import os
import torch
import random
from torch_geometric.nn import MessagePassing
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops
import os.path as osp
import math
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
import ignite
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.utils import convert_tensor
from ignite.metrics import Loss, MeanAbsoluteError
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan

from data import get_train_val_test
from quotientcomplex import get_train_val_test_loader
from model import QCformer

import yaml
import argparse
import json
import time

import pickle




# torch config
torch.set_default_dtype(torch.float32)
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda:2")



def count_parameters(model):
    total_params = 0
    for parameter in model.parameters():
        total_params += parameter.element_size() * parameter.nelement()
    for parameter in model.buffers():
        total_params += parameter.element_size() * parameter.nelement()
    total_params = total_params / 1024 / 1024
    print(f"Total Trainable Params: {total_params} MB")
    return total_params



def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print('  ' * indent + str(key) + ':')
            print_dict(value, indent + 1)
        else:
            print('  ' * indent + str(key) + ': ' + str(value))


def load_dataset(config,dataset,prop,fold,batch_size,file_path):
    print(file_path)
    if os.path.exists(file_path):
        print('Data file exists, loading data from pickle file...')
        with open(file_path, 'rb') as handle:
            data = pickle.load(handle)
        train_loader = data['train_loader']
        val_loader = data['val_loader']
        test_loader = data['test_loader']
        prepare_batch = data['prepare_batch']
        mean_train = data['mean_train']
        std_train = data['std_train']
        n_test = len(test_loader.dataset)
        print('Data loaded!')
    else:
        print('Data file does not exist, loading data from scratch...')
        # if dataset == 'matbench':
        #     train_loader,val_loader,test_loader,prepare_batch,mean_train,std_train = get_train_val_test_loader_matbench(config,prop,fold,batch_size)
        #     n_test = len(test_loader.dataset)
        # else:
        dataset_train,dataset_val,dataset_test = get_train_val_test(dataset,prop,split_seed=123)
        train_loader,val_loader,test_loader,prepare_batch,mean_train,std_train = get_train_val_test_loader(config,dataset,dataset_train,dataset_val,dataset_test,prop,batch_size)
        n_test = len(dataset_test)
        try: 
            data_to_pickle = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'prepare_batch': prepare_batch,
            'mean_train': mean_train,
            'std_train': std_train
        }
            with open(file_path, 'wb') as handle:
                pickle.dump(data_to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            print('Failed to save data to pickle file')

        print('Dataset loaded!')
    return train_loader,val_loader,test_loader,prepare_batch,mean_train,std_train,n_test


# def train(dataset,epoch,learning_rate,batch_size,weight_decay,property,split_seed=123,fold=0):
def train(config_path):

    config = load_config(config_path)
    print_dict(config)

    dataset = config['training']['dataset']
    epoch = config['training']['epoch']
    learning_rate = config['training']['learning_rate']
    batch_size = config['training']['batch_size']
    weight_decay = float(config['training']['weight_decay'])
    prop = config['training']['property']
    split_seed = config['training']['split_seed']
    fold = config['training']['fold']
    write_predictions = config['training']['write_predictions']
    model_criterion = config['model']['criterion']


    if write_predictions:
        output_dir = 'results/' + dataset + '-' + prop
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


    deterministic = False
     

    file_path = dataset + '-preprocessed/' + prop + '-' + str(batch_size) + '.pickle'
    train_loader,val_loader,test_loader,prepare_batch,mean_train,std_train,n_test = load_dataset(config,dataset,prop,fold,batch_size,file_path)
    print('mean_train:',mean_train)
    print('std_train:',std_train)
    print('n_test:',n_test)
    print('------------------------------------')
    print('Start training...')

    ignite.utils.manual_seed(split_seed)
    
    net = QCformer(config)
    count_parameters(net)
    net.to(device)
    if model_criterion == 'mse':
        criterion = torch.nn.MSELoss()
    elif model_criterion == 'mae':
        criterion = torch.nn.L1Loss()
    params = group_decay(net)
    optimizer = torch.optim.AdamW(params,lr=learning_rate,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=epoch,pct_start=0.3)
    

    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError() * std_train, "neg_mae": -1.0 * MeanAbsoluteError() * std_train}
    
    trainer = create_supervised_trainer(
        net,
        optimizer,
        criterion,
        prepare_batch=prepare_batch,
        device=device,
        deterministic=deterministic,
    )
    val_evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )
    train_evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )
    test_evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )
    
    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )
    
    '''
    # model checkpointing
    to_save = {"model": net,"optimizer": optimizer,"lr_scheduler": scheduler,"trainer": trainer}
    handler = Checkpoint(to_save, DiskSaver('saved/' + model_detail, create_dir=True, require_empty=False),
            n_saved=2, global_step_transform=lambda *_: trainer.state.epoch)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)
    
    # evaluate save
    to_save = {"model": net}
    handler = Checkpoint(to_save, DiskSaver('saved/' + model_detail, create_dir=True, require_empty=False),
            n_saved=2,filename_prefix='best',score_name="neg_mae",global_step_transform=lambda *_: trainer.state.epoch)
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)
    '''
    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
        "test": {m: [] for m in metrics.keys()},
    }
    
    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        """Print training and validation metrics to console."""
        val_evaluator.run(val_loader)
        vmetrics = val_evaluator.state.metrics
        for metric in metrics.keys():
            vm = vmetrics[metric]
            t_metric = metric
            if metric == "roccurve":
                vm = [k.tolist() for k in vm]
            if isinstance(vm, torch.Tensor):
                vm = vm.cpu().numpy().tolist()

            history["validation"][metric].append(vm)

        
        
        epoch_num = len(history["validation"][t_metric])
        if epoch_num % 10 == 0:
            # train
            train_evaluator.run(train_loader)
            tmetrics = train_evaluator.state.metrics
            for metric in metrics.keys():
                tm = tmetrics[metric]
                if metric == "roccurve":
                    tm = [k.tolist() for k in tm]
                if isinstance(tm, torch.Tensor):
                    tm = tm.cpu().numpy().tolist()

                history["train"][metric].append(tm)
            
        else:
            tmetrics = {}
            tmetrics['mae'] = -1
            #test_metrics = {}
            #test_metrics['mae'] = -1
        
        test_mae = 0
        if epoch_num==epoch:
            # test
            net.eval()
            if write_predictions:
                output_file = f"results/{dataset}-{prop}/test_prediction.csv"
                with open(output_file, 'w') as f:
                    f.write("target,prediction\n")
                    with torch.no_grad():  
                        for dat in test_loader:
                            g,target = dat
                            target=target.to(device)
                            out_data = net(g.to(device))
                            pre1 = target.tolist()
                            pre2 = out_data.tolist()
                            
                            for true_val, pred_val in zip(pre1, pre2):
                                f.write(f"{true_val:.6f}, {pred_val:.6f}\n")

            with torch.no_grad():  
                for dat in test_loader:
                    g,target = dat
                    target=target.to(device)
                    out_data = net(g.to(device))
                    #print(target.shape,out_data.shape)
                    test_mae = test_mae + torch.abs(target-out_data).sum().data.item()
                


        if epoch_num<epoch:
            print('epoch:',epoch_num,f"Val_MAE: {vmetrics['mae']:.4f}",f"Train_MAE: {tmetrics['mae']:.4f}")
        else:
            print('epoch:',epoch_num,f"Val_MAE: {vmetrics['mae']:.4f}",f"Train_MAE: {tmetrics['mae']:.4f}","Test MAE",test_mae*std_train/n_test)

        

    trainer.run(train_loader, max_epochs=epoch)
    
    
def main():
    parser = argparse.ArgumentParser(description="Run model with config file")
    parser.add_argument('config_file', type=str, help='Path to the config file')
    args = parser.parse_args()
    
    import time
    start_time = time.time()

    train(args.config_file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    seconds = elapsed_time % 60
    print(f"Elapsed time: {int(hours)} hrs, {int(minutes)} mins, {seconds:.2f} secs")



if __name__ == '__main__':
    main() 

    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    #     print('The code uses GPU...')
    # else:
    #     device = torch.device('cpu')
    #     print('The code uses CPU!!!')

    





        
    