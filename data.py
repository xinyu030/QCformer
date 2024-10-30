from jarvis.db.figshare import data as jdata
import math
import numpy as np
import random
import pandas as pd

def get_id_train_val_test(
    total_size=1000,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]  
    id_test = ids[-n_test:]
    return id_train, id_val, id_test




def prepare_data_moduli(typ):
    if typ[0]=='b':
        typ1 = 'bulk'
    else:
        typ1 = 'shear'
    import pickle
    filename = 'data/' + typ1 + '_megnet_train.pkl'
    d_train = pickle.load(open(filename,'rb'))
    
    filename = 'data/' + typ1 + '_megnet_val.pkl'
    d_val = pickle.load(open(filename,'rb'))
    
    filename = 'data/' + typ1 + '_megnet_test.pkl'
    d_test = pickle.load(open(filename,'rb'))
    
    property = typ1 + ' modulus'
    for item in d_train:
        item[property] = item[property].item()
    for item in d_val:
        item[property] = item[property].item()
    for item in d_test:
        item[property] = item[property].item()
    return d_train,d_val,d_test


def get_train_val_test(dataset,property,split_seed):
    if property=='bulk modulus' or property=='shear modulus':
        dataset_train,dataset_val,dataset_test = prepare_data_moduli(property)
    
    else:
        if dataset=='mp':
            d = jdata('megnet')
        elif dataset=='jarvis':
            d = jdata('cfid_3d')
    
        # extract specific data
        all_targets = []
        dat = []
        index = []
    
        for i in range(len(d)):
        # for i in range(500):
            if ( d[i][property] is not None and d[i][property] != "na" and not math.isnan(d[i][property]) ):
                index.append(i)
                all_targets.append(d[i][property])
                dat.append(d[i])
    
        # split train,val,test
        if dataset=='jarvis':
            id_train, id_val, id_test = get_id_train_val_test( total_size=len(index), split_seed=split_seed,
                train_ratio=0.8,val_ratio=0.1,test_ratio=0.1)
        elif dataset=='mp':
            id_train, id_val, id_test = get_id_train_val_test( total_size=len(index), split_seed=split_seed,
                train_ratio=0.866564,val_ratio=0.072214,test_ratio=0.061223)
    
    
        dataset_train = [dat[x] for x in id_train]
        dataset_val = [dat[x] for x in id_val]
        dataset_test = [dat[x] for x in id_test]
    
    return dataset_train,dataset_val,dataset_test


if __name__ == "__main__":
    dataset = 'mp'