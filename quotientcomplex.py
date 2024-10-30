import pandas as pd
from functools import partial
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
import os.path as osp
import os
import numpy as np
import random
import math,time
import pandas as pd
from jarvis.core.specie import chem_data, get_node_attributes
from jarvis.core.atoms import Atoms
from jarvis.core.atoms import pmg_to_atoms
from pathlib import Path
from typing import Optional
from typing import List, Tuple, Sequence, Optional
# from matbench.bench import MatbenchBenchmark
import pickle


from pandarallel import pandarallel
pandarallel.initialize()


def prepare_pyg_batch(
    batch: Tuple[Data, torch.Tensor], device=None, non_blocking=False
):
    """Send batched dgl crystal graph to device."""
    g, t = batch
    batch = (
        g.to(device),
        t.to(device, non_blocking=non_blocking),
    )

    return batch

class PygStandardize(torch.nn.Module):
    """Standardize atom_features: subtract mean and divide by std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Register featurewise mean and standard deviation."""
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, g: Data):
        """Apply standardization to atom_features."""
        h = g.x
        g.x = (h - self.mean) / self.std
        return g


class PygStructureDataset(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
        self,
        df: pd.DataFrame,
        graphs: Sequence[Data],
        target: str,
        atom_features="atomic_number",
        transform=None,
        classification=False,
        id_tag="id",
        neighbor_strategy="",
        lineControl=True,
        mean_train=None,
        std_train=None,
        dataname=None,
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        `graphs`: DGLGraph representations corresponding to rows in `df`
        `target`: key for label column in `df`
        """
        self.df = df
        self.graphs = graphs
        self.target = target
        if dataname=='jarvis':
            self.ids = self.df['jid']
        elif dataname in ['mp','matbench']:
            self.ids = self.df['id']
        self.atoms = self.df['atoms']
        self.labels = torch.tensor(self.df[target]).type(torch.get_default_dtype())
        #print("mean %f std %f"%(self.labels.mean(), self.labels.std()))
        
        
        if mean_train == None:
            mean = self.labels.mean()
            std = self.labels.std()
            self.labels = (self.labels - mean) / std
            #print("normalize using training mean but shall not be used here %f and std %f" % (mean, std))
        else:
            self.labels = (self.labels - mean_train) / std_train
            #print("normalize using training mean %f and std %f" % (mean_train, std_train))
        
        #print('start transform')
        self.transform = transform

        
        
        
        '''
        #use new feature to replace this part
        
        features = self._get_attribute_lookup(atom_features)

        # load selected node representation
        # assume graphs contain atomic number in g.ndata["atom_features"]
        for g in graphs:
            z = g.x
            g.atomic_number = z
            z = z.type(torch.IntTensor).squeeze()
            f = torch.tensor(features[z]).type(torch.FloatTensor)
            if g.x.size(0) == 1:
                f = f.unsqueeze(0)
            g.x = f
        '''
        
        
        #print('start batch')
        self.prepare_batch = prepare_pyg_batch
        #print('batch ok')

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        """Build a lookup array indexed by atomic number."""
        max_z = max(v["Z"] for v in chem_data.values())

        # get feature shape (referencing Carbon)
        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features

    def __len__(self):
        """Get length."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        g = self.graphs[idx]
        label = self.labels[idx]

        if self.transform:
            g = self.transform(g)

        return g, label

    def setup_standardizer(self, ids):
        """Atom-wise feature standardization transform."""
        x = torch.cat(
            [
                g.x
                for idx, g in enumerate(self.graphs)
                if idx in ids
            ]
        )
        self.atom_feature_mean = x.mean(0)
        self.atom_feature_std = x.std(0)

        self.transform = PygStandardize(
            self.atom_feature_mean, self.atom_feature_std
        )

    @staticmethod
    def collate(samples: List[Tuple[Data, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        return batched_graph, torch.tensor(labels)
        


def area_from_edges(dis):
    dis = [ round(dis[0],4),round(dis[1],4),round(dis[2],4) ]
    if abs(dis[0]+dis[1]-dis[2])<=0.01 or abs(dis[0]+dis[2]-dis[1])<=0.01 or abs(dis[1]+dis[2]-dis[0])<=0.01:
        return 0
    p = ( dis[0] +dis[1] +dis[2] )/2
    return pow( p * (p-dis[0]) * (p-dis[1]) * (p-dis[2]) ,0.5)

 
def angle_from_edges(dis,i):
    temp = 0
    index = []
    for j in [0,1,2]:
        if j!=i:
            index.append(j)
            temp = temp + pow(dis[j],2)
    temp = temp - pow(dis[i],2)
    temp = round( temp/(2*dis[ index[0] ]*dis[ index[1] ]),2)
    return math.acos(temp)
    
    
    
# calculate actural position of node i in the new_point set
def calculate_coords(atoms,new_point,i):
    lattice = atoms.lattice_mat
    coords = atoms.cart_coords

    idx = new_point[i][0]
    image = new_point[i][1:]
    pos = coords[idx] + np.dot(image,lattice)
    return pos



from itertools import combinations_with_replacement
def k_polynomial(dis, k):
    terms = dis.copy()  # Add original variables
    
    # Loop through dimensions from 2 to k
    for dim in range(2, k + 1):
        # Generate combinations with replacement for the current dimension
        combs = combinations_with_replacement(dis, dim)
        for comb in combs:
            term = 1
            for var in comb:
                term *= var
            terms.append(term)
    
    return terms
   


def get_complex(config,atoms,cutoff=None):
    if cutoff is None:
        cutoff = config['features']['cutoff']
    max_neighbors = config['features']['max_neighbors']
    use_triangle_potential = config['features']['use_triangle_potential']

    lat = atoms.lattice
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    # attempt = 0
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        # attempt += 1

        return get_complex(config=config,atoms=atoms,cutoff=r_cut)

    
    
    atom_name = []
    for ii, s in enumerate(atoms.elements):
        atom_name.append(s)
    
    
    
    # point
    new_point = []
    point_dict = {}
    for i in range(atoms.num_atoms):
        new_point.append([i,0,0,0])
        point_dict[str([ i,0,0,0 ])] = i
    pn = atoms.num_atoms
    for item in all_neighbors:
        neighbor = sorted(item,key=lambda x: x[2])
        for i in range(max_neighbors):
            one = neighbor[i][0]
            two = neighbor[i][1]
            val = neighbor[i][2]
            coor = neighbor[i][3]
            temp = [two,round(coor[0]),round(coor[1]),round(coor[2])]
            if str(temp) not in point_dict:
                new_point.append(temp)
                atom_name.append( atom_name[two] )
                point_dict[str(temp)] = pn
                pn = pn + 1
                
    
    
    
    
    # edge
    edge_index = [ [],[] ]
    edge_dis = []
    E = []
    E_dict = {}
    en = 0
    for item in all_neighbors:
        neighborlist = sorted(item, key=lambda x: x[2])       
        for i in range(max_neighbors):
            one = neighborlist[i][0]
            two = neighborlist[i][1]
            val = neighborlist[i][2]
            coor = neighborlist[i][3]
            pos = [round(coor[0]),round(coor[1]),round(coor[2])]
            
            edge_index[0].append(two)
            edge_index[1].append(one)
            temp = point_dict[str([ two,round(coor[0]),round(coor[1]),round(coor[2]) ])]
            E.append([temp,one])
            a1 = get_node_attributes(atom_name[one],atom_features='atomic_number')
            a2 = get_node_attributes(atom_name[two],atom_features='atomic_number')

            edge_dis.append(a1+a2+[-0.75/val])

            E_dict[str([temp,one])] = [en,val]
            en = en + 1
            
            
                
        
    
    # triangle
    T = []
    T_dis = []
    triangle_index = [ [],[] ]
    triangle_dis = []
    for i in range(atoms.num_atoms,len(new_point)):
        feat1 = get_node_attributes(atom_name[i],atom_features='atomic_number')
        for j in range(atoms.num_atoms):
            if str([i,j]) in E_dict:
                dis1 = E_dict[str([i,j])][1]
                feat2 = get_node_attributes(atom_name[j],atom_features='atomic_number')
                for k in range(atoms.num_atoms):
                    feat3 = get_node_attributes(atom_name[k],atom_features='atomic_number')
                    if j!=k and str([i,k]) in E_dict:
                        dis2 = E_dict[str([i,k])][1]
                        if str([j,k]) in E_dict:
                            
                            dis3 = E_dict[str([j,k])][1]
                            T.append([i,j,k])

                            if use_triangle_potential:
                                dis = [dis1,dis2,dis3]
                                feat5 = k_polynomial(dis, 2)
                                T_dis.append(feat5)
                            else:
                                feat4 = area_from_edges([dis1,dis2,dis3])
                                T_dis.append(feat1+feat2+feat3+[feat4])                                
                            
                            
                            
                        if str([k,j]) in E_dict:
                            dis3 = E_dict[str([k,j])][1]
                            T.append([i,k,j])

                            if use_triangle_potential:
                                dis = [dis2,dis1,dis3]
                                feat5 = k_polynomial(dis, 2)
                                T_dis.append(feat5)
                            else:   
                                feat4 = area_from_edges([dis2,dis1,dis3])
                                T_dis.append(feat1+feat3+feat2+[feat4])
                            
    for i in range(len(T)):
        one = T[i][0]
        two = T[i][1]
        thr = T[i][2]
        
        index1 = E_dict[str([one,two])][0]
        index2 = E_dict[str([one,thr])][0]
        index3 = E_dict[str([two,thr])][0]
        
        triangle_index[0].append(index1)
        triangle_index[1].append(index2)
        triangle_dis.append(T_dis[i])        
        
        triangle_index[0].append(index1)
        triangle_index[1].append(index3)
        triangle_dis.append(T_dis[i])
        
        triangle_index[0].append(index2)
        triangle_index[1].append(index3)
        triangle_dis.append(T_dis[i])

    edge_index = torch.tensor(edge_index,dtype=torch.long)
    edge_dis = torch.tensor(edge_dis,dtype=torch.get_default_dtype())
    triangle_index = torch.tensor(triangle_index,dtype=torch.long)
    triangle_dis = torch.tensor(triangle_dis,dtype=torch.get_default_dtype())
    return edge_index,edge_dis,triangle_index,triangle_dis



def atom_quotient_complex(
        config: Optional[dict] = None,
        atoms=None,
        neighbor_strategy="k-nearest",
        # cutoff=8.0, 
        # max_neighbors=12,
        atom_features="cgcnn",        
    ):
        
        
        edge_index,edge_dis,triangle_index,triangle_dis = get_complex(config,atoms)
        
        
        # cgcnn feature
        sps_features = []
        for ii, s in enumerate(atoms.elements):
            feat = list(get_node_attributes(s, atom_features='atomic_number'))
            sps_features.append(feat)
        
        sps_features = np.array(sps_features)
        
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )
        
        
        g = Data(x=node_features, edge_index=edge_index, edge_dis=edge_dis, triangle_index=triangle_index, triangle_dis=triangle_dis,label=0 )
        
        return g
        
        


def load_complexes(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    name: str = "dft_3d",
    neighbor_strategy: str = "k-nearest",
    # cutoff: float = 8,
    # max_neighbors: int = 12,
):
    """Construct quotient simplicial complex.

    """

    def atoms_to_complex(atoms, config):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms)
        return atom_quotient_complex(
            config = config,
            atoms=structure,
            neighbor_strategy=neighbor_strategy,
            # cutoff=cutoff,
            #atom_features="atomic_number",
            atom_features='cgcnn',
            # max_neighbors=max_neighbors,

            
        )
        
    complexes = df['atoms'].parallel_apply(lambda atoms: atoms_to_complex(atoms, config)).values
    return complexes



def get_train_val_test_loader(config,dataset,dataset_train,dataset_val,dataset_test,prop,batch_size):
    # train
    df = pd.DataFrame(dataset_train)
    vals = df[prop].values
    complexes = load_complexes(df,name=dataset,config=config)
    
    mean_train = np.mean(vals)
    std_train = np.std(vals)
    
    
    train_data = PygStructureDataset(
            df,
            complexes,
            target=prop,
            atom_features='cgcnn',
            mean_train=mean_train,
            std_train=std_train,
            dataname=dataset,
        )
    
    # val
    df = pd.DataFrame(dataset_val)
    vals = df[prop].values
    complexes = load_complexes(df,name=dataset,config=config)
    val_data = PygStructureDataset(
            df,
            complexes,
            target=prop,
            atom_features='cgcnn',
            mean_train=mean_train,
            std_train=std_train,
            dataname=dataset,
        )
        
    # test
    df = pd.DataFrame(dataset_test)
    vals = df[prop].values
    complexes = load_complexes(df,name=dataset,config=config)
    test_data = PygStructureDataset(
            df,
            complexes,
            target=prop,
            atom_features='cgcnn',
            mean_train=mean_train,
            std_train=std_train,
            dataname=dataset,
        )
    
    
    collate_fn = train_data.collate
    
    # use a regular pytorch dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        follow_batch = ['x','edge_dis']
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        follow_batch = ['x','edge_dis']
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        follow_batch = ['x','edge_dis']
    )
    
    prepare_batch = partial(train_loader.dataset.prepare_batch)
    return train_loader,val_loader,test_loader,prepare_batch,mean_train,std_train



# def load_matbench_complexes_and_df(   
#     config: Optional[dict] = None,
#     task_name: str = "matbench_mp_e_form",
#     fold: int = 0,
#     neighbor_strategy: str = "k-nearest",
#     # cutoff: float = 8,
#     # max_neighbors: int = 12,
# ):
#     """Construct quotient simplicial complex
#        with matbench dataset, 5 folds cross validation.
#     """

#     def atoms_to_complex(atoms, config):
#         """Convert structure dict to DGLGraph."""
#         structure = pmg_to_atoms(atoms)
#         return atom_quotient_complex(
#             config=config,
#             atoms=structure,
#             neighbor_strategy=neighbor_strategy,
#             # cutoff=cutoff,
#             #atom_features="atomic_number",
#             atom_features='cgcnn',
#             # max_neighbors=max_neighbors,
#         )
        
#     mb = MatbenchBenchmark(autoload=False)
#     task_names = ['matbench_dielectric',
#     'matbench_expt_gap',
#     'matbench_expt_is_metal',
#     'matbench_glass',
#     'matbench_jdft2d',
#     'matbench_log_gvrh',
#     'matbench_log_kvrh',
#     'matbench_mp_e_form',
#     'matbench_mp_gap',
#     'matbench_mp_is_metal',
#     'matbench_perovskites',
#     'matbench_phonons',
#     'matbench_steels']
#     for ii, task in enumerate(mb.tasks):
#         if ii == task_names.index(task_name):
#             task.load()


#             train_and_val_inputs, train_and_val_outputs = task.get_train_and_val_data(fold)
#             train_inputs = train_and_val_inputs.iloc[: int(0.75 * len(train_and_val_inputs))]
#             train_outputs = train_and_val_outputs.iloc[: int(0.75 * len(train_and_val_inputs))]
#             val_inputs = train_and_val_inputs.iloc[int(0.75 * len(train_and_val_inputs)) :]
#             val_outputs = train_and_val_outputs.iloc[int(0.75 * len(train_and_val_inputs)) :]
#             test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

#             train_complexes = train_inputs.parallel_apply(lambda atoms: atoms_to_complex(atoms, config)).values
#             val_complexes = val_inputs.parallel_apply(lambda atoms: atoms_to_complex(atoms, config)).values
#             test_complexes = test_inputs.parallel_apply(lambda atoms: atoms_to_complex(atoms, config)).values

#             # ids: 'mb-dielectric-0001', 'mb-dielectric-0002', ...
#             train_dict = {'id':train_inputs.index,'atoms':train_inputs, 'target':train_outputs}
#             val_dict = {'id':val_inputs.index,'atoms':val_inputs, 'target':val_outputs}
#             test_dict = {'id':test_inputs.index,'atoms':test_inputs, 'target':test_outputs}
#             train_df = pd.DataFrame(train_dict)
#             val_df = pd.DataFrame(val_dict)
#             test_df = pd.DataFrame(test_dict)

#     return train_complexes, val_complexes, test_complexes, train_df, val_df, test_df


# def get_train_val_test_loader_matbench(config,prop,fold,batch_size):
#     train_complexes, val_complexes, test_complexes, train_df, val_df, test_df = load_matbench_complexes_and_df(
#         config=config,
#         task_name=prop,
#         fold=fold,
#         neighbor_strategy="k-nearest",
#         # cutoff=8,
#         # max_neighbors=12,
#     )
    
#     train_outputs = train_df['target'].values
#     mean_train = np.mean(train_outputs)
#     std_train = np.std(train_outputs)
    

#     train_data = PygStructureDataset(
#             train_df,
#             train_complexes,
#             target='target',
#             atom_features='cgcnn',
#             mean_train=mean_train,
#             std_train=std_train,
#             dataname='matbench',
#         )
    
#     val_data = PygStructureDataset(
#             val_df,
#             val_complexes,
#             target='target',
#             atom_features='cgcnn',
#             mean_train=mean_train,
#             std_train=std_train,
#             dataname='matbench',
#         )
    
#     test_data = PygStructureDataset(
#             test_df,
#             test_complexes,
#             target='target',
#             atom_features='cgcnn',
#             mean_train=mean_train,
#             std_train=std_train,
#             dataname='matbench',
#         )
    
#     collate_fn = train_data.collate
    
#     # use a regular pytorch dataloader
#     train_loader = DataLoader(
#         train_data,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=collate_fn,
#         drop_last=True,
#         follow_batch = ['x','edge_dis'],
#         pin_memory=True
#     )
    
#     val_loader = DataLoader(
#         val_data,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=collate_fn,
#         drop_last=True,
#         follow_batch = ['x','edge_dis'],
#         pin_memory=True
#     )

#     test_loader = DataLoader(
#         test_data,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=collate_fn,
#         drop_last=False,
#         follow_batch = ['x','edge_dis'],
#         pin_memory=True
#     )
    
#     prepare_batch = partial(train_loader.dataset.prepare_batch)

#     return train_loader,val_loader,test_loader,prepare_batch,mean_train,std_train


if __name__ == '__main__':
    mb = MatbenchBenchmark(autoload=False)
    task_names = ['matbench_dielectric',
    'matbench_expt_gap',
    'matbench_expt_is_metal',
    'matbench_glass',
    'matbench_jdft2d',
    'matbench_log_gvrh',
    'matbench_log_kvrh',
    'matbench_mp_e_form',
    'matbench_mp_gap',
    'matbench_mp_is_metal',
    'matbench_perovskites',
    'matbench_phonons',
    'matbench_steels']
    task_name = 'matbench_jdft2d'
    for ii, task in enumerate(mb.tasks):
        if ii == task_names.index(task_name):
            task.load()
            train_and_val_inputs, train_and_val_outputs = task.get_train_and_val_data(0)
            test_inputs, test_outputs = task.get_test_data(0, include_target=True)
    data_to_pickle = {
        'train_and_val_inputs': train_and_val_inputs,
        'train_and_val_outputs': train_and_val_outputs,
        'test_inputs': test_inputs,
        'test_outputs': test_outputs
    }
    with open('matbench_data/'+ task_name+'_raw.pickle', 'wb') as handle:
        pickle.dump(data_to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # config = {
    #     'features': {
    #         'cutoff': 8.0,
    #         'max_neighbors': 12,
    #         'use_edge_potential': False,
    #         'use_triangle_potential': True,
    #     }
    # }

    # data = train_and_val_inputs.iloc[0]
    # atoms = pmg_to_atoms(data)
    # edge_index,edge_dis,triangle_index,triangle_dis = get_complex(config,atoms)
    # print(edge_dis)
