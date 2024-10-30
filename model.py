from functools import partial
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import os.path as osp
import os
import numpy as np
import random
import math,time
import pandas as pd
from jarvis.core.specie import chem_data, get_node_attributes
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jdata

from pathlib import Path
from typing import List, Tuple, Sequence, Optional

from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops
import os.path as osp
import math
import torch.nn.functional as F
from utils import RBFExpansion_node,RBFExpansion_edge,RBFExpansion_triangle, RBFExpansion_triangle_dis
from torch_geometric.nn.norm import BatchNorm




device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda:2")

        
class QCBlock(MessagePassing):
    def __init__(self,in_size,out_size):
        super(QCBlock,self).__init__(aggr='add')
        self.in_size = in_size
        self.out_size = out_size

        self.K_v2v = torch.nn.Parameter(torch.zeros(in_size,out_size,device=device))
        torch.nn.init.xavier_uniform_(self.K_v2v)
        self.K_e2v = torch.nn.Parameter(torch.zeros(in_size,out_size,device=device))
        torch.nn.init.xavier_uniform_(self.K_e2v)
        self.V_v2v = torch.nn.Parameter(torch.zeros(in_size,out_size,device=device))
        torch.nn.init.xavier_uniform_(self.V_v2v)
        self.V_e2v = torch.nn.Parameter(torch.zeros(in_size,out_size,device=device))
        torch.nn.init.xavier_uniform_(self.V_e2v)
        
        self.key_update = torch.nn.Sequential(torch.nn.Linear(out_size*2,out_size*2,device=device),torch.nn.SiLU(),torch.nn.Linear(out_size*2,out_size*2,device=device))
        self.linear_update = torch.nn.Sequential(torch.nn.Linear(out_size*2,out_size*2,device=device),torch.nn.SiLU(),torch.nn.Linear(out_size*2,out_size*2,device=device))
        # self.layernorm = torch.nn.LayerNorm(out_size*2,device=device)
        self.sigmoid = torch.nn.Sigmoid()
        self.msg_layer = torch.nn.Sequential(torch.nn.Linear(out_size*2,out_size,device=device),torch.nn.LayerNorm(out_size,device=device) )
        self.bn = torch.nn.BatchNorm1d(out_size*2)      
        self.act = torch.nn.SiLU()
        
        
    def forward(self,x,edge_index,edge_feature):
        K_v = torch.mm(x,self.K_v2v)
        V_v = torch.mm(x,self.V_v2v)
        
        
        if min(edge_feature.shape)==0:
            return V_v
        else:
            out = self.propagate(edge_index,query_v=K_v,key_v=K_v,value_v=V_v,edge_feature=edge_feature)
            return out
        
    
    def message(self,query_v_i,key_v_i,key_v_j,edge_feature,value_v_i,value_v_j):
        K_E = torch.mm(edge_feature,self.K_e2v)
        V_E = torch.mm(edge_feature,self.V_e2v)
        
        query_i = torch.cat([ query_v_i,query_v_i  ],dim=1)
        key_j = torch.cat([ key_v_j,K_E ],dim=1)
        key_j = self.key_update(key_j)
        alpha = ( query_i * key_j ) / math.sqrt(self.out_size * 2)
        alpha = F.dropout(alpha,p=0,training=self.training)

        out = torch.cat([ value_v_j,V_E  ],dim=1)
        out = self.linear_update(out) * self.sigmoid( self.bn(alpha.view(-1,2*self.out_size)) )
        out = self.act(self.msg_layer(out))
        return out
        
        



class QCConv(torch.nn.Module):
    def __init__(self,in_size,out_size,head,dropout=0):
        super(QCConv,self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.head = head
        self.attention = torch.nn.ModuleList()
        for i in range(self.head):
            self.attention.append(QCBlock(in_size,out_size))
        self.linear_concate_v = torch.nn.Linear(out_size*head,out_size,device=device)
        self.bn_v = torch.nn.BatchNorm1d(out_size,device=device)
        self.act = torch.nn.SiLU()
        self.dropout = dropout
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.linear_concate_v.reset_parameters()        
    
    def forward(self,x,edge_index,edge_feature):
        hidden_v = []
        for atten in self.attention:
            hv = atten(x,edge_index,edge_feature)
            hidden_v.append(hv)
            
        hv = torch.cat(hidden_v,dim=1)
        out = self.linear_concate_v(hv)

        out = self.act(self.bn_v(out))
        out = x + F.dropout(out, p=self.dropout, training=self.training)
        return out
    




class QCformer(torch.nn.Module): 
    def __init__(self,config):
        super(QCformer,self).__init__()
        self.head1 = config['model']['head_v']
        self.head2 = config['model']['head_e']
        self.layer_number = config['model']['layer_number']
        self.inner_layer = config['model']['inner_layer']
        self.in_size = config['model']['hidden_dim']
        self.out_size = config['model']['hidden_dim']
        self.atom_feature_size = config['features']['atom_feature_size']
        self.edge_feature_size = config['features']['edge_feature_size']
        self.triangle_feature_size = config['features']['triangle_feature_size']
        use_triangle_potential = config['features']['use_triangle_potential']
        self.inner_dropout = config['model']['inner_dropout']
        self.dropout = config['model']['dropout']

        self.atom_init = torch.nn.Sequential(
            RBFExpansion_node() ,
            torch.nn.Linear(self.atom_feature_size, self.in_size,device=device),
            torch.nn.SiLU(),
            torch.nn.Linear(self.in_size, self.in_size,device=device),
        )

 
        self.edge_init = torch.nn.Sequential(
            RBFExpansion_edge(vmin=-4,vmax=0,bins=64),
            torch.nn.Linear(self.edge_feature_size, self.in_size,device=device),
            torch.nn.SiLU(),
            torch.nn.Linear(self.in_size, self.in_size,device=device),
        )  

        if use_triangle_potential:
            self.triangle_init = torch.nn.Sequential(
                RBFExpansion_triangle_dis(vmin=0,vmax=5.0,bins=8),
                torch.nn.Linear(self.triangle_feature_size, self.in_size,device=device),
                torch.nn.SiLU(),
                torch.nn.Linear(self.in_size, self.in_size,device=device),
            )  
        else:
            self.triangle_init = torch.nn.Sequential(
                RBFExpansion_triangle(vmin=0,vmax=8.0,bins=80),
                torch.nn.Linear(self.triangle_feature_size, self.in_size,device=device),
                torch.nn.SiLU(),
                torch.nn.Linear(self.in_size, self.in_size,device=device),
            )
        

        self.layer0 = torch.nn.ModuleList( [ QCConv( self.in_size,  self.out_size,  self.head1, self.inner_dropout) for i in range(self.inner_layer) ] )          
        self.layer1 = torch.nn.ModuleList( [ QCConv( self.in_size,  self.out_size,  self.head1, self.inner_dropout) for i in range(self.layer_number) ] )
        self.layer2 = torch.nn.ModuleList( [ QCConv( self.in_size,  self.out_size,  self.head2, self.inner_dropout) for i in range(self.layer_number) ] )

        # self.downpool = torch.nn.Linear( self.out_size, 48, device=device)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear( self.out_size*2,  self.out_size,device=device), torch.nn.SiLU()
        )
        self.fc_out = torch.nn.Linear( self.out_size, 1,device=device)
    
        
    def forward(self,data):
        x = self.atom_init(data.x)
        edge_feature = self.edge_init(data.edge_dis)
        triangle_feature = self.triangle_init(data.triangle_dis)

        for i in range(self.inner_layer):
            x = self.layer0[i](x,data.edge_index,edge_feature)
        
        for i in range(self.layer_number):
            edge_feature = self.layer2[i](edge_feature,data.triangle_index,triangle_feature)
            x = self.layer1[i](x,data.edge_index,edge_feature)
            
        
        
        feature1 = global_mean_pool(x,data.x_batch)
        feature2 = global_mean_pool(edge_feature,data.edge_dis_batch)
        # feature2 = self.downpool(feature2)
        feature = torch.cat([feature1,feature2],dim=1)
        # feature = feature1

        feature = F.dropout(feature, p=self.dropout, training=self.training)
        
        
        feature = self.fc(feature)
        
        out = self.fc_out(feature)
        return torch.squeeze(out)