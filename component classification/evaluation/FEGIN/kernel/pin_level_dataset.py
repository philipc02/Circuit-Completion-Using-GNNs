import os.path as osp
import sys, os
from shutil import rmtree
import torch
#from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree,from_networkx
import torch_geometric.transforms as T
sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
sys.path.append('%s/' % os.path.dirname(os.path.realpath(__file__)))
from utils import create_subgraphs, return_prob
from tu_dataset import TUDataset
from torch_geometric.data import Data, InMemoryDataset
import pdb
import pickle
import numpy as np
import networkx as nx

class PinLevelDataset(InMemoryDataset):
    def __init__(self, root, name,h, max_nodes_per_hop, node_label, use_rd,transform=None, pre_transform=None, pre_filter=None):
        self.h,self.max_nodes_per_hop, self.node_label, self.use_rd,self.name = h,max_nodes_per_hop, node_label, use_rd,name
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        
        
#     @property
#     def raw_file_names(self):
#         return ['ltspice_examples_torch.pt']


    @property
    def processed_file_names(self):
        return [self.name+'_pin_level_processed.pt']

#     def download(self):
#         # Download to `self.raw_dir`.
#         download_url(url, self.raw_dir)
#         ...

    def process(self):
        # Read data into huge `Data` list.
#         data_list = [...]
        print("loading pin-level data now!")

        # Load the dataset created by preparation script
        with open(f'data/{self.name}_pin_level_GC.pkl', 'rb') as f:
            dataset_dict = pickle.load(f)
        
        with open(f'data/{self.name}_pin_level_label_mapping.pkl', 'rb') as f:
            label_mapping = pickle.load(f)
        
        data_list = []

        # Process training graphs
        for i, (G, label) in enumerate(zip(dataset_dict['train_x'], dataset_dict['train_y'])):
            data = self.convert_pin_graph_to_pyg(G)
            if data is not None:
                data.y = torch.tensor([label], dtype=torch.long)
                data.set = 'train'
                data_list.append(data)
        
        # Process test graphs
        for i, (G, label) in enumerate(zip(dataset_dict['test_x'], dataset_dict['test_y'])):
            data = self.convert_pin_graph_to_pyg(G)
            if data is not None:
                data.y = torch.tensor([label], dtype=torch.long)
                data.set = 'test'
                data_list.append(data)
        
        print(f"Loaded {len(data_list)} examples "
              f"({sum(1 for d in data_list if d.set == 'train')} train, "
              f"{sum(1 for d in data_list if d.set == 'test')} test)")
                    
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])

    def convert_pin_graph_to_pyg(self, G):
        # Convert pin-level graph to PyG Data object
        if G.number_of_nodes() == 0:
            return None
            
        # Create node features
        node_features = []
        node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
        
        for node, attr in G.nodes(data=True):
            feat = np.zeros(10, dtype=np.float32)  # Feature vector
            
            node_type = attr.get('type', '')
            comp_type = attr.get('comp_type', '')
            pin_type = attr.get('pin', '')
            
            # Encode node type
            if node_type in {'component', 'subcircuit'}:
                feat[0] = 1.0
                # Encode component type
                if comp_type in ['R', 'C', 'V', 'X']:
                    comp_idx = ['R', 'C', 'V', 'X'].index(comp_type)
                    feat[1 + comp_idx] = 1.0
            elif node_type == 'pin':
                feat[5] = 1.0
                # Encode pin type
                if pin_type in ['1', '2', 'pos', 'neg']:
                    pin_idx = ['1', '2', 'pos', 'neg'].index(pin_type)
                    feat[6 + pin_idx] = 1.0
            elif node_type == 'net':
                feat[9] = 1.0
                
            node_features.append(feat)
        
        # Create edges
        edges = []
        edge_attrs = []
        
        for u, v, attr in G.edges(data=True):
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            edges.append((u_idx, v_idx))
            
            edge_kind = attr.get('kind', '')
            if edge_kind == 'component_connection':
                edge_attrs.append([1, 0])  # Internal
            elif edge_kind == 'net_connection':
                edge_attrs.append([0, 1])  # External
            else:
                edge_attrs.append([0, 0])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
