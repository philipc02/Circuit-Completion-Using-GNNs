from .circuit_datasets import BaseCircuitDataset
import torch
import numpy as np
from torch_geometric.data import Data

class ComponentComponentDataset(BaseCircuitDataset):
    # Dataset for component-component representation
    def convert_graph_to_pyg(self, G):
        if G.number_of_nodes() == 0:
            return None
        
        # Node features (component type and normalized node degree)
        node_features = []
        node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
        degrees = dict(G.degree())
        
        for node, attr in G.nodes(data=True):
            feat = np.zeros(2, dtype=np.float32)
            
            comp_type = attr.get('comp_type', 'R')
            if comp_type in ['R', 'C', 'V', 'X']:
                comp_idx = ['R', 'C', 'V', 'X'].index(comp_type)
                feat[0] = comp_idx / 4.0  # Normalize
            
            feat[1] = degrees[node] / 10.0  # Normalized degree
            
            node_features.append(feat)
        
        # Edges
        edges = []
        for u, v, attr in G.edges(data=True):
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            edges.append((u_idx, v_idx))
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index)