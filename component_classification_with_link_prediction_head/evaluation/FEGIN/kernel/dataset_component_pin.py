from .circuit_datasets import BaseCircuitDataset
import torch
import numpy as np
from torch_geometric.data import Data


class ComponentPinDataset(BaseCircuitDataset):
    @property
    def num_features(self):
        return 17

    @property 
    def num_classes(self):
        return 5
    # Dataset for component - pin - pin - component representation
    def convert_graph_to_pyg(self, G):
        if G.number_of_nodes() == 0:
            return None
        
        # Node features with edge attributes
        node_features = []
        node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
        degrees = dict(G.degree())
        
        for node, attr in G.nodes(data=True):
            node_type = attr.get('type', '')
            comp_type = attr.get('comp_type', '')
            pin_type = attr.get('pin', '')
            
            feat = np.zeros(17, dtype=np.float32) # bigger feature vector for encoding pin type
            
            if node_type == 'component':
                feat[0] = 1.0  # node type: component
                feat[1] = degrees[node] / 10.0
                if comp_type in ['R', 'C', 'V', 'X', 'M']:
                    comp_idx = ['R', 'C', 'V', 'X', 'M'].index(comp_type)
                    feat[2 + comp_idx] = 1.0
            elif node_type == 'pin':
                feat[7] = 1.0  # node type: pin
                feat[8] = degrees[node] / 5.0
                if pin_type in ['1', '2', 'pos', 'neg', 'p', 'drain', 'gate', 'source']:
                    pin_types = ['1', '2', 'pos', 'neg', 'p', 'drain', 'gate', 'source']
                    pin_idx = pin_types.index(pin_type)
                    feat[8 + pin_idx] = 1.0
            
            node_features.append(feat)
        
        # Edges with attributes
        # TODO: integrate edge attributes into message passing in model
        edges = []
        edge_attrs = []
        
        for u, v, attr in G.edges(data=True):
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            edges.append((u_idx, v_idx))
            
            edge_kind = attr.get('kind', '')
            if edge_kind == 'internal':
                edge_attrs.append([1, 0])  # Internal
            elif edge_kind == 'external':
                edge_attrs.append([0, 1])  # External
            else:
                edge_attrs.append([0, 0])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)