import os
import pickle
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset

class BaseCircuitDataset(InMemoryDataset):
    # Base class for all circuit representations
    def __init__(self, root, name, representation, h, max_nodes_per_hop, node_label, use_rd,transform=None, pre_transform=None, pre_filter=None):
        self.representation = representation
        self.h,self.max_nodes_per_hop, self.node_label, self.use_rd,self.name = h,max_nodes_per_hop, node_label, use_rd,name
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.name}_{self.representation}_processed.pt']
    
    def process(self):
        print(f"Loading {self.representation} representation data...")
        
        # Load the pre-parsed graphs
        graph_folder = f"graphs_{self.name}/graphs_{self.representation}"
        
        # Load dataset splits
        with open(f'data/{self.name}_dataset.pkl', 'rb') as f:
            dataset_info = pickle.load(f)
        
        data_list = []
        
        # Load and convert graphs
        for split in ['train', 'test']:
            files = dataset_info[f'{split}_files']
            
            for graph_file in files:
                graph_path = os.path.join(graph_folder, graph_file)
                
                if not os.path.exists(graph_path):
                    print(f"Warning: {graph_path} not found")
                    continue
                
                with open(graph_path, 'rb') as f:
                    G = pickle.load(f)
                
                # Get masked examples for this circuit
                examples = self.create_masked_examples_from_circuit(G, graph_file)
                
                for example in examples:
                    data = self.convert_graph_to_pyg(example['masked_graph'])
                    if data is not None:
                        data.y = torch.tensor([example['label_idx']], dtype=torch.long)
                        data.set = split
                        data_list.append(data)
        
        print(f"Loaded {len(data_list)} examples ({self.representation})")
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])
    
    def create_masked_examples_from_circuit(self, G, graph_name):
        # Masked examples for component classification
        examples = []
        comp_type_to_idx = {'R': 0, 'C': 1, 'V': 2, 'X': 3}
        
        component_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'component']
        
        for comp_node in component_nodes:
            comp_type = G.nodes[comp_node].get('comp_type')
            if comp_type not in comp_type_to_idx:
                continue
            
            # Create masked graph
            G_masked = self.create_masked_graph(G, comp_node)
            if G_masked is None or G_masked.number_of_nodes() < 2:
                continue
            
            examples.append({
                'original_graph': graph_name,
                'target_component': comp_node,
                'label': comp_type,
                'label_idx': comp_type_to_idx[comp_type],
                'masked_graph': G_masked
            })
        
        return examples
    
    def create_masked_graph(self, G, target_component):
        G_masked = G.copy()
        
        if self.representation in ["component_component", "component_net"]:
            G_masked.remove_node(target_component)
        elif self.representation in ["component_pin", "component_pin_net"]:
            # For pin-based representations: remove component and its pins
            pin_nodes = [node for node in G_masked.neighbors(target_component) 
                        if G_masked.nodes[node].get('type') == 'pin']
            nodes_to_remove = [target_component] + pin_nodes
            G_masked.remove_nodes_from(nodes_to_remove)
        
        # Remove any isolated nodes
        G_masked.remove_nodes_from(list(nx.isolates(G_masked)))
        
        return G_masked
    
    def convert_graph_to_pyg(self, G):
        # Convert graph to PyG Data object: to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement this method")