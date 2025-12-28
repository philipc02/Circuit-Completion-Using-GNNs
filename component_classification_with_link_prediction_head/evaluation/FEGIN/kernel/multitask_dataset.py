import os
import pickle
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset


class MultiTaskCircuitDataset(InMemoryDataset):
    # Remove component (and corresponding pins depending on representation)
    # Use masked graph for component classification
    # Generate candidate edges for link prediction
    # Label which edges should exist based on original graph (positive, negative)

    
    def __init__(self, root, name, representation, h, max_nodes_per_hop, 
                 node_label, use_rd, neg_sampling_ratio=5.0, max_pins=2, split='train',
                 transform=None, pre_transform=None, pre_filter=None):
        self.representation = representation
        self.h,self.max_nodes_per_hop, self.node_label, self.use_rd,self.name = h,max_nodes_per_hop, node_label, use_rd,name
        self.neg_sampling_ratio = neg_sampling_ratio  # ratio of negative to positive samples
        self.max_pins = max_pins
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter)
        # self.data, self.slices = torch.load(self.processed_paths[0])

        loaded = torch.load(self.processed_paths[0])
        all_class_data = loaded['class_data']
        all_pin_predictions = loaded['pin_predictions']
        all_candidate_edges = loaded['candidate_edges']
        all_edge_labels = loaded['edge_labels']
        all_pin_positions = loaded['pin_positions']
        
        # Filter by split
        self.class_data = []
        self.pin_predictions = []
        self.candidate_edges = []
        self.edge_labels = []
        self.pin_positions = []
        
        for i in range(len(all_class_data)):
            if hasattr(all_class_data[i], 'set') and all_class_data[i].set == split:
                self.class_data.append(all_class_data[i])
                self.pin_predictions.append(all_pin_predictions[i])
                self.candidate_edges.append(all_candidate_edges[i])
                self.edge_labels.append(all_edge_labels[i])
                self.pin_positions.append(all_pin_positions[i])
        
        self.num_examples = len(self.class_data)
        
    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # Return a dictionary with all elements for this example
        return {
            'classification': self.class_data[idx],
            'pin_predictions': self.pin_predictions[idx],
            'candidate_edges': self.candidate_edges[idx],
            'edge_labels': self.edge_labels[idx],
            'pin_positions': self.pin_positions[idx]
        }


    
    @property
    def processed_file_names(self):
        return [f'{self.name}_{self.representation}_multitask_processed.pt']
    
    def process(self):
        print(f"Processing {self.representation} representation for multi-task learning...")
        
        graph_folder = f"../../../graph_parsers/graphs_{self.name}/graphs_{self.representation}"
        
        with open(f'data/{self.name}_dataset.pkl', 'rb') as f:
            dataset_info = pickle.load(f)
        
        examples_list = []
        circuit_to_files = dataset_info['circuit_to_files']
        
        # Process training circuits
        for circuit_name in dataset_info['train_files']:
            rep_files = circuit_to_files.get(circuit_name, {})
            graph_file = rep_files.get(self.representation)
            
            if not graph_file:
                continue
            
            graph_path = os.path.join(graph_folder, graph_file)
            if not os.path.exists(graph_path):
                continue
            
            with open(graph_path, 'rb') as f:
                G = pickle.load(f)
            
            examples = self.create_multitask_examples(G, graph_file, 'train')
            examples_list.extend(examples)
        
        # Process test circuits
        for circuit_name in dataset_info['test_files']:
            rep_files = circuit_to_files.get(circuit_name, {})
            graph_file = rep_files.get(self.representation)
            
            if not graph_file:
                continue
            
            graph_path = os.path.join(graph_folder, graph_file)
            if not os.path.exists(graph_path):
                continue
            
            with open(graph_path, 'rb') as f:
                G = pickle.load(f)
            
            examples = self.create_multitask_examples(G, graph_file, 'test')
            examples_list.extend(examples)
        
        print(f"Created {len(examples_list)} multi-task examples ({self.representation})")
        
        if self.pre_filter is not None:
            examples_list = [example for example in examples_list if self.pre_filter(example)]
        
        if self.pre_transform is not None:
            examples_list = [self.pre_transform(example) for example in examples_list]
        
        # - class_data (single graph)
        # - pin_predictions (list of graphs)
        # - candidate_edges (list of tensors)
        # - edge_labels (list of tensors)
        # - pin_positions (list of indices)
        
        all_class_data = []
        all_pin_predictions = []  
        all_candidate_edges = []
        all_edge_labels = []    
        all_pin_positions = []

        for example_dict in examples_list:
            all_class_data.append(example_dict['classification'])
            all_pin_predictions.append(example_dict['pin_predictions'])
            all_candidate_edges.append(example_dict['candidate_edges'])
            all_edge_labels.append(example_dict['edge_labels'])
            all_pin_positions.append(example_dict['pin_positions'])

        # Save everything together
        torch.save({
            'class_data': all_class_data,          # List of Data objects
            'pin_predictions': all_pin_predictions, # List of lists of Data objects
            'candidate_edges': all_candidate_edges, # List of lists of tensors
            'edge_labels': all_edge_labels,         # List of lists of tensors
            'pin_positions': all_pin_positions,     # List of lists of indices
            'num_examples': len(all_class_data)
        }, self.processed_paths[0])
    
    def create_multitask_examples(self, G, graph_name, split):
        # examples with both classification labels and edge prediction targets
        examples = []
        comp_type_to_idx = {'R': 0, 'C': 1, 'V': 2, 'X': 3}
        
        component_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'component']
        
        for comp_node in component_nodes:
            comp_type = G.nodes[comp_node].get('comp_type')
            if comp_type not in comp_type_to_idx:
                continue

            # TASK 1: Component Classification (mask component + ALL pins)
            G_class_masked = self.mask_for_classification(G, comp_node)
            if G_class_masked is None or G_class_masked.number_of_nodes() < 2:
                continue
            
            # Create classification example
            class_data = self.convert_graph_to_pyg(G_class_masked)
            if class_data is None:
                continue

            class_data.set = split
            class_data.y = torch.tensor([comp_type_to_idx[comp_type]], dtype=torch.long)
            class_data.target_component = comp_node
            class_data.comp_type = comp_type

            # TASK 2: Pin Connection Prediction (one example per pin)
                        
            # Get all pins of this component
            pin_nodes = [n for n in G.neighbors(comp_node) 
                        if G.nodes[n].get('type') == 'pin']
            pin_nodes.sort(key=lambda n: G.nodes[n].get('pin', ''))

            pin_predictions = []
            all_candidate_edges = []
            all_edge_labels = []
            pin_positions = []

            for pin_idx, pin_node in enumerate(pin_nodes):
                if pin_idx >= self.max_pins:  # Only handle up to max_pins
                    break
                
                # Get the net this pin should connect to
                connected_nets = [n for n in G.neighbors(pin_node) if G.nodes[n].get('type') == 'net']
                
                if not connected_nets:
                    continue
                
                target_net = connected_nets[0]  # For our current format just one net should be connected to each pin 
                
                # Create graph with only this pin masked
                G_pin_masked = self.mask_for_pin_prediction(G, comp_node, pin_node)
                if G_pin_masked is None or G_pin_masked.number_of_nodes() < 2:
                    continue
                
                # Generate candidate edges from node to all nets
                candidate_edges, edge_labels = self.generate_candidate_edges(G_pin_masked, target_net)
                
                pin_data = self.convert_graph_to_pyg(G_pin_masked)
                if pin_data is not None:
                    pin_data.pin_position = torch.tensor([pin_idx], dtype=torch.long)
                    pin_predictions.append(pin_data)
                    all_candidate_edges.append(candidate_edges)
                    all_edge_labels.append(edge_labels)
                    pin_positions.append(pin_idx)
                    
            if pin_predictions:
                examples.append({
                    'classification': class_data,
                    'pin_predictions': pin_predictions,  # LIST of pin graphs
                    'candidate_edges': all_candidate_edges,  # LIST of candidate edges
                    'edge_labels': all_edge_labels,  # LIST of edge labels
                    'pin_positions': pin_positions  # LIST of pin positions
                })
        
        return examples
    
    def mask_for_classification(self, G, comp_node):
        # Mask component and ALL its pins for classification
        G_masked = G.copy()
        removed_nodes = [comp_node]
        
        pin_nodes = [n for n in G.neighbors(comp_node) if G.nodes[n].get('type') == 'pin']
        removed_nodes.extend(pin_nodes)
        
        G_masked.remove_nodes_from(removed_nodes)
        G_masked.remove_nodes_from(list(nx.isolates(G_masked)))
        
        if G_masked.number_of_nodes() == 0:
            return None
        return G_masked
    
    def mask_for_pin_prediction(self, G, comp_node, target_pin):
        # Mask only specific pin for connection prediction
        G_masked = G.copy()
        
        # Remove only this pin as we want to find all connection between pin and other net nodes
        G_masked.remove_node(target_pin)
        # G_masked.remove_nodes_from(list(nx.isolates(G_masked)))
        
        if G_masked.number_of_nodes() == 0:
            return None
        return G_masked
    
    def get_component_connections(self, G, comp_node):
        connections = set()
        
        if self.representation in ["component_component", "component_net"]:
            connections = set(G.neighbors(comp_node))        
        # For the pin level nodes, we look at the connections from the pin nodes (which will be reinstated)
        elif self.representation in ["component_pin", "component_pin_net"]:
            pin_nodes = [n for n in G.neighbors(comp_node) 
                        if G.nodes[n].get('type') == 'pin']
            for pin in pin_nodes:
                for neighbor in G.neighbors(pin):
                    if neighbor != comp_node:  # add every other connected component other than itself
                        connections.add(neighbor)
        
        return connections
    
    def create_masked_graph_with_tracking(self, G, target_component):
        # Create masked graph and track which nodes were removed

        G_masked = G.copy()
        removed_nodes = [target_component]
        
        if self.representation in ["component_pin", "component_pin_net"]:
            # Remove component and corresponding pins
            pin_nodes = [node for node in G_masked.neighbors(target_component) 
                        if G_masked.nodes[node].get('type') == 'pin']
            removed_nodes.extend(pin_nodes)
        
        G_masked.remove_nodes_from(removed_nodes)
        G_masked.remove_nodes_from(list(nx.isolates(G_masked)))
        
        return G_masked, set(removed_nodes)
    
    def generate_candidate_edges(self, G_masked, target_net):
        # returns candidate_edges: [2, num_candidates] tensor with node pairs
        # and edge_labels: [num_candidates] to differentiate positice and negative edges

        target_net = [target_net]

        node_mapping = {node: i for i, node in enumerate(G_masked.nodes())}
                
        if len(target_net) == 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float)
        
        VIRTUAL_NODE_IDX = -1
        
        # Positive samples
        # Represented as edges from 'virtual new pin node of component node' to existing nodes
        num_pos = len(target_net)
        pos_edges = []
        new_comp_idx = len(node_mapping)  # Virtual new pin node of component node
        for conn in target_net:
            pos_edges.append([VIRTUAL_NODE_IDX, node_mapping[conn]])
        
        # Negative samples
        num_neg = int(num_pos * self.neg_sampling_ratio)
        all_nodes = list(G_masked.nodes())
        negative_candidates = [n for n in all_nodes if n not in target_net]
        
        if len(negative_candidates) > num_neg:
            neg_samples = np.random.choice(negative_candidates, num_neg, replace=False)
        else:
            neg_samples = negative_candidates
        
        neg_edges = []
        for node in neg_samples:
            neg_edges.append([VIRTUAL_NODE_IDX, node_mapping[node]])
        
        # Combine positive and negative samples
        all_edges = pos_edges + neg_edges
        all_labels = [1.0] * num_pos + [0.0] * len(neg_edges)
        
        if len(all_edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float)
        
        candidate_edges = torch.tensor(all_edges, dtype=torch.long).t()
        edge_labels = torch.tensor(all_labels, dtype=torch.float)
        
        return candidate_edges, edge_labels
    
    def convert_graph_to_pyg(self, G):
        if G.number_of_nodes() == 0:
            return None
        
        # Node features
        node_features = []
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        
        node_features = self.get_node_features(G, self.representation)
        
        try:
            x = torch.tensor(np.array(node_features), dtype=torch.float)
        except Exception as e:
            # If conversion fails, dump debug info to help locate ragged vectors.
            print("ERROR: failed to convert node_features list to numpy array. Debug info:")
            for i, feat in enumerate(node_features):
                try:
                    print(f"  node {i}: type={type(feat)}, shape={getattr(feat, 'shape', None)}")
                except Exception:
                    print(f"  node {i}: unable to inspect feature")
            raise

        representation = self.representation
        if representation == "component_component":
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
            
            data = Data(x=x, edge_index=edge_index)
        elif representation == "component_net":            
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
            
            data = Data(x=x, edge_index=edge_index)
        elif representation == "component_pin":
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
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        elif representation == "component_pin_net":
            # Edges with attributes
            edges = []
            edge_attrs = []
            
            for u, v, attr in G.edges(data=True):
                u_idx = node_mapping[u]
                v_idx = node_mapping[v]
                edges.append((u_idx, v_idx))
                
                edge_kind = attr.get('kind', '')
                if edge_kind == 'component_connection':  # internal
                    edge_attrs.append([1, 0])
                elif edge_kind == 'net_connection':  # external
                    edge_attrs.append([0, 1])
                else:
                    edge_attrs.append([0, 0])
            
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 2), dtype=torch.float)
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data
    
    def get_node_features(self, G, representation):
        if G.number_of_nodes() == 0:
                return None
        

        if representation == "component_component":
            # Node features (component type and normalized node degree)
            node_features = []
            degrees = dict(G.degree())
            
            for node, attr in G.nodes(data=True):
                node_type = attr.get('type', '')
                comp_type = attr.get('comp_type', '')
                feat = np.zeros(6, dtype=np.float32)
                
                feat[0] = 1.0  # node type: component
                feat[1] = degrees[node] / 10.0
                if comp_type in ['R', 'C', 'V', 'X']:
                    comp_idx = ['R', 'C', 'V', 'X'].index(comp_type)
                    feat[2 + comp_idx] = 1.0
                
                node_features.append(feat)
        elif representation == "component_net":            
            # Node features (node type, component type, normalized node degree)
            node_features = []
            node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
            degrees = dict(G.degree())
            
            for node, attr in G.nodes(data=True):
                node_type = attr.get('type', '')
                comp_type = attr.get('comp_type', '')
                
                feat = np.zeros(8, dtype=np.float32)
                
                if node_type == 'component':
                    feat[0] = 1.0  # node type: component
                    feat[1] = degrees[node] / 10.0
                    if comp_type in ['R', 'C', 'V', 'X']:
                        comp_idx = ['R', 'C', 'V', 'X'].index(comp_type)
                        feat[2 + comp_idx] = 1.0
                else:  # net node
                    feat[6] = 1.0  # node type: net
                    feat[7] = degrees[node] / 20.0
                                
                node_features.append(feat)
        elif representation == "component_pin":
            # Node features with edge attributes
            node_features = []
            node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
            degrees = dict(G.degree())
            
            for node, attr in G.nodes(data=True):
                node_type = attr.get('type', '')
                comp_type = attr.get('comp_type', '')
                pin_type = attr.get('pin', '')
                
                feat = np.zeros(16, dtype=np.float32) # bigger feature vector for encoding pin type
                
                if node_type == 'component':
                    feat[0] = 1.0  # node type: component
                    feat[1] = degrees[node] / 10.0
                    if comp_type in ['R', 'C', 'V', 'X']:
                        comp_idx = ['R', 'C', 'V', 'X'].index(comp_type)
                        feat[2 + comp_idx] = 1.0
                elif node_type == 'pin':
                    feat[6] = 1.0  # node type: pin
                    feat[7] = degrees[node] / 5.0
                    if pin_type in ['1', '2', 'pos', 'neg', 'p']:
                        pin_types = ['1', '2', 'pos', 'neg', 'p']
                        pin_idx = pin_types.index(pin_type)
                        feat[8 + pin_idx] = 1.0
                
                node_features.append(feat)
        elif representation == "component_pin_net":
            # Node features
            node_features = []
            node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
            degrees = dict(G.degree())
            
            for node, attr in G.nodes(data=True):
                node_type = attr.get('type', '')
                comp_type = attr.get('comp_type', '')
                pin_type = attr.get('pin', '')
                
                feat = np.zeros(16, dtype=np.float32)  # bigger feature vector for encoding pin type

                if node_type == 'component':
                    feat[0] = 1.0  # node type: component
                    feat[1] = degrees[node] / 10.0
                    if comp_type in ['R', 'C', 'V', 'X']:
                        comp_idx = ['R', 'C', 'V', 'X'].index(comp_type)
                        feat[2 + comp_idx] = 1.0
                elif node_type == 'pin':
                    feat[6] = 1.0  # node type: pin
                    feat[7] = degrees[node] / 5.0
                    if pin_type in ['1', '2', 'pos', 'neg', 'p']:
                        pin_idx = ['1', '2', 'pos', 'neg', 'p'].index(pin_type)
                        feat[8 + pin_idx] = 1.0
                elif node_type == 'net':
                    feat[13] = 1.0  # node type: net
                    feat[14] = degrees[node] / 20.0

                node_features.append(feat)
        
        return node_features