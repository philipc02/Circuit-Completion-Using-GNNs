import pickle
import os
import random
from collections import defaultdict, Counter
import networkx as nx

def create_dataset_splits():
    # Create train/test splits and save to files
    print("Creating dataset splits...")
    
    graph_folder = "../data/graphs_star_filtered"
    graph_files = [f for f in os.listdir(graph_folder) if f.endswith('_star_filtered.gpickle')]
    
    # Split at circuit level (80:20 train:Test like baseline)
    random.seed(42)
    random.shuffle(graph_files)
    split_idx = int(0.8 * len(graph_files))
    train_files = graph_files[:split_idx]
    test_files = graph_files[split_idx:]
    
    # Save file splits
    dataset_info = {
        'train_files': train_files,
        'test_files': test_files,
        'all_files': graph_files
    }
    
    with open('../data/pin_level_dataset_splits.pkl', 'wb') as f:
        pickle.dump(dataset_info, f)
    
    print(f"Created splits: {len(train_files)} train, {len(test_files)} test circuits")
    return dataset_info

def create_masked_examples(dataset_info):
    # Create masked training examples from the splits
    print("Creating masked training examples...")
    
    graph_folder = "../data/graphs_star_filtered"
    
    train_examples = []
    test_examples = []
    
    # Component type mapping
    comp_type_to_idx = {'R': 0, 'C': 1, 'V': 2, 'X': 3}
    
    # Process training circuits
    for graph_file in dataset_info['train_files']:
        graph_path = os.path.join(graph_folder, graph_file)
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        
        examples = create_masked_examples_from_circuit(G, graph_file, comp_type_to_idx)
        train_examples.extend(examples)
    
    # Process test circuits  
    for graph_file in dataset_info['test_files']:
        graph_path = os.path.join(graph_folder, graph_file)
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        
        examples = create_masked_examples_from_circuit(G, graph_file, comp_type_to_idx)
        test_examples.extend(examples)
    
    # Create the final dataset files (matching baseline format)
    dataset_dict = {
        'train_x': [ex['masked_graph'] for ex in train_examples],
        'test_x': [ex['masked_graph'] for ex in test_examples],
        'train_y': [ex['label_idx'] for ex in train_examples],
        'test_y': [ex['label_idx'] for ex in test_examples],
        'train_files': dataset_info['train_files'],
        'test_files': dataset_info['test_files'],
        'label_mapping': comp_type_to_idx
    }
    
    with open('../data/ltspice_demos_pin_level_GC.pkl', 'wb') as f:
        pickle.dump(dataset_dict, f)
    
    # Save label mapping separately (like baseline)
    with open('../data/ltspice_demos_pin_level_label_mapping.pkl', 'wb') as f:
        pickle.dump(comp_type_to_idx, f)
    
    print(f"Created {len(train_examples)} train and {len(test_examples)} test masked examples")
    
    train_labels = [ex['label'] for ex in train_examples]
    test_labels = [ex['label'] for ex in test_examples]
    
    print("Train label distribution:", dict(Counter(train_labels)))
    print("Test label distribution:", dict(Counter(test_labels)))

def create_masked_examples_from_circuit(G, graph_name, comp_type_to_idx):
    examples = []
    
    component_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') in {'component', 'subcircuit'}]
    
    # One masked example for each component
    for comp_node in component_nodes:
        comp_type = G.nodes[comp_node].get('comp_type')
        if comp_type not in comp_type_to_idx:
            continue
            
        # Create masked graph by removing component and corresponding pins
        G_masked = create_masked_graph(G, comp_node)
        if G_masked is None or G_masked.number_of_nodes() < 3:
            continue
            
        example = {
            'original_graph': graph_name,
            'target_component': comp_node,
            'label': comp_type,
            'label_idx': comp_type_to_idx[comp_type],
            'masked_graph': G_masked
        }
        examples.append(example)
    
    return examples

def create_masked_graph(G, target_component):
    G_masked = G.copy()
    
    # All pin nodes connected to target component
    pin_nodes = [node for node in G_masked.neighbors(target_component) if G_masked.nodes[node].get('type') == 'pin']
    
    # Remove component and coresponding pins
    nodes_to_remove = [target_component] + pin_nodes
    G_masked.remove_nodes_from(nodes_to_remove)
    
    return G_masked

if __name__ == "__main__":
    dataset_info = create_dataset_splits()
    create_masked_examples(dataset_info)