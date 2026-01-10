"""
Debug script to understand why AMS dataset produces 0 training samples
"""
import sys
import os

# Find where the code actually is
search_paths = [
    '/home/philip/projects/Circuit-Completion-Using-GNNs/component_classification_with_link_prediction_head/evaluation/FEGIN',
    '/home/philip/projects/Circuit-Completion-Using-GNNs',
]

for path in search_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)
        print(f"Added to path: {path}")

# Now try to import the dataset classes
try:
    from kernel.multitask_dataset import MultiTaskCircuitDataset
    print("✓ Successfully imported MultiTaskCircuitDataset")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    print("\nPlease run this script from the FEGIN directory or adjust paths")
    sys.exit(1)

import pickle
import networkx as nx
import torch

# Try to load the dataset pickle
dataset_paths = [
    'data/amsnet_dataset.pkl',
    'component_classification_with_link_prediction_head/evaluation/FEGIN/data/amsnet_dataset.pkl',
]

dataset_info = None
dataset_path_found = None
for path in dataset_paths:
    if os.path.exists(path):
        with open(path, 'rb') as f:
            dataset_info = pickle.load(f)
        dataset_path_found = path
        print(f"✓ Loaded dataset from: {path}")
        break

if dataset_info is None:
    print("✗ Could not find amsnet_dataset.pkl")
    print("  Please run prepare_all_datasets.py first")
    sys.exit(1)

print(f"\nDataset info:")
print(f"  Train files: {len(dataset_info['train_files'])}")
print(f"  Test files: {len(dataset_info['test_files'])}")
print(f"  Representations: {dataset_info['representations']}")

# Manually inspect graphs before trying dataset creation
representation = 'component_component'
graph_folder = 'graphs_amsnet/graphs_component_component'

# Find the actual graph folder
graph_paths = [
    graph_folder,
    f'../../../graph_parsers/{graph_folder}',
    f'graph_parsers/{graph_folder}',
]

actual_graph_folder = None
for path in graph_paths:
    if os.path.exists(path):
        actual_graph_folder = path
        print(f"✓ Found graphs at: {path}")
        break

if actual_graph_folder is None:
    print(f"✗ Could not find graph folder: {graph_folder}")
    sys.exit(1)

# Load and inspect all training graphs
print("\n" + "="*60)
print("INSPECTING ALL TRAINING GRAPHS")
print("="*60)

comp_type_to_idx = {'R': 0, 'C': 1, 'V': 2, 'X': 3}
valid_examples_count = 0
invalid_examples_count = 0

for i, circuit_name in enumerate(dataset_info['train_files']):
    filename = dataset_info['circuit_to_files'][circuit_name][representation]
    graph_path = os.path.join(actual_graph_folder, filename)
    
    print(f"\n[{i+1}/{len(dataset_info['train_files'])}] {filename}")
    
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    print(f"  Total nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    
    # Count node types
    component_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'component']
    print(f"  Component nodes: {len(component_nodes)}")
    
    # Count component types
    comp_types = {}
    for node in component_nodes:
        comp_type = G.nodes[node].get('comp_type')
        comp_types[comp_type] = comp_types.get(comp_type, 0) + 1
    print(f"  Component types: {comp_types}")
    
    # Check how many valid examples could be created
    valid_comps = 0
    for comp_node in component_nodes:
        comp_type = G.nodes[comp_node].get('comp_type')
        
        # Check if component type is valid
        if comp_type not in comp_type_to_idx:
            print(f"    ✗ Component {comp_node} has invalid type: {comp_type}")
            continue
        
        # Simulate masking for classification
        G_masked = G.copy()
        G_masked.remove_node(comp_node)
        G_masked.remove_nodes_from(list(nx.isolates(G_masked)))
        
        if G_masked.number_of_nodes() < 2:
            print(f"    ✗ Component {comp_node} ({comp_type}): masked graph too small ({G_masked.number_of_nodes()} nodes)")
            invalid_examples_count += 1
        else:
            print(f"    ✓ Component {comp_node} ({comp_type}): masked graph OK ({G_masked.number_of_nodes()} nodes)")
            valid_comps += 1
            valid_examples_count += 1
    
    print(f"  → Valid examples from this graph: {valid_comps}")

print("\n" + "="*60)
print(f"TOTAL VALID EXAMPLES (manual count): {valid_examples_count}")
print(f"TOTAL INVALID EXAMPLES: {invalid_examples_count}")
print("="*60)

# Now try to actually create the dataset using InMemoryDataset
print("\n" + "="*60)
print("Attempting to create MultiTaskCircuitDataset (InMemoryDataset)...")
print("="*60)

try:
    # Determine root directory from dataset path
    if 'component_classification_with_link_prediction_head/evaluation/FEGIN' in dataset_path_found:
        root_dir = 'component_classification_with_link_prediction_head/evaluation/FEGIN/data'
    else:
        root_dir = 'data'
    
    print(f"Using root directory: {root_dir}")
    
    # Create train dataset with proper InMemoryDataset parameters
    train_dataset = MultiTaskCircuitDataset(
        root=root_dir,
        name='amsnet',
        representation=representation,
        h=2,  # enclosing subgraph hops
        max_nodes_per_hop=10,
        node_label='drnl',
        use_rd=True,
        neg_sampling_ratio=5.0,
        max_pins=2,
        split='train'  # This is the key parameter
    )
    
    print(f"\n✓ Dataset created successfully!")
    print(f"  Total samples: {len(train_dataset)}")
    print(f"  Num features: {train_dataset.num_features}")
    print(f"  Num classes: {train_dataset.num_classes}")
    
    if len(train_dataset) == 0:
        print("\n✗ PROBLEM: Dataset has 0 samples!")
        print("\nPossible causes:")
        print("  1. Graphs are too small (< 2 nodes after masking)")
        print("  2. Component types don't match expected types (R, C, V, X)")
        print("  3. Processing failed during create_multitask_examples()")
        
        # Check if processed file exists
        processed_file = os.path.join(root_dir, 'processed', f'amsnet_{representation}_multitask_processed.pt')
        if os.path.exists(processed_file):
            print(f"\n  Processed file exists: {processed_file}")
            print("  Loading to inspect...")
            loaded = torch.load(processed_file)
            print(f"  Keys in processed file: {loaded.keys()}")
            print(f"  Num examples in file: {loaded.get('num_examples', 'unknown')}")
            print(f"  Num class_data: {len(loaded.get('class_data', []))}")
        else:
            print(f"\n  Processed file NOT found: {processed_file}")
    else:
        print(f"\n✓ SUCCESS: Dataset has {len(train_dataset)} samples")
        
        # Try to get one sample
        sample = train_dataset[0]
        print(f"  Sample keys: {sample.keys()}")
        print(f"  Classification data: {sample['classification']}")
        print(f"  Number of pin predictions: {len(sample['pin_predictions'])}")
        
except Exception as e:
    print(f"\n✗ Failed to create dataset: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Debug complete")
print("="*60)