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

# Try to load the dataset pickle
dataset_paths = [
    'data/amsnet_dataset.pkl',
    'component_classification_with_link_prediction_head/evaluation/FEGIN/data/amsnet_dataset.pkl',
]

dataset_info = None
for path in dataset_paths:
    if os.path.exists(path):
        with open(path, 'rb') as f:
            dataset_info = pickle.load(f)
        print(f"✓ Loaded dataset from: {path}")
        break

if dataset_info is None:
    print("✗ Could not find amsnet_dataset.pkl")
    print("  Please run prepare_all_datasets.py first")
    sys.exit(1)

print(f"\nDataset info:")
print(f"  Train files: {len(dataset_info['train_files'])}")
print(f"  Test files: {len(dataset_info['test_files'])}")

# Try to create the dataset
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

# Load one graph to inspect
first_train = dataset_info['train_files'][0]
first_filename = dataset_info['circuit_to_files'][first_train][representation]
graph_path = os.path.join(actual_graph_folder, first_filename)

print(f"\nInspecting first training graph: {first_filename}")
with open(graph_path, 'rb') as f:
    G = pickle.load(f)

print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")
print(f"  Node attributes: {list(G.nodes(data=True))[0]}")

# Try to create dataset
print("\n" + "="*60)
print("Attempting to create MultiTaskCircuitDataset...")
print("="*60)

try:
    train_dataset = MultiTaskCircuitDataset(
        dataset_info['train_files'],
        dataset_info['circuit_to_files'],
        representation,
        actual_graph_folder,
        is_test=False,
        reprocess=True
    )
    
    print(f"\n✓ Dataset created successfully!")
    print(f"  Total samples: {len(train_dataset)}")
    
    if len(train_dataset) == 0:
        print("\n✗ PROBLEM: Dataset has 0 samples!")
        print("\nPossible causes:")
        print("  1. Graphs are too small (< minimum required nodes)")
        print("  2. Masking strategy fails on these graphs")
        print("  3. Component types don't match expected types")
        
        # Check component types in graphs
        print("\nChecking component types in graphs...")
        all_comp_types = set()
        for circuit_name in dataset_info['train_files'][:3]:  # Check first 3
            filename = dataset_info['circuit_to_files'][circuit_name][representation]
            with open(os.path.join(actual_graph_folder, filename), 'rb') as f:
                G = pickle.load(f)
            for _, attr in G.nodes(data=True):
                if 'comp_type' in attr:
                    all_comp_types.add(attr['comp_type'])
        
        print(f"  Found component types: {all_comp_types}")
        print(f"  Expected component types: {train_dataset.component_types if hasattr(train_dataset, 'component_types') else 'unknown'}")
    
    else:
        print(f"\n✓ SUCCESS: Dataset has {len(train_dataset)} samples")
        # Try to get one sample
        sample = train_dataset[0]
        print(f"  Sample keys: {sample.keys()}")
        
except Exception as e:
    print(f"\n✗ Failed to create dataset: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Debug complete")
print("="*60)