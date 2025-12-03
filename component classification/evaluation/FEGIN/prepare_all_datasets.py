import pickle
import os
from collections import Counter

def prepare_all_datasets():
    # Create dataset splits and masked examples for all representations
    representations = [
        "component_component",
        "component_net", 
        "component_pin",
        "component_pin_net"
    ]
    
    # Load all graph files
    base_graph_folder = "../../../graph_parsers/graphs_ltspice_demos"
    
    # TODO: Common list of circuits across all representations for fair comparison
    common_circuits = None
    
    for rep in representations:
        folder = f"{base_graph_folder}/graphs_{rep}"
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found")
            continue
        
        circuits = [f for f in os.listdir(folder) if f.endswith('.gpickle')]
        circuits = [f.split('_')[0] + '.net' for f in circuits]  # get base circuit names
        
        if common_circuits is None:
            common_circuits = set(circuits)
        else:
            common_circuits = common_circuits.intersection(set(circuits))
    
    if common_circuits:
        common_circuits = list(common_circuits)
        # Split into train:test (80:20)
        import random
        random.seed(42)
        random.shuffle(common_circuits)
        split_idx = int(0.8 * len(common_circuits))
        train_files = common_circuits[:split_idx]
        test_files = common_circuits[split_idx:]
        
        dataset_info = {
            'train_files': train_files,
            'test_files': test_files,
            'all_files': common_circuits,
            'representations': representations
        }
        
        with open('data/ltspice_demos_dataset.pkl', 'wb') as f:
            pickle.dump(dataset_info, f)
        
        print(f"Created splits with {len(train_files)} train, {len(test_files)} test circuits")
        print(f"Common across all representations: {len(common_circuits)} circuits")
    else:
        print("No common circuits found across representations")

if __name__ == "__main__":
    prepare_all_datasets()