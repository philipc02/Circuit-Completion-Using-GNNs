import pickle
import os
from collections import Counter, defaultdict

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
    
    # Dictionary to store files for each representation
    rep_files = defaultdict(list)
    
    # Collect all files for each representation
    for rep in representations:
        folder = f"{base_graph_folder}/graphs_{rep}"
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found")
            continue
        
        # Get all gpickle files
        files = [f for f in os.listdir(folder) if f.endswith('.gpickle')]
        rep_files[rep] = files
        
        print(f"{rep}: {len(files)} files")
    
    # Find common circuits (by base name) across all representations
    common_circuits = None
    
    for rep, files in rep_files.items():
        # Extract base names
        base_names = set()
        for f in files:
            base = f
            for r in representations:
                if f.endswith(f"_{r}.gpickle"):
                    base = f[:-len(f"_{r}.gpickle") - 1]  # -1 for underscore
                    break
            else:
                base = f[:-8]  # Remove .gpickle
            base_names.add(base)
        
        if common_circuits is None:
            common_circuits = base_names
        else:
            common_circuits = common_circuits.intersection(base_names)
    
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
            'train_files': defaultdict(list),
            'test_files': defaultdict(list),
            'all_files': common_circuits,
            'representations': representations
        }
        
        for rep in representations:
            for file in train_files:
                # Find the file for this base in this representation
                for f in rep_files[rep]:
                    if file in f:
                        dataset_info['train_files'][rep].append(f)
                        break
            
            for file in test_files:
                for f in rep_files[rep]:
                    if file in f:
                        dataset_info['test_files'][rep].append(f)
                        break
        
        with open('data/ltspice_demos_dataset.pkl', 'wb') as f:
            pickle.dump(dataset_info, f)
        
        print(f"\nCreated splits:")
        for rep in representations:
            print(f"  {rep}: {len(dataset_info['train_files'][rep])} train, "
                f"{len(dataset_info['test_files'][rep])} test")
        
        print(f"\nTotal common circuits: {len(common_circuits)}")
    else:
        print("No common circuits found across representations")

if __name__ == "__main__":
    prepare_all_datasets()