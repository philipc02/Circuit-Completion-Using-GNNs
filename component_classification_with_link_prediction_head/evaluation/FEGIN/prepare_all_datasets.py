import pickle
import os
from collections import Counter

def extract_circuit_name(filename):
    # Remove .gpickle
    if filename.endswith('.gpickle'):
        filename = filename[:-8]
    
    # Remove representation suffixes
    representations = ["component_component", "component_net", "component_pin", "star_filtered"]  #component_pin_net graphs have old naming scheme, using this workaround for now to not have to parse again
    
    for rep in representations:
        if filename.endswith(f"_{rep}"):
            return filename[:-len(f"_{rep}")]
    
    return filename

def prepare_all_datasets():
    # Create dataset splits and masked examples for all representations
    representations = [
        "component_component",
        "component_net", 
        "component_pin",
        "component_pin_net"
    ]
    
    # Load all graph files
    base_graph_folder = "../../../graph_parsers/graphs_ltspice_examples"
    
    # TODO: Common list of circuits across all representations for fair comparison
    # Dictionary to store circuit -> representation -> actual filenames
    circuit_to_files = {}
    
    for rep in representations:
        folder = f"{base_graph_folder}/graphs_{rep}"
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found")
            continue
        
        files = [f for f in os.listdir(folder) if f.endswith('.gpickle')]
        print(f"{rep}: {len(files)} files")
        
        for filename in files:
            circuit_name = extract_circuit_name(filename)
            
            if circuit_name not in circuit_to_files:
                circuit_to_files[circuit_name] = {}
            
            circuit_to_files[circuit_name][rep] = filename
    
    # Find circuits that exist in ALL 4 representations
    common_circuits = []
    for circuit_name, rep_dict in circuit_to_files.items():
        if len(rep_dict) == len(representations):  # any entries in the dict that have a filename for all four representations
            common_circuits.append(circuit_name)
    
    print(f"\nFound {len(common_circuits)} circuits common to all representations")
    
    # if nothing is there
    if not common_circuits:
        print("\nChecking what we have:")
        # Show circuits that exist in at least 3 representations
        for circuit_name, rep_dict in circuit_to_files.items():
            if len(rep_dict) >= 3:
                missing = [r for r in representations if r not in rep_dict]
                print(f"  {circuit_name}: in {len(rep_dict)} reps, missing {missing}")
        return
    

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
        'circuit_to_files': circuit_to_files,  # save mapping of circuit -> rep -> filename 
        'representations': representations
    }
    
    with open('data/ltspice_examples_dataset.pkl', 'wb') as f:
        pickle.dump(dataset_info, f)
    
    print(f"Created splits with {len(train_files)} train, {len(test_files)} test circuits")

if __name__ == "__main__":
    prepare_all_datasets()