import pickle
import os

print("=== Dataset Analysis ===")

# 1. Check dataset.pkl
with open('data/amsnet_dataset.pkl', 'rb') as f:
    dataset_info = pickle.load(f)

print(f"Train files: {len(dataset_info['train_files'])}")
print(f"Test files: {len(dataset_info['test_files'])}")

# 2. Check if graphs exist
base_path = "graphs_amsnet/graphs_component_component"
missing_files = []

for circuit in dataset_info['train_files']:
    if circuit in dataset_info['circuit_to_files']:
        filename = dataset_info['circuit_to_files'][circuit].get('component_component')
        if filename:
            full_path = os.path.join(base_path, filename)
            if not os.path.exists(full_path):
                missing_files.append((circuit, filename))
        else:
            print(f"No component_component file for {circuit}")
    else:
        print(f"Circuit {circuit} not in circuit_to_files")

if missing_files:
    print(f"\nMissing {len(missing_files)} graph files:")
    for circuit, filename in missing_files:
        print(f"  {circuit}: {filename}")
else:
    print("\nAll graph files exist!")