import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import re

def extract_representation(dir_name):
    dir_name_lower = dir_name.lower()
    
    if 'component_pin_net' in dir_name_lower:
        return 'component_pin_net'
    elif 'component_pin' in dir_name_lower:
        return 'component_pin'
    elif 'component_net' in dir_name_lower:
        return 'component_net'
    elif 'component_component' in dir_name_lower:
        return 'component_component'
    

    match = re.search(r'fegin_(.+?)_L\d', dir_name_lower)
    if match:
        rep = match.group(1)
        known_reps = ['component_component', 'component_net', 'component_pin', 'component_pin_net']
        if rep in known_reps:
            return rep
    
    return None

def aggregate_by_representation():
    base_dir = Path("fegin_experiments")
    class_names = ['R', 'C', 'V', 'X']
    
    aggregated = {}

    print("Scanning directories...")
    dirs_list = []
    for exp_dir in base_dir.iterdir():
        if exp_dir.is_dir():
            dirs_list.append(exp_dir.name)
    print(f"Found {len(dirs_list)} directories")
    
    for exp_dir in base_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        dir_name = exp_dir.name
        representation = extract_representation(dir_name)
        
        if representation and (exp_dir / "confusion_matrix.npy").exists():
            cm = np.load(exp_dir / "confusion_matrix.npy")
            
            if representation not in aggregated:
                aggregated[representation] = []
            aggregated[representation].append(cm)
    
    for rep, matrices in aggregated.items():
        total_cm = np.sum(matrices, axis=0)
        
        print(f"\n{rep.upper()}: {len(matrices)} experiments")
        print("Aggregated Confusion Matrix:")
        print(total_cm)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{rep} - Aggregated ({len(matrices)} experiments)')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'aggregated_{rep}.png', dpi=150)
        plt.show()
        
        np.save(f'aggregated_cm_{rep}.npy', total_cm)

if __name__ == "__main__":
    aggregate_by_representation()