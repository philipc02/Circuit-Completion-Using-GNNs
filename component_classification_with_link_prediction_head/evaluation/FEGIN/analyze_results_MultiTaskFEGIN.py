import os

import json

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# --- Global Plot Configuration for Papers ---

plt.rc('font', size=16)

plt.rc('axes', titlesize=20)

plt.rc('axes', labelsize=18)

plt.rc('xtick', labelsize=16)

plt.rc('ytick', labelsize=16)

plt.rc('legend', fontsize=16)

plt.rc('figure', titlesize=22)



def load_results(results_dir):

    """Loads results from a specific directory path."""

    all_results_path = os.path.join(results_dir, 'all_results.json')

    

    if not os.path.exists(all_results_path):

        print(f"Warning: Could not find {all_results_path}")

        return [], {}



    with open(all_results_path, 'r', encoding='utf-8') as f:

        all_results = json.load(f)

    

    best_results_path = os.path.join(results_dir, 'best_results.json')

    if os.path.exists(best_results_path):

        with open(best_results_path, 'r', encoding='utf-8') as f:

            try:

                best_results = json.load(f)

            except json.JSONDecodeError:

                best_results = {}

    else:

        best_results = {}

    

    if not best_results:

        print(f"Extracting best results from: {os.path.basename(results_dir)}")

        best_results = extract_best_results(all_results)

    

    return all_results, best_results



def extract_best_results(all_results):

    best_results = {}

    by_representation = {}

    for result in all_results:

        if result.get('success') and result.get('combined_score') is not None:

            rep = result['representation']

            if rep not in by_representation:

                by_representation[rep] = []

            by_representation[rep].append(result)

    

    for rep, results in by_representation.items():

        if results:

            best_result = max(results, key=lambda x: x['combined_score'])

            best_results[rep] = best_result

    return best_results



def create_comparison_chart(best_results, output_dir):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    if not best_results: return
    
    reps, combined_scores, node_f1_scores, edge_auc_scores = [], [], [], []
    label_map = {'component': 'comp.', 'component_net': 'comp-net', 
                 'component_pin': 'comp-pin', 'component_pin_net': 'comp-pin-net'}

    for rep, result in best_results.items():
        if result and result.get('success'):
            reps.append(label_map.get(rep, rep))
            combined_scores.append(result['combined_score'])
            node_f1_scores.append(result['node_f1'])
            edge_auc_scores.append(result['edge_auc'])

    # Create figure with two subplots of different widths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [3, 1]})

    # --- LEFT PANEL: MultiTaskFEGIN Results ---
    n = len(reps)
    width = 0.25
    gap = 0.02
    x_pos = np.arange(n) * (3 * width + 0.4)

    ax1.bar(x_pos - width - gap, combined_scores, width, label='Combined', color='#1f77b4', alpha=0.85)
    ax1.bar(x_pos, node_f1_scores, width, label='Node F1', color='#ff7f0e', alpha=0.85)
    ax1.bar(x_pos + width + gap, edge_auc_scores, width, label='Edge AUC', color='#2ca02c', alpha=0.85)

    ax1.set_title("(a) Multi-Task Joint Prediction", fontweight='bold', pad=20)
    ax1.set_ylabel("Score", fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(reps)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.2, axis='y', linestyle='--')

    # --- RIGHT PANEL: Link Prediction Comparison ---
    # Find your best Edge AUC to compare
    best_own_edge = max(edge_auc_scores)
    baseline_edge = 0.990
    
    comp_labels = ['Best\n(Ours)', 'Pan et al.\n[11]']
    comp_values = [best_own_edge, baseline_edge]
    comp_colors = ['#2ca02c', '#d62728'] # Green vs Red

    ax2.bar(comp_labels, comp_values, color=comp_colors, alpha=0.8)
    ax2.set_title("(b) Link Prediction", fontweight='bold', pad=20)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.2, axis='y', linestyle='--')
    
    # Add labels on top of the comparison bars
    for i, v in enumerate(comp_values):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

    # Add annotation about the ground-truth disparity
    ax2.text(0.5, 0.05, "*Baseline uses\nGround-Truth inputs", 
             transform=ax2.transAxes, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'representation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()



def main():

    # --- YOUR SPECIFIC FOLDER PATHS ---

    path_pin = r"C:\Users\chris\OneDrive\Desktop\Chrissa\UNIVER~1\BACHEL~1\SEM7~1\BACHEL~1\CIRCUI~2\COMPON~1\EVALUA~1\FEGIN\MULTIT~3"

    path_pin_net = r"C:\Users\chris\OneDrive\Desktop\Chrissa\UNIVER~1\BACHEL~1\SEM7~1\BACHEL~1\CIRCUI~2\COMPON~1\EVALUA~1\FEGIN\MULTIT~4"

    

    results_dirs = [path_pin, path_pin_net]

    output_dir = os.path.join(path_pin_net, "analysis_combined")

    os.makedirs(output_dir, exist_ok=True)



    all_results_combined = []

    best_results_combined = {}



    print("Loading data...")

    for d in results_dirs:

        if os.path.exists(d):

            all_r, best_r = load_results(d)

            all_results_combined.extend(all_r)

            best_results_combined.update(best_r)

        else:

            print(f"Skipping missing directory: {d}")



    if not all_results_combined:

        print("No data found.")

        return



    create_comparison_chart(best_results_combined, output_dir)

    

    # Best Printout

    if best_results_combined:

        best_rep = max(best_results_combined, key=lambda k: best_results_combined[k]['combined_score'])

        print(f"\nBest Overall: {best_rep} ({best_results_combined[best_rep]['combined_score']:.4f})")



if __name__ == "__main__":

    main()