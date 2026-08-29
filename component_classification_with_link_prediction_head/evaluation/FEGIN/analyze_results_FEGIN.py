import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Global Plot Configuration for Papers ---
plt.rc('font', size=17)          
plt.rc('axes', titlesize=21)     
plt.rc('axes', labelsize=19)     
plt.rc('xtick', labelsize=17)    
plt.rc('ytick', labelsize=17)    
plt.rc('legend', fontsize=17)    
plt.rc('figure', titlesize=23)   

def load_results(results_dir='hyperparameter_search_ltspice_examples_round_3'):
    with open(os.path.join(results_dir, 'all_results.json'), 'r') as f:
        all_results = json.load(f)
    with open(os.path.join(results_dir, 'best_results.json'), 'r') as f:
        best_results = json.load(f)
    return all_results, best_results

def create_comparison_chart(best_results, output_dir='analysis'):
    os.makedirs(output_dir, exist_ok=True)
    
    reps = []
    f1_scores = []
    
    # 1. Add Experimental Results
    for rep, result in best_results.items():
        if result and result.get('success'):
            label_map = {
                'component_component': 'comp.',
                'component_net':       'comp-net',
                'component_pin':       'comp-pin',
                'component_pin_net':   'comp-pin\n-net',
            }
            reps.append(label_map.get(rep, rep.replace('_', ' ').title()))
            f1_scores.append(result['f1'])

    num_experimental = len(reps)
    
    # 2. Add LLM baseline only
    reps.append('LLM\n(baseline)')
    f1_scores.append(0.220)
    
    fig, ax1 = plt.subplots(figsize=(10, 7))
    
    bar_width = 0.85  
    group_gap = 0.4   
    
    x_pos = []
    current_pos = 0
    for i in range(len(reps)):
        if i == num_experimental:
            current_pos += group_gap
        x_pos.append(current_pos)
        current_pos += 1.0 
    
    x_pos = np.array(x_pos)
    colors = ['#1f77b4', '#d62728', '#9467bd', '#2ca02c', '#bcbd22']
    
    bars = ax1.bar(x_pos, f1_scores, alpha=0.85, width=bar_width, color=colors[:len(reps)])
    
    ax1.set_ylabel('F1 Score', fontweight='bold', labelpad=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(reps, fontsize=12, rotation=0, fontweight='medium')
    ax1.tick_params(axis='x', pad=5)
    ax1.grid(True, alpha=0.2, axis='y', linestyle='--')
    
    ax1.set_ylim([0.1, 0.9])
    ax1.set_yticks(np.arange(0.1, 1.0, 0.1))
    
    for bar, score, color in zip(bars, f1_scores, colors):
        height = bar.get_height()
        # clip label inside plot if bar reaches top
        label_y = min(height + 0.01, 0.87)
        ax1.text(bar.get_x() + bar.get_width()/2., label_y,
                 f'{score:.3f}', ha='center', va='bottom',
                 fontsize=13, fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'representation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_parameter_analysis(all_results, output_dir='analysis'):
    os.makedirs(output_dir, exist_ok=True)
    df_data = []
    for result in all_results:
        if result.get('success') and result.get('f1') is not None:
            row = {'representation': result['representation'], 'f1': float(result['f1'])}
            params = result['params']
            row.update({
                'layers': int(params['layers']), 'hiddens': int(params['hiddens']),
                'batch_size': int(params['batch_size']), 'lr': float(params['lr']),
                'emb_size': int(params['emb_size'])
            })
            df_data.append(row)
    
    df = pd.DataFrame(df_data)
    if df.empty: return
    
    reps = df['representation'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, rep in enumerate(reps):
        ax = axes[idx//2, idx%2]
        rep_data = df[df['representation'] == rep]
        pivot = rep_data.pivot_table(values='f1', index='layers', columns='hiddens', aggfunc='mean')
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel('Hidden Channels')
        ax.set_ylabel('Layers')
        ax.set_title(f'{rep.replace("_", " ").title()}')
        
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                if not np.isnan(pivot.iloc[i, j]):
                    ax.text(j, i, f'{pivot.iloc[i, j]:.3f}', ha="center", va="center", fontsize=12)
        
        plt.colorbar(im, ax=ax).set_label('Mean F1', size=14)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'heatmaps.png'), dpi=300)
    plt.show()

def paramter_effects(all_results, output_dir='analysis'):
    df_data = []
    for result in all_results:
        if result.get('success') and result.get('f1') is not None:
            row = {'representation': result['representation'], 'f1': result['f1']}
            row.update(result['params'])
            df_data.append(row)
    
    df = pd.DataFrame(df_data)
    if df.empty: return
    
    for rep in df['representation'].unique():
        rep_data = df[df['representation'] == rep]
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        parameters = ['layers', 'hiddens', 'batch_size', 'lr', 'emb_size']
        param_names = ['Layers', 'Hidden', 'Batch Size', 'Learning Rate', 'Emb Size']
        
        for idx, (param, param_name) in enumerate(zip(parameters, param_names)):
            ax = axes[idx]
            grouped = rep_data.groupby(param)['f1'].agg(['mean', 'std', 'count'])
            grouped.index = pd.to_numeric(grouped.index, errors='coerce')
            grouped = grouped.sort_index()
            
            x, y, y_err = grouped.index, grouped['mean'], grouped['std']
            ax.errorbar(x, y, yerr=y_err, fmt='o-', capsize=6, linewidth=2, markersize=10)
            ax.set_xlabel(param_name)
            ax.set_ylabel('F1 Score')
            ax.grid(True, alpha=0.3)
            
            for xi, yi, count in zip(x, y, grouped['count']):
                ax.text(xi, yi + 0.005, f'n={count}', ha='center', va='bottom', fontsize=12)
        
        axes[5].axis('off')
        plt.suptitle(f'Parameter Analysis: {rep.replace("_", " ").title()}', fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f'parameter_effects_{rep}.png'), dpi=300)
        plt.show()

def main():
    output_dir = 'analysis_ltspice_examples_round_3'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading results...")
    all_results, best_results = load_results('hyperparameter_search_ltspice_examples_round_3')
    
    print("Creating visualizations (Single Chart with F1 ± STD labels)...")
    create_comparison_chart(best_results, output_dir)
    create_parameter_analysis(all_results, output_dir)
    paramter_effects(all_results, output_dir)
    
    print(f"\nAnalysis complete! Output directory: '{output_dir}'")

if __name__ == "__main__":
    main()