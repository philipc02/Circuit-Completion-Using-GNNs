import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    f1_stds = []
    params_list = []
    
    for rep, result in best_results.items():
        if result and result.get('success'):
            reps.append(rep.replace('_', ' ').title())
            f1_scores.append(result['f1'])
            f1_stds.append(result['f1_std'])
            params_list.append(result['params'])
    
    # Create bar chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    x_pos = np.arange(len(reps))
    # Map representations against theur best f1 scores
    bars = ax1.bar(x_pos, f1_scores, capsize=10, alpha=0.8, color=['#1f77b4', '#ac386a', '#2ca02c', '#d62728'])
    
    ax1.set_xlabel('Representation', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('Best F1 Scores by Circuit Representation', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(reps, rotation=0)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0.5, 0.8]) # Show just this range on the y axis
    
    # Value labels  (with standard deviation) on bars
    for i, (bar, f1, std) in enumerate(zip(bars, f1_scores, f1_stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{f1:.3f} ± {std:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Parameter table
    param_data = []
    for rep, params in zip(reps, params_list):
        param_data.append([
            rep,
            params['layers'],
            params['hiddens'],
            params['batch_size'],
            f"{params['lr']:.4f}",
            params['emb_size']
        ])
    
    column_labels = ['Representation', 'Layers', 'Hidden', 'Batch Size', 'Learning Rate', 'Emb Size']
    table = ax2.table(cellText=param_data, colLabels=column_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax2.axis('off')
    ax2.set_title('Best Hyperparameters for Each Representation', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'representation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison chart saved to analysis_ltspice_examples_round_3/representation_comparison.png")

def create_parameter_analysis(all_results, output_dir='analysis'):
    os.makedirs(output_dir, exist_ok=True)
    
    df_data = []
    for result in all_results:
        if result.get('success') and result.get('f1') is not None:
            row = {
                'representation': result['representation'],
                'f1': float(result['f1']),
                'f1_std': float(result['f1_std']),
                'elapsed': float(result['elapsed'])
            }
            # Convert all parameter values to appropriate types
            params = result['params']
            row.update({
                'layers': int(params['layers']),
                'hiddens': int(params['hiddens']),
                'batch_size': int(params['batch_size']),
                'lr': float(params['lr']),
                'emb_size': int(params['emb_size']),
                'epochs': int(params['epochs']),
                'h': int(params['h'])
            })
            df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    if df.empty:
        print("No successful results to analyze")
        return
    
    # Heatmap of best parameters (only combination of layers and hidden channels, can be done for other combinations too)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    reps = df['representation'].unique()
    
    for idx, rep in enumerate(reps):
        ax = axes[idx//2, idx%2]
        rep_data = df[df['representation'] == rep]
        
        if 'layers' in rep_data.columns and 'hiddens' in rep_data.columns:
            pivot = rep_data.pivot_table(values='f1', index='layers', columns='hiddens', aggfunc='mean')
            
            im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel('Hidden Channels')
            ax.set_ylabel('Layers')
            ax.set_title(f'{rep.replace("_", " ").title()}: Layers vs Hidden')
            
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    if not np.isnan(pivot.iloc[i, j]):
                        text = ax.text(j, i, f'{pivot.iloc[i, j]:.3f}', ha="center", va="center", color="black", fontsize=8)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('F1 Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmaps.png'), dpi=300)
    plt.show()


def paramter_effects(all_results, output_dir='analysis'):
    df_data = []
    for result in all_results:
        if result.get('success') and result.get('f1') is not None:
            row = {
                'representation': result['representation'],
                'f1': result['f1'],
                'f1_std': result['f1_std'],
            }
            row.update(result['params'])
            df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    if df.empty:
        return
    
    numeric_cols = ['layers', 'hiddens', 'batch_size', 'lr', 'emb_size', 'f1', 'f1_std']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # For each representation: analyze parameter effects
    for rep in df['representation'].unique():
        rep_data = df[df['representation'] == rep]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        parameters = ['layers', 'hiddens', 'batch_size', 'lr', 'emb_size']
        param_names = ['Layers', 'Hidden', 'Batch Size', 'Learning Rate', 'Emb Size']
        
        for idx, (param, param_name) in enumerate(zip(parameters, param_names)):
            ax = axes[idx]
            
            # Group by parameter value
            grouped = rep_data.groupby(param)['f1'].agg(['mean', 'std', 'count'])
            # Sort by parameter value
            grouped.index = pd.to_numeric(grouped.index, errors='coerce')
            grouped = grouped.sort_index()
            
            x = grouped.index
            y = grouped['mean']
            y_err = grouped['std']
            
            ax.errorbar(x, y, yerr=y_err, fmt='o-', capsize=5, linewidth=2, markersize=8)
            ax.set_xlabel(param_name)
            ax.set_ylabel('F1 score')
            ax.set_title(f'{rep}: Effect of {param_name}')
            ax.grid(True, alpha=0.3)
            
            # Add count labels
            for i, (xi, yi, count) in enumerate(zip(x, y, grouped['count'])):
                ax.text(xi, yi + 0.005, f'n={count}', ha='center', va='bottom', fontsize=8)
        
        ax = axes[5]
        sorted_data = rep_data.sort_values('batch_size')
        ax.scatter(sorted_data['batch_size'], sorted_data['f1'], alpha=0.7, s=50)
        ax.set_xlabel('Batch size')
        ax.set_ylabel('F1 score')
        ax.set_title(f'{rep}: Batch size vs f1')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Parameter analysis: {rep.replace("_", " ").title()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'parameter_effects_{rep}.png'), dpi=300)
        plt.show()

def main():
    output_dir = 'analysis_ltspice_examples_round_3'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading results...")
    all_results, best_results = load_results('hyperparameter_search_ltspice_examples_round_3')
    
    print("Creating visualizations...")
    create_comparison_chart(best_results, output_dir)
    create_parameter_analysis(all_results, output_dir)
    paramter_effects(all_results, output_dir)
    
    print(f"\nAnalysis complete! Output directory: '{output_dir}'")

if __name__ == "__main__":
    main()