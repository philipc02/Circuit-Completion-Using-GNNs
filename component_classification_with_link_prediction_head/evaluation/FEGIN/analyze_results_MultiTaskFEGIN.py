import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_results(results_dir='multitask_hyperparameter_search_component_pin_net'):
    with open(os.path.join(results_dir, 'all_results.json'), 'r') as f:
        all_results = json.load(f)
    
    # Create best_results from all_results if best_results.json is empty or doesn't exist
    best_results_path = os.path.join(results_dir, 'best_results.json')
    if os.path.exists(best_results_path):
        with open(best_results_path, 'r') as f:
            best_results = json.load(f)
    else:
        best_results = {}
    
    # If best_results is empty or invalid, extract best from all_results
    if not best_results or len(best_results) == 0:
        print("Best results file is empty. Extracting best results from all results...")
        best_results = extract_best_results(all_results)
    
    return all_results, best_results

def extract_best_results(all_results):
    """Extract the best result for each representation from all_results"""
    best_results = {}
    
    # Group by representation
    by_representation = {}
    for result in all_results:
        if result.get('success') and result.get('combined_score') is not None:
            rep = result['representation']
            if rep not in by_representation:
                by_representation[rep] = []
            by_representation[rep].append(result)
    
    # Find best for each representation
    for rep, results in by_representation.items():
        if results:
            best_result = max(results, key=lambda x: x['combined_score'])
            best_results[rep] = best_result
            print(f"Best for {rep}: Combined Score = {best_result['combined_score']:.4f}")
    
    return best_results

def create_comparison_chart(best_results, output_dir='analysis_multitask_hyperparameter_search_component_pin_net'):
    # Convert to absolute path to ensure consistency
    output_dir = os.path.abspath(output_dir)
    
    # Ensure directory exists with explicit error checking
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory created/verified: {output_dir}")
        print(f"Directory exists: {os.path.exists(output_dir)}")
    except Exception as e:
        print(f"Error creating directory: {e}")
        return
    
    # Check if we have any best results
    if not best_results:
        print("No best results to create comparison chart!")
        return
    
    # Check if we have any best results
    if not best_results:
        print("No best results to create comparison chart!")
        return
    
    reps = []
    combined_scores = []
    node_f1_scores = []
    edge_auc_scores = []
    node_f1_stds = []
    edge_auc_stds = []
    params_list = []
    
    for rep, result in best_results.items():
        if result and result.get('success'):
            reps.append(rep.replace('_', ' ').title())
            combined_scores.append(result['combined_score'])
            node_f1_scores.append(result['node_f1'])
            edge_auc_scores.append(result['edge_auc'])
            node_f1_stds.append(result.get('node_f1_std', 0))
            edge_auc_stds.append(result.get('edge_auc_std', 0))
            params_list.append(result['params'])
    
    # Create bar chart
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    x_pos = np.arange(len(reps))
    width = 0.25
    # Map representations against their best scores
    bars1 = axes[0].bar(x_pos - width, combined_scores, width, 
                        label='Combined Score', capsize=5, alpha=0.8, color='#1f77b4')
    bars2 = axes[0].bar(x_pos, node_f1_scores, width, 
                        label='Node F1', capsize=5, alpha=0.8, color='#ff7f0e')
    bars3 = axes[0].bar(x_pos + width, edge_auc_scores, width, 
                        label='Edge AUC', capsize=5, alpha=0.8, color='#2ca02c')
    
    axes[0].set_xlabel('Representation', fontsize=12)
    axes[0].set_ylabel('Scores', fontsize=12)
    axes[0].set_title('Best Scores by Circuit Representation', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(reps, rotation=0)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Value labels on bars
    for bars, scores, stds, color in zip(
        [bars2, bars3],  # Only Node F1 and Edge AUC have std
        [node_f1_scores, edge_auc_scores],
        [node_f1_stds, edge_auc_stds],
        ['#ff7f0e', '#2ca02c']
    ):
        for i, (bar, score, std) in enumerate(zip(bars, scores, stds)):
            height = bar.get_height()
            label = f'{score:.3f} ± {std:.3f}'
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.005, 
                        label, ha='center', va='bottom', fontsize=9, color=color)
    
    # Combined score labels (no std)
    for i, (bar, score) in enumerate(zip(bars1, combined_scores)):
        height = bar.get_height()
        label = f'{score:.3f}'
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.005, 
                    label, ha='center', va='bottom', fontsize=9, color='#1f77b4')
    
    # Parameter table - only create if we have data
    if params_list:
        param_data = []
        for rep, params in zip(reps, params_list):
            param_data.append([
                rep,
                params['layers'],
                params['hiddens'],
                params['batch_size'],
                f"{params['lr']:.4f}",
                params['emb_size'],
                params['lambda_node'],
                params['lambda_edge'],
                params['neg_sampling_ratio']
            ])
        
        column_labels = ['Representation', 'Layers', 'Hidden', 'Batch Size', 'Learning Rate', 
                        'Emb Size', 'Lambda Node', 'Lambda Edge', 'Neg Sample Ratio']
        table = axes[1].table(cellText=param_data, colLabels=column_labels, 
                             cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        axes[1].axis('off')
        axes[1].set_title('Best Hyperparameters for Each Representation', fontsize=12, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'No parameter data available', 
                    ha='center', va='center', fontsize=12)
        axes[1].axis('off')
    
    plt.tight_layout()
    
    # Create the full path for saving with additional verification
    save_path = os.path.join(output_dir, 'representation_comparison.png')
    print(f"Attempting to save to: {save_path}")
    
    # Double-check the directory exists right before saving
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        print(f"Directory doesn't exist, creating: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Chart saved successfully to: {save_path}")
    except Exception as e:
        print(f"Error saving figure: {e}")
        # Try alternative: save to current directory as backup
        backup_path = 'representation_comparison.png'
        plt.savefig(backup_path, dpi=300, bbox_inches='tight')
        print(f"✓ Chart saved to backup location: {backup_path}")
    
    plt.show()
    
    print(f"Comparison chart saved to {save_path}")

def create_parameter_analysis(all_results, output_dir='analysis_multitask_hyperparameter_search_component_pin_net'):  # Changed here
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    df_data = []
    for result in all_results:
        if result.get('success') and result.get('combined_score') is not None:
            row = {
                'representation': result['representation'],
                'combined_score': float(result['combined_score']),
                'node_f1': float(result['node_f1']),
                'edge_auc': float(result['edge_auc']),
                'elapsed': float(result.get('elapsed', 0))
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
                'h': int(params['h']),
                'lambda_node': float(params['lambda_node']),
                'lambda_edge': float(params['lambda_edge']),
                'neg_sampling_ratio': float(params['neg_sampling_ratio'])
            })
            df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    if df.empty:
        print("No successful results to analyze")
        return
    
    print(f"Total successful experiments: {len(df)}")
    print(f"Representations found: {df['representation'].unique()}")
    
    reps = df['representation'].unique()
    
    # Heatmap of best parameters (layers vs hidden)
    if len(reps) > 0:
        fig, axes = plt.subplots(1, min(len(reps), 2), figsize=(10, 5) if len(reps) == 1 else (12, 5))
        if len(reps) == 1:
            axes = [axes]
        
        for idx, rep in enumerate(reps[:2]):  # Limit to first 2 representations
            ax = axes[idx]
            rep_data = df[df['representation'] == rep]
            
            # Pivot: layers vs hiddens with combined_score
            if len(rep_data) >= 2:  # Need at least 2 data points for heatmap
                pivot = rep_data.pivot_table(values='combined_score', index='layers', columns='hiddens', aggfunc='mean')
                
                im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
                ax.set_xticks(np.arange(len(pivot.columns)))
                ax.set_yticks(np.arange(len(pivot.index)))
                ax.set_xticklabels(pivot.columns)
                ax.set_yticklabels(pivot.index)
                ax.set_xlabel('Hidden Channels')
                ax.set_ylabel('Layers')
                ax.set_title(f'{rep.replace("_", " ").title()}: Layers vs Hidden')
                
                # Add text annotations
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        if not np.isnan(pivot.iloc[i, j]):
                            text = ax.text(j, i, f'{pivot.iloc[i, j]:.3f}', 
                                          ha="center", va="center", color="black", fontsize=8)
                
                plt.colorbar(im, ax=ax)
            else:
                ax.text(0.5, 0.5, f'Not enough data for {rep}', 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'{rep.replace("_", " ").title()}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmaps_layers_hidden.png'), dpi=300)
        plt.show()

def parameter_effects(all_results, output_dir='analysis_multitask_hyperparameter_search_component_pin_net'):  # Changed here
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    df_data = []
    for result in all_results:
        if result.get('success') and result.get('combined_score') is not None:
            row = {
                'representation': result['representation'],
                'combined_score': result['combined_score'],
                'node_f1': result['node_f1'],
                'edge_auc': result['edge_auc'],
            }
            row.update(result['params'])
            df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    if df.empty:
        print("No successful results for parameter effects analysis")
        return
    
    # Convert numeric columns
    numeric_cols = ['layers', 'hiddens', 'batch_size', 'lr', 'emb_size', 
                   'lambda_node', 'lambda_edge', 'neg_sampling_ratio',
                   'combined_score', 'node_f1', 'edge_auc']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # For each representation: analyze parameter effects
    for rep in df['representation'].unique():
        rep_data = df[df['representation'] == rep]
        
        if len(rep_data) < 3:  # Need at least 3 data points for meaningful analysis
            print(f"Not enough data for {rep} (only {len(rep_data)} points)")
            continue
        
        # Create scatter plots for key parameters
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        parameters = ['layers', 'hiddens', 'batch_size', 'lr', 'emb_size', 
                     'lambda_node', 'lambda_edge', 'neg_sampling_ratio']
        param_names = ['Layers', 'Hidden', 'Batch Size', 'Learning Rate', 'Emb Size',
                      'Lambda Node', 'Lambda Edge', 'Neg Sample Ratio']
        
        for idx, (param, param_name) in enumerate(zip(parameters, param_names)):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Simple scatter plot
            x = rep_data[param]
            y = rep_data['combined_score']
            
            ax.scatter(x, y, alpha=0.7, s=50)
            
            # Add trend line if enough points
            if len(x) > 2:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_sorted = np.sort(x)
                ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.5)
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('Combined Score')
            ax.set_title(f'{rep}: {param_name} vs Combined Score')
            ax.grid(True, alpha=0.3)
        
        # Last subplot: Show all parameter values for this representation
        ax = axes[-1]
        # Get parameter names excluding the ones already plotted
        remaining_params = [p for p in parameters if p not in parameters[:len(axes)-1]]
        
        if remaining_params:
            param_text = f"Parameters for {rep}:\n"
            param_text += f"Experiments: {len(rep_data)}\n\n"
            for param in parameters:
                if param in rep_data.columns:
                    unique_vals = rep_data[param].unique()
                    if len(unique_vals) <= 5:
                        param_text += f"{param}: {sorted(unique_vals)}\n"
            
            ax.text(0.1, 0.5, param_text, fontsize=10, 
                   verticalalignment='center', transform=ax.transAxes)
        
        ax.axis('off')
        
        plt.suptitle(f'Parameter Analysis: {rep.replace("_", " ").title()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(output_dir, f'parameter_effects_{rep}.png'), dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving figure: {e}")
            if rep == 'component_pin':
                backup_path = 'parameter_effects_component_pin.png'
            else:
                backup_path = 'parameter_effects_component_pin_net.png'
            plt.savefig(backup_path, dpi=300, bbox_inches='tight')
            print(f"✓ Chart saved to backup location: {backup_path}")
        
        plt.show()

def main():
    output_dir = os.path.abspath('analysis_multitask_hyperparameter_search_ltspice_demos')
    os.makedirs(output_dir, exist_ok=True)

    results_dirs = [
    'multitask_hyperparameter_search_ltspice_demos_component_pin',
    'multitask_hyperparameter_search_ltspice_demos_component_pin_net'
    ]

    all_results_combined = []
    best_results_combined = {}

    print("Loading results...")
    for d in results_dirs:
        all_r, best_r = load_results(d)
        all_results_combined.extend(all_r)
        best_results_combined.update(best_r)
        
    print(f"\nTotal experiments: {len(all_results_combined)}")
    successful = [r for r in all_results_combined if r.get('success') and r.get('combined_score') is not None]
    print(f"Successful experiments: {len(successful)}")
    
    if len(successful) > 0:
        print(f"Representations found: {list(best_results_combined.keys())}")
        
        print("\nCreating visualizations...")
        create_comparison_chart(best_results_combined, output_dir)
        create_parameter_analysis(all_results_combined, output_dir)
        parameter_effects(all_results_combined, output_dir)
        
        print("\n" + "="*80)
        print("BEST OVERALL CONFIGURATION")
        print("="*80)
        
        # Find the best across all representations
        best_overall = None
        for rep, result in best_results_combined.items():
            if result and result['success']:
                if best_overall is None or result['combined_score'] > best_overall['combined_score']:
                    best_overall = result
                    best_overall['representation'] = rep
        
        if best_overall:
            print(f"\nBest Representation: {best_overall['representation'].replace('_', ' ').title()}")
            print(f"Combined Score: {best_overall['combined_score']:.4f}")
            print(f"Node F1: {best_overall['node_f1']:.4f} ± {best_overall.get('node_f1_std', 0):.4f}")
            print(f"Edge AUC: {best_overall['edge_auc']:.4f} ± {best_overall.get('edge_auc_std', 0):.4f}")
            print("\nParameters:")
            for param, value in best_overall['params'].items():
                print(f"  {param}: {value}")
        else:
            print("No successful experiments found!")
    else:
        print("\nNo successful experiments to analyze!")
        
        # Show what went wrong
        print("\nFailure analysis:")
        failures = [r for r in all_results_combined if not r.get('success') or r.get('combined_score') is None]
        for i, fail in enumerate(failures[:5]):  # Show first 5 failures
            print(f"{i+1}. Representation: {fail.get('representation', 'N/A')}")
            if 'error' in fail:
                print(f"   Error: {fail['error']}")
    
    print(f"\nAnalysis complete! Output directory: '{output_dir}'")

if __name__ == "__main__":
    main()