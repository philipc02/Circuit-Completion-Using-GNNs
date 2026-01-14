#!/usr/bin/env python3
# Hyperparameter search for MultiTaskFEGIN model (pin-level representations only)

import os
import sys
import time
import json
import argparse
import itertools
import subprocess
import numpy as np
from collections import defaultdict

# Hyperparameter search space for MultiTaskFEGIN
HYPERPARAMETERS = {
    'layers': [2, 3, 4, 5, 6],
    'hiddens': [16, 32, 64, 128],
    'batch_size': [16, 32, 64],
    'lr': [0.01, 0.001, 0.0001],
    'emb_size': [128, 250],
    'epochs': [100],
    'h': [2],
    'lambda_node': [0.5, 1.0, 2.0, 3.0],
    'lambda_edge': [0.3, 0.5, 1.0, 1.5],
    'neg_sampling_ratio': [2.0, 5.0, 10.0],
}

REPRESENTATIONS = [
    'component_pin',
    'component_pin_net'
]

def parse_results_from_stdout(stdout_text):
    combined_score = None
    node_f1 = None
    edge_auc = None
    node_f1_std = None
    edge_auc_std = None
    
    lines = stdout_text.split('\n')
    for line in lines:
        # Look for the summary statistics at the end
        if 'Node F1:' in line and '±' in line:
            # Example: "Node F1: 0.6584 ± 0.0186"
            parts = line.split('Node F1:')[1].split('±')
            if len(parts) == 2:
                node_f1 = float(parts[0].strip())
                node_f1_std = float(parts[1].strip())
        
        if 'Edge AUC:' in line and '±' in line:
            # Example: "Edge AUC: 0.9720 ± 0.0139"
            parts = line.split('Edge AUC:')[1].split('±')
            if len(parts) == 2:
                edge_auc = float(parts[0].strip())
                edge_auc_std = float(parts[1].strip())
        
        if 'Combined:' in line and '±' in line:
            # Example: "Combined: 0.7525 ± 0.0136"
            parts = line.split('Combined:')[1].split('±')
            if len(parts) == 2:
                combined_score = float(parts[0].strip())
    
    return combined_score, node_f1, edge_auc, node_f1_std, edge_auc_std

def load_existing_result(exp_dir):
    """Load results from a previously completed experiment"""
    output_file = os.path.join(exp_dir, 'output.txt')
    if not os.path.exists(output_file):
        return None
    
    try:
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Parse the stdout section
        if 'STDOUT:' in content:
            stdout_section = content.split('STDOUT:')[1].split('\n\nSTDERR:')[0]
        else:
            stdout_section = content
        
        combined, node_f1, edge_auc, node_f1_std, edge_auc_std = parse_results_from_stdout(stdout_section)
        
        if combined is not None:
            return {
                'combined_score': combined,
                'node_f1': node_f1,
                'edge_auc': edge_auc,
                'node_f1_std': node_f1_std,
                'edge_auc_std': edge_auc_std,
                'success': True,
                'resumed': True
            }
    except Exception as e:
        print(f"⚠️  Warning: Could not parse existing results from {exp_dir}: {e}")
    
    return None

def run_experiment(representation, params, output_dir, force_rerun=False):
    """Run a single experiment with given parameters"""
    exp_name = (f"{representation}_L{params['layers']}_H{params['hiddens']}_"
                f"BS{params['batch_size']}_LR{params['lr']}_E{params['emb_size']}_"
                f"LN{params['lambda_node']}_LE{params['lambda_edge']}_"
                f"NSR{params['neg_sampling_ratio']}")
    
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Check if experiment already completed (auto-resume support)
    if not force_rerun and os.path.exists(os.path.join(exp_dir, 'output.txt')):
        print("=" * 80)
        print(f"⏩ Skipping {exp_name} (already completed)")
        print("=" * 80)
        
        existing_result = load_existing_result(exp_dir)
        if existing_result:
            print(f"  Loaded results: Combined={existing_result['combined_score']:.4f}, "
                  f"Node F1={existing_result['node_f1']:.4f}, "
                  f"Edge AUC={existing_result['edge_auc']:.4f}")
            
            return {
                'representation': representation,
                'params': params,
                'combined_score': existing_result['combined_score'],
                'node_f1': existing_result['node_f1'],
                'edge_auc': existing_result['edge_auc'],
                'node_f1_std': existing_result['node_f1_std'],
                'edge_auc_std': existing_result['edge_auc_std'],
                'elapsed': 0,  # Not tracked for resumed experiments
                'success': True,
                'resumed': True
            }
        else:
            print(f"⚠️  Could not load results, will re-run experiment")
    
    cmd = [
        'python3', 'main.py',
        '--data', 'ltspice_demos',
        '--representation', representation,
        '--model', 'MultiTaskFEGIN',
        '--layers', str(params['layers']),
        '--hiddens', str(params['hiddens']),
        '--batch_size', str(params['batch_size']),
        '--lr', str(params['lr']),
        '--emb_size', str(params['emb_size']),
        '--epochs', str(params['epochs']),
        '--lambda_node', str(params['lambda_node']),
        '--lambda_edge', str(params['lambda_edge']),
        '--neg_sampling_ratio', str(params['neg_sampling_ratio']),
        '--save_appendix', exp_name,
        '--no_val'
    ]
    
    print("=" * 80)
    print(f"Running experiment: {exp_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=7200  # 2 hour timeout per experiment
        )
        
        # Save output
        with open(os.path.join(exp_dir, 'output.txt'), 'w') as f:
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
        
        combined, node_f1, edge_auc, node_f1_std, edge_auc_std = parse_results_from_stdout(result.stdout)
        
        elapsed = time.time() - start_time
        
        if combined is not None:
            print(f"✓ Completed in {elapsed:.1f}s")
            print(f"  Combined Score: {combined:.4f}")
            print(f"  Node F1: {node_f1:.4f} ± {node_f1_std:.4f}")
            print(f"  Edge AUC: {edge_auc:.4f} ± {edge_auc_std:.4f}")
            return {
                'representation': representation,
                'params': params,
                'combined_score': combined,
                'node_f1': node_f1,
                'edge_auc': edge_auc,
                'node_f1_std': node_f1_std,
                'edge_auc_std': edge_auc_std,
                'elapsed': elapsed,
                'success': True,
                'resumed': False
            }
        else:
            print(f"✗ Failed to parse results (completed in {elapsed:.1f}s)")
            print(f"⚠️  stdout preview: {result.stdout[:500]}")
            return {
                'representation': representation,
                'params': params,
                'combined_score': None,
                'node_f1': None,
                'edge_auc': None,
                'elapsed': elapsed,
                'success': False,
                'resumed': False
            }
            
    except subprocess.TimeoutExpired:
        print(f"✗ Experiment timed out after 2 hours")
        return {
            'representation': representation,
            'params': params,
            'combined_score': None,
            'node_f1': None,
            'edge_auc': None,
            'elapsed': 7200,
            'success': False,
            'error': 'timeout',
            'resumed': False
        }
    except Exception as e:
        print(f"✗ Experiment failed with error: {e}")
        return {
            'representation': representation,
            'params': params,
            'combined_score': None,
            'node_f1': None,
            'edge_auc': None,
            'elapsed': time.time() - start_time,
            'success': False,
            'error': str(e),
            'resumed': False
        }

def grid_search(representations, hyperparams, output_dir, force_rerun=False):
    """Exhaustive grid search over all parameter combinations"""
    all_results = []
    best_by_representation = {}
    
    # Get all parameter combinations
    param_names = list(hyperparams.keys())
    param_values = [hyperparams[name] for name in param_names]
    
    total_combinations = len(list(itertools.product(*param_values))) * len(representations)
    print(f"Total experiments to run: {total_combinations}")
    print(f"Estimated time (assuming 2h per experiment): {total_combinations * 2:.1f} hours")
    
    completed = 0
    skipped = 0
    
    for representation in representations:
        print("\n" + "=" * 80)
        print(f"SEARCHING HYPERPARAMETERS FOR: {representation}")
        print("=" * 80 + "\n")
        
        best_for_rep = None
        rep_results = []
        
        for param_combo in itertools.product(*param_values):
            params = dict(zip(param_names, param_combo))
            
            result = run_experiment(representation, params, output_dir, force_rerun)
            rep_results.append(result)
            all_results.append(result)
            
            if result.get('resumed', False):
                skipped += 1
            
            if result['success'] and result['combined_score'] is not None:
                if best_for_rep is None or result['combined_score'] > best_for_rep['combined_score']:
                    best_for_rep = result
                    print(f"\n🎯 NEW BEST for {representation}: {result['combined_score']:.4f}")
            
            completed += 1
            print(f"\nProgress: {completed}/{total_combinations} ({completed/total_combinations*100:.1f}%) [Resumed: {skipped}]\n")
            
            # Save intermediate results after each experiment
            save_results(all_results, best_by_representation, output_dir)
        
        if best_for_rep:
            best_by_representation[representation] = best_for_rep
            print(f"\n{'='*80}")
            print(f"BEST FOR {representation.upper()}:")
            print(f"  Combined Score: {best_for_rep['combined_score']:.4f}")
            print(f"  Node F1: {best_for_rep['node_f1']:.4f}")
            print(f"  Edge AUC: {best_for_rep['edge_auc']:.4f}")
            print(f"  Parameters: {best_for_rep['params']}")
            print(f"{'='*80}\n")
        else:
            print(f"\n✗ No successful experiments for {representation}\n")
    
    return all_results, best_by_representation

def random_search(representations, hyperparams, output_dir, n_trials=30, force_rerun=False):
    """Random search (more efficient than grid search)"""
    all_results = []
    best_by_representation = {}
    
    total_experiments = n_trials * len(representations)
    print(f"Total experiments to run: {total_experiments}")
    print(f"Estimated time (assuming 2h per experiment): {total_experiments * 2:.1f} hours")
    
    np.random.seed(42)
    completed = 0
    skipped = 0
    
    for representation in representations:
        print("\n" + "=" * 80)
        print(f"RANDOM SEARCH FOR: {representation}")
        print("=" * 80 + "\n")
        
        best_for_rep = None
        rep_results = []
        
        for trial in range(n_trials):
            # Randomly sample parameters
            params = {}
            for param_name, param_values in hyperparams.items():
                params[param_name] = np.random.choice(param_values)
            
            print(f"\nTrial {trial+1}/{n_trials} for {representation}")
            
            result = run_experiment(representation, params, output_dir, force_rerun)
            rep_results.append(result)
            all_results.append(result)
            
            if result.get('resumed', False):
                skipped += 1
            
            if result['success'] and result['combined_score'] is not None:
                if best_for_rep is None or result['combined_score'] > best_for_rep['combined_score']:
                    best_for_rep = result
                    print(f"\n🎯 NEW BEST for {representation}: {result['combined_score']:.4f}")
            
            completed += 1
            print(f"\nProgress: {completed}/{total_experiments} ({completed/total_experiments*100:.1f}%) [Resumed: {skipped}]\n")
            
            # Save intermediate results after each experiment
            save_results(all_results, best_by_representation, output_dir)
        
        if best_for_rep:
            best_by_representation[representation] = best_for_rep
            print(f"\n{'='*80}")
            print(f"BEST FOR {representation.upper()}:")
            print(f"  Combined Score: {best_for_rep['combined_score']:.4f}")
            print(f"  Node F1: {best_for_rep['node_f1']:.4f}")
            print(f"  Edge AUC: {best_for_rep['edge_auc']:.4f}")
            print(f"  Parameters: {best_for_rep['params']}")
            print(f"{'='*80}\n")
        else:
            print(f"\n✗ No successful experiments for {representation}\n")
    
    return all_results, best_by_representation

def save_results(results, best_results, output_dir):
    """Save all results to JSON and text files"""
    # Save all results
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save best results
    with open(os.path.join(output_dir, 'best_results.json'), 'w') as f:
        json.dump(best_results, f, indent=2, default=str)
    
    # Create human-readable summary
    summary_file = os.path.join(output_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTITASK FEGIN HYPERPARAMETER SEARCH SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("BEST CONFIGURATIONS BY REPRESENTATION:\n")
        f.write("-"*80 + "\n\n")
        
        for rep, result in best_results.items():
            if result and result['success']:
                f.write(f"{rep.upper()}:\n")
                f.write(f"  Combined Score: {result['combined_score']:.4f}\n")
                f.write(f"  Node F1: {result['node_f1']:.4f} ± {result['node_f1_std']:.4f}\n")
                f.write(f"  Edge AUC: {result['edge_auc']:.4f} ± {result['edge_auc_std']:.4f}\n")
                f.write(f"  Parameters:\n")
                for param_name, param_value in result['params'].items():
                    f.write(f"    {param_name}: {param_value}\n")
                if not result.get('resumed', False):
                    f.write(f"  Training time: {result['elapsed']:.1f}s ({result['elapsed']/3600:.2f}h)\n")
                f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ALL EXPERIMENTS SUMMARY:\n")
        f.write("-"*80 + "\n\n")
        
        successful = [r for r in results if r['success'] and r['combined_score'] is not None]
        failed = [r for r in results if not r['success'] or r['combined_score'] is None]
        resumed = [r for r in results if r.get('resumed', False)]
        
        f.write(f"Total experiments: {len(results)}\n")
        f.write(f"Successful experiments: {len(successful)}/{len(results)}\n")
        f.write(f"Failed experiments: {len(failed)}/{len(results)}\n")
        f.write(f"Resumed experiments: {len(resumed)}/{len(results)}\n\n")
        
        if successful:
            combined_scores = [r['combined_score'] for r in successful]
            node_f1_scores = [r['node_f1'] for r in successful]
            edge_auc_scores = [r['edge_auc'] for r in successful]
            
            f.write("COMBINED SCORE Statistics:\n")
            f.write(f"  Mean: {np.mean(combined_scores):.4f}\n")
            f.write(f"  Std: {np.std(combined_scores):.4f}\n")
            f.write(f"  Min: {np.min(combined_scores):.4f}\n")
            f.write(f"  Max: {np.max(combined_scores):.4f}\n\n")
            
            f.write("NODE F1 Statistics:\n")
            f.write(f"  Mean: {np.mean(node_f1_scores):.4f}\n")
            f.write(f"  Std: {np.std(node_f1_scores):.4f}\n")
            f.write(f"  Min: {np.min(node_f1_scores):.4f}\n")
            f.write(f"  Max: {np.max(node_f1_scores):.4f}\n\n")
            
            f.write("EDGE AUC Statistics:\n")
            f.write(f"  Mean: {np.mean(edge_auc_scores):.4f}\n")
            f.write(f"  Std: {np.std(edge_auc_scores):.4f}\n")
            f.write(f"  Min: {np.min(edge_auc_scores):.4f}\n")
            f.write(f"  Max: {np.max(edge_auc_scores):.4f}\n")
    
    print(f"\n✓ Results saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter search for MultiTaskFEGIN model'
    )
    parser.add_argument('--search_method', type=str, default='random', 
                       choices=['grid', 'random'],
                       help='Search method: grid or random')
    parser.add_argument('--n_trials', type=int, default=30,
                       help='Number of random trials per representation (for random search)')
    parser.add_argument('--output_dir', type=str, default='multitask_hyperparameter_search',
                       help='Output directory for results')
    parser.add_argument('--reps', type=str, nargs='+', default=REPRESENTATIONS,
                       help='Representations to search (default: component_pin component_pin_net)')
    parser.add_argument('--force_rerun', action='store_true', default=False,
                       help='Force re-run of all experiments (ignore existing results)')
    
    args = parser.parse_args()
    
    # Validate representations
    for rep in args.reps:
        if rep not in REPRESENTATIONS:
            print(f"✗ Error: {rep} is not a valid pin-level representation")
            print(f"  Valid representations: {REPRESENTATIONS}")
            sys.exit(1)
    
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        args.output_dir
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save search configuration
    config = {
        'search_method': args.search_method,
        'n_trials': args.n_trials if args.search_method == 'random' else None,
        'representations': args.reps,
        'hyperparameters': HYPERPARAMETERS,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'force_rerun': args.force_rerun
    }
    
    with open(os.path.join(output_dir, 'search_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("="*80)
    print("MULTITASK FEGIN HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Representations: {args.reps}")
    print(f"Search method: {args.search_method}")
    if args.search_method == 'random':
        print(f"Trials per representation: {args.n_trials}")
    print(f"Output directory: {output_dir}")
    print(f"Force rerun: {args.force_rerun}")
    print(f"Hyperparameters:")
    for key, values in HYPERPARAMETERS.items():
        print(f"  {key}: {values}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    if args.search_method == 'grid':
        all_results, best_results = grid_search(args.reps, HYPERPARAMETERS, output_dir, args.force_rerun)
    else:
        all_results, best_results = random_search(args.reps, HYPERPARAMETERS, output_dir, args.n_trials, args.force_rerun)
    
    total_time = time.time() - start_time
    
    # Save final results
    save_results(all_results, best_results, output_dir)
    
    # Print final summary
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH COMPLETED")
    print("="*80)
    print(f"Total time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
    print(f"Results saved to: {output_dir}")
    
    resumed_count = sum(1 for r in all_results if r.get('resumed', False))
    if resumed_count > 0:
        print(f"Resumed {resumed_count} previously completed experiments")
    
    print(f"\nBEST CONFIGURATIONS:")
    print("-"*80)
    for rep, result in best_results.items():
        if result and result['success']:
            print(f"\n{rep.upper()}:")
            print(f"  Combined Score: {result['combined_score']:.4f}")
            print(f"  Node F1: {result['node_f1']:.4f} ± {result['node_f1_std']:.4f}")
            print(f"  Edge AUC: {result['edge_auc']:.4f} ± {result['edge_auc_std']:.4f}")
            print(f"  Parameters:")
            print(f"    Layers: {result['params']['layers']}")
            print(f"    Hidden: {result['params']['hiddens']}")
            print(f"    Batch Size: {result['params']['batch_size']}")
            print(f"    Learning Rate: {result['params']['lr']}")
            print(f"    Embedding Size: {result['params']['emb_size']}")
            print(f"    Lambda Node: {result['params']['lambda_node']}")
            print(f"    Lambda Edge: {result['params']['lambda_edge']}")
            print(f"    Neg Sampling Ratio: {result['params']['neg_sampling_ratio']}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()