#!/usr/bin/env python3
# Hyperparameter search for FEGIN model across all circuit representations

import os
import sys
import time
import json
import argparse
import itertools
import subprocess
import numpy as np
from collections import defaultdict

# Hyperparameter search space
HYPERPARAMETERS = {
    'layers': [2, 3, 4, 5, 6],
    'hiddens': [16, 32, 64, 128],
    'batch_size': [32, 64, 128, 256],
    'lr': [0.1, 0.01, 0.001, 0.0001],
    'emb_size': [128, 250, 512],
    'epochs': [100],
    'h': [2],
}

REPRESENTATIONS = [
    'component_component',
    'component_net', 
    'component_pin',
    'component_pin_net'
]

def parse_results_from_stdout(stdout_text):
    # parse F1 score from stdout
    best_f1 = None
    f1_std = None
    
    lines = stdout_text.split('\n')
    for line in lines:
        '''if 'Best result - f1:' in line:
            # Parse line example 'Best result - f1:0.634 ± 0.015, with 4 layers and 32 hidden units and h = 2'
            parts = line.split('f1:')[1].split('±')
            if len(parts) == 2:
                best_f1 = float(parts[0].strip())
                f1_std = float(parts[1].split(',')[0].strip())  # standard deviation -> extract from line as well
            break
            '''
        line = line.strip()
        if line.startswith("FEGIN weighted F1:"):
            # supports both with and without ±
            parts = line.split(":")[1].strip().split("±")
            best_f1 = float(parts[0].strip())
            if len(parts) > 1:
                f1_std = float(parts[1].strip())
            else:
                f1_std = 0.0
            break
    
    return best_f1, f1_std

def run_experiment(representation, params, output_dir):
    exp_name = f"{representation}_L{params['layers']}_H{params['hiddens']}_BS{params['batch_size']}_LR{params['lr']}_E{params['emb_size']}"
    
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    cmd = [
        'python3', 'main.py',
        '--data', 'ltspice_examples',
        '--representation', representation,
        '--model', 'FEGIN',
        '--layers', str(params['layers']),
        '--hiddens', str(params['hiddens']),
        '--batch_size', str(params['batch_size']),
        '--lr', str(params['lr']),
        '--emb_size', str(params['emb_size']),
        '--epochs', str(params['epochs']),
        '--save_appendix', exp_name,
        '--no_val'
    ]
    
    print("**********************************************************")
    print(f"Running experiment: {exp_name}")
    print(f"Command: {' '.join(cmd)}")
    print("**********************************************************")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Save output
        with open(os.path.join(exp_dir, 'output.txt'), 'w') as f:
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
        
        best_f1, f1_std = parse_results_from_stdout(result.stdout)
        
        elapsed = time.time() - start_time
        
        if best_f1 is not None:
            print(f"Completed in {elapsed:.1f}s - Best F1: {best_f1:.4f} ± {f1_std:.4f}")
            return {
                'representation': representation,
                'params': params,
                'f1': best_f1,
                'f1_std': f1_std,
                'elapsed': elapsed,
                'success': True
            }
        else:
            print(f"Failed to parse results (completed in {elapsed:.1f}s)")
            return {
                'representation': representation,
                'params': params,
                'f1': None,
                'f1_std': None,
                'elapsed': elapsed,
                'success': False
            }
            
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        return {
            'representation': representation,
            'params': params,
            'f1': None,
            'f1_std': None,
            'elapsed': time.time() - start_time,
            'success': False,
            'error': str(e)
        }

def grid_search(representations, hyperparams, output_dir):
    # Detailed search
    all_results = []
    best_by_representation = {}
    
    # Parameter combinations
    param_names = list(hyperparams.keys())
    param_values = [hyperparams[name] for name in param_names]
    
    total_combinations = len(list(itertools.product(*param_values))) * len(representations)
    print(f"Total experiments to run: {total_combinations}")
    
    completed = 0
    
    for representation in representations:
        print("**********************************************************************")
        print(f"Searching hyperparameters for representation: {representation}")
        print("**********************************************************************")
        
        best_for_rep = None
        rep_results = []
        
        for param_combo in itertools.product(*param_values):
            params = dict(zip(param_names, param_combo))
            
            result = run_experiment(representation, params, output_dir)
            rep_results.append(result)
            all_results.append(result)
            
            if result['success'] and result['f1'] is not None:
                if best_for_rep is None or result['f1'] > best_for_rep['f1']:
                    best_for_rep = result
            
            completed += 1
            print(f"Progress: {completed}/{total_combinations} ({completed/total_combinations*100:.1f}%)")
        
        if best_for_rep:
            best_by_representation[representation] = best_for_rep
            print(f"\nBest for {representation}: F1={best_for_rep['f1']:.4f}")
            print(f"  Parameters: {best_for_rep['params']}")
        else:
            print(f"\nNo successful experiments for {representation}")
    
    return all_results, best_by_representation

def random_search(representations, hyperparams, output_dir, n_trials=50):
    # Random search (more efficient)
    all_results = []
    best_by_representation = {}
    
    total_experiments = n_trials * len(representations)
    print(f"Total experiments to run: {total_experiments}")
    
    np.random.seed(42)
    completed = 0
    
    for representation in representations:
        print("**********************************************************************")
        print(f"Random search for representation: {representation}")
        print("**********************************************************************")
        
        best_for_rep = None
        rep_results = []
        
        for trial in range(n_trials):
            params = {}
            for param_name, param_values in hyperparams.items():
                params[param_name] = np.random.choice(param_values)
            
            result = run_experiment(representation, params, output_dir)
            rep_results.append(result)
            all_results.append(result)
            
            if result['success'] and result['f1'] is not None:
                if best_for_rep is None or result['f1'] > best_for_rep['f1']:
                    best_for_rep = result
            
            completed += 1
            print(f"Progress: {completed}/{total_experiments} ({completed/total_experiments*100:.1f}%)")
        
        if best_for_rep:
            best_by_representation[representation] = best_for_rep
            print(f"\nBest for {representation}: F1={best_for_rep['f1']:.4f}")
            print(f"  Parameters: {best_for_rep['params']}")
        else:
            print(f"\nNo successful experiments for {representation}")
    
    return all_results, best_by_representation

def save_results(results, best_results, output_dir):
    # Save all results
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save best results
    with open(os.path.join(output_dir, 'best_results.json'), 'w') as f:
        json.dump(best_results, f, indent=2, default=str)
    
    summary_file = os.path.join(output_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        print("**********************************************************************")
        f.write("HYPERPARAMETER SEARCH SUMMARY\n")
        print("**********************************************************************")
        
        f.write("BEST CONFIGURATIONS BY REPRESENTATION:\n")
        print("----------------------------------------------------------------------")
        
        for rep, result in best_results.items():
            if result and result['success']:
                f.write(f"\n{rep.upper()}:\n")
                f.write(f"  F1 Score: {result['f1']:.4f} ± {result['f1_std']:.4f}\n")
                f.write(f"  Parameters:\n")
                for param_name, param_value in result['params'].items():
                    f.write(f"    {param_name}: {param_value}\n")
                f.write(f"  Training time: {result['elapsed']:.1f}s\n")
        
        print("**********************************************************************")
        f.write("ALL EXPERIMENTS SUMMARY:\n")
        print("----------------------------------------------------------------------")
        
        successful = [r for r in results if r['success'] and r['f1'] is not None]
        failed = [r for r in results if not r['success'] or r['f1'] is None]
        
        f.write(f"\nSuccessful experiments: {len(successful)}/{len(results)}\n")
        f.write(f"Failed experiments: {len(failed)}/{len(results)}\n")
        
        if successful:
            f1_scores = [r['f1'] for r in successful]
            f.write(f"\nF1 Score Statistics (successful runs only):\n")
            f.write(f"  Mean: {np.mean(f1_scores):.4f}\n")
            f.write(f"  Std: {np.std(f1_scores):.4f}\n")
            f.write(f"  Min: {np.min(f1_scores):.4f}\n")
            f.write(f"  Max: {np.max(f1_scores):.4f}\n")
    
    print(f"\nResults saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search for FEGIN model')
    parser.add_argument('--search_method', type=str, default='random', 
                       choices=['grid', 'random'],
                       help='Search method: grid or random')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of random trials per representation (for random search)')
    parser.add_argument('--output_dir', type=str, default='hyperparameter_search_ltspice_demos',
                       help='Output directory for results')
    parser.add_argument('--reps', type=str, nargs='+', default=REPRESENTATIONS,
                       help='Representations to search (default: all)')
    
    args = parser.parse_args()
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save search configuration
    config = {
        'search_method': args.search_method,
        'n_trials': args.n_trials,
        'representations': args.reps,
        'hyperparameters': HYPERPARAMETERS
    }
    
    with open(os.path.join(output_dir, 'search_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Starting hyperparameter search for representations: {args.reps}")
    print(f"Search method: {args.search_method}")
    print(f"Output directory: {output_dir}")
    print(f"Hyperparameters: {HYPERPARAMETERS}")
    
    start_time = time.time()
    
    if args.search_method == 'grid':
        all_results, best_results = grid_search(args.reps, HYPERPARAMETERS, output_dir)
    else:
        all_results, best_results = random_search(args.reps, HYPERPARAMETERS, output_dir, args.n_trials)
    
    total_time = time.time() - start_time
    
    # Save results
    save_results(all_results, best_results, output_dir)
    
    # Print final summary
    print("**********************************************************************")
    print(f"HYPERPARAMETER SEARCH COMPLETED")
    print("**********************************************************************")
    print(f"Total time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
    print(f"Results saved to: {output_dir}")
    
    print(f"\nBEST CONFIGURATIONS:")
    for rep, result in best_results.items():
        if result and result['success']:
            print(f"\n{rep.upper()}:")
            print(f"  F1: {result['f1']:.4f} ± {result['f1_std']:.4f}")
            print(f"  Layers: {result['params']['layers']}")
            print(f"  Hidden: {result['params']['hiddens']}")
            print(f"  Batch Size: {result['params']['batch_size']}")
            print(f"  Learning Rate: {result['params']['lr']}")
            print(f"  Embedding Size: {result['params']['emb_size']}")

if __name__ == "__main__":
    main()