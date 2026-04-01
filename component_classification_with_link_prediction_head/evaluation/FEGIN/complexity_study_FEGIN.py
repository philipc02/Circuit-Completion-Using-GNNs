#!/usr/bin/env python3

import os
import time
import json
import argparse
import subprocess
import numpy as np
from collections import defaultdict

REPRESENTATIONS = [
    'component_component',
    'component_net',
    'component_pin',
    'component_pin_net'
]

LAYER_RANGE = [2, 3, 4, 5, 6, 7, 8]
SEEDS = [0, 1, 2, 3, 4]

DEFAULT_PARAMS = {
    'hiddens': 128,
    'batch_size': 64,
    'lr': 0.001,
    'emb_size': 250,
    'epochs': 100,
    'h': 2,
    'lambda_node': 1.0,
    'lambda_edge': 1.0,
    'neg_sampling_ratio': 2.0,
    'max_pins': 3,
}


def parse_results_from_stdout(stdout_text, model):
    if model == 'FEGIN':
        for line in stdout_text.split('\n'):
            line = line.strip()
            if line.startswith("FEGIN weighted F1:"):
                return {'f1': float(line.split(":")[1].split("±")[0].strip())}
        return {'f1': None}

    elif model == 'MultiTaskFEGIN':
        results = {'f1': None, 'edge_auc': None, 'combined': None}
        for line in stdout_text.split('\n'):
            line = line.strip()
            if line.startswith("Component Classification F1:"):
                results['f1'] = float(line.split(":")[1].strip())
            elif line.startswith("Link Prediction AUC:"):
                results['edge_auc'] = float(line.split(":")[1].strip())
            elif line.startswith("Combined Score:"):
                results['combined'] = float(line.split(":")[1].strip())
        return results

    return {'f1': None}


def run_experiment(representation, layers, seed, output_dir, model, extra_params):
    exp_name = f"{representation}_L{layers}_seed{seed}"
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    cmd = [
        'python3', 'main.py',
        '--data', 'amsnet',
        '--representation', representation,
        '--model', model,
        '--layers', str(layers),
        '--hiddens', str(DEFAULT_PARAMS['hiddens']),
        '--batch_size', str(DEFAULT_PARAMS['batch_size']),
        '--lr', str(DEFAULT_PARAMS['lr']),
        '--emb_size', str(DEFAULT_PARAMS['emb_size']),
        '--epochs', str(DEFAULT_PARAMS['epochs']),
        '--seed', str(seed),
        '--save_appendix', exp_name,
        '--no_val'
    ]

    if model == 'MultiTaskFEGIN':
        cmd += [
            '--lambda_node', str(extra_params.get('lambda_node', DEFAULT_PARAMS['lambda_node'])),
            '--lambda_edge', str(extra_params.get('lambda_edge', DEFAULT_PARAMS['lambda_edge'])),
            '--neg_sampling_ratio', str(extra_params.get('neg_sampling_ratio', DEFAULT_PARAMS['neg_sampling_ratio'])),
        ]

    print(f"Running: {exp_name}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    with open(os.path.join(exp_dir, 'output.txt'), 'w') as f:
        f.write(result.stdout)

    f1 = parse_results_from_stdout(result.stdout, model)

    return f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default='model_complexity_study')
    parser.add_argument('--model', type=str, default='MultiTaskFEGIN',
                        choices=['FEGIN', 'MultiTaskFEGIN'])
    parser.add_argument('--lambda_node', type=float, default=DEFAULT_PARAMS['lambda_node'])
    parser.add_argument('--lambda_edge', type=float, default=DEFAULT_PARAMS['lambda_edge'])
    parser.add_argument('--neg_sampling_ratio', type=float, default=DEFAULT_PARAMS['neg_sampling_ratio'])
    args = parser.parse_args()

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        args.output_dir
    )
    os.makedirs(output_dir, exist_ok=True)

    extra_params = {
        'lambda_node': args.lambda_node,
        'lambda_edge': args.lambda_edge,
        'neg_sampling_ratio': args.neg_sampling_ratio,
    }

    raw_results = []
    aggregated_results = []

    start_time = time.time()

    for representation in REPRESENTATIONS:
        print("\n====================================================")
        print(f"Representation: {representation}  |  Model: {args.model}")
        print("====================================================")

        for layers in LAYER_RANGE:
            seed_scores = {'f1': [], 'edge_auc': [], 'combined': []}

            for seed in SEEDS:
                metrics = run_experiment(representation, layers, seed, output_dir, args.model, extra_params)

                for key in seed_scores:
                    if metrics.get(key) is not None:
                        seed_scores[key].append(metrics[key])

                raw_results.append({
                    'representation': representation,
                    'layers': layers,
                    'seed': seed,
                    **metrics
                })

            agg_entry = {'representation': representation, 'layers': layers}
            summary_parts = []
            for key, scores in seed_scores.items():
                if scores:
                    agg_entry[f'mean_{key}'] = float(np.mean(scores))
                    agg_entry[f'std_{key}'] = float(np.std(scores))
                    summary_parts.append(f"{key}={np.mean(scores):.4f}±{np.std(scores):.4f}")
                else:
                    agg_entry[f'mean_{key}'] = None
                    agg_entry[f'std_{key}'] = None

            aggregated_results.append(agg_entry)
            print(f"  L={layers} -> {' | '.join(summary_parts) if summary_parts else 'no results'}")

    total_time = time.time() - start_time

    # Save raw results
    with open(os.path.join(output_dir, 'raw_results.json'), 'w') as f:
        json.dump(raw_results, f, indent=2)

    # Save aggregated results
    with open(os.path.join(output_dir, 'aggregated_results.json'), 'w') as f:
        json.dump(aggregated_results, f, indent=2)

    print("\n====================================================")
    print("COMPLEXITY STUDY COMPLETED")
    print("====================================================")
    print(f"Total time: {total_time/3600:.2f} hours")


if __name__ == "__main__":
    main()
