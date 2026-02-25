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
    'h': 2
}


def parse_results_from_stdout(stdout_text):
    best_f1 = None

    for line in stdout_text.split('\n'):
        line = line.strip()
        if line.startswith("FEGIN weighted F1:"):
            best_f1 = float(line.split(":")[1].split("±")[0].strip())
            break

    return best_f1


def run_experiment(representation, layers, seed, output_dir):
    exp_name = f"{representation}_L{layers}_seed{seed}"
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    cmd = [
        'python3', 'main.py',
        '--data', 'amsnet',
        '--representation', representation,
        '--model', 'FEGIN',
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

    print(f"Running: {exp_name}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    with open(os.path.join(exp_dir, 'output.txt'), 'w') as f:
        f.write(result.stdout)

    f1 = parse_results_from_stdout(result.stdout)

    return f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default='model_complexity_study')
    args = parser.parse_args()

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        args.output_dir
    )
    os.makedirs(output_dir, exist_ok=True)

    raw_results = []
    aggregated_results = []

    start_time = time.time()

    for representation in REPRESENTATIONS:
        print("\n====================================================")
        print(f"Representation: {representation}")
        print("====================================================")

        for layers in LAYER_RANGE:
            seed_scores = []

            for seed in SEEDS:
                f1 = run_experiment(representation, layers, seed, output_dir)

                if f1 is not None:
                    seed_scores.append(f1)

                raw_results.append({
                    'representation': representation,
                    'layers': layers,
                    'seed': seed,
                    'f1': f1
                })

            if len(seed_scores) > 0:
                mean_f1 = np.mean(seed_scores)
                std_f1 = np.std(seed_scores)
            else:
                mean_f1 = None
                std_f1 = None

            aggregated_results.append({
                'representation': representation,
                'layers': layers,
                'mean_f1': mean_f1,
                'std_f1': std_f1
            })

            print(f"L={layers} -> Mean F1={mean_f1:.4f} ± {std_f1:.4f}")

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
