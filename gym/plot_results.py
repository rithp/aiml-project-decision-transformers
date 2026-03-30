"""
plot_results.py
---------------
Load result JSON files produced by run_noise_sweep.py and plot the
'return vs reward noise' decay curves for DT and BC side-by-side.

Usage:
    python plot_results.py --env halfcheetah --dataset medium

Output:
    figures/noise_robustness_<env>-<dataset>.png
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_results(env, dataset, results_dir='results'):
    """
    Load all JSON result files matching the given env/dataset.
    Returns a nested dict:
        data[model_type][target_return] = [(noise_std, mean, std), ...]
    """
    pattern = os.path.join(results_dir, f'{env}-{dataset}-*.json')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f'No result files found matching: {pattern}\n'
            f'Make sure you have run run_noise_sweep.py first.'
        )

    # data[model_type][target_return] → list of (noise_std, return_mean, return_std)
    data = defaultdict(lambda: defaultdict(list))

    for fpath in files:
        with open(fpath) as f:
            records = json.load(f)
        for rec in records:
            model_type   = rec['model_type']
            target       = rec['target_return']
            noise_std    = rec['noise_std']
            return_mean  = rec['return_mean']
            return_std   = rec['return_std']
            if return_mean is not None:
                data[model_type][target].append((noise_std, return_mean, return_std))

    # sort each list by noise_std
    for model_type in data:
        for target in data[model_type]:
            data[model_type][target].sort(key=lambda x: x[0])

    return data


def plot(env, dataset, data, out_dir='figures'):
    os.makedirs(out_dir, exist_ok=True)

    # collect all unique target returns across models
    all_targets = sorted({
        target
        for model_data in data.values()
        for target in model_data
    }, reverse=True)

    model_styles = {
        'dt': {'color': '#2196F3', 'marker': 'o', 'label': 'Decision Transformer'},
        'bc': {'color': '#FF5722', 'marker': 's', 'label': 'MLP-BC (baseline)'},
    }

    fig, axes = plt.subplots(
        1, len(all_targets),
        figsize=(6 * len(all_targets), 5),
        sharey=False,
    )
    if len(all_targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, all_targets):
        for model_type, model_data in data.items():
            if target not in model_data:
                continue
            points = model_data[target]
            noise_vals   = np.array([p[0] for p in points])
            return_means = np.array([p[1] for p in points])
            return_stds  = np.array([p[2] for p in points])

            style = model_styles.get(model_type, {'color': 'gray', 'marker': 'x', 'label': model_type})
            ax.plot(noise_vals, return_means,
                    color=style['color'], marker=style['marker'],
                    linewidth=2, markersize=7, label=style['label'])
            ax.fill_between(noise_vals,
                            return_means - return_stds,
                            return_means + return_stds,
                            color=style['color'], alpha=0.15)

        ax.set_xlabel('Reward Noise Std (σ)', fontsize=13)
        ax.set_ylabel('Episode Return', fontsize=13)
        ax.set_title(f'Target Return = {target}', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)

    env_display = f'{env}-{dataset}'
    fig.suptitle(
        f'Robustness to Noisy Reward Annotations\n({env_display})',
        fontsize=15, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'noise_robustness_{env_display}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'[plot] Saved → {out_path}')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--out_dir', type=str, default='figures')
    args = parser.parse_args()

    data = load_results(args.env, args.dataset, results_dir=args.results_dir)
    plot(args.env, args.dataset, data, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
