"""
plot_gamma_results.py
---------------------
Load result JSONs from the gamma sweep and plot:
  1. Mean episode return (undiscounted) vs gamma  — main result
  2. Mean episode length vs gamma                 — does discounting shorten episodes?

Usage:
    python plot_gamma_results.py --env hopper --dataset medium

Output:
    figures/gamma_impact_<env>-<dataset>.png
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
    Load all gamma-experiment JSON files.
    Returns:
        data[target_label] = [(gamma, return_mean, return_std, length_mean, length_std), ...]
    """
    pattern = os.path.join(results_dir, f'{env}-{dataset}-dt-gamma*.json')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f'No gamma result files found matching: {pattern}\n'
            f'Run run_gamma_sweep.py first.'
        )

    # group by target_return label (high / medium)
    # We'll rank targets: highest target → "High Target", second → "Medium Target"
    raw = []  # list of dicts

    for fpath in files:
        with open(fpath) as f:
            records = json.load(f)
        for rec in records:
            if rec.get('return_mean') is not None:
                raw.append(rec)

    if not raw:
        raise ValueError('All result files are empty.')

    # determine the two target labels per gamma
    # For each gamma, sort targets descending → ["High Target (pXX)", "Med Target (pYY)"]
    gamma_targets = defaultdict(list)
    for rec in raw:
        gamma_targets[rec['gamma']].append(rec['target_return'])

    # build clean data structure
    # data[target_rank] = [(gamma, mean, std), ...]  where rank 0=high, 1=medium
    data = defaultdict(list)
    for rec in raw:
        gamma = rec['gamma']
        targets_for_gamma = sorted(gamma_targets[gamma], reverse=True)
        rank = targets_for_gamma.index(rec['target_return'])
        label = 'High Target (p90)' if rank == 0 else 'Medium Target (p50)'
        data[label].append((
            gamma,
            rec['return_mean'],
            rec['return_std'],
        ))

    # sort each list by gamma descending (so x-axis reads 1.0 → 0.90)
    for label in data:
        data[label].sort(key=lambda x: x[0], reverse=True)

    return data


def plot(env, dataset, data, out_dir='figures'):
    os.makedirs(out_dir, exist_ok=True)

    colors = {
        'High Target (p90)':    '#2196F3',
        'Medium Target (p50)':  '#FF9800',
    }
    markers = {
        'High Target (p90)':    'o',
        'Medium Target (p50)':  's',
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, points in data.items():
        gammas       = np.array([p[0] for p in points])
        return_means = np.array([p[1] for p in points])
        return_stds  = np.array([p[2] for p in points])

        color  = colors.get(label, 'gray')
        marker = markers.get(label, 'x')

        ax.plot(gammas, return_means,
                color=color, marker=marker,
                linewidth=2, markersize=8, label=label)
        ax.fill_between(gammas,
                        return_means - return_stds,
                        return_means + return_stds,
                        color=color, alpha=0.15)

        # annotate gamma=1.0 (baseline) with a dashed vertical line
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.6,
               label='γ=1.0 (original DT)')

    ax.set_xlabel('Discount Factor (γ)', fontsize=13)
    ax.set_ylabel('Episode Return (true, undiscounted)', fontsize=13)
    ax.invert_xaxis()   # x-axis: 1.0 on left → 0.90 on right (increasing discounting)
    ax.set_title(
        f'Impact of Discount Factor on Decision Transformer\n({env}-{dataset})',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'gamma_impact_{env}-{dataset}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'[plot] Saved → {out_path}')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--out_dir', type=str, default='figures')
    args = parser.parse_args()

    data = load_results(args.env, args.dataset, results_dir=args.results_dir)
    plot(args.env, args.dataset, data, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
