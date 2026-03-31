"""
plot_gamma_results.py
---------------------
Load result JSONs from the gamma sweep and plot:
  1. Mean episode return (undiscounted) vs gamma  — main result

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
    pattern = os.path.join(results_dir, f'{env}-{dataset}-dt-gamma*.json')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f'No gamma result files found matching: {pattern}\n'
            f'Run run_gamma_sweep.py first.'
        )

    raw = []
    for fpath in files:
        with open(fpath) as f:
            records = json.load(f)
        for rec in records:
            if rec.get('return_mean') is not None:
                raw.append(rec)

    if not raw:
        raise ValueError('All result files are empty.')

    # For each gamma, collect unique targets and sort descending
    gamma_targets = defaultdict(list)
    for rec in raw:
        gamma_targets[rec['gamma']].append(rec['target_return'])

    data = defaultdict(list)
    for rec in raw:
        gamma = rec['gamma']
        targets_for_gamma = sorted(gamma_targets[gamma], reverse=True)
        rank = targets_for_gamma.index(rec['target_return'])
        label = 'High Target (p90)' if rank == 0 else 'Medium Target (p50)'
        data[label].append((gamma, rec['return_mean'], rec['return_std']))

    # For gamma levels where p90==p50 (both collapsed to same target),
    # Medium Target entry is missing. Fill it with the same value as High Target
    # so the dotted orange line spans the full x-axis — making collapse visible.
    all_gammas = sorted(gamma_targets.keys(), reverse=True)
    high_by_gamma = {p[0]: p for p in data['High Target (p90)']}
    med_by_gamma  = {p[0]: p for p in data.get('Medium Target (p50)', [])}

    for g in all_gammas:
        if g not in med_by_gamma and g in high_by_gamma:
            data['Medium Target (p50)'].append(high_by_gamma[g])  # collapsed → same point

    for label in data:
        data[label].sort(key=lambda x: x[0], reverse=True)

    return data


def plot(env, dataset, data, out_dir='figures'):
    os.makedirs(out_dir, exist_ok=True)

    styles = {
        'High Target (p90)': dict(
            color='#2196F3', marker='o', linestyle='-',
            linewidth=2.2, markersize=8,
            label='High Target (p90)',
        ),
        'Medium Target (p50)': dict(
            color='#FF9800', marker='s', linestyle='--',
            linewidth=1.8, markersize=7,
            label='Medium Target (p50)  [dashed where targets collapsed]',
        ),
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, points in data.items():
        gammas       = np.array([p[0] for p in points])
        return_means = np.array([p[1] for p in points])
        return_stds  = np.array([p[2] for p in points])
        s = styles.get(label, dict(color='gray', marker='x', linestyle='-',
                                   linewidth=2, markersize=8, label=label))
        ax.plot(gammas, return_means,
                color=s['color'], marker=s['marker'], linestyle=s['linestyle'],
                linewidth=s['linewidth'], markersize=s['markersize'],
                label=s['label'])
        ax.fill_between(gammas,
                        return_means - return_stds,
                        return_means + return_stds,
                        color=s['color'], alpha=0.12)

    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.7,
               label='γ=1.0 (original DT baseline)')

    ax.set_xlabel('Discount Factor (γ)', fontsize=13)
    ax.set_ylabel('Episode Return (true, undiscounted)', fontsize=13)
    ax.invert_xaxis()  # 1.0 on left → 0.90 on right = increasing discounting
    ax.set_title(
        f'Impact of Discount Factor on Decision Transformer\n({env}-{dataset})',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=10)
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
