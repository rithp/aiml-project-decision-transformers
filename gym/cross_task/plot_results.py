"""
Plot results from the cross-task generalization experiment.

Generates:
  1. Learning curves: fine-tuned vs from-scratch return over gradient steps
  2. Sample efficiency bar chart: steps to reach threshold return
  3. Summary table

Usage:
    python cross_task/plot_results.py --results_file results/transfer_results.json
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Will print text summary only.")


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_curves(logs, target_return=5000):
    """Extract (gradient_steps, mean_return) from a list of iteration logs."""
    steps = []
    returns = []
    for log in logs:
        steps.append(log["total_gradient_steps"])
        key = f"evaluation/target_{target_return}_return_mean"
        if key in log:
            returns.append(log[key])
        else:
            # Try to find any return key
            ret_keys = [k for k in log if "return_mean" in k]
            if ret_keys:
                returns.append(log[ret_keys[0]])
            else:
                returns.append(0.0)
    return np.array(steps), np.array(returns)


def find_steps_to_threshold(steps, returns, threshold):
    """Find the first gradient step count where return >= threshold."""
    above = np.where(returns >= threshold)[0]
    if len(above) == 0:
        return None  # never reached
    return steps[above[0]]


def plot_learning_curves(results, output_dir, target_return=5000):
    """Plot fine-tune vs scratch learning curves for each experiment."""
    if not HAS_MPL:
        return

    for exp_key, exp_data in results.items():
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        ft_steps, ft_returns = extract_curves(
            exp_data["finetune_logs"], target_return
        )
        sc_steps, sc_returns = extract_curves(
            exp_data["scratch_logs"], target_return
        )

        ax.plot(ft_steps, ft_returns, "b-o", label="Fine-tuned", linewidth=2,
                markersize=4)
        ax.plot(sc_steps, sc_returns, "r-s", label="From scratch", linewidth=2,
                markersize=4)

        ax.set_xlabel("Gradient Steps", fontsize=12)
        ax.set_ylabel(f"Mean Return (target={target_return})", fontsize=12)
        ax.set_title(
            f"Cross-Task Transfer: {exp_data['source_dataset']} → "
            f"{exp_data['target_domain']}",
            fontsize=13,
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        out_path = os.path.join(output_dir, f"learning_curve_{exp_key}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_sample_efficiency(results, output_dir, target_return=5000,
                           threshold_fraction=0.5):
    """
    Bar chart comparing gradient steps to reach threshold_fraction * target_return.
    """
    if not HAS_MPL:
        return

    threshold = target_return * threshold_fraction
    exp_keys = []
    ft_steps_list = []
    sc_steps_list = []

    for exp_key, exp_data in results.items():
        ft_steps, ft_returns = extract_curves(
            exp_data["finetune_logs"], target_return
        )
        sc_steps, sc_returns = extract_curves(
            exp_data["scratch_logs"], target_return
        )

        ft_to_thresh = find_steps_to_threshold(ft_steps, ft_returns, threshold)
        sc_to_thresh = find_steps_to_threshold(sc_steps, sc_returns, threshold)

        exp_keys.append(
            f"{exp_data['source_dataset']}\n→ {exp_data['target_domain']}"
        )
        ft_steps_list.append(ft_to_thresh if ft_to_thresh else ft_steps[-1])
        sc_steps_list.append(sc_to_thresh if sc_to_thresh else sc_steps[-1])

    x = np.arange(len(exp_keys))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(exp_keys) * 2), 6))
    bars1 = ax.bar(x - width / 2, ft_steps_list, width, label="Fine-tuned",
                   color="steelblue")
    bars2 = ax.bar(x + width / 2, sc_steps_list, width, label="From scratch",
                   color="indianred")

    ax.set_ylabel("Gradient Steps to Threshold", fontsize=12)
    ax.set_title(
        f"Sample Efficiency: Steps to Reach {threshold:.0f} Return "
        f"({threshold_fraction*100:.0f}% of target)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(exp_keys, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    out_path = os.path.join(output_dir, "sample_efficiency_comparison.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def print_summary_table(results, target_return=5000):
    """Print a text summary table of all experiments."""
    print(f"\n{'='*90}")
    print("SUMMARY: Cross-Task Generalization Results")
    print(f"{'='*90}")
    header = (
        f"{'Experiment':<30} | {'FT Final Return':>16} | "
        f"{'Scratch Final':>14} | {'FT Speedup':>11}"
    )
    print(header)
    print("-" * len(header))

    for exp_key, exp_data in results.items():
        ft_steps, ft_returns = extract_curves(
            exp_data["finetune_logs"], target_return
        )
        sc_steps, sc_returns = extract_curves(
            exp_data["scratch_logs"], target_return
        )

        ft_final = ft_returns[-1] if len(ft_returns) else 0
        sc_final = sc_returns[-1] if len(sc_returns) else 0

        # Compute speedup: steps for scratch to reach FT's iteration-1 return
        if len(ft_returns) > 0:
            ft_first = ft_returns[0]
            sc_to_ft_first = find_steps_to_threshold(
                sc_steps, sc_returns, ft_first
            )
            if sc_to_ft_first and ft_steps[0] > 0:
                speedup = f"{sc_to_ft_first / ft_steps[0]:.1f}x"
            else:
                speedup = "N/A"
        else:
            speedup = "N/A"

        label = f"{exp_data['source_dataset']} → {exp_data['target_domain']}"
        print(f"{label:<30} | {ft_final:>16.1f} | {sc_final:>14.1f} | {speedup:>11}")

    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str,
                        default="results/transfer_results.json")
    parser.add_argument("--output_dir", type=str, default="results/plots")
    parser.add_argument("--target_return", type=float, default=5000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_results(args.results_file)

    print_summary_table(results, args.target_return)

    if HAS_MPL:
        print("\nGenerating plots...")
        plot_learning_curves(results, args.output_dir, args.target_return)
        plot_sample_efficiency(results, args.output_dir, args.target_return)
    else:
        print("\nSkipping plots (install matplotlib: pip install matplotlib)")


if __name__ == "__main__":
    main()
