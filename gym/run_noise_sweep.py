"""
run_noise_sweep.py
------------------
Sweep over a list of reward-noise levels and call experiment() for each.
Results are saved per run to results/<env>-<dataset>-<model_type>-noise<σ>.json
and checkpoints to checkpoints/<env>-<dataset>-<model_type>-noise<σ>/<run_id>/.

Example usage (DT sweep):
    python run_noise_sweep.py \\
        --env halfcheetah --dataset medium \\
        --noise_levels 0.0 0.5 1.0 2.0 5.0 \\
        --model_type dt \\
        --max_iters 10 --num_steps_per_iter 10000 \\
        --num_eval_episodes 20

Then run for baseline:
    python run_noise_sweep.py ... --model_type bc

Then plot:
    python plot_results.py --env halfcheetah --dataset medium
"""

import argparse
import os
import sys

from experiment import experiment


def parse_args():
    parser = argparse.ArgumentParser(description='Noise sweep for Decision Transformer')

    # sweep-specific
    parser.add_argument('--noise_levels', type=float, nargs='+',
                        default=[0.0, 0.5, 1.0, 2.0, 5.0],
                        help='List of reward noise std values to sweep over')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    # passed through to experiment() unchanged
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', type=bool, default=False)

    return parser.parse_args()


def result_exists(env, dataset, model_type, noise_std, results_dir='results'):
    """Return True if a results JSON was already written for this noise level."""
    path = os.path.join(
        results_dir,
        f'{env}-{dataset}-{model_type}-noise{noise_std}.json'
    )
    return os.path.exists(path)


def main():
    args = parse_args()
    base_variant = vars(args)
    noise_levels = base_variant.pop('noise_levels')  # remove sweep-only arg

    print('=' * 60)
    print(f'Noise sweep: {noise_levels}')
    print(f'Model: {base_variant["model_type"]}  |  Env: {base_variant["env"]}-{base_variant["dataset"]}')
    print('=' * 60)

    for noise_std in noise_levels:
        # Resume support: skip if this noise level already has a results file
        if result_exists(base_variant['env'], base_variant['dataset'],
                         base_variant['model_type'], noise_std):
            print(f'\n[skip] noise_std={noise_std} — results already exist, skipping.')
            continue

        print(f'\n{"=" * 60}')
        print(f'>>> Starting run: reward_noise_std = {noise_std}')
        print(f'{"=" * 60}')

        variant = dict(base_variant)
        variant['reward_noise_std'] = noise_std

        experiment('gym-experiment', variant=variant)

    print('\n[sweep] All noise levels complete.')


if __name__ == '__main__':
    main()
