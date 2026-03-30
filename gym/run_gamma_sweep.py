"""
run_gamma_sweep.py
------------------
Sweep over discount factors γ and call experiment() for each.
Results saved to results/<env>-<dataset>-dt-gamma<γ>.json

Example:
    python run_gamma_sweep.py \\
        --env hopper --dataset medium \\
        --gamma_levels 1.0 0.99 0.97 0.95 0.90 \\
        --max_iters 10 --num_steps_per_iter 10000 \\
        --num_eval_episodes 100

Then plot:
    python plot_gamma_results.py --env hopper --dataset medium
"""

import argparse
import os

from experiment import experiment


def parse_args():
    parser = argparse.ArgumentParser(description='Gamma sweep for Decision Transformer')

    # sweep-specific
    parser.add_argument('--gamma_levels', type=float, nargs='+',
                        default=[1.0, 0.99, 0.97, 0.95, 0.90],
                        help='List of discount factors to sweep over')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    # passed through to experiment() unchanged
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
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


def result_exists(env, dataset, gamma, results_dir='results'):
    """Return True if a result JSON already exists for this gamma."""
    path = os.path.join(results_dir, f'{env}-{dataset}-dt-gamma{gamma}.json')
    return os.path.exists(path)


def main():
    args = parse_args()
    base_variant = vars(args)
    gamma_levels = base_variant.pop('gamma_levels')

    print('=' * 60)
    print(f'Gamma sweep: {gamma_levels}')
    print(f'Env: {base_variant["env"]}-{base_variant["dataset"]}  |  Model: DT only')
    print('=' * 60)

    for gamma in gamma_levels:
        if result_exists(base_variant['env'], base_variant['dataset'], gamma):
            print(f'\n[skip] gamma={gamma} — results already exist, skipping.')
            continue

        print(f'\n{"=" * 60}')
        print(f'>>> Starting run: gamma = {gamma}')
        print(f'{"=" * 60}')

        variant = dict(base_variant)
        variant['model_type'] = 'dt'        # DT only — BC doesn't use RTGs
        variant['gamma'] = gamma
        variant['auto_targets'] = True       # use percentile-based targets
        variant['gamma_experiment'] = True   # use gamma filename even at gamma=1.0
        variant['reward_noise_std'] = 0.0   # no noise in this experiment

        experiment('gym-experiment', variant=variant)

    print('\n[sweep] All gamma levels complete.')


if __name__ == '__main__':
    main()
