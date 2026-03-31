import gymnasium as gym
import numpy as np
import torch
import wandb

import argparse
import copy
import json
import os
import pickle
import random
import sys

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def add_reward_noise(trajectories, noise_std):
    """
    Deep-copy trajectories and add Gaussian noise N(0, noise_std^2) to every
    reward in the dataset.  noise_std=0.0 is a no-op (clean run).
    """
    if noise_std == 0.0:
        return trajectories  # nothing to do
    noisy = copy.deepcopy(trajectories)
    for traj in noisy:
        noise = np.random.normal(0.0, noise_std, traj['rewards'].shape)
        traj['rewards'] = traj['rewards'] + noise
    print(f'[noise] Added Gaussian reward noise with std={noise_std:.3f}')
    return noisy


def save_checkpoint(model, optimizer, scheduler, iter_num, save_dir, variant):
    """Save a training checkpoint to save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'checkpoint_iter{iter_num}.pt')
    torch.save({
        'iter': iter_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'variant': variant,
    }, path)
    print(f'[checkpoint] Saved → {path}')


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)
    reward_noise_std = variant.get('reward_noise_std', 0.0)
    checkpoint_dir = variant.get('checkpoint_dir', 'checkpoints')
    gamma = variant.get('gamma', 1.0)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-gamma{gamma}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'hopper':
        env = gym.make('Hopper-v5')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v5')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v5')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    else:
        raise NotImplementedError

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # inject reward noise (deep-copied; original file untouched)
    trajectories = add_reward_noise(trajectories, reward_noise_std)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    disc_returns = []  # discounted returns under the configured gamma
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
        disc_returns.append(discount_cumsum(path['rewards'], gamma=gamma)[0])  # full-traj discounted return
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    disc_returns = np.array(disc_returns)

    # --- Option A: derive evaluation targets from dataset discounted return percentiles ---
    # 90th percentile → high target, 50th percentile → medium target
    # This ensures the target is always in-distribution for whatever gamma is used.
    if variant.get('auto_targets', True) and model_type == 'dt':
        high_target = float(np.percentile(disc_returns, 90))
        med_target  = float(np.percentile(disc_returns, 50))
        # Adaptive rounding: use granularity proportional to the magnitude of the returns
        # so that p90 and p50 remain distinct even for small discounted returns (low gamma).
        magnitude = max(abs(high_target), 1.0)
        if magnitude >= 500:
            gran = 100
        elif magnitude >= 50:
            gran = 10
        elif magnitude >= 5:
            gran = 1
        else:
            gran = 0.1
        high_target = round(high_target / gran) * gran
        med_target  = round(med_target  / gran) * gran
        # Ensure they stay distinct; if still equal, nudge med_target down by one granule
        if high_target == med_target:
            med_target = high_target - gran
        env_targets = [high_target, med_target]
        print(f'[gamma={gamma}] Auto-derived targets from discounted returns: {env_targets}')
        print(f'  (disc_return p90={np.percentile(disc_returns, 90):.2f}, p50={np.percentile(disc_returns, 50):.2f}, gran={gran})')

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'Gamma: {gamma}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return (undiscounted): {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Average return (discounted):   {np.mean(disc_returns):.2f}, std: {np.std(disc_returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=gamma)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(dtype=torch.float32, device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    # directory for checkpoints specific to this run
    run_checkpoint_dir = os.path.join(
        checkpoint_dir,
        f'{env_name}-{dataset}-{model_type}-noise{reward_noise_std}',
        exp_prefix,
    )

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    # collect final results across all eval targets for result JSON
    all_results = []

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)

        # save checkpoint after every iteration
        save_checkpoint(model, optimizer, scheduler, iter + 1, run_checkpoint_dir, variant)

        # collect eval metrics on the final iteration
        if iter == variant['max_iters'] - 1:
            for target_rew in env_targets:
                key_mean = f'evaluation/target_{target_rew}_return_mean'
                key_std  = f'evaluation/target_{target_rew}_return_std'
                all_results.append({
                    'gamma': gamma,
                    'noise_std': reward_noise_std,
                    'model_type': model_type,
                    'env': env_name,
                    'dataset': dataset,
                    'target_return': target_rew,
                    'return_mean': outputs.get(key_mean, None),
                    'return_std':  outputs.get(key_std,  None),
                })

    # save results JSON
    os.makedirs('results', exist_ok=True)
    # gamma experiment results use a dedicated filename
    if gamma != 1.0 or variant.get('gamma_experiment', False):
        result_path = os.path.join('results', f'{env_name}-{dataset}-{model_type}-gamma{gamma}.json')
    else:
        result_path = os.path.join('results', f'{env_name}-{dataset}-{model_type}-noise{reward_noise_std}.json')
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'[results] Saved → {result_path}')

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
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
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    # --- new args ---
    parser.add_argument('--reward_noise_std', type=float, default=0.0,
                        help='Std of Gaussian noise added to rewards (0.0 = clean)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Root directory for saving checkpoints')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Discount factor for RTG computation (1.0 = original DT)')
    parser.add_argument('--auto_targets', type=lambda x: x.lower() != 'false', default=True,
                        help='Derive evaluation targets from dataset discounted-return percentiles')
    parser.add_argument('--gamma_experiment', action='store_true',
                        help='Flag to use gamma-specific result filename even at gamma=1.0')

    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
