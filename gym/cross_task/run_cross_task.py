"""
Cross-Task Generalization and Fine-Tuning Experiment for Decision Transformer.

This script runs the full experiment from the proposal:
  1. Pre-train DT on source domain (Walker2d with standard dynamics)
  2. Fine-tune pre-trained DT on small target domain dataset
  3. Train DT from scratch on the same target domain dataset
  4. Compare sample efficiency (gradient steps to reach threshold return)

Usage:
    # Step 1: Pre-train on all three source datasets
    python cross_task/run_cross_task.py --phase pretrain

    # Step 2: Run fine-tune vs from-scratch comparison
    python cross_task/run_cross_task.py --phase transfer --target heavy

    # Or run everything end-to-end
    python cross_task/run_cross_task.py --phase all
"""

import argparse
import copy
import json
import os
import pickle
import random
import sys
import time

import gymnasium as gym
import numpy as np
import torch

# Add parent dir to path so we can import decision_transformer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode_rtg,
)
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.training.seq_trainer import SequenceTrainer
from cross_task.modified_envs import make_modified_walker2d, get_all_target_names


# =============================================================================
# Default hyperparameters (matching the original DT paper for Walker2d)
# =============================================================================

DEFAULT_CONFIG = {
    "env": "walker2d",
    "K": 20,               # context length
    "batch_size": 64,
    "embed_dim": 128,
    "n_layer": 3,
    "n_head": 1,
    "activation_function": "relu",
    "dropout": 0.1,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "warmup_steps": 10000,
    "max_ep_len": 1000,
    "scale": 1000.0,
    "env_targets": [5000, 2500],
    "num_eval_episodes": 5,   # reduced aggressively for speed
    "device": "mps",
}

# Pre-training config
PRETRAIN_CONFIG = {
    **DEFAULT_CONFIG,
    "max_iters": 5,              # 5 pre-training iterations
    "num_steps_per_iter": 2500,  # 2.5K steps each
}

# Fine-tuning config
FINETUNE_CONFIG = {
    **DEFAULT_CONFIG,
    "learning_rate": 1e-5,        # lower LR for fine-tuning
    "warmup_steps": 1000,         # shorter warmup
    "max_iters": 10,              # 10 iterations
    "num_steps_per_iter": 1000,   # 1K steps each
}

# From-scratch config (same budget as fine-tuning for fair comparison)
SCRATCH_CONFIG = {
    **DEFAULT_CONFIG,
    "max_iters": 10,
    "num_steps_per_iter": 1000,
}


def discount_cumsum(x, gamma):
    out = np.zeros_like(x)
    out[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        out[t] = x[t] + gamma * out[t + 1]
    return out


def load_dataset(dataset_path):
    """Load a trajectory dataset from pickle."""
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)
    return trajectories


def compute_dataset_stats(trajectories):
    """Compute state mean/std and return statistics."""
    states = np.concatenate([t["observations"] for t in trajectories], axis=0)
    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0) + 1e-6
    returns = np.array([t["rewards"].sum() for t in trajectories])
    traj_lens = np.array([len(t["observations"]) for t in trajectories])
    return state_mean, state_std, returns, traj_lens


def make_get_batch(trajectories, state_mean, state_std, state_dim, act_dim,
                   scale, max_ep_len, K, device):
    """Create the get_batch function for the trainer."""
    traj_lens = np.array([len(t["observations"]) for t in trajectories])
    returns = np.array([t["rewards"].sum() for t in trajectories])
    num_timesteps = sum(traj_lens)

    sorted_inds = np.argsort(returns)
    num_trajectories = len(trajectories)
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=64, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories), size=batch_size,
            replace=True, p=p_sample,
        )
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj["rewards"].shape[0] - 1)

            s.append(traj["observations"][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj["actions"][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj["rewards"][si:si + max_len].reshape(1, -1, 1))
            if "terminals" in traj:
                d.append(traj["terminals"][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj["dones"][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1
            rtg.append(
                discount_cumsum(traj["rewards"][si:], gamma=1.0)[
                    : s[-1].shape[1] + 1
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
            )
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate(
                [np.ones((1, max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
            )
            r[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1
            )
            d[-1] = np.concatenate(
                [np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1
            )
            rtg[-1] = (
                np.concatenate(
                    [np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1
                )
                / scale
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device
        )
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device
        )
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device
        )
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device
        )
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(dtype=torch.float32, device=device)

        return s, a, r, d, rtg, timesteps, mask

    return get_batch


def create_model(config):
    """Instantiate a fresh DecisionTransformer."""
    return DecisionTransformer(
        state_dim=config["state_dim"],
        act_dim=config["act_dim"],
        max_length=config["K"],
        max_ep_len=config["max_ep_len"],
        hidden_size=config["embed_dim"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_inner=4 * config["embed_dim"],
        activation_function=config["activation_function"],
        n_positions=1024,
        resid_pdrop=config["dropout"],
        attn_pdrop=config["dropout"],
    )


def make_eval_fn(env, state_dim, act_dim, max_ep_len, scale, target_rew,
                 state_mean, state_std, device, num_eval_episodes):
    """Create an evaluation function for a given target return."""
    def fn(model):
        returns, lengths = [], []
        for _ in range(num_eval_episodes):
            with torch.no_grad():
                ret, length = evaluate_episode_rtg(
                    env, state_dim, act_dim, model,
                    max_ep_len=max_ep_len, scale=scale,
                    target_return=target_rew / scale,
                    mode="normal",
                    state_mean=state_mean, state_std=state_std,
                    device=device,
                )
            returns.append(ret)
            lengths.append(length)
        return {
            f"target_{target_rew}_return_mean": np.mean(returns),
            f"target_{target_rew}_return_std": np.std(returns),
            f"target_{target_rew}_length_mean": np.mean(lengths),
            f"target_{target_rew}_length_std": np.std(lengths),
        }
    return fn


def train_model(model, config, trajectories, eval_env, state_mean, state_std,
                save_path=None, log_prefix=""):
    """
    Train a DT model and return per-iteration logs.
    
    Returns:
        all_logs: list of dicts, one per iteration
    """
    device = config["device"]
    state_dim = config["state_dim"]
    act_dim = config["act_dim"]

    model = model.to(device=device)

    get_batch = make_get_batch(
        trajectories, state_mean, state_std,
        state_dim, act_dim,
        config["scale"], config["max_ep_len"], config["K"], device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / config["warmup_steps"], 1),
    )

    eval_fns = [
        make_eval_fn(
            eval_env, state_dim, act_dim,
            config["max_ep_len"], config["scale"], tar,
            state_mean, state_std, device,
            config["num_eval_episodes"],
        )
        for tar in config["env_targets"]
    ]

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=config["batch_size"],
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        eval_fns=eval_fns,
    )

    all_logs = []
    total_steps = 0
    for iter_num in range(config["max_iters"]):
        logs = trainer.train_iteration(
            num_steps=config["num_steps_per_iter"],
            iter_num=iter_num + 1,
            print_logs=True,
        )
        total_steps += config["num_steps_per_iter"]
        logs["total_gradient_steps"] = total_steps
        logs["iteration"] = iter_num + 1
        all_logs.append(logs)

        # Print summary
        ret_key = f"evaluation/target_{config['env_targets'][0]}_return_mean"
        if ret_key in logs:
            print(f"{log_prefix} Iter {iter_num+1}: "
                  f"loss={logs['training/train_loss_mean']:.4f}, "
                  f"return={logs[ret_key]:.1f}, "
                  f"steps={total_steps}")

    # Save checkpoint
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "state_mean": state_mean,
            "state_std": state_std,
            "logs": all_logs,
        }, save_path)
        print(f"  Checkpoint saved to: {save_path}")

    return all_logs


# =============================================================================
# Phase 1: Pre-training
# =============================================================================

def run_pretrain(args):
    """Pre-train DT on source domain Walker2d datasets."""
    datasets = args.source_datasets.split(",")
    device = args.device

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"PRE-TRAINING on walker2d-{dataset_name}-v2")
        print(f"{'='*70}")

        dataset_path = f"data/walker2d-{dataset_name}-v2.pkl"
        if not os.path.exists(dataset_path):
            print(f"ERROR: Dataset not found at {dataset_path}")
            print("Run: python data/download_d4rl_datasets.py")
            continue

        trajectories = load_dataset(dataset_path)
        state_mean, state_std, returns, traj_lens = compute_dataset_stats(trajectories)

        print(f"  {len(trajectories)} trajectories, {sum(traj_lens)} timesteps")
        print(f"  Return: mean={returns.mean():.1f}, max={returns.max():.1f}")

        # Source eval env (standard Walker2d)
        eval_env = make_modified_walker2d("source")
        state_dim = eval_env.observation_space.shape[0]
        act_dim = eval_env.action_space.shape[0]

        config = {
            **PRETRAIN_CONFIG,
            "state_dim": state_dim,
            "act_dim": act_dim,
            "device": device,
        }

        model = create_model(config)
        save_path = os.path.join(
            args.checkpoint_dir,
            f"pretrained_walker2d_{dataset_name}.pt",
        )

        train_model(
            model, config, trajectories, eval_env,
            state_mean, state_std,
            save_path=save_path,
            log_prefix=f"[pretrain/{dataset_name}]",
        )
        eval_env.close()


# =============================================================================
# Phase 2: Transfer (fine-tune vs from-scratch)
# =============================================================================

def run_transfer(args):
    """Run fine-tune vs from-scratch comparison on target domains."""
    source_datasets = args.source_datasets.split(",")
    target_domains = (
        get_all_target_names() if args.target == "all"
        else args.target.split(",")
    )
    device = args.device

    results = {}

    for source_ds in source_datasets:
        for target_name in target_domains:
            exp_key = f"{source_ds}_to_{target_name}"
            print(f"\n{'='*70}")
            print(f"TRANSFER: {source_ds} -> {target_name}")
            print(f"{'='*70}")

            # Load pre-trained checkpoint
            ckpt_path = os.path.join(
                args.checkpoint_dir,
                f"pretrained_walker2d_{source_ds}.pt",
            )
            if not os.path.exists(ckpt_path):
                print(f"ERROR: Checkpoint not found: {ckpt_path}")
                print("Run --phase pretrain first.")
                continue

            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            pretrained_config = checkpoint["config"]
            # Use source domain stats for normalization (important!)
            source_state_mean = checkpoint["state_mean"]
            source_state_std = checkpoint["state_std"]

            # Load target domain dataset
            target_data_path = os.path.join(
                "data/target", f"walker2d-{target_name}.pkl"
            )
            if not os.path.exists(target_data_path):
                print(f"ERROR: Target dataset not found: {target_data_path}")
                print("Run: python cross_task/collect_target_data.py --target all")
                continue

            target_trajectories = load_dataset(target_data_path)
            target_state_mean, target_state_std, target_returns, target_lens = (
                compute_dataset_stats(target_trajectories)
            )
            print(f"  Target data: {len(target_trajectories)} trajectories, "
                  f"{sum(target_lens)} timesteps")
            print(f"  Target returns: mean={target_returns.mean():.1f}, "
                  f"max={target_returns.max():.1f}")

            # Target eval env
            eval_env = make_modified_walker2d(target_name)
            state_dim = eval_env.observation_space.shape[0]
            act_dim = eval_env.action_space.shape[0]

            # -----------------------------------------------------------------
            # A) Fine-tune from pre-trained checkpoint
            # -----------------------------------------------------------------
            print(f"\n--- Fine-tuning from {source_ds} checkpoint ---")
            ft_config = {
                **FINETUNE_CONFIG,
                "state_dim": state_dim,
                "act_dim": act_dim,
                "device": device,
            }
            ft_model = create_model(ft_config)
            ft_model.load_state_dict(checkpoint["model_state_dict"])

            ft_logs = train_model(
                ft_model, ft_config, target_trajectories, eval_env,
                target_state_mean, target_state_std,
                save_path=os.path.join(
                    args.checkpoint_dir,
                    f"finetuned_{source_ds}_to_{target_name}.pt",
                ),
                log_prefix=f"[finetune/{exp_key}]",
            )

            # -----------------------------------------------------------------
            # B) Train from scratch on same target data
            # -----------------------------------------------------------------
            print(f"\n--- Training from scratch on {target_name} ---")
            scratch_config = {
                **SCRATCH_CONFIG,
                "state_dim": state_dim,
                "act_dim": act_dim,
                "device": device,
            }
            scratch_model = create_model(scratch_config)

            scratch_logs = train_model(
                scratch_model, scratch_config, target_trajectories, eval_env,
                target_state_mean, target_state_std,
                save_path=os.path.join(
                    args.checkpoint_dir,
                    f"scratch_{target_name}.pt",
                ),
                log_prefix=f"[scratch/{target_name}]",
            )

            # Store results
            results[exp_key] = {
                "finetune_logs": ft_logs,
                "scratch_logs": scratch_logs,
                "source_dataset": source_ds,
                "target_domain": target_name,
            }

            eval_env.close()

    # Save all results
    results_path = os.path.join(args.output_dir, "transfer_results.json")
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert to JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\nAll results saved to: {results_path}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Task Generalization Experiment for Decision Transformer"
    )
    parser.add_argument(
        "--phase", type=str, required=True,
        choices=["pretrain", "transfer", "all"],
        help="Which phase to run",
    )
    parser.add_argument(
        "--source_datasets", type=str,
        default="medium,medium-replay,medium-expert",
        help="Comma-separated list of D4RL dataset types for pre-training",
    )
    parser.add_argument(
        "--target", type=str, default="all",
        help="Target domain(s), comma-separated or 'all'",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Directory to save/load model checkpoints",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to save experiment results",
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.phase in ("pretrain", "all"):
        run_pretrain(args)

    if args.phase in ("transfer", "all"):
        run_transfer(args)


if __name__ == "__main__":
    main()
