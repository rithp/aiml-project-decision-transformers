"""
Collect offline trajectory data from modified Walker2d environments.

Strategy: Load D4RL's medium policy (SAC checkpoint) and roll it out in the
modified environment. If that's unavailable, we use an alternative approach:
run the *original* Walker2d-medium dataset's behavioral policy by loading the
pre-trained DT itself (or fall back to random + replay).

Usage:
    python collect_target_data.py \
        --target heavy \
        --num_trajectories 50 \
        --output_dir data/target
"""

import argparse
import collections
import os
import pickle

import gymnasium as gym
import numpy as np

from cross_task.modified_envs import make_modified_walker2d, get_all_target_names


def collect_random_trajectories(env, num_trajectories=50, max_ep_len=1000):
    """Collect trajectories using a random policy. Useful as a baseline."""
    paths = []
    for i in range(num_trajectories):
        data = collections.defaultdict(list)
        obs, _ = env.reset()
        for t in range(max_ep_len):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            data["observations"].append(obs)
            data["next_observations"].append(next_obs)
            data["actions"].append(action)
            data["rewards"].append(reward)
            data["terminals"].append(done)
            obs = next_obs
            if done:
                break
        episode_data = {k: np.array(v) for k, v in data.items()}
        paths.append(episode_data)
        ret = episode_data["rewards"].sum()
        if (i + 1) % 10 == 0:
            print(f"  Collected {i+1}/{num_trajectories} trajectories, "
                  f"latest return: {ret:.1f}")
    return paths


def collect_with_source_policy(env, source_dataset_path, num_trajectories=50,
                                max_ep_len=1000):
    """
    Replay actions from source dataset trajectories in the modified environment.
    
    This is a simple but effective approach: take the action sequences from the
    source domain dataset and execute them open-loop in the target environment.
    The dynamics mismatch means the resulting trajectories will differ from the
    source, creating a valid target-domain dataset.
    
    For a more principled approach, you could train a policy (SAC/PPO) in the
    target environment, but this simple method suffices for the experiment.
    """
    with open(source_dataset_path, "rb") as f:
        source_trajectories = pickle.load(f)

    # Sort by return (highest first) and pick top trajectories
    source_returns = [t["rewards"].sum() for t in source_trajectories]
    sorted_indices = np.argsort(source_returns)[::-1]

    paths = []
    for i in range(min(num_trajectories, len(source_trajectories))):
        src_traj = source_trajectories[sorted_indices[i % len(sorted_indices)]]
        data = collections.defaultdict(list)
        obs, _ = env.reset()
        src_actions = src_traj["actions"]

        for t in range(min(max_ep_len, len(src_actions))):
            action = src_actions[t]
            # Clip to action space bounds
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            data["observations"].append(obs)
            data["next_observations"].append(next_obs)
            data["actions"].append(action)
            data["rewards"].append(reward)
            data["terminals"].append(done)
            obs = next_obs
            if done:
                break

        episode_data = {k: np.array(v) for k, v in data.items()}
        paths.append(episode_data)
        ret = episode_data["rewards"].sum()
        if (i + 1) % 10 == 0:
            print(f"  Collected {i+1}/{num_trajectories} trajectories, "
                  f"latest return: {ret:.1f}")

    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True,
                        choices=get_all_target_names() + ["all"],
                        help="Target domain config name")
    parser.add_argument("--num_trajectories", type=int, default=50)
    parser.add_argument("--source_dataset", type=str, default=None,
                        help="Path to source domain .pkl for action replay. "
                             "If not provided, uses random policy.")
    parser.add_argument("--output_dir", type=str, default="data/target")
    parser.add_argument("--max_ep_len", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    targets = get_all_target_names() if args.target == "all" else [args.target]

    for target_name in targets:
        print(f"\n{'='*60}")
        print(f"Collecting data for target: {target_name}")
        print(f"{'='*60}")

        env = make_modified_walker2d(target_name)

        if args.source_dataset and os.path.exists(args.source_dataset):
            print(f"Using source policy replay from: {args.source_dataset}")
            paths = collect_with_source_policy(
                env, args.source_dataset,
                num_trajectories=args.num_trajectories,
                max_ep_len=args.max_ep_len,
            )
        else:
            print("Using random policy (no source dataset provided)")
            paths = collect_random_trajectories(
                env, num_trajectories=args.num_trajectories,
                max_ep_len=args.max_ep_len,
            )

        # Print statistics
        returns = np.array([p["rewards"].sum() for p in paths])
        lengths = np.array([len(p["rewards"]) for p in paths])
        print(f"\nDataset statistics for '{target_name}':")
        print(f"  Trajectories: {len(paths)}")
        print(f"  Total timesteps: {lengths.sum()}")
        print(f"  Return: mean={returns.mean():.1f}, std={returns.std():.1f}, "
              f"max={returns.max():.1f}, min={returns.min():.1f}")
        print(f"  Length: mean={lengths.mean():.1f}")

        # Save
        out_path = os.path.join(args.output_dir,
                                f"walker2d-{target_name}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(paths, f)
        print(f"  Saved to: {out_path}")

        env.close()


if __name__ == "__main__":
    main()
