# # """
# # PSDT vs DT on D4RL MuJoCo (Normal + Delayed Rewards)

# # This experiment compares PSDT against standard DT on the D4RL MuJoCo
# # benchmarks, with a focus on the delayed (sparse) reward setting where
# # memory provides the most advantage.

# # In delayed mode, all intermediate rewards are zero and the total return
# # is given only at the final timestep. This makes the RTG signal within
# # a context window less informative, requiring the model to integrate
# # information over longer horizons — exactly what PSDT's cell state does.

# # Setup:
# #     1. Download D4RL datasets: python download_d4rl.py
# #     2. Run PSDT:  python mujoco_experiment.py --model_type psdt --env hopper --dataset medium --mode delayed
# #     3. Run DT:    python mujoco_experiment.py --model_type dt --env hopper --dataset medium --mode delayed
# #     4. Compare results

# # The code reuses the original DT codebase's GPT2 backbone and adds PSDT
# # components on top. Continuous actions with MSE loss — no discrete action issues.
# # """

# # import gymnasium as gym
# # import numpy as np
# # import torch
# # import torch.nn.functional as F

# # import argparse
# # import pickle
# # import random
# # import os
# # import time

# # from decision_transformer.models.decision_transformer import DecisionTransformer
# # from decision_transformer.models.psdt import PSDTModel


# # def discount_cumsum(x, gamma):
# #     ret = np.zeros_like(x)
# #     ret[-1] = x[-1]
# #     for t in reversed(range(x.shape[0] - 1)):
# #         ret[t] = x[t] + gamma * ret[t + 1]
# #     return ret


# # def evaluate_episode_rtg(env, state_dim, act_dim, model, model_type,
# #                          max_ep_len=1000, scale=1000., state_mean=0., state_std=1.,
# #                          device='cuda', target_return=None, mode='normal'):
# #     """Evaluate one episode. For PSDT, maintains cell state across steps."""
# #     model.eval()
# #     model.to(device=device)

# #     state_mean_t = torch.from_numpy(state_mean).to(device=device)
# #     state_std_t = torch.from_numpy(state_std).to(device=device)

# #     state, _ = env.reset()
# #     if mode == 'noise':
# #         state = state + np.random.normal(0, 0.1, size=state.shape)

# #     states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
# #     actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
# #     rewards = torch.zeros(0, device=device, dtype=torch.float32)

# #     ep_return = target_return
# #     target_return_t = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
# #     timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

# #     # PSDT: initialize cell state
# #     cell_state = None
# #     if model_type == 'psdt':
# #         cell_state = model.init_cell_state(batch_size=1, device=device)

# #     episode_return, episode_length = 0, 0

# #     for t in range(max_ep_len):
# #         actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
# #         rewards = torch.cat([rewards, torch.zeros(1, device=device)])

# #         if model_type == 'psdt':
# #             action, cell_state = model.get_action(
# #                 (states.to(dtype=torch.float32) - state_mean_t) / state_std_t,
# #                 actions.to(dtype=torch.float32),
# #                 rewards.to(dtype=torch.float32),
# #                 target_return_t.to(dtype=torch.float32),
# #                 timesteps.to(dtype=torch.long),
# #                 cell_state=cell_state,
# #             )
# #             cell_state = cell_state.detach()
# #         else:
# #             action = model.get_action(
# #                 (states.to(dtype=torch.float32) - state_mean_t) / state_std_t,
# #                 actions.to(dtype=torch.float32),
# #                 rewards.to(dtype=torch.float32),
# #                 target_return_t.to(dtype=torch.float32),
# #                 timesteps.to(dtype=torch.long),
# #             )

# #         actions[-1] = action
# #         action_np = action.detach().cpu().numpy()


# #         state, reward, terminated, truncated, _ = env.step(action_np)
# #         done = terminated or truncated
        
# #         cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
# #         states = torch.cat([states, cur_state], dim=0)
# #         rewards[-1] = reward

# #         if mode != 'delayed':
# #             pred_return = target_return_t[0, -1] - (reward / scale)
# #         else:
# #             pred_return = target_return_t[0, -1]
# #         target_return_t = torch.cat([target_return_t, pred_return.reshape(1, 1)], dim=1)
# #         timesteps = torch.cat([
# #             timesteps,
# #             torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)
# #         ], dim=1)

# #         episode_return += reward
# #         episode_length += 1

# #         if done:
# #             break

# #     return episode_return, episode_length


# # def experiment(variant):
# #     device = variant.get('device', 'cuda')
# #     env_name = variant['env']
# #     dataset = variant['dataset']
# #     model_type = variant['model_type']
# #     mode = variant.get('mode', 'normal')

# #     # Environment setup
# #     if env_name == 'hopper':
# #         env = gym.make('Hopper-v4')
# #         max_ep_len = 1000
# #         env_targets = [3600, 1800]
# #         scale = 1000.
# #     elif env_name == 'halfcheetah':
# #         env = gym.make('HalfCheetah-v4')
# #         max_ep_len = 1000
# #         env_targets = [12000, 6000]
# #         scale = 1000.
# #     elif env_name == 'walker2d':
# #         env = gym.make('Walker2d-v4')
# #         max_ep_len = 1000
# #         env_targets = [5000, 2500]
# #         scale = 1000.
# #     else:
# #         raise NotImplementedError(f'Unknown env: {env_name}')

# #     state_dim = env.observation_space.shape[0]
# #     act_dim = env.action_space.shape[0]

# #     # Load dataset
# #     dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
# #     if not os.path.exists(dataset_path):
# #         print(f'Dataset not found at {dataset_path}')
# #         print(f'Run: python download_d4rl.py')
# #         return

# #     with open(dataset_path, 'rb') as f:
# #         trajectories = pickle.load(f)

# #     # Apply delayed reward mode
# #     if mode == 'delayed':
# #         for path in trajectories:
# #             path['rewards'][-1] = path['rewards'].sum()
# #             path['rewards'][:-1] = 0.

# #     states_all, traj_lens, returns = [], [], []
# #     for path in trajectories:
# #         states_all.append(path['observations'])
# #         traj_lens.append(len(path['observations']))
# #         returns.append(path['rewards'].sum())
# #     traj_lens, returns = np.array(traj_lens), np.array(returns)

# #     states_concat = np.concatenate(states_all, axis=0)
# #     state_mean, state_std = np.mean(states_concat, axis=0), np.std(states_concat, axis=0) + 1e-6

# #     num_timesteps = sum(traj_lens)
# #     K = variant['K']
# #     batch_size = variant['batch_size']
# #     n_pred_steps = variant.get('n_pred_steps', 3)

# #     print('=' * 60)
# #     print(f'D4RL MuJoCo Experiment')
# #     print(f'  Env: {env_name}-{dataset}')
# #     print(f'  Mode: {mode} {"(all rewards moved to final step)" if mode == "delayed" else ""}')
# #     print(f'  Model: {model_type}')
# #     print(f'  K: {K}, State dim: {state_dim}, Act dim: {act_dim}')
# #     print(f'  {len(traj_lens)} trajectories, {num_timesteps} timesteps')
# #     print(f'  Returns: mean={np.mean(returns):.1f}, std={np.std(returns):.1f}, '
# #           f'max={np.max(returns):.1f}, min={np.min(returns):.1f}')
# #     print('=' * 60)

# #     # Select top trajectories for training
# #     pct_traj = variant.get('pct_traj', 1.)
# #     num_timesteps_target = max(int(pct_traj * num_timesteps), 1)
# #     sorted_inds = np.argsort(returns)
# #     num_trajectories = 1
# #     timesteps_count = traj_lens[sorted_inds[-1]]
# #     ind = len(trajectories) - 2
# #     while ind >= 0 and timesteps_count + traj_lens[sorted_inds[ind]] <= num_timesteps_target:
# #         timesteps_count += traj_lens[sorted_inds[ind]]
# #         num_trajectories += 1
# #         ind -= 1
# #     sorted_inds = sorted_inds[-num_trajectories:]
# #     p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

# #     def get_batch(batch_size=256, max_len=K):
# #         """Sample batch with future ground truth for PSDT prediction loss."""
# #         batch_inds = np.random.choice(
# #             np.arange(num_trajectories), size=batch_size, replace=True, p=p_sample)

# #         s, a, r, d, rtg, timesteps_b, mask = [], [], [], [], [], [], []
# #         fut_s, fut_a, fut_rtg = [], [], []

# #         for i in range(batch_size):
# #             traj = trajectories[int(sorted_inds[batch_inds[i]])]
# #             traj_len = traj['rewards'].shape[0]
# #             si = random.randint(0, traj_len - 1)

# #             # Current segment
# #             s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
# #             a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
# #             r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
# #             if 'terminals' in traj:
# #                 d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
# #             else:
# #                 d.append(traj['dones'][si:si + max_len].reshape(1, -1))
# #             timesteps_b.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
# #             timesteps_b[-1][timesteps_b[-1] >= max_ep_len] = max_ep_len - 1
# #             rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
# #             if rtg[-1].shape[1] <= s[-1].shape[1]:
# #                 rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

# #             # Pad current segment
# #             tlen = s[-1].shape[1]
# #             s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
# #             s[-1] = (s[-1] - state_mean) / state_std
# #             a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
# #             r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
# #             d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
# #             rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
# #             timesteps_b[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps_b[-1]], axis=1)
# #             mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

# #             # Future ground truth for prediction loss
# #             fut_start = si + max_len
# #             fs_raw = traj['observations'][fut_start:fut_start + n_pred_steps]
# #             fa_raw = traj['actions'][fut_start:fut_start + n_pred_steps]
# #             fr_raw_rewards = traj['rewards'][fut_start:]
# #             n_avail = min(len(fs_raw), len(fa_raw), len(fr_raw_rewards))

# #             if n_avail > 0:
# #                 fs = (fs_raw[:n_avail] - state_mean) / state_std
# #                 fa = fa_raw[:n_avail]
# #                 fr = discount_cumsum(fr_raw_rewards, gamma=1.)[:n_avail] / scale
# #                 n_avail = min(n_avail, n_pred_steps)
# #                 fs, fa, fr = fs[:n_avail], fa[:n_avail], fr[:n_avail]
# #             else:
# #                 n_avail = 0
# #                 fs = np.zeros((0, state_dim))
# #                 fa = np.zeros((0, act_dim))
# #                 fr = np.zeros((0,))

# #             pad = n_pred_steps - n_avail
# #             if pad > 0:
# #                 fs = np.concatenate([fs, np.zeros((pad, state_dim))])
# #                 fa = np.concatenate([fa, np.zeros((pad, act_dim))])
# #                 fr = np.concatenate([fr, np.zeros(pad)])

# #             fut_s.append(fs.reshape(1, n_pred_steps, state_dim))
# #             fut_a.append(fa.reshape(1, n_pred_steps, act_dim))
# #             fut_rtg.append(fr.reshape(1, n_pred_steps, 1))

# #         s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
# #         a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
# #         r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
# #         d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
# #         rtg_t = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
# #         ts = torch.from_numpy(np.concatenate(timesteps_b, axis=0)).to(dtype=torch.long, device=device)
# #         mask_t = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
# #         fut_s_t = torch.from_numpy(np.concatenate(fut_s, axis=0)).to(dtype=torch.float32, device=device)
# #         fut_a_t = torch.from_numpy(np.concatenate(fut_a, axis=0)).to(dtype=torch.float32, device=device)
# #         fut_rtg_t = torch.from_numpy(np.concatenate(fut_rtg, axis=0)).to(dtype=torch.float32, device=device)

# #         return s, a, r, d, rtg_t, ts, mask_t, fut_s_t, fut_a_t, fut_rtg_t

# #     # Evaluation function
# #     def eval_episodes(target_rew):
# #         def fn(model):
# #             returns_list, lengths_list = [], []
# #             for _ in range(variant['num_eval_episodes']):
# #                 with torch.no_grad():
# #                     ret, length = evaluate_episode_rtg(
# #                         env, state_dim, act_dim, model, model_type,
# #                         max_ep_len=max_ep_len, scale=scale,
# #                         target_return=target_rew / scale,
# #                         mode=mode,
# #                         state_mean=state_mean, state_std=state_std,
# #                         device=device,
# #                     )
# #                 returns_list.append(ret)
# #                 lengths_list.append(length)
# #             return {
# #                 f'target_{target_rew}_return_mean': np.mean(returns_list),
# #                 f'target_{target_rew}_return_std': np.std(returns_list),
# #                 f'target_{target_rew}_length_mean': np.mean(lengths_list),
# #             }
# #         return fn

# #     # Create model
# #     embed_dim = variant['embed_dim']

# #     if model_type == 'psdt':
# #         model = PSDTModel(
# #             state_dim=state_dim, act_dim=act_dim,
# #             hidden_size=embed_dim,
# #             n_actions=None,  # continuous actions
# #             cell_size=variant.get('cell_size', embed_dim),
# #             n_pred_steps=n_pred_steps,
# #             pred_loss_weight=variant.get('pred_loss_weight', 0.5),
# #             max_length=K, max_ep_len=max_ep_len,
# #             n_layer=variant['n_layer'],
# #             n_head=variant['n_head'],
# #             n_inner=4 * embed_dim,
# #             activation_function=variant['activation_function'],
# #             n_positions=1024,
# #             resid_pdrop=variant['dropout'],
# #             attn_pdrop=variant['dropout'],
# #         )
# #     else:
# #         model = DecisionTransformer(
# #             state_dim=state_dim, act_dim=act_dim,
# #             hidden_size=embed_dim,
# #             n_actions=None,
# #             max_length=K, max_ep_len=max_ep_len,
# #             n_layer=variant['n_layer'],
# #             n_head=variant['n_head'],
# #             n_inner=4 * embed_dim,
# #             activation_function=variant['activation_function'],
# #             n_positions=1024,
# #             resid_pdrop=variant['dropout'],
# #             attn_pdrop=variant['dropout'],
# #         )

# #     model = model.to(device=device)
# #     total_params = sum(p.numel() for p in model.parameters())
# #     print(f'Total parameters: {total_params:,}')

# #     optimizer = torch.optim.AdamW(
# #         model.parameters(), lr=variant['learning_rate'], weight_decay=variant['weight_decay'])
# #     scheduler = torch.optim.lr_scheduler.LambdaLR(
# #         optimizer, lambda steps: min((steps + 1) / variant['warmup_steps'], 1))

# #     # Training loop
# #     eval_fns = [eval_episodes(tar) for tar in env_targets]
# #     start_time = time.time()

# #     for iter_num in range(variant['max_iters']):
# #         model.train()
# #         train_losses = []
# #         action_losses = []
# #         pred_losses = []

# #         for _ in range(variant['num_steps_per_iter']):
# #             batch = get_batch(batch_size)
# #             states_b, actions_b, rewards_b, dones_b, rtg_b, ts_b, mask_b = batch[:7]
# #             fut_s_b, fut_a_b, fut_rtg_b = batch[7], batch[8], batch[9]

# #             action_target = torch.clone(actions_b)

# #             if model_type == 'psdt':
# #                 _, action_preds, _, _, pred_loss = model.forward(
# #                     states_b, actions_b, rewards_b, rtg_b[:, :-1], ts_b,
# #                     attention_mask=mask_b, cell_state=None,
# #                     future_states=fut_s_b, future_actions=fut_a_b, future_rtg=fut_rtg_b,
# #                 )
# #                 # MSE action loss (continuous actions)
# #                 act_d = action_preds.shape[2]
# #                 ap = action_preds.reshape(-1, act_d)[mask_b.reshape(-1) > 0]
# #                 at = action_target.reshape(-1, act_d)[mask_b.reshape(-1) > 0]
# #                 action_loss = torch.mean((ap - at) ** 2)
# #                 loss = action_loss + variant.get('pred_loss_weight', 0.5) * pred_loss
# #                 pred_losses.append(pred_loss.item())
# #             else:
# #                 _, action_preds, _ = model.forward(
# #                     states_b, actions_b, rewards_b, rtg_b[:, :-1], ts_b,
# #                     attention_mask=mask_b,
# #                 )
# #                 act_d = action_preds.shape[2]
# #                 ap = action_preds.reshape(-1, act_d)[mask_b.reshape(-1) > 0]
# #                 at = action_target.reshape(-1, act_d)[mask_b.reshape(-1) > 0]
# #                 action_loss = torch.mean((ap - at) ** 2)
# #                 loss = action_loss

# #             optimizer.zero_grad()
# #             loss.backward()
# #             torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
# #             optimizer.step()
# #             scheduler.step()

# #             train_losses.append(loss.item())
# #             action_losses.append(action_loss.item())

# #         # Evaluate
# #         model.eval()
# #         eval_results = {}
# #         for eval_fn in eval_fns:
# #             outputs = eval_fn(model)
# #             for k, v in outputs.items():
# #                 eval_results[k] = v

# #         # Print results
# #         elapsed = time.time() - start_time
# #         avg_loss = np.mean(train_losses)
# #         avg_act_loss = np.mean(action_losses)

# #         print(f'Iter {iter_num + 1}/{variant["max_iters"]} ({elapsed:.0f}s) | '
# #               f'Loss: {avg_loss:.4f} | Act loss: {avg_act_loss:.4f}', end='')
# #         if pred_losses:
# #             print(f' | Pred loss: {np.mean(pred_losses):.4f}', end='')

# #         for target in env_targets:
# #             ret_key = f'target_{target}_return_mean'
# #             if ret_key in eval_results:
# #                 normalized = eval_results[ret_key] / target * 100
# #                 print(f' | T={target}: {eval_results[ret_key]:.1f} ({normalized:.1f}%)', end='')
# #         print()

# #     # Save model
# #     os.makedirs('checkpoints', exist_ok=True)
# #     save_path = f'checkpoints/{model_type}_{env_name}_{dataset}_{mode}.pt'
# #     torch.save(model.state_dict(), save_path)
# #     print(f'\nModel saved to {save_path}')


# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser()

# #     # Environment
# #     parser.add_argument('--env', type=str, default='hopper',
# #                         choices=['hopper', 'halfcheetah', 'walker2d'])
# #     parser.add_argument('--dataset', type=str, default='medium',
# #                         choices=['medium', 'medium-replay', 'medium-expert', 'expert'])
# #     parser.add_argument('--mode', type=str, default='delayed',
# #                         choices=['normal', 'delayed'],
# #                         help='delayed = sparse reward (all reward at final step)')

# #     # Model
# #     parser.add_argument('--model_type', type=str, default='psdt',
# #                         choices=['psdt', 'dt'])
# #     parser.add_argument('--K', type=int, default=20)
# #     parser.add_argument('--embed_dim', type=int, default=128)
# #     parser.add_argument('--n_layer', type=int, default=3)
# #     parser.add_argument('--n_head', type=int, default=1)
# #     parser.add_argument('--activation_function', type=str, default='relu')
# #     parser.add_argument('--dropout', type=float, default=0.1)

# #     # PSDT specific
# #     parser.add_argument('--cell_size', type=int, default=128)
# #     parser.add_argument('--n_pred_steps', type=int, default=3)
# #     parser.add_argument('--pred_loss_weight', type=float, default=0.5)

# #     # Training
# #     parser.add_argument('--batch_size', type=int, default=64)
# #     parser.add_argument('--learning_rate', type=float, default=1e-4)
# #     parser.add_argument('--weight_decay', type=float, default=1e-4)
# #     parser.add_argument('--warmup_steps', type=int, default=10000)
# #     parser.add_argument('--max_iters', type=int, default=10)
# #     parser.add_argument('--num_steps_per_iter', type=int, default=10000)
# #     parser.add_argument('--num_eval_episodes', type=int, default=100)
# #     parser.add_argument('--pct_traj', type=float, default=1.)

# #     # System
# #     parser.add_argument('--device', type=str, default='cuda')

# #     args = parser.parse_args()
# #     experiment(vars(args))


# """
# PSDT vs DT on D4RL MuJoCo (Normal + Delayed Rewards)

# This experiment compares PSDT against standard DT on the D4RL MuJoCo
# benchmarks, with a focus on the delayed (sparse) reward setting where
# memory provides the most advantage.

# In delayed mode, all intermediate rewards are zero and the total return
# is given only at the final timestep. This makes the RTG signal within
# a context window less informative, requiring the model to integrate
# information over longer horizons — exactly what PSDT's cell state does.

# Setup:
#     1. Download D4RL datasets: python download_d4rl.py
#     2. Run PSDT:  python mujoco_experiment.py --model_type psdt --env hopper --dataset medium --mode delayed
#     3. Run DT:    python mujoco_experiment.py --model_type dt --env hopper --dataset medium --mode delayed
#     4. Compare results

# The code reuses the original DT codebase's GPT2 backbone and adds PSDT
# components on top. Continuous actions with MSE loss — no discrete action issues.
# """

# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn.functional as F

# import argparse
# import pickle
# import random
# import os
# import time

# from decision_transformer.models.decision_transformer import DecisionTransformer
# from decision_transformer.models.psdt import PSDTModel


# def discount_cumsum(x, gamma):
#     ret = np.zeros_like(x)
#     ret[-1] = x[-1]
#     for t in reversed(range(x.shape[0] - 1)):
#         ret[t] = x[t] + gamma * ret[t + 1]
#     return ret


# def evaluate_episode_rtg(env, state_dim, act_dim, model, model_type,
#                          max_ep_len=1000, scale=1000., state_mean=0., state_std=1.,
#                          device='cuda', target_return=None, mode='normal'):
#     """Evaluate one episode. For PSDT, maintains cell state across steps."""
#     model.eval()
#     model.to(device=device)

#     state_mean_t = torch.from_numpy(state_mean).to(device=device)
#     state_std_t = torch.from_numpy(state_std).to(device=device)

#     state = env.reset()
#     if mode == 'noise':
#         state = state + np.random.normal(0, 0.1, size=state.shape)

#     states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
#     actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
#     rewards = torch.zeros(0, device=device, dtype=torch.float32)

#     ep_return = target_return
#     target_return_t = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
#     timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

#     # PSDT: initialize cell state
#     cell_state = None
#     if model_type == 'psdt':
#         cell_state = model.init_cell_state(batch_size=1, device=device)

#     episode_return, episode_length = 0, 0

#     for t in range(max_ep_len):
#         actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
#         rewards = torch.cat([rewards, torch.zeros(1, device=device)])

#         if model_type == 'psdt':
#             action, cell_state = model.get_action(
#                 (states.to(dtype=torch.float32) - state_mean_t) / state_std_t,
#                 actions.to(dtype=torch.float32),
#                 rewards.to(dtype=torch.float32),
#                 target_return_t.to(dtype=torch.float32),
#                 timesteps.to(dtype=torch.long),
#                 cell_state=cell_state,
#             )
#             cell_state = cell_state.detach()
#         else:
#             action = model.get_action(
#                 (states.to(dtype=torch.float32) - state_mean_t) / state_std_t,
#                 actions.to(dtype=torch.float32),
#                 rewards.to(dtype=torch.float32),
#                 target_return_t.to(dtype=torch.float32),
#                 timesteps.to(dtype=torch.long),
#             )

#         actions[-1] = action
#         action_np = action.detach().cpu().numpy()

#         state, reward, done, _ = env.step(action_np)

#         cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
#         states = torch.cat([states, cur_state], dim=0)
#         rewards[-1] = reward

#         if mode != 'delayed':
#             pred_return = target_return_t[0, -1] - (reward / scale)
#         else:
#             pred_return = target_return_t[0, -1]
#         target_return_t = torch.cat([target_return_t, pred_return.reshape(1, 1)], dim=1)
#         timesteps = torch.cat([
#             timesteps,
#             torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)
#         ], dim=1)

#         episode_return += reward
#         episode_length += 1

#         if done:
#             break

#     return episode_return, episode_length


# def experiment(variant):
#     device = variant.get('device', 'cuda')
#     env_name = variant['env']
#     dataset = variant['dataset']
#     model_type = variant['model_type']
#     mode = variant.get('mode', 'normal')

#     # Environment setup
#     if env_name == 'hopper':
#         env = gym.make('Hopper-v3')
#         max_ep_len = 1000
#         env_targets = [3600, 1800]
#         scale = 1000.
#     elif env_name == 'halfcheetah':
#         env = gym.make('HalfCheetah-v3')
#         max_ep_len = 1000
#         env_targets = [12000, 6000]
#         scale = 1000.
#     elif env_name == 'walker2d':
#         env = gym.make('Walker2d-v3')
#         max_ep_len = 1000
#         env_targets = [5000, 2500]
#         scale = 1000.
#     else:
#         raise NotImplementedError(f'Unknown env: {env_name}')

#     state_dim = env.observation_space.shape[0]
#     act_dim = env.action_space.shape[0]

#     # Load dataset
#     dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
#     if not os.path.exists(dataset_path):
#         print(f'Dataset not found at {dataset_path}')
#         print(f'Run: python download_d4rl.py')
#         return

#     with open(dataset_path, 'rb') as f:
#         trajectories = pickle.load(f)

#     # Apply delayed reward mode
#     if mode == 'delayed':
#         for path in trajectories:
#             path['rewards'][-1] = path['rewards'].sum()
#             path['rewards'][:-1] = 0.

#     states_all, traj_lens, returns = [], [], []
#     for path in trajectories:
#         states_all.append(path['observations'])
#         traj_lens.append(len(path['observations']))
#         returns.append(path['rewards'].sum())
#     traj_lens, returns = np.array(traj_lens), np.array(returns)

#     states_concat = np.concatenate(states_all, axis=0)
#     state_mean, state_std = np.mean(states_concat, axis=0), np.std(states_concat, axis=0) + 1e-6

#     num_timesteps = sum(traj_lens)
#     K = variant['K']
#     batch_size = variant['batch_size']
#     n_pred_steps = variant.get('n_pred_steps', 3)

#     print('=' * 60)
#     print(f'D4RL MuJoCo Experiment')
#     print(f'  Env: {env_name}-{dataset}')
#     print(f'  Mode: {mode} {"(all rewards moved to final step)" if mode == "delayed" else ""}')
#     print(f'  Model: {model_type}')
#     print(f'  K: {K}, State dim: {state_dim}, Act dim: {act_dim}')
#     print(f'  {len(traj_lens)} trajectories, {num_timesteps} timesteps')
#     print(f'  Returns: mean={np.mean(returns):.1f}, std={np.std(returns):.1f}, '
#           f'max={np.max(returns):.1f}, min={np.min(returns):.1f}')
#     print('=' * 60)

#     # Select top trajectories for training
#     pct_traj = variant.get('pct_traj', 1.)
#     num_timesteps_target = max(int(pct_traj * num_timesteps), 1)
#     sorted_inds = np.argsort(returns)
#     num_trajectories = 1
#     timesteps_count = traj_lens[sorted_inds[-1]]
#     ind = len(trajectories) - 2
#     while ind >= 0 and timesteps_count + traj_lens[sorted_inds[ind]] <= num_timesteps_target:
#         timesteps_count += traj_lens[sorted_inds[ind]]
#         num_trajectories += 1
#         ind -= 1
#     sorted_inds = sorted_inds[-num_trajectories:]
#     p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

#     def get_batch(batch_size=256, max_len=K):
#         """Sample batch with future ground truth for PSDT prediction loss."""
#         batch_inds = np.random.choice(
#             np.arange(num_trajectories), size=batch_size, replace=True, p=p_sample)

#         s, a, r, d, rtg, timesteps_b, mask = [], [], [], [], [], [], []
#         fut_s, fut_a, fut_rtg = [], [], []

#         for i in range(batch_size):
#             traj = trajectories[int(sorted_inds[batch_inds[i]])]
#             traj_len = traj['rewards'].shape[0]
#             si = random.randint(0, traj_len - 1)

#             # Current segment
#             s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
#             a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
#             r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
#             if 'terminals' in traj:
#                 d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
#             else:
#                 d.append(traj['dones'][si:si + max_len].reshape(1, -1))
#             timesteps_b.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
#             timesteps_b[-1][timesteps_b[-1] >= max_ep_len] = max_ep_len - 1
#             rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
#             if rtg[-1].shape[1] <= s[-1].shape[1]:
#                 rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

#             # Pad current segment
#             tlen = s[-1].shape[1]
#             s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
#             s[-1] = (s[-1] - state_mean) / state_std
#             a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
#             r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
#             d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
#             rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
#             timesteps_b[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps_b[-1]], axis=1)
#             mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

#             # Future ground truth for prediction loss
#             fut_start = si + max_len
#             fs_raw = traj['observations'][fut_start:fut_start + n_pred_steps]
#             fa_raw = traj['actions'][fut_start:fut_start + n_pred_steps]
#             fr_raw_rewards = traj['rewards'][fut_start:]
#             n_avail = min(len(fs_raw), len(fa_raw), len(fr_raw_rewards))

#             if n_avail > 0:
#                 fs = (fs_raw[:n_avail] - state_mean) / state_std
#                 fa = fa_raw[:n_avail]
#                 fr = discount_cumsum(fr_raw_rewards, gamma=1.)[:n_avail] / scale
#                 n_avail = min(n_avail, n_pred_steps)
#                 fs, fa, fr = fs[:n_avail], fa[:n_avail], fr[:n_avail]
#             else:
#                 n_avail = 0
#                 fs = np.zeros((0, state_dim))
#                 fa = np.zeros((0, act_dim))
#                 fr = np.zeros((0,))

#             pad = n_pred_steps - n_avail
#             if pad > 0:
#                 fs = np.concatenate([fs, np.zeros((pad, state_dim))])
#                 fa = np.concatenate([fa, np.zeros((pad, act_dim))])
#                 fr = np.concatenate([fr, np.zeros(pad)])

#             fut_s.append(fs.reshape(1, n_pred_steps, state_dim))
#             fut_a.append(fa.reshape(1, n_pred_steps, act_dim))
#             fut_rtg.append(fr.reshape(1, n_pred_steps, 1))

#         s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
#         a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
#         r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
#         d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
#         rtg_t = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
#         ts = torch.from_numpy(np.concatenate(timesteps_b, axis=0)).to(dtype=torch.long, device=device)
#         mask_t = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
#         fut_s_t = torch.from_numpy(np.concatenate(fut_s, axis=0)).to(dtype=torch.float32, device=device)
#         fut_a_t = torch.from_numpy(np.concatenate(fut_a, axis=0)).to(dtype=torch.float32, device=device)
#         fut_rtg_t = torch.from_numpy(np.concatenate(fut_rtg, axis=0)).to(dtype=torch.float32, device=device)

#         return s, a, r, d, rtg_t, ts, mask_t, fut_s_t, fut_a_t, fut_rtg_t

#     # ====================================================================
#     # Segment-threaded batch sampler for PSDT.
#     # Returns N consecutive segments per trajectory so cell state can be
#     # threaded across them during training (mirrors eval behavior).
#     # Shapes: (B, N, K, ...) for segment tensors; future tensors stay
#     # (B, n_pred_steps, ...) and are aligned with the END of the LAST segment.
#     # ====================================================================
#     n_segments = variant.get('n_segments', 3)

#     def get_batch_threaded(batch_size=256, max_len=K):
#         N = n_segments
#         batch_inds = np.random.choice(
#             np.arange(num_trajectories), size=batch_size, replace=True, p=p_sample)

#         s_all, a_all, r_all, d_all = [], [], [], []
#         rtg_all, ts_all, mask_all = [], [], []
#         fut_s_all, fut_a_all, fut_rtg_all = [], [], []

#         for i in range(batch_size):
#             traj = trajectories[int(sorted_inds[batch_inds[i]])]
#             traj_len = traj['rewards'].shape[0]
#             si = random.randint(0, max(0, traj_len - 1))

#             s_segs, a_segs, r_segs, d_segs = [], [], [], []
#             rtg_segs, ts_segs, mask_segs = [], [], []

#             for seg in range(N):
#                 seg_start = si + seg * max_len
#                 seg_end = seg_start + max_len

#                 if seg_start >= traj_len:
#                     # Fully-padded segment (past trajectory end)
#                     s_seg = (np.zeros((max_len, state_dim)) - state_mean) / state_std
#                     a_seg = np.ones((max_len, act_dim)) * -10.0
#                     r_seg = np.zeros((max_len, 1))
#                     d_seg = np.ones((max_len,)) * 2
#                     rtg_seg = np.zeros((max_len + 1, 1))
#                     ts_seg = np.zeros((max_len,))
#                     mk = np.zeros((max_len,))
#                 else:
#                     real_end = min(seg_end, traj_len)
#                     real_len = real_end - seg_start
#                     pad_len = max_len - real_len

#                     s_real = traj['observations'][seg_start:real_end]
#                     a_real = traj['actions'][seg_start:real_end]
#                     r_real = traj['rewards'][seg_start:real_end].reshape(-1, 1)
#                     if 'terminals' in traj:
#                         d_real = traj['terminals'][seg_start:real_end]
#                     else:
#                         d_real = traj['dones'][seg_start:real_end]

#                     rtg_full = discount_cumsum(traj['rewards'][seg_start:], gamma=1.)
#                     rtg_real = rtg_full[:real_len + 1].reshape(-1, 1)
#                     if rtg_real.shape[0] < real_len + 1:
#                         rtg_real = np.concatenate(
#                             [rtg_real, np.zeros((real_len + 1 - rtg_real.shape[0], 1))], axis=0)

#                     ts_real = np.arange(seg_start, seg_start + real_len)
#                     ts_real = np.clip(ts_real, 0, max_ep_len - 1)

#                     s_seg = np.concatenate([np.zeros((pad_len, state_dim)), s_real], axis=0)
#                     s_seg = (s_seg - state_mean) / state_std
#                     a_seg = np.concatenate([np.ones((pad_len, act_dim)) * -10.0, a_real], axis=0)
#                     r_seg = np.concatenate([np.zeros((pad_len, 1)), r_real], axis=0)
#                     d_seg = np.concatenate([np.ones((pad_len,)) * 2, d_real], axis=0)
#                     rtg_seg = np.concatenate([np.zeros((pad_len, 1)), rtg_real], axis=0) / scale
#                     ts_seg = np.concatenate([np.zeros((pad_len,)), ts_real], axis=0)
#                     mk = np.concatenate([np.zeros((pad_len,)), np.ones((real_len,))], axis=0)

#                 s_segs.append(s_seg); a_segs.append(a_seg); r_segs.append(r_seg)
#                 d_segs.append(d_seg); rtg_segs.append(rtg_seg); ts_segs.append(ts_seg)
#                 mask_segs.append(mk)

#             s_all.append(np.stack(s_segs, axis=0))
#             a_all.append(np.stack(a_segs, axis=0))
#             r_all.append(np.stack(r_segs, axis=0))
#             d_all.append(np.stack(d_segs, axis=0))
#             rtg_all.append(np.stack(rtg_segs, axis=0))
#             ts_all.append(np.stack(ts_segs, axis=0))
#             mask_all.append(np.stack(mask_segs, axis=0))

#             # Future targets aligned with end of LAST segment
#             fut_start = si + N * max_len
#             fs_raw = traj['observations'][fut_start:fut_start + n_pred_steps]
#             fa_raw = traj['actions'][fut_start:fut_start + n_pred_steps]
#             fr_raw_rewards = traj['rewards'][fut_start:]
#             n_avail = min(len(fs_raw), len(fa_raw), len(fr_raw_rewards))

#             if n_avail > 0:
#                 fs_f = (fs_raw[:n_avail] - state_mean) / state_std
#                 fa_f = fa_raw[:n_avail]
#                 fr_f = discount_cumsum(fr_raw_rewards, gamma=1.)[:n_avail] / scale
#                 n_avail = min(n_avail, n_pred_steps)
#                 fs_f, fa_f, fr_f = fs_f[:n_avail], fa_f[:n_avail], fr_f[:n_avail]
#             else:
#                 n_avail = 0
#                 fs_f = np.zeros((0, state_dim))
#                 fa_f = np.zeros((0, act_dim))
#                 fr_f = np.zeros((0,))

#             pad = n_pred_steps - n_avail
#             if pad > 0:
#                 fs_f = np.concatenate([fs_f, np.zeros((pad, state_dim))])
#                 fa_f = np.concatenate([fa_f, np.zeros((pad, act_dim))])
#                 fr_f = np.concatenate([fr_f, np.zeros(pad)])

#             fut_s_all.append(fs_f)
#             fut_a_all.append(fa_f)
#             fut_rtg_all.append(fr_f.reshape(-1, 1))

#         s = torch.from_numpy(np.stack(s_all, axis=0)).to(dtype=torch.float32, device=device)
#         a = torch.from_numpy(np.stack(a_all, axis=0)).to(dtype=torch.float32, device=device)
#         r = torch.from_numpy(np.stack(r_all, axis=0)).to(dtype=torch.float32, device=device)
#         d = torch.from_numpy(np.stack(d_all, axis=0)).to(dtype=torch.long, device=device)
#         rtg_t = torch.from_numpy(np.stack(rtg_all, axis=0)).to(dtype=torch.float32, device=device)
#         ts = torch.from_numpy(np.stack(ts_all, axis=0)).to(dtype=torch.long, device=device)
#         mask_t = torch.from_numpy(np.stack(mask_all, axis=0)).to(device=device)
#         fut_s_t = torch.from_numpy(np.stack(fut_s_all, axis=0)).to(dtype=torch.float32, device=device)
#         fut_a_t = torch.from_numpy(np.stack(fut_a_all, axis=0)).to(dtype=torch.float32, device=device)
#         fut_rtg_t = torch.from_numpy(np.stack(fut_rtg_all, axis=0)).to(dtype=torch.float32, device=device)

#         return s, a, r, d, rtg_t, ts, mask_t, fut_s_t, fut_a_t, fut_rtg_t

#     # Evaluation function
#     def eval_episodes(target_rew):
#         def fn(model):
#             returns_list, lengths_list = [], []
#             for _ in range(variant['num_eval_episodes']):
#                 with torch.no_grad():
#                     ret, length = evaluate_episode_rtg(
#                         env, state_dim, act_dim, model, model_type,
#                         max_ep_len=max_ep_len, scale=scale,
#                         target_return=target_rew / scale,
#                         mode=mode,
#                         state_mean=state_mean, state_std=state_std,
#                         device=device,
#                     )
#                 returns_list.append(ret)
#                 lengths_list.append(length)
#             return {
#                 f'target_{target_rew}_return_mean': np.mean(returns_list),
#                 f'target_{target_rew}_return_std': np.std(returns_list),
#                 f'target_{target_rew}_length_mean': np.mean(lengths_list),
#             }
#         return fn

#     # Create model
#     embed_dim = variant['embed_dim']

#     if model_type == 'psdt':
#         model = PSDTModel(
#             state_dim=state_dim, act_dim=act_dim,
#             hidden_size=embed_dim,
#             n_actions=None,  # continuous actions
#             cell_size=variant.get('cell_size', embed_dim),
#             n_pred_steps=n_pred_steps,
#             pred_loss_weight=variant.get('pred_loss_weight', 0.5),
#             max_length=K, max_ep_len=max_ep_len,
#             n_layer=variant['n_layer'],
#             n_head=variant['n_head'],
#             n_inner=4 * embed_dim,
#             activation_function=variant['activation_function'],
#             n_positions=1024,
#             resid_pdrop=variant['dropout'],
#             attn_pdrop=variant['dropout'],
#         )
#     else:
#         model = DecisionTransformer(
#             state_dim=state_dim, act_dim=act_dim,
#             hidden_size=embed_dim,
#             n_actions=None,
#             max_length=K, max_ep_len=max_ep_len,
#             n_layer=variant['n_layer'],
#             n_head=variant['n_head'],
#             n_inner=4 * embed_dim,
#             activation_function=variant['activation_function'],
#             n_positions=1024,
#             resid_pdrop=variant['dropout'],
#             attn_pdrop=variant['dropout'],
#         )

#     model = model.to(device=device)
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f'Total parameters: {total_params:,}')

#     optimizer = torch.optim.AdamW(
#         model.parameters(), lr=variant['learning_rate'], weight_decay=variant['weight_decay'])
#     scheduler = torch.optim.lr_scheduler.LambdaLR(
#         optimizer, lambda steps: min((steps + 1) / variant['warmup_steps'], 1))

#     # Training loop
#     eval_fns = [eval_episodes(tar) for tar in env_targets]
#     start_time = time.time()

#     for iter_num in range(variant['max_iters']):
#         model.train()
#         train_losses = []
#         action_losses = []
#         pred_losses = []
#         cell_norms = []

#         for _ in range(variant['num_steps_per_iter']):
#             if model_type == 'psdt':
#                 # Segment-threaded training: (B, N, K, ...) tensors
#                 batch = get_batch_threaded(batch_size)
#             else:
#                 batch = get_batch(batch_size)

#             if model_type == 'psdt':
#                 (states_b, actions_b, rewards_b, dones_b, rtg_b, ts_b, mask_b,
#                  fut_s_b, fut_a_b, fut_rtg_b) = batch

#                 B, N, K_seg = states_b.shape[:3]
#                 cell_state = None
#                 total_action_loss = 0.0
#                 pred_loss_final = torch.tensor(0.0, device=device)

#                 for seg_idx in range(N):
#                     s = states_b[:, seg_idx]
#                     a_seg = actions_b[:, seg_idx]
#                     r_seg = rewards_b[:, seg_idx]
#                     rtg_seg = rtg_b[:, seg_idx]        # (B, K+1, 1)
#                     ts_seg = ts_b[:, seg_idx]
#                     mask_seg = mask_b[:, seg_idx]
#                     is_last = (seg_idx == N - 1)

#                     action_target_seg = torch.clone(a_seg)

#                     _, action_preds, _, new_cell_state, seg_pred_loss = model.forward(
#                         s, a_seg, r_seg, rtg_seg[:, :-1], ts_seg,
#                         attention_mask=mask_seg,
#                         cell_state=cell_state,
#                         future_states=fut_s_b if is_last else None,
#                         future_actions=fut_a_b if is_last else None,
#                         future_rtg=fut_rtg_b if is_last else None,
#                     )

#                     act_d = action_preds.shape[2]
#                     flat_mask = mask_seg.reshape(-1) > 0
#                     if flat_mask.any():
#                         ap = action_preds.reshape(-1, act_d)[flat_mask]
#                         at = action_target_seg.reshape(-1, act_d)[flat_mask]
#                         seg_action_loss = torch.mean((ap - at) ** 2)
#                     else:
#                         seg_action_loss = torch.tensor(0.0, device=device)

#                     total_action_loss = total_action_loss + seg_action_loss
#                     if is_last:
#                         pred_loss_final = seg_pred_loss

#                     # Detach between segments: threads memory, blocks BPTT
#                     cell_state = new_cell_state.detach()

#                 # Track cell norm as diagnostic that memory is active
#                 cell_norms.append(cell_state.norm(dim=-1).mean().item())

#                 action_loss = total_action_loss / N
#                 pred_loss = pred_loss_final
#                 loss = action_loss + variant.get('pred_loss_weight', 0.5) * pred_loss
#                 pred_losses.append(pred_loss.item())
#             else:
#                 states_b, actions_b, rewards_b, dones_b, rtg_b, ts_b, mask_b = batch[:7]
#                 action_target = torch.clone(actions_b)
#                 _, action_preds, _ = model.forward(
#                     states_b, actions_b, rewards_b, rtg_b[:, :-1], ts_b,
#                     attention_mask=mask_b,
#                 )
#                 act_d = action_preds.shape[2]
#                 ap = action_preds.reshape(-1, act_d)[mask_b.reshape(-1) > 0]
#                 at = action_target.reshape(-1, act_d)[mask_b.reshape(-1) > 0]
#                 action_loss = torch.mean((ap - at) ** 2)
#                 loss = action_loss

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
#             optimizer.step()
#             scheduler.step()

#             train_losses.append(loss.item())
#             action_losses.append(action_loss.item())

#         # Evaluate
#         model.eval()
#         eval_results = {}
#         for eval_fn in eval_fns:
#             outputs = eval_fn(model)
#             for k, v in outputs.items():
#                 eval_results[k] = v

#         # Print results
#         elapsed = time.time() - start_time
#         avg_loss = np.mean(train_losses)
#         avg_act_loss = np.mean(action_losses)

#         print(f'Iter {iter_num + 1}/{variant["max_iters"]} ({elapsed:.0f}s) | '
#               f'Loss: {avg_loss:.4f} | Act loss: {avg_act_loss:.4f}', end='')
#         if pred_losses:
#             print(f' | Pred loss: {np.mean(pred_losses):.4f}', end='')
#         if cell_norms:
#             print(f' | Cell norm: {np.mean(cell_norms):.3f}', end='')

#         for target in env_targets:
#             ret_key = f'target_{target}_return_mean'
#             if ret_key in eval_results:
#                 normalized = eval_results[ret_key] / target * 100
#                 print(f' | T={target}: {eval_results[ret_key]:.1f} ({normalized:.1f}%)', end='')
#         print()

#     # Save model
#     os.makedirs('checkpoints', exist_ok=True)
#     suffix = '_fixed' if model_type == 'psdt' else ''
#     save_path = f'checkpoints/{model_type}_{env_name}_{dataset}_{mode}{suffix}.pt'
#     torch.save(model.state_dict(), save_path)
#     print(f'\nModel saved to {save_path}')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()

#     # Environment
#     parser.add_argument('--env', type=str, default='hopper',
#                         choices=['hopper', 'halfcheetah', 'walker2d'])
#     parser.add_argument('--dataset', type=str, default='medium',
#                         choices=['medium', 'medium-replay', 'medium-expert', 'expert'])
#     parser.add_argument('--mode', type=str, default='delayed',
#                         choices=['normal', 'delayed'],
#                         help='delayed = sparse reward (all reward at final step)')

#     # Model
#     parser.add_argument('--model_type', type=str, default='psdt',
#                         choices=['psdt', 'dt'])
#     parser.add_argument('--K', type=int, default=20)
#     parser.add_argument('--embed_dim', type=int, default=128)
#     parser.add_argument('--n_layer', type=int, default=3)
#     parser.add_argument('--n_head', type=int, default=1)
#     parser.add_argument('--activation_function', type=str, default='relu')
#     parser.add_argument('--dropout', type=float, default=0.1)

#     # PSDT specific
#     parser.add_argument('--cell_size', type=int, default=128)
#     parser.add_argument('--n_pred_steps', type=int, default=3)
#     parser.add_argument('--pred_loss_weight', type=float, default=0.5)
#     parser.add_argument('--n_segments', type=int, default=3,
#                         help='Number of consecutive segments per sample for PSDT segment-threaded training')

#     # Training
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--learning_rate', type=float, default=1e-4)
#     parser.add_argument('--weight_decay', type=float, default=1e-4)
#     parser.add_argument('--warmup_steps', type=int, default=10000)
#     parser.add_argument('--max_iters', type=int, default=10)
#     parser.add_argument('--num_steps_per_iter', type=int, default=10000)
#     parser.add_argument('--num_eval_episodes', type=int, default=100)
#     parser.add_argument('--pct_traj', type=float, default=1.)

#     # System
#     parser.add_argument('--device', type=str, default='cuda')

#     args = parser.parse_args()
#     experiment(vars(args))


"""
PSDT vs DT on D4RL MuJoCo (Normal + Delayed Rewards)

This experiment compares PSDT against standard DT on the D4RL MuJoCo
benchmarks, with a focus on the delayed (sparse) reward setting where
memory provides the most advantage.

In delayed mode, all intermediate rewards are zero and the total return
is given only at the final timestep. This makes the RTG signal within
a context window less informative, requiring the model to integrate
information over longer horizons — exactly what PSDT's cell state does.

Setup:
    1. Download D4RL datasets: python download_d4rl.py
    2. Run PSDT:  python mujoco_experiment.py --model_type psdt --env hopper --dataset medium --mode delayed
    3. Run DT:    python mujoco_experiment.py --model_type dt --env hopper --dataset medium --mode delayed
    4. Compare results

The code reuses the original DT codebase's GPT2 backbone and adds PSDT
components on top. Continuous actions with MSE loss — no discrete action issues.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import argparse
import pickle
import random
import os
import time

from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.psdt import PSDTModel


def discount_cumsum(x, gamma):
    ret = np.zeros_like(x)
    ret[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        ret[t] = x[t] + gamma * ret[t + 1]
    return ret


def evaluate_episode_rtg(env, state_dim, act_dim, model, model_type,
                         max_ep_len=1000, scale=1000., state_mean=0., state_std=1.,
                         device='cuda', target_return=None, mode='normal'):
    """Evaluate one episode. For PSDT, maintains cell state across steps."""
    model.eval()
    model.to(device=device)

    state_mean_t = torch.from_numpy(state_mean).to(device=device)
    state_std_t = torch.from_numpy(state_std).to(device=device)

    state, _ = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return_t = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    # PSDT: initialize cell state
    cell_state = None
    if model_type == 'psdt':
        cell_state = model.init_cell_state(batch_size=1, device=device)

    episode_return, episode_length = 0, 0

    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        if model_type == 'psdt':
            action, cell_state = model.get_action(
                (states.to(dtype=torch.float32) - state_mean_t) / state_std_t,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return_t.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                cell_state=cell_state,
            )
            cell_state = cell_state.detach()
        else:
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean_t) / state_std_t,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return_t.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )

        actions[-1] = action
        action_np = action.detach().cpu().numpy()

        state, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return_t[0, -1] - (reward / scale)
        else:
            pred_return = target_return_t[0, -1]
        target_return_t = torch.cat([target_return_t, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([
            timesteps,
            torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)
        ], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def experiment(variant):
    device = variant.get('device', 'cuda')
    env_name = variant['env']
    dataset = variant['dataset']
    model_type = variant['model_type']
    mode = variant.get('mode', 'normal')

    # Environment setup
    if env_name == 'hopper':
        env = gym.make('Hopper-v4')
        max_ep_len = 1000
        env_targets = [3600, 1800]
        scale = 1000.
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v4')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v4')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    else:
        raise NotImplementedError(f'Unknown env: {env_name}')

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Load dataset
    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    if not os.path.exists(dataset_path):
        print(f'Dataset not found at {dataset_path}')
        print(f'Run: python download_d4rl.py')
        return

    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # Apply delayed reward mode
    if mode == 'delayed':
        for path in trajectories:
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.

    states_all, traj_lens, returns = [], [], []
    for path in trajectories:
        states_all.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    states_concat = np.concatenate(states_all, axis=0)
    state_mean, state_std = np.mean(states_concat, axis=0), np.std(states_concat, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)
    K = variant['K']
    batch_size = variant['batch_size']
    n_pred_steps = variant.get('n_pred_steps', 3)

    print('=' * 60)
    print(f'D4RL MuJoCo Experiment')
    print(f'  Env: {env_name}-{dataset}')
    print(f'  Mode: {mode} {"(all rewards moved to final step)" if mode == "delayed" else ""}')
    print(f'  Model: {model_type}')
    print(f'  K: {K}, State dim: {state_dim}, Act dim: {act_dim}')
    print(f'  {len(traj_lens)} trajectories, {num_timesteps} timesteps')
    print(f'  Returns: mean={np.mean(returns):.1f}, std={np.std(returns):.1f}, '
          f'max={np.max(returns):.1f}, min={np.min(returns):.1f}')
    print('=' * 60)

    # Select top trajectories for training
    pct_traj = variant.get('pct_traj', 1.)
    num_timesteps_target = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)
    num_trajectories = 1
    timesteps_count = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps_count + traj_lens[sorted_inds[ind]] <= num_timesteps_target:
        timesteps_count += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        """Sample batch with future ground truth for PSDT prediction loss."""
        batch_inds = np.random.choice(
            np.arange(num_trajectories), size=batch_size, replace=True, p=p_sample)

        s, a, r, d, rtg, timesteps_b, mask = [], [], [], [], [], [], []
        fut_s, fut_a, fut_rtg = [], [], []

        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            traj_len = traj['rewards'].shape[0]
            si = random.randint(0, traj_len - 1)

            # Current segment
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps_b.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps_b[-1][timesteps_b[-1] >= max_ep_len] = max_ep_len - 1
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # Pad current segment
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps_b[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps_b[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

            # Future ground truth for prediction loss
            fut_start = si + max_len
            fs_raw = traj['observations'][fut_start:fut_start + n_pred_steps]
            fa_raw = traj['actions'][fut_start:fut_start + n_pred_steps]
            fr_raw_rewards = traj['rewards'][fut_start:]
            n_avail = min(len(fs_raw), len(fa_raw), len(fr_raw_rewards))

            if n_avail > 0:
                fs = (fs_raw[:n_avail] - state_mean) / state_std
                fa = fa_raw[:n_avail]
                fr = discount_cumsum(fr_raw_rewards, gamma=1.)[:n_avail] / scale
                n_avail = min(n_avail, n_pred_steps)
                fs, fa, fr = fs[:n_avail], fa[:n_avail], fr[:n_avail]
            else:
                n_avail = 0
                fs = np.zeros((0, state_dim))
                fa = np.zeros((0, act_dim))
                fr = np.zeros((0,))

            pad = n_pred_steps - n_avail
            if pad > 0:
                fs = np.concatenate([fs, np.zeros((pad, state_dim))])
                fa = np.concatenate([fa, np.zeros((pad, act_dim))])
                fr = np.concatenate([fr, np.zeros(pad)])

            fut_s.append(fs.reshape(1, n_pred_steps, state_dim))
            fut_a.append(fa.reshape(1, n_pred_steps, act_dim))
            fut_rtg.append(fr.reshape(1, n_pred_steps, 1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg_t = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        ts = torch.from_numpy(np.concatenate(timesteps_b, axis=0)).to(dtype=torch.long, device=device)
        mask_t = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        fut_s_t = torch.from_numpy(np.concatenate(fut_s, axis=0)).to(dtype=torch.float32, device=device)
        fut_a_t = torch.from_numpy(np.concatenate(fut_a, axis=0)).to(dtype=torch.float32, device=device)
        fut_rtg_t = torch.from_numpy(np.concatenate(fut_rtg, axis=0)).to(dtype=torch.float32, device=device)

        return s, a, r, d, rtg_t, ts, mask_t, fut_s_t, fut_a_t, fut_rtg_t

    # ====================================================================
    # Segment-threaded batch sampler for PSDT.
    # Returns N consecutive segments per trajectory so cell state can be
    # threaded across them during training (mirrors eval behavior).
    # Shapes: (B, N, K, ...) for segment tensors; future tensors stay
    # (B, n_pred_steps, ...) and are aligned with the END of the LAST segment.
    # ====================================================================
    n_segments = variant.get('n_segments', 3)

    def get_batch_threaded(batch_size=256, max_len=K):
        N = n_segments
        batch_inds = np.random.choice(
            np.arange(num_trajectories), size=batch_size, replace=True, p=p_sample)

        s_all, a_all, r_all, d_all = [], [], [], []
        rtg_all, ts_all, mask_all = [], [], []
        fut_s_all, fut_a_all, fut_rtg_all = [], [], []

        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            traj_len = traj['rewards'].shape[0]
            si = random.randint(0, max(0, traj_len - 1))

            s_segs, a_segs, r_segs, d_segs = [], [], [], []
            rtg_segs, ts_segs, mask_segs = [], [], []

            for seg in range(N):
                seg_start = si + seg * max_len
                seg_end = seg_start + max_len

                if seg_start >= traj_len:
                    # Fully-padded segment (past trajectory end)
                    s_seg = (np.zeros((max_len, state_dim)) - state_mean) / state_std
                    a_seg = np.ones((max_len, act_dim)) * -10.0
                    r_seg = np.zeros((max_len, 1))
                    d_seg = np.ones((max_len,)) * 2
                    rtg_seg = np.zeros((max_len + 1, 1))
                    ts_seg = np.zeros((max_len,))
                    mk = np.zeros((max_len,))
                else:
                    real_end = min(seg_end, traj_len)
                    real_len = real_end - seg_start
                    pad_len = max_len - real_len

                    s_real = traj['observations'][seg_start:real_end]
                    a_real = traj['actions'][seg_start:real_end]
                    r_real = traj['rewards'][seg_start:real_end].reshape(-1, 1)
                    if 'terminals' in traj:
                        d_real = traj['terminals'][seg_start:real_end]
                    else:
                        d_real = traj['dones'][seg_start:real_end]

                    rtg_full = discount_cumsum(traj['rewards'][seg_start:], gamma=1.)
                    rtg_real = rtg_full[:real_len + 1].reshape(-1, 1)
                    if rtg_real.shape[0] < real_len + 1:
                        rtg_real = np.concatenate(
                            [rtg_real, np.zeros((real_len + 1 - rtg_real.shape[0], 1))], axis=0)

                    ts_real = np.arange(seg_start, seg_start + real_len)
                    ts_real = np.clip(ts_real, 0, max_ep_len - 1)

                    s_seg = np.concatenate([np.zeros((pad_len, state_dim)), s_real], axis=0)
                    s_seg = (s_seg - state_mean) / state_std
                    a_seg = np.concatenate([np.ones((pad_len, act_dim)) * -10.0, a_real], axis=0)
                    r_seg = np.concatenate([np.zeros((pad_len, 1)), r_real], axis=0)
                    d_seg = np.concatenate([np.ones((pad_len,)) * 2, d_real], axis=0)
                    rtg_seg = np.concatenate([np.zeros((pad_len, 1)), rtg_real], axis=0) / scale
                    ts_seg = np.concatenate([np.zeros((pad_len,)), ts_real], axis=0)
                    mk = np.concatenate([np.zeros((pad_len,)), np.ones((real_len,))], axis=0)

                s_segs.append(s_seg); a_segs.append(a_seg); r_segs.append(r_seg)
                d_segs.append(d_seg); rtg_segs.append(rtg_seg); ts_segs.append(ts_seg)
                mask_segs.append(mk)

            s_all.append(np.stack(s_segs, axis=0))
            a_all.append(np.stack(a_segs, axis=0))
            r_all.append(np.stack(r_segs, axis=0))
            d_all.append(np.stack(d_segs, axis=0))
            rtg_all.append(np.stack(rtg_segs, axis=0))
            ts_all.append(np.stack(ts_segs, axis=0))
            mask_all.append(np.stack(mask_segs, axis=0))

            # Future targets aligned with end of LAST segment
            fut_start = si + N * max_len
            fs_raw = traj['observations'][fut_start:fut_start + n_pred_steps]
            fa_raw = traj['actions'][fut_start:fut_start + n_pred_steps]
            fr_raw_rewards = traj['rewards'][fut_start:]
            n_avail = min(len(fs_raw), len(fa_raw), len(fr_raw_rewards))

            if n_avail > 0:
                fs_f = (fs_raw[:n_avail] - state_mean) / state_std
                fa_f = fa_raw[:n_avail]
                fr_f = discount_cumsum(fr_raw_rewards, gamma=1.)[:n_avail] / scale
                n_avail = min(n_avail, n_pred_steps)
                fs_f, fa_f, fr_f = fs_f[:n_avail], fa_f[:n_avail], fr_f[:n_avail]
            else:
                n_avail = 0
                fs_f = np.zeros((0, state_dim))
                fa_f = np.zeros((0, act_dim))
                fr_f = np.zeros((0,))

            pad = n_pred_steps - n_avail
            if pad > 0:
                fs_f = np.concatenate([fs_f, np.zeros((pad, state_dim))])
                fa_f = np.concatenate([fa_f, np.zeros((pad, act_dim))])
                fr_f = np.concatenate([fr_f, np.zeros(pad)])

            fut_s_all.append(fs_f)
            fut_a_all.append(fa_f)
            fut_rtg_all.append(fr_f.reshape(-1, 1))

        s = torch.from_numpy(np.stack(s_all, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.stack(a_all, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.stack(r_all, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.stack(d_all, axis=0)).to(dtype=torch.long, device=device)
        rtg_t = torch.from_numpy(np.stack(rtg_all, axis=0)).to(dtype=torch.float32, device=device)
        ts = torch.from_numpy(np.stack(ts_all, axis=0)).to(dtype=torch.long, device=device)
        mask_t = torch.from_numpy(np.stack(mask_all, axis=0)).to(device=device)
        fut_s_t = torch.from_numpy(np.stack(fut_s_all, axis=0)).to(dtype=torch.float32, device=device)
        fut_a_t = torch.from_numpy(np.stack(fut_a_all, axis=0)).to(dtype=torch.float32, device=device)
        fut_rtg_t = torch.from_numpy(np.stack(fut_rtg_all, axis=0)).to(dtype=torch.float32, device=device)

        return s, a, r, d, rtg_t, ts, mask_t, fut_s_t, fut_a_t, fut_rtg_t

    # Evaluation function
    def eval_episodes(target_rew):
        def fn(model):
            returns_list, lengths_list = [], []
            for _ in range(variant['num_eval_episodes']):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        env, state_dim, act_dim, model, model_type,
                        max_ep_len=max_ep_len, scale=scale,
                        target_return=target_rew / scale,
                        mode=mode,
                        state_mean=state_mean, state_std=state_std,
                        device=device,
                    )
                returns_list.append(ret)
                lengths_list.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns_list),
                f'target_{target_rew}_return_std': np.std(returns_list),
                f'target_{target_rew}_length_mean': np.mean(lengths_list),
            }
        return fn

    # Create model
    embed_dim = variant['embed_dim']

    if model_type == 'psdt':
        model = PSDTModel(
            state_dim=state_dim, act_dim=act_dim,
            hidden_size=embed_dim,
            n_actions=None,  # continuous actions
            cell_size=variant.get('cell_size', embed_dim),
            n_pred_steps=n_pred_steps,
            pred_loss_weight=variant.get('pred_loss_weight', 0.5),
            max_length=K, max_ep_len=max_ep_len,
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4 * embed_dim,
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    else:
        model = DecisionTransformer(
            state_dim=state_dim, act_dim=act_dim,
            hidden_size=embed_dim,
            n_actions=None,
            max_length=K, max_ep_len=max_ep_len,
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4 * embed_dim,
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )

    model = model.to(device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=variant['learning_rate'], weight_decay=variant['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / variant['warmup_steps'], 1))

    # Training loop
    eval_fns = [eval_episodes(tar) for tar in env_targets]
    start_time = time.time()

    for iter_num in range(variant['max_iters']):
        model.train()
        train_losses = []
        action_losses = []
        pred_losses = []
        cell_norms = []

        for _ in range(variant['num_steps_per_iter']):
            if model_type == 'psdt':
                # Segment-threaded training: (B, N, K, ...) tensors
                batch = get_batch_threaded(batch_size)
            else:
                batch = get_batch(batch_size)

            if model_type == 'psdt':
                (states_b, actions_b, rewards_b, dones_b, rtg_b, ts_b, mask_b,
                 fut_s_b, fut_a_b, fut_rtg_b) = batch

                B, N, K_seg = states_b.shape[:3]
                cell_state = None
                total_action_loss = 0.0
                pred_loss_final = torch.tensor(0.0, device=device)

                for seg_idx in range(N):
                    s = states_b[:, seg_idx]
                    a_seg = actions_b[:, seg_idx]
                    r_seg = rewards_b[:, seg_idx]
                    rtg_seg = rtg_b[:, seg_idx]        # (B, K+1, 1)
                    ts_seg = ts_b[:, seg_idx]
                    mask_seg = mask_b[:, seg_idx]
                    is_last = (seg_idx == N - 1)

                    action_target_seg = torch.clone(a_seg)

                    _, action_preds, _, new_cell_state, seg_pred_loss = model.forward(
                        s, a_seg, r_seg, rtg_seg[:, :-1], ts_seg,
                        attention_mask=mask_seg,
                        cell_state=cell_state,
                        future_states=fut_s_b if is_last else None,
                        future_actions=fut_a_b if is_last else None,
                        future_rtg=fut_rtg_b if is_last else None,
                    )

                    act_d = action_preds.shape[2]
                    flat_mask = mask_seg.reshape(-1) > 0
                    if flat_mask.any():
                        ap = action_preds.reshape(-1, act_d)[flat_mask]
                        at = action_target_seg.reshape(-1, act_d)[flat_mask]
                        seg_action_loss = torch.mean((ap - at) ** 2)
                    else:
                        seg_action_loss = torch.tensor(0.0, device=device)

                    total_action_loss = total_action_loss + seg_action_loss
                    if is_last:
                        pred_loss_final = seg_pred_loss

                    # Detach between segments: threads memory, blocks BPTT
                    cell_state = new_cell_state.detach()

                # Track cell norm as diagnostic that memory is active
                cell_norms.append(cell_state.norm(dim=-1).mean().item())

                action_loss = total_action_loss / N
                pred_loss = pred_loss_final
                loss = action_loss + variant.get('pred_loss_weight', 0.5) * pred_loss
                pred_losses.append(pred_loss.item())
            else:
                states_b, actions_b, rewards_b, dones_b, rtg_b, ts_b, mask_b = batch[:7]
                action_target = torch.clone(actions_b)
                _, action_preds, _ = model.forward(
                    states_b, actions_b, rewards_b, rtg_b[:, :-1], ts_b,
                    attention_mask=mask_b,
                )
                act_d = action_preds.shape[2]
                ap = action_preds.reshape(-1, act_d)[mask_b.reshape(-1) > 0]
                at = action_target.reshape(-1, act_d)[mask_b.reshape(-1) > 0]
                action_loss = torch.mean((ap - at) ** 2)
                loss = action_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            action_losses.append(action_loss.item())

        # Evaluate
        model.eval()
        eval_results = {}
        for eval_fn in eval_fns:
            outputs = eval_fn(model)
            for k, v in outputs.items():
                eval_results[k] = v

        # Print results
        elapsed = time.time() - start_time
        avg_loss = np.mean(train_losses)
        avg_act_loss = np.mean(action_losses)

        print(f'Iter {iter_num + 1}/{variant["max_iters"]} ({elapsed:.0f}s) | '
              f'Loss: {avg_loss:.4f} | Act loss: {avg_act_loss:.4f}', end='')
        if pred_losses:
            print(f' | Pred loss: {np.mean(pred_losses):.4f}', end='')
        if cell_norms:
            print(f' | Cell norm: {np.mean(cell_norms):.3f}', end='')

        for target in env_targets:
            ret_key = f'target_{target}_return_mean'
            if ret_key in eval_results:
                normalized = eval_results[ret_key] / target * 100
                print(f' | T={target}: {eval_results[ret_key]:.1f} ({normalized:.1f}%)', end='')
        print()

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    suffix = '_fixed' if model_type == 'psdt' else ''
    save_path = f'checkpoints/{model_type}_{env_name}_{dataset}_{mode}{suffix}.pt'
    torch.save(model.state_dict(), save_path)
    print(f'\nModel saved to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--env', type=str, default='hopper',
                        choices=['hopper', 'halfcheetah', 'walker2d'])
    parser.add_argument('--dataset', type=str, default='medium',
                        choices=['medium', 'medium-replay', 'medium-expert', 'expert'])
    parser.add_argument('--mode', type=str, default='delayed',
                        choices=['normal', 'delayed'],
                        help='delayed = sparse reward (all reward at final step)')

    # Model
    parser.add_argument('--model_type', type=str, default='psdt',
                        choices=['psdt', 'dt'])
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)

    # PSDT specific
    parser.add_argument('--cell_size', type=int, default=128)
    parser.add_argument('--n_pred_steps', type=int, default=3)
    parser.add_argument('--pred_loss_weight', type=float, default=0.5)
    parser.add_argument('--n_segments', type=int, default=3,
                        help='Number of consecutive segments per sample for PSDT segment-threaded training')

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--pct_traj', type=float, default=1.)

    # System
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    experiment(vars(args))