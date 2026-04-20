"""
PSDT Sequence Trainer

Modified trainer that:
1. Samples segments with extra future steps for prediction targets
2. Manages cell state across segments within a trajectory
3. Computes combined action + prediction loss
"""

import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class PSDTSequenceTrainer(Trainer):
    """
    Trainer for Predictive State Decision Transformer.

    Compared to the standard SequenceTrainer, this:
    - Passes future ground truth data for the prediction loss
    - Manages cell state across training (detached, no BPTT)
    - Logs both action loss and prediction loss separately
    """

    def __init__(self, n_pred_steps=3, pred_loss_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.n_pred_steps = n_pred_steps
        self.pred_loss_weight = pred_loss_weight

    def train_step(self):
        # get_batch now returns extra future data for prediction targets
        batch = self.get_batch(self.batch_size)
        states, actions, rewards, dones, rtg, timesteps, attention_mask = batch[:7]

        # Future ground truth (may be None if get_batch doesn't provide them)
        future_states = batch[7] if len(batch) > 7 else None
        future_actions = batch[8] if len(batch) > 8 else None
        future_rtg = batch[9] if len(batch) > 9 else None

        action_target = torch.clone(actions)

        # Forward pass with no cell state (each batch sample is independent)
        # Cell state starts at zero for each sampled segment
        # This is the simplest training mode - no cross-segment BPTT needed
        state_preds, action_preds, reward_preds, new_cell_state, pred_loss = \
            self.model.forward(
                states, actions, rewards, rtg[:, :-1], timesteps,
                attention_mask=attention_mask,
                cell_state=None,  # start fresh each batch (no BPTT)
                future_states=future_states,
                future_actions=future_actions,
                future_rtg=future_rtg,
            )

        # Action loss (same as standard DT)
        act_dim = action_preds.shape[2]
        action_preds_masked = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target_masked = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        action_loss = torch.mean((action_preds_masked - action_target_masked) ** 2)

        # Combined loss
        total_loss = action_loss + self.pred_loss_weight * pred_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean(
                (action_preds_masked - action_target_masked) ** 2
            ).detach().cpu().item()
            self.diagnostics['training/pred_loss'] = pred_loss.detach().cpu().item()
            self.diagnostics['training/action_loss'] = action_loss.detach().cpu().item()

        return total_loss.detach().cpu().item()
