"""
Predictive State Decision Transformer (PSDT) - With Discrete Action Support

Changes from original:
  - Added n_actions parameter for discrete action spaces
  - Action head outputs n_actions logits when discrete
  - get_action() returns argmax of logits when discrete
"""

import numpy as np
import torch
import torch.nn as nn
import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


class GateMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def __call__(self, x):
        return self.net(x)


class PredictionHead(nn.Module):
    def __init__(self, input_dim, state_dim, act_dim, hidden_dim=None):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        output_dim = state_dim + act_dim + 1
        if hidden_dim is None:
            hidden_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        out = self.net(x)
        state_pred = out[..., :self.state_dim]
        action_pred = out[..., self.state_dim:self.state_dim + self.act_dim]
        rtg_pred = out[..., -1:]
        return state_pred, action_pred, rtg_pred


class PSDTModel(TrajectoryModel):

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            n_actions=None,
            cell_size=None,
            n_pred_steps=3,
            pred_loss_weight=0.5,
            gate_hidden_dim=None,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.cell_size = cell_size if cell_size is not None else hidden_size
        self.n_pred_steps = n_pred_steps
        self.pred_loss_weight = pred_loss_weight
        self.n_actions = n_actions

        if gate_hidden_dim is None:
            gate_hidden_dim = self.cell_size

        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            **kwargs
        )
        if not hasattr(config, 'n_ctx'):
            config.n_ctx = config.n_positions

        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(self.state_dim, hidden_size)
        self.embed_action = nn.Linear(self.act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        self.predict_return = nn.Linear(hidden_size, 1)

        if self.n_actions is not None:
            self.predict_action = nn.Linear(hidden_size, self.n_actions)
        else:
            self.predict_action = nn.Sequential(
                *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            )

        gate_input_dim = hidden_size + self.cell_size
        self.forget_gate = GateMLP(gate_input_dim, self.cell_size, gate_hidden_dim)
        self.input_gate = GateMLP(gate_input_dim, self.cell_size, gate_hidden_dim)
        self.output_gate = GateMLP(gate_input_dim, self.cell_size, gate_hidden_dim)
        self.cell_candidate = GateMLP(gate_input_dim, self.cell_size, gate_hidden_dim)
        self.memory_inject = nn.Linear(self.cell_size, hidden_size)

        pred_input_dim = hidden_size + self.cell_size
        self.pred_heads = nn.ModuleList([
            PredictionHead(pred_input_dim, state_dim, act_dim, hidden_dim=gate_hidden_dim)
            for _ in range(n_pred_steps)
        ])

        self._init_forget_gate_bias(bias_value=1.0)

    def _init_forget_gate_bias(self, bias_value=1.0):
        last_layer = self.forget_gate.net[-1]
        nn.init.constant_(last_layer.bias, bias_value)

    def init_cell_state(self, batch_size, device):
        return torch.zeros(batch_size, self.cell_size, device=device)

    def forward(self, states, actions, rewards, returns_to_go, timesteps,
                attention_mask=None, cell_state=None,
                future_states=None, future_actions=None, future_rtg=None):

        batch_size, seq_length = states.shape[0], states.shape[1]
        device = states.device

        if cell_state is None:
            cell_state = self.init_cell_state(batch_size, device)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)

        h_mem = self.memory_inject(torch.tanh(cell_state))

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)

        stacked_inputs = stacked_inputs + h_mem.unsqueeze(1)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])
        action_preds = self.predict_action(x[:, 1])

        z_pool = x[:, 2, -1]

        gate_input = torch.cat([z_pool, cell_state], dim=-1)
        f = torch.sigmoid(self.forget_gate(gate_input))
        i = torch.sigmoid(self.input_gate(gate_input))
        c_candidate = torch.tanh(self.cell_candidate(gate_input))
        new_cell_state = f * cell_state + i * c_candidate

        ogate_input = torch.cat([z_pool, new_cell_state], dim=-1)
        o = torch.sigmoid(self.output_gate(ogate_input))

        pred_input = torch.cat([z_pool, new_cell_state], dim=-1)
        pred_loss = torch.tensor(0.0, device=device)

        if future_states is not None and future_actions is not None and future_rtg is not None:
            for j, head in enumerate(self.pred_heads):
                if j < future_states.shape[1]:
                    s_pred, a_pred, r_pred = head(pred_input)
                    pred_loss = pred_loss + (
                        torch.mean((s_pred - future_states[:, j]) ** 2) +
                        torch.mean((a_pred - future_actions[:, j]) ** 2) +
                        torch.mean((r_pred - future_rtg[:, j]) ** 2)
                    )
            pred_loss = pred_loss / max(1, min(self.n_pred_steps, future_states.shape[1]))

        return state_preds, action_preds, return_preds, new_cell_state, pred_loss

    def get_action(self, states, actions, rewards, returns_to_go, timesteps,
                   cell_state=None, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            attention_mask = torch.cat([
                torch.zeros(self.max_length - states.shape[1]),
                torch.ones(states.shape[1])
            ]).to(dtype=torch.long, device=states.device).reshape(1, -1)

            states = torch.cat([
                torch.zeros((1, self.max_length - states.shape[1], self.state_dim), device=states.device),
                states], dim=1).to(dtype=torch.float32)
            actions = torch.cat([
                torch.zeros((1, self.max_length - actions.shape[1], self.act_dim), device=actions.device),
                actions], dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat([
                torch.zeros((1, self.max_length - returns_to_go.shape[1], 1), device=returns_to_go.device),
                returns_to_go], dim=1).to(dtype=torch.float32)
            timesteps = torch.cat([
                torch.zeros((1, self.max_length - timesteps.shape[1]), device=timesteps.device),
                timesteps], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, _, new_cell_state, _ = self.forward(
            states, actions, None, returns_to_go, timesteps,
            attention_mask=attention_mask, cell_state=cell_state,
        )

        if self.n_actions is not None:
            action = torch.argmax(action_preds[0, -1]).unsqueeze(0).float()
        else:
            action = action_preds[0, -1]

        return action, new_cell_state
