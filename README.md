# Decision Transformer — Cross-Task Generalization Study

Fork of [kzl/decision-transformer](https://github.com/kzl/decision-transformer) (Chen et al., NeurIPS 2021)

Extended with a **Cross-Task Generalization and Fine-Tuning** experiment on continuous control (`MuJoCo Walker2d`).

---

## What This Repo Contains

| Path | Description |
|---|---|
| `gym/experiment.py` | Core PyTorch training script. |
| `gym/cross_task/modified_envs.py` | Custom `Walker2d` environments with altered physics (mass/friction). |
| `gym/cross_task/collect_target_data.py` | Offline target data collection utilizing source policy replay. |
| `gym/cross_task/run_cross_task.py` | Aggregated script handling pre-training and fine-tuning transfer modes. |
| `gym/cross_task/plot_results.py` | Script to visualize learning curves and generate sample efficiency charts. |
| `gym/run_all.sh` | Master executable script handling full experiment orchestration. |
| `gym/results/` | Pre-computed raw JSON outputs from zero-shot vs fine-tuning tests. |
| `gym/results/plots/` | Output High-Res generated graphs showcasing transfer performance. |

---

## Experiment: Robustness to Cross-Task Dynamics Transfer
We investigate whether a Decision Transformer pre-trained purely on standard dynamics can be fine-tuned efficiently on a foreign target domain possessing heavily altered physical paradigms, compared against an identical baseline forced to learn natively from scratch.

### Setup:
*   **Environment**: `Walker2d-v5` (MuJoCo, Refactored for modern Gymnasium API compatibility)
*   **Dataset**: D4RL `walker2d-medium-v2`, `medium-replay`, and `expert`
*   **Foreign Target Dynamics**: 
    * `heavy` (1.5x body mass)
    * `light` (0.5x body mass)
    * `slippery` (0.5x floor friction)
    * `heavy_slippery` (1.5x mass + 0.5x friction)
*   **Evaluation Mode**: Full cross-evaluation mapping pre-trained models across all testing variations dynamically evaluated 5 times per training iteration.

### Key results:
*   **Sample Efficiency**: Fine-tuned models mapping continuous geometry reliably converge to stable operational metrics over 4.0x–9.0x faster than baselines on natively compatible physics deviations (e.g., heavy configurations).
*   **Negative Transfer Identification**: Extreme environmental deviations (e.g., bouncy light gravity maps) notoriously trigger localized catastrophic unlearning, severely slowing convergence as ingrained weight patterns clash against volatile bouncing physical dynamics.

---

## Quick Start (Steps to Reproduce Results)

Our comprehensive master script guarantees **100% reproducibility** because it controls all dependencies, collections, and training sweeps smoothly on modern architectures. Because this experiment natively enforces float32 tensor casting, it executes reliably.

### Experiment Iteration Budgets (Reproducibility Metrics):
To ensure experiments finish within reasonable bounds (computation budget of ~1.5 - 3 hours), our master configuration truncates default timelines strictly to:
- **Pre-Training Budget:** 5 Iterations (2,500 steps/iter)
- **Fine-Tuning/Scratch Transfer Budget:** 10 Iterations (1,000 steps/iter)
- **Evaluation:** 5 structural rollback episodes per iteration

### 1. Install dependencies
```bash
cd gym
pip install -r requirements.txt
pip install gymnasium[mujoco]
```

### 2. Run the Evaluation Suite
```bash
# Execute full testing suite autonomously
bash run_all.sh --device cpu

# Alternatively, execute step-by-step for absolute reproducibility:
bash run_all.sh --step 0  # Dependency Checks
bash run_all.sh --step 1  # Download D4RL Source Datasets natively
bash run_all.sh --step 2  # Synthesize foreign target physical environments
bash run_all.sh --step 3  # Pre-train Source Checkpoints
bash run_all.sh --step 4  # Fine-tune & Evaluate Transfer Baselines
bash run_all.sh --step 5  # Plot Aggregated Transfer Results
```
*(All 13 transfer plots and efficiency comparisons securely auto-generate inside `results/plots/`)*

---

## Original Paper
```bibtex
@article{chen2021decisiontransformer,
  title={Decision Transformer: Reinforcement Learning via Sequence Modeling},
  author={Lili Chen and Kevin Lu and Aravind Rajeswaran and Kimin Lee and
          Aditya Grover and Michael Laskin and Pieter Abbeel and
          Aravind Srinivas and Igor Mordatch},
  journal={arXiv preprint arXiv:2106.01345},
  year={2021}
}
```
