# Decision Transformer — Noisy Reward Robustness Study

> **Fork of** [kzl/decision-transformer](https://github.com/kzl/decision-transformer) (Chen et al., NeurIPS 2021)  
> Extended with a **reward-noise robustness experiment** on continuous control (MuJoCo Hopper).

---

## What This Repo Contains

| Path | Description |
|---|---|
| `gym/experiment.py` | Core training script (extended with `--reward_noise_std`) |
| `gym/run_noise_sweep.py` | Sweeps over noise levels, saves per-run JSON results |
| `gym/plot_results.py` | Aggregates JSONs → plot |
| `gym/results/` | Pre-computed results (DT + BC × 5 noise levels) |
| `gym/figures/` | Output plots |
| `report.tex` | LaTeX report covering the full experiment |

---

## Experiment: Robustness to Reward Annotation Noise

We investigate how well the Decision Transformer handles **corrupted reward labels** in the offline dataset, compared to an MLP Behaviour Cloning baseline.

**Setup:**
- Environment: `Hopper-v5` (MuJoCo, continuous 3-DOF control)
- Dataset: D4RL `hopper-medium-v2`
- Noise model: $\tilde{r}_t = r_t + \mathcal{N}(0, \sigma^2)$ added to training rewards only
- Noise levels: `σ ∈ {0.0, 0.5, 1.0, 2.0, 5.0}`
- Evaluation: noiseless true environment, 100 rollouts per target

**Key results:**
- **DT** retains >90% of clean performance at all noise levels (non-monotonic, near-flat curve)
- **MLP-BC** collapses at σ=0.5 (return drops to 378 from 903 clean) before recovering at higher noise

---

## Quick Start

### 1. Install dependencies
```bash
cd gym
pip install -r requirements.txt   # or use conda_env.yml
```

### 2. Download dataset
```bash
python data/fetch_datasets.py
```

### 3. Run the noise sweep
```bash
# Decision Transformer
python run_noise_sweep.py --env hopper --dataset medium --model_type dt \
    --noise_levels 0.0 0.5 1.0 2.0 5.0 --max_iters 10 --num_steps_per_iter 10000

# MLP-BC baseline
python run_noise_sweep.py --env hopper --dataset medium --model_type bc \
    --noise_levels 0.0 0.5 1.0 2.0 5.0 --max_iters 10 --num_steps_per_iter 10000
```

> Results are saved to `results/` automatically. Runs are resumable — existing JSONs are skipped.

### 4. Plot results
```bash
python plot_results.py --env hopper --dataset medium
# → figures/noise_robustness_hopper-medium.png
```

### 5. Compile report
```bash
cd ..
pdflatex report.tex
```

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

## License

MIT
