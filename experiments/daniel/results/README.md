# experiments/daniel/results/

This directory contains all output files produced by Daniel's M1 iMac training
and evaluation runs for the ANA 699 Robotics Capstone. Files are organized by
the notebook section that produced them.

---

## Normalization Statistics
Required for inference — these files must be present before running any
evaluation or video render notebook.

| File | Description |
|------|-------------|
| `state_mean.npy` | Per-dimension mean across all 2,000,000 training timesteps |
| `state_std.npy` | Per-dimension std across all 2,000,000 training timesteps |

---

## EDA Figures
Produced by `daniel_EDA_DT.ipynb`. Exploratory analysis of the combined
Minari medium-expert dataset prior to training.

| File | Description |
|------|-------------|
| `action_distribution.png` | Distribution of actions across all 6 joint dimensions |
| `correlation_heatmap.png` | Correlation matrix across state dimensions |
| `episode_lengths.png` | Episode length distribution (all 1,000 steps) |
| `observation_heatmap.png` | Observation density across state dimensions |
| `reward_distribution.png` | Reward distribution across medium and expert splits |
| `rtg_distribution.png` | Return-to-go distribution across the combined dataset |
| `loss_curves_comparison.png` | Early training loss curve comparisons |
| `training_loss.png` | Training loss from initial EDA training run |

---

## Training Run Outputs
Produced by `daniel_DT_M1_baseline_pipeline.ipynb` — the primary training
notebook. 100,000 gradient steps on MPS, batch size 64, K=20, cosine LR.

| File | Description |
|------|-------------|
| `hw_quickval_M1-iMac.csv` | Hardware metrics from 500-iteration quick validation run |
| `hw_quickval_M1-iMac.png` | BPS / loss / grad norm plots from quick validation |
| `hw_full_M1-iMac.csv` | Hardware tracker CSV — 100 checkpoint records covering BPS, loss, RAM, GPU memory across 100,000 iterations. Primary input for Section 5.4 cross-platform comparison |
| `training_history_M1-iMac.csv` | Per-checkpoint loss, BPS, and gradient norm history |
| `training_curves_M1-iMac.png` | Training loss (log scale), BPS, and gradient norm plots |
| `bps_history_M1-iMac.csv` | Batches-per-second history across training |
| `bps_curve_M1-iMac.png` | BPS throughput curve |
| `checkpoint_loss_M1-iMac.csv` | Loss at each 10,000-iteration checkpoint |
| `checkpoint_loss_M1-iMac.png` | Checkpoint loss curve |
| `loss_distribution_M1-iMac.png` | Distribution of per-batch MSE loss values |
| `pre_norm_state_stats_M1-iMac.png` | Raw state statistics before normalization |
| `reward_distributions_M1-iMac.png` | Reward distributions across medium and expert splits |

---

## Evaluation Outputs
Produced by the evaluation section of `daniel_DT_M1_baseline_pipeline.ipynb`.
50-episode MuJoCo rollout conditioned on target RTG = 16,585.

| File | Description |
|------|-------------|
| `eval_summary_M1-iMac.csv` | Headline evaluation results — mean/std raw return, D4RL normalized score, success gap across 50 episodes |
| `eval_figures_M1-iMac.png` | Figures 7a (return distribution), 7b (normalized return vs benchmarks), 7c (per-episode success gap), 8 (RTG profile during rollout) |
| `rtg_sweep_M1-iMac.csv` | RTG sweep results across 5 candidates (6,000 / 9,000 / 12,000 / 14,000 / 16,585), 10 episodes each |
| `rtg_sweep_M1-iMac.png` | RTG sweep figure |

---

## Video Renders
Produced by `daniel DT Half Cheetah Video Run.ipynb`. MuJoCo HalfCheetah-v4
policy rollouts recorded as MP4, target RTG = 16,585.

| File | Description |
|------|-------------|
| `videos_M1-iMac/halfcheetah_ep1_rtg16585_M1-iMac.mp4` | Episode 1 policy rollout |
| `videos_M1-iMac/halfcheetah_ep2_rtg16585_M1-iMac.mp4` | Episode 2 policy rollout |
| `videos_M1-iMac/halfcheetah_ep3_rtg16585_M1-iMac.mp4` | Episode 3 policy rollout |
| `videos_M1-iMac/halfcheetah_highlight_reel_M1-iMac.mp4` | Highlight reel — best moments across all evaluated episodes |

---

## Headline Result

**D4RL Normalized Score: 139.1 ± 1.0** across 50 evaluation episodes
Training time: ~58 minutes | Hardware: M1 iMac, 8.6 GB unified memory
Cumulative Interaction Cost (CIC): **0 environment steps**
Chen et al. (2021) benchmark: ~86.8 | Delta: **+52.3 normalized points**
