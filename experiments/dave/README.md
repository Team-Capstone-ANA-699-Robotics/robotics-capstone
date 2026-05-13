# Eliminating the High-Risk Interaction Cost of Autonomous Robotics Through Offline Sequence Modeling: A Decision Transformer Approach

**Darwin Juan · Dan Kast · David Terando**  
Master of Data Science — National University · ANA699 Capstone · Spring 2026

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA%20%7C%20CPU-orange.svg)](https://pytorch.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-HalfCheetah--v4-green.svg)](https://mujoco.org/)
[![uv](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://github.com/astral-sh/uv)

---

## Abstract

Online reinforcement learning algorithms require 800,000–1,200,000 live environment interactions to reach proficiency on standard locomotion benchmarks, posing a safety and economic barrier to autonomous robotics training on physical hardware. This study investigated whether offline sequence modeling via the Decision Transformer can eliminate this high-risk Cumulative Interaction Cost (CIC) while producing a competent baseline policy. A 727,558-parameter Decision Transformer was trained on the Minari HalfCheetah Medium-Expert dataset across three hardware platforms — Apple M5 Max, Apple M1 iMac, and NVIDIA cloud GPU — and evaluated in the MuJoCo physics simulator. The production configuration achieved a D4RL normalized score of 139.33 ± 0.36 at CIC = 0, with zero catastrophic failures across 150 evaluation episodes and cross-platform mean returns agreeing within 0.27 normalized points. A six-component cost model projected per-run savings of $182–$1,260. The findings establish that offline sequence modeling delivers competent, deterministic policies on consumer-grade hardware without incurring online environment-interaction costs.

---

## Repository Structure

```
robotics-capstone/
│
├── README.md
│
├── training_handcoded_K30_bs256_primary.ipynb   # PRIMARY production notebook (M5 Max)
├── training_handcoded_K20_bs64.ipynb            # Baseline handcoded notebook
├── eda_dataset_comparison_restructured.ipynb    # Foundational EDA notebook
├── evaluation_rtg_sweep_primary.ipynb           # RTG sweep evaluation
├── validation_diagnostic.ipynb                  # Generalization diagnostic (train vs. val loss)
│
├── notebooks_archive/                           # Historical development runs
│   ├── hf_progression/                          # HuggingFace backbone progression
│   └── hf_ablations/                            # Architecture ablation studies
│
├── checkpoints/                                 # Trained model weights (.pt)
├── checkpoints_valdiag/                         # Validation diagnostic checkpoint
├── results/                                     # CSVs and evaluation artifacts
├── videos/                                      # MuJoCo rollout recordings
├── data/                                        # Dataset (not tracked by git — see below)
└── docs/
    ├── chapters/                                # Paper chapters
    └── references/                              # Master reference list
```

---

## Key Results

| Platform | Backbone | K | bs | n | D4RL Norm. Score | CIC |
|---|---|---|---|---|---|---|
| Apple M5 Max | Hand-coded GPT | 30 | 256 | 50 | **139.33 ± 0.36** | 0 |
| Apple M1 iMac | Hand-coded GPT | 20 | 64 | 50 | **139.06 ± 0.96** | 0 |
| NVIDIA RTX 6000 | Hand-coded GPT | 30 | 256 | 50 | **139.33 ± 2.28** | 0 |

- **Competency threshold** (D4RL ≥ 70): cleared by 69 normalized points
- **Catastrophic failures**: 0 of 150 episodes across all platforms
- **Projected fiscal savings**: $182–$1,260 per run vs. online RL baselines (PPO/SAC)
- **Cross-platform portability**: M5 Max and RTX 6000 mean returns agree to within 0.16 raw return units under identical configuration

---

## Novel Contribution: Cumulative Interaction Cost (CIC)

This study introduces the **Cumulative Interaction Cost (CIC)** metric:

```
CIC = Σ(e=1 to E) L_e · 𝟙(training_active)
```

Where E = number of live episodes, L_e = episode length, and 𝟙 is an indicator function equal to 1 when the agent is training via live interaction. Online RL baselines (PPO, SAC) incur CIC of 800,000–1,200,000 steps. The Decision Transformer achieves **CIC = 0** by training exclusively on static offline data.

---

## Notebooks

### Root-Level (Primary Deliverables)

| Notebook | Description |
|---|---|
| `training_handcoded_K30_bs256_primary.ipynb` | Primary production run: hand-coded GPT backbone, K=30, bs=256, cosine LR, 100k iterations on M5 Max. Produces headline result. |
| `training_handcoded_K20_bs64.ipynb` | Baseline handcoded run: K=20, bs=64. Used for cross-platform comparison baseline. |
| `eda_dataset_comparison_restructured.ipynb` | Full EDA comparing D4RL vs. Minari datasets; derives target RTG=16,585 from expert sub-split. |
| `evaluation_rtg_sweep_primary.ipynb` | RTG sweep evaluation across conditioning range; characterizes policy response to target RTG. |
| `validation_diagnostic.ipynb` | Generalization diagnostic: trains on 90/10 episode-stratified split with held-out validation loss tracked every 2,000 iterations over 100k steps. Confirms absence of overfitting (final val/train loss ratio = 1.14). |

### Archived Notebooks (`notebooks_archive/`)

| Notebook | Description |
|---|---|
| `hf_progression/training_hf_m5_standard.ipynb` | Early HuggingFace GPT-2 backbone, standard config |
| `hf_progression/training_hf_m5_samplevol.ipynb` | Sample volume exploration |
| `hf_progression/training_hf_m5_optimized.ipynb` | HF optimized LR run |
| `hf_ablations/ablation_hf_wpe_zeroed.ipynb` | WPE embedding zeroed |
| `hf_ablations/ablation_hf_uniform_init.ipynb` | Uniform weight initialization |
| `hf_ablations/ablation_hf_gelu.ipynb` | GELU activation |
| `hf_ablations/ablation_hf_gelu_uniform_init.ipynb` | GELU + uniform init combined |
| `hf_ablations/ablation_hf_cosine_lr.ipynb` | Cosine LR schedule |
| `hf_ablations/ablation_hf_constant_lr_baseline.ipynb` | Constant LR baseline |
| `hf_ablations/ablation_hand_coded_reference.ipynb` | Hand-coded architecture reference |

---

## Setup & Installation

### Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) package manager
- MuJoCo (see data section for install notes)
- Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU — auto-detected at runtime

### Install

```bash
git clone https://github.com/Team-Capstone-ANA-699-Robotics/robotics-capstone.git
cd robotics-capstone
uv sync
```

### Data

The HDF5 dataset is not tracked in this repository. Download via Minari before running any training notebook:

```python
import minari
minari.download_dataset("mujoco/halfcheetah/medium-v0")
minari.download_dataset("mujoco/halfcheetah/expert-v0")
```

The notebooks combine these two sub-splits at load time to replicate the D4RL `halfcheetah-medium-expert-v2` composite. If you prefer the HDF5 file directly, place `halfcheetah_medium_expert-v2.hdf5` in the `data/` directory — the notebooks include an HDF5 fallback loader.

> **Note on d4rl:** The `d4rl` package does not install on Apple Silicon due to a `pybullet` arm64 compilation failure. This project uses Minari throughout. The EDA notebook loads D4RL via HDF5 fallback for comparison purposes only.

---

## Checkpoints

Trained model weights are stored in `checkpoints/`. Each `.pt` file contains the model state dictionary, optimizer state, training configuration, iteration counter, and embedded state normalization statistics (`state_mean`, `state_std`).

| File | Corresponds To |
|---|---|
| `dt_full_final_M5-Max-Cosine-bs256-K30-RTG16585-nh2-newDTarch.pt` | `training_handcoded_K30_bs256_primary.ipynb` |
| `dt_full_final_M5-Max-Cosine-bs64-K20-RTG16585-nh2-newDTarch.pt` | `training_handcoded_K20_bs64.ipynb` |
| `checkpoints_valdiag/dt_valdiag_final.pt` | `validation_diagnostic.ipynb` |

To load a checkpoint for evaluation:

```python
import torch
ckpt = torch.load("checkpoints/dt_full_final_M5-Max-Cosine-bs256-K30-RTG16585-nh2-newDTarch.pt",
                  map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
state_mean = ckpt["state_mean"]
state_std  = ckpt["state_std"]
```

---

## Paper

The capstone paper is in final preparation. Chapter drafts are available in `docs/chapters/`.

| Chapter | Status |
|---|---|
| Chapter 1 — Introduction | Complete |
| Chapter 2 — Literature Review | Complete |
| Chapter 3 — Methodology | Complete |
| Chapter 4 — Results | Complete (tables/figures in progress) |
| Chapter 5 — Discussion | Complete (tables/figures in progress) |

The full compiled paper will be posted to `docs/` upon final submission.

---

## Citation

```bibtex
@mastersthesis{juan_kast_terando_2026,
  title   = {Eliminating the High-Risk Interaction Cost of Autonomous Robotics
             Through Offline Sequence Modeling: A Decision Transformer Approach},
  author  = {Juan, Darwin and Kast, Dan and Terando, David},
  school  = {National University},
  year    = {2026},
  program = {Master of Data Science},
  course  = {ANA699}
}
```

---

## License

This project is submitted in partial fulfillment of the requirements for the Master of Data Science degree at National University. Code is made available for academic and research purposes.
