# Eliminating the High-Risk Interaction Cost of Autonomous Robotics Through Offline Sequence Modeling: A Decision Transformer Approach

**Darwin Juan · Dan Kast · David Terando**
Master of Data Science — National University · ANA699 Capstone · Spring 2026

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS-orange.svg)](https://pytorch.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-HalfCheetah--v4-green.svg)](https://mujoco.org/)
[![Hardware](https://img.shields.io/badge/Hardware-M1%20iMac%208.6GB-lightgrey.svg)](https://www.apple.com)

---

## Abstract

Online reinforcement learning algorithms require 800,000–1,200,000 live environment interactions to reach proficiency on standard locomotion benchmarks, posing a safety and economic barrier to autonomous robotics training on physical hardware. This study investigated whether offline sequence modeling via the Decision Transformer can eliminate this high-risk Cumulative Interaction Cost (CIC) while producing a competent baseline policy. A 727,558-parameter Decision Transformer was trained on the Minari HalfCheetah Medium-Expert dataset across three hardware platforms — Apple M5 Max, Apple M1 iMac, and NVIDIA cloud GPU — and evaluated in the MuJoCo physics simulator. The production configuration achieved a D4RL normalized score of 139.33 ± 0.36 at CIC = 0, with zero catastrophic failures across 150 evaluation episodes and cross-platform mean returns agreeing within 0.27 normalized points. The findings establish that offline sequence modeling delivers competent, deterministic policies on consumer-grade hardware without incurring online environment-interaction costs.

---

## This Folder — Daniel Kast (M1 iMac, Consumer Baseline)

This experiment space contains Daniel's contribution to the ANA 699 three-tier
hardware benchmarking experiment. The M1 iMac (8.6 GB unified memory) represents
the consumer baseline tier — the lowest-cost hardware on which meaningful offline
RL research can be conducted.

The central claim under test: offline RL via Decision Transformer achieves zero
Cumulative Interaction Cost (CIC) during training while producing a competent
MuJoCo locomotion policy — and that this is reproducible on consumer-grade
Apple Silicon hardware.

---

## Folder Structure

    experiments/daniel/
    │
    ├── daniel_DT_M1_baseline_pipeline.ipynb   # PRIMARY training notebook
    ├── daniel_minari_DT.ipynb                 # Original baseline notebook
    ├── daniel_EDA_DT.ipynb                    # Exploratory data analysis
    ├── daniel DT Half Cheetah Video Run.ipynb # Policy video renderer
    ├── daniel_minari_DT-Copy1.ipynb           # Development iteration
    │
    ├── checkpoints/
    │   └── dt_full_final_M1-iMac.pt           # Final trained model weights
    │
    ├── results/                               # All outputs and metrics
    │   ├── state_mean.npy                     # Normalization stats (required for inference)
    │   ├── state_std.npy
    │   ├── hw_full_M1-iMac.csv                # Hardware tracker — cross-platform comparison input
    │   ├── hw_quickval_M1-iMac.csv/.png       # Quick validation metrics
    │   ├── training_history_M1-iMac.csv       # Loss/BPS/grad norm history
    │   ├── training_curves_M1-iMac.png        # Training diagnostic figure
    │   ├── eval_summary_M1-iMac.csv           # Headline evaluation results
    │   ├── eval_figures_M1-iMac.png           # Evaluation figures (7a/7b/7c/8)
    │   ├── rtg_sweep_M1-iMac.csv/.png         # RTG sweep results
    │   └── videos_M1-iMac/                    # MuJoCo rollout recordings
    │
    ├── configs/
    │   └── params.json                        # Hyperparameters
    ├── models/
    │   └── model.py                           # Model definition
    ├── requirements.txt
    └── README.md

---

## Key Results

| Metric | Value |
|---|---|
| Hardware | M1 iMac, 8.6 GB unified memory |
| D4RL Normalized Score | **139.1 ± 1.0** (n=50 episodes) |
| Mean Raw Return | 16,985 ± 119 |
| Min Raw Return | 16,419 |
| Target RTG | 16,585 |
| Success Gap | −400 (target exceeded) |
| Catastrophic Failures | 0 of 50 episodes |
| Training Time | ~58 minutes |
| Mean BPS | 33.9 |
| Training CIC | **0 environment steps** |
| Chen et al. (2021) benchmark | ~86.8 |
| Delta vs benchmark | **+52.3 normalized points** |

---

## Novel Contribution: Cumulative Interaction Cost (CIC)

This study introduces the **Cumulative Interaction Cost (CIC)** metric:

    CIC = Σ(e=1 to E) L_e · 𝟙(training_active)

Where E = number of live episodes, L_e = episode length, and 𝟙 is an indicator
function equal to 1 when the agent is training via live interaction. Online RL
baselines (PPO, SAC) incur CIC of 800,000–1,200,000 steps. The Decision
Transformer achieves CIC = 0 by training exclusively on static offline data.

---

## Notebooks

| Notebook | Description |
|---|---|
| `daniel_DT_M1_baseline_pipeline.ipynb` | Primary production notebook — hand-coded DT backbone, K=20, bs=64, cosine LR, 100k iterations on M1 iMac. Produces headline result of 139.1 ± 1.0 normalized score. |
| `daniel_minari_DT.ipynb` | Original baseline notebook — earlier pipeline used for initial results and RTG sweep. |
| `daniel_EDA_DT.ipynb` | Full exploratory data analysis of the Minari medium-expert dataset. Derives target RTG=16,585 from expert sub-split max return. |
| `daniel DT Half Cheetah Video Run.ipynb` | Policy video renderer — loads trained checkpoint and records MuJoCo rollouts as MP4. |

---

## Checkpoints

Trained model weights are stored in `checkpoints/`. Each `.pt` file contains the
model state dictionary, training configuration, and embedded state normalization
statistics (`state_mean`, `state_std`).

| File | Corresponds To |
|---|---|
| `dt_full_final_M1-iMac.pt` | `daniel_DT_M1_baseline_pipeline.ipynb` — primary result |

To load the checkpoint for evaluation:

    import torch
    ckpt = torch.load(
        "experiments/daniel/checkpoints/dt_full_final_M1-iMac.pt",
        map_location="cpu",
        weights_only=False
    )
    model.load_state_dict(ckpt["model_state"])
    state_mean = ckpt["state_mean"]
    state_std  = ckpt["state_std"]

Note: weights_only=False is required — PyTorch 2.11 changed the default and
will block loading custom dataclass objects without this flag.

---

## Setup & Installation

### Prerequisites

- Python 3.10
- conda (Anaconda or Miniconda)
- Apple Silicon Mac (MPS acceleration auto-detected)

### Install

    git clone https://github.com/Team-Capstone-ANA-699-Robotics/robotics-capstone.git
    cd robotics-capstone
    conda create -n robotics-capstone python=3.10
    conda activate robotics-capstone
    conda install pytorch -c pytorch
    pip install minari[hf] gymnasium[mujoco] mujoco transformers numpy pandas matplotlib seaborn tqdm psutil

Known issue on Apple Silicon — if kernel crashes on torch import:

    conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
    conda activate robotics-capstone

### Launching Jupyter

Always launch from the correct conda environment:

    cd ~/Desktop/robotics-capstone
    conda activate robotics-capstone
    caffeinate &
    jupyter notebook

Never launch from the base environment — PyTorch MPS, MuJoCo, and Minari
will not be available.

---

## Dataset

All experiments use Minari datasets. Do not use D4RL — deprecated and
incompatible with Apple Silicon arm64 due to pybullet compilation failure.

    import minari
    ds = minari.load_dataset('mujoco/halfcheetah/medium-v0', download=True)
    ds = minari.load_dataset('mujoco/halfcheetah/expert-v0', download=True)

Datasets are cached to ~/.minari/datasets/ after first download.
Combined: 2,000 episodes, 2,000,000 timesteps.

---

## Citation

    @mastersthesis{juan_kast_terando_2026,
      title   = {Eliminating the High-Risk Interaction Cost of Autonomous Robotics
                 Through Offline Sequence Modeling: A Decision Transformer Approach},
      author  = {Juan, Darwin and Kast, Dan and Terando, David},
      school  = {National University},
      year    = {2026},
      program = {Master of Data Science},
      course  = {ANA699}
    }

---

## License

This project is submitted in partial fulfillment of the requirements for the
Master of Data Science degree at National University. Code is made available
for academic and research purposes.
