<div align="center">

# 🤖 ANA 699 — Robotics Capstone

### Offline Reinforcement Learning for Robotic Control
### via Decision Transformers

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-Simulation-00ADD8?style=for-the-badge)](https://mujoco.org)
[![Minari](https://img.shields.io/badge/Minari-Dataset-6C63FF?style=for-the-badge)](https://minari.farama.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

---

*Can a robot learn to move — not from trial and error — but from studying the past?*
*This project explores that question — three ways.*

---

</div>

## 📌 Overview

This capstone project investigates **offline reinforcement learning** for robotic locomotion using **Decision Transformers** — a sequence modeling approach that reframes RL as a conditional sequence prediction problem.

Each team member independently trains and evaluates their own model variant on identical hardware tiers, with results compared in a shared evaluation framework. Rather than learning from live environment interaction, our agents learn from **pre-collected datasets** (Minari), treating trajectories as language-like sequences and leveraging transformer architectures to generate goal-conditioned behavior.

The central research claim: **offline RL via Decision Transformer achieves zero Cumulative Interaction Cost (CIC) during training** while producing a competent MuJoCo locomotion policy — and that this is reproducible across consumer, mid-tier, and high-performance hardware.

---

## 🧠 Core Concepts

| Concept | Description |
|---|---|
| **Offline RL** | Learning from fixed datasets without live environment interaction |
| **Decision Transformer** | GPT-style causal transformer applied to RL trajectory sequences |
| **Return Conditioning** | Agent is prompted with a desired return-to-go (RTG) to guide behavior |
| **CIC** | Cumulative Interaction Cost — total live environment steps during training. Our DT = 0 |
| **MuJoCo** | Physics-based simulation environment for robot locomotion tasks |
| **Minari** | Modern offline RL dataset library (replaces deprecated D4RL) |

---

## 📊 Results

All three platforms trained the same hand-coded Decision Transformer architecture on the Minari mujoco/halfcheetah/medium-v0 + mujoco/halfcheetah/expert-v0 combined dataset (2,000 episodes, 2,000,000 timesteps).

### Cross-Platform Evaluation Results

| Team Member | Hardware | K | Batch Size | n_heads | LR Schedule | D4RL Norm. Score | Training Time | BPS |
|---|---|---|---|---|---|---|---|---|
| Daniel Kast | M1 iMac (8.6 GB) | 20 | 64 | 2 | Cosine | **139.1 ± 1.0** | ~58 min | ~33.9 |
| Darwin Juan | NVIDIA RTX 6000 Pro (Colab) | 30 | 64 | 1 | Cosine | **82.4 ± 30.7** | — | — |
| Dave Terando | M5 Max (128 GB) | 30 | 256 | 2 | Cosine | **139.3 ± 0.4** | ~45 min | ~49.4 |

**Chen et al. (2021) HalfCheetah Medium-Expert benchmark: ~86.8**

### Key Findings

- **CIC = 0** across all three platforms — zero live environment steps during training
- Daniel's M1 iMac (consumer hardware, 8.6 GB RAM) matched Dave's M5 Max result within noise, exceeding the paper benchmark by ~52 normalized points
- Darwin's Colab GPU result shows higher variance (±30.7), indicating sensitivity to the K=30 context window and target RTG selection on that hardware tier
- The hand-coded backbone outperformed the HuggingFace GPT2Model backbone by ~59 normalized points at matched hyperparameters (136.2 vs 77.4)

---

## 🗂️ Project Structure

    robotics-capstone/
    │
    ├── data/                        # Shared dataset loading & preprocessing
    ├── models/                      # Shared base model architecture
    ├── experiments/
    │   ├── daniel/                  # M1 iMac — consumer baseline
    │   │   ├── daniel_DT_M1_baseline_pipeline.ipynb
    │   │   ├── daniel_minari_DT.ipynb
    │   │   ├── daniel_EDA_DT.ipynb
    │   │   ├── daniel DT Half Cheetah Video Run.ipynb
    │   │   ├── checkpoints/
    │   │   └── results/
    │   ├── darwin/                  # Google Colab — mid-tier GPU
    │   └── dave/                    # M5 Max MacBook Pro — high performance
    ├── evaluation/                  # Shared evaluation framework
    ├── comparison/                  # Cross-platform result comparison
    ├── requirements.txt
    └── README.md

---

## ⚙️ Installation

### Apple Silicon (M1 iMac / M5 Max) — conda

    git clone https://github.com/Team-Capstone-ANA-699-Robotics/robotics-capstone.git
    cd robotics-capstone
    conda create -n robotics-capstone python=3.10
    conda activate robotics-capstone
    conda install pytorch -c pytorch
    pip install minari[hf] gymnasium[mujoco] mujoco transformers numpy pandas matplotlib seaborn tqdm psutil

Known issue on Apple Silicon — if kernel crashes on torch import:

    conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
    conda activate robotics-capstone

### Google Colab (Darwin)

    pip install minari[hf] gymnasium[mujoco] mujoco transformers torch numpy pandas matplotlib seaborn tqdm psutil

### M5 Max (Dave) — uv

    uv pip install torch minari[hf] gymnasium[mujoco] mujoco transformers numpy pandas matplotlib seaborn tqdm psutil

---

## 📁 Dataset

All experiments use Minari datasets. Do not use D4RL — deprecated and incompatible with Apple Silicon arm64.

    import minari
    ds = minari.load_dataset('mujoco/halfcheetah/medium-v0', download=True)
    ds = minari.load_dataset('mujoco/halfcheetah/expert-v0', download=True)

Datasets are cached to ~/.minari/datasets/ after first download. Combined: 2,000 episodes, 2,000,000 timesteps.

---

## 👥 Team

| Name | Role | Hardware | GitHub |
|---|---|---|---|
| Daniel Kast | Consumer Baseline | M1 iMac, 8.6 GB | [@Daniel-Kast](https://github.com/Daniel-Kast) |
| Darwin Juan | Mid-Tier GPU | Google Colab, NVIDIA RTX 6000 Pro | [@darwinjuan](https://github.com/darwinjuan) |
| Dave Terando | High Performance | M5 Max MacBook Pro, 128 GB | [@DaveT-Git](https://github.com/DaveT-Git) |

---

## 📚 References

- Chen, L. et al. (2021). Decision Transformer: Reinforcement Learning via Sequence Modeling. https://arxiv.org/abs/2106.01345
- Fu, J. et al. (2020). D4RL: Datasets for Deep Data-Driven Reinforcement Learning. https://arxiv.org/abs/2004.07219
- Farama Foundation. Minari — Offline RL Datasets. https://minari.farama.org
- Todorov, E. et al. MuJoCo: A physics engine for model-based control. https://mujoco.org

---

## 📄 License

This project is licensed under the MIT License. See LICENSE for details.

---

*ANA 699 Capstone · 2026*
