<div align="center">

```
██████╗  ██████╗ ██████╗  ██████╗ ████████╗██╗ ██████╗███████╗
██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝██║██╔════╝██╔════╝
██████╔╝██║   ██║██████╔╝██║   ██║   ██║   ██║██║     ███████╗
██╔══██╗██║   ██║██╔══██╗██║   ██║   ██║   ██║██║     ╚════██║
██║  ██║╚██████╔╝██████╔╝╚██████╔╝   ██║   ██║╚██████╗███████║
╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝   ╚═╝   ╚═╝ ╚═════╝╚══════╝
```

# 🤖 ANA 699 — Robotics Capstone

### Offline Reinforcement Learning for Robotic Control  
### via Decision Transformers

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-Simulation-00ADD8?style=for-the-badge)](https://mujoco.org)
[![D4RL](https://img.shields.io/badge/D4RL-Dataset-FF6B35?style=for-the-badge)](https://github.com/Farama-Foundation/d4rl)
[![Minari](https://img.shields.io/badge/Minari-Dataset-6C63FF?style=for-the-badge)](https://minari.farama.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

---

*Can a robot learn to move — not from trial and error — but from studying the past?*  
*This project explores that question.*

---

</div>

## 📌 Overview

This capstone project investigates **offline reinforcement learning** for robotic locomotion and manipulation tasks using **Decision Transformers** — a sequence modeling approach that reframes RL as a conditional sequence prediction problem.

Rather than learning from live environment interaction, our agent learns from **pre-collected datasets** (D4RL / Minari), treating trajectories as language-like sequences and leveraging transformer architectures to generate goal-conditioned behavior.

---

## 🧠 Core Concepts

| Concept | Description |
|---|---|
| **Offline RL** | Learning from fixed datasets without environment interaction |
| **Decision Transformer** | GPT-style architecture applied to RL trajectory sequences |
| **Return Conditioning** | Agent is prompted with a desired return to guide behavior |
| **MuJoCo** | Physics-based simulation environment for robot tasks |
| **D4RL / Minari** | Benchmark datasets of pre-collected robot trajectories |

---

## 🗂️ Project Structure

```
robotics-capstone/
│
├── 📁 data/                  # Dataset loading and preprocessing
│   ├── loader.py
│   └── preprocessing.py
│
├── 📁 models/                # Model architecture
│   ├── decision_transformer.py
│   └── gpt2_backbone.py
│
├── 📁 training/              # Training loops and configs
│   ├── trainer.py
│   └── config.py
│
├── 📁 evaluation/            # Evaluation and metrics
│   ├── evaluate.py
│   └── metrics.py
│
├── 📁 notebooks/             # Exploratory analysis & demos
│   └── demo.ipynb
│
├── 📁 results/               # Saved checkpoints and plots
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Team-Capstone-ANA-699-Robotics/robotics-capstone.git
cd robotics-capstone
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install MuJoCo
Follow the official MuJoCo installation guide: [mujoco.org](https://mujoco.org)

---

## 🚀 Usage

```bash
# Train the Decision Transformer
python training/trainer.py --env hopper-medium-v2 --epochs 10

# Evaluate a trained model
python evaluation/evaluate.py --checkpoint results/checkpoint.pt

# Run the demo notebook
jupyter notebook notebooks/demo.ipynb
```

---

## 👥 Team

| Name | Role | GitHub |
|---|---|---|
| Daniel Kast | Project Lead | [@Daniel-Kast](https://github.com/Daniel-Kast) |
| Teammate 2 | Role TBD | — |
| Teammate 3 | Role TBD | — |

---

## 📚 References

- Chen, L. et al. (2021). [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
- Fu, J. et al. (2020). [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://arxiv.org/abs/2004.07219)
- Farama Foundation. [Minari — Offline RL Datasets](https://minari.farama.org)
- Todorov, E. et al. [MuJoCo: A physics engine for model-based control](https://mujoco.org)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

*ANA 699 Capstone · University Project · 2026*

</div>
