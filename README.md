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
*This project explores that question — three ways.*

---

</div>

## 📌 Overview

This capstone project investigates **offline reinforcement learning** for robotic locomotion and manipulation tasks using **Decision Transformers** — a sequence modeling approach that reframes RL as a conditional sequence prediction problem.

Each team member independently trains and tunes their own model variant, with results compared in a shared evaluation framework. Rather than learning from live environment interaction, our agents learn from **pre-collected datasets** (D4RL / Minari), treating trajectories as language-like sequences and leveraging transformer architectures to generate goal-conditioned behavior.

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
├── 📁 data/                        # Shared dataset loading & preprocessing
│   ├── loader.py
│   ├── preprocessing.py
│   └── __init__.py
│
├── 📁 models/                      # Shared base model architecture
│   ├── base_transformer.py
│   └── __init__.py
│
├── 📁 experiments/                 # Individual model experiments
│   │
│   ├── 📁 daniel/                  # Daniel's experiment space
│   │   ├── models/model.py         # Daniel's model variant
│   │   ├── configs/params.json     # Hyperparameters & settings
│   │   ├── results/                # Daniel's output & metrics
│   │   └── README.md               # Notes on Daniel's approach
│   │
│   ├── 📁 darwin/                  # Darwin's experiment space
│   │   ├── models/model.py
│   │   ├── configs/params.json
│   │   ├── results/
│   │   └── README.md
│   │
│   └── 📁 dave/                    # Dave's experiment space
│       ├── models/model.py
│       ├── configs/params.json
│       ├── results/
│       └── README.md
│
├── 📁 evaluation/                  # Shared evaluation framework
│   ├── evaluate.py
│   ├── metrics.py
│   └── __init__.py
│
├── 📁 comparison/                  # Cross-model result comparison
│   ├── compare_results.py
│   └── visualize.py
│
├── 📁 notebooks/                   # Demos & exploratory analysis
│   └── demo.ipynb
│
├── 📁 docs/                        # Project notes & documentation
│   └── notes.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔀 Branching Strategy

Each team member works on their own branch and opens a Pull Request to merge into `main`:

```
main              ← stable, reviewed code only
daniel/model      ← Daniel's experiments
darwin/model      ← Darwin's experiments
dave/model        ← Dave's experiments
```

**Ground Rules:**
- ❌ Never commit directly to `main`
- ✅ Always work on your personal branch
- ✅ Open a Pull Request when merging to `main`
- ✅ At least one teammate must review before merging
- ✅ Document your hyperparameters in `configs/params.json`
- ✅ Save all outputs to your own `results/` folder
- ✅ Discuss changes to shared files before committing

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
# Train your model (run from your experiments folder)
python experiments/daniel/models/model.py --config experiments/daniel/configs/params.json

# Evaluate a trained model
python evaluation/evaluate.py --checkpoint experiments/daniel/results/checkpoint.pt

# Compare all three models
python comparison/compare_results.py

# Run the demo notebook
jupyter notebook notebooks/demo.ipynb
```

---

## 📊 Experiment Comparison

| Team Member | Model Variant | Key Hyperparameters | Best Score |
|---|---|---|---|
| Daniel Kast | TBD | TBD | TBD |
| Darwin Juan | TBD | TBD | TBD |
| Dave Terando | TBD | TBD | TBD |

*This table will be updated as experiments are completed.*

---

## 👥 Team

| Name | Role | GitHub |
|---|---|---|
| Daniel Kast | Co-Lead | [@Daniel-Kast](https://github.com/Daniel-Kast) |
| Darwin Juan | Co-Lead | — |
| Dave Terando | Co-Lead | — |

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
