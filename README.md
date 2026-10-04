# Monopoly-MARL: Multi-Agent Reinforcement Learning in a Complex Game Environment

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-Parallel_API-orange.svg)](https://pettingzoo.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Project_Status-14--Day_Sprint-success.svg)](#14-tage-sprint-roadmap)

Ein 14-tägiges Deep-Tech-Forschungsprojekt zur Erforschung autonomer Multi-Agent Reinforcement Learning (MARL) Algorithmen im klassischen Brettspiel **Monopoly**. Das Projekt nutzt Centralized Training with Decentralized Execution (CTDE) mit **Independent PPO (IPPO)** und **Multi-Agent PPO (MAPPO)** im PettingZoo-Standard, um komplexe Strategien in Ressourcenmanagement, Farbgruppen-Akkumulation, dynamischem Trading und Verhandlungsführung zu erlernen.

---

## Projekt-Dashboard & Architektur

```
                                  +------------------------------------+
                                  |         Global Board State         |
                                  | (Tiles, Ownership, Bank, Jail, ...) |
                                  +-----------------+------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
          +------------------------------+                      +------------------------------+
          |   Local Observation (Dict)   |                      |     Action Mask Validator    |
          |  [Board + Cash + Pos + Net]  |                      | [Filtered Valid Action Space]|
          +--------------+---------------+                      +--------------+---------------+
                         |                                                     |
                         +--------------------------+--------------------------+
                                                    |
                                                    v
                                      +---------------------------+
                                      |     Actor-Critic Policy   |
                                      | (Decentralized Execution) |
                                      +-------------+-------------+
                                                    |
                      +-----------------------------+-----------------------------+
                      |                                                           |
                      v                                                           v
       +-------------------------------+                           +-------------------------------+
       |   IPPO (Decentralized Value)  |                           |     MAPPO (Central Critic)    |
       |     V(s_i) from Local State   |                           |    V(s_global) from Full Obs  |
       +-------------------------------+                           +-------------------------------+
```

### Modul-Struktur

```
monopoly-marl-14days/
├── daily_trackers/           # 14 tägliche Sprint-Tracker (DAY_01.md - DAY_14.md)
│   ├── DAY_01.md ... DAY_14.md
├── src/
│   ├── envs/                 # PettingZoo ParallelEnv Implementierung
│   │   ├── monopoly_env.py
│   ├── agents/               # Random, Heuristik, IPPO, MAPPO
│   │   ├── random_agent.py
│   │   ├── heuristic_agent.py
│   ├── utils/                # Action Masking, Reward Wrappers, GAE
│   │   ├── action_masking.py
│   ├── visualization/        # Board-Render, Elo-Matrix, Loss-Curves
├── tests/                    # Pytest Suite
│   ├── test_env.py
├── CHEATSHEET_CORE.md        # Mathematische Formeln, PettingZoo API & Pitfalls
├── TROUBLESHOOTING_PREVENTIVE.md # Präventive Problemlösungsmatrix
├── requirements.txt          # Projekt-Abhängigkeiten
└── README.md                 # Dieses Dokument
```

---

## Schnellstart & Installation

### 1. Repository klonen & Virtual Environment erstellen
```bash
# In das Projektverzeichnis wechseln
cd monopoly-marl-14days

# Virtual Environment erstellen
python -m venv .venv

# Aktivieren (Windows PowerShell):
.venv\Scripts\Activate.ps1
# Aktivieren (Linux/macOS):
source .venv/bin/activate
```

### 2. Dependencies installieren
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Schneller Funktionscheck (Unit Tests)
```bash
pytest tests/ -v
```

### 4. Baseline-Simulation ausführen
```bash
python -c "
from src.envs.monopoly_env import MonopolyEnv
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.random_agent import RandomAgent

env = MonopolyEnv(render_mode='human')
obs, _ = env.reset()
agents = {
    'player_0': HeuristicAgent('player_0'),
    'player_1': RandomAgent('player_1'),
    'player_2': RandomAgent('player_2'),
    'player_3': RandomAgent('player_3')
}

for step in range(20):
    actions = {a: agents[a].select_action(obs[a]) for a in env.agents}
    obs, rewards, term, trunc, _ = env.step(actions)
    env.render()
    if not env.agents:
        break
"
```

---

## 14-Tage-Sprint-Roadmap

Jeder Tag besitzt einen dedizierten, detaillierten Tracker mit Checklisten, präventiver Fehlerbehandlung und Fallback-Prioritäten:

| Tag | Phase | Fokus & Meilenstein | Dokument |
|---|---|---|---|
| **01** | Env Scaffold | PettingZoo API & Dev-Setup: ParallelEnv Dummy | [DAY_01.md](daily_trackers/DAY_01.md) |
| **02** | Game Logic | Monopoly Board & Game Logic: 40 Tiles, Eigentum, Cash, Jail | [DAY_02.md](daily_trackers/DAY_02.md) |
| **03** | Constraints | Action Space & Action Masking: Illegale Züge filtern | [DAY_03.md](daily_trackers/DAY_03.md) |
| **04** | Rewards | Reward Engineering: Dense vs. Sparse Net-Worth Balancing | [DAY_04.md](daily_trackers/DAY_04.md) |
| **05** | Baselines | Baselines (Random & Heuristik): 95% Win-Rate der Heuristik | [DAY_05.md](daily_trackers/DAY_05.md) |
| **06** | Single RL | Single-Agent Benchmark: PPO schlägt Heuristik | [DAY_06.md](daily_trackers/DAY_06.md) |
| **07** | Performance | Profiling & Vectorization: >500 Steps/Sekunde Durchsatz | [DAY_07.md](daily_trackers/DAY_07.md) |
| **08** | MARL Core | Independent PPO (IPPO): Erstes dezentrales Multi-Agent Training | [DAY_08.md](daily_trackers/DAY_08.md) |
| **09** | CTDE | MAPPO: Centralized Critic $V(s_{global})$ & Joint Observation | [DAY_09.md](daily_trackers/DAY_09.md) |
| **10** | MLOps | Hyperparameter Tuning & W&B Monitoring Dashboard | [DAY_10.md](daily_trackers/DAY_10.md) |
| **11** | Negotiation | Trading & Negotiation Logic: Bilateraler Asset-Tausch | [DAY_11.md](daily_trackers/DAY_11.md) |
| **12** | Self-Play | Advanced Self-Play & Opponent Pool mit historischen Checkpoints | [DAY_12.md](daily_trackers/DAY_12.md) |
| **13** | Analysis | Evaluation, Elo-Rating & Statistische Dominanzanalyse | [DAY_13.md](daily_trackers/DAY_13.md) |
| **14** | Release | Clean Code, Visualisierung, Demo & Finaler Forschungsbericht | [DAY_14.md](daily_trackers/DAY_14.md) |

---

## Begleitdokumentation

- [STUDY_GUIDE.md](STUDY_GUIDE.md): **Start hier!** Grundlagen zu RL, PPO, CTDE, Action Masking und Verständnis-Quiz.
- [CHEATSHEET_CORE.md](CHEATSHEET_CORE.md): Formeln zu PPO, GAE, MAPPO Critic sowie PettingZoo Code-Patterns.
- [TROUBLESHOOTING_PREVENTIVE.md](TROUBLESHOOTING_PREVENTIVE.md): Präventive Notfall-Matrix für Trainingsinstabilitäten, FPS-Bottlenecks und Non-Stationarity.
