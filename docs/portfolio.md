# Portfolio and interview notes

## Project summary

A Python/PyTorch multi-agent reinforcement learning prototype with a custom PettingZoo environment, IPPO/MAPPO training, legal-action masking and independently replayed evaluation. The work emphasizes reproducibility, automated testing and transparent research limitations.

Development was AI-assisted. Describe only the parts you personally understand, reviewed and can explain; a repository demonstrates a project, not automatically sole authorship of every implementation detail.

## Lebenslauf-Vorschlag

**Monopoly MARL — Python-/PyTorch-Portfolio-Projekt**  
AI-gestütztes Multi-Agent-RL-Projekt mit eigener PettingZoo-Spielumgebung, IPPO-/MAPPO-Training und reproduzierbarer Evaluation. Zwei unabhängige CPU-Trainingsläufe liefern identische Modellparameter und Einzelspiel-Ergebnisse; automatisierte Regressionstests und Linux-CI sichern die Implementierung ab.

Kurzfassung:

> Multi-Agent-RL-Prototyp mit Python, PyTorch und PettingZoo; reproduzierbares PPO-Training, Action Masking und automatisierte Tests/CI.

## English CV suggestion

**Monopoly MARL — Python/PyTorch portfolio project**  
Built and validated an AI-assisted multi-agent RL prototype with a custom PettingZoo environment, IPPO/MAPPO training and reproducible evaluation. Verified exact parameter equality across independent same-seed CPU runs and supported the implementation with regression tests and Linux CI.

Use “Built and validated” only if it accurately describes your personal contribution. Otherwise use “Developed and reviewed with AI assistance” or “Portfolio project: ...”. Do not claim the agents beat the heuristic: the short evaluation does not establish that.

## Be ready to explain

1. Why only one player's action executes despite using the Parallel API.
2. How legal-action masks prevent illegal choices, and why server-side validation is still useful.
3. How a misaligned `done` flag can leak advantages across episodes.
4. The difference between IPPO's observation critic and MAPPO's global/player-conditioned critic.
5. Why exactly repeated weights are a reproducibility result, not evidence of a strong policy.
6. Why the eight-game smoke result cannot reliably rank algorithms.
7. Why the unilateral trading placeholder was disabled.
8. Which game rules remain simplified, and what you would implement next.

## Honest next step

Longer multi-seed training with held-out evaluation seeds, learning curves and fixed baseline matchups. Add rule coverage incrementally rather than claiming a complete Monopoly engine.
