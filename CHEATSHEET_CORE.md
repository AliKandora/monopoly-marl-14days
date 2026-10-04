# Monopoly-MARL: Core Mathematics, API & Architecture Cheat Sheet

Dieses Dokument dient als zentrale Referenz für mathematische Formulierungen, PettingZoo API-Konventionen, PyTorch Action Masking und wiederkehrende MARL-Pitfalls.

---

## 1. Mathematische Grundlagen (MARL & PPO)

### 1.1 PPO Clipped Surrogate Objective

Das Ziel ist es, Policy-Updates zu begrenzen, um destruktive Gradientenschritte zu verhindern. Das Clipped Surrogate Objective für Agent $i$ lautet:

$$L^{CLIP}_i(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_{i,t}, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_{i,t} \right) \right]$$

Wobei das Wahrscheinlichkeitsverhältnis definiert ist als:
$$r_t(\theta) = \frac{\pi_\theta(a_{i,t} \mid o_{i,t})}{\pi_{\theta_{old}}(a_{i,t} \mid o_{i,t})}$$
- $\epsilon$: Clipping-Parameter (typischerweise $\epsilon \in [0.1, 0.2]$)
- $\hat{A}_{i,t}$: Vorteilsschätzung (Advantage) zum Zeitschritt $t$.

---

### 1.2 Generalized Advantage Estimation (GAE)

Zur Balancierung von Bias und Varianz in den Advantage-Schätzungen berechnen wir $\hat{A}_t^{GAE(\gamma, \lambda)}$ über den TD-Residual $\delta_t^V$:

$$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$

Typische Hyperparameter: $\gamma = 0.99$, $\lambda = 0.95$.

---

### 1.3 MAPPO Centralized Value Function (CTDE)

Im **Centralized Training with Decentralized Execution (CTDE)** Paradigma operieren die Actor-Netzwerke dezentral auf lokalen Beobachtungen $o_i$, während der Critic-Loss die globale Zustandsrepräsentation $s$ (alle Spieler, alle Besitztümer, Bankzustand) nutzt:

$$L(V_\phi) = \hat{\mathbb{E}}_{t} \left[ \max \left( (V_\phi(s_t) - \hat{R}_t)^2, \, (V_{\phi_{old}}(s_t) + \text{clip}(V_\phi(s_t) - V_{\phi_{old}}(s_t), -\epsilon_v, \epsilon_v) - \hat{R}_t)^2 \right) \right]$$

Wobei $\hat{R}_t = \hat{A}_t + V_{\phi_{old}}(s_t)$ die Target-Returns darstellen.

---

## 2. PettingZoo API Cheat Sheet

### 2.1 Parallel API (Empfohlen für Vectorized PPO)

In der Parallel API übermitteln alle zu diesem Zeitschritt handlungsfähigen Agenten gleichzeitig ihre Aktionen als Dictionary:

```python
from src.envs.monopoly_env import MonopolyEnv

env = MonopolyEnv(max_turns=200)
observations, infos = env.reset(seed=42)

while env.agents:
    # Aktionen pro aktivem Agenten generieren
    actions = {
        agent: policy_net(observations[agent]["observation"], observations[agent]["action_mask"])
        for agent in env.agents
    }
    
    # Simultaner Step
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

### 2.2 AEC API (Agent-Environment-Cycle für sequenzielle Turns)

Falls strikte Zug-Reihenfolgen (z. B. Würfeln -> Kaufen -> Bauen) als sequenzieller Graph modelliert werden:

```python
from pettingzoo.classic import tictactoe_v3 # Vergleichsbeispiel

# env.agent_selection definiert den aktuellen Spieler:
for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        mask = info["action_mask"]
        action = select_action(obs, mask)
    env.step(action)
```

---

## 3. PyTorch Action Masking Code-Snippets

Das Verhindern illegaler Züge (z. B. Kaufen ohne ausreichendes Bargeld oder Bauen ohne Farbgruppe) ist zwingend über Logit-Maskierung umzusetzen:

```python
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class MaskedActorCritic(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        logits = self.actor(obs)
        
        # Maskierung: Unzulässige Aktionen auf -1e8 setzen
        neg_inf = torch.tensor(-1e8, device=logits.device, dtype=logits.dtype)
        masked_logits = torch.where(mask.bool(), logits, neg_inf)
        
        dist = Categorical(logits=masked_logits)
        value = self.critic(obs)
        return dist, value
```

---

## 4. Top Pitfalls & Gegenmaßnahmen in Monopoly-MARL

| Pitfall | Ursache | Lösung & Best Practice |
|---|---|---|
| **Endlose Episoden** | Spieler sammeln Zinsen/GO-Geld ohne Aggression. | Hartes Limit `max_turns = 200` & moderater Schrittkosten-Abzug (`-0.01` pro Passivität). |
| **Trading Action-Explosion** | Naives Tauschen: $40 \text{ Assets} \times 40 \text{ Gegen-Assets} \times \text{Cash}$ führt zu $>100.000$ Aktionen. | 2-Phasen Trading: Erst Tauschangebot mit diskreten Stufen (z. B. 1 Asset gegen feste Cash-Optionen), dann binäres Accept/Reject. |
| **Sparse Reward Trap** | Belohnung erst beim Sieg führt dazu, dass Millionen Steps ohne Gradienten verpuffen. | Reward Shaping über Net-Worth-Differenzen: $\Delta \text{NetWorth} = (\text{Cash}_t + \text{Immobilienwert}_t) - (\text{Cash}_{t-1} + \text{Immobilienwert}_{t-1})$. |
| **Deadlock beim Häuserbau** | Agent baut keine Häuser, weil Farbgruppe übersehen wird. | Action-Masking darf `BUILD_HOUSE` nur dann auf 1 setzen, wenn alle Straßen der Farbe im Besitz sind. |
| **Policy Collapse / Overfitting** | Agent lernt gegen feste Strategie und verliert gegen neue Taktiken. | Opponent Pool mit älteren Modell-Checkpoints (Self-Play League). |
