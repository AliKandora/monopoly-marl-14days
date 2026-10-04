# Monopoly-MARL: Präventives Troubleshooting & Notfall-Matrix

Dieses Dokument bietet sofort anwendbare Lösungen für die typischen Fehlermuster und Performance-Bottlenecks, die während des 14-tägigen MARL-Entwicklungszyklus auftreten können.

---

## 1. Problem-Lösungs-Matrix

| Symptom / Hürde | Wahrscheinliche Ursache | Sofortige Korrekturmaßnahme |
|---|---|---|
| **1. Agent lernt nicht / Loss explodiert (NaN)** | - Zu hohe Learning Rate<br>- Explodierende Value-Vorhersagen bei großen Net-Worth-Werten<br>- Fehlendes Gradient Clipping | 1. **Gradient Clipping:** `torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)`<br>2. **Value Clipping:** Value Loss wie im PPO-Paper clippen (`clip_range_vf=0.2`).<br>3. **Observation Normalization:** Cash und Preise durch $1000$ oder $1500$ teilen, damit Inputs im Bereich $[-3, 3]$ liegen.<br>4. **LR Warmup & Schedule:** Linearer Decay von $3 \cdot 10^{-4}$ auf $1 \cdot 10^{-5}$. |
| **2. Environment läuft zu langsam (< 100 FPS)** | - Python `for`-Schleifen für 40 Spielfelder<br>- Rendering-Aufrufe im Trainings-Loop<br>- Zu häufiges Erzeugen temporärer Dictionaries | 1. **NumPy Vektorisierung:** Board-States als feste NumPy-Arrays (`int8` / `float32`) halten.<br>2. **Render-Entkopplung:** Kein `render()` während des Trainings, nur alle $N$ Evaluationsepisoden.<br>3. **Vectorized Envs:** Gymnasium `AsyncVectorEnv` oder PettingZoo Parallel Wrappers nutzen, um parallele Prozesse zu nutzen. |
| **3. Invalid Action Execution & Crashs** | - Policy wählt verbotenen Index (z.B. Kaufen ohne Geld)<br>- Maske im Forward-Pass vergessen oder nicht auf logits angewendet | 1. **Strict Mask Validation:** Assertion vor `env.step()`: `assert mask[action] == 1, f"Illegal action {action}"`<br>2. **Logit Clamping:** Ungültige Aktionen strikt mit `-1e8` überschreiben, um `exp(logit) = 0` zu erzwingen.<br>3. **Fallback in Env:** Führt ein Agent dennoch eine illegale Aktion aus, wird diese im Env transparent als `PASS_TURN` gewertet und mit moderater Strafe (`-5.0`) belegt. |
| **4. Non-Stationarity bei Self-Play (Zyklen/Regression)** | - Agenten überpassen sich gegenseitig an aktuelle Schwächen<br>- Vergessen früherer effektiver Strategien | 1. **Opponent Pool:** 80% der Trainingsspiele gegen die aktuellste Policy, 20% gegen fixierte Checkpoints aus früheren Epochen.<br>2. **Elo-Tracking:** Evaluation der Checkpoints in einer Round-Robin-Turnier-Matrix zur Messung echter Spielstärke statt bloßer Self-Play-Rewards. |

---

## 2. Detaillierte Code-Rezepte zur Fehlerbehebung

### A. Gradient Clipping & Value Loss Protection in PyTorch

```python
# Critic Loss mit Clipping (MAPPO Best Practice)
v_clipped = v_old + torch.clamp(v_pred - v_old, -clip_range_vf, clip_range_vf)
vf_loss1 = (v_pred - returns) ** 2
vf_loss2 = (v_clipped - returns) ** 2
loss_critic = 0.5 * torch.mean(torch.max(vf_loss1, vf_loss2))

# Gradient Norm Clipping vor dem Optimizer-Step
loss_total = loss_actor + 0.5 * loss_critic - 0.01 * entropy_loss
optimizer.zero_grad()
loss_total.backward()
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
optimizer.step()
```

### B. Überprüfung der Simulationsgeschwindigkeit (FPS Benchmark)

Falls die Trainingszeit unerwartet hoch ist, starte diesen Check im Terminal:

```python
import time
from src.envs.monopoly_env import MonopolyEnv
from src.agents.random_agent import RandomAgent

env = MonopolyEnv()
obs, _ = env.reset()
agents = {a: RandomAgent(a) for a in env.possible_agents}

start = time.time()
steps = 10000
for _ in range(steps):
    if not env.agents:
        obs, _ = env.reset()
    actions = {a: agents[a].select_action(obs[a]) for a in env.agents}
    obs, _, _, _, _ = env.step(actions)

fps = steps / (time.time() - start)
print(f"Aktuelle Simulationsgeschwindigkeit: {fps:.1f} Steps/Sekunde (Soll: >500)")
```

---

## 3. Sanity-Checkliste vor längeren Trainingsläufen

- [ ] Wurden alle Zufallsquellen (`seed`) im Environment und PyTorch deterministisch gesetzt?
- [ ] Werden W&B Runs eindeutig getaggt (z.B. `ippo-v1-baseline`, `mappo-v1-ctde`)?
- [ ] Werden Checkpoints automatisch bei verbesserter Win-Rate gegen den Heuristic-Bot gespeichert?
- [ ] Ist das Action-Masking bei `BUILD_HOUSE` und `BUY_PROPERTY` mathematisch und regeltechnisch wasserdicht?
