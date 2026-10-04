# Tag 06: Single-Agent RL Benchmark

- **Titel & Fokus des Tages:** Single-Agent PPO Pipeline gegen 3 fixierte Heuristik-Bots
- **Pflicht-Milestone (Hauptziel):** Erfolgreiches Training eines einzelnen PPO-Agents in der 4-Spieler-Umgebung, der eine signifikant überlegene Win-Rate gegen 3 Heuristik-Gegner erzielt.

---

## Detaillierte Checkliste der Tagesaufgaben

- [ ] Single-Agent Trainings-Wrapper schreiben: 1 lernender Agent (`player_0`) trainiert via PPO, `player_1..3` werden von statischen `HeuristicAgent` Instanzen gesteuert.
- [ ] Actor-Critic Netzwerk mit PyTorch definieren (MLP: 2 Hidden Layers mit je 128 Units, Tanh-Aktivierung).
- [ ] Rollout-Buffer für Trajektoriensammlung implementieren (Obs, Action, LogProb, Reward, Done, Mask).
- [ ] PPO-Trainingsschleife implementieren (GAE-Berechnung, Clipped Loss, Orthogonale Gewichtsinitialisierung).
- [ ] Trainingslauf über $500.000$ Steps starten und Win-Rate-Kurve über Zeit tracken.
- [ ] Checkpoint des besten Modells unter `checkpoints/ppo_single_best.pt` persistieren.

---

## Präventive Hilfe für HEUTE: Spezifische Hürden & Sofortlösungen

- **Hürde:** PPO-Agent lernt nichts und wählt stur `PASS_TURN`.
  - *Lösung:* Entropy Bonus Koeffizient anheben (von $0.001$ auf $0.02$), um initiale Exploration im Discrete Action Space zu erzwingen. Zudem sicherstellen, dass die Net-Worth-Differenz als dichter Zwischen-Reward aktiv ist.
- **Hürde:** Advantage-Schätzungen driften ab.
  - *Lösung:* Advantages vor dem Policy-Update standardisieren: `advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)`.

---

## "Früher fertig?" (Puffer- & Bonus-Tasks)

- [ ] Learning Rate Scheduler (Linearer Decay mit PyTorch `LambdaLR`) einbinden.
- [ ] Value-Loss und Policy-Loss Kurven in TensorBoard / W&B visualisieren.

---

## "Rückstand?" (Notfall-Priorisierung)

- **MUSS fertig werden:** PPO-Trainingsschleife ohne NaNs, sichtbarer Reward-Anstieg über $200.000$ Steps.
- **KANN entfallen:** Perfektionierung der Win-Rate (>50% reicht als Machbarkeitsnachweis gegen 3 Heuristik-Bots).

---

## Notizen, Hyperparameter-Logs & W&B Run-IDs

- **Datum / Arbeitszeit:**
- **Git Commit Hash:**
- **PPO Hyperparameter (LR, Clip, Batch Size):**
- **W&B Run ID:**
