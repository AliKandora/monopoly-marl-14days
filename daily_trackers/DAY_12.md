# Tag 12: Advanced Self-Play & Opponent Pool

- **Titel & Fokus des Tages:** Self-Play League Training, Opponent Sampling & Vermeidung von Policy Cyclicity
- **Pflicht-Milestone (Hauptziel):** Aufbau eines Opponent Pools mit eingefrorenen Checkpoints früherer Epochen; stabiles Training gegen diversifizierte Gegner-Policies.

---

## Detaillierte Checkliste der Tagesaufgaben

- [ ] Opponent-Pool-Manager `src/agents/opponent_pool.py` implementieren:
  - Speichert alle $K$ Epochen einen Snapshot der aktuellen Policy-Gewichte ab.
  - Hält ein Reservoir der letzten $M$ Modell-Checkpoints vor.
- [ ] Matchmaking-Logik für 4 Spieler:
  - 1 Slot: Aktuell lernende Policy.
  - 2 Slots: Zufällig gezogene historische Checkpoints aus dem Pool.
  - 1 Slot: HeuristicAgent als verankernder "Anchor-Bot" gegen Strategiedrift.
- [ ] Self-Play Trainingslauf über 2 Millionen Total Steps initiieren.
- [ ] Überwachung der Policy-Entropie (Verhinderung von vorzeitigem Strategiekollaps).

---

## Präventive Hilfe für HEUTE: Spezifische Hürden & Sofortlösungen

- **Hürde:** Catastrophic Forgetting / Strategy Cycling (Policy $A$ schlägt $B$, $B$ schlägt $C$, $C$ schlägt $A$, ohne echten Fortschritt).
  - *Lösung:* Der statische `HeuristicAgent` im Matchmaking dient als unbestechlicher Benchmark ("Anchor"). Nur Checkpoints, die den Anchor-Bot mit $\ge 80\%$ schlagen, werden in den offiziellen League-Pool aufgenommen.
- **Hürde:** Hoher GPU/CPU-Overhead durch ständiges Neuladen von Checkpoints.
  - *Lösung:* Checkpoints als PyTorch Module im RAM cachen und per State-Dict referenzieren, anstatt Dateien im Disk-IO-Takt nachzuladen.

---

## "Früher fertig?" (Puffer- & Bonus-Tasks)

- [ ] Elo-Rating-System für Checkpoints implementieren (`scripts/compute_elo.py`).
- [ ] Heatmap der Siegchancen (Generation $X$ vs. Generation $Y$) generieren.

---

## "Rückstand?" (Notfall-Priorisierung)

- **MUSS fertig werden:** Einfacher Pool mit den letzten 3 Checkpoints (FIFO) + Heuristic Bot.
- **KANN entfallen:** Komplexe dynamische League-Sampling-Algorithmen (AlphaStar-Stil).

---

## Notizen, Hyperparameter-Logs & W&B Run-IDs

- **Datum / Arbeitszeit:**
- **Git Commit Hash:**
- **Größe des Opponent Pools:**
- **W&B Run ID:**
