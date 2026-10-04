# Tag 10: Hyperparameter Tuning & Logging

- **Titel & Fokus des Tages:** Experiment-Tracking mit Weights & Biases (W&B) & Systematisches Hyperparameter-Tuning
- **Pflicht-Milestone (Hauptziel):** Vollständig instrumentiertes Dashboard in W&B mit Live-Tracking von Win-Rates, Cumulative Rewards, Episode Lengths, Value/Policy Losses und Hyperparameter-Sweeps.

---

## Detaillierte Checkliste der Tagesaufgaben

- [ ] W&B Logger in Trainingsskripte integrieren (`wandb.init(project="monopoly-marl", config=...)`).
- [ ] Tracking-Metriken definieren:
  - `train/actor_loss`, `train/critic_loss`, `train/entropy`, `train/approx_kl`
  - `game/episode_length`, `game/bankruptcies_per_episode`
  - `eval/win_rate_vs_heuristic`, `eval/mean_net_worth`
- [ ] Sweep-Konfiguration (`sweep.yaml`) anlegen für Schlüsselparameter:
  - Learning Rate: $[1\cdot 10^{-4}, 3\cdot 10^{-4}, 5\cdot 10^{-4}]$
  - Entropy Coeff: $[0.005, 0.01, 0.02]$
  - GAE Lambda: $[0.90, 0.95, 0.98]$
  - Clip Range: $[0.1, 0.2]$
- [ ] Mindestens 4 Sweep-Runs über Nacht starten und Ergebnisse vergleichend auswerten.

---

## Präventive Hilfe für HEUTE: Spezifische Hürden & Sofortlösungen

- **Hürde:** W&B synchronisiert zu viele Daten und bremst den Trainingsloop aus.
  - *Lösung:* Logge Trainingsverluste nicht bei jedem Step, sondern nur einmal pro Update-Epoche (`log_freq = 100`). Berechne Evaluationsmetriken separat alle 50 Episoden.
- **Hürde:** Offline-Umgebung oder fehlender API-Key blockiert Script.
  - *Lösung:* W&B Fallback-Modus unterstützen: `wandb.init(mode="disabled" if args.no_wandb else "online")` und paralleles Schreiben von TensorBoard-Logs.

---

## "Früher fertig?" (Puffer- & Bonus-Tasks)

- [ ] Automatisches Speichern des besten Modells als W&B Model Artifact.
- [ ] Diagramm mit Korrelation zwischen Häuserbau-Häufigkeit und Endplatzierung erstellen.

---

## "Rückstand?" (Notfall-Priorisierung)

- **MUSS fertig werden:** Stabiles W&B / TensorBoard Logging der Kern-Metriken für 1 Referenz-Run.
- **KANN entfallen:** Umfassender Bayes'scher Hyperparameter-Sweep (manueller Test von 2-3 Konfigurationen genügt).

---

## Notizen, Hyperparameter-Logs & W&B Run-IDs

- **Datum / Arbeitszeit:**
- **Git Commit Hash:**
- **Beste gefundene Konfiguration:**
- **W&B Sweep URL:**
