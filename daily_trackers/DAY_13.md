# Tag 13: Evaluation & Statistische Analyse

- **Titel & Fokus des Tages:** Rigorose Benchmarking-Matrix, Statistische Signifikanz & Strategie-Profiling
- **Pflicht-Milestone (Hauptziel):** Erstellung einer vollständigen Kreuztabelle (Win-Rate & Elo Matrix) aller entwickelten Agenten (Random, Heuristik, IPPO, MAPPO) über mindestens 500 Test-Spiele.

---

## Detaillierte Checkliste der Tagesaufgaben

- [ ] Evaluations-Harness in `scripts/run_tournament.py` aufsetzen:
  - Lässt alle Modellvarianten im Round-Robin-Modus gegeneinander antreten (je 100 Partien pro Paarung/Vierer-Konstellation).
  - Tauscht Spieler-Sitzpositionen gleichmäßig durch (um First-Player-Bias zu eliminieren).
- [ ] Erfassung und Aggregation statistischer Kennzahlen:
  - Win-Rate (%) mit 95%-Konfidenzintervall (Wilson-Score-Intervall).
  - Mittlere Rundenanzahl bis zum Bankrott der Gegner.
  - Mittleres Vermögen (Net-Worth) bei Spielende.
  - Bevorzugte Farbstraßen der RL-Agenten vs. Heuristik.
- [ ] Visualisierungs-Skripte in `src/visualization/` finalisieren:
  - Win-Rate Matrix Heatmap mit Seaborn / Matplotlib.
  - Vermögensverlaufs-Graphen über typische 100 Runden.
- [ ] Automatischer Export der Tabellen nach Markdown und CSV (`results/`).

---

## Präventive Hilfe für HEUTE: Spezifische Hürden & Sofortlösungen

- **Hürde:** Zu lange Laufzeit des 500-Spiele Turniers.
  - *Lösung:* Schalte Rendering strikt aus, nutze Python `multiprocessing.Pool` über 4 CPU-Kerne, und setze `max_turns = 150` mit automatischem Net-Worth-Entscheid bei Timeouts.
- **Hürde:** Signifikanz-Verzerrung durch Würfelglück.
  - *Lösung:* Paired Testing: Nutze denselben Satz an Pseudo-Random-Seeds für unterschiedliche Algorithmen, um identische Würfel-Sequenzen zu testen.

---

## "Früher fertig?" (Puffer- & Bonus-Tasks)

- [ ] Feature Importance / Shapley-Value-Analyse der Policy-Inputs (welche Board-Features beeinflussen Kaufentscheidungen am stärksten?).
- [ ] PDF-Report-Generator für Turnierergebnisse.

---

## "Rückstand?" (Notfall-Priorisierung)

- **MUSS fertig werden:** 100 Spiele MAPPO vs. Heuristik vs. Random mit dokumentierter Win-Rate-Tabelle.
- **KANN entfallen:** Umfassendes 500-Spiele Turnier aller Zwischen-Checkpoints.

---

## Notizen, Hyperparameter-Logs & W&B Run-IDs

- **Datum / Arbeitszeit:**
- **Git Commit Hash:**
- **Finale Win-Rates:**
  - MAPPO vs Heuristik:
  - IPPO vs Heuristik:
  - MAPPO vs IPPO:
- **Notizen:**
