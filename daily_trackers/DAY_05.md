# Tag 05: Baselines (Random & Heuristik)

- **Titel & Fokus des Tages:** Stochastische & Deterministische Baseline-Agenten & Evaluierungs-Harness
- **Pflicht-Milestone (Hauptziel):** Der regelbasierte `HeuristicAgent` schlägt 3 `RandomAgent`-Gegner in 100 Test-Spielen mit einer Win-Rate von mindestens 95%.

---

## Detaillierte Checkliste der Tagesaufgaben

- [ ] `RandomAgent` (`src/agents/random_agent.py`) finalisieren: Mask-konforme, gleichverteilte Aktionsauswahl.
- [ ] `HeuristicAgent` (`src/agents/heuristic_agent.py`) optimieren:
  - Kaufe alle verfügbaren Straßen, wenn Cash-Puffer $\ge \$200$ bleibt.
  - Baue Häuser prioritär auf Monopolen, wenn Cash $\ge \$400$.
  - Gehe nur in Hypothek, wenn akute Zahlungsunfähigkeit droht.
- [ ] Evaluierungs-Skript `scripts/evaluate_baselines.py` erstellen:
  - Führe 100 Spiele mit 1 HeuristicAgent vs. 3 RandomAgents durch.
  - Erfasse Sieg-Wahrscheinlichkeiten, durchschnittliche Rundenanzahl bis zum Sieg und Endvermögen.
- [ ] Win-Rate statistisch auswerten und Konsolen-Report ausgeben.

---

## Präventive Hilfe für HEUTE: Spezifische Hürden & Sofortlösungen

- **Hürde:** Heuristik-Bot erreicht nur 70-80% Win-Rate gegen Random.
  - *Lösung:* Überprüfe das Häuserbau-Verhalten. In Monopoly gewinnt nicht der mit dem meisten Bargeld, sondern derjenige, der schnell Häuser baut (Mietsteigerung von \$10 auf \$500+). Senke den Spar-Puffer des Heuristik-Bots, wenn er ein Monopol besitzt.
- **Hürde:** Match-Deadlocks (alle Spieler passen, keine Aktionen finden statt).
  - *Lösung:* Erzwinge Würfeln als Priorität 1 zu Rundenbeginn; `PASS_TURN` darf erst nach Durchführung aller Aktionen des Zuges erlaubt sein.

---

## "Früher fertig?" (Puffer- & Bonus-Tasks)

- [ ] Aggressiven Heuristik-Bot ("Property Tycoon") vs. defensiven Heuristik-Bot ("Cash Hoarder") gegeneinander antreten lassen.
- [ ] Matplotlib Balkendiagramm der Gewinnverteilung in `src/visualization/` generieren.

---

## "Rückstand?" (Notfall-Priorisierung)

- **MUSS fertig werden:** Lauffähiger `RandomAgent` und `HeuristicAgent`, Mindest-Win-Rate > 85% über 50 Episoden.
- **KANN entfallen:** Feintuning unterschiedlicher Heuristik-Varianten.

---

## Notizen, Hyperparameter-Logs & W&B Run-IDs

- **Datum / Arbeitszeit:**
- **Git Commit Hash:**
- **Win-Rate Heuristik vs. Random (100 Spiele):**
- **Notizen:**
