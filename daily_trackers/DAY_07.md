# Tag 07: Profiling, Refactoring & Env Optimization

- **Titel & Fokus des Tages:** Performance Profiling, NumPy Vektorisierung & Skalierbarkeits-Optimierung
- **Pflicht-Milestone (Hauptziel):** Simulationsdurchsatz des Monopoly-Environments erreicht nach Profiling und Vektorisierung mindestens 500 Steps/Sekunde auf einer CPU-Instanz.

---

## Detaillierte Checkliste der Tagesaufgaben

- [ ] Performance-Profiling mit Python `cProfile` oder `py-spy` durchführen:
  ```bash
  python -m cProfile -o profile.pstats -s cumtime scripts/benchmark_env.py
  ```
- [ ] Bottlenecks identifizieren (z.B. tiefe Dictionary-Kopien, redundante Masken-Berechnungen, String-Lookups).
- [ ] Datenstrukturen refaktorisieren:
  - Property-Lookups von String-Dictionaries auf kompakte 1D-NumPy Arrays umstellen.
  - Spieler-Identifikatoren intern rein integer-basiert (`0..3`) handhaben und nur an der PettingZoo-Schnittstelle als Strings exponieren.
- [ ] Benchmark-Skript schreiben und Vorher/Nachher-FPS dokumentieren.
- [ ] Regressionstests laufen lassen, um sicherzustellen, dass keine Spiellogik gebrochen wurde.

---

## Präventive Hilfe für HEUTE: Spezifische Hürden & Sofortlösungen

- **Hürde:** Speicherallokationen in jedem `step()` bremsen den Python Garbage Collector aus.
  - *Lösung:* Observation- und Info-Buffer vorallokieren und in-place beschreiben, anstatt in jedem Step neue NumPy-Arrays und Dictionaries im Speicher zu erzeugen.
- **Hürde:** Redundante Masken-Checks.
  - *Lösung:* Maskenberechnung nur für den aktuell aktiven Spieler ausführen; für inaktive Spieler ein statisches Singleton-Array `[0, 0, 1, 0, 0, 0, 0]` (nur Pass legal) wiederverwenden.

---

## "Früher fertig?" (Puffer- & Bonus-Tasks)

- [ ] Vektorisierung mehrerer Environments parallel via `AsyncVectorEnv` oder `SyncVectorEnv` testen.
- [ ] Numba JIT-Kompilierung für kernintensive Berechnungen (z. B. Miete-Matrizen) evaluieren.

---

## "Rückstand?" (Notfall-Priorisierung)

- **MUSS fertig werden:** Mindestens 350 Steps/Sekunde, Eliminierung der gröbsten Python-Schleifen.
- **KANN entfallen:** Numba / Multiprocessing-Parallelisierung (NumPy-Optimierung genügt vorerst).

---

## Notizen, Hyperparameter-Logs & W&B Run-IDs

- **Datum / Arbeitszeit:**
- **Git Commit Hash:**
- **FPS Vorher vs. Nachher:**
- **Notizen:**
