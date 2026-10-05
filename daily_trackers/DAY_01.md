# Tag 01: PettingZoo API & Dev-Setup

- **Titel & Fokus des Tages:** PettingZoo Parallel API Architektur & Minimales Dummy Environment
- **Pflicht-Milestone (Hauptziel):** Lauffähiges PettingZoo Parallel Environment mit fehlerfreiem `reset()`- und `step()`-Zyklus für 4 Agenten.

---

## Detaillierte Checkliste der Tagesaufgaben

- [x] Python Virtual Environment (`.venv`) aufsetzen und Abhängigkeiten aus `requirements.txt` installieren.
- [x] Git-Repository initialisieren und Remote-Verknüpfung zu GitHub herstellen.
- [x] PettingZoo `ParallelEnv`-Basisklasse implementieren (`src/envs/monopoly_env.py`).
- [x] `reset()`-Methode so konfigurieren, dass sie initiale Beobachtungen für alle 4 Agenten (`player_0` bis `player_3`) liefert.
- [x] Aktionsverarbeitung in `step()` implementieren (Würfeln, Kaufen, Turn-Fortschaltung).
- [x] Termination- und Truncation-Flags für Episodenende definieren (`max_turns` Limit).
- [x] Unit-Test in `tests/test_env.py` ausführen und sicherstellen, dass alle Tests grün durchlaufen.

---

## Präventive Hilfe für HEUTE: Spezifische Hürden & Sofortlösungen

- **Hürde:** PettingZoo API-Versionierung (Dict vs. Tuple Return in Gym/Gymnasium).
  - *Lösung:* PettingZoo $\ge 1.24.0$ verwendet `reset()` mit `(obs, infos)` und `step()` mit 5 Rückgabewerten `(obs, rewards, terminations, truncations, infos)`. Achte darauf, dass alle Dictionaries dieselben aktiven Keys (`env.agents`) enthalten.
- **Hürde:** Typinkonsistenzen in NumPy Arrays im Observation Space.
  - *Lösung:* Observation Space strikt als `np.float32` deklarieren und in `_get_obs()` explizit mit `.astype(np.float32)` zurückgeben.

---

## "Früher fertig?" (Puffer- & Bonus-Tasks)

- [x] Textbasierte `render()`-Methode mit ANSI-Farben zur optischen Inspektion im Terminal ergänzen.
- [x] PettingZoo Konformitätstest in `tests/test_env.py` erfolgreich integriert.

---

## "Rückstand?" (Notfall-Priorisierung)

- **MUSS fertig werden:** Das Environment muss deterministisch initialisieren, einen Dummy-Step ohne Crash ausführen und `obs, rewards, term, trunc, info` zurückgeben.
- **KANN entfallen:** Ausgefeiltes Rendering und `infos`-Zusatzmetriken auf Tag 02 verschieben.

---

## Notizen, Hyperparameter-Logs & W&B Run-IDs

- **Status:** **Erfolgreich abgeschlossen**
- **Git Commit:** `7c1fbcf` / `6ee50ed`
- **Ergebnis:** PettingZoo ParallelEnv initialisiert 4 Agenten mit 144-dim Observation Space und 7 diskreten Aktionen.
