# Tag 01: PettingZoo API & Dev-Setup

- **Titel & Fokus des Tages:** PettingZoo Parallel API Architektur & Minimales Dummy Environment
- **Pflicht-Milestone (Hauptziel):** Lauffähiges PettingZoo Parallel Environment mit fehlerfreiem `reset()`- und `step()`-Zyklus für 4 Agenten.

---

## Detaillierte Checkliste der Tagesaufgaben

- [ ] Python Virtual Environment (`.venv`) aufsetzen und Abhängigkeiten aus `requirements.txt` installieren.
- [ ] Git-Repository überprüfen und ersten Feature-Branch anlegen (`git checkout -b day-01-env-scaffold`).
- [ ] PettingZoo `ParallelEnv`-Basisklasse implementieren (`src/envs/monopoly_env.py`).
- [ ] `reset()`-Methode so konfigurieren, dass sie initiale Beobachtungen für alle 4 Agenten (`player_0` bis `player_3`) liefert.
- [ ] Dummy-Aktionsverarbeitung in `step()` implementieren (z. B. Dummy-Würfeln und Turn-Fortschaltung).
- [ ] Termination- und Truncation-Flags für Episodenende definieren (z. B. Max Steps Limit = 100).
- [ ] Unit-Test in `tests/test_env.py` ausführen und sicherstellen, dass Pytest mit Status grün durchläuft.

---

## Präventive Hilfe für HEUTE: Spezifische Hürden & Sofortlösungen

- **Hürde:** PettingZoo API-Versionierung (Dict vs. Tuple Return in Gym/Gymnasium).
  - *Lösung:* PettingZoo $\ge 1.24.0$ verwendet `reset()` mit `(obs, infos)` und `step()` mit 5 Rückgabewerten `(obs, rewards, terminations, truncations, infos)`. Achte darauf, dass alle Dictionaries dieselben aktiven Keys (`env.agents`) enthalten.
- **Hürde:** Typinkonsistenzen in NumPy Arrays im Observation Space.
  - *Lösung:* Observation Space strikt als `np.float32` deklarieren und in `_get_obs()` explizit mit `.astype(np.float32)` zurückgeben.

---

## "Früher fertig?" (Puffer- & Bonus-Tasks)

- [ ] Textbasierte `render()`-Methode mit ANSI-Farben zur optischen Inspektion im Terminal ergänzen.
- [ ] PettingZoo `api_test` aus `pettingzoo.test` importieren und das Environment gegen offizielle API-Spezifikationen prüfen.

---

## "Rückstand?" (Notfall-Priorisierung)

- **MUSS fertig werden:** Das Environment muss deterministisch initialisieren, einen Dummy-Step ohne Crash ausführen und `obs, rewards, term, trunc, info` zurückgeben.
- **KANN entfallen:** Ausgefeiltes Rendering und `infos`-Zusatzmetriken auf Tag 02 verschieben.

---

## Notizen, Hyperparameter-Logs & W&B Run-IDs

- **Datum / Arbeitszeit:**
- **Git Commit Hash:**
- **Notizen:**
