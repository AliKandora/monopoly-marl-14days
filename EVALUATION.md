# Evaluation und Abschluss Version 1.0

Prüfung am 8. Oktober 2026. Ausgangspunkt: Commit `c93c328dffbe972f64d2753195eb0e1a29832d9d`.

## Gesamturteil

Für ein erstes GitHub-Projekt ist dies eine ambitionierte, sinnvoll modularisierte Lernbasis. Besonders positiv: getrennte Umgebung/Agenten/Features, Action Masking, zwei PPO-Varianten, Baselines, Demo und umfangreiche Lernunterlagen. Der ursprüngliche Stand war jedoch ein Prototyp mit methodischen Fehlern, nicht ein abgeschlossener Forschungsnachweis. Dokumentation und Roadmap versprachen mehr Regelumfang, Verhandlung und wissenschaftliche Validierung als implementiert war.

Version 1.0 schließt den **vereinbarten funktionellen Prototyp-Umfang** ab: getestetes Training, begrenzte Simulation, nachvollziehbare Evaluation, Anleitung und Regressionstests. Vollständige Monopoly-Regeln, eine Trading-Liga und statistisch belegte Überlegenheit sind ausdrücklich kein Bestandteil dieses Abschlusses.

## Nachgewiesene Probleme und Korrekturen

| Bereich | Ausgangsproblem | Änderung |
|---|---|---|
| GAE | `dones[t+1]` statt des terminalen Übergangs `dones[t]` | Episoden werden korrekt getrennt; numerischer Regressionstest |
| Rollout-Bootstrap | Critic bewertete am Ende den vorherigen Zustand | Nächste Observation bzw. aktueller globaler Zustand |
| IPPO Global State | Nach dem Schritt gespeichert | Zustand vor dem Schritt gespeichert |
| Tote Spieler | Fiktive Nullzustände in PPO-Minibatches | Valid-Maske schließt ausgeschiedene Spieler aus |
| Schrittzähler | Vier Schritte gezählt, obwohl nur eine Aktion ausgeführt wird | Tatsächliche Umgebungs-Mikroschritte |
| Zufall | `env.reset()` setzte globales NumPy-RNG | Lokaler Generator je Umgebung; Trainingsseed für NumPy/Torch |
| Endlosschleifen | Wiederholte Verwaltung konnte einen Zug unbegrenzt halten | Maximal zwölf Verwaltungsaktionen pro Zug |
| Action Validation | Negative Indizes oder Indexfehler möglich | Explizite Typ-/Bereichs- und Vollständigkeitsprüfung |
| Handel | Automatischer Verkauf ohne Zustimmung | Platzhalter deaktiviert, keine falsche Verhandlungsbehauptung |
| Turnier | Fehlende Checkpoints wurden still ignoriert | Expliziter Abbruch; Checkpoint-Hashes und Seeds im Bericht |
| Sieger | Episodenflags wurden nicht über alle Schritte akkumuliert; Gleichstand zugunsten erster Sitz | Überlebende nach Spielzustand; Net-Worth-Gleichstände explizit |
| Checkpoints | Deterministic-Argument ignoriert | Maskiertes Argmax; `weights_only=True` beim Laden |
| Heuristik | Reserve vor statt nach Kauf geprüft | Kaufkosten von verfügbarer Reserve abziehen |
| Dokumentation | Roadmap und Ergebnisse wirkten wie garantierte Leistungen | Unterstützte Regeln, offene Ziele und historische Evidenz getrennt |

## Validierung

Lokales System: Windows, Python 3.12.7, CPU-PyTorch. Konkrete Paketversionen: [requirements-tested.txt](requirements-tested.txt). Diese Datei dokumentiert den Teststand, nicht alle unterstützten Versionskombinationen. Für den CPU-Build beim Installieren `--extra-index-url https://download.pytorch.org/whl/cpu` verwenden.

- Vollständige Pytest-Suite: **28 Tests bestanden**. Zwei erwartbare PettingZoo-AEC-Hinweise betreffen Dict-Observations mit Action Mask.
- Native Parallel-API und AEC-Konvertierung geprüft.
- IPPO und MAPPO je 2.048 Umgebungs-Mikroschritte mit Seed 42 erfolgreich trainiert und gespeichert. Kurze Tests prüfen zusätzlich endliche Modellparameter und Checkpoint-Roundtrip.
- Turnier: acht Spiele, 50 Turns, Seed 100; gleiche Checkpoints und Seeds ergeben beim Wiederholen identische JSON-Daten. Details in [results/smoke/tournament.json](results/smoke/tournament.json).
- Demo mit 20 Schritten und ohne Wartezeit erfolgreich ausgeführt; Baseline-Skript mit acht Spielen erfolgreich ausgeführt.
- Quellcode mit `compileall` geprüft.
- GitHub-Actions-Workflow hinzugefügt und erfolgreich auf Linux ausgeführt: [Lauf 37797157724](https://github.com/AliKandora/monopoly-marl-14days/actions/runs/37797157724), Commit `5925aa1d7696020a419551c5fd7c038e428a1c49`. Alle Job-Schritte erfolgreich.
- Entscheidungen und ihre Änderungsmöglichkeiten: [DECISIONS.md](DECISIONS.md).

Reproduktion der funktionellen Turnierprüfung:

```bash
python scripts/train_ippo.py --steps 2048 --seed 42 --save-path checkpoints/ippo_smoke.pt
python scripts/train_mappo.py --steps 2048 --seed 42 --save-path checkpoints/mappo_smoke.pt
python scripts/run_tournament.py --games 8 --max-turns 50 --seed 100 --ippo-path checkpoints/ippo_smoke.pt --mappo-path checkpoints/mappo_smoke.pt --results-dir results/smoke
```

Die Smoke-Checkpoints liegen lokal, werden nicht als fertige starke Modelle ins Repository aufgenommen. Die acht Spiele sind ein Funktionstest, **kein** aussagekräftiger Algorithmusvergleich. Die historischen 85 % IPPO-Siege bleiben archiviert, sind aber mangels Checkpoints/Seeds nicht reproduzierbar belegt und stammen aus dem fehlerhaften Evaluationspfad.

## Offene Grenzen

Siehe [README](README.md) für die konkrete Regelliste. Besonders wichtig: keine Karten, Auktionen oder echten Verhandlungen; vereinfachte Insolvenz; finite-horizon- statt Time-Limit-Bootstrapping; Shared-Parameter-IPPO; öffentliches Board im Actor; Opponent Pool noch nicht im Trainingsloop; kein W&B-Tuning oder Elo-Experiment. Die Beobachtung enthält die verbleibende Verwaltungsaktionszahl nicht separat. Eine Open-Source-Lizenz muss der Eigentümer noch bewusst wählen.

## Sinnvolle nächste Forschungsstufe

1. Drei oder mehr Trainingsseeds, länger trainieren und Konfigurationen/Lernkurven speichern.
2. Unabhängige Evaluationsseeds, sitzbalancierte Spiele und separate Tests gegen feste Baselines.
3. Ein Regelmodul nach dem anderen mit gezielten Tests ergänzen; danach bilateralen Handel als eigene Entscheidungsphase modellieren.
4. Erst anhand dieser Daten beurteilen, ob MAPPO tatsächlich einen Vorteil besitzt.
