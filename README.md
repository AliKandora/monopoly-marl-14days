# Monopoly-MARL — Version 1.0

Ein erstes Python-Lernprojekt für Multi-Agent Reinforcement Learning: vier Agenten spielen in einer **vereinfachten Monopoly-Umgebung**. Enthalten sind Random- und Heuristik-Baselines, parametergeteiltes IPPO, MAPPO mit zentralem Critic, reproduzierbare Auswertung und eine Terminal-Demo.

**Status:** funktionell getesteter Forschungsprototyp, kein vollständiger Monopoly-Simulator und kein Nachweis, dass MAPPO oder IPPO die Heuristik übertreffen. Die ursprüngliche 14-Tage-Roadmap bleibt als Lernplan erhalten; ihre Ziele sind nicht sämtlich implementiert oder validiert.

## Schnellstart

Python 3.10+; getestet mit Python 3.12 auf Windows. Befehle im Repository-Verzeichnis ausführen.

```bash
git clone https://github.com/AliKandora/monopoly-marl-14days.git
cd monopoly-marl-14days
python -m venv .venv
```

Aktivieren auf Windows PowerShell: `.venv\Scripts\Activate.ps1`; auf Linux/macOS: `source .venv/bin/activate`.

```bash
python -m pip install --upgrade pip
# CPU-Version; für CUDA die passende Installation von pytorch.org verwenden.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest tests -q
python demo.py --delay 0 --steps 60
python scripts/evaluate_baselines.py --games 100 --max_turns 150
```

Die Heuristik wird zwischen den vier Sitzen rotiert. Eine Zielquote wie „95 %“ ist kein garantiertes Ergebnis.

## Training und Turnier

```bash
python scripts/train_ippo.py --steps 20000 --seed 42 --save-path checkpoints/ippo_best.pt
python scripts/train_mappo.py --steps 20000 --seed 42 --save-path checkpoints/mappo_best.pt
python scripts/run_tournament.py --games 100 --max-turns 150 --seed 1000
```

`--steps` zählt **Umgebungs-Mikroschritte**, nicht vier künstlich mitgezählte Agentenaktionen. Rollouts werden vollständig gesammelt; der tatsächliche Umfang kann auf das nächste Vielfache von 128 aufgerundet werden. `_best` ist ein historischer Dateiname: gespeichert wird das letzte Modell, nicht ein durch Validierung ausgewähltes bestes Modell.

Das Turnier benötigt beide Checkpoints und bricht bei fehlenden Dateien ab. Es erzeugt JSON mit Seeds, Einzelspielen und SHA-256 der Modelle, eine Tabelle mit Wilson-Intervallen und eine Grafik. Unentschieden werden nicht dem ersten Sitz zugesprochen. Für exakt ausgeglichene Sitze muss die Spielzahl durch vier teilbar sein. Modelle während Training und Evaluation getrennt halten; für belastbare Vergleiche mehrere Trainingsseeds und unabhängige Evaluationsseeds verwenden. Wilson-Intervalle decken die Variabilität zwischen Trainingsläufen nicht ab.

## Unterstützte Regeln

- 40 Felder; Eigentum, Kauf, Miete, vollständige Farbgruppen, Bahnhöfe und Versorger.
- Gleichmäßiger automatischer Haus-/Hotelaufbau, Hypotheken und Rückkauf mit Gebühr.
- Los-Bonus, Steuern, Gefängnisfeld und Freiwürfeln beziehungsweise Pflichtzahlung.
- Vier Spieler; Zugphasen Würfeln, Kaufentscheidung und Verwaltung.
- Action Masks und Prüfung des diskreten Aktionsbereichs; höchstens zwölf Verwaltungsaktionen pro Zug.
- Ende durch Insolvenz oder endlichen Turn-Horizont; bei Timeout entscheidet das Vermögen, Gleichstände bleiben erhalten.

## Bewusste Vereinfachungen und offene Ausbauziele

Keine Ereignis-/Gemeinschaftskarten, Auktionen, freiwillige Gefängniszahlung, Zusatzwürfe bei Pasch, Drei-Pasch-Regel, begrenzter Haus-/Hotelvorrat oder Notverkäufe vor Insolvenz. Negatives Bargeld bedeutet sofortige Insolvenz; Vermögen geht an die Bank, nicht an den Gläubiger. Die Miet- und Vermögensmodelle sind Forschungsvereinfachungen, keine zertifizierte Umsetzung offizieller Regeln.

`PROPOSE_TRADE` bleibt als reservierte Aktion erhalten, ist aber deaktiviert: der frühere Platzhalter zwang Gegner zum Kauf und war keine Verhandlung. Handel benötigt einen eigenen Vorschlags-/Zustimmungsmechanismus. Grundstücksauswahl bei Bauen und Hypotheken erfolgt deterministisch, nicht durch den Agenten.

Die Parallel-API transportiert vier Aktionen, führt aber nur die Aktion des aktuellen Spielers aus. Inaktive lebende Spieler geben eine maskierte Pass-Aktion ab; ausgeschiedene Spieler werden aus Trainings-Minibatches entfernt. Alle Actor-Beobachtungen enthalten öffentlich sichtbare Board- und Spielerinformationen; „dezentral“ bedeutet hier Ausführung ohne zentralen Critic, **nicht** partielle Beobachtbarkeit. IPPO teilt Actor/Critic-Parameter zwischen Spielern. MAPPO nutzt denselben Actor-Stil und einen globalen Critic mit Spielerkennung.

Der GAE-Puffer behandelt den endlichen Horizont als Episodenende ohne Bootstrap. Das ist eine dokumentierte finite-horizon-Konvention, keine Implementierung von Time-Limit-Bootstrapping für unendliche Aufgaben. Die Begrenzung von Verwaltungsaktionen ist eine technische Schutzregel; ihre verbleibende Anzahl ist derzeit kein eigenes Beobachtungsfeature.

Opponent Pool existiert als Hilfsklasse, ist aber nicht an den Trainingsloop angeschlossen. W&B-Tracking, Elo-Liga, systematisches Hyperparameter-Tuning und ein statistischer Forschungsnachweis bleiben offen. Keine Langzeittrainings- oder Dominanzbehauptung folgt aus den Smoke-Tests.

## Aufbau

```text
src/envs/             Spielzustand, Regeln und Action Masks
src/agents/           Baselines, Netze, PPO-Puffer und Checkpoints
src/utils/            Features, Rewards und Evaluation
src/visualization/    Diagramme
scripts/              Training, Turnier und Benchmark
 tests/               API-, Regel- und Regressionstests
 daily_trackers/      ursprünglicher 14-Tage-Lernplan
```

- [Abschlussprüfung und Änderungen](EVALUATION.md)
- [Lernleitfaden](STUDY_GUIDE.md)
- [Formeln und Konzepte](CHEATSHEET_CORE.md)
- [Fehlerbehebung](TROUBLESHOOTING_PREVENTIVE.md)
- [Historische Ergebnisse: nicht reproduzierbar belegt](results/tournament_summary.md)

## Tests und Wartung

`python -m pytest tests -q` prüft beide PettingZoo-API-Wege, reproduzierbare Seeds, Aktionsvalidierung, Episodenende, Insolvenz, GAE, Ausschluss ausgeschiedener Spieler, Checkpoint-Roundtrip und kurze IPPO/MAPPO-Trainingsläufe. Die GitHub-Actions-Konfiguration führt diese Suite bei Push/PR aus. Zwei Hinweise des AEC-Tests über Dict-Beobachtungen sind erwartbar, da Action Masks Teil der Observation sind.

PyTorch-Checkpoints nur aus vertrauenswürdiger Herkunft laden; der Loader nutzt `weights_only=True`. Ergebnisse aus langen Läufen gehören mit Konfiguration, Seeds und Modell-Hashes dokumentiert. Im Repository ist noch keine Open-Source-Lizenz gewählt; öffentlich lesbarer Code ist nicht automatisch uneingeschränkt nachnutzbar.
