# Entscheidungsübersicht — Abschluss Version 1.0

Stand: 8. Oktober 2026. Repository: `AliKandora/monopoly-marl-14days`; Zielbranch: `main`.

## Ergebnis

Der vereinbarte funktionelle Prototyp wurde fertiggestellt und direkt auf `main` bereitgestellt. 28 lokale Tests bestanden; beide PPO-Varianten wurden über je 2.048 Umgebungs-Mikroschritte geprüft. Der erste GitHub-Actions-Lauf auf Linux war ebenfalls erfolgreich: [Testlauf 37797157724](https://github.com/AliKandora/monopoly-marl-14days/actions/runs/37797157724), Commit `5925aa1d7696020a419551c5fd7c038e428a1c49`.

## Von dir ausdrücklich festgelegt

| Entscheidung | Festlegung |
|---|---|
| Abschlussumfang | Empfohlene, getestete Version 1.0 statt vollständiger Umsetzung aller 14 Roadmap-Tage |
| Bereitstellung | GitHub verbinden und direkt pushen |
| Ziel | `AliKandora/monopoly-marl-14days`, Branch `main` |
| Testumgebung | Installation einer getrennten CPU-PyTorch-Testumgebung erlaubt |
| Workflow-Fallback | Fehlenden Workflow über die GitHub-Webseite speichern erlaubt |
| Weitere Entscheidungen | Autonom im bestehenden Umfang weiterarbeiten; Entscheidungen dokumentieren; bei notwendiger Mitwirkung warten |

## Meine technischen Entscheidungen

| Entscheidung | Warum | Später änderbar durch |
|---|---|---|
| GAE-Episodengrenzen und Bootstrap-Zustände korrigieren | Vorher konnte Lernen Episoden vermischen und falsche Folgezustände bewerten | Nur mit neuer RL-Konvention und entsprechenden Tests |
| Endlichen Turn-Horizont als echtes Episodenende behandeln | Konsistenter, begrenzter Forschungsprototyp | Time-Limit-Bootstrapping mit Speicherung finaler Zustände implementieren |
| Ausgeschiedene Spieler aus Minibatches ausschließen | Keine fiktiven Nullzustände trainieren | Einen expliziten absorbing-state-Ansatz separat modellieren |
| Tatsächliche Umgebungs-Mikroschritte zählen | Vier Eingaben bedeuteten nicht vier ausgeführte Züge | Andere Zählkonvention ausdrücklich benennen |
| Lokalen Zufallsgenerator je Umgebung verwenden | Unabhängige, reproduzierbare Simulationen | Keine globale RNG-Abhängigkeit wieder einführen |
| Verwaltungsaktionen auf zwölf pro Zug begrenzen | Verhindert Endlosschleifen durch Hypotheken/Rückkauf | Limit anpassen oder Budget als Beobachtungsfeature ergänzen |
| Handelsplatzhalter deaktivieren; Aktionsdimension erhalten | Gegner wurden zuvor ohne Zustimmung zum Kauf gezwungen; kompatible Netzdimension beibehalten | Bilaterale Vorschlags- und Annahmephasen entwickeln |
| Fehlende Modelle im Turnier als Fehler behandeln | Keine untrainierten Netze als trainierte PPO-Agenten bewerten | Expliziten, separat beschrifteten Untrained-Baseline-Modus ergänzen |
| Seeds, Modell-Hashes, Einzelspiele und Wilson-Intervalle speichern | Ergebnisse nachvollziehbar machen | Zusätzlich Mehrseed-Statistik und Lernkurven ergänzen |
| Sieger aus überlebenden Net-Worth-Leadern bestimmen; Gleichstände beibehalten | Kein falscher Gewinner durch alte Termination-Flags oder Sitzvorteil bei Gleichstand | Andere, dokumentierte Turnierwertung wählen |
| Deterministische Aktionswahl tatsächlich implementieren | Das vorhandene Argument wurde ignoriert | Stochastische Evaluation bleibt Standard im Turnier |
| Checkpoints mit `weights_only=True` laden | Unnötiges Pickle-Risiko reduzieren | Nur begründet und niemals für untrusted Modelle erweitern |
| Bargeldreserve der Heuristik nach Kaufkosten prüfen | Vorher war die behauptete Reserve nicht garantiert | Cash-Sicherheitsmarge verändern |
| Nur kurze Trainings- und Turnierchecks ausführen | Funktion prüfen, ohne lange Rechenarbeit oder angebliche Konvergenz | Separate längere Forschungsstudie starten |
| Historische Resultate erhalten und warnend kennzeichnen | Keine Herkunft löschen; alte 85 % sind nicht reproduzierbar belegt | Bei vorhandenen Originalmodellen und Seeds nachvalidieren |
| Regelumfang im README ehrlich einschränken | Karten, Auktionen, vollständige Insolvenz und Verhandlung fehlen | Regelmodule gezielt ergänzen und testen |
| CPU-PyTorch und isolierte lokale Umgebung | Reproduzierbare Tests, kein Eingriff ins übrige Python-Setup | GPU-Umgebung separat einrichten |
| W&B nicht als Pflichtabhängigkeit behalten | Kein angeschlossener W&B-Trainingspfad vorhanden | Monitoring bewusst implementieren und optional installieren |
| Paketbereiche plus getesteten Snapshot dokumentieren | Praktischer Einstieg und nachvollziehbarer Teststand | Strikten Lockfile-Workflow ergänzen |
| Workflow zuletzt hochladen | Während der dateiweisen API-Commits keine unvollständigen Zwischenstände automatisch testen | Künftige Änderungen vorzugsweise atomar oder per PR machen |
| Workflow über Browser speichern | Die App wies ausschließlich den Workflow-Schreibzugriff mit 404 ab; genaue Ursache unbestätigt | Workflow-Berechtigung der Integration später prüfen |
| Workflow in gültiger JSON-/YAML-Flow-Notation speichern | GitHub-Webeditor rückte mehrzeilige Eingaben automatisch falsch ein; Inhalt vor Commit exakt verglichen | In übliches mehrzeiliges YAML umformatieren, ohne Semantik zu ändern |
| Keine Open-Source-Lizenz selbst wählen | Lizenz ist eine bewusste Eigentümerentscheidung | Du wählst z. B. MIT oder Apache-2.0 nach Prüfung |
| Keine zusätzlichen Langzeitexperimente nach deinem Weggehen starten | Autonomie ersetzt nicht den festgelegten Projektumfang | Separate Forschungsstufe mit Budget und Ziel beginnen |

## Was bewusst nicht als fertig behauptet wird

- Vollständiger Monopoly-Regelumfang, Ereigniskarten und Auktionen.
- Echte Handelsverhandlungen, integrierte Opponent-Pool-Liga, Elo und W&B-Tuning.
- Überlegenheit von IPPO/MAPPO oder abgeschlossene statistische Forschungsarbeit.
- Ein als „bestes“ validiertes Modell: `*_best.pt` sind historische Dateinamen für das letzte Modell.
- Uneingeschränkte Nachnutzbarkeit ohne Lizenzwahl.

## Nachvollziehen und Änderungen zurücknehmen

Ausgangscommit vor meinen Änderungen: `c93c328dffbe972f64d2753195eb0e1a29832d9d`. Die App konnte nur einzelne Dateien pro Commit schreiben; daher entstanden mehrere nachvollziehbare Dateicommits statt eines atomaren Gesamtcommits. Vor Abschluss wurden die ersten 20 Dateien über Git-Blob-Hashes geprüft, danach der Workflow separat verifiziert.

Für spätere Änderungen einzelne Entscheidungen und Dateien gezielt bearbeiten oder betroffene Commits mit `git revert` rückgängig machen. Kein Force-Push und kein Zurücksetzen von `main` ohne vorherige Prüfung: dadurch könnten spätere eigene Änderungen verloren gehen.

Die lokale Testumgebung und Smoke-Modelle liegen in der Arbeitskopie dieses Chats, nicht im GitHub-Repository. Einstieg und Reproduktionsbefehle: [README](README.md). Fachliche Bewertung: [EVALUATION](EVALUATION.md). Prüfnachweise: [results/validation.json](results/validation.json).
