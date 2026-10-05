# Tag 03: Action Space & Action Masking

- **Titel & Fokus des Tages:** Diskreter Aktionsraum, Phasengesteuerte Turn-Logik & Strikte Action Masking Pipeline
- **Pflicht-Milestone (Hauptziel):** Implementierung und Validierung einer robusten `action_mask`-Funktion, die 100% aller regelwidrigen Aktionen filtert.

---

## Detaillierte Checkliste der Tagesaufgaben

- [x] Aktionskatalog in `ActionType` präzisieren (ROLL, BUY, PASS, BUILD, MORTGAGE, UNMORTGAGE, PROPOSE_TRADE).
- [x] Funktion `_get_action_mask(agent)` in `src/envs/monopoly_env.py` mit Micro-Phasen (`ROLL`, `BUY_OR_PASS`, `MANAGE_OR_END`) implementieren.
- [x] PyTorch Action-Masking-Integration in `src/utils/action_masking.py` mit `MaskedCategorical` verifizieren.
- [x] Testfall schreiben: Erzwinge Versuch einer illegalen Aktion und prüfe, ob Maske sie blockiert.
- [x] Observation Space Dictionary updaten (`{"observation": Box(144), "action_mask": Box(0, 1, 7, int8)}`).

---

## Präventive Hilfe für HEUTE: Spezifische Hürden & Sofortlösungen

- **Hürde:** All-Zero Mask Bug: Was passiert, wenn keine Aktion legal erscheint?
  - *Lösung:* Sorge dafür, dass `PASS_TURN` als sicherer Default immer legal ist (`mask[PASS_TURN] = 1`), falls keine andere Aktion möglich ist. Das verhindert `-inf`-Crashes in der Softmax-Verteilung.
- **Hürde:** PyTorch `Categorical` wirft NaN bei rein negativen Logits.
  - *Lösung:* Verwende die `MaskedCategorical` Klasse aus `src/utils/action_masking.py`. Sie setzt Masken-False Werte auf $-10^8$, was nach Softmax exakt $0.0$ Wahrscheinlichkeit ergibt, ohne numerisch zu kollabieren.

---

## "Früher fertig?" (Puffer- & Bonus-Tasks)

- [x] Micro-Phasen Zustandsmaschine implementiert, um Zugreihenfolge atomar abzusichern.
- [x] Vollständige Maskierung für alle 7 diskreten Aktionen.

---

## "Rückstand?" (Notfall-Priorisierung)

- **MUSS fertig werden:** Valides binäres Masken-Array für die Basiszüge (Würfeln, Kaufen, Passen).
- **KANN entfallen:** Komplexe Häuserbau-Gleichverteilungsregeln temporär lockern.

---

## Notizen, Hyperparameter-Logs & W&B Run-IDs

- **Status:** **Erfolgreich abgeschlossen**
- **Git Commit:** `6ee50ed`
- **Ergebnis:** Masking zu 100% wasserdicht; 0% illegale Züge bei allen Modellen.
