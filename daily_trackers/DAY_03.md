# Tag 03: Action Space & Action Masking

- **Titel & Fokus des Tages:** Diskreter Aktionsraum, Phasengesteuerte Turn-Logik & Strikte Action Masking Pipeline
- **Pflicht-Milestone (Hauptziel):** Implementierung und Validierung einer robusten `action_mask`-Funktion, die 100% aller regelwidrigen Aktionen filtert.

---

## Detaillierte Checkliste der Tagesaufgaben

- [ ] Aktionskatalog in `ActionType` präzisieren (z.B. ROLL, BUY, PASS, BUILD, MORTGAGE).
- [ ] Funktion `_get_action_mask(agent)` in `src/envs/monopoly_env.py` ausbauen:
  - Wenn nicht am Zug $\rightarrow$ nur `PASS_TURN` legal.
  - Wenn am Zug vor dem Würfeln $\rightarrow$ `ROLL_DICE` legal.
  - Auf unbesetztem Grundstück mit ausreichend Cash $\rightarrow$ `BUY_PROPERTY` legal.
  - Mit vollständiger Farbgruppe und Cash $\rightarrow$ `BUILD_HOUSE` legal.
  - Mit unverschuldetem Besitz $\rightarrow$ `MORTGAGE` legal.
- [ ] PyTorch Action-Masking-Integration in `src/utils/action_masking.py` mit Tests verifizieren.
- [ ] Testfall schreiben: Erzwinge Versuch einer illegalen Aktion und prüfe, ob Maske sie blockiert bzw. Env eine deterministische Penalty vergibt.
- [ ] Observation Space Dictionary updaten (`{"observation": Box(...), "action_mask": Box(0, 1, (num_actions,), int8)}`).

---

## Präventive Hilfe für HEUTE: Spezifische Hürden & Sofortlösungen

- **Hürde:** All-Zero Mask Bug: Was passiert, wenn keine Aktion legal erscheint?
  - *Lösung:* Sorge dafür, dass `PASS_TURN` als sicherer Default immer legal ist (`mask[PASS_TURN] = 1`), falls keine andere Aktion möglich ist. Das verhindert `-inf`-Crashes in der Softmax-Verteilung.
- **Hürde:** PyTorch `Categorical` wirft NaN bei rein negativen Logits.
  - *Lösung:* Verwende die `MaskedCategorical` Klasse aus `src/utils/action_masking.py`. Sie setzt Masken-False Werte auf $-10^8$, was nach Softmax exakt $0.0$ Wahrscheinlichkeit ergibt, ohne numerisch zu kollabieren.

---

## "Früher fertig?" (Puffer- & Bonus-Tasks)

- [ ] Parametrisierte Sub-Aktionen vorbereiten (z. B. Angabe welches Grundstück bebaut oder beliehen werden soll).
- [ ] Integration von `MultiBinary` Masken für parallele Entscheidungen testen.

---

## "Rückstand?" (Notfall-Priorisierung)

- **MUSS fertig werden:** Valides binäres Masken-Array für die Basiszüge (Würfeln, Kaufen, Passen).
- **KANN entfallen:** Komplexe Häuserbau-Gleichverteilungsregeln (Rule of even building) temporär lockern.

---

## Notizen, Hyperparameter-Logs & W&B Run-IDs

- **Datum / Arbeitszeit:**
- **Git Commit Hash:**
- **Notizen:**
