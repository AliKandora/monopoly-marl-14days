# Tag 11: Trading & Negotiation Logic

- **Titel & Fokus des Tages:** Bilaterale Handelslogik, Diskretes Tausch-Protokoll & Verhandlungs-Heuristik
- **Pflicht-Milestone (Hauptziel):** Implementierung eines vereinfachten, handhabbaren Tausch-Interfaces (Asset gegen Bargeld oder 1-zu-1 Asset-Swap) ohne kombinatorische Aktionsraum-Explosion.

---

## Detaillierte Checkliste der Tagesaufgaben

- [ ] Trading-Architektur entwerfen:
  - Phase 1: Anbieter wählt Angebot (Eigenes Grundstück $X$ + Preisforderung $Y$).
  - Phase 2: Zielspieler erhält Angebot in seiner Observation und entscheidet binär: `[ACCEPT_TRADE, REJECT_TRADE]`.
- [ ] Diskretisierung der Cash-Beträge (z.B. in \$50-Schritten: \$50, \$100, \$200, \$300), um den Aktionsraum kompakt zu halten.
- [ ] Action-Masking für Trading:
  - Tauschangebot nur legal, wenn der Spieler tatsächlich unhypothekiertes Eigentum besitzt.
  - Annahme nur legal, wenn der Empfänger über ausreichende Liquidität verfügt.
- [ ] Trade-Akzeptanz-Logik im HeuristicBot implementieren (akzeptiert, wenn Trade ein Monopol vervollständigt und Liquiditätspuffer intakt bleibt).
- [ ] Evaluierung: Führt Trading zu schnelleren Monopolbildungen und kürzeren Spieldauern?

---

## Präventive Hilfe für HEUTE: Spezifische Hürden & Sofortlösungen

- **Hürde:** Tausch-Deadlocks oder repetitive Spam-Angebote zwischen Agenten.
  - *Lösung:* Maximal 1 Tauschangebot pro Spieler und Runde erlauben (`has_proposed_trade_this_turn`-Flag). Nach einer Ablehnung ist kein weiteres Angebot im selben Zug möglich.
- **Hürde:** RL-Agent verschenkt Grundstücke unter Wert ("Dumping").
  - *Lösung:* Net-Worth Reward greift sofort! Wenn ein Agent ein Grundstück für \$50 weggibt, das einen fairen Wert von \$200 hat, erfährt er ein negatives $\Delta \text{NetWorth}$ Signal von $-\$150$, was Dumping rasch eliminiert.

---

## "Früher fertig?" (Puffer- & Bonus-Tasks)

- [ ] Visualisierung von Tausch-Netzwerken (wer handelt am häufigsten mit wem?) in `src/visualization/trade_graph.py`.
- [ ] 1-zu-1 Tausch (Straße gegen Straße zur beidseitigen Monopolbildung) implementieren.

---

## "Rückstand?" (Notfall-Priorisierung)

- **MUSS fertig werden:** Unilateraler Tausch: Angebot einer Immobilie zum Festpreis an den Höchstbietenden bzw. Zielspieler.
- **KANN entfallen:** Freies Verhandeln über mehrere Runden (Single-Shot Proposal genügt vollkommen).

---

## Notizen, Hyperparameter-Logs & W&B Run-IDs

- **Datum / Arbeitszeit:**
- **Git Commit Hash:**
- **Durchschnittliche Trades pro Spiel:**
- **W&B Run ID:**
