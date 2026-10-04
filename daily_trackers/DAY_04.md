# Tag 04: Reward Engineering

- **Titel & Fokus des Tages:** Reward Shaping, Sparse vs. Dense Balancing & Net-Worth Dynamics
- **Pflicht-Milestone (Hauptziel):** Implementierung einer ausgewogenen Reward-Struktur, die stabiles Lernen ermöglicht und weder zu lethargischem Passieren noch zu irrationalem Verkaufen führt.

---

## Detaillierte Checkliste der Tagesaufgaben

- [ ] Reward-Komponenten im Environment formalisieren:
  - **Terminal Win/Loss:** $+100.0$ für alleinigen Überlebenden, $-50.0$ für Bankrott.
  - **Net-Worth Delta ($\Delta \text{NW}$):** Differenz aus $(\text{Cash}_t + \text{Immobilienwert}_t) - (\text{Cash}_{t-1} + \text{Immobilienwert}_{t-1})$ skaliert mit Faktor $0.005$.
  - **Mietgewinn / Mietverlust:** Kleine proportionale Signale bei Mieteinnahmen ($+0.5$) bzw. Mietzahlungen ($-0.5$).
  - **Passivität / Time-Penalty:** Minimale Zeitstrafe ($-0.01$ pro Runde) zur Förderung aktiver Spielzüge.
- [ ] Reward-Clipping und Normalisierung: Reward-Werte auf den Bereich $[-10.0, +10.0]$ clampen (außer Terminal-Reward).
- [ ] Unit-Tests für monotone Belohnungssteigerung bei Vermögenszuwachs erstellen.
- [ ] Simulation über 100 Episoden laufen lassen und kumulative Rewards der Agenten plotten.

---

## Präventive Hilfe für HEUTE: Spezifische Hürden & Sofortlösungen

- **Hürde:** Reward Hacking (z. B. Agent hypothekiert Immobilie und kauft sie sofort wieder zurück, um temporäre Cash-Boni abzugreifen).
  - *Lösung:* Belohne NIEMALS isolierte Cash-Gewinne, sondern ausschließlich den kombinierten **Net-Worth** (Bargeld + Immobilienwert abzüglich Hypothekenschulden). Eine Hypothek ändert den Net-Worth nicht (Cash steigt um M, Immobilienwert sinkt um M).
- **Hürde:** Explodierende Q-Werte bei langen Episoden.
  - *Lösung:* Discount-Faktor $\gamma$ auf $0.99$ setzen und sicherstellen, dass die Net-Worth-Differenzen als $\Delta = \text{NW}_t - \text{NW}_{t-1}$ und nicht als absoluter Betrag übergeben werden.

---

## "Früher fertig?" (Puffer- & Bonus-Tasks)

- [ ] Reward-Wrapper-Klasse (`src/utils/reward_wrappers.py`) schreiben, um verschiedene Reward-Varianten per Config-Parameter austauschen zu können.
- [ ] Farbgruppen-Vervollständigungs-Bonus ($+10.0$ Einmalbelohnung bei Monopolbildung) evaluieren.

---

## "Rückstand?" (Notfall-Priorisierung)

- **MUSS fertig werden:** Net-Worth Delta Reward + Bankrott-Terminal Penalty.
- **KANN entfallen:** Komplexe Mieteinnahmen-Zusatzboni (Net-Worth deckt Mieten automatisch ab).

---

## Notizen, Hyperparameter-Logs & W&B Run-IDs

- **Datum / Arbeitszeit:**
- **Git Commit Hash:**
- **Notizen:**
