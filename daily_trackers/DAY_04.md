# Tag 04: Reward Engineering

- **Titel & Fokus des Tages:** Reward Shaping, Sparse vs. Dense Balancing & Net-Worth Dynamics
- **Pflicht-Milestone (Hauptziel):** Implementierung einer ausgewogenen Reward-Struktur, die stabiles Lernen ermöglicht und weder zu lethargischem Passieren noch zu irrationalem Verkaufen führt.

---

## Detaillierte Checkliste der Tagesaufgaben

- [x] Reward-Komponenten im Environment formalisieren:
  - **Terminal Win/Loss:** $+100.0$ für alleinigen Überlebenden, $-50.0$ für Bankrott.
  - **Net-Worth Delta ($\Delta \text{NW}$):** Differenz aus $(\text{Cash}_t + \text{Immobilienwert}_t) - (\text{Cash}_{t-1} + \text{Immobilienwert}_{t-1})$ skaliert mit Faktor $0.001$.
  - **Passivität / Time-Penalty:** Minimale Zeitstrafe ($-0.005$ pro Runde) zur Förderung aktiver Spielzüge.
- [x] Reward-Berechnung in `src/utils/reward_wrappers.py` gekapselt (`RewardConfig`, `compute_shaped_rewards`).
- [x] Monotone Net-Worth-Kalkulation in `tests/test_env.py` verifiziert.
- [x] Simulation über 50 Episoden mit lückenlosem Net-Worth Tracking durchgeführt.

---

## Präventive Hilfe für HEUTE: Spezifische Hürden & Sofortlösungen

- **Hürde:** Reward Hacking (z. B. Agent hypothekiert Immobilie und kauft sie sofort wieder zurück).
  - *Lösung:* Belohne NIEMALS isolierte Cash-Gewinne, sondern ausschließlich den kombinierten **Net-Worth**. Eine Hypothek ändert den Net-Worth nicht.
- **Hürde:** Explodierende Q-Werte bei langen Episoden.
  - *Lösung:* Discount-Faktor $\gamma$ auf $0.99$ setzen und Net-Worth-Differenzen als $\Delta = (\text{NW}_t - \text{NW}_{t-1})/1000$ normalisieren.

---

## "Früher fertig?" (Puffer- & Bonus-Tasks)

- [x] Reward-Wrapper-Klasse (`src/utils/reward_wrappers.py`) geschrieben und voll integriert.
- [x] Terminal-Reward-Boni für Alleinsieg ($+100.0$) implementiert.

---

## "Rückstand?" (Notfall-Priorisierung)

- **MUSS fertig werden:** Net-Worth Delta Reward + Bankrott-Terminal Penalty.
- **KANN entfallen:** Komplexe Mieteinnahmen-Zusatzboni.

---

## Notizen, Hyperparameter-Logs & W&B Run-IDs

- **Status:** **Erfolgreich abgeschlossen**
- **Git Commit:** `6ee50ed`
- **Ergebnis:** Stabiler, dichter $\Delta\text{NetWorth}$-Reward ohne Explodieren der Gradienten.
