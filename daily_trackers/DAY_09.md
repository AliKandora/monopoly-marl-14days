# Tag 09: MAPPO & Centralized Critic

- **Titel & Fokus des Tages:** Centralized Training with Decentralized Execution (CTDE) mit Multi-Agent PPO (MAPPO)
- **Pflicht-Milestone (Hauptziel):** Implementierung einer zentralisierten Value-Function $V(s_{global})$, die den vollständigen Spielbrett- und Spieler-Zustand verarbeitet, während die Aktionsauswahl rein dezentral bleibt.

---

## Detaillierte Checkliste der Tagesaufgaben

- [ ] Globalen State-Extractor in `src/utils/feature_extraction.py` implementieren:
  - Konkateniert alle lokalen Beobachtungen oder extrahiert den allwissenden Board-State (inkl. unsichtbarer Metriken, Rest-Bargeld aller Gegner, Bankkredit-Limits).
- [ ] MAPPO Actor-Critic Architektur in `src/agents/mappo_agent.py` erstellen:
  - Dezentraler Actor: $a_i \sim \pi(o_i, \text{mask}_i)$
  - Zentralisierter Critic: $V_\phi(s_{global})$
- [ ] Value-Loss Berechnung auf Basis von $s_{global}$ mit Value Clipping adaptieren.
- [ ] Vergleichs-Experiment: IPPO (Tag 08) vs. MAPPO (Tag 09) unter identischen Seeds starten.
- [ ] Konvergenzgeschwindigkeit und Varianz der Advantage-Schätzung vergleichen.

---

## Präventive Hilfe für HEUTE: Spezifische Hürden & Sofortlösungen

- **Hürde:** Dimension des globalen States sprengt den Speicher oder verlangsamt das Backpropagation.
  - *Lösung:* Normalisiere alle globalen Werte rigoros ($[0, 1]$ für Cash relativ zum Gesamtgeld im Spiel) und halte das Critic-Netzwerk schlank (2x256 Neuronen genügen für Monopoly).
- **Hürde:** "Lazy Critic" lernt nur den Mittelwert des Spiels und ignoriert individuelle Agenten-Dynamiken.
  - *Lösung:* Übergebe dem Critic neben $s_{global}$ auch den One-Hot-Vektor des jeweiligen Agenten $i$, sodass er $V_\phi(s_{global}, i)$ schätzen kann.

---

## "Früher fertig?" (Puffer- & Bonus-Tasks)

- [ ] Attention-Mechanismus (Self-Attention über alle 4 Spieler-Zustände) im zentralen Critic testen.
- [ ] Plot der Value-Schätzung über eine gesamte Beispielpartie anfertigen.

---

## "Rückstand?" (Notfall-Priorisierung)

- **MUSS fertig werden:** Lauffähiger MAPPO-Loss mit globalem State-Input für den Critic.
- **KANN entfallen:** Attention-Critic (Standard-MLP reicht für den Milestone).

---

## Notizen, Hyperparameter-Logs & W&B Run-IDs

- **Datum / Arbeitszeit:**
- **Git Commit Hash:**
- **Vergleich Advantage-Varianz (IPPO vs MAPPO):**
- **W&B Run ID:**
