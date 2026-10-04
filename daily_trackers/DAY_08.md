# Tag 08: Independent PPO (IPPO) Setup

- **Titel & Fokus des Tages:** Dezentrales Multi-Agent Training mit Independent PPO (IPPO)
- **Pflicht-Milestone (Hauptziel):** Erfolgreicher Start und Konvergenz eines echten 4-Agenten Self-Play Trainings mit IPPO (Shared Policy oder 4 individuelle Policies).

---

## Detaillierte Checkliste der Tagesaufgaben

- [ ] Architektur-Entscheidung: Parameter-Sharing (1 Policy steuert alle 4 Spieler mit Agent-ID One-Hot) vs. heterogene Policies. Empfehlung: Parameter-Sharing für 4x höhere Sample-Effizienz.
- [ ] IPPO-Trainer in `src/agents/ippo_trainer.py` implementieren:
  - Trajektorien aller 4 Agenten in gemeinsamen Rollout-Buffer zusammenführen.
  - Dezentrale Value-Function $V_i(o_i)$ nutzt ausschließlich die lokale Beobachtung $o_i$.
- [ ] Multi-Agent Loss Aggregation: Mittelwertbildung des Policy- und Value-Loss über alle aktiven Agenten.
- [ ] Trainingslauf über 1 Million Total Environment Steps starten.
- [ ] Metriken loggen: Mittlerer Gesamt-Reward pro Agent, Episodenlänge, Bankrott-Rate.

---

## Präventive Hilfe für HEUTE: Spezifische Hürden & Sofortlösungen

- **Hürde:** Asymmetrische Startvorteile (z. B. Spieler 0 agiert zuerst und hat statistisch höhere Chancen auf frühe Straßen).
  - *Lösung:* Rotiere die Startreihenfolge in jedem `reset()` (`current_agent_idx = np.random.randint(0, 4)`), damit kein Bias im Parameter-Sharing entsteht.
- **Hürde:** Instabile Trainingskurven durch Nicht-Stationarität (alle Agenten verändern ihre Policy simultan).
  - *Lösung:* Kleine Learning Rate ($1.5 \cdot 10^{-4}$ bis $2.5 \cdot 10^{-4}$) und PPO-Clip-Range auf $0.1$ oder $0.15$ reduzieren.

---

## "Früher fertig?" (Puffer- & Bonus-Tasks)

- [ ] Checkpoint-Saving alle $100.000$ Steps mit automatischer Evaluierungsrunde gegen den Heuristik-Bot.
- [ ] Generierung eines ersten W&B Dashboard-Links.

---

## "Rückstand?" (Notfall-Priorisierung)

- **MUSS fertig werden:** Lauffähiger IPPO-Trainingsloop mit Parameter-Sharing über mindestens $300.000$ Steps.
- **KANN entfallen:** Training separater Netzwerke für jeden der 4 Spieler (Parameter-Sharing reicht vollkommen aus).

---

## Notizen, Hyperparameter-Logs & W&B Run-IDs

- **Datum / Arbeitszeit:**
- **Git Commit Hash:**
- **Shared Policy (Ja/Nein):**
- **W&B Run ID:**
