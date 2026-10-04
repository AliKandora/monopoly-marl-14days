# Tag 02: Monopoly Board & Game Logic

- **Titel & Fokus des Tages:** Monopoly Board State, Spielfeld-Topologie & Grundregeln
- **Pflicht-Milestone (Hauptziel):** Vollständige Modellierung des 40-Felder-Spielbretts mit Besitzverhältnissen, Cash-Transaktionen, GO-Durchlauf und Gefängnis-Mechanik.

---

## Detaillierte Checkliste der Tagesaufgaben

- [ ] Spielfeld-Definition anlegen: 40 Felder (Straßen mit Farbgruppen, Bahnhöfe, Werke, Steuern, GO, Jail, Free Parking, Go to Jail).
- [ ] Datenstrukturen für Board-Status in `src/envs/monopoly_env.py` implementieren (`property_owner`, `property_houses`, `property_mortgaged`).
- [ ] Spieler-Attribute verwalten: `cash` (Startkapital \$1500), `position` (0–39), `in_jail` (Turns in Jail).
- [ ] Würfel-Logik integrieren: 2 Würfel ($2 \times 1\text{--}6$), Pasch-Erkennung (optional vereinfacht), zyklische Positionsaktualisierung modulo 40.
- [ ] GO-Regel umsetzen: Beim Passieren oder Landen auf Feld 0 erhält der Spieler +\$200.
- [ ] Miete-Kassieren: Landet Spieler $A$ auf dem Grundstück von Spieler $B$, wird der entsprechende Betrag transferiert.
- [ ] Bankrott-Kondition: Fällt Cash $< 0$ und keine Hypothek ist mehr möglich, scheidet der Spieler aus.

---

## Präventive Hilfe für HEUTE: Spezifische Hürden & Sofortlösungen

- **Hürde:** Zirkuläre Miete-Schulden bei gleichzeitigem Bankrott.
  - *Lösung:* Wenn Spieler $A$ an Spieler $B$ zahlen muss, aber nicht genug Cash hat, wird sein gesamtes verbleibendes Restvermögen an $B$ übertragen und $A$ sofort auf `terminations[A] = True` gesetzt.
- **Hürde:** Speicherineffizienz durch Objektorientierung für jedes Feld.
  - *Lösung:* Verwende flache NumPy-Arrays (`int8` für Farbgruppen-Besitz, `int32` für Preise und Mieten) statt 40 individueller Python-Klasseninstanzen. Dies garantiert maximale Simulationsgeschwindigkeit.

---

## "Früher fertig?" (Puffer- & Bonus-Tasks)

- [ ] Ereignis- und Gemeinschaftskarten mit 5 Standard-Effekten (Geld gewinnen, Geld zahlen, ins Gefängnis gehen) implementieren.
- [ ] Farbgruppen-Multiplikator für Miete (doppelte Miete bei vollem Besitz) aktivieren.

---

## "Rückstand?" (Notfall-Priorisierung)

- **MUSS fertig werden:** 40 Felder, Cash-Abzüge bei Miete, GO-Auszahlung und Bankrott-Ausscheiden.
- **KANN entfallen:** Bahnhöfe/Werke vorerst als Standard-Grundstücke mit Festmiete behandeln; Kartenstapel weglassen.

---

## Notizen, Hyperparameter-Logs & W&B Run-IDs

- **Datum / Arbeitszeit:**
- **Git Commit Hash:**
- **Notizen:**
