# Monopoly-MARL: Der Ultimative Study Guide & Deep-Dive Vorbereitung

> **Herzlichen Glückwunsch zu deinem allerersten GitHub-Repository!**  
> Dieses Dokument ist dein persönlicher Spickzettel und Wissensfundament. Hier findest du alle theoretischen und praktischen Konzepte kompakt, verständlich und mit anschaulichen Beispielen erklärt, damit du mit vollem Selbstvertrauen in das 14-Tage-Projekt startest.

---

## Inhaltsverzeichnis

1. [Reinforcement Learning (RL) Grundlagen](#1-reinforcement-learning-rl-grundlagen)
2. [Policy Gradients & PPO (Proximal Policy Optimization)](#2-policy-gradients--ppo)
3. [Multi-Agent RL (MARL) & Spieltheorie](#3-multi-agent-rl-marl--spieltheorie)
4. [PettingZoo & Environment Engineering](#4-pettingzoo--environment-engineering)
5. [PyTorch & Action Masking in der Praxis](#5-pytorch--action-masking-in-der-praxis)
6. [Die goldenen Faustregeln für Monopoly-MARL](#6-die-goldenen-faustregeln)
7. [Verständnis-Check: 5 Quiz-Fragen mit Musterlösungen](#7-verst%C3%A4ndnis-check)

---

## 1. Reinforcement Learning (RL) Grundlagen

### Was ist Reinforcement Learning im Kern?
Im Gegensatz zu *Supervised Learning* (wo ein Lehrer die richtige Antwort vorgibt) lernt ein RL-Agent durch **Trial and Error** (Versuch und Irrtum). Er interagiert mit einer Umgebung (*Environment*), beobachtet Zustände, wählt Aktionen und erhält Belohnungen (*Rewards*).

```
                      +-------------------+
                      |    Environment    |
                      |  (Monopoly Board) |
                      +---------+---------+
                                |
             Observation o_t    |    Reward r_t
                                v
                      +-------------------+
                      |       Agent       |
                      |   (Neural Net)    |
                      +---------+---------+
                                |
                                | Action a_t (z. B. BUY_PROPERTY)
                                v
                      +-------------------+
                      |    Environment    |
                      +-------------------+
```

### Die Kernbegriffe, die du im Schlaf kennen musst:

| Begriff | Symbol | Bedeutung in Monopoly |
|---|---|---|
| **State / Observation** | $s$ bzw. $o$ | Das Spielfeld: Wer besitzt welche Straßen? Wie viel Bargeld hat jeder? Wer sitzt im Gefängnis? |
| **Action** | $a$ | Was der Agent tut: Würfeln, Straße kaufen, Haus bauen, Hypothek aufnehmen, Runde beenden. |
| **Reward** | $r$ | Das Feedback-Signal: $+2.0$ für den Kauf einer Straße, $+100.0$ für den Gesamtsieg, $-50.0$ bei Bankrott. |
| **Policy** | $\pi(a \mid s)$ | Die Strategie / das "Gehirn" des Agenten: Gibt die Wahrscheinlichkeit an, im Zustand $s$ die Aktion $a$ zu wählen. |
| **Value Function** | $V(s)$ | Die Erwartung: *"Wie gut ist mein Zustand langfristig?"* (Wie viel Gesamtgewinn werde ich von hier an noch erzielen?). |
| **Q-Function** | $Q(s, a)$ | *"Wie gut ist es, im Zustand $s$ spezifisch Aktion $a$ zu wählen?"*. |
| **Discount Factor** | $\gamma \in [0, 1)$ | Zeitpräferenz: Belohnungen heute sind mehr wert als Belohnungen in 50 Zügen ($\gamma = 0.99$). |

---

## 2. Policy Gradients & PPO

### Warum nicht einfach Deep Q-Learning (DQN)?
DQN schätzt den Wert jeder Aktion. In Multi-Agenten-Systemen und komplexen Spielen mit wechselnd legalen Aktionen sind **Policy Gradient Methoden** (konkret **Actor-Critic**) deutlich stabiler, weil sie direkt die Strategie $\pi_\theta(a \mid s)$ optimieren und mit stochastischen Policies besser explorieren.

### Der Aufbau eines Actor-Critic Netzwerks:
1. **Der Actor (Spieler):** Gibt Wahrscheinlichkeiten über die Aktionen aus: $\pi_\theta(a \mid o)$.
2. **Der Critic (Bewerter):** Schätzt den Erwartungswert der aktuellen Situation: $V_\phi(s)$.

### Das Herzstück von PPO: Das Clipped Objective
Wenn ein neurales Netz in einem Schritt zu viel lernt, kann die Strategie schlagartig kollabieren ("Policy Collapse"). PPO verhindert das mit einer simplen, genialen mathematischen Schranke (dem **Clipping**):

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$

- Wenn ein Zug gut war ($\hat{A}_t > 0$), erhöht PPO dessen Wahrscheinlichkeit – aber **maximal um $(1 + \epsilon)$** (meist $\epsilon = 0.2$).
- Das garantiert extrem stabiles Lernen ohne böse Überraschungen über Nacht!

### Was ist der Advantage $\hat{A}_t$?
Der Advantage misst: *"War diese Aktion besser oder schlechter als das, was der Critic im Durchschnitt für diesen Zustand erwartet hat?"*
$$\hat{A}_t = Q(s_t, a_t) - V(s_t)$$
Wird berechnet über **GAE (Generalized Advantage Estimation)**, um Rauschen zu filtern.

---

## 3. Multi-Agent RL (MARL) & Spieltheorie

### Warum ist Multi-Agent so viel schwerer als Single-Agent?
Im Single-Agent RL (z. B. Super Mario) ist die Spielwelt statisch: Springt Mario gegen eine Wand, prallt er ab – immer gleich.
Im Multi-Agent RL verändern sich **alle Agenten gleichzeitig**:
1. Wenn Spieler 0 lernt, aggressiv alle Straßen aufzukaufen,
2. müssen Spieler 1, 2 und 3 lernen, sich mit Bargeld zu schützen oder gezielt Gegenmonopole zu blockieren.
3. Die Umgebung ist aus Sicht eines einzelnen Agenten **nicht-stationär** (die Spielregeln der Gegner verändern sich permanent!).

### Das CTDE-Paradigma (Centralized Training with Decentralized Execution)

```
========================================================================
TRAININGS-PHASE (Centralized - "Im Simulator mit Röntgenblick")
 Critic sieht ALLES: Global Board, Geld aller Spieler, verdeckte Karten
 Critic berechnet präzise Value-Updates: V(s_global)
========================================================================
                                   |
                                   v
========================================================================
SPIEL-PHASE / INFERENZ (Decentralized - "Am realen Spieltisch")
 Jeder Actor sieht NUR seine eigene Hand & offene Board-Infos
 Wählt autonom Züge: a_i ~ pi(o_i) ohne fremde Hilfe!
========================================================================
```

- **IPPO (Independent PPO):** Jeder Agent tut so, als wären alle anderen Teil der Umwelt. Einfach, überraschend stark!
- **MAPPO (Multi-Agent PPO):** Actor arbeitet dezentral, aber der Critic nutzt $s_{\text{global}}$. Hilft enorm bei der "Credit Assignment"-Frage: *"Habe ich verloren, weil ich schlecht gespielt habe, oder weil Spieler 1 und 2 sich gegenseitig reich getauscht haben?"*.

---

## 4. PettingZoo & Environment Engineering

### Was ist PettingZoo?
PettingZoo ist der weltweite Standard für Multi-Agent Gym-Environments (entwickelt von der Farama Foundation, den Machern von Gymnasium).

### Parallel API vs. AEC API:
- **Parallel API (nutzen wir hier):** Alle aktiven Spieler geben gleichzeitig ihre Aktionen in einem Dictionary ab: `env.step({"player_0": act0, "player_1": act1, ...})`. Perfekt für schnelles Batch-Training auf GPUs/Multicore-CPUs.
- **AEC API (Agent-Environment Cycle):** Sequenziell wie Schach (`agent_iter()`).

### Action Masking: Warum es unverzichtbar ist
In Monopoly kannst du nicht in jedem Zug ein Hotel bauen – du brauchst erst alle 3 Straßen derselben Farbe und genug Bargeld!
- **Falscher Weg:** Den Agenten raten lassen und bei verbotenen Zügen mit $-1000$ bestrafen $\rightarrow$ Agent verbringt 95% der Zeit damit, sinnlos gegen Wände zu rennen.
- **Richtiger Weg (Action Masking):** Wir übergeben dem neuronalen Netz eine binäre Maske (z. B. `[1, 0, 1, 0, 0]`). Unmögliche Aktionen werden vor der Softmax-Berechnung auf $-\infty$ gesetzt. Ihre Wahrscheinlichkeit ist exakt $0.0\%$. Der Agent kann **nur legale Züge** auswählen!

---

## 5. PyTorch & Action Masking in der Praxis

So sieht der entscheidende Code-Block in PyTorch aus:

```python
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class MaskedActor(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        # 1. Rohe Logits (unnormalisierte Vorhersagen)
        logits = self.network(obs)
        
        # 2. Illegale Aktionen mit sehr großer negativer Zahl maskieren
        neg_inf = torch.tensor(-1e8, device=logits.device)
        masked_logits = torch.where(mask.bool(), logits, neg_inf)
        
        # 3. Softmax-Verteilung erzeugen (nur legale Aktionen erhalten > 0%)
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        return action, dist.log_prob(action)
```

---

## 6. Die goldenen Faustregeln für Monopoly-MARL

1. **Belohne Net-Worth ($\Delta \text{Vermögen}$), nicht isoliertes Bargeld!**  
   Wenn du nur Bargeld belohnst, verkauft der Agent alle Grundstücke, freut sich kurz über \$1000 und geht drei Runden später pleite.
2. **Begrenze Rundenzeiten hart!**  
   Monopoly neigt zu Endlos-Partien, wenn 4 vorsichtige Bots aufeinandertreffen. Setze `max_turns = 200`. Wer dann das meiste Vermögen hat, gewinnt.
3. **Traue keiner einzelnen Gewinnquote!**  
   Weil Würfel im Spiel sind, gibt es Varianz. Teste Checkpoints immer über mindestens 100 Spiele mit getauschten Spieler-Startpositionen.
4. **Frozen Opponent Pool gegen "Kreis-Strategien":**  
   Trainiere die Agenten nicht nur gegen die aktuelle Version, sondern mische frühere Checkpoints aus Tag 05 oder Tag 08 ein.

---

## 7. Verständnis-Check: 5 Quiz-Fragen

Teste dein Wissen! Versuche zuerst selbst die Antwort zu formulieren, bevor du in die Lösung schaust.

---

### Frage 1: Action Masking
**Frage:** Ein Agent hat \$50 auf dem Konto. Vor ihm liegt die Schlossallee (Kosten: \$400). Warum bestrafen wir den Agenten nicht einfach mit einem negativen Reward (z. B. $-10.0$), wenn er `BUY_PROPERTY` wählt, sondern nutzen stattdessen eine Action-Maske?

<details>
<summary>👉 Lösung anzeigen</summary>

**Antwort:**  
Weil negatives Bestrafen ("Penalty-Shaping") den Policy-Gradienten verzerrt. Der Agent müsste erst tausende Male zufällig den Fehler machen und lernen, den negativen Reward zu meiden, anstatt sofort zu lernen, welche *legalen* Entscheidungen (z. B. Sparen oder Würfeln) taktisch klug sind. Action Masking nimmt unmögliche Aktionen sofort aus dem Lösungsraum heraus, spart bis zu 90% der Rechenzeit und verhindert, dass das neuronale Netz degeneriert.
</details>

---

### Frage 2: Reward-Design & Net-Worth
**Frage:** Spieler A hypothekiert eine Straße und erhält dafür von der Bank sofort \$100 Bargeld. Warum sollte sich sein Reward in diesem Moment **nicht** um $+100$ erhöhen?

<details>
<summary>👉 Lösung anzeigen</summary>

**Antwort:**  
Weil sein Gesamtvermögen (Net-Worth) sich durch die Hypothek nicht vergrößert hat: Er hat zwar +\$100 Cash, aber gleichzeitig den Wert der unbelasteten Immobilie verloren und Schulden bei der Bank. Würden wir +\$100 belohnen, würde der Agent lernen, in jeder Runde all seine Straßen zu hypothekieren ("Reward Hacking"), was ihn strategisch handlungsunfähig macht.
</details>

---

### Frage 3: PPO Clipping
**Frage:** Was bezweckt der Parameter $\epsilon$ (z. B. $\epsilon = 0.2$) in der PPO-Clip-Formel? Was würde passieren, wenn $\epsilon = 5.0$ wäre?

<details>
<summary>👉 Lösung anzeigen</summary>

**Antwort:**  
$\epsilon$ begrenzt das Verhältnis $r_t(\theta) = \frac{\pi_\theta}{\pi_{\text{old}}}$ auf das Intervall $[1 - \epsilon, 1 + \epsilon]$ (bei $0.2$ also auf $[0.8, 1.2]$). Dadurch darf sich die Wahrscheinlichkeit einer Aktion pro Trainingsschritt um höchstens 20% ändern. Wäre $\epsilon = 5.0$, könnte ein einziger glücklicher Zufallswurf das Netz so stark in eine Richtung verbiegen, dass alle zuvor gelernten Nuancen überschrieben werden (Policy Collapse).
</details>

---

### Frage 4: CTDE (Centralized Training vs. Decentralized Execution)
**Frage:** Warum darf der Critic im MAPPO-Algorithmus während des Trainings wissen, wie viel Geld die Gegner auf dem Konto haben, der Actor während des Spiels aber nicht zwingend?

<details>
<summary>👉 Lösung anzeigen</summary>

**Antwort:**  
Weil der Critic nur während der Trainingsphase im Simulator existiert, um die Value-Funktion $V(s_{\text{global}})$ präzise zu schätzen und den Vorteil $\hat{A}_t$ zu berechnen. Sobald das Training beendet ist, wird der Critic weggeworfen! Im echten Spiel läuft nur noch der dezentrale Actor ($\pi(o_i)$), der ausschließlich seine eigenen lokalen Informationen benötigt.
</details>

---

### Frage 5: Non-Stationarity im Self-Play
**Frage:** Warum kann ein Agent, der 5 Millionen Schritte lang ausschließlich gegen sich selbst (Self-Play) trainiert hat, plötzlich gegen einen simplen Anfänger-Bot haushoch verlieren?

<details>
<summary>👉 Lösung anzeigen</summary>

**Antwort:**  
Das Phänomen heißt **Overfitting an die eigene Nische** (oder Policy Cyclicity). Durch das ständige Self-Play haben beide Seiten hochspezialisierte Gegenstrategien zueinander entwickelt (z. B. "Wenn ich nicht handle, passt du auch"). Ein simpler Random- oder Heuristik-Bot spielt unberechenbar und hält sich nicht an diese stillschweigenden Annahmen. Die Lösung hierfür ist ein **Opponent Pool** (Tag 12), bei dem der Agent regelmäßig auch gegen historische Versionen und den Heuristik-Bot antreten muss.
</details>

---

## Du bist startklar!

Lies dir vor Beginn jedes Tages kurz den entsprechenden Tracker in [`daily_trackers/`](daily_trackers/) durch. Alles, was du brauchst, ist Schritt für Schritt aufgebaut. Viel Erfolg beim Coden!
