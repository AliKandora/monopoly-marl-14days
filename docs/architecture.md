# Architecture and scope

## Environment contract

`MonopolyEnv` exposes the PettingZoo Parallel API for four players, while processing only the active player's action per micro-step. The three other living players provide a masked pass action. Turn phases are `ROLL`, `BUY_OR_PASS` and `MANAGE_OR_END`.

The observation is a dictionary containing 144 public-state features and a seven-entry legal-action mask. Actor policies see ownership, buildings, mortgages, player cash/positions/jail/net worth, turn progress, active-player identity and phase. MAPPO's critic receives 146 global features plus a four-entry player-ID vector. The actor can execute without the centralized critic.

## Training

IPPO shares actor and observation-critic parameters across players. MAPPO shares actor parameters and uses a centralized player-conditioned critic. Both use clipped PPO, entropy regularization, value clipping, GAE and gradient clipping.

Terminal transition flags are aligned with their own transition. Rollout-end bootstrap values use the following state, not the preceding state. Already eliminated players are excluded from minibatches. Living inactive players still contribute value targets; their forced pass action has no choice entropy.

The turn limit is a finite-horizon task boundary and is treated as terminal, rather than bootstrapping an infinite task across a time limit. Requested training budgets count environment micro-steps and may be rounded up to a complete rollout.

## Reliability safeguards

- Local random generator per environment; explicit NumPy/Torch training seeds.
- Action type/range validation and required actions for living players.
- At most twelve management actions per turn, preventing mortgage/unmortgage deadlocks.
- Both PPO checkpoints required before a tournament can start.
- Tie-preserving net-worth scoring, rotated seats and per-game evidence.
- Tensor comparison between independently initialized training processes.

The remaining management-action budget is not currently an observation feature. Consequently this safety cap is not a fully represented state variable; expose it when extending the environment or making formal Markov-state claims.

## Rules implemented

40 board tiles; property purchase; simplified street/railroad/utility rents; color-group ownership; automatic even house/hotel building; mortgages and paid redemption; GO income; taxes; basic jail entry/freeing; immediate insolvency; finite-horizon net-worth scoring.

## Deliberate limitations

No event/community cards, auctions, voluntary jail payment, extra doubles turns, three-doubles rule, finite building stock or liquidation before bankruptcy. Negative cash means immediate bankruptcy; assets revert to the bank, not the creditor. Management target selection is deterministic rather than chosen explicitly by the policy.

The trade action is reserved and disabled. Its previous forced-sale placeholder was removed because a transfer without opponent consent is not negotiation. The opponent-pool helper is available but not integrated into the current trainers. W&B tracking, systematic tuning, Elo leagues and multi-seed performance studies are not implemented.

This is a Monopoly-inspired engineering prototype, not a certified implementation of official game rules.
