# Engineering decisions

## Portfolio refresh — 9 October 2026

- **MIT license:** a short, permissive license for this educational codebase. Copyright uses the public GitHub identity AliKandora. It does not license third-party trademarks or override dependency licenses.
- **Evidence before presentation:** reproduce both trainers from fresh initialization in independent processes before updating claims in the README.
- **Same-seed replay, not a performance claim:** compare all model parameters and per-game outcomes; retain the small-sample result even though PPO does not beat the heuristic.
- **Clean landing page:** English README for an international Python/ML-engineering audience, with a runnable demo, architecture diagram, CI badge and evidence links.
- **Remove obsolete clutter from the current tree:** fourteen daily sprint templates, the three general learning documents, and the old unsupported result table/plot. No Git history rewrite; all remain available in earlier commits.
- **Preserve technical provenance:** previous evaluation and decision records are retained under `docs/history/`, clearly marked as historical.
- **No exaggerated CV claims:** describe AI-assisted development, reproducible engineering and test coverage—not sole manual authorship, research-grade convergence or full game-rule support.
- **Fix preview labeling:** a demo that stops at its step budget reports a preview limit and current leaders, not a completed game or final winner.
- **Portable evidence:** record checkpoint paths relative to the repository; no user-machine paths or model binaries in published artifacts.
- **Keep the algorithm stable:** no RL-rule changes during presentation cleanup; v1.0 weights and game results are independently reproduced.
- **No unsolicited long study, release or rename:** longer experiments, a public release tag and a repository rename require a separate decision.

## Core v1.0 choices

Finite-horizon episodes; aligned terminal transitions in GAE; next-state bootstrap; dead-agent exclusion from minibatches; local environment RNG; bounded management steps; disabled non-consensual trading; mandatory trained checkpoints; rotated seats and explicit ties.

Implementation details: [architecture](architecture.md). Evidence: [independent reproduction](reproducibility.md). Historical rationale: [v1.0 decisions](history/decisions-v1.md).
