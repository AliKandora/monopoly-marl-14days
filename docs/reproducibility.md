# Independent reproduction

Verified on **9 October 2026** from the published v1.0 source commit `c0639c2cf6fd88b10be50798c862836ea4847924`.

## What was actually checked

The Git transport failed in the host Windows TLS layer. A fresh working copy was therefore downloaded from GitHub at the fixed commit, with every file verified against its Git blob SHA. A separate virtual environment was created from `requirements-tested.txt` using cached packages. No old checkpoint was used as training input.

Two independent runs each launched IPPO and MAPPO in separate Python processes from seeded random initialization. Each model trained for **2,048 environment micro-steps**, with training seed **42**. Each pair of models then played **eight games**, maximum **50 turns**, evaluation seed **100**. Seats were rotated; ties were retained instead of assigned to the first seat.

## Findings

| Check | IPPO | MAPPO |
|---|---:|---:|
| Actor parameters exactly equal between fresh runs | Yes | Yes |
| Critic parameters exactly equal between fresh runs | Yes | Yes |
| Maximum absolute tensor difference | 0.0 | 0.0 |
| Training metadata equal | Yes | Yes |

All evaluation games, step/turn counts, net-worth records, winners, ties and confidence intervals matched between the two new runs. They also matched the previous v1.0 smoke report, which was used only as a comparison reference—not as training data or checkpoint input.

The eight-game result was Random 2 wins (25%), Heuristic 5 wins (62.5%), IPPO 0 wins and MAPPO 1 win (12.5%). There were no ties. These tiny-sample numbers are disclosed for traceability, **not promoted as evidence of algorithmic superiority**.

## Environment

- Windows, Python 3.12.7.
- PyTorch 2.14.1+cpu; NumPy 2.5.3; PettingZoo 1.27.0; Gymnasium 1.4.0.
- One Torch CPU thread inside each training/evaluation process.
- Full pinned package snapshot: [requirements-tested.txt](../requirements-tested.txt).

## Re-run it

After activating a fresh Python environment:

```bash
python -m pip install -r requirements-tested.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pytest tests -q
python scripts/reproduce.py --steps 2048 --games 8 --max-turns 50
```

The script starts four fresh training processes, writes local models and per-run evaluation files, then exits with an error if model parameters or evaluation results differ. It overwrites files inside its output directory; choose a new `--output` directory to retain another experiment.

Evidence:
- [Machine-readable comparison](../results/reproduction/comparison.json)
- [First fresh evaluation](../results/reproduction/run-a/tournament.json)
- [Second fresh evaluation](../results/reproduction/run-b/tournament.json)
- [Prior v1.0 reference](../results/smoke/tournament.json)

Weights are intentionally excluded from Git. Serialized checkpoint file hashes can differ even when all tensors are identical; the reproduction check compares actual tensor values. Tournament JSON still records the hashes of the specific files it loaded.

## What this does not establish

This is same-seed deterministic replay in one recorded hardware/software environment. It is not an independent implementation of PPO, not a cross-platform or GPU guarantee, not robustness across training seeds, and not convergence. Eight short games cannot support a reliable ranking of the algorithms. Larger, held-out, multi-seed experiments remain a separate research step.

Tests pass on the recorded Windows environment; the existing GitHub Actions suite also runs on Linux. A successful test suite on Linux does not by itself prove identical learned weights across operating systems.

## Portfolio regression check

After adding the reproduction comparator and correcting demo preview labels, the expanded local suite passed **31 tests** with two expected PettingZoo AEC warnings. The full two-run reproduction was repeated on the final local source and still passed.
