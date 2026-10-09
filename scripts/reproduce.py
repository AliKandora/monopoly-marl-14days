"""Run two independent seeded CPU trainings and compare weights and evaluation.

Usage: python scripts/reproduce.py --steps 2048 --games 8 --max-turns 50
Generated model files stay local; only reports and logs belong in the repository.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def run_logged(arguments: list[str], log_path: Path) -> None:
    with log_path.open('w', encoding='utf-8') as log:
        subprocess.run(
            [sys.executable, *arguments], cwd=ROOT, stdout=log,
            stderr=subprocess.STDOUT, text=True, check=True,
        )


def compare_checkpoint_weights(first: Path, second: Path) -> dict:
    a = torch.load(first, map_location='cpu', weights_only=True)
    b = torch.load(second, map_location='cpu', weights_only=True)
    report = {}
    for group in ('actor_state_dict', 'critic_state_dict'):
        if set(a[group]) != set(b[group]):
            raise AssertionError(f'Different parameter keys in {group}')
        exact = all(torch.equal(a[group][key], b[group][key]) for key in a[group])
        max_delta = max(float((a[group][key] - b[group][key]).abs().max()) for key in a[group])
        report[group] = {'exact': exact, 'max_absolute_difference': max_delta}
    report['metadata_equal'] = a.get('metadata') == b.get('metadata')
    return report


def reproduce(steps: int = 2048, games: int = 8, max_turns: int = 50,
              training_seed: int = 42, evaluation_seed: int = 100,
              output: str = 'results/reproduction') -> dict:
    if min(steps, games, max_turns) < 1:
        raise ValueError('steps, games and max-turns must be positive')
    directory = ROOT / output
    directory.mkdir(parents=True, exist_ok=True)
    reports = []
    for label in ('run-a', 'run-b'):
        folder = directory / label
        folder.mkdir(parents=True, exist_ok=True)
        for algorithm in ('ippo', 'mappo'):
            run_logged([
                f'scripts/train_{algorithm}.py', '--steps', str(steps),
                '--seed', str(training_seed), '--save-path', str((folder / f'{algorithm}.pt').relative_to(ROOT)),
            ], folder / f'{algorithm}-training.log')
        run_logged([
            'scripts/run_tournament.py', '--games', str(games),
            '--max-turns', str(max_turns), '--seed', str(evaluation_seed),
            '--ippo-path', str((folder / 'ippo.pt').relative_to(ROOT)),
            '--mappo-path', str((folder / 'mappo.pt').relative_to(ROOT)),
            '--results-dir', str(folder.relative_to(ROOT)),
        ], folder / 'evaluation.log')
        reports.append(json.loads((folder / 'tournament.json').read_text(encoding='utf-8')))
    weights = {
        algorithm: compare_checkpoint_weights(directory / 'run-a' / f'{algorithm}.pt',
                                               directory / 'run-b' / f'{algorithm}.pt')
        for algorithm in ('ippo', 'mappo')
    }
    compared_fields = ('games', 'wins', 'ties', 'win_rates', 'confidence_95')
    evaluation_equal = all(reports[0][field] == reports[1][field] for field in compared_fields)
    previous_path = ROOT / 'results/smoke/tournament.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else None
    same_config = previous is not None and previous['seed'] == evaluation_seed and previous['max_turns'] == max_turns and len(previous['games']) == games
    previous_equal = all(reports[0][field] == previous[field] for field in compared_fields) if same_config else None
    passed = evaluation_equal and all(
        comparison['metadata_equal'] and all(comparison[group]['exact'] for group in ('actor_state_dict', 'critic_state_dict'))
        for comparison in weights.values()
    )
    result = {
        'protocol': 'Two separate Python processes per algorithm, fresh initialized models; no checkpoint input',
        'training': {'requested_env_steps': steps, 'seed': training_seed, 'device': 'cpu'},
        'evaluation': {'games': games, 'max_turns': max_turns, 'seed': evaluation_seed},
        'environment': {
            'python': platform.python_version(), 'platform': platform.system(),
            'packages': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'pettingzoo', 'gymnasium')},
        },
        'weights': weights,
        'evaluation_fields_compared': list(compared_fields),
        'evaluation_equal': evaluation_equal,
        'previous_v1_smoke_equal': previous_equal,
        'win_rates': reports[0]['win_rates'],
        'ties': reports[0]['ties'],
        'passed': passed,
        'limits': 'Same-seed CPU replay within this software/hardware environment only. Not cross-platform determinism, training-seed robustness, convergence or superiority.',
        'checkpoint_hash_note': 'Serialization files may differ even when every parameter is identical. Compare tensor values, not just file hashes.',
    }
    (directory / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not passed:
        raise AssertionError('Independent reproduction failed; inspect comparison.json and logs')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=2048)
    parser.add_argument('--games', type=int, default=8)
    parser.add_argument('--max-turns', type=int, default=50)
    parser.add_argument('--training-seed', type=int, default=42)
    parser.add_argument('--evaluation-seed', type=int, default=100)
    parser.add_argument('--output', default='results/reproduction')
    args = parser.parse_args()
    reproduce(args.steps, args.games, args.max_turns, args.training_seed, args.evaluation_seed, args.output)
