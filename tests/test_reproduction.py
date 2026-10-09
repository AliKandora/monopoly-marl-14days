import torch

from demo import run_demo
from scripts.reproduce import compare_checkpoint_weights


def test_checkpoint_comparison_uses_tensors(tmp_path):
    a, b = tmp_path / 'first.pt', tmp_path / 'second.pt'
    state = {
        'actor_state_dict': {'weight': torch.tensor([1.0, 2.0])},
        'critic_state_dict': {'weight': torch.tensor([3.0])},
        'metadata': {'seed': 42},
    }
    torch.save(state, a)
    torch.save(state, b)
    result = compare_checkpoint_weights(a, b)
    assert result['actor_state_dict']['exact']
    assert result['critic_state_dict']['exact']
    assert result['actor_state_dict']['max_absolute_difference'] == 0
    assert result['metadata_equal']


def test_checkpoint_comparison_detects_changes(tmp_path):
    a, b = tmp_path / 'first.pt', tmp_path / 'second.pt'
    state = {
        'actor_state_dict': {'weight': torch.tensor([1.0])},
        'critic_state_dict': {'weight': torch.tensor([3.0])},
    }
    torch.save(state, a)
    state['actor_state_dict']['weight'] += 1
    torch.save(state, b)
    result = compare_checkpoint_weights(a, b)
    assert not result['actor_state_dict']['exact']
    assert result['actor_state_dict']['max_absolute_difference'] == 1


def test_demo_preview_does_not_claim_final_winner(capsys):
    run_demo(delay=0, max_steps=1)
    text = capsys.readouterr().out
    assert 'DEMO PREVIEW LIMIT REACHED' in text
    assert 'not final winners' in text
    assert 'GAME OVER' not in text
