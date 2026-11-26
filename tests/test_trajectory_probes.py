from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from cosmos3_h_embodied.scripts import pretrain_trajectory_codes as train
from cosmos3_h_embodied.scripts import trajectory_probes as probes


def test_mixed_windows_preserve_exact_inclusion_probabilities():
    n, count = 147, 8
    before, after = Counter(), Counter()
    mixed = ordinary = 0
    for start in range(n - count + 1):
        orig = range(start, start + count)
        pos = probes.mixed_positions(n, count, start, step=37)
        assert len(pos) == len(set(pos)) == count
        assert all(a // 64 == b // 64 for a, b in zip(orig, pos))
        before.update(orig); after.update(pos)
        # A fixture with adjacent episodes grouped by root.
        mixed += len({i // 16 for i in pos}) > 1
        ordinary += len({i // 16 for i in orig}) > 1
    assert before == after and mixed > ordinary


def test_no_camera_reads_only_kinematics(monkeypatch):
    p, a = torch.randn(42, 20), torch.randn(42, 20)
    monkeypatch.setattr(train, 'kinematics', lambda e, d, phase, normalize: (p, a))
    monkeypatch.setattr(train.Observations, 'images', lambda *args: pytest.fail('decoded camera'))
    obs = train.Observations(None, {'mean': [0]*20, 'std': [1]*20}, camera=False)
    rgb, pose, act = obs.sequence('episode')
    model = train.construct(64, 'cpu', camera=False)
    prev = torch.nn.functional.pad(act[None, :-1], (0, 0, 1, 0))
    features = model.features(rgb[None], pose[None], prev)
    assert model.recurrent.input_size == 40 and torch.equal(features, torch.cat([pose[None], prev], -1))
    h, _, _ = model(rgb[None], pose[None], act[None], torch.tensor([42]))
    assert h.shape == (1, 42, 64)


def test_action_scaling_uses_pose_fit_keys_and_excludes_terminal(monkeypatch):
    episodes = [SimpleNamespace(key=('a', i)) for i in range(3)]
    def raw(e, dataset, phase=0, *, normalize):
        assert not normalize
        return torch.ones(3, 20), torch.tensor([1., 3., 999.]).add(e.key[1]*2)[:, None].repeat(1, 20)
    monkeypatch.setattr(probes, 'kinematics', raw)
    fit = {'episodes': [['a', 0], ['a', 2]], 'frames': 6}
    stats = probes.action_fit(SimpleNamespace(_split='train'), episodes, fit)
    assert stats['mean'] == [4]*20 and stats['std'] == pytest.approx([5**.5]*20)
    with pytest.raises(ValueError, match='training split'):
        probes.action_fit(SimpleNamespace(_split='val'), episodes, fit)
    # Apply the frozen scale to a held-out row without recalibration.
    monkeypatch.setattr(train, 'kinematics', lambda e, d, phase, normalize: (torch.ones(2, 20), torch.cat([torch.full((1, 20), 6.), torch.zeros(1, 20)])))
    obs = train.Observations(None, {'mean': [0]*20, 'std': [1]*20}, camera=False, action_stats=stats)
    _, p, a = obs.sequence('held-out')
    torch.testing.assert_close(a[0], torch.full((20,), 2/5**.5)); assert not a[-1].any() and (p == 1).all()


def test_diagnostic_cannot_export(tmp_path):
    torch.save({'factor': 'no-camera'}, tmp_path / 'best.pt')
    with pytest.raises(ValueError, match='cannot be exported'):
        train.export(None, tmp_path, torch.device('cpu'))
