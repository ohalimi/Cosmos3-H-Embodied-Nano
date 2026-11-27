"""Check affine calibration and tail clipping independently, including episode boundaries."""
from types import SimpleNamespace

import pytest
import torch

from cosmos3_h_embodied.data.action_normalization import normalize_action
from cosmos3_h_embodied.scripts import pretrain_trajectory_codes as train


@pytest.mark.parametrize('split', ['train', 'val'])
def test_affine_and_clipping_factors_change_only_their_axis(monkeypatch, split):
    scale = {'q01': torch.full((20,), -4.), 'q99': torch.full((20,), 4.)}
    # Exercise the denominator floor on one held-constant coordinate.
    scale['q99'][-1] = scale['q01'][-1]
    ds = SimpleNamespace(_split=split, _action_normalization='quantile', _load_norm_stats=lambda: scale)
    pose = torch.arange(80).reshape(4,20).float()
    raw = torch.tensor([-8., 0., 8.])[:,None].expand(-1,20).clone()
    terminal = torch.zeros(1,20)
    def kinematics(e, dataset, phase, *, normalize):
        assert dataset is ds and phase == 0
        body = normalize_action(raw, 'quantile', scale) if normalize else raw
        return pose, torch.cat((body,terminal))
    monkeypatch.setattr(train, 'kinematics', kinematics)
    monkeypatch.setattr(train.Observations, 'images', lambda *_: torch.arange(4)[:,None])
    fit = {'mean': [1.]*20, 'std': [2.]*20}; z = (raw-1.)/2.
    # Provide a hashable episode key to the sequence reader's cache.
    class Episode:
        stride = 1
    e = Episode()
    q = raw/4.; q[:,-1] = (raw[:,-1]+4.)*2/1e-8-1.
    for factor, scaling, expected in [(None,None,q.clamp(-1.5,1.5)),
                                      ('clipped-z-actions',fit,z.clamp(-1.5,1.5)),
                                      ('unclipped-quantile-actions',None,q),
                                      ('internal-action-scaling',fit,z)]:
        obs = train.make_observations(ds, {'mean':[0.]*20,'std':[1.]*20}, factor, scaling)
        rgb, p, action = obs.sequence(e)
        assert obs.camera and obs.max_steps is None and rgb[:,0].tolist() == [0,1,2,3]
        assert torch.equal(p, pose) and torch.equal(action[-1:],terminal)
        assert torch.equal(action[:-1],expected)
        prev = torch.nn.functional.pad(action[:-1], (0,0,1,0))
        assert not prev[0].any() and torch.equal(prev[1:],expected)
    with pytest.raises(ValueError, match='training action fit'):
        train.make_observations(ds, {'mean':[0.]*20,'std':[1.]*20}, 'clipped-z-actions', None)
    with pytest.raises(ValueError, match='training action fit'):
        train.make_observations(ds, {'mean':[0.]*20,'std':[1.]*20}, 'unclipped-quantile-actions', fit)
