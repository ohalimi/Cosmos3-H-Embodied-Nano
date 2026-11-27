"""Independent recipe-axis and numeric checks for the five single-factor diagnostics."""
from collections import namedtuple
import math

import pytest
import torch
from torch import nn

from cosmos3_h_embodied.model.trajectory_codes import OFFSETS
from cosmos3_h_embodied.scripts import pretrain_trajectory_codes as train
from cosmos3_h_embodied.scripts.trajectory_probes import FACTORS


BASE = (64, 0., 619, OFFSETS, None)
CASES = {
    'wide-hidden': (128, 0., 619, OFFSETS, None),
    'frequency-decay': (64, 1., 619, OFFSETS, None),
    'short-dct-window': (64, 0., 512, OFFSETS, None),
    'single-consistency-lag': (64, 0., 619, (16,), None),
    'cap-sequences': (64, 0., 619, OFFSETS, 512),
}
FIT = {'mean': [0.] * 20, 'std': [1.] * 20}


@pytest.mark.parametrize('factor,expected', [(None, BASE), ('fast-learning-rate', BASE), *CASES.items()])
def test_only_requested_axis_changes_and_recipe_restores(monkeypatch, factor, expected):
    monkeypatch.setattr(train, 'fixed_camera_encoder', lambda *args: (nn.Identity(), 84))
    torch.manual_seed(17)
    model = train.construct(619, 'cpu', factor=factor)
    obs = train.make_observations(None, FIT, factor)
    width = model.recurrent.hidden_size
    assert (width, model.gamma, model.basis.size(0), model.offsets, obs.max_steps) == expected
    assert model.recurrent.input_size == 104 and model.recurrent.num_layers == 2
    assert model.encoder is not None and obs.camera and obs.action_stats is None
    assert model.dictionary.centers.shape == model.dictionary.moment.shape == (512, width)
    assert model.hidden_readout.in_features == model.to_rows[0].in_features == width
    assert model.to_rows[0].out_features == model.code_readout.in_features == 60
    assert model.hidden_readout.out_features == model.code_readout.out_features == 640
    if factor in CASES:
        assert factor in FACTORS and sum(x != y for x, y in zip(expected, BASE)) == 1
    # Save/restore requires just window + factor. The derived basis is nonpersistent.
    restored = train.construct(model.basis.size(0), 'cpu', factor=factor)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored.basis, model.basis, rtol=0., atol=0.)
    assert restored.gamma == model.gamma and restored.offsets == model.offsets
    if width == BASE[0]:
        # A changed objective, window, or sequence cap must not consume RNG and silently
        # change the initial weights. Check actual tensors, not just sizes.
        torch.manual_seed(17)
        ordinary = train.construct(619, 'cpu')
        assert list(ordinary.state_dict()) == list(model.state_dict())
        for name, value in ordinary.state_dict().items():
            assert torch.equal(value, model.state_dict()[name])
    else:
        h = torch.randn(2, 5, width)
        model.dictionary(h, torch.ones(h.shape[:2], dtype=torch.bool))
        assert model.dictionary.assign(h).shape == h.shape[:2]
        with pytest.raises(ValueError, match='hidden width'):
            model.dictionary.assign(h[..., :64])


def test_frequency_weighting_reaches_both_heads_but_not_overlap_or_commitment():
    torch.manual_seed(10)
    flat = train.construct(48, 'cpu', camera=False)
    weighted = train.construct(48, 'cpu', camera=False, factor='frequency-decay')
    weighted.load_state_dict(flat.state_dict(), strict=True)
    hidden = torch.randn(2, 40, 64); code = torch.randn(2, 40, 60)
    poses = torch.randn(2, 40, 20); anchors = torch.tensor([2, 20]); lengths = torch.tensor([39, 40])
    commitment = torch.tensor(.375)
    old = flat.losses(hidden, code, commitment, poses, anchors, lengths)
    got = weighted.losses(hidden, code, commitment, poses, anchors, lengths)
    weights = [math.exp(-k/31.) for k in range(32)]
    for key, head, values in [('reconstruction', weighted.hidden_readout, hidden),
                              ('code_reconstruction', weighted.code_readout, code)]:
        # Independent target construction includes repeated frame zero at both anchors.
        by_anchor = []
        for t in anchors.tolist():
            frames = [poses[:, max(0, j)] for j in range(t-47, t+1)]
            target = weighted.basis.T @ torch.stack(frames, dim=1)
            err = head(values[:, t]).reshape(-1, 32, 20) - target
            per_band = [err[:, k, :].square().mean() for k in range(32)]
            by_anchor.append(sum(w*v for w,v in zip(weights, per_band)) / sum(weights))
        torch.testing.assert_close(got[key], torch.stack(by_anchor).mean())
        assert not torch.isclose(got[key], old[key])
    assert torch.equal(got['overlap'], old['overlap']) and got['commitment'] is commitment


def test_single_lag_ignores_invalid_pairs_and_returns_zero_for_none():
    m = train.construct(48, 'cpu', camera=False, factor='single-consistency-lag')
    h = torch.randn(2, 40, 64); c = torch.randn(2, 40, 60); p = torch.randn(2, 40, 20)
    lengths = torch.tensor([35, 40]); commit = torch.tensor(.7)
    # 18 + 16 = 34 is valid, 19 + 16 = 35 is padding for the shorter sequence.
    result = m.losses(h, c, commit, p, torch.tensor([18,19]), lengths)
    early, late = (m.basis @ m.hidden_readout(h[:, i]).reshape(2, 32, 20) for i in (18,34))
    torch.testing.assert_close(result['overlap'], (early[:, 16:] - late[:, :-16]).square().mean())
    none = m.losses(h, c, commit, p, torch.tensor([19,34]), lengths)
    assert none['overlap'].shape == () and none['overlap'].item() == 0.


def test_sequence_cap_after_clock_selection_before_padding(monkeypatch):
    Episode = namedtuple('Episode', ['size','stride','root'])
    pool = [Episode(n, 2, str(n)) for n in (1023, 1025, 1037)]
    def clock(e, phase):
        return torch.arange(e.size)[phase::e.stride].float()
    def kinematics(e, ds, phase, normalize):
        assert normalize
        pose = clock(e, phase)[:, None].expand(-1,20)
        return pose, pose + 1
    monkeypatch.setattr(train, 'kinematics', kinematics)
    monkeypatch.setattr(train.Observations, 'images', lambda _,e: torch.arange(e.size)[:,None])
    base = train.make_observations(None, FIT)
    capped = train.make_observations(None, FIT, 'cap-sequences')
    for phase in (0, 1):
        for e in pool:
            full = base.sequence(e, phase); short = capped.sequence(e, phase)
            assert len(full[0]) == len(clock(e, phase))
            for x,y in zip(short, full):
                assert torch.equal(x, y[:512])
    import random
    uncapped_batch = base.batch(pool, random.Random(4), 3, 'cpu')
    capped_batch = capped.batch(pool, random.Random(4), 3, 'cpu')
    assert uncapped_batch[-1].tolist() == [512, 513, 519]
    assert capped_batch[-1].tolist() == [512, 512, 512]
    assert capped.last_root_count == base.last_root_count == 3
    assert all(torch.equal(x, y[:, :512]) for x,y in zip(capped_batch[:-1], uncapped_batch[:-1]))


def test_short_dct_keeps_long_anchors_and_does_not_decode_an_empty_overlap():
    model = train.construct(619, 'cpu', camera=False, factor='short-dct-window').double()
    # Compare independent contractions in double: this fixture sums 511 identical large values.
    poses = torch.zeros(1, 516, 20, dtype=torch.float64); poses[:, 0] = 100.; poses[:, -1] = 1.
    h = poses.new_zeros(1, 516, 64); c = poses.new_zeros(1, 516, 60)
    with torch.no_grad():
        model.hidden_readout.weight.zero_(); model.hidden_readout.bias.zero_()
    anchors = torch.tensor([1,515]); lengths = torch.tensor([516])
    terms = model.losses(h, c, torch.tensor(0.), poses, anchors, lengths)
    at_first = torch.cat((poses[:, :1].expand(-1,511,-1), poses[:,1:2]), 1)
    at_last = poses[:, -512:]
    expected = sum((model.basis.T @ x).square().mean() for x in (at_first,at_last)) / 2
    torch.testing.assert_close(terms['reconstruction'], expected)
    # 1+512 is a valid *time* but there is no overlapping history at lag=window.
    assert terms['overlap'].item() == 0.
