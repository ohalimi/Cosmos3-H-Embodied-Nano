"""CPU-only graph and export contracts; tensors here are not experiment data."""
import pytest
import torch
from torch import nn

from cosmos3_h_embodied.model.trajectory_codes import EpisodeDictionary, TrajectoryCodes


class TinyObs(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(3, 4)

    def forward(self, obs):
        return torch.cat([self.visual(obs['left'].mean((-2, -1))), obs['proprio']], -1)


def sample():
    torch.manual_seed(41)
    n, t = 2, 42
    return (TrajectoryCodes(TinyObs(), 9, 48), torch.rand(n, t, 3, 4, 4), torch.randn(n, t, 5),
            torch.randn(n, t, 20), torch.tensor([t - 3, t]))


def test_auxiliary_is_the_pretraining_gradient_for_projection():
    m, rgb, p, a, lengths = sample()
    h, codes, commit = m(rgb, p, a, lengths)
    losses = m.losses(h, codes, commit, a, torch.tensor([4, 12, 17, 30]), lengths)
    main = losses['reconstruction'] + losses['overlap']
    assert torch.autograd.grad(main, m.to_rows[0].weight, allow_unused=True, retain_graph=True)[0] is None
    losses['code_reconstruction'].backward(retain_graph=True)
    for layer in [m.to_rows[0], m.code_readout, m.encoder.visual]:
        assert layer.weight.grad.isfinite().all() and layer.weight.grad.norm() > 0
    m.zero_grad(set_to_none=True)
    sum(losses.values()).backward()
    assert m.hidden_readout.weight.grad.norm() > 0
    assert m.dictionary.centers.grad is None
    assert m.dictionary.centers.shape == (512, 64)
    torch.testing.assert_close(m.basis.T @ m.basis, torch.eye(32))


def test_export_chunking_is_causal_and_frozen():
    m, rgb, p, a, lengths = sample()
    m(rgb, p, a, lengths)
    m.eval()
    h, codes, _ = m(rgb, p, a, lengths)
    snapshot = {k: v.clone() for k, v in m.state_dict().items()}
    expected = codes[0].reshape(-1, 3, 20)
    track = m.track(rgb[0], p[0], a[0], chunk=3)
    torch.testing.assert_close(track, expected)
    # At decision t, observation t is available; command t and observation t+1 are not.
    rgb[:, 14:] = 0
    p[:, 14:] += 10
    a[:, 13:] += 50
    other, _, _ = m(rgb, p, a, lengths)
    torch.testing.assert_close(other[:, :14], h[:, :14])
    assert not torch.allclose(other[:, 14:], h[:, 14:])
    for k, v in m.state_dict().items():
        torch.testing.assert_close(v, snapshot[k])


def test_objective_matches_independent_window_reference():
    m, rgb, p, a, lengths = sample()
    h, c, commitment = m(rgb, p, a, lengths)
    anchors = torch.tensor([2, 16, 30])
    losses = m.losses(h, c, commitment, a, anchors, lengths)
    def coeff(t, head, values):
        return head(values[:, t]).view(len(a), 32, 20)
    ref1, ref2, ref_overlap = [], [], []
    for t in anchors.tolist():
        past = a[:, max(0, t - 47):t + 1]
        past = torch.cat([a[:, :1].expand(-1, 48 - past.size(1), -1), past], dim=1)
        target = m.basis.T @ past
        ref1.append((coeff(t, m.hidden_readout, h) - target).square().mean())
        ref2.append((coeff(t, m.code_readout, c) - target).square().mean())
    for delta in (1, 2, 4, 8, 16, 32):
        comparisons = []
        for t in anchors.tolist():
            if t + delta >= int(lengths.min()):
                continue
            left = m.basis @ coeff(t, m.hidden_readout, h)
            right = m.basis @ coeff(t + delta, m.hidden_readout, h)
            comparisons.append((left[:, delta:] - right[:, :-delta]).square().mean())
        if comparisons:
            ref_overlap.append(torch.stack(comparisons).mean())
    for name, reference in [('reconstruction', ref1), ('code_reconstruction', ref2), ('overlap', ref_overlap)]:
        torch.testing.assert_close(losses[name], torch.stack(reference).mean())


def test_padding_cannot_influence_dictionary_or_commitment():
    x = torch.randn(2, 80, 64)
    mask = torch.arange(80) < torch.tensor([[78], [80]])
    altered = x.clone()
    altered[~mask] = -10000
    a, b = EpisodeDictionary(), EpisodeDictionary()
    for i in range(2):
        torch.manual_seed(i)
        _, loss = a(x, mask)
        torch.manual_seed(i)
        _, alternate = b(altered, mask)
        torch.testing.assert_close(loss, alternate)
    for k, v in a.state_dict().items():
        torch.testing.assert_close(v, b.state_dict()[k])
    assert a.usage() == b.usage()
    assert 1 <= a.usage()['perplexity'] <= a.usage()['used_codes'] + 1e-4
    assert a.usage()['used_codes'] <= mask.sum() and a.usage()['dictionary_size'] == 512
    # A large EMA mass is not evidence that the current batch uses those codes.
    a(torch.zeros_like(x), mask)
    assert a.usage()['used_codes'] == a.usage()['perplexity'] == 1
    with pytest.raises(RuntimeError, match='no trained dictionary'):
        EpisodeDictionary().eval()(x, mask)
