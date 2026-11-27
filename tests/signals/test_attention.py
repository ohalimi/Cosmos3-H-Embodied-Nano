"""Synthetic tests, never a trainer/loader/evaluator. No pretrained constructors."""
import inspect
import random

import numpy as np
import pytest
import torch
from cosmos_framework.data.generator.sequence_packing.runtime import (
    get_all_seq_unpadded, sequence_pack_from_packed_sequence,
)
from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear

from openh_signals.attention import attention_summary, observe_two_way


# Absolute mixture tolerance fixed per storage dtype (inputs cast, reference stays FP32).
TOL = {torch.float32: 2e-6, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}


@pytest.mark.parametrize('dtype', list(TOL))
def test_cpu_sdpa(dtype):
    g = torch.Generator().manual_seed(4)
    q, k, v = [torch.randn(9, h, 16, generator=g).to(dtype) for h in (4, 2, 2)]
    ids = torch.arange(9) % 3
    p = attention_summary(q, k, v, ids, ids, 4, tile=2, causal=True, calibration=True)
    qh, kh, vh = [a.transpose(0, 1)[None] for a in (q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1))]
    expected = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
    torch.testing.assert_close(p['mixture'], expected[0].transpose(0, 1).float(), atol=TOL[dtype], rtol=0)
    assert p['mass'][:, 3].isnan().all() and p['mass'][:, :3, 3].count_nonzero() == 0
    assert p['max_row_error'] <= 1e-5
    # Independent oracle for the head and group axes, not only the mixture.
    logits = qh[0].double() @ kh[0].double().transpose(1, 2) / 4
    probs = logits.masked_fill(torch.ones(9, 9, dtype=torch.bool).triu(1), -torch.inf).softmax(-1)
    for a in range(3):
        rows = probs[:, ids == a]
        for b in range(3):
            torch.testing.assert_close(p['mass'][:, a, b], rows[:, :, ids == b].sum(-1).mean(1).float())
        torch.testing.assert_close(p['entropy'][:, a], -torch.special.xlogy(rows, rows).sum(-1).mean(1).float())


def packs(q, k, v, ctor=sequence_pack_from_packed_sequence):
    ci = torch.tensor([0, 1, 6, 7, 8], device=q.device)
    fi = torch.tensor([2, 3, 4, 5, 9, 10], device=q.device)
    return {f'packed_{key}_states': ctor(a, ['causal', 'full']*2, [2, 4, 3, 2], [6, 5], ci, fi)
            for key, a in zip(('query', 'key', 'value'), (q, k, v))}


def labels(device):
    return torch.tensor([0, 1, 2, 2, 3, 3, 0, 1, 1, 2, 3], device=device)


def test_real_packer_blocks_and_normalized_keys():
    torch.manual_seed(10)
    q, k, v = [torch.randn(11, h, 16) for h in (4, 2, 2)]
    args = packs(q, k, v)
    args['packed_key_states_normalized'] = packs(q, -k, v)['packed_key_states']
    obs = observe_two_way(args, labels('cpu'), 4, 0, tile=2, calibration=True)
    assert len(obs) == 2
    for start, end, c, record in zip((0, 6), (6, 11), (2, 3), obs):
        assert record['mass'][:, :2, 2:].count_nonzero() == record['legal'][:2, 2:].count_nonzero() == 0
        torch.testing.assert_close(record['mass'].sum(-1), torch.ones(4, 4), atol=1e-5, rtol=0)
        for idx in range(start, end):
            n = idx + 1 if idx < start+c else end
            keys = k if idx < start+c else -k
            z = torch.einsum('hd,khd->hk', q[idx], keys[start:n].repeat_interleave(2, 1)) / 4
            ref = torch.einsum('hk,khd->hd', z.softmax(-1), v[start:n].repeat_interleave(2, 1))
            torch.testing.assert_close(ref, record['mixture'][idx-start])
    with pytest.raises(ValueError, match='group'):
        observe_two_way(args, torch.zeros(11, dtype=torch.long), 4, 0)


def test_value_cancellation_full_head_and_prompt_integrity():
    from openh_signals.attention import FrozenText
    q = torch.zeros(11, 32, 4)
    k = torch.zeros(11, 8, 4)
    v = torch.ones_like(k); v[3] = -1
    args = packs(q, k, v)
    for layer in (0, 35):
        r, other = observe_two_way(args, labels('cpu'), 4, layer, tile=1, sentinel={0})
        assert 'full_map' not in other
        matrix = r['full_map']
        assert matrix.shape == (6, 6) and matrix.dtype == torch.float16
        assert matrix[0, 0] == 1 and matrix[0, 1:].count_nonzero() == 0
        assert torch.all(matrix[1, :2] == .5) and matrix[1, 2:].count_nonzero() == 0
        assert torch.all(matrix[2:] == torch.tensor(1/6, dtype=torch.float16))
        torch.testing.assert_close(r['mass'][:, 3, 2], torch.full((32,), 1/3))
        assert r['read_strength'][3, 2] == 0  # opposite V, same key-group mass
        index = FrozenText(); first = index.apply('same-binding', r, 'first')
        assert first['mass'].shape == (32, 4, 4)
        restored = FrozenText(index.state_dict())
        later = restored.apply('same-binding', r, 'later')
        assert later['mass'].shape == (32, 2, 4) and later['text_reference'] == 'first'
        assert later['mot_query_groups'].tolist() == [2, 3]
        assert torch.equal(later['full_map'], matrix)
        args2 = packs(q+1, k.clone(), v)
        args2['packed_key_states']['causal_seq'][1] += 1
        changed = observe_two_way(args2, labels('cpu'), 4, layer, tile=1)[0]
        with pytest.raises(ValueError, match='frozen text'):
            restored.apply('same-binding', changed, 'corrupt')


def bytecopy(x):
    return x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).clone()


def protocol(enabled, device='cpu', dtype=torch.float32, native=False, after=None):
    """Three updates from identical seeds. CPU fixture uses the real action projector
    and PyTorch SDPA; on GPU use the actual unmodified two-way kernel dispatch.
    """
    torch.manual_seed(183); random.seed(5); np.random.seed(6)
    ctor = sequence_pack_from_packed_sequence
    if native:
        from cosmos3._src.vfm.datasets.sequence_packing import factored_from_joint_sequence, get_all_seq
        from cosmos3._src.vfm.models.mot.attention import two_way_attention
        ctor, unpack = factored_from_joint_sequence, get_all_seq
    else:
        from cosmos_framework.model.generator.mot.attention import two_way_attention
        unpack = get_all_seq_unpadded
    d = 64
    # FP32 master parameters; the tiny projected Q/K/V exercise the selected compute dtype.
    m = DomainAwareLinear(8, 8*d, 2).to(device)
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    for step in range(1, 4):
        opt.zero_grad(set_to_none=True)
        x = torch.randn(11, 8, device=device)
        y = m(x, torch.arange(11, device=device) % 2).to(dtype).view(11, 8, d)
        q, k, v = y.split((4, 2, 2), 1)
        args = packs(q, k, v, ctor)
        obs = observe_two_way(args, labels(device), 4, 0, tile=2, calibration=True, sentinel=(0,)) if enabled else None
        if device == 'cpu':
            parts = []
            for a, c, z in ((0, 2, 6), (6, 9, 11)):
                for qs, ks, causal in ((slice(a, c), slice(a, c), True), (slice(c,z), slice(a,z), False)):
                    tensors = [t.transpose(0, 1).unsqueeze(0) for t in
                               (q[qs], k[ks].repeat_interleave(2, 1), v[ks].repeat_interleave(2, 1))]
                    parts.append(torch.nn.functional.scaled_dot_product_attention(*tensors, is_causal=causal)[0].transpose(0, 1))
            pred = torch.cat(parts)
        else:
            pred = unpack(two_way_attention(**args)).reshape(11, 4, d)
        if obs:
            torch.testing.assert_close(pred.float(), torch.cat([r['mixture'] for r in obs]), rtol=0, atol=TOL[dtype])
        loss = pred.float().square().mean()
        loss.backward()
        opt.step()
        if enabled and after:
            after(step, obs)
        record = {'batch':bytecopy(x), 'prediction':bytecopy(pred), 'loss':bytecopy(loss),
                  'torch_rng':torch.get_rng_state(), 'py_rng':random.getstate(), 'np_rng':repr(np.random.get_state())}
        if device == 'cuda':
            record['device_rng'] = torch.cuda.get_rng_state()
        for name, par in m.named_parameters():
            record[name], record['grad:'+name] = bytecopy(par), bytecopy(par.grad)
        yield record


def assert_protocol(**kwargs):
    base, observed = [list(protocol(flag, **kwargs)) for flag in (False, True)]
    assert not torch.equal(base[0]['fc.weight'], base[-1]['fc.weight'])
    for step, (a, b) in enumerate(zip(base, observed), 1):
        for name in a:
            equal = torch.equal(a[name], b[name]) if torch.is_tensor(a[name]) else a[name] == b[name]
            assert equal, f'first mismatch: step {step}, {name}'


def test_cpu_bitwise():
    assert_protocol()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires the installed real varlen GPU backend')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('native', [False, True])
def test_real_gpu_kernel(dtype, native):
    assert_protocol(device='cuda', dtype=dtype, native=native)
