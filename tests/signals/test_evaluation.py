"""Synthetic, no source windows or saved model state. GPU tests never allocate a job."""

from contextlib import nullcontext
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch
from cosmos_framework.data.generator.sequence_packing.runtime import (
    sequence_pack_from_packed_sequence,
)
from cosmos_framework.model.generator.mot import inference_text_kv_memory as kv
from cosmos_framework.model.generator.mot.attention import SplitInfo
from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear

from openh_signals.evaluation import (
    ClipStrata,
    cached_attention,
    cached_summary,
    eval_score_hook,
)
from openh_signals.runtime import Runtime, scoped


def test_native_cache_install_and_restore():
    rt = Runtime({}, lambda *a: None)
    original = kv.dispatch_attention
    attn = SimpleNamespace(dispatch_attention_fn=original)
    net = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])
        )
    )
    with cached_attention(rt, kv):
        previous = kv.install_inference_memory_attention_dispatch(net)
        assert attn.dispatch_attention_fn is kv.dispatch_attention_with_text_kv_memory
        kv.restore_inference_attention_dispatch(previous)
    assert attn.dispatch_attention_fn is original


def test_equal_clip_not_equal_token_weight():
    s = ClipStrata([("same",)], max_instances=2)
    s.add(("same",), "a", mass=torch.tensor([0.25, torch.nan]))
    s.add(("same",), "b", mass=torch.tensor([0.75, 0.0]))
    assert s.mean(("same",))["mass"].tolist() == [0.5, 0.0]
    assert s.counts(("same",))["mass"].tolist() == [2, 1]
    with pytest.raises(ValueError, match="replayed"):
        s.add(("same",), "a", mass=torch.zeros(2))


def test_real_scorer(tmp_path):
    from cosmos3_h_embodied.scripts.eval_utils import compute_sample_metrics

    # The evaluator imports the reference by value; wrap that binding, not the definition.
    mod = SimpleNamespace(compute_sample_metrics=compute_sample_metrics)
    gt = torch.zeros(5, 4)
    pred = gt + 2
    pred[:3] = 100
    sample = SimpleNamespace(
        outputs=[SimpleNamespace(content={"action": pred.tolist()})]
    )
    seen = []
    args = ("synthetic/inverse_dynamics/sample", None, gt, sample, tmp_path, ".mp4")
    mask = torch.arange(5) >= 3
    base = mod.compute_sample_metrics(*args, action_prediction_mask=mask)
    with eval_score_hook(mod, lambda name, row: seen.append(row)):
        out = mod.compute_sample_metrics(*args, action_prediction_mask=mask)
    assert base == out and seen[0]["action_mse"] == base["action_mse"] == 4.0
    assert (seen[0]["excluded_rows"], seen[0]["valid_elements"]) == (3, 8)


def trace(mode, device="cpu", dtype=torch.float32, multi=True):
    torch.manual_seed(849)
    random.seed(849)
    np.random.seed(849)
    batches = [[0, 1, 3, 7, 7], [0, 1, 1, 4, 6]] if multi else [[0, 1, 3, 7, 7]]
    lens = [len(g) for g in batches]
    texts = [sum(i < 3 for i in g) for g in batches]
    groups = torch.tensor(sum(batches, []), device=device)
    spans = ["causal", "full"] * len(batches)
    ci = (groups < 3).nonzero().flatten()
    fi = (groups >= 3).nonzero().flatten()
    masks = []
    for n, t in zip(lens, texts):
        masks += [t, n - t]
    net = DomainAwareLinear(8, 8 * 64, 2).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=0.2)

    def operands():
        x = torch.randn(sum(lens), 8, device=device)
        y = (
            net(x, torch.arange(len(x), device=device) % 2)
            .to(dtype)
            .reshape(len(x), 8, 64)
        )
        return tuple(
            sequence_pack_from_packed_sequence(z, spans, masks, lens, ci, fi)
            for z in y.split((4, 2, 2), 1)
        )

    rows = []
    rng = []
    state = kv.InferenceTextKVMemoryState(kv.make_inference_text_kv_cache(1))
    rt = Runtime({}, lambda n, row: rows.append(row))
    recs = []
    start = 0
    for i, n in enumerate(lens):
        recs.append(dict(offset=start, length=n, text_digest="s" + str(i)))
        start += n
    branches = ("positive", "negative") if multi else ("positive",)
    cell = dict(kind="scored", branches=branches, cache_sentinels=(0,), full_map=(0,))
    ctor = cached_attention(rt, kv) if mode and device == "cuda" else nullcontext()
    with (
        scoped(rt.current, cell),
        scoped(rt.groups, (groups, set(), recs)),
        scoped(rt.layer, 0),
        ctor,
    ):
        pq, pk, pv = operands()
        state.init(pq, torch.device(device))
        # The scorer is unchanged. Only GPU has an installed release attention kernel.
        if device == "cuda":
            with scoped(rt.solver, (0, 3, True, 3)):
                kv.dispatch_attention_with_text_kv_memory(
                    pq,
                    pk,
                    pv,
                    SplitInfo(masks, spans, lens, sum(lens)),
                    memory_value=state.read_for_layer(0),
                )
        # Native cache writes trim both alignment tails before storage.
        state.write_for_layer(
            0,
            (
                pk["full_only_seq"][: len(fi)][None],
                pv["full_only_seq"][: len(fi)][None],
                pk["causal_seq"][: len(ci)][None],
                pv["causal_seq"][: len(ci)][None],
            ),
        )
        memory = state.read_for_layer(0)
        layout = [(torch.tensor(g, device=device), t) for g, t in zip(batches, texts)]
        for i in (1, 2):
            opt.zero_grad(set_to_none=True)
            pq, pk, pv = operands()
            q = pq["full_only_seq"][None]
            k = torch.cat(
                (memory.und_k_cached, pk["full_only_seq"][: len(fi)][None]), 1
            )
            v = torch.cat(
                (memory.und_v_cached, pv["full_only_seq"][: len(fi)][None]), 1
            )
            if multi:
                q = q[:, : len(fi)]
                k = k.index_select(1, memory.kv_reorder_indices)
                v = v.index_select(1, memory.kv_reorder_indices)
            kwargs = dict(
                query=q,
                key=k,
                value=v,
                cumulative_seqlen_Q=memory.cumulative_seqlen_q,
                cumulative_seqlen_KV=memory.cumulative_seqlen_kv,
            )
            if device == "cpu":
                # CPU fixture uses NATIVE cache write/reorder + separate unmodified PyTorch SDPA per sample.
                pred = []
                a = b = 0
                for g, t in layout:
                    nq = len(g) - t
                    qs, ks, vs = (
                        q[:, a : a + nq],
                        k[:, b : b + len(g)].repeat_interleave(2, 2),
                        v[:, b : b + len(g)].repeat_interleave(2, 2),
                    )
                    pred.append(
                        torch.nn.functional.scaled_dot_product_attention(
                            qs.transpose(1, 2), ks.transpose(1, 2), vs.transpose(1, 2)
                        ).transpose(1, 2)
                    )
                    a += nq
                    b += len(g)
                pred = torch.cat(pred, 1)[0]
                if mode:
                    summaries = cached_summary(kwargs, layout, 0, calibration=True)
                    torch.testing.assert_close(
                        torch.cat([r["mixture"] for r in summaries]),
                        pred,
                        atol=2e-6,
                        rtol=0,
                    )
                    assert all(
                        r["counts"][:3].sum() == 0 and r["mass"][:, :3].isnan().all()
                        for r in summaries
                    )
                    with pytest.raises(ValueError, match="unmeasured"):
                        cached_summary(kwargs, layout, 0, sentinel=(0,))
            else:
                rt.groups.get()[1].clear()
                with scoped(rt.solver, (i, 3, True, 3)):
                    out, _ = kv.dispatch_attention_with_text_kv_memory(
                        pq, pk, pv, None, memory_value=memory
                    )
                pred = out["full_only_seq"]
                if mode:
                    row = rows[-len(lens)]
                    assert (
                        row["cell"]["solver_step"] == i
                        and row["cell"]["branch"] == "positive"
                    )
                    matrix = row["full_map"]
                    assert matrix.shape == (lens[0], lens[0])
                    assert matrix[: texts[0], texts[0] :].count_nonzero() == 0
                    # Named head from its stored fp16 probabilities reproduces actual kernel output.
                    mix = matrix[texts[0] :].float() @ v[0, : lens[0], 0].float()
                    torch.testing.assert_close(
                        mix, pred[: lens[0] - texts[0], :64].float(), atol=0.01, rtol=0
                    )
            loss = pred.float().square().mean()
            loss.backward()
            opt.step()

            def bits(x):
                return (
                    x.detach()
                    .cpu()
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )

            rng.append(
                dict(
                    prediction=bits(pred),
                    loss=bits(loss),
                    weights=[bits(p) for p in net.parameters()],
                    torch_rng=bits(torch.get_rng_state()),
                    cuda_rng=bits(torch.cuda.get_rng_state())
                    if device == "cuda"
                    else None,
                    python_rng=random.getstate(),
                    numpy_rng=repr(np.random.get_state()),
                )
            )
    return rng


@pytest.mark.parametrize("multi", [False, True])
def test_real_cache_cpu_noninterference(multi):
    assert trace(False, multi=multi) == trace(True, multi=multi)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="pinned cached-kernel dispatcher requires GPU"
)
@pytest.mark.parametrize("multi", [False, True])
def test_real_cached_kernel(multi):
    assert trace(False, "cuda", torch.bfloat16, multi) == trace(
        True, "cuda", torch.bfloat16, multi
    )
