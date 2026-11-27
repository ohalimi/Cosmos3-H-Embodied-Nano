"""No weights, tokenizer assets or data. All modules are randomly initialized."""

from contextlib import ExitStack
from hashlib import sha256
import os
import random

import numpy as np
import pytest
import torch

from openh_signals.features import (
    bridge_features,
    codec_features,
    decoder_features,
    pack_coverage,
    summarize,
    wan_features,
)


def _run(factory, forward, observe):
    # Construct independent graphs from identical states. Check the first mismatch
    # by tensor name rather than concatenating rounded values or comparing losses only.
    runs = []
    for enabled in (False, True):
        torch.manual_seed(1271)
        random.seed(23)
        np.random.seed(28)
        model = factory()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        results = {}
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            records = []
            with ExitStack() as stack:
                if enabled:
                    observe(
                        stack,
                        model,
                        lambda name, fields: records.append((name, fields)),
                    )
                y = forward(model)
            # Never install a hook or substitute a kernel in the off graph.
            loss = y.float().square().mean()
            loss.backward()
            opt.step()
            results[step, "prediction"] = y.detach().clone()
            results[step, "loss"] = loss.detach().clone()
            results[step, "cpu_rng"] = torch.get_rng_state()
            if y.is_cuda:
                results[step, "device_rng"] = torch.cuda.get_rng_state(y.device)
            results[step, "host_rng"] = (random.getstate(), repr(np.random.get_state()))
            for k, v in model.state_dict().items():
                results[step, k] = v.detach().clone()
            if enabled:
                assert records
        runs.append(results)
    for k, a in runs[0].items():
        b = runs[1][k]
        if not isinstance(a, torch.Tensor):
            assert a == b, k
        else:
            assert torch.equal(
                a.reshape(-1).contiguous().view(torch.uint8),
                b.reshape(-1).contiguous().view(torch.uint8),
            ), k


def test_partition_sketch_and_absent_groups():
    labels = torch.tensor([0, 0, -1, 2, 0])
    assert pack_coverage(labels, 3) == {
        "ranges": [(0, 0, 2), (2, 3, 4), (0, 4, 5)],
        "counts": [3, 0, 1],
    }
    with pytest.raises(ValueError):
        pack_coverage(torch.tensor([5]), 4)
    x = torch.tensor(
        [[1.0, 2.0], [2.0, 4.0], [900.0, 800.0], [0.0, 0.0], [3.0, 6.0]],
        requires_grad=True,
    )
    state = torch.get_rng_state().clone()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        s = summarize(x, labels, 3, projection=True)
    assert torch.equal(torch.get_rng_state(), state)
    assert s["groups"] == [0, 2] and s["rows"] == [3, 1]
    expected = [
        sum(
            z * (1 if sha256(f"v1|2|{i}|{j}".encode()).digest()[0] >> 7 else -1)
            for i, z in enumerate((2.0, 4.0))
        )
        / 4
        for j in range(16)
    ]
    assert s["repr16"].dtype == torch.float16
    assert s["repr16"][0].tolist() == expected
    assert s["pooled_l2"].tolist() == pytest.approx([20**0.5, 0])
    assert s["rms"][0].item() == pytest.approx((70 / 6) ** 0.5)
    assert not any(v.requires_grad for v in s.values() if isinstance(v, torch.Tensor))


def test_bridge_real_projector():
    from cosmos_framework.model.generator.mot.domain_aware_linear import (
        DomainAwareLinear,
    )

    labels = torch.tensor([0, 1, 2, -1, -1, -1])
    sources = {}

    def attach(stack, m, emit):
        def collect(name, fields):
            assert fields["rows"] == [1, 1, 1]
            assert fields["repr16"].shape == (3, 16)
            sources[name] = True
            emit(name, fields)

        stack.enter_context(bridge_features(m, "action", labels, 3, collect))

    _run(
        lambda: DomainAwareLinear(4, 8, 2),
        lambda m: m(torch.randn(2, 3, 4), torch.tensor([0, 1])),
        attach,
    )
    assert set(sources) == {"action.in", "action.out"}


def test_wan_spatial_reference():
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import (
        AttentionBlock,
    )

    torch.manual_seed(13)
    block = AttentionBlock(4)
    x = torch.randn(1, 4, 2, 3, 5)
    content = torch.ones(1, 2, 3, 5, dtype=torch.bool)
    content[:, :, 0, :] = False
    events = []
    with wan_features(block, lambda k, v: events.append((k, v)), content_mask=content):
        block(x)
    records = [r for k, r in events if k == "encoder.spatial"]
    assert [r["plane"] for r in records] == [0, 1]
    # Reference every real key on the plane, not per-quadrant softmax.
    for t in range(2):
        q, k, v = (
            block.to_qkv(block.norm(x[:, :, t])).squeeze(0).flatten(1).T.chunk(3, 1)
        )
        p = torch.softmax(q.float() @ k.float().T / 2, 1)
        groups = torch.tensor(
            [
                2 * (2 * i + 1 >= 3) + (2 * j + 1 >= 5) + 4 * (not content[0, t, i, j])
                for i in range(3)
                for j in range(5)
            ]
        )
        for i in set(groups.tolist()):
            for j in range(8):
                expected = p[groups == i][:, groups == j].sum(1).mean()
                assert records[t]["mass"][0, i, j].item() == pytest.approx(
                    expected.item(), abs=2e-6
                )
            expected = -torch.xlogy(p[groups == i], p[groups == i]).sum(1).mean()
            assert records[t]["entropy"][0, i].item() == pytest.approx(
                expected.item(), abs=2e-6
            )
        rms = (p @ v).square().mean().sqrt()
        assert events[-1][1]["plane_rms"][0, t].item() == pytest.approx(
            rms.item(), abs=1e-6
        )


def test_real_dense_ffn_parity():
    from types import SimpleNamespace
    from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import (
        Qwen3VLTextMLP,
    )
    from openh_signals.features import ffn_features

    labels = torch.tensor([0, -1, 0, 1])

    def factory():
        return Qwen3VLTextMLP(
            SimpleNamespace(hidden_size=8, intermediate_size=16, hidden_act="silu")
        )

    def attach(stack, m, emit):
        def check(name, fields):
            assert fields["groups"] == [0, 1] and fields["rows"] == [2, 1]
            emit(name, fields)

        stack.enter_context(ffn_features(m, labels, 2, check))

    _run(factory, lambda m: m(torch.randn(5, 8)), attach)


def test_real_codec_parity():
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import WanVAE_

    seen = set()

    def observe(stack, m, emit):
        def capture(name, fields):
            seen.add(name)
            emit(name, fields)

        stack.enter_context(codec_features(m, capture, encode=True, decode=True))
        for tower in (m.encoder, m.decoder):
            stack.enter_context(wan_features(tower.middle[1], capture))

    def forward(m):
        pred, z = m(torch.randn(1, 3, 1, 32, 32), (torch.zeros(4), torch.ones(4)))
        return torch.cat([z.flatten(), pred.flatten()])

    _run(
        lambda: WanVAE_(
            dim=4, dec_dim=4, z_dim=4, dim_mult=[1, 1, 1, 1], num_res_blocks=1
        ),
        forward,
        observe,
    )
    assert seen == {
        "encoder.spatial",
        "encoder.spatial.value",
        "encoder.bridge.encode.in",
        "encoder.bridge.encode.out",
        "encoder.bridge.mu",
        "encoder.bridge.decode.in",
        "encoder.bridge.decode.out",
    }


@pytest.mark.skipif(
    os.getenv("OPENH_TEST_CUDA") != "1",
    reason="opt-in GPU calibration inside allocation only",
)
def test_real_decoder_parity():
    from cosmos_framework.model.generator.mot.unified_mot import (
        LayerTypes,
        MoTDecoderLayer,
        Qwen3VLTextConfig,
    )
    from cosmos_framework.model.generator.mot.attention import build_packed_sequence
    from cosmos_framework.data.generator.sequence_packing.runtime import (
        from_und_gen_splits,
    )

    device = "cuda"
    und, gen = torch.tensor([0, 0, 1]), torch.tensor([2, 2, 3, 3, 3])
    seen = set()

    def factory():
        config = Qwen3VLTextConfig(
            hidden_size=128,
            intermediate_size=128,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            num_hidden_layers=1,
        )
        return MoTDecoderLayer(
            config,
            layer_idx=0,
            layer_types=LayerTypes("qwen3_vl_dense"),
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
        ).to(device, torch.bfloat16)

    def forward(m):
        p, mask, _ = build_packed_sequence(
            "two_way",
            packed_sequence=torch.randn(8, 128, device=device, dtype=torch.bfloat16),
            attn_modes=["causal", "full"],
            split_lens=[3, 5],
            sample_lens=[8],
            packed_und_token_indexes=torch.arange(3, device=device),
            packed_gen_token_indexes=torch.arange(3, 8, device=device),
            num_heads=2,
            head_dim=64,
            num_layers=1,
        )

        def rotary(fn):
            return from_und_gen_splits(
                fn(torch.randn_like(p["causal_seq"][:, :64])),
                fn(torch.randn_like(p["full_only_seq"][:, :64])),
                p,
            )

        o = m(p, mask, (rotary(torch.cos), rotary(torch.sin)))[0]
        return torch.cat([o["causal_seq"][:3], o["full_only_seq"][:5]])

    def attach(stack, m, emit):
        def capture(name, fields):
            seen.add((name, fields["tower"]))
            emit(name, fields)

        stack.enter_context(decoder_features(m, 0, und, gen, 4, capture))

    _run(factory, forward, attach)
    assert seen == {
        (name, side)
        for name in [
            "attn.out",
            "ffn_gate.silu",
            "ffn_gate.product",
            "residual",
            "embedding",
            "block_exit",
        ]
        for side in ["und", "gen"]
    }
