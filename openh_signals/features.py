"""Scoped eager feature taps. Bind *one logical cell* and only an original forward.

Row labels come from the caller's physical pack/tokenizer binding. A label of -1
excludes a different sample; it is not a padding, history or caption class.
Emit accepts reduced, detached fields; it must not retain model inputs or outputs.
"""

from contextlib import ExitStack, contextmanager
from functools import lru_cache, wraps
from hashlib import sha256
from unittest.mock import patch

import torch


def _labels(groups, n_groups):
    if (
        groups.ndim != 1
        or groups.dtype != torch.long
        or not 1 <= n_groups <= 12
        or ((groups < -1) | (groups >= n_groups)).any()
    ):
        raise ValueError("invalid physical token partition")


def pack_coverage(groups, n_groups):
    """Half-open physical ranges (not a dump of token IDs) for one bound cell."""
    _labels(groups, n_groups)
    values = groups.detach().cpu().tolist()
    ranges = []
    counts = [0] * n_groups
    start = 0
    while start < len(values):
        group = values[start]
        end = start + 1
        while end < len(values) and values[end] == group:
            end += 1
        if group >= 0:
            counts[group] += end - start
            ranges.append((group, start, end))
        start = end
    return {"ranges": ranges, "counts": counts}


@lru_cache(maxsize=8)
def _signs(width):
    # Big-endian first bit; set -> +1, clear -> -1. No PRNG, on any rank.
    return torch.tensor(
        [
            [
                1 if sha256(f"v1|{width}|{i}|{j}".encode()).digest()[0] & 128 else -1
                for j in range(16)
            ]
            for i in range(width)
        ],
        device="cpu",
        dtype=torch.int8,
    )


@torch.no_grad()
def summarize(x, groups, n_groups, *, projection=False):
    """Grouped FP32 moments and optional signed sketch, over physical real rows."""
    _labels(groups, n_groups)
    if (
        x.ndim != 2
        or x.size(0) != len(groups)
        or not x.is_floating_point()
        or x.size(1) == 0
    ):
        raise ValueError("expected a floating feature matrix aligned to labels")
    groups = groups.to(x.device)
    ids, counts, means, rms = [], [], [], []
    sketches, scales, norms = [], [], []
    basis = _signs(x.size(1)).to(x.device) if projection else None
    for g in range(n_groups):
        rows = x.detach()[groups == g].float()
        if not rows.size(0):
            continue
        ids.append(g)
        counts.append(rows.size(0))
        means.append(rows.mean())
        rms.append(rows.square().mean().sqrt())
        if projection:
            pooled = rows.mean(0)
            # Elementwise reduction avoids autocast/TF32 matmul in the observer.
            z = (pooled[:, None] * basis).sum(0) * 0.25
            scale = (z.abs().max() / torch.finfo(torch.float16).max).clamp_min(1.0)
            sketches.append(z / scale)
            scales.append(scale)
            norms.append(torch.linalg.vector_norm(pooled))

    def vector(xs):
        return (
            torch.stack(xs).cpu()
            if xs
            else torch.empty(0, dtype=torch.float32, device="cpu")
        )

    record = {"groups": ids, "rows": counts, "mean": vector(means), "rms": vector(rms)}
    if projection:
        record.update(
            repr16=(
                torch.stack(sketches).to(device="cpu", dtype=torch.float16)
                if sketches
                else torch.empty(0, 16, dtype=torch.float16, device="cpu")
            ),
            scale=vector(scales),
            pooled_l2=vector(norms),
        )
    return record


def _tower(pack, side):
    key = "causal_seq" if side == "und" else "full_only_seq"
    if key in pack:
        return pack[key]
    return pack["packed_sequence"][pack[f"packed_{side}_token_indexes"]]


@contextmanager
def ffn_features(mlp, groups, n_groups, emit):
    """Observe the actual dense gate output and the product entering down_proj.

    groups has the true tower length (decoder packs may have trailing alignment).
    """
    if getattr(mlp.config, "hidden_act", None) != "silu":
        raise ValueError("only dense SiLU MLPs are bound")

    @torch.no_grad()
    def gate(_, args, out):
        emit("ffn_gate.silu", summarize(out[: len(groups)], groups, n_groups))

    @torch.no_grad()
    def product(_, args):
        emit("ffn_gate.product", summarize(args[0][: len(groups)], groups, n_groups))

    with (
        mlp.act_fn.register_forward_hook(gate),
        mlp.down_proj.register_forward_pre_hook(product),
    ):
        yield


@contextmanager
def decoder_features(block, layer, und_groups, gen_groups, n_groups, emit):
    """Observe both residual additions of a dense MoTDecoderLayer, no replay.

    Bind one non-sharded, non-cached original call. The sequences may include
    alignment rows, but the true per-stream counts must agree with the label
    vector lengths. Ratios with zero input norm are NaN (undefined).
    """
    labels = {"und": und_groups, "gen": gen_groups}
    for g in labels.values():
        _labels(g, n_groups)
    state = {}
    calls = 0

    @torch.no_grad()
    def begin(_, args, kwargs):
        nonlocal calls
        calls += 1
        pack = args[0] if args else kwargs["input"]
        if (
            calls != 1
            or pack.get("is_sharded", False)
            or kwargs.get("gen_only", False)
            or (len(args) > 5 and args[5])
        ):
            raise ValueError(
                "feature cell needs one complete original dual-tower forward"
            )
        for side, cnt in [("und", "_num_causal_tokens"), ("gen", "_num_full_tokens")]:
            x = _tower(pack, side)
            if pack[cnt] != len(labels[side]) or x.ndim != 2 or x.size(0) < pack[cnt]:
                raise ValueError("real token count and feature labels disagree")
            state[side] = {"input": x[: len(labels[side])].detach()}

    def record(side, stage, values):
        return emit(stage, {"layer": layer, "tower": side, **values})

    def mid(side):
        @torch.no_grad()
        def hook(_, args):
            state[side]["post_attn"] = args[0][: len(labels[side])].detach()

        return hook

    def capture(side, stage):
        @torch.no_grad()
        def hook(_, args, out):
            record(
                side, stage, summarize(out[: len(labels[side])], labels[side], n_groups)
            )

        return hook

    @torch.no_grad()
    def end(_, args, output):
        for side, groups in labels.items():
            st = state[side]
            a, b = st["input"], st["post_attn"]
            c = _tower(output[0], side)[: len(groups)].detach()
            x = summarize(a, groups, n_groups, projection=layer == 0)
            if layer == 0:
                record(side, "embedding", x)
            y = summarize(b, groups, n_groups)
            # Deltas are differences of the actual rounded residuals, not the
            # mathematical pre-addend; o_proj is observed separately below.
            d1 = summarize(b.float() - a.float(), groups, n_groups)["rms"]
            d2 = summarize(c.float() - b.float(), groups, n_groups)["rms"]

            def ratio(d, base):
                return torch.where(base > 0, d / base, float("nan"))

            record(
                side,
                "residual",
                {
                    "groups": x["groups"],
                    "rows": x["rows"],
                    "input_rms": x["rms"],
                    "attn_delta_rms": d1,
                    "mlp_delta_rms": d2,
                    "attn_ratio": ratio(d1, x["rms"]),
                    "mlp_ratio": ratio(d2, y["rms"]),
                },
            )
            record(side, "block_exit", summarize(c, groups, n_groups, projection=True))
        state.clear()

    with ExitStack() as stack:
        stack.enter_context(block.register_forward_pre_hook(begin, with_kwargs=True))
        for side, suffix in [("und", ""), ("gen", "_moe_gen")]:
            mlp = getattr(block, "mlp" + suffix)
            stack.enter_context(
                getattr(
                    block, "post_attention_layernorm" + suffix
                ).register_forward_pre_hook(mid(side))
            )
            stack.enter_context(
                getattr(block.self_attn, "o_proj" + suffix).register_forward_hook(
                    capture(side, "attn.out")
                )
            )
            stack.enter_context(
                ffn_features(
                    mlp,
                    labels[side],
                    n_groups,
                    lambda name, fields, s=side: record(s, name, fields),
                )
            )
        stack.enter_context(block.register_forward_hook(end))
        yield
        if calls != 1 or state:
            raise ValueError("decoder feature boundary did not execute completely")


@contextmanager
def bridge_features(module, name, groups, n_groups, emit):
    """An observed token-aligned input/output projection (never a dummy row).

    Supports [N,D] and [B,T,D]; pass labels in the corresponding flattened
    row order. For embedding tables only observe the output, never raw IDs.
    """
    _labels(groups, n_groups)

    @torch.no_grad()
    def hook(_, args, kwargs, out):
        x = args[0] if args else kwargs["x"]

        def features(t):
            if t.ndim not in (2, 3):
                raise ValueError("unsupported projection layout")
            return summarize(
                t.reshape(-1, t.size(-1)), groups, n_groups, projection=True
            )

        if not isinstance(module, torch.nn.Embedding):
            emit(name + ".in", features(x))
        emit(name + ".out", features(out))

    with module.register_forward_hook(hook, with_kwargs=True):
        yield


@contextmanager
def wan_features(block, emit, *, content_mask=None, first_chunk=0):
    """Observe the actual single-head Wan (encoder or decoder) spatial block.

    Each event has one record per actual time-plane. Quadrants use pixel
    centers, with ties assigned to the lower/right half. A bool content mask
    [B,T,H,W] if available is at this block's actual resolution (no inferred
    crop). When supplied, split each quadrant into content / reflected keys.
    """
    from .attention import attention_summary

    shape = None
    if first_chunk < 0:
        raise ValueError("invalid Wan chunk ordinal")
    chunk = first_chunk - 1

    @torch.no_grad()
    def begin(_, args):
        nonlocal shape, chunk
        shape = args[0].shape
        chunk += 1
        if len(shape) != 5:
            raise ValueError("Wan expects [B,C,T,H,W]")

    @torch.no_grad()
    def qkv(_, args, out):
        b, c, t, h, w = shape
        if out.shape != (b * t, 3 * c, h, w):
            raise ValueError("unexpected Wan qkv geometry")
        yy = (2 * torch.arange(h, device=out.device) + 1 >= h).long()
        xx = (2 * torch.arange(w, device=out.device) + 1 >= w).long()
        quad = (2 * yy[:, None] + xx[None, :]).flatten()
        mask = content_mask(chunk) if callable(content_mask) else content_mask
        if mask is not None and (
            mask.shape != (b, t, h, w) or mask.dtype != torch.bool
        ):
            raise ValueError("Wan content mask is not on the observed canvas")
        for i, plane in enumerate(out.detach()):
            g = (
                quad
                if mask is None
                else quad + 4 * (~mask[i // t, i % t]).to(out.device).flatten().long()
            )
            n = 4 if mask is None else 8
            q, k, v = plane.flatten(1).T.unsqueeze(1).chunk(3, -1)
            s = attention_summary(q, k, v, g, g, n)
            emit(
                "encoder.spatial",
                {
                    "chunk": chunk,
                    "batch": i // t,
                    "plane": i % t,
                    "hw": (h, w),
                    "content_mask_known": mask is not None,
                    "counts": torch.bincount(g, minlength=n).cpu(),
                    "mass": s["mass"].cpu(),
                    "entropy": s["entropy"].cpu(),
                },
            )

    @torch.no_grad()
    def mixed(_, args):
        b, c, t, h, w = shape
        v = args[0].detach().float()
        if v.shape != (b * t, c, h, w):
            raise ValueError("unexpected SDPA mixture geometry")
        emit(
            "encoder.spatial.value",
            {
                "chunk": chunk,
                "plane_rms": v.square().mean((1, 2, 3)).sqrt().reshape(b, t).cpu(),
            },
        )

    with ExitStack() as stack:
        stack.enter_context(block.register_forward_pre_hook(begin))
        stack.enter_context(block.to_qkv.register_forward_hook(qkv))
        stack.enter_context(block.proj.register_forward_pre_hook(mixed))
        yield


@contextmanager
def codec_features(codec, emit, *, encode=True, decode=False):
    """Observe Wan pixel input, pre-normalization mu, normalized latent, and
    inverse path when it actually runs. Bind the raw VAE, not its interface.
    No codec calls, cache edits, tensor logging or synthetic reconstruction.
    """
    if getattr(codec, "_aot_chunk_fns", None) or getattr(
        codec, "_compiled_decoder_forward", None
    ):
        raise ValueError("compiled codec requires its own verified tap")

    def planes(name, tensor):
        if tensor.ndim != 5:
            raise ValueError("Wan bridge expects [B,C,T,H,W]")
        x = tensor.detach().float()
        emit(
            name,
            {
                "shape": tuple(x.shape),
                "plane_rms": x.square().mean((1, 3, 4)).sqrt().cpu(),
            },
        )

    def wrap(orig, name):
        @wraps(orig)
        def wrapped(*args, **kwargs):
            # Do not change grad mode, return type or cache state of original.
            result = orig(*args, **kwargs)
            with torch.no_grad():
                x = args[0] if args else kwargs["x" if name == "encode" else "z"]
                planes("encoder.bridge." + name + ".in", x)
                planes("encoder.bridge." + name + ".out", result)
            return result

        return wrapped

    @torch.no_grad()
    def mu(_, args, out):
        if out.size(1) != 2 * codec.z_dim:
            raise ValueError("unexpected posterior head width")
        planes("encoder.bridge.mu", out[:, : codec.z_dim])

    with ExitStack() as stack:
        if encode:
            stack.enter_context(codec.conv1.register_forward_hook(mu))
            stack.enter_context(
                patch.object(codec, "encode", wrap(codec.encode, "encode"))
            )
        if decode:
            stack.enter_context(
                patch.object(codec, "decode", wrap(codec.decode, "decode"))
            )
        yield
