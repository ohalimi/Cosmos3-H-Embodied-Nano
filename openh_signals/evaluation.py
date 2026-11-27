"""Read sites for the public request-local CFG cache and the final scorer."""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from hashlib import sha256
import weakref

import torch

from .attention import attention_summary
from .runtime_pack import GROUPS, TEXT


def _layout(groups, records):
    layout = []
    for rec in records:
        start = rec["offset"]
        stop = start + rec["length"]
        g = groups[start:stop]
        text = torch.isin(g, torch.tensor(sorted(TEXT), device=g.device))
        nt = int(text.sum())
        if not (0 < nt < len(g) and text[:nt].all() and not text[nt:].any()):
            raise ValueError("cached labels need contiguous actual UND/GEN splits")
        layout.append((g, nt))
    return layout


def _branches(cell, count):
    branches = cell.get("branches")
    if branches is None and count == 1:
        branches = (cell.get("branch"),)
    if (
        branches is None
        or len(branches) != count
        or not all(isinstance(s, str) and s for s in branches)
    ):
        raise ValueError("each executed sample needs its actual CFG branch binding")
    return tuple(branches)


@torch.no_grad()
def _prefix(bound, layout, layer, selected):
    """Measure only the sentinel head's causal submatrix *while UND queries execute*."""
    head = {0: 0, 35: 31}.get(layer)
    if head is None or not selected:
        return {}
    from cosmos_framework.data.generator.sequence_packing.runtime import get_und_seq

    q = get_und_seq(bound["packed_query_states"])
    k = get_und_seq(bound["packed_key_states"])
    offset = 0
    matrices = {}
    for i, (g, nt) in enumerate(layout):
        if i in selected:
            if len(g) > 4096 or head >= q.shape[1] or q.shape[1] % k.shape[1]:
                raise ValueError("requested sentinel prefix cannot be represented")
            qh = q[offset : offset + nt, head : head + 1]
            kvhead = head // (q.shape[1] // k.shape[1])
            kh = k[offset : offset + nt, kvhead : kvhead + 1]
            # V does not participate in probability measurement; pass only this head's real values.
            vh = get_und_seq(bound["packed_value_states"])[
                offset : offset + nt, kvhead : kvhead + 1
            ]
            matrix = torch.empty(nt, nt, device="cpu", dtype=torch.float16)

            def save(a, b, p, _):
                matrix[a : a + p.size(1), b : b + p.size(2)] = (
                    p[0].to(torch.float16).cpu()
                )

            attention_summary(
                qh, kh, vh, g[:nt], g[:nt], len(GROUPS), causal=True, visit=save
            )
            matrices[i] = matrix
        offset += nt
    return matrices


@torch.no_grad()
def cached_summary(
    bound, layout, layer, *, sentinel=(), prefixes=None, calibration=False
):
    """At the actual dense/varlen attention call, after concat/index_select of cached keys.

    Each entry reports only the real GEN queries that executed; missing text query
    groups have zero count and NaN summaries. A full matrix additionally needs a
    measured causal prefix tied to the SAME live cache and CFG branch at fill.
    """
    q, k, v = (bound[n] for n in ("query", "key", "value"))
    if any(x.ndim != 4 or x.shape[0] != 1 for x in (q, k, v)) or bound.get("is_causal"):
        raise ValueError("unsupported cached attention layout")
    q, k, v = q[0], k[0], v[0]
    qends = [0]
    kends = [0]
    for g, nt in layout:
        qends.append(qends[-1] + len(g) - nt)
        kends.append(kends[-1] + len(g))
    if len(k) != kends[-1] or len(v) != len(k) or len(q) < qends[-1]:
        raise ValueError("actual cache length and tagged physical inputs disagree")
    if len(layout) > 1:
        if len(q) != qends[-1] or any(
            bound.get(key) is None or bound[key].tolist() != expected
            for key, expected in [
                ("cumulative_seqlen_Q", qends),
                ("cumulative_seqlen_KV", kends),
            ]
        ):
            raise ValueError(
                "cached kernel offsets disagree with actual per-sample labels"
            )
    elif any(
        bound.get(key) is not None
        for key in ("cumulative_seqlen_Q", "cumulative_seqlen_KV")
    ):
        raise ValueError("unexpected varlen single-sample cache call")
    out = []
    for i, (g, nt) in enumerate(layout):
        qq = q[qends[i] : qends[i + 1]]
        kk = k[kends[i] : kends[i + 1]]
        vv = v[kends[i] : kends[i + 1]]
        head = {0: 0, 35: 31}.get(layer) if i in sentinel else None
        matrix = None
        if head is not None:
            prefix = (prefixes or {}).get(i)
            if (
                len(g) > 4096
                or prefix is None
                or prefix.shape != (nt, nt)
                or head >= qq.shape[1]
            ):
                raise ValueError(
                    "unmeasured cache-fill prefix or oversized map: cannot emit full map"
                )
            matrix = torch.zeros(len(g), len(g), device="cpu", dtype=torch.float16)
            matrix[:nt, :nt] = prefix
        squares = qq.new_zeros((len(GROUPS), len(GROUPS)), dtype=torch.float32)
        parts = None

        def visit(a, b, p, values):
            nonlocal parts
            h, r, c = p.shape
            if b == 0:
                parts = p.new_zeros((len(GROUPS), h, r, values.size(-1)))
            keys = g[b : b + c]
            for group in keys.unique().tolist():
                mask = keys == group
                parts[group] += p[:, :, mask] @ values[:, mask]
            if b + c == len(g):
                queries = torch.nn.functional.one_hot(
                    g[nt + a : nt + a + r], len(GROUPS)
                ).float()
                squares.add_(queries.T @ parts.square().sum((1, 3)).T)
                parts = None
            if matrix is not None:
                matrix[nt + a : nt + a + r, b : b + c] = p[head].to(torch.float16).cpu()

        row = attention_summary(
            qq,
            kk,
            vv,
            g[nt:],
            g,
            len(GROUPS),
            scale=bound.get("scale"),
            visit=visit,
            calibration=calibration,
        )
        den = row["counts"].float() * qq.size(1) * vv.size(-1)
        row.update(
            read_strength=(
                squares / den.masked_fill(den == 0, torch.nan)[:, None]
            ).sqrt(),
            cache_hit=True,
            text_queries_executed=False,
            padded_queries_excluded=len(q) - qends[-1],
        )
        if matrix is not None:
            row.update(full_map=matrix, text_matrix_source="same_cache_fill")
        out.append(row)
    return out


@contextmanager
def cached_attention(runtime, owner):
    """Scoped hooks on the pinned cache module, not a change to its backend/dispatch.

    Attest branch, prompt and K/V object identity at the actual cache write. The
    optional two small fill matrices live only as long as their native K cache.
    """
    from .runtime import wrap, scoped

    active_value = ContextVar("cache_value", default=None)
    cache = {}
    pending = {}

    def on_fill(original, args, kwargs, bound):
        active = runtime.groups.get()
        cell = runtime.current.get()
        value = bound.get("memory_value")
        if (
            active is not None
            and cell is not None
            and type(value) is owner.InferenceTextKVMemoryValue
            and value.frame_idx == 0
        ):
            layer = runtime.layer.get()
            groups, _, records = active
            selected = tuple(cell.get("cache_sentinels", ()))
            if (
                len(set(selected)) != len(selected)
                or len(selected) > 2
                or not set(selected) <= set(range(len(records)))
            ):
                raise ValueError("invalid cache sentinel selection")
            pending[layer] = (
                _branches(cell, len(records)),
                tuple(r["text_digest"] for r in records),
                _prefix(bound, _layout(groups, records), layer, selected),
            )
        return original(*args, **kwargs)

    def on_write(original, args, kwargs, bound):
        i = bound["layer_idx"]
        result = original(*args, **kwargs)
        if i in pending and runtime.groups.get() is not None:
            k, v = bound["self"]._text_kv_cache[i].get()
            ident = id(k)

            def forget(ref):
                if ident in cache and cache[ident][0] is ref:
                    cache.pop(ident)

            cache[ident] = (weakref.ref(k, forget), weakref.ref(v), *pending.pop(i))
        return result

    def on_hit(original, args, kwargs, bound):
        with scoped(active_value, bound["memory_value"]):
            return original(*args, **kwargs)

    def on_kernel(original, args, kwargs, bound):
        cell = runtime.current.get()
        active = runtime.groups.get()
        if cell is None or active is None:
            return original(*args, **kwargs)
        value = active_value.get()
        groups, seen, records = active
        layer = runtime.layer.get()
        if value is None or type(layer) is not int or layer in seen:
            raise ValueError("unknown or repeated cached layer")
        k, v = value.und_k_cached, value.und_v_cached
        entry = cache.get(id(k))
        branches = _branches(cell, len(records))
        if (
            entry is None
            or entry[0]() is not k
            or entry[1]() is not v
            or entry[2] != branches
            or entry[3] != tuple(r["text_digest"] for r in records)
        ):
            raise ValueError("unattested cache fill, branch or prompt mismatch")
        solver = runtime.solver.get()
        if solver is None or not solver[2]:
            raise ValueError(
                "cached observation outside an actual scheduled solver call"
            )
        selected = cell.get("full_map", False)
        if selected is True and len(records) != 1:
            raise ValueError("multi-sample full map needs explicit sentinel index")
        sentinel = (0,) if selected is True else selected or ()
        if (
            len(sentinel) > 2
            or len(set(sentinel)) != len(sentinel)
            or not set(sentinel) <= set(range(len(records)))
        ):
            raise ValueError("invalid full map sample index")
        rows = cached_summary(
            bound, _layout(groups, records), layer, sentinel=sentinel, prefixes=entry[4]
        )
        seen.add(layer)
        for i, row in enumerate(rows):
            runtime.emit(
                "mot",
                dict(
                    **row,
                    layer=layer,
                    sample=i,
                    cell=dict(
                        cell,
                        branch=branches[i],
                        solver_step=solver[0],
                        solver_count=solver[1],
                    ),
                ),
            )
        return original(*args, **kwargs)

    with ExitStack() as stack:
        # Keep owner.dispatch_attention identical to every unwrapped layer's default:
        # the native cache installer checks that identity before binding the memory dispatcher.
        for obj, name, hook in (
            (owner, "dispatch_attention_with_text_kv_memory", on_fill),
            (owner.InferenceTextKVMemoryState, "write_for_layer", on_write),
            (owner, "_attention_gen_with_cached_text", on_hit),
            (owner, "attention", on_kernel),
        ):
            stack.enter_context(wrap(obj, name, hook))
        yield


class ClipStrata:
    """Equal-clip accumulation over predeclared full cell keys. NaN means structural absence.

    Include mode/seed/condition/chunk/solver/branch/layer in key. Supply the completed
    clip's summaries once per key; request duplicate detection via its private digest.
    """

    def __init__(self, allowed, *, max_instances):
        self.allowed = frozenset(allowed)
        self.limit = max_instances
        self.bins = {}
        self.seen = {}

    @torch.no_grad()
    def add(self, key, clip_digest, **fields):
        if key not in self.allowed or not fields:
            raise ValueError("unbound stratum or empty clip")
        members = self.seen.setdefault(key, set())
        if clip_digest in members or len(members) >= self.limit:
            raise ValueError("replayed clip or too many instances")
        values = {
            n: t.detach().to(device="cpu", dtype=torch.float64)
            for n, t in fields.items()
        }
        if any(torch.isinf(v).any() for v in values.values()):
            raise ValueError("nonfinite observed clip summary")
        bucket = self.bins.setdefault(
            key,
            {
                n: (torch.zeros_like(v), torch.zeros_like(v, dtype=torch.int32))
                for n, v in values.items()
            },
        )
        if bucket.keys() != values.keys() or any(
            bucket[n][0].shape != v.shape for n, v in values.items()
        ):
            raise ValueError("incompatible group/head schema across clips")
        for n, v in values.items():
            keep = v.isfinite()
            bucket[n][0].add_(v.masked_fill(~keep, 0))
            bucket[n][1].add_(keep)
        members.add(clip_digest)

    def mean(self, key):
        return {
            n: (s / c.masked_fill(c == 0, 1)).masked_fill(c == 0, torch.nan).float()
            for n, (s, c) in self.bins[key].items()
        }

    def counts(self, key):
        return {n: c.clone() for n, (_, c) in self.bins[key].items()}


@torch.no_grad()
def scorer_record(bound, result):
    """Observe the scorer RETURN, not a fresh rollout, decoder invocation or media read."""
    row = {"mode": result["mode"]}
    if "action_mse" in result:
        gt, mask = bound["gt_action_td"], bound["action_prediction_mask"]
        if (
            gt is None
            or gt.ndim != 2
            or mask is None
            or mask.dtype != torch.bool
            or mask.shape != gt.shape[:1]
        ):
            raise ValueError("cannot attest future-only action scoring")
        if not mask.any() or not bool((mask.int().diff() >= 0).all()):
            raise ValueError("expected future-only suffix of raw valid channels")
        target = gt[mask.to(gt.device)].float().contiguous().cpu()
        row.update(
            action_mse=float(result["action_mse"]),
            valid_elements=target.numel(),
            future_rows=len(target),
            raw_width=target.size(-1),
            excluded_rows=len(mask) - len(target),
            target_digest=sha256(target.numpy().tobytes()).hexdigest(),
            action_scope="future_rows_valid_channels",
        )
    if "psnr" in result:
        row.update(
            psnr=float(result["psnr"]),
            video_scope="native_overlap_crop_all_scored_frames",
        )
    return row


@contextmanager
def eval_score_hook(eval_module, emit):
    """Wrap the scorer *binding used by the evaluator*, not the defining module's alias."""
    from .runtime import wrap

    def capture(original, args, kwargs, bound):
        result = original(*args, **kwargs)
        emit(
            "eval.error",
            dict(sample_name=bound["name"], **scorer_record(bound, result)),
        )
        return result

    with wrap(eval_module, "compute_sample_metrics", capture):
        yield
