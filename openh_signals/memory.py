"""Observe already executed CAMP lookups, producer assignments and paired scores.

No dataset, track, codebook or checkpoint is opened here. Producer hooks are a
separate scope: the policy never constructs a writer or guesses its labels.
"""

from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
from pathlib import Path
import sys

import torch
from torch.nn import functional as F


# Exact source blobs with verified public function-local contracts.
TRACK = "5500e10919d2ff3cf2b29a287775da5c65f3d213"
TRIPLET = "8a6535a5d84ab1ec67b974d9c624b0cc215ddad9"
PRODUCER = "c6ed816f3dc6146d993a3a5181b3eb0a5536ef0e"


def _source(fn, blob):
    raw = Path(inspect.getsourcefile(fn)).read_bytes()
    actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if actual != blob:
        raise ValueError("CAMP hook source differs from the approved binding")


@torch.no_grad()
def row_summary(rows):
    if rows.shape != (3, 20) or not torch.isfinite(rows).all() or rows.abs().max() > 1:
        raise ValueError("expected three finite bounded projected rows of width twenty")
    x = rows.float()
    return dict(
        norm=x.norm(dim=1),
        mean=x.mean(1),
        max_abs=x.abs().amax(1),
        zero_fraction=x.eq(0).float().mean(1),
        saturation_fraction=(x.abs() >= 0.99).float().mean(1),
        saturation_threshold=0.99,
        code_use="not present in policy input",
    )


@torch.no_grad()
def memory_prefix(raw, projected, history):
    """One policy payload only, at the actual projection; do not pool an entire pack."""
    if (
        raw.ndim != 2
        or projected.ndim != 2
        or raw.size(1) < 20
        or not 0 <= history <= 16
        or len(raw) != len(projected)
        or len(raw) != 3 + history + 24
    ):
        raise ValueError("unbound memory/history/target boundary")
    prefix = 3 + history
    record = row_summary(raw[:3, :20])
    record.update(
        history=history,
        clean_prefix=prefix,
        future_targets=24,
        raw_prefix_norm=raw[:prefix, :20].float().norm(dim=-1),
        projected_prefix_norm=projected[:prefix].float().norm(dim=-1),
        history_lags=tuple(range(1 - history, 1)),
    )
    return record


@torch.no_grad()
def memory_reads(summary):
    """Derived from the ordinary all-key denominator, before FrozenText compaction."""
    counts, mass, strength = (summary[k] for k in ("counts", "mass", "read_strength"))
    if len(counts) != 12 or not (counts[8:11] == 1).all() or counts[7] != 24:
        raise ValueError("memory groups must identify the three observed physical rows")
    latest = mass[:, 7, 6:7]
    return dict(
        future_to_memory=mass[:, 7, 8:11],
        memory_to_future=mass[:, 8:11, 7],
        relative_to_latest=(
            mass[:, 7, 8:11] / latest.masked_fill(latest == 0, torch.nan)
        )
        if counts[6] == 1
        else None,
        future_read_rms=strength[7, 8:11],
        latest_valid=int(counts[6]),
    )


@contextmanager
def track_reads(cls, receive):
    """Register in the process that actually performs the lookup (possibly a worker).

    `receive` joins by (split, root, actual episode, exact frame) plus writer ID,
    never by the ordinal of the requested window. An unsuccessful lookup emits nothing.
    """
    original = cls.at
    _source(original, TRACK)

    @wraps(original)
    def at(self, root, episode, frame):
        rows = original(self, root, episode, frame)
        key = (self.folder.name, Path(root).name, int(episode), int(frame))
        receive(key, dict(writer=self.folder.parent.name, **row_summary(rows)))
        return rows

    cls.at = at
    try:
        yield
    finally:
        cls.at = original


@torch.no_grad()
def code_use(labels, valid):
    if (
        labels.dtype != torch.long
        or labels.shape != valid.shape
        or valid.dtype != torch.bool
    ):
        raise ValueError(
            "one actual producer assignment and a validity flag per anchor required"
        )
    ids = labels[valid]
    if (ids < 0).any() or (ids >= 512).any():
        raise ValueError("invalid dictionary winner")
    hist = ids.bincount(minlength=512)
    p = hist[hist > 0].double() / max(1, len(ids))
    h = -(p * p.log()).sum()
    return dict(
        histogram=hist.to(torch.uint32),
        possible=512,
        distinct=len(p),
        valid=len(ids),
        top_share=p.max().item() if len(p) else None,
        entropy=h.item() if len(p) else None,
        perplexity=h.exp().item() if len(p) else None,
    )


@contextmanager
def _local_return(fn, blob, capture):
    """PEP 669 local PY_RETURN event; no global tracer or change to the called function.

    Only used in eager producer/scorer processes. The monitored source is checked
    before relying on its locals. Do not disable another profiler to acquire an ID.
    """
    _source(fn, blob)
    mon = sys.monitoring
    tool = next((i for i in (3, 4) if mon.get_tool(i) is None), None)
    if tool is None:
        raise RuntimeError("no free local monitoring slot")
    code = fn.__code__
    mon.use_tool_id(tool, "openh-signals")

    def on_return(current_code, offset, value):
        frame = sys._getframe(1)
        if current_code is not code or frame.f_code is not code:
            raise ValueError("unexpected frame at monitored return")
        capture(frame.f_locals, value)

    try:
        mon.register_callback(tool, mon.events.PY_RETURN, on_return)
        mon.set_local_events(tool, code, mon.events.PY_RETURN)
        yield
    finally:
        mon.set_local_events(tool, code, 0)
        mon.register_callback(tool, mon.events.PY_RETURN, None)
        mon.free_tool_id(tool)


def public_assignments(dictionary, selected, receive):
    """Observe actual pre-EMA labels, not nearest neighbors recomputed after return.

    selected(labels_shape) supplies the authorized anchor mask and private binding;
    if None, nothing is retained. Register only at authorized producer observation.
    """

    def capture(local, value):
        if local["self"] is not dictionary:
            return
        valid = local["valid"]
        cell = selected(valid.shape)
        if cell is None:
            return
        binding, mask = cell
        if mask.shape != valid.shape or mask.dtype != torch.bool:
            raise ValueError("writer selection must bind the actual anchor layout")
        receive(binding, code_use(local["labels"].reshape_as(valid), valid & mask))

    return _local_return(type(dictionary).forward, PRODUCER, capture)


def northstar_assignments(vq, selected, receive):
    """Observe returned IDs[1] on the other producer; no encoder invocation."""
    raw = Path(inspect.getsourcefile(type(vq).forward)).read_bytes()
    if (
        hashlib.sha256(raw).hexdigest()
        != "e1e829dc0054be69e90c597ee9310fdbbae772bb898aeffe8bfbc9219533d0b0"
    ):
        raise ValueError("unverified returned-ID producer source")

    def hook(module, args, kwargs, result):
        b = inspect.signature(module.forward).bind(*args, **kwargs)
        b.apply_defaults()
        ids = result[1]
        valid = b.arguments["mask"]
        cell = selected(ids.shape)
        if cell is not None:
            binding, mask = cell
            if mask.shape != ids.shape or mask.dtype != torch.bool:
                raise ValueError("writer selection must bind the actual anchor layout")
            receive(binding, code_use(ids, mask if valid is None else valid & mask))

    return vq.register_forward_hook(hook, with_kwargs=True)


@torch.no_grad()
def writer_gates(lstm, inputs, initial, actual, selected):
    """Validated observer-side LSTM recurrence on an existing deterministic writer call.

    No gate is reported unless *all* final states and the entire top-layer
    sequence agree with the actual fused outputs. Supports the FP32 producers.
    """
    if (
        not lstm.batch_first
        or lstm.bidirectional
        or lstm.proj_size
        or lstm.dropout
        or inputs.dtype != torch.float32
        or inputs.ndim != 3
        or selected.shape != inputs.shape[:2]
        or selected.dtype != torch.bool
        or not selected.any()
    ):
        raise ValueError("unsupported LSTM diagnostic configuration")
    seq = inputs.detach()
    target, (ht, ct) = actual
    summaries = []
    with torch.autocast(device_type=seq.device.type, enabled=False):
        for i in range(lstm.num_layers):
            if initial is None:
                h = seq.new_zeros((len(seq), lstm.hidden_size))
                c = torch.zeros_like(h)
            else:
                h, c = initial[0][i].detach(), initial[1][i].detach()
            projected = F.linear(
                seq, getattr(lstm, f"weight_ih_l{i}"), getattr(lstm, f"bias_ih_l{i}")
            )
            gates = seq.new_zeros(4)
            states = []
            for t in range(seq.shape[1]):
                ai, af, ag, ao = (
                    projected[:, t]
                    + F.linear(
                        h,
                        getattr(lstm, f"weight_hh_l{i}"),
                        getattr(lstm, f"bias_hh_l{i}"),
                    )
                ).chunk(4, -1)
                ing, forget, candidate, out = (
                    ai.sigmoid(),
                    af.sigmoid(),
                    ag.tanh(),
                    ao.sigmoid(),
                )
                c = forget * c + ing * candidate
                h = out * c.tanh()
                states.append(h)
                gates += torch.stack(
                    [x[selected[:, t]].sum() for x in (ing, forget, candidate, out)]
                )
            seq = torch.stack(states, 1)
            torch.testing.assert_close(h, ht[i], atol=2e-5, rtol=1e-5)
            torch.testing.assert_close(c, ct[i], atol=2e-5, rtol=1e-5)
            summaries.append(gates / (selected.sum() * lstm.hidden_size))
        torch.testing.assert_close(seq, target, atol=2e-5, rtol=1e-5)
    return dict(
        gate_order=("input", "forget", "candidate", "output"),
        mean=torch.stack(summaries),
        valid=int(selected.sum()),
        hidden_rms=seq[selected].square().mean().sqrt(),
    )


@torch.no_grad()
def writer_spatial(logits, xy, selected):
    """One producer-side anchor; preserve the 32 x 9 softmax, not any RGB."""
    if (
        logits.shape[1:] != (32, 3, 3)
        or xy.shape != (9, 2)
        or not 0 <= selected < len(logits)
    ):
        raise ValueError("unknown producer camera or spatial layout")
    p = logits[selected].float().flatten(1).softmax(-1)
    return dict(
        probabilities=p.half(),
        entropy=-torch.special.xlogy(p, p).sum(-1),
        coordinate_norm=(p @ xy.float()).norm(dim=-1),
    )


def intervention_scores(fn, bind, receive):
    """Observe a completed, separately authorized scorer triplet. Never call generate.

    bind(batch, seed) supplies (actual private window key, actual donor window key)
    from the already-bound evaluator, both (root, episode, frame).
    """

    @torch.no_grad()
    def capture(local, value):
        source, donor = bind(local["batch"], local["seed"])
        if source[:2] == donor[:2] or len(source) != 3 or len(donor) != 3:
            raise ValueError("intervention needs a physically distinct donor")
        mask = local["mask"].cpu()
        gt = local["gt"].detach().cpu()[mask]
        preds, mse = local["preds"], value["mse"]
        if gt.shape != (24, 20) or not mask[-24:].all() or mask[:-24].any():
            raise ValueError("unexpected future-only scorer layout")
        digest = hashlib.sha256(
            gt.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        row = dict(
            source=source,
            donor=donor,
            target_digest=digest,
            seed=int(local["seed"]),
            donor_code_equal=not value["donor_code_differs"],
            mse=mse.copy(),
            target_elements=480,
        )
        for condition in ("zero", "shuffled"):
            row[condition + "_delta_mse"] = mse[condition] - mse["real"]
            row[condition + "_changed_fraction"] = (
                (preds["real"] != preds[condition]).float().mean().item()
            )
        receive(row)

    return _local_return(fn, TRIPLET, capture)
