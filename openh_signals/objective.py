"""Read-only taps on the two-way recipe's natural objective and backward pass.

Install the flow tap on the *consumer* (omni_mot_model), before the public
history adapter installs its temporary action override. Capture the weight
returned by the actual scheduler; never call a scheduler a second time. These
APIs do not keep predictions, individual gradients, or module parameters.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from inspect import signature
import math
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


@dataclass(frozen=True)
class FlowSample:
    modality: str
    family: str
    mode: str
    sigma: object  # actual base sigma(s), not nominal timestep bins
    coefficient: float
    sample: int  # logical index within this microbatch, not a dataset index


def _numbers(value, n, field):
    item = torch.as_tensor(value).detach().flatten()
    if item.numel() not in (1, n) or not torch.isfinite(item).all():
        raise ValueError(f"invalid {field} schedule for {n} frames")
    return item.cpu().tolist()


@torch.no_grad()
def _flow_rows(args, outputs, weights, samples):
    predictions = args["pred"]
    valid = bool(args["has_valid_tokens"])
    if len(samples) != len(args["condition_mask"]) or (
        valid and len(samples) != len(predictions)
    ):
        raise ValueError("loss metadata must identify every logical item")
    if len(weights) != (len(samples) if valid else 0):
        raise ValueError("time weight capture does not match native calls")
    if args.get("time_axis", 0) != 0:
        raise ValueError("v1 covers action and vision only")
    excluded = bool(args.get("exclude_fully_conditioned_items", False))
    used_items = (
        sum(bool((c == 0).any()) for c in args["condition_mask"])
        if excluded
        else len(samples)
    )
    for i, (mask, meta) in enumerate(zip(args["condition_mask"], samples, strict=True)):
        if meta.modality not in ("action", "vision") or not math.isfinite(
            meta.coefficient
        ):
            raise ValueError("unsupported objective component")
        n = mask.size(0)
        if mask.numel() != n or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("v1 requires one binary conditioning flag per row/plane")
        mask = mask.reshape(n)
        t = args["timesteps"][i].flatten()[:n]
        row = dict(
            modality=meta.modality,
            family=meta.family,
            mode=meta.mode,
            sample=meta.sample,
            coefficient=meta.coefficient,
            dummy=not valid,
            timesteps=_numbers(t, n, "t"),
            sigma=_numbers(meta.sigma, n, "sigma"),
            clean=mask.bool().tolist(),
            mean_items=used_items,
            normalize_by_active=bool(args.get("normalize_by_active")),
        )
        if not valid:
            # Fully conditioned native dummy: no true prediction or time weight.
            # Do not use its tensor geometry to claim a target or target error.
            if not mask.bool().all():
                raise ValueError("no valid tokens but logical item has noisy positions")
            yield dict(
                row,
                valid_count=0,
                per_row_valid=[0] * n,
                squared_error_sum=None,
                training_denominator=None,
                active_mse=None,
                objective_contribution=0.0,
                time_weights=None,
            )
            continue
        p = predictions[i]
        if p.shape != args["target"][i].shape:
            raise ValueError("prediction and target shape differ")
        widths = args.get("raw_action_dim")
        slots = args.get("action_valid_mask")
        if meta.modality == "action":
            if p.ndim != 2 or p.size(0) != n:
                raise ValueError("expected T,D action")
            d = p.size(1) if widths is None or widths[i] is None else int(widths[i])
            if not 0 < d <= p.size(1):
                raise ValueError("invalid real action width")
            v = None if slots is None or slots[i] is None else slots[i].flatten()[:d]
            if v is not None and (len(v) != d or not ((v == 0) | (v == 1)).all()):
                raise ValueError("invalid channel validity")
            err = (p[:, :d] - args["target"][i][:, :d]).square()
            if v is not None:
                v = v.to(err)
                err = err * v
            row_size = d if v is None else int(v.sum())
            row["raw_width"] = d
        else:
            if p.ndim != 4 or p.size(1) != n or widths is not None:
                raise ValueError("expected C,T,H,W vision")
            err = (p - args["target"][i]).square()
            row_size = err.numel() // n
            v = None
        active = ~mask.bool()
        rows = err if meta.modality == "action" else err.transpose(0, 1).flatten(1)
        total = float(rows.sum(1, dtype=torch.float32).masked_fill(~active, 0).sum())
        count = int(active.sum()) * row_size
        all_count = n * row_size
        denom = max(1, count if row["normalize_by_active"] else all_count)
        # Preserve native dtypes, broadcasting and reduction order for the
        # weighted objective. A train-time FP16 sum need not equal an FP32 sum.
        noisy = 1.0 - args["condition_mask"][i]
        tw = weights[i].reshape(-1, *([1] * (noisy.ndim - 1)))
        weighted_err = err * tw * noisy
        if row["normalize_by_active"] or v is not None:
            numerator_rows = (
                noisy.sum() if row["normalize_by_active"] else noisy.numel()
            )
            channels = v.sum() if v is not None else row_size
            effective_den = (numerator_rows * channels).clamp(min=1)
            weighted = float(weighted_err.sum() / effective_den)
        else:
            weighted = float(weighted_err.mean())
        counts = (active.to(torch.int64) * row_size).tolist()
        yield dict(
            row,
            valid_count=count,
            all_count=all_count,
            training_denominator=denom,
            per_row_valid=counts,
            squared_error_sum=total,
            active_mse=total / count if count else None,
            time_weights=_numbers(tw, n, "weight"),
            native_item_loss=float(outputs[1][i]),
            weighted_item_loss=weighted,
            objective_contribution=meta.coefficient * weighted / max(1, used_items),
        )


@contextmanager
def observe_flow(omni_module, time_weight_cls, describe, emit):
    """describe(bound args) -> list[FlowSample] or None for an unsampled call.

    ``emit(signal, row)`` consumes summaries synchronously (e.g. compact rows
    in the step envelope). Neither the forward nor its outputs are replaced.
    """
    receiver = ContextVar("time_weights", default=None)
    original = omni_module.compute_flow_matching_loss
    sig = signature(original)
    weight_call = time_weight_cls.__call__

    @wraps(weight_call)
    def weight(self, *a, **kw):
        out = weight_call(self, *a, **kw)
        buf = receiver.get()
        if buf is not None:
            buf.append(out.detach().clone())
        return out

    @wraps(original)
    def loss(*a, **kw):
        bound = sig.bind(*a, **kw)
        bound.apply_defaults()
        samples = describe(bound.arguments)
        if samples is None:
            return original(*a, **kw)
        weights = []
        token = receiver.set(weights)
        try:
            out = original(*a, **kw)
        finally:
            receiver.reset(token)
        for row in _flow_rows(bound.arguments, out, weights, samples):
            emit("flow.parts", row)
        return out

    with (
        patch.object(time_weight_cls, "__call__", weight),
        patch.object(omni_module, "compute_flow_matching_loss", loss),
    ):
        yield


def _scale(scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid AMP scale")
    return float(scale)


def _square_sum(x, scale):
    # vector_norm reads the local shard; scalar arithmetic does not touch .grad.
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    return (float(torch.linalg.vector_norm(x.detach(), dtype=dtype)) / scale) ** 2


@torch.no_grad()
def module_gradients(model, amp_scale, emit):
    """Emit disjoint per-module *local* sums for reduction across all ranks.

    Call at entry to the callback group's pre-optimizer hook, before clipping
    and before any callback-side unscale. A full-mesh DTensor contributes each
    local shard once (coordinate 0 on replica axes); unsharded DDP only rank 0.
    Unknown subgroup/partial layouts cannot be turned into logical norms here.
    """
    scale = _scale(amp_scale)
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if world > 1 else 0
    ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    buckets = {}
    for path, p in model.named_parameters():
        name = path.rpartition(".")[0] or "<root>"
        # Root embeddings may be parameters, not nn.Module leaves.
        if path.rsplit(".", 1)[-1] not in ("weight", "bias"):
            name = path
        stat = buckets.setdefault(
            name, dict(square_sum=0.0, grad_numel=0, frozen_numel=0, missing_numel=0)
        )
        g = p.grad
        if isinstance(p, DTensor):
            if g is not None and (
                not isinstance(g, DTensor) or p.device_mesh != g.device_mesh
            ):
                raise ValueError("unverified gradient placement")
            layout = g if g is not None else p
            mesh = layout.device_mesh
            coord = mesh.get_coordinate()
            if mesh.mesh.numel() != world or coord is None:
                raise ValueError(
                    "cannot deduplicate a submesh without global ownership metadata"
                )
            owner = True
            for i, placement in enumerate(layout.placements):
                if placement.is_replicate():
                    owner &= coord[i] == 0
                elif not placement.is_shard():
                    raise ValueError("partial/unreduced gradient at pre-clip boundary")
            if not owner:
                continue
            if g is not None:
                g = g.to_local()
            size = (p.to_local() if g is None else g).numel()
        else:
            if world != 1 and not ddp:
                raise ValueError("plain tensor with unverified cross-rank ownership")
            if rank != 0:
                continue
            size = p.numel()
        if not p.requires_grad:
            stat["frozen_numel"] += size
        elif g is None:
            stat["missing_numel"] += size
        else:
            stat["grad_numel"] += size
            stat["square_sum"] += _square_sum(g, scale)
    if len(buckets) > 4096:
        raise ValueError("v1 gradient bucket reservation exceeded")
    for name, row in buckets.items():
        emit("grad.module", dict(name=name, amp_scale=scale, **row))


def logical_module_gradients(model, amp_scale, emit):
    """Collective on EVERY world rank at the same sampled pre-clip callback.

    An identical sorted parameter-owner schema on every rank is required; do
    not use pipeline partitions or call this only on the writer rank. Emits
    globally reduced logical norms once, on global rank zero.
    """
    rows = []
    module_gradients(model, amp_scale, lambda _, row: rows.append(row))
    rows.sort(key=lambda row: row["name"])
    fields = ("square_sum", "grad_numel", "frozen_numel", "missing_numel")
    multi = dist.is_initialized() and dist.get_world_size() > 1
    if multi:
        # Check identities, not just the number of buckets, before reducing.
        # Numeric hash exchange works with NCCL, unlike a CPU tensor collective.
        import hashlib

        device = (
            next(model.parameters()).device
            if dist.get_backend() == "nccl"
            else torch.device("cpu")
        )
        digest = hashlib.sha256(
            b"\0".join(row["name"].encode() for row in rows)
        ).digest()
        schema = torch.tensor(list(digest), device=device, dtype=torch.uint8)
        peers = [torch.empty_like(schema) for _ in range(dist.get_world_size())]
        dist.all_gather(peers, schema)
        if any(not torch.equal(p, schema) for p in peers):
            raise ValueError("gradient ownership buckets differ across ranks")
        # FP64 is exact for integer parameter counts up to 2**53.
        totals = torch.tensor(
            [[r[f] for f in fields] for r in rows], dtype=torch.float64, device=device
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if dist.get_rank() != 0:
            return
        for row, values in zip(rows, totals.tolist(), strict=True):
            row.update(zip(fields, [values[0], *map(int, values[1:])], strict=True))
    for row in rows:
        emit(
            "grad.module",
            dict(row, l2=math.sqrt(row["square_sum"]) if row["grad_numel"] else None),
        )


@contextmanager
def observe_preclip(callback_group, enabled, emit):
    original = callback_group.on_before_optimizer_step

    @wraps(original)
    def before(model, optimizer, scheduler, grad_scaler, iteration):
        if enabled(iteration + 1):
            logical_module_gradients(model, grad_scaler.get_scale(), emit)
        return original(model, optimizer, scheduler, grad_scaler, iteration=iteration)

    with patch.object(callback_group, "on_before_optimizer_step", before):
        yield


def watch_group(tensor, groups, name, amp_scale, emit, *, accumulation=1):
    """Observe grads w.r.t. real row outputs; -1 means packing/alignment pad.

    Caller owns the (microstep, boundary) identity and chooses exactly one
    executed tap for that identity. No gradients are requested on inference
    outputs. The square sum is unscaled for AMP, not multiplied by accumulation.
    """
    scale = _scale(amp_scale)
    if not tensor.requires_grad:
        return None
    if isinstance(tensor, DTensor):
        raise ValueError("group IDs must describe owned local activation rows")
    labels = torch.as_tensor(groups, device=tensor.device).detach().clone().reshape(-1)
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64) or len(
        labels
    ) != math.prod(tensor.shape[:-1]):
        raise ValueError("group IDs do not match activation rows")
    present = sorted(set(labels.tolist()) - {-1})
    if any(g < 0 or g >= 12 for g in present):
        raise ValueError("invalid v1 token group")

    @torch.no_grad()
    def measure(grad):
        flat = grad.detach().reshape(len(labels), -1)
        for group in present:
            values = flat[labels == group]
            emit(
                "grad.group",
                dict(
                    name=name,
                    group=group,
                    rows=len(values),
                    count=values.numel(),
                    square_sum=_square_sum(values, scale),
                    amp_scale=scale,
                    accumulation=accumulation,
                ),
            )
        return None

    return tensor.register_hook(measure)
