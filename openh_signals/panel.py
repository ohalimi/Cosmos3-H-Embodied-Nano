"""Plan the fixed panel for an isolated diagnostic worker; never drive the live model.

The data owner supplies digests of *physically bound* held-out windows. This is
neither a sampler nor a source of training batches. A missing cell is a hold.
"""

from dataclasses import dataclass
import hashlib
import re

import torch

from .events import Event, PLAN_SHA256

SIGMA = {"low": 0.15, "mid": 0.50, "high": 0.85}
MODES = frozenset(("forward_dynamics", "inverse_dynamics", "policy"))


@dataclass(frozen=True)
class Window:
    identity: str  # digest of frozen physical root/index/episode/frame/view binding
    target: str  # mode-independent target digest
    modes: tuple
    sentinel: bool


@dataclass(frozen=True)
class Cell:
    window: Window
    mode: str
    level: str
    full_map: bool

    @property
    def sigma(self):
        return SIGMA[self.level]

    def generator(self, modality, device):
        """Same base noise across weights/modes/levels, independent across modalities."""
        if modality not in ("action", "vision"):
            raise ValueError("unknown observer noise modality")
        key = f"v1|{self.window.identity}|{modality}|20260908".encode()
        return torch.Generator(device=device).manual_seed(
            int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        )


class ProbePanel:
    def __init__(self, windows):
        w = self.windows = tuple(windows)
        if (
            not 1 <= len(w) <= 13
            or len({p.identity for p in w}) != len(w)
            or sum(p.sentinel for p in w) > 2
        ):
            raise ValueError("duplicate/unbounded fixed probe selection")
        for p in w:
            if not (
                re.fullmatch("[0-9a-f]{64}", p.identity)
                and re.fullmatch("[0-9a-f]{64}", p.target)
                and set(p.modes) <= MODES
                and len(set(p.modes)) == len(p.modes)
                and p.modes
            ):
                raise ValueError("unbound probe window/eligibility")
            if p.sentinel and "policy" not in p.modes:
                raise ValueError(
                    "training sentinel needs the approved conditional policy cell"
                )

    def select(self, event: Event):
        return tuple(
            Cell(
                w,
                mode,
                level,
                event.deep and w.sentinel and mode == "policy" and level == "mid",
            )
            for w in self.windows
            for mode, level in event.cells
            if mode in w.modes
        )

    def complete(
        self,
        event,
        *,
        snapshot_digest,
        cells,
        writer,
        protocol,
        evidence,
        step_envelope,
        reduced,
        rank,
        wandb_run=None,
        scalars=None,
    ):
        """Join completed step-owned summaries with private, already reduced probe records.

        cells is an iterable of (Cell, attestation, bounded S bytes, named M tensors).
        The worker, not this coordinator, executes native forward/noising on an isolated
        snapshot. This sink cannot substitute a different source, target or step on resume.
        Never call on a microstep, from an individual shard, or with an incomplete panel.
        """
        if not reduced:
            raise ValueError("distributed observations not yet reduced")
        if rank != 0:
            return None
        if not re.fullmatch("[0-9a-f]{64}", snapshot_digest):
            raise ValueError("missing exact snapshot digest")
        if not isinstance(step_envelope, bytes) or len(step_envelope) >= 1 << 20:
            raise ValueError("step envelope exceeds plan")
        if event.kind == "entry" and step_envelope:
            raise ValueError("stage entry is not an optimizer update")
        wanted = {c: i for i, c in enumerate(self.select(event))}
        summaries = {"step": step_envelope}
        maps = {}
        bindings = []
        for cell, actual, summary, matrices in cells:
            if cell not in wanted:
                raise ValueError("unexpected/replayed panel cell")
            i = wanted.pop(cell)
            required = dict(
                plan_sha256=PLAN_SHA256,
                window=cell.window.identity,
                target=cell.window.target,
                mode=cell.mode,
                nominal_sigma=cell.sigma,
                step=event.step,
                phase=event.phase,
                snapshot=snapshot_digest,
                branch="positive",
            )
            if any(actual.get(k) != v for k, v in required.items()):
                raise ValueError("probe target/snapshot/recipe or noise cell mismatch")
            if not isinstance(summary, bytes) or not 0 < len(summary) <= 2 << 20:
                raise ValueError("missing or oversized summary")
            summaries[f"cell.{i}"] = summary
            bindings.append((i, required))
            if set(matrices) != ({(0, 0), (35, 31)} if cell.full_map else set()):
                raise ValueError("expected exactly the selected sentinel heads")
            for (layer, head), m in matrices.items():
                if (
                    m.dtype != torch.float16
                    or m.ndim != 2
                    or not 0 < len(m) <= 4096
                    or m.shape != (len(m), len(m))
                    or len(m) != actual.get("real_tokens")
                ):
                    raise ValueError("not a complete canonical fp16 sentinel matrix")
                maps[f"cell.{i}.L{layer}.H{head}"] = (
                    m.detach().contiguous().cpu().numpy().tobytes()
                )
        if wanted:
            raise ValueError("incomplete isolated probe panel")
        return writer.publish(
            event,
            protocol,
            evidence=dict(evidence, snapshot=snapshot_digest, cells=sorted(bindings)),
            summaries=summaries,
            maps=maps,
            reduced=True,
            rank=rank,
            wandb_run=wandb_run,
            scalars=scalars,
        )
