"""Single-worker scorer coordinator. Private fetch attestations are inputs, never source data."""

from contextlib import contextmanager, ExitStack
from hashlib import sha256, sha1
import importlib
import io
import json
from pathlib import Path
import re

import numpy as np
import torch

from .attention import FrozenText
from .evaluation import eval_score_hook
from .events import Event, MIB, evaluation_reservation
from .runtime import FRAMEWORK, require_local
from .runtime_eval import EvalCalls


def _identity(x):
    """Hash live tensor bytes and geometry only. Never open media or weights."""
    t = x.detach().contiguous().cpu()
    h = sha256(f"{t.dtype}|{tuple(t.shape)}|".encode())
    h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
    return h.hexdigest()


def _blob(path):
    raw = path.read_bytes()
    return sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def _archive(arrays):
    # One bounded uncompressed payload per cell/stratum; outer container
    # compresses it. No pickle, source tensors, prompts or clips.
    out = io.BytesIO()
    np.savez(out, **arrays)
    return out.getvalue()


def _canonical_text(arr, meta, bank):
    """Read compact *executed* UND rows through their retained first observation."""
    ptr = meta.get("text_reference")
    if ptr is None:
        return {}  # cached unexecuted UND remains NA in native layout
    ids = arr["mot_query_groups"].astype(np.intp)
    tg = arr["text_rows"].astype(np.intp)
    if len(np.unique(ids)) != len(ids) or not np.isin(ids, np.arange(12)).all():
        raise ValueError("invalid compact query-group coordinates")
    if ptr == meta["observation_reference"]:
        if ptr in bank or not np.array_equal(ids, np.arange(12)):
            raise ValueError("first frozen text record does not own all query groups")
        bank[ptr] = (tg.copy(), arr["mass"][:, tg].copy(), arr["entropy"][:, tg].copy())
    if ptr not in bank:
        raise ValueError("missing retained frozen text owner")
    stored, mass, entropy = bank[ptr]
    if not np.array_equal(stored, tg):
        raise ValueError("frozen text group layout differs")
    out = {}
    for field, base in [("mass", mass), ("entropy", entropy)]:
        value = arr[field]
        dense = np.full(
            (value.shape[0], 12, *value.shape[2:]), np.nan, dtype=value.dtype
        )
        dense[:, ids] = value
        dense[:, tg] = base
        out[field] = dense
    return out


class SingleWorkerEvaluation:
    """Concrete binding for one pinned README scorer invocation, one process,
    sequential single-clip generation. `qualified` is code-owned, not JSON.

    Each catalog instance identifies an actual (name, seed, chunk, ablation)
    plus two modality digests from the gated crosswalk. This adapter does not
    create the crosswalk. One process must own every instance in the event.
    """

    memory_rows = 0
    WORKER_BLOB = "46fe5810cffffeffc8f262615fc81ea883da7f45"
    SCORER_BLOB = "5308cd9c29a19d4432d9608d8f0619dac6ef2a63"

    @classmethod
    def preflight(cls, target, spec, probes, writer):
        require_local()
        if (
            not isinstance(target, Path)
            or _blob(target) != cls.WORKER_BLOB
            or spec["source_commit"] != FRAMEWORK
        ):
            raise ValueError("ordinary single-worker scorer/source only")
        if spec["protocol"] != "readme-eval" or spec[
            "reservation"
        ] != evaluation_reservation(*spec["eval_counts"]):
            raise ValueError(
                "inconsistent scorer protocol or complete event reservation"
            )
        return cls(spec, probes, writer)

    def __init__(self, spec, probes, writer):
        self.spec, self.writer, self.text_spans = spec, writer, probes["text_spans"]
        self.frozen_text = FrozenText()  # evaluated checkpoint is a distinct event
        self.catalog = {}
        for row in probes["instances"]:
            key = row["name"], row["seed"]
            if key in self.catalog or row["chunk"] != 0 or row["ablation"] != "real":
                raise ValueError("repeated or unsupported evaluation instance")
            for name in (
                "target_digest",
                "action_digest",
                "video_digest",
                "root_digest",
                "index_digest",
                "window_digest",
            ):
                if not re.fullmatch("[a-f0-9]{64}", row[name]):
                    raise ValueError("incomplete physical evaluation binding")
            self.catalog[key] = row
        r, p, j, n = spec["eval_counts"]
        if (
            len(self.catalog) != n
            or len({s["family"] for s in self.catalog.values()}) > r
            or len({s["window_digest"] for s in self.catalog.values() if s["probe"]})
            > p
            or len(
                {
                    (s["mode"], s["seed"], s["chunk"], s["ablation"])
                    for s in self.catalog.values()
                }
            )
            > j
        ):
            raise ValueError("actual instances exceed preflight reservation")
        self.active = None
        self.scored, self.strata, self.individual, self.maps, self.per_clip = (
            set(),
            {},
            {},
            {},
            [],
        )
        self.actual_event_bytes = 0
        self.text_baselines = {}

    def identify(self, sample, batch):
        key = sample.name, sample.seed
        row = self.catalog.get(key)
        if row is None or key in self.scored or self.active is not None:
            raise ValueError("unexpected, repeated or overlapping evaluation instance")
        if (
            _identity(batch["video"][0]) != row["video_digest"]
            or _identity(batch["action"][0]) != row["action_digest"]
        ):
            raise ValueError("actual input differs from frozen physical window")
        self.active, self.current = key, {}
        self.current_bytes, self.current_meta, self.map_info = (
            0,
            {},
            getattr(self, "map_info", {}),
        )
        return dict(
            row,
            reference=":".join(
                (
                    self.spec["protocol"],
                    str(self.spec["step"]),
                    row["window_digest"],
                    row["name"],
                    str(row["seed"]),
                    str(row["chunk"]),
                    row["ablation"],
                )
            ),
            kernel=self.spec["kernel"],
        )

    def emit(self, name, row):
        if name == "eval.error":
            return self.score(row)
        if self.active is None:
            raise ValueError("signal outside a physically bound clip")
        cell = row.get("cell", {})
        tags = tuple(
            row.get(k)
            for k in ("sample", "layer", "tower", "codec", "chunk", "plane", "batch")
        )
        path = json.dumps(
            (name, cell.get("solver_call"), cell.get("branch"), tags),
            separators=(",", ":"),
        )
        if path in self.current:
            raise ValueError("duplicate per-clip boundary")
        arr, meta = {}, {}
        for k, v in row.items():
            if k == "cell" or k in (
                "sample",
                "layer",
                "tower",
                "codec",
                "chunk",
                "plane",
                "batch",
            ):
                continue
            if k == "full_map":
                if (
                    not cell.get("full_map")
                    or v.dtype != torch.float16
                    or v.ndim != 2
                    or v.shape[0] != v.shape[1]
                    or len(v) > 4096
                ):
                    raise ValueError("unqualified or oversized full map")
                if len(self.maps) >= 4:
                    raise ValueError("too many deep maps")
                mid = "head" + str(len(self.maps))
                self.maps[mid] = v.detach().contiguous().cpu().numpy().tobytes()
                self.map_info[mid] = dict(
                    window=cell["window_digest"],
                    layer=row["layer"],
                    head={0: 0, 35: 31}[row["layer"]],
                    branch=cell["branch"],
                    solver_call=cell["solver_call"],
                    seed=cell["seed"],
                    mode=cell["mode"],
                    shape=list(v.shape),
                )
            elif isinstance(v, torch.Tensor):
                arr[k] = v.detach().to("cpu").contiguous().numpy().copy()
            elif isinstance(v, (bool, int, float, list, tuple)):
                a = np.asarray(v)
                if a.dtype.kind in "biuf":
                    arr[k] = a
                else:
                    meta[k] = v
            elif isinstance(v, (str, type(None))):
                meta[k] = v
            else:
                raise ValueError("unbounded or unsupported observation type")
        if cell.get("kind") == "eval":
            meta.update(
                t=cell["t"],
                solver_calls=cell["solver_calls"],
                cache_hit=cell["cache_hit"],
            )
        self.current_bytes += sum(a.nbytes for a in arr.values()) + len(
            json.dumps(meta)
        )
        # Up to three solver locations x two CFG branches and a codec reference.
        if self.current_bytes > 6 * 2 * MIB + MIB:
            raise ValueError("clip summaries exceed plan size")
        self.current[path] = arr
        self.current_meta[path] = meta

    def score(self, row):
        if self.active is None or row["sample_name"] != self.active[0]:
            raise ValueError("unmatched scorer return")
        ref = self.catalog[self.active]
        k = self.active
        if (
            "target_digest" in row
            and row["target_digest"] != ref["future_action_digest"]
        ):
            raise ValueError("future action scorer target mismatch")
        # Real positive branches must observe the entire stack at all three solver calls.
        layers = {}
        for path in self.current:
            name, step, branch, tags = json.loads(path)
            if name == "mot":
                layers.setdefault((step, branch), set()).add(tags[1])
        expected = {(i, b) for i, bs in ref["solver_branches"] for b in bs}
        if not expected or row["mode"] != ref["mode"]:
            raise ValueError("unbound solver schedule or scorer mode")
        for support in ref["required_signals"]:
            if not any(json.loads(p)[0] == support for p in self.current):
                raise ValueError(f"missing required scored signal: {support}")
        if ref.get("sentinel_first_seed_real"):
            seen_maps = {
                v["layer"]
                for v in self.map_info.values()
                if v["window"] == ref["window_digest"]
            }
            if seen_maps != {0, 35}:
                raise ValueError("incomplete selected full maps")
        if layers.keys() != expected or not all(
            v == set(range(36)) for v in layers.values()
        ):
            raise ValueError("missing or repeated protocol/layer evidence")
        group = tuple(ref[s] for s in ("family", "mode", "seed", "ablation", "chunk"))
        bins = self.strata.setdefault(group, {})
        fields = {}
        descriptors = {}
        focus = []
        for idx, (path, arr) in enumerate(self.current.items()):
            name = json.loads(path)[0]
            canonical = (
                _canonical_text(arr, self.current_meta[path], self.text_baselines)
                if name == "mot"
                else {}
            )
            if name == "mot" and "mass" in arr:
                qids = arr.get("mot_query_groups", np.arange(arr["mass"].shape[-2]))
                fq = np.flatnonzero(qids == 7)
                if len(fq):
                    focus.extend(arr["mass"][:, fq[0], 1].reshape(-1).tolist())
            for field, a in arr.items():
                # Stratum means of token-level vectors with variable length would
                # be meaningless. Keep exact ranges/counts for individual probes;
                # only aggregate homologous layer/group coordinates for others.
                stat = canonical.get(field, a)
                if (
                    "groups" in arr
                    and field not in ("groups", "scale")
                    and a.ndim
                    and len(a) == len(arr["groups"])
                ):
                    ids = np.asarray(arr["groups"], dtype=np.intp)
                    stat = np.full((12, *a.shape[1:]), np.nan, dtype=np.float32)
                    stat[ids] = a * arr["scale"][:, None] if field == "repr16" else a
                elif field not in (
                    "mass",
                    "entropy",
                    "read_strength",
                    "legal",
                    "counts",
                    "future_to_memory",
                    "memory_to_future",
                    "max_row_error",
                    "plane_rms",
                ):
                    stat = None
                if stat is not None:
                    if (path, field) not in bins:
                        bins[path, field] = [
                            np.zeros_like(stat, dtype=np.float64),
                            np.zeros_like(stat, dtype=np.int32),
                        ]
                    total, count = bins[path, field]
                    if stat.shape != total.shape:
                        raise ValueError("stratum group coordinates changed")
                    if np.isinf(stat).any():
                        raise ValueError("nonfinite clip statistic")
                    valid = np.isfinite(stat)
                    total += np.where(valid, stat, 0)
                    count += valid
                if ref["probe"]:
                    tag = f"r{idx}_{field}"
                    fields[tag] = a
                    descriptors[tag] = (path, field)
        archive_name = None
        if ref["probe"]:
            archive_name = "probe" + str(len(self.individual))
            fields["metadata"] = np.frombuffer(
                json.dumps(
                    (descriptors, self.current_meta), separators=(",", ":")
                ).encode(),
                dtype=np.uint8,
            )
            payload = _archive(fields)
            if len(payload) > 12 * MIB:
                raise ValueError("serialized clip exceeds six-cell reservation")
            self.individual[archive_name] = payload
            self.actual_event_bytes += len(payload)
            if self.actual_event_bytes > self.spec["reservation"]:
                raise ValueError("uncompressed reservation exceeded")
        clip_row = dict(
            **row,
            window_digest=ref["window_digest"],
            family=ref["family"],
            seed=ref["seed"],
            root_digest=ref["root_digest"],
            index_digest=ref["index_digest"],
            chunk=ref["chunk"],
            ablation=ref["ablation"],
            probe_archive="S/" + archive_name if archive_name else None,
        )
        focus = np.asarray(focus)
        focus = focus[np.isfinite(focus)]
        if focus.size:
            clip_row.update(
                future_to_task_mean=float(focus.mean()),
                future_to_task_min=float(focus.min()),
                future_to_task_max=float(focus.max()),
                future_to_task_count=int(focus.size),
            )
        if len(json.dumps(clip_row).encode()) > 4096:
            raise ValueError("oversized scorer row")
        self.per_clip.append(clip_row)
        self.scored.add(k)
        self.active = None
        self.current = {}
        self.current_meta = {}

    def finish(self):
        if (
            self.scored != set(self.catalog)
            or self.active is not None
            or not self.scored
        ):
            raise ValueError("incomplete evaluation; no event published")
        summaries = dict(self.individual)
        summaries["clips"] = json.dumps(self.per_clip, separators=(",", ":")).encode()
        for i, (key, bins) in enumerate(sorted(self.strata.items())):
            fields = {}
            desc = {}
            for j, ((path, field), (total, count)) in enumerate(sorted(bins.items())):
                tag = f"f{j}"
                desc[tag] = (path, field)
                fields[tag] = np.where(
                    count > 0, total / np.maximum(count, 1), np.nan
                ).astype("float32")
                fields[tag + "_count"] = count
            fields["schema"] = np.frombuffer(
                json.dumps([key, desc], separators=(",", ":")).encode(), dtype=np.uint8
            )
            summaries[f"stratum{i}"] = _archive(fields)
        evidence = {
            k: self.spec[k]
            for k in ("source_commit", "config_digest", "probe_digest", "kernel")
        }
        evidence.update(full_maps=self.map_info)
        event = Event(self.spec["step"], "eval", "score", True, ())
        import sys

        existing_run = getattr(sys.modules.get("wandb"), "run", None)
        scalars = {"scored_instances": len(self.scored)}
        for field in ("action_mse", "psnr", "future_to_task_mean"):
            values = [
                r[field] for r in self.per_clip if field in r and np.isfinite(r[field])
            ]
            if values:
                scalars["clip_mean." + field] = float(np.mean(values))
        return self.writer.publish(
            event,
            self.spec["protocol"],
            evidence=evidence,
            summaries=summaries,
            maps=self.maps,
            reduced=True,
            scalars=scalars,
            wandb_run=existing_run,
        )

    @contextmanager
    def attach(self, runtime):
        scorer = importlib.import_module("cosmos3_h_embodied.scripts.eval_utils")
        if _blob(Path(scorer.__file__)) != self.SCORER_BLOB:
            raise ValueError("unqualified scorer source")
        with ExitStack() as stack:
            # Wrap the defining function BEFORE the entrypoint imports by value.
            stack.enter_context(eval_score_hook(scorer, self.emit))
            stack.enter_context(EvalCalls(runtime, self.identify).install(FRAMEWORK))
            yield
        self.finish()
