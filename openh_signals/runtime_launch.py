"""Private worker preflight and launch-time wiring (no training/evaluation imports when off)."""

from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path

from .events import Event, EventWriter, PLAN_SHA256, read_probe_list
from .runtime import FRAMEWORK, Runtime


# Independently approved full-manifest hashes (immutable plan, source, config,
# budget and probe digests); never a truthy user-supplied `approved` field.
# The concrete single-worker scorer adapter exists below the qualification
# gate; other protocols without an implemented coordinator remain closed.
QUALIFIED_MANIFESTS = frozenset()
CALIBRATION = (
    Path(__file__).resolve().parents[1] / "tests/signals/test_runtime_reference.py"
)


def _parse_manifest(path):
    with path.open("rb") as f:
        raw = f.read(65537)
    if len(raw) > 65536:
        raise ValueError("oversize worker manifest")
    spec = json.loads(raw)
    if not isinstance(spec, dict) or spec.get("plan") != PLAN_SHA256:
        raise ValueError("unrecognized signal plan")
    return spec, hashlib.sha256(raw).hexdigest()


@contextmanager
def attach_worker(path, target, argv=()):
    """Attach before invoking the worker. Never import/execute a worker to discover its identity.

    Real worker binding requires a trusted adapter with preflight(), emit(),
    scoped train/eval metadata and calibrated storage/reduction ownership. No
    manifest boolean promotes an unqualified release or a synthetic pass.
    """
    spec, manifest_digest = _parse_manifest(path)
    protocol = spec["protocol"]
    calibration = protocol == "wan-cpu-calibration" and target == CALIBRATION
    if not calibration and (
        protocol != "readme-eval" or manifest_digest not in QUALIFIED_MANIFESTS
    ):
        raise ValueError(
            "unqualified worker: missing verified fixed-probe/physical binding, "
            "global shard ownership and backend parity for this exact source/config/protocol"
        )
    if (
        not calibration
        and hashlib.sha256(
            json.dumps(list(argv), separators=(",", ":")).encode()
        ).hexdigest()
        != spec["argv_digest"]
    ):
        raise ValueError("worker arguments differ from qualified invocation")
    probes = read_probe_list(spec["probes"], expected_sha256=spec["probe_digest"])
    writer = EventWriter(
        spec["run_dir"],
        reservation=spec["reservation"],
        kept_bytes=spec["kept_bytes"],
        previous_bytes=spec.get("previous_bytes", 0),
    )
    with ExitStack() as stack:
        if calibration:
            if (
                spec["source_commit"] != FRAMEWORK
                or spec["config_digest"]
                != hashlib.sha256(CALIBRATION.read_bytes()).hexdigest()
            ):
                raise ValueError("calibration source/config differs")
            rows = []

            def emit(name, fields):
                if (
                    name not in ("encoder.spatial", "encoder.spatial.value")
                    or len(rows) >= 8
                ):
                    raise ValueError("unexpected calibration signal")
                rows.append(
                    (
                        name,
                        {
                            k: v.tolist() if hasattr(v, "tolist") else v
                            for k, v in fields.items()
                        },
                    )
                )

            rt = Runtime(probes["text_spans"], emit)
            stack.enter_context(rt.install(spec["source_commit"]))
            stack.enter_context(rt.cell({"kind": "calibration"}))
            yield
            if not rows:
                raise ValueError("no executed calibration hook")
            writer.publish(
                Event(0, "calibration", "synthetic", False, ()),
                protocol,
                evidence={
                    "source_commit": spec["source_commit"],
                    "probe_digest": spec["probe_digest"],
                    "config_digest": spec["config_digest"],
                    "manifest_digest": manifest_digest,
                },
                summaries={"codec": json.dumps(rows, allow_nan=False).encode()},
                reduced=True,
            )
        else:
            # The adapter computes checked capacity from its resolved plan,
            # verifies input attestations, source/patch/config and topology,
            # and supplies the actual completed-event coordinator, not a flag.
            from .runtime_eval_store import SingleWorkerEvaluation

            binding = SingleWorkerEvaluation.preflight(target, spec, probes, writer)
            rt = Runtime(
                binding.text_spans,
                binding.emit,
                memory_rows=binding.memory_rows,
                frozen_text=binding.frozen_text,
            )
            stack.enter_context(rt.install(spec["source_commit"]))
            stack.enter_context(binding.attach(rt))
            yield
