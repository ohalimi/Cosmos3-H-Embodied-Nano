"""Process-local, reversible read sites. No trainer, loader or model is constructed here."""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import subprocess

import torch

from .attention import observe_two_way
from .events import solver_calls
from .runtime_pack import GROUPS, label_pack

FRAMEWORK = "7cfaf7854b2114eda5deb36250a7e8f00c64d69e"
NATIVE = frozenset(
    (
        "d3183b2db52afbde536ee662696cd5418f2002d9",
        "aeb0f03e8a652d346dcd20b24d88e6eb558b2ba8",
        "6f8eb4dcd41ed7e0170d115d6bc88cefd638c398",
        "2be2de744a862aa6e8bfbcb4b5583c550c36ea7a",
        "86ac356c89a08623471d22d216dfddec6236a4e4",
    )
)


S1_PATCH = "2c8139766ce19162478f573a3ffac5f04f79ffd9f7978dcda066c9a67bce643c"


def hook_blob(commit, path, tracked, patch_digest):
    if patch_digest is None:
        return tracked
    if commit != "aeb0f03e8a652d346dcd20b24d88e6eb558b2ba8" or patch_digest != S1_PATCH:
        raise ValueError("unattested release patch")
    if path == "cosmos3/_src/imaginaire/trainer.py":
        if tracked != "774398a5238ab9d79a0c94726b1d72875778644b":
            raise ValueError("wrong release patch base")
        return "5d9cfcf2a7b91799e2999dd151838c2284353bb1"
    return tracked


def pinned(name, commit, patch_digest=None):
    """Check actual import origin and hook-site bytes, not just a user-supplied revision string."""
    if commit not in NATIVE | {FRAMEWORK}:
        raise ValueError("unknown source pin")
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        raise ValueError(f"missing source module: {name}")
    path = Path(spec.origin).resolve(strict=True)

    def git(*args):
        return (
            subprocess.check_output(
                ["git", "--no-replace-objects", "-C", str(path.parent), *args]
            )
            .decode()
            .strip()
        )

    root = Path(git("rev-parse", "--show-toplevel"))
    raw = path.read_bytes()
    relative = path.relative_to(root).as_posix()
    expected = hook_blob(
        commit, relative, git("rev-parse", f"{commit}:{relative}"), patch_digest
    )
    if (
        git("rev-parse", "HEAD") != commit
        or expected
        != hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    ):
        raise ValueError(f"source commit or hook-site bytes differ: {name}")
    mod = importlib.import_module(name)
    if Path(mod.__file__).resolve() != path:
        raise ValueError(f"loaded source differs: {name}")
    return mod


@contextmanager
def wrap(owner, name, interceptor):
    """Wrap a class method/module binding; never monkeypatch torch or the attention backend."""
    original = getattr(owner, name)
    signature = inspect.signature(original)
    inherited = name not in vars(owner)

    @wraps(original)
    def call(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return interceptor(original, args, kwargs, bound.arguments)

    setattr(owner, name, call)
    try:
        yield
    finally:
        if getattr(owner, name) is not call:
            raise RuntimeError(f"{name} replaced during observer scope")
        if inherited:
            delattr(owner, name)
        else:
            setattr(owner, name, original)


@contextmanager
def scoped(variable, value):
    token = variable.set(value)
    try:
        yield value
    finally:
        variable.reset(token)


def require_local():
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or (
        torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1
    ):
        raise ValueError("capture requires an attested global shard-reduction binding")
    if (
        torch.compiler.is_compiling()
        or torch.cuda.is_initialized()
        and torch.cuda.is_current_stream_capturing()
    ):
        raise ValueError("compiled/captured graph has no calibrated observer binding")


class Runtime:
    """Observe only calls inside an explicitly selected cell. ``emit`` is synchronous.

    A cell contains private metadata; it is NOT a request to fetch a sample,
    generate noise, add a forward, or re-run an encoder. The adapter refuses
    unknown pack layouts rather than silently dropping attention boundaries.
    """

    def __init__(self, text_spans, emit, *, memory_rows=0, frozen_text=None):
        self.spans, self.memory_rows, self.frozen_text = (
            text_spans,
            memory_rows,
            frozen_text,
        )
        self.receiver = ContextVar("signal_receiver", default=emit)
        self.emit = lambda name, row: self.receiver.get()(name, row)
        self.current = ContextVar("signal_cell", default=None)
        self.groups = ContextVar("signal_groups", default=None)
        self.layer = ContextVar("signal_layer", default=None)
        self.solver = ContextVar("signal_solver", default=None)
        self.chunk_counts = ContextVar("signal_codec_chunks", default=None)
        self.codec_cell = ContextVar("signal_codec_cell", default=None)
        self.cache_class = None

    @contextmanager
    def cell(self, metadata):
        require_local()
        if self.current.get() is not None:
            raise ValueError("overlapping observation cells")
        with scoped(self.current, metadata), scoped(self.chunk_counts, {}):
            yield

    def network(self, original, args, kwargs, bound):
        cell = self.current.get()
        if cell is None:
            return original(*args, **kwargs)
        model = bound["self"]
        memory = bound.get("memory")
        has_cache = memory is not None and type(memory) is self.cache_class
        if (
            (memory is not None and not has_cache)
            or bound.get("video_temporal_causal")
            or getattr(model, "video_temporal_causal", False)
            or model.config.joint_attn_implementation != "two_way"
        ):
            raise ValueError(
                "cached/alternate attention not equivalent to executed two-way calls"
            )
        groups, records = label_pack(
            bound["packed_seq"], self.spans, memory_rows=self.memory_rows
        )
        layers = model.language_model.model.layers
        # The set is per network forward. Backward checkpoint replay runs outside this scope.
        seen = set()
        if cell.get("kind") != "natural":
            for i, record in enumerate(records):
                self.emit("pack.coverage", dict(**record, sample=i, cell=cell))
        from .features import bridge_features, decoder_features
        from .objective import watch_group

        pack = bound["packed_seq"]
        with ExitStack() as stack:
            stack.enter_context(scoped(self.groups, (groups, seen, records)))
            # Per-sample summaries do not pool across a packed batch. Ignore
            # other logical samples explicitly, not by cropping an activation.
            natural = cell.get("kind") == "natural"
            if natural:

                def bind_grad(name, labels, is_block=False):
                    def tap(module, args, output):
                        if is_block:
                            pieces = []
                            for stream, idx in (
                                ("causal_seq", "_causal_indices"),
                                ("full_only_seq", "_full_indices"),
                            ):
                                p = output[0]
                                if stream not in p:
                                    raise ValueError(
                                        "unbound residual pack for gradient"
                                    )
                                x = p[stream]
                                g = groups[p[idx]]
                                if len(g) < len(x):
                                    g = torch.nn.functional.pad(
                                        g, (0, len(x) - len(g)), value=-1
                                    )
                                pieces.append((x, g, name + "." + stream))
                        else:
                            pieces = [(output, labels, name)]
                        for x, g, key in pieces:
                            watch_group(
                                x,
                                g,
                                key,
                                cell["amp_scale"],
                                self.emit,
                                accumulation=cell["accumulation"],
                            )

                    return tap

                for j, block in enumerate(layers):
                    stack.enter_context(
                        block.register_forward_hook(bind_grad(f"layer.{j}", None, True))
                    )
                for name, index in (
                    ("action2llm", pack.action.sequence_indexes),
                    ("vae2llm", pack.vision.sequence_indexes),
                ):
                    stack.enter_context(
                        getattr(model, name).register_forward_hook(
                            bind_grad(name, groups[index])
                        )
                    )
            else:
                if self.memory_rows:
                    from .memory import memory_prefix

                    @torch.no_grad()
                    def prefix_tap(module, args, output):
                        raw = args[0]
                        sizes = [3 + r["history"] + 24 for r in records]
                        if (
                            raw.ndim != 2
                            or sum(sizes) != raw.size(0)
                            or output.size(0) != raw.size(0)
                        ):
                            raise ValueError(
                                "CAMP projection does not cover these physical payloads"
                            )
                        for i, (x, y, r) in enumerate(
                            zip(
                                raw.split(sizes),
                                output.split(sizes),
                                records,
                                strict=True,
                            )
                        ):
                            self.emit(
                                "memory.prefix",
                                dict(
                                    **memory_prefix(x, y, r["history"]),
                                    sample=i,
                                    cell=cell,
                                ),
                            )

                    stack.enter_context(
                        model.action2llm.register_forward_hook(prefix_tap)
                    )
                for i, rec in enumerate(records):
                    labels = torch.full_like(groups, -1)
                    a, b = rec["offset"], rec["offset"] + rec["length"]
                    labels[a:b] = groups[a:b]

                    def emit(name, row, i=i):
                        self.emit(name, dict(**row, sample=i, cell=cell))

                    text = torch.isin(
                        groups, torch.tensor([0, 1, 2, 11], device=groups.device)
                    )
                    if has_cache and memory.is_gen_only():
                        emit(
                            "cache.coverage",
                            {"und_queries_executed": False, "decoder_full_tap": False},
                        )
                    else:
                        for j, block in enumerate(layers):
                            stack.enter_context(
                                decoder_features(
                                    block,
                                    j,
                                    labels[text],
                                    labels[~text],
                                    len(GROUPS),
                                    emit,
                                )
                            )
                    for name, index in (
                        ("action2llm", pack.action.sequence_indexes),
                        ("vae2llm", pack.vision.sequence_indexes),
                    ):
                        stack.enter_context(
                            bridge_features(
                                getattr(model, name),
                                name,
                                labels[index],
                                len(GROUPS),
                                emit,
                            )
                        )
            output = original(*args, **kwargs)
        if seen != set(range(len(layers))):
            raise ValueError("unobserved/replayed layer or alternate attention path")
        return output

    def layer_call(self, original, args, kwargs, bound):
        if self.groups.get() is None:
            return original(*args, **kwargs)
        with scoped(self.layer, bound["self"].layer_idx):
            return original(*args, **kwargs)

    def attention(self, original, args, kwargs, bound):
        active = self.groups.get()
        if active is None:
            return original(*args, **kwargs)
        groups, seen, records = active
        count = len(records)
        layer = self.layer.get()
        if not isinstance(layer, int) or layer in seen:
            raise ValueError("unknown/duplicate layer under packed forward")
        seen.add(layer)
        cell = self.current.get()
        if cell.get("kind") == "natural":
            return original(
                *args, **kwargs
            )  # only objective/gradients on natural optimizer microbatches
        # Mixed sentinel packs need per-item selection; never retain extra maps.
        sentinel = bool(cell.get("full_map", False))
        if sentinel and count != 1:
            raise ValueError("sentinel requires a unique physical cell")
        rows = observe_two_way(bound, groups, len(GROUPS), layer, sentinel=sentinel)
        if len(rows) != count:
            raise ValueError("softmax/pack sample boundary mismatch")
        for i, row in enumerate(rows):
            if self.memory_rows:
                from .memory import memory_reads

                self.emit(
                    "memory.read",
                    dict(**memory_reads(row), sample=i, layer=layer, cell=cell),
                )
            # Non-probe evaluation clips are reduced to stratum means, not
            # retained as individual observations, so cannot own a durable row.
            if self.frozen_text is not None and cell.get("probe", True):
                key = "|".join(
                    (
                        records[i]["text_digest"],
                        str(layer),
                        cell["branch"],
                        cell["kernel"],
                    )
                )
                reference = "/".join(
                    map(
                        str,
                        (
                            cell["reference"],
                            cell.get("solver_call", "fixed"),
                            cell["branch"],
                            layer,
                            i,
                        ),
                    )
                )
                row = self.frozen_text.apply(key, row, reference)
                row["observation_reference"] = reference
            self.emit("mot", dict(**row, layer=layer, sample=i, cell=cell))
        return original(*args, **kwargs)

    def wan_call(self, original, args, kwargs, bound):
        cell = self.current.get() or self.codec_cell.get()
        if cell is None or cell.get("kind") == "natural":
            return original(*args, **kwargs)
        from .features import wan_features

        codec = bound["self"]
        stream = type(codec).__name__
        chunks = self.chunk_counts.get()
        i = chunks.get(stream, 0)
        chunks[stream] = i + 1

        def emit(n, r):
            self.emit(n, dict(**r, codec=stream, cell=cell))

        with wan_features(codec.middle[1], emit, first_chunk=i):
            return original(*args, **kwargs)

    def solver_call(self, original, args, kwargs, bound):
        """Wrap only the passed solver callable; count actual calls (not CFG forwards)."""
        fn = bound["velocity_fn"]
        n = bound["num_steps"]
        keep = solver_calls(n)
        calls = 0

        @wraps(fn)
        def velocity(*a, **kw):
            nonlocal calls
            i, calls = calls, calls + 1
            with scoped(self.solver, (i, n, i in keep, bound.get("seed"))):
                return fn(*a, **kw)

        # Preserve original invocation (including parameters that originally took defaults).
        if "velocity_fn" in kwargs:
            kwargs = {**kwargs, "velocity_fn": velocity}
        else:
            args = (args[0], velocity, *args[2:])
        result = original(*args, **kwargs)
        if calls != n:
            raise ValueError("actual and requested UniPC call counts differ")
        return result

    @contextmanager
    def install(self, commit):
        """Install at launch in THIS Python worker, on the exact source revision."""
        public = commit == FRAMEWORK
        if not public and commit not in NATIVE:
            raise ValueError("unapproved hook source")
        model = (
            "cosmos_framework.model.generator" if public else "cosmos3._src.vfm.models"
        )
        diffusion = model if public else "cosmos3._src.vfm"
        attention = pinned(model + ".mot.attention", commit)
        tower = pinned(model + ".mot.unified_mot", commit)
        net = pinned(model + ".mot.cosmos3_vfm_network", commit)
        sampler = pinned(diffusion + ".diffusion.samplers.unipc", commit)
        codec = pinned(diffusion + ".tokenizers.wan2pt2_vae_4x16x16", commit)
        with ExitStack() as stack:
            if public:
                from .evaluation import cached_attention

                cache_mod = pinned(model + ".mot.inference_text_kv_memory", commit)
                self.cache_class = cache_mod.InferenceTextKVMemoryState
                stack.callback(setattr, self, "cache_class", None)
                stack.enter_context(cached_attention(self, cache_mod))
            for owner, method, func in (
                (attention, "two_way_attention", self.attention),
                (codec.Encoder3d, "forward", self.wan_call),
                (codec.Decoder3d, "forward", self.wan_call),
                (tower.PackedAttentionMoT, "forward", self.layer_call),
                (net.Cosmos3VFMNetwork, "forward", self.network),
                (sampler.UniPCSampler, "forward", self.solver_call),
            ):
                stack.enter_context(wrap(owner, method, func))
            yield self
