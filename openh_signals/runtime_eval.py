"""Bind an existing, single-sample, sequential-CFG scored pass; never create probes."""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar

from .runtime import FRAMEWORK, pinned, scoped, wrap
from .runtime_pack import token_digest


class EvalCalls:
    """`identify(sample_args, batch)` must attest physical target and branch identities.

    Return the *current* private clip/seed/mode/chunk/ablation identifiers,
    positive and negative token digests, kernel and reference. An ordinal alone
    is not a physical identity. `emit` consumes detached compact diagnostics.
    A divergent, cached or fused CFG layout is a hold, not an extra forward.
    """

    def __init__(self, runtime, identify):
        self.rt, self.identify = runtime, identify
        self.sample = ContextVar("scored_sample", default=None)
        self.seen = ContextVar("scored_velocity_calls", default=None)

    def generate(self, original, args, kwargs, bound):
        batch = bound["sample_args_list"]
        if len(batch) != 1:
            raise ValueError("unbound multi-sample evaluation")
        context = self.identify(batch[0], bound["data_batch"])
        keys = {
            "target_digest",
            "mode",
            "chunk",
            "seed",
            "ablation",
            "kernel",
            "reference",
            "branches",
        }
        if not keys.issubset(context) or not context["target_digest"]:
            raise ValueError("unattested evaluation source/target/branch")
        seen = set()
        with (
            scoped(self.sample, context),
            scoped(self.seen, seen),
            scoped(
                self.rt.codec_cell,
                {
                    "kind": "eval_codec",
                    "target_digest": context["target_digest"],
                    "mode": context["mode"],
                    "chunk": context["chunk"],
                },
            ),
            scoped(self.rt.chunk_counts, {}),
        ):
            result = original(*args, **kwargs)
        if not seen:
            raise ValueError("no selected native velocity calls from this generation")
        from .events import solver_calls

        lengths = {n for i, n, branch in seen}
        if len(lengths) != 1 or {i for i, n, branch in seen} != set(
            solver_calls(lengths.pop())
        ):
            raise ValueError("scored pass missing solver locations")
        return result

    def velocity(self, original, args, kwargs, bound):
        meta, loc = self.sample.get(), self.rt.solver.get()
        if meta is None:
            return original(*args, **kwargs)
        if loc is None:
            raise ValueError("unbound solver; no guessed call index")
        i, n, selected, seed = loc
        if not selected:
            return original(*args, **kwargs)
        tokens = bound["text_tokens"]
        cache = bound.get("memory")
        if (
            len(tokens) != 1
            or (cache is not None and type(cache) is not self.rt.cache_class)
            or bound.get("skip_text_tokens")
        ):
            raise ValueError(
                "unknown cache, fused CFG or skipped text lacks a bound prompt identity"
            )
        branch = meta["branches"].get(token_digest(tokens[0]))
        if branch not in ("positive", "negative"):
            raise ValueError("conditional branch identity not attested")
        if seed != meta["seed"] and seed != [meta["seed"]]:
            raise ValueError("actual solver seed differs from physical sample binding")
        cell = {k: v for k, v in meta.items() if k != "branches"}
        # Solver t is not necessarily a flow sigma; never invent the latter.
        cell.update(
            kind="eval",
            branch=branch,
            solver_call=i,
            solver_calls=n,
            t=bound["timestep"].detach().flatten().tolist(),
            cache_hit=bool(cache and cache.is_gen_only()),
        )
        chosen = meta.get("sentinel_first_seed_real", False) and branch == "positive"
        cell["cache_sentinels"] = (0,) if chosen else ()
        cell["full_map"] = bool(chosen and i == (n - 1) // 2)
        key = (i, n, branch)
        if key in self.seen.get():
            raise ValueError(
                "multiple unresolved velocity branches at one solver location"
            )
        self.seen.get().add(key)
        with self.rt.cell(cell):
            return original(*args, **kwargs)

    @contextmanager
    def install(self, commit):
        if commit != FRAMEWORK:
            raise ValueError("no qualified native scorer adapter")
        pipe = pinned("cosmos_framework.inference.inference", commit)
        omni = pinned("cosmos_framework.model.generator.omni_mot_model", commit)
        with ExitStack() as stack:
            stack.enter_context(
                wrap(pipe.OmniInference, "generate_batch", self.generate)
            )
            stack.enter_context(wrap(omni.OmniMoTModel, "_get_velocity", self.velocity))
            yield self
