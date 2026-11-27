"""Trainer boundary adapter: stage natural microbatches; publish only confirmed updates."""

from contextlib import ExitStack, contextmanager


from .events import event_for
from .objective import FlowSample, observe_flow, observe_preclip
from .runtime import FRAMEWORK, pinned, require_local, scoped, wrap


class TrainSteps:
    """Pass `finish(event, rows)` to the single-writer coordinator.

    `finish` receives only detached summaries, and only *after* a real global
    update. It is not a callback for running diagnostic forwards on the live
    model. This binding requires ordinary, non-fused optimizers on one rank.
    """

    def __init__(self, runtime, phase, final_step, finish):
        self.runtime, self.phase, self.final_step, self.finish = (
            runtime,
            phase,
            final_step,
            finish,
        )
        self.pending = None
        self.micro = None
        self.sigmas = None
        self.pack = None

    def record(self, name, row):
        if self.pending is None or self.micro is None:
            raise ValueError("record outside sampled training microstep")
        rows = self.pending[1]
        rows.append((name, dict(**row, microstep=self.micro)))
        if len(rows) > 65536:
            raise ValueError("sampled step exceeds bounded diagnostic row limit")

    def noise(self, original, args, kwargs, bound):
        if self.micro is not None:
            v, a = bound["sigmas"], bound.get("sigmas_action")
            self.sigmas = (v.detach(), v.detach() if a is None else a.detach())
        return original(*args, **kwargs)

    def losses(self, original, args, kwargs, bound):
        if self.micro is None:
            return original(*args, **kwargs)
        cfg = bound["self"].config.rectified_flow_training_config
        if bound["is_image_batch"] or self.sigmas is None:
            raise ValueError("unexpected image-only or unbound noise schedule")
        self.pack = (bound["data_batch_packed"], cfg)
        try:
            return original(*args, **kwargs)
        finally:
            self.pack = None

    def describe(self, args):
        if self.pack is None:
            return None
        pack, cfg = self.pack
        if args["condition_mask"] is pack.action.condition_mask:
            modality, coeff, sigmas = "action", cfg.action_loss_weight, self.sigmas[1]
        elif args["condition_mask"] is pack.vision.condition_mask:
            modality, coeff, sigmas = "vision", cfg.loss_scale, self.sigmas[0]
        else:
            raise ValueError("flow mask is not a bound physical stream")
        if len(self.families) != len(pack.sample_lens) or len(sigmas) != len(
            self.families
        ):
            raise ValueError("cannot bind every sample to its actual family/sigma")
        out = []
        for i, family in enumerate(self.families):
            if not isinstance(family, str) or not family:
                raise ValueError("unbound family")
            vc = pack.vision.condition_mask[i].flatten().bool()
            ac = pack.action.condition_mask[i].flatten().bool()
            if len(vc) < 2 or len(ac) < 24 or not bool(ac[:-24].all() and vc[0]):
                raise ValueError("unexpected target geometry")
            if bool(vc.all() and (~ac[-24:]).all()):
                mode = "inverse_dynamics"
            elif bool((~vc[1:]).all() and ac.all()):
                mode = "forward_dynamics"
            elif bool((~vc[1:]).all() and (~ac[-24:]).all()):
                mode = "policy"
            else:
                raise ValueError("unrecognized joint masks")
            sigma = sigmas[i]
            if modality == "action" and sigma.numel() != 1:
                raise ValueError("action per-row sigma not attested for this schedule")
            out.append(FlowSample(modality, family, mode, sigma, coeff, i))
        return out

    def step(self, original, args, kwargs, bound):
        trainer, step, micro = (
            bound["self"],
            bound["iteration"] + 1,
            bound["grad_accum_iter"],
        )
        event = event_for(self.phase, step, final_step=self.final_step)
        if event is None:
            return original(*args, **kwargs)
        require_local()
        if micro == 0:
            if self.pending is not None:
                raise ValueError("unclosed accumulation window")
            self.pending = (event, [])
        if self.pending is None or self.pending[0] != event:
            raise ValueError("resumed inside unbound accumulation window")
        optimizer, scaler = bound["optimizer"], bound["grad_scaler"]
        if getattr(optimizer, "_step_supports_amp_scaling", False):
            raise ValueError("fused optimizer may skip an update inside step")
        calls = []

        def after(*unused):
            calls.append(True)

        self.micro = micro
        self.families = bound["data"].get("_action_family", ())
        if isinstance(self.families, str):
            self.families = [self.families]
        try:
            with ExitStack() as stack:
                stack.callback(optimizer.register_step_post_hook(after).remove)
                stack.enter_context(
                    observe_preclip(trainer.callbacks, lambda i: i == step, self.record)
                )
                stack.enter_context(scoped(self.runtime.receiver, self.record))
                stack.enter_context(
                    self.runtime.cell(
                        dict(
                            kind="natural",
                            microstep=micro,
                            step=step,
                            amp_scale=float(scaler.get_scale()),
                            accumulation=trainer.config.trainer.grad_accum_iter,
                        )
                    )
                )
                result = original(*args, **kwargs)
            if result[2] == 0:
                if calls != [True]:
                    raise ValueError(
                        "no confirmed optimizer update; cannot use trainer iteration as completed-step counter"
                    )
                complete = self.pending
                self.pending = None
                self.finish(*complete)
            elif calls:
                raise ValueError("unexpected optimizer update during accumulation")
            return result
        finally:
            self.micro = self.sigmas = self.pack = self.families = None

    @contextmanager
    def install(self, commit, *, patch_digest=None):
        public = commit == FRAMEWORK
        prefix = "cosmos_framework.model.generator" if public else "cosmos3._src.vfm"
        omni = pinned(
            prefix + ("" if public else ".models") + ".omni_mot_model", commit
        )
        weight = pinned(prefix + ".algorithm.loss.time_weight", commit).TrainTimeWeight
        trainer = pinned(
            "cosmos_framework.trainer" if public else "cosmos3._src.imaginaire.trainer",
            commit,
            patch_digest=patch_digest,
        )
        with ExitStack() as stack:
            stack.enter_context(observe_flow(omni, weight, self.describe, self.record))
            stack.enter_context(
                wrap(omni.OmniMoTModel, "_add_noise_to_input", self.noise)
            )
            stack.enter_context(wrap(omni.OmniMoTModel, "_compute_losses", self.losses))
            stack.enter_context(
                wrap(trainer.ImaginaireTrainer, "training_step", self.step)
            )
            yield self
