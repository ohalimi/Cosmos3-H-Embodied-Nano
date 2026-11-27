import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import zipfile

import pytest
import torch

from openh_signals.events import PLAN_SHA256
from openh_signals.runtime import FRAMEWORK, Runtime, wrap
from openh_signals.runtime_pack import label_pack, token_digest
from openh_signals.runtime_steps import TrainSteps


def test_launch(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = Path(__file__).with_name("test_runtime_reference.py").resolve()
    launcher = root / "openh_signals/launch.py"
    probes = tmp_path / "bindings.json"
    probes.write_text('{"text_spans":{}}')
    config = dict(
        source_commit=FRAMEWORK,
        plan=PLAN_SHA256,
        protocol="wan-cpu-calibration",
        run_dir=str(tmp_path),
        probes=str(probes),
        probe_digest=hashlib.sha256(probes.read_bytes()).hexdigest(),
        reservation=1 << 20,
        kept_bytes=50 << 20,
        config_digest=hashlib.sha256(script.read_bytes()).hexdigest(),
    )
    flags = tmp_path / "manifest.json"
    flags.write_text(json.dumps(config))
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
    )

    def invoke(args):
        out = subprocess.run(
            [sys.executable, str(launcher), *map(str, args)],
            capture_output=True,
            text=True,
            env=env,
        )
        assert out.returncode == 0, out.stderr
        return json.loads(out.stdout.splitlines()[-1])

    off = invoke([script])
    assert not off["imported"] and not (tmp_path / "signals").exists()
    on = invoke(["--internal-signals", flags, "--", script])
    assert on["imported"] and off["states"] == on["states"]
    [output] = [p for p in (tmp_path / "signals").iterdir() if zipfile.is_zipfile(p)]
    with zipfile.ZipFile(output) as z:
        key = next(k for k in z.namelist() if k.endswith("codec"))
        rows = json.loads(z.read(key))
    assert sum(n == "encoder.spatial" for n, _ in rows) == 4
    assert sorted({r["chunk"] for n, r in rows}) == [0, 1]


def test_refuse_before_entry(tmp_path):
    from openh_signals.runtime_launch import attach_worker

    manifest = tmp_path / "private.json"
    manifest.write_text(
        json.dumps(
            dict(
                plan=PLAN_SHA256,
                protocol="stage1",
                source_commit=FRAMEWORK,
                config_digest="0" * 64,
            )
        )
    )
    with pytest.raises(ValueError, match="unqualified"):
        with attach_worker(manifest, "module:unbound"):
            pytest.fail("unqualified worker started")


def test_pack_multi():
    # Two samples of unequal history; second sample's memory is not at pack start.
    n1, n2 = 2 + 4 + 3 + 1 + 24, 3 + 4 + 3 + 24
    text, vision, action = [], [], []
    for off, nt, n in [(0, 2, n1), (n1, 3, n2)]:
        text.extend(range(off, off + nt))
        vision.extend(range(off + nt, off + nt + 4))
        action.extend(range(off + nt + 4, off + n))
    p = NS(
        sample_lens=[n1, n2],
        sequence_length=n1 + n2,
        attn_modes=["causal", "full"] * 2,
        split_lens=[2, n1 - 2, 3, n2 - 3],
        text_ids=torch.tensor([8, 2, 6, 2, 7]),
        text_indexes=torch.tensor(text),
        vision=NS(
            sequence_indexes=torch.tensor(vision),
            token_shapes=[(2, 1, 2)] * 2,
            condition_mask=[torch.tensor([1, 0])] * 2,
        ),
        action=NS(
            sequence_indexes=torch.tensor(action),
            token_shapes=[(28,), (27,)],
            condition_mask=[
                torch.tensor([1] * 4 + [0] * 24),
                torch.tensor([1] * 3 + [0] * 24),
            ],
        ),
    )
    spans = {token_digest([8, 2]): [0, 1], token_digest([6, 2, 7]): [0, 1, 2]}
    groups, rows = label_pack(p, spans, memory_rows=3)
    assert [r["history"] for r in rows] == [1, 0]
    assert (
        groups.tolist()
        == [0, 1, 3, 3, 4, 4, 8, 9, 10, 6]
        + [7] * 24
        + [0, 1, 2, 3, 3, 4, 4, 8, 9, 10]
        + [7] * 24
    )
    with pytest.raises(ValueError, match="digest"):
        label_pack(p, {}, memory_rows=3)
    p.action.sequence_indexes[0] = p.text_indexes[0]
    with pytest.raises(ValueError, match="partition"):
        label_pack(p, spans, memory_rows=3)


def test_step_lifecycle():
    param = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.SGD([param], lr=0.1)
    callbacks = NS(on_before_optimizer_step=lambda *a, **kw: None)
    rt = Runtime({}, lambda *a: None)
    saved = []
    emitted = TrainSteps(
        rt, "suturebot", 2, lambda e, r: saved.append((e.step, len(r)))
    )
    trainer = NS(config=NS(trainer=NS(grad_accum_iter=2)), callbacks=callbacks)
    scaler = NS(get_scale=lambda: 1.0)

    def call(m, skip=False):
        def work():
            rt.emit("mock", {"stat": 0})
            if m and not skip:
                opt.step()
            return None, None, 0 if m else 1

        return emitted.step(
            work,
            (),
            {},
            dict(
                self=trainer,
                iteration=0,
                grad_accum_iter=m,
                optimizer=opt,
                grad_scaler=scaler,
                data={},
            ),
        )

    call(0)
    assert not saved
    call(1)
    assert saved == [(1, 2)]
    call(0)
    with pytest.raises(ValueError, match="no confirmed"):
        call(1, True)
    assert saved == [(1, 2)]


def test_native_trainer_boundary():
    from contextlib import nullcontext
    from cosmos_framework.trainer import ImaginaireTrainer
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import (
        AttentionBlock,
    )

    # No trainer construction, data iteration, model initialization from files or train() call.
    # Exercise the pinned trainer's actual microstep/optimizer methods on one in-memory leaf.
    class Toy(AttentionBlock):
        def training_step(self, data, iteration):
            y = self(data["x"])
            return {"pred": y}, y.square().mean()

        def on_after_backward(self):
            pass

        def on_before_optimizer_step(self, *a, **kw):
            pass

        def on_before_zero_grad(self, *a, **kw):
            pass

    cb = (
        "on_before_forward",
        "on_after_forward",
        "on_before_backward",
        "on_after_backward",
        "on_before_optimizer_step",
        "on_before_zero_grad",
    )

    def run(observe):
        torch.manual_seed(52)
        trainer = object.__new__(ImaginaireTrainer)
        trainer.callbacks = NS(**{key: (lambda *a, **kw: None) for key in cb})
        trainer.config = NS(
            trainer=NS(
                grad_accum_iter=2,
                distributed_parallelism="none",
                straggler_detection=NS(
                    analyze_forward=False,
                    analyze_backward=False,
                    analyze_optimizer=False,
                ),
            )
        )
        trainer.training_timer = lambda *a, **kw: nullcontext()
        trainer.straggler_detector = NS(profile_section=lambda *a, **kw: nullcontext())
        model = Toy(4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        saved = []
        states = []
        rt = Runtime({}, lambda *a: None)
        adapter = TrainSteps(rt, "suturebot", 2, lambda e, r: saved.append((e.step, r)))
        with adapter.install(FRAMEWORK) if observe else nullcontext():
            for j in range(4):
                batch = {"x": torch.randn(1, 4, 1, 2, 2)}
                out, loss, index = trainer.training_step(
                    model,
                    opt,
                    sched,
                    scaler,
                    batch,
                    iteration=j // 2,
                    grad_accum_iter=j % 2,
                )
                states.append(
                    [
                        out["pred"].detach(),
                        loss.detach(),
                        torch.get_rng_state(),
                        *[p.detach().clone() for p in model.parameters()],
                    ]
                )
        return saved, states

    off, baseline = run(False)
    on, observed = run(True)
    assert not off and [step for step, rows in on] == [1, 2]
    assert all(
        any(name == "grad.module" and row["l2"] > 0 for name, row in rows)
        for _, rows in on
    )
    for a, b in zip(baseline, observed, strict=True):
        for v, w in zip(a, b, strict=True):
            assert torch.equal(v, w)


def test_solver_routing():
    from openh_signals.runtime_eval import EvalCalls

    rt = Runtime({}, lambda *a: None)
    m = {
        "target_digest": "a" * 64,
        "mode": "policy",
        "chunk": 0,
        "seed": 3,
        "ablation": "real",
        "kernel": "synthetic",
        "reference": "r",
        "branches": {token_digest([3]): "positive", token_digest([4]): "negative"},
        "sentinel_first_seed_real": True,
    }
    route = EvalCalls(rt, lambda a, b: m)
    observed = []

    class Solver:
        def forward(self, velocity_fn, num_steps=5, seed=3):
            for i in range(num_steps):
                velocity_fn(torch.tensor(i))

    def score():
        def velocity(t):
            for token in (3, 4):
                route.velocity(
                    lambda: observed.append(rt.current.get()),
                    (),
                    {},
                    dict(text_tokens=[[token]], timestep=t),
                )

        Solver().forward(velocity)

    with wrap(Solver, "forward", rt.solver_call):
        route.generate(score, (), {}, dict(sample_args_list=[object()], data_batch={}))
    samples = [x for x in observed if x]
    assert [(x["solver_call"], x["branch"]) for x in samples] == [
        (i, b) for i in (0, 2, 4) for b in ("positive", "negative")
    ]
    assert [(x["solver_call"], x["branch"]) for x in samples if x["full_map"]] == [
        (2, "positive")
    ]
