"""CPU-only synthetic calibration. Select the native dependency with
OPENH_SIGNALS_OBJECTIVE_BACKEND=native; otherwise use the public adapter.
"""

from contextlib import ExitStack, nullcontext
import os
import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from openh_signals.objective import (
    FlowSample,
    observe_flow,
    observe_preclip,
    watch_group,
)

NATIVE = os.environ.get("OPENH_SIGNALS_OBJECTIVE_BACKEND") == "native"
if NATIVE:
    from cosmos3._src.vfm.models import omni_mot_model as omni
    from cosmos3._src.vfm.models.mot.domain_aware_linear import DomainAwareLinear
    from cosmos3._src.vfm.diffusion.rectified_flow import RectifiedFlow
    from cosmos3._src.imaginaire.utils.callback import CallBackGroup
else:
    from cosmos_framework.model.generator import omni_mot_model as omni
    from cosmos_framework.model.generator.mot.domain_aware_linear import (
        DomainAwareLinear,
    )
    from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow
    from cosmos_framework.utils.callback import CallBackGroup
    from cosmos3_h_embodied.model.omni_mot_model import _active_normalized_action_loss


def exercise(enabled):
    # This is a unit test of one optimizer window, repeated twice. Neither an
    # OmniMoTModel nor a trainer, dataset, processor, or save is constructed.
    random.seed(88)
    np.random.seed(88)
    torch.manual_seed(88)
    model = DomainAwareLinear(3, 4, num_domains=2)
    model.bias.weight.requires_grad_(False)
    model.unused = torch.nn.Parameter(torch.tensor(1.0))
    rf = RectifiedFlow(None)
    cb = CallBackGroup(SimpleNamespace(trainer=SimpleNamespace(callbacks={})), None)
    scale = torch.amp.GradScaler("cpu", init_scale=8.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.1)
    baseline = []
    events = []
    loss_checks = []
    schedules = torch.tensor([[0.25], [0.75]])

    def describe(args):
        modality = "action" if args.get("raw_action_dim") is not None else "vision"
        return [
            FlowSample(modality, "synthetic", mode, schedules[i], 3.0, i)
            for i, mode in enumerate(modes)
        ]

    def emit(key, row):
        events.append((key, row))

    original = omni.compute_flow_matching_loss
    with ExitStack() as stack:
        if enabled:
            stack.enter_context(
                observe_flow(omni, type(rf.train_time_weight), describe, emit)
            )
            stack.enter_context(observe_preclip(cb, lambda s: s in (1, 2), emit))
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            for micro in range(2):
                modes = ("policy", "inverse") if micro == 0 else ("forward", "forward")
                x = torch.randn(2, 3, 3)
                act = model(x, torch.arange(2))
                if enabled:
                    watch_group(
                        act,
                        torch.tensor([[1, 2, 2], [2, 2, -1]]),
                        "action",
                        scale.get_scale(),
                        emit,
                        accumulation=2,
                    )
                kw = dict(
                    rectified_flow=rf,
                    timesteps=schedules * 1000,
                    tensor_kwargs_fp32={"device": "cpu"},
                )
                action_target = list(torch.randn_like(act))
                action_mask = (
                    [torch.tensor([[1.0], [0.0], [0.0]]), torch.zeros(3, 1)]
                    if micro == 0
                    else [torch.ones(3, 1)] * 2
                )
                native_active = NATIVE
                adapter = (
                    nullcontext() if NATIVE else _active_normalized_action_loss(rf)
                )
                with adapter:
                    la, _ = omni.compute_flow_matching_loss(
                        pred=list(act),
                        target=action_target,
                        condition_mask=action_mask,
                        raw_action_dim=[2, 3],
                        normalize_by_active=native_active,
                        has_valid_tokens=(micro == 0),
                        **kw,
                    )
                vis = act.transpose(1, 2)[..., None, None]
                vision_masks = [
                    torch.tensor([1.0, 0.0, 0.0]).reshape(3, 1, 1),
                    torch.full((3, 1, 1), float(micro == 0)),
                ]
                lv, _ = omni.compute_flow_matching_loss(
                    pred=list(vis),
                    target=list(torch.randn_like(vis)),
                    condition_mask=vision_masks,
                    normalize_by_active=False,
                    has_valid_tokens=True,
                    **kw,
                )
                loss = 3.0 * (la + lv)
                loss_checks.append(float(loss.detach()))
                baseline.extend(
                    [
                        (f"prediction/{step}/{micro}", act.detach().clone()),
                        (f"loss/{step}/{micro}", loss.detach().clone()),
                    ]
                )
                scale.scale(loss / 2).backward()
            baseline.append((f"preclip/{step}", model.fc.weight.grad.clone()))
            cb.on_before_optimizer_step(model, opt, None, scale, iteration=step)
            scale.step(opt)
            scale.update()
            for name, p in model.named_parameters():
                baseline.append((f"parameter/{step}/{name}", p.detach().clone()))
            baseline.append((f"rng/{step}", torch.random.get_rng_state()))
    assert omni.compute_flow_matching_loss is original
    return baseline, events, loss_checks, (random.getstate(), np.random.get_state())


def _shard_check(rank, path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard, Replicate
    from openh_signals.objective import logical_module_gradients

    dist.init_process_group(
        "gloo", init_method="file://" + path, rank=rank, world_size=2
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        m = torch.nn.Module()
        for name, placement in [("sharded", Shard(0)), ("replicated", Replicate())]:
            d = distribute_tensor(
                torch.arange(5, dtype=torch.float32), mesh, [placement]
            )
            m.register_parameter(name, torch.nn.Parameter(d))
            getattr(m, name).grad = d.detach() * 4.0
        rows = []
        logical_module_gradients(m, 4.0, lambda _, row: rows.append(row))
        assert len(rows) == (2 if rank == 0 else 0)
        assert all(
            r["grad_numel"] == 5 and abs(r["square_sum"] - 30.0) < 1e-5 for r in rows
        )
    finally:
        dist.destroy_process_group()


class ObjectiveTest(unittest.TestCase):
    def test_replica_not_counted_twice_or_uneven_shard_missed(self):
        import tempfile
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as d:
            mp.start_processes(
                _shard_check, (d + "/rendezvous",), nprocs=2, start_method="fork"
            )

    def test_tiny_optimizer_bitwise_and_numerical(self):
        off, off_rows, _, off_rng = exercise(False)
        on, rows, losses, on_rng = exercise(True)
        self.assertFalse(off_rows)
        for (name, before), (tag, after) in zip(off, on, strict=True):
            self.assertEqual(name, tag)
            self.assertTrue(
                torch.equal(
                    before.reshape(-1).view(torch.uint8),
                    after.reshape(-1).view(torch.uint8),
                ),
                name,
            )
        self.assertEqual(off_rng[0], on_rng[0])
        np.testing.assert_equal(off_rng[1], on_rng[1])
        flow = [r for k, r in rows if k == "flow.parts"]
        self.assertEqual(len(flow), 16)
        self.assertEqual([r["valid_count"] for r in flow[:4]], [4, 9, 8, 0])
        self.assertEqual([r["training_denominator"] for r in flow[:4]], [4, 9, 12, 12])
        self.assertEqual(flow[0]["per_row_valid"], [0, 2, 2])
        self.assertEqual(flow[0]["time_weights"], [1.0])
        self.assertTrue(flow[4]["dummy"])
        self.assertIsNone(flow[4]["active_mse"])
        for i, loss in enumerate(losses):
            self.assertAlmostEqual(
                sum(r["objective_contribution"] for r in flow[i * 4 : i * 4 + 4]),
                loss,
                places=5,
            )
        groups = [r for k, r in rows if k == "grad.group"]
        self.assertEqual([r["rows"] for r in groups], [1, 4] * 4)
        self.assertEqual([r["accumulation"] for r in groups], [2] * 8)
        modules = [r for k, r in rows if k == "grad.module"]
        self.assertEqual(
            [r["frozen_numel"] for r in modules if r["name"] == "bias"], [8, 8]
        )
        self.assertEqual(
            [r["missing_numel"] for r in modules if r["name"] == "unused"], [1, 1]
        )
        preclip = [g for tag, g in on if tag.startswith("preclip/")]
        for r, g in zip(
            (r for r in modules if r["name"] == "fc"), preclip, strict=True
        ):
            self.assertAlmostEqual(
                r["square_sum"], float((g.double() / 8).square().sum()), places=5
            )

    def test_group_excludes_pad_and_hook_does_not_modify_grad(self):
        x = torch.ones(3, 2, requires_grad=True)
        out = []
        watch_group(
            x, torch.tensor([2, -1, 2]), "residual", 4.0, lambda k, r: out.append(r)
        )
        grad = torch.tensor([[12.0, 16.0], [80.0, 80.0], [0.0, 4.0]])
        x.backward(grad)
        self.assertTrue(torch.equal(grad, x.grad))
        self.assertEqual(out[0]["group"], 2)
        self.assertEqual(out[0]["count"], 4)
        self.assertAlmostEqual(out[0]["square_sum"], 26.0, places=5)
        with self.assertRaises(ValueError):
            watch_group(x, torch.tensor([12, 0, 1]), "bad", 4.0, print)

    @unittest.skipIf(NATIVE, "channel mask belongs to public helper")
    def test_noncontiguous_valid_channels(self):
        rf = RectifiedFlow(None)
        rows = []
        with observe_flow(
            omni,
            type(rf.train_time_weight),
            lambda a: [FlowSample("action", "synthetic", "policy", 0.5, 1.0, 0)],
            lambda k, r: rows.append(r),
        ):
            loss, _ = omni.compute_flow_matching_loss(
                pred=[torch.ones(3, 5)],
                target=[torch.zeros(3, 5)],
                condition_mask=[torch.tensor([[1.0], [0.0], [0.0]])],
                raw_action_dim=[4],
                action_valid_mask=[torch.tensor([0, 1, 0, 1, 1])],
                timesteps=torch.ones(1, 3),
                has_valid_tokens=True,
                tensor_kwargs_fp32={},
                rectified_flow=rf,
            )
        self.assertEqual(rows[0]["all_count"], 6)
        self.assertEqual(rows[0]["valid_count"], 4)
        self.assertEqual(rows[0]["time_weights"], [1.0, 1.0, 1.0])
        self.assertEqual(rows[0]["weighted_item_loss"], float(loss))


if __name__ == "__main__":
    unittest.main(verbosity=2)
