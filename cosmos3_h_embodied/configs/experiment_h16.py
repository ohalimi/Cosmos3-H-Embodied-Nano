# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""H16 task-prompt recipe: Cosmos3-Nano with 16 action-history rows on the Open-H mixture.

Experiments:

* ``openh_h16_task_prompt``: Open-H roots available on Hugging Face, per-root caption contexts.
* ``openh_h16_task_prompt_reference``: the exact training mixture of the released checkpoint
  (55 roots, some not public) with the caption contexts it was trained with.

Both continue from a model checkpoint (``checkpoint.load_path``, a DCP directory; convert the
released weights with ``tools/convert_checkpoint_to_dcp.py``). Data paths come from
``OPENH_DATA_ROOT``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import cosmos_framework

from hydra.core.config_store import ConfigStore

from cosmos3_h_embodied.callbacks.action_dataloader_state import ActionDataLoaderStateCallback
from cosmos3_h_embodied.callbacks.nano_warm_start import InitEmaFromNano
from cosmos3_h_embodied.callbacks.packed_sigma_loss import PackedSigmaLossAnalysis
from cosmos3_h_embodied.callbacks.repro_record import ReproRecordCallback
from cosmos3_h_embodied.configs.openh_datasets import FAMILIES, build_dataset_entries
from cosmos3_h_embodied.data.dataloaders import InfiniteDataLoader
from cosmos3_h_embodied.data.joint_dataloader import HistoryMatchedJointDataLoader
from cosmos3_h_embodied.data.unified_dataset import wrap_dataset
from cosmos3_h_embodied.model.omni_mot_model import HistoryOmniMoTModel
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

MAX_SEQUENCE_LENGTH = 5040

# Experiment overrides for evaluating on the val/full split: video augmentation is training-only
# (the eval must condition on and score clean frames). cosmos-framework's dataset loading
# disables only caption dropout for these splits.
EVAL_DATASET_OVERRIDES = (
    "dataloader_train.dataloaders.action_data.dataloader.dataset.video_augment=null",
    "dataloader_train.dataloaders.action_data.dataloader.dataset.video_augment_cuda_families=[]",
)


def h16_model_config() -> dict:
    """Cosmos3-Nano model config with the H16 recipe's changes."""
    cfg = copy.deepcopy(NANO_MODEL_CONFIG)
    cfg["resolution"] = "256"
    cfg["max_num_tokens_after_packing"] = -1
    cfg["activation_checkpointing"] = dict(mode="full", save_ops_regex=["fmha"])
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["diffusion_expert_config"]["max_vae_latent_side_after_patchify"] = 52
    cfg["parallelism"]["data_parallel_shard_degree"] = 32
    # Gradients were reduced in fp32 (CF's default reduces in bf16).
    cfg["parallelism"]["fsdp_reduce_dtype"] = "float32"
    rf = cfg["rectified_flow_training_config"]
    rf["loss_scale"] = 10.0
    rf["image_loss_scale"] = None
    rf["action_loss_weight"] = 10.0
    rf["normalize_loss_by_active"] = False
    # 25-frame clips (24 + 1) are encoded unpadded.
    cfg["tokenizer"]["encode_exact_durations"] = [25]
    cfg["tokenizer"]["encode_chunk_frames"] = {"256": 68, "480": 24, "720": 12}
    # CF's path is relative to its repository root; resolve it so any working directory works.
    base_config = cfg["vlm_config"]["model_instance"]["config"]["base_config"]
    base_config["json_file"] = str(Path(cosmos_framework.__file__).resolve().parent.parent / base_config["json_file"])
    return cfg


def h16_dataloader(
    *,
    include_nonpublic: bool,
    caption_prefix_mode: str,
    families: list[str] | None = None,
    shard_across_workers: bool = True,
    num_workers: int = 12,
    prefetch_factor: int = 16,
    augment: bool = True,
    **dataset_overrides,
) -> LazyDict:
    """The action dataloader of the recipe. ``augment=False`` (validation) turns off video augmentation and
    caption dropout."""
    families = build_dataset_entries(
        include_nonpublic=include_nonpublic,
        caption_prefix_mode=caption_prefix_mode,
        families=families,
        **dataset_overrides,
    )
    family_names = [entry.name for entry in families]
    dataset = L(wrap_dataset)(
        list_of_datasets=families,
        action_channel_masking=True,
        append_duration_fps_timestamps=True,
        append_idle_frames=False,
        append_viewpoint_info=False,
        append_resolution_info=True,
        caption_key="ai_caption",
        cfg_dropout_rate=0.1 if augment else 0.0,
        format_prompt_as_json=False,
        idle_frames_dropout=0.05,
        keep_aspect_ratio=True,
        video_augment=dict(enabled=True, erasing_prob=0.0, frame_chunk_size=1) if augment else None,
        video_augment_cuda_families=family_names if augment else [],
        max_action_dim="${model.config.max_action_dim}",
        pad_keys=None,
        resolution=None,
        # Rank-based family mixing (each rank serves one family). Set False for per-sample
        # mixing when there are fewer ranks than families.
        shard_across_workers=shard_across_workers,
        text_token_key="text_token_ids",
        tokenizer_config="${model.config.vlm_config.tokenizer}",
        video_temporal_downsample=4,
    )
    return L(HistoryMatchedJointDataLoader)(
        dataloaders=dict(
            action_data=dict(
                dataloader=L(InfiniteDataLoader)(
                    dataset=dataset,
                    batch_size=1,
                    in_order=True,
                    multiprocessing_context="spawn",
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=True,
                    pin_memory=True,
                    seed=42,
                    use_deterministic_seed=True,
                ),
                ratio=1,
            )
        ),
        audio_sample_rate=48000,
        max_samples_per_batch=None,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        patch_spatial=2,
        seed=42,
        sound_latent_fps=0,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        # Spawn workers after the checkpoint (and the dataloader resume state) is loaded.
        lazy_initialize_child_iterators=True,
        prewarm=True,
        chunk_length=24,
        max_history_actions=16,
    )


# Learning-rate schedules (factors of optimizer.lr = 2e-4).
# "continue": constant 1e-5 (factor 0.05) after a 100-step warm-up -- the rate the released
#   checkpoint finished its training at; the default when training on from it.
# "as_trained": the schedule of the training run that produced the checkpoint (linear decay
#   from 0.4 to 0.05 over 50,000 iterations); used by the reference experiment.
LR_SCHEDULES = {
    "continue": dict(cycle_lengths=[10_000_000], f_max=[0.05], f_min=[0.05], f_start=[1.0e-06], warm_up_steps=[100]),
    "as_trained": dict(cycle_lengths=[50000], f_max=[0.4], f_min=[0.05], f_start=[1.0e-06], warm_up_steps=[100]),
}


def _experiment(
    name: str,
    *,
    include_nonpublic: bool,
    caption_prefix_mode: str,
    lr_schedule: str = "continue",
    dataloader_val: LazyDict | None = None,
    callbacks: dict | None = None,
    **dataloader_kwargs,
) -> LazyDict:
    return LazyDict(
        dict(
            defaults=[
                {"override /model": "mot_fsdp"},
                {"override /data_train": None},
                {"override /data_val": None},
                {"override /optimizer": "fusedadamw"},
                {"override /scheduler": "lambdalinear"},
                {"override /checkpoint": "local"},
                {"override /callbacks": ["basic", "optimization", "job_monitor"]},
                {"override /ema": "power"},
                {"override /tokenizer": "wan2pt2_tokenizer"},
                {"override /sound_tokenizer": None},
                {"override /vlm_config": None},
                {"override /ckpt_type": "dcp"},
                "_self_",
            ],
            job=dict(project="cosmos3_h_embodied", group="openh_h16", name=name, wandb_mode="disabled"),
            model=dict(
                _target_=HistoryOmniMoTModel,
                normalize_action_loss_by_active=True,
                config=h16_model_config(),
            ),
            optimizer=dict(
                betas=[0.9, 0.99],
                eps=1.0e-08,
                fused=True,
                keys_to_select=[
                    "moe_gen",
                    "time_embedder",
                    "vae2llm",
                    "llm2vae",
                    "action2llm",
                    "llm2action",
                    "action_modality_embed",
                ],
                lr=2.0e-04,
                lr_multipliers={"action2llm": 5.0, "llm2action": 5.0, "action_modality_embed": 5.0},
                optimizer_type="FusedAdam",
                weight_decay=0.05,
            ),
            scheduler=dict(lr_scheduler_type="LambdaLinear", verbosity_interval=0, **LR_SCHEDULES[lr_schedule]),
            trainer=dict(
                distributed_parallelism="fsdp",
                grad_accum_iter=2,
                logging_iter=10,
                max_iter=50000,
                run_validation=False,
                run_validation_on_start=False,
                save_zero_checkpoint=False,
                seed=0,
                timeout_period=999999999,
                compile_config=dict(recompile_limit=100, use_duck_shape=False),
                cudnn=dict(benchmark=True, deterministic=False),
                ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
                grad_scaler_args=dict(enabled=False),
                callbacks=dict(
                    action_dataloader_state=L(ActionDataLoaderStateCallback)(),
                    grad_clip=dict(clip_norm=1.0, force_finite=True),
                    manual_gc=dict(every_n=200),
                    skip_nan_step=dict(max_consecutive_nan=100),
                    training_stats=dict(log_freq=10),
                    **(callbacks or {}),
                ),
            ),
            checkpoint=dict(
                load_path="???",
                load_training_state=False,
                strict_resume=True,
                keys_to_skip_loading=[],
                load_ema_to_reg=False,
                save_iter=100,
                broadcast_via_filesystem=True,
                type=dict(disable_async=True),
            ),
            model_parallel=dict(cpu_offloading_activations=False),
            dataloader_train=h16_dataloader(
                include_nonpublic=include_nonpublic, caption_prefix_mode=caption_prefix_mode, **dataloader_kwargs
            ),
            dataloader_val=dataloader_val,
            upload_reproducible_setup=False,
        ),
        flags={"allow_objects": True},
    )


openh_h16_task_prompt = _experiment("openh_h16_task_prompt", include_nonpublic=False, caption_prefix_mode="fixed")
openh_h16_task_prompt_reference = _experiment(
    "openh_h16_task_prompt_reference",
    include_nonpublic=True,
    caption_prefix_mode="legacy",
    lr_schedule="as_trained",
)
# Smoke test with one GPU on two families whose roots need only the v2.1 -> v3.0 conversion
# (e.g. samples from tools/download_openh_samples.py): per-sample family mixing (fewer ranks
# than families), a window cap, and few workers. Combine with COSMOS_SMOKE=1 (2-layer model).
openh_h16_task_prompt_smoke = _experiment(
    "openh_h16_task_prompt_smoke",
    include_nonpublic=False,
    caption_prefix_mode="fixed",
    families=["dvrk_obuda", "dvrk_ucsd"],
    shard_across_workers=False,
    num_workers=2,
    prefetch_factor=2,
    max_windows=64,
)

cs = ConfigStore.instance()
for _name, _node in (
    ("openh_h16_task_prompt", openh_h16_task_prompt),
    ("openh_h16_task_prompt_reference", openh_h16_task_prompt_reference),
    ("openh_h16_task_prompt_smoke", openh_h16_task_prompt_smoke),
):
    cs.store(group="experiment", package="_global_", name=_name, node=_node)


def register_experiment(
    name: str, *, include_nonpublic: bool = False, caption_prefix_mode: str = "fixed", **kwargs
) -> LazyDict:
    """Build an experiment like ``openh_h16_task_prompt`` with changes, and register it as ``name``.

    ``kwargs``: ``lr_schedule`` (a key of ``LR_SCHEDULES``), ``dataloader_val`` (a validation dataloader,
    default none), ``callbacks`` (extra trainer callbacks), the options of :func:`h16_dataloader`
    (``families``, ``shard_across_workers``, ``num_workers``, ...), and dataset options applied to
    every family (e.g. ``mode="forward_dynamics"``, ``joint_mode_weights``, ``max_windows``). Example::

        register_experiment("my_fd", families=["my_family"], shard_across_workers=False, mode="forward_dynamics")
    """
    node = _experiment(name, include_nonpublic=include_nonpublic, caption_prefix_mode=caption_prefix_mode, **kwargs)
    cs.store(group="experiment", package="_global_", name=name, node=node)
    return node


# SutureBot memory experiments on the opt-in ``suturebot_oshri`` family. The launch sets OPENH_DATA_ROOT to the
# directory that holds the family's roots and REPRO_DIR to the run's reproducibility record. Validation uses the
# family's val split (the split seed and ratio of training) without augmentation.
_SUTUREBOT = dict(
    include_nonpublic=True, caption_prefix_mode="fixed", families=["suturebot_oshri"], shard_across_workers=False
)
_SUTUREBOT_FAMILY = next(f for f in FAMILIES if f.name == "suturebot_oshri")
for _name in ("suturebot-h16", "suturebot-h16-smoke"):
    register_experiment(
        _name,
        **_SUTUREBOT,
        dataloader_val=h16_dataloader(**_SUTUREBOT, augment=False, num_workers=4, prefetch_factor=2, split="val"),
        callbacks=dict(
            repro=L(ReproRecordCallback)(
                normalizer=_SUTUREBOT_FAMILY.kwargs["normalizer_filename"],
                data_roots=[r.path for r in _SUTUREBOT_FAMILY.roots],
            )
        ),
    )

# Train from Nano rather than continuing from the released embodied weights. Keep
# the two families' native video keys, caption contexts, ratios and normalizers.
_SUTUREBOT_ROOTS = (
    "jhu/imerse/wound_closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled",
    "jhu/hf_suturebot",
)


def _two_root_experiment(name: str, *, caption_prefix_mode: str, grad_accum_iter: int) -> LazyDict:
    roots = dict(
        families=["dvrk_jhu", "dvrk_suturebot_mono"], root_allowlist=_SUTUREBOT_ROOTS,
        include_nonpublic=True, caption_prefix_mode=caption_prefix_mode,
    )
    node = _experiment(
        name, **roots, lr_schedule="as_trained",
        dataloader_val=h16_dataloader(**roots, augment=False, split="val", num_workers=4, prefetch_factor=2),
        callbacks=dict(
            nano_ema=L(InitEmaFromNano)(),
            sigma_loss_analysis=dict(_target_=PackedSigmaLossAnalysis),
            repro=L(ReproRecordCallback)(
                normalizer=[f.kwargs["normalizer_filename"] for f in FAMILIES if f.name in roots["families"]],
                data_roots=list(_SUTUREBOT_ROOTS),
            ),
        ),
    )
    node.checkpoint.keys_to_skip_loading = ["net_ema."]
    node.trainer.grad_accum_iter = grad_accum_iter
    node.trainer.run_validation = True
    node.trainer.run_validation_on_start = True
    cs.store(group="experiment", package="_global_", name=name, node=node)
    return node


nano_h16_suturebot = _two_root_experiment("nano-h16-suturebot", caption_prefix_mode="fixed", grad_accum_iter=1)
nano_h16_suturebot_ref32 = _two_root_experiment("nano-h16-suturebot-ref32", caption_prefix_mode="legacy", grad_accum_iter=2)
