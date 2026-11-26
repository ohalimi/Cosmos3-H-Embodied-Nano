# Reproduction ledger: SutureBot memory

Job numbers below are Slurm allocations, not training steps. Slide titles identify the result across deck renumberings. Script digests are SHA-256 of the submitted copy unless explicitly labeled **source**. Commands are the recorded application invocations with local paths and rendezvous hosts replaced; they do not contain the launch wrappers.

- **Data (seven roots):** `hf_suturebot`, `ood`, `suture_bot_failure`, `cosmos_fail_filtered`, `cosmos_throw_fail_demo`, `cosmos_knot_fail_demo`, `suturebot_act_throw_eval`, in the `suturebot_oshri` family.
- **Split:** the pinned loader's `split_episode_ids` permutes each root's episode indices with seed 42, takes the first `round(0.05 × total_episodes)` for validation and the remainder for training, before filtering. The committed exclusions overlay has SHA-256 `c2eff8a6c6113d32b8db00b0910b978a8ed4feff346fd5c9b8541955892d27b7`; it is **not** a split manifest.
- **Command names:** `CODE` is the pinned checkout (plus any recorded diff); `DATA` contains the initial checkpoint; `OUTPUT_ROOT` is the training output base. `RUN_DIR` is the per-token output folder, `JOB_DIR` its nested training job folder, and `HOST` the rendezvous host. For evaluations, `MODEL` is the named checkpoint's `model` directory and `EVAL_OUT` the output folder for that checkpoint.
- **W&B entity/project:** `holoscan-advanced-workflow-team/cosmos3_suturebot_memory`. A source training link is not an evaluation run.

## suturebot-h16-smoke — pipeline smoke test

- Slide: “Smoke run checks the pipeline, not the model” (slide 21 in the smoke packet).
- State: **finished at step 2**. Job 418038: COMPLETED `0:0`.
- Branch: `ohalimi/suturebot-memory`. Code commit: `63455e9854f6dfd64e313c2a18108409d90dde85`. Framework commit: `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`.
- W&B: [71kzbz0q](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/71kzbz0q).
- Launch script SHA-256: `29faf81307da904f0aaf6cf5292006855e1dd6e15ad957e31b0e8e1b70a3ca4f`.
- Data roots: the seven listed above. Split manifest SHA-256: missing — no frozen manifest in the run’s config, repro directory or W&B record; the loader computes the split.
- Trainer seed: `0`. Start checkpoint: `h16_50000_dcp` (weights only).
- Dirty diff: validation step in `omni_mot_model.py` and new `packed_batch_val.py`; full patch in W&B `repro/diff` (run-folder patch SHA-256 `924742391a262267715a2ebf10eb43f35b36a9e7161db056c2b2757833797b98`). To reconstruct the effective tree, apply that patch to the recorded base, **or** take just those two files from `7e8a0e8f7f866344175267a417526c3cafedefba`. Do not substitute the entire later commit.

All three attempts used the same W&B ID and initial weights, not a resumed optimizer checkpoint. Slurm reports zero scheduler restarts on each allocation.

| Job | State | Submitted script SHA-256 | Code at that attempt |
|---|---|---|---|
| 417610 | FAILED `1:0` | `a31c6ff2f85611f43ac2133e0f8b23caff853e24bca7847acee8d6d4622c432d` | Clean base; validation returned no training pair. |
| 417987 | FAILED `1:0` | `a31c6ff2f85611f43ac2133e0f8b23caff853e24bca7847acee8d6d4622c432d` | Validation-step fix only; patch SHA-256 `6da4db6b528372868b1c796a1710231326c7dce6b9cfe7481abd37708d411c28`. Packed validation then rejected multiple dataset names. |
| 418038 | COMPLETED `0:0` | `29faf81307da904f0aaf6cf5292006855e1dd6e15ad957e31b0e8e1b70a3ca4f` | Both fixes; checkpoint `iter_000000002`. |


Recorded config export and trainer command for 418038 (`COSMOS_SMOKE=1` for both commands). For 417610 and 417987 the same commands used their own rendezvous ID/host and omitted the `trainer.callbacks.wandb_val._target_` override:

```sh
python -m cosmos_framework.scripts.export_config -o $RUN_DIR/exported_config.yaml --config-file \
  cosmos3_h_embodied/configs/config.py --experiment suturebot-h16-smoke
bash $CODE/tools/train_with_prune.sh --checkpoint-dir $JOB_DIR/checkpoints --keep-last 2 --keep-every 0 \
  --milestones 500,1000,2000 --save-iter 100 -- torchrun --nnodes=1 --nproc_per_node=1 --rdzv-backend=c10d \
  --rdzv-endpoint=$HOST:29500 --rdzv-id=418038 --rdzv-conf=timeout=120 -m cosmos_framework.scripts._train \
  --config-file $RUN_DIR/exported_config.yaml -o $OUTPUT_ROOT --config-overrides \
  checkpoint.load_path=$DATA/Cosmos3-H-Embodied-Nano/h16_50000_dcp checkpoint.load_training_state=false \
  checkpoint.save_iter=100 job.project=cosmos3_suturebot_memory job.group=suturebot-memory \
  job.name=suturebot-h16-smoke job.wandb_mode=online model.config.parallelism.data_parallel_shard_degree=1 \
  model.config.compile.enabled=false trainer.run_validation=true trainer.run_validation_on_start=true \
  trainer.callbacks.wandb_val._target_=cosmos3_h_embodied.callbacks.packed_batch_val.PackedBatchValLog \
  trainer.validation_iter=100 trainer.max_val_iter=4 trainer.callbacks.norm_monitor.every_n=100 \
  trainer.callbacks.sigma_loss_analysis.every_n=250 trainer.callbacks.sigma_loss_analysis.every_n_viz=250 \
  trainer.callbacks.device_monitor.every_n=50 trainer.callbacks.sampled_media.enabled=true \
  trainer.callbacks.sampled_media.output_uri=$JOB_DIR/sampled_media.lance \
  trainer.callbacks.every_n_sample_reg._target_=cosmos_framework.callbacks.every_n_draw_sample.EveryNDrawSample \
  trainer.callbacks.every_n_sample_reg.every_n=250 \
  trainer.callbacks.every_n_sample_reg.do_x0_prediction=false \
  trainer.callbacks.every_n_sample_ema._target_=cosmos_framework.callbacks.every_n_draw_sample.EveryNDrawSample \
  trainer.callbacks.every_n_sample_ema.every_n=250 trainer.callbacks.every_n_sample_ema.is_ema=true \
  trainer.callbacks.every_n_sample_ema.do_x0_prediction=false trainer.callbacks.norm_monitor.every_n=1 \
  trainer.callbacks.sigma_loss_analysis.every_n=1 trainer.callbacks.every_n_sample_reg.every_n=1 \
  trainer.callbacks.every_n_sample_ema.every_n=1
```

## suturebot-h16 — arm A training

- Slide: “Arm A vs t8funnhc: public recipe on SutureBot” (slide 21 in the intermediate-checkpoint assignment).
- State: **finished at step 2,000**. Job 418157: COMPLETED `0:0`.
- Branch: `ohalimi/suturebot-memory`. Code commit: `63455e9854f6dfd64e313c2a18108409d90dde85`. Framework commit: `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`.
- W&B: [gzluhczu](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/gzluhczu).
- Launch script SHA-256: `d1f6ec3ac3e0cf8b5b4b2a0b5d020937520b7f4683fa2c9f4fb44e08993da525`.
- Data roots: the seven listed above. Split manifest SHA-256: missing — no frozen manifest in the run’s config, repro directory or W&B record; the loader computes the split.
- Trainer seed: `0`. Start checkpoint: `h16_50000_dcp` (weights only).
- Dirty diff: validation step in `omni_mot_model.py` and new `packed_batch_val.py`; full patch in W&B `repro/diff` (run-folder patch SHA-256 `924742391a262267715a2ebf10eb43f35b36a9e7161db056c2b2757833797b98`). To reconstruct the effective tree, apply that patch to the recorded base, **or** take just those two files from `7e8a0e8f7f866344175267a417526c3cafedefba`. Do not substitute the entire later commit.

Recorded config export and trainer command (`COSMOS_SMOKE` unset):

```sh
python -m cosmos_framework.scripts.export_config -o $RUN_DIR/exported_config.yaml --config-file \
  cosmos3_h_embodied/configs/config.py --experiment suturebot-h16
bash $CODE/tools/train_with_prune.sh --checkpoint-dir $JOB_DIR/checkpoints --keep-last 2 --keep-every 0 \
  --milestones 500,1000,2000 --save-iter 100 -- torchrun --nnodes=8 --nproc_per_node=4 --rdzv-backend=c10d \
  --rdzv-endpoint=$HOST:29500 --rdzv-id=418157 --rdzv-conf=timeout=120 -m cosmos_framework.scripts._train \
  --config-file $RUN_DIR/exported_config.yaml -o $OUTPUT_ROOT --config-overrides \
  checkpoint.load_path=$DATA/Cosmos3-H-Embodied-Nano/h16_50000_dcp checkpoint.load_training_state=false \
  checkpoint.save_iter=100 job.project=cosmos3_suturebot_memory job.group=suturebot-memory \
  job.name=suturebot-h16 job.wandb_mode=online model.config.parallelism.data_parallel_shard_degree=32 \
  model.config.compile.enabled=false trainer.run_validation=true trainer.run_validation_on_start=true \
  trainer.callbacks.wandb_val._target_=cosmos3_h_embodied.callbacks.packed_batch_val.PackedBatchValLog \
  trainer.validation_iter=100 trainer.max_val_iter=3 trainer.callbacks.norm_monitor.every_n=100 \
  trainer.callbacks.sigma_loss_analysis.every_n=250 trainer.callbacks.sigma_loss_analysis.every_n_viz=250 \
  trainer.callbacks.device_monitor.every_n=50 trainer.callbacks.sampled_media.enabled=true \
  trainer.callbacks.sampled_media.output_uri=$JOB_DIR/sampled_media.lance \
  trainer.callbacks.every_n_sample_reg._target_=cosmos_framework.callbacks.every_n_draw_sample.EveryNDrawSample \
  trainer.callbacks.every_n_sample_reg.every_n=250 \
  trainer.callbacks.every_n_sample_reg.do_x0_prediction=false \
  trainer.callbacks.every_n_sample_ema._target_=cosmos_framework.callbacks.every_n_draw_sample.EveryNDrawSample \
  trainer.callbacks.every_n_sample_ema.every_n=250 trainer.callbacks.every_n_sample_ema.is_ema=true \
  trainer.callbacks.every_n_sample_ema.do_x0_prediction=false trainer.max_iter=2000
```

## suturebot-released-h16 — released-checkpoint evaluation

- Slide: “Step 0: score the released model, no training” (slide 20 in the step-zero packet); step-zero reference for “Arm A vs t8funnhc: public recipe on SutureBot” (slide 21 in the intermediate-checkpoint assignment).
- State: **finished at fine-tuning step 0 (released weights)**. Job 417609: COMPLETED `0:0`.
- Branch: `ohalimi/suturebot-memory`. Code commit: `63455e9854f6dfd64e313c2a18108409d90dde85`. Framework commit: `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`.
- Evaluation W&B ID and URL: missing — the project run listing and evaluation folder contain no evaluation run. The input was the release, not a training run from this project.
- Launch script SHA-256: `f6c6f635dd5156ec3280a9ceb0f453564eb789ba6b5ae9da5ba5a3120413e292`.
- Start checkpoint (`MODEL`), with SHA-256 of its metadata: `h16_50000_dcp/model` (`14b2c97c9a8db8b3b55a47611931995e3e6cee1dc5394221e80a1c10e1dc3aca`).
- Data roots: the seven listed above, validation only. Split manifest SHA-256: missing — no frozen split in the evaluation plan, resolved config, or evaluator output.
- Dirty diff: missing — the evaluation plan pins the code revision but neither the plan nor the evaluation log records a launch-time diff.
- Protocol per checkpoint: joint forward dynamics, inverse dynamics and policy, **32 clips per mode**, seed `20260908`. Forward rollouts: up to three validation episodes per root, **16 windows × 2 seeds = 32 rollouts**, base seed `0` (recorded seeds `0` and `1000`), three chunks each. Both evaluators: **35 sampling steps, guidance 6.0**. Counts are from `metrics_aggregate.json` and `summary.json`.

Recorded evaluation commands:

```sh
python -m cosmos3_h_embodied.scripts.eval --no-guardrails --checkpoint-path $MODEL --config-file \
  cosmos3_h_embodied/configs/config.py --experiment suturebot-h16 --dataset.dataset suturebot_oshri \
  --num-steps 35 --guidance 6.0 --dataset.model-mode joint --dataset.dataset-split val \
  --dataset.num-samples 32 --seed 20260908 -o $EVAL_OUT/eval_joint
python -m cosmos3_h_embodied.scripts.quant_eval --no-guardrails --checkpoint-path $MODEL --config-file \
  cosmos3_h_embodied/configs/config.py --experiment suturebot-h16 --dataset.dataset suturebot_oshri \
  --num-steps 35 --guidance 6.0 --dataset.dataset-split val --episodes-per-root 3 --num-seeds 2 --seed 0 \
  --num-chunks 3 --save-videos -o $EVAL_OUT/quant_eval
```

## suturebot-h16-eval — arm A final-checkpoint evaluation

- Slide: “Arm A vs t8funnhc: public recipe on SutureBot” (slide 21 in the intermediate-checkpoint assignment); also “One held-out episode at step 0 and step 2,000”.
- State: **finished at step 2,000**. Job 431929: COMPLETED `0:0`.
- Branch: `ohalimi/suturebot-memory`. Code commit: `63455e9854f6dfd64e313c2a18108409d90dde85`. Framework commit: `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`.
- Evaluation W&B ID and URL: missing — the project run listing and evaluation folder contain no evaluation run. Source training run: [gzluhczu](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/gzluhczu) (checkpoint only).
- Launch script SHA-256: `ab60035a51ce5e92ca171d49a9dbd58642b0b967b2338e71bfaaa0773a37ee47`.
- Start checkpoint (`MODEL`), with SHA-256 of its metadata: `iter_000002000/model` (`54de8de93b4c1a6f10075f142d872412070eee9194e5f23de51e23c808600e75`).
- Data roots: the seven listed above, validation only. Split manifest SHA-256: missing — no frozen split in the evaluation plan, resolved config, or evaluator output.
- Dirty diff: missing — the evaluation plan pins the code revision but neither the plan nor the evaluation log records a launch-time diff.
- Protocol per checkpoint: joint forward dynamics, inverse dynamics and policy, **32 clips per mode**, seed `20260908`. Forward rollouts: up to three validation episodes per root, **16 windows × 2 seeds = 32 rollouts**, base seed `0` (recorded seeds `0` and `1000`), three chunks each. Both evaluators: **35 sampling steps, guidance 6.0**. Counts are from `metrics_aggregate.json` and `summary.json`.

Recorded evaluation commands:

```sh
python -m cosmos3_h_embodied.scripts.eval --no-guardrails --checkpoint-path $MODEL --config-file \
  cosmos3_h_embodied/configs/config.py --experiment suturebot-h16 --dataset.dataset suturebot_oshri \
  --num-steps 35 --guidance 6.0 --dataset.model-mode joint --dataset.dataset-split val \
  --dataset.num-samples 32 --seed 20260908 -o $EVAL_OUT/eval_joint
python -m cosmos3_h_embodied.scripts.quant_eval --no-guardrails --checkpoint-path $MODEL --config-file \
  cosmos3_h_embodied/configs/config.py --experiment suturebot-h16 --dataset.dataset suturebot_oshri \
  --num-steps 35 --guidance 6.0 --dataset.dataset-split val --episodes-per-root 3 --num-seeds 2 --seed 0 \
  --num-chunks 3 --save-videos -o $EVAL_OUT/quant_eval
```

## suturebot-h16-eval-steps-500-1000-r2 — arm A intermediate-checkpoint evaluation

- Slide: “Arm A vs t8funnhc: public recipe on SutureBot” (slide 21 in the intermediate-checkpoint assignment).
- State: **finished at steps 500 and 1,000**. Job 444850: COMPLETED `0:0`. Failed preflight 444816 is recorded separately below.
- Branch: `ohalimi/suturebot-memory`. Code commit: `63455e9854f6dfd64e313c2a18108409d90dde85`. Framework commit: `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`.
- Evaluation W&B ID and URL: missing — the project run listing and evaluation folder contain no evaluation run. Source training run: [gzluhczu](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/gzluhczu) (checkpoint only).
- Launch script SHA-256: `99ed4bdd2f780231b39c9d22d9019ea80db42eb8318c1933eea3ab3f0166f32b`. Source SHA-256: `fdc972a8148a56538999167f22154daebd43905ca549475ebd3cf1103a59b82e`.
- Start checkpoint (`MODEL`), with SHA-256 of its metadata: `iter_000000500/model` (`c99bb6f0fa284e161f7b1ab6faee939a132e51f8be0b73a184081508ad94109f`); `iter_000001000/model` (`67f18b0bd434fa93a8657ec772fae026018cb1101fd2dec0233bc43bf00bb351`).
- Data roots: the seven listed above, validation only. Split manifest SHA-256: missing — no frozen split in the evaluation plan, resolved config, or evaluator output.
- Dirty diff: missing — the evaluation plan pins the code revision but neither the plan nor the evaluation log records a launch-time diff.
- Protocol per checkpoint: joint forward dynamics, inverse dynamics and policy, **32 clips per mode**, seed `20260908`. Forward rollouts: up to three validation episodes per root, **16 windows × 2 seeds = 32 rollouts**, base seed `0` (recorded seeds `0` and `1000`), three chunks each. Both evaluators: **35 sampling steps, guidance 6.0**. Counts are from `metrics_aggregate.json` and `summary.json`.

Recorded evaluation commands (one pair for each `MODEL`, with separate `EVAL_OUT` directories):

```sh
python -m cosmos3_h_embodied.scripts.eval --no-guardrails --checkpoint-path $MODEL --config-file \
  cosmos3_h_embodied/configs/config.py --experiment suturebot-h16 --dataset.dataset suturebot_oshri \
  --num-steps 35 --guidance 6.0 --dataset.model-mode joint --dataset.dataset-split val \
  --dataset.num-samples 32 --seed 20260908 -o $EVAL_OUT/eval_joint
python -m cosmos3_h_embodied.scripts.quant_eval --no-guardrails --checkpoint-path $MODEL --config-file \
  cosmos3_h_embodied/configs/config.py --experiment suturebot-h16 --dataset.dataset suturebot_oshri \
  --num-steps 35 --guidance 6.0 --dataset.dataset-split val --episodes-per-root 3 --num-seeds 2 --seed 0 \
  --num-chunks 3 --save-videos -o $EVAL_OUT/quant_eval
```

## suturebot-h16-eval-steps-500-1000 — failed evaluation preflight

- Slide: “Arm A vs t8funnhc: public recipe on SutureBot” (slide 21 in the intermediate-checkpoint assignment).
- State: **failed**. Job 444816: FAILED `1:0` before either evaluator started; job 444850 superseded it under a new name.
- Branch: `ohalimi/suturebot-memory`. Code commit: `63455e9854f6dfd64e313c2a18108409d90dde85`. Framework commit: `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`.
- W&B run ID and URL: missing — no matching project run or run folder; [gzluhczu](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/gzluhczu) is the intended checkpoint source only.
- Launch command (from accounting): `sbatch <submitted-wrapper>`. SHA-256 of the submitted copy at the accounting command path: `a02788cbcdf59d3a5beda2c5146894c8715557ff53bf8d319a6a62dee4d7824b`; source: `a091bfb36c1a17539b490e31f8fb196908d7684fc00de582aef8418093c0f412`. The preflight checked the reference pins, then failed looking up the watch file relative to the submitting home directory.
- Intended inputs (not loaded): arm A `iter_000000500/model` and `iter_000001000/model`, the seven roots above. Intended protocol: job 431929’s validation protocol (32 joint clips/mode, seed 20260908; up to three episodes/root, two rollout seeds from base 0, three chunks; 35 steps, guidance 6.0). No clips were scored.
- Split manifest SHA-256: missing — no run folder or evaluation plan was produced. Dirty diff: missing — no worktree status or patch in the preflight log.

## suturebot-current-row — reverted launch

- Slide: “Arm B vs A: does action history help?” (slide 21 in the arm B assignment).
- State: **failed before training; not to be relaunched**. Job 432418: FAILED `143:0`.
- Branch: `ohalimi/suturebot-memory`. Code commit: `f69c31f67e064b29c7b545a67ebc43d9e65e5035`. Framework commit: `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`.
- W&B run ID and URL: missing — no run in the project listing or ID in the job folder; config import failed before W&B initialized.
- Launch script SHA-256: `b99e7158640d1e7c5f497b7f04f81f07887429d0caad6e2bf863204799105940`.
- Data roots: the seven listed above. Split manifest SHA-256: missing — no frozen manifest in the run’s exported config or repro directory; the loader computes the split.
- Trainer seed: `0`. Start checkpoint: `h16_50000_dcp` (weights only; configured, not loaded).
- Dirty diff: none — clean worktree and empty recorded patch.
- Disposition: the pinned commit lacked `cosmos3_h_embodied.callbacks.packed_batch_val`. Revert `d89f3e817318b19a8710a5748858e1799df7d476` preserves the failed commit in history. The command below documents the failure, not a recovery procedure.

Recorded config export and trainer command (`COSMOS_SMOKE` unset):

```sh
python -m cosmos_framework.scripts.export_config -o $RUN_DIR/exported_config.yaml --config-file \
  cosmos3_h_embodied/configs/config.py --experiment suturebot-current-row
bash $CODE/tools/train_with_prune.sh --checkpoint-dir $JOB_DIR/checkpoints --keep-last 2 --keep-every 0 \
  --milestones 500,1000,2000 --save-iter 100 -- torchrun --nnodes=8 --nproc_per_node=4 --rdzv-backend=c10d \
  --rdzv-endpoint=$HOST:29500 --rdzv-id=432418 --rdzv-conf=timeout=120 -m cosmos_framework.scripts._train \
  --config-file $RUN_DIR/exported_config.yaml -o $OUTPUT_ROOT --config-overrides \
  checkpoint.load_path=$DATA/Cosmos3-H-Embodied-Nano/h16_50000_dcp checkpoint.load_training_state=false \
  checkpoint.save_iter=100 job.project=cosmos3_suturebot_memory job.group=suturebot-memory \
  job.name=suturebot-current-row job.wandb_mode=online \
  model.config.parallelism.data_parallel_shard_degree=32 model.config.compile.enabled=false \
  trainer.run_validation=true trainer.run_validation_on_start=true \
  trainer.callbacks.wandb_val._target_=cosmos3_h_embodied.callbacks.packed_batch_val.PackedBatchValLog \
  trainer.validation_iter=100 trainer.max_val_iter=3 trainer.callbacks.norm_monitor.every_n=100 \
  trainer.callbacks.sigma_loss_analysis.every_n=250 trainer.callbacks.sigma_loss_analysis.every_n_viz=250 \
  trainer.callbacks.device_monitor.every_n=50 trainer.callbacks.sampled_media.enabled=true \
  trainer.callbacks.sampled_media.output_uri=$JOB_DIR/sampled_media.lance \
  trainer.callbacks.every_n_sample_reg._target_=cosmos_framework.callbacks.every_n_draw_sample.EveryNDrawSample \
  trainer.callbacks.every_n_sample_reg.every_n=250 \
  trainer.callbacks.every_n_sample_reg.do_x0_prediction=false \
  trainer.callbacks.every_n_sample_ema._target_=cosmos_framework.callbacks.every_n_draw_sample.EveryNDrawSample \
  trainer.callbacks.every_n_sample_ema.every_n=250 trainer.callbacks.every_n_sample_ema.is_ema=true \
  trainer.callbacks.every_n_sample_ema.do_x0_prediction=false trainer.max_iter=2000
```
