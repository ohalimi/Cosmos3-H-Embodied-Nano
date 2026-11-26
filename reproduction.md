# Reproduction ledger: SutureBot history mask

Job numbers below are Slurm allocations, not training steps. Slide titles identify the result across deck renumberings. Script digests are SHA-256 of the submitted copy unless explicitly labeled **source**. Commands are the recorded application invocations with local paths and rendezvous hosts replaced; they do not contain the launch wrappers.

- **Data (seven roots):** `hf_suturebot`, `ood`, `suture_bot_failure`, `cosmos_fail_filtered`, `cosmos_throw_fail_demo`, `cosmos_knot_fail_demo`, `suturebot_act_throw_eval`, in the `suturebot_oshri` family.
- **Split:** the pinned loader's `split_episode_ids` permutes each root's episode indices with seed 42, takes the first `round(0.05 × total_episodes)` for validation and the remainder for training, before filtering. The committed exclusions overlay has SHA-256 `c2eff8a6c6113d32b8db00b0910b978a8ed4feff346fd5c9b8541955892d27b7`; it is **not** a split manifest.
- **Command names:** `CODE` is the pinned checkout (plus any recorded diff); `DATA` contains the initial checkpoint; `OUTPUT_ROOT` is the training output base. `RUN_DIR` is the per-token output folder, `JOB_DIR` its nested training job folder, and `HOST` the rendezvous host. For evaluations, `MODEL` is the named checkpoint's `model` directory and `EVAL_OUT` the output folder for that checkpoint.
- **W&B entity/project:** `holoscan-advanced-workflow-team/cosmos3_suturebot_memory`. A source training link is not an evaluation run.

## suturebot-history-mask — arm B

Jobs 443070 and 444259 shared the token, not the W&B recorder. The first trained the model; the second evaluated its step-2,000 checkpoint. Neither allocation was requeued.

### Training — job 443070

- Slide: “Arm B vs A: does action history help?” (slide 21 in the arm B assignment).
- State: **finished at step 2,000**. Job 443070: COMPLETED `0:0`.
- Branch: `ohalimi/suturebot-history-mask`. Code commit: `54ead497244ce77a45e284e32261bff6d1408082`. Framework commit: `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`.
- W&B: [cyi0yxho](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/cyi0yxho).
- Launch script SHA-256: `8f0d5db9d5f473c49143ee691fd0e279df168aaac4268ed479333c8fb0ee4c96`.
- Data roots: the seven listed above. Split manifest SHA-256: missing — no frozen manifest in the run’s config, repro directory or W&B record; the loader computes the split.
- Trainer seed: `0`. Start checkpoint: `h16_50000_dcp` (weights only).
- Dirty diff: none — clean worktree and empty recorded patch.

Recorded config export and trainer command (`COSMOS_SMOKE` unset):

```sh
python -m cosmos_framework.scripts.export_config -o $RUN_DIR/exported_config.yaml --config-file \
  cosmos3_h_embodied/configs/config.py --experiment suturebot-history-mask
bash $CODE/tools/train_with_prune.sh --checkpoint-dir $JOB_DIR/checkpoints --keep-last 2 --keep-every 0 \
  --milestones 500,1000,2000 --save-iter 100 -- torchrun --nnodes=8 --nproc_per_node=4 --rdzv-backend=c10d \
  --rdzv-endpoint=$HOST:29500 --rdzv-id=443070 --rdzv-conf=timeout=120 -m cosmos_framework.scripts._train \
  --config-file $RUN_DIR/exported_config.yaml -o $OUTPUT_ROOT --config-overrides \
  checkpoint.load_path=$DATA/Cosmos3-H-Embodied-Nano/h16_50000_dcp checkpoint.load_training_state=false \
  checkpoint.save_iter=100 job.project=cosmos3_suturebot_memory job.group=suturebot-memory \
  job.name=suturebot-history-mask job.wandb_mode=online \
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

### Evaluation — job 444259

- Slide: “Arm B vs A: does action history help?” (slide 21 in the arm B assignment).
- State: **finished at step 2,000**. Job 444259: COMPLETED `0:0`.
- Branch: `ohalimi/suturebot-history-mask`. Code commit: `54ead497244ce77a45e284e32261bff6d1408082`. Framework commit: `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`.
- Evaluation W&B ID and URL: missing — the project run listing and evaluation folder contain no evaluation run. Source training run: [cyi0yxho](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/cyi0yxho) (checkpoint only).
- Launch script SHA-256: `7887cbe08924ae972e86a6f966f4fbe28c49040c0c7815d0be2272650464b530`.
- Start checkpoint (`MODEL`), with SHA-256 of its metadata: `iter_000002000/model` (`d62c778076982aa68aad6a7d91dbd260bd8dbcb7eacabf4ac43b46462c10fa02`).
- Data roots: the seven listed above, validation only. Split manifest SHA-256: missing — no frozen split in the evaluation plan, resolved config, or evaluator output.
- Dirty diff: missing — the evaluation plan pins the code revision but neither the plan nor the evaluation log records a launch-time diff.
- Protocol per checkpoint: joint forward dynamics, inverse dynamics and policy, **32 clips per mode**, seed `20260908`. Forward rollouts: up to three validation episodes per root, **16 windows × 2 seeds = 32 rollouts**, base seed `0` (recorded seeds `0` and `1000`), three chunks each. Both evaluators: **35 sampling steps, guidance 6.0**. Counts are from `metrics_aggregate.json` and `summary.json`.
- Paired windows: `arm-a-rollout-plan.json`, SHA-256 `53ca9ed9d575a556f380c312f9675f98afaf9ef2e09722f642cd534b82b3c982`, from arm A job 431929. This selects rollout windows; it is not the dataset split.

Recorded evaluation commands:

```sh
python -m cosmos3_h_embodied.scripts.eval --no-guardrails --checkpoint-path $MODEL --config-file \
  cosmos3_h_embodied/configs/config.py --experiment suturebot-history-mask --dataset.dataset \
  suturebot_oshri --num-steps 35 --guidance 6.0 --dataset.model-mode joint --dataset.dataset-split val \
  --dataset.num-samples 32 --seed 20260908 -o $EVAL_OUT/eval_joint
python -m cosmos3_h_embodied.scripts.quant_eval --no-guardrails --checkpoint-path $MODEL --config-file \
  cosmos3_h_embodied/configs/config.py --experiment suturebot-history-mask --dataset.dataset \
  suturebot_oshri --num-steps 35 --guidance 6.0 --dataset.dataset-split val --plan \
  $RUN_DIR/arm-a-rollout-plan.json --episodes-per-root 3 --num-seeds 2 --seed 0 --num-chunks 3 \
  --save-videos -o $EVAL_OUT/quant_eval
```
