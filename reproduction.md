# Reproduction record

## nano-h16-suturebot

**Running at step 20,280 of 50,000** (Forge, 26 Nov 2026 17:24 PT); the latest complete save at this check was 20,200. The evaluation inputs at steps 0 (Cosmos3-Nano), 100, 1,000, 5,000 and 10,000 are present. Steps 25,000 and 50,000 have not been saved. Slides 7 and 18 have no held-out scores from this run yet: a training validation loss is not a rollout score.

| Job | Starting state | Outcome |
| --- | --- | --- |
| 438465 | Clean base `11a02c275e2a24c30cb1f8ca697164d0ce28c140`, initialized from **Cosmos3-Nano** | Failed after saving step 8,400. On rank 62 the sigma logger paired 9 vision sigmas with a single loss. |
| 442417 | Same base, output directory and Forge run; restored model, optimizer, scheduler, trainer and data-loader state from `iter_000008400` | Running. First new iteration 8,401; first complete post-resume save 8,500. |

**Code and run.** Branch `ohalimi/nano-h16-suturebot`; framework commit `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. W&B: [holoscan-advanced-workflow-team / cosmos3_suturebot_memory / zrhxudfh](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/zrhxudfh). Both jobs used 16 nodes × 4 ranks. Trainer seed (`repro/seed`): **0**; data and split seed: **42**.

**Resume diff.** The first job has no patch. For the second, keep the base above and apply exactly `git diff 11a02c275e2a 5433f75f1421670963c9c99ee17b9cd734767251`. This is the tree change recorded by Forge as `repro/dirty=true`, `repro/diff`, and `ops/fix_commit`. `experiment_h16.py` selects `PackedSigmaLossAnalysis`; `packed_sigma_loss.py` excludes missing or mismatched per-item vision losses from the logger without dropping its distributed gathers; `test_packed_sigma_loss.py` covers that behavior. No optimizer, objective, split, or step-cap change. The archived resume `diff.patch` has SHA-256 `e267d0e79a3a4403c734e595d4ffb713f21061a5ff70c20dc8797cea56ace763` (including its final newline).

**Inputs.** Set `OPENH_DATA_ROOT` to the parent of the two prepared v3.0 roots. Use `tools/data_prep` for the public point-labeled wound-closure root; do not use the raw download or substitute a clean-video twin. These are the only eligible roots:

| Public root | Train episodes | Validation episodes | Exclusions |
| --- | ---: | ---: | --- |
| `jhu/hf_suturebot` | 1,385 | 73 | None. Copies 298/579/323/628/350/675 and originals 46/47/48/49/50/51 are all train-side. |
| `jhu/imerse/wound_closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled` | 202 | 11 | 198, 199, 201 |

Frozen `val-episodes.json` SHA-256: `c06581a2348a981eb0100960daedb0fd684cef8539e9dc1a75961a761a6133f4` for both jobs. Split whole episodes with ratio 0.05 before making windows; within each root the train and validation IDs are disjoint. Normalizer SHA-256 values from `repro/normalizer_sha256`: `dvrk_jhu_backward_anchored_worldtrans_rot6d.json` = `309209ce315b3fd5b6b40176eae2060dfbb2399b4944aff8d60179a2808842cf`; `suturebot_backward_anchored_worldtrans_rot6d.json` = `d7ecb2e6d64b5810982032bc61df8796698107367e6661d9b9e7deee37512260`.

**Launch record.** Source launcher SHA-256 `22fd70653fe564b5636c7c609fc07041c7a00a496c8213798e689800bcf80a49`; executed launcher SHA-256 `39d1bcf66f3c8121cde4c80853f0063963b000d75c7cab11e77925eea97e514e` in both jobs. Only the interpreter line was rewritten by the site; the resume used the identical archived executable. Forge `repro/command` records the export and training invocation. Below, `CODE` is the clean or patched checkout for the job, `OUT` the experiment output parent, `RUN=$OUT/nano-h16-suturebot`, `JOB=$RUN/cosmos3_suturebot_memory/suturebot-memory/nano-h16-suturebot`, and `NANO` the Cosmos3-Nano directory **containing** `model/`. `HOST` is the allocation's rendezvous host and `JOB_ID` is the applicable job above. This is the training payload, not a new submission.

```sh
cd "$CODE"
python -m cosmos_framework.scripts.export_config -o "$RUN/exported_config.yaml" \
  --config-file cosmos3_h_embodied/configs/config.py --experiment nano-h16-suturebot
bash tools/train_with_prune.sh --checkpoint-dir "$JOB/checkpoints" --keep-last 2 --keep-every 0 \
  --milestones 100,1000,5000,10000,25000,50000 --save-iter 100 -- \
  torchrun --nnodes=16 --nproc_per_node=4 --rdzv-backend=c10d --rdzv-endpoint="$HOST:29500" \
  --rdzv-id="$JOB_ID" --rdzv-conf=timeout=120 -m cosmos_framework.scripts._train \
  --config-file "$RUN/exported_config.yaml" -o "$OUT" --config-overrides \
  checkpoint.load_path="$NANO" checkpoint.load_training_state=false checkpoint.save_iter=100 \
  job.project=cosmos3_suturebot_memory job.group=suturebot-memory job.name=nano-h16-suturebot job.wandb_mode=online \
  model.config.parallelism.data_parallel_shard_degree=64 model.config.compile.enabled=false \
  trainer.run_validation=true trainer.run_validation_on_start=true \
  trainer.callbacks.wandb_val._target_=cosmos3_h_embodied.callbacks.packed_batch_val.PackedBatchValLog \
  trainer.validation_iter=100 trainer.max_val_iter=3 trainer.callbacks.norm_monitor.every_n=100 \
  trainer.callbacks.sigma_loss_analysis.every_n=250 trainer.callbacks.sigma_loss_analysis.every_n_viz=250 \
  trainer.callbacks.device_monitor.every_n=50 trainer.callbacks.sampled_media.enabled=true \
  trainer.callbacks.sampled_media.output_uri="$JOB/sampled_media.lance" \
  trainer.callbacks.every_n_sample_reg._target_=cosmos_framework.callbacks.every_n_draw_sample.EveryNDrawSample \
  trainer.callbacks.every_n_sample_reg.every_n=250 trainer.callbacks.every_n_sample_reg.do_x0_prediction=false \
  trainer.callbacks.every_n_sample_ema._target_=cosmos_framework.callbacks.every_n_draw_sample.EveryNDrawSample \
  trainer.callbacks.every_n_sample_ema.every_n=250 trainer.callbacks.every_n_sample_ema.is_ema=true \
  trainer.callbacks.every_n_sample_ema.do_x0_prediction=false
```

For the clean launch there was no latest checkpoint. On resume the run's own latest pointer took precedence over the Nano initialization. Saves are every 100 steps. The best validation checkpoint observed so far (step 3,200) was pruned before it was requested. Steps 25,000 and 50,000 are retention milestones. Step 46,937 is not a scheduled save; a neighboring step is not equivalent. No separate evaluation is recorded here.
