# Reproduction record

## nano-h16-suturebot

**Running, 27 Nov 2026 11:00 PT.** Slurm job 442417 was running on 16 nodes; the original Forge run reported step **44,820 of 50,000**. The latest checkpoint pointer named **44,800**. Both that save and the retained 25,000 save had nonempty model, optimizer, scheduler and trainer metadata and 64 nonempty model shards. The five one-node evaluations through step 10,000 finished; their Forge plans and summaries matched the saved files. **Step 25,000 is saved but not scored:** the input gate again exited 1, so no evaluation job or W&B record was created. The 50,000 save did not yet exist.

**Data gate: failed.** At the 25,000 check both dataset READMEs and the Cosmos3-Nano README explicitly said `kosher: no`; the retained checkpoint directory had no README. Both training attempts, their checkpoints, and all Slide 7 and 18 results derived from them are **provisional**. The existing training job continues; no new training or evaluation may be submitted unless the gate passes for every input. Completion and matching digests establish what ran, not that the inputs match the reference team's data.

| Job | Starting state | Outcome |
| --- | --- | --- |
| 438465 | Clean base `11a02c275e2a24c30cb1f8ca697164d0ce28c140`, initialized from **Cosmos3-Nano** | Failed after saving step 8,400. On rank 62 the sigma logger paired 9 vision sigmas with a single loss. |
| 442417 | Same base, output directory and Forge run; restored model, optimizer, scheduler, trainer and data-loader state from `iter_000008400` | Running. First new iteration 8,401; first complete post-resume save 8,500. |

**Code and run.** Branch `ohalimi/nano-h16-suturebot`; framework commit `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. W&B: [holoscan-advanced-workflow-team / cosmos3_suturebot_memory / zrhxudfh](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/zrhxudfh). Both jobs used 16 nodes × 4 ranks. Trainer seed (`repro/seed`): **0**; data and split seed: **42**.

**Resume diff.** The first job has no patch. For the second, keep the base above and apply exactly `git diff 11a02c275e2a 5433f75f1421670963c9c99ee17b9cd734767251`. This is the tree change recorded by Forge as `repro/dirty=true`, `repro/diff`, and `ops/fix_commit`. `experiment_h16.py` selects `PackedSigmaLossAnalysis`; `packed_sigma_loss.py` excludes missing or mismatched per-item vision losses from the logger without dropping its distributed gathers; `test_packed_sigma_loss.py` covers that behavior. No optimizer, objective, split, or step-cap change. The archived resume `diff.patch` has SHA-256 `e267d0e79a3a4403c734e595d4ffb713f21061a5ff70c20dc8797cea56ace763` (including its final newline).

**Historical inputs (not verified for reuse).** `OPENH_DATA_ROOT` pointed to the parent of the two v3.0 roots below. The recorded run used the public point-labeled wound-closure root built with `tools/data_prep`, not the raw download or a clean-video twin. This describes the run; it is not a preparation instruction or evidence of equivalence to the reference team's copy.

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

For the clean launch there was no latest checkpoint. On resume the run's own latest pointer took precedence over the Nano initialization. Saves are every 100 steps. The best validation checkpoint observed so far (step 3,200) was pruned before it was requested. Steps 25,000 and 50,000 are retention milestones. Step 46,937 is not a scheduled save; a neighboring step is not equivalent. See the separate evaluation runs below.

## Slide 18 evaluation protocol

All entries below use branch `ohalimi/nano-h16-suturebot`, evaluator commit `ee497ad8464162157ddc1aacb0ac9595a8493e78`, and the framework commit and frozen split in the training entry above. The 84 eligible episodes are not 84 sampled clips: the joint evaluation scores 32 physical windows (28 mono, 4 wound closure), each in forward dynamics, inverse dynamics and WAM. `make_panel` selects evenly spaced sorted validation episode IDs per root, at source frame 48. The first three selected per root also get three consecutive rollout chunks. Panel SHA-256: `1c502b6bb020fbe5028f19608b1f120c6349493cf0e7b16fbea01c130dadc30e`. A substituted window is an error. No panel, model weights, or per-sample files are stored in this repository.

One node per checkpoint, one GPU per seed, seeds **20260908, 20260909, 20260910**, 35 denoising steps and guidance 6.0. Report WAM policy MSE, FD PSNR, ID MSE (joint summarizer, All), and rollout PSNR (quantitative rollout summary, All). Report all three seed values, their mean and max minus min. The step-0 baseline uses regular Nano weights; later saves use their own EMA. Source launcher SHA-256 `5b1eeee38471e7c0845659a5896ba6dc8941c76f1ed7ba57c4a82665b4fd5792`; executed launcher `f7cf122dc06c9c2777be0c3c374769b4b35456992718eab9316c28ed420ac760` (site interpreter only). Each Forge eval record has its plan and receives its summary when complete. The command for each seed is:

```sh
python -m tools.benchmark.heldout_panel --checkpoint "$MODEL_DIR" \
  --manifest "$FROZEN_MANIFEST" --manifest-sha256 c06581a2348a981eb0100960daedb0fd684cef8539e9dc1a75961a761a6133f4 \
  --seed "$SEED" --weights "$WEIGHTS" --output "$SEED_OUTPUT"
```

## nano-h16-suturebot-eval-s0

Slide 18, checkpoint **0** (`Cosmos3-Nano/model`, regular); model metadata SHA-256 `21dea15c9aac109bbd0c0b6efa55df101fb21b33f1264a0b284b7ea4c84481ae`. Job **446447**, [Forge eval 37589c88](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/37589c88), finished; verified against the online plan and summary files. Same code, launcher, panel and three-seed command as above.

## nano-h16-suturebot-eval-s100

Slide 18, checkpoint **100** (`iter_000000100/model`, ema); model metadata SHA-256 `73c7a75272d6a1717fb2b576e63e34c57f6efcf1a4424ea17775e3e94f161928`. Job **446449**, [Forge eval bf5edf36](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/bf5edf36), finished; verified against the online plan and summary files. Same code, launcher, panel and three-seed command as above.

## nano-h16-suturebot-eval-s1000

Slide 18, checkpoint **1,000** (`iter_000001000/model`, ema); model metadata SHA-256 `d5058119de32ea6e5cd5a24ea216356da071abf3d0b30bbb52ab8446cf925996`. Job **446451**, [Forge eval 384348b5](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/384348b5), finished; verified against the online plan and summary files. Same code, launcher, panel and three-seed command as above.

## nano-h16-suturebot-eval-s5000

Slide 18, checkpoint **5,000** (`iter_000005000/model`, ema); model metadata SHA-256 `91a5cfe95b1689f9745259786fb53c997da993ffbb8fe42fe525600c31573adb`. Job **446453**, [Forge eval 8cebe80a](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/8cebe80a), finished; verified against the online plan and summary files. Same code, launcher, panel and three-seed command as above.

## nano-h16-suturebot-eval-s10000

Slide 18, checkpoint **10,000** (`iter_000010000/model`, ema); model metadata SHA-256 `473b619bcd14d97bf79b530d1c0d985acb5dea3afc5326aa5573b65465c1ccfc`. Job **446666**, [Forge eval 747fd3d0](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/747fd3d0), finished; verified against the online plan and summary files. Same code, launcher, panel and three-seed command as above.
