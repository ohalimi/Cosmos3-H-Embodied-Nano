# Reproducing the CAMP memory arm

Slide numbering: the Arm C vs A comparison (originally 26) is now slide 38; the alive, harmless and used checks (originally 27) are now slide 39. The original numbers below identify the run records. The three short diagnostics are not policy results or export candidates.

## `suturebot-camp-pretrain` — slides 26 and 27

**State:** running; step 2,250 logged locally (Forge 2,200 at 2026-11-26 16:10 PT), cap 20,000. Slurm job **443703**. No selected-checkpoint gate verdict or downstream result yet.

- Branch `ohalimi/suturebot-camp`, clean code commit `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff. The later dictionary readout rebuild at `c35b0c0d633b3e21a5954cb20a2ddde0ecadd2b9` is **not** the training revision.
- W&B: `holoscan-advanced-workflow-team / cosmos3_suturebot_memory / 9qq6q9tb` ([run and reproduction files](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/9qq6q9tb)).
- Launch: `sbatch <project-root>/<verified-launch-script>`; SHA-256 of the executed script: `69c50c1b20a6b6a576f22447e6322f209de587ab67fe87125df86101e4f02f52`. Retrieve the script and resolved `control.yaml` from the run's `repro/` files. At the pinned commit the underlying CLI is `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda`. The launcher logs the metrics and invokes the corresponding `export` mode on the checkpoint with the lowest sampled validation reconstruction. Do not confuse the final training step with the selected step.
- Fresh trajectory model; no policy checkpoint loaded. Model, pose-fit and episode sampling seed 0. Control split seed 42, validation fraction 0.05. Use the registered H16 control at `63455e9854f6dfd64e313c2a18108409d90dde85`: `hf_suturebot`, `ood`, `suture_bot_failure`, `cosmos_fail_filtered`, `cosmos_throw_fail_demo`, `cosmos_knot_fail_demo`, `suturebot_act_throw_eval`. No validation fitting.
- Registered split manifest SHA-256: `e702f9c54d8a95d69d942459f19445936b401f9b1ab65f6e523daea9850569de`. Canonical bytes: UTF-8 JSON, sorted object keys `train` and `val`, each holding lexicographically sorted `[root_basename, episode_index, native_frame_count]` arrays, compact separators, no newline. Use `registered(control, split)` before the memory trainer's minimum-length filter: 2,188 train and 118 validation episodes. Resolved control SHA-256 `e116ed3ed155cccf7b3ad7debfbeafb655cb0e0bdd433591aa0961b8bd8886c0`; train-only 200-episode pose fit `386bec77130e831e006724e20706e02f963e7dd2839613fe136a4c5845451c5e`; action normalizer `d7ecb2e6d64b5810982032bc61df8796698107367e6661d9b9e7deee37512260`.
- Prospective release gate, registered 15:14 PT on 2026-11-26: compare the **actual export input** with the internal reference at that exact optimizer step. Freeze both dictionaries; take one nearest-center vote per valid state on the same phase-zero panel (first min(512, length) states of each of 114 eligible validation episodes, in root/episode order, 13,524 states, manifest `b9608d6bd9629a8d6834788af2191155ce6bd537796f82b3c171c94dc50cd8db`). Public must have at least as many winning codes, at least as high pooled perplexity, and no more votes for its most common code. Require finite primary reconstruction below the public zero predictor and batch perplexity >6 at all scheduled log rows from 400 steps before selection through selection. No EMA-based count or interim checkpoint substitutes. This encoder probe has no diffusion sampler or guidance.

## `suturebot-camp-internal-alive-1k` — slide 27 diagnostic only

**State:** training and fixed-panel evaluation finished at step 1,000; Slurm job **445121** exited 1:0 in a subsequent, separate public readout. The public readout was completed offline with the corrected comparison. No resume and no policy scoring.

- **Not reproducible from git by request.** Reference commit `8347c01eb179d86a12f312dc522ace41bacf6c10`; the diagnostic adapter was uncommitted, not part of the public branch. Reader from `ohalimi/suturebot-camp` at `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`; same framework environment as the pretrain. No internal source or patch is included here.
- W&B: `holoscan-advanced-workflow-team / cosmos3_suturebot_memory / 3fo9nx2h` ([diagnostic run](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/3fo9nx2h)). Results and adapter checksums: Drive `open-h-remote-agent / status / 2026-11-26T1513-arm-c-internal-diagnostic / internal-diagnostic.txt` (results-only packet).
- Launch: `sbatch <project-root>/<verified-diagnostic-launch-script>`; executed SHA-256 `a896f6ae3ae8cdee73cfc0506ec03d7eb1c9b7d6e008764d95bf90ecadc12dd0`. Registered-split adapter, training mode, `--steps 1000 --device cuda`; fresh initialization, seed 0; same seven roots, split manifest and pose fit as above. Frozen panel at 600, 800 and 1,000; no diffusion sampling or guidance. The 461-winner diagnostic criterion was not met. This is not a fine-tune result or a decision about the pretrain's eventual export.

## Pinpoint runs — diagnostic protocol

These are controlled pretraining diagnostics, not policy runs or candidates for track release. Use the seven registered roots, split manifest, control configuration, 200 training episodes, fresh initialization and seed 0 listed for job 443703. Each uses the clean public rebuild, changes one factor, stops at step 1,000 and does not export. At 600, 800 and 1,000, read 512 assignment counts on CPU from the identical phase-zero validation panel described above. No sampling steps or guidance. W&B files include the launcher, resolved control and (after completion) the three histograms. Job 443703 saved no 600/800 weights or panel summaries; do not substitute a batch metric at those steps.

## `suturebot-camp-mixed-root-batches-1k` — slide 27 diagnostic

**State:** running at step 300; job **445349**. Forge running, step 250.

- Branch `ohalimi/suturebot-camp`, commit `193c90b64f870f068f711f711d2bd653dff717aa`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. Clean; no diff. W&B `holoscan-advanced-workflow-team / cosmos3_suturebot_memory / 3f8e443e` ([run](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/3f8e443e)).
- Launch: `sbatch <project-root>/<verified-launch-script>`, executed SHA-256 `442424a8ea9ef86094f94830c944b1b3d31fde9fa8c0ba94e27c0110258cdf9f`. Underlying training command: `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --mixed-root-batches` with the online logger. Data, model seed, start, panel and probe cadence as above.

## `suturebot-camp-no-camera-1k` — slide 27 diagnostic

**State:** finished at step 1,000; job **445351**. Forge finished, step 1,000.

- Branch `ohalimi/suturebot-camp`, commit `193c90b64f870f068f711f711d2bd653dff717aa`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. Clean; no diff. W&B `holoscan-advanced-workflow-team / cosmos3_suturebot_memory / 219576a3` ([run](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/219576a3)).
- Launch: `sbatch <project-root>/<verified-launch-script>`, executed SHA-256 `9f6c889447ea8720665dacacdccfbf7a4a51eaeb8fa2473626000662cf6dd1b0`. Underlying training command: `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --no-camera` with the online logger. Data, model seed, start, panel and probe cadence as above.

## `suturebot-camp-internal-action-scaling-1k` — slide 27 diagnostic

**State:** running at step 350; job **445353**. Forge running, step 300.

- Branch `ohalimi/suturebot-camp`, commit `193c90b64f870f068f711f711d2bd653dff717aa`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. Clean; no diff. W&B `holoscan-advanced-workflow-team / cosmos3_suturebot_memory / ed014e63` ([run](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/ed014e63)).
- Launch: `sbatch <project-root>/<verified-launch-script>`, executed SHA-256 `6a7786547d9a179c75707be7ab5e3b800c3a5b26f7dd11bd4cb43d9e615572f9`. Underlying training command: `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --internal-action-scaling` with the online logger. Data, model seed, start, panel and probe cadence as above.

<!-- end of pinpoint diagnostics -->

<!-- diagnosis owner records -->

## Differential diagnosis — reference and noise protocol (slide 39, originally 27)

Same seven registered roots, split manifest `e702f9c54d8a95d69d942459f19445936b401f9b1ab65f6e523daea9850569de`, and frozen 200-episode pose fit as job 443703 above. Each starts fresh; no policy checkpoint, sampler or guidance. No episode tracks. Frozen CPU evaluation at completed multiples of 1,000 over the same 114 validation episodes and 13,524 real states. Source-free results packet: `camp-diagnosis/run-evidence.md` and matched measurements (owner-local; no Drive delivery by this owner). Read the observer, scorer, adapter and split hashes from the run config.

### `suturebot-camp-diagnosis-reference-20k`

- State: completed, last local counter step 20,000 / 20,000, Slurm job **445774**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`53ab9aa0`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/53ab9aa0).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `d3c25e78fd68be211439a16a2e54e73bead8f04ba149db4faef63b2eac800803`. In-place registered adapter `train --seed 0 --steps 20000 --device cuda`. Data and protocol as above.

### `suturebot-camp-diagnosis-noise-s1-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445788**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`a41486f9`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/a41486f9).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `d7fdc6c22e3da363654bfc974d47157b55253166367d15ab14c6c8c42ecf6c32`. In-place registered adapter `train --seed 1 --steps 1000 --device cuda`. Data and protocol as above.

### `suturebot-camp-diagnosis-noise-s2-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445790**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`71242437`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/71242437).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `3473f3e8ffa4925047c630191bf1bd18c36ecd43077e89299b66a6e79b0d2b05`. In-place registered adapter `train --seed 2 --steps 1000 --device cuda`. Data and protocol as above.

### `suturebot-camp-diagnosis-noise-s3-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445792**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`a48c3515`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/a48c3515).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `50f7b2b5e47659fbd962812375aaab067b1201e4db877d6ee67dfd49f42a3d1a`. In-place registered adapter `train --seed 3 --steps 1000 --device cuda`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-public-lr-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445808**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`49362249`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/49362249).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `d7fddd67b2281fe5775567820aa7b5dabd1b21f88b78311339f4e3063866113c`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda --lr 0.0001`. Data and protocol as above.

### `suturebot-camp-diagnosis-public-fast-lr-1k`

- State: running, last local counter step 1/1,000. Slurm job **445815**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `703f0d922e1f7801369c42be916c683cc97a37a9`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`7e2526e3`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/7e2526e3); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `a9917072e6b2cb5d39b40ae171a8c64aa3359ffc90aa5254b34a757fc38ea0e3`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --fast-learning-rate` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.
