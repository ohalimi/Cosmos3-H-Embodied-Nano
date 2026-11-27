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

- State: running, last local counter step 150/1,000. Slurm job **445815**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `703f0d922e1f7801369c42be916c683cc97a37a9`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`7e2526e3`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/7e2526e3); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `a9917072e6b2cb5d39b40ae171a8c64aa3359ffc90aa5254b34a757fc38ea0e3`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --fast-learning-rate` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-reference-narrow-hidden-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445822**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`7889b441`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/7889b441).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `49753852954c44c5dadda0b236ee1456f59c333fd6b4a48eb9134b9577f10aa9`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda --hidden-dim 64`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-flat-frequency-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445824**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`e487f039`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/e487f039).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `8622c705e8042a68387f183606d362df19500bc3665228f3c5833e6cff02adb0`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda --gamma 0`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-long-window-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445826**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`58621bdf`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/58621bdf).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `84a80a4cd4092c81cdaf196de5997ba7e6bffbd4e153570fa69724a29511b889`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda --recon-len 619`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-multi-lag-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445828**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`63e70a06`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/63e70a06).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `e78c6d18b19bc000c28054320a0943ddce9c17d29fe766d61283497b9d733dfb`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda (multi-lag overlap observer)`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-uncapped-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445830**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`05bf63dc`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/05bf63dc).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `6295fa9f7c15284e65b8646c1fbadecae3e8d254139dcdb2d73c671353ef2fab`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda --cap-len 619`. Data and protocol as above.

### `suturebot-camp-diagnosis-public-wide-hidden-1k`

- State: running, last local counter step 100/1,000. Slurm job **445841**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `8190b91bf61793cd553d2c4817aebbcbe56ae388`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`d2cf1baa`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/d2cf1baa); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `bb30ee964660de152bb6193189bd4b87c5e202ee3693b183b332d393bc22ae57`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --wide-hidden` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-public-frequency-decay-1k`

- State: running, last local counter step 100/1,000. Slurm job **445843**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `8190b91bf61793cd553d2c4817aebbcbe56ae388`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`86bfa8b1`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/86bfa8b1); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `9f7fa833aec719b37fbcda0992fd8b004db9814c179e0b50f4cf28957cb5bbff`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --frequency-decay` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-public-short-dct-window-1k`

- State: running, last local counter step 100/1,000. Slurm job **445845**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `8190b91bf61793cd553d2c4817aebbcbe56ae388`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`5df5ec91`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/5df5ec91); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `76eeedd7e45e11b41cb0f4b5b8182d3b33c4dcb6a1fb827c84667954f747b465`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --short-dct-window` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-public-single-consistency-lag-1k`

- State: running, last local counter step 100/1,000. Slurm job **445847**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `8190b91bf61793cd553d2c4817aebbcbe56ae388`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`26c133c9`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/26c133c9); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `9e1e582274f0b16b08bcbe6ec0f583760afcc7c943d61fdd10b829a2eab579be`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --single-consistency-lag` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-public-cap-sequences-1k`

- State: running, last local counter step 100/1,000. Slurm job **445849**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `8190b91bf61793cd553d2c4817aebbcbe56ae388`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`69a6f732`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/69a6f732); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `cc3991a0f9b4dc03944e350f68bfd3d3313974fbad769f578da9426e17f46808`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --cap-sequences` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-reference-with-camera-1k`

- State: running, last local counter step 50 / 1,000, Slurm job **445881**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`b58bef67`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/b58bef67).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `9b0bc498a7c33f43a80af876877937bd6f2a9783cfeb7b43477e81ec6d652032`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda (with left-camera front end)`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-unclipped-quantile-actions-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445885**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`e2dd5dc0`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/e2dd5dc0).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `a6d463d45ab312b0434d8dee1b37ec50fd202cdf7d5126dc958ddeef8c46b28b`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda (single run-specific treatment: unclipped-quantile-actions)`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-clipped-z-actions-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445887**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`4594117b`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/4594117b).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `4785f901e169c38e1b4350cc0fd3c2f4c644ec5e450dcca97a23b4d0d619f51a`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda (single run-specific treatment: clipped-z-actions)`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-whole-public-action-scaling-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445889**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`8d83ddbb`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/8d83ddbb).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `7893830e308754cf63398280868eedb28a0691655697d5d6b8c91c44bddf332f`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda (single run-specific treatment: whole-public-action-scaling)`. Data and protocol as above.

### `suturebot-camp-diagnosis-public-clipped-z-actions-1k`

- State: running, last local counter step 50/1,000. Slurm job **445892**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `e99bc8f1230c170ff1ca917c0bde9ba3048b9238`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`668a33f3`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/668a33f3); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `3b1ac19dff17d5364d0d99156c8cf6f9e2eff260e7cf085b4e0036c739c052cc`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --clipped-z-actions` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-public-unclipped-quantile-actions-1k`

- State: running, last local counter step 50/1,000. Slurm job **445894**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `e99bc8f1230c170ff1ca917c0bde9ba3048b9238`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`a6488ea1`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/a6488ea1); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `85fa809e4f3fb20fe1afecffb6b3b84117f01314b59382a8f52a787db6b3cde6`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --unclipped-quantile-actions` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-reference-camera-init-draws-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445912**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`23a77989`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/23a77989).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `9adb11fb0d57978153f4166a168208db2cd8d5dae8084ec3daaaf10e90f2a368`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda (single run-specific treatment: camera-init-draws)`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-no-embedding-init-draws-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445914**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`bcca442d`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/bcca442d).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `7d143be18a37d21d232268111fa2c55a6dfd83acfc30157c0f959d8043c2b278`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda (single run-specific treatment: no-embedding-init-draws)`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-crop-draws-before-vq-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445916**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`cf284781`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/cf284781).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `8d7ac32cf40af37341e26113349804979e0955d8553b08c41d06eb21c32a1337`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda (single run-specific treatment: crop-draws-before-vq)`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-cdist-training-assignment-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445918**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`864b4791`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/864b4791).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `b11dfe2ae8a7511e558677c47e76c01217286b8281e3a7bb3e562f3454b5b5c6`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda (single run-specific treatment: cdist-training-assignment)`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-indexed-center-sums-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445920**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`dfd54a6f`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/dfd54a6f).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `75704510004b72693d79f3a928151df84d77374fc08d99b49d58dfeaf99c9799`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda (single run-specific treatment: indexed-center-sums)`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-linear-interpolation-ema-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445922**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`25846c07`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/25846c07).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `b076dd80b98d2111fae704c416d0e0ef19663b447bcd9f1abad8cfb8e21d80e9`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda (single run-specific treatment: linear-interpolation-ema)`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-multiply-ratio-smoothing-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445924**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`bac2414a`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/bac2414a).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `9b8c6c48c0b5b018d1e2b870e51c9d62d8767cbc55bef98c161abe0faf985e9d`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda (single run-specific treatment: multiply-ratio-smoothing)`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-double-precision-basis-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445926**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`d291f7cd`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/d291f7cd).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `1de51a57c78ac7b0e8a042f847ac637e2b45d2a0f66007f7a3e3b8f38c64ae2d`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda (single run-specific treatment: double-precision-basis)`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-slow-lr-narrow-hidden-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445932**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`f5a612a6`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/f5a612a6).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `1bf58e0ec735ec5f239024d26d978d1d313bd2bdfbe8c7214c9e185d98cbcf56`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda (single run-specific treatment: slow-lr-narrow-hidden)`. Data and protocol as above.

### `suturebot-camp-diagnosis-reference-slow-lr-flat-frequency-1k`

- State: completed, last local counter step 1,000 / 1,000, Slurm job **445934**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`bb0bad3d`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/bb0bad3d).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `dec4b7722c5e33fb838408bec6bb30ce5dd944c7402874f6df5e871cb595216a`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda (single run-specific treatment: slow-lr-flat-frequency)`. Data and protocol as above.

### `suturebot-camp-diagnosis-public-squared-training-assignment-1k`

- State: running, last local counter step 1/1,000. Slurm job **445938**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `f62327a8f81d7e012059b8dce060c4c622a76fd1`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`760b3334`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/760b3334); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `150e76a1659a0a67187ce6ca4280ef71de1b52485d8c1ff7a1c015a740a38efd`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --squared-training-assignment` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-public-independent-camera-init-1k`

- State: running, last local counter step 1/1,000. Slurm job **445954**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `d8992661e703f43cf87ad9021fb52076edf244ff`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`64338446`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/64338446); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `883cbc6f5ccb9b6f3e4a3b3155d113365957b64e61849c470685c216752cf5f3`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --independent-camera-init` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-public-embedding-init-draws-1k`

- State: running, last local counter step 1/1,000. Slurm job **445956**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `d8992661e703f43cf87ad9021fb52076edf244ff`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`379a41b9`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/379a41b9); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `f07460ee7c67ceec19c2032826a39d4b8e79154547031084ff85054569969d13`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --embedding-init-draws` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-public-center-training-crop-1k`

- State: running, last local counter step 1/1,000. Slurm job **445958**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `d8992661e703f43cf87ad9021fb52076edf244ff`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`5b778504`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/5b778504); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `563acaa81fc2509ba43373f61abbdec19f232debdaf6e27f90007df6fb3c1f25`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --center-training-crop` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-public-independent-vq-draws-1k`

- State: running, last local counter step 1/1,000. Slurm job **445960**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `d8992661e703f43cf87ad9021fb52076edf244ff`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`5a63dffa`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/5a63dffa); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `99ac9aee4196952858ac54975b4e11078135778440719ecae0280f5a641812f7`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --independent-vq-draws` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-public-matrix-center-sums-1k`

- State: running, last local counter step 0/1,000. Slurm job **445965**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `f62327a8f81d7e012059b8dce060c4c622a76fd1`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`4577350e`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/4577350e); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `2c8c9d20d131c3f6f877f5d6988dba65ee246b5883f881f28d5a71ae44b65a27`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --matrix-center-sums` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-public-multiply-add-ema-1k`

- State: running, last local counter step 0/1,000. Slurm job **445969**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `f62327a8f81d7e012059b8dce060c4c622a76fd1`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`90797bdf`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/90797bdf); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `57aedae3ef72c5a14e068b7d722a4940fda01a94d4d036dfd5270db01d24fb03`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --multiply-add-ema` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-public-divide-first-smoothing-1k`

- State: running, last local counter step 0/1,000. Slurm job **445971**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `f62327a8f81d7e012059b8dce060c4c622a76fd1`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`09e5e85a`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/09e5e85a); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `cedd435e61a6ef45bcd6edea011a8f4c5b633709eabe888acc87a072ba884f18`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --divide-first-smoothing` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-public-single-precision-basis-1k`

- State: running, last local counter step 0/1,000. Slurm job **445973**.
- Clean public branch `ohalimi/suturebot-camp-diagnosis`, training commit `f62327a8f81d7e012059b8dce060c4c622a76fd1`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No dirty diff.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`e1074db2`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/e1074db2); reproduction files are attached to the run.
- `sbatch <project-root>/<verified-diagnostic-script>`, SHA-256 `943698e8b45fb0a8e1dab8b854f929d9e3dea6f3acd0acd732bad5828405f84b`. Public `python -m cosmos3_h_embodied.scripts.pretrain_trajectory_codes pretrain --control-config <control.yaml> --out <run-dir> --device cuda --single-precision-basis` with online logger. Seed 0; fresh model; same roots, split, panel and evaluation as above.

### `suturebot-camp-diagnosis-reference-camera-center-crop-1k`

- State: running, last local counter step 0 / 1,000, Slurm job **445975**.
- Not reproducible from public git by request: in-place commit `8347c01eb179d86a12f312dc522ace41bacf6c10`, framework `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`; uncommitted private instrumentation, no source diff distributed. Public reader `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`. Ledger branch `ohalimi/suturebot-camp-diagnosis`.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [`472fb4d9`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/472fb4d9).
- Launch: `sbatch <project-root>/<verified-diagnostic-script>`, executed script SHA-256 `1b15637f00fc88f24684f6f8b1cf761f7e3a7804667a7f0f4b601f208aba404d`. In-place registered adapter `train --seed 0 --steps 1000 --device cuda (single run-specific treatment: camera-center-crop)`. Data and protocol as above.

### `suturebot-camp-diagnosis-public-443703-readout` — frozen public comparator

- CPU-only readout, not a new training job. Source job 443703, W&B `9qq6q9tb`, clean source training commit `5ef4a62e37a2b2b84f59e30b0880a3a4b73cd5e6`, framework and registered split as above. Reads immutable DCP checkpoints, not tracks, with the same private scorer as the reference. Observer uncommitted by request. Source-free evidence packet as above. Public training run is never modified.
- W&B: holoscan-advanced-workflow-team / cosmos3_suturebot_memory / [87c19fc4](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/87c19fc4). Latest shared readout: optimizer step 2000. No optimizer, sampler, guidance, diffusion checkpoint, training seed change or scheduler submission.
