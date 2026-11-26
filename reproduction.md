# Reproducing the two-root recipe check

## nano-h16-suturebot-ref32 — slides 18–20, Same, 32 × 2

- State: submitted, awaiting first logged step. Training job: **445512** (first attempt; no resumes).
- Branch: `ohalimi/nano-h16-suturebot-ref32`. Clean training commit: `e3ca2a82a7442c3aacced51d4d79e198c4b95602`, based on the packed sigma logger fix `5433f75f1421670963c9c99ee17b9cd734767251`. Framework: `7cfaf7854b2114eda5deb36250a7e8f00c64d69e`. No uncommitted training diff.
- Forge W&B: `holoscan-advanced-workflow-team/cosmos3_suturebot_memory`; ID and link pending trainer initialization. The run's repro record contains the exact expanded training command and the launch wrapper.
- Launch command: `sbatch <project-root>/scripts/<launch-wrapper>`; launch wrapper SHA-256 `805c68286dcba7f81005a971e29ad831ccdc62f4e331ee484960d1d566fee15e`. Export `cosmos3_h_embodied/configs/config.py` with experiment `nano-h16-suturebot-ref32`; start 8 nodes × 4 GPUs, shard degree 32, accumulation 2.
- Weights: **Cosmos3-Nano DCP**, not the released H16 weights. Trainer seed 0; deterministic loader and split seed 42. Train and validate only on `jhu/hf_suturebot` (family `dvrk_suturebot_mono`, ratio 0.28775) and `jhu/imerse/wound_closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled` (`dvrk_jhu`, ratio 2.5928). Frozen `val-episodes.json` SHA-256 `c06581a2348a981eb0100960daedb0fd684cef8539e9dc1a75961a761a6133f4`; held-out fraction 0.05.
- `legacy` captions on both splits; validation enabled on start and every 100 steps. H16, chunk 24, resolution 256; `as_trained` LR schedule, optimizer LR 0.0002, factors 0.4 → 0.05 over 50,000 iterations, cap 50,000. Save every 100; protected milestones 100, 1,000, 5,000, 10,000, 25,000 and 50,000.
- Held-out evaluation (planned, not yet scored): joint/WAM, 32 validation samples; seeds 20260908, 20260909 and 20260910, 35 sampling steps, guidance 6.0 and three rollout chunks. Compare only checkpoints at the same milestone.
