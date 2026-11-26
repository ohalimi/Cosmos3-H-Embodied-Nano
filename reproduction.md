# Reproduction

## `release-all-modes` — not submitted

**Slide 10: “Step 50,000: the release's public eval is the target.”** This entry is for the *released* H16 checkpoint at step 50,000. It is a plan, not a measured result. The plus-families checkpoint is not part of this run.

| Record | Value |
|---|---|
| Evaluation branch / pinned code | `ohalimi/release-all-modes` / `48fd710163aca224e5f1a65e55132a16b8460c48` (clean detached checkout required) |
| Framework | `cosmos-framework`, `7cfaf7854b2114eda5deb36250a7e8f00c64d69e` at preflight; record the actual revision at launch |
| Start checkpoint | Released `Cosmos3-H-Embodied-Nano/h16_50000_dcp/model` (DCP) |
| Local launcher SHA-256 | `2c11666d6d5e048fb46aaa0687319bca37934d34582f9b1d30d4dce2b3197ff7` |
| Job IDs, including retries | None; not submitted |
| W&B | Configured entity/project: `holoscan-advanced-workflow-team/cosmos3_suturebot_memory`; no run ID or run URL |
| Runtime diff | None; the pinned evaluation checkout passed the clean-tree check |

### Inputs and submission gate

Use the public roots in `cosmos3_h_embodied/configs/openh_datasets.py` at the pinned commit. In the table, each suffix denotes one distinct directory under its prefix, relative to `OPENH_DATA_ROOT`. Do not include nonpublic roots. `dvrk_rob_surgical` has no held-out episodes and is not evaluated.

| Family | Root prefix | Suffixes |
|---|---|---|
| `cmr_clinical` | `cmr_surgical/` | `cholecystectomy`, `hysterectomy`, `inguinal_hernia`, `prostatectomy` |
| `dvrk_hamlyn` | `hamlyn/` | `knot_tying`, `needle_grasp_and_handover`, `peg_transfer`, `suturing_1`, `suturing_2`, `tissue_retraction` |
| `dvrk_jhu` | `jhu/` | `imerse/wound_closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled`, `imerse/nephfat/nephfat`, `imerse/srt_tissue_lift`, `imerse/cao_cautery_combined`, `imerse/srth_porcine_chole`, `lcsr/arcade/cholecystectomy` |
| `dvrk_obuda` | `obuda/` | `pork_1`, `frs_dome_1`, `needlethreading_1`, `needlethreading_2`, `pegtransfer_1`, `pegtransfer_2`, `rollercoaster_1`, `seaspike_1`, `seaspike_2`, `seaspike_3`, `skinphantom_1` |
| `dvrk_smarts` | `jhu/lcsr/smarts/SurgSync-stitch-coldcut/` | `P1`, `P2`, `P3` |
| `dvrk_stanford` | `stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/` | `peg_transfer`, `needle_transfer`, `tissue_retraction` |
| `dvrk_tud_grasping_retraction` | `tud/260131_tundra_dataset/` | `grasping_retraction` |
| `dvrk_ucberkeley` | `ucberkeley/` | `debridement_lerobot` |
| `dvrk_ucsd` | `ucsd/` | `surgical_learning_dataset2`, `surgical_learning_dataset` |
| `turin_mitic` | `turin/` | `mitic_lerobot_plastic_tube`, `mitic_lerobot_ex_vivo`, `mitic_lerobot_plastic_pad`, `mitic_lerobot_plastic_pad_3dmed` |

Submit once, only after the data owner publishes **all 41 prepared roots** and the nonempty `indices/cmr_valid_windows_c24_stride8_reject_any_v2.npz`, and the pinned launcher's `--check-data` exits 0. At 15:15 PT on November 26, 2026 it exited 3: only 5/41 roots (all JHU) and no index. Raw downloads do not satisfy this gate.

**Split:** use `val` as resolved by the pinned release loader, including its shipped exclusions; no custom split or overlay. The release config uses split seed 42 and a 0.05 validation fraction. **Realized split/clip manifest SHA-256: pending.** There is no complete manifest of the evaluated clips before publication and evaluation. A hash of the config would not identify the actual clips.

### Planned invocation and scoring

Run from the pinned checkout in the matching framework environment. The placeholders below are public data, the released checkpoint directory, and a new output directory respectively. The one-node launcher distributes these independent family/seed invocations across four GPUs; it does not shard a single rollout. Its planned submission command is `sbatch <launcher-with-SHA-256-above>`.

```bash
export OPENH_DATA_ROOT='<prepared-public-data>'
ckpt='<released-h16-50000-dcp-model>'
out='<evaluation-output>'
families=(cmr_clinical dvrk_hamlyn dvrk_jhu dvrk_obuda dvrk_smarts dvrk_stanford dvrk_tud_grasping_retraction dvrk_ucberkeley dvrk_ucsd turin_mitic)
for family in "${families[@]}"; do
  for s in 0 1 2 3; do
    mode=wam; if ((s == 0)); then mode=joint; fi
    python -m cosmos3_h_embodied.scripts.eval --no-guardrails --checkpoint-path "$ckpt" \
      --config-file cosmos3_h_embodied/configs/config.py --experiment openh_h16_task_prompt \
      --dataset.dataset "$family" --dataset.model-mode "$mode" --dataset.dataset-split val \
      --dataset.num-samples 64 --spread-samples --seed "$((20260908+s))" \
      --num-steps 35 --guidance 6.0 -o "$out/seed$s/$family"
  done
done
python tools/benchmark/summarize_eval.py --runs "$out"/seed{0,1,2,3} \
  -o "$out/summary" --all-modes-families "${families[@]}"
```

This is 64 evenly spread validation clips per family, 640 distinct clips total (25 frames and 24 action rows each). Seed 0 (`20260908`) evaluates forward dynamics, inverse dynamics and policy; seeds 1–3 (`20260909`–`20260911`) add policy rollouts on the same clips. All use 35 sampling steps and guidance 6.0. Report FD PSNR, ID MSE, policy MSE and PSNR (mean and per-clip best of four), baseline MSE, and the spread across policy seeds. Report a mismatch from the published All baseline MSE of 0.0747 as a clip-set difference; do not substitute the published values for measurements.
