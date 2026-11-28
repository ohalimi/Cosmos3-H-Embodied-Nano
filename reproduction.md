# Reproduction

## `release-all-modes` — running (released step 50,000; retry job 459480)

Job `459480` started at 19:24 PT on 27 Nov 2026. One submission covers all three README repeats. Its four workers share one `HF_DATASETS_CACHE` at `<attempt-output>/runtime/datasets`, outside all release inputs and tokenizer snapshots. Attempt 1’s outputs were not reused. Only the runtime storage changed; clips, checkpoint, seeds and scoring command are unchanged. The retry launcher (SHA-256 `751c56bd45a2718866fbafe9789aa65457c350697ef7f3b967de0eb52dde648d`) requires at least 1 TiB and 100,000 free inodes before submission and at job start. `bash <output-cache-launcher> --preflight` passed on all 67 inputs (54 release-side, 11 packaged, two tokenizer snapshots); `sbatch <output-cache-launcher>` submitted the retry. **No measured scores at submission.**

Job `459308` was the first submission for the “Release / our eval” column (slides 44–45; earlier slides 10 and 28). It passed the input gate and started four workers, but all four stopped on the first family (`cmr_clinical`) while the loader wrote its Arrow cache to node-local temporary storage: `OSError: [Errno 28] No space left on device`. Slurm recorded `FAILED`, exit `1:0`, after 10 min 21 s. **No clip scores or aggregate metrics were produced for any repeat.** No rebuild or plus-families checkpoint was scored.

| Record | Value |
|---|---|
| Branch / clean evaluation commit | `ohalimi/release-all-modes` / `48fd710163aca224e5f1a65e55132a16b8460c48` |
| Framework commit | `7cfaf7854b2114eda5deb36250a7e8f00c64d69e` |
| Start checkpoint | Released native `iter_000050000/model`, regular `net.*` (not EMA). Native regular tensors match the published BF16 network. |
| Launcher SHA-256 | `e44198fcf1ca2def05a83bd315f42c1ef745cbaa7311a3e78e88ed9622e8472f` (one node, four single-GPU workers). Reads the published offline Qwen `0c351dd01ed8` and Wan `921dbaf3f167` snapshots from the release layout. |
| Jobs / W&B | Forge `holoscan-advanced-workflow-team/cosmos3_suturebot_memory`: Slurm `459308`, failed, run [`stocjz6a`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/stocjz6a); Slurm `459480`, running, run [`omx7bgeq`](https://app.forge.coreweave.com/holoscan-advanced-workflow-team/cosmos3_suturebot_memory/runs/omx7bgeq). |
| Classification / signals | **Provisional, kosher light** until all inputs are stamped. Internal signals deferred: no proven native-kernel hooks, 0 bytes planned. |

### Inputs and submission gate

Set `OPENH_DATA_ROOT` to the release team's processed copies, laid out at the public relative paths in `cosmos3_h_embodied/configs/openh_datasets.py` at the pinned commit. The table selects 41 roots, not the whole training mixture. Use the release's repaired copy where specified (including row-index-repaired Stanford peg transfer) and its clean wound-closure stream. Do not substitute a local rebuild or a labeled video for a missing input. Each suffix below denotes one directory under its prefix. Exclude nonpublic roots; `dvrk_rob_surgical` has no held-out episodes and is not evaluated.

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

**Gate:** require the complete public layout and the nonempty CMR valid-window index. Run `data_gate.py roots <launcher>`, then `data_gate.py check --scope released-recipe --light <kosher-light-release.tsv> <resolved-inputs>`. Expand the 41 public roots from the pinned family config; also check the native model, the CMR index file, and the evaluation normalizers and exclusions against their release evidence. The offline evaluator also opens the Wan VAE and Qwen tokenizer: check the actual snapshot paths against owner-supplied source evidence. A pass on the release table alone does not clear inputs it does not list. Require all checks to pass immediately before the single submission and again at job start. No downloading, preparing data, or substituting weights.

The README evaluator opens the normalizers and relative-path exclusion list shipped with the public package. Gate those actual files using their upstream Git blobs at `14d1626700b5`; do not substitute the training overlay, whose absolute paths would not select the same exclusions here. For the 41 public evaluation roots the two overlays select the same 301 excluded root/episode pairs. The numeric statistics loaded by the evaluator match the release sidecars.

**Split:** shipped `val` (seed 42, fraction 0.05), exclusions and CMR windows unchanged. Realized clip-manifest hash: pending. One panel of 64 evenly spread clips per family × 10 families = 640. Each clip has 25 frames and 24 predicted action rows.

### Command and scoring

Run the [README all-modes protocol](cosmos3_h_embodied/README_EVALUATION.md#all-modes-on-held-out-clips) three times. The first repeat is the published seed sequence. Each repeat has one FD/ID rollout and four policy rollouts, **not** one policy rollout compared with the published four-rollout mean. Explicitly disable EMA when reading the native DCP: in the published evaluation DCP the regular weights occupy both network slots; that is not true of the native format. Keep all other inference settings unchanged.

```bash
export OPENH_DATA_ROOT='<team-processed-release-data>'
ckpt='<released-native-iter_000050000-model>'; out='<evaluation-output>'
families=(cmr_clinical dvrk_hamlyn dvrk_jhu dvrk_obuda dvrk_smarts dvrk_stanford dvrk_tud_grasping_retraction dvrk_ucberkeley dvrk_ucsd turin_mitic)
for base in 20260908 20260909 20260910; do
  for family in "${families[@]}"; do
    for s in 0 1 2 3; do
      mode=wam; if ((s == 0)); then mode=joint; fi
      python -m cosmos3_h_embodied.scripts.eval --no-guardrails --no-use-ema-weights --checkpoint-path "$ckpt" \
        --config-file cosmos3_h_embodied/configs/config.py --experiment openh_h16_task_prompt \
        --dataset.dataset "$family" --dataset.model-mode "$mode" --dataset.dataset-split val \
        --dataset.num-samples 64 --spread-samples --seed "$((base+s))" --num-steps 35 --guidance 6.0 \
        -o "$out/$base/seed$s/$family" --experiment-overrides \
        "model.config.tokenizer.vae_path=<verified-Wan-VAE-file>" \
        "model.config.vlm_config.tokenizer.pretrained_model_name=<verified-Qwen-tokenizer-folder>"
    done
  done
  python tools/benchmark/summarize_eval.py --runs "$out/$base"/seed{0,1,2,3} \
    --all-modes-families "${families[@]}" -o "$out/$base/summary"
done
```

For each base seed report all ten family rows and All, with the spread of its four policy rollouts. Require 64 matched finite scores per expected family/mode/rollout and identical policy ground truth. Compare the three All rows and report their mean and max minus min. The consecutive base seeds produce overlapping policy seed quartets; the policy range describes these three repeats, **not** three independent four-rollout estimates or a confidence interval.

| Published reference only | FD PSNR | ID MSE | Policy MSE mean / best of 4 | Policy PSNR mean / best of 4 | Baseline MSE |
|---|---:|---:|---:|---:|---:|
| README All (640 clips) | 26.09 dB | 0.0087 | 0.0518 / 0.0331 | 23.00 / 23.88 dB | 0.0747 |

Flag a measured baseline different from 0.0747 at four decimals as a clip-set mismatch. Do not replace a measurement with its reference.
