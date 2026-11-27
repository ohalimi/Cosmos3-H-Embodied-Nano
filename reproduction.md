# Reproduction

## `release-all-modes` — not submitted

This single evaluation of the **released H16 checkpoint at step 50,000** serves slide 10 (“Step 50,000: the release's public eval is the target”) and the “Release, our eval” column of slide 28. It is a plan, not a measured result. It does not evaluate a rebuild, a plus-families checkpoint, or step 46,937.

| Record | Value |
|---|---|
| Evaluation branch / pinned code | `ohalimi/release-all-modes` / `48fd710163aca224e5f1a65e55132a16b8460c48` (clean detached checkout required) |
| Framework | `cosmos-framework`, `7cfaf7854b2114eda5deb36250a7e8f00c64d69e` at preflight; record the actual revision at launch |
| Start checkpoint | Released `Cosmos3-H-Embodied-Nano/h16_50000_dcp/model` (DCP) |
| Local launcher SHA-256 | `afae9dadca2af5c48d7501fb4232752c2a94edfd2e54abdf0187eb798c16b466` (three joint seeds plus a fourth policy rollout) |
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
    mode=wam; if ((s < 3)); then mode=joint; fi
    python -m cosmos3_h_embodied.scripts.eval --no-guardrails --checkpoint-path "$ckpt" \
      --config-file cosmos3_h_embodied/configs/config.py --experiment openh_h16_task_prompt \
      --dataset.dataset "$family" --dataset.model-mode "$mode" --dataset.dataset-split val \
      --dataset.num-samples 64 --spread-samples --seed "$((20260908+s))" \
      --num-steps 35 --guidance 6.0 -o "$out/seed$s/$family"
  done
done
```

The 40 invocations use the same 640 distinct clips: 64 per family, 25 frames and 24 predicted action rows per clip. Seeds `20260908`, `20260909`, and `20260910` each run **all three modes** to measure the slide 28 noise floor. Seed `20260911` adds **policy only**, retaining the README's four policy rollouts without counting the fourth seed in that noise floor. Every invocation uses 35 sampling steps and guidance 6.0.

**Scoring.** Use `load_runs`, `seed_spread` and `summarize` from the pinned `tools/benchmark/summarize_eval.py`. For each of the first three seeds in turn, validate its complete joint result against the other three seeds' policy rows with `seed_spread` (64 matched clip IDs per family, finite scores, identical policy ground truth). Reject missing or extra modes, duplicate scores, or changed clips; do not silently average partial outputs.

From the clean pinned checkout, with `out` and `families` as above, the score command is:

```bash
python - "$out" "${families[@]}" <<'SCORE'
import json, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from tools.benchmark.summarize_eval import load_runs, seed_spread, summarize, write_table

out = Path(sys.argv[1]); families = set(sys.argv[2:])
runs = [load_runs([out / f'seed{s}']) for s in range(4)]
policy = [{key: v for key, v in run.items() if key[1] == 'wam'} for run in runs]
if runs[3].keys() != policy[3].keys():
    raise ValueError('fourth seed must contain policy only')
for s in range(3):
    aligned = [runs[s], *(policy[k] for k in range(4) if k != s)]
    spread = seed_spread(aligned, families)
    if s == 0:
        policy_seeds = spread  # order 0,1,2,3
        readme = defaultdict(list)
        for run in aligned:
            for key, rows in run.items():
                readme[key].extend(rows)
        table = summarize(readme)

per_seed = [summarize(run)['All'] for run in runs[:3]]
floor = {'seeds': [20260908, 20260909, 20260910]}
for key in ('fd_psnr', 'id_mse', 'policy_mse_mean'):
    values = [row[key] for row in per_seed]
    floor[key] = dict(values=values, mean=mean(values), range=max(values)-min(values))
summary = out / 'summary'; summary.mkdir(exist_ok=True)
for name, data in [('eval_summary', table), ('policy_seeds', policy_seeds), ('noise_floor', floor)]:
    (summary / f'{name}.json').write_text(json.dumps(data, allow_nan=False, indent=2) + '\n')
write_table(table, summary / 'eval_summary.md')
SCORE
```

- **Slide 10, README comparison:** FD PSNR and ID MSE from seed `20260908`. Across all four policy rollouts, report mean MSE and PSNR, per-clip best of four (minimum MSE, maximum PSNR), and each seed's clip mean. Use the pinned scorer's baseline per family and sample-weighted All baseline. If the measured All baseline differs from 0.0747 at the README precision, report a clip-set difference; never substitute README reference scores for measurements.
- **Slide 28, release at 50,000:** for each joint seed separately, take the All-row FD PSNR, ID MSE and policy MSE on the same 640 clips. Record the three values for each metric, their arithmetic mean, and their range (maximum minus minimum). This range is the empirical noise floor, not a confidence interval. Keep this three-seed summary distinct from slide 10's single-seed dynamics and four-seed policy.
