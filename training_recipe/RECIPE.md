# How the released 50,000-step H16 checkpoint was trained

**The released weights descend from `cw9jrqm0` at 46,937, then jobs `6078067` → `6080332` → `6083383`.** The September 13–14 four-job continuation also reached 50,000, but used a generic prompt and is not this checkpoint. This is a record of the team's recipe, not a launcher or a claim that it has been reproduced. [Run status](../reproduction.md): no runs yet.

| Part of the pin | Where to find it | What was checked |
|---|---|---|
| H0, including interruptions | [Five resolved config records](stage1/README.md), [restart order](stage1/restarts.tsv), [setting changes](stage1/changes.tsv) | All 43 logged dictionaries, their signatures and adjacent differences. Not all restore payloads or checkout identities are known. |
| H16, including recovery | [Three resolved configs and receipt chain](stage2/README.md) | Parent receipt digests, training revisions, last durable steps and native config digests, not published tensor bytes. |
| Data for both stages | [55 roots, per-stage variants, normalizers, index and exclusions](DATA.md) | Selected paths and available profile checksums. A path, a table hash and a whole-root integrity check are three different things. |

The public reference point is `14d1626700b532e1dad27c0f99f0d46389b5234b`. The team's [training differences](../cosmos3_h_embodied/training_differences.md) explain why starting with the public H16 experiment, a public-only mixture or the released weights does **not** reproduce the two stages. The config files here preserve the settings that were logged, in public form; site-specific locations and implementation symbols are translated or omitted as documented beside each set. Use the original code at its recorded revision, not these files as a replacement trainer. Private paths and original records are indexed by name in `released-h16-checkpoint.md` and the three evidence packets below.

## 1. H0: Cosmos3-Nano → 46,937

Start with Cosmos3-Nano and a **fresh optimizer**, no action history, C24 at resolution 256. Use all 15 families on 32 data-parallel ranks, accumulation 2. The resolved optimizer is FusedAdam: base LR `2e-4`, betas `(0.9, 0.99)`, weight decay `0.05`; `action2llm`, `llm2action` and `action_modality_embed` have 5× LR. LambdaLinear has a single 50,000-update cycle: warm up for 100 steps from factor `1e-6` to `0.4`, then decay to `0.05`. Do not restart that schedule at the H16 boundary. **Checked:** `cw9jrqm0` config and [stage-one snapshots](stage1/README.md); the defaults selector alone does not identify the optimizer or scheduler that ran.

Keep H0's original packing loader, idle-frame counts in captions (`append_idle_frames: true`), Hamlyn/UCSD XYZW interpretation and matching normalizers, and the older Stanford peg-transfer root. These are properties of the baseline, not defects to correct retrospectively. The logical paths alone do not capture changes such as the SMARTS camera overlay at 2,200. **Sources:** run config, dated media handoff and [training differences §0](../cosmos3_h_embodied/training_differences.md#0-training-history-of-the-released-checkpoint).

| Historical boundary | What the evidence establishes |
|---|---|
| Launch, job `5972250` | The launch audit names `d3183b2db52afbde536ee662696cd5418f2002d9`. The final run-level commit was **not** the Nano launch commit. |
| First 14 starts | A: initial settings; B: optimizer removed from skip list; C: final save and callback family names; D: temporary profiling settings; back to C. Starts 2–10 have resolved settings but lack direct per-job source bindings. |
| Save 2,200 → 3,669 | Job `5982753` is the first bound to `aeb0f03e8a652d346dcd20b24d88e6eb558b2ba8`. Camera overlay changed without renaming the roots. |
| Start 15 onward | On that base, job `5985107` restored 3,669 with the separate **246,596-byte** augmentation patch, SHA-256 `2c8139766ce19162478f573a3ffac5f04f79ffd9f7978dcda066c9a67bce643c`; no separate patch commit is evidenced. E enables all-family GPU augmentation and one-frame chunks. The job reached 3,826; after a no-update retry, `5985164` resumed **3,800**, not 3,826. |
| Finish, job `6020929` | Trainer iteration and Adam step both reached 46,937. The historical `iter_000046937` listing contains **164 files / 147,048,365,668 bytes**, ranks 0–31: model, optimizer, scheduler, trainer and per-rank loader state. Listed, not load-tested. |

**Sources:** `released-stage1-records` (launch audit, timed configs, handoffs and job receipts); `reproduce-last-stage/run-plan.txt` for the original order; checkpoint inventory in `slide-27-evidence.txt`. The [restart ledger](stage1/restarts.tsv) distinguishes job-specific confirmation, inference and unknowns. A future rebuild must save **its own** complete 46,937 and preserve its state for Stage 2. The historical parent is a comparison, not a shortcut.

## 2. H16: our full 46,937 state → 50,000

Restore model/EMA, optimizer, scheduler, trainer/RNG and rank-local dataloader positions. H16 uses 16 causal past actions, original dataset task/instruction prompts (no fixed sentence), no future idle-frame count, WXYZ for Hamlyn/UCSD with the matching normalizers, and the row-index repair and updated normalizer for Stanford. The profile is `openh-exp232-wxyz-h16-v1`; wound closure requires `video_source: clean`. Preserve the trained caption-prefix behavior, not the corrected default. **Checked:** the contributing job configs, profile and [public reference differences](../cosmos3_h_embodied/training_differences.md).

| Contributing job | Start → last durable save | Updates retained | Source and public config |
|---|---|---:|---|
| `6078067` | 46,937 → 47,437 | 500 | `6f8eb4dc` · [pilot](stage2/6078067.yaml) |
| `6080332` | 47,437 → 49,000 | 1,563 | `2be2de74` · [full allocation](stage2/6080332.yaml); reached 49,008 at deadline, lost eight updates |
| `6083383` | 49,000 → 50,000 | 1,000 | `86ac356c89a08623471d22d216dfddec6236a4e4` · [recovery](stage2/6083383.yaml) |

The registrations bind each parent path **and** its receipt digest. The last receipt records 1,000 updates and 32 ranks at 50,000. Job `6079595` was cancelled while pending; `6083331` failed preflight. Neither trained. The three contributing configs share data, model and optimization settings but differ in orchestration. The four comparison jobs (`6022574`, `6024589`, `6024787`, `6025164`) share only the 46,937 parent with the release. Their loss curves, intermediate checkpoints and final YAML must not fill any release result. **Attribution:** Lukas identifies the 50,000 checkpoint of `6083383` as the source of the published regular network weights; lineage is settled by the receipts. A tensor comparison from that checkpoint through export to the public weights remains open. See [native hashes, differences and evidence](stage2/README.md).

## Pass checks, fixed before a rebuild

1. **At 46,937: compare training trajectories, not a single raw loss.** Align `train@2_detail/flow_matching_loss_action` with `cw9jrqm0` on optimizer step. Compare the full curves with the same centered 21-point mean of the 20-step log, clipping the window at each end. Tabulate unsmoothed means within ±50 steps of 1k, 5k, 10k, 25k and 46.9k. The reference rounds to **0.0204, 0.0134, 0.0107, 0.0073, 0.0049**. The last number is the mean of four observations at 46,860–46,920 (`0.00493101`), **not** a logged loss at 46,937 or the centered smooth at 46,900 (`0.0048045`). Checked directly against the 2,346-point `hist_cw9jrqm0.json` archive and the milestone rule in the 26 Nov recipe notes. Require an intact 46,937 save and comparable shape and level throughout, without unexplained restart jumps or a growing gap. There is only one historical trajectory, not a measured H0 run-to-run tolerance; do not declare a statistical match on four-decimal endpoint agreement. Hold for review if the trajectory disagrees. The matched evaluation against the historical 46,937 parent is a separate pending check.
2. **At 50,000: match the released checkpoint on the same held-out clips.** Use the [README single-clip evaluation](../cosmos3_h_embodied/README_EVALUATION.md#all-modes-on-held-out-clips), not its long-rollout benchmark. Freeze the clip list/order, family normalizers, script revision and seed schedule before comparison. Use 64 spread validation clips × 10 families = **640 clips**, 35 sampling steps, guidance 6.0; FD PSNR includes the conditioning frame, ID and policy MSE use 24 normalized action rows. Printed targets: **FD PSNR 26.09 dB; ID MSE 0.0087; policy MSE 0.0518 (mean of four rollouts)**. Best-of-four `0.0331` is not that target.
3. **Measure the noise floor on the release first.** The evaluation owner's plan (`slide-10-evidence.txt` and `slide-28-evidence.txt`) names base seeds `20260908`, `20260909`, `20260910`; report per-seed values, mean and range (max − min) for each metric. Keep the four-rollout policy statistic distinct from this three-seed variation; compare like aggregations on the same clips and seeds. Check the release against the README before using it as a ruler. For each of the three headline metrics, compare rebuilt and released scores against the measured metric-specific range, reporting the signed differences (higher is better for PSNR, lower for MSE). A difference beyond the noise floor, a changed clip set or an unexplained release/README mismatch is **not a reproduction pass**. No range has been measured and no rebuild score is reported here.

## Compute and storage

**Planning estimate, not a measured run.** The rebuild plan is 8 nodes × 4 GPUs, or 32 simultaneous ranks. The archived final H0 job instead reports 4 nodes × 8 GPUs; the rank count agrees, placement is not the same. At the plan's 4.69 s/update, H0 takes 61.15 h and H16 3.99 h: 65.14 h (~2,084 GPU-hours) of useful updates. Budget about three days before evaluation; replays, restarts, I/O and queue delays are not included in the rate. Source: `run-plan.txt`, final H0 metadata.

**Observed storage, decimal bytes.** The copy inventory sets a lower bound of **659,787,218,915 B** for H0 roots and **659,787,219,901 B** for H16 roots; 44 H16 roots have exact sizes, six have incomplete media inventories, five have no size under the current policy. Do not sum both stage totals: 54 source paths are shared, and linked media must be counted once per physical copy. Start weights, sidecars, temporary writes and a separately retained comparison checkpoint also need space.

The team's checkpoint inventory of 23 Sep at 12:27 UTC (compressed inventory SHA-256 `f4aa559aadf60a6cd3f9322e27aa60f1a8df98fc74fe9c6a1689bbefe3d7bc85`) gives a **lower bound for the corrected lineage**, not the final peak:

| Source at snapshot | Saves present | Logical bytes |
|---|---:|---:|
| H0 through 46,937 | 55 | 8,087,660,120,216 |
| `6078067` | 7 | 1,029,338,566,009 |
| `6080332` | 16 | 2,352,773,858,517 |
| `6083383`, **only through 49,200 at snapshot** | 2 | 294,096,731,665 |
| **Coexisting release-chain saves** | **80** | **11,763,869,276,407** |

Checked by selecting only the four output checkpoint roots, deduplicating by device/inode: 13,120 distinct files, 11,763,989,098,496 allocated bytes. **Saves plus just the known H0 inputs already exceed 12.42 TB**, before the missing bytes or later saves. The earlier 91-save, 13.381-TB count used the September 14 comparison campaign; it is not the release's retention count. Neither clears the explicit 10-TB run check. The separate shared project quota is 100 TB (escalate before 85 TB), not a reservation or a change to the run check. Sources: `release-data-copy`, team's inventory, `storage-audit-20261127T0145.txt`. Do not assume a new pruning policy.

## What is missing before starting Stage 1

- **Inputs and weights:** no whole-root sign-off. At the 27 Nov 02:38 PT receipt check, 29 of 55 H16 packages had arrived: 20 with the listed regular files and no known missing video link; nine lacked their linked video. Twenty more roots exceeded the 10-GB transfer limit, five CMR paths were policy-blocked, and the H16 Stanford repair was pending. The old Stanford copy is separate. Nine repaired H16 table pins, the H0 Stanford table pin and the exclusion file matched; none checks all modalities. The clinical index/normalizer remain blocked; other normalizers and direct profile records await receipts. Source: `release-data-copy` copy table and receipt check.
- **Gate:** the copy owner's gate failed on all 57 intended paths (55 H16, extra H0 Stanford, Nano). The evaluation owner separately requires verified release weights and its own inputs. Only the data owners can certify whole roots; differing team copies, including SutureBot mono, remain held. No public or newer copy, missing video, altered timestamp, dropped root or hand-built repair is an acceptable substitute. Before any future submission enumerate the actual config's roots and check every stamp, plus start weights; a failure stops the run.
- **H0 provenance:** bind the unresolved early jobs/commits and remaining parent saves, restart states, worker/RNG and effective media mounts to original records. The delivered augmented tree is short seven policy-blocked files. The final config/commit cannot stand in for earlier starts. Sources: `released-stage1-records` and `reproduce-last-stage`.
- **H16 provenance:** the job-lineage question **is settled**. Direct offline histories, direct source-record comparisons and public tensor/export equality are separate outstanding checks, not evidence for the old four-job plan. For the matched 46,937 evaluation, the historical checkpoint has been listed, not delivered or loaded.
- **Execution:** obtain exact input and peak-save space, clearance for the corrected plan, and the pre-run release baseline. No training or evaluation is authorized by this pin.

**Next, in order:** (1) evaluate the verified released weights ourselves on the frozen README panel and measure the three-seed floor; (2) rebuild H0 from verified Nano inputs and compare at our own 46,937; (3) continue that full state through the three H16 segments to 50,000; (4) compare both checkpoints and report pass or discrepancy before proposing any recipe change.
