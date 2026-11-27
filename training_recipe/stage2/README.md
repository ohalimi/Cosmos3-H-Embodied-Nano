# Stage 2: H16 task prompts, 46,937 → 50,000

The three YAML files are the **public form of the resolved training configs**. They record the original settings, not launch instructions. The update counts below are updates retained in the published lineage, not simply updates observed by a logger. `86ac356c` is the **final** job's source commit, not the source commit of all three jobs.

| Job / config | Training source commit | Input → last update seen → retained checkpoint | Retained updates |
|---|---|---|---:|
| [6078067](6078067.yaml), pilot | `6f8eb4dcd41ed7e0170d115d6bc88cefd638c398` | Stage 1 at 46,937 → 47,437 → 47,437 | 500 |
| [6080332](6080332.yaml), full allocation | `2be2de744a862aa6e8bfbcb2b5583c550c36ea7a` | 6078067 at 47,437 → 49,008 → 49,000 | 1,563 |
| [6083383](6083383.yaml), recovery | `86ac356c89a08623471d22d216dfddec6236a4e4` | 6080332 at 49,000 → 50,000 → 50,000 | 1,000 |

All three use a strict full-state restore: `load_training_state: true`, empty checkpoint skip lists, no scheduler-only load, and an explicit expected starting trainer iteration **and** Adam step. The completion records include model, optimizer, trainer, scheduler and 32 dataloader states. In a rebuild the initial input is **the rebuild's own** complete Stage 1 checkpoint; each later job takes the corresponding save from that same chain. The numbers alone do not identify the input.

**Checked:** the registered parent of 6080332 names the 47,437 pilot checkpoint and pins the pilot completion receipt (`4e7863794a52a5ff4d22166de6f483521e4e7af0df6a40916fe08b30deb53b30`, SHA-256). The parent of 6083383 names the recovered 49,000 checkpoint and pins its recovery receipt (`33a5a35ab5d6682948857e029832a014cdbd5f6aed20b22b767df851c211eb8d`). Both receipt digests match the archived records. 6080332 did 1,571 updates, then failed at its deadline; eight unsaved updates had to be replayed. A planned job, 6079595, was cancelled without starting. Job 6083331 (source `ac86f39f7901e892d23d83924040be61ba5ace33`) failed preflight before training; its registered YAML is semantically identical to the recovery job's. Neither contributed weights.

**Published checkpoint:** Lukas Zbinden identifies `iter_000050000` from 6083383 as the published source. Its completion receipt (`83c31402d2f0fc476bd58b53aefa1d2e3c036bf6b5f01a4b1a4fb5e902941477`) records 1,000 updates and 32 dataloader states. The registration, recovery and completion records establish the job lineage. They are **not** hashes of the published weight tensors; no equality check against those tensors is claimed here.

## Why the September 13–14 jobs are not the release

6022574, 6024589, 6024787 and 6025164 were a different continuation from the same 46,937 parent. Do not take their 47,437 or 49,000 save, or use their loss history as that of the released checkpoint. At the same 49,000 start, the resolved config of 6025164 (native SHA-256 `83cad5876c21783381780c9a1a9d27d64ef2d906fe0d4193829593bab8becd96`) differs from 6083383 at exactly 17 settings: the 15 family-level overrides `fixed_instruction: Continue the ongoing surgical task.` are absent in 6083383, the intermediate 49,050 milestone was removed, and `H16FullAllocationCallback` replaced `H16ProductionTrialCallback`. The release reads each dataset's task or instruction text; both continuations had `append_idle_frames: false`.

Within the actual release chain, the 6078067 → 6080332 config changes six settings: milestone list, job name, history-exposure callback (`H16TaskPromptCallback` → `H16FullAllocationCallback`), expected start, expected Adam step and trainer maximum iteration (47,437 → 50,000). From 6080332 → 6083383 only milestones, job name, expected start and expected Adam step change. Optimizer, scheduler, model and data settings agree in all three. This comparison is of resolved settings, not code identity.

## Sources and public projection

Sources: the team's stage registrations and their embedded YAML, the native invocation digests, the pilot completion, recovery and final completion receipts; see the private `released-stage2-records` packet, especially `jobs.tsv` and `config-and-receipt-provenance.tsv`. For published-weight attribution see the 27 Nov exact-training-data record and the private `released-h16-checkpoint.md` index. These native SHA-256 values identify the **original**, not the projected files:

| Job | Native resolved YAML SHA-256 | Independent digest available from |
|---|---|---|
| 6078067 | `7f8e44d3b9ac40338feb97e28c5a326543483332a9e13e7001055597146cae58` | Training completion |
| 6080332 | `a1fb4598d975375606ff9b1307d1a3cc3e81902792ecb6e3339a17f257c48833` | Native dry-run receipt; the failed job has no training completion |
| 6083383 | `3566a32c6439e7fc1ce0609d7d255886911320412f262166747adf4eedc56e89` | Training completion |

The records owner recovered the native byte order from the registered values and verified it against each invocation digest. Before projection each digest was checked again, and the native and registered YAMLs parsed to the same values. The public YAML preserves the keys, list order and scalar types. Per file, 160 string values were translated: checkout-specific import targets/types are module and symbol labels, not resolvable imports; the model-config reference is its filename; a disabled weights-store location and tracking names are symbolic. `<parent-checkpoint>` means the input in the table. No command, launcher, receipt, episode, parameter tensor or normalizer payload is included.

The 55 `roots` in each file are **relative to the prepared data directory**. Each was checked against its family's ordered entry in [`openh_datasets.py`](../../cosmos3_h_embodied/configs/openh_datasets.py). This includes the nonpublic roots; a public *path name* does not mean public *bytes*. The index is relative to the same directory; the exclusion-list reference is relative to the repository. Family normalizer filenames are unchanged. All three configs require clean wound-closure video and the repaired Stanford peg-transfer root. The release's original locations and variants remain in the private provenance record. Projection does not authorize substitution, downloading data or running these configs.
