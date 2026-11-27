# Which data went into the two stages?

Profile `openh-exp232-wxyz-h16-v1` at `86ac356c89a08623471d22d216dfddec6236a4e4`: **58,067 bytes**, SHA-256 `ca7361e575a454355c484488f7f72bc0a2afb1dfd6958dfb0e485fe3368a188d`. The archived launcher pins that digest. The source profile identifies the training manifest by SHA-256 `2bc72304ca2a1529e63b4feed3c0167fe89cb3cb1d1657e780dce30daea1866a` and the loader revision by `b46ab8f6eb8a620d558929a901f918049544709c`. These are reference identifiers, not local stamps.

## Ordered inputs

Checked by comparing the profile’s selected values (not its accepted alternatives), the 55 paths in Lukas’s 27 Nov release record, the copy ledger, and the `cw9jrqm0` config. All 55 H16 source paths agree; 54 of 55 H0 source paths are the same. All paths below are relative to the prepared directory in [`openh_datasets.py`](../cosmos3_h_embodied/configs/openh_datasets.py), **not** a declaration that the bytes are public or locally present. Physical paths are kept in the private `released-h16-checkpoint.md` source index and Lukas’s record.

“Preserved” means the team’s August 28 snapshot; “release mono” is the dedicated SutureBot mono copy. The eleven repaired roots have linked video: the repaired tables alone are not a complete input. The SHA-256 column is the **reference H16 data-table digest** (ordered relative paths and bytes of `data/**/*.parquet`); it is **not** a hash of a whole root, its metadata or its video. A dash means the profile does not supply a repaired-table pin for this root, not that it passed. For row 28 the older H0 table pin is `db5149061c43d74511fa976ea7249eb62fdfeb99404355b6af66ac0b3b4aa8da`. Keep those two physical inputs distinct even though they have the same logical name.

| # | Family | Logical root in both stages | Source version (H0 → H16 if different) | H16 table SHA-256 |
|---:|---|---|---|---|
| 1 | `cmr_clinical` | `cmr_surgical/cholecystectomy` | preserved | — |
| 2 | `cmr_clinical` | `cmr_surgical/hysterectomy` | preserved | — |
| 3 | `cmr_clinical` | `cmr_surgical/inguinal_hernia` | preserved | — |
| 4 | `cmr_clinical` | `cmr_surgical/prostatectomy` | preserved | — |
| 5 | `dvrk_obuda` | `obuda/pork_1` | preserved | — |
| 6 | `dvrk_obuda` | `obuda/frs_dome_1` | preserved | — |
| 7 | `dvrk_obuda` | `obuda/needlethreading_1` | preserved | — |
| 8 | `dvrk_obuda` | `obuda/needlethreading_2` | preserved | — |
| 9 | `dvrk_obuda` | `obuda/pegtransfer_1` | preserved | — |
| 10 | `dvrk_obuda` | `obuda/pegtransfer_2` | preserved | — |
| 11 | `dvrk_obuda` | `obuda/rollercoaster_1` | preserved | — |
| 12 | `dvrk_obuda` | `obuda/seaspike_1` | timestamp-repaired | ab2b4c54346cd2ef62c0245665bf1d5c740c2c824bcb75fb7500eba4f8ade883 |
| 13 | `dvrk_obuda` | `obuda/seaspike_2` | preserved | — |
| 14 | `dvrk_obuda` | `obuda/seaspike_3` | preserved | — |
| 15 | `dvrk_obuda` | `obuda/skinphantom_1` | preserved | — |
| 16 | `dvrk_ucsd` | `ucsd/surgical_learning_dataset2` | preserved | — |
| 17 | `dvrk_ucsd` | `ucsd/surgical_learning_dataset` | preserved | — |
| 18 | `dvrk_jhu` | `jhu/imerse/wound_closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled` | preserved | — |
| 19 | `dvrk_jhu` | `jhu/imerse/nephfat/nephfat` | timestamp-repaired | 3a91637cb89a9f57396a418fc4ac0f9d1a35fa2c4700588cf50422498a038598 |
| 20 | `dvrk_jhu` | `jhu/imerse/srt_tissue_lift` | preserved | — |
| 21 | `dvrk_jhu` | `jhu/imerse/cao_cautery_combined` | timestamp-repaired | 61f9487b28aa727040c947f539382a5bd65789656ba4c1c298319985dc4d9474 |
| 22 | `dvrk_jhu` | `jhu/imerse/srth_porcine_chole` | preserved | — |
| 23 | `dvrk_jhu` | `jhu/suturebot_canon/suturebot_tissue_2` | timestamp-repaired | 376cf33ce90650fc690f2f8bee274680077dcc74fc41146d3b9acf56ed754150 |
| 24 | `dvrk_jhu` | `jhu/suturebot_canon/suturing_jesse` | timestamp-repaired | 878ae9ec6aa133c8424dcae012be05726fed5af06bcf6f0acea706eb9199158f |
| 25 | `dvrk_jhu` | `jhu/suturebot_canon/jesse_pickup` | timestamp-repaired | 9f4972b4444e6bb28315336ee636c1e4cf62273c550e3c5f1683e9db8298dc96 |
| 26 | `dvrk_jhu` | `jhu/lcsr/arcade/cholecystectomy` | preserved | — |
| 27 | `dvrk_suturebot_mono` | `jhu/hf_suturebot` | release mono | — |
| 28 | `dvrk_stanford` | `stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/peg_transfer` | timestamp-repaired → row-index repair | 248da71a161ea90fe15560876d8e9d36bed2d6ce6a15582705ea8161a791a5ab |
| 29 | `dvrk_stanford` | `stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/needle_transfer` | timestamp-repaired | 7e3624a60942289aa53d3346f94487213ebaf03bec942ab4e8a7afbb675e2642 |
| 30 | `dvrk_stanford` | `stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/tissue_retraction` | timestamp-repaired | 9bb9f51ab96ab59fb1947939ef336d5fa12ada306da9d930ee42faf87f5362e8 |
| 31 | `dvrk_rob_surgical` | `rob_surgical/all_merged_data` | preserved | — |
| 32 | `dvrk_ustc` | `ustc/needle_pickup` | preserved | — |
| 33 | `dvrk_ustc` | `ustc/Needle_handover` | preserved | — |
| 34 | `dvrk_ustc` | `ustc/ustc_knot_tying` | preserved | — |
| 35 | `dvrk_ustc` | `ustc/grasp_on_liver` | preserved | — |
| 36 | `dvrk_ustc` | `ustc/exvivo_liver_sep` | preserved | — |
| 37 | `dvrk_ustc` | `ustc/invivo_liver_sep` | preserved | — |
| 38 | `dvrk_ustc` | `ustc/tissue_lifting` | preserved | — |
| 39 | `dvrk_hamlyn` | `hamlyn/knot_tying` | preserved | — |
| 40 | `dvrk_hamlyn` | `hamlyn/needle_grasp_and_handover` | preserved | — |
| 41 | `dvrk_hamlyn` | `hamlyn/peg_transfer` | preserved | — |
| 42 | `dvrk_hamlyn` | `hamlyn/suturing_1` | preserved | — |
| 43 | `dvrk_hamlyn` | `hamlyn/suturing_2` | preserved | — |
| 44 | `dvrk_hamlyn` | `hamlyn/tissue_retraction` | preserved | — |
| 45 | `dvrk_ucberkeley` | `ucberkeley/debridement_lerobot` | preserved | — |
| 46 | `turin_mitic` | `turin/mitic_lerobot_plastic_tube` | preserved | — |
| 47 | `turin_mitic` | `turin/mitic_lerobot_ex_vivo` | preserved | — |
| 48 | `turin_mitic` | `turin/mitic_lerobot_plastic_pad` | preserved | — |
| 49 | `turin_mitic` | `turin/mitic_lerobot_plastic_pad_3dmed` | preserved | — |
| 50 | `cmr_versius_drybox_peg_transfer` | `cmr_surgical/drybox_peg_transfer` | timestamp-repaired | 791a81092fb089996b81891c916af1255cc57991246c77ba7a0a7ab06eed4f08 |
| 51 | `dvrk_star_il` | `jhu/star_il_261` | timestamp-repaired | 884c709f7806b3872264b7451553d7ab9eed415aaaf665831aa8ed8eb2ba8f63 |
| 52 | `dvrk_tud_grasping_retraction` | `tud/260131_tundra_dataset/grasping_retraction` | preserved | — |
| 53 | `dvrk_smarts` | `jhu/lcsr/smarts/SurgSync-stitch-coldcut/P1` | preserved | — |
| 54 | `dvrk_smarts` | `jhu/lcsr/smarts/SurgSync-stitch-coldcut/P2` | preserved | — |
| 55 | `dvrk_smarts` | `jhu/lcsr/smarts/SurgSync-stitch-coldcut/P3` | preserved | — |

## Normalizers and indices

The first-stage names below come from the archived final H0 config; “same name” is not proof of byte identity at every restart. Digests are of original **H16** normalizer files, not of public JSON serializations. The task does not distribute normalizers, indices, or their contents.

| Family | H0 filename | H16 filename | H16 file SHA-256 |
|---|---|---|---|
| `cmr_clinical` | same name | `cmr_clinical_all4_backward_anchored_worldtrans_rot6d.json` | `7407feda37ce61aec87ed63b5f81ca50e4207394a3ca22118bb26de97bd790f0` |
| `dvrk_obuda` | same name | `dvrk_obuda_backward_anchored_worldtrans_rot6d.json` | `0d7b90d6caca552266adb1ec24881ecb7628ac64331e25fe879166991e16e75d` |
| `dvrk_ucsd` | `dvrk_ucsd_backward_anchored_worldtrans_rot6d.json` | `dvrk_ucsd_backward_anchored_worldtrans_rot6d_wxyz_c24_20260910.json` | `b03def7774050bb5a5d48da60266e1628b22d26c3405a3cfabb95ed6ab3a3a02` |
| `dvrk_jhu` | same name | `dvrk_jhu_backward_anchored_worldtrans_rot6d.json` | `6617b6c04c52dd38a66243417cff3655f6d180022652ad209bf3c5cd1dd5101b` |
| `dvrk_suturebot_mono` | same name | `suturebot_backward_anchored_worldtrans_rot6d.json` | `18ad4d232e0b99e009740fb1101847467d24599574efe56578d61c590434ba00` |
| `dvrk_stanford` | `dvrk_stanford_backward_anchored_worldtrans_rot6d.json` | `dvrk_stanford_backward_anchored_worldtrans_rot6d_shardfixed_c24_20260908.json` | `03d900ccebc3d3efd0ac2d21f6c4ff565f908ac3562d7fa9ff6b1e39d9d4ec86` |
| `dvrk_rob_surgical` | same name | `dvrk_rob_surgical_backward_anchored_worldtrans_rot6d.json` | `586a3b953eeb50ba039cd9a6a97aa5263a054129878a0dbd5899892b4e2475f8` |
| `dvrk_ustc` | same name | `dvrk_ustc_backward_anchored_worldtrans_rot6d.json` | `24c410ca14a0adbfbeef1bfd75449417ebd2a84083af4ed52a9c9823797a3e30` |
| `dvrk_hamlyn` | `dvrk_hamlyn_backward_anchored_worldtrans_rot6d.json` | `dvrk_hamlyn_backward_anchored_worldtrans_rot6d_wxyz_c24_20260910.json` | `7dd6650a89dbedc8d91895692c6a3491d5174bf30d5d9869e936c1a280c1e9ae` |
| `dvrk_ucberkeley` | same name | `dvrk_ucberkeley_backward_anchored_worldtrans_rot6d.json` | `cc1fc6b716681339d4216cb7b7c705b77568116ea8e72ea84ab55f4722817633` |
| `turin_mitic` | same name | `turin_mitic_backward_anchored_worldtrans_rot6d.json` | `2d6bc3be96d9e0deea06b73a011063ca91d9fa6e625ba757293f3a4fe952ed3e` |
| `cmr_versius_drybox_peg_transfer` | same name | `drybox_backward_anchored_worldtrans_rot6d.json` | `2c3f15575699dc7d7a47d7fe15f136848fa110c3a98b03b33de92612ac57501b` |
| `dvrk_star_il` | same name | `star_il_backward_anchored_worldtrans_rot6d.json` | `770bed9cd3c20eed222f18886f24877e963b85a5c6368c1928fc815096c5ce72` |
| `dvrk_tud_grasping_retraction` | same name | `dvrk_tud_grasping_retraction_backward_anchored_worldtrans_rot6d.json` | `856f9d889fdb4e020f8c8d2ded8e9bcea1f1a4392efcdd62247df66aa7cfe539` |
| `dvrk_smarts` | same name | `dvrk_smarts_backward_anchored_worldtrans_rot6d.json` | `060ac08e2474465bf610541236f7532667f2964ddcb9763142bc8e46492eb807` |

- **CMR valid-window index, relative to the prepared directory:** `indices/cmr_valid_windows_c24_stride8_reject_any_v2.npz`. Original file SHA-256 `397076576238f647dbb8cbd0ac046ceedec66d4b9ca91a2021e8a01070f7ac8f`.
- **exclusions for every family, public equivalent relative to the repository:** `cosmos3_h_embodied/data/exclusions/openh_h16_exclusions.json`. Original file SHA-256 `bae99f1ec9690014891d88f47555c6252f0a8175b0c83a4c03fbe63d6f5f009c`.

For wound closure (row 18) the release used `video_source: clean`, the twin without burned-in points (`observation.images.endoscope.left_clean`, 960 × 540), not the public labeled stream. The reference needs clinical instructions, the four nonpublic families and the three canonical SutureBot roots within JHU. An 11-family public run is not this experiment. Check the SMARTS camera overlays and time-dependent repairs at each H0 restart; an unchanged logical path does not prove unchanged media. See [the recipe and remaining gates](RECIPE.md).
