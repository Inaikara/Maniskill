# Final retained artifacts — three obstacle tasks

## Cleanup status

The 2026-10-23 cleanup retains one final valid paired experiment per task,
the six selected Base DP / ConstrainNet weights, and the formal data, training
records, fixed scenes and verification needed to reproduce those experiments.
Safety success is the primary comparison metric; the paired Base DP comparator
is retained even where its safety result is worse. No checkpoint or scene was
reselected by looking at individual evaluation outcomes.

Pilot/calibration/probe/interface-smoke artifacts, failed or superseded runs,
the failed PlaceSphere ConstrainNet checkpoint and the periodic PickCube Base DP
checkpoint were deleted. The cleanup removed **992 files / 450,850,395 bytes**
(about 430 MiB); **765 retained file identities** were verified unchanged.

Cleanup record:
`outputs/exp/20261023_105159_final_artifact_cleanup/cleanup_manifest.json`.
It records all retained and deleted file paths, sizes and SHA-256 hashes, and
each completed deletion. `previous_OUTPUTS.md` in that directory preserves the
previous full output index. Historical manifests/reports were not rewritten:
references to removed diagnostics are historical, and resolve to deletion
records rather than current files. The earlier completion audit describes the
pre-cleanup acceptance evidence, including pilots that have now been removed.
`final_verification.json` in the cleanup directory records a second retained-hash
check and an independent recomputation of all six policy/task metric rows from
the 600 retained episode records; all match their authoritative summaries.

## Selected weights — exactly six files

| Task | Model | Current path |
| --- | --- | --- |
| StackCube (`panda_wristcam` evaluation) | Base DP | `outputs/ckpt/20260930_base_dp_stackcube_980/best_eval_success_at_end.pt` |
| StackCube | ConstrainNet | `outputs/ckpt/20261021_constrainnet_wristcam_seed1/final.pt` |
| PickCube | Base DP | `outputs/ckpt/20261022_144000_pickcube_base_dp/final.pt` |
| PickCube | ConstrainNet | `outputs/ckpt/20261022_144835_pickcube_constrain/final.pt` |
| PlaceSphere | Base DP | `outputs/ckpt/20261022_193900_placesphere_base_dp/final.pt` |
| PlaceSphere | ConstrainNet | `outputs/ckpt/20261022_201100_placesphere_constrain/final.pt` |

The weight files keep their original bytes and hashes. StackCube uses the
previously selected `ema_agent` / final EMA; the two new tasks use their
preselected final EMA after 30,000 updates per selected model.

StackCube's historical runs have date-level provenance, so their weight folders
use `YYYYMMDD_` prefixes (2026-09-30 and 2026-10-21) without inventing a start
time. The 2026-10-23 renames preserve both checkpoint hashes. Their exact
old/new file mappings and hashes are in
`outputs/exp/20261023_105159_final_artifact_cleanup/path_migrations.json`;
the cleanup verifier uses this sidecar to resolve the original retained paths.

## Final data and validation

| Task | Retained formal data | Retained validation records |
| --- | --- | --- |
| StackCube | `demos/StackCube-obstacle/` (100 obstacle demos); Base DP source `demos/StackCube-v1/motionplanning/trajectory.filtered.state.pd_ee_delta_pos.physx_cpu.h5` (980 demos) | Dataset `validation.json`, training and scene audits below |
| PickCube | `demos/PickCube-obstacle/20261022_132753_formal/` (1,000 direct + 100 obstacle) | `outputs/exp/20261022_132753_pickcube_formal/` |
| PlaceSphere | `demos/PlaceSphere-obstacle/20261022_184558_formal/` (1,000 direct + 100 obstacle) | `outputs/exp/20261022_184558_placesphere_formal/` |

The historical `demos/StackCube-v1/` source/replay archive is preserved under
the repository's original-data traceability rules. No retained data was moved
or reserialized. PickCube and PlaceSphere pilot/intermediate datasets are gone.

## Final experiment directories

All paths in this table are below `outputs/exp/`.

| Task | Base training | ConstrainNet training | Fixed scenes | Final paired evaluation |
| --- | --- | --- | --- | --- |
| StackCube | `20260930_base_dp_stackcube_980_seed1/` | `20261021_constrainnet_wristcam_seed1/` | `20261021_wristcam_scenes/` | `20261021_wristcam_paired_eval/` |
| PickCube | `20261022_144000_pickcube_base_dp/` | `20261022_144835_pickcube_constrain/` | `20261022_152157_pickcube_scenes/` | `20261022_162002_pickcube_paired_eval/` |
| PlaceSphere | `20261022_193900_placesphere_base_dp/` | `20261022_201100_placesphere_constrain/` | `20261022_205500_placesphere_scenes/` | `20261022_213500_placesphere_paired_eval/` |

Each paired experiment retains both policies' complete 100-episode records,
summaries, parsed configuration, identity/restore evidence and representative
videos. PickCube's `termination_correction.json` remains authoritative for its
18/17 historical 500-step-limit outcomes; original flags were not reconstructed.
PlaceSphere's real time-limit truncations are 3/13 at the 200-step limit.

Also retained:

- `outputs/exp/20261022_221641_multitask_obstacle_completion_audit/`: original
  12-ticket acceptance audit, whose pre-cleanup scope is unchanged.
- `outputs/exp/20261023_105159_final_artifact_cleanup/`: current pruning and
  retained-identity audit. This supersedes historical *retention* descriptions.
- `outputs/exp/20261023_122842_pickcube_contact_diagnosis/`: Ticket 01 contact
  diagnosis for 28 old PickCube scenes. It does not replace
  `20261022_162002_pickcube_paired_eval/`. The earlier incomplete run
  `20261023_122333_pickcube_contact_diagnosis/` is not the evidence set.
- `outputs/exp/20261023_141712_placesphere_failure_diagnosis/`: Ticket 02
  crossing and terminal diagnosis for 28 old PlaceSphere scenes. It does not
  replace `20261022_213500_placesphere_paired_eval/`.
- `outputs/exp/20261023_153133_pickcube_transport_detour/`: retained Ticket 03
  batch. It does not close the parent end-sweep, join, or source-identity gaps.
  Bytes are unchanged. Matching trajectories remain
  `demos/PickCube-transport-detour/20261023_153133/`.
- `outputs/exp/20261023_165956_pickcube_transport_detour/`: Ticket 03 acceptance
  batch, 8/22 attempts. It supersedes `20261023_153133` for those three gaps
  and does not replace `20261022_152157_pickcube_scenes/` or
  `20261022_162002_pickcube_paired_eval/`. Matching trajectories are
  `demos/PickCube-transport-detour/20261023_165956/`. Incomplete same-day runs
  `20261023_164921_pickcube_transport_detour/` and
  `20261023_165430_pickcube_transport_detour/` are not this evidence set.
- `outputs/exp/20261023_235744_pickcube_learning_pilot_r2/`: Ticket 05
  learnability pilot, round 2 of 2. Gate is **pause**. It does not replace
  `20261022_162002_pickcube_paired_eval/` and does not admit Ticket 07.
  Round 1 `outputs/exp/20261023_183243_pickcube_learning_pilot/` is retained
  and was not used for training. Weights:
  `outputs/ckpt/20261023_235744_pickcube_pilot_base/final.pt`,
  `outputs/ckpt/20261023_235744_pickcube_pilot_constrain/final.pt`.
   Training demos: `demos/PickCube-transport-detour/20261023_235744_pilot_r2/`.
   These are not formal performance results.
- `outputs/exp/20261024_104348_pickcube_data_expansion/`: Ticket 05 authorized
  data-expansion round on the same frozen 30 dev scenes. Gate is **pause**.
  It does not replace `20261022_162002_pickcube_paired_eval/` or the round-2
  pilot, and does not admit Ticket 07. Weights:
  `outputs/ckpt/20261024_104348_pickcube_expansion_base/final.pt`,
  `outputs/ckpt/20261024_104348_pickcube_expansion_constrain/final.pt`.
   Training demos: `demos/PickCube-transport-detour/20261024_104348_expansion/`
   (1,000 direct + 100 obstacle, including the original 20+20). Not a formal
   performance or generalization result.
- `outputs/exp/20261024_202858_pickcube_proximity_weight/`: Ticket 05
  authorized training-optimization slice on the same 1,000 direct + 100
  obstacle data and the same frozen 30 dev scenes. One control-only retrain
  from the expansion Base EMA, with a frozen TCP-obstacle proximity weight on
  the epsilon loss. Gate is **pause**. It does not replace the expansion run
  or admit Ticket 07. No new demos. Weight:
  `outputs/ckpt/20261024_202858_pickcube_proximity_weight_constrain/final.pt`
  `22d3624941645ecac5bab4bd8679a2e40f77498b531b40f5e1d3c4be79d2a037`.
   Same-scene result versus expansion Constrain is worse: task 22/26, safe
   11/13, contact 18/17, XY 4/3. Not a formal performance result.
 - `outputs/exp/20261024_213325_pickcube_obstacle300/`: Ticket 05 authorized
   100→300 obstacle increment on the same frozen 30 dev scenes. The existing
   1,000 direct arrays were copied, not re-collected, and Base was not
   retrained. Unweighted epsilon; the rejected proximity weight stayed off.
   Gate is **pause**. It does not replace the expansion run or admit Ticket 07.
   Weight: `outputs/ckpt/20261024_213325_pickcube_obstacle300_constrain/final.pt`
   `bd7757122b75ad0bcc9cc5bd88a4b5174d48dfc520d1c8459332eea157732801`.
   Training demos: `demos/PickCube-transport-detour/20261024_213325_obstacle300/`
   `3c15f50993b5ab5967b1a40695e53288b0f9464f4b61504a93d9989499ded7e6`
   (1,000 copied direct, not claimed unique; 300 obstacle = retained 100 plus
   200 new unique scenes). Same-scene result versus expansion Constrain
   26/13/17/3 is 25/12/18/2 task/safe/contact/XY. Not a formal performance
   result. Preregistration
   `3d37d27ac2145733c091094a5dd5245e9a237294abaebef6c37c1d29e28060a9`.
    Gate `296088e5e0b7118952d7dc020ca603dcfe98eeda3a1b0dc7cba430c56828e71a`.
    Comparison `f20d5c182c3d7079ad50e2411857670d96c77e6d7ee3b6b0e6f1474b7ee5d92e`.
  - `outputs/exp/20261025_062737_pickcube_window_gap/`: Ticket 05 expert-window
    diagnosis of the frozen 100 and 300 EMAs. No collection and no training.
    Same sample seeds and diffusion timesteps. Index
    `diagnosis/index.json`
    `35d9525892d3a2fc36dc75b905f0675b09669a82195867a8ff4d7edca44f8ed8`.
    Preregistration
    `db3a3479632664205578ab8f6eba00ed4bc0053f751ecb5d96c31661f2019bfb`.
    It does not replace the 300 trial or admit Ticket 07.
  - `outputs/exp/20261025_065159_pickcube_exposure90000/`: Ticket 05 authorized
    scratch 90,000-update trial from the matched expansion Base EMA. The saved
    30,000-step cosine last learning rate was 0, so resume was not used.
    Unweighted epsilon; proximity stayed off. No new demos. Gate is **pause**.
    It does not replace the 100 or 300 runs or admit Ticket 07. Weight:
    `outputs/ckpt/20261025_065159_pickcube_exposure90000_constrain/final.pt`
    `8a5de3ac5f8c4caba5617c30605bc17201f6beb4e4ea139fe0b8cfa8bb6ed5db`.
    Same-scene result versus expansion Constrain 26/13/17/3 is 27/15/15/3
    task/safe/contact/XY. Not a formal performance result. Preregistration
    `4e1be76719c9c22388db372c464cb3f222a4d1665343ef15c6c975ecfc3cfeca`.
    Gate `4f93ef34f0e6670246c7b5fd46c90e60fe9f54c31ab54dda9545093e22c42081`.
    Comparison `a8f213c327731fa2688b533c396a96c20f8b41b3932e5985ec6b2f6d67aad0c5`.
   - `outputs/exp/20261025_081913_pickcube_closed_loop_diagnosis/`: Ticket 05
     closed-loop diagnosis of the 15 exposure contact episodes. No collection,
     no training, and no stride trial. It does not replace the 90k eval or
     admit Ticket 07. Checkpoint unchanged
     `8a5de3ac5f8c4caba5617c30605bc17201f6beb4e4ea139fe0b8cfa8bb6ed5db`.
     Preregistration
     `68e84320a807face3b1a1f25a3da6246137e656725330627399913eda95679b6`.
     Index `diagnosis/index.json`
     `46a00982c870c3af71b897f7dedd20285849088d6cdf2bfc307a4dfc6ccf30ae`.
     Gate remains pause. Staleness was not isolated (0/15 queued overshoot).
  - `outputs/exp/20261024_014459_placesphere_learning_pilot/`: Ticket 06
  learnability pilot, round 1 of 2. Gate is **pause**. It does not replace
  `20261022_213500_placesphere_paired_eval/` and does not admit Ticket 08.
  No second round was started. Weights:
  `outputs/ckpt/20261024_014459_placesphere_pilot_base/final.pt`,
  `outputs/ckpt/20261024_014459_placesphere_pilot_constrain/final.pt`.
  Training demos: `demos/PlaceSphere-transport-detour/20261024_014459_pilot_r1/`.
  These are not formal performance results.
- `outputs/exp/20261024_155244_placesphere_data_expansion/`: Ticket 06
  authorized data-expansion round on the same frozen 30 dev scenes. Gate is
  **pause**. It does not replace `20261022_213500_placesphere_paired_eval/` or
  the round-1 pilot, and does not admit Ticket 08. Weights:
  `outputs/ckpt/20261024_155244_placesphere_expansion_base/final.pt`
  `2bbb7f5e1c1f4f93d177cf12b63d182a2d4cc1a239387f51d03de6afe01c8640`,
  `outputs/ckpt/20261024_155244_placesphere_expansion_constrain/final.pt`
  `36518c84c0aacaadaf33eb7e35161b0e869c0d18d62e9425337021a763bfa43c`.
  Training demos: `demos/PlaceSphere-transport-detour/20261024_155244_expansion/`
  `df01aec45a5fcf6f04b0f0e2cb91d4bdf58d238ce41163b40cf15d1388d253ed`
  (1,000 direct + 100 obstacle, including the original 20+20). The packaged
  1,000 is not 1,000 unique task layouts (greedy unique 425, conflicts 575).
  Not a formal performance or generalization result. Preregistration
  `270ecc8712aef405f94cfdf14c6ad5f7cc1e30f4a7faf9d1d696a0185da45100`.
  Gate `fb768c089f124780eda1a477fbe5db26544d161b7faf9822f06b3032aa854391`.
  Comparison `3864d18c94ad775b18dfd2e5a0093df7ccae1ae7f342be574f6e79ac7612fbb0`.
  Coverage-writer fix validation, not a rerun:
  `outputs/exp/20261024_185754_placesphere_coverage_writer_fix/`.
  It does not replace this experiment or change its bytes. Experiment source
  `63d17eb3889b92b3638a42ee84ae3ada765b3dcf9b013307f3e4bba0d222d844` stays the
  preregistration identity. New writer identity
  `c370f36b3c1bb1d3ceb87fe6db200c754c0cf934d7cbe91fac566c6850c1b655` is not
  retroactive. Validation
  `2a59194181b48870889492aae53424523b5eca99958ece386fc3e166d65aec61`.
- `outputs/exp/20261025_085101_placesphere_obstacle300_90k/`: Ticket 06
  authorized combined 100→300 obstacle and 90,000-update slice on the same
  frozen 30 dev scenes. Factors cannot be separated. Gate is **pause**. It
  does not replace the expansion run or admit Ticket 08. No automatic
  300→1000. Base was not retrained. Weight:
  `outputs/ckpt/20261025_085101_placesphere_obstacle300_90k_constrain/final.pt`
  `70cf5e6f00bfec8b7d0945f65805fba06bbddd95356778d73009bdc5fed2326b`.
  Training demos:
  `demos/PlaceSphere-transport-detour/20261025_085101_obstacle300/`
  `f9dd209868d800fbce41eae276112a23dffc9aec02e91196ca900d46fc42aa02`
  (1,000 copied direct + 100 copied obstacle, arrays equal to
  `df01aec45a5fcf6f04b0f0e2cb91d4bdf58d238ce41163b40cf15d1388d253ed`;
  200 new unique obstacle). Same-scene Constrain versus expansion 25/21/7/0
  is 27/21/6/1 task/safe/contact/XY. Not a formal performance result.
  Preregistration
  `a2708b53736ad21d4242d94afa371cab37ab8456d454f6aea525d04e12779382`.
  Gate `b640d91352b28dc503303c35c3a96085835a4c761e75727700d0cd51d839c9e1`.
   Comparison `e0bd371ae6c32dfe7331f1d44a918bebe8b3fc2fa9f4d3640271f8e26597bb3f`.
- `outputs/exp/20261025_114030_placesphere_release_diagnosis/`: Ticket 06
  indexed diagnosis of the frozen `70cf5e6f` EMA. No training and no new data.
  It does not replace the 300/90k eval. Index
  `10a2ec98844bf9afaa7ea28f1ccf48c8049d017d28b8ed6d90d54e17f55cab38`.
  Evidence `8e93d6c9886e7e69e86accfc12622cc3b20f7205d4ccc28d06d6c947fa1a6bec`.
  Window index is `window_index.jsonl`, 40104 entries.
- `outputs/exp/20261025_114648_placesphere_phase_balance/`: Ticket 06
  one phase-inverse-frequency trial on the unchanged 300 demos, from Base EMA
  `2bbb7f5e1c1f4f93d177cf12b63d182a2d4cc1a239387f51d03de6afe01c8640`, 90000
  updates, final EMA. Gate is **pause**. It does not replace the 300/90k run
  or admit Ticket 08. Weight:
  `outputs/ckpt/20261025_114648_placesphere_phase_balance_constrain/final.pt`
  `3385414d139d587252dfca36fdd6dd4501d7f1ac7fdfcd220e95fa08ed3a58bf`.
  Same-scene Constrain versus `70cf5e6f` 27/21/6/1 is 27/21/7/0
  task/safe/contact/XY. Contact cleared/introduced/both/neither = 2/3/4/21.
  Not a formal performance result. Preregistration
  `9c1a630dd7f4e29c09463fe29e49652f1daf08298aa0eb90fc8542723032432b`.
  Gate `1f0dd11faa7893d7e8b55f388e753da988cffa43902e932cc8d4cbee7f6ca493`.
  Comparison `e2444277f5d755d0b4a230e0631ac31828c25ed4541f8c97d71ba0b518f36b75`.
- `outputs/exp/20261025_131040_placesphere_coverage/`: Ticket 06 offline
  coverage of the six frozen `70cf5e6f` contacts against the existing 300
  obstacle windows only. No new demos, no training, no dev labels in the
  reference set. Prior diagnosis traces had no full observation window, so
  the frozen 30 were replayed; contact steps matched 1/90, 3/146, 5/113,
  9/146, 19/92, 21/162. Train windows 40104 on episodes 1000..1299:
  approach 16899, transport 17952, held_near_bin 4953, release 300.
  Execution-state rule would collect; policy-window rule would not
  (5/6 vs 3/6 outside). Nearest expert actions are still closed-grip
  negative delta-z. Not a reward or formal result. Preregistration
  `c4d7df16feae6bc5d9bf208f040769b8a7c2b4df6f99b888a0945e7351335480`.
  Coverage `566542a0146eda49df58420cf0fec371e5215cb0e57dcdbaaff5d9e78f281089`.
  Pilot identity `f238ddc29af22eab6daee9adf9e74d3d2132b0d2e2694e23530fc3cfcc2043e3`.
  Labeler identity `c53a60af0f3f698ffc2e497a6acba202d94e01f3dcbe53866c5612d6e328a6ce`.
- `outputs/exp/20261025_131730_placesphere_recovery_probe/`: one train-scene
  feasibility probe, not a packaged demo. A 2 cm lateral offset at the
  closed-grip container approach still commands a closed descent to the same
  target. Collection of this expert is not a recovery dataset. Probe
   `49facecbf0e06a09f75e970830f4401d017cf6a26b39b491232d99410985397e`.
- `outputs/exp/20261025_134916_placesphere_matched_terminal/`: Ticket 06
  opt-in above-release feasibility. Not a training set and not collection-ready.
  Legacy `terminal_mode` default remains `legacy`. Selected margin 0.015 m on
  8/8 calibration obstacle runs after 0.005 m failed the vertex filter on 2/8.
  Confirmation obstacle 16/16 native-safe, 0 contact. Dev30 after the freeze,
  not used for selection or training: 30/30 native-safe, 0 contact.
  Perturbation denominator 64 compared on 8 calibration layouts: legacy
  contact 32, new contact 10, introduced 0, cleared 22. Pre-terminal ±1/±2 cm
  contact was 0/32 for both. The contact drop is only at nearest-approach
  midstates, where new still contacts 10/32. Direct suffix accepted 0/8 and
  0/16. `confirmation.json` `collection_ready: true` is obstacle-only and is
  overruled by `collection_decision.json`. No demos written. Old 300 and
  direct h5 files were not overwritten. Run identity
  `aa7d2821ce537475051e30895e04a9acbdeaecae2dd8dc1da04ee4da5fa80eed`.
   The later flag fix is not that identity. Radius, native thresholds, and
   cylinder geometry were not changed. No training. That run's
   `confirmation.json` `collection_ready: true` is obstacle-only and is not
   rewritten. It is not this round's identity.
- `outputs/exp/20261026_100200_placesphere_matched_terminal_r19/`: Ticket 06
   round 19 role split. Not a training set. Collection is **not ready**.
   Source identity `75e474697e1960c655d548945d7423ad28c6087ee65281f20ad5620ab7e33bf3`.
   Previous runtime `aa7d2821ce537475051e30895e04a9acbdeaecae2dd8dc1da04ee4da5fa80eed`
   was not rewritten. Margin 0.015 m selected on obstacle positives only
   (0.005 failed 2/8 `hand_below_cylinder_top`). Calibration role bundle 8/8.
   Confirmation admission 13/16, `collection_ready` false in both
    `confirmation.json` and `collection_decision.json`. Held-out dev30 26/30,
    not used for selection or training. Negative direct 16/16 confirmation
    transport-blocked and admits no demos. No full collection and no training.
- `outputs/exp/20261026_103421_placesphere_terminal_round20_phase_a/`: Ticket 06
  Round-20 Phase A measurement before the motion correction. Final trace
  identity `4cc1780646a8ffe6514fd911a393279f4e4d466808591ac10598d45d519692b2`;
  diagnosis SHA `4977e33094005879408f17d1890dcf0d0c49dfbbf2979afd48c35d2948eb80a4`.
  R19 full failures are angular: 085 was 0.001252 m/s linear and 1.87655 rad/s
  angular; dev 003 was 0.000353 m/s and 4.54203 rad/s; dev 019 was 0.000361
  m/s and 7.39852 rad/s. After 200 neutral steps, 003 remained at 2.16161
  rad/s and 019 at 4.48883 rad/s. 085 reached native at +69 steps. Actual-
  cylinder suffixes 046/123 reached their own first native endpoints; dev-006
  did so after five neutral steps. Dev-000 measured shared z 0.151770565 m
  against direct requirement 0.151873754 m, a 0.103189 mm shortfall (vertex
  gap is approximate, not a surface proof).
- `outputs/exp/20261026_110038_placesphere_round20_long_settle_probe/`: focused
  continuation from retained full R19 actions, diagnostic only. Dev-003 reached
  native after 361 additional neutral steps; dev-019 after 785. No thresholds or
  geometry changed.
- `outputs/exp/20261026_104648_placesphere_matched_terminal_r20/`: preserved
  earlier complete Round-20 validation, identity
  `b437c3f2d2bb426936459921de248c5e0bb32a459e4dcfaa03865e560135993b`.
  Calibration 8/8 and confirmation 16/16 passed, but dev30 was 27/30. This is
  a historical recovery point and was not rewritten.
- `outputs/exp/20261026_105656_placesphere_round20_dev_engineering_feedback/`:
  non-selecting repeated-dev feedback for predeclared 8/16-step holds on
  dev-000/003/019. It did not repair those cases and is not independent
  validation. Recovery files are in
  `outputs/exp/20261026_110010_placesphere_matched_terminal_r20_recovery/`.
- `outputs/exp/20261026_110408_placesphere_matched_terminal_r20b/`: final
  Round-20 validation identity
  `3edc929330f644c407b94f6a23244520ae96b639706b954258336f82c3441958`.
  Margin 0.015 m; selected 0-step open hold on calibration 8 after all 0/8/16
  candidates tied 8/8. Confirmation 16/16 matched positives and 16/16 blocked
  negatives. Frozen dev30 admitted 30/30 without selection/training use and
  had 30/30 blocked negatives. Direct/obstacle release z matched exactly.
  Expert neutral settle is capped at 800 steps (1,000 total expert actions) for
  the measured spin cases; policy evaluation remains 200. Perturbation scope:
  64/64 finite comparisons on calibration 8; new contact 10/64, native 32/64,
  crossing 0/64 versus legacy contact 32/64, native 30/64; cleared/introduced/
  both/neither contact transitions 22/0/10/32. Preterminal native 30/32→32/32;
  nearest-approach native 0/32 for both. This is a bounded
  calibration-layout scope, not a broad rate. No full collection or training.
- `demos/PlaceSphere-transport-detour/20261026_111952_round20_matched_tplus1/`
  and `outputs/exp/20261026_111952_placesphere_round20_tplus1/`: one matched
  direct/obstacle T+1 state/controller/action artifact after the gates. Both
  roles exactly restored initial state/controller and first-action state,
  controller, policy observation, terminated, and truncated values. HDF5 SHA
  `51cb2f0109898c7477d6d522251ea58086559e3dd11c921aedc40e332544dacc`; runner
  SHA `f3001d7fa8032d526453c1bf41ccefcb33d9fa39b57c551d7c28e673d162ed39`.
  This is a T+1 proof artifact, not the 1,000+300 collection and not used for
  training. The existing formal generator still uses the legacy terminal path;
  that was the state before the Round-21 opt-in integration below.
- Round 21 matched-terminal collection, source integration and exact commands:
  `outputs/exp/20261026_133000_placesphere_matched_collection_r21b/`.
  The final pre-collection registration is
  `preregistration.json` (`a1f6ff02fac852ea1d73bfce951047e17f6dc74b22a2f4f3643e4dd176ad0a29`),
  source identity `6a8091fc694e43fffe84fd50d7d60280f649d927986a143445f26d91a420b048`.
  It pins the collector and all experiment helpers, native PlaceSphere source
  hashes, environment package versions, field schemas, action/observation
  dimensions, source layout catalogs, task-identity dedup and finite budgets.
  ManiSkill source and historical datasets were not modified. Existing geometry
  challenge, 3 cm cylinder radius, native thresholds, legacy terminal default,
  and 200-step policy budget remain fixed.

  The bounded native pilot at `pilot_subset/` regenerated 80 DIRECT + 25
  OBSTACLE demos across 25 matched pair layouts from the frozen final-candidate
  lists. All 105 T+1 initial/first-transition restore audits passed. There was
  one rejected expert case, retained in the journal; no policy-performance
  filter was used. 6/80 DIRECT and 2/25 OBSTACLE episodes exceeded 200 actions;
  median/p90/p95/max lengths were 105/147/160/909 and 116/179/529/633. Actions
  beyond the 200-step horizon were 13.4% and 20.9% of each role's action volume.
  Open near-zero-motion actions were 19.8% DIRECT and 26.9% OBSTACLE. The
  preregistered native-static 8-action windows were 0/1,765; action-neutral
  fractions and phase counts are retained in `pilot_subset/report.json`. This
  does not imply policy convergence; all complete native-success tails remain
  in the data.

  The first full 1,000+300 attempt at
  `demos/PlaceSphere-transport-detour/20261026_123000_matched_terminal_r21/`
  is preserved as a nonqualifying diagnostic package: its mixed-cohort selector
  produced only 213 greedy-unique OBSTACLE task layouts (87 duplicates). Its
  trajectory SHA was `61d5d06543bb1e4db13a9083bb10910c65c0d0b4ae0dc532fa72f6747809efd7`;
  it is superseded below and must not be used for training.

  The corrected, qualifying candidate package is
  `demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/`.
  Primary OBSTACLE sources were frozen as the 100 matched pair-catalog layouts
  plus the 200 `catalog_full` obstacle layouts; all 300 pass the task-identity
  dedup rule. A separately frozen 212-layout task-unique replacement pool was
  available; 3 replacements were used after 3 positive OBSTACLE native-terminal
  failures. The DIRECT class contains 300 matched positive DIRECT partners plus
  700 DIRECT-only trajectories from the separate source pool. Actual source
  counts: DIRECT = 703 `direct_scenes/assembled`, 99 `pair_scenes/merged`, 198
  `catalog_full`; OBSTACLE = 3 `direct_scenes/assembled` replacements, 99
  `pair_scenes/merged`, 198 `catalog_full`. All source scene IDs, seeds, and
  role-to-episode mappings are in `collection_full/layout_id_mapping.json`.

  Collection completed in 3,165.2 s: 1,000 DIRECT + 300 OBSTACLE, 300 paired
  layouts, 1,005 layout attempts, 3 preregistered obstacle replacements, and 5
  rejected cases (3 positive obstacle terminal failures; 2 positive direct
  native-placement failures). No exception, runtime error, state-restore
  failure, timeout, or dev/Ticket04 layout conflict was admitted. All 300
  accepted OBSTACLE layouts are task-unique and all 300 have paired DIRECT,
  negative transport-block, and suffix-replay proofs. The DIRECT class has
  472 greedy-unique task layouts and 528 within-class task repeats; source
  identity repeats are not presented as new unique layouts. This is comparable
  in scale to the prior 425 unique / 575 repeat DIRECT package, with the changed
  paired source mix disclosed above. External conflicts against the frozen
  dev30 and Ticket 04 catalogs are 0.

  Independent saved-artifact audit passed all 1,300 episodes: 7-D legal native
  actions, named 39-D policy / 42-D raw observations, T actions and T+1 native
  state/controller/info/termination records, first native success, and 1,300/1,300
  exact initial + first-transition restore audits. Every obstacle prefix is
  contact/crossing-free from execution start. Radius/native thresholds and
  obstacle geometry are unchanged; vertex clearances remain approximations,
  with PhysX execution as the safety evidence. Full layout mapping and per-episode
  findings are in `collection_full/dataset_validation.json`.

  Across all 1,300 episodes, 89 exceeded 200 actions: 60/1,000 DIRECT and
  29/300 OBSTACLE. Role medians/p90/p95/max are DIRECT 105/147/321/909 and
  OBSTACLE 126/185/544/889. Above-200 actions account for 15.8% DIRECT and
  20.7% OBSTACLE action volume. Near-zero open actions account for 21.6% and
  25.5%; the strict 8-action native-static window count is 0/22,364 before the
  first native-success stopping point. Step-phase action totals (DIRECT /
  OBSTACLE) are grasp approach 42,100 / 12,562; grasp pose 5,000 / 1,500;
  gripper close 8,000 / 2,400; lift 8,028 / 2,404; transport 29,015 / 15,021;
  clearance approach 4,112 / 1,228; stabilize 2,001 / 600; release 1,000 / 300;
  retreat 6,101 / 1,807; settle 27,723 / 12,555. Phase labels refer to the
  expert's current executed phase, with no future-state phase inference. Full
  per-phase current-action labels, episode lengths and action-window records are
  in `report.json`; no success tail was truncated or removed. Do not infer that
  the unchanged 200-step policy budget solves these longer expert tails.

  Identity hashes: `trajectory.h5`
  `eecfcdac7e357334eb68921c1e784e6858c3ccdd11d69c17e4e79b5cbf4556d0`,
  `trajectory.json`
  `ff971fb2b9278126c20d37a7b43a3f9ad74d0a86adc5ce926922b0ad9f796e86`,
  `attempts.jsonl`
  `b13b796325af0a838a803bf169d9dfc0daade6a394a7f9ef264da318d47ff42a`,
  dataset report `9d8e4c9231e37017df17c916df8a00b8cfad06d502f51a86e9f67d077fa3e44d`,
  independent validation `da77e933fa6f887594800c187d169255724c65ef82957513f3db4d9a84bfe089`,
  dataset identity `f0b9dd6d5b966744f44608528aa4bbab89baf9ebeb5b01972d315ad5ad78150e`,
  and layout mapping `a519b542faf497c49d0342851343c67dc7edb19a37c881d84f747ecf65811825`.
  Complete preregistration, pilot and full-collection command logs are retained
  in the matching `outputs/exp/20261026_133000_placesphere_matched_collection_r21b/`
  directory. No training or policy evaluation was run. The dataset is ready for
  independent parent review. Its HDF5 IDs follow collection order and are
  interleaved DIRECT/OBSTACLE, so the legacy trusted-formal trainers (fixed to
  the historical 1,000/100 IDs and hashes) must not be pointed at this version.
  After review, the matched training slice should add an opt-in reader keyed by
  the new manifest's class IDs, preregister the 1,000/300 identities, then run
  Base DP for 30,000 updates on DIRECT and ConstrainNet for 90,000 updates from
  that matched Base EMA on OBSTACLE. Use separate `outputs/exp/` and
  `outputs/ckpt/` paths. This is a proposed next plan, not a training result or
  an executed command.
- `outputs/exp/20261023_175529_placesphere_transport_detour/`: Ticket 04
  acceptance batch, 4/8 attempts. It does not replace
  `20261022_205500_placesphere_scenes/` or
  `20261022_213500_placesphere_paired_eval/`. Matching trajectories are
  `demos/PlaceSphere-transport-detour/20261023_175529/`. Same-day incomplete
  runs `20261023_174606_placesphere_transport_detour/` and
  `20261023_175100_placesphere_transport_detour/`, and the smaller successful
  probe `20261023_175419_placesphere_transport_detour/`, are not this evidence
  set.

## Reports

- [Current three-task results](THREE_TASK_RESULTS.md)
- [StackCube paired results](WRISTCAM_EVALUATION.md)
- [PickCube paired results](PICKCUBE_EVALUATION.md)
- [PlaceSphere paired results](PLACE_SPHERE_EVALUATION.md)
- [PickCube / PlaceSphere collection and training summary](MULTITASK_OBSTACLE_RESULTS.md)

Older pilot and task reports remain source-controlled documentation. Their
statements that diagnostic artifacts were retained describe their original
publication state, not the post-cleanup filesystem.

## Historical relocation map

These mappings still resolve retained final artifacts without rewriting old
manifest/checkpoint paths. More specific checkpoint mappings take precedence.

| Historical path | Current path |
| --- | --- |
| `outputs/ckpt/base_dp_stackcube_980/` | `outputs/ckpt/20260930_base_dp_stackcube_980/` |
| `outputs/ckpt/constrainnet_wristcam_seed1/` | `outputs/ckpt/20261021_constrainnet_wristcam_seed1/` |
| `outputs/runs/dp_StackCube-v1_980_1/checkpoints/best_eval_success_at_end.pt` | `outputs/ckpt/20260930_base_dp_stackcube_980/best_eval_success_at_end.pt` |
| `outputs/runs/dp_StackCube-v1_980_1/` (training records) | `outputs/exp/20260930_base_dp_stackcube_980_seed1/` |
| `outputs/constrain_runs/wristcam_full30000_seed1/checkpoints/final.pt` | `outputs/ckpt/20261021_constrainnet_wristcam_seed1/final.pt` |
| `outputs/constrain_runs/wristcam_full30000_seed1/` (training records) | `outputs/exp/20261021_constrainnet_wristcam_seed1/` |
| `outputs/wristcam_formal_20261021/` or `outputs/data/wristcam_formal_20261021/` | `demos/StackCube-obstacle/` |
| `outputs/wristcam_eval_scenes_20261021/` | `outputs/exp/20261021_wristcam_scenes/` |
| `outputs/wristcam_paired_eval_20261021_recovery/` | `outputs/exp/20261021_wristcam_paired_eval/` |

Activate `maniskill` before any execution. New data must use an explicit path
under `demos/`; new weights belong in `outputs/ckpt/`, and experiment records in
  `outputs/exp/YYYYMMDD_HHMMSS_<name>/`. Do not overwrite the retained final runs.

## Round 22 — matched PlaceSphere DIRECT/OBSTACLE training and paired dev

This is the authorized matched-data optimization continuation of Ticket 06,
not formal evaluation or generalization evidence. The branch remains
`goal/transport-detour-optimization` at base `730b744495914b82f3c19415cfaf30f536779ba2`;
no checkout, branch change, commit, or push occurred. ManiSkill source and all
historical datasets/checkpoints were left unchanged. The earlier
`20261026_123000_matched_terminal_r21` dataset (213 unique obstacle layouts / 87
repeats) was excluded. The only training data were the qualifying
`20261026_133000_matched_terminal_r21b` package:

- HDF5 `eecfcdac7e357334eb68921c1e784e6858c3ccdd11d69c17e4e79b5cbf4556d0`;
  manifest `ff971fb2b9278126c20d37a7b43a3f9ad74d0a86adc5ce926922b0ad9f796e86`.
- Manifest-selected IDs are pinned in preregistration; the actual HDF5 order is
  interleaved (DIRECT IDs are even, OBSTACLE IDs odd). Counts are exactly 1,000
  DIRECT / 300 OBSTACLE, disjoint and covering 0–1299. The opt-in versioned
  reader cross-checks the manifest labels against HDF5 `kind`, rejects missing
  labels, wrong counts and unpinned identities, and verifies T+1 39/42 state,
  seven-channel native actions and successful terminal evidence. Historical
  formal readers remain locked to their original hashes and 1,000/100 IDs.
- Actual official training windows: 132,080 DIRECT / 50,077 OBSTACLE. The
  separate stride-8 diagnostic has 16,189 / 6,175 windows and is not the
  training-window denominator. Episode length median/p90/p95/max is
  105/147/321/909 DIRECT and 126/185/544/889 OBSTACLE. Actions beyond 200 are
  15.83% / 20.71% of action volume; open near-zero actions are 21.58% / 25.52%
  (28,723 / 12,855 actions). Since the tails are below 50% and the action-neutral
  share is independently substantial, the preregistered choice was uniform
   official DP window sampling: no phase weight, trajectory reweighting, or
   truncation. Expert tails do not prove policy sufficiency at 200 steps.

  A separate, non-selecting quality audit was calculated from each episode's
  saved current-action phase and per-step linear/angular speeds in the HDF5. It
  uses fixed non-overlapping 8-action windows and no future phase inference. The
  action-only open/near-zero predicate is all eight gripper actions open and
  max norm of the six pose channels ≤0.02; linear and angular native speed checks
  are counted independently (≤0.01 m/s and ≤0.5 rad/s):

  | Role | All stride-8 windows | Open/near-zero action-only windows | Linear static among action-only | Angular static among action-only | Linear-static / angular-only failure | Both native-static |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | DIRECT | 16,189 | 3,311 (20.45%) | 3,301/3,311 (99.70%) | 0/3,311 | 3,301/3,311 (99.70%) | 0/3,311 |
  | OBSTACLE | 6,175 | 1,513 (24.50%) | 1,506/1,513 (99.54%) | 0/1,513 | 1,506/1,513 (99.54%) | 0/1,513 |

  Reproduction command (read-only against the pinned HDF5):

  ```bash
  source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 300s \
    python -u -m examples.baselines.constrain.audit_placesphere_matched_windows \
    --data demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.h5 \
    --output outputs/exp/20261026_153000_placesphere_matched_paired_eval/matched_window_quality_posthoc.json
  ```

  All action-only windows are current-phase `settle`; 10 DIRECT and 7 OBSTACLE
  action-only windows fail the linear static check. Full report:
  `outputs/exp/20261026_153000_placesphere_matched_paired_eval/matched_window_quality_posthoc.json`.
  SHA `b0afc25631f84c745ddf486d203dd009438b19c71e15772fc35b203d2c0a6f5c`.
  This analysis was completed after preregistered training/evaluation, so it did
  not affect sampling, checkpoint choice, or gate interpretation. It is a
  process deviation from the requested order to resolve this quality gap before
  training; these rates are post-hoc, not preregistered independent
  angular-window coverage. Expert angular settling remains a bottleneck for
  parent review.

The before-training registration is
`outputs/exp/20261026_153000_placesphere_matched_paired_eval/preregistration_matched_r3.json`
(SHA `d0b7b88d5514198a5cae44ec4aff41b5bb1968b3162f580ff2956bc8144f0e02`),
which pins the dataset IDs/hashes, actual source file bytes, final-EMA selection,
30-scene seeds, budgets, and output paths. Training-source identity
`7c9bec6b3310ac6e3f2715a11f63a4b889b433f79dbce7f80ebbf98d6d0d0f79` was
verified before the Base and 90k runs. The earlier r1/r2 registration files in
the same experiment folder are superseded preregistration drafts: r1 had not
yet recorded calculated DP window totals, and r2 was generated before a trainer
syntax cleanup. Neither was used for any training or evaluation command.
Before evaluation, the adverse contact/XY
transition labels in the summary writer were corrected (episode inference was
unchanged) and re-frozen in preregistration r4, SHA
`f1a49951d652cfb30fd40a33b867f0664d0cc19efd1039d95ff63cf6b9a53adf`, source
identity `8fab4e962a21c8b67de70bd5272e6994f85595ad29ee7cf0e4e8b71b97527a0e`.

Preflight loaded the new Base EMA strictly into the PlaceSphere control model:
epsilon and full 100-step disabled-control equivalence both had max absolute
difference 0.0. The 300 obstacle IDs yielded 50,077 windows. Two finite
preflight updates verified finite loss, movement in the zero-initialized
control path, nonzero zero-convolution gradients and an unchanged frozen base.
An initial overly strong preflight assertion expected gradients in all upstream
and hint modules before those zero convolutions moved; those gradients are
structurally zero at initialization. No training had started in those attempts;
the corrected two-update check passed. All attempts terminated cleanly.
Before training, `py_compile` also caught an indentation error in the optional
trainer snapshot list; it was fixed and compilation/focused tests passed before
the final training registration. The first reader sample assertion expected a
42-D policy observation; the actual contract is 39-D policy plus 42-D raw, and
the assertion was corrected before registration. One process-inspection shell
check found `rg` unavailable; it was rerun successfully with `ps` piped to
Python. None of these issues changed dataset bytes or launched an unregistered
training/evaluation run. No `.tmp` checkpoints or experiment processes remained.

Exact finite-budget commands (all after activating `maniskill`):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=30s 2400s \
  python -u -m examples.baselines.constrain.train_placesphere_base_dp \
  --demo-path demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.h5 \
  --manifest demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.json \
  --run-dir outputs/exp/20261026_153000_placesphere_matched_base_dp \
  --checkpoint-dir outputs/ckpt/20261026_153000_placesphere_matched_base_dp \
  --device cuda --seed 1 --total-iters 30000 --batch-size 1024 --lr 1e-4 \
  --warmup 500 --ema-power 0.75 --matched-r21b

source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=30s 6000s \
  python -u -m examples.baselines.constrain.placesphere_constrain \
  --demo-path demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.h5 \
  --manifest demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.json \
  --base-checkpoint outputs/ckpt/20261026_153000_placesphere_matched_base_dp/final.pt \
  --run-dir outputs/exp/20261026_153000_placesphere_matched_constrain_90k \
  --checkpoint-dir outputs/ckpt/20261026_153000_placesphere_matched_constrain_90k \
  --device cuda --total-iters 90000 --batch-size 1024 --seed 1 --lr 1e-4 \
  --warmup 500 --ema-power 0.75 --matched-r21b
```

Base completed 30,000 updates in 886.36 s; losses were finite (first
1.16264045, last 0.00248973). Final EMA:
`outputs/ckpt/20261026_153000_placesphere_matched_base_dp/final.pt`, SHA
`4484b76f1767a4723c7a60f17c1cdd2b25afc6876bc0f732f66a0997919c6942`.
Base run config SHA `6cf5c963f1bb475489b826c7515b93591522c4b8f54d5f78ca692461ff7e6aff`;
summary SHA `a5cede581b8cb747d942159eff4f675c8162399c34428da88ef200dcaef49eb6`.
Control completed 90,000 updates in 2,533.85 s; losses were finite (first
0.02094785, last 0.00165384), frozen Base unchanged, control parameters moved,
and control-down, mid and hint gradients were nonzero. Pre/post disabled-control
epsilon and sampling equivalence max difference was 0.0. Final EMA:
`outputs/ckpt/20261026_153000_placesphere_matched_constrain_90k/final.pt`, SHA
`4c8f18c79c00188da785740bae5750c5989b8516a2f741712f3d5c833d915b88`.
Control run config SHA `ef7fe1925a735817c57263d8376759d301b703500749d93366dfbc53dd700be6`;
summary SHA `ed67aa7c98d3c11ce50714dd64fd8887e1dde6cec9ca3186784296d6a5a8d042`.

Paired evaluation reused the original frozen 30-scene development set from
`outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes` (the same
set used by the 2026-10-25 300/90k run), with identical restored states,
controllers, observations and per-scene seeds; max 200 control steps, 120 s per
episode, 8,000 s overall. The primary comparison is the old 300/90k EMA; the
old 100-obstacle EMA is secondary. New evaluation command:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s 8000s \
  python -u -m examples.baselines.constrain.evaluate_placesphere_matched \
  --scene-dir outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes \
  --base-checkpoint outputs/ckpt/20261026_153000_placesphere_matched_base_dp/final.pt \
  --constrain-checkpoint outputs/ckpt/20261026_153000_placesphere_matched_constrain_90k/final.pt \
  --output-dir outputs/exp/20261026_153000_placesphere_matched_paired_eval/paired_eval \
  --primary-baseline outputs/exp/20261025_085101_placesphere_obstacle300_90k/paired \
  --secondary-baseline outputs/exp/20261024_155244_placesphere_data_expansion/paired \
  --device cuda
```

Both new policies evaluated all 30/30 episodes, zero runtime errors. Base was
6/30 task, 0/30 safe, 30/30 contact, 24/30 XY; old 300/90k Base was
5/30, 0/30, 30/30, 24/30. New Constrain was 26/30 task, 25/30 safe, 1/30
contact and 0/30 XY. Against the old 300/90k Constrain
27/21/6/1: task improved/regressed/both/neither = 2/3/24/1; safe = 7/3/18/2;
contact cleared/introduced/both/neither = 6/1/0/23; XY = 1/0/0/29. Against
the old 100-obstacle Constrain 25/21/7/0: task = 5/4/21/0; safe = 8/4/17/1;
contact = 6/0/1/23; XY = 0/0/0/30. New Base versus old 300/90k Base task
transitions = 4/3/2/21; safe = 0/0/0/30; contact = 0/0/30/0; XY = 1/1/23/5.

The frozen gate is **pause / fail**: required task ≥27/30 and safe ≥26/30,
contact ≤3/30 and XY ≤3/30; observed task 26 and safe 25 fail, while contact
and XY pass. These are development results, not formal/generalization claims.
Compared with the old 300/90k run, this matched dataset plus layout-replacement
package improves safety/contact outcomes but does not isolate a pure expert
causal effect. Do not proceed to Ticket 08 or PickCube formal work from this
result without parent review.

Episode-level evidence is retained in
`paired_eval/{base_dp,constrain_dp}/episode_*.json`; derived summaries are in
`paired_summary.json` (SHA `4ea9288fa5fb6506d92b5fb0d753b21a4747857c1a92a9ac2b517a53dbcf67b7`)
and `episode_diagnostics.json` (SHA
`eb241ecd060233a743d3c8607b40bed97a8a17eed06ab12a81c4235fc9081537`). New
Constrain had one first contact: grasp-approach, Panda left finger/gripper on
the top cap. It ended ungrasped; 30 releases were recorded, with 2
`incorrect_release` and 2 `not_stabilized` failures. Successful native completion
steps: 26/30, median 107, p90 160, max 165; all successes occurred within the
200-step policy budget. Base first contact occurred 30/30 (29 transport, one
release/stabilization); Base had 24 incorrect-release failures. Terminal speed,
grasp, contact entity/phase/location and release records remain per-episode.
Frozen evaluation config SHA
`edc7e458d9caad81bb7e73d6d0bf638ac7442c659a8225960295b7c5f1bc81f6`. Base and
Constrain episode-directory identity hashes are respectively
`f2d668093cdb60cdd0420feb0ad7c8f1cc2f558b9dac67fa6c27c8e21c4fe896` and
`877cd6ad6c5faca05da52cea50970227a1f06ddaee4bc7d66c10c40dc6b8b2da`.

Full constrain test suite after the post-hoc window-audit check:
`timeout --signal=TERM --kill-after=20s 700s python -m pytest
examples/baselines/constrain/tests -q --tb=line` — **484 passed, 28 warnings,
83.23 s**. `git diff --check` passed. GPU training/evaluation processes
were gone at verification; the RTX 5070 returned to idle. No runtime checkpoint
or evaluation errors occurred. Files changed are limited to
`examples/baselines/constrain/` plus this index and Ticket 06 execution notes.

## Round 23 — measured PlaceSphere hold-8 continuation

User-authorized continuous Ticket 06 optimization on
`goal/transport-detour-optimization`, starting HEAD
`a402e6c6b01bd2622468e8f827fb13a076795d2a`. No checkout, commit, or push.
ManiSkill source and all retained Round-22 artifacts were unchanged. This is a
development experiment, not formal evaluation or generalization evidence.

### Current matched-checkpoint diagnosis

`placesphere_matched_diagnosis.py` is pinned to the Round-22 Base EMA
`4484b76f1767a4723c7a60f17c1cdd2b25afc6876bc0f732f66a0997919c6942`, Constrain
EMA `4c8f18c79c00188da785740bae5750c5989b8516a2f741712f3d5c833d915b88`, and the
unchanged 30-scene paired evaluation manifest. It traces indices 12/13/21/22/25/28
and safe Constrain controls 0/2/4/6 for both policies, with full raw observations,
queued seven-D actions, TCP/sphere/bin/cylinder fields, action queue/replan
indices, grasp/release/contact/crossing transitions, and post-action linear and
angular velocity vectors. Controller/action slice `[1,9)`, horizons 2/16/8,
scene-seed inference, 200-step horizon, and native limits (0.01 m/s, 0.5 rad/s)
are unchanged. All 20/20 traces per policy reproduced the saved success/safety/
contact/crossing/termination/step outcomes across 10 selected scenes per policy;
0 mismatches and 0 runtime errors.

| Constrain dev index | Result | First contact / release | Terminal linear / angular speed |
| ---: | --- | --- | ---: |
| 12 | native task success, unsafe | grasp-approach contact step 4, queue 3; left finger/top cap | 0.008181 m/s / 0.470576 rad/s |
| 13 | release/stability timeout | release step 86, queue 5 | 0.025526 m/s / 1.276365 rad/s |
| 21 | release/stability timeout | release step 169, queue 0 | 0.581996 m/s / 29.099939 rad/s |
| 22 | release/stability timeout | release step 114, queue 1 | 0.000720 m/s / 4.987443 rad/s |
| 25 | safe native success | release step 111, queue 6 | 0.002481 m/s / 0.126532 rad/s |
| 28 | release/stability timeout | release step 99, queue 2 | 0.001383 m/s / 5.122324 rad/s |

The sole contact is early grasp approach, not release. Four current 200-step
failures are angular/stability failures, including two with sub-threshold
terminal linear speed. That did not isolate queued-action staleness as the shared
cause. Full corrected diagnostic:
`outputs/exp/20261026_210000_placesphere_matched_diagnosis_r23d/`; diagnosis SHA
`d3780c27bd156af222527e7e0a63b9d34d5807e47b747e92a017c2e4b840711e`, preregistration
SHA `9f66f2827b4db02f0defc45cb9e460d76aaee2c790f303dde22a204fc48f11f8`.
The full T+1 expert velocity/action join SHA is
`956e5a5ad0b00018a1c37294709536d7969631d98288cea48b51e3d15569fa4e`. It aligns
each action to its *post-action* native state and checks against recorded
per-action velocity norms. Among the 300 matched OBSTACLE training releases,
linear-speed median/p90/max was 0.368188/0.373907/0.375557 m/s (300/300 above
native linear limit); angular-speed median/p90/max was 0.403734/0.438300/0.501188
rad/s (1/300 above angular limit). Of the 12,555 post-release settle
transitions, 12,457 exceeded the native angular limit and 138 exceeded the
linear limit; retained expert tails do eventually reach their first native
endpoints.

### Physics selection, data regeneration, and training

Before any dev-set result was inspected, a physics-only replay compared 0/8/16
open-neutral holds after the actual release transition on frozen training
OBSTACLE IDs 1/3/5/7/9/11/13/15. Each candidate reused the same full initial
environment/controller state and native action prefix, inserting only the
registered hold. Safe native success was 6/8, 7/8, 7/8; contact and crossing were
0/8 for all candidates; terminal speed-threshold passes were 6/8, 7/8, 7/8.
The frozen tie-break selected the shorter 8-step hold. Probe result
`outputs/exp/20261026_181000_placesphere_matched_hold_probe_r23f/result.json`
SHA `eb45994e780b7553b79ec966fb6ccf37f1628eb7ee4c6a78830d313f0b29b769`;
preregistration SHA
`adb7dfdb2bc4ef8f3c65ac378d3e2ab0d4da90c1c650f701df19368e37fadbbf`.

The full collector registration
`outputs/exp/20261026_183000_placesphere_hold8_registration_r23.json` SHA
`0ae6445a848cf5a046b1602fdebcca1364233f766a42a70a5a1aa73574e69af6` freezes
hold 8, source-catalog identity, direct/obstacle role rules, task-layout dedup,
and finite serial budgets. A 20+20 paired native collector pilot completed in
73 s and validated 20 unique paired obstacle layouts and 40/40 restore audits;
it is retained separately at `demos/PlaceSphere-transport-detour/20261026_183500_hold8_pilot_r23/`
and excluded from full training.

The regenerated full data is
`demos/PlaceSphere-transport-detour/20261026_184500_matched_terminal_hold8_r23/`:
1,000 DIRECT + 300 OBSTACLE, 1,004 layout attempts, 2 predeclared replacements,
4 rejected layouts, 3,104.39 s, 300 unique matched obstacle layouts, and 474/1,000
greedy-unique direct layouts (526 repeats, not claimed unique coverage). No old
trajectory arrays were copied. Independent validation passed 1,300/1,300 exact
T+1 restore audits, native seven-D actions, 39/42 state observation schema,
first-native-success evidence, full-prefix contact/crossing checks, and paired
negative/replay proofs. Dataset identity: HDF5 SHA
`2bd725179ed61c8c32b97cdc9b89ce348add0bbf6b4d7a912cfa8bad30b9a5a1`, manifest
SHA `8685d2303cb269c5e9ccda9964586c25406e40afdcd83c49ad6fd3adbb24fbad`,
training identity SHA `76fd854fcea553bd020f8e22ea25ab078552c903afe5bafd0bf3125ad61af70a`.
Independent validation SHA
`c7508ec2709252c33ad534df24be390a902c922baf8fde622dc90a6a11e607a1`;
collection report SHA `aa292bcaec9a0671a3df3074ddab29ec4cbb493b610d8d5a0349160ed5c0c08f`.

The before-training trial registration is
`outputs/exp/20261026_195500_placesphere_hold8_training_prereg_r23b.json` SHA
`32057e428d24e301b78e485a8cbfd1c8ddec56cd7c6d3962d80dab314e27ac84`; registered
source identity `8baa3958d6341d6e5e632b74fd8150297dc3fae9bedeb703c96f43ad6eafa059`
was rechecked with no source drift before training. It pins uniform official
window sampling, new matching data identity, seed 1, Base 30,000 updates,
Constrain 90,000 updates from the new Base final EMA, final-EMA weight selection,
the frozen 30 dev scenes, 200 policy steps, 120 s per episode, 8,000 s overall,
output paths, and original gate thresholds.

Base: 30,000 updates in 881.56 s, all losses finite (1.159990 → 0.002054), final
EMA SHA `92ac831b7dc1b1cdad4ddc9d7003813e174cb329507bdeb019e23f74d3aac19c`.
Run config SHA `bfcde1a99d908b29cdb569eb3af401ae33119aca8f96b3db100fbec449ffbaee`;
summary SHA `ddab69156d38f80e2f524497659470d2ecec56d6ca36878d1774ac1831cda35c`.
Constrain: 90,000 updates in 2,518.40 s, all losses finite (0.015319 →
0.001245), frozen Base unchanged, control branch moved with nonzero gradients,
pre/post disabled-control epsilon and 100-step sampler equivalence exact.
Final EMA SHA `c8d1404f68cf153176035c8d7d95d032517838f9df877b295bbefd5102f57ada`;
run config SHA `7bbe0dc6c4e89e9f1f5b5df332a083d6eda19a0cc398c041f2a938dce8385d2f`;
summary SHA `8fab4eead8d240205a6db64b3f9395b2899d115453eb47a82d89784aae12303d`.

### One paired dev trial and gate

The one registered evaluation used the exact Round-22 frozen 30-scene manifest,
200 steps, both policies, and zero runtime errors (30/30 each). Hold-8 Constrain
scored 25/30 task, 25/30 safe, 0/30 contact, 0/30 XY; successful native
completion steps were median/p90/max 113/160/184. Against the primary current
matched Round-22 Constrain (26/25/1/0): task improved/regressed/both/neither
4/5/21/0; safe 5/5/20/0; contact cleared/introduced/both/neither 1/0/0/29;
XY 0/0/0/30. The previous contact cleared at scene 12, but native task success
regressed at scenes 5/10/11/17/24 while former failures 13/21/22/28 recovered.
The five current failures were all release/stability timeouts at the 200-step
limit: scenes 5/10/24 stayed on-bin but spinning; scenes 11/17 were high-motion
off-bin releases. The hold removes the single contact but does not clear the
remaining release/stability bottleneck.

**Gate: pause/fail.** Task 25 < 27 and safe 25 < 26; contact and XY pass.
This is a development result, not formal evaluation. Paired summary SHA
`63f316e6188c1ce2ea4b0c454c1552de67b63510bbca0aae13aa7348a54c8f04`; frozen
config SHA `fef33e7ca0f8319d043cfc7bdf5fcb74654a1ac12bf104e7825df94a6f8c3d88`.
The matched current EMA baseline and per-episode files remain intact.
Persistent Round-23 argv/budget/outcome ledger:
`outputs/exp/20261026_195000_placesphere_hold8_paired_eval_r23/round23_execution.json`
SHA `a30b1436e315413befa5013e30ee906d3310e27d9f0325c647d5556739f5f41f`;
all frozen training source hashes still match.

Final full constrain suite after Round-23 code additions:
`timeout --signal=TERM --kill-after=20s 700s python -m pytest examples/baselines/constrain/tests -q --tb=line`
— **498 passed, 28 warnings, 82.41 s**. `git diff --check` passed. Training,
evaluation, diagnostic and collection processes exited; no leftover `.tmp`
checkpoints or experiment processes were present. Ticket 08 remains blocked. The
next data-backed seam is current angular release/stability under the 200-step
policy budget; no additional R23 factor or formal evaluation was started.

## Round 24 — indexed release mechanics and exploratory execution stride 2

User-authorized single-developer continuation on
`goal/transport-detour-optimization`, starting at base
`9f190f80665810d55e35c3436710a0aeb2b2d95b` (integration branch `dev`). No
checkout, branch change, commit, or push. ManiSkill source, training data, and
checkpoint bytes were not changed. No training, data collection, formal
evaluation, or PickCube work was run.

### Release-mechanics evidence

Native Base DP and Constrain episodes were instrumented with full per-step
seven-channel actions, observation/replan step, queue index, TCP xyz/quaternion,
sphere xyz/quaternion and linear/angular velocity, grasp/open state, obstacle
contact/crossing flags, control timestep, and termination flags. The Constrain
trace reproduces all five R23 failures (scenes 5, 10, 24, 11, 17) and includes
the preregistered matched-success controls (0–4). Full traces remain inside
`paired_eval/{base_dp,constrain_dp}/episode_*.json`; indexed open-command and
physical `is_grasped: true→false` event joins are in
`release_causal_measurements.json`.

The training-expert audit reads the regenerated R23 hold-8 300-OBSTACLE HDF5,
not the older R22 expert. It verifies 48,982 action/state pairs: action index
`t` is compared to the T+1 native state `t+1`, and each of the saved linear and
angular speed values matches that same post-action state. The 300 open-action /
ungrasped transitions have pre-open linear speed median/p90 0.00549/0.00678 m/s
and angular speed 0.02665/0.03084 rad/s; same-index post-action medians/p90 are
0.36818/0.37391 m/s and 0.40389/0.43830 rad/s. This is a synchronized
before/after kinematic association, not an impulse or proof of cause. Expert
per-action timestep is absent, so expert `g·dt` is not calculated.

The five Constrain failure replays show pre-open motion ranging from 0.0030 to
0.0223 m/s linear and 0.0247 to 0.3560 rad/s angular. At the physical
ungrasped transition, downward linear velocity magnitude is about 0.364–0.378
m/s. The following 0.05 s displacement is about −0.0329 to −0.0336 m, compared
with −0.0123 m under the zero-initial-velocity `½g·dt²` comparison; observed
`dz/dt` and `g·dt` are recorded per event. This does not isolate release
actuation, contact, gravity, or pre-existing motion. Per-step sphere/robot
contact is not exposed in saved native step info (obstacle contact is); joint
torque and impulse are not measured. Do not infer a dynamic impulse from these
records.

### Preregistered stride-2 trial

Before evaluation, registration
`examples/baselines/constrain/placesphere_stride2_preregistration.json` pinned
the 30-scene manifest, exact source hashes, Base EMA
`92ac831b7dc1b1cdad4ddc9d7003813e174cb329507bdeb019e23f74d3aac19c`, and
Constrain EMA
`c8d1404f68cf153176035c8d7d95d032517838f9df877b295bbefd5102f57ada`. One
factor changed: each policy call still predicts its original 16×7 sequence and
selects the original `[1:9]` actions; execution consumes two actions, appends
the latest two observations, and replans. Observation history remains 2, all 7
action dimensions, weights, sampler, scene seeds, clamps and safety behavior
are unchanged. More calls naturally consume more sequential diffusion RNG
draws; this coupled sampler-noise change is disclosed. This is explicitly
**EXPLORATORY**: more frequent feedback may help release errors, but R23 did not
establish that stale queued actions caused them.

Both policies ran all 30 scenes with the same restored state/controller/seed,
200 control steps, native first-success termination, a 300 s episode wall cap,
and a fixed 30 denominator. The 18,000 s outer timeout is compute-only; no
scene was dropped and no result-based stopping/selection occurred. The exact
command and exit/cleanup details are in `execution_record.json`.

| Policy | Task | Safe | Contact | XY crossing | Runtime errors | Successful steps median/p90/max |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Base DP stride 2 | 8/30 | 0/30 | 30/30 | 25/30 | 0 | 160.5/185.5/189 |
| Constrain stride 2 | 26/30 | 26/30 | 1/30 | 1/30 | 0 | 135.5/158.5/171 |

Against primary R23 Constrain stride 8 (25/25/0/0), stride 2 has task/safe
transitions improved/regressed/both/neither = 4/3/22/1, and contact and XY
cleared/introduced/both/neither = 0/1/0/29 for each. Four of the five former
release/stability failures (5, 10, 24, 11) now succeed; scene 17 remains a
release/stability timeout at step 200. The one new contact/XY episode is scene
25. Against secondary R22 matched hold-0 Constrain (26/25/1/0), transitions are
task 4/4/22/0, safe 5/4/21/0, contact 1/1/0/28, and XY 0/1/0/29. R22 is
secondary and not a one-factor comparison.

The frozen gate remains **pause/fail**: task 26/30 is below 27; safe 26/30,
contact 1/30 and XY 1/30 meet their thresholds. The remaining scene-17 failure
and single introduced contact/crossing are retained; no rerun or additional
stride trial was started. New result directory:
`outputs/exp/20261026_235900_placesphere_stride2_r24/` (registration,
`paired_eval/`, per-episode full traces, `release_causal_measurements.json`, and
`execution_record.json`). Paired summary SHA
`3c537c1ba646e5d483e2fd4075524ba36bc0ea08eae4356cc4e0231276c2b0cd`; expert
HDF5 SHA `2bd725179ed61c8c32b97cdc9b89ce348add0bbf6b4d7a912cfa8bad30b9a5a1`.

Verification: constrain suite **502 passed, 28 warnings, 83.65 s**; `git
diff --check` passed. Full evaluation exited 0 with 60/60 episodes and no
runtime errors; model/evaluation child processes exited. Next minimal slice is
parent review of the indexed event traces, especially scene 17 and why the
post-action expert velocity jump appears in the matching T+1 record. Do not
claim causality or launch further trials in this round.

## Round 25 — candidate-collector interruption diagnosis and bounded resume canary

The obstacle1000 collection did **not** reach matched demonstration collection,
training, or paired evaluation. The checked-out branch remains
`goal/transport-detour-optimization` at HEAD
`6aeee1ecf559aa9e247e3348ff11f308595e94fe`; no branch operation, commit, push,
or ManiSkill source edit occurred. All native commands ran after activating
`maniskill` and had finite outer/per-attempt timeouts.

### Exact stall and retained work

The before-collection plan is
`outputs/exp/20261026_231800_placesphere_obstacle1000_r25/preregistration_collection.json`
(SHA `34d0b21272b34fefb7a5f0b5a4fc2469d73541877c57ca84455d2e87dc8e1cbe`). The
first geometry-candidate run in
`outputs/exp/20261026_231800_placesphere_obstacle1000_r25/candidate_source/`
requested 700 layouts under a 21,600 s wall / 10,000-attempt cap. It stopped
after 21,648.57 s: 2,941 attempts, 223 accepted safe geometry layouts (7.58%),
and 2,718 retained rejections. These are scene/layout proposals, not the new
hold-8 matched training episodes. Its seed interval was 2026102712–2026105652.
HDF5 SHA `cc92ddcc86faf5ebb6a2c2ac1b21fe30a55975bc6072ada81a55b9dbc477a2d4`,
manifest SHA `910cb1c8db1d5bd675624e7854c0f9f0c4f0f73ccd9bd0e05192a67175946fca`,
attempt journal SHA `6676d01ac0752b040a25f053fb8952127f214bf68b5c7fcad83d05426b4720d5`.
Rejection counts: 1,036 task-layout duplicate; 638 direct-template
unsuccessful; 552 side-detour column infeasible; 262 direct block at endpoint;
160 direct container unsuccessful; 27 waypoint out of bounds; 15 obstacle
native unsuccessful; 15 side-template unsuccessful; 13 obstacle contact or
crossing.

The next run reused the initial seed argument rather than starting after the
last journaled seed. Its 3,197 attempts were 2026102713–2026105909, overlapping
2,940 of the first run’s seeds and replaying those expensive physics searches.
It consumed a further 24,114.62 s, retained 16 accepted scenes and all 3,197
journal rows, but was interrupted before its final report/manifest write. The
root cause was the resume cursor, not training or GPU capacity. The incomplete
continuation HDF5 SHA is
`4a7394345c576994eb7709c3d6696ab8c42689ca6881ab6aa018966886ee45a9`; journal
SHA `4c81fdbfbdbdd08baa883930da6e79dfab037a35947c285dc95c91ecacc47db6`.
Those 16 journal-accepted scenes are at IDs `placesphere-000..015`; each has
matching seed metadata, finite 7-D expert actions, T/T+1 39/42 observations,
native success, and full-prefix safe-success flags. They were task-unique against
the first 223 and the preregistered exclusion catalogs. They are retained as
candidate-scene provenance only, not as training demonstrations. Total valid
candidate-layout records recovered: **239/700**, not 223+16 discarded or
recollected.

Recovery manifest
`outputs/exp/20261027_100700_placesphere_obstacle1000_r25/recovered_candidates_239/manifest.json`
(SHA `89b180a4d2cd6e3685d4b7b1eec4a76b0312bf4c7917b5d3ed530251af246ba3`)
joins the 223 complete first-run records and 16 safely recovered continuation
groups without editing either source directory. All 239 pass the frozen
seed-or-both-endpoints-within-0.02-m identity rule. Their native source arrays
remain in the two original candidate HDF5s.

### Bounded resume repair and canary

The repaired cursor scans every prior attempt journal, including rejected
attempts, rejects overlapping seed histories, and resumes strictly at
`max(attempted_seed)+1`. For the two long-run journals this is seed
2026105910; the canary then consumed seeds 2026105910–2026105917, so the next
full-batch first seed is **2026105918**. A cheap preregistered proposal replay
uses the same `default_rng(seed)` start/goal draws and geometry checks before
opening PhysX. It skips only geometry-exhausted or task-identity duplicate
proposals; all other proposals continue through the unchanged expert physics
and post-execution safety predicate. The candidate generator now atomically
writes `recovery_checkpoint.json` each attempt with accepted scene IDs/layouts,
last/next seed, attempted count, wall, source identity and journal byte count.
Forensic reconciliation checks accepted journal rows against recovered HDF5
groups and fails closed on a count/identity mismatch.

The resume/canary registration is
`outputs/exp/20261027_100700_placesphere_obstacle1000_r25/candidate_resume_canary_preregistration_r2.json`
(SHA `a6a7cd357d6aaa214a6b75b30f2f8e1e4558209a1a35cbf6f5a4b380172e8ba8`). It
records the 2,940-seed overlap, all source/artifact hashes and the exact cursor.
The 8-proposal / 300-s canary ran 20.67 s: 5 known duplicates were rejected
before environment construction, 3 new proposals reached physics, 0 were
accepted (one direct-template failure and two side-column failures), and all 8
attempts were durably recorded. This proves fresh-seed progression and the
preflight/progress path; it does not establish a new expert acceptance rate.
Canary journal SHA
`bcdffec1bb21b7be66f3a4dd8f815a1b1900f423978ed46bf62a1699d4f58499`, report SHA
`e7ca605d33c6f16c854fe5f8dc9c806bc31262efd84b5773493a03bf716a05db`, recovery
checkpoint SHA `3f5f4bbfd0ec98f14642a8ffa19153a5f2198b7a311625ba1a4f41612f6b464b`.

The next bounded continuation plan is frozen at
`outputs/exp/20261027_100700_placesphere_obstacle1000_r25/full_candidate_continuation_preregistration_r3.json`
(SHA `3895c80446ee163fc0c32ef599944145b20487acc7565a35ee2eebfaac64dd67`). It
uses first seed 2026105918, targets the remaining 461 candidate layouts, and
has an 8,836-attempt / 65,000-s budget derived from 16 accepts in 265 unseen
continuation+canary seeds, plus a fixed 1,200-attempt buffer. There was **no
full continuation, matched 700-pair collection, dataset assembly, training,
checkpoint verification, or R25 paired evaluation** in this bounded repair
round. The original 1,000 DIRECT + 300 OBSTACLE hold-8 dataset and both R23
checkpoints remain unchanged. R25 gate status is therefore not evaluated; the
fixed R24 stride-2 gate remains the last measured result.

The above R25 continuation is **superseded** by the user's newer fixed 1,000
DIRECT / 500 OBSTACLE limit. No 461-layout continuation, candidate expansion,
or 1,000-obstacle training was launched. The following R26 section records the
active frozen-data continuation and its final development result.

Verification after the resume fix: focused recovery/profile suite **5 passed**;
full constrain suite **507 passed, 28 warnings, 617.70 s**; `compileall` and
`git diff --check` passed. The canary and test child processes exited; no
collector, trainer or evaluator process remained. No episode was selected using
policy metrics. Detailed source edits are confined to
`examples/baselines/constrain/`; hashes are in the preregistrations and
canary/checkpoint records above.

## Round 26 — frozen 1,000/500 matched-data continuation

The user's newest constraint is fixed at **1,000 DIRECT / 500 OBSTACLE** and
prohibits adding layout candidates. This supersedes the historical 1,000-obstacle
target and the R25 461-candidate continuation plan above. No layout proposal was
sampled in R26; only layouts in the recovered, immutable 239-candidate pool were
replayed. The branch remained `goal/transport-detour-optimization` at
`6aeee1ecf559aa9e247e3348ff11f308595e94fe`. No checkout, branch change,
commit, push, or ManiSkill-source edit occurred. The existing Lightnav process
was left untouched.

### Frozen candidate replay, external-identity correction, and final data

The frozen-pool replay registration is
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/frozen_replay_selection_r5.json`
(SHA `42eba7cd617cc7d47b8fc31d3f9fdaf4a35f48a8b2b2877a091748fe562f96a6`). It
pins the recovered pool manifest (`89b180a4d2cd6e3685d4b7b1eec4a76b0312bf4c7917b5d3ed530251af246ba3`),
both candidate-source HDF5 hashes, replay code hashes, deterministic
`(seed, original_id)` ordering, margin 0.015 m, hold-8 terminal contract, and a
one-pass maximum of 239 existing layouts. The ordered first 200 candidates were
replayed serially in 1,289.89 s; all 200 matched direct/obstacle/negative/suffix
admissions and both positive T+1 restore audits passed. No expert/restore
rejection or timeout occurred. The separate candidate-obstacle staging HDF5 SHA
is `dd5de15459201edb275ada9002c557d84474e95acd0ad718d1ec3a58a86cbd8a`; its
manifest SHA is `35652421107ec68a5f62b800f59c24e63217404ae047354e123c95f5616c569a`.
The complete first-pass ledger SHA is
`4ac08310aa66cf82c1b01d852ebbf70f07349baa320452019a03f1facdd28721`.

An independent external task-identity audit then found four of those admitted
layouts conflicted with Ticket 04: candidate IDs **1, 80, 84, 159**. They were
excluded from training regardless of their positive expert outcome. Their prior
package and hashes remain preserved as non-training history; the correction
sidecar explicitly supersedes its stale admission claim. Before replay, a second
registration
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/replacement_replay_preregistration_r2.json`
(SHA `f97a60c43ca7893596f82987cbf81e0182927925b610ec9f08755fe02a236392`)
froze only the remaining existing pool IDs, external manifests, replacement
order, code hashes, and four-replacement target. Existing candidates **200–203**
were replayed in order in 46.92 s; all four passed. Candidate IDs **204–238**
remain unused and are not training data. No new layout was sampled. An initial
standalone candidate-0 state-restore smoke was retained separately as
`first_candidate_attempt.{json,h5}` and is not a second training demonstration.

The active audited package is
`demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/`.
The four Ticket-04-conflicting candidates are absent. The new OBSTACLE class
uses source candidate IDs 0–199 excluding 1/80/84/159, plus 200–203, mapped to
final HDF5 IDs **1300–1499**. Its final HDF5 SHA is
`796c7a65f4ac3d9b8f434f71764ab393eec4257c217a79d4351f338e78993961`, manifest
SHA `aebbb0d79eed376b8c6a79e10b297751ee51077d168e03ceb12db643358f9fc9`, and
training-identity SHA
`648f59863a747d603ec8dc6ccf056995e60a93f35b1875bf0b230a261f28fe78`. The
independent saved-artifact audit
`independent_r26_audit.json` (SHA
`35bd68b10bd4139e0e2033394e69f936fa1a568e40a89897bf5088d7dbb7e4e3`)
verified 1,500 total episodes, 1,000/500 class counts, exact original IDs
0–1299 HDF5 group-value preservation, all 200 new T/T+1 native 7-D / 39-D / 42-D
contracts, first-native-success and no-truncation evidence, 400 positive role
T+1 restore proofs, 200 negative transport-block proofs, 200 actual-cylinder
terminal-suffix proofs, and full-prefix obstacle contact/crossing safety.
Final task-identity intersections are **0** against the original obstacle-300,
frozen dev30, and Ticket 04 sets. OBSTACLE layouts are **500 unique / 0 repeats**.
The retained DIRECT class has **474 greedy-unique / 526 repeated** layouts;
these repeats are historical DIRECT content, not new unique-layout claims.

The initial r2 assembled package remains unchanged but is **not** an active
training input because of the four Ticket 04 overlaps. Its correction record is
`training_package_r3/external_task_identity_correction.json` (SHA
`4d5b987011bbd3130c17da57cfb8766fb3d129ed8f88ab31da4d21a94d8abbc3`). Neither
the historical HDF5 nor its original manifest/training-identity hashes were
rewritten.

### Before-training window audit and control-only training

Before the training registration, the active 500-OBSTACLE identity reader loaded
exactly 500 IDs and **80,151 official windows**. The matched R23 300-OBSTACLE
source has 48,682 windows; the ratio is 1.646. Training uses the preregistered
150,000 updates (`90,000 × 500/300`), yielding 1,916.38 cohort-equivalent window
passes versus 1,893.10 for the old 90k/300 run (about 1.2% higher). Data volume
and update budget change together; this is not a separated causal data-volume
effect.

The final package's complete, non-truncated action/window audit is in
`dataset_validation.json` (SHA
`edd744337e4ee1cdc0a6d1a29788ec7277f9196e08dd972293d966f5b0182e89`). DIRECT /
OBSTACLE episodes longer than 200 actions: **64/1,000** and **44/500**; action
volume beyond step 200: **17.83% / 17.59%**. Open near-zero actions: **27.17%
/ 24.97%** of action volume. Fixed stride-8 open/near-zero action-only windows:
DIRECT **3,739/16,610**, OBSTACLE **2,055/9,859**. Separate linear-static counts
are **3,647/3,739** and **2,012/2,055**; angular-static counts are **0/3,739**
and **0/2,055**; both-native-static counts are zero. Uniform official-window
sampling was retained; no phase weighting, trajectory weighting, tail
truncation, policy-performance filter, or artificial neutral window was used.

The before-training registration is
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/training_preregistration_r1.json`
(SHA `08f361e950586758283ebd7f472d264fbb77b2a1e5ea29e55047feffcebb7a7a`). It
pins all dataset/source/EMA/dev/baseline hashes, the exact 150,000-update
training and single stride-2 evaluation plan, exposure calculations, budgets,
and gate. The two-update CUDA preflight record is
`training_preflight.json`: finite losses, moved zero-convolution weights,
nonzero zero-convolution gradients, moved trainable control, bit-unchanged
Base, and zero disabled-control epsilon/sampler difference. No preflight
checkpoint was selected or used.

Exact training command (9000-s timeout):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s 9000s \
  python -u -m examples.baselines.constrain.placesphere_constrain \
  --demo-path demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/trajectory.h5 \
  --manifest demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/trajectory.json \
  --base-checkpoint outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23/final.pt \
  --run-dir outputs/exp/20261027_140000_placesphere_hold8_500_r26/training_control_150k \
  --checkpoint-dir outputs/ckpt/20261027_140000_placesphere_hold8_500_r26_constrain_150k \
  --device cuda --total-iters 150000 --batch-size 1024 --seed 1 --lr 1e-4 \
  --warmup 500 --ema-power 0.75 \
  --matched-data-identity demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/training_identity.json
```

Training completed 150,000 updates in **4,316.23 s**. Losses were finite
(0.02049344 → 0.00134501); frozen Base unchanged; control parameters and all
control-down/mid/hint gradient families moved; disabled-control epsilon and
100-step sampler maximum difference were both 0.0. Final EMA:
`outputs/ckpt/20261027_140000_placesphere_hold8_500_r26_constrain_150k/final.pt`,
SHA `ff388efb87e4fde6acaaf21778aaee25b93883af990b99ad510fa82f97ccbf71`.
Run-config SHA `a560490d198d7b62ab790556a05a243ba739beae4fe98a41518ccd29ea744489`;
training-summary SHA `7dd5ba8bc1ca5bcfc7440fd1256028d47dffe798be29bdbceeba43e3218365bc`.

### One paired stride-2 dev run

The sole evaluation was preregistered in
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/evaluation_preregistration.json`
(SHA `1b304218b462e6cde8ff63dab1463e8a2da51cb73920d7a3a19839fdb6513a81`). It
reuses the current frozen 30-scene dev states/controllers/observations/seeds,
stride 2, prediction horizon 16, executed action slice [1,9], 200 control steps,
300 s per episode, fixed denominator 30, and the R24 stride-2 baseline. No
stride grid, scene addition, checkpoint selection, or second paired run occurred.

Exact evaluation command (18,000-s outer timeout):

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s 18000s \
  python -u -m examples.baselines.constrain.evaluate_placesphere_matched \
  --scene-dir outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes \
  --base-checkpoint outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23/final.pt \
  --constrain-checkpoint outputs/ckpt/20261027_140000_placesphere_hold8_500_r26_constrain_150k/final.pt \
  --output-dir outputs/exp/20261027_140000_placesphere_hold8_500_r26/paired_eval_stride2 \
  --primary-baseline outputs/exp/20261026_235900_placesphere_stride2_r24/paired_eval \
  --device cuda --replan-stride 2
```

Both policies completed **30/30** with zero runtime errors. Base: **8 task / 0
safe / 30 contact / 25 XY**. Constrain: **25 task / 25 safe / 0 contact / 0
XY**. Against R24 stride-2 Constrain (26/26/1/1), task improved/regressed/both/
neither = **2/3/23/2**, safe = **2/3/23/2**, contact cleared/introduced/both/
neither = **1/0/0/29**, XY = **1/0/0/29**. The five new failures are release/
stability timeouts, with no contact or crossing: scenes 8 (release step 155,
0.3740 m/s, 0.3204 rad/s, off-bin), 13 (109, 0.3341 m/s, 1.2531 rad/s,
off-bin), 15 (106, 0.3717 m/s, 0.2945 rad/s, off-bin; terminal on-bin but
spinning at 3.6033 rad/s), 17 (73, 0.3892 m/s, 0.3047 rad/s, off-bin), and 25
(117, 0.3183 m/s, 0.9206 rad/s, off-bin). Successful native steps were median
114, p90 173, max 179. Episode-level release/contact/terminal records and file
hashes are indexed in `paired_eval_index.json` (SHA
`e860a8e8721b4392e0ac18be708f85c0d19a3d8f4495d654e9ede75e428f48cd`). Paired
summary SHA `cac80011768efd2f94ed823cfcdef6646f4c01d0e1348abd9747f0fc95cef07e`;
frozen config SHA `e82bc5df0e70e4ca7bf9286d1cfe36fbfcedbe81f1d864020c1b00c7abb97d9d`.

**Gate: pause/fail.** The fixed requirements are task ≥27/30 and safe ≥26/30;
observed 25/30 for each. Contact and XY are 0/30 and pass. This is repeated
development-set evidence only, not a formal/generalization claim. No new-data
collection is authorized by this result. Continue only with optimization on the
frozen 1,000/500 data and the same dev set after parent review.

Final constrain suite: **512 passed, 28 warnings, 346.62 s**. `git diff --check`
passed. Training, replay, and evaluation children exited; the RTX 5070 returned
idle. Machine-readable command/budget/outcome and input/output hashes are in
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/round26_execution_record.json`
(SHA `acee5cade32864ac2af672fa72877117e63e85711c34d69cdf0147070b56ec97`).

After the data/training/evaluation run, all old R25 expansion/canary/461-layout
planning and execution entry points in `placesphere_obstacle1000_expansion.py`
were made fail-closed under the active freeze. No data, training, or evaluation
was rerun after that guard-only change. The R26 replay and replacement
registrations retain the exact source hashes used at those stages; the
post-run source transition and final verification are recorded in
`postrun_obsolete_candidate_guard_audit.json` (SHA
`deeea2e823295cecd9121f2e2d616fd8eec56adceb42870c140d8eab601ff1a8`). The final
post-guard test run is **512 passed, 28 warnings, 309.59 s**.

## Round 27 — frozen-data DDIM inference trial (preregistered)

The active authorization remains fixed at **1,000 DIRECT / 500 OBSTACLE**; no
new data, layouts, or trajectory collection is included. Round 27 registers
exactly one inference-only change for both already-trained policies: runtime
`DDIMScheduler`, 100 inference steps, `eta=0`, created from the loaded
training-time DDPM scheduler config (squared-cosine betas, epsilon prediction,
sample clipping). The training checkpoint contract remains DDPM. Reverse
variance draws are removed, so later global RNG positions also differ; this is
one sampler-configuration comparison, not an isolated variance-only causal
effect. Default DDPM adapter call paths remain unchanged.

The preregistration is
`outputs/exp/20261027_204500_placesphere_ddim100_r27/inference_preregistration.json`.
It pins the exact r26 HDF5/manifest/training-identity, Base EMA and final
Constrain EMA hashes and paths, all 30 frozen dev scenes, R24 stride-2 baseline,
runtime source hashes, one 60-policy-episode paired budget, per-episode limits,
and the gate. Both policies retain stride 2, prediction horizon 16, executed
slice `[1,9]`, same saved state/controller/observation/seed reset, native 7-D
actions, and existing action mapping/clamps. Runtime errors remain in the fixed
denominator. The evaluator fails closed on any preregistered artifact/source
identity mismatch before creating an evaluation output or executing a scene.

The sole allowed command is:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s 18000s \
  python -u -m examples.baselines.constrain.evaluate_placesphere_matched \
  --scene-dir outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes \
  --base-checkpoint outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23/final.pt \
  --constrain-checkpoint outputs/ckpt/20261027_140000_placesphere_hold8_500_r26_constrain_150k/final.pt \
  --output-dir outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval \
  --primary-baseline outputs/exp/20261026_235900_placesphere_stride2_r24/paired_eval \
  --device cuda --replan-stride 2 --sampler ddim \
  --inference-preregistration outputs/exp/20261027_204500_placesphere_ddim100_r27/inference_preregistration.json
```

Registration-time checks: focused sampler/PlaceSphere/pilot tests **64 passed**;
full constrain suite **517 passed, 28 warnings, 328.18 s**; CLI import/help and
`git diff --check` passed.

### Single paired run and result

The sole registered command completed normally from **2026-10-27 21:00:48.570
to 21:22:35.919 +08:00** (1,307.348 s, below the 18,000 s cap). It evaluated
**30/30 episodes for each policy**, retained zero runtime errors, and had 60/60
bit-exact scene restores with matching paired scene IDs/seeds. No episode was
dropped/retried. The RTX 5070 returned idle; no unrelated compute process was
started or stopped. The frozen data, training identity, and both checkpoint
hashes were identical before and after.

DDIM Base achieved **6 task / 0 safe / 30 contact / 24 XY**. DDIM Constrain
achieved **26 task / 26 safe / 0 contact / 0 XY**. Its 26 successful native
episodes had median / p90 / max success step **125 / 171 / 171**. Against R24
stride-2 Constrain (**26/26/1/1**), task and safe transitions were
improved/regressed/both/neither **3/3/23/1**; contact and XY were each
cleared/introduced/both/neither **1/0/0/29**. Three R24 failures (indices
17/23/25) cleared and three prior successes (6/11/14) regressed; index 9 failed
in both. All four new failures were incorrect releases, with no contact or
crossing: indices **6, 9, 11, 14**, release steps **136, 161, 78, 82**. Release
linear speeds were **0.389, 0.405, 0.425, 0.367 m/s**; every release was off-bin
and high above the bin (relative z **0.151, 0.155, 0.109, 0.140 m**). Per-step
contact/release/terminal mechanism traces are retained in their episode JSONs.

**Gate: fail/pause.** Task success 26/30 is one short of ≥27; safe 26/30 passes
≥26; contact 0 and XY 0 pass ≤3. This is frozen-development-set evidence only.
No formal 08 evaluation or new collection follows. Parent review selects the
next frozen-data optimization slice before any further experiment.

The machine-readable execution record is
`outputs/exp/20261027_204500_placesphere_ddim100_r27/round27_execution_record.json`;
SHA `5a388a8231cb2f1b7bfbc99487abb6d8663182e23304f205c9f5ce0af5335b8f`. It pins
the exact command/budgets/timestamps, all input and output summary hashes, the
64-file evaluation-tree digest, resource/cleanup state, comparisons, failure
indices/mechanisms, and gate transition. The full paired summary is
`outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval/paired_summary.json`
(SHA `ef24daa752f99ce31d5063325397c1276f3e6bc3d085dcaeafa824904378346e`); the
evaluation log SHA is `fed54e6d0256e7878b164a32a7fc6e776a0d5c0f5a23ddf999a9d3fb12a989cd`;
the paired-output tree SHA is
`6c47aab68cfe5947d0671b491a317e61355b422ee4d02559933fbecb15cd80c5`.

## Round 27 comparison correction — R26 is primary; R24 is secondary

**Correction to the Round 27 interpretation above:** the DDIM trial was compared
there to R24 as if it were the primary baseline. That was wrong. The required
primary is the immediately preceding **R26 DDPM100 stride-2** run, which uses
the exact same Base and Constrain checkpoints, frozen scenes, seeds, and rollout
limits. R24 has different data/checkpoint generation and is secondary context,
not evidence for this sampler comparison.

The actual R26 episode path is
`outputs/exp/20261027_140000_placesphere_hold8_500_r26/paired_eval_stride2/`.
Its frozen config SHA is
`e82bc5df0e70e4ca7bf9286d1cfe36fbfcedbe81f1d864020c1b00c7abb97d9d`, paired
summary SHA `cac80011768efd2f94ed823cfcdef6646f4c01d0e1348abd9747f0fc95cef07e`,
and its registered R26 evaluation preregistration SHA is
`1b304218b462e6cde8ff63dab1463e8a2da51cb73920d7a3a19839fdb6513a81`. R26
records Base **8/0/30/25** and Constrain **25/25/0/0** (task/safe/contact/XY).
Its original command omitted a sampler override; the registered R26 evaluator,
Base adapter, and Constrain adapter source hashes establish the original
DDPM100 path. The R26 and R27 runs pin the same Base EMA SHA
`92ac831b7dc1b1cdad4ddc9d7003813e174cb329507bdeb019e23f74d3aac19c`, Constrain
EMA SHA `ff388efb87e4fde6acaaf21778aaee25b93883af990b99ad510fa82f97ccbf71`,
scene manifest SHA `9e1d8ee4786d2336f01e07d664ca37c3dcc7ca455e7876176def65c18b63c119`,
and scene HDF5 SHA `adc7d1b2a16745c459db53d2c8f818ac87d1ffffe21fd44c54509f77cef342d7`.
Both use 30 scenes, stride 2, prediction horizon 16, executed slice `[1,9]`,
200 control steps, and 300 s per-episode wall cap. The correction analysis
verified all per-policy scene IDs/seeds, bit-exact restores, and exact initial
policy observations across R26 and R27.

Correct primary transitions, ordered improved/regressed/both/neither for
task/safe and cleared/introduced/both/neither for contact/XY:

| Policy | R26 DDPM100 task/safe/contact/XY | R27 DDIM100 task/safe/contact/XY | Task | Safe | Contact | XY |
|---|---:|---:|---:|---:|---:|---:|
| Base | 8/0/30/25 | 6/0/30/24 | 3/5/3/19 | 0/0/0/30 | 0/0/30/0 | 3/2/22/3 |
| Constrain | 25/25/0/0 | 26/26/0/0 | 5/4/21/0 | 5/4/21/0 | 0/0/0/30 | 0/0/0/30 |

Constrain failures changed from R26 indices **8, 13, 15, 17, 25** to R27
indices **6, 9, 11, 14**. The corrected comparison therefore shows task and safe
success each increased by one (25→26), with five R26 failures resolved and four
previous successes regressed. Contact and XY remained zero in both. The fixed
gate still fails task ≥27 by one; safe ≥26 and both safety ceilings pass.
Base failure-index sets and both policies' complete metric transitions are in
the machine-readable correction result.

R24 remains preserved as secondary comparison: it reports 26/26/1/1 for its
Constrain checkpoint/data and its old R27 transition counts remain in the
original execution record, but are confounded for sampler inference. The
original R27 preregistration still names R24 as primary; it has **not** been
rewritten or represented as retrospectively corrected.

The read-only correction was preregistered separately at
`outputs/exp/20261027_213400_placesphere_r27_primary_correction/analysis_registration.json`
(SHA `5594ff8af2a1831fbb78a8f3d60c1dcfd1699e57b34b1b2a2f43bda5d43ea8e9`). Its
result is `primary_comparison.json` (SHA
`b37577c8442f6ee1a1749c3fd2283df5670d6e9dc7710691fadd49f91cf78b26`). The
timestamped record hashes both 64-file episode trees and all source/input
registrations. A prior analyzer preflight stopped before computing transitions;
its log is preserved, and the successful analysis used the superseding
registration. The correction was analysis-only: no native evaluation, replay,
training, checkpoint/data mutation, or additional scene execution occurred.
The Round 27 execution record was appended with this correction, the R24
reclassification, provenance hashes, and corrected transition table; current
execution-record SHA is
`5a388a8231cb2f1b7bfbc99487abb6d8663182e23304f205c9f5ce0af5335b8f`. The
bounded analysis regression tests passed: **2 passed**.

### R27 correction provenance rerun — immutable registered execution-record input

The first correction registration is retained unchanged at
`outputs/exp/20261027_213400_placesphere_r27_primary_correction/analysis_registration.json`
(SHA `5594ff8af2a1831fbb78a8f3d60c1dcfd1699e57b34b1b2a2f43bda5d43ea8e9`), as is
its result (SHA `b37577c8442f6ee1a1749c3fd2283df5670d6e9dc7710691fadd49f91cf78b26`).
Its registered R27 execution-record input pinned SHA
`bc3483a62d0a1496b93d003cd73bb2dcdce1081c08c2d608b6aa90ff3187fd9d`, but the
same path was later appended with the correction and had SHA
`5a388a8231cb2f1b7bfbc99487abb6d8663182e23304f205c9f5ce0af5335b8f`. A direct
rerun therefore failed its registered-input hash gate at
`r27_original_execution_record`; the earlier registration/result remain honest
historical records, not silently repaired artifacts.

The superseding registration is
`outputs/exp/20261027_220000_placesphere_r27_primary_correction_rerun/analysis_registration.json`
(SHA `7d294ee9876420d5ef9d26bb0dc9575a05c5ab3ae43f63a9287652cb51da4249`). It
registers a read-only snapshot at
`outputs/exp/20261027_220000_placesphere_r27_primary_correction_rerun/inputs/round27_execution_record.current.json`
(SHA `5a388a8231cb2f1b7bfbc99487abb6d8663182e23304f205c9f5ce0af5335b8f`),
copied byte-for-byte from the R27 record before adding the outputs-only pointer
to this rerun. The current mutable R27 execution record SHA is now
`a10abed8c1e96cb66e898397c3e0dfcd25b3654e3ad50e43709c47c814ec3562`; it is not
an analysis input. All other registered file and 64-file episode-tree hashes,
including the R26/R27 scenes and checkpoint identities, pass unchanged.

The reproducible CLI, run twice with exit 0 and byte-identical stdout SHA
`43d56e96dbe930ad98b47397db3117df831777fb00e4b2b58ad50ea24f6cda6b`, was:

```sh
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && timeout --signal=TERM --kill-after=15s 180s python -m examples.baselines.constrain.placesphere_r27_primary_analysis --registration outputs/exp/20261027_220000_placesphere_r27_primary_correction_rerun/analysis_registration.json
```

The saved result is
`outputs/exp/20261027_220000_placesphere_r27_primary_correction_rerun/primary_comparison.json`
(SHA `43d56e96dbe930ad98b47397db3117df831777fb00e4b2b58ad50ea24f6cda6b`). It
reproduces R26 Base **8/0/30/25** vs R27 Base **6/0/30/24**, and R26 Constrain
**25/25/0/0** vs R27 Constrain **26/26/0/0** (task/safe/contact/XY), including
the transition counts above. R24 remains secondary and the original R27
preregistration's R24-primary deviation is explicitly preserved.

The analyzer source SHA is
`149035f8fc8715219956f1f15350de96e04498f53f7f1116ed1b5360600012d6`; the
regression-test source SHA is
`691d8ae35b7b42fa7cbc60f4ca7f9f5bef81f365179e9515d34e7e8a6fb87ea8`. Focused
tests passed **4/4**, including reproducing the retained stale-pin failure,
accepting the real frozen snapshot, and rejecting post-freeze mutation. No
native episodes, replay, training, data, checkpoint, or scene were created or
changed.

## Round 29 — fixed four-plan DDIM ensemble (inference-only)

Continued on `goal/transport-detour-optimization`, HEAD/base
`a1dcc014fe38976e384610df1f933719672256de`; no checkout, commit, or push.
Training remains frozen at **1,000 DIRECT / 500 OBSTACLE**. No new demos, layouts,
weights, or ManiSkill source edits. One registered inference change was used for
both policies: four independent Gaussian `(16,7)` initial plans, identical latest
two-frame condition, 100-step DDIM eta 0 denoising in one batch, arithmetic mean
of all seven channels after denoising over `[1:9]`, then the existing action map
and first-two-action execution/replan. No sample ranking, clamping changes, or
route selection. Defaults remain one plan and existing DDPM call paths.

Registration: `outputs/exp/20261027_230000_placesphere_ensemble4_r29/`
`inference_preregistration_r2.json` (SHA
`25b4a8d1e34d52b7dd410f7546a4fd807182321f4592c31b5192744ca531ae49`). The
initial registration was retained at `inference_preregistration.json`; its first
launch stopped before output creation or any scene execution because the
ensemble-size guard expected the field under `protocol`. The corrected r2
registration moved that same fixed value there; the failed preflight log remains
at `evaluation_command.log` (SHA
`14aa56cb8b6341ca4ae48d62459a534bbe6733ab530723f21c883c60f33498ef`). The one
native trial then ran with the fixed 18,000 s outer limit, 300 s/episode, 200
control steps, serial CUDA inference and 30-scene denominator per policy. Exact
command and complete input/source identities are in `round29_execution_record.json`.

The immutable primary is R27 DDIM100 eta0 single-plan at
`outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval/`, with identical
weights and frozen scenes. The evaluator preflight checked all 30 scene IDs,
seeds, bit-exact restore flags, checkpoint hashes, and equal initial 42-D policy
observations. After the run, all 30 initial observations per policy matched R27
again. All 60 episodes completed; 0 runtime errors, dropped episodes, or retries.
Elapsed wall time was approximately 1,260 s (22:02:59–22:23:59 local). Maximum
single synchronized ensemble inference time was 0.3745 s for Constrain and 0.4712
s for Base. The GPU returned idle after execution.

| Policy | Task | Safe | Contact | XY | Runtime errors |
|---|---:|---:|---:|---:|---:|
| Base | 3/30 | 0/30 | 24/30 | 23/30 | 0 |
| Constrain | 25/30 | 25/30 | 0/30 | 0/30 | 0 |

R27 Constrain was 26/26/0/0. R29 vs R27 transitions (task and safe are
improved/regressed/both/neither; contact and XY are cleared/introduced/both/neither)
were task **4/5/21/0**, safe **4/5/21/0**, contact **0/0/0/30**, XY **0/0/0/30**.
The five current Constrain failures are scenes 2, 12, 16, 19, 29: four
`incorrect_release` (2/12/16/29), one `not_stabilized` (19). The four incorrect
releases were off-bin; scene 19 released with 0.484 m/s linear speed and remained
unstabilized. There were no contacts or XY crossings. Base vs R27 was task
1/4/2/23, safe 0/0/0/30, contact 6/0/24/0, XY 5/4/19/2.

**Gate: fail/pause.** Task 25/30 is below 27 and safe 25/30 is below 26;
contact and XY ceilings pass at 0. No formal evaluation follows from this
repeated-development-set result. Keep the dataset frozen and diagnose the actual
release/stabilization regressions before selecting a next optimization slice.
This trial changes ensemble size, random draw consumption, and compute together;
it does not establish that stochastic dispersion caused the earlier XY release
errors or that action averaging preserves route topology.

The evaluator's unmodified raw `paired_summary.json` has legacy R24 comparison
labels and an old 90% gate even though the supplied primary directory was R27.
Those episode outcomes were not edited. The separately registered read-only
`primary_analysis.json` is authoritative for R27 transitions and the 27/26/3/3
gate. Analysis registration SHA
`55c1a454f4c5e8c6e0f72069f80d453d0f11fed1f598c7b4c41c2466b6c94e2d`; analysis SHA
`33fa28df35422b703c2880eb2980273cbec7dee3ef765109e70663023a7ee846`. R29 paired
summary SHA `459d169f4e22317a9056a8304302f212c98ddace90cd8da48c5abac85f204c09`.
The Base EMA, Constrain EMA, HDF5, manifest and training-identity hashes were
recomputed after the run and remain the registered R26 identities. The complete
run, failure list, paired transitions, hashes and cleanup record are in
`round29_execution_record.json`.

Verification after the changes:

```bash
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=20s 700s \
  python -m pytest examples/baselines/constrain/tests -q --tb=line
```

Result: **528 passed, 28 warnings, 82.59 s**. This includes a bitwise equality
check of the existing single-plan DDIM path against the pre-ensemble known-latent
reference loop. `git diff --check` passed. No
training, replay, or additional native episode was started after the registered
trial. Execution record SHA
`44ee060c7a29be94c9ab52007809a6c31be377634c49f071b831e3025dce4637`.
