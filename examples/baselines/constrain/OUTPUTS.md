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
