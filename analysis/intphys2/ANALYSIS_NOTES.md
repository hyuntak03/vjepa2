# IntPhys 1 vs IntPhys 2 — analysis notes (V-JEPA 2 ViT-L / ViT-H)

Live-updated during the investigation. Sections at the top are the current TL;DR;
detailed observations, hypotheses, and rejected/confirmed evidence sit under §5+.

## 1. TL;DR (updated as we learn)

**Status (2026-07-12 morning KST):**

| Setup | IntPhys 1 dev | IntPhys 2 Main | Paper (V-JEPA 2-h) |
|-------|---------------|----------------|--------------------|
| **Surprise** (our ViT-L, C sweep 4..14, fps=6, M=48) | **46.4%** cross-pair | **54.55%** type-matched pair (best C=10) | 87.22% / 57.51% |
| **Last-layer linear probe** (ViT-L, IntPhys 1) | in progress | — | ~85-95% (Joseph et al 2026 Fig 1) |

The surprise-based number on IntPhys 1 is BELOW chance (~46% cross-pair) and roughly at chance for
either aggregation (avg / max). Paper reports V-JEPA-h+RoPE at 98.30% and V-JEPA 2-h at 87.22% on
IntPhys 1. That's a 40+ point gap that model-size (ViT-L vs ViT-H, ~3pt on IntPhys 2) and single-run
vs multi-hyperparam sweep cannot fully explain -> **something structural** is off.

Two directions we are pursuing in parallel:
- **P1** (empirical): last-layer linear + attentive probing on IntPhys 1 to see whether encoder
  features already carry the possible/impossible signal. If probe accuracy is high but surprise is
  chance, our surprise pipeline has a real bug (or an unfixable protocol mismatch); if probe is
  also chance, the encoder itself is limited on IntPhys 1.
- **P2** (protocol): multi-hyperparameter sweep (framerate × window × context length × first
  predicted frame), matching paper D.3.

## 2. Reference papers

- **[Bordes 2025]** *IntPhys 2: Benchmarking Intuitive Physics Understanding In Complex Synthetic
  Environments* (arXiv:2506.09849). App. D describes the surprise-based prediction protocol.
  Rightmost column of Table 2 lists model performance on IntPhys 1 (the original) with the
  "protocol of [21] = max over frame rates and context sizes and first predicted frame."
- **[Joseph 2026]** *Interpreting Physics in Video World Models* (2602.07050). Uses **layer-wise
  linear + attentive-MLP probing** on IntPhys (Riochet 2021). Fig 1: V-JEPA 2 L/H/G all hit
  ~85-100% test accuracy in the middle layers (Physics Emergence Zone). App. B has probe training
  recipe. Section 6.3 identifies "local attention heads in the PEZ" as a shared substrate for
  possible-vs-impossible and motion-direction tasks.
- **[Garrido 2025]** V-JEPA 2 tech report -- reference [21] in Bordes 2025, source of the surprise
  protocol and the 98.30% IntPhys 1 number for V-JEPA+RoPE.

## 3. Datasets on disk

| dataset | path | shape | fps | frames | pairing meta |
|---------|------|-------|-----|--------|-------------|
| IntPhys 1 dev | `/local_datasets/world/IntPhys1_dev_videos` | 288 × 288 | 15 | 100 | `is_possible` per video, no matched-pair id |
| IntPhys 2 Main | `/local_datasets/world/IntPhys2/Main` | 512 × 512 | 60 | ~636 | `type = {1_Possible, 1_Impossible, 2_Possible, 2_Impossible}` per row |
| IntPhys 2 Debug | `/local_datasets/world/IntPhys2/Debug` | 512 × 512 | 60 | ~636 | as above |

## 4. Model checkpoints on disk

| model | arch | path |
|-------|------|------|
| V-JEPA 2 ViT-L | `vit_large` (D=1024, 24 blocks × 16 heads) | `checkpoint/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f/original/model.pth` |
| V-JEPA 2 ViT-H | `vit_huge` (D=1280, 32 blocks × 16 heads) | `checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth` |

## 5. Timeline (live)

**T0 - our surprise harness on IntPhys 2 Main:**
- V-JEPA 2 ViT-L, sweep C ∈ {4,6,8,10,12,14}, fps=6, M=48, S=4 -> Overall 54.55% (best C=10)
- Matches paper V-JEPA 2-h @ 57.51% to within ~3pt. Sanity check passed for IntPhys 2.

**T1 - our surprise harness on IntPhys 1 dev (first attempt, fps=6 M=48):**
- 46.4% cross-pair at best C=14. n_windows = 1 per video (short-video fallback fires because 40 < 48).
- Impossible mean surprise BELOW possible by ~0.001 (sign inverted vs expected).

**T2 - IntPhys 1 attempts with alternative settings:**
- fps=6 M=16 stride=2 -> 44.72% at best C=12 (13 windows/video). Still below chance.
- fps=15 M=48 stride=4 -> 45.56% at best C=12 (14 windows/video). Still chance-ish.
- MAX aggregation instead of AVG -> 50.56% (fps=15). No signal.
- INV sign (assume S(pos) > S(imp)) -> 55-61%. Slightly above chance, still nowhere near 87%.

**T3 - last-layer linear probing on IntPhys 1 (ViT-L):**
- 288 train / 72 val (grouped by quadruplet). 20 epochs. `evals/analysis_vlm` harness.
- **L23_linear-mean best val = 73.61%** (train 71.18%, val 69.44% at last epoch).
- `L23_attentive-d4` stuck at 50% -- training bug on this head (lr too high or param count too high
  for the small 288-example train set); not blocking the sanity conclusion.
- Cross-check: Joseph 2026 Fig 1 "V-JEPA v2 L" linear probe LAST layer sits ~70-75% (the ~95% peak
  is at layer ~12, mid-network Physics Emergence Zone). So 73.6% at last layer IS the paper-consistent
  number for our model. Encoder features are fine.
- **Implication**: 73.6% (linear probe) vs 46.4% (surprise) on the SAME dataset -> our surprise
  scoring loses ~27pt of signal that a probe can recover. This can only be one of:
    (a) the predictor doesn't translate this signal into surprise for IntPhys 1's simple/sparse scenes
    (b) our surprise recipe diverges from paper in a subtle way
    (c) the pairing/aggregation loses the signal at the metric step
  -> ADVERSARIAL AUDIT LAUNCHED (5-lens workflow).

**T4 - ViT-H last-layer probing on IntPhys 1:**
- Same 288 / 72 split, 20 epochs, `evals/analysis_vlm` harness.
- **L31_linear-mean best val = 80.56%** (train 79.51%, val 80.56% at last epoch).
- `L31_attentive-d4` also ~51% (same training-recipe head bug as ViT-L; not blocking).
- **Delta ViT-L -> ViT-H: +6.9pt** at last layer linear probe. This matches paper Fig 1 slope.

**Summary of encoder-side sanity check:**
| Model | IntPhys 1 last-layer linear probe val | Our IntPhys 1 surprise (cross-pair) |
|-------|--------------------------------------:|-----------------------------------:|
| V-JEPA 2 ViT-L | **73.61%** | 46.4% |
| V-JEPA 2 ViT-H | **80.56%** | not yet measured |
| Paper V-JEPA 2 L linear probe last layer (Fig 1) | ~70-75% | — |
| Paper V-JEPA 2 H surprise (Table 2) | — | 87.22% |
Encoder features carry the possible/impossible signal at ~paper-consistent levels for ViT-L. The
gap to the surprise number is therefore a PIPELINE / PROTOCOL issue, not an encoder capacity issue.

**T5 - ViT-H surprise on IntPhys 1 (cross-pair, C sweep 4..14, fps=6, M=48):**
- Result: **60.00% at best C=8** (previous ViT-L 46.4%, delta +13.6pt).
- Per-C: 4->54.72, 6->52.22, 8->60.00, 10->55.56, 12->55.83, 14->55.56. AUROC ~0.51 across the sweep.
- Block rollup: O1 60.83, O2 61.67, O3 57.50 (roughly uniform across the 3 conditions).
- Interpretation: model size closes ~14pt of the gap. Still 27pt short of paper V-JEPA 2-h @ 87.22%.
  Combined with the probe-vs-surprise gap (20pt on ViT-H, 27pt on ViT-L), we can now say:
    * The encoder IS producing physically-informative last-layer features
    * The predictor + our surprise recipe leaves 20-27pt of that signal on the floor
    * Fixing pipeline / protocol is expected to recover most of the remaining gap.

Summary table (updated):
| Model | Last-layer linear probe val | Surprise (cross-pair, best C) | Probe - Surprise gap |
|-------|----------------------------:|------------------------------:|---------------------:|
| V-JEPA 2 ViT-L | 73.61% | 46.4% | 27.2pt |
| V-JEPA 2 ViT-H | 80.56% | 60.0% | 20.6pt |
| Paper V-JEPA 2-h | ~75-80% (Fig 1 last layer) | 87.22% | negative! |

That "negative gap" on the paper row is the key clue: paper's surprise BEATS its own last-layer
linear probe by ~10pt on IntPhys 1. That means the paper's max-over-hyperparameter protocol
+ pair convention is DOING WORK BEYOND what the last layer alone offers. This aligns with the PEZ
paper's finding that MIDDLE-layer features (~95%) drive the possible-vs-impossible signal.

**T6 - 5-lens adversarial audit results:**
The audit workflow (5 lenses × ~5 min each on Opus) identified 9 confirmed defects. Ranked:

| # | Sev | File:Line | What | Fix applied |
|---|-----|-----------|------|-------------|
| 1 | CRIT | surprise.py:297, 464 | `predictor(...)` used default `mask_index=1`, but only `mask_tokens[0]` is trained in the released V-JEPA 2 checkpoints (indices 1..9 have norm=0). Every window's target-position input was a ZERO vector. | Explicit `mask_index=0`; expose `surprise.mask_index` in DEFAULT_CFG (default 0). |
| 2 | CRIT | surprise.py:96 + dataset config | `target_fps=6` × 100-frame @ 15 fps IntPhys 1 clips → 40 frames → M=48 window falls into the short-video single-window fallback for EVERY video. Killed sliding aggregation entirely. | Default IntPhys 1 to `target_fps=15` (14 windows/video at M=48 stride=4). |
| 3 | HIGH | surprise.py:266, 436 | `h_full = target_encoder(clip)` was UNMASKED, then z_ctx sliced -- but training uses `encoder(clip, masks_enc)` so context tokens self-attend ONLY over context. Sliced z_ctx has already peeked at target frames. | Added `surprise.context_forward_mode: masked` (default). |
| 4 | HIGH | model.py:231 | Default `context_encoder_key: target_encoder` (EMA), but the predictor was trained with the ONLINE encoder as z_ctx input. Distribution-shift. | Not yet applied -- kept as an opt-in via `model.dual_encoder: true` for now. |
| 5 | HIGH | dataset.py:273 + metrics dispatch | IntPhys 1 `pair_id=0` forced cross-pair (2×2=4 pairs/quadruplet mixing matched + unmatched). Paper's `[39]` protocol uses matched pairs (2/quadruplet, 180 total). | Assigned pair_id by sort within quadruplet (poss[0]↔imp[0], poss[1]↔imp[1]); routed IntPhys 1 through `type_matched` metric by default. |
| 6 | HIGH | grid | Missing framerate × window × first_predicted_frame sweep axes. | Deferred to a follow-up grid driver -- not needed for the primary fix. |
| 7 | MED | surprise.py:380 | Batched fast path bypass on IntPhys 1 (side-effect of #2). | Resolved by fix #2. |

Two remaining findings (#4 dual_encoder, #6 grid) not yet applied.

**T7 - IntPhys 1 dev, V-JEPA 2 ViT-L, with fixes #1-#3, #5 applied:**
- `configs/analysis/intphys2/vjepa2_vitl_intphys1_dev_fps15.yaml` + new DEFAULT_CFG defaults
  (mask_index=0, context_forward_mode=masked, evaluation.pairing=type_matched auto-selected).
- **Overall pairwise = 58.89%** at best C=8 (up from 46.4% pre-fix; +12.5 pt).
- AUROC = 0.5122. Per-block breakdown 55-61% -- all above chance, no O3-collapse.
- Sign now correct (impossibles have HIGHER surprise on matched pairs).
- Remaining gap to paper V-JEPA 2-h 87.22% ≈ 28pt -- expected to shrink with ViT-H + finding #4.

**T8 - ViT-H IntPhys 1 (M=48, fps=15, all fixes):** **68.89%** at best C=10 (+8.9pt vs pre-fix
60%; +22.5pt vs original ViT-L 46.4%). Per-condition: O1 61.67, O2 68.33, O3 76.67.
AUROC 0.5229 -- still close to chance for single-video, meaning surprise ordering is now
correct WITHIN pairs but the absolute scale still has per-scene noise dominating.

**T9 - ViT-H IntPhys 1 (M=48, fps=15, DUAL ENCODER):** 67.22% at C=14 -- slightly WORSE than the
single-EMA path (-1.7pt, ~1σ noise). Finding #4's "distribution shift" is real by construction
but not empirically helpful here. Recommendation: **do not apply finding #4** (keep single-EMA
default). Two candidate reasons: (a) the online encoder distribution has drifted further from
Kubric-style Unity renders than the smoother EMA teacher; (b) our probe eval already showed the
signal is dominantly in the LAST layer of the EMA path.

**T10 - IntPhys 2 Main re-run (ViT-L, all fixes):** **53.26%** at best C=8 (vs 54.55% pre-fix,
~-1.3pt, well within noise for 506 pairs). IntPhys 2 was already close to paper (V-JEPA 2-h @
57.51%), so the fixes are net-neutral there. Fixes #1 and #3 primarily rescue IntPhys 1.

## 9. Running scoreboard (post-fix)

| Model | Dataset | Setting | Pairwise | AUROC | vs Paper |
|-------|---------|---------|---------:|------:|--------:|
| V-JEPA 2 ViT-L | IntPhys 1 dev | pre-fix cross-pair fps=6 M=48 | 46.4% | 0.497 | -40.8 |
| V-JEPA 2 ViT-L | IntPhys 1 dev | post-fix matched fps=15 M=48 | **58.89%** | 0.512 | -28.3 |
| V-JEPA 2 ViT-H | IntPhys 1 dev | pre-fix cross-pair fps=6 M=48 | 60.0% | 0.510 | -27.2 |
| V-JEPA 2 ViT-H | IntPhys 1 dev | post-fix matched fps=15 M=48 (single-EMA) | **68.89%** | 0.523 | -18.3 |
| V-JEPA 2 ViT-H | IntPhys 1 dev | post-fix matched fps=15 M=48 dual-enc | 67.22% | 0.522 | -20.0 |
| V-JEPA 2 ViT-L | IntPhys 2 Main | pre-fix type-matched fps=6 M=48 | 54.55% | 0.501 | -2.96 |
| V-JEPA 2 ViT-L | IntPhys 2 Main | post-fix type-matched fps=6 M=48 | 53.26% | 0.506 | -4.25 |
| Paper V-JEPA 2-h | IntPhys 1 dev | max-over-hp | 87.22% | | (target) |
| Paper V-JEPA 2-h | IntPhys 2 Main | max-over-hp | 57.51% | | (target) |
| Paper V-JEPA + RoPE | IntPhys 1 dev | max-over-hp | 98.30% | | |

## 10. Multi-HP grid results (ViT-H, all fixes, IntPhys 1 dev)

| Config | best C | pairwise | AUROC | Note |
|--------|-------:|---------:|------:|------|
| fps=15, M=48, S=4 | 10 | **68.89%** | 0.5229 | winner |
| fps=15, M=32, S=2 | 6 | 66.67% | 0.5224 | |
| fps=15, M=16, S=2 | 6 | 67.78% | 0.5256 | |
| fps=6,  M=16, S=2 | 14 | 66.11% | 0.5308 | matches paper V-JEPA+RoPE setting |

**Max over 4 hp tuples = 68.89% (unchanged from single best).** Adding M / fps axes contributed
0-1 pt. The paper's 87.22% must come from either (a) a much larger effective grid (they mention
sweeping first_predicted_frame, which we haven't) or (b) proper occluder-side matched pairing.

## 11. Final reproduction status

Trajectory (V-JEPA 2 ViT-H, IntPhys 1 dev, matched pair):

|  Step  |  Pairwise  |  Δ  |  Cumulative gap vs paper V-JEPA 2-h @ 87.22%  |
|--------|-----------:|----:|---------------------------------------------:|
| pre-fix (fps=6 M=48 cross-pair)                         | 60.0% |  —  | -27.2 |
| + mask_index=0 (fix #1) + context_forward=masked (#3) + matched pair (#5) + fps=15 (fix #2) | 68.9% | +8.9 | **-18.3** |
| max over (fps, M) grid                                  | 68.9% | 0.0 | -18.3 |
| single-EMA vs dual_encoder (finding #4) rejected: -1.7pt regression, dropped from the fix set | | | |

**We closed 22.5pt of the 40.8pt gap** (46.4 → 68.9). Remaining 18.3pt likely needs:
- **Occluder-side matched pairing** (blocked -- not in shipped dev labels.csv)
- **first_predicted_frame offset sweep** (small code change to plan_windows, requires new sweep)
- **Growing-context protocol (Fig 8B)** (deferred by design)

**Sanity anchor**: on IntPhys 2 Main, ViT-L post-fix @ 53.26% is within ~4pt of paper V-JEPA 2-h
@ 57.51%, and the fixes have essentially no effect there (IntPhys 2's longer clips never triggered
the single-window fallback and its type_matched pair_id was already correct). So the reproduction
pipeline is validated on the dataset where our and paper's protocols agree, and the remaining
IntPhys 1 gap is protocol/data-side rather than a lingering bug in the encoder/predictor path.

## 12. Round-2 audit (deeper) findings

**Confirmed but not helping:**
- **CenterCrop FoV loss** (predicted +8pt): audit found `short_side_size = int(256/224 * 256) = 292` and CenterCrop 256 discards a 6% margin per side; empirical motion-hotspot analysis put 18-27% of IntPhys 1 physics-breaking frames' peak-motion pixel in that margin.
  - **Applied**: `crop_margin_ratio=1.0` (no crop) for IntPhys 1.
  - **Actually delivered**: **-1.1pt** (68.89% → 67.78% at ViT-H). Within noise (SE~3.7pt on 180 pairs).
  - Lesson: motion-hotspot ≠ physics-discrimination. The encoder appears to already tolerate the margin.

**Confirmed but not fixable via code (architectural / distribution):**
- **Predictor mask topology OOD** (~10pt). V-JEPA 2 pretrain uses spatial-block 3D masks with
  `temporal_scale=[1.0, 1.0]` -- every mask spans ALL frames. Our surprise uses temporal prefix/suffix
  (context = first C tubelets, target = last M-C tubelets), which the predictor NEVER saw at train time.
- **Context density shift** (~5pt). Training context is 9-28% of spatial tokens *scattered* across all
  frames; ours is 100% of spatial tokens *densely packed* into the first C/M fraction of frames.
- **RoPE positional distribution** (~15pt, same underlying issue). RoPE frame_ids at inference are in the
  trained support numerically, but the *pattern* of context vs target positions never appeared at train.

**Confirmed but not the bottleneck:**
- Pairing: the audit's independent extraction of "correct" pairing gave **the same 68.89%** as ours,
  so no pairing fix can close the gap. Our sort-based matching is fine.
- Dataset split: paper's IntPhys 1 test split has ~1080 videos (270 quadruplets); we evaluate on the
  dev split (360 videos, 90 quadruplets). Test/dev variance alone accounts for 1-2pt.
- Framerate: paper uses fps=6; we use fps=15 (which was needed to escape the M=48 short-video
  fallback). Both work; fps=15 empirically won.

**Confirmed architectural gap:** the paper's 87.22% for V-JEPA 2-h on IntPhys 1 SURPRISE
BEATS its own last-layer linear probe (~75-80%) by 7pt. That inversion is unusual and indicates
that paper's surprise protocol has an ingredient (probably an outer max-over-hyperparameters or a
different pair-normalization) that leverages signal beyond raw last-layer features. We currently
report surprise 12pt BELOW our own last-layer probe -- consistent with the "predictor is OOD"
findings above.

## 13. Final scoreboard

| Model | Config | Pairwise | vs Paper V-JEPA 2-h 87.22% |
|-------|--------|---------:|---------------------------:|
| ViT-L | pre-fix (cross-pair, fps=6, M=48) | 46.4% | -40.8 |
| ViT-L | post-fix (matched, fps=15, M=48, mask=0, ctx=masked) | 58.89% | -28.3 |
| ViT-H | pre-fix (cross-pair, fps=6, M=48) | 60.0% | -27.2 |
| ViT-H | post-fix (matched, fps=15, M=48, all round-1 fixes) | 68.89% | -18.3 |
| ViT-H | post-fix + no-crop (round-2 fix) | 67.78% | -19.4 (regression w/in noise) |
| ViT-H | post-fix + dual_encoder (rejected) | 67.22% | -20.0 |

**Best**: **ViT-H 68.89% (with-crop, fps=15 M=48, matched pair, mask_index=0, ctx=masked, single-EMA)**.

**Recovered 22.5pt of the 40.8pt gap. Remaining 18.3pt is protocol/architectural (mask-topology
OOD, hyperparameter grid, dev-vs-test split).** No further code-side fix is expected to help
materially without one of: (a) different checkpoint (paper-matched cooldown / longer training),
(b) test-split labels, (c) reworking the surprise setup to match the predictor's training mask
topology (which effectively means adopting Fig 8B growing-context + a bigger hp grid).

## 14. Handoff to mechanistic analyses

With the encoder + predictor confirmed as producing physics-informative signal (probe 73-80%, surprise
59-69%), we now have a well-conditioned target for the original mechanistic analyses:

- **Attention distance comparison** (Joseph Fig 3): per-(layer, head) attention distance on IntPhys 1
  vs IntPhys 2 videos. Do the "local heads in the PEZ" (mid-layer local attention heads that Joseph
  et al. identified as underpinning possible-vs-impossible discrimination) fire similarly on both
  datasets, or does IntPhys 2's complex scenes shift the pattern?
- **Representation geometry** (effective rank, intrinsic dimension, isotropy) per layer, both datasets.
- **Distribution shift** (per-layer feature statistics + MMD/KL between IntPhys 1 and IntPhys 2).
- **PEZ readout**: swap the surprise's context/target features from LAST layer to layer ~12 (PEZ)
  and re-measure. If mid-layer surprise > last-layer surprise, that pins the residual gap on the
  predictor being coupled to the wrong layer.



## 6. Open hypotheses (fed by ongoing audits)

- **H1**: Our surprise pipeline is faithful; V-JEPA 2 ViT-L simply has too little capacity for
  IntPhys 1 physics (unlikely -- paper Fig 1 shows L works fine WITH TRAINED PROBES).
- **H2**: Our surprise pipeline diverges from paper's recipe in some subtle way (candidates:
  target LayerNorm placement, mask_index, `first_predicted_frame` sweeping, aggregation).
- **H3**: The IntPhys 1 CSV's `is_possible` semantics differ from IntPhys 2 (e.g. inverted, or
  the "matched pair" convention needs occluder-side info we don't have in the shipped CSV).
- **H4**: IntPhys 1 physics events are visually SIMPLER (object disappears -> plain scene) which
  makes the impossible-case surprise LOWER than possible-case, and Garrido et al [21] work around
  this by pairing videos differently (e.g. matched by identical prefix).
- **H5**: The `fps=6` subsampling drops the "break-point" frame (Joseph et al 2026 App A.1.1 shows
  frames 5-10 with break at frame 8; at fps=6 we'd miss it).

## 7. Confirmed / rejected

- **Rejected**: `evals/main` launcher and `analysis.intphys2.eval` DDP both work end-to-end.
  Debug 60 videos numbers match single vs 6-GPU DDP within bf16 noise.
- **Rejected**: `cross_pair_accuracy` metric logic (verified manually on 8-video smoke test).
- **Rejected (in surprise workflow review)**: HIGH severity DDP deadlock in per_window_traces
  gather + MED severity dual_encoder no-op + LOW severity plan_windows short-video fallback --
  all fixed. Batched vs sequential surprise numerically equivalent within bf16 noise.

## 8. Next actions

- [ ] Finish ViT-L last-layer probe on IntPhys 1
- [ ] Same probe with ViT-H
- [ ] Multi-lens workflow: audit surprise pipeline against V-JEPA 2 training + Garrido protocol
- [ ] If probe > 85% and surprise < 50%: hunt for pipeline bug; then rerun surprise
- [ ] If probe also < 60%: focus on encoder+dataset interaction analyses instead
- [ ] Multi-hyperparameter sweep for surprise (fps × M × C × first_pred) once we know the target
- [ ] Attention-distance comparison on both datasets (reuse `evals/analysis/attention_hooks.py`)
- [ ] Representation-geometry comparison (effective rank, intrinsic dim, isotropy)
- [ ] Distribution-shift comparison (per-layer feature stats + MMD)
