# 4. Experiments (draft)

**Question**: which results does the paper report, against which references, and which ones do we still lack?
**Answer (v3.2, after the full-budget test)**: sections 4.2, 4.3 and 4.5 are written from measurements, with three same-space references for every cell. Placement is judged with an object-region score that a true-position oracle passes; cells where the oracle fails carry no failure claim, and Cosmos is not used as evidence. Section 4.5 reports a small pilot and a pre-registered full-budget test of the input-occluder task, which is negative at far blocks. Sections 4.4, 4.6 and 4.7 describe experiments without results yet (expected).
**Next**: an evaluation set that a tracker cannot solve (the ledge) becomes the main axis for any method.

Version: v3.2 (full-budget test, 2026-10-23). Terms follow the team glossary (STORY.md section 0, English column).
Markers: `[ev N]` evidence claim (evidence table v3), `unverified` other-team result, `(expected)` no result yet, `[source table]` fact taken from documents.

---

## 4.1 Protocol

**Models.**
We compare the released V-JEPA 2 predictor, the observation-only block-AR baseline, two prior observation-only block-AR predictors (output feedback and hidden feedback), a video-context predictor in two versions (block targets and clip targets), and predictors fine-tuned on our testbed.
All use the frozen V-JEPA 2 ViT-H encoders.
We also report V-JEPA 2.1 g, DINO-Foresight and VideoMAEv2-g.

**Data.**
VOBench is our controlled synthetic testbed [source table].
Each group shares one context and has possible and impossible futures, which gives matched pairs with pixel-identical contexts.
It varies the object set (primitive or realistic, reported separately), the motion, the violation and the occlusion timing; under late occlusion the object is hidden at the last observation.
The gravity set adds contexts with several candidate futures: a ledge (2 candidates) and an arc (the true parabola and 8 alternatives); we use its 108-group test split.
External benchmarks are IntPhys 1 (dev, 180 pairs), IntPhys 2 (Main, 506 pairs), GRASP and InfLevel, each on its official evaluation grid.
For natural video we use a held-out part of K710.
Our SSv2 split is a held-out 1 percent of SSv2 that is not the official validation set; it is clean only for the prior predictors, because the baseline and the released predictor trained on nearly all of SSv2.

**Scores.**
Surprise is the token-averaged L1 distance between predicted and target tokens over the future blocks.
Pairwise accuracy is the share of matched pairs where the impossible future has higher surprise (chance 50 percent).
Future-choice accuracy is the share of groups where the lowest-surprise candidate is the true future (chance one over the number of candidates).
Carry gain is the projection of the predicted displacement on the true displacement, divided by the squared true displacement, with a parameter-free template readout; we report the same readout on true future tokens as the ceiling.

**Three references for every cell.**
Every score comes with a copy baseline, a context-average copy (both prediction-free, in the same scoring space) and a true-position oracle, a training-free latent prediction that moves the last fully visible object tokens (3 by 3 cells) to their true future cells, with the last observation or background-average tokens elsewhere.
We make a failure claim only on cells where the oracle clearly beats the baselines; elsewhere we say that the score cannot judge the cell.

**Which score for which claim.**
The all-token score does not pass the oracle check for placement: the oracle equals copy under late occlusion (53.3) and visible motion (89.2 vs 90.0) [ev 20, 56].
An object-region score, computed only on the 3 by 3 cells of the object in the two futures, passes it (95.3 under late occlusion, 99.2 for visible motion) [ev 56, 57].
We therefore judge latent placement (are the predicted tokens at the object cells closer to the possible future's target tokens?) and carrying with the object-region score, and report the all-token score next to it; presence and external benchmarks use the all-token score. All scores are computed on latent tokens; nothing is decoded to pixels.
Carry gain becomes a primary measure of distance only where its own ceiling is at least 0.5 (E7).

**Disclosures.**
The released predictor and the block-AR predictors use different scoring spaces and different context encoders [ev 50]; we compare each model only against its own same-space baselines.
One block is 0.5 s at the training frame rate and 0.2 s on VOBench.
On 108-pair cells the sampling half-width is about 8 points and repeated runs differ by about 1 point [ev 51]; we do not read differences below 2 points there.

## 4.2 How much of each benchmark needs prediction

**Table 1. Prediction-free baselines.** Pairwise accuracy in the released predictor's scoring space.

| Benchmark (cell) | Copy | Released predictor | Released minus copy |
|---|---|---|---|
| IntPhys 1, all six cells | 62.2 to 85.0 [ev 2] | | |
| IntPhys 1, best cell (stride 2, window 32) | 85.0 | 88.9 | +3.9 [−0.6, +8.9] [ev 1] |
| IntPhys 1, official cell (stride 2, window 16) | 73.9 | 83.3 | +9.4 [+3.3, +15.6] [ev 2] |
| IntPhys 2, Main | 55.5 | 54.3 | −1.2 [−5.1, +3.0] [ev 4] |
| VOBench, late occlusion (primitive / realistic) | 56.5 / 61.1 | 57.9 / 65.7 | +1.4 / +4.6, intervals include 0 [ev 5] |
| GRASP, InfLevel | missing | 70.4 / 72.8 (continuity) | (gap G6) [ev 9] |

Much of the IntPhys 1 score is reached without prediction, although the predictor adds 10 points at the official cell.
On IntPhys 2 and under late occlusion the released predictor stays near copy.
Garrido et al. [2025] report high GRASP scores on support and gravity; a GRASP copy baseline is the cheapest way to see whether those cells need prediction (gap G6).
The block-AR baseline clears its own block-space copy only at the official cell (90.0 vs 83.9, +6.1 [+2.2, +10.6]; window 32: +3.9 [−1.1, +8.3]) [ev 3].

## 4.3 Diagnosis

**One harness, three references.**
We measure the block-AR baseline and a prior observation-only block-AR predictor with one evaluation harness, the object-region score, and three same-space references [ev 67 to 69, table 1-3a].

| Cell | Score | Baseline | Prior predictor | Copy | Context-average copy | Oracle or ceiling | Chance |
|---|---|---|---|---|---|---|---|
| Visible moving objects, carry distance | carry gain, steps with true displacement above one cell (median) | 0.22 (interquartile 0.08 to 0.42) | 0.65 | −0.02 | −0.50 | ceiling 0.90 | n/a |
| Visible moving objects | object-region pairwise accuracy | 79.2 [72, 86] | 92.5 | 68.3 | 56.7 | 99.2 | 50 |
| Late occlusion, moving objects, placement | object-region pairwise accuracy | 51.4 | 51.4 | 51.1 | 50.8 | 95.3 | 50 |
| Late occlusion, static objects, placement | object-region pairwise accuracy | 88.9 | 93.1 | 63.9 | 100 | 90.3 | 50 |
| Late occlusion, moving objects, presence | all-token pairwise accuracy | 94.2 | 99.2 | 53.3 | 94.2 | n/a | 50 |
| Ledge (144 groups) | object-region future choice | 30.6 [23, 38] | 7.6 | 31.2 | 41.0 | 100 | 50 |
| Arc (72 groups) | object-region future choice | 11.1 [4, 18] | 25.0 | 0.0 | 16.7 | 100 | 11.1 |
| Natural video (K400 validation, 200 clips) | block-target L1, lower is better | 0.561 | 0.563 | 0.720 | 0.571 | n/a | n/a |

The two models differ in training block length (0.5 s for the baseline, 0.17 s for the prior predictor; VOBench blocks are 0.2 s), so we compare each against its own references rather than against each other.

**Neither model carries the tokens of hidden moving objects to their future cells.**
Both score 51.4 under late occlusion, the same as copy and the context average, while the oracle reaches 95.3; this holds for vanishing, shape and color violations alike [ev 67].
For hidden static objects the object-cell tokens are right (88.9 and 93.1), but a training-free context average already reaches 100 there, so this is not an ability.

**Visible carrying depends on the model.**
Both beat copy on visible moving objects. The baseline's predicted object tokens move by only 0.22 of the true displacement, the prior predictor's by 0.65, against a ceiling of 0.90 [ev 67, table 1-3a].

**Presence confirms prior work.**
All-token presence scores (94.2 and 99.2) are at or near a training-free context average (94.2) [ev 67], so they are not an ability.
The released predictor is rarely surprised when a hidden object vanishes (55.6 / 61.1, against 61.1 / 61.1 for copy in its own space) [ev 16], which confirms Xie and Alanwar [2026].

**Falls are not chosen beyond prediction-free baselines.**
With the object-region score, the oracle picks the true future in every group of the ledge and the arc, so the score can see success [ev 68].
On the ledge, both models pick the fall less often than chance and than the context average (30.6 and 7.6, against 50 and 41.0).
On the arc, neither is separable from the context average (11.1 and 25.0, against 16.7; intervals overlap).
With all-token scores the ledge cannot be judged (oracle 11.8) [ev 23].

**Natural video.**
On held-out K400 clips both models' losses (0.561 and 0.563) are within 2 percent of a training-free context average (0.571), against 0.720 for copy [ev 69, table 1-3a].

**Part of the material is in the input.**
For an object hidden at the last observation, its last seen position and velocity are readable from the context tokens within 4 to 11 pixels and 0.5 to 0.9 pixels per cell (true speed 15 to 18) [ev 27].
Acceleration and the current position of a hidden object are not measured (gap G13).

**Where the future comes from.**
When we change only the last observation in pixels, the predicted future follows it (follow index 0.63 to 0.88 for a freeze); changing only the block before it moves the future by at most 0.06 cells for the prior predictors [ev 30].
Cutting the context from 8 to 2 blocks loses only 1 to 2 percent of the gain over copy, for a prior predictor on our SSv2 split and for the baseline on held-out K710 [ev 31, 58].
History is useful nonetheless: for a training-free context average, the history value is 0.30 to 0.49 [ev 58].
On static cells the baseline's prediction is close to the last observation shrunk toward an average, with a factor of 0.54 (data-optimal 0.49 to 0.52) [ev 58, unverified].
The last-observation swap on the baseline is E3.

**What the loss rewards.**
When two futures differ only in one object, the object's 3 by 3 cells hold 3.1 to 5.4 percent of the target-token difference, about their 3.5 percent share of the image [ev 35]; another team reproduces this on a separate sample [ev 54, unverified].
Block tokens move by 32 to 39 percent of the distance between different videos under invisible ±1/255 perturbations [ev 36, unverified].
On synthetic 3D clips a same-cell context average beats copy by 12.6 percent while a carry oracle adds only 0.7 percent [ev 37, unverified].
On VOBench, the reward for the context average grows with the step, from −2.3 percent at the first step to 20.8 percent at the eighth; with noise-averaged inputs and targets it goes from −8.8 to 21.2 percent, so noise-averaging removes the short-step part but not the growing part [ev 64].
The growing part comes from smearing moving objects; carrying the object to its true position adds only 1.3 percent overall (35 percent on the object's own cells) [ev 64].
Target-side jitter that is independent and symmetric does not move the optimal prediction (theory) [ev 63].
Measured separately, input-side jitter is what rewards averaging and shrinkage: cleaning the inputs alone cuts the averaging reward from 16.8 to 6.8 percent and the shrinkage reward from 4.6 to 0.3 percent; cleaning the targets alone raises the averaging reward to 24.4 percent [ev 66].

**Retraining helps in domain only.**
Fine-tuning on VOBench and the gravity set solves the arc in domain (32 of 36 for the released predictor); at matched IntPhys 1 cells the same model drops by 6.7 points [−12.2, −1.7] at window 32 and stays level at window 16 [ev 24, table 1-8a].
Fine-tuning on 2D synthetic physics leaves the out-of-domain arc at 1 of 36, below copy [ev 25].
Variants of the baseline (mixing in the last observation, occlusion clips, a temporal-difference loss, object-label supervision) do not raise carrying [ev 42, 43].
The video-context predictor shows a trade-off: about twice the carry gain of the baseline (0.59 vs 0.29) [ev 12, unverified], but the clip-target version drops to 48.6 / 58.3 on late-occlusion presence and to 47.8 on IntPhys 1 [ev 21].
Other latent models score low in the same direction on all-token scores [ev 7, 48]; these scores cannot see placement, so we do not use them as evidence of generality.
Cosmos-Predict1-4B, a generative model that predicts reconstruction latents, reaches 91.7 on visible cells [ev 71]. Its scores on late-occlusion moving objects and at the ledge are all-token scores; in our space the same cells fail the oracle check, and Cosmos has no oracle in its own space, so under our rules these scores are not evidence either way.

## 4.4 Training-free tests

- **Noise-averaged targets (E1).** Noise-averaging inputs and targets cuts the reward for context averaging by only 19 percent on natural video and 15 percent on VOBench, below the 25 percent line set before the run; a target-only fix is therefore rejected [ev 64; the natural-video value is from plan v2]. A 64-dimensional jitter-robust projection cuts it by 32 percent on natural video (16.8 to 11.5 percent), removes the reward for shrinking toward the origin (12.6 to −0.7 percent) and doubles the carry reward (3.5 to 6.8 percent). At the same size of 256 it differs only a little from a principal-component projection (shrinkage 7.5 vs 9.6 percent, carry 5.0 vs 4.1 percent), so the effect of reducing dimension and the effect of removing jitter directions are not yet separated [plan section 12; corrected in round 2: an earlier draft compared the 64-dimensional projection with a 256-dimensional principal-component projection and called them the same size].
- **Context-average copy (E2).** It solves all-token presence on hidden moving objects (95.0, oracle 95.8) but not object-region placement (34.2 against copy 33.6, oracle 80.3) [ev 16, 57].
- **Last-observation swap on the baseline (E3, E3b).** Changing only the block before the last observation moves the predicted position by 0.05 to 0.06 cells at step 1, which matches a last-observation-only prediction (0.00) and has the opposite sign to a weighted-average prediction (−0.09); freezing the last observation moves it by 1.30 cells, more than a static prediction (at most 0.86), so the model also continues the last observation's motion [ev 65].
- **Input-by-target 2 x 2 (E1b).** Input-side jitter rewards averaging and shrinkage; target-side jitter does not [ev 66].
- **Carry-gain readout check (E7).** Ceiling per test set (at least 0.5), steps above one cell, presence threshold.

## 4.5 Training results: a small-budget pilot

Section 3.5 gives the full table [ev 72 to 76].
Each pilot arm saw 24,000 clips, 0.23 percent of the clips seen by the baseline, with one tenth of its peak learning rate and one seed [ev 76].
At this budget the binary primary measure stays between 50.6 and 52.6 for every arm, against 51.1 for copy and 95.3 for the oracle.
The continuous placement margin moves slightly toward the true position for the three occluder arms, by 1.0 to 1.5 percent of the oracle margin with intervals above zero, and the gap grows from 1,000 to 2,000 steps [ev 76]. The margin was added after the results; the pre-registered rule is binary.
The occluder arms raise hidden static objects (+9.7, +11.1, +9.7), the outcome that Xie and Alanwar [2026] report as freezing [ev 73, 76].
The two arms with a jitter-robust projection fail the pre-registered bias checks in the original space because of the change of space [ev 74].
The out-of-domain checks do not separate the occluder arms from the same-step baseline: IntPhys 1 late occlusion (180 pairs) 87.2 to 87.8, the arc at apex 48 20.8, the ledge 29.9, and the same natural-video loss and carry gain on K400 validation clips [ev 76, table 1-11a].
There are four independent tests: A2c barely moved from AX during training, and A1 and A1′ always combine weights and projection [ev 74, 76].
The pilot rules out only effects of about 3 percent of the oracle margin or more at this budget, by an approximate calibration [ev 76].
In a pre-registered full-budget test, with the baseline's peak learning rate and global batch, 5 percent of its training steps and two seeds, the input occluder still does not carry hidden moving objects: against a same-condition baseline arm it keeps them slightly more near the last observation (+0.4 and +0.6 percent of the oracle margin) and slightly less at far blocks (−0.45 and −0.53 percent, intervals below zero), the binary measure stays within 1.4 points, and the far-block margin does not rise over training [ev 85]. Its carry share stays at the baseline's level (0.007 to 0.008 at 4 fps) [ev 81].
The effect is 0.5 percent of the oracle margin, so we do not call it harmful. The result covers the input-occluder task with the same loss and the same target space under continued training at 5 percent of the baseline's steps; a change of target space was not tested at this budget. Natural-video loss (0.559 against 0.561), IntPhys 1 late occlusion (75.0, equal to the baseline) and false carry do not drop; carry gain on visible moving objects rises slightly (0.31 and 0.29 against 0.24 for the same-condition arm).

## 4.6 Natural video and downstream use (expected)

We report carry gain on held-out K400 validation clips (optical-flow pseudo-labels), by distance from the last observation, with the true-target ceiling.
The baseline's carry gain there is 0.45, against 0.21 for copy, 0.29 for the context-average copy and a ceiling of 0.78; because copy already scores 0.21 under this readout, we read only differences from copy and the context average [table 1-3a].
We also check false carrying, motion that appears in cells that should not move (static clips 0.399, far cells of moving clips 0.334, against 0.414 and 0.413 for copy); a method that carries more must not raise these values [table 1-3a].
Downstream action anticipation on EK100 does not separate predictors (replacing the predicted future with copy or an average changes action recall by 1.3 to 2.6 points; four predictors span 0.36 points) [ev 49]; we use it only as a check that nothing drops.

## 4.7 Action-conditioned post-training (future work, not a contribution)

We replicate DINO-world's comparison (from scratch vs post-training on an observation-trained predictor) on our predictor and add 10, 30 and 100 percent of the action data, scored by held-out action-conditioned rollout error.
Our small Push-T test found no difference (0.60 vs 0.60 over 50 episodes) [ev 45]; the two results are reported side by side with their conditions.

## 4.8 Limitations

- One encoder family and two block-AR predictors carry every placement and fall claim. Other latent models were scored only with all-token scores [ev 48], and Cosmos has no oracle in its space [ev 71], so neither supports generality yet (G10, G20, G21).
- Placement and the ledge are judged with the object-region score, which the oracle passes; the all-token score cannot judge them [ev 20, 23, 57, 68]. Carry gain on natural video reads 0.21 for copy, so it is read only as a difference from copy and the context average.
- Late occlusion in VOBench is solved by one simple baseline: tracking the object with labels and pasting its last fully visible tokens on a quadratic extrapolation of its path reaches 95.3, equal to the oracle (86.7 with constant velocity), against 51.4 for the baseline [plan section 11; perfect perception assumed]. All motions in VOBench have constant velocity or constant acceleration. The test set therefore shows the failure of trained predictors, but any method gain must be reported next to this baseline and on clips whose paths are not deterministic.
- Small cells: 108 pairs per object set under late occlusion, 12 groups per apex time on the arc.
- Several mechanism numbers come from another team and are not yet committed [ev 12, 33, 36, 37, 40, 54, 55, 58].
- The small pilot is not a verdict [ev 76]. The full-budget test covers one task (the input occluder) with the same loss and target space at 5 percent of the baseline's steps, below the 10 percent a reviewer asked for [ev 85]; a change of target space was not tested at this budget.
- A tracking baseline without labels already beats the trained predictor on late occlusion in VOBench: tracking the object in the context frames and pasting its tokens on a quadratic extrapolation reaches 76.1 (far blocks 69.0) against 51.4 (41.3), though it fails on hidden static objects (63.9) [ev 86]. On the ledge the same baseline scores 5.6 percent [2.1, 9.7], and 0 percent even with labeled positions, against 30.6 for the trained predictor and 100 for the oracle [ev 86], so the ledge is the main axis for judging any method.
- The six latent target spaces with low carry reliability were measured by another team, are not committed and do not pass our tool check [ev 78].
- The carry-share result was found post hoc and then replicated on new clips under a pre-registered rule [ev 81]. Natural video has no oracle check for it, and flow error lowers the data value, so 4 to 13 percent is a generous estimate of what the models carry.
- Training-free tests do not answer whether a video encoder or an image encoder is the better block encoder: a positive control that reads two raw frames with a small nonlinear readout reaches only R² 0.072, below the pre-set bar of 0.3, so the motion readout is not a valid tool here [ev 84; corrected in this round from "no reason to prefer V-JEPA 2"]. We keep only the jitter result: jitter is a property of the V-JEPA 2 encoder itself (0.36 for a duplicated single frame, 0.26 for two frames, 0.05 for DINOv2). The encoder question needs a training run that swaps the encoder.
