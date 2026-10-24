# 1. Introduction (draft)

**Question**: what does the introduction claim, in which order, and on which evidence?
**Answer (v3.2, after the full-budget test)**: two observation-only block-AR predictors on a frozen V-JEPA 2 encoder do not carry the tokens of moving objects hidden at the last observation to their future cells, and they do not choose falls beyond prediction-free baselines. Freezing and presence failures are the results of Xie and Alanwar [2026], which we treat as prior work. Trained predictors carry only about 4 to 13 percent of the motion-shifted content that natural video rewards, and a larger learning rate and budget do not change this. A pre-registered full-budget test of the input-occluder task is negative at far blocks. The cause is open.
**Next**: user decisions on the direction (D8), a structural carry operation judged on the ledge (D10) and a DINOv2-target block-AR (D11); a second latent family before the scope widens.

Version: v3.2 (full-budget test, 2026-10-23). Terms follow the team glossary (STORY.md section 0, English column).
Markers for reviewers (removed in the final text): `[ev N]` = evidence claim N (evidence table v3); `unverified` = other-team result, not committed; `(expected)` = no result yet; `[cite, to verify]` = citation not checked against the original.

---

**Working title (placeholder).** Block-AR Predictors on a Frozen V-JEPA 2 Encoder Shrink the Last Observation Instead of Carrying What Is Hidden

## 1.1 Paragraphs

**P1. Intuitive physics lets people imagine plausible futures.**
A child who sees a ball roll across a table expects it to keep rolling.
A ball that rolls behind a box should come out on the other side.
These expectations belong to intuitive physics.
Infants show them early, and they acquire them largely by watching [cite, to verify].
They let people imagine plausible futures and adapt quickly to new scenes and tasks [cite, to verify].

**P2. Latent world models are credited with intuitive physics, and a fair test needs prediction-free baselines.**
Latent world models of the JEPA family try to learn the same knowledge from video without generating pixels.
An encoder turns frames into tokens, and a predictor forecasts the tokens of future frames in that latent space; nothing is decoded back to pixels.
We study this latent prediction setting only.
We separate self-supervised latents, which are never decoded (V-JEPA 2, DINOv2), from reconstruction latents of generative tokenizers, which are trained to be decoded back to pixels (Cosmos). Both are latent prediction, and both let one choose the target space.
These models now support planning, forecasting and action anticipation [V-JEPA 2; V-JEPA 2-AC; DINO-world].
Intuitive-physics benchmarks credit them through violation of expectation: the model should be more surprised by an impossible future than by a possible one [Garrido et al. 2025; IntPhys 2].
Part of this credit does not require prediction.
A copy baseline, which repeats the tokens of the last observation, reaches 62 to 85 percent pairwise accuracy across the six IntPhys 1 evaluation cells [ev 2].
At the best cell the released V-JEPA 2 predictor is not separable from copy (88.9 vs 85.0); at the official cell it leads by 9.4 points [+3.3, +15.6] (83.3 vs 73.9) [ev 1, 2].
We therefore report every score next to two prediction-free baselines, copy and a context average, and next to a true-position oracle that checks whether the score can see success at all.
This is an evaluation protocol, not a contribution on its own; related checks appear in Garrido et al. [2025], Xie and Alanwar [2026] and DINO-world [2025].

**P3. Two observation-only block-AR predictors on V-JEPA 2 do not carry hidden objects to their future cells and do not beat prediction-free baselines on falls.**
We sort each test by what the last observation contains, and we first check which score can see success.
We say a predictor places an object when, at the cells where the object should be, its predicted tokens are closer to the target tokens of the possible future than of the impossible one; this is a statement about latent tokens, not pixels.
An all-token score cannot see placement: an oracle that moves the last fully visible object tokens to their true future cells scores the same as copy (53.3) under late occlusion [ev 20].
An object-region score can: the same oracle reaches 95.3 under late occlusion, 99.2 for visible motion and 100 on our gravity set [ev 57, 67, 68].
With this score and three same-space references (copy, context-average copy, oracle), we measure two observation-only block-AR predictors in one harness [ev 67 to 69, table 1-3a].
Neither puts the tokens of a moving object hidden at the last observation at its future cells: both score 51.4, the same as copy (51.1) and the context average (50.8), against 95.3 for the oracle.
How far their predicted object tokens move for visible objects depends on the model: the carry gain is 0.22 for the baseline and 0.65 for a prior predictor trained on shorter blocks, against a ceiling of 0.90. The two models were trained on blocks of 0.5 s and 0.17 s, so part of this gap may come from the time scale.
Its high all-token presence score (94.2) equals that of a training-free context average (94.2), and the released predictor fails presence as Xie and Alanwar [2026] report [ev 16, 67].
Neither chooses the fall at a ledge beyond prediction-free baselines: 30.6 and 7.6, against chance 50 and a context average of 41.0, while the oracle reaches 100. On the arc, neither is separable from the context average (11.1 and 25.0, against 16.7).
On held-out natural video its loss is only 2 percent below that of the context average (0.561 vs 0.571; copy 0.720).
For hidden static objects the object-cell tokens are right (88.9 against 63.9 for copy), but a training-free context average already reaches 100 there [table 1-3a].

**P4. Retraining helps only inside the training domain.**
Fine-tuning on our gravity set solves the arc in domain (32 of 36 groups), but on IntPhys 1 the same model drops by 6.7 points [−12.2, −1.7] at the window-32 cell and stays level at the window-16 cell [ev 24, table 1-8a].
Variants of the baseline that mix in the last observation, add occlusion clips, add a temporal-difference loss or add object-label supervision do not raise carrying [ev 42, 43].
A video-context predictor carries about twice as far (0.59 vs 0.29) but loses presence under late occlusion and falls on IntPhys 1 [ev 12, 21].
So the question is not only how much to train, but what the training signal rewards.

**P5. The trained predictor shrinks the last observation and carries far less than the data rewards.**
History is useful: a training-free average over the context blocks beats a two-block average, with a history value of 0.30 to 0.49 [ev 58].
Yet the trained predictors barely use it: cutting the context from 8 to 2 blocks costs 1 to 2 percent of their gain over copy on held-out K710 and SSv2 clips [ev 31, 58].
Instead, on static cells the prediction is close to the last observation shrunk toward an average, with a factor of 0.54, near the data-optimal 0.49 to 0.52 [ev 58, unverified].
A hidden moving object exists only in the tokens of earlier blocks, so a predictor that shrinks the last observation has no object tokens to carry to the future cells.
Why shrink? The loss rewards it.
Observation-only predictors are trained with a point-estimate loss: for every cell of a future block the predictor outputs one vector and is scored by its L1 distance to the one target token of the future that actually happened.
When the future is uncertain, the best single point is a summary of the possible targets (for L1, the per-dimension conditional median), so a prediction that moves an object only part of the way can be optimal.
Block tokens change under perturbations that are invisible in pixels, by 32 to 39 percent of the distance between different videos [ev 36, unverified]. This jitter is a property of the V-JEPA 2 encoder itself: it is about five times that of DINOv2 (0.26 vs 0.05), and a duplicated single frame shows it too (0.36), so it does not come from the two-frame block [ev 84].
Target-side jitter of this kind, if independent and symmetric, does not move the L1-optimal prediction (theory) [ev 63].
We measure the two sides separately: removing jitter from the inputs alone cuts the reward for context averaging from 16.8 to 6.8 percent and the reward for shrinkage from 4.6 to 0.3 percent, while removing it from the targets alone raises the averaging reward to 24.4 percent [ev 66].
On our testbed, a training-free context average beats copy by an amount that grows with the step, from about 0 at the first step to 21 percent at the eighth [ev 64].
Noise-averaging the tokens removes the short-step part but not the growing part, which comes from smearing moving objects over their possible positions; carrying the object to its true position adds only 1.3 percent on top [ev 64].
The trained baseline behaves like this solution at short steps: its next-step position follows the last observation and continues its motion, while the block before it barely matters [ev 65].
This describes the short-step behaviour: shrinkage plus first-order continuation of the last observation.
For moving content, however, the trained predictors carry far less than even this point-estimate solution would.
We measure the carry share with a placebo check: the weight on the last block shifted along the true flow, minus the weight on the same block shifted along a direction rotated by 90 degrees, which removes spatial correlation that does not depend on direction.
On natural video the data reward a carry share of 0.134 to 0.189, while every trained block-AR predictor carries 0.007 to 0.025, about 4 to 13 percent of it, regardless of its training data, budget or window [ev 81]. We found this post hoc and then replicated it on new clips and another frame rate under a rule fixed before the run.
An earlier claim from another team, that the model carries about as much as the data optimum (0.18 vs 0.21), does not survive this check: without the placebo our models also score 0.19 to 0.21, but the same share appears for the rotated direction [ev 70, 81].
Why trained predictors carry so little is open. We make no causal claim.
A generative model trained with a distributional loss on reconstruction latents (Cosmos) also scores low on the same cells, but those cells cannot be judged under our rules, so it is neither support nor a counterexample for any cause [ev 71].

**P6. Changing the training task does not make the predictor carry hidden objects, even at a fair budget.**
Prior observation-only predictors, including ours, train with a point-estimate loss on clips that rarely require history.
We ran four independent changes while keeping the block-AR architecture and natural-video data: an input occluder at random places, an input occluder on cells that were moving, carry weights combined with a jitter-robust projection, and short contexts.
Each pilot arm saw 24,000 clips, 0.23 percent of the clips seen by the baseline, with one tenth of its peak learning rate and one seed [ev 76].
At this budget the binary primary measure stays at copy level for every arm, between 50.6 and 52.6 against 51.1 [ev 72, 76].
A continuous placement margin moves slightly: the occluder arms move toward the true position by 1.0 to 1.5 percent of the oracle distance, with intervals above zero, and the gap grows during training [ev 76]. This margin was added after the results; the pre-registered rule is binary and stays as it is.
The occluder arms also raise hidden static objects, as in the freezing that Xie and Alanwar [2026] report [ev 73, 76].
The pilot alone rules out only effects of about 3 percent of the oracle distance or more, by an approximate calibration [ev 76], so we ran a fair test.
In a pre-registered full-budget test, with the baseline's peak learning rate and global batch, 5 percent of its training steps and two seeds, the input occluder still does not carry hidden moving objects: against a same-condition baseline arm it keeps them slightly more near the last observation (+0.4 and +0.6 percent of the oracle margin) and slightly less at far blocks (−0.45 and −0.53 percent, intervals below zero), the binary measure stays within 1.4 points, and the far-block margin does not rise over training [ev 85]. Its carry share stays at the baseline's level (0.007 to 0.008 at 4 fps) [ev 81].
The effect is 0.5 percent of the oracle margin, so we do not call it harmful. The result covers the input-occluder task with the same loss and the same target space under continued training at 5 percent of the baseline's steps; a change of target space was not tested at this budget.

## 1.2 Closest work

**Xie and Alanwar [2026]** show that a frozen V-JEPA 2 predictor loses a hidden moving object within about 0.3 seconds while the encoder still detects it.
Their three-world readout (keeps moving, frozen at the last seen place, vanished) also shows that continued predictor training with mixed masks freezes objects at their last position.
Our results for the released predictor confirm theirs on our occluder testbed, and we add block-AR predictors trained for future prediction.
We do not claim these results as new. Their paper shaped this submission, so we treat it as prior work.
Our occlusion-only arm adapts their masking prescription to a block-AR predictor. It draws an occluder on the input frames instead of masking predictor tokens, so it is not a reproduction of their method.

**Frozen Flows Forget [2026]** tells a story close to ours, and it also shaped this submission.
A world model on a frozen latent space does not move objects, the cause lies in the training signal rather than the representation, and a still prediction is rewarded by the score.
The mechanisms differ: there, supervision is sparse; here, we hypothesize that input-side jitter makes averaging pay.
The remedies differ: they add pixel-decoder supervision; our candidates change the predictor's inputs or targets in latent space.
They do not evaluate intuitive physics or violation of expectation.

**Slow features.** Sobal et al. [2022] and MotionJEPA [2026, cite, to verify] describe JEPA objectives that drift toward slow features while the encoder is trained.
Our encoder is frozen, and the problem runs the other way: the tokens change too fast under invisible perturbations.

**Observation-only pretraining before actions.** DINO-world [2025] trains an observation-only predictor on a frozen encoder and post-trains it with actions; its Table 4 reports higher planning success after observation pretraining (Push-T 46.9 to 59.4) [cite, to verify].
We do not claim this as new.
Our own small Push-T test shows no difference (0.60 vs 0.60 over 50 episodes) [ev 45]; we report the two results side by side with their conditions.

**Feature forecasting with alignment.** DF3 [2026, abstract checked] and F2MF [Šarić et al., CVPR 2020] align past features along motion before forecasting; a trajectory-aligned predictor we considered is close to them.

## 1.3 Contributions

1. **A diagnosis with a checked score.** Two observation-only block-AR predictors on a frozen V-JEPA 2 encoder do not carry the tokens of hidden moving objects to their future cells and do not choose falls beyond prediction-free baselines; we measure with an object-region score that a true-position oracle passes and with three same-space references [ev 57, 67, 68]. Freezing and presence failures are results of Xie and Alanwar [2026]; our share is the checked score, the block-AR setting and the first judgeable test of falls.
2. **How much is carried.** A placebo-checked measurement shows that trained block-AR predictors carry about 4 to 13 percent of the motion-shifted content that natural video rewards, found post hoc and replicated under a pre-registered rule [ev 81]. The earlier claim that they sit at the point-estimate optimum does not reproduce. They also shrink the last observation instead of using history [ev 58, partly unverified]. Natural video has no oracle check for this measurement.
3. **A pre-registered negative result at a fair budget.** The input-occluder task, the closest adaptation of the masking prescription of Xie and Alanwar [2026] to a block-AR predictor, does not carry hidden moving objects to far blocks at 5 percent of the baseline's training steps with two seeds, and the carry share does not change [ev 81, 85]. The scope is this task with the same loss and target space.
4. **An evaluation protocol**: three same-space references (copy, context-average copy, true-position oracle) for every score, and failure claims only where the oracle passes.

The small-budget pilot (Section 4.5) is reported as context for item 3. Action-conditioned post-training is future work, not a contribution.

## 1.4 Notes for the next revision

- P1 has no verified citation yet. "Animals" and "adaptation" stay out until a citation is checked.
- P3 must give time scales in the final text: one block is 0.5 s in training for the baseline, 0.17 s for the prior predictor, and 0.2 s on VOBench.
- P3 compares models with different scoring spaces and context encoders [ev 50]; each number is read against its own same-space baseline.
- The leakage word stays out of the paper text until the user decides (decision D7). P2 says "prediction-free baselines".
- Baseline numbers come from one evaluation harness [ev 67 to 69]; the prior predictor's harness row is being measured.
- P6 counts four independent tests: A2c barely moved from AX during training (6 percent of the AX to A0 distance), and A1 and A1′ always combine weights and projection, so weighting itself is not judged [ev 74, 76].
- The continuous margin, the budget ratio and the approximate 3 percent bound are in the evidence table [ev 76]; the bound is a reviewer calculation with an L1 nonlinearity caveat.
- The scope is V-JEPA 2 block-AR predictors until a second latent family is measured with the object-region score (G20, G21).
- The contribution list has no method item. Whether a method returns depends on the user decisions D8, D10 and D11.
- Item 3 is a negative result; whether it counts as a contribution is the leader's and the user's call.
