# 3. Method (draft)

**Question**: what must the method change, which arms must it beat, and how do the arms test the cause?
**Answer (v3.2, after the full-budget test)**: the diagnosis asks for a latent block-AR predictor that uses history to carry the tokens of hidden moving objects to their future cells. A small pilot of training-signal changes was not a verdict, so we ran a pre-registered full-budget test of the input-occluder task. It is negative at far blocks (−0.5 percent of the oracle margin), and the carry share stays at the baseline's level. The next direction (a structural carry operation, or a DINOv2 target) is a user decision.
**Next**: user decisions D8, D10 and D11.

Version: v3.2 (full-budget test, 2026-10-23). Results from evidence v3.2 [ev 72 to 76, 78, 81, 85]. Terms follow the team glossary (STORY.md section 0, English column).
Markers: `[ev N]` evidence claim (evidence table v3), `unverified` other-team result, `(expected)` no result yet, `[plan v2]` result reported in plan v2.

---

## 3.1 Setup

**Frozen encoders.**
We use the context encoder and the target encoder of V-JEPA 2 ViT-H and keep both frozen.
Only the predictor is trained.

**Blocks and block tokens.**
A block is two consecutive frames, the temporal size of one encoder tubelet.
We encode each block alone with the target encoder and apply a parameter-free layer norm.
This gives block tokens $h_i \in \mathbb{R}^{256 \times 1280}$, one token per cell of a $16 \times 16$ grid.
One block is 0.5 s at the training frame rate and 0.2 s on VOBench.

**Block-AR predictor.**
The predictor $f_\theta$ reads context blocks $h_1, \dots, h_C$ with block-causal attention and predicts one block at a time.
In a rollout it feeds its own output back as the next input: $p_{C+k} = f_\theta(h_{1:C}, \tilde p_{C+1:C+k-1})$, where $\tilde p$ is either the predicted tokens (output feedback) or the hidden vector before the output projection (hidden feedback).
This is the interface of the action-conditioned predictor of V-JEPA 2-AC, so the predictor can be post-trained with actions and used for planning.

**Observation-only training.**
The loss is a point-estimate loss: for every cell of a future block the predictor outputs one vector, and the loss is the L1 distance to the one target token of the future that actually happened, averaged over cells and dimensions.
When the future is uncertain, the optimal point is a summary of the possible targets (for L1, the per-dimension conditional median), so a prediction that carries an object only part of the way can be optimal.
The loss is applied under teacher forcing and over a rollout of up to 8 steps:
$$\mathcal{L} = \sum_{k} \frac{1}{256} \sum_{c} \lVert p_{C+k}[c] - h_{C+k}[c] \rVert_1 .$$
No action labels and no object labels are used.
The observation-only block-AR baseline is trained this way for 70 epochs on natural video, with hidden feedback [source table, to confirm: data mix and frame rate].

## 3.2 What the diagnosis requires

**Two places where jitter enters.**
Block tokens jitter under invisible pixel perturbations [ev 36, unverified].
In a block-AR predictor the jitter appears in the inputs $h_{1:C}$ and in the targets $h_{C+k}$.
Write $h_i = \bar h_i + n_i$, where $\bar h_i$ is the content and $n_i$ the jitter.
If $n_{C+k}$ is independent of the inputs and symmetric, the L1-optimal prediction (the conditional median) does not change; target-side jitter only raises the loss floor (theory).
Input-side jitter does change it: with noisy inputs, the optimal prediction shrinks toward an average, and averaging several context blocks reduces the input noise (theory).
Averaging and shrinking keep static content and leave moving content behind.
This matches what trained predictors do: they barely use history, although history is useful, and they shrink the last observation by a factor of 0.54 on static cells [ev 58, unverified]; they place hidden static objects but not hidden moving ones [ev 57].

| Requirement | Why | Kind |
|---|---|---|
| State in one sentence how the method differs from the block-AR baseline and the prior observation-only predictors | The user requires beating the prior predictors | rule |
| Removing jitter is not enough, and a target-only change fails | Noise-averaging cuts the reward for context averaging by only 15 percent, below the pre-set 25 percent line; the part that grows with the step is smearing, not jitter [ev 64]. The short-step reward comes from input-side jitter [ev 66] | measured |
| Make history pay: the prediction for a hidden moving object must come from earlier blocks | Trained predictors use 1 to 2 percent of history while a training-free average uses 30 to 49 percent [ev 58]; the tokens of hidden moving objects are not carried to their future cells [ev 57] | measured |
| Make carrying pay more than averaging or shrinkage | Trained predictors sit near averaging and shrinkage [ev 30, 32, 58] | hypothesis |
| Keep the block-AR architecture and token space | Planning needs the action-conditioned interface | rule |
| Train on natural video only | Synthetic clips would make the synthetic testbed in domain | rule |
| Keep training and rollout inputs consistent | In a rollout the predictor's own output is the next input; removing jitter only from teacher-forced inputs creates a mismatch | hypothesis |

## 3.3 Training-signal changes we tried

Each change targets one link of the diagnosis. All changes keep the architecture, the token spaces, the rollout and the natural-video data, so any trained predictor still plugs into action-conditioned post-training.

**Input occluder (a task that needs history).**
On a fraction of training steps we draw a solid rectangle on the input frames only, before encoding, from a random context block to the last observation, covering 15 to 40 percent of the frame. The targets come from the unoccluded frames, and prediction, targets and loss stay in latent space. The occluder produces real occluder tokens, as in our late-occlusion test set; the block-AR predictor has no mask token, so we do not zero tokens. It adapts the masking prescription of Xie and Alanwar [2026] to a block-AR predictor. The intervention differs: they mask predictor tokens with a mask token, and we draw an occluder on the input frames.

**Motion-placed occluder (a task where hidden content moves).**
A random occluder mostly covers static background, so the hidden content rarely moves during training. One arm therefore places the occluder on the cells whose pixels changed most in the two frames before the occluder starts. The placement uses the inputs only, so it does not depend on the targets and does not change the loss optimum. No mask of Xie and Alanwar [2026] chooses its place from motion; the closest one follows a detected hand.

**Carry-weighted prediction loss (a loss that pays for carrying).**
Under the point-estimate loss, the optimal prediction uses motion-shifted tokens in proportion to how well they predict the next block; we call this proportion the carry reliability κ. It is 0.21 on natural video and 0.32 on deterministic synthetic 3D clips, and the baseline's own carry coefficient (0.18) sits near the natural-video optimum [ev 70, other team, uncommitted]. This motivated the weighted arms. Our placebo-checked measurement later did not reproduce it: the baseline carries about 5 percent of what natural video rewards [ev 81]. We weight each cell by how much its target departs from the mean of the previous four target blocks (target-dependent weights), or, as a control that does not move the L1 optimum, by how much the last input block departs from the previous ones (context-only weights). One arm uses the context-only weights in the original token space, without any projection:
$$\mathcal{L}_w = \sum_k \frac{1}{256}\sum_c w_k[c]\, \lVert p_{C+k}[c] - h_{C+k}[c] \rVert_1, \qquad \tfrac{1}{256}\textstyle\sum_c w_k[c] = 1 .$$

**Jitter-robust projection (a different target space).**
Input-side jitter rewards averaging and shrinkage [ev 66], so two arms pass inputs and targets through a fixed 64-dimensional projection that suppresses jitter directions.

## 3.4 Arms

All pilot arms start from the block-AR baseline and train for 2,000 steps on natural video (K400 train and SSv2 train) with the same data order and seed. This is a small budget: with a global batch of 12, each arm sees 24,000 clips, 0.23 percent of the clips seen by the baseline, with a peak learning rate of 5e-5 (one tenth of the baseline's), decayed to 1e-6, and one seed [ev 76]. A user run with short contexts is reported alongside; it differs from the baseline in window, feedback, loss, data and training amount, but not in time scale (4 fps, 0.5 s blocks).

| Arm | What changes |
|---|---|
| A0 same-step baseline | nothing |
| AX occlusion only | input occluder at random places (adapts the masking prescription of Xie and Alanwar) |
| A1 | input occluder + target-dependent carry weights + jitter-robust projection |
| A1′ | input occluder + context-only carry weights + jitter-robust projection |
| A2c | input occluder + context-only carry weights, original space, no projection |
| A4 motion-placed occluder | input occluder placed on cells whose pixels changed most just before the occluder starts (chosen from the input only), no weights, no projection |
| Short-context run | contexts of 1 to 2 blocks, long rollouts, 30 of 70 epochs |

## 3.5 Results of a small-budget pilot (not a verdict)

The primary measure is object-region pairwise accuracy on moving objects hidden at the last observation (360 pairs), where copy scores 51.1, the context-average copy 50.8 and the oracle 95.3. Arms are compared by paired differences with block-level bootstrap 95 percent intervals [ev 72 to 76, tables 1-11a and 1-11b]. Because the binary measure hides small moves, we also report the placement margin: the error to the impossible future minus the error to the possible one at the object cells, read as a share of the oracle's margin [ev 76]. This margin was added after the results; the pre-registered rule is binary.

| Arm | Primary (vs A0) | Hidden static (B1) | False carry, static · far cells (B2, lower is better) | K400 validation L1 (B3, lower is better) | Visible moving · carry gain |
|---|---|---|---|---|---|
| Baseline (start) | 51.4 | 88.9 | 0.399 · 0.334 | 0.561 | 79.2 · 0.22 |
| A0 same-step baseline | 52.2 | 84.7 | 0.403 · 0.338 | 0.560 | 83.3 · 0.25 |
| AX occlusion only | 52.2 (0.0 [−1.4, +1.4]) | 94.4 (+9.7 [+4.2, +16.7]) | 0.397 · 0.337 | 0.560 | 81.7 · 0.28 |
| A1 | 50.6 (−1.7 [−3.6, +0.3]) | 61.1 (−23.6) | 0.591 · 0.521 | 0.620 | 70.0 · 0.19 |
| A1′ | 50.6 (−1.7 [−3.3, 0.0]) | 68.1 (−16.7) | 0.593 · 0.522 | 0.621 | 63.3 · 0.05 |
| A2c | 52.2 (0.0 [−1.4, +1.4]) | 95.8 (+11.1 [+4.2, +19.4]) | 0.397 · 0.337 | 0.560 | 81.7 · 0.28 |
| A4 motion-placed occluder | 51.7 (−0.6 [−1.9, +0.6]) | 94.4 (+9.7 [+4.2, +16.7]) | 0.399 · 0.337 | 0.560 | 80.8 · 0.28 |
| Short-context run, 8-block context | 52.6 (+1.2 [−0.8, +3.2] vs baseline) | 100.0 | 0.421 · 0.370 | 0.562 | 72.5 · 0.10 |
| Short-context run, 2-block context | 50.6 (−0.8 [−2.2, +0.3] vs baseline) | 79.2 | 0.448 · 0.384 | 0.565 | 82.5 · 0.17 |
| Copy · context average · oracle | 51.1 · 50.8 · 95.3 | 63.9 · 100 · 90.3 | 0.414 · 0.413 (copy) | 0.720 · 0.571 | 68.3 · 56.7 · 99.2; ceiling 0.90 |

- **On the binary measure, occlusion alone looks like "keep it where it was".** AX leaves the primary measure unchanged and raises hidden static objects, the outcome that Xie and Alanwar report as freezing [ev 73]. The context average already reaches 100 on static cells.
- **The continuous margin moves slightly in the right direction for the occluder arms** [ev 76]. Against A0, AX moves toward the true position by 1.45 percent of the oracle margin [+1.01, +1.91], A2c by 1.50 percent [+1.05, +1.96] and A4 by 0.99 percent [+0.66, +1.33]; the two projected arms move away (−3.56 and −3.40 percent). Only 8 of 360 binary pairs differ between the occluder arms and A0, so the narrow binary intervals mostly reflect how little the models changed.
- **Both projected arms fail the same way, and the projection is the common cause.** In the original space they lose static placement, invent motion and raise natural-video loss; scored in their own projected space these collapses vanish (A1′) or shrink (A1), while the primary measure stays unchanged there too (51.9 and 50.8) [ev 74]. The target-dependent weights are therefore not needed to explain the failure.
- **Moving the occluder onto moving cells does not move the binary measure** [ev 76]. A4 targets the pre-registered guess that random occluders teach freezing because they mostly cover static background. With the occluder on moving cells the primary measure is −0.6 [−1.9, +0.6] against A0. At this budget this does not reject the guess, since the margin of A4 is above zero. The three occluder arms (AX, A2c, A4) all raise only hidden static objects (+9.7, +11.1, +9.7), and they match A0 on false carry, natural-video loss and carry, and the out-of-domain checks (IntPhys 1 late occlusion, the arc at apex 48, the ledge).
- **Weighting is not judged.** A1 and A1′ always combine weights and projection, and even in their projected space they lose hidden static objects (−8.3 [−18.1, +1.4]) [ev 74]. **Context-only weights in the original space are not a clean test either.** A2c matches AX, but its weights moved only 0.25 from AX during training, 6 percent of the distance between AX and A0 [ev 76]. We read A2c as "these weights barely changed training", not as "weighting has no effect".
- **Short contexts do not help either** [ev 75]: the primary measure stays at copy level; a 2-block context improves visible carrying (82.5) but loses static placement (79.2) and the arc (0 of 72).
- **The binary gap is flat, but the margin grows.** The binary gap between AX and A0 does not change from epoch 5 to epoch 10 (−0.6 [−2.2, +0.8]) [ev 72], while the margin gap grows from 1.23 to 1.45 percent (+0.22 [+0.12, +0.32]) although the learning rate decays [ev 76]. A pre-registered rule that asked for a 4-point binary rise could hardly pass under this decay, so we withdraw our earlier decision not to scale up.
- **What the pilot rules out.** By an approximate calibration, a move of about 3 percent of the oracle margin or more would show on the binary measure, so the pilot rules out only effects of that size at this budget [ev 76]. The calibration is approximate, because L1 is not linear in mixing.
- We state the budget with the result: 24,000 clips per pilot arm (0.23 percent of the baseline's), one tenth of its peak learning rate, one seed; 30 of 70 epochs for the short-context run. We do not claim that these changes fail, and we do not claim that no training signal can work.

## 3.6 What the pilot leaves open

We fix the success rule for A4 before the run: the primary measure must rise above A0 and AX with paired intervals that exclude zero, and the bias checks and four out-of-domain checks must pass. A4 does not pass the first condition at this budget.

**Full-budget test.** We continued training the occlusion-only arm and a same-condition baseline arm with the baseline's peak learning rate (5.2e-4) and global batch, for 5 percent of the baseline's steps, with two seeds, and fixed the decision rule before the run: success needed a far-block margin gain of at least 10 percent of the oracle margin, above zero in both seeds, and a binary interval above zero. In a pre-registered full-budget test, with the baseline's peak learning rate and global batch, 5 percent of its training steps and two seeds, the input occluder still does not carry hidden moving objects: against a same-condition baseline arm it keeps them slightly more near the last observation (+0.4 and +0.6 percent of the oracle margin) and slightly less at far blocks (−0.45 and −0.53 percent, intervals below zero), the binary measure stays within 1.4 points, and the far-block margin does not rise over training [ev 85]. Its carry share stays at the baseline's level (0.007 to 0.008 at 4 fps) [ev 81]. The effect is 0.5 percent of the oracle margin, so we do not call it harmful. The result covers the input-occluder task with the same loss and the same target space under continued training at 5 percent of the baseline's steps; a change of target space was not tested at this budget. The re-warmed baseline arm itself lost 2.0 to 2.5 percent of margin against the original baseline, so we compare only with the same-condition arm.

We read the pilot with a placebo-checked carry measurement. Natural video rewards a carry share of 0.134 to 0.189, and every trained predictor carries 0.007 to 0.025, including the pilot arms and the baseline arm of the full-budget test at its first checkpoint [ev 81; found post hoc, replicated under a pre-registered rule]. The pilot arms carry the same small share as the baseline, which fits a binary measure that does not move. Why trained predictors carry so much less than the data rewards is open. The low natural-to-synthetic ratios of carry reliability in six target spaces come from another team and do not pass our tool check, so we do not use them [ev 78].

A generative model trained with a distributional loss on reconstruction latents (Cosmos) also scores low on the same cells, decoded with argmax [ev 71]. Those cells cannot be judged under our rules, so this is neither support for nor evidence against any cause.

Any method gain on VOBench late occlusion must be reported next to a tracking baseline: pasting the last fully visible object tokens on a quadratic extrapolation of the labeled path already reaches the oracle (95.3), because every VOBench path has constant velocity or constant acceleration [plan section 11], and the same baseline without labels reaches 76.1 against 51.4 for the trained predictor [ev 86]. On the ledge this baseline scores only 5.6 percent [ev 86], so any method is judged mainly there. A method claim therefore also needs clips whose paths are not deterministic.

The next direction is open (user decision D8). Options are a loss on object position and presence with deterministic synthetic targets, which gives up label-free training; an analysis paper built around the law; or both together.

## 3.7 Action-conditioned post-training (future work, not a contribution)

DINO-world [2025] already reports that observation pretraining helps action-conditioned planning (Push-T 46.9 to 59.4) [cite, to verify].
Our own small Push-T test shows no difference (0.60 vs 0.60 over 50 episodes) [ev 45].
We therefore do not claim this as new.
We replicate the comparison on our predictor and add one axis: 10, 30 and 100 percent of the action data.
The action data and the encoder for this study are open decisions.
