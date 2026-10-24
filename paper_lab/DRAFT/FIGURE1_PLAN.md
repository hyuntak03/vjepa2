# Figure 1: plan

**Question**: what should Figure 1 show in one look, using only scores that pass the oracle check, and without repeating Xie and Alanwar [2026]?
**Answer (v3.1, round 2)**: three panels in one row, all read in latent space. (a) Failure: for two observation-only block-AR predictors on V-JEPA 2, the predicted tokens at a hidden moving ball's future cells are no closer to the true future than copy, and at a ledge they do not choose the fall beyond a prediction-free average (a hidden still ball is a reference). (b) Mechanism: history is useful, yet the trained predictor shrinks the last observation instead. (c) Open: the carry share of trained predictors against the share natural video rewards (pre-registered replication), or the pre-registered full-budget test (near and far-block margins, two seeds).
**Next**: choose panel (c) between the two measured options below.

Version: v3.2 (full-budget test, 2026-10-23). Terms follow the team glossary (STORY.md section 0, English column).
Markers: `[ev N]` evidence claim (evidence table v3), `unverified` other-team result, `(expected)` no result yet.

---

## 1. The sentence the figure carries

A person expects a ball that rolls behind a box to come out further along.
In an observation-only block-AR predictor on a frozen V-JEPA 2 encoder, the predicted tokens at the hidden moving ball's future cells are no closer to the true future than a copy of the last observation, and at a ledge its tokens do not favor the fall more than a prediction-free average.
For a hidden ball that stood still the object-cell tokens are right, but averaging the past tokens alone already does that.
A small pilot of training-signal changes (0.23 percent of the baseline's training clips) leaves the binary score at copy level and moves a continuous margin only slightly, so the figure does not claim that these changes fail.

## 2. What changed from v1 and why

| v1 panel | Change | Reason |
|---|---|---|
| (b) hidden ball, three candidate futures, all tokens | Replaced by panel (a) with the object-region score and an oracle bar | The three-world readout is the design of Xie and Alanwar [2026], and the all-token score does not pass the oracle check (oracle = copy, 53.3) [ev 20]. The object-region score passes it (95.3) [ev 57] |
| (c) ledge fall | Removed in the first v2; **back as an option** | With all-token scores even the oracle fails (7 of 72) [ev 23], but with the object-region score the oracle picks the fall in every group and the baseline picks it in 30.6 percent (chance 50, context average 41.0) [ev 68]. Option: a fourth bar group in panel (a), so the figure shows one absent event as well |
| (c) arc | Moved to Table 2 | With the object-region score the arc can be judged (oracle 100): the baseline scores 11.1, equal to chance, and does not separate from the context average (16.7) [ev 68]. It moves to Table 2 for space, not because it cannot be judged. The all-token view (only apex 48 and 66 judgeable) is in the appendix [ev 23] |

Overlap we state in the caption: Xie and Alanwar also report freezing at the last seen place after masked training.
What panel (a) adds: a block-AR predictor trained for future prediction, a score checked by an oracle, and the still-object contrast.

## 3. Panels

| Panel | Shows | Numbers (object-region score, pairwise accuracy) | Status |
|---|---|---|---|
| (a) Failure | Short filmstrips with the last observation outlined: hidden moving ball, ball at a ledge, and (reference) hidden still ball. For each, bars for copy, context-average copy, oracle and the two block-AR predictors | hidden moving: copy 51.1 · context average 50.8 · baseline 51.4 · prior predictor 51.4 · oracle 95.3. ledge: 31.2 · 41.0 · 30.6 · 7.6 · 100 (chance 50). hidden still: 63.9 · 100 · 88.9 · 93.1 · 90.3 [ev 67, 68, table 1-3a] | measured |
| (b) Mechanism | Left: history value of a training-free context average vs the trained predictor. Right: reward of the context average over copy by step, original vs noise-averaged tokens, with the carry reward on top | history value 0.30 to 0.49 vs 0.011 to 0.014 [ev 58, unverified] · context-average reward −2.3 to 20.8 percent (original) and −8.8 to 21.2 percent (noise-averaged), carry reward 1.3 percent [ev 64] | left from another team, not committed; right recomputed |
| (c) Open (two options) | Option 1, the carry share: for natural video at 4 and 8 fps, the carry share that the data reward (0.134, 0.189) next to the share each trained predictor carries (0.007 to 0.025), with intervals [ev 81]. The caption states that the result was found post hoc and replicated under a pre-registered rule, and that natural video has no oracle check. Option 2, the full-budget test: for each seed, the occluder arm minus the same-condition baseline arm on the binary score, the near-block margin and the far-block margin, with the learning curve of the far-block margin as an inset (−0.23, −0.49, −0.77, −0.49 percent) [ev 85]. The occlusion-only arm is named as an adaptation of the masking prescription of Xie and Alanwar, not a reproduction | Option 2 numbers: far-block margin −0.45 % [−0.78, −0.17] and −0.53 % [−0.80, −0.30]; all blocks +0.39 % and +0.57 % (seeds 0 and 1); binary within ±1.4 points [ev 85] | open (leader and user) |

- Panel (a) is the human hook: the expected future of a hidden rolling ball is obvious, and the predictor scores at copy level.
- The still-ball bar is a reference: for hidden content that does not move, even a training-free average of past tokens is right at the object cells, so the failure is moving content whose tokens must be carried from history.
- Panel (b) states the mechanism with measured numbers; the cause (input-side jitter) stays in the text as a hypothesis.
- Whichever option panel (c) takes, its caption states the budget (5 percent of the baseline's steps, two seeds) and the scope (input-occluder task, same loss and target space). Option 1 does not use the other team's κ values [ev 70, 78].

## 4. Readout rules

- Object-region score: surprise computed on the 3 by 3 cells of the object in the two futures (rendered labels), same pairs and same tie rule as the all-token score.
- Every bar group shows the oracle, so the reader sees that the score can see success.
- All-token values go to the appendix next to each bar.
- No decoded pixels are drawn as if they were the prediction. Filmstrips show the input frames only; every number in the figure is computed on latent tokens.

## 5. Corner-case line for the caption

A moving object hidden at the last observation is rare in IntPhys 1 (moving objects are hidden for about 0.2 s) [ev 47].
The caption states this and points to the planning use case: a latent rollout that loses a hidden moving object's tokens keeps them wrong in every later step.
The use-case sentence is a hypothesis until the action-conditioned study runs.

## 6. Style

- Labels in English; Nimbus Roman; vector PDF; double-column width 7.0 in.
- Panel labels (a), (b), (c) centered under each panel. No narrative text inside the figure.
- Colors: palette slots 1 to 3 only (#2a78d6, #eb6834, #1baf7a); copy in gray; oracle as an outlined bar. In panel (c), option 2 uses slot 1 for occluder arms, slot 2 for projected arms and slot 3 for the short-context run, with a direct label on every bar.
- Object sets: one set per panel, named in the caption.
