# Parabolic Prediction — V-JEPA2 Predictor-Surprise Physical-Plausibility Probe

A controlled physical-reasoning eval that asks: **given a video of a ball falling
toward the ground, does the V-JEPA2 *predictor* expect it to keep falling (physical)
rather than float or fly back up (impossible)?** We measure this entirely in the
pretrained model's latent space — no training, no fine-tuning.

---

## 1. Motivation & core idea

V-JEPA2 is a **video encoder + latent predictor**. The predictor is trained to
predict, in representation space, the tokens of *future* space-time patches from a
*context* of earlier patches. We reuse that exact machinery as a physics probe:

> Feed the model the context (a ball launched from the ground, now descending).
> Have the predictor imagine the future. Then compare its imagined future to the
> *encoded* representation of several **candidate** futures. If the model understands
> gravity, its prediction should be **closest to the physically-correct continuation**
> (ball keeps falling and lands) and **far from impossible ones** (ball freezes / flies up).

The comparison is a plain L1 distance in latent space — the same quantity as the
V-JEPA training loss.

### Why not just feed single candidate *images*?
Because the encoder is a **video** encoder (spatio-temporal tubelets). A standalone
image is out-of-distribution and its latent is not comparable to the predictor's
output. The target must be the encoder's representation of the future frames **within
a full clip** (contextualized). Hence every candidate is a **full video**.

### The leakage subtlety (and its fix)
The encoder uses **full bidirectional attention**. If you encode a full clip and slice
out the context tokens, those tokens have already attended to (leaked) the future.
The predictor, however, was trained on a **masked** context (future tokens dropped
*before* the transformer). So:

- **context (predictor input)** = `context_encoder(clip, masks=[ctx_idx])` — masked,
  leak-free. Tubelets are non-overlapping, so context tokens depend only on context
  frames.
- **target (comparison)** = `target_encoder(clip)` (full, unmasked) → gather the
  future-token positions. Leakage here is fine: it is the *target*, never the input,
  and a contextualized target is exactly what JEPA is trained to match.

### Train-faithful encoder roles (dual encoder)
Training uses TWO encoders (`app/vjepa/train.py`): the **online encoder** feeds the
predictor (`z = encoder(clip, masks)`), and the **EMA target encoder** produces the
regression target (`h = target_encoder(clip)`). We mirror that exactly
(`dual_encoder: true`): context → checkpoint key `encoder`, target → `target_encoder`.
The two weight sets genuinely differ in the released checkpoint (EMA), so using the
EMA encoder for both roles would feed the predictor a distribution it was not trained
on.

This is precisely what `analysis/intphys2/surprise.py` implements; we reuse it.

---

## 2. Dataset design (`data_gen/make_parabolic_dataset.py`)

One **scene** = a diverse parabolic arc, rendered as **3 variants that share the exact
same context** and differ only in the future:

```
frame:  0 ─────────── ~20 ──────── 31 │ 32 ───────── 39
        launch(ground)  apex     ctx-end │  target/future  land
        └──────── context (32 frames, shared) ──────┘└─ 8 frames ─┘
```

| variant   | future (f32..f39)                     | physical? | violation      |
|-----------|---------------------------------------|-----------|----------------|
| `possible`| keeps falling → **lands** on ground   | ✅ yes    | none           |
| `higher`  | **reverses upward** (anti-gravity)    | ❌ no     | anti_gravity   |
| `frozen`  | **completely static** (hovers)        | ❌ no     | static         |

**Shared context is rendered once**: the `possible` clip's frames 0..39 are rendered,
then only frames 32..39 are re-rendered for `higher`/`frozen` (overwriting those PNGs).
So frames 0..31 are byte-identical across the 3 mp4s at encode time, and the eval
additionally splices the decoded `possible` context onto every variant to stay
identical after lossy decode.

### Timeline / why 40 frames, context 32
- `N=40 @ 16fps` → launch f0, apex ≈ f20, **land f39**. Even frame count → clean
  tubelet(2) alignment (20 tubelets; context 16, target 4).
- **context = 32 frames** (f0..f31) — fixed to match the EK100 pipeline.
- **target = 8 frames** (f32..f39) placed in the **fast late-descent**, where the ball
  moves most, so `possible` vs `impossible` diverge strongly even over a short horizon.

### Divergence (why the signal is strong)
The discriminating signal is not "position change" but **possible-vs-impossible
divergence**, which grows toward landing. Measured from the trajectories (render-free):

| apex | context-end (f31) height | at landing (f39): frozen div. | higher div. |
|------|--------------------------|-------------------------------|-------------|
| 3.0 m | 1.96 m (desc. 2.9 m/s)  | **1.96 m** (~1.1 ball ⌀)      | **2.90 m** (~1.6 ⌀) |
| 2.2 m | 1.43 m                  | 1.43 m (~0.8 ⌀)               | 2.13 m |

`higher` is a stronger violation than `frozen` (opposite direction of motion).

### Diversity (200 scenes, seeded)
`apex ∈ [2.2, 3.0] m` (→ per-clip gravity `g ∈ [2.8, 3.8] m/s²`), horizontal
`|vx| ∈ [2.0, 3.5] m/s`, direction L/R, start x kept in-frame. Ball appearance and
camera are **fixed** (controlled — no appearance shortcut). **Shadow-free**
(removes a shadow-based height shortcut). Rendered at **256px** (= eval `img_size`).

### Output layout
```
data_gen/parabolic_dataset/
├── videos/scene_XXXXXX_{possible,higher,frozen}.mp4   # 200 × 3 = 600 clips, 40f @16fps, 256²
├── metadata.csv        # scene_id, variant, file, is_possible, violation, apex_m, g_mps2,
│                       #   vx_mps, vz0_mps, direction, ctx_end, target_start, n_frames, fps, ...
└── dataset_info.json   # n_scenes, variants, frames, ctx_end, fps, size
```

### (Re)generate
```bash
# vll6, 4 GPUs (~10 min for 200 scenes). --mem is required or SLURM grabs the whole node.
sbatch --nodelist=vll6 --partition=batch_vll --gres=gpu:4 --cpus-per-task=24 --mem=64G \
       --export=ALL,N=200,SAMPLES=32,SHARDS_PER_GPU=3 \
       data_gen/run_parabolic_dataset.sbatch
# knobs (env): N, SAMPLES, FRAMES, CTX_END, SIZE, SEED, SHARDS_PER_GPU
```

---

## 3. Eval code (`evals/analysis_vlm/parabolic/`)

Self-contained, config-driven, **imports** `analysis.intphys2` for the model loader +
low-level forward helpers (never modifies existing code).

```
evals/analysis_vlm/parabolic/
├── dataset.py   ParabolicScenes — decode 3 variants, splice shared context, ImageNet-normalize
├── forward.py   scene_surprises — masked-ctx encode ×1 → predictor ×1 → per-variant target L1
├── scoring.py   score_argmin (3-way min-L1) + score_pairwise (2-way surprise)
├── eval.py      entry point: config → model → scenes → surprises → scoring → outputs
└── configs/parabolic_vitl.yaml
```

### Forward (per scene) — `forward.py`
```
ctx_idx, tgt_idx = context/target token indices (temporal-tubelet-major split at context_length)
z_ctx  = context_encoder(possible_clip, masks=[ctx_idx])      # (1) masked, leak-free, ONCE
z_pred = predictor(z_ctx, ctx_idx, tgt_idx, mask_index=0)     # (2) imagined future, ONCE
for v in {possible, higher, frozen}:                         # (3) per-variant target
    h      = target_encoder(clip_v)                          #     full clip, unmasked
    h_tgt  = gather(h, tgt_idx); (LayerNorm)
    surprise[v] = L1(z_pred, h_tgt)
```
`clip_v` = `possible`'s context ⊕ variant `v`'s future (identical context guaranteed).
1 masked-context encode + 1 predictor + 3 target encodes per scene.

### Scoring — `scoring.py` (both configurable in yaml)
- **argmin (3-way ranking)**: the variant with the **smallest** surprise should be
  `possible`. `accuracy = P(argmin == possible)`, chance `1/3`.
- **pairwise (2-way)**: for each `(possible, impossible)` pair, correct iff
  `surprise(impossible) > surprise(possible)`. ties = 0.5, chance `0.5`.
  Default pairs: `possible-vs-frozen`, `possible-vs-higher`.

### Run
```bash
# in the V-JEPA env (needs torch + decord + the repo deps; NOT the blender env)
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
python -m evals.analysis_vlm.parabolic.eval \
    --config evals/analysis_vlm/parabolic/configs/parabolic_vitl.yaml --device cuda:0
```
Outputs → `z_exp/parabolic_prediction/results/vitl/`:
`per_scene.csv` (surprises + argmin per scene), `summary.json`, `summary.txt`.

---

## 4. Config reference (everything is editable)

`evals/analysis_vlm/parabolic/configs/parabolic_vitl.yaml`

| block | key | meaning |
|-------|-----|---------|
| model | `checkpoint`, `arch_name`, `window_size(=40)`, `predictor{...}`, `dtype` | pretrained V-JEPA2 (schema = `analysis.intphys2.model.build_from_config`) |
|       | `dual_encoder: true`, `context_encoder_key: encoder`, `target_encoder_key: target_encoder` | train-faithful roles: online encoder → predictor input, EMA encoder → target |
| eval  | `context_length(=32)` | context = frames 0..31; target = 32..39 |
|       | `distance` | `l1` \| `smoothl1` \| `l2` \| `cosine` |
|       | `target_layer_norm`, `mask_index(=0)` | train-faithful target LN; only `mask_tokens[0]` is trained |
| scoring | `modes` | any of `[argmin, pairwise]` |
|         | `argmin_candidates`, `argmin_correct` | which variants ranked; which is "correct" |
|         | `pairwise_pairs` | list of `[possible, impossible]` pairs to score |

To use a different model (e.g. ViT-H / dual-encoder), copy the config and change the
`model:` block (same schema as the IntPhys2 eval configs under
`configs/analysis/intphys2/`). To change context/target split, edit `window_size`
(and the dataset `FRAMES`/`CTX_END`) so both stay tubelet(2)-aligned.

---

## 4b. Faithfulness vs the official IntPhys2 release

Audited line-by-line against Meta's released prediction-eval code
(`/data/hyuntak/project/2026/2027_cvpr/IntPhys2/prediction_evals`, esp.
`app/vjepa/modelcustom/default_wrapper.py` + `evals/intphys2/eval.py` +
`evals/intphys2/configs/vjepa_2.yaml`). Our parabolic forward matches the official
V-JEPA2 surprise computation on every axis that matters:

| axis | official | ours (parabolic) |
|------|----------|------------------|
| context encoding | `encoder(x, masks_enc)` — ONLINE encoder, masked | same (`dual_encoder: true`, key `encoder`) |
| target encoding | `target_encoder(x, full)` → gather masks_pred | same (key `target_encoder`) |
| token split | temporal-contiguous first (C/tubelet)·spatial indices | same (`_context_target_indices`) |
| target LayerNorm | targets only, after gather; preds NOT normed | same |
| loss | plain L1 mean over (tokens, dim) | same (`l1`, `loss_exp=1`) |
| mask token | `mask_tokens[0]` (MultiMask wrapper i=0), 10 tokens | same (`mask_index: 0`) |
| precision | bf16 autocast (fp32 weights) | bf16 weight-cast — tiny numeric diff, documented |
| windowing | sliding M=48 windows + `max_context_mode` beginning losses (they don't know WHEN the violation happens) | single window over the whole 40f clip, context 32 — our violation is BY CONSTRUCTION at the first target frame (f32), so beginning/sliding coverage is unnecessary |

Known bug in the official code (NOT copied): their `get_time_masks()` hardcodes
`spatial_dim=(224,224)` while `vjepa_2.yaml` runs at resolution 256 — masks come out
sized 196 tokens/frame instead of 256, misaligning context/target. We compute the
split from `img_size` (the intended semantics).

Official-only mechanisms we deliberately do NOT adopt: the context-length sweep,
sliding windows, and `max_context_mode` all exist because IntPhys2 does not know
WHEN a violation happens inside a long video. Our probe is single-window with the
violation pinned to the first target frame (f32), so none of them apply — the only
official ingredient that matters here is the dual-encoder role split, which we use.

## 5. Design decisions & caveats

- **context = 32 frames, fixed** (EK100 consistency). Target placed in the fast
  descent so an 8-frame horizon still gives large divergence.
- **Masked context is mandatory** (leakage). We always use the masked forward, never
  the leaked full-clip slice, matching how the predictor was trained.
- **`mask_index=0`**: the released checkpoint only trained `mask_tokens[0]`; other
  indices are zero-init and would inject a null query.
- **Shared context is rendered once** and spliced at eval time → the only thing that
  differs across candidates is the future (clean, controlled comparison).
- **Physics is analytic** (exact ballistic), constant gravity per clip; the *value* of
  g is a toy value chosen for framing, but within each clip the motion is exact
  constant-acceleration (see `data_gen/make_projectile_blender.py`).
- **Expected result if the model has physics understanding**: argmin accuracy > 1/3
  and pairwise accuracy > 0.5, with `higher` easier than `frozen` (stronger violation).
  A near-chance result is itself an informative finding (predictor does not encode
  gravity direction over this horizon).

---

## 6. File map

| path | role |
|------|------|
| `data_gen/make_parabolic_dataset.py` | dataset generator (possible/higher/frozen, shared context) |
| `data_gen/run_parabolic_dataset.sbatch` | multi-GPU render → merge → sanity |
| `data_gen/make_projectile_blender.py` | Blender scene/camera/encoder (reused) |
| `evals/analysis_vlm/parabolic/` | eval package (dataset/forward/scoring/eval + config) |
| `analysis/intphys2/{model,surprise}.py` | reused (imported, unmodified): model loader + forward helpers |
| `z_exp/parabolic_prediction/` | this doc + eval results |
