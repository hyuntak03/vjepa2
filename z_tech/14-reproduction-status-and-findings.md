# Reproduction status & findings

## Purpose

This section records **what of the paper's physics-interpretability results this fork has
actually reproduced**, the two root-cause debugging wins that made the reproduction work,
one correctness invariant for linear probing, and the phased roadmap for the remaining
paper experiments. It is a *status + findings* note, not a code walkthrough — see the
cross-referenced sections for the machinery it leans on:

- Feature cache / pooling granularities → `z_tech/03-feature-caching-and-pooling.md`
- Probe heads, regression task, NaN-masking, R² → `z_tech/04-probes-regression-nanmask.md`
- `uniform_sampling` / `frame_step` / `resize_mode` data knobs → `z_tech/06-data-pipeline-changes.md`
- Blender toy-physics dataset generator → `z_tech/09-blender-toy-dataset.md`
- Attention hooks (distance/ablation substrate) → `z_tech/11-attention-hooks.md`
- Additive analysis-mode dispatch → `z_tech/12-analysis-modes.md`

The headline result: **the frozen V-JEPA 2-L layer-wise R² dissociation of Fig. 2c
reproduces on the paper-faithful Blender toy dataset** — speed is decodable from the very
first block, direction emerges sharply at the ~one-third-depth Physics Emergence Zone (PEZ),
and acceleration magnitude sits in between.

## What changed vs upstream V-JEPA2

Baseline = commit `204698b` (no `evals/analysis*` tree at all). The reproduction depended
on three concrete fork changes, all additive and default-off:

| File | Kind | Delta that enabled the reproduction |
|------|------|-------------------------------------|
| `src/datasets/video_dataset.py` | **modified** | `uniform_sampling` param + early-return branch in `loadvideo_decord` (`:334-341`) — fixes the contiguous-window sampling bug (finding #2). `git diff 204698b` = pure param-threading + this one branch. |
| `src/datasets/data_manager.py`, `evals/video_classification_frozen/eval.py` | **modified** | `uniform_sampling` threaded through `init_data` / `make_dataloader` (2 lines each). |
| `data_gen/make_physics_blender.py`, `data_gen/sanity_check_blender.py`, `data_csv/make_blender_targets.py` | **new** | Paper-faithful single-fixed-red-sphere Blender generator + `blender_targets.npy` builder (finding #1). |
| `evals/analysis_vlm/eval.py` | **new** | The regression probing driver; reads `data.uniform_sampling`, per-column target standardization (`:201-203`), NaN-masked R². Also holds the additive analysis-mode dispatch (`:566-588`). |
| `evals/analysis_vlm/modes/` | **new** | Phase-0 dispatch scaffold + Phase-1 `attention_distance` (finding #5). |
| `configs/analysis/blender_toy_dataset/*.yaml` | **new** | The reproduction configs (`frame_step: 1` + `uniform_sampling: true`). |

Everything is routed via `eval_name: analysis_vlm`; `main.py`/`scaffold.py`/existing evals
are untouched. Removing the fork's config knobs reproduces upstream sampling byte-for-byte.

---

## (a) What reproduced — Fig. 2c layer-wise R² dissociation

Run (single-GPU, frozen V-JEPA 2-L, `vit_large`, d=1024, 24 blocks):

```bash
python -m evals.main --fname configs/analysis/blender_toy_dataset/vjepa_combined.yaml \
    --devices cuda:0 --debugmode True
```

One combined run trains a per-layer **linear-mean** probe (`pre_norm: true`) for each of
three variables against `blender_targets.npy` `(672,4) = [speed, sinθ, cosθ, accel_mag]`,
with per-variable NaN-masking (speed defined on the 392 constant-velocity clips, accel_mag
on the 280 constant-acceleration clips, direction on all 672).

**Measured best-val R² by layer** (from
`configs/analysis/blender_toy_dataset/logs/analysis_vlm/vjepa-blender-polar_regression/summary.json`):

| Variable | L0 | L2 | L4 | L6 | **L7** | **L8** | L10 | L12 | L23 | best |
|----------|----|----|----|----|----|----|----|----|----|------|
| **speed** | **0.68** | 0.74 | 0.81 | 0.89 | 0.89 | 0.91 | 0.92 | 0.92 | 0.96 | 0.96 (L23) |
| **direction** | **0.28** | 0.43 | 0.51 | 0.68 | **0.68** | **0.98** | 0.94 | 0.97 | 0.97 | 0.98 (L11) |
| **accel_mag** | **0.53** | 0.58 | 0.64 | 0.75 | 0.76 | 0.88 | 0.79 | 0.85 | 0.92 | 0.92 (L23) |

This is the paper's dissociation:

- **Speed is decodable from block 0** (R² ≈ 0.68) and only creeps up with depth.
- **Direction is near-chance early** (R² ≈ 0.28 at L0) and **jumps sharply at the PEZ**:
  L7 → L8 goes 0.68 → 0.98, i.e. layer fraction 8/23 ≈ 0.35, squarely inside the shaded
  PEZ band `plot_pez: [0.2, 0.4]`.
- **Acceleration magnitude is intermediate** (0.53 at L0), tracking neither the flat-early
  speed curve nor the late-emerging direction curve.

**Plots produced:** `stage_val_acc.png` in the log dir — per-layer R² curves for all three
variables on one axis, with the PEZ shaded (plotting seam: `z_tech/07-plotting.md`). The
same recipe has been run for the LLaVA-Video backend (`llavavideo_combined.yaml`,
`llavavideo-blender-combined/`).

> Caveat on absolute numbers: R² magnitudes are dataset-dependent (this Blender set is
> ~4-way-ish directions, not the paper's exact synthetic-ball set). The **qualitative
> ordering + the sharp direction jump at ~one-third depth** is the reproduction criterion,
> and that holds.

---

## (b) The two root-cause debugging wins

Both failures produced a *flat / non-dissociating* speed curve — direction looked fine but
speed and accel were not decodable early, breaking the Fig. 2c signature. Two independent
root causes:

### 1. Random-appearance toy generator → paper-faithful single sphere

An earlier "anti-shortcut" toy generator randomized shape / color / size per clip (intended
to prevent the probe from cheating on appearance). It **did not reproduce early-speed**: with
the object's identity changing every clip, the encoder's early layers had no stable
retinotopic substrate to read instantaneous motion from. **Fix:** the paper-faithful
generator (`data_gen/make_physics_blender.py`) renders a **single fixed red sphere**
(r = 0.3 m, 8 m floor, overhead cam at (0,0,10), 16 f @ 24 fps, 256²) — matching the paper's
Kubric-style setup. With a constant appearance, layer-0 speed becomes decodable (R² ≈ 0.68
above). See `z_tech/09-blender-toy-dataset.md`.

### 2. `frame_step` contiguous-window sampling bug → `frame_step` / `uniform_sampling`

`VideoDataset.loadvideo_decord` (`src/datasets/video_dataset.py:293`) with the stock
`frame_step` path samples a **contiguous window** of `clip_len = fpc * frame_step` frames.
On a 64-frame clip with `frame_step=1`, `fpc=16`: `clip_len = 16`, `partition_len = 64`, so
`partition_len > clip_len` takes the first branch and returns
`indices = linspace(0, 16, 16) → 0..15` — the **first 1/4 of the trajectory**. The object
then moves only a **sub-patch distance per tubelet**, so layer-0 tokens carry essentially no
motion and speed/accel are undecodable early.

**Two fixes, both in-tree:**

- `frame_step=4`: `clip_len = fpc*4 = 64` spans the whole 64-frame clip (stride-4 sampling
  covers the full trajectory).
- **`uniform_sampling: true`** (this fork's additive knob): evenly sample `fpc` frames over
  the *entire* video, length-agnostic — `video_dataset.py:337-341`:

  ```python
  if getattr(self, "uniform_sampling", False):
      n = len(vr)
      indices = np.clip(np.linspace(0, n - 1, num=fpc).round(), 0, n - 1).astype(np.int64)
      buffer = vr.get_batch(list(indices)).asnumpy()
      return buffer, [indices]
  ```

The **Blender clips are natively 16 frames**, so their configs use `frame_step: 1` +
`uniform_sampling: true` (a 16-frame `linspace(0,15,16)` = every frame), which is robust to
clip length and sidesteps the contiguous-window branch entirely. See
`z_tech/06-data-pipeline-changes.md`.

---

## (c) Linear-probe normalization (`pre_norm`) is required — correctness finding

`LinearProbe(pre_norm=True)` (`evals/analysis/probes.py:34-39`) applies
`nn.LayerNorm(in_dim)` over the feature dim **per sample** before the linear layer:

```python
self.norm = nn.LayerNorm(in_dim) if pre_norm else nn.Identity()
self.linear = nn.Linear(in_dim, num_classes, bias=True)
...
return self.linear(self.norm(self._pool(x)))
```

(Same for the pooled-cache `PooledLinearProbe`, `evals/analysis_vlm/cache.py:196-214`, and
the framewise `TemporalLinearProbe`, `evals/analysis_vlm/probes.py:84-90`.)

**Why it is required for a *cross-layer* scan.** V-JEPA activation scale differs by **orders
of magnitude across the 24 layers**. A single fixed learning rate (the analysis harness
sweeps one probe config across all layers) cannot fit both a small-norm early layer and a
large-norm late layer without input normalization — the effective step size is wrong for
most layers. Per-sample LayerNorm rescales every layer's pooled feature to unit-ish scale so
one lr fits all of them. A competitor fork set **`pre_norm=False` (raw pooled features) + a
fixed lr** and reported poor linear-probe accuracy for exactly this reason.

**Valid alternatives** (the choice is normalize-or-fail, not LayerNorm-specifically):

| Normalization | Semantics | OK? |
|---------------|-----------|-----|
| `nn.LayerNorm` over D (this fork) | per-sample, over feature dim | ✅ |
| per-feature `StandardScaler` | per-feature, over the dataset | ✅ (equivalent goal, fit on train) |
| none (`pre_norm=False`) + fixed lr | raw pooled features | ❌ — the wrong choice |

R² itself is scale-invariant; the standardization at `eval.py:201-203` is on the *targets*
and independent of `pre_norm` on the *inputs*. Both matter for different reasons — target
standardization keeps MSE/lr well-scaled across variables (pixels vs sin/cos), input
`pre_norm` keeps them well-scaled across layers.

---

## (d) Roadmap for the remaining paper experiments

The remaining paper analyses are being added as **additive, config-driven modes** under
`experiment.analysis.modes` (full design in
`evals/analysis_vlm/modes/REPRODUCTION_PLAN.md`; dispatch in `z_tech/12-analysis-modes.md`).
**Invariant / default-off guarantee:** the key is absent from every existing config, so
`modes_cfg == {}` (`evals/analysis_vlm/eval.py:568`) ⇒ the whole dispatch block is skipped ⇒
`summary.json` / `log_r*.csv` / `stage_val_acc.png` are byte-identical to a pre-change run.
`skip_base_probe` defaults `False` (`eval.py:504`) ⇒ the base probe loop is unchanged.

| Phase | Mode | Paper ref | Status | Blender run |
|-------|------|-----------|--------|-------------|
| 0 | dispatch scaffold (`modes/__init__.py`, empty registry, default-off) | — | **DONE** | proven byte-identical with no `modes:` key |
| 1 | `attention_distance` | C.6 / Fig. 19 / Fig. 3 | **DONE** (self-registers, `attention_distance.py`) | `vjepa_attn_distance.yaml` |
| 2 | `orthogonal_probe_sequence` | C.11 / Fig. 22 / Table 3 | pending | `vjepa_ortho_probe.yaml` (`cache_pooling: tokens`) |
| 3 | `steering` | C.12 / Fig. 24 | pending | `vjepa_steering.yaml` (layer 8, `tokens`) |
| 4 | `direction_tuning` (circular geometry) | C.7/C.8/C.10 / Fig. 20/21/23 | pending | `vjepa_direction_tuning.yaml` (`tokens`, stages [0,8,12,23]) |
| 5 | `attention_ablation` | C.6 / Table 4 | pending | `vjepa_attn_ablation.yaml` (`cache_features: false`) |

**How each runs on Blender** (all: single-GPU, frozen V-JEPA 2-L, `velocity_*.csv` for
direction/speed):

```bash
python -m evals.main --fname configs/analysis/blender_toy_dataset/<mode>.yaml \
    --devices cuda:0 --debugmode True
# writes <folder>/<mode>/*.json + *.png
```

- **attention_distance** — plots per-layer mean attention distance `D̄⁽ˡ⁾` and head
  specialization `S⁽ˡ⁾` (patch units) vs layer. Pass criterion: `D̄` dips + `S` spikes in the
  PEZ (~L5-8), not the absolute values.
- **orthogonal_probe_sequence** — deflation loop counting orthogonal direction/speed probes
  before decode hits chance. Pass: `K(direction) ≫ K(speed)` at every layer.
- **steering** — 70/30 split, least-squares coordinates in the K-probe subspace, held-out
  eval probe. Pass: MAE-to-target falls monotonically as N grows while MAE-to-true-label
  rises (the curves cross).
- **direction_tuning** — per-neuron sin/cos GLM. Pass: Layer-0 tuning vectors
  sporadic/short vs a dense 360°-tiling fan at L8; direction "sawtooth" redundancy vs smooth
  speed decay.
- **attention_ablation** — re-extract features under a spatial/temporal attention mask, then
  re-decode the frozen direction head. Pass: spatial-only spares global direction but kills
  per-patch R²; combined collapses direction even at the mildest (s=3, t=1).

---

## Config

The reproduction config (`configs/analysis/blender_toy_dataset/vjepa_combined.yaml`), trimmed
to the load-bearing keys:

```yaml
eval_name: analysis_vlm
tag: vjepa-blender-polar_regression

experiment:
  analysis:
    model: vjepa
    task: regression
    regression:
      targets_npy: .../data_csv/blender_toy/blender_targets.npy   # (672,4)=[speed,sinθ,cosθ,accel_mag], NaN=undefined
      variables:
        - {name: speed,      cols: [0]}     # velocity clips only (accel clips NaN -> masked)
        - {name: direction,  cols: [1, 2]}  # all clips
        - {name: accel_mag,  cols: [3]}     # accel clips only
    stages: {vision_encoder: all}           # layer-wise scan, all 24 blocks
    plot: true
    plot_pez: [0.2, 0.4]                     # PEZ shading (layer fraction); remove/false to disable
    probes:
      - {type: linear, pooling: mean, pre_norm: true,   # pre_norm REQUIRED (finding c)
         optimization: {lr: 0.001, weight_decay: 0.1, warmup: 2.0}}

  data:
    dataset_type: VideoDataset
    resolution: 256
    resize_mode: resize
    frame_step: 1                # blender clips are native 16f
    uniform_sampling: true       # 16 evenly-sampled frames = every frame (finding b#2)
    num_segments: 1
    frames_per_clip: 16          # paper inference = 16f @ 24fps
    dataset_train: .../data_csv/blender_toy/combined_train.csv
    dataset_val:   .../data_csv/blender_toy/combined_val.csv

  optimization:
    batch_size: 8
    num_epochs: 40
    use_bfloat16: true
    cache_features: true
    cache_pooling: pooled        # [mean‖max]; fine for linear-mean probes (finding #4)
    cache_max_gb: 80
```

---

## Gotchas / invariants

- **`cache_pooling` degrades direction if you go token-blind.** `pooled` = `[mean‖max]`
  over tokens (2D, **collapses the time axis** → degrades direction), `tokens` = full
  `(N,D)`, `framewise` = `(T,D)` (VLM-only, needs `num_temporal`). The Fig. 2c linear-mean
  scan tolerates `pooled` because it uses a global mean anyway, but **all token-level
  reproduction modes (ortho probe / steering / direction tuning) require
  `cache_pooling: tokens`** — see `z_tech/03-feature-caching-and-pooling.md`.
- **`attention_ablation` must set `cache_features: false`** — ablation changes the features,
  so a cache built from the un-ablated forward would silently reuse baseline features.
- **`pre_norm: false` is the one wrong probe choice** (finding c). LayerNorm-over-D or a
  per-feature StandardScaler both work; no normalization does not.
- **Frame sampling is the silent Fig. 2c killer** (finding b#2). A flat early-speed curve is
  almost always a sampling problem (contiguous sub-window), not a model problem — check
  `frame_step` / `uniform_sampling` before anything else.
- **Default-off is a hard invariant.** No `modes:` key ⇒ zero behavior change; the analysis
  modes never run for existing configs.
- **Absolute R²/MAE numbers are dataset-dependent.** The reproduction criteria are the
  qualitative signatures (early-speed, sharp direction jump at ~one-third depth, direction ≫
  speed subspace dim, monotone steering), not the paper's exact values on its own synthetic
  set.
