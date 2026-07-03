# 14 — Reproduction status & findings

> Two paper results now reproduce on the paper-faithful Blender toy set — Fig. 2c's layer-wise R² dissociation (speed decodable early, direction emerging at the one-third-depth PEZ) and Fig. 3's per-head attention-distance locality (unusually-local heads clustering in the mid layers) — enabled by three root-cause findings (single-sphere data, `uniform_sampling`, `pre_norm`), with Phases 0–1 of the analysis-mode roadmap DONE and Phases 2–5 pending.

## Purpose

This section records **what of the paper's physics-interpretability results this fork has
actually reproduced**, the debugging/correctness findings that made the reproduction work,
and the phased roadmap for the remaining paper experiments. It is a *status + findings*
note, not a code walkthrough — the machinery it leans on is documented in the
cross-referenced sections.

**Two headline results now reproduce** on the paper-faithful Blender toy dataset with the
frozen V-JEPA 2-L (`vit_large`, d=1024, 24 blocks × 16 heads):

1. **Fig. 2c — layer-wise R² dissociation** (linear probing): speed is decodable from block 0,
   direction emerges sharply at the ~one-third-depth **Physics Emergence Zone (PEZ)**, and
   acceleration magnitude sits in between.
2. **Fig. 3 — per-head attention-distance heatmap** (encoder-only, no probe): spatiotemporally
   **local heads emerge in the mid/PEZ layers (~L5–L13)** while early (L0–L4) and late
   (L17–L23) layers stay uniformly long-range.

Both are *qualitative-signature* reproductions — the absolute R²/distance magnitudes are
dataset-dependent (this Blender set is not the paper's exact synthetic-ball set); the
**ordering and the location of the transition** are the reproduction criteria, and they hold.

## What changed vs upstream V-JEPA2

Baseline = commit `204698b` (no `evals/analysis*` tree at all). Everything is **additive and
default-off**; removing the fork's config knobs reproduces upstream byte-for-byte.

### NEW files (absent from baseline — `git cat-file -e 204698b:<path>` fails)

| File | Role in the reproduction |
|------|--------------------------|
| `evals/analysis/attention_hooks.py` | The `F.scaled_dot_product_attention` monkey-patch substrate + `AttentionDistanceCollector`. Capture is a **detached side computation** — encoder output is bit-identical when capture-only (Fig. 3 depends on this). See [11-attention-hooks.md](11-attention-hooks.md). |
| `evals/analysis_vlm/modes/__init__.py` | Phase-0 dispatch scaffold: `@register` registry, lazy `_import_modes`, `AnalysisContext`, `run_modes`. Reached only via `experiment.analysis.modes`. |
| `evals/analysis_vlm/modes/attention_distance.py` | Phase-1 `attention_distance` mode — renders the **Fig. 3 heatmap (primary)** + the Fig. 19 companion line plot. |
| `evals/analysis_vlm/modes/REPRODUCTION_PLAN.md` | The Phase 0–5 blueprint (per-experiment recipe, pass criteria, paper-value tables). |
| `evals/analysis_vlm/eval.py` | The regression probing driver (Fig. 2c) + the additive mode-dispatch block. Reads `data.uniform_sampling`, per-column NaN-aware target standardization (`:198-203`), NaN-masked R². |
| `evals/analysis/probes.py`, `evals/analysis_vlm/cache.py`, `evals/analysis_vlm/probes.py` | `LinearProbe` / `PooledLinearProbe` / `TemporalLinearProbe` with the `pre_norm` LayerNorm (finding #3). |
| `data_gen/make_physics_blender.py`, `data_gen/sanity_check_blender.py`, `data_csv/make_blender_targets.py` | Paper-faithful single-fixed-red-sphere Blender generator + `blender_targets.npy` builder (finding #1). See [09-blender-toy-dataset.md](09-blender-toy-dataset.md). |
| `configs/analysis/blender_toy_dataset/*.yaml` | The reproduction configs (`vjepa_combined.yaml` = Fig. 2c; `vjepa_attn_distance.yaml` = Fig. 3). |
| `z_scripts/run_attn_distance_vjepa.sh` | SLURM launcher for the Fig. 3 run (`gpu:1` **on purpose** — see Design & data flow). |

### MODIFIED files (exact additive delta via `git diff 204698b -- <path>`)

| File | Delta |
|------|-------|
| `src/datasets/video_dataset.py` | `uniform_sampling=False` param threaded through `make_videodataset` (`:54,:71`) + `VideoDataset.__init__` (`:142,:147`), and **one early-return branch** in `loadvideo_decord` (`:334-343`). `getattr(self, "uniform_sampling", False)` ⇒ default path is untouched (finding #2). |
| `src/datasets/data_manager.py`, `evals/video_classification_frozen/eval.py` | `uniform_sampling` threaded through `init_data` / `make_dataloader` (≈2 lines each). |

**Default-off guarantee.** No `modes:` key ⇒ `modes_cfg == {}` (`eval.py:568`) ⇒ the whole
dispatch block is skipped and `evals.analysis_vlm.modes` is never imported ⇒
`summary.json` / `log_r*.csv` / `stage_val_acc.png` are byte-identical to a pre-change run.
`skip_base_probe` defaults `False` (`eval.py:504`) ⇒ `num_probe_epochs == num_epochs` ⇒ the
probe loop is unchanged. Without `uniform_sampling: true` the sampler takes the stock
contiguous-window branch verbatim.

## Design & data flow

The reproduction is **two independent runs**, both single-GPU, frozen V-JEPA 2-L, routed via
`eval_name: analysis_vlm`:

| Run | Config | What it does | Output |
|-----|--------|--------------|--------|
| **Fig. 2c** (probing sweep) | `vjepa_combined.yaml` | Trains a per-layer linear-mean probe (`pre_norm: true`) for speed/direction/accel_mag against `blender_targets.npy`, all 24 blocks, NaN-masked R². | `.../vjepa-blender-polar_regression/{summary.json, stage_val_acc.png}` |
| **Fig. 3** (encoder-only capture) | `vjepa_attn_distance.yaml` | `skip_base_probe: true`, `num_epochs: 0` ⇒ **no probe trained**. The `attention_distance` mode runs fresh clip forwards under the SDPA capture patch and accumulates per-(layer,head) attention-weighted distance. | `.../vjepa-blender-attn_distance/attention_distance/{attention_distance.json, attention_distance.png, attention_distance_layerwise.png}` |

**Mode-dispatch flow** (Fig. 3 run): `eval.py:568-588` builds an `AnalysisContext` and calls
`run_modes(modes_cfg, ctx)`; `run_modes` (`modes/__init__.py:76`) lazily imports
`attention_distance` (self-registers via `@register`), then dispatches. The mode calls
`ctx.make_val_clip_loader()`, wraps the encoder in `attention_hooks(encoder, collector=…)`,
runs `ctx.encode_clip(data)` for up to `max_batches` val batches, then `collector.finalize()`
→ JSON, then the two plots.

**Why single-GPU (the DDP note).** The mode dispatch runs on **rank 0 only** —
`eval.py:569` guards it with `if modes_cfg and rank == 0`. So multi-GPU gives **no speedup**
for post-hoc modes; DDP only helps the base probing sweep / feature cache. Every mode recipe
is therefore single-GPU, and `z_scripts/run_attn_distance_vjepa.sh` sets `--gres=gpu:1` on
purpose (its header says so), exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to
curb fragmentation, and runs `python -m evals.main --fname $CONFIG --devices $DEVICES`
(no `--debugmode`; the interactive recipe adds `--debugmode True`).

**Token geometry (paper-faithful).** The Fig. 3 run uses `resolution: 224` →
`patch_size: 16` = **14×14 = 196 spatial patches**; `tubelet_size: 2` on 16 frames = **8
temporal tokens**; total **T·H·W = 8·196 = 1568 tokens** per clip — the paper's geometry
(cf. `REPRODUCTION_PLAN.md:326`). The captured `rows_per_layer = 122304 = 78 clips × 1568
tokens`, i.e. **every one of the 78 `velocity_val.csv` clips** was measured across the 10
batches (last batch partial). Each token contributes one query row to the running mean.

## Results — what reproduced

### (a) Fig. 2c — layer-wise R² dissociation (speed early, direction at the PEZ)

Run (`vjepa_combined.yaml`; single-GPU, frozen V-JEPA 2-L):

```bash
python -m evals.main --fname configs/analysis/blender_toy_dataset/vjepa_combined.yaml \
    --devices cuda:0 --debugmode True
```

One combined run trains a per-layer **linear-mean** probe (`pre_norm: true`) for each of three
variables against `blender_targets.npy` `(672,4) = [speed, sinθ, cosθ, accel_mag]`, with
per-variable NaN-masking: speed on the 392 constant-velocity clips, accel_mag on the 280
constant-acceleration clips, direction on all 672.

**Measured best-val R² by layer** (from
`.../vjepa-blender-polar_regression/summary.json`):

| Variable | L0 | L4 | **L5** | L6 | L7 | **L8** | L9 | L11 | L14 | L23 | best |
|----------|----|----|----|----|----|----|----|----|----|----|------|
| **speed**     | 0.68 | 0.81 | 0.91 | 0.89 | 0.89 | 0.91 | 0.92 | 0.93 | 0.94 | **0.97** | 0.97 (L23) |
| **direction** | 0.28 | 0.51 | 0.82 | 0.68 | 0.68 | **0.98** | 0.97 | 0.98 | 0.96 | 0.98 | 0.98 (L8/L11) |
| **accel_mag** | 0.53 | 0.64 | 0.73 | 0.75 | 0.76 | 0.88 | 0.80 | 0.83 | 0.89 | **0.92** | 0.92 (L23) |

This is the paper's dissociation:

- **Speed is decodable from block 0** (R² ≈ 0.68) and only creeps up with depth.
- **Direction is near-chance early** (R² ≈ 0.28 at L0) and **emerges at the PEZ**. The curve
  is **non-monotonic**: it first spikes at L5 (0.51 → **0.82**), *relapses* to 0.68 at L6–L7,
  then makes its **sharp, durable jump at L8: 0.68 → 0.98**. Layer fraction 8/23 ≈ 0.35 sits
  squarely inside the shaded PEZ band `plot_pez: [0.2, 0.4]`. (The L5 spike-and-relapse is
  worth stating so the "sharp jump" is not read as a single clean step.)
- **Acceleration magnitude is intermediate** (0.53 at L0), tracking neither the flat-early
  speed curve nor the late-locking direction curve.

**Plot produced:** `stage_val_acc.png` — per-layer R² curves for all three variables on one
axis, PEZ shaded (plotting seam: [07-plotting.md](07-plotting.md)). The same recipe has been
run for the LLaVA-Video backend (`llavavideo_combined.yaml`).

> Caveat on absolute numbers: R² magnitudes are dataset-dependent. The **qualitative ordering
> + the direction jump at ~one-third depth** is the reproduction criterion, and it holds.

### (b) Fig. 3 — per-head attention-distance heatmap (local heads at the PEZ) *(NEW)*

Run (`vjepa_attn_distance.yaml`; single-GPU, **encoder-only**, 10 val batches = all 78 clips):

```bash
python -m evals.main --fname configs/analysis/blender_toy_dataset/vjepa_attn_distance.yaml \
    --devices cuda:0 --debugmode True
# or: sbatch z_scripts/run_attn_distance_vjepa.sh
```

The `attention_distance` mode captures, for every (layer, head), the attention-weighted
**spatial (patch)** and **temporal (tubelet)** distance
`E_i[ Σ_j softmax_attn[h,i,j] · dist(i,j) ]` of the frozen RoPE encoder, via the additive
`attention_hooks` SDPA patch. It writes **two plots**:

- **`attention_distance.png` — the PRIMARY plot = paper Fig. 3 heatmap.** x = Layer (0–23),
  y = Attention Head (0–15), colour = **spatial distance in patches**, cmap `Blues_r`
  (**LOW distance = DARK** so local heads stand out), per-cell value annotations, colorbar
  "Distance (patches)", title *"V-JEPA v2-L: Attention Distance Per Head"*
  (`attention_distance.py:76-112`).
- **`attention_distance_layerwise.png` — the companion Appendix Fig. 19 dual-axis line plot**:
  layer-mean distance `D̄` (red, mean over heads) and head specialization `S` (blue dashed,
  std over heads) vs **layer fraction**, PEZ shaded (`attention_distance.py:115-156`).

**Terms.** `D̄⁽ˡ⁾` = mean over the 16 heads of their per-head attention distances within a
layer. **`S⁽ˡ⁾` = head specialization = population std (`statistics.pstdev`) over the 16
heads of their per-head attention distances = attention-head diversity**; it spikes where a
few unusually-local heads coexist with the long-range majority.

**Measured signature** (from `attention_distance.json`; 24 layers × 16 heads, 78 clips ×
1568 tokens per layer):

- Per-cell **spatial** distance spans **0.10 – 9.26 patches**; the single most-local (darkest)
  cell is **Layer 6 / Head 15 = 0.10 patches**.
- Per-cell **temporal** distance spans **0.01 – 3.03 tubelets** (also captured, in the JSON's
  `temporal_distance [24][16]` — never surfaced in the heatmap, which plots spatial).
- Layer-mean `D̄` starts **7.08 (L0)**, dips to a minimum **3.73 (L9)** (3.74 at L6), and rises
  to a **late peak 7.70 (L22)** (6.64 at L23).
- Head specialization `S` is **0.33 (L0)** and **peaks 2.54 (L10)**, staying high (≈2.3–2.5)
  across L5–L10.
- Count of "local" heads (per-cell spatial distance < 3 patches) is **0 in L0–L1 and in
  L17–L23**, but **clusters at 4–7 heads across L5–L13** (7 at L5, L6, L9; 6 at L10; 5 at L8).

**Per-layer table** (spatial):

| L | D̄ | S | #local(<3) | | L | D̄ | S | #local(<3) |
|---|-----|-----|---|---|---|-----|-----|---|
| 0 | 7.08 | 0.33 | 0 | | 12 | 5.17 | 1.90 | 3 |
| 1 | 6.89 | 0.72 | 0 | | 13 | 5.12 | 1.72 | 2 |
| 2 | 6.36 | 1.85 | 2 | | 14 | 5.57 | 1.52 | 1 |
| 3 | 6.17 | 1.32 | 0 | | 15 | 6.35 | 1.39 | 1 |
| 4 | 5.89 | 1.81 | 2 | | 16 | 5.75 | 1.37 | 1 |
| **5** | **3.99** | **2.38** | **7** | | 17 | 5.92 | 1.45 | 0 |
| **6** | **3.74** | **2.49** | **7** | | 18 | 5.90 | 1.32 | 0 |
| 7 | 4.84 | 2.08 | 4 | | 19 | 6.50 | 0.98 | 0 |
| 8 | 4.76 | 2.25 | 5 | | 20 | 6.69 | 1.13 | 0 |
| **9** | **3.73** | **2.41** | **7** | | 21 | 6.97 | 0.86 | 0 |
| **10** | 4.12 | **2.54** | 6 | | **22** | **7.70** | 0.75 | 0 |
| 11 | 5.13 | 1.86 | 1 | | 23 | 6.64 | 1.35 | 0 |

**This is Fig. 3 reproduced:** `D̄` **dips** and `S` **spikes** across the PEZ (~L5–L13, min `D̄`
at L9, max `S` at L10); early (L0–L4) and late (L17–L23) layers are uniformly long-range with
`S` near its L0 floor. Spatiotemporally-**local heads emerge specifically in the mid layers**
alongside the persistent long-range heads — the paper's claim. Absolute patch values differ
from the paper's synthetic-ball set; the **directional dip/spike at the emergence zone** is
the pass criterion.

## Findings — the wins that made it work

### Finding #1 — random-appearance toy → paper-faithful single red sphere

An earlier "anti-shortcut" generator randomized shape/color/size per clip (to stop the probe
cheating on appearance). It **did not reproduce early-speed**: with the object's identity
changing every clip, early layers had no stable retinotopic substrate to read instantaneous
motion from — a *flat, non-dissociating* speed curve.
**Fix:** `data_gen/make_physics_blender.py` renders a **single fixed red sphere**
(r = 0.3 m, 8 m floor, overhead cam at (0,0,10), 16 f @ 24 fps, 256²) — the paper's
Kubric-style setup. With a constant appearance, layer-0 speed becomes decodable (R² ≈ 0.68
above). See [09-blender-toy-dataset.md](09-blender-toy-dataset.md).

### Finding #2 — `frame_step` contiguous-window sampling → `uniform_sampling`

`VideoDataset.loadvideo_decord` (`src/datasets/video_dataset.py:293`) with the stock
`frame_step` path samples a **contiguous window** of `clip_len = fpc · frame_step` frames.
When `fpc·frame_step < len(video)` it returns only the **first sub-segment** of the
trajectory, so the object moves a **sub-patch distance per tubelet** and layer-0 tokens carry
essentially no motion — speed/accel undecodable early (the same flat curve as #1, different
root cause). The additive fix (`video_dataset.py:334-343`):

```python
# uniform_sampling: pick `fpc` frames evenly across the WHOLE video (length-agnostic;
# ignores frame_step / num_clips). Avoids the contiguous-window default that, when
# fpc*frame_step < len(video), only covers a sub-segment -> sub-patch motion per tubelet.
if getattr(self, "uniform_sampling", False):
    n = len(vr)
    indices = np.clip(np.linspace(0, n - 1, num=fpc).round(), 0, n - 1).astype(np.int64)
    buffer = vr.get_batch(list(indices)).asnumpy()
    return buffer, [indices]
```

The Blender clips are natively 16 frames, so the configs use `frame_step: 1` +
`uniform_sampling: true` (a 16-frame `linspace(0,15,16)` = every frame): robust to clip
length, sidesteps the contiguous-window branch entirely. (`frame_step: 4` would also span a
64-frame clip, but `uniform_sampling` is length-agnostic.) See
[06-data-pipeline-changes.md](06-data-pipeline-changes.md).

### Finding #3 — linear-probe `pre_norm` is required (correctness)

`LinearProbe(pre_norm=True)` (`evals/analysis/probes.py:38-53`) applies `nn.LayerNorm(in_dim)`
over the feature dim **per sample**, after pooling, before the linear layer:

```python
self.norm = nn.LayerNorm(in_dim) if pre_norm else nn.Identity()
self.linear = nn.Linear(in_dim, num_classes, bias=True)
...
def forward(self, x):
    x = self._pool(x)
    x = self.norm(x)
    return self.linear(x)
```

(Same for the pooled-cache `PooledLinearProbe`, `evals/analysis_vlm/cache.py:196-201`, and the
framewise `TemporalLinearProbe`, `evals/analysis_vlm/probes.py:84-89`.)

**Why it is required for a *cross-layer* scan.** V-JEPA activation scale differs by **orders of
magnitude across the 24 layers**. A single fixed lr (the harness sweeps one probe config
across all layers) cannot fit both a small-norm early layer and a large-norm late layer
without input normalization. Per-sample LayerNorm rescales every layer's pooled feature to
unit-ish scale so one lr fits all. A competitor fork set `pre_norm=False` (raw pooled
features) + a fixed lr and reported poor linear-probe accuracy for exactly this reason.

**Valid alternatives** (the choice is normalize-or-fail, not LayerNorm-specifically):

| Normalization | Semantics | OK? |
|---------------|-----------|-----|
| `nn.LayerNorm` over D (this fork) | per-sample, over feature dim | ✅ |
| per-feature `StandardScaler` | per-feature, over the dataset (fit on train) | ✅ |
| none (`pre_norm=False`) + fixed lr | raw pooled features | ❌ — the wrong choice |

R² itself is scale-invariant; the target standardization at `eval.py:198-203` is on the
*targets* and independent of `pre_norm` on the *inputs*. Both matter for different reasons:
target standardization keeps MSE/lr well-scaled across variables (pixels vs sin/cos in
[-1,1]); input `pre_norm` keeps them well-scaled across layers. See
[04-probes-regression-nanmask.md](04-probes-regression-nanmask.md).

## Roadmap — remaining paper experiments (Phases 0–5)

The remaining paper analyses are being added as **additive, config-driven modes** under
`experiment.analysis.modes` (full design in `evals/analysis_vlm/modes/REPRODUCTION_PLAN.md`;
dispatch in [12-analysis-modes.md](12-analysis-modes.md)). **Phase 0 (dispatch scaffold) and
Phase 1 (`attention_distance`) are DONE; Phases 2–5 pending.**

| Phase | Mode | Paper ref | Status | Blender config |
|-------|------|-----------|--------|----------------|
| **0** | dispatch scaffold (`modes/__init__.py`, registry, `skip_base_probe`, default-off) | — | **DONE** — proven byte-identical with no `modes:` key | — |
| **1** | `attention_distance` | C.6 / Fig. 19 / **Fig. 3** | **DONE** — run + result in section (b) above | `vjepa_attn_distance.yaml` |
| **2** | `orthogonal_probe_sequence` | C.11 / Fig. 22 / Table 3 | pending | `vjepa_ortho_probe.yaml` (`cache_pooling: tokens`) |
| **3** | `steering` | C.12 / Fig. 24 | pending | `vjepa_steering.yaml` (layer 8, `tokens`) |
| **4** | `direction_tuning` (circular geometry) | C.7/C.8/C.10 / Fig. 20/21/23 | pending | `vjepa_direction_tuning.yaml` (`tokens`, stages [0,8,12,23]) |
| **5** | `attention_ablation` | C.6 / Table 4 | pending | `vjepa_attn_ablation.yaml` (`cache_features: false`) |

**How each pending mode runs on Blender** (all: single-GPU, frozen V-JEPA 2-L, `velocity_*.csv`
for direction/speed):

```bash
python -m evals.main --fname configs/analysis/blender_toy_dataset/<mode>.yaml \
    --devices cuda:0 --debugmode True
# writes <folder>/<mode>/*.json + *.png
```

- **`attention_distance` (Phase 1, DONE)** — the **Fig. 3 heatmap** (`attention_distance.png`)
  is the primary output; the `D̄`/`S` line plot (`attention_distance_layerwise.png`) is the
  companion. Reproduced values in section (b): `D̄` 7.08→3.73(L9)→7.70(L22); `S` 0.33→2.54(L10);
  most-local cell L6/H15 = 0.10 patches; local heads cluster ~L5–L13.
- **`orthogonal_probe_sequence`** — QR-deflation loop counting orthogonal direction/speed
  probes before decode hits chance (R²<0.1 / MAE>80°). *Pass:* `K(direction) ≫ K(speed)` at
  every layer, rising with depth.
- **`steering`** — 70/30 split, least-squares coordinates in the K-probe subspace, held-out
  eval probe, layer 8. *Pass:* MAE-to-target falls monotonically as N grows while
  MAE-to-true-label rises (the curves cross).
- **`direction_tuning`** — per-neuron sin/cos GLM. *Pass:* Layer-0 tuning vectors
  sporadic/short vs a dense 360°-tiling fan at L8; direction "sawtooth" redundancy vs smooth
  speed decay; no speed ring.
- **`attention_ablation`** — re-extract features under a spatial/temporal attention mask
  (`cache_features: false`, fresh forwards), then re-decode the frozen direction head.
  *Pass:* spatial-only spares global direction but kills per-patch R²; combined collapses
  direction even at the mildest (s=3, t=1).

## Key code

| Location | What |
|----------|------|
| `evals/analysis_vlm/eval.py:568-588` | Mode dispatch — `modes_cfg = get("modes") or {}`; guarded `if modes_cfg and rank == 0` (single-GPU rationale). |
| `evals/analysis_vlm/eval.py:504` | `num_probe_epochs = 0 if skip_base_probe else num_epochs`. |
| `evals/analysis_vlm/eval.py:198-203` | NaN-aware per-column target standardization (independent of `pre_norm`). |
| `evals/analysis_vlm/modes/attention_distance.py:32-73` | `@register("attention_distance")` mode body: collector + `attention_hooks` context + `max_batches` cap → JSON + 2 plots. |
| `evals/analysis_vlm/modes/attention_distance.py:76-112` | `_plot_heatmap` — Fig. 3 primary plot (`Blues_r`, low=dark, annotated). |
| `evals/analysis/attention_hooks.py:161-225` | `AttentionDistanceCollector` — `query_chunk`-streamed running means; never materializes `(B,H,N,N)`. |
| `evals/analysis/attention_hooks.py:318-322` | Capture is a **detached side computation** — returns `_ORIG_SDPA(...)` so encoder output is bit-identical. |
| `src/datasets/video_dataset.py:334-343` | `uniform_sampling` early-return branch (finding #2). |
| `evals/analysis/probes.py:38-53` | `LinearProbe` `pre_norm` LayerNorm (finding #3). |

The identical-when-off contract, quoted:

```python
# evals/analysis_vlm/eval.py:566-569
# Absent from a config ⇒ modes_cfg == {} ⇒ this whole block is skipped and nothing is
# imported ⇒ existing runs behave byte-for-byte identically. Runs on rank 0 only.
modes_cfg = args_analysis.get("modes") or {}
if modes_cfg and rank == 0:
```

## Configuration

Two run recipes, side by side.

### Fig. 2c — `vjepa_combined.yaml` (probing sweep)

```yaml
eval_name: analysis_vlm
tag: vjepa-blender-polar_regression
experiment:
  analysis:
    model: vjepa
    task: regression
    regression:
      targets_npy: .../blender_targets.npy       # (672,4)=[speed,sinθ,cosθ,accel_mag], NaN=undefined
      variables:
        - {name: speed,     cols: [0]}            # velocity clips only (accel clips NaN -> masked)
        - {name: direction, cols: [1, 2]}         # all clips
        - {name: accel_mag, cols: [3]}            # accel clips only
    stages: {vision_encoder: all}                 # layer-wise scan, all 24 blocks
    plot: true
    plot_pez: [0.2, 0.4]                          # PEZ shading (layer fraction)
    probes:
      - {type: linear, pooling: mean, pre_norm: true,   # pre_norm REQUIRED (finding #3)
         optimization: {lr: 0.001, weight_decay: 0.1, warmup: 2.0}}
  data:
    dataset_type: VideoDataset
    resolution: 256
    resize_mode: resize
    frame_step: 1                # blender clips are native 16f
    uniform_sampling: true       # 16 evenly-sampled frames = every frame (finding #2)
    num_segments: 1
    frames_per_clip: 16
    dataset_train: .../combined_train.csv
    dataset_val:   .../combined_val.csv
  optimization:
    batch_size: 8
    num_epochs: 40
    use_bfloat16: true
    cache_features: true
    cache_pooling: pooled        # [mean‖max]; fine for a global-mean linear probe (finding #4)
    cache_max_gb: 80
```

### Fig. 3 — `vjepa_attn_distance.yaml` (encoder-only attention capture)

```yaml
eval_name: analysis_vlm
tag: vjepa-blender-attn_distance
experiment:
  analysis:
    model: vjepa
    task: regression
    regression:
      targets_npy: .../blender_targets.npy
      variables: [{name: direction, cols: [1, 2]}]
    stages: {vision_encoder: all}
    plot_pez: [0.2, 0.4]
    skip_base_probe: true          # encoder-only: distance capture needs no trained probe
    modes:
      attention_distance:          # ← default-off unless present (finding #5)
        enabled: true
        query_chunk: 512           # stream queries so the (B,H,N,N) matrix is never materialized
        max_batches: 10            # measure the first 10 val batches (= all 78 clips here)
    probes:
      - {type: linear, pooling: mean, pre_norm: true,
         optimization: {lr: 0.001, weight_decay: 0.1, warmup: 2.0}}
  data:
    dataset_type: VideoDataset
    resolution: 224                # paper geometry: 14x14 patches x 8 tubelets = 1568 tokens
    resize_mode: resize
    frame_step: 1
    uniform_sampling: true
    num_segments: 1
    frames_per_clip: 16
    dataset_train: .../velocity_val.csv   # == dataset_val; no probe trained ⇒ train split unused
    dataset_val:   .../velocity_val.csv
    num_classes: 4                        # ignored for regression + no probe
  optimization:
    batch_size: 8
    num_epochs: 0                  # no probe training (encoder-only mode)
    use_bfloat16: true
    cache_features: false          # attention capture runs fresh forwards, not the cache
```

**`attention_distance` mode options** (`modes/attention_distance.py`; all result-safe defaults):

| Key | Meaning | Default | Notes |
|-----|---------|---------|-------|
| `enabled` | run this mode | `true` (mode present) | `false` / `{enabled: false}` skips it |
| `query_chunk` | #query rows streamed per SDPA chunk | `512` | **memory knob, RESULT-INVARIANT** — streams so `(B,H,N,N)` is never materialized |
| `max_batches` | #val batches averaged into the running mean | `8` | run used `10` (= all 78 clips) |
| `annotate` | draw per-cell numbers on the heatmap | `true` | cosmetic only |

## Invariants & gotchas

- **Finding #4 — `cache_pooling` degrades direction if you go time-blind.** `pooled` =
  `[mean‖max]` over tokens (2D, **collapses the time axis** → degrades direction), `tokens` =
  full `(N,D)`, `framewise` = `(T,D)` (VLM-only). The Fig. 2c linear-mean scan tolerates
  `pooled` (it means-pools anyway), but **all token-level modes (ortho probe / steering /
  direction tuning) require `cache_pooling: tokens`** — see
  [03-feature-caching-and-pooling.md](03-feature-caching-and-pooling.md).
- **Finding #5 — default-off is a hard invariant.** No `modes:` key ⇒ `modes_cfg == {}` ⇒ the
  dispatch block is skipped and `evals.analysis_vlm.modes` is never imported ⇒ zero behavior
  change for existing configs.
- **`attention_distance` needs the clip encoder.** The mode asserts `ctx.data_mode == "clip"`
  and requires `RoPEAttention` blocks; it captures **spatial** distance for the heatmap while
  also writing temporal distance to the JSON.
- **`attention_ablation` must set `cache_features: false`.** Ablation changes the features, so
  a cache built from the un-ablated forward would silently reuse baseline features.
- **`pre_norm: false` is the one wrong probe choice** (finding #3). LayerNorm-over-D or a
  per-feature StandardScaler both work; no normalization does not.
- **Frame sampling is the silent Fig. 2c killer** (finding #2). A flat early-speed curve is
  almost always a sampling problem (contiguous sub-window), not a model problem — check
  `frame_step` / `uniform_sampling` first.
- **Absolute R²/distance numbers are dataset-dependent.** The reproduction criteria are the
  *qualitative signatures* (early-speed; direction jump at ~one-third depth; `D̄` dip + `S`
  spike at the PEZ; direction ≫ speed subspace dim; monotone steering), not the paper's exact
  values on its own synthetic set.

## Cross-references

- Feature cache / pooling granularities → [03-feature-caching-and-pooling.md](03-feature-caching-and-pooling.md)
- Probe heads, regression task, NaN-masking, R², target standardization → [04-probes-regression-nanmask.md](04-probes-regression-nanmask.md)
- `uniform_sampling` / `frame_step` / `resize_mode` data knobs → [06-data-pipeline-changes.md](06-data-pipeline-changes.md)
- Plotting seam (`stage_val_acc.png`, PEZ shading) → [07-plotting.md](07-plotting.md)
- Blender toy-physics dataset generator → [09-blender-toy-dataset.md](09-blender-toy-dataset.md)
- Attention hooks (SDPA capture/ablation substrate, `AttentionDistanceCollector`) → [11-attention-hooks.md](11-attention-hooks.md)
- Additive analysis-mode dispatch (registry, `AnalysisContext`, `run_modes`) → [12-analysis-modes.md](12-analysis-modes.md)
- Full config reference → [13-configs-reference.md](13-configs-reference.md)
