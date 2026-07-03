# Analysis modes subpackage & reproduction roadmap

## Purpose

`evals/analysis_vlm/modes/` is an **additive, config-driven** post-hoc analysis layer bolted onto the
existing `analysis_vlm` layer-wise probing harness. It hosts paper-reproduction analyses of the frozen
V-JEPA2 encoder (attention-distance/ablation, orthogonal-probe steering, circular direction geometry)
behind a single config selector `experiment.analysis.modes`. The design invariant is strict:
**when that key is absent, the harness is byte-for-byte identical to before** — the mode package is never
even imported.

Exactly one mode is implemented today (`attention_distance`, Appendix C.6 / Fig. 19 & 3). The remaining
phases are specified but pending; the plan lives in
`evals/analysis_vlm/modes/REPRODUCTION_PLAN.md`.

> Scope note: this doc covers the **modes subpackage** and its two seams into `eval.py`. The base
> `analysis_vlm` probing harness (linear probes, `cache_pooling`, target standardization, `pre_norm`) is
> documented separately; see the `vlm-encoder-probing` memory / earlier `z_tech` sections. Cross-refs at
> the end.

---

## What changed vs upstream V-JEPA2

Against upstream commit `204698b` the **entire** `evals/analysis/` and `evals/analysis_vlm/` trees are
fork additions (they do not exist upstream — `git ls-tree 204698b -- evals/analysis_vlm/` is empty, and
`git diff 204698b -- evals/analysis_vlm/eval.py` reports the file as 814 new insertions). So "vs upstream"
for the probing harness means "all of it." The delta **this** subsystem introduces on top of the
pre-existing fork harness is small and enumerated below.

| Path | Status | Delta |
|------|--------|-------|
| `evals/analysis_vlm/modes/__init__.py` | **new** (untracked) | Mode registry, `AnalysisContext` dataclass, `run_modes()` dispatch. |
| `evals/analysis_vlm/modes/attention_distance.py` | **new** (untracked) | The one implemented mode: per-(layer,head) attention-weighted spatial/temporal distance + dual-axis plot. |
| `evals/analysis_vlm/modes/REPRODUCTION_PLAN.md` | **new** (untracked) | Phase 0–5 roadmap + consolidated config schema + adversarially-checked paper specs. |
| `evals/analysis/attention_hooks.py` | **new** (untracked) | SDPA monkey-patch context manager + `AttentionDistanceCollector` + ablation bias builder. **Reused as-is** by the mode, not modified. |
| `evals/analysis_vlm/eval.py` | **modified** | Two additive seams only: (1) `skip_base_probe` gate at `eval.py:501-507`; (2) the mode dispatch block at `eval.py:565-590`. Everything else is the pre-existing harness. |

Note: at time of writing the `modes/` package and `attention_hooks.py` are **untracked** working-tree
files; the `eval.py` seams are already committed (last touched by `778de51`).

---

## The three pieces

### 1. Registry + lazy import (`evals/analysis_vlm/modes/__init__.py`)

A mode is any callable `fn(cfg: dict, ctx: AnalysisContext) -> None` registered by name.

```python
# __init__.py:22-27
def register(name):
    """Decorator: register a mode implementation under `name`."""
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco
```

Imports are deferred so that `from evals.analysis_vlm.modes import AnalysisContext` has **no side effects**
and never fails on a half-written future mode file — mode modules self-register only when `run_modes`
actually runs (`__init__.py:30-36`). Only `attention_distance` is wired in today; the rest are commented
placeholders.

### 2. `AnalysisContext` — the read-only handle passed to every mode (`__init__.py:39-74`)

A frozen dataclass snapshot of `eval.py`'s local scope: the encoder, trained probe heads + metadata,
standardization stats, optional cached features, and two **closures** back into `eval.py` so a mode can
run fresh encoder forwards or build a val clip loader without re-implementing them.

Key fields:

| Field | Meaning |
|-------|---------|
| `encoder`, `device`, `folder`, `rank`, `world_size`, `use_bfloat16`, `plot_pez` | run context; each mode writes under `<folder>/<name>/`. |
| `heads`, `best_val`, `stages`, `embed_dims`, `reg_vars` | trained probes + their layer/stage/variable metadata. |
| `targets_t` | **standardized** targets on device (indexed by CSV label). |
| `targets_npy`, `col_mu`, `col_sd` | path to **raw** targets + the per-column `nanmean`/`nanstd` used to standardize — needed to recover true (unstandardized) angles/speeds. |
| `cache_pooling`, `data_mode` | granularity of cached features; `data_mode` is `"clip"` for V-JEPA. |
| `tr_feats/tr_labels/va_feats/va_labels` | cached features, **`None` when `cache_features=false`**. |
| `encode_clip(data_batch)` | closure: `_encode(encoder, d, device, data_mode, use_bfloat16)`. |
| `make_val_clip_loader()` | closure: builds the val clip `DataLoader` (workers=0). |

### 3. `run_modes(modes_cfg, ctx)` dispatch (`__init__.py:76-94`)

```python
# __init__.py:80-94  (abridged)
if not modes_cfg:
    return
_import_modes()
for name, cfg in modes_cfg.items():
    if cfg is False or (isinstance(cfg, dict) and cfg.get("enabled") is False):
        continue                                  # per-mode opt-out
    if name not in _REGISTRY:
        raise ValueError(f"unknown analysis mode {name!r}; valid: {sorted(_REGISTRY)}")
    cfg = {} if cfg in (True, None) else dict(cfg)
    cfg.pop("enabled", None)
    out = os.path.join(ctx.folder, name); os.makedirs(out, exist_ok=True)
    _REGISTRY[name](cfg, ctx)
```

Per-entry config semantics: `True`/`{}`/`None` ⇒ run with defaults; a dict ⇒ options; `False` or
`{enabled: false}` ⇒ skip that mode. Unknown names **raise** with the valid list. Modes run in insertion
(dict) order.

---

## The two `eval.py` seams

Both are additive and default-off.

**Seam 1 — `skip_base_probe`** (`eval.py:501-507`). Encoder-only modes (like attention distance) don't
need the layer probes trained, so the flag can shorten the train loop to 0 epochs:

```python
# eval.py:504
num_probe_epochs = 0 if args_analysis.get("skip_base_probe", False) else num_epochs
```

Default `False` ⇒ `num_probe_epochs == num_epochs` ⇒ the loop is byte-identical to before.

**Seam 2 — dispatch block** (`eval.py:565-590`). Placed after the plotting block, **rank 0 only**:

```python
# eval.py:568-570
modes_cfg = args_analysis.get("modes") or {}
if modes_cfg and rank == 0:
    from evals.analysis_vlm.modes import AnalysisContext, run_modes
    ctx = AnalysisContext(
        encoder=encoder, device=device, folder=folder, ...,
        targets_npy=(tpath if task == "regression" else None),
        col_mu=(mu if task == "regression" else None),
        col_sd=(sd if task == "regression" else None),
        tr_feats=(tr_feats if cache_features else None), ...,
        encode_clip=lambda d: _encode(encoder, d, device, data_mode, use_bfloat16),
        make_val_clip_loader=lambda: _split_loader(val_data_path[0], training=False, workers=0)[0],
    )
    run_modes(modes_cfg, ctx)
if torch.distributed.is_initialized():
    torch.distributed.barrier()
```

`args_analysis.get("modes") or {}` is `{}` for every existing config, so the block (and the mode import)
is skipped entirely. The trailing `barrier()` is a no-op unless DDP is active.

**Default-off guarantee (invariant):** no `modes` key ⇒ `modes_cfg == {}` ⇒ dispatch skipped, package
never imported; `skip_base_probe` absent ⇒ full training loop. `summary.json`, `log_r*.csv`, and
`stage_val_acc.png` are unchanged. `col_mu`/`col_sd` are captured by pure read (no mutation of the
existing standardization).

---

## Implemented mode: `attention_distance`

`evals/analysis_vlm/modes/attention_distance.py` (Appendix C.6 / Fig. 19 & 3). It measures, per
(layer, head), the **attention-weighted spatial (patch) and temporal (tubelet) distance** of the frozen
RoPE encoder, captured as a **detached side computation** inside the `attention_hooks` SDPA monkey-patch —
the encoder's actual output is bit-identical to baseline (`attention_hooks.py:47-48`, verified with
`torch.equal`).

Flow (`attention_distance.py:32-71`):

1. `assert ctx.data_mode == "clip"` — requires the V-JEPA clip encoder.
2. `_find_rope_attn(ctx.encoder)` (`attention_hooks.py:239-245`) walks `model.modules()` for all
   `RoPEAttention` blocks (24 for ViT-L); `num_heads` read from block 0 (16).
3. Build an `AttentionDistanceCollector(num_layers, num_heads, query_chunk, max_batches)`
   (`attention_hooks.py:161`).
4. `with attention_hooks(ctx.encoder, collector=collector):` run `ctx.encode_clip(data)` over the first
   `max_batches` val batches — the patched SDPA streams queries in `query_chunk` blocks so the
   `(B,H,N,N)` matrix is never fully materialized.
5. `collector.finalize()` (`attention_hooks.py:217-225`) returns
   `{spatial_distance:[L][H], temporal_distance:[L][H], num_layers, num_heads, rows_per_layer}`, dumped to
   `<folder>/attention_distance/attention_distance.json`.
6. `_plot(...)` writes a dual-axis PNG of layer-mean distance `Dbar = mean_h` (red) and head
   specialization `S = std_h` (blue, `statistics.pstdev`) vs **layer fraction**, with PEZ shading.

**Paper signature to validate:** `Dbar` **dips** to a minimum and `S` **spikes** around one-third depth
(the Physics Emergence Zone) as spatiotemporally-local heads emerge alongside the long-range heads. The
plot uses only `spatial_distance`; `temporal_distance` is captured but not plotted.

**Gotchas:**
- **`data_mode` must be `clip`** — asserted; VLM/framewise encoders are rejected.
- **Fresh forwards, not the cache** — the mode calls the encoder through `encode_clip`; set
  `cache_features: false`.
- **Token→(t,x,y) layout** matches `RoPEAttention.separate_positions` exactly
  (`attention_hooks.py:84-90`); `resolution: 224` gives the paper geometry (14×14×8 = 1568 tokens).
- Plotting is optional: if matplotlib is unavailable it warns and skips rather than failing the analysis
  (`attention_distance.py:79-81`).

---

## Config

A mode config clones the working `vjepa_combined.yaml` and adds only the `modes:` block (plus a
`cache_pooling` change where token-level modes need it). The real, in-repo example
`configs/analysis/blender_toy_dataset/vjepa_attn_distance.yaml`:

```yaml
eval_name: analysis_vlm
experiment:
  analysis:
    model: vjepa
    task: regression
    regression:
      targets_npy: /data/.../data_csv/blender_toy/blender_targets.npy
      variables:
        - {name: direction, cols: [1, 2]}
    stages: { vision_encoder: all }
    plot_pez: [0.2, 0.4]
    skip_base_probe: true          # encoder-only: distance capture needs no trained probe
    modes:
      attention_distance:          # ← default-off unless present; absent ⇒ existing behavior
        enabled: true
        query_chunk: 512           # stream queries so (B,H,N,N) is never materialized
        max_batches: 8             # first 8 val batches (cheap, stable)
    probes:
      - { type: linear, pooling: mean, pre_norm: true,
          optimization: { lr: 0.001, weight_decay: 0.1, warmup: 2.0 } }
  data:
    dataset_type: VideoDataset
    resolution: 224                # 14x14 patches x 8 tubelets = 1568 tokens (paper geometry)
    resize_mode: resize
    frame_step: 1
    uniform_sampling: true         # 16-frame Blender clips: sample evenly over the whole video
    num_segments: 1
    dataset_train: /data/.../blender_toy/velocity_val.csv
    dataset_val:   /data/.../blender_toy/velocity_val.csv
    num_classes: 4
    frames_per_clip: 16
  optimization:
    batch_size: 8
    num_epochs: 0                  # no probe training
    use_bfloat16: true
    cache_features: false          # attention capture runs fresh forwards, not the cache
model_kwargs:
  module_name: evals.analysis.modelcustom.vit_encoder_multilayer
  pretrain_kwargs:
    encoder: { model_name: vit_large, patch_size: 16, tubelet_size: 2, use_rope: true, ... }
```

Config keys introduced by this subsystem: **`experiment.analysis.modes`** (mapping mode→cfg) and
**`experiment.analysis.skip_base_probe`** (bool, default `false`). Per-mode keys for `attention_distance`:
`enabled`, `query_chunk` (default 512), `max_batches` (default 8).

> `frame_step: 1 + uniform_sampling: true` is deliberate for the natively-16-frame Blender clips — see the
> frame-sampling gotcha in the findings below and the frame-sampling `z_tech` section.

---

## Reproduction roadmap (`REPRODUCTION_PLAN.md`)

The plan collapses three earlier module designs into **one** selector `experiment.analysis.modes` and has
each mode **reuse** existing machinery rather than re-implement. Consolidated schema: one default-off
`modes:` block; **removing it reproduces `vjepa_combined.yaml` behavior exactly.** Optional
`skip_base_probe` shortens the base loop for encoder-only modes.

Planned modes and paper sections:

| Phase | Mode | Paper | Status | Reuses |
|-------|------|-------|--------|--------|
| 0 | dispatch scaffold (`__init__` + `eval.py` seams) | — | **DONE** | — |
| 1 | `attention_distance` | C.6 / Fig. 19, 3 | **DONE** | `attention_hooks.py` (as-is) |
| 5 | `attention_ablation` | C.6 / Table 4 | pending | `attention_hooks` ablation bias; re-eval frozen `ctx.heads` |
| 2 | `orthogonal_probe_sequence` | C.11 | pending | new `subspace.py` (QR deflation) |
| 3 | `steering` | C.12 | pending | `subspace.py` least-squares steering |
| 4 | `direction_tuning` | C.7 / C.8 / C.10 | pending | new `plotting.plot_direction_tuning` |

Dependency DAG: `__init__/dispatch → attention_hooks → {attention_distance, attention_ablation}`;
`subspace.py → {orthogonal_probe_sequence, steering}`; `direction_tuning → plotting`.

Planned per-mode config knobs (from the plan's schema, subject to change until implemented):
- **attention_ablation** — `ablate_layers` (default = `stages`), `spatial: [1,3,5,7,9,11,13]`,
  `temporal: [1,2,3,4,5,6]`, `combined: [[3,1],…]`, `per_patch`. **Requires `cache_features: false`**
  (ablation changes features; a cache would reuse unmasked baselines). A known open question: the paper's
  collapse at (s=3,t=1) implies **UNION** semantics, whereas current `build_ablation_bias` uses AND for
  `combined` — flagged as a knob to add.
- **orthogonal_probe_sequence** — `variable`, `cols`, `max_dims`, `epochs` (100 direction / 50 speed),
  `lr: 0.001`, `weight_decay: 0.0001`, `stop: {r2_below, mae_above_deg}`. **Needs `cache_pooling: tokens`.**
- **steering** — `stage: 8`, `orthogonal_probe: {…}`, `steer: {train_frac: 0.7, thetas_deg, n_sweep,
  coord_from, forward_intervention: false}`. Default `forward_intervention: false` = offline re-decode
  (the validated path). **Needs `cache_pooling: tokens`.**
- **direction_tuning** — `variables`, `contrast`, `site`, `n_bins`, `cv_folds`, `ridge_alpha`,
  `repr_layers`, `sawtooth`. **Needs `cache_pooling: tokens`.**

Two future touched files (not yet applied): `evals/analysis/plotting.py` gains
`plot_direction_tuning(...)` (Phase 4); everything else is new files under `modes/`.

---

## Key project findings

1. **Fig. 2c dissociation reproduces on the paper-faithful Blender toy dataset.** Frozen V-JEPA2-L
   layer-wise R² shows **SPEED decodable early** (R² ≈ 0.68 at layer 0), **DIRECTION emerging sharply at
   the Physics Emergence Zone** (≈ 0.28 at L0 → ≈ 0.9 by layer fraction 0.3–0.4), with `accel_mag` in
   between. An earlier anti-shortcut generator (random shape/color/size) did **not** reproduce
   early-speed; the fix was (a) a paper-faithful single fixed **red sphere** and (b) correct frame
   sampling.

2. **`frame_step` sampling bug.** `VideoDataset.loadvideo_decord` with `frame_step=1` on a 64-frame clip
   samples 16 **contiguous** frames (first 1/4 of the trajectory) → sub-patch motion per tubelet → layer 0
   cannot encode speed/accel. Fix: `frame_step=4` (span the whole clip) **or** the new `uniform_sampling`
   option (evenly sample `fpc` frames over the whole video, length-agnostic). Blender clips are natively
   16 frames, so their configs use `frame_step: 1 + uniform_sampling: true`.

3. **Linear-probe normalization (`pre_norm`) is REQUIRED.** `PooledLinearProbe(pre_norm=True)` applies
   `nn.LayerNorm` over the feature dim per sample. A competitor fork set `pre_norm=False` (raw pooled
   features) + a fixed lr and got poor linear-probe accuracy, because V-JEPA activation scale differs by
   orders of magnitude across layers, so one lr cannot fit all layers without input normalization. Valid
   alternatives: LayerNorm (per-sample over D) **or** per-feature StandardScaler; "no normalization" is
   the only wrong choice.

4. **`cache_pooling` granularity.** `'pooled'` = `[mean‖max]` (2D, collapses time, **degrades
   direction**); `'tokens'` = full `(N,D)`; `'framewise'` = `(T,D)`, VLM-only. **Token-level modes**
   (orthogonal-probe, steering, direction-tuning) require `cache_pooling: tokens`.

5. **Reproduction roadmap** (`evals/analysis_vlm/modes/REPRODUCTION_PLAN.md`): additive config-driven
   modes for attention distance (C.6), attention ablation (C.6 Table 4), orthogonal probe sequence
   (C.11) + steering (C.12), and circular direction geometry (C.7/C.10). **Phase 0** (dispatch scaffold,
   default-off) and **Phase 1** (`attention_distance`) are **DONE**; **Phases 2–5** are pending.

---

## Cross-refs

- **Base `analysis_vlm` probing harness** (probes, `pre_norm`, `cache_pooling`, target standardization) —
  `vlm-encoder-probing` memory / earlier `z_tech` section.
- **Frame sampling** (`uniform_sampling`, `frame_step`) — the frame-sampling `z_tech` section.
- **Blender toy-physics data generation** — the Blender dataset `z_tech` section.
- `evals/analysis/attention_hooks.py` — the SDPA capture/ablation primitive reused by `attention_distance`.
