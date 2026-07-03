I now have the full ground truth. `evals/analysis/attention_hooks.py` (Design A) is already implemented and verified default-off; `eval.py` is 783 lines with the dispatch seam at line 560. Here is the consolidated, implementation-ready blueprint.

---

# Implementation Blueprint — Physics-Interpretability Analyses on the Blender Toy-Physics Dataset

## 0. Reconciliation decision (why one seam, not three)

The three module designs overlap and use three different config namespaces (`analysis.attention`, `analysis.steering`, `analysis.modes`). I collapse them into **one** selector — `experiment.analysis.modes` (Design B's registry) — and have each mode **reuse the strongest existing machinery** rather than re-implement:

- `attention_distance` + `attention_ablation` → thin adapters over the **already-written, verified** `evals/analysis/attention_hooks.py` (Design A). No rewrite.
- `orthogonal_probe_sequence` + `steering` → a new shared `subspace.py` implementing C.11 deflation + C.12 least-squares steering (Design C's math).
- `direction_tuning` → new module (C.7/C.8/C.10), rendered by a new plotting fn.

Net additive surface to existing code: **exactly two touched files** — one dispatch block at the tail of `evals/analysis_vlm/eval.py:main()` and one new function in `evals/analysis/plotting.py`. `main.py`/`scaffold.py`/`src/models/**` untouched. `eval_name` stays `analysis_vlm`.

Ground-truth facts this relies on (verified by reading source):
- `eval.py` standardizes targets in place at lines **201–203** (`mu=nanmean`, `sd=nanstd`, `targets_arr=(x−mu)/clip(sd)`) → the trained direction probe predicts **standardized** (sin,cos). Modes needing true angle must re-load `blender_targets.npy` raw.
- `blender_targets.npy` = `(672,4)` = `[speed, sinθ, cosθ, accel_mag]`; NaN counts `[280, 0, 0, 392]`. Direction defined for **all** 672; speed for velocity rows 0–391; accel_mag for accel rows 392–671. Val split = 134 clips (78 velocity + 56 accel).
- `_encode` (line 618) returns `feats: list[(B,N,D)]` per stage; `run_one_epoch` (643) computes per-head masked R². `heads[i]` carries `name, layer, layer_pos, stage, series, module, tcols`.
- Cache granularities (`cache.py`): `pooled` (n,2D mean‖max — collapses time, **degrades direction**), `tokens` (n,N,D — full), `framewise` (VLM-only, needs `num_temporal`). V-JEPA has no `num_temporal` → token-level modes require `cache_pooling: tokens`.
- `attention_hooks.py` monkey-patches `F.scaled_dot_product_attention` scoped to a context manager, byte-identical when off; `_find_rope_attn` walks `model.modules()` and finds all 24 `RoPEAttention` blocks through the `MultiLayerClipAggregation` wrapper. `encoder.num_heads=16` (vit_large), depth 24.
- Plotting seam: `plot_layer_val_acc` + `_elbow_x`; PEZ shading inline at plotting.py **76–80**; dispatch insertion point is **after line 559** (end of `if rank==0 and make_plot:`).

---

## 1. Ordered incremental implementation plan (by dependency + risk)

**Phase 0 — Dispatch scaffold (foundation; nothing runs without it).**
Build `evals/analysis_vlm/modes/__init__.py` (registry, `run_modes`, `AnalysisContext`) + the single `eval.py` dispatch block. Ship with an **empty registry** first and prove default-off: run the existing `vjepa_combined.yaml` and confirm `summary.json`/`log_r*.csv`/`stage_val_acc.png` are byte-identical to a pre-change run. This is the guarantee gate — do not proceed until it passes.

**Phase 1 — `attention_distance` (lowest risk; no probe deps; exercises the hook + dispatch path end-to-end).**
Thin adapter over the existing `AttentionDistanceCollector` + `attention_hooks(...)`. Reuses `evals/analysis/attention_hooks.py` verbatim. Validates the whole seam with the least moving parts. Depends only on Phase 0 + `hooks.py` (a 20-line re-export wrapper).

**Phase 2 — `subspace.py` (C.11) → `orthogonal_probe_sequence` mode.**
Pure feature-space (no hooks). Establishes: raw-angle target extraction from `targets_npy`, fp16→fp32 upcast, QR deflation, empirical circular chance. `orthogonal_probe_sequence` is a thin wrapper. Steering depends on this, so it comes first.

**Phase 3 — `steering` mode (C.12).**
Consumes `subspace.py`'s `{W_k}`, builds `V=QR([W_1ᵀ…W_Kᵀ])`, least-squares `c*`, offline re-decode with a held-out probe, sweep N. Optional real-intervention path via `steer_residual` CM (default off).

**Phase 4 — `direction_tuning` mode + `plotting.plot_direction_tuning`.**
Independent of hooks/subspace; uses token-level activations + raw θ. Per-neuron sin/cos GLM (C.7), quadratic speed GLM (C.8), sawtooth reuse of `subspace.py`. Add the new plotting function here.

**Phase 5 — `attention_ablation` mode (heaviest; done last).**
Combines the ablation branch of `attention_hooks` with re-extraction under mask + re-eval of frozen `ctx.heads` (Direction R²) and an optional per-patch probe (Per-patch R²). Depends on Phases 0–1 machinery + trained heads.

Dependency DAG: `__init__/dispatch → hooks.py → {attention_distance, attention_ablation}`; `subspace.py → {orthogonal_probe_sequence, steering}`; `direction_tuning → plotting`; `steering` also depends on `subspace.py`.

---

## 2. Consolidated config schema (one default-off block; one YAML per experiment)

Single new key: `experiment.analysis.modes` (mapping mode→cfg). **Absent ⇒ `{}` ⇒ dispatch skipped ⇒ today's behavior byte-for-byte.** Per-entry `enabled:false` / `false` / omit ⇒ that mode skipped; `true`/`{}` ⇒ run with defaults. Optional `experiment.analysis.skip_base_probe` (default `false`) shortens the base train loop for encoder-only modes.

All five clone the working `vjepa_combined.yaml` and add only the `modes:` block (+ a `cache_pooling` change where token-level is needed).

### 2a. attention-distance — `configs/analysis/blender_toy_dataset/vjepa_attn_distance.yaml`
```yaml
experiment:
  analysis:
    model: vjepa
    task: regression
    regression:
      targets_npy: /data/hyuntak/.../data_csv/blender_toy/blender_targets.npy
      variables: [{name: direction, cols: [1,2]}]
    stages: {vision_encoder: all}
    probes: [{type: linear, pooling: mean, pre_norm: true}]
    skip_base_probe: true            # encoder-only: no probe needed for distance capture
    modes:
      attention_distance: {enabled: true, query_chunk: 512, max_batches: 8}
  data: {dataset_type: VideoDataset, resolution: 224, resize_mode: resize, frames_per_clip: 16,
         uniform_sampling: true, num_segments: 1,
         dataset_val: /data/.../data_csv/blender_toy/velocity_val.csv,
         dataset_train: /data/.../data_csv/blender_toy/velocity_train.csv, num_classes: 4}
  optimization: {batch_size: 8, num_epochs: 0, use_bfloat16: true, cache_features: false}
```
Note: `resolution: 224` (paper geometry: 14×14×8=1568 tokens); the default `256` gives 16×16 patches — either works, distance is in patch units, but 224 matches the paper.

### 2b. attention-ablation — `vjepa_attn_ablation.yaml`
```yaml
    task: regression
    regression:
      targets_npy: .../blender_targets.npy
      variables: [{name: direction, cols: [1,2]}, {name: speed, cols: [0]}]
    stages: {vision_encoder: [8]}    # read out / ablate at the PEZ (~one-third of 24)
    probes: [{type: linear, pooling: mean, pre_norm: true}]
    modes:
      attention_ablation:
        enabled: true
        ablate_layers: [8]           # PEZ layers to mask (default = stages)
        spatial:  [1,3,5,7,9,11,13]
        temporal: [1,2,3,4,5,6]
        combined: [[3,1],[5,2],[7,3],[9,4],[11,5],[13,6]]
        variables: [direction]       # metrics recomputed under each mask
        per_patch: true              # also compute per-patch Direction R² (needs velocity clips)
  optimization: {cache_features: false}   # REQUIRED: ablation changes features per setting
```

### 2c. orthogonal-probe (C.11) — `vjepa_ortho_probe.yaml`
```yaml
    task: regression
    regression: {targets_npy: .../blender_targets.npy, variables: [{name: direction, cols: [1,2]}]}
    stages: {vision_encoder: all}
    probes: [{type: linear, pooling: mean, pre_norm: true}]
    skip_base_probe: true
    modes:
      orthogonal_probe_sequence:
        enabled: true
        variable: direction          # speed also supported (1-D)
        source: regression           # atan2 over raw cols [1,2]
        cols: [1,2]
        max_dims: 32
        epochs: 100                   # paper: 100 (direction) / 50 (speed)
        lr: 0.001                     # Adam
        weight_decay: 0.0001
        batch_size: 256
        stop: {r2_below: 0.1, mae_above_deg: 80}   # C.11 text thresholds (Fig-22 set: r2_below 0.3)
        standardize: true
  optimization: {cache_features: true, cache_pooling: tokens, cache_max_gb: 80}
```

### 2d. steering (C.12) — `vjepa_steering.yaml`
```yaml
    task: regression
    regression: {targets_npy: .../blender_targets.npy, variables: [{name: direction, cols: [1,2]}]}
    stages: {vision_encoder: [8]}    # paper reports layer 8
    probes: [{type: linear, pooling: mean, pre_norm: true}]
    skip_base_probe: true
    modes:
      steering:
        enabled: true
        stage: 8
        source: regression
        cols: [1,2]
        orthogonal_probe: {max_dims: 25, epochs: 100, lr: 0.001, weight_decay: 0.0001,
                           batch_size: 256, stop: {r2_below: 0.1}}
        steer:
          train_frac: 0.7            # C.12: 70/30 disjoint; held-out eval probe on test
          thetas_deg: [90]           # target θ*
          n_sweep: [1,2,3,5,8,12,16,20,25]
          coord_from: invert         # least-squares c* (centroid fallback if too few near θ*)
          alpha: 1.0
          forward_intervention: false  # false = offline re-decode (default/validated)
        plot: true
  optimization: {cache_features: true, cache_pooling: tokens, cache_max_gb: 80}
```

### 2e. circular-direction-geometry (C.7/C.8/C.10) — `vjepa_direction_tuning.yaml`
```yaml
    task: regression
    regression:
      targets_npy: .../blender_targets.npy
      variables: [{name: speed, cols: [0]}, {name: direction, cols: [1,2]}, {name: accel_mag, cols: [3]}]
    stages: {vision_encoder: [0, 8, 12, 23]}   # Layer 0 (baseline) vs 8 (PEZ) for Fig-20 contrast
    probes: [{type: linear, pooling: mean, pre_norm: true}]
    plot: true
    plot_pez: [0.2, 0.4]
    modes:
      direction_tuning:
        enabled: true
        variables: [direction]       # circular target
        contrast: [speed]            # non-circular baseline (no sawtooth / no ring)
        site: block_out              # block_out | mlp_fc1 | mlp_fc2 (per-neuron GLM site)
        n_bins: 24
        cv_folds: 5
        ridge_alpha: 0.001
        repr_layers: [0, 8]          # Fig-20 tuning-vector fans
        sawtooth: true               # reuse orthogonal_probe_sequence for Fig-23 redundancy
  optimization: {cache_features: true, cache_pooling: tokens, cache_max_gb: 80}
```

Removing the `modes:` block from any of the above reproduces `vjepa_combined.yaml` behavior exactly.

---

## 3. NEW files and MINIMAL additive touches

### NEW files
```
evals/analysis_vlm/modes/__init__.py                       # REGISTRY + run_modes() + AnalysisContext
evals/analysis_vlm/modes/hooks.py                          # re-export attention_hooks + steer_residual CM
evals/analysis_vlm/modes/subspace.py                       # C.11 deflation + C.12 steering core
evals/analysis_vlm/modes/attention_distance.py             # Phase 1
evals/analysis_vlm/modes/attention_ablation.py             # Phase 5
evals/analysis_vlm/modes/orthogonal_probe_sequence.py      # Phase 2
evals/analysis_vlm/modes/steering.py                       # Phase 3
evals/analysis_vlm/modes/direction_tuning.py               # Phase 4
configs/analysis/blender_toy_dataset/vjepa_attn_distance.yaml
configs/analysis/blender_toy_dataset/vjepa_attn_ablation.yaml
configs/analysis/blender_toy_dataset/vjepa_ortho_probe.yaml
configs/analysis/blender_toy_dataset/vjepa_steering.yaml
configs/analysis/blender_toy_dataset/vjepa_direction_tuning.yaml
```
`evals/analysis/attention_hooks.py` **already exists** — reused, not modified.

**`__init__.py` contract:**
- `_REGISTRY: dict[str, callable]`; `@register(name)` decorator; `_import_modes()` imports the five mode files to populate it.
- `AnalysisContext` (dataclass, read-only): `encoder, heads, best_val, stages, embed_dims, reg_vars, task, num_classes, targets_t, targets_npy, col_mu, col_sd, tr_feats, tr_labels, va_feats, va_labels, cache_pooling, data_mode, use_bfloat16, device, rank, world_size, folder, plot_pez, encode_clip (closure), make_val_clip_loader (closure)`.
- `run_modes(modes_cfg, ctx)`: iterate in insertion order; skip `False`/`{enabled:false}`; normalize `True/None→{}`; raise on unknown name (list valid); `os.makedirs(ctx.folder/<mode>)`; call `_REGISTRY[name]({...cfg}, ctx)`. Each mode writes `<folder>/<mode>/*.json` + `*.png`.

**`hooks.py` contract:** `from evals.analysis.attention_hooks import attention_hooks, AttentionDistanceCollector, build_ablation_bias, _find_rope_attn` (re-export). Add `steer_residual(encoder, layer, vector, alpha)` CM = `register_forward_pre_hook` on the underlying ViT block adding `alpha*vector`, removed in `finally`. Add `resolve_blocks(encoder)` → the ViT's `.blocks` (via `encoder.model.blocks` or module walk).

**`subspace.py` contract (C.11/C.12):**
`build_angle_targets(targets_npy, cols, row_index)`, `flatten_stage(feat_fp16)→X_fp32`, `fit_linear_probe(Xtr,Ytr,Xva,Yva,out_dim,lr,wd,epochs,bs)` (bare `nn.Linear` + `torch.optim.Adam`, MSE on (sin,cos), returns raw `W`), `angular_error`, `circular_chance_error`, `orthonormalize` (modified Gram-Schmidt + periodic QR), `fit_orthogonal_probe_sequence(...)→{basis,W_list,marginal_err,cumulative_err,k_selected,chance}`, `build_subspace_V(W_list)=QR(stack)`, `steer_toward(X,V,W_list,theta_star,coord_from)` (least-squares `c*`, `x*=Vc*+x⊥`), `evaluate_steering_vs_dims(...)→(dims,err_before,err_after)`.

### Touched file 1 — `evals/analysis_vlm/eval.py`
**Insert ONE block after line 559** (immediately after the `if rank == 0 and make_plot:` plot block, before the `class _DirectResizeClipTransform` at 562):
```python
    # ── ADDITIVE: post-hoc analysis modes. Default absent ⇒ skipped entirely. ──
    modes_cfg = args_analysis.get("modes") or {}
    if modes_cfg and rank == 0:
        from evals.analysis_vlm.modes import run_modes, AnalysisContext
        ctx = AnalysisContext(
            encoder=encoder, heads=heads, best_val=best_val, stages=stages,
            embed_dims=embed_dims, reg_vars=reg_vars, task=task, num_classes=num_classes,
            targets_t=targets_t,
            targets_npy=(reg_cfg.get("targets_npy") or reg_cfg.get("targets")) if task == "regression" else None,
            col_mu=(mu if task == "regression" else None), col_sd=(sd if task == "regression" else None),
            tr_feats=(tr_feats if cache_features else None), tr_labels=(tr_labels if cache_features else None),
            va_feats=(va_feats if cache_features else None), va_labels=(va_labels if cache_features else None),
            cache_pooling=cache_pooling, data_mode=data_mode, use_bfloat16=use_bfloat16,
            device=device, rank=rank, world_size=world_size, folder=folder, plot_pez=plot_pez,
            encode_clip=lambda d: _encode(encoder, d, device, "clip", use_bfloat16),
            make_val_clip_loader=lambda: _split_loader(val_data_path[0], training=False, workers=0)[0],
        )
        run_modes(modes_cfg, ctx)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
```
**One-line change to the base loop** for `skip_base_probe` (line 503):
```python
    num_probe_epochs = 0 if args_analysis.get("skip_base_probe", False) else num_epochs
    for epoch in range(start_epoch, num_probe_epochs):
```
`mu`, `sd`, `tpath`, `tr_feats/va_feats`, `_split_loader`, `val_data_path` are all already in scope at line 560. `reg_cfg` too.

**Why additive/identical-when-off:** `args_analysis.get("modes") or {}` is `{}` for every existing config (grep-confirmed the key is unused) ⇒ block skipped, no import. `skip_base_probe` defaults `False` ⇒ `num_probe_epochs == num_epochs` ⇒ identical loop. `mu/sd` are already computed (pure read, no mutation). The `barrier()` only fires under DDP and is a no-op for correctness. RNG draw order, `summary.json`, `log_r*.csv`, `stage_val_acc.png` unchanged.

### Touched file 2 — `evals/analysis/plotting.py`
Add `plot_direction_tuning(per_layer, out_path, subtitle=None, pez=None, repr_layers=None)` (3 panels: per-neuron sin/cos tuning-vector fans Layer 0 vs 8; decoded-θ̂ sawtooth scatter vs flat speed; per-layer sawtooth-index / circular-corr curve with `_elbow_x` + PEZ shading) and a private `_shade_pez(ax, depth, pez)` refactored from lines 76–80. **Leave `plot_layer_val_acc` and `_elbow_x` byte-identical** (do not route them through `_shade_pez`). Nothing executes at import; the new fn is reached only when `direction_tuning` runs.

---

## 4. Per-experiment run recipe on Blender + expected result to validate

Common: single GPU (`--devices cuda:0 --debugmode True`), frozen V-JEPA 2-L (`vit_large`, d=1024, 24 blocks, 16 heads), `velocity_*.csv` for direction/speed, `blender_targets.npy` cols `[speed, sinθ, cosθ, accel_mag]`.

### (A) attention-distance
`python -m evals.main --fname configs/analysis/blender_toy_dataset/vjepa_attn_distance.yaml --devices cuda:0 --debugmode True`
→ `<folder>/attention_distance/attention_distance.json` = `{spatial_distance:[24][16], temporal_distance:[24][16], ...}`. The mode computes `Dbar^(l)=mean_h`, `S^(l)=std_h` and plots vs layer 0–23.
**Validate (Fig. 19/3):** `Dbar` starts ~8 patches (L0), **dips to a minimum ~5** around the one-third-depth PEZ (L5–8), rises to a peak ~9 late; `S` **spikes** (~2.5–2.7) in the PEZ and is ~0.3 at L0. Per-head cells span ~1.3–10 patches with local+global heads coexisting at the PEZ. Directional sign of the dip/spike at the emergence zone is the pass criterion (absolute values are dataset-dependent).

### (B) attention-ablation
`... --fname .../vjepa_attn_ablation.yaml ...`
→ `<folder>/attention_ablation/ablation.json`: for each of baseline + 7 spatial + 6 temporal + 6 combined settings, `{Direction_R2, Per_patch_R2}` (IntPhys/ImageNet columns need external datasets — out of scope for Blender). Uses `attention_hooks(encoder, ablation=spec)` around fresh clip forwards, re-reads the frozen `ctx.heads[direction]`.
**Validate (Table 4 signatures):** (a) spatial-only barely moves global Direction R² (baseline→ small drop across s) but **collapses Per-patch R²** (0.72→…→<0 by large s); (b) temporal-only **hurts global Direction R²** more (down toward ~0.8); (c) **combined destroys direction even at the mildest (s=3,t=1)** (R²→~0.14 then <0). Exact numbers won't match the paper's synthetic ball set; the **qualitative ordering** (spatial spares global/kills per-patch; combined collapses direction) is the pass criterion.

### (C) orthogonal-probe (C.11)
`... --fname .../vjepa_ortho_probe.yaml ...` (`cache_pooling: tokens`)
→ `<folder>/orthogonal_probe_sequence/ortho.json`: per layer `{k_selected, marginal_err[], cumulative_err[], chance}`. Deflation loop with Adam(1e-3, wd 1e-4), stop at R²<0.1 or MAE>80°.
**Validate (Fig. 22 / Table 3):** direction K **grows with depth** and **exceeds speed K** at every layer; direction effective-dim (2K) is large and rises; speed stays modest and roughly flat. On the 4-way-ish Blender directions expect smaller K than the paper's 8-direction set, but **direction ≫ speed** must hold.

### (D) steering (C.12)
`... --fname .../vjepa_steering.yaml ...` (`cache_pooling: tokens`, layer 8)
→ `<folder>/steering/steering_summary.json` + `steering_error_vs_dims.png`. 70/30 split, up to 25 train probes, held-out eval probe on the test split, offline re-decode of the edited cache, sweep N.
**Validate (Fig. 24):** held-out **MAE-to-target falls monotonically** as N grows (baseline ≈ chance ~80–90° → small with all probes), while **MAE-to-true-label rises** (the curves cross), confirming a genuine shift. 1 probe ≈ little effect (>50°); many-probe ≈ small error. The **monotone target-error reduction + true-label-error increase** is the pass criterion (absolute 11.9° endpoint is dataset-specific).

### (E) circular-direction-geometry (C.7/C.8/C.10)
`... --fname .../vjepa_direction_tuning.yaml ...` (`cache_pooling: tokens`, stages [0,8,12,23])
→ `<folder>/direction_tuning/direction_tuning.json` + `direction_tuning.png`. Per-neuron sin/cos GLM (5-fold ΔR², ridge 1e-3), PD=atan2(β_sin,β_cos), speed quadratic GLM, sawtooth via reused C.11.
**Validate (Fig. 20/21/23):** (1) Layer-0 tuning vectors sporadic/short/disorganized vs **Layer-8 dense organized fan tiling 360°**; (2) direction redundancy curve shows the **jagged sawtooth** (paired sin/cos) while **speed R² decays smoothly** with no oscillation; (3) speed preferred-direction heatmap shows **no ring**. The direction-vs-speed qualitative contrast is the pass criterion.

---

## 5. Top correctness risks + how to test each

**Global (must gate before anything else): default-off guarantee.**
Risk: the dispatch/`skip_base_probe`/`col_mu` capture silently alters the baseline run.
Test: `git stash`-run `vjepa_combined.yaml`, save `summary.json`+`log_r*.csv`; apply the two touches; re-run with **no `modes:` key**; `diff` all three outputs → must be byte-identical. Also assert `args_analysis.get("modes")` is `None` for all existing configs (`grep -rL "modes:" configs/`).

**attention-distance.**
Risk 1: capture materializes softmax the SDPA fast path skips → OOM at N≈1568–4096. Mitigation is in place (`query_chunk`, `max_batches`). Test: run at `batch_size:8, resolution:224`; watch peak GPU mem; confirm no OOM and JSON has 24×16 finite entries.
Risk 2: side-softmax scale vs returned SDPA scale diverges under custom `qk_scale`. Test: stock V-JEPA has `qk_scale=None`; assert `module.scale == head_dim**-0.5`. Correctness of "capture doesn't change features" is guaranteed by attention_hooks returning the original SDPA output — verify with a `torch.equal` probe on one batch with/without the collector.
Risk 3: token→(t,x,y) indexing must match `RoPEAttention.separate_positions` (`t=idx//(H*W)`). Already matched in `_coords`. Test: assert `ds.max()≈sqrt(2)*(grid−1)` and `dt.max()==T−1`.

**attention-ablation.**
Risk 1 (highest): ablation changes features, so **`cache_features:true` would reuse unmasked baseline features**. Test: the mode must `assert not ctx.cache_pooling or fresh-forward`; force `cache_features:false` in the config and assert the mode re-runs the encoder (not the cached path) — verify baseline row reproduces the un-ablated Direction R² and s=3,t=1 differs.
Risk 2: combined regime boolean (UNION vs INTERSECTION). `build_ablation_bias` uses **AND** for `combined`; the paper's collapse at (s=3,t=1) implies UNION (apply spatial OR temporal). Test: run both; the setting that collapses direction at the mildest pair matches the paper → adopt it. (Flag as a knob; current `attention_hooks` uses AND — likely needs a UNION option added, which is an additive change to `build_ablation_bias(mode='combined_or')`.)
Risk 3: fully-masked query row → NaN. `build_ablation_bias` already un-masks full rows. Test: run s=13,t=6 (most aggressive) and assert outputs are NaN-free.

**orthogonal-probe.**
Risk 1: pooled cache degeneracy. Test: assert `cache_pooling=='tokens'`; error loudly on `pooled`.
Risk 2: standardization — atan2 over standardized sin/cos warps the circle. Test: `subspace.build_angle_targets` re-loads raw `targets_npy`; assert `angle ∈ [−π,π]` and matches `arctan2(raw_sin, raw_cos)` for 5 known clips.
Risk 3: Gram-Schmidt drift + fp16. Test: after fitting, assert `‖BᵀB − I‖ < 1e-4` and inputs upcast to fp32.
Risk 4: chance threshold — hard-coded 90° wrong for discrete directions. Test: `circular_chance_error` = val error of predicting train circular mean; assert stop fires within a few dims of where marginal R²<0.1 empirically.

**steering.**
Risk 1: 70/30 disjointness — steering probes and the eval probe must not share activations. Test: assert train/test index sets disjoint; eval probe R² on test ≈ high (~0.9+).
Risk 2: coord_from='centroid' fails for off-grid θ* on 4-way data. Test: request θ*=90 (on-grid) first; assert `invert` fallback path runs when centroid support <5 samples.
Risk 3: offline vs forward-intervention layout. Keep `forward_intervention:false` as the validated default; test the offline path first (edit cached X, re-decode). Only enable the hook path after asserting the delta reshapes to `(B, T*S, D)` temporal-major.

**direction-tuning.**
Risk 1: standardized readout → un-standardize with `col_sd/col_mu` before atan2; ground-truth θ from raw npy. Test: decoded θ̂ vs true θ scatter is near-diagonal at Layer 8.
Risk 2: atan2 column order — `cols:[1,2]` is `[sin,cos]` → `atan2(readout[...,0], readout[...,1])`. Test: swapping cols must visibly mirror the tuning fan; assert the mode keys off `reg_vars` cols, not positions.
Risk 3: small val set (78 velocity val) with 24 bins ≈ 3 samples/bin. Test: use adaptive `n_bins`, report SEM, and rely on the harmonic-fit sawtooth-index (all-sample) as the quantitative metric, not per-bin means.
Risk 4: mean-pool collapses time → per-neuron GLM should use **token-level** (`cache_pooling:tokens`) activations; velocity clips have constant θ so pooling is valid but per-neuron Fig-20 needs tokens. Test: run `site: block_out` first; confirm PD distribution tiles 360° at L8 and is clustered/short at L0.

**DDP note (all modes):** modes run on rank 0 only; `make_val_clip_loader` uses the closure's `world_size/rank` so under `world_size>1` it is sharded. For faithful reproduction launch the analysis **single-GPU** (consistent with the paper's single clip-set and the repo's documented single-GPU debug path), or extend `run_modes` to all-gather cached shards to rank 0. Test: assert `world_size==1` in mode entry (warn otherwise).

---

**Files referenced (all absolute):**
- Reused as-is: `/data/hyuntak/project/2026/2027_cvpr/vjepa2/evals/analysis/attention_hooks.py`
- Touched (2): `/data/hyuntak/project/2026/2027_cvpr/vjepa2/evals/analysis_vlm/eval.py` (dispatch after L559 + L503 `skip_base_probe`), `/data/hyuntak/project/2026/2027_cvpr/vjepa2/evals/analysis/plotting.py` (`plot_direction_tuning`+`_shade_pez`)
- New package: `/data/hyuntak/project/2026/2027_cvpr/vjepa2/evals/analysis_vlm/modes/` (8 files) + 5 configs under `/data/hyuntak/project/2026/2027_cvpr/vjepa2/configs/analysis/blender_toy_dataset/`
- Data: `/data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/blender_toy/{velocity,acceleration,combined}_{train,val}.csv`, `blender_targets.npy` (672,4)

---

# Appendix — Verified experiment specs (from paper, adversarially checked)

## attn-distance  (CONFIRMED)
- **goal**: Show that the Physics Emergence Zone (~one-third depth) coincides with a mechanistic change in the attention pattern of the V-JEPA 2-L encoder: the layer-average attention head distance DROPS while head specialization (spread of distances across heads) SPIKES, because unusually spatiotemporally-local attention heads emerge alongside the pre-existing long-range heads. This localized-attention emergence is the shared circuit substrate hypothesized to support both direction decoding and possible-vs-impossible (IntPhys) physics judgments.
- **metric**: Per-layer average attention head distance Dbar^{(l)} in patches (Fig. 19 red / Fig. 3 per-head cells), and per-layer head specialization S^{(l)} in patches (Fig. 19 blue), defined on the attention-weighted spatial distance ds. Table 4 additionally uses downstream direction R2, IntPhys accuracy, per-patch direction R2, and ImageNet accuracy as ablation metrics.
- **expected**: Qualitative (paper's claim, Fig. 19 caption): in the Physics Emergence Zone the average attention head distance DROPS and head specialization SPIKES as spatiotemporally-local heads emerge among the longer-distance heads. Quantitative curve values read from Fig. 19 (approximate): red Attention Distance starts ~8.1 patches at layer 0, falls sharply to a minimum ~4.9 at layers 5-6, stays low (~5-6) across layers 5-10, then rises to a peak ~9.2 at layer 22 (ending ~8.0 at layer 23). Blue Head Specialization is ~0.3 at layer 0, spikes into the emergence zone reaching ~2.7 at layer 5 and ~2.6 at layers 9-10, then declines to a min ~0.8 at layer 22. Fig. 3 per-head heatmap shows individual-head distances spanning ~2 to ~10 patches, with local (low) and long-range (high) heads coexisting specifically around the emergence zone. Causal follow-up (Table 4): local-attention ablation at the emergence zone leaves ImageNet ~unchanged (33.7 -> ~33 across s) but degrades direction and IntPhys; combined s=3,t=1 collapses direction R2 from 0.97 to 0.14, confirming local attention is functionally required for the physics representations.
- **hyperparams**: model=V-JEPA 2-L (ViT-Large encoder); num_layers=24 (indices 0-23); num_heads_per_layer=16 (indices 0-15); input_frames=16; tubelet_temporal_stride=2 (16 frames -> 8 temporal tokens); spatial_grid=14x14 = 196 patches; patch_size=16x16 pixels; image_resolution=224x224; total_tokens=T x N = 8 x 196 = 1568; max_spatial_distance=~18 patches (14x14 diagonal); temporal_distance_range=0 to 7 tubelets; physics_emergence_zone=~one-third depth, layer ~8 (direction per-patch transition at Layer 7->8); ablation_spatial_thresholds_s (Table 4 only)={1,3,5,7,9,11,13} patches; ablation_temporal_thresholds_t (Table 4 only)={1,2,3,4,5,6} tubelets
- **verify corrections**: Fig. 3 per-head cell values actually span roughly 1.3 to 10.3 patches (visible cells: 1.3, 1.4, 1.5, 1.6, 1.7 at the low end; 10.1, 10.3 at the high end). The spec's stated '~2 to ~10 patches' tracks the colorbar tick range, not the true cell min/max; the darkest local heads dip to ~1.3, below the spec's ~2. | Minor scope note: per Section 6.3 (p.6) the causal ablation in Table 4/Table 2 is applied 'exclusively in the Physics Emergence Zone' layers, not globally. The spec's Table-4 method step describes the row-masking correctly but omits that it is restricted to emergence-zone layers (the expected_result field does say 'at the emergence zone', so this is only a completeness gap in the method step). | Fig. 19 left-axis (red) range: spec says '~5-9 patches' and Dbar 'ranging ~5-9'; the actual red curve slightly exceeds this on both ends (min ~4.9 at layers 5-6, peak ~9.2 at layer 22). Cosmetic, not a contradiction.

## attn-ablation  (CONFIRMED)
- **goal**: Test whether spatiotemporally LOCAL attention is causally responsible for physics encoding in V-JEPA 2. By zeroing (and renormalizing) attention between tokens whose spatial and/or temporal distance falls at or below a threshold, and measuring how four downstream capabilities degrade, the experiment localizes which capability depends on which attention scale. Headline phenomenon: spatial-local ablation barely touches global direction R2 but wipes out per-patch (retinotopic) direction localization; temporal-local ablation strongly hurts both direction and IntPhys (intuitive physics); combined spatiotemporal ablation destroys direction encoding entirely, while IntPhys and ImageNet decay only gradually — evidence that local heads emerging at the Physics Emergence Zone (PEZ) underpin spatiotemporal physical processing.
- **metric**: Four metrics reported per ablation condition: (1) Direction R2 (global/mean-pooled direction prediction on synthetic velocity data); (2) IntPhys Acc (%) possible-vs-impossible; (3) Per-patch R2 (per-patch direction decoding, can go below 0 = worse than mean predictor, shown as '<0'); (4) ImageNet Acc (%) top-1. Degradation from the s=0,t=0 baseline is the read-out; larger drop = greater reliance on local attention at that spatial/temporal scale.
- **expected**: Table 4 (V-JEPA 2-L), columns [Direction R2 | IntPhys Acc | Per-patch R2 | ImageNet Acc]. BASELINE s=0,t=0: 0.97 | 78.3 | 0.72 | 33.7. SPATIAL-ONLY (temporal preserved): s=1: 0.96|71.4|0.65|33.8; s=3: 0.95|67.2|0.53|33.3; s=5: 0.95|63.9|0.43|33.5; s=7: 0.93|62.2|0.30|33.5; s=9: 0.92|60.6|0.14|33.8; s=11: 0.91|61.1|<0|33.7; s=13: 0.88|60.8|<0|33.9. TEMPORAL-ONLY (spatial preserved): t=1: 0.94|76.4|0.64|33.3; t=2: 0.85|60.6|0.48|33.0; t=3: 0.83|51.9|0.41|30.3; t=4: 0.82|50.6|0.36|28.0; t=5: 0.81|50.8|0.29|27.0; t=6: 0.80|50.8|0.24|25.6. SPATIOTEMPORAL (both knocked out): s=3,t=1: 0.14|61.7|<0|33.1; s=5,t=2: <0|60.3|<0|31.8; s=7,t=3: <0|56.4|<0|29.7; s=9,t=4: <0|56.7|<0|27.3; s=11,t=5: <0|58.1|<0|19.5; s=13,t=6: <0|50.8|<0|11.2. QUALITATIVE SIGNATURES: (a) SPATIAL-ONLY minimally affects global direction R2 (0.97 -> only 0.88 at s=13) and leaves ImageNet essentially flat (~33.3-33.9 throughout), but strongly degrades per-patch direction localization (0.72 -> 0.14 at s=9, <0 by s>=11); IntPhys drops moderately (78.3 -> ~60). (b) TEMPORAL-ONLY strongly hurts BOTH global direction R2 (0.97 -> 0.80) and IntPhys (78.3 -> 50.8, near chance/floor by t>=3), also lowers per-patch R2 (0.72 -> 0.24) and ImageNet (33.7 -> 25.6). (c) COMBINED destroys direction encoding entirely (R2 collapses to 0.14 at the mildest s=3,t=1 and goes <0 for all stronger pairs) and per-patch R2 is <0 throughout; IntPhys and ImageNet degrade only gradually (IntPhys ~50-62, ImageNet 33.1 down to 11.2 at the strongest s=13,t=6). Interpretation: global direction relies on non-local (long-range) spatial attention but on local temporal attention; per-patch/retinotopic direction depends on local spatial attention; IntPhys depends on local temporal attention; ImageNet (static) is robust except to the strongest combined ablation. Note the main-text Table 2 subset matches exactly: BASE 0.97/78.3/33.7, SPATIAL s=7 0.93/62.2/33.5, TEMPORAL t=3 0.83/51.9/30.3, COMBINED s=3,t=1 0.14/61.7/33.1.
- **hyperparams**: model=V-JEPA 2-L (ViT-L video encoder); input_frames=16; tubelet_temporal_stride=2 (16 frames -> 8 temporal tokens; each tubelet = 2 frames); temporal_tokens_T=8 (tubelet index 0..7); spatial_patches_N=196 (14x14 grid); patch_size=16x16; image_size=224x224; sequence_length=1568 tokens (T x N = 8 x 196); spatial_distance_ds=Euclidean in patch coords, units=patches, max ~18 (sqrt(2)*13 diagonal); temporal_distance_dt=|t_i - t_j| tubelets, range 0..7; masking_rule=zero attention where ds(q,k)<=s (spatial) and/or dt(q,k)<=t (temporal), then renormalize surviving weights to sum to 1; spatial_thresholds_s={1, 3, 5, 7, 9, 11, 13} patches (spatial-only regime); temporal_thresholds_t={1, 2, 3, 4, 5, 6} tubelets (temporal-only regime); combined_paired_thresholds=(3,1),(5,2),(7,3),(9,4),(11,5),(13,6) as (s,t); baseline_condition=s=0, t=0 (no masking); ablation_layer_scope=Physics Emergence Zone layers only (~one-third depth; ~Layer 8 for V-JEPA 2-L) — not all layers
- **verify corrections**: Spatial max-distance: spec writes 'max ~= 18.4 = sqrt(2)*13'; the PDF only states 'maximum ~18 on the diagonal' (p.27). The derivation is consistent, but report it as the paper's '~18' rather than the more precise 18.4 to avoid implying the paper stated it. | Model attribution: the C.6 text and the Table 4 caption say only 'V-JEPA v2' / name no variant; the 'V-JEPA 2-L (ViT-L)' label is inferred from Fig 18 (same synthetic-velocity dataset, explicitly V-JEPA 2-L) and from the Layer 7->8 transition being one-third of a 24-layer ViT-L. Well-supported but not stated in C.6/Table 4 itself; keep it labeled as inferred. | Layer scope of Table 4: the PEZ-exclusive scope is stated only in main-text Sec 6.3 and implied by the Table 2 caption's pointer to Table 4; Appendix C.6 does not restate it and never enumerates the exact ablated layer index/band. Treat '~Layer 8 only' as an assumption, not a specified hyperparameter.

## ortho-probe-sequence  (PARTIAL)
- **goal**: Measure the effective feature dimensionality with which a video world model (V-JEPA 2-L) encodes a physical variable, by counting how many mutually orthogonal linear directions in a layer's activation space independently carry that variable. K probes are trained sequentially, each on activations with all previously-found probe directions projected out (deflation), until decoding drops to chance. K (or 2K for the 2D direction output) is the reported subspace dimensionality. Demonstrates that physics variables are high-dimensional/redundant (tens of independent features), that direction is much higher-dim than speed which is higher than IntPhys, and (via C.12) that this subspace causally controls the variable.
- **metric**: "Number of orthogonal probes K trainable before decoding falls to chance = effective subspace dimensionality (2K for direction's 2D output; K for speed/IntPhys 1D output). Per-probe decode quality measured by: Direction — R^2 and circular (angular) MAE in degrees; Speed — R^2 and MAE (vs random baseline); IntPhys — accuracy and AUC. Reported per layer 0–23."
- **expected**: "Physical variables occupy tens of independent dimensions (redundant/high-dimensional coding). Across layers 0–23 (C.11 text): direction subspaces of dimension 14–136, speed subspaces 16–31, IntPhys subspaces 1–15. Table 3 (reported layer subset) gives direction 66–136, speed 21–29, IntPhys 7–15 dims. Dimensionality rises with depth (Fig 22): direction climbs from ~0–2 probes at layer 0 to ~80 probes by the last layer; speed stays ~15–30; IntPhys ~0–15. Direction shows a jagged 'sawtooth' redundancy curve (Fig 23, accuracy-within-15°) while speed decays smoothly (Fig 23 R^2). Direction ≫ speed ≫ IntPhys in dimensionality. (C.12 corollary: steering with ~20 orthogonal direction probes moves a held-out probe's read from MAE 82.9° baseline to ≈11.9° to target at layer 8.)"
- **hyperparams**: optimizer=Adam; learning_rate (η)=1e-3; weight_decay (λ)=1e-4; epochs (direction)=100; epochs (speed)=50; epochs (IntPhys)=50; train/test split=80/20, fixed random seed; embedding dim d=1024 (V-JEPA 2-L); probe type=single-layer linear (no hidden layers, bias present); direction output=2D (sin θ, cos θ), circular regression, MSE; speed output=1D scalar, linear regression, MSE; IntPhys output=1D logit, binary logistic regression, cross-entropy; orthogonalization=QR decomposition of probe weights -> Q_k; deflation X(k+1)=X(k)-X(k)Q_kQ_k^T; STOP threshold — Direction (C.11 text)=R^2 < 0.1  OR  circular MAE > 80° (chance ≈ 90°); STOP threshold — Speed (C.11 text)=R^2 < 0.05  OR  MAE > 90% of random baseline; STOP threshold — IntPhys (C.11 text)=accuracy < 55%  OR  AUC < 0.55 (chance = 50%); STOP threshold — Direction (Fig 22 caption, CONFLICTS)=R^2 < 0.3; STOP threshold — Speed (Fig 22 caption, CONFLICTS)=R^2 < 0.1; STOP threshold — IntPhys (Fig 22 caption)=accuracy < 55%; layers analyzed=0–23 (all 24 encoder layers of V-JEPA 2-L)
- **verify corrections**: Table 3 is on printed page 26, not p.25. Page 25 (C.4) only DISCUSSES it. Fix figure_or_table 'Table 3 ... p.25' -> p.26 (C.4 text p.25). | 'probe type: single-layer linear (no hidden layers, bias present)' — the 'bias present' clause is invented. The PDF (C.11 Probe Architecture) says only 'All probes are single-layer linear models'; it never states whether a bias term is used. Drop 'bias present' or mark it as an assumption. | The expected_result / uncertainty attribute 'Table 3 gives direction 66-136, speed 21-29, IntPhys 7-15' but these are the C.4 TEXT ranges (p.25: 'Direction subspace (k = 66-136) ... Speed subspace (k = 21-29) ... IntPhys subspace (k = 7-15)'), NOT the literal min-max of Table 3's columns. Table 3's raw Dir Dim column spans 14-400: layers 20-23 all report Dir Dim = 400 (an apparent probe/dim cap the spec never mentions). Its raw IP Dim column reaches 32 at layer 23 (and 19 at layer 21), and its Spd Dim column spans 16-31. So the stated upper bounds (136 dir, 15 IntPhys) are exceeded by the actual table at late layers. | The uncertainty note 'Table 3 covers only a subset of layers' is factually wrong: Table 3 lists ALL 24 layers (0-23). It is the STATED dimension ranges (C.11's 14-136/16-31/1-15 and C.4's 66-136/21-29/7-15) that are approximations/subsets, and they UNDER-report the late-layer values (dir=400, IntPhys=32), not over-report.

## steering  (CONFIRMED)
- **goal**: Demonstrate that the motion-direction variable in a frozen video encoder is encoded as a HIGH-dimensional (distributed) population code that can only be causally controlled (steered toward a target angle theta*) by a COORDINATED intervention across many tens of orthogonal probe directions — not a single/low-rank direction. This directly contrasts with low-dimensional activation steering in language models (where even complex behaviors like refusal move along one or a few directions). The experiment causally validates that the K-probe orthogonal direction subspace controls direction decoding, using a strict held-out probe so the effect is a true generalization, not probe-overfitting.
- **metric**: circular mean angular error (degrees) of held-out direction probe to target theta*
- **expected**: At layer 8 of V-JEPA 2-L, steering 8-direction videos toward theta*=90: BASELINE (no steering) held-out probe MAE = 4.9 deg to ground truth but 82.9 deg to target. AFTER steering with 20 train-set probes: held-out probe MAE = 11.9 deg to target — a ~71 deg improvement over the 82.9 deg baseline, and this generalizes to a probe trained on entirely disjoint data. Sweep (Fig. 24): 1-5 probes give only modest improvement (MAE > 50 deg); ~20 probes reach MAE ~ 12 deg; steering across ALL probe directions reaches < 0.5 deg error to target, versus > 80 deg for a single-probe intervention. The MAE-to-target and MAE-to-true-label curves cross as N grows (target-error falling from ~78 to ~12, true-label-error rising from ~5 to ~65), confirming a genuine shift of the representation rather than injection into a separate subspace. KEY FINDING: effective steering requires coordinated intervention across many tens of orthogonal dimensions (direction subspace is ~40-50 dims at the Physics Emergence Zone, up to ~80 near output; across layers 0-23 direction subspaces span 14-136 dims), unlike low-dimensional / single-direction steering in language models.
- **hyperparams**: layer (reported result)=8 (of V-JEPA 2-L, blocks 0-23); embedding dim d=1024; target angle theta*=90 deg (single fixed target); source motion directions=8 discrete: 0, 45, 90, 135, 180, 225, 270, 315 deg; required angular shift=0 to 180 deg, average ~90 deg; train/test split (C.12 steering)=70% / 30% = 240 / 103 videos (total 343); steering probe count K (max)=25 probes, trained until R^2 < 0.1; probe-count sweep N (Fig. 24)=1 to ~20-25 (x-axis 0-20 shown); held-out eval probe fit quality=R^2 = 0.99 (on test set); probe type=single-layer linear circular-regression, output (sin theta, cos theta), MSE loss; probe optimizer (C.11)=Adam; probe learning rate (C.11)=1e-3; probe weight decay (C.11)=1e-4; probe epochs, direction (C.11)=100; direction stopping criterion=R^2 < 0.1 OR circular MAE > 80 deg (chance ~ 90 deg)
- **verify corrections**: Figure 24 title is paraphrased, not verbatim. The spec quotes it as 'Steering Generalizes to Held-Out Probe (Layer 8, V-JEPA v2-L)'. The actual caption reads: 'Figure 24. Steering generalizes to held-out data and probes.' The '(Layer 8, V-JEPA 2-L)' info appears in the caption body, not the title. | expected_result's per-N plot numbers 'true-label-error rising from ~5 to ~65' and 'target falling from ~78 to ~12' are only partially textual. The paper states only baseline target MAE=82.9, baseline true-label MAE=4.9 (~5 is fine), and post-20-probe target=11.9, plus the qualitative 'as MAE to the target decreases, MAE to the true labels increases correspondingly.' The true-label endpoint '~65' and target start '~78' are plot estimates, not stated numerically — treat as read-from-figure, not paper facts (the spec's uncertainties note flags this, keep it). | Table 3 characterization: spec labels it '(per-layer subspace dimensions).' Its actual title is 'Subspace overlap between motion encoding (direction/speed) and IntPhys probes (V-JEPA v2-L, d=1024)'; per-layer subspace dimensions are columns within it (Dir Dim, IP Dim, Spd Dim). Note the paper-internal tension the spec should be aware of: Table 3 lists layer-8 direction Dim = 136 (and C.9 text says 'direction subspace is high-dimensional (66–136 dimensions)'), whereas the main text / Fig. 22 say direction decoding needs 'roughly 40–50 features … up to 80 near the output.' The spec faithfully reports both the '40–50/up-to-80' and '14–136' numbers, but they coexist inconsistently in the paper (Fig. 22 counts K probes at a different threshold R^2<0.3, C.11 uses R^2<0.1, Table 3 reports dimension≈2K). | The explicit stacked A=[W_1 V;…;W_K V], b=[t*−W_k x_perp], c*=argmin||Ac−b|| normal-equation form in Method Step 2b/equations is the spec's own construction. The paper only writes 'Solve for target coordinates c* via least squares such that all probes predict θ*.' Keep it flagged (spec already does) as a natural reading, not verbatim; probe bias terms and target normalization remain unspecified in the PDF.

## direction-circular-geometry  (CONFIRMED)
- **goal**: Demonstrate that at the Physics Emergence Zone (~one-third depth; layer 8 for V-JEPA 2-L, 24 blocks) motion DIRECTION is encoded as a CIRCULAR population code: individual MLP neurons show smooth sinusoidal (sin/cos) tuning to ground-truth direction theta, and the population of per-neuron tuning vectors (beta_cos, beta_sin) tiles the full 360-degree unit circle. This circular geometry is ABSENT/sporadic at Layer 0 and emerges sharply at the transition (Fig 20). Direction is further shown to be a high-dimensional 'sawtooth' sin-cosine code via successive orthogonalization (Fig 4c, Fig 23 left), a redundancy signature that is UNIQUE to direction and NOT present for SPEED (Fig 23 right, smooth monotonic decay; and no circular geometry for speed, Fig 21). Speed is instead modeled with a quadratic (preferred-speed) tuning curve, not a sinusoidal/circular one.
- **metric**: Primary tuning metric: cross-validated Delta R^2 (k=5) of the per-neuron GLM (direction sin/cos GLM; speed quadratic GLM). Population descriptors: preferred direction PD=arctan2(beta_sin,beta_cos), direction gain=sqrt(beta_cos^2+beta_sin^2); preferred speed r*=-beta_r/(2 beta_r2), speed gain=|beta_r|. Redundancy/sawtooth curves: Direction = 'Accuracy within 15 degrees' vs orthogonal-probe number; Speed = 'Validation R^2' vs orthogonal-probe number. Dimensionality = number of orthogonal probes K before chance (effective dim = 2K for direction, K for speed/IntPhys).
- **expected**: Direction: at the Physics Emergence Zone, MLP units are strongly direction-tuned (high GLM R^2) with preferred directions tiling 360 deg; the population of per-neuron (beta_cos,beta_sin) vectors forms a unit-circle/ring geometry. Fig 20: Layer 0 tuning vectors are sporadic, short and clustered (disorganized, do not cover all angles) whereas Layer 8 vectors densely and uniformly fan out over all 360 deg (organized circular population code) - the geometry is absent early and emerges sharply at the transition. Sawtooth (Fig 4c / Fig 23 left): direction 'Accuracy within 15 deg' oscillates jaggedly between ~100% and ~40% across ~100 successive orthogonalizations while decaying - the signature of paired sine-cosine features. Speed (Fig 23 right): 'Validation R^2' decays smoothly/monotonically from ~95% to ~10% over ~28 probes with NO sawtooth; and the preferred-speed heatmap (Fig 21) shows no ring geometry - a qualitative direction-vs-speed difference. Dimensionality (Fig 22, V-JEPA 2-L): direction jumps from ~5 pre-zone to ~44 at layer fraction ~0.35, ~40-50 features at the zone, rising to ~80 near the output layers; speed stays roughly ~15-31 (flat); IntPhys ~1-15. Reported subspace-dimension ranges across layers 0-23 (Table 3, C.11): direction 14-136, speed 16-31, IntPhys 1-15. IntPhys (possible-vs-impossible) needs ~20 independent features at the zone.
- **hyperparams**: GLM basis (direction)=sinusoidal: [cos(theta), sin(theta)] (Eq 4); GLM basis (speed)=quadratic: [r, r^2] (Eq 6); CV folds k (both GLMs)=5; GLM regularization=ridge, alpha = 1e-3; theta range=[-pi, pi] radians; PD bins (direction heatmap)=24 bins spanning [-pi, pi]; preferred-speed bins (speed heatmap)=24 bins spanning 1st-99th percentile of observed preferred speeds; position aggregation (heatmaps)=MAX Delta R^2 across spatiotemporal positions within a bin; neuron sort (heatmaps)=by peak preferred bin, then by tuning strength within bin; embedding dim d=1024 (V-JEPA 2-L); orthogonal-probe optimizer=Adam, lr eta = 1e-3, weight decay lambda = 1e-4 (C.11); orthogonal-probe epochs=100 (direction) / 50 (speed, IntPhys) (C.11); orthogonal-probe train/test split=80/20, fixed seed (C.11); subspace projection=QR decomposition of stacked probe weights; X^(k+1)=X^(k)-X^(k) Q Q^T; stopping thresholds (C.11 stated)=Direction: R^2<0.1 or circular MAE>80deg (chance ~90deg); Speed: R^2<0.05 or MAE>90% of random baseline; IntPhys: acc<55% or AUC<0.55; stopping thresholds (Fig 22 caption stated)=Direction R^2<0.3; Speed R^2<0.1; IntPhys accuracy<55% (INCONSISTENT with C.11 - see uncertainties); emergence-zone / analyzed layer=~one-third depth; Layer 8 for V-JEPA 2-L (compared against Layer 0); MLP site analyzed=MLP fc1/fc2 units at end of Physics Emergence Zone
- **verify corrections**: Fig 22 IntPhys curve reading: spec's expected_result reports 'IntPhys ~1-15' as the Fig 22 dimensionality. The plotted IntPhys curve (page 31) actually rises to ~30 near layer fraction 1.0, and Table 3 (page 25) IP-Dim reaches 32 at layer 23 (19 at layer 21). The '1-15' figure is only the range STATED in C.11 text; it is contradicted by the paper's own Fig 22 and Table 3. | Direction subspace range attribution: spec's expected_result says 'Reported subspace-dimension ranges ... (Table 3, C.11): direction 14-136'. C.11 text does say 14-136, but Table 3 itself (page 25) has Direction-Dim = 400 at layers 20, 21, 22, 23 (and Dir=14 min at layer 2), i.e. actual Table 3 range is 14-400, not 14-136. Also note C.4 (page 25) states the direction subspace as '66-136 dimensions' and speed as '21-29', both inconsistent with C.11's '14-136' and '16-31'. The spec should flag this three-way internal inconsistency (C.11 14-136 vs C.4 66-136 vs Table 3 up to 400), not present 14-136 as the Table 3 value. | Fig 22 y-axis semantics: spec says the plotted quantity is 'effective feature dimensionality' and 'for direction the effective dimensionality is 2K'. Fig 22's y-axis (page 31) is literally labeled 'Number of Orthogonal Probes' (= K), and C.11 defines effective dim = 2K only in text. So the direction curve values (~44 at zone, ~80 at output) are K (number of probes), matching Section 7.2's '40-50 ... up to 80', NOT 2K. These do not reconcile with Table 3's Dir-Dim of 400, another paper-internal inconsistency worth noting. | Orthogonal-probe projection hyperparam: spec's 'subspace projection: QR decomposition of stacked probe weights' is imprecise for the per-step C.11 projection. C.11 step 2-3 (page 32) forms Q_k from a SINGLE probe's weights W_k each iteration (X^(k+1)=X^(k)-X^(k)Q_kQ_k^T). Stacking ALL probe weights [W_1^T,...,W_K^T] and applying QR is the separate C.12 STEERING construction (Eq 8, page 32), not the redundancy-sweep projection step.

## tasks-metrics-hparams  (CONFIRMED)
- **goal**: Demonstrate that physical information (possible-vs-impossible "IntPhys" discrimination and motion direction) emerges at a consistent "Physics Emergence Zone" ~one-third through frozen video encoders (V-JEPA 2 L/H/G, VideoMAE-v2), that this is NOT a generic depth effect (control tasks: ImageNet classification, CLEVRER object counting, SSv2 action recognition do not show the one-third signature), and that the zone's spatiotemporal computation is carried by local attention heads (attention-ablation table with four reused metrics). This spec pins down (1) the linear-probe training protocol behind the main layer-wise R2/accuracy curves, (2) how the control-task probes are set up, and (3) precise, implementable definitions of the four metrics reused in the attention-ablation tables (Table 2 main body, Table 4 appendix).
- **metric**: Four metrics reused in the attention-ablation tables: (1) Direction R2 = coefficient of determination of a mean-pooled linear circular-regression probe predicting (sin theta, cos theta) [baseline 0.97]; (2) IntPhys Accuracy = test accuracy (%) of a mean-pooled linear binary-logistic possible-vs-impossible probe, chance 50% [baseline 78.3%]; (3) Per-patch Direction R2 = R2 of a per-patch (no-pooling) linear direction probe [baseline 0.72, may go <0]; (4) ImageNet Accuracy = top-1 accuracy (%) of a linear ImageNet-classification probe [baseline 33.7%]. Layer-wise main results report probe R2 (regression targets) and accuracy (classification targets) as mean +/- std over 5 grouped folds, best-of-20-config sweep.
- **expected**: MAIN CURVES: IntPhys probe accuracy jumps sharply from ~chance (~50%) to ~85-95% at ~one-third depth ('Physics Emergence Zone'), consistent across V-JEPA 2 L/H/G (and a weaker version in VideoMAE-v2-G); accuracy peaks in the MIDDLE third and degrades toward the output. Direction R2 is low early and rises sharply to ~0.97 at the emergence zone (V-JEPA 2-L layer 8), while speed and acceleration MAGNITUDE are decodable with high R2 from EARLY layers; direction is the variable whose emergence marks the zone. CONTROL TASKS (Fig 12, V-JEPA 2-L, Test Acc vs Layer Fraction): IntPhys (~40%->~100% with sharp ~0.3 jump) and Shuffled-Video (~73%->~100%, emergence-like) show the one-third signature; CLEVRER Count (~0%->~80%), ImageNet (~0%->~37%), and SSv2 (~5%->~37%) rise only monotonically with depth (NO one-third signature) -> zone is not a generic depth effect. ATTENTION ABLATION -- Table 2 (emergence zone, V-JEPA 2-L): BASE 0.97/78.3/33.7; SPATIAL s=7 -> 0.93/62.2/33.5; TEMPORAL t=3 -> 0.83/51.9/30.3; COMBINED s=3,t=1 -> 0.14/61.7/33.1 (columns = Dir R2 / IntPhys% / ImageNet%). Table 4 (full sweep, columns Dir R2 / IntPhys Acc / Per-patch R2 / ImageNet Acc): Baseline s=0,t=0 = 0.97/78.3/0.72/33.7. Spatial-only degrades per-patch R2 sharply (0.72->0.65->0.53->0.30->0.14->'<0'->'<0' for s=1,3,5,7,9,11,13) while barely touching direction R2 (0.96->0.88) and holding ImageNet ~33-34%. Temporal-only strongly hurts direction (0.94->0.80) and IntPhys (76.4->50.8 = chance) and starts hurting ImageNet at large t (33.3->25.6). Combined destroys direction (0.14 then '<0') and per-patch ('<0'), with IntPhys ~50-62% and ImageNet degrading gradually (33.1->11.2 by s=13,t=6). Interpretation: direction encoding depends on LOCAL (esp. temporal) attention in the zone; ImageNet (static) is largely unaffected until extreme combined ablation.
- **hyperparams**: probe_learning_rates (sweep)={1e-4, 3e-4, 1e-3, 3e-3, 5e-3}; probe_weight_decay (sweep)={0.01, 0.1, 0.4, 0.8}; total_sweep_configs=20 (5 lr x 4 wd); model_selection=best on validation performance; cross_validation=5-fold grouped CV; report mean +/- std across folds; probe_form=linear f(h_l)=W h_l + b on spatiotemporally mean-pooled activations; direction_probe_head=circular regression, 2 outputs (sin theta, cos theta), MSE loss; speed/accel_probe_head=scalar linear regression, MSE loss; intphys_probe_head=binary logistic regression, cross-entropy loss; V-JEPA2-L_layers=24 (indices 0-23), d=1024; emergence_zone_layer (V-JEPA2-L)=~layer 8 (one-third of 24); direction analyses & steering use layer 8; tokenization=16 frames -> 8 tubelets (temporal stride 2); 14x14=196 spatial; 1568 tokens; 224x224 input, 16x16 patches; ablation_spatial_thresholds_s={1,3,5,7,9,11,13} patches; ablation_temporal_thresholds_t={1,2,3,4,5,6} tubelets; ablation_combined_pairs=(s,t) in {(3,1),(5,2),(7,3),(9,4),(11,5),(13,6)}; velocity_dataset=392 videos = 8 dir x 7 speed x 7 start-pos; 16 frames @24fps; 256x256; acceleration_dataset=280 videos = 8 dir x 5 accel x 7 start-pos; identical resolution
- **verify corrections**: expected_result mis-transcribes the per-patch (Per-patch R2) column of Table 4 for the SPATIAL-only sweep: it writes '0.72->0.65->0.53->0.30->0.14->"<0"->"<0" for s=1,3,5,7,9,11,13', dropping the s=5 value and folding the s=0 baseline into the sweep. The correct Table 4 values are: baseline s=0 = 0.72; then s=1,3,5,7,9,11,13 = 0.65, 0.53, 0.43, 0.30, 0.14, <0, <0 (the 0.43 at s=5 is missing from the spec). | The three probe-head definitions (circular (sin,cos) regression + MSE; scalar regression + MSE; binary logistic + cross-entropy) are stated only in App. C.11 (the orthogonal-probe-sequence experiment), NOT in App. B. App. B literally states only 'linear probes of the form f(h_l)=W h_l + b on spatiotemporally pooled activations'. The spec's method attributes these heads to the 'MAIN LINEAR PROBE (protocol behind ALL layer-wise curves, App. B)', which is a reasonable inference but is not explicitly stated for the main sweep; the spec should note the head form is imported from C.11. | expected_result calls VideoMAE-v2-G 'a weaker version' of the emergence. The paper (p.4) says VideoMAE-v2-G 'also exhibits a similar depth-dependent transition' and reserves 'fail to exhibit reliable emergence' for the SMALLER VideoMAE-v2 variants, not G. Minor characterization mismatch.
