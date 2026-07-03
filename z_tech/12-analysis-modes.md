# 12 — Analysis modes & reproduction roadmap

> An additive, config-driven post-hoc analysis layer (`evals/analysis_vlm/modes/`) bolted onto the frozen-encoder probing harness — one mode (`attention_distance`) is implemented and has already reproduced the paper's Figure 3 attention-locality heatmap on the Blender toy set; four more are specified but pending.

## Purpose

`evals/analysis_vlm/modes/` hosts paper-reproduction analyses of the **frozen** V-JEPA2 encoder
(attention-distance / ablation, orthogonal-probe steering, circular direction geometry) behind a single
config selector `experiment.analysis.modes`. It sits on top of — and reuses — the existing `analysis_vlm`
layer-wise probing harness rather than re-implementing feature extraction, standardization, or plotting.

The design invariant is strict: **when `experiment.analysis.modes` is absent, the harness is byte-for-byte
identical to before** — the `modes` package is never even imported, and the base training/eval/plot path
is unchanged.

Exactly one mode is implemented today: **`attention_distance`** (Appendix C.6 / Fig. 3 & Fig. 19). It has
been **run and reproduces Fig. 3** on the Blender velocity set (see [Reproduced result](#reproduced-result-fig-3)).
The remaining phases (2–5) are specified but pending; the full plan lives in
`evals/analysis_vlm/modes/REPRODUCTION_PLAN.md`.

> Scope note: this section covers the **modes subpackage** and its two seams into `eval.py`. The base
> `analysis_vlm` probing harness (linear probes, `cache_pooling`, target standardization, `pre_norm`) is
> [section 02](02-analysis-vlm-harness.md); the SDPA capture/ablation primitive it reuses is
> [section 11](11-attention-hooks.md). Cross-refs at the end.

---

## What changed vs upstream V-JEPA2

Against upstream commit `204698b`, the **entire** `evals/analysis/` and `evals/analysis_vlm/` trees are
fork additions — they do not exist upstream (`git ls-tree 204698b -- evals/analysis_vlm/` is empty, and
`git diff 204698b -- evals/analysis_vlm/eval.py` reports the file as **814 insertions, 0 deletions** — a
pure new file). So "vs upstream" for the probing harness means "all of it." The delta **this** subsystem
adds on top of the pre-existing fork harness is small and enumerated below.

| Path | Status | Delta / last commit |
|------|--------|---------------------|
| `evals/analysis_vlm/modes/__init__.py` | **new** (committed `4c76f65`) | Mode registry + `register()` decorator, `AnalysisContext` dataclass, `run_modes()` dispatch. |
| `evals/analysis_vlm/modes/attention_distance.py` | **new** (committed `c296428`) | The one implemented mode: per-(layer,head) attention-weighted spatial/temporal distance → **Fig. 3 heatmap** (`attention_distance.png`, primary) **+ Fig. 19 dual-axis line plot** (`attention_distance_layerwise.png`). |
| `evals/analysis_vlm/modes/REPRODUCTION_PLAN.md` | **new** (committed `4c76f65`) | Phase 0–5 roadmap + consolidated config schema + adversarially-checked paper specs. |
| `evals/analysis/attention_hooks.py` | **new** (committed `4c76f65`) | SDPA monkey-patch context manager + `AttentionDistanceCollector` + ablation-bias builder. **Reused as-is** by the mode, not modified. Documented in [section 11](11-attention-hooks.md). |
| `configs/analysis/blender_toy_dataset/vjepa_attn_distance.yaml` | **new** (committed `c296428`) | The real, in-repo config for the mode. |
| `evals/analysis_vlm/eval.py` | **modified** (last touched `4c76f65`) | **Two additive seams only:** (1) `skip_base_probe` gate at `eval.py:501-507`; (2) the mode-dispatch block at `eval.py:565-590`. Everything else is the pre-existing harness. |
| `z_scripts/run_attn_distance_vjepa.sh` | **new** (untracked — `z_scripts/` is gitignored: `.gitignore:38`) | SLURM launcher for the mode (single GPU on purpose). Lives alongside the other launchers (`run_analysis_vjepa.sh`, `run_analysis_vlm.sh`), all under the gitignored `z_scripts/`. |

**Tracking note:** the `modes/` package, `attention_hooks.py`, the config, and the two `eval.py` seams are
all **committed and the working tree is clean**. The only working-tree-only artifact is the launcher, which
is untracked because the whole `z_scripts/` directory is gitignored — the same as every other run script.

**Exact `eval.py` additive delta.** Because upstream has no `eval.py` at all, the "delta" that matters is
the *two seams* inside the fork's file. Both are guarded so they no-op when `modes` is absent:

```python
# Seam 1 — eval.py:504  (default False => num_probe_epochs == num_epochs => identical loop)
num_probe_epochs = 0 if args_analysis.get("skip_base_probe", False) else num_epochs

# Seam 2 — eval.py:568  (empty for every existing config => block + import skipped)
modes_cfg = args_analysis.get("modes") or {}
if modes_cfg and rank == 0:
    ...
```

**Default-off guarantee.** No `modes` key ⇒ `modes_cfg == {}` ⇒ dispatch block and package import skipped;
`skip_base_probe` absent ⇒ full training loop. `summary.json`, `log_r*.csv`, and `stage_val_acc.png` are
unchanged. `col_mu`/`col_sd` are captured by pure read (no mutation of the existing standardization).

---

## Design & data flow

Three pieces make up the subsystem: a **registry** (`__init__.py`), a read-only **context** dataclass, and
a **dispatch** function — plus the two `eval.py` seams that build the context and call dispatch.

### 1. Registry + lazy import (`evals/analysis_vlm/modes/__init__.py`)

A mode is any callable `fn(cfg: dict, ctx: AnalysisContext) -> None` registered by name via a decorator.

```python
# __init__.py:22-27
def register(name):
    """Decorator: register a mode implementation under `name`."""
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco
```

Imports are deferred (`_import_modes()`, `__init__.py:30-36`) so that
`from evals.analysis_vlm.modes import AnalysisContext` has **no side effects** and never fails on a
half-written future mode file — mode modules self-register only when `run_modes` actually runs. Only
`attention_distance` is wired in today; the other four phases are commented placeholders.

### 2. `AnalysisContext` — the read-only handle passed to every mode (`__init__.py:39-73`)

A dataclass snapshot of `eval.py`'s local scope: the encoder, trained probe heads + metadata,
standardization stats, optional cached features, and two **closures** back into `eval.py` so a mode can run
fresh encoder forwards or build a val clip loader without re-implementing them.

| Field | Meaning |
|-------|---------|
| `encoder`, `device`, `folder`, `rank`, `world_size`, `use_bfloat16`, `plot_pez` | run context; each mode writes under `<folder>/<name>/`. |
| `task`, `num_classes`, `heads`, `best_val`, `stages`, `embed_dims`, `reg_vars` | trained probes + their layer/stage/variable metadata (`reg_vars` = `[(name, [cols…]), …]`). |
| `targets_t` | **standardized** targets on device (indexed by CSV label). |
| `targets_npy` (`tpath`), `col_mu` (`mu`), `col_sd` (`sd`) | path to **raw** targets + the per-column `nanmean`/`nanstd` used to standardize (`eval.py:201-203`) — needed to recover true (unstandardized) angles/speeds. |
| `cache_pooling`, `data_mode` | granularity of cached features; `data_mode == "clip"` for V-JEPA. |
| `tr_feats`/`tr_labels`/`va_feats`/`va_labels` | cached features, **`None` when `cache_features=false`**. |
| `encode_clip(data_batch)` | closure: `_encode(encoder, d, device, data_mode, use_bfloat16)` — one encoder forward returning per-stage `feats`. |
| `make_val_clip_loader()` | closure: builds the val clip `DataLoader` (`workers=0`). |

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

Per-entry config semantics: `True` / `{}` / `None` ⇒ run with defaults; a dict ⇒ options; `False` or
`{enabled: false}` ⇒ skip that mode. Unknown names **raise** with the valid list. Modes run in insertion
(dict) order.

### 4. The two `eval.py` seams

Both are additive and default-off; both are placed **after** the base train/eval/plot loop.

**Seam 1 — `skip_base_probe`** (`eval.py:501-507`). Encoder-only modes (like attention distance) don't need
the layer probes trained, so this flag shortens the train loop to 0 epochs:

```python
# eval.py:504
num_probe_epochs = 0 if args_analysis.get("skip_base_probe", False) else num_epochs
```

Default `False` ⇒ `num_probe_epochs == num_epochs` ⇒ the loop is byte-identical to before.

**Seam 2 — dispatch block** (`eval.py:565-590`), placed after the plotting block, **rank 0 only**:

```python
# eval.py:568-590  (abridged)
modes_cfg = args_analysis.get("modes") or {}
if modes_cfg and rank == 0:
    from evals.analysis_vlm.modes import AnalysisContext, run_modes
    ctx = AnalysisContext(
        encoder=encoder, device=device, folder=folder, rank=rank, world_size=world_size,
        use_bfloat16=use_bfloat16, plot_pez=plot_pez, task=task, num_classes=num_classes,
        heads=heads, best_val=best_val, stages=stages, embed_dims=embed_dims, reg_vars=reg_vars,
        targets_t=targets_t,
        targets_npy=(tpath if task == "regression" else None),
        col_mu=(mu if task == "regression" else None),
        col_sd=(sd if task == "regression" else None),
        cache_pooling=cache_pooling, data_mode=data_mode,
        tr_feats=(tr_feats if cache_features else None), ...,
        encode_clip=lambda d: _encode(encoder, d, device, data_mode, use_bfloat16),   # :585
        make_val_clip_loader=lambda: _split_loader(val_data_path[0], training=False, workers=0)[0],  # :586
    )
    run_modes(modes_cfg, ctx)
if torch.distributed.is_initialized():          # :589-590
    torch.distributed.barrier()
```

`args_analysis.get("modes") or {}` is `{}` for every existing config, so the block (and the mode import)
is skipped entirely. The trailing `barrier()` is a no-op unless DDP is active. **`encode_clip` passes
`data_mode`** (not a hard-coded `"clip"`), so the closure follows whatever the run configured.

---

## Implemented mode: `attention_distance`

`evals/analysis_vlm/modes/attention_distance.py` (Appendix C.6 / Fig. 3 & Fig. 19). It measures, per
(layer, head), the **attention-weighted spatial (patch) and temporal (tubelet) distance** of the frozen
RoPE encoder, captured as a **detached side computation** inside the `attention_hooks` SDPA monkey-patch —
the encoder's actual output is bit-identical to baseline (`attention_hooks.py:46-48`, verified with
`torch.equal`).

### Two outputs

The mode writes **three files** under `<folder>/attention_distance/`:

| File | What it is |
|------|-----------|
| `attention_distance.json` | Raw capture: `spatial_distance[L][H]`, `temporal_distance[L][H]`, `num_layers`, `num_heads`, `rows_per_layer`, and `n_batches`. |
| **`attention_distance.png`** | **PRIMARY — the paper's Fig. 3 heatmap** (`_plot_heatmap`, `attention_distance.py:76-112`). |
| `attention_distance_layerwise.png` | Companion — the Appendix **Fig. 19** dual-axis line plot (`_plot_layerwise`, `attention_distance.py:115-156`). |

**Fig. 3 heatmap (`_plot_heatmap`).** A per-head attention-locality map so the unusually-local heads stand
out:

| Aspect | Spec (`attention_distance.py`) |
|--------|-------------------------------|
| Data | `Z = np.array(out["spatial_distance"]).T` → shape `(H, L)`, rows = head, cols = layer (`:88`). |
| x-axis | **Layer** `0…L-1` (0–23 for ViT-L), integer ticks fontsize 6 (`:96,:98`). |
| y-axis | **Attention Head** `0…H-1` (0–15), integer ticks fontsize 6 (`:97,:99`). |
| Colour | `spatial_distance` in **patches**; `cmap="Blues_r"` so **LOW distance = DARK blue** (local heads pop as dark cells); `origin="lower"` (`:94`). |
| Range | `vmin/vmax = nanmin/nanmax(Z)`; annotation-text threshold `mid = 0.5*(vmin+vmax)` (`:90-91`). |
| Cell labels | `annotate` (default True): `f"{v:.1f}"` in each cell, **white** text on dark cells (`v < mid`) else `#222` on light (`:100-105`). |
| Colorbar | label **"Distance (patches)"** (`:106-107`). |
| Title | **"V-JEPA v2-L: Attention Distance Per Head"** + subtitle `(vjepa \| N val batches)` (`:108`, `:70`). |
| Size / dpi | `figsize=(max(9, L*0.44), max(4.5, H*0.34))`, `dpi=150` (`:93,:110`). |

**Fig. 19 companion (`_plot_layerwise`).** A dual-axis line plot vs **layer fraction** (`layer/(L-1)`):

- **`Dbar` = mean over heads** of per-head distance — red solid line, marker `o`, left axis "attention
  distance (patches)" (`:131,:140-143`).
- **`S` = head specialization = std over the 16 heads** of their per-head distances within a layer (=
  attention-head diversity), computed via `statistics.pstdev` — blue dashed line, marker `s`, right axis
  "head specialization (patches)" (`:132,:146-149`).
- **PEZ shading**: gray `axvspan(pez[0], pez[1])` + "PEZ" label when `ctx.plot_pez` is set (`:135-138`).
- Title "Attention distance & head specialization", `dpi=130` (`:152,:154`).

Both plots use **only `spatial_distance`** — `temporal_distance` is captured to JSON but neither plots it
(heatmap `Z = out["spatial_distance"].T`; layerwise `sd = out["spatial_distance"]`).

### Flow (`attention_distance.py:32-73`)

1. `assert ctx.data_mode == "clip"` — requires the V-JEPA clip encoder (`:34`).
2. `_find_rope_attn(ctx.encoder)` (`attention_hooks.py:239-245`) walks `model.modules()` for all
   `RoPEAttention` blocks (24 for ViT-L); `num_heads` read from block 0 (16) (`:38-42`).
3. Build an `AttentionDistanceCollector(num_layers, num_heads, query_chunk, max_batches)`
   (`attention_hooks.py:161`, `:45-50`).
4. `with attention_hooks(ctx.encoder, collector=collector):` run `ctx.encode_clip(data)` over the first
   `max_batches` val batches (`:54-59`); the patched SDPA streams queries in `query_chunk` blocks so the
   `(B,H,N,N)` matrix is never fully materialized.
5. `collector.finalize()` (`attention_hooks.py:217-225`) returns
   `{spatial_distance, temporal_distance, num_layers, num_heads, rows_per_layer}`; the mode then sets
   **`out["n_batches"] = n`** (`:62`) and dumps `attention_distance.json` (`:63-66`).
6. `_plot_heatmap(...)` → `attention_distance.png` (`:71-72`) and `_plot_layerwise(...)` →
   `attention_distance_layerwise.png` (`:73`), both with `subtitle = "vjepa | {n} val batches"` (`:70`).

**Paper signature to validate:** per-layer `Dbar` **dips** to a minimum and `S` **spikes** around
one-third depth (the Physics Emergence Zone) as spatiotemporally-local heads emerge alongside the
long-range heads. On the heatmap this shows up as a **band of dark (low-distance) cells clustered in the
middle layers**, with early and late layers uniformly light (long-range).

### Reproduced result (Fig. 3)

The mode was **actually run** on the Blender velocity set — single GPU, 10 val batches — and **reproduces
Fig. 3**. Outputs (all three files present):

```
configs/analysis/blender_toy_dataset/logs/analysis_vlm/vjepa-blender-attn_distance/attention_distance/
  ├── attention_distance.json            (24×16, n_batches=10)
  ├── attention_distance.png             (Fig. 3 heatmap)
  └── attention_distance_layerwise.png   (Fig. 19 companion)
```

Verified from the JSON (`num_layers=24`, `num_heads=16`, `n_batches=10`, `rows_per_layer=122304` each):

| Quantity | Observed |
|----------|----------|
| Per-layer `Dbar` | **7.08** at L0 → **dips to ~3.7 minimum** at L6 (3.74) and L9 (3.73) → back up to ~7.7 late (L22). |
| Head specialization `S` | **0.33** at L0 → **spikes to ~2.5** across the middle band (peak **2.54** at L10; 2.38–2.49 at L5/L6/L9) → falls to ~0.8–1.3 late. |
| Per-head spatial distance | spans **~0.10 → ~9.26 patches** overall; local (dark) heads with `d < 1` patch appear only in the middle layers (min-per-layer drops below 0.5 for L4–L11). |
| Early / late layers | **uniformly long-range** — min-per-layer head distance stays ≳4 patches at L0–L1 and L19–L23; no dark cells. |

The shaded PEZ on the layerwise plot is `plot_pez: [0.2, 0.4]` (layer fractions 0.2–0.4 ≈ layers 5–9); the
`Dbar` dip and `S` spike fall squarely in/around that band — the "Physics Emergence Zone" where the
locally-attending heads emerge (roughly L5–L13).

### How to run — `z_scripts/run_attn_distance_vjepa.sh`

A SLURM launcher (`env vjepa2`, config `vjepa_attn_distance.yaml`) runs the reproduction:

```bash
sbatch z_scripts/run_attn_distance_vjepa.sh
# override the config:
sbatch --export=ALL,CONFIG=<path.yaml> z_scripts/run_attn_distance_vjepa.sh
```

Key facts about the launcher:

- **Single GPU on purpose** (`--gres=gpu:1`). The modes dispatch runs on **rank 0 only**
  (`eval.py:569`, `if modes_cfg and rank == 0`), so multi-GPU gives **no speedup** for post-hoc modes.
  Multi-GPU only helps the base probing **sweep / feature cache**, which an encoder-only mode
  (`skip_base_probe: true`, `num_epochs: 0`) doesn't run. Single-GPU is the faithful, simplest setup.
- **Env knobs**: `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` (single-node NCCL safety, harmless at 1 GPU),
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (less CUDA fragmentation on the ~1568-token forwards).
- Resource request: `--cpus-per-gpu=12`, `--mem-per-gpu=80G`, partition `batch_vll`.
- It builds `--devices` from `CUDA_VISIBLE_DEVICES` and calls
  `python -m evals.main --fname "$CONFIG" --devices $DEVICES`.
- **Known cosmetic mismatch:** the header comment says `node : vll4`, but the SBATCH directive is `-w vll6`.
  The directive wins; treat the comment as stale.

---

## Configuration

The mode config clones the working `vjepa_combined.yaml` and adds only the `modes:` block (plus
`skip_base_probe`). The real, in-repo config
`configs/analysis/blender_toy_dataset/vjepa_attn_distance.yaml`:

```yaml
eval_name: analysis_vlm
folder: /data/hyuntak/project/2026/2027_cvpr/vjepa2/configs/analysis/blender_toy_dataset/logs
tag: vjepa-blender-attn_distance         # outputs land under <folder>/analysis_vlm/<tag>/
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
        query_chunk: 512           # stream queries so (B,H,N,N) is never materialized (memory only)
        max_batches: 10            # measure the first 10 val batches (cheap, stable)
        # annotate: true           # (default) per-cell numbers on the heatmap
    probes:
      - { type: linear, pooling: mean, pre_norm: true,
          optimization: { lr: 0.001, weight_decay: 0.1, warmup: 2.0 } }
  data:
    dataset_type: VideoDataset
    resolution: 224                # paper geometry: 14×14 patches × 8 tubelets = 1568 tokens
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
    num_epochs: 0                  # no probe training (encoder-only mode)
    use_bfloat16: true
    cache_features: false          # attention capture runs fresh forwards, not the cache
model_kwargs:
  checkpoint: /data/.../vjepa2-vitl-fpc64-256/.../original/model.pth
  module_name: evals.analysis.modelcustom.vit_encoder_multilayer
  pretrain_kwargs:
    encoder: { checkpoint_key: target_encoder, model_name: vit_large, patch_size: 16,
               tubelet_size: 2, uniform_power: true, use_rope: true }
  wrapper_kwargs: { max_frames: 128, use_pos_embed: false }
```

### Config keys introduced by this subsystem

| Key | Meaning | Default | Allowed values |
|-----|---------|---------|----------------|
| `experiment.analysis.modes` | Mapping `mode → cfg`. **Absent ⇒ `{}` ⇒ dispatch skipped, package never imported.** | absent | `{ <mode>: true \| false \| null \| {…} }` |
| `experiment.analysis.skip_base_probe` | Shorten the base probe loop to 0 epochs for encoder-only modes. | `false` | `true` / `false` |
| `modes.attention_distance` (value) | Run/skip + options. `true`/`{}`/`null` ⇒ run with defaults; `false`/`{enabled:false}` ⇒ skip. | — | bool / dict |
| `modes.attention_distance.query_chunk` | Query-streaming block size. **Memory knob only — result-invariant** (running sums→means). | `512` | positive int |
| `modes.attention_distance.max_batches` | # val batches averaged; caps the number of measured encoder forwards. `null` ⇒ measure all. | `8` (code) — config uses `10` | positive int / `null` |
| `modes.attention_distance.annotate` | Per-cell numeric labels on the Fig. 3 heatmap. | `true` | `true` / `false` |

> `frame_step: 1` + `uniform_sampling: true` is deliberate for the natively-16-frame Blender clips — see
> the frame-sampling gotcha in [section 06](06-data-pipeline-changes.md) / [section 14](14-reproduction-status-and-findings.md).

---

## Invariants & gotchas

- **Default-off is the prime invariant.** No `modes` key ⇒ `modes_cfg == {}` ⇒ dispatch skipped, package
  never imported; `skip_base_probe` absent ⇒ full training loop. `summary.json`, `log_r*.csv`,
  `stage_val_acc.png` unchanged. `col_mu`/`col_sd` are captured by pure read (no mutation of the existing
  standardization at `eval.py:201-203`).
- **`data_mode` must be `clip`** — asserted (`attention_distance.py:34`); VLM/framewise encoders are
  rejected.
- **Fresh forwards, not the cache** — the mode calls the encoder through `encode_clip`; set
  `cache_features: false`. (The attention distance is measured *during* the forward via the SDPA patch, so
  a feature cache would bypass the capture entirely.)
- **Encoder output is bit-identical** with capture on — the collector computes softmax as a detached side
  computation and still returns the original SDPA output (`attention_hooks.py:46-48`,
  `:318-322`).
- **`query_chunk` is result-invariant** — it only controls how many query rows are softmaxed at once; the
  accumulated means are identical. Shrink it if the ~1568-token attention chunk OOMs.
- **Token→(t,x,y) layout** matches `RoPEAttention.separate_positions` exactly (`attention_hooks.py:84-90`,
  `:94-101`); `resolution: 224` gives the paper geometry (14×14×8 = 1568 tokens). A `resolution: 256`
  default would give 16×16 patches — still valid (distance is in patch units) but off the paper geometry.
- **Plots are optional** — if matplotlib/numpy are unavailable each plotter warns and skips rather than
  failing the analysis: heatmap try/except at `attention_distance.py:80-87`, layerwise at `:118-124`. The
  JSON is always written first (`:65-66`), before either plot.
- **`max_batches` default mismatch** — the *code* default is `8` (`attention_distance.py:43`), but the
  in-repo config and the reproduction run both use `10` (JSON `n_batches == 10`). Set it explicitly to be
  reproducible.
- **Single-GPU is not a limitation** — modes run on rank 0 only; the launcher's `--gres=gpu:1` is
  intentional (see [How to run](#how-to-run--z_scriptsrun_attn_distance_vjepash)).

---

## Reproduction roadmap (`REPRODUCTION_PLAN.md`)

The plan collapses three earlier module designs into **one** selector `experiment.analysis.modes` and has
each mode **reuse** existing machinery rather than re-implement. Consolidated schema: one default-off
`modes:` block; **removing it reproduces `vjepa_combined.yaml` behavior exactly.** Optional
`skip_base_probe` shortens the base loop for encoder-only modes.

Planned modes and paper sections (phase numbers follow the plan's **dependency-DAG** ordering, so Phase 5
precedes Phases 2–4 — it only depends on the already-built `attention_hooks` + trained heads, whereas 2–4
need a new `subspace.py`):

| Phase | Mode | Paper | Status | Reuses |
|-------|------|-------|--------|--------|
| 0 | dispatch scaffold (`__init__` + `eval.py` seams) | — | **DONE** | — |
| 1 | `attention_distance` | C.6 / Fig. 3, 19 | **DONE** ✔ reproduced | `attention_hooks.py` (as-is) |
| 5 | `attention_ablation` | C.6 / Table 4 | pending | `attention_hooks` ablation bias; re-eval frozen `ctx.heads` |
| 2 | `orthogonal_probe_sequence` | C.11 | pending | new `subspace.py` (QR deflation) |
| 3 | `steering` | C.12 | pending | `subspace.py` least-squares steering |
| 4 | `direction_tuning` | C.7 / C.8 / C.10 | pending | new `plotting.plot_direction_tuning` |

Dependency DAG: `__init__/dispatch → attention_hooks → {attention_distance, attention_ablation}`;
`subspace.py → {orthogonal_probe_sequence, steering}`; `direction_tuning → plotting`.

Planned per-mode config knobs (from the plan's schema, subject to change until implemented):

- **attention_ablation** — `ablate_layers` (default = `stages`), `spatial: [1,3,5,7,9,11,13]`,
  `temporal: [1,2,3,4,5,6]`, `combined: [[3,1],…]`, `per_patch`. **Requires `cache_features: false`**
  (ablation changes features; a cache would reuse unmasked baselines). Open question: the paper's collapse
  at (s=3,t=1) implies **UNION** semantics, whereas `build_ablation_bias` currently uses **AND** for
  `combined` (`attention_hooks.py:141-143`) — flagged as a knob to add.
- **orthogonal_probe_sequence** — `variable`, `cols`, `max_dims`, `epochs` (100 direction / 50 speed),
  `lr: 0.001`, `weight_decay: 0.0001`, `stop: {r2_below, mae_above_deg}`. **Needs `cache_pooling: tokens`.**
- **steering** — `stage: 8`, `orthogonal_probe: {…}`, `steer: {train_frac: 0.7, thetas_deg, n_sweep,
  coord_from, forward_intervention: false}`. Default `forward_intervention: false` = offline re-decode (the
  validated path). **Needs `cache_pooling: tokens`.**
- **direction_tuning** — `variables`, `contrast`, `site`, `n_bins`, `cv_folds`, `ridge_alpha`,
  `repr_layers`, `sawtooth`. **Needs `cache_pooling: tokens`.**

Two future touched files (not yet applied): `evals/analysis/plotting.py` gains `plot_direction_tuning(...)`
(Phase 4); everything else is new files under `modes/`.

---

## Cross-references

- [11 — Attention hooks](11-attention-hooks.md) — the SDPA capture/ablation primitive
  (`AttentionDistanceCollector`, `attention_hooks`, `build_ablation_bias`) reused verbatim by this mode.
- [02 — analysis_vlm harness](02-analysis-vlm-harness.md) — the base layer-wise probing harness this
  subsystem seams into (`_encode`, `run_one_epoch`, probe heads, standardization).
- [03 — Feature caching & pooling](03-feature-caching-and-pooling.md) — `cache_pooling`
  (`pooled`/`tokens`/`framewise`) that the pending token-level modes require.
- [04 — Probes / regression / NaN-mask](04-probes-regression-nanmask.md) — `pre_norm`, target
  standardization (`mu`/`sd`), and the `reg_vars` metadata carried in `AnalysisContext`.
- [06 — Data pipeline changes](06-data-pipeline-changes.md) — `uniform_sampling` / `frame_step` sampling.
- [07 — Plotting](07-plotting.md) — the base `plot_layer_val_acc` + PEZ shading; future
  `plot_direction_tuning` (Phase 4).
- [09 — Blender toy dataset](09-blender-toy-dataset.md) / [10 — CSV targets](10-datasets-csv-targets.md) —
  the velocity clips and `blender_targets.npy` this mode was run on.
- [13 — Configs reference](13-configs-reference.md) — where `vjepa_attn_distance.yaml` sits among the
  analysis configs.
- [14 — Reproduction status & findings](14-reproduction-status-and-findings.md) — the broader
  Fig-2c/Fig-3 reproduction narrative and the `frame_step` bug.
