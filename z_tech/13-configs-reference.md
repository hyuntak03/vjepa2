# 13 — Config reference

> The complete YAML surface of the `configs/analysis/` probing subsystem: every top-level / `experiment.analysis` / `experiment.data` / `experiment.optimization` key (meaning, default, allowed values), the full config-family map, the `attention_distance` mode's three outputs + JSON schema, and all **three** `z_scripts/` SLURM launchers with the `--export CONFIG=…` override pattern.

## Purpose

This fork adds two **layer-wise probing** eval subsystems on top of stock V-JEPA2, both driven entirely by YAML under `configs/analysis/`. This section is the authoritative reference for that config surface.

The active harness is **`evals/analysis_vlm/eval.py`** (config `eval_name: analysis_vlm`). It supports three model backends (`vjepa`, `llavavideo`, `qwen3vl`) and two tasks (classification, regression). Almost every current YAML targets this harness; only `analysis_TEMPLATE.yaml` documents the older `eval_name: analysis` harness (`evals/analysis/eval.py`), whose key space differs (`layers` instead of `stages`, no `model`/`task`/`regression`/`modes`).

There are exactly **22 `.yaml` files** under `configs/analysis` = **21 experiment configs + 1 template** (`analysis_TEMPLATE.yaml`), across **four families** (root, `InsPhys2/`, `toy_dataset/`, `blender_toy_dataset/`).

---

## What changed vs upstream V-JEPA2

Baseline = commit `204698b`. **Everything below is net-new — no upstream file was modified.** `git diff --stat 204698b` confirms `evals/main.py`, `evals/main_distributed.py`, `evals/scaffold.py` are **unchanged**: routing still happens through the stock `eval_name → evals/<eval_name>/eval.py` scaffold, so no upstream code needed touching. `git cat-file -e 204698b:evals/analysis/attention_hooks.py` fails ("Not a valid object name") — every file below is absent at the baseline.

| Path | Status | What it is |
|---|---|---|
| `evals/analysis_vlm/eval.py` | **new** (814 L) | Unified probing harness; parses every key documented here. Entry `main(args_eval)` at `evals/analysis_vlm/eval.py:85`. |
| `evals/analysis_vlm/{cache,data,loadutil,probes}.py` | **new** | Feature cache, dataloaders (clip + raw), model-dir resolver, probe builders. |
| `evals/analysis_vlm/modelcustom/{llava_video_encoder,qwen3vl_encoder}.py` | **new** | VLM vision-tower wrappers exposing per-layer stages. |
| `evals/analysis_vlm/modes/__init__.py` | **new** (94 L) | Modes **registry** (`@register`), `run_modes()` dispatcher, and the read-only `AnalysisContext` dataclass passed to every mode. |
| `evals/analysis_vlm/modes/attention_distance.py` | **new** (156 L) | The single registered mode today (Phase 1); heatmap + layerwise plots + JSON. |
| `evals/analysis_vlm/modes/REPRODUCTION_PLAN.md` | **new** | The 5-phase paper-reproduction roadmap (blueprint + phase status; see below). |
| `evals/analysis/attention_hooks.py` | **new** (390 L) | Core of `attention_distance`: SDPA monkey-patch, `AttentionDistanceCollector`, `_find_rope_attn`, `build_ablation_bias`. Additive, default-off, torn down on exit (encoder left byte-identical). |
| `evals/analysis/` (eval.py, probes.py, plotting.py, modelcustom/vit_encoder_multilayer.py) | **new** | Older `analysis` harness + the shared V-JEPA multilayer ViT wrapper and probe classes (`build_probe`, `probe_name`) that `analysis_vlm` imports. |
| `configs/analysis/**` | **new** | All 22 YAMLs surveyed here (21 experiment + 1 template). |
| `z_scripts/run_analysis_{vjepa,vlm}.sh`, `run_attn_distance_vjepa.sh` | **new** | Three SLURM launchers (none present upstream). |

**Default-off guarantee.** The `analysis_vlm` harness reuses the shared multilayer wrapper `evals/analysis/modelcustom/vit_encoder_multilayer.py` and the probe classes in `evals/analysis/probes.py` (imported at `evals/analysis_vlm/eval.py:40`). The two `probes.py` files are distinct; the active one is `evals/analysis/probes.py`. The `modes` dispatch block (`eval.py:565–589`) and `skip_base_probe` (`eval.py:504`) are additive: a config **without** `experiment.analysis.modes` yields `modes_cfg == {}`, so the block is skipped, the modes package is **never imported**, and `skip_base_probe` defaults `false` ⇒ `num_probe_epochs == num_epochs` ⇒ the train loop is byte-for-byte identical to a pre-change run.

---

## Design & data flow

### Routing: how a config reaches the harness

`eval_name` is the scaffold key. `eval_name: analysis_vlm` ⇒ `evals/analysis_vlm/eval.py:main()`. Standard CLI overrides from `evals/main.py` still apply (`--val_only`, `--checkpoint`, `--model_name`, `--batch_size`, `--folder --override_config_folder`, `--devices`, `--debugmode`). Output goes to `<folder>/analysis_vlm/<tag>/` → `latest.pt`, `log_r{rank}.csv`, `summary.json`, and (if `plot`) `stage_val_acc.png`; a `modes` run additionally writes `<folder>/analysis_vlm/<tag>/<mode_name>/`.

### Config families

`ls -R configs/analysis` groups into four families. All current configs use `eval_name: analysis_vlm`.

| Family (dir) | Data | Task | Model × variant matrix |
|---|---|---|---|
| `configs/analysis/*.yaml` (root) | `data_csv/R2R_4way_1500*` | **classification** (`num_classes: 4`, R2R 4-way shape/color) + `vjepa_regression` | `{vjepa,llavavideo,qwen3vl}_analysis`, `vjepa_regression`, `analysis_TEMPLATE` (docs the *older* `analysis` harness) |
| `InsPhys2/` | `data_csv/IntPhys2/IntPhys2_2way*` | **classification** (2-way possible/impossible; attentive probing) | `{vjepa,llavavideo,qwen3vl}_analysis` |
| `toy_dataset/` | `data_csv/toy_physics/*` — 64-frame synthetic ball clips (stride-4 → 16f) | **regression** (`speed`, `direction`(sin,cos), `accel_mag`) | `vjepa_{velocity, acceleration, combined, combined_attentive}`, `llavavideo_{velocity, acceleration, combined}`, `qwen3vl_{velocity, acceleration, combined}` (10 configs; only `vjepa` has `combined_attentive`) |
| `blender_toy_dataset/` | `data_csv/blender_toy/*` — paper-faithful Kubric-style Blender, **native 16f** (`frame_step: 1`) | **regression** + `vjepa_attn_distance` (encoder-only attention-distance mode) | `vjepa_combined`, `vjepa_attn_distance`, `llavavideo_combined`, `qwen3vl_combined` |

**Backend ↔ config ↔ conda env** (each backend needs its own env):

| Backend | env | Encoder module | Checkpoint form |
|---|---|---|---|
| `vjepa` | `vjepa2` | `evals.analysis.modelcustom.vit_encoder_multilayer` | local `.pth` snapshot |
| `llavavideo` | `lmms_eval_llavavideo` | `evals.analysis_vlm.modelcustom.llava_video_encoder` | HF repo id `lmms-lab/LLaVA-Video-7B-Qwen2` + `cache_dir`; `wrapper_kwargs.llava_repo` required |
| `qwen3vl` | `lmms_eval_py311_2.7` | `evals.analysis_vlm.modelcustom.qwen3vl_encoder` | HF repo id `Qwen/Qwen3-VL-4B-Instruct` + `cache_dir` |

### Modes: dispatch & 5-phase roadmap

`experiment.analysis.modes` is a `{name: cfg}` map dispatched by `run_modes` (`evals/analysis_vlm/modes/__init__.py:76`) **on rank 0 only** (`eval.py:569`). The registry is intentionally **single-entry today** — this is Phase 1 of a five-phase reproduction plan (`evals/analysis_vlm/modes/REPRODUCTION_PLAN.md`):

| Phase | Mode | Paper ref | Status |
|---|---|---|---|
| 0 | Dispatch scaffold (`modes/__init__.py` registry + `run_modes` + `AnalysisContext`; the additive `eval.py:565–589` block; `skip_base_probe`) | — | **DONE** (default-off, verified) |
| 1 | `attention_distance` | Fig 3 / Appendix C.6 & Fig 19 | **DONE** (run & reproduced, see below) |
| 2 | `orthogonal_probe_sequence` | C.11 | pending |
| 3 | `steering` | C.12 | pending |
| 4 | `direction_tuning` | C.7 / C.10 | pending |
| 5 | `attention_ablation` | C.6 / Table 4 | pending |

Pending phases self-register in `_import_modes()` as their files are added (`modes/__init__.py:34–36`); `REPRODUCTION_PLAN.md` gives the dependency-ordered blueprint (subspace deflation → orthogonal/steering; direction_tuning; ablation last).

---

## Key code

The whole config surface is parsed in `evals/analysis_vlm/eval.py:main()`; line anchors below are verified against the source.

### Top-level keys

| Key | Default | Meaning |
|---|---|---|
| `eval_name` | — (req) | Scaffold route. `analysis_vlm` for the current harness. |
| `folder` | — (req) | Log/ckpt root. Actual path = `<folder>/analysis_vlm/<tag>/`. |
| `num_workers` | `12` | DataLoader workers (`eval.py:94`). |
| `resume_checkpoint` | `false` | Resume from `<…>/<tag>/latest.pt` (`eval.py:92`). |
| `val_only` | `false` | `true` = eval only, skip probe training (`eval.py:90`). |
| `tag` | `None` | Run name / subfolder. Same tag reuses the folder (`eval.py:93`). |

### `experiment.analysis`

Parsed at `evals/analysis_vlm/eval.py:104–214`, `:504`, `:568`.

| Key | Default | Allowed | Meaning |
|---|---|---|---|
| `model` | `""` | `vjepa` \| `llavavideo` \| `qwen3vl` | Selects backend module **and** data mode via `_BACKENDS` (`eval.py:62`). Unknown value raises. Omit only if you set `model_kwargs.module_name` explicitly. |
| `data_mode` | derived | `clip` \| `raw` | Rarely set. Derived: `clip` for vjepa, `raw` for VLMs (`eval.py:109–117`). `clip` = the shared V-JEPA `VideoDataset`; `raw` = per-frame native VLM preprocessing. |
| `stages` | — (req) | dict / `"all"` / `[int,…]` | Which encoder outputs to probe (see below). Legacy alias: `layers` (`eval.py:124`). |
| `probes` | — (req) | non-empty list | One head per `(stage × probe)`; asserted non-empty (`eval.py:148`). |
| `task` | `classification` | `classification` \| `regression` | CrossEntropy→accuracy vs MSE→R² (`eval.py:189`). |
| `regression` | `{}` | dict | Required iff `task: regression` (see below). |
| `plot` | `false` | bool | After training, write `stage_val_acc.png` (x=layer/stage, y=best val metric). |
| `plot_pez` | `None` | `[lo,hi]`, `0≤lo<hi≤1` | Shade a "Physics Emergence Zone" band (layer-fraction). Validated at `eval.py:151–153`. Consumed **twice**: (1) shades `stage_val_acc.png`; (2) passed as `ctx.plot_pez` into the `attention_distance` layerwise plot to shade `attention_distance_layerwise.png` (`eval.py:574` → `attention_distance.py:73,135–138`). |
| `skip_base_probe` | `false` | bool | `true` ⇒ `num_probe_epochs = 0`, i.e. **no probe training** (encoder-only runs). `eval.py:504`. |
| `modes` | `{}` | `{name: cfg}` | Default-off post-hoc modes (see below). Dispatched rank-0-only at `eval.py:568–588`. |

#### `stages`

Structured form (recommended): a dict where **only `vision_encoder` carries a per-layer selection** (`list[int]` or `"all"`); every other key is a boolean toggle for a single named stage.

```yaml
stages:
  vision_encoder: all          # or [3, 7, 11, 15, 19, 23]
  after_merger: true           # qwen3vl toggle
  deepstack: [5, 11, 17]       # qwen3vl per-index
```

- **clip / vjepa** (`eval.py:125–139`): backbone == `vision_encoder` only; the dict *must* contain `vision_encoder` (missing key raises with a typo hint), extra toggle keys are ignored with a warning. Resolved to `out_layers` via `_resolve_layers` (`eval.py:71`); `"all"` expands to `range(depth)` using `_VIT_DEPTH` (`vit_large=24`, `vit_huge=32`, `vit_giant=40`). Shorthand `stages: all` or `stages: [5,11,23]` also accepted.
- **raw / VLM** (`eval.py:140–145`): the whole spec is passed to the backend as `out_stages`; the backend resolves `"all"` + toggles and reports the final `.stages` back. Plot x-axis uses stage *position* (not block index) to avoid collisions.

| Backend | `vision_encoder` layers | Toggle stages | Source |
|---|---|---|---|
| `vjepa` (clip) | ViT blocks `0..depth-1` (vit_large ⇒ 0..23) | — (backbone only) | `_VIT_DEPTH`, `eval.py:68` |
| `llavavideo` (raw) | SigLIP `layer_0..layer_25` (1152-d) | `after_projector` (3584-d), `after_vision_encoder_pool2` (1152-d), `after_projector_pool2` (3584-d). `"all"` = `layer_0..25 + after_projector`. | `llava_video_encoder.py:12–18,44–47` |
| `qwen3vl` (raw) | ViT `block_0..block_23` (1024-d, pre-merger) | `before_merger` (=block_23, 1024-d), `after_merger` (2560-d), `deepstack: [5,11,17]`/`all`/`true` (2560-d). `"all"` = `block_0..23 + after_merger`. | `qwen3vl_encoder.py:11–16,54` |

#### `regression`

Required when `task: regression` (`eval.py:190–214`). The CSV integer label **indexes** an `(N,D)` targets array, so the standard dataloaders are unchanged — the harness maps `label → target vector`.

```yaml
regression:
  targets_npy: /.../combined_targets.npy   # (N,D) float; NaN = undefined for that clip
  variables:                               # each becomes its own R² curve on the SAME plot
    - {name: speed,     cols: [0]}         # valid only on constant-velocity clips (NaN elsewhere → masked)
    - {name: direction, cols: [1, 2]}      # sin,cos of angle (circular)
    - {name: accel_mag, cols: [3]}
```

- Targets are **per-column standardized NaN-aware** (`eval.py:201–203`, `mu=nanmean`, `sd=nanstd`); R² is invariant to this affine transform. NaNs stay NaN and are masked per head. (`mu`/`sd`/`tpath` are handed to modes via `ctx.col_mu`/`ctx.col_sd`/`ctx.targets_npy` so a mode can recover raw targets — `eval.py:577–579`.)
- `variables` optional: if omitted, one variable spanning all columns (`eval.py:205–206`). Column indices are range-checked (`eval.py:210`).

#### `modes` (default-off post-hoc analysis)

`run_modes` (`modes/__init__.py:76–94`) dispatches the `{name: cfg}` map. A run **without this key never imports or executes any mode** (`modes/__init__.py:80` early-return). Per-entry `cfg` may be `true`/`{}`/`None` (run with defaults), a dict of options, or `false`/`{enabled: false}` (skip, `:84`). Unknown names **raise** with the valid list (`:88`). Each mode writes under `<folder>/<name>/`.

The only registered mode is **`attention_distance`** (`modes/attention_distance.py:32`), used by `blender_toy_dataset/vjepa_attn_distance.yaml`. It asserts the V-JEPA **clip** encoder (`data_mode == "clip"`, `attention_distance.py:34`), finds the RoPEAttention blocks (`_find_rope_attn`, `:38`; raises if none), streams val batches through `attention_hooks(...)` (`:54`), and finalizes a per-(layer,head) statistic. `num_layers=24`, `num_heads=16` for vit_large.

```python
# attention_distance.py:43,48 — config knobs
max_batches = cfg.get("max_batches", 8)                 # code default 8; the shipped config uses 10
collector = AttentionDistanceCollector(
    num_layers=num_layers, num_heads=num_heads,
    query_chunk=int(cfg.get("query_chunk", 512)),       # memory knob — RESULT-INVARIANT
    max_batches=max_batches)                            # #val batches actually measured
```

**Config options** (`attention_distance.py`):

| Key | Code default | Shipped config | Meaning |
|---|---|---|---|
| `enabled` | — | `true` | `false` / `{enabled: false}` skips the mode. |
| `query_chunk` | `512` | `512` | Query-chunk size for the streamed softmax so the `(B,H,N,N)` matrix is **never materialized**. Pure memory knob — **does not change the result** (`:48`). |
| `max_batches` | `8` | `10` | Number of val batches averaged (cap on encoder forwards actually measured; `:43,58`). |
| `annotate` | `true` | (default) | Render per-cell numeric distance on the heatmap (`:72,100–105`). |

**Three outputs** written to `<folder>/attention_distance/`:

| File | What it is | Source |
|---|---|---|
| `attention_distance.png` | **PRIMARY** — paper **Fig 3 heatmap**. x=`Layer` (0–23), y=`Attention Head` (0–15), colour = **spatial** distance in patches, cmap `Blues_r` (**LOW distance = DARK blue**, so unusually-local heads stand out), per-cell value annotations (when `annotate`), colorbar `"Distance (patches)"`, title `"V-JEPA v2-L: Attention Distance Per Head"`. | `attention_distance.py:76–112` |
| `attention_distance_layerwise.png` | Companion **Appendix Fig 19** — dual-axis line plot: layer-mean distance **D̄** (mean over heads, red solid, left axis) + **head specialization S** (std over heads, blue dashed, right axis) vs **layer fraction**, with PEZ shading from `ctx.plot_pez`. | `attention_distance.py:115–156` |
| `attention_distance.json` | Raw stats (schema below). | `attention_distance.py:61–66`; keys from `attention_hooks.py:217–225` + `:62` |

**JSON schema** (`AttentionDistanceCollector.finalize`, `attention_hooks.py:217–225`; `n_batches` added at `attention_distance.py:62`) — the mode captures **both** spatial and temporal distance; the heatmap uses `spatial_distance.T`:

| Key | Type | Meaning |
|---|---|---|
| `spatial_distance` | `[24][16]` float | Per (layer, head) attention-weighted **Euclidean patch** distance. |
| `temporal_distance` | `[24][16]` float | Per (layer, head) attention-weighted `|Δt|` **tubelet** distance. |
| `num_layers` | int (`24`) | ViT depth. |
| `num_heads` | int (`16`) | Heads per block. |
| `rows_per_layer` | `[24]` float | `#(query, batch)` rows averaged per layer = `n_clips × N_tokens`. |
| `n_batches` | int | Number of val batches actually measured. |

> **head specialization S** = std over the 16 heads of their per-head attention distances **within a layer** (attention-head diversity). It is the blue dashed curve in `attention_distance_layerwise.png` (`attention_distance.py:132,146–148`) and **spikes at the PEZ** — local and global heads coexist there, so the spread widens.

**Validated result (Fig 3 reproduced).** `attention_distance` was run on the Blender velocity set (single-GPU, `n_batches = 10`, 78 velocity val clips × 1568 tokens = 122304 rows/layer) — outputs under `configs/analysis/blender_toy_dataset/logs/analysis_vlm/vjepa-blender-attn_distance/attention_distance/`. Observed signature: mean spatial distance **D̄** dips from ≈7.1 patches (L0) to ≈3.7–4.0 in **layers 5–10** (the Physics Emergence Zone, ~one-third depth) and rises back to ≈7 late; head specialization **S** spikes from ≈0.33 (L0) to ≈2.5 (L6/L10). The per-head minimum reaches ≈0.1 patches (an extremely local head at L6). On the heatmap the unusually-local (dark) heads **cluster in the middle layers** while early/late layers are uniformly long-range — the paper's signature.

### `experiment.data`

Parsed at `evals/analysis_vlm/eval.py:156–175`.

| Key | Default | Allowed | Meaning |
|---|---|---|---|
| `dataset_type` | `VideoDataset` | src/datasets loader | Usually left as-is. |
| `dataset_train` / `dataset_val` | — (req) | path | CSV, no header, space-separated: `<abs_video_path> <int_label>`. |
| `num_classes` | — (req) | int | Class count for classification; **ignored for regression** (kept for compat). |
| `resolution` | `224` | int | Input resolution. fpc64-256 ckpt ⇒ `256`. **Clip path only** — raw VLM path does native preprocessing. |
| `frames_per_clip` | `16` | even int | Frames per clip (even, `tubelet_size=2`). |
| `frame_step` | `4` | int | Contiguous sampling stride. Ignored if `uniform_sampling: true`. |
| `uniform_sampling` | `false` | bool | Sample `frames_per_clip` **evenly across the whole video**. Clip path only; raw path is always uniform (`eval.py:165–167`). |
| `clip_duration` | `None` | float | Seconds-based fixed-length sampling. |
| `num_segments` | `1` | int | Temporal clips per video (analysis ⇒ 1). |
| `num_views_per_segment` | `1` | int | Spatial multi-crop views (analysis ⇒ 1). |
| `normalization` | `None` | `((mean),(std))` | `None` ⇒ ImageNet defaults. |
| `resize_mode` | `crop` | `crop` \| `resize` | **Clip path only.** `crop` = shorter-side resize + center-crop (stock). `resize` = direct squash to `resolution²`. Invalid value **raises** (no silent fallback, `eval.py:174`). |

> **Raw (VLM) path ignores `resolution` / `resize_mode` / `frame_step`** — the backend runs native preprocessing (SigLIP 384, Qwen `smart` resize). VLM configs set only `dataset_*`, `num_classes`, `frames_per_clip`.

### `experiment.optimization`

Parsed at `evals/analysis_vlm/eval.py:217–241`.

| Key | Default | Allowed | Meaning |
|---|---|---|---|
| `batch_size` | — (req) | int | Per-GPU batch. |
| `num_epochs` | — (req) | int | Probe training epochs. `0` (or `skip_base_probe`) ⇒ no training. |
| `use_bfloat16` | — (req) | bool | AMP autocast (float16 in practice, per the template annotation). |
| `save_optimizer` | `false` | bool | `true` saves AdamW state in `latest.pt` (10s of GB for big attentive heads). |
| `cache_features` | `false` | bool | Encode once (frozen encoder, deterministic no-aug) then train probes over the cache — big speedup. |
| `cache_pooling` | `tokens` | `pooled` \| `tokens` \| `framewise` | What the cache stores. `pooled` = mean‖max only (tiny, **linear probes only** — attentive raises, `eval.py:302–308`). `tokens` = all tokens (large, any probe). `framewise` = per-frame spatial pool `(B,T,D)` (VLM temporal probes). |
| `cache_max_gb` | `64` | number | Abort if estimated per-rank cache RAM exceeds this (hard guard). |
| `default_head` | `{}` | dict | Base optimizer values for probes that omit them. |

`default_head` (and per-probe `optimization`) fields, defaults from `_opt_kwargs` (`eval.py:234–241`):

| Field | Default | Note |
|---|---|---|
| `lr` | `0.001` | ref_lr |
| `weight_decay` | `0.01` | ref_wd |
| `final_weight_decay` | `0.01` | cosine-end wd |
| `start_lr` | `0.0` | warmup start |
| `final_lr` | `0.0` | cosine-end lr |
| `warmup` | `1.0` | warmup epochs |

Per-probe `optimization` overrides `default_head` (dict merge, `eval.py:231–233`).

### Probe spec (`experiment.analysis.probes[*]`)

Parsed at `eval.py:295–343`; built by `build_probe` / named by `probe_name` (`evals/analysis/probes.py:56,84`).

| Key | Default | Allowed | Applies to |
|---|---|---|---|
| `type` | `attentive` | `linear` \| `attentive` | both |
| `pooling` | `mean` | `mean` \| `max` \| `meanmax` \| `framewise_mean` \| `framewise_max` | linear. `meanmax` doubles input dim. `framewise_*` keeps temporal order (VLM; needs `cache_pooling` ≠ `pooled`). |
| `pre_norm` | `true` | bool | linear (LayerNorm on pooled feature; corrects cross-layer scale). |
| `num_heads` | `16` | int | attentive pooler heads. |
| `num_probe_blocks` (alias `depth`) | `1` | int | attentive depth = `depth-1` self-attn + 1 cross-attn. |
| `temporal_pos` | `none` | `none` \| `learnable` \| `rope` | attentive **VLM only** (V-JEPA RoPE already carries time). |
| `name` | auto | str | Log column; auto = e.g. `L23_linear-mean`, `L11_attentive-d4`. |
| `optimization` | inherits `default_head` | dict | per-head optimizer. |

---

## Configuration

### Annotated example — `blender_toy_dataset/vjepa_combined.yaml` (V-JEPA regression, all 24 layers)

```yaml
eval_name: analysis_vlm
folder: /.../configs/analysis/blender_toy_dataset/logs   # → <folder>/analysis_vlm/<tag>/
num_workers: 8
resume_checkpoint: false
val_only: false
tag: vjepa-blender-polar_regression

experiment:
  analysis:
    model: vjepa
    task: regression
    regression:
      targets_npy: /.../blender_toy/blender_targets.npy   # (672,4)=[speed,sinθ,cosθ,accel_mag], NaN=undefined
      variables:
        - {name: speed,     cols: [0]}
        - {name: direction, cols: [1, 2]}
        - {name: accel_mag, cols: [3]}
    stages:
      vision_encoder: all       # ViT blocks 0..23
    plot: true
    plot_pez: [0.2, 0.4]        # shade PEZ band
    probes:
      - type: linear
        pooling: mean
        pre_norm: true
        optimization: { lr: 0.001, weight_decay: 0.1, warmup: 2.0 }

  data:
    dataset_type: VideoDataset
    resolution: 256
    resize_mode: resize
    frame_step: 1               # blender is native 16f
    uniform_sampling: true
    num_segments: 1
    num_views_per_segment: 1
    dataset_train: /.../blender_toy/combined_train.csv
    dataset_val:   /.../blender_toy/combined_val.csv
    num_classes: 4              # ignored for regression
    frames_per_clip: 16

  optimization:
    batch_size: 8
    num_epochs: 40
    use_bfloat16: true
    save_optimizer: false
    cache_features: true
    cache_pooling: pooled       # linear-only, tiny → optimal for all-layer scan
    cache_max_gb: 80
    default_head: { start_lr: 0.0, final_lr: 0.0, final_weight_decay: 0.0 }

model_kwargs:
  checkpoint: /.../models--facebook--vjepa2-vitl-fpc64-256/.../original/model.pth
  module_name: evals.analysis.modelcustom.vit_encoder_multilayer
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder   # EMA weights (eval standard) vs `encoder`=online
      model_name: vit_large            # used to expand stages:"all" to depth 24
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
      img_temporal_dim_size: null
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false               # do NOT set out_layers here — injected from stages
```

**VLM `model_kwargs` differ**: `checkpoint` = HF repo id, add `cache_dir`, `pretrain_kwargs.encoder: {}`, and backend `wrapper_kwargs`:
- llavavideo: `llava_repo` (LLaVA-NeXT path, required), `spatial_pool_stride: 2`, `dtype: float16`.
- qwen3vl: `resize_mode: smart`, `min_pixels: 8192`, `max_pixels: 112896`, `attn_implementation: sdpa`, `dtype: float16`.

### Distinctive knobs — `blender_toy_dataset/vjepa_attn_distance.yaml` (encoder-only mode)

This is the only config that trips the `modes` path. It clones `vjepa_combined.yaml` and adds `modes:` + turns off probe training and the cache. Its non-obvious deviations from the general rules:

| Key | Value here | Why (differs from the norm) |
|---|---|---|
| `data.resolution` | `224` | Paper geometry: 14×14 patches × 8 tubelets = **1568 tokens** — an intentional exception to the "`256` for the fpc64-256 ckpt" rule. Distance is in patch units either way, but 224 matches the paper. |
| `analysis.skip_base_probe` | `true` | Encoder-only: distance capture needs no trained probe (`eval.py:504`). |
| `optimization.num_epochs` | `0` | No probe training. |
| `optimization.cache_features` | `false` | Attention capture runs **fresh forwards**, not the cache. |
| `analysis.plot_pez` | `[0.2, 0.4]` | Shades the PEZ band on `attention_distance_layerwise.png` via `ctx.plot_pez`. |

```yaml
experiment:
  analysis:
    model: vjepa
    task: regression
    regression:
      targets_npy: /.../blender_toy/blender_targets.npy
      variables:
        - {name: direction, cols: [1, 2]}
    stages: { vision_encoder: all }
    plot_pez: [0.2, 0.4]
    skip_base_probe: true          # encoder-only: distance capture needs no trained probe
    modes:
      attention_distance:          # ← default-off unless present; absent ⇒ existing behavior
        enabled: true
        query_chunk: 512           # stream queries so (B,H,N,N) is never materialized (result-invariant)
        max_batches: 10            # measure the first 10 val batches (code default is 8)
    probes:
      - { type: linear, pooling: mean, pre_norm: true,
          optimization: { lr: 0.001, weight_decay: 0.1, warmup: 2.0 } }
  data:
    dataset_type: VideoDataset
    resolution: 224                # paper geometry: 14x14 x 8 tubelets = 1568 tokens
    resize_mode: resize
    frame_step: 1
    uniform_sampling: true
    dataset_train: /.../blender_toy/velocity_val.csv
    dataset_val:   /.../blender_toy/velocity_val.csv
    num_classes: 4
    frames_per_clip: 16
  optimization:
    batch_size: 8
    num_epochs: 0                  # no probe training (encoder-only mode)
    use_bfloat16: true
    cache_features: false          # attention capture runs fresh forwards, not the cache
```

### SLURM launchers (`z_scripts/`)

**Three** thin SLURM wrappers. All `set -euo pipefail`, activate the conda env, export `PYTHONPATH=$PROJECT`, disable NCCL P2P/IB (`NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` — some nodes lack GPU peer-access), set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, convert the SLURM-assigned `$CUDA_VISIBLE_DEVICES` into `--devices cuda:i …`, and run:

```bash
python -m evals.main --fname "$CONFIG" --devices $DEVICES
```

All read `CONDA_ENV` and `CONFIG` from the environment with defaults, so you override them via `sbatch --export`.

| Launcher | `CONDA_ENV` default | `CONFIG` default | GPUs | Node / mem |
|---|---|---|---|---|
| `run_analysis_vjepa.sh` | `vjepa2` | `blender_toy_dataset/vjepa_combined.yaml` | `--gres=gpu:4` (comment recommends 8 for `cache_pooling: tokens`) | `-w vll4`, `mem-per-gpu 80G` |
| `run_analysis_vlm.sh` | `lmms_eval_llavavideo` | `blender_toy_dataset/llavavideo_combined.yaml` | `--gres=gpu:4` | `--nodelist vll4`, `mem-per-gpu 70G` |
| `run_attn_distance_vjepa.sh` | `vjepa2` | `blender_toy_dataset/vjepa_attn_distance.yaml` | **`--gres=gpu:1` (single by design)** | `-w vll6`, `mem-per-gpu 80G` |

> **Why `run_attn_distance_vjepa.sh` is single-GPU on purpose.** The modes dispatch runs on **rank 0 only** (`eval.py:569` `if modes_cfg and rank == 0`), so multi-GPU gives **no speedup for post-hoc modes** — extra ranks would sit idle. Multi-GPU only helps the base probing sweep / feature cache, which this encoder-only config doesn't run (`skip_base_probe: true`, `num_epochs: 0`, `cache_features: false`). Single-GPU is the faithful, simplest setup.

#### `--export CONFIG=` override pattern

```bash
# default (env + config baked in)
sbatch z_scripts/run_analysis_vjepa.sh

# swap config only (vjepa env fixed)
sbatch --export=ALL,CONFIG=configs/analysis/InsPhys2/vjepa_analysis.yaml \
       z_scripts/run_analysis_vjepa.sh

# VLM: env and config MUST be paired (each backend has its own env)
sbatch --export=ALL,CONDA_ENV=lmms_eval_llavavideo,CONFIG=configs/analysis/InsPhys2/llavavideo_analysis.yaml \
       z_scripts/run_analysis_vlm.sh
sbatch --export=ALL,CONDA_ENV=lmms_eval_py311_2.7,CONFIG=configs/analysis/blender_toy_dataset/qwen3vl_combined.yaml \
       z_scripts/run_analysis_vlm.sh

# attention-distance mode (single GPU by design)
sbatch z_scripts/run_attn_distance_vjepa.sh
```

> `ALL,` keeps the full submitting environment; append only the vars you override. Forgetting `ALL` drops `PATH` etc.

---

## Invariants & gotchas

- **Modes are default-off.** No `experiment.analysis.modes` key ⇒ modes package is never imported or run (`modes/__init__.py:80`). A mode set to `false` / `{enabled: false}` is skipped; unknown mode names **raise** (`:88`). Modes dispatch runs on **rank 0 only** (`eval.py:569`).
- **`attention_distance` requires the clip encoder.** It asserts `data_mode == "clip"` and raises if no RoPEAttention blocks are found (`attention_distance.py:34,38–40`). It writes **three** files (heatmap `.png` primary, `_layerwise.png`, `.json`) — not one. `query_chunk` is a **result-invariant** memory knob; `max_batches` controls how many val batches are averaged (code default **8**, the shipped config uses **10**).
- **`cache_pooling: pooled` is linear-only.** An attentive probe with `pooled` cache **raises** (`eval.py:302–308`). `framewise_*` pooling needs `tokens` or `framewise`, never `pooled`.
- **`cache_max_gb` is a hard guard**, not advice: estimated per-rank cache RAM over the limit **aborts** the run. `tokens` × many stages × 32f/256 ≈ 88 GB/rank on 4 GPUs (aborts at default 64) vs ≈ 44 GB on 8 GPUs. Use `pooled` or more GPUs, or raise the cap.
- **`resize_mode` invalid ⇒ raise** (`eval.py:174`); no silent fallback to `crop`.
- **`stages` typos raise** on the clip path if the dict lacks `vision_encoder` (`eval.py:129–133`). Do **not** set `wrapper_kwargs.out_layers` / `out_stages` — they are injected from `stages`.
- **`num_classes` is ignored for regression** but kept for dataloader compatibility; output dim comes from `variables[*].cols`.
- **VLM raw path ignores `resolution` / `resize_mode` / `frame_step`.** Those only affect the clip (V-JEPA) path.
- **`checkpoint_key: target_encoder`** loads the EMA weights (evaluation standard); `encoder` would load the online branch.
- **`use_bfloat16` is float16 autocast** in practice despite the name (per the template annotation).
- **Backend ↔ conda env must match** the config's `model` (V-JEPA / LLaVA / Qwen each need a different env) — hence the paired `--export` on `run_analysis_vlm.sh`.
- **`analysis_TEMPLATE.yaml` documents the *older* `analysis` harness** (`layers` key, no `model`/`task`/`regression`/`modes`). The `analysis_vlm` parser accepts `layers` as a legacy alias for `stages` (`eval.py:124`), but the rest of the template's key space does not carry over — use `stages:` and `eval_name: analysis_vlm`.

---

## Cross-references

- [`02-analysis-vlm-harness.md`](02-analysis-vlm-harness.md) — the `analysis_vlm/eval.py` harness that parses this key space.
- [`03-feature-caching-and-pooling.md`](03-feature-caching-and-pooling.md) — `cache_features` / `cache_pooling` / `cache_max_gb`.
- [`04-probes-regression-nanmask.md`](04-probes-regression-nanmask.md) — probe spec + regression / NaN masking.
- [`05-analysis-clip-harness.md`](05-analysis-clip-harness.md) — the older `analysis` harness (`analysis_TEMPLATE.yaml`).
- [`06-data-pipeline-changes.md`](06-data-pipeline-changes.md) — `experiment.data`, `resize_mode`, sampling.
- [`07-plotting.md`](07-plotting.md) — `plot` / `plot_pez` / `stage_val_acc.png`.
- [`08-vlm-encoders.md`](08-vlm-encoders.md) — backend stage vocabularies.
- [`09-blender-toy-dataset.md`](09-blender-toy-dataset.md) — the Blender toy dataset.
- [`10-datasets-csv-targets.md`](10-datasets-csv-targets.md) — CSV format + `targets_npy`.
- [`11-attention-hooks.md`](11-attention-hooks.md) — `attention_hooks.py` internals (SDPA patch, collector, ablation bias).
- [`12-analysis-modes.md`](12-analysis-modes.md) — the modes registry + `attention_distance`.
- [`14-reproduction-status-and-findings.md`](14-reproduction-status-and-findings.md) — phase roadmap + validated findings.
