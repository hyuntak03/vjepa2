# Overview & Architecture

## Purpose

This fork turns the stock [V-JEPA2](https://github.com/facebookresearch/vjepa2) repo into a
**mechanistic-interpretability harness** for studying *where and how physical variables (speed,
direction, acceleration, "possible vs. impossible") are encoded across the layers of a frozen video
world model*. It reproduces the analyses of *"Interpreting Physics in Video World Models"* (the PDF
is checked in at repo root) and extends them from V-JEPA2 to two video-VLM vision encoders.

Concretely, the fork adds machinery to:

- Take a **frozen** encoder (never fine-tuned), tap its **per-layer** features, and attach a small
  **probe** (linear or attentive) to *each* layer independently.
- Train all the per-layer probes jointly and report a **layer × probe** metric matrix +
  a **layer-wise metric plot** (accuracy for classification, R² for continuous-variable regression).
- Do this uniformly across three model families — **V-JEPA2 ViT**, **LLaVA-Video SigLIP**,
  **Qwen3-VL ViT** — plus post-hoc analysis "modes" (attention-distance capture / ablation).

Everything is **additive and default-off**: with no analysis config present, the upstream repo
behaves byte-for-byte as before (see [Design principle](#design-principle-additive--default-off)).

## What changed vs upstream V-JEPA2

Base commit for all diffs below: `204698b` ("Fix figure (#143)").

The fork adds **two new eval packages** and touches only a handful of upstream files, each with a
tiny passthrough:

| Kind | Path | Delta |
|---|---|---|
| **new package** | `evals/analysis/` | Clip / V-JEPA layer-wise probing harness (`eval_name: analysis`). |
| **new package** | `evals/analysis_vlm/` | Unified vjepa / llavavideo / qwen3vl probing harness (`eval_name: analysis_vlm`); superset of the above. |
| modified | `src/datasets/video_dataset.py` | +13 lines: `uniform_sampling` option — sample `fpc` frames *evenly across the whole video* instead of a contiguous `frame_step` window (`src/datasets/video_dataset.py:334`). Default `False`. |
| modified | `src/datasets/data_manager.py` | +2 lines: thread `uniform_sampling` through `init_data` → `make_videodataset`. |
| modified | `evals/video_classification_frozen/eval.py` | +2 lines: thread `uniform_sampling` through `make_dataloader`. |
| modified | `src/models/vision_transformer.py` | **+1 comment only.** The multi-layer feature tap (`out_layers`) is **already upstream** (`src/models/vision_transformer.py:204-207`); the fork just reuses it. |
| modified | `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py` | comment-only annotations. |
| build artifact | `build/lib/**` | A mechanical `pip`-build copy of `src/` that got committed. **Not a deliverable** — ignore it. |

> **Key accuracy note:** the ability to return features from *multiple* ViT blocks
> (`out_layers`) is native to upstream V-JEPA2's `VisionTransformer`. This fork adds *no* core-model
> code to get per-layer features; it only wraps the existing `out_layers` API (see
> `evals/analysis/modelcustom/vit_encoder_multilayer.py:57`).

### Routing (why no scaffold/main edits)

`evals/scaffold.py` dynamically imports the eval module named by the config's `eval_name`:

```python
# evals/scaffold.py:18
import_path = f"evals.{eval_name}.eval"
return importlib.import_module(import_path).main(args_eval=args_eval, ...)
```

So `eval_name: analysis` → `evals/analysis/eval.py` and `eval_name: analysis_vlm` →
`evals/analysis_vlm/eval.py`, with **zero changes** to `main.py` / `main_distributed.py` /
`scaffold.py`.

## The two analysis harnesses

Both share the same core loop (frozen encoder → per-layer/stage features → one probe head per
`(stage × probe-spec)` → joint train → `stage_val_acc.png` + `summary.json`). `analysis_vlm` is a
strict superset.

| | `evals/analysis` | `evals/analysis_vlm` |
|---|---|---|
| `eval_name` | `analysis` | `analysis_vlm` |
| Backbones | V-JEPA2 ViT only (`data_mode="clip"`) | `vjepa` (reuses the clip backend) · `llavavideo` · `qwen3vl` (`_BACKENDS`, `evals/analysis_vlm/eval.py:62`) |
| Data path | stock clip `VideoDataset` (normalized 5-D clips) | clip *or* **raw-frame** loader (`evals/analysis_vlm/data.py`) chosen by `data_mode` |
| Task | classification (accuracy) | classification **or** regression → R² (`evals/analysis_vlm/eval.py:189`) |
| Probes | `linear`, `attentive` (`evals/analysis/probes.py`) | + temporal-aware `TemporalLinearProbe` / `TemporalAttentiveClassifier` (`evals/analysis_vlm/probes.py`) |
| Feature cache | no (re-encodes each epoch) | **yes** — one deterministic pre-pass, then epochs run over the cache (`evals/analysis_vlm/cache.py`) |
| Optimizer | one AdamW per head | one **fused** AdamW, one param-group per head (`_init_opt_fused`, `evals/analysis_vlm/eval.py:617`) |
| Post-hoc modes | — (`attention_hooks.py` lives here but is driven by the vlm harness) | `experiment.analysis.modes` registry (`evals/analysis_vlm/modes/`) |

The `analysis_vlm` harness is where active development is; `analysis` is the simpler, original
clip-only version. The two share `probes.py` and `plotting.py` (the vlm harness imports
`evals.analysis.probes` / `evals.analysis.plotting`).

## Data flow

```
video (.mp4)                             CSV: "<abs_path> <int_label>"  (no header, space-delim)
   │
   ├─ clip path  (V-JEPA):  VideoDataset → normalized clip tensors  (5-D, shorter-side crop or direct resize)
   └─ raw path   (VLM):     RawVideoDataset → list of (T,H,W,C) uint8 frames  (backend preprocesses natively)
   │
   ▼
FROZEN encoder  (torch.no_grad, never trained)
   ├─ vjepa:      vit_encoder_multilayer.MultiLayerClipAggregation  → list over out_layers, each (B,N,D)
   ├─ llavavideo: SigLIP tower (no 7B LLM)  → list over stages,  (B, T*729, 1152) / projector 3584
   └─ qwen3vl:    Qwen3-VL ViT (no 4B LLM)  → list over stages,  block/​merger/​deepstack features
   │
   ▼
(optional) FEATURE CACHE  — one deterministic pre-pass, per-stage features held in RAM (fp16)
   │                         cache_pooling: tokens | pooled | framewise   (evals/analysis_vlm/cache.py)
   ▼
one PROBE head per (stage × probe-spec × regressed-variable)   — trained JOINTLY, encoder detached
   │   classification → CrossEntropy → accuracy
   │   regression     → per-column MSE (NaN-masked) → R² (all-reduced)
   ▼
per-layer metric  →  summary.json  +  stage_val_acc.png  (x = layer fraction, one line per probe/variable)
   │
   └─ (optional) modes: attention_distance capture / ablation, …   (evals/analysis_vlm/modes/)
```

Since the integer CSV label is reused as a **row index into an `(N, D)` targets `.npy`**, the *same*
dataloaders serve both classification and regression — the harness maps `label → target vector`
(`evals/analysis_vlm/eval.py:192-214`). No dataset code changes between tasks.

## Design principle: additive / default-off

Every capability is gated so an unconfigured run is unchanged:

- **New eval packages, not edits.** Selected only via `eval_name`; the stock evals are untouched.
- **`modes` block** (`evals/analysis_vlm/eval.py:568`): absent ⇒ `modes_cfg == {}` ⇒ the whole block
  is skipped and nothing is imported. Comment in-code: *"existing runs behave byte-for-byte
  identically."*
- **Attention hooks** (`evals/analysis/attention_hooks.py`): a monkey-patch of
  `F.scaled_dot_product_attention` installed **only** inside a context manager, fully torn down on
  exit. Disabled (`attention.enable: false`) ⇒ `nullcontext`, nothing patched. Capture-only is a
  *detached side computation* → encoder output is bit-identical (verified via `torch.equal`).
- **Regression** off by default (`task: classification`); **feature cache** off by default
  (`cache_features: false`); **`uniform_sampling`** default `False`; **`save_optimizer`** default
  `False`; **`skip_base_probe`** default `False` (loop runs the normal number of epochs).

## Change surface (new vs. modified, by area)

Legend: **N** = new file · **M** = modified upstream · **U** = untracked (working-tree only, not yet
committed) · git-ignored dirs are real deliverables produced/consumed by the harness.

### Analysis harness — `evals/`

| Status | Path | Role |
|---|---|---|
| N | `evals/analysis/eval.py` | Clip/V-JEPA layer-wise probing driver (`eval_name: analysis`). |
| N | `evals/analysis/probes.py` | `LinearProbe` + `build_probe` / `probe_name` (shared by both harnesses). |
| N | `evals/analysis/plotting.py` | Layer-fraction metric plot (accuracy or R²), PEZ shading, peak/elbow markers. |
| N | `evals/analysis/modelcustom/vit_encoder_multilayer.py` | `MultiLayerClipAggregation` — returns one `(B,N,D)` per requested block. |
| U | `evals/analysis/attention_hooks.py` | SDPA monkey-patch: per-head attention-distance capture + local-attention ablation. |
| N | `evals/analysis_vlm/eval.py` | Unified harness (vjepa/llavavideo/qwen3vl; classification+regression; cache; modes). |
| N | `evals/analysis_vlm/cache.py` | Frozen-feature RAM cache (`tokens`/`pooled`/`framewise`) + thread prefetcher. |
| N | `evals/analysis_vlm/data.py` | Raw-frame `RawVideoDataset` + unpadded DDP shard sampler for VLMs. |
| N | `evals/analysis_vlm/loadutil.py` | HF weight-location resolver (repo-id + cache_dir, offline-first). |
| N | `evals/analysis_vlm/probes.py` | `TemporalLinearProbe`, `TemporalAttentiveClassifier` (learnable / RoPE temporal pos). |
| N | `evals/analysis_vlm/modelcustom/llava_video_encoder.py` | LLaVA-Video SigLIP tower + projector, no 7B LLM. |
| N | `evals/analysis_vlm/modelcustom/qwen3vl_encoder.py` | Qwen3-VL vision ViT (blocks / merger / deepstack), no 4B LLM. |
| U | `evals/analysis_vlm/modes/__init__.py` | Mode registry + `AnalysisContext` + `run_modes` dispatch. |
| U | `evals/analysis_vlm/modes/attention_distance.py` | Mode: layer×head spatial/temporal attention distance (paper App. C.6). |
| U | `evals/analysis_vlm/modes/REPRODUCTION_PLAN.md` | Roadmap for remaining paper-repro modes. |

### `src/` data pipeline & models (upstream, modified)

| Status | Path | Delta |
|---|---|---|
| M | `src/datasets/video_dataset.py` | `uniform_sampling` frame selection (`+13`, default off). |
| M | `src/datasets/data_manager.py` | `uniform_sampling` passthrough (`+2`). |
| M | `src/models/vision_transformer.py` | comment only (`+1`); `out_layers` was already upstream. |
| M | `evals/video_classification_frozen/eval.py` | `uniform_sampling` passthrough in `make_dataloader` (`+2`). |
| M | `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py` | comment annotations only. |

### Data generation — `data_gen/` (git-ignored, deliverable)

| Path | Role |
|---|---|
| `data_gen/make_physics_toy.py` | Lightweight 2D PIL/ffmpeg toy-ball generator (velocity / acceleration grids; anti-shortcut nuisance randomization). |
| `data_gen/make_physics_blender.py` | Paper-faithful **Blender (bpy)** photorealistic sphere generator (Cycles render). |
| `data_gen/run_blender_toy.sh`, `sanity_check_blender.py` | Blender launch + sanity checks. |
| `data_gen/{blender_toy_dataset,physics_data}/` | Generated `.mp4`s + `metadata.csv` / `kinematics.json` (+ `.tar` bundles). |

### Target/CSV builders — `data_csv/` (CSV/JSON git-ignored, deliverable)

| Path | Role |
|---|---|
| `data_csv/make_regression_targets.py` | Build `targets.npy` + rewrite split CSVs so `label = npy row`; emit the `regression.variables` block (R2R). |
| `data_csv/make_blender_targets.py` | Same for the Blender toy set; combined velocity+accel space, `NaN` for undefined variables. |
| `data_csv/{toy_physics,blender_toy,R2R_4way_1500,IntPhys2}/` | Split CSVs + `targets.npy` per dataset. |

### Configs — `configs/analysis/` (YAMLs **tracked**; `logs/` subdirs git-ignored)

| Path | Role |
|---|---|
| `configs/analysis/analysis_TEMPLATE.yaml` | Documented template for all knobs. |
| `configs/analysis/{vjepa,llavavideo,qwen3vl}_analysis.yaml` | Per-model classification probing. |
| `configs/analysis/vjepa_regression.yaml` | R² regression (speed / direction / accel). |
| `configs/analysis/{toy_dataset,blender_toy_dataset,InsPhys2}/*.yaml` | Per-dataset variants (incl. `vjepa_attn_distance.yaml`, **U**, using `modes`). |

*(21 analysis YAMLs are tracked in the diff; `configs/analysis/logs/` and
`configs/analysis/toy_dataset/logs/` are git-ignored run outputs — checkpoints + plots.)*

### Scripts — `z_scripts/` (git-ignored, deliverable)

| Path | Role |
|---|---|
| `z_scripts/run_analysis_vjepa.sh`, `run_analysis_vlm.sh` | Slurm launchers (set `CONDA_ENV` + `CONFIG`). |

### Misc (repo root)

| Status | Path | Role |
|---|---|---|
| N | `debug_infer.py` | Minimal single-GPU encoder-forward debug script (no probes/labels). |
| N | `Interpreting Physics in Video World Models.pdf` | The reference paper being reproduced. |

## Config

Both harnesses read one YAML (via `python -m evals.main --fname <cfg>`). A representative
V-JEPA2 classification run (`configs/analysis/vjepa_analysis.yaml`, comments condensed to English):

```yaml
eval_name: analysis_vlm            # -> evals/analysis_vlm/eval.py
folder: /.../configs/analysis/logs # outputs at <folder>/analysis_vlm/<tag>/
num_workers: 8
resume_checkpoint: false
val_only: false
tag: vjepa2_shape_color

experiment:
  analysis:
    model: vjepa                   # vjepa | llavavideo | qwen3vl  (picks backend + data_mode)
    stages:
      vision_encoder: all          # ViT block indices 0..23, or e.g. [3,5,7,11,15,19,23]
    plot: true                     # write stage_val_acc.png (x = layer fraction, y = best val acc)
    probes:                        # one head per (stage x probe-spec), trained jointly
      - type: linear               # linear | attentive
        pooling: mean              # mean | max | meanmax
        pre_norm: true             # LayerNorm the pooled vector (corrects per-layer feature scale)
        optimization: { lr: 0.01, weight_decay: 0.0, warmup: 1.0 }

  data:
    dataset_type: VideoDataset
    dataset_train: /.../R2R_4way_1500_shape_color_train.csv   # "<abs_mp4> <int_label>"
    dataset_val:   /.../R2R_4way_1500_shape_color_val.csv
    num_classes: 4
    resolution: 256                # fpc64-256 checkpoint
    resize_mode: crop              # crop (shorter-side + center crop) | resize (direct, aspect squash)
    frames_per_clip: 32            # even (tubelet_size=2)
    frame_step: 1
    num_segments: 1
    num_views_per_segment: 1

  optimization:
    batch_size: 8
    num_epochs: 20
    use_bfloat16: true
    save_optimizer: false          # store probe weights only (smaller latest.pt)
    cache_features: true           # frozen encoder -> encode ONCE, then train probes over the cache
    cache_pooling: pooled          # pooled (mean||max; LINEAR only, tiny) | tokens (all probes, large)
    cache_max_gb: 130              # abort if estimated per-rank cache RAM exceeds this
    default_head: { start_lr: 0.0, final_lr: 0.0, final_weight_decay: 0.01 }

model_kwargs:
  checkpoint: /.../vjepa2-vitl-fpc64-256/.../model.pth
  module_name: evals.analysis.modelcustom.vit_encoder_multilayer
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder   # target_encoder (EMA, eval standard) | encoder (online)
      model_name: vit_large            # used to expand stages:"all" -> depth 24
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
    # out_layers is injected automatically from stages.vision_encoder
```

For **regression** (`configs/analysis/vjepa_regression.yaml`): set `task: regression` and add a
`regression:` block pointing at `targets_npy` with named `variables` (each a column-slice → its own
R² curve). For **VLMs**, set `model: llavavideo|qwen3vl`, point `checkpoint` at an HF repo id + set
`wrapper_kwargs.cache_dir`, and use structured `stages` (e.g. `{vision_encoder: all,
after_projector: true}` for LLaVA; `{vision_encoder: all, after_merger: true, deepstack: [5,11,17]}`
for Qwen).

## Gotchas / invariants / default-off guarantees

- **`stages` is polymorphic.** Accepts `"all"`, a `[int,...]` block list, a structured dict
  (`{vision_encoder: [...]|all, <toggles>: true}`), or a legacy list of concrete stage-name strings.
  For the V-JEPA clip path only `vision_encoder` is meaningful; extra keys are warned + ignored
  (`evals/analysis_vlm/eval.py:125-138`).
- **Plot x-axis differs by path.** Clip path uses the *block index* as x; raw/VLM path uses the
  *stage position* as x (so `block_5` vs `deepstack_5` don't collide on x=5), with the stage name on
  the tick. `summary.json` always keeps exact per-stage metrics
  (`evals/analysis_vlm/eval.py:287-294`).
- **`cache_pooling='pooled'` ⇒ linear probes only.** It caches only `[mean‖max]`, so attentive /
  framewise probes are rejected up front (`evals/analysis_vlm/eval.py:302-313`). `tokens` supports
  all probes but scales with `N × #stages` and can OOM host RAM — guarded by `cache_max_gb`
  (`evals/analysis_vlm/cache.py:139`).
- **`tokens` cache needs uniform token counts.** Videos of differing resolution/length produce
  different `N` and fail to concat. For Qwen use `resize_mode: fixed` + `qwen_fixed_h/w`; for LLaVA
  SigLIP is fixed 384² so it's fine (`evals/analysis_vlm/cache.py:151-159`).
- **Cache = "no-augment probing."** The cache pre-pass is deterministic (`training=False`) for *both*
  splits, so train-time augmentation is dropped. Intended for frozen-encoder probing; consistent
  with the VLM paths and with val/inference.
- **Regression standardization + NaN masking.** Targets are per-column standardized (NaN-aware); R²
  is invariant to this affine map. A variable undefined on some videos is `NaN` there and *masked
  per-head*, so one combined dataset can host variables defined on disjoint subsets
  (`evals/analysis_vlm/eval.py:195-210`, `run_one_epoch` masking at `:730-766`).
- **Temporal probes exist because pooling is permutation-invariant.** A depth-1 attentive pooler
  ignores frame order; LLaVA-Video's per-frame SigLIP bakes no temporal order into token *values*,
  so "up" vs "down" is invisible without a temporal pos-encoding. V-JEPA / Qwen3-VL encode time via
  RoPE / temporal patches and usually don't need it (`evals/analysis_vlm/probes.py:1-20`). These
  probes require `encoder.num_temporal`, which only the VLM backends expose.
- **VLM backends run in separate conda envs.** `vjepa2` / `lmms_eval_llavavideo` /
  `lmms_eval_py311_2.7`. Only the selected backend is imported (lazily via `module_name`), so
  conflicting heavy deps never co-load (`evals/analysis_vlm/eval.py:16-18`).
- **`out_layers` is upstream, not fork code.** Per-layer feature extraction reuses V-JEPA2's native
  `VisionTransformer(out_layers=...)`; the fork's ViT edit is a lone comment.
- **Attention hooks: identical-when-off.** With capture-only (no ablation) the encoder output is
  bit-identical to baseline; ablation *deliberately* changes it (that is the experiment). Any
  attention call outside a live RoPE-block context is a straight pass-through
  (`evals/analysis/attention_hooks.py:42-49, 295-322`).
- **Small-split drop_last trap.** The stock `init_data`/`make_videodataset` ignore `drop_last` and
  default to `True`, silently dropping partial batches — fatal for tiny val splits. The vlm harness
  force-sets `drop_last=False` after loader creation (`evals/analysis_vlm/eval.py:414-419`).
- **Working-tree state (uncommitted at time of writing):** the `modes` dispatch block and
  `attention_hooks.py` / `modes/` are **untracked** additions; the committed `analysis_vlm/eval.py`
  is 783 lines with a `+32` working-tree delta adding the `modes` hook.
