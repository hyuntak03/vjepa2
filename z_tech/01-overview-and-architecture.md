# 01 — Overview & Architecture

> This fork turns stock V-JEPA2 into a **default-off, config-driven mechanistic-interpretability harness** that taps a *frozen* video encoder layer-by-layer, probes where physical variables live, and reproduces *"Interpreting Physics in Video World Models"* — adding two eval packages (`analysis`, `analysis_vlm`) and post-hoc "modes" without editing a single line of the upstream training/eval core.

This section is the **map** of the fork. Its tables are authoritative: every tracked change is enumerated against an exact `git` filter, every modified upstream file is given its precise additive delta, and the default-off guarantee is stated for each.

---

## Purpose

The fork studies *where and how* physical variables — speed, direction, acceleration, and "possible vs. impossible" — are encoded across the layers of a **frozen** video world model. It reproduces the analyses of *"Interpreting Physics in Video World Models"* (PDF checked in at repo root) and generalizes them from V-JEPA2 to two additional video-VLM vision encoders.

Concretely, the fork adds machinery to:

- Take a **frozen** encoder (never fine-tuned), tap its **per-layer** features, and attach a small **probe** (linear or attentive) to *each* layer independently.
- Train all per-layer probes jointly and report a **layer × probe** metric matrix plus a **layer-fraction metric plot** (accuracy for classification, R² for continuous-variable regression).
- Do this uniformly across three model families — **V-JEPA2 ViT**, **LLaVA-Video SigLIP**, **Qwen3-VL ViT**.
- Run **post-hoc "modes"** on the same frozen encoder — currently `attention_distance` (per-head attention locality, paper Fig. 3 / App. C.6), with four more paper-repro modes scaffolded but pending.

Everything is **additive and default-off**: with no analysis config present, the upstream repo behaves byte-for-byte as before (see [The additive / default-off principle](#the-additive--default-off-principle)).

---

## What changed vs upstream V-JEPA2

Base commit for all diffs: **`204698b`** ("Fix figure (#143)"). Current HEAD: **`c296428`**. The working tree is **clean** — everything described here is committed.

The fork adds **two new eval packages** plus data/config/tooling, and touches only **five upstream files**, each with a tiny passthrough or a comment.

### New vs modified — headline

| Kind | Path | What / delta |
|---|---|---|
| **new package** | `evals/analysis/` | Clip / V-JEPA layer-wise probing harness (`eval_name: analysis`) + shared `probes.py` / `plotting.py` + `attention_hooks.py`. |
| **new package** | `evals/analysis_vlm/` | Unified vjepa / llavavideo / qwen3vl probing harness (`eval_name: analysis_vlm`); **strict superset** of `analysis`, and home of the `modes/` registry. |
| modified | `src/datasets/video_dataset.py` | **+13 lines**: `uniform_sampling` frame selection (default `False`). |
| modified | `src/datasets/data_manager.py` | **+2 lines**: thread `uniform_sampling` through `init_data` → `make_videodataset`. |
| modified | `evals/video_classification_frozen/eval.py` | **+2 lines**: thread `uniform_sampling` through `make_dataloader`. |
| modified | `src/models/vision_transformer.py` | **+1 comment only** (line 188). The multi-layer tap (`out_layers`) is **already upstream**. |
| modified | `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py` | comment/annotation only (`+3 −2`, all `#!` notes). |

### Exact additive deltas of the modified files

Each modified upstream file is a pure passthrough or comment — no behavior changes unless the new flag is explicitly set.

- **`src/datasets/video_dataset.py`** (`+13`) — a new `uniform_sampling` kwarg on `make_videodataset` + `VideoDataset.__init__`, plus one early-return branch in the frame sampler:

  ```python
  # src/datasets/video_dataset.py:331-345  (guarded, default off)
  if getattr(self, "uniform_sampling", False):
      n = len(vr)
      indices = np.clip(np.linspace(0, n - 1, num=fpc).round(), 0, n - 1).astype(np.int64)
      buffer = vr.get_batch(list(indices)).asnumpy()
      return buffer, [indices]
  ```
  Picks `fpc` frames **evenly across the whole video** (ignores `frame_step` / `num_clips`), avoiding the contiguous-window default that covers only a sub-segment. **Default `uniform_sampling=False` ⇒ this branch is never entered ⇒ sampling is byte-identical to upstream.**

- **`src/datasets/data_manager.py`** (`+2`) — adds `uniform_sampling=False` to `init_data`'s signature and forwards it to `make_videodataset`. No-op when unset.

- **`evals/video_classification_frozen/eval.py`** (`+2`) — adds `uniform_sampling=False` to `make_dataloader` and forwards it to `init_data`. No-op when unset.

- **`src/models/vision_transformer.py`** (`+1`) — a lone comment at line 188 (`#! patch embedding만 하고 pos embedding은 self.blocks에서 처리 (RoPE 쓰니까)`). **No code change.** The per-layer feature tap the harness relies on is native upstream:

  ```python
  # src/models/vision_transformer.py:205-209  (UPSTREAM, not fork code)
  if self.out_layers is not None and i in self.out_layers:
      outs.append(self.norm(x))
  ...
  if self.out_layers is not None:
      return outs
  ```

- **`evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py`** (`+3 −2`) — only `#!` annotation comments added inside `ClipAggregation.forward`; no logic change.

> **Key accuracy note.** Returning features from *multiple* ViT blocks (`out_layers`) is **native** to upstream V-JEPA2's `VisionTransformer`. This fork adds **no** core-model code to get per-layer features; it only wraps the existing `out_layers` API (`evals/analysis/modelcustom/vit_encoder_multilayer.py:51-58`, which injects `out_layers` from the config's `stages`).

### Edit-free `eval_name` routing (why `main.py` / `scaffold.py` are untouched)

`evals/scaffold.py` dynamically imports the eval module named by the config's `eval_name`:

```python
# evals/scaffold.py:18-19
import_path = f"evals.{eval_name}.eval"
return importlib.import_module(import_path).main(args_eval=args_eval, resume_preempt=resume_preempt)
```

So `eval_name: analysis` → `evals/analysis/eval.py` and `eval_name: analysis_vlm` → `evals/analysis_vlm/eval.py`, with **zero changes** to `main.py` / `main_distributed.py` / `scaffold.py`. New capability is a new package selected by a string, never a patch to the launcher.

### Complete change surface (by area)

The table below is **exhaustive** against this exact filter so a reader can reproduce it from `git`:

```bash
git diff --name-status 204698b HEAD \
  | grep -vE 'build/lib/' | grep -v 'configs/eval_2_1' | grep -v '/logs/'
```

Legend: **A** = added file · **M** = modified upstream file. (There are **no** untracked in-scope files — the tree is clean; all deliverables that live outside version control are listed separately below.)

**`evals/analysis/` — clip / V-JEPA harness (7 files, all A)**

| Status | Path | Role |
|---|---|---|
| A | `evals/analysis/__init__.py` | Package marker. |
| A | `evals/analysis/eval.py` | Clip/V-JEPA layer-wise probing driver (`eval_name: analysis`). |
| A | `evals/analysis/probes.py` | `LinearProbe` + `build_probe` / `probe_name` (**shared** by both harnesses). |
| A | `evals/analysis/plotting.py` | Layer-fraction metric plot (accuracy or R²), PEZ shading, peak/elbow markers (**shared**). |
| A | `evals/analysis/modelcustom/__init__.py` | Package marker. |
| A | `evals/analysis/modelcustom/vit_encoder_multilayer.py` | `MultiLayerClipAggregation` — one `(B,N,D)` per requested block via native `out_layers`. |
| A | `evals/analysis/attention_hooks.py` | SDPA monkey-patch: per-head attention-distance **capture** + local-attention **ablation** machinery (`build_ablation_bias`, `AttentionDistanceCollector`, `_find_rope_attn`). |

**`evals/analysis_vlm/` — unified harness + modes (12 files, all A)**

| Status | Path | Role |
|---|---|---|
| A | `evals/analysis_vlm/__init__.py` | Package marker. |
| A | `evals/analysis_vlm/eval.py` | Unified harness: vjepa/llavavideo/qwen3vl · classification+regression · feature cache · modes dispatch (**814 lines**). |
| A | `evals/analysis_vlm/cache.py` | Frozen-feature RAM cache (`tokens`/`pooled`/`framewise`) + thread prefetcher. |
| A | `evals/analysis_vlm/data.py` | Raw-frame `RawVideoDataset` + unpadded DDP shard sampler for VLMs. |
| A | `evals/analysis_vlm/loadutil.py` | HF weight-location resolver (repo-id + cache_dir, offline-first). |
| A | `evals/analysis_vlm/probes.py` | `TemporalLinearProbe`, `TemporalAttentiveClassifier` (learnable / RoPE temporal pos). |
| A | `evals/analysis_vlm/modelcustom/__init__.py` | Package marker. |
| A | `evals/analysis_vlm/modelcustom/llava_video_encoder.py` | LLaVA-Video SigLIP tower + projector, no 7B LLM. |
| A | `evals/analysis_vlm/modelcustom/qwen3vl_encoder.py` | Qwen3-VL vision ViT (blocks / merger / deepstack), no 4B LLM. |
| A | `evals/analysis_vlm/modes/__init__.py` | Mode registry + `AnalysisContext` dataclass + `run_modes` dispatch. |
| A | `evals/analysis_vlm/modes/attention_distance.py` | **Phase 1** mode: layer×head attention distance → Fig-3 heatmap + Fig-19 line plot. |
| A | `evals/analysis_vlm/modes/REPRODUCTION_PLAN.md` | Roadmap for remaining paper-repro modes (Phases 2–5). |

**`src/` + `evals/video_classification_frozen/` — upstream, modified (5 files, all M)**

| Status | Path | Delta |
|---|---|---|
| M | `src/datasets/video_dataset.py` | `uniform_sampling` frame selection (`+13`, default off). |
| M | `src/datasets/data_manager.py` | `uniform_sampling` passthrough (`+2`). |
| M | `evals/video_classification_frozen/eval.py` | `uniform_sampling` passthrough in `make_dataloader` (`+2`). |
| M | `src/models/vision_transformer.py` | comment only (`+1`, line 188); `out_layers` was already upstream. |
| M | `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py` | `#!` annotations only (`+3 −2`). |

**Configs — `configs/analysis/` (22 YAMLs, all A; `logs/` subdirs excluded by the `/logs/` filter)**

| Status | Path | Role |
|---|---|---|
| A | `configs/analysis/analysis_TEMPLATE.yaml` | Documented template for every knob. |
| A | `configs/analysis/{vjepa,llavavideo,qwen3vl}_analysis.yaml` | Per-model classification probing (3). |
| A | `configs/analysis/vjepa_regression.yaml` | R² regression (speed / direction / accel). |
| A | `configs/analysis/InsPhys2/{vjepa,llavavideo,qwen3vl}_analysis.yaml` | IntPhys-2 per-model runs (3). Dir is literally spelled **`InsPhys2`** in git (typo for IntPhys2). |
| A | `configs/analysis/toy_dataset/{vjepa,llavavideo,qwen3vl}_{velocity,acceleration,combined}.yaml` + `vjepa_combined_attentive.yaml` | PIL/ffmpeg toy-ball variants (10). |
| A | `configs/analysis/blender_toy_dataset/{vjepa_combined,vjepa_attn_distance,llavavideo_combined,qwen3vl_combined}.yaml` | Blender toy-set variants (4), incl. `vjepa_attn_distance.yaml` (the `modes` run). |

*(Exactly **22** `configs/analysis/*.yaml` are tracked — `git ls-files 'configs/analysis/*.yaml' | wc -l` = 22, and all 22 appear in the diff. `configs/analysis/logs/` and `configs/analysis/toy_dataset/logs/` are git-ignored run outputs — checkpoints + plots — and are removed by the `/logs/` filter.)*

**Misc tracked (repo root + docs, all A except `.gitignore`)**

| Status | Path | Role |
|---|---|---|
| A | `debug_infer.py` | Minimal single-GPU encoder-forward debug script (no probes/labels) — smoke-tests a checkpoint + `MultiLayerClipAggregation` forward in isolation. |
| A | `Interpreting Physics in Video World Models.pdf` | The reference paper being reproduced. |
| A | `z_tech/*.md` + `z_tech/README.md` | This 15-part technical reference (you are reading `01`). |
| M | `.gitignore` | Adds `checkpoint/`, `sample_video/`, `z_scripts`, `*csv`, `*.json`, `configs/analysis/logs`, `configs/analysis/toy_dataset/logs/`, `data_gen/`, `configs/z_tak_attentive_probing`. Note `*csv` is what ignores the whole **`data_csv/`** directory. |

### Deliverables outside the diff (git-ignored)

These are real, load-bearing deliverables the harness produces/consumes, but they **do not appear in the diff** because `.gitignore` excludes them (`z_scripts`, `data_gen/`, and the `*csv` glob covering `data_csv/`). Listed for completeness:

| Path | Role |
|---|---|
| `data_gen/make_physics_toy.py` | 2D PIL/ffmpeg toy-ball generator (velocity/accel grids; anti-shortcut nuisance randomization). |
| `data_gen/make_physics_blender.py` | Paper-faithful Blender (bpy) photorealistic sphere generator (Cycles). |
| `data_csv/make_regression_targets.py`, `make_blender_targets.py` | Build `targets.npy` + rewrite split CSVs so `label = npy row`; emit the `regression.variables` block. |
| `data_csv/{toy_physics,blender_toy,R2R_4way_1500,IntPhys2}/` | Split CSVs + `targets.npy` per dataset. Data dir is spelled **`IntPhys2`** (vs the `InsPhys2` config dir — same dataset, two spellings). |
| `z_scripts/run_analysis_vjepa.sh`, `run_analysis_vlm.sh` | SLURM launchers for the probing sweeps (set `CONDA_ENV` + `CONFIG`). |
| `z_scripts/run_attn_distance_vjepa.sh` | SLURM launcher for the `attention_distance` mode (**single-GPU on purpose** — see gotchas). |

---

## Design & data flow

### The two harnesses (and why `analysis_vlm` is a strict superset)

Both share one core loop: **frozen encoder → per-layer/stage features → one probe head per `(stage × probe-spec × variable)` → joint train → `stage_val_acc.png` + `summary.json`**. They share `probes.py` and `plotting.py` (the vlm harness imports `evals.analysis.probes` / `evals.analysis.plotting`). `analysis_vlm` adds every axis the paper needs on top.

| | `evals/analysis` | `evals/analysis_vlm` |
|---|---|---|
| `eval_name` | `analysis` | `analysis_vlm` |
| Backbones | V-JEPA2 ViT only (`data_mode="clip"`) | `vjepa` (reuses the clip backend) · `llavavideo` · `qwen3vl` (`_BACKENDS`, `evals/analysis_vlm/eval.py:62`) |
| Data path | stock clip `VideoDataset` (normalized 5-D clips) | clip **or** raw-frame loader (`evals/analysis_vlm/data.py`) chosen by `data_mode` |
| Task | classification (accuracy) | classification **or** regression → R² (NaN-masked) |
| Probes | `linear`, `attentive` (`evals/analysis/probes.py`) | + temporal-aware `TemporalLinearProbe` / `TemporalAttentiveClassifier` (`evals/analysis_vlm/probes.py`) |
| Feature cache | no (re-encodes each epoch) | **yes** — one deterministic pre-pass, epochs run over the cache (`evals/analysis_vlm/cache.py`) |
| Optimizer | one AdamW per head | one **fused** AdamW, one param-group per head (`_init_opt_fused`, `evals/analysis_vlm/eval.py:617`) |
| Post-hoc modes | — (`attention_hooks.py` lives here, but is *driven* by the vlm harness) | `experiment.analysis.modes` registry (`evals/analysis_vlm/modes/`) |

Active development lives in `analysis_vlm`; `analysis` is the simpler, original clip-only version that the vlm harness borrows probes/plotting/hooks from.

### End-to-end data flow

```
video (.mp4)                 CSV: "<abs_path> <int_label>"  (no header, space-delimited)
   │
   ├─ clip path (V-JEPA):  VideoDataset → normalized clip tensors (5-D; shorter-side crop OR direct resize)
   └─ raw path  (VLM):     RawVideoDataset → list of (T,H,W,C) uint8 frames (backend preprocesses natively)
   │
   ▼
FROZEN encoder  (torch.no_grad, never trained)
   ├─ vjepa:      vit_encoder_multilayer.MultiLayerClipAggregation → list over out_layers, each (B,N,D)
   ├─ llavavideo: SigLIP tower (no 7B LLM)  → list over stages, (B, T*729, 1152) / projector 3584
   └─ qwen3vl:    Qwen3-VL ViT (no 4B LLM)  → list over stages, block / merger / deepstack features
   │
   ▼
(optional) FEATURE CACHE — one deterministic pre-pass, per-stage features held in RAM (fp16)
   │                        cache_pooling: tokens | pooled | framewise   (evals/analysis_vlm/cache.py)
   ▼
one PROBE head per (stage × probe-spec × regressed-variable) — trained JOINTLY, encoder detached
   │   classification → CrossEntropy → accuracy
   │   regression     → per-column MSE (NaN-masked) → R² (all-reduced)
   ▼
per-layer metric → summary.json + stage_val_acc.png   (x = layer fraction, one line per probe/variable)
   │
   └─ (optional, RANK 0 ONLY) modes: attention_distance (Phase 1) → heatmap + line plot
                                      [attention_ablation, orthogonal_probe_sequence,
                                       steering, direction_tuning — Phases 2–5, pending]
```

The integer CSV label doubles as a **row index into an `(N, D)` targets `.npy`**, so the *same* dataloaders serve both classification and regression — the harness maps `label → target vector`. No dataset code changes between tasks.

### The additive / default-off principle

Every capability is gated so an **unconfigured run is byte-for-byte unchanged**:

- **New eval packages, not edits.** Selected only via `eval_name`; the stock evals are untouched.
- **`modes` block** (`evals/analysis_vlm/eval.py:565-590`): absent ⇒ `modes_cfg == {}` ⇒ the whole block is skipped and `evals.analysis_vlm.modes` is never even imported. In-code comment: *"existing runs behave byte-for-byte identically."* Runs on **rank 0 only**.
- **`skip_base_probe`** (default `False`, `eval.py:501-504`): when true, sets `num_probe_epochs = 0` so an encoder-only mode (like `attention_distance`) skips probe training entirely; default keeps the normal epoch count.
- **Attention hooks** (`evals/analysis/attention_hooks.py`): a monkey-patch of `F.scaled_dot_product_attention` installed **only** inside a context manager, torn down on exit (`_ORIG_SDPA` restored in `finally`). When capture-only, the encoder output is a **detached side computation** → bit-identical to baseline.
- **Per-flag defaults**: `task: classification`, `cache_features: false`, `uniform_sampling: False`, `save_optimizer: false`, `skip_base_probe: false`.

---

## Key code

- **Routing** — `evals/scaffold.py:18` builds `import_path = f"evals.{eval_name}.eval"` and calls its `main`. This is the *only* thing that couples a config to a package.
- **Backend registry** — `evals/analysis_vlm/eval.py:62` `_BACKENDS = { vjepa | llavavideo | qwen3vl → (default module, data_mode) }`; unknown `analysis.model` raises with the valid list (`:111-113`).
- **Modes dispatch (default-off seam)** — `evals/analysis_vlm/eval.py:565-590`:

  ```python
  # evals/analysis_vlm/eval.py:565-590
  modes_cfg = args_analysis.get("modes") or {}
  if modes_cfg and rank == 0:
      from evals.analysis_vlm.modes import AnalysisContext, run_modes
      ctx = AnalysisContext(
          encoder=encoder, device=device, folder=folder, ...,
          targets_npy=(tpath if task == "regression" else None),
          col_mu=(mu if task == "regression" else None), col_sd=(sd if ...),
          encode_clip=lambda d: _encode(encoder, d, device, data_mode, use_bfloat16),
          make_val_clip_loader=lambda: _split_loader(val_data_path[0], training=False, workers=0)[0],
      )
      run_modes(modes_cfg, ctx)
  ```
  `AnalysisContext` is a **read-only** dataclass handing each mode the frozen encoder, trained heads + metadata, cached features (if any), the standardization stats to recover raw targets, and two closures back into `eval.py` scope (`encode_clip`, `make_val_clip_loader`).

- **Mode registry** — `evals/analysis_vlm/modes/__init__.py`: `@register(name)` populates `_REGISTRY`; `_import_modes()` lazily imports mode files so they self-register (today: only `attention_distance`; Phases 2–5 are commented placeholders). `run_modes` skips `False` / `{enabled: false}`, normalizes `True`/`None` → `{}`, and raises on an unknown name.
- **The one wired mode** — `evals/analysis_vlm/modes/attention_distance.py`: asserts `ctx.data_mode == "clip"`, finds the 24 `RoPEAttention` blocks (`_find_rope_attn`), runs `max_batches` val batches under `attention_hooks(...)`, and writes `attention_distance.json` + two plots. See [12 — Analysis modes](12-analysis-modes.md).
- **Native per-layer tap wrapper** — `evals/analysis/modelcustom/vit_encoder_multilayer.py:51-58` reads `wrapper_kwargs.out_layers` (injected from `stages`) and passes it straight to the upstream `VisionTransformer`.

---

## Configuration

Both harnesses read one YAML via `python -m evals.main --fname <cfg>`. A representative V-JEPA2 classification run (`configs/analysis/vjepa_analysis.yaml`, comments condensed):

```yaml
eval_name: analysis_vlm            # -> evals/analysis_vlm/eval.py  (routing key)
folder: /.../configs/analysis/logs # outputs at <folder>/analysis_vlm/<tag>/
num_workers: 8
resume_checkpoint: false
val_only: false
tag: vjepa2_shape_color

experiment:
  analysis:
    model: vjepa                   # vjepa | llavavideo | qwen3vl (picks backend + data_mode)
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

Top-level and cross-cutting keys:

| Key | Meaning | Default | Allowed values |
|---|---|---|---|
| `eval_name` | Routing key → `evals/<name>/eval.py` | — (required) | `analysis`, `analysis_vlm` |
| `experiment.analysis.model` | Backend selector (`_BACKENDS`) | — (required for `analysis_vlm`) | `vjepa`, `llavavideo`, `qwen3vl` |
| `experiment.analysis.task` | Probe objective | `classification` | `classification`, `regression` |
| `experiment.analysis.skip_base_probe` | Skip probe training (encoder-only modes) | `false` | `true`, `false` |
| `experiment.analysis.modes` | Post-hoc mode map (`{name: cfg}`) | *absent* → skipped | `attention_distance: {...}` (Phase 1); Phases 2–5 pending |
| `data.uniform_sampling` | Evenly sample `fpc` frames across the whole video | `false` | `true`, `false` |
| `optimization.cache_features` | Encode once, train over RAM cache | `false` | `true`, `false` |
| `optimization.cache_pooling` | Cache granularity | `tokens` | `tokens`, `pooled`, `framewise` |

- **Regression** (`configs/analysis/vjepa_regression.yaml`): set `task: regression` and add a `regression:` block with `targets_npy` + named `variables` (each a column-slice → its own R² curve). See [04 — Probes, regression & NaN masking](04-probes-regression-nanmask.md).
- **VLMs**: set `model: llavavideo|qwen3vl`, point `checkpoint` at an HF repo id, set `wrapper_kwargs.cache_dir`, and use structured `stages` (e.g. `{vision_encoder: all, after_projector: true}` for LLaVA; `{vision_encoder: all, after_merger: true, deepstack: [5,11,17]}` for Qwen). See [08 — VLM encoders](08-vlm-encoders.md).
- **Modes**: the only complete example today is `configs/analysis/blender_toy_dataset/vjepa_attn_distance.yaml` (encoder-only, `skip_base_probe: true`, `num_epochs: 0`, `cache_features: false`) — detailed in [12 — Analysis modes](12-analysis-modes.md) and [13 — Configs reference](13-configs-reference.md).

---

## Invariants & gotchas

- **`stages` is polymorphic.** Accepts `"all"`, a `[int,...]` block list, a structured dict (`{vision_encoder: [...]|all, <toggles>: true}`), or a legacy list of concrete stage-name strings. For the V-JEPA clip path only `vision_encoder` is meaningful; extra keys are warned + ignored (`eval.py:125-138`).
- **Plot x-axis differs by path.** Clip path uses the block index as x; raw/VLM path uses the stage position (so `block_5` vs `deepstack_5` don't collide on x=5), with the stage name on the tick. `summary.json` always keeps exact per-stage metrics.
- **`cache_pooling='pooled'` ⇒ linear probes only.** It caches only `[mean‖max]`, so attentive/framewise probes are rejected up front (`eval.py:302-313`). `tokens` supports all probes but scales with `N × #stages` and can OOM host RAM — guarded by `cache_max_gb` (`cache.py:139`).
- **Cache = "no-augment probing."** The pre-pass is deterministic (`training=False`) for *both* splits, so train-time augmentation is dropped — intended for frozen-encoder probing.
- **Small-split `drop_last` trap.** Stock `init_data`/`make_videodataset` ignore `drop_last` and default it to `True`, silently dropping partial batches — fatal for tiny val splits. The vlm harness force-sets `drop_last=False` after loader creation (`eval.py:414-419`).
- **`out_layers` is upstream, not fork code.** Per-layer extraction reuses V-JEPA2's native `VisionTransformer(out_layers=...)` (`src/models/vision_transformer.py:205-209`); the fork's ViT edit is a lone comment (line 188).
- **Ablation machinery exists but is not yet an exposed mode.** `evals/analysis/attention_hooks.py:120` defines `build_ablation_bias` (spatial/temporal/combined local-attention masks), but **no mode dispatches it** — `attention_ablation` is **Phase 5, pending**. Only `attention_distance` (capture-only) is wired into the registry today. Capture is bit-identical to baseline; ablation *deliberately* changes the encoder output (that is the experiment).
- **Post-hoc modes run on rank 0 only.** The dispatch block is gated by `rank == 0` (`eval.py:569`); other ranks proceed to the DDP `barrier`. Therefore **multi-GPU gives no speedup for modes** — see the launcher gotcha below. Multi-GPU only accelerates the base probing sweep / feature-cache pre-pass.
- **`attention_distance` is clip-only.** It asserts `ctx.data_mode == "clip"` and requires `RoPEAttention` blocks — it works on the V-JEPA backend, not the VLM towers.
- **VLM backends run in separate conda envs.** `vjepa2` / `lmms_eval_llavavideo` / `lmms_eval_py311_2.7`. Only the selected backend is imported (lazily via `module_name`), so conflicting heavy deps never co-load.
- **Naming inconsistency (by design of the repo, flagged here):** the config dir is `configs/analysis/InsPhys2/` while the data dir is `data_csv/IntPhys2/`. Both refer to the **IntPhys-2** dataset; the config-dir name is a typo. Use the exact on-disk spelling per location.
- **`data_csv/` is git-ignored via the `*csv` glob.** Any path ending in `csv` (including the directory name `data_csv`) is ignored, which is why the target-builder `.py` files there are not tracked.

---

## Cross-references

- [02 — analysis_vlm harness](02-analysis-vlm-harness.md) — the unified driver in full.
- [03 — Feature caching & pooling](03-feature-caching-and-pooling.md) — `tokens`/`pooled`/`framewise`, prefetch, `cache_max_gb`.
- [04 — Probes, regression & NaN masking](04-probes-regression-nanmask.md) — heads, R², per-column masking, temporal probes.
- [05 — analysis (clip) harness](05-analysis-clip-harness.md) — the simpler original harness.
- [06 — Data pipeline changes](06-data-pipeline-changes.md) — `uniform_sampling`, `resize_mode`, loaders.
- [07 — Plotting](07-plotting.md) — `plot_layer_val_acc`, PEZ shading, elbow markers.
- [08 — VLM encoders](08-vlm-encoders.md) — LLaVA-Video / Qwen3-VL towers, stages, `loadutil`.
- [09 — Blender toy dataset](09-blender-toy-dataset.md) — the photorealistic sphere generator.
- [10 — Datasets, CSVs & targets](10-datasets-csv-targets.md) — CSV format, `targets.npy`, `regression.variables`.
- [11 — Attention hooks](11-attention-hooks.md) — the SDPA monkey-patch, capture + ablation internals.
- [12 — Analysis modes](12-analysis-modes.md) — the registry, `AnalysisContext`, `attention_distance`.
- [13 — Configs reference](13-configs-reference.md) — every knob across all 22 YAMLs.
- [14 — Reproduction status & findings](14-reproduction-status-and-findings.md) — Phase status + the reproduced Fig-3 result.
