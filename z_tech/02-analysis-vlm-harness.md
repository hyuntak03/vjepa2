# 02 — `analysis_vlm` harness (eval flow)

## Purpose

`evals/analysis_vlm/` is a **unified frozen-encoder probing harness** across model
families. For a chosen encoder it extracts features at a set of **stages** (ViT
layers / VLM sub-modules), builds one probe head per **(stage × probe-spec ×
variable)**, trains all heads **jointly** on the frozen features, and reports a
`[stage × probe]` accuracy (classification) or R² (regression) matrix plus a
per-stage plot.

Three encoder backends are selectable from config; each lives in its **own conda
env** and is imported lazily so heavy/conflicting deps never co-load:

| `analysis.model` | backend module | data mode | env |
|---|---|---|---|
| `vjepa` | `evals.analysis.modelcustom.vit_encoder_multilayer` | `clip` | `vjepa2` |
| `llavavideo` | `evals.analysis_vlm.modelcustom.llava_video_encoder` | `raw` | `lmms_eval_llavavideo` |
| `qwen3vl` | `evals.analysis_vlm.modelcustom.qwen3vl_encoder` | `raw` | `lmms_eval_py311_2.7` |

Routed by `eval_name: analysis_vlm` — `evals/scaffold.py:19` dynamically imports
`evals.analysis_vlm.eval` and calls `main()`, so upstream `evals/main.py` /
`scaffold.py` are untouched.

## What changed vs upstream V-JEPA2

The **entire `evals/analysis_vlm/` package is new** (not present at upstream
`204698b`; `git ls-tree -r 204698b -- evals/analysis_vlm` returns nothing —
added in commit `91fa127 "analysis module added"`). The sibling
`evals/analysis/` (V-JEPA layer-wise probing) is likewise new. No upstream file
is modified for routing — `scaffold.py`'s existing `eval_name` dispatch picks it
up.

**New files in this subsystem (this doc's scope):**

| File | Role |
|---|---|
| `evals/analysis_vlm/eval.py` | `main()` control flow — config → encoder → heads → train/eval → outputs |
| `evals/analysis_vlm/data.py` | raw-frame dataloader for VLM backends (`RawVideoDataset`, `make_raw_dataloader`) |
| `evals/analysis_vlm/loadutil.py` | HF-weight location resolver shared by VLM backends |
| `evals/analysis_vlm/cache.py` | one-shot feature cache (see **§03**) |
| `evals/analysis_vlm/probes.py` | temporal-aware probe heads (see **§04**) |
| `evals/analysis_vlm/modelcustom/{llava_video_encoder,qwen3vl_encoder}.py` | VLM backends (see **§12** for stages) |
| `evals/analysis_vlm/modes/` | post-hoc analysis modes (see **§12**) |

**Reused from upstream (imported, not copied):** `init_module`
(`evals/video_classification_frozen/models.py:14`), `make_dataloader` / `init_data`,
`WarmupCosineLRSchedule` / `CosineWDSchedule`, `AllReduceSum`, `init_distributed`,
`CSVLogger`, `robust_checkpoint_loader`, and `build_probe` / `probe_name`
(`evals/analysis/probes.py:56,84`). The V-JEPA (`clip`) path reuses the stock
`VideoDataset` + `make_dataloader` verbatim.

---

## `main()` control flow — `evals/analysis_vlm/eval.py:85`

### 1. Config parsing

Top-level (`args_eval`): `val_only`, `folder`, `resume_checkpoint` (or preempt),
`tag`, `num_workers` (default 12). `model_kwargs` yields `checkpoint`
(local `.pth` for vjepa **or** HF repo id for VLM), `cache_dir`, `module_name`,
`pretrain_kwargs` (`args_model`), `wrapper_kwargs` (`args_wrapper`).

Three `experiment.*` sub-sections drive the run: `experiment.analysis`,
`experiment.data`, `experiment.optimization`.

### 2. Model registry → backend + data mode — `eval.py:62`, `:108`

```python
_BACKENDS = {
    "vjepa":      ("evals.analysis.modelcustom.vit_encoder_multilayer", "clip"),
    "llavavideo": ("evals.analysis_vlm.modelcustom.llava_video_encoder", "raw"),
    "qwen3vl":    ("evals.analysis_vlm.modelcustom.qwen3vl_encoder",     "raw"),
}
```

- `analysis.model` (if set) picks `module_name` and `data_mode` from `_BACKENDS`
  (each defaultable/overridable via `model_kwargs.module_name` /
  `analysis.data_mode`).
- If `data_mode` is still unset: `"raw"` when `module_name` contains
  `"analysis_vlm"`, else `"clip"` (`eval.py:117`).
- `module_name` must resolve or it asserts.

`data_mode` selects **both** the dataloader and how the encoder forward is called
(see `_encode`, below). `clip` = V-JEPA tensor path; `raw` = VLM native-preprocess
path.

### 3. Stages resolution — `eval.py:124`

`analysis.stages` (legacy alias `analysis.layers`). Accepted forms:

- **structured dict** `{vision_encoder: [ints]|"all", <toggle>: true|[ints]}` —
  only `vision_encoder` carries the per-layer list; other keys are backend stage
  toggles (`after_merger`, `deepstack`, …).
- **shorthand** `"all"` or `[int, …]`.

Path split:

- **`clip`/vjepa** (`eval.py:125`): dict must contain `vision_encoder` (else
  raises; extra keys warn+ignored — the ViT backbone has no other stages).
  Resolved via `_resolve_layers` (`eval.py:71`) — `"all"` expands to
  `range(_VIT_DEPTH[model_name])` (`vit_large=24`, `vit_huge=32`, `vit_giant=40`,
  …; `"all"` on an unknown model name raises). Result injected as
  `wrapper_kwargs.out_layers`; `stages` is the concrete int list.
- **`raw`/VLM** (`eval.py:140`): the whole spec is handed to the backend as
  `wrapper_kwargs.out_stages`; `cache_dir` is copied into `wrapper_kwargs` if
  present (lets the backend resolve a repo-id checkpoint). `stages = None` here,
  then read back from `encoder.stages` **after** `init_module` (`eval.py:276`).

### 4. Probes / plot / task

- `analysis.probes` — non-empty list; each spec builds a head per stage
  (`eval.py:147`).
- `analysis.plot` (default `false`), `analysis.plot_pez` (`[lo,hi]` layer-fraction
  band to shade; validated `0≤lo<hi≤1`) → **§07**.
- `analysis.task` — `"classification"` (CrossEntropy → accuracy%) or
  `"regression"` (MSE → R²). Regression config (`eval.py:189`):
  - `regression.targets_npy` — `(N,D)` `.npy`; **the CSV integer label indexes
    this array** (label → target vector), so the dataloaders stay unchanged.
  - Columns are **standardized per-column, NaN-aware** (`nanmean`/`nanstd`); R² is
    invariant to this affine transform. NaNs stay NaN → masked out per head.
  - `regression.variables` — list of `{name, cols}`; each becomes its own head /
    R² curve. Default: one variable spanning all columns. See **§04**.

### 5. Optimization — `eval.py:216`

`experiment.optimization`: `batch_size`, `num_epochs`, `use_bfloat16`,
`default_head` (per-probe optimization fallback), plus:

- `save_optimizer` (default **false**) — AdamW state in `latest.pt` is huge for
  high-dim attentive probes; off = probe weights only.
- `cache_features` (default **false**), `cache_pooling` (`"tokens"`|`"pooled"`|
  `"framewise"`, default `"tokens"`), `cache_max_gb` (default 64) → **§03**.

`_opt_kwargs(spec)` (`eval.py:231`) merges `default_head` under each probe's own
`optimization` block → `ref_wd/final_wd/start_lr/ref_lr/final_lr/warmup`.

### 6. Encoder init — `eval.py:267`

`init_module(module_name, …, model_kwargs=args_model, wrapper_kwargs=args_wrapper)`
builds the encoder, `.eval()`s it and sets `requires_grad=False` (frozen). For
`raw`, `stages = list(encoder.stages)` is read back; `embed_dims` come from
`encoder.embed_dims` (VLM backends expose per-stage dims; V-JEPA falls back to
`[embed_dim]*len(stages)`). Backends also expose `num_temporal` (used by
temporal-aware probes).

### 7. Head construction — one per (stage × probe × variable) — `eval.py:287`

For each stage, a plot **x-value** and **tag** are chosen:

- `clip`: `stage_tag = f"L{block:02d}"`, `layer_val = block index` (unique).
- `raw`: `stage_tag = str(stage_name)`, `layer_val = stage_position` (position
  avoids `block_5` vs `deepstack_5` colliding on x=5; the name labels the tick).

`_build(out_dim)` (`eval.py:315`) dispatches the head type:

| condition | head |
|---|---|
| `cache_features` & `cache_pooling=="pooled"` | `cache.PooledLinearProbe` (linear only) |
| `type=linear` & `pooling` starts `framewise` | `probes.TemporalLinearProbe` (needs `encoder.num_temporal`) |
| `type=attentive` & `temporal_pos∈{learnable,rope}` | `probes.TemporalAttentiveClassifier` (needs `num_temporal`) |
| else | upstream `build_probe(spec, …)` (linear / attentive) |

`out_dim = len(var_cols)` for regression, else `num_classes`. Each head is
`.to(device)` and DDP-wrapped (`static_graph=True`) when a process group exists.
Head record:

```python
heads.append(dict(name=name, layer=layer_val, layer_pos=stage_pos,
                  probe=pname, series=series,
                  stage=stage_tag, module=module, tcols=var_cols))
```

- `name = f"{stage_tag}_{pname}{__var}"`, de-collided with `#2`, `#3`, … on
  duplicate specs.
- `layer_pos` indexes into the per-stage feature list at forward time.
- `series` = plot-line grouping: `pname` (classification), `var_name`
  (regression, single probe), or `var_name·pname` (regression, multiple probes).

**Guards:** `cache_pooling="pooled"` caches only pooled vectors → attentive or
`framewise` probes raise (use `cache_pooling="tokens"` or `cache_features=false`).
`temporal_pos`/`framewise` require a backend exposing `num_temporal` (VLM only;
V-JEPA already encodes time via RoPE) → raises otherwise.

### 8. Data loaders — `eval.py:383`

`_split_loader(root, training, …)` builds a single-split loader:

- `clip` + `resize_mode="resize"` → upstream `init_data` with a
  `_DirectResizeClipTransform` (direct resize to `resolution²`, aspect squashed,
  SigLIP-like; deterministic).
- `clip` + `resize_mode="crop"` (default) → upstream `make_dataloader`
  (shorter-side resize + center-crop).
- `raw` → `data.make_raw_dataloader` (see below).

> **Gotcha:** the clip path force-sets `ld.batch_sampler.drop_last = False`
> (`eval.py:418`) — `init_data`/`make_videodataset` default `drop_last=True`,
> which silently drops partial batches and can yield **0 batches** on small val
> splits. DDP `DistributedSampler` pads to equal per-rank counts, so batch counts
> stay aligned.

If `cache_features` (`eval.py:426`): one **deterministic pre-pass per split**
(`training=False`, `workers=0`) encodes features into a per-rank cache, then
`make_cached_loader` serves them; `run_mode="cached"`. Loaders are built
sequentially (train dropped before val) with `persistent_workers=False` to avoid
spawn-multiprocessing deadlock. Details in **§03**.

### 9. Optimizer — one fused AdamW — `eval.py:617`

`_init_opt_fused` builds **one** `AdamW` with **one param-group per head** (each
carrying its own `mc_*` LR/WD schedule keys), a single `WarmupCosineLRSchedule`,
`CosineWDSchedule`, and `GradScaler`. Numerically identical to
one-optimizer-per-head but collapses N `step()`/`zero_grad()` launches into one
(~25% off the cached step on many-head scans). Returned as **length-1 lists** so
the loop/checkpoint code (which iterates these lists) is unchanged.

### 10. Train / eval loop — `eval.py:498`

- `num_probe_epochs = 0 if analysis.skip_base_probe else num_epochs`
  (default-off; off ⇒ loop byte-identical to before) — used by encoder-only
  analysis modes that don't need trained probes.
- Each epoch: `train_sampler.set_epoch`, `run_one_epoch(training=True)` (skipped
  under `val_only`), then `run_one_epoch(training=False)`; `best_val[name]`
  tracked (floor `-inf`, since R² can be negative).
- Rank-0 side effects each epoch: append CSV row, rewrite `summary.json`,
  `save_checkpoint`. `val_only` breaks after the first pass.

`run_one_epoch` (`eval.py:674`):

- Calls `_encode` → per-stage feature list; each head reads `feats[h["layer_pos"]]`.
- **classification:** `CrossEntropyLoss`, returns per-head accuracy% via
  `AllReduceSum(correct)/total`.
- **regression:** masked-mean MSE per head (NaN target rows contribute 0 but stay
  in the graph so DDP static-graph structure is rank-identical); returns per-head
  R² = `1 − SS_res/SS_tot`, computed from all-reduced `ss_res / sum_y / sum_y2 /
  cnt` accumulators (per-head valid-sample stats, so one combined dataset can hold
  variables defined on different video subsets).
- Validation runs heads under `torch.no_grad()` — avoids arming the DDP
  static-graph reducer (a grad-enabled forward with no backward trips it next
  forward).

`_encode` (`eval.py:649`) has three branches by `data_mode`:

| mode | input | encoder call |
|---|---|---|
| `cached` | `(list[stage] tensors, labels)` | none — features already cached |
| `clip` | `(clips, labels, clip_indices)` | `encoder(clips, clip_indices)` under fp16 autocast, `no_grad` |
| `raw` | `(frames_list, labels)` | `encoder(frames_list)` in the backend's own (half) dtype, `no_grad` |

Returns `(feats: list[(B,N,D) detached], labels, bsz)`. Features keep the encoder
dtype (fp16) — upcasting all stages to fp32 here would double peak GPU memory
(OOM on all-layer scans).

### 11. Outputs — `<folder>/analysis_vlm/<tag>/`

| File | Written by | Contents |
|---|---|---|
| `log_r{rank}.csv` | `CSVLogger` (`eval.py:474`) | `epoch`, then `<head>_train` / `<head>_val` per head |
| `summary.json` | rank 0 each epoch (`eval.py:542`) | `epoch`, `model`, `data_mode`, `task`, `metric` (`r2`/`accuracy`), `variables`, `stages`, `head_names`, `val_acc`, `train_acc`, `best_val_acc` |
| `latest.pt` | `save_checkpoint` (`eval.py:485`) | `classifiers` (state dicts), `epoch`, `head_names`, `stages`, `batch_size`, `world_size`; `opt`/`scaler` only if `save_optimizer` |
| `stage_val_acc.png` | if `analysis.plot`, rank 0 (`eval.py:556`) | `plot_layer_val_acc(heads, best_val, …, metric, pez)` — see **§07** |

The folder path is `os.path.join(folder, "analysis_vlm/")` then `/<tag>` if `tag`
is set (`eval.py:259`).

### 12. Post-hoc analysis modes — `eval.py:565`

If `analysis.modes` is present (and rank 0), `evals.analysis_vlm.modes.run_modes`
runs with an `AnalysisContext` carrying encoder, heads, cache tensors, targets,
etc. **Absent ⇒ `modes_cfg == {}` ⇒ the whole block is skipped and nothing is
imported ⇒ existing runs behave byte-for-byte identically.** See **§12**.

Checkpoint resume (`eval.py:477`): loads `latest.pt`, replays the LR/WD schedule
`start_epoch*ipe` steps; weights-only checkpoints restart the optimizer.

---

## `evals/analysis_vlm/data.py` — raw-frame dataloader (VLM path)

VLM vision encoders need their **own native preprocessing** (SigLIP 384 resize /
Qwen smart-resize), so this loader does **no** normalize/crop — it just samples
raw uint8 frames and hands them to the backend.

- `RawVideoDataset` (`data.py:28`): reads the same CSV as the V-JEPA path
  (`<abs_mp4_path> <int_label>`, space-delimited, no header; falls back to `::`).
  `_load` uniformly samples `frames_per_clip` frames across the whole video via
  `np.linspace(0, n-1, T)` (repeats gracefully if `n < T`) → `(T,H,W,C)` uint8.
  `__getitem__` retries up to 8× on corrupt/missing videos with a random fallback.
- `_collate` (`data.py:63`): keeps frames as a **list** of `(T,H,W,C)` tensors
  (videos may differ in H/W — the backend resizes each natively); labels stacked.
- `_UnpaddedShardSampler` (`data.py:69`): strided `rank::world_size` shard with
  **no padding**, so the rank-union is exactly the dataset once each — required so
  `AllReduceSum`'d accuracy is exact (a `DistributedSampler` pads with duplicates
  and skews it). Used for **eval / cache pre-pass**.
- `make_raw_dataloader` (`data.py:84`): `DistributedSampler(shuffle=True)` for
  training, `_UnpaddedShardSampler` for eval. `persistent_workers` defaults on for
  training but is forced off for the one-shot cache pre-pass (`persistent=False`).

> **Invariant:** raw sampling is **always uniform** across the full clip
> (`frame_step` is a clip-path-only concept). The clip path's `uniform_sampling`
> flag is what makes V-JEPA match this behavior.

## `evals/analysis_vlm/loadutil.py` — weight resolver (VLM backends)

Lets a VLM config point at weights flexibly. Used **only** by the two VLM
backends, not by `eval.py` directly.

- `resolve_model_dir(checkpoint, wrapper_kwargs)` (`loadutil.py:30`): candidate =
  `wrapper_kwargs.pretrained or checkpoint`. If it's a local dir →
  `find_snapshot`; otherwise treat it as an **HF repo id** and
  `snapshot_download(..., cache_dir=cache_dir, local_files_only=True)`
  (offline-first, downloads only on miss).
- `find_snapshot(path)` (`loadutil.py:18`): accepts a snapshot dir, an HF-cache
  `models--…` root (globs `snapshots/*`), or any dir with `config.json`.

This is why VLM configs set `checkpoint: <HF repo id>` + `cache_dir: <cache root>`
(e.g. `Qwen/Qwen3-VL-4B-Instruct` + `/data/dataset/…`), and why `eval.py` copies
`cache_dir` into `wrapper_kwargs` for the raw path (`eval.py:143`).

---

## Config

Full schema for a run. Keys read by `eval.py` (defaults in brackets; `[req]` =
required/asserted).

### Top level
| key | default | notes |
|---|---|---|
| `eval_name` | `[req]` | must be `analysis_vlm` (scaffold routing) |
| `folder` | `[req]` | output root; run dir = `<folder>/analysis_vlm/<tag>/` |
| `tag` | `None` | sub-folder / run name |
| `num_workers` | `12` | DataLoader workers |
| `resume_checkpoint` | `false` | resume from `latest.pt` |
| `val_only` | `false` | eval only (skip probe training) |

### `experiment.analysis`
| key | default | notes |
|---|---|---|
| `model` | — | `vjepa` \| `llavavideo` \| `qwen3vl` (picks backend + data mode) |
| `data_mode` | auto | `clip` \| `raw` (override) |
| `stages` (alias `layers`) | `[req]` | dict `{vision_encoder: […]\|"all", <toggle>}` or `"all"`/`[int,…]` |
| `probes` | `[req]` | non-empty list of probe specs (**§04**) |
| `task` | `classification` | or `regression` |
| `regression.targets_npy` | — | `(N,D)` `.npy`; CSV int label indexes it |
| `regression.variables` | all cols | `[{name, cols:[…]}, …]` |
| `plot` | `false` | write `stage_val_acc.png` |
| `plot_pez` | `None` | `[lo,hi]` layer-fraction shade band |
| `skip_base_probe` | `false` | 0 probe epochs (encoder-only modes) |
| `modes` | `{}` | post-hoc modes (**§12**); absent ⇒ skipped |

### `experiment.data`
| key | default | notes |
|---|---|---|
| `dataset_type` | `VideoDataset` | clip path only |
| `dataset_train` / `dataset_val` | `[req]` | CSV `"<mp4> <int_label>"` |
| `num_classes` | `[req]` (classification) | regression head dim = `len(cols)` instead |
| `resolution` | `224` | clip path |
| `resize_mode` | `crop` | `crop`\|`resize` (clip path only; raises otherwise) |
| `frames_per_clip` | `16` | even (tubelet=2 for V-JEPA); raw: even, `grid_t=T/2` |
| `frame_step` | `4` | clip contiguous stride (ignored if `uniform_sampling`) |
| `uniform_sampling` | `false` | clip: sample T frames evenly across whole video (raw is always uniform) |
| `num_segments` / `num_views_per_segment` | `1` / `1` | clip path |
| `clip_duration` / `normalization` | `None` / `None` | clip path |

### `experiment.optimization`
| key | default | notes |
|---|---|---|
| `batch_size` / `num_epochs` | `[req]` | per-GPU batch / epochs |
| `use_bfloat16` | `[req]` | fp16 autocast (misnamed) |
| `default_head` | `{}` | per-probe optimizer fallback (`lr/weight_decay/warmup/start_lr/final_lr/final_weight_decay`) |
| `save_optimizer` | `false` | store AdamW state in `latest.pt` |
| `cache_features` | `false` | one-shot feature cache (**§03**) |
| `cache_pooling` | `tokens` | `tokens`\|`pooled`\|`framewise` |
| `cache_max_gb` | `64` | abort if est. per-rank cache RAM exceeds |

### `model_kwargs`
`checkpoint` (local `.pth` for vjepa **or** HF repo id for VLM), `cache_dir` (HF
cache root), `module_name` (auto from `model`), `pretrain_kwargs` (`args_model`,
e.g. `encoder: {checkpoint_key, model_name, patch_size, tubelet_size, …}`),
`wrapper_kwargs` (`out_layers`/`out_stages` auto-injected; VLM: `resize_mode`,
`min_pixels`/`max_pixels`, `attn_implementation`, `dtype`, `pretrained`, …).

### Example A — V-JEPA classification, all layers, cached
```yaml
eval_name: analysis_vlm
folder: /…/logs
tag: vjepa2_shape_color
num_workers: 8
experiment:
  analysis:
    model: vjepa
    stages: { vision_encoder: all }     # 24 blocks (vit_large)
    plot: true
    probes:
      - { type: linear, pooling: mean, pre_norm: true,
          optimization: { lr: 0.01, weight_decay: 0.0, warmup: 1.0 } }
  data:
    dataset_type: VideoDataset
    dataset_train: /…/train.csv
    dataset_val:   /…/val.csv
    num_classes: 4
    resolution: 256
    resize_mode: resize
    frames_per_clip: 32
    frame_step: 1
  optimization:
    batch_size: 8
    num_epochs: 20
    use_bfloat16: true
    cache_features: true
    cache_pooling: pooled               # linear-only, tiny cache
    cache_max_gb: 130
    default_head: { start_lr: 0.0, final_lr: 0.0, final_weight_decay: 0.01 }
model_kwargs:
  checkpoint: /…/model.pth
  module_name: evals.analysis.modelcustom.vit_encoder_multilayer
  pretrain_kwargs:
    encoder: { checkpoint_key: target_encoder, model_name: vit_large,
               patch_size: 16, tubelet_size: 2, uniform_power: true, use_rope: true }
  wrapper_kwargs: { max_frames: 128, use_pos_embed: false }
```

### Example B — Qwen3-VL raw path, mixed stages, regression variables
```yaml
eval_name: analysis_vlm
folder: /…/logs
tag: qwen3vl-toy
experiment:
  analysis:
    model: qwen3vl
    stages:
      vision_encoder: [3, 7, 11, 15, 19, 23]   # ViT blocks (1024-d)
      after_merger: true                        # 2560-d LLM input
      deepstack: [5, 11, 17]                     # injection points (2560-d)
    task: regression
    regression:
      targets_npy: /…/combined_targets.npy       # CSV int label indexes rows
      variables:
        - { name: speed,     cols: [0] }
        - { name: direction, cols: [1, 2] }       # sin,cos (circular)
        - { name: accel_mag, cols: [3] }
    plot: true
    plot_pez: [0.2, 0.4]
    probes:
      - { type: linear, pooling: mean, pre_norm: true,
          optimization: { lr: 0.01, weight_decay: 0.0, warmup: 1.0 } }
  data:
    dataset_train: /…/train.csv
    dataset_val:   /…/val.csv
    num_classes: 4                    # informational under regression
    frames_per_clip: 8                # even; grid_t = T/2
  optimization:
    batch_size: 8
    num_epochs: 20
    use_bfloat16: true
    cache_features: true
    cache_pooling: tokens
    cache_max_gb: 64
    default_head: { start_lr: 0.0, final_lr: 0.0, final_weight_decay: 0.01 }
model_kwargs:
  checkpoint: Qwen/Qwen3-VL-4B-Instruct        # HF repo id
  cache_dir: /data/dataset/LLaVA-Video-100K-Subset/
  module_name: evals.analysis_vlm.modelcustom.qwen3vl_encoder
  pretrain_kwargs: { encoder: {} }
  wrapper_kwargs:
    resize_mode: smart
    min_pixels: 8192
    max_pixels: 112896
    attn_implementation: sdpa
    dtype: float16
```

---

## Gotchas / invariants / default-off guarantees

- **Default-off additive features.** `cache_features`, `save_optimizer`,
  `skip_base_probe`, `plot`, and `analysis.modes` all default off/empty; each was
  added so that with it unset the harness behaves as before. `skip_base_probe:
  false` ⇒ `num_probe_epochs == num_epochs` ⇒ loop byte-identical. Empty `modes`
  ⇒ nothing imported.
- **`data_mode` drives everything.** It selects the dataloader **and** the encoder
  call signature (`_encode`). `clip` expects `(clips, labels, clip_indices)`;
  `raw` expects `(frames_list, labels)`.
- **Stage x-axis differs by mode.** `clip` plots by block index; `raw` plots by
  stage **position** (so `block_5` and `deepstack_5` don't collide). `summary.json`
  keeps exact per-head values regardless.
- **Regression label semantics.** The CSV integer label is a **row index** into
  `targets_npy`, not a class. Targets are standardized per-column (NaN-aware); NaN
  rows are masked per head, so one dataset can mix variables defined on disjoint
  video subsets.
- **`cache_pooling="pooled"` ⇒ linear probes only.** Attentive / `framewise`
  probes raise (pooled cache already collapsed the tokens). Use `tokens` or
  `cache_features=false` for those.
- **Temporal probes are VLM-only.** `temporal_pos ∈ {learnable,rope}` and
  `pooling=framewise_*` require `encoder.num_temporal`; V-JEPA doesn't expose it
  (time is already in RoPE) → raises.
- **Clip loader `drop_last` is force-cleared** to avoid silently dropping partial
  batches (0 batches on tiny val splits). Raw eval uses an unpadded shard sampler
  for exact all-reduced accuracy.
- **Cache pre-pass uses `workers=0`, `persistent=False`.** A second DataLoader's
  worker respawn deadlocks at the train→val transition under spawn
  multiprocessing.
- **Validation runs heads under `no_grad`** to avoid tripping the DDP static-graph
  reducer on the next forward.
- **Features stay fp16.** `_encode` deliberately does not upcast all stages to
  fp32 (would double peak GPU memory / OOM on all-layer scans); the cache stores
  `.half()` and the non-cached probe runs under autocast.
- **Env isolation.** Only the selected backend module is imported (lazily). Do not
  mix a `vjepa` config with a VLM env or vice-versa — each backend's deps live in
  its own conda env.
- **`num_classes` is required for classification** but only informational under
  regression (head out-dim = `len(var_cols)`); it is still recorded in
  `summary.json` and passed to the plot.

## Cross-references
- **§03** feature caching (`cache.py`, `cache_pooling`, RAM guard)
- **§04** probes & regression heads (`probes.py`, R² math, variables)
- **§07** plotting (`plot_layer_val_acc`, `plot_pez`)
- **§12** post-hoc analysis modes + VLM backend stage definitions
