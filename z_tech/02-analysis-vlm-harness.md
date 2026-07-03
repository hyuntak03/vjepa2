# 02 — analysis_vlm harness (eval flow)

> `evals/analysis_vlm/eval.py::main()` is a unified frozen-encoder probing harness: it selects one of three encoder backends (each in its own conda env), extracts features at a set of **stages**, builds one probe head per **(stage × probe-spec × variable)**, trains them **jointly** on the frozen features, and reports a `[stage × probe]` accuracy / R² matrix — then optionally dispatches default-off **post-hoc analysis modes** on rank 0.

## Purpose

`evals/analysis_vlm/` is a **model-family-agnostic** layer/stage probing harness. For a
chosen encoder it:

1. Loads a **frozen** encoder that exposes intermediate features at a configurable set
   of **stages** (ViT blocks for V-JEPA; ViT-blocks / after-merger / deepstack for VLMs).
2. Builds **one probe head per `(stage × probe-spec × regression-variable)`** and trains
   them **all jointly** in a single fused optimizer over the frozen features.
3. Reports a `[stage × probe]` matrix — **accuracy %** (classification, CrossEntropy) or
   **R²** (regression, masked MSE) — as `summary.json`, per-rank CSV, a checkpoint, and an
   optional per-stage plot.
4. Optionally runs **post-hoc analysis modes** (attention distance, ablation, steering, …)
   on the same frozen encoder — a strictly additive, default-off block.

Three encoder backends are selectable from config; each lives in its **own conda env** and
is imported **lazily** (via `importlib` on `module_name`) so heavy/conflicting deps never
co-load:

| `analysis.model` | backend module | `data_mode` | conda env |
|---|---|---|---|
| `vjepa` | `evals.analysis.modelcustom.vit_encoder_multilayer` | `clip` | `vjepa2` |
| `llavavideo` | `evals.analysis_vlm.modelcustom.llava_video_encoder` | `raw` | `lmms_eval_llavavideo` |
| `qwen3vl` | `evals.analysis_vlm.modelcustom.qwen3vl_encoder` | `raw` | `lmms_eval_py311_2.7` |

Routed by `eval_name: analysis_vlm` — `evals/scaffold.py:19` dynamically imports
`evals.analysis_vlm.eval` and calls `main(args_eval, resume_preempt)`, so upstream
`evals/main.py` / `evals/scaffold.py` are **untouched**.

## What changed vs upstream V-JEPA2

The **entire `evals/analysis_vlm/` package is new** — not present at upstream `204698b`
(`git ls-tree -r 204698b -- evals/analysis_vlm` returns nothing; added in commit
`91fa127 "analysis module added"`). No upstream file is modified for routing:
`git diff 204698b -- evals/scaffold.py evals/main.py` is **empty** — the stock `eval_name`
dispatch (`scaffold.py:16-19`, `import_path = f"evals.{eval_name}.eval"`) picks the package
up unchanged.

**NEW files in this doc's scope:**

| File | Role | Detailed in |
|---|---|---|
| `evals/analysis_vlm/eval.py` | `main()` control flow — config → encoder → heads → train/eval → outputs → modes | this doc |
| `evals/analysis_vlm/data.py` | raw-frame dataloader for VLM backends (`RawVideoDataset`, `make_raw_dataloader`) | this doc |
| `evals/analysis_vlm/loadutil.py` | HF-weight location resolver shared by VLM backends | this doc |
| `evals/analysis_vlm/cache.py` | one-shot feature cache | §03 |
| `evals/analysis_vlm/probes.py` | temporal-aware probe heads | §04 |
| `evals/analysis_vlm/modelcustom/{llava_video_encoder,qwen3vl_encoder}.py` | VLM backends + stage definitions | §08 |
| `evals/analysis_vlm/modes/` | post-hoc analysis modes (dispatch + `attention_distance`, …) | §12 |

**MODIFIED upstream files:** none. Routing rides the existing scaffold; there is no upstream
delta to quote.

**Reused from upstream (imported, not copied):**

| Symbol | Source | Note |
|---|---|---|
| `init_module` | `evals/video_classification_frozen/models.py:14` | builds encoder, then `.eval()` + `requires_grad=False` on all params (models.py:42-44) → **frozen** |
| `build_probe`, `probe_name` | `evals/analysis/probes.py:56,84` | upstream-style linear/attentive head + stable head name |
| `make_dataloader`, `init_data`, `DEFAULT_NORMALIZATION` | `evals/video_classification_frozen/eval.py`, `src/datasets/data_manager` | clip data path |
| `WarmupCosineLRSchedule`, `CosineWDSchedule` | `evals/video_classification_frozen/eval.py` | per-param-group LR/WD schedules |
| `AllReduceSum`, `init_distributed` | `src/utils/distributed` | exact metric reduction / DDP init |
| `CSVLogger` | `src/utils/logging` | per-rank CSV |
| `robust_checkpoint_loader` | `src/utils/checkpoint_loader` | resume |

**Additive, default-off features (within the new file):** `cache_features`,
`save_optimizer`, `skip_base_probe`, `plot`, and `analysis.modes` all default off/empty.
Each was added so that with it unset the harness runs exactly as its first version:
`skip_base_probe: false` ⇒ `num_probe_epochs == num_epochs` ⇒ loop byte-identical
(eval.py:504); an absent `modes` key ⇒ `modes_cfg == {}` ⇒ the dispatch block is skipped and
**nothing under `modes/` is imported** (eval.py:568-570).

## Design & data flow

`main()` executes the steps below **in this order** (source anchors in each header). Internal
step numbers are labelled **Step N**; cross-references to sibling tech-report sections use
**§NN** — the two numbering spaces are distinct.

### Step 1 — Config parsing — `eval.py:85`

Top-level (`args_eval`): `val_only`, `folder`, `resume_checkpoint`
(`or resume_preempt`), `tag`, `num_workers` (default **12**). `model_kwargs` yields
`checkpoint` (local `.pth` for vjepa **or** HF repo id for VLM), `cache_dir`, `module_name`,
`pretrain_kwargs` → `args_model`, `wrapper_kwargs` → `args_wrapper` (copied into a fresh
dict). `enc_model_name = args_model.encoder.model_name`. Three sub-sections drive the run:
`experiment.analysis`, `experiment.data`, `experiment.optimization`.

### Step 2 — Model registry → backend + data_mode — `eval.py:62,108`

```python
_BACKENDS = {
    "vjepa":      ("evals.analysis.modelcustom.vit_encoder_multilayer", "clip"),
    "llavavideo": ("evals.analysis_vlm.modelcustom.llava_video_encoder", "raw"),
    "qwen3vl":    ("evals.analysis_vlm.modelcustom.qwen3vl_encoder",     "raw"),
}
```

- `analysis.model` (if set) picks `module_name` + `data_mode` from `_BACKENDS`; each is
  overridable via `model_kwargs.module_name` / `analysis.data_mode`. An unknown model raises.
- If `data_mode` is still unset: `"raw"` when `module_name` contains `"analysis_vlm"`, else
  `"clip"` (eval.py:116-117).
- `module_name` must resolve or it asserts (eval.py:118).

`data_mode` selects **both** the dataloader **and** how the encoder forward is called
(see `_encode`, Step 10). `clip` = V-JEPA tensor path; `raw` = VLM native-preprocess path.

### Step 3 — Stages resolution — `eval.py:124`

`analysis.stages` (legacy alias `analysis.layers`). Accepted forms:

- **structured dict** `{vision_encoder: [ints]|"all", <toggle>: true|[ints]}` — only
  `vision_encoder` carries the per-layer list; other keys are backend stage toggles
  (`after_merger`, `deepstack`, …).
- **shorthand** `"all"` or `[int, …]`.

Path split by `data_mode`:

- **`clip`/vjepa** (eval.py:125-139): a dict **must** contain `vision_encoder` or it raises
  (the ViT backbone has no other stages); any extra truthy keys **warn + are ignored**.
  The `vision_encoder` value is resolved via `_resolve_layers` (eval.py:71): `"all"` expands
  to `range(_VIT_DEPTH[model_name])`; a plain list is `int()`-cast. Result is injected as
  `wrapper_kwargs.out_layers` and `stages` becomes the concrete int list.

  ```python
  _VIT_DEPTH = {"vit_large": 24, "vit_huge": 32, "vit_giant": 40,
                "vit_giant_xformers": 40, "vit_gigantic": 48}
  ```
  `"all"` on a model name absent from `_VIT_DEPTH` raises (give an explicit list).
- **`raw`/VLM** (eval.py:140-145): the whole spec is handed to the backend as
  `wrapper_kwargs.out_stages`; `cache_dir` is copied into `wrapper_kwargs` if present (lets
  the backend resolve a repo-id checkpoint via `loadutil`). `stages = None` here, then read
  back from `encoder.stages` **after** `init_module` (Step 6, eval.py:276-277).

### Step 4 — probes / plot / task — `eval.py:147`

- `analysis.probes` — **non-empty list** (asserted); each spec builds a head per stage.
- `analysis.plot` (default `false`); `analysis.plot_pez` = `[lo,hi]` layer-fraction band to
  shade, validated `0 ≤ lo < hi ≤ 1` (eval.py:150-153) → §07.
- `analysis.task` (default `classification`; must be `classification` or `regression`,
  else raises):
  - **classification** → CrossEntropy → **accuracy %**.
  - **regression** → masked MSE → **R²** (eval.py:189-212):
    - `regression.targets_npy` (alias `targets`) — required; `(N,D)` `.npy`
      (`ndim==1` promoted to `(N,1)`). **The CSV integer label indexes rows of this array**
      (`label → target vector`), so all dataloaders stay unchanged.
    - Columns are **standardized per-column, NaN-aware**: `μ=nanmean`, `σ=nanstd`,
      `x=(x-μ)/clip(σ,1e-6,None)`. R² is invariant to this affine transform; NaNs stay NaN →
      masked out per head. `μ`/`σ` are kept (`col_mu`/`col_sd`) and later handed to modes so
      they can recover raw targets.
    - `regression.variables` — list of `{name, cols}`; default is one variable spanning all
      columns. Each becomes its own head / R² curve. Column bounds are asserted `0 ≤ c < D`.
      `reg_vars` = `[(name, cols), …]`; classification uses a single dummy `(None, None)`.

### Step 5 — Optimization config — `eval.py:216`

`experiment.optimization`: `batch_size`, `num_epochs`, `use_bfloat16`, `default_head` (the
per-probe optimizer fallback), plus:

- `save_optimizer` (default **false**) — AdamW state in `latest.pt` is huge for high-dim
  attentive probes; off = probe weights only.
- `cache_features` (default **false**), `cache_pooling` (`tokens`|`pooled`|`framewise`,
  default `tokens`), `cache_max_gb` (default **64**) → §03.

`_opt_kwargs(spec)` (eval.py:231) merges `default_head` under each probe's own
`optimization` block →
`ref_wd`(`weight_decay`,0.01) / `final_wd`(`final_weight_decay`,0.01) /
`start_lr`(0.0) / `ref_lr`(`lr`,0.001) / `final_lr`(0.0) / `warmup`(1.0).

**Distributed / device setup** (eval.py:243-264): `mp.set_start_method("spawn")`,
`device = cuda:0`, `world_size, rank = init_distributed()`. Regression targets are moved
on-device as `targets_t` `(N,D)`. Output dir = `os.path.join(folder, "analysis_vlm/")`, then
`/<tag>` if `tag` is set; `log_file = log_r{rank}.csv`, `latest_path = latest.pt`.

### Step 6 — Encoder init (frozen) — `eval.py:267`

`init_module(module_name, frames_per_clip, resolution, checkpoint, model_kwargs=args_model,
wrapper_kwargs=args_wrapper, device)` builds the encoder, calls `.eval()` and sets
`requires_grad=False` on every parameter (`models.py:42-44`) — the encoder is **frozen**.
For `raw`, `stages = list(encoder.stages)` is read back. `embed_dims =
getattr(encoder,"embed_dims",None) or [encoder.embed_dim]*len(stages)` (VLM backends expose
per-stage dims; V-JEPA falls back to a uniform dim); asserted equal-length to `stages`.
Backends may also expose `num_temporal` (drives temporal-aware probes).

### Step 7 — Head construction: one per (stage × probe × variable) — `eval.py:287`

For each stage a plot **x-value** and **tag** are chosen:

- `clip`: `stage_tag = f"L{block:02d}"`, `layer_val = block index` (unique).
- `raw`: `stage_tag = str(stage_name)`, `layer_val = stage_position` (position avoids
  `block_5` vs `deepstack_5` colliding on x=5; the name labels the tick).

Per probe spec, defaults are read that **determine which `_build` branch fires**:
`type` defaults to **`attentive`** (eval.py:296), `temporal_pos` to `none` (eval.py:297),
`pooling` to `mean` (eval.py:298). Derived flags:
`framewise = (type=="linear" and pooling.startswith("framewise"))`;
`use_tpos = (type=="attentive" and temporal_pos in {learnable,rope})`.
`pname = probe_name(spec) + (f"-{tpos}" if use_tpos else "")`.

`_build(out_dim)` (eval.py:315) dispatches the head type:

| condition | head |
|---|---|
| `cache_features` & `cache_pooling=="pooled"` | `cache.PooledLinearProbe` (linear only) |
| `type=linear` & `pooling` starts `framewise` | `probes.TemporalLinearProbe` (needs `encoder.num_temporal`) |
| `type=attentive` & `temporal_pos∈{learnable,rope}` | `probes.TemporalAttentiveClassifier` (needs `num_temporal`) |
| else | upstream `build_probe(spec, …)` (linear / attentive) |

`out_dim = len(var_cols)` for regression, else `num_classes`. Each head is `.to(device)` and
DDP-wrapped (`static_graph=True`) when a process group exists. Head record:

```python
heads.append(dict(name=name, layer=layer_val, layer_pos=stage_pos,
                  probe=pname, series=series,
                  stage=stage_tag, module=module, tcols=var_cols))
```

- `name = f"{stage_tag}_{pname}{__var}"`, de-collided with `#2`, `#3`, … on duplicate specs.
- `layer_pos` indexes into the per-stage feature list at forward time.
- `series` = plot-line grouping: `pname` (classification), `var_name` (regression, single
  probe), or `var_name·pname` (regression, multiple probes).

**Guards (raise loudly):** `cache_pooling="pooled"` caches only pooled vectors → attentive or
`framewise` probes raise; `temporal_pos`/`framewise` require a backend exposing
`num_temporal` (VLM only; V-JEPA already encodes time via RoPE).

### Step 8 — Data loaders — `eval.py:383`

`_split_loader(root, training, persistent=True, workers=None)` builds one split's loader:

- `clip` + `resize_mode="resize"` → upstream `init_data` with a `_DirectResizeClipTransform`
  (direct resize to `resolution²`, aspect squashed, SigLIP-like; deterministic; reuses the
  stock `Compose/Resize/ClipToTensor/Normalize` primitives).
- `clip` + `resize_mode="crop"` (default) → upstream `make_dataloader` (shorter-side resize +
  center-crop).
- `raw` → `data.make_raw_dataloader`.

> **Gotcha:** the clip path force-sets `ld.batch_sampler.drop_last = False` (eval.py:418-419)
> — `init_data`/`make_videodataset` default `drop_last=True`, which silently drops partial
> batches and can yield **0 batches** on small val splits. DDP `DistributedSampler` pads to
> equal per-rank counts, so batch counts stay aligned.

If `cache_features` (eval.py:426-455): one **deterministic pre-pass per split**
(`training=False`, `workers=0`, `persistent=False`) encodes features into a per-rank cache
via `build_feature_cache`, then `make_cached_loader` serves them; `run_mode="cached"` and
`train_sampler=None`. The two pre-pass loaders are built **sequentially** (train dropped
before val) to avoid spawn-multiprocessing worker pile-up / deadlock. Otherwise (eval.py:457)
the normal train/val loaders are built. `ipe = len(train_loader)`. Details in §03.

### Step 9 — Optimizer: one fused AdamW — `eval.py:465,617`

`_init_opt_fused` builds **one** `AdamW` with **one param-group per head** (each carrying its
own `mc_*` LR/WD schedule keys), a single `WarmupCosineLRSchedule`, `CosineWDSchedule`, and
`GradScaler`. Numerically identical to one-optimizer-per-head (the schedules already iterate
`optimizer.param_groups`), but collapses N `step()`/`zero_grad()`/`scaler.step()` launches
into one (~25% off the cached step on many-head scans). Returned as **length-1 lists** so the
loop/checkpoint code (which iterates these lists) is unchanged. `T = num_epochs · ipe` (note:
uses the **full** `num_epochs`, not `num_probe_epochs`). `scaler = None` when `use_bfloat16`
is false.

The rank-0 `CSVLogger` is opened here (eval.py:470-474): columns are `epoch` then
`<head>_train`/`<head>_val` per head.

### Step 10 — Checkpoint resume + train / eval loop — `eval.py:477`

**Resume** (eval.py:477-483, executes **before** the epoch loop): if `resume_checkpoint` and
`latest.pt` exists, `load_checkpoint` restores head weights (and optimizer/scaler if the
checkpoint has them — weights-only checkpoints restart the optimizer), then the LR/WD schedule
is fast-forwarded `start_epoch · ipe` steps so the resumed run continues on the exact schedule.
`val_only` resume returns `start_epoch=0`.

`save_checkpoint(epoch)` (eval.py:485) writes on rank 0: `classifiers` (unwrapped state
dicts), `epoch`, `head_names`, `stages`, `batch_size`, `world_size`; `opt`/`scaler` only if
`save_optimizer`.

**Loop** (eval.py:498-554):

- `num_probe_epochs = 0 if analysis.skip_base_probe else num_epochs` (default-off; off ⇒ loop
  byte-identical to before) — used by encoder-only analysis modes that don't need trained
  probes.
- Each epoch: `train_sampler.set_epoch(epoch)`, `run_one_epoch(training=True)` (skipped and
  faked as `-1.0` under `val_only`), then `run_one_epoch(training=False)`; `best_val[name]`
  tracked with floor `-inf` (R² can be negative).
- Rank-0 side effects each epoch: append CSV row, **rewrite `summary.json`**, `save_checkpoint`.
  `val_only` breaks after the first pass.

`run_one_epoch` (eval.py:674):

- Calls `_encode` → per-stage feature list; each head reads `feats[h["layer_pos"]]`.
- **classification:** `CrossEntropyLoss`; returns per-head accuracy % via
  `AllReduceSum(correct)/total`.
- **regression:** masked-mean MSE per head (NaN target rows contribute 0 but stay in the graph
  so the DDP static-graph structure is rank-identical); returns per-head
  R² = `1 − SS_res/SS_tot` from all-reduced `ss_res / sum_y / sum_y2 / cnt` **per-head**
  accumulators (so one combined dataset can hold variables defined on different video subsets).
- **Validation runs heads under `torch.no_grad()`** — a grad-enabled DDP forward with no
  backward trips the static-graph reducer on the *next* forward under multi-GPU.

`_encode` (eval.py:649) has three branches by `data_mode`:

| mode | input | encoder call | dtype on return |
|---|---|---|---|
| `cached` | `(list[stage] tensors, labels)` | none — features already cached | **upcast to fp32 per batch** (`.to(device).float()`, eval.py:653) |
| `clip` | `(clips, labels, clip_indices)` | `encoder(clips, clip_indices)` under fp16 autocast, `no_grad` | encoder dtype (fp16) |
| `raw` | `(frames_list, labels)` | `encoder(frames_list)` in the backend's own (half) dtype, `no_grad` | encoder dtype (fp16) |

Returns `(feats: list[(B,N,D) detached], labels, bsz)`. The **non-cached** branches keep the
encoder dtype (fp16) — upcasting all stages to fp32 here would double peak GPU memory (OOM on
all-layer scans); the probe forward runs under autocast anyway. The **cached** branch stores
`.half()` in the cache (§03) but re-floats each batch to fp32 on load, so cached-path probes
see fp32 inputs.

**Plot** (eval.py:556-563): if `analysis.plot` and rank 0, `plot_layer_val_acc(heads,
best_val, "stage_val_acc.png", subtitle, num_classes, metric, pez=plot_pez)`. Each regression
variable is its own curve (legend). → §07.

### Step 11 — Outputs — `<folder>/analysis_vlm/[<tag>/]`

| File | Written by | Contents |
|---|---|---|
| `log_r{rank}.csv` | `CSVLogger` (eval.py:474) | `epoch`, then `<head>_train` / `<head>_val` per head |
| `summary.json` | rank 0 each epoch (eval.py:542) | `epoch`, **`num_epochs`**, `model`, `data_mode`, **`num_classes`**, `task`, `metric` (`r2`/`accuracy`), `variables` (`[{name,cols}]` or `null`), `stages` (stringified), `head_names`, `val_acc`, `train_acc`, `best_val_acc` |
| `latest.pt` | `save_checkpoint` (eval.py:485) | `classifiers` (state dicts), `epoch`, `head_names`, `stages`, `batch_size`, `world_size`; `opt`/`scaler` only if `save_optimizer` |
| `stage_val_acc.png` | if `analysis.plot`, rank 0 (eval.py:556) | `plot_layer_val_acc(…)` — §07 |
| `<mode>/…` | post-hoc modes, rank 0 (Step 12) | one subdir per mode — §12 |

### Step 12 — Post-hoc analysis modes (additive) — `eval.py:565`

```python
modes_cfg = args_analysis.get("modes") or {}
if modes_cfg and rank == 0:
    from evals.analysis_vlm.modes import AnalysisContext, run_modes
    ctx = AnalysisContext(encoder=…, heads=…, best_val=…, stages=…, embed_dims=…,
                          reg_vars=…, targets_t=…, targets_npy=…, col_mu=…, col_sd=…,
                          tr_feats=…, va_feats=…, cache_pooling=…, data_mode=…,
                          encode_clip=lambda d: _encode(encoder, d, device, data_mode, use_bfloat16),
                          make_val_clip_loader=lambda: _split_loader(val_data_path[0], training=False, workers=0)[0])
    run_modes(modes_cfg, ctx)
if torch.distributed.is_initialized():
    torch.distributed.barrier()
```

- **Absent `modes` key ⇒ `modes_cfg == {}` ⇒ the whole block is skipped and nothing under
  `modes/` is imported ⇒ existing runs behave byte-for-byte identically.**
- The block runs on **rank 0 only**. The final `torch.distributed.barrier()` (eval.py:589-590)
  is load-bearing: under multi-GPU it holds every non-rank-0 rank at the barrier while rank 0
  runs `run_modes`, so no rank exits `main()` (and tears down the process group) mid-mode.
- **Practical consequence:** because modes execute on a single rank, **multi-GPU gives NO
  speedup for post-hoc modes** — only the base probing sweep and the feature-cache pre-pass
  parallelize across ranks. This is why the mode launcher `z_scripts/run_attn_distance_vjepa.sh`
  requests **`--gres=gpu:1`** on purpose (the single-GPU setup is the faithful and simplest one
  for modes).
- `AnalysisContext` (a read-only dataclass) carries the frozen encoder, the trained heads +
  metadata, cached feature tensors (or `None`), standardization stats (`col_mu`/`col_sd`) and
  the raw `targets_npy` path, plus two closures back into `main()`'s scope: `encode_clip`
  (encode one batch) and `make_val_clip_loader` (build a fresh deterministic val clip loader).

Mode internals (`attention_distance` heatmap/line plots, ablation, steering, …) are deferred
to §12.

---

### `evals/analysis_vlm/data.py` — raw-frame dataloader (VLM path)

VLM vision encoders need their **own native preprocessing** (SigLIP 384 resize /
Qwen smart-resize), so this loader does **no** normalize/crop — it just samples raw uint8
frames and hands them to the backend.

- `RawVideoDataset` (data.py:28): reads the same CSV as the V-JEPA path
  (`<abs_mp4_path> <int_label>`, space-delimited, no header; falls back to `::` on a parser
  error). `_load` uniformly samples `frames_per_clip` frames across the whole video via
  `np.linspace(0, n-1, T).round()` (repeats gracefully if `n < T`) → `(T,H,W,C)` uint8 via
  decord. `__getitem__` retries up to **8×** on corrupt/missing videos with a random fallback
  index, else raises.
- `_collate` (data.py:63): keeps frames as a **list** of `(T,H,W,C)` tensors (videos may
  differ in H/W — the backend resizes each natively); labels stacked to a `long` tensor.
- `_UnpaddedShardSampler` (data.py:69): strided `rank::world_size` shard with **no padding**,
  so the rank-union is exactly the dataset once each — required so `AllReduceSum`'d accuracy is
  exact (a `DistributedSampler` pads with duplicates and skews it). Used for **eval / cache
  pre-pass**.
- `make_raw_dataloader` (data.py:84): sampler selection is **guarded by `world_size > 1`**
  (data.py:102):
  - `world_size > 1` & `training` → `DistributedSampler(shuffle=True, drop_last=False)`.
  - `world_size > 1` & eval → `_UnpaddedShardSampler`.
  - **`world_size == 1` → `sampler = None`**; then `shuffle=(training and sampler is None)` ⇒
    plain in-process shuffle for training, sequential for eval. This is the **common
    single-GPU path** (the norm for post-hoc modes and small toy sets).
  - `persistent_workers` defaults on for training but is forced **off** for the one-shot cache
    pre-pass (`persistent=False`); `prefetch_factor=4` when workers > 0.

> **Invariant:** raw sampling is **always uniform** across the full clip (`frame_step` is a
> clip-path-only concept). The clip path's `uniform_sampling` flag is what makes V-JEPA match
> this behavior.

### `evals/analysis_vlm/loadutil.py` — weight resolver (VLM backends)

Lets a VLM config point at weights flexibly. Used **only** by the two VLM backends, not by
`eval.py` directly.

- `resolve_model_dir(checkpoint, wrapper_kwargs)` (loadutil.py:30): candidate =
  `wrapper_kwargs.pretrained or checkpoint`. A local dir → `find_snapshot`; otherwise treated
  as an **HF repo id** → `snapshot_download(..., cache_dir=cache_dir, local_files_only=True)`
  (offline-first; downloads only on a cache miss).
- `find_snapshot(path)` (loadutil.py:18): accepts a snapshot dir, an HF-cache `models--…` root
  (globs `snapshots/*`), or any dir with `config.json`.

This is why VLM configs set `checkpoint: <HF repo id>` + `cache_dir: <cache root>` (e.g.
`Qwen/Qwen3-VL-4B-Instruct` + `/data/dataset/…`), and why `eval.py` copies `cache_dir` into
`wrapper_kwargs` for the raw path (eval.py:143-144).

## Key code

**Data-mode fallback + module assertion** — `eval.py:110-118`:

```python
if model_sel:
    default_module, default_mode = _BACKENDS[model_sel]
    module_name = module_name or default_module
    data_mode = data_mode or default_mode
if data_mode is None:
    data_mode = "raw" if (module_name and "analysis_vlm" in module_name) else "clip"
assert module_name, "must set experiment.analysis.model or model_kwargs.module_name"
```

**NaN-aware per-column target standardization** — `eval.py:201-203`:

```python
mu = np.nanmean(targets_arr, axis=0, keepdims=True)
sd = np.nanstd(targets_arr, axis=0, keepdims=True)
targets_arr = (targets_arr - mu) / np.clip(sd, 1e-6, None)
```

**Force-clear `drop_last` on the clip loader** — `eval.py:418-419`:

```python
if getattr(ld, "batch_sampler", None) is not None:
    ld.batch_sampler.drop_last = False
```

**Per-head masked-mean regression loss (NaN rows kept in graph)** — `eval.py:732-736`:

```python
yh = yfull[:, head_cols[hi]]
m = (~torch.isnan(yh).any(dim=1)).float()                       # (B,)
err = ((preds[hi] - torch.nan_to_num(yh)) ** 2).sum(dim=1) * m  # (B,)
losses.append(err.sum() / m.sum().clamp(min=1.0))
```

**`summary.json` payload (exhaustive)** — `eval.py:542-550`:

```python
json.dump({"epoch": epoch + 1, "num_epochs": num_epochs, "model": model_sel,
           "data_mode": data_mode, "num_classes": num_classes,
           "task": task, "metric": ("r2" if task == "regression" else "accuracy"),
           "variables": ([{"name": n, "cols": c} for n, c in reg_vars]
                         if task == "regression" else None),
           "stages": [str(s) for s in stages],
           "head_names": head_names, "val_acc": val_acc, "train_acc": train_acc,
           "best_val_acc": best_val}, f, indent=2)
```

**Additive modes dispatch + rank-0 barrier** — `eval.py:568-590`:

```python
modes_cfg = args_analysis.get("modes") or {}
if modes_cfg and rank == 0:
    from evals.analysis_vlm.modes import AnalysisContext, run_modes
    ctx = AnalysisContext(...)
    run_modes(modes_cfg, ctx)
if torch.distributed.is_initialized():
    torch.distributed.barrier()
```

## Configuration

Full schema for a run. Keys read by `eval.py` (defaults in brackets; `[req]` =
required/asserted). Probe-spec internals → §04; VLM `wrapper_kwargs` → §08; the consolidated
mode configs → §12/§13.

### Top level
| key | default | notes |
|---|---|---|
| `eval_name` | `[req]` | must be `analysis_vlm` (scaffold routing) |
| `folder` | `[req]` | output root; run dir = `<folder>/analysis_vlm/[<tag>/]` |
| `tag` | `None` | sub-folder / run name |
| `num_workers` | `12` | DataLoader workers |
| `resume_checkpoint` | `false` | resume from `latest.pt` |
| `val_only` | `false` | eval only (skip probe training) |

### `experiment.analysis`
| key | default | allowed / notes |
|---|---|---|
| `model` | — | `vjepa` \| `llavavideo` \| `qwen3vl` (picks backend + data mode) |
| `data_mode` | auto | `clip` \| `raw` (override) |
| `stages` (alias `layers`) | `[req]` | dict `{vision_encoder: […]\|"all", <toggle>}` or `"all"`/`[int,…]` |
| `probes` | `[req]` | non-empty list of probe specs (§04) |
| `task` | `classification` | `classification` \| `regression` |
| `regression.targets_npy` | — | `(N,D)` `.npy`; CSV int label indexes rows |
| `regression.variables` | all cols | `[{name, cols:[…]}, …]` |
| `plot` | `false` | write `stage_val_acc.png` |
| `plot_pez` | `None` | `[lo,hi]` layer-fraction shade band, `0≤lo<hi≤1` |
| `skip_base_probe` | `false` | 0 probe epochs (encoder-only modes) |
| `modes` | `{}` | post-hoc modes (§12); absent ⇒ skipped, nothing imported |

### `experiment.data`
| key | default | notes |
|---|---|---|
| `dataset_type` | `VideoDataset` | clip path only |
| `dataset_train` / `dataset_val` | `[req]` | CSV `"<mp4> <int_label>"` |
| `num_classes` | `[req]` (classification) | regression head dim = `len(cols)` instead; still recorded in `summary.json` |
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
| `cache_features` | `false` | one-shot feature cache (§03) |
| `cache_pooling` | `tokens` | `tokens`\|`pooled`\|`framewise` |
| `cache_max_gb` | `64` | abort if est. per-rank cache RAM exceeds |

### `model_kwargs`
`checkpoint` (local `.pth` for vjepa **or** HF repo id for VLM), `cache_dir` (HF cache root),
`module_name` (auto from `model`), `pretrain_kwargs` → `args_model`
(e.g. `encoder: {checkpoint_key, model_name, patch_size, tubelet_size, …}`), `wrapper_kwargs`
→ `args_wrapper` (`out_layers`/`out_stages` auto-injected; VLM adds `resize_mode`,
`min_pixels`/`max_pixels`, `attn_implementation`, `dtype`, `pretrained`, … → §08).

### Example A — V-JEPA classification, all layers, cached (pooled)
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

### Example C — encoder-only mode run (V-JEPA + `attention_distance`)
```yaml
experiment:
  analysis:
    model: vjepa
    task: regression
    regression:
      targets_npy: /…/blender_targets.npy
      variables: [{ name: direction, cols: [1, 2] }]
    stages: { vision_encoder: all }
    probes: [{ type: linear, pooling: mean, pre_norm: true }]
    skip_base_probe: true                       # encoder-only: no probe trained
    modes:
      attention_distance: { enabled: true, query_chunk: 512, max_batches: 8 }
  optimization: { batch_size: 8, num_epochs: 0, use_bfloat16: true, cache_features: false }
```
Launched single-GPU via `z_scripts/run_attn_distance_vjepa.sh` (`--gres=gpu:1` on purpose;
`python -m evals.main --fname <cfg> --devices cuda:0`). Mode internals + outputs → §12.

## Invariants & gotchas

- **Default-off additive features.** `cache_features`, `save_optimizer`, `skip_base_probe`,
  `plot`, `analysis.modes` all default off/empty; unset ⇒ harness behaves as its first
  version. `skip_base_probe: false` ⇒ `num_probe_epochs == num_epochs` ⇒ loop byte-identical.
  Empty `modes` ⇒ nothing under `modes/` imported.
- **`data_mode` drives everything.** It selects the dataloader **and** the encoder call
  signature (`_encode`). `clip` expects `(clips, labels, clip_indices)`; `raw` expects
  `(frames_list, labels)`.
- **Stage x-axis differs by mode.** `clip` plots by block index; `raw` plots by stage
  **position** (so `block_5` and `deepstack_5` don't collide). `summary.json` keeps exact
  per-head values regardless.
- **Regression label semantics.** The CSV integer label is a **row index** into `targets_npy`,
  not a class. Targets are standardized per-column (NaN-aware); NaN rows are masked per head,
  so one dataset can mix variables defined on disjoint video subsets.
- **`cache_pooling="pooled"` ⇒ linear probes only.** Attentive / `framewise` probes raise
  (pooled cache already collapsed the tokens). Use `tokens` or `cache_features=false`.
- **Temporal probes are VLM-only.** `temporal_pos ∈ {learnable,rope}` and `pooling=framewise_*`
  require `encoder.num_temporal`; V-JEPA doesn't expose it (time is in RoPE) → raises.
- **Probe-spec defaults choose the branch.** `type` defaults to `attentive`, `pooling` to
  `mean`, `temporal_pos` to `none`; a bare `{}` spec builds a mean-pooled attentive probe via
  `build_probe`.
- **Clip loader `drop_last` is force-cleared** to avoid silently dropping partial batches
  (0 batches on tiny val splits). Raw eval uses an unpadded shard sampler for exact
  all-reduced accuracy; single-GPU (`world_size==1`) uses `sampler=None` + plain shuffle.
- **Cache pre-pass uses `workers=0`, `persistent=False`.** A second DataLoader's worker respawn
  deadlocks at the train→val transition under spawn multiprocessing.
- **Validation runs heads under `no_grad`** to avoid tripping the DDP static-graph reducer on
  the next forward.
- **Non-cached features stay fp16.** `_encode` deliberately does not upcast the `clip`/`raw`
  stages to fp32 (would double peak GPU memory / OOM on all-layer scans); the cache stores
  `.half()` but the **cached** branch re-floats each batch to fp32 on load (eval.py:653), and
  the non-cached probe runs under autocast.
- **Modes run on rank 0 only; the trailing `barrier()` gates teardown.** Multi-GPU gives **no
  speedup** for post-hoc modes — only the base probing sweep and the feature-cache pre-pass
  parallelize. Hence the single-GPU mode launcher.
- **Env isolation.** Only the selected backend module is imported (lazily). Do not mix a
  `vjepa` config with a VLM env or vice-versa — each backend's deps live in its own conda env.
- **`num_classes` is required for classification** but only informational under regression
  (head out-dim = `len(var_cols)`); it is still recorded in `summary.json` and passed to the
  plot.

## Cross-references
- **§03** feature caching (`cache.py`, `cache_pooling`, RAM guard, `PooledLinearProbe`)
- **§04** probes & regression heads (`probes.py`, `build_probe`, R² math, variables)
- **§05** V-JEPA layer-wise probing harness (`evals/analysis/`, sibling of this package)
- **§06** data pipeline changes (`_DirectResizeClipTransform`, `uniform_sampling`, clip loaders)
- **§07** plotting (`plot_layer_val_acc`, `plot_pez` / PEZ shading)
- **§08** VLM encoder backends + stage definitions (`llava_video_encoder`, `qwen3vl_encoder`)
- **§11** attention hooks (`attention_hooks.py`, `AttentionDistanceCollector`) used by modes
- **§12** post-hoc analysis modes (dispatch, `AnalysisContext`, `attention_distance` internals)
- **§13** configs reference (per-mode YAML schema)
- **§14** reproduction status & findings (attention-distance Fig. 3/19 results)
