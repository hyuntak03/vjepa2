# 06 — Data-pipeline changes

> Two additive, **default-off** video-loading knobs — `uniform_sampling` (whole-clip frame coverage) and `resize_mode: 'resize'` (direct H×W squash instead of shorter-side-crop) — plus the `frame_step` contiguous-window bug they exist to fix; every hop restores upstream behavior when the config does not opt in.

## Purpose

The fork adds two knobs to the shared video data pipeline so that short, fixed-length toy clips (the Blender / toy-physics datasets used for the Fig.2c layer-wise R² reproduction and the attention-distance analysis) are **sampled** and **resized** in a way that preserves whole-clip motion and the full spatial frame:

1. **`uniform_sampling`** — evenly sample `frames_per_clip` frames across the *entire* video, length-agnostic, ignoring `frame_step`, `num_clips`, and `random_clip_sampling`. Threaded from config through `init_data` → `make_videodataset` → `VideoDataset.loadvideo_decord`.
2. **`resize_mode: 'resize'`** — direct `H×W` resize (aspect ratio squashed, full frame kept) instead of the stock shorter-side-resize + center-crop. Implemented as a drop-in transform (`_DirectResizeClipTransform`) inside the `analysis_vlm` eval **only** — the shared `src/datasets` transforms are untouched.

Both knobs default to the **upstream** behavior, so existing runs are byte-identical unless a config opts in. These same two knobs are what the newer post-hoc **analysis-modes** subsystem (`attention_distance` and its siblings) consumes — see [Cross-references](#cross-references).

## What changed vs upstream V-JEPA2

Baseline = commit **`204698b`**. The full fork surface touched by this section is **four source files + the config tree**. To reproduce every byte of the delta:

```bash
git diff 204698b -- \
  src/datasets/video_dataset.py \
  src/datasets/data_manager.py \
  evals/video_classification_frozen/eval.py \
  evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py
```

| File | Kind | Delta | Default-off guarantee |
|------|------|-------|-----------------------|
| `src/datasets/video_dataset.py` | **modified** | **(a) threading:** new `uniform_sampling=False` param on `make_videodataset` (`:54`, forwarded to `VideoDataset` at `:71`) and on `VideoDataset.__init__` (`:142`, stored as `self.uniform_sampling` at `:147`). **(b) behavior:** early-return branch in `loadvideo_decord` (`:334–341`) that evenly samples `fpc` frames over the whole video. | Param defaults `False`; branch guarded by `getattr(self, "uniform_sampling", False)`. |
| `src/datasets/data_manager.py` | **modified** | Threading only: new `uniform_sampling=False` param on `init_data` (`:40`), forwarded to `make_videodataset` **inside the `videodataset` branch only** (`:87`). The ImageNet path is untouched. | Param defaults `False`. |
| `evals/video_classification_frozen/eval.py` | **modified** | Threading only: new `uniform_sampling=False` param on `make_dataloader` (`:431`), forwarded to `init_data` (`:465`). **Note:** the classification `main()` does *not* read this from any config — only the plumbing exists here, so classification evals are unaffected regardless of the config file. | Param defaults `False`; never wired from YAML. |
| `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py` | **modified** | **Annotation-only / NO-OP.** Three `#!` comment lines added inside `ClipAggregation.forward` (`#! vit_encoder` before the method; two `#!` breadcrumbs before `outputs = self.model(x)`). **Zero behavioral change** — listed here only so a reader diffing `204698b` does not mistake it for a functional fork edit. | Comments compile away; run is byte-identical. |
| `evals/analysis_vlm/eval.py` | **new** (814 lines; absent at `204698b`) | The only entrypoint that actually wires **both** knobs from YAML: reads `data.uniform_sampling` (`:167`) and validates `data.resize_mode` (`:173–175`); defines `_DirectResizeClipTransform` (`:593`) for `resize_mode:'resize'`. | `resize_mode` defaults `'crop'`; `uniform_sampling` defaults `False`. |
| `configs/analysis/**` | **new** | Toy/Blender configs set `uniform_sampling: true`, `resize_mode: resize`, `frame_step: 1`. | Opt-in per config. |

**Classification of the four source deltas:** `data_manager.py`, `video_classification_frozen/eval.py`, and the *param-threading half* of `video_dataset.py` are pure parameter-threading; the *single behavioral change* is the `loadvideo_decord` branch in `video_dataset.py`; `vit_encoder_multiclip.py` is comments-only. Nothing else in the shared pipeline moved.

### The behavioral branch (the only functional change)

```python
# src/datasets/video_dataset.py:334-341  (inside VideoDataset.loadvideo_decord, after vr.seek(0))
# uniform_sampling: pick `fpc` frames evenly across the WHOLE video (length-agnostic;
# ignores frame_step / num_clips). Avoids the contiguous-window default that, when
# fpc*frame_step < len(video), only covers a sub-segment -> sub-patch motion per tubelet.
if getattr(self, "uniform_sampling", False):
    n = len(vr)
    indices = np.clip(np.linspace(0, n - 1, num=fpc).round(), 0, n - 1).astype(np.int64)
    buffer = vr.get_batch(list(indices)).asnumpy()
    return buffer, [indices]
```

Everything **above** this branch still runs unchanged: the `fps`/`duration`/`frame_step` validation (`:158`), `clip_len = int(fpc * fstp)` (`:326`), the `filter_short_videos` guard (`:328`), and `vr.seek(0)` (`:332`). Only the segment-partition sampler **below** it (`:345–398`) is bypassed.

### The annotation-only delta (verbatim)

```diff
# evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py
@@ class ClipAggregation(nn.Module):
-
+    #! vit_encoder
     def forward(self, x, clip_indices=None):
@@
         x = torch.cat(x, dim=0)
-
+        #! x.shape = [2,3,32,256,256] -> 3D conv 전
+        #! self.model(x) 해서 이제 모델 들어감
         outputs = self.model(x)
```

These are debugging breadcrumbs marking where the concatenated 5-D clip tensor `[B, C, T, H, W]` enters the 3-D patch-embed conv of `self.model`. The `[2,3,32,256,256]` shape note is illustrative annotation, **not** an authoritative contract. No code path, RNG draw, or output is affected.

## Design & data flow

```
config (experiment.data.*)
        │  uniform_sampling, resize_mode, frame_step, resolution, num_segments, frames_per_clip
        ▼
evals/analysis_vlm/eval.py  main()          ← the ONLY entrypoint that reads BOTH knobs
        │
        ├── resize_mode == 'crop'  ──► make_dataloader(...) ──► stock VideoTransform
        │                                (shorter-side resize + CenterCrop)
        │
        └── resize_mode == 'resize' ─► init_data(transform=_DirectResizeClipTransform, …)
                                         (direct resolution² resize; full frame kept)
                          │
                          ▼  (both paths forward uniform_sampling=…)
        src/datasets/data_manager.py  init_data(uniform_sampling)  [videodataset branch only]
                          ▼
        src/datasets/video_dataset.py  make_videodataset ► VideoDataset(uniform_sampling)
                          ▼
        VideoDataset.loadvideo_decord(sample, fpc)
                  ├── uniform_sampling=True  ► linspace(0,len-1,fpc) → return  (branch :337)
                  └── uniform_sampling=False ► stock num_clips segment-partition sampler (:345-398)
```

### `uniform_sampling` — semantics

When `uniform_sampling=True`, `loadvideo_decord` returns exactly `fpc` frame indices `round(linspace(0, len-1, fpc))` — the first and last frame of the video are always endpoints, and the middle `fpc-2` are evenly spaced. This is **length-agnostic and fully deterministic**: `frame_step`, `num_clips`, and `random_clip_sampling` are all ignored on this path. For a native 16-frame clip with `fpc=16`, it selects **all 16 frames in order**; `frame_step` becomes moot.

### `resize_mode` — semantics

| `resize_mode` | Code path | Spatial op | Effect |
|---------------|-----------|-----------|--------|
| `'crop'` (default, upstream) | `video_classification_frozen.make_dataloader` → stock `VideoTransform` | shorter-side resize to `int(crop_size*256/224)` then `CenterCrop(crop_size)` (`utils.py:68–72`) | model's trained pipeline; **edges cropped away** |
| `'resize'` | `_DirectResizeClipTransform` + direct `init_data` call, mirroring `make_dataloader`'s args exactly (`eval.py:389–403`) | `vt.Resize((crop, crop))` then `ClipToTensor` + `Normalize` (`eval.py:607–611`) | aspect squashed, **full frame preserved** (no edge loss); matches the VLM SigLIP square-resize path; deterministic |

`vt.Resize` with a **tuple** size resizes to that exact `(width, height)` (`src/datasets/utils/video/transforms.py:856–866`, docstring: `size (tuple): (width, height)`), so with `(crop, crop)` it squashes to a square. Only the transform is swapped — the original `VideoTransform` / `make_transforms` are left byte-identical; `resize` is a *separate* transform class.

### `frame_step` — the contiguous-window bug `uniform_sampling` fixes

`frame_step` (config `frame_step` → `init_data(frame_sample_rate=…)` → `VideoDataset.frame_step`) sets the temporal stride of the **stock** sampler. Exactly one of `frame_step`, `fps`, `duration` may be non-None (`video_dataset.py:158`). With `frame_step`, the per-clip window length is `clip_len = fpc * frame_step` (`:326`), and each of `num_clips` segments samples `np.linspace(start, end, fpc)` inside a `partition_len = len(vr) // num_clips` window (`:345–398`).

When `fpc * frame_step < len(video)`, that window covers only a **sub-segment** of the trajectory:

| Video len | `fpc` | `frame_step` | Coverage |
|-----------|-------|--------------|----------|
| 64 | 16 | 1 | first 16 frames ≈ **first ¼** of the trajectory (buggy for motion) |
| 64 | 16 | 4 | frames 0…63 — **whole clip** |
| 16 | 16 | 1 + `uniform_sampling` | all 16 frames (stride moot) |

On a 64-frame toy clip with `frame_step=1`, `fpc=16`, the sampler grabs 16 frames spanning only the **first ¼** of the trajectory — the object barely moves within a tubelet, so per-tubelet motion is sub-patch and **layer-0 cannot encode speed or acceleration**, suppressing the early-speed signal in Fig.2c. Two valid fixes:

- `frame_step = 4` (so `16 × 4 = 64` spans the whole 64-frame clip), **or**
- `uniform_sampling: true` (span `[0, len-1]` regardless of length).

The **Blender clips are natively 16 frames**, so their configs pair `frame_step: 1` with `uniform_sampling: true` — 16 frames requested from a 16-frame video picks every frame, and `frame_step` is irrelevant.

## Key code

- `src/datasets/video_dataset.py:337` — the branch (quoted above). Consumed after `vr.seek(0)`; returns `(buffer, [indices])` mirroring the stock return shape.
- `src/datasets/video_dataset.py:158` — the guard that forces exactly one of `fps`/`duration`/`frame_step`, evaluated in `__init__` **before** the branch is ever reached:

```python
# src/datasets/video_dataset.py:158-161
if sum([v is not None for v in (fps, duration, frame_step)]) != 1:
    raise ValueError(
        f"Must specify exactly one of either {fps=}, {duration=}, or {frame_step=}."
    )
```

- `evals/analysis_vlm/eval.py:173-175` — fail-loud `resize_mode` validator (never silently falls back to `crop`):

```python
clip_resize_mode = args_data.get("resize_mode", "crop")
if clip_resize_mode not in ("crop", "resize"):  # fail loud
    raise ValueError(f"data.resize_mode must be 'crop' or 'resize', got {clip_resize_mode!r}")
```

- `evals/analysis_vlm/eval.py:389-403` — the `resize` branch that swaps only the transform and calls `init_data` directly, forwarding `uniform_sampling=uniform_sampling` and forcing `drop_last=False` on the batch sampler (partial-batch fix for tiny val splits).
- `evals/analysis_vlm/eval.py:593,607-611` — `_DirectResizeClipTransform`; the `Compose` core:

```python
# evals/analysis_vlm/eval.py:607-611  (_DirectResizeClipTransform.__init__)
self.eval_transform = vt.Compose([
    vt.Resize((crop_size, crop_size), interpolation="bilinear"),  # (w,h) tuple -> direct resize
    vvt.ClipToTensor(),
    vt.Normalize(mean=normalize[0], std=normalize[1]),
])
```

## Configuration

Real example — `configs/analysis/blender_toy_dataset/vjepa_combined.yaml` (`data` block, lines 40–51):

```yaml
experiment:
  data:
    dataset_type: VideoDataset
    resolution: 256            # squashed to 256x256 by resize (16x16 patches)
    resize_mode: resize        # crop (default, shorter-side+center-crop) | resize (direct HxW squash)
    frame_step: 1              # native 16f clips (legacy 64f toy used step 4 to subsample)
    uniform_sampling: true     # 16 frames evenly across the whole video == all frames
    num_segments: 1            # MUST stay 1 with uniform_sampling
    num_views_per_segment: 1
    dataset_train: /.../blender_toy/combined_train.csv
    dataset_val:   /.../blender_toy/combined_val.csv
    frames_per_clip: 16        # paper inference = 16f @ 24fps
    num_classes: 4             # ignored for regression
```

`resolution` is **not** universal across Blender configs — `resize` squashes the frame to `resolution²` whatever the value:

| Config | `resolution` | Grid | Note |
|--------|-------------|------|------|
| `vjepa_combined.yaml:42` | `256` | 16×16 patches | default toy geometry |
| `vjepa_attn_distance.yaml:45` | `224` | 14×14 patches × 8 tubelets = **1568 tokens** | paper geometry (attention-distance) |

Both use the identical `resize_mode: resize` + `uniform_sampling: true`; attention distance is measured in **patch units**, so either grid works — 224 just matches the paper.

### Key reference

| Key | Meaning | Default | Allowed values |
|-----|---------|---------|----------------|
| `data.uniform_sampling` | Sample `fpc` frames evenly over the whole video; ignore `frame_step`/`num_clips` | `false` | `true` / `false` |
| `data.resize_mode` | Clip spatial op (V-JEPA clip path only) | `'crop'` | `'crop'` \| `'resize'` |
| `data.frame_step` | Stock-sampler temporal stride; `clip_len = fpc*frame_step` | (must set exactly one of `frame_step`/`fps`/`clip_duration`) | int ≥ 1 |
| `data.resolution` | Target square edge; `resize` → `resolution²` | `224` | int (e.g. `224`, `256`) |
| `data.num_segments` | `num_clips` for the stock sampler | `1` | int ≥ 1 (**keep 1** with `uniform_sampling`) |
| `data.frames_per_clip` | `fpc` requested per clip | `16` | int |

> **Note on `resize_mode` values `'smart'` / `'fixed'`:** these appear in the **Qwen/VLM** raw-frame configs (e.g. `qwen3vl_analysis.yaml`) and are a *different* code path (min/max-pixels vs fixed H/W), **not** the clip `'crop'|'resize'` handled here. The clip validator would raise on them.

Omitting `resize_mode` → `'crop'`; omitting `uniform_sampling` → `false`; either restores upstream behavior.

## Invariants & gotchas

- **`frame_step` (or `fps`/`duration`) must still be set even with `uniform_sampling=True`.** The `__init__` guard (`:158`) requires exactly one, and `clip_len` is still computed for the `filter_short_videos` check (`:326–330`) *before* the branch. Blender configs satisfy this with `frame_step: 1`.
- **Keep `num_segments`/`num_clips = 1` with `uniform_sampling`.** The branch returns exactly `fpc` frames as one clip; downstream `split_into_clips` (`:251–255`) would slice an `fpc`-length buffer into `num_clips` sub-clips and yield empty/short clips if `num_clips > 1`.
- **`getattr(self, "uniform_sampling", False)`** is defensive — a `VideoDataset` pickled before the field existed still works (treated as off).
- **`resize_mode` defaults to `'crop'`**, so existing V-JEPA analysis runs are unchanged unless the config sets `resize`. Invalid values fail loud rather than silently defaulting.
- **Why `resize` for toy physics:** the object can hug the frame edges; `crop` throws those pixels away and loses motion, so toy configs use `resize` (see the canonical comment at `configs/analysis/vjepa_analysis.yaml:58` and the legacy `configs/analysis/toy_dataset/vjepa_acceleration.yaml:39` "full-frame preserve; crop → edge loss → motion loss").
- **Classification is never affected.** The stock `video_classification_frozen` `main()` never reads `uniform_sampling` from config — the `make_dataloader` param is inert plumbing there.
- **Default-off end-to-end.** Every threading hop defaults `False`; `resize_mode` defaults `'crop'`; the `vit_encoder_multiclip.py` delta is comments-only. A config that sets none of these produces a byte-identical run to `204698b`.

## Cross-references

- [02 — analysis-vlm harness](02-analysis-vlm-harness.md) — the `analysis_vlm/eval.py` entrypoint that reads both knobs.
- [05 — analysis clip harness](05-analysis-clip-harness.md) — the clip data path (`make_dataloader`, `VideoTransform`) these knobs branch around.
- [09 — Blender toy dataset](09-blender-toy-dataset.md) & [10 — datasets / CSV / targets](10-datasets-csv-targets.md) — the native-16-frame clips that motivate `uniform_sampling`, and the CSV/`.npy` targets.
- [11 — attention hooks](11-attention-hooks.md) & [12 — analysis modes](12-analysis-modes.md) — the post-hoc `attention_distance` mode **consumes exactly these two knobs** (`uniform_sampling: true`, `resize_mode: resize`, `resolution: 224`) via `configs/.../vjepa_attn_distance.yaml`; the Fig.3 heatmap / Fig.19 layerwise plot, `query_chunk`/`max_batches`/`annotate` options, the `run_attn_distance_vjepa.sh` launcher, and the `modes` dispatch scaffold (`skip_base_probe`) live there, not here.
- [13 — configs reference](13-configs-reference.md) — the full `experiment.data` schema.
- [14 — reproduction status & findings](14-reproduction-status-and-findings.md) — the Fig.2c early-speed / direction-emergence result that this `uniform_sampling` fix unblocked, and the attention-distance PEZ reproduction.
