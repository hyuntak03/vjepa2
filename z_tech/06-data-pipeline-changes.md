# Data-pipeline changes (`uniform_sampling`, `resize_mode`, `frame_step`)

## Purpose

This fork adds two additive, default-off knobs to the shared video data pipeline so that
short, fixed-length toy clips (the Blender / toy-physics datasets used for the Fig.2c
layer-wise R² reproduction) are sampled and resized in a way that preserves whole-clip
motion:

1. **`uniform_sampling`** — evenly sample `frames_per_clip` frames across the *entire*
   video, length-agnostic, ignoring `frame_step` and `num_clips`. Threaded from config
   through `init_data` → `make_videodataset` → `VideoDataset.loadvideo_decord`.
2. **`resize_mode: 'resize'`** — direct `H×W` resize (aspect squashed) instead of the stock
   shorter-side-resize + center-crop. Implemented as a drop-in transform in the
   `analysis_vlm` eval only.

Both default to the *upstream* behavior, so existing runs are byte-identical unless the
config opts in.

## What changed vs upstream V-JEPA2

Baseline = commit `204698b`.

| File | Kind | Delta |
|------|------|-------|
| `src/datasets/video_dataset.py` | **modified** | New `uniform_sampling` param on `make_videodataset` + `VideoDataset.__init__`; early-return branch in `loadvideo_decord` that evenly samples `fpc` frames over the whole video. |
| `src/datasets/data_manager.py` | **modified** | New `uniform_sampling` param on `init_data`, forwarded to `make_videodataset` (VideoDataset branch only). |
| `evals/video_classification_frozen/eval.py` | **modified** | New `uniform_sampling` param on `make_dataloader`, forwarded to `init_data`. **Note:** the classification `main()` does *not* read it from config — only the plumbing exists here. |
| `evals/analysis_vlm/eval.py` | **new (fork dir)** | Reads `data.uniform_sampling` and `data.resize_mode` from YAML; defines `_DirectResizeClipTransform` for `resize_mode:'resize'`. This is the only entrypoint that actually wires both knobs from config. |
| `configs/analysis/**` | **new** | Toy/Blender configs set `uniform_sampling: true`, `resize_mode: resize`, `frame_step: 1`. |

The exact additive deltas (`git diff 204698b -- src/datasets/video_dataset.py src/datasets/data_manager.py`) are all pure parameter-threading plus this one branch:

```python
# src/datasets/video_dataset.py:334-341  (inside VideoDataset.loadvideo_decord)
# uniform_sampling: pick `fpc` frames evenly across the WHOLE video (length-agnostic;
# ignores frame_step / num_clips). Avoids the contiguous-window default that, when
# fpc*frame_step < len(video), only covers a sub-segment -> sub-patch motion per tubelet.
if getattr(self, "uniform_sampling", False):
    n = len(vr)
    indices = np.clip(np.linspace(0, n - 1, num=fpc).round(), 0, n - 1).astype(np.int64)
    buffer = vr.get_batch(list(indices)).asnumpy()
    return buffer, [indices]
```

Everything above this branch (the `fps`/`duration`/`frame_step` validation, the
`filter_long_videos`/`filter_short_videos` guards, `vr.seek(0)`) still runs; only the
segment-partition sampling below it is bypassed.

---

## 1. `uniform_sampling`

### Threading

- `src/datasets/data_manager.py:40` — `init_data(..., uniform_sampling=False)`, forwarded at
  `data_manager.py:87` to `make_videodataset` **only in the `videodataset` branch**
  (ImageNet path is untouched).
- `src/datasets/video_dataset.py:54` — `make_videodataset(..., uniform_sampling=False)` →
  passed to `VideoDataset(...)` at `video_dataset.py:71`.
- `src/datasets/video_dataset.py:142` — `VideoDataset.__init__(..., uniform_sampling=False)`
  stored as `self.uniform_sampling` (`video_dataset.py:147`).
- Consumed by the branch at `src/datasets/video_dataset.py:337`.

### Semantics

When `uniform_sampling=True`, `loadvideo_decord` returns exactly `fpc` frame indices
`round(linspace(0, len-1, fpc))` — the first and last frame of the video are always
endpoints, and the middle `fpc-2` are evenly spaced. This is **length-agnostic and fully
deterministic**: `frame_step`, `num_clips`, and `random_clip_sampling` are all ignored on
this path. For a native 16-frame clip with `fpc=16`, this selects *all 16 frames* in order.

### Why it exists (frame_step reproduction bug)

The stock (`uniform_sampling=False`) path partitions the video into `num_clips` equal
segments and, per segment, samples a window of `clip_len = fpc * frame_step` contiguous-ish
frames (`video_dataset.py:343-398`). When `fpc * frame_step < len(video)`, that window
covers only a **sub-segment** of the trajectory. On a 64-frame toy clip with `frame_step=1`
and `fpc=16`, the sampler grabs 16 frames spanning the **first 1/4 of the trajectory** — the
object barely moves within a tubelet, so per-tubelet motion is sub-patch and **layer-0
cannot encode speed or acceleration**. This directly suppressed the early-speed signal in
Fig.2c.

Two valid fixes:
- `frame_step = 4` (so `16 × 4 = 64` spans the whole 64-frame clip), or
- `uniform_sampling: true` (span the whole video regardless of length).

The **Blender clips are natively 16 frames**, so their configs use `frame_step: 1` +
`uniform_sampling: true` — with 16 frames requested from a 16-frame video, uniform sampling
picks every frame, and `frame_step` is moot.

### Gotchas / invariants

- **`frame_step` (or `fps`/`duration`) must still be set even with `uniform_sampling=True`.**
  `VideoDataset.__init__` enforces "exactly one of `fps`/`duration`/`frame_step` is
  non-None" (`video_dataset.py:158-161`) *before* the branch is reached, and `clip_len` is
  still computed for the `filter_short_videos` guard (`video_dataset.py:326-330`). The
  configs satisfy this with `frame_step: 1`.
- **Keep `num_segments`/`num_clips = 1`.** The branch returns exactly `fpc` frames as a
  single clip; downstream `split_into_clips` (`video_dataset.py:251-255`) would slice an
  `fpc`-length buffer into `num_clips` sub-clips and produce empty/short clips if
  `num_clips > 1`. Toy configs set `num_segments: 1`.
- **`getattr(self, "uniform_sampling", False)`** is defensive — a `VideoDataset` pickled
  before this field existed still works (treated as off).
- **Default-off:** every hop defaults to `False`; the stock segment-partition sampler is
  untouched when the flag is not passed.
- The stock `video_classification_frozen` `main()` never reads this from config, so
  classification evals are unaffected regardless of the config file.

---

## 2. `resize_mode: 'resize'`

Lives entirely in `evals/analysis_vlm/eval.py`; **not** in the shared `src/datasets` code.

- Parsed at `evals/analysis_vlm/eval.py:173` with a fail-loud validator:

```python
# evals/analysis_vlm/eval.py:173-175
clip_resize_mode = args_data.get("resize_mode", "crop")
if clip_resize_mode not in ("crop", "resize"):  # fail loud (don't silently fall back to crop)
    raise ValueError(f"data.resize_mode must be 'crop' or 'resize', got {clip_resize_mode!r}")
```

- **`crop` (default, upstream behavior):** the clip loader routes through
  `video_classification_frozen.make_dataloader` → stock `VideoTransform`, i.e. shorter-side
  resize to `int(crop_size * 256/224)` then `CenterCrop(crop_size)`
  (`evals/video_classification_frozen/utils.py:68-72`). The model's trained spatial pipeline;
  edges are cropped.
- **`resize`:** swaps *only* the transform for `_DirectResizeClipTransform` and calls
  `init_data` directly, mirroring `make_dataloader`'s args exactly
  (`evals/analysis_vlm/eval.py:389-403`). The transform does a direct `(crop, crop)` resize:

```python
# evals/analysis_vlm/eval.py:607-611  (_DirectResizeClipTransform.__init__)
self.eval_transform = vt.Compose([
    vt.Resize((crop_size, crop_size), interpolation="bilinear"),  # (w,h) tuple -> direct resize
    vvt.ClipToTensor(),
    vt.Normalize(mean=normalize[0], std=normalize[1]),
])
```

`vt.Resize` with a **tuple** size resizes to that exact `(w, h)` (`transforms.py:856-872`),
so aspect ratio is squashed and the **full frame is preserved** (no edge loss) — matching the
VLM SigLIP square-resize path. It is deterministic (no train-time augmentation).

### Gotchas / invariants

- **Default is `'crop'`**, so existing V-JEPA analysis runs are unchanged unless the config
  sets `resize_mode: resize`. The stock `VideoTransform` / `make_transforms` are left
  untouched — `resize` uses a *separate* transform class.
- For toy physics the object can hug the frame edges; `crop` throws those pixels away and
  loses motion, so toy configs use `resize` (see `configs/.../vjepa_acceleration.yaml:39`:
  "384² 전체 보존 (crop은 가장자리 손실 → motion 손해)"). Blender configs use `resolution: 256`,
  so `resize` produces `256×256`.
- `resize_mode` values `'smart'`/`'fixed'` seen in some configs are the **Qwen/VLM** raw-path
  knobs (min/max-pixels vs fixed H/W), a *different* code path — not the clip `'crop'|'resize'`
  handled here.

---

## 3. `frame_step` / clip-sampling semantics

`frame_step` (config `frame_step` → `init_data(frame_sample_rate=...)` →
`VideoDataset.frame_step`) sets the temporal stride of the stock sampler. Exactly one of
`frame_step`, `fps`, `duration` may be set (`video_dataset.py:158`). With `frame_step` the
per-clip window length is `clip_len = fpc * frame_step` (`video_dataset.py:326`), and each of
`num_clips` segments samples `np.linspace(start, end, fpc)` inside a `partition_len =
len(vr) // num_clips` window (`video_dataset.py:343-398`).

Consequences that drove the toy configs:

| Video len | `fpc` | `frame_step` | Coverage |
|-----------|-------|--------------|----------|
| 64 | 16 | 1 | first 16 frames ≈ **first 1/4** of trajectory (buggy for motion) |
| 64 | 16 | 4 | frames 0…63 — **whole clip** |
| 16 | 16 | 1 + `uniform_sampling` | all 16 frames (stride moot) |

`uniform_sampling` sidesteps `frame_step` entirely by spanning `[0, len-1]`, which is why the
native-16-frame Blender clips pair `frame_step: 1` with `uniform_sampling: true`.

---

## Config

Real example — `configs/analysis/blender_toy_dataset/vjepa_combined.yaml` (`data` block):

```yaml
experiment:
  data:
    dataset_type: VideoDataset
    resolution: 256
    resize_mode: resize        # crop (default, shorter-side+center-crop) | resize (direct HxW)
    frame_step: 1              # native 16f clips (legacy 64f toy used step 4 to subsample)
    uniform_sampling: true     # 16 frames evenly across whole video == all frames
    num_segments: 1            # MUST stay 1 with uniform_sampling
    num_views_per_segment: 1
    dataset_train: /.../blender_toy/combined_train.csv
    dataset_val:   /.../blender_toy/combined_val.csv
    frames_per_clip: 16        # paper inference = 16f @ 24fps
    num_classes: 4             # ignored for regression
```

Omitting `resize_mode` → `'crop'`; omitting `uniform_sampling` → `False`; both restore
upstream behavior.

---

## Key project findings

1. **Fig.2c reproduction (positive result).** On the paper-faithful Blender toy dataset, the
   frozen V-JEPA2-L layer-wise R² reproduces the paper's dissociation: **SPEED** is decodable
   early (R² ≈ 0.68 at layer 0); **DIRECTION** emerges sharply in the Physics Emergence Zone
   (≈ 0.28 at L0 → ≈ 0.9 by layer-fraction 0.3–0.4); **accel_mag** sits in between. An earlier
   anti-shortcut toy generator (random shape/color/size) did **not** reproduce early-speed;
   the fixes were (a) a paper-faithful single fixed red sphere and (b) correct frame sampling
   (this doc's `uniform_sampling`).
2. **`frame_step` reproduction bug (the reason `uniform_sampling` exists).**
   `VideoDataset.loadvideo_decord` with `frame_step=1` on a 64-frame clip samples 16
   **contiguous** frames — the first 1/4 of the trajectory — so per-tubelet motion is
   sub-patch and **layer-0 cannot encode speed/accel**. Fix: `frame_step=4` (span the whole
   clip) **or** `uniform_sampling=true` (evenly sample `fpc` frames over the whole video,
   length-agnostic). Blender clips are natively 16 frames, so their configs use `frame_step:
   1` + `uniform_sampling: true`.
