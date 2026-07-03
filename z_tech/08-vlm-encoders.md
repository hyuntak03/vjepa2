# VLM encoder backends (LLaVA-Video, Qwen3-VL)

## Purpose

Two frozen-encoder wrappers that expose **per-layer vision features** of large
VLMs to the unified probing harness (`evals/analysis_vlm/`, see `07-*`). Each loads
**only the vision tower** of a VLM (the multi-billion-param LLM is never
instantiated) and returns a `list[Tensor(B, N, D)]` — one tensor per requested
*stage* — that the harness caches and probes layer-by-layer.

- `evals/analysis_vlm/modelcustom/llava_video_encoder.py` — LLaVA-Video-7B-Qwen2, a **SigLIP** vision tower + `mm_projector`. Per-frame (framewise) encoding.
- `evals/analysis_vlm/modelcustom/qwen3vl_encoder.py` — Qwen3-VL-4B-Instruct vision ViT, with the spatial **merger** and **deepstack** taps.

Both plug into the **stock upstream** `init_module` dispatcher
(`evals/video_classification_frozen/models.py:14`, unchanged vs upstream) via
`importlib` on `module_name`; the harness picks the backend from
`experiment.analysis.model` (`evals/analysis_vlm/eval.py:62`).

## What changed vs upstream V-JEPA2

Upstream `204698b` has **no** `evals/analysis_vlm/` tree at all — the entire
subsystem is new (added in commit `91fa127`, "analysis module added";
`git ls-tree 204698b -- evals/analysis_vlm` is empty). Nothing that ships upstream
is modified for these backends.

| File | Status | Delta |
|---|---|---|
| `evals/analysis_vlm/modelcustom/llava_video_encoder.py` | **new** | SigLIP tower loader + stage taps (this doc) |
| `evals/analysis_vlm/modelcustom/qwen3vl_encoder.py` | **new** | Qwen3-VL ViT loader + merger/deepstack taps (this doc) |
| `evals/analysis_vlm/loadutil.py` | **new** | shared weight-location resolver (`resolve_model_dir`) |
| `configs/analysis/llavavideo_analysis.yaml` | **new** | LLaVA backend config |
| `configs/analysis/qwen3vl_analysis.yaml` | **new** | Qwen3-VL backend config |
| `evals/video_classification_frozen/models.py` | **unchanged** | `init_module` dispatcher reused as-is (`git diff 204698b` is empty) |

The `init_module` API contract both backends satisfy is upstream's
(`models.py:31`): `importlib.import_module(module_name).init_module(frames_per_clip,
resolution, checkpoint, model_kwargs, wrapper_kwargs)` → an `nn.Module`, then
`.to(device).eval()` and all params frozen (`requires_grad=False`).

## Per-env: each backend has its OWN conda env

The two families have mutually incompatible dependency stacks, so **each runs in
its own conda env** and only the *selected* backend is imported (lazily, via
`importlib` on `module_name`) — so heavy/conflicting deps never co-load
(`evals/analysis_vlm/eval.py:16-18`).

| Backend | conda env | extra requirement |
|---|---|---|
| `llavavideo` | `lmms_eval_llavavideo` | LLaVA-NeXT source repo on `sys.path` (for `llava.model...siglip_encoder`) via `wrapper_kwargs.llava_repo` |
| `qwen3vl` | `lmms_eval_py311_2.7` | `transformers` with `qwen3_vl` (`Qwen3VLVisionModel`) |
| `vjepa` (baseline) | `vjepa2` | — |

Launcher pairs env+config: `z_scripts/run_analysis_vlm.sh` (`CONDA_ENV=... CONFIG=...`,
lines 18-20). The env is **not** auto-selected by the config; picking the wrong
env just fails at import.

## Shared entry contract

`init_module(resolution, frames_per_clip, checkpoint, model_kwargs, wrapper_kwargs)`
(LLaVA `llava_video_encoder.py:73`, Qwen `qwen3vl_encoder.py:43`). Both read all
knobs from `wrapper_kwargs` (aka `model_kwargs.wrapper_kwargs`), resolve weights via
`loadutil.resolve_model_dir`, and return an encoder exposing:

- `.stages : list[str]` — **resolved** (`"all"` already expanded, toggles applied)
- `.embed_dims : list[int]` — aligned with `.stages`
- `.num_temporal : int` — temporal positions (enables `framewise` cache; VLM-only)
- `.tubelet_size : int`
- `forward(frames_list) -> list[Tensor(B, N, D)]` aligned with `.stages`;
  `frames_list` is a list of `B` tensors, each `(T, H, W, C)` **uint8 raw RGB frames**
  (the harness's `raw` data mode — native VLM preprocessing happens *inside* the wrapper).

**Weight resolution** (`loadutil.py:30`): `wrapper_kwargs.pretrained` (or `checkpoint`)
— if a local dir, `find_snapshot` locates `config.json` (accepts an HF-cache
`models--*/snapshots/*` root); otherwise it is treated as an HF **repo id** resolved
under `cache_dir` **offline-first** (`snapshot_download(..., local_files_only=True)`,
downloading only on miss). The harness copies `model_kwargs.cache_dir` into
`wrapper_kwargs.cache_dir` for the raw path (`eval.py:143`).

## Stage spec — one shape, three encoders

All three backends accept the **same** `analysis.stages` grammar
(`llava..._resolve_stages:172`, `qwen..._resolve_stages:162`):

- **structured dict** (preferred): `{vision_encoder: [i,...] | "all", <toggle>: true, ...}` — only `vision_encoder` carries a per-layer selection; everything else is a boolean toggle (Qwen's `deepstack` additionally accepts a subset list of its merge indexes).
- **shorthand**: `"all"` == `{vision_encoder: all}`; `[int, ...]` == `{vision_encoder: [...]}`.
- **legacy**: a list of concrete stage-name strings, e.g. `["after_vision_encoder","after_projector"]`.

Invariant: an empty selection raises `analysis.stages selected nothing`
(`llava:197`, `qwen:196`). Out-of-range layer/block indices and unknown deepstack
indexes fail loud.

---

## LLaVA-Video encoder (SigLIP)

Loads **LLaVA-Video-7B-Qwen2** vision tower + projector, **no LLM**. Verified
recipe (`llava_video_encoder.py:1-24`):

- Vision-tower weights are **double-nested** under `model.vision_tower.vision_tower.*`;
  projector (`mlp2x_gelu`) under `model.mm_projector.*` (`:108`).
- Build `SigLipVisionModel(SigLipVisionConfig())`, **delete the last encoder layer**
  (→ **26** layers, index 0..25), set `head = Identity`, then `load_state_dict`
  with a strict assert that **no keys are missing/unexpected** (`:120-124`):

```python
vt = SigLipVisionModel(SigLipVisionConfig())
del vt.vision_model.encoder.layers[-1:]        # -> 26 layers
vt.vision_model.head = nn.Identity()
miss, unexp = vt.load_state_dict(vis_sd, strict=False)
assert not miss and not unexp, f"vision tower key mismatch: ..."
```

- Projector is `Linear(1152,3584) -> GELU -> Linear(3584,3584)`, loaded `strict=True` (`:140`).

**Framewise**: SigLIP is applied **per frame**, so each frame yields a fixed
**729-token** (27×27) grid at 1152-d, and the wrapper reshapes to `(B, T*Ntok, D)`
(`:262-263`). `num_temporal = frames_per_clip` (`:149`), `tubelet_size = 1`.

### Stages (dim / token layout)

| stage | dim | tokens | source (`forward._stage`, `:248`) |
|---|---|---|---|
| `layer_<i>`, i∈0..25 | 1152 | T·729 | `hidden_states[i+1]` |
| `after_vision_encoder` | 1152 | T·729 | `hidden_states[-1]` (== `layer_25`) |
| `after_projector` | 3584 | T·729 | projector(`hidden_states[-1]`) |
| `after_vision_encoder_pool2` | 1152 | T·196 | 2× bilinear spatial pool of final |
| `after_projector_pool2` | 3584 | T·196 | 2× bilinear pool of projector out |
| `"all"` | — | — | `[layer_0 .. layer_25, after_projector]` |

Pooling (`_pool2d:208`) uses `spatial_pool_stride` (default **2**, bilinear
`F.interpolate`) so 729 → 196 (ceil(27/2)²=14²).

### LLaVA-only knobs (`wrapper_kwargs`)

| key | default | meaning |
|---|---|---|
| `llava_repo` | `/data/hyuntak/project/2026/vlm_direction/LLaVA-NeXT` | path prepended to `sys.path` for `llava` SigLIP code (`:84-86`) |
| `pretrained` / `cache_dir` | — | weight location (repo id + HF cache, or local dir) |
| `out_stages` | `["after_vision_encoder","after_projector"]` | stage spec (harness passes `analysis.stages` here) |
| `spatial_pool_stride` | `2` | bilinear pool stride for `*_pool2` |
| `vision_chunk` | `32` | frames per SigLIP sub-forward (bounds attention memory) |
| `attn_implementation` | `sdpa` | `sdpa` (fused) or `eager` |
| `dtype` | `float16` | encoder forward dtype (`float32` for CPU debug) |

### Gotchas / invariants (LLaVA)

- **SDPA monkeypatch** (default on): `SigLipAttention.forward` is swapped for a
  fused `scaled_dot_product_attention` version (`_siglip_sdpa_forward:59`, same
  math, ~2-3× faster, avoids the `(B,heads,729,729)` fp32 score matrix). Opt out
  with `attn_implementation: eager`.
- **Frame chunking** (`:234`): with `B*T > vision_chunk` the tower runs in
  frame-chunks and per-layer hidden states are concatenated back — required to
  avoid OOM on the eager score matrix; result is numerically identical.
- Projector is applied only when a `*projector*` stage is selected
  (`self._need_proj`, `:151`, `:246`).
- Strict key assert (`:124`) makes a **different LLaVA/SigLIP variant hard-fail** rather than silently mis-load.
- `conv_template` / `video_decode_backend` / `mm_spatial_pool_mode` / `max_frames_num` / `force_sample` from lmms-eval **do not** affect vision-only extraction (`:80-82`).

---

## Qwen3-VL encoder (merger / deepstack)

Loads **Qwen3-VL-4B-Instruct** vision ViT, **no LLM**. Recipe
(`qwen3vl_encoder.py:1-27`): build **only**
`Qwen3VLVisionModel(Qwen3VLConfig.from_pretrained(snap).vision_config)` and load
just `model.visual.*` (`:76-90`, strict no-missing/unexpected assert). Real 4B
vision config: `hidden_size=1024`, `out_hidden_size=2560`, `depth=24`,
`spatial_merge_size=2`, `patch_size=16`, `temporal_patch_size=2`,
`deepstack_visual_indexes=[5,11,17]`.

`num_temporal = frames_per_clip // temporal_patch_size` (`:103`, e.g. 8//2 = **4**
grid_t positions), `tubelet_size = temporal_patch_size` (2 frames per token).

### Stages → layer mapping

Raw per-block outputs (1024-d, **pre-merge**) are captured via **forward hooks**
(`_mk_hook:200`, registered only for the blocks a stage needs, `:106-115`). The
merger halves each spatial dim (÷`merge²`=4) and lifts 1024→2560.

| stage | dim | tokens | source (`forward`, `:254`) |
|---|---|---|---|
| `block_<i>`, i∈0..23 | 1024 | grid_t·gh·gw | hook capture `self._captured[i]` |
| `before_merger` | 1024 | grid_t·gh·gw | `self._captured[depth-1]` (== `block_23`) |
| `after_merger` | 2560 | grid_t·(gh/2)·(gw/2) | `image_embeds` (fed to the LLM) |
| `deepstack_<i>`, i∈[5,11,17] | 2560 | grid_t·(gh/2)·(gw/2) | `deepstack[deepstack_idx.index(i)]` |
| `"all"` | — | — | `[block_0 .. block_23, after_merger]` |

`self.visual(pixel_values, grid_thw)` returns `(image_embeds, deepstack)`; the
wrapper splits each along dim 0 by per-video token counts (`after_sizes` /
`before_sizes`, `:239-241`) and stacks to `(B, N, D)`.

### Qwen-only knobs (`wrapper_kwargs`)

| key | default | meaning |
|---|---|---|
| `pretrained` / `cache_dir` | — | weight location |
| `out_stages` | `["before_merger","after_merger"]` | stage spec |
| `resize_mode` | `smart` | `smart` (processor `smart_resize` within `[min_pixels,max_pixels]`) or `fixed` |
| `min_pixels` / `max_pixels` | `8192` / `112896` | smart-resize token budget (lmms-eval defaults) |
| `qwen_fixed_h` / `qwen_fixed_w` | — | **required** for `resize_mode=fixed`; must be multiples of `patch*merge = 32` |
| `attn_implementation` | `sdpa` | `sdpa` / `eager` / `flash_attention_2` |
| `dtype` | `float16` | forward dtype (`float32` for CPU) |

`smart` builds the processor with `min/max_pixels` and also force-sets the attrs
(`:128-133`); `fixed` pre-resizes with `F.interpolate` to `(fixed_h, fixed_w)` and
passes `do_resize=False` (`_preprocess:214`).

### Gotchas / invariants (Qwen)

- **Variable token count under `smart`** (the key invariant): different input
  resolutions → different `grid_thw` → different `N`, so `torch.stack` across
  videos fails. `forward` catches this and raises a directive to use
  `resize_mode: fixed` + `qwen_fixed_h/w` (`_stack:243-251`). Same failure surfaces
  in the `tokens` cache concat (`cache.py:153-159`). **If your videos are already
  uniform resolution, `smart` yields a uniform grid and batches fine.**
- `deepstack` list is validated against the real `deepstack_visual_indexes`
  ([5,11,17]); anything else fails loud (`:175-176`, `:193-194`).
- Hooks are registered **only** for needed blocks, and `self._captured` is reset
  each `forward` (`:236`) — no cross-batch leakage.
- Strict key assert (`:89-90`) hard-fails on a mismatched Qwen variant/config.

---

## Config

Both are config-driven under `eval_name: analysis_vlm`. Real examples live at
`configs/analysis/llavavideo_analysis.yaml` and `configs/analysis/qwen3vl_analysis.yaml`.

### LLaVA-Video (`configs/analysis/llavavideo_analysis.yaml`)

```yaml
eval_name: analysis_vlm
experiment:
  analysis:
    model: llavavideo
    stages:
      vision_encoder: all              # SigLIP layers 0..25 (or e.g. [3,7,11,15,19,23,25])
      # after_projector: true          # 3584-d LLM-input space (toggle)
      # after_vision_encoder_pool2: true
    probes:
      - type: linear
        pooling: framewise_mean        # framewise -> keeps temporal order (VLM-only)
        pre_norm: true
        optimization: { lr: 0.01, weight_decay: 0.0, warmup: 1.0 }
  data:
    frames_per_clip: 16                # -> num_temporal = 16
    num_classes: 4
  optimization:
    cache_features: true
    cache_pooling: framewise           # (B,T,D) spatial-mean per frame; needs num_temporal
    cache_max_gb: 70
model_kwargs:
  checkpoint: lmms-lab/LLaVA-Video-7B-Qwen2      # HF repo id (or local snapshot dir)
  cache_dir: /data/dataset/LLaVA-Video-100K-Subset/
  module_name: evals.analysis_vlm.modelcustom.llava_video_encoder
  wrapper_kwargs:
    llava_repo: /data/hyuntak/project/2026/vlm_direction/LLaVA-NeXT
    spatial_pool_stride: 2
    dtype: float16
```

### Qwen3-VL (`configs/analysis/qwen3vl_analysis.yaml`)

```yaml
eval_name: analysis_vlm
experiment:
  analysis:
    model: qwen3vl
    stages:
      vision_encoder: [3, 7, 11, 15, 19, 23]   # ViT blocks 0..23 (raw, 1024-d)
      after_merger: true                        # 2560-d, LLM input
      deepstack: [5, 11, 17]                     # 2560-d injection points
    probes:
      - { type: linear, pooling: mean, pre_norm: true, optimization: { lr: 0.01 } }
      - { type: attentive, num_heads: 16, num_probe_blocks: 1, optimization: { lr: 0.002 } }
  data:
    frames_per_clip: 8                 # even; grid_t = 8/2 = 4
    num_classes: 4
  optimization:
    cache_features: true
    cache_pooling: tokens              # full tokens; needs uniform N (see gotcha)
    cache_max_gb: 64
model_kwargs:
  checkpoint: Qwen/Qwen3-VL-4B-Instruct
  cache_dir: /data/dataset/LLaVA-Video-100K-Subset/
  module_name: evals.analysis_vlm.modelcustom.qwen3vl_encoder
  wrapper_kwargs:
    resize_mode: smart                 # or: fixed + qwen_fixed_h/w (multiples of 32)
    min_pixels: 8192
    max_pixels: 112896
    attn_implementation: sdpa
    dtype: float16
```

### Checkpoints / cache_dir summary

| backend | `checkpoint` (repo id) | `cache_dir` |
|---|---|---|
| LLaVA-Video | `lmms-lab/LLaVA-Video-7B-Qwen2` | `/data/dataset/LLaVA-Video-100K-Subset/` |
| Qwen3-VL | `Qwen/Qwen3-VL-4B-Instruct` | `/data/dataset/LLaVA-Video-100K-Subset/` |

Either `checkpoint` may instead be a **local snapshot dir** (then `cache_dir` is
unused). Resolution is offline-first: cached weights are found via
`snapshot_download(local_files_only=True)` and only downloaded on miss
(`loadutil.py:49-57`).

## Framewise cache path

The `framewise` feature-cache mode is **VLM-only** and depends on the encoder
exposing `num_temporal`:

- `cache.py:88` `reduce_feature(feat, "framewise", num_temporal)` reshapes
  `(B, N, D)` → `(B, T, S, D)` and spatial-means over `S` → **`(B, T, D)`** — keeps
  temporal order (unlike `pooled`'s global mean‖max) while staying small. It
  **requires** `N % num_temporal == 0`, else raises.
- `eval.py:436` pulls `enc_num_temporal = getattr(encoder, "num_temporal", None)`
  and passes it into `build_feature_cache` (`:442-449`). LLaVA sets it to
  `frames_per_clip`; Qwen to `frames_per_clip // temporal_patch_size`.
- The matching probe side is the `framewise_*` linear pooling (`eval.py:299`,
  `TemporalLinearProbe`), which also needs `encoder.num_temporal` and errors if
  absent.
- **V-JEPA has no `num_temporal`**, so `cache_pooling: framewise` and the
  `framewise_*` pooling are only valid for these two VLM backends
  (`modes/REPRODUCTION_PLAN.md:21`).

## Default-off guarantees

- The entire `analysis_vlm` subsystem is reached **only** via `eval_name: analysis_vlm`; stock `evals/` and `main.py`/`scaffold.py` are untouched (`eval.py:13-14`).
- Only the **selected** backend module is imported (lazy `importlib` on `module_name`), so the two conda-env-specific dependency stacks never co-load (`eval.py:16-18`).
- `*_pool2` (LLaVA) and `deepstack` / `before_merger` / `after_merger` (Qwen) are **off unless explicitly toggled**; the projector is computed only when a projector stage is selected.
- SDPA is **on by default** but is a mathematically-neutral swap; `attn_implementation: eager` restores stock attention.
