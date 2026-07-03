# 08 — VLM encoder backends

> Two frozen, vision-only wrappers — LLaVA-Video (SigLIP tower + projector, per-frame) and Qwen3-VL (ViT + spatial merger + deepstack) — that expose the **per-layer / per-stage** features of large VLMs to the `analysis_vlm` probing harness, each loaded WITHOUT its multi-billion-param LLM and each run in its OWN conda env.

## Purpose

The unified probing harness (`evals/analysis_vlm/`, see [§02](02-analysis-vlm-harness.md)) selects one of **three** frozen encoders through `experiment.analysis.model`. Two of them are these new VLM backends; both run on the harness's **`raw`** data path (`data_mode='raw'`) — the dataloader hands each encoder a list of raw uint8 RGB frames and **all native VLM preprocessing happens inside the wrapper**. The third backend, `vjepa` (`data_mode='clip'`), is the V-JEPA2 ViT documented in [§05](05-analysis-clip-harness.md)/[§02](02-analysis-vlm-harness.md).

Each VLM wrapper:

- loads **only the vision tower** (the LLM is never instantiated — no weights, no memory),
- resolves a config-driven **stage** spec into concrete named taps, and
- `forward(frames_list)` returns a `list[Tensor(B, N, D)]` — **one tensor per requested stage**, aligned with `.stages` — that the harness caches and probes layer-by-layer.

| Backend (`analysis.model`) | File | Vision tower | Notable taps |
|---|---|---|---|
| `llavavideo` | `evals/analysis_vlm/modelcustom/llava_video_encoder.py` | **SigLIP** (LLaVA-Video-7B-Qwen2) + `mm_projector`; encoded **per frame** | `layer_0..25`, `after_projector`, `*_pool2` |
| `qwen3vl` | `evals/analysis_vlm/modelcustom/qwen3vl_encoder.py` | **Qwen3-VL-4B-Instruct** ViT + spatial **merger** + **deepstack** | `block_0..23`, `after_merger`, `deepstack_{5,11,17}` |

---

## What changed vs upstream V-JEPA2

Upstream `204698b` has **no** `evals/analysis_vlm/` tree at all — verified: `git ls-tree 204698b -- evals/analysis_vlm` is empty. The entire subsystem is new, added in commit **`91fa127`** ("analysis module added"). Nothing that ships upstream is modified for these backends; the stock `init_module` dispatcher is reused verbatim (`git diff 204698b -- evals/video_classification_frozen/models.py` is empty).

| File | Status | Delta |
|---|---|---|
| `evals/analysis_vlm/modelcustom/llava_video_encoder.py` | **new** (`91fa127`) | SigLIP tower + projector loader, framewise stage taps (this doc) |
| `evals/analysis_vlm/modelcustom/qwen3vl_encoder.py` | **new** (`91fa127`) | Qwen3-VL ViT loader, merger/deepstack hook taps (this doc) |
| `evals/analysis_vlm/loadutil.py` | **new** (`91fa127`) | shared offline-first weight resolver (`resolve_model_dir`, `find_snapshot`) |
| `configs/analysis/llavavideo_analysis.yaml` | **new** | LLaVA-Video backend config template |
| `configs/analysis/qwen3vl_analysis.yaml` | **new** | Qwen3-VL backend config template |
| `evals/video_classification_frozen/models.py` | **unchanged** | `init_module` dispatcher reused as-is (upstream diff empty) |

**Default-off guarantee.** The whole subsystem is reached **only** via `eval_name: analysis_vlm` (`scaffold.py` dynamic import); `evals/main.py`, `scaffold.py`, and the stock `evals/` paths are untouched (`eval.py:13-14`). Within a run, **only the selected backend module is imported** — lazily, via `importlib` on `module_name` (`models.py:32`) — so the two conda-env-specific dependency stacks never co-load (`eval.py:16-18`).

---

## Design & data flow

```
config: experiment.analysis.model = llavavideo | qwen3vl
        │
        ▼  eval.py:62-66  _BACKENDS[model] -> (module_name, data_mode='raw')
init_module(module_name, device, frames_per_clip, resolution, checkpoint,
            model_kwargs, wrapper_kwargs)                       (models.py:14)
        │  importlib.import_module(module_name).init_module(**kw)   (models.py:32)
        ▼
<backend>.init_module(...) -> LLaVAVideoEncoder | Qwen3VLEncoder (nn.Module)
        │  .to(device).eval(); every param requires_grad=False    (models.py:40-44)
        ▼
raw dataloader (evals/analysis_vlm/data.py) yields frames_list =
        list of B tensors, each (T, H, W, C) uint8 RGB
        │  _encode(...) calls encoder(frames_list)                 (eval.py:662-666)
        ▼
encoder.forward -> list[Tensor(B, N, D)]  aligned with .stages
        │
        ▼  harness caches (framewise/tokens/pooled) + trains one probe / (stage × probe × variable)
```

### Backend dispatch & `data_mode`

`experiment.analysis.model` maps to a `(module_name, data_mode)` pair (`eval.py:62-66`). Both VLM backends resolve to `data_mode='raw'`. Either half is overridable via `model_kwargs.module_name` / `experiment.analysis.data_mode`; if only `module_name` is set, `data_mode` defaults to `raw` when `"analysis_vlm"` is in the module path, else `clip` (`eval.py:116-117`). In `raw` mode the harness forwards `analysis.stages` verbatim to `wrapper_kwargs.out_stages` (`eval.py:142`) and copies `model_kwargs.cache_dir` into `wrapper_kwargs.cache_dir` so a repo-id checkpoint resolves (`eval.py:143-144`); it then reads the **resolved** stage list back off the encoder (`eval.py:276-277`).

### Per-env: each backend has its OWN conda env

The two families have mutually incompatible dependency stacks, so **each runs in its own conda env** and only the *selected* backend is imported — so heavy/conflicting deps never co-load.

| Backend | conda env | extra requirement |
|---|---|---|
| `llavavideo` | `lmms_eval_llavavideo` | LLaVA-NeXT source repo on `sys.path` (for `llava.model...siglip_encoder`) via `wrapper_kwargs.llava_repo` |
| `qwen3vl` | `lmms_eval_py311_2.7` | `transformers` with `qwen3_vl` (`Qwen3VLVisionModel`) |
| `vjepa` (baseline) | `vjepa2` | — |

The launcher `z_scripts/run_analysis_vlm.sh` pairs env+config (`CONDA_ENV=... CONFIG=...`). The env is **not** auto-selected by the config — picking the wrong env just fails at import.

```bash
sbatch --export=ALL,CONDA_ENV=lmms_eval_llavavideo,\
CONFIG=configs/analysis/llavavideo_analysis.yaml z_scripts/run_analysis_vlm.sh
# Qwen3-VL:  CONDA_ENV=lmms_eval_py311_2.7  CONFIG=configs/analysis/qwen3vl_analysis.yaml
```

### Shared entry contract (`init_module`)

Both backends define the same signature — **note the argument order matches the encoders' `def`, NOT the docstring in `models.py`**:

```python
def init_module(resolution, frames_per_clip, checkpoint, model_kwargs, wrapper_kwargs): ...
#   LLaVA: llava_video_encoder.py:73     Qwen: qwen3vl_encoder.py:43
```

The stock dispatcher `models.py:32-39` calls this by **keyword** for every argument (`frames_per_clip=…, resolution=…, checkpoint=…, model_kwargs=…, wrapper_kwargs=…`), so the positional order is irrelevant — `resolution` first (encoder def) vs `frames_per_clip` first (dispatcher's own signature) both resolve correctly. Both backends ignore `resolution` (native VLM preprocessing sets its own size) and read every knob from `wrapper_kwargs` (aka `model_kwargs.wrapper_kwargs`).

### Encoder API surface (attributes exposed)

Every backend returns an `nn.Module` exposing exactly this surface (consumed across `eval.py` + `cache.py`):

| attribute | type | meaning |
|---|---|---|
| `.stages` | `list[str]` | **resolved** stage names (`"all"` already expanded, toggles applied); order defines forward output order |
| `.embed_dims` | `list[int]` | per-stage feature dim, aligned 1:1 with `.stages` |
| `.embed_dim` | `int` | `== embed_dims[0]` (fallback single-dim) |
| `.num_temporal` | `int` | temporal positions; enables the `framewise` cache/probe (**VLM-only**; V-JEPA does not expose it) |
| `.tubelet_size` | `int` | frames folded into one temporal token (LLaVA 1, Qwen 2) |
| `forward(frames_list)` | `list[Tensor(B,N,D)]` | one tensor per `.stages` entry; `frames_list` = list of B tensors `(T,H,W,C)` uint8 raw RGB |

`init_module` (`models.py:40-44`) then `.to(device).eval()`s it and sets `requires_grad=False` on every parameter — the encoders are strictly frozen feature extractors.

### Weight resolution (`loadutil.resolve_model_dir`)

Both backends locate weights through `resolve_model_dir(checkpoint, wrapper_kwargs)` (`loadutil.py:30`). Candidate = `wrapper_kwargs.pretrained` **or** `checkpoint` (`loadutil.py:39`):

- **local dir** → `find_snapshot` (`loadutil.py:18`) accepts a dir with `config.json` directly, OR an HF-cache `models--*/snapshots/*` root (globs `snapshots/*` for a `config.json`).
- **HF repo id** → resolved under `cache_dir` **offline-first**: `snapshot_download(cand, cache_dir=…, local_files_only=True)` (`loadutil.py:52`), downloading only on a cache miss (`loadutil.py:57`).

### Stage spec — one grammar, three encoders

All three backends accept the **same** `analysis.stages` grammar (`llava._resolve_stages:172`, `qwen._resolve_stages:162`):

- **structured dict** (preferred): `{vision_encoder: [i,…] | "all", <toggle>: true, …}` — only `vision_encoder` carries a per-layer/per-block selection; everything else is a boolean toggle (Qwen's `deepstack` additionally accepts a subset list of its merge indexes, or `true`/`"all"` for all of them).
- **shorthand**: the string `"all"` == `{vision_encoder: "all"}`; a bare `[int, …]` == `{vision_encoder: […]}`.
- **legacy**: a list of concrete stage-name strings, e.g. `["after_vision_encoder","after_projector"]` (validated per-name).

> **`"all"` expands to the encoder layers/blocks ONLY.** `{vision_encoder: "all"}` (or the `"all"` shorthand) yields `layer_0..layer_25` (LLaVA, 26 stages) / `block_0..block_23` (Qwen, 24 stages) and **nothing else** — `after_projector` / `after_merger` / `deepstack` are appended only when their toggle is **explicitly** set (`llava:188-197`, `qwen:180-198`). The two wrapper module docstrings (`llava:18`, `qwen:16`) claim `"all" -> [… , after_projector]` / `[… , after_merger]`; that is **stale/aspirational and does not match the code** — treat the code behavior above as authoritative.

**Invariant:** an empty selection raises `analysis.stages selected nothing` (`llava:197`, `qwen:197`). Out-of-range layer/block indices and unknown deepstack indexes fail loud.

---

## Key code

### LLaVA-Video encoder (SigLIP + projector, framewise)

Loads **LLaVA-Video-7B-Qwen2**'s SigLIP vision tower + `mm_projector`, **no LLM** (`llava_video_encoder.py`).

**Load recipe** (`:100-142`):
- Vision-tower weights are **double-nested** under `model.vision_tower.vision_tower.*`; the projector (`mlp2x_gelu`) under `model.mm_projector.*` (`:108`). Both are gathered across `model-*-of-*.safetensors` shards and upcast to fp32 (`:110-116`).
- Build `SigLipVisionModel(SigLipVisionConfig())`, **delete the last encoder layer** (→ **26** layers, index 0..25), set `head = Identity`, then `load_state_dict(strict=False)` with an assert that **no keys are missing/unexpected** (`:120-124`):

```python
vt = SigLipVisionModel(SigLipVisionConfig())
del vt.vision_model.encoder.layers[-1:]        # -> 26 layers (index 0..25)
vt.vision_model.head = nn.Identity()
miss, unexp = vt.load_state_dict(vis_sd, strict=False)
assert not miss and not unexp, f"vision tower key mismatch: ..."   # :124
```

- Projector is `Linear(1152,3584) -> GELU -> Linear(3584,3584)`, constructed at `:140` and loaded **`strict=True`** at `:141` — a mismatched projector hard-fails.

**Framewise encoding.** SigLIP is applied **per frame**, so each frame yields a fixed **729-token** (27×27) grid at 1152-d; the wrapper reshapes each stage to `(B, T·Ntok, D)` (`:262-263`). `num_temporal = frames_per_clip` (`:149`, passed positionally at `:91`), `tubelet_size = 1` (`:148`).

**Stages (dim / token layout).** Layer `i` reads `hidden_states[i+1]` (index 0 is the embeddings), so `layer_25 == after_vision_encoder == hidden_states[-1]`:

| stage | dim | tokens | source (`forward._stage`, `:248`) |
|---|---|---|---|
| `layer_<i>`, i∈0..25 | 1152 | T·729 | `hidden_states[i+1]` (`:251`) |
| `after_vision_encoder` | 1152 | T·729 | `hidden_states[-1]` (== `layer_25`) (`:253`) |
| `after_projector` | 3584 | T·729 | `projector(hidden_states[-1])` (`:255`) |
| `after_vision_encoder_pool2` | 1152 | T·196 | `_pool2d(hidden_states[-1])` (`:257`) |
| `after_projector_pool2` | 3584 | T·196 | `_pool2d(projector out)` (`:259`) |
| `"all"` shorthand | — | — | expands to `[layer_0 .. layer_25]` **only** |

Pooling (`_pool2d:208`) uses `spatial_pool_stride` (default **2**, bilinear `F.interpolate`) so 729 → 196 (`ceil(27/2)² = 14² = 196`).

**LLaVA-only knobs** (`wrapper_kwargs`):

| key | default | allowed | meaning |
|---|---|---|---|
| `pretrained` / `cache_dir` | — | path / repo-id + cache | weight location (falls back to `model_kwargs.checkpoint`) |
| `out_stages` | `["after_vision_encoder","after_projector"]` | stage grammar | stage spec (harness overrides with `analysis.stages`, `:88`) |
| `llava_repo` | `/data/hyuntak/project/2026/vlm_direction/LLaVA-NeXT` | path | prepended to `sys.path` for `llava` SigLIP code (`:84-86`) |
| `spatial_pool_stride` | `2` | int ≥ 1 | bilinear pool stride for `*_pool2` (`:90`, `:145`) |
| `vision_chunk` | `32` | int | frames per SigLIP sub-forward (bounds attention memory, `:92`, `:150`) |
| `attn_implementation` | `sdpa` | `sdpa` \| `eager` | `sdpa` = fused SDPA monkeypatch; `eager` = stock SigLIP attention (`:93`, `:130`) |
| `dtype` | `float16` | torch dtype name | encoder forward dtype (`float32` for CPU debug) (`:89`, `:153`) |

### Qwen3-VL encoder (merger / deepstack)

Loads **Qwen3-VL-4B-Instruct**'s vision ViT, **no LLM** (`qwen3vl_encoder.py`).

**Load recipe** (`:76-90`): build **only** `Qwen3VLVisionModel(Qwen3VLConfig.from_pretrained(snap).vision_config)`, set `vcfg._attn_implementation = attn_impl` (`:78`), then load just the `model.visual.*` tensors from the shards (fp32) with the same strict no-missing/no-unexpected assert (`:89-90`). Real 4B vision config (read via `getattr`, `:92-97`): `hidden_size=1024`, `out_hidden_size=2560`, `depth=24`, `spatial_merge_size=2`, `patch_size=16`, `temporal_patch_size=2`, `deepstack_visual_indexes=[5,11,17]`.

**Temporal layout.** `num_temporal = frames_per_clip // temporal_patch_size` (`:103`, e.g. `8//2 = 4` grid_t positions); `tubelet_size = temporal_patch_size = 2` (`:102`) — two frames fold into each temporal token.

**Stages → layer mapping.** Raw per-block outputs (1024-d, **pre-merge**) are captured via **forward hooks** registered only for the blocks a stage needs (`_mk_hook:200`, registered `:114-115`; `_need_blocks` built at `:106-112`). The spatial **merger** halves each spatial dim (`÷merge² = 4`) and lifts `1024 → 2560`:

| stage | dim | tokens | source (`forward`, `:254`) |
|---|---|---|---|
| `block_<i>`, i∈0..23 | 1024 | grid_t·gh·gw | hook capture `self._captured[i]` (`:260`) |
| `before_merger` | 1024 | grid_t·gh·gw | `self._captured[depth-1]` (== `block_23`) (`:258`) |
| `after_merger` | 2560 | grid_t·(gh/2)·(gw/2) | `image_embeds` (fed to the LLM) (`:256`) |
| `deepstack_<i>`, i∈{5,11,17} | 2560 | grid_t·(gh/2)·(gw/2) | `deepstack[deepstack_idx.index(i)]` (`:263`) |
| `"all"` shorthand | — | — | expands to `[block_0 .. block_23]` **only** |

`self.visual(pixel_values, grid_thw)` returns `(image_embeds, deepstack)` (`:237`); the wrapper `torch.split`s each along dim 0 by per-video token counts — `after_sizes = grid_thw.prod(-1)//merge²` and `before_sizes = grid_thw.prod(-1)` (`:239-241`) — then `_stack`s to `(B, N, D)` (`:243-266`).

**Qwen-only knobs** (`wrapper_kwargs`):

| key | default | allowed | meaning |
|---|---|---|---|
| `pretrained` / `cache_dir` | — | path / repo-id + cache | weight location |
| `out_stages` | `["before_merger","after_merger"]` | stage grammar | stage spec (harness overrides with `analysis.stages`, `:54`) |
| `resize_mode` | `smart` | `smart` \| `fixed` | `smart` = processor `smart_resize` within `[min_pixels,max_pixels]`; `fixed` = pre-resize + `do_resize=False` |
| `min_pixels` / `max_pixels` | `8192` / `112896` | int | smart-resize token budget (lmms-eval defaults) (`:59-60`) |
| `qwen_fixed_h` / `qwen_fixed_w` | — | multiples of `patch·merge = 32` | **required** for `resize_mode=fixed` (asserted `:120-123`) |
| `attn_implementation` | `sdpa` | `sdpa` \| `eager` \| `flash_attention_2` | passed into `vcfg._attn_implementation` (`:78`) |
| `dtype` | `float16` | torch dtype name | forward dtype (`float32` for CPU) (`:55`, `:135`) |

Under `smart`, the `AutoVideoProcessor` is built with `min/max_pixels` **and** the attrs are force-set afterward (`:128-133`). Under `fixed`, `_preprocess` pre-resizes each clip with `F.interpolate` to `(fixed_h, fixed_w)` and passes `do_resize=False` (`:214-218`).

---

## Configuration

Both are config-driven under `eval_name: analysis_vlm`. Real templates live at `configs/analysis/llavavideo_analysis.yaml` and `configs/analysis/qwen3vl_analysis.yaml` (values below match those files; only `pretrain_kwargs`/`folder`/`tag` are elided).

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
    plot: true
    probes:
      - type: linear
        pooling: framewise_mean        # framewise -> keeps temporal order (VLM-only)
        pre_norm: true
        optimization: { lr: 0.01, weight_decay: 0.0, warmup: 1.0 }
  data:
    num_classes: 4
    frames_per_clip: 16                # -> num_temporal = 16
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
      deepstack: [5, 11, 17]                     # 2560-d injection points ("all"/true = all)
    plot: true
    probes:
      - { type: linear, pooling: mean, pre_norm: true, optimization: { lr: 0.01 } }
      - { type: attentive, num_heads: 16, num_probe_blocks: 1, optimization: { lr: 0.002 } }
  data:
    num_classes: 4
    frames_per_clip: 8                 # even; grid_t = 8/2 = 4
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

### Checkpoints / `cache_dir` summary

| backend | `checkpoint` (repo id) | `cache_dir` |
|---|---|---|
| LLaVA-Video | `lmms-lab/LLaVA-Video-7B-Qwen2` | `/data/dataset/LLaVA-Video-100K-Subset/` |
| Qwen3-VL | `Qwen/Qwen3-VL-4B-Instruct` | `/data/dataset/LLaVA-Video-100K-Subset/` |

Either `checkpoint` may instead be a **local snapshot dir** (then `cache_dir` is unused). Resolution is offline-first: cached weights are found via `snapshot_download(local_files_only=True)` and downloaded only on miss (`loadutil.py:49-57`).

---

## Invariants & gotchas

### Framewise cache path (`num_temporal`)

The `framewise` feature-cache mode is **VLM-only** and depends on the encoder exposing `num_temporal`:

- `reduce_feature(feat, "framewise", num_temporal)` (`cache.py:77`, framewise branch `:88-95`) reshapes `(B, N, D) → (B, T, S, D)` and spatial-means over `S` → **`(B, T, D)`** — keeps temporal order (unlike `pooled`'s global mean‖max) while staying small. It **requires** `N % num_temporal == 0`, else raises (`cache.py:92-93`).
- `eval.py:436` reads `enc_num_temporal = getattr(encoder, "num_temporal", None)` and passes it into `build_feature_cache` (`:442-449`). LLaVA sets it to `frames_per_clip`; Qwen to `frames_per_clip // temporal_patch_size`.
- The matching probe side is the `framewise_*` linear pooling (`eval.py:299`, `TemporalLinearProbe` at `:321-330`), which also needs `encoder.num_temporal` and raises if absent (`:323-325`).
- **V-JEPA has no `num_temporal`**, so `cache_pooling: framewise` and the `framewise_*` pooling are valid **only** for these two VLM backends. (V-JEPA already encodes time via RoPE.)

### LLaVA-specific

- **SDPA monkeypatch (default on):** `SigLipAttention.forward` is swapped for a fused `scaled_dot_product_attention` version (`_siglip_sdpa_forward:59`, patch loop `:130-138`) — same math (SDPA's default `1/√head_dim` scale == the eager `self.scale`), ~2-3× faster, avoids materializing the `(B,heads,729,729)` fp32 score matrix. Opt out with `attn_implementation: eager`.
- **Frame chunking (`:234-243`):** with `B·T > vision_chunk` the tower runs in frame-chunks and per-layer hidden states are concatenated back — required to bound the eager score-matrix memory (e.g. `8·16=128` frames); result is numerically identical.
- The projector is computed **only** when a `*projector*` stage is selected (`self._need_proj:151`, used `:246`).
- The strict key assert (`:124`) makes a **different LLaVA/SigLIP variant hard-fail** rather than silently mis-load.
- lmms-eval knobs `conv_template` / `video_decode_backend` / `mm_spatial_pool_mode` / `max_frames_num` / `force_sample` **do not** affect vision-only feature extraction (`:80-82`).

### Qwen-specific

- **Variable token count under `smart` (the key invariant):** different input resolutions → different `grid_thw` → different `N`, so `torch.stack` across videos fails. `forward` catches this and raises a directive to use `resize_mode: fixed` + `qwen_fixed_h/w` (`_stack:243-251`). The same failure surfaces later in the `tokens` cache concat (`cache.py:151-159`). **If your videos are already uniform resolution, `smart` yields a uniform grid and batches fine.**
- The `deepstack` list is validated against the real `deepstack_visual_indexes` (`[5,11,17]`) — anything else fails loud (legacy `:175-176`, dict `:193-194`).
- Hooks are registered **only** for needed blocks, and `self._captured` is reset each `forward` (`:236`) — no cross-batch leakage.
- The strict key assert (`:89-90`) hard-fails on a mismatched Qwen variant/config.

### Cross-cutting

- **`"all"` never includes toggles** — see the stage-grammar box above; the wrapper docstrings (`llava:18`, `qwen:16`) are wrong on this point.
- `*_pool2` (LLaVA) and `deepstack` / `before_merger` / `after_merger` (Qwen) are **off unless explicitly toggled**.
- SDPA is on by default but is a mathematically-neutral swap; `attn_implementation: eager` restores stock attention.
- Stages are returned in encoder-dtype (fp16) views into the hidden states — `_encode` keeps them fp16 to avoid doubling peak GPU memory on the all-layer scan (`eval.py:667-670`); the cache stores `.half()`.

---

## Cross-references

- [§02 `analysis_vlm` harness](02-analysis-vlm-harness.md) — the eval flow that selects these backends, builds one head per (stage × probe × variable), and reports the [stage × probe] matrix.
- [§03 Feature caching & pooling](03-feature-caching-and-pooling.md) — the `tokens` / `pooled` / `framewise` cache granularities and the `num_temporal` dependency these encoders satisfy.
- [§04 Probes, regression & NaN-masking](04-probes-regression-nanmask.md) — the probe heads (incl. `TemporalLinearProbe` / `TemporalAttentiveClassifier`) that consume `num_temporal`.
- [§05 `analysis` clip harness](05-analysis-clip-harness.md) — the third backend, the V-JEPA2 ViT (`data_mode='clip'`).
- [§07 Plotting](07-plotting.md) — the layer-wise `stage_val_acc.png` these stages feed.
- [§13 Config reference](13-configs-reference.md) — the full `analysis_vlm` YAML key-space and the `z_scripts` launcher / `--export CONFIG` pattern.

**Scope note — post-hoc analysis modes are a different backend.** The `experiment.analysis.modes` layer (`attention_distance`, and the roadmapped ablation / ortho-probe / steering / direction-tuning) operates on the **V-JEPA clip encoder** (`data_mode='clip'`), **not** these VLM (`data_mode='raw'`) wrappers, so it is intentionally out of scope here. The dispatch seam (`eval.py:565-590`, default-off when `experiment.analysis.modes` is absent) and `skip_base_probe` (`eval.py:504`) are shared plumbing but only fire for clip-mode configs. The one implemented mode, `attention_distance` (`evals/analysis_vlm/modes/attention_distance.py`), renders the paper's **Fig. 3 heatmap** as its primary plot (`attention_distance.png`; x=Layer 0-23, y=Head 0-15, colour=spatial distance in patches, `Blues_r` so local heads are dark, per-cell annotations) plus the Appendix **Fig. 19** dual-axis line plot (`attention_distance_layerwise.png`; layer-mean distance + head specialization vs layer fraction) — and it runs on **rank 0 only** (so multi-GPU gives no speedup; single-GPU, e.g. `z_scripts/run_attn_distance_vjepa.sh`, is the faithful setup). See [§11 Attention hooks](11-attention-hooks.md), [§12 Analysis modes & roadmap](12-analysis-modes.md), and [§14 Reproduction status & findings](14-reproduction-status-and-findings.md).
