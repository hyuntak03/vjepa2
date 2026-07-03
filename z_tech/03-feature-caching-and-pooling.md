# 03 — Feature caching & pooling

> Encode the frozen encoder **once** into an fp16 host-RAM cache, then train every probe for many epochs over the cache with no decode and no forward — turning the 24-stage all-layer linear scan from minutes-per-epoch into seconds-per-epoch.

## Purpose

The layer-wise probing harness (`evals/analysis_vlm`, see [02](02-analysis-vlm-harness.md)) freezes the encoder and trains a **separate** probe per (stage × probe-spec) for many epochs. Because the encoder is frozen **and** preprocessing is deterministic (no augmentation), every video's features are byte-identical every epoch — so re-encoding per epoch is pure waste.

`evals/analysis_vlm/cache.py` implements the **encode-once pre-pass**: run the encoder a single deterministic time, reduce + cache the per-stage features in host RAM (fp16), then train probes over the cache. Epochs collapse from minutes to seconds. This file is what makes the `vision_encoder: all` all-layer linear scan (24 V-JEPA-L blocks × N samples) tractable on a single GPU.

The cache is also the **hand-off surface** for the post-hoc analysis modes ([12](12-analysis-modes.md)): feature-space modes read this rank's cached tokens directly off `AnalysisContext`. Note the split — hook-based modes (`attention_distance`) deliberately **do not** use the cache; see [Two mode families](#two-mode-families-relative-to-the-cache) below.

## What changed vs upstream V-JEPA2

Upstream `204698b` has **no `evals/analysis_vlm/` at all** — its `evals/` tree is only `action_anticipation_frozen`, `image_classification_frozen`, `video_classification_frozen`, `hub`, `main.py`, `main_distributed.py`, `scaffold.py`. Everything below is **net-new fork code**; there is no upstream counterpart to diff against, so `git diff 204698b -- <path>` reports each file as *added in full* (e.g. `cache.py` = `214 insertions(+)`, 1 file changed).

| File | Status | Delta |
|---|---|---|
| `evals/analysis_vlm/cache.py` | **new (+214)** | The entire caching subsystem documented here: `build_feature_cache`, `reduce_feature`, `_ThreadPrefetcher`, `CachedTensorDataset`, `_cached_collate`, `make_cached_loader`, `PooledLinearProbe`. |
| `evals/analysis_vlm/eval.py` | **new** | Driver: reads `optimization.cache_features / cache_pooling / cache_max_gb` (`eval.py:227-229`), wires the sequential pre-pass (`eval.py:426-455`), selects `PooledLinearProbe` under `cache_pooling='pooled'` (`eval.py:302-320`), runs the cached `_encode` branch (`eval.py:651-655`), and hands the cache to modes via `AnalysisContext` (`eval.py:565-590`). |
| `evals/analysis_vlm/probes.py` | **new** | Sibling probe heads (`TemporalLinearProbe`, `TemporalAttentiveClassifier`) that also take the `pre_norm` LayerNorm (`probes.py:84-90`). |
| `evals/analysis_vlm/modes/__init__.py` | **new** | `AnalysisContext` dataclass carries the cache to modes; `run_modes` dispatch. See [12](12-analysis-modes.md). |

**Default-off guarantee.** `cache_features` defaults to `false` (`eval.py:227`) → the harness runs the ordinary per-epoch encode path and none of `cache.py` is exercised. Caching is strictly opt-in.

## Design & data flow

```
data loader (workers=0, shuffle=False shard)
        │  decode (decord, releases GIL)
        ▼  _ThreadPrefetcher (daemon thread, depth=2)  ── overlaps ──┐
build_feature_cache: encode_fn(batch) → feats list[(B,N,D)]         │ GPU encode
        │  reduce_feature(f, cache_pooling) → .half().cpu()          │
        ▼  concat per stage                                         ─┘
feats_cat: list[stage] of (n_local, …) fp16 CPU  +  labels (n_local,) long CPU
        │
        ├─▶ CachedTensorDataset → make_cached_loader (shuffle=training, workers=0)
        │        └─ train probes many epochs (run_mode='cached', no decode, no encoder)
        └─▶ AnalysisContext.{tr,va}_feats/labels ─▶ feature-space modes (cache_pooling: tokens)
```

### `build_feature_cache` — the encode-once pre-pass

`cache.py:99`. One `@torch.no_grad()` deterministic pass: encode every sample, reduce, accumulate on CPU as fp16.

- **fp16 on CPU.** Each reduced stage tensor is `.half().cpu()` before accumulation (`cache.py:131`) — the cache lives in host RAM, halving footprint vs fp32. Training re-floats to fp32 on device (see [cached `_encode` branch](#the-cached-_encode-branch)).
- **Per-rank shard.** Under DDP each rank pre-passes *its own fixed shard* (`shuffle=False DistributedSampler`; the pre-pass loaders are built `training=False`, `eval.py:441/447`), caches it, then shuffles *locally* at train time. Metrics all-reduce as usual; there is no cross-rank cache exchange.
- **Sequential split builds.** `eval.py` builds the train pre-pass loader, caches, `del`s it, *then* builds val — with `persistent=False, workers=0` — to avoid spawn-multiprocessing worker pile-up/deadlock at the train→val transition (`eval.py:426-451`).
- **Up-front size guard.** Estimates full-cache size after batch 0 and aborts before OOM (`cache.py:134-144`; see [Memory tradeoffs](#memory-tradeoffs--cache_max_gb)).
- **N-mismatch guard.** `torch.cat` over stages fails if token count `N` differs across batches (mixed resolutions/lengths under `tokens`); caught and re-raised with a fix hint (`cache.py:151-159`).
- **Built-size log.** On success it logs the realized cache: `"[<label>] feature cache built: {n} samples x {stages} stages ({pooling}) -> {MB} MB RAM (this rank)"` (`cache.py:161-163`).

### Cache granularities — `reduce_feature`

`cache.py:77`. Input is `feat: (B, N, D)`; output depends on `cache_pooling`:

| `cache_pooling` | Output shape | What it stores | Probe types | Size |
|---|---|---|---|---|
| `tokens` | `(B, N, D)` | full token set (`cache.py:85`) | **all** (incl. attentive) + feature-space modes | large — scales with `N × #stages` |
| `pooled` | `(B, 2D)` | `[mean ‖ max]` over tokens (`cache.py:87`) | **linear only** | tiny |
| `framewise` | `(B, T, D)` | spatial-mean per frame (`cache.py:88-95`) | linear / framewise / attentive over the `T` vectors | small |

Notes / invariants:

- **`pooled` collapses time.** Mean/max over *all* `N` tokens erases the temporal token structure. Fine for `speed` / `accel_mag` but **degrades `direction`** decodability — feature-space token modes therefore require `cache_pooling: tokens`.
- **`framewise` is VLM-only.** It needs `encoder.num_temporal`; V-JEPA does not expose it (time is folded into RoPE), so `framewise` raises `ValueError` on V-JEPA (`cache.py:89-90`). It also requires `N % num_temporal == 0` (`cache.py:92-93`).
- `pooled` **always stores 2D = `[mean‖max]`** regardless of the probe's `pooling` setting — the probe slices out `mean`, `max`, or both at forward time (see [`PooledLinearProbe`](#pooledlinearprobe--the-pre_norm-layernorm)).

### `_ThreadPrefetcher`

`cache.py:34`. A background **thread** (not subprocess) prefetch over a `num_workers=0` loader. A daemon thread decodes the next batch into a small `queue.Queue(maxsize=depth)` (default `depth=2`) while the main thread runs the GPU encode. decord releases the GIL during decode, so decode overlaps encode without any worker subprocesses.

Why a thread and not `num_workers>0`: the pre-pass hits the train→val loader transition under spawn multiprocessing, where DataLoader worker respawn deadlocks. A daemon thread sidesteps that entirely (`cache.py:34-39`, wired at `eval.py:441/447` with `workers=0`). Decode exceptions are captured and re-raised in the consuming thread (`cache.py:57-58, 71-72`); `len()` is proxied so tqdm shows a progress bar (`cache.py:62-65, 123-124`).

### `CachedTensorDataset` / `make_cached_loader`

`cache.py:167`. A trivial in-RAM dataset over the already-sharded cache: `__getitem__` returns `([f[i] for f in self.feats], int(self.labels[i]))` (`cache.py:175-176`). `make_cached_loader` (`cache.py:186`) wraps it in a plain `DataLoader(shuffle=training, num_workers=0, collate_fn=_cached_collate)` where `_cached_collate` (`cache.py:179`) stacks per-stage tensors. Because the cache is *already this rank's shard*, no `DistributedSampler` is used — shuffling is purely local.

### `PooledLinearProbe` + the `pre_norm` LayerNorm

`cache.py:193`. Linear probe over a pre-pooled cached vector `x = (B, 2D) = [mean‖max]`.

- The probe **slices the requested view** (`mean` / `max` / `meanmax`) out of the always-2D cache at forward time, so `in_dim` is `D` or `2D` (`cache.py:204-214`).
- **`pre_norm` applies `nn.LayerNorm` over the feature dim, per sample** (`cache.py:201`). LayerNorm normalizes each sample's vector across `D` (→ mean 0, unit variance *per sample, over the feature dimension*) then applies a learned affine (γ, β). It is **not** batch/per-feature statistics — each row is standardized independently.

**Why per-sample-over-D normalization is required (not cosmetic).** V-JEPA activation magnitudes differ by **orders of magnitude across layers**, so a single shared learning rate cannot fit every layer's raw pooled features — deep-layer probes explode while early-layer probes barely move. LayerNorm rescales every stage's input to a common scale, letting one `lr` fit all 24 stages in the all-layer scan. A competitor fork set `pre_norm=False` (raw pooled features) with a fixed `lr` and got poor linear-probe accuracy for exactly this reason.

**Valid choices:** LayerNorm (per-sample over `D`, what this code does) **or** per-feature `StandardScaler`. **"No normalization" is the only wrong choice.** The sibling `TemporalLinearProbe` (`probes.py:84-90`) takes the same `pre_norm` LayerNorm.

### sklearn StandardScaler + LogisticRegression solver path — NOT present

The `StandardScaler` alternative above is **conceptual**, not code in this fork. There is no sklearn path: a full-tree grep for `sklearn`, `StandardScaler`, `LogisticRegression`, `fit_sklearn_linear_probe`, `apply_fitted_linear_probe` returns nothing. All probes are torch modules trained by the harness's fused-AdamW loop ([04](04-probes-regression-nanmask.md)); the per-feature-`StandardScaler` + `LogisticRegression`-solver idiom is not wired in. (If a sklearn solver path is later added, update this subsection.)

### The cached `_encode` branch

`eval.py:649`. When `cache_features` is on, the pre-pass sets `run_mode = "cached"` (`eval.py:455`) and the train loop is driven by `_encode(..., data_mode="cached", ...)`. The cached branch **makes no encoder call** — it re-floats the fp16 cache to fp32 on device (`eval.py:651-655`):

```python
if data_mode == "cached":                                        # eval.py:651
    feats = [f.to(device, non_blocking=True).float() for f in data[0]]   # :653
    labels = data[1].to(device, non_blocking=True)               # :654
    return feats, labels, labels.size(0)
```

(`data_mode == "cached"` is the runtime `run_mode`; the config-level `data_mode` — `clip`/`raw`/VLM — is preserved separately and still recorded in `summary.json` at `eval.py:544`.)

### Two mode families relative to the cache

The post-hoc modes ([12](12-analysis-modes.md)) split into two families with **opposite** cache requirements:

| Family | Example modes | Cache setting | How they read features |
|---|---|---|---|
| **Hook-based fresh-forward** | `attention_distance` (done), `attention_ablation` (pending) | `cache_features: false` | Re-run the encoder under `attention_hooks` / masks on `ctx.make_val_clip_loader()` + `ctx.encode_clip`; **never call `build_feature_cache`.** |
| **Feature-space token** | `orthogonal_probe_sequence`, `steering`, `direction_tuning` (all pending) | `cache_features: true, cache_pooling: tokens` | Read this rank's cached tokens off `AnalysisContext.{tr,va}_feats/labels`. |

`attention_distance` is **encoder-only**: it sets `skip_base_probe: true` (`num_probe_epochs=0`, `eval.py:501-507`) and `cache_features: false`, and captures per-head distance on *fresh* forwards via the SDPA `attention_hooks` patch ([11](11-attention-hooks.md)) — the output stays bit-identical (capture is a detached side computation). Because it changes/reads nothing through the cache, it **bypasses `build_feature_cache` entirely**. The token-cache consumers are the *still-pending* feature-space modes (`orthogonal_probe_sequence` / `steering` / `direction_tuning`), per `modes/REPRODUCTION_PLAN.md` §2c/2d/2e. Any mode that **changes** features per setting (attention ablation) must likewise set `cache_features: false`, else it would reuse the unmasked baseline cache (REPRODUCTION_PLAN Risk 1).

### Cache hand-off to modes via `AnalysisContext`

`modes/__init__.py:39-73`. On rank 0 only, after the base probe loop, `eval.py:565-590` builds an `AnalysisContext` and calls `run_modes`. The cache is surfaced through six read-only fields — `cache_pooling`, `data_mode`, `tr_feats`, `tr_labels`, `va_feats`, `va_labels` — populated **only when `cache_features` is true** (else `None`, `eval.py:580-584`). Feature-space modes consume `{tr,va}_feats` (this rank's fp16 token cache) plus the two closures `encode_clip` / `make_val_clip_loader`. This is the single point at which the token cache leaves the caching subsystem.

## Key code

`build_feature_cache` per-batch loop and reduce (`cache.py:99, 129-131`):

```python
@torch.no_grad()
def build_feature_cache(encode_fn, loader, cache_pooling, num_temporal=None, max_gb=None,
                        label="cache", rank=0):
    ...
    for data in iterator:
        feats, labels, bsz = encode_fn(data)             # (list[(B,N,D)], (B,), bsz)
        reduced = [reduce_feature(f, cache_pooling, num_temporal).half().cpu() for f in feats]
```

Granularity reducer (`cache.py:84-95`):

```python
if mode == "tokens":                                          # (B, N, D)  — all probes + modes
    return feat
if mode == "pooled":                                          # (B, 2D) = [mean ‖ max]
    return torch.cat([feat.mean(dim=1), feat.max(dim=1).values], dim=-1)
if mode == "framewise":                                       # (B, T, D) — VLM only
    ...
    return feat.view(b, num_temporal, s, d).mean(dim=2)
```

Up-front RAM guard (`cache.py:134-144`):

```python
per_sample_mb = sum(r[:1].element_size() * r[:1].nelement() for r in reduced) / 1024.0**2
est_gb = per_sample_mb * n_target / 1024.0
if max_gb and est_gb > max_gb:
    raise RuntimeError(f"[{label}] estimated feature cache {est_gb:.0f} GB exceeds cache_max_gb={max_gb} ...")
```

`PooledLinearProbe` view-slice + norm (`cache.py:196-214`):

```python
def __init__(self, embed_dim, num_classes, pooling="mean", pre_norm=True):
    in_dim = embed_dim * (2 if pooling == "meanmax" else 1)
    self.norm   = nn.LayerNorm(in_dim) if pre_norm else nn.Identity()   # per-sample over D
    self.linear = nn.Linear(in_dim, num_classes, bias=True)

def forward(self, x):                         # x: (B, 2D) = [mean ‖ max]
    if   self.pooling == "mean":    z = x[..., :d]
    elif self.pooling == "max":     z = x[..., d:2*d]
    elif self.pooling == "meanmax": z = x[..., :2*d]
    return self.linear(self.norm(z))
```

`pooled` ⇒ linear-only guard + probe substitution (`eval.py:302-320`):

```python
if cache_features and cache_pooling == "pooled":
    if ptype != "linear":
        raise ValueError("cache_pooling='pooled' caches only pooled vectors -> linear probes only. ...")
    ...
    return PooledLinearProbe(embed_dim=embed_dims[stage_pos], num_classes=out_dim,
                             pooling=pooling, pre_norm=spec.get("pre_norm", True))
```

## Configuration

Config-driven via `experiment.optimization` (parsed at `eval.py:227-229`):

| key | meaning | default | allowed values |
|---|---|---|---|
| `cache_features` | master switch; `false` → normal per-epoch encode path | `false` | `true` \| `false` |
| `cache_pooling` | cache granularity | `tokens` | `pooled` \| `tokens` \| `framewise` |
| `cache_max_gb` | abort if estimated **per-rank** cache RAM exceeds this | `64` | any positive number |

Real example — the Blender toy Fig-2c all-layer linear scan (`configs/analysis/blender_toy_dataset/vjepa_combined.yaml`):

```yaml
experiment:
  analysis:
    stages:
      vision_encoder: all        # scan every V-JEPA-L block (24 stages)
    probes:
      - type: linear
        pooling: mean            # V-JEPA is spatio-temporal → global mean OK
        pre_norm: true           # REQUIRED (per-sample LayerNorm over D)
        optimization: { lr: 0.001, weight_decay: 0.1, warmup: 2.0 }
  data:
    frame_step: 1                # Blender clips are natively 16f
    uniform_sampling: true       # evenly sample fpc frames over the whole clip
    frames_per_clip: 16
  optimization:
    batch_size: 8
    num_epochs: 40
    cache_features: true
    cache_pooling: pooled        # tiny 2D cache → fast all-layer scan
    cache_max_gb: 80
```

Shipping configs in `configs/analysis/blender_toy_dataset/` are `vjepa_combined.yaml`, `llavavideo_combined.yaml`, `qwen3vl_combined.yaml` (all `cache_pooling: pooled` linear scans) and `vjepa_attn_distance.yaml` (encoder-only, `cache_features: false`). **No `cache_pooling: tokens` config ships yet** — the token-cache configs for the per-neuron / orthogonal-probe / steering / direction-tuning modes are Phases 2–4, still pending (blueprints in `modes/REPRODUCTION_PLAN.md` §2c/2d/2e). When they land they will set `cache_pooling: tokens` + a higher `cache_max_gb`.

## Invariants & gotchas

- **Default OFF.** `cache_features` defaults `false` (`eval.py:227`); the ordinary per-epoch encode path runs and `cache.py` is untouched.
- **`pooled` ⇒ linear only.** `eval.py:302-313` raises if a non-linear probe (or a `framewise`-pooling probe) is requested under `cache_pooling='pooled'` — the token structure it needs is already collapsed. `PooledLinearProbe` is substituted only when `cache_features and cache_pooling=='pooled'` (`eval.py:316-320`).
- **`tokens` needs uniform token count.** Mixed video resolution/length ⇒ varying `N` ⇒ `torch.cat` fails; the error names the fixes: `cache_pooling='pooled'`, Qwen `resize_mode='fixed'` + `qwen_fixed_h/w`, or uniform inputs (`cache.py:151-159`).
- **`framewise` is VLM-only.** V-JEPA has no `num_temporal` → `ValueError` (`cache.py:89-90`); needs `N % num_temporal == 0` (`cache.py:92-93`).
- **fp16 cache, fp32 train.** Cache is stored `.half()` (`cache.py:131`); the cached `_encode` branch re-floats to fp32 on device (`eval.py:653`). Expect fp16 rounding vs a live fp32 forward.
- **Size guard fires after batch 0.** The `cache_max_gb` abort needs one batch to measure `per_sample_mb`, so it triggers early but not before the first encode (`cache.py:132-144`). `max_gb` is **per-rank** host RAM (default `64`, `eval.py:229`); escape hatches in the message: reduce stages/frames, shard across more GPUs, use `cache_pooling='pooled'`, or raise `cache_max_gb`.
- **`num_workers=0` in the pre-pass is deliberate** — subprocess workers deadlock at the train→val transition under spawn; overlap comes from `_ThreadPrefetcher` instead (`eval.py:439-440/441/447`).
- **Hook-based modes must NOT cache.** Any mode that changes features per setting (attention ablation) or captures on fresh forwards (attention distance) sets `cache_features: false` and bypasses `build_feature_cache` — otherwise it reuses the unmasked baseline cache (REPRODUCTION_PLAN Risk 1).
- **Cache reaches modes only on rank 0, only when on.** `AnalysisContext.{tr,va}_feats/labels` and `cache_pooling`/`data_mode` are populated only if `cache_features` (else `None`); the modes dispatch runs on rank 0 (`eval.py:569`).

## Memory tradeoffs & `cache_max_gb`

Per-rank host-RAM cost ≈ `n_samples × #stages × (per-sample tensor bytes, fp16)`:

| pooling | per-sample, per-stage | typical use |
|---|---|---|
| `pooled` | `2D` fp16 (e.g. `2·1024·2 B ≈ 4 KB`) | all-layer linear scan — tiny even at 24 stages |
| `framewise` | `T·D` fp16 | temporal-preserving, still small |
| `tokens` | `N·D` fp16 (N in the hundreds–thousands) | attentive / feature-space token modes — large |

`build_feature_cache` estimates full-cache size after batch 0 and aborts *up front* if it exceeds `max_gb`, so a `tokens` selection with many stages/tokens cannot silently OOM (`cache.py:134-144`).

## Key findings

1. **Fig-2c reproduction.** On the paper-faithful Blender toy dataset (fixed red sphere, r=0.3 m, overhead cam, 16f@24fps) the frozen V-JEPA2-L layer-wise R² reproduces the dissociation: **SPEED** decodable early (R² ≈ 0.68 at L0); **DIRECTION** emerges sharply in the Physics Emergence Zone (R² ≈ 0.28 at L0 → ≈ 0.9 by layer-fraction 0.3–0.4); **accel_mag** in between. Uses the `pooled` cache. Full write-up in [14](14-reproduction-status-and-findings.md).
2. **`frame_step` reproduction bug.** `frame_step=1` on a 64-frame clip samples 16 contiguous frames (first ¼ trajectory) → only sub-patch motion → L0 cannot encode speed/accel. Fix: `frame_step=4` **or** `uniform_sampling: true`. Blender clips are natively 16f, so configs use `frame_step=1 + uniform_sampling=true`. Details in [06](06-data-pipeline-changes.md) / [14](14-reproduction-status-and-findings.md).
3. **`pre_norm` LayerNorm is REQUIRED** (per-sample over `D`, `cache.py:201`); one shared `lr` cannot fit 24 layers whose activation scale differs by orders of magnitude. Valid alternatives: LayerNorm **or** per-feature `StandardScaler`; "no normalization" is the only wrong choice. No sklearn path is implemented.
4. **`cache_pooling` semantics.** `pooled` = `[mean‖max]` (2D, collapses time, **degrades direction**), `tokens` = full `(N,D)`, `framewise` = `(T,D)` (VLM-only). Feature-space token modes (per-neuron, orthogonal-probe, steering) require `cache_pooling: tokens`; hook-based modes (attention distance/ablation) require `cache_features: false`.
5. **Reproduction roadmap** (`modes/REPRODUCTION_PLAN.md`): additive, config-driven modes for attention distance (C.6), attention ablation (C.6 Table 4), orthogonal probe sequence (C.11) + steering (C.12), circular direction geometry (C.7/C.10). **Phase 0 (dispatch scaffold — `run_modes`/`AnalysisContext` in `modes/__init__.py` + the additive dispatch block & `skip_base_probe` in `eval.py`, default-off) and Phase 1 (`attention_distance`) are DONE; Phases 2–5 pending.**

   **Phase-1 `attention_distance` — as-built.** It is an **encoder-only, hook-based** mode (`skip_base_probe: true`, `cache_features: false`) that captures per-(layer,head) attention-weighted spatial/temporal distance via the SDPA `attention_hooks` patch on *fresh* forwards — it never touches the cache. Its **primary** output is the paper's **Figure 3 heatmap** `attention_distance.png` (`modes/attention_distance.py:76`, `_plot_heatmap`): x = Layer (0–23), y = Attention Head (0–15), colour = spatial distance in patches, `cmap="Blues_r"` (LOW distance = DARK blue so unusually-local heads stand out), per-cell value annotations, colorbar `"Distance (patches)"`, title `"V-JEPA v2-L: Attention Distance Per Head"`. It **also** writes the Appendix **Fig-19** dual-axis line plot `attention_distance_layerwise.png` (`:115`, `_plot_layerwise`): layer-mean distance `Dbar = mean over heads` (red) + **head specialization** `S = std over the 16 heads` (blue dashed, = attention-head diversity, spikes at the PEZ) vs layer fraction, with PEZ shading. New cfg knobs: `query_chunk` (memory knob, streams queries so the `(B,H,N,N)` matrix is never materialized — **result-invariant**, default 512), `max_batches` (# val batches averaged, default 8), `annotate` (per-cell numbers, default True). **RUN** on the Blender velocity set (single GPU, 10 val batches, `resolution: 224` → 14×14×8 = 1568 tokens) and **reproduced Fig 3**: unusually-local low-distance (dark) heads cluster in the **middle layers ~L5–13 (Physics Emergence Zone)** — measured per-layer mean spatial distance dips to ≈ 3.7–5.2 patches at L5–13 vs ≈ 7 at the edges — while early/late layers are uniformly long-range. Output under `configs/analysis/blender_toy_dataset/logs/analysis_vlm/vjepa-blender-attn_distance/attention_distance/` (`attention_distance.{json,png}` + `attention_distance_layerwise.png`; JSON = `{spatial_distance:[24][16], temporal_distance:[24][16], num_layers, num_heads, rows_per_layer, n_batches}`). Launcher `z_scripts/run_attn_distance_vjepa.sh` is **single-GPU on purpose**: the modes dispatch runs on **rank 0 only** (`eval.py:569`), so multi-GPU gives **no** speedup for post-hoc modes — multi-GPU only helps the base probing sweep / the per-rank feature cache. Full mode docs in [12](12-analysis-modes.md); hook mechanics in [11](11-attention-hooks.md).

## Cross-references

- [02 — `analysis_vlm` harness (eval flow)](02-analysis-vlm-harness.md) — the driver that wires this cache and the probe loop.
- [04 — Probes, regression task & NaN-masking](04-probes-regression-nanmask.md) — the probe heads (`PooledLinearProbe`, `TemporalLinearProbe`) trained over the cache; fused-AdamW loop.
- [06 — Data pipeline changes](06-data-pipeline-changes.md) — `frame_step` / `uniform_sampling` sampling that makes the pre-pass deterministic.
- [11 — Attention hooks (distance + ablation)](11-attention-hooks.md) — the SDPA patch the hook-based modes use instead of the cache.
- [12 — Analysis modes subpackage & reproduction roadmap](12-analysis-modes.md) — `AnalysisContext` cache hand-off, `attention_distance`, and the pending token-cache modes.
- [14 — Reproduction status & findings](14-reproduction-status-and-findings.md) — Fig-2c / `frame_step` findings in full.
