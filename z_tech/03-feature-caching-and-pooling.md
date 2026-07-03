# Feature caching & pooling

## Purpose

The layer-wise probing harness (`evals/analysis_vlm`) freezes the encoder and trains a
*separate* probe per (stage × probe-spec) for many epochs. Because the encoder is frozen
**and** preprocessing is deterministic (no augmentation), every video's features are
byte-identical every epoch — so re-encoding per epoch is pure waste.

`evals/analysis_vlm/cache.py` implements the **encode-once pre-pass**: run the encoder a
single deterministic time, reduce+cache the per-stage features in host RAM (fp16), then train
probes over the cache with no decode and no encoder forward. Epochs collapse from
minutes to seconds. This file is what makes the `vision_encoder: all` all-layer linear scan
(24 stages × N samples) tractable on a single GPU.

## What changed vs upstream V-JEPA2

Upstream `204698b` has **no `evals/analysis_vlm/` at all** — its `evals/` tree is only
`action_anticipation_frozen`, `image_classification_frozen`, `video_classification_frozen`,
`hub`, `main.py`, `main_distributed.py`, `scaffold.py`. Everything below is **net-new fork code**.

| File | Status | Delta |
|---|---|---|
| `evals/analysis_vlm/cache.py` | **new** | The entire caching subsystem documented here: `build_feature_cache`, `reduce_feature`, `_ThreadPrefetcher`, `CachedTensorDataset`, `make_cached_loader`, `PooledLinearProbe`. |
| `evals/analysis_vlm/eval.py` | **new** | Driver that reads `optimization.cache_features / cache_pooling / cache_max_gb`, wires the pre-pass, and selects `PooledLinearProbe` when pooling is precomputed. |
| `evals/analysis_vlm/probes.py` | **new** | Sibling probe heads (`TemporalLinearProbe`, `TemporalAttentiveClassifier`) that also take `pre_norm`. |

There is **no upstream counterpart** to diff against; `git diff 204698b -- evals/analysis_vlm/cache.py`
reports the file as added in full.

## Pipeline overview

```
data loader (workers=0, shuffle=False shard)
        │  decode (decord, releases GIL)
        ▼  _ThreadPrefetcher (daemon thread, depth=2)  ── overlaps ──┐
build_feature_cache: encode_fn(batch) → feats list[(B,N,D)]         │ GPU encode
        │  reduce_feature(f, cache_pooling) → .half().cpu()          │
        ▼  concat per stage                                         ─┘
feats_cat: list[stage] of (n_local, …) fp16 CPU  +  labels (n_local,) long CPU
        │
        ▼  CachedTensorDataset → make_cached_loader (shuffle=training, workers=0)
train probes many epochs (no decode, no encoder)
```

## `build_feature_cache` — encode-once pre-pass

`cache.py:99`

```python
@torch.no_grad()
def build_feature_cache(encode_fn, loader, cache_pooling, num_temporal=None, max_gb=None,
                        label="cache", rank=0):
    # encode_fn(data) -> (feats: list[(B,N,D)] per stage, labels: (B,), bsz)
    # returns: (feats_cat: list[stage] of (n_local, ...) fp16 CPU, labels: (n_local,) long CPU)
```

Per-batch loop (`cache.py:129`):

```python
for data in iterator:
    feats, labels, bsz = encode_fn(data)
    reduced = [reduce_feature(f, cache_pooling, num_temporal).half().cpu() for f in feats]
    ...
    labels_acc.append(labels.cpu())
```

Key properties:

- **fp16 on CPU.** Each reduced stage tensor is `.half().cpu()` before accumulation — the cache lives in host RAM, halving footprint vs fp32. `_encode` re-floats to fp32 on device at train time (`eval.py:649`, cached branch: `f.to(device).float()`).
- **Per-rank shard.** Under DDP each rank pre-passes *its own fixed shard* (`shuffle=False` `DistributedSampler`, i.e. the loaders are built `training=False`, `eval.py:441/447`), caches it, and later shuffles *locally*. Metrics all-reduce as usual. No cross-rank cache exchange.
- **Sequential split builds.** `eval.py` builds the train pre-pass loader, caches, `del`s it, *then* builds val — with `persistent_workers=False, num_workers=0` — to avoid spawn-multiprocessing worker pile-up/deadlock at the train→val transition (`eval.py:426-451`).
- **N-mismatch guard.** `torch.cat` over stages can fail if token count `N` differs across batches (mixed resolutions/lengths under `tokens`); it is caught and re-raised with a fix hint (`cache.py:151-159`).

## Cache granularities — `reduce_feature`

`cache.py:77`. Input is `feat: (B, N, D)`; output depends on `cache_pooling`:

| `cache_pooling` | Output shape | What it stores | Probe types | Size |
|---|---|---|---|---|
| `tokens` | `(B, N, D)` | full token set (`cache.py:85`) | **all** (incl. attentive) | large — scales with `N × #stages` |
| `pooled` | `(B, 2D)` | `[mean ‖ max]` over tokens (`cache.py:87`) | **linear only** | tiny |
| `framewise` | `(B, T, D)` | spatial-mean per frame (`cache.py:88-95`) | linear / framewise / attentive over the `T` vectors | small |

```python
if mode == "pooled":
    return torch.cat([feat.mean(dim=1), feat.max(dim=1).values], dim=-1)   # (B, 2D)
...
if mode == "framewise":
    ...
    return feat.view(b, num_temporal, s, d).mean(dim=2)   # (B, T, D)
```

Notes / invariants:

- **`pooled` collapses time.** Mean/max over *all* `N` tokens erases the temporal token structure. This is fine for `speed`/`accel_mag` but **degrades `direction`** decodability — token-level modes (per-neuron, attention geometry) therefore require `cache_pooling: tokens`.
- **`framewise` is VLM-only.** It needs `encoder.num_temporal`; V-JEPA does not expose it (time is folded into RoPE), so `framewise` raises `ValueError` on V-JEPA (`cache.py:89-90`). It also requires `N % num_temporal == 0` (`cache.py:92-93`).
- `pooled` **always stores 2D = `[mean‖max]`** regardless of the probe's `pooling` setting — the probe slices out `mean`, `max`, or both at forward time (see `PooledLinearProbe` below).

## `_ThreadPrefetcher`

`cache.py:34`. A background **thread** (not subprocess) prefetch over a `num_workers=0` loader.
A daemon thread decodes the next batch into a small `queue.Queue(maxsize=depth)` (default
`depth=2`) while the main thread runs the GPU encode. decord releases the GIL during decode, so
decode overlaps encode without any worker subprocesses.

Why a thread and not `num_workers>0`: the pre-pass hits the train→val loader transition under
spawn multiprocessing, where DataLoader worker respawn deadlocks. A daemon thread sidesteps that
entirely (`cache.py:34-39`, `eval.py:439-440`). Decode exceptions are captured and re-raised in the
consuming thread (`cache.py:57-58, 71-72`); `len()` is proxied so tqdm shows a progress bar (`cache.py:62-65`).

## `CachedTensorDataset` / loader

`cache.py:167`. Trivial in-RAM dataset over the already-sharded cache:

```python
def __getitem__(self, i):
    return [f[i] for f in self.feats], int(self.labels[i])
```

`make_cached_loader` (`cache.py:186`) wraps it in a plain `DataLoader(shuffle=training,
num_workers=0)` with `_cached_collate` that stacks per-stage tensors. Because the cache is
*already this rank's shard*, no `DistributedSampler` is used — shuffling is purely local.

## `PooledLinearProbe` and the `pre_norm` LayerNorm

`cache.py:193`. Linear probe over a pre-pooled cached vector `x = (B, 2D) = [mean‖max]`.

```python
def __init__(self, embed_dim, num_classes, pooling="mean", pre_norm=True):
    ...
    in_dim = embed_dim * (2 if pooling == "meanmax" else 1)
    self.norm   = nn.LayerNorm(in_dim) if pre_norm else nn.Identity()
    self.linear = nn.Linear(in_dim, num_classes, bias=True)

def forward(self, x):                # x: (B, 2D)
    if   self.pooling == "mean":    z = x[..., :d]
    elif self.pooling == "max":     z = x[..., d:2*d]
    elif self.pooling == "meanmax": z = x[..., :2*d]
    return self.linear(self.norm(z))
```

- The probe slices the requested view (`mean` / `max` / `meanmax`) out of the always-2D cache, so `in_dim` is `D` or `2D`.
- **`pre_norm` applies `nn.LayerNorm` over the feature dim, per sample.** LayerNorm normalizes each sample's vector across `D` (→ mean 0, unit variance *per sample, over the feature dimension*) and then applies a learned affine (γ, β). It is **not** batch/per-feature statistics — each row is standardized independently.

### Why per-sample-over-D normalization matters (finding #3)

`pre_norm=True` is **required**, not cosmetic. V-JEPA activation magnitudes differ by **orders
of magnitude across layers**, so a single shared learning rate cannot fit every layer's raw
pooled features — deep-layer probes explode while early-layer probes barely move. LayerNorm
rescales every stage's input to a common scale, letting one `lr` fit all 24 stages in the
all-layer scan.

A competitor fork set `pre_norm=False` (raw pooled features) with a fixed `lr` and got poor
linear-probe accuracy for exactly this reason. **Valid choices:** LayerNorm (per-sample over `D`,
what this code does) **or** per-feature `StandardScaler`. **"No normalization" is the only wrong
choice.** The sibling `TemporalLinearProbe` (`probes.py:84-89`) takes the same `pre_norm`
LayerNorm.

## sklearn StandardScaler + LogisticRegression path — NOT present

The task asked about `fit_sklearn_linear_probe` / `apply_fitted_linear_probe`. **These do not
exist** anywhere in the repo — a full-tree grep for `sklearn`, `StandardScaler`,
`LogisticRegression`, `fit_sklearn_linear_probe`, `apply_fitted_linear_probe` returns nothing.
All probes are torch modules trained by the harness's fused-AdamW loop. The per-feature
`StandardScaler` mentioned in finding #3 is a *conceptually valid alternative* to LayerNorm, not
code implemented here. (If a sklearn solver path is later added, this section should be updated.)

## Memory tradeoffs & `cache_max_gb` guard

Per-rank host-RAM cost ≈ `n_samples × #stages × (per-sample tensor bytes, fp16)`:

| pooling | per-sample, per-stage | typical use |
|---|---|---|
| `pooled` | `2D` fp16 (e.g. `2·1024·2 B ≈ 4 KB`) | all-layer linear scan — tiny even at 24 stages |
| `framewise` | `T·D` fp16 | temporal-preserving, still small |
| `tokens` | `N·D` fp16 (N in the hundreds–thousands) | attentive/token-level — large |

`build_feature_cache` **estimates the full-cache size after the first batch** and aborts *up front*
if it exceeds `max_gb`, so a `tokens` selection with many stages/tokens cannot silently OOM
(`cache.py:134-144`):

```python
est_gb = per_sample_mb * n_target / 1024.0
if max_gb and est_gb > max_gb:
    raise RuntimeError(f"[{label}] estimated feature cache {est_gb:.0f} GB exceeds cache_max_gb={max_gb} ...")
```

The error message lists the escape hatches: reduce stages/frames, shard across more GPUs, switch to
`cache_pooling='pooled'` (linear only), or raise `optimization.cache_max_gb`. `max_gb` is
**per-rank** host RAM (default `64`, `eval.py:229`).

## Config

Config-driven via `experiment.optimization` (parsed at `eval.py:227-229`):

| key | default | meaning |
|---|---|---|
| `cache_features` | `false` | master switch; `false` → normal per-epoch encode path |
| `cache_pooling` | `tokens` | `pooled` \| `tokens` \| `framewise` |
| `cache_max_gb` | `64` | abort if estimated per-rank cache RAM exceeds this |

Real example — the Blender toy Fig-2c all-layer linear scan
(`configs/analysis/blender_toy_dataset/vjepa_combined.yaml`):

```yaml
experiment:
  analysis:
    stages:
      vision_encoder: all        # scan every V-JEPA-L block
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

Token-level reproduction modes instead use `cache_pooling: tokens` (+ a higher `cache_max_gb`), e.g.
the attention-distance / per-neuron configs.

## Gotchas / invariants / default-off guarantees

- **Default OFF.** `cache_features` defaults to `false` (`eval.py:227`); without it the harness runs the ordinary per-epoch encode path. Caching is strictly opt-in.
- **`pooled` ⇒ linear only.** `eval.py:302-313` raises if a non-linear probe (or a `framewise`-pooling probe) is requested under `cache_pooling='pooled'` — the token structure it needs is already collapsed. `PooledLinearProbe` is only substituted when `cache_features and cache_pooling=='pooled'` (`eval.py:316-320`).
- **Ablation modes must NOT cache.** Any mode that *changes the features per setting* (e.g. attention ablation) must set `cache_features: false`, otherwise it would reuse the unmasked baseline cache. This is called out in `modes/REPRODUCTION_PLAN.md` (Risk 1).
- **`tokens` needs uniform token count.** Mixed video resolution/length ⇒ varying `N` ⇒ `torch.cat` fails; fix by `cache_pooling='pooled'`, Qwen `resize_mode='fixed'`, or uniform inputs (`cache.py:151-159`).
- **fp16 cache, fp32 train.** Cache is stored `.half()`; training re-floats to fp32 on device. Expect fp16 rounding in cached features vs a live fp32 forward.
- **Guard fires after batch 0.** The `cache_max_gb` abort needs one batch to measure `per_sample_mb`, so it triggers early but not before the first encode.
- **`num_workers=0` in the pre-pass is deliberate** — subprocess workers deadlock at the train→val transition under spawn; overlap comes from `_ThreadPrefetcher` instead.

## Key project findings

1. **Fig-2c reproduction.** On the paper-faithful Blender toy dataset (single fixed red sphere, r=0.3 m, overhead cam, 16f@24fps, 256²) the frozen V-JEPA2-L layer-wise R² reproduces the paper's dissociation: **SPEED** is decodable early (R² ≈ 0.68 at layer 0); **DIRECTION** emerges sharply in the Physics Emergence Zone (R² ≈ 0.28 at L0 → ≈ 0.9 by layer-fraction 0.3–0.4); **accel_mag** sits in between. An earlier anti-shortcut generator (random shape/color/size) did **not** reproduce early-speed; the fix was (a) the paper-faithful fixed red sphere and (b) correct frame sampling.

2. **`frame_step` reproduction bug.** `VideoDataset.loadvideo_decord` with `frame_step=1` on a 64-frame clip samples 16 **contiguous** frames (first ¼ of the trajectory) → only sub-patch motion per tubelet → layer 0 cannot encode speed/accel. Fix: `frame_step=4` (span the whole clip) **or** the new `uniform_sampling` option (evenly sample `fpc` frames over the whole video, length-agnostic). Blender clips are natively 16 frames, so their configs use `frame_step=1 + uniform_sampling=true`.

3. **Linear-probe normalization (`pre_norm`) is REQUIRED.** `PooledLinearProbe(pre_norm=True)` applies `nn.LayerNorm` over the feature dim **per sample** (`cache.py:201`). A competitor fork set `pre_norm=False` (raw pooled features) + a fixed `lr` and got poor accuracy, because V-JEPA activation scale differs by orders of magnitude across layers, so one `lr` cannot fit all layers without input normalization. Valid alternatives: **LayerNorm (per-sample over D)** *or* **per-feature StandardScaler**; "no normalization" is the only wrong choice. (The StandardScaler alternative is conceptual — no sklearn path is implemented in this fork.)

4. **`cache_pooling` semantics.** `pooled` = `[mean‖max]` (2D, collapses time, **degrades direction**), `tokens` = full `(N,D)`, `framewise` = `(T,D)` (VLM-only, needs `num_temporal`). Token-level modes (per-neuron, attention geometry) require `cache_pooling: tokens`.

5. **Reproduction roadmap** (`evals/analysis_vlm/modes/REPRODUCTION_PLAN.md`): additive, config-driven modes for attention distance (C.6), attention ablation (C.6 Table 4), orthogonal probe sequence (C.11) + steering (C.12), circular direction geometry (C.7/C.10). Phase 0 (dispatch scaffold, default-off) and Phase 1 (`attention_distance`) are **DONE**; Phases 2–5 pending.
