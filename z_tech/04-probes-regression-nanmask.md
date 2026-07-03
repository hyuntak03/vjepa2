# 04 — Probes, regression & NaN-masking

> The probe zoo (five head classes) trained on a frozen encoder, plus the `task=regression` path that turns a CSV integer label into a standardized continuous target vector and trains **NaN-masked, DDP-safe** MSE heads whose validation metric is a per-head, all-reduced R².

## Purpose

The analysis subsystem trains lightweight **probe heads** on top of a **frozen** encoder to measure *where* (which layer/stage) a given signal becomes linearly (or attentively) decodable. This section documents two things end-to-end:

1. **The probe zoo** — `linear`, `attentive`, `framewise` (temporal-linear), `temporal-attentive`, and the cache-optimized `PooledLinearProbe` — split across `evals/analysis/probes.py` (stock heads + factory), `evals/analysis_vlm/probes.py` (VLM temporal heads), and `evals/analysis_vlm/cache.py` (the pooled-cache probe).
2. **The regression task** added to `evals/analysis_vlm/eval.py`: continuous targets from a `targets.npy` **row-indexed by the CSV integer label**, per-column NaN-aware standardization, multi-variable heads (one R² curve per variable), a **NaN-masked masked-mean MSE** loss that keeps heterogeneous clips (e.g. velocity vs. acceleration) trainable in one combined DDP run, and an R² metric assembled from padded per-head sufficient statistics.

## What changed vs upstream V-JEPA2

The **entire** `evals/analysis/` and `evals/analysis_vlm/` trees are new in this fork. Base commit `204698b` has no `evals/analysis*` at all — `git ls-tree 204698b evals/` returns only the upstream eval dirs (`action_anticipation_frozen`, `hub`, `image_classification_frozen`, `main.py`, `main_distributed.py`, `scaffold.py`, `video_classification_frozen`). So **everything below is new-file addition; no upstream file was modified.** Routing is purely by `eval_name: analysis_vlm` through `evals/scaffold.py`'s dynamic import, so `main.py` / `scaffold.py` / the existing evals stay byte-identical.

| File | Status | What it is |
|------|--------|-----------|
| `evals/analysis/probes.py` | **new** | `LinearProbe` (the only *new* probe class in this file) + `build_probe()` / `probe_name()` factory that also forwards to the stock `AttentiveClassifier`. |
| `evals/analysis_vlm/probes.py` | **new** | `TemporalAttentiveClassifier` (learnable / rope temporal PE) + `TemporalLinearProbe` (framewise). |
| `evals/analysis_vlm/cache.py` | **new** | `PooledLinearProbe` (5th head class: linear probe over a pre-pooled `[mean‖max]` cache vector) + the feature-cache machinery. |
| `evals/analysis_vlm/eval.py` | **new** | Unified frozen-encoder probing driver; hosts the `task=regression` path, the NaN-masked loss, and the R² metric. |

Within the fork, the regression path landed incrementally (commit `b6fb82a` "regression module added"). It is **additive and default-off**. Two later additive insertions into this same driver are relevant here and both preserve byte-identical behavior when their config keys are absent:

| Insertion | `eval.py` lines | Default-off guarantee |
|-----------|-----------------|-----------------------|
| `skip_base_probe` flag | 501–504 | `num_probe_epochs = 0 if skip_base_probe else num_epochs`. Absent ⇒ `num_probe_epochs == num_epochs` ⇒ the train/val loop is byte-identical. Used only by encoder-only post-hoc modes that don't need the layer probes trained. |
| post-hoc `modes` dispatch | 565–590 | `modes_cfg = args_analysis.get("modes") or {}`; empty ⇒ the whole block (and every `from evals.analysis_vlm.modes …` import) is skipped. Runs on **rank 0 only**. It receives the regression standardization internals (see [Design & data flow](#design--data-flow)). |

## Design & data flow

### Probe dispatch — five head classes

Probe selection is config-driven via a `probes:` list, each entry a `spec` dict with a `type`. Inside the per-`(stage × spec)` loop, `eval.py` computes three boolean flags from the spec (`eval.py:296-300`) and the `_build(out_dim)` closure (`eval.py:315-346`) dispatches to **one of five** head classes:

| `type` | `pooling` / `temporal_pos` (+ cache) | Class | Defined at | Order-aware? |
|--------|--------------------------------------|-------|------------|--------------|
| `linear` | `mean` \| `max` \| `meanmax` | `LinearProbe` | `analysis/probes.py:25` | No — global pool |
| `linear` | `framewise_mean` \| `framewise_max` | `TemporalLinearProbe` | `analysis_vlm/probes.py:74` | **Yes** — keeps T |
| `linear` | `mean`\|`max`\|`meanmax` **and** `cache_features` + `cache_pooling: pooled` | `PooledLinearProbe` | `analysis_vlm/cache.py:193` | No — reads pre-pooled `[mean‖max]` cache vector |
| `attentive` | `temporal_pos: none` | `AttentiveClassifier` (stock) | `src/models/attentive_pooler.py` | No — set pooling |
| `attentive` | `temporal_pos: learnable` \| `rope` | `TemporalAttentiveClassifier` | `analysis_vlm/probes.py:29` | **Yes** — temporal PE |

Flag derivation (`eval.py:299-300`):

```python
framewise = ptype == "linear" and pooling.startswith("framewise")  # spatial-pool per frame, keep T
use_tpos  = ptype == "attentive" and tpos in ("learnable", "rope")
```

The pooled-cache branch takes priority inside `_build` (`eval.py:316-320`): when `cache_features` + `cache_pooling == "pooled"`, a `linear` spec becomes a `PooledLinearProbe`; `attentive`/`framewise` specs are rejected up front (`eval.py:302-313`) because the pooled cache has already collapsed the token set they need.

### Regression data flow

```
CSV integer label  ──index──▶  targets_arr[label]           (N,D) raw target rows
        │                            │
        │                    per-column NaN-aware standardize (eval.py:195-203)
        │                            ▼
   labels (B,)              targets_t on device (N,D)        eval.py:257
        └──────────────┬─────────────┘
                       ▼
         yfull = targets_t[labels]  (B,D)                     eval.py:718
                       │  per head hi: slice columns head_cols[hi]
        ┌──────────────┴──────────────┐
        ▼                             ▼
  masked-mean MSE (train)      per-head R² sufficient stats (train+val)
  eval.py:729-736              eval.py:757-766  →  AllReduceSum  →  eval.py:782-793
```

One probe head is built **per `(stage × spec × variable)`**: the head-construction loop nests `for var_name, var_cols in reg_vars` (`eval.py:350`) inside the `(stage, spec)` loops, sets `out_dim = len(var_cols)` (`eval.py:351`), and stores the variable's column slice on the head as `tcols` (`eval.py:375`). Classification uses a single dummy variable `(None, None)` (`eval.py:191`) so the same loop serves both tasks. Every head also carries a `series` key (`eval.py:367-372`) that groups plot curves — the **variable** for regression, the **probe** for classification, and `f"{var_name}·{pname}"` when multiple probe specs coexist so each combination is its own curve.

## Key code

### `linear` — `LinearProbe` (`analysis/probes.py:25`)

Global-pool tokens `(B,N,D) → (B,D)`, optional LayerNorm, one `nn.Linear`. `analysis/probes.py:41-53`:

```python
def _pool(self, x):  # x: (B, N, D)
    if self.pooling == "mean":   return x.mean(dim=1)
    if self.pooling == "max":    return x.max(dim=1).values
    if self.pooling == "meanmax":return torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
```

`meanmax` doubles `in_dim` (`embed_dim*2`, `analysis/probes.py:37`). `LinearProbe` is the **only new probe class defined in `analysis/probes.py`**; the `build_probe()` factory (`analysis/probes.py:56-81`) constructs it for `type: linear` and *also* forwards `type: attentive` straight to the stock `AttentiveClassifier`.

### `attentive` — stock `AttentiveClassifier` (`analysis/probes.py:72-79`)

`build_probe()` forwards untouched to the stock V-JEPA `AttentiveClassifier` (cross-attention pooling + Linear), identical to `evals/video_classification_frozen`. `depth` is read from `num_probe_blocks` (falling back to `depth`, default `1`; `analysis/probes.py:76`). This is a **set** pooler — permutation-invariant over tokens.

### `framewise` — `TemporalLinearProbe` (`analysis_vlm/probes.py:74`)

Pools spatial tokens **within each frame** but keeps the temporal axis, then concatenates per-frame vectors → `Linear`. Use for temporal tasks (direction, velocity) where a global mean would wash out "up" vs. "down". `analysis_vlm/probes.py:92-107`:

```python
def forward(self, x):  # x: (B, T*S, D), temporal-major
    b, n, d = x.shape; t = self.num_temporal
    if n % t != 0: raise ValueError(...)
    s = n // t
    x = x.view(b, t, s, d)
    x = x.mean(dim=2) if self.spatial_pool == "mean" else x.max(dim=2).values  # (B,T,D)
    return self.linear(self.norm(x.reshape(b, t * d)))         # in_dim = D * T
```

Triggered when `type: linear` **and** `pooling` starts with `framewise` (`eval.py:299`). The spatial pool is the suffix — `framewise_mean` → `mean` via `pooling.split("_", 1)[1]` (`eval.py:329`). Requires `encoder.num_temporal` (VLM backends only) — raises otherwise (`eval.py:323-325`).

### `temporal-attentive` — `TemporalAttentiveClassifier` (`analysis_vlm/probes.py:29`)

Applies a **per-frame temporal positional encoding** *before* the stock attentive pooler, restoring order a depth-1 pooler would ignore. Needed for encoders that do **not** bake temporal order into token values (notably LLaVA-Video's per-frame SigLIP); V-JEPA / Qwen3-VL bake time via RoPE / temporal patches and usually don't need it. `analysis_vlm/probes.py:53-59`:

```python
def _apply_temporal(self, x):  # x: (B, T, S, D)
    if self.mode == "learnable":
        return x + self.temporal_pos                 # learnable (1,T,1,D), absolute
    d = self.embed_dim
    x1, x2 = x[..., : d//2], x[..., d//2:]            # rope: rotate-half, angle = t*inv_freq
    cos, sin = self.rope_cos.to(x.dtype), self.rope_sin.to(x.dtype)
    return torch.cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)
```

- `mode="learnable"` → a learnable `(1,T,1,D)` embedding added per frame (absolute), init `trunc_normal_(std=0.02)` (`analysis_vlm/probes.py:39-40`).
- `mode="rope"` → rotary (relative) temporal structure inside the pooler self-attn; no learnable params, needs even `embed_dim` (`analysis_vlm/probes.py:42`).
- Assumes **temporal-major** token layout `[frame0 S tokens, frame1 S tokens, …]` — how the VLM backends flatten `(B,T,S,D) → (B,T*S,D)`. Guards `n % t == 0` (`analysis_vlm/probes.py:64`).

Triggered by `type: attentive` + `temporal_pos in {learnable, rope}` (`eval.py:300`), gated on `encoder.num_temporal` (`eval.py:331-337`).

### `PooledLinearProbe` — cache-optimized linear (`analysis_vlm/cache.py:193`)

Built inline at `eval.py:316-320` when `cache_features` + `cache_pooling: pooled`. It reads a **pre-pooled** cache vector `x=(B, 2D)=[mean‖max]` and slices the half its `pooling` asks for, so no token set survives. This is why attentive/framewise probes are incompatible with `cache_pooling: pooled` (`eval.py:302-313`). See [03 — Feature caching & pooling](03-feature-caching-and-pooling.md).

### `pre_norm` — linear heads only

A LayerNorm over the pooled feature **before** the linear layer — recommended `True` for cross-layer comparison because different encoder layers have very different feature scales.

- `LinearProbe`: default `True` (`analysis/probes.py:34,38`); factory reads `spec.get("pre_norm", True)` (`analysis/probes.py:69`).
- `TemporalLinearProbe`: default `True`, over the *concatenated* `T*D` vector (`analysis_vlm/probes.py:84,89`); `eval.py:330` passes `spec.get("pre_norm", True)`.
- `PooledLinearProbe`: default `True` (`analysis_vlm/cache.py:196,201`); `eval.py:320` passes `spec.get("pre_norm", True)`.
- **Not** applicable to `attentive` / `temporal-attentive` heads — the attentive pooler carries its own internal norm; those specs silently ignore `pre_norm`.

`False` swaps in `nn.Identity()`.

### Regression targets & per-column standardization (`eval.py:189-214`)

Set `experiment.analysis.task: regression`. The dataloaders (both the shared clip `VideoDataset` and the VLM raw path) are **unchanged** — the CSV still carries an integer label. That integer **row-indexes** an `(N,D)` targets array. `eval.py:195-203`:

```python
targets_arr = np.load(tpath).astype(np.float32)          # (N,D)
if targets_arr.ndim == 1: targets_arr = targets_arr[:, None]
mu = np.nanmean(targets_arr, axis=0, keepdims=True)      # NaN-aware column mean
sd = np.nanstd(targets_arr, axis=0, keepdims=True)
targets_arr = (targets_arr - mu) / np.clip(sd, 1e-6, None)
```

- **`nanmean` / `nanstd`**: a column may be defined on only a subset of videos and `NaN` elsewhere; NaN-aware stats ignore the undefined rows. NaNs **stay NaN** after the transform → masked out per head later.
- Standardization keeps MSE/lr well-scaled regardless of units (pixels vs. sin/cos in `[-1,1]`); **R² is invariant** to this affine transform, so the reported metric is unaffected.
- The array is moved on-device once (`targets_t`, `eval.py:257`); per batch the harness gathers `yfull = targets_t[labels]` (`eval.py:718`).
- **Downstream:** `mu`, `sd`, the raw `tpath`, `targets_t`, and `reg_vars` are handed to the post-hoc modes system via `AnalysisContext(col_mu=mu, col_sd=sd, targets_npy=tpath, targets_t=…, reg_vars=…)` (`eval.py:572-579`). Modes that report **raw** (unstandardized) angles/speeds recover them with these stats. See [12 — Analysis modes](12-analysis-modes.md).

### Multi-variable heads (`eval.py:204-212`)

`variables:` lists named column-slices of the target array; **each becomes its own R² curve** on the same plot (paper Fig. 2c: speed / direction / accel together).

```python
reg_vars = [(v["name"], [int(c) for c in v["cols"]]) for v in var_cfg]
```

Default (no `variables`): one variable spanning all D columns (`eval.py:205-206`). Each `col` is bounds-checked against `D` (`eval.py:209-210`).

### NaN-masking — combined heterogeneous run (`eval.py:729-736`)

One combined dataset can hold **velocity clips** (speed defined, accel NaN) **and** acceleration clips (accel defined, speed NaN); the speed head trains only on velocity clips, the accel head only on acceleration clips — in a **single** run. Training loss is a **masked-mean MSE per head**: NaN target rows contribute `0` but are **kept in the graph** so the DDP static-graph structure is byte-identical across ranks regardless of which rows are valid on each rank.

```python
# masked-mean MSE per head: NaN target rows contribute 0 (kept in the graph so the
# DDP static-graph structure is identical across ranks regardless of which rows are valid).
losses = []
for hi in range(n_heads):
    yh = yfull[:, head_cols[hi]]
    m = (~torch.isnan(yh).any(dim=1)).float()                       # (B,)
    err = ((preds[hi] - torch.nan_to_num(yh)) ** 2).sum(dim=1) * m  # (B,)
    losses.append(err.sum() / m.sum().clamp(min=1.0))
```

- `nan_to_num(yh)` prevents `0*NaN = NaN` from poisoning the graph; the multiply by `m` (not boolean indexing) keeps tensor shapes and the autograd graph identical every step.
- `m.sum().clamp(min=1.0)` avoids div-by-zero when a rank's batch has **no** valid rows for that head (loss is then `0`, gradient `0`).
- All heads' losses are summed (`loss_total = sum(losses)`, `eval.py:741`) and back-propped through one fused AdamW.
- **Validation runs under `torch.no_grad()`** (`grad_ctx` at `eval.py:723`): a grad-enabled DDP forward with no backward would trip the `static_graph` reducer on the next forward under multi-GPU, so heads are run without arming the reducer.

### R² metric — padded per-head sufficient statistics

R² = `1 − SS_res / SS_tot`, computed **per head over that head's valid samples**, with all sufficient statistics all-reduced across ranks. The buffers are declared over `Dmax = max(len(c) for c in head_cols)` (**defined at `eval.py:691`**; the four buffers `ss_res`/`sum_y`/`sum_y2`/`cnt` are allocated at `eval.py:695-698`) so a 1-column head (speed) and a 2-column head (direction sin,cos) share one tensor. Accumulation `eval.py:757-766`:

```python
for hi in range(n_heads):
    d = len(head_cols[hi])
    yh = yfull[:, head_cols[hi]]
    m = ~torch.isnan(yh).any(dim=1)            # valid rows for this variable
    if m.any():
        p, y = preds[hi][m].float(), yh[m]
        ss_res[hi] += ((p - y) ** 2).sum()     # Σ‖pred-y‖²
        sum_y[hi, :d] += y.sum(dim=0)          # Σy   (padded to Dmax)
        sum_y2[hi] += (y ** 2).sum()           # Σ‖y‖²
        cnt[hi] += m.sum()                     # #valid
```

After `AllReduceSum` over all four buffers (`eval.py:782-787`), SS_tot is recovered from the sufficient stats and R² formed per head (`eval.py:789-793`):

```python
d, nh = len(head_cols[hi]), cnt[hi].clamp(min=1)
sst = (sum_y2[hi] - (sum_y[hi, :d] ** 2).sum() / nh).clamp(min=1e-12)  # Σ(y-ȳ)²
r2.append((1.0 - ss_res[hi] / sst).item())
```

- `SS_tot = Σy² − (Σy)²/n = Σ(y−ȳ)²`, using each head's **own** valid-subset mean (`Σy/cnt`), summed over that head's `d` columns.
- `.clamp(min=1e-12)` guards a degenerate (constant-target) head.
- Reported per-head; R² **can be negative** (a probe worse than predicting the mean), so `best_val` is seeded with `-inf` (`eval.py:505`), and the plot draws an R²=0 "predict mean" baseline (`analysis/plotting.py:129-132`).
- The tqdm postfix shows the best head's running R² every 20 iters (`eval.py:771-778`).

`summary.json` records `metric: r2` and the `variables` list when regressing (`eval.py:544-547`).

## Configuration

Real YAML for a combined velocity+acceleration+direction run (raw VLM path). Only the `experiment.analysis` block is regression-specific; the rest is the standard `analysis_vlm` scaffold.

```yaml
eval_name: analysis_vlm
folder: /path/to/runs/toyball
tag: reg_combined

model_kwargs:
  checkpoint: lmms-lab/LLaVA-Video-7B-Qwen2   # HF repo id (vlm) or local .pth (vjepa)
  cache_dir: /path/to/hf_cache

experiment:
  analysis:
    model: llavavideo            # -> backend module + raw data mode
    stages: {vision_encoder: all}   # or [int,...] / "all"
    task: regression
    regression:
      targets_npy: /path/to/toyball_targets.npy   # (N, D) float array
      variables:
        - {name: speed,     cols: [0]}       # velocity clips (accel cols NaN)
        - {name: direction, cols: [1, 2]}    # sin,cos of angle (both clip types)
        - {name: accel_mag, cols: [3]}       # acceleration clips (speed col NaN)
    probes:
      - {type: linear,    pooling: framewise_mean, pre_norm: true}
      - {type: attentive, temporal_pos: rope, num_probe_blocks: 1}
    plot: true

  data:
    dataset_train: /path/to/train.csv   # CSV integer label indexes targets_npy
    dataset_val:   /path/to/val.csv
    num_classes: 4          # ignored for regression heads (out_dim = len(cols))
    frames_per_clip: 16
    resolution: 224

  optimization:
    batch_size: 16
    num_epochs: 30
    use_bfloat16: true
    default_head: {lr: 0.001, weight_decay: 0.01, warmup: 1.0}
```

| Key | Meaning | Default | Allowed values |
|-----|---------|---------|----------------|
| `analysis.task` | probe objective | `classification` | `classification` \| `regression` |
| `analysis.regression.targets_npy` (alias `targets`) | path to the `(N,D)` float target array (1-D promoted to `(N,1)`) | — (required for regression) | any `.npy` path |
| `analysis.regression.variables[*].name` | curve/series label for this variable | `target` (single default var) | any string |
| `analysis.regression.variables[*].cols` | column indices into `targets_npy` for this head | all columns | ints in `[0, D)` |
| `analysis.probes[*].type` | head family | `attentive` | `linear` \| `attentive` |
| `analysis.probes[*].pooling` | linear pooling | `mean` | `mean` \| `max` \| `meanmax` \| `framewise_mean` \| `framewise_max` |
| `analysis.probes[*].temporal_pos` | temporal PE (attentive) | `none` | `none` \| `learnable` \| `rope` |
| `analysis.probes[*].pre_norm` | LayerNorm before linear (linear heads only) | `true` | bool |
| `analysis.probes[*].num_probe_blocks` (alias `depth`) | attentive pooler depth | `1` | int |
| `analysis.skip_base_probe` | skip probe training (0 epochs) for encoder-only modes | `false` | bool |
| `optimization.cache_features` | encode once, train over cache | `false` | bool |
| `optimization.cache_pooling` | cache granularity | `tokens` | `tokens` \| `pooled` (linear only) \| `framewise` |

Notes:

- `variables[*].cols` are column indices into `targets_npy`; each variable spawns its own head per stage and its own R² curve.
- `num_classes` is only used for classification heads; a regression head's output dim is `len(cols)`.

## Invariants & gotchas

- **Default-off (task).** `task` defaults to `classification` (`eval.py:189`); the regression path is entered only on `task: regression`, and `targets_npy` is asserted present (`eval.py:194`). No config change ⇒ byte-identical classification behavior.
- **Default-off (`skip_base_probe`).** Additive flag (`eval.py:504`); absent ⇒ `num_probe_epochs == num_epochs` ⇒ the train/val loop is byte-identical. Only set it for encoder-only post-hoc modes that don't need the layer probes trained.
- **CSV label = target row index.** The integer label is *not* a class — it row-indexes `targets_npy`. Row count `N` must cover every label in train+val CSVs.
- **NaN is meaningful, not an error.** A NaN target row = "this variable undefined for this clip"; it is masked out of loss *and* R² for that head. A column that is NaN for *all* rows breaks `nanstd` (→ near-zero, clamped to `1e-6`) — don't include empty columns.
- **DDP static-graph invariance.** The train loss keeps NaN rows in the graph (multiply-by-mask + `nan_to_num`) rather than indexing them out, so every rank runs an identical graph each step (`static_graph=True`, `eval.py:354`). Do **not** "optimize" the loss to skip invalid rows via boolean indexing — that desyncs ranks.
- **Validation runs under `no_grad`** to keep the DDP reducer disarmed (`eval.py:723`).
- **R² can be < 0**; `best_val` floors at `-inf`, not `0` (`eval.py:505`); the plot draws the R²=0 "predict mean" baseline (`analysis/plotting.py:129-132`).
- **Temporal heads need `encoder.num_temporal`.** `framewise` and `temporal_pos` probes raise on encoders that don't expose it (V-JEPA already encodes time via RoPE); they are VLM-backend-only (`eval.py:323-325`, `eval.py:331-337`).
- **Temporal-major layout assumed.** `TemporalAttentiveClassifier` / `TemporalLinearProbe` require token count divisible by `num_temporal` and the `[frame0…, frame1…]` order the VLM backends produce; they raise on mismatch (`analysis_vlm/probes.py:64,95`).
- **`pre_norm` only affects linear heads** (`LinearProbe` / `TemporalLinearProbe` / `PooledLinearProbe`). Attentive specs silently ignore it.
- **Feature cache + probe compatibility.** `cache_pooling: pooled` caches only `[mean‖max]` vectors → `PooledLinearProbe` (non-framewise linear) only; attentive and framewise probes need `cache_pooling: tokens` (or `cache_features: false`) and raise otherwise (`eval.py:302-313`).
- **Standardization stats leave the driver.** `mu`/`sd`/`targets_t`/`targets_npy`/`reg_vars` are surfaced to the post-hoc modes via `AnalysisContext` (`eval.py:572-579`); a mode that reports raw quantities must un-standardize with `col_mu`/`col_sd`.

## Cross-references

- [02 — Analysis-VLM harness](02-analysis-vlm-harness.md) — the surrounding driver (config parse, DDP, fused optimizer, train/eval loop).
- [03 — Feature caching & pooling](03-feature-caching-and-pooling.md) — `cache_features`, `cache_pooling`, and `PooledLinearProbe`.
- [05 — Analysis-CLIP harness](05-analysis-clip-harness.md) — the V-JEPA clip data path that feeds these probes.
- [07 — Plotting](07-plotting.md) — the R² curve rendering, R²=0 baseline, and PEZ shading.
- [10 — Datasets, CSVs & targets](10-datasets-csv-targets.md) — how `targets.npy` and the label-indexed CSVs are built.
- [12 — Analysis modes](12-analysis-modes.md) — where the standardized targets (`col_mu`/`col_sd`/`targets_t`/`reg_vars`) and `skip_base_probe` feed the additive post-hoc modes subsystem.
