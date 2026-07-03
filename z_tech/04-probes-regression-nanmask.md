# Probes, regression task & NaN-masking

## Purpose

The analysis subsystem trains lightweight **probe heads** on top of a **frozen**
encoder to measure *where* (which layer/stage) a given signal becomes linearly (or
attentively) decodable. This section documents:

1. The probe zoo — `linear`, `attentive`, `framewise` (temporal-linear), and
   `temporal-attentive` heads — split across `evals/analysis/probes.py` (stock heads
   + factory) and `evals/analysis_vlm/probes.py` (VLM-specific temporal heads).
2. The **regression** task added to `evals/analysis_vlm/eval.py`: continuous targets
   from a `targets.npy` indexed by the CSV integer label, per-column standardization,
   multi-variable heads, **NaN-masking** so heterogeneous clips (e.g. velocity vs.
   acceleration) train in one combined DDP run, and the R² metric.

## What changed vs upstream V-JEPA2

The **entire** `evals/analysis/` and `evals/analysis_vlm/` trees are new in this fork
— base commit `204698b` has no `evals/analysis*` at all (`git ls-tree 204698b evals/`
returns only the upstream eval dirs). So everything below is **new-file** additions;
nothing upstream was modified to build this. It is routed purely by `eval_name:
analysis_vlm` via `evals/scaffold.py`'s dynamic import, so `main.py` / `scaffold.py` /
existing evals are untouched.

| File | Status | What it is |
|------|--------|-----------|
| `evals/analysis/probes.py` | **new** | `LinearProbe` + `build_probe()`/`probe_name()` factory; wraps stock `AttentiveClassifier`. |
| `evals/analysis_vlm/probes.py` | **new** | `TemporalAttentiveClassifier` (learnable / rope temporal PE) + `TemporalLinearProbe` (framewise). |
| `evals/analysis_vlm/eval.py` | **new** | Unified frozen-encoder probing driver; adds the `task=regression` path + NaN-masked loss + R² metric. |

Within-fork, the regression path itself landed incrementally (commit `b6fb82a`
"regression module added"); it is **additive** and default-off (see Gotchas).

---

## Probe types

Probe selection is config-driven via a `probes:` list, each entry a `spec` dict with a
`type`. `eval.py` inspects the spec and dispatches to one of four head classes.
Dispatch logic: `evals/analysis_vlm/eval.py:296-346`.

| `type` | `pooling` / `temporal_pos` | Class | File | Order-aware? |
|--------|----------------------------|-------|------|--------------|
| `linear` | `mean` \| `max` \| `meanmax` | `LinearProbe` | `analysis/probes.py:25` | No (global pool) |
| `linear` | `framewise_mean` \| `framewise_max` | `TemporalLinearProbe` | `analysis_vlm/probes.py:74` | **Yes** (keeps T) |
| `attentive` | `temporal_pos: none` | `AttentiveClassifier` (stock) | `src/models/attentive_pooler.py` | No (set pooling) |
| `attentive` | `temporal_pos: learnable` \| `rope` | `TemporalAttentiveClassifier` | `analysis_vlm/probes.py:29` | **Yes** (temporal PE) |

### `linear` — `LinearProbe`

Global-pool tokens `(B,N,D) → (B,D)`, optional LayerNorm, one `nn.Linear`.
`evals/analysis/probes.py:41-53`:

```python
def _pool(self, x):  # x: (B, N, D)
    if self.pooling == "mean":   return x.mean(dim=1)
    if self.pooling == "max":    return x.max(dim=1).values
    if self.pooling == "meanmax":return torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
```

`meanmax` doubles `in_dim` (`embed_dim*2`). This is the *only* probe whose factory
lives in `build_probe()` (`analysis/probes.py:56`).

### `attentive` — stock `AttentiveClassifier`

`build_probe()` (`evals/analysis/probes.py:72-79`) forwards straight to the stock
V-JEPA `AttentiveClassifier` (cross-attention pooling + Linear), identical to
`evals/video_classification_frozen`. `depth` is read from `num_probe_blocks` (falling
back to `depth`, default `1`). This is a **set** pooler — permutation-invariant over
tokens.

### `framewise` — `TemporalLinearProbe`

A linear probe that **pools spatial tokens within each frame but keeps the temporal
axis**, then concatenates per-frame vectors → `Linear`. Use for temporal tasks
(direction, velocity) where a global mean would wash out "up" vs. "down".
`evals/analysis_vlm/probes.py:92-107`:

```python
def forward(self, x):  # x: (B, T*S, D), temporal-major
    b, n, d = x.shape; t = self.num_temporal
    if n % t != 0: raise ValueError(...)
    s = n // t
    x = x.view(b, t, s, d)
    x = x.mean(dim=2) if self.spatial_pool == "mean" else x.max(dim=2).values  # (B,T,D)
    return self.linear(self.norm(x.reshape(b, t * d)))         # in_dim = D * T
```

Triggered when `type: linear` **and** `pooling` starts with `framewise`
(`eval.py:299`). The spatial pool is the suffix: `framewise_mean` → `mean`
(`eval.py:329`, `pooling.split("_", 1)[1]`). Requires `encoder.num_temporal` (VLM
backends only) — raises otherwise (`eval.py:323-325`).

### `temporal-attentive` — `TemporalAttentiveClassifier`

Applies a **per-frame temporal positional encoding** *before* the stock attentive
pooler, restoring order that a depth-1 pooler would ignore. Needed for encoders that
do **not** bake temporal order into token values (notably LLaVA-Video's per-frame
SigLIP); V-JEPA / Qwen3-VL bake time via RoPE/temporal patches and usually don't need
it. `evals/analysis_vlm/probes.py:53-71`:

```python
def _apply_temporal(self, x):  # x: (B, T, S, D)
    if self.mode == "learnable":
        return x + self.temporal_pos                 # learnable (1,T,1,D), absolute
    d = self.embed_dim
    x1, x2 = x[..., : d//2], x[..., d//2:]            # rope: rotate-half, angle = t*inv_freq
    cos, sin = self.rope_cos.to(x.dtype), self.rope_sin.to(x.dtype)
    return torch.cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)
```

- `mode="learnable"` → a learnable `(1,T,1,D)` embedding added per frame.
- `mode="rope"` → rotary (relative) temporal structure inside the pooler self-attn;
  no learnable params, needs even `embed_dim`.
- Assumes **temporal-major** token layout `[frame0 S tokens, frame1 S tokens, …]`
  — how the VLM backends flatten `(B,T,S,D) → (B,T*S,D)`. Guards `n % t == 0`
  (`probes.py:64`).

Triggered by `type: attentive` + `temporal_pos in {learnable, rope}`
(`eval.py:300`), also gated on `encoder.num_temporal` (`eval.py:331-337`).

### `pre_norm`

A LayerNorm over the pooled feature **before** the linear layer — recommended `True`
for cross-layer comparison because different encoder layers have very different
feature scales.

- `LinearProbe`: `pre_norm` default `True` (`analysis/probes.py:34,38`); factory reads
  `spec.get("pre_norm", True)` (`analysis/probes.py:69`).
- `TemporalLinearProbe`: `pre_norm` default `True`, over the *concatenated* `T*D`
  vector (`analysis_vlm/probes.py:84,89`); `eval.py:330` passes
  `spec.get("pre_norm", True)`.
- **Not** applicable to `attentive` / `temporal-attentive` heads — the attentive
  pooler carries its own internal norm; those specs ignore `pre_norm`.

`False` swaps in `nn.Identity()`.

---

## Regression task (`evals/analysis_vlm/eval.py`)

Set `experiment.analysis.task: regression`. The dataloaders (both the shared clip
`VideoDataset` and the VLM raw path) are **unchanged** — the CSV still carries an
integer label. That integer **indexes** an `(N,D)` targets array, and the harness maps
`label → target vector`. Config parse + validation: `eval.py:189-214`.

### Targets & per-column standardization

`eval.py:195-203`:

```python
targets_arr = np.load(tpath).astype(np.float32)          # (N,D)
if targets_arr.ndim == 1: targets_arr = targets_arr[:, None]
mu = np.nanmean(targets_arr, axis=0, keepdims=True)      # NaN-aware column mean
sd = np.nanstd(targets_arr, axis=0, keepdims=True)
targets_arr = (targets_arr - mu) / np.clip(sd, 1e-6, None)
```

- **`nanmean`/`nanstd`**: a column may be defined on only a subset of videos and `NaN`
  elsewhere; NaN-aware stats ignore the undefined rows. NaNs **stay NaN** after the
  transform → masked out per head later.
- Standardization keeps MSE/lr well-scaled regardless of units (pixels vs. sin/cos in
  `[-1,1]`); **R² is invariant** to this affine transform, so the reported metric is
  unaffected.
- The array is moved on-device once; per batch the harness gathers
  `targets_t[labels]` (`eval.py:257`, `eval.py:718`).

### Multi-variable heads (one head/curve per variable)

`variables:` lists named column-slices of the target array; **each becomes its own R²
curve** on the same plot (paper Fig. 2c: speed / direction / accel together). Parsed at
`eval.py:204-212`:

```python
reg_vars = [(v["name"], [int(c) for c in v["cols"]]) for v in var_cfg]
```

Default (no `variables`): one variable spanning all D columns (`eval.py:205-206`).
Head construction loops `for var_name, var_cols in reg_vars` (`eval.py:350`), builds
one probe per `(stage × spec × variable)` with `out_dim = len(var_cols)`, and stores
the column slice as `tcols` (`eval.py:375`). The plot line grouping is the `series`
key (`eval.py:367-372`): the **variable** for regression; if multiple probe specs
coexist, `series = f"{var_name}·{pname}"` so each is its own curve.
Classification uses a single dummy variable `(None, None)` (`eval.py:191`).

### NaN-masking (combined heterogeneous run)

The point: one combined dataset can hold **velocity clips** (speed defined, accel NaN)
**and** acceleration clips (accel defined, speed NaN); the speed head trains only on
velocity clips, the accel head only on acceleration clips — in a **single** run.

Training loss is a **masked-mean MSE per head** — NaN target rows contribute `0` but
are **kept in the graph** so the DDP static-graph structure is byte-identical across
ranks regardless of which rows are valid on each rank. `eval.py:729-736`:

```python
for hi in range(n_heads):
    yh = yfull[:, head_cols[hi]]
    m = (~torch.isnan(yh).any(dim=1)).float()                       # (B,) valid-row mask
    err = ((preds[hi] - torch.nan_to_num(yh)) ** 2).sum(dim=1) * m  # NaN->0, then masked
    losses.append(err.sum() / m.sum().clamp(min=1.0))               # mean over valid rows
```

Key invariants:
- `nan_to_num(yh)` prevents `NaN → 0*NaN = NaN` poisoning the graph; the multiply by
  `m` (not indexing) keeps tensor shapes and the autograd graph identical every step.
- `m.sum().clamp(min=1.0)` avoids div-by-zero when a rank's batch has **no** valid rows
  for that head (loss is then `0`, gradient `0`).
- All heads' losses are summed (`eval.py:741`) and back-propped through one fused
  AdamW; validation runs heads under `torch.no_grad()` so the DDP reducer isn't armed
  (`eval.py:723`, avoids the static-graph reducer tripping on a grad-forward with no
  backward).

### R² metric (padded per-head stats)

R² = `1 − SS_res / SS_tot`, computed **per head over that head's valid samples**, with
all sufficient statistics all-reduced across ranks. Accumulation `eval.py:757-766`:

```python
for hi in range(n_heads):
    d = len(head_cols[hi]); yh = yfull[:, head_cols[hi]]
    m = ~torch.isnan(yh).any(dim=1)                 # valid rows for this variable
    if m.any():
        p, y = preds[hi][m].float(), yh[m]
        ss_res[hi]  += ((p - y) ** 2).sum()         # Σ‖pred-y‖²
        sum_y[hi,:d] += y.sum(dim=0)                # Σy   (padded to Dmax)
        sum_y2[hi]  += (y ** 2).sum()               # Σ‖y‖²
        cnt[hi]     += m.sum()                      # #valid
```

The per-head buffers are padded to `Dmax = max(len(c))` (`eval.py:692-698`) so a
1-column head (speed) and a 2-column head (direction sin,cos) live in one tensor.
After `AllReduceSum`, SS_tot is recovered from the sufficient stats and R² formed
(`eval.py:782-793`):

```python
sst = (sum_y2[hi] - (sum_y[hi,:d] ** 2).sum() / nh).clamp(min=1e-12)  # Σ(y-ȳ)²
r2.append((1.0 - ss_res[hi] / sst).item())
```

- `SS_tot = Σy² − (Σy)²/n = Σ(y−ȳ)²`, using each head's **own** valid-subset mean
  (`Σy/cnt`), summed over that head's `d` columns.
- `.clamp(min=1e-12)` guards a degenerate (constant-target) head.
- Reported per-head; R² **can be negative** (a probe worse than predicting the mean),
  so `best_val` is seeded with `-inf` (`eval.py:505`), and the plot draws an R²=0
  "predict mean" baseline (`analysis/plotting.py:129-131`).

`summary.json` records `metric: r2` and the `variables` list when regressing
(`eval.py:544-547`).

---

## Config

Real YAML for a combined velocity+acceleration+direction run (raw VLM path). Only the
`experiment.analysis` block is regression-specific; the rest is the standard
`analysis_vlm` scaffold.

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

Notes:
- `variables[*].cols` are column indices into `targets_npy`; each variable spawns its
  own head per stage and its own R² curve.
- `num_classes` is only used for classification heads; a regression head's output dim
  is `len(cols)`.

---

## Gotchas / invariants / default-off guarantees

- **Default-off**: `task` defaults to `classification` (`eval.py:189`); the regression
  path is entered only on `task: regression`, and `targets_npy` is asserted present
  (`eval.py:194`). No config change ⇒ byte-identical classification behavior.
- **CSV label = target row index.** The integer label is *not* a class — it indexes
  `targets_npy` row-wise. Row count `N` must cover every label in train+val CSVs.
- **NaN is meaningful, not an error.** A NaN target row = "this variable undefined for
  this clip"; it is masked out of loss *and* R² for that head. A column that is NaN for
  *all* rows breaks `nanstd` (→ near-zero, clamped to `1e-6`) — don't include empty
  columns.
- **DDP static-graph invariance.** The train loss keeps NaN rows in the graph
  (multiply-by-mask, `nan_to_num`) rather than indexing them out, so every rank runs an
  identical graph each step (`static_graph=True`, `eval.py:354`). Do **not** "optimize"
  the loss to skip invalid rows via boolean indexing — that desyncs ranks.
- **Validation runs under `no_grad`** to keep the DDP reducer disarmed (`eval.py:723`).
- **R² can be < 0**; `best_val` floors at `-inf`, not `0` (`eval.py:505`).
- **Temporal heads need `encoder.num_temporal`.** `framewise` and `temporal_pos`
  probes raise on encoders that don't expose it (V-JEPA already encodes time via RoPE);
  they are VLM-backend-only (`eval.py:323-325`, `eval.py:331-337`).
- **Temporal-major layout assumed.** `TemporalAttentiveClassifier` /
  `TemporalLinearProbe` require token count divisible by `num_temporal` and the
  `[frame0…, frame1…]` order the VLM backends produce; they raise on mismatch.
- **`pre_norm` only affects linear heads.** Attentive specs silently ignore it.
- **Feature cache + probe compatibility.** `cache_pooling: pooled` caches only
  `[mean‖max]` vectors → linear (non-framewise) probes only; attentive and framewise
  probes need `cache_pooling: tokens` (or `cache_features: false`) and raise otherwise
  (`eval.py:302-313`).
