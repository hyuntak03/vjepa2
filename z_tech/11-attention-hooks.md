# 11 — Attention hooks (distance + ablation)

> An additive, default-off, **zero-core-edit** instrumentation layer that monkey-patches `F.scaled_dot_product_attention` (plus forward pre/post hooks on every `RoPEAttention` block) to (a) capture per-head spatial+temporal attention distance and (b) inject a local-attention ablation bias — all while leaving `src/models/**` byte-for-byte upstream and restoring everything on context exit.

## Purpose

`evals/analysis/attention_hooks.py` instruments the **frozen V-JEPA2 RoPE encoder** without forking a single line of model code. It provides two features, both installed at runtime as a monkey-patch on `torch.nn.functional.scaled_dot_product_attention` combined with forward-pre/-post hooks on each `RoPEAttention` module:

1. **Distance capture** — per `(layer, head)`, the attention-weighted **spatial** (patch-grid Euclidean) and **temporal** (tubelet `|Δt|`) distance, i.e. `E_i[ Σ_j softmax_attn[h,i,j] · dist(i,j) ]`. This is the standard "mean attention distance" diagnostic, streamed over query chunks so the full `(B,H,N,N)` attention matrix is never materialized.
2. **Local-attention ablation** — an additive attention bias built from the `(T,H,W)` token grid that sets `-inf` on every query/key pair *within* a distance threshold, injected via the SDPA `attn_mask` argument. Used to knock out local heads and measure the downstream effect (paper Appendix C.6 / Table 4).

The whole point is that **`src/models/**` is never functionally touched** — the encoder runs upstream code verbatim, and on context exit everything is restored so the model is left byte-for-byte identical. `src/models/utils/modules.py` is confirmed **byte-identical to upstream** (`git diff 204698b -- src/models/utils/modules.py` is empty).

The one **live caller** today is the `attention_distance` analysis mode (`evals/analysis_vlm/modes/attention_distance.py`), which runs **capture-only** (no ablation) and reproduces the paper's Figure 3 per-head heatmap on the Blender toy set. The ablation path is implemented and unit-usable but not yet wired to any pipeline (scheduled for reproduction Phase 5).

## What changed vs upstream V-JEPA2

Base for comparison: upstream commit **`204698b`** (`Fix figure (#143)`).

| File | Status | Delta |
|---|---|---|
| `evals/analysis/attention_hooks.py` | **new** | The entire subsystem: SDPA monkey-patch, `RoPEAttention` pre/post hooks, `AttentionDistanceCollector`, `build_ablation_bias`, geometry caches, and a dormant `parse_config`/`maybe_context` config front-end. |
| `evals/analysis_vlm/modes/attention_distance.py` | **new** | The only live caller (**Phase 1**). Registers the `attention_distance` mode; runs **capture-only** over `max_batches` val clips; writes **three** artifacts — `attention_distance.json` + the **primary** Fig-3 heatmap `attention_distance.png` + the **secondary** Fig-19 dual-axis line plot `attention_distance_layerwise.png`. |
| `evals/analysis_vlm/modes/__init__.py` | **new** | Modes registry + `run_modes()` + `AnalysisContext` (**Phase 0** scaffold). `@register("attention_distance")` self-registers here. See ch 12. |
| `evals/analysis_vlm/eval.py` | **modified** | Two *additive* touches (default-off). `git diff 204698b`: (1) `:504` `num_probe_epochs = 0 if args_analysis.get("skip_base_probe", False) else num_epochs` then `for epoch in range(start_epoch, num_probe_epochs)`; (2) `:565–589` the rank-0-only dispatch block `modes_cfg = args_analysis.get("modes") or {}; if modes_cfg and rank == 0: … run_modes(modes_cfg, ctx)`. Both no-ops when the keys are absent. |
| `z_scripts/run_attn_distance_vjepa.sh` | **new** | SLURM launcher for the distance run (single-GPU on purpose — see below). |
| `configs/analysis/blender_toy_dataset/vjepa_attn_distance.yaml` | **new** | The one config that actually exercises the hook (`experiment.analysis.modes.attention_distance`). |
| `src/models/utils/modules.py` | **unchanged** | `git diff 204698b -- src/models/utils/modules.py` is **empty** (0 lines). The hooks work purely because upstream `RoPEAttention.forward` already threads the grid dims and `attn_mask` into SDPA. |
| `src/models/vision_transformer.py` | **cosmetic only** | The *sole* change anywhere under `src/models/` is a single non-functional comment at `:187` (a Korean note on patch-vs-pos embedding); it does not affect execution and is unrelated to the hooks. |

**Default-off guarantee (eval.py):** every existing config lacks the `modes` key, so `args_analysis.get("modes") or {}` is `{}` → the dispatch block is skipped and *nothing is imported*. `skip_base_probe` defaults `False` → `num_probe_epochs == num_epochs` → the probe loop is byte-identical. Existing runs behave exactly as before.

## Design & data flow

### How it hooks without editing `src/models/**`

Three upstream facts make the patch possible; none required a fork edit.

**1. `RoPEAttention.forward` already threads the grid + `attn_mask`.** `src/models/utils/modules.py:331`:

```python
def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
```

and its SDPA branch (guard at `modules.py:372`, call at `:374`) runs on the **already-RoPE-rotated** `q,k`:

```python
if attn_mask is not None or self.use_sdpa:
    with torch.backends.cuda.sdp_kernel():
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
        )
```

So `(T, H_patches, W_patches)` arrive as forward kwargs, and any additive bias placed in `attn_mask` is honored by the real kernel. The RoPE rotation is applied to `q,k` **before** this call, so the captured softmax logits match what the model actually computes.

**2. The `F` symbol is shared.** `modules.py:8` does `import torch.nn.functional as F` at module scope; `attention_hooks.py:65` captures and later rebinds that same attribute:

```python
_ORIG_SDPA = F.scaled_dot_product_attention
...
F.scaled_dot_product_attention = patched_sdpa   # observed by modules.py at call time
```

Because `RoPEAttention.forward` resolves `F.scaled_dot_product_attention` at call time, rebinding the function-module attribute is enough — no wrapping of the `nn.Module` needed.

**3. A pre-hook stashes per-call context; SDPA reads the top of a thread-local stack.** `attention_hooks.py:270` registers `pre_hook(..., with_kwargs=True)` on every `RoPEAttention` module; it reads `T/H_patches/W_patches` from the kwargs and pushes a frame; `post_hook` (`:285`) pops it. SDPA is called **exactly once** inside each `RoPEAttention.forward`, so the top of stack is always the correct layer while the patched SDPA runs.

```python
def pre_hook(mod, args, kwargs):
    T = kwargs.get("T"); H = kwargs.get("H_patches"); W = kwargs.get("W_patches")
    tok_mask = kwargs.get("mask")
    ok = (T is not None and H is not None and W is not None and tok_mask is None)
    _stack().append(dict(active=ok, layer=mod._attn_layer_idx, scale=mod._attn_scale, T=T, H=H, W=W))
```

When no active frame is on the stack — any non-`RoPEAttention` SDPA call (plain `Attention`, `CrossAttention`, `ACRoPEAttention`, probe heads) — the patched SDPA (`:298`) is a straight pass-through to `_ORIG_SDPA`.

### Distance capture — `AttentionDistanceCollector` (`:161`)

Accumulates running float64 sums of expected distance per `(layer, head)`; `finalize()` divides by the per-layer row count.

- **Streaming over query chunks** (`:199`, default `query_chunk=512`): for each chunk it recomputes `logits = (q@kᵀ)·scale`, `softmax`, then `Σ_j attn·dist` for both spatial (`ds`) and temporal (`dt`) — so the full `(B,H,N,N)` is never held. `q,k` are cast to `compute_dtype=float32` at **`:194–195`** for a stable statistic even when the model runs bf16/fp16.
- **Detached side computation** (`@torch.no_grad()`, `:188`). It does *not* feed the encoder — the patched SDPA still returns the original kernel's output. It roughly doubles attention FLOPs for the measured batches.
- **Batch gating** (`:178`, `:182`): `note_forward` increments `_forward_calls` only when it sees **layer 0** (≈ one increment per encoder forward); `enabled_for` returns `True` while `_forward_calls <= max_batches`, so only the first `max_batches` forwards are measured. `enabled_for` returns `True` unconditionally when `max_batches is None`.
- **Scale consistency**: `eff_scale` (`:302`) falls back to `ctx["scale"] = m.scale = head_dim**-0.5` when SDPA is called with `scale=None` (which the encoder does), matching the kernel's internal default since the last dim equals `head_dim`.

`finalize()` (`:217`) output shape:

```json
{"spatial_distance": [[per-head]... per-layer],
 "temporal_distance": [[...]...],
 "num_layers": L, "num_heads": H, "rows_per_layer": [...]}
```

**Geometry.** Token index → `(t,h,w)` must match `RoPEAttention.separate_positions` (`modules.py:316`) exactly: `t = idx // (H·W)`, `h = (idx − t·H·W) // W`, `w = remainder`. This is reproduced in `_coords` (`:90`); pairwise distances come from `_dist_rows` (`:104`) with spatial = Euclidean patch distance, temporal = `|Δt|` tubelets. `_COORD_CACHE`/`_DIST_CACHE` memoize by `(T,H,W,device,…)`.

### Ablation bias — `build_ablation_bias` (`:120`)

Returns an additive `(1,1,N,N)` bias, `0` = keep / `-inf` = drop, cached by full key.

```python
if   mode == "spatial":  drop = ds <= float(spatial)
elif mode == "temporal": drop = dt <= float(temporal)
elif mode == "combined": drop = (ds <= float(spatial)) & (dt <= float(temporal))
...
full = drop.all(dim=1)          # never fully mask a row -> avoid all -inf softmax = NaN
if full.any():
    drop = drop.clone(); drop[full] = False
bias = torch.zeros_like(ds, dtype=dtype)
bias.masked_fill_(drop, float("-inf"))
```

Injected in the patched SDPA (`:305`): the bias is summed with any caller `attn_mask` (both additive, `:312–317`) and passed as `attn_mask` to `_ORIG_SDPA`. When capture is *also* on, the collector receives `extra_bias=combined` (`:320`) so the measured distance reflects the ablated distribution.

### Live caller & wiring (the path that actually runs)

The mode is reached through the **modes registry**, *not* the dormant `attention:` config front-end:

1. A config sets `experiment.analysis.modes.attention_distance` (see Configuration).
2. `eval.py:568` reads `modes_cfg`; `:569` guards `if modes_cfg and rank == 0`; `:588` calls `run_modes(modes_cfg, ctx)`. `skip_base_probe: true` short-circuits the base probe loop to 0 epochs at `eval.py:504` (encoder-only — no probe needed for distance capture).
3. `run_modes` (`modes/__init__.py:76`) dispatches to `@register("attention_distance")` (`attention_distance.py:32`).
4. `run` builds an `AttentionDistanceCollector`, wraps the val loop in `attention_hooks(ctx.encoder, collector=collector)`, and encodes `max_batches` clips.

It writes **three** artifacts under `<folder>/attention_distance/`:

| Artifact | Producer | Role | Content |
|---|---|---|---|
| `attention_distance.json` | `collector.finalize()` + `n_batches` (`:61–66`) | data | `spatial_distance[24][16]`, `temporal_distance[24][16]`, `num_layers`, `num_heads`, `rows_per_layer`, `n_batches`. |
| `attention_distance.png` | `_plot_heatmap` (`:76`) | **PRIMARY** — paper **Fig. 3** | Per-head heatmap. `x = Layer (0–23)`, `y = Attention Head (0–15)`, colour = **spatial** distance in patches. `cmap "Blues_r"` (`:94`) so **low distance = DARK blue** → unusually-local heads stand out. Per-cell numeric annotations (toggle via `annotate`, `:100`), colorbar `"Distance (patches)"`, title `"V-JEPA v2-L: Attention Distance Per Head"`. `Z` is transposed to `(H, L)` at `:88`. |
| `attention_distance_layerwise.png` | `_plot_layerwise` (`:115`) | **SECONDARY** — Appendix **Fig. 19** | Dual-axis line plot vs **layer fraction**: red = `Dbar` = mean over heads (`:131`); blue dashed = `S` = **head specialization** (`:132`); PEZ band shaded from `ctx.plot_pez` (`:135–138`). |

**Head specialization `S`** = the standard deviation over the 16 heads of their per-head attention distances *within* a layer — i.e. attention-head diversity. Computed as `statistics.pstdev(row)` per layer (`attention_distance.py:132`). It **spikes** at the Physics Emergence Zone (PEZ) as spatiotemporally-local heads emerge alongside the long-range heads.

### Reproduction result (Blender velocity set)

`attention_distance` was **run** (single-GPU, 10 val batches, `max_batches: 10` → `n_batches = 10` in the output JSON) and **reproduced Fig. 3**: unusually-local low-distance (dark) heads cluster in the **middle layers (~5–13, the PEZ)**, while early and late layers are uniformly long-range. Verified in the JSON: layer-mean `Dbar` dips to **3.73 at layer 9** vs **7.08 at L0** and **6.64 at the last layer**. Output lives under
`configs/analysis/blender_toy_dataset/logs/analysis_vlm/vjepa-blender-attn_distance/attention_distance/` (`.json` + both `.png`). Full write-up is in ch 14.

### SLURM launcher (`z_scripts/run_attn_distance_vjepa.sh`)

Submits the distance run on the Blender set with `--gres=gpu:1`. **Single-GPU is deliberate**: the modes dispatch runs on **rank 0 only** (`eval.py:569`), so post-hoc modes get **no speedup** from multi-GPU — multi-GPU only helps the base probing sweep / feature cache, both skipped here (`skip_base_probe: true`, `cache_features: false`). The script activates the `vjepa2` env and runs `python -m evals.main --fname "$CONFIG" --devices …`.

## Key code

**Pass-through vs instrumented SDPA** — `attention_hooks.py:295`:

```python
def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    s = _stack(); ctx = s[-1] if s else None
    if ctx is None or not ctx["active"]:                 # non-RoPE / masked / unknown-grid
        return _ORIG_SDPA(query, key, value, attn_mask=attn_mask, ...)
    ...
    if collector is not None and collector.enabled_for(layer):
        collector.update(layer, query, key, eff_scale, T, H, W, extra_bias=combined)  # detached
    return _ORIG_SDPA(query, key, value, attn_mask=combined, ...)   # returned value unchanged when combined is None
```

**Streaming distance + float32 cast** — `attention_hooks.py:188`:

```python
@torch.no_grad()
def update(self, layer_idx, q, k, scale, T, H, W, extra_bias=None):
    q = q.to(self.compute_dtype); k = k.to(self.compute_dtype)   # :194-195 (stable in bf16 runs)
    for start in range(0, N, self.query_chunk):                  # :199 stream, never (B,H,N,N)
        logits = (q[:, :, start:stop] @ k.transpose(-2, -1)) * scale
        if extra_bias is not None: logits = logits + extra_bias[..., start:stop, :].to(logits.dtype)
        attn = logits.softmax(dim=-1)
        ds, dt = _dist_rows(T, H, W, device, rows=rows)
        acc_sd += (attn * ds[None, None]).sum(-1).sum(dim=(0, 2)).double().cpu()
        acc_td += (attn * dt[None, None]).sum(-1).sum(dim=(0, 2)).double().cpu()
```

**Ablation `combined` = AND + never-fully-mask** — `attention_hooks.py:141`:

```python
elif mode == "combined":                                  # :143  AND — paper may want UNION
    drop = (ds <= float(spatial)) & (dt <= float(temporal))
...
full = drop.all(dim=1)                                     # :147  un-drop any all-masked row
if full.any(): drop = drop.clone(); drop[full] = False
```

**Heatmap semantics** — `attention_distance.py:88`:

```python
Z = np.array(out["spatial_distance"], dtype=float).T       # (H, L): rows=head, cols=layer
im = ax.imshow(Z, cmap="Blues_r", origin="lower", ...)     # :94  low distance = DARK blue
```

**Layer-mean + head specialization** — `attention_distance.py:131`:

```python
dbar   = [sum(row) / len(row) for row in sd]                       # Dbar = mean over heads
spread = [statistics.pstdev(row) if len(row) > 1 else 0.0 for row in sd]   # S = std over heads
```

**Live capture loop (bounded)** — `attention_distance.py:54`:

```python
n = 0
with attention_hooks(ctx.encoder, collector=collector):
    for data in loader:
        ctx.encode_clip(data)      # SDPA patch captures per-head distance
        n += 1
        if max_batches is not None and n >= max_batches:   # :57-59 caller cap
            break
```

Note the run is bounded **twice**: this explicit `break` in the caller **and** the collector's `enabled_for` gate (`:182`) — the loader is *not* consumed whole.

## Configuration

### Live surface — the modes registry (what runs)

The distance mode is configured under `experiment.analysis.modes.attention_distance`. Real example — `configs/analysis/blender_toy_dataset/vjepa_attn_distance.yaml`:

```yaml
eval_name: analysis_vlm
tag: vjepa-blender-attn_distance
experiment:
  analysis:
    model: vjepa
    task: regression
    regression:
      targets_npy: .../blender_toy/blender_targets.npy
      variables: [{name: direction, cols: [1, 2]}]
    stages: {vision_encoder: all}
    plot_pez: [0.2, 0.4]          # PEZ band drawn on the layerwise plot
    skip_base_probe: true         # encoder-only: distance capture needs no trained probe
    modes:
      attention_distance:         # default-off unless present; absent => existing behavior
        enabled: true
        query_chunk: 512          # memory knob (result-invariant): stream queries
        max_batches: 10           # measure the first 10 val batches (cheap, stable)
  data:
    resolution: 224               # paper geometry: 14x14 patches x 8 tubelets = 1568 tokens
    frames_per_clip: 16
    dataset_val: .../blender_toy/velocity_val.csv
  optimization:
    num_epochs: 0                 # no probe training (encoder-only)
    use_bfloat16: true
    cache_features: false         # attention capture runs fresh forwards, not the cache
```

Mode-level keys read by `attention_distance.py::run`:

| Key | Meaning | Default | Allowed |
|---|---|---|---|
| `enabled` | Run this mode (`false`/`{enabled: false}`/absent ⇒ skipped by `run_modes`). | — | `true` \| `false` |
| `query_chunk` | Query-tile size for streaming softmax. **Memory knob only — result-invariant** (`:48`). | `512` | positive int |
| `max_batches` | Number of val batches averaged; the loop caps at `n >= max_batches` (`:43`,`:58`) and the collector measures only the first `max_batches` encoder forwards. `null` ⇒ all. | `8` (code) / **`10`** (this run) | positive int \| `null` |
| `annotate` | Draw per-cell numeric annotations on the Fig-3 heatmap (`:72`,`:100`). | `true` | `true` \| `false` |

> The code default for `max_batches` is `8`; the executed Blender run used **`10`** (yaml `:35`, confirmed `n_batches=10` in the output JSON).

### Dormant alternative — the `attention:` front-end (not wired)

`attention_hooks.py` also ships `parse_config` (`:342`) + `maybe_context` (`:385`), intended for a separate `experiment.analysis.attention.*` block that would also drive **ablation** sweeps:

```yaml
experiment:
  analysis:
    attention:
      enable: true
      capture: {query_chunk: 512, max_batches: 8}
      ablation:
        spatial:  [1, 2]      # one run per s: drop where patch_dist <= s
        temporal: [1]         # one run per t: drop where tubelet_dist <= t
        combined: [[2, 1]]    # one run per [s, t]: drop where (spatial<=s) AND (temporal<=t)
```

`parse_config` flattens the ablation lists into labeled settings (`spatial_s2`, `temporal_t1`, `combined_s2.0_t1.0`); with `ablation` absent it yields a single `("baseline", None)` so a caller can always iterate. `maybe_context` returns `nullcontext()` when disabled.

> **Not wired.** As of this writing **nothing in the repo calls `parse_config`/`maybe_context`**, and no YAML defines an `attention:` analysis block. The live mode bypasses this front-end entirely and constructs `AttentionDistanceCollector` directly. The ablation path + this schema are implemented and unit-usable but **not yet exercised** — scheduled for **reproduction Phase 5** (`attention_ablation`, C.6/Table 4); see `evals/analysis_vlm/modes/REPRODUCTION_PLAN.md`.

## Invariants & gotchas

- **Default-off / bit-identical.** Feature disabled → `attention_hooks(...)` installs nothing observable; capture-only → the patched SDPA computes softmax as a *detached side computation* and still returns `_ORIG_SDPA(q,k,v, attn_mask=None)`, so the encoder output is **bit-identical to baseline** (verified in-code with `torch.equal`). An installed-but-`active=False` frame (token mask present or grid unknown) also falls through unchanged. **Ablation** deliberately changes the output — that is the experiment.
- **`combined` uses AND** (`:143`): drop only where *both* spatial≤s *and* temporal≤t. **The paper may instead want UNION** (spatial≤s *or* temporal≤t) — its collapse at the mildest `(s=3,t=1)` pair implies UNION. This is the one semantic choice to revisit before publishing; flagged in-code and in `REPRODUCTION_PLAN.md` (risk note).
- **Never-fully-masked row** (`:147`): any query row whose keys are all dropped is left entirely un-dropped, preventing an all-`-inf` softmax row → NaN. Practical trigger: `temporal` with `t ≥ T−1`, or `spatial` with `s` spanning the grid.
- **Self is always dropped** for any threshold ≥ 0 (diagonal distance is 0), so under ablation a token does not attend to itself — unless the whole row would be masked and the safety rule un-drops it.
- **Additive-mask assumption**: combining with a caller `attn_mask` uses `attn_mask + bias`, correct only if the caller mask is *additive* (float bias), not boolean. In the eval path the encoder passes `attn_mask=None`, so this is currently moot — but do not enable ablation on a path passing a boolean `attn_mask`.
- **`dtype` = `query.dtype`**: the bias is built in the query dtype (bf16/fp16 `-inf` is fine).
- **Dense grid only** (`:277`): a frame is `active` only when `T/H_patches/W_patches` are all present **and** the block's token `mask` is `None`. Under masked/token-dropped forwards the frame is inert → pass-through; capture and ablation apply only to the full `T·H·W` grid indexable by `arange`.
- **One push per forward.** SDPA is invoked once per `RoPEAttention.forward`, so stack depth tracks the current block. Non-RoPE attention runs with an empty (or inactive-top) stack → untouched.
- **`max_batches` is double-gated** in the live caller: the explicit `break` (`attention_distance.py:57–59`) **and** `enabled_for` (`:182`). The loader is never fully consumed.
- **Caches persist across contexts.** `_COORD_CACHE`/`_DIST_CACHE` are module globals, *not* cleared on teardown; they key on `(T,H,W,device,dtype,thresholds,mode)` and only grow if you sweep many grid sizes / thresholds in one process (minor memory note, not correctness).
- **`_find_rope_attn` returns modules in `model.modules()` order** (`:239`), which for the ViT is block order; `layer_idx` is assigned from that enumeration (`:264`).
- **Teardown** — `attention_hooks` is a `@contextmanager` (`:248`); the `finally` block (`:327`) always restores: rebind `F.scaled_dot_product_attention = _ORIG_SDPA`, `remove()` every pre/post hook, `del` the temp `_attn_layer_idx`/`_attn_scale` attributes, and reset `_TLS.stack = []`.

## Cross-references

- [12 — Analysis modes subpackage & reproduction roadmap](12-analysis-modes.md) — the `modes` registry, `run_modes`, `AnalysisContext`, the `eval.py` dispatch seam, and Phase 0–5 status.
- [13 — Config reference](13-configs-reference.md) — full schema for `experiment.analysis.modes.attention_distance` and `vjepa_attn_distance.yaml`.
- [14 — Reproduction status & findings](14-reproduction-status-and-findings.md) — the Fig-3 heatmap reproduction on Blender velocity (Dbar dip at L9, PEZ dark-head cluster).
- [02 — `analysis_vlm` harness (eval flow)](02-analysis-vlm-harness.md) — where the encoder, loader closures, and `skip_base_probe` live.
- `evals/analysis_vlm/modes/REPRODUCTION_PLAN.md` — Phase roadmap; the AND-vs-UNION ablation risk note.
