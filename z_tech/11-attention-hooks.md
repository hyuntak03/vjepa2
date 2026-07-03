# Attention hooks (distance + ablation)

## Purpose

`evals/analysis/attention_hooks.py` is an **additive, default-off, zero-core-edit** instrumentation layer for the frozen V-JEPA2 RoPE encoder. It provides two features, both installed at runtime as a monkey-patch on `torch.nn.functional.scaled_dot_product_attention` plus forward-pre/-post hooks on each `RoPEAttention` block:

1. **Distance capture** — per `(layer, head)`, the attention-weighted **spatial** (patch-grid Euclidean) and **temporal** (tubelet `|Δt|`) distance, i.e. `E_i[ Σ_j softmax_attn[h,i,j] · dist(i,j) ]`. This is the standard "mean attention distance" diagnostic, streamed over query chunks so the full `(B,H,N,N)` attention matrix is never materialized.
2. **Local-attention ablation** — an additive attention bias built from the `(T,H,W)` token grid that sets `-inf` on every query/key pair *within* a distance threshold, injected via the SDPA `attn_mask` argument. Used to knock out local heads and measure the downstream effect.

The whole point is that **`src/models/**` is never touched** — the encoder runs upstream code verbatim, and on context exit everything is restored so the model is left byte-for-byte identical.

## What changed vs upstream V-JEPA2

| File | Status | Delta |
|---|---|---|
| `evals/analysis/attention_hooks.py` | **new file** | The entire subsystem: SDPA monkey-patch, `RoPEAttention` pre/post hooks, `AttentionDistanceCollector`, `build_ablation_bias`, and a `parse_config`/`maybe_context` config front-end. |
| `evals/analysis_vlm/modes/attention_distance.py` | **new file** | The only live caller. Registers an `attention_distance` analysis mode that runs **capture-only** (no ablation) over `max_batches` val clips and writes `attention_distance.json` + a dual-axis plot. |
| `src/models/utils/modules.py` | **unchanged** | `git diff 204698b -- src/models/utils/modules.py` is **empty** — byte-identical to upstream. The hooks work purely because upstream `RoPEAttention.forward` already threads the grid dims and `attn_mask` into SDPA (see below). |

> Base for comparison: upstream commit `204698b` (`Fix figure (#143)`).

## How it hooks without editing `src/models/**`

Three upstream facts make the patch possible; none required a fork edit.

**1. `RoPEAttention.forward` already threads the grid + `attn_mask`.** `src/models/utils/modules.py:331`:

```python
def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
```

and its SDPA branch (`src/models/utils/modules.py:372`) calls, on the **already-RoPE-rotated** `q,k`:

```python
if attn_mask is not None or self.use_sdpa:
    with torch.backends.cuda.sdp_kernel():
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
        )
```

So `(T, H_patches, W_patches)` arrive as forward kwargs, and any additive bias placed in `attn_mask` is honored by the real kernel. The rotation is already applied to `q,k` before this call, so the captured softmax logits match what the model actually computes.

**2. The `F` symbol is shared.** `modules.py:8` does `import torch.nn.functional as F` at module scope, and `attention_hooks.py:65` captures and later rebinds that same attribute:

```python
_ORIG_SDPA = F.scaled_dot_product_attention
...
F.scaled_dot_product_attention = patched_sdpa   # observed by modules.py at call time
```

Because `RoPEAttention.forward` resolves `F.scaled_dot_product_attention` at call time, rebinding the function-module attribute is enough — no wrapping of the `nn.Module` needed.

**3. A pre-hook stashes the per-call context; SDPA reads the top of a thread-local stack.** `attention_hooks.py:270` registers `pre_hook(..., with_kwargs=True)` on every `RoPEAttention` module; it reads `T/H_patches/W_patches` from the kwargs and pushes a frame, `post_hook` (`:285`) pops it. SDPA is called **exactly once** inside each `RoPEAttention.forward`, so the top of stack is always the correct layer while the patched SDPA runs.

```python
def pre_hook(mod, args, kwargs):
    T = kwargs.get("T"); H = kwargs.get("H_patches"); W = kwargs.get("W_patches")
    tok_mask = kwargs.get("mask")
    ok = (T is not None and H is not None and W is not None and tok_mask is None)
    _stack().append(dict(active=ok, layer=mod._attn_layer_idx, scale=mod._attn_scale, T=T, H=H, W=W))
```

When no active frame is on the stack (any non-`RoPEAttention` SDPA call — plain `Attention`, `CrossAttention`, `ACRoPEAttention`, probe heads), the patched SDPA (`:298`) is a straight pass-through to `_ORIG_SDPA`.

## Distance capture — `AttentionDistanceCollector`

`attention_hooks.py:161`. Accumulates running float64 sums of expected distance per `(layer, head)`; `finalize()` divides by the per-layer row count.

- **Streaming over query chunks** (`:199`, default `query_chunk=512`): for each chunk it recomputes `logits = (q@kᵀ)·scale`, `softmax`, then `Σ_j attn·dist` for both spatial (`ds`) and temporal (`dt`) — so `(B,H,N,N)` is never held whole. `q,k` are cast to `compute_dtype=float32` (`:190`) for a stable statistic even when the model runs bf16/fp16.
- **Detached side computation** (`@torch.no_grad()`, `:188`). It does *not* feed the encoder — the patched SDPA still returns the original kernel's output. It roughly doubles attention FLOPs for measured batches.
- **Batch gating** (`:178`, `:182`): `note_forward` increments a counter on layer 0; `enabled_for` measures only the first `max_batches` encoder forwards. `enabled_for` returns `True` for all when `max_batches is None`.
- **Scale consistency**: `eff_scale` falls back to `ctx["scale"] = m.scale = head_dim**-0.5` when SDPA is called with `scale=None` (which the encoder does), matching the kernel's internal default since the last dim equals `head_dim`.

`finalize()` (`:217`) output shape:

```json
{"spatial_distance": [[per-head]... per-layer],
 "temporal_distance": [[...]...],
 "num_layers": L, "num_heads": H, "rows_per_layer": [...]}
```

**Geometry.** Token index → `(t,h,w)` must match `RoPEAttention.separate_positions` (`modules.py:316`) exactly: `t = idx // (H·W)`, `h = (idx − t·H·W) // W`, `w = remainder`. This is reproduced in `_coords` (`:90`); pairwise distances come from `_dist_rows` (`:104`) with spatial = Euclidean patch distance, temporal = `|Δt|` tubelets. `_COORD_CACHE`/`_DIST_CACHE` memoize by `(T,H,W,device,…)`.

## Ablation bias — `build_ablation_bias`

`attention_hooks.py:120`. Returns an additive `(1,1,N,N)` bias, `0` = keep / `-inf` = drop, cached by full key.

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

Injected in the patched SDPA (`:305`): the bias is summed with any caller `attn_mask` (both additive) and passed as `attn_mask` to `_ORIG_SDPA`. When capture is *also* on, the collector receives `extra_bias=combined` (`:320`) so the measured distance reflects the ablated distribution.

### Invariants / gotchas for ablation

- **`combined` uses AND** (`:143`): drop only where *both* spatial≤s *and* temporal≤t. **The paper may instead want UNION** (drop where spatial≤s *or* temporal≤t). This is the one semantic choice to revisit before publishing — flagged in-code and here.
- **Never-fully-masked-row** (`:147`): any query row whose keys are all dropped is left entirely un-dropped, preventing an all-`-inf` softmax row → NaN. Practical trigger: `temporal` with `t ≥ T−1`, or `spatial` with `s` spanning the grid.
- **Self is always dropped** for any threshold ≥ 0 (diagonal distance is 0), so under ablation a token does not attend to itself — unless the whole row would be masked and the safety rule un-drops it.
- **Additive-mask assumption**: combining with a caller `attn_mask` uses `attn_mask + bias`. This is only correct if the caller mask is *additive* (float bias), not a boolean mask. In the eval path the encoder passes `attn_mask=None`, so this is currently moot — but do not enable ablation on a path that passes a boolean `attn_mask`.
- **`dtype` = `query.dtype`**: the bias is built in the query dtype (bf16/fp16 `-inf` is fine).

## Config

The module ships a config front-end, `parse_config` (`attention_hooks.py:342`) + `maybe_context` (`:385`), intended for an `experiment.analysis.attention.*` block:

```python
def parse_config(att_cfg, num_layers, num_heads):
    if not att_cfg or not att_cfg.get("enable", False):
        return None
    ...
```

Example YAML (schema as parsed):

```yaml
experiment:
  analysis:
    attention:
      enable: true            # master switch; false/absent -> feature is a no-op
      capture:                # omit for no distance capture
        query_chunk: 512
        max_batches: 8        # measure only first N encoder forwards; null -> all
      ablation:               # omit for capture-only
        spatial:  [1, 2]      # one run per s: drop where patch_dist <= s
        temporal: [1]         # one run per t: drop where tubelet_dist <= t
        combined: [[2, 1]]    # one run per [s, t]: drop where (spatial<=s) AND (temporal<=t)
```

`parse_config` flattens the ablation lists into labeled settings (`spatial_s2`, `temporal_t1`, `combined_s2.0_t1.0`); when `ablation` is absent it yields a single `("baseline", None)` so the caller can always iterate. `maybe_context` returns a `nullcontext()` when disabled.

> **Uncertain / not wired:** as of this writing **nothing in the repo calls `parse_config` or `maybe_context`**, and no YAML defines an `attention:` analysis block (verified by grep). The one live consumer, `evals/analysis_vlm/modes/attention_distance.py`, bypasses the config front-end and constructs `AttentionDistanceCollector` directly from a flat `cfg` (`query_chunk`, `max_batches`, default `8`) with **no ablation**. So the ablation path and the YAML schema above are implemented and unit-usable but not yet exercised by any pipeline.

Live usage (capture-only), `attention_distance.py:54`:

```python
with attention_hooks(ctx.encoder, collector=collector):
    for data in loader:
        ctx.encode_clip(data)   # SDPA patch captures per-head distance
```

## Default-off / bit-identical guarantee

- **Feature disabled** → `parse_config` returns `None`, `maybe_context` returns `nullcontext()`; **nothing is patched**, encoder runs upstream verbatim.
- **Capture-only** → the patched SDPA computes softmax as a *detached side computation* and still returns `_ORIG_SDPA(q,k,v, attn_mask=None)`. The encoder output is **bit-identical** to baseline (the in-code comment notes this is verified with `torch.equal`). Even an installed-but-`active=False` frame (token mask present, or grid unknown) falls through to `_ORIG_SDPA` unchanged.
- **Ablation** → *deliberately* changes the output; that is the experiment.

## Teardown & invariants

`attention_hooks` is a `@contextmanager` (`:248`). Setup tags each `RoPEAttention` with `_attn_layer_idx`/`_attn_scale` and registers the hooks; the `finally` block (`:327`) restores everything:

```python
finally:
    F.scaled_dot_product_attention = _ORIG_SDPA   # un-patch
    for h in handles: h.remove()                  # drop pre/post hooks
    for m in rope_mods:                            # strip the temp attributes
        del m._attn_layer_idx; del m._attn_scale
    _TLS.stack = []                                # reset thread-local
```

Other invariants worth remembering:

- **Dense grid only.** A frame is `active` only when `T/H_patches/W_patches` are all present **and** the block's token `mask` is `None` (`:277`). Under masked/token-dropped forwards the frame is inert → pass-through, so capture and ablation apply only to the full `T·H·W` grid indexable by `arange`.
- **One push per forward.** SDPA is invoked once per `RoPEAttention.forward`, so the stack depth tracks the current block. Non-RoPE attention runs with an empty (or inactive-top) stack → untouched.
- **Caches persist across contexts.** `_COORD_CACHE`/`_DIST_CACHE` are module globals and are *not* cleared on teardown. They key on `(T,H,W,device,dtype,thresholds,mode)`, so they only grow if you sweep many grid sizes / thresholds in one process (minor memory note, not a correctness issue).
- **`_find_rope_attn` returns modules in `model.modules()` order**, which for the ViT is block order; `layer_idx` is assigned from that enumeration (`:264`).
