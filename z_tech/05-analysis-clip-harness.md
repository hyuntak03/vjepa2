# 05 — analysis (clip / V-JEPA) harness

> The lean, V-JEPA-only **clip** probing harness: a multilayer wrapper returns each requested encoder block as a **separate** tensor, and one independent probe head is trained per `(layer × probe-spec)` in a single shared forward pass, yielding a `[layer × probe]` validation-accuracy matrix.

## Purpose

Layer-wise linear/attentive **probing** of a *frozen* V-JEPA2 ViT encoder. For a chosen set of encoder blocks, extract each block's token features **separately**, attach one independent probe head per `(layer × probe-spec)`, train them all in a single forward pass, and report a `[layer × probe]` validation-accuracy matrix (plus an optional `x=layer, y=best-val-acc` plot). Answers: *"which depth of the encoder linearly encodes property X?"*

This is the **clip path**: it decodes video clips and feeds them through the ViT itself **every step** (no feature cache). It is the original, V-JEPA-only variant.

The sibling `evals/analysis_vlm` harness (see [§02](02-analysis-vlm-harness.md)) is the unified, **feature-caching**, **VLM-capable**, **regression-capable** successor that also hosts the post-hoc analysis-**modes** subsystem. Everything in `evals/analysis` is the lean predecessor; the two harnesses share `probes.py` and `plotting.py` *by copy* (each package has its own file, not a shared import).

**See also — where the rest of the story lives:** the `evals/analysis` package now *also* physically hosts `attention_hooks.py`, the runtime SDPA-patch machinery that powers the `attention_distance` mode. That file is **out of scope for this clip harness** (the clip harness never calls it) but it is documented in full in [§11 attention hooks](11-attention-hooks.md); the mode dispatch that consumes it lives in [§12 analysis modes](12-analysis-modes.md), and what it reproduced is in [§14 reproduction status](14-reproduction-status-and-findings.md).

## What changed vs upstream V-JEPA2

Baseline: upstream commit `204698b`. This subsystem is **almost entirely new files**; it reuses stock machinery by *import only*, so the standard eval code paths are untouched.

### New files (the `evals/analysis/` package)

| File | Lines | Role |
|---|---:|---|
| `evals/analysis/eval.py` | 395 | Harness `main()` + build-heads + `run_one_epoch` train/eval loop + checkpoint I/O. |
| `evals/analysis/modelcustom/vit_encoder_multilayer.py` | 161 | Wrapper returning a **per-layer feature list** (one `(B,N,D)` per requested block). |
| `evals/analysis/probes.py` | 93 | `build_probe` → `LinearProbe` / `AttentiveClassifier`; `probe_name` for stable labels. |
| `evals/analysis/plotting.py` | 153 | `plot_layer_val_acc` — shared layer-fraction plotter (its full feature surface is documented in [§07](07-plotting.md)). |
| `evals/analysis/attention_hooks.py` | 390 | **Not used by the clip harness.** Runtime SDPA monkey-patch + RoPEAttention hooks (attention distance capture + ablation bias) that feed the `attention_distance` **mode**. Documented in [§11](11-attention-hooks.md). Added Jul 3 (commit `4c76f65`/`c296428`). |
| `evals/analysis/__init__.py`, `.../modelcustom/__init__.py` | 0 | Empty package markers. |

> **Why `attention_hooks.py` is listed but out of scope here:** the section opener presents this table as the package inventory, and the file physically lives in `evals/analysis/`. But it is imported by the *modes* dispatch (§12), never by `eval.py` in this package. This file names it and forwards the reader; it does **not** absorb any distance/heatmap content.

### Modified upstream files (exact additive deltas)

| File | Kind | Exact delta (`git diff 204698b`) | Default-off guarantee |
|---|---|---|---|
| `evals/video_classification_frozen/eval.py` | **modified** | **+2 lines**: a `uniform_sampling=False` kwarg added to `make_dataloader`'s signature and threaded into the `init_data(...)` call. | New param defaults `False`; unrelated to this harness (belongs to the data-pipeline fix, [§06](06-data-pipeline-changes.md)). |
| `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py` | **comment-only** | 3 insertions / 2 deletions — three `#!` annotation comments (`#! vit_encoder`, `#! x.shape…`, `#! self.model(x)…`) replacing two blank lines. **No behavior change.** | Comments only; the wrapper is byte-equivalent at runtime. |
| `src/models/vision_transformer.py` | **comment-only (fork)** | The `out_layers` multi-layer tap is **already upstream** in `204698b` (verified: `out_layers` occurs 0× in the fork diff, 4× in baseline). The fork adds only a Korean `#!` comment near `patch_embed`. | The tap this harness relies on is stock; the fork adds nothing to it. |

### Routing

No change to `evals/main.py` or `evals/scaffold.py` is needed. `scaffold.main` dynamically imports `evals.{eval_name}.eval`, so a config with `eval_name: analysis` lands in `evals/analysis/eval.py` automatically (`evals/scaffold.py:15-19`).

The `out_layers` plumbing in the ViT (`src/models/vision_transformer.py:205-209`) already exists upstream and is shared with the `_multilevel` wrapper — **this fork does not add it**, it only consumes it.

## Design & data flow

```
config (eval_name: analysis)
   │  scaffold dynamic import
   ▼
evals/analysis/eval.py :: main()
   │  _resolve_layers("all" | [ints]) ─────────────► out_layers injected into wrapper_kwargs
   │
   ├─ init_module(...vit_encoder_multilayer)  ── frozen ViT, out_layers set ──┐
   │                                                                          │
   ├─ build heads: for layer in layers × for spec in probes → one probe each  │
   │       (LinearProbe | AttentiveClassifier)                                │
   │                                                                          │
   ├─ init_opt(list-of-classifiers, list-of-opt-kwargs)                       │
   │       → per-head AdamW + cosine LR + cosine WD + GradScaler              │
   ▼                                                                          ▼
run_one_epoch:  clips ──► encoder(clips, clip_indices) ──► List[(B,N,D)] over layers
                            (torch.no_grad + detach: encoder frozen)
                                      │
                 logits = [ head(feats[head.layer_pos]) for head in heads ]   ← one forward per head
                 loss   = sum(CrossEntropy(logit, label))                      ← ONE summed backward
                            │
                 acc accumulated on-GPU → AllReduceSum once/epoch
   ▼
outputs: log_r{rank}.csv, summary.json ([layer×probe] matrix), latest.pt, layer_val_acc.png
```

The single load-bearing idea: **encode once, feed every head.** One frozen ViT forward produces the whole layer list; every probe reads its own slice of that list. Because features are detached, the summed backward touches only probe parameters.

### The multilayer wrapper — and how it differs

`evals/analysis/modelcustom/vit_encoder_multilayer.py` builds the ViT with the requested block indices and keeps each block's output **as a separate tensor**.

The ViT collects intermediates when `out_layers` is set — note **every** returned layer is passed through the *same final* `self.norm` (`src/models/vision_transformer.py:205-209`):

```python
if self.out_layers is not None and i in self.out_layers:
    outs.append(self.norm(x))
...
if self.out_layers is not None:
    return outs           # list, one (B', N, D) per requested block
```

`MultiLayerClipAggregation.forward` runs the multi-clip / multi-view unrolling once **per layer** and returns a list, same order as `out_layers` (`vit_encoder_multilayer.py:121-161`):

```python
layer_outputs = self.model(x)                       # list over layers, each (B', N, D)
if not isinstance(layer_outputs, list):
    layer_outputs = [layer_outputs]
...
return [multiviews_postprocess(lo) for lo in layer_outputs]   # List[(B, N, D)]
```

Contrast with the two `video_classification_frozen` wrappers:

| Wrapper | `out_layers` passed? | ViT returns | Post-processing | Probe count |
|---|---|---|---|---|
| `vit_encoder_multiclip.py` | no | single tensor (final block, normed) | one `(B,N,D)` | **last layer only** |
| `vit_encoder_multiclip_multilevel.py` | yes | list | `torch.cat(outputs, dim=1)` → tokens **concatenated** into one long sequence (`…multilevel.py:123`) | **one probe over the concat** |
| `vit_encoder_multilayer.py` (this) | yes | list | per-layer `multiviews_postprocess`, **kept separate** | **one probe per layer** |

So `_multilevel` fuses depth into a single richer token sequence (you cannot attribute accuracy to a specific block); **`_multilayer` keeps depths apart** so accuracy is attributed to a specific block — the whole point of a layer-wise scan.

`init_module` (`vit_encoder_multilayer.py:36-88`) **requires** `out_layers` and raises if missing (`:53-55`). It pops two keys before constructing the aggregation module (`:76-86`): `out_layers` (passed explicitly to both the ViT ctor and the wrapper) and an optional compute-`dtype` knob (e.g. `float32` for CPU debug) used to `.to(dtype=…)` the assembled module. The wrapper exposes `embed_dim` / `num_heads` for the probes.

## Key code

### `main()` control flow (`evals/analysis/eval.py`)

**1. Resolve layers** — `experiment.analysis.layers` may be a list or the literal `"all"`. `_resolve_layers` (`eval.py:77-83`) expands `"all"` to `range(depth)` via a `model_name → depth` table `_VIT_DEPTH` (`eval.py:68-74`):

```python
_VIT_DEPTH = {
    "vit_large": 24,
    "vit_huge": 32,
    "vit_giant": 40,
    "vit_giant_xformers": 40,
    "vit_gigantic": 48,
}
```

The resolved list is injected into `wrapper_kwargs["out_layers"]` (`eval.py:113-114`); the config's `wrapper_kwargs` leaves it blank.

**2. Build the frozen encoder** via the stock `init_module` (`eval.py:172-180`) — same loader the normal eval uses; it just points `module_name` at `…vit_encoder_multilayer`.

**3. Build heads** — nested loop over `layers × probe_specs`, one probe module each (`eval.py:190-198`):

```python
for layer_pos, layer in enumerate(layers):
    for spec in probe_specs:
        module = build_probe(spec, embed_dim=encoder.embed_dim, num_classes=num_classes,
                             use_activation_checkpointing=True).to(device)
        name = f"L{layer:02d}_{probe_name(spec)}"
        heads.append(dict(name=name, layer=layer, layer_pos=layer_pos, module=module))
```

`layer_pos` (not `layer`) indexes into the wrapper's returned list — layer `23` might be `layer_pos` `3` if `layers=[5,11,17,23]`.

**4. Optimizers** — reuses stock `init_opt` with a **list** of classifiers and a **list** of per-head opt kwargs (`eval.py:221-227`), yielding one AdamW + cosine LR + cosine WD schedule + GradScaler **per head**. Each `probe.optimization` overlays `optimization.default_head` via `_opt_kwargs` (`eval.py:137-147`).

**5. Train/eval loop** (`run_one_epoch`, `eval.py:317-380`) — encode once, feed every head (`eval.py:339-354`):

```python
with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_bfloat16):
    ...
    with torch.no_grad():
        feats = encoder(clips, clip_indices)   # list over layers, each (B, N, D)
    feats = [f.detach() for f in feats]

    logits = [h["module"](feats[h["layer_pos"]]) for h in heads]
    losses = [criterion(o, labels) for o in logits]
...
loss_total = sum(losses)                        # ONE combined backward; heads are independent
```

Because `feats` are detached, the summed backward only touches probe params — heads do not interfere and the encoder gets no gradient. Accuracy is accumulated **on-GPU** and all-reduced **once per epoch** (`AllReduceSum`, `eval.py:377-380`) to avoid per-iter host syncs.

> **`use_bfloat16` actually drives a *float16* autocast.** Despite the name, the autocast is `dtype=torch.float16` (`eval.py:339`); the flag only toggles `enabled=`. The same flag also decides whether `init_opt` returns real `GradScaler`s (needed for fp16) — see the scaler gotcha below.

**6. Outputs** (rank 0):

| File | Written | Content |
|---|---|---|
| `log_r{rank}.csv` | per epoch | per-head `{name}_train`, `{name}_val` columns (`eval.py:230-234, 294-298`). |
| `summary.json` | per epoch | `{epoch, num_epochs, layers, head_names, val_acc, train_acc, best_val_acc}` — current + best-so-far matrix (`eval.py:300-303`). |
| `latest.pt` | per epoch | classifier state dicts + optimizer + scaler + epoch + head names + layers (`eval.py:245-257`). |
| `layer_val_acc.png` | **once, after training** | if `analysis.plot` (`eval.py:310-314`). |

All under `<folder>/analysis/[<tag>/]`.

### Probes (`evals/analysis/probes.py`)

`build_probe(spec, embed_dim, num_classes, use_activation_checkpointing=False)` (`probes.py:56-81`) dispatches on `type`:

- **`linear`** → `LinearProbe` (`probes.py:25-53`): pool tokens (`mean` | `max` | `meanmax`), optional `pre_norm` LayerNorm over the pooled feature, then `nn.Linear`. `pre_norm: true` is the **default** and is **recommended for cross-layer comparison** — different blocks have very different feature scales (they all share the same final `norm`, but residual magnitude grows with depth). For `meanmax` the input dim is `2·embed_dim`.
- **`attentive`** → the stock `AttentiveClassifier` (cross-attention pooling + Linear), identical to `video_classification_frozen`. Depth key is `num_probe_blocks` (alias `depth`, default `1`); `num_heads` defaults `16`. Carries its own internal norm (`pre_norm` is irrelevant to it).

`probe_name(spec)` (`probes.py:84-93`) gives the stable column/label suffix: an explicit `name` if given, else `linear-{pooling}` or `attentive-d{num_probe_blocks}`.

### How this shares `probes.py` & `plotting.py`

Both `evals/analysis` and `evals/analysis_vlm` carry their **own copy** of `probes.py` and `plotting.py` (independent files, not a shared module). The VLM copies are supersets: `analysis_vlm/probes.py` adds regression/temporal head types ([§04](04-probes-regression-nanmask.md)); `analysis_vlm/plotting.py` is where the R²/PEZ/elbow logic was first grown. This `evals/analysis/plotting.py` was later extended (Jul 1) to the **same** superset surface (`_elbow_x`, `r2` metric, PEZ band, `num_classes` chance line), so the two plotters are now feature-parallel — **but the clip harness only exercises the accuracy path** (see the plotting note below).

## Configuration

Real example: `configs/z_tak_attentive_probing/R2R_4way_analysis.yaml` (the clip-harness config; note `eval_name: analysis`, module `…vit_encoder_multilayer`). Trimmed:

```yaml
eval_name: analysis                       # -> evals/analysis/eval.py (scaffold dynamic import)
folder: /…/configs/z_tak_attentive_probing/logs   # outputs land in <folder>/analysis/<tag>/
num_workers: 6
resume_checkpoint: false
val_only: false
tag: r2r-4way-analysis

experiment:
  analysis:
    layers: [5, 11, 17, 23]               # encoder block indices (0-indexed; 23 = last for vit_large). Or "all".
    plot: true                            # save layer_val_acc.png ONCE at the end
    probes:                               # one head per (layer × probe) is trained simultaneously
      - type: linear
        pooling: mean                     # mean | max | meanmax
        pre_norm: true
        optimization: { lr: 0.01, weight_decay: 0.0, warmup: 1.0 }
      - type: attentive
        num_heads: 16
        num_probe_blocks: 4
        optimization: { lr: 0.002, weight_decay: 0.01, warmup: 2.0 }

  data:
    dataset_type: VideoDataset
    dataset_train: /…/R2R_4way_1500_shape_color_train.csv   # "<video_path> <int_label>", space-sep, no header
    dataset_val:   /…/R2R_4way_1500_shape_color_val.csv
    num_classes: 4
    frames_per_clip: 32                   # even (tubelet_size=2)
    frame_step: 1
    num_segments: 1
    num_views_per_segment: 1
    resolution: 256                       # match the fpc64-256 checkpoint

  optimization:
    batch_size: 8
    num_epochs: 20
    use_bfloat16: true                    # NAME says bf16, but drives a float16 AMP autocast
    default_head:                         # fills any value a probe.optimization omits
      start_lr: 0.0
      final_lr: 0.0
      final_weight_decay: 0.01

model_kwargs:
  checkpoint: /…/vjepa2-vitl-fpc64-256/…/model.pth
  module_name: evals.analysis.modelcustom.vit_encoder_multilayer
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder      # EMA weights (eval standard) | encoder = online
      model_name: vit_large               # also used to expand layers: "all" (depth 24)
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
    # out_layers is injected automatically from experiment.analysis.layers
```

Run (single-GPU debug):

```bash
python -m evals.main --fname configs/z_tak_attentive_probing/R2R_4way_analysis.yaml \
       --devices cuda:0 --debugmode True
```

### Config key reference

| Key | Meaning | Default | Allowed / notes |
|---|---|---|---|
| `eval_name` | scaffold dispatch target | — (required) | must be `analysis` for this harness |
| `experiment.analysis.layers` | encoder blocks to tap | — (required) | list of 0-based ints, or `"all"` (needs a `_VIT_DEPTH` model) |
| `experiment.analysis.probes` | probe specs (one head per layer each) | — (required) | non-empty list of `{type: linear|attentive, …}` |
| `experiment.analysis.plot` | save `layer_val_acc.png` once at end | `false` | bool |
| `probe.type` | probe kind | `attentive` | `linear` \| `attentive` |
| `probe.pooling` (linear) | token pooling | `mean` | `mean` \| `max` \| `meanmax` |
| `probe.pre_norm` (linear) | LayerNorm before Linear | `true` | keep `true` for cross-layer scans |
| `probe.num_heads` (attentive) | attention heads | `16` | int |
| `probe.num_probe_blocks` (attentive) | cross-attn depth (alias `depth`) | `1` | int |
| `probe.name` | explicit label override | derived | string |
| `probe.optimization` | per-head opt overlay on `default_head` | `{}` | `lr, weight_decay, final_weight_decay, start_lr, final_lr, warmup` |
| `optimization.use_bfloat16` | enable AMP (float16 autocast) + real GradScaler | — | bool; **name is misleading** |
| `optimization.default_head` | opt defaults filling omitted probe values | `{}` | see `_opt_kwargs` |
| `wrapper_kwargs.out_layers` | **do not set** — injected from `analysis.layers` | injected | overwritten if set by hand |
| `wrapper_kwargs.dtype` | optional compute-dtype cast (CPU debug) | none | e.g. `float32` |

## Invariants & gotchas

- **Default-off by construction.** Nothing here runs unless a config sets `eval_name: analysis`. The only upstream edits are a `uniform_sampling` passthrough (+2 lines, its own default `False`) and comment-only `#!` lines in `vit_encoder_multiclip.py` / `vision_transformer.py` — stock eval behavior is byte-unchanged.
- **`out_layers` is required** by the multilayer wrapper; missing → `ValueError` (`vit_encoder_multilayer.py:53-55`). Leave it out of the config's `wrapper_kwargs`; the harness injects it from `analysis.layers` (`eval.py:113-114`). Setting it by hand is overwritten anyway.
- **`layers: "all"` needs a known depth.** Only the five models in `_VIT_DEPTH` are recognized (`eval.py:68-74`: `vit_large=24`, `vit_huge=32`, `vit_giant=40`, `vit_giant_xformers=40`, `vit_gigantic=48`); an unknown `model_name` with `"all"` raises — pass an explicit list. Indices are 0-based and refer to the output **after** that block.
- **All returned layers share the same final `norm`.** The ViT applies `self.norm` to *every* collected intermediate (`vision_transformer.py:206`), so probes see post-final-LayerNorm features, not per-block-normed ones. Keep `pre_norm: true` on linear probes to neutralize residual cross-layer scale differences.
- **Heads are truly independent.** `feats` are `.detach()`ed and the loop uses one summed backward (`eval.py:347-354`); this is valid *only because* heads share no parameters and never backprop into the frozen encoder. Do not add a head that reuses another head's tensors.
- **Shared-scaler assumption.** `run_one_epoch` uses `scaler[0]` for all heads (`eval.py:327`), relying on `init_opt` producing identical `GradScaler`s. Fine as-is; a heterogeneous scaler list would break it. With `use_bfloat16: false`, `scaler[0] is None` and the loop takes the unscaled `.backward()` branch.
- **`frames_per_clip` must be even** (tubelet_size 2). Token count per layer is `T·S` with `T = F/2`.
- **Single-GPU debug** skips DDP gracefully when no process group exists (`eval.py:184-186`); multi-GPU wraps each head in `DistributedDataParallel(static_graph=True)` (`eval.py:194-195`).
- **`val_only: true`** evaluates once and breaks after the first epoch (`eval.py:269-270, 306-307`); **resume** replays scheduler/WD steps to re-sync LR (`eval.py:237-243`).
- **Cost scales with `len(layers) × len(probes)`** heads, but the encoder forward runs **once per batch** regardless. The clip path **re-encodes every epoch** — there is no feature cache (that optimization lives in `analysis_vlm`, [§03](03-feature-caching-and-pooling.md)). This is the main reason to prefer the VLM harness for a full all-layer sweep.
- **The plot is nearly inert in the clip path.** The clip call passes **only** `heads, best_val, out_path, subtitle` (`eval.py:310-314`) — it does **not** pass `num_classes`, `metric`, `target_label`, or `pez`. Therefore in `plot_layer_val_acc`:
  - the **random-chance line** is gated on `if num_classes:` (`plotting.py:137`) and `num_classes` defaults `None` → **never drawn**;
  - `metric` defaults `'accuracy'` → fixed `0–120%` y-axis, no R² axis;
  - **no PEZ band** (`pez=None`) and **no elbow** (the elbow is only computed for a `probe_label == "direction"` series (`plotting.py:104`), which the clip flow never produces — its series labels are probe types like `linear-mean`, `attentive-d4`).
  So the clip plot is just one **peak-starred** accuracy line per probe with a layer-fraction x-axis. The chance-line / R² / PEZ / elbow machinery in `plotting.py` exists for the regression/VLM harness, which passes those extra kwargs — see [§07 plotting](07-plotting.md). Per-line label is parsed from the head name (`name.split("_", 1)[1]`, the probe part after `L{layer}_`).

## Cross-references

- [§02 `analysis_vlm` harness](02-analysis-vlm-harness.md) — the unified successor (feature cache, VLM backends, regression, modes); shares `probes.py`/`plotting.py` by copy.
- [§03 Feature caching & pooling](03-feature-caching-and-pooling.md) — the encode-once cache the clip path lacks.
- [§04 Probes, regression & NaN-masking](04-probes-regression-nanmask.md) — the VLM probe superset that extends this file's two head types.
- [§06 Data-pipeline changes](06-data-pipeline-changes.md) — the `uniform_sampling` +2-line delta touching `video_classification_frozen/eval.py`.
- [§07 Plotting](07-plotting.md) — the full `plot_layer_val_acc` feature surface (chance line, elbow, PEZ, R²) that is inert in the clip path.
- [§11 Attention hooks](11-attention-hooks.md) — `evals/analysis/attention_hooks.py`, the SDPA-patch machinery hosted in this package but unused by the clip harness.
- [§12 Analysis modes](12-analysis-modes.md) — the `experiment.analysis.modes` dispatch (in `analysis_vlm`) that consumes `attention_hooks.py`.
- [§14 Reproduction status](14-reproduction-status-and-findings.md) — what the probing scans (and the `attention_distance` mode) actually reproduced.
