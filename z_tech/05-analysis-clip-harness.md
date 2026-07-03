# analysis (clip / V-JEPA layer-wise) harness

## Purpose

Layer-wise linear/attentive **probing** of a *frozen* V-JEPA2 ViT encoder. For a
chosen set of encoder blocks, extract each block's token features **separately**,
attach one independent probe head per `(layer × probe-spec)`, train them all in a
single forward pass, and report a `[layer × probe]` validation-accuracy matrix (plus
an optional `x=layer, y=best-val-acc` plot). Answers "which depth of the encoder
linearly encodes property X?".

This is the **clip path**: it feeds decoded video clips through the ViT itself every
step (no feature cache). The sibling `evals/analysis_vlm` harness is the unified,
feature-caching, VLM-capable variant; this one is the lean V-JEPA-only original.

## What changed vs upstream V-JEPA2

Baseline: upstream commit `204698b`. This subsystem is **almost entirely new files**;
it reuses stock machinery by *import only*, so the standard eval code paths are
untouched.

| File | Status | Delta |
|---|---|---|
| `evals/analysis/eval.py` | **new** (395 lines) | The harness `main()` + train/eval loop. |
| `evals/analysis/modelcustom/vit_encoder_multilayer.py` | **new** (161) | Wrapper returning per-layer feature list. |
| `evals/analysis/probes.py` | **new** (93) | `build_probe` → `LinearProbe` / `AttentiveClassifier`. |
| `evals/analysis/plotting.py` | **new** (153) | `plot_layer_val_acc` (layer-fraction curves). |
| `evals/analysis/__init__.py`, `.../modelcustom/__init__.py` | **new** (empty) | Package markers. |
| `evals/video_classification_frozen/eval.py` | **modified** | `+2` lines only: a `uniform_sampling=False` kwarg threaded through `make_dataloader` (unrelated to this harness; default-off). |
| `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py` | **modified** | Comments only (`#!` annotations); no behavior change. |

Routing needs **no** change to `evals/main.py` or `evals/scaffold.py`: `scaffold.main`
dynamically imports `evals.{eval_name}.eval`, so a config with `eval_name: analysis`
lands in `evals/analysis/eval.py` automatically (`evals/scaffold.py:15-19`).

The `out_layers` plumbing in the ViT itself (`src/models/vision_transformer.py:205-209`)
already exists upstream and is shared with the `_multilevel` wrapper — this fork does
not add it.

## The multilayer wrapper — and how it differs

`evals/analysis/modelcustom/vit_encoder_multilayer.py` builds the ViT with the requested
block indices and keeps each block's output **as a separate tensor**.

The ViT collects intermediates when `out_layers` is set — note every returned layer is
passed through the **same final `self.norm`** (`src/models/vision_transformer.py:205-209`):

```python
if self.out_layers is not None and i in self.out_layers:
    outs.append(self.norm(x))
...
if self.out_layers is not None:
    return outs           # list, one (B', N, D) per requested block
```

`MultiLayerClipAggregation.forward` runs the multi-clip / multi-view unrolling once **per
layer** and returns a list, same order as `out_layers`
(`evals/analysis/modelcustom/vit_encoder_multilayer.py:130-161`):

```python
layer_outputs = self.model(x)                       # list over layers, each (B', N, D)
if not isinstance(layer_outputs, list):
    layer_outputs = [layer_outputs]
...
return [multiviews_postprocess(lo) for lo in layer_outputs]   # List[(B, N, D)]
```

Contrast with the two `video_classification_frozen` wrappers:

| Wrapper | `out_layers` | ViT returns | Post-processing | Probe count |
|---|---|---|---|---|
| `vit_encoder_multiclip.py` | not passed | single tensor (final block, normed) | one `(B,N,D)` | **last layer only** |
| `vit_encoder_multiclip_multilevel.py` | passed | list | `torch.cat(outputs, dim=1)` → tokens **concatenated** into one long sequence (`...multilevel.py:123`) | **one probe over concat** |
| `vit_encoder_multilayer.py` (this) | passed | list | per-layer `multiviews_postprocess`, **kept separate** | **one probe per layer** |

So `_multilevel` fuses depth into a single richer token sequence; **`_multilayer` keeps
depths apart** so accuracy can be attributed to a specific block.

`init_module` requires `out_layers` and raises if missing
(`vit_encoder_multilayer.py:53-55`). It also pops two keys before constructing the
aggregation module (`vit_encoder_multilayer.py:76-86`): `out_layers` (passed explicitly)
and an optional compute-`dtype` knob (e.g. `float32` for CPU debug) used to `.to(dtype=…)`
the assembled module. The wrapper exposes `embed_dim` / `num_heads` for the probes.

## `eval.py` control flow

`main(args_eval)` in `evals/analysis/eval.py`:

1. **Resolve layers** — `experiment.analysis.layers` may be a list or the literal
   `"all"`. `_resolve_layers` (`eval.py:77-83`) expands `"all"` to `range(depth)` via a
   `model_name → depth` table (`_VIT_DEPTH`, `eval.py:68-74`; `vit_large=24`,
   `vit_huge=32`, `vit_giant=40`, `vit_gigantic=48`). The resolved list is injected into
   `wrapper_kwargs["out_layers"]` (`eval.py:113-114`) — the config's `wrapper_kwargs`
   leaves it blank.

2. **Build frozen encoder** via the stock `init_module` from
   `evals.video_classification_frozen.models` (`eval.py:172-180`) — same loader the
   normal eval uses; it just points `module_name` at `...vit_encoder_multilayer`.

3. **Build heads** — nested loop over `layers × probe_specs`, one probe module each
   (`eval.py:190-198`):

   ```python
   for layer_pos, layer in enumerate(layers):
       for spec in probe_specs:
           module = build_probe(spec, embed_dim=encoder.embed_dim, num_classes=num_classes, ...)
           name = f"L{layer:02d}_{probe_name(spec)}"
           heads.append(dict(name=name, layer=layer, layer_pos=layer_pos, module=module))
   ```

   `layer_pos` (not `layer`) indexes into the wrapper's returned list.

4. **Optimizers** — reuses stock `init_opt` with a **list** of classifiers and a **list**
   of per-head opt kwargs (`eval.py:221-227`), yielding one AdamW + cosine LR + cosine WD
   schedule + GradScaler per head. Each `probe.optimization` overlays
   `optimization.default_head` via `_opt_kwargs` (`eval.py:137-147`).

5. **Train/eval loop** (`run_one_epoch`, `eval.py:317-380`) — the crucial efficiency
   point: **encode once, feed every head** (`eval.py:345-354`):

   ```python
   with torch.no_grad():
       feats = encoder(clips, clip_indices)   # list over layers, each (B, N, D)
   feats = [f.detach() for f in feats]
   logits = [h["module"](feats[h["layer_pos"]]) for h in heads]
   losses = [criterion(o, labels) for o in logits]
   ...
   loss_total = sum(losses)                    # one combined backward; heads are independent
   ```

   Because `feats` are detached, the summed backward only touches probe params — heads do
   not interfere and the encoder gets no gradient. Accuracy is accumulated on-GPU and
   all-reduced **once per epoch** (`AllReduceSum`, `eval.py:377-380`) to avoid per-iter
   host syncs.

6. **Outputs** (rank 0) — `log_r{rank}.csv` (per-head train/val each epoch),
   `summary.json` (current + best-so-far matrix, `eval.py:300-303`), `latest.pt`
   checkpoint, and — after training, if `analysis.plot` — `layer_val_acc.png`
   (`eval.py:310-314`). All under `<folder>/analysis/[<tag>/]`.

## Probes

`build_probe(spec, embed_dim, num_classes, …)` (`evals/analysis/probes.py:56-81`)
dispatches on `type`:

- **`linear`** → `LinearProbe`: pool tokens (`mean` | `max` | `meanmax`), optional
  `pre_norm` LayerNorm, then `nn.Linear`. `pre_norm: true` is the default and is
  **recommended for cross-layer comparison** — different blocks have very different
  feature scales (`probes.py:25-53`).
- **`attentive`** → the stock `AttentiveClassifier` (cross-attention pooling + Linear),
  identical to `video_classification_frozen`. Depth key is `num_probe_blocks` (alias
  `depth`).

`probe_name(spec)` gives the stable column/label suffix (`linear-mean`, `attentive-d4`,
or an explicit `name`).

## Plotting / outputs

`plot_layer_val_acc` (`evals/analysis/plotting.py:36`) draws one line per probe series,
**x-axis as layer fraction 0..1** (block index / deepest index), a random-chance line
(`100/num_classes`), a peak star per curve, and — for a `direction` series — an "elbow"
(saturation) marker via the parameter-free max-chord-distance heuristic (`_elbow_x`,
`plotting.py:17-33`). It also supports an `r2` metric mode and a shaded "PEZ" band; those
are exercised mainly by the regression / VLM harnesses, not the default clip flow. For
the clip harness the per-line label is parsed from the head name (`name.split("_", 1)[1]`,
i.e. the probe part after `L{layer}_`).

## Config

Real example: `configs/z_tak_attentive_probing/R2R_4way_analysis.yaml` (the clip-harness
config; note `eval_name: analysis`, module `…vit_encoder_multilayer`). Trimmed:

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
    plot: true                            # save layer_val_acc.png once at the end
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
    use_bfloat16: true                    # float16 AMP autocast
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

```
python -m evals.main --fname configs/z_tak_attentive_probing/R2R_4way_analysis.yaml \
       --devices cuda:0 --debugmode True
```

## Gotchas / invariants / default-off

- **Default-off by construction.** Nothing here runs unless a config sets
  `eval_name: analysis`. The only upstream file edits are a `uniform_sampling` passthrough
  (+2 lines, its own default `False`) and comment-only lines in `vit_encoder_multiclip.py`
  — stock eval behavior is unchanged.
- **`out_layers` is required** by the multilayer wrapper; missing → `ValueError`
  (`vit_encoder_multilayer.py:53-55`). Leave it out of the config's `wrapper_kwargs`; the
  harness injects it from `analysis.layers` (`eval.py:113-114`). Setting it by hand in the
  config would be overwritten anyway.
- **`layers: "all"` needs a known depth.** Only the models in `_VIT_DEPTH` are recognized
  (`eval.py:68-74`); an unknown `model_name` with `"all"` raises — pass an explicit list
  instead. Indices are 0-based and refer to the output **after** that block.
- **All returned layers share the same final `norm`.** The ViT applies `self.norm` to
  every collected intermediate (`vision_transformer.py:206`), so probes see post-final-
  LayerNorm features, not per-block-normed ones. Keep `pre_norm: true` on linear probes to
  neutralize residual cross-layer scale differences.
- **Heads are truly independent.** `feats` are `.detach()`ed and the loop uses one summed
  backward; this is valid *only because* heads share no parameters and never backprop into
  the frozen encoder. Do not add a head that reuses another head's tensors.
- **Shared-scaler assumption.** `run_one_epoch` uses `scaler[0]` for all heads
  (`eval.py:327`), relying on `init_opt` producing identical `GradScaler`s. Fine as-is;
  a heterogeneous scaler list would break it.
- **`frames_per_clip` must be even** (tubelet_size 2). Token count per layer is
  `T·S` with `T = F/2`.
- **Single-GPU debug** skips DDP gracefully when no process group exists
  (`eval.py:184-186`); multi-GPU wraps each head in `DistributedDataParallel(static_graph=True)`.
- **`val_only: true`** evaluates once and breaks after the first epoch; **resume** replays
  scheduler/WD steps to re-sync LR (`eval.py:237-243`).
- **Cost scales with `len(layers) × len(probes)`** heads, but the encoder forward is run
  **once per batch** regardless — the clip path re-encodes every epoch (no feature cache;
  that optimization lives in `analysis_vlm`).
