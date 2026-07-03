# V-JEPA2 fork — physics-interpretability probing harness

## What this fork is

This is an **additive, default-off mechanistic-interpretability harness** built on top of
upstream V-JEPA2 (baseline commit `204698b`). It reproduces *"Interpreting Physics in Video
World Models"* by tapping the **per-layer features of frozen video encoders** — the V-JEPA2
ViT, the LLaVA-Video SigLIP tower, and the Qwen3-VL ViT — attaching **one probe head per
(layer/stage × probe-spec × variable)**, training those heads jointly on frozen features
(classification *or* regression), and plotting a **layer-wise accuracy / R² curve** that shows
where physical variables (speed, direction, acceleration) become linearly decodable. It ships
its own **Blender/CYCLES toy-physics dataset generator** (paper-faithful single fixed red
sphere, 672 clips), a **feature cache** that makes the all-layer scan tractable, and a
**post-hoc analysis-modes** layer (attention distance/ablation, steering, …). Everything is
routed through the stock `eval_name` dispatch (`evals/main.py`/`scaffold.py` are **untouched**);
two new eval packages (`evals/analysis`, `evals/analysis_vlm`) carry all the new code, and the
only upstream source edits are a handful of `uniform_sampling` passthroughs and comments — remove
the fork's config knobs and upstream behavior is byte-for-byte identical. Headline result: the
frozen V-JEPA2-L **Fig. 2c dissociation reproduces** — speed decodable from block 0, direction
emerging sharply at the ~one-third-depth Physics Emergence Zone, acceleration in between.

---

## Table of contents

Read top-to-bottom for a full tour; jump by area otherwise.

| # | Doc | One-liner |
|---|-----|-----------|
| 01 | [Overview & architecture](01-overview-and-architecture.md) | Top-level map of the fork: what is probed, the two harnesses, edit-free routing, the data flow, and the minimal upstream delta. |
| 02 | [`analysis_vlm` harness (eval flow)](02-analysis-vlm-harness.md) | The unified frozen-encoder harness (vjepa/llavavideo/qwen3vl): builds one head per (stage × probe × variable), trains jointly, reports a [stage × probe] matrix. |
| 03 | [Feature caching & pooling](03-feature-caching-and-pooling.md) | Encode-once per-rank fp16 feature cache with three pooling granularities (`tokens`/`pooled`/`framewise`) plus the required pooled-probe LayerNorm. |
| 04 | [Probes, regression & NaN-masking](04-probes-regression-nanmask.md) | The four probe head types, the `targets.npy`-indexed regression task, per-column standardization, and the DDP-safe NaN-masked masked-mean MSE / R². |
| 05 | [`analysis` clip harness (V-JEPA)](05-analysis-clip-harness.md) | The V-JEPA-only clip harness: a multilayer wrapper returns each block separately and trains one probe per (layer × probe) in a single forward. |
| 06 | [Data-pipeline changes](06-data-pipeline-changes.md) | The two additive knobs — `uniform_sampling` and `resize_mode: resize` — that fix the `frame_step` contiguous-window sampling bug behind the reproduction. |
| 07 | [Plotting](07-plotting.md) | The single-function layer-wise plotter: peak star, direction elbow, metric-aware axes, PEZ shading, layer-fraction x-axis (all default-off). |
| 08 | [VLM encoder backends](08-vlm-encoders.md) | The two new frozen vision-only wrappers (LLaVA-Video SigLIP+projector, Qwen3-VL ViT+merger/deepstack), each in its own conda env. |
| 09 | [Blender toy-physics dataset](09-blender-toy-dataset.md) | The paper-faithful Blender/bpy CYCLES generator: 672 single-fixed-sphere clips with analytic motion and exact physical+pixel ground truth. |
| 10 | [CSV / `targets.npy` builders](10-datasets-csv-targets.md) | The scripts that turn per-video metadata into a `targets.npy` plus index CSVs whose integer label is a row index into that array. |
| 11 | [Attention hooks (distance + ablation)](11-attention-hooks.md) | The runtime SDPA monkey-patch + RoPEAttention hooks capturing per-head attention distance and injecting a distance-threshold ablation bias. |
| 12 | [Analysis modes & reproduction roadmap](12-analysis-modes.md) | The additive `experiment.analysis.modes` dispatch layer (registry + context), the one implemented mode, and the Phase 0–5 roadmap. |
| 13 | [Config reference](13-configs-reference.md) | Full YAML key-space for `eval_name: analysis_vlm`, the config-family map, and the `z_scripts` SLURM launchers with the `--export CONFIG` pattern. |
| 14 | [Reproduction status & findings](14-reproduction-status-and-findings.md) | What actually reproduced (Fig. 2c), the two root-cause debugging wins, the `pre_norm` correctness finding, and the remaining-experiments roadmap. |

---

## Quickstart

### 1. Run a layer-wise probing job

The active harness is `evals/analysis_vlm` (`eval_name: analysis_vlm`). Everything is one
`python -m evals.main --fname <config>` call; the config selects backend, data, probes, and
the layer scan.

**Single-GPU debug** (frozen V-JEPA2-L, all 24 blocks, regression on the Blender set):

```bash
python -m evals.main --fname configs/analysis/blender_toy_dataset/vjepa_combined.yaml \
    --devices cuda:0 --debugmode True
# writes <folder>/analysis_vlm/<tag>/: summary.json, log_r0.csv, latest.pt, stage_val_acc.png
```

**Multi-GPU via SLURM** (launchers convert `CUDA_VISIBLE_DEVICES` → `--devices` and pick the
conda env; env **must** be paired with the config):

```bash
# V-JEPA (env vjepa2; default config = blender vjepa_combined.yaml)
sbatch z_scripts/run_analysis_vjepa.sh

# override env+config for a VLM backend
sbatch --export=ALL,CONDA_ENV=lmms_eval_llavavideo,\
CONFIG=configs/analysis/blender_toy_dataset/llavavideo_combined.yaml \
    z_scripts/run_analysis_vlm.sh
# Qwen3-VL: CONDA_ENV=lmms_eval_py311_2.7, CONFIG=configs/analysis/.../qwen3vl_combined.yaml
```

> The older V-JEPA-only **clip harness** (`eval_name: analysis`, classification) runs the same
> way: `python -m evals.main --fname configs/z_tak_attentive_probing/R2R_4way_analysis.yaml --devices cuda:0`.

Note (from `run_analysis_vjepa.sh`): the default config uses `cache_pooling: tokens` + 7 stages,
so per-rank cache RAM ≈ 88 GB on 4 GPU and trips the `cache_max_gb: 64` guard — use 8 GPU, or
switch to `cache_pooling: pooled` (linear-only), or raise `cache_max_gb`.

### 2. Generate the Blender toy-physics dataset

One shell script renders all 672 clips (velocity 392 + acceleration 280), merges the shards,
and runs the sanity checker. It shards one Blender process per GPU (round-robin,
`SHARDS_PER_GPU` procs each):

```bash
bash data_gen/run_blender_toy.sh
# overrides:
DEVICE=CPU  bash data_gen/run_blender_toy.sh     # OPTIX fails on vll4; CUDA is the default
SAMPLES=32  bash data_gen/run_blender_toy.sh     # more CYCLES samples/pixel (slower)
NGPU=2      bash data_gen/run_blender_toy.sh     # cap GPU count
# outputs: data_gen/blender_toy_dataset/{velocity,acceleration}/*.mp4, metadata.csv, kinematics.json
```

Then build the regression targets + index CSVs the probing configs consume
(`data_csv/`, see [§10 CSV / targets builders](10-datasets-csv-targets.md)): `make_blender_targets.py` emits
`blender_targets.npy` `(672,4)=[speed, sinθ, cosθ, accel_mag]` (NaN-masked) plus 6 split CSVs.

---

## Change surface vs upstream V-JEPA2 (baseline `204698b`)

Additive and default-off throughout. `evals/main.py`, `evals/scaffold.py`,
`evals/video_classification_frozen/models.py`, and `src/models/utils/modules.py` are
**byte-identical to upstream**.

| Area | Path | Kind | What / why |
|------|------|------|-----------|
| Clip harness | `evals/analysis/` (eval.py, `modelcustom/vit_encoder_multilayer.py`, probes.py, plotting.py) | **new** | V-JEPA-only per-layer probing; routed by `eval_name: analysis`. |
| Unified harness | `evals/analysis_vlm/` (eval.py, cache.py, probes.py, data.py, loadutil.py, `modes/`, attention_hooks.py) | **new** | vjepa/llavavideo/qwen3vl, +regression, +feature cache, +temporal probes, +post-hoc modes; `eval_name: analysis_vlm`. |
| Src pipeline | `src/datasets/video_dataset.py` | **modified** | `uniform_sampling` param + one early-return branch in `loadvideo_decord` (`:337-341`). Default off. |
| Src pipeline | `src/datasets/data_manager.py`, `evals/video_classification_frozen/eval.py` | **modified** | `uniform_sampling` passthrough (~2 lines each). |
| Src models | `src/models/vision_transformer.py` | **comment only** | `out_layers` multi-layer tap already upstream (`:204-207`); the fork only wraps it. |
| Configs | `configs/analysis/**.yaml` (21 files) | **new, tracked** | Probing configs. Only `configs/analysis/logs/` is gitignored. |
| Data gen | `data_gen/` (`make_physics_blender.py`, `run_blender_toy.sh`, `sanity_check_blender.py`) | **new, gitignored** | Blender toy-physics generator + checker. |
| Datasets | `data_csv/` (`make_blender_targets.py`, `make_regression_targets.py`, outputs) | **new, gitignored** | `targets.npy` + index-CSV builders (`*csv` glob ignores the dir). |
| Launchers | `z_scripts/` (`run_analysis_vjepa.sh`, `run_analysis_vlm.sh`) | **new, gitignored** | SLURM launchers with the `--export CONFIG` override pattern. |

---

## Glossary

- **PEZ (Physics Emergence Zone)** — the ~one-third-depth band of layers where *direction*
  suddenly becomes linearly decodable (empirically V-JEPA2-L L7→L8, layer fraction ≈ 0.35).
  Rendered as a gray `axvspan` from `plot_pez: [lo, hi]` in **layer-fraction** units (e.g.
  `[0.2, 0.4]`); a reproduction "passes" when the direction curve jumps inside it.

- **`cache_pooling`** — granularity of the fp16 feature cache (`experiment.optimization`):
  `tokens` = full `(B,N,D)` (all probe types, large), `pooled` = `(B,2D)=[mean‖max]`
  (**linear-only**, tiny, collapses time), `framewise` = `(B,T,D)` spatial-mean per frame
  (VLM-only). Token-level modes (ortho/steering/direction-tuning) require `tokens`.

- **`pre_norm`** — per-sample `nn.LayerNorm` over the feature dim `D` applied before the
  linear/framewise probe (default `True`). **Required** for a cross-layer scan: V-JEPA
  activation scale varies by orders of magnitude across depth, so one shared lr cannot fit raw
  features. LayerNorm-over-D or a per-feature StandardScaler both work; `pre_norm: false` is
  the one wrong choice. (Attentive heads carry their own norm and ignore it.)

- **`uniform_sampling`** — additive, default-off data knob (`experiment.data`): evenly sample
  `frames_per_clip` frames across the **whole** video (`round(linspace(0, len-1, fpc))`),
  ignoring `frame_step`/`num_clips`. Sidesteps the stock contiguous-window sampler that (with
  `frame_step=1` on a 64-frame clip) reads only the first quarter of the trajectory and flattens
  the early-speed curve.

- **`modes`** — the additive post-hoc analysis layer (`experiment.analysis.modes`, a
  `{name: cfg}` map). Absent key ⇒ nothing imported ⇒ outputs byte-identical. One mode
  implemented (`attention_distance`); Phases 2–5 (ablation, ortho-probe, steering, direction
  tuning) are on the roadmap. See [§12 Analysis modes](12-analysis-modes.md).

- **`series`** — the plotting group key: one curve per `h['series']` (falling back to `probe`,
  then a name split). The VLM harness sets `series = variable` (speed/direction/accel) so each
  physical variable gets its own layer-wise curve instead of colliding on a stage-tag split.

- **`framewise`** — a temporal token layout: `(B,N,D) → (B,T,D)` by spatial-mean per frame,
  used both as a cache pooling mode and as a `TemporalLinearProbe` pooling that concatenates
  the `T` frames order-aware. **VLM-only** — requires `encoder.num_temporal` (V-JEPA does not
  expose it), giving VLM pooling the temporal ordering it otherwise lacks.
