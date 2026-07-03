# Plotting

## Purpose

`evals/analysis/plotting.py` renders the **layer-wise probing** figure: x = encoder layer
(shown as *layer fraction* 0–1), y = a per-layer metric (classification accuracy % or
regression R²), with **one curve per series** (probe type, or regressed variable). It adds
publication-style annotations used by this fork's physics-emergence analysis: a **peak**
marker (argmax) and an **elbow/saturation** marker, a metric-aware y-axis with a baseline
line, and an optional **PEZ** (Physics Emergence Zone) shaded band.

It is a pure rendering helper — one public function, `plot_layer_val_acc(...)` — called at
the end of a probing run by the two analysis harnesses. It never raises into training:
matplotlib import is wrapped and failure just logs a warning and returns `None`
(`plotting.py:45-52`).

## What changed vs upstream V-JEPA2

| File | Status | Delta |
|------|--------|-------|
| `evals/analysis/plotting.py` | **New file** (153 lines, added this fork) | Entire module; `git diff 204698b -- evals/analysis/plotting.py` = `1 file changed, 153 insertions(+)`. Upstream has no analysis plotting. |

The module is consumed by (also fork additions):
- `evals/analysis/eval.py:311` — clip/V-JEPA probing harness. Calls with **defaults only**
  (`metric="accuracy"`, no `num_classes`, no `pez`).
- `evals/analysis_vlm/eval.py:562-563` — VLM/multi-variable harness. Passes `metric`,
  `num_classes`, `pez=plot_pez`, and per-head `series`/`probe`/`stage` keys.

## Public API

```python
def plot_layer_val_acc(heads, val_acc, out_path, subtitle=None, num_classes=None,
                       metric="accuracy", target_label=None, pez=None):
```
(`plotting.py:36-37`)

- `heads`: list of dicts, each with at least `layer` and `name` (`name == "L<layer>_<probe>"`).
  Optional keys `series`, `probe`, `stage` are honored when present.
- `val_acc`: dict `head_name -> value` (accuracy % or R²). Despite the name, the harnesses
  pass **best-so-far** values (`best_val`).
- `metric`: `"accuracy"` (default) or `"r2"` — controls the entire y-axis regime.
- `pez`: `[lo, hi]` layer-fraction band to shade, or falsy → no shading.

## Series grouping (one curve per series/variable)

Each head is bucketed by a **series label**, chosen with this precedence
(`plotting.py:63-67`):

```python
probe_label = (h.get("series") or h.get("probe")
               or (h["name"].split("_", 1)[1] if "_" in h["name"] else h["name"]))
series[probe_label].append((h["layer"], val_acc[h["name"]]))
```

- **Explicit `series`** wins — the VLM harness sets it to the regressed *variable*
  (speed / direction / accel_mag) for R², or the probe type for classification
  (`evals/analysis_vlm/eval.py:363-374`), so e.g. speed/direction/accel become **separate
  curves on one plot**.
- Fallback splits `name` on the **first** `_`. The code comments flag that this is *wrong*
  for VLM stage tags like `block_5_linear-mean` — which is exactly why the explicit `series`
  key exists.
- Points within a series are sorted by layer before plotting; the line is drawn with
  `linewidth=2` and **no markers** (`plotting.py:88`).

Optional `stage` keys populate `stage_by_x` (`plotting.py:70`) so peak/elbow annotations can
use a human stage name (e.g. `block_5`) instead of `L<layer>`.

## Peak marker (argmax)

For each series, the peak = the point of maximum metric (first stage at the global max if
tied, since `max` keeps the first). It is drawn as a **filled star** on the curve plus a
dashed vertical guide and a bold label (`plotting.py:92-99`):

```python
bx, by = max(pts, key=lambda t: t[1])
peak_label = stage_by_x.get(bx, f"L{bx}")
ax.axvline(bx, color=color, linestyle="--", alpha=0.35, linewidth=1)
ax.scatter([bx], [by], s=90, marker="*", color=color, zorder=5,
           edgecolors="black", linewidths=0.6)
ax.annotate(f"peak: {peak_label} ({vfmt(by)})", (bx, by), ...)
```

## Elbow / saturation marker (`_elbow_x`)

The elbow is a parameter-free "knee": the point of **maximum perpendicular distance from the
chord** joining the first and last point of the curve — the saturation layer where the metric
stops improving meaningfully (`plotting.py:17-33`):

```python
def _elbow_x(xs, ys):
    if len(xs) < 3:
        return None
    x0, y0, x1, y1 = xs[0], ys[0], xs[-1], ys[-1]
    dx, dy = x1 - x0, y1 - y0
    denom = math.hypot(dx, dy)
    if denom == 0:
        return None
    best_x, best_d = None, -1.0
    for x, y in zip(xs, ys):
        d = abs(dy * (x - x0) - dx * (y - y0)) / denom  # |cross| / |chord|
        if d > best_d:
            best_d, best_x = d, x
    return best_x
```

Rendered as a **hollow diamond** (`facecolors="none"`) with a dotted vertical guide, visually
distinct from the filled-star peak (`plotting.py:104-113`).

### Direction-only elbow rule (a hard invariant)

The elbow is computed **only for the series literally named `"direction"`**
(`plotting.py:104`):

```python
ex = _elbow_x(xs, ys) if probe_label == "direction" else None
if ex is not None and ex != bx:
    ...
```

Rationale (from the code comment): speed/accel are high from the first layer, so their elbow
is not informative; direction is the meaningful emergence marker. The elbow is also **skipped
when it coincides with the peak** (`ex != bx`), and when the series has `< 3` points (`_elbow_x`
returns `None`).

## Metric-aware axes

`is_r2 = metric == "r2"` (`plotting.py:54`) switches the value formatter and the whole y-axis
regime.

| Aspect | `accuracy` (default) | `r2` |
|--------|----------------------|------|
| value fmt (`vfmt`) | `"{v:.1f}%"` | `"R²={v:.3f}"` |
| y-label | `val accuracy (%)` | `validation R²` |
| ylim | fixed `0 – 120` | `min(0, min(all_y))-0.03 … 1.03` (dynamic lower bound) |
| yticks | `range(0,121,20)` → 0,20,…,120 | steps of 0.2 from `floor(lo/0.2)*0.2` up to 1.0 |
| baseline line (red dashed) | `chance = 100/num_classes`, **only if `num_classes` given** | `R²=0` (predict-mean) baseline, always |
| baseline label | `"chance {:.1f}%"` | `"R²=0 (predict mean)"` |

Accuracy branch (`plotting.py:133-141`):

```python
ax.set_ylim(0, 120)                 # fixed y-scale so plots are comparable across runs
ax.set_yticks(range(0, 121, 20))
if num_classes:                     # random-chance baseline (= 100 / #classes)
    chance = 100.0 / num_classes
    ax.axhline(chance, color="red", linestyle="--", ...)
```

R² branch (`plotting.py:123-132`): ylim floor is `min(0.0, min(all_y)) - 0.03` so
below-baseline probes (R² < 0, worse than predicting the mean) remain visible, and the red
`R²=0` line at `axhline(0.0)` marks the mean-prediction baseline.

The title reflects the metric and, for R² with a `target_label`, names the variable
(`plotting.py:143-146`). Note: the VLM harness intentionally passes **no** `target_label`
because each variable is already its own curve/legend entry
(`evals/analysis_vlm/eval.py:561`).

## Layer-fraction x-axis

Data, PEZ band, and peak/elbow positions all live in **layer-index** space; only the *ticks*
are relabeled to fractions of the deepest layer, so the linear axis reads 0.0–1.0
(`plotting.py:117-121`):

```python
depth = max((h["layer"] for h in heads), default=0) or 1
ax.set_xticks([depth * f for f in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)])
ax.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
ax.set_xlim(-0.02 * depth, 1.02 * depth)
ax.set_xlabel("layer fraction")
```

The `or 1` guards against `depth == 0` (single layer / empty) to avoid divide-by-zero.

## PEZ gray shading (`plot_pez`)

The Physics Emergence Zone is an optional gray band (`color="gray", alpha=0.15, zorder=0`)
plus a "PEZ" text label, mapping fractional `[lo, hi]` onto the layer axis via the deepest
layer index (`plotting.py:76-80`):

```python
if pez:
    depth = max((h["layer"] for h in heads), default=0) or 1
    ax.axvspan(pez[0] * depth, pez[1] * depth, color="gray", alpha=0.15, zorder=0)
    ax.text((pez[0] + pez[1]) / 2 * depth, 0.99, "PEZ", ...)
```

Driven by config key `plot_pez` under `experiment.analysis`. The VLM harness validates it
before use (`evals/analysis_vlm/eval.py:150-153`):

```python
plot_pez = args_analysis.get("plot_pez")   # [lo,hi] band; None/false -> no shading
if plot_pez:
    assert len(plot_pez) == 2 and 0.0 <= plot_pez[0] < plot_pez[1] <= 1.0, ...
```

## Legend

Single legend, **lower-right**, titled by metric (`plotting.py:148`):

```python
ax.legend(title=("variable" if is_r2 else "probe"), loc="lower right")
```

Output is written with `fig.savefig(out_path, dpi=130)` and the figure is closed
(`plotting.py:150-151`). Filenames differ per harness: `layer_val_acc.png` (clip) vs
`stage_val_acc.png` (VLM).

## Config

All plotting behavior is config/argument controlled — the function has no side effects unless
the harness both enables plotting and passes the relevant knobs.

- `experiment.analysis.plot` (bool, **default `false`**) gates the entire call.
  `make_plot = args_analysis.get("plot", False)` in both harnesses
  (`evals/analysis/eval.py:111`, `evals/analysis_vlm/eval.py:149`). The plot is emitted only
  when `make_plot and rank == 0`.
- `experiment.analysis.plot_pez` (`[lo, hi]` or omitted/`false`) — PEZ band. **VLM harness
  only**; the clip harness never forwards `pez`.

Real YAML (from `configs/analysis/toy_dataset/vjepa_combined.yaml`, R² regression, multiple
variables → one curve each):

```yaml
experiment:
  analysis:
    task: regression
    regression:
      targets_npy: /.../toy_physics/combined_targets.npy   # (672,4)=[speed, sinθ, cosθ, accel]
      variables:
        - {name: speed,     cols: [0]}
        - {name: direction, cols: [1, 2]}   # <- the only series that gets an elbow marker
        - {name: accel_mag, cols: [3]}
    stages:
      vision_encoder: all
    plot: true
    plot_pez: [0.2, 0.4]        # Physics Emergence Zone shading (layer-fraction); false/omit = off
    probes:
      - type: linear
        pooling: mean
        pre_norm: true
        optimization: { lr: 0.001, weight_decay: 0.1, warmup: 2.0 }
```

Here `metric` is derived by the harness (`"r2"` because `task: regression`), not set directly
in YAML (`evals/analysis_vlm/eval.py:559`).

## Gotchas / invariants / default-off guarantees

- **Default off**: `plot` defaults to `False`; with no `plot:` key nothing is rendered and
  matplotlib is never imported.
- **Never breaks a run**: matplotlib import failure is caught and only logged
  (`plotting.py:50-52`); rendering is optional by design.
- **rank-0 only**: both harnesses guard the call with `rank == 0`.
- **Elbow is direction-only** — a deliberate rule (`plotting.py:104`), not a general knee
  detector; it also requires ≥ 3 points and a peak-mismatch (`ex != bx`).
- **`chance` line requires `num_classes`** — the clip harness passes none, so accuracy plots
  from `evals/analysis/eval.py` show **no** chance baseline.
- **Series grouping key matters**: relying on the `name.split("_", 1)` fallback misgroups VLM
  stage tags; harnesses must set explicit `series` (they do). See
  `evals/analysis_vlm/eval.py:363-374`.
- **Fraction axis is cosmetic**: ticks are fractions but all geometry (peak/elbow/PEZ) stays
  in integer layer-index units; `depth == 0` is guarded by `or 1`.
- **`val_acc` argument carries *best*-so-far values** in both call sites, despite the
  parameter name.

## Cross-references

- `02-analysis-probing` (clip harness `evals/analysis/eval.py` that emits `layer_val_acc.png`)
- `06-analysis-vlm` (VLM harness `evals/analysis_vlm/eval.py`, `series`/`stage`/`plot_pez` wiring)
