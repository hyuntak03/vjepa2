# 07 — Plotting

> Two plotting surfaces in this fork: the base layer-wise probing figure (`evals/analysis/plotting.py::plot_layer_val_acc`) with peak-star / direction-only-elbow / metric-aware axes / PEZ shading / layer-fraction x-axis, **plus** the newer per-mode attention-locality plots (`evals/analysis_vlm/modes/attention_distance.py::_plot_heatmap` + `_plot_layerwise`) that render the paper's Fig 3 heatmap and Fig 19 dual-axis line plot **without routing through `plotting.py`**. Everything is config/argument controlled and default-off.

## Purpose

There are now **two independent plotting code paths** in the analysis subsystem; a reader should not assume `plotting.py` is the only one.

**Surface 1 — base probing figure (`evals/analysis/plotting.py`).**
`plot_layer_val_acc(...)` renders the **layer-wise probing** figure: x = encoder layer (shown as *layer fraction* 0–1), y = a per-layer metric (classification accuracy % or regression R²), with **one curve per series** (probe type, or regressed variable). It adds this fork's physics-emergence annotations: a **peak** marker (argmax), a direction-only **elbow/saturation** marker, a metric-aware y-axis with a baseline line, and an optional **PEZ** (Physics Emergence Zone) shaded band. It is a pure rendering helper — one public function — called at the end of a probing run by **both** analysis harnesses (clip + VLM). It never raises into training: the matplotlib import is wrapped and failure just logs a warning and returns `None` (`evals/analysis/plotting.py:45-52`).

**Surface 2 — attention-distance mode plots (`evals/analysis_vlm/modes/attention_distance.py`).**
The post-hoc `attention_distance` mode renders **its own figures inline** (two private functions, `_plot_heatmap` + `_plot_layerwise`) and has **zero references** to `plotting.py` / `plot_layer_val_acc`. Its **primary** output is the paper's **Figure 3 heatmap** (`attention_distance.png`); its companion is the **Appendix Figure 19 dual-axis** line plot (`attention_distance_layerwise.png`). These read the `spatial_distance` matrix captured by the SDPA attention hook (see `11-attention-hooks`) rather than probe metrics.

> **Reproduction-plan status.** `REPRODUCTION_PLAN.md` "Touched file 2" intended to *add* a `plot_direction_tuning` function to `plotting.py` and *refactor* a shared `_shade_pez(ax, depth, pez)` helper. **Neither is done yet (Phase 4 pending).** `plotting.py` is still **exactly** `plot_layer_val_acc` + `_elbow_x` — byte-identical, unchanged by the reproduction work. All mode-specific plotting currently lives **inline in each mode file**.

## What changed vs upstream V-JEPA2

| File | Status | Delta |
|------|--------|-------|
| `evals/analysis/plotting.py` | **NEW file** (153 lines) | Entire module. `git diff 204698b -- evals/analysis/plotting.py` = `new file mode 100644 … @@ -0,0 +1,153 @@` (153 insertions). Upstream has **no** analysis plotting. Contents: `plot_layer_val_acc` + `_elbow_x` only. **Unchanged** by the physics-reproduction work so far. |
| `evals/analysis_vlm/modes/attention_distance.py` | **NEW file** (157 lines) | The `attention_distance` mode (`run`) **and** its two inline plotters `_plot_heatmap` (Fig 3) + `_plot_layerwise` (Fig 19). Did not exist at `204698b` (`git ls-tree 204698b` returns nothing). Does **not** import or call `plotting.py`. |
| `evals/analysis/eval.py` | MODIFIED (consumer) | Adds `from evals.analysis.plotting import plot_layer_val_acc` (`:46`) and one **defaults-only** call (`:310-314`) → `layer_val_acc.png`. Plotting delta only; full harness delta in `05-analysis-clip-harness`. |
| `evals/analysis_vlm/eval.py` | MODIFIED (consumer) | Lazy import + call with `metric`/`num_classes`/`pez` (`:556-563`) → `stage_val_acc.png`; sets per-head `series`/`probe`/`stage` (`:363-374`); validates `plot_pez` (`:150-153`); dispatches modes (`:565-590`). Full delta in `02-analysis-vlm-harness`. |

**Default-off guarantee.** `experiment.analysis.plot` defaults `False` → the base plot is never rendered and matplotlib is never imported. `experiment.analysis.modes` is absent from every existing config → the mode dispatch block is skipped entirely (`modes_cfg == {}`), so `attention_distance.py` is never imported and its plotters never run. Both plot call sites and the mode dispatch are additionally gated on `rank == 0`.

---

## Design & data flow

### Surface 1 · `plot_layer_val_acc` — the base probing figure

Signature (`evals/analysis/plotting.py:36-37`):

```python
def plot_layer_val_acc(heads, val_acc, out_path, subtitle=None, num_classes=None,
                       metric="accuracy", target_label=None, pez=None):
```

| Param | Meaning |
|-------|---------|
| `heads` | list of dicts, each with at least `layer` and `name` (`name == "L<layer>_<probe>"`); optional `series`, `probe`, `stage` honored when present. |
| `val_acc` | dict `head_name -> value` (accuracy % or R²). **Despite the name, the harnesses pass *best-so-far* values** (`best_val`). |
| `metric` | `"accuracy"` (default) or `"r2"` — switches the entire y-axis regime. Derived by the VLM harness (`"r2" if task == "regression"`, `evals/analysis_vlm/eval.py:559`), **not** set in YAML. |
| `num_classes` | classification chance baseline `= 100/num_classes`; VLM passes `data.num_classes`, clip passes none. |
| `target_label` | R²-plot title suffix; VLM passes **none** (each variable is its own curve). |
| `pez` | `[lo, hi]` layer-fraction band to shade, or falsy → no shading. |

**Series grouping (one curve per series/variable).** Each head is bucketed by a series label with this precedence (`plotting.py:63-67`):

```python
probe_label = (h.get("series") or h.get("probe")
               or (h["name"].split("_", 1)[1] if "_" in h["name"] else h["name"]))
series[probe_label].append((h["layer"], val_acc[h["name"]]))
```

- **Explicit `series`** wins — the VLM harness sets it to the regressed *variable* (speed / direction / accel_mag) for R², or the probe type for classification, and to `var·probe` when multiple probes coexist (`evals/analysis_vlm/eval.py:363-374`). So speed/direction/accel become **separate curves on one plot**.
- Fallback splits `name` on the **first** `_`. The code comment flags this is *wrong* for VLM stage tags like `block_5_linear-mean` — which is exactly why the explicit `series` key exists.
- Points within a series are sorted by layer; the line is `linewidth=2`, **no markers** (`plotting.py:88`).
- Optional `stage` keys populate `stage_by_x` (`plotting.py:70`) so peak/elbow annotations show a human stage name (e.g. `block_5`) instead of `L<layer>`.

**Peak marker (argmax).** Per series, peak = point of maximum metric (first stage at the global max if tied, since `max` keeps the first). Drawn as a **filled star** + dashed vertical guide + bold label (`plotting.py:92-99`).

**Elbow / saturation marker — direction-only (hard invariant).** The elbow is a parameter-free "knee": the point of **maximum perpendicular distance from the chord** joining the first and last point of the curve (`_elbow_x`, `plotting.py:17-33`). It is computed **only for the series literally named `"direction"`**, is **skipped when it coincides with the peak** (`ex != bx`), and needs **≥ 3 points** (`_elbow_x` returns `None` otherwise). Rendered as a **hollow diamond** (`facecolors="none"`) + dotted vertical guide, distinct from the filled-star peak (`plotting.py:100-113`). Rationale (code comment): speed/accel are high from the first layer, so their elbow is uninformative; direction is the meaningful emergence marker.

**Metric-aware axes.** `is_r2 = (metric == "r2")` (`plotting.py:54`) switches the value formatter and whole y-axis regime:

| Aspect | `accuracy` (default) | `r2` |
|--------|----------------------|------|
| value fmt (`vfmt`) | `"{v:.1f}%"` | `"R²={v:.3f}"` |
| y-label | `val accuracy (%)` | `validation R²` |
| ylim | fixed `0 – 120` | `min(0, min(all_y)) − 0.03 … 1.03` (dynamic lower bound so below-baseline probes stay visible) |
| yticks | `range(0,121,20)` → 0,20,…,120 | steps of 0.2 from `floor(lo/0.2)*0.2` up to 1.0 |
| baseline (red dashed) | `chance = 100/num_classes`, **only if `num_classes` given** | `R²=0` (predict-mean) baseline, **always** at `axhline(0.0)` |
| baseline label | `"chance {:.1f}%"` | `"R²=0 (predict mean)"` |

The title reflects the metric and, for R² with a `target_label`, names the variable (`plotting.py:143-146`).

**Layer-fraction x-axis (cosmetic relabel).** Data, PEZ band, and peak/elbow positions all live in **layer-index** space; only the *ticks* are relabeled to fractions of the deepest layer, so the linear axis reads 0.0–1.0 (`plotting.py:117-121`). `depth = max(layer) or 1` guards `depth == 0` against divide-by-zero.

**PEZ gray shading.** Optional gray band (`color="gray", alpha=0.15, zorder=0`) + a "PEZ" text label, mapping fractional `[lo, hi]` onto the **integer layer axis** via `pez[i] * depth` (`plotting.py:76-80`). Driven by `experiment.analysis.plot_pez`; the VLM harness validates it (`0<=lo<hi<=1`, `evals/analysis_vlm/eval.py:150-153`). The clip harness never forwards `pez`.

**Legend & output.** Single legend, **lower-right**, titled `"variable"` (R²) or `"probe"` (accuracy) (`plotting.py:148`). Saved with `fig.savefig(out_path, dpi=130)`, figure closed (`plotting.py:150-151`). Filenames differ per harness: `layer_val_acc.png` (clip) vs `stage_val_acc.png` (VLM).

### Surface 2 · `attention_distance` mode plots

The mode `run()` (`attention_distance.py:32-73`) captures a `(num_layers × num_heads)` **spatial** distance matrix (and a temporal one) via the SDPA hook, writes `attention_distance.json`, then calls **two** inline plotters. Input JSON keys (verified on the real Blender run): `spatial_distance` `[24][16]`, `temporal_distance` `[24][16]`, `num_layers=24`, `num_heads=16`, `rows_per_layer`, `n_batches`. **Both plotters use `spatial_distance` only; `temporal_distance` is captured but not plotted.**

**`_plot_heatmap` (PRIMARY, paper Fig 3 → `attention_distance.png`)** (`attention_distance.py:76-112`):

- `Z = np.array(out["spatial_distance"]).T` → shape `(H, L) = (16, 24)`; rows = head, cols = layer.
- `imshow(Z, aspect="auto", cmap="Blues_r", origin="lower", vmin, vmax, extent=[-0.5, L-0.5, -0.5, H-0.5])`. **`Blues_r` means LOW distance = DARK blue**, so the unusually-local heads that emerge at the PEZ stand out as dark cells; `origin="lower"` puts head 0 at the bottom.
- Axes: **x = "Layer"** ticks `0…23`, **y = "Attention Head"** ticks `0…15` (fontsize 6).
- **Per-cell value annotations** gated by cfg **`annotate`** (default `True`): each cell prints `f"{v:.1f}"`; text colour is `"white"` on dark cells and `"#222"` on light cells, thresholded at the **mid value** `mid = 0.5*(vmin+vmax)` (`color = "white" if v < mid else "#222"`).
- Colorbar label **`"Distance (patches)"`**; title **`"V-JEPA v2-L: Attention Distance Per Head"`** (+ subtitle `vjepa | {n} val batches`).
- `figsize=(max(9, L*0.44), max(4.5, H*0.34))` — **auto-scales with L and H** (≈10.6×5.4 in for 24×16); `dpi=150`.

**`_plot_layerwise` (companion, Appendix Fig 19 → `attention_distance_layerwise.png`)** (`attention_distance.py:115-156`):

- `L = out["num_layers"]`; `depth = (L-1) or 1`; **x = layer fraction** `xs = [layer/depth for layer in range(L)]` (already 0..1).
- **Left axis (`tab:red`, solid, marker `o`)** = layer-mean distance **`Dbar`** = `mean over the 16 heads` (`[sum(row)/len(row) for row in sd]`), y-label `"attention distance (patches)"`.
- **Right axis (`twinx`, `tab:blue`, dashed, marker `s`)** = **head specialization** **`S`** = `statistics.pstdev(row)` = **population std over the 16 heads** of their per-head attention distances within a layer (`0.0` if a layer has ≤1 head). y-label `"head specialization (patches)"`.
  - **"Head specialization" = attention-head diversity** — how differently the 16 heads attend within a layer. It **SPIKES at the PEZ** (spatiotemporally-local heads emerge alongside long-range heads), while `Dbar` **DIPS** there.
- **PEZ shading** drawn **directly** at `axvspan(pez[0], pez[1])` (no `*depth`) because the x-axis is *already* a fraction; `pez` comes from `ctx.plot_pez` = `experiment.analysis.plot_pez`.
- Combined legend of both axes' lines, `loc="upper center"`; `figsize=(8, 4.5)`; `dpi=130`.

### Comparison of the three plotting surfaces

| | `plotting.py` `plot_layer_val_acc` | `_plot_heatmap` (Fig 3) | `_plot_layerwise` (Fig 19) |
|---|---|---|---|
| Purpose | layer-wise probe metric per series | per-(layer,head) attention locality | layer-mean distance + head specialization |
| Output PNG | `layer_val_acc.png` / `stage_val_acc.png` | `attention_distance.png` (PRIMARY) | `attention_distance_layerwise.png` |
| Data source | probe `best_val` metrics | `out["spatial_distance"]ᵀ` | `out["spatial_distance"]` |
| x-axis | **layer fraction** (ticks relabeled; data in **index** space) | **Layer index 0–23** | **layer fraction 0–1** (data already fraction) |
| y-axis | metric (acc % or R²) | Attention Head index 0–15 | `Dbar` (left) + `S` (right `twinx`) |
| colour | one hue per series | `Blues_r` spatial distance | red / blue lines |
| PEZ | `pez*depth` on **integer** axis | none | `pez[0..1]` **directly** |
| DRY status | canonical PEZ + frac-axis source | no PEZ | **duplicates** PEZ + frac-axis inline |
| dpi | 130 | 150 | 130 |

### DRY / structural note

`_plot_layerwise` **duplicates** two pieces of `plotting.py` inline instead of sharing a helper: (1) the **`"layer fraction"` x-axis label**, and (2) the **PEZ gray-band shading** (`color="gray", alpha=0.15, zorder=0`, "PEZ" text at `y=0.99` via `ax.get_xaxis_transform()`). Subtle behavioral difference worth flagging:

- **`plotting.py` maps PEZ via `pez*depth`** onto the **integer layer-index** axis (its data lives in index space).
- **`_plot_layerwise` draws `pez[0..1]` directly** because its x-axis is **already** fraction 0..1.

Both produce the same *visual* band, but the coordinate math differs because the axis units differ. The intended shared `_shade_pez` refactor (REPRODUCTION_PLAN Touched file 2) is **not done**.

---

## Key code

**Elbow / knee (`evals/analysis/plotting.py:17-33`).**

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

**Direction-only elbow rule (`plotting.py:104-105`).**

```python
ex = _elbow_x(xs, ys) if probe_label == "direction" else None
if ex is not None and ex != bx:   # skip when the elbow coincides with the peak
    ...
```

**PEZ shading on the integer layer axis (`plotting.py:76-80`).**

```python
if pez:
    depth = max((h["layer"] for h in heads), default=0) or 1
    ax.axvspan(pez[0] * depth, pez[1] * depth, color="gray", alpha=0.15, zorder=0)
    ax.text((pez[0] + pez[1]) / 2 * depth, 0.99, "PEZ", color="dimgray", fontsize=8,
            ha="center", va="top", transform=ax.get_xaxis_transform())
```

**Layer-fraction x-axis (cosmetic tick relabel, `plotting.py:117-121`).**

```python
depth = max((h["layer"] for h in heads), default=0) or 1
ax.set_xticks([depth * f for f in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)])
ax.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
ax.set_xlim(-0.02 * depth, 1.02 * depth)
ax.set_xlabel("layer fraction")
```

**R² vs accuracy branches (`plotting.py:123-141`).**

```python
if is_r2:
    ax.set_ylabel("validation R²")
    lo = min(0.0, min(all_y) if all_y else 0.0)
    ax.set_ylim(lo - 0.03, 1.03)
    ...
    ax.axhline(0.0, color="red", linestyle="--", ...)   # R²=0 predict-mean baseline (always)
else:
    ax.set_ylabel("val accuracy (%)")
    ax.set_ylim(0, 120)                 # fixed y-scale so plots are comparable across runs
    ax.set_yticks(range(0, 121, 20))
    if num_classes:                     # chance line ONLY when num_classes given
        chance = 100.0 / num_classes
        ax.axhline(chance, color="red", linestyle="--", ...)
```

**VLM call site + metric derivation (`evals/analysis_vlm/eval.py:556-563`).**

```python
if rank == 0 and make_plot:
    from evals.analysis.plotting import plot_layer_val_acc
    metric = "r2" if task == "regression" else "accuracy"
    sub = f"{model_sel} | best {'R²' if metric == 'r2' else 'val'} over {last_epoch} epoch(s)"
    plot_layer_val_acc(heads, best_val, os.path.join(folder, "stage_val_acc.png"),
                       subtitle=sub, num_classes=num_classes, metric=metric, pez=plot_pez)
```

**Clip call site — defaults only (`evals/analysis/eval.py:310-314`).** `metric`/`num_classes`/`pez` all defaulted → accuracy regime, **no chance line**, no PEZ:

```python
if rank == 0 and make_plot:
    plot_layer_val_acc(
        heads, best_val, os.path.join(folder, "layer_val_acc.png"),
        subtitle=f"best val over {last_epoch} epoch(s)",
    )
```

**Heatmap colormap + annotation threshold (`attention_distance.py:88-108`).**

```python
Z = np.array(out["spatial_distance"], dtype=float).T   # (H, L): rows=head, cols=layer
H, L = Z.shape
vmin, vmax = float(np.nanmin(Z)), float(np.nanmax(Z))
mid = 0.5 * (vmin + vmax)
im = ax.imshow(Z, aspect="auto", cmap="Blues_r", origin="lower",  # low dist = dark blue
               vmin=vmin, vmax=vmax, extent=[-0.5, L - 0.5, -0.5, H - 0.5])
if annotate:
    for h in range(H):
        for lyr in range(L):
            v = Z[h, lyr]
            ax.text(lyr, h, f"{v:.1f}", ha="center", va="center", fontsize=4.2,
                    color=("white" if v < mid else "#222"))  # dark cell -> white text
cbar.set_label("Distance (patches)")
ax.set_title("V-JEPA v2-L: Attention Distance Per Head" + ...)
```

**Dual-axis Dbar + head specialization, PEZ drawn directly (`attention_distance.py:127-152`).**

```python
sd = out["spatial_distance"]; L = out["num_layers"]
depth = (L - 1) or 1
xs = [layer / depth for layer in range(L)]                       # already 0..1
dbar   = [sum(row) / len(row) for row in sd]                     # mean over heads
spread = [statistics.pstdev(row) if len(row) > 1 else 0.0 for row in sd]  # std over heads
if pez:
    ax.axvspan(pez[0], pez[1], color="gray", alpha=0.15, zorder=0)   # NOT *depth
ax.plot(xs, dbar, color="tab:red", lw=2, marker="o", ms=3,
        label="attention distance (mean over heads)")
ax2 = ax.twinx()
ax2.plot(xs, spread, color="tab:blue", lw=2, ls="--", marker="s", ms=3,
         label="head specialization (std over heads)")
```

---

## Configuration

All plotting is config/argument controlled: nothing renders unless the harness enables it and passes the relevant knobs.

### Base probing plot knobs (`plot_layer_val_acc`)

| Key | Meaning | Default | Allowed |
|-----|---------|---------|---------|
| `experiment.analysis.plot` | gates the **entire** base plot call (`make_plot and rank==0`) | `false` | `true` / `false` |
| `experiment.analysis.plot_pez` | `[lo, hi]` PEZ band (layer-fraction). **VLM harness only** — clip never forwards `pez` | none | `[lo,hi]`, `0<=lo<hi<=1` (asserted `evals/analysis_vlm/eval.py:150-153`) |
| `metric` (derived) | `"r2"` if `task==regression` else `"accuracy"` | — | derived at `eval.py:559`, not YAML |
| `num_classes` (derived) | chance-line denominator; VLM passes `data.num_classes`, clip passes none | none | int |

### `attention_distance` mode plot knobs (`experiment.analysis.modes.attention_distance`)

| Key | Meaning | Default | Effect class |
|-----|---------|---------|--------------|
| `enabled` | run the mode; `false` / `enabled:false` / omit → skip | (presence) | dispatch |
| `query_chunk` | stream queries so the `(B,H,N,N)` attention matrix is never materialized (OOM guard) | `512` | **memory only — result-invariant** |
| `max_batches` | number of val batches averaged into the distance matrices | `8` | **sampling** (more = more stable average; changes numbers, converges) |
| `annotate` | per-cell numeric annotations on the heatmap | `true` | **rendering only** |
| `experiment.analysis.plot_pez` | reused for the layerwise PEZ band via `ctx.plot_pez` | none | rendering (layerwise) |

**Real YAML — `attention_distance` mode** (`configs/analysis/blender_toy_dataset/vjepa_attn_distance.yaml`, ran single-GPU on the Blender velocity set, 10 val batches, and **reproduced Fig 3**):

```yaml
experiment:
  analysis:
    model: vjepa
    task: regression
    regression:
      targets_npy: /.../data_csv/blender_toy/blender_targets.npy
      variables: [{name: direction, cols: [1, 2]}]
    stages: {vision_encoder: all}
    plot_pez: [0.2, 0.4]           # layerwise PEZ band (~1/3 depth)
    skip_base_probe: true          # encoder-only: distance capture needs no trained probe
    modes:
      attention_distance:          # default-off unless present
        enabled: true
        query_chunk: 512           # memory knob (result-invariant)
        max_batches: 10            # #val batches averaged
        # annotate: true           # (default) per-cell numbers on the heatmap
  data:
    resolution: 224                # paper geometry: 14x14 patches x 8 tubelets = 1568 tokens
    num_classes: 4
  optimization:
    num_epochs: 0                  # no probe training
    cache_features: false          # attention capture runs fresh forwards, not the cache
```

**Real YAML — base R² probing plot** (`configs/analysis/toy_dataset/vjepa_combined.yaml`, multiple variables → one curve each; `direction` is the only series that gets an elbow marker):

```yaml
experiment:
  analysis:
    task: regression
    regression:
      variables:
        - {name: speed,     cols: [0]}
        - {name: direction, cols: [1, 2]}   # <- the only series that gets an elbow marker
        - {name: accel_mag, cols: [3]}
    stages: {vision_encoder: all}
    plot: true
    plot_pez: [0.2, 0.4]           # PEZ shading; false/omit = off
    probes: [{type: linear, pooling: mean, pre_norm: true}]
```

Output of the real `attention_distance` run:
`configs/analysis/blender_toy_dataset/logs/analysis_vlm/vjepa-blender-attn_distance/attention_distance/` → `attention_distance.json` (24×16), `attention_distance.png`, `attention_distance_layerwise.png`. Launched by `z_scripts/run_attn_distance_vjepa.sh` (**single GPU on purpose**: post-hoc modes dispatch on **rank 0 only**, so multi-GPU gives *no* speedup for modes — multi-GPU only helps the base probing sweep / feature cache).

---

## Invariants & gotchas

- **Default off (both surfaces).** `plot` defaults `False`; `modes` absent → mode dispatch skipped, `attention_distance.py` never imported, matplotlib never touched.
- **Never breaks a run.** All three plotters wrap the matplotlib/numpy import in `try/except` and only log a warning on failure (`plotting.py:50-52`, `attention_distance.py:85-87`, `:122-124`).
- **rank-0 only.** Both plot call sites and the mode dispatch are guarded by `rank == 0`.
- **Elbow is direction-only** — a deliberate rule (`plotting.py:104`), not a general knee detector; also requires ≥ 3 points and `ex != bx`.
- **`chance` line requires `num_classes`** — the clip harness passes none, so its accuracy plots show **no** chance baseline.
- **Series grouping key matters.** The `name.split("_", 1)` fallback misgroups VLM stage tags; harnesses must set explicit `series` (they do, `evals/analysis_vlm/eval.py:363-374`).
- **Fraction axis is cosmetic (base plot)** — ticks are fractions but all geometry (peak/elbow/PEZ) stays in integer layer-index units; `depth == 0` guarded by `or 1`. The **layerwise** plot is the opposite: its data is *already* a fraction, so it shades PEZ with `pez[0..1]` directly.
- **`val_acc` carries *best*-so-far values** in both base call sites, despite the parameter name.
- **`temporal_distance` is captured but not plotted** — both mode plotters read `spatial_distance` only. The JSON still contains the temporal matrix.
- **`Blues_r` + `origin="lower"`** → **LOW distance = DARK** cells, so local heads stand out; annotation text flips to white on those dark cells at the mid-value threshold.
- **Mode plotters bypass `plotting.py` entirely** — no shared helper yet; the frac-axis label and PEZ band are duplicated inline (see DRY note). `plotting.py` remains exactly `plot_layer_val_acc` + `_elbow_x`.
- **Mode knob semantics:** `query_chunk` is purely a memory knob (result-invariant); `max_batches` changes how many val batches are averaged (a sampling knob, converges); `annotate` is render-only. None of them changes the encoder output — capture is a detached side computation (see `11-attention-hooks`).
- **Fig 3 reproduced on Blender:** unusually-local low-distance (dark) heads cluster in the **middle layers (~5–13 = Physics Emergence Zone)**; early/late layers are uniformly long-range. `Dbar` dips and `S` (head specialization) spikes at the PEZ.

---

## Cross-references

- [`02-analysis-vlm-harness`](02-analysis-vlm-harness.md) — VLM harness: `series`/`stage`/`plot_pez` wiring (`eval.py:363-374`, `:150-153`), the `plot_layer_val_acc` call (`:556-563`), and the modes dispatch (`:565-590`) → `stage_val_acc.png`.
- [`05-analysis-clip-harness`](05-analysis-clip-harness.md) — clip/V-JEPA harness (`evals/analysis/eval.py`) that emits `layer_val_acc.png` with the **defaults-only** call.
- [`11-attention-hooks`](11-attention-hooks.md) — the SDPA attention patch + `AttentionDistanceCollector` that produce the `spatial_distance` matrix the mode plotters consume.
- [`12-analysis-modes`](12-analysis-modes.md) — the modes registry / `run_modes` dispatch and the `attention_distance` mode `run()`; where the two inline plotters are invoked.
- [`13-configs-reference`](13-configs-reference.md) — full YAML knob reference, including the `modes.attention_distance` block and `plot` / `plot_pez`.
- [`14-reproduction-status-and-findings`](14-reproduction-status-and-findings.md) — Fig 3 heatmap + Fig 19 dual-axis reproduction on Blender, phase status (Phases 0–1 done; Phase 4 `plot_direction_tuning` pending).
