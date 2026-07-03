# CSV / targets.npy builders

## Purpose

Two small offline scripts that turn a per-video **metadata** source into the two artifacts the
`analysis_vlm` regression probing harness expects:

1. a `targets.npy` of shape `(N, D)` holding the **continuous** ground-truth values, and
2. index CSVs of the form `<video_path> <int_label>`, where the integer label is a **row index**
   into that `.npy` (not a class id).

- `data_csv/make_blender_targets.py` — purpose-built for the Blender toy-physics dataset
  (`data_gen/blender_toy_dataset`). Emits one shared `blender_targets.npy` `(672, 4)` =
  `[speed, sinθ, cosθ, accel_mag]` with NaN masking, and 6 split CSVs using a **global combined
  row index** (velocity rows `0..391`, acceleration rows `392..671`).
- `data_csv/make_regression_targets.py` — a **generic** scalar/angle target builder for R2R /
  IntPhys style datasets, where the `.npy` row is simply the video-id (filename stem).

Both feed `experiment.analysis.task: regression` in the encoder-probing harness. See
`z_tech` section *vlm-encoder-probing* / `evals/analysis_vlm` for the consumer side.

## What changed vs upstream V-JEPA2

These are **entirely new, fork-local tooling** — upstream V-JEPA2 (base `204698b`, "Fix figure
(#143)") has no `data_csv/` tree at all.

| Item | Status | Notes |
|------|--------|-------|
| `data_csv/make_blender_targets.py` | **new file** | Blender-specific builder |
| `data_csv/make_regression_targets.py` | **new file** | generic R2R/IntPhys builder |
| `data_csv/blender_toy/*` | **new (generated)** | targets.npy + 6 split CSVs |
| `data_csv/toy_physics/*` | **new (generated)** | older `physics_data` variant (see below) |

> **Gotcha — these files are git-ignored.** `.gitignore:40` is a bare `*csv` glob, which matches
> the directory name `data_csv` itself (it ends in "csv"). Consequence:
> - the **entire** `data_csv/` tree is untracked, **including both builder `.py` scripts**;
> - every `*.csv` in the whole repo is ignored.
>
> So `git diff 204698b -- data_csv/…` shows nothing and `git log` for these files is empty — they
> exist only in the working tree. Treat them as local-only build artifacts + build scripts, not
> committed fork history. (`data_gen/blender_toy_dataset` is also ignored, via `data_gen/` at
> `.gitignore:48`.)

The tracked, committed part of this fork that these builders serve is the
`evals/analysis_vlm/` regression harness.

---

## The CSV ↔ targets contract

Defined by the consumer, not the builders. Key lines:

- `evals/analysis_vlm/eval.py:178-179` — *"The CSV integer label INDEXES regression.targets_npy…
  the harness maps label→target vector"*, so the shared clip/VLM dataloaders stay unchanged.
- `evals/analysis_vlm/data.py:56` — `return self._load(i), int(self.labels[i])` — the dataset
  yields `(clip, int_label)`; that label is used verbatim to gather `targets_t[label]`.
- `evals/analysis_vlm/eval.py:195-207` — load + **per-column, NaN-aware standardization**:

```python
targets_arr = np.load(tpath).astype(np.float32)          # (N, D)
mu = np.nanmean(targets_arr, axis=0, keepdims=True)
sd = np.nanstd(targets_arr, axis=0, keepdims=True)
targets_arr = (targets_arr - mu) / np.clip(sd, 1e-6, None)  # R^2 invariant; NaN stays NaN
var_cfg = reg_cfg.get("variables") or [{"name": ..., "cols": list(range(D))}]
reg_vars = [(v["name"], [int(c) for c in v["cols"]]) for v in var_cfg]
```

Contract summary:

- **Row = label.** The int after the path in each CSV line is the row of `targets.npy`.
- **Column-slices = variables.** `regression.variables` names each variable and its `cols`; each
  becomes its own R² curve on the same plot (paper Fig. 2c: speed / direction / accel together).
- **NaN = masked.** A column defined only on a subset of videos (e.g. `speed` only on velocity
  rows) is `NaN` elsewhere; standardization is NaN-aware and that head is masked per-row.
- **Angles are `(sin, cos)`**, a 2-column circular encoding — never a raw degree scalar.

---

## `make_blender_targets.py` (Blender toy-physics)

Signature / CLI (`data_csv/make_blender_targets.py:26-33`):

```
--data_dir     dataset dir with metadata.csv, velocity/, acceleration/   (required)
--out_dir      output dir                                                (required)
--path_prefix  abs prefix prepended to CSV video paths (default = abspath(data_dir))
--val_frac     0.2      --seed 0
```

### Input

`<data_dir>/metadata.csv`, one row per video. Relevant columns (observed header):
`vid, file, dataset, direction_deg, speed, accel, …`. `dataset ∈ {velocity, acceleration}`;
`velocity` rows have `speed` filled + empty `accel`, `acceleration` rows the reverse; **all** rows
have `direction_deg`. In the current dataset: **392 velocity + 280 acceleration = 672**.

### Targets array — `(672, 4)`

`blender_targets.npy` columns = `[speed(m/s), sinθ, cosθ, accel_mag(m/s²)]`, `float32`,
NaN-initialised (`make_blender_targets.py:40-52`):

```python
n_vel = sum(r["dataset"] == "velocity" for r in rows)   # 392
targets = np.full((n, 4), np.nan, dtype=np.float32)
for r in rows:
    v  = int(r["vid"])
    gi = v if r["dataset"] == "velocity" else n_vel + v   # GLOBAL row index
    th = np.deg2rad(float(r["direction_deg"]))
    targets[gi, 1] = np.sin(th); targets[gi, 2] = np.cos(th)   # direction: ALL rows
    if r["dataset"] == "velocity": targets[gi, 0] = float(r["speed"])   # accel col -> NaN
    else:                          targets[gi, 3] = float(r["accel"])   # speed col -> NaN
```

- **Global combined index** (invariant): velocity `vid v → row v` (`0..391`);
  acceleration `vid a → row 392 + a` (`392..671`).
- NaN pattern: velocity rows → `accel_mag = NaN`; acceleration rows → `speed = NaN`;
  `sinθ, cosθ` present everywhere. Asserted at the end (`:86-88`).

### Split CSVs

Deterministic per-dataset split, then combined = concat (`:58-79`):

```python
rng = random.Random(args.seed)
def split(lst):
    lst = sorted(lst); rng.shuffle(lst)
    k = int(round(len(lst) * (1 - args.val_frac)))
    return lst[:k], lst[k:]
```

Each line is `f"{path} {gi}\n"` where `path = os.path.join(prefix, r["file"])` and `gi` is the
**global** row index. Produces 6 files:

| File | Rows | Labels |
|------|------|--------|
| `velocity_train.csv` / `velocity_val.csv` | 314 / 78 | `0..391` |
| `acceleration_train.csv` / `acceleration_val.csv` | 224 / 56 | `392..671` |
| `combined_train.csv` / `combined_val.csv` | 538 / 134 | `0..671` |

`blender_targets.npy` is shared by all six (indexed by the global label).

### `path_prefix` (the vll4 detail)

The CSV stores **absolute** video paths, baked at build time from `path_prefix` (default =
`abspath(data_dir)`). On the **vll4** cluster the data lives at
`/local_datasets/world/blender_toy_dataset`, so the CSVs were built with
`--path_prefix /local_datasets/world/blender_toy_dataset`, e.g.:

```
/local_datasets/world/blender_toy_dataset/velocity/000028.mp4 28
/local_datasets/world/blender_toy_dataset/acceleration/000017.mp4 409
```

> **Invariant:** the CSVs and the `.npy` are only valid **as a pair from the same run** — the
> `392` offset is `n_vel`, which is recomputed from the metadata. Change the velocity count and
> the global indices shift; a stale `.npy` will silently mislabel. Moving the data also requires a
> rebuild (or a matching prefix) because paths are absolute.

---

## `make_regression_targets.py` (generic R2R / IntPhys)

A general builder where the `.npy` **row is the video-id = filename stem** (not a combined space).

Signature / CLI (`data_csv/make_regression_targets.py:54-62`):

```
--metadata    metadata json/csv (per-video fields)                       (required)
--split_csv   one or more existing "<path> <class>" CSVs to relabel      (required, nargs=+)
--out_dir  --out_prefix                                                  (required)
--var         name:type:field   (repeatable; type = scalar | angle)      (required)
--cat-map     "right=0,up=90,left=180,down=270"   (map category -> degrees)
```

Core logic:

- `vid_of(path) = int(os.path.splitext(os.path.basename(path))[0])` (`:32-33`) — **filename stem
  must be an int**.
- `load_metadata` (`:36-51`) accepts a JSON list of dicts (keyed by `id`, or `vid_of(e["video"])`)
  or a CSV with an `id`/`video`/`file_name` column.
- Var typing (`:80-101`): `scalar:<field>` → 1 col `float(raw)`; `angle:<field>` → 2 cols
  `(sin, cos)` of `np.deg2rad(deg)`, where `deg` is the field value or, if categorical, its
  `--cat-map` mapping.
- `n = max(ids) + 1`; `targets = np.full((n, D), np.nan)`. Asserts **no NaN among referenced ids**
  (`:104`) — every video in the splits must have a value for every variable.
- Rewrites each input split `<stem>.csv → <stem>_reg.csv` as `"<path> <vid_of(path)>"`
  (`:110-121`), and **prints the `regression.variables` YAML block** to paste into the config
  (`:123-128`).

> **Difference from the Blender builder:** rows are self-indexed by filename stem, so this is
> unsuitable for *combining* two datasets whose stems overlap (they would collide on the same
> row). That collision is exactly why the Blender dataset needs its own builder with the explicit
> `392 + vid` offset.

---

## Produced files

### `data_csv/blender_toy/` (source: `data_gen/blender_toy_dataset`, vll4 prefix `/local_datasets/world/blender_toy_dataset`)

| File | Shape / lines | Notes |
|------|---------------|-------|
| `blender_targets.npy` | `(672, 4)` float32 | `[speed, sinθ, cosθ, accel_mag]`, NaN-masked |
| `velocity_train.csv` / `velocity_val.csv` | 314 / 78 | labels `0..391` |
| `acceleration_train.csv` / `acceleration_val.csv` | 224 / 56 | labels `392..671` |
| `combined_train.csv` / `combined_val.csv` | 538 / 134 | labels `0..671` |

### `data_csv/toy_physics/` (older `physics_data` source, prefix `/local_datasets/world/physics_data`)

An earlier variant over a related `physics_data` set. **Provenance is not in git** (ignored), but
the on-disk shapes reveal two labelling schemes coexisting — the generic per-task style and the
Blender combined style — plus full unsplit `*_all.csv` listings:

| File | Shape / lines | Scheme |
|------|---------------|--------|
| `velocity_targets.npy` | `(392, 3)` | per-task, self-indexed; cols `[speed, sinθ, cosθ]` |
| `acceleration_targets.npy` | `(280, 3)` | per-task, self-indexed; cols `[accel, sinθ, cosθ]` |
| `combined_targets.npy` | `(672, 4)` | combined/global; cols `[speed, sinθ, cosθ, accel_mag]`, NaN-masked |
| `velocity_all/train/val.csv` | 392 / 314 / 78 | labels `0..391` (index `velocity_targets.npy`) |
| `acceleration_all/train/val.csv` | 280 / 224 / 56 | labels `0..279` (**local**, index `acceleration_targets.npy`) |
| `combined_all/train/val.csv` | 672 / 538 / 134 | labels `0..671` (global, index `combined_targets.npy`) |

> **Gotcha:** in `toy_physics` the per-task acceleration CSVs use **local** labels `0..279` (they
> pair with the 3-col `acceleration_targets.npy`), whereas the `combined_*` CSVs use **global**
> `392..671` (pairing with `combined_targets.npy`). **Never cross-pair** a CSV with the wrong
> `.npy` — the label range must match the array's row space. The newer `blender_toy` folder
> avoids this by shipping a single combined `.npy` for all splits.

---

## Config (regression harness)

The builders don't read a YAML; they *produce* the artifacts a probing run points at.
`make_regression_targets.py` even prints the block to paste. Minimal example wiring the combined
Blender targets into `experiment.analysis`:

```yaml
experiment:
  analysis:
    task: regression                 # default is 'classification'; MUST be set
    regression:
      targets_npy: /data/.../vjepa2/data_csv/blender_toy/blender_targets.npy   # (672,4)
      variables:
        - {name: speed,     cols: [0]}     # velocity rows only (NaN on accel rows -> masked)
        - {name: direction, cols: [1, 2]}  # sin, cos (circular) — all rows
        - {name: accel_mag, cols: [3]}     # acceleration rows only
data:
  # dataloader reads these CSVs; the int label indexes targets_npy above
  train: /data/.../vjepa2/data_csv/blender_toy/combined_train.csv
  val:   /data/.../vjepa2/data_csv/blender_toy/combined_val.csv
```

Notes / defaults:
- `task` defaults to `classification` (`eval.py:189`); regression is **opt-in** and asserts
  `regression.targets_npy` is present (`eval.py:194`).
- `regression.targets` is accepted as an alias for `targets_npy` (`eval.py:193`).
- If `variables` is omitted, one variable spanning **all** columns is used (`eval.py:205-206`).
- Standardization (mean/std) is applied for training stability only; **R² is invariant** to it, so
  reported metrics are in raw-target terms. Different units across columns (m/s vs m/s² vs sin/cos
  in `[-1,1]`) are therefore fine.

## Gotchas / invariants (summary)

- **Local-only:** everything under `data_csv/` (scripts + outputs) is git-ignored via `*csv`
  (`.gitignore:40`); not in upstream, not in fork history.
- **Row == label**, always. Column-slices == variables. NaN == per-row masked head.
- **Blender global index** = `vid` for velocity, `392 + vid` for acceleration; valid only with the
  `.npy` from the same build (`392 = n_vel`, recomputed from metadata).
- **Absolute paths** are baked into CSVs at build time (`path_prefix`); relocating data ⇒ rebuild.
- **Generic builder** self-indexes by filename stem ⇒ cannot merge stem-overlapping datasets.
- **Never cross-pair** a CSV label range with a mismatched `.npy` (esp. `toy_physics` local vs
  global variants).
