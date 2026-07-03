# 10 — CSV / targets.npy builders

> Two offline scripts that convert per-video **metadata** into the artifact pair the `analysis_vlm`
> regression harness consumes: an `(N, D)` **`targets.npy`** of continuous ground truth plus
> `<video_path> <int_label>` **index CSVs**, where the integer is a **row index** into the `.npy`
> (never a class id).

## Purpose

The `analysis_vlm` layer-wise probing harness runs `task: regression` by asking each CSV line for an
**integer label** and treating that integer as a **row index** into a shared `(N, D)` `targets.npy`
of continuous ground truth. Two fork-local builders produce that pair from raw per-video metadata:

| Builder | Scope | Row-index scheme | `targets.npy` |
|---------|-------|------------------|---------------|
| `data_csv/make_blender_targets.py` | purpose-built for the Blender toy-physics set (`data_gen/blender_toy_dataset`) | **global combined** index (velocity `0..391`, acceleration `392..671`) | one shared `blender_targets.npy` `(672, 4)` = `[speed, sinθ, cosθ, accel_mag]`, NaN-masked |
| `data_csv/make_regression_targets.py` | **generic** scalar/angle builder for R2R / IntPhys-style sets | **self-index by filename stem** (`row = int(stem)`) | `<prefix>_targets.npy` `(max_stem+1, D)`, columns = the requested `--var`s |

Both feed `experiment.analysis.task: regression`. See section
[02 — analysis_vlm harness](./02-analysis-vlm-harness.md) and
[04 — Probes, regression & NaN-masking](./04-probes-regression-nanmask.md) for the consumer side.

## What changed vs upstream V-JEPA2

Everything here is **entirely new, fork-local tooling**. Upstream V-JEPA2 (base `204698b`, "Fix
figure (#143)") has no `data_csv/` tree at all — `git diff 204698b -- data_csv/…` returns **nothing**
because the tree is git-ignored (see the gotcha below), so there is no upstream baseline to delta
against. These are additive build scripts + their generated outputs.

| Item | Status | Notes |
|------|--------|-------|
| `data_csv/make_blender_targets.py` | **new file** | Blender-specific builder (93 lines) |
| `data_csv/make_regression_targets.py` | **new file** | generic R2R/IntPhys builder (132 lines) |
| `data_csv/blender_toy/*` | **new (generated)** | `blender_targets.npy` + 6 split CSVs |
| `data_csv/toy_physics/*` | **new (generated)** | older `physics_data` variant (12 files) |
| `data_csv/R2R_4way_1500/*` | **new (generated)** | R2R 4-way class CSVs + `regression/` reg outputs |
| `data_csv/IntPhys2/*` | **new (staged)** | IntPhys2 2-way class CSVs + `metadata.csv` (never regressed) |

**Default-off guarantee (consumer side).** These are offline scripts with no runtime flag, but the
harness they feed is opt-in: `experiment.analysis.task` **defaults to `classification`**
(`eval.py:189`); `regression` is entered only when explicitly set, and then asserts
`regression.targets_npy` is present (`eval.py:194`). A run that never sets `task: regression` never
touches any of these artifacts.

> **Gotcha — the whole tree is git-ignored.** `.gitignore:40` is a bare `*csv` glob. It matches the
> directory name `data_csv` itself (it ends in "csv"), so `git check-ignore` reports the **entire**
> `data_csv/` subtree as ignored — including both builder `.py` scripts and every `.npy`:
> ```
> .gitignore:40:*csv   data_csv
> .gitignore:40:*csv   data_csv/make_blender_targets.py
> .gitignore:40:*csv   data_csv/blender_toy/blender_targets.npy
> ```
> Consequences: `git ls-files data_csv` is empty, `git log`/`git diff` show nothing, and every
> `*.csv` in the whole repo is ignored. Treat these as **local-only** build scripts + artifacts, not
> committed fork history. (`data_gen/blender_toy_dataset` is separately ignored via `data_gen/` at
> `.gitignore:48`.)

The tracked, committed part of the fork these builders serve is the `evals/analysis_vlm/` harness.

## Design & data flow

```
metadata (csv/json)  ──►  builder script  ──►  targets.npy  (N, D)  continuous ground truth
                                          └──►  <split>.csv  lines: "<abs_video_path> <row_index>"
                                                                                    │
analysis_vlm dataloader reads CSV ── yields (clip, int_label) ── gathers targets[label] ── probe → R²
```

### The CSV ↔ targets contract (defined by the **consumer**, not the builders)

| Rule | Meaning | Where enforced |
|------|---------|----------------|
| **Row = label** | the int after the path in each CSV line is a **row of `targets.npy`** | `eval.py:178` comment; `data.py:56` yields `int(self.labels[i])` |
| **Column-slice = variable** | `regression.variables` names each variable + its `cols`; each is its own R² curve on one plot | `eval.py:204-207` |
| **NaN = per-row masked head** | a column defined on only a subset of videos is `NaN` elsewhere; standardization is NaN-aware and that head is masked per-row | `eval.py:201-203` |
| **Angle = `(sin, cos)`** | direction is a 2-column circular encoding, never a raw-degree scalar | both builders |

The relevant consumer comment block is `eval.py:177-188`; the load / NaN-aware standardization is
`eval.py:195-207`:

```python
# eval.py:178  "The CSV integer label INDEXES regression.targets_npy ((N,D) .npy), so the
# eval.py:180   dataloaders stay unchanged — the harness maps label->target vector."
targets_arr = np.load(tpath).astype(np.float32)             # :195  (N, D)
mu = np.nanmean(targets_arr, axis=0, keepdims=True)         # :201
sd = np.nanstd(targets_arr, axis=0, keepdims=True)          # :202
targets_arr = (targets_arr - mu) / np.clip(sd, 1e-6, None)  # :203  R^2 invariant; NaN stays NaN
var_cfg = reg_cfg.get("variables")                          # :204
if not var_cfg:                                             # :205  default: one var over all cols
    var_cfg = [{"name": reg_cfg.get("name", "target"), "cols": list(range(targets_arr.shape[1]))}]
reg_vars = [(v["name"], [int(c) for c in v["cols"]]) for v in var_cfg]   # :207
```

- `data.py:56` — `return self._load(i), int(self.labels[i])` — the dataset yields `(clip, int_label)`
  and that label is used verbatim to gather `targets_arr[label]`. Both the shared clip
  `VideoDataset` and the VLM raw path are untouched.
- Standardization is for **training stability only**; **R² is invariant** to the affine transform, so
  reported metrics are in raw-target terms and columns with different units (m/s vs m/s² vs
  `sin/cos ∈ [-1,1]`) coexist safely.

### Why two builders

The generic builder self-indexes by filename stem, so it **cannot merge two datasets whose stems
overlap** — they would collide on the same `.npy` row. The Blender set combines two subsets
(velocity + acceleration) whose `vid`s both start at 0, so it needs its own builder with an explicit
`392 + vid` offset. That single difference is the reason both scripts exist.

## Key code

### `make_blender_targets.py` — Blender toy-physics

CLI (`make_blender_targets.py:26-33`):

| Flag | Meaning | Default |
|------|---------|---------|
| `--data_dir` | dataset dir with `metadata.csv`, `velocity/`, `acceleration/` | **required** |
| `--out_dir` | output dir | **required** |
| `--path_prefix` | abs prefix prepended to CSV video paths | `abspath(data_dir)` |
| `--val_frac` | val fraction of the per-dataset split | `0.2` |
| `--seed` | RNG seed for the shuffle | `0` |

**Input** — `<data_dir>/metadata.csv`, one row per video. Columns used: `vid, file, dataset,
direction_deg, speed, accel`. `dataset ∈ {velocity, acceleration}`; velocity rows fill `speed` (empty
`accel`), acceleration rows the reverse; **all** rows carry `direction_deg`. Current set: **392
velocity + 280 acceleration = 672**.

**Targets `(672, 4)` = `[speed(m/s), sinθ, cosθ, accel_mag(m/s²)]`**, NaN-initialised
(`make_blender_targets.py:37-52`):

```python
n_vel = sum(r["dataset"] == "velocity" for r in rows)   # 392
targets = np.full((n, 4), np.nan, dtype=np.float32)
for r in rows:
    v = int(r["vid"])
    gi = v if r["dataset"] == "velocity" else n_vel + v         # GLOBAL row index
    th = np.deg2rad(float(r["direction_deg"]))
    targets[gi, 1] = np.sin(th); targets[gi, 2] = np.cos(th)    # direction: ALL rows
    if r["dataset"] == "velocity": targets[gi, 0] = float(r["speed"])   # accel col -> NaN
    else:                          targets[gi, 3] = float(r["accel"])   # speed col -> NaN
```

- **Global combined index** (invariant): velocity `vid v → row v` (`0..391`); acceleration
  `vid a → row 392 + a` (`392..671`).
- **NaN pattern** (verified on disk — NaN-per-column `[280, 0, 0, 392]`): 280 velocity rows have
  `accel_mag = NaN`; 392 acceleration rows have `speed = NaN`; `sinθ, cosθ` present everywhere.
  Asserted at `make_blender_targets.py:86-88`.

**Split CSVs** — deterministic per-dataset split, then `combined = velocity ⧺ acceleration`
(`make_blender_targets.py:58-79`):

```python
rng = random.Random(args.seed)
def split(lst):
    lst = sorted(lst); rng.shuffle(lst)
    k = int(round(len(lst) * (1 - args.val_frac)))
    return lst[:k], lst[k:]
```

Each line is `f"{path} {gi}\n"` with `path = os.path.join(prefix, r["file"])` and `gi` the **global**
row index. Six files are written (`:75-82`); `blender_targets.npy` is shared by all six.

**`path_prefix` (the vll4 detail).** CSVs store **absolute** video paths baked at build time from
`path_prefix` (default `abspath(data_dir)`). On the **vll4** cluster the data lives at
`/local_datasets/world/blender_toy_dataset`, so the CSVs were built with
`--path_prefix /local_datasets/world/blender_toy_dataset`. Verified on disk:

```
/local_datasets/world/blender_toy_dataset/velocity/000028.mp4 28
/local_datasets/world/blender_toy_dataset/acceleration/000017.mp4 409     # 409 = 392 + 17
```

### `make_regression_targets.py` — generic R2R / IntPhys

CLI (`make_regression_targets.py:54-62`):

| Flag | Meaning | Default |
|------|---------|---------|
| `--metadata` | per-video metadata `.json` (list of dicts) or `.csv` | **required** |
| `--split_csv` | one or more existing `"<path> <class>"` CSVs to relabel (`nargs="+"`) | **required** |
| `--out_dir` | output dir | **required** |
| `--out_prefix` | stem for `<prefix>_targets.npy` | **required** |
| `--var` | `name:type:field`, repeatable (`type = scalar \| angle`) | **required** |
| `--cat-map` | category→degrees map, e.g. `"right=0,up=90,left=180,down=270"` | `None` |

Core logic:

- `vid_of(path) = int(os.path.splitext(os.path.basename(path))[0])` (`:32-33`) — **the filename stem
  must be an integer.**
- `load_metadata` (`:36-51`) accepts a JSON list of dicts (keyed by `e["id"]` or `vid_of(e["video"])`)
  or a CSV with an `id` / `video` / `file_name` column.
- Var typing (`:80-101`): `scalar:<field>` → 1 col `float(raw)`; `angle:<field>` → 2 cols
  `(sin, cos)` of `np.deg2rad(deg)`, where `deg` is the field value or, if categorical, its
  `--cat-map` mapping.
- `n = max(ids) + 1`; `targets = np.full((n, D), np.nan)`; asserts **no NaN among referenced ids**
  (`:104`) — every video in the splits must have a value for every variable.
- Rewrites each input split `<stem>.csv → <stem>_reg.csv` as `"<path> <vid_of(path)>"` (`:110-121`),
  then **prints the `regression.variables` YAML block** to paste into the config (`:123-128`).

The script's header example (`make_regression_targets.py:17-23`) is the one that was actually run —
it produced `data_csv/R2R_4way_1500/regression/` (see Produced files):

```
python data_csv/make_regression_targets.py \
  --metadata /local_datasets/vlm_direction/vlm_direction_testbed/R2R_video_1500/shape_color_metadata.json \
  --split_csv data_csv/R2R_4way_1500_shape_color_train.csv data_csv/R2R_4way_1500_shape_color_val.csv \
  --out_dir data_csv/regression --out_prefix R2R_shape_color \
  --var speed:scalar:speed --var direction:angle:direction \
  --cat-map "right=0,up=90,left=180,down=270"
```

## Configuration

The builders don't read a YAML; they *produce* the artifacts a probing run points at, and
`make_regression_targets.py` prints the block to paste. Minimal wiring of the combined Blender
targets into `experiment.analysis`:

```yaml
experiment:
  analysis:
    task: regression                 # default is 'classification'; MUST be set to opt in
    regression:
      targets_npy: /data/.../vjepa2/data_csv/blender_toy/blender_targets.npy   # (672, 4)
      variables:
        - {name: speed,     cols: [0]}     # velocity rows only (NaN on accel rows -> masked)
        - {name: direction, cols: [1, 2]}  # sin, cos (circular) — all rows
        - {name: accel_mag, cols: [3]}     # acceleration rows only
data:
  # dataloader reads these CSVs; the int label indexes targets_npy above
  train: /data/.../vjepa2/data_csv/blender_toy/combined_train.csv
  val:   /data/.../vjepa2/data_csv/blender_toy/combined_val.csv
```

| Key | Meaning | Default | Allowed values |
|-----|---------|---------|----------------|
| `analysis.task` | task type | `classification` | `classification` \| `regression` |
| `regression.targets_npy` | path to `(N, D)` `.npy` | — (asserted present) | any readable `.npy` |
| `regression.targets` | alias for `targets_npy` (`eval.py:193`) | — | — |
| `regression.variables` | list of `{name, cols}` column-slices | one var over **all** cols (`eval.py:205-206`, name = `reg_cfg.get("name","target")`) | list of dicts |
| `regression.name` | fallback var name when `variables` omitted | `target` | any string |

## Produced files (on-disk inventory)

Four output trees currently exist under `data_csv/`. `blender_toy/` + `toy_physics/` come from
`make_blender_targets.py`; `R2R_4way_1500/regression/` from `make_regression_targets.py`; the R2R and
IntPhys2 classification CSVs are hand-staged inputs.

### `data_csv/blender_toy/` — `make_blender_targets.py` output (prefix `/local_datasets/world/blender_toy_dataset`)

| File | Shape / lines | Notes |
|------|---------------|-------|
| `blender_targets.npy` | `(672, 4)` float32 | `[speed, sinθ, cosθ, accel_mag]`; NaN-per-col `[280, 0, 0, 392]` |
| `velocity_train.csv` / `velocity_val.csv` | 314 / 78 | labels `0..391` |
| `acceleration_train.csv` / `acceleration_val.csv` | 224 / 56 | labels `392..671` |
| `combined_train.csv` / `combined_val.csv` | 538 / 134 | labels `0..671` |

### `data_csv/toy_physics/` — older `physics_data` variant (prefix `/local_datasets/world/physics_data`)

An earlier build over a related `physics_data` set. Provenance is not in git, but the on-disk shapes
show **two labelling schemes coexisting** (per-task self-indexed *and* Blender-style combined) plus
full unsplit `*_all.csv` listings:

| File | Shape / lines | Scheme |
|------|---------------|--------|
| `velocity_targets.npy` | `(392, 3)` | per-task, self-indexed; `[speed, sinθ, cosθ]` (no NaN) |
| `acceleration_targets.npy` | `(280, 3)` | per-task, self-indexed; `[accel, sinθ, cosθ]` (no NaN) |
| `combined_targets.npy` | `(672, 4)` | combined/global; `[speed, sinθ, cosθ, accel_mag]`, NaN-per-col `[280, 0, 0, 392]` |
| `velocity_all/train/val.csv` | 392 / 314 / 78 | labels `0..391` → `velocity_targets.npy` |
| `acceleration_all/train/val.csv` | 280 / 224 / 56 | labels `0..279` (**local**) → `acceleration_targets.npy` |
| `combined_all/train/val.csv` | 672 / 538 / 134 | labels `0..671` (**global**) → `combined_targets.npy` |

> **Gotcha:** in `toy_physics` the per-task acceleration CSVs use **local** labels `0..279` (paired
> with the 3-col `acceleration_targets.npy`), whereas `combined_*` uses **global** `392..671` (paired
> with `combined_targets.npy`). Verified on disk: `acceleration_train` max label = `279`,
> `combined_train` max label = `671`. **Never cross-pair** a CSV with the wrong `.npy`. The newer
> `blender_toy/` avoids this by shipping a single combined `.npy` for all splits.

### `data_csv/R2R_4way_1500/` — R2R 4-way (prefix `/local_datasets/vlm_direction/vlm_direction_testbed/R2R_video_1500`)

Two 4-way **classification** attributes (`shape_color`, `obj_place`), each a balanced 6000-clip set
(labels `0..3`, 1200 per class in each split), plus a **regression** subtree derived from
`shape_color` by the generic builder:

| File | Shape / lines | Notes |
|------|---------------|-------|
| `R2R_4way_1500_obj_place_all/train/val.csv` | 6000 / 4800 / 1200 | classification, labels `0..3` — **no regression output derived** |
| `R2R_4way_1500_shape_color_all/train/val.csv` | 6000 / 4800 / 1200 | classification, labels `0..3` (input to the generic builder) |
| `regression/R2R_shape_color_targets.npy` | `(6000, 3)` float32 | `[speed(scalar), sin(direction), cos(direction)]`; **no NaN** (all 6000 rows referenced); `--var speed:scalar:speed --var direction:angle:direction --cat-map "right=0,up=90,left=180,down=270"` |
| `regression/R2R_4way_1500_shape_color_train_reg.csv` | 4800 | labels = **filename stem int** (e.g. `…/003355.mp4 3355`) |
| `regression/R2R_4way_1500_shape_color_val_reg.csv` | 1200 | same |

This `regression/` subtree is the **one concrete on-disk output of `make_regression_targets.py`** —
it closes the loop of the script's own header example. Note `obj_place` was **never** run through the
generic builder (no `_reg.csv`, no `_targets.npy`).

### `data_csv/IntPhys2/` — IntPhys2 2-way (prefix `/local_datasets/world/IntPhys2/Main/Videos`)

Binary **possible/impossible** classification CSVs plus a raw `metadata.csv`. **No `_reg.csv` and no
`_targets.npy` exist here** — IntPhys2 is staged as *input* for the generic builder but was **never
run** (only R2R has been):

| File | Shape / lines | Notes |
|------|---------------|-------|
| `IntPhys2_2way_all/train/val.csv` | 1012 / 808 / 204 | binary labels `0/1`, balanced (train 404/404) |
| `IntPhys2_2way_easy_all/train/val.csv` | 104 / 84 / 20 | "Easy"-difficulty subset, binary |
| `metadata.csv` | 1013 lines (1012 rows + header) | cols `SceneIndex, name, file_name, game_name, condition, env, type, occluder, Difficulty, Camera` |

> **Freshness/accuracy nuance:** only **R2R (`shape_color`)** has actually been run through
> `make_regression_targets.py`. IntPhys2 is a classification task and, additionally, its filenames are
> 64-char hex hashes (e.g. `56c044…c47c.mp4`) — `vid_of()` does `int(stem)` and would raise on a hex
> stem — so IntPhys2 cannot be regressed by this builder as-is. Treat `IntPhys2/` and
> `R2R_4way_1500/*_obj_place_*` as classification data only.

## Invariants & gotchas

- **Local-only.** Everything under `data_csv/` (scripts + outputs) is git-ignored via `*csv`
  (`.gitignore:40`); not in upstream, not in fork history. `data_gen/` is ignored via `.gitignore:48`.
- **Row == label**, always. Column-slices == variables. NaN == per-row masked head.
- **Blender global index** = `vid` for velocity, `392 + vid` for acceleration; valid **only** with the
  `.npy` from the same build (`392 = n_vel`, recomputed from metadata each run). A stale `.npy` will
  silently mislabel. Change the velocity count and every acceleration index shifts.
- **Absolute paths** are baked into CSVs at build time (`path_prefix`); relocating the data requires a
  rebuild (or a matching prefix).
- **Generic builder self-indexes by filename stem** → cannot merge stem-overlapping datasets, and the
  stem **must be an int** (`vid_of` = `int(stem)`).
- **Never cross-pair** a CSV's label range with a mismatched `.npy` (especially the `toy_physics`
  local-`0..279` vs global-`392..671` acceleration variants).
- **Standardization is affine and R²-invariant** — reported metrics are raw-target; mixed units across
  columns are fine.
- **Consumed by** the `attention_distance` reproduction run, which points at
  `data_csv/blender_toy/velocity_*.csv` + `blender_targets.npy` (see section
  [12 — Analysis modes](./12-analysis-modes.md)).

## Cross-references

- [02 — analysis_vlm harness (eval flow)](./02-analysis-vlm-harness.md) — the consumer that reads
  these CSVs + `targets.npy`.
- [04 — Probes, regression task & NaN-masking](./04-probes-regression-nanmask.md) — how `variables` /
  NaN masking / R² are computed downstream.
- [06 — Data-pipeline changes](./06-data-pipeline-changes.md) — the dataloader (`data.py`) that yields
  `(clip, int_label)`.
- [09 — Blender toy-physics dataset generator](./09-blender-toy-dataset.md) — the `data_gen/` source
  that produces the `metadata.csv` feeding `make_blender_targets.py`.
- [12 — Analysis modes subpackage](./12-analysis-modes.md) — post-hoc modes (incl. `attention_distance`)
  that consume the Blender velocity CSVs.
- [13 — Config reference](./13-configs-reference.md) — full `experiment.analysis.regression` schema.
