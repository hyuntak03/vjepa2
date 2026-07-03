# 09 — Blender toy-physics dataset

> A paper-faithful synthetic ball-motion dataset (672 photorealistic CYCLES clips) with a **single fixed red sphere** and analytic frictionless motion, whose exact metre + pixel ground truth plugs straight into the layer-wise probing CSV builders — the fixed-appearance control that restored the paper's early-layer speed result and, downstream, reproduced its Fig. 3 attention-locality heatmap.

## Purpose

Generate a **paper-faithful** synthetic ball-motion dataset for probing motion
variables (speed, acceleration, direction) in V-JEPA2 and VLM encoders. It
replicates the controlled toy datasets of *"Interpreting Physics in Video World
Models"* (App. A.1.2) using **Blender / `bpy` + CYCLES** so the frames are
photorealistic (in-distribution for a video encoder) rather than flat 2D PIL
sprites. Ball motion is computed **analytically** (frictionless, so no physics
engine is needed) and Blender projects metres → pixels itself, so ground truth
is exact in both physical and pixel units.

Output plugs directly into the layer-wise probing CSV builders (same on-disk
layout as the older `make_physics_toy.py`):

```
<save_dir>/velocity/000000.mp4 …        <save_dir>/acceleration/000000.mp4 …
<save_dir>/metadata.csv                 <save_dir>/kinematics.json
```

**Downstream validation (the control paid off end-to-end).** This generator was
run for real (`data_gen/blender_toy_dataset/`, **672 clips, Jul 1**) and the
`velocity` split was then consumed by the `attention_distance` analysis mode to
**reproduce the paper's Fig. 3 heatmap** on frozen V-JEPA2-L: the unusually-local
(dark, low-distance) attention heads cluster in the **middle layers (~5–13, the
Physics Emergence Zone)** while early/late layers stay uniformly long-range.
Output is co-located at
`configs/analysis/blender_toy_dataset/logs/analysis_vlm/vjepa-blender-attn_distance/attention_distance/`
(`attention_distance.png` + `attention_distance_layerwise.png` + `.json`,
24 layers × 16 heads, 10 val batches). The mode itself is owned by sections
[11](11-attention-hooks.md)/[12](12-analysis-modes.md)/[13](13-configs-reference.md)/[14](14-reproduction-status-and-findings.md);
this is only the "consumed by" pointer that anchors the Purpose claim — the
**single fixed sphere** here is exactly what restored the paper's layer-0 speed /
middle-layer locality result.

## What changed vs upstream V-JEPA2

Upstream commit `204698b` has **no `data_gen/` directory at all** (verified:
`git cat-file -e 204698b:data_gen/make_physics_blender.py` → *ABSENT*) — the
entire data generator is new to this fork. All three files below are **NEW**;
none is a modification of an upstream file.

| File | Status | Role |
|------|--------|------|
| `data_gen/make_physics_blender.py` | **new** | The `bpy`/CYCLES generator (this doc). |
| `data_gen/run_blender_toy.sh` | **new** | Multi-GPU / multi-process driver + merge + sanity check. |
| `data_gen/sanity_check_blender.py` | **new** | Post-merge dataset validator. |

**Default-off / isolation guarantee.** This is a standalone offline tool under
`data_gen/`, invoked manually; it touches **no** training or eval code path, so
its existence cannot change any model run. Within the generator, GPU is itself
opt-in: `--device` defaults to **`CPU`** (`make_physics_blender.py:228`), and the
grid/scene defaults are the paper values, so a bare invocation reproduces the
paper dataset on CPU with no surprises.

**Relationship to the earlier `data_gen/make_physics_toy.py`** (also fork-only):
that predecessor is a lightweight **2D PIL** generator with an explicit
*anti-shortcut* design — it randomizes shape / colour / size / orientation
**independently** of the motion label (`make_physics_toy.py:20-46`). That
randomization is exactly why the Blender generator was written: randomizing
appearance **decorrelated early-layer motion-energy from speed**, so the paper's
layer-0 speed result did not reproduce. `make_physics_blender.py` instead uses
**one fixed sphere appearance** (a single red glossy material, identical for
every clip) to restore the paper's control (`make_physics_blender.py:64-73`,
header note lines 18-20).

## Design & data flow

```
build_specs()  ──►  672 specs  ──►  shard slice (i % nshards == shard)
  seeded RNG        vid counter      each process renders its own subset
  starts~U([-2,2]²) per dataset
        │
        ▼  per spec
  positions()  ──►  keyframe sphere along trajectory  ──►  ONE render(animation=True)
  analytic (x,y)      sph.location per frame               Cycles loops frames in C
        │                                                        │
        ▼                                                        ▼  f0000.png … f0015.png
  kinematics()  GT (metres + Blender-projected pixels)     ffmpeg CLI  ──►  <dset>/NNNNNN.mp4
        │                                                        │
        └──►  row appended to metadata_<shard>.csv  ◄────────────┘  encode_ok flag
        └──►  clip entry in kinematics_<shard>.json

  (all shards done)  ──►  --merge  ──►  metadata.csv + kinematics.json
                          rm -rf _frames_*  ──►  sanity_check_blender.py  (exit≠0 on any fault)
```

- **One Blender per process** (`bpy` is a singleton), so parallelism is *across
  processes*, sharded by `i % nshards == shard` (`make_physics_blender.py:255`).
- **`vid` is assigned before sharding** — a global contiguous counter *per
  dataset* (`:196-207`) — so clip filenames are stable regardless of shard count.
- The scene is built **once per process** and reused for every clip in that
  shard's slice; only the sphere's keyframes change between clips.

## Scene

Built once per process in `build_scene()` (`make_physics_blender.py:42-121`),
from factory-empty settings (`read_factory_settings(use_empty=True)`).

| Element | Setting | Code |
|---------|---------|------|
| **Floor** | 8×8 m plane at origin, grey Principled BSDF (roughness 0.9) + procedural `ShaderNodeTexNoise` (scale 6) → colour-ramp `(0.40,0.40,0.42)`→`(0.52,0.52,0.55)`, giving static reference features | `:48-62` |
| **Sphere** | UV sphere `r=0.3 m` (segments 48, rings 24), smooth-shaded, **fixed** red-glossy material `(0.85,0.18,0.15)`, roughness 0.35 | `:65-73` |
| **Camera** | perspective, at `(0,0,10)` looking straight down (`rotation_euler=(0,0,0)`) | `:76-80` |
| **Light** | single `SUN`, energy 3.5, **`angle=0.1`** (soft-shadow penumbra), tilted `(25°,15°)` → shading + shadow depth cue | `:83-85` |
| **World** | soft ambient background `(0.05,0.06,0.08)` so floor isn't pitch black | `:88-89` |
| **Render** | `CYCLES`, `samples` (default 24 spp), 256×256, **`resolution_percentage=100`**, PNG out (RGB, zlib compression 15), **`film_transparent=False`** (opaque background) | `:91-97, :117` |

**Camera FOV = Kubric default.** The code **never sets `cam_data.lens` or
`sensor_width`**, so the camera keeps Blender's factory defaults — **50 mm lens /
36 mm sensor** — which is exactly the Kubric default and the paper's FOV. This is
an *implicit* invariant: the paper-exact FOV depends on Blender defaults not
changing across versions (see Invariants & gotchas).

## Analytic motion & grid

`positions()` (`make_physics_blender.py:124-139`) returns per-frame world `(x,y)`
in metres, frictionless:

```python
t = i / args.fps                         # seconds
if spec["dataset"] == "velocity":
    x, y = sx + v*ux*t,        sy + v*uy*t          # constant velocity
else:
    x, y = sx + 0.5*a*ux*t*t,  sy + 0.5*a*uy*t*t    # start at rest, const accel
```

Heading `(ux,uy) = (cos θ, sin θ)`. This matches PyBullet with `dv=0` (velocity)
and `F=ma` from rest (acceleration).

The grid is built in `build_specs()` (`:184-209`): directions =
`round(360·k/n_directions, 4)` (→ `0,45,…,315` for `n=8`), start positions drawn
from a **seeded** RNG `~ U([-2,2]²)` m. `--dataset` gates which grids are built
(`:191-194`).

| Dataset | directions × magnitudes × starts | count | magnitudes |
|---------|----------------------------------|-------|-----------|
| `velocity` | 8 × 7 × 7 | **392** | speeds `{1,2,3,4,5,6,7}` m/s |
| `acceleration` | 8 × 5 × 7 | **280** | accels `{2,4,6,8,10}` m/s² |
| **total** | | **672** | 16 frames @ 24 fps (0.67 s), 256² |

## Ground truth

Two sibling artifacts describe every clip. They deserve **parallel** treatment:
`metadata.csv` is what the layer-wise probing CSV builders and
`sanity_check_blender.py` key on, while `kinematics.json` carries the dense
per-frame physical + pixel trajectory.

### `metadata.csv` — one row per clip (16 columns)

Written per shard as `metadata_<shard>.csv` (`:284-295`) and concatenated by
`--merge` (sorted by `(dataset, vid)`, `:245`). The 673-line file = 1 header +
672 rows.

| Column | Meaning | Notes |
|--------|---------|-------|
| `vid` | per-dataset contiguous clip index | resets to 0 for each dataset |
| `file` | relative path, e.g. `velocity/000123.mp4` | key into `kinematics.json` |
| `dataset` | `velocity` \| `acceleration` | selects the split |
| `direction_deg` | heading in `{0,45,…,315}` | |
| `speed` | m/s label (velocity only) | **blank for acceleration rows** (`:286`) |
| `accel` | m/s² label (acceleration only) | **blank for velocity rows** (`:287`) |
| `start_x`, `start_y` | start position in metres, `∈[-2,2]` | |
| `sphere_r` | `0.3` | scene constant echoed per row |
| `floor_m` | `8.0` | |
| `cam_z` | `10.0` | |
| `size` | `256` | |
| `frames` | `16` | |
| `fps` | `24` | |
| `samples` | CYCLES spp (`24`) | |
| `encode_ok` | `1` if ffmpeg succeeded else `0` | sanity check requires all `1` |

Because `speed`/`accel` are written blank for the non-applicable dataset, a
velocity row has `speed=7.0, accel=` and an acceleration row has `speed=,
accel=2.0`. Example rows (real file):

```
vid,file,dataset,direction_deg,speed,accel,start_x,start_y,sphere_r,floor_m,cam_z,size,frames,fps,samples,encode_ok
0,acceleration/000000.mp4,acceleration,0.0,,2.0,-1.1953,-0.8134,0.3,8.0,10.0,256,16,24,24,1
389,velocity/000389.mp4,velocity,315.0,7.0,,1.7378,-0.7216,0.3,8.0,10.0,256,16,24,24,1
```

### `kinematics.json` — per-frame trajectory

`kinematics()` (`make_physics_blender.py:142-173`) writes both **physical** and
**projected-pixel** ground truth. The file is a dict keyed by the same relative
`file` path → a list of 16 per-frame records. Pixel projection uses Blender's own
camera model, so it is consistent with the actual rendered frames:

```python
from bpy_extras.object_utils import world_to_camera_view
co = world_to_camera_view(scn, cam, Vector((x, y, args.sphere_r)))  # normalized (u,v,depth)
px.append((co.x * args.size, (1 - co.y) * args.size))               # image px, y flipped
```

Per-frame record keys (13):

| Key | Meaning | Rounding |
|-----|---------|----------|
| `frame` | frame index `0..15` | — |
| `x_m`, `y_m` | world position (m) | 4 dp |
| `x_px`, `y_px` | image position (px, **y flipped**) | 3 dp |
| `vx_mps`, `vy_mps` | world velocity components (m/s) | 4 dp |
| `vx_px`, `vy_px` | finite-diff pixel motion (px/frame) | 4 dp |
| `speed_mps` | scalar speed (`a·t` for accel) | 4 dp |
| `accel_mps2` | scalar accel (0 for velocity) | 4 dp |
| `direction_deg` | heading | 3 dp |
| `speed_px` | `hypot(vx_px, vy_px)` (px/frame) | 4 dp |

Because projection is done by Blender, **no manual px/m conversion is needed** and
the pixel GT matches the rendered pixels.

## Key code

### Single-render animation loop + persistent-data speedup (`:263-291`, `:118-120`)

The sphere `location` is **keyframed** along the trajectory, then the whole clip
renders in a **single** `bpy.ops.render.render(animation=True)` call so Cycles
loops the frames in C:

```python
sph.animation_data_clear()
for i, (x, y) in enumerate(wxy):
    sph.location = (x, y, args.sphere_r)
    sph.keyframe_insert(data_path="location", frame=i)
scn.frame_start = 0; scn.frame_end = len(wxy) - 1
...
bpy.ops.render.render(animation=True)          # -> f0000.png .. f0015.png
```

Two decisive optimizations:

- **`scn.render.use_persistent_data = True`** (`:118-120`): reuse geometry/BVH
  across frames so only the moving sphere re-syncs each frame. Without this,
  every render rebuilds the whole scene on the GPU — this is *the* big speedup.
- **PNG then ffmpeg CLI** (`:93`, `encode()` `:176-181`): this `bpy` build has no
  FFMPEG muxer, so frames are written as fast low-zlib PNGs and encoded to
  H.264 (`libx264`, `yuv420p`, `crf 18`) via the `ffmpeg` binary on `PATH`.

### GPU selection (`:98-114`)

`--device` (default `CPU`) chooses the Cycles backend:

```python
if args.device.upper() in ("CUDA", "OPTIX", "HIP"):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = args.device.upper()
    used = [d.name for d in prefs.devices if d.type == args.device.upper() and (setattr(d,"use",True) or True)]
    for d in prefs.devices:                    # disable CPU when a GPU backend is chosen
        if d.type == "CPU": d.use = False
    scn.cycles.device = "GPU"
    if not used: scn.cycles.device = "CPU"     # graceful fallback, prints NO DEVICE FOUND
```

**OPTIX on `vll4` fails** (driver ABI error 7804) and silently falls back to CPU,
so the driver script defaults to **`DEVICE=CUDA`** (`run_blender_toy.sh:25`). Note
that `bpy` prints `"OptiX initialization failed"` even in CUDA mode (it probes all
backends), so `run_blender_toy.sh` only treats it as a real fallback if the
*selected* backend failed at runtime (`run_blender_toy.sh:93-105`).

### Multi-process sharding & GPU oversubscription (`run_blender_toy.sh`)

`bpy` is one Blender per process, so parallelism is across processes. The driver
resolves GPUs (SLURM `CUDA_VISIBLE_DEVICES` first, else `nvidia-smi -L`), then
**oversubscribes** each GPU because the tiny scene makes render *overhead-bound*
(scene-sync + PNG I/O), not GPU-bound (GPU was only ~2× CPU):

```bash
# pack several Blender procs onto each GPU to overlap that overhead
if [ "$DEVICE" = "CPU" ]; then SHARDS_PER_GPU=${SHARDS_PER_GPU:-8}
else                           SHARDS_PER_GPU=${SHARDS_PER_GPU:-4}; fi   # 4 GPU x 4 = 16 procs
```

Processes are launched round-robin over GPUs via per-process
`CUDA_VISIBLE_DEVICES` (`run_blender_toy.sh:55-84`), with `--threads = NCPU/TOTAL`
so parallel CPU threads don't oversubscribe cores (`:58`,
`make_physics_blender.py:115-116`). After all shards finish, the script runs the
generator's `--merge` mode (`make_physics_blender.py:239-250`) to concatenate
`metadata_*.csv` / `kinematics_*.json` → `metadata.csv` / `kinematics.json`,
removes the `_frames_*` PNG scratch dirs, then runs the sanity check.

### Sanity check (`sanity_check_blender.py`)

```
python sanity_check_blender.py <dataset_dir> <exp_velocity> <exp_accel> [frames]
```

Run post-merge; exits non-zero on any failure:

- **Counts** (`:29-49`): metadata rows / mp4 on disk / kinematics clips all
  `== exp_vel+exp_acc`; split correct.
- **Files** (`:36-43`): every row's `encode_ok == 1`; every `.mp4` present and
  non-empty.
- **Grid** (`:51-58`): directions exactly `{0,45,…,315}`; every start within
  `[-2,2]`.
- **Per-clip physics** (all 672 clips, `:65-88`):
  - `velocity`: speed constant, `speed[0] == label`, accel `≈ 0`.
  - `acceleration`: accel constant `== label`, starts at rest (`v0≈0`), final
    speed matches `a·(F−1)/fps`.

## Configuration

**This subsystem is CLI/env-driven, not YAML-driven.** The generator is
configured entirely through `argparse` (`make_physics_blender.py:212-235`); the
`run_blender_toy.sh` driver exposes a subset as env vars. The YAML files under
`configs/analysis/blender_toy_dataset/*.yaml` are **downstream probing configs**
that *consume* this dataset (see [13](13-configs-reference.md)) — they do **not**
configure the generator.

Paper-exact run (672 videos), overriding only common knobs:

```bash
DEVICE=CUDA  bash data_gen/run_blender_toy.sh          # OPTIX fails on vll4 -> CUDA
SAMPLES=32   bash data_gen/run_blender_toy.sh          # more CYCLES spp (slower)
NGPU=2       bash data_gen/run_blender_toy.sh          # cap GPU count
```

**Env-var knobs** (`run_blender_toy.sh:25-33, 45, 52-54`):

| Env var | Default | Meaning |
|---------|---------|---------|
| `DEVICE` | `CUDA` | Cycles backend (`CUDA`\|`OPTIX`\|`CPU`) |
| `SAMPLES` | `24` | CYCLES samples/pixel |
| `SEED` | `0` | RNG for start positions |
| `NGPU` | (all) | cap number of GPUs used |
| `SHARDS_PER_GPU` | `4` GPU / `8` CPU | processes packed per GPU (oversubscription) |

The paper-exact grid (`N_DIRECTIONS=8`, `SPEEDS`, `ACCELS`, `N_STARTS=7`) is
**hardcoded** in the driver — do not change it if you want paper-exact output.

**Generator flags** (`make_physics_blender.py:212-235`; defaults are all paper
values):

| Flag | Default | Allowed / meaning |
|------|---------|-------------------|
| `--save_dir` | *(required)* | output directory |
| `--dataset` | `both` | `velocity`\|`acceleration`\|`both`; gates which grids are built (`:191-194`) |
| `--size` / `--frames` / `--fps` | `256` / `16` / `24` | paper resolution / clip length / fps |
| `--n_directions` / `--n_starts` | `8` / `7` | grid dims |
| `--speeds` / `--accels` | `1,2,3,4,5,6,7` / `2,4,6,8,10` | magnitude grids (m/s, m/s²) |
| `--start_m` / `--floor_m` / `--cam_z` / `--sphere_r` | `2.0` / `8.0` / `10.0` / `0.3` | scene metres |
| `--samples` | `24` | CYCLES spp |
| `--device` / `--threads` | `CPU` / `0` | backend / Cycles threads (0=all) |
| `--shard` / `--nshards` / `--merge` | `0` / `1` / off | sharding + merge mode |
| `--smoke` | off | 6-clip eyeball mode; keeps only directions `{0,90,180,270}`, first 6 specs (`:233,253-254`) |
| `--seed` | `0` | RNG for start positions |

### Why it matches the paper exactly

- **Start positions**: `~ U([-2,2]²)` m (`build_specs :201-202`, `--start_m 2.0`).
- **Speeds / accels**: `{1..7}` m/s and `{2,4,6,8,10}` m/s² — the paper's grids.
- **Directions**: 8 evenly spaced headings `{0,45,…,315}` (`:189`).
- **FOV**: overhead camera at `(0,0,10)` with Blender's default 50 mm / 36 mm
  sensor = Kubric default = the paper's rendering setup.
- **Motion**: analytic frictionless kinematics reproduce PyBullet's `dv=0`
  (velocity) and constant-`a`-from-rest (acceleration).
- **Fixed appearance**: a single sphere material for all 672 clips — the paper's
  control that keeps early motion-energy correlated with speed.

## Invariants & gotchas

- **The fixed sphere is load-bearing.** The single fixed red material is
  deliberate. The predecessor `make_physics_toy.py` randomizes appearance, which
  **decorrelates early-layer motion-energy from speed** and breaks the paper's
  layer-0 speed result. Do not re-introduce per-clip appearance randomness here.
  *(This control is exactly what let the downstream `attention_distance` mode
  reproduce Fig. 3 — see Purpose / Cross-references.)*
- **Filenames are unique only within a subdirectory.** `vid` is a per-dataset
  counter, so **both** `velocity/` and `acceleration/` restart at `000000.mp4`
  (verified on disk: both `velocity/000000.mp4` and `acceleration/000000.mp4`
  exist). A basename is **not** globally unique — always disambiguate with the
  `dataset` / `file` column (`velocity/000000.mp4` vs `acceleration/000000.mp4`).
- **FOV depends on Blender factory defaults.** Lens/sensor are never set (`grep`
  confirms no `lens`/`sensor_width` in the file). If a future Blender changes
  default lens (50 mm) or sensor width (36 mm), the pixel projection and FOV drift
  silently away from paper/Kubric. Treat 50 mm / 36 mm as an invariant.
- **`--device` default is `CPU`.** Running the Python directly renders on **CPU**
  unless `--device CUDA` is passed — GPU is *default-off*. The
  `run_blender_toy.sh` driver flips the default to `CUDA`.
- **Silent GPU→CPU fallback.** If no GPU device enumerates, the generator prints
  `NO DEVICE FOUND` and runs on CPU (`make_physics_blender.py:110-112`); a "GPU
  run" can secretly be CPU. `run_blender_toy.sh:93-105` greps the logs to catch
  this (and the OPTIX err-7804 case) and warn.
- **ffmpeg must be on PATH.** This `bpy` build has no FFMPEG muxer; encoding is
  delegated to the `ffmpeg` binary. Missing ffmpeg → `encode_ok=0` for every row,
  which the sanity check will fail on.
- **Seed + grid must be identical across shards.** Start positions come from a
  seeded RNG over the *full* grid, then sliced by shard index. All shards must use
  the same `--seed` and grid args or the merged dataset is inconsistent;
  `run_blender_toy.sh` passes one `SEED` and grid to every shard.
- **`speed`/`accel` columns are conditionally blank.** A velocity row leaves
  `accel` empty and an acceleration row leaves `speed` empty
  (`make_physics_blender.py:286-287`); downstream readers must treat the blank as
  "not applicable to this split", not zero.
- **Pixel y is flipped**: `y_px = (1 − co.y) · size` (image origin top-left).
- **Sanity check is mandatory in the pipeline** and exits non-zero on any
  mismatch, so `run_blender_toy.sh` reports `DONE with SANITY FAILURES` and
  propagates the failure code (`run_blender_toy.sh:114-120`).

## Cross-references

- [06 — Data-pipeline changes](06-data-pipeline-changes.md) — how the resulting
  mp4s are sampled/resized/frame-stepped by the loader.
- [10 — CSV / targets.npy builders](10-datasets-csv-targets.md) — the builders
  that turn `metadata.csv` + `kinematics.json` into the probing CSVs and
  `blender_targets.npy`.
- [11 — Attention hooks](11-attention-hooks.md) — the additive SDPA patch /
  `AttentionDistanceCollector` that this dataset feeds.
- [12 — Analysis modes subpackage & reproduction roadmap](12-analysis-modes.md) —
  the `attention_distance` mode (Phase 1, done) that consumed the `velocity` set.
- [13 — Config reference](13-configs-reference.md) — `vjepa_attn_distance.yaml`
  and the other downstream probing configs.
- [14 — Reproduction status & findings](14-reproduction-status-and-findings.md) —
  the Fig. 3 heatmap reproduction (dark local heads in the ~5–13 PEZ) that this
  fixed-sphere dataset made possible.
