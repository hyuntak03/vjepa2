# Blender toy-physics dataset generator

## Purpose

Generate a **paper-faithful** synthetic ball-motion dataset for probing motion
variables (speed, acceleration, direction) in V-JEPA2 and VLM encoders. It
replicates the controlled toy datasets of *"Interpreting Physics in Video World
Models"* (App. A.1.2) using **Blender / `bpy` + CYCLES** so the frames are
photorealistic (in-distribution for a video encoder) rather than flat 2D PIL
sprites. Ball motion is computed **analytically** (frictionless, so no physics
engine is needed) and Blender projects metres→pixels itself, so ground truth is
exact in both physical and pixel units.

Output plugs directly into the layer-wise probing CSV builders (same on-disk
layout as the older `make_physics_toy.py`):

```
<save_dir>/velocity/000000.mp4 …        <save_dir>/acceleration/000000.mp4 …
<save_dir>/metadata.csv                 <save_dir>/kinematics.json
```

## What changed vs upstream V-JEPA2

Upstream commit `204698b` has **no `data_gen/` directory at all** — the entire
data generator is new to this fork. This section documents three new files:

| File | Status | Role |
|------|--------|------|
| `data_gen/make_physics_blender.py` | **new** | The bpy/CYCLES generator (this doc). |
| `data_gen/run_blender_toy.sh` | **new** | Multi-GPU / multi-process driver + merge + sanity check. |
| `data_gen/sanity_check_blender.py` | **new** | Post-merge dataset validator. |

Relationship to the earlier `data_gen/make_physics_toy.py` (also fork-only): that
predecessor is a lightweight **2D PIL** generator with an explicit *anti-shortcut*
design — it randomizes shape / colour / size / orientation **independently** of
the motion label (`make_physics_toy.py:20-46`). That randomization is exactly why
the Blender generator was written: randomizing appearance **decorrelated
early-layer motion-energy from speed**, so the paper's layer-0 speed result did
not reproduce. `make_physics_blender.py` instead uses **one fixed sphere
appearance** (a single red glossy material, same for every clip) to restore the
paper's control (`make_physics_blender.py:64-73`, header note lines 18-20).

## Scene

Built once per process in `build_scene()` (`make_physics_blender.py:42-121`),
from factory-empty settings (`read_factory_settings(use_empty=True)`).

| Element | Setting | Code |
|---------|---------|------|
| **Floor** | 8×8 m plane at origin, grey Principled BSDF (roughness 0.9) + procedural `ShaderNodeTexNoise` (scale 6) → colour-ramp, giving static reference features | `:48-62` |
| **Sphere** | UV sphere `r=0.3 m`, smooth-shaded, **fixed** red-glossy material `(0.85, 0.18, 0.15)`, roughness 0.35 | `:65-73` |
| **Camera** | perspective, at `(0,0,10)` looking straight down (`rotation_euler=(0,0,0)`) | `:76-80` |
| **Light** | single `SUN`, energy 3.5, tilted `(25°,15°)` → shading + shadow depth cue | `:83-85` |
| **World** | soft ambient background `(0.05,0.06,0.08)` so floor isn't pitch black | `:88-89` |
| **Render** | `CYCLES`, `samples` (default 24 spp), 256×256, PNG out (RGB, zlib compression 15) | `:91-97` |

**Camera FOV = Kubric default.** The code **never sets `cam_data.lens` or
`sensor_width`**, so the camera keeps Blender's factory defaults — **50 mm lens /
36 mm sensor** — which is exactly the Kubric default and the paper's FOV. This is
an *implicit* invariant: the paper-exact FOV depends on Blender defaults not
changing across versions. (See Gotchas.)

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

The grid is built in `build_specs()` (`:184-209`), directions =
`round(360·k/n_directions, 4)` (→ `0,45,…,315` for `n=8`), start positions drawn
from a **seeded** RNG `~ U([-2,2]²)` m:

| Dataset | directions × magnitudes × starts | count | magnitudes |
|---------|----------------------------------|-------|-----------|
| `velocity` | 8 × 7 × 7 | **392** | speeds `{1,2,3,4,5,6,7}` m/s |
| `acceleration` | 8 × 5 × 7 | **280** | accels `{2,4,6,8,10}` m/s² |
| **total** | | **672** | 16 frames @ 24 fps (0.67 s), 256² |

`vid` is assigned **before** sharding (a global contiguous counter per dataset,
`:207`), so clip filenames are stable regardless of shard count.

## Ground truth (`kinematics.json`)

`kinematics()` (`make_physics_blender.py:142-173`) writes both **physical** and
**projected-pixel** ground truth per frame. Pixel projection uses Blender's own
camera model:

```python
from bpy_extras.object_utils import world_to_camera_view
co = world_to_camera_view(scn, cam, Vector((x, y, args.sphere_r)))  # normalized (u,v,depth)
px.append((co.x * args.size, (1 - co.y) * args.size))               # image px, y flipped
```

Per-frame record keys: `x_m,y_m` (world), `x_px,y_px` (image, y-flipped),
`vx_mps,vy_mps,speed_mps,accel_mps2,direction_deg` (physical), and finite-diff
pixel motion `vx_px,vy_px,speed_px`. Because projection is done by Blender, **no
manual px/m conversion is needed** and the pixel GT is consistent with the actual
rendered frames.

## Rendering & speed optimizations

Per clip (`:263-291`): the sphere `location` is **keyframed** along the
trajectory, then the whole clip renders in a **single** `bpy.ops.render.render(
animation=True)` call so Cycles loops the frames in C:

```python
sph.animation_data_clear()
for i, (x, y) in enumerate(wxy):
    sph.location = (x, y, args.sphere_r)
    sph.keyframe_insert(data_path="location", frame=i)
scn.frame_start = 0; scn.frame_end = len(wxy) - 1
...
bpy.ops.render.render(animation=True)          # -> f0000.png .. f0015.png
```

Two key optimizations:

- **`scn.render.use_persistent_data = True`** (`:118-120`): reuse geometry/BVH
  across frames so only the moving sphere re-syncs each frame. Without this,
  every render rebuilds the whole scene on the GPU — this is *the* big speedup.
- **PNG then ffmpeg CLI** (`:93`, `encode()` `:176-181`): this `bpy` build has no
  FFMPEG muxer, so frames are written as fast low-zlib PNGs and encoded to
  H.264 (`libx264`, `yuv420p`, `crf 18`) via the `ffmpeg` binary on `PATH`.

## GPU selection

`--device` (default `CPU`) chooses the Cycles backend (`:98-114`):

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
so the driver script defaults to **`DEVICE=CUDA`** (`run_blender_toy.sh:25`).
Note that `bpy` prints `"OptiX initialization failed"` even in CUDA mode (it
probes all backends), so `run_blender_toy.sh` only treats it as a real fallback
if the *selected* backend failed (`run_blender_toy.sh:93-105`).

## Multi-process sharding (`run_blender_toy.sh`)

`bpy` is one Blender per process, so parallelism is **across processes**, sharded
by `i % nshards == shard` (`make_physics_blender.py:255`). The driver
(`run_blender_toy.sh`) resolves GPUs (SLURM `CUDA_VISIBLE_DEVICES` first, else
`nvidia-smi -L`), then **oversubscribes** each GPU:

```bash
# render is OVERHEAD-bound (scene-sync + PNG I/O), not GPU-bound (GPU only ~2x CPU).
# => pack several Blender procs onto each GPU to overlap that overhead.
SHARDS_PER_GPU=${SHARDS_PER_GPU:-4}            # 4 GPU x 4 = 16 procs; CPU mode defaults to 8
```

Processes are launched round-robin over GPUs via per-process
`CUDA_VISIBLE_DEVICES` (`:78-84`), with `--threads = NCPU/TOTAL` so parallel CPU
threads don't oversubscribe cores (`:58`, `make_physics_blender.py:115-116`).
After all shards finish, the script runs the generator's `--merge` mode
(`make_physics_blender.py:239-250`) to concatenate `metadata_*.csv` /
`kinematics_*.json` → `metadata.csv` / `kinematics.json`, removes the
`_frames_*` PNG scratch dirs, then runs the sanity check.

## Sanity check (`sanity_check_blender.py`)

```
python sanity_check_blender.py <dataset_dir> <exp_velocity> <exp_accel> [frames]
```

Run post-merge. Exits non-zero on any failure (`sanity_check_blender.py`):

- **Counts**: metadata rows / mp4 on disk / kinematics clips all `== exp_vel+exp_acc`, split correct (`:29-49`).
- **Files**: every row's `encode_ok == 1`, every `.mp4` present and non-empty (`:36-43`).
- **Grid**: directions exactly `{0,45,…,315}`; every start within `[-2,2]` (`:51-58`).
- **Per-clip physics** (all 672 clips, `:65-88`):
  - `velocity`: speed constant, `speed[0] == label`, accel `≈ 0`.
  - `acceleration`: accel constant `== label`, starts at rest (`v0≈0`), final speed matches `a·(F−1)/fps`.

## Config

**This subsystem is CLI/env-driven, not YAML-driven.** The generator is
configured entirely through `argparse` (`make_physics_blender.py:212-235`); the
`run_blender_toy.sh` driver exposes a subset as env vars. (The YAML files under
`configs/analysis/blender_toy_dataset/*.yaml` are **downstream probing configs**
that *consume* this dataset — they do not configure the generator.)

Paper-exact run (672 videos), overriding only the device:

```bash
DEVICE=CUDA  bash data_gen/run_blender_toy.sh          # OPTIX fails on vll4 -> CUDA
SAMPLES=32   bash data_gen/run_blender_toy.sh          # more CYCLES spp (slower)
NGPU=2       bash data_gen/run_blender_toy.sh          # cap GPU count
```

Env-var knobs (`run_blender_toy.sh:25-33, 52-54`): `DEVICE` (default `CUDA`),
`SAMPLES` (24), `SEED` (0), `NGPU`, `SHARDS_PER_GPU` (4 GPU / 8 CPU). The
paper-exact grid (`N_DIRECTIONS=8`, `SPEEDS`, `ACCELS`, `N_STARTS=7`) is hardcoded
in the driver — **do not change it if you want paper-exact output**.

Key generator flags (defaults are all paper values):

| Flag | Default | Meaning |
|------|---------|---------|
| `--size` / `--frames` / `--fps` | `256` / `16` / `24` | paper resolution / clip length |
| `--speeds` / `--accels` | `1,2,3,4,5,6,7` / `2,4,6,8,10` | magnitude grids |
| `--n_directions` / `--n_starts` | `8` / `7` | grid dims |
| `--start_m` / `--floor_m` / `--cam_z` / `--sphere_r` | `2.0` / `8.0` / `10.0` / `0.3` | scene metres |
| `--samples` | `24` | CYCLES spp |
| `--device` / `--threads` | `CPU` / `0` | backend / Cycles threads (0=all) |
| `--shard` / `--nshards` / `--merge` | `0` / `1` / off | sharding + merge mode |
| `--seed` | `0` | RNG for start positions |

## Why it matches the paper exactly

- **Start positions**: `~ U([-2,2]²)` m, matching the paper's start distribution (`build_specs` `:201-202`, `--start_m 2.0`).
- **Speeds / accels**: `{1..7}` m/s and `{2,4,6,8,10}` m/s² — the paper's exact magnitude grids.
- **Directions**: 8 evenly spaced headings `{0,45,…,315}` (`:189`).
- **FOV**: overhead camera at `(0,0,10)` with Blender's default 50 mm / 36 mm sensor = Kubric default = the paper's rendering setup.
- **Motion**: analytic frictionless kinematics reproduce PyBullet's `dv=0` (velocity) and constant-`a`-from-rest (acceleration).
- **Fixed appearance**: a single sphere material for all 672 clips — the paper's control that keeps early motion-energy correlated with speed.

## Gotchas / invariants / default-off guarantees

- **Fixed sphere is load-bearing.** The single fixed red material is deliberate.
  The predecessor `make_physics_toy.py` randomizes appearance, which
  **decorrelates early-layer motion-energy from speed** and breaks the paper's
  layer-0 speed result. Do not re-introduce per-clip appearance randomness here.
- **FOV depends on Blender factory defaults.** Lens/sensor are never set
  (`grep` confirms no `lens`/`sensor_width` in the file). If a future Blender
  changes default lens (50 mm) or sensor width (36 mm), the pixel projection and
  FOV drift silently away from paper/Kubric. Treat 50 mm / 36 mm as an invariant.
- **`--device` default is `CPU`.** Running the Python directly renders on **CPU**
  unless `--device CUDA` is passed — GPU is *default-off*. The `run_blender_toy.sh`
  driver flips the default to `CUDA`.
- **Silent GPU→CPU fallback.** If no GPU device enumerates, the generator prints
  `NO DEVICE FOUND` and runs on CPU (`make_physics_blender.py:110-112`); a "GPU
  run" can secretly be CPU. `run_blender_toy.sh:93-105` greps the logs to catch
  this (and the OPTIX err-7804 case) and warn.
- **ffmpeg must be on PATH.** This `bpy` build has no FFMPEG muxer; encoding is
  delegated to the `ffmpeg` binary. Missing ffmpeg → `encode_ok=0` for every row.
- **Seed + grid must be identical across shards.** Start positions come from a
  seeded RNG over the *full* grid, then sliced by shard index. All shards must
  use the same `--seed` and grid args or the merged dataset is inconsistent;
  `run_blender_toy.sh` passes one `SEED` and grid to every shard.
- **Pixel y is flipped**: `y_px = (1 − co.y) · size` (image origin top-left).
- **Sanity check is mandatory in the pipeline** and exits non-zero on any
  mismatch, so `run_blender_toy.sh` reports `DONE with SANITY FAILURES` and
  propagates the failure code (`run_blender_toy.sh:114-120`).
