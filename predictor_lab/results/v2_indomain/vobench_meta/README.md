# VOBench_v1

Violation-of-expectation benchmark rendered in Unreal Engine 5.4 (TestBed, `tb/export.py`).

720 blocks -> 2880 arm videos (1440 possible, 1440 impossible). Each block renders two variants
A and B with the same map, camera, path and occluder, spliced at frame 48:

| arm | frames 0-47 | frames 48-95 | label |
|---|---|---|---|
| pos_a | A | A | possible |
| pos_b | B | B | possible |
| imp_ab | A | B | impossible |
| imp_ba | B | A | impossible |

Video: 96 frames, 30 fps, 512 px PNG. Model input: stride 3 -> 32 samples (context 16 + predict 16).

## Factors

- objset: primitive (sphere, cube, cone, cylinder, torus, capsule); realistic (basketball, tennis_ball, apple, car, scooter, bus)
- violation: permanence (B = empty scene), shape (B = another object of the set), color (B = a colour >= 60 deg of hue from A's)
- motion: flat_static, flat_vel, flat_acc, flat_dec, ramp_down, ramp_up
- background: v11 (floor + wall), assets (buildings, bushes, rocks, sky)
- occlusion: visible, early / mid / late x k = 1-3: a panel stands in front of the object for exactly 2k model-input samples (it turns between samples). Only `late` hides the A|B switch.

Blocks per objset x violation: primitive/color 120, primitive/permanence 120, primitive/shape 120, realistic/color 120, realistic/permanence 120, realistic/shape 120; per occlusion: early 216, late 216, mid 216, visible 72.

Objects and colours are balanced over the factors (each object equally often per violation; every shape pair (A, B) equally often); panel size and path depend only on (objset, background, motion, occlusion, k). The four arms of a block share every factor, so no factor predicts possible / impossible. Split train / test by block, never by arm.

## Layout

```
<objset>/<violation>/<occlusion>/<background>_<motion>_<A>[_to_<B>]/
    A/ B/ pos_a/ pos_b/ imp_ab/ imp_ba/   000000.png ... (arms are hard links into A/ B/)
    label.json                            design, geometry, per-frame centre / u,v / radius / hidden share
metadata.csv   one row per arm: file_name = the arm folder
blocks.csv     one row per block
review/        GIFs (2x2 arms) and model-input sheets, INDEX.md
```

metadata.csv `samples`: per model-input sample `.` visible, `p` partly hidden, `H` hidden, `o` partly off-frame, `|` = splice. `edge_dE`: object-edge contrast (Lab dE76).

## gravity/ (added 2026-10-10)

Gravity violations in the same frame system (96 frames, 30 fps, 512 px, stride 3, splice 48), built by
the same harness (TestBed spec `specs/VO_gravity.json`). 216 blocks -> 936 arm videos. Self-contained:
its own `README.md`, `metadata.csv`, `blocks.csv` and `review/`, with `file_name` relative to `gravity/`.
The top-level `metadata.csv` / `blocks.csv` above cover the permanence / shape / color blocks only.

A gravity block is ONE past and several futures (not the A/B four arms): every arm shares frames 0-47
with `pos` and differs only after the splice. g = 375 cm/s^2 for both laws (V11_gravity and
gravity_realistic moved to this frame system).

- `ledge` (144 blocks, V11_gravity's cells: platform 300 / 400 cm x speed 200 / 260 / 320 cm/s):
  `pos` falls off the platform, `imp_float` keeps going at its height.
- `arc` (72 blocks, gravity_realistic's cells: apex on frame 66 / 48 / 30 = rising / at the apex /
  falling at the splice; 9 arms): `pos`, `imp_rise`, `imp_float`, `imp_line`, `imp_stop`, and
  `imp_stop2` / `4` / `8` / `12` (the true flight for K frames after the splice, then frozen in mid-air).

See `gravity/README.md` for the laws and cells.
