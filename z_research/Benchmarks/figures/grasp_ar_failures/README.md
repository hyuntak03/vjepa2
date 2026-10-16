# GRASP — AR 이 못하는 시나리오의 실패 쌍 (2026-10-16)

채점: 공식 규칙 (A.8 격자 최고 combo · 시나리오마다 최고 C · 창 평균 surprise). combo — release `skip10_w32`, AR `skip10_w32`, AR+VOB `skip10_w32`.
정답 = 불가능 (IP) 영상의 surprise 가 가능 (P) 영상보다 크다. margin = (s_IP − s_P) / s_P × 100 (%) — **음수면 틀림**.
고른 쌍: AR 이 틀리고 release 가 맞힌 쌍 중 AR margin 이 가장 나쁜 순 (모자라면 AR 이 틀린 쌍).

GIF: 원본 mp4 (MPEG-4 Part 2, VSCode 재생 불가) 를 12 fps · 320×240 로 줄여 P | IP 를 나란히 붙였다. 원본은 `/workspace/world_model/grasp_ar_failures/`.

재현: `python z_research/scripts/analysis/grasp_failure_samples.py`

## Collision — 정확도 release 68.8 · AR 3.1 · AR+VOB 1.6 (AR 이 틀린 쌍 124 / 128)

| 순위 | 쌍 | GIF (왼쪽 P · 오른쪽 IP) | margin release | margin AR | margin AR+VOB |
|---|---|---|---:|---:|---:|
| 1 | Collision_071 | `Collision/1_071.gif` | +1.38 | -11.60 | -13.40 |
| 2 | Collision_045 | `Collision/2_045.gif` | +0.76 | -10.33 | -12.22 |
| 3 | Collision_119 | `Collision/3_119.gif` | +0.75 | -9.34 | -10.95 |

## ObjectPermanence3 — 정확도 release 98.4 · AR 50.0 · AR+VOB 55.5 (AR 이 틀린 쌍 64 / 128)

| 순위 | 쌍 | GIF (왼쪽 P · 오른쪽 IP) | margin release | margin AR | margin AR+VOB |
|---|---|---|---:|---:|---:|
| 1 | ObjectPermanence3_086 | `ObjectPermanence3/1_086.gif` | +0.59 | -7.25 | -7.32 |
| 2 | ObjectPermanence3_011 | `ObjectPermanence3/2_011.gif` | +0.55 | -6.38 | -6.15 |
| 3 | ObjectPermanence3_007 | `ObjectPermanence3/3_007.gif` | +0.33 | -6.38 | -6.51 |

## Unchangeableness2 — 정확도 release 91.4 · AR 67.2 · AR+VOB 71.1 (AR 이 틀린 쌍 42 / 128)

| 순위 | 쌍 | GIF (왼쪽 P · 오른쪽 IP) | margin release | margin AR | margin AR+VOB |
|---|---|---|---:|---:|---:|
| 1 | Unchangeableness2_017 | `Unchangeableness2/1_017.gif` | +0.46 | -3.29 | -3.05 |
| 2 | Unchangeableness2_075 | `Unchangeableness2/2_075.gif` | +0.85 | -3.21 | -3.01 |
| 3 | Unchangeableness2_032 | `Unchangeableness2/3_032.gif` | +0.15 | -2.20 | -2.95 |

## SolidityContinuity2 — 정확도 release 82.0 · AR 69.5 · AR+VOB 77.3 (AR 이 틀린 쌍 39 / 128)

| 순위 | 쌍 | GIF (왼쪽 P · 오른쪽 IP) | margin release | margin AR | margin AR+VOB |
|---|---|---|---:|---:|---:|
| 1 | SolidityContinuity2_117 | `SolidityContinuity2/1_117.gif` | +0.05 | -1.39 | +0.45 |
| 2 | SolidityContinuity2_125 | `SolidityContinuity2/2_125.gif` | +0.33 | -1.04 | -1.19 |
| 3 | SolidityContinuity2_116 | `SolidityContinuity2/3_116.gif` | +0.24 | -1.00 | -1.59 |

## GravityInertia2 — 정확도 release 53.1 · AR 44.5 · AR+VOB 50.0 (AR 이 틀린 쌍 71 / 128)

| 순위 | 쌍 | GIF (왼쪽 P · 오른쪽 IP) | margin release | margin AR | margin AR+VOB |
|---|---|---|---:|---:|---:|
| 1 | GravityInertia2_083 | `GravityInertia2/1_083.gif` | +0.08 | -15.09 | -12.26 |
| 2 | GravityInertia2_085 | `GravityInertia2/2_085.gif` | +1.85 | -11.62 | -13.77 |
| 3 | GravityInertia2_012 | `GravityInertia2/3_012.gif` | +1.44 | -11.02 | -7.71 |

## Inertia — 정확도 release 53.9 · AR 47.7 · AR+VOB 49.2 (AR 이 틀린 쌍 67 / 128)

| 순위 | 쌍 | GIF (왼쪽 P · 오른쪽 IP) | margin release | margin AR | margin AR+VOB |
|---|---|---|---:|---:|---:|
| 1 | Inertia_023 | `Inertia/1_023.gif` | +0.73 | -8.29 | -5.59 |
| 2 | Inertia_012 | `Inertia/2_012.gif` | +0.30 | -5.18 | -6.16 |
| 3 | Inertia_123 | `Inertia/3_123.gif` | +0.07 | -4.12 | -4.44 |

## GravityInertia — 정확도 release 22.7 · AR 35.2 · AR+VOB 32.0 (AR 이 틀린 쌍 83 / 128)

| 순위 | 쌍 | GIF (왼쪽 P · 오른쪽 IP) | margin release | margin AR | margin AR+VOB |
|---|---|---|---:|---:|---:|
| 1 | GravityInertia_079 | `GravityInertia/1_079.gif` | +0.63 | -2.15 | -2.83 |
| 2 | GravityInertia_092 | `GravityInertia/2_092.gif` | +0.15 | -0.95 | -1.23 |
| 3 | GravityInertia_007 | `GravityInertia/3_007.gif` | +0.05 | -0.41 | -0.39 |

## GravityContinuity — 정확도 release 32.8 · AR 48.4 · AR+VOB 54.7 (AR 이 틀린 쌍 66 / 128)

| 순위 | 쌍 | GIF (왼쪽 P · 오른쪽 IP) | margin release | margin AR | margin AR+VOB |
|---|---|---|---:|---:|---:|
| 1 | GravityContinuity_058 | `GravityContinuity/1_058.gif` | +0.00 | -0.50 | -0.84 |
| 2 | GravityContinuity_017 | `GravityContinuity/2_017.gif` | +0.08 | -0.38 | -0.12 |
| 3 | GravityContinuity_043 | `GravityContinuity/3_043.gif` | +0.04 | -0.36 | -0.31 |
