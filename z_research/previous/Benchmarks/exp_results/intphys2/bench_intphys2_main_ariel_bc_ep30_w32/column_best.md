# IntPhys 2 열별 최고 선택 — `bench_intphys2_main_ariel_bc_ep30_w32`

격자 6 개: C8/avg, C12/avg, C16/avg, C20/avg, C24/avg, C28/avg

| 열 | 짝 | 열별 최고 (설정) | overall 최고 설정 (C8/avg) 로 읽은 값 | 논문 V-JEPA 2-h |
|---|---:|---:|---:|---:|
| Easy | 52 | 61.54 (C8/avg) | 61.54 | 54.00 |
| Medium | 200 | 53.50 (C20/avg) | 53.00 | 58.50 |
| Hard | 168 | 53.57 (C8/avg) | 53.57 | 59.38 |
| Unknown | 86 | 51.16 (C8/avg) | 51.16 |  |
| Overall | 506 | 53.75 (C8/avg) | 53.75 | 57.51 |
| permanence|Fixed | 52 | 71.15 (C8/avg) | 71.15 | 59.62 |
| permanence|Moving | 68 | 47.06 (C28/avg) | 41.18 | 57.35 |
| immutability|Fixed | 52 | 55.77 (C20/avg) | 48.08 | 55.77 |
| immutability|Moving | 68 | 48.53 (C8/avg) | 48.53 | 58.72 |
| continuity|Fixed | 52 | 65.38 (C28/avg) | 61.54 | 57.69 |
| continuity|Moving | 68 | 63.24 (C8/avg) | 63.24 | 75.00 |
| solidity|Fixed | 52 | 48.08 (C8/avg) | 48.08 | 46.15 |
| solidity|Moving | 94 | 52.13 (C8/avg) | 52.13 | 58.51 |

## C 별 Overall

| 설정 | Overall | 동점 |
|---|---:|---:|
| C8/avg | 53.75 | 0 |
| C12/avg | 50.20 | 0 |
| C16/avg | 49.41 | 0 |
| C20/avg | 51.19 | 0 |
| C24/avg | 49.80 | 0 |
| C28/avg | 50.99 | 0 |

⚠️ 열별 최고는 같은 평가셋에서 격자를 고른 값 — descriptive (CLAUDE.md §1-3).

## 재현

```
python z_research/scripts/analysis/intphys2_column_best.py --run z_research/Benchmarks/exp_results/intphys2/bench_intphys2_main_ariel_bc_ep30_w32
```