# IntPhys 2 열별 최고 선택 — `bench_intphys2_main_ariel_bc_ep30_w48`

격자 6 개: C12/avg, C18/avg, C24/avg, C30/avg, C36/avg, C42/avg

| 열 | 짝 | 열별 최고 (설정) | overall 최고 설정 (C42/avg) 로 읽은 값 | 논문 V-JEPA 2-h |
|---|---:|---:|---:|---:|
| Easy | 52 | 61.54 (C42/avg) | 61.54 | 54.00 |
| Medium | 200 | 53.00 (C42/avg) | 53.00 | 58.50 |
| Hard | 168 | 51.79 (C24/avg) | 49.40 | 59.38 |
| Unknown | 86 | 56.98 (C30/avg) | 54.65 |  |
| Overall | 506 | 52.96 (C42/avg) | 52.96 | 57.51 |
| permanence|Fixed | 52 | 57.69 (C30/avg) | 57.69 | 59.62 |
| permanence|Moving | 68 | 47.06 (C42/avg) | 47.06 | 57.35 |
| immutability|Fixed | 52 | 65.38 (C36/avg) | 59.62 | 55.77 |
| immutability|Moving | 68 | 42.65 (C18/avg) | 33.82 | 58.72 |
| continuity|Fixed | 52 | 53.85 (C36/avg) | 51.92 | 57.69 |
| continuity|Moving | 68 | 73.53 (C30/avg) | 70.59 | 75.00 |
| solidity|Fixed | 52 | 48.08 (C12/avg) | 46.15 | 46.15 |
| solidity|Moving | 94 | 56.38 (C42/avg) | 56.38 | 58.51 |

## C 별 Overall

| 설정 | Overall | 동점 |
|---|---:|---:|
| C12/avg | 49.41 | 0 |
| C18/avg | 50.20 | 0 |
| C24/avg | 51.78 | 0 |
| C30/avg | 52.37 | 0 |
| C36/avg | 52.77 | 0 |
| C42/avg | 52.96 | 0 |

⚠️ 열별 최고는 같은 평가셋에서 격자를 고른 값 — descriptive (CLAUDE.md §1-3).

## 재현

```
python z_research/scripts/analysis/intphys2_column_best.py --run z_research/Benchmarks/exp_results/intphys2/bench_intphys2_main_ariel_bc_ep30_w48
```