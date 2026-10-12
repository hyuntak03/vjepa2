# IntPhys 2 열별 최고 선택 — `bench_intphys2_main_vith`

격자 11 개: C4/avg, C6/avg, C8/avg, C10/avg, C12/avg, C14/avg, C18/avg, C24/avg, C30/avg, C36/avg, C42/avg

| 열 | 짝 | 열별 최고 (설정) | overall 최고 설정 (C24/avg) 로 읽은 값 | 논문 V-JEPA 2-h |
|---|---:|---:|---:|---:|
| Easy | 52 | 57.69 (C30/avg) | 51.92 | 54.00 |
| Medium | 200 | 54.50 (C24/avg) | 54.50 | 58.50 |
| Hard | 168 | 54.76 (C42/avg) | 53.57 | 59.38 |
| Unknown | 86 | 61.63 (C36/avg) | 56.98 |  |
| Overall | 506 | 54.35 (C24/avg) | 54.35 | 57.51 |
| permanence|Fixed | 52 | 59.62 (C24/avg) | 59.62 | 59.62 |
| permanence|Moving | 68 | 51.47 (C42/avg) | 45.59 | 57.35 |
| immutability|Fixed | 52 | 61.54 (C4/avg) | 57.69 | 55.77 |
| immutability|Moving | 68 | 50.00 (C12/avg) | 45.59 | 58.72 |
| continuity|Fixed | 52 | 55.77 (C30/avg) | 51.92 | 57.69 |
| continuity|Moving | 68 | 69.12 (C24/avg) | 69.12 | 75.00 |
| solidity|Fixed | 52 | 50.00 (C6/avg) | 50.00 | 46.15 |
| solidity|Moving | 94 | 55.32 (C24/avg) | 55.32 | 58.51 |

## C 별 Overall

| 설정 | Overall | 동점 |
|---|---:|---:|
| C4/avg | 48.81 | 0 |
| C6/avg | 49.21 | 0 |
| C8/avg | 49.60 | 0 |
| C10/avg | 48.81 | 0 |
| C12/avg | 51.19 | 0 |
| C14/avg | 50.59 | 0 |
| C18/avg | 51.78 | 0 |
| C24/avg | 54.35 | 0 |
| C30/avg | 54.15 | 0 |
| C36/avg | 53.75 | 0 |
| C42/avg | 53.75 | 0 |

⚠️ 열별 최고는 같은 평가셋에서 격자를 고른 값 — descriptive (CLAUDE.md §1-3).

## 재현

```
python z_research/scripts/analysis/intphys2_column_best.py --run z_research/Benchmarks/exp_results/intphys2/bench_intphys2_main_vith
```