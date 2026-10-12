# IntPhys 2 열별 최고 선택 — `bench_intphys2_main_vith_w16`

격자 6 개: C4/avg, C6/avg, C8/avg, C10/avg, C12/avg, C14/avg

| 열 | 짝 | 열별 최고 (설정) | overall 최고 설정 (C10/avg) 로 읽은 값 | 논문 V-JEPA 2-h |
|---|---:|---:|---:|---:|
| Easy | 52 | 55.77 (C8/avg) | 50.00 | 54.00 |
| Medium | 200 | 57.50 (C10/avg) | 57.50 | 58.50 |
| Hard | 168 | 51.79 (C10/avg) | 51.79 | 59.38 |
| Unknown | 86 | 50.00 (C4/avg) | 47.67 |  |
| Overall | 506 | 53.16 (C10/avg) | 53.16 | 57.51 |
| permanence|Fixed | 52 | 57.69 (C10/avg) | 57.69 | 59.62 |
| permanence|Moving | 68 | 58.82 (C14/avg) | 52.94 | 57.35 |
| immutability|Fixed | 52 | 53.85 (C12/avg) | 48.08 | 55.77 |
| immutability|Moving | 68 | 54.41 (C10/avg) | 54.41 | 58.72 |
| continuity|Fixed | 52 | 57.69 (C6/avg) | 55.77 | 57.69 |
| continuity|Moving | 68 | 61.76 (C4/avg) | 57.35 | 75.00 |
| solidity|Fixed | 52 | 53.85 (C14/avg) | 51.92 | 46.15 |
| solidity|Moving | 94 | 53.19 (C4/avg) | 48.94 | 58.51 |

## C 별 Overall

| 설정 | Overall | 동점 |
|---|---:|---:|
| C4/avg | 51.38 | 0 |
| C6/avg | 51.19 | 0 |
| C8/avg | 50.79 | 0 |
| C10/avg | 53.16 | 0 |
| C12/avg | 51.28 | 1 |
| C14/avg | 50.20 | 0 |

⚠️ 열별 최고는 같은 평가셋에서 격자를 고른 값 — descriptive (CLAUDE.md §1-3).

## 재현

```
python z_research/scripts/analysis/intphys2_column_best.py --run z_research/Benchmarks/exp_results/intphys2/bench_intphys2_main_vith_w16
```