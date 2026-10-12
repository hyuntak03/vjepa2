# IntPhys 2 열별 최고 선택 — `bench_intphys2_main_vith_w32`

격자 6 개: C8/avg, C12/avg, C16/avg, C20/avg, C24/avg, C28/avg

| 열 | 짝 | 열별 최고 (설정) | overall 최고 설정 (C24/avg) 로 읽은 값 | 논문 V-JEPA 2-h |
|---|---:|---:|---:|---:|
| Easy | 52 | 57.69 (C8/avg) | 46.15 | 54.00 |
| Medium | 200 | 55.00 (C24/avg) | 55.00 | 58.50 |
| Hard | 168 | 55.36 (C8/avg) | 51.79 | 59.38 |
| Unknown | 86 | 59.30 (C28/avg) | 53.49 |  |
| Overall | 506 | 52.77 (C24/avg) | 52.77 | 57.51 |
| permanence|Fixed | 52 | 61.54 (C12/avg) | 57.69 | 59.62 |
| permanence|Moving | 68 | 51.47 (C24/avg) | 51.47 | 57.35 |
| immutability|Fixed | 52 | 61.54 (C24/avg) | 61.54 | 55.77 |
| immutability|Moving | 68 | 50.00 (C8/avg) | 41.18 | 58.72 |
| continuity|Fixed | 52 | 55.77 (C8/avg) | 55.77 | 57.69 |
| continuity|Moving | 68 | 63.24 (C28/avg) | 60.29 | 75.00 |
| solidity|Fixed | 52 | 48.08 (C8/avg) | 46.15 | 46.15 |
| solidity|Moving | 94 | 54.26 (C8/avg) | 51.06 | 58.51 |

## C 별 Overall

| 설정 | Overall | 동점 |
|---|---:|---:|
| C8/avg | 51.78 | 0 |
| C12/avg | 51.78 | 0 |
| C16/avg | 50.99 | 0 |
| C20/avg | 50.99 | 0 |
| C24/avg | 52.77 | 0 |
| C28/avg | 52.77 | 0 |

⚠️ 열별 최고는 같은 평가셋에서 격자를 고른 값 — descriptive (CLAUDE.md §1-3).

## 재현

```
python z_research/scripts/analysis/intphys2_column_best.py --run z_research/Benchmarks/exp_results/intphys2/bench_intphys2_main_vith_w32
```