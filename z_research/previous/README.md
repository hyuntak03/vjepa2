# z_research/previous — 2026-10-10 이전 연구 세트 (보관)

2026-10-10 에 `z_research/` 최상위의 연구 세트를 전부 이 폴더로 내렸다. 새 세트는 `z_research/VOBench_v1/` 부터 시작한다.
**`z_research/scripts/` 는 옮기지 않았다** — `run.sh` 표준 진입점이고 새 세트도 같은 스크립트를 쓴다.

| 옮긴 것 | 옛 자리 → 새 자리 |
|---|---|
| `anticipation` · `Benchmarks` · `context_encoder_analysis` · `IntPhys` · `IntPhysGenV8` · `IntPhysGenV10` · `IntPhysGenV11` · `IntPhysGenV11_occlusion_timing_ablation` · `OcclusionStats` · `predictor_training` · `RollOutV2` · `RollOutV3` · `TrainingEffects` · `v11_realistic` · `v11_roll_out` | `z_research/<셋>` → `z_research/previous/<셋>` |
| `HANDOFF_2026-09-22.md` · `OVERVIEW_2026-09-22.md` | 같은 규칙 |

## 경로 갱신

- **고친 것** (147 파일, `z_research/<셋>` → `z_research/previous/<셋>`, 정확한 폴더 이름 일치만): `*.py` · `*.sh` · `*.sbatch` · `*.yaml` (z_research · cvpr · z_training · analysis · configs/protocols), `configs/protocols/datasets.md` (`results_root`), `CLAUDE.md`.
  손으로 고친 둘: `anticipation/EK100/selftest.py` (`parents[3]` → `parents[4]`), `scripts/analysis/intphys2_grid_best.py` (`parents[2] / "previous/Benchmarks/…"`).
- **안 고친 것**: 각 세트 안의 `.md` 문서 본문, `auto_research/`, `z_archive_10/`, `cvpr/` · `z_training/` 의 `.md`. 그 안의 `z_research/<셋>/…` 표기는 위 표대로 `previous/` 를 끼워 읽는다. 세트끼리의 상대 링크 (`../RollOutV2/…`) 는 같이 옮겼으므로 그대로 산다.
- 원래 없던 폴더를 가리키는 `results_root` (`IntPhysGenV13_Black_occluder`, `IntPhysGenV10_low_occlusion`, `IntPhys-Like_data`) 는 손대지 않았다.

## 검증 (2026-10-10, vll5)

- `python cvpr/harness/check_equivalence.py` → 동일 27 · 차이 0 · 건너뜀 1 (옮기기 전과 같음)
- `python z_research/scripts/analysis/check_oracle.py` → 통과 (IntPhys 1 88.89 재현)
- `DRYRUN=1 bash z_research/scripts/run.sh surprise_c16t32 v11 vith` → `output_dir` = `z_research/previous/IntPhysGenV11/exp_results/…`
- `python z_research/previous/anticipation/EK100/selftest.py` → 전부 PASS

## 재현

```bash
# 옮기기 전 파일 144 개의 사본 (경로 수정 전): 세션 스크래치패드 before_repath.tar (세션 종료 시 사라짐)
# 되돌리기: mv z_research/previous/<셋> z_research/ 후, 같은 정규식을 반대로 (z_research/previous/<셋> → z_research/<셋>)
```
