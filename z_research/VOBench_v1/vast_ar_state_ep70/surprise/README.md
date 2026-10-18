# VOBench_v1 · vast AR-state predictor ep70 (ViT-H frozen encoder) · surprise

> vast 브랜치에서 학습한 AR-state predictor (70 epoch · 20,580 step, `z_training/vast/ar_state/latest.pt`, 레지스트리 `vith_vast_ar_state_ep70`).
> 파일 구성은 [`../../ariel_ar_state_ep40/surprise/`](../../ariel_ar_state_ep40/surprise/README.md) 와 같다 (`tables.{md,json}` · `figures/fig_*`). 채점 경로 `analysis/predictors/ar_scoring.py`.
> 폴더는 2026-10-14 에 만들었다 (채점 실행은 그 전에 있었다).

## 결과 (쌍 정확도 %, matched, VOBench 전체 1,440 쌍)

| | primitive | realistic |
|---|---:|---:|
| overall | 81.5 [79, 84] | 85.3 [83, 88] |
| AR 공간 복사 (copy_blk) | 77.6 | 81.4 |
| **late 전체** | **73.6** [69, 79] | **77.3** [73, 81] |
| late · permanence (A / B) | 93.1 (94 / 92) | 95.8 (100 / 92) |
| late · shape | 61.1 | 65.3 |
| late · colour | 66.7 | 70.8 |

- **ariel AR-state ep40 과 같은 그림이다** (overall 81.4 / 85.0, late permanence 97 대 · shape · colour 60 대). 데이터 · 창 · rollout 길이를 바꾸고 70 epoch 학습해도 late 의 모양 · 색은 오르지 않는다.
- late permanence 는 거의 다 맞히지만, late shape · colour 는 release (late 57.9 / 65.7) 와 같은 수준.
- 복사 대비 Δ +3.9 (두 objset 모두 CI 가 0 을 안 포함).

## VOBench post-FT (2026-10-14, `vobench_vast_ar_state_postft`)

split v2 train 가능 영상 720 clip · 30 epoch · 450 step (ariel ar_state ep40 FT 와 같은 레시피, 출발만 vast ep70). 채점 split v2 test 720 쌍.

| split v2 test | release | release+FT | vast ep70 | **vast ep70 + FT** |
|---|---:|---:|---:|---:|
| 전체 | 75.7 | 91.8 | 82.8 | **91.0** |
| late | 62.0 | 89.4 | 75.9 | **90.3** |
| late permanence (A) | 54.2 (13.9) | 98.6 (97.2) | 95.8 (97.2) | 100 (100) |
| late shape · colour | 65.3 · 66.7 | 87.5 · 81.9 | 63.9 · 68.1 | **87.5 · 83.3** |

- post-FT 뒤에는 release+FT · ariel ar_state+FT 와 같은 자리 (late 89–90) — 출발점 (release · ariel · vast) 이 결과를 가르지 않는다. 같은 분포 안의 결과.
- 채점 산출물 `z_training/runs/vobench_vast_ar_state_postft/eval/surprise_c16t32__vobench_v1_split2_test_latest/`.

## 단서

- 타깃이 블록별 LN(target_encoder) 이라 surprise 크기 · margin 은 릴리즈와 비교하지 않는다. 쌍 정확도만.
- 복사 기준선은 ar ep40 의 복사 실행 `copy__vobench_v1_vith_ariel_ar_ep40` 의 `copy_blk` 를 그대로 쓴다 (encoder 만으로 정해진다).

원시 산출물: `../../exp_results/surprise_c16t32__vobench_v1_vith_vast_ar_state_ep70/` · c16t24 창 `..._wc16t24/`.

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
GPUS=4 DECODE_WORKERS=8 BATCH_SIZE=16 bash z_research/scripts/run.sh surprise_c16t32 vobench_v1 vith_vast_ar_state_ep70
$P z_research/scripts/analysis/vobench_tables.py --model vast_ar_state_ep70     # → tables.{md,json}
$P z_research/scripts/figures/plot_vobench.py --model vast_ar_state_ep70        # → figures/fig_*
```
