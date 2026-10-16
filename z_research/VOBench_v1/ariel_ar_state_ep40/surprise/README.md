# VOBench_v1 · Ariel AR-state predictor ep40 (ViT-H frozen encoder) · surprise

> `ar` 와 같고 rollout 되먹임만 predictor 상태 (predictor_norm 출력, 384) 로 바꾼 팔 (ariel `c973560`, 레지스트리 `vith_ariel_ar_state_ep40`).
> 파일 구성은 [`../../ariel_ar_ep40/surprise/`](../../ariel_ar_ep40/surprise/README.md) 와 같다 (`tables.{md,json}` · `figures/fig_*`). 채점 경로 `analysis/predictors/ar_scoring.py` (kind=ar 와 같다).

## 결과 (2026-10-11, 쌍 정확도 %, matched)

| objset | ar ep40 | **ar_state ep40** | AR 공간 복사 (copy_blk) | 쌍 단위 비교 (ar 만 맞춤 / ar_state 만 맞춤, McNemar) |
|---|---:|---:|---:|---|
| primitive | 81.9 | **81.5** | 77.6 | 11 / 8, p 0.65 |
| realistic | 86.4 | **85.4** | 81.4 | 10 / 3, p 0.09 |

- **ar 와 구분되지 않는다.** 위반 × 가림 시점 칸이 모두 ±4pt 안이고, 쌍 단위 판정 차이는 유의하지 않다.
- late permanence 는 ar 와 같이 거의 다 맞힌다 (primitive 97.2 · realistic 95.8; ar 98.6 · 100). 떨어진 몇 쌍은 ramp 운동의 B 방향에서 나왔다 (칸당 6 쌍).
- late shape · colour 는 ar 와 같이 chance 근처다 (정지 물체만 높다).

## 단서

- ar 와 학습 데이터 · 예산도 다르다 (K400 fps 6 + SSv2 stride 1 · 32 장 창 · C {2,4,8} → K {4,6,8} · 11,760 step). 차이 (또는 차이 없음) 를 되먹임 하나의 효과로 읽지 않는다.
- 복사 기준선은 ar 의 복사 실행 `copy__vobench_v1_vith_ariel_ar_ep40` 의 `copy_blk` 를 그대로 쓴다 — encoder 만으로 정해져 predictor 와 무관하다. 그 실행 안의 predictor 재계산 대조 (`own`) 는 predictor 가 달라 표에서 뺐다.
- 타깃이 블록별 LN(target_encoder) 이라 surprise 크기 · margin 은 릴리즈와 비교하지 않는다. 쌍 정확도만.

원시 산출물: `../../exp_results/surprise_c16t32__vobench_v1_vith_ariel_ar_state_ep40/` (+ `.log`).

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
GPUS=4 DECODE_WORKERS=8 BATCH_SIZE=16 bash z_research/scripts/run.sh surprise_c16t32 vobench_v1 vith_ariel_ar_state_ep40   # vll5, 54 분
$P z_research/scripts/analysis/vobench_tables.py --model ariel_ar_state_ep40     # → tables.{md,json}
$P z_research/scripts/figures/plot_vobench.py --model ariel_ar_state_ep40        # → figures/
```
