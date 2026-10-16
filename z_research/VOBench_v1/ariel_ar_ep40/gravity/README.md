# VOBench_v1 · Ariel AR predictor ep40 (ViT-H frozen encoder) · gravity (ledge · arc)

> 결과 요약 · 해석 · 단서는 상위 [`../../README.md`](../../README.md) §3-14. 이 폴더는 정리본만 둔다. V-JEPA 2 릴리즈 판은 [`../../vjepa2/gravity/`](../../vjepa2/gravity/) (§3-12).

**주 결과 = L1 retrieval** (미래 9 개 중 L1 이 가장 낮은 것 = predictor 가 그린 미래, 전체 토큰, 사용자 10-11).
⚠️ AR 은 타깃이 **블록별 LN(target_encoder)** (`analysis/predictors/ar_scoring.py`) 이라 surprise 크기는 릴리즈와 비교하지 않는다 — 고른 미래 · 쌍 판정만. 복사 = 같은 공간의 `copy_blk` (문맥 마지막 블록 h 를 미래 블록마다).

| 파일 | 무엇 |
|---|---|
| `figures/fig_gravity_choice_<objset>` | **주 그림** — 문맥 상태 3 패널 · 미래 9 개 고른 비율 [block 부트스트랩 95 % CI] · 궤적 썸네일 (V-JEPA 2 판과 같은 판형) |
| `figures/fig_gravity_choice_ball_<objset>` | 보조 — 공 칸만 |
| `figures/fig_arc_choice` | 탐색용 — 모델 (전체 · 공 칸) · 복사 (copy_blk) 막대 |
| `tables.md` · `tables.json` | ledge 쌍 정확도 + 복사 · arc 고른 미래 (전체 · post · 공 칸 · 복사) · 쌍 정확도 · 튜블릿별 · 튜블릿 묶음 · 위치 규칙 기준선 · 멈춤 D. summary.json assert |

원시 산출물: `../../exp_results/surprise_c16t32__vobench_v1_gravity_vith_ariel_ar_ep40/` (summary.json · per_block.json · token_surprise.npz (e · ec = copy_blk)). 로그 `../../exp_results/logs/{gravity,token_gravity}_ariel_ar_ep40_c16t32.log`.

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
GPUS=4 RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1_gravity vith_ariel_ar_ep40          # 4 GPU 약 4 분
$P z_research/scripts/analysis/token_surprise_maps.py --run z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_gravity_vith_ariel_ar_ep40 --gpus 4 --copy   # kind=ar 분기 (채점기와 같은 ar_window 경로)
$P z_research/scripts/analysis/vobench_gravity_tables.py --model ariel_ar_ep40
$P z_research/scripts/figures/plot_vobench_gravity_choice.py --model ariel_ar_ep40
$P z_research/scripts/figures/plot_vobench_gravity.py --model ariel_ar_ep40
```
