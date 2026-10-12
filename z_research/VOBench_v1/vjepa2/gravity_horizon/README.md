# VOBench_v1 · V-JEPA 2 ViT-H 릴리즈 · gravity_horizon — 예측 길이 M 을 줄이면

> 결과 요약 · 해석 · 단서는 상위 [`../../README.md`](../../README.md) §3-13. 이 폴더는 정리본만 둔다.

**묻는 것**: gravity arc 에서 p 가 "진짜 운동을 몇 샘플 잇다 멈춘다" (§3-12) 가 긴 예측 (16 샘플) 이라서인가.
문맥 16 샘플 (stride 3, 문맥 끝 f45) 고정, 미래 M = 16 · 8 · 4 · 2 샘플 (창 c16t32 · c16t24 · c16t20 · c16t18, 미래 끝 f93 · f69 · f57 · f51).
절대 (문맥 끝에서 일정 거리만 잇는다) 면 곡선이 절대 프레임에서 겹치고, 상대 (예측 길이에 비례) 면 M 이 짧을수록 일찍 멈춘다.

| 파일 | 무엇 |
|---|---|
| `tables.md` · `tables.json` | §1 창 안 멈춤 시점 D (고른 후보의 멈춘 프레임을 창 끝에서 자름 − 45; `끝까지` = 창 안 모든 샘플에서 arc 와 1 프레임 이내) · §2 튜블릿별 "정지 (f45) 보다 진짜 운동에 가까운 block %" — 전체 토큰 · 공 칸, objset × 문맥 상태 × M |
| `figures/fig_horizon_all_<objset>` | **주 그림 (전체 토큰 = 표준 채점 기준, 사용자 10-11)**. objset 마다 한 장, 패널 = 문맥 상태. x = 예측 순번 (문맥 끝에서 몇 번째 샘플), 선 = 예측 길이 16 · 8 · 4 · 2 (선이 끝나는 곳 = 창 끝, 옅은 띠 = block 부트스트랩 95 % CI). 어느 길이든 창의 마지막 튜블릿이 정지 쪽으로 떨어진다 |
| `figures/fig_horizon_ball_<objset>` | 보조 — 같은 판형, 공 칸만. 창 끝 효과가 없고 고정 순번 (6–8 번째, 오르는 중 10–12 번째) 에서 떨어진다 |
| `figures/_superseded/` | 1 차 그림 (2×3, 선 4 개 + 오차 막대) — 보기 어려워 내림. 그 README 에 이유. 2 차 (절대 프레임축 · 모양 다른 표지) 는 같은 파일명으로 덮어씀 |

원시 산출물: `../../exp_results/surprise_c16t32__vobench_v1_gravity_vith{,_wc16t24,_wc16t20,_wc16t18}/` (token_surprise.npz e · ec). 로그 `../../exp_results/logs/{gravity,token_gravity}_c16t{32,24,20,18}.log`.

캡션 초안 — `fig_horizon_all_<objset>`:
> *Does the prediction stop following the true motion because the horizon is long? The context is fixed (16 samples); the predictor forecasts M = 16, 8, 4 or 2 samples. y: blocks whose prediction is closer to the true motion than to the ball frozen at the last context frame, per future tubelet (all tokens, as in the standard surprise; 12 blocks; shading: 95 % block-bootstrap CI). Each line ends where its prediction window ends. Whatever the horizon, the last tubelets of the window turn towards the frozen future.*

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
for n in 24 20 18; do                                                     # M = 8 · 4 · 2 (M 16 = 표준 창, gravity 와 공유)
  SET="window.n_frames=$n window.context=16" GPUS=4 RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1_gravity vith
  $P z_research/scripts/analysis/token_surprise_maps.py --run z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_gravity_vith_wc16t$n --gpus 4 --copy
done
$P z_research/scripts/analysis/vobench_gravity_horizon.py       # → tables.{md,json}
$P z_research/scripts/figures/plot_vobench_gravity_horizon.py   # → figures/fig_horizon_{ball,all}_<objset>
```
(창 하나 4 GPU 채점 약 4 분 + 토큰 지도 약 3 분)
