# VOBench_v1 · V-JEPA 2 ViT-H 릴리즈 · gravity (ledge · arc)

> 결과 요약 · 해석 · 단서는 상위 [`../../README.md`](../../README.md) §3-12. 이 폴더는 정리본만 둔다.
> 2026-10-11 재렌더판 (g 375, 꼭대기 f66 · f48 · f30). 옛 판 (g 981 + sliding) 은 `_previous_20261010_g981/`.

| 파일 | 무엇 |
|---|---|
| `tables.md` · `tables.json` | §1 ledge 쌍 정확도 (모델 · 복사 · Δ, 높이 × 속도 셀) · §2 arc: 고른 미래 (전체 토큰 · post · 공 칸 · 복사) · 쌍 정확도 · 튜블릿별 · **튜블릿 묶음 (앞 t0–t2 / 뒤 t3–t7)** · 위치 규칙 기준선 · §4 고른 멈춤 후보의 D (= 멈춘 프레임 − 문맥 끝). per_block.json + token_surprise.npz 에서 재계산, summary.json 과 assert |
| `figures/fig_gravity_choice_<objset>` | **논문용** (Nimbus Roman, 7.0 in). 패널 = 문맥 끝 상태 (오르는 중 · 꼭대기 · 내려가는 중), 막대 = 미래 9 개 중 p 가 가장 가까운 비율 [block 부트스트랩 95 % CI], 막대 아래 = 그 미래의 궤적 (대표 block). 전체 토큰 = 채점값 |
| `figures/fig_gravity_choice_ball_<objset>` | 같은 판형, L1 을 공 칸만 (후보 9 개 공 자리 합집합 3×3) |
| `figures/fig_arc_choice` | 탐색용 한 장 — 행 = objset, 열 = 셀, 모델 (전체 · 공 칸) · 복사 막대 |

원시 산출물: `../../exp_results/surprise_c16t32__vobench_v1_gravity_vith/` (summary.json · per_block.json · token_surprise.npz (e · ec) · _resolved.yaml). 로그 `../../exp_results/logs/{gravity,token_gravity}_c16t32.log`.
대표 block (썸네일): primitive = v11 배경 capsule, realistic = v11 배경 basketball (셀마다 이름순 첫 block).

캡션 초안 — `fig_gravity_choice_<objset>`:
> *Which of nine futures sharing one context the V-JEPA 2 predictor output is closest to (lowest L1 surprise over all future tokens; 12 blocks per panel; error bars: 95 % block-bootstrap CI; dashed: chance 1/9). The context ends (a) 21 frames before the apex, (b) at the apex (the apex is on the splice, 3 frames after the last context frame), (c) 15 frames after it. Below each bar: that future in one representative block (grey: context seen by the model; blue: the possible future; orange: impossible futures; dotted: end of context). stop +Kf follows the true motion for K frames after the splice, then freezes in mid-air.*

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/vobench_gravity_tables.py          # → tables.{md,json}
$P z_research/scripts/figures/plot_vobench_gravity_choice.py      # → figures/fig_gravity_choice{,_ball}_<objset>
$P z_research/scripts/figures/plot_vobench_gravity.py             # → figures/fig_arc_choice
```
(채점 · 토큰 지도 명령은 상위 README `## 재현`)
