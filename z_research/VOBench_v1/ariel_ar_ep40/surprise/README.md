# VOBench_v1 · Ariel AR predictor ep40 (ViT-H frozen encoder) · surprise

> 결과 요약 · 해석 · 단서는 상위 [`../../README.md`](../../README.md) §3-8. 이 폴더는 정리본만 둔다. gravity 판은 [`../gravity/`](../gravity/) (§3-14).

| 파일 | 무엇 |
|---|---|
| `tables.md` · `tables.json` | objset (primitive · realistic) 별 채점표 — 위반 × 가림 시점 (A/B 방향) · 위반 × k · 운동 · 배경 · 물체 · 복사 기준선. per_block.json 에서 재계산 + summary.json assert |
| `figures/fig_timing` | 행 = objset, 열 = 위반, x = 가림 시점 (회색 = late). 막대 = 정확도 [block 부트스트랩 95 % CI], 점 = A · B 방향 |
| `figures/fig_k` | x = k (0 = 가림 없음), 선 = early · mid · late |
| `figures/fig_late_motion` | late 만, x = 운동 6 종 |
| `figures/fig_motion_{visible,early,mid}` | `fig_late_motion` 과 같은 꼴, 가림 시점만 바꿈 (2026-10-11). ⚠️ 칸당 쌍 수: visible **4** (block 2) · early / mid / late 12 (block 6) — visible 은 경향으로도 약하다 |
| `figures/fig_permanence_context` | permanence 만: 물체 문맥 (있는데→있는) · 빈 문맥 (없는데→없는), 가림 시점별 · late × 운동 |

| `figures/token_surprise_vs_vjepa2/{primitive,realistic}.gif · _strip.png` | **토큰별 Δ = e_imp − e_pos 를 불가능 영상 위에 — 왼쪽 V-JEPA 2 릴리즈 · 오른쪽 AR** (late · k = 2 전형 쌍: permanence 등속 · permanence 정지 · shape 등속). **눈금 하나** (두 모델 · 두 objset · 모든 튜블릿 공통, late 미래 토큰 |Δ| 99 분위 = 0.213; AR 은 타깃 공간이 달라 크기는 참고), 흰 원 = 가능 영상의 물체 자리. strip = 문맥 t0–t7 (원본) + 미래 t8–t15 (Δ) 전체 시퀀스. `*_static.gif` · `*_static_strip.png` = 행을 전부 정지 물체로 (permanence · shape · colour, late k=2; `gif_vobench_token_surprise.py _static`). `z_research/scripts/figures/gif_vobench_token_surprise.py` (2026-10-11) |

**AR 의 late permanence 는 물체 자리에서 이기지 않는다 (2026-10-11, late permanence 물체 문맥 쌍 전부, 재등장 뒤 튜블릿).**
움직이는 물체: 물체 자리 Δ primitive −0.159 · realistic −0.103 (Δ > 0 인 쌍 **0 %**) · 나머지 토큰 +0.042 · +0.034 (**100 %**) → 쌍 전체 +0.031 · +0.025 (100 %).
정지 물체: 물체 자리 +0.192 · +0.138 (100 %). 릴리즈는 움직이는 물체에서 물체 자리 · 나머지 모두 음수.
즉 AR 은 움직이는 물체가 다시 나타나는 자리에서는 오히려 "사라진" 미래에 가깝고, **물체 밖 토큰** 에서 이긴다 — v11 의 presence-without-placement (`auto_research/Archive/AR_PREDICTOR_2026-09-27.md` §2-6) 와 같다. 정지 물체만 자리까지 맞는다.
토큰 지도 검증: 두 실행 모두 토큰 평균 = per_video_surprise (상대 ≤ 2.3e-7), 쌍 판정 1,440 개 전부 채점과 같음.

원시 산출물: `../../exp_results/surprise_c16t32__vobench_v1_vith_ariel_ar_ep40 · copy__vobench_v1_vith_ariel_ar_ep40/` (레지스트리 이름 `vith_ariel_ar_ep40`).

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/vobench_tables.py --model ariel_ar_ep40     # → tables.{md,json}
$P z_research/scripts/figures/plot_vobench.py --model ariel_ar_ep40        # → figures/
```
(채점 명령은 상위 README `## 재현`)
