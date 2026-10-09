# OcclusionStats — 벤치마크의 가림은 얼마나 길고, 가려진 동안 물체가 움직이나

> 마지막 갱신 2026-09-26 · 바뀐 것: 세트 개설. IntPhys 1 dev 완료. IntPhys 2 · GRASP · InfLevel · 합성 세트는 다음 단계.

**왜 이 세트가 있나.** IntPhys 2 논문 (Fig. 2) 은 벤치마크마다 가림 시간 분포를 그리고, 모델이 IntPhys 2 에서 무너지는 이유로
"더 엄격한 기억 요구" 를 든다. 이 그림은 가려진 **시간**만 재고, 가려진 동안 물체가 **움직이는지**는 가르지 않는다.
여기서는 두 축을 같이 잰다. 우리 이야기 (상태를 진화시키며 운반하는가) 에는 움직임 축이 필요하다.

## 결론 (IntPhys 1 dev, 180 쌍)

**IntPhys 1 의 가림은 양극단이다.**
- 움직이는 물체는 **0.2 초** 가려진다 (2–6 프레임 @ 15 fps). 88.89 칸 (skip 2) 에서 모델에게는 1–2 장이다.
- 5 초 가림은 **정지 물체**뿐이다 (가림막이 올라가 덮는 장면).
- 움직이는 물체가 1 초 넘게 가려지는 사건은 없다.

| | 사건 가림 | 가려진 동안 이동 | V-JEPA 2 (88.89 칸) |
|---|---|---|---|
| 이동 + 가림 (60 쌍) | 0.13–0.4 초 (중앙 0.2), skip-2 로 1–3 장 | 중앙 4.6 물체 폭 | 75.0 |
| 정지 + 가림 (30 쌍) | 1.8–5.9 초 (중앙 4.9) | 중앙 0.14 폭 (안 움직임) | 100 |

근거·방법·검증·정정은 [`Archive/INTPHYS1_OCCLUSION_2026-09-26.md`](Archive/INTPHYS1_OCCLUSION_2026-09-26.md).

**미결:** IntPhys 2 에 같은 두 축을 거는 것. mask 가 없어 추적기가 필요하다. 추적기는 IntPhys 1 (mask 정답) 로 먼저 검증한다.

## 구성

| 경로 | 내용 |
|---|---|
| `Archive/INTPHYS1_OCCLUSION_2026-09-26.md` | IntPhys 1 정본 문서 (방법 · 검증 · 표 · 해석 · 못 하는 주장 · 재현) |
| `intphys1/summary.json` | 조건별 통계, 문턱 민감도, 가림 길이별 정답 |
| `intphys1/gaps.csv` | 가림 하나당 한 줄 (영상, 시작·끝, 길이, skip-2/5 장수, 이동 거리, 사건 여부) |
| `intphys1/pairs.csv` | 쌍 하나당 한 줄 (d, 사건 종류, 사건 가림, V-JEPA 2 정답) |
| `intphys1/videos.json` | possible 영상별 프레임 가시성 · 물체 위치 · 크기 |
| `intphys1/figures/` | `fig_visibility_raster.png` (쌍 × 프레임), `fig_gap_stats.png` (길이 · 이동 · 정답) |
| `intphys1/_superseded/`, `intphys1/figures/_superseded/` | 같은 날 정정된 v1 · v2 산출물 (그곳 README 에 이유) |

스크립트: `z_research/scripts/analysis/occlusion_intphys1.py`, `z_research/scripts/figures/plot_occlusion_intphys1.py`.
