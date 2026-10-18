# bottleneck — predictor 병목 진단 (2026-10-13)

정본 문서: [`auto_research/Archive/PREDICTOR_BOTTLENECK_2026-10-13.md`](../../../auto_research/Archive/PREDICTOR_BOTTLENECK_2026-10-13.md).

| 파일 | 무엇 |
|---|---|
| `<name>/shard*.npz` | VOB_readout 걸음별 자 읽기 · 은닉 풀링 · 표류 · 오차 (name = release · ar40 · ar_state40; roll · tf · injL · injO) — `predictor_bottleneck.py` |
| `analysis.json` | 걸음 이득 · 지연 · ridge probe (속도 · 모양 · 색) · 오차 몫 · 학습 신호 — `predictor_bottleneck_analyze.py` |
| `natural_<name>.json` | SSv2 val 학습 신호 배분 (변한 토큰 vs 정적) — `predictor_signal_natural.py` |
| `late_identity/` | VOBench late 정체: 은닉 블록 평균 vs 참 자리 — `predictor_late_identity.py` |
| `steer/<name>.json` | 상태 속도 코드 조향 — `predictor_velocity_steer.py` |
| `figures/` | 그림 다섯 장 — `z_research/scripts/figures/plot_predictor_bottleneck.py` |

## 2026-10-13 저녁 추가

- `anchor/` — 마지막 문맥 블록 픽셀 조작 (predictor_anchor_test.py), `history/` — 문맥 길이별 손실 (predictor_history_value.py). 해석 정본 `auto_research/paper/DESIGN_CHOICES_2026-10-13.md` §2-2 · §2-6.
- 조립 상한 · 표적 국소성 산출물은 `../headroom/` (design_headroom.py · token_locality.py).

## attention/ (2026-10-13) — ⚠️ 2026-10-14 `../analysis/attention_map/` 로 옮겼다 (모델별 폴더, 그림 포함)

아래는 옮기기 전 기록이다. `ar_attention_history.py` 산출 — AR 셋 (vast AR-state ep70 · Ariel AR ep40 · Ariel AR-state ep40) × (vob_visible · vob_late · ssv2). `<set>_<model>.json` 요약, `_r*.npz` 원자료 (bm_all · bm_obj · om_obj · om_mir · hid · yx · objmap · 예시 ex_A). 해석은 `auto_research/paper/STATUS_2026-10-13.md` B13. 그림 `figures/fig_ar_attention_{blocks,maps}`.
