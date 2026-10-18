# compare/_superseded

| 그림 | 언제 · 왜 물러났나 | 대체 |
|---|---|---|
| `fig_release_vs_postft_k{1,4,8}` · `fig_vast_ar_state_vs_postft_k{1,4,8}` | 2026-10-14. 미래 블록 하나 (k) 마다 층 × 블록 행렬이라 "t9 를 만들 때, t10 을 만들 때 …" 를 한눈에 못 본다 (사용자: "그래프가 전혀 와닿지 않는다") | `../fig_{release,vast_ar_state}_vs_postft_rows_{visible,late}` — 행 = 걸음 (making t9 … t16) |

물러났지만 **이 그림에만 있는 것**: 층별 곡선 (d)(h) 의 **물체 자리 vs 같은 좌표 대조** (물체가 3 토큰 이상 움직인 clip). post-FT 가 "물체를 본다" 는 주장의 대조는 여기와 `../../README.md` 표에 있다.
다시 그리기: `python z_research/scripts/figures/plot_attention_layers.py --pair {release,vast} --step {0,3,7}` (출력이 이 폴더로 간다).
