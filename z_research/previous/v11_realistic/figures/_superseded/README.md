# _superseded — v11_realistic 옛 그림 (2026-10-06)

**물러난 이유: 보기 어렵다 (사용자 지적 2026-10-06).** 본 그림은 `../fig_rl_summary` 로 바꿨다
(`IntPhysGenV11_occlusion_timing_ablation/figures/01_scoring/fig_timing_step` 과 같은 판형 — 패널 3, 선 = 운동 팔, 값 라벨).

| 그림 | 무엇 | 아직 쓸 데 |
|---|---|---|
| `fig_rl_occlusion` | 운동 팔 × (기하 v11 / 실물) × (가림 없음 / 가림) 막대 + 95 % CI | CI 가 필요할 때 |
| `fig_rl_direction` | k × 방향 (물체→빈 / 빈→물체), 실물 + 기하 v11 회색 참조선 | **A/B 방향 분해는 여기에만 있다** (CLAUDE.md §1-7) |
| `fig_rl_ledge` | ledge 예시 프레임 · 쌍별 surprise 산점도 · 상대 차 | 프레임 예시 · 차이 크기 |

다시 그리기: `$P z_research/scripts/figures/plot_v11_realistic.py --all` (본 그림과 함께 figures/ 최상위에 다시 떨어진다 — 그 뒤 여기로 옮길 것).

| `fig_gravity_choice_4futures` | gravity_realistic **미래 넷** 판 (2026-10-07 17:40 채점). 데이터가 같은 날 20:36 에 **미래 다섯** (`imp_stop` 추가) · 꼭대기 f66/f48/f30 · g 120 으로 다시 렌더됐다 | 옛 데이터의 결과 — 수치 `exp_results/_previous_20261007_4futures/` |
