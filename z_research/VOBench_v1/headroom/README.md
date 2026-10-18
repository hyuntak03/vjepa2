# headroom — 학습 없는 조립 상한 · 표적 국소성 (2026-10-13)

해석 정본: `auto_research/paper/DESIGN_CHOICES_2026-10-13.md` §2-4 · §2-5.

| 파일 | 무엇 | 만든 것 |
|---|---|---|
| `{vobench,gravity}_r*.npz` | 렌더 (copy · ar · move_or · move_cv · move_ca) 별 블록 S 와 토큰 오차 지도 `E_*` (8×16×16, fp16), 옮김 가능성 `transport` | `z_research/scripts/analysis/design_headroom.py extract` |
| `headroom.json` | VOBench matched 쌍 정확도 (셀 · 방향별), gravity 고른 미래, ledge 쌍 | `design_headroom.py analyze` |
| `locality_r*.npz` · `token_locality.json` | 두 미래 (물체만 다름) 표적 차의 물체 자리 몫 | `token_locality.py` |
| `run.log` | 추출 로그 | — |

⚠️ 공간은 Ariel AR 과 같은 **블록별 LN(target_encoder)** 다 (표준 채점의 창 단위 표적과 다르다). move_* 는 문맥 라벨 위치를 쓴다 (지각 완벽 가정).
