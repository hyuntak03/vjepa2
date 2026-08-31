# surprise 그림 — 카테고리

`PAPER_STORY_2026-08-31.md` 의 beat 순서로 나눴다.
전부 `z_research/scripts/figures/plot_v11_surprise.py` 가 만들고, 폴더 배정은
그 파일 상단의 `FIGDIR` 표가 정한다. **여기 없는 이름은 최상위에 떨어진다** —
새 그림이 눈에 띄라고 일부러 그렇게 뒀다.

## `01_condition/` — 조건 주효과 (beat 1–2)

| 그림 | 무엇 |
|---|---|
| `fig_occlusion` | 가림 유무 × 위반 종류. 무너지는 곳이 어디인지 |
| `fig_ramp_vs_flat` | 등속 vs 등가속. **물리를 안다면 쉬워야 할 쪽이 더 나쁘다** |

## `02_occlusion_k/` — 가림 길이 (beat 2·4)

| 그림 | 무엇 |
|---|---|
| `fig_k_dose_arm` | `k` × 운동 유형 3팔. 주력 |
| `fig_k_dose` | 운동을 평균낸 `k` 반응. `fig_k_dose_arm` 과 축·폭을 맞춰 나란히 본다 |
| `fig_k_sensitivity` | 같은 축을 민감도로. 보조 |

핵심: **`k=0` 과 `k>0` 사이에서 끊기고 `k=1…4` 는 평평하다.**
가려지는 시간이 아니라 관측이 끊겼다는 사실이 전부다.

## `03_direction/` — 방향 비대칭 (beat 4)

| 그림 | 무엇 |
|---|---|
| `fig_vanish_direction` | 영속성 `빈→물체` 100 / `물체→빈` 0. 50% 가 평균임을 드러낸다 |
| `fig_direction_split` | 조건별 방향 분해 |
| `fig_direction_gap` | 방향 간 격차 |

## `04_object_order/` — predictor 가 무엇을 만드나 (beat 4)

| 그림 | 무엇 |
|---|---|
| `fig_predictor_vote` | 7×7. **모든 칸 = 출력이 열 물체 쪽으로 간 비율.** 대각만 진한 것이 좋은 상태 |
| `fig_vote_grid_shape` | 같은 행렬을 3팔 × `k`=0…4 로 |
| `fig_vote_grid_color` | 색 판 |

행·열은 지킴 순으로 정렬돼 있다. **하삼각만 채워지는 것이 1차원 순서의 모습**이고,
어긋난 칸은 42 중 1~4개다 (등가속+가림).

## `_superseded/` — 대체됨 (지우지 않는다)

| 그림 | 왜 |
|---|---|
| `fig_keep_matrix`, `fig_keep_matrix_visible` | 같은 행렬의 "지켜냈나" 판. `fig_predictor_vote` 가 **"무엇을 만드나"** 로 틀을 뒤집어 대체 |
| `fig_keep_dose` | 지킴 점수 × `k`. `fig_vote_grid_*` 가 같은 것을 행렬로 담는다 |
| `fig_make_prob`, `fig_make_prob_grid_*` | 2지선다를 Luce 로 **7지선다 확률**로 바꾼 판. 아래 참조 |

### `fig_make_prob*` 이 물러난 이유

수식은 맞지만 **최저 상대 하나가 행 전체를 지배한다.** `s = (1-p)/p` 가 `p` 가 작을 때 폭발한다.

```
등가속 · 가림없음   짝평균acc   최저 상대       7지선다 P
cube                 83.3     cylinder  6         8.7
capsule              69.8     cylinder  0         2.8
pyramid              64.6     cylinder 12         9.4
```

스무딩 탓이 아니다 (Jeffreys vs add-1 차이 0.1~6.0pt). 그리고 `k` 별 판은 비대각이
**방향당 n=4** 인데 대각이 그 6칸 전부의 함수라, "개별 칸을 읽지 말라" 던 잡음을
대각선이 통째로 물려받는다. IIA 가정도 행 안에서 검정 불가능하다.

**다만 여기서 나온 사실 하나는 살린다** — 등가속·비가림에서 `cylinder` 가 나머지 여섯을
거의 전부 압도한다 (cube 6%, capsule 0%, pyramid 12% 로 이긴다). 짝평균(cube 83.3)은
이 지배 구조를 가린다. beat 4 의 순서가 **한 물체에 몰려 있다**는 뜻이라 기록해 둔다.

## 재현

```bash
python z_research/scripts/figures/plot_v11_surprise.py \
  --report z_research/IntPhysGenV11/exp_results/report.json \
  --outdir z_research/IntPhysGenV11/figures/surprise
```
