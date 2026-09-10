# 그림 — 읽는 순서

`01`~`03` 은 `z_research/scripts/figures/plot_v11_timing.py`, `06_knockout/` 은
`plot_v11_knockout.py` 가 만든다. **그림 하나 = 주장 하나.** 폴더 번호가 논증 순서다.

```bash
python z_research/scripts/figures/plot_v11_timing.py \
  --exp    z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results \
  --index  data_csv/intphysgen_v11_full/index_probe.csv \
  --outdir z_research/IntPhysGenV11_occlusion_timing_ablation/figures
```

실행할 때마다 `report.json` 의 `cells` 를 재합산해 `overall` 과 대조한다 (틀리면 죽는다).

```
figures/
├── 01_scoring/          fig_timing_step
│   └── by_k/            fig_timing_k
├── 02_probing/
│   ├── 01_self/         fig_self_{shape,color}
│   │   └── by_k/        fig_self_k_{shape,color}
│   ├── 02_vis_occ/      fig_vis_occ_{shape,color}
│   │   └── by_k/        fig_vis_occ_k_{shape,color}
├── 03_mechanism/        fig_margin · fig_margin_vs_acc
└── 06_knockout/         fig_ko_layers · fig_ko_drift
    ├── by_violation/    fig_ko_violation
    ├── by_motion/       fig_ko_motion
    ├── by_timing/       fig_ko_timing · fig_ko_timing_{vanish,shape,color}
    └── by_k/            fig_ko_k · fig_ko_k_{vanish,shape,color}
```

```bash
python z_research/scripts/figures/plot_v11_knockout.py \
  --results z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/knockout__v11_full_vith/results.json \
  --outdir  z_research/IntPhysGenV11_occlusion_timing_ablation/figures
```

---

## 규칙

### 수치 출처는 하나씩이다

| | 출처 |
|---|---|
| 채점 | `surprise_c16t32__v11_full_vith` → `report.json` |
| probing **전부** | **`attn_probe_xfer__v11_full_vith` 한 run** |
| 기전 | `surprise_c16t32__v11_full_vith/per_block.json` |
| knockout | `knockout__v11_full_vith/results.json` (`merge.py` 가 `score_blocks` 로 재채점한 `breakdown`) |

⚠️ probing 을 두 run 에서 섞어 읽지 말 것. `attn_probe_xfer` 가 상위집합이다
(`z` self + `h` self·`h→p` + `p` self·`p→h`). 전에 `attn_probe` 와 섞었더니
같은 self 칸이 108 중 1개에서 **6.1pt** 벌어졌다 (미수렴 head).

### 색

**선은 전부 서로 다른 색** — 파랑 `#2a78d6` · 주황 `#eb6834` · 초록 `#1baf7a`.
명도만 다른 램프는 범례에서 안 읽혀서 안 쓴다. 대신 **한 그림에서 팔레트를 쓰는 축은 하나뿐**이다:

| 그림 | 선이 뜻하는 것 | 나머지는 |
|---|---|---|
| `fig_timing_step` | 운동 3종 | 위반=패널, 타이밍=x축 |
| `by_k/fig_timing_k` | 타이밍 3종 | 위반=패널(제목 **중립색**), k=x축 |
| `fig_self_*` · `fig_vis_occ_*` | 지점 z / h / p | 운동=패널, 타이밍=x축 |
| `by_k/*` (probing) | 타이밍 3종 | **한 수치만** 그린다, 축 라벨 중립색 |
| `fig_margin` | 위반 3종 | 타이밍=x축 |
| `06_knockout/*` | 간선 3종 (mask reads ctx / mask reads mask / ctx reads ctx) | 위반·운동·타이밍·k = **패널**, 층 = x축 |

선은 파스텔(명도만 +26%, 채도 유지), **값 라벨·제목은 원색**. 대비가 낮으므로
**모든 점에 값을 직접 단다** (CLAUDE.md §8-4). 라벨은 같은 x 에서 세로로 쌓고
다른 계열 마커를 피해 밀어낸다.

### `by_k/` 는 전부 같은 판이다

`x = k (0 = 가림막 없음)` · `선 = 타이밍` · `패널 = 운동` · **나머지는 전부 고정**.
공용 함수 `_by_k()` 하나가 셋을 다 그리므로 세 폴더를 나란히 놓고 같은 방식으로 읽는다.

---

## `01_scoring/` — 무엇이 무너지는가

**`fig_timing_step`** — 계단은 위반마다 다르다.

| 위반 | vis | early | mid | late |
|---|---:|---:|---:|---:|
| Object permanence | 100.0 | 97.8 | 95.8 | **57.5** ← 깨끗한 계단 |
| Shape consistency | 84.3 | 78.1 | 76.0 | **55.4** ← 두 번 떨어진다 |
| Colour consistency | 81.5 | **67.4** | 68.7 | 65.1 ← 계단이 아니다 |

운동별로는 `late` 에서 58.9 / 60.0 / 59.7 로 **셋이 한 점에 모인다**
(비가림에서 11pt 벌어져 있던 차이가 사라진다).

**`by_k/fig_timing_k`** — `k=0 → 1` 한 칸이 전부다. 그 뒤로는 평평하다.

## `02_probing/` — 정보인가 자리인가

### `01_self/` — 정보는 어디서도 안 없어진다

`p` self probe (chance 12.5):

| | vis | early | mid | late |
|---|---:|---:|---:|---:|
| shape | 100 | 100 | 100 | 95.2 ~ 98.9 |
| colour | 100 | 99.5 ~ 100 | 98.9 ~ 100 | 86.3 ~ 99.2 |

`env` 는 36칸 전부 100.0. **채점이 떨어지는 것은 정보 손실이 아니다.**

### `02_vis_occ/` — 비가림에서 배운 readout 을 가림에 걸면

`p`, colour: `early` 98.6 / 99.9 / 100 → `mid` 98.2 / 99.4 / 98.9 → **`late` 15.1 / 20.6 / 16.4**
(static / flat / ramp). `z` · `h` 는 `late` 에서도 93~100 을 유지한다.

**`p` 만, `late` 에서만 자리가 바뀐다.** `by_k/` 를 보면 그 붕괴가 **`k=0 → 1` 한 칸**에서 일어난다.

### GT 미래(h) ↔ 예측 미래(p) 이식은 그림에서 뺐다 (2026-09-03)

`h→h` / `p→p` 는 전부 100 이라 정보가 없고, `h→p` 의 `late` 붕괴는 `02_vis_occ` 와 겹친다.
건질 것은 하나 — **`h→p` 가 가림막 없는 조건에서 이미 낮다** (static 95 / flat 79 / ramp 57).
그 숫자는 아래 `03_mechanism` 에 넣었다. 전수는 `../README.md` §전수 기록 8.
`p→h` 가 잘 되는 비대칭은 부분공간 포함인지 probe 지름길인지 못 가려서 주장으로 안 쓴다.

## `03_mechanism/` — 왜 떨어지는가

채점은 `|p − h(가능)|` vs `|p − h(불가능)|` 의 대소다. 그 차이(margin)는
**base 의 0.08 ~ 3.7% 밖에 안 된다.** 벤치마크가 재는 게 전부 그 얇은 층이다.

base 가 왜 그렇게 큰가 — **`p` 와 `h` 는 가림막이 없어도 같은 좌표계가 아니다.**
GT 미래에서 배운 readout 을 예측 미래에 걸면(`h→p`, colour) static 95 / flat 79 / **ramp 57** 이다
(`h→h`·`p→p` 는 100). `|p − h|` 의 대부분은 이 좌표 불일치이고, 물체 신호는 그 위의 얇은 층이다.

| | base `|p−h|` | margin (% of base) | 효과크기 |
|---|---:|---:|---:|
| no occluder | 0.573 ~ 0.578 | 0.27 ~ 3.73 | 0.76 ~ 1.30 |
| early / mid | 0.578 ~ 0.583 | 0.11 ~ 2.39 | 0.47 ~ 1.17 |
| **late** | **0.584 ~ 0.589** | **0.08 ~ 0.77** | **0.22 ~ 0.44** |

정확도는 `margin > 0` 인 비율이므로 **효과크기가 곧 정확도**다 —
`fig_margin_vs_acc` 의 **Spearman ρ = 0.986** (12셀).

⚠️ `fig_margin` (a) 에서 shape 와 colour 의 base 는 소수 셋째 자리까지 같다
(둘 다 물체 하나만 바꾼 것이라 장면 통계가 같다). 선이 겹쳐 x 를 조금 밀어 그렸다.

---

## `06_knockout/` — 어느 경로가 채점을 나르는가

predictor 의 attention 간선을 층 창(**±3, 폭 7**)마다 끊고 채점 정확도를 다시 쟀다.
x = 창의 **중심 층** (0 = L0-3 … 11 = L8-11), 맨 오른쪽 `all` = 12층 전부.
y 는 **절대 정확도**, 회색 점선이 그 패널 부분집합의 기준선(`clean_null`), 검은 점선이 chance.

| 그림 | 패널 | 읽을 것 |
|---|---|---|
| `fig_ko_layers` | 전체 | `mask reads ctx` 만 떨어진다 (75.9 → 38.3). 중심 0~4 평평(−10), 5부터 꺾여 8 이후 포화 |
| `fig_ko_drift` | 전체 | 예측 자체의 이동 `|p−p_clean|/|p_clean|`. **정확도가 안 변한 `ctx reads ctx` 도 여기서는 움직인다** |
| `by_violation/` | vanish / shape / colour | colour 는 앞 층을 끊으면 오히려 오른다 (70 → 74) |
| `by_motion/` | static / flat / ramp | 앞 창(L0-3)의 효과는 운동에서만 (static −3 vs moving −9~−16) |
| `by_timing/` | visible / early / mid / late | 꺾이는 자리는 같다. late 는 기준선이 낮아 폭이 작을 뿐 |
| `by_k/` | k = 1~4 (가림 조건만) | 곡선이 겹친다 — k 무관 |

⚠️ **Δ 가 아니라 절대값을 그린 이유** — late 처럼 기준선이 chance 근처인 칸은 떨어질 여지가
없어 Δ 만 보면 "영향이 없다" 로 잘못 읽힌다.
⚠️ **38% 는 chance 아래다.** 문맥을 못 읽은 예측은 무작위가 아니라 고정 예측이고 불가능 미래에
더 가깝다 (visible/vanish 13%). 절대값이 아니라 **꺾이는 자리**를 읽을 것.
⚠️ 간선 이름은 `<query>-><key>` 다. 사용자 표기 "context → mask" = 하네스 `mask->ctx` = 그림 "mask reads ctx".

---

## 합치면

1. **채점이 떨어진다** — 위반마다 다른 모양으로 (01)
2. **정보 때문이 아니다** — `p` 에서 다 읽힌다 (02/01_self)
3. **`late` 는 자리가 바뀐다** — `p` 이식만 무너진다 (02/02_vis_occ)
4. **`early`/`mid` 는 자리도 그대로다** — 이식이 98~100 인데 채점은 떨어진다 (02+01)
5. **`p` 와 `h` 는 애초에 같은 좌표계가 아니다** — 비가림에서도 `h→p` 57~95 (02/03_gt_pred)
6. → 그 전부를 설명하는 건 **효과크기** 하나다 (03)

**`early`/`mid` 와 `late` 는 서로 다른 고장이다.**
early/mid 는 신호가 묽어진 것뿐이고, late 는 거기에 **자리 변화가 더해진다.**

---

## 단서

- `(조건, k)` 셀은 172~215 clip 이다. 점이 아니라 **선의 모양**을 읽을 것
- **shape 은 colour 만큼 깨끗하지 않다** — `flat` 팔에서 `vis→` 가 early/mid 에도
  56~62 로 떨어진다. **"계단" 은 colour 에서 확정, shape 에서는 경향**이다
- self 는 `evals[fit]["per_group"][fit_group]` 이다. `evals[fit]["overall"]` 은
  **12조건 평균**이라 이식 실패가 섞인다 (2026-09-02 에 이걸 혼동한 적이 있다)
- 미수렴 head 1개 (`shape / p / static_occlusion`, train_acc 94.4). 그 칸은 95.2 이고
  chance 12.5 보다 훨씬 위라 결론에는 영향이 없다
- `z` 는 표현 이식에서 뺐다 — `z` 는 문맥 구간(1–16), `h`·`p` 는 미래 구간(17–32) 이라
  토큰의 시각이 다르다

## 검증

수치는 네 층으로 대조했다 (2026-09-02, 전부 **불일치 0**):

| 대조 | |
|---|---|
| 채점 cell 117개 ↔ `per_block.json` 원값 재채점 | n 까지 일치 |
| probing `per_group` ↔ `predictions.json` clip 별 재계산 | 두 run 전부 |
| 그림에 찍히는 값 전수 ↔ 독립 경로 재계산 | |
| `by_k` 를 k 로 다시 합친 값 ↔ 합산 판 | 서로 다른 파일에서 온다 |
| knockout `breakdown` 을 전부 합친 값 ↔ `specs[].metrics` 전체값 (39개 전부) | `plot_v11_knockout.py` 가 매번 대조, 불일치 0 (2026-09-03) |
