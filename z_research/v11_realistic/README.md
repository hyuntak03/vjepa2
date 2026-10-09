# v11_realistic — 실물 메시로 다시 렌더한 v11 의 surprise 채점 (2026-10-06 개설)

**무엇을 묻나.** v11 의 vanish 결과 (가림이 문맥 경계에 걸리면 무너지고, 무너질 때 A 물체→빈 0 % / B 빈→물체 100 %) 가
기하 도형 (cube · sphere …) 대신 **실물 메시** (공 · 자전거 · …) 에서도 같은가. 그리고 선반 낙하 (gravity) 를 따로 잰다.

**프로토콜** — 표준 `surprise_c16t32` 그대로 (CLAUDE.md §1-7): ViT-H, raw 100 프레임 stride 3 → 32 샘플,
문맥 16 (raw 0–45) · 예측 16 (raw 48–93), target encoder 32 장 전부, latent L1, `target_layer_norm`, matched pair.

## 데이터

| 세트 | 레지스트리 | clip | block | matched pair | 위반 | 문맥 무결성 |
|---|---|---:|---:|---:|---|---|
| `IntPhysGen_v11_realistic` | `v11_realistic` | 1,440 | 360 | 720 | vanish 만 | 720 쌍 mismatch 0 |
| `IntPhysGen_v11_realistic_ledge` | `v11_realistic_ledge` | 240 | 120 | 120 | gravity (낙하 vs 공중 직진) | 120 쌍 mismatch 0 |

- 프레임 `/local_datasets/world/world_analysis/IntPhysGen_v11_realistic{,_ledge}` (vll4 에서 렌더, 2026-10-06), 288×288, 16 fps
- index `data_csv/intphysgen_v11_realistic{,_ledge}/index.csv` — `z_research/scripts/data/build_intphysgen_index.py` 가 metadata 에서 만든다
  (v11 index 와 같은 14 열 + `sym_k`. `violation_type` 은 block 단위)
- **realistic**: v11 과 같은 6 조건 × 240 clip. 물체 15 종 (실물 메시), 색은 `native` (메시 고유). 배경 4 종. 물체 겉보기 35.6 px.
  `sym_k` 0 (가림 없음, 720 clip) / 1–4 (각 180). block = 4 중항 (pos_a 물체 · pos_b 빈 · imp_ab 사라짐 · imp_ba 나타남)
- **ledge**: 6 조건 = 수평 속도 3 (140 · 180 · 220 cm/s) × 깊이 2 (z 300 · 400), 조건당 20 block. 물체 20 종.
  block = 2 clip (`pos_fall` 포물선 낙하 · `imp_float` 같은 높이로 직진), 과거 공유. 가림 없음
- ⚠️ 규모가 v11 의 **1/4** (쌍 기준, vanish 행 2,688 대비 720) 다. 조건 × 방향 칸 60 쌍, 가림 조건 × k × 방향 칸 **15 쌍** — k 칸은 선 모양만 읽는다

## 결과 (2026-10-06, vll3 · ViT-H · `surprise_c16t32`)

수치는 `exp_results/tables.{md,json}` — `v11_realistic_tables.py` 가 per_block.json 에서 다시 세고 summary.json 의 overall 과 대조 검증했다.
[ ] = block 부트스트랩 95 % CI.

### 1. 실물 메시에서도 v11 vanish 결과가 그대로 나온다

overall **79.6 % [77, 82]** (720 쌍, tie 0) · 원본 v11 vanish 행만 78.8 % [77, 80] (2,688 쌍).

| condition | 실물 전체 | A 물체→빈 | B 빈→물체 | 기하 v11 전체 | 기하 A | 기하 B |
|---|---:|---:|---:|---:|---:|---:|
| `static_visible` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `static_occlusion` | 72.5 [66, 79] | 61.7 | 83.3 | 70.5 [67, 74] | 67.0 | 74.1 |
| `moving_visible_flat` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `moving_occlusion_flat` | 55.0 [52, 59] | 10.0 | 100.0 | 52.0 [51, 53] | 4.0 | 100.0 |
| `moving_visible` (ramp) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `moving_occlusion` (ramp) | 50.0 [50, 50] | **0.0** | **100.0** | 50.0 | 0.0 | 100.0 |

- 가림 없음 세 팔 100 · 가림이 문맥 경계에 걸리면 무너짐 · 무너질 때 **A 물체→빈 0–10 / B 빈→물체 100** — 셋 다 기하 도형 v11 과 같다.
  CI 가 모든 칸에서 겹친다
- k 1–4 는 평평하다 (이동 팔 A: flat 20 / 20 / 0 / 0, ramp 0 / 0 / 0 / 0, n=15 / 칸) — v11 과 같은 모양
- 물체 15 종 모두 비가림 100. 가림 A 는 물체마다 8–42 % (n=12 / 물체, CI 폭 ±25) — 물체 간 차이를 주장할 표본이 아니다

### 2. 선반 낙하 (ledge) — 120 쌍 전부 틀린다

overall **0.0 %** (120 / 120 쌍에서 surprise(공중 직진) < surprise(낙하)). 속도 3 × 깊이 2 × 물체 20 **모든 칸 0 %**.
상대 차 (불가능 − 가능) / 가능 의 중앙값 −1.7 % (조건별 −1.1 ~ −2.5 %), 가장 덜 틀린 쌍도 −0.3 %.
느릴수록 · 멀수록 (z 400) 더 크게 틀린다 (140 cm/s z400 중앙 −2.5 % ↔ 220 z300 −1.1 %).

- 프레임은 맞다 (`figures/fig_rl_ledge` (a)): 가능 clip 은 단 끝에서 포물선으로 떨어지고 (y 95 → 152 px), 불가능 clip 은 같은 높이로 계속 간다.
  둘 다 화면 안에 남는다
- RollOutV2 위치 자의 ledge 결과 (CLAUDE.md §5-5: p 는 선반 높이 유지, 부유 궤적 쪽 attention) 와 같은 방향이다.
  **채점으로도 "p 의 미래 = 문맥의 수평 운동을 같은 높이로 이어 감"** 이 드러난다
- ⚠️ 낙하는 문맥에 없는 사건 (단 끝) 이다. ~~state evolution 정의 (CLAUDE.md §0) 에서는 장면 요소 · 사건별 변화로 정의 밖 (common sense) 이라,
  이 결과를 "전이 능력 실패" 로 쓰지 않는다~~
  **⚠️ 2026-10-08 번복 (사용자) — ledge · wall · 꼭대기는 논문에 넣는다 (3 전이의 3b 사건).** 단서: 떠 가는 미래는 복사와 닮아 VoE 쌍 하나로는 "불가능한 연속" 과 "머묾" 을 못 가른다 (H23 §3-7) — 정본 `auto_research/paper/PAPER_STORY_2026-10-08.md` §2-3 (iii). 이 결과는 **3b 사건 실패** 의 근거로 쓴다 — 아래 '예측 구간 내용량' 단서와 함께.
- ⚠️ 예측 구간의 내용량이 다르다 — 낙하 clip 은 물체가 화면에서 더 많이 움직인다 (수직 이동 + 그림자 분리). 0 % 가
  "낙하를 모른다" 인지 "덜 움직이는 미래가 늘 싸다" 인지는 이 데이터로 못 가른다 (같은 높이 직진은 문맥 끝 속도 그대로라 L1 이 작을 수 있다)

### 그림 — `figures/`

**`fig_rl_summary`** (본 그림) — 캡션:
> Pairwise accuracy of V-JEPA 2 ViT-H (`surprise_c16t32`, matched pairs). (a) Object permanence with realistic meshes and
> (b) with the geometric shapes of v11, by occlusion length k at the context end (k = 0: no occluder; shaded: occluded);
> lines are motion types. (c) Falling off a ledge (realistic meshes), by horizontal speed (two depths pooled, n = 40 pairs per speed):
> the impossible continuation that floats on at the ledge height is judged less surprising than the fall in all 120 pairs.
> Dashed line: chance. Direction split (object→empty / empty→object): `tables.md` and `_superseded/fig_rl_direction`.

옛 세 장 (`fig_rl_occlusion` · `fig_rl_direction` · `fig_rl_ledge`) 은 보기 어려워 `figures/_superseded/` 로 내렸다 (그 README 에 이유와 쓸 데).

### 3. 토큰별 surprise 차 GIF (2026-10-07) — `figures/token_surprise/`

채점 차 s_imp − s_pos 를 미래 토큰 (8 × 16 × 16) 으로 풀어 원본 위에 겹쳤다 (ledge 3 · vanish 6 조건). 토큰 평균이 채점과 같음을 검증했다.
**쌍 차이의 대부분은 물체 토큰 (3–6 %) 이 아니라 배경 토큰에서 온다** — ledge 배경 −0.0079 vs 물체 −0.0024. 정지 · 가림은 물체 자리에서도
|Δ| 가 배경과 같다 (p 가 두 미래에서 똑같이 멀다). 자세한 표 · 그리는 규칙 · 검증은 `figures/token_surprise/README.md`.

### 4. 포물선의 미래 다섯 — p 는 어느 미래를 만드나 (2026-10-07 20:36 재렌더판, `gravity_realistic`)

데이터 `IntPhysGen_gravity_realistic` (720 clip / 144 block, 가림 없음, **PNG 전부 재렌더**). block = 과거 하나 (공이 포물선으로 난다) + 미래 다섯:
arc (포물선 계속, 가능) · rise (포물선을 문맥 끝 높이에서 위아래로 뒤집음) · float (연결 지점 f48 높이 유지, 수평만) · line (arc 와 같은 끝점까지 직선) ·
**stop (문맥 끝 f45 자리에서 멈춤, 새로 추가)**. 조건 = 꼭대기 위치: `arc_pre` (f66 — 문맥은 오르는 중만) · `arc_apex` (f48 = 문맥 경계) ·
`arc_post` (f30 — 문맥이 꼭대기를 지나 내려가는 중). 생성 로그 `UnrealEngine/gravreal_219837.log`: g 120 cm/s², 문맥 끝 수직 속도 ±135 cm/s.
문맥 픽셀 동일 (감사 144 × 4, mismatch 0) 이라 `scoring.pairing=cross` = 문맥 일치 쌍 전부. 고른 미래 = argmin_k s_k (chance 20 %).
수치 `exp_results/gravity_retrieval.{md,json}` (summary.json 과 쌍 정확도 일치 검증). 토큰 지도 검증 1 · 2 통과 (상대 오차 2e-7, 판정 불일치 0 / 576).
궤적은 렌더 프레임 위에 겹쳐 대조했다 (metadata 점 = 공 위치).

| 조건 | arc (가능) | rise | float | line | stop |
|---|---:|---:|---:|---:|---:|
| `arc_pre` (오르는 중) | 10.4 | 4.2 | **83.3** | 2.1 | 0 |
| `arc_apex` (꼭대기) | **62.5** | 0 | 33.3 | 2.1 | 2.1 |
| `arc_post` (내려가는 중) | **85.4** | 0 | 14.6 | 0 | 0 |
| 전체 (144) | 52.8 | 1.4 | 43.8 | 1.4 | 0.7 |

**위치 기준선 — 그 규칙으로 미래를 만들면 무엇을 고를까** (block 수 / 48, `gravity_retrieval.md` §5):

| 조건 | 문맥 끝 수직 속도 (px/샘플, − = 위) | 등속 이어 가기 | 높이 유지 | 그 자리 멈춤 | **p** |
|---|---:|---|---|---|---|
| `arc_pre` | −6.0 | arc 48 | float 48 | stop 48 | **float 40** · arc 5 · rise 2 · line 1 |
| `arc_apex` | −1.2 | rise 48 | float 48 | stop 48 | **arc 30** · float 16 · line 1 · stop 1 |
| `arc_post` | +3.6 | float 48 | float 48 | stop 48 | **arc 41** · float 7 |

- ~~**오르는 중이면 높이를 유지한다** — … 위로 가는 속도를 잇지 않는다~~ **정정 (2026-10-08, 튜블릿별 분해)** — 합으로는 float 83 % 이지만
  **앞 4 튜블릿 (t0–t3, 꼭대기 f66 까지) 은 arc 를 따라간다** (물체 토큰 기준 92–100 % block 이 arc 에 가장 가깝다). 꼭대기 뒤 (t4–t6) 에 **내려오지 않고**
  (rise / float 쪽), 마지막 t7 은 **stop 에 가장 가깝다 (100 %)**. float 은 그 사이의 절충이라 합에서 이긴다 — arc 와 float 은 이 조건에서 위치로 평균 9 px 밖에
  안 떨어져 (arc 가 15 px 올랐다가 11 px 아래로 내려오는 정도) 뒤쪽 튜블릿이 승부를 낸다. **위로 가는 속도는 잇는다 · 꼭대기에서 돌아 내려오지 않는다** 가 맞는 문장이다
- **내려가는 중이면 포물선을 고른다** (85 %) — 등속 기준선 (float) 보다 더 많이 떨어지는 미래다. 같은 끝점의 직선 (line) 은 0 %
- 꼭대기 (수직 속도 0) 는 arc 62.5 / float 33.3 으로 갈린다
- **튜블릿별 (`gravity_retrieval.md` §6)** — 세 조건 모두 **처음 1–4 튜블릿은 진짜 운동 (arc) 을 따라가고, 그 뒤는 진짜 운동에서 벗어난다.**
  내려가는 중 (post) 도 t0–t1 은 arc 100 % 지만 t3 부터 물체 토큰이 stop (문맥 끝 자리) 쪽으로 간다 (94 %). 합에서 arc 가 이기는 건 이 조건에서 arc 가 다른 미래와 크게 떨어져 (float 과 66 px)
  앞 튜블릿의 일치가 많이 쌓이기 때문이다. 마지막 튜블릿 t7 은 **세 조건 모두 stop 에 가장 가깝다** (전체 토큰 100 %)
  — §5-5 거리 한계 ("p 의 물체다운 토큰은 마지막 관측에서 2–3 칸 안에서만") 와 같은 방향
- **⚠️ 고른 미래 (§1 표) 는 공 자리의 판단과 다를 수 있다 (2026-10-08, `gravity_retrieval.md` §7)** — arc vs float 를 공 자리 · 그림자 · 나머지 (두 미래 픽셀이 거의 같은 칸) 로 나누면:

  | 조건 | 전체 (= 채점) | 공 자리 | 그림자 | 나머지 (픽셀 ≤ 16) |
  |---|---:|---:|---:|---:|
  | 오르는 중 | arc 12 % | **arc 75 %** | 62 % | **0 %** |
  | 꼭대기 | arc 67 % | **arc 2 %** | 35 % | **85 %** |
  | 내려가는 중 | arc 85 % | 81 % | 79 % | 77 % |

  (block 중 p 가 arc 쪽인 비율.) **꼭대기에서 arc 가 이기는 것은 공 자리가 아니라 픽셀이 같은 칸 때문이다** — 공 자리에서는 98 % block 이 float (높이 유지) 쪽이다.
  그래서 "꼭대기에서 내려오는 걸 본 적 없는데 내려온다고 예측한다" 는 **공 자리 근거로는 안 선다**. 오르는 중은 거꾸로 공 자리가 arc, 나머지가 float 를 고른다.
  그림자 (공 밖 픽셀 차 > 16) 는 결정하지 않는다. 나머지 칸의 차는 픽셀이 아니라 target encoder 의 전역 attention 이 만든 h 차이다 (§3 과 같은 기제).
  ⚠️ vanish · ledge (§3) 에서는 배경 부호가 물체와 96–100 % 같았는데 여기 (오르는 중 · 꼭대기) 는 **반대로 간다** — 두 결과를 지우지 말고 조건 차이로 읽는다 (가림 없는 큰 수직 운동 vs 물체 유무 · 낙하)
- ⚠️ t7 의 stop 100 % 는 predictor 의 **마지막 시간 위치 자체의 성질** 일 수도 있다 (마지막 튜블릿이 문맥 끝 상태로 되돌아가는 경향). 다른 세트에서 튜블릿별로 확인하지 않았다.
  튜블릿 하나의 argmin 은 차이가 작아 흔들린다 — 한 칸 숫자보다 줄의 흐름으로 읽을 것
- **stop (멈춤) 은 거의 안 고른다** (0.7 %) — p 는 수평 운동은 늘 잇는다
- 쌍 정확도 전체 85.2 % = vs rise 93.1 · vs float 54.9 · vs line 94.4 · vs stop 98.6 의 평균. float 만 반반이다
- ⚠️ **옛 판 (미래 넷, 아래) 과 꼭대기 결과가 다르다** — 옛 판 apex float 94 % ↔ 새 판 float 33 %. 데이터 설계 (g 170 → 120, 꼭대기 f57/f42 → f66/f30,
  문맥 끝 속도) 가 같이 바뀌어 무엇이 차이를 만들었는지 이 둘로는 못 가른다. 두 판을 섞어 읽지 말 것
- ⚠️ float 은 화면 변화가 작은 미래다 — "높이 유지" 와 "덜 변하는 미래가 L1 로 싸다" 를 채점만으로 못 가른다. 다만 stop (변화가 더 작다) 을 0.7 % 만 고르므로 "가장 덜 변하는 미래" 만으로는 설명되지 않는다
- ⚠️ 생성 로그 pixel check **FAIL** (렌더 위치가 계획과 최대 6.3 / 6.4 / 16.0 px). 프레임 위 대조에서는 공과 점이 겹쳤다. `dataset.json` 의 "g 101" 문구는 로그의 g 120 과 다르다

**그림 `figures/fig_gravity_choice`** — 캡션:
> Which future the V-JEPA 2 predictor output is closest to (lowest L1 surprise among five futures sharing one context; 48 blocks per panel,
> chance 20 %, dashed). (a) The context shows the ball rising, (b) the context ends at the apex, (c) the context shows it descending.
> Below each bar: that future in one representative left-to-right block (grey dots: context seen by the model; coloured dots: the future;
> dotted line: end of context). Error bars: 95 % block-bootstrap CI.

**짝 그림 `figures/fig_gravity_choice_ball`** (2026-10-08) — 같은 판형, 같은 L1 을 **공 칸만** (다섯 미래 물체 자리 합집합 3×3) 평균해 고른 비율:
오르는 중 arc 71 · float 29 / 꼭대기 arc 15 · **float 83** / 내려가는 중 arc 65 · line 17 · float 15.
위 그림 (화면 전체 = 표준 채점) 과 오르는 중 · 꼭대기에서 순위가 뒤집힌다 — 두 그림은 **다른 것을 잰다** (전체 = 채점이 고르는 미래, 공 칸 = p 가 공을 그리는 미래에 가까운 쪽).
캡션: *Same as fig_gravity_choice, but the L1 distance is averaged over the ball tokens only (union of the five futures' ball positions, 3×3 tokens).*

### 4-옛판. 포물선의 미래 넷 (2026-10-07 17:40 채점) — **대체됨**

> **대체됨 (2026-10-07 20:36)** — 데이터가 미래 다섯 (`imp_stop` 추가) · 꼭대기 f66/f48/f30 · g 120 으로 다시 렌더됐다. 정본은 위 §4.
> 아래는 옛 프레임으로 낸 기록이다. 수치 `exp_results/_previous_20261007_4futures/`, 그림 `figures/_superseded/fig_gravity_choice_4futures`.


데이터 `IntPhysGen_gravity_realistic` (576 clip / 144 block, 가림 없음). block = 과거 하나 (공이 포물선으로 난다) + 미래 넷:
arc (포물선 계속, 가능) · rise (꼭대기 뒤에도 계속 올라감) · float (높이 유지, 수평으로만) · line (같은 끝점까지 직선).
조건 = 꼭대기 위치: `arc_pre` (f57, 문맥은 오르는 중만) · `arc_apex` (f48 = 문맥 경계) · `arc_post` (f42, 문맥이 꼭대기와 내려가기 시작까지).
문맥 픽셀 동일 (감사 144 × 3, mismatch 0) 이라 `scoring.pairing=cross` = 문맥 일치 쌍 전부. 고른 미래 = argmin_k s_k (chance 25 %).
수치 `exp_results/gravity_retrieval.{md,json}` (`gravity_retrieval.py`, summary.json 과 쌍 정확도 일치 검증). 토큰 지도도 뽑았다 (`token_surprise.npz`, 검증 1 · 2 통과, 상대 오차 2e-7).

| 조건 | arc (가능) | rise | float | line |
|---|---:|---:|---:|---:|
| `arc_pre` | 0 | 20.8 | **79.2** | 0 |
| `arc_apex` | 4.2 | 2.1 | **93.8** | 0 |
| `arc_post` | 37.5 | 0 | **62.5** | 0 |
| 전체 (144) | 13.9 | 7.6 | **78.5** | 0 |

- **p 가 가장 가까운 미래는 거의 늘 float (높이 유지)** — 78.5 % [72, 85]. 꼭대기가 경계에 오면 94 %
- 문맥이 오르는 중만 보여도 (arc_pre) 올라가는 미래 (rise) 는 21 % 뿐이고 진짜 포물선은 0 % — 수직 속도를 잇지 않는다.
  문맥이 내려가기 시작까지 보여야 (arc_post) 포물선이 38 % 로 나온다
- line (끝점이 포물선과 같은 직선) 은 한 번도 안 고르고 평균 순위 3.8 (가장 멀다) — 쌍 정확도 vs line 100 %
- 쌍 정확도 전체 52.5 % 는 vs line 100 · vs rise 43.8 · vs float 13.9 의 평균이다 (`gravity_retrieval.md` §2)
- ledge (§2, 0 / 120) · RollOutV2 위치 자 (p 는 선반 높이 유지) 와 같은 방향 — **받침이 없어도 (공중 비행) p 의 미래는 높이를 유지한다**
- ⚠️ float 은 문맥 끝 높이를 그대로 두는 미래라 화면 변화가 가장 작다. "높이를 유지한다" 와 "덜 변하는 미래가 L1 로 싸다" 는 이 채점만으로 못 가른다 (§2 단서와 같다)
- ~~⚠️ metadata `object_px_*_by_sample` 이 불가능 셋에서 똑같이 적혀 있다~~ **정정 (2026-10-07 18:07)** — 생성 쪽에서 metadata 를 다시 조립해 지금은 셋이 다르다
  (rise 상승 · float 높이 고정 · line 직선). **프레임은 그대로다** (PNG 시각 16:46 · 렌더 단계 `converted 0` · 같은 쌍 픽셀 차 0.34 / 2.59 / 3.46 재측정 일치) → 채점 결과 유효.
  index 를 다시 만들어도 바이트 동일, 문맥 감사 mismatch 0

**그림 `figures/fig_gravity_choice`** — 캡션:
> Which future the V-JEPA 2 predictor output is closest to (lowest L1 surprise among four futures sharing one context; 48 blocks per panel,
> chance 25 %, dashed). (a) The context shows the ball rising, (b) the context ends at the apex, (c) the context shows the start of the descent.
> Below each bar: that future in one representative left-to-right block (grey dots: context seen by the model; coloured dots: the future;
> dotted line: end of context). Error bars: 95 % block-bootstrap CI.

**위치 기준선 — 그 규칙으로 미래를 만들면 무엇을 고를까** (metadata 궤적, 미래 16 샘플 평균 위치 거리; `gravity_retrieval.md` §5):

| 조건 | 문맥 끝 수직 속도 (px/샘플, − = 위) | 등속 이어 가기 | 높이 유지 (수평만 이어 감) | **p (surprise)** |
|---|---:|---|---|---|
| `arc_pre` | −5.2 (올라가는 중) | rise 48 | float 48 | rise 10 · **float 38** |
| `arc_apex` | −1.7 | rise 48 | float 48 | arc 2 · rise 1 · **float 45** |
| `arc_post` | +0.6 (막 내려가기 시작) | float 48 | float 48 | **arc 18** · float 30 |

- 오르는 중 (pre · apex) 에 **등속으로 이었다면 전부 rise 를 골랐을 것**인데 p 는 float — **수평 속도는 잇고 수직 속도는 버린다** (높이 유지)
- 내려가기 시작 (post) 은 두 기준선 모두 float 인데 p 는 38 % 를 진짜 포물선에 둔다 — 문맥에 내려가는 모습이 보이면 기준선보다 물리 쪽으로 간다
- CLAUDE.md 의 "전이 ◐: 방향 · 속도는 이어 가고 가속도는 반영 안 함" 을 **수직 방향에서는 속도도 잇지 않는다** 로 좁힌다 (이 세트 기준, 단서 위와 같음)
- ⚠️ metadata 는 계획 위치다. 생성 로그 (`UnrealEngine/gravreal_219796.log`) 의 pixel check 가 **FAIL** — 렌더 위치가 계획과 최대 6.5 (apex · post) / 13.7 px (pre),
  이동 거리 기울기 0.88–0.96. 미래끼리의 위치 차가 수십 px 라 기준선 판정은 안 바뀐다고 봤다 (렌더 위치로 다시 재지는 않았다)
- ⚠️ `dataset.json` 은 "g 는 ledge 와 같은 101 cm/s²" 라 적었는데 생성 로그 · provenance 는 **g 170** (`arc_az −170`). 설명 문구가 옛 값이다 (생성 쪽 확인 필요)

## 읽는 법 · 단서

- **A/B 방향을 항상 같이 본다** (CLAUDE.md §1-7). vanish 전체 50 % 는 A 0 / B 100 의 평균일 수 있다
- ledge 의 낙하는 문맥에 없는 사건 (선반 끝) 이다 — ~~state evolution 네 능력 정의 밖 (common sense) 으로 읽는다~~ → 10-08 번복: **3b 사건** 으로 읽는다 (정본 §2-3 iii 의 단서와 함께)
- 원본 v11 대조는 같은 `surprise_c16t32` 실행 (`IntPhysGenV11/exp_results/surprise_c16t32__v11_vith`) 의 **vanish 행만** (2,688 쌍)

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
# index + 문맥 무결성 감사 (GPU 불필요)
$P z_research/scripts/data/build_intphysgen_index.py --frames-root /local_datasets/world/world_analysis/IntPhysGen_v11_realistic --out data_csv/intphysgen_v11_realistic
$P z_research/scripts/data/build_intphysgen_index.py --frames-root /local_datasets/world/world_analysis/IntPhysGen_v11_realistic_ledge --out data_csv/intphysgen_v11_realistic_ledge
# 채점 (config = configs/protocols/surprise_c16t32.yaml + datasets.md 의 v11_realistic{,_ledge} + models.md 의 vith)
GPUS=8 bash z_research/scripts/run.sh surprise_c16t32 v11_realistic vith
GPUS=8 bash z_research/scripts/run.sh surprise_c16t32 v11_realistic_ledge vith
# 표 (summary.json 과 overall 일치 검증 포함) → exp_results/tables.{md,json}
$P z_research/scripts/analysis/v11_realistic_tables.py
# 포물선 미래 넷 (cross = 문맥 일치 쌍 전부) → exp_results/gravity_retrieval.{md,json}
$P z_research/scripts/data/build_intphysgen_index.py --frames-root /local_datasets/world/world_analysis/IntPhysGen_gravity_realistic --out data_csv/intphysgen_gravity_realistic
SET="scoring.pairing=cross" GPUS=4 bash z_research/scripts/run.sh surprise_c16t32 gravity_realistic vith
$P z_research/scripts/analysis/token_surprise_maps.py --run z_research/v11_realistic/exp_results/surprise_c16t32__gravity_realistic_vith --gpus 4
$P z_research/scripts/analysis/gravity_retrieval.py
# 그림 → figures/fig_rl_summary.{pdf,png}  (옛 세 장은 --all)
$P z_research/scripts/figures/plot_v11_realistic.py
```

vll3 실행 (2026-10-06): 프레임은 vll4 렌더 → vll5 `/local_datasets` → NFS tar (`/data/hyuntak/tmp_xfer/v11_realistic_both.tar`) → vll3 `/local_datasets` 로 풀어 썼다.
로그 `exp_results/logs/`. 채점 시간: realistic 3.5 분 · ledge 1.2 분 (GPU 8 장, 모델 로드 포함).
