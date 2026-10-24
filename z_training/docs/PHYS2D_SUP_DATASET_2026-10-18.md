# phys2d_sup — 2D 중력 · 영속 · 고체성 학습셋 + 프레임별 상태 라벨 (2026-10-18)

> **정본은 §V `phys2d_sup_v2`** (vast, 같은 날 저녁). §0 이하는 vll5 의 `phys2d_sup_v1` 기록이다 —
> ⚠️ v1 의 창 인덱스 (`phys2d_sup_v1_future`, 67,334 창) 는 **문맥에 전환이 들어 있는 창이 56.6 %** 다 (§V-1). 다시 만들 것.

## V. phys2d_sup_v2 (vast, 2026-10-18 저녁) — 전환은 예측 구간에, 가림은 어디든

**사용자 결정** (이 판의 기준):
- "충돌이면 충돌 직전까지만 context 에 보여주고 그 다음 충돌하고 멈추거나 그런 걸 예측해야지. 아니면 딱 충돌 시점까지만 context 로 주던가"
- "permanence 는 context 에서 가려져도 되고 다양하게 여러 번 가려져도 돼 한 clip 내부에서" · "가려지는 게 화면 밖으로 나가는 거면 조금 그럴 수도"
- 쓰임: predictor 출력 (T·N, D) → attention pooling (T, D) → 장마다 속도 (cos, sin, 크기) · 위치 (x, y) · 정체 (모양 × 색 48 또는 없음) head

**위치**: `/workspace/dataset/training_data/phys2d_sup_v2/{train,val}` — train **38,343** clip (20 시나리오 × 2,000 시도, 통과 95.9 %, 4.9 GB) · val **1,156** (seed 2, 겹침 0, 149 MB).
생성 16 분 (CPU 240 프로세스, nice 19 — GPU 학습과 같이 돌았다).

### V-1. 왜 다시 만들었나 — v1 창 인덱스 실측 (main 코드로 시나리오당 10 편 생성, 솎기 전 창 2,987)

| 문맥 안에 있는 것 | 창 비율 |
|---|---:|
| json 사건 아무거나 | 56.6 % |
| 충돌 dv ≥ 60 (방해 물체 제외) | 41.9 % |
| 예측 구간에 화면 밖 물체가 처음 보임 | 34.1 % |

원인: `build_phys2d_window_index.py` 가 사건을 `boundary.tsv` 에서만 읽었는데 거기엔 **행으로 고른 사건만** (충돌 영상당 ≤ 2) 있다.
그리고 생성기 boundary 행 자체가 `b = 사건 뒤 첫 장 + 1 + j` 로 전환을 **문맥 끝**에 넣었다 (vast `phys2d_v2` 와 같은 관례 — 결정과 반대).

### V-2. 바꾼 것 (`gen_phys2d_sup.py` · `build_phys2d_window_index.py` · `app/vjepa_frozen/var_window.py`)

- **경계** — post = 전환 뒤 첫 장 (전환은 post−1 렌더 뒤 substep 에서 일어난다). 예측 첫 장 b = post + j, j ∈ {−2, −1, 0, +1} 균등:
  문맥이 전환 2 · 1 · 0 장 전에서 끝나거나 (j ≤ 0), 문맥 마지막 장이 딱 전환 순간 (j = +1). 실측 b − post 분포 41,344 / 42,570 / 42,983 / 48,315.
- **`ctx_start` 열** — 이 행보다 앞선 **충돌 (dv ≥ 60, 방해 물체 제외) · 받침 이탈** 의 post. 로더 (var_window boundary) 가 문맥 C ≤ (b − ctx_start)/s 인 행만 쓴다
  (열이 없는 manifest 는 예전과 같다). **가림 · 다시 보임은 문맥을 막지 않는다** (결정대로). 깨끗한 문맥 길이 ≥ 4 / 8 / 16 / 32 장: 100 / 74 / 54 / 27 % 행.
- **예측 구간 끝 (`n_frames`)** — 화면 밖에서 생긴 물체가 **처음 보이는 장** 직전에서도 끊는다 (생성 시각만 보던 것은 생성이 문맥 안이면 놓쳤다).
- **창 인덱스** — json `events` 전체를 읽는다. ① 문맥에 충돌 · 이탈 전환 없음 (문맥 마지막 장이 딱 전환 순간인 것은 허용) ② 예측 구간에 전환 (충돌 · 이탈 · 가림 · 다시 보임) ≥ 1
  ③ 예측 구간에 화면 밖 물체 첫 등장 없음. 산출 `<split>/windows_T32C16s1.csv`.
- **spawn 여부를 재시도와 무관하게** (v1: 설계 50 % → rejection 뒤 76.5 %). 그대로 고정하면 15 % 가 200 회 안에 통과 못 하고 실패가 모양과 얽혀
  (원이 굴러서 잘 통과) 주 물체 원이 18.9 % 가 됐다 (val 시험판, 지웠다) → **100 회째 시도부터는 다시 뽑는다** (`SPAWN_FIXED_ATTEMPTS`).

### V-3. 행 검사 (train 전수 175,212 행, `z_training/data/verify_phys2d_sup_rows.py <split 폴더>`)
- 문맥 [ctx_start, b) 안에 충돌 · 이탈 전환: **0 행**. 사건이 경계 밖: 0 행.
- 예측 구간에 화면 밖 물체 첫 보임: 171 행 (0.1 %) — 전부 문맥 마지막 장에 이미 2 % 미만으로 비친 경우 (검사는 전체 면적 비, 생성기는 화면 안 면적 비로 잰다). 위반 아님.
- 사라짐의 원인: **가림막 99,422 vs 화면 밖 3,646 (3.5 %)**. 본 물체가 화면 밖에 있는 물체·장 0.9 %.

### V-4. 편향 점검 (`train/audit.md`)

| 항목 | v2 | v1 (vll5) |
|---|---|---|
| 주 물체 모양 · 색 | 16.4–17.3 % · 12.4–12.6 % (48 칸 777–833 편) | 16.6–16.7 · 12.4–12.5 % |
| 전체 물체 모양 | 원 22.1 % (나머지 14.2–16.9) | 원 20.4 % |
| g 5 등분 | 19.2–20.7 % | 19.8–20.3 % |
| 들어오는 물체 있는 클립 | **65.1 %** (설계 50) | 76.5 % |
| 가림 구간 / 영상 | 평균 2.10 · 0 / 1 / 2 / 3+ = 29 / 20 / 21 / **31 %** · 한 물체 2 번 이상 38.2 % | 24 % (3+) · 29.9 % |
| 가려진 동안 움직임 | 86.1 % (이동 중앙 18 px) | 87.6 % |
| boundary 행 | 175,111 — 가림 37.0 · 다시 보임 32.3 · 충돌 22.8 · 이탈 8.0 % | 가림 33 · 다시 보임 29 · 충돌 29 · 이탈 10 |
| 예측 앞 8 장 복사 / 등속으로 풀리는 행 (< 0.5 반폭) | 8.2 / 17.6 % (충돌 2.3 / 2.8 · 이탈 0.7 / 1.4 · 가림 10.4 / 21.8 · 다시 보임 11.6 / 27.4) | 7.1 / 13.5 % |
| 창 (T32 C16, clip 당 ≤ 4) | **142,182** (clip 95.1 %) · 첫 전환 충돌 58 · 가림 22 · 다시 보임 14 · 이탈 6 % | 67,334 (규칙 깨짐) |
| 라벨 정합 | 위치 중앙 0.19 px · p99 0.85 · 속도 p99 0.68 px/장 | 같음 |
| 누수 | seed 중복 0 · val–train 0 | 같음 |

⚠️ **남은 치우침 — 알고 쓸 것**
1. **rejection 이 세다** — 재시도 평균 container 73.7 · stack_topple 73.3 · drop_behind 69.5 … (v1 보다 많다). 거의 전부 "too short" (64 장 전에 움직임이 끝남, 시도 단위 집계 5,258 / 7,100).
   오래 움직이는 변수 (반발 · 속도) 가 골라진다. 실패 1,657 편 (stack_topple 22 · container 19.5 %) → 시나리오 편수 1,636–2,000.
2. **가림 · 다시 보임 행은 등속으로 꽤 풀린다** (21.8 / 27.4 %) — 가려진 동안 등속으로 굴러가는 경우. 영속 supervision 으로는 맞는 신호지만 "외삽으로 못 푼다" 는 충돌 · 이탈 행에만 성립.
3. 가림 시작 시각이 앞쪽 (5 등분 35 / 22 / 17 / 17 / 9 %), y 아래 두 칸 71 %, 가림막 없는 영상 (16.5 %) 은 가림 0 % — v1 과 같은 성격.
4. 창 인덱스의 첫 전환은 충돌이 58 % (창은 영상 어디서나 시작하므로 잦은 충돌이 많이 걸린다). 사건 종류 균형이 필요하면 boundary 행 (`boundary.tsv`, var_window) 을 쓴다.

⚠️ **head 설계와의 관계** — 영상당 물체 평균 2.9 개 (1 개 12 %). (T, D) 하나로 위치 · 속도 · 정체를 내면 **물체가 여럿인 장에서 정답이 하나로 정해지지 않는다.**
라벨은 물체마다 다 있다 (`window_labels`, max_obj 10). 쓰는 쪽에서 (a) 주 물체 하나 (`role == main`, 정체 층화된 쪽) 를 정답으로 정하거나
(b) 정체 48 개 query 로 (T, 48, D) — vll5 의 `state_head` 방식 — 로 풀어야 한다. 어느 쪽인지는 정하지 않았다.

**학습 레지스트리 · 상태 라벨 로더는 vast 에 없다** (`phys2d_sup` 타입 · `Phys2DSupDataset` 은 main/vll5 에만). 데이터를 vll5 로 옮기거나 로더를 이식해야 쓴다.

### V-재현
```bash
cd /workspace/world_model/vjepa2
R=/workspace/dataset/training_data/phys2d_sup_v2
nice -n 19 /venv/vjepa/bin/python -W ignore z_training/data/gen_phys2d_sup.py --per-scenario 60   --seed 2 --workers 240 --preview-n 0 --out $R/val     # 0.6 분
nice -n 19 /venv/vjepa/bin/python -W ignore z_training/data/gen_phys2d_sup.py --per-scenario 2000 --seed 1 --workers 240 --preview-n 3 --debug-overlay --out $R/train   # 16 분
nice -n 19 /venv/vjepa/bin/python -W ignore z_training/data/audit_phys2d_sup.py --root $R/train --val $R/val --workers 100 > $R/train/audit.md
/venv/vjepa/bin/python z_training/data/build_phys2d_window_index.py --root $R/train --max-per-clip 4   # → $R/train/windows_T32C16s1.csv
/venv/vjepa/bin/python z_training/data/build_phys2d_window_index.py --root $R/val   --max-per-clip 2
```
var_window boundary 로 쓸 때: `boundary: {manifest: $R/train/boundary.tsv, ...}` — `ctx_start` 열을 로더가 읽는다.

---

# (기록) phys2d_sup_v1 — vll5

> 생성기 `z_training/data/gen_phys2d_sup.py` · 편향 점검 `z_training/data/audit_phys2d_sup.py` · 라벨 읽기 `z_training/data/phys2d_sup_labels.py`.
> vast 브랜치 `gen_phys2d.py` (174bf30, `phys2d_v2`) 를 포크했다 — 물리 · 렌더 · 사건 행 관례는 같고, **라벨 · 정체 · 여러 번 가림 · 치우침 수정**이 다르다.
> 수치는 전부 산출물 (`clips/*/*.json`, `*_states.npz`, `*_seg.npz`) 에서 점검 스크립트로 다시 계산했다 (`train/audit.md`, `val/audit.md`).
> **학습은 아직 안 돌렸다.** 이 문서는 데이터의 구성 · 라벨 · 편향 기록이다.

## 0. 요약

- **목적** (사용자 2026-10-18): predictor 학습에 **매 장 상태 supervision** — 물체 정체 (모양 × 색) · 위치 (x, y) · 속도 (vx, vy) — 을 주고,
  영속 (여러 번 가려졌다 나옴) · 중력 (자유 낙하 · 받침 끝 낙하 · 포물선 · 꼭대기에서 되돌아옴 · 진자 · 용수철 · 경사) · 고체성 (벽 · 물체 · 선반 · 통을 통과 못 함) 이 데이터에 고르게 들어가게.
- **규모**: train **19,953 clip** (20 시나리오 × 1,000 시도, 통과 99.8 %) · val **996 clip** (seed 2, 겹침 0). 2.6 GB + 129 MB.
- **생성**: vll5 CPU 40 프로세스, train 22 분 (클립당 CPU 약 2.6 초). pymunk 7.3.1 (vjepa2 env 에 설치).
- **위치**: `/local_datasets/world/training_data/phys2d_sup_v1/{train,val}` (→ `/data2/local_datasets/world/training_data/`, 로컬 xfs).
- **학습 레지스트리** (`configs/training/datasets.md`): `phys2d_sup_v1_future` (67,334 창) · `phys2d_sup_v1_future_val` (1,701 창) · `phys2d_sup_v1_clips` (19,953 clip).

## 1. 무엇인가

- **가능 (plausible) 영상만.** pymunk 강체 물리 + cv2 렌더 (2 배 슈퍼샘플), 256², 1 장 = 시뮬 1/12 초, mp4 는 4 fps 로 적는다
  (학습 로더 fps 4 · fstp 1 이 렌더한 장을 하나도 안 건너뛴다 — vast phys2d 와 같은 관례). 길이 64–128 장 (중앙 106).
- 클립마다 무작위: g (350–900 px/s², **클립마다 고정 · 균등**) · 크기 · 반발 · 마찰 · 배경 5 종 · 지면 높이 · 좌우 반전 · 시작 위상 · 옆벽 (75 %) ·
  방해 물체 (25 %) · 화면 밖에서 들어오는 물체 (50 %) · 궤적과 무관한 고정 가림막 0–3 개 · 떠다니는 가림막 (12 %) · 기운 선반 (30 %).

| 범주 | 시나리오 (각 1,000 시도) |
|---|---|
| 중력 7 | `drop_bounce` (자유 낙하 · 튕김) · `projectile` (포물선) · **`toss_up`** (위로 던져 꼭대기에서 되돌아옴, 신규) · `pendulum` (단 · 이중) · `spring` · `ramp_roll` · `ledge_fall` (받침 끝 낙하) |
| 영속 6 | `occ_pass` (궤적 위 가림막 **1–3 개**) · **`osc_behind`** (진자 · 용수철 · 왕복 공이 같은 가림막 뒤를 **2 번 이상** 드나듦, 신규) · `moving_screen` (오가는 가림막) · `cup_cover` · `drop_behind` · `tunnel` |
| 고체성 6 | `shelf_land` · `wall_bounce` · `collide` · `stack_topple` · `container` · `occ_wall` (가려진 벽에 튕김) |
| 섞음 1 | **`sandbox`** (신규) — 지형 (경사 · 선반 · 벽 · 통) · 장치 (진자 · 용수철 · 가림막 · 움직이는 가림막) 2–4 개 + 물체 2–4 개 무작위 조합, 사건 종류 2 가지 이상 |

## 2. 라벨 — `<clip>_states.npz` (장 T × 물체 n), `<clip>_seg.npz`, `<clip>.json`

좌표: 화면 px (원점 왼쪽 위, y 아래, 픽셀 j = [j, j+1)). 시간 단위 = 렌더 1 장 (1/12 초). 좌우 반전 클립은 라벨도 반전돼 있다 (x → 256 − x, vx · 각도 · 각속도 부호 반전).

| 키 | 모양 | 뜻 |
|---|---|---|
| `pos` | (T,n,2) | 질량 중심 (px). **가려져 있어도 참값** (영속 supervision) |
| `vel` | (T,n,2) | 순간 속도 (px/장). 방향 = atan2(vy, vx), 크기 = ‖v‖ — **각을 직접 회귀하지 말고 (vx, vy) 로** (±π 불연속 · 정지 시 정의 안 됨) |
| `acc` | (T,n,2) | 가속도 (px/장², 속도의 중앙 차분 — 충돌 장에서 크다). 자유 낙하 중 ≈ (0, g/144) |
| `angle` · `omega` | (T,n) | 각도 (rad) · 각속도 (rad/장) |
| `bbox` | (T,n,4) | amodal 상자 x0 y0 x1 y1 |
| `present` | (T,n) | 시뮬에 있음. 화면 밖에서 들어오는 물체는 들어오기 전 False, 그 칸 연속 값은 NaN |
| `vis_frac` · `inframe_frac` | (T,n) | 전체 면적 중 화면에 보이는 비율 · 화면 안 비율 (가림 무시) |
| `hidden` | (T,n) | 화면 안인데 완전히 가려짐 (보이는 비율 < 2 %) |
| `seen` | (T,n) | 지금까지 30 % 넘게 보인 적 있음 — 한 번도 안 본 물체의 위치 손실을 끄는 데 쓴다 |
| `impact` | (T,n) | 그 장에 시작된 충돌의 속도 변화 (px/장) |
| `identity` · `shape_id` · `color_id` | (n,) | 정체 = shape_id × 8 + color_id. 모양 (circle · square · rectangle · triangle · pentagon · hexagon) × 색 (red · orange · yellow · green · cyan · blue · purple · magenta). **한 영상 안 정체는 모두 다르다** |
| `size` · `area` | (n,) | 크기 (반지름 상당 px) · 면적. 여섯 모양의 면적을 원의 ±8 % 로 맞춰 크기가 모양과 얽히지 않는다 |
| `occluder_rect` | (T,m,4) | 가림막 · 컵 상자 |
| `seg` (`_seg.npz`) | (T,256,256) u8 | 보이는 픽셀 인스턴스 세그 — 0 배경 · 지형, i+1 물체 i, 255 가림막 · 컵 |

json: 물체 (역할 main · bob · spawn · distractor · stack, 질량 · 반발 · 마찰) · 가림막 (출처 scenario · auto · extra · drift) · 정적 지형 다각형 · 줄 ·
**사건 목록** (`events`: impact · leave · occlude · reappear, 장 번호) · boundary 행 · 검사 지표.

**읽기**: `window_labels(load_clip(mp4), start, n_frames, stride, tubelet=2, max_obj=10)` → 튜블릿마다 (`reduce=last` 기본) 라벨 + 손실 마스크
`mask_state` (본 적 있고 화면 안 — 가려져 있어도 위치 · 속도를 건다) · `mask_visible` (보일 때만) · `mask_identity`. 영상당 물체 최대 9 (평균 3.24).

### 라벨 정합 (H, train 표본 998 clip)
- 위치 라벨 vs 완전히 보이는 물체의 세그 무게중심: **중앙 0.19 px · p99 0.82 px** · max 2.4 (n = 52,746 물체·장, 반전 클립 포함)
- 속도 라벨 vs 위치 중앙 차분 (충돌 앞뒤 장 제외): 중앙 0.000 · p95 0.28 · p99 0.73 px/장 (차이는 진자 · 용수철의 곡률)

## 3. 편향 점검 (train 19,953 clip, `train/audit.md`)

### 3-1. 균형이 맞는 것 ✅

| 항목 | 측정 |
|---|---|
| 주 물체 정체 (층화) | 48 칸 각 410–419 편 · 모양 6 개 각 16.6–16.7 % · 색 8 개 각 12.4–12.5 % |
| 전체 물체 정체 | 48 칸 max/min 1.48 · 모양 14.4–20.4 % (원이 가장 많다 — 아래 3-2 #3) |
| 정체 ↔ 사건 | 충돌 · 이탈 · 가림 을 겪은 물체의 모양 · 색 비율이 전체 비율과 ±3 %p 안 (색은 ±0.5 %p) |
| g | 5 등분 칸 19.8–20.3 % (vast v2 는 23.8 → 17.5 %) |
| 반발 계수 | 모양별 중앙 0.63–0.67 |
| 좌우 | 반전 50.1 % · 주 물체 순 수평 이동 오른쪽 46.3 / 왼쪽 47.0 % · x 분포 가운데 6 칸 13.6–14.7 % (양 끝 칸은 옆벽 두께만큼 7.5 %) |
| 배경 | 5 종 각 19.3–20.6 % |
| 영속 (가림 구간: 본 뒤 완전히 가려졌다 다시 보임, 2 장 이상) | **34,526 구간** · 영상당 0 / 1 / 2 / 3+ = 37 / 21 / 18 / **24 %** · 한 물체가 **2 번 이상** 가려졌다 나온 영상 29.9 % (3 번 이상 13.5 %) · 가려진 길이 중앙 5 장 (2–3: 34 · 4–8: 38 · 9–16: 21 · >16: 7 %) · **가려진 동안 움직인 구간 87.6 %** (이동 중앙 18 px) |
| 가림의 범주 | 가림 구간 있는 영상: permanence 98.7 · mixed 78.3 · solidity 51.2 · gravity 39.7 % — 영속 시나리오 밖에서도 가림이 생긴다 (extra · drift 가림막) |
| 복사 · 등속 지름길 (boundary 행 예측 앞 8 장) | 복사 오차 < 0.5 반폭인 행 7.1 % · 등속 외삽 13.5 % (vast v2: 3.3 / 10.9 %). 이탈 행은 0.4 / 1.8 % |
| 누수 | seed 중복 0 · val–train 겹침 0 |

### 3-2. 치우침 — 알고 쓸 것 ⚠️

| # | 치우침 | 측정 | 위험 | 대응 |
|---|---|---|---|---|
| 1 | **들어오는 물체** | 들어오는 물체 있는 클립 설계 50 % → **76.5 %**, 전체 물체의 38.5 % 가 spawn. 입구는 대부분 위 | 낙하하는 물체가 많다 · "새 물체는 위에서" prior | rejection 이 운동을 더해 주는 클립을 고른다 (g · drift 와 같은 기제). 등장 장은 `seen=0` · boundary 예측 구간 제외. 고치려면 spawn 여부도 시도 번호와 무관하게 정한다 (재생성 22 분) |
| 2 | **세로 위치 = 바닥 근처** | y 아래 두 칸 71 % · 위 세 칸 7 % | 바닥 위 굴림 · 튕김이 대부분 | 중력이면 자연스럽다 (vast v2 74 %) |
| 3 | **영속 · 고체성에서 원이 조금 많다** | 전체 물체 원 20.4 % (균등 16.7) · 가림 구간 물체 원 24.5 % · 고체성은 상자 고정 (stack_topple) 으로 square · rectangle 20 % | 굴러서 멀리 가는 원이 가림막 뒤를 더 자주 지난다 | 주 물체는 층화로 정확히 균등. 원 쏠림은 물리 (구름) 에서 온다 |
| 4 | **가림 시작 시각이 앞쪽** | 영상 5 등분 37 / 21 / 16 / 16 / 9 % | 영상 앞쪽에 가림이 많다 | 학습 창을 영상 전체에서 뽑으면 창 안 위치는 덜 치우친다. 창 인덱스는 사건 장 기준이라 영향 작음 |
| 5 | **가림막 존재 → 가림** | 가림막 없는 영상 (22 %) 의 가림 0 % · 무관한 extra 가림막 영상 77 % 에 가림 · occ_pass 미끼 설계 25 % → 15.7 % | 가림막이 보이면 가림이 온다 | 실세계 통계와 같은 방향 (가림막 없이는 못 가린다) |
| 6 | **사건 종류 (json events)** | 충돌 77 % · 이탈 9 % · 가림 7 % · 다시 보임 7 % | — | boundary 행은 가림 33 · 다시 보임 29 · 충돌 29 · 이탈 10 % 로 맞췄다 (영상당 충돌 행 ≤ 2) |
| 7 | **물체는 채도 높은 8 색, 지형 · 가림막 · 배경은 채도 낮은 색** | 설계 | "채도 = 물체" 단서 | 색 정체가 늘 읽히게 하려는 의도된 설계 |
| 8 | **rejection** | 재시도 평균 stack_topple 36.9 · occ_wall 27.3 · … moving_screen 0.6, 실패 47 편 (stack_topple 19 · occ_wall 16) | 오래 움직이는 변수만 남는다 | 실패 편은 `failed.tsv` |

### 3-3. 이 점검이 못 본 것 (단서)

- **2D 그림 ↔ 실사 encoder 도메인 차이는 재지 않았다** (vast 문서와 같은 단서). 학습 전에 frozen encoder 로 이 영상에서 위치 · 정체 probe 와 잠재 복사 기준선을 먼저 재는 것이 싸다.
- 복사 · 외삽 오차는 물체 중심 좌표로 쟀다. 픽셀 · 잠재 공간 복사 오차는 따로 재야 한다.
- 정체 라벨은 렌더 색 (원은 반쪽 음영, 다각형은 회전 표시 점) 에서 팔레트 색이 지배적이라는 것만 확인했다 (`verify` 표본) — encoder 가 48 정체를 가르는지는 안 쟀다.

## 4. 쓰는 법

```yaml
data:
  datasets: [phys2d_sup_v1_future]       # 또는 phys2d_sup_v1_clips (video_csv, frame_step 1)
```
학습 로더 (`video_window`) 는 영상만 읽는다. 상태 supervision 은 로더가 같은 (path, start, stride) 로 `window_labels` 를 불러 붙여야 한다 — **로더 · 손실 연결은 아직 안 했다.**

```python
from z_training.data.phys2d_sup_labels import load_clip, window_labels
y = window_labels(load_clip(path), start, n_frames=32, stride=1, tubelet=2, max_obj=10)
# y["pos"] (16,10,2) · y["vel"] (16,10,2) · y["shape_id"] / y["color_id"] (10,) · y["mask_state"] (16,10) …
```
⚠️ 논문 문장과의 관계 (CLAUDE.md §12 질문 6): 상태 라벨은 action 은 아니지만 **관측만이 아닌 특권 정보**다. "action 없이 관측만으로 학습한 predictor" 를 주장하려면
이 supervision 을 쓴 판과 안 쓴 판을 나눠 보고하거나, 라벨을 보조 손실 · 진단 상한으로 위치시켜야 한다.

## 5. 정정 · 폐기 (생성기 개발 중, 기록)

- 첫 스모크에서 ledge_fall · toss_up · stack_topple · cup_cover 클립이 t ≈ 36 이후 **정지 화면**이었다 — 화면 밖으로 나간 떠다니는 가림막의 움직임이 운동 검사 · 끝 자르기를 통과시켰다. **수정**: 화면 안 가림막만 센다.
- 첫 본 생성 (2,319 편에서 중단) 의 중간 점검: g 5 등분 27 → 15 % (rejection 이 낮은 g 를 고름) · drift 가림막 설계 12 % → 30 % (무관한 가림막의 움직임이 운동 검사를 대신 통과시킴). **수정**: g 를 시도 번호와 무관한 난수로 고정 · 운동 검사는 시나리오 가림막만 센다. 그 판은 **지우고** 다시 만들었다 (지금 판: g 19.8–20.3 %, drift 12.5 %).
- osc_behind 의 1 차 궤적이 짧을 때 난수 범위가 비는 예외 — **수정** (T < 24 면 다시 뽑기), 한 편의 예외가 전체를 멈추지 않게 `failed.tsv` 로 돌린다.

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python      # pymunk 7.3.1 필요
R=/local_datasets/world/training_data/phys2d_sup_v1
$PY -W ignore z_training/data/gen_phys2d_sup.py --per-scenario 2 --out /tmp/p2s --debug-overlay        # 미리보기 + 라벨 겹친 그림 (_preview/*_labels.png)
$PY -W ignore z_training/data/gen_phys2d_sup.py --per-scenario 50   --seed 2 --workers 40 --preview-n 0 --out $R/val      # 1 분
$PY -W ignore z_training/data/gen_phys2d_sup.py --per-scenario 1000 --seed 1 --workers 40 --preview-n 3 --debug-overlay --out $R/train   # 22 분
$PY -W ignore z_training/data/audit_phys2d_sup.py --root $R/train --val $R/val --workers 40 > $R/train/audit.md
$PY z_training/data/build_phys2d_window_index.py --root $R --split train --max-per-clip 4 --out data_csv/phys2d_sup_v1/train_T32C16s1.csv
$PY z_training/data/build_phys2d_window_index.py --root $R --split val   --max-per-clip 2 --out data_csv/phys2d_sup_v1/val_T32C16s1.csv
cp $R/train/index.csv data_csv/phys2d_sup_v1/train_clips.csv; cp $R/val/index.csv data_csv/phys2d_sup_v1/val_clips.csv
```
생성은 이어 하기를 지원한다 (json · seg 가 있는 clip 은 건너뛴다). 같은 seed 면 같은 영상이 나온다.
