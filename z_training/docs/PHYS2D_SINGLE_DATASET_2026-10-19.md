# phys2d_single_v1 — 물체 1 개 · 시간 고른 가림 · 장마다 상태 라벨 (2026-10-19)

**목적 (사용자 2026-10-19)**: "물체 1 개로만 하자 (object permanence, gravity, solidity) 배울 수 있도록. object bias 없도록 …
가려지는 것도 많아도 돼 … 결국 각 시점별로 obj position, velocity, identity 등 가려져도 맞출 수 있도록 하려는 것" /
"초기에 가려져도 되고 나중에 가려져도 되고 둘 다 … 저런 구분 head 를 N, D 즉 블록 단위로 걸 거라 시간에 대한 편향도 없도록" /
"데이터는 코드만 있으면 어디서 만들어도 똑같이 만들어지도록".

| | |
|---|---|
| 위치 | `/workspace/dataset/training_data/phys2d_single_v1/{train,val}` (3.3 GB) |
| 영상 | train **39,593** (39,600 중, 실패 7 은 모두 cup_cover) · val **1,440** (전부) |
| 창 행 | `uniform.tsv` train **511,782** (영상당 12.9) · val 18,683 / `boundary.tsv` (사건 경계) train 164,704 · val 5,994 |
| 생성 시간 | 240 워커 13 분 + 재시도 4 분 (3D 생성을 멈춘 상태) |
| 지문 | train `30927cd84cdb0954` (39,593) · val `3eb717ec360a5353` (1,440) — val 은 `z_training/data/fingerprints/phys2d_single_v1_val.tsv` |

생성기 `z_training/data/gen_phys2d_single.py` 는 `gen_phys2d_sup.py` (phys2d_sup_v2, [`PHYS2D_SUP_DATASET_2026-10-18.md`](PHYS2D_SUP_DATASET_2026-10-18.md)) 를
import 하고 바뀐 것만 담는다. 렌더 · 라벨 (`_states.npz` 키 · 단위) · 검사 · 사건 행 규칙은 sup 판과 같다. 물체 축 길이는 언제나 1.

## 1. 시나리오 18 (물체는 언제나 1 개)

| 범주 | 시나리오 |
|---|---|
| 중력 7 | drop_bounce · projectile · toss_up · ramp_roll · ledge_fall · pendulum (단진자) · spring |
| 영속 5 | osc_behind · moving_screen · cup_cover · drop_behind · tunnel (시나리오 고유 가림막) |
| 고체성 5 | wall_bounce · **obstacle_bounce (새로)** · shelf_land · container · occ_wall — **정적 지형과의 충돌만** |
| 섞음 1 | sandbox (경사 · 선반 · 벽 · 통 2~4 개 + 물체 하나) |

sup 판에서 뺀 것: 방해 물체 · 화면 밖에서 들어오는 물체 · 이중 진자 · collide · stack_topple · occ_pass (아래 시간 고른 가림이 대신한다).

## 2. 가림 — 시간에 고르게

- 영상마다 가림 구간 수 n ∈ {0..4} (확률 8 · 27 · 30 · 22 · 13 %) — **seed 로 고정**, 시도 번호와 무관 (rejection 이 가림 수를 고르지 않게).
- 구간 시작 a ~ U[−L+1, T−1], 길이 L ~ U[2, 16] → 장 t 가 구간에 들 확률이 t 와 무관 (가장자리 포함). 그 구간 궤적 (물체가 화면 안) 을 모두 덮는 고정 가림막.
  가림막은 정적이라 물체가 다시 지나면 또 가려진다 — 라벨은 래스터로 다시 잰다. 실제 가림 구간 수 0 ~ 15+ (0: 4 %, 1: 24 %, 2: 21 %, 3: 18 %, 4: 12 %, 5+: 21 %).
- 첫 장부터 가려진 채 시작하는 영상도 있다 (영속 시나리오 밖에서 48~64 %).
- **멈춘 자리를 덮는 가림막은 30 % 만 받는다** (`P_COVER_REST`). 1 일 때 끝 장 가림이 70~88 % 여서 "가려짐 = 멈춤" 지름길이 됐다 (시험 432 편).
- 영상에서 가려진 장 비율 ≤ 0.7, 한 번은 보여야 한다 (vis > 0.3). 궤적과 무관한 고정 가림막 0~3 · 떠다니는 가림막 20 % 는 sup 판 그대로 (미끼 역할).

## 3. 편향 점검 (train 전수, `train/audit.md` 에서 다시 계산)

**시간** — `uniform.tsv` 창 C16 K16, 블록 16 (tubelet 2, 블록 라벨 = 마지막 장) 자리별 P(가려짐):

| | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 | p8 | p9 | p10 | p11 | p12 | p13 | p14 | p15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 42.6 | 43.1 | 43.3 | 43.0 | 42.4 | 41.9 | 41.5 | 41.4 | 41.3 | 41.4 | 41.6 | 41.8 | 41.9 | 41.8 | 41.6 | 41.3 |
| boundary (대조) | 44.4 | 44.4 | 44.2 | 44.6 | 45.4 | 47.0 | 49.0 | 50.1 | 48.6 | 48.1 | 48.1 | 47.2 | 46.0 | 44.7 | 43.5 | 42.2 |

uniform 폭 **1.9 pt** (val 2.8 pt). 사건 행 창은 경계 근처로 가림이 몰린다 (폭 7.9 pt) → **블록 단위 상태 head 에는 `uniform.tsv`.**
영상 장 기준으로는 첫 8 장 34.6 % · 마지막 구간 30.2 % 로 낮다 (영속 시나리오는 첫 장 · 끝 장 가림 0 % — sup 판 검사가 "가려지기 전 · 뒤에 보임" 을 요구). 창 안 자리별로는 위처럼 평평하다.

**속도와 가림** — 멈춘 물체가 더 가려지지 않는다:

| 속력 px/장 | 장 비율 % | P(가려짐) % |
|---|---:|---:|
| 0–0.1 | 17.1 | 39.3 |
| 0.1–1 | 6.2 | 42.7 |
| 1–3 | 17.5 | 40.5 |
| 3–6 | 16.7 | 44.9 |
| 6–10 | 17.5 | 42.6 |
| 10+ | 25.0 | 38.1 |

**정체 48** — 시나리오마다 48 칸 층화 (sup 판과 같다) + **실패하면 같은 정체로 seed 만 바꿔 최대 12 번** (`SEED_RETRIES`).
정체당 820~828 편, 모양당 6,594~6,600 편. 정체별 가림 비율 36.6~43.8 %, 평균 x 125~131 px, 평균 y 179~192 px.

| 모양 | 영상 | 가림 % | 속력 px/장 |
|---|---:|---:|---:|
| circle | 6,600 | **37.4** | **5.13** |
| square | 6,599 | 42.2 | 6.62 |
| rectangle | 6,600 | 41.4 | 6.54 |
| triangle | 6,600 | 41.8 | 6.61 |
| pentagon | 6,600 | 41.7 | 6.54 |
| hexagon | 6,594 | 41.0 | 6.41 |

⚠️ **남은 편향 — 원**: 속력 5.1 (나머지 6.4~6.6), 가림 37.4 % (나머지 41~42). 원은 굴러 에너지를 회전에 쓰고, 나머지는 굴림 시나리오에서 마찰을 낮춰 미끄러진다 (sup 판의 `slide`).
물리 차이라 두었다. 가림이 낮은 이유는 재지 않았다 (추측: 굴러서 화면 밖에 있는 장이 많거나, 가림막 크기 상한에 걸리는 빠른 구간이 다르다).

**시나리오별** (`train/audit.md` §3): 가림 26.8 % (drop_behind) ~ 46.2 % (pendulum). 영상 길이는 물체가 멈추면 짧다 — 48~59 장, 진자 · 용수철 · moving_screen · osc_behind 만 121~128 장.
seed 재시도가 많은 시나리오: cup_cover 56 % · moving_screen 33 % · occ_wall 29 % · drop_behind 28 % · tunnel 24 %.

## 4. sup 판과 다른 문턱

| | sup_v2 | single_v1 | 이유 |
|---|---|---|---|
| 최소 길이 | 64 | **40** | 물체 하나가 멈추면 장면이 멈춘다 — 64 면 시험 51/108 통과 |
| 멈춘 뒤 남기는 장 | 4 | **16** | 가려진 채 멈춰 있는 물체도 영속 감독 재료 |
| motion / displaced 문턱 | 0.7 / 0.6 | **0.6 / 0.5** | 꼬리가 길어진 만큼 |
| C16K16 사건 행 요구 | 있음 | **없음** | 가림 없는 진자 영상을 떨어뜨려 가림 수와 시나리오를 얽는다. 상태 라벨은 모든 장에 있다 |

## 5. 재현성 — 어디서 만들어도 같은 데이터

- 라벨은 (코드, seed, `z_training/data/requirements_phys_gen.txt` 버전 — numpy 2.5.2 · opencv-python 5.0.0.93 · pymunk 7.3.1) 로 정해진다. 워커 수 · 실행 순서 · 기계와 무관.
- **확인 (2026-10-19)**: 워커 90 / 13, `PYTHONHASHSEED` 0 / 777 로 val 설정 90 편을 두 번 → 라벨 · 세그 · boundary 행 **전부 일치**.
- 고친 것: `gen_phys2d_sup.record_contacts` 가 받침 이탈 태그를 **문자열 집합 순서**로 돌았다 (= `PYTHONHASHSEED` 에 따라 같은 장 사건 순서 → 행 난수가 바뀔 수 있다). `sorted` 로 고쳤다. 실행 스크립트도 `PYTHONHASHSEED=0`.
- 실행 스크립트가 버전을 검사하고 다르면 멈춘다 (`ALLOW_VERSION_MISMATCH=1`). 끝에 val 지문을 레포 기준과 대조한다 (`val/fingerprint_check.txt`).
- 같지 않을 수 있는 것: mp4 바이트 (ffmpeg/libx264 버전). 지문은 `_states.npz` 배열 · `_seg.npz` · json 의 사건 · 행 · 물체만 잰다.

## 6. 학습에 붙이려면 (아직 안 했다)

`configs/training/natural_ar_vw_state_phys2d.yaml` 의 `data.var_window.boundary.manifest` 를
`/workspace/dataset/training_data/phys2d_single_v1/train/uniform.tsv` 로 (`ctx_start` 0, 경계 4 장 간격, 예측 16 장이 들어가는 자리만). 라벨 키는 sup 판과 같아 `labels: phys2d_sup` 그대로.

## 재현

```bash
cd /workspace/world_model/vjepa2
# 환경: python 3.12 + requirements_phys_gen.txt, ffmpeg
python -m venv /venv/physgen && /venv/physgen/bin/pip install -r z_training/data/requirements_phys_gen.txt
# 미리보기 (라벨 겹친 그림 _preview/*_labels.png)
PYTHONHASHSEED=0 /venv/physgen/bin/python -W ignore z_training/data/gen_phys2d_single.py --per-scenario 4 --out /tmp/p2one --debug-overlay
# 전체 (val → train → audit → 지문 → val 지문 대조). 끊기면 같은 명령으로 이어 한다
PY=/venv/physgen/bin/python OUT=/workspace/dataset/training_data/phys2d_single_v1 WORKERS=240 bash z_training/data/run_gen_phys2d_single_v1.sh
# 대조만
/venv/physgen/bin/python z_training/data/fingerprint_phys_data.py --root $OUT/val --check z_training/data/fingerprints/phys2d_single_v1_val.tsv
```
