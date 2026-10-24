# phys3d_single_v3 — 놓기 전 공중에 떠 있지 않음 · 렌더 4 fps (2026-10-23, ⚠️ 시험까지 · 전체 렌더 미시작)

**왜 (사용자 2026-10-23)**: v2 `drop_behind_00000.mp4` — "처음에 왜 공중에 떠 있음…? 이상하잖아." · "렌더 시간 왜 이리 오래 걸려. 좀 더 빠르게 안 돼?"

v2 ([`PHYS3D_SINGLE_V2_DATASET_2026-10-21.md`](PHYS3D_SINGLE_V2_DATASET_2026-10-21.md)) 는 사건이 영상 곳곳에 오게 놓는 시각을 0~120 장 늦췄는데
그동안 물체를 그 자리에 붙잡아 두어 공중이면 떠 있었다. v2 train 실측 (시작부터 12 장 넘게 바닥 위 0.3 m 이상에 멈춤):
drop_behind 91 % · drop_bounce 89 % · shelf_land 89 % · ramp_roll 90 % · obstacle_bounce 88 % · ledge_fall 86 % · projectile 67 % (평균 44~61 장).
v2 는 지우지 않고 둔다 (`/workspace/dataset/training_data/phys3d_single_v2`, 학습에 쓰지 말 것을 권한다).

| | |
|---|---|
| 위치 (예정) | `/workspace/dataset/training_data/phys3d_single_v3/{train,val}` |
| 생성기 | `z_training/data/gen_phys3d_single_v3_blender.py` (v2 를 import · 붙잡기와 저장만 바꿈) · `gen_phys3d.py --single-v3 --stride 1 --frames 288` · 실행 `run_gen_phys3d_single_v3.sh` |
| 시간 축 | 시뮬 · 라벨 · 사건은 12 fps 로 만들고 **렌더 · 저장은 3 장마다 1 장 (4 fps)**. mp4 · `states.npz` · tsv 행 전부 4 fps 장 단위 → **학습 config 는 `stride: 1`** |

## 1. v2 와 다른 것

1. **붙잡는 자리** (놓는 시각 delay > 0 일 때만, `env["hold"]`):
   - `rest` — 아래 면 (바닥 · 선반 · 경사면 · 장애물 위) 과 0.15 m 안이면 그 면에 내려놓는다. 놓는 순간 v2 와 같은 초속도를 받는다 (차인다).
   - `throw` — 떠 있고 초속도 > 1.5 m/s: 같은 포물선을 거꾸로 따라가 아래 면에 닿는 자리에서 출발 (바닥 · 받침에서 던진 것 — 놓은 뒤 궤적이 v2 와 같다).
   - `throw_shift` — 거꾸로 간 자리가 방 밖이거나 길이 막히면: 바로 아래 면에서 같은 수평 속도 · 같은 높이까지 오르는 속도로 (포물선을 옆으로 옮긴 것).
   - `rod` — 떠 있고 느리면: 화면 위 밖에서 내려온 막대 + 집게가 쥐고 있다 놓는 순간 위로 빠진다 (충돌 없음, 3 장 (렌더) 안에 8 m 위).
   - delay = 0 이면 v2 와 같다 (붙잡는 장이 렌더 앞 준비 장 안).
2. **4 fps 저장** (`S.SAVE_STRIDE = 3`, `S.subsample`): 렌더 장 0, 3, 6, … 만 렌더 (`frame_step`). 라벨은 남긴 장에서 속도 · 가속도 · 각속도를 다시 미분 (단위 = 저장 장당).
   행: 경계 round(b/3) · 끝 ceil(end/3) · 가려짐 행 pred_min 은 다시 보이는 렌더 장을 덮는 최소 K (2 미만이면 행을 뺀다 — 경계에서 이미 보임).
   나중에 12 · 6 fps 학습은 못 한다.
3. 나머지 (길이 최대 288 렌더 장 = 96 저장 장, 안 보임 연속 ≤ 45 렌더 장 = 15 저장 장, 미끄러지는 가림막, 사건 행) 는 v2 그대로.

## 2. 시험 결과 (2026-10-23)

**라벨만 624 편** (13 × 48, 렌더 생략, 2 분) — 전부 통과. `audit.md` 와 별도 집계에서 (아래 §2-1 의 상한 수정 뒤 다시 만든 것):

| | v3 시험 | v2 val (렌더) |
|---|---:|---:|
| 가림 (전체 장) | 41.6 % | 39.1 % |
| uniform 창 블록 자리 폭 | 2.2 pt | 1.8 pt |
| 안 보임 연속 최대 (저장 장) | **15** (= 렌더 45) | 45 렌더 장 ⚠️ §2-1 |
| C4 / C8 / C16 문맥 전부 안 보임 | 29.0 / 14.1 / 0.0 % | 26.9 / 13.1 / 0.6 % |
| 멈춘 장 (< 0.1 px/장) | 57.9 % | 64.5 % |
| 사건 행 (영상당) | 2,382 (3.8) | 4,124 (4.0) |
| 사건 경계 분포 (학습 16 장 폭마다) | 22.6 · 28.6 · 26.1 · 12.1 · 7.6 · 3.0 % | 21.7 · 28.5 · 26.0 · 13.8 · 7.5 · 2.4 % |
| C32 · K32 적격 행 (impact · leave · occlude) | 97 · 48 · 442 | (val 1,040 편) 174 · 150 · 696 |
| 모양별 가림 | 36.5 ~ 48.5 % | 30.4 ~ 40.7 % |

- 모양별 폭은 v2 시험 (36~50 %) 과 같은 표본 효과로 본다 (시험은 정체마다 같은 seed k 를 시나리오에 걸쳐 쓴다 — v2 train 에서 34.7~38.3 % 로 좁아졌다). **추측** — v3 train 에서 확인.

### 2-1. ⚠️ 안 보임 상한이 4 fps 에서 새고 있었다 (v2 에도 해당)

첫 시험에서 4 fps 저장 장의 안 보임 연속이 최대 28 장 (15 넘는 영상 43 / 624) 이었다. 12 fps 에서 재면 판 가장자리에서 `vis_frac` 가 0.3 을
오르내려 한 장씩 끊기는데, 3 장마다 남기면 그 끊김이 빠진다 (pendulum seed 15: v2 12 fps 42 장 → v3 4 fps 26 장 = 렌더 78 장).
**v2 를 stride 3 으로 학습하면 같은 일이 생긴다** — v2 의 "≤ 45 렌더 장" 은 12 fps 에서만 맞다 (v2 train 을 3 장마다 다시 재지는 않았다).
수정: 상한 검사 (판 놓기 · 마지막 검사) 를 저장하는 장에서 잰다 (`S.blind_run` = 저장 장 연속 × 3 ≤ 45). SAVE_STRIDE 1 (v1 · v2) 이면 전과 같다.
수정 뒤 최대 15 저장 장 · C16 문맥 전부 안 보임 0.0 %.

- 붙잡는 자리 (624 편): drop_behind rod 48/48 · shelf_land rod 46 · ledge_fall · ramp_roll rest 48/48 · projectile throw_shift 39 · wall_bounce rest 27 / throw_shift 20 · pendulum none.
- "막대 없이 바닥 위 0.3 m 이상에서 4 저장 장 넘게 멈춤": ledge_fall 83 % · ramp_roll 85 % · obstacle_bounce 33 % — **선반 · 경사면 · 장애물 위에 놓인 것** (그림으로 확인: obstacle_bounce 8 은 위 판 위, 9 는 기둥 위). 나머지 ≤ 4 %.
- 렌더 예시 (drop_behind · shelf_land · projectile · ramp_roll seed 0): 막대에 매달린 구 → 막대가 위로 빠지며 낙하, 바닥에 놓인 공 → 던져짐. 눈으로 확인.

**렌더 169 편** (13 × 13, CPU 워커 169, 8 분): 영상당 평균 325 초 (중앙 332) — v2 train 704 초의 **2.16 배 빠름**.
3 배가 안 되는 건 영상당 고정 비용 (시뮬 · 장마다 보임 재기 · 가림막 재시도, 12 fps 그대로) 이 약 130 초라서다 (두 점으로 푼 추정).
→ 216 코어에서 train 15,600 편 ≈ 6.5 시간 · val 1,040 편 ≈ 0.5 시간.

## 3. 단서

- `rest` 는 "가만히 있다 저절로 출발" 이 남는다 (놓는 장은 예측 못 하는 사건으로 info unpred 에 넣어 행이 넘지 않게 함). 막대 `rod` 도 놓는 순간 v2 의 작은 초속도 (≤ 1.5 m/s) 를 준다.
- 닫힌 방이라 "화면 밖에서 굴러 들어오기" 는 못 했다 (양옆 벽이 화면 안).
- 미끄러지는 가림막 · 미끼 판은 v2 처럼 공중 높이에 선다 (받침 없음).
- 멈춘 장은 57.9 % 로 여전히 많다 (놓기 전 정지 + 멈춘 뒤 꼬리).
- 로더는 아직 `pred_min` 을 읽지 않는다.

## 재현

```bash
cd /workspace/world_model/vjepa2
# 라벨만 시험 (렌더 생략, 624 편 ≈ 2 분)
PYTHONHASHSEED=0 PHYS3D_NORENDER=1 python3 z_training/data/gen_phys3d.py --single-v3 --stride 1 --frames 288 \
  --out /tmp/p3v3 --per-scenario 48 --gpus "" --cpu-workers 180
python3 z_training/data/audit_phys2d_single.py --three-d --stride 1 --root /tmp/p3v3 > /tmp/p3v3/audit.md
# 렌더 시험 (169 편 ≈ 8 분) — manifest.tsv 의 sec 열이 영상당 초
PYTHONHASHSEED=0 python3 z_training/data/gen_phys3d.py --single-v3 --stride 1 --frames 288 --out /tmp/p3v3r --per-scenario 13 --gpus= --cpu-workers 169
# 예시 한 편
/workspace/tools/blender-4.2.3-linux-x64/blender -t 8 -b -noaudio -P z_training/data/gen_phys3d_single_v3_blender.py -- \
  --scenario drop_behind --seed 0 --frames 288 --out /tmp/p3v3one
# 전체 (val → audit → 지문 → train). val 만: VAL_ONLY=1
TRAIN_PER=1200 CPU_WORKERS=216 bash z_training/data/run_gen_phys3d_single_v3.sh
```
