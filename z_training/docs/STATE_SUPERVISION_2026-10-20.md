# 상태 감독 (state supervision) — 동기 · 설계 · 결과 · 고민 (2026-10-18 ~ 10-20)

> 블록 AR predictor (`kind=ar`, 상태 되먹임) 를 학습할 때, 합성 데이터의 **물체 상태 라벨** (정체 · 위치 · 속도) 로
> predictor 출력에 보조 손실을 거는 실험들을 한 곳에 모은다. 수치는 각 run 의 `train.log` · `metrics.jsonl` · 채점 산출물에서 다시 계산했다.

---

## 1. 동기와 철학

**목표** — 실제 영상을 보고 latent 예측 손실로 직관 물리를 배워 그럴듯한 미래를 그리는 predictor.
지금 블록 AR 은 두 가지가 안 된다 (`PROBLEMS_AND_DIRECTION_2026-10-18.md`):

- **영속** — 가림이 문맥 끝에 걸리면 predictor 미래에는 물체다운 토큰이 처음부터 없다 (CLAUDE.md §5-5). 가려지지 않아도 문맥에서 멀어질수록 물체가 흐려진다
- **뚜렷함** — L1 은 애매하면 중간값을 내라고 해서, 미래가 평균적인 물체 · 복사에 가까워진다

**상태 감독의 생각** — "예측한 미래에서 물체의 **존재 → 정체 (모양 × 색) → 위치 · 속도** 가 읽혀야 한다" 를 손실로 건다.
그러면 predictor 는 미래를 복사하거나 흐리게 만들어서는 맞힐 수 없다 (복사하면 위치가 안 움직이고, 흐리면 정체가 안 읽힌다).

**철학 (사용자 결정)**

| | 내용 |
|---|---|
| 주인공은 latent 예측 | 자연 영상 latent 손실이 학습의 중심이다. 상태 감독은 그 위의 **제약** — "자연 영상으로 배우되 지름길로 배우지 마라" |
| 압력은 출력에 | 감독 head 를 predictor 출력 (TF · rollout 둘 다, 문맥 자리 포함) 에 건다 — 출력 · 마지막 proj 까지 압력이 가도록 |
| head 는 꾀를 못 부리게 | head 가 크면 predictor 가 안 바뀌어도 head 가 혼자 읽는다 → head 는 작게 |
| 가려져도 감독 | 가려진 물체의 위치 · 정체도 맞히게 한다 — 영속 · 내부 상태 추정 |
| 한 번에 학습 | head 를 미리 학습해 고정하지 않고 predictor 와 같이 학습 (정답 토큰으로 미리 학습한 자는 가림을 못 가르친다) |
| config 로만 켜고 끈다 | `state_head:` · `data.var_window.boundary:` 키. 없으면 기준 AR 과 같다 |

---

## 2. 구현 (요약)

### 2-1. 데이터 — 합성 step 에만 감독

| 데이터 | 내용 | 라벨 |
|---|---|---|
| `phys2d_sup_v2` (10-18) | 2D 물리 38,343 편, 물체 여러 개 (평균 2.5, 최대 9), 정체 48 = 모양 6 × 색 8 | 장마다 위치 · 속도 · 보임 · 가림 · 정체 (`_states.npz`) |
| `phys2d_single_v1` (10-19) | 2D, **물체 1 개**, 39,593 편 | 같은 형식 |
| `phys3d_single_v1` (10-20) | **3D Blender 렌더**, 물체 1 개 | 같은 형식 (+ 3D 위치 · 속도) |

- step 의 일부 (`boundary.p`, 10 % → 20 %) 를 합성 창으로 바꾼다. 나머지는 자연 영상 (k710 · ssv2 · how, 기준 AR 과 같다)
- 창: 충돌 · 이탈은 예측 첫 장을 사건에 맞춘 행, 나머지는 무작위 시작 행 (가림은 어디서든) — `data_csv/<set>/train_mix.tsv`
- 라벨은 블록 마지막 장 상태. TF 출력 블록 b → 라벨 블록 b+1, rollout 걸음 r → 블록 C+r (latent 정답과 같은 정렬). 좌표 라벨이라 자르기 금지, 반전 끔

### 2-2. head 설계의 변천

| head | 구조 | 크기 | 결과 |
|---|---|---:|---|
| `attn` (main 425d67d) | 정체 query 48 × multi-head attention (투영 Q·K·V·O) + MLP → 존재 · 위치 · 속도 | 755 만 | 잘 읽는다 (rollout 위치 14 px). **채점 그대로** — head 가 혼자 읽는다 |
| `light` | query 48 · 투영 없는 내적 pooling · linear 1 개 | 6.9 만 | 같이 학습하면 못 읽음 (위치 53 px · 있음 recall 0 %) — "없음" 95 % 불균형 + clip 에 묶임 |
| `query3` | query 48 + 위치 · 속도 · 물체 49 지선다 linear 3 개 | 13 만 | 단독 학습 1,500 step 정체 47~54 % — 느림 |
| **`single`** (지금) | 물체 1 개 장면: **query 1 개** attention pooling → `(T, D)` → **linear 3 개** (물체 49 = 정체 48 + 없음 · 위치 · 속도) | 7 만 | 정체 80 % · 위치 21 px (복사 28 px 보다 좋음) |

### 2-3. 학습 설정 (지금 기준, `configs/training/natural_ar_vw_state_phys2d_single.yaml`)

| 키 | 값 | 이유 |
|---|---|---|
| `state_head.type` | `single` | 위 |
| `state_head.weight` | **0.05** | 제약 크기. 1.0 이면 합성 step 이 Adam 을 장악해 latent 학습이 멈췄다 (3-3) |
| `state_head.lr_mult` | 2 | head 만 약 1e-3 (단독 검증에서 잘 된 값) |
| `state_head.clip_separate` | true | clip 1.0 은 predictor gradient 에만. head 까지 같이 자르면 head 가 못 배웠다 |
| `optimization.clip_grad` | 1.0 | clip 없이 epoch 4 에 predictor 가 무너졌다 (3-3) |
| `boundary.p` | 0.20 | 합성 step 비율 |

코드: `app/vjepa_frozen/state_head.py` (head · 손실) · `var_window.py` (라벨 싣기) · `train.py` (손실 · clip 분리 · 로그) · `z_training/monitor_ar.sh` (head 구역).

---

## 3. 결과

### 3-1. head 가 읽는 것 (single run, 합성 step 학습 로그)

| epoch | 6 | 14 | 60 |
|---|---:|---:|---:|
| 정체 정확도 (49 지선다, 우연 2 %) | 29 % | 58 % | **80 %** |
| rollout 위치 오차 보임 / 가림 (px, 복사 기준선 28) | 64 / 68 | 30 / 27 | **21 / 22** |
| attention 집중도 (균등 = 1) | 46 | 139 | 200 |

가려진 물체의 위치도 보이는 물체와 같은 수준으로 읽힌다.

### 3-2. 채점 — 기준 AR 과 같은 수준

| | VOBench late P / R | gravity P / R (우연 37) | IntPhys1 | IntPhys2 | InfLevel |
|---|---|---|---:|---:|---:|
| 기준 AR | 73.6 / 77.3 | 20.4 / 8.3 | 90.0 | 51.8 | 58.4 |
| attn head + phys2d 10 % | 75.0 / 76.9 (in-loop) | — | 90.6 | — | — |
| **single + phys2d_single 20 %** | 73.6 / 76.4 | 17.6 / 7.4 | 91.7 | 51.8 | 58.8 |

VOBench late 위반별 (primitive / realistic): 영속 94.4 / 94.4 · 모양 58.3 / 63.9 · 색 68.1 / 70.8 — 기준 AR (93.1 / 95.8 · 61.1 / 65.3 · 66.7 / 70.8) 과 칸마다 1~3 쌍 차이.
자연 영상 latent (tf+ar) 는 기준 AR 보다 0.02 높게 끝났다 (e69: 1.098 vs 1.078).

### 3-3. 실패와 정정 (기록)

| 시도 | 일 | 교훈 |
|---|---|---|
| attn head · lr ×10 · clip 없음 | epoch 4 의 합성 step 하나에서 손실 46 → predictor 가 입력과 무관한 출력으로 무너짐 (val 동점 50 %) | 합성 step grad norm 13~22 vs 자연 0.25~0.85 — clip 필요 |
| light head (같이 학습) | 위치 53 px · 있음 recall 0 % | "없음" 95 % 불균형, clip 이 head 까지 잘랐다 |
| ⚠️ **정정** — "light head 가 위치를 못 읽는 건 RoPE 라 토큰 내용에 위치가 없어서" | main 의 IdReadout (같은 pooling + linear) 이 위치를 약 1 칸으로 읽었다 → **철회**. 원인은 학습 조건 | |
| single · weight 1 · lr ×3 | 자연 latent 1.25 로 멈춤 · val 50 (margin 0) | 합성 step 이 Adam 2 차 모멘트를 정해 자연 step 갱신이 약 1/5 |
| single · weight 0.05 · lr ×20 | head 발산 (위치 176 → 277 px) | ⚠️ **정정** — "손실을 줄인 만큼 lr 로 보상" 은 틀렸다. AdamW 는 gradient 크기와 무관하게 lr 만큼 움직인다 |
| ⚠️ **정정** — "상태 384 에 걸면 출력 1280 과 달리 가림 충돌이 없다" | 출력 = linear(상태) 라 같은 정보 → **철회**. 분리하려면 출력으로 안 그려지는 칸이 필요 | |

### 3-4. 그림으로 본 릴리즈 predictor (pixel decoder, `z_pixel_decoder/`)

블록 latent → 픽셀 decoder 로 릴리즈 V-JEPA 2 predictor 의 VOBench 미래를 그렸다 (`z_pixel_decoder/runs/vobench_vits/renders/`).
가림이 없어도 물체가 미래 처음 2~3 블록만 남고 사라지고, late 가림에서는 다시 나타나지 않는다 — §5-5 와 같다.
(이 decoder 는 블록별 latent 로 학습해 릴리즈의 32 장 window latent 는 모자이크로 그린다. 두 공간을 함께 배우는 ViT-B decoder 를 학습 중)

---

## 4. 고민 (열린 문제)

| # | 고민 | 근거 | 다음 |
|---|---|---|---|
| 1 | **읽히는 것 ≠ 채점이 오르는 것** | head 정체 80 % · 위치 21 px 인데 채점 그대로. 물체는 화면의 1~2 % (VOBench 물체 2~5 칸 / 256) 라 화면 평균 L1 에 묻힌다 (추측). 또는 "읽히는 방향" 으로만 담기고 정답 토큰과의 거리는 그대로 (추측) | 물체 자리 채점 (`obj_region_margin.py`), pixel decoder 로 미래 비교 |
| 2 | **가려진 물체 감독이 latent 손실과 부딪힌다** | 정답 토큰은 가려지면 가림막 — "가림막을 그려라" vs "물체를 남겨라". 출력이 상태의 linear 라 상태에 걸어도 같다 | **기억 토큰**: 블록마다 출력으로 안 그려지는 `(1, D)` 토큰 (AC predictor 처럼 앞에), head 는 그 토큰에 linear 3 개 |
| 3 | **합성 → 3D · 실제 영상으로 넘어가나** | 2D 감독은 VOBench (3D) 로 안 넘어갔다. 09 월 post-FT 세 번도 도메인 안에서만 올랐다 (`RESULTS_2026-09-19.md`). predictor 가 도메인별로 다르게 행동할 수 있다 (추측) | **phys3d 를 섞은 run** (`natural_ar_vw_state_phys23d_single`, 10-20 대기열) |
| 4 | **제약이어야 하는데 주인공이 되기 쉽다** | weight 0.05 에서도 합성 / 자연 grad norm 이 5~6 배 (기준 3 배) | 상태 gradient 를 latent gradient 의 일정 비율로 맞추는 장치 (`grad_ratio`, 미구현) |
| 5 | **head 의 세기** | 강하면 혼자 읽고, 약하면 못 배운다 | single 은 균형이 맞았다 — 기억 토큰으로 가면 pooling 도 필요 없다 |
| 6 | **기억이 생겨도 예측에 쓰이나** | head 만 만족시키는 곁가지가 될 수 있다 | 기억 끄기 실험. 쓰이게 만드는 힘: 다시 나타남 창 (`reappear` 6 만 행) · 실제 영상의 문맥 구멍 (`loss.ctx_hole`) |

**정답 (latent 목표) 을 바꾸는 것은 하지 않는다** — 정답은 관측이고, 가려진 물체가 안 보이는 게 맞다.
필요한 건 predictor 안에 **관측과 따로 노는 상태 공간** 을 두는 것 (Dreamer 의 잠재 상태 · 관측 모델 분리와 같은 생각).
encoder 를 여는 것 (LoRA) 은 하지 않는다 — 정답 쪽을 열면 붕괴 위험, 문맥 쪽은 이미 상태를 잘 읽는다 (속도 R² 0.996, probe 98~100 %).

---

## 5. 다음 순서 (제안)

1. `natural_ar_vw_state_phys23d_single` — 2D + 3D 합성 (고민 3). 10-20 pixel decoder 학습 뒤 자동 시작
2. 기억 토큰 run — 고민 2 · 6. head 를 출력 pooling 에서 기억 토큰 linear 로 옮기는 것만 다르게
3. 그 뒤 한 키씩: 변하는 칸 가중 L1 (뚜렷함) · 다시 나타남 창 · 실제 영상 문맥 구멍

---

## 재현

```bash
cd /workspace/world_model/vjepa2
# 학습 (config 맨 위 주석에 설계 이유)
GPUS=8 bash z_training/tmux_train.sh natural_ar_vw_state_phys2d_single       # single head · 2D
GPUS=8 bash z_training/tmux_train.sh natural_ar_vw_state_phys23d_single      # single head · 2D + 3D
watch -c -n 2 bash z_training/monitor_ar.sh <run>                            # head 구역: 손실 · 위치 · 정체 · 집중도 · 합성/자연 grad norm

# head 만 따로 (predictor 고정) — 자 검증
CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_training/tools/state_head_fit.py --run natural_ar_vw_state_phys2d \
    --head '{"type":"single"}' --root /workspace/dataset/training_data/phys2d_single_v1 --rows uniform --tag single_fit --steps 1500
# 자리별 (컨텍스트 TF · 예측 TF · rollout) 위치 · 존재
CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_training/tools/state_head_probe.py --run <run> --ckpt latest.pt

# 채점 (GPU 8)
GPUS=8 bash z_training/eval_bench.sh <run> latest.pt                                          # IntPhys1 · IntPhys2
RESULTS_ROOT=legacy GPUS=8 bash cvpr/eval/testbed/run.sh vobench_v1_late    vast_ar_vw_state_phys2d_single
RESULTS_ROOT=legacy GPUS=8 bash cvpr/eval/testbed/run.sh vobench_v1_gravity vast_ar_vw_state_phys2d_single   # 점수는 block 정의
# 2D 렌더 잡음 · latent 잡음 바닥
CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_research/scripts/analysis/latent_noise_floor.py
```
