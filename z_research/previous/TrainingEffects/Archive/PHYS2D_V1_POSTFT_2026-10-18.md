# phys2d_v1 로 릴리즈 predictor post-FT — 두 목적함수 (encoder frozen)

2026-10-18 사용자 지시로 시작. **진행 중** (팔 A 23:08 시작, 팔 B 는 GPU 가 비면 chain 으로 자동).

> "/local_datasets/world/training_data/phys2d_v1 여기에 train set 있는데. 이거 release predictor 위에다가
>  pft 해보자 encoder frozen이고. 학습 방식은 미래 예측 (대신 그러면 어떤 물리적 특성이 context 에 들어가면
>  안되고 미래 예측부분에 들어가도록 event transition 이 context 에 있으면 안돼) 이거랑 predictor 가 원래
>  학습한 tublet 형식 {대신 encoder frozen} 으로 pft 2개 진행 가능하도록."

## 1. 두 팔

| | 팔 A `phys2d_future_postft` | 팔 B `phys2d_tubelet_postft` |
|---|---|---|
| 목적함수 | 미래 예측 (temporal_prefix C16 -> 16) | 릴리즈 사전학습 tubelet 마스크 (block3d 2 종) |
| 창 | 32 프레임 (stride 1 = 4 fps 원본) | 16 프레임 (= 사전학습 `dataset_fpcs`) |
| 데이터 | **사건이 예측 구간에만 있는 창 98,298 개** | clip 28,900 전체, 창 무작위 |
| 시작점 | 릴리즈 predictor | 릴리즈 predictor |
| encoder | **동결** (context <- `encoder`, target <- `target_encoder`) | 〃 |
| global batch | 96 (6 x 16) | 96 (6 x 16) |
| optimizer step | 1,024 x 10 = **10,240** | **10,240** (같게 맞췄다) |
| lr | 5e-5 (warmup 1, cos -> 1e-6), wd 0.04 | 〃 |

두 팔의 차이가 **목적함수만**이 되도록 step 수 · global batch · lr · 데이터 원본을 맞췄다.
남은 교락 하나: **창 길이가 다르다** (32 vs 16). 각자 자기 기준에 충실한 쪽을 택했다 —
팔 B 는 "원래 학습한 형식", 팔 A 는 채점 토폴로지 (`surprise_c16t32`).

## 2. ⚠️ 세트의 `boundary_frame` 을 쓰지 않았다

세트에 `boundary.tsv` 가 있고 `boundary_frame` 컬럼이 문맥/예측 분할점처럼 보인다. **쓰면 지시의 반대가 된다.**

전수 150,381 행에서 `boundary_frame − event_frame` 은 **정확히 {1, 2, 3}** 이고 거의 균등하다
(48,547 / 50,124 / 51,710). 즉 그 자리에서 자르면 사건이 **문맥의 마지막 1~3 프레임 안**에 들어온다.
(생성기가 "충돌은 보이고 결과만 예측" 을 의도한 것으로 보이나, 지시는 사건 자체를 문맥에서 빼라는 것이다.)

대신 `event_frame` 을 기준으로 창을 직접 골랐다. 그 컬럼이 **실제 운동 불연속**인지 먼저 쟀다
(컬럼 이름을 믿지 않는다):

| | event_frame | 무작위 프레임 대조군 |
|---|---:|---:|
| 그 시점 가속도의 백분위 중앙값 | **0.92** | 0.74 |
| 백분위 > 0.9 비율 | **0.57** | 0.20 |

사건이 정확히 하나인 clip 329 개 중 무작위 60 개. 전역 argmax 로는 안 된다 — 장면에 물체가 여럿이라
다른 물체의 사건을 집는다 (첫 시도에서 중앙값 차이 +37 프레임이 나왔다. 그 측정은 폐기).

## 3. 창 규칙 (팔 A)

`z_training/data/build_phys2d_window_index.py`. stride s, 길이 T, 문맥 C 프레임에서
문맥 = `[start, start+C*s)`, 예측 = `[start+C*s, start+T*s)` 일 때

1. 문맥 안에 **그 clip 의 어떤 event_frame 도 없다** — 겨냥한 사건 하나만 빼는 게 아니다
2. 예측 구간 안에 사건이 하나 이상 있다
3. `0 <= start`, `start + T*s <= video_frames`

T=32 · C=16 · stride 1 에서:

| | 값 |
|---|---|
| 창을 갖는 clip | **25,112 / 28,900 (86.9 %)** |
| 유효 창 | 437,761 → clip 당 4 개로 고르게 솎아 **98,298** |
| 카테고리 | solidity 34,736 · gravity 33,325 · permanence 30,237 |
| 예측 구간 사건 수 | 1 개 54,425 · 2 개 26,585 · 3 개 12,315 · 4+ 4,851 |

다른 후보 (참고): T32/C16/s2 53.1 % · T16/C8/s1 97.6 % · T32/C8/s1 97.6 %.
**T=32·C=16·s=1 을 택한 이유**: 채점 토폴로지 (`surprise_c16t32`) 와 같아서 학습 결과를
`z_training/eval.sh` 로 바로 잴 수 있고, stride 1 이 원본 4 fps = 릴리즈 사전학습 fps 와 같다.

눈으로도 확인했다 (3 창 contact sheet): `ledge_fall` 문맥 [0..15] 에서 공이 선반 위를 굴러가고,
낙하는 예측 [16..31] 안에서만 일어난다 (사건 21).

**이 규칙의 부수 효과 — 문맥은 늘 "사건 없는" 구간이다.** 그래서 팔 A 는 정확히
"마지막 관측에 **없는** 변화를 만들어 내는가" 를 학습 신호로 준다 (정본 `PAPER_STORY_2026-10-08` P3 의 3a·3b 축).

## 4. 데이터 (원본)

`/local_datasets/world/training_data/phys2d_v1` · 2.6 GB · 2D 물리 sandbox.
train 28,900 clip / val 1,020. **256×256 · 4 fps · 64~128 프레임** (mp4).
시나리오 17 종 × 카테고리 3, 각 1,700 clip:

- **solidity** (6): collide · container · occ_wall · shelf_land · stack_topple · wall_bounce
- **permanence** (5): cup_cover · drop_behind · moving_screen · occ_pass · tunnel
- **gravity** (6): drop_bounce · ledge_fall · pendulum · projectile · ramp_roll · spring

사건 종류 impact 127,376 · occlude 15,619 · leave 7,386. `manifest.tsv` 에 clip 마다
`copy_l1` · `pos_copy_err_px` · `linear_extrap_err_px` 가 이미 들어 있다 (복사 기준선 재료).
⚠️ 생성기는 이 레포에 없다 (`/workspace/...` 에서 만들어 옮긴 것). boundary 의미는 위처럼 **실측으로** 정했다.

## 5. 하드웨어 — VRAM 을 채워도 안 빨라진다 (실측)

사용자 요청으로 rank batch 를 쓸어 봤다 (팔 A, 1 GPU, compile 켬, 한계 iteration 의 **증분** 시간):

| rank batch | VRAM | iter | clip/s | 상대 |
|---:|---:|---:|---:|---:|
| 16 | 18.9 GB | 954 ms | **16.8** | x1.00 |
| 32 | 35.2 GB | 1,966 ms | 16.3 | x0.97 |
| 64 | 67.7 GB | 4,072 ms | 15.7 | x0.94 |

VRAM 3.6 배를 써도 처리량 **−6 %**. batch 16 에서 이미 SM 100 % · **541~578 W** 다
(EK100 포화 때 505~536 W 보다 높다). 샘플 하나가 32 프레임 x 256px = **4,096 토큰**이라
batch 16 만으로 65k 토큰이 ViT-H 에 들어간다 — EK100 때와 상황이 다르다.

그래서 **두 팔을 동시에 띄우지 않고 순차로** 돌린다. 각 팔이 자기 GPU 를 이미 포화시키므로
3+3 으로 쪼개면 각자 2 배 느려질 뿐이다: 순차 2.7 h + 1.4 h = **4.1 h** vs 동시 **5.4 h**.

`torch.compile` 은 켰다 (증분 1,263 → 954 ms, **x1.32**; 첫 step 에 ~25 s).

## 6. 고친 코드

| 파일 | 무엇 | 왜 |
|---|---|---|
| `app/vjepa_frozen/data.py` | **`video_window` 데이터셋 타입 신설** (`VideoWindowDataset`) | csv 한 행 = 창 하나. decord 로 `start, start+stride, ...` 를 정확히 읽는다. 행의 `n_frames` 가 `data.n_frames` 와 다르면 죽는다 |
| `app/vjepa_frozen/train.py` | `sync_masks` 창 자르기를 **block3d 에서 건너뛴다** | 자르기는 enc\|pred 가 앞쪽 C+K 블록을 빈틈없이 덮을 때만 뜻이 있다. block3d 는 `enc + pred < N` (16 프레임에서 288+1536=1824 < 2048) 이라 자르면 gather 가 범위를 넘어 **device-side assert** 로 죽었다 |
| `z_training/harness/resolve_train.py` | ① `video_window` 검사 (컬럼·실물·`n_frames` 일치) ② **`val.n_frames > data.n_frames` 를 모델 로드 전에 거부** | ②: encoder·predictor 가 `data.n_frames` 로 지어진다. 팔 B (16) 에 기본 val (32) 를 걸었더니 **학습을 다 끝낸 뒤 val 에서** device-side assert 로 죽었다 |
| `configs/training/datasets.md` | `phys2d_v1_future` · `phys2d_v1_future_val` · `phys2d_v1_clips` 등록 | |
| `z_vll7_training/configs/phys2d_{future,tubelet}_postft.yaml` | 두 팔 | |
| `z_training/data/build_phys2d_window_index.py` | 창 인덱스 생성기 | |

## 7. in-loop val (수렴 감시)

IntPhys1 dev 60 쌍. **정식 수치가 아니다** — 학습 뒤 `z_training/eval.sh` 로 낸다.

⚠️ **88.89 와 같은 줄에 두지 말 것** (2026-10-18 사용자 지적). 프로토콜이 다르다:

| | in-loop val | 88.89 (CLAUDE.md §5-4) |
|---|---|---|
| 창 | **고정 단창** C16/T32 stride 3 (raw 1~94) | Garrido **sliding A.8 격자 최고** `skip2_w32` |
| 쌍 | **60** (`n_blocks_per_type: 10`, seed 0 부분표본) | **180** 전수 |
| 필터 | 없음 | **Filtered** (논문 A.7/A.8) |
| 코드 | `app/vjepa_frozen/val.py` | `run.sh intphys1_sliding` |

같은 릴리즈 predictor 를 **고정 단창 전수 180 쌍**으로 재면 **66.67 %** 다 (§5-4). 지금의 73.3 은
그 집안의 60 쌍 부분표본이지 88.89 집안이 아니다. 창 하나 대 격자 최고의 간격 22.2 pt 는 §1-3 에 적힌 그것이다.
**88.89 와 비교 가능한 줄은 학습 뒤의 이것이다**:
`GPUS=6 bash z_training/eval.sh phys2d_future_postft intphys1_dev intphys1_sliding`

| | 창 | epoch 0 (= 릴리즈 출발선) |
|---|---|---:|
| 팔 A | 32 프레임 stride 3 (raw 1~94), 문맥 16 | **73.3 %** |
| 팔 B | 16 프레임 stride 6 (raw 1~91), 문맥 8 | **75.0 %** |

⚠️ **두 팔의 in-loop val 은 서로 비교할 수 없다** (창 길이·문맥이 다르다. 팔 B 를 32 프레임으로 걸 수
없는 이유는 §6 의 ②). 팔 B 에서 stride 3 (raw 1~46) 을 썼더니 12 쌍 중 **8 쌍이 tie** 라 (위반이 창 밖)
stride 6 으로 올렸다.

## 7-b. 궤적 — **두 팔 모두 IntPhys1 이 단조로 떨어진다** (2026-10-19)

⚠️ 팔 A 는 epoch 5/10 까지다 (진행 중, 03:55 종료 예정). 팔 B 는 10 epoch **완료**.

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** `future` val acc | 73.3 | 70.0 | 68.3 | 68.3 | 68.3 | 66.7 | | | | | |
| A margin ×10⁴ | 37 | 29 | 24 | 20 | 19 | 18 | | | | | |
| A train loss | — | .5256 | .4965 | .4880 | .4842 | .4817 | | | | | |
| **B** `tubelet` val acc | 76.7 | 78.3 | 76.7 | 75.0 | 73.3 | 70.0 | 68.3 | 68.3 | 65.0 | 65.0 | **65.0** |
| B margin ×10⁴ | **133** | 59 | 41 | 32 | 26 | 21 | 19 | 18 | 16 | 16 | **16** |
| B train loss | — | .5583 | .5360 | .5311 | .5279 | .5259 | .5231 | .5225 | .5213 | .5213 | .5209 |

**읽는 것은 acc 가 아니라 margin 이다.** acc 는 60 쌍이라 1 쌍 = 1.7 pt 이고 한두 칸 움직임은 잡음이다.
margin (`surprise(neg) − surprise(pos)`) 은 연속량이고 **두 팔 모두 10 epoch 내내 한 번도 올라가지 않았다**:

- 팔 B **+0.0133 → +0.0016 (8.3 배 축소)** · 팔 A **+0.0037 → +0.0018 (2 배)**
- 같은 구간에 train loss 는 내려가 평평해진다 (B 는 epoch 8 부터 .5213 고정 = 수렴)
- **붙는 방향이 neg 쪽이다** — 팔 B 에서 pos .5932 → .6003 (+0.0071) 인데 neg .6065 → .6020 (−0.0045).
  둘 다 오르는 게 아니라 **불가능 장면의 surprise 가 내려와서 가능 쪽에 붙었다**

즉 phys2d_v1 의 목적은 배웠는데 (loss 수렴), 배우는 동안 IntPhys1 의 가능/불가능 분리가 거의 사라졌다.
**`z_training/RESULTS_2026-09-19.md` 의 "post-FT 는 학습 도메인 안에서만 오른다" 의 네 번째 사례다.**
목적함수를 미래 예측 (팔 A) 으로 바꿔도 tubelet (팔 B) 과 **방향이 같다** — 즉 이 결과를 목적함수 탓으로 돌릴 수 없다.

⚠️ **단서 (결론과 같은 비중)**
- in-loop val 은 §7 의 고정 단창 60 쌍이다. **절대값은 88.89 와 비교 금지**, 궤적만 읽는다
- 60 쌍 궤적 하나로 "나빠졌다" 를 확정하지 않는다. 정식 수치는 §8 의 `eval.sh` (복사 기준선 포함) 로 낸다
- margin 축소 자체는 "분리가 나빠졌다" 와 "전체 surprise 규모가 바뀌었다" 를 아직 못 가른다 —
  채점은 부호만 보므로 acc 가 같이 떨어진 것이 분리 악화 쪽 증거이지만, 정식 채점으로 확인한다
- 학습 데이터 (phys2d_v1 2D 합성) 안에서 올랐는지는 **아직 안 쟀다.** 그것이 없으면 "도메인 안에서만 오른다" 의
  앞쪽 절반이 비어 있다 — `eval.sh` 에 phys2d held-out 줄을 같이 돌려야 한다

## 8. 아직 말할 수 없는 것

- 어느 목적함수가 나은가 — **학습이 안 끝났다.**
- 끝나도 in-loop val 로는 못 말한다. `z_training/eval.sh` 로 v11 · IntPhys1 · VOBench 를 **복사 기준선과 같이** 내야 한다
  (정본 하네스 질문 5). 합성 2D 세트로 학습했으므로 **학습 도메인 밖에서 재는 것**이 핵심이다
  (`RESULTS_2026-09-19.md`: post-FT 세 팔 모두 도메인 안에서만 올랐다).
- 창 길이 교락 (32 vs 16) 이 남는다. 결과가 갈리면 팔 B 를 32 프레임으로 한 번 더 돌려 교락을 떼야 한다.

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
# 1) 창 인덱스 (CPU, 1 분)
python z_training/data/build_phys2d_window_index.py --split train --n-frames 32 --context 16 --stride 1 --max-per-clip 4
python z_training/data/build_phys2d_window_index.py --split val   --n-frames 32 --context 16 --stride 1 --max-per-clip 2
#    팔 B 용 clip 목록 (경로 접두사만 교체)
sed 's#^/workspace/dataset/training_data/phys2d_v1#/local_datasets/world/training_data/phys2d_v1#' \
  /local_datasets/world/training_data/phys2d_v1/train/index.csv > data_csv/phys2d_v1/train_clips.csv

# 2) 검사만 (GPU 0 장, 몇 초)
DRYRUN=1 bash z_vll7_training/train.sh phys2d_future_postft
DRYRUN=1 bash z_vll7_training/train.sh phys2d_tubelet_postft

# 3) 학습 (6 GPU, 순차. 팔 B 는 GPU 가 비면 자동으로 올라간다)
GPUS=6 bash z_vll7_training/tmux_train.sh phys2d_future_postft
tmux new-session -d -s chain_phys2d_B "GPUS=6 bash z_vll7_training/chain.sh phys2d_tubelet_postft"

# 4) 채점 (학습 뒤)
GPUS=6 bash z_training/eval.sh phys2d_future_postft v11
GPUS=6 bash z_training/eval.sh phys2d_future_postft intphys1_dev intphys1_sliding   # <- 88.89 와 비교 가능한 줄
```
