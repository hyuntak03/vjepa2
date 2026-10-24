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

## 7-b. 궤적 — in-loop val 은 두 팔 모두 단조로 떨어진다 (2026-10-19)

> ⚠️ **정정 (2026-10-19, §9 정식 채점 뒤).** 이 절의 **측정은 맞다** (in-loop val 은 실제로 단조 하락했다).
> 틀린 것은 여기서 끌어낸 추론이다 — 나는 이것을 "post-FT 가 도메인 밖을 망친다" 로 읽었는데,
> 정식 채점에서 팔 A 는 IntPhys1 · VOBench 가 **거의 안 변했고** (±2 pt) gravity 는 **크게 올랐다**.
> in-loop val 은 **고정 단창 60 쌍**이라 팔 A 가 실제로 바꾼 축 (운동 외삽) 을 거의 재지 않는다.
> **결론은 §9 를 볼 것.** 이 절은 "60 쌍 궤적으로 일반화하면 안 된다" 의 기록으로 남긴다.


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

## 9. 정식 채점 (2026-10-19) — **팔 A 가 바꾼 것은 중력이 아니라 기본 미래다**

하네스 `cvpr/eval/{intphys1,testbed}/run.sh`, 6 GPU, vll7. 모델 셋을 **같은 판에서** 재어 비교한다
(문서의 88.89 는 옛 `Benchmarks/` 산출물이라 §1-3 의 프로토콜 섞임을 피하려고 릴리즈를 다시 돌렸다).
**릴리즈가 IntPhys1 88.89 를 정확히 재현했다** (O1 85.00 / O2 96.67 / O3 85.00) — 하네스 검증 완료.

| | IntPhys1 A.8 best | VOBench_v1 (1,440 쌍) | gravity (720 쌍) |
|---|---:|---:|---:|
| release `vith` | **88.89** | 76.39 | **8.74** |
| 팔 A `futureA` | 91.11 (+2.2) | 74.72 (−1.7) | **36.46 (+27.7)** |
| 팔 B `tubeletB` | 69.44 (−19.4) | 68.89 (−7.5) | 6.89 (−1.9) |
| **복사 기준선** | **85.00** | — (§9-6 못 돌림) | **12.85** |

### 9-1. IntPhys1 의 +2.2 는 **격자 최고값 효과**다

| 칸 | release | 팔 A | 팔 B |
|---|---:|---:|---:|
| skip2_w16 | 83.33 | **91.11** | 58.89 |
| skip2_w32 | **88.89** | 87.78 | 69.44 |
| skip5_w16 | 61.67 | 61.11 | 58.89 |

한 칸만 +7.8 이고 나머지는 평평하거나 살짝 내렸는데 **최고값을 고르니** 올라 보인다 (§1-3: sliding best 는
held-out 추정치가 아니라 descriptive). in-loop val 과 배치가 가장 가까운 `skip2_w32` 는 88.89 → 87.78 로
**§7-b 와 방향이 같다.** 팔 A 는 IntPhys1 을 올리지도 내리지도 않았다고 읽는 것이 맞다.

### 9-2. VOBench 는 세 축 모두 잡음 안 (팔 A) / 광범위 하락 (팔 B)

| | n | release | 팔 A | 팔 B |
|---|---:|---:|---:|---:|
| permanence | 480 | 83.33 | 81.46 (−1.9) | 67.71 (**−15.6**) |
| shape | 480 | 72.71 | 70.00 (−2.7) | 65.83 (−6.9) |
| color | 480 | 73.12 | 72.71 (−0.4) | 73.12 (±0.0) |
| late (문맥 끝 가림) | 432 | 61.81 | 59.49 (−2.3) | 56.94 (−4.9) |

### 9-3. ⚠️ gravity 의 +27.7 은 **중력이 아니다** — 변이별로 쪼개면 드러난다

gravity 는 block 하나에 과거 하나 + 미래 여럿이다. 미래 변이별 정확도 (`per_block.json` 의
`per_video_surprise` 를 index 의 `variant` 로 다시 묶어 계산):

| variant | n 쌍 | **copy** | release | 팔 A | 팔 B | 팔A−copy | rel−copy | 불가능 쪽 움직임 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `imp_stop` | 72 | 36.11 | 63.89 | 79.17 | 22.22 | **+43.1** | +27.8 | 적음 |
| `imp_rise` | 72 | 11.11 | 31.94 | 56.94 | 27.78 | **+45.8** | +20.8 | 많음 |
| `imp_float` | 216 | 1.39 | 5.56 | 34.26 | 2.31 | **+32.9** | +4.2 | 적음 |
| `imp_stop2` | 72 | 43.06 | 12.50 | 68.06 | 19.44 | +25.0 | −30.6 | 적음 |
| `imp_stop4` | 72 | 51.39 | 5.56 | 62.50 | 13.89 | +11.1 | −45.8 | 적음 |
| `imp_stop8` | 72 | 44.44 | 0.00 | 51.39 | 6.94 | +6.9 | −44.4 | 적음 |
| `imp_stop12` | 72 | 52.78 | 0.00 | 45.83 | 4.17 | −6.9 | −52.8 | 적음 |
| **`imp_line`** | 72 | **65.28** | 79.17 | **29.17** | 63.89 | **−36.1** | +13.9 | **비슷 (등속)** |
| 미시 평균 (쌍) | 720 | 30.83 | 20.97 | 49.58 | 16.53 | +18.8 | −9.9 | |

⚠️ 위 표는 **쌍 미시 평균**이고, 표 맨 위의 8.74 / 36.46 은 하네스의 `block_pairwise` (**블록 거시 평균**) 다.
ledge 블록이 144 / 216 (67 %) 인데 쌍은 1 개씩뿐이라 두 집계가 갈린다. 거시로 다시 내면
(하네스 값과 소수점까지 일치 — release 8.74 · 팔 A 36.46 · 팔 B 6.89 재현):

| 거시 (block_pairwise) | n_block | copy | release | 팔 A | 팔 B |
|---|---:|---:|---:|---:|---:|
| OVERALL | 216 | **12.85** | 8.74 | **36.46** | 6.89 |
| ledge (`pos` vs `imp_float` 뿐) | 144 | 0.00 | 0.00 | 27.08 | 0.00 |
| arc (미래 8 개) | 72 | **38.54** | 26.22 | **55.21** | 20.66 |

**`imp_line` 이 유일하게 중력을 재는 칸이고, 거기서만 무너졌다.** `imp_line` = 등속 직선 = 속도는 정답
(포물선) 과 같고 **가속만 없는** 미래다. 중력을 배웠다면 이 칸을 가장 잘 걸러야 하는데 79.17 → 29.17 로
chance 한참 아래다. 나머지 7 칸 중 6 칸이 "불가능 쪽이 **덜** 움직임" 이고 그게 **720 쌍 중 576 쌍 (80 %)**
이라 overall 이 +27.7 로 뜬다.

**복사 기준선이 이것을 확정한다 (2026-10-19 측정, `cvpr/eval/testbed/copy_baseline.py`):**

- **릴리즈는 gravity 에서 복사보다 못하다** — 거시 8.74 vs copy 12.85, arc 26.22 vs copy 38.54.
  즉 릴리즈의 낮은 점수는 "예측을 안 해서" 가 아니라 **복사보다도 틀린 방향으로 예측해서** 다
- **팔 A 는 복사를 분명히 넘는다** (거시 +23.6, arc +16.7, 미시 +18.8) — 8 변이 중 6 개에서 copy 초과.
  즉 팔 A 는 진짜로 **예측하고 있고**, 복사로 설명되지 않는다
- **그런데 `imp_line` 에서만 복사보다 36.1 pt 아래다** (29.17 vs copy 65.28). 복사조차 65 % 로 거르는
  등속 미래를, 팔 A 는 **정답 포물선보다 더 그럴듯하게 여긴다.** 소극적 실패가 아니라 **방향이 있는 실패**다

**맞는 해석:** 팔 A 는 기본 미래를 **"머묾" 에서 "속도 이어가기" 로 옮겼다. 중력이 아니라 1 차 외삽이다.**

- 릴리즈 기본값 ≈ 복사/머묾 → `float`·`stop` 이 안 놀랍다 (0~6 %) · `line` 은 놀랍다 (79 %)
- 팔 A 기본값 ≈ 속도 이어가기 → `float`·`stop` 이 놀랍다 (34~79 %) · `line` 은 안 놀랍다 (29 %)

§3 (도메인 안) 이 이것을 뒷받침한다 — 팔 A 의 이득이 **17 시나리오 전부에 균일**했다 (Δ +0.3348,
칸별 편차 ±0.008). 중력이 없는 `pendulum` · `spring` · `collide` · `occ_pass` 도 똑같이 올랐다.
**물리별 특화가 아니라 전반적 예측력 향상이다.**

⚠️ **철회.** 2026-10-19 작업 중 "ledge 12 칸이 전부 0 → 8~50 %, 설계한 대로 맞은 축" 이라고 한 번 적었다.
   **틀렸다** — ledge 의 변이는 `pos` vs `imp_float` 뿐이라 전부 '덜 움직임' 축이고, CLAUDE.md §5-6 이
   경고한 교란 (`"ledge 0 % 는 '덜 움직이는 미래가 L1 로 싸다' 와 못 가른다"`) 그 자체다. 다시 쓰지 말 것.

### 9-4. 도메인 안 (phys2d_v1 held-out 1,774 창) — §3 과 같은 결론

`z_research/scripts/analysis/phys2d_heldout_l1.py`, 복사 기준선 포함. ⚠️ **VoE 가 아니라 예측 정확도다**
(phys2d_v1 에 가능/불가능 쌍이 없다, §4).

| | copy | release | 팔 A | 팔 B |
|---|---:|---:|---:|---:|
| 예측 L1 (17 시나리오 평균) | 0.8089 | 0.5748 | **0.4742** | 0.5584 |
| Δ (copy − L1) | — | +0.2341 | **+0.3348** | +0.2506 |

**두 팔 다 도메인 안에서는 올랐다.** 즉 "학습이 망했다" 가 아니다 — §7-b 의 in-loop val 하락은
학습 실패가 아니라 다른 것을 재고 있었다.

### 9-5. 그래서 무엇이 섰나 / 안 섰나

✅ **섰다** — 관측만으로 한 frozen-encoder post-FT 가 릴리즈 predictor 의 **운동 외삽 한계를 실제로 밀어냈다**
(§5-5 의 "p 는 마지막 관측 자리에서 2~3 칸 (40~60 px) 안에서만 유지" 가 느슨해졌다). `imp_stop8` · `imp_stop12`
가 **0.00 → 51.39 / 45.83** 인 것이 가장 직접적인 증거다 (릴리즈는 그 칸을 한 번도 못 맞혔다).

❌ **안 섰다** — intuitive physics 를 따르는 미래는 아니다. 가속 (3a) 을 재는 유일한 칸 `imp_line` 에서
**오히려 크게 나빠졌다** (79.17 → 29.17). 정본 §2-4 의 합격선 (Ariel 을 넘는다) 근처에도 가지 않는다.

### 9-6. 못 낸 것

- **VOBench_v1 복사 기준선** — 옛 복사 엔진이 쓰는 `configs/protocols/datasets.md` 의 `vobench_v1`
  `frames_root` 가 `/data2/...` (vll5) 인데 이 노드엔 `/data2` 가 없다. gravity 는 항목을 새로 추가해 뚫었지만
  (`## vobench_v1_gravity`, 2026-10-19), `vobench_v1` 은 기존 항목을 고치면 vll5 가 깨져 손대지 않았다.
  **고치는 법**: 복사 엔진 (`cvpr/eval/testbed/copy_baseline.py`) 을 cvpr resolver 로 옮기면 두 노드 다 된다
- **IntPhys 2** — 안 돌렸다 (시간). leakage 가 작은 세트라 §9-5 의 ❌ 를 더 세게 걸 수 있는 자리다
- **gravity 복사 기준선은 `--no-check` 로 받았다** — 두 resolver 차이가 (1) `frames_root` 문자열
  (심볼릭으로 **같은 디렉토리**임을 `readlink -f` 로 확인) 과 (2) `scoring.pairing` matched vs cross 뿐이고,
  (2) 는 엔진의 쌍 정확도를 안 쓰고 `copy.npy` 를 variant 로 다시 묶었으므로 무관하다.
  ⚠️ `copy_baseline.py` 는 gravity 에서 마지막 출력 한 줄 (`by violation_type`, `n_pair=None`) 에 죽는다 —
  **배열은 그 전에 다 저장된다.** 고치려면 `summarize()` 의 포맷 문자열에 None 가드를 넣을 것

⚠️ **두 팔은 한 변수 대조가 아니다.** 마스킹 목적 (temporal_prefix vs block3d) 외에 **창 선택**
(사건 강제 vs 무작위) 과 **프레임 수** (32 vs 16) 가 같이 다르다. "목적함수가 원인" 이라고 쓸 수 없다.
가르려면 팔 B 를 32 프레임 + 같은 창으로 다시 돌려야 한다 (§8 의 교락과 같은 항목).

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
