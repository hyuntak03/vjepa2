# VOBench post-FT 설정 (2026-10-14)

`vjepa2_pft/` 의 학습 · 채점 설정을 한곳에 모은 문서다. 결과는 `README.md` 에 있다.
아래 값은 run 폴더의 `config.yaml` · `metrics.jsonl` · `log_r0.csv` 에서 다시 읽어 확인했다.

## 1. 학습 run

| run | 출발 predictor | 학습 데이터 | epoch | step/epoch · 총 step | 상태 |
|---|---|---|---:|---|---|
| **`vobench_release_postft`** (held-out) | 릴리즈 ViT-H predictor (oneshot, mask token) | `vobench_split2_train` — split v2 train 절반의 가능 영상 **720 clip / 360 block** | **30** | 22 · **660** | 완료 (14:42–15:09, 약 27 분) |
| **`vobench_release_postft_all`** (상한선) | 〃 | `vobench_allpos_train` — VOBench 전체 가능 영상 **1,440 clip / 720 block** (held-out 없음) | **30** | 45 · 1,350 | 완료 (16:32–17:25) · 채점 VOBench 전체 |
| `vobench_ar_state_postft_v2` (대조) | AR-state ariel ep40 (`z_training/ariel/ar_state_future_1e_5`) | `vobench_split2_train` (720 clip) | 30 | 15 · 450 | 완료 · 채점 split v2 test |
| `vobench_vast_ar_state_postft` (대조) | vast AR-state ep70 (`z_training/vast/ar_state/latest.pt`) | `vobench_split2_train` (720 clip) | 30 | 15 · 450 | 완료 · 채점 split v2 test |
| `vobench_ar_state_postft` (v1, ⚠️ 누수) | 〃 | `vobench_split_train` — split v1 (test 가능 영상 52 개가 train 과 같은 설정 · 궤적) | 30 | 15 · 450 | 완료 · 기록용 |

AR-state ep70 (vast) 은 **post-FT 하지 않았다.** 그림의 ep70 막대는 VOBench 전체 채점에서 같은 test block 만 고른 값이다.

## 2. release post-FT 공통 설정

config 는 `configs/training/vobench_release_postft.yaml` (extends `frozen_predictor_postft` → `_base_frozen_predictor`) 이고, 상한선 판 `vobench_release_postft_all.yaml` 은 이 파일에서 데이터만 바꿨다.

| 항목 | 값 |
|---|---|
| 학습 대상 | **predictor 만.** encoder · target encoder 는 frozen 이고 릴리즈 ViT-H 체크포인트 그대로 (`load_predictor: true`). 옵티마이저 상태는 새로 만든다 |
| 역할 | 문맥 ← `encoder` (online) · 타깃 ← `target_encoder` (EMA). 채점 `surprise_c16t32` 와 같다 |
| 과제 | `temporal_prefix`, `context_frames: [16]` → 앞 16 장 (8 튜블릿) 문맥, 뒤 16 장 (8 튜블릿) 예측. 채점 창과 같은 꼴 |
| 손실 | L1 (`loss_exp 1.0`), 타깃 = LN(target_encoder(32 장)) 의 미래 토큰 (`target_layer_norm: true`) |
| 프레임 | raw 96 장 · stride 3 → 32 장, 256 px. 시작점은 `frames_start_choices [0, 1, 2]` 중 무작위 (채점은 0 고정) |
| 증강 | 좌우 뒤집기 p 0.5. **무작위 자르기 없음** (합성 장면은 물체가 작아서, 자르면 화면 밖으로 나간다) |
| batch | rank 당 8 × 4 GPU = **32** (vll5, 24 GB × 4) |
| 최적화 | AdamW, lr 5e-5 (warmup 1 epoch, 시작 1e-6 → cosine → 1e-6), wd 0.04, betas (0.9, 0.999), grad clip 없음 |
| dtype | bfloat16 (encoder · predictor 모두), SDPA |
| 저장 | 5 epoch 마다 (`e5.pt` … `e30.pt`) + `latest.pt`. 채점은 `latest.pt` (= epoch 30) |
| in-loop 감시 | IntPhys 1 dev 60 쌍 (유형당 10 block), 매 epoch. held-out 판: 71.7 (epoch 1) → 70.0 (epoch 30), 시작 전 출발선은 66.7 |
| train loss | held-out 판: 0.575 (epoch 1) → 0.482 (epoch 30) |

## 3. 데이터 분할

스크립트는 `z_training/data/build_vobench_split_index.py` 이다.

| 판 | 만드는 법 | train | test | 누수 검사 |
|---|---|---|---|---|
| **v2** (기본) | 가능 arm 설정 키 (objset · 운동 · 배경 · 가림 · k · 물체) 를 하나라도 공유하는 block 을 union-find 로 묶는다. 묶음째 train/test 에 보내고, 셀 (objset × 위반 × 가림 × 운동) 마다 반반이 되게 탐욕 배정한다 (seed 0) | 360 block 의 가능 영상 720 clip | 360 block × 4 변이 = 1,440 clip = **matched 720 쌍** | 공유 설정 키 0 |
| `all` (상한선) | 전부 train | 720 block 의 가능 영상 1,440 clip | — (채점은 VOBench 전체) | 해당 없음 — test 쌍의 가능 영상을 학습에서 본다 |
| v1 (⚠️ 누수) | block 단위 · 셀 층화 | 720 clip | 720 쌍 | test 가능 영상 52/720 이 train 과 같은 설정 · 궤적 (렌더 노이즈만 다름) |

v2 test 의 가림 시점별 block 수는 late 108 · early 108 · mid 108 · visible 36 이다. 따라서 운동별 그림의 칸 하나는 다음 크기다:
- late **12 쌍** (6 block × A/B 2 방향)
- visible **4 쌍**
- seen (visible · early · mid) **28 쌍**

위반 · 운동 · objset 구성은 train 과 test 가 같다 — "같은 분포의 held-out" 이다.

## 4. 채점

| 항목 | 값 |
|---|---|
| 프로토콜 | `surprise_c16t32` — 문맥 16 장 → 미래 16 장, latent L1, matched pairing, `mask_index 0`, float32 + autocast fp16 |
| held-out 판 | `vobench_v1_split2_test` (720 쌍) — `z_training/eval.sh vobench_release_postft vobench_v1_split2_test` |
| 상한선 판 | `vobench_v1` 전체 (1,440 쌍). held-out 판과는 **같은 v2 test 720 쌍** 에서 견준다 |
| 복사 기준선 | release 공간 (LN(z) 마지막 문맥 튜블릿), v2 test block 에서 다시 셈: primitive 72.5 · realistic 82.2 |
| CI | block 부트스트랩 95 % (2,000 회) |
| 비교 스크립트 | `vobench_split_compare.py --split v2` · 그림 `plot_vobench_postft.py --split v2 --base release` · `plot_vobench_pft_late_motion.py --occ {late,visible,seen} [--ar70]` |

## 5. 자 (decoder) 분석 설정

| 항목 | 값 |
|---|---|
| 자 학습셋 | `VOB_readout_primitive` (8,820 clip, `data_csv/rollout_v2_vob_readout_primitive`) — VOBench 와 다른 렌더 세트 |
| 자 | 위치 + 정체 (56 조합 + 없음) attentive 자 (`rollout2_identity_readout.py`, identity_r8 레시피, split block 단위 50/15/35) |
| 표현 | p = 그 predictor 의 미래 8 튜블릿, h = LN(target_encoder) — 자는 predictor 마다 따로 학습 (`identity_vobprim_rel` · `identity_vobprim_relft`) |
| 자 검증 (held-out) | release: p 정체 87.4 · 중심 1.26 칸 / release+FT: p 정체 87.7 · 중심 1.02 칸 / h 둘 다 93.8 · 0.75 칸 |
| VOBench 읽기 | `vobench_ar_readout.py extract --name {rel,relft}` → `plot_vobench_ar_readout.py --test-only` (v2 test block 만) |

## 재현

```bash
python z_training/data/build_vobench_split_index.py                 # split v2
python z_training/data/build_vobench_split_index.py --version all   # 상한선용 전체 가능 영상
GPUS=4 bash z_training/train.sh vobench_release_postft
GPUS=4 bash z_training/train.sh vobench_release_postft_all
GPUS=4 DECODE_WORKERS=8 BATCH_SIZE=16 bash z_training/eval.sh vobench_release_postft vobench_v1_split2_test
GPUS=4 DECODE_WORKERS=8 BATCH_SIZE=16 bash z_training/eval.sh vobench_release_postft_all vobench_v1
```
config: `configs/training/vobench_release_postft{,_all}.yaml` · 데이터셋 `configs/training/datasets.md ## vobench_split2_train · ## vobench_allpos_train` · test `configs/protocols/datasets.md ## vobench_v1_split2_test`.
