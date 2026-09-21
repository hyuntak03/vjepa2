# EK100 action anticipation 재현 — 시작점 (2026-09-14 개정)

> ## 2026-09-20 결정 (사용자): **기준은 공개 코드 규약 (`released`) 이다.**
> 논문 Table 5 수치와 공개 probe 체크포인트가 그 규약에서 나왔다 (§2-A, §A-1). 비교 가능한 축은 그것 하나이므로
> 재현·보고의 기준선은 `ek100_vith_official256_released_grid8` 의 **verb 56.36 / noun 56.16 / action 35.34** (논문 ViT-H 59.2 / 54.6 / 36.5) 로 둔다.
> `paper` 규약 (문맥이 action 시작 1 s 전에 끝남) 수치는 **정직한 anticipation** 으로 병기하되 논문 비교선으로 쓰지 않는다.

> ⚠️ 정정 (2026-09-15) — 아래 "아직 본 학습은 안 돌렸다" 는 지난 서술이다. `ek100_vith_lr3e-4` (paper 규약 · timestamp · head 1 · **옛 데이터**) 가 20 epoch 를 마쳤다:
> verb 32.05 / noun 34.81 / action 18.41 (최고 action 18.59 @ epoch 19). 논문 36.5 는 릴리즈 규약 (context 가 action 끝 1 s 전) 이라 비교 대상이 아니다.
> 새 데이터 (§3-1) 로는 `train_vith.sh paper|released` 로 다시 돌린다.
>
> 아직 본 학습(20 epoch)은 돌리지 않았다. 1-GPU smoke(비디오 4개, 1 epoch)는 끝까지 통과했다 (§5).
> 레포 규칙은 루트 `CLAUDE.md`. 논문은 레포 루트의 `V-JEPA 2- Self-Supervised Video Models Enable Understanding, Prediction and Planning.pdf`
> (§6, Appendix C.1·D.1, Table 5·19·20). arXiv 2506.09985 판과 §6·Appendix D 본문이 글자 단위로 같다.

## 0. 실행

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
DRYRUN=1 GPUS=8 bash z_research/anticipation/EK100/run.sh        # 검사만 (몇 초, GPU 0장)
GPUS=1 SMOKE=1 bash z_research/anticipation/EK100/run.sh         # 배관 점검 (~3분 + 모델 로딩)
GPUS=8 bash z_research/anticipation/EK100/run.sh                 # 본 설정 = configs/ek100_vith.yaml
GPUS=8 bash z_research/anticipation/EK100/sbatch.sh              # SLURM (vll5)
watch -n 1 bash z_research/anticipation/EK100/monitor.sh         # 모니터 (가장 최근 run, 또는 인자로 TAG)

# 2026-09-15 부터 본 학습은 이것 — 공식 기하 재인코딩본 (§3-1) + 규약을 인자로 고정. SLURM 8 GPU
bash z_research/anticipation/EK100/train_vith.sh paper      # context = action 시작 1 s 전에 끝 (논문 글)   TAG ek100_vith_official256_paper
bash z_research/anticipation/EK100/train_vith.sh released   # context = action 끝 1 s 전에 끝 (릴리즈 코드) TAG ek100_vith_official256_released
bash z_research/anticipation/EK100/eval_released_vitl.sh    # 공개 ViT-L probe val 만 (released + frame_fixfps)
HEADS=grid8 NODE=vll3 MEM_PER_GPU=40G bash z_research/anticipation/EK100/train_vith.sh released
                                                            # 2026-09-15 제출: head 8 개 (lr {3e-4,1e-4} x wd {1e-4,1e-3,1e-2,1e-1}), vll3
# ⚠️ head 20 개 (HEADS=sweep) 는 RTX 4090 24 GB 에 안 들어간다: head 1 개 = 82.1M 파라미터, 가중치+grad+Adam+DDP 버퍼 1.53 GB → 20 개 30.6 GB.
#    head 8 개도 원본 루프 (전부 fwd 뒤 전부 bwd) 는 activation 이 쌓여 안 들어가서, eval.py 를 head 마다 fwd→bwd 로 바꿨다
#    (CPU·DDP·가짜 데이터에서 원본 방식과 파라미터 업데이트 차이 0.0 확인, bf16 경로 포함).
```

예전 명령 `EK100_ROOT=... SET="data.spatial_mode=short_side data.time_source=timestamp model_kwargs.module_name=..._nonsquare model_kwargs.wrapper_kwargs.mask_index=0" TAG=ek100_vith_ts_mask0 GPUS=8 bash run.sh ek100_vith`
의 설정이 **이제 config 기본값**이다. 위 세 번째 줄과 같은 계산이다.

| 환경변수 | 뜻 |
|---|---|
| `GPUS` | GPU 수 (기본 1). 8 이면 global batch 128 = 논문 |
| `TAG` | 결과 폴더 이름 (기본 `ek100_vith`). **설정을 바꾸면 TAG 도 바꾼다** — 같은 TAG 에 다른 설정의 `latest.pt` 가 있으면 resolve 가 막는다 |
| `SET="a.b=1 c.d=2"` | config 덮어쓰기. `data.` `optimization.` `classifier.` `evaluation.` 는 `experiment.` 생략 |
| `HEADS=sweep` | 논문의 probe head 20개 (lr 5종 x wd 4종) |
| `SMOKE=1` `VAL_ONLY=1` `DRYRUN=1` | 배관 점검 / `latest.pt` 로 val 만 / 검사만 |

자주 쓸 변형:
```bash
TAG=ek100_vith_mask1   SET="model_kwargs.wrapper_kwargs.mask_index=1"      GPUS=8 bash .../run.sh   # 릴리즈 코드의 mask token
TAG=ek100_vith_encoder SET="model_kwargs.wrapper_kwargs.no_predictor=true" GPUS=8 bash .../run.sh   # encoder 만 (논문 Table 20 대조군)
TAG=ek100_vith_sweep   HEADS=sweep                                          GPUS=8 bash .../run.sh   # 논문 head 20개
```

출력 `exp_results/action_anticipation_frozen/<TAG>/`: `_resolved.yaml` · `stdout.log` · `log_r0.csv` · `latest.pt` ·
**`val_metrics.jsonl`** (epoch 마다 한 줄: mean class recall@5 verb/noun/action, 센 clip 수, head 별 값). **보고 값은 마지막 epoch.**

---

## 1. 논문이 한 것 (원문 기준)

| 항목 | 논문 | 어디 |
|---|---|---|
| 과제 | action segment **시작 전**의 context clip 으로 verb / noun / action 예측. "The interval between the end of the context and the beginning of the action segment is the anticipation time, which is set to 1 second by default." | §6 Task |
| 입력 | "we sample a video clip that ends 1 second before an action starts". 32 frames @ 8 fps, **256×256** (ViT-L/H/g), 384×384 (ViT-g384) | §6, D.1 |
| 모델 | frozen encoder → predictor 가 "the mask tokens corresponding to the frame 1 second into the future" 로 미래 표현 예측 → **encoder 출력과 predictor 출력을 토큰 축으로 concat** | §6 Anticipation Probe |
| probe | 4 transformer block, 16 head, 앞 3개 self-attn + 마지막 cross-attn. cross-attn 의 **query 3개** → verb / noun / action 각각 linear | §6, C.1, D.1 |
| loss | focal (α 0.25, γ 2.0), 세 분류기 독립 합산 | §6, D.1 |
| train | anticipation time **0.25–1.75 s**, anticipation point **0.0–0.25** | D.1, Table 19 |
| val | anticipation time **1 s**, anticipation point **0.0** | D.1, Table 19 |
| anticipation point 정의 | "an anticipation point of 0 means that we predict the representation of the **first frame in the action segment** … 1 means … the last frame" | D.1 |
| 최적화 | 20 epoch, warmup 0, global batch 128, **head 20개** (lr [5e-3 3e-3 1e-3 3e-4 1e-4] × wd [1e-4 1e-3 1e-2 1e-1]), "reporting the accuracy of the **best-performing classifier**", cosine lr | Table 19, C.1 Optimization |
| 지표 | "mean-class recall-at-5 for verb, noun and action on the **validation set** of EK100" | Table 5 |
| 논문이 **안** 적은 것 | spatial 전처리(crop 방식), mask token 번호, 프레임 번호 vs 타임스탬프, 비디오 fps 가 8 의 배수가 아닐 때의 샘플링 | — |

**보고 수치** (Table 5, Frozen Backbone, R@5 verb / noun / action):

| 모델 | verb | noun | action |
|---|---:|---:|---:|
| ViT-L | 57.8 | 53.8 | 32.7 |
| **ViT-H** | **59.2** | **54.6** | **36.5** |
| ViT-g | 61.2 | 55.7 | 38.0 |
| ViT-g384 | 63.6 | 57.1 | 39.7 |

Table 20 (ViT-g384): encoder 만 61.3 / 57.0 / 39.1, predictor 만 48.7 / 34.7 / 20.2, 둘 다 63.6 / 57.1 / 39.7.
레포 `README.md` 에는 ViT-L (32.7) · ViT-g384 (39.7) 의 **학습된 probe checkpoint** 가 공개돼 있다 (§7).

---

## 2. 릴리즈 코드가 논문과 다른 곳 → 우리 선택

전부 실물에서 재계산했다 (2026-09-14, annotation csv + `_meta/*.json` 의 실제 fps).

| # | 무엇 | 릴리즈 코드 | 논문 / 옳은 쪽 | **config 기본** | 스위치 |
|---|---|---|---|---|---|
| A | anticipation point 방향 | `af = sf·ap + (1−ap)·ef − at` → **ap=0 이 action 끝** | ap=0 = action 첫 프레임 | **`paper`** (사용자 결정 09-12) | `data.anticipation_point_mode` |
| B | 구간 경계 | csv 의 `start_frame` 을 비디오 프레임 번호로 그대로 | 타임스탬프 × 실제 fps | **`timestamp`** | `data.time_source` |
| C | 미래 mask token | `predictor.forward` 기본 `mask_index=1` = **0 벡터** (ckpt 에서 1~9 전부 norm 0.0000, [0] 만 0.6706) | 논문 미기재 | **`0`** (학습된 토큰) | `model_kwargs.wrapper_kwargs.mask_index` |
| D | probe head | 20개, 지표는 head 중 max | 20개 sweep best | **1개 (lr 3e-4, wd 1e-2)** (사용자 결정 09-13, lr 은 09-14 에 1e-3 → 3e-4) | `HEADS=sweep` |
| E | val 을 세는 법 | `ipe` 배치만 돌고 모자라면 다시 감는다 → clip 중복·누락 (시뮬레이션 8 GPU 기준 15% 안 셈) | "on the validation set" | **정확히 한 번** (`exact_val.py`) | `evaluation.exact_val_pass` |
| F | spatial | 원본 1080p → 짧은변 292 → 가운데 256 (val); train 은 원본 전체에서 random resized crop | 미기재 | **256×256 재인코딩본 + `short_side`(항등)** | `data.spatial_mode`, `data.base_path` |

### A. anticipation point — action 프레임이 문맥에 들어가는 비율

| 규약 × 경계 | val (9,296 segment) | train (67,217, 한 번 샘플) |
|---|---|---|
| released × frame (= 릴리즈 코드 그대로) | **88.6%**, 평균 14.00 / 32 장 | 69.0%, 10.05 장 |
| released × timestamp | 78.8%, 12.27 장 | 64.4%, 9.13 장 |
| paper × frame | 6.9%, 0.27 장 | 20.1%, 1.64 장 |
| **paper × timestamp (기본)** | **0.1%**, 0.03 장 | 9.2%, 0.77 장 (ap ≤ 0.25 로 의도된 것) |

릴리즈 규약으로 나온 수치는 anticipation 성능으로 읽을 수 없다 (val 의 대부분이 정답 action 을 보고 있다).
val 0.1% 는 영상 시작 1초 안에 action 이 시작해 인덱스가 0 으로 잘린 경우다.

### A-1. 규약을 바꿔 평가하면 — 학습 규약을 벗어나면 이식되지 않는다 (2026-09-20)

`val_metrics.jsonl` 에서 읽은 값. 전부 val 9,296 clip 전수 (`exact_val`), mean class recall@5.

| 체크포인트 (TAG) | 학습 규약 | 평가 규약 | verb | noun | action |
|---|---|---|---:|---:|---:|
| `ek100_vith_official256_released_grid8` (20 epoch, head 8) | released | released | 56.36 | 56.16 | **35.34** |
| `ek100_vith_rel_grid8_valpaper` (위 체크포인트, val 만) | released | **paper** | 26.90 | 35.28 | **14.14** |
| `ek100_vith_official256_paper_grid8` (20 epoch, head 8) | paper | paper | 33.25 | 35.24 | **19.07** |
| `ek100_vitl_rel_resized_mask0` (공개 probe) | released | released | 48.58 | 47.61 | 28.51 |
| `ek100_vitl_paper_resized_mask0` (같은 공개 probe) | released | **paper** | 24.25 | 31.48 | 11.92 |

- **같은 체크포인트에서 평가 규약만 바꾸면 action 이 35.34 → 14.14.** 공개 ViT-L 도 28.51 → 11.92 로 폭이 같다
- paper 규약으로 **처음부터 학습하면 19.07** 이다 (20 epoch 완주, 2026-09-20). 같은 평가에서 released 학습 head 를 가져오면 14.14 → **"released 로 학습하고 paper 로 평가" 는 손해다**
- **가장 큰 수치는 여기다: 같은 모델·같은 probe 인데 문맥에 동작이 보이면 35.34, 안 보이면 19.07.** 릴리즈 규약 수치의 절반 가까이가 "이미 시작된 동작을 보고 있다" 에서 온다.
  (우연은 action recall@5 = 5/1114 = 0.45 이므로 19.07 도 우연의 42 배다. "못 한다" 가 아니라 **문맥이 동작을 보여줄 때와 아닐 때가 두 배 차이** 라는 뜻이다)
- paper arm 의 수렴: 5 epoch 당 +6.58 → +5.44 → **+1.39** 로 마지막 구간에서 꺾였다 (cosine lr → 0). 20 epoch 예산 안에서 대체로 수렴했다
- 왜: released 문맥에는 정답 action 프레임이 val 평균 **14 / 32 장** 들어 있다 (§2-A). 두 규약의 차이는 곧 **action 길이**다 —
  annotation 에서 계산한 val 구간 길이는 중앙값 **1.96 s** · 평균 **3.68 s**, 1 초 미만은 **15.7%** 뿐이다 (train 1.57 / 3.12 s, 25.9%).
  1 초보다 짧은 구간에서만 두 규약이 같은 문맥이 된다
- 예측 대상도 같이 옮겨간다: released 는 action 의 **마지막** 프레임, paper 는 **첫** 프레임 (predictor 는 1 초를 건너뛰고 **한 시점** 을 만든다 — `num_output_frames: 2` = 1 tubelet, 공식 기본값)
- ⚠️ 이 val 은 경계를 `timestamp` 로 환산했고 체크포인트는 `frame_fixfps` 로 학습했다. 규약과 경계 환산이 함께 바뀐다
  (paper 규약에서 프레임 번호를 쓰면 59.94 fps 영상이 최대 2 s 밀려 문맥이 action 안으로 들어간다, §2-B). 규약만의 효과를 보려면 `frame_fixfps` arm 을 더 돌린다 (val 12 분)

재현: `VAL_ONLY=1 HEADS=grid8 TAG=ek100_vith_rel_grid8_valpaper SET="data.anticipation_point_mode=paper data.time_source=timestamp" NODE=vll3 MEM_PER_GPU=30G GPUS=8 bash z_research/anticipation/EK100/sbatch.sh`
(학습된 `latest.pt` 를 새 TAG 폴더에 심볼릭 링크로 걸어 두고 val 만 돈다 — `eval_released_vitl.sh` 와 같은 방식.
⚠️ vll3 는 RealMemory 329 GB 라 `MEM_PER_GPU=40G` (= 320 G) 면 다른 job 20 G 때문에 영원히 대기한다.)

### B. 프레임 번호 vs 타임스탬프 — ⚠️ 이전 README 의 "8개 비디오, 0.9%, 무해" 는 틀렸다

annotation 의 `start_frame` 은 비디오 fps 와 상관없이 **60 fps (50 fps 비디오는 50) 기준**으로 매겨져 있다.
decord 는 실제 fps 로 프레임을 세므로 `start_frame / 실제 fps − start_timestamp` 만큼 시점이 어긋난다:

| 실제 fps | val segment (video) | train segment (video) | 프레임 번호 방식의 시점 오차 (평균 / 최대) |
|---|---|---|---|
| 50.00 | — | 37,455 (201) | 0.006 s / 0.02 s |
| **59.94** | 9,167 (133) | 29,167 (290) | **0.37 s / 2.0 s (val)**, 0.59 s / 3.7 s (train) — 시점이 늦어진다 (영상 뒤로 갈수록 커짐) |
| 29.97 | 63 (3) | 506 (1) | 63 s / 187 s (val), 최대 3,347 s (train) |
| 47.95 | 25 (1) | 89 (3) | 56 s / 134 s |
| 90.00 | 41 (1, P18_09) | — | 37 s / 90 s |

val 로 보면 프레임 번호 방식에서 오차 > 0.5 s 인 segment 가 26.7%, > 1 s 가 10.9% 다. anticipation time 이 1 s 이므로
**문맥이 action 시작을 넘어가는 누수가 생긴다** (위 표 paper × frame 6.9%). 타임스탬프 방식의 오차는 최대 0.03 s.

### C. mask token — 사용자 명령대로 0 을 기본으로 뒀다

논문 수치가 어느 쪽으로 나왔는지는 **모른다** (코드 기본은 1). 공개 수치와 비교하려면 `mask_index=1` arm 을 같이 돌린다.

### 릴리즈 코드 그대로 둔 것 (알고 둔다)

- **프레임 간격** `fstp = int(vfps / 8)`: 실제 문맥 길이가 fps 마다 다르다 — 59.94 fps 는 7장 간격 = **3.74 s** (8.56 fps), 50 fps 3.84 s,
  29.97 fps 3.20 s, 47.95 fps 3.34 s, 90 fps 3.91 s. predictor 는 8 fps (0.125 s/frame) 로 가정한다. 논문의 "8 fps" 와 7% 차이.
- **anticipation step 양자화** `int(at · 8 / 2)` tubelet: train 에서 at 0.25~0.49 s 는 1 step (0.25 s) 로 잘린다. val (1 s) 은 정확.
- **train random resized crop** 은 원본 16:9 전체가 아니라 256×256 재인코딩본 안에서 고른다 (F). 공식과 다르다.
- `resolution`·`frames_per_clip`·probe 구조·loss·lr 스케줄은 공식 config (`configs/eval/vitl/ek100.yaml`) 와 같다. 공식은 64 GPU × batch 2, 우리는 8 × 16 (global 128 동일).

---

## 3. 영상 검증 (2026-09-14, `/data2/local_datasets/EPIC-KITCHENS_resized`)

| 검사 | 결과 |
|---|---|
| 파일 | 700 / 700, 256×256, `_meta` 700개 전부 `ok`, 임시 `.tmp.MP4` 0개 (인코더 로그 `*.tmp.MP4.enc.log` 4개 남음) |
| 헤더 대조 (700개) | 프레임 수 = 원본 `nb_frames` = 재인코딩본 decord 길이, fps 그룹 원본 = 재인코딩본 (59.94 423 · 50 268 · 29.97 4 · 47.95 4 · **90 1**). 원본 avg fps 59.939 vs 재인코딩 59.940 (프레임 간격 `int(fps/8)` 동일) |
| **픽셀 정렬** (decord, dataloader 와 같은 인덱스) | 25개 비디오 × 3 clip × 32장 (복구 4개 + fps 그룹별 표본). 재인코딩 k 번째가 원본 k+d 중 어디와 가장 같은지: 확실히 갈린 906장 중 **d=0 854**, ±1 50 (23 / 27 대칭), ±2 2. 동률 1,173장 중 d=0 이 빠진 동률 21 (1.8%, 움직임 큰 50 fps 비디오에 몰림) |
| decord 임의 접근 (재인코딩본) | ffmpeg 순차 디코드와 **픽셀 차 0.0** (P21_01) |
| 순차 디코드끼리 (ffmpeg) | P07_101 3,000장 중 움직이는 2,869장에서 d=0 2,867 |

→ **프레임 밀림은 없다.** 동률이 많은 것은 원본 성질이다: 59.94 fps 비디오 상당수가 **같은 프레임을 2장씩** 담고 있다
(P21_01 연속 프레임 차가 11.1 / 0.1 / 10.0 / 0.1 … 로 번갈아, P13_06 은 움직이는 프레임의 43%). 재인코딩본도 같은 패턴이다.
이런 비디오의 실제 시간 해상도는 30 fps 급이다 — 공식 경로에서도 같다.

- 복구한 4개 (P29_01 · P29_05 · P30_05 · P30_08) 는 원본에 디코드 불가 패킷이 있어 해당 번호를 앞 프레임으로 채웠다 (2 / 2 / 1 / 16장).
  공식 md5 와 일치하므로 배포본 자체의 성질이다. 깨진 패킷 뒤는 다음 키프레임까지 번짐으로 디코드된다 — val 에서 그 구간이 입력 32장에 걸리는 clip 14개.
- ⚠️ `_info.json` 은 `host: vll6` 이라고 적혀 있지만 **vll5 의 `/data2` 에도 700개가 그대로 있다** (2026-09-14 vll5 에서 위 검사 전부 수행, smoke 도 vll5).
  이전 README 의 "vll6 에만 있다" 는 지금 상태와 맞지 않는다. 다른 노드에서 돌릴 때는 `DRYRUN=1` 이 비디오 존재를 먼저 검사한다.
- 재인코딩 과정의 틀린 진단 기록은 `z_research/scripts/data/build_ek100_resized.py` docstring 에 있다.

---

### 3-1. 공식 val 입력과 같게 만든 세트 (2026-09-14 val · 2026-09-15 train, 지금 경로 `/data2/local_datasets/EPIC-KITCHENS_resized`)

> ⚠️ **경로 변경 (2026-09-15).** 처음엔 `/data2/local_datasets/epic_test` 에 만들었고 (이름과 달리 **validation 138 개**, test 67 개는 정답 비공개라 안 씀),
> train 495 개를 같은 기하로 더한 뒤 사용자가 `EPIC-KITCHENS_resized` 로 옮겼다. **그 자리에 있던 옛 데이터 (§3, 짧은변 256 → 256) 는 지워졌다.**
> 2026-09-15 이전 run (`ek100_vith_lr3e-4`, `ek100_vitl_rel_resized*`, `ek100_vitl_paper_resized_mask0`) 은 옛 데이터 기준이다.
> 옛 데이터 전용 스크립트 (`build_ek100_resized.py`, `verify_ek100_resized_*.py`, `ek100_resized_progress.py`) 는 새 `_meta` 형식을 못 읽는다.
>
> **현재 내용 (2026-09-15, `_meta` 633 개 전수 확인):** train 495 + val 138 전부 검증 통과 (프레임 수·fps 일치, MAD ≤ 2.99).
> 압축: val crf 12 · train crf 15 459 개 / crf 12 34 개 / crf 8 2 개 (P27_105, P28_103 은 crf 12 에서도 MAD 3.30 / 3.10).
> 색 허용치: val 0.5 (P18_08 만 1.0), train 1.0. 앞 프레임으로 채운 프레임: P30_08 289 · P30_05 50 · P29_01 46 · P29_05 36.
> 기하: 1920×1080·1280×720 → 519×292, crop (132, 18) / 1920×1440 (2 개) → 389×292, crop (66, 18).
> 공식 코드가 원본 1080p 에서 만드는 입력과의 차이 (val 3 비디오 × 2 구간, 256² 텐서): 평균 1.7~2.2 / 99% 7~10 (옛 데이터는 평균 13~30 / 99% 97~163).
> 공개 ViT-L probe (released · frame_fixfps · mask 1): 옛 데이터 49.43 / 48.28 / **29.50** → 이 세트 52.43 / 51.45 / **31.01** (verb / noun / action, 9,296 clip). 논문 57.8 / 53.8 / 32.7.
> ⚠️ train 은 가운데 256 만 저장했으므로 random resized crop 이 원본 16:9 전체가 아니라 그 안에서만 일어난다 (val 입력은 공식과 같다).

**왜:** 공개 ViT-L probe 를 공식 조건 (released · 프레임 번호 · mask 1) 으로 256 재인코딩본에서 평가하면
action 29.48 / verb 49.28 / noun 48.22 로 논문 (32.7 / 57.8 / 53.8) 보다 낮았다. 원본 1080p 평가는 이 셸이 학습 job 의
메모리 그룹 안이라 decord worker 가 OOM 으로 죽었다. 재인코딩본에서 두 가지 차이를 찾았다:
- **기하:** 기존 판은 짧은변 256 → 256 crop. 공식 val 은 짧은변 292 (`int(256*256/224)`) → 가운데 256 (물체 1.14 배)
- **색:** 기존 판은 `yuv420p`, 색 태그 없음 (원본은 `yuvj420p` BT.709). 표본 비디오에서 decord 로 되읽은 RGB 가 R·G 약 −2, B +0.5~0.8

**만드는 법** (`z_research/scripts/data/build_ek100_val_official.py`):
- 공식 변환 그대로: decord RGB → `cv2.INTER_LINEAR` 짧은변 292 (1920×1080 → 519×292) → `CenterCrop(256)` (x 132, y 18)
- 인코딩: BT.709 행렬 (정밀 반올림), tv range, bt709 태그, **yuv444p**, crf 12, gop 32.
  4:2:0 은 무손실이어도 되읽은 RGB 가 −0.9 / −1.4 / −0.9 로 치우쳤고 (BT.601 태그로 바꿔도 같음), 4:4:4 는 치우침 0.4 이내
- 프레임: decord 인덱스 k = 출력 k (프레임 수 = 원본 decord 길이). decord 가 못 읽는 프레임은 앞 프레임으로 채움 (val 에서는 P29_05·P30_08)
- 비디오마다 저장 직후 되읽어 프레임 수·fps·색 (표본 8 장, 채널 평균 차 ≤ 0.5, MAD ≤ 3) 을 검사하고 `_meta/<video>.json` 에 기록
- 시험: P09_07 통과 (1,655 / 1,655, 색 +0.38 / +0.07 / +0.20, MAD 1.88)

**쓰는 법:** config 기본값 (`base_path: /data2/local_datasets/EPIC-KITCHENS_resized`, `spatial_mode: short_side`) 그대로. 학습은 `train_vith.sh`, 공개 probe 평가는 `eval_released_vitl.sh`.
(2026-09-14 당시: `data.base_path=/data2/local_datasets/epic_test` + `VAL_ONLY=1`. train 비디오가 없어 `resolve.py` 는 `VAL_ONLY` 일 때 val 비디오만 요구하도록 고쳤다.)

## 4. 레이블 공간

`filter_annotations` 가 train 에 있는 (verb, noun) 조합만 남긴다: train 67,217 segment / 495 video, **verb 97 · noun 289 · action 3,568**;
val 9,668 → **9,296** segment / 138 video (372개 버림). 논문의 "3,568 action / 97 verb / 300 noun" 과 맞는다 (noun 은 289개만 쓰인다).

---

## 5. smoke 결과 (배관 점검, 수치를 읽는 용도 아님)

`GPUS=1 SMOKE=1` (2026-09-14, vll5): 비디오 4개 (train 548 clip / val 580 clip), 1 epoch, 274 iter.
로그에서 확인한 것 — mask token norm `[0]=0.6706, [1..9]=0.0000`, `AnticipativeWrapper grid=(16,16) mask_index=0`,
**exact val `n_clips 580/580`**, `val_metrics.jsonl` 기록, `latest.pt` 저장. 이전 실측 (8 GPU, batch 16): 2.4 s/iter → epoch 525 iter ≈ 21 분, 20 epoch ≈ 7 시간 + val.

---

## 6. 파일

| 파일 | 역할 |
|---|---|
| `configs/ek100_vith.yaml` | **본 설정 하나** (손으로 관리. 생성기 없음) |
| `run.sh` | 진입점. resolve → 빈 DDP 포트 → `evals.main` (포트 충돌 시 조용한 world_size=1 폴백을 `ANT_EXPECT_WS` 로 막는다) |
| `resolve.py` | 덮어쓰기(`SET`/`TAG`/`SMOKE`/`HEADS`) + 검사 (annotation·ckpt·비디오 633개 존재, module·mask 조합, 프레임 예산) + **같은 TAG 다른 설정 이어하기 차단** |
| `sbatch.sh` | SLURM. `ANT_RUN=1` 로 제출/본체 분기, `SET` 은 base64 |
| `monitor.sh` | `watch -n 1` 용 모니터: SLURM 상태, epoch·iter 진행률과 남은 시간, 최신 train 지표, epoch 별 val 과 상수 예측 경고, GPU, 에러 줄 |
| `selftest.py` | CPU 자체 검사 (spatial 모드, 비정사각 wrapper, 정사각에서 원본 module 과 비트 동일, ap 산술, exact val == `ClassMeanRecall`). 2026-09-14 전부 PASS |

`evals/action_anticipation_frozen/` 에서 upstream (45d025f) 대비 바꾼 것 — 전부 **추가 인자**, 인자 기본값은 릴리즈 동작:

| 파일 | 무엇 |
|---|---|
| `epickitchens.py` | `anticipation_point_mode` (A), `time_source` (B) |
| `dataloader.py` | `spatial_mode` · `crop_width` · `train_spatial_mode` 전달 |
| `spatial.py` (새 파일) | `short_side` / `center_crop` / `letterbox` |
| `exact_val.py` (새 파일) | val 정확히 한 번 + all_reduce 한 번 (E) |
| `modelcustom/vit_encoder_predictor_concat_ar_nonsquare.py` (새 파일) | `mask_index` 인자 (C) + 비정사각 grid. 정사각·mask 1 이면 원본과 비트 동일 |
| `eval.py` | 위 인자 전달, `limit_videos`, `ANT_EXPECT_WS` 가드, `evaluation` 블록·`val_metrics.jsonl`, rank≠0 로그 끔 (2026-09-14: GPU 수만큼 같은 줄이 찍히던 것) |

영상 검증 스크립트: `z_research/scripts/data/verify_ek100_resized_frames.py` (§3 픽셀 정렬).

---

## 7. 아직 안 한 것

### ⚠️ lr 1e-3 은 학습이 무너졌다 (2026-09-14, TAG `ek100_vith`)

| val mean class recall@5 | verb | noun | action |
|---|---:|---:|---:|
| epoch 1 / 2 / 3 | 6.85 / 6.85 / 6.85 | 2.46 / 2.48 / 2.48 | 0.46 / 0.45 / 0.45 |
| 모든 clip 에 같은 top-5 (계산값: 5 / val 에 나온 클래스 수 73 · 201 · 1,114) | 6.85 | 2.49 | 0.45 |

- 영상 (clip 마다 다름), encoder·predictor 특징 (clip 간 토큰 std 1.84, NaN 없음), gradient (모든 층, GradScaler 건너뛴 step 0 / 1,575) 는 정상.
- 무너진 곳은 probe 몸통: 학습된 pooler 출력의 clip 간 차이 0.019 (새로 초기화한 probe 0.168). residual 가지 가중치가 줄었다 (attn.proj 크기 19.7 → 14.9 → 13.0, 초기 약 25.6).
- 시점: train recall verb 가 10 iter 7.9 → 80 iter 5.1 로 떨어진 뒤 그대로. warmup 없는 lr 1e-3 이 큰 것으로 판단 (가설, 짧은 lr 비교는 하지 않았다).
- 논문은 head 20개 (lr 5 × wd 4) 중 best 를 보고하므로 무너지는 lr 이 섞여도 드러나지 않는다. head 하나로 고정하면 이 안전장치가 없다.
- → 기본 lr 을 **3e-4** 로 바꾸고 TAG `ek100_vith_lr3e-4` 로 다시 제출 (사용자 결정). 첫 epoch val 이 위 상수값에서 벗어나는지부터 확인한다.

#### ⚠️ 2026-09-20 재확인 — lr 1e-3 · 3e-3 은 **새 데이터·head 2 개에서도** 첫 epoch 에 붕괴한다

grid8 두 arm 에서 최적이 lr 격자 위쪽 끝 (3e-4) 에 붙어 있어 (released: 3e-4 넷이 1e-4 넷을 4.9 pt 차로, paper: 2.7 pt 차로 전부 이김)
한 칸 위를 밟아 보았다 — TAG `ek100_vith_official256_paper_hi2`, **lr {1e-3, 3e-3} × wd 1e-2, warmup 0 (논문대로), val 매 epoch**, vll3 8 GPU.

| head | action | verb | noun |
|---|---:|---:|---:|
| lr 1e-3 wd 1e-2 | **0.45** | 7.14 | **2.49** |
| lr 3e-3 wd 1e-2 | **0.45** | 5.48 | **2.49** |
| (붕괴 상수 = 모든 clip 에 같은 top-5) | 0.45 | 6.85 | 2.49 |

epoch 1 val 에서 둘 다 상수값이라 job 215141 을 취소했다 (16 시간 절약). **결론: warmup 0 에서 우리가 쓸 수 있는 lr 상한은 3e-4 다.**
논문이 lr 5e-3~1e-4 5 종을 쓰고도 괜찮은 것은 20 head 중 best 를 보고하기 때문이다 (붕괴한 head 는 조용히 버려진다) — head 수가 적으면 그 안전장치가 없다.
남은 수렴 지렛대는 lr 이 아니라 **epoch 예산**이다 (paper arm 은 20 epoch 에서도 5 epoch 당 +5 pt 로 오르는 중, §A-1).
다시 시도하려면 warmup 을 1~2 epoch 넣어야 하고 그건 논문 optimizer 설정에서 벗어나므로 **별도 arm 으로 적는다**. 격자는 `resolve.py --heads hi2`.


- ❌ 본 학습 (20 epoch). 결과가 나오면 **마지막 epoch** 값을 Table 5 ViT-H (59.2 / 54.6 / 36.5) 옆에 두되, §2 의 A·B·C·D·E·F 를 같이 적는다.
  우리 쪽이 더 어렵다 (A: 누수 0.1% vs 88.6%, D: head 1개 vs 20개 중 best). "재현 실패" 로 읽지 말 것 (CLAUDE.md §1-3).
- ✅ **논문이 실제로 어느 규약으로 돌렸는지 확인 (2026-09-20 완료).** 공개 ViT-L probe 가 released 규약에서 action **31.01** (논문 32.7), paper 규약에서 **11.92**.
  → 논문 수치는 공식 코드 그대로의 규약이다. §A-1.
- ❌ encoder 만 (`no_predictor=true`) arm — "predictor 가 기여하는가" (논문 Table 20 은 +0.6 action).

---

## 8. 기록 — 정정한 서술 (다시 쓰지 말 것)

| 이전 서술 | 정정 |
|---|---|
| "annotation 프레임 번호와 fps 가 어긋나는 비디오는 8개 (590 segment, 0.9%), 59.94 fps 는 오차 0.1% 라 무해" | 59.94 fps 도 영상 길이에 비례해 최대 2 s (val) / 3.7 s (train) 어긋나고, 90 fps 비디오 P18_09 가 val 에 있다. 기본값을 `timestamp` 로 바꿨다 (§2-B) |
| "EK100 해상도 3종 (1080p 694개 등)" 표만 있고 fps 는 59.94/50 두 종류로 서술 | fps 는 5종: 59.94 · 50 · 29.97 · 47.95 · 90 |
| "재인코딩본은 vll6 에만 있다" | 2026-09-14 vll5 `/data2` 에 700개 있음 확인 |
| "기본 `center_crop`, `short_side` 는 SET 으로" + `make_configs.py` 생성기 + `ek100_smoke.yaml` | config 한 파일에 결정된 설정을 기본으로 박았다. smoke 는 `SMOKE=1`, sweep 은 `HEADS=sweep` |
| "ViT-H 비교 수치가 레포에 없다, PDF 가 없다" | PDF 가 레포 루트에 있다. ViT-H 는 59.2 / 54.6 / 36.5 |

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
python z_research/anticipation/EK100/selftest.py                                   # CPU, 몇 초
DRYRUN=1 GPUS=8 bash z_research/anticipation/EK100/run.sh
GPUS=1 SMOKE=1 bash z_research/anticipation/EK100/run.sh
python z_research/scripts/data/verify_ek100_resized_frames.py --jobs 16 --out /tmp/ek100_frames.json   # §3 (~10분, NFS 원본 읽음)
```
§2 의 누수·시점 오차 표는 `epic-kitchens-100-annotations/EPIC_100_{train,validation}.csv` 와
`/data2/local_datasets/EPIC-KITCHENS_resized/_meta/<video>.json` 의 fps 로 `epickitchens.decode_videos_to_clips` 와 같은 산술을 numpy 로 돌려 얻었다 (2026-09-14 세션).
데이터 원본 `/data/dataset/EPIC-KITCHENS` (NFS, 1.2 T) · annotation `/data/hyuntak/project/2026/2027_cvpr/epic-kitchens-100-annotations` · ckpt `configs/protocols/models.md` 의 `vith`.
