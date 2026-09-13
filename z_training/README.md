# z_training — frozen encoder 위에서 predictor 학습

> **2026-09-10 결정: 릴리즈 ViT-H predictor 에서 post-FT, 데이터는 IntPhysGen v11 의 block 단위 train 절반(가능만),
> 채점은 test 절반(가능+불가능, 10,752 쌍).** 그게 `v11_postft` 다. 대조군 `v11_scratch`. 분할은 §3.

```bash
# 출발선은 다시 돌리지 않는다 — 기존 v11 채점의 per_block.json 에서 test 절반만 골라 낸 값: **75.83%** (10,752 쌍)
#   z_training/runs/release_vith/eval/v11_split_test_from_existing_runs.json  (같은 C16/T32 프로토콜)
GPUS=8 bash z_training/train.sh v11_postft                      # post-FT (epoch 마다 e{N}.pt, predictor 만)
GPUS=8 bash z_training/eval.sh v11_postft v11_split_test        # held-out 채점 (CKPT=e3.pt 로 epoch 선택)
GPUS=8 bash z_training/train.sh v11_scratch                     # 대조군
```

그 밖의 일반 명령 (다른 데이터를 쓸 때는 `SET="data.datasets=[이름]"` 으로 준다):

```bash
bash z_training/train.sh --list                                  # config · 데이터셋 후보 · run 목록
DRYRUN=1 SET="data.datasets=[<이름>]" bash z_training/train.sh frozen_predictor_scratch   # 병합·검사만 (몇 초)
GPUS=2 bash z_training/train.sh --smoke-ddp                      # 모델·데이터 없이 spawn/포트/join 점검
LIMIT=16 SET="data.datasets=[<이름>]" DEBUG=1 bash z_training/train.sh debug             # 배관 점검 (GPU 1장, ~4분)
GPUS=8 SET="data.datasets=[<이름>]" bash z_training/train.sh frozen_predictor_scratch    # predictor 새로 학습
GPUS=8 SET="data.datasets=[<이름>]" bash z_training/train.sh frozen_predictor_postft     # 릴리즈 predictor post-FT
GPUS=8 SET="data.datasets=[<이름>]" bash z_training/sbatch.sh frozen_predictor_scratch   # SLURM (vll5, 8 GPU, 24h)
GPUS=8 bash z_training/eval.sh frozen_predictor_scratch v11      # 채점 (= run.sh surprise_c16t32 v11)
```

| 파일 | 역할 |
|---|---|
| `train.sh` | 진입점. 병합·검사 → DDP 포트 탐색 → 스레드 제한 → `harness/launch.py` |
| `harness/launch.py` | rank 프로세스 spawn (app/main.py 의 흐름 + 포트/timeout/join/split-brain 가드). **app/main.py 는 안 건드린다** |
| `sbatch.sh` | 위를 SLURM 으로. `TRAIN_RUN=1` 로 제출/본체가 갈린다 (`z_research/scripts/sbatch.sh` 와 같은 구조) |
| `eval.sh` | 학습한 predictor 를 **기존 채점 하네스** 로 잰다 (`model.predictor_checkpoint`) |
| `harness/resolve_train.py` | config 병합기 (`extends`, 레지스트리, `--set`, 실물 검사) |
| `harness/export_ckpt.py` | predictor-only ckpt → 릴리즈 형식 통짜 파일 (models.md 등록·공유용, 평소 불필요) |
| `harness/extract_predictor.py` | 릴리즈 model.pth 의 predictor 만 떼어 `runs/release_vith/latest.pt` 로 (출발선 채점용, 89 MB) |
| `data/build_v11_split_index.py` | v11 block 단위 train/test 분할 (§3) |
| `data/build_intphys1_train_index.py` | IntPhys1 train 인덱스 (`data_csv/intphys1_train/index.csv`) |
| `runs/<NAME>/` | `config.yaml` `latest.pt` `e{N}.pt` `train.log` `metrics.jsonl` `log_r*.csv` `eval/` (gitignore) |

학습 코드는 `app/vjepa_frozen/` (`train.py` 루프 · `data.py` 데이터/마스크 · `utils.py` 모델/옵티마이저 ·
`val.py` in-loop 채점). config 스키마는 `configs/training/README.md`.

---

## 1. 무엇을 하는가

```
릴리즈 model.pth ──encoder(online)────────► context_encoder  (frozen)
                 ──target_encoder(EMA)────► target_encoder   (frozen)
                 ──predictor ──(post-FT 만)► predictor        (학습)   scratch 는 새로 초기화

clip (32f, stride 3)  ─► target_encoder ─► LN ─► h[future]        ─┐
                      ─► context_encoder(앞 16f 토큰만) ─► z ─► predictor ─► p ─┤ loss = mean|p − h|
```

- 손실·토큰 배치·역할 분리가 **`surprise_c16t32` 채점과 같다.** 학습한 predictor 는 구조도 릴리즈와
  같아서 `run.sh surprise_c16t32 / intphys1_sliding / attn_probe` 에 그대로 들어간다.
- encoder 두 벌은 `no_grad` 라 메모리는 predictor 것만 든다 (rank 당 batch 8 이 24 GB 에 여유 있게 든다 — debug 실측 §5).
- 체크포인트는 **predictor 만** (`latest.pt`, ~90 MB + Adam). encoder 는 `base_checkpoint` 경로로 가리킨다.
- `val:` 블록이 있으면 epoch 마다 작은 matched-pair 채점을 찍는다 (수렴 감시용, 표본 작음).

## 2. 두 모드

| | scratch | post-FT |
|---|---|---|
| config | `frozen_predictor_scratch` | `frozen_predictor_postft` |
| predictor 초기값 | 새로 (zero-init mask token) | 릴리즈 `predictor` 를 strict 로드 |
| lr (미검증) | 3e-4, warmup 2 ep, 30 ep | 5e-5, warmup 1 ep, 10 ep |
| 출발점 확인 | — | `val.at_start` 가 epoch 0 에 릴리즈 predictor 점수를 먼저 찍는다 |

옵티마이저는 둘 다 새로 만든다 (릴리즈 ckpt 의 `opt` 는 world_size 256, encoder 포함 파라미터 그룹이라 못 쓴다).

## 3. 데이터

**v11 block split** (`z_training/data/build_v11_split_index.py`, `data_csv/intphysgen_v11_split/`):
v11_full 12조건 10,752 block 을 (condition × violation_type × sym_k) 117 셀에서 층화해 seed 0 으로 50/50.
train 절반은 가능 변이만(`index_train.csv`, 10,752 clip) → `v11_split_train`; test 절반은 4 변이 전부
(`index_test.csv`, 21,504 clip = 10,752 matched pair) → `configs/protocols/datasets.md ## v11_split_test`.
검증: block 4행 같은 split, pair 1+1, train/test block 겹침 0 (`split_report.json`). `--holdout-shape/--holdout-color` 로
안 본 외형 block 을 통째로 test 에 보낼 수 있다 (`index_test_holdout.csv`).
⚠️ 여전히 v11/v11_full **전체** 채점과는 문맥을 공유한다. 학습 후 점수는 `v11_split_test` 로만 읽는다.

그 밖의 후보 (`configs/training/datasets.md`):

| 이름 | clip | 주의 |
|---|---:|---|
| `intphys1_train` | 3,750 | IntPhys 2019 train 분할(가능만). 채점 세트와 문맥을 안 나누는 유일한 로컬 세트. **인덱스 미생성** (`data/build_intphys1_train_index.py`) |
| `rollout_v2_training` | 896 | 무중력 직선 운동만 |
| `rollout_v2_possible` | ~4.6k | 법칙 7종. 위치 readout 결과와 겹친다 |
| `v11_possible` | 10,752 | ⚠️ **v11 채점과 문맥이 픽셀 단위로 같다.** v11 점수는 학습셋 점수가 된다 |
| `intphys2_train_mp4` | 808 | mp4 경로 (`video_csv`). 미검증 |

- **가능 변이만 쓴다** (`include: {plausible: ["1"]}` 기본). 불가능 변이는 위반을 정상으로 가르친다.
- 시간 jitter: `frames_start_choices` 에서 시작 프레임을 샘플마다 고른다 (v11 은 저장 프레임이 0,3,…,99 라 0/3/6 만).
- 자연 영상(K710·EK100)은 로컬에 없다. EK100 은 `/data/.../ek100_download` 로 받는 중(1.2 TB, NFS).
  `video_csv` 타입으로 붙이면 되지만 NFS 디코딩 속도는 미측정.

## 4. 채점 연결

```bash
GPUS=8 bash z_training/eval.sh <run> v11                    # surprise_c16t32, matched, 10,752 쌍
GPUS=8 bash z_training/eval.sh <run> intphys1_dev intphys1_sliding
CKPT=e10.pt GPUS=8 bash z_training/eval.sh <run> v11
```
- `analysis/intphys2/model.py` 에 `model.predictor_checkpoint` 를 추가했다. encoder 는 `models.md` 의
  base(`vith`)에서, predictor 만 run 파일에서 읽는다. 빠진 키가 있으면 죽는다.
- 결과는 `<run>/eval/<프로토콜>__<데이터셋>_<ckpt>/summary.json`. 기본 `results_root` 와 캐시 TAG 를 건드리지 않는다.
- 릴리즈 기준선: `GPUS=8 bash z_research/scripts/run.sh surprise_c16t32 v11 vith` → 73.37% (CLAUDE.md §5-1).
- 통짜 체크포인트가 필요하면 `python z_training/harness/export_ckpt.py <run>` (5 GB, NFS 90 초).

## 5. 함정 · 검증

- **기존 코드는 안 고쳤다.** 런처는 `harness/launch.py` 로 따로 뒀다 (DDP 포트 `TRAIN_DDP_PORT`, NCCL timeout,
  rank join + exitcode, `TRAIN_EXPECT_WS` 가드). 기존 파일에 손댄 곳은 둘뿐이고 둘 다 **새 키가 있을 때만** 도는 분기다:
  `analysis/intphys2/model.py` 의 `model.predictor_checkpoint`, `z_research/scripts/harness/resolve.py` 의 그 키 실물 검사 한 줄.
- `app/vjepa/train.py` 의 import 시 `CUDA_VISIBLE_DEVICES = SLURM_LOCALID` 트릭은 **넣지 않았다** —
  단일 task 안에서 프로세스를 spawn 하는 우리 방식에서는 모든 rank 를 GPU 0 으로 몰아 버린다.
- DDP: predictor 는 mask token 10개 중 1개만 쓰므로 `find_unused_parameters=True`.
  `TRAIN_EXPECT_WS` 로 world_size 를 확인해 조용한 split-brain 을 죽인다.
- `latest.pt` 는 `.tmp` 에 쓰고 rename 한다. 같은 NAME 으로 다시 부르면 이어서 돈다.
- **rank 당 CPU 스레드** 를 `OMP_NUM_THREADS = min(12, nproc/GPUS)` 로 제한한다 (CLAUDE.md §7-1).
- 인덱스 csv 는 `.gitignore` 대상 (`*csv`) — `build_intphys1_train_index.py` 로 다시 만든다.
- **멈추기: `pkill -f "launch.py --fname .*<run 이름>"`** — 런처가 SIGTERM 을 받으면 rank 8개를 같이 정리한다.
  rank 프로세스는 cmdline 이 `python -c from multiprocessing.spawn ...` 이라 폴더 이름으로는 안 잡힌다.
  `<run>/pids.txt` 에 런처+rank PID 가 있다. 런처가 SIGKILL 로 죽어도 rank 는 PR_SET_PDEATHSIG 로 따라 죽는다.
  (2026-09-10 사고: 런처만 죽여 rank 8개가 고아로 GPU 를 잡고 같은 폴더에 계속 썼다. 항상 `nvidia-smi` 로 확인할 것.)
- 배관 실측 (2026-09-10, 16 clip · 2 epoch × 4 step · GPU 1장, 산출물은 지웠다): ViT-H 로딩 ~2분, step 당
  GPU 1.1 s (batch 2, 아직 워밍업), 최대 메모리 4.7 GB, loss 0.78 → 0.69, latest.pt/e{N}.pt 저장·val 정상.
  `--smoke-ddp` 로 런처(spawn/포트/join)만 따로 점검할 수 있다.

## 6. 다음 (PAPER_STORY_2026-09-06 beat 6·7)

이 하네스는 **마스크 토폴로지**(`temporal_prefix` vs 릴리즈 `block3d`) 축을 이미 갖는다.
아직 없는 것: (1) **다단계 예측** — predictor 출력을 문맥으로 되먹여 다음 구간을 예측하는 rollout 손실
(`app/vjepa_droid/train.py` 의 `auto_steps` 방식), (2) **시간 인과 attention** — 미래 토큰이 뒤 토큰을
못 보게 하는 attn_mask (`src/models/utils/modules.py` 의 `attn_mask` 인자로 붙일 수 있다).
둘 다 `app/vjepa_frozen/train.py` 의 `train_step` 과 `src/models/predictor.py` 만 건드리면 된다.
하네스 5번 — "그냥 더 학습시킨 것 아닌가" 를 막는 ablation 은 `temporal_prefix` 단독 vs `block3d` 단독 vs
혼합이 첫 줄이다.
