# z_AC_training — V-JEPA 2-AC (action-conditioned predictor) 학습 파악

> **2026-09-19 개설. 아직 아무것도 돌리지 않았다.** 이 문서는 릴리즈 코드 (`app/vjepa_droid/`, `src/models/ac_predictor.py`,
> `configs/train/vitg16/droid-256px-8f.yaml`) 와 논문 §3 (V-JEPA 2 PDF, `pdftotext` 760–960 줄) 을 읽고 정리한 것이다.
> 수치는 config·코드에서 직접 읽었고, 코드로 확인하지 못한 것은 "미검증" 으로 적었다.
>
> **결론 3줄**
> 1. AC 학습 = **frozen ViT-g encoder** 위에서 **새 predictor (300M, frame-causal)** 를 DROID 로봇 영상 + end-effector 상태로 학습.
>    손실 = teacher forcing L1 (다음 프레임) + 2-step rollout L1. 릴리즈 predictor 는 **로드하지 않는다** (`load_predictor: false`).
> 2. **로컬에 없는 것:** DROID 데이터, ViT-g 체크포인트 (`vitg.pt`), AC 체크포인트 (`vjepa2-ac-vitg.pt`). 로컬 ckpt 는 ViT-H / ViT-L 뿐.
> 3. **코드 ≠ 논문 3곳:** clip 길이 (8f vs 16f), augmentation (사실상 고정 center crop), rollout 손실 구성. §4.
>
> **2026-09-20 추가:** 평가(planning) 프로토콜 정리 + Jongseo RoboCasa AC 실험 검토 → [`Archive/PLANNING_AND_ROBOCASA_2026-09-20.md`](Archive/PLANNING_AND_ROBOCASA_2026-09-20.md)
>
> **미결 1줄:** 이 폴더에서 **무엇을** 학습할지 (DROID 재현 / ViT-H 로 바꾼 AC / action 없는 우리 물리 데이터에 AC 구조만) — §6.

---

## 0. 폴더 양식 (계획 — `z_training/` 과 같은 모양)

| 파일 | 역할 | 상태 |
|---|---|---|
| `README.md` | 시작점 (이 문서) | ✅ |
| `train.sh` / `sbatch.sh` / `eval.sh` / `monitor.sh` | 진입점. `z_training/` 것을 그대로 본뜬다 (DDP 포트·`TRAIN_EXPECT_WS`·스레드 제한·`TRAIN_RUN=1`) | ⏳ §6 결정 후 |
| `harness/launch.py` | rank spawn + join. **`app/vjepa_droid/train.py` 는 안 고친다** (§5 의 `SLURM_LOCALID` 트릭 때문에 우회 필요) | ⏳ |
| `harness/resolve_train.py` | config 병합 (`extends`·레지스트리·`SET`·실물 검사) — `z_training/harness/` 재사용 가능 | ⏳ |
| `data/` | DROID 경로 csv 생성 (`droid_train_paths.csv`, 한 줄 = trajectory 폴더) | ⏳ |
| `runs/<NAME>/`, `slurm_logs/` | 산출물 (gitignore) | — |

config 는 `configs/training/` 가 아니라 원 레포 관례대로 `configs/train/<arch>/droid-*.yaml` 이 정본이다.
디버그용은 이미 있다: `configs/train/vitl16/droid_ac_debug.yaml` + 더미 데이터 `dummy_droid_data/` (`z_scripts/gen_dummy_droid.py`).

---

## 1. 무엇을 하는가

```
DROID trajectory ─ mp4 (외부 카메라 1개, 4 fps, 8 장) ─► target_encoder (frozen, 프레임마다 따로) ─► LN ─► z_1..z_8
                 └ trajectory.h5 ─ end-effector 상태 s_k (7D) ─► a_k = s_{k+1} − s_k  (7D, 7개)

predictor 입력 (프레임마다 끼워 넣기):  [a_1, s_1, z_1(256 tok)] [a_2, s_2, z_2] … [a_7, s_7, z_7]
block-causal mask: 프레임 k 의 토큰은 프레임 ≤ k 의 a·s·z 전부를 본다
출력: 각 프레임 자리 → 다음 프레임 ẑ_{k+1}     (a·s 토큰 자리는 버림)

jloss (teacher forcing) = mean| ẑ_{2..8} − z_{2..8} |
sloss (rollout)         = mean| [ẑ_2 , ẑ_3^AR] − z_{2..3} |      ẑ_3^AR = P([z_1, ẑ_2], a_{1:2}, s_{1:2})
loss = jloss + sloss
```

| 항목 | 값 | 출처 |
|---|---|---|
| encoder | ViT-g (`vit_giant_xformers`), **context·target 둘 다 `target_encoder` 키** | config `meta.context_encoder_key/target_encoder_key` |
| encoder 학습 여부 | **사실상 frozen** — optimizer 에는 들어가지만 forward 에서 `encoder` 를 안 쓴다 (target_encoder 만, `no_grad`). grad 가 None 이라 AdamW 가 건너뛴다. EMA 갱신도 없다 | `train.py` `forward_target`, `utils.py` `init_opt` |
| 프레임 인코딩 | 각 프레임을 2번 복제해 tubelet 2 를 채움 → **이미지 인코더처럼** 프레임마다 256 토큰 | `train.py` `.repeat(1,1,2,1,1)` |
| predictor | `vit_ac_predictor` depth 24, dim 1024, heads 16, GELU, 3D-RoPE (a·s 토큰은 시간 축 RoPE 만) | config `model`, `ac_predictor.py`, 논문 §3.1 |
| action / state | 7D = xyz · Euler xyz · gripper. action 은 **회전은 상대 회전행렬 → Euler**, 나머지는 차분 | `droid.py` `poses_to_diffs` |
| extrinsics | 읽기는 하지만 `use_extrinsics: false` 라 predictor 에 안 들어감. `camera_frame: false` → 상태는 로봇 base 기준 | config, `droid.py` |
| 손실 | L1 (`loss_exp 1.0`), `normalize_reps: true` (target 과 예측 출력 둘 다 LN) | config `loss` |
| 정밀도 | bf16 autocast | `meta.dtype` |
| 최적화 | AdamW, WSD 스케줄 (warmup 15 ep → 상수 4.25e-4 → anneal 15 ep → 0), wd 0.04 | config `optimization` |
| 규모 | 4 노드 × 8 GPU × batch 8 = **global 256**, `ipe 300` × **315 epoch = 94,500 step ≈ 24.2M clip** | config |
| 초기화 | encoder ← `vitg.pt`, predictor **새로** (`load_predictor: false`) | config `meta` |

**우리 연구와의 연결** — 이 predictor 는 `z_training/README.md` §6 에서 "아직 없는 것" 이라 적은 두 가지
(**다단계 rollout 손실**, **시간 인과 attention**)를 이미 갖춘 구현이다. 또 `PAPER_STORY_2026-09-06.md` 가
"AC 는 pretrained predictor 를 버리고 새로 학습했다" 를 beat 7 의 근거로 쓰고 있다.

## 2. 데이터 — DROID

| 항목 | 값 | 출처 |
|---|---|---|
| 데이터셋 | DROID (Khazatsky et al., 2024), Franka Panda 팔 원격조작, 외부 고정 카메라 | 논문 §3 |
| 사용량 | 영상 **4 초 미만 버림 → 62 시간 미만**. 성공/실패·과제 라벨 안 씀 (비교 실험은 23k trajectory) | 논문 §3.1, §4 |
| 포맷 | **raw 포맷** (RLDS 아님): trajectory 폴더마다 `*.json` 메타 + `trajectory.h5` + `recordings/MP4/<카메라시리얼>.mp4` | `droid.py` `loadvideo_decord` |
| h5 에서 읽는 키 | `observation/robot_state/cartesian_position` [T,6], `.../gripper_position` [T], `observation/camera_extrinsics/<시리얼>_left` | `droid.py` |
| 카메라 | `camera_views: [left_mp4_path]` — 메타 json 의 그 키로 mp4 를 찾는다. **실제 DROID 메타 키 이름과 맞는지 미검증** (config 주석이 GitHub issue #65 를 가리킨다) | config |
| 입력 목록 | `datasets: [droid_train_paths.csv]` — 한 줄에 trajectory 폴더 하나, 헤더 없음 | config, `droid.py` |
| 시간 샘플링 | 영상 원 fps (DROID 15 fps 로 알려짐, 미검증) 에서 `ceil(vfps/4)` 칸 간격 8 장 = **약 2 초 창** 을 무작위 위치에서 | `droid.py` |
| 해상도 | 256 × 256 | config `crop_size` |

**로컬 상태 (2026-09-19 확인):** DROID 없음 (`/local_datasets`, `/data2` 모두). 디스크 여유 `/` 130G · `/data2` **99G (99%)** →
DROID 를 받는다면 NFS `/data` 밖에 자리가 없다 (쓰기 57MB/s, 디코딩 속도 미측정). 

**DROID 배포본 크기** (공식 페이지 droid-dataset.github.io, 2026-09-19 확인):

| 버전 | 크기 | 경로 | AC 코드가 읽나 |
|---|---:|---|---|
| RLDS 전체 | 1.7 TB | `gs://gresearch/robotics/droid` | ❌ (TFRecord, `droid.py` 는 raw 폴더를 읽는다) |
| RLDS 샘플 100 ep | 2 GB | `gs://gresearch/robotics/droid_100` | ❌ |
| raw (stereo HD MP4 + SVO 포함) | 8.7 TB | `gs://gresearch/robotics/droid_raw` | ✅ |
| raw MP4-only (stereo·SVO 제외) | 5.6 TB | 같은 버킷, rsync 필터 | ✅ **최소 필요분** |

- 공식 페이지 주의: raw 는 face-blur 복사 때 **에피소드 약 20% 가 빠졌다** (RLDS 는 완전).
- AC 는 외부 카메라 1 개 mp4 + `trajectory.h5` + 메타 json 만 쓴다 → 카메라 3 대 중 1 대만 골라 받으면 **약 2 TB 로 추정** (5.6/3, 미검증).
  논문이 쓴 62 시간 부분집합만이면 더 작다 (수백 GB 추정, 미검증). 둘 다 메타 json 을 먼저 받아 파일 목록을 만들어야 가능.
- 메타 json 의 카메라 키 이름 (`left_mp4_path` 가 실제로 있는지) 은 여전히 미검증 — json 몇 개만 먼저 받아 확인할 것.

**체크포인트:** `checkpoint/` 에 ViT-H·ViT-L HF 캐시뿐. `vitg.pt`, `vjepa2-ac-vitg.pt` 없음 (README 의 다운로드 링크).

## 3. 계획 (추론) 쪽 — 학습된 AC 를 쓰는 법

`notebooks/utils/world_model_wrapper.py` + `mpc_utils.py` (`energy_landscape_example.ipynb`):
현재 프레임·목표 이미지를 각각 encoder 로 인코딩 → CEM 으로 action 열을 샘플 → predictor 로 rollout →
`|ẑ_T − z_goal|` L1 최소 top-k 로 평균/분산 갱신 → 첫 action 만 실행 (receding horizon).
rollout 중 상태는 `compute_new_pose` 로 action 을 적분해 만든다 (학습 때는 **진짜 상태**를 넣는다 — 차이 주의).
기본 `rollout 2, samples 400, topk 10, cem_steps 10, maxnorm 0.05`, 회전 action 은 0 으로 고정.

**실제 로봇 실험은 `planning horizon = 1` 이다** (논문 부록 B.2: 800 samples · 10 refinement · horizon 1,
"tasks are relatively greedy … short planning horizon to be sufficient". Table 3 의 Cosmos 대조도 horizon 1).
노트북 기본값 `rollout 2` 는 energy landscape 예시용이다. pick-and-place 는 긴 horizon 대신 **하위목표 이미지 3 장**
(4 step → 10 step → 4 step) 으로 끊는다.

**따라서 배포 시 AC 가 실제로 쓰는 능력은 두 가지로 쪼개진다** (해석, 우리가 측정한 것 아님):

1. **action 조건부 1-step 전이** — eq (5) 는 `P(â_{1:T}; s_k, z_k)`, 즉 **현재 프레임 1 장 + 현재 end-effector 상태**에서
   시작한다. 영상 문맥에서 운동 상태를 읽어낼 필요가 거의 없다 (고정 카메라·느린 팔). 대신 action → latent 변화 사상이
   **실제 물리와 정합**해야 한다. 내부적으로만 일관되고 현실과 어긋나면 CEM 은 여전히 에너지 최소값을 찾지만 팔은 엉뚱한 데로 간다.
2. **latent 거리가 쓸 만한 에너지인가** — `|ẑ − z_goal|` 이 목표에 가까워질수록 줄어야 하고, 그 값이 팔·물체 위치에
   지배돼야 한다. 이건 예측 정확도와 **다른 성질**이고, 우리 논문이 재는 "거리로 재면 무엇이 드러나는가" 와 같은 층이다.

**AC 평가는 "미래가 물리적으로 그럴듯한가" 를 한 번도 묻지 않는다** — "어떤 action 이 latent 거리를 줄이는가" 만 묻는다.
우리 surprise 채점과 재는 축이 다르다.
⚠️ Jongseo RoboCasa 실험의 "AC 가 action 을 안 읽는다" 는 **인용 불가** (공식 ckpt 버그·proprio 불일치·zero-shot).

## 4. 코드 ≠ 논문 (코드에서 확인)

| 항목 | 논문 §3.1 | 릴리즈 코드/config | 확인 |
|---|---|---|---|
| clip 길이 | 4 초, 16 장, T=15 | `dataset_fpcs: [8]` @ 4 fps → **8 장 (약 2 초)**, T=7 | config |
| augmentation | random-resize-crop, 종횡비 (0.75, 1.35) | `random_resize_scale: [1.777, 1.777]` → 면적이 원본보다 커서 **10 번 다 실패 → 항상 center crop** (16:9 입력이면 종횡비 1.35 고정, 1280×720 → 972×720). **무작위성 0** | `_get_param_spatial_crop` 2000 회 실측, 결과 1 종 |
| rollout 손실 | `‖P(a_{1:2}, s_1, z_1) − z_3‖` 하나 | `z_ar = [ẑ_2(TF), ẑ_3(AR)]` 둘 다 → ẑ_2 가 jloss·sloss 에 **두 번** 들어간다 | `train.py` `forward_predictions` |
| encoder | frozen | optimizer 에 있지만 grad 없음 → 결과적으로 frozen | `train.py` |

## 5. 함정 (돌리기 전에)

- **`app/vjepa_droid/train.py` import 시 `CUDA_VISIBLE_DEVICES = SLURM_LOCALID`** — `z_training` 에서 이미 한 번 밟은 것
  (단일 task 에서 spawn 하면 모든 rank 가 GPU 0). 원 진입점 `app.main` 으로 돌리면 SLURM 안에서 이 문제가 난다. 런처를 따로 둬야 한다.
- 원 config 는 **32 GPU (4 노드)** 기준. vll5 8 GPU 면 global batch 64 → lr 을 줄이거나 grad accumulation (코드에 없음).
- ViT-g 는 `mem_per_gpu 220G`, `use_activation_checkpointing: true` 로 돌았다. 우리 GPU 메모리에서 ViT-g + 300M predictor 가 드는지 **미측정**.
- `load_pretrained` 는 `strict=False` — 키가 안 맞아도 조용히 넘어간다. 로드 메시지 (`missing/unexpected`) 를 반드시 볼 것.
- 데이터 로더는 실패한 영상을 **무작위 다른 영상으로 조용히 대체**한다 (`__getitem__` while 루프). 경로가 전부 틀려도 무한 루프만 돈다.
- 체크포인트는 encoder + target_encoder + predictor + opt 통째로 저장 (ViT-g 면 수십 GB 추정). `z_training` 처럼 predictor 만 저장하도록 바꾸는 게 맞다.

## 5-1. AC 는 "가림에서 상태를 이어가는가" 의 측정 대상이 못 된다 (2026-09-20 사용자 지적, 동의)

- **배포 시 autoregressive 가 아니다** — horizon 1, 문맥은 현재 프레임 1 장 (§3). 가림을 건너는 구간이 아예 안 생긴다.
  길게 가야 할 때도 rollout 대신 **하위목표 이미지**로 끊는다.
- **모델은 굴릴 수 있지만 (frame-causal + 2-step rollout 손실로 학습), 굴려도 해석이 안 된다:**
  AC 의 세계는 **agent 가 일으킨 변화**다. 장면은 준정적이고, 물체가 스스로 움직이는 자유 동역학은 학습에 거의 없다.
  우리 장면(UE5)에 걸려면 action 을 0 으로 넣어야 하는데 그건 "아무 일도 안 일어난다" 를 뜻한다 → **실패해도 증거가 아니다.**
  DROID 안에서 재려 해도 진실 위치 라벨이 없다.
- **그래서 AC 의 쓸모는 두 가지로 한정한다:** (1) **설계 증거** — 계획용으로 만든 world model 조차 horizon 1 로 쓰고,
  논문 스스로 "Long horizon planning … limited" 라고 적는다. (2) **레시피** — AR rollout 손실 + frame-causal attention
  (`z_training/README.md` §6 이 "아직 없는 것" 으로 적은 둘). 우리 장면에는 action 이 없으므로, 우리에게 남는 건 이 구조축이다.

## 6. 미결 — 이 폴더에서 무엇을 할지 (사용자 결정)

| 선택지 | 필요한 것 | 하네스 질문 (CLAUDE.md §12) |
|---|---|---|
| A. 논문 재현 (ViT-g + DROID) | DROID raw 수 TB (미검증), `vitg.pt`, 32 GPU 급 | beat 를 못 댄다 — 재현 자체는 우리 논문 주장이 아니다 |
| B. **ViT-H 로 바꾼 AC** + DROID | DROID, 기존 ViT-H ckpt. 채점 하네스가 ViT-H 기준 | "기준 모델 ViT-H" 원칙과 맞다. 단 데이터 확보 비용이 큼 |
| C. **AC 구조만** (frame-causal + rollout) 을 v11 / RollOut_v2 에, action 은 0 또는 물체 상태 | 데이터 이미 있음. `ac_predictor` 에 action/state 를 넣을 규약을 정해야 | state evolution 4 능력 중 **지속·전이** 를 직접 겨냥. 단 재학습이고, `z_training` 결과 "학습 도메인 안에서만 오른다" 를 먼저 넘어야 |
| D. **릴리즈 AC ckpt 를 받아 분석만** (학습 없음) | `vjepa2-ac-vitg.pt` 다운로드 (ViT-g) | frozen 원칙과 맞고 가장 싸다. 단 ViT-g 라 기존 ViT-H 수치와 직접 비교 불가 |

---

## 재현

이 문서의 확인 명령 (모델 로드 없음):

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
# augmentation 이 고정 center crop 인지
/data/hyuntak/anaconda3/envs/vjepa2/bin/python -c "
from src.datasets.utils.video.transforms import _get_param_spatial_crop as g
import collections
print(collections.Counter(g((1.777,1.777),(0.75,1.35),720,1280) for _ in range(2000)))"
# 논문 AC 절
pdftotext 'V-JEPA 2- Self-Supervised Video Models Enable Understanding, Prediction and Planning.pdf' - | sed -n 760,960p
# 디버그 경로 (더미 데이터, GPU 1장, 무작위 초기화 ViT-L)
python z_scripts/gen_dummy_droid.py
python -m app.main --fname configs/train/vitl16/droid_ac_debug.yaml --devices cuda:0 --debugmode True
```

읽은 파일: `app/vjepa_droid/{train,droid,utils,transforms}.py`, `src/models/ac_predictor.py`,
`src/models/utils/modules.py` (`build_action_block_causal_attention_mask`, `ACBlock`),
`configs/train/vitg16/droid-256px-8f.yaml`, `configs/train/vitl16/droid_ac_debug.yaml`,
`notebooks/utils/{world_model_wrapper,mpc_utils}.py`, `src/hub/backbones.py` (`vjepa2_ac_vit_giant`).
