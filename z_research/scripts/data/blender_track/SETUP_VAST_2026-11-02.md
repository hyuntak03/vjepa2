# BlenderTrack 생성기 — Vast 머신 이식 메모 (2026-11-02)

- **질문**: 연구실 SLURM 클러스터용 BlenderTrack 생성기 (v1 → v3 → v4vc → v5mc) 를 이 Vast 머신 (GPU 8 장, SLURM 없음) 에서 돌릴 수 있나.
- **답**: 된다. v5mc 를 설계 → 고르기 → 렌더 → 묶기까지 작게 끝까지 돌렸다 (영상 141 + 시험 VoE 영상 178, 7 분) [측정].
- **다음 할 일**: 전체 v5mc (영상 약 47.6K) 를 언제 돌릴지 정한다. GPU 를 혼자 쓰면 약 2.5–3 시간, 지금처럼 학습과 겹치면 약 10–12 시간 [측정에서 계산].

용어
| 용어 | 뜻 |
|---|---|
| 판 | 데이터 세대 (v1 · v3 · v4vc · v5mc …). 설정 파일 하나 = 판 하나 |
| 작업 공간 | 렌더 중간 산출 자리 (설정의 out). v5mc 는 `…/v5_motion_control/work` |
| 최종 폴더 | 묶기가 옮겨 놓는 tar 친화 폴더 (설정의 data_root). v5mc 는 `…/v5_motion_control/data` |
| 묶음 | 같은 장면에서 운동만 바꾼 영상 여러 개 + 물체만 뺀 빈 장면 |
| 작업자 | 렌더 프로세스 하나 (GPU 하나에 여러 개, clip 을 잠금 파일로 나눠 가진다) |
| 학습 겹침 | 같은 GPU 에서 predictor 학습이 함께 돌던 상태 |

## 1. 환경

왜: 렌더는 bpy 5.0.1 모듈 (Cycles) 이 필요한데 이 머신 파이썬에는 없고, 독립 실행형 Blender 4.2 는 vast 브랜치 생성기 전용이라 건드리면 안 된다.

| 항목 | 값 | 확인 |
|---|---|---|
| 렌더 파이썬 | 새 venv `/venv/blender` — Python 3.11.16 (uv 관리판) · bpy 5.0.1 · numpy 1.26.4 | import 됨, 시스템 라이브러리 추가 설치 없음 [측정] |
| Cycles GPU | OptiX · CUDA 둘 다 RTX PRO 6000 Blackwell 을 본다. 렌더러는 OptiX 를 먼저 고른다 | 256² 16 샘플 한 장 0.3 초, 그림 정상 [측정] |
| HDRI | bpy 모듈판 `…/bpy/5.0/datafiles/studiolights/world/` 에 8 개 (city · courtyard · forest · interior · night · studio · sunrise · sunset). 설정이 쓰는 7 개 다 있음 | 렌더 결과에 배경으로 보임 [측정] |
| 설계 · 묶기 파이썬 | 기존 `/venv/vjepa` (Python 3.12 · numpy · pyyaml · decord · PIL) — 새로 설치 없음 | [측정] |
| ffmpeg | 시스템 6.1.1, libx264 있음 — 설치 없음 | [측정] |
| 경고 | `HIPEW initialization failed` 는 AMD 장치 탐색 실패로 무해 | |

## 2. 바꾼 파일

원칙: 파이썬 생성 코드는 거의 안 고쳤다. 경로는 환경 파일 하나에서 읽고, 설정은 "출력 자리만 바꾼 사본" 으로 돈다. 클러스터에서는 예전과 같게 돈다 (자리 자동 판별, 가짜 sbatch 로 제출 명령 확인 [측정]).

| 파일 | 무엇 |
|---|---|
| `env.sh` (새) | 경로 정본. 파이썬 두 개 · 출력 뿌리 · 목록 자리. 클러스터 파이썬이 있으면 클러스터 값, 없으면 Vast 값. 클러스터 원래 값은 주석으로 |
| `localize_cfg.py` (새) | 설정 사본을 `<새 out>/config_*.yaml` 에 쓴다. out · data_root 앞부분 `/data/hyuntak/…/nfs_cache/blender_track` 을 출력 뿌리로 바꾼다 (주석 유지). 스모크용으로 분할 고르기 · 값 덮기도 받는다. 클러스터 모드에서는 원본과 바이트까지 같다 (설정 8 개 확인 [측정]) |
| `run_v5mc.sh` | env.sh · 설정 사본 사용. sbatch 가 있으면 예전 제출 그대로, 없으면 (또는 강제 로컬 지정) 한 머신 판: 설계 → 고르기 → GPU N × 작업자 W 렌더 (기존 잠금 방식) → 묶기를 백그라운드로, 로그 · PID 출력. 스모크용 "분할마다 묶음 N 개만 렌더" |
| `run_v4vc.sh` | 위와 같은 구조 (평면 · 빗면 설정 둘 다). 로컬 판은 건식 점검만 함 |
| `gen_render.sh` · `gen_render_extra.sh` | 경로를 env.sh 에서. sbatch 가 스크립트를 spool 로 복사하므로 자리는 드라이버가 넘긴 값 → 클러스터 자리 → 자기 자리 순으로 찾는다. 렌더 GPU 번호 목록 (선택) 추가 |
| `run_render.sh` · `run_v3t.sh` | 경로를 env.sh 에서 (v3t 는 설정 사본도) |
| `pipeline.sh` · `run_v3_main.sh` · `run_v3t_multi.sh` | 경로 · 설정 사본. sbatch 가 없으면 분명한 메시지를 내고 멈춘다 (로컬 판 없음) |
| `compose_v4vc.py` | 목록 자리를 환경변수로 (비우면 레포 data_csv, 예전과 같음) |
| `make_v3_meta.py` · `make_v3t_meta.py` · `make_ctrl_meta.py` | 출력 뿌리를 환경변수로 (비우면 클러스터 자리) |

Vast 기본값: 출력 뿌리 `/workspace/dataset/training_data/blender_track` → v5mc 작업 공간 `…/v5_motion_control/work`, 최종 폴더 `…/v5_motion_control/data`. v4vc 목록은 `…/blender_track/data_csv/` (레포 data_csv 대신).

## 3. 스모크

왜: 설계 → 렌더 → 묶기 사슬이 이 머신에서 끊기지 않는지, clip 당 렌더 시간이 얼마인지.

무엇을: v5mc 9 분할 전부를 분할마다 후보 52 · 묶음 26 으로 줄여 설계하고, 고른 묶음 중 분할마다 2 개 (상황이 퍼지게 처음 · 끝) 만 렌더했다. GPU 0 · 1 × 작업자 8. 출력 `/workspace/dataset/training_data/blender_track/_smoke_20261102/` (50 MB).

결과 [측정]
| 단계 | 결과 | 시간 |
|---|---|---|
| 설계 (200 프로세스) | 분할마다 후보 52 중 성공: speed · accel 39–42, grid 19–23 | 분할당 2–5 초, 9 분할 29 초 |
| 고르기 | 묶음 26 목표 중 15–24 고름 (grid 가 가장 모자람, 빗면 × 판 · 벽에서 실패 — 클러스터 설계 시험과 같은 경향) | 1 초 미만 |
| 렌더 | 영상 141 개, 오류 0, 투영 오차 0.000 px | 6 분 40 초 |
| 묶기 | 렌더한 묶음 18 개 전부 렌더 타이밍 통과. 학습 목록 47 · 시험 VoE 225 줄 (만든 불가능 영상 178) | 7 초 |

clip 하나 (64 장) 의 시간은 거의 다 GPU 렌더다 (라벨 0.1–0.2 초 · 인코딩 0.1 초).
| 구간 | GPU 하나의 동시 작업자 | clip 당 시간 |
|---|---|---|
| 학습이 아직 안 돌 때 (첫 네 분할) | 6–8 | **10.8–15.0 초** |
| 학습 겹침 (뒤 다섯 분할; 같은 GPU 에서 predictor 학습 100 % 사용) | 8 | **29–60 초** |

해석: 학습과 겹치면 같은 작업자 수에서 clip 이 4–5 배 느려진다 [측정]. 작업자 하나가 GPU 메모리 약 4 GB 를 쓴다 [측정] — 8 개면 32 GB, 학습 (장당 15–28 GB) 과 함께 96 GB 안에 들어간다.

출력 구조 (최종 폴더, 설계대로)
- `{train,test,probe}/{speed,accel,grid}/shard_000/<clip>.{mp4,npz,json}` — mp4 256² · 20 fps · 64 장 h264. npz = 위치 · 화면 중심 · 상자 · amodal/visible 마스크 (64 × 64 칸) · 판 각도. json = 설계 + 렌더 설정 (bpy 5.0.1).
- `test_voe/<종류>/<묶음>/{vanish,appear,swap}_*.mp4` — 프레임 32 에서 이어 붙인 불가능 영상 (decord 로 64 장 읽힘).
- `lists/` — 학습 목록 npz (clip 상대 경로 · 보임 비율 (64,) · 뿌리 절대 경로) · 분할별 manifest csv · 시험 VoE 목록 csv · 개수 json. 맨 위 README 자동 작성.

## 4. 전체 v5mc 예상

영상 수: 스모크의 묶음당 영상 수 (speed · accel 약 6.1–6.3, grid 약 10.5–10.7) 와 채움 비율을 전체 설정에 곱하면 약 **47.6K** (학습 43.5K · 시험 3.7K · 확인 0.4K) [측정에서 계산]. 채움 비율은 후보가 3 배인 전체 설정에서 더 높을 수 있다 [추측].

| 조건 | GPU 하나 처리량 | 8 GPU 렌더 | 근거 |
|---|---|---|---|
| GPU 를 렌더만 씀, 작업자 8 | 약 0.55–0.67 clip/초 | **약 2.5–3 시간** | 학습 전 구간 clip 당 12 초 안팎 [측정] → 8 / 12–14.5 |
| 지금처럼 학습 겹침, 작업자 8 | 약 0.13–0.16 clip/초 | **약 10–12 시간** (학습도 느려진다) | 뒤 구간 50–60 초 [측정] |

- 설계 (CPU 240): 후보 약 28.4K, 스모크 속도로 약 5–10 분 [추측]. 묶기 (VoE 이어 붙이기 약 14K 개 포함) 수 분 [추측].
- 작업자 12 이상이 더 빠른지는 못 쟀다 (학습이 GPU 를 다 쓰고 있었다). 학습이 끝난 뒤 첫 분할 렌더 로그로 확인할 것.
- 디스크: clip 하나 (mp4 + npz + json) 약 107 KB [측정] → 영상 5.1 GB + 시험 VoE 약 1.5 GB + 설계 파일 약 0.8 GB = **약 7–8 GB** (여유 457 GB).

## 5. 안 되는 것 · 단서

- `/workspace` 는 이 인스턴스에서 **볼륨이 아니다** (`workspace_is_volume = false`). 인스턴스를 recycle · destroy 하면 렌더 결과가 사라진다 → 렌더 뒤 최종 폴더를 tar 로 밖에 옮겨야 한다.
- v1 · v2 (`pipeline.sh`) · v3 (`run_v3_main.sh` · `run_v3t_multi.sh`) 는 로컬 판이 없다 — 경로만 맞췄고 sbatch 없으면 멈춘다. v3t 한 노드 판 (`run_v3t.sh`) 은 로컬에서도 돌 수 있지만 안 돌려 봤다.
- v4vc 로컬 판은 건식 점검 · 가짜 sbatch 제출만 했다 (렌더 안 함). v4vc 목록은 레포 data_csv 가 아니라 `…/blender_track/data_csv/` 에 써지므로 학습 쪽 경로를 맞춰야 한다.
- 스모크는 후보 수를 줄인 설정이라 묶음 설계가 전체 실행의 같은 번호와 다르다 (설계 일정이 후보 수에 따라 바뀐다). 스모크 결과를 학습에 쓰지 말 것.
- 렌더 시간 측정의 뒤 다섯 분할은 학습 겹침 상태다. 학습 전 구간은 1 분 남짓 · 작업자 6–8 이라 정상 상태 처리량은 ±20 % 정도 불확실 [추측].
- 파이썬 파일 docstring 안의 예시 명령 (`P=/data/hyuntak/…`) 은 그대로 뒀다 (설명일 뿐).
- 독립 실행형 Blender 4.2.3 · vast 체크아웃은 건드리지 않았다.

## 부록: 재현

```bash
cd /workspace/world_model/vjepa2_main/z_research/scripts/data/blender_track
source env.sh                                   # 자리 · 파이썬 · 출력 뿌리 (BT_SITE=vast 자동)
DRY=1 bash run_v5mc.sh                          # 아무것도 안 쓰고 out · 분할 · 렌더 여유 GPU 만

# 환경 다시 만들기
uv venv /venv/blender --python /.uv/python_install/cpython-3.11-linux-x86_64-gnu/bin/python3.11
uv pip install --python /venv/blender/bin/python "bpy==5.0.1"

# 스모크 (이 메모의 숫자)
SETS=""; for s in s_speed s_accel s_grid t_speed t_accel t_grid; do SETS="$SETS splits.$s.families=26 splits.$s.candidates=52"; done
BT_ROOT=/workspace/dataset/training_data/blender_track/_smoke_20261102 BT_CFG_SET="$SETS" SMOKE_FAMS=2 \
  GPU_LIST=0,1 PER_GPU=8 DESIGN_PROCS=200 PACK_WORKERS=32 bash run_v5mc.sh

# 전체 v5mc (띄우지 않았다) — 백그라운드, PID · 로그 경로를 찍는다
GPU_LIST=0,1,2,3,4,5,6,7 PER_GPU=8 bash run_v5mc.sh
tail -f /workspace/dataset/training_data/blender_track/v5_motion_control/work/logs/local_*.out
kill -- -$(cat /workspace/dataset/training_data/blender_track/v5_motion_control/work/logs/local.pid)   # 멈추기
STAGE=render bash run_v5mc.sh                   # 이어 하기 (끝난 clip 은 건너뜀). STAGE = design | select | render | pack

# v4vc (평면 / 빗면)
bash run_v4vc.sh ;  V4VC_CFG=v4vc_ramp.yaml bash run_v4vc.sh
```

손잡이: `GPU_LIST` (기본 = 빈 메모리 ≥ `MIN_FREE_GB` (= 작업자 × 5 GB) 인 GPU 전부) · `PER_GPU` (8) · `THREADS` (2) · `DESIGN_PROCS` (nproc − 16) · `PACK_WORKERS` (48) · `FG=1` (이 셸에서) · `BT_LOCAL=1` (sbatch 가 있어도 로컬) · `BT_ROOT` / `BT_OUT` / `BT_DATA_ROOT` (출력 자리) · `BT_CFG_ONLY` / `BT_CFG_SET` (설정 사본에서 분할 고르기 · 값 덮기).
