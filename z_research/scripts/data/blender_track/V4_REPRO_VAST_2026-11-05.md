# BlenderTrack v4 생성기 — vast 레포 이식 · 재현 검사 · 데이터 ablation 하는 법 (2026-11-05)

- **질문**: v4 (같은 장면 · 속도 4 단계) 학습 데이터를 이 머신에서 똑같이 다시 만들고, 설정을 바꿔 데이터 ablation 을 할 수 있나.
- **답**: 된다. 생성기를 이 레포로 옮겼고, 설계 · 라벨은 원본 v4 와 **값까지 같게** 나온다. 영상 픽셀은 렌더러 자체의 흔들림 (같은 머신에서 두 번 그려도 생기는 차이) 만큼만 다르다 [측정].
- **다음**: ablation 축 (무엇을 바꿀지) 을 정한다. 엄밀 비교가 필요하면 기준 v4 도 이 머신에서 한 번 다시 렌더해 같은 렌더러로 맞춘다.

## 용어

| 용어 | 뜻 |
|---|---|
| 설계 | 영상 하나를 그리기 전에 정하는 모든 값 (카메라 · 가림막 · 물체 · 움직임 · 속도) 을 담은 json. 무작위 시드로 정해진다 |
| 묶음 | 장면 하나를 고정하고 target 속도만 4 단계로 바꾼 영상 4 개 + target 만 뺀 빈 장면 1 개 |
| 라벨 | 영상마다 물체 위치 · 화면 중심 · 상자 · 보이는 / 가려진 마스크를 담은 npz |
| 렌더 | 설계를 Blender (Cycles, GPU) 로 그려 mp4 로 만드는 단계 |
| 학습 목록 | 학습기가 읽는 영상 경로 목록 (`data_csv/<이름>/arms/all_mp4list.csv`) |
| 평면 · 빗면 | v4 의 두 설정. 평면 = `v4vc.yaml`, 빗면 = `v4vc_ramp.yaml` (이름 v4vcr) |

## 1. 왜 했나

사용자가 v4 를 기준으로 학습 데이터를 바꿔 가며 (데이터 ablation) 성능을 맞추려 한다. 그러려면 (1) v4 와 같은 데이터를 이 머신에서 만들 수 있어야 하고, (2) 설정 한 곳만 바꿔 변형을 만들 수 있어야 한다.

## 2. 무엇을 옮겼나

- 생성기 폴더 전체를 main 작업 사본 (`/workspace/world_model/vjepa2_main/z_research/scripts/data/blender_track`) 에서 이 레포의 같은 자리 `z_research/scripts/data/blender_track/` 로 복사했다 (43 파일, 1.4 MB, `__pycache__` 제외).
  - 출처: main 커밋 fc59160 (11-02) + 11-02 에 이 머신에 맞춘 수정 (`env.sh` · `localize_cfg.py` · 드라이버 로컬 실행). 그 내용은 같은 폴더 `SETUP_VAST_2026-11-02.md`.
  - 폴더 안에서만 서로 import 한다 (다른 레포 코드에 기대지 않는다).
- 환경: 설계 · 묶기 = `/venv/vjepa`, 렌더 = `/venv/blender` (Python 3.11 · bpy 5.0.1). 원본 v4 도 bpy 5.0.1 로 그렸다 (설계 json 의 render 항목) [측정].
- 학습 목록 생성기 `z_training/data/build_v4vc_index.py` 를 고쳤다. 기본값으로 쓰면 예전과 같다 (§3 회귀 검사).
  - `--gen-dir` 기본값 = 이 레포의 생성기 폴더
  - `--name`: 목록 폴더 이름. 기본 `blender_track_v4vc` 이고, ablation 판이 원본 목록을 덮지 않게 이름을 바꿔 쓴다
  - `--cfg-flat` · `--cfg-ramp`: 원본 v4 폴더 (`config.yaml`) 와 새로 생성한 폴더 (`config_v4vc.yaml`) 를 둘 다 읽는다. 평면만 · 빗면만도 된다

## 3. 재현 검사 [측정]

| 검사 | 방법 | 결과 |
|---|---|---|
| 설정 | 레포 `configs/v4vc.yaml` · `v4vc_ramp.yaml` 를 원본 폴더에 함께 저장된, 생성 때 쓴 설정과 비교 (주석 제외) | 둘 다 같다 |
| 설계 | 가장 작은 분할 (평면 probe seed 79 · 빗면 rprobe seed 89) 의 설계를 새로 만들어 원본 설계 json 과 모든 값 비교 | **130 / 130 같음**, 묶음 파일 52 / 52 같음 |
| 라벨 | 원본 4 편 (평면 3 · 빗면 1) 을 같은 설계로 다시 렌더 | **4 / 4 비트 단위로 같음** (위치 · 중심 · 상자 · 마스크) |
| 픽셀 (원본 대 다시 그린 것) | 64 장 디코드 후 비교 | 평균 차 0.51–0.71 / 255, 같은 픽셀 55–66 %, PSNR 46–48 dB |
| 픽셀 (이 머신에서 두 번 그린 것) | 같은 영상을 두 번 렌더 | 평균 차 0.54 / 255, 같은 픽셀 64 % — **위와 같은 크기** |
| 학습 목록 | 고친 목록 생성기를 기본값으로 원본 v4 에 돌림 | 영상 목록 바이트 단위로 같음, 보임 비율 배열 같음 (빈 값 위치 포함) |

해석
- 설계 · 라벨 · 목록 규칙은 원본과 똑같이 나온다. 같은 설정이면 같은 장면 · 같은 궤적 · 같은 가림 시점이다.
- 픽셀 차이는 이식 문제가 아니라 Cycles GPU 렌더 (디노이저 · GPU 병렬 합) 가 매번 조금씩 다르게 그리는 데서 온다. 원본 대 재렌더 차이가 같은 머신 두 번 차이와 같은 크기다. 비트 단위로 같은 픽셀은 원본 파이프라인을 그대로 다시 돌려도 얻을 수 없다.
- 그래서 ablation 비교를 엄밀하게 하려면 기준 v4 도 이 머신에서 다시 렌더해 같은 렌더러 흔들림 아래에 두는 것이 깔끔하다. 차이가 PSNR 47 dB 수준이라 학습 결과에 영향이 있을 가능성은 낮다 [추측].

## 4. 데이터 ablation 하는 법

1. `configs/v4vc.yaml` (빗면이면 `v4vc_ramp.yaml`) 를 새 이름으로 복사하고 바꿀 값만 고친다. **`name` 과 `out` 을 반드시 바꾼다** (out 의 앞부분 클러스터 경로는 그대로 두면 이 머신의 `BT_ROOT` 아래로 바뀐다).
2. 생성: `V4VC_CFG=<새 설정>.yaml bash run_v4vc.sh` (설계 → 렌더 → 묶기, 백그라운드, PID · 로그 경로를 찍는다).
3. 학습 목록: `build_v4vc_index.py --root <out> --name <새 이름>` → `data_csv/<새 이름>/arms/all_mp4list.csv`.
4. 학습: `configs/training/datasets.md` 에 `type: mp4_list` 항목 하나 (`blender_track_v4vc_all` 을 본떠 csv 만 바꿈), 학습 설정은 `v4vc_release_postft.yaml` 등을 extends 해서 `data.datasets` 만 바꾼다.

바꿀 수 있는 값 (설정 키)

| 축 | 키 | v4 값 | 뜻 |
|---|---|---|---|
| 상황 | `cells.objset` · `camera` · `occ` · `exclude` | primitive/realistic · front/side · board/wall/trapdoor/object | 물체 셋 × 카메라 × 가리는 것 |
| 속도 단계 | `velocity.levels` · `ratio_step` | 4 · 1.26 | 묶음 안 속도 단계 수 · 단계 사이 화면 속력 비 |
| 속력 범위 | `velocity.u0_rel` | [0.105, 0.135] | 가장 느린 단계의 화면 속력 (물체 지름 대비) |
| 감속 | `velocity.mu_roll` · `mu_slide` · `keep_speed` | 작음 · 0.8 | 구름 · 미끄럼 마찰, 가려질 때 남는 속력 비 |
| 가림 시점 · 길이 | `velocity.hide` · `fast_hide` | 샘플 14–15 · 1.6–2.4 | 가려지기 시작하는 샘플, 가장 빠른 단계의 완전 가림 길이 |
| 규모 | `splits.<분할>.families` · `candidates` · `n` · `seed` | train_fam 1,190 묶음 등 | 묶음 수 · 후보 수 · 시드 |
| 영상 | `fps` · `frames` · `size` · `render.samples` · `denoise` | 20 · 64 · 256 · 16 · true | |

- ⚠️ 설계 단계의 값 (상황 · 속도 · 가림) 을 바꾸면 같은 시드라도 무작위 뽑기 순서가 달라져 장면 자체가 바뀔 수 있다 [추측]. "장면은 같고 한 축만 다르다" 가 필요하면 생성 후 `design_fam/` 을 원본과 비교해 확인한다.
- 평면 v4 는 묶음 분할 (`train_fam`) 과 대조 분할 (`train_ind`, 장면마다 속도 하나) 을 함께 설계한다. 학습 목록은 묶음 분할만 쓰지만 길이를 대조에 맞추는 규칙 때문에 대조 분할도 렌더해야 목록이 원본 규칙대로 나온다.

## 5. 시간 · 단서

- 렌더 시간: GPU 를 렌더만 쓸 때 작업자 하나가 clip 하나에 약 11–15 초 (11-02 측정, `SETUP_VAST_2026-11-02.md`). v4 원본 전체 (설계 파일 20,847) 를 GPU 8 × 작업자 8 로 그리면 약 1 시간 안팎 [추측]. 학습과 겹치면 4–5 배 느려진다. 이번 검사는 학습 다섯 개와 겹쳐 clip 당 약 2 분이었다 [측정].
- ⚠️ `run_v4vc.sh` 의 로컬 (sbatch 없음) 전체 실행 — 설계 → 렌더 → 묶기를 끝까지 — 은 아직 돌려 보지 않았다. 이번에는 설계 · 렌더 단계를 각각 직접 불러 검사했다. 첫 ablation 은 작은 분할 (probe) 로 사슬 전체를 먼저 돌려 볼 것.
- `/workspace` 는 이 인스턴스에서 볼륨이 아니다. recycle · destroy 하면 생성물이 사라진다.
- 독립 실행형 Blender 4.2 (`/workspace/tools/blender-4.2.3-linux-x64`) 는 다른 생성기 (phys3d) 용이다. BlenderTrack 은 `/venv/blender` 의 bpy 5.0.1 을 쓴다.
- 검사 산출물: `/workspace/dataset/training_data/blender_track/_repro_v4_20261105` (설계 · 재렌더 4 편) · `…_rerun` (같은 머신 두 번째 렌더).

## 부록: 재현

```bash
cd /workspace/world_model/vjepa2/z_research/scripts/data/blender_track
source env.sh
# 설계 재현 검사 (CPU, 수 초)
export BT_OUT=/workspace/dataset/training_data/blender_track/_repro_v4_20261105
for c in v4vc.yaml v4vc_ramp.yaml; do $BT_PY localize_cfg.py configs/$c --write 'config_{name}.yaml'; done
$BT_PY design_families_v4vc.py --config $BT_OUT/config_v4vc.yaml  --split probe  --procs 32 && $BT_PY design_families_v4vc.py --config $BT_OUT/config_v4vc.yaml  --split probe  --select
$BT_PY design_families_v4vc.py --config $BT_OUT/config_v4vcr.yaml --split rprobe --procs 32 && $BT_PY design_families_v4vc.py --config $BT_OUT/config_v4vcr.yaml --split rprobe --select
# 특정 clip 만 렌더 (설정 json 필요): 번호 목록 파일을 --extra_list 로, 몫은 비게
for n in v4vc v4vcr; do $BT_PY -c "import yaml,json,sys; json.dump(yaml.safe_load(open(sys.argv[1])), open(sys.argv[2],'w'), indent=1)" $BT_OUT/config_$n.yaml $BT_OUT/config_$n.json; done
echo "0 1 5" > $BT_OUT/extra_probe.txt
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 $BT_BLENDER_PY gen_render.py --config $BT_OUT/config_v4vc.json --split probe --worker 99999 --nworkers 100000 --extra_list $BT_OUT/extra_probe.txt

# v4 전체 다시 만들기 (평면 · 빗면) — 출력 자리를 새로
BT_OUT=/workspace/dataset/training_data/blender_track/v4_repro GPU_LIST=0,1,2,3,4,5,6,7 PER_GPU=8 bash run_v4vc.sh
BT_OUT=/workspace/dataset/training_data/blender_track/v4_repro GPU_LIST=0,1,2,3,4,5,6,7 PER_GPU=8 V4VC_CFG=v4vc_ramp.yaml bash run_v4vc.sh
# ablation 판: configs/<새>.yaml (name · out · 바꿀 값) → V4VC_CFG=<새>.yaml bash run_v4vc.sh

# 학습 목록 (원본 v4 는 기본값 = 예전과 같은 결과)
/venv/vjepa/bin/python /workspace/world_model/vjepa2/z_training/data/build_v4vc_index.py --root /workspace/dataset/training_data/v4_velocity_control
/venv/vjepa/bin/python /workspace/world_model/vjepa2/z_training/data/build_v4vc_index.py --root <out> --name <새 이름>
```
