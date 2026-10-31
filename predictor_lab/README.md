# predictor_lab — BlenderTrack 으로 predictor 를 학습하고 시트 열 그대로 바로 재는 공간 (2026-10-31, vll7 기준)

> **무엇** — 아이디어 (손실 항 · head · 창 고르기) 를 config 한 장으로 적으면, 같은 데이터로 학습하고 Google 시트 (2027_CVPR_world_model, Overall) 와 같은 열 (VOBench P / R · VOBench Gravity P / R · rubric office · IntPhys 1 · IntPhys1 late occ) 로 재서, 시트의 기준 4 행과 같은 이름 · 같은 열 순서의 표로 낸다.
> **검증** — 표 계산기가 시트 기준 4 행의 점수 28 칸 중 23 칸을 그대로 다시 낸다 (다른 5 칸은 10-15 평가 행렬 문서 값과 같다, §3). 학습 · 평가는 끝까지 돌려 확인했다 (vll1 · vll6), vll7 은 설정 병합까지 확인.
> **처음 할 일** — `paths.env` 의 데이터셋 경로 · 파이썬을 자기 노드에 맞게 고친다 (지금 값은 vll7 기준: Blackwell 이라 vjepa2_bw, 학습셋은 NFS).

---

## 용어

| 용어 | 뜻 |
|---|---|
| encoder · predictor | V-JEPA 2 ViT-H. encoder (문맥 · 목표 두 개) 는 고정, predictor (22.3M) 만 학습한다. 출발점은 릴리즈 predictor 가중치 |
| BlenderTrack | 학습 데이터. 눈높이 Blender 렌더 4,000 clip (6 초, 30 fps, 256²), 연속 무작위 궤적 · 가림 사건, 프레임마다 target 위치 · 64×64 amodal / visible 칸 라벨. 설계 · 렌더 문서 `z_analysis/release/06_blender_track/README.md` |
| 창 | 학습 때 clip 에서 잘라 쓰는 32 샘플. 배치마다 보폭 (2 · 3 · 4 프레임) 과 문맥 길이 (8–24 샘플 = 4–12 블록) 를 고르고 나머지를 미래로 예측한다 |
| late 창 | 문맥 마지막 샘플에서 target 이 가려져 있고, 문맥 앞쪽에서 보였고, 미래에서 다시 보이는 창. 기본으로 샘플의 50 % 를 late 창으로 채운다 |
| binding 항 | 참 미래 target 자리 (블록마다 중심 칸 3×3) 토큰만의 L1 을 표준 L1 에 더한 것 |
| 점유 head | predictor 마지막 층 토큰마다 "target 이 이 토큰을 차지하는 비율" 을 내는 선형 head. 목표는 amodal 칸 (가려져도 물체가 있는 곳). 기울기가 predictor 로 흐른다 |
| 쌍 정답률 | 문맥이 같은 가능 · 불가능 영상 한 쌍에서 surprise (예측 미래와 실제 미래 latent 의 L1 평균) 가 불가능 쪽이 크면 정답. 우연 50 % |

## 1. 흐름

1. **학습** — config 이름 하나로 sbatch 잡을 띄운다. 데이터 경로는 `paths.env` 에서 들어간다.
   - 기본 팔 셋: 표준 L1 만 (대조군) · L1 + binding 항 · L1 + binding + 점유 head.
   - 결과는 `z_training/runs/<이름>/` 이다.
2. **평가** — run 이름 하나로 다섯 벤치를 차례로 채점하는 sbatch 잡을 띄운다.
   - encoder 는 릴리즈 그대로, predictor 만 그 run 의 것으로 바꿔 끼운다. 채점기는 저장소의 표준 하네스 그대로다.
   - 결과는 `<run>/eval_lab/` 이다.
3. **표** — 시트 기준 4 행 (V-JEPA 2 ViT-H · V-JEPA2 w/ VoBench · AR predictor (vast AR-state ep70) · AR predictor w/ VoBench) 과 새 run 을 시트 B–R 열 그대로 낸다. 새 run 의 이름은 config 의 `lab.sheet_name`, Step · Seen clip · Data scale · Epoch 는 run 폴더에서 계산한다. `--tsv` 는 시트에 바로 붙이는 탭 구분 출력이다.

## 2. 벤치 다섯 (시트 열과 같다)

| 열 | 무엇 | 쌍 | 창 · 규칙 |
|---|---|---:|---|
| VOBench P / R | VOBench_v1 split v2 test 중 문맥 끝 가림 (late). P = 단순 도형, R = 실물 메시 | 108 / 108 쌍 | 문맥 16 + 미래 16, 보폭 3 (표준 채점) |
| VOBench Gravity P / R | gravity test: 문맥 하나 + 미래 여럿 (ledge 2 · arc 9). 진짜 미래의 surprise 가 가장 작으면 정답 | 54 / 54 block | 같은 표준 채점 |
| IntPhys 1 | IntPhys 1 dev 전체 | 180 | Garrido A.8: 창 16 · 32 × skip 2 · 5 · 10, 시작점마다 문맥 길이 최솟값 (Filtered). **격자 최고 칸**이라 held-out 추정치가 아니다 |
| IntPhys 1 late occ | IntPhys 1 균일 창 판 (문맥 16 = 위반 전 균일, 미래 16 = 위반 뒤 균일) 중 가림 쌍 | 90 | 움직이는 물체 · 가림 60 + 정지 물체 · 가림 30 |
| rubric office | 실제 영상 (TCOW Rubric Office) 20 개, 사라짐 짝 20 + 생겨남 짝 20 | 40 | violation 앞뒤 균일 16 + 16 장. 원본 mp4 에서 라벨 (영상별 프레임 번호 32 개) 대로 바로 읽는다 (예전에 뽑은 PNG 와 화소 차 0). 같은 영상의 두 짝은 독립이 아니다 |

## 3. 검증 [측정, 2026-10-31]

**표 계산기** — 시트 기준 4 행의 점수를 기존 결과 폴더에서 이 공간의 규칙으로 다시 계산해 시트 (2027_CVPR_world_model, Overall 탭) 와 대조했다 (`table.py --check`).

| 시트 행 | VOBench P | VOBench R | Gravity P | Gravity R | rubric office | IntPhys 1 | IntPhys1 late occ |
|---|---:|---:|---:|---:|---:|---:|---:|
| V-JEPA 2 ViT-H | **58.3** (시트 57.9) | 65.7 | 0.0 | 0.0 | 52.5 | 88.9 | 57.8 |
| V-JEPA2 w/ VoBench | 88.0 | 91.7 | 100.0 | 92.6 | 50.0 | 83.3 | 62.2 |
| AR predictor (vast AR-state ep70) | **75.9** (73.6) | **75.9** (77.3) | **18.5** (20.4) | **11.1** (8.3) | 52.5 | 90.0 | 81.1 |
| AR predictor w/ VoBench | 89.8 | 89.8 | 57.4 | 59.3 | 52.5 | 90.0 | 83.3 |

- 28 칸 중 23 칸이 시트와 소수 첫째 자리까지 같다.
- 굵은 5 칸은 시트와 다르다. 다시 계산한 값은 10-15 평가 행렬 문서 (`z_research/VOBench_v1/Archive/EVAL_MATRIX_2026-10-15.md`) 와 같다. 시트 값을 내는 결과 폴더는 레포에서 못 찾았다 (다른 실행에서 온 것으로 보인다 [추측]). 어느 쪽을 기준으로 할지는 사용자 결정이다.

**설정 병합 (GPU 0 장)**
- 학습: 세 팔 config 가 데이터 경로 · 창 노브 · 손실 노브를 맞게 합친다.
- 평가: 다섯 벤치 모두 경로 · 창 · predictor 교체가 맞게 합쳐진다 (vll5 · vll7).
  - IntPhys 1 late 프레임은 기록된 창 정의 (영상마다 32 장의 프레임 번호) 로 dev PNG 를 가리키는 링크를 만들어 쓴다 (`setup_intphys1_late.py`, 360 영상 · 11,520 링크 · 빠진 원본 0).
  - Rubric 은 원본 mp4 에서 라벨 (영상별 원본 프레임 번호 32 개, `make_rubric_index.py`) 대로 바로 읽는다. 예전에 뽑아 둔 PNG 와 화소 차이 0 (60 fps 중복 영상 포함 5 영상 대조).

**끝까지 돌린 시험 (GPU, 2026-10-31)**

| 시험 | 결과 |
|---|---|
| 학습 스모크 (점유 head 팔, GPU 1 장 · clip 64 · 4 step) | 손실 셋 (L1 · binding · 점유) 이 기록되고 체크포인트 저장까지 끝남 |
| 릴리즈 predictor → IntPhys 1 late occ | **57.8** (기준 · 시트 57.8) |
| 릴리즈 predictor → Rubric | **50.0** (기준 52.5). 한 쌍 차이다: 그 쌍의 surprise 차가 상대 0.002 % 로 사실상 동점이고, 기준값을 낸 한 번 통과 채점기와 표준 하네스의 수치 차 (영상별 surprise 상대차 중앙 0.004 %) 로 뒤집힌다 |
| 학습한 체크포인트 (위 스모크, 점유 head 포함) → Rubric | predictor 만 골라 불러오고 채점 · 표까지 정상 (52.5 — 4 step 스모크라 값 자체는 의미 없음). 결과 `z_training/runs/lab_bind_occ_smoke64/eval_lab/` |
| 다른 노드 (vll6) 에서 IntPhys 1 late | 프레임 링크를 평가 잡이 자동으로 만들고 돈다 |

**설정 대조 (코드 수준)** — 이 공간이 만드는 병합 설정을 기준 실행이 실제로 쓴 설정 파일과 키 단위로 비교했다 (경로 · 출력 폴더 · predictor 경로는 빼고).

| 벤치 | 대조 키 | 다른 키 |
|---|---:|---|
| VOBench late (기준: 릴리즈 + VOBench PFT 평가) | 38 | 0 |
| IntPhys 1 창 16 · 32 (기준: 릴리즈 A.8) | 46 · 46 | 0 · 0 |
| IntPhys 1 late (기준: 릴리즈) | 38 | 0 |
| Rubric (기준: vast AR 평가) | 39 | 1 — predictor 구조 표시 (기준이 AR 모델이라 생긴 차이, 정상) |

IntPhys 1 격자 · VOBench 는 오래 걸리거나 vll5 GPU 를 기다려야 해서 끝까지 돌리지 않았다. 위 대조로 원 실행과 같은 설정임을 확인했다.

## 4. 아이디어를 넣는 곳

| 바꾸려는 것 | 어디서 | 코드 수정 |
|---|---|---|
| 창 · late 비율 · 보폭 · 문맥 길이 | config 의 데이터 노브 (`_template.yaml`) | 없음 |
| binding · 점유 · 운동 head 가중 | config 의 손실 노브 | 없음 |
| 학습 길이 · lr · seed | config 의 최적화 노브 | 없음 |
| 새 손실 항 · 새 head | `app/vjepa_frozen/where_head.py` 에 항을 쓰고 config 키로 켠다. 라벨 (target 위치 · 가려진 비율 · amodal 점유) 은 이미 배치에 실려 온다 | 한 파일 |
| 새 라벨 · 다른 창 규칙 | `app/vjepa_frozen/blender_track_data.py` (수집기) | 한 파일 |
| 새 학습 데이터 | 렌더러 `z_research/scripts/data/blender_track/` (설계 · 렌더 · 렌더 전 검사) → 메타 `z_training/data/build_blender_track_index.py` → `paths.env` | 설계 파일 |

## 5. 단서

- **평가 데이터는 노드 로컬이다 (vll7 기준).** VOBench · gravity `/local_datasets/world/testbed/VOBench_v1` · IntPhys 1 dev `/local_datasets/world/world_analysis` · IntPhys 1 late `/local_datasets/world/IntPhys1_late_uni` (없으면 dev PNG 링크를 자동으로 만든다) · Rubric `/local_datasets/world/testbed/real_bench/rubric_office_quads`. 학습셋 (BlenderTrack) 은 NFS 에서 읽는다 (vll7 디스크 여유 ~40 GB).
- **AR predictor 행 (vast AR-state ep70) 의 VOBench · Gravity 4 칸과 V-JEPA 2 ViT-H 의 VOBench P 는 시트와 다르다.** 다시 계산한 값은 10-15 평가 행렬과 같고, 시트 값을 내는 결과 폴더는 레포에서 못 찾았다. 어느 쪽을 기준으로 할지는 사용자 결정.
- **IntPhys 1 은 격자 최고 칸이다.** 같은 격자에서 고른 값이라 차이가 작으면 (±3) 읽지 않는다.
- **표본이 작다.** Rubric 40 쌍 (영상 20) · IntPhys 1 late occ 90 쌍. ±10–15 pt 안의 차이는 가르지 못한다.
- **복사 기준선은 표에 없다.** 예측 없이 마지막 문맥을 복사해도 VOBench 사라짐 · IntPhys 1 은 꽤 풀린다. 주장에는 복사 대비 차이를 같이 낸다 (저장소 CLAUDE.md §0).
- **학습 로그의 "masks: context_frames [16]" 은 무시해도 된다.** BlenderTrack 수집기가 배치마다 문맥 길이를 다시 고른다. 실제 값은 로그의 "mask bt s<보폭> C<문맥>" 이다.
- **AR (자기회귀) predictor 도 평가는 된다** (체크포인트의 구조 표시를 읽는다). 다만 surprise 가 블록별 목표 latent 공간이라 쌍 정답률만 비교한다. 학습 config 는 릴리즈 predictor post-FT 만 준비돼 있다.

---

## 부록: 명령 · 파일

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python        # paths.env 의 LAB_PY (vll7)
# 0. 경로 확인: predictor_lab/paths.env 를 자기 환경에 맞게 수정 (IntPhys 1 late 링크 · Rubric 라벨 index 는 평가가 없으면 알아서 만든다)
# 1. 학습 (sbatch, 기본 GPU 2 장 · paths.env 의 LAB_NODE · LAB_PY)
bash predictor_lab/train.sh lab_bind_occ
NAME=my_idea_s1 SET="meta.seed=1" bash predictor_lab/train.sh my_idea
DRYRUN=1 bash predictor_lab/train.sh lab_bind_occ                       # 병합 config 만 (GPU 0 장)
LOCAL=1 GPUS=1 LIMIT=64 SET="optimization.epochs=1" bash predictor_lab/train.sh lab_bind_occ   # 스모크 (4 step)
# 2. 평가 (sbatch, GPU 1 장) — 벤치: vobench gravity rubric intphys1 intphys1_late (골라도 된다)
bash predictor_lab/eval.sh lab_bind_occ
bash predictor_lab/eval.sh lab_bind_occ vobench gravity rubric
DRYRUN=1 bash predictor_lab/eval.sh z_training/runs/release_vith/latest.pt   # 경로 · 설정 검사만
# 3. 표 (시트 B–R 과 같은 이름 · 열)
$P predictor_lab/table.py lab_bind_occ lab_bind lab_base
$P predictor_lab/table.py lab_bind_occ --no-refs --tsv                    # 시트에 바로 붙이는 행
$P predictor_lab/table.py --check                                          # 기준 4 행 재계산 ↔ 시트 대조
```

| 파일 | 무엇 |
|---|---|
| `paths.env` | 데이터 · 체크포인트 경로 (이것만 고친다) |
| `train.sh` · `train_job.sh` | 학습 실행기 (sbatch) · 본체 (resolve_train → harness/launch.py, 파이썬은 LAB_PY). config 는 `configs/` 를 먼저 찾고 없으면 `configs/training/` |
| `configs/lab_base.yaml` · `lab_bind.yaml` · `lab_bind_occ.yaml` | 기본 세 팔 (L1 · + binding · + 점유 head) |
| `configs/_template.yaml` | 새 아이디어 틀 (노브 목록 · 코드 넣는 곳) |
| `eval.sh` · `eval_job.sh` | 평가 실행기 (`cvpr/harness/launch.sh` 를 벤치마다 부른다) |
| `table.py` | 시트 형태 채점표 (기준 4 행 + run, B–R 열) · `--tsv` · `--check` |
| `refs.json` | 시트 기준 4 행: 이름 · C–J 메타 · 결과 폴더 · 시트 점수 |
| `setup_intphys1_late.py` | IntPhys 1 late 프레임 폴더 = 기록된 창 정의로 dev PNG 링크 |
| `make_rubric_index.py` | Rubric 라벨 index (원본 mp4 + 영상별 프레임 번호 32 개) |
| `CLAUDE.md` | Claude 가 이 폴더에서 일할 때의 지침 |
| `app/vjepa_frozen/blender_track_data.py` | BlenderTrack 데이터셋 · 수집기 (창 · late 과표집 · 라벨) |
| `app/vjepa_frozen/where_head.py` | binding 항 · 점유 head · 운동 head |
