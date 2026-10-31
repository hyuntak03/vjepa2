# predictor_lab — Claude 작업 지침

이 폴더는 "아이디어 → BlenderTrack 학습 → 네 벤치 평가 → 기준 모델과 한 표" 를 바로 돌리는 공간이다. 시작점은 `README.md`.
저장소 루트 `CLAUDE.md` 의 규칙 (금지 표현 · 복사 기준선 · matched 쌍 · `sed -i` 금지 · 긴 작업은 sbatch) 이 그대로 적용된다.

## 처음에 할 일
1. **`paths.env` 의 데이터셋 경로 · 파이썬을 자기 환경에 맞게 고친다.** 지금 값은 vll7 기준이다 (Blackwell GPU 라 vjepa2_bw, 학습셋은 NFS).
   - 각 경로가 지금 노드에 있는지 `ls` 로 확인하고, 없으면 사용자에게 그 데이터가 어디 있는지 묻는다.
   - 레지스트리 (`configs/protocols/*`, `cvpr/registry/*`, `configs/training/datasets.md`) 는 고치지 않는다. 경로는 `paths.env` 에서만 덮어쓴다.
2. 평가 데이터는 프레임을 새로 뽑지 않는다. IntPhys 1 late 는 dev PNG 링크, Rubric 은 원본 mp4 + 프레임 번호 라벨로 읽는다 (없으면 평가가 알아서 만든다).
3. `DRYRUN=1 bash predictor_lab/train.sh lab_base` 와 `DRYRUN=1 bash predictor_lab/eval.sh z_training/runs/release_vith/latest.pt` 로 경로 · 설정을 GPU 없이 검사한다.
4. `python predictor_lab/table.py --check` 로 표 계산기가 기준 값을 내는지 본다.
   - 차이가 3 칸 (README §3) 을 넘으면 경로나 결과 폴더가 바뀐 것이다.

## 아이디어를 넣을 때
- 노브로 되는 것은 `configs/_template.yaml` 을 복사해 config 한 장으로만 한다 (코드 수정 없음).
- 새 손실 · head 는 `app/vjepa_frozen/where_head.py` 에 config 키로 켜고 끄게 넣는다. 키가 없을 때 기존 팔의 결과가 바뀌면 안 된다.
- 새 config 에는 맨 위에 "무엇을 바꿨나 · 왜 · 어느 팔과 비교하나" 를 주석 세 줄로 적는다.
- 비교는 **같은 seed · 같은 step 의 대조 팔** (`lab_base`) 과 한다.
  - seed 가 하나면 결론에 "seed 1 개" 를 붙인다.

## 실행할 때
- 학습 · 평가는 `train.sh` · `eval.sh` (sbatch) 로만 띄운다. 대화를 막는 foreground 실행 · `sleep` 대기는 하지 않는다.
- 다른 사람의 잡은 건드리지 않는다. `scancel` 은 자기 잡만.
- GPU 를 오래 쓰는 새 학습은 사용자가 요청했을 때만 띄운다.

## 결과를 쓸 때
- 표는 시트 (2027_CVPR_world_model, Overall 탭) 와 같은 이름 · B–R 열 순서다. 새 run 의 이름은 config 의 `lab.sheet_name` 에 적는다.
- 사용자가 시트에 넣어 달라고 하면 `table.py <run> --no-refs --tsv` 의 행을 Google Sheets 커넥터로 해당 섹션 아래에 넣는다 (기존 행은 안 고친다).
- 표는 `table.py` 출력을 그대로 쓰고, 그 값은 산출물에서 다시 계산한 것임을 밝힌다.
- 차이에는 표본 크기를 같이 쓴다.
  - Rubric 은 40 쌍이지만 독립 단위는 영상 20 개다.
  - IntPhys 1 late occ 는 90 쌍이다.
  - IntPhys 1 은 격자 최고 칸이다 (descriptive).
- "학습 도메인 안 (BlenderTrack 검증 400 clip)" 과 "도메인 밖 (네 벤치)" 을 섞어 결론 내지 않는다. 이 공간의 목적은 도메인 밖이다.
- 사용자 문서는 루트 `CLAUDE.md` 의 "사용자에게 보여줄 문서 작성 규칙" 을 따른다.
