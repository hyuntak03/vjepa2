#!/bin/bash
# predictor_lab/eval.sh 의 본체 (sbatch 안 또는 LOCAL=1). 직접 부르지 말고 eval.sh 를 쓴다.
# 입력 (환경변수): CK (predictor .pt) · E (결과 폴더) · NAME · BENCHES · GPUS [· DRYRUN · FORCE]
# 채점 엔진은 cvpr/harness/launch.sh (표준 하네스 그대로). 데이터 경로만 paths.env 값으로 덮어쓴다.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f $HERE/paths.env ]] || HERE=/data/hyuntak/project/2026/2027_cvpr/vjepa2/predictor_lab   # sbatch 가 스크립트를 복사해 돌릴 때
source "$HERE/paths.env"
cd "$LAB_REPO"
# cvpr 하네스가 쓰는 파이썬 · torchrun · 노드 로컬 루트를 paths.env 값으로 (vll7 Blackwell = vjepa2_bw, z_vll7_training/eval.sh 와 같은 처리)
source "$LAB_REPO/cvpr/env.sh"
export CVPR_PY="$LAB_PY" CVPR_TORCHRUN="$(dirname "$LAB_PY")/torchrun"
[[ -n ${LAB_CVPR_DATA2:-} ]] && export CVPR_DATA2="$LAB_CVPR_DATA2"
L=$LAB_REPO/cvpr/harness/launch.sh
P="model.predictor_checkpoint=$CK"
fail=0; WIN=
run() {   # run <벤치 폴더 이름> <protocol.yaml> <데이터셋> <SET …>
  local out=$1 proto=$2 ds=$3; shift 3
  echo "### $out  ($(date '+%H:%M:%S'))"
  env ${WIN:+WINDOW=$WIN} OUTDIR="$E/$out" TAG="lab_${NAME}_$out" SET="$P $*" GPUS=${GPUS:-1} bash "$L" "$proto" "$ds" vith || { echo "!! 실패 $out"; fail=1; }
}
for b in $BENCHES; do
  case $b in
    vobench)
      # late 행만 고른 index (data_csv/predictor_lab/vobench_late/index_test.csv) — 없으면 paths.env 의 VOB_INDEX 에서 만든다
      VL=$LAB_REPO/data_csv/predictor_lab/vobench_late
      if [[ ! -f $VL/index_test.csv ]]; then
        mkdir -p "$VL"
        "$LAB_PY" -c "import csv,sys; r=csv.DictReader(open(sys.argv[1])); w=csv.DictWriter(open(sys.argv[2],'w',newline=''),r.fieldnames); w.writeheader(); w.writerows(x for x in r if x['occlusion']=='late')" "$VOB_INDEX" "$VL/index_test.csv"
      fi
      run vobench_late cvpr/eval/testbed/protocol.yaml vobench_v1_split2_test \
        "data.root=$VL data.index_csv=index_test.csv data.frames_root=$VOB_FRAMES" ;;
    gravity)
      # gravity test 108 block (문맥 하나 + 미래 여럿: ledge 2 · arc 9). 레지스트리가 scoring.pairing=cross 로 둔다 (문맥 일치 쌍 전부)
      run gravity_test cvpr/eval/testbed/protocol.yaml vobench_v1_gravity \
        "data.root=$GRAV_INDEX_DIR data.index_csv=index_test.csv data.frames_root=$GRAV_FRAMES" ;;
    intphys1)
      for WIN in garrido_w16 garrido_w32; do          # A.8: 창 (C+M) 마다 실행을 나눈다 (공식은 창마다 모델을 그 프레임 수로 짓는다)
        run "intphys1_dev_$WIN" cvpr/eval/intphys1/protocol.yaml intphys1_dev \
          "data.root=$IP1_INDEX_DIR data.frames_root=$IP1_FRAMES"
      done; WIN= ;;
    intphys1_late)
      # 프레임 폴더 = 기록된 창 정의로 dev PNG 를 가리키는 링크. 이 노드에 없으면 만든다 (몇 초)
      [[ -d $IP1L_FRAMES ]] || "$LAB_PY" "$HERE/setup_intphys1_late.py" || { echo "!! IntPhys 1 late 링크 실패"; fail=1; }
      run intphys1_late_uni cvpr/eval/intphys1_late/protocol.yaml intphys1_late_uni \
        "data.root=$IP1L_INDEX_DIR data.frames_root=$IP1L_FRAMES" ;;
    rubric)
      # 원본 mp4 를 그대로 읽는다: index 의 frames 열 (영상마다 원본 프레임 번호 32 개) 만 고른다 — 프레임을 따로 뽑지 않는다
      [[ -f $RUB_INDEX ]] || "$LAB_PY" "$HERE/make_rubric_index.py" || { echo "!! Rubric index 실패"; fail=1; }
      run rubric_uniform cvpr/eval/testbed/protocol.yaml vobench_v1 \
        "data.root=$RUB_SRC data.index_csv=$RUB_INDEX data.frames_root=null data.frames_column=frames window.stride=1 window.frames_start=0" ;;
    *) echo "!! 모르는 벤치: $b (vobench | gravity | intphys1 | intphys1_late | rubric)"; fail=1 ;;
  esac
done
[[ -n ${DRYRUN:-} ]] || echo "표: $LAB_PY predictor_lab/table.py $E"
exit $fail
