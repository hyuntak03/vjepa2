#!/bin/bash
# (사본, 11-01) 칸 채우기 번호를 먼저 그리는 렌더 잡 — gen_render.sh 와 같고 EXTRA_DIR 만 더 받는다. 실행 중인 gen_render.sh 를 고치면 bash 가 바이트 위치로 이어 읽다 깨진다.
# BlenderTrack v1 렌더 잡 (sbatch 본체, pipeline.sh 가 띄운다). 노드 하나에서 GPU 마다 작업자 PER_GPU 개, 분할 여러 개를 차례로.
# 입력 (환경변수): CFG (설정 yaml) · SPLITS ("train val ood") · NODE_I (이 노드 번호) · N_NODES · GPUS · PER_GPU · THREADS
# 작업자 번호는 노드를 가로질러 전역이다 (노드 i 의 작업자 w → 전역 W_OFF + w, 기본 W_OFF = i × GPUS × PER_GPU). 이어 하기: 끝난 clip 은 건너뛴다.
set -u
B=/data/hyuntak/anaconda3/envs/blender/bin/python
HERE=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/data/blender_track
OUT=$(/data/hyuntak/anaconda3/envs/vjepa2/bin/python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['out'])" "$CFG")
mkdir -p "$OUT/logs"
NL=$((GPUS * PER_GPU)); NW=${NW_TOTAL:-$((N_NODES * NL))}; WOFF=${W_OFF:-$((NODE_I * NL))}   # 노드마다 GPU 수가 다르면 pipeline 이 W_OFF · NW_TOTAL 을 준다
echo "host $(hostname) | splits $SPLITS | node $NODE_I/$N_NODES | workers $NL (global $NW) | $(date)"
for SPLIT in $SPLITS; do
  for ((w = 0; w < NL; w++)); do
    g=$((w % GPUS)); W=$((WOFF + w))
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 $B "$HERE/gen_render.py" --config "$CFG" --split "$SPLIT" --worker $W --nworkers $NW ${CLAIM:+--claim} ${EXTRA_DIR:+--extra_list $EXTRA_DIR/extra_$SPLIT.txt} \
      > "$OUT/logs/${SPLIT}_w${W}.log" 2>&1 &
  done
  wait
  echo "split $SPLIT done $(date)"
done
fail=$(grep -l "Traceback" "$OUT"/logs/*_w*.log 2>/dev/null | head -3)
[[ -n $fail ]] && { echo "!! 오류 로그: $fail"; exit 1; }
exit 0
