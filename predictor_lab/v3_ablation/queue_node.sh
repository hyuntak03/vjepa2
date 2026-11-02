#!/bin/bash
# v3 ablation 대기열 — 노드 하나를 통째로 잡고, 칸 (GPU 쌍) 마다 대기열 (runs.tsv 순서) 에서 다음 학습을 가져와 돈다 (2026-11-01).
#   두 노드가 같은 대기열을 나눠 쓴다: 학습마다 lock 디렉터리 (mkdir 은 원자적) 로 한 번만 잡힌다. 끝난 학습 (latest.pt + done) 은 건너뛴다.
#   입력 (환경변수): SLOTS ("0 1|2 3|..." 칸마다 GPU 묶음, 같은 묶음을 여러 번 쓰면 겹쳐 돈다) · BATCH (GPU 당, 전역 = BATCH × GPU 수 = 32 로 맞춘다) · LAB_PY · THREADS
#   sbatch -w vll7 --gres=gpu:6 ... --export=ALL,SLOTS="0 1|0 1|0 1|2 3|2 3|2 3|4 5|4 5|4 5",LAB_PY=... predictor_lab/v3_ablation/queue_node.sh
set -u
R=/data/hyuntak/project/2026/2027_cvpr/vjepa2
Q=$R/predictor_lab/v3_ablation/queue; mkdir -p "$Q"
RUNS=$R/predictor_lab/v3_ablation/runs.tsv
CLIPS=/data/hyuntak/project/2026/2027_cvpr/nfs_cache/blender_track/v3/clips
IFS='|' read -r -a SL <<<"$SLOTS"
echo "$(hostname) $(date '+%F %T') 칸 ${#SL[@]}: ${SLOTS}"
slot() {
  local k=$1 devs=$2
  while true; do
    local got=""
    while IFS=$'\t' read -r name arm seed; do
      [[ -z $name || $name == \#* ]] && continue
      [[ -f $Q/$name.done ]] && continue
      mkdir "$Q/$name.lock" 2>/dev/null || continue
      got=$name; break
    done < "$RUNS"
    [[ -z $got ]] && { echo "칸 $k ($devs): 대기열 빔 $(date '+%T')"; return 0; }
    arm=$(awk -F'\t' -v n="$got" '$1==n {print $2}' "$RUNS"); seed=$(awk -F'\t' -v n="$got" '$1==n {print $3}' "$RUNS")
    META=$R/data_csv/blender_track_v3/arms/$arm.npz
    # 11-01 v3t: 클립 폴더는 목록 (npz) 의 root 를 쓴다 (v3t · 섞은 팔은 v3 가 아니다). v3 팔은 같은 값이라 동작 그대로
    ROOT=$(/data/hyuntak/anaconda3/envs/vjepa2/bin/python -c "import numpy as np,sys; print(np.load(sys.argv[1])['root'])" "$META" 2>/dev/null || echo "$CLIPS")
    SET="data.datasets=[{\"type\":\"blender_track\",\"meta\":\"$META\",\"root\":\"$ROOT\"}] meta.seed=$seed data.batch_size=${BATCH:-16}"
    port=$(( ${PORT_BASE:-$(( 29000 + (${SLURM_JOB_ID:-0} % 200) * 20 ))} + k ))   # 같은 잡에 칸을 더 붙일 때는 PORT_BASE 로 피한다
    echo "칸 $k ($devs): 시작 $got (목록 $arm, seed $seed, port $port) $(date '+%T')"
    CONFIG=lab_v3_base NAME=$got GPUS=$(wc -w <<<"$devs") DEVS="$devs" TRAIN_DDP_PORT=$port THREADS=${THREADS:-4} SET_B64=$(printf %s "$SET" | base64 -w0) \
      bash $R/predictor_lab/train_job.sh > "$Q/$got.$(hostname).log" 2>&1
    rc=$?
    if [[ $rc -eq 0 && -f $R/z_training/runs/$got/latest.pt ]]; then touch "$Q/$got.done"; echo "칸 $k: 끝 $got $(date '+%T')"
    else
      nf=$(( $(cat "$Q/$got.nfail" 2>/dev/null || echo 0) + 1 )); echo $nf > "$Q/$got.nfail"
      if (( nf < 3 )); then                                    # 11-01: 메모리 순간 초과 (rank -9) 로 죽은 학습을 대기열에서 잃지 않게 — 되돌리고 칸은 쉰다
        rm -rf "$Q/$got.lock" "$R/z_training/runs/$got"
        echo "칸 $k: 실패 $got (rc $rc, $nf 번째) — 대기열로 되돌리고 3 분 쉼"; sleep 180
      else echo "칸 $k: 실패 $got (rc $rc, $nf 번째) — 그만 (로그 $Q/$got.$(hostname).log)"; fi
    fi
  done
}
for k in "${!SL[@]}"; do slot "$k" "${SL[$k]}" & sleep 20; done
wait
echo "$(hostname) 모든 칸 끝 $(date '+%F %T')"
