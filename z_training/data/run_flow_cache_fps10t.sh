#!/usr/bin/env bash
# fps 10 시간 기준 flow 캐시 — K400 · SSv2 를 GPU 4 장에 나눠 뽑는다 (2026-10-27).
#   GPU i 가 K400 shard i → SSv2 shard i 를 차례로 (GPU 당 1 프로세스, chunk 64).
#   출력은 RAM 디스크 ($TMP_OUT, 기본 /dev/shm/flow_cache) 에 쓰고, shard 하나가 끝나면 $OUT (NFS) 로 복사한 뒤 RAM 에서 지운다.
#
# 벤치 · 실측 (vll7 RTX PRO 6000, RAFT-large 12 iter · 256 px · 전후 일관성):
#   * GPU 당 약 250 쌍/s 에서 GPU 연산이 병목 — 프로세스 2 · 3 개, 정역방향 합치기, torch.compile, batch 128 모두 더 빠르지 않았다
#     (batch 256 은 torchvision RAFT 의 grid_sample 이 cuDNN 한계로 실패).
#   * OMP 스레드를 안 묶으면 프로세스마다 torch 스레드 풀 72 개가 디코드와 코어를 다툰다 (GPU 8–45 %) → 4 개로 묶는다.
#   * 출력 memmap 을 NFS (/data) 에 바로 쓰면 파일이 커지면서 메인 스레드가 시간의 약 19 % 를 nfs_wait_on_request 로 보내
#     250 → 154 쌍/s 로 떨어졌다 → RAM 디스크에 쓴다.
#   예상: K400 약 6.3 h + SSv2 약 1.5 h (+ 복사).
#
#   bash z_training/data/run_flow_cache_fps10t.sh            # 백그라운드로 띄우고 바로 돌아온다
#   tail -n 1 /data/hyuntak/flow_cache/logs_fps10t/*.log     # 진행
# 표: z_training/data/build_time_fps_table.py → data_csv/vll7_{k400_320,ssv2}/fps10t_table.tsv
# ⚠️ RAM 디스크는 재부팅 · 노드 장애 때 사라진다. 끝난 shard 만 NFS 에 남는다.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY=${PY:-/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python}
OUT=${OUT:-/data/hyuntak/flow_cache}
TMP_OUT=${TMP_OUT:-/dev/shm/flow_cache}
NG=${NG:-4}
DW=${DW:-14}                                  # 프로세스당 디코드 스레드
LOG="$OUT/logs_fps10t"; mkdir -p "$LOG" "$TMP_OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4}
cd "$REPO"
for i in $(seq 0 $((NG - 1))); do
  setsid nohup bash -c "
    set -e
    for ds in k400_320 ssv2; do
      name=vll7_\${ds}_fps10t
      CUDA_VISIBLE_DEVICES=$i $PY z_training/data/build_flow_cache.py --table data_csv/vll7_\${ds}/fps10t_table.tsv \
          --out $TMP_OUT/\$name/s$i --grid 32 --chunk 64 --decode-workers $DW --num-shards $NG --shard $i --device cuda:0 \
          > $LOG/\${ds}_s$i.log 2>&1
      mkdir -p $OUT/\$name
      cp -r $TMP_OUT/\$name/s$i $OUT/\$name/ && echo \"복사 끝 \$name/s$i \$(date)\" >> $LOG/\${ds}_s$i.log
      rm -r $TMP_OUT/\$name/s$i
    done
  " > "$LOG/runner_$i.out" 2>&1 < /dev/null &
  echo "GPU $i 시작 (pid $!)"
done
