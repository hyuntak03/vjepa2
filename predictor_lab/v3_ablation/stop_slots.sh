#!/bin/bash
# 칸 (slot) 끄기: 그 칸이 지금 돌리는 학습 이름들을 받아, 그 학습의 train_job.sh 를 낳은 칸 subshell 과 학습 전체를 끝낸다 (vll7 안에서 srun --overlap 으로).
for name in "$@"; do
  lp=$(pgrep -f "z_training/runs/$name/config.yaml" | head -1)
  [[ -z $lp ]] && { echo "$name: 실행 중 아님"; continue; }
  tj=$(ps -o ppid= -p $lp | tr -d ' ')                       # launch.py 의 부모 = train_job.sh (bash)
  sl=$(ps -o ppid= -p $tj | tr -d ' ')                       # 그 부모 = 칸 subshell
  echo "$name: launch $lp · train_job $tj · 칸 $sl"
  kill -STOP $sl 2>/dev/null; pkill -TERM -P $tj; kill -TERM $tj $lp 2>/dev/null; sleep 5
  pkill -KILL -f "z_training/runs/$name/config.yaml"; kill -KILL $sl 2>/dev/null
done
