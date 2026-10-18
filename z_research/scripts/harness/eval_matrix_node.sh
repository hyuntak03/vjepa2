#!/bin/bash
# 평가 행렬 칸을 vll5 밖 노드 (데이터 없는 노드) 에서 돌린다 (2026-10-15). sbatch 로 cvpr/harness/submit.sh 가 부른다.
#   NODE=vll1 GPUS=3 JOBNAME=mx_ip2 bash cvpr/harness/submit.sh z_research/scripts/harness/eval_matrix_node.sh intphys2:vith_vast_ar_state_ep70
#   인자 = <벤치>:<모델> 를 순서대로 (벤치 = intphys2 · inflevel · grasp). 한 job 안에서 차례로 돈다.
# 노드 로컬 디스크에 쓰지 않는다 (vll1 은 / · /data2 가 100 % 찼다 — 10-15 실측). 벤치 데이터는 NFS 사본을 바로 읽는다:
#   IntPhys 2 → SET data.root=/data/dataset/world/IntPhys2 (Main 파일 목록이 vll5 /data2 사본과 같음, 10-15 확인)
#   GRASP     → SET data.root=data_csv/grasp_level2_nfs (index 의 /data2/.../GRASP/ 를 /data/dataset/world/Benchmarks/GRASP/ 로 바꾼 사본, 파일 목록 같음)
#   InfLevel  → index 가 원래 NFS 경로
# 결과 폴더 이름은 SET 과 무관하게 레지스트리 이름 그대로 (intphys2_main_<모델>_wip2_w* · grasp_level2_<모델>_wgarrido_w*).
set -uo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
INF="inflevel_continuity inflevel_gravity inflevel_solidity"
for job in "$@"; do
  b=${job%%:*}; m=${job#*:}
  echo "[node $(hostname) $(date +%m-%d_%H:%M:%S)] 시작 $b $m"
  case $b in
    intphys2) SET="data.root=/data/dataset/world/IntPhys2 ${SET:-}" bash cvpr/eval/intphys2/run.sh intphys2_main "$m" ;;
    inflevel) bash cvpr/eval/intphys1/run.sh "$INF" "$m" ;;
    grasp)    SET="data.root=$PWD/data_csv/grasp_level2_nfs ${SET:-}" bash cvpr/eval/intphys1/run.sh grasp_level2 "$m" ;;
  esac
  echo "[node $(hostname) $(date +%m-%d_%H:%M:%S)]   끝 $b $m rc=$?"
done
