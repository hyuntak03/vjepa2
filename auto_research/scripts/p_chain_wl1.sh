#!/bin/bash
# [P] [T] 가중 L1 이어 학습 체크포인트가 나오면 차례로 판정 (GPU 하나 · 스레드 8). 사용: bash p_chain_wl1.sh
cd /workspace/world_model/vjepa2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${GPU:-5}
PY=/venv/vjepa/bin/python; D=/workspace/auto_research_cache/ar_ft
while pgrep -f "stab_mechanism.py --ft ft_l1_lr1e4" > /dev/null; do sleep 20; done
stable() { [ -f $1 ] || return 1; s1=$(stat -c %s $1); sleep 30; s2=$(stat -c %s $1); [ "$s1" = "$s2" ]; }
$PY -W ignore auto_research/scripts/sample_presence.py --run natural_ar_vw_state latest.pt --ft $D/ft_l1_lr1e4.pt --K 1 > auto_research/logs/sp_ft_l1_lr1e4.log 2>&1
echo "sp ft_l1_lr1e4 $(date +%H:%M)"
for n in ft_wl1 ft_wl1_lr1e4; do
  until stable $D/$n.pt; do sleep 30; done
  echo "found $n $(date +%H:%M)"
  $PY -W ignore auto_research/scripts/sample_presence.py --run natural_ar_vw_state latest.pt --ft $D/$n.pt --K 1 > auto_research/logs/sp_$n.log 2>&1
  $PY -W ignore auto_research/scripts/stab_mechanism.py --ft $n --only_ft 1 > auto_research/logs/stab_mechanism_$n.log 2>&1
  echo "done $n $(date +%H:%M)"
done
$PY -W ignore auto_research/scripts/vob_late_hyp.py --motion flat_static --ft ft_l1 ft_l1_lr1e4 ft_wl1 ft_wl1_lr1e4 --out auto_research/results/vob_late_wl1_static.json > auto_research/logs/vob_late_wl1_static.log 2>&1
echo "vob done $(date +%H:%M)"
