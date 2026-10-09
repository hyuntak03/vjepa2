#!/bin/bash
# ⚠️ 2026-09-27: ep5/19/30/41/43 체크포인트가 지워져 **재실행 불가** (기록용). 산출물은 auto_research/_stage/results_vll3/scene_*_arielep/ 에만 남는다.
# 2026-09-26 묶음 R (REVIEW_v2 1 순위): Ariel epoch × reach 포화 검정 — ep5 · 19 · 30 · 41 (ep43 은 기본 팔 A) 을 팬 · SSv2 · EK100 같은 판에.
#   구조 문제 (포화) 인지 학습 부족 (계속 증가) 인지 가른다. 학습 없음.
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; S=auto_research/scripts; C=/data2/local_datasets/world/world_analysis/cache/auto_research
A=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/ariel/block_causal_future_only_1e_5_ep
run() { echo "[$(date)] >>> $*"; "$@"; r=$?; echo "[$(date)] <<< exit $r"; return $r; }
unset ARLIB_PRED_CKPT
export SCENE_EXTRA="E5=${A}_5/latest.pt,E19=${A}_19/latest.pt,E30=${A}_30/latest.pt,E41=${A}_41/latest.pt" SCENE_ONLY_EXTRA=1
run $P $S/e_scene_reach.py --gpus 1 --limit 4 --out $C/_smoke/scene_ep || exit 1
run $P $S/e_scene_reach.py --gpus ${G:-8} --n 800 --out $C/scene_pan_arielep
for src in ssv2 ek100; do run $P $S/e_scene_reach_nat.py --source $src --gpus ${G:-8} --out $C/scene_nat_${src}_arielep; done
