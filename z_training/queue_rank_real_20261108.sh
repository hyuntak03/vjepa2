#!/bin/bash
# 2026-11-08 (사용자 "1"): 짝 순위 손실 + 실제 영상 사건 판 (v4bc4f_ar_vw_state_rank_real) — 다른 세션의 tmem 학습 · 채점 사슬이 끝나고 GPU 가 비면 올린다.
#   학습 → 첫 step 확인 (오류면 멈춤) → 시트 벤치 · Surrogate → 보류 24 사건 채점 (이 판 · 순위 판 · 음의 손실 판) → 시트 표 · 사건별 표 · 보류 요약.
cd /workspace/world_model/vjepa2
ts() { TZ=Asia/Seoul date '+%F %T KST'; }
RUN=v4bc4f_ar_vw_state_rank_real; M=vast_ar_vw_state_v4bc4f_rank_real
echo "[queue_rr] $(ts) 대기 시작 — chain_tmem_20261108 · tmem 학습이 끝날 때까지"
while pgrep -f '[c]hain_tmem_20261108.sh' >/dev/null || pgrep -f '[l]aunch.py --fname .*natural_ar_vw_state_tmem' >/dev/null; do sleep 120; done
echo "[queue_rr] $(ts) tmem 사슬 끝 — GPU 가 3 분 연속 빌 때까지"
idle=0
while (( idle < 3 )); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
  if (( busy == 0 )); then idle=$((idle + 1)); else idle=0; fi
  sleep 60
done
echo "[queue_rr] $(ts) GPU 빔 → 학습"
GPUS=8 bash z_training/tmux_train.sh $RUN || { echo "⚠️ 학습을 못 올림 $(ts)"; exit 1; }
L=z_training/runs/$RUN/train.log
for i in $(seq 1 40); do
  sleep 30
  if grep -q -E 'Traceback|OutOfMemoryError|rank 실패' $L 2>/dev/null; then
    echo "⚠️ $(ts) 학습 오류 — 멈춤"; grep -m3 -E 'Error|rank 실패' $L; tmux kill-session -t train_$RUN 2>/dev/null; exit 1
  fi
  grep -q ' rank n ' $L 2>/dev/null && break
done
echo "[queue_rr] $(ts) 학습 시작 확인: $(grep ' rank n ' $L | tail -1 | sed -E 's/.*\]\[main +\] //' | cut -c1-160)"
RUN=$RUN MODEL=$M EP=20 GPU_IDS="0 1 2 3 4 5 6 7" bash z_training/chain_full_eval.sh > z_training/logs/chain_full_eval_$M.log 2>&1
GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh intphys1_surrogate $M > z_training/logs/surrogate_eval_$M.log 2>&1
for m in $M vast_ar_vw_state_v4bc4f_rank vast_ar_vw_state_v4bc4f_negimp; do
  GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 bash cvpr/eval/testbed/run.sh realneg_v1_holdout $m > z_training/logs/realneg_holdout_$m.log 2>&1
done
/venv/vjepa/bin/python z_research/scripts/analysis/sheet_table.py --md vast_ar_vw_state_v4bc4_negimp vast_ar_vw_state_v4bc4f_negimp vast_ar_vw_state_v4bc4f_rank $M
/venv/vjepa/bin/python z_research/scripts/analysis/realbench_by_case.py vast_ar_vw_state_v4bc4f_negimp vast_ar_vw_state_v4bc4f_rank $M
/venv/vjepa/bin/python - <<'P'
import csv, json, os
idx = list(csv.DictReader(open("data_csv/realneg_v1_holdout/index.csv")))
blk = {}
for r in idx:
    blk.setdefault(r["block_id"], {})[r["variant"]] = r["video_id"]
print("보류 24 사건 | 사라짐 (가능 vs 불가능) | 흔적 대조 (가능 vs 흔적 가능, 50 % 가 흔적 둔감)")
for m in ("vast_ar_vw_state_v4bc4f_negimp", "vast_ar_vw_state_v4bc4f_rank", "vast_ar_vw_state_v4bc4f_rank_real"):
    f = f"cvpr/results/eval/testbed/realneg_v1_holdout_{m}/per_block.json"
    if not os.path.exists(f):
        print(m, "결과 없음"); continue
    S = {k: (v["all"] if isinstance(v, dict) else v) for k, v in json.load(open(f))["per_video_surprise"].items()}
    a = [S[v["imp_ab"]] > S[v["pos_a"]] for v in blk.values()]
    t = [S[v["imp_ba"]] > S[v["pos_b"]] for v in blk.values()]
    print(f"{m} | {100 * sum(a) / len(a):.1f} | {100 * sum(t) / len(t):.1f}")
P
echo "QUEUE_RR_DONE $(ts)"
