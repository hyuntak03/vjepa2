#!/bin/bash
# 2026-11-06 사용자 "Vobench v3 하라니까": v4b 다섯 판의 e5 체크포인트를 VOBench v3 로 중간 채점 (학습과 GPU 를 같이 씀, 채점만).
cd /workspace/world_model/vjepa2
for pair in v4b_ar_vw_state_postft:vast_ar_vw_state_v4b v4b_ar_vw_state_negimp:vast_ar_vw_state_v4b_negimp v4b_ar_vw_state_negpair_w3:vast_ar_vw_state_v4b_negpair_w3 \
            v4b_release_postft:vith_release_v4b v4b_release_negpair_w3:vith_release_v4b_negpair_w3; do
  R=${pair%%:*}; M=${pair##*:}
  SET="model.predictor_checkpoint=/workspace/world_model/vjepa2/z_training/runs/$R/e5.pt" SUFFIX=_e5 GPU_IDS="0 1 2 3 4 5 6 7" GPUS=8 \
    bash cvpr/eval/testbed/run.sh vobench_v3 $M > z_training/logs/vob3_e5_$M.log 2>&1
  echo "$(TZ=Asia/Seoul date '+%T KST') $M exit $?"
done
echo VOB3_E5_DONE
