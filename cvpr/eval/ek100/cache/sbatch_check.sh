#!/bin/bash
#SBATCH --job-name=ek_check
#SBATCH --partition=batch_vll
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/cvpr/logs/%x_%j.out
# 캐시 검사 (그 노드 /data2 의 캐시): train view 재현 · int8 오차 · meta 행 수.  sbatch -w vll6 --export=ALL,CFG=<yaml> cvpr/eval/ek100/cache/sbatch_check.sh
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2; source /data/hyuntak/anaconda3/bin/activate vjepa2
for v in 0 13; do python cvpr/eval/ek100/cache/check_train_q8.py --config $CFG --view $v --chunk 7 2>&1 | grep -E "재현|상대|rror"; done
python - <<PY
import numpy as np, yaml, pathlib
d=yaml.safe_load(open("$CFG"))["experiment"]["data"]
C=pathlib.Path("/data2/local_datasets/world/world_analysis/cache/ek100_feats")/f"{d['anticipation_point_mode']}_{d['time_source']}_{pathlib.Path(d['dataset_train']).stem}"
for sp in ("train","val"):
    m=np.load(C/sp/"meta.npz"); print(sp, "rows", len(m["row"]), "valid", int(m["valid"].sum()), "views", len(set(m["view"].tolist())))
PY
df -h /data2 | tail -1
