#!/bin/bash
#SBATCH --job-name=mergecache
#SBATCH --partition=batch_vll
#SBATCH -w vll5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/slurm_logs/%x_%j.out
#SBATCH --error=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/slurm_logs/%x_%j.err
# -----------------------------------------------------------------------------
# 토큰 캐시 A 뒤에 B 를 이어 붙이고, 인덱스 순서와 맞는지 **검증까지** 한다.
# GPU 는 안 쓴다 (파티션에 CPU 전용이 없어 1장만 잡는다). 순수 디스크 I/O 다.
#
#   sbatch --export=ALL,INTO=<A>,SRC=<B>,RENAME=<새이름>,IDX=<index.csv> \
#          z_research/scripts/harness/sbatch_merge_cache.sh
# -----------------------------------------------------------------------------
set -euo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
source /data/hyuntak/anaconda3/bin/activate vjepa2
: "${INTO:?INTO=<합칠 대상 캐시 디렉토리> 필요}"
: "${SRC:?SRC=<붙일 캐시 디렉토리> 필요}"
: "${RENAME:?RENAME=<병합 후 디렉토리 이름> 필요}"
: "${IDX:?IDX=<대조할 index.csv> 필요}"

echo "== 0) 병합 전 검사 (dry-run)"
python z_research/scripts/harness/merge_token_cache.py --into "$INTO" --from "$SRC" --dry-run

echo "== 1) 병합"
python z_research/scripts/harness/merge_token_cache.py \
  --into "$INTO" --from "$SRC" --rename-to "$RENAME" --free-as-you-go

echo "== 2) 검증: 캐시 video_ids 순서 == index.csv 순서"
NEW="$(dirname "$INTO")/$RENAME"
python - "$NEW" "$IDX" <<'PY'
import csv, json, sys
import numpy as np
cache, idx = sys.argv[1], sys.argv[2]
m = json.load(open(f"{cache}/meta.json"))
ids = [r["video_id"] for r in csv.DictReader(open(idx))]
assert m["video_ids"] == ids, "치명: 캐시 순서와 index 순서가 다르다"
print(f"  video_ids {len(ids)} 일치")
for b, tok in m["base_counts"].items():
    a = np.load(f"{cache}/{b}.npy", mmap_mode="r")
    assert a.shape == (len(ids), tok, m["embed_dim"]), f"{b}: shape {a.shape} 이상"
    # 이어붙인 경계 앞뒤가 0 으로 채워지지 않았는지 (쓰기 누락 탐지)
    lo, hi = a[0], a[-1]
    assert np.isfinite(np.asarray(lo, np.float32)).all(), f"{b}: 첫 clip 에 NaN/Inf"
    assert np.abs(np.asarray(hi, np.float32)).max() > 0, f"{b}: 마지막 clip 이 전부 0"
    print(f"  {b:12s} {a.shape} OK  (첫/마지막 clip 검사 통과)")
print("  검증 통과")
PY
echo "== 끝"
