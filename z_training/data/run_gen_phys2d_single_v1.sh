#!/bin/bash
# phys2d_single_v1 (2026-10-19) — 물체 1 개 · 시간 고른 가림. train 18 시나리오 × 2200 · val × 80. 끊기면 같은 명령으로 이어 한다.
#
#   PY=<python> OUT=<저장 위치> WORKERS=<n> bash z_training/data/run_gen_phys2d_single_v1.sh
#
# 재현: 라벨은 (코드, seed, requirements_phys_gen.txt 버전) 으로 정해진다 — 워커 수 · 기계 · 실행 순서와 무관.
#   버전이 다르면 멈춘다 (ALLOW_VERSION_MISMATCH=1 로 무시). 끝나면 지문을 만들고, val 은 레포의 기준 지문과 대조한다.
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-/venv/vjepa/bin/python}
OUT=${OUT:-/workspace/dataset/training_data/phys2d_single_v1}
WORKERS=${WORKERS:-$(( $(nproc) - 8 ))}
REF=z_training/data/fingerprints/phys2d_single_v1_val.tsv
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $OUT
$PY - <<'EOF' | tee $OUT/versions.txt
import sys, numpy, cv2, pymunk, subprocess, os
want = dict(l.split("==") for l in open("z_training/data/requirements_phys_gen.txt").read().split() if "==" in l)
have = {"numpy": numpy.__version__, "opencv-python": cv2.__version__, "pymunk": pymunk.version}
print("python", sys.version.split()[0], "|", " ".join(f"{k}={v}" for k, v in have.items()), "|",
      subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0])
bad = [k for k in have if not want[k].startswith(have[k])]
if bad and os.environ.get("ALLOW_VERSION_MISMATCH") != "1":
    sys.exit(f"버전 불일치 {bad} — requirements_phys_gen.txt 를 맞추거나 ALLOW_VERSION_MISMATCH=1")
EOF
nice -n 10 $PY -W ignore z_training/data/gen_phys2d_single.py --per-scenario 80   --seed 2 --workers $WORKERS --preview-n 0 --out $OUT/val   > $OUT/gen_val.log 2>&1
nice -n 10 $PY -W ignore z_training/data/gen_phys2d_single.py --per-scenario 2200 --seed 1 --workers $WORKERS --preview-n 3 --debug-overlay --out $OUT/train > $OUT/gen_train.log 2>&1
$PY z_training/data/audit_phys2d_single.py --root $OUT/val   > $OUT/val/audit.md
$PY z_training/data/audit_phys2d_single.py --root $OUT/train > $OUT/train/audit.md
$PY z_training/data/fingerprint_phys_data.py --root $OUT/val   --write $OUT/val/fingerprint.tsv
$PY z_training/data/fingerprint_phys_data.py --root $OUT/train --write $OUT/train/fingerprint.tsv
if [ -f $REF ]; then $PY z_training/data/fingerprint_phys_data.py --root $OUT/val --check $REF | tee $OUT/val/fingerprint_check.txt; fi
