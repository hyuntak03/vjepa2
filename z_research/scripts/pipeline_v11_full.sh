#!/bin/bash
# v11_full probing 파이프라인 — 캐시를 다시 뽑지 않고 **이어 붙여서** 만든다.
#
#   1) extract   v11_earlymid 캐시만 새로 뽑는다        8 GPU  ~32분
#   2) merge     v11_vith + v11_earlymid_vith 이어붙임  1 GPU  ~20분 (순수 I/O)
#   3) probe     v11_full 12조건 probing                12 job x 1 GPU  ~20분
#
# **job 이 job 을 제출하지 않는다.** 사람이 이 스크립트를 돌릴 때 전부 미리 제출하고
# --dependency=afterok 로 잇는다 (2026-08-30 의 자기제출 폭주를 구조적으로 막는다).
#
#   bash z_research/scripts/pipeline_v11_full.sh          # 검사만 하고 제출
#   CHECK_ONLY=1 bash z_research/scripts/pipeline_v11_full.sh   # 검사만
set -euo pipefail
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
CACHE=/local_datasets/world/world_analysis/cache
GPUS_EX=${GPUS_EX:-8}

echo "===== 사전 검사 ====="
fail=0
chk() { if eval "$2"; then echo "  OK   $1"; else echo "  실패 $1"; fail=1; fi; }

chk "v11 캐시 존재"            "[[ -f $CACHE/v11_vith/meta.json ]]"
chk "earlymid 캐시 없음(새로 뽑을 것)" "[[ ! -d $CACHE/v11_earlymid_vith ]]"
chk "v11_full 캐시 없음"       "[[ ! -d $CACHE/v11_full_vith ]]"
chk "v11_full index 존재"      "[[ -f data_csv/intphysgen_v11_full/index.csv ]]"
chk "v11_full index_probe 존재" "[[ -f data_csv/intphysgen_v11_full/index_probe.csv ]]"
chk "earlymid index 존재"      "[[ -f data_csv/intphysgen_v11_earlymid/index.csv ]]"

# 순서 검증 — 이게 어긋나면 병합해도 캐시가 안 맞는다
python - <<'PY' || fail=1
import csv, json, sys
cache = json.load(open("/local_datasets/world/world_analysis/cache/v11_vith/meta.json"))["video_ids"]
em   = [r["video_id"] for r in csv.DictReader(open("data_csv/intphysgen_v11_earlymid/index.csv"))]
full = [r["video_id"] for r in csv.DictReader(open("data_csv/intphysgen_v11_full/index.csv"))]
ok = (cache + em) == full
print(f"  {'OK  ' if ok else '실패'} index 순서 == (v11 캐시 순서 + earlymid 순서)  [{len(full)} clip]")
sys.exit(0 if ok else 1)
PY

# 디스크 — 추출에 필요한 양 + 병합 중 순간 증가(target base 하나치)
python - <<'PY' || fail=1
import os, sys, json
import numpy as np
st = os.statvfs("/local_datasets/world/world_analysis/cache")
free = st.f_bavail * st.f_frsize / 2**30
need_extract = sum(os.path.getsize(f"/local_datasets/world/world_analysis/cache/v11_vith/{b}.npy")
                   for b in ("ctx_masked", "target", "predictor")) / 2**30
peak = os.path.getsize("/local_datasets/world/world_analysis/cache/v11_vith/target.npy") / 2**30
ok = free > need_extract + peak + 50
print(f"  {'OK  ' if ok else '실패'} 디스크 여유 {free:.0f} GiB  "
      f"(추출 {need_extract:.0f} + 병합 순간 {peak:.0f} + 여유 50 필요)")
sys.exit(0 if ok else 1)
PY

echo "  -- DRYRUN 으로 두 config 를 해석해 본다"
DRYRUN=1 bash z_research/scripts/run.sh attn_probe v11_earlymid vith >/tmp/_dry_em.txt 2>&1 \
  && echo "  OK   attn_probe v11_earlymid 해석" || { echo "  실패 v11_earlymid"; fail=1; }
DRYRUN=1 bash z_research/scripts/run.sh attn_probe v11_full vith >/tmp/_dry_full.txt 2>&1 \
  && echo "  OK   attn_probe v11_full 해석" || { echo "  실패 v11_full"; fail=1; }
NG=$(grep -cE "^  - - " /tmp/_dry_full.txt || true)
chk "v11_full sweep 이 12 조건 ($NG)" "[[ $NG -eq 12 ]]"
BASE=$(awk '$1=="output_dir:"{print $2}' /tmp/_dry_full.txt | tail -1)
chk "output_dir 확인 ($BASE)" "[[ -n '$BASE' ]]"

# GSPLIT 의 조건 이름이 실제 sweep 에 다 있는지
GSPLIT_ALL="static_visible,moving_visible_flat,moving_visible \
static_occlusion,moving_occlusion_flat,moving_occlusion \
static_occlusion_early,moving_occlusion_flat_early,moving_occlusion_early \
static_occlusion_mid,moving_occlusion_flat_mid,moving_occlusion_mid"
miss=0
for gs in $GSPLIT_ALL; do
  for c in ${gs//,/ }; do grep -q -- "- - $c$" /tmp/_dry_full.txt || { echo "  실패 GSPLIT 조건 없음: $c"; miss=1; }; done
done
[[ $miss -eq 0 ]] && echo "  OK   GSPLIT 조건 12개 전부 sweep 에 있다" || fail=1

[[ $fail -eq 0 ]] || { echo; echo "사전 검사 실패 — 제출하지 않는다"; exit 1; }
echo "===== 사전 검사 통과 ====="
[[ -z "${CHECK_ONLY:-}" ]] || { echo "CHECK_ONLY=1 — 제출하지 않는다"; exit 0; }

echo
echo "===== 제출 ====="
EMBASE=$(DRYRUN=1 bash z_research/scripts/run.sh attn_probe v11_earlymid vith 2>/dev/null \
         | awk '$1=="output_dir:"{print $2}' | tail -1)
_b64() { printf %s "$1" | base64 -w0; }
EXSET="probing.fit_groups_sweep=[null] probing.optims.attn_30.num_epochs=1"
JEX=$(sbatch --parsable --job-name=ex_earlymid --gres=gpu:"$GPUS_EX" \
  --export=ALL,WMA_RUN=1,P=attn_probe,D=v11_earlymid,M=vith,GPUS="$GPUS_EX",OUTDIR="$EMBASE/_prep",SET_B64="$(_b64 "$EXSET")" \
  z_research/scripts/sbatch.sh)
echo "1) extract  $JEX   (GPUS=$GPUS_EX)  -> $EMBASE/_prep"

JMG=$(sbatch --parsable --job-name=mergecache --dependency=afterok:"$JEX" \
  --export=ALL,INTO="$CACHE/v11_vith",SRC="$CACHE/v11_earlymid_vith",RENAME=v11_full_vith,IDX=data_csv/intphysgen_v11_full/index.csv \
  z_research/scripts/harness/sbatch_merge_cache.sh)
echo "2) merge    $JMG   (afterok:$JEX)"

PREP_JID="$JMG" SPLIT="shape color env" GSPLIT="$GSPLIT_ALL" \
  GPUS=1 GPUS_PROBE=1 P=attn_probe D=v11_full M=vith \
  bash z_research/scripts/sbatch.sh

echo
echo "진행:    watch -n 1 bash z_research/scripts/monitor.sh"
echo "합치기:  python z_research/scripts/analysis/merge_probe_runs.py --base $BASE"
