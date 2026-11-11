#!/bin/bash
# BlenderTrack 생성기 경로 — 드라이버 · 렌더 잡이 전부 이 파일 하나에서 읽는다 (2026-11-02, Vast 이식).
#   source z_research/scripts/data/blender_track/env.sh        (드라이버가 알아서 한다)
#   값을 바꾸려면 실행 전에 같은 이름의 환경변수로 덮는다 (예: BT_ROOT=/elsewhere bash run_v5mc.sh).
#
# 자리 (BT_SITE) 는 클러스터 파이썬이 있으면 cluster, 없으면 vast 로 저절로 정한다.
#   cluster = 원래 값 그대로 (yaml 의 out · data_root 를 안 바꾼다) → 클러스터에서는 예전과 똑같이 돈다.
#   vast    = 이 머신 값 (아래). yaml 의 out · data_root 앞부분 BT_CLUSTER_ROOT 를 BT_ROOT 로 바꾼 사본으로 돈다 (localize_cfg.py).
#
# 클러스터 (vll 노드) 원래 값 — 기록:
#   BT_PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python          설계 · 고르기 · 묶기 (numpy · pyyaml · decord · PIL)
#   BT_BLENDER_PY=/data/hyuntak/anaconda3/envs/blender/bin/python  렌더 (bpy 5.0.1 모듈, Cycles)
#   HERE=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/data/blender_track
#   out / data_root = /data/hyuntak/project/2026/2027_cvpr/nfs_cache/blender_track/<판>[/work|/data]
#   잡 분배 = sbatch -p batch_vll -w vllN

: "${BT_CLUSTER_HERE:=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/data/blender_track}"
: "${BT_CLUSTER_ROOT:=/data/hyuntak/project/2026/2027_cvpr/nfs_cache/blender_track}"   # yaml out · data_root 의 공통 앞부분
if [[ -z ${BT_SITE:-} ]]; then
  if [[ -x /data/hyuntak/anaconda3/envs/vjepa2/bin/python ]]; then BT_SITE=cluster; else BT_SITE=vast; fi
fi
if [[ -z ${BT_HERE:-} || ! -f ${BT_HERE}/env.sh ]]; then
  BT_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

if [[ $BT_SITE == cluster ]]; then
  : "${BT_PY:=/data/hyuntak/anaconda3/envs/vjepa2/bin/python}"
  : "${BT_BLENDER_PY:=/data/hyuntak/anaconda3/envs/blender/bin/python}"
  : "${BT_ROOT:=}"                                       # 비움 = yaml 의 out · data_root 그대로
  : "${BT_DATA_CSV:=}"                                   # 비움 = <레포>/data_csv (compose_v4vc · make_*_meta 의 목록 자리)
else
  : "${BT_PY:=/venv/vjepa/bin/python}"                   # Python 3.12 · numpy · pyyaml · decord · PIL · matplotlib
  : "${BT_BLENDER_PY:=/venv/blender/bin/python}"         # Python 3.11 · bpy 5.0.1 (uv venv, 2026-11-02)
  : "${BT_ROOT:=/workspace/dataset/training_data/blender_track}"
  : "${BT_DATA_CSV:=${BT_ROOT}/data_csv}"                # 레포 data_csv/ 대신 (v4vc 목록 · v3 팔 목록)
fi
# BT_OUT · BT_DATA_ROOT 를 주면 yaml 의 out · data_root 를 통째로 그 값으로 (BT_ROOT 앞부분 바꾸기보다 먼저).
# BT_CFG_ONLY="s1 s2" (그 분할만 남김) · BT_CFG_SET="a.b=1 c.d=x" (점 경로로 덮기) — 스모크용, localize_cfg.py 가 읽는다.
export BT_SITE BT_HERE BT_PY BT_BLENDER_PY BT_ROOT BT_DATA_CSV BT_CLUSTER_ROOT BT_CLUSTER_HERE
