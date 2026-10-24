#!/bin/bash
#SBATCH --job-name=pixdec_render
#SBATCH --partition=batch_vll
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=45G
#SBATCH --time=0-00:30:00             # 짧아야 backfill 로 바로 들어간다 (10-21: 2 시간이면 Priority 대기)
#SBATCH --output=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_pixel_decoder/runs/slurm_logs/%x_%j.out
# render_future.py 를 다른 노드에서 (2026-10-21). 그 노드엔 VOBench 가 없으므로 쓸 프레임만 NFS 로 복사해 둔 자리를 ROOT 로 준다:
#   python z_pixel_decoder/render_future.py --list-frames ... (vll5) → runs/_frames/VOBench_v1 로 복사 →
#   sbatch -w vll3 --time=0-00:30:00 --export=ALL,SETS="late gravity",MODELS="vith vith_release_vobftg" z_pixel_decoder/render.sbatch
#   VideoMAE 는 SCRIPT=render_videomae.py (MODELS 없이)
#   ⚠️ --export 는 쉼표로 변수를 끊는다 → 목록은 공백으로 주고 여기서 쉼표로 바꾼다 (10-21: ARGS="--sets late,gravity" 가 late 만 갔다)
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
source /data/hyuntak/anaconda3/bin/activate vjepa2
export PYTHONPATH=$PWD
ROOT=${ROOT:-$PWD/z_pixel_decoder/runs/_frames/VOBench_v1}
echo "node $(hostname) · CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES · root $ROOT · args $ARGS"
ARGS="${ARGS:-} ${SETS:+--sets $(tr ' ' ',' <<<"$SETS")} ${MODELS:+--models $(tr ' ' ',' <<<"$MODELS")}"
echo "args $ARGS"
python z_pixel_decoder/${SCRIPT:-render_future.py} --root "$ROOT" ${ARGS}
