#!/usr/bin/env python3
"""[E] IntPhys 2 공유 토큰 캐시 (auto_research, 2026-10-22). 학습 없음.

  /venv/vjepa/bin/python auto_research/scripts/ip2_cache.py --split Main --init          # 메타 + 빈 memmap 만들기
  for k in 0 1; do CUDA_VISIBLE_DEVICES=$((2+k)) /venv/vjepa/bin/python -W ignore auto_research/scripts/ip2_cache.py --split Main --shard $k --nshard 2 & done; wait

출력 <out>/<split>/ (Debug 는 배관 확인용):
  meta.json   영상 순서 = 행 순서. scene_index · name · type · pair_id · is_impossible · condition · camera · difficulty · occluder · n_raw · file
  h.npy       (N, 32, 256, 1280) fp16  블록별 LN(목표 encoder), bf16 계산 (가중치 · 입력 bf16 + autocast; 학습 정답과 같은 처리)
  pix.npy     (N, 64, 256, 256, 3) uint8  채점기 전처리 (10 장마다 = 6 fps, 짧은 변 256 bilinear → 가운데 256) 를 되돌린 픽셀
  p_tf.npy    (N, 31, 256, 1280) fp16  기준 블록 AR (natural_ar_vw_state latest) TF 한 걸음: 행 b = 블록 0..b 로 블록 b+1 예측 (자기 문맥, 32 블록 한 번에)
  done.npy    (N,) bool  행이 채워졌는지
전처리는 analysis.intphys2.dataset 의 함수를 그대로 쓴다 (수정 없음)."""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="Main")
ap.add_argument("--root", default="/workspace/dataset/benchmarks/IntPhys2")
ap.add_argument("--out", default="/workspace/auto_research_cache/intphys2")
ap.add_argument("--init", action="store_true")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--nb", type=int, default=32)
args = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis.intphys2.dataset import build_video_transform, frame_step_indices   # noqa: E402
OUT = os.path.join(args.out, args.split)
NB, NF, S, D, R = args.nb, 2 * args.nb, 256, 1280, 256
m = pd.read_csv(os.path.join(args.root, args.split, "metadata.csv"), dtype=str)
N = len(m)

if args.init:
    os.makedirs(OUT, exist_ok=True)
    meta = []
    for i, r in m.iterrows():
        meta.append(dict(row=int(i), scene_index=r["SceneIndex"], name=r["name"], file=r["file_name"], type=r["type"], pair_id=r["type"].split("_")[0],
                         is_impossible=r["type"].endswith("Impossible"), condition=r["condition"], camera=r["Camera"], difficulty=r["Difficulty"],
                         occluder=r["occluder"], game_name=r["game_name"]))
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=0)
    np.lib.format.open_memmap(os.path.join(OUT, "h.npy"), mode="w+", dtype=np.float16, shape=(N, NB, S, D))
    np.lib.format.open_memmap(os.path.join(OUT, "pix.npy"), mode="w+", dtype=np.uint8, shape=(N, NF, R, R, 3))
    np.lib.format.open_memmap(os.path.join(OUT, "p_tf.npy"), mode="w+", dtype=np.float16, shape=(N, NB - 1, S, D))
    np.save(os.path.join(OUT, "done.npy"), np.zeros(N, bool))
    print(f"init {OUT}: {N} 영상", flush=True)
    sys.exit(0)

_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                          # noqa: E402  (기준 AR 불러오기 · TF 래퍼)
sys.argv = _argv
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
import decord                                                                    # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda")
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc, _load_checkpoint(CK)["target_encoder"], tag="t"); enc = enc.to(dev, torch.bfloat16).eval()
W = FT.NoiseWrap(FT.load_pred("natural_ar_vw_state", "latest.pt"), 32, 0.0).to(dev).eval()
tfm = build_video_transform(256, crop_margin_ratio=1.0)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
Hm = np.load(os.path.join(OUT, "h.npy"), mmap_mode="r+"); Pm = np.load(os.path.join(OUT, "pix.npy"), mmap_mode="r+")
Tm = np.load(os.path.join(OUT, "p_tf.npy"), mmap_mode="r+")
done = np.zeros(N, bool)
rows = [i for i in range(N) if i % args.nshard == args.shard]
for j, i in enumerate(rows):
    vr = decord.VideoReader(os.path.join(args.root, args.split, m.loc[i, "file_name"]), num_threads=2)
    idx = frame_step_indices(vr, 10)[:NF]
    fr = vr.get_batch(idx).asnumpy()
    if len(idx) < NF:                                                            # 짧으면 마지막 장 반복 (기록)
        fr = np.concatenate([fr, np.repeat(fr[-1:], NF - len(idx), 0)])
    x = tfm(fr)                                                                  # (3, T, 256, 256) 정규화
    pix = ((x * STD + MEAN) * 255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0).numpy()
    with torch.no_grad():
        xb = x.view(3, NB, 2, R, R).permute(1, 0, 2, 3, 4).to(dev)               # (NB, 3, 2, R, R)
        hs = []
        for k in range(0, NB, 8):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = enc(xb[k:k + 8].to(torch.bfloat16))
            hs.append(F.layer_norm(h.float(), (h.size(-1),)))
        h = torch.cat(hs)                                                        # (NB, S, D)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = W.tf(h.reshape(1, NB * S, D)).float()                            # (1, NB*S, D): 위치 b → 블록 b+1
    Hm[i] = h.half().cpu().numpy(); Pm[i] = pix; Tm[i] = p.view(NB, S, D)[:-1].half().cpu().numpy()
    done[i] = True
    if j % 20 == 0:
        print(f"[s{args.shard}] {j}/{len(rows)} 영상 {i} 원래 장 {len(vr)} → {len(idx)}", flush=True)
Hm.flush(); Pm.flush(); Tm.flush()
np.save(os.path.join(OUT, f"done_s{args.shard}.npy"), done)
print(f"[s{args.shard}] done {len(rows)}", flush=True)
