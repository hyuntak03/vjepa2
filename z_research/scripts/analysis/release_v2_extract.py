#!/usr/bin/env python3
"""VOBench_v2 late · visible 가능 영상의 z · p · h 추출 (릴리즈 ViT-H) — 블록별 probing 용 (2026-10-31, z_analysis/release).

정의 (VOBench_v1 캐시와 같은 함수 · 같은 관례 — vobench_tubelet_probe.py cfg_probe · _worker, transfer_gap_extract.py):
  z[t] = LN(context_encoder(문맥 16 장))[t]        t = 블록 1..8     (fp16, (N, 8, 256, 1280))
  p[t] = predictor(context_encoder 출력, mask_index 0)[t]  t = 블록 9..16  (fp16, 원값)
  h_fut[t] = LN(target_encoder(32 장))[t]            t = 블록 9..16 (fp16) — 판독기 이식 (encoder 판독기 → p) 용 (10-31 추가, --what h · zph)
  pooled.npz = z · p 의 블록 평균 토큰 (256 칸 평균, fp32, (N, 8, 1280))
  probing 관례 bfloat16 · autocast none. 프레임은 NFS 바이트 사본 (stage_vobench_v2_late.py), 리사이즈는 하네스 그대로.
출력: /data/hyuntak/project/2026/2027_cvpr/nfs_cache/release_{late,visible}_vobench_v2/{z.npy, p.npy, h_fut.npy, pooled.npz, meta.json, _DONE}

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  sbatch -p batch_vll -w vll2 --gres=gpu:1 -c 8 --mem=48G --wrap "$P z_research/scripts/analysis/release_v2_extract.py"      # late z · p (NFS 프레임 사본)
  sbatch -p batch_vll -w vll5 --gres=gpu:1 -c 8 --mem=48G --wrap "$P z_research/scripts/analysis/release_v2_extract.py --occ late --what h --frames /data2/local_datasets/world/Benchmark/VOBench_v2"
  sbatch -p batch_vll -w vll5 --gres=gpu:1 -c 8 --mem=48G --wrap "$P z_research/scripts/analysis/release_v2_extract.py --occ visible --what zph --frames /data2/local_datasets/world/Benchmark/VOBench_v2"
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
CFG = REPO / "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/_resolved.yaml"
FRAMES = "/data/hyuntak/project/2026/2027_cvpr/nfs_cache/VOBench_v2_late_frames"
OUT_OF = lambda occ: Path(f"/data/hyuntak/project/2026/2027_cvpr/nfs_cache/release_{occ}_vobench_v2")
NF, S, D = 8, 256, 1280
KEYS = ("block_id", "source_block", "variant", "objset", "violation", "occlusion", "k", "motion", "background", "obj_a", "obj_b", "samples")


def main(occ="late", what="zp", frames=FRAMES, pred_ckpt=None, out_dir=None):
    OUT = Path(out_dir) if out_dir else OUT_OF(occ)
    from evals.world_model_analysis.data import WMADataset
    from evals.world_model_analysis.attn_probe import _ClipDS, _ac_dtype, _autocast
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    assert frames != FRAMES or Path(FRAMES, "_DONE").exists(), "프레임 사본 먼저 (stage_vobench_v2_late.py)"
    cfg = yaml.safe_load(open(CFG))
    cfg["data"]["root"] = str(REPO / "data_csv/vobench_v2"); cfg["data"]["frames_root"] = frames
    cfg["model"]["dtype"] = "bfloat16"; cfg["model"]["autocast"] = "none"; cfg["features"] = {"decode_workers": 6}
    if pred_ckpt:                                                       # 학습한 predictor (z_training run 의 latest.pt) — 2026-10-31
        cfg["model"]["predictor_checkpoint"] = pred_ckpt
    ds = WMADataset(cfg)
    idx = [i for i, r in enumerate(ds.records) if r.plausible == "1" and r.raw["occlusion"] == occ]
    meta = [dict(video_id=ds.records[i].video_id, **{k: ds.records[i].raw[k] for k in KEYS}) for i in idx]
    print(f"{occ} 가능 {len(idx)} clip · {what}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "meta.json").exists():                                    # 이미 뽑은 배열과 clip 순서가 같아야 한다
        old = [m["video_id"] for m in json.load(open(OUT / "meta.json"))["meta"]]
        assert old == [m["video_id"] for m in meta], "기존 meta 와 clip 순서가 다르다"
    do_zp, do_h, keep_z = "p" in what, "h" in what, "z" in what          # what = p 면 z 는 계산만 하고 안 저장 (학습한 predictor 판)
    Z = np.lib.format.open_memmap(OUT / "z.npy.tmp", "w+", np.float16, (len(idx), NF, S, D)) if keep_z else None
    P = np.lib.format.open_memmap(OUT / "p.npy.tmp", "w+", np.float16, (len(idx), NF, S, D)) if do_zp else None
    H = np.lib.format.open_memmap(OUT / "h_fut.npy.tmp", "w+", np.float16, (len(idx), NF, S, D)) if do_h else None
    dev = torch.device("cuda:0"); torch.set_num_threads(4)
    model = build_from_config(cfg["model"], dev)
    bs = 4
    ci, ti = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=S, batch_size=bs, device=dev)
    loader = torch.utils.data.DataLoader(_ClipDS(cfg, idx), batch_size=bs, shuffle=False, num_workers=6, pin_memory=True)
    ac = _ac_dtype(cfg["model"].get("autocast", "none"))
    w, t0 = 0, time.time()
    with torch.inference_mode():
        for x in loader:
            b = x.size(0)
            x = x.to(dev, model.dtype, non_blocking=True)
            with _autocast(ac):
                if do_zp:
                    z = model.context_encoder(x, masks=[ci[:b]]); z = z[-1] if isinstance(z, list) else z
                    p = model.predictor(z, ci[:b], ti[:b], mask_index=0)
                if do_h:
                    h = model.target_encoder(x); h = h[-1] if isinstance(h, list) else h
                    h = nn.functional.layer_norm(h, (h.size(-1),))
            if keep_z:
                Z[w:w + b] = nn.functional.layer_norm(z.float(), (z.size(-1),)).cpu().numpy().reshape(b, NF, S, -1).astype(np.float16)
            if do_zp:
                P[w:w + b] = p.float().cpu().numpy().reshape(b, NF, S, -1).astype(np.float16)
            if do_h:
                H[w:w + b] = h.float().cpu().numpy().reshape(b, 2 * NF, S, -1)[:, NF:].astype(np.float16)
            w += b
            if w % 80 < bs:
                print(f"{w}/{len(idx)} ({time.time() - t0:.0f}s)", flush=True)
    if do_h:
        H.flush(); del H; (OUT / "h_fut.npy.tmp").rename(OUT / "h_fut.npy")
    if not do_zp:
        print("done h", time.time() - t0, flush=True); return
    P.flush(); del P; (OUT / "p.npy.tmp").rename(OUT / "p.npy")
    if keep_z:
        Z.flush(); del Z; (OUT / "z.npy.tmp").rename(OUT / "z.npy")
    pooled = {}
    for name in (("z", "p") if keep_z else ("p",)):
        A = np.load(OUT / f"{name}.npy", mmap_mode="r"); m = np.zeros((len(A), NF, D), np.float32)
        for i in range(0, len(A), 32):
            m[i:i + 32] = np.asarray(A[i:i + 32], np.float32).mean(2)
        pooled[name] = m
    np.savez(OUT / "pooled.npz", **pooled)
    json.dump(dict(meta=meta, cfg=cfg, frames_root=frames), open(OUT / "meta.json", "w"))
    (OUT / "_DONE").write_text(f"{occ} clips {len(idx)} · {time.strftime('%F %T')}\n")
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--occ", default="late", choices=["late", "visible", "early", "mid"])
    ap.add_argument("--what", default="zp", choices=["zp", "h", "zph", "p"])
    ap.add_argument("--frames", default=FRAMES, help="vll5 에서는 원본 /data2/local_datasets/world/Benchmark/VOBench_v2")
    ap.add_argument("--pred-ckpt", default=None, help="학습한 predictor 체크포인트 (z_training/runs/<run>/latest.pt). 주면 --out 필수")
    ap.add_argument("--out", default=None, help="출력 폴더 (기본 nfs_cache/release_<occ>_vobench_v2)")
    a = ap.parse_args()
    if a.pred_ckpt and not a.out:
        raise SystemExit("--pred-ckpt 를 주면 --out 도 줘야 한다 (릴리즈 캐시를 덮어쓰지 않게)")
    main(a.occ, a.what, a.frames, a.pred_ckpt, a.out)
