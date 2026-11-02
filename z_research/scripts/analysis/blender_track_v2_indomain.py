#!/usr/bin/env python3
"""BlenderTrack v2 test 도메인 안 점검 — predictor 여럿의 미래 예측 오차를 clip 마다 잰다 (2026-11-01).

질문: v2 로 이어 학습한 predictor 가 자기 도메인 (학습에 안 쓴 v2 test) 에서는 배웠나, 어느 칸 (타이밍 · 가림 형태 · 운동 · 카메라) 에서 배웠나.
창: 학습과 같다 — 20 fps 64 장을 간격 2 로 32 샘플 (0, 2, …, 62), 문맥 16 샘플 (블록 8) + 미래 16 샘플 (블록 8).
오차 (블록 b, 토큰 n): e = mean_d |p − LN(h)|  (학습 손실과 같은 거리, h = target encoder, p = predictor(문맥 encoder 토큰))
  · 전체 = 미래 8 블록 × 256 토큰 평균 (표준 채점과 같은 기준)
  · target 자리 = amodal 점유 비율 (블록 2 장 평균, 16×16) 로 가중한 평균 — 가려져도 물체가 있는 곳
  · 복사 기준선 = 마지막 문맥 블록의 LN(h) 를 미래 블록마다 그대로 (예측 없이)
산출: <out>/shard<k>.npz — clip · e_all (M, N, 8) · e_obj (M, N, 8) · c_all (N, 8) · c_obj (N, 8) · hid (N, 8) · occ_sum (N, 8) · models
  P=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python   (vll7)
  $P z_research/scripts/analysis/blender_track_v2_indomain.py --shard 0 --nshards 6 --out <dir>
  집계: z_research/scripts/analysis/blender_track_v2_indomain_table.py --out <dir>
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from app.vjepa_frozen import utils as U  # noqa: E402
from app.vjepa_frozen.blender_track_data import BlenderTrackDataset  # noqa: E402
from app.vjepa_frozen.data import ClipTransform  # noqa: E402

MODELS = {   # 이름 → predictor 체크포인트 (encoder 는 모두 릴리즈 ViT-H 고정)
    "release": "z_training/runs/release_vith/latest.pt",
    "v1": "z_training/runs/btrack_v1ns_l1/latest.pt",
    "v2": "z_training/runs/btrack_v2_l1/latest.pt",
    "v2late": "z_training/runs/btrack_v2late_l1/latest.pt",
}


class Clips(torch.utils.data.Dataset):
    def __init__(self, ds, ids, idx, tf):
        self.ds, self.ids, self.idx, self.tf = ds, ids, idx, tf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        c = self.ids[k]
        x = self.tf(self.ds.read(c, self.idx))
        lab = self.ds.labels(c, self.idx, flip=False)
        return x, lab["occ"], lab["hid"], c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=f"{REPO}/data_csv/blender_track_v2/test_meta_ok.npz")
    ap.add_argument("--clips", default="/data/hyuntak/project/2026/2027_cvpr/nfs_cache/blender_track/v2/clips")
    ap.add_argument("--config", default=f"{REPO}/z_training/runs/btrack_v2_l1/config.yaml", help="encoder · predictor 구조 (학습 config)")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda:0")
    MD = yaml.safe_load(open(a.config))["model"]
    n_frames, tub, S = 32, 2, 256
    state = U.load_base_checkpoint(MD["checkpoint"])
    ctx_enc, tgt_enc = U.build_frozen_encoders(MD, state, dev, torch.bfloat16, n_frames)
    preds = {}
    for m in a.models:
        p = U.build_predictor(MD, ctx_enc.embed_dim, None, dev, n_frames)
        ck = torch.load(os.path.join(REPO, MODELS[m]), map_location="cpu", weights_only=False)
        sd = {k.replace("module.", "", 1): v for k, v in ck["predictor"].items()}
        p.load_state_dict(sd, strict=True)
        preds[m] = p.eval()
    del state
    ds = BlenderTrackDataset({"meta": a.meta, "root": a.clips}, n_frames, None)
    ids = list(range(len(ds.clips)))[a.shard::a.nshards]
    if a.limit:
        ids = ids[:a.limit]
    idx = np.arange(n_frames) * 2                                    # 0, 2, …, 62 (설계의 샘플 = 칸 타이밍의 기준)
    dl = torch.utils.data.DataLoader(Clips(ds, ids, idx, ClipTransform(256)), batch_size=a.batch, num_workers=8, shuffle=False)
    nb, nc = n_frames // tub, 16 // tub                             # 블록 16, 문맥 블록 8
    n_ctx = nc * S
    ci = torch.arange(n_ctx, device=dev).unsqueeze(0)
    ti = torch.arange(n_ctx, nb * S, device=dev).unsqueeze(0)
    M = len(a.models)
    E_all, E_obj, C_all, C_obj, HID, OCC, CL = [], [], [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for it, (x, occ, hid, c) in enumerate(dl):
            x = x.to(dev, non_blocking=True)
            n = x.size(0)
            occ_f = occ[:, nc:].to(dev).reshape(n, -1)                  # (n, 8·256) 미래 블록 점유
            with torch.autocast("cuda", dtype=torch.bfloat16):
                xe = x.to(torch.bfloat16)
                h = tgt_enc(xe)
                h = torch.nn.functional.layer_norm(h.float(), (h.size(-1),))
                z = ctx_enc(xe, masks=[ci.expand(n, -1)])
                hf = h[:, n_ctx:]                                     # (n, 8·256, D)
                cp = h[:, (nc - 1) * S:nc * S].repeat(1, nb - nc, 1)   # 마지막 문맥 블록 복사
                ea, eo = [], []
                for m in a.models:
                    p = preds[m](z, ci.expand(n, -1), ti.expand(n, -1), mask_index=int(MD.get("mask_index", 0)))
                    e = (p.float() - hf).abs().mean(-1)                # (n, 8·256)
                    ea.append(e.reshape(n, nb - nc, S).mean(-1))
                    eo.append((e * occ_f).reshape(n, nb - nc, S).sum(-1) / occ_f.reshape(n, nb - nc, S).sum(-1).clamp_min(1e-6))
                ec = (cp - hf).abs().mean(-1)
            E_all.append(torch.stack(ea, 0).cpu()); E_obj.append(torch.stack(eo, 0).cpu())
            C_all.append(ec.reshape(n, nb - nc, S).mean(-1).cpu())
            C_obj.append(((ec * occ_f).reshape(n, nb - nc, S).sum(-1) / occ_f.reshape(n, nb - nc, S).sum(-1).clamp_min(1e-6)).cpu())
            HID.append(hid[:, nc:]); OCC.append(occ[:, nc:].reshape(n, nb - nc, -1).sum(-1))
            CL += [ds.clips[int(k)] for k in c]
            if it % 50 == 0:
                print(f"shard {a.shard}: {len(CL)}/{len(ids)} | {time.time() - t0:.0f}s", flush=True)
    np.savez(f"{a.out}/shard{a.shard}.npz", clip=np.array(CL), models=np.array(a.models),
             e_all=torch.cat(E_all, 1).numpy(), e_obj=torch.cat(E_obj, 1).numpy(), c_all=torch.cat(C_all).numpy(),
             c_obj=torch.cat(C_obj).numpy(), hid=torch.cat(HID).numpy(), occ_sum=torch.cat(OCC).numpy())
    print(f"shard {a.shard} 끝 {len(CL)} clip | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
