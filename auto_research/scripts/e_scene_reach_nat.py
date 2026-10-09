#!/usr/bin/env python3
"""I-scene 자연 영상판 (2026-09-25) — e_scene_reach.py (인공 팬) 의 개입을 **실제 움직임** 에 건다.

출처 u_i(s): 미래 튜블릿 i 의 첫 프레임 (16+2i) 에서 칸 s 중심을 **역방향 optical flow** (Farneback, 프레임마다 t → t−1) 로 프레임 15 까지 추적한 자리.
  --source ssv2 : SSv2 좌/우 677 clip (e_ssv2_extract.load_clip, 균등 32 장)
  --source ek100: EPIC-KITCHENS resized 256² 60 fps, 영상마다 임의 구간 32 장 stride 4 (≈ 15 fps; 1 인칭, 카메라 운동)
  --source v3   : RollOut_v3 raw stride 2 (P2 s2_C16 과 같은 프레임). 출처는 GT: 물체 3×3 칸은 물체 변위만큼, 나머지는 제자리 (배경 정지)
판독 · 개입 · 저장은 e_scene_reach 와 같고, 다음을 더 저장한다: src.npy (n, 8, 256) int16 · disp.npy (n, 8, 256) float16 (Chebyshev 칸)
  · srccv.npy (n, 8, 256) int16 = 문맥 끝 flow (프레임 12→15) 로 등속 외삽한 **인과** 출처 추정 (bias_cv 팔).
팔: release base(+기록) · rule_prefix · bias4 (flow 출처) · bias4_cv (인과 추정 출처) · anticopy4 · temp2 / Ariel base(+기록) · bias4 · bias4_cv · anticopy4 / pv1 base
  (v3 는 GT 출처라 bias4 = 정답, bias4_cv = 문맥 GT 속도 외삽)

  python e_scene_reach_nat.py --source ssv2 --gpus 8 --out <dir>
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arlib  # noqa: E402
from h4_single_query_v3 import idx_range  # noqa: E402
from e_ssv2_extract import load_clip, SSV2, MEAN, STD  # noqa: E402
import scene_hooks as SH  # noqa: E402
import e_scene_reach as ES  # noqa: E402
from e_flowtrack import to_gray, V3_FR, V3_SCALE, FREE  # noqa: E402

S_TOK, D, NF, TC, G, PX = 256, 1280, 32, 8, 16, 16
EK = Path("/data2/local_datasets/EPIC-KITCHENS_resized")
ARMS = [("R", "base", dict(rec=True)), ("R", "rule_prefix", dict(rule="prefix")), ("R", "bias4", dict(bias=4.0)), ("R", "bias4_cv", dict(bias=4.0, cv=True)),
        ("R", "anticopy4", dict(anticopy=4.0)), ("R", "temp2", dict(temp=2.0)),
        ("R", "qshift", dict(qshift="src")), ("R", "qshift_cv", dict(qshift="cv")),
        ("A", "base", dict(rec=True)), ("A", "bias4", dict(bias=4.0)), ("A", "qshift", dict(qshift="src")), ("A", "qshift_cv", dict(qshift="cv")), ("A", "bias4_cv", dict(bias=4.0, cv=True)), ("A", "anticopy4", dict(anticopy=4.0)),
        ("P", "base", dict(rec=True))]
REC_PREDS = ["R", "A", "P"]
EXTRA = [kv.split("=", 1) for kv in os.environ.get("SCENE_EXTRA", "").split(",") if "=" in kv]
for _k, _p in EXTRA:
    ARMS += [(_k, "base", dict(rec=True)), (_k, "bias4", dict(bias=4.0)), (_k, "anticopy4", dict(anticopy=4.0))]
    REC_PREDS.append(_k)
ARM_NAMES = [f"{p}:{a}" for p, a, _ in ARMS]


def items_for(source, limit):
    if source == "ek100" and os.environ.get("SCENE_EK_STAGE"):                  # 2026-09-29: 다른 노드용 스테이징 (data/stage_ek100.py)
        out = json.load(open(Path(os.environ["SCENE_EK_STAGE"]) / "items.json"))
        if limit:
            k = np.linspace(0, len(out) - 1, limit).round().astype(int); out = [out[i] for i in k]
        return out
    if source == "ssv2":
        import decord
        out = []
        for c in ("left", "right"):
            for f in sorted((SSV2 / c).glob("*.mp4")):
                if len(decord.VideoReader(str(f), num_threads=1)) >= NF: out.append(dict(file=str(f)))
    elif source == "ek100":
        import decord
        vids = sorted(EK.glob("P*/videos/*.MP4")); rng = np.random.default_rng(0); out = []
        per = max(1, 800 // len(vids) + 1)
        for f in vids:
            try:
                L = len(decord.VideoReader(str(f), num_threads=1))
            except Exception:
                continue
            for _ in range(per):
                s0 = int(rng.integers(0, max(1, L - 4 * NF))); out.append(dict(file=str(f), start=s0, stride=4))
        out = out[:: max(1, len(out) // 800)][:800]
    else:
        R = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))).records
        out = [dict(rec=i) for i, r in enumerate(R) if r.plausible == "1" and r.raw["scenario"] in FREE][::4]
    if limit:
        k = np.linspace(0, len(out) - 1, limit).round().astype(int); out = [out[i] for i in k]
    return out


def load_item(source, it, ds64=None):
    if source == "ssv2":
        return load_clip(it["file"]), None
    if source == "ek100":
        if it.get("npy"):                                                        # 스테이징된 uint8 (32, 256, 256, 3)
            x = torch.from_numpy(np.load(it["npy"])).permute(3, 0, 1, 2).float() / 255.0
            return (x - MEAN) / STD, None
        import decord
        vr = decord.VideoReader(it["file"], num_threads=2); idx = [it["start"] + it["stride"] * k for k in range(NF)]
        x = torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(3, 0, 1, 2).float() / 255.0
        if x.shape[-1] != 256 or x.shape[-2] != 256:
            x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).transpose(0, 1)
        return (x - MEAN) / STD, None
    from h23_extract_v3 import arr
    i = it["rec"]; x = ds64.clip(i)[:, V3_FR].float(); raw = ds64.records[i].raw
    gx, gy = arr(raw["px_x_by_sample"]) * V3_SCALE, arr(raw["px_y_by_sample"]) * V3_SCALE
    return x, (gx[V3_FR], gy[V3_FR])                                          # 32 장 위치 (256 격자 px)


def _sample(fl, pts):
    """fl (H, W, 2) np, pts (P, 2) x,y → (P, 2) 쌍선형."""
    t = torch.from_numpy(fl).permute(2, 0, 1)[None].float()
    g = (torch.from_numpy(pts).float() + 0.5) / 128.0 - 1.0                   # align_corners=False 규약 (화소 중심)
    return F.grid_sample(t, g[None, None], align_corners=False, mode="bilinear", padding_mode="border")[0, :, 0].T.numpy()


def src_maps(source, x, gt):
    """→ src (8,256) 출처 flat (−1 무효), disp (8,256) 칸, srccv (8,256) 인과 등속 추정 출처."""
    cx = (np.arange(S_TOK) % G) * PX + PX / 2; cy = (np.arange(S_TOK) // G) * PX + PX / 2; ctr = np.stack([cx, cy], 1)
    src = np.full((8, S_TOK), -1, np.int64); disp = np.zeros((8, S_TOK), np.float32); scv = np.full((8, S_TOK), -1, np.int64)
    to_cell = lambda p: np.where((p[:, 0] >= 0) & (p[:, 0] < 256) & (p[:, 1] >= 0) & (p[:, 1] < 256),
                                 (np.clip(p[:, 1], 0, 255) // PX).astype(int) * G + (np.clip(p[:, 0], 0, 255) // PX).astype(int), -1)
    if source == "v3":
        gx, gy = gt
        ol = np.array([gx[15], gy[15]])
        vcv = (ol - np.array([gx[12], gy[12]])) / 3.0
        for i in range(8):
            f = 16 + 2 * i; og = np.array([gx[f], gy[f]])
            if not np.all(np.isfinite(og)) or not np.all(np.isfinite(ol)):
                src[i] = np.arange(S_TOK); scv[i] = np.arange(S_TOK); continue
            near = np.maximum(np.abs(cx - og[0]), np.abs(cy - og[1])) <= PX * 1.0     # 물체 3×3 칸
            p = ctr.copy(); p[near] = ctr[near] - (og - ol); src[i] = to_cell(p)
            pc = ctr.copy(); pc[near] = ctr[near] - vcv * (f - 15); scv[i] = to_cell(pc)
            disp[i] = np.where(near, np.abs(og - ol).max() / PX, 0.0)
        return src, disp, scv
    import cv2
    g = to_gray(x.cpu()); bf = {t: cv2.calcOpticalFlowFarneback(g[t], g[t - 1], None, 0.5, 3, 15, 3, 5, 1.2, 0) for t in range(16, NF)}
    ff = [cv2.calcOpticalFlowFarneback(g[t], g[t + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0) for t in (12, 13, 14)]
    vfield = np.mean(ff, 0)                                                     # 문맥 끝 속도장 (px/프레임)
    for i in range(8):
        f = 16 + 2 * i; p = ctr.copy()
        for t in range(f, 15, -1):
            p = p + _sample(bf[t], p)
        src[i] = to_cell(p); disp[i] = np.abs(p - ctr).max(1) / PX
        scv[i] = to_cell(ctr - _sample(vfield, ctr) * (f - 15))
    return src, disp, scv


def _worker(rank, world, a, items):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    cfg = arlib.v3_cfg(16, 16); os.environ.pop("ARLIB_PRED_CKPT", None)
    bundle = arlib.build_bundle(cfg, dev, window=NF); ac = arlib.autocast_ctx(cfg)
    preds = {"R": bundle.predictor}
    preds["A"], _ = ES.load_predictor(cfg["model"], ES.ARIEL, dev, bundle.dtype, NF)
    preds["P"], _ = ES.load_predictor(cfg["model"], ES.PV1, dev, bundle.dtype, NF)
    for _k, _p in EXTRA:
        preds[_k], _ = ES.load_predictor(cfg["model"], _p, dev, bundle.dtype, NF)
    hooks = {k: SH.patch(preds[k], SH.SceneHook()) for k in preds}
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))) if a.source == "v3" else None
    ci = idx_range(0, TC, 1, dev); ti = idx_range(TC, TC, 1, dev)
    O = {nm: np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="r+") for nm in ("arm_c", "arm_ct", "arm_cc", "enc", "att", "src", "disp", "srccv") + (("cosmap",) if a.cosmaps else ())}
    mine = list(range(rank, len(items), world)); t0 = time.time()
    for n_done, row in enumerate(mine):
        x0, gt = load_item(a.source, items[row], ds64)
        src, disp, scv = src_maps(a.source, x0, gt)
        O["src"][row] = src; O["disp"][row] = disp; O["srccv"][row] = scv
        x = x0[None].to(dev, bundle.dtype)
        with torch.inference_mode(), ac():
            h = F.layer_norm(bundle.target_encoder(x), (D,)).float().reshape(16, S_TOK, D)
            hc = F.layer_norm(bundle.target_encoder(x[:, :, :16]), (D,)).float().reshape(8, S_TOK, D)
            z = bundle.context_encoder(x, masks=[ci]); hcL = hc[7]
            ec, ect, ecc = ES.readout(h[8:], hcL, src)
            O["enc"][row] = np.stack([ec.astype(np.float32), ect.astype(np.float32), ecc.astype(np.float32)], 0)
            rec = ES.rec_spec(src, dev)
            for ai, (pk, name, kw) in enumerate(ARMS):
                hk = hooks[pk]; hk.reset(); hk.n_ctx_blocks = TC; hk.rule = kw.get("rule"); hk.temp = kw.get("temp")
                if "bias" in kw: hk.bias = ES.bias_matrix(scv if kw.get("cv") else src, kw["bias"], dev)
                if "anticopy" in kw: hk.bias = ES.anticopy_matrix(kw["anticopy"], dev)
                if "qshift" in kw: hk.qshift = ES.qshift_matrix(scv if kw["qshift"] == "cv" else src, dev)
                if kw.get("rec"): hk.rec = rec
                p = preds[pk](z, ci, ti, mask_index=0).float().reshape(8, S_TOK, D)
                c_, ct_, cc_ = ES.readout(p, hcL, src)
                if a.cosmaps and kw.get("rec"):
                    O["cosmap"][row, REC_PREDS.index(pk)] = (F.normalize(p, dim=-1) @ F.normalize(hcL, dim=-1).T).half().cpu().numpy()
                O["arm_c"][row, ai] = c_; O["arm_ct"][row, ai] = ct_; O["arm_cc"][row, ai] = cc_
                if kw.get("rec"):
                    R_ = torch.stack(hk.rec_out).float(); Lh = R_.shape[0]
                    g4 = [R_.mean(0), R_[:Lh // 3].mean(0), R_[Lh // 3:2 * Lh // 3].mean(0), R_[2 * Lh // 3:].mean(0)]
                    O["att"][row, REC_PREDS.index(pk)] = torch.stack(g4).reshape(4, 3, 8, S_TOK).half().cpu().numpy()
                hk.reset()
        if rank == 0 and n_done % 10 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분 남은 {el/(n_done+1)*(len(mine)-n_done-1)/60:.1f} 분", flush=True)
    for v_ in O.values(): v_.flush()
    print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("ssv2", "ek100", "v3"), required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count()); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cosmaps", action="store_true", help="base 팔의 cos 지도 전체 (n, |rec_preds|, 8, 256, 256) fp16 저장")
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    items = items_for(a.source, a.limit); n = len(items)
    shapes = dict(arm_c=((n, len(ARMS), 8, S_TOK), np.int16), arm_ct=((n, len(ARMS), 8, S_TOK), np.float16), arm_cc=((n, len(ARMS), 8, S_TOK), np.float16),
                  enc=((n, 3, 8, S_TOK), np.float32), att=((n, len(REC_PREDS), 4, 3, 8, S_TOK), np.float16),
                  src=((n, 8, S_TOK), np.int16), disp=((n, 8, S_TOK), np.float16), srccv=((n, 8, S_TOK), np.int16))
    if a.cosmaps: shapes["cosmap"] = ((n, len(REC_PREDS), 8, S_TOK, S_TOK), np.float16)
    for nm, (sh, dt) in shapes.items():
        m = np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="w+", dtype=dt, shape=sh); m[:] = (-1 if dt == np.int16 else np.nan); del m
    json.dump(dict(n=n, source=a.source, items=items, arms=ARM_NAMES, rec_preds=REC_PREDS), open(Path(a.out) / "meta.json", "w"))
    print("clips", n, "arms", len(ARMS), flush=True)
    torch.set_num_threads(4)
    mp.spawn(_worker, args=(a.gpus, a, items), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
