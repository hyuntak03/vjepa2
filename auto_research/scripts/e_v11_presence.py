#!/usr/bin/env python3
"""v11 late 가림 · vanish 쌍에서 "presence vs placement" — 채점 margin 이 어느 토큰에서 나오나 (2026-09-29, AR_PREDICTOR §2-6 의 모순 검정).

    SCENE_V11_STAGE=<dir> python e_v11_presence.py --out <dir> [--gpus N] [--limit K]

쌍 = (pos_a, imp_ab) · (pos_b, imp_ba): 문맥 16 장이 픽셀 단위로 같다 → predictor 출력 p 가 쌍 안에서 같다.
predictor 마다 저장 (쌍, 8 슬롯, 256 칸):
  M   = mean_D |p − h_imp| − mean_D |p − h_pos|      토큰별 margin (+ = 가능 쪽을 고름 = 정답 방향). 채점 = mean(M) > 0
  Mc  = 같은 것을 복사 (문맥 마지막 튜블릿) 로     복사 기준선의 토큰별 margin
  Dh  = mean_D |h_pos − h_imp|                       두 미래가 다른 자리 (물체가 다시 나타나는 칸 ↔ 빈 칸) — encoder 가 정한 "자리"
표준 팔 (R · F · B): z = online(문맥 16), h = LN(target(32 장))[미래], 복사 = LN(z)[튜블릿 7] 반복.
AR 팔 (AR ep40 · ctx_ar ep16): 타깃 = 블록별 LN(target(2 장)), AR 문맥 = 블록별, ctx_ar 문맥 = z; 복사 = h_blk[7] 반복.
읽기 (analyze): 정답률; margin 질량이 Dh 상위 칸에 몰리는 비율 (placement share) vs 퍼짐; 슬롯별. AR 이 맞히는 쌍에서 질량이 Dh 상위에 있으면 placement, 아니면 presence-without-placement.
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
import e_scene_reach as ES  # noqa: E402
import e_scene_ar as EA  # noqa: E402
from e_ssv2_extract import MEAN, STD  # noqa: E402
from app.vjepa_frozen.ar import encode_blocks  # noqa: E402

S_TOK, D, NF, TC = 256, 1280, 32, 8
Z = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training")
CK = {"F": Z / "ariel/full_attn_future_pred_1e_5/latest.pt", "B": Z / "ariel/block_causal_future_only_1e_5/latest.pt",
      "AR": Z / "ariel_eval/ckpt/ar_ep40.pt", "CX": Z / "ariel_eval/ckpt/ctx_ar_ep16.pt"}
PREDS = ["R", "F", "B", "AR", "CX"]
SPACE = {"R": "std", "F": "std", "B": "std", "AR": "blk", "CX": "blk"}


def load_clip(it):
    x = torch.from_numpy(np.load(it["npy"])).permute(3, 0, 1, 2).float() / 255.0
    return (x - MEAN) / STD


def pairs_from(items):
    by = {}
    for it in items: by.setdefault(it["block_id"], {})[it["variant"]] = it
    out = []
    for b, d in sorted(by.items()):
        if all(v in d for v in ("pos_a", "imp_ab", "pos_b", "imp_ba")):
            out.append(dict(block=b, pos=d["pos_a"], imp=d["imp_ab"], kind="obj", condition=d["pos_a"]["condition"], sym_k=d["pos_a"]["sym_k"]))
            out.append(dict(block=b, pos=d["pos_b"], imp=d["imp_ba"], kind="empty", condition=d["pos_a"]["condition"], sym_k=d["pos_a"]["sym_k"]))
    return out


def _worker(rank, world, a, pairs):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    cfg = arlib.v3_cfg(16, 16); os.environ.pop("ARLIB_PRED_CKPT", None)
    bundle = arlib.build_bundle(cfg, dev, window=NF); ac = arlib.autocast_ctx(cfg); bf = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    preds = {"R": bundle.predictor}; kind = {}
    for k in ("F", "B", "AR", "CX"):
        preds[k], kind[k] = ES.load_predictor(cfg["model"], CK[k], dev, bundle.dtype, NF)
    ci = idx_range(0, TC, 1, dev); ti = idx_range(TC, TC, 1, dev)
    O = {nm: np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="r+") for nm in ("M", "Mc", "Dh")}
    mine = list(range(rank, len(pairs), world)); t0 = time.time()
    for n_done, row in enumerate(mine):
        pr = pairs[row]; xp = load_clip(pr["pos"])[None].to(dev, bundle.dtype); xi = load_clip(pr["imp"])[None].to(dev, bundle.dtype)
        assert torch.equal(xp[:, :, :16], xi[:, :, :16]), "matched pair 문맥이 다르다"
        with torch.inference_mode():
            with ac():
                hp = F.layer_norm(bundle.target_encoder(xp), (D,)).float().reshape(16, S_TOK, D)[8:]
                hi = F.layer_norm(bundle.target_encoder(xi), (D,)).float().reshape(16, S_TOK, D)[8:]
                z = bundle.context_encoder(xp, masks=[ci]); zc = F.layer_norm(z.float(), (D,)).reshape(8, S_TOK, D)[7]
                p_std = {k: preds[k](z, ci, ti, mask_index=0).float().reshape(8, S_TOK, D) for k in ("R", "F", "B")}
            with bf():
                hbp = encode_blocks(bundle.target_encoder, xp, 2); hbi = encode_blocks(bundle.target_encoder, xi, 2)
                p_ar, _ = EA.ar_predict(preds["AR"], hbp); p_cx, _ = EA.ctx_ar_predict(preds["CX"], z, hbp)
            hbp = hbp.float().reshape(16, S_TOK, D); hbi = hbi.float().reshape(16, S_TOK, D)
            tgt = {"std": (hp, hi, zc), "blk": (hbp[8:], hbi[8:], hbp[7])}
            for pi, k in enumerate(PREDS):
                p = {"R": p_std["R"], "F": p_std["F"], "B": p_std["B"], "AR": p_ar, "CX": p_cx}[k]; Hp, Hi, Tc = tgt[SPACE[k]]
                O["M"][row, pi] = ((p - Hi).abs().mean(-1) - (p - Hp).abs().mean(-1)).half().cpu().numpy()
                O["Mc"][row, pi] = ((Tc[None] - Hi).abs().mean(-1) - (Tc[None] - Hp).abs().mean(-1)).half().cpu().numpy()
                O["Dh"][row, pi] = (Hp - Hi).abs().mean(-1).half().cpu().numpy()
        if rank == 0 and n_done % 20 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분", flush=True)
    for v_ in O.values(): v_.flush()
    print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count()); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    items = json.load(open(Path(os.environ["SCENE_V11_STAGE"]) / "items.json")); pairs = pairs_from(items)
    if a.limit: pairs = pairs[: a.limit]
    n = len(pairs)
    for nm in ("M", "Mc", "Dh"):
        m = np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="w+", dtype=np.float16, shape=(n, len(PREDS), 8, S_TOK)); m[:] = np.nan; del m
    json.dump(dict(n=n, preds=PREDS, space=SPACE, ckpt={k: str(v) for k, v in CK.items()}, pairs=[{k: (v if not isinstance(v, dict) else v["video_id"]) for k, v in p.items()} for p in pairs]),
              open(Path(a.out) / "meta.json", "w"))
    print("pairs", n, flush=True); torch.set_num_threads(4)
    mp.spawn(_worker, args=(a.gpus, a, pairs), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
