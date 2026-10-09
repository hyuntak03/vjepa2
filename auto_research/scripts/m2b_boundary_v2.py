#!/usr/bin/env python3
"""M2b — 경계 패칭 2 판 (적대 검증 2026-09-25 반영, v11).

1 판에서 살아남은 것: 움직이는 물체의 미래 "있음" 은 predictor 입력의 경계 1–2 튜블릿 토큰이 정한다 (회복 0.94).
검증이 요구한 것:
  (a) 천장 — late · early · visible 의 **진짜 미래 LN(h)** 에 같은 E 템플릿을 건 물체 신호 (L 은 처음 k 장 가려짐)
  (b) 경계 토큰의 역사 섞임 — 문맥 encoder 는 문맥 16 장 전체를 보고 경계 토큰을 만든다. **경계 튜블릿 토큰만 마스크로 남겨 인코딩한 토큰** (역사 프레임 안 봄, RoPE 위치 보존) 으로 교체해도 넘어가나
  (c) E 짝을 **같은 block** 에서 (matched pair 유지), 가능하면 같은 sym_k
  (d) 가림 없는 쌍둥이 V (visible) 기준선: V, V+Lb, L+Vb
  (e) 화면 밖 진실 칸은 NaN (1 판은 가장자리로 잘라 넣었다)
팔 (predictor 입력; 위치 인덱스 원래 자리):
  L, E, V, L+Eb, E+Lb, Eb_only, Lb_only        (1 판과 같은 정의)
  Lbt   = [L 역사 ; L 경계를 경계 2m 프레임만으로 인코딩]     (절단 인코딩 대조)
  L+Ebt = [L 역사 ; E 경계를 경계 2m 프레임만으로 인코딩]     (역사 정보 없는 경계 관측)
  V+Lb, L+Vb
천장 (predictor 없음): hL, hE, hV = 각자 진짜 미래 h 에 E 템플릿.
저장: loc (n, 팔+천장, 8, 4)  y, x, maxcos, median;  scores (n, 팔, 2, 8) = matched pos/imp L1 (L 계열은 L 의 h, E 계열은 E, V 계열은 V).

  python m2b_boundary_v2.py --gpus 8   /   --gpus 1 --limit 8 --out /tmp/m2b_smoke
"""
from __future__ import annotations
import argparse, collections, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arlib  # noqa: E402
from h7_extract_v11 import cfg_v11  # noqa: E402
from h4_single_query_v3 import idx_range, locate  # noqa: E402

S_TOK, D, TC, K, G, CELL = 256, 1280, 8, 8, 16, 18.0
META = "/local_datasets/world/world_analysis/IntPhysGen_v11/metadata.csv"
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v11_m2b"
ARMS = ["L", "E", "V", "L+Eb", "E+Lb", "Eb_only", "Lb_only", "Lbt", "L+Ebt", "V+Lb", "L+Vb"]
CEIL = ["hL", "hE", "hV"]


def build_pairs():
    M = {r["name"]: r for r in csv.DictReader(open(META))}
    key = lambda r: (r["motion"], r["env"], r["shape_pre"], r["color_pre"], r["shape_post"], r["color_post"], r["travel_direction"], r["v0_cm_s"],
                     r["object_px_x_by_sample"], r["object_px_y_by_sample"], r["cam_x"], r["cam_yaw"], r["violation_type"], r["role"])
    g = collections.defaultdict(list)
    for r in M.values(): g[key(r)].append(r)
    idx = list(csv.DictReader(open(arlib.ROOT / "data_csv/intphysgen_v11_full/index_probe.csv")))
    blocks = collections.defaultdict(dict); vid_bp = {}
    for r in idx:
        blocks[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r["video_id"]
        vid_bp[r["video_id"]] = (r["block_id"], r["pair_id"])
    def twin(lp, timing):
        c = [t for t in g[key(lp)] if t["occ_timing"] == timing and t["name"] in vid_bp]
        same = [t for t in c if t["sym_k"] == lp["sym_k"]]
        for t in (same or c):
            b = blocks.get(vid_bp[t["name"]], {})
            if len(b) == 2 and b["pos"] == t["name"]: return b, t
        return None, None
    out = []
    for (bid, pid), d in blocks.items():
        if len(d) != 2: continue
        lp = M[d["pos"]]
        if lp["occ_timing"] != "late": continue
        Eb, et = twin(lp, "early"); Vb, _ = twin(lp, "")
        if Eb is None or Vb is None: continue
        k = int(lp["sym_k"]); m = 1 if k <= 2 else 2
        out.append(dict(L_pos=d["pos"], L_imp=d["imp"], E_pos=Eb["pos"], E_imp=Eb["imp"], V_pos=Vb["pos"], V_imp=Vb["imp"],
                        m=m, k=k, same_k=et["sym_k"] == lp["sym_k"], motion=lp["motion"], block=bid, pair_id=pid,
                        violation=M[d["imp"]]["violation_type"], px=lp["object_px_x_by_sample"], py=lp["object_px_y_by_sample"]))
    return out


def cells(px, py):
    x = np.array([float(v) for v in px.split()]); y = np.array([float(v) for v in py.split()]); c = []
    for t in range(K):
        s0 = 16 + 2 * t; cx, cy = np.nanmean(x[s0:s0 + 2]), np.nanmean(y[s0:s0 + 2])
        c.append((-1, -1) if (np.isnan(cx) or cx < 0 or cx >= 288 or cy < 0 or cy >= 288) else (int(cx // CELL), int(cy // CELL)))
    return c


def _worker(rank, world, a, pairs):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    cfg = cfg_v11(); ds = arlib.WMADataset(cfg); bundle = arlib.build_bundle(cfg, dev, window=32); pred, enc, tgt = bundle.predictor, bundle.context_encoder, bundle.target_encoder
    vid2i = {r.video_id: i for i, r in enumerate(ds.records)}; ac = arlib.autocast_ctx(cfg)
    S = np.lib.format.open_memmap(Path(a.out) / "scores.npy", mode="r+"); LOC = np.lib.format.open_memmap(Path(a.out) / "loc.npy", mode="r+")
    mine = list(range(rank, len(pairs), world)); t0 = time.time()
    ci = idx_range(0, TC, 1, dev); ti = idx_range(TC, K, 1, dev)
    for n_done, pi in enumerate(mine):
        P = pairs[pi]; names = ("L_pos", "L_imp", "E_pos", "E_imp", "V_pos", "V_imp")
        clips = torch.stack([ds.clip(vid2i[P[k]]) for k in names]).to(dev, bundle.dtype)
        m = P["m"]; nb = m * S_TOK
        with torch.inference_mode(), ac():
            h = F.layer_norm(tgt(clips), (D,)).float().reshape(6, 16, S_TOK, D)[:, TC:]
            zL = enc(clips[0:1], masks=[ci]); zE = enc(clips[2:3], masks=[ci]); zV = enc(clips[4:5], masks=[ci])
            # 경계 튜블릿 토큰만 남기고 인코딩 (encoder 가 역사 프레임을 안 본다; RoPE 위치는 원래 자리 — 교차 비평 C12)
            tr = lambda c: enc(c, masks=[idx_range(TC - m, m, 1, dev)])
            zLt, zEt = tr(clips[0:1]), tr(clips[2:3])
            hist = lambda z: z[:, :-nb]
            inp = {"L": (zL, ci), "E": (zE, ci), "V": (zV, ci),
                   "L+Eb": (torch.cat([hist(zL), zE[:, -nb:]], 1), ci), "E+Lb": (torch.cat([hist(zE), zL[:, -nb:]], 1), ci),
                   "Eb_only": (zE[:, -2 * S_TOK:], idx_range(TC - 2, 2, 1, dev)), "Lb_only": (zL[:, -2 * S_TOK:], idx_range(TC - 2, 2, 1, dev)),
                   "Lbt": (torch.cat([hist(zL), zLt], 1), ci), "L+Ebt": (torch.cat([hist(zL), zEt], 1), ci),
                   "V+Lb": (torch.cat([hist(zV), zL[:, -nb:]], 1), ci), "L+Vb": (torch.cat([hist(zL), zV[:, -nb:]], 1), ci)}
            p = {k_: pred(z, c, ti, mask_index=0).float().reshape(K, S_TOK, D) for k_, (z, c) in inp.items()}
        cl = cells(P["px"], P["py"])
        for ai, arm in enumerate(ARMS):
            fam = {"L": 0, "Lb_only": 0, "Lbt": 0, "L+Eb": 0, "L+Ebt": 0, "L+Vb": 0, "E": 2, "E+Lb": 2, "Eb_only": 2, "V": 4, "V+Lb": 4}[arm]
            hp, hi = h[fam], h[fam + 1]
            S[pi, ai, 0] = (p[arm] - hp).abs().mean((-1, -2)).cpu().numpy(); S[pi, ai, 1] = (p[arm] - hi).abs().mean((-1, -2)).cpu().numpy()
        for t in range(K):
            cx, cy = cl[t]
            if cx < 0: continue
            tpl = h[2, t, cy * G + cx][None]
            for ai, arm in enumerate(ARMS): LOC[pi, ai, t] = locate(p[arm][t][None], tpl).cpu().numpy()[0]
            for ci_, hh in enumerate((h[0], h[2], h[4])): LOC[pi, len(ARMS) + ci_, t] = locate(hh[t][None], tpl).cpu().numpy()[0]
        if rank == 0 and n_done % 100 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분 남은 {el/(n_done+1)*(len(mine)-n_done-1)/60:.1f} 분", flush=True)
    for x in (S, LOC): x.flush()
    print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count()); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    pairs = build_pairs()
    if a.limit: pairs = pairs[:: max(1, len(pairs) // a.limit)][: a.limit]
    n = len(pairs); print("pairs", n, collections.Counter((p["motion"], p["violation"]) for p in pairs), "same_k", sum(p["same_k"] for p in pairs), flush=True)
    for nm, sh in (("scores", (n, len(ARMS), 2, K)), ("loc", (n, len(ARMS) + len(CEIL), K, 4))):
        x = np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="w+", dtype=np.float32, shape=sh); x[:] = np.nan; del x
    json.dump(dict(n=n, arms=ARMS, ceil=CEIL, pairs=pairs), open(Path(a.out) / "meta.json", "w"))
    mp.spawn(_worker, args=(a.gpus, a, pairs), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
