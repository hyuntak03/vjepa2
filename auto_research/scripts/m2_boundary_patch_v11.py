#!/usr/bin/env python3
"""M2 — 경계 패칭 (인과): late 가림의 붕괴는 "가려진 경계 튜블릿" 하나로 설명되나.

v11 에는 같은 물체·궤적·배경·카메라에서 **가림 타이밍만 다른 쌍둥이** 가 있다 (metadata 내용 키로 짝지음; 가림막의 자리·폭만 다르다).
late = 문맥 끝 (경계) 튜블릿에서 물체가 가려짐, early = 문맥 앞쪽에서 가려지고 경계에서는 보임. 둘의 역사 (경계 앞 문맥) 에는 모두 물체가 보인다.

가설 (병목 = 위치로 질의하는 최근 문맥 조회, 상태 없음) 이 맞으면: late 에 early 의 **경계 토큰만** 넣으면 early 수준으로 돌아오고,
early 에 late 의 경계만 넣으면 late 수준으로 무너진다. 역사 (앞쪽 문맥) 는 거의 기여하지 않는다 (경계만 남겨도 같다).
대립 (상태는 역사에서 통합되는데 배치만 실패) 이면: 경계만 바꾼 회복은 부분적이고, 역사를 지우면 달라진다.

팔 (predictor 입력 = 문맥 encoder 출력 토큰; 위치 인덱스는 원래 자리 유지):
  L        late 그대로                              E        early 쌍둥이 그대로
  L+Eb     late 에 early 경계 m 튜블릿             E+Lb     early 에 late 경계 m 튜블릿
  Lb_only  late 의 마지막 2 튜블릿만 (역사 삭제)     Eb_only  early 의 마지막 2 튜블릿만
  m = late 의 가려진 문맥 샘플이 든 튜블릿 수 (k ≤ 2 → 1, k ≥ 3 → 2)
측정 (미래 8 튜블릿):
  채점  S = mean|p − LN(h)|: L 계열은 late pos/imp 의 h, E 계열은 early 쌍둥이 pos/imp 의 h (matched pair; 둘은 문맥이 같다)
  물체  템플릿 = early pos 의 h 진실 칸 토큰 (궤적이 같으니 진실 칸도 같다) → 각 팔 p 의 argmax · contrast
  닫힘  D_arm,t = mean|p_arm,t − p_E,t| (전 토큰)  → R = 1 − D_(L+Eb) / D_L
표준 수치 경로 (C16/P16 stride 3, fp32 + autocast fp16, mask_index 0).
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
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/v11_m2"
ARMS = ["L", "L+Eb", "Lb_only", "E", "E+Lb", "Eb_only"]


def build_pairs():
    M = {r["name"]: r for r in csv.DictReader(open(META))}
    key = lambda r: (r["motion"], r["env"], r["shape_pre"], r["color_pre"], r["travel_direction"], r["v0_cm_s"],
                     r["object_px_x_by_sample"][:80], r["object_px_y_by_sample"][:40], r["cam_x"], r["cam_yaw"],
                     r["violation_type"], r["role"])
    g = collections.defaultdict(list)
    for r in M.values():
        g[key(r)].append(r)
    idx = list(csv.DictReader(open(arlib.ROOT / "data_csv/intphysgen_v11_full/index_probe.csv")))
    blocks = collections.defaultdict(dict)
    for r in idx:
        blocks[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r["video_id"]
    out = []
    for (b, pid), d in blocks.items():
        if len(d) != 2: continue
        lp, li = M[d["pos"]], M[d["imp"]]
        if lp["occ_timing"] != "late": continue
        def tw(r):
            c = [t for t in g[key(r)] if t["occ_timing"] == "early"]
            same = [t for t in c if t["sym_k"] == r["sym_k"]]
            return (same or c or [None])[0]
        ep, ei = tw(lp), tw(li)
        if ep is None or ei is None: continue
        k = int(lp["sym_k"]); m = 1 if k <= 2 else 2
        out.append(dict(L_pos=d["pos"], L_imp=d["imp"], E_pos=ep["name"], E_imp=ei["name"], m=m, k=k,
                        motion=lp["motion"], violation=lp["violation_type"], pair_id=pid, role=lp["role"],
                        same_k=ep["sym_k"] == lp["sym_k"],
                        px=lp["object_px_x_by_sample"], py=lp["object_px_y_by_sample"]))
    return out


def cells(px, py):
    x = np.array([float(v) for v in px.split()]); y = np.array([float(v) for v in py.split()])
    c = []
    for t in range(K):
        s0 = 16 + 2 * t
        cx, cy = np.nanmean(x[s0:s0 + 2]), np.nanmean(y[s0:s0 + 2])
        c.append((-1, -1) if np.isnan(cx) else (int(np.clip(cx // CELL, 0, G - 1)), int(np.clip(cy // CELL, 0, G - 1))))
    return c


def _worker(rank, world, a, pairs):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    cfg = cfg_v11(); ds = arlib.WMADataset(cfg); bundle = arlib.build_bundle(cfg, dev, window=32); pred = bundle.predictor
    vid2i = {r.video_id: i for i, r in enumerate(ds.records)}
    mine = list(range(rank, len(pairs), world))
    ac = arlib.autocast_ctx(cfg)
    S = np.lib.format.open_memmap(Path(a.out) / "scores.npy", mode="r+")     # (n, 6 arm, 2 [pos,imp], 8 slot)
    LOC = np.lib.format.open_memmap(Path(a.out) / "loc.npy", mode="r+")      # (n, 6 arm, 8, 4)
    DD = np.lib.format.open_memmap(Path(a.out) / "dist.npy", mode="r+")      # (n, 6 arm, 8) = |p_arm − p_E|
    t0 = time.time()
    for n_done, pi in enumerate(mine):
        P = pairs[pi]
        clips = torch.stack([ds.clip(vid2i[P[k]]) for k in ("L_pos", "L_imp", "E_pos", "E_imp")]).to(dev, bundle.dtype)
        with torch.inference_mode(), ac():
            h = F.layer_norm(bundle.target_encoder(clips), (D,)).float().reshape(4, 16, S_TOK, D)[:, TC:]
            ci = idx_range(0, TC, 1, dev); ti = idx_range(TC, K, 1, dev)
            zL = bundle.context_encoder(clips[0:1], masks=[ci]); zE = bundle.context_encoder(clips[2:3], masks=[ci])
            m = P["m"]; nb = m * S_TOK
            inp = {"L": (zL, ci), "E": (zE, ci),
                   "L+Eb": (torch.cat([zL[:, :-nb], zE[:, -nb:]], 1), ci), "E+Lb": (torch.cat([zE[:, :-nb], zL[:, -nb:]], 1), ci),
                   "Lb_only": (zL[:, -2 * S_TOK:], idx_range(TC - 2, 2, 1, dev)), "Eb_only": (zE[:, -2 * S_TOK:], idx_range(TC - 2, 2, 1, dev))}
            p = {a_: pred(z, c, ti, mask_index=0).float().reshape(K, S_TOK, D) for a_, (z, c) in inp.items()}
        cl = cells(P["px"], P["py"])
        for ai, arm in enumerate(ARMS):
            hp, hi = (h[0], h[1]) if arm.startswith("L") else (h[2], h[3])
            S[pi, ai, 0] = (p[arm] - hp).abs().mean((-1, -2)).cpu().numpy()
            S[pi, ai, 1] = (p[arm] - hi).abs().mean((-1, -2)).cpu().numpy()
            DD[pi, ai] = (p[arm] - p["E"]).abs().mean((-1, -2)).cpu().numpy()
            for t in range(K):
                cx, cy = cl[t]
                if cx < 0: continue
                tpl = h[2, t, cy * G + cx][None]
                LOC[pi, ai, t] = locate(p[arm][t][None], tpl).cpu().numpy()[0]
        if rank == 0 and n_done % 100 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분 남은 {el/(n_done+1)*(len(mine)-n_done-1)/60:.1f} 분", flush=True)
    for x in (S, LOC, DD): x.flush()
    print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    pairs = build_pairs()
    if a.limit: pairs = pairs[:: max(1, len(pairs) // a.limit)][: a.limit]
    n = len(pairs); print("pairs", n, collections.Counter((p["motion"], p["violation"]) for p in pairs), flush=True)
    for nm, sh in (("scores", (n, 6, 2, K)), ("loc", (n, 6, K, 4)), ("dist", (n, 6, K))):
        x = np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="w+", dtype=np.float32, shape=sh); x[:] = np.nan; del x
    json.dump(dict(n=n, arms=ARMS, pairs=pairs), open(Path(a.out) / "meta.json", "w"))
    mp.spawn(_worker, args=(a.gpus, a, pairs), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
