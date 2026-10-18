#!/usr/bin/env python3
"""설계 상한 (headroom) — 학습 없이 frozen encoder 토큰을 '적분해서 옮기기' 만 하면 어디까지 풀리나 (2026-10-13).

진단 (PREDICTOR_BOTTLENECK_2026-10-13): predictor 는 속도를 알지만 위치로 적분하지 않고, 자리는 '가까운 최근 내용 가져오기' 로 정한다.
그렇다면 설계 후보 '운동 · 외형 분리 (자리 = 마지막 본 자리 + 운동 적분, 외형 = 마지막으로 보인 물체 토큰을 그 자리로 옮김)' 는
학습 없이 조립만 해도 무엇을 푸나? — 그 상한이 설계 선택의 근거다.

공간: Ariel AR 과 같은 **블록별 LN(target_encoder)** (블록 = 2 장, 16 블록 = 32 샘플 stride 3). 문맥 C = 8 블록, 미래 K = 8 블록.
렌더 (모두 같은 문맥에서 만들어 matched 쌍의 두 미래에 같게 쓰인다):
  copy      문맥 마지막 블록 × 8
  ar        Ariel AR ep40 rollout (비교용 학습 모델)
  move_or   자리 = **정답 궤적** (라벨, 그 문맥 arm 의 물체) — 위치가 완벽할 때의 상한
  move_cv   자리 = 문맥 안 마지막 두 '보인' 샘플로 등속 외삽 (1 차 적분)
  move_ca   자리 = 문맥 안 '보인' 샘플에 2 차 맞춤 (가속까지, 2 차 적분)
  옮기기 = 캔버스 (문맥 마지막 블록; 물체가 보이면 그 3×3 을 물체가 멀리 있던 문맥 블록의 같은 칸으로 지움) 위에
           '마지막으로 완전히 보인 문맥 블록' 의 물체 3×3 토큰을 예측 자리 (블록 평균 중심) 에 붙인다. 가림은 고려하지 않는다 (쌍의 두 미래에 같게 들어간다).
  ⚠️ move_* 는 문맥의 **라벨 위치** (u, v) 를 쓴다 = 지각이 완벽하다는 가정. encoder 가 위치 · 속도를 R² 0.99 로 담는다는 기존 결과에 기댄다.
채점: S = mean |렌더 − LN(h)[미래 8 블록]| (블록별 S 도 저장). VOBench 는 matched 쌍, gravity 는 arm 들 중 가장 가까운 미래 (cross).

  python design_headroom.py extract --set vobench --gpus 4
  python design_headroom.py extract --set gravity --gpus 4
  python design_headroom.py analyze
산출: z_research/VOBench_v1/headroom/{vobench,gravity}_r*.npz · headroom.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
FR = Path("/data2/local_datasets/world/Benchmark/VOBench_v1")
OUT = REPO / "z_research/VOBench_v1/headroom"
SETS = {
    "vobench": dict(index=REPO / "data_csv/vobench_v1/index.csv", cfg=REPO / "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/_resolved.yaml", root=FR),
    "gravity": dict(index=REPO / "data_csv/vobench_v1_gravity/index.csv", cfg=REPO / "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_gravity_vith/_resolved.yaml", root=FR / "gravity"),
}
S, NB, C, K, D, G, CELL = 256, 16, 8, 8, 1280, 16, 32.0
RENDERS = ("copy", "ar", "move_or", "move_cv", "move_ca")
MAPS = os.environ.get("HEADROOM_MAPS", "1") == "1"
GRAV_ARM = {"pos": "P", "imp_rise": "R", "imp_float": "F", "imp_line": "L", "imp_stop": "S", "imp_stop2": "S2", "imp_stop4": "S4", "imp_stop8": "S8", "imp_stop12": "S12"}


def track_vobench(r):
    lab = json.load(open(FR / Path(r["file_name"]).parent / "label.json"))["labels"]
    arm = "A" if r["variant"] in ("pos_a", "imp_ab") else "B"                     # 문맥 arm (matched 쌍은 같은 문맥)
    L = lab.get(arm)
    if not L or L[0] is None: return None
    P = np.array([[L[3 * s]["u"], L[3 * s]["v"]] for s in range(32)], np.float64)
    hid = np.array([L[3 * s]["hidden"] for s in range(32)], np.float64)
    return P, hid


def track_gravity(r):
    lab = json.load(open(FR / "gravity" / Path(r["file_name"]).parent / "label.json"))["labels"]
    L = lab["P"]                                                                      # 문맥은 모든 arm 이 같다 (f0–47); 정답 궤적 = P
    P = np.array([[L[3 * s]["u"], L[3 * s]["v"]] for s in range(32)], np.float64)
    hid = np.array([L[3 * s]["hidden"] for s in range(32)], np.float64)
    return P, hid


def plan_positions(P, hid):
    """미래 블록 8 개의 중심 (or · cv · ca) — 샘플 16..31 을 블록으로 평균. 문맥은 샘플 0..15 중 보인 것만."""
    vis = [s for s in range(16) if hid[s] < 0.5]
    out = {"move_or": P[16:].reshape(8, 2, 2).mean(1)}
    if len(vis) >= 2:
        s1, s0 = vis[-1], vis[-2]; v = (P[s1] - P[s0]) / (s1 - s0)
        fut = np.arange(16, 32)
        out["move_cv"] = (P[s1][None] + v[None] * (fut - s1)[:, None]).reshape(8, 2, 2).mean(1)
        w = vis[-8:] if len(vis) >= 3 else vis
        if len(w) >= 3:
            t = np.array(w, np.float64); A = np.stack([np.ones_like(t), t, t ** 2], 1)
            coef = np.linalg.lstsq(A, P[w], rcond=None)[0]
            out["move_ca"] = (np.stack([np.ones(16), fut, fut ** 2], 1) @ coef).reshape(8, 2, 2).mean(1)
        else:
            out["move_ca"] = out["move_cv"]
    else:
        out["move_cv"] = out["move_ca"] = np.repeat(P[vis[-1] if vis else 15][None], 8, 0)
    blk_vis = [b for b in range(C) if hid[2 * b] < 0.5 and hid[2 * b + 1] < 0.5]
    return out, (blk_vis[-1] if blk_vis else None), P[:16].reshape(8, 2, 2).mean(1)


def tok(xy):
    return int(np.clip(xy[1] // CELL, 0, G - 1)), int(np.clip(xy[0] // CELL, 0, G - 1))


def _worker(rank, world, setname, bs):
    import torch
    import yaml
    from importlib.util import module_from_spec, spec_from_file_location
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import block_indices, encode_blocks
    from evals.world_model_analysis.data import WMADataset
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    os.environ.setdefault("PRESENCE_SET", "vob_readout_primitive")
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir)
    model = dict(ir.rp.MODEL, predictor_checkpoint=str(REPO / "z_training/ariel/ar_future_1e_5/latest.pt"))
    bundle = build_from_config(model, dev); pred = bundle.predictor
    sp = SETS[setname]; cfg = yaml.safe_load(open(sp["cfg"]))
    ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=model, features={"cache_dir": "/tmp"}, surprise={}))
    rows = {r["video_id"]: r for r in csv.DictReader(open(sp["index"]))}
    ids = [i for i, rec in enumerate(ds.records) if rec.video_id in rows]
    mine = ids[rank * len(ids) // world:(rank + 1) * len(ids) // world]
    tr = track_vobench if setname == "vobench" else track_gravity
    rec = defaultdict(list); t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b0 in range(0, len(mine), bs):
            ch = mine[b0:b0 + bs]
            x = torch.stack([ds.clip(i) for i in ch]).to(dev, dtype=bundle.dtype); B = x.size(0)
            hb = encode_blocks(bundle.target_encoder, x, 2).float()                        # (B, 16S, D)
            ar = pred.rollout(hb[:, :C * S].to(torch.bfloat16), block_indices(B, C, S, dev), K).float()
            hblk = hb.view(B, NB, G, G, D); fut = hblk[:, C:]
            for b, i in enumerate(ch):
                r = rows[ds.records[i].video_id]
                t = None
                renders = {"copy": hblk[b, C - 1].unsqueeze(0).expand(K, -1, -1, -1), "ar": ar[b].view(K, G, G, D)}
                t = tr(r)
                if t is None:                                                              # 빈 문맥 (물체 없음): 옮길 것 없음 = 복사
                    for m in ("move_or", "move_cv", "move_ca"): renders[m] = renders["copy"]
                else:
                    P, hid = t; pos, last_vis, ctx_c = plan_positions(P, hid)
                    canvas = hblk[b, C - 1].clone()
                    if hid[14] < 0.5 and hid[15] < 0.5:                                    # 문맥 마지막 블록에 물체가 보이면 지운다
                        ry, rx = tok(ctx_c[C - 1])
                        far = [bb for bb in range(C) if np.abs(np.array(tok(ctx_c[bb])) - np.array((ry, rx))).max() >= 3]
                        if far:
                            src = hblk[b, far[0]]
                            for dy in (-1, 0, 1):
                                for dx in (-1, 0, 1):
                                    yy, xx = ry + dy, rx + dx
                                    if 0 <= yy < G and 0 <= xx < G: canvas[yy, xx] = src[yy, xx]
                    if last_vis is None:
                        for m in ("move_or", "move_cv", "move_ca"): renders[m] = canvas.unsqueeze(0).expand(K, -1, -1, -1)
                    else:
                        oy, ox = tok(ctx_c[last_vis]); src = hblk[b, last_vis]
                        for m in ("move_or", "move_cv", "move_ca"):
                            frames = canvas.unsqueeze(0).repeat(K, 1, 1, 1)
                            for k in range(K):
                                cy, cx = tok(pos[m][k]) if (0 <= pos[m][k][0] < 512 and 0 <= pos[m][k][1] < 512) else (None, None)
                                if cy is None: continue
                                for dy in (-1, 0, 1):
                                    for dx in (-1, 0, 1):
                                        sy, sx, ty, tx = oy + dy, ox + dx, cy + dy, cx + dx
                                        if 0 <= sy < G and 0 <= sx < G and 0 <= ty < G and 0 <= tx < G: frames[k, ty, tx] = src[sy, sx]
                            renders[m] = frames
                for m in RENDERS:
                    em = (renders[m].float() - fut[b]).abs().mean(-1)                      # (K, G, G)
                    rec[f"S_{m}"].append(em.mean((-1, -2)).cpu().numpy())
                    if MAPS: rec[f"E_{m}"].append(em.half().cpu().numpy())
                # 옮김 가능성: 같은 물체가 문맥 안 두 자리 (≥ 3 칸 떨어짐) 에 보일 때, 옮긴 토큰이 그 자리의 참 토큰에 얼마나 가까운가
                tv = np.full(4, np.nan, np.float32)
                if t is not None:
                    vb = [bb for bb in range(C) if hid[2 * bb] < 0.5 and hid[2 * bb + 1] < 0.5]
                    if len(vb) >= 2:
                        i0, j0 = vb[0], vb[-1]; ci, cj = tok(ctx_c[i0]), tok(ctx_c[j0])
                        if max(abs(ci[0] - cj[0]), abs(ci[1] - cj[1])) >= 3 and all(1 <= c <= G - 2 for c in ci + cj):
                            def patch(bb, c): return hblk[b, bb, c[0] - 1:c[0] + 2, c[1] - 1:c[1] + 2].float()
                            tv[0] = (patch(j0, cj) - patch(i0, ci)).abs().mean().item()            # 옮긴 물체 vs 참 물체
                            tv[1] = (patch(j0, cj) - patch(i0, cj)).abs().mean().item()            # 그 자리의 이전 배경 vs 참 물체 (= 옮기지 않으면)
                            tv[2] = (patch(j0, ci) - patch(i0, ci)).abs().mean().item()            # 물체가 떠난 자리 (배경) vs 이전 물체
                            far = (int((ci[0] + G // 2) % G), int((ci[1] + G // 2) % G))
                            far = (min(max(far[0], 1), G - 2), min(max(far[1], 1), G - 2))
                            tv[3] = (patch(j0, far) - patch(i0, far)).abs().mean().item()          # 먼 배경끼리 (같은 자리, 다른 시점)
                rec["transport"].append(tv)
                rec["video_id"].append(ds.records[i].video_id)
            if rank == 0 and (b0 // bs) % 10 == 0: print(f"[r0] {setname} {b0 + B}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{setname}_r{rank}.npz", video_id=np.array(rec.pop("video_id")), **{k: np.stack(v) for k, v in rec.items()})
    print(f"[r{rank}] {setname} done ({time.time() - t0:.0f}s)", flush=True)


def load(setname):
    Z = defaultdict(list)
    for f in sorted(OUT.glob(f"{setname}_r*.npz")):
        z = np.load(f, allow_pickle=True)
        for k in z.files: Z[k].append(z[k])
    return {k: np.concatenate(v) for k, v in Z.items()}


def analyze():
    res = {}
    # ---- VOBench matched 쌍
    Z = load("vobench"); vid = {v: i for i, v in enumerate(Z["video_id"])}
    rows = {r["video_id"]: r for r in csv.DictReader(open(SETS["vobench"]["index"]))}
    by_blk = defaultdict(dict)
    for v, r in rows.items(): by_blk[r["source_block"]][r["variant"]] = v
    acc = defaultdict(lambda: defaultdict(list))
    for sb, arms in by_blk.items():
        if not all(a in arms for a in ("pos_a", "pos_b", "imp_ab", "imp_ba")) or not all(arms[a] in vid for a in arms): continue
        r = rows[arms["pos_a"]]
        mot = "static" if r["motion"] == "flat_static" else "moving"
        for pos, imp, dirn in (("pos_a", "imp_ab", "A"), ("pos_b", "imp_ba", "B")):
            ip, ii = vid[arms[pos]], vid[arms[imp]]
            for m in RENDERS:
                sp_, si_ = Z[f"S_{m}"][ip].mean(), Z[f"S_{m}"][ii].mean()
                hit = 1.0 if si_ > sp_ else (0.5 if si_ == sp_ else 0.0)
                for key in ("all", f"{r['objset']}", f"{r['occlusion']}", f"{r['violation']}/{r['occlusion']}", f"{r['violation']}/{r['occlusion']}/{mot}",
                            f"{r['occlusion']}/{mot}", f"{r['violation']}/{r['occlusion']}/{mot}/{dirn}"):
                    acc[key][m].append(hit)
    res["vobench"] = {k: {m: [float(np.mean(v)), len(v)] for m, v in d.items()} for k, d in acc.items()}
    # ---- gravity: 가장 가까운 미래 (arm 들 중 argmin S)
    Zg = load("gravity"); vidg = {v: i for i, v in enumerate(Zg["video_id"])}
    rowsg = {r["video_id"]: r for r in csv.DictReader(open(SETS["gravity"]["index"]))}
    blk = defaultdict(dict)
    for v, r in rowsg.items(): blk[r["source_block"]][r["variant"]] = v
    ch = defaultdict(lambda: defaultdict(lambda: defaultdict(int))); ledge = defaultdict(list)
    for sb, arms in blk.items():
        r = rowsg[arms["pos"]]; cell = f"{r['law']}/{r['cell']}" if r["law"] == "arc" else "ledge"
        if not all(a in vidg for a in arms.values()): continue
        for m in RENDERS:
            sc = {a: Zg[f"S_{m}"][vidg[v]].mean() for a, v in arms.items()}
            best = min(sc, key=sc.get); ch[cell][m][best] += 1
            if r["law"] == "ledge": ledge[m].append(1.0 if sc["imp_float"] > sc["pos"] else 0.0)
    res["gravity_choice"] = {c: {m: dict(d) for m, d in dd.items()} for c, dd in ch.items()}
    res["ledge_pair_acc"] = {m: float(np.mean(v)) for m, v in ledge.items()}
    json.dump(res, open(OUT / "headroom.json", "w"), indent=1)
    V = res["vobench"]
    print("VOBench matched 쌍 정확도 (AR 공간) — copy · ar · move_or · move_cv · move_ca  [쌍 수]")
    keys = ["all", "primitive", "realistic", "visible", "early", "mid", "late"] + sorted(k for k in V if k.count("/") == 2 and not k.endswith(("/A", "/B")))
    for k in keys:
        if k in V: print(f"  {k:32s} " + "  ".join(f"{100 * V[k][m][0]:5.1f}" for m in RENDERS) + f"  [{V[k]['copy'][1]}]")
    print("late 방향별 (A = 물체→빈 / 다른 물체):")
    for k in sorted(k for k in V if k.startswith(("permanence/late", "shape/late", "color/late")) and k.endswith(("/A", "/B"))):
        print(f"  {k:32s} " + "  ".join(f"{100 * V[k][m][0]:5.1f}" for m in RENDERS))
    print("\ngravity — 고른 미래 (arm: block 수)")
    for c, dd in sorted(res["gravity_choice"].items()):
        for m in RENDERS: print(f"  {c:14s} {m:8s} {dict(sorted(dd[m].items(), key=lambda x: -x[1]))}")
    print("ledge 쌍 (pos vs imp_float) 정확도:", {m: round(100 * v, 1) for m, v in res["ledge_pair_acc"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["extract", "analyze"]); ap.add_argument("--set", default="vobench")
    ap.add_argument("--gpus", type=int, default=4); ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    if a.phase == "extract":
        import torch
        torch.multiprocessing.spawn(_worker, args=(a.gpus, a.set, a.bs), nprocs=a.gpus, join=True)
    else:
        analyze()
