#!/usr/bin/env python3
"""transfer_gap R4 E39b — 영속 칸의 자리 지도 (R2 규칙 1): 다시 나타난 물체의 존재 대비가 참 자리에 서나 (2026-10-30, 실험 A [X1]).

정의 (block 마다, 같은 장면 두 가능 영상: 물체 팔 문맥 pos_a · 빈 장면 팔 문맥 pos_b):
  D[t, y, x] = ‖p(pos_a 문맥)[t, y, x] − p(pos_b 문맥)[t, y, x]‖₂  (미래 튜블릿 t, 칸 y · x)
  다시 나타난 미래 튜블릿 (참 물체가 두 장 모두 보임 · 화면 안) 마다 3×3 상자 합 (중심 1–14) 최대 자리가 참 자리 (블록 평균 토큰)
  1 칸 안 (체비쇼프 ≤ 1) 이면 맞음. W = 맞음 비율 (튜블릿 단위, CI = block bootstrap 2,000).
  기준선: 복사 = 문맥 마지막 튜블릿 LN(z) 의 두 팔 차 지도 (모든 미래 튜블릿에 같다) ·
          전형 자리 = 같은 판 · 같은 칸 (조건 · k) · 같은 문맥 운동 방향 (가로 부호) 의 다른 block 들의 그 튜블릿 참 자리 중앙값 (한 block 빼기, 학습 없음).
판: vob = VOBench split v2 test late 이동 permanence (256² 사본) · v11 = v11 split test 가림 이동 사라짐 (원본 화소 사본) · v11r = v11_realistic 가림 이동 사라짐 (원본 화소 사본).
모델: E39 의 te 행 10 개 (릴리즈 + PFT 9). float32 + autocast float16, 문맥 = online encoder.
재현:
  sbatch -p batch_vll -w vll4 --gres=gpu:1 ... --wrap "$P z_research/scripts/analysis/transfer_gap_r4_presence.py run"
  $P z_research/scripts/analysis/transfer_gap_r4_presence.py summarize     # → results/r4_x1/E39b_presence.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
S, NB, C, K, G = 256, 16, 8, 8, 16
OUTR = REPO / "z_analysis/transfer_gap/results/r4_x1"
OUT = OUTR / "E39b"
STAGE = OUTR / "stage"
FR = REPO / "z_pixel_decoder/runs/_frames"
SETS = {"vob": FR / "VOBench_v1_256_x1r4", "v11": FR / "IntPhysGen_v11_sub", "v11r": FR / "IntPhysGen_v11_realistic_sub"}


def preds_list():
    out = []
    for kv in open(OUTR / "te/preds.txt").read().strip().split(","):
        t, p = kv.split("="); out.append((t, p))
    return out


def geom_px(px, py, hid, cell_px):
    tok = np.stack([np.clip(py // cell_px, 0, G - 1), np.clip(px // cell_px, 0, G - 1)], 1).reshape(NB, 2, 2).mean(1)
    yx = np.clip(np.floor(tok + 0.5).astype(int), 1, G - 2)
    lim = cell_px * G
    ins = (px >= 0) & (px < lim) & (py >= 0) & (py < lim)
    vb = ((hid < 0.5) & ins).reshape(NB, 2).all(1)
    return yx, vb


def blocks(setname):
    """[(block, cell, dir, row_a, row_b, yx, vb)]"""
    out = []
    if setname == "vob":
        T = np.load(STAGE / "vobench_tracks.npz", allow_pickle=True); ti = {v: i for i, v in enumerate(T["vid"])}
        te = {r["video_id"] for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_test.csv"))}
        by = defaultdict(dict)
        for r in csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv")):
            if r["occlusion"] == "late" and r["motion"] != "flat_static" and r["violation"] == "permanence" and r["video_id"] in te:
                by[(r["objset"], r["block_id"])][r["variant"]] = r
        for key, v in by.items():
            j = ti[v["pos_a"]["video_id"]]; P, hid = T["P"][j], T["hid"][j]
            yx, vb = geom_px(P[:, 0], P[:, 1], hid, 32)
            out.append(("|".join(key), f"{v['pos_a']['motion']}|k{v['pos_a']['k']}", np.sign(P[15, 0] - P[0, 0]), v["pos_a"], v["pos_b"], yx, vb))
    else:
        import pandas as pd
        if setname == "v11":
            idx = STAGE / "idx_v11sub3/index.csv"; meta = pd.read_csv(STAGE / "v11_meta_sub.csv").set_index("name")
        else:
            idx = REPO / "data_csv/intphysgen_v11_realistic/index.csv"; meta = pd.read_csv(STAGE / "v11r_meta.csv").set_index("name")
        by = defaultdict(dict)
        for r in csv.DictReader(open(idx)):
            if r["condition"] in ("moving_occlusion", "moving_occlusion_flat"):
                by[r["block_id"]][r["variant"]] = r
        for b, v in by.items():
            m = meta.loc[v["pos_a"]["video_id"]]
            px = np.array([float(x) for x in str(m["object_px_x_by_sample"]).split()]); py = np.array([float(x) for x in str(m["object_px_y_by_sample"]).split()])
            k = int(v["pos_a"]["sym_k"]); hid = np.zeros(32); hid[16 - k:16 + k] = 1.0
            yx, vb = geom_px(px, py, hid, 18)
            out.append((b, f"{v['pos_a']['condition']}|k{k}", np.sign(px[15] - px[0]), v["pos_a"], v["pos_b"], yx, vb))
    return out


def _worker(rank, world, sets, bs):
    import torch
    import copy as _copy
    from analysis.intphys2.model import build_from_config, _clean_backbone_key, _load_state_dict
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.data import WMADataset
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    model = dict(checkpoint=str(REPO / "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth"),
                 arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, window_size=32, use_rope=True, uniform_power=False,
                 dual_encoder=True, context_encoder_key="encoder", target_encoder_key="target_encoder",
                 predictor=dict(embed_dim=384, depth=12, num_heads=12, num_mask_tokens=10), dtype="float32")
    bundle = build_from_config(model, dev)
    preds = {}
    for t, p in preds_list():
        if t == "release": preds[t] = bundle.predictor; continue
        pm = _copy.deepcopy(bundle.predictor); st = torch.load(REPO / p, map_location="cpu", weights_only=False)
        assert (st.get("arch") or {}).get("kind") in (None, "default", "oneshot"), t
        assert not (set(pm.state_dict()) - set(_clean_backbone_key(st["predictor"]))), t
        _load_state_dict(pm, st["predictor"], tag=t); preds[t] = pm.to(dev).eval()
    ac = torch.autocast("cuda", dtype=torch.float16)
    for setname in sets:
        B_ = blocks(setname)[rank::world]
        rows = [b[3] for b in B_] + [b[4] for b in B_]
        idir = OUT / f"index_{setname}_r{rank}"; idir.mkdir(parents=True, exist_ok=True)
        with open(idir / "index.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        ds = WMADataset(dict(data=dict(root=str(idir), index_csv="index.csv", frames_root=str(SETS[setname]), frames_pattern="{file_name}/{frame:06d}.png",
                                       frames_start=0, frames_stride=3, n_frames=32, resolution=256, block_column="block_id", pair_column="pair_id",
                                       variant_column="variant", plausible_column="plausible", type_column="condition"),
                             model=model, features={"cache_dir": "/tmp"}, surprise={}))
        di = {r.video_id: i for i, r in enumerate(ds.records)}
        acc = defaultdict(list); t0 = time.time()
        for g0 in range(0, len(B_), bs):
            gg = B_[g0:g0 + bs]; n = len(gg)
            x = torch.stack([ds.clip(di[b[3]["video_id"]]) for b in gg] + [ds.clip(di[b[4]["video_id"]]) for b in gg]).to(dev)
            ci, ti = _context_target_indices(ctx_frames=2 * C, tgt_frames=2 * K, tubelet_size=2, spatial_tokens=S, batch_size=2 * n, device=dev)
            with torch.no_grad(), ac:
                z = bundle.context_encoder(x, masks=[ci]); z = z[-1] if isinstance(z, list) else z
                zl = torch.nn.functional.layer_norm(z.float(), (z.size(-1),))[:, (C - 1) * S:C * S]
                Dc = (zl[:n] - zl[n:]).norm(dim=-1).view(n, 1, G, G).expand(n, K, G, G)
                acc["copy"].append(torch.nn.functional.avg_pool2d(Dc.reshape(n * K, 1, G, G), 3, 1, 1).view(n, K, G, G)[:, :, 1:-1, 1:-1].cpu().numpy())
                for t, pm in preds.items():
                    p = pm(z, ci, ti, mask_index=0).float()
                    D = (p[:n] - p[n:]).norm(dim=-1).view(n, K, G, G)
                    acc[t].append(torch.nn.functional.avg_pool2d(D.reshape(n * K, 1, G, G), 3, 1, 1).view(n, K, G, G)[:, :, 1:-1, 1:-1].cpu().numpy())
            acc["block"].append(np.array([b[0] for b in gg])); acc["cell"].append(np.array([b[1] for b in gg])); acc["dir"].append(np.array([b[2] for b in gg]))
            acc["yx"].append(np.stack([b[5] for b in gg])); acc["vb"].append(np.stack([b[6] for b in gg]))
            if rank == 0: print(f"[r0] {setname} {g0 + n}/{len(B_)} ({time.time() - t0:.0f}s)", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / f"{setname}_r{rank}.npz", **{k: np.concatenate(v) for k, v in acc.items()})


def load(setname):
    Z = {}
    for f in sorted(OUT.glob(f"{setname}_r*.npz")):
        z = np.load(f)
        for k in z.files: Z.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in Z.items()}


def boot(x, b, ref=None, nb=2000, seed=0):
    xs = x if ref is None else x - ref
    u, inv = np.unique(b, return_inverse=True)
    s = np.bincount(inv, xs, len(u)); n = np.bincount(inv, None, len(u))
    rng = np.random.default_rng(seed); bs = []
    for _ in range(nb):
        ii = rng.integers(0, len(u), len(u)); bs.append(s[ii].sum() / n[ii].sum())
    return [round(100 * float(xs.mean()), 1), round(100 * float(np.percentile(bs, 2.5)), 1), round(100 * float(np.percentile(bs, 97.5)), 1)]


def summarize():
    E39 = json.load(open(OUTR / "E39_matrix.json"))
    out = {"definition": __doc__.split("정의")[1].split("판:")[0].strip(), "sets": {}, "verdicts": {}}
    for setname in ("vob", "v11", "v11r"):
        if not list(OUT.glob(f"{setname}_r*.npz")): continue
        Z = load(setname); yx, vb, blk, cell, dr = Z["yx"], Z["vb"], Z["block"], Z["cell"], Z["dir"]
        names = [k for k in Z if k not in ("yx", "vb", "block", "cell", "dir")]
        hits = {nm: [] for nm in names}; hits["prior"] = []; bl = []
        for i in range(len(blk)):
            others = [j for j in range(len(blk)) if j != i and cell[j] == cell[i] and dr[j] == dr[i]]
            for t in range(K):
                if not vb[i, C + t]: continue
                y, x = yx[i, C + t]
                for nm in names:
                    m = Z[nm][i, t]; a = np.unravel_index(np.argmax(m), m.shape)
                    hits[nm].append(float(max(abs(a[0] + 1 - y), abs(a[1] + 1 - x)) <= 1))
                oj = [j for j in others if vb[j, C + t]]
                if oj:
                    med = np.median(yx[oj, C + t], 0)
                    hits["prior"].append(float(max(abs(med[0] - y), abs(med[1] - x)) <= 1))
                else:
                    hits["prior"].append(np.nan)
                bl.append(blk[i])
        bl = np.array(bl); res = {}
        for nm, h in hits.items():
            h = np.array(h); ok = ~np.isnan(h)
            res[nm] = {"W": boot(h[ok], bl[ok]), "n_step": int(ok.sum())}
            if nm not in ("release", "copy", "prior"):
                res[nm]["dW_vs_release"] = boot(h[ok], bl[ok], np.array(hits["release"])[ok])
        out["sets"][setname] = dict(res, n_block=int(len(np.unique(blk))))
        col = {"vob": "vob", "v11": "v11", "v11r": "v11r"}[setname]
        for nm, r in res.items():
            if nm in ("release", "copy", "prior"): continue
            v = E39["verdicts"].get(nm, {}).get(col)
            if not v or "영속 이득" not in v["verdict"]: continue
            W, dW = r["W"], r["dW_vs_release"]
            if W[0] >= 50 and dW[0] >= 20 and dW[1] > 0:
                vv = "영속 (자리까지)" + (" · 자리는 전형 자리 사전과 구분 안 됨" if res["prior"]["W"][0] >= W[0] - 5 else "")
            else:
                vv = "사라짐 점수 이득 (자리 없음)"
            out["verdicts"].setdefault(nm, {})[col] = dict(E39=v["verdict"], W=W, dW=dW, W_prior=res["prior"]["W"], W_copy=res["copy"]["W"], verdict=vv)
        print(f"\n== {setname} (block {len(np.unique(blk))})")
        for nm, r in res.items():
            print(f"  {nm:10s} W {r['W'][0]:5.1f} [{r['W'][1]:.0f},{r['W'][2]:.0f}]" + (f"  ΔW {r['dW_vs_release'][0]:+.1f} [{r['dW_vs_release'][1]:+.0f},{r['dW_vs_release'][2]:+.0f}]" if "dW_vs_release" in r else "") + f"  (step {r['n_step']})")
    json.dump(out, open(OUTR / "E39b_presence.json", "w"), indent=1, ensure_ascii=False)
    print("\n판정:")
    for nm, v in out["verdicts"].items():
        print(f"  {nm:10s} " + " | ".join(f"{c}: {d['verdict']} (W {d['W'][0]:.0f}, ΔW {d['dW'][0]:+.0f}, 사전 {d['W_prior'][0]:.0f})" for c, d in v.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--sets", default="vob,v11,v11r"); ap.add_argument("--gpus", type=int, default=1); ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "run":
        import torch
        torch.multiprocessing.spawn(_worker, args=(a.gpus, a.sets.split(","), a.bs), nprocs=a.gpus, join=True)
    summarize()
