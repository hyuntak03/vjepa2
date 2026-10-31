#!/usr/bin/env python3
"""[X1] E8 문맥 토큰 바꿔 넣기 — predictor 미래의 정체는 어느 문맥 튜블릿에서 오나 (H2-2) (2026-10-30).

모델: 릴리즈 ViT-H (probing 관례 bfloat16 · autocast none, vobench_tubelet_probe 와 같은 경로). 대상: 모양 · 색 block 전부 (두 objset, 모든 가림 조건).
A 영상의 문맥 encoder 출력 z_A (튜블릿 8 × 칸 256) 중 튜블릿 집합 J 의 토큰을 같은 block B 영상 (같은 궤적 · 다른 물체) 의 z_B 로 바꿔 predictor 에 넣는다.
  J ∈ {∅, occ = 물체가 가려진 샘플이 하나라도 있는 문맥 튜블릿, end = 마지막으로 다 보인 문맥 튜블릿, vis = 다 보인 문맥 튜블릿 전부, all = 0..7}
  π_J = ⟨vec(p̃_J − p̃_A), vec(p̃_B − p̃_A)⟩ / ‖vec(p̃_B − p̃_A)‖²   (p̃ = 토큰 LN, 미래 8 걸음 · 전 칸) · 물체 영역판 π^O (참 물체 3×3/외곽 상자, 걸음별)
  건전성: π_∅ = 0, π_all = 1 (± 1e-3) — 같은 batch 안에서 계산.
  late 안 ridge: split v2 train block 의 원래 late p (블록 평균, 6 지선다 모양) 로 맞춘 one-hot ridge 가 test block 의 p_J 를 B 의 모양으로 읽는 비율.
판정 (PLAN §3): H2-2 = π_occ ≤ 0.1 이고 π_end ≥ 0.5. 기각 = π_occ ≥ 0.3.
출력: results/q1q2/E8_swap.json · E8_rows.jsonl

  $P z_research/scripts/analysis/transfer_gap_swap.py --gpu 0
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_gap_common as C  # noqa: E402

NT, NF, S, D = C.NT, C.NF, C.S, C.D


def main(gpu, limit):
    import torch
    sys.path.insert(0, str(C.REPO))
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.data import WMADataset
    import yaml
    dev = torch.device(f"cuda:{gpu}"); torch.cuda.set_device(dev)
    cfg = yaml.safe_load(open(C.REPO / "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/_resolved.yaml"))
    cfg["model"]["dtype"] = "bfloat16"; cfg["model"]["autocast"] = "none"; cfg["features"] = {"decode_workers": 6}
    ds = WMADataset(cfg); vid2rec = {r.video_id: i for i, r in enumerate(ds.records)}
    model = build_from_config(cfg["model"], dev)
    M = C.meta(); trb = C.train_blocks(); bmap, UV, HID, EXT = C.load_tracks()
    blocks = defaultdict(dict)
    for j, m in enumerate(M):
        blocks[m["block_id"]][m["variant"]] = j
    bl = sorted([b for b in blocks if M[blocks[b]["pos_a"]]["violation"] in ("shape", "color")], key=int)[: limit or None]
    LN = lambda x: torch.nn.functional.layer_norm(x.float(), (x.shape[-1],))
    rows, feats = [], []
    t0 = time.time()
    for n_, b in enumerate(bl):
        ja, jb = blocks[b]["pos_a"], blocks[b]["pos_b"]; m = M[ja]
        st = C.tub_state(m["samples"])
        Jsets = {"none": [], "occ": [t for t in range(NF) if st[t] != "vis"], "all": list(range(NF))}
        visc = [t for t in range(NF) if st[t] == "vis"]
        Jsets["vis"] = visc
        Jsets["end"] = visc[-1:] if visc else []
        x = torch.stack([ds.clip(vid2rec[M[ja]["video_id"]]), ds.clip(vid2rec[M[jb]["video_id"]])]).to(dev, model.dtype)
        names = list(Jsets)
        with torch.inference_mode():
            ci, ti = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=S, batch_size=2 + len(names), device=dev)
            ci2, _ = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=S, batch_size=2, device=dev)
            z = model.context_encoder(x, masks=[ci2]); z = z[-1] if isinstance(z, list) else z            # (2, 8·256, D)
            zA, zB = z[0].reshape(NF, S, -1), z[1].reshape(NF, S, -1)
            zs = [zA, zB]
            for nm in names:
                zz = zA.clone()
                if Jsets[nm]:
                    zz[Jsets[nm]] = zB[Jsets[nm]]
                zs.append(zz)
            zin = torch.stack([q.reshape(NF * S, -1) for q in zs])
            p = model.predictor(zin, ci, ti, mask_index=0)
            pt = LN(p).reshape(len(zs), NF, S, -1)
        pA, pB = pt[0], pt[1]
        dBA = (pB - pA); den = float((dBA * dBA).sum())
        bi = bmap[b]; uv, ext = UV[bi, 0], EXT[bi, 0]
        Ocells = [C.region(uv[t], ext[t]) for t in range(NF, NT)]
        denO = sum(float((dBA[k, Ocells[k]] ** 2).sum()) for k in range(NF))
        r = dict(block=b, objset=m["objset"], violation=m["violation"], occlusion=m["occlusion"], k=m["k"], motion=m["motion"],
                 test=b not in trb, J={nm: Jsets[nm] for nm in names}, obj_a=m["obj_a"], obj_b=m["obj_b"])
        for q, nm in enumerate(names):
            d = pt[2 + q] - pA
            r[f"pi_{nm}"] = float((d * dBA).sum()) / den
            r[f"piO_{nm}"] = sum(float((d[k, Ocells[k]] * dBA[k, Ocells[k]]).sum()) for k in range(NF)) / max(denO, 1e-12)
            feats.append((b, nm, pt[2 + q].mean(1).cpu().numpy().astype(np.float32)))   # (8, D) 블록 평균
        r["pi_B_check"] = float(((pt[1] - pA) * dBA).sum()) / den
        rows.append(r)
        if n_ % 40 == 0:
            print(f"swap {n_}/{len(bl)} {time.time() - t0:.0f}s · {r['occlusion']} π: " + " ".join(f"{nm} {r['pi_' + nm]:.3f}" for nm in names), flush=True)
    return rows, feats


def ridge_read(rows, feats):
    """late 안 ridge: train block 의 원래 late p (tubeprobe, 블록 평균, 걸음 평균) → 모양 6 지선다. test block 의 p_J 가 B 의 모양으로 읽히나."""
    M = C.meta(); trb = C.train_blocks()
    P = np.load(C.TUBE / "p.npy", mmap_mode="r")
    out = {}
    for o in ("primitive", "realistic"):
        tr = [j for j, m in enumerate(M) if m["objset"] == o and m["occlusion"] == "late" and C.label_of(m) != "empty" and m["block_id"] in trb]
        Xtr = np.stack([np.asarray(P[j], np.float32) for j in tr])                           # (n, 8, 256, D)
        Xtr = (Xtr - Xtr.mean(-1, keepdims=True)) / (Xtr.std(-1, keepdims=True) + 1e-6)     # LN
        Xtr = Xtr.mean(2).mean(1)                                                             # 블록 · 걸음 평균 (n, D)
        ytr = np.array([C.label_of(M[j]).split("/")[0] for j in tr]); cls = np.unique(ytr)
        Y = (ytr[:, None] == cls[None]).astype(np.float64)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; A = (Xtr - mu) / sd
        W = np.linalg.solve(A.T @ A + 100.0 * np.eye(A.shape[1]), A.T @ (Y - Y.mean(0)))
        res = defaultdict(list)
        rowd = {r["block"]: r for r in rows}
        for b, nm, f in feats:
            r = rowd[b]
            if r["objset"] != o or not r["test"]:
                continue
            pr = cls[int((((f.mean(0) - mu) / sd) @ W).argmax())]
            res[(r["occlusion"], nm)].append((pr == r["obj_b"].split("/")[0], pr == r["obj_a"].split("/")[0]))
        out[o] = {f"{oc}|{nm}": dict(n=len(v), read_as_B=100 * float(np.mean([a for a, _ in v])), read_as_A=100 * float(np.mean([c for _, c in v])))
                  for (oc, nm), v in sorted(res.items())}
    return out


def summarize(rows):
    out = {}
    names = ["none", "occ", "end", "vis", "all"]
    for o in ("primitive", "realistic", "both"):
        for oc in ("visible", "early", "mid", "late"):
            for mv in ("move", "static", "all"):
                sel = [r for r in rows if (o == "both" or r["objset"] == o) and r["occlusion"] == oc
                       and (mv == "all" or (r["motion"] != "flat_static") == (mv == "move"))]
                if not sel:
                    continue
                d = dict(n_block=len(sel))
                for nm in names:
                    for pre in ("pi", "piO"):
                        v = [r[f"{pre}_{nm}"] for r in sel if (nm != "occ" or r["J"]["occ"]) and (nm != "end" or r["J"]["end"])]
                        if v:
                            d[f"{pre}_{nm}"] = C.bootstrap_mean(v, [r["block"] for r in sel if (nm != "occ" or r["J"]["occ"]) and (nm != "end" or r["J"]["end"])])
                out[f"{o}|{oc}|{mv}"] = d
    chk = dict(pi_none_absmax=float(max(abs(r["pi_none"]) for r in rows)), pi_all_absdev_max=float(max(abs(r["pi_all"] - 1) for r in rows)))
    return out, chk


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows, feats = main(a.gpu, a.limit)
    summ, chk = summarize(rows)
    print("sanity", chk, flush=True)
    rr = ridge_read(rows, feats)
    suf = "" if not a.limit else f"_smoke{a.limit}"
    C.RES.mkdir(parents=True, exist_ok=True)
    json.dump(dict(definition=__doc__, sanity=chk, summary=summ, ridge_read=rr), open(C.RES / f"E8_swap{suf}.json", "w"), indent=1, ensure_ascii=False)
    with open(C.RES / f"E8_rows{suf}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("→", C.RES / f"E8_swap{suf}.json")
