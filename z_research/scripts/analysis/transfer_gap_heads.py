#!/usr/bin/env python3
"""[X1] E3 판독기 다시 맞춤 (split v2) — h 판독기 · p 판독기 · 같은 시점 판독기 + late 안 ridge · 복사 판독 (2026-10-30).

판독기 (vobench_tubelet_probe.run_objset 과 같은 구조 · 같은 학습 설정, split 만 VOBench split v2 의 block 으로):
  AttentiveClassifier (depth 1, query 1, head 16) · 35 epoch · batch 16 · lr 1e-4 OneCycle · wd 0.05 · 라벨 = (모양/색) 결합 + 없음 (empty)
  h      : train block 의 h (= LN(target_encoder(32 장))) 보이는 튜블릿 t0..15 (물체 + 빈 장면)
  p      : train block 의 보임 조건 (visible · early · mid) 릴리즈 p 미래 튜블릿 t8..15 중 보이는 것
  h_tX   : 같은 시점 판독기 — h 의 튜블릿 X 하나 (X = 8, 11, 15), 보이는 것, 모든 조건
  p_tX   : p 의 튜블릿 X 하나, 보임 조건
  시드 0, 1. objset (primitive · realistic) 따로.
평가 (test block, 고정 판독기): 원천 h · p (릴리즈) · pf (릴리즈 + PFT) × 칸 (보임 · 가림 · 조건 · 걸음).
late 안 ridge (phase ridge): late · 이동 · 물체 영상, 걸음마다 블록 평균 (g) · 참 자리 3×3 (l) 특징 → 6 지선다 모양, block 5 겹 one-hot ridge (λ 100)
  (predictor_late_identity.ridge_cv_acc 그대로). 원천: 릴리즈 p · PFT p · h (천장) · 복사 = LN(z)[t7] (마지막 문맥 튜블릿, late 에서는 가려짐)
  · LN(z)[마지막 보인 문맥 튜블릿].
출력: TG/heads/<objset>_<name>_s<seed>.pt · results/q1q2/E3_heads_<objset>.json (+ 표본 예측 npz) · results/q1q2/E3_ridge.json

  $P z_research/scripts/analysis/transfer_gap_heads.py heads --objset primitive --gpu 0
  $P z_research/scripts/analysis/transfer_gap_heads.py ridge
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_gap_common as C  # noqa: E402

NT, NF = C.NT, C.NF


def train_head(X, pairs, y, n_cls, dev, seed, src_off=0, epochs=None):
    import torch, torch.nn as nn
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    clf = C.make_head(n_cls, dev)
    o = C.OPT; ep_n = epochs or o["epochs"]; bs = o["batch_size"]
    opt = torch.optim.AdamW(clf.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    steps = ep_n * ((len(pairs) + bs - 1) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, o["lr"], total_steps=steps)
    yt = torch.tensor(y, device=dev)
    for ep in range(ep_n):
        clf.train(); perm = rng.permutation(len(pairs)); tot = 0.0
        for s0 in range(0, len(pairs), bs):
            sel = perm[s0:s0 + bs]
            xb = torch.from_numpy(np.stack([X[pairs[j][0]][pairs[j][1] - src_off] for j in sel]).astype(np.float32)).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = nn.functional.cross_entropy(clf(xb), yt[sel])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); tot += float(loss) * len(sel)
    clf.eval()
    return clf, tot / len(pairs)


def predict(clf, X, pairs, dev, off=0, fn=None, bs=64):
    import torch
    out = []
    with torch.no_grad():
        for s0 in range(0, len(pairs), bs):
            pp = pairs[s0:s0 + bs]
            x = torch.from_numpy(np.stack([X[i][t - off] for i, t in pp]).astype(np.float32)).to(dev)
            if fn is not None:
                x = fn(x, [t for _, t in pp], [i for i, _ in pp])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(clf(x).float().argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, int)


def heads(objset, gpu, seeds=(0, 1)):
    import torch
    dev = torch.device(f"cuda:{gpu}")
    M = C.meta(); trb = C.train_blocks()
    rows = [i for i, m in enumerate(M) if m["objset"] == objset]
    labs, L = C.classes(M, objset); NONE = L["empty"]
    st = {i: C.tub_state(M[i]["samples"]) for i in rows}
    empty = {i: C.label_of(M[i]) == "empty" for i in rows}
    late = {i: M[i]["occlusion"] == "late" for i in rows}
    tr_v = [i for i in rows if M[i]["block_id"] in trb]; te_v = [i for i in rows if M[i]["block_id"] not in trb]
    t0 = time.time()
    ridx = np.array(rows); pos = {i: j for j, i in enumerate(rows)}
    H = np.load(C.TUBE / "h.npy", mmap_mode="r"); P = np.load(C.TUBE / "p.npy", mmap_mode="r"); PF = np.load(C.TG / "p_relftg.npy", mmap_mode="r")
    Hs, Ps, PFs = np.asarray(H[ridx]), np.asarray(P[ridx]), np.asarray(PF[ridx])
    print(f"[{objset}] loaded {Hs.shape} {Ps.shape} {PFs.shape} {time.time() - t0:.0f}s · train videos {len(tr_v)} test {len(te_v)}", flush=True)
    lab = lambda i: L[C.label_of(M[i])]
    loc = lambda pairs: [(pos[i], t) for i, t in pairs]

    specs = {"h": ("h", [(i, t) for i in tr_v for t in range(NT) if st[i][t] == "vis"]),
             "p": ("p", [(i, t) for i in tr_v if not late[i] for t in range(NF, NT) if st[i][t] == "vis"])}
    for X in (8, 11, 15):
        specs[f"h_t{X}"] = ("h", [(i, X) for i in tr_v if st[i][X] == "vis"])
        specs[f"p_t{X}"] = ("p", [(i, X) for i in tr_v if not late[i] and st[i][X] == "vis"])

    # 평가 칸 (test block)
    obj_te = [i for i in te_v if not empty[i]]
    cats = {
        "h_vis_obj": ("h", [(i, t) for i in obj_te for t in range(NT) if st[i][t] == "vis"]),
        "h_vis_empty": ("h", [(i, t) for i in te_v if empty[i] for t in range(NT) if st[i][t] == "vis"]),
        "h_occ_obj": ("h", [(i, t) for i in obj_te for t in range(NT) if st[i][t] == "occ"]),
    }
    for src in ("p", "pf"):
        cats[f"{src}_vis_obj"] = (src, [(i, t) for i in obj_te if not late[i] for t in range(NF, NT) if st[i][t] == "vis"])
        cats[f"{src}_vis_empty"] = (src, [(i, t) for i in te_v if empty[i] and not late[i] for t in range(NF, NT)])
        cats[f"{src}_late_occ"] = (src, [(i, t) for i in obj_te if late[i] for t in range(NF, NT) if st[i][t] == "occ"])
        cats[f"{src}_late_reapp"] = (src, [(i, t) for i in obj_te if late[i] for t in range(NF, NT) if st[i][t] == "vis"])
        cats[f"{src}_late_empty"] = (src, [(i, t) for i in te_v if empty[i] and late[i] for t in range(NF, NT)])
    arr = {"h": (Hs, 0), "p": (Ps, NF), "pf": (PFs, NF)}

    res = dict(objset=objset, classes=labs, n_train_videos=len(tr_v), n_test_videos=len(te_v), heads={})
    C.HEADS.mkdir(parents=True, exist_ok=True); C.RES.mkdir(parents=True, exist_ok=True)
    samples = {}
    for seed in seeds:
        for name, (src, pairs) in specs.items():
            Xa, off = arr[src]
            y = np.array([lab(i) for i, _ in pairs])
            t1 = time.time()
            clf, loss = train_head(Xa, loc(pairs), y, len(labs), dev, seed, src_off=off)
            torch.save(clf.state_dict(), C.HEADS / f"{objset}_{name}_s{seed}.pt")
            hr = dict(n_train=len(pairs), final_loss=loss, train_s=time.time() - t1, cats={})
            tsel = None if not name.endswith(("t8", "t11", "t15")) else int(name.split("_t")[1])
            for cn, (csrc, cp) in cats.items():
                if tsel is not None:
                    cp = [(i, t) for i, t in cp if t == tsel]
                if not cp:
                    continue
                Xc, coff = arr[csrc]
                pred = predict(clf, Xc, loc(cp), dev, off=coff)
                gt = np.array([lab(i) for i, _ in cp])
                r = dict(n=len(cp), acc=100 * float((pred == gt).mean()), none=100 * float((pred == NONE).mean()))
                if (gt != NONE).all():
                    r["shape"] = 100 * float(np.mean([labs[a].split("/")[0] == labs[b].split("/")[0] for a, b in zip(pred, gt)]))
                    r["color"] = 100 * float(np.mean([labs[a].split("/")[-1] == labs[b].split("/")[-1] for a, b in zip(pred, gt)]))
                byt = defaultdict(list)
                for (i, t), a, b in zip(cp, pred, gt):
                    byt[t].append((a == b, a == NONE))
                r["by_t"] = {int(t): dict(n=len(v), acc=100 * float(np.mean([x for x, _ in v])), none=100 * float(np.mean([z for _, z in v])))
                             for t, v in sorted(byt.items())}
                hr["cats"][cn] = r
                samples[f"{name}_s{seed}|{cn}"] = np.array([[i, t, a, b] for (i, t), a, b in zip(cp, pred, gt)], np.int32)
            res["heads"][f"{name}_s{seed}"] = hr
            brief = {k: round(v["acc"], 1) for k, v in hr["cats"].items()}
            print(f"[{objset}] {name}_s{seed} n_train {len(pairs)} loss {loss:.3f} ({hr['train_s']:.0f}s) {brief}", flush=True)
            json.dump(res, open(C.RES / f"E3_heads_{objset}.json", "w"), indent=1, ensure_ascii=False)
    np.savez_compressed(C.RES / f"E3_samples_{objset}.npz", **samples)
    print(f"[{objset}] done {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ late 안 ridge
def ridge():
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("pli", Path(__file__).resolve().parent / "predictor_late_identity.py")
    pli = module_from_spec(spec); spec.loader.exec_module(pli)
    M = C.meta(); bmap, UV, HID, EXT = C.load_tracks()
    zm = json.load(open(C.TG / "z_late_meta.json")); zslot = {j: q for q, j in enumerate(zm["tp"])}
    H = np.load(C.TUBE / "h.npy", mmap_mode="r"); P = np.load(C.TUBE / "p.npy", mmap_mode="r")
    PF = np.load(C.TG / "p_relftg.npy", mmap_mode="r"); Z = np.load(C.TG / "z_late_ctx.npy", mmap_mode="r")
    sel = [j for j, m in enumerate(M) if m["occlusion"] == "late" and C.label_of(m) != "empty" and m["motion"] != "flat_static"]
    feats = defaultdict(list); meta_rows = []
    LN = lambda x: (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)
    t0 = time.time()
    for n_, j in enumerate(sel):
        m = M[j]; bi = bmap[m["block_id"]]; arm = 0 if m["variant"] == "pos_a" else 1
        uv, hid = UV[bi, arm], HID[bi, arm]
        lv = C.last_visible_ctx(hid)
        p = np.asarray(P[j], np.float32); pf = np.asarray(PF[j], np.float32); h = np.asarray(H[j], np.float32); z = np.asarray(Z[zslot[j]], np.float32)
        for k in range(NF):
            t = NF + k; loc = C.region(uv[t])
            for nm, x in (("p", p[k]), ("pf", pf[k]), ("pn", LN(p[k])), ("pfn", LN(pf[k])), ("h", h[t])):
                feats[f"{nm}_g{k}"].append(x.mean(0)); feats[f"{nm}_l{k}"].append(x[loc].mean(0))
        for nm, tt in (("z7", 7), ("zlv", lv)):
            loc = C.region(uv[tt]); feats[f"{nm}_g"].append(z[tt].mean(0)); feats[f"{nm}_l"].append(z[tt][loc].mean(0))
        loc = C.region(uv[lv]); feats["hlv_g"].append(h[lv].mean(0)); feats["hlv_l"].append(h[lv][loc].mean(0))
        meta_rows.append(m)
        if n_ % 50 == 0:
            print(f"ridge feats {n_}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
    F = {k: np.stack(v).astype(np.float64) for k, v in feats.items()}
    shape = np.array([C.label_of(m).split("/")[0] for m in meta_rows]); color = np.array([C.label_of(m).split("/")[-1] for m in meta_rows])
    objset = np.array([m["objset"] for m in meta_rows]); blk = np.array([m["block_id"] for m in meta_rows])
    out = {}
    for o in ("primitive", "realistic"):
        mk = objset == o
        cell = dict(n=int(mk.sum()), n_block=int(len(np.unique(blk[mk]))), chance_shape=float(1 / len(np.unique(shape[mk]))))
        for nm in ("p", "pf", "pn", "pfn", "h"):
            for kind in ("g", "l"):
                cell[f"{nm}_{kind}"] = [100 * pli.ridge_cv_acc(F[f"{nm}_{kind}{k}"][mk], shape[mk], blk[mk]) for k in range(NF)]
        for nm in ("z7", "zlv", "hlv"):
            for kind in ("g", "l"):
                cell[f"{nm}_{kind}"] = 100 * pli.ridge_cv_acc(F[f"{nm}_{kind}"][mk], shape[mk], blk[mk])
        if o == "primitive":
            for nm in ("p", "pf", "h"):
                cell[f"color_{nm}_g"] = [100 * pli.ridge_cv_acc(F[f"{nm}_g{k}"][mk], color[mk], blk[mk]) for k in range(NF)]
            cell["color_z7_g"] = 100 * pli.ridge_cv_acc(F["z7_g"][mk], color[mk], blk[mk])
            cell["chance_color"] = float(1 / len(np.unique(color[mk])))
        out[o] = cell
        print(o, {k: (np.round(v, 1).tolist() if isinstance(v, list) else v) for k, v in cell.items()}, flush=True)
    json.dump(dict(definition="late · 이동 · 물체 가능 영상, 걸음 k=0..7 (t8..15) 마다 block 5-fold one-hot ridge (λ 100) 모양 정확도 %. "
                              "p/pf = 릴리즈/PFT 원값, pn/pfn = 토큰 LN, h = 참 목표 encoder, g = 블록 평균, l = 참 자리 3×3 평균. "
                              "z7 = LN(문맥 encoder)[t7] (마지막 문맥 튜블릿 = 복사 판독), zlv = 마지막 보인 문맥 튜블릿, hlv = 같은 튜블릿 h.",
                   results=out), open(C.RES / "E3_ridge.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["heads", "ridge"])
    ap.add_argument("--objset", default="primitive")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args()
    if a.phase == "heads":
        heads(a.objset, a.gpu, tuple(int(s) for s in a.seeds.split(",")))
    else:
        ridge()
