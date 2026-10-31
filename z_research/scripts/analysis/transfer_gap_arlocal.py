#!/usr/bin/env python3
r"""[X1] R3 E34 — 국소 판독 검사를 AR predictor 에: late 예측의 정체가 참 자리에 몰리나, 화면에 퍼지나 (2026-10-30).

왜: [R] R2 local_id 는 릴리즈 late p 의 정체가 참 자리 = 대조 자리 (퍼짐), h · PFT p 는 참 자리 ≫ 대조 (몰림) 임을 보였다.
    관측만으로 학습한 AR predictor (vast AR ep70 · Ariel AR) 와 그 PFT 가 어느 꼴인지가 목표 문단의 첫 근거다.
추출 (phase extract, GPU 1, 모델마다): VOBench late · 이동 · 물체 있는 가능 영상 (tubeprobe 번호, local_id 와 같은 행).
  vobench_ar_token_transfer.py 와 같은 경로: h_blk = encode_blocks(target_encoder, 32 장, 2) (블록별 LN, 16 블록),
  p = rollout(predictor, h_blk 문맥 8 블록) 8 블록. autocast bfloat16. 저장 = p (n, 8, 256, 1280) fp16 · (--save_h) h_blk 미래 8 블록.
분석 (phase analyze, CPU): [R] R2 local_id 와 같은 판독기 (ridge 6 · 8 지선다, block 5 겹, 안쪽 3 겹으로 λ) · 같은 행
  (h 에서 물체 보이는 미래 걸음 t ≥ 8). 특징 = 토큰 LN 뒤 화면 평균 / 참 자리 3×3 / **대조 3 곳** (같은 행, 참 자리에서 ≥ 5 칸, 무작위,
  각각 판독기, 정확도 평균) / 마지막 본 자리 3×3. Δ = 참 자리 − 대조 평균 (pt). CI = block bootstrap 2,000 (행별 정답 차의 block 평균).
  표현: 릴리즈 p · h (32 장 LN) · 릴리즈 PFT p (캐시 bf16) / AR 판: h_blk (블록별 encoder 참 미래) · 모델별 p.
  ⚠️ AR 은 encoder 공간이 다르다 (블록별 LN(target_encoder)) — 같은 모델 (같은 공간) 안의 참 자리 − 대조 만 비교한다.
판정 (BOARD [X1] R3 사전 등록, 모델마다): 퍼짐 = 세 과제 모두 CI 가 0 을 포함하거나 |Δ| < 5 · 몰림 = 둘 이상에서 Δ ≥ +10 & CI 하한 > 0 · 그 사이 약한 국소 성분.
출력  /data2/local_datasets/world/world_analysis/cache/transfer_gap/E34/{p_<model>.npy, h_blk.npy, list.json}
      z_analysis/transfer_gap/results/q1q2/E34_arlocal.json

  PY=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python
  $PY z_research/scripts/analysis/transfer_gap_arlocal.py extract --model vast70 --save_h [--limit 8]
  $PY z_research/scripts/analysis/transfer_gap_arlocal.py extract --model vast70_pft
  $PY z_research/scripts/analysis/transfer_gap_arlocal.py extract --model ariel_ar38
  $PY z_research/scripts/analysis/transfer_gap_arlocal.py analyze
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import transfer_gap_common as C  # noqa: E402

ARR = C.TG / "E34"
OUT = C.RES
MODELS = {"vast70": "z_training/vast/ar_state/latest.pt",
          "vast70_pft": "z_training/runs/vobench_vast_ar_state_postft_g/latest.pt",
          "ariel_ar38": "z_training/ariel_eval/ckpt/ar_ep38.pt"}
S, T, D = 256, 8, 1280


def clip_list():
    M = C.meta(); out = []
    for i, m in enumerate(M):
        if m["occlusion"] != "late" or m["motion"] == "flat_static" or C.label_of(m) == "empty":
            continue
        out.append(i)
    return out


def extract(model_name, gpu, limit, save_h):
    import torch, yaml
    sys.path.insert(0, str(REPO))
    from analysis.intphys2.model import build_from_config
    from analysis.predictors.ar_scoring import rollout
    from app.vjepa_frozen.ar import block_indices, encode_blocks
    from evals.world_model_analysis.data import WMADataset
    from evals.world_model_analysis.attn_probe import _ClipDS
    torch.set_num_threads(6); dev = torch.device(f"cuda:{gpu}"); torch.cuda.set_device(dev)
    cfg = yaml.safe_load(open(REPO / "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/_resolved.yaml"))
    cfg["features"] = {"decode_workers": 6}
    model = dict(cfg["model"], predictor_checkpoint=str(REPO / MODELS[model_name]))
    bundle = build_from_config(model, dev)
    ds = WMADataset(dict(cfg, model=model)); vid2rec = {r.video_id: i for i, r in enumerate(ds.records)}
    M = C.meta(); tp = clip_list()[: limit or None]
    ARR.mkdir(parents=True, exist_ok=True)
    suf = "_smoke" if limit else ""
    json.dump(dict(tp=tp, video_id=[M[j]["video_id"] for j in tp]), open(ARR / f"list{suf}.json", "w"))
    P = np.lib.format.open_memmap(ARR / f"p_{model_name}{suf}.npy", "w+", np.float16, (len(tp), T, S, D))
    H = np.lib.format.open_memmap(ARR / f"h_blk{suf}.npy", "w+", np.float16, (len(tp), T, S, D)) if save_h else None
    loader = torch.utils.data.DataLoader(_ClipDS(dict(cfg, model=model), [vid2rec[M[j]["video_id"]] for j in tp]), batch_size=8, shuffle=False,
                                         num_workers=5, pin_memory=True)
    w, t0 = 0, time.time()
    with torch.no_grad():
        for x in loader:
            B = x.size(0); x = x.to(dev, dtype=bundle.dtype, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hb = encode_blocks(bundle.target_encoder, x, 2)
                idx = block_indices(B, 2 * T, S, dev)
                p = rollout(bundle.predictor, hb[:, :T * S], idx[:, :T * S], T)
            P[w:w + B] = p.float().view(B, T, S, D).half().cpu().numpy()
            if H is not None:
                H[w:w + B] = hb[:, T * S:].float().view(B, T, S, D).half().cpu().numpy()
            w += B
            if w % 80 < 8:
                print(f"{model_name} {w}/{len(tp)} {time.time() - t0:.0f}s", flush=True)
    P.flush()
    if H is not None:
        H.flush()
    (ARR / f"DONE_{model_name}{suf}").write_text("ok")
    print(f"done {model_name} {len(tp)} in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- 분석

def LN(x):
    x = x.astype(np.float32); mu = x.mean(-1, keepdims=True); sd = x.std(-1, keepdims=True); return (x - mu) / np.sqrt(sd ** 2 + 1e-5)


def box(r, c):
    r = min(max(r, 1), 14); c = min(max(c, 1), 14); return [rr * 16 + cc for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)]


def ridge_cv_pred(X, y, g, lams=(1, 10, 100, 1e3, 1e4), k=5, seed=0):
    """[R] R2 local_id 의 ridge_cv 와 같은 절차 — 행별 정답 여부를 돌려준다."""
    cls = sorted(set(y)); Y = np.array([[1.0 if a == c else 0.0 for c in cls] for a in y]); yi = np.array([cls.index(a) for a in y])
    ub = np.unique(g); rng = np.random.default_rng(seed); rng.shuffle(ub); folds = np.array_split(ub, k); pred = np.zeros(len(y), int)
    for f in folds:
        te = np.isin(g, f); trn = ~te; Xt = X[trn]; mu = Xt.mean(0); sd = Xt.std(0) + 1e-6; Xs = (Xt - mu) / sd; Xe = (X[te] - mu) / sd
        inner = np.array_split(rng.permutation(np.unique(g[trn])), 3); best = None
        for lam in lams:
            acc = []
            for fi in inner:
                v = np.isin(g[trn], fi); A = Xs[~v]; W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y[trn][~v]); acc.append(((Xs[v] @ W).argmax(1) == yi[trn][v]).mean())
            if best is None or np.mean(acc) > best[0]:
                best = (np.mean(acc), lam)
        W = np.linalg.solve(Xs.T @ Xs + best[1] * np.eye(Xs.shape[1]), Xs.T @ Y[trn]); pred[te] = (Xe @ W).argmax(1)
    return (pred == yi).astype(np.float64), len(cls)


def block_boot(d, g, n=2000, seed=0):
    ub = np.unique(g); bm = np.array([d[g == b].mean() for b in ub]); w = np.array([(g == b).sum() for b in ub])
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        ix = rng.integers(0, len(ub), len(ub)); vals.append((bm[ix] * w[ix]).sum() / w[ix].sum())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def analyze(min_t=8, n_ctrl=3):
    M = C.meta(); tr = np.load(C.TG / "tracks.npz"); bmap = {b: i for i, b in enumerate(tr["block_id"])}
    L = json.load(open(ARR / "list.json")); pos = {tp: q for q, tp in enumerate(L["tp"])}
    srcs = {"rel_p": ("tube_p", None), "h": ("tube_h", None), "rel_pft_p": ("pft", None), "ar_h_blk": ("arr", ARR / "h_blk.npy")}
    for mn in MODELS:
        if (ARR / f"p_{mn}.npy").exists() and (ARR / f"DONE_{mn}").exists():
            srcs[f"{mn}_p"] = ("arr", ARR / f"p_{mn}.npy")
    TP = np.load(C.TUBE / "p.npy", mmap_mode="r"); TH = np.load(C.TUBE / "h.npy", mmap_mode="r"); PF = np.load(C.TG / "p_relftg.npy", mmap_mode="r")
    AR = {k: np.load(v[1], mmap_mode="r") for k, v in srcs.items() if v[0] == "arr" and v[1].exists()}
    srcs = {k: v for k, v in srcs.items() if v[0] != "arr" or k in AR}
    cell = lambda uv: (int(np.clip(uv[1] // 32.0, 0, 15)), int(np.clip(uv[0] // 32.0, 0, 15)))  # noqa: E731
    rows = []; F = defaultdict(list); rng = np.random.default_rng(7)
    for i in L["tp"]:
        m = M[i]; lab = C.label_of(m)
        bi = bmap[m["block_id"]]; arm = 0 if m["variant"] == "pos_a" else 1
        uv = tr["uv"][bi, arm]; hid = tr["hidden"][bi, arm]
        vis = [t for t in range(8) if not hid[t]]; tl = vis[-1] if vis else 0
        steps = [t for t in range(min_t, 16) if not hid[t]]
        if not steps:
            continue
        X = {}
        for k, (kind, _) in srcs.items():
            if kind == "tube_p":
                X[k] = np.asarray(TP[i])
            elif kind == "tube_h":
                X[k] = np.asarray(TH[i, 8:])
            elif kind == "pft":
                X[k] = np.asarray(PF[i])
            else:
                X[k] = np.asarray(AR[k][pos[i]])
        rl, cl = cell(uv[tl])
        for t in range(min_t, 16):
            if hid[t]:
                continue
            r, c = cell(uv[t])
            elig = [cc for cc in range(1, 15) if abs(cc - c) >= 5]
            ctrls = list(rng.choice(elig, size=min(n_ctrl, len(elig)), replace=False))
            idx = {"true": box(r, c), "last": box(rl, cl)}
            for q, cc in enumerate(ctrls):
                idx[f"ctrl{q}"] = box(r, cc)
            for k, Xk in X.items():
                Xl = LN(Xk[t - 8]); F[k + "|avg"].append(Xl.mean(0))
                for nm, ix in idx.items():
                    F[k + "|" + nm].append(Xl[ix].mean(0))
            rows.append(dict(block=m["block_id"], objset=m["objset"], shape=lab.split("/")[0], color=lab.split("/")[-1], t=t, n_ctrl=len(ctrls)))
    F = {k: np.stack(v) for k, v in F.items()}
    g = np.array([r["block"] for r in rows])
    out = {"definition": __doc__.split("분석 (phase analyze, CPU):")[1].split("출력")[0].strip(), "n_rows": len(rows), "min_t": min_t, "n_ctrl": n_ctrl,
           "sources": list(srcs.keys()), "tasks": {}}
    for obj, attr in (("primitive", "shape"), ("realistic", "shape"), ("primitive", "color")):
        s = np.array([r["objset"] == obj for r in rows]); y = [r[attr] for r, ok in zip(rows, s) if ok]; gs = g[s]
        res = {"n": int(s.sum()), "n_block": int(len(np.unique(gs))), "chance": round(100 / len(set(y)), 1)}
        for src in srcs:
            cor = {}
            for nm in ["avg", "true", "last"] + [f"ctrl{q}" for q in range(n_ctrl)]:
                cor[nm], _ = ridge_cv_pred(F[f"{src}|{nm}"][s], y, gs)
            ctrl = np.mean([cor[f"ctrl{q}"] for q in range(n_ctrl)], 0)
            d = cor["true"] - ctrl
            e = dict(avg=round(cor["avg"].mean() * 100, 1), true=round(cor["true"].mean() * 100, 1), ctrl_mean=round(ctrl.mean() * 100, 1),
                     ctrl_each=[round(cor[f"ctrl{q}"].mean() * 100, 1) for q in range(n_ctrl)], last=round(cor["last"].mean() * 100, 1),
                     delta_true_minus_ctrl=round(d.mean() * 100, 1), delta_CI=[round(v * 100, 1) for v in block_boot(d, gs)],
                     delta_true_minus_last=round((cor["true"] - cor["last"]).mean() * 100, 1))
            res[src] = e; print(obj, attr, src, e, flush=True)
        # 섞은 라벨 대조 (block 단위) — 화면 평균 특징, 대표로 릴리즈 p 와 AR 첫 모델
        rng2 = np.random.default_rng(1); ub = np.unique(gs); perm = dict(zip(ub, rng2.permutation(ub))); lb = {}
        for r, ok in zip(rows, s):
            if ok:
                lb.setdefault(r["block"], r[attr])
        ysh = [lb[perm[r["block"]]] for r, ok in zip(rows, s) if ok]
        res["shuffled_avg"] = {src: round(ridge_cv_pred(F[f"{src}|avg"][s], ysh, gs)[0].mean() * 100, 1) for src in srcs if src in ("rel_p", "vast70_p", "ar_h_blk")}
        out["tasks"][f"{obj}|{attr}"] = res
    ver = {}
    for src in srcs:
        ds = [out["tasks"][t][src] for t in out["tasks"]]
        spread = all((e["delta_CI"][0] <= 0 <= e["delta_CI"][1]) or abs(e["delta_true_minus_ctrl"]) < 5 for e in ds)
        conc = sum(1 for e in ds if e["delta_true_minus_ctrl"] >= 10 and e["delta_CI"][0] > 0) >= 2
        ver[src] = "몰림 (h · PFT 꼴)" if conc else "퍼짐 (릴리즈 꼴)" if spread else "약한 국소 성분"
    out["verdict"] = dict(rule="퍼짐 = 세 과제 모두 CI 가 0 포함 또는 |Δ| < 5 · 몰림 = 둘 이상에서 Δ ≥ +10 & CI 하한 > 0 · 그 사이 약한 국소 성분", by_source=ver)
    print(json.dumps(out["verdict"], ensure_ascii=False), flush=True)
    json.dump(out, open(OUT / "E34_arlocal.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract", "analyze"])
    ap.add_argument("--model", default="vast70")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_h", action="store_true")
    a = ap.parse_args()
    if a.phase == "extract":
        extract(a.model, a.gpu, a.limit, a.save_h)
    else:
        analyze()
