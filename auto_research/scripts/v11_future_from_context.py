#!/usr/bin/env python3
"""IntPhysGen v11 **late 가림** — 과거 16 frames 만 본 문맥 encoder (z) 위의 작은 자로 미래 8 슬롯 (x, y) 를 회귀한다 (2026-10-02, 사용자 지시).
"predictor 는 안 되는 조건 (문맥 끝 k 샘플 가림 → p 미래에 물체 토큰 없음)" 에서 encoder 과거 표현에 미래 위치가 읽히는가.

대상: `data_csv/intphysgen_v11_vanish_all/index_probe.csv` 의 visible 3 조건 (k=0, 672) + late 가림 3 조건 (k=1..4, 672) — vanish pos_a (물체 있음). 1,344 clip.
표현 (셋 다 같은 창 · 같은 자 · 같은 라벨):  z = ctx_masked (문맥 16 샘플만, LN, 8 튜블릿 × 256) / p = predictor 미래 8 튜블릿 / h = target encoder 32 샘플의 미래 8 튜블릿 (상한).
자: rollout2_future_from_context 의 HeadA (튜블릿별 N 풀링 쿼리 1, 3,842 파라미터) · HeadB (슬롯별 쿼리 8, 12,802).
라벨: metadata `object_px_{x,y}_by_sample` 32 샘플 → 미래 슬롯 j = 샘플 (16+2j, 17+2j) 평균 /144 − 1.  가려진 샘플 = hidden_start..hidden_end (raw) → late k 는 샘플 16−k..16+k−1.
실험:  fit_vis = visible 블록 절반으로 학습 → visible 나머지 절반 + late k=1..4 전부에 test (가림을 본 적 없는 자의 이식)
        fit_all = 전체 블록 절반 (condition × k 층화) 학습 → 나머지 절반, k 별 (가림 문맥에서 정보가 있기는 한가)
기준선 (라벨만): CV = 마지막 **보인** 문맥 슬롯 둘로 외삽 / COPY = 마지막 보인 슬롯 고정 / TEMPLATE = condition(motion) 별 train 평균 궤적.
지표: 슬롯별 유클리드 오차 px (288 px 화면), (condition, k) 별, 미래 슬롯을 '가려진 (j < ⌈k/2⌉)' 과 '보이는' 으로 나눠서. block bootstrap CI.

  python v11_future_from_context.py extract --frames <IntPhysGen_v11 dir> --out /dev/shm/v11_ffc [--bs 4]      (GPU 1 장)
  python v11_future_from_context.py probe --feat /dev/shm/v11_ffc --out <dir> [--device cuda --epochs 60]
"""
import argparse, csv, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "z_research/scripts/analysis")); sys.path.insert(0, str(ROOT / "auto_research/scripts"))
INDEX = ROOT / "data_csv/intphysgen_v11_vanish_all/index_probe.csv"; META = ROOT / "auto_research/_stage/v11_metadata.csv"
VIS = ["static_visible", "moving_visible_flat", "moving_visible"]; OCC = ["static_occlusion", "moving_occlusion_flat", "moving_occlusion"]
NS, CS, S, D, RES, W = 32, 16, 256, 1280, 144.0, 288; T = (NS - CS) // 2
arr = lambda s: np.array([float(v) for v in re.split(r"[;,\s|]+", str(s).strip()) if v], np.float32)


def select():
    csv.field_size_limit(10 ** 9)
    rows = [r for r in csv.DictReader(INDEX.open()) if r["plausible"] == "1" and r["condition"] in VIS + OCC and r["occ_timing"] in ("", "late")]
    meta = {m["name"]: m for m in csv.DictReader(META.open())}; n = len(rows)
    L = np.zeros((n, NS, 2), np.float32); hid = np.zeros((n, NS), bool)
    for i, r in enumerate(rows):
        m = meta[r["video_id"]]; L[i, :, 0], L[i, :, 1] = arr(m["object_px_x_by_sample"]), arr(m["object_px_y_by_sample"])
        try:
            hs, he = float(m["hidden_start"]), float(m["hidden_end"])
            if he >= hs: hid[i] = [hs <= 3 * t <= he for t in range(NS)]
        except (KeyError, ValueError): pass
        k = int(r["sym_k"])
        if k > 0:
            want = np.zeros(NS, bool); want[CS - k:CS + k] = True; assert (hid[i] == want).all(), (r["video_id"], np.where(hid[i])[0])
    inf = (L[..., 0] >= 0) & (L[..., 0] < W) & (L[..., 1] >= 0) & (L[..., 1] < W)
    P = L.reshape(n, NS // 2, 2, 2).mean(2) / RES - 1                                   # (n, 16 슬롯, 2) 정규화
    inf_s = inf.reshape(n, NS // 2, 2).all(2); hid_s = hid.reshape(n, NS // 2, 2).any(2)  # 슬롯: 둘 다 화면 안 / 하나라도 가려짐
    info = dict(vid=np.array([r["video_id"] for r in rows]), block=np.array([r["block_id"] for r in rows]), cond=np.array([r["condition"] for r in rows]),
                k=np.array([int(r["sym_k"]) for r in rows]), motion=np.array([r["motion"] for r in rows]))
    return rows, P, inf_s, hid_s, info


def extract(a):
    from analysis.intphys2.model import build_from_config
    from evals.world_model_analysis.data import WMADataset
    from evals.analysis_vlm.occlusion_identity.forward import extract_batch
    from rollout3_window_readout import MODEL, prefetch
    rows, P, inf_s, hid_s, info = select(); n = len(rows); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with (out / "index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    DATA = dict(root=str(out), index_csv="index.csv", frames_root=a.frames, frames_pattern="{file_name}/{frame:06d}.png", frames_start=0, frames_stride=3, n_frames=NS, resolution=256)
    dev = torch.device("cuda:0"); bundle = build_from_config({**MODEL, "window_size": NS}, dev)
    ds = WMADataset(dict(data=DATA, model={**MODEL, "window_size": NS}, features={"cache_dir": "/tmp"}, surprise={}))
    mm = {r: np.lib.format.open_memmap(out / f"{r}.npy", mode="w+", dtype=np.float16, shape=(n, T * S, D)) for r in ("z", "p", "h")}
    ln = lambda x: F.layer_norm(x.float(), (x.size(-1),)); pool = ThreadPoolExecutor(max_workers=6); t0 = time.time(); done = 0
    with torch.no_grad():
        for ids, clips in prefetch(ds, list(range(n)), a.bs, pool):
            ids = np.asarray(ids); clips = clips.to(dev, dtype=bundle.dtype, non_blocking=True)
            o = extract_batch(clips, bundle, [{"base": "ctx_masked"}, {"base": "predictor"}], context_length=CS, mask_index=0, out_dtype=torch.float16)
            mm["z"][ids] = o["ctx_masked"].cpu().numpy(); mm["p"][ids] = o["predictor"].cpu().numpy()
            fo = bundle.target_encoder(clips); fo = fo[-1] if isinstance(fo, list) else fo
            mm["h"][ids] = ln(fo)[:, T * S:].half().cpu().numpy()
            done += len(ids)
            if done % 64 < a.bs: print(f"  {done}/{n} {time.time()-t0:.0f}s", flush=True)
    for m in mm.values(): m.flush()
    np.savez(out / "labels.npz", P=P, inf=inf_s, hid=hid_s, **info); print("extracted", n, out, flush=True)


def baselines(P, inf, hid, train_m, cond):
    n = len(P); ctx = P[:, :T]; fut = P[:, T:]; cv = np.zeros_like(fut); copy = np.zeros_like(fut)
    for i in range(n):
        vis = np.where(~hid[i, :T])[0]; l1, l0 = vis[-1], vis[-2]; v = (ctx[i, l1] - ctx[i, l0]) / (l1 - l0)
        kk = np.arange(T) + (T - l1); cv[i] = ctx[i, l1] + v[None] * kk[:, None]; copy[i] = ctx[i, l1]
    tpl = np.zeros_like(fut)
    for c in np.unique(cond): m = train_m & (cond == c); tpl[cond == c] = fut[m].mean(0) if m.any() else fut[train_m].mean(0)
    return dict(CV=cv, COPY=copy, TEMPLATE=tpl)


def cell_table(pred, fut, inf, hid, info, sel, rng, B=300):
    """(condition, k) 별 유클리드 오차 px: 전체 / 가려진 미래 슬롯 / 보이는 미래 슬롯 (+ 슬롯별), block bootstrap CI (전체)."""
    e = np.linalg.norm(pred - fut, axis=-1) * RES; out = {}
    hf = hid[:, T:]; ok = inf[:, T:]
    for c in VIS + OCC:
        for k in sorted(set(info["k"][sel & (info["cond"] == c)])):
            m = sel & (info["cond"] == c) & (info["k"] == k); em, okm, hm, bl = e[m], ok[m], hf[m], info["block"][m]
            def mean_of(mask, rows=None):
                mk = mask if rows is None else mask[rows]; ee = em if rows is None else em[rows]; return float(ee[mk].mean()) if mk.any() else None
            ub = np.unique(bl); ix = {u: np.where(bl == u)[0] for u in ub}
            bs = [mean_of(okm, np.concatenate([ix[u] for u in rng.choice(ub, len(ub))])) for _ in range(B)]
            out[f"{c}|k{k}"] = dict(n=int(m.sum()), all=mean_of(okm), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], hidden=mean_of(okm & hm), visible=mean_of(okm & ~hm),
                                     slot=[float(em[:, j][okm[:, j]].mean()) if okm[:, j].any() else None for j in range(T)], n_hidden_slots=int((okm & hm).sum()))
    return out


def probe(a):
    from rollout2_future_from_context import HeadA, HeadB
    feat = Path(a.feat); Lz = np.load(feat / "labels.npz", allow_pickle=True); P, inf, hid = Lz["P"], Lz["inf"], Lz["hid"]; info = {k: Lz[k] for k in ("vid", "block", "cond", "k", "motion")}
    n = len(P); fut = P[:, T:]; dev = torch.device(a.device); rng = np.random.default_rng(0); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # 블록 절반 split (condition × k 층화)
    half = np.zeros(n, bool)
    for key in sorted(set(zip(info["cond"], info["k"]))):
        m = (info["cond"] == key[0]) & (info["k"] == key[1]); ub = np.unique(info["block"][m]); rng.shuffle(ub); tr_b = set(ub[: len(ub) // 2]); half[m] = np.array([b in tr_b for b in info["block"][m]])
    is_vis = info["k"] == 0
    exps = {"fit_vis": (half & is_vis, ~half | ~is_vis), "fit_all": (half, ~half)}            # (train mask, test mask)
    Y = torch.from_numpy(fut).to(dev); Wt = torch.from_numpy((inf[:, T:] & ~hid[:, T:]).astype(np.float32)).to(dev)   # 학습 손실: 화면 안 · **안 가려진** 미래 슬롯만 (가려진 슬롯은 test 에서만 본다)
    res = dict(n=int(n), epochs=a.epochs, experiments={})
    for ename, (trm, tem) in exps.items():
        R = {"n_train": int(trm.sum()), "n_test": int(tem.sum()), "rows": {}}
        for bn, bp in baselines(P, inf, hid, trm, info["cond"]).items(): R["rows"][bn] = cell_table(bp, fut, inf, hid, info, tem, rng)
        for rep in ("z", "p", "h"):
            X = torch.from_numpy(np.ascontiguousarray(np.load(feat / f"{rep}.npy", mmap_mode="r"))).to(dev)          # (n, 2048, D) fp16, 7 GB
            for hn, cls in (("A", HeadA), ("B", HeadB)):
                torch.manual_seed(0); head = cls().to(dev); opt = torch.optim.Adam(head.parameters(), lr=a.lr); tri = np.where(trm)[0]
                for ep in range(a.epochs):
                    perm = tri[rng.permutation(len(tri))]
                    for i in range(0, len(perm), a.bs):
                        b = torch.from_numpy(perm[i:i + a.bs]).to(dev); pred, _ = head(X[b].float()); w = Wt[b]
                        loss = ((pred - Y[b]).abs().sum(-1) * w).sum() / w.sum().clamp(min=1); opt.zero_grad(); loss.backward(); opt.step()
                head.eval()
                with torch.no_grad(): pr = torch.cat([head(X[i:i + 128].float())[0] for i in range(0, n, 128)]).cpu().numpy()
                R["rows"][f"{rep}-{hn}"] = cell_table(pr, fut, inf, hid, info, tem, rng); np.save(out / f"pred_{ename}_{rep}{hn}.npy", pr)
                print(f"[{ename}] {rep}-{hn} " + " ".join(f"{c.split('|')[0][:14]}|{c.split('|')[1]} {v['all']:.1f}" for c, v in R["rows"][f"{rep}-{hn}"].items()), flush=True)
            del X; torch.cuda.empty_cache() if dev.type == "cuda" else None
        res["experiments"][ename] = R
    np.savez(out / "labels.npz", P=P, inf=inf, hid=hid, half=half, **info); json.dump(res, open(out / "result.json", "w"), indent=1); print("saved", out / "result.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--frames", required=True); e.add_argument("--out", required=True); e.add_argument("--bs", type=int, default=4)
    p = sub.add_parser("probe"); p.add_argument("--feat", required=True); p.add_argument("--out", required=True); p.add_argument("--device", default="cuda"); p.add_argument("--epochs", type=int, default=60); p.add_argument("--bs", type=int, default=32); p.add_argument("--lr", type=float, default=1e-3)
    s = sub.add_parser("plan")
    a = ap.parse_args()
    if a.cmd == "extract": extract(a)
    elif a.cmd == "probe": probe(a)
    else:
        rows, P, inf, hid, info = select(); import collections; print(len(rows), collections.Counter(zip(info["cond"], info["k"]))); print("hidden future slots by k:", {k: int(hid[info["k"] == k][:, T:].sum(1).mean()) for k in range(5)}, "in_frame future", inf[:, T:].mean().round(3))
