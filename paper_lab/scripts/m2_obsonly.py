#!/usr/bin/env python3
"""M2-t 관측만 판 하한 = '물체 추적 + 2 차 외삽' 기준선 (진행자 2, 2026-10-22, PLAN §11-2 의 미리 정한 다음 수).

하한 A (m2_lowerbound.py) 는 문맥의 **라벨 위치** 를 썼다 (지각 완벽 가정). 여기서는 라벨 없이 **문맥 프레임만** 으로 물체를 추적한다.
  추적기 (픽셀, 학습 없음): 문맥 16 장의 화소별 중앙값 = 배경 → 장마다 배경과 다른 화소의 연결 성분 → 첫 장의 가장 큰 성분 = 물체
  (색 · 면적 본) → 장마다 색 · 면적 · 예측 자리에 가장 맞는 성분, 면적이 본의 60 % 미만이면 '가려짐'. 미래 프레임은 보지 않는다.
  렌더 (design_headroom 의 move_cv · move_ca 와 같은 규칙, 위치만 추적값): 마지막 문맥 블록에 물체가 보이면 그 3x3 을 물체가 ≥ 3 칸
  떨어져 있던 블록의 같은 칸으로 지우고, 마지막으로 다 보인 블록의 물체 3x3 토큰을 외삽 자리 (등속 · 2 차) 에 붙인다.
  같은 실행에서 라벨 판 (move_cv_lab · move_ca_lab) 도 다시 만들어 맞댄다.
채점 = m2_lowerbound.py 와 같다 (R2 자리 여유 · 먼 칸 48 px, 기준 블록 AR = 하네스 vast70).

  python paper_lab/scripts/m2_obsonly.py track                       # CPU: 추적 + 추적 오차 (라벨 대비, 진단용)
  CUDA_VISIBLE_DEVICES=<g> python paper_lab/scripts/m2_obsonly.py render   # GPU: 블록 토큰 → 렌더 → 칸별 오차 지도
  python paper_lab/scripts/m2_obsonly.py score                       # CPU: 표 + 미리 정한 판정
산출: paper_lab/_scratch/m2/obs_tracks.json · obs_render.npz · obs_score.json
"""
import argparse, csv, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p_ in (str(REPO), str(REPO / "z_research/scripts/analysis"), str(REPO / "paper_lab/scripts"), str(REPO / "paper_lab/eval")):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
OUT = REPO / "paper_lab/_scratch/m2"
G, C, K, NB, D, S = 16, 8, 8, 16, 1280, 256
IMNET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1)
IMNET_STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1)
FG_THR, MIN_AREA, VIS_FRAC, COL_THR = 0.12, 12, 0.6, 0.25   # 결과 전에 고정 (2026-10-22 18:04). 화소 값 [0, 1], 256 화면


def _ds():
    import yaml
    import design_headroom as H
    from evals.world_model_analysis.data import WMADataset
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir)
    cfg = yaml.safe_load(open(H.SETS["vobench"]["cfg"]))
    model = dict(ir.rp.MODEL)
    ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=model, features={"cache_dir": "/tmp"}, surprise={}))
    rows = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
    return H, ds, rows, model


def blocks_of(rows, occ_mot=(("late", True), ("late", False))):
    blk = defaultdict(dict)
    for v, r in rows.items():
        blk[r["source_block"]][r["variant"]] = v
    out = []
    for sb, arms in sorted(blk.items()):
        if not all(a in arms for a in ("pos_a", "pos_b", "imp_ab", "imp_ba")):
            continue
        r = rows[arms["pos_a"]]
        if (r["occlusion"], r["motion"] != "flat_static") in occ_mot:
            out.append((sb, arms))
    return out


def track_pixels(x):
    """x (3, 32, 256, 256) 정규화 tensor → (P (16, 2) 512 라벨 척도, hid (16,)) — 문맥 16 장만 쓴다."""
    import scipy.ndimage as ndi
    a = (x[:, :16].cpu().float().numpy() * IMNET_STD + IMNET_MEAN).clip(0, 1)          # (3, 16, H, W)
    bg = np.median(a, axis=1)                                                           # (3, H, W)
    P = np.full((16, 2), np.nan); hid = np.ones(16)
    tmpl = None; last = None; vel = np.zeros(2)
    for t in range(16):
        diff = np.abs(a[:, t] - bg).max(0)
        lab, n = ndi.label(diff > FG_THR)
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        area = ndi.sum(np.ones_like(diff), lab, idx)
        cy, cx = np.array(ndi.center_of_mass(np.ones_like(diff), lab, idx)).T
        col = np.stack([ndi.mean(a[c, t], lab, idx) for c in range(3)], 1)              # (n, 3)
        ok = area >= MIN_AREA
        if tmpl is None:                                                                # 본: 처음 보이는 가장 큰 성분
            if not ok.any():
                continue
            j = int(np.argmax(np.where(ok, area, -1)))
            tmpl = dict(col=col[j], area=float(area[j]))
        else:
            cd = np.linalg.norm(col - tmpl["col"][None], axis=1)
            pred = last + vel if last is not None else None
            dist = np.hypot(cx - pred[0], cy - pred[1]) / 64.0 if pred is not None else np.zeros(n)
            score = cd + 0.5 * np.abs(np.log(np.maximum(area, 1) / tmpl["area"])) + dist
            score[~ok | (cd > COL_THR)] = np.inf
            if not np.isfinite(score).any():
                continue
            j = int(np.argmin(score))
        if area[j] < VIS_FRAC * tmpl["area"]:                                           # 거의 다 가려짐 → 가려짐으로 (중심이 치우친다)
            continue
        pos = np.array([cx[j], cy[j]])
        if last is not None:
            vel = pos - last
        last = pos
        P[t] = pos * 2.0                                                                # 256 화면 → 512 라벨 척도
        hid[t] = 0.0
    return P, hid


def track():
    H, ds, rows, _ = _ds()
    idx = {r.video_id: i for i, r in enumerate(ds.records)}
    res, errs, agree = {}, [], []
    t0 = time.time()
    for sb, arms in blocks_of(rows):
        for pos, arm in (("pos_a", "A"), ("pos_b", "B")):
            v = arms[pos]
            x = ds.clip(idx[v])
            P, hid = track_pixels(x)
            res[v] = dict(P=[[None if np.isnan(c) else float(c) for c in p] for p in P], hid=hid.tolist())
            t = H.track_vobench(rows[v])
            if t is not None:
                Pl, hl = t
                for s in range(16):
                    agree.append(float((hl[s] >= 0.5) == (hid[s] >= 0.5)))
                    if hl[s] < 0.5 and hid[s] < 0.5:
                        errs.append(float(np.linalg.norm(Pl[s] - P[s])) / 2.0)          # 256 화면 px
        if len(res) % 40 == 0:
            print(f"[track] {len(res)} contexts ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    diag = dict(n_contexts=len(res), median_err_px256=float(np.median(errs)) if errs else None, p90_err_px256=float(np.percentile(errs, 90)) if errs else None,
                hidden_agreement=float(np.mean(agree)) if agree else None, params=dict(FG_THR=FG_THR, MIN_AREA=MIN_AREA, VIS_FRAC=VIS_FRAC, COL_THR=COL_THR))
    json.dump(dict(diag=diag, tracks=res), open(OUT / "obs_tracks.json", "w"))
    print(json.dumps(diag, indent=1))


def plan(P, hid):
    """design_headroom.plan_positions 와 같은 규칙 (cv = 마지막 두 '보인' 샘플, ca = 보인 샘플 최대 8 개 2 차 맞춤). P 의 미래 칸은 안 쓴다."""
    vis = [s for s in range(16) if hid[s] < 0.5]
    fut = np.arange(16, 32)
    if len(vis) >= 2:
        s1, s0 = vis[-1], vis[-2]; v = (P[s1] - P[s0]) / (s1 - s0)
        cv = (P[s1][None] + v[None] * (fut - s1)[:, None]).reshape(8, 2, 2).mean(1)
        w = vis[-8:]
        if len(w) >= 3:
            t = np.array(w, float); A = np.stack([np.ones_like(t), t, t ** 2], 1)
            coef = np.linalg.lstsq(A, P[w], rcond=None)[0]
            ca = (np.stack([np.ones(16), fut, fut ** 2], 1) @ coef).reshape(8, 2, 2).mean(1)
        else:
            ca = cv
    elif vis:
        cv = ca = np.repeat(P[vis[-1]][None], 8, 0)
    else:
        return None
    blk_vis = [b for b in range(C) if hid[2 * b] < 0.5 and hid[2 * b + 1] < 0.5]
    ctx_c = np.array([np.nanmean(P[2 * b:2 * b + 2], 0) if not np.isnan(P[2 * b:2 * b + 2]).all() else [np.nan, np.nan] for b in range(C)])
    return dict(move_cv=cv, move_ca=ca), (blk_vis[-1] if blk_vis else None), ctx_c


def render_moves(hblk_b, P, hid, H):
    """hblk_b (16, G, G, D) torch → {move_cv, move_ca: (K, G, G, D)} — design_headroom 의 렌더와 같은 규칙, 위치는 (P, hid) 에서."""
    out = plan(P, hid)
    copy = hblk_b[C - 1].unsqueeze(0).expand(K, -1, -1, -1)
    if out is None:
        return {"move_cv": copy, "move_ca": copy}
    pos, last_vis, ctx_c = out
    canvas = hblk_b[C - 1].clone()
    if hid[14] < 0.5 and hid[15] < 0.5 and not np.isnan(ctx_c[C - 1]).any():
        ry, rx = H.tok(ctx_c[C - 1])
        far = [bb for bb in range(C) if not np.isnan(ctx_c[bb]).any() and np.abs(np.array(H.tok(ctx_c[bb])) - np.array((ry, rx))).max() >= 3]
        if far:
            src = hblk_b[far[0]]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = ry + dy, rx + dx
                    if 0 <= yy < G and 0 <= xx < G:
                        canvas[yy, xx] = src[yy, xx]
    res = {}
    for m in ("move_cv", "move_ca"):
        frames = canvas.unsqueeze(0).repeat(K, 1, 1, 1)
        if last_vis is not None and not np.isnan(ctx_c[last_vis]).any():
            oy, ox = H.tok(ctx_c[last_vis]); src = hblk_b[last_vis]
            for k in range(K):
                p = pos[m][k]
                if not (0 <= p[0] < 512 and 0 <= p[1] < 512):
                    continue
                cy, cx = H.tok(p)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        sy, sx, ty, tx = oy + dy, ox + dx, cy + dy, cx + dx
                        if 0 <= sy < G and 0 <= sx < G and 0 <= ty < G and 0 <= tx < G:
                            frames[k, ty, tx] = src[sy, sx]
        res[m] = frames
    return res


def render():
    import torch
    import passline as PL
    from app.vjepa_frozen.ar import encode_blocks
    torch.set_num_threads(4)
    H, ds, rows, model = _ds()
    dev = torch.device("cuda", 0)
    _, bundle = PL._load_model("z_training/vast/ar_state/latest.pt", dev)          # 표적 encoder 만 쓴다
    tr = json.load(open(OUT / "obs_tracks.json"))["tracks"]
    idx = {r.video_id: i for i, r in enumerate(ds.records)}
    rec = defaultdict(list); t0 = time.time()
    order = ("pos_a", "imp_ab", "pos_b", "imp_ba")
    ctx_of = {"pos_a": "pos_a", "imp_ab": "pos_a", "pos_b": "pos_b", "imp_ba": "pos_b"}
    for j, (sb, arms) in enumerate(blocks_of(rows)):
        x = torch.stack([ds.clip(idx[arms[v]]) for v in order]).to(dev, dtype=bundle.dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hb = encode_blocks(bundle.target_encoder, x, 2).float().view(4, NB, G, G, D)
        rend = {}
        for v in ("pos_a", "pos_b"):                                                     # 문맥은 pos 와 그 짝 imp 가 같다 (matched)
            ci = order.index(v)
            o = tr[arms[v]]
            Po = np.array([[np.nan if c is None else c for c in p] for p in o["P"]]); ho = np.array(o["hid"])
            Po_full = np.vstack([Po, np.full((16, 2), np.nan)])
            ho_full = np.concatenate([ho, np.ones(16)])
            r_obs = render_moves(hb[ci], Po_full, ho_full, H)
            t = H.track_vobench(rows[arms[v]])
            r_lab = render_moves(hb[ci], *t, H) if t is not None else {m: hb[ci, C - 1].unsqueeze(0).expand(K, -1, -1, -1) for m in ("move_cv", "move_ca")}
            rend[v] = {"obs_cv": r_obs["move_cv"], "obs_ca": r_obs["move_ca"], "lab_cv": r_lab["move_cv"], "lab_ca": r_lab["move_ca"]}
        for ci, v in enumerate(order):
            fut = hb[ci, C:]
            for m, R in rend[ctx_of[v]].items():
                rec[f"E_{m}"].append((R.float() - fut).abs().mean(-1).half().cpu().numpy())
            rec["video_id"].append(arms[v])
        if j % 20 == 0:
            print(f"[render] {j + 1} blocks ({time.time() - t0:.0f}s)", flush=True)
    np.savez(OUT / "obs_render.npz", video_id=np.array(rec.pop("video_id")), **{k: np.stack(v) for k, v in rec.items()})
    print(f"[render] done ({time.time() - t0:.0f}s)", flush=True)


def score():
    import m2_lowerbound as LB
    P = LB.pairs("late", True)
    Ps = LB.pairs("late", False)
    z = np.load(OUT / "obs_render.npz", allow_pickle=True)
    obs = ({v: i for i, v in enumerate(z["video_id"])}, {k: z[k] for k in z.files if k.startswith("E_")})
    base = LB.load_harness("vast70")
    res = {"diag": json.load(open(OUT / "obs_tracks.json"))["diag"], "rows": {}}
    print("tracking diag:", res["diag"])
    for cellname, PP in (("late_moving", P), ("late_static", Ps)):
        for far in ((False, True) if cellname == "late_moving" else (False,)):
            tag = f"{cellname}|{'far48' if far else 'all'}"
            mo = LB.margin(base, "oracle", PP, far); scale = np.nanmean(np.abs(mo))
            mb = LB.margin(base, "model", PP, far)
            print(f"\n== {tag}: oracle scale {scale:.4f}")
            for lab, src, rnd in (("copy", base, "copy"), ("ctx-mean8", base, "mean8"), ("oracle", base, "oracle"), ("base block AR", base, "model"),
                                  ("label track + const-vel", obs, "lab_cv"), ("label track + quadratic", obs, "lab_ca"),
                                  ("observed track + const-vel", obs, "obs_cv"), ("observed track + quadratic", obs, "obs_ca")):
                m = LB.margin(src, rnd, PP, far)
                d = (m - mb) / scale
                mean, lo, hi, n = LB.boot_diff(d, PP)
                share = np.nanmean(m) / scale
                print(f"  {lab:28s} acc {LB.acc(m):5.1f}  margin share {100 * share:+6.1f} %   vs base AR {100 * mean:+6.1f} % [{100 * lo:+6.1f}, {100 * hi:+6.1f}]  n {n}")
                res["rows"][f"{tag}|{lab}"] = dict(acc=LB.acc(m), margin_share=float(share), diff_vs_base=[float(mean), float(lo), float(hi)], n=n)
    json.dump(res, open(OUT / "obs_score.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["track", "render", "score"])
    a = ap.parse_args()
    {"track": track, "render": render, "score": score}[a.phase]()
