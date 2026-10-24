#!/usr/bin/env python3
"""추적기가 못 푸는 판인가 — 2D 합성 사건 창 (진행자 2, 2026-10-22, PLAN §13 A 의 첫 실험). 학습 없음.

창: 2D 합성 상태 라벨판 v1 검증 세트 사건 표 (충돌 · 이탈), 시작 = 사건 장 − 18 → 문맥 16 장은 사건 전, 사건은 두 번째 미래 블록 첫 장.
채점: 사건 물체의 참 자리 3x3 칸에서 렌더 − 실제 미래 블록 토큰 오차. R = 1 − 오차(렌더) / 오차(복사), 사건 전 (첫 미래 블록) · 사건 뒤 (나머지 7).
렌더: 복사 · 문맥 평균 · 오라클 (참 궤적에 붙이기) · 추적 기준선 (라벨 추적 = 지각 완벽, 등속 · 2 차) · 모델 rollout (기준 블록 AR · sup_v1 · nosup · mix).
판 쓰기 규칙 (결과 전 19:37, 해시 사본 _scratch/prereg/PLAN_2026-10-22_event2d_1937.*): 사건 뒤 블록에서 추적 기준선 (라벨 · 2 차) R < 0.5 × 오라클 R.

  CUDA_VISIBLE_DEVICES=<g> python paper_lab/scripts/event2d.py prep --n 400
  CUDA_VISIBLE_DEVICES=<g> python paper_lab/scripts/event2d.py model --ckpt <path> --name <이름>
  python paper_lab/scripts/event2d.py score
산출: 캐시 /local_datasets/world/world_analysis/cache/paper_lab_event2d/ · 통계 paper_lab/_scratch/event2d/
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p_ in (str(REPO), str(REPO / "z_research/scripts/analysis"), str(REPO / "paper_lab/scripts"), str(REPO / "paper_lab/eval")):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
ROOT = Path("/local_datasets/world/training_data/phys2d_sup_v1/val")
CACHE = Path("/local_datasets/world/world_analysis/cache/paper_lab_event2d")
OUT = REPO / "paper_lab/_scratch/event2d"
G, C, K, NB, D, S = 16, 8, 8, 16, 1280, 256


def windows(n, seed=0):
    rows = list(csv.DictReader(open(ROOT / "boundary.tsv"), delimiter="\t"))
    ok, seen = [], set()
    for r in rows:
        if r["event"] not in ("impact", "leave"):
            continue
        ef, T = int(r["event_frame"]), int(r["video_frames"])
        st = ef - 18
        if st < 0 or st + 32 > T or (r["path"], ef) in seen:
            continue
        seen.add((r["path"], ef)); ok.append(dict(path=r["path"], ef=ef, st=st, event=r["event"], scenario=r["scenario"]))
    rng = np.random.default_rng(seed); rng.shuffle(ok)
    return ok[:n]


def labels(w):
    """사건 물체의 (P (32, 2) 512 척도, hid (32,)) 또는 None."""
    z = np.load(w["path"].replace(".mp4", "_states.npz"))
    ef, st = w["ef"], w["st"]
    f = np.arange(st, st + 32)
    # 사건 물체 = 창 안에 계속 있으면서 사건 전후 (±2 장) 속도 변화가 가장 큰 물체 (스모크에서 '충돌 세기 최대' 가 맞은 쪽 정지 물체를 고르는 것을 보고 결과 전에 바꿈)
    T = z["vel"].shape[0]
    a, b = max(ef - 2, 0), min(ef + 2, T - 1)
    dv = np.nan_to_num(np.linalg.norm(z["vel"][b] - z["vel"][a], axis=-1))
    pres = z["present"][f].all(0) & ~np.isnan(z["pos"][f]).any((0, 2))
    if not pres.any():
        return None
    dv[~pres] = -1
    j = int(np.argmax(dv))
    pos = z["pos"][f, j]
    if np.isnan(pos).any() or not z["present"][f, j].all():
        return None
    hid = (z["hidden"][f, j] | (z["vis_frac"][f, j] < 0.9)).astype(float)
    return pos.astype(np.float64) * 2.0, hid, j


def obj_cells(P):
    """미래 블록 k 마다 참 자리 3x3 칸 (8, 9, 2) 과 유효 (8, 9)."""
    cells = np.zeros((8, 9, 2), np.int64); valid = np.zeros((8, 9), bool)
    fut = P[16:].reshape(8, 2, 2).mean(1)
    for k in range(8):
        u, v = fut[k]
        cy, cx = int(v // 32), int(u // 32)
        i = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = cy + dy, cx + dx
                if 0 <= yy < G and 0 <= xx < G and 0 <= u < 512 and 0 <= v < 512:
                    cells[k, i] = (yy, xx); valid[k, i] = True
                i += 1
    return cells, valid


def paste_render(hblk_b, P, hid, posK):
    """m2_obsonly.render_moves 와 같은 규칙 (캔버스 지우기 + 마지막으로 다 보인 블록의 물체 3x3 을 posK (8, 2) 자리에)."""
    import design_headroom as H
    ctx_c = P[:16].reshape(8, 2, 2).mean(1)
    canvas = hblk_b[C - 1].clone()
    if hid[14] < 0.5 and hid[15] < 0.5:
        ry, rx = H.tok(ctx_c[C - 1])
        far = [bb for bb in range(C) if np.abs(np.array(H.tok(ctx_c[bb])) - np.array((ry, rx))).max() >= 3]
        if far:
            src = hblk_b[far[0]]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = ry + dy, rx + dx
                    if 0 <= yy < G and 0 <= xx < G:
                        canvas[yy, xx] = src[yy, xx]
    blk_vis = [b for b in range(C) if hid[2 * b] < 0.5 and hid[2 * b + 1] < 0.5]
    frames = canvas.unsqueeze(0).repeat(K, 1, 1, 1)
    if blk_vis:
        lv = blk_vis[-1]; oy, ox = H.tok(ctx_c[lv]); src = hblk_b[lv]
        for k in range(K):
            p = posK[k]
            if not (0 <= p[0] < 512 and 0 <= p[1] < 512):
                continue
            cy, cx = H.tok(p)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    sy, sx, ty, tx = oy + dy, ox + dx, cy + dy, cx + dx
                    if 0 <= sy < G and 0 <= sx < G and 0 <= ty < G and 0 <= tx < G:
                        frames[k, ty, tx] = src[sy, sx]
    return frames


def obj_err(R, futobj, cells, valid):
    """R (K, G, G, D) torch, futobj (K, 9, D) torch → (K,) 칸 평균 L1 (유효 칸만)."""
    import torch
    e = []
    for k in range(K):
        vk = valid[k]
        if not vk.any():
            e.append(float("nan")); continue
        dv = R.device
        rr = R[k][torch.as_tensor(cells[k, vk, 0], device=dv), torch.as_tensor(cells[k, vk, 1], device=dv)]
        e.append(float((rr.float() - futobj[k][torch.as_tensor(vk, device=futobj.device)].float().to(dv)).abs().mean()))
    return np.array(e)


def prep(n, bs):
    import torch, decord
    import passline as PL
    from analysis.intphys2.dataset import build_video_transform
    from app.vjepa_frozen.ar import encode_blocks
    import m2_obsonly as MO
    torch.set_num_threads(4)
    dev = torch.device("cuda", 0)
    _, bundle = PL._load_model("z_training/vast/ar_state/latest.pt", dev)
    tf_ = build_video_transform(256, 1.0)
    W = windows(n)
    CACHE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    keep, hctx, futobj, cells_all, valid_all, E = [], [], [], [], [], {m: [] for m in ("copy", "mean8", "oracle", "trk_cv", "trk_ca")}
    t0 = time.time(); buf = []

    def flush():
        if not buf:
            return
        x = torch.stack([b[1] for b in buf]).to(dev, dtype=bundle.dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hb = encode_blocks(bundle.target_encoder, x, 2).float().view(len(buf), NB, G, G, D)
        for b, (w, _, (P, hid, j)) in enumerate(buf):
            cells, valid = obj_cells(P)
            fut = hb[b, C:]
            fo = torch.stack([fut[k][torch.as_tensor(cells[k, :, 0], device=dev), torch.as_tensor(cells[k, :, 1], device=dev)] for k in range(K)])   # (K, 9, D)
            pl = MO.plan(P, hid)
            renders = {"copy": hb[b, C - 1].unsqueeze(0).expand(K, -1, -1, -1), "mean8": hb[b, :C].mean(0).unsqueeze(0).expand(K, -1, -1, -1),
                       "oracle": paste_render(hb[b], P, hid, P[16:].reshape(8, 2, 2).mean(1))}
            if pl is None:
                renders["trk_cv"] = renders["trk_ca"] = renders["copy"]
            else:
                renders["trk_cv"] = paste_render(hb[b], P, hid, pl[0]["move_cv"])
                renders["trk_ca"] = paste_render(hb[b], P, hid, pl[0]["move_ca"])
            for m, R in renders.items():
                E[m].append(obj_err(R, fo, cells, valid))
            keep.append(dict(w, obj=int(j))); hctx.append(hb[b, :C].half().cpu().numpy()); futobj.append(fo.half().cpu().numpy())
            cells_all.append(cells); valid_all.append(valid)
        buf.clear()
        print(f"[prep] {len(keep)} windows ({time.time() - t0:.0f}s)", flush=True)

    for w in W:
        lab = labels(w)
        if lab is None:
            continue
        try:
            vr = decord.VideoReader(w["path"], num_threads=1)
            x = tf_(vr.get_batch(list(range(w["st"], w["st"] + 32))).asnumpy())
        except Exception:
            continue
        buf.append((w, x, lab))
        if len(buf) == bs:
            flush()
    flush()
    np.save(CACHE / "hctx.npy", np.stack(hctx))
    np.savez(CACHE / "meta.npz", futobj=np.stack(futobj), cells=np.stack(cells_all), valid=np.stack(valid_all))
    json.dump(keep, open(OUT / "windows.json", "w"))
    np.savez(OUT / "E_data.npz", **{m: np.stack(v) for m, v in E.items()})
    print(f"[prep] done {len(keep)}", flush=True)


def model(ckpt, name, bs):
    import torch
    import passline as PL
    from app.vjepa_frozen.ar import block_indices
    torch.set_num_threads(4)
    dev = torch.device("cuda", 0)
    _, bundle = PL._load_model(ckpt, dev)
    pred = bundle.predictor
    Hc = np.load(CACHE / "hctx.npy", mmap_mode="r"); M = np.load(CACHE / "meta.npz")
    E = []
    for b0 in range(0, len(Hc), bs):
        h = torch.as_tensor(np.asarray(Hc[b0:b0 + bs]), device=dev).float()
        B = h.shape[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ar = pred.rollout(h.reshape(B, C * S, D).to(torch.bfloat16), block_indices(B, C, S, dev), K).float().view(B, K, G, G, D)
        for b in range(B):
            i = b0 + b
            E.append(obj_err(ar[b], torch.as_tensor(M["futobj"][i], device=dev), M["cells"][i], M["valid"][i]))
    np.save(OUT / f"E_{name}.npy", np.stack(E))
    print(f"[model {name}] {len(E)} windows", flush=True)


def score():
    W = json.load(open(OUT / "windows.json"))
    Ed = np.load(OUT / "E_data.npz")
    E = {m: Ed[m] for m in Ed.files}
    for f in sorted(OUT.glob("E_*.npy")):
        E[f.stem[2:]] = np.load(f)
    vids = np.array([w["path"] for w in W]); uv = sorted(set(vids)); vidx = {v: i for i, v in enumerate(uv)}
    gid = np.array([vidx[v] for v in vids])
    rng = np.random.default_rng(0); B = 1000
    picks = [rng.integers(0, len(uv), len(uv)) for _ in range(B)]
    members = [np.where(gid == g)[0] for g in range(len(uv))]

    def R(m, ks):
        e = np.nanmean(E[m][:, ks], 1); c = np.nanmean(E["copy"][:, ks], 1)
        ok = ~np.isnan(e) & ~np.isnan(c)
        pt = 1 - e[ok].sum() / c[ok].sum()
        bs = []
        for pk in picks:
            ii = np.concatenate([members[g] for g in pk]); ii = ii[ok[ii]]
            bs.append(1 - e[ii].sum() / c[ii].sum())
        return float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    res = {"n_windows": len(W), "n_videos": len(uv), "rows": {}}
    print(f"windows {len(W)} · videos {len(uv)}")
    print(f"{'render':12s} {'R pre-event (block 8)':>28s} {'R post-event (blocks 9-15)':>30s}")
    for m in ["copy", "mean8", "oracle", "trk_cv", "trk_ca"] + sorted(k for k in E if k not in ("copy", "mean8", "oracle", "trk_cv", "trk_ca")):
        pre, post = R(m, [0]), R(m, list(range(1, 8)))
        res["rows"][m] = dict(pre=pre, post=post)
        print(f"{m:12s} {pre[0]:+.3f} [{pre[1]:+.3f}, {pre[2]:+.3f}]      {post[0]:+.3f} [{post[1]:+.3f}, {post[2]:+.3f}]")
    o, t = res["rows"]["oracle"]["post"][0], res["rows"]["trk_ca"]["post"][0]
    res["usable"] = bool(t < 0.5 * o)
    print(f"판 쓰기 규칙: 추적 (라벨 · 2 차) R_post {t:+.3f} vs 0.5 × 오라클 {0.5 * o:+.3f} → {'추적기가 못 푸는 판' if res['usable'] else '추적기가 푼다 (판으로 못 씀)'}")
    by = {}
    for sc in sorted({w["scenario"] for w in W}):
        ii = [i for i, w in enumerate(W) if w["scenario"] == sc]
        row = {}
        for m in ("oracle", "trk_ca", "vast70", "sup_v1"):
            if m in E:
                e = np.nanmean(E[m][ii][:, 1:], 1); c = np.nanmean(E["copy"][ii][:, 1:], 1); ok = ~np.isnan(e) & ~np.isnan(c)
                row[m] = float(1 - e[ok].sum() / c[ok].sum()) if ok.any() else float("nan")
        by[sc] = dict(n=len(ii), **row)
    res["by_scenario_post"] = by
    json.dump(res, open(OUT / "score.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--n", type=int, default=400); p.add_argument("--bs", type=int, default=8)
    p = sub.add_parser("model"); p.add_argument("--ckpt", required=True); p.add_argument("--name", required=True); p.add_argument("--bs", type=int, default=8)
    sub.add_parser("score")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.n, a.bs)
    elif a.cmd == "model":
        model(a.ckpt, a.name, a.bs)
    else:
        score()
