#!/usr/bin/env python3
"""고리 1 [E] — nuisance 와 신호는 시공간 일관성으로 갈리나 (auto_research, 2026-10-21).

  /venv/vjepa/bin/python auto_research/scripts/enc_noise_coherence.py

이웃 칸 (가로 · 세로 1 칸) 두 벡터의 상관 = E[u·v] / sqrt(E|u|² E|v|²) (표본 평균), 소스별:
  nuisance  ±1/255 반복 차 (h(x+n1) − h(x+n2)) · 정지 칸 블록 간 차 (|Δ픽셀| < 1 인 이웃 두 칸)
  변화 신호  둘 다 크게 바뀐 이웃 두 칸 (≥ 16 단계) 의 블록 간 차
  장면 신호  다른 창 같은 칸 차 (h_i − h_j), 이웃 두 칸
  (phys3d 캐시) 물체 대비: 정지 물체 칸 토큰 − 그 칸 배경 (물체 없던 블록 평균), 이웃 두 물체 칸 · 운반 변화: 물체가 움직인 쌍의 이웃 두 물체 칸 h_{t+1} − h_t
시간 일관성: 같은 칸 이웃 블록 (b, b+1) 두 벡터의 상관 — nuisance 는 ±1/255 반복 차 (블록마다 다른 잡음), 정지 칸 잔차 (h_b − 창 평균)."""
import glob, json, os
import numpy as np
G = 16
rng = np.random.default_rng(0)


def corr(U, V):
    U = np.concatenate(U); V = np.concatenate(V)
    return float((U * V).sum(-1).mean() / np.sqrt((U * U).sum(-1).mean() * (V * V).sum(-1).mean())), len(U)


def pairs_h(mask):                      # mask (G,G) bool → 이웃 쌍 인덱스
    a_, b_ = [], []
    m = mask
    for dy, dx in ((0, 1), (1, 0)):
        y, x = np.where(m[:G - dy, :G - dx] & m[dy:, dx:])
        a_.append(y * G + x); b_.append((y + dy) * G + (x + dx))
    return np.concatenate(a_), np.concatenate(b_)


RES = {}
D = "/workspace/auto_research_cache/enc_noise"
import sys
LABELS_ONLY = "--labels_only" in sys.argv
if LABELS_ONLY:
    _prev = json.load(open(os.path.join(os.path.dirname(__file__), "../results/enc_noise_coherence.json")))
    RES.update({k: v for k, v in _prev.items() if k != "phys3d_labels"})
for src in (() if LABELS_ONLY else ("phys3d", "vobench", "k710", "ssv2", "ht100m", "ek100")):
    tf = sorted(glob.glob(f"{D}/{src}_target_encoder_s*_tok.npy"))
    if not tf:
        continue
    acc = {k: ([], []) for k in ("nuis_rep", "nuis_static", "change", "scene", "time_rep", "time_static_resid")}
    prev = None
    for f in tf:
        T = np.load(f, mmap_mode="r"); Z = np.load(f[:-8] + ".npz")
        for i in range(0, T.shape[0], 2):
            t = np.asarray(T[i, :4], np.float32); h0, hf, n1, n2 = t
            pixD = Z["pixD"][i]
            dn = n1 - n2
            ia, ib = pairs_h(np.ones((G, G), bool))
            sel = rng.choice(len(ia), 64, replace=False)
            for b in range(0, h0.shape[0], 3):
                acc["nuis_rep"][0].append(dn[b, ia[sel]]); acc["nuis_rep"][1].append(dn[b, ib[sel]])
            acc["time_rep"][0].append(dn[:-1, :32].reshape(-1, dn.shape[-1])); acc["time_rep"][1].append(dn[1:, :32].reshape(-1, dn.shape[-1]))
            dh = h0[1:] - h0[:-1]
            for b in range(dh.shape[0]):
                st = (pixD[b] < 1).reshape(G, G); mv = (pixD[b] >= 16).reshape(G, G)
                for key, m in (("nuis_static", st), ("change", mv)):
                    pa, pb = pairs_h(m)
                    if len(pa):
                        s = rng.choice(len(pa), min(16, len(pa)), replace=False)
                        acc[key][0].append(dh[b, pa[s]]); acc[key][1].append(dh[b, pb[s]])
            allst = (pixD < 2).all(0)
            if allst.sum():
                r = h0[:, allst] - h0[:, allst].mean(0, keepdims=True)
                acc["time_static_resid"][0].append(r[:-1].reshape(-1, r.shape[-1])[:64]); acc["time_static_resid"][1].append(r[1:].reshape(-1, r.shape[-1])[:64])
            if prev is not None:
                d = h0[0] - prev
                s = rng.choice(len(ia), 64, replace=False)
                acc["scene"][0].append(d[ia[s]]); acc["scene"][1].append(d[ib[s]])
            prev = h0[0]
    RES[src] = {k: corr(*v) for k, v in acc.items() if v[0]}
    print(src, " · ".join(f"{k} {v[0]:+.3f} (n {v[1]})" for k, v in RES[src].items()), flush=True)

# phys3d 캐시: 물체 신호
P3 = "/workspace/auto_research_cache/phys3d"
def cover(bbox):
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return oy[..., :, None] * ox[..., None, :]
def dil(m):
    o = m.copy(); o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
    return o
acc = {k: ([], []) for k in ("obj_contrast", "transport_change", "obj_contrast_time")}
for f in sorted(glob.glob(f"{P3}/h_val_target_s*.npy")):
    sh = f.split("_s")[-1].split(".")[0]
    lab = np.load(f"{P3}/lab_val_s{sh}.npz"); H = np.load(f, mmap_mode="r")
    for w in range(H.shape[0]):
        cv = cover(np.nan_to_num(lab["bbox"][w], nan=-100.0))
        okf = lab["inframe_frac"][w] > 0.9
        visb = (lab["vis_frac"][w][0::2] > 0.7) & (lab["vis_frac"][w][1::2] > 0.7) & okf[0::2] & okf[1::2]
        ins = (np.minimum(cv[0::2], cv[1::2]) > 0.6) & visb[:, None, None]             # 보임 조건 (2026-10-21 정정)
        touch = (cv[0::2] > 0) | (cv[1::2] > 0)
        free = ~np.stack([dil(touch[b]) for b in range(16)])
        h = None
        pos = lab["pos"][w]
        for b in range(16):
            pa, pb = pairs_h(ins[b])
            if not len(pa):
                continue
            if h is None:
                h = np.asarray(H[w], np.float32)
            nf = free.reshape(16, -1)
            okp = (nf[:, pa].sum(0) >= 3) & (nf[:, pb].sum(0) >= 3)
            for x, y in zip(pa[okp][:4], pb[okp][:4]):
                bgx = h[nf[:, x], x].mean(0); bgy = h[nf[:, y], y].mean(0)
                acc["obj_contrast"][0].append((h[b, x] - bgx)[None]); acc["obj_contrast"][1].append((h[b, y] - bgy)[None])
            if b < 15:
                d = pos[2 * b + 3] - pos[2 * b + 1]
                if np.isfinite(d).all() and np.linalg.norm(d) > 8:
                    acc["transport_change"][0].append(h[b + 1, pa[:4]] - h[b, pa[:4]]); acc["transport_change"][1].append(h[b + 1, pb[:4]] - h[b, pb[:4]])
                # 같은 칸 · 이웃 블록 물체 대비 (정지 물체: 두 블록 모두 안쪽)
                if np.isfinite(d).all() and np.linalg.norm(d) < 1:
                    cells = np.where(ins[b].reshape(-1) & ins[b + 1].reshape(-1) & (nf.sum(0) >= 3))[0][:4]
                    for c in cells:
                        bg = h[nf[:, c], c].mean(0)
                        acc["obj_contrast_time"][0].append((h[b, c] - bg)[None]); acc["obj_contrast_time"][1].append((h[b + 1, c] - bg)[None])
RES["phys3d_labels"] = {k: corr(*v) for k, v in acc.items() if v[0]}
print("phys3d_labels", " · ".join(f"{k} {v[0]:+.3f} (n {v[1]})" for k, v in RES["phys3d_labels"].items()))
json.dump(RES, open(os.path.join(os.path.dirname(__file__), "../results/enc_noise_coherence.json"), "w"), indent=1)
