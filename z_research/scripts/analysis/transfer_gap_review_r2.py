#!/usr/bin/env python3
"""[R] R2 적대 검증 — 새 측정 셋 (2026-10-30, CPU, 학습 없음 · GPU 없음).

  track_prior : E32 예측 물체 자리 (존재 대비 지도 3×3 최대 상자) 를 '방향 + 학습셋 전형 끝 자리' 기준선과 비교.
                기준선 = 문맥 h 자리로 정한 운동 방향 (앞 3 · 뒤 3 튜블릿 열 중앙값 부호) 이 같은 다른 block 의 t15 참 자리 중앙값 (한 개 빼기).
                끌림 = 참 끝 자리가 VOBench (보폭 3) 방향별 전형 끝 자리에서 ≥ 2 칸 떨어진 block 에서, 예측이 전형 끝 자리 쪽에 더 가까운 비율.
  flip_null   : E31c 뒤집힘 몰림을 '원래 정답 / 오답 쌍의 뒤집힘 비율' 로 다시 검정 (50:50 귀무 대신 정답 여부 조건부, Fisher).
  local_id    : 릴리즈 late p 의 정체가 참 자리에 국소로 있나. late 안에서 학습한 ridge (h 판독기 이식 없음),
                특징 = LN 토큰의 화면 평균 / 참 자리 3×3 / 대조 3×3 (같은 행, 참 자리에서 ≥ 5 칸) / 마지막 본 자리 3×3.
                block 5 겹, 섞은 라벨 대조. 같은 판을 h (참 미래) · PFT p 에.
출력: z_analysis/transfer_gap/results/review_r2/*.json
  $P z_research/scripts/analysis/transfer_gap_review_r2.py track_prior
  $P z_research/scripts/analysis/transfer_gap_review_r2.py flip_null
  $P z_research/scripts/analysis/transfer_gap_review_r2.py local_id late 8      # (visible 11 · early,mid 11 도)
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "z_analysis/transfer_gap/results/review_r2"
TG = Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap")
TUBE = "/local_datasets/world/world_analysis/cache/vobench_v1_vith_tubeprobe"
ARR = TG / "E32"


def loc(E):
    G = E.reshape(*E.shape[:-1], 16, 16)
    box = sum(G[..., i:i + 14, j:j + 14] for i in range(3) for j in range(3))
    flat = box.reshape(*box.shape[:-2], -1); k = flat.argmax(-1)
    return k // 14 + 1, k % 14 + 1


def w1(a, b, c, d):
    return np.maximum(np.abs(a - c), np.abs(b - d)) <= 1


def _late(sub, name):
    Z = np.load(ARR / (f"{sub}/{name}.npz" if sub else f"{name}.npz")); meta = json.load(open(ARR / f"{name}_meta.json"))
    rh, ch = loc(Z["Eh"]); ix = np.array([i for i, m in enumerate(meta) if m["occ"] == "late"])
    d = np.sign(np.median(ch[ix, 3:6], 1) - np.median(ch[ix, 0:3], 1))
    P = {k: tuple(a[ix, 7] for a in loc(Z[k])) for k in Z.keys() if k != "Eh"}
    return rh[ix, 15], ch[ix, 15], d, P


def track_prior():
    Tr, Tc, d, _ = _late(None, "vob")
    typ = {s: (float(np.median(Tr[d == s])), float(np.median(Tc[d == s]))) for s in (-1, 1) if (d == s).any()}
    res = {"typical_end_vob_late_by_dir(row,col)": {str(k): v for k, v in typ.items()}, "sets": {}}
    for sub, name in [(None, "vob"), (None, "vob_s2"), (None, "v11sub"), (None, "v11real"), ("v11pft", "vob"), ("v11pft", "v11sub")]:
        Tr, Tc, d, P = _late(sub, name)
        hit = []
        for j in range(len(Tr)):
            s = d == d[j]; s[j] = False
            hit.append(bool(s.any() and w1(np.median(Tr[s]), np.median(Tc[s]), Tr[j], Tc[j])))
        o = {"n": int(len(Tr)), "dir_prior_within1": round(float(np.mean(hit)), 3), "true_t15_col_hist": Counter(Tc.tolist()).most_common(6)}
        tr_ = np.array([typ.get(s, (np.nan, np.nan))[0] for s in d]); tc_ = np.array([typ.get(s, (np.nan, np.nan))[1] for s in d])
        far = np.hypot(Tr - tr_, Tc - tc_) >= 2
        for k, (pr, pc) in P.items():
            dt = np.hypot(pr - Tr, pc - Tc); dy = np.hypot(pr - tr_, pc - tc_)
            o[k] = {"within1_true": round(float(w1(pr, pc, Tr, Tc).mean()), 3), "t15_col_hist": Counter(pc.tolist()).most_common(6),
                    "n_far": int(far.sum()),
                    "far_within1_true": round(float(w1(pr, pc, Tr, Tc)[far].mean()), 3) if far.any() else None,
                    "far_closer_to_typical_end": round(float((dy[far] < dt[far]).mean()), 3) if far.any() else None}
        if name == "vob_s2" and not sub:
            right = d > 0; pc = P["Ef"][1]
            o["s2_right_truth_col_le11"] = int((Tc[right] <= 11).sum()); o["s2_right_pft_col_ge12_when_truth_le11"] = int(((pc[right] >= 12) & (Tc[right] <= 11)).sum())
        res["sets"][f"{sub or 'main'}:{name}"] = o; print(sub or "main", name, json.dumps(o, ensure_ascii=False), flush=True)
    res["note"] = "Er = 릴리즈, Ef = 그 폴더의 PFT (main = VOBench PFT, v11pft = v11 PFT). 기준선 · 끌림은 사후 검사 (PLAN_R2 에 없음)."
    json.dump(res, open(OUT / "track_prior.json", "w"), indent=1, ensure_ascii=False)


def flip_null():
    from scipy.stats import fisher_exact
    d = json.load(open(REPO / "z_analysis/transfer_gap/results/q3_r2/E31c_symnoise.json")); res = {}
    for sec, models in [("intphys1_skip2_w32", ["release", "vob_pft"]), ("vobench_test", ["release", "vobftg", "copy"])]:
        V = d[sec]["versions"]; n = d[sec]["n_pairs"]
        for m in models:
            nc = round(V["orig"][m] / 100 * n); nw = n - nc
            L = sum(V[f"sr{s}"]["flips_" + m]["lost"] for s in range(1, 5)); G = sum(V[f"sr{s}"]["flips_" + m]["gained"] for s in range(1, 5))
            pool = (L + G) / (4 * n)
            res[f"{sec}|{m}"] = dict(orig_correct=nc, orig_wrong=nw, lost=L, gained=G, flip_rate_correct=round(L / (4 * nc), 3), flip_rate_wrong=round(G / (4 * nw), 3),
                                    expected_lost_if_uniform=round(pool * 4 * nc, 1), expected_gained_if_uniform=round(pool * 4 * nw, 1),
                                    fisher_p=float(fisher_exact([[L, 4 * nc - L], [G, 4 * nw - G]])[1]))
            print(sec, m, res[f"{sec}|{m}"], flush=True)
    res["note"] = "seed 4 개를 모은 쌍 × seed 단위 (같은 쌍이 여러 seed 에서 뒤집히면 독립이 아님 → p 는 낙관적)."
    json.dump(res, open(OUT / "flip_null.json", "w"), indent=1)


def local_id(which="late", min_t=8):
    M = json.load(open(TUBE + "/meta.json"))["meta"]
    tr = np.load(TG / "tracks.npz"); bmap = {b: i for i, b in enumerate(tr["block_id"])}
    P = np.load(TUBE + "/p.npy", mmap_mode="r"); H = np.load(TUBE + "/h.npy", mmap_mode="r"); PF = np.load(TG / "p_relftg.npy", mmap_mode="r")
    cell = lambda uv: (int(np.clip(uv[1] // 32.0, 0, 15)), int(np.clip(uv[0] // 32.0, 0, 15)))  # noqa: E731

    def box(r, c):
        r = min(max(r, 1), 14); c = min(max(c, 1), 14); return [rr * 16 + cc for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)]

    def LN(x):
        x = x.astype(np.float32); mu = x.mean(-1, keepdims=True); sd = x.std(-1, keepdims=True); return (x - mu) / np.sqrt(sd ** 2 + 1e-5)
    rows = []; F = defaultdict(list)
    for i, m in enumerate(M):
        if m["occlusion"] not in which.split(",") or m["motion"] == "flat_static":
            continue
        lab = m["obj_a"] if m["variant"] == "pos_a" else m["obj_b"]
        if lab == "empty":
            continue
        bi = bmap[m["block_id"]]; arm = 0 if m["variant"] == "pos_a" else 1
        uv = tr["uv"][bi, arm]; hid = tr["hidden"][bi, arm]
        vis = [t for t in range(8) if not hid[t]]; tl = vis[-1] if vis else 0
        steps = [t for t in range(min_t, 16) if not hid[t]]
        if not steps:
            continue
        p = np.asarray(P[i]); h = np.asarray(H[i, 8:]); pf = np.asarray(PF[i]); rl, cl = cell(uv[tl])
        for t in steps:
            r, c = cell(uv[t]); cc = 15 - c if abs((15 - c) - c) >= 5 else (c + 7 if c + 7 <= 14 else c - 7)
            idx = {"true": box(r, c), "ctrl": box(r, cc), "last": box(rl, cl)}
            for nm, X in (("rel", p[t - 8]), ("h", h[t - 8]), ("pft", pf[t - 8])):
                Xl = LN(X); F[nm + "|avg"].append(Xl.mean(0))
                for k, ix in idx.items():
                    F[nm + "|" + k].append(Xl[ix].mean(0))
            rows.append(dict(block=m["block_id"], objset=m["objset"], shape=lab.split("/")[0], color=lab.split("/")[-1]))
    F = {k: np.stack(v) for k, v in F.items()}

    def ridge_cv(X, y, g, lams=(1, 10, 100, 1e3, 1e4), k=5, seed=0):
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
        return float((pred == yi).mean()), len(cls)
    out = {"which": which, "min_t": min_t, "n_rows": len(rows)}; g = np.array([r["block"] for r in rows])
    for obj, attr in (("primitive", "shape"), ("realistic", "shape"), ("primitive", "color")):
        s = np.array([r["objset"] == obj for r in rows]); y = [r[attr] for r, ok in zip(rows, s) if ok]; res = {}
        for k in F:
            res[k] = round(ridge_cv(F[k][s], y, g[s])[0] * 100, 1)
        rng = np.random.default_rng(1); ub = np.unique(g[s]); perm = dict(zip(ub, rng.permutation(ub))); lb = {}
        for r, ok in zip(rows, s):
            if ok:
                lb.setdefault(r["block"], r[attr])
        res["rel|avg_shuffled"] = round(ridge_cv(F["rel|avg"][s], [lb[perm[r["block"]]] for r, ok in zip(rows, s) if ok], g[s])[0] * 100, 1)
        res["chance"] = round(100 / len(set(y)), 1); res["n"] = int(s.sum()); res["n_block"] = int(len(np.unique(g[s])))
        out[f"{obj}|{attr}"] = res; print(obj, attr, res, flush=True)
    json.dump(out, open(OUT / f"local_id_{which.replace(',', '+')}_t{min_t}.json", "w"), indent=1)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "track_prior":
        track_prior()
    elif cmd == "flip_null":
        flip_null()
    elif cmd == "local_id":
        local_id(sys.argv[2] if len(sys.argv) > 2 else "late", int(sys.argv[3]) if len(sys.argv) > 3 else 8)
