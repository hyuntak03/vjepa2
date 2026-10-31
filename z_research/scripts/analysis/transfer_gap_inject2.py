#!/usr/bin/env python3
r"""[X1] R2 E24 주입 격자 (신탁 없는 자리) + late 문맥 encoder 속도 판독 — H2-4 · H2-5 · PFT 크기 몫 vs 자리 몫 (2026-10-30).

기호: p = 릴리즈 predictor 원값 (t8–15), p̃ = LN(p), z = LN(context_encoder(문맥 16 장)) t0–7, h = LN(target_encoder(32 장)). 채점 = 표준 (전체 토큰 L1, matched 짝).
t끝 = 문맥 안 마지막으로 보인 튜블릿 (label). 칸 이동 = 원천 중심 칸 → 목표 중심 칸 평행 이동 (R1 E7 d3 과 같은 구성).
내용 셋 (주입할 토큰):
  z  = z[t끝] 의 물체 칸 (외곽 상자 + 1 칸)                                   (R1 d2 · d3 의 내용)
  p1 = 릴리즈 p 가 O_p(t) 에 둔 자기 토큰 (3×3)
  p2 = p1 의 대비를 2γ 배: μ^p + 2γ_t (p̃ − μ^p)  (μ^p · γ_t = E4 맞춤값, 보임 조건 train block)
자리 넷 (튜블릿 t 마다 목표 중심):
  O끝  = t끝 의 물체 중심 칸 (마지막으로 본 자리)
  O_p  = 릴리즈 출력의 대비 에너지 최대 칸 (permanence = 존재 대비 LN(p_물체문맥) − LN(p_빈문맥), 모양 = 두 팔 정체 대비) — 출력만 씀
  O등속 = 마지막으로 보인 두 문맥 튜블릿 중심 (label) 의 등속 외삽: c(t) = u(t끝) + (u(t끝) − u(t끝′))·(t − t끝)/(t끝 − t끝′) — 관측 가능한 정보만
  O등가속 = 마지막으로 보인 문맥 튜블릿 셋의 중심에 맞춘 2 차식 외삽 (PLAN 밖 추가 — 등속이 무너지는 가속 · 감속 · 경사 칸 대조)
  O참  = 참 미래 중심 (label) — 신탁, 상한
예측 x = p 원값 복사 → 목표 칸에 내용을 덮어씀 (원래 자리는 지우지 않는다 — R1 d2 · d3 와 같음). 12 판 + 기준 (rel · copy · pft · d1).
PFT 되돌림 (test block): i = PFT 의 O참 물체 토큰을 O끝 으로 옮기고 O참 은 빈 장면 문맥의 PFT 예측으로 채움 ·
  ii = PFT 의 O참 존재 대비를 릴리즈 크기로 줄임 (x[O참] = pf_빈 + s_t (pf − pf_빈), s_t = ‖(p − p_빈)[O참]‖ / ‖(pf − pf_빈)[O참]‖) · iii = ii 뒤 i.
대상: late · 이동 · permanence (A · B, 60 block) · late · 이동 · 모양 (A, 60 block). 운동 종류별 (등속 · 가속 · 감속 · 경사 내림 · 경사 오름, 각 12).
판정 (PLAN_R2 §5 E24): H2-4 = A(z, O등속) ≥ 62 & A(z, O참) 와 10 pt 안 (등속 칸만 되면 '1 차 외삽은 등속에서만'). H2-5 = A(p2, O_p) ≥ 50.
  되돌림: i 로 A ≤ 30 → 자리 몫 · ii 로 ≤ 30 → 크기 몫 · iii 에서만 → 둘 다 필요.
속도 (phase velocity): late · 이동 · 물체 영상 (permanence A 팔 + 모양 · 색 두 팔) 의 z_6 · z_7 → 문맥 끝 속도 v = u(t7) − u(t6) (px / 튜블릿, label).
  특징 = [블록 평균 z_6, z_7] (2560) 또는 [O끝 평균 z_6, z_7]. ridge (λ = train 안 5 겹으로 고름) · block 5 겹 R² (가로 · 세로).
  대조 = 보임 조건 · 이동 · 모양 · 색 (E21 추출 z_vis_ctx) 같은 식. 판정: R² ≥ 0.8 → 'late 에서도 encoder 가 속도를 읽는다' · ≤ 0.3 → 사용자 전제 수정.
출력: results/q1q2_r2/E24_inject.json · E24_rows.jsonl · E24_velocity.json

  $P z_research/scripts/analysis/transfer_gap_inject2.py inject [--limit 4]
  $P z_research/scripts/analysis/transfer_gap_inject2.py velocity
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_gap_common as C  # noqa: E402
import transfer_gap_margin as MG  # noqa: E402

NT, NF, S, D, G = C.NT, C.NF, C.S, C.D, C.GRID
OUT = MG.OUT
MOTIONS = ["flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up"]


def place(x, k, content, src_center, tgt_center):
    """content: {칸: 토큰} (원천 칸 번호) → x[k] 의 (칸 − src_center + tgt_center) 에 덮어씀."""
    dr, dc = tgt_center[0] - src_center[0], tgt_center[1] - src_center[1]
    for cc, tok in content.items():
        rr, c2 = divmod(cc, G); r2, c3 = rr + dr, c2 + dc
        if 0 <= r2 < G and 0 <= c3 < G:
            x[k, r2 * G + c3] = tok


def cell_rc(uv):
    return C.cell_of(uv)


def cv_centers(uv, hid, tl):
    vis = [t for t in range(NF) if not hid[t]]
    prev = [t for t in vis if t < tl]
    if not prev:
        return [cell_rc(uv[tl])] * NF, (0.0, 0.0)
    t2 = prev[-1]
    v = (uv[tl] - uv[t2]) / (tl - t2)
    return [cell_rc(uv[tl] + v * (t - tl)) for t in range(NF, NT)], (float(v[0]), float(v[1]))


def ca_centers(uv, hid, tl):
    """등가속 외삽 (추가, PLAN 밖): 마지막으로 보인 문맥 튜블릿 셋 (t끝 포함) 의 중심에 2 차식을 맞춰 미래로 늘인다. 셋이 없으면 등속."""
    vis = [t for t in range(NF) if not hid[t] and t <= tl]
    if len(vis) < 3:
        return cv_centers(uv, hid, tl)[0]
    ts = np.array(vis[-3:], float); X = np.stack([uv[int(t)] for t in ts])
    cu = np.polyfit(ts, X[:, 0], 2); cv_ = np.polyfit(ts, X[:, 1], 2)
    return [cell_rc(np.array([np.polyval(cu, t), np.polyval(cv_, t)])) for t in range(NF, NT)]


def argmax_cell(e):
    r, c = divmod(int(e.argmax()), G)
    return r, c


def box3(rc):
    r, c = rc
    return [rr * G + cc for rr in range(max(r - 1, 0), min(r + 1, G - 1) + 1) for cc in range(max(c - 1, 0), min(c + 1, G - 1) + 1)]


def inject(limit):
    cx = MG.Ctx(); M = cx.M
    sel = []
    for b, vv in cx.blocks.items():
        m = M[vv["pos_a"]]
        if m["occlusion"] == "late" and m["motion"] != "flat_static" and m["violation"] in ("permanence", "shape"):
            sel.append(b)
    sel.sort(key=int)
    if limit:
        sel = [b for b in sel if M[cx.blocks[b]["pos_a"]]["violation"] == "permanence"][:limit] + [b for b in sel if M[cx.blocks[b]["pos_a"]]["violation"] == "shape"][:2]
    rows = []
    t0 = time.time()
    for n_, b in enumerate(sel):
        test = b not in cx.trb
        geo = cx.geo(b); uv, ext, hid, tl = geo["uv"], geo["ext"], geo["hid"], geo["tl"]
        ja, jb = cx.blocks[b]["pos_a"], cx.blocks[b]["pos_b"]; m = M[ja]; o = m["objset"]; mp = cx.maps[o]
        pa, pb = cx.arr(cx.P, ja), cx.arr(cx.P, jb)
        pta, ptb = MG.LNnp(pa), MG.LNnp(pb)
        dP = pta - ptb
        Op = [argmax_cell((dP[k] ** 2).sum(-1)) for k in range(NF)]       # 릴리즈가 둔 자리 (출력만)
        cv, vel = cv_centers(uv, hid, tl)
        ends = [cell_rc(uv[tl])] * NF
        trues = [cell_rc(uv[t]) for t in range(NF, NT)]
        ca = ca_centers(uv, hid, tl)
        places = {"Oend": ends, "Op": Op, "Ocv": cv, "Oca": ca, "Otrue": trues}
        src_cells = C.region(uv[tl], ext[tl]); src_c = cell_rc(uv[tl])
        if test:
            pfa, pfb = cx.arr(cx.PF, ja), cx.arr(cx.PF, jb)
        for var, j, p, pt in (("pos_a", ja, pa, pta), ("pos_b", jb, pb, ptb)):
            if m["violation"] == "shape" and var == "pos_b":
                continue
            hp = cx.arr(cx.H, j)[NF:]; hi = cx.arr(cx.HI, cx.imp_of[j]); z = cx.arr(cx.Z, cx.zslot[j])
            sc = lambda x: dict(acc=float(np.abs(x - hi).mean()) > float(np.abs(x - hp).mean()),
                                m=(float(np.abs(x - hi).mean()) - float(np.abs(x - hp).mean())) / float(np.abs(x - hp).mean()))
            r = dict(block=b, objset=o, violation=m["violation"], motion=m["motion"], k=m["k"], dirn="A" if var == "pos_a" else "B", test=test,
                     tl=tl, vel_ctx=vel,
                     d_cv_true=float(np.mean([np.hypot(a[0] - c[0], a[1] - c[1]) for a, c in zip(cv, trues)])),
                     d_ca_true=float(np.mean([np.hypot(a[0] - c[0], a[1] - c[1]) for a, c in zip(ca, trues)])),
                     d_p_true=float(np.mean([np.hypot(a[0] - c[0], a[1] - c[1]) for a, c in zip(Op, trues)])),
                     d_end_true=float(np.mean([np.hypot(a[0] - c[0], a[1] - c[1]) for a, c in zip(ends, trues)])))
            r["rel"] = sc(p)
            r["copy"] = sc(np.broadcast_to(z[7], (NF, S, D)).copy())
            d1 = p.copy()
            for k in range(NF):
                cells = list(geo["Ot"][k]); d1[k, cells] = hp[k, cells]
            r["d1"] = sc(d1)
            for cn in ("z", "p1", "p2"):
                for pn, centers in places.items():
                    x = p.copy()
                    for k in range(NF):
                        if cn == "z":
                            content = {cc: z[tl, cc] for cc in src_cells}; sc0 = src_c
                        else:
                            cells = box3(Op[k]); sc0 = Op[k]
                            if cn == "p1":
                                content = {cc: p[k, cc] for cc in cells}
                            else:
                                content = {cc: mp["mu_p"][k, cc] + 2 * mp["gam_t"][k] * (pt[k, cc] - mp["mu_p"][k, cc]) for cc in cells}
                        place(x, k, content, sc0, centers[k])
                    r[f"{cn}@{pn}"] = sc(x)
            if test:
                pf = pfa if var == "pos_a" else pfb
                pfe = pfb if var == "pos_a" else pfa       # 빈 장면 (permanence) / 다른 물체 (모양) 문맥의 PFT 예측
                r["pft"] = sc(pf)
                if m["violation"] == "permanence":
                    def revert_place(xin):
                        x = xin.copy()
                        for k in range(NF):
                            cells = list(geo["Ot"][k])
                            content = {cc: xin[k, cc].copy() for cc in cells}
                            x[k, cells] = pfe[k, cells] if var == "pos_a" else xin[k, cells]
                            place(x, k, content, trues[k], ends[k])
                        return x

                    def revert_size(xin):
                        x = xin.copy()
                        for k in range(NF):
                            cells = list(geo["Ot"][k])
                            dpf = (pfa - pfb)[k, cells]; drl = (pa - pb)[k, cells]
                            s_ = float(np.linalg.norm(drl) / max(np.linalg.norm(dpf), 1e-9))
                            x[k, cells] = pfe[k, cells] + s_ * (xin[k, cells] - pfe[k, cells]) if var == "pos_a" else xin[k, cells]
                        return x
                    r["rev_i_place"] = sc(revert_place(pf))
                    r["rev_ii_size"] = sc(revert_size(pf))
                    r["rev_iii_both"] = sc(revert_place(revert_size(pf)))
            rows.append(r)
        if n_ % 20 == 0:
            print(f"{n_}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
    return rows


def summarize(rows):
    out = {}
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, dict) and "acc" in v})
    for viol in ("permanence", "shape"):
        for o in ("primitive", "realistic", "both"):
            for dirn in ("A", "B"):
                for split in ("all", "test"):
                    for mv in ["all"] + MOTIONS:
                        sel = [r for r in rows if r["violation"] == viol and (o == "both" or r["objset"] == o) and r["dirn"] == dirn
                               and (split == "all" or r["test"]) and (mv == "all" or r["motion"] == mv)]
                        if not sel:
                            continue
                        if mv != "all" and o != "both":
                            continue
                        d = dict(n=len(sel))
                        for k in keys:
                            v = [r for r in sel if k in r]
                            if v:
                                d[k] = C.bootstrap_mean([100 * float(r[k]["acc"]) for r in v], [r["block"] for r in v])
                                d[k]["m_median"] = float(np.median([r[k]["m"] for r in v]))
                        for g in ("d_cv_true", "d_ca_true", "d_p_true", "d_end_true"):
                            d[g] = float(np.mean([r[g] for r in sel]))
                        out[f"{viol}|{o}|{dirn}|{split}|{mv}"] = d
    # 판정
    ver = {}
    for split in ("all", "test"):
        d = out.get(f"permanence|both|A|{split}|all", {})
        if not d:
            continue
        g = lambda k: d.get(k, {}).get("mean", float("nan"))
        ver[split] = dict(A_z_Ocv=g("z@Ocv"), A_z_Otrue=g("z@Otrue"), A_z_Oend=g("z@Oend"), A_z_Op=g("z@Op"), A_copy=g("copy"), A_rel=g("rel"),
                          A_p2_Op=g("p2@Op"), A_p2_Ocv=g("p2@Ocv"), A_p2_Otrue=g("p2@Otrue"), A_p1_Otrue=g("p1@Otrue"),
                          H2_4=bool(g("z@Ocv") >= 62 and abs(g("z@Ocv") - g("z@Otrue")) <= 10), H2_5=bool(g("p2@Op") >= 50))
        if split == "test":
            ver[split].update(A_pft=g("pft"), A_rev_i_place=g("rev_i_place"), A_rev_ii_size=g("rev_ii_size"), A_rev_iii_both=g("rev_iii_both"))
            ver[split]["pft_share"] = ("자리 몫" if g("rev_i_place") <= 30 else "") + (" 크기 몫" if g("rev_ii_size") <= 30 else "") or \
                ("둘 다 필요" if g("rev_iii_both") <= 30 else "어느 되돌림으로도 30 아래로 안 감")
        ver[split]["by_motion_z_Ocv"] = {mv: out.get(f"permanence|both|A|{split}|{mv}", {}).get("z@Ocv", {}).get("mean") for mv in MOTIONS}
        ver[split]["by_motion_z_Oca"] = {mv: out.get(f"permanence|both|A|{split}|{mv}", {}).get("z@Oca", {}).get("mean") for mv in MOTIONS}
        ver[split]["A_z_Oca"] = g("z@Oca")
        ver[split]["by_motion_z_Otrue"] = {mv: out.get(f"permanence|both|A|{split}|{mv}", {}).get("z@Otrue", {}).get("mean") for mv in MOTIONS}
        ver[split]["by_motion_copy"] = {mv: out.get(f"permanence|both|A|{split}|{mv}", {}).get("copy", {}).get("mean") for mv in MOTIONS}
    return out, ver


# ------------------------------------------------------------------ 속도
def ridge_cv(X, Y, groups, lams=(1e-1, 1, 10, 100, 1000, 1e4), k=5, seed=0):
    """block 5 겹 R² (출력 차원별). λ 는 바깥 train 안에서 다시 5 겹으로 고른다."""
    ub = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(ub)
    folds = np.array_split(ub, k)
    pred = np.zeros_like(Y)

    def fit(Xa, Ya, lam):
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; A = (Xa - mu) / sd; ym = Ya.mean(0)
        if A.shape[0] < A.shape[1]:
            K = A @ A.T
            W = A.T @ np.linalg.solve(K + lam * np.eye(len(K)), Ya - ym)
        else:
            W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ya - ym))
        return lambda Xb: ((Xb - mu) / sd) @ W + ym

    lam_used = []
    for f in folds:
        te = np.isin(groups, f); tr = ~te
        gtr = groups[tr]; ubt = np.unique(gtr); rng.shuffle(ubt); inner = np.array_split(ubt, 4)
        best, bl = None, None
        for lam in lams:
            err = 0.0
            for g in inner:
                v = np.isin(gtr, g)
                fn = fit(X[tr][~v], Y[tr][~v], lam); err += ((fn(X[tr][v]) - Y[tr][v]) ** 2).sum()
            if best is None or err < best:
                best, bl = err, lam
        lam_used.append(bl)
        pred[te] = fit(X[tr], Y[tr], bl)(X[te])
    r2 = 1 - ((pred - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    return [float(v) for v in r2], lam_used


def velocity():
    M = C.meta(); bmap, UV, HID, EXT = C.load_tracks()
    Z = np.load(C.TG / "z_late_ctx.npy", mmap_mode="r"); zm = json.load(open(C.TG / "z_late_meta.json"))
    out = {"definition": "z_6 · z_7 (LN, 블록 평균 / O끝 평균) → v = u(t7) − u(t6) (px@512 / 튜블릿), ridge block 5 겹 R² (가로, 세로)"}
    sets = {"late": (Z, zm["tp"])}
    fz = C.TG / "z_vis_ctx.npy"
    if fz.exists():
        sets["vis"] = (np.load(fz, mmap_mode="r"), json.load(open(C.TG / "z_vis_meta.json"))["tp"])
    for nm, (ZZ, tps) in sets.items():
        feats = defaultdict(list); Y = []; grp = []; objs = []; ks = []
        for q, j in enumerate(tps):
            m = M[j]
            if m["motion"] == "flat_static" or C.label_of(m) == "empty":
                continue
            bi = bmap[m["block_id"]]; arm = 0 if m["variant"] == "pos_a" else 1
            uv = UV[bi, arm]; hid = HID[bi, arm]
            tl = C.last_visible_ctx(hid)
            z = np.asarray(ZZ[q], np.float32)
            Oe = C.region(uv[tl], EXT[bi, arm][tl])
            feats["blockmean"].append(np.concatenate([z[6].mean(0), z[7].mean(0)]))
            feats["Oend"].append(np.concatenate([z[6, Oe].mean(0), z[7, Oe].mean(0)]))
            feats["blockmean_z7only"].append(z[7].mean(0))
            Y.append(uv[7] - uv[6]); grp.append(m["block_id"]); objs.append(m["objset"]); ks.append((hid[6], hid[7]))
        Y = np.array(Y, np.float64); grp = np.array(grp); objs = np.array(objs)
        res = dict(n=len(Y), n_block=len(np.unique(grp)), hidden_t6_frac=float(np.mean([a for a, _ in ks])), hidden_t7_frac=float(np.mean([b for _, b in ks])),
                   v_std=Y.std(0).tolist())
        for fn, F in feats.items():
            X = np.stack(F).astype(np.float64)
            for o in ("primitive", "realistic", "both"):
                s = (objs == o) if o != "both" else np.ones(len(Y), bool)
                r2, lam = ridge_cv(X[s], Y[s], grp[s])
                # 섞은 라벨 대조 (block 단위로 섞음)
                rng = np.random.default_rng(1); ub = np.unique(grp[s]); perm = dict(zip(ub, rng.permutation(ub)))
                Ysh = np.stack([Y[s][grp[s] == perm[g]][0] if (grp[s] == perm[g]).any() else y for g, y in zip(grp[s], Y[s])])
                r2s, _ = ridge_cv(X[s], Ysh, grp[s])
                res[f"{fn}|{o}"] = dict(R2_u=r2[0], R2_v=r2[1], lam=lam, shuffled_R2=r2s)
                print(nm, fn, o, [round(v, 3) for v in r2], "shuffled", [round(v, 3) for v in r2s], flush=True)
        out[nm] = res
    d = out["late"].get("blockmean|both", {})
    out["verdict"] = dict(late_R2_u=d.get("R2_u"), rule="R² ≥ 0.8 → late 에서도 encoder 가 속도를 읽는다 · ≤ 0.3 → 사용자 전제 수정",
                          verdict=("읽는다" if (d.get("R2_u") or 0) >= 0.8 else "전제 수정" if (d.get("R2_u") or 0) <= 0.3 else "중간"))
    json.dump(out, open(OUT / "E24_velocity.json", "w"), indent=1, ensure_ascii=False)
    print(out["verdict"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["inject", "velocity"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.phase == "inject":
        t0 = time.time()
        rows = inject(a.limit)
        summ, ver = summarize(rows)
        suf = f"_smoke{a.limit}" if a.limit else ""
        json.dump(dict(definition=__doc__, verdict=ver, summary=summ), open(OUT / f"E24_inject{suf}.json", "w"), indent=1, ensure_ascii=False, default=float)
        with open(OUT / f"E24_rows{suf}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        print(json.dumps(ver, ensure_ascii=False, default=float), f"{time.time() - t0:.0f}s")
    else:
        velocity()
