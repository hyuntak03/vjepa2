#!/usr/bin/env python3
"""[X1] E2 쌍 차이 지도 — Q1 (방향 · 크기 · 자리) 의 파라미터 없는 판정 + E8b (PFT 이득 = 운반?) (2026-10-30).

같은 block 의 가능 영상 A · B 는 같은 장면 · 같은 궤적이고 물체만 다르다 (모양 · 색 block = 정체 대비, permanence block = 존재 대비 물체 − 빈).
  Δh_t = h_A,t − h_B,t            (h = LN(target_encoder(32 장)), 256 칸 × 1280)
  Δp_t = LN(p_A,t) − LN(p_B,t)    (p = predictor 미래, 토큰별 학습 계수 없는 LN = 판독기가 보는 것). 릴리즈 (p) · PFT (pf)
지표 (걸음 t = 8..15 마다, X = 예측 쪽 차이, Y = 참 차이):
  ρ = ‖X‖/‖Y‖ (전 칸) · ρ_O (참 물체 영역 합집합)
  κ고정 = cos(X, Y) · κ이동 = max_{s ∈ [−9,9]×[−3,3] 칸} cos(X, T_s Y), s* = 최선 이동 (T_s = 칸 평행 이동, 빈 칸 0)
  κ끝 = max_s cos(X, T_s Δh_t끝) (t끝 = 문맥 안 마지막으로 보인 튜블릿) · late 는 Δz_t끝 (z = LN(문맥 encoder)) 판도
  κ모음 = cos(Σ_{c∈Op} X_c, Σ_{c∈O참} Y_c), Op = X 에너지 최대 칸의 3×3
  τ* = argmax_{t′∈0..15} κ이동(X, Δh_t′)   ⚠️ PLAN 은 물체 영역 L1 이었다 — 여기는 쌍 차이 cos 판 (배경이 빠진 같은 척도)
  N = (Σ_c e_c)²/Σ_c e_c² (유효 칸 수, e_c = ‖X_c‖²) — 비 N(X)/N(Y)
  질량 m_참 · m_끝 · m_숨 · m_나머지 (우선순위 참 > 끝 > 숨, 영역 = 3×3 / 균등 기준 9/256 = 0.035)
  주성분별 r_j = Σ(u_jᵀX)²/Σ(u_jᵀY)² (u_j = split v2 train h 토큰 주성분), 지수 = median r (분산 위 10 %) / median r (아래 50 %)
  따라감 f = 1 + ⟨s*, d참⟩/|d참|² (d참 = 칸 단위 참 이동 t끝 → t; f = 1 제자리 추적, 0 = 마지막 본 자리에 머묾), |d참| ≥ 1 칸만
  정규화 κ̂ = (κ − κ바닥)/(κ천장 − κ바닥): κ천장 = κ이동(Δh_t, Δh_{t±1}) (같은 block), κ바닥 = κ이동(X_b, Δh_b′) (같은 조건의 다른 block)
대조 S5 (encoder 보임 → 가림): X = Δh_{t가림} (두 팔 모두 가려진 튜블릿), Y = Δh_{가장 가까운 보인 튜블릿}.
시나리오: S1 보임 조건 · 이동 · t8–10 보임 / S2 t11–15 / S4 late · 이동 재등장 (+ late 가림) / S5 / S6 = PFT (test block) / 정지 대조.
통계: block bootstrap 2000 (평균의 95 % CI).
출력: results/q1q2/E2_pairdiff.json · E2_rows.npz (block × 걸음 행) · results/q1q2/fig/

  $P z_research/scripts/analysis/transfer_gap_pairdiff.py --gpu 0
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_gap_common as C  # noqa: E402

NT, NF, S, D, G = C.NT, C.NF, C.S, C.D, C.GRID
SH = [(dr, dc) for dr in range(-3, 4) for dc in range(-9, 10)]


def torch_tools(dev):
    import torch

    def LN(x):
        return torch.nn.functional.layer_norm(x.float(), (x.shape[-1],))

    def shift_stats(X, Y):
        """X (B,256,D), Y (B,256,D) → cos(X, T_s Y) for all s (B, |SH|), 그리고 ⟨X,T_sY⟩."""
        Gm = torch.einsum("bcd,bed->bce", X, Y).reshape(-1, G, G, G, G)     # [b, r, c, r', c']
        eY = (Y * Y).sum(-1).reshape(-1, G, G)
        nX = X.flatten(1).norm(dim=1)
        out = []
        for dr, dc in SH:
            # T_s Y at (r,c) = Y at (r-dr, c-dc)
            r0, r1 = max(dr, 0), G + min(dr, 0); c0, c1 = max(dc, 0), G + min(dc, 0)
            rr = torch.arange(r0, r1, device=X.device); cc = torch.arange(c0, c1, device=X.device)
            sub = Gm[:, rr][:, :, cc]                                          # (b, nr, nc, G, G)
            ip = sub[:, torch.arange(len(rr))[:, None], torch.arange(len(cc))[None, :], (rr - dr)[:, None], (cc - dc)[None, :]].sum((1, 2))
            nY = eY[:, (rr - dr)][:, :, (cc - dc)].sum((1, 2)).clamp_min(1e-12).sqrt()
            out.append(ip / (nX * nY).clamp_min(1e-12))
        return torch.stack(out, 1)

    SHT = torch.tensor(SH, device=dev)

    def region_shift(X, Y, O):
        """물체 영역판: cos(X[O+s], Y[O]) for all s (|SH|,). O = 칸 번호 목록."""
        O = torch.tensor(sorted(O), device=dev)
        r, c = O // G, O % G
        rr = r[None] + SHT[:, :1]; cc = c[None] + SHT[:, 1:]
        valid = ((rr >= 0) & (rr < G) & (cc >= 0) & (cc < G)).float()
        idx = (rr.clamp(0, G - 1) * G + cc.clamp(0, G - 1))
        Xs = X[idx] * valid[..., None]                                   # (|SH|, |O|, D)
        Yo = Y[O]
        ip = torch.einsum("sod,od->s", Xs, Yo)
        return ip / (Xs.flatten(1).norm(dim=1) * Yo.norm()).clamp_min(1e-12)

    return LN, shift_stats, region_shift


def neff(e):
    return float(e.sum() ** 2 / max((e ** 2).sum(), 1e-12))


def regions_for(uv, hid, t, ext):
    """참 (O참) · 끝 (O끝) · 숨 (O숨) 칸 집합 — 우선순위 참 > 끝 > 숨."""
    tl = C.last_visible_ctx(hid)
    o_true = set(C.region(uv[t], ext[t]))
    o_end = set(C.region(uv[tl], ext[tl])) - o_true
    o_hid = set()
    for tt in range(NT):
        if hid[tt] and tt <= t:
            o_hid |= set(C.region(uv[tt]))
    o_hid -= o_true | o_end
    return o_true, o_end, o_hid, tl


def main(gpu, objsets, limit):
    import torch
    dev = torch.device(f"cuda:{gpu}")
    LN, shift_stats, region_shift = torch_tools(dev)
    M = C.meta(); trb = C.train_blocks(); bmap, UV, HID, EXT = C.load_tracks()
    zm = json.load(open(C.TG / "z_late_meta.json")); zslot = {j: q for q, j in enumerate(zm["tp"])}
    H = np.load(C.TUBE / "h.npy", mmap_mode="r"); P = np.load(C.TUBE / "p.npy", mmap_mode="r")
    PF = np.load(C.TG / "p_relftg.npy", mmap_mode="r"); Z = np.load(C.TG / "z_late_ctx.npy", mmap_mode="r")
    tp = {(m["block_id"], m["variant"]): j for j, m in enumerate(M)}

    # 주성분 (split v2 train h 토큰, 표본)
    t0 = time.time()
    rng = np.random.default_rng(0)
    trv = [j for j, m in enumerate(M) if m["block_id"] in trb]
    pick = rng.choice(trv, size=min(240, len(trv)), replace=False)
    tok = []
    for j in sorted(pick):
        x = np.asarray(H[j], np.float32).reshape(-1, D)
        tok.append(x[rng.choice(len(x), 512, replace=False)])
    T = torch.from_numpy(np.concatenate(tok)).to(dev)
    T = T - T.mean(0)
    evals, U = torch.linalg.eigh(T.T @ T / len(T))
    order = torch.argsort(evals, descending=True); evals, U = evals[order], U[:, order]
    top = slice(0, D // 10); bot = slice(D // 2, D)
    print(f"PCA {T.shape} {time.time() - t0:.0f}s; var top10% {float(evals[top].sum() / evals.sum()):.3f}", flush=True)
    del T

    # 대상 block
    blocks = defaultdict(dict)
    for j, m in enumerate(M):
        blocks[m["block_id"]][m["variant"]] = j
    bl = [b for b in blocks if M[blocks[b]["pos_a"]]["objset"] in objsets]
    bl = sorted(bl, key=int)[: limit or None]
    # 바닥용 짝 block: 같은 (objset, 위반, 가림, k, 운동) 의 다른 block (물체 조합이 다른 것)
    grp = defaultdict(list)
    for b in bl:
        m = M[blocks[b]["pos_a"]]
        grp[(m["objset"], m["violation"], m["occlusion"], m["k"], m["motion"])].append(b)
    floor_of = {}
    for g, bs in grp.items():
        for q, b in enumerate(bs):
            ma = M[blocks[b]["pos_a"]]
            cand = [c for c in bs[q + 1:] + bs[:q] if (M[blocks[c]["pos_a"]]["obj_a"], M[blocks[c]["pos_a"]]["obj_b"]) != (ma["obj_a"], ma["obj_b"])]
            floor_of[b] = cand[0] if cand else None

    rows = []; pcs = defaultdict(lambda: np.zeros((2, D)))
    load = lambda arr, j: torch.from_numpy(np.asarray(arr[j], np.float32)).to(dev)
    t0 = time.time()
    for n_, b in enumerate(bl):
        ja, jb = blocks[b]["pos_a"], blocks[b]["pos_b"]; m = M[ja]
        bi = bmap[b]; uvA, hidA, extA = UV[bi, 0], HID[bi, 0], EXT[bi, 0]
        hidB = HID[bi, 1] if m["violation"] != "permanence" else hidA
        hid = hidA | hidB
        st = C.tub_state(m["samples"])
        hA, hB = load(H, ja), load(H, jb)
        dH = hA - hB                                                                   # (16,256,D)
        dP = LN(load(P, ja)) - LN(load(P, jb)); dPF = LN(load(PF, ja)) - LN(load(PF, jb))
        late = m["occlusion"] == "late"
        dZ = (load(Z, zslot[ja]) - load(Z, zslot[jb])) if late else None
        fb = floor_of.get(b)
        dHf = (load(H, blocks[fb]["pos_a"]) - load(H, blocks[fb]["pos_b"])) if fb else None
        tl = C.last_visible_ctx(hid)
        ts = torch.arange(NF, NT)
        Y = dH[NF:]                                                                    # (8,256,D)
        ceilY = dH[torch.tensor([t + 1 if t < NT - 1 else t - 1 for t in range(NF, NT)])]
        cs_ceil = shift_stats(Y, ceilY).max(1).values
        res_m = {}
        for mdl, X in (("rel", dP), ("pft", dPF)):
            cs = shift_stats(X, Y)                                                     # (8, |SH|)
            kmax, kidx = cs.max(1)
            kfix = cs[:, SH.index((0, 0))]
            kend = shift_stats(X, dH[tl].expand(NF, -1, -1)).max(1).values
            kendz = shift_stats(X, dZ[tl].expand(NF, -1, -1)).max(1).values if late else None
            kfloor = shift_stats(X, dHf[NF:]).max(1).values if dHf is not None else None
            # τ*: 모든 참 걸음과 (한 번에: 8 × 16 짝)
            ktau = shift_stats(X.repeat_interleave(NT, 0), dH.repeat(NF, 1, 1)).max(1).values.reshape(NF, NT)   # (8,16)
            px = (torch.einsum("tcd,dj->tcj", X, U) ** 2).sum(1)                      # (8, D)
            py = (torch.einsum("tcd,dj->tcj", Y, U) ** 2).sum(1)
            res_m[mdl] = (cs, kmax, kidx, kfix, kend, kendz, kfloor, ktau, px, py)
        # 대조 S5: encoder 가림 튜블릿 (두 팔 모두 가려짐)
        s5 = []
        occ_t = [t for t in range(NT) if st[t] == "occ"]
        vis_t = [t for t in range(NT) if st[t] == "vis"]
        if occ_t and vis_t:
            for t in occ_t:
                tv = min(vis_t, key=lambda v: (abs(v - t), -v))
                cs = shift_stats(dH[t][None], dH[tv][None])[0]
                eX = (dH[t] ** 2).sum(-1).cpu().numpy(); eY = (dH[tv] ** 2).sum(-1).cpu().numpy()
                s5.append(dict(t=t, tv=tv, rho=float(dH[t].norm() / dH[tv].norm()), kmove=float(cs.max()), kfix=float(cs[SH.index((0, 0))]),
                               neff_ratio=neff(eX) / max(neff(eY), 1e-9)))
        for k, t in enumerate(range(NF, NT)):
            o_true, o_end, o_hid, _ = regions_for(uvA, hid, t, extA)
            eY = (Y[k] ** 2).sum(-1).cpu().numpy()
            dtrue = np.array(C.cell_of(uvA[t])) - np.array(C.cell_of(uvA[tl]))       # (dr, dc)
            base = dict(block=b, objset=m["objset"], violation=m["violation"], occlusion=m["occlusion"], k=m["k"], motion=m["motion"],
                        moving=m["motion"] != "flat_static", t=t, state=st[t], test=b not in trb, t_end=tl, d_true=dtrue.tolist(),
                        ceil=float(cs_ceil[k]), nY=float(Y[k].norm()), neffY=neff(eY),
                        mY_true=float(eY[list(o_true)].sum() / eY.sum()), mY_end=float(eY[list(o_end)].sum() / eY.sum()) if o_end else 0.0)
            for mdl in ("rel", "pft"):
                cs, kmax, kidx, kfix, kend, kendz, kfloor, ktau, px, py = res_m[mdl]
                X = dP[k] if mdl == "rel" else dPF[k]
                eX = (X ** 2).sum(-1).cpu().numpy()
                rr_, cc_ = divmod(int(eX.argmax()), G)
                op = C.region(((cc_ + 0.5) * C.CELL, (rr_ + 0.5) * C.CELL))
                ot = list(o_true)
                kcol = float(torch.nn.functional.cosine_similarity(X[op].sum(0), Y[k][ot].sum(0), dim=0))
                s = SH[int(kidx[k])]
                f = None
                if np.hypot(*dtrue) >= 1:
                    f = float(1 + (s[0] * dtrue[0] + s[1] * dtrue[1]) / (dtrue ** 2).sum())
                rest = set(range(S)) - o_true - o_end - o_hid
                # 물체 영역판 (Y 의 O참 패턴을 X 에서 찾는다)
                cso = region_shift(X, Y[k], o_true); kOm, kOi = cso.max(0)
                sO = SH[int(kOi)]
                fO = float(1 + (sO[0] * dtrue[0] + sO[1] * dtrue[1]) / (dtrue ** 2).sum()) if np.hypot(*dtrue) >= 1 else None
                dend = tuple(int(v) for v in (np.array(C.cell_of(uvA[tl])) - np.array(C.cell_of(uvA[t]))))
                kO_atend = float(cso[SH.index(dend)]) if dend in SH else None
                tn = t + 1 if t < NT - 1 else t - 1
                kO_ceil = float(region_shift(dH[t], dH[tn], set(C.region(uvA[tn], extA[tn]))).max())
                kO_floor = None
                if dHf is not None:
                    bf = bmap[fb]; kO_floor = float(region_shift(X, dHf[t], set(C.region(UV[bf, 0][t], EXT[bf, 0][t]))).max())
                r = dict(base, model=mdl, rho=float(X.norm() / Y[k].norm()),
                         rho_O=float(X[ot].norm() / Y[k][ot].norm()), kmove=float(kmax[k]), kfix=float(kfix[k]), s_star=list(s), follow=f,
                         kend=float(kend[k]), kendz=float(kendz[k]) if kendz is not None else None,
                         kfloor=float(kfloor[k]) if kfloor is not None else None, kcol=kcol,
                         tau=int(ktau[k].argmax()), ktau=ktau[k].cpu().numpy().round(4).tolist(),
                         neff_ratio=neff(eX) / max(neff(eY), 1e-9),
                         m_true=float(eX[ot].sum() / eX.sum()), m_end=float(eX[list(o_end)].sum() / eX.sum()) if o_end else 0.0,
                         m_hid=float(eX[list(o_hid)].sum() / eX.sum()) if o_hid else 0.0, m_rest=float(eX[list(rest)].sum() / eX.sum()),
                         n_true=len(o_true), n_end=len(o_end), n_hid=len(o_hid),
                         kO_move=float(kOm), kO_fix=float(cso[SH.index((0, 0))]), sO_star=list(sO), followO=fO, kO_atend=kO_atend,
                         kO_ceil=kO_ceil, kO_floor=kO_floor)
                rows.append(r)
                key = (mdl, m["objset"], "id" if m["violation"] != "permanence" else "perm", m["occlusion"] == "late", r["moving"], st[t], t >= 11)
                pcs[key][0] += px[k].cpu().numpy(); pcs[key][1] += py[k].cpu().numpy()
        for d5 in s5:
            rows.append(dict(block=b, objset=m["objset"], violation=m["violation"], occlusion=m["occlusion"], k=m["k"], motion=m["motion"],
                             moving=m["motion"] != "flat_static", model="enc_occ", test=b not in trb, **d5))
        if n_ % 40 == 0:
            print(f"block {n_}/{len(bl)} {time.time() - t0:.0f}s", flush=True)
    return rows, pcs, evals.cpu().numpy()


def summarize(rows, pcs, evals):
    out = {}
    fut = [r for r in rows if r.get("model") in ("rel", "pft")]

    def agg(sel, keys=("rho", "rho_O", "kmove", "kfix", "kend", "kendz", "kfloor", "ceil", "kcol", "neff_ratio", "m_true", "m_end", "m_hid", "m_rest", "follow", "mY_true",
                       "kO_move", "kO_fix", "followO", "kO_atend", "kO_ceil", "kO_floor")):
        d = dict(n=len(sel), n_block=len({r["block"] for r in sel}))
        if not sel:
            return d
        for k in keys:
            v = [(r[k], r["block"]) for r in sel if r.get(k) is not None]
            if v:
                d[k] = C.bootstrap_mean([a for a, _ in v], [b for _, b in v])
        if "kmove" in d and "kfloor" in d and "ceil" in d:
            km, kf, kc = d["kmove"]["mean"], d["kfloor"]["mean"], d["ceil"]["mean"]
            d["kmove_hat"] = (km - kf) / (kc - kf) if kc != kf else None
            d["kfix_hat"] = (d["kfix"]["mean"] - kf) / (kc - kf) if kc != kf else None
        if "kO_move" in d and "kO_floor" in d and "kO_ceil" in d:
            km, kf, kc = d["kO_move"]["mean"], d["kO_floor"]["mean"], d["kO_ceil"]["mean"]
            d["kO_move_hat"] = (km - kf) / (kc - kf) if kc != kf else None
            d["kO_fix_hat"] = (d["kO_fix"]["mean"] - kf) / (kc - kf) if kc != kf else None
        taus = [r["tau"] for r in sel if "tau" in r]
        if taus:
            d["tau_mode"] = int(np.bincount(taus, minlength=NT).argmax()); d["tau_hist"] = np.bincount(taus, minlength=NT).tolist()
            d["tau_minus_t_median"] = float(np.median([r["tau"] - r["t"] for r in sel]))
        ss = [r["s_star"] for r in sel if "s_star" in r]
        if ss:
            d["s_star_median"] = np.median(np.array(ss), 0).tolist()
            d["d_true_median"] = np.median(np.array([r["d_true"] for r in sel]), 0).tolist()
        return d

    def scen(model, objset, viol, cond):
        base = [r for r in fut if r["model"] == model and r["objset"] == objset and ((r["violation"] != "permanence") == (viol == "id"))]
        return {k: [r for r in base if f(r)] for k, f in cond.items()}

    cond = {
        "S1_vis_move_t8-10": lambda r: r["occlusion"] != "late" and r["moving"] and r["t"] <= 10 and r["state"] == "vis",
        "S2_vis_move_t11-15": lambda r: r["occlusion"] != "late" and r["moving"] and r["t"] >= 11 and r["state"] == "vis",
        "S4_late_move_reapp": lambda r: r["occlusion"] == "late" and r["moving"] and r["state"] == "vis",
        "S4_late_move_occ": lambda r: r["occlusion"] == "late" and r["moving"] and r["state"] == "occ",
        "static_vis_t8-10": lambda r: r["occlusion"] != "late" and not r["moving"] and r["t"] <= 10 and r["state"] == "vis",
        "static_vis_t11-15": lambda r: r["occlusion"] != "late" and not r["moving"] and r["t"] >= 11 and r["state"] == "vis",
        "static_late_reapp": lambda r: r["occlusion"] == "late" and not r["moving"] and r["state"] == "vis",
    }
    for viol in ("id", "perm"):
        for o in ("primitive", "realistic"):
            for model in ("rel", "pft"):
                for split in ("all", "test"):
                    if model == "pft" and split == "all":
                        continue
                    sc = scen(model, o, viol, cond)
                    for k, sel in sc.items():
                        if split == "test":
                            sel = [r for r in sel if r["test"]]
                        out[f"{viol}|{o}|{model}|{split}|{k}"] = agg(sel)
            # 걸음별 (릴리즈, 보임 조건 이동 / late 이동)
            for model in ("rel", "pft"):
                for t in range(NF, NT):
                    for lab, f in (("vis_move", lambda r: r["occlusion"] != "late" and r["moving"]),
                                   ("late_move", lambda r: r["occlusion"] == "late" and r["moving"])):
                        sel = [r for r in fut if r["model"] == model and r["objset"] == o and ((r["violation"] != "permanence") == (viol == "id"))
                               and r["t"] == t and f(r) and (r["test"] or model == "rel") and (r["state"] == "vis" or lab == "late_move")]
                        if sel:
                            out[f"{viol}|{o}|{model}|bystep|{lab}|t{t}"] = agg(sel, keys=("rho", "rho_O", "kmove", "kfix", "kend", "m_true", "m_end", "follow", "neff_ratio", "kO_move", "kO_fix", "followO", "kO_atend"))
            enc = [r for r in rows if r.get("model") == "enc_occ" and r["objset"] == o and ((r["violation"] != "permanence") == (viol == "id"))]
            for lab, f in (("all", lambda r: True), ("late", lambda r: r["occlusion"] == "late"), ("early_mid", lambda r: r["occlusion"] in ("early", "mid"))):
                sel = [r for r in enc if f(r)]
                out[f"{viol}|{o}|enc_occ|S5_{lab}"] = agg(sel, keys=("rho", "kmove", "kfix", "neff_ratio"))
    # 주성분 지수
    pc = {}
    for key, (sx, sy) in pcs.items():
        r = sx / np.maximum(sy, 1e-12)
        pc["|".join(map(str, key))] = dict(index=float(np.median(r[: D // 10]) / max(np.median(r[D // 2:]), 1e-12)),
                                          r_top10_median=float(np.median(r[: D // 10])), r_bot50_median=float(np.median(r[D // 2:])),
                                          r_first20=r[:20].round(4).tolist())
    out["_pc_index"] = pc
    out["_pc_var_top10_frac"] = float(evals[: D // 10].sum() / evals.sum())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--objsets", default="primitive,realistic")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows, pcs, ev = main(a.gpu, tuple(a.objsets.split(",")), a.limit)
    C.RES.mkdir(parents=True, exist_ok=True)
    suf = "" if not a.limit else f"_smoke{a.limit}"
    summ = summarize(rows, pcs, ev)
    json.dump(dict(definition=__doc__, summary=summ), open(C.RES / f"E2_pairdiff{suf}.json", "w"), indent=1, ensure_ascii=False, default=float)
    with open(C.RES / f"E2_rows{suf}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    print("→", C.RES / f"E2_pairdiff{suf}.json", len(rows))
