#!/usr/bin/env python3
"""[X1] E4 정렬 사다리 (Q1 학습된 판정) + E6 정렬 공간 VoE 재채점 + E7 주입 (Q2) — 한 번의 평가 패스 (2026-10-30).

기호: p̃ = LN(p) (토큰별 학습 계수 없는 LN = 판독기가 실제로 보는 것) · h = LN(target_encoder(32 장)) · t = 미래 걸음 8..15 · c = 칸.
맞춤 데이터 (fit): split v2 **train** block · 보임 조건 (visible · early · mid, late 제외) · 물체가 보이는 걸음 ('vis') · 모든 칸의 (p̃, h) 짝.
  μ_{t,c}, μ^p_{t,c} = train 평균 토큰.  Xr = p̃ − μ^p_{t,c}, Yr = h − μ_{t,c}.
사상 M(p̃):
  L0   p̃ 그대로                                          (VoE 는 원값 p 판 L0raw 도)
  L1   채널별 평균 · 분산 (걸음별): m^h_{t,d} + (s^h_{t,d}/s^p_{t,d})(p̃_d − m^p_{t,d})
  L2   전역 대비 (걸음별 스칼라): μ_{t,c} + γ_t Xr,  γ_t = RMS(Yr_t)/RMS(Xr_t)
  L2b  주성분별 대비: μ + U diag(σ^h_j/σ^p_j) Uᵀ Xr  (U = Yr 주성분)
  L3   직교 Procrustes: μ + γ Xr R  (R = argmin ‖Yr − γ Xr R‖, RᵀR = I; γ = 전체 RMS 비)
  L4   ridge 선형: μ + Xr W  (λ 는 train block 20 % 보류로 고름)
  L4t  걸음별 ridge (같은 λ)
  L5   L4 + 토큰별 MLP f(p̃) (1280→2560→1280 GELU, L1 손실, 3 epoch)
  L4s  섞은 짝 L4 (대조: 같은 조건 · 같은 걸음의 다른 영상 h 와 짝)
  Lsh  자리: p̃ 지도를 E2 의 물체 영역 최선 이동 s*_O 만큼 되돌림 (빈 칸 = μ^p) — 파라미터 0 (단 s* 는 참 Δh 로 찾았다 = 신탁)
  L4sh Lsh 뒤 L4
  L3late · L4late  late train block 의 재등장 걸음 짝으로 맞춘 판 (H1-C 대조)
평가 (test block, 고정 판독기 = E3 의 h · p · 같은 시점 판독기, 시드 0 · 1):
  S1 보임 조건 · 이동 · t8–10 · 정답 보임 / S2 t11–15 / S4r late · 이동 · 재등장 / S4o late · 이동 · 가림 / static (보임 조건 정지) / S3 같은 시점 판독기 (t8 · t11 · t15)
  결합 정확도 · 모양 · 없음 % · 회복률 R = (acc_M − acc_L0)/(acc_천장 − acc_L0), 천장 = 같은 표본의 p 판독기 (L0). block bootstrap 2000 CI.
  대조: 사상을 h 에 건 판 (h 판독기 h 보임 정확도 유지 ≥ 85 %?) · L4 극분해 (회전 tr(U)/1280, 크기 P 의 고유값 분위).
E6: s_M(i) = mean_{t,c,d}|M(p)_{i} − h_i| (전체 토큰 L1, 짝 (pos, imp) 이 p 를 공유). matched 쌍 정답 = s(imp) > s(pos). 방향 A (pos_a, imp_ab) · B.
E7 (late, 물체 영역 = 외곽 상자 + 1 칸, 나머지 칸은 원래 p 원값): d1 = h_A,t (참 미래 정답) 를 O참(t) 에 · d2 = LN(z)[t끝] 물체 칸을 모든 걸음 O끝 에 ·
  d3 = d2 칸을 O참(t) 로 옮김 · d4 = 다른 팔 (B) 의 LN(z)[t끝] 같은 칸을 O참(t) 에.  VoE (E6 과 같은 채점) 와 h 판독기 판독.
출력: TG/maps/<objset>.pt · results/q1q2/E4_ladder_<objset>.json · E6E7_rows_<objset>.jsonl

  $P z_research/scripts/analysis/transfer_gap_ladder.py fit  --objset primitive --gpu 0
  $P z_research/scripts/analysis/transfer_gap_ladder.py eval --objset primitive --gpu 0
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_gap_common as C  # noqa: E402

NT, NF, S, D, G = C.NT, C.NF, C.S, C.D, C.GRID


def _LN(x):
    import torch
    return torch.nn.functional.layer_norm(x.float(), (x.shape[-1],))


def fit(objset, gpu):
    import torch
    dev = torch.device(f"cuda:{gpu}")
    M = C.meta(); trb = C.train_blocks()
    H = np.load(C.TUBE / "h.npy", mmap_mode="r"); P = np.load(C.TUBE / "p.npy", mmap_mode="r")
    rows = [j for j, m in enumerate(M) if m["objset"] == objset and m["block_id"] in trb]
    st = {j: C.tub_state(M[j]["samples"]) for j in rows}
    vis_fit = [j for j in rows if M[j]["occlusion"] != "late"]
    late_fit = [j for j in rows if M[j]["occlusion"] == "late"]
    blocks = sorted({M[j]["block_id"] for j in vis_fit}); rng = np.random.default_rng(0); rng.shuffle(blocks)
    ho = set(blocks[: len(blocks) // 5])
    t0 = time.time()
    # 1) 평균 토큰 · L1 통계 (걸음별)
    sx = torch.zeros(NF, S, D, device=dev, dtype=torch.float64); sy = torch.zeros_like(sx); cnt = torch.zeros(NF, device=dev, dtype=torch.float64)
    cache = {}
    for j in vis_fit:
        p = _LN(torch.from_numpy(np.asarray(P[j], np.float32)).to(dev)); h = torch.from_numpy(np.asarray(H[j, NF:], np.float32)).to(dev)
        mask = torch.tensor([st[j][t] == "vis" for t in range(NF, NT)], device=dev)
        cache[j] = (p.half(), h.half(), mask)
        sx[mask] += p[mask].double(); sy[mask] += h[mask].double(); cnt += mask.double()
    mu_p = (sx / cnt[:, None, None]).float(); mu_h = (sy / cnt[:, None, None]).float()
    print(f"[{objset}] fit videos {len(vis_fit)} · tokens/step {cnt.tolist()} · {time.time() - t0:.0f}s", flush=True)
    # L1 통계 · 공분산
    s1x = torch.zeros(NF, D, device=dev, dtype=torch.float64); s2x = torch.zeros_like(s1x); s1y = torch.zeros_like(s1x); s2y = torch.zeros_like(s1x)
    rx = torch.zeros(NF, device=dev, dtype=torch.float64); ry = torch.zeros_like(rx)
    Cxx = {k: torch.zeros(D, D, device=dev, dtype=torch.float64) for k in ("all", "tr", "ho")}
    Cxy = {k: torch.zeros(D, D, device=dev, dtype=torch.float64) for k in ("all", "tr", "ho")}
    Cyy = {k: torch.zeros(D, D, device=dev, dtype=torch.float64) for k in ("all", "ho")}
    Cxx_t = torch.zeros(NF, D, D, device=dev, dtype=torch.float64); Cxy_t = torch.zeros_like(Cxx_t)
    ntok = defaultdict(float)
    # 섞은 짝: 같은 (위반, 가림, k, 운동) 의 다른 영상 h
    grp = defaultdict(list)
    for j in vis_fit:
        m = M[j]; grp[(m["violation"], m["occlusion"], m["k"], m["motion"])].append(j)
    partner = {}
    for g, js in grp.items():
        for q, j in enumerate(js):
            cand = [x for x in js[q + 1:] + js[:q] if M[x]["block_id"] != M[j]["block_id"]]
            partner[j] = cand[0] if cand else None
    Cxy_s = torch.zeros(D, D, device=dev, dtype=torch.float64); Cxx_s = torch.zeros_like(Cxy_s)
    for j in vis_fit:
        p, h, mask = cache[j]; p, h = p.float()[mask], h.float()[mask]
        Xr = (p - mu_p[mask]); Yr = (h - mu_h[mask])
        s1x[mask] += p.sum(1).double(); s2x[mask] += (p * p).sum(1).double(); s1y[mask] += h.sum(1).double(); s2y[mask] += (h * h).sum(1).double()
        rx[mask] += (Xr * Xr).sum((1, 2)).double(); ry[mask] += (Yr * Yr).sum((1, 2)).double()
        Xf, Yf = Xr.reshape(-1, D), Yr.reshape(-1, D)
        part = "ho" if M[j]["block_id"] in ho else "tr"
        xx, xy, yy = (Xf.T @ Xf).double(), (Xf.T @ Yf).double(), (Yf.T @ Yf).double()
        for k_ in ("all", part):
            Cxx[k_] += xx; Cxy[k_] += xy; ntok[k_] += len(Xf)
        Cyy["all"] += yy
        if part == "ho":
            Cyy["ho"] += yy
        ts = torch.nonzero(mask).flatten()
        for q, tt in enumerate(ts.tolist()):
            Cxx_t[tt] += (Xr[q].T @ Xr[q]).double(); Cxy_t[tt] += (Xr[q].T @ Yr[q]).double()
        pj = partner.get(j)
        if pj is not None:
            p2, h2, m2 = cache[pj]
            both = mask & m2
            if both.any():
                Xs = (cache[j][0].float()[both] - mu_p[both]).reshape(-1, D)
                Ys = (h2.float()[both] - mu_h[both]).reshape(-1, D)
                Cxx_s += (Xs.T @ Xs).double(); Cxy_s += (Xs.T @ Ys).double()
    nt = cnt * S
    m_p = (s1x / nt[:, None]).float(); m_h = (s1y / nt[:, None]).float()
    sd_p = (s2x / nt[:, None] - (s1x / nt[:, None]) ** 2).clamp_min(1e-8).sqrt().float()
    sd_h = (s2y / nt[:, None] - (s1y / nt[:, None]) ** 2).clamp_min(1e-8).sqrt().float()
    gam_t = (ry / rx).sqrt().float(); gam = float((ry.sum() / rx.sum()).sqrt())
    # L2b
    ev, U = torch.linalg.eigh(Cyy["all"]); o = torch.argsort(ev, descending=True); ev, U = ev[o], U[:, o]
    sig_h = (torch.diagonal(U.T @ Cyy["all"] @ U) / ntok["all"]).clamp_min(1e-12).sqrt()
    sig_p = (torch.diagonal(U.T @ Cxx["all"] @ U) / ntok["all"]).clamp_min(1e-12).sqrt()
    M2b = (U * (sig_h / sig_p)[None]) @ U.T
    # L3
    A, Sg, Bt = torch.linalg.svd(Cxy["all"]); R = A @ Bt
    # L4: λ 보류 선택
    tr_scale = float(torch.trace(Cxx["tr"]) / D)
    lam_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    mse = {}
    for lf in lam_grid:
        W = torch.linalg.solve(Cxx["tr"] + lf * tr_scale * torch.eye(D, device=dev, dtype=torch.float64), Cxy["tr"])
        e = torch.trace(W.T @ Cxx["ho"] @ W) - 2 * torch.trace(W.T @ Cxy["ho"]) + torch.trace(Cyy["ho"])
        mse[lf] = float(e / ntok["ho"])
    lf = min(mse, key=mse.get); lam = lf * float(torch.trace(Cxx["all"]) / D)
    W = torch.linalg.solve(Cxx["all"] + lam * torch.eye(D, device=dev, dtype=torch.float64), Cxy["all"])
    Wt = torch.stack([torch.linalg.solve(Cxx_t[t] + lf * float(torch.trace(Cxx_t[t]) / D) * torch.eye(D, device=dev, dtype=torch.float64), Cxy_t[t]) for t in range(NF)])
    Ws = torch.linalg.solve(Cxx_s + lf * float(torch.trace(Cxx_s) / D) * torch.eye(D, device=dev, dtype=torch.float64), Cxy_s)
    # 극분해 W = Up P
    a_, s_, bt_ = torch.linalg.svd(W); Up = a_ @ bt_
    polar = dict(rot_trace_over_D=float(torch.trace(Up) / D), P_eig_median=float(s_.median()), P_eig_p10=float(torch.quantile(s_.float(), 0.1)),
                 P_eig_p90=float(torch.quantile(s_.float(), 0.9)), P_eig_max=float(s_.max()))
    print(f"[{objset}] λ factor {lf} mse {mse} · γ {gam:.3f} γ_t {gam_t.tolist()} · polar {polar} · {time.time() - t0:.0f}s", flush=True)
    # late 판 L3 · L4 (재등장 걸음)
    Cxx_l = torch.zeros(D, D, device=dev, dtype=torch.float64); Cxy_l = torch.zeros_like(Cxx_l); nl = 0
    for j in late_fit:
        mask = torch.tensor([st[j][t] == "vis" for t in range(NF, NT)], device=dev)
        if not mask.any():
            continue
        p = _LN(torch.from_numpy(np.asarray(P[j], np.float32)).to(dev))[mask]; h = torch.from_numpy(np.asarray(H[j, NF:], np.float32)).to(dev)[mask]
        Xf = (p - mu_p[mask]).reshape(-1, D); Yf = (h - mu_h[mask]).reshape(-1, D)
        Cxx_l += (Xf.T @ Xf).double(); Cxy_l += (Xf.T @ Yf).double(); nl += len(Xf)
    Wl = torch.linalg.solve(Cxx_l + lf * float(torch.trace(Cxx_l) / D) * torch.eye(D, device=dev, dtype=torch.float64), Cxy_l)
    Al, _, Btl = torch.linalg.svd(Cxy_l); Rl = Al @ Btl
    # L5: L4 위 잔차 MLP
    W32 = W.float()
    mlp = torch.nn.Sequential(torch.nn.Linear(D, 2 * D), torch.nn.GELU(), torch.nn.Linear(2 * D, D)).to(dev)
    torch.nn.init.zeros_(mlp[2].weight); torch.nn.init.zeros_(mlp[2].bias)
    opt = torch.optim.AdamW(mlp.parameters(), lr=3e-4, weight_decay=0.01)
    toks = []
    for j in vis_fit:
        p, h, mask = cache[j]
        toks.append((p[mask].reshape(-1, D), h[mask].reshape(-1, D), (mu_p[mask]).reshape(-1, D).half(), (mu_h[mask]).reshape(-1, D).half()))
    Xall = torch.cat([a for a, _, _, _ in toks]); Yall = torch.cat([b for _, b, _, _ in toks])
    MPall = torch.cat([c for _, _, c, _ in toks]); MHall = torch.cat([d for _, _, _, d in toks]); del toks
    n = len(Xall); bs = 4096; losses = []
    for ep in range(3):
        perm = torch.randperm(n, device=dev); tot = 0.0
        for s0 in range(0, n, bs):
            ix = perm[s0:s0 + bs]
            x = Xall[ix].float(); y = Yall[ix].float()
            base = MHall[ix].float() + (x - MPall[ix].float()) @ W32
            loss = (base + mlp(x) - y).abs().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += float(loss) * len(ix)
        losses.append(tot / n)
        print(f"[{objset}] L5 epoch {ep + 1} L1 {tot / n:.4f}", flush=True)
    with torch.no_grad():
        l4_l1 = float(np.mean([float(((MHall[s0:s0 + 65536].float() + (Xall[s0:s0 + 65536].float() - MPall[s0:s0 + 65536].float()) @ W32) - Yall[s0:s0 + 65536].float()).abs().mean())
                               for s0 in range(0, n, 65536)]))
    C.MAPS.mkdir(parents=True, exist_ok=True)
    torch.save(dict(mu_p=mu_p.half().cpu(), mu_h=mu_h.half().cpu(), m_p=m_p.cpu(), m_h=m_h.cpu(), sd_p=sd_p.cpu(), sd_h=sd_h.cpu(),
                    gam_t=gam_t.cpu(), gam=gam, M2b=M2b.float().cpu(), R=R.float().cpu(), W=W.float().cpu(), Wt=Wt.float().cpu(),
                    Ws=Ws.float().cpu(), Wl=Wl.float().cpu(), Rl=Rl.float().cpu(), mlp=mlp.state_dict(), lam_factor=lf, lam_mse=mse,
                    polar=polar, n_tok=ntok["all"], n_tok_late=nl, l5_losses=losses, l4_train_l1=l4_l1,
                    sig_ratio=(sig_h / sig_p).float().cpu(), ev=ev.float().cpu()), C.MAPS / f"{objset}.pt")
    print(f"[{objset}] saved maps · L4 train L1 {l4_l1:.4f} · L5 {losses} · {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ eval
def evaluate(objset, gpu, seeds=(0, 1)):
    import torch
    dev = torch.device(f"cuda:{gpu}")
    M = C.meta(); trb = C.train_blocks(); bmap, UV, HID, EXT = C.load_tracks()
    labs, L = C.classes(M, objset); NONE = L["empty"]
    mp = torch.load(C.MAPS / f"{objset}.pt", map_location=dev)
    f32 = lambda k: mp[k].float().to(dev)
    mu_p, mu_h, m_p, m_h, sd_p, sd_h = f32("mu_p"), f32("mu_h"), f32("m_p"), f32("m_h"), f32("sd_p"), f32("sd_h")
    gam_t, gam = f32("gam_t"), mp["gam"]
    M2b, R, W, Wt, Ws, Wl, Rl = f32("M2b"), f32("R"), f32("W"), f32("Wt"), f32("Ws"), f32("Wl"), f32("Rl")
    mlp = torch.nn.Sequential(torch.nn.Linear(D, 2 * D), torch.nn.GELU(), torch.nn.Linear(2 * D, D)).to(dev); mlp.load_state_dict(mp["mlp"]); mlp.eval()
    # E2 의 물체 영역 최선 이동 (릴리즈 · PFT)
    sstar = {}
    for line in open(C.RES / "E2_rows.jsonl"):
        r = json.loads(line)
        if r.get("model") in ("rel", "pft") and "sO_star" in r:
            sstar[(r["model"], r["block"], r["t"])] = tuple(r["sO_star"])

    def shift_map(x, s, base):
        """x (8,256,D) 걸음별 이동 s_t: 결과(c) = x(c + s) — Δp ≈ T_s Δh 를 되돌린다. 빈 칸 = base."""
        out = base.clone()
        for k in range(NF):
            dr, dc = s[k]
            xr = x[k].reshape(G, G, D); o = out[k].reshape(G, G, D)
            r0, r1 = max(0, -dr), G - max(0, dr); c0, c1 = max(0, -dc), G - max(0, dc)
            if r1 > r0 and c1 > c0:
                o[r0:r1, c0:c1] = xr[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        return out

    def maps(pt, s_rel):
        """pt = LN(p) (8,256,D) → {사상 이름: x̂}"""
        Xr = pt - mu_p
        out = {"L0": pt,
               "L1": m_h[:, None] + (sd_h / sd_p)[:, None] * (pt - m_p[:, None]),
               "L2": mu_h + gam_t[:, None, None] * Xr,
               "L2b": mu_h + Xr @ M2b,
               "L3": mu_h + gam * (Xr @ R),
               "L4": mu_h + Xr @ W,
               "L4t": mu_h + torch.einsum("tcd,tde->tce", Xr, Wt),
               "L4s": mu_h + Xr @ Ws,
               "L3late": mu_h + gam * (Xr @ Rl),
               "L4late": mu_h + Xr @ Wl}
        with torch.no_grad():
            out["L5"] = out["L4"] + mlp(pt)
        if s_rel is not None:
            sh = shift_map(pt, s_rel, mu_p)
            out["Lsh"] = sh
            out["L4sh"] = mu_h + (sh - mu_p) @ W
        return out

    # 판독기
    heads = {}
    for seed in seeds:
        for nm in ("h", "p", "h_t8", "h_t11", "h_t15", "p_t8", "p_t11", "p_t15"):
            f = C.HEADS / f"{objset}_{nm}_s{seed}.pt"
            if f.exists():
                clf = C.make_head(len(labs), dev); clf.load_state_dict(torch.load(f, map_location=dev)); clf.eval(); heads[(nm, seed)] = clf

    def read(clf, x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return clf(x.float()).float().argmax(1).cpu().numpy()

    H = np.load(C.TUBE / "h.npy", mmap_mode="r"); P = np.load(C.TUBE / "p.npy", mmap_mode="r")
    PF = np.load(C.TG / "p_relftg.npy", mmap_mode="r"); HI = np.load(C.TG / "h_imp_fut.npy", mmap_mode="r"); Z = np.load(C.TG / "z_late_ctx.npy", mmap_mode="r")
    IM = json.load(open(C.TG / "imp_meta.json"))["meta"]; imp_of = {m["pos_tp"]: q for q, m in enumerate(IM)}
    zm = json.load(open(C.TG / "z_late_meta.json")); zslot = {j: q for q, j in enumerate(zm["tp"])}
    tpi = {(m["block_id"], m["variant"]): j for j, m in enumerate(M)}
    copy = np.load(C.REPO / "z_research/VOBench_v1/exp_results/copy__vobench_v1_vith/copy.npy").mean(1)
    cvid = json.load(open(C.REPO / "z_research/VOBench_v1/exp_results/copy__vobench_v1_vith/meta.json"))["video_id"]; cidx = {v: i for i, v in enumerate(cvid)}

    rows_v = [j for j, m in enumerate(M) if m["objset"] == objset]
    reads = []      # (j, t, map, head, seed, pred)
    scores = []     # 영상별 VoE
    t0 = time.time()
    tens = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(dev)
    for n_, j in enumerate(rows_v):
        m = M[j]; test = m["block_id"] not in trb; late = m["occlusion"] == "late"
        st = C.tub_state(m["samples"]); is_empty = C.label_of(m) == "empty"
        p_raw = tens(P[j]); pt = _LN(p_raw); pf_raw = tens(PF[j]); pft = _LN(pf_raw)
        hf = tens(H[j, NF:]); q = imp_of[j]; hi = tens(HI[q])
        s_rel = [sstar.get(("rel", m["block_id"], t), (0, 0)) for t in range(NF, NT)]
        has_s = all(("rel", m["block_id"], t) in sstar for t in range(NF, NT))
        mm = maps(pt, s_rel if has_s else None)
        mm_pf = {"PFT_L0": pft}
        if has_s and all(("pft", m["block_id"], t) in sstar for t in range(NF, NT)):
            mm_pf["PFT_Lsh"] = shift_map(pft, [sstar[("pft", m["block_id"], t)] for t in range(NF, NT)], mu_p)
        # E6 VoE
        sc = dict(j=j, imp=q, block=m["block_id"], variant=m["variant"], dirn="A" if m["variant"] == "pos_a" else "B", violation=m["violation"],
                  occlusion=m["occlusion"], k=m["k"], motion=m["motion"], test=test,
                  copy_pos=float(copy[cidx[m["video_id"]]]), copy_imp=float(copy[cidx[IM[q]["video_id"]]]))
        allm = dict(mm, **mm_pf, L0raw=p_raw, PFT_L0raw=pf_raw)
        for nm, x in allm.items():
            sc[f"pos_{nm}"] = float((x - hf).abs().mean()); sc[f"imp_{nm}"] = float((x - hi).abs().mean())
        # E7 주입 (late)
        if late:
            bi = bmap[m["block_id"]]; arm = 0 if m["variant"] == "pos_a" else 1
            uv = UV[bi, 0]; ext = EXT[bi, 0]; hid = HID[bi, 0] | (HID[bi, 1] if m["violation"] != "permanence" else HID[bi, 0])
            tl = C.last_visible_ctx(hid)
            other = tpi[(m["block_id"], "pos_b" if arm == 0 else "pos_a")]
            z_self = tens(Z[zslot[j]]); z_oth = tens(Z[zslot[other]])      # 캐시는 이미 LN(z)
            src_cells = C.region(uv[tl], ext[tl])
            r_end, c_end = C.cell_of(uv[tl])
            inj = {}
            for dn in ("d1", "d2", "d3", "d4"):
                x = p_raw.clone()
                for k, t in enumerate(range(NF, NT)):
                    if dn == "d1":
                        cells = C.region(uv[t], ext[t]); x[k, cells] = hf[k, cells]
                    elif dn == "d2":
                        x[k, src_cells] = z_self[tl, src_cells]
                    else:
                        r_t, c_t = C.cell_of(uv[t]); dr, dc = r_t - r_end, c_t - c_end
                        src = z_self if dn == "d3" else z_oth
                        for cc in src_cells:
                            rr, c2 = divmod(cc, G); r2, c3 = rr + dr, c2 + dc
                            if 0 <= r2 < G and 0 <= c3 < G:
                                x[k, r2 * G + c3] = src[tl, cc]
                inj[dn] = x
                sc[f"pos_{dn}"] = float((x - hf).abs().mean()); sc[f"imp_{dn}"] = float((x - hi).abs().mean())
            mm_inj = {f"inj_{k}": _LN(v) for k, v in inj.items()}
        else:
            mm_inj = {}
        scores.append(sc)
        # 판독 (test block 만 — 판독기는 train block 으로 학습)
        if test:
            gt = L[C.label_of(m)]
            readset = dict(mm, **mm_pf, **mm_inj)
            for (nm, seed), clf in heads.items():
                if nm in ("h", "p"):
                    srcs = readset if nm == "h" else {"L0": pt, "PFT_L0": pft}
                    for mn, x in srcs.items():
                        pr = read(clf, x)
                        for k, t in enumerate(range(NF, NT)):
                            reads.append((j, t, mn, nm, seed, int(pr[k]), gt, st[t]))
                    if nm == "h":       # 대조: 사상을 h 에 (h 보임 걸음)
                        hm = maps(hf, None)
                        for mn in ("L1", "L2", "L2b", "L3", "L4", "L4t", "L5", "L4s"):
                            pr = read(clf, hm[mn])
                            for k, t in enumerate(range(NF, NT)):
                                reads.append((j, t, "onH_" + mn, nm, seed, int(pr[k]), gt, st[t]))
                        pr = read(clf, hf)
                        for k, t in enumerate(range(NF, NT)):
                            reads.append((j, t, "H", nm, seed, int(pr[k]), gt, st[t]))
                else:
                    tt = int(nm.split("_t")[1]); k = tt - NF
                    srcs = readset if nm.startswith("h_") else {"L0": pt}
                    for mn, x in srcs.items():
                        pr = read(clf, x[k:k + 1])
                        reads.append((j, tt, mn, nm, seed, int(pr[0]), gt, st[tt]))
        if n_ % 60 == 0:
            print(f"[{objset}] eval {n_}/{len(rows_v)} {time.time() - t0:.0f}s", flush=True)
    C.RES.mkdir(parents=True, exist_ok=True)
    with open(C.RES / f"E6E7_rows_{objset}.jsonl", "w") as f:
        for s in scores:
            f.write(json.dumps(s) + "\n")
    mapnames = sorted({r[2] for r in reads})
    np.savez_compressed(C.RES / f"E4_reads_{objset}.npz", reads=np.array([[r[0], r[1], mapnames.index(r[2]), ["h", "p", "h_t8", "h_t11", "h_t15", "p_t8", "p_t11", "p_t15"].index(r[3]),
                                                                          r[4], r[5], r[6], ["vis", "occ", "mix"].index(r[7])] for r in reads], np.int32),
                        mapnames=np.array(mapnames), classes=np.array(labs), none=NONE)
    print(f"[{objset}] eval done {time.time() - t0:.0f}s · reads {len(reads)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["fit", "eval"])
    ap.add_argument("--objset", default="primitive")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    fit(a.objset, a.gpu) if a.phase == "fit" else evaluate(a.objset, a.gpu)
