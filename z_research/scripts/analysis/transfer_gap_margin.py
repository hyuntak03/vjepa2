#!/usr/bin/env python3
r"""[X1] R2 E22 채점 마진 분해 + E23 영역 교환 — 사라짐 판정은 어느 칸에서 정해지나, PFT 이득은 물체 자리인가 화면 전체인가 (H2-6) (2026-10-30).

기호: p = predictor 원값 (8 걸음 × 256 칸 × 1280), h = LN(target_encoder(32 장)) 의 미래 t8–15, 짝 (pos, imp) 은 같은 문맥 → 같은 p.
  s(p; h) = mean_{t,c,d} |p − h| (표준 채점, 전체 토큰). 채점 마진 m = (s(imp) − s(pos)) / s(pos). m > 0 이면 정답.
  칸 기여 e_{t,c} = Σ_d (|p − h_imp| − |p − h_pos|) → m = Σ_R m_R, m_R = Σ_{t, c∈R_t} e_{t,c} / (N · s(pos)), N = 8·256·1280.
영역 (block 의 팔 A 궤적, label 에서; t끝 = 문맥 안 마지막으로 보인 튜블릿):
  O참(t) = 참 물체 외곽 상자 + 1 칸 · O끝 = t끝 자리의 같은 영역 · 통로(t) = ∪_{τ=t끝..t} 영역(τ) (물체가 지나는 칸, 두 영역 포함)
  E22 분할: R1 = O참 · R2 = O끝 \ O참 · R3 = 통로 \ (O참 ∪ O끝) · R4 = 나머지 칸
예측 (E22): rel (릴리즈 원값) · rel_ln (LN(p)) · pft (VOBench PFT 원값, test block) · copy (LN(z)[t7] 를 8 걸음에 복사) ·
  d1 (rel 의 O참 칸을 그 영상 자신의 참 h 로 — 정답 상한) · d3 (마지막 보인 문맥 z 의 물체 칸을 O참(t) 로 옮김 — 신탁 자리, R1 E7 과 같은 구성) ·
  s2g (μ^p + 2γ_t (p̃ − μ^p), μ^p · γ_t = E4 맞춤값 (보임 조건 train block))
E23 영역 교환 (test block): mixP(R) = R 칸은 pft, 나머지는 rel / mixR(R) = R 칸은 rel, 나머지는 pft. R ∈ {O참, O참+1칸, 통로, 나머지, 전부}.
대상: late · 이동 · permanence (A · B) · late · 이동 · 모양/색 (A · B) · 보임 조건 · 이동 · permanence (대조) / E23 은 + 보임 조건 test 전부 (부작용).
판정 (PLAN_R2 §5): E22 — Δm = m(pft) − m(rel) 의 영역 몫 (평균 Δm_R / 평균 Δm). 통로 (R1+R2+R3) ≥ 60 % → '물체 자리', 나머지 ≥ 60 % → '화면 전체'.
  E23 — A(통로만 PFT) ≥ 80 & A(나머지만 PFT) ≤ 30 → 물체 자리 (운반) · A(나머지만 PFT) ≥ 70 → 화면 전체 · 둘 다 30–70 → 나눠 갖는다.
⚠️ 캐시 (bfloat16) 채점은 공식과 짝 판정 92.6 % 일치 (R1 G0) → 같은 캐시 안의 비교만 읽는다.
출력: results/q1q2_r2/E22_margin.json · E23_swapregion.json · E22_rows.jsonl

  sbatch ... --wrap "$P z_research/scripts/analysis/transfer_gap_margin.py [--limit 6]"
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_gap_common as C  # noqa: E402

NT, NF, S, D, G = C.NT, C.NF, C.S, C.D, C.GRID
OUT = C.REPO / "z_analysis/transfer_gap/results/q1q2_r2"
N_ALL = NF * S * D


class Ctx:
    """공통 자원 — 배열 mmap · 메타 · 궤적 · 맞춤값."""

    def __init__(self):
        import torch
        self.M = C.meta(); self.trb = C.train_blocks(); self.bmap, self.UV, self.HID, self.EXT = C.load_tracks()
        self.H = np.load(C.TUBE / "h.npy", mmap_mode="r"); self.P = np.load(C.TUBE / "p.npy", mmap_mode="r")
        self.PF = np.load(C.TG / "p_relftg.npy", mmap_mode="r"); self.HI = np.load(C.TG / "h_imp_fut.npy", mmap_mode="r")
        self.Z = np.load(C.TG / "z_late_ctx.npy", mmap_mode="r")
        self.IM = json.load(open(C.TG / "imp_meta.json"))["meta"]; self.imp_of = {m["pos_tp"]: q for q, m in enumerate(self.IM)}
        zm = json.load(open(C.TG / "z_late_meta.json")); self.zslot = {j: q for q, j in enumerate(zm["tp"])}
        self.blocks = defaultdict(dict)
        for j, m in enumerate(self.M):
            self.blocks[m["block_id"]][m["variant"]] = j
        self.maps = {}
        for o in ("primitive", "realistic"):
            mp = torch.load(C.MAPS / f"{o}.pt", map_location="cpu")
            self.maps[o] = dict(mu_p=mp["mu_p"].float().numpy(), mu_h=mp["mu_h"].float().numpy(), gam_t=mp["gam_t"].float().numpy())

    def arr(self, a, j):
        return np.asarray(a[j], np.float32)

    def geo(self, b):
        """block b 의 팔 A 궤적 영역."""
        m = self.M[self.blocks[b]["pos_a"]]
        bi = self.bmap[b]; uv, ext = self.UV[bi, 0], self.EXT[bi, 0]
        hid = self.HID[bi, 0] | (self.HID[bi, 1] if m["violation"] != "permanence" else self.HID[bi, 0])
        tl = C.last_visible_ctx(hid)
        Ot = [set(C.region(uv[t], ext[t])) for t in range(NF, NT)]
        Ot1 = [dilate(o) for o in Ot]
        Oe = set(C.region(uv[tl], ext[tl]))
        cor = []
        for t in range(NF, NT):
            s_ = set()
            for tau in range(tl, t + 1):
                s_ |= set(C.region(uv[tau], ext[tau]))
            cor.append(s_)
        return dict(uv=uv, ext=ext, hid=hid, tl=tl, Ot=Ot, Ot1=Ot1, Oe=Oe, cor=cor)


def dilate(cells):
    out = set()
    for c in cells:
        r, cc = divmod(c, G)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if 0 <= r + dr < G and 0 <= cc + dc < G:
                    out.add((r + dr) * G + cc + dc)
    return out


def LNnp(x):
    mu = x.mean(-1, keepdims=True); sd = x.std(-1, keepdims=True)
    return (x - mu) / np.sqrt(sd ** 2 + 1e-5)


def masks(geo):
    """E22 분할 마스크 (8, 256) bool × 4 + E23 영역."""
    R1 = np.zeros((NF, S), bool); R2 = np.zeros_like(R1); R3 = np.zeros_like(R1); T1 = np.zeros_like(R1); COR = np.zeros_like(R1)
    for k in range(NF):
        R1[k, list(geo["Ot"][k])] = True
        R2[k, list(geo["Oe"] - geo["Ot"][k])] = True
        R3[k, list(geo["cor"][k] - geo["Ot"][k] - geo["Oe"])] = True
        T1[k, list(geo["Ot1"][k])] = True
        COR[k, list(geo["cor"][k] | geo["Oe"])] = True
    R4 = ~(R1 | R2 | R3)
    return dict(R1=R1, R2=R2, R3=R3, R4=R4), dict(Otrue=R1, Otrue1=T1, corridor=COR, rest=~COR, all=np.ones_like(R1))


def margin(x, hp, hi, parts):
    """x, hp, hi (8,256,D) → m, {R: m_R}."""
    e = (np.abs(x - hi) - np.abs(x - hp)).sum(-1)          # (8,256)
    sp = float(np.abs(x - hp).mean())
    m = float(e.sum() / (N_ALL * sp))
    return m, {k: float(e[v].sum() / (N_ALL * sp)) for k, v in parts.items()}, sp


def preds_for(cx, j, geo, with_pft):
    m = cx.M[j]; o = m["objset"]; mp = cx.maps[o]
    p = cx.arr(cx.P, j); pt = LNnp(p)
    out = {"rel": p, "rel_ln": pt,
           "s2g": mp["mu_p"] + 2 * mp["gam_t"][:, None, None] * (pt - mp["mu_p"])}
    if with_pft:
        out["pft"] = cx.arr(cx.PF, j)
    if m["occlusion"] == "late" and j in cx.zslot:
        z = cx.arr(cx.Z, cx.zslot[j])
        out["copy"] = np.broadcast_to(z[7], (NF, S, D)).copy()
        hf = cx.arr(cx.H, j)[NF:]
        tl = geo["tl"]; uv = geo["uv"]
        src_cells = C.region(uv[tl], geo["ext"][tl]); r_end, c_end = C.cell_of(uv[tl])
        d1 = p.copy(); d3 = p.copy()
        for k, t in enumerate(range(NF, NT)):
            cells = list(geo["Ot"][k]); d1[k, cells] = hf[k, cells]
            r_t, c_t = C.cell_of(uv[t]); dr, dc = r_t - r_end, c_t - c_end
            for cc in src_cells:
                rr, c2 = divmod(cc, G); r2, c3 = rr + dr, c2 + dc
                if 0 <= r2 < G and 0 <= c3 < G:
                    d3[k, r2 * G + c3] = z[tl, cc]
        out["d1"] = d1; out["d3"] = d3
    return out


def main(limit):
    cx = Ctx()
    M = cx.M
    sel = []
    for b, vv in cx.blocks.items():
        m = M[vv["pos_a"]]
        if m["motion"] == "flat_static":
            cat = None
            if m["occlusion"] != "late":
                cat = "vis_static"
        elif m["occlusion"] == "late":
            cat = "late_move_" + ("perm" if m["violation"] == "permanence" else "id")
        else:
            cat = "vis_move_" + ("perm" if m["violation"] == "permanence" else "id")
        if cat is None:
            continue
        sel.append((b, cat))
    sel.sort(key=lambda x: int(x[0]))
    if limit:
        sel = [s for s in sel if s[1] == "late_move_perm"][:limit] + [s for s in sel if s[1] == "vis_move_id"][:2]
    rows22, rows23 = [], []
    t0 = time.time()
    for n_, (b, cat) in enumerate(sel):
        test = b not in cx.trb
        geo = cx.geo(b); parts, regs = masks(geo)
        do22 = cat in ("late_move_perm", "late_move_id", "vis_move_perm")
        if not do22 and not test:
            continue
        for var in ("pos_a", "pos_b"):
            j = cx.blocks[b][var]; m = M[j]
            hp = cx.arr(cx.H, j)[NF:]; hi = cx.arr(cx.HI, cx.imp_of[j])
            pr = preds_for(cx, j, geo, with_pft=test)
            base = dict(block=b, cat=cat, objset=m["objset"], violation=m["violation"], occlusion=m["occlusion"], k=m["k"], motion=m["motion"],
                        dirn="A" if var == "pos_a" else "B", test=test, j=j)
            if do22:
                r = dict(base)
                for nm, x in pr.items():
                    mm, mR, sp = margin(x, hp, hi, parts)
                    r[f"m_{nm}"] = mm; r[f"mR_{nm}"] = mR
                r["n_cells"] = {k: int(v.sum()) for k, v in parts.items()}
                rows22.append(r)
            if test:
                r = dict(base)
                rel, pft = pr["rel"], pr["pft"]
                sc = lambda x: (float(np.abs(x - hi).mean()) > float(np.abs(x - hp).mean()))
                r["rel"] = sc(rel); r["pft"] = sc(pft)
                for rn, msk in regs.items():
                    xm = rel.copy(); xm[msk] = pft[msk]; r[f"mixP_{rn}"] = sc(xm)
                    xr = pft.copy(); xr[msk] = rel[msk]; r[f"mixR_{rn}"] = sc(xr)
                r["n_cells"] = {k: int(v.sum()) for k, v in regs.items()}
                rows23.append(r)
        if n_ % 40 == 0:
            print(f"{n_}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
    return rows22, rows23


def boot_ratio(num, den, grp, n=2000, seed=0):
    num, den, grp = np.asarray(num, float), np.asarray(den, float), np.asarray(grp)
    ub = np.unique(grp); idx = {b: np.where(grp == b)[0] for b in ub}
    sn = np.array([num[idx[b]].sum() for b in ub]); sd = np.array([den[idx[b]].sum() for b in ub])
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n):
        s = rng.integers(0, len(ub), len(ub)); bs.append(sn[s].sum() / sd[s].sum())
    return dict(ratio=float(num.sum() / den.sum()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(len(num)))


def summarize(rows22, rows23):
    out22, out23 = {}, {}
    RN = ["R1", "R2", "R3", "R4"]
    for cat in ("late_move_perm", "late_move_id", "vis_move_perm"):
        for o in ("primitive", "realistic", "both"):
            for dirn in ("A", "B"):
                for split in ("all", "test"):
                    sel = [r for r in rows22 if r["cat"] == cat and (o == "both" or r["objset"] == o) and r["dirn"] == dirn and (split == "all" or r["test"])]
                    if not sel:
                        continue
                    grp = [r["block"] for r in sel]
                    d = dict(n=len(sel))
                    for nm in ("rel", "rel_ln", "pft", "copy", "d1", "d3", "s2g"):
                        v = [r for r in sel if f"m_{nm}" in r]
                        if not v:
                            continue
                        g = [r["block"] for r in v]
                        mm = np.array([r[f"m_{nm}"] for r in v])
                        e = dict(acc=C.bootstrap_mean(100 * (mm > 0), g), m_median=float(np.median(mm)), m_mean=C.bootstrap_mean(mm, g))
                        for R in RN:
                            vals = np.array([r[f"mR_{nm}"][R] for r in v])
                            e[f"{R}_median"] = float(np.median(vals)); e[f"{R}_mean"] = float(vals.mean())
                        # O참 만으로 부호가 정해지는 짝: 부호가 같고 |m_R1| > |나머지 합|
                        r1 = np.array([r[f"mR_{nm}"]["R1"] for r in v])
                        e["R1_decides_frac"] = float(np.mean((np.sign(r1) == np.sign(mm)) & (np.abs(r1) > np.abs(mm - r1))))
                        e["corridor_mean"] = float(np.mean([r[f"mR_{nm}"]["R1"] + r[f"mR_{nm}"]["R2"] + r[f"mR_{nm}"]["R3"] for r in v]))
                        d[nm] = e
                    # PFT − 릴리즈 마진 변화의 영역 몫 (test)
                    v = [r for r in sel if "m_pft" in r]
                    if v:
                        g = [r["block"] for r in v]
                        dm = np.array([r["m_pft"] - r["m_rel"] for r in v])
                        sh = {}
                        for R in RN:
                            dR = np.array([r["mR_pft"][R] - r["mR_rel"][R] for r in v])
                            sh[R] = boot_ratio(dR, dm, g)
                        dcor = np.array([sum(r["mR_pft"][R] - r["mR_rel"][R] for R in ("R1", "R2", "R3")) for r in v])
                        sh["corridor"] = boot_ratio(dcor, dm, g)
                        d["delta_pft_minus_rel"] = dict(dm_mean=C.bootstrap_mean(dm, g), share=sh)
                    out22[f"{cat}|{o}|{dirn}|{split}"] = d
    keys = None
    for cat_name, f in (("late_move_perm", lambda r: r["cat"] == "late_move_perm"), ("late_move_id", lambda r: r["cat"] == "late_move_id"),
                        ("vis_all", lambda r: r["occlusion"] != "late")):
        for o in ("primitive", "realistic", "both"):
            for dirn in ("A", "B", "AB"):
                sel = [r for r in rows23 if f(r) and (o == "both" or r["objset"] == o) and (dirn == "AB" or r["dirn"] == dirn)]
                if not sel:
                    continue
                keys = [k for k in sel[0] if k in ("rel", "pft") or k.startswith(("mixP_", "mixR_"))]
                g = [r["block"] for r in sel]
                out23[f"{cat_name}|{o}|{dirn}"] = {k: C.bootstrap_mean([100 * float(r[k]) for r in sel], g) for k in keys} | {"n": len(sel)}
    # 판정
    v22 = {}
    for o in ("both", "primitive", "realistic"):
        d = out22.get(f"late_move_perm|{o}|A|test", {}).get("delta_pft_minus_rel")
        if d:
            cor = d["share"]["corridor"]["ratio"]; rest = d["share"]["R4"]["ratio"]
            v22[o] = dict(corridor_share=cor, rest_share=rest,
                          verdict="물체 자리" if cor >= 0.6 else "화면 전체" if rest >= 0.6 else "나눠 갖는다")
    v23 = {}
    for o in ("both", "primitive", "realistic"):
        d = out23.get(f"late_move_perm|{o}|A")
        if d:
            a_cor, a_rest = d["mixP_corridor"]["mean"], d["mixP_rest"]["mean"]
            v23[o] = dict(A_corridor_only_pft=a_cor, A_rest_only_pft=a_rest,
                          verdict="둘 다 충분 (중복 — 어느 쪽 칸만 PFT 여도 뒤집힘)" if (a_cor >= 80 and a_rest >= 70) else "물체 자리 (운반)" if (a_cor >= 80 and a_rest <= 30) else "화면 전체" if a_rest >= 70 else
                          ("나눠 갖는다" if 30 <= a_cor <= 70 and 30 <= a_rest <= 70 else "판정 밖"))
    return out22, out23, v22, v23


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    r22, r23 = main(a.limit)
    o22, o23, v22, v23 = summarize(r22, r23)
    OUT.mkdir(parents=True, exist_ok=True)
    suf = f"_smoke{a.limit}" if a.limit else ""
    json.dump(dict(definition=__doc__, verdict=v22, summary=o22), open(OUT / f"E22_margin{suf}.json", "w"), indent=1, ensure_ascii=False)
    json.dump(dict(definition=__doc__, verdict=v23, summary=o23), open(OUT / f"E23_swapregion{suf}.json", "w"), indent=1, ensure_ascii=False)
    with open(OUT / f"E22_rows{suf}.jsonl", "w") as f:
        for r in r22:
            f.write(json.dumps(r) + "\n")
    with open(OUT / f"E23_rows{suf}.jsonl", "w") as f:
        for r in r23:
            f.write(json.dumps(r) + "\n")
    print("E22 verdict", v22, "\nE23 verdict", v23, f"\n{time.time() - t0:.0f}s")
