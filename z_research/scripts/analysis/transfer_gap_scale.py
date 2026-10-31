#!/usr/bin/env python3
r"""[X1] R2 E19 배율 곡선 — 대비를 몇 배 키우면 h 판독기가 얼마나 되살아나나, 포화점은 어디인가 (H1-S) (2026-10-30).

transfer_gap_review_r1.py ([R] R1) 의 확장. 고정 h 판독기 (E3, 시드 0 · 1, attention 풀링, 위치 정보 없음) 를 CPU float32 로 건다.
사상 (p̃ = LN(p), μ^p_{t,c} · γ_t = E4 맞춤값 (보임 조건 train block), m_i = 그 영상 · 걸음의 256 칸 평균):
  Pk    μ^p + k·γ_t (p̃ − μ^p)          (p 자기 평균 유지, 배율만)      k ∈ {0.5, 1, 2, 3, 4, 8}
  Sk    m_i + k·γ_t (p̃ − m_i)          (영상 자기 평균판, 맞춘 평균 없음)
  Hk    μ^h + k (h − μ^h)                (같은 꼴을 h 에 — 판독기의 배율 허용 한계)
표본 (test block, 물체 있는 영상, 물체 보이는 걸음):
  S2   보임 조건 · 이동 · t11–15 (릴리즈 p)     · S2pft 같은 표본의 PFT p · S4 late · 이동 · 다시 보이는 걸음 (릴리즈 p) · Hvis 보임 조건 h t8–15
회복률 R = (acc_M − acc_L0) / (천장 − acc_L0), 천장 = 같은 표본의 p 판독기 L0 (E4_reads 의 'p' 판독기 · 'L0' / 'PFT_L0'). block bootstrap 2000.
판정: 포화 R ≥ 0.9 → 크기가 전부 · ≈ 0.6 → 나머지는 크기가 아님 (E17 로). Hk 가 k ≥ 4 에서 90 % 아래면 포화점은 판독기의 배율 허용 한계로 적는다.
출력: results/q1q2_r2/E19_scale.json

  sbatch ... --wrap "$P z_research/scripts/analysis/transfer_gap_scale.py"
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_gap_common as C  # noqa: E402
import transfer_gap_report as RP  # noqa: E402

NF, NT = C.NF, C.NT
OUT = C.REPO / "z_analysis/transfer_gap/results/q1q2_r2"
KS = (0.5, 1, 2, 3, 4, 8)


def main():
    import torch
    torch.set_num_threads(12)
    M = C.meta(); trb = C.train_blocks()
    P = np.load(C.TUBE / "p.npy", mmap_mode="r"); PF = np.load(C.TG / "p_relftg.npy", mmap_mode="r"); H = np.load(C.TUBE / "h.npy", mmap_mode="r")
    res = {"definition": __doc__, "objsets": {}}
    t0 = time.time()
    for objset in ("primitive", "realistic"):
        labs, L = C.classes(M, objset)
        mp = torch.load(C.MAPS / f"{objset}.pt", map_location="cpu")
        mu_p, mu_h, gam_t = mp["mu_p"].float(), mp["mu_h"].float(), mp["gam_t"].float()
        heads = []
        for s in (0, 1):
            clf = C.make_head(len(labs), "cpu"); clf.load_state_dict(torch.load(C.HEADS / f"{objset}_h_s{s}.pt", map_location="cpu")); clf.eval()
            heads.append(clf)
        z = np.load(C.RES / f"E4_reads_{objset}.npz", allow_pickle=True)
        maps_x1 = list(z["mapnames"]); hn = RP.HEADN
        ceil = {}
        for j, t, mi, hi, seed, pr, gt, stt in z["reads"]:
            if hn[hi] == "p" and maps_x1[mi] in ("L0", "PFT_L0"):
                ceil.setdefault((maps_x1[mi], int(j), int(t)), []).append(pr == gt)
        sel = [j for j, m in enumerate(M) if m["objset"] == objset and m["block_id"] not in trb and C.label_of(m) != "empty" and m["motion"] != "flat_static"]
        rows = {"S2": [], "S2pft": [], "S4": [], "Hvis": []}
        for n_, j in enumerate(sel):
            m = M[j]; st = C.tub_state(m["samples"]); gt = L[C.label_of(m)]; late = m["occlusion"] == "late"
            if late:
                ks = [k for k, t in enumerate(range(NF, NT)) if st[t] == "vis"]
                scen = [("S4", P, "L0")]
            else:
                ks = [k for k, t in enumerate(range(NF, NT)) if t >= 11 and st[t] == "vis"]
                scen = [("S2", P, "L0"), ("S2pft", PF, "PFT_L0")]
            if not ks:
                continue
            for sc, arr, cname in scen:
                pt = torch.nn.functional.layer_norm(torch.from_numpy(np.asarray(arr[j], np.float32)), (C.D,))
                fm = pt.mean(1, keepdim=True); g = gam_t[:, None, None]
                mm = {"L0": pt}
                for k in KS:
                    mm[f"P{k}"] = mu_p + k * g * (pt - mu_p)
                    mm[f"S{k}"] = fm + k * g * (pt - fm)
                with torch.no_grad():
                    acc = {nm: np.stack([h(x[ks]).argmax(1).numpy() for h in heads]) for nm, x in mm.items()}
                for q, k in enumerate(ks):
                    c = ceil.get((cname, j, NF + k))
                    rows[sc].append((m["block_id"], {nm: float((a[:, q] == gt).mean()) for nm, a in acc.items()},
                                     float(np.mean(c)) if c else np.nan,
                                     {nm: float(np.mean([labs[x].split("/")[0] == labs[gt].split("/")[0] for x in a[:, q]])) for nm, a in acc.items()}))
            if not late:
                kv = [k for k, t in enumerate(range(NF, NT)) if st[t] == "vis"]
                h = torch.from_numpy(np.asarray(H[j, NF:], np.float32))
                mm = {"L0": h}
                for k in KS:
                    mm[f"H{k}"] = mu_h + k * (h - mu_h)
                with torch.no_grad():
                    acc = {nm: np.stack([hd(x[kv]).argmax(1).numpy() for hd in heads]) for nm, x in mm.items()}
                for q, k in enumerate(kv):
                    rows["Hvis"].append((m["block_id"], {nm: float((a[:, q] == gt).mean()) for nm, a in acc.items()}, np.nan, {}))
            if n_ % 50 == 0:
                print(f"[{objset}] {n_}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
        cell = {}
        for sc, R in rows.items():
            if not R:
                continue
            grp = [r[0] for r in R]; base = np.array([r[1]["L0"] for r in R]); cv = np.array([r[2] for r in R])
            d = dict(n=len(R), n_block=len(set(grp)), maps={})
            if not np.isnan(cv).all():
                d["ceiling"] = 100 * float(np.nanmean(cv))
            for nm in R[0][1]:
                a = np.array([r[1][nm] for r in R])
                e = dict(acc=C.bootstrap_mean(100 * a, grp))
                if R[0][3]:
                    e["shape"] = C.bootstrap_mean([100 * r[3][nm] for r in R], grp)
                if not np.isnan(cv).any() and sc in ("S2", "S2pft"):
                    e["R"] = RP.boot_ratio(a, base, cv, grp)
                d["maps"][nm] = e
            cell[sc] = d
            print(objset, sc, {nm: (round(v["acc"]["mean"], 1), round((v.get("R") or {}).get("R") or 0, 3)) for nm, v in d["maps"].items()}, flush=True)
        res["objsets"][objset] = cell
    # 판정
    ver = {}
    for o, cell in res["objsets"].items():
        if "S2" not in cell:
            continue
        mp_ = cell["S2"]["maps"]
        Rp = {k: mp_[f"P{k}"].get("R", {}).get("R") for k in KS}
        Rs = {k: mp_[f"S{k}"].get("R", {}).get("R") for k in KS}
        hk = {k: cell["Hvis"]["maps"][f"H{k}"]["acc"]["mean"] for k in KS} if "Hvis" in cell else {}
        best = max(v for v in list(Rp.values()) + list(Rs.values()) if v is not None)
        ver[o] = dict(R_P=Rp, R_S=Rs, H_acc=hk, R_max=best,
                      verdict="크기가 전부" if best >= 0.9 else "크기 + 나머지 (E17)" if best >= 0.5 else "크기 몫 작음",
                      reader_limit=[k for k, v in hk.items() if k >= 4 and v < 90])
    res["verdict"] = ver
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "E19_scale.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(ver, ensure_ascii=False), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
