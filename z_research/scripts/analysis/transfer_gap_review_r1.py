#!/usr/bin/env python3
"""[R] R1 적대 검증 — E4 'L2 = 크기' 판정의 분해 (2026-10-30, 검토 에이전트).

왜: E4 의 L2 사상은 x̂ = μ^h_{t,c} + γ_t (p̃ − μ^p_{t,c}) 다. 즉 '걸음별 스칼라 γ_t' 하나가 아니라
    (a) 칸 · 걸음마다의 평균 토큰 바꿔치기 μ^p → μ^h (8 × 256 × 1280 값) 와 (b) 스칼라 γ_t 를 함께 건다.
    [X1] 은 회복 (R 0.62 / 0.58) 을 '크기' 로 읽었다. (a) 만으로 회복되면 '크기' 가 아니라 '공통 성분 (평균 토큰) 의 차이' 다.
무엇: S2 (보임 조건 · 이동 · t11–15 · 물체 보임 · test block) 에서 고정 h 판독기 (E3, 시드 0 · 1) 정확도를
    L0      p̃
    L2      μ^h + γ_t (p̃ − μ^p)          ([X1] 재현 확인용)
    Lmu     μ^h + (p̃ − μ^p)              평균 토큰만 바꿈 (γ = 1)
    Lgam    μ^p + γ_t (p̃ − μ^p)          스칼라만 (p 자기 평균 유지)
    Lgam2   μ^p + 2γ_t (p̃ − μ^p)         스칼라 두 배 (단조성)
    Lself   m + γ_t (p̃ − m), m = 그 영상 · 걸음의 256 칸 평균  (맞춘 평균 없음, γ_t 만)
  회복률 R = (acc_M − acc_L0)/(acc_천장 − acc_L0), 천장 = [X1] 의 같은 표본 p 판독기 L0 판독 (E4_reads 에서 읽음). block bootstrap 2000.
  CPU (float32) 로 돌린다 — [X1] 은 GPU bf16 autocast. L0 · L2 가 [X1] 값과 맞는지 먼저 본다.
출력: z_analysis/transfer_gap/results/review_r1/L2_decomp.json

  sbatch -p batch_vll -w vll5 --gres=gpu:0 --cpus-per-task=12 --mem=24G -o z_analysis/transfer_gap/logs/%x_%j.out -J rvL2 \
    --wrap "$P z_research/scripts/analysis/transfer_gap_review_r1.py"
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_gap_common as C  # noqa: E402
import transfer_gap_report as RP  # noqa: E402

NF, NT = C.NF, C.NT
OUT = C.REPO / "z_analysis/transfer_gap/results/review_r1"


def main():
    import torch
    torch.set_num_threads(12)
    M = C.meta(); trb = C.train_blocks()
    P = np.load(C.TUBE / "p.npy", mmap_mode="r")
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
        # 천장: [X1] E4_reads 의 p 판독기 L0 (시드 평균)
        z = np.load(C.RES / f"E4_reads_{objset}.npz", allow_pickle=True)
        maps_x1 = list(z["mapnames"]); hn = RP.HEADN
        ceil, x1acc = {}, {}
        for j, t, mi, hi, seed, pr, gt, stt in z["reads"]:
            if maps_x1[mi] == "L0" and hn[hi] == "p":
                ceil.setdefault((int(j), int(t)), []).append(pr == gt)
            if maps_x1[mi] in ("L0", "L2") and hn[hi] == "h":
                x1acc.setdefault((maps_x1[mi], int(j), int(t)), []).append(pr == gt)
        sel = [j for j, m in enumerate(M) if m["objset"] == objset and m["block_id"] not in trb and C.label_of(m) != "empty"
               and m["occlusion"] != "late" and m["motion"] != "flat_static"]
        rows = []   # (j, t, block, {map: acc_seed_mean})
        for n_, j in enumerate(sel):
            m = M[j]; st = C.tub_state(m["samples"]); gt = L[C.label_of(m)]
            ks = [k for k, t in enumerate(range(NF, NT)) if t >= 11 and st[t] == "vis"]
            if not ks:
                continue
            p = torch.from_numpy(np.asarray(P[j], np.float32))
            pt = torch.nn.functional.layer_norm(p, (p.shape[-1],))
            Xr = pt - mu_p
            fm = pt.mean(1, keepdim=True)
            g = gam_t[:, None, None]
            mm = {"L0": pt, "L2": mu_h + g * Xr, "Lmu": mu_h + Xr, "Lgam": mu_p + g * Xr, "Lgam2": mu_p + 2 * g * Xr,
                  "Lself": fm + g * (pt - fm)}
            acc = {}
            with torch.no_grad():
                for nm, x in mm.items():
                    pr = np.stack([h(x[ks]).argmax(1).numpy() for h in heads])     # (2, len(ks))
                    acc[nm] = (pr == gt).mean(0)
            for q, k in enumerate(ks):
                rows.append((j, NF + k, m["block_id"], {nm: float(a[q]) for nm, a in acc.items()}))
            if n_ % 40 == 0:
                print(f"[{objset}] {n_}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
        keys = [(j, t) for j, t, _, _ in rows]
        grp = [b for _, _, b, _ in rows]
        cvec = np.array([np.mean(ceil[k]) if k in ceil else np.nan for k in keys])
        base = np.array([r[3]["L0"] for r in rows])
        cell = {"n": len(rows), "n_block": len(set(grp)), "ceiling_p_reader": 100 * float(np.nanmean(cvec)),
                "x1_L0": 100 * float(np.mean([np.mean(x1acc[("L0",) + k]) for k in keys if ("L0",) + k in x1acc])),
                "x1_L2": 100 * float(np.mean([np.mean(x1acc[("L2",) + k]) for k in keys if ("L2",) + k in x1acc])),
                "maps": {}}
        for nm in ("L0", "L2", "Lmu", "Lgam", "Lgam2", "Lself"):
            a = np.array([r[3][nm] for r in rows])
            e = dict(acc=C.bootstrap_mean(100 * a, grp))
            if not np.isnan(cvec).any():
                e["R"] = RP.boot_ratio(a, base, cvec, grp)
            cell["maps"][nm] = e
        res["objsets"][objset] = cell
        print(objset, json.dumps({k: (round(v["acc"]["mean"], 1), round(v.get("R", {}).get("R") or 0, 3)) for k, v in cell["maps"].items()}),
              "x1 L0/L2", round(cell["x1_L0"], 1), round(cell["x1_L2"], 1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "L2_decomp.json", "w"), indent=1, ensure_ascii=False)
    print("→", OUT / "L2_decomp.json", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
