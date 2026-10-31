#!/usr/bin/env python3
r"""[X1] R2 E23b — 나머지 칸의 존재 대비는 h 의 전역 존재 서명과 같은 방향인가 (E23 의 [추측] 시험, 계획 밖 추가) (2026-10-30).

permanence block 의 두 가능 영상 (물체 팔 A · 빈 장면 팔 B, 같은 장면 · 같은 궤적) 에서 존재 대비 Δx_t = x_A,t − x_B,t (토큰 LN).
영역: 물체 통로 (마지막 본 자리 ~ 참 자리, transfer_gap_margin.py 와 같은 정의) 와 나머지 칸.
  κ_rest = cos(vec Δp[나머지], vec Δh[나머지]) (미래 8 걸음 · 나머지 칸 · 1280 채널을 펼침) · ρ_rest = ‖Δp[나머지]‖ / ‖Δh[나머지]‖ · 통로판도 같이.
  칸 평균판: 나머지 칸 평균 Δ (걸음별 1280 차원) 끼리 cos — 화면 공통 성분.
대상: late · 이동 · permanence (60 block; PFT 는 test 30) · 보임 조건 · 이동 · permanence (대조). 릴리즈 · PFT · 복사 (LN(z)[t7], late 만).
출력: results/q1q2_r2/E23b_restsig.json

  $P z_research/scripts/analysis/transfer_gap_restsig.py
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_gap_common as C  # noqa: E402
import transfer_gap_margin as MG  # noqa: E402

NF = C.NF


def main():
    cx = MG.Ctx(); M = cx.M; rows = []
    for b, vv in sorted(cx.blocks.items(), key=lambda x: int(x[0])):
        m = M[vv["pos_a"]]
        if m["violation"] != "permanence" or m["motion"] == "flat_static":
            continue
        cat = "late" if m["occlusion"] == "late" else "vis"
        test = b not in cx.trb
        geo = cx.geo(b); parts, regs = MG.masks(geo)
        cor = regs["corridor"]; rest = ~cor
        ja, jb = vv["pos_a"], vv["pos_b"]
        dH = cx.arr(cx.H, ja)[NF:] - cx.arr(cx.H, jb)[NF:]
        preds = {"rel": (MG.LNnp(cx.arr(cx.P, ja)), MG.LNnp(cx.arr(cx.P, jb)))}
        if test:
            preds["pft"] = (MG.LNnp(cx.arr(cx.PF, ja)), MG.LNnp(cx.arr(cx.PF, jb)))
        if cat == "late":
            za, zb = cx.arr(cx.Z, cx.zslot[ja])[7], cx.arr(cx.Z, cx.zslot[jb])[7]
            preds["copy"] = (np.broadcast_to(za, dH.shape), np.broadcast_to(zb, dH.shape))
        r = dict(block=b, cat=cat, objset=m["objset"], test=test)
        for nm, (a, bb) in preds.items():
            dP = a - bb
            for rn, msk in (("rest", rest), ("corridor", cor)):
                X, Y = dP[msk], dH[msk]
                r[f"{nm}_k_{rn}"] = float((X * Y).sum() / (np.linalg.norm(X) * np.linalg.norm(Y) + 1e-12))
                r[f"{nm}_rho_{rn}"] = float(np.linalg.norm(X) / (np.linalg.norm(Y) + 1e-12))
            mx = np.stack([dP[k][rest[k]].mean(0) for k in range(NF)]); my = np.stack([dH[k][rest[k]].mean(0) for k in range(NF)])
            r[f"{nm}_k_restmean"] = float(np.mean([(x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12) for x, y in zip(mx, my)]))
        r["h_rest_energy_frac"] = float((dH[rest] ** 2).sum() / (dH ** 2).sum())
        rows.append(r)
    out = {"definition": __doc__, "summary": {}}
    for cat in ("late", "vis"):
        for split in ("all", "test"):
            sel = [r for r in rows if r["cat"] == cat and (split == "all" or r["test"])]
            g = [r["block"] for r in sel]
            d = dict(n=len(sel), h_rest_energy_frac=C.bootstrap_mean([r["h_rest_energy_frac"] for r in sel], g))
            for k in sorted({k for r in sel for k in r if k.startswith(("rel_", "pft_", "copy_"))}):
                v = [(r[k], r["block"]) for r in sel if k in r]
                d[k] = C.bootstrap_mean([a for a, _ in v], [x for _, x in v])
            out["summary"][f"{cat}|{split}"] = d
            print(cat, split, {k: round(v["mean"], 3) for k, v in d.items() if isinstance(v, dict)}, flush=True)
    json.dump(out, open(MG.OUT / "E23b_restsig.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
