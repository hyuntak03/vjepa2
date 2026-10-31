#!/usr/bin/env python3
"""[X1] E2c late 메아리 검사 — late 예측의 쌍 차이 Δp_t 는 어느 encoder 대비를 닮았나 (2026-10-30, E8 결과로 추가).

E8 에서 late 의 p 는 가려진 마지막 문맥 튜블릿 (t7) 의 토큰을 바꾸면 B 쪽으로 26–38 % 움직였다 (마지막 '보인' 튜블릿은 10 %).
그렇다면 late Δp 는 참 미래의 보이는 대비 Δh_t 보다 가려진 문맥 대비 Δz_7 (문맥 encoder 가 가린 물체에 실은 꼴) 을 닮았나.
  Δp_t = LN(p_A,t) − LN(p_B,t) (릴리즈 · PFT), Δh_t = h_A,t − h_B,t, Δz_τ = LN(z_A)_τ − LN(z_B)_τ (τ = 7 가려진 마지막 · τ = t끝 마지막으로 보인)
  κ(X, Y) = max_{s∈[−9,9]×[−3,3]} cos(X, T_s Y) (전 칸) · κ_O(X, Y; O) = max_s cos(X[O+s], Y[O]) (Y 의 물체 영역 O 를 X 에서 찾음)
  비교: κ(Δp_t, Δh_t) vs κ(Δp_t, Δz_7) vs κ(Δp_t, Δz_t끝) vs κ(Δp_t, Δh_7) (목표 encoder 의 가린 t7) · 대조 κ(Δz_7, Δh_t) (encoder 가림 꼴 ↔ 보이는 미래)
  O 는 각각의 물체 자리 (Δh_t → 참 자리 t, Δz_7 · Δh_7 → t7 의 가려진 참 자리, Δz_t끝 → t끝 자리).
대상: late block 전부 (모양 · 색 = 정체, permanence = 존재), 이동 / 정지 따로. block bootstrap.
출력: results/q1q2/E2c_echo.json

  $P z_research/scripts/analysis/transfer_gap_echo.py --gpu 0
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_gap_common as C  # noqa: E402
import transfer_gap_pairdiff as PD  # noqa: E402

NT, NF = C.NT, C.NF


def main(gpu):
    import torch
    dev = torch.device(f"cuda:{gpu}")
    LN, shift_stats, region_shift = PD.torch_tools(dev)
    M = C.meta(); bmap, UV, HID, EXT = C.load_tracks()
    zm = json.load(open(C.TG / "z_late_meta.json")); zslot = {j: q for q, j in enumerate(zm["tp"])}
    H = np.load(C.TUBE / "h.npy", mmap_mode="r"); P = np.load(C.TUBE / "p.npy", mmap_mode="r")
    PF = np.load(C.TG / "p_relftg.npy", mmap_mode="r"); Z = np.load(C.TG / "z_late_ctx.npy", mmap_mode="r")
    blocks = defaultdict(dict)
    for j, m in enumerate(M):
        blocks[m["block_id"]][m["variant"]] = j
    load = lambda arr, j: torch.from_numpy(np.asarray(arr[j], np.float32)).to(dev)
    rows = []
    for b in sorted(blocks, key=int):
        ja, jb = blocks[b]["pos_a"], blocks[b]["pos_b"]; m = M[ja]
        if m["occlusion"] != "late":
            continue
        bi = bmap[b]; uv, ext = UV[bi, 0], EXT[bi, 0]
        hid = HID[bi, 0] | (HID[bi, 1] if m["violation"] != "permanence" else HID[bi, 0])
        tl = C.last_visible_ctx(hid)
        dH = load(H, ja) - load(H, jb); dZ = load(Z, zslot[ja]) - load(Z, zslot[jb])
        O7 = set(C.region(uv[7], ext[7])); Otl = set(C.region(uv[tl], ext[tl]))
        st = C.tub_state(m["samples"])
        for mdl, arr in (("rel", P), ("pft", PF)):
            dP = LN(load(arr, ja)) - LN(load(arr, jb))
            for k in range(NF):
                t = NF + k; X = dP[k]; Ot = set(C.region(uv[t], ext[t]))
                r = dict(block=b, objset=m["objset"], violation=m["violation"], moving=m["motion"] != "flat_static", k=m["k"], t=t, state=st[t], model=mdl, t_end=tl)
                for nm, Y, O in (("h_t", dH[t], Ot), ("z_7", dZ[7], O7), ("z_end", dZ[tl], Otl), ("h_7", dH[7], O7)):
                    r[f"k_{nm}"] = float(shift_stats(X[None], Y[None]).max())
                    r[f"kO_{nm}"] = float(region_shift(X, Y, O).max())
                rows.append(r)
        for k in range(NF):
            t = NF + k; Ot = set(C.region(uv[t], ext[t]))
            rows.append(dict(block=b, objset=m["objset"], violation=m["violation"], moving=m["motion"] != "flat_static", k=m["k"], t=t, state=st[t], model="enc",
                             k_z7_vs_ht=float(shift_stats(dZ[7][None], dH[t][None]).max()), kO_z7_vs_ht=float(region_shift(dZ[7], dH[t], Ot).max()),
                             k_zend_vs_ht=float(shift_stats(dZ[tl][None], dH[t][None]).max()), kO_zend_vs_ht=float(region_shift(dZ[tl], dH[t], Ot).max())))
    out = {}
    for viol in ("id", "perm"):
        for o in ("primitive", "realistic"):
            for mv in (True, False):
                for mdl in ("rel", "pft", "enc"):
                    for stt in ("vis", "occ", "all"):
                        sel = [r for r in rows if r["model"] == mdl and r["objset"] == o and ((r["violation"] != "permanence") == (viol == "id"))
                               and r["moving"] == mv and (stt == "all" or r["state"] == stt)]
                        if not sel:
                            continue
                        keys = [k for k in sel[0] if k.startswith(("k_", "kO_"))]
                        out[f"{viol}|{o}|{'move' if mv else 'static'}|{mdl}|{stt}"] = {
                            kk: C.bootstrap_mean([r[kk] for r in sel], [r["block"] for r in sel]) for kk in keys} | {"n": len(sel)}
    return rows, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0); a = ap.parse_args()
    rows, out = main(a.gpu)
    json.dump(dict(definition=__doc__, summary=out), open(C.RES / "E2c_echo.json", "w"), indent=1, ensure_ascii=False)
    print("→", C.RES / "E2c_echo.json", len(rows))
