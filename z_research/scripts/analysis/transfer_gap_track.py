#!/usr/bin/env python3
"""transfer_gap Q3 · E32 — 진행 거리 · 방향 끌림: PFT 는 문맥 끝에 가려진 물체를 문맥의 운동대로 옮기나, VOBench 전형 거리로 옮기나. 2026-10-30, [X2] R2.

왜: 사용자 Q3 의 "진행 방향을 외웠나" 를 직접 잰다. E12 의 역재생 · 보폭은 점수만 봤다. 학습 분포 밖 속력 (v11 · 보폭 2) 에서도
    PFT 가 예측 물체를 VOBench 전형 진행 거리 (t7 → t15 중앙 5.7 칸) 쪽으로 옮기면 H3-V (진행 거리 외우기).
정의 (파라미터 없음, 자 없음)
  존재 대비 지도: 같은 block 의 두 가능 영상 (물체 문맥 pos_a, 빈 문맥 pos_b; 같은 장면 · 같은 가림막).
    E^h_{t,c} = ‖h_a − h_b‖₂ (t = 0..15, 칸 c = 16×16) · E^p_{t,c} = ‖p_a − p_b‖₂ (미래 t = 8..15, 릴리즈 · PFT).
  자리 loc(E_t) = 3×3 상자 합이 최대인 상자의 중심 칸 (행, 열). 몰림 = 그 상자 합 ÷ 전체 합 (균등이면 9/256 = 0.035).
  마지막 본 자리 L = loc(E^h_{t_last}), t_last = 문맥 튜블릿 중 두 장이 다 보이는 마지막 튜블릿 (VOBench samples 문자열 · v11 n_context_hidden).
  참 진행 d = ‖loc(E^h_15) − L‖ · 예측 진행 d̂ = ‖loc(E^p_15) − L‖ (칸).  기울기 b: d̂ = a + b·d (최소제곱).
  방향 일치 = (예측 변위 · 참 변위) > 0 인 비율 (참 변위 ≥ 1 칸만).  끌림 = |d̂ − 5.7| < |d̂ − d| 인 비율 (|d − 5.7| ≥ 1.5 인 block 만).
  관문: VOBench 보이는 튜블릿에서 loc(E^h_t) 가 label 중심과 1 칸 (체비셰프) 안 ≥ 90 %.
  ⚠️ 정정 (2026-10-30, 첫 실행 뒤) — 처음엔 "예측 몰림 ≥ 균등의 3 배 (0.105)" 인 block 만 썼는데, 참 h 의 존재 대비도 몰림이 0.047 뿐이라
     (에너지가 화면 전체에 퍼짐, R2 계획 §해석) 예측 block 의 0–10 % 만 남았다. 대신 정점의 두드러짐 z = (최대 상자 − 상자 평균) / 상자 표준편차
     를 내고 (h 2.3–2.9, 예측 2.9–4.0), 모든 block 으로 잰다. 예측 자리 읽기의 검증 = 보임 조건에서 loc(E^p_t) 가 loc(E^h_t) 와 1 칸 안인 비율.
세트: VOBench split v2 test (원본) · VOBench test 보폭 2 (원프레임 15–77) · v11 부분 (문맥 끝 가림 이동 vanish + 보임 이동) · v11 실물 메시.
출력: 배열 /data2/local_datasets/world/world_analysis/cache/transfer_gap/E32/<set>.npz · 결과 z_analysis/transfer_gap/results/q3_r2/E32_track.json

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/transfer_gap_track.py extract --sets vob,vob_s2,v11sub,v11real     (GPU 1, 약 10 분)
  $P z_research/scripts/analysis/transfer_gap_track.py report
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

ARR = Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap/E32")
OUT = REPO / "z_analysis/transfer_gap/results/q3_r2/E32_track.json"
REL = str(REPO / "z_training/runs/release_vith/latest.pt")
PFT = str(REPO / "z_training/runs/vobench_release_postft_g/latest.pt")
TYP = 5.7
SETS = {
    "vob": ("vobench_v1_split2_test", []),
    "vob_s2": ("vobench_v1_split2_test", ["data.frames_start=15", "data.frames_stride=2"]),
    "v11sub": ("v11_split_test", ["data.root=" + str(REPO / "data_csv/transfer_gap_v11sub"), "data.index_csv=index.csv"]),
    "v11real": ("v11_realistic", []),
}


def pick_blocks(W, name):
    by = defaultdict(dict)
    for i, r in enumerate(W.records):
        by[r.block_id][r.raw.get("variant")] = i
    out = []
    for b, v in by.items():
        if "pos_a" not in v or "pos_b" not in v:
            continue
        raw = W.records[v["pos_a"]].raw
        if name.startswith("vob"):
            if raw.get("violation") != "permanence" or raw.get("motion") == "flat_static":
                continue
            samp = raw.get("samples", "").replace("|", "")
            hid = [samp[2 * t] == "H" or samp[2 * t + 1] == "H" for t in range(16)] if len(samp) == 32 else [False] * 16
            vis_ctx = [t for t in range(8) if not hid[t]]
            out.append({"block": b, "a": v["pos_a"], "b": v["pos_b"], "occ": raw.get("occlusion"), "k": raw.get("k"),
                        "motion": raw.get("motion"), "objset": raw.get("objset"), "bg": raw.get("background"),
                        "t_last": max(vis_ctx) if vis_ctx else -1, "hidden": hid})
        else:
            if raw.get("violation_type") != "vanish" or raw.get("condition", "").startswith("static"):
                continue
            k = int(raw.get("n_context_hidden") or raw.get("sym_k") or 0)
            if not raw.get("condition", "").startswith("moving_occlusion"):
                k = 0
            out.append({"block": b, "a": v["pos_a"], "b": v["pos_b"], "occ": "late" if k > 0 else "visible", "k": str(k),
                        "motion": raw.get("condition"), "objset": "v11", "bg": "v11", "t_last": 7 - math.ceil(k / 2),
                        "hidden": [False] * (8 - math.ceil(k / 2)) + [True] * math.ceil(k / 2) + [None] * 8})
    return out


def extract(sets):
    import torch
    import te_v11_score as te
    from transfer_gap_te import resolve
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    sc = None
    ARR.mkdir(parents=True, exist_ok=True)
    for name in sets:
        ds, st = SETS[name]
        cfg, _ = resolve(ds, "vith", st, None, 0)
        W = te.WMADataset(cfg)
        blocks = pick_blocks(W, name)
        if sc is None:
            sc = te.Scorer(cfg, dev, [("release", REL), ("pft", PFT)])
        D, S, nc, nf = sc.D, sc.S, sc.nc, sc.nf
        n = len(blocks)
        Eh = np.zeros((n, nc + nf, S), np.float32); Er = np.zeros((n, nf, S), np.float32); Ef = np.zeros((n, nf, S), np.float32)
        t0 = time.time()
        bs = 4
        for b0 in range(0, n, bs):
            bl = blocks[b0:b0 + bs]
            clips = torch.stack([W.clip(x[y]) for x in bl for y in ("a", "b")])
            with torch.inference_mode():
                z, h, ci, ti, _ = sc.encode(clips)
                B = z.size(0)
                hh = h.float().reshape(B, nc + nf, S, D)
                pr = sc.predict(sc.mods[0], z, ci, ti).float().reshape(B, nf, S, D)
                pf = sc.predict(sc.mods[1], z, ci, ti).float().reshape(B, nf, S, D)
                dn = lambda x: (x[0::2] - x[1::2]).norm(dim=-1).cpu().numpy()  # noqa: E731
                Eh[b0:b0 + len(bl)] = dn(hh); Er[b0:b0 + len(bl)] = dn(pr); Ef[b0:b0 + len(bl)] = dn(pf)
            if (b0 // bs) % 20 == 0:
                print(f"[{name}] {b0 + len(bl)}/{n} {time.time() - t0:.0f}s", flush=True)
        meta = [{k: v for k, v in x.items() if k not in ("a", "b")} | {"vid_a": W.records[x["a"]].video_id, "vid_b": W.records[x["b"]].video_id}
                for x in blocks]
        np.savez(ARR / f"{name}.npz", Eh=Eh, Er=Er, Ef=Ef)
        json.dump(meta, open(ARR / f"{name}_meta.json", "w"), ensure_ascii=False)
        print(f"[{name}] done {n} blocks {time.time() - t0:.0f}s", flush=True)


def loc(E):
    """E (..., 256) → (행, 열, 몰림). 3×3 상자 합 최대 (14×14 중심)."""
    G = E.reshape(*E.shape[:-1], 16, 16)
    box = sum(G[..., i:i + 14, j:j + 14] for i in range(3) for j in range(3))
    flat = box.reshape(*box.shape[:-2], -1)
    k = flat.argmax(-1)
    r, c = k // 14 + 1, k % 14 + 1
    share = flat.max(-1) / np.maximum(E.sum(-1), 1e-9)
    return r, c, share


def report():
    res = {"definition": __doc__.split("정의 (파라미터 없음, 자 없음)")[1].split("세트:")[0].strip(), "typical_progress_cells": TYP, "sets": {}}
    # 관문: VOBench label
    tr = np.load(Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap/tracks.npz"))
    uvmax = float(np.nanmax(tr["uv"]))
    scale = 512 / 16 if uvmax > 260 else 256 / 16
    lab = {str(b): i for i, b in enumerate(tr["block_id"])}
    for name in SETS:
        f = ARR / f"{name}.npz"
        if not f.exists():
            continue
        Z = np.load(f); meta = json.load(open(ARR / f"{name}_meta.json"))
        Eh, Er, Ef = Z["Eh"], Z["Er"], Z["Ef"]
        rh, ch, sh = loc(Eh); rr, cr, sr = loc(Er); rf, cf, sf = loc(Ef)
        R = {"n_blocks": len(meta)}
        if name == "vob":   # 관문 (보이는 튜블릿, A = 물체 영상)
            hit, tot = 0, 0
            for i, m in enumerate(meta):
                j = lab.get(str(m["block"]))
                if j is None or not tr["present"][j, 0]:
                    continue
                for t in range(16):
                    if tr["hidden"][j, 0, t] or (m["hidden"][t] if t < len(m["hidden"]) and m["hidden"][t] is not None else False):
                        continue
                    u, v = tr["uv"][j, 0, t] / scale          # (x, y) 칸
                    tot += 1
                    hit += max(abs(ch[i, t] + 0.5 - u), abs(rh[i, t] + 0.5 - v)) <= 1.5   # 칸 중심 기준 1 칸 안 (체비셰프)
            R["gate_label_within_1cell"] = round(hit / max(tot, 1), 4); R["gate_n"] = tot; R["uv_scale_px_per_cell"] = scale
        # 데이터로 정한 t_last: 문맥 튜블릿 중 존재 대비 총량이 앞 4 튜블릿 중앙의 절반 이상인 마지막 튜블릿 (가린 튜블릿은 두 영상이 같아 Δh ≈ 0)
        Et = Eh.sum(-1)
        ref = np.median(Et[:, :4], 1)
        tl_data = np.array([max([t for t in range(8) if Et[i, t] >= 0.5 * ref[i]] or [-1]) for i in range(len(meta))])
        tl_meta = np.array([m["t_last"] for m in meta])
        R["t_last_meta_vs_data_agree"] = round(float((tl_data == tl_meta).mean()), 3)
        R["t_last_source"] = "data" if name == "vob_s2" else "meta"
        if name == "vob_s2":       # samples 문자열은 보폭 3 창 기준이라 보폭 2 판에는 데이터 규칙을 쓴다
            for i, m in enumerate(meta):
                m["t_last"] = int(tl_data[i])
        groups = defaultdict(list)
        for i, m in enumerate(meta):
            groups[f"occ={m['occ']}"].append(i)
            if m["occ"] == "late":
                groups[f"late|motion={m['motion']}"].append(i)
                groups[f"late|k={m['k']}"].append(i)
        for g, ix in sorted(groups.items()):
            ix = np.array(ix)
            tl = np.array([meta[i]["t_last"] for i in ix])
            ok = tl >= 0
            ix, tl = ix[ok], tl[ok]
            if len(ix) == 0:
                continue
            Lr, Lc = rh[ix, tl], ch[ix, tl]
            T = (rh[ix, 15] - Lr, ch[ix, 15] - Lc)
            d = np.hypot(*T)
            out = {"n": int(len(ix)), "true_d_median": round(float(np.median(d)), 2), "true_d_q10_q90": [round(float(np.percentile(d, q)), 2) for q in (10, 90)],
                   "h_share_t15_median": round(float(np.median(sh[ix, 15])), 3)}
            for mn, (r_, c_, s_) in (("release", (rr, cr, sr)), ("pft", (rf, cf, sf))):
                Pv = (r_[ix, 7] - Lr, c_[ix, 7] - Lc)
                dh = np.hypot(*Pv)
                conc = s_[ix, 7] >= 3 * 9 / 256
                dot = Pv[0] * T[0] + Pv[1] * T[1]
                mv = d >= 1
                mm = np.ones(len(ix), bool)                 # 정정: 몰림 문턱 없이 모든 block
                Pm = {f"t{8 + q}": round(float((np.maximum(np.abs(r_[ix, q] - rh[ix, 8 + q]), np.abs(c_[ix, q] - ch[ix, 8 + q])) <= 1).mean()), 3) for q in (0, 3, 7)}
                Lm = round(float((np.maximum(np.abs(r_[ix, 7] - Lr), np.abs(c_[ix, 7] - Lc)) <= 1).mean()), 3)
                o = {"pred_share_median": round(float(np.median(s_[ix, 7])), 3), "frac_share_ge_3x_uniform": round(float(conc.mean()), 3),
                     "pred_loc_within1_of_true": Pm, "pred_t15_within1_of_last_seen": Lm,
                     "pred_d_median_all": round(float(np.median(dh)), 2)}
                if mm.sum() >= 3:
                    dd, dhh = d[mm], dh[mm]
                    b, a = np.polyfit(dd, dhh, 1) if np.ptp(dd) > 0 else (float("nan"), float("nan"))
                    far = np.abs(dd - TYP) >= 1.5
                    o.update({"n_conc": int(mm.sum()), "pred_d_median": round(float(np.median(dhh)), 2), "slope_b": round(float(b), 3),
                              "intercept_a": round(float(a), 3), "corr": round(float(np.corrcoef(dd, dhh)[0, 1]), 3) if np.ptp(dd) > 0 and np.ptp(dhh) > 0 else None,
                              "ratio_pred_over_true_median": round(float(np.median(dhh / np.maximum(dd, 0.5))), 3),
                              "dir_agree": round(float((dot[mm & mv] > 0).mean()), 3) if (mm & mv).sum() else None, "n_dir": int((mm & mv).sum()),
                              "pull_to_typ": round(float((np.abs(dhh[far] - TYP) < np.abs(dhh[far] - dd[far])).mean()), 3) if far.sum() else None,
                              "n_pull": int(far.sum()),
                              "err_to_true_median": round(float(np.median(np.hypot(r_[ix, 7][mm] - rh[ix, 15][mm], c_[ix, 7][mm] - ch[ix, 15][mm]))), 2)})
                out[mn] = o
            R[g] = out
        res["sets"][name] = R
        print(name, json.dumps({k: v for k, v in R.items() if k.startswith(("gate", "occ=late", "occ=visible"))}, ensure_ascii=False), flush=True)
    v = {}
    for name in ("v11sub", "vob_s2", "v11real"):
        x = res["sets"].get(name, {}).get("occ=late", {}).get("pft", {})
        if x.get("slope_b") is not None:
            v[name] = {"b": x["slope_b"], "pull": x.get("pull_to_typ"), "dir": x.get("dir_agree"),
                       "H3-V": bool(x["slope_b"] <= 0.5 and (x.get("pull_to_typ") or 0) >= 0.7),
                       "follows_context": bool(x["slope_b"] >= 0.8 and (x.get("dir_agree") or 0) >= 0.9)}
    res["verdict_inputs"] = v
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(v, ensure_ascii=False)); print("→", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["extract", "report"])
    ap.add_argument("--sets", default="vob,vob_s2,v11sub,v11real")
    ap.add_argument("--pft", default=None, help="PFT 자리에 다른 체크포인트 (예: v11 PFT). 배열 · 결과 이름에 --tag 를 붙인다")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    global PFT, ARR, OUT
    if a.pft:
        PFT = a.pft
    if a.tag:
        ARR = ARR / a.tag
        OUT = OUT.with_name(f"E32_track_{a.tag}.json")
    extract(a.sets.split(",")) if a.part == "extract" else report()


if __name__ == "__main__":
    main()
