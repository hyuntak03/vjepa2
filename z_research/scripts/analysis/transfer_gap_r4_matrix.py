#!/usr/bin/env python3
"""transfer_gap R4 E39 — 지금까지 학습한 PFT 들의 late 영속 전이 행렬 (2026-10-30, 실험 A [X1]).

물음: 어떤 학습 데이터가 학습 생성기 밖의 late 영속 이득을 냈나?
경로: 열마다 **한 번의 encoder 통과 (te 경로, transfer_gap_te.py)** 로 모든 predictor + 복사 기준선을 같은 화소 · 같은 수치로 채점했다
      (같은 열 안의 행끼리는 같은 경로라 비교된다). 기존 하네스 산출물 (per_block.json) 과의 쌍 판정 일치율을 관문으로 같이 낸다.
정의:
  쌍 = 같은 장면 (block) · 같은 pair_id 의 (가능, 불가능). 판정 = s(불가능) > s(가능) (같으면 0.5). s = 미래 8 튜블릿 평균 L1 (표준 surprise).
  A = 사라짐 방향 (pair A: 물체가 있던 문맥 → 빈 미래가 불가능) · B = 나타남 방향 (빈 문맥 → 물체 미래).
  열: vob = VOBench split v2 test late 이동 permanence (30 block, 256² 사본) · v11 = v11 split test 가림 이동 (평지 + 경사) 사라짐 (224 block, 원본 화소 사본)
      · v11st = 같은 판의 정지 가림 (112 block, 참고) · v11r = v11_realistic 가림 이동 사라짐 · ip1 = IntPhys 1 late uni 이동 · 가림 (60 쌍; A/B 없음 → 쌍 정확도)
      · ip2 = IntPhys 2 late uni permanence (120 쌍; 쌍 정확도. 가능 · 불가능 문맥이 화소로 같지 않다).
  G = acc(행) − acc(릴리즈) (같은 쌍), Δcopy = acc(행) − acc(복사). CI = 장면 bootstrap 2,000 (2.5–97.5).
  판정 (사전 등록, BOARD "[X1] R4 — 사전 등록" E39): 자기 생성기 밖 칸에서 G ≥ +10 이고 CI 하한 > 0 이고 B 하락 ≤ 10 → 영속 이득,
      + Δcopy CI 하한 > 0 → 복사를 넘는 이득. B 가 10 넘게 떨어지면 편향 이동. (ip1 · ip2 는 B 를 따로 못 내므로 쌍 정확도로만.)
입력: z_analysis/transfer_gap/results/r4_x1/te/<열>/{meta.json, l1.npy, copy.npy, done.npy}
재현: $P z_research/scripts/analysis/transfer_gap_r4_matrix.py   (CPU, 이 셸) → results/r4_x1/E39_matrix.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TE = REPO / "z_analysis/transfer_gap/results/r4_x1/te"
OUT = REPO / "z_analysis/transfer_gap/results/r4_x1/E39_matrix.json"
NB = 2000
# 행 → 자기 생성기 (열 이름들). '반쯤' 은 따로 표시
OWN = {"release": set(), "vobftg": {"vob"}, "vobft": {"vob"}, "voball": {"vob"}, "vobobjho": {"vob"},
       "v11pft": {"v11", "v11st"}, "ip1pft": {"ip1"}, "ip1late_f0": {"ip1"}, "ip2pair": {"ip2"}, "ip2e80": {"ip2"}}
HALF = {"vobftg": {"v11", "v11st", "v11r"}, "vobft": {"v11", "v11st", "v11r"}, "voball": {"v11", "v11st", "v11r"}, "vobobjho": {"v11", "v11st", "v11r"},
        "v11pft": {"v11r"}}
INSAMPLE = {("voball", "vob"): "가능 영상 전부로 학습 (test 포함)", ("ip1pft", "ip1"): "IntPhys 1 dev 로 학습 (같은 영상의 다른 창)",
            ("ip1late_f0", "ip1"): "fold 목록이 없어 held-out 을 못 고름", ("ip2pair", "ip2"): "Main 짝 split 학습 (일부 영상 겹침)",
            ("ip2e80", "ip2"): "IntPhys 2 로 학습"}
GATES = {  # 열 → (하네스 산출물, 행) 쌍 판정 일치 관문
    "v11": [("z_research/previous/IntPhysGenV11/exp_results/surprise_c16t32__v11_vith/per_block.json", "release"),
            ("z_training/runs/v11_postft/eval/surprise_c16t32__v11_split_test_e10/per_block.json", "v11pft")],
    "vob": [("z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/per_block.json", "release"),
            ("z_training/runs/vobench_release_postft_g/eval/surprise_c16t32__vobench_v1_split2_test_latest/per_block.json", "vobftg")],
    "ip1": [("cvpr/results/eval/intphys1_late/intphys1_late_uni_vith/per_block.json", "release"),
            ("cvpr/results/eval/intphys1_late/intphys1_late_uni_vith_release_vobftg/per_block.json", "vobftg")],
    "ip2": [("cvpr/results/eval/intphys2_late/intphys2_late_uni_vith/per_block.json", "release"),
            ("cvpr/results/eval/intphys2_late/intphys2_late_uni_vith_release_vobftg/per_block.json", "vobftg")],
    "v11r": [("z_research/previous/v11_realistic/exp_results/surprise_c16t32__v11_realistic_vith/per_block.json", "release")],
}


VAST = {  # 열 → (vast AR per_block, vast AR + PFT per_block) — 하네스 산출물, AR 공간 (블록별 LN(target)) 이라 둘끼리만 비교
    "vob": ("z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith_vast_ar_state_ep70/per_block.json",
            "z_training/runs/vobench_vast_ar_state_postft_g/eval/surprise_c16t32__vobench_v1_split2_test_latest/per_block.json"),
    "v11": ("z_analysis/transfer_gap/results/q3/runs/E11_v11sub_vith_vast_ar_state_ep70/per_block.json",
            "z_analysis/transfer_gap/results/q3/runs/E11_v11sub_vith_vast_ar_state_ep70_vobftg/per_block.json"),
    "v11st": ("z_analysis/transfer_gap/results/q3/runs/E11_v11sub_vith_vast_ar_state_ep70/per_block.json",
              "z_analysis/transfer_gap/results/q3/runs/E11_v11sub_vith_vast_ar_state_ep70_vobftg/per_block.json"),
    "v11r": ("z_analysis/transfer_gap/results/q3/runs/E11_v11real_vith_vast_ar_state_ep70/per_block.json",
             "z_analysis/transfer_gap/results/q3/runs/E11_v11real_vith_vast_ar_state_ep70_vobftg/per_block.json"),
    "ip1": ("cvpr/results/eval/intphys1_late/intphys1_late_uni_vith_vast_ar_state_ep70/per_block.json",
            "cvpr/results/eval/intphys1_late/intphys1_late_uni_vith_vast_ar_state_ep70_vobftg/per_block.json"),
    "ip2": ("cvpr/results/eval/intphys2_late/intphys2_late_uni_vith_vast_ar_state_ep70/per_block.json",
            "cvpr/results/eval/intphys2_late/intphys2_late_uni_vith_vast_ar_state_ep70_vobftg/per_block.json"),
}


def vast_rows(col, m, P, sc, dd):
    fa, fb = (REPO / x for x in VAST.get(col, ("", "")))
    if not fa.is_file() or not fb.is_file(): return None
    va, vb = json.load(open(fa))["per_video_surprise"], json.load(open(fb))["per_video_surprise"]
    vid = m["video_id"]
    ok = np.array([vid[p["pos"]] in va and vid[p["imp"]] in va and vid[p["pos"]] in vb and vid[p["imp"]] in vb for p in P])
    if ok.sum() == 0: return None
    ja = np.array([1.0 if va[vid[p["imp"]]] > va[vid[p["pos"]]] else 0.0 for p, o in zip(P, ok) if o])
    jb = np.array([1.0 if vb[vid[p["imp"]]] > vb[vid[p["pos"]]] else 0.0 for p, o in zip(P, ok) if o])
    s2, d2 = sc[ok], dd[ok]
    out = {"n_pair": int(ok.sum()), "src": [str(fa.relative_to(REPO)), str(fb.relative_to(REPO))]}
    for di in sorted(set(d2)):
        k = d2 == di
        out[f"vastAR_{di}"] = boot(ja[k], s2[k]); out[f"vastAR_pft_{di}"] = boot(jb[k], s2[k]); out[f"G_{di}"] = boot(jb[k], s2[k], ja[k])
    return out


def load(col_dir):
    m = json.load(open(col_dir / "meta.json"))
    l1 = np.load(col_dir / "l1.npy").mean(-1)                        # (n, K)
    cp = np.load(col_dir / "copy.npy").mean(-1)                       # (n,)
    assert np.load(col_dir / "done.npy").all(), col_dir
    return m, l1, cp


def pairs(m, col):
    import csv
    idx = {r["video_id"]: r for r in csv.DictReader(open(m["index_csv"]))}
    by = defaultdict(dict)
    n = len(m["video_id"])
    for i in range(n):
        blk = (m.get("objset") or [None] * n)[i] or ""
        key = (blk, m["block_id"][i], m["pair_id"][i])
        by[key]["pos" if str(m["plausible"][i]) == "1" else "imp"] = i
    out = []
    for (o, b, pid), v in by.items():
        if "pos" not in v or "imp" not in v: continue
        i = v["pos"]
        if col in ("v11", "v11st", "v11r"):
            cond = m["condition"][i]
            if col == "v11st" and cond != "static_occlusion": continue
            if col in ("v11", "v11r") and cond not in ("moving_occlusion", "moving_occlusion_flat"): continue
            d = "A" if m["variant"][i] == "pos_a" else "B"
        elif col == "vob":
            d = "A" if m["variant"][i] == "pos_a" else "B"
        elif col == "ip1":
            if idx[m["video_id"][i]]["group"] != "moving/occluded": continue
            d = "P"
        elif col == "ip2":
            if m["condition"][i] != "permanence": continue
            d = "P"
        out.append(dict(pos=v["pos"], imp=v["imp"], scene=f"{o}|{b}", d=d))
    return out


def judge(s, P):
    return np.array([1.0 if s[p["imp"]] > s[p["pos"]] else 0.5 if s[p["imp"]] == s[p["pos"]] else 0.0 for p in P])


def boot(x, sc, ref=None, seed=0):
    xs = x if ref is None else x - ref
    u, inv = np.unique(sc, return_inverse=True)
    s = np.bincount(inv, xs, len(u)); n = np.bincount(inv, None, len(u))
    rng = np.random.default_rng(seed); b = []
    for _ in range(NB):
        ii = rng.integers(0, len(u), len(u)); b.append(s[ii].sum() / n[ii].sum())
    return [round(100 * float(xs.mean()), 2), round(100 * float(np.percentile(b, 2.5)), 2), round(100 * float(np.percentile(b, 97.5)), 2)]


def gate(col, m, l1, tags, P):
    out = []
    for path, row in GATES.get(col, []):
        f = REPO / path
        if not f.exists() or row not in tags: continue
        pv = json.load(open(f))["per_video_surprise"]
        vid = m["video_id"]; k = tags.index(row)
        ok = [p for p in P if vid[p["pos"]] in pv and vid[p["imp"]] in pv]
        if not ok: continue
        a = judge(l1[:, k], ok); h = np.array([1.0 if pv[vid[p["imp"]]] > pv[vid[p["pos"]]] else 0.0 for p in ok])
        out.append(dict(src=path, row=row, n=len(ok), agree=round(float((a == h).mean()), 4), te_acc=round(100 * float(a.mean()), 2), harness_acc=round(100 * float(h.mean()), 2)))
    return out


def main():
    res = {"definition": __doc__.split("정의:")[1].split("입력:")[0].strip(), "cols": {}, "verdicts": {}}
    cols = [c for c in ("vob", "v11", "v11st", "v11r", "ip1", "ip2") if (TE / ({"v11st": "v11sub3", "v11": "v11sub3"}.get(c, c))).exists()]
    for col in cols:
        d = TE / {"v11st": "v11sub3", "v11": "v11sub3"}.get(col, col)
        if not (d / "done.npy").exists() or not np.load(d / "done.npy").all():
            print("미완", col); continue
        m, l1, cp = load(d)
        tags = m["pred_tags"]
        P = pairs(m, col)
        sc = np.array([p["scene"] for p in P]); dd = np.array([p["d"] for p in P])
        J = {t: judge(l1[:, k], P) for k, t in enumerate(tags)}; J["copy"] = judge(cp, P)
        cr = {"n_pair": len(P), "gates": gate(col, m, l1, tags, P), "rows": {}}
        dirs = ["A", "B"] if "A" in dd else ["P"]
        for t in list(tags) + ["copy"]:
            r = {}
            for di in dirs:
                s = dd == di
                r[di] = boot(J[t][s], sc[s])
                if t not in ("release", "copy"):
                    r[f"G_{di}"] = boot(J[t][s], sc[s], J["release"][s])
                if t != "copy":
                    r[f"dcopy_{di}"] = boot(J[t][s], sc[s], J["copy"][s])
            r["own"] = "자기" if col in OWN.get(t, set()) else ("반쯤" if col in HALF.get(t, set()) else "밖")
            if (t, col) in INSAMPLE: r["note"] = INSAMPLE[(t, col)]
            cr["rows"][t] = r
        cr["vast"] = vast_rows(col, m, P, sc, dd)
        res["cols"][col] = cr
    # 판정
    for t in OWN:
        if t == "release": continue
        v = {}
        for col, cr in res["cols"].items():
            r = cr["rows"].get(t)
            if r is None or r["own"] == "자기": continue
            main_d = "A" if "A" in r else "P"
            G = r[f"G_{main_d}"]; dc = r[f"dcopy_{main_d}"]
            bdrop = (r["B"][0] - cr["rows"]["release"]["B"][0]) if "B" in r else 0.0
            if G[0] >= 10 and G[1] > 0 and bdrop >= -10:
                vv = "복사를 넘는 영속 이득" if dc[1] > 0 else "영속 이득 (복사 못 넘음)"
            elif G[0] >= 10 and G[1] > 0:
                vv = "편향 이동 (B 하락)"
            else:
                vv = "이득 없음"
            v[col] = dict(verdict=vv, own=r["own"], G=G, dcopy=dc, B_drop=round(bdrop, 2))
        res["verdicts"][t] = v
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    for col, cr in res["cols"].items():
        print(f"\n== {col} (쌍 {cr['n_pair']}) 관문 {[(g['row'], g['agree'], g['te_acc'], g['harness_acc']) for g in cr['gates']]}")
        for t, r in cr["rows"].items():
            ks = [k for k in r if k in ("A", "B", "P")]
            print(f"  {t:10s} " + " · ".join(f"{k} {r[k][0]:5.1f} [{r[k][1]:.0f},{r[k][2]:.0f}]" for k in ks)
                  + "".join(f" · G_{k[2:]} {r[k][0]:+.1f} [{r[k][1]:+.0f},{r[k][2]:+.0f}]" for k in r if k.startswith("G_"))
                  + "".join(f" · Δcopy_{k[6:]} {r[k][0]:+.1f} [{r[k][1]:+.0f},{r[k][2]:+.0f}]" for k in r if k.startswith("dcopy_") and k[6:] in ("A", "P"))
                  + f"  ({r['own']}{' · ' + r['note'] if 'note' in r else ''})")
    print("\n판정:")
    for t, v in res["verdicts"].items():
        print(f"  {t:10s} " + " | ".join(f"{c}: {d['verdict']} ({d['own']}, G {d['G'][0]:+.1f} [{d['G'][1]:+.0f},{d['G'][2]:+.0f}])" for c, d in v.items()))


if __name__ == "__main__":
    main()
