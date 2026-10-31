#!/usr/bin/env python3
"""[X1] E1 관문 — 캐시 채점 재현 (G0) · A/B 궤적 같음 (G1) · PFT p 재현 (G3) (2026-10-30).

정의:
  s(i) = mean_{t∈8..15, c, d} |p_i − h_i| (표준 채점, 전체 토큰 L1, p 원값 · h = LN(target_encoder(32)) — bfloat16 캐시)
  가능 영상 i: p = tubeprobe p (릴리즈) 또는 p_relftg (PFT), h = tubeprobe h[t8..15]
  불가능 영상 q: p = 짝 가능 영상의 p (같은 문맥 → 비트 동일, CLAUDE.md §1-2), h = h_imp_fut[q]
  쌍 정답 = s(imp) > s(pos), matched (pos_a, imp_ab) · (pos_b, imp_ba).
  G0: 공식 per_video_surprise (float32 + autocast float16) 와 쌍 판정 일치 ≥ 97 %, 칸 (objset × 위반 × 가림) 정확도 차 ≤ 2 pt.
  G1: 모양 · 색 block 의 A · B 튜블릿 중심 차 |du| ≤ 8, |dv| ≤ 18 px@512 (t0..15 최대).
  G3: PFT p 로 split v2 test late 를 채점 → 공식 PFT per_video (split2 test) 와 일치 · late objset 별 88.0 / 91.7 ± 2.
출력: z_analysis/transfer_gap/results/q1q2/E1_gates.json · E1_scores.npz (영상별 s: release · PFT, pos · imp)

  $P z_research/scripts/analysis/transfer_gap_gates.py      # sbatch CPU (memmap 순차 읽기 ~30 GB)
"""
from __future__ import annotations
import csv, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
TG = Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap")
TUBE = Path("/local_datasets/world/world_analysis/cache/vobench_v1_vith_tubeprobe")
RES = REPO / "z_analysis/transfer_gap/results/q1q2"
OFF_REL = REPO / "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/per_block.json"
OFF_PFT = REPO / "z_training/runs/vobench_release_postft_g/eval/surprise_c16t32__vobench_v1_split2_test_latest/per_block.json"
SPLIT_TEST = REPO / "data_csv/vobench_v1_split_v2/index_test.csv"
NF = 8


def l1(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def main():
    RES.mkdir(parents=True, exist_ok=True)
    M = json.load(open(TUBE / "meta.json"))["meta"]
    IM = json.load(open(TG / "imp_meta.json"))["meta"]
    H = np.load(TUBE / "h.npy", mmap_mode="r"); P = np.load(TUBE / "p.npy", mmap_mode="r")
    PF = np.load(TG / "p_relftg.npy", mmap_mode="r"); HI = np.load(TG / "h_imp_fut.npy", mmap_mode="r")
    n, nI = len(M), len(IM)
    s_pos_rel = np.zeros(n); s_pos_pft = np.zeros(n); s_imp_rel = np.zeros(nI); s_imp_pft = np.zeros(nI)
    t0 = time.time()
    for i in range(n):
        hf = np.asarray(H[i, NF:]); s_pos_rel[i] = l1(P[i], hf); s_pos_pft[i] = l1(PF[i], hf)
        if i % 200 == 0:
            print(f"pos {i}/{n} {time.time() - t0:.0f}s", flush=True)
    for q, m in enumerate(IM):
        hi = np.asarray(HI[q]); j = m["pos_tp"]
        s_imp_rel[q] = l1(P[j], hi); s_imp_pft[q] = l1(PF[j], hi)
        if q % 200 == 0:
            print(f"imp {q}/{nI} {time.time() - t0:.0f}s", flush=True)
    np.savez(RES / "E1_scores.npz", s_pos_rel=s_pos_rel, s_pos_pft=s_pos_pft, s_imp_rel=s_imp_rel, s_imp_pft=s_imp_pft,
             pos_vid=np.array([m["video_id"] for m in M]), imp_vid=np.array([m["video_id"] for m in IM]),
             imp_pos_tp=np.array([m["pos_tp"] for m in IM]))

    def pairs_table(s_pos, s_imp, off, keep=None):
        rows = []
        for q, m in enumerate(IM):
            pm = M[m["pos_tp"]]
            if keep is not None and m["block_id"] not in keep:
                continue
            ours = s_imp[q] > s_pos[m["pos_tp"]]
            o = off[m["video_id"]] > off[pm["video_id"]] if off is not None else None
            rows.append(dict(cell=f'{m["objset"]}/{m["violation"]}/{m["occlusion"]}', dirn="A" if m["variant"] == "imp_ab" else "B",
                             moving=pm["motion"] != "flat_static", ours=bool(ours), off=None if o is None else bool(o),
                             d_ours=float(s_imp[q] - s_pos[m["pos_tp"]]),
                             d_off=None if off is None else float(off[m["video_id"]] - off[pm["video_id"]])))
        return rows

    def cells(rows, key="ours"):
        by = defaultdict(list)
        for r in rows:
            by[r["cell"]].append(r[key])
        return {c: 100 * float(np.mean(v)) for c, v in sorted(by.items())}

    out = {}
    # G0 — 릴리즈 캐시 채점 vs 공식
    off = json.load(open(OFF_REL))["per_video_surprise"]
    rows = pairs_table(s_pos_rel, s_imp_rel, off)
    agree = 100 * float(np.mean([r["ours"] == r["off"] for r in rows]))
    co, cf = cells(rows, "ours"), cells(rows, "off")
    dmax = max(abs(co[c] - cf[c]) for c in co)
    pv = np.array([s_pos_rel[j] for j in range(n)]); pvo = np.array([off[M[j]["video_id"]] for j in range(n)])
    out["G0"] = dict(n_pair=len(rows), pair_agree_pct=agree, cell_acc_maxdiff_pt=dmax, overall_ours=100 * float(np.mean([r["ours"] for r in rows])),
                     overall_off=100 * float(np.mean([r["off"] for r in rows])), cells_ours=co, cells_off=cf,
                     pos_surprise_corr=float(np.corrcoef(pv, pvo)[0, 1]), pos_surprise_rel_diff_median=float(np.median(np.abs(pv - pvo) / pvo)),
                     pass_=bool(agree >= 97 and dmax <= 2))
    # late 방향별 (릴리즈, 전 block)
    lat = defaultdict(list)
    for r in rows:
        if r["cell"].endswith("/late"):
            lat[(r["cell"].split("/")[0], r["cell"].split("/")[1], "move" if r["moving"] else "static", r["dirn"])].append((r["ours"], r["off"]))
    out["G0"]["late_dir"] = {"/".join(k): dict(n=len(v), ours=100 * float(np.mean([a for a, _ in v])), off=100 * float(np.mean([b for _, b in v])))
                             for k, v in sorted(lat.items())}
    print("G0", {k: v for k, v in out["G0"].items() if not isinstance(v, dict)}, flush=True)

    # G3 — PFT
    test_blocks = {r["block_id"] for r in csv.DictReader(SPLIT_TEST.open())}
    offp = json.load(open(OFF_PFT))["per_video_surprise"]
    rows3 = pairs_table(s_pos_pft, s_imp_pft, offp, keep=test_blocks)
    agree3 = 100 * float(np.mean([r["ours"] == r["off"] for r in rows3]))
    late = lambda rs, o, k: 100 * float(np.mean([r[k] for r in rs if r["cell"].startswith(o) and r["cell"].endswith("late")]))
    out["G3"] = dict(n_pair=len(rows3), pair_agree_pct=agree3,
                     late_ours={o: late(rows3, o, "ours") for o in ("primitive", "realistic")},
                     late_off={o: late(rows3, o, "off") for o in ("primitive", "realistic")},
                     overall_ours=100 * float(np.mean([r["ours"] for r in rows3])), overall_off=100 * float(np.mean([r["off"] for r in rows3])),
                     cells_ours=cells(rows3, "ours"), cells_off=cells(rows3, "off"))
    out["G3"]["pass_"] = bool(agree3 >= 97 and all(abs(out["G3"]["late_ours"][o] - out["G3"]["late_off"][o]) <= 2 for o in ("primitive", "realistic")))
    rows3r = pairs_table(s_pos_rel, s_imp_rel, off, keep=test_blocks)
    out["G3"]["release_same_test_late"] = {o: late(rows3r, o, "ours") for o in ("primitive", "realistic")}
    print("G3", {k: v for k, v in out["G3"].items() if not k.startswith("cells")}, flush=True)

    # G1 — 궤적
    tr = np.load(TG / "tracks.npz")
    bid = list(tr["block_id"]); uv = tr["uv"]
    viol = {m["block_id"]: m["violation"] for m in M}
    bad, mx = [], []
    for bi, b in enumerate(bid):
        if viol.get(b) not in ("shape", "color"):
            continue
        d = np.abs(uv[bi, 0] - uv[bi, 1])
        mx.append(d.max(0))
        if d[:, 0].max() > 8 or d[:, 1].max() > 18:
            bad.append(dict(block_id=b, du=float(d[:, 0].max()), dv=float(d[:, 1].max())))
    mx = np.array(mx)
    out["G1"] = dict(n_block=len(mx), du_max=float(mx[:, 0].max()), dv_max=float(mx[:, 1].max()),
                     du_p95=float(np.percentile(mx[:, 0], 95)), dv_p95=float(np.percentile(mx[:, 1], 95)), excluded=bad, pass_=len(bad) == 0)
    print("G1", {k: v for k, v in out["G1"].items() if k != "excluded"}, len(bad), flush=True)
    json.dump(out, open(RES / "E1_gates.json", "w"), indent=1, ensure_ascii=False)
    print("→", RES / "E1_gates.json")


if __name__ == "__main__":
    main()
