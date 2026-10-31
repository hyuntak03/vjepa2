#!/usr/bin/env python3
"""transfer_gap R4 [X2] — 진단 PFT 의 끝 자리 지표 (E35 자 그대로) + 전이 판의 존재 대비 지도. 2026-10-30.

왜: R3 E35 는 VOBench PFT 의 뒤쪽 미래 가로 자리가 학습셋 시간표를 따른다는 것을 보였다 (보폭 2 끌림 비 λ 1.0 · 가로 따라감 비 β 0.12).
    R4 진단 PFT (T0 s0 · s1 · T1 끝 자리 흩뜨리기 · T2 생성기 섞기) 에서 이 두 비가 어떻게 바뀌는지, 그리고 전이 판 (v11 · v11_realistic)
    에서 사라짐 정답과 같이 낼 존재 대비 지도의 참 자리 1 칸 안 비율 (R2 규칙 1) 을 잰다.
정의: transfer_gap_endprior.py (E35) 머리말과 같다 — 존재 대비 지도 E = ‖x_물체팔 − x_빈장면팔‖₂ (칸별), 자리 = 3×3 상자 합 최대,
    참 자리 = 같은 튜블릿 h 의 자리, λ = (예측 열 − 보폭 2 참 열) / (보폭 3 참 열 − 보폭 2 참 열) (t15, 두 참 열 차 ≥ 1.5 칸 block 중앙값),
    β = Σ(예측 열 차) / Σ(참 열 차) (잘라 옮기기 창 x0 · x64, t15). 전형 끝 자리 = R3 E35 의 VOBench train h 배열 (vobtrain, 원본 프레임).
E35 코드를 import 해서 경로 · predictor 목록 · 세트만 바꾼다 (E35 파일 수정 없음).
  · 프레임: VOBench test late = 256² NFS 사본 (원판 · 보폭 2), 잘라 옮기기 = R3 창 프레임의 NFS 바이트 사본 (late permanence 이동 realistic 15 block),
    v11 · v11_realistic = 원본 NFS 바이트 사본. 모든 predictor 를 같은 프레임 · 같은 하드웨어 (vll3) 로 잰다 → 같은 판 안 비교만 한다.
  · v11 판 block = 이동 가림 (moving_occlusion · moving_occlusion_flat) 사라짐 위반 block 의 pos_a (물체) · pos_b (빈 장면).
출력: 배열 z_analysis/transfer_gap/results/r4_x2/E35r4/<set>.npz (+_meta.json) · 결과 z_analysis/transfer_gap/results/r4_x2/E35r4_endprior.json

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/transfer_gap_r4_endprior.py extract --sets orig,T1,T3_x0y0,T3_x64y0,T3_x0y64,T3_x64y64,v11late,v11real   (GPU 1)
  $P z_research/scripts/analysis/transfer_gap_r4_endprior.py extract --sets orig --limit 2 --preds base     (배관 확인)
  $P z_research/scripts/analysis/transfer_gap_r4_endprior.py report
  $P z_research/scripts/analysis/transfer_gap_r4_endprior.py presence_ci     (참 자리 1 칸 안의 모델 간 차 CI → E35r4_presence_ci.json)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
sys.path.insert(0, str(REPO))
import transfer_gap_endprior as ep  # noqa: E402

NFS = REPO / "z_pixel_decoder/runs/_frames"
R = REPO / "z_training/runs"
RES = REPO / "z_analysis/transfer_gap/results/r4_x2"
ARR = RES / "E35r4"
E35_ARR = Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap/E35")     # vobtrain (h 만) 원본 (vll5)
PRED_ALL = [("release", R / "release_vith/latest.pt"), ("vobpft", R / "vobench_release_postft_g/latest.pt"),
            ("t0s0", R / "tg4_t0_s0/latest.pt"), ("t0s1", R / "tg4_t0_s1/latest.pt"),
            ("t1", R / "tg4_t1_endscatter/latest.pt"), ("t2", R / "tg4_t2_genmix/latest.pt")]
VOB = [f"data.root={REPO}/data_csv/transfer_gap_r4_vob", "data.index_csv=index_test_late.csv",
       f"data.frames_root={NFS}/VOBench_v1_256full"]
SETS = {
    "orig": ("vobench_v1_split2_test", VOB),
    "T1": ("vobench_v1_split2_test", VOB + ["data.frames_start=15", "data.frames_stride=2"]),
    **{f"T3_{w}": ("vobench_v1_split2_test", [f"data.root={REPO}/data_csv/transfer_gap_r4_vob", "data.index_csv=index_T3_lateperm.csv",
                                             f"data.frames_root={NFS}/tg4_frames_T3_{w}"]) for w in ("x0y0", "x64y0", "x0y64", "x64y64")},
    "v11late": ("v11_split_test", [f"data.root={REPO}/data_csv/transfer_gap_r4_v11", "data.index_csv=index_test_latevanish.csv",
                                   f"data.frames_root={NFS}/tg4_IntPhysGen_v11"]),
    "v11real": ("v11_realistic", [f"data.root={REPO}/data_csv/transfer_gap_r4_v11real", "data.index_csv=index_occ.csv",
                                  f"data.frames_root={NFS}/tg4_IntPhysGen_v11_realistic"]),
    "vobtrain": ("vobench_v1_split2_test", ["data.index_csv=index_train.csv"]),      # h 만 — R3 배열을 복사해 쓴다 (다시 뽑지 않음)
}
_orig_pick = ep.pick_blocks


def pick_blocks(W, name):
    if name not in ("v11late", "v11real"):
        return _orig_pick(W, name)
    from collections import defaultdict
    by = defaultdict(dict)
    for i, r in enumerate(W.records):
        by[r.block_id][r.raw.get("variant")] = i
    out = []
    for b, v in by.items():
        if "pos_a" not in v or "pos_b" not in v:
            continue
        raw = W.records[v["pos_a"]].raw
        if raw.get("condition") not in ("moving_occlusion", "moving_occlusion_flat") or raw.get("violation_type") != "vanish":
            continue
        out.append({"block": b, "a": v["pos_a"], "b": v["pos_b"], "occ": "late", "k": str(raw.get("sym_k")),
                    "motion": raw.get("condition"), "objset": name, "violation": "permanence", "source_block": raw.get("source_block"),
                    "file_name": raw.get("file_name"), "dir_meta": raw.get("direction")})
    return sorted(out, key=lambda x: int(x["block"]) if str(x["block"]).isdigit() else str(x["block"]))


def setup(preds: str):
    ARR.mkdir(parents=True, exist_ok=True)
    for f in ("vobtrain.npz", "vobtrain_meta.json"):
        if not (ARR / f).exists() and (E35_ARR / f).exists():
            shutil.copyfile(E35_ARR / f, ARR / f)
    sel = PRED_ALL if preds == "all" else PRED_ALL[:2] if preds == "base" else [p for p in PRED_ALL if p[0] in preds.split(",")]
    ep.PREDS = [(t, str(p)) for t, p in sel]
    ep.SETS = SETS
    ep.ARR = ARR
    ep.TG = ARR                      # E35b (v11 PFT 의 v11 끝 자리) 절은 E32 배열이 없으므로 건너뛴다
    ep.OUT = RES / "E35r4_endprior.json"
    ep.pick_blocks = pick_blocks
    # E35 extract 는 모델을 짓는 첫 config 를 VOBench 원본 경로 (vll5 /data2) 로 푼다 → 다른 노드에서는 사본 경로로 바꿔 준다
    import transfer_gap_te as tte
    _res = tte.resolve

    def resolve(dataset, model, sets, window, limit):
        if dataset == "vobench_v1_split2_test" and not sets:
            sets = VOB
        return _res(dataset, model, sets, window, limit)
    tte.resolve = resolve


def presence_ci():
    """참 자리 1 칸 안 (t15 열 · 2 차원) 의 모델 간 차 — block bootstrap 2,000 (R2 규칙 1 표에 CI 를 붙인다).
    출력: results/r4_x2/E35r4_presence_ci.json"""
    import json
    import numpy as np
    out = {}
    for name in ("orig", "T1", "v11late", "v11real"):
        Z, M = ep.load(name)
        rh, ch, _, _ = ep.loc(Z["Eh"])
        nc, nf = M["nc"], M["nf"]
        tl = nc + nf - 1
        d = ep.direction(ch)
        Tc, Tr = ch[:, tl], rh[:, tl]
        n = len(Tc)
        base = np.array([bool((s := (d == d[j])).sum() > 1 and abs(np.median(np.delete(Tc, j)[np.delete(s, j)]) - Tc[j]) <= 1)
                         for j in range(n)], float)
        hit, hit2 = {"baseline_dir_typical": base}, {}
        for t in M["tags"]:
            r_, c_, _, _ = ep.loc(Z[f"E_{t}"])
            hit[t] = (np.abs(c_[:, nf - 1] - Tc) <= 1).astype(float)
            hit2[t] = (np.maximum(np.abs(c_[:, nf - 1] - Tc), np.abs(r_[:, nf - 1] - Tr)) <= 1).astype(float)
        rng = np.random.default_rng(0)
        B = rng.integers(0, n, (2000, n))
        o = {"n_block": n, "within1col": {k: round(float(v.mean()), 3) for k, v in hit.items()},
             "within1_2d": {k: round(float(v.mean()), 3) for k, v in hit2.items()}, "diff_ci": {}}
        for a in ("t1", "t2", "t0s1", "vobpft"):
            for b in ("t0s0", "t0s1", "baseline_dir_typical"):
                if a in hit and b in hit and a != b:
                    dd = (hit[a] - hit[b])[B].mean(1)
                    o["diff_ci"][f"{a}-{b}"] = {"v": round(float((hit[a] - hit[b]).mean()), 3),
                                                "ci": [round(float(np.percentile(dd, 2.5)), 3), round(float(np.percentile(dd, 97.5)), 3)]}
        out[name] = o
        print(name, o["within1col"], {k: v for k, v in o["diff_ci"].items() if k.startswith(("t1", "t2"))}, flush=True)
    json.dump(out, open(RES / "E35r4_presence_ci.json", "w"), indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["extract", "report", "presence_ci"])
    ap.add_argument("--sets", default="orig,T1,T3_x0y0,T3_x64y0,T3_x0y64,T3_x64y64,v11late,v11real")
    ap.add_argument("--preds", default="all", help="all | base | 콤마 태그")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    setup(a.preds)
    if a.part == "extract":
        ep.extract(a.sets.split(","), a.limit)
    elif a.part == "report":
        ep.report()
    else:
        presence_ci()


if __name__ == "__main__":
    main()
