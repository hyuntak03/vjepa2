#!/usr/bin/env python3
"""transfer_gap Q3 · E16 — PFT 가 분포 안 · 가까운 · 밖 영상에서 predictor 출력을 얼마나 바꾸나 (GPU 1, 약 15 분). 2026-10-30, 실험 B [X2].

왜: E9 에서 VOBench PFT 의 분포 밖 가능 영상 오차 비 ε 가 1.00–1.03 이었다. 두 경우가 갈린다 —
  (가) PFT 갱신이 분포 밖 입력에서는 출력을 거의 안 바꾼다 (갱신이 VOBench 같은 입력에서만 켜짐)
  (나) 출력은 크게 바뀌지만 오차를 줄이지도 늘리지도 않는 방향으로 바뀐다
정의 (가능 영상만, 표준 c16t32 창, 미래 8 튜블릿 × 256 칸 × 1280)
  e_rel = mean | p_rel − h |,  e_pft = mean | p_pft − h |,  d = mean | p_pft − p_rel |   (h = LN(목표 encoder(32 장))[미래])
  비 d / e_rel  (출력 변화량을 릴리즈 오차 크기로 잰 것) · 집중도 = 변화량 상위 5 % 토큰이 전체 변화량에서 차지하는 몫
  ρ_move = 토큰별 변화량 ‖p_pft − p_rel‖₁ 과 토큰별 실제 변화 ‖h − LN(z)[문맥 끝]‖₁ 의 상관 (영상마다 Spearman 대신 Pearson, 평균)
  cos = cos(p_pft − p_rel, h − p_rel) (영상 하나를 펼친 벡터) — 변화가 릴리즈 오차를 줄이는 방향인가 (0 = 직교)
  d_offset = |튜블릿마다 256 칸 평균 변화| 의 평균 (화면 전체가 같이 움직이는 몫), d_centered = 그것을 뺀 변화량
같은 encoder · 같은 창 · 같은 영상에서 두 predictor 를 비교한다 (te_v11_score.Scorer, keep_p).

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/transfer_gap_pdiff.py --per 160   → z_analysis/transfer_gap/results/q3/E16_pdiff.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

import te_v11_score as te  # noqa: E402

OUT = REPO / "z_analysis/transfer_gap/results/q3/E16_pdiff.json"
import os
if os.environ.get("TG_OUT"):
    OUT = Path(os.environ["TG_OUT"])
SETS = {   # 이름 → (레지스트리 데이터셋, --set 목록, 분류)
    "vobench_split2_test": ("vobench_v1_split2_test", [], "in-dist"),
    "v11sub": ("v11_split_test", ["data.root=" + str(REPO / "data_csv/transfer_gap_v11sub"), "data.index_csv=index.csv"], "near"),
    "v11_realistic": ("v11_realistic", [], "near"),
    "intphys1_dev": ("intphys1_dev", [], "external"),
    "grasp_level2_s5": ("grasp_level2", ["data.frames_stride=5"], "external"),
    "inflevel_continuity_s5": ("inflevel_continuity", ["data.frames_stride=5"], "external"),
}


def resolve(ds, sets):
    with tempfile.TemporaryDirectory() as td:
        o = Path(td) / "c.yaml"
        args = [sys.executable, str(REPO / "cvpr/harness/resolve.py"), "eval/testbed", ds, "vith", "-o", str(o), "--quiet"]
        for kv in sets:
            args += ["--set", kv]
        r = subprocess.run(args, text=True, capture_output=True, cwd=str(REPO))
        if r.returncode:
            sys.exit(f"resolve 실패 {ds}: {r.stderr}")
        return yaml.safe_load(o.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=160, help="데이터셋마다 가능 영상 수 (고르게)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pft", default=str(REPO / "z_training/runs/vobench_release_postft_g/latest.pt"))
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    cfg0 = resolve("vobench_v1_split2_test", [])
    preds = [("release", str(REPO / "z_training/runs/release_vith/latest.pt")), ("pft", a.pft)]
    sc = te.Scorer(cfg0, dev, preds)
    D, S, nc, nf = sc.D, sc.S, sc.nc, sc.nf
    res = {"pft": a.pft, "per": a.per, "definition": __doc__.split("정의")[1].split("같은 encoder")[0].strip(), "sets": {}}
    for name, (ds, sets, kind) in SETS.items():
        cfg = resolve(ds, sets)
        for k in ("n_frames",):
            assert int(cfg["data"][k]) == int(cfg0["data"][k])
        W = te.WMADataset(cfg)
        pos = [i for i, r in enumerate(W.records) if str(r.plausible) == "1"]
        pick = [pos[int(x)] for x in np.linspace(0, len(pos) - 1, min(a.per, len(pos)))]
        rows = []
        for b0 in range(0, len(pick), a.batch):
            idx = pick[b0:b0 + a.batch]
            clips = torch.stack([W.clip(i) for i in idx])
            with torch.inference_mode():
                r = sc.run(clips, keep_p=True)
                z, B = r["_z"], clips.size(0)
            # 목표 h: Scorer.encode 와 같은 식 (autocast 안에서 LN(target_encoder(32 장)))
            with torch.inference_mode(), sc.autocast():
                x = clips.to(dev, sc.bundle.dtype)
                hh = F.layer_norm(sc.bundle.target_encoder(x), (D,)).float().reshape(B, nc + nf, S, D)
                zT = F.layer_norm(z.float(), (D,)).reshape(B, nc, S, D)[:, nc - 1:nc].float()
            hf = hh[:, nc:]
            pr, pf = r["_p"]["release"], r["_p"]["pft"]
            e_rel = (pr - hf).abs().mean((-1, -2, -3)); e_pft = (pf - hf).abs().mean((-1, -2, -3))
            dt = (pf - pr).abs().mean(-1)                    # (B, nf, S) 토큰별 변화량
            mv = (hf - zT).abs().mean(-1)                    # (B, nf, S) 토큰별 실제 변화 (문맥 끝 대비)
            dl = pf - pr                                     # (B, nf, S, D) 변화
            er = hf - pr                                     # (B, nf, S, D) 릴리즈 오차 방향
            off = dl.mean(2, keepdim=True)                   # 튜블릿마다 칸 평균 변화 (전역 이동)
            for j in range(B):
                v = dt[j].flatten(); m = mv[j].flatten()
                top = torch.topk(v, max(1, int(0.05 * v.numel()))).values.sum() / v.sum()
                rho = torch.corrcoef(torch.stack([v, m]))[0, 1]
                cos = F.cosine_similarity(dl[j].flatten(), er[j].flatten(), dim=0)
                d_off = off[j].abs().mean(); d_cen = (dl[j] - off[j]).abs().mean()
                rows.append([float(e_rel[j]), float(e_pft[j]), float(v.mean()), float(top), float(rho), float(cos), float(d_off), float(d_cen)])
            del r
        A = np.array(rows)
        res["sets"][name] = {"kind": kind, "dataset": ds, "sets": sets, "n_video": len(rows),
                             "e_rel": round(float(A[:, 0].mean()), 5), "e_pft": round(float(A[:, 1].mean()), 5),
                             "eps": round(float(A[:, 1].mean() / A[:, 0].mean()), 4),
                             "d": round(float(A[:, 2].mean()), 5), "d_over_e_rel": round(float(A[:, 2].mean() / A[:, 0].mean()), 4),
                             "d_over_e_rel_q10_q90": [round(float(np.percentile(A[:, 2] / A[:, 0], q)), 4) for q in (10, 90)],
                             "top5pct_share": round(float(A[:, 3].mean()), 4), "rho_change_vs_motion": round(float(np.nanmean(A[:, 4])), 4),
                             "cos_change_vs_error": round(float(A[:, 5].mean()), 4),
                             "d_global_offset": round(float(A[:, 6].mean()), 5), "d_centered": round(float(A[:, 7].mean()), 5)}
        print(name, res["sets"][name], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("→", OUT)


if __name__ == "__main__":
    main()
