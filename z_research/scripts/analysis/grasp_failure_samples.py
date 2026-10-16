#!/usr/bin/env python
"""GRASP 에서 AR 이 못하는 시나리오의 실패 쌍을 골라 영상을 한 폴더로 복사한다 (2026-10-16, 사용자 요청 "샘플 3 개씩").

    python z_research/scripts/analysis/grasp_failure_samples.py [--out <폴더>] [--n 3]

채점은 garrido_rescore 의 공식 규칙 그대로 (A.8 격자 최고 combo · 시나리오마다 최고 C · AvgSurprise).
시나리오 고르기: AR 정확도 < 55 이거나 release 보다 8 pt 넘게 낮은 시나리오.
쌍 고르기: AR 이 틀리고 (s_IP <= s_P) release 가 맞힌 쌍을 AR 의 상대 margin (s_IP − s_P) / s_P 이 가장 나쁜 순으로 n 개.
           모자라면 AR 이 틀린 쌍으로 채운다.
출력: <out>/<시나리오>/<순위>_<번호>_{P,IP}.mp4 + <out>/README.md (쌍마다 세 모델의 surprise · margin)
"""
import argparse, os, pathlib, shutil, sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import garrido_rescore as G  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
MODELS = [  # (이름, 결과 폴더 base, collect 의 model 인자)
    ("release", REPO / "z_research/Benchmarks/exp_results", "vith"),
    ("AR", REPO / "z_training/runs/natural_ar_vw_state/eval", "latest"),
    ("AR+VOB", REPO / "z_training/runs/natural_ar_vw_state_vob/eval", "latest"),
]


def video_scores(base, model):
    """-> (best, {scenario: {video_id: AvgSurprise}}, meta) — 공식 규칙으로 고른 combo · 시나리오별 C 의 영상 점수."""
    r = G.collect(base, "grasp_level2", model)
    best = r["best"]
    combo = best["combo"]
    d = next(pathlib.Path(x) for x in r["dirs"] if combo.split("_")[1] in pathlib.Path(x).name.split("_")[-1])
    import json
    pw = json.load(open(d / "per_window.json"))["windows"]
    _, meta = G.load_index(d)
    vids, _, Cs, arr = G.score_table(pw, combo)
    out = {}
    for g, v in best["per_group"].items():
        red = v["reduction"]
        m = np.nanmin(arr, axis=2) if red == "filtered" else arr[:, :, Cs.index(int(red[1:]))]
        out[g] = {vid: float(np.nanmean(m[i])) for i, vid in enumerate(vids) if meta[vid]["block_type"] == g}
    return best, out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/world_model/grasp_ar_failures")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    S, B, meta = {}, {}, None
    for name, base, model in MODELS:
        B[name], S[name], meta = video_scores(base, model)
    acc = {n: {g: v["acc"] for g, v in B[n]["per_group"].items()} for n in S}
    scen = [g for g in acc["AR"] if acc["AR"][g] < 55 or acc["release"][g] - acc["AR"][g] > 8]
    scen.sort(key=lambda g: acc["AR"][g] - acc["release"][g])
    out = pathlib.Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    lines = ["# GRASP — AR 이 못하는 시나리오의 실패 쌍 (2026-10-16)", "",
             "채점: 공식 규칙 (A.8 격자 최고 combo · 시나리오마다 최고 C · 창 평균 surprise). "
             f"combo — release `{B['release']['combo']}`, AR `{B['AR']['combo']}`, AR+VOB `{B['AR+VOB']['combo']}`.",
             "정답 = 불가능 (IP) 영상의 surprise 가 가능 (P) 영상보다 크다. margin = (s_IP − s_P) / s_P × 100 (%) — **음수면 틀림**.",
             "고른 쌍: AR 이 틀리고 release 가 맞힌 쌍 중 AR margin 이 가장 나쁜 순 (모자라면 AR 이 틀린 쌍).", "",
             "재현: `python z_research/scripts/analysis/grasp_failure_samples.py`", ""]
    for g in scen:
        pairs = {}
        for vid, r in meta.items():
            if r["block_type"] == g and vid in S["AR"][g]:
                pairs.setdefault(r["pair_id"], {})["P" if r["plausible"] == "1" else "IP"] = vid
        rows = []
        for pid, pr in pairs.items():
            if set(pr) != {"P", "IP"}:
                continue
            mg = {n: 100 * (S[n][g][pr["IP"]] - S[n][g][pr["P"]]) / S[n][g][pr["P"]] for n in S}
            rows.append((pid, pr, mg))
        wrong = [x for x in rows if x[2]["AR"] <= 0]
        pick = sorted([x for x in wrong if x[2]["release"] > 0], key=lambda x: x[2]["AR"])[: a.n]
        if len(pick) < a.n:
            pick += sorted([x for x in wrong if x not in pick], key=lambda x: x[2]["AR"])[: a.n - len(pick)]
        lines += [f"## {g} — 정확도 release {acc['release'][g]:.1f} · AR {acc['AR'][g]:.1f} · AR+VOB {acc['AR+VOB'][g]:.1f} "
                  f"(AR 이 틀린 쌍 {len(wrong)} / {len(rows)})", "",
                  "| 순위 | 쌍 | 파일 (P / IP) | margin release | margin AR | margin AR+VOB |", "|---|---|---|---:|---:|---:|"]
        for k, (pid, pr, mg) in enumerate(pick, 1):
            num = pathlib.Path(meta[pr["P"]]["file"]).stem
            dst = out / g
            dst.mkdir(parents=True, exist_ok=True)
            names = []
            for tag in ("P", "IP"):
                fn = f"{k}_{num}_{tag}.mp4"
                shutil.copy2(meta[pr[tag]]["file"], dst / fn)
                names.append(f"`{g}/{fn}`")
            lines.append(f"| {k} | {pid} | {' / '.join(names)} | {mg['release']:+.2f} | {mg['AR']:+.2f} | {mg['AR+VOB']:+.2f} |")
        lines.append("")
    (out / "README.md").write_text("\n".join(lines))
    print(out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
