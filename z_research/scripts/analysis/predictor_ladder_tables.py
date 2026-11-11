#!/usr/bin/env python3
"""predictor 사다리 + 끊기 T1 · T2 + 인과 문맥 (R1) 표 (2026-11-06, cvpr_auto 요청). GPU 불필요.

입력 = z_analysis/results/release_ladder_20261106/run_predictor_ladder_v2.sh 의 채점 결과 (cvpr/results/eval/testbed/<벤치>_<모델><접미>).
판 접미: 기본 _fb (없으면 접미 없는 기존 채점) · T1 _kopast · T2 _kolast · R1 _ctxcausal_fb · R1+T2 _ctxcausal_kolast.
점수 = 시트 점수 (사라짐 쌍 A 와 나타남 쌍 B 의 평균), 칸 '점수 (A/B)'. 미래 블록별 = per_video_surprise 의 fb1..fb8 로 같은 짝 판정.
  /venv/vjepa/bin/python z_research/scripts/analysis/predictor_ladder_tables.py [--out <md>]
"""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from scene_paired_tests import _blocks, boards, scene_stats, summarize, RES  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
D = f"{REPO}/data_csv"

# (이름, 문맥 encoder, 문맥 입력, 예측 방식, 학습량)
MODELS = [
    ("vith", "V-JEPA 2 (릴리즈)", "online", "영상 통째", "한 번에 · 양방향", "릴리즈 사전학습"),
    ("vast_tube_cd64", "tube (릴리즈 레시피 재학습)", "online", "영상 통째", "한 번에 · 양방향", "63 ep + 64 장 cooldown 7 ep"),
    ("vast_ar_oneshot_vidctx", "영상 문맥 one-shot", "target", "영상 통째", "한 번에 · 양방향", "70 ep"),
    ("vast_ar_oneshot_vidctx_causal", "영상 문맥 one-shot · 블록 인과", "target", "영상 통째", "한 번에 · 블록 인과", "70 ep"),
    ("vast_ar_oneshot_causal", "블록 문맥 one-shot · 블록 인과", "target", "블록별", "한 번에 · 블록 인과", "14 ep (학습량 1/5)"),
    # ("vast_ar_vw_state_vidctx", ...) — 2026-11-06 사용자 "영상 문맥 AR 필요 없어" 로 표에서 뺐다 (채점 결과는 남아 있다)
    ("vast_ar_vw_state", "블록 AR", "target", "블록별", "한 걸음씩 (상태 되먹임)", "70 ep"),
    ("vast_ar_vw_state_v4vc", "블록 AR + v4 (L1)", "target", "블록별", "한 걸음씩", "70 ep + v4 2,000 step"),
    ("vast_ar_vw_state_v4vc_negpair", "블록 AR + v4 (짝 손실 λ 1)", "target", "블록별", "한 걸음씩", "70 ep + v4 2,000 step"),
]
VARIANTS = [("기본", "_fb"), ("T1 과거 끊기", "_kopast"), ("T2 블록 8 끊기", "_kolast"), ("R1 인과 문맥", "_ctxcausal_fb"), ("R1 + T2", "_ctxcausal_kolast")]


def load(pat, m, suf, key="all"):
    for s in ([suf, ""] if suf == "_fb" else [suf]):
        f = f"{RES}/{pat.format(m=m)}{s}/per_block.json"
        if os.path.exists(f):
            pv = json.load(open(f))["per_video_surprise"]
            if key != "all" and not isinstance(next(iter(pv.values())), dict):
                continue
            return {k: (v[key] if isinstance(v, dict) else v) for k, v in pv.items()}
    return None


def score(S, blk):
    if S is None or not blk:
        return None
    r = scene_stats(S, blk)
    if r is None:
        return None
    s = summarize(r)
    return (s["A"] + s["B"]) / 2, s["A"], s["B"], s["n"]


def fmt(v, ab=True):
    if v is None:
        return "—"
    return f"{v[0]:.1f} ({v[1]:.0f}/{v[2]:.0f})" if ab else f"{v[1]:.0f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); a = ap.parse_args()
    out = []
    P = lambda *x: out.append(" ".join(str(y) for y in x))  # noqa: E731
    B = boards()
    mv3 = {r["block_id"] for r in csv.DictReader(open(f"{D}/vobench_v3/index.csv")) if r["motion"] != "flat_static"}
    mvS = {r["block_id"] for r in csv.DictReader(open(f"{D}/intphys1_surrogate/index.csv")) if r["motion"] != "flat_static"}
    v3 = {k: B[k] for k in ("VOBench v3 late · panel", "VOBench v3 late · moving occluder")}
    v3_all = ("testbed/vobench_v3_{m}", {**v3["VOBench v3 late · panel"][1], **v3["VOBench v3 late · moving occluder"][1]})
    v1 = f"{D}/vobench_v1_perm/index.csv"
    v1cells = {}
    if os.path.exists(v1):
        for occ in ("visible", "early", "mid", "late"):
            for mov in (True, False):
                v1cells[(occ, mov)] = _blocks(v1, lambda r, o=occ, mm=mov: r["occlusion"] == o and (r["motion"] != "flat_static") == mm)

    # (5) 모델 표
    P("## 판 (문맥 encoder · 문맥 입력 · 예측 방식 · 학습량)\n")
    P("| 판 | 문맥 encoder | 문맥 입력 | 예측 방식 | 학습량 |"); P("|---|---|---|---|---|")
    for m, nm, enc, inp, pred, amt in MODELS:
        P(f"| {nm} | {enc} | {inp} | {pred} | {amt} |")

    # (1) VOB3 late
    P("\n## (1) VOBench v3 late — 시트 점수 (A 사라짐 / B 나타남)\n")
    P("| 판 | 조건 | 판 가림막 · 움직임 | 판 가림막 · 정지 | 지나가는 가림막 · 움직임 | 지나가는 가림막 · 정지 | 전체 움직임 | 전체 정지 |")
    P("|---|---|---|---|---|---|---|---|")
    for m, nm, *_ in MODELS:
        for vn, suf in VARIANTS:
            S = load("testbed/vobench_v3_{m}", m, suf)
            if S is None:
                continue
            cells = []
            for bn in v3:
                for mov in (True, False):
                    cells.append(score(S, {b: x for b, x in v3[bn][1].items() if (b in mv3) == mov}))
            for mov in (True, False):
                cells.append(score(S, {b: x for b, x in v3_all[1].items() if (b in mv3) == mov}))
            P(f"| {nm} | {vn} | " + " | ".join(fmt(c) for c in cells) + " |")

    # (2) VOB1 permanence 가시 대조
    if v1cells:
        P("\n## (2) VOBench v1 permanence — 가시 vs 가림 (시트 점수, A/B). 블록 수: "
          + ", ".join(f"{o}·{'움직' if mm else '정지'} {len(v)}" for (o, mm), v in v1cells.items()) + "\n")
        P("| 판 | 조건 | 가시 · 움직임 | 가시 · 정지 | early · 움직임 | mid · 움직임 | late · 움직임 | late · 정지 |")
        P("|---|---|---|---|---|---|---|---|")
        order = [("visible", True), ("visible", False), ("early", True), ("mid", True), ("late", True), ("late", False)]
        for m, nm, *_ in MODELS:
            for vn, suf in VARIANTS:
                S = load("testbed/vobench_v1_perm_{m}", m, suf)
                if S is None:
                    continue
                P(f"| {nm} | {vn} | " + " | ".join(fmt(score(S, v1cells[c])) for c in order) + " |")

    # (3) 미래 블록별 사라짐 A
    P("\n## (3) 미래 블록 9..16 별 사라짐 쌍 정확도 A (%) — 기본 판, 블록 j 의 surprise 만으로 같은 짝 판정\n")
    sets = [("VOB3 late · 움직임", "testbed/vobench_v3_{m}", {b: x for b, x in v3_all[1].items() if b in mv3})]
    if v1cells:
        sets += [("VOB1 가시 · 움직임", "testbed/vobench_v1_perm_{m}", v1cells[("visible", True)]),
                 ("VOB1 late · 움직임", "testbed/vobench_v1_perm_{m}", v1cells[("late", True)])]
    sets += [("Surrogate · 움직임", "testbed/intphys1_surrogate_{m}", {b: x for b, x in B["IntPhys1 Surrogate"][1].items() if b in mvS})]
    for sn, pat, blk in sets:
        P(f"\n**{sn}** (장면 {len(blk)})\n")
        P("| 판 | " + " | ".join(f"블록 {9 + j}" for j in range(8)) + " | 전체 |"); P("|---|" + "---:|" * 9)
        for m, nm, *_ in MODELS:
            row = []
            for j in range(8):
                row.append(fmt(score(load(pat, m, "_fb", f"fb{j + 1}"), blk), ab=False))
            if all(x == "—" for x in row):
                continue
            row.append(fmt(score(load(pat, m, "_fb"), blk), ab=False))
            P(f"| {nm} | " + " | ".join(row) + " |")

    # (4) 다른 벤치 기본 칸
    P("\n## (4) 다른 벤치 — 기본 판 시트 점수 (A/B)\n")
    other = [("Surrogate 움직임", "IntPhys1 Surrogate", mvS, True), ("Surrogate 정지", "IntPhys1 Surrogate", mvS, False),
             ("real95", "real world (all 95)", None, None), ("real48 보류", "real world (held-out 48)", None, None),
             ("IP2 late", "IntPhys2 late · permanence (A=pair 1, B=pair 2)", None, None)]
    P("| 판 | 조건 | " + " | ".join(o[0] for o in other) + " |"); P("|---|---|" + "---|" * len(other))
    for m, nm, *_ in MODELS:
        for vn, suf in VARIANTS:
            row, any_ = [], False
            for _, bn, mvset, mov in other:
                pat, blk = B[bn]
                if mvset is not None:
                    blk = {b: x for b, x in blk.items() if (b in mvset) == mov}
                v = score(load(pat, m, suf), blk)
                any_ |= v is not None
                row.append(fmt(v))
            if any_:
                P(f"| {nm} | {vn} | " + " | ".join(row) + " |")
    txt = "\n".join(out)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
