#!/usr/bin/env python3
"""정답 공간 교차 채점 (2026-11-05). GPU 불필요.
질문: AR 이 합성 보드에서 릴리즈 + v4 보다 잘하는 것이 '블록별 정답 공간' 덕인가, AR 자체 (블록 문맥 입력 · 굴리기) 덕인가.
  - AR 판:   기본 = 블록별 정답 (각 2 장 블록을 따로 target encoder 에) / 교차 = 창 전체 정답 (*_artgtwin, model.predictor.ar_target=window)
  - 릴리즈 판: 기본 = 창 전체 정답 / 교차 = 미래만 정답 (*_futonly, surprise.target_future_only) · 블록별 정답 (*_tgtblk, surprise.target_blockwise)
  - 블록별 복사 기준선은 두 경로 (AR ar_copy · 릴리즈 target_blockwise 의 copy_all) 가 상대 차 ≤ 2e-7 로 같다 (정합 검사).
두 교차 모두 학습 때와 다른 정답 공간이라 예측과 정답 사이에 체계적 어긋남이 있다 — 대칭 설계.
장면 단위: A (사라짐 짝 정답) · B (나타남 짝 정답) · D = 1[m_A + m_B > 0] · c (치우침, > 0 = 빈 미래가 싸다).
  /venv/vjepa/bin/python z_research/scripts/analysis/ar_target_space_tables.py > z_analysis/results/scene_paired_20261105/AR_TARGET_SPACE.md
"""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from scene_paired_tests import boards, scene_stats, summarize, RES

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# (행 이름, 모델 키, 결과 접미사, 정답 공간) — "(학습)" = 그 판이 학습한 공간
ROWS = [("V-JEPA 2 (릴리즈 그대로)", "vith", "", "창 전체 (학습)"),
        ("V-JEPA 2 (릴리즈 그대로)", "vith", "_tgtblk", "블록별"),
        ("릴리즈 + v4", "vith_release_v4vc", "", "창 전체 (학습)"),
        ("릴리즈 + v4", "vith_release_v4vc", "_futonly", "미래만"),
        ("릴리즈 + v4", "vith_release_v4vc", "_tgtblk", "블록별"),
        ("AR + v4", "vast_ar_vw_state_v4vc", "", "블록별 (학습)"),
        ("AR + v4", "vast_ar_vw_state_v4vc", "_artgtwin", "창 전체"),
        ("AR (v4 없음)", "vast_ar_vw_state", "", "블록별 (학습)"),
        ("AR (v4 없음)", "vast_ar_vw_state", "_artgtwin", "창 전체")]
# 영역 채점이 있는 (모델, 접미사, 공간). 릴리즈 + v4 의 창 전체 영역은 *_region
REGION_ROWS = [("릴리즈 + v4", "vith_release_v4vc", "_region", "창 전체 (학습)"), ("릴리즈 + v4", "vith_release_v4vc", "_futonly", "미래만"),
               ("릴리즈 + v4", "vith_release_v4vc", "_tgtblk", "블록별"),
               ("AR + v4", "vast_ar_vw_state_v4vc", "_region", "블록별 (학습)"), ("AR + v4", "vast_ar_vw_state_v4vc", "_artgtwin", "창 전체")]
# 복사 기준선 (마지막 문맥 블록 / 튜블릿을 미래에 그대로): (공간, 모델, 접미사, 키). 체크포인트 무관.
COPY = [("블록별", "vast_ar_vw_state_v4vc", "_arcopyblk", "all"), ("미래만", "vith", "_futonly", "copy_all"), ("창 전체", "vith", "_copychk", "copy_all")]


def load(pat, m, suf, key="all"):
    f = f"{RES}/{pat.format(m=m)}{suf}/per_block.json"
    if not os.path.exists(f):
        return None
    S = json.load(open(f))["per_video_surprise"]
    out = {}
    for k, v in S.items():
        if isinstance(v, dict):
            if key not in v:
                return None
            out[k] = v[key]
        else:
            if key != "all":
                return None
            out[k] = v
    return out


def moving_blocks(board):
    idx = list(csv.DictReader(open(f"{REPO}/data_csv/{board}/index.csv")))
    return {r["block_id"] for r in idx if r["motion"] != "flat_static"}


def fmt(S, blk):
    if S is None:
        return "— | — | — | —"
    r = scene_stats(S, blk)
    if r is None:
        return "영상 빠짐 | | |"
    s = summarize(r)
    return f"{s['A']:.1f} | {s['B']:.1f} | {s['C']:.1f} | {s['c']:+.2f}"


def main():
    B = boards()
    print("# 정답 공간 교차 채점 — AR 의 합성 우위는 정답 공간 덕인가 (2026-11-05, vast)\n")
    print(__doc__.split("  /venv")[0].replace('"""', ""))
    subsets = []
    for bname, board in (("VOBench v3 late · panel", "vobench_v3"), ("VOBench v3 late · moving occluder", "vobench_v3"),
                         ("IntPhys1 Surrogate", "intphys1_surrogate")):
        pat, blk = B[bname]; mv = moving_blocks(board)
        subsets.append((f"{bname} · 움직임", pat, {b: v for b, v in blk.items() if b in mv}))
        subsets.append((f"{bname} · 정지", pat, {b: v for b, v in blk.items() if b not in mv}))
    for bname in ("IntPhys1 late · O1 occluded", "IntPhys2 late · permanence (A=pair 1, B=pair 2)", "real world (all 95)"):
        pat, blk = B[bname]; subsets.append((bname, pat, blk))
    for sname, pat, blk in subsets:
        print(f"\n## {sname} (장면 {len(blk)})\n")
        print("| 판 | 정답 공간 | A | B | D | c |\n|---|---|---:|---:|---:|---:|")
        for lab, m, suf, sp in ROWS:
            print(f"| {lab} | {sp} | {fmt(load(pat, m, suf), blk)} |")
        for sp, m, suf, key in COPY:
            S = load(pat, m, suf, key)
            if S is not None:
                print(f"| 복사 (predictor 없음) | {sp} | {fmt(S, blk)} |")
    # 영역 (물체 3×3 · 먼 배경) — 움직임 장면, 영역 채점이 있는 두 판만
    print("\n## 영역별 — 움직임 장면 (물체 = 투영 중심 3×3 패치, 먼 배경 = 중심에서 3 칸 이상)\n")
    for sname, pat, blk in subsets[:6:2]:
        print(f"\n### {sname} (장면 {len(blk)})\n")
        print("| 판 | 정답 공간 | 영역 | A | B | D | c |\n|---|---|---|---:|---:|---:|---:|")
        for lab, m, suf, sp in REGION_ROWS:
            for reg in ("object", "far_bg"):
                print(f"| {lab} | {sp} | {'물체' if reg == 'object' else '먼 배경'} | {fmt(load(pat, m, suf, reg), blk)} |")
        for sp, m, suf, key in COPY:
            for reg in ("object", "far_bg"):
                S = load(pat, m, suf, key.replace("all", reg))
                if S is not None:
                    print(f"| 복사 (predictor 없음) | {sp} | {'물체' if reg == 'object' else '먼 배경'} | {fmt(S, blk)} |")


if __name__ == "__main__":
    main()
