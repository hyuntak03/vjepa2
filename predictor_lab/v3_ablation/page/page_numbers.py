"""v3 결과 페이지의 수치 — 정답률 + 같은 짝에서 한 팔만 맞힌 수 (McNemar 정확 검정, 양측).

table_v3.py 의 panel_scores (배치 병합 규칙 그대로) 를 다시 쓴다. 출력은 stdout 표 하나.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P predictor_lab/v3_ablation/page/page_numbers.py
"""
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import table_v3 as T  # noqa: E402

ARMS = ["copy", "release", "v3_mix680_s0", "v3_mix1600_s0", "v3_full_s0", "v3_mix1600_t25_s0", "v3t_only_s0", "vobftg"]
CELLS = [("v3test", "all"), ("v3test", "A"), ("v3ttest", "all"), ("v3ttest", "A"), ("v3ttest", "B"),
         ("clevrer", "all"), ("vomo", "all"), ("vomo", "A"), ("vomo", "B"), ("vob_late", "all"), ("vob_late", "A"), ("vob_late", "B"),
         ("ip1uni", "all"), ("ip2uni", "all"), ("rubric", "all")]
CMP = [("v3_mix1600_s0", "release"), ("v3_mix1600_t25_s0", "v3_mix1600_s0"), ("v3t_only_s0", "v3_mix1600_s0"),
       ("v3_mix1600_s0", "v3_mix680_s0"), ("v3_full_s0", "v3_mix1600_s0")]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    cache = {}
    for pn, sub in CELLS:
        if pn not in cache:
            cache[pn] = T.panel_scores(pn)
        S, P = cache[pn]
        f = dict(T.PANELS[pn])[sub]
        sel = [(a, b) for a, b, r in P if f(r)]
        J = {t: T._judge(S[t], sel) for t in ARMS if t in S}
        acc = " · ".join(f"{t.replace('_s0', '')} {100 * J[t].mean():.1f}" for t in ARMS if t in J)
        print(f"\n{pn}/{sub}  n={len(sel)}\n  {acc}")
        for x, y in CMP:
            if x in J and y in J:
                b = int(((J[x] == 1) & (J[y] < 1)).sum()); c = int(((J[y] == 1) & (J[x] < 1)).sum())
                print(f"  {x.replace('_s0', ''):>16} 만 {b:4d}  대  {y.replace('_s0', ''):<16} 만 {c:4d}   p={mcnemar(b, c):.3g}")


if __name__ == "__main__":
    main()
