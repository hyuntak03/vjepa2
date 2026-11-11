#!/usr/bin/env python3
"""전이 판정 표 (2026-11-09, 사용자 "데이터를 바꿀 거면 그 데이터로 학습된 L1 만 판도 있어야 돼. 그에 비해 다른 건 유지하고 IP2 와 real 이 올라야").
판마다 자기 데이터의 L1 만 이어 학습 판 (기준) 대비 시트 칸 차이를 내고 통과 여부를 매긴다. GPU 불필요 (sheet_table 과 같은 칸 계산).

  유지 칸 (합성): VOBench v3 4 칸 (판 · 지나가는 × 움직임 · 정지) · IntPhys1 Surrogate 2 칸 — 기준보다 TOL 점 넘게 떨어지면 탈락
  참고 칸: IntPhys1 late (판정 안 함)
  올라야 할 칸: IntPhys2 late · real 95 — 기준보다 높아야 통과 (real 보류 48 은 같이 보인다)
  /venv/vjepa/bin/python z_research/scripts/analysis/transfer_table.py <판>=<기준> ... [--tol 3]
"""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from sheet_table import COLS, REPO, boards, cell

KEEP = ["VOB판움직", "VOB판정지", "VOB지나움직", "VOB지나정지", "Sur움직", "Sur정지"]
RISE = ["IP2late", "real95"]
INFO = ["IP1late", "real48"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="<판>=<기준 L1 판>")
    ap.add_argument("--tol", type=float, default=3.0)
    a = ap.parse_args()
    B = boards()
    mv = {d: {r["block_id"] for r in csv.DictReader(open(f"{REPO}/data_csv/{d}/index.csv")) if r["motion"] != "flat_static"}
          for d in ("vobench_v3", "intphys1_surrogate")}
    get = lambda m: {c[0]: cell(B, mv, m, "", c[1], c[2], c[3]) for c in COLS}
    cols = KEEP + RISE + INFO
    print("| 판 (기준 대비 차이, 괄호 = 판의 점수 A/B) | " + " | ".join(cols) + " | 판정 |\n|---|" + "---|" * (len(cols) + 1))
    for pr in a.pairs:
        m, base = pr.split("=", 1)
        x, b = get(m), get(base)
        txt, ok, why = [], True, []
        for c in cols:
            if x[c] is None or b[c] is None:
                txt.append("—"); ok = ok and c in INFO
                if c not in INFO:
                    why.append(f"{c} 없음")
                continue
            d = x[c][0] - b[c][0]
            txt.append(f"{d:+.1f} ({x[c][0]:.1f}: {x[c][1]:.0f}/{x[c][2]:.0f})")
            if c in KEEP and d < -a.tol:
                ok = False; why.append(f"{c} {d:+.1f}")
            if c in RISE and d <= 0:
                ok = False; why.append(f"{c} 안 오름")
        print(f"| {m} vs {base} | " + " | ".join(txt) + f" | {'✅ 통과' if ok else '❌ ' + ', '.join(why)} |")


if __name__ == "__main__":
    main()
