"""attn_block9_object.py 결과 → 표 (열 = 문맥 블록 1–8). 2026-11-06."""
import argparse
import json

import numpy as np

NAMES = {"release": "V-JEPA 2", "release_causal": "V-JEPA 2 + 인과 문맥 (R1)", "ar": "AR",
         "ar_v4bc4": "AR → v4bc4 L1", "ar_v4bc4_neg": "AR → v4bc4 음의 손실"}   # 2026-11-07: 같은 구조, 학습 데이터만 다름


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--layers", default="all", help="all 또는 0 부터 센 층 범위 예: 0-0")
    a = ap.parse_args()
    d = json.load(open(a.json))
    meta = d["meta"]
    lay = slice(None) if a.layers == "all" else slice(int(a.layers.split("-")[0]), int(a.layers.split("-")[1]) + 1)

    def groups():
        out = {}
        for i, m in enumerate(meta):
            mv = "정지" if m["motion"] == "flat_static" else "움직임"
            out.setdefault(f"{m['occlusion']} · {mv}", []).append(i)
            if m["occlusion"] == "late" and mv == "움직임":
                out.setdefault(f"late k{m['k']} · 움직임", []).append(i)
        return out

    G = groups()
    order = ["visible · 움직임", "early · 움직임", "mid · 움직임", "late · 움직임", "late k1 · 움직임", "late k2 · 움직임",
             "late k3 · 움직임", "visible · 정지", "late · 정지"]
    for g in order:
        ids = G.get(g)
        if not ids:
            continue
        vis = np.array([meta[i]["vis"] for i in ids]).mean(0)
        print(f"\n**{g}** (클립 {len(ids)}) — 블록 9 를 만드는 query (블록 9 시점 물체 자리 3×3) → 블록 t 의 물체 자리 3×3 attention 질량 (%)\n")
        print("| 모델 | " + " | ".join(f"블록 {t+1}" for t in range(8)) + " | 문맥 합 | 미래 자리 9–16 합 |")
        print("|---|" + "---:|" * 10)
        for mdl, per in d["models"].items():
            x = np.array([per[i] for i in ids])[:, lay].mean((0, 1))        # (NB, 2)
            o = x[:, 0]
            print(f"| {NAMES[mdl]} | " + " | ".join(f"{100*v:.2f}" for v in o[:8]) +
                  f" | {100*o[:8].sum():.2f} | {'—' if mdl == 'ar' else f'{100*o[8:].sum():.2f}'} |")
        for mdl, per in d["models"].items():
            x = np.array([per[i] for i in ids])[:, lay].mean((0, 1))
            lift = x[:8, 0] / np.maximum(x[:8, 1], 1e-9) / (9 / 256)
            print(f"| ({NAMES[mdl]}: 블록 안 물체 자리 몫 ÷ 면적 비) | " + " | ".join(f"×{v:.1f}" for v in lift) + " | | |")
        print("| (물체 가시 비율) | " + " | ".join(f"{v:.2f}" for v in vis[:8]) + f" | | 블록 9: {vis[8]:.2f} |")


if __name__ == "__main__":
    main()
