#!/usr/bin/env python3
"""IntPhys1 방향 균형 감사 — 우리 데이터의 축퇴가 외부 벤치마크에도 있는가.

우리 설계는 모든 전이를 A/B 양방향으로 돌린다. IntPhys1 은 그렇게 만들어지지 않았다.
방향이 불균형하면 점수의 상당 부분이 appearance bias 기여일 수 있다.

방향 정의 (IntPhys1 에는 A/B 라벨이 없으므로 관측량에서 만든다)
  O1/O3 (vanish/appear) : imp 가 pos 보다 물체 보이는 프레임이 **적으면** '사라짐',
                          많으면 '나타남'   (videos.csv 의 obj_visible_frames)
  O2 (mesh change)      : 불가능 영상이 바뀐 결과 메시(magic_mesh)로 방향을 잡는다

  sensitivity = (acc_사라짐 + acc_나타남)/2 - 50
  bias        = |acc_사라짐 - acc_나타남|/2

  python z_research/scripts/intphys1_direction_audit.py \
      --result-dir z_exp/world_model_analysis/results/intphys1_ourproto_vith
"""
from __future__ import annotations
import argparse, collections, csv, json
from pathlib import Path
import numpy as np

B = Path("/local_datasets/world/world_analysis/IntPhys1_dev_by_scene")
MO = {"정지": "static", "이동": "moving"}
VI = {"눈앞": "visible", "가려짐": "occluded"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--combo", default=None, help="sliding 결과면 'skip2_w32/avg' 등")
    a = ap.parse_args()

    S = json.loads((a.result_dir / "per_block.json").read_text())["per_video_surprise"]
    def sc(v):
        x = S[v]
        return x[a.combo] if isinstance(x, dict) else x

    V = {r["video_id"]: r for r in csv.DictReader((B / "videos.csv").open())}
    P = list(csv.DictReader((B / "pairs.csv").open()))
    rows = []
    for r in P:
        p, i = r["pos"], r["imp"]
        dv = float(V[i]["obj_visible_frames"]) - float(V[p]["obj_visible_frames"])
        if r["principle"].startswith("O2"):
            d = f"->{V[i]['magic_mesh'] or '?'}"          # 바뀐 결과 메시
        else:
            d = "사라짐" if dv < 0 else ("나타남" if dv > 0 else "동수")
        rows.append(dict(pr=r["principle"].split("_")[0], d=d,
                         cond=f"{MO[r['label_motion']]}_{VI[r['label_vis']]}",
                         ok=1.0 if sc(i) > sc(p) else 0.0))

    overall = 100 * np.mean([x["ok"] for x in rows])
    print(f"{a.result_dir.name}   overall {overall:.2f}%   n={len(rows)}"
          + (f"   combo={a.combo}" if a.combo else "") + "\n")

    print("=== 1. 방향 균형 자체 ===")
    for pr in ["O1", "O2", "O3"]:
        c = collections.Counter(x["d"] for x in rows if x["pr"] == pr)
        n = sum(c.values())
        bal = " / ".join(f"{k} {v}({100*v/n:.0f}%)" for k, v in sorted(c.items()))
        print(f"  {pr}  n={n:<4} {bal}")

    print("\n=== 2. 방향별 정확도 ===")
    print(f"  {'principle':<5}{'condition':<20}" + "".join(f"{d:>14}" for d in ["사라짐", "나타남", "동수"])
          + f"{'sens':>8}{'|bias|':>8}")
    for pr in ["O1", "O3"]:
        for c in ["static_visible", "moving_visible", "moving_occluded", "static_occluded"]:
            sel = [x for x in rows if x["pr"] == pr and x["cond"] == c]
            if not sel: continue
            cells, acc = [], {}
            for d in ["사라짐", "나타남", "동수"]:
                s = [x["ok"] for x in sel if x["d"] == d]
                acc[d] = 100 * np.mean(s) if s else None
                cells.append(f"{acc[d]:.1f}(n{len(s)})" if s else "-")
            if acc["사라짐"] is not None and acc["나타남"] is not None:
                sens = (acc["사라짐"] + acc["나타남"]) / 2 - 50
                bias = abs(acc["사라짐"] - acc["나타남"]) / 2
                tail = f"{sens:>+8.1f}{bias:>8.1f}"
            else:
                tail = f"{'-':>8}{'-':>8}"
            print(f"  {pr:<5}{c:<20}" + "".join(f"{x:>14}" for x in cells) + tail)

    print("\n=== 3. O2 (mesh change) 방향별 ===")
    for c in ["static_visible", "moving_visible", "moving_occluded", "static_occluded"]:
        sel = [x for x in rows if x["pr"] == "O2" and x["cond"] == c]
        if not sel: continue
        cc = collections.defaultdict(list)
        for x in sel: cc[x["d"]].append(x["ok"])
        print(f"  {c:<20}" + "  ".join(f"{k} {100*np.mean(v):.0f}%(n{len(v)})"
                                       for k, v in sorted(cc.items())))


if __name__ == "__main__":
    main()
