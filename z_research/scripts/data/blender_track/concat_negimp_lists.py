#!/usr/bin/env python3
"""음의 손실 학습 목록 합치기 (2026-11-07): 묶음 정보 (families.json) 가 있는 목록 여러 개를 반복 횟수와 함께 이어 붙인다.
번호 (A · B) 는 합친 목록 기준으로 다시 매긴다. ImpTable 검사 (묶음 영상 합 = 목록 길이, 같은 목록) 를 통과하는 꼴.
  $BT_PY concat_negimp_lists.py --name blender_track_v4bc4f \
      --part v4bc4:data_csv/blender_track_v4bc4_negimp/families.json:1 --part v4f:data_csv/blender_track_v4f_cover_negimp/families.json:4
출력: data_csv/<name>/arms/all_mp4list.csv · data_csv/<name>_negimp/families.json
"""
import argparse
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--part", action="append", required=True, help="이름:families.json:반복")
    a = ap.parse_args()
    paths, fams, parts = [], [], []
    for spec in a.part:
        tag, fj, rep = spec.split(":")
        fj = fj if os.path.isabs(fj) else os.path.join(REPO, fj)
        d = json.load(open(fj))
        L = [ln.strip() for ln in open(d["list"], encoding="utf-8") if ln.strip()]
        if sum(1 + len(f["A"]) for f in d["families"]) != len(L):
            raise ValueError(f"{fj}: 묶음 영상 합 {sum(1 + len(f['A']) for f in d['families'])} ≠ 목록 {len(L)}")
        for r in range(int(rep)):
            off = len(paths)
            paths += L
            for f in d["families"]:
                fams.append({**f, "A": [off + i for i in f["A"]], "B": off + f["B"], "src": tag})
        parts.append([tag, d["list"], int(rep)])
    d1, d2 = f"{REPO}/data_csv/{a.name}/arms", f"{REPO}/data_csv/{a.name}_negimp"
    os.makedirs(d1, exist_ok=True); os.makedirs(d2, exist_ok=True)
    open(f"{d1}/all_mp4list.csv", "w").writelines(p + "\n" for p in paths)
    s = {"name": a.name, "videos": len(paths), "families": len(fams), "parts": parts}
    json.dump(s, open(f"{d1}/all_summary.json", "w"), indent=1)
    json.dump({"list": os.path.abspath(f"{d1}/all_mp4list.csv"), "splice": 32, "families": fams, "summary": s}, open(f"{d2}/families.json", "w"))
    print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
