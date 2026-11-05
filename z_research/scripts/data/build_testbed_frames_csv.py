#!/usr/bin/env python3
"""TestBed 보드 (IntPhys1_surrogate · VOBench_v3) 의 label.json → data_csv/<보드>/frames.csv (2026-11-05).

채점기 token_subset (evals/world_model_analysis/eval.py `_token_subset_masks`) 이 읽는 표:
  video_id, frame, x_px, y_px, present
  frame  = 채점 표본 번호 (raw = frames_start + frame × frames_stride, 두 보드 모두 0 + 3 × frame, 32 장)
  x/y_px = 채점 해상도 (256) 픽셀. label 의 u, v 는 원본 512 px 화면 좌표 → × 256 / 512
  present = 투영 중심이 화면 안 (0 ≤ u, v < 512) 이면 1. **가림막 뒤 (hidden) 여도 1** — "물체가 있어야 할 자리" (투영 중심) 정의.
block 의 네 변이 모두 trajectory A (물체가 있는 판) 를 쓴다. 채점기가 block 단위 합집합을 쓰므로 결과는 같다.

  /venv/vjepa/bin/python z_research/scripts/data/build_testbed_frames_csv.py intphys1_surrogate vobench_v3
"""
import csv, json, os, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
ROOTS = {"intphys1_surrogate": "/workspace/dataset/benchmarks/IntPhys1_surrogate", "vobench_v3": "/workspace/dataset/benchmarks/VOBench_v3"}
N, START, STRIDE, SRC, RES = 32, 0, 3, 512, 256


def main():
    for name in sys.argv[1:]:
        idx = f"{REPO}/data_csv/{name}/index.csv"
        rows = list(csv.DictReader(open(idx)))
        out, n_in, n_all, miss = [], 0, 0, 0
        cache = {}
        for r in rows:
            rel = r.get("file") or r.get("file_name")
            d = os.path.dirname(os.path.join(ROOTS[name], rel))
            if d not in cache:
                lab = json.load(open(os.path.join(d, "label.json")))["labels"]
                cache[d] = {int(e["f"]): e for e in lab["A"]}
            tr = cache[d]
            for i in range(N):
                e = tr.get(START + STRIDE * i)
                if e is None:
                    miss += 1; continue
                u, v = float(e["u"]), float(e["v"])
                pres = int(0 <= u < SRC and 0 <= v < SRC)
                n_in += pres; n_all += 1
                out.append((r["video_id"], i, round(u * RES / SRC, 2), round(v * RES / SRC, 2), pres))
        p = f"{REPO}/data_csv/{name}/frames.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["video_id", "frame", "x_px", "y_px", "present"]); w.writerows(out)
        print(f"{name}: {len(rows)} videos → {p} ({len(out)} rows, in-frame {100 * n_in / max(1, n_all):.1f} %, missing {miss})")


if __name__ == "__main__":
    main()
