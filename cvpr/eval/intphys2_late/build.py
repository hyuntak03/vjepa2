#!/usr/bin/env python3
"""intphys2_late_uni 데이터 만들기 (2026-10-22, 사용자 지시) — IntPhys 2 Main 506 쌍, 사람이 표시한 위반 시작 프레임 d 앞을 문맥으로.

라벨 = IntPhys2_labeling 시트 (labels 탭) 의 violation_start_frame (원본 60 fps, 0-based). 불가능 영상마다 한 줄.
  사용자 (10-22): "같은 시나리오면 pair 1, 2 가 사실상 violation start frame 도 같더라" → 쌍의 라벨이 비었거나 not_found 면 같은 장면 다른 쌍의 라벨을 빌린다.
  둘 다 있으면 자기 라벨. 둘 다 없으면 뺀다.
쌍 = 같은 장면의 (k_Possible, k_Impossible), k ∈ {1, 2} (analysis/intphys2/README §3 — 쌍 1 · 2 는 장애물 유무가 달라 가로질러 비교하지 않는다).
창 (IntPhys 1 uni 와 같은 규칙, cvpr/eval/intphys1_late/build.py --uniform-d):
  n = min(가능 영상 프레임 수, 불가능 영상 프레임 수)
  문맥 16 장 = round(linspace(0, d − 1, 16)),  미래 16 장 = round(linspace(d, n − 1, 16))   — 두 영상에 같은 번호
⚠️ IntPhys 2 는 가능 · 불가능 영상이 첫 프레임부터 픽셀이 다르다 (조명 · 카메라 · 배치가 따로 렌더됨, IntPhys2_labeling/README).
  그래서 matched 쌍이어도 문맥이 같지 않고 p 도 같지 않다 (IntPhys 1 · VOBench 와 다른 점). context_integrity.json 에 쌍마다 문맥 차이를 적는다.
산출
  프레임: <frames_out>/<video_id>/000000.png … 000031.png  (decord 로 원본 512² 그대로, 리사이즈는 채점기가 한다)
  index : data_csv/intphys2_late_uni/index.csv  (+ labels_snapshot.csv = 이번에 쓴 시트 사본, context_integrity.json)

  python cvpr/eval/intphys2_late/build.py --labels <시트 labels 탭 CSV>
  --d-offset -10 (2026-10-23, 사용자 "violation start 를 10 frame 씩 모두 빼서"): d = 라벨 − 10 (60 fps 기준 10 장 = 1/6 초)
    → --out data_csv/intphys2_late_uni_m10 --frames-out <NFS>/IntPhys2_late_uni_m10 (레지스트리 intphys2_late_uni_m10)
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MAP = Path("/data/hyuntak/project/2026/2027_cvpr/IntPhys2_labeling/labels.csv")       # file → SceneIndex · pair · 원본 경로 (impossible_src / possible_src)
SRC_ROOT = Path("/data2/local_datasets/world/Benchmark/IntPhys2/Main/Videos")          # vll5 로컬 사본 (MAP 경로는 NFS /data/dataset/world/IntPhys2/Main/Videos)
N_CTX, N_FUT = 16, 16


NFS_ROOT = Path("/data/dataset/world/IntPhys2/Main/Videos")                         # 어느 노드에서나 읽히는 원본 사본


def nfs(p):
    return str(NFS_ROOT / os.path.basename(p))


def local(p):
    return str(SRC_ROOT / os.path.basename(p))


def nframes(p):
    import decord
    return len(decord.VideoReader(p, num_threads=1))


def dump(job):
    import decord
    from PIL import Image
    path, frames, out = job
    tag = " ".join(map(str, frames))
    if (Path(out) / "frames.txt").exists() and (Path(out) / "frames.txt").read_text() == tag:   # 같은 프레임이면 건너뜀 (라벨이 바뀐 쌍만 다시 뽑는다)
        return out, None
    os.makedirs(out, exist_ok=True)
    arr = decord.VideoReader(path, num_threads=1).get_batch(frames).asnumpy()
    for i, a in enumerate(arr):
        Image.fromarray(a).save(f"{out}/{i:06d}.png")
    Path(out, "frames.txt").write_text(tag)
    return out, arr[:N_CTX].astype(np.int16)


def build_final(a, mp):
    """labels_final.csv 의 행 = 쌍 하나. impossible_file 로 대응표 (원본 경로) 를 찾고, 적힌 32 장을 그대로 쓴다."""
    by = {r["file"]: r for r in mp}
    rows, jobs = [], []
    for x in csv.DictReader(open(a.final)):
        r = by[x["impossible_file"]]
        assert (r["SceneIndex"], r["pair"]) == (x["scene"], x["pair"]), x["impossible_file"]
        pos, imp = local(r["possible_src"]), local(r["impossible_src"])
        fr = list(map(int, x["context_frames"].split())) + list(map(int, x["future_frames"].split()))
        n_pos, n_imp = nframes(pos), nframes(imp)
        assert len(fr) == N_CTX + N_FUT and max(fr) < n_imp, (x["impossible_file"], max(fr), n_imp)
        fr_pos = [min(f, n_pos - 1) for f in fr]          # 미래 번호는 불가능 영상 길이 기준 — 가능 영상이 짧은 36 쌍은 끝 1–2 장을 가능 영상 마지막 장으로 (clamp_pos 열)
        n = min(n_pos, n_imp)
        base = {"block_id": f"s{x['scene']}", "pair_id": x["pair"], "block_type": x["condition"], "condition": x["condition"], "game_name": x["game_name"],
                "Difficulty": r["Difficulty"], "Camera": r["Camera"], "occluder": r["occluder"], "d": int(x["violation_start"]),
                "d_sheet": x["sheet_violation_start"], "d_src": x["source"], "annotator": x["annotator"], "occluded": "0", "n": n, "clamp_pos": sum(f != g for f, g in zip(fr, fr_pos))}
        for plaus, path, ff in (("1", pos, fr_pos), ("0", imp, fr)):
            vid = f"s{x['scene']}_p{x['pair']}_{'pos' if plaus == '1' else 'imp'}"
            rows.append({"video_id": vid, "file": nfs(path) if a.index_only else path, "variant": f"{'pos' if plaus == '1' else 'imp'}_{x['pair']}", "plausible": plaus, **base,
                         "frames": " ".join(map(str, ff))})
            jobs.append((path, ff, f"{a.frames_out}/{vid}"))
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    shutil.copy(a.final, f"{a.out}/labels_final_snapshot.csv")
    if not a.index_only:
        with ProcessPoolExecutor(a.workers) as ex:
            list(ex.map(dump, jobs, chunksize=4))
    print(f"쌍 {len(rows) // 2} · 프레임 {a.frames_out} · index {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", help="시트 labels 탭 CSV (gviz export)")
    ap.add_argument("--index-only", action="store_true", help="--final 과 함께: PNG 를 만들지 않고 index 만 (file = NFS mp4 절대경로, 채점기가 frames 열의 번호로 mp4 를 직접 읽는다 — data.frames_column)")
    ap.add_argument("--final", help="labels_final.csv (2026-10-23 사용자 재라벨): 쌍마다 violation_start · context_frames · future_frames 가 적혀 있다 → 그 프레임 그대로")
    ap.add_argument("--frames-out", default="/data2/local_datasets/world/world_analysis/IntPhys2_late_uni")
    ap.add_argument("--out", default=str(REPO / "data_csv/intphys2_late_uni"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--d-offset", type=int, default=0, help="라벨 d 에 더한다 (2026-10-23 사용자: 위반 시작을 10 프레임씩 빼서 → -10). index 의 d_label 열에 원래 값")
    a = ap.parse_args()

    mp = list(csv.DictReader(open(MAP)))
    if a.final:
        return build_final(a, mp)
    lab = {r["file"]: r for r in csv.DictReader(open(a.labels))}
    assert set(lab) == {r["file"] for r in mp}, "시트와 대응표의 file 목록이 다르다"

    def own(r):
        x = lab[r["file"]]
        return int(x["violation_start_frame"]) if x["violation_start_frame"].strip() and not x["not_found"].strip() else None

    scene = collections.defaultdict(dict)
    for r in mp:
        scene[r["SceneIndex"]][r["pair"]] = r
    rows, skipped, jobs = [], [], []
    for sc, prs in sorted(scene.items()):
        for k, r in sorted(prs.items()):
            d, src = own(r), "own"
            other = prs.get("2" if k == "1" else "1")
            if d is None and other is not None and own(other) is not None:
                d, src = own(other), "other_pair"
            if d is None:
                skipped.append(r["file"]); continue
            d_label, d = d, d + a.d_offset
            pos, imp = local(r["possible_src"]), local(r["impossible_src"])
            n = min(nframes(pos), nframes(imp))
            fr = [int(round(x)) for x in np.linspace(0, d - 1, N_CTX)] + [int(round(x)) for x in np.linspace(d, n - 1, N_FUT)]
            x = lab[r["file"]]
            base = {"block_id": f"s{sc}", "pair_id": k, "block_type": r["condition"], "condition": r["condition"], "game_name": r["game_name"],
                    "Difficulty": r["Difficulty"], "Camera": r["Camera"], "occluder": r["occluder"], "d": d, "d_label": d_label, "d_src": src,
                    "d_other": "" if other is None or own(other) is None else own(other), "occluded": x["occluded"].strip() or "0",
                    "n": n, "frames": " ".join(map(str, fr))}
            for plaus, path in (("1", pos), ("0", imp)):
                vid = f"s{sc}_p{k}_{'pos' if plaus == '1' else 'imp'}"
                rows.append({"video_id": vid, "file": path, "variant": f"{'pos' if plaus == '1' else 'imp'}_{k}", "plausible": plaus, **base})
                jobs.append((path, fr, f"{a.frames_out}/{vid}"))

    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    shutil.copy(a.labels, f"{a.out}/labels_snapshot.csv")
    print(f"쌍 {len(rows) // 2} (자기 라벨 {sum(r['d_src'] == 'own' for r in rows) // 2} · 다른 쌍 라벨 {sum(r['d_src'] == 'other_pair' for r in rows) // 2}) · 뺀 쌍 {len(skipped)}")

    ctx = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for out, c in ex.map(dump, jobs, chunksize=4):
            if c is not None:
                ctx[os.path.basename(out)] = c
    prev = Path(a.out) / "context_integrity.json"
    integ = json.load(open(prev))["pairs"] if prev.exists() else {}
    for r in rows[::2]:
        p, i = f"s{r['block_id'][1:]}_p{r['pair_id']}_pos", f"s{r['block_id'][1:]}_p{r['pair_id']}_imp"
        if p in ctx and i in ctx:
            dd = np.abs(ctx[p] - ctx[i])
            integ[f"{r['block_id']}_p{r['pair_id']}"] = {"mean_abs": round(float(dd.mean()), 3), "frac_px_gt10": round(float((dd.max(-1) > 10).mean()), 4)}
    json.dump({"note": "문맥 16 장의 가능 − 불가능 RGB 차이 (0–255). IntPhys 2 는 영상이 따로 렌더되어 0 이 아니다", "skipped": skipped, "pairs": integ},
              open(f"{a.out}/context_integrity.json", "w"), indent=1, ensure_ascii=False)
    print("프레임:", a.frames_out, "· index:", a.out)


if __name__ == "__main__":
    main()
