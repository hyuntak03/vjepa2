#!/usr/bin/env python3
"""IntPhys2 라벨 확인용 영상 — 라벨 시트에 기록된 불가능 영상 중 조건 · 게임을 골라, 가능 | 불가능 나란히 + 프레임 번호 + 위반 시작 표시 (2026-10-22, 사용자 지시).

  /venv/vjepa/bin/python z_research/scripts/figures/intphys2_label_videos.py <라벨 JSON> <condition> <game_name> <출력 폴더>

영상마다 <출력>/scene<S>_pair<P>_<condition>_<game>.mp4 (60 fps, 1024×512 + 위 띠).
띠: 프레임 번호 · 시트 위반 시작 d_sheet · 채점에 쓰는 d (index_mp4.csv — −10 · 사용자 지정 반영). 프레임 ≥ d_sheet 이면 번호가 빨강."""
import csv, json, os, re, subprocess, sys
import numpy as np, decord, cv2
LAB, COND, GAME, OUT = sys.argv[1:5]
ROOT = "/workspace/dataset/benchmarks/IntPhys2/Main"
os.makedirs(OUT, exist_ok=True)
v = json.load(open(LAB))["values"]; h = v[0]
meta = {(m["condition"], m["SceneIndex"], m["type"]): m for m in csv.DictReader(open(f"{ROOT}/metadata.csv"))}
used = {r["block_id"] + "|" + r["variant"]: int(r["d"]) for r in csv.DictReader(open("data_csv/intphys2_violation_uni/index_mp4.csv"))}
for r in v[1:]:
    r = dict(zip(h, r + [""] * (len(h) - len(r))))
    if r["condition"] != COND or r["game_name"] != GAME or not r["violation_start_frame"].strip():
        continue
    sc, p = re.search(r"scene(\d+(?:_\d+)?)_pair(\d)_", r["file"]).groups()
    ds = int(r["violation_start_frame"]); du = used.get(f"{COND}|{sc}|{p}|pos")
    vp = decord.VideoReader(os.path.join(ROOT, meta[(COND, sc, f"{p}_Possible")]["file_name"]))
    vi = decord.VideoReader(os.path.join(ROOT, meta[(COND, sc, f"{p}_Impossible")]["file_name"]))
    N = min(len(vp), len(vi)); W = 512
    fn = os.path.join(OUT, f"scene{sc}_pair{p}_{COND}_{GAME}.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * W}x{W + 40}", "-r", "60", "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", fn], stdin=subprocess.PIPE)
    for s in range(0, N, 64):
        A = vp.get_batch(list(range(s, min(N, s + 64)))).asnumpy(); B = vi.get_batch(list(range(s, min(N, s + 64)))).asnumpy()
        for k in range(len(A)):
            f = s + k
            top = np.zeros((40, 2 * W, 3), np.uint8)
            col = (255, 60, 60) if f >= ds else (255, 255, 255)
            cv2.putText(top, f"frame {f}/{N - 1}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            cv2.putText(top, f"sheet d={ds} ({r['annotator']})  used d={du}", (300, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(top, "possible | impossible", (760, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            fr = np.concatenate([A[k], B[k]], 1)
            ff.stdin.write(np.concatenate([top, fr], 0).tobytes())
    ff.stdin.close(); ff.wait()
    print(fn, f"d_sheet {ds} · used {du} · N {N}", flush=True)
