#!/usr/bin/env python3
"""build_ek100_resized.py 진행 상황 — 완료 파일 + 지금 도는 ffmpeg 가 읽은 양으로 % 와 남은 시간을 낸다.

`_build.log` 는 파일이 하나 끝날 때마다 한 줄씩 쌓인다. 큰 파일부터 처리하므로 **처음 수십 분은 비어 있는 게 정상**이다.
그동안은 이 스크립트로 본다.

    python z_research/scripts/data/ek100_resized_progress.py
    watch -n 30 /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/data/ek100_resized_progress.py
"""
import glob, json, os, re, subprocess, time
from pathlib import Path

D = Path("/data2/local_datasets/EPIC-KITCHENS_resized")
SRC = Path("/data/dataset/EPIC-KITCHENS")

sizes = {p.stem: p.stat().st_size for p in SRC.glob("P*/videos/*.MP4")}
total = sum(sizes.values())
metas = [json.load(open(f)) for f in glob.glob(str(D / "_meta" / "*.json"))]
ok = [m for m in metas if m.get("ok")]
bad = [m for m in metas if not m.get("ok")]
done_b = sum(sizes.get(m["video_id"], 0) for m in ok)

running = []  # (video_id, 읽은 바이트, 원본 크기)
for p in subprocess.run(["ps", "-C", "ffmpeg", "-o", "pid="], capture_output=True, text=True).stdout.split():
    try:
        cmd = open(f"/proc/{p}/cmdline", "rb").read().decode(errors="ignore")
        if "EPIC-KITCHENS_resized" not in cmd:
            continue
        vid = re.search(r"/(P\d+_\d+)\.MP4", cmd).group(1)
        rchar = int([l for l in open(f"/proc/{p}/io") if l.startswith("rchar")][0].split()[1])
        running.append((vid, min(rchar, sizes.get(vid, rchar)), sizes.get(vid, 0)))
    except Exception:
        pass
inprog_b = sum(r[1] for r in running)
processed = done_b + inprog_b

info = D / "_info.json"
t0 = time.mktime(time.strptime(json.load(open(info))["created"], "%Y-%m-%d %H:%M:%S")) if info.exists() else None
el = time.time() - t0 if t0 else None
rate = processed / el if el else 0
eta = (total - processed) / rate if rate else float("nan")

print(time.strftime("%H:%M:%S"), f"| 경과 {el/60:.0f}분" if el else "")
print(f"완료 {len(ok)}/{len(sizes)} 개   실패 {len(bad)}   진행 중 {len(running)} 개")
print(f"처리량 {processed/1e9:.0f} / {total/1e9:.0f} GB  ({100*processed/total:.1f}%)   평균 {rate/1e6:.0f} MB/s")
if rate:
    print(f"남은 시간 ≈ {eta/60:.0f}분  (끝 ≈ {time.strftime('%H:%M', time.localtime(time.time()+eta))})")
if bad:
    print("실패:", ", ".join(f"{m['video_id']} {m.get('problems')}" for m in bad[:5]))
if running:
    top = sorted(running, key=lambda r: -r[2])[:5]
    print("진행 중 (큰 것 5개): " + "  ".join(f"{v} {100*b/max(s,1):.0f}%" for v, b, s in top))
