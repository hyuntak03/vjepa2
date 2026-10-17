#!/usr/bin/env python3
"""Kinetics-710 train 을 받아 **짧은 변 320 · keyint 24** 로 바로 재인코딩한다 (2026-10-17, 사용자 결정 — vast 브랜치 K710_320 과 같은 사본).

    nohup python z_training/data/download_k710.py --out /data3/local_datasets/world/training_datasets/K710_320 \\
        --dl 8 --enc 40 > z_training/logs/download_k710.log 2>&1 &
    python z_training/data/download_k710.py --out ... --only-tar <url> --limit 50     # 스모크 (tar 하나, 영상 50 개)

K710 = K400 ∪ K600 ∪ K700-2020 **train** (UniFormerV2 정의). 같은 클립 (파일 이름 <ytid>_<start>_<end>) 은 한 번만,
세 세트의 **val 에 있는 YouTube id 는 뺀다** (평가 누수 방지). 원본은 CVDF 공개 S3 tar (https://s3.amazonaws.com/kinetics/).
원본 tar 는 디스크에 쌓지 않는다: tar 를 받아 풀고 → 영상마다 ffmpeg 재인코딩 → 원본 삭제. 끝난 tar 는 done_tars.txt 에 남겨 이어 받는다.
출력: <out>/videos/<ytid>_<start>_<end>.mp4 · <out>/manifest.csv (file,label,set) · <out>/failed.txt
재인코딩은 z_research/scripts/data/resize_videos.sh 와 같은 설정 (libx264 veryfast crf 23, -g 24, 오디오 없음, faststart).
"""
from __future__ import annotations

import argparse, csv, io, os, queue, shutil, subprocess, sys, tarfile, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://s3.amazonaws.com/kinetics"
SETS = [  # (이름, train tar 목록, val 주석) — 큰 세트부터 (중복 클립은 먼저 받은 쪽이 남는다)
    ("k700", f"{BASE}/700_2020/train/k700_2020_train_path.txt", f"{BASE}/700_2020/annotations/val.csv"),
    ("k600", f"{BASE}/600/train/k600_train_path.txt", f"{BASE}/600/annotations/val.csv"),
    ("k400", f"{BASE}/400/train/k400_train_path.txt", f"{BASE}/400/annotations/val.csv"),
]
S = 320


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode()


def encode(src: str, dst: str) -> bool:
    tmp = dst + ".tmp.mp4"
    cmd = ["nice", "-n", "15", "ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-threads", "1", "-i", src,
           "-vf", f"scale=w='if(gt(iw,ih),-2,{S})':h='if(gt(iw,ih),{S},-2)'",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-g", "24", "-pix_fmt", "yuv420p", "-an",
           "-movflags", "+faststart", tmp]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode == 0 and os.path.getsize(tmp) > 1000:
        os.replace(tmp, dst)
        return True
    if os.path.exists(tmp):
        os.remove(tmp)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None, help="tar 를 푸는 곳 (기본 <out>/_tmp)")
    ap.add_argument("--dl", type=int, default=8, help="동시에 받는 tar 수")
    ap.add_argument("--enc", type=int, default=40, help="동시 ffmpeg 수 (nice 15, 스레드 1)")
    ap.add_argument("--max-inflight", type=int, default=24, help="풀어 둔 채 재인코딩을 기다리는 tar 수 상한 (디스크 보호)")
    ap.add_argument("--only-tar", default=None, help="이 tar URL 하나만 (스모크)")
    ap.add_argument("--limit", type=int, default=0, help="tar 당 영상 수 상한 (스모크)")
    a = ap.parse_args()
    out = os.path.abspath(a.out); vid_dir = os.path.join(out, "videos"); tmp_root = a.tmp or os.path.join(out, "_tmp")
    os.makedirs(vid_dir, exist_ok=True); os.makedirs(tmp_root, exist_ok=True)
    done_path, man_path, fail_path = (os.path.join(out, n) for n in ("done_tars.txt", "manifest.csv", "failed.txt"))
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    lock = threading.Lock()

    # ---- val 제외 목록 (세 세트의 val YouTube id)
    val_ids = set()
    for name, _, val_url in SETS:
        rows = list(csv.DictReader(io.StringIO(fetch_text(val_url))))
        val_ids.update(r["youtube_id"] for r in rows)
        print(f"[val] {name}: {len(rows):,} 줄 → 제외 id 누적 {len(val_ids):,}", flush=True)

    # ---- 받을 tar 목록
    jobs = []
    if a.only_tar:
        jobs = [("smoke", a.only_tar)]
    else:
        for name, path_url, _ in SETS:
            urls = [u.strip() for u in fetch_text(path_url).split() if u.strip()]
            jobs += [(name, u) for u in urls if u not in done]
            print(f"[tar] {name}: {len(urls)} 개 (남은 것 {sum(1 for u in urls if u not in done)})", flush=True)
    print(f"[plan] tar {len(jobs)} 개, 동시 받기 {a.dl}, 동시 ffmpeg {a.enc}", flush=True)

    enc_pool = ThreadPoolExecutor(max_workers=a.enc)
    inflight = threading.Semaphore(a.max_inflight)
    stats = {"tar": 0, "ok": 0, "dup": 0, "val": 0, "fail": 0, "bytes": 0}
    t0 = time.time()

    def process_tar(name: str, url: str):
        inflight.acquire()
        work = os.path.join(tmp_root, os.path.basename(url).replace(".tar.gz", ""))
        try:
            shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
            tar_path = work + ".tar.gz"
            for attempt in range(5):
                r = subprocess.run(["curl", "-sS", "--retry", "5", "--retry-delay", "10", "-o", tar_path, url])
                if r.returncode == 0:
                    break
                time.sleep(30)
            else:
                print(f"[fail-dl] {url}", flush=True); return
            with lock:
                stats["bytes"] += os.path.getsize(tar_path)
            with tarfile.open(tar_path) as tf:
                tf.extractall(work)
            os.remove(tar_path)
            files = []
            for root, _, fs in os.walk(work):
                for f in fs:
                    if f.endswith(".mp4"):
                        files.append((os.path.basename(root), os.path.join(root, f)))
            if a.limit:
                files = files[: a.limit]
            futs = []
            for label, src in files:
                key = os.path.basename(src)[:-4]
                ytid = key[:11]
                dst = os.path.join(vid_dir, key + ".mp4")
                if ytid in val_ids:
                    with lock: stats["val"] += 1
                    os.remove(src); continue
                if os.path.exists(dst):
                    with lock: stats["dup"] += 1
                    os.remove(src); continue
                futs.append((label, src, dst, enc_pool.submit(encode, src, dst)))
            for label, src, dst, fu in futs:
                ok = fu.result()
                with lock:
                    if ok:
                        stats["ok"] += 1
                        with open(man_path, "a") as fm:
                            fm.write(f"{os.path.basename(dst)},{label},{name}\n")
                    else:
                        stats["fail"] += 1
                        with open(fail_path, "a") as ff:
                            ff.write(f"{url}\t{os.path.basename(src)}\n")
                if os.path.exists(src):
                    os.remove(src)
            if not a.only_tar:
                with lock, open(done_path, "a") as fd:
                    fd.write(url + "\n")
            with lock:
                stats["tar"] += 1
                el = time.time() - t0
                print(f"[{time.strftime('%m-%d %H:%M')}] tar {stats['tar']}/{len(jobs)} ({name}) | 영상 ok {stats['ok']:,} 중복 {stats['dup']:,} "
                      f"val제외 {stats['val']:,} 실패 {stats['fail']:,} | 받은 {stats['bytes']/1e9:.0f} GB "
                      f"({stats['bytes']/1e6/max(el,1):.0f} MB/s) | 경과 {el/3600:.1f} h", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
            inflight.release()

    with ThreadPoolExecutor(max_workers=a.dl) as dl_pool:
        list(dl_pool.map(lambda j: process_tar(*j), jobs))
    enc_pool.shutdown(wait=True)
    print(f"[done] {stats}", flush=True)


if __name__ == "__main__":
    main()
