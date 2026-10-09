#!/usr/bin/env python3
"""자연 영상 학습셋 -> `video_csv` 인덱스 (predictor 학습용).

    python z_training/data/build_video_index.py --list
    python z_training/data/build_video_index.py ssv2 --write
    python z_training/data/build_video_index.py k400 --write --probe --min-frames 120

형식은 `src/datasets/video_dataset.VideoDataset` 이 요구하는 **"<mp4 절대경로> <라벨>"**
공백 구분이다. **라벨은 전부 0** — frozen-encoder JEPA 손실은 라벨을 안 쓴다
(`app/vjepa_frozen/train.py`). 행동 라벨은 필요 없다.

⚠️ val 분할은 **공식 분할이 아니라** seed 0 무작위다. 학습 중 loss 감시용으로만 읽을 것.

⚠️ **`--probe` 의 `--min-frames` 는 config 의 `n_frames × fstp` 와 맞춰야 한다.**
   `fstp = 원본fps // data.fps` 라 데이터셋마다 다르다 (아래 표). 안 맞추면 학습 중
   `filter_short_videos` 가 조용히 버리고 epoch 크기가 줄어든다.

실측 (2026-09-22)

| 세트 | clip | fps | 길이 median | 해상도 |
|---|---:|---|---:|---|
| `ssv2` | 220,847 | 12 | 45 (max **76**) | 높이 320 고정 |
| `k400` | ~138,000 | 30 주류 (25·24 섞임) | 112 (p95 300) | **혼재** (720×1280 / 360×480 …) |

⚠️ **SSv2 는 최대 76장(6.3초)이라 긴 창을 못 준다.** `fps 6 × 24장` = raw 48 이 현실적인 상한이다.
⚠️ **K400 은 해상도·화면비가 섞여 있다.** `ClipTransform` 은 (r,r) 로 바로 리사이즈하므로
   **`aug.random_resized_crop` 을 반드시 켠다** (안 켜면 16:9 가 1:1 로 눌려 가로 속도가 1.78배 압축된다).

2026-09-27 vll5 이식 (Ariel `z_ariel/vjepa2_train_code_20260927/z_training/data/build_video_index.py`)
  * `paths.py` (`DATA_CSV`, `TRAIN_DATA_ROOT`) import 를 뺐다 — 우리 레포에는 paths.env 가 없다.
    대신 세트마다 vll5 기본 `root` 를 SETS 에 두고 `--root` / `--out` 으로 덮는다.
    출력 기본값 = `<레포>/data_csv/<set>` (gitignore — 이 스크립트로 다시 만든다).
  * ssv2: `/data2/local_datasets/something-something/something-something-v2-mp4` (vll5 로컬 HDD, 220,850 편).
    vll5 에서 헤더 probe 는 16 스레드로 약 0.6 ms/편 (400 편 표본) — 캐시 예열 없이 수 분이면 끝난다.
  * k400: **vll5 에 쓸 수 있는 사본이 없다** (`/data2/local_datasets/Kinetics-400` 비어 있음). `--root` 를 반드시 준다.
    Ariel 의 `k400_320_min96` 은 장면 단위 서브클립을 320p 로 재인코딩한 사본이라 우리 K400 (10 초 원본) 과 분포가 다르다.

    python z_training/data/build_video_index.py ssv2 --probe --min-frames 48 --workers 16 --write
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import random
import re
import sys
import time

# 2026-09-27: Ariel 의 `from paths import DATA_CSV, TRAIN_DATA_ROOT` 대신 상수 (우리 레포에는 paths.env 가 없다)
ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_CSV = str(ROOT / "data_csv")

# 세트별 차이는 **데이터로만** 둔다 (코드 분기 금지).
SETS = {
    "ssv2": {
        "root": "/data2/local_datasets/something-something",     # vll5 로컬 (2026-09-27)
        "glob": "something-something-v2-mp4/*.mp4",
        "note": "Something-Something v2. 12 fps, 높이 320, median 45장(max 76).",
    },
    "k400": {
        "root": None,             # vll5 에 없다 — --root 필수 (2026-09-27)
        "glob": "K400/videos/*/*/*.mp4",
        "note": "Kinetics-400. fps 30 주류, 해상도 혼재, median 112장(p95 300).",
        # ⚠️ 이 K400 사본은 장면 단위로 잘린 서브클립이고 파일명이 `..._clip_<i>_<start>_<dur>.mp4` 다.
        #    **컨테이너 헤더 길이는 원본 것이라 신뢰할 수 없고, 파일명의 <dur> 가 실제 길이(초)** 다.
        #    2026-09-22: `_clip_2_7.107_2.870.mp4`(2.87 초) 가 헤더 ≥96 장으로 통과했다가 decord 가
        #    EOF 너머에서 무한정 spin -> rank 로더 정지 -> DDP 전체 정지. 그래서 초 단위로 거른다.
        "dur_from_name": r"_clip_\d+_[\d.]+_([\d.]+)\.mp4$",
        "fps_floor": 24,          # 24/25/30 fps 가 섞여 있다. 가장 낮은 fps 기준으로 필요한 초를 계산
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", nargs="?", choices=sorted(SETS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--root", default=None, help="기본 SETS[<set>]['root'] (vll5 경로)")
    ap.add_argument("--out", default=None, help=f"기본 {DATA_CSV}/<set>")
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", action="store_true", help="decord 로 길이를 재서 lengths.tsv 를 만든다")
    ap.add_argument("--workers", type=int, default=32, help="--probe / --deep-probe 병렬도")
    ap.add_argument("--deep-probe", action="store_true",
                    help="파일마다 **실제로 프레임을 디코드**해(첫/중간/끝) 멈추거나 실패하는 파일을 decode_bad.txt 에 적는다. "
                         "이후 --min-frames 쓰기는 그 파일을 뺀다")
    ap.add_argument("--deep-timeout", type=float, default=30.0, help="--deep-probe 파일당 타임아웃(초)")
    ap.add_argument("--min-frames", default="0",
                    help="쉼표로 여러 개. 각각 train_min<N>.csv / val_min<N>.csv 를 쓴다. "
                         "lengths.tsv 가 있으면 probe 없이 즉시 만든다")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    if a.list or not a.set:
        print("세트:")
        for k, v in sorted(SETS.items()):
            print(f"  {k:<6} {v['note']}")
        return

    spec = SETS[a.set]
    root = a.root or spec.get("root")
    if not root:
        sys.exit(f"ERROR: '{a.set}' 는 기본 root 가 없다 (이 기계에 사본이 없다) — --root 를 준다")
    out = a.out or os.path.join(DATA_CSV, a.set)
    files = sorted(glob.glob(os.path.join(root, spec["glob"])))
    print(f"[{a.set}] mp4 {len(files):,}  <- {os.path.join(root, spec['glob'])}")
    if not files:
        sys.exit("ERROR: mp4 가 하나도 없다")

    # ── 길이 표 (lengths.tsv) ────────────────────────────────────────────────────
    # ⚠️ **런타임 거부-재추첨은 치명적이다.** `filter_short_videos` 는 짧은 영상을 만나면
    #    무작위 인덱스로 다시 뽑는데, 통과율이 낮으면 (k400 @fps6 은 14.2%) 성공 1 건당
    #    영상을 7 번 열게 되어 rank 하나가 배치를 못 채우고 **DDP 전체가 멈춘다**
    #    (2026-09-22 실측: skipping 1,368 회 / step 로그 0 회, GPU 7 장이 100% 로 공회전).
    #    그래서 길이를 **미리** 걸러 csv 를 만든다. 임계값은 config 의 `n_frames x fstp` 다.
    lp = os.path.join(out, "lengths.tsv")
    lengths = {}
    if a.probe:
        from concurrent.futures import ThreadPoolExecutor
        import decord

        def _len(f):
            try:
                return f, len(decord.VideoReader(f, num_threads=1))
            except Exception:
                return f, -1

        print(f"probe {len(files):,} 편 (workers {a.workers}) ...", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for i, (f, n) in enumerate(ex.map(_len, files, chunksize=64)):
                lengths[f] = n
                if (i + 1) % 10000 == 0:            # 2026-09-27: 25000 -> 10000, 경과 시간 (백그라운드 빌드 감시용)
                    print(f"  {i+1:,}/{len(files):,}  {time.time()-t0:.0f}s", flush=True)
        os.makedirs(out, exist_ok=True)
        with open(lp, "w", encoding="utf-8") as f:
            for k, v in lengths.items():
                f.write(f"{k}\t{v}\n")
        print("썼다:", lp)
    elif os.path.isfile(lp):
        for ln in open(lp, encoding="utf-8"):
            k, v = ln.rstrip("\n").rsplit("\t", 1)
            lengths[k] = int(v)
        print(f"lengths.tsv 재사용: {len(lengths):,} 편")

    # ── 실제 디코드 검증 (decode_bad.txt) ────────────────────────────────────────
    # ⚠️ `len(VideoReader)` 는 **컨테이너 헤더**를 읽는다. K400 의 잘린 서브클립
    #    (`_clip_N_<start>_<2~3초>.mp4`) 은 헤더가 240 장이라 해도 실제 스트림은 ~75 장이라,
    #    그 너머를 읽으면 decord 가 **무한정 spin** 한다 (2026-09-22: 893 표본 중 8 개, 전부 그 형태).
    #    DataLoader 는 배치를 순서대로만 내놓으므로 워커 하나가 멈추면 그 rank 가 통째로 죽는다.
    #    그래서 첫·중간·**끝** 프레임을 실제로 읽어 본다. 멈추면 워커를 죽이고 그 파일을 bad 로 적는다.
    bp = os.path.join(out, "decode_bad.txt")
    if a.deep_probe:
        import multiprocessing as mp, queue as _q
        def _dec_worker(inq, outq):
            import decord
            while True:
                f = inq.get()
                if f is None:
                    return
                outq.put(("start", f))
                try:
                    vr = decord.VideoReader(f, num_threads=1); L = len(vr)
                    vr.get_batch([0, L // 2, L - 1]).asnumpy()
                    outq.put(("ok", f))
                except Exception as e:
                    outq.put(("bad", f))
        todo = [f for f in files if lengths.get(f, 1) > 0]
        inq, outq = mp.Queue(), mp.Queue()
        for f in todo:
            inq.put(f)
        procs, cur, bad, done, t0 = {}, {}, [], 0, time.time()
        def _spawn():
            pr = mp.Process(target=_dec_worker, args=(inq, outq), daemon=True); pr.start(); procs[pr.pid] = pr
        for _ in range(a.workers):
            _spawn()
        print(f"deep-probe {len(todo):,} 편 (workers {a.workers}, timeout {a.deep_timeout}s) ...", flush=True)
        while done < len(todo):
            try:
                kind, f = outq.get(timeout=1.0)
                if kind == "start":
                    cur[f] = time.time()
                else:
                    cur.pop(f, None); done += 1
                    if kind == "bad":
                        bad.append(f)
                    if done % 20000 == 0:
                        print(f"  {done:,}/{len(todo):,}  bad {len(bad)}  {time.time()-t0:.0f}s", flush=True)
            except _q.Empty:
                pass
            # 타임아웃: 그 파일을 잡고 있는 워커를 찾아 죽이고 새 워커를 띄운다
            for f, ts in list(cur.items()):
                if time.time() - ts > a.deep_timeout:
                    for pid, pr in list(procs.items()):
                        # 어느 워커인지 모르므로 살아 있는 워커 중 가장 오래 응답 없는 것을 죽인다:
                        # start 신호 뒤 timeout 을 넘긴 파일은 하나뿐이므로 그 파일의 워커만 죽으면 된다
                        pass
                    # 워커-파일 매핑이 없으니 전부 확인: 각 워커에 'ping' 은 불가 -> 가장 단순한 방법: 전원 재시작
                    for pid, pr in list(procs.items()):
                        pr.terminate(); pr.join(); procs.pop(pid)
                    bad.append(f); cur.pop(f, None); done += 1
                    # 진행 중이던 다른 파일들은 큐에 다시 넣는다
                    for g in list(cur):
                        inq.put(g); cur.pop(g)
                    for _ in range(a.workers):
                        _spawn()
                    print(f"  !! HANG {f[-80:]}  -> bad ({len(bad)})", flush=True)
                    break
        for _ in range(a.workers):
            inq.put(None)
        os.makedirs(out, exist_ok=True)
        with open(bp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(bad) + ("\n" if bad else ""))
        print(f"deep-probe 완료: bad {len(bad):,} / {len(todo):,} ({time.time()-t0:.0f}s) -> {bp}", flush=True)
    badset = set()
    if os.path.isfile(bp):
        badset = {ln.rstrip("\n") for ln in open(bp, encoding="utf-8") if ln.strip()}
        if badset:
            print(f"decode_bad.txt: {len(badset):,} 편을 뺀다")

    mins = sorted({int(x) for x in str(a.min_frames).split(",")})
    if mins != [0] and not lengths:
        sys.exit("ERROR: --min-frames 를 쓰려면 lengths.tsv 가 필요하다 (--probe 를 먼저 돌린다)")

    # 파일명 길이(초) 필터 — 헤더가 거짓말하는 세트용 (SETS[...]["dur_from_name"])
    dur_re = re.compile(spec["dur_from_name"]) if spec.get("dur_from_name") else None
    fps_floor = float(spec.get("fps_floor", 24))
    def _dur(f):
        m = dur_re.search(os.path.basename(f)) if dur_re else None
        return float(m.group(1)) if m else None

    for mn in mins:
        cand = files if mn <= 0 else [f for f in files if lengths.get(f, -1) >= mn]
        if dur_re and mn > 0:
            need_sec = mn / fps_floor * 1.05
            n0 = len(cand)
            cand = [f for f in cand if (_dur(f) is None) or (_dur(f) >= need_sec)]
            print(f"  파일명 길이 >= {need_sec:.2f}s 필터: {n0:,} -> {len(cand):,}  (헤더는 통과했지만 실제로 짧은 서브클립 {n0-len(cand):,} 제거)")
        keep = [f for f in cand if f not in badset]
        sfx = "" if mn <= 0 else f"_min{mn}"
        if mn > 0:
            print(f"길이 >= {mn:>4}: {len(keep):>8,} / {len(files):,} ({100*len(keep)/len(files):.1f} %)")
        if not keep:
            print(f"  !! min_frames={mn} 에 남는 영상이 0 — csv 를 쓰지 않는다")
            continue
        rng = random.Random(a.seed)
        idx = list(range(len(keep)))
        rng.shuffle(idx)
        n_val = int(len(keep) * a.val_frac)
        val = sorted(keep[i] for i in idx[:n_val])
        train = sorted(keep[i] for i in idx[n_val:])
        if not a.write:
            print(f"  train {len(train):,} / val {len(val):,}   (--write 를 줘야 쓴다)")
            continue
        os.makedirs(out, exist_ok=True)
        for name, rows in ((f"train{sfx}.csv", train), (f"val{sfx}.csv", val)):
            pth = os.path.join(out, name)
            with open(pth, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(f"{r} 0\n")
            print("  썼다:", pth, f"({len(rows):,} 줄)")


if __name__ == "__main__":
    main()
