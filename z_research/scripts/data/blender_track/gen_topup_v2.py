#!/usr/bin/env python3
"""BlenderTrack v2 칸 채우기 — 설계에 실패한 clip 만큼 같은 칸 · 같은 방해 물체 수의 새 번호를 설계해 채운다 (2026-11-01).

왜: 다른 물체 가림 × 정지 · 감속 경사 target × 낮은 · 비스듬 카메라 같은 칸은 기하로 어려워 (화면 크기 비 ≤ 1.8 · 점프 금지)
    설계 실패가 칸 안에서 최대 88 % 였다 → 그 칸만 얇아져 축 조합 분석이 안 된다.
새 번호: B + 972 j + (원래 번호 mod 972), B = 97,200 — 972 = 칸 324 × 방해 물체 수 3 이라 칸 · 방해 물체 수가 그대로다 (cell_of 와 같은 규칙).
산출: <out>/design/<id>.json (성공) · <out>/design/extra_<split>.txt (그릴 새 번호, 한 줄에 하나) · 모자란 칸 보고.
  $P gen_topup_v2.py --config configs/v2.yaml --splits train test --procs 70 --max_ratio 40
"""
import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_design_v2 as G  # noqa: E402

B, P972 = 97200, 972
CFG = {}


def _init(path):
    for sp in ("train", "test"):
        CFG[sp] = G.load_cfg(path, sp)


def one(job):
    split, i = job
    cfg = CFG[split]
    out, sd = cfg["out"], cfg["splits"][split]["seed"]
    cid = f"{split}{sd}_{i:06d}"
    if os.path.exists(f"{out}/design/{cid}.json"):
        return split, i, True
    try:
        spec = G.sample_clip(cfg, split, i, max_tries=1500, streams=int(cfg["checks"].get("design_streams", 10)))
    except RuntimeError:
        return split, i, False
    tmp = f"{out}/design/{cid}.{os.getpid()}.tmp"
    json.dump(spec, open(tmp, "w"), default=lambda x: x.item() if hasattr(x, "item") else str(x))
    os.replace(tmp, f"{out}/design/{cid}.json")
    return split, i, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--max_ratio", type=int, default=40, help="모자란 clip 하나에 새 번호를 최대 몇 개까지 시도")
    a = ap.parse_args()
    cfg = G.load_cfg(a.config, "train")
    out = cfg["out"]
    t0 = time.time()
    rep = {}
    for sp in a.splits:
        sd = cfg["splits"][sp]["seed"]
        need = collections.Counter()
        for f in os.listdir(f"{out}/design"):
            if f.startswith(f"{sp}{sd}_") and f.endswith(".fail"):
                i = int(f[len(f"{sp}{sd}_"):-5])
                if i < B:
                    need[i % P972] += 1
        got = collections.Counter(); tried = collections.Counter(); extra = []
        j = 0
        with ProcessPoolExecutor(a.procs, initializer=_init, initargs=(a.config,)) as ex:
            while True:
                open_g = [r for r in need if got[r] < need[r] and tried[r] < a.max_ratio * need[r]]
                if not open_g:
                    break
                # 그룹마다 모자란 만큼 (최소 1) 새 번호를 한 번에
                jobs = []
                for r in open_g:
                    k = max(1, min(need[r] - got[r], 8)) * 2
                    for _ in range(k):
                        jobs.append((sp, B + P972 * (j + tried[r]) + r)); tried[r] += 1
                for s_, i, ok in ex.map(one, jobs, chunksize=1):
                    r = i % P972
                    if ok and got[r] < need[r]:
                        got[r] += 1; extra.append(i)
                print(f"{sp}: 채움 {sum(got.values())}/{sum(need.values())} · 시도 {sum(tried.values())} | {time.time() - t0:.0f}s", flush=True)
        with open(f"{out}/design/extra_{sp}.txt", "w") as f:
            f.write("\n".join(str(i) for i in sorted(extra)) + "\n")
        short = {}
        for r in need:
            if got[r] < need[r]:
                c, _ = G.cell_of(cfg, r)
                short["|".join([c["timing"], c["form"], c["motion"], c["camera"], f"n{r // 324}"])] = need[r] - got[r]
        rep[sp] = {"need": sum(need.values()), "filled": sum(got.values()), "tried": sum(tried.values()), "short": short}
        print(json.dumps(rep[sp], ensure_ascii=False), flush=True)
    json.dump(rep, open(f"{out}/design/topup_report.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
