#!/usr/bin/env python3
"""BlenderTrack v1 렌더 전 검사 — 설계만으로 (렌더 없이) 통과 기준을 잰다. 하나라도 어기면 종료 코드 1 (pipeline.sh 가 멈춘다).

재는 것 (설정의 checks: 블록이 기준)
  운동 종류 비율 (거절로 사라진 종류가 없나) · 경로 위 가림막 수 · target 이 한 번이라도 다 가려지는 clip 비율
  late 창: 보폭마다 (렌더 프레임 단위) 32 샘플 창 (문맥 16 + 미래 16) 중 문맥 끝 target 이 가려지고 앞에서 보였고 미래에 다시 보이는 창
           → late 창을 가진 clip 비율 · 서로 다른 late 미래 경로 수 (미래 8 블록 중심 칸, 16×16)
  외우기 대 계산 (참고, 기준 아님): late 창에서 미래 블록 4 의 target 중심을
           (a) 문맥의 마지막으로 보인 두 샘플로 등속 외삽 · (b) 학습 절반에서 문맥 경로가 가장 비슷한 창의 미래를 그대로 (최근접 이웃)
           으로 맞힌 오차 (px). (b) 가 (a) 보다 훨씬 작으면 "꺼내 쓰기" 로 풀리는 데이터다.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P gen_check.py --config configs/v1.yaml --out <json>
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_design as G  # noqa: E402

CTX = FUT = 16


def windows(ctr, hid, s, T, size):
    span = (CTX + FUT - 1) * s + 1
    out = []
    for f0 in range(0, T - span + 1, max(1, s)):
        idx = f0 + s * np.arange(CTX + FUT)
        h, c = hid[idx], ctr[idx]
        late = bool(h[CTX - 1] >= 0.9 and (h[:CTX - 1] < 0.3).any() and (h[CTX:] < 0.3).any())
        if not late:
            out.append((False, None, None)); continue
        blk = c[CTX:].reshape(FUT // 2, 2, 2).mean(1)
        path = tuple(map(tuple, np.clip(np.floor(blk / (size / 16)), -1, 16).astype(int)))
        vis = np.nonzero(h[:CTX] < 0.3)[0]
        feat = None
        if len(vis) >= 2:
            a, b = vis[-2], vis[-1]
            lastv = c[b]; vel = (c[b] - c[a]) / (b - a)
            tgt_i = CTX + 8                                            # 미래 블록 4 (샘플 8)
            if h[tgt_i] < 0.9 or True:
                extrap = lastv + vel * (tgt_i - b)
                hist = c[vis[-min(len(vis), 6):]] - lastv              # 마지막으로 보인 자리 기준 문맥 경로 (최대 6 샘플)
                hist = np.pad(hist, ((6 - len(hist), 0), (0, 0)), mode="edge").ravel()
                feat = (np.r_[lastv / size, hist / size, (CTX - 1 - b) / CTX], c[tgt_i] - lastv, np.linalg.norm(extrap - c[tgt_i]))
        out.append((True, path, feat))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = G.load_cfg(a.config, a.split)
    K = cfg["checks"]
    n = a.n or K["n_designs"]
    seed = cfg["splits"][a.split]["seed"]
    t0 = time.time()
    specs = [G.sample_clip(cfg, seed, i) for i in range(n)]
    T, size = specs[0]["T"], specs[0]["size"]
    st = [G.target_state(s) for s in specs]
    res = {"config": os.path.abspath(a.config), "split": a.split, "n": n, "design_s": round(time.time() - t0, 1)}
    kinds = collections.Counter(s["objects"][0]["motion"]["kind"] for s in specs)
    res["target_kind_frac"] = {k: round(v / n, 3) for k, v in kinds.items()}
    res["event_panels_per_clip"] = round(float(np.mean([sum(p["role"] == "event" for p in s["panels"]) for s in specs])), 2)
    ev = [p for s in specs for p in s["panels"] if p["role"] == "event"]
    sl = [p for p in ev if p.get("mode") == "slide"]
    rel = collections.Counter(p.get("rel") for p in sl)
    res["event_panel_mode_frac"] = {k: round(v / max(len(ev), 1), 3) for k, v in collections.Counter(p.get("mode") for p in ev).items()}
    res["slide_event_rel_frac"] = {k: round(v / max(len(sl), 1), 3) for k, v in rel.items()}
    res["extra_slide_per_clip"] = round(float(np.mean([sum(p["role"] == "extra" and p.get("mode") == "slide" for p in s["panels"]) for s in specs])), 2)
    res["clips_with_full_hide"] = round(float(np.mean([(h >= 0.9).any() for _, h in st])), 3)
    per_stride = {}
    for s in K["strides"]:
        W = [windows(c, h, s, T, size) for c, h in st]
        late = [(i, p, f) for i, w in enumerate(W) for (l, p, f) in w if l]
        per_stride[s] = {"late_windows": len(late), "clips_with_late": round(len({i for i, _, _ in late}) / n, 3),
                         "distinct_late_paths": len({p for _, p, _ in late})}
        # 외우기 대 계산 (참고): 짝수 clip = 기억 창고, 홀수 clip = 질의
        A = [(f[0], f[1]) for i, _, f in late if f is not None and i % 2 == 0]
        B = [(f[0], f[1], f[2]) for i, _, f in late if f is not None and i % 2 == 1]
        if len(A) > 50 and len(B) > 50:
            FA, YA = np.array([x[0] for x in A]), np.array([x[1] for x in A])
            FB, YB, EX = np.array([x[0] for x in B]), np.array([x[1] for x in B]), np.array([x[2] for x in B])
            err = []
            for j in range(0, len(FB), 512):
                d = ((FB[j:j + 512, None, :] - FA[None]) ** 2).sum(-1)
                nn = d.argmin(1)
                err.append(np.linalg.norm(YA[nn] - YB[j:j + 512], axis=1))
            err = np.concatenate(err)
            per_stride[s]["future_err_px_median"] = {"nearest_neighbor_lookup": round(float(np.median(err)), 1),
                                                     "constant_velocity_extrap": round(float(np.median(EX)), 1)}
    res["late_by_stride"] = {str(k): v for k, v in per_stride.items()}
    s2 = per_stride.get(2) or per_stride[K["strides"][0]]
    gates = {
        "kind_min_frac": all(kinds.get(k, 0) / n >= K["kind_min_frac"] for k, p in cfg["motion"]["kinds"].items() if p > 0),
        "clips_with_late_min": s2["clips_with_late"] >= K["clips_with_late_min"],
        "distinct_late_paths_per_clip_min": s2["distinct_late_paths"] / n >= K["distinct_late_paths_per_clip_min"],
        "slide_event_min_frac": len(sl) / max(len(ev), 1) >= K.get("slide_event_min_frac", 0.0),
        "slide_rel_min_frac": all(rel.get(k, 0) / max(len(sl), 1) >= K.get("slide_rel_min_frac", 0.0) for k in ("same", "opposite")),
    }
    res["gates"] = gates
    res["pass"] = all(gates.values())
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
