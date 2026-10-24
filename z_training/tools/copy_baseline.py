#!/usr/bin/env python3
"""상태 head 위치 오차의 **복사 기준선** — 정답 상태만으로 (GPU · 모델 불필요) (2026-10-20).

  /venv/vjepa/bin/python z_training/tools/copy_baseline.py <run> [--n 3000]

run 의 config 에서 합성 창 설정 (manifest · 간격 · (C, K) 조합) 을 그대로 읽어, 학습과 같은 방식으로 창을 뽑는다.
굴린 미래 블록 (C .. C+K/2−1) 마다 물체 위치 오차 (px) 를 보임 / 가림으로 나눠:
  copy       마지막 문맥 블록 위치를 그대로
  const_vel  마지막 두 문맥 블록의 변위로 등속 외삽 (문맥 블록이 1 개면 copy 와 같다)
결과를 <run>/copy_baseline.json 에 쓴다 — z_training/monitor_ar.sh 가 읽어 기준선으로 쓴다 (없으면 옛 값 28 px)."""
import argparse, bisect, json, os, random, sys
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "z_research/scripts/harness"))
from app.vjepa_frozen.var_window import VarWindowBatches            # noqa: E402
from app.vjepa_frozen.state_head import pack_labels                 # noqa: E402
from z_training.data.phys2d_sup_labels import load_clip, window_labels   # noqa: E402
import yaml                                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--n", type=int, default=3000, help="뽑는 창 수")
a = ap.parse_args()
D = os.path.join(REPO, "z_training/runs", a.run)
cfg = yaml.safe_load(open(os.path.join(D, "config.yaml")))
bx = dict(cfg["data"]["var_window"]["boundary"])
from paths import expand                                             # noqa: E402
bx["manifest"] = expand(bx["manifest"])
ds = VarWindowBatches.__new__(VarWindowBatches)
ds.tubelet = 2
ds.transform = type("T", (), {"resolution": 256})()
ds._init_boundary(bx)
s = ds.bnd_stride
rng = random.Random(0)
res = {f"{m}_{v}": [] for m in ("copy", "const_vel") for v in ("vis", "hid")}
for _ in range(a.n):
    c, K = rng.choice(ds.bnd_combos)
    lo = bisect.bisect_left(ds.bnd_cmax, c)
    while True:
        j = rng.randrange(lo, len(ds.bnd_items))
        if ds.bnd_kmax[j] >= K:
            break
    _, path, b = ds.bnd_items[j]
    idx0 = b - s * c
    lab = pack_labels(window_labels(load_clip(path), start=idx0, n_frames=c + K, stride=s, tubelet=2, reduce="last"))
    C, R = c // 2, K // 2
    P, m, h = lab["pos"] * 256, lab["m_state"], lab["hidden"]
    for k in np.nonzero(m[C - 1])[0]:
        last = P[C - 1, k]
        v = (P[C - 1, k] - P[C - 2, k]) if C >= 2 and m[C - 2, k] else 0
        for t in range(C, C + R):
            if not m[t, k]:
                continue
            key = "hid" if h[t, k] else "vis"
            res["copy_" + key].append(float(np.linalg.norm(P[t, k] - last)))
            res["const_vel_" + key].append(float(np.linalg.norm(P[t, k] - (last + v * (t - C + 1)))))
out = {k: {"mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None, "n": len(v)} for k, v in res.items()}
out["setting"] = {"manifest": bx["manifest"], "stride": s, "combos": ds.bnd_combos, "n_windows": a.n}
json.dump(out, open(os.path.join(D, "copy_baseline.json"), "w"), indent=1)
for k, v in res.items():
    print(f"{k:14s} 평균 {np.mean(v):6.1f} px  중앙 {np.median(v):6.1f}  n {len(v)}")
print(f"→ {os.path.join(D, 'copy_baseline.json')}")
