#!/usr/bin/env python3
"""decoder 검증 — 자 없이 (학습 파라미터 0) v11 p 안의 물체 검사 — 같은 문맥의 두 세계 h_pos (물체 있음) / h_imp (48 부터 물체 없음) 와 p 의 토큰별 L1.
(1) 진실 물체 자리 3x3 에서  r = |p - h_imp| / |p - h_pos|.  r > 1 이면 p 가 그 자리에 물체를 두었다 (물체 세계에 더 가깝다), r < 1 이면 빈 자리.
    같은 비율을 물체 밖 토큰에서 재면 ≈ 1 이어야 한다 (대조).
(2) s_i = |p - h_imp|_i - |p - h_pos|_i 의 최대 토큰 = 자 없이 찾은 물체 자리. 진실·마지막관측 과의 거리, 봉우리 세기 (max s / std s).
조건: flat/ramp × k=0 / early k4 / mid k4 / late k4 (+ late k1). 채점기와 같은 L1·같은 LN(h).
출력: figures/<train>/v11_vanish/timing/token_test.{json,md}
  ROLLOUT2_TRAIN=v5 python z_research/scripts/analysis/v11_token_object_test.py [n_per_cond]"""
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2"); sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
import rollout2_test_readout as rt
C = Path("/local_datasets/world/world_analysis/cache/v11_full_vith")
idx = list(csv.DictReader((ROOT / "data_csv/intphysgen_v11_full/index_probe.csv").open()))
meta = {r["name"]: r for r in csv.DictReader(open("/local_datasets/world/world_analysis/IntPhysGen_v11/metadata.csv"))}
row = {v: i for i, v in enumerate(json.loads((C / "meta.json").read_text())["video_ids"])}
P = np.load(C / "predictor.npy", mmap_mode="r"); H = np.load(C / "target.npy", mmap_mode="r")
CONDS = [("flat k=0", "moving_visible_flat", "0"), ("flat early k=4", "moving_occlusion_flat_early", "4"), ("flat mid k=4", "moving_occlusion_flat_mid", "4"), ("flat late k=1", "moving_occlusion_flat", "1"), ("flat late k=4", "moving_occlusion_flat", "4"),
         ("ramp k=0", "moving_visible", "0"), ("ramp early k=4", "moving_occlusion_early", "4"), ("ramp mid k=4", "moving_occlusion_mid", "4"), ("ramp late k=4", "moving_occlusion", "4")]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 56


def n3(x, y):
    cx, cy = int(x // 18), int(y // 18); return [r * 16 + c for r in range(cy - 1, cy + 2) for c in range(cx - 1, cx + 2) if 0 <= r < 16 and 0 <= c < 16]


res = {}
for name, cond, K in CONDS:
    cell = lambda v: {r["block_id"]: r for r in idx if r["condition"] == cond and r["sym_k"] == K and r["violation_type"] == "vanish" and r["variant"] == v}
    pos, imp = cell("pos_a"), cell("imp_ab"); blocks = sorted(pos)[:N]
    R_obj, R_bg, D_tr, D_last, PK, INF = [], [], [], [], [], []
    for b in blocks:
        p = np.asarray(P[row[pos[b]["video_id"]]], np.float32).reshape(8, 256, -1)
        hp = np.asarray(H[row[pos[b]["video_id"]]][2048:], np.float32).reshape(8, 256, -1); hi = np.asarray(H[row[imp[b]["video_id"]]][2048:], np.float32).reshape(8, 256, -1)
        dp = np.abs(p - hp).mean(-1); di = np.abs(p - hi).mean(-1)                       # (8, 256)
        t32 = np.stack([rt.arr(meta[pos[b]["video_id"]]["object_px_x_by_sample"]), rt.arr(meta[pos[b]["video_id"]]["object_px_y_by_sample"])], -1); tub = t32.reshape(16, 2, 2).mean(1); last = tub[7]
        ro, rb, dt, dl, pk, inf = [], [], [], [], [], []
        for t in range(8):
            sel = n3(*tub[8 + t]); oth = np.setdiff1d(np.arange(256), sel)
            ro.append(di[t, sel].mean() / dp[t, sel].mean()); rb.append(di[t, oth].mean() / dp[t, oth].mean())
            s = di[t] - dp[t]; i = int(s.argmax()); est = np.array([(i % 16) * 18 + 9, (i // 16) * 18 + 9])
            dt.append(np.linalg.norm(est - tub[8 + t])); dl.append(np.linalg.norm(est - last)); pk.append(s.max() / (s.std() + 1e-9)); inf.append(0 <= tub[8 + t, 0] < 288)
        R_obj.append(ro); R_bg.append(rb); D_tr.append(dt); D_last.append(dl); PK.append(pk); INF.append(inf)
    INF = np.array(INF); f = lambda a: [float(v) for v in np.array(a).mean(0)]
    res[name] = dict(n=len(blocks), r_obj=f(R_obj), r_bg=f(R_bg), d_truth=f(D_tr), d_last=f(D_last), peak=f(PK), in_frame=f(INF),
                     frac_r_obj_gt1=[float(v) for v in (np.array(R_obj) > 1).mean(0)])
    r = res[name]
    print(f"{name} (n={r['n']})"); print("   r_obj (>1 = 물체 있음)", " ".join(f"{v:5.2f}" for v in r["r_obj"])); print("   frac r_obj>1        ", " ".join(f"{v:5.2f}" for v in r["frac_r_obj_gt1"]))
    print("   r_bg (대조, ≈1)      ", " ".join(f"{v:5.2f}" for v in r["r_bg"])); print("   argmax s − 진실 px   ", " ".join(f"{v:5.0f}" for v in r["d_truth"]))
    print("   argmax s − 마지막관측", " ".join(f"{v:5.0f}" for v in r["d_last"])); print("   봉우리 세기 max/std  ", " ".join(f"{v:5.1f}" for v in r["peak"]), flush=True)
OUT = rt.FIG_ROOT / "v11_vanish/timing"; OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT / "token_test.json", "w"), indent=1)
md = ["# 자 없는 물체 검사 — 진실 물체 3×3 에서 r = |p − h_imp| / |p − h_pos| (>1 = p 가 그 자리에 물체를 둠), 물체 밖 토큰은 대조 (≈1)", "", "| 조건 | 지표 | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 |", "|---|---|---|---|---|---|---|---|---|---|"]
for name, r in res.items():
    md.append(f"| {name} (n={r['n']}) | r_obj | " + " | ".join(f"{v:.2f}" for v in r["r_obj"]) + " |"); md.append(f"| | frac r_obj>1 | " + " | ".join(f"{v:.2f}" for v in r["frac_r_obj_gt1"]) + " |")
    md.append(f"| | r_bg | " + " | ".join(f"{v:.2f}" for v in r["r_bg"]) + " |"); md.append(f"| | argmax s − 진실 px | " + " | ".join(f"{v:.0f}" for v in r["d_truth"]) + " |")
(OUT / "token_test.md").write_text("\n".join(md) + "\n"); print(f"→ {OUT}/token_test.{{json,md}}")
