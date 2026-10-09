"""균등 지도 (x̂ = 화면 중심) 의 ρ 귀무값 — 합의 토큰 d≥1, 구간별 중앙값."""
import numpy as np
from vcommon import *
out = {}
for name in ("scene_pan_cm", "scene_nat_ssv2_cm", "scene_nat_ek100_cm", "scene_nat_v3_cm"):
    R = Reach(name); m = R.HE & (R.D >= 1)
    sx = np.arange(256)[None, None, :] % G; sy = np.arange(256)[None, None, :] // G; ux, uy = R.SRC % G, R.SRC // G
    dvx, dvy = ux - sx, uy - sy; d2 = np.maximum(dvx ** 2 + dvy ** 2, 1e-9)
    rho0 = ((7.5 - sx) * dvx + (7.5 - sy) * dvy) / d2
    out[name] = {f"[{lo},{hi})": round(float(np.median(rho0[m & (R.D >= lo) & (R.D < hi)])), 3) for lo, hi in BINS[1:] if (m & (R.D >= lo) & (R.D < hi)).sum() > 50}
    out[name]["by_slot"] = [round(float(np.median(rho0[m & (R.SL == i)])), 3) if (m & (R.SL == i)).sum() > 50 else None for i in range(8)]
dump("null_rho", out); print(out)
