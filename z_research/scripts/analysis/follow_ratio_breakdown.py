"""AR 예측이 물체를 왜 일부만 옮기나 — 따라 움직인 비율을 실제 이동 거리 · 장면 · 속도별로 (2026-10-26 저녁, CPU).
사용자: "왜 그런 거지.. 실제로 조금 움직이긴 한다는 건데".
입력: position_readout_meaningless_samples.npz (attentive probe 가 낸 위치; enc = encoder 토큰, ro = AR 예측 출력 토큰, val, 블록 7 · b 물체 보임), ar_layers val_meta (장면).
비율 = Σ(예측 변위 · 실제 변위) / Σ|실제 변위|² (변위 = 위치 − 블록 7 위치). 실제로 16 px 이상 움직인 표본만. 예측 변위 크기 vs 실제 변위 크기도.
  /venv/vjepa/bin/python z_research/scripts/analysis/follow_ratio_breakdown.py   → z_analysis/results/follow_ratio_breakdown.json
"""
import json
import numpy as np
S = np.load("/workspace/world_model/vjepa2/z_analysis/results/position_readout_meaningless_samples.npz")
L = np.load("/workspace/auto_research_cache/attnprobe/val_ro_lab.npy"); L = L[(L[:, 7] >= 8) & (L[:, 7] <= 15)]
meta = {m["i"]: m for m in json.load(open("/workspace/auto_research_cache/ar_layers/val_meta.json"))}
scen = np.array([meta[int(i)]["scen"] for i in L[:, 6]])
ratio = lambda p, t: float((p * t).sum() / (t ** 2).sum()) if len(t) else float("nan")
res = {}
for nm in ("enc", "ro"):
    P, Y, Y7, st = S[f"{nm}_pred"], S[f"{nm}_true"], S[f"{nm}_last"], S[f"{nm}_step"]
    ok = ~np.isnan(Y7).any(1); t = Y - Y7; p = P - Y7; d = np.linalg.norm(t, axis=1); mv = ok & (d >= 16)
    r = {"by_distance": {}, "by_step_group_and_distance": {}}
    for lo, hi in ((16, 32), (32, 64), (64, 999)):
        m = mv & (d >= lo) & (d < hi); r["by_distance"][f"{lo}-{hi}px"] = {"ratio": round(ratio(p[m], t[m]), 3), "n": int(m.sum()),
                                                                         "pred_move_px_median": round(float(np.median(np.linalg.norm(p[m], axis=1))), 1), "true_move_px_median": round(float(np.median(d[m])), 1)}
    for gname, ks in (("steps 2-3", (2, 3)), ("steps 6-8", (6, 7, 8))):
        for lo, hi in ((16, 32), (32, 999)):
            m = mv & np.isin(st, ks) & (d >= lo) & (d < hi); r["by_step_group_and_distance"][f"{gname} | {lo}-{hi}px"] = {"ratio": round(ratio(p[m], t[m]), 3), "n": int(m.sum())}
    if nm == "ro":
        sc = scen if len(scen) == len(st) else None
        r["by_scene"] = {s: {"ratio": round(ratio(p[mv & (sc == s)], t[mv & (sc == s)]), 3), "n": int((mv & (sc == s)).sum())} for s in sorted(set(sc[mv]))} if sc is not None else {}
    res[nm] = r
json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/follow_ratio_breakdown.json", "w"), indent=1)
for nm in res:
    print("==", nm)
    for k, v in res[nm]["by_distance"].items(): print("  거리", k, v)
    for k, v in res[nm]["by_step_group_and_distance"].items(): print("  ", k, v)
for s, v in res["ro"].get("by_scene", {}).items(): print("  장면", s, v)
