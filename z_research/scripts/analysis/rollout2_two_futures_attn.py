#!/usr/bin/env python3
"""ledge / wall 두 미래 판정의 슬롯별 근거 — p 자의 attention 이 가능 궤적 / 불가능 궤적 어느 쪽 3×3 에 실리는지, 읽기가 '기본값' (균등 attention 읽기 = Linear(토큰 평균)) 에서
얼마나 떨어져 있는지, 기본값 자체는 두 궤적 중 어디에 가까운지. 위치 주장은 이 진단과 같이 읽는다 (물체가 없으면 자는 기본값을 낸다).
입력: exp_results/<train>/attentive_pooling/p/{preds.npz, attn.pt} + v2 predictor 캐시 (기본값 계산). 출력: figures/<train>/summary/two_futures_attn.{json,md}
  ROLLOUT2_TRAIN=v5 python z_research/scripts/analysis/rollout2_two_futures_attn.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout2_test_readout as rt                       # noqa: E402
from rollout2_attn_readout import AttnReadout            # noqa: E402
RES = rt.RES_ROOT / "attentive_pooling/p"; OUT = rt.FIG_ROOT / "summary"
KEYS = ("mass_pos", "mass_imp", "entropy", "d_read_uniform", "d_read_pos", "d_read_imp", "d_uniform_pos", "d_uniform_imp", "read_closer_imp", "uniform_closer_imp")


def mass3(w, x, y):
    cx, cy = int(x // 18), int(y // 18); sel = [r * 16 + c for r in range(cy - 1, cy + 2) for c in range(cx - 1, cx + 2) if 0 <= r < 16 and 0 <= c < 16]; return float(w[sel].sum())


def main():
    z = np.load(RES / "preds.npz", allow_pickle=True); a = z["attn"].astype(np.float32); tr = (z["truth"] + 1) * rt.RES; pr = (z["pred"] + 1) * rt.RES; sc = z["scenario"]; pl = z["plausible"]
    idx = {r["video_id"]: r for r in csv.DictReader(rt.INDEX.open())}; blk = np.array([idx[v]["block_id"] for v in z["video_id"]])
    P = np.load(rt.CACHE / "predictor.npy", mmap_mode="r"); model = AttnReadout(); model.load_state_dict(torch.load(RES / "attn.pt", map_location="cpu")); model.eval()
    res = {}
    for s in ("ledge", "wall"):
        rows = np.where((sc == s) & (pl == 1))[0]; M = {k: [] for k in KEYS}
        for i in rows:
            j = np.where((blk == blk[i]) & (pl == 0))[0][0]                                   # 같은 block 의 불가능 clip (p 는 공유, 진실 궤적만 다름)
            with torch.no_grad():
                un = (model.out(torch.from_numpy(np.asarray(P[i], np.float32)).reshape(8, 256, -1).mean(1)).numpy() + 1) * rt.RES
            M["mass_pos"].append([mass3(a[i, t], *tr[i, t]) for t in range(8)]); M["mass_imp"].append([mass3(a[i, t], *tr[j, t]) for t in range(8)])
            M["entropy"].append(-(a[i] * np.log(a[i] + 1e-12)).sum(-1)); M["d_read_uniform"].append(np.linalg.norm(pr[i] - un, axis=-1))
            dp = np.linalg.norm(pr[i] - tr[i], axis=-1); di = np.linalg.norm(pr[i] - tr[j], axis=-1); up = np.linalg.norm(un - tr[i], axis=-1); ui = np.linalg.norm(un - tr[j], axis=-1)
            M["d_read_pos"].append(dp); M["d_read_imp"].append(di); M["d_uniform_pos"].append(up); M["d_uniform_imp"].append(ui); M["read_closer_imp"].append(di < dp); M["uniform_closer_imp"].append(ui < up)
        res[s] = {"n": int(len(rows)), **{k: [float(v) for v in np.mean(M[k], 0)] for k in KEYS}}
    OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT / "two_futures_attn.json", "w"), indent=1)
    md = ["# ledge / wall — 슬롯별 attention 근거 (p 자, 학습셋 v5)", "", "mass_* = 가능/불가능 궤적 3×3 칸의 attention 질량 (균등 0.035). d_* px. uniform = Linear(토큰 평균) = 자가 물체를 못 찾을 때의 기본값.", ""]
    for s, r in res.items():
        md += [f"## {s} (n={r['n']})", "", "| | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 |", "|---|---|---|---|---|---|---|---|---|"]
        for k in KEYS:
            md.append(f"| {k} | " + " | ".join((f"{v:.2f}" if k.startswith(("mass", "entropy")) or k.endswith("closer_imp") else f"{v:.0f}") for v in r[k]) + " |")
        md.append(""); print(f"{s} n={r['n']}"); [print(f"   {k:<18}", " ".join(f"{v:5.2f}" if k.startswith(('mass', 'entropy')) or k.endswith('closer_imp') else f"{v:5.0f}" for v in r[k])) for k in KEYS]
    (OUT / "two_futures_attn.md").write_text("\n".join(md) + "\n"); print(f"→ {OUT}/two_futures_attn.{{json,md}}")


if __name__ == "__main__":
    main()
