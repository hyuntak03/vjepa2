#!/usr/bin/env python3
"""e_v11_presence.py 산출물 → presence vs placement. python e_v11_presence_analyze.py <dir>  (OUT_TAG)

쌍마다: 정답 = mean(M) > 0. 복사 정답 = mean(Mc) > 0.
placement share = Σ_{Dh 상위 q 칸} M⁺ / Σ M⁺  (M⁺ = max(M, 0), q = 상위 5 % 칸, 슬롯별 Dh 순위). 균등이면 0.05.
Dh 상위 칸에서의 margin 부호 = 물체 자리에서 가능 쪽을 고르나 (placement 의 직접 신호); Dh 하위 90 % 칸의 margin 합 = 자리 밖 (presence · 전역) 신호.
kind = obj (pos_a vs imp_ab: 문맥에 물체, 가능 = 다시 나타남) / empty (pos_b vs imp_ba). 조건 · sym_k 별. block bootstrap CI.
"""
import json, os, sys
from pathlib import Path
import numpy as np

I = Path(sys.argv[1]); TAG = os.environ.get("OUT_TAG", ""); ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/scene"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(I / "meta.json")); preds = meta["preds"]; pairs = meta["pairs"]; n = len(pairs)
M = np.load(I / "M.npy").astype(np.float32); Mc = np.load(I / "Mc.npy").astype(np.float32); Dh = np.load(I / "Dh.npy").astype(np.float32)
done = np.isfinite(M[:, 0, 0, 0]); print("완료", int(done.sum()), "/", n)
K = np.array([p["kind"] for p in pairs]); COND = np.array([p["condition"] for p in pairs]); BLK = np.array([p["block"] for p in pairs]); SK = np.array([p["sym_k"] for p in pairs])
rng = np.random.default_rng(0)


def boot(v, blk, B=1000):
    ok = np.isfinite(v); v, blk = v[ok], blk[ok]
    if len(v) == 0: return [np.nan] * 3
    ub = np.unique(blk); ix = {b: np.where(blk == b)[0] for b in ub}
    bs = [v[np.concatenate([ix[b] for b in rng.choice(ub, len(ub))])].mean() for _ in range(B)]
    return [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


res = dict(n=int(done.sum()), preds=preds, rows=[])
q = 0.05; kq = int(round(256 * q))
for pi, pk in enumerate(preds):
    m = M[:, pi]; dh = Dh[:, pi]; mc = Mc[:, pi]
    correct = m.mean((1, 2)) > 0; ccopy = mc.mean((1, 2)) > 0
    top = np.argsort(-dh, axis=-1)[..., :kq]                                       # (n, 8, kq) Dh 상위 칸 (슬롯별)
    mp_ = np.maximum(m, 0); tot = mp_.sum((1, 2)) + 1e-9
    share = np.take_along_axis(mp_, top, -1).sum((1, 2)) / tot                     # placement share
    m_top = np.take_along_axis(m, top, -1).mean((1, 2)); m_rest = (m.sum((1, 2)) - np.take_along_axis(m, top, -1).sum((1, 2))) / (256 * 8 - kq * 8)
    for kind in ("obj", "empty", "all"):
        sel = done & ((K == kind) if kind != "all" else True)
        for cond in ("all",) + tuple(sorted(set(COND))):
            s2 = sel & ((COND == cond) if cond != "all" else True)
            if s2.sum() < 8: continue
            res["rows"].append(dict(pred=pk, kind=kind, condition=cond, n=int(s2.sum()), acc=boot(correct[s2].astype(float), BLK[s2]), acc_copy=boot(ccopy[s2].astype(float), BLK[s2]),
                                    placement_share=boot(share[s2], BLK[s2]), margin_top=boot(m_top[s2], BLK[s2]), margin_rest=boot(m_rest[s2], BLK[s2]),
                                    share_when_correct=boot(share[s2 & correct], BLK[s2 & correct]) if (s2 & correct).sum() >= 8 else None,
                                    slot_margin=[round(float(m[s2, i].mean()), 4) for i in range(8)]))
json.dump(res, open(OUT / f"v11_presence{TAG}.json", "w"), indent=1, default=float)
print("kind  cond  pred | acc  copy | placement share (균등 .05) | margin top / rest | share when correct")
for r in res["rows"]:
    if r["condition"] == "all":
        print(f"{r['kind']:6s} {r['condition']:6s} {r['pred']:3s} | {r['acc'][0]:.2f} {r['acc_copy'][0]:.2f} | {r['placement_share'][0]:.3f} [{r['placement_share'][1]:.3f},{r['placement_share'][2]:.3f}] | {r['margin_top'][0]:+.4f} / {r['margin_rest'][0]:+.4f} | {r['share_when_correct'][0] if r['share_when_correct'] else float('nan'):.3f}")
