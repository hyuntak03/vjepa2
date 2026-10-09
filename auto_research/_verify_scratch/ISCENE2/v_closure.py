"""(C) closure 재계산: 허용폭 (≤1 / 정확) · 슬롯 제외 · 복사 붕괴 · pan_cm base 와 교차 · 합의 필터 (pan_cm enc)."""
import json, numpy as np
from vcommon import *
rng = np.random.default_rng(2)
d = ST / "scene_closure"; m = json.load(open(d / "meta.json")); items = m["items"]; preds = m["preds"]; arms = m["arms"]; n = len(items)
AM = np.load(d / "am.npy").astype(np.int64); CT = np.load(d / "ct.npy").astype(np.float32); CC = np.load(d / "cc.npy").astype(np.float32)
SRC = np.stack([pan_src(it["v"], it["dir"]) for it in items]); D = np.stack([np.broadcast_to((it["v"] * (np.arange(8) + 1) / 8.0)[:, None], (8, 256)) for it in items]).astype(np.float32)
done = AM[:, 0, 0, 0, 0] >= 0; VAL = (SRC >= 0) & done[:, None, None]
R = Reach("scene_pan_cm"); AG = R.HE[:n] & VAL
res = dict(n_done=int(done.sum()), per_pred={})
def slot_rate(h, msk, j): return rate_ci((h & msk)[:, j].sum(1).astype(float), msk[:, j].sum(1).astype(float), rng)
def slots_delta(hA, hB, msk, js):
    mm = msk.copy(); sel = np.zeros(8, bool); sel[js] = True; mm &= sel[None, :, None]
    return rate_ci(((hA & mm).sum((1, 2)) - (hB & mm).sum((1, 2))).astype(float), mm.sum((1, 2)).astype(float), rng)
for pi, pk in enumerate(preds):
    rec = {}
    for tol, tn in ((1, "cheb1"), (0, "exact")):
        H = {a: (cheb(AM[:, pi, ai], SRC) <= tol) & VAL for ai, a in enumerate(arms)}
        for msk, mn in ((VAL, "all"), (AG, "agree")):
            r = dict(by_slot={a: [slot_rate(H[a], msk, j)[0] for j in range(8)] for a in arms})
            r["delta_slot"] = {f"{a}-os": [slot_rate(H[a], msk, j)[0] - slot_rate(H["os"], msk, j)[0] for j in range(8)] for a in arms[1:]}
            r["delta_pooled"] = {f"{a}-os_j{js[0]}-{js[-1]}": slots_delta(H[a], H["os"], msk, js) for a in arms[1:] for js in ([3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7], [5, 6, 7])}
            rec[f"{tn}_{mn}"] = r
    # 복사 붕괴: argmax == 자기 칸 (d ≥ 1 토큰), 팔·슬롯별;  cos(복사) − cos(출처)
    own = {a: ((AM[:, pi, ai] == np.arange(256)[None, None, :]) & VAL & (D >= 1)) for ai, a in enumerate(arms)}
    rec["own_frac_by_slot"] = {a: [round(float(own[a][:, j].sum() / max((VAL & (D >= 1))[:, j].sum(), 1)), 3) for j in range(8)] for a in arms}
    rec["cc_minus_ct_by_slot"] = {a: [round(float(np.nanmean((CC[:, pi, ai] - CT[:, pi, ai])[:, j][(VAL & (D >= 1))[:, j]])), 3) for j in range(8)] for ai, a in enumerate(arms)}
    rec["ct_by_slot"] = {a: [round(float(np.nanmean(CT[:, pi, ai][:, j][VAL[:, j]])), 3) for j in range(8)] for ai, a in enumerate(arms)}
    # os 와 pan_cm base 교차 (같은 clip): argmax 일치율, hit 일치
    if f"{pk}:base" in R.ai:
        cb = R.C[:n, R.ai[f"{pk}:base"]]
        rec["os_vs_pan_base_argmax_agree"] = round(float((cb == AM[:, pi, 0])[VAL].mean()), 4)
        rec["os_vs_pan_base_hit"] = [round(float(((cheb(cb, SRC) <= 1) & VAL)[VAL].mean()), 4), round(float(((cheb(AM[:, pi, 0], SRC) <= 1) & VAL)[VAL].mean()), 4)]
    res["per_pred"][pk] = rec
dump("closure", res)
for pk, rec in res["per_pred"].items():
    for key in ("cheb1_all", "exact_all", "cheb1_agree"):
        r = rec[key]; print(pk, key, {a: [round(x, 2) for x in v] for a, v in r["by_slot"].items()}); print("   Δ pooled", r["delta_pooled"])
    print(pk, "own_frac", rec["own_frac_by_slot"]); print(pk, "cc-ct", rec["cc_minus_ct_by_slot"]); print(pk, "xcheck", rec.get("os_vs_pan_base_argmax_agree"), rec.get("os_vs_pan_base_hit"))
