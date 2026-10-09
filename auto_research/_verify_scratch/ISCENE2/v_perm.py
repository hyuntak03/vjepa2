"""(P) permanence 재계산 + 공격: 템플릿 의존 · encoder 대조 · k 단조 · 회색 대조 · 비가림 대조 (pan_cm 같은 clip)."""
import json, numpy as np
from vcommon import *
rng = np.random.default_rng(1)
d = ST / "scene_perm"; m = json.load(open(d / "meta.json")); items = m["items"]; preds = m["preds"]; n = len(items)
AM = np.load(d / "argmax_prev.npy").astype(np.int64); CP = np.load(d / "cos_prev.npy").astype(np.float32); CLs = np.load(d / "cos_last.npy").astype(np.float32)
HID = np.load(d / "hid.npy"); EP = np.load(d / "enc_prev.npy").astype(np.float32); EL = np.load(d / "enc_last.npy").astype(np.float32)
K = np.array([it["k"] for it in items]); V = np.array([it["v"] for it in items]); DR = np.array([it["dir"] for it in items]); Mm = (K + 1) // 2
SRCm = np.stack([pan_src(V[c], DR[c], extra=Mm[c]) for c in range(n)])
SRC0 = np.stack([pan_src(V[c], DR[c]) for c in range(n)])
D = np.stack([np.broadcast_to((V[c] * (np.arange(8) + 1) / 8.0)[:, None], (8, 256)) for c in range(n)]).astype(np.float32)
done = AM[:, 0, 0, 0] >= 0; VAL = (SRCm >= 0) & done[:, None, None]
CL = np.broadcast_to(np.arange(n)[:, None, None], SRCm.shape)
print("done", done.sum(), "/", n, " hid 비율", HID[VAL].mean().round(3))
# 비가림 대조: pan_cm 의 같은 clip (앞 800) base 팔
R = Reach("scene_pan_cm"); NU = min(n, R.n); assert all(R.items[c]["file"] == items[c]["file"] and R.items[c]["v"] == items[c]["v"] for c in range(NU))
INU = (np.arange(n) < NU)[:, None, None]
UN = {pk: np.concatenate([R.hit(f"{pk}:base")[:NU], np.zeros((n - NU, 8, 256), bool)]) for pk in ("R", "A", "P")}; UNE = np.concatenate([R.HE[:NU], np.zeros((n - NU, 8, 256), bool)])
res = dict(n_done=int(done.sum()), rows=[], k_mono={}, gray={}, unocc={}, template={})
def sub(v, msk): return rate_ci((v & msk).sum((1, 2)).astype(float), msk.sum((1, 2)).astype(float), rng)
def subm(v, msk):  # 평균 (연속값)
    num = np.where(msk, v, 0).sum((1, 2)); den = msk.sum((1, 2)).astype(float); return rate_ci(num, den, rng)
for pi, pk in enumerate(preds):
    hit = (cheb(AM[:, pi], SRCm) <= 1) & VAL
    hit0 = (cheb(AM[:, pi], SRCm) == 0) & VAL
    for k in (2, 4, 8):
        mk = VAL & (K == k)[:, None, None]
        for lo, hi in BINS[:5]:
            mb = mk & (D >= lo) & (D < hi); mh, mv = mb & HID, mb & ~HID
            if mh.sum() < 50 or mv.sum() < 50: continue
            num = ((hit & mh).sum((1, 2)) / np.maximum(mh.sum((1, 2)), 1) - (hit & mv).sum((1, 2)) / np.maximum(mv.sum((1, 2)), 1))
            okc = (mh.sum((1, 2)) > 0) & (mv.sum((1, 2)) > 0)
            gap = rate_ci(np.where(okc, num, 0), okc.astype(float), rng)      # clip 별 (hid − vis) 평균
            res["rows"].append(dict(pred=pk, k=k, lo=lo, hi=hi, n_hid=int(mh.sum()), n_vis=int(mv.sum()), A_hid=sub(hit, mh), A_vis=sub(hit, mv), gap_clipmean=gap,
                                    A0_hid=sub(hit0, mh)[0], A0_vis=sub(hit0, mv)[0],
                                    cos_prev_hid=subm(CP[:, pi], mh)[0], cos_prev_vis=subm(CP[:, pi], mv)[0], cos_last_hid=subm(CLs[:, pi], mh)[0], cos_last_vis=subm(CLs[:, pi], mv)[0],
                                    enc_prev_hid=subm(EP, mh)[0], enc_prev_vis=subm(EP, mv)[0], enc_last_hid=subm(EL, mh)[0], enc_last_vis=subm(EL, mv)[0]))
    # k 단조 (근거리 0–2 칸 합쳐서): hid / vis 비율 + CI
    for k in (2, 4, 8):
        mb = VAL & (K == k)[:, None, None] & (D < 2); mh, mv = mb & HID, mb & ~HID
        ah, av = sub(hit, mh), sub(hit, mv)
        ratio = []
        for _ in range(300):
            s = rng.choice(np.where(done)[0], done.sum()); 
            ratio.append((hit & mh)[s].sum() / max((mh[s]).sum(), 1) / max((hit & mv)[s].sum() / max(mv[s].sum(), 1), 1e-9))
        res["k_mono"][f"{pk}_k{k}_d<2"] = dict(A_hid=ah, A_vis=av, ratio=[round(ah[0] / max(av[0], 1e-9), 3), round(float(np.percentile(ratio, 2.5)), 3), round(float(np.percentile(ratio, 97.5)), 3)])
    # 회색 대조: hid 토큰에서 predictor cos(prev 내용) vs cos(회색 띠 토큰) 와 같은 토큰의 encoder 실제 미래 토큰 (가림 없는 미래) 의 두 cos
    for k in (4, 8):
        mh = VAL & (K == k)[:, None, None] & (D < 2) & HID
        res["gray"][f"{pk}_k{k}_hid_d<2"] = dict(pred_cos_prev=subm(CP[:, pi], mh), pred_cos_gray=subm(CLs[:, pi], mh), enc_cos_prev=subm(EP, mh), enc_cos_gray=subm(EL, mh),
                                                 frac_pred_gray_gt_prev=round(float((CLs[:, pi] > CP[:, pi])[mh].mean()), 3), frac_enc_gray_gt_prev=round(float((EL > EP)[mh].mean()), 3))
    # 비가림 대조 (같은 clip, 띠 없음, 판독 = hc_L 템플릿): hid 자리 vs vis 자리 적중, 그리고 vis 토큰의 가림 run 대비
    if pk in UN:
        for k in (2, 4, 8):
            for lo, hi in BINS[:3]:
                mb = VAL & INU & (K == k)[:, None, None] & (D >= lo) & (D < hi); mh, mv = mb & HID, mb & ~HID
                res["unocc"][f"{pk}_k{k}_[{lo},{hi})"] = dict(unocc_hidcells=sub(UN[pk], mh), unocc_viscells=sub(UN[pk], mv), occ_hid=sub(hit, mh), occ_vis=sub(hit, mv),
                                                              enc_unocc_hidcells=sub(UNE, mh), enc_unocc_viscells=sub(UNE, mv))
    # 템플릿 의존: vis 토큰에서 cos_prev (L−m 템플릿) vs cos_last (L 템플릿, 같은 내용) — 같으면 템플릿 탓이 아님
    for k in (2, 4, 8):
        mv = VAL & (K == k)[:, None, None] & (D < 2) & ~HID
        res["template"][f"{pk}_k{k}_vis_d<2"] = dict(cos_prev=subm(CP[:, pi], mv)[0], cos_last=subm(CLs[:, pi], mv)[0], enc_prev=subm(EP, mv)[0], enc_last=subm(EL, mv)[0])
# 합의 근사: encoder 가 L−m 템플릿과 cos ≥ 0.5 인 토큰만 (판독이 성립하는 자리)
res["agree_approx"] = {}
for pi, pk in enumerate(preds):
    hit = (cheb(AM[:, pi], SRCm) <= 1) & VAL
    for k in (4, 8):
        mb = VAL & (K == k)[:, None, None] & (D < 2) & (EP >= 0.5); mh, mv = mb & HID, mb & ~HID
        res["agree_approx"][f"{pk}_k{k}_d<2_encprev>=0.5"] = dict(n_hid=int(mh.sum()), n_vis=int(mv.sum()), A_hid=sub(hit, mh), A_vis=sub(hit, mv))
dump("perm", res)
for r in res["rows"]:
    if r["lo"] < 2: print(f"{r['pred']:3s} k{r['k']} [{r['lo']},{r['hi']}) hid {r['A_hid'][0]:.3f} vis {r['A_vis'][0]:.3f} gap {r['gap_clipmean']}  cosprev h/v {r['cos_prev_hid']:.2f}/{r['cos_prev_vis']:.2f} coslast h/v {r['cos_last_hid']:.2f}/{r['cos_last_vis']:.2f} encprev h/v {r['enc_prev_hid']:.2f}/{r['enc_prev_vis']:.2f} enclast h/v {r['enc_last_hid']:.2f}/{r['enc_last_vis']:.2f}")
print(json.dumps(res["k_mono"], indent=0)); print(json.dumps(res["gray"], indent=0)); print(json.dumps(res["unocc"], indent=0)); print(json.dumps(res["template"], indent=0)); print(json.dumps(res["agree_approx"], indent=0))
