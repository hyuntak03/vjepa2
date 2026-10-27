"""모양 × 색 판독 정확도가 왜 낮은가 — encoder 1280, 블록 8–14 합쳐 학습 (2026-10-26, CPU).
사용자: "shape × color 조합 acc 가 왜 이렇게 낮지?"
판독기 = identity_1280_vs_384_same_samples.py 의 "합침" 과 같다 (선형층, 표준화, AdamW lr 1e-3 · wd 1e-2, 최대 100 epoch). 모양 (6) · 색 (8) · 모양×색 (48) 따로.
보는 것: 각 정확도 · 따로 맞힌 둘이 다 맞을 확률 · 헷갈림 상위 · 물체 크기 (블록에서 물체가 걸친 칸 수, ar_layers val_meta) 별 정확도.
  /venv/vjepa/bin/python z_research/scripts/analysis/identity_readout_why_low.py   → z_analysis/results/identity_readout_why_low.json
"""
import copy, json
from collections import Counter
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(32)
P = "/workspace/auto_research_cache/ar_layers_pooled"
TR, VA = np.load(f"{P}/train.npz"), np.load(f"{P}/val.npz")
meta = {m["i"]: m for m in json.load(open("/workspace/auto_research_cache/ar_layers/val_meta.json"))}
SH = ["sphere", "cube", "cylinder", "cone", "pyramid", "torus"]; CO = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
BLOCKS = list(range(8, 15))


def fit(X, y, clip, K):
    torch.manual_seed(0); X, y = torch.from_numpy(X), torch.from_numpy(y); h = torch.from_numpy(clip % 5 == 0)
    head = torch.nn.Linear(X.shape[1], K); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2); best = (-1, None)
    Xa, ya, Xh, yh = X[~h], y[~h], X[h], y[h]
    for ep in range(100):
        perm = torch.randperm(len(Xa))
        for i in range(0, len(Xa), 256):
            idx = perm[i:i + 256]; loss = F.cross_entropy(head(Xa[idx]), ya[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            a = (head(Xh).argmax(1) == yh).float().mean().item()
        if a > best[0]:
            best = (a, copy.deepcopy(head.state_dict()))
    head.load_state_dict(best[1]); return head


def gather(Z, with_cells=False):
    X, s, c, cl, nc = [], [], [], [], []
    for b in BLOCKS:
        m = Z["lab"][:, b, 0] > 0; X.append(Z["enc"][m, b].astype(np.float32)); s.append(Z["lab"][m, b, 1].astype(np.int64)); c.append(Z["lab"][m, b, 2].astype(np.int64)); cl.append(np.nonzero(m)[0])
        if with_cells:
            nc.append(np.array([len(meta[int(i)]["blocks"][b]["cells"]) for i in Z["i"][m]]))
    return np.concatenate(X), np.concatenate(s), np.concatenate(c), np.concatenate(cl), (np.concatenate(nc) if with_cells else None)


Xt, st, ct, clt, _ = gather(TR); Xv, sv, cv, _, ncell = gather(VA, True)
mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Xt, Xv = (Xt - mu) / sd, (Xv - mu) / sd
pred = {}
for name, yt, K in (("shape", st, 6), ("color", ct, 8), ("joint", st * 8 + ct, 48)):
    h = fit(Xt, yt, clt, K)
    with torch.no_grad():
        pred[name] = h(torch.from_numpy(Xv)).argmax(1).numpy()
ok_s, ok_c, ok_j = pred["shape"] == sv, pred["color"] == cv, pred["joint"] == sv * 8 + cv
res = {"n_val": int(len(sv)), "shape_acc": round(ok_s.mean() * 100, 1), "color_acc": round(ok_c.mean() * 100, 1), "joint_acc": round(ok_j.mean() * 100, 1),
       "both_separate_heads_correct": round((ok_s & ok_c).mean() * 100, 1), "product": round(ok_s.mean() * ok_c.mean() * 100, 1),
       "joint_head_shape_part_acc": round((pred["joint"] // 8 == sv).mean() * 100, 1), "joint_head_color_part_acc": round((pred["joint"] % 8 == cv).mean() * 100, 1)}
res["shape_confusions_top"] = [(f"{SH[a]}→{SH[b]}", n) for (a, b), n in Counter(zip(sv[~ok_s], pred["shape"][~ok_s])).most_common(6)]
res["color_confusions_top"] = [(f"{CO[a]}→{CO[b]}", n) for (a, b), n in Counter(zip(cv[~ok_c], pred["color"][~ok_c])).most_common(6)]
res["shape_acc_per_class"] = {SH[k]: round(ok_s[sv == k].mean() * 100, 1) for k in range(6)}
res["color_acc_per_class"] = {CO[k]: round(ok_c[cv == k].mean() * 100, 1) for k in range(8)}
bins = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 999)]
res["by_object_cells"] = {f"{lo}-{hi}": {"n": int(((ncell >= lo) & (ncell <= hi)).sum()), "shape": round(ok_s[(ncell >= lo) & (ncell <= hi)].mean() * 100, 1),
                                         "color": round(ok_c[(ncell >= lo) & (ncell <= hi)].mean() * 100, 1), "joint": round(ok_j[(ncell >= lo) & (ncell <= hi)].mean() * 100, 1)}
                          for lo, hi in bins if ((ncell >= lo) & (ncell <= hi)).sum() > 0}
res["object_cells_quantiles"] = np.percentile(ncell, [10, 25, 50, 75, 90]).tolist()
json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/identity_readout_why_low.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
