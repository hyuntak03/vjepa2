"""AR predictor 의 예측 미래 p 에 물체가 '어디에도 없나' 아니면 '엉뚱한 자리에 있나' (2026-10-26).

vobench_ar_token_transfer.py extract 가 남긴 토큰 memmap (h 16 블록 · p 8 블록) 을 그대로 쓴다. GPU 추출 없음.

판정기 둘 (모두 참 h 에서만 정한다, p 는 안 본다):
  (A) 토큰 물체다움 자: 토큰마다 선형 1280 → 1 (logistic). 물체 토큰 = 칸 중심이 물체 상자 (label ext_u · ext_v) 안,
      배경 토큰 = 상자에서 1.5 칸 넘게 떨어진 것. train 영상 h 의 보이는 블록에서 학습.
  (B) 본 맞춤 (파라미터 없음): 마지막으로 보인 문맥 블록의 물체 칸 h 토큰과의 코사인.
문턱: test 영상 h 의 보이는 블록에서 '배경 토큰 최대값' 의 99 백분위 → 물체가 없는 블록이 넘을 확률 ≈ 1 %.
걸음마다 p 블록을 분류:
  none      = 문턱을 넘는 토큰이 없다
  at_truth  = 최고 토큰이 참 위치에서 ≤ 1.5 칸
  at_last   = 마지막 관측 위치에서 ≤ 1.5 칸 (참에서는 > 1.5)
  on_path   = 마지막 관측 → 참 선분에서 수직 ≤ 1.5 칸 (위 둘 아님)
  elsewhere = 나머지
+ 문턱 넘는 토큰 수 (번짐), kymograph (운동 축 좌표 −4…12 칸 × 걸음, 축에서 수직 ≤ 1.5 칸 토큰의 평균 점수).

  $PY z_research/scripts/analysis/vobench_ar_token_where.py --tok_dir <scratch>/tok_vast70 [--name vast70]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import vobench_ar_layer_velocity_probe as lv      # noqa: E402
import vobench_ar_token_transfer as tt             # noqa: E402

S, T, C, G, D = 256, 8, 8, 16, 1280
CATS = ["none", "at_truth", "at_last", "on_path", "elsewhere"]
KY = np.arange(-4, 12.01, 0.5)                     # kymograph 운동 축 칸


def labels_ext(vids):
    """블록마다 상자 반폭 (칸)."""
    rows = {r["video_id"]: r for r in lv.rows_obj()}
    lab, E = {}, []
    for v in vids:
        r = rows[v]; arm = "A" if r["variant"] == "pos_a" else "B"
        b = lv.FR / Path(r["file_name"]).parent
        if b not in lab:
            lab[b] = json.load(open(b / "label.json"))["labels"]
        L = lab[b][arm]
        E.append([[max(L[3 * (2 * k)]["ext_u"], L[3 * (2 * k) + 3]["ext_u"]) / tt.CELL,
                   max(L[3 * (2 * k)]["ext_v"], L[3 * (2 * k) + 3]["ext_v"]) / tt.CELL] for k in range(2 * T)])
    return np.array(E, np.float32)


def main(name, tok_dir, steps=3000):
    import torch
    dev = "cuda"
    vids, Hs, Ps = tt.load_tokens(tok_dir)
    XY, VIS, meta = tt.block_labels(vids); EXT = labels_ext(vids)
    g = np.array([m["block"] for m in meta]); mot = np.array([m["motion"] for m in meta]); occ = np.array([m["occ"] for m in meta])
    tr = lv.block_split(g, mot); te = ~tr
    cy, cx = np.meshgrid(np.arange(G) + 0.5, np.arange(G) + 0.5, indexing="ij")
    cen = np.stack([cx.ravel(), cy.ravel()], 1).astype(np.float32)               # (256, 2)

    def tok_labels(xy, ext):                                                        # (b,2),(b,2) → obj (b,256), bg (b,256)
        d = np.abs(cen[None] - xy[:, None])
        obj = (d[..., 0] <= np.maximum(ext[:, None, 0], 0.5)) & (d[..., 1] <= np.maximum(ext[:, None, 1], 0.5))
        bg = (d[..., 0] > ext[:, None, 0] + 1.5) | (d[..., 1] > ext[:, None, 1] + 1.5)
        return obj, bg

    # ---- (A) 물체다움 자 학습 (train h 보이는 블록)
    itr = np.where(tr)[0]
    Htr = tt.cat_rows(Hs, itr)[VIS[itr]]                                            # (nb, 256, D)
    obj, bg = tok_labels(XY[itr][VIS[itr]], EXT[itr][VIS[itr]])
    rng = np.random.default_rng(0)
    bi, ti = np.where(obj); Xo = Htr[bi, ti]
    bi, ti = np.where(bg); k = rng.choice(len(bi), min(len(bi), 8 * len(Xo)), replace=False); Xb = Htr[bi[k], ti[k]]
    del Htr
    X = torch.tensor(np.concatenate([Xo, Xb]), device=dev).float(); Y = torch.cat([torch.ones(len(Xo)), torch.zeros(len(Xb))]).to(dev)
    mu, sd = X.mean(0), X.std(0) + 1e-6; X = (X - mu) / sd
    head = torch.nn.Linear(D, 1).to(dev); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor(len(Xb) / len(Xo), device=dev); gen = torch.Generator(device=dev).manual_seed(0)
    for it in range(steps):
        b = torch.randint(0, len(X), (4096,), device=dev, generator=gen)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(X[b]).squeeze(-1), Y[b], pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"[objness] train tokens obj {len(Xo)} bg {len(Xb)} loss {loss.item():.3f}", flush=True)
    del X

    def score(Xb):                                                                  # (b, 256, D) → (b, 256)
        with torch.no_grad():
            return head((torch.tensor(Xb, device=dev).float() - mu) / sd).squeeze(-1).cpu().numpy()

    # ---- test 영상 점수 맵
    ite = np.where(te)[0]; n = len(ite)
    SH = np.zeros((n, 2 * T, S), np.float32); SP = np.zeros((n, T, S), np.float32)
    CH = np.full((n, T, S), np.nan, np.float32); CP = np.full((n, T, S), np.nan, np.float32)
    last = np.zeros(n, int)
    for k0 in range(0, n, 32):
        sel = ite[k0:k0 + 32]; Hb = tt.cat_rows(Hs, sel); Pb = tt.cat_rows(Ps, sel); B = len(sel)
        SH[k0:k0 + B] = score(Hb.reshape(B * 2 * T, S, D)).reshape(B, 2 * T, S)
        SP[k0:k0 + B] = score(Pb.reshape(B * T, S, D)).reshape(B, T, S)
        Hn = torch.nn.functional.normalize(torch.tensor(Hb, device=dev).float(), dim=-1)
        Pn = torch.nn.functional.normalize(torch.tensor(Pb, device=dev).float(), dim=-1)
        for j, i in enumerate(sel):
            vc = np.where(VIS[i, :C])[0]
            if len(vc) == 0:
                continue
            lb = int(vc[-1]); last[k0 + j] = lb; x, y = XY[i, lb]
            tm = Hn[j, lb, int(np.clip(y, 0, G - 1e-3)) * G + int(np.clip(x, 0, G - 1e-3))]
            CH[k0 + j] = (Hn[j, C:] @ tm).cpu().numpy(); CP[k0 + j] = (Pn[j] @ tm).cpu().numpy()
    XYt, VISt, EXTt = XY[ite], VIS[ite], EXT[ite]; mott, occt = mot[ite], occ[ite]
    lastxy = XYt[np.arange(n), last]

    # ---- 문턱: test h 보이는 미래 블록의 배경 토큰 최대값 99 백분위
    def bg_max(Smap, xy, ext):
        _, bg = tok_labels(xy, ext); return np.where(bg, Smap, -np.inf).max(-1)
    vf = VISt[:, C:]
    xyf = XYt[:, C:][vf]; exf = EXTt[:, C:][vf]
    thrA = float(np.quantile(bg_max(SH[:, C:][vf], xyf, exf), 0.99))
    thrB = float(np.quantile(bg_max(CH[vf], xyf, exf), 0.99))
    obj_f, _ = tok_labels(xyf, exf)
    detA_h = float((np.where(obj_f, SH[:, C:][vf], -np.inf).max(-1) > thrA).mean())
    detB_h = float((np.where(obj_f, CH[vf], -np.inf).max(-1) > thrB).mean())
    res = dict(n_test=n, thr_objness=thrA, thr_template=thrB,
               check_h_visible_detect_objness=detA_h, check_h_visible_detect_template=detB_h,
               check_h_hidden_any_over_thr_objness=float((SH[:, C:][~vf].max(-1) > thrA).mean()) if (~vf).any() else None)

    def classify(Smap, thr, truth, lxy):                                            # (m,256) → 범주 · 문턱 넘는 토큰 수 · 최고 칸
        over = Smap > thr; cnt = over.sum(-1); top = cen[np.nanargmax(np.where(np.isnan(Smap), -np.inf, Smap), -1)]
        dt = np.linalg.norm(top - truth, axis=-1); dl = np.linalg.norm(top - lxy, axis=-1)
        seg = truth - lxy; L2 = (seg ** 2).sum(-1) + 1e-6; u = np.clip(((top - lxy) * seg).sum(-1) / L2, 0, 1)
        dperp = np.linalg.norm(top - (lxy + u[:, None] * seg), axis=-1)
        cat = np.where(cnt == 0, 0, np.where(dt <= 1.5, 1, np.where(dl <= 1.5, 2, np.where(dperp <= 1.5, 3, 4))))
        return cat, cnt, top

    groups = {}
    for o, osel in (("not_late", occt != "late"), ("late", occt == "late")):
        for m_ in np.unique(mott):
            s = np.where(osel & (mott == m_))[0]
            if len(s) == 0:
                continue
            d = dict(n=int(len(s)))
            for tag, Smap, thr in (("pA", SP, thrA), ("pB", CP, thrB), ("hA", SH[:, C:], thrA), ("hB", CH, thrB)):
                fr = np.zeros((T, len(CATS))); cnts = np.zeros(T)
                for t in range(T):
                    cat, cnt, _ = classify(Smap[s, t], thr, XYt[s, C + t], lastxy[s])
                    fr[t] = np.bincount(cat, minlength=len(CATS)) / len(s); cnts[t] = cnt.mean()
                d[f"{tag}_frac"] = fr.tolist(); d[f"{tag}_n_over"] = cnts.tolist()
            # kymograph: 운동 축 좌표 (마지막 관측 기준) × 걸음, 축 수직 ≤ 1.5 칸 토큰의 평균 점수
            ax = XYt[s, C + T - 1] - lastxy[s]; nr = np.linalg.norm(ax, axis=-1, keepdims=True)
            ax = np.where(nr > 1e-6, ax / np.maximum(nr, 1e-6), np.array([1.0, 0.0]))
            rel = cen[None] - lastxy[s][:, None]                                        # (m, 256, 2)
            along = (rel * ax[:, None]).sum(-1); perp = np.abs(rel[..., 0] * ax[:, None, 1] - rel[..., 1] * ax[:, None, 0])
            bins = np.digitize(along, KY) - 1; ok = (perp <= 1.5) & (bins >= 0) & (bins < len(KY) - 1)
            for tag, Smap in (("pA", SP[s]), ("hA", SH[s, C:]), ("pB", CP[s]), ("hB", CH[s])):
                ky = np.full((T, len(KY) - 1), np.nan)
                for t in range(T):
                    v = Smap[:, t]; acc = np.zeros(len(KY) - 1); cn = np.zeros(len(KY) - 1)
                    np.add.at(acc, bins[ok], np.nan_to_num(v[ok])); np.add.at(cn, bins[ok], 1)
                    ky[t] = np.where(cn > 0, acc / np.maximum(cn, 1), np.nan)
                d[f"ky_{tag}"] = ky.tolist()
            d["true_prog"] = [float(np.mean(((XYt[s, C + t] - lastxy[s]) * ax).sum(-1))) for t in range(T)]
            groups[f"{o}/{m_}"] = d
    res["groups"] = groups; res["cats"] = CATS; res["ky_edges"] = KY.tolist()
    od = REPO / f"z_analysis/ar_token_where/{name}"; od.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(od / "where.json", "w"), indent=1)
    torch.save(dict(head=head.state_dict(), mu=mu.cpu(), sd=sd.cpu()), od / "objness.pt")
    print(json.dumps({k: v for k, v in res.items() if k not in ("groups", "cats", "ky_edges")}, indent=1))
    for k, d in groups.items():
        print(f"\n{k} (n {d['n']})   범주 = " + " / ".join(CATS))
        for tag in ("pA", "pB", "hA"):
            print(f"  {tag} " + "  ".join(f"t{t}:" + ",".join(f"{x * 100:3.0f}" for x in d[f'{tag}_frac'][t]) for t in (0, 2, 4, 7))
                  + "  | 넘는 토큰 수 " + " ".join(f"{x:4.1f}" for x in d[f"{tag}_n_over"]))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok_dir", required=True)
    ap.add_argument("--name", default="vast70")
    a = ap.parse_args()
    main(a.name, a.tok_dir)
