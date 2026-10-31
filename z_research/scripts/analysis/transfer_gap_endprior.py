#!/usr/bin/env python3
"""transfer_gap Q3 · E35 — 끝 자리 사전 확인 (사전 등록 시험). 2026-10-30, [X2] R3.

왜: R2 검토가 E32 를 사후 검사로 뒤집었다 — "PFT 는 방향은 문맥을 따르지만 다시 나타나는 물체를 학습셋의 전형 끝 자리에 둔다"
    (보폭 2 판 10 block). 이것을 미리 정한 판정 규칙으로 다른 변환 판 (잘라 옮기기 · 역재생 · 창 이동) 에서 시험한다.
정의 (학습 없음, E32 와 같은 자)
  존재 대비 지도: 같은 block 두 가능 영상 (pos_a 물체 · pos_b 빈 장면, 또는 v11 / VOBench train 의 모양 · 색 block 은 서로 다른 물체 둘)
    E^h_{t,c} = ‖h_a − h_b‖₂ (t = 0..nc+nf−1, 칸 c = 16×16) · E^k_{t,c} = ‖p^k_a − p^k_b‖₂ (미래 튜블릿, predictor k).
  자리 loc(E) = 3×3 상자 합이 최대인 상자의 중심 칸 (행, 열) — 중심은 1..14 (E32 와 같다). 참 자리 = 같은 튜블릿 h 의 자리.
  방향 d = sign(중앙 열(h, 문맥 튜블릿 3–5) − 중앙 열(h, 0–2)) (R2 track_prior 와 같다; late 는 이 튜블릿이 다 보인다).
  전형 끝 자리 = VOBench split v2 train late 이동 block 의 h t15 자리 방향별 중앙값 (주) · test 판 (R2 대용) 도 낸다.
  변환 좌표 (T3, 448 px 창 (x0, y0) → 256): 칸 c → ((c + 0.5)·32 − x0)/28 − 0.5 (행은 y0). 나머지 변환은 같은 좌표.
  (i)   잘라 옮기기 따라감 비 β = Σ(예측 열 차) / Σ(참 열 차), 같은 block 에서 x0 (또는 y0) 만 64 px 다른 두 창 (기하 −2.29 칸).
  (ii)  보폭 2 끌림 비 λ = (예측 열 − 보폭 2 참 열) / (보폭 3 참 열 − 보폭 2 참 열), t15, 두 참 열 차 ≥ 1.5 칸 block 의 중앙값.
  (iii) T5 시점 끌림 비 = (예측 − T5 참) / (원판 같은 튜블릿 참 − T5 참), 마지막 미래 튜블릿 (t13), 차 ≥ 1 칸.
  1 칸 규칙 표: |예측 열 − 참 열| ≤ 1 · |예측 열 − 전형 끝 열 (변환 좌표)| ≤ 1 · 방향 + 전형 끝 자리 기준선 (같은 세트 다른 block 한 개 빼기).
세트 (test late · 이동 · permanence): orig · T1 (보폭 2, raw 15–77) · T3_{x0y0,x64y0,x0y64,x64y64} · T2 (역재생) · T5 (창 이동, c16t28)
       h 만: vobtrain (split v2 train 이동 block 전부) · v11train (v11 split train 이동 4 조건 × 모양 위반, 672 block)
모델: release · vobpft (VOBench + gravity PFT) · objho (처음 보는 물체 PFT) · v11pft (v11 PFT e10)
출력: 배열 /data2/local_datasets/world/world_analysis/cache/transfer_gap/E35/<set>.npz (+ _meta.json) ·
      결과 z_analysis/transfer_gap/results/q3/E35_endprior.json

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/transfer_gap_endprior.py extract --sets orig,T1,T2,T3_x0y0,T3_x64y0,T3_x0y64,T3_x64y64,T5,vobtrain,v11train   (GPU 1, 약 20 분)
  $P z_research/scripts/analysis/transfer_gap_endprior.py extract --sets orig --limit 2      (배관 확인)
  $P z_research/scripts/analysis/transfer_gap_endprior.py report
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

TG = Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap")
ARR = TG / "E35"
OUT = REPO / "z_analysis/transfer_gap/results/q3/E35_endprior.json"
R = REPO / "z_training/runs"
PREDS = [("release", str(R / "release_vith/latest.pt")), ("vobpft", str(R / "vobench_release_postft_g/latest.pt")),
         ("objho", str(R / "vobench_release_postft_objholdout/latest.pt")), ("v11pft", str(R / "v11_postft/e10.pt"))]
VOBX = str(REPO / "data_csv/transfer_gap_vob")
SETS = {
    "orig": ("vobench_v1_split2_test", []),
    "T1": ("vobench_v1_split2_test", ["data.frames_start=15", "data.frames_stride=2"]),
    "T2": ("vobench_v1_split2_test", [f"data.root={VOBX}", "data.index_csv=index_T2.csv", f"data.frames_root={TG}/frames_T2"]),
    **{f"T3_{w}": ("vobench_v1_split2_test", [f"data.root={VOBX}", "data.index_csv=index_T3.csv", f"data.frames_root={TG}/frames_T3_{w}"])
       for w in ("x0y0", "x64y0", "x0y64", "x64y64")},
    "T5": ("vobench_v1_asymmetry", [f"data.root={VOBX}", "data.index_csv=index_T5.csv", "window.n_frames=28"]),
    "vobtrain": ("vobench_v1_split2_test", ["data.index_csv=index_train.csv"]),
    "v11train": ("v11_split_test", ["data.index_csv=index_train.csv"]),
}
H_ONLY = ("vobtrain", "v11train")
CROP = {"T3_x0y0": (0, 0), "T3_x64y0": (64, 0), "T3_x0y64": (0, 64), "T3_x64y64": (64, 64)}
V11_CONDS = ("moving_occlusion", "moving_occlusion_flat", "moving_visible", "moving_visible_flat")
NB = 2000


def pick_blocks(W, name):
    by = defaultdict(dict)
    for i, r in enumerate(W.records):
        by[r.block_id][r.raw.get("variant")] = i
    out = []
    for b, v in by.items():
        if "pos_a" not in v or "pos_b" not in v:
            continue
        raw = W.records[v["pos_a"]].raw
        if name == "v11train":
            if raw.get("condition") not in V11_CONDS or raw.get("violation_type") != "shape":
                continue
            k = int(raw.get("n_context_hidden") or 0) if raw["condition"].startswith("moving_occlusion") else 0
            out.append({"block": b, "a": v["pos_a"], "b": v["pos_b"], "occ": "late" if k > 0 else "visible", "k": str(k),
                        "motion": raw.get("condition"), "objset": "v11", "bg": raw.get("env"), "dir_meta": raw.get("direction")})
            continue
        if raw.get("motion") == "flat_static":
            continue
        if name != "vobtrain" and (raw.get("violation") != "permanence" or raw.get("occlusion") != "late"):
            continue
        out.append({"block": b, "a": v["pos_a"], "b": v["pos_b"], "occ": raw.get("occlusion"), "k": raw.get("k"),
                    "motion": raw.get("motion"), "objset": raw.get("objset"), "bg": raw.get("background"),
                    "violation": raw.get("violation"), "source_block": raw.get("source_block"), "file_name": raw.get("file_name")})
    return sorted(out, key=lambda x: int(x["block"]) if str(x["block"]).isdigit() else str(x["block"]))


def extract(sets, limit=0):
    import torch
    import te_v11_score as te
    from transfer_gap_te import resolve
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    ARR.mkdir(parents=True, exist_ok=True)
    cfg0, _ = resolve("vobench_v1_split2_test", "vith", [], None, 0)
    sc = te.Scorer(cfg0, dev, PREDS)
    tags = sc.tags
    D, S = sc.D, sc.S
    for name in sets:
        ds, st = SETS[name]
        cfg, _ = resolve(ds, "vith", st, None, 0)
        N, C = int(cfg["data"]["n_frames"]), int(cfg["surprise"]["context_length"])
        sc.ctx, sc.tgt = C, N - C
        sc.nc, sc.nf = C // 2, (N - C) // 2
        nc, nf = sc.nc, sc.nf
        W = te.WMADataset(cfg)
        blocks = pick_blocks(W, name)
        if limit:
            blocks = blocks[:limit]
        n = len(blocks)
        Eh = np.zeros((n, nc + nf, S), np.float32)
        Ep = {t: np.zeros((n, nf, S), np.float32) for t in tags} if name not in H_ONLY else {}
        t0 = time.time()
        bs = 4
        for b0 in range(0, n, bs):
            bl = blocks[b0:b0 + bs]
            clips = torch.stack([W.clip(x[y]) for x in bl for y in ("a", "b")])
            with torch.inference_mode():
                z, h, ci, ti, _ = sc.encode(clips)
                B = z.size(0)
                dn = lambda x: (x[0::2] - x[1::2]).norm(dim=-1).cpu().numpy()  # noqa: E731
                Eh[b0:b0 + len(bl)] = dn(h.float().reshape(B, nc + nf, S, D))
                for k, t in enumerate(tags if name not in H_ONLY else []):
                    p = sc.predict(sc.mods[k], z, ci, ti).float().reshape(B, nf, S, D)
                    Ep[t][b0:b0 + len(bl)] = dn(p)
            if (b0 // bs) % 25 == 0:
                print(f"[{name}] {b0 + len(bl)}/{n} {time.time() - t0:.0f}s", flush=True)
        meta = [{k: v for k, v in x.items() if k not in ("a", "b")} | {"vid_a": W.records[x["a"]].video_id, "vid_b": W.records[x["b"]].video_id}
                for x in blocks]
        sfx = f"_lim{limit}" if limit else ""
        np.savez(ARR / f"{name}{sfx}.npz", Eh=Eh, **{f"E_{t}": v for t, v in Ep.items()})
        json.dump({"blocks": meta, "nc": nc, "nf": nf, "tags": tags if name not in H_ONLY else [], "dataset": ds, "sets": st},
                  open(ARR / f"{name}{sfx}_meta.json", "w"), ensure_ascii=False)
        print(f"[{name}] done {n} blocks nc={nc} nf={nf} {time.time() - t0:.0f}s", flush=True)
    print("E35_EXTRACT_DONE", flush=True)


# ----------------------------------------------------------------------------- report (CPU)
def loc(E):
    """E (..., 256) → (행, 열, 몰림, 두드러짐 z). 3×3 상자 합 최대 (중심 1..14)."""
    G = E.reshape(*E.shape[:-1], 16, 16)
    box = sum(G[..., i:i + 14, j:j + 14] for i in range(3) for j in range(3))
    flat = box.reshape(*box.shape[:-2], -1)
    k = flat.argmax(-1)
    share = flat.max(-1) / np.maximum(E.sum(-1), 1e-9)
    zz = (flat.max(-1) - flat.mean(-1)) / np.maximum(flat.std(-1), 1e-9)
    return k // 14 + 1, k % 14 + 1, share, zz


def load(name):
    f = ARR / f"{name}.npz"
    if not f.exists():
        return None
    Z = np.load(f); M = json.load(open(ARR / f"{name}_meta.json"))
    return Z, M


def direction(ch):
    return np.sign(np.median(ch[:, 3:6], 1) - np.median(ch[:, 0:3], 1)).astype(int)


def mapc(c, off):
    return ((np.asarray(c, float) + 0.5) * 32 - off) / 28 - 0.5


def boot_ci(fn, n, nb=NB, seed=0):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(nb):
        ix = rng.integers(0, n, n)
        x = fn(ix)
        if x is not None and np.isfinite(x):
            v.append(x)
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if len(v) > 50 else None


def typical(Z, M, t=15, occ_keep=("late",)):
    rh, ch, _, _ = loc(Z["Eh"])
    # ⚠️ 정정 (첫 report 뒤): 모양 · 색 block (서로 다른 물체 둘) 의 대비 지도는 물체가 아니라 화면 맨 위 줄 (행 1–2) 에 정점이 선다
    #    (VOBench train 모양 · 색 120 block 중 행 1–2 가 약 80 %, v11 train 모양 block 도 같음) → 물체 · 빈 장면 대비 (permanence) 만 쓴다
    ix = np.array([i for i, m in enumerate(M["blocks"]) if m["occ"] in occ_keep and m.get("violation", "permanence") == "permanence"])
    d = direction(ch[ix])
    res = {}
    for s in (-1, 1):
        j = ix[d == s]
        if len(j):
            res[s] = {"row": float(np.median(rh[j, t])), "col": float(np.median(ch[j, t])), "n": int(len(j)),
                      "col_hist": Counter(ch[j, t].tolist()).most_common(5)}
    return res


def report():
    res = {"definition": __doc__.split("정의 (학습 없음, E32 와 같은 자)")[1].split("세트 (test")[0].strip(), "sets": {}}
    data = {n: load(n) for n in SETS}
    # 전형 끝 자리 (주 = train late 이동, 대용 = test 원판)
    typ_train = typical(*data["vobtrain"], 15) if data["vobtrain"] else None
    typ_train_all = typical(*data["vobtrain"], 15, ("late", "early", "mid", "visible")) if data["vobtrain"] else None
    typ_test = typical(*data["orig"], 15) if data["orig"] else None
    typ_train_t13 = typical(*data["vobtrain"], 13) if data["vobtrain"] else None
    res["typical_end"] = {"train_late_t15": typ_train, "train_all_moving_t15": typ_train_all, "test_orig_late_t15": typ_test,
                          "train_late_t13": typ_train_t13}
    TYP = {s: (v["row"], v["col"]) for s, v in (typ_train or typ_test).items()}
    print("typical end (train late t15):", TYP, flush=True)
    cache = {}
    for name in [n for n in SETS if n not in H_ONLY]:
        if data[name] is None:
            continue
        Z, M = data[name]
        nf, nc = M["nf"], M["nc"]
        tl = nc + nf - 1                          # 마지막 미래 튜블릿 (t15, T5 는 t13)
        rh, ch, sh, zh = loc(Z["Eh"])
        d = direction(ch)
        Tc, Tr = ch[:, tl], rh[:, tl]
        x0, y0 = CROP.get(name, (0, 0))
        # 전형 끝 열 (변환 좌표). ⚠️ 정정 (첫 report 뒤): 448 창 변환은 T3 에만 건다 (처음엔 x0 = 0 인 비-T3 세트에도 축척을 걸었다)
        pc = np.array([(mapc(TYP[s][1], x0) if name in CROP else TYP[s][1]) if s in TYP else np.nan for s in d])
        pcS = np.array([TYP[s][1] if s in TYP else np.nan for s in d])                  # 화면 좌표 그대로
        n = len(d)
        base = []
        for j in range(n):                          # 방향 + 전형 끝 자리 기준선 (같은 세트, 한 개 빼기)
            s = d == d[j]; s[j] = False
            base.append(bool(s.any() and abs(np.median(Tc[s]) - Tc[j]) <= 1))
        o = {"n": n, "last_future_tubelet": int(tl), "dir_counts": dict(Counter(d.tolist())),
             "true_col_hist": Counter(Tc.tolist()).most_common(6), "prior_col_mapped_unique": sorted({round(float(x), 2) for x in pc}),
             "true_minus_prior_col_abs_median": round(float(np.median(np.abs(Tc - pc))), 2),
             "n_true_ge2_from_prior": int((np.abs(Tc - pc) >= 2).sum()),
             "baseline_dir_typical_within1col": round(float(np.mean(base)), 3), "h_share_median": round(float(np.median(sh[:, tl])), 3),
             "h_z_median": round(float(np.median(zh[:, tl])), 2)}
        cache[name] = {"Tc": Tc, "Tr": Tr, "d": d, "blocks": [m["block"] for m in M["blocks"]], "pred": {}, "rh": rh, "ch": ch, "tl": tl}
        for t in M["tags"]:
            r_, c_, s_, z_ = loc(Z[f"E_{t}"])
            P = c_[:, nf - 1]
            cache[name]["pred"][t] = (r_, c_)
            wt = np.abs(P - Tc) <= 1; wp = np.abs(P - pc) <= 1; wS = np.abs(P - pcS) <= 1
            dec = np.abs(Tc - pc) >= 2
            verdict = ("구분 불가 (둘 다 ≥ 70 %)" if wt.mean() >= 0.7 and wp.mean() >= 0.7 else "문맥을 따른다" if wt.mean() >= 0.7
                       else "끝 자리 사전" if wp.mean() >= 0.7 else "판정 불가")
            o[t] = {"valid_near_true_or_prior": round(float((wt | wp).mean()), 3),
                    "within1col_true": round(float(wt.mean()), 3), "within1col_prior_mapped": round(float(wp.mean()), 3),
                    "within1col_prior_screen": round(float(wS.mean()), 3),
                    "within1_2d_true": round(float((np.maximum(np.abs(P - Tc), np.abs(r_[:, nf - 1] - Tr)) <= 1).mean()), 3),
                    "col_err_median": round(float(np.median(P - Tc)), 2), "col_err_mean": round(float(np.mean(P - Tc)), 2),
                    "pred_col_hist": Counter(P.tolist()).most_common(6),
                    "dir_agree_vs_context": round(float(np.mean(np.sign(P - ch[:, nc - 1]) == d)), 3),
                    "share_median": round(float(np.median(s_[:, nf - 1])), 3), "z_median": round(float(np.median(z_[:, nf - 1])), 2),
                    "decisive_n": int(dec.sum()),
                    "decisive_within1col_true": round(float(wt[dec].mean()), 3) if dec.any() else None,
                    "decisive_within1col_prior": round(float(wp[dec].mean()), 3) if dec.any() else None,
                    "literal_verdict": verdict}
        res["sets"][name] = o
        print(name, json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("within1col_true", "within1col_prior_mapped", "col_err_median", "literal_verdict")})
                                for k, v in o.items() if k in ("n", "baseline_dir_typical_within1col", "n_true_ge2_from_prior") or k in M["tags"]}, ensure_ascii=False), flush=True)
    # (i) 잘라 옮기기 따라감 비 β
    beta = {}
    if all(k in cache for k in CROP):
        pairs = {"x": [("T3_x0y0", "T3_x64y0"), ("T3_x0y64", "T3_x64y64")], "y": [("T3_x0y0", "T3_x0y64"), ("T3_x64y0", "T3_x64y64")]}
        bl = cache["T3_x0y0"]["blocks"]
        assert all(cache[k]["blocks"] == bl for k in CROP)
        nB = len(bl)
        for ax, pl in pairs.items():
            key = "ch" if ax == "x" else "rh"
            tl = cache["T3_x0y0"]["tl"]
            dT = np.stack([cache[b][key][:, tl] - cache[a][key][:, tl] for a, b in pl], 1)        # (nB, 2)
            o = {"geometric_delta_cells": round(-64 / 28, 3), "h_delta_mean": round(float(dT.mean()), 3),
                 "beta_h_vs_geometry": round(float(dT.mean() / (-64 / 28)), 3)}
            for t in cache["T3_x0y0"]["pred"]:
                ix_ = 1 if ax == "x" else 0
                nf = cache["T3_x0y0"]["pred"][t][ix_].shape[1]
                dP = np.stack([cache[b]["pred"][t][ix_][:, nf - 1] - cache[a]["pred"][t][ix_][:, nf - 1] for a, b in pl], 1)
                f = lambda ix, dP=dP: float(dP[ix].sum() / dT[ix].sum()) if abs(dT[ix].sum()) > 0 else None  # noqa: E731
                b_ = f(np.arange(nB))
                ci = boot_ci(f, nB)
                v = ("화면 고정 끝 자리 사전 아님 (장면 · 문맥을 따라 옮긴다)" if b_ >= 0.7 and ci and ci[0] > 0.4 else
                     "화면 고정 끝 자리 사전" if b_ <= 0.3 and ci and ci[1] < 0.6 else "판정 불가")
                # 유효성 관문 (첫 report 뒤 추가, 사후): 네 창 모두에서 예측이 참 자리 또는 전형 끝 열 1 칸 안인 block ≥ 50 % 일 때만 판정을 읽는다.
                #   아니면 자리는 물체 없는 지도의 기본값 (화면 가장자리) 이다 — 릴리즈 · v11 PFT.
                valid = min(res["sets"][k][t]["valid_near_true_or_prior"] for k in CROP)
                if valid < 0.5:
                    v = f"해당 없음 (물체 자리 없음 · 기본값, 유효 {valid:.2f})"
                # 다른 미래 튜블릿에서도 (t8, t11)
                other = {}
                for q in range(nf):                         # 미래 튜블릿별 (late 는 앞 k 샘플이 가려져 t8–t9 의 참 자리가 기본값일 수 있다 → h 차와 같이 읽는다)
                    dPq = np.stack([cache[b]["pred"][t][ix_][:, q] - cache[a]["pred"][t][ix_][:, q] for a, b in pl], 1)
                    dTq = np.stack([cache[b][key][:, cache["T3_x0y0"]["tl"] - (nf - 1) + q] - cache[a][key][:, cache["T3_x0y0"]["tl"] - (nf - 1) + q] for a, b in pl], 1)
                    other[f"t{8 + q}"] = {"beta": round(float(dPq.sum() / dTq.sum()), 3) if abs(dTq.sum()) > 0 else None,
                                          "h_delta_mean": round(float(dTq.mean()), 2), "pred_delta_mean": round(float(dPq.mean()), 2)}
                o[t] = {"beta": round(b_, 3), "ci95": ci, "pred_delta_mean": round(float(dP.mean()), 3), "verdict": v, "beta_other_tubelets": other,
                        "per_pair_beta": [round(float(dP[:, i].sum() / dT[:, i].sum()), 3) for i in range(dP.shape[1])]}
            beta[ax] = o
        # block 별 t15 열 (참 · 모델) — 문서 표용
        tlc = cache["T3_x0y0"]["tl"]
        res["crop_per_block_t15_col"] = [{"block": b, "dir": int(cache["T3_x0y0"]["d"][i]),
                                          **{k[3:]: {"true": int(cache[k]["ch"][i, tlc]), **{t: int(cache[k]["pred"][t][1][i, -1]) for t in cache[k]["pred"]}}
                                             for k in CROP}} for i, b in enumerate(bl)]
        res["crop_follow_beta"] = beta
        print("beta", json.dumps(beta, ensure_ascii=False)[:1500], flush=True)
    # (ii) 보폭 2 끌림 비 λ
    if "T1" in cache and "orig" in cache:
        A, B = cache["orig"], cache["T1"]
        assert A["blocks"] == B["blocks"]
        T3_, T2_ = A["Tc"], B["Tc"]
        sep = np.abs(T3_ - T2_) >= 1.5
        o = {"n_sep": int(sep.sum()), "n": int(len(sep)), "true_s3_minus_s2_median": round(float(np.median(T3_ - T2_)), 2)}
        for t, (r_, c_) in B["pred"].items():
            P = c_[:, -1]
            lam = (P - T2_) / np.where(sep, T3_ - T2_, np.nan)
            js = np.where(sep)[0]
            f = lambda ix, js=js, lam=lam: float(np.median(lam[js][ix]))  # noqa: E731
            med = float(np.median(lam[sep])) if sep.any() else float("nan")
            ci = boot_ci(f, len(js)) if len(js) > 3 else None
            pooled = float(((P - T2_)[sep]).sum() / ((T3_ - T2_)[sep]).sum()) if sep.any() else None
            v = ("문맥 속도를 따른다" if med <= 0.3 and ci and ci[1] < 0.5 else "보폭 3 끝 자리 사전 (시간 척도에 묶임)" if med >= 0.7 and ci and ci[0] > 0.5 else "부분")
            valid = float(((np.abs(P - T2_) <= 1) | (np.abs(P - T3_) <= 1)).mean())      # 유효성 관문 (사후 추가, β 와 같은 뜻)
            if valid < 0.5:
                v = f"해당 없음 (물체 자리 없음 · 기본값, 유효 {valid:.2f})"
            Po = A["pred"][t][1][:, -1]
            o[t] = {"valid_near_s2_or_s3_true": round(valid, 3), "lambda_median": round(med, 3), "ci95": ci, "lambda_pooled": round(pooled, 3) if pooled is not None else None, "verdict": v,
                    "pred_s2_minus_pred_s3_col_median": round(float(np.median(P - Po)), 2),
                    "within1col_true_s2_sep": round(float((np.abs(P - T2_) <= 1)[sep].mean()), 3) if sep.any() else None,
                    "within1col_true_s3_sep": round(float((np.abs(P - T3_) <= 1)[sep].mean()), 3) if sep.any() else None}
        # 튜블릿별 끌림 비 (모은 비 Σ / Σ, 두 참 열 차 ≥ 1.5 인 block) — 끝 자리에서만 끌리나, 궤적 전체가 보폭 3 시간표를 따르나
        byq = {}
        for q in range(8):
            ta, tb = A["ch"][:, 8 + q], B["ch"][:, 8 + q]
            sq = np.abs(ta - tb) >= 1.5
            row = {"n_sep": int(sq.sum()), "true_s3_minus_s2_mean": round(float((ta - tb).mean()), 2)}
            for t, (r_, c_) in B["pred"].items():
                Pq = c_[:, q]
                row[t] = round(float(((Pq - tb)[sq]).sum() / ((ta - tb)[sq]).sum()), 3) if sq.sum() >= 3 else None
            byq[f"t{8 + q}"] = row
        o["lambda_pooled_by_tubelet"] = byq
        o["per_block_t15_col"] = [{"block": b, "dir": int(A["d"][i]), "true_s3": int(T3_[i]), "true_s2": int(T2_[i]), "sep": bool(sep[i]),
                                   **{t: {"s3": int(A["pred"][t][1][i, -1]), "s2": int(B["pred"][t][1][i, -1])} for t in B["pred"]}}
                                  for i, b in enumerate(A["blocks"])]
        res["stride2_pull_lambda"] = o
        print("lambda", json.dumps(o, ensure_ascii=False), flush=True)
    # (iii) T5 시점 끌림 비
    if "T5" in cache and "orig" in cache:
        A, B = cache["orig"], cache["T5"]
        # T5 block_id 는 같은 원 block 번호를 쓰나? source_block 으로 맞춘다
        Zo, Mo = data["orig"]; Z5, M5 = data["T5"]
        so = {m["source_block"]: i for i, m in enumerate(Mo["blocks"])}
        pairs = [(so.get(m["source_block"]), j) for j, m in enumerate(M5["blocks"])]
        pairs = [(i, j) for i, j in pairs if i is not None]
        io, j5 = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        tl5 = B["tl"]                                                  # 13
        To = A["ch"][io, tl5]; T5c = B["ch"][j5, tl5]
        sep = np.abs(To - T5c) >= 1
        o = {"n_matched": int(len(io)), "n_sep": int(sep.sum()), "true_orig_minus_T5_col_median": round(float(np.median(To - T5c)), 2)}
        for t, (r_, c_) in B["pred"].items():
            P = c_[j5, -1]
            lam = (P - T5c) / np.where(sep, To - T5c, np.nan)
            js = np.where(sep)[0]
            o[t] = {"valid_near_T5_or_orig_true": round(float(((np.abs(P - T5c) <= 1) | (np.abs(P - To) <= 1)).mean()), 3),
                    "lambda_median": round(float(np.median(lam[sep])), 3) if sep.any() else None,
                    "ci95": boot_ci(lambda ix, js=js, lam=lam: float(np.median(lam[js][ix])), len(js)) if len(js) > 3 else None,
                    "within1col_true_T5": round(float((np.abs(P - T5c) <= 1).mean()), 3),
                    "within1col_true_orig_t13": round(float((np.abs(P - To) <= 1).mean()), 3)}
        res["T5_pull_lambda"] = o
        print("T5", json.dumps(o, ensure_ascii=False), flush=True)
    # E35b: v11 PFT 의 끝 자리 (VOBench 원판, 방향별) vs v11 train 방향별 끝 열
    if "orig" in cache and (TG / "E32/v11sub.npz").exists():
        # ⚠️ 정정 (첫 report 뒤): v11 split train 에는 물체 · 빈 장면 block (vanish) 이 없고 모양 block 의 대비 지도는 화면 맨 위에 정점이 선다
        #    (행 1–2 가 약 80 %) → v11 끝 열은 E32 v11sub (v11 split test 의 vanish 가능 영상 둘, 같은 생성기 · block 단위 무작위 split) 로 낸다.
        Zv = np.load(TG / "E32/v11sub.npz"); Mv = {"blocks": json.load(open(TG / "E32/v11sub_meta.json"))}
        rv, cv, sv, zv = loc(Zv["Eh"])
        dv = direction(cv)
        occ = np.array([m["occ"] for m in Mv["blocks"]])
        dm = np.array([m.get("motion") for m in Mv["blocks"]])
        tv = {}
        for s in (-1, 1):
            for oc in ("late", "visible"):
                j = (dv == s) & (occ == oc)
                if j.any():
                    tv[f"dir{s:+d}|{oc}"] = {"n": int(j.sum()), "t15_col_median": float(np.median(cv[j, 15])), "t15_col_hist": Counter(cv[j, 15].tolist()).most_common(5),
                                             "dir_meta": Counter(dm[j].tolist()).most_common(3)}
        o = {"v11_end_by_dir_source": "E32 v11sub (v11 split test vanish, 가능 영상 둘)", "v11_end_by_dir": tv,
             "v11_h_share_median": round(float(np.median(sv[:, 15])), 3)}
        if data["v11train"]:
            Zt, Mt = data["v11train"]
            rt, ct, _, _ = loc(Zt["Eh"])
            o["v11train_shape_contrast_diag"] = {"t15_row_hist": Counter(rt[:, 15].tolist()).most_common(4), "t15_col_hist": Counter(ct[:, 15].tolist()).most_common(4),
                                                 "note": "모양 block 대비 지도 — 정점이 화면 맨 위 줄 (물체 자리 아님) 이라 끝 열 기준으로 안 쓴다"}
        if data["vobtrain"]:
            Zt, Mt = data["vobtrain"]
            rt, ct, _, _ = loc(Zt["Eh"])
            vio = np.array([m["violation"] for m in Mt["blocks"]])
            o["vobtrain_contrast_row_diag"] = {v: Counter(rt[vio == v, 15].tolist()).most_common(3) for v in ("permanence", "shape", "color")}
        A = cache["orig"]
        d = A["d"]
        for t in A["pred"]:
            Zo, Mo = data["orig"]
            r_, c_, s_, z_ = loc(Zo[f"E_{t}"])
            P = c_[:, -1]
            by = {}
            for s in (-1, 1):
                j = d == s
                if not j.any():
                    continue
                key = f"dir{s:+d}|late"
                ref = tv.get(key, {}).get("t15_col_median")
                # v11 PFT 학습셋은 v11 이 전부다 — 같은 방향 v11 끝 열 (late) 기준
                by[f"dir{s:+d}"] = {"n": int(j.sum()), "pred_col_hist": Counter(P[j].tolist()).most_common(6), "pred_col_median": float(np.median(P[j])),
                                    "true_col_median": float(np.median(A["Tc"][j])),
                                    "within1col_v11_typical_end": round(float((np.abs(P[j] - ref) <= 1).mean()), 3) if ref is not None else None,
                                    "within1col_true": round(float((np.abs(P[j] - A["Tc"][j]) <= 1).mean()), 3),
                                    "share_median": round(float(np.median(s_[j, -1])), 3), "z_median": round(float(np.median(z_[j, -1])), 2)}
            meds = [v["pred_col_median"] for v in by.values()]
            okdir = [v["within1col_v11_typical_end"] for v in by.values() if v["within1col_v11_typical_end"] is not None]
            verdict = ("v11 끝 자리 사전 (방향별)" if okdir and min(okdir) >= 0.7 else
                       "방향 무관 자리 (기본값 후보)" if len(meds) == 2 and abs(meds[0] - meds[1]) <= 1 else "판정 불가")
            o[t] = {"by_dir": by, "verdict": verdict}
        res["E35b_v11pft_endprior"] = o
        print("E35b", json.dumps({t: o[t]["verdict"] for t in A["pred"]}, ensure_ascii=False), flush=True)
        # E32b 배열과 같은지 (v11 PFT 를 같은 경로로 다시 뽑았는가)
        f = TG / "E32/v11pft/vob.npz"
        if f.exists():
            Zb = np.load(f); Mb = json.load(open(TG / "E32/v11pft/vob_meta.json"))
            late = [i for i, m in enumerate(Mb) if m["occ"] == "late"]
            vb = [Mb[i]["vid_a"] for i in late]
            va = [m["vid_a"] for m in data["orig"][1]["blocks"]]
            if vb == va:
                cb = loc(Zb["Ef"][late])[1][:, -1]
                res["E35b_v11pft_endprior"]["agree_with_E32b_arrays_col"] = round(float((cb == loc(data["orig"][0]["E_v11pft"])[1][:, -1]).mean()), 3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print("→", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["extract", "report"])
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    extract(a.sets.split(","), a.limit) if a.part == "extract" else report()


if __name__ == "__main__":
    main()
