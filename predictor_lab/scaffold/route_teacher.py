#!/usr/bin/env python3
"""경로 선생 진단 2/2 (CPU) — 목표 경로 A* 의 정확도 (v3 · v3t · VOBench) 와 릴리즈 · PFT attention 의 일치도 (2026-11-01).

경로 선생 (학습 때만 쓸 모듈 후보. 돌리기 전에 고정한 정의)
  ① 물체다움 o_t(j) = σ((w·LN(z_t(j)) + b − τ) / κ)   — 1 판 0 단계의 물체다움 선형 (BlenderTrack v1 학습, 1,281 수) · τ · κ 그대로.
     문턱을 정할 때 학습 없는 신호 (대응 왕복 오차 · 봉우리) 는 물체 / 배경을 AUC 0.66–0.76 으로만 갈랐고 이 선형은 0.995 (BlenderTrack v1 검증, 11-01).
  ② 속도 v_t(j) = 대응 변위 / 칸 간격 의 중앙값 — 앞으로 (t → t+k) 와 뒤로 (t → t−k, 부호 반대), k = 1–3, 창 반경 3, 포물선 보간 (학습 없음).
  ③ 착지 y_{t,j}(τ) = 자리(j) + v_t(j) · (τ − t)        — 문맥 칸 t (0–7) 의 토큰이 미래 칸 τ (8–15) 에 있을 자리 (등속).
  ④ 목표 경로 A*(τ, r)(t, j) ∝ o_t(j) · exp(−‖r − y_{t,j}(τ)‖² / 2σ²),  σ = 1 토큰.  미래 칸 τ 의 자리 r 에 올 물체는 문맥의 어느 토큰인가.
정확도 (라벨로, 평가 전용)
  속도 읽기: 마지막으로 보인 칸 쪽 (보이는 비율 ≥ 0.9) 에서 target 중심의 속도 vs 라벨 (가려진 부분 포함 마스크 무게중심) 속도.
  착지: target 토큰 (보이는 점유 ≥ 0.25, 그 칸 보이는 비율 ≥ 0.5) 의 o 무게 착지 평균 vs 참 중심 (미래 칸별 오차, 토큰). 대조 = 마지막 보인 자리에 머무름 · 라벨 속도 등속.
  경로 정밀도: A*(τ, 참 자리) 가 target 토큰에 준 질량 (나머지는 배경 · 다른 물체).
일치도 (모델 attention, route_extract.py 가 기록): 미래 칸 τ 의 참 자리 3×3 query 의 문맥 key 분포 Â (문맥 질량으로 정규화, 층 3–11 평균)
  겹침 = Σ min(Â, A*) (0–1, 균등 Â 의 겹침과 같이) · target 토큰 질량 ÷ 균등 몫 (경로 세기) — 모델과 A* 를 같은 자로.
  VOBench 1/3 · 2/3 배: 같은 query 를 1 배 판 참 자리 (= PFT 가 외운 시간표 자리) 에서도 — 어느 쪽에서 경로가 켜지나.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P predictor_lab/scaffold/route_teacher.py      # → predictor_lab/scaffold/results/route_teacher.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.vjepa_frozen import scaffold as SC  # noqa: E402
import adapter_step1_velocity as A1  # noqa: E402

RT = Path("/data/hyuntak/project/2026/2027_cvpr/nfs_cache/route_teacher")
STAGE0 = Path("/data/hyuntak/project/2026/2027_cvpr/nfs_cache/scaffold_stage0")
RES = REPO / "predictor_lab/scaffold/results"
S, G, C, K = 256, 16, 8, 8
SIG = 1.0
BAND = list(range(3, 12))
XY = SC.token_coords().numpy()                                   # (256, 2) (열, 행)
_st = json.load(open(RES / "stage0.json"))["read"]
TAU, KAP = _st["tau"], _st["kappa"]
_sd = torch.load(RES / "object_score_w.pt", map_location="cpu")
OBJ = SC.ObjectScore(_sd["lin.weight"].shape[1]); OBJ.load_state_dict(_sd)


def teacher(z: np.ndarray):
    """z (8, 256, D) → 물체다움 o (8, 256) · 속도 v (8, 256, 2)."""
    zt = torch.from_numpy(z).float()
    with torch.no_grad():
        o = torch.sigmoid((OBJ(zt) - TAU) / KAP).numpy()
    est = [[[] for _ in range(S)] for _ in range(C)]
    acc = np.zeros((C, S, 2)); cnt = np.zeros((C, S))
    vs = {t: [] for t in range(C)}
    for k in (1, 2, 3):
        d, _ = A1.flow(zt, k)                                      # 앞으로: 칸 t → t+k (t = 0..7−k)
        d = d.reshape(C - k, S, 2).numpy() / k
        for t in range(C - k):
            vs[t].append(d[t])
        db, _ = A1.flow(torch.flip(zt, [0]), k)                    # 뒤로: 뒤집은 순서에서 칸 (7−t) → (7−t−k) = 원래 t → t−k
        db = db.reshape(C - k, S, 2).numpy() / k
        for tt in range(C - k):
            t = C - 1 - tt
            vs[t].append(-db[tt])
    v = np.stack([np.median(np.stack(vs[t]), 0) for t in range(C)])  # (8, 256, 2)
    return o, v


def landing(o, v, tau):
    """미래 칸 tau 에서 각 문맥 토큰의 착지 (8, 256, 2)."""
    t = np.arange(C)[:, None, None]
    return XY[None] + v * (tau - t)


def astar(o, v, tau, r):
    """A*(tau, r) (8·256,) 정규화 · 정규화 전 합."""
    y = landing(o, v, tau)
    w = o * np.exp(-((y - np.asarray(r)[None, None]) ** 2).sum(-1) / (2 * SIG ** 2))
    s = w.sum()
    return (w / s).ravel() if s > 0 else np.full(C * S, 1.0 / (C * S)), float(s)


def bilinear(field, xy):
    """field (256, C) 또는 (16, 16, C) 를 토큰 좌표 xy (열, 행) 에서 이중선형으로."""
    return A1.bilinear(np.asarray(field).reshape(G, G, -1), xy)


def box_tokens(c):
    """VOBench: 중심 c (열, 행) 3×3 토큰 마스크 (256,)."""
    m = np.zeros((G, G), bool)
    x, y = int(np.clip(np.round(c[0]), 1, G - 2)), int(np.clip(np.round(c[1]), 1, G - 2))
    m[y - 1:y + 2, x - 1:x + 2] = True
    return m.ravel()


def clip_record(z, ctr, vis, tok_target, tok_other, att=None, att_tt=None, ctr_tt=None):
    """한 영상의 정확도 · 일치도. ctr (16, 2) target 중심 · vis (16,) · tok_target (8, 256) bool · tok_other (8, 256) bool.
    att: {model: (K, L, 2048)} 참 자리 query · att_tt: 1 배 시간표 자리 query (VOBench 느린 판)."""
    o, v = teacher(z)
    vis_c = np.nonzero(vis[:C] >= 0.9)[0]
    rec = {}
    if len(vis_c) >= 2:
        lv = int(vis_c.max())
        ts = [t for t in vis_c if t >= lv - 4]
        vt = np.polyfit(ts, ctr[ts, 0], 1)[0]
        vh = np.median([bilinear(v[t], ctr[t])[0] for t in ts])
        rec["v_read_ratio"] = float(vh / vt) if abs(vt) > 0.05 else np.nan
        rec["v_true"] = float(vt)
    else:
        lv = None
    # 착지 (target 토큰 o 무게)
    w = o * tok_target
    land_err, stay_err, cv_err, prec, cover, mass_other, ovl = [], [], [], [], [], [], {}
    for r in range(K):
        tau = C + r
        if not np.isfinite(ctr[tau, 0]):
            continue
        y = landing(o, v, tau)
        if w.sum() > 0:
            L_ = (w[..., None] * y).sum((0, 1)) / w.sum()
            land_err.append(float(np.linalg.norm(L_ - ctr[tau])))
        if lv is not None:
            stay_err.append(float(np.linalg.norm(ctr[lv] - ctr[tau])))
            ts = [t for t in vis_c if t >= lv - 4]
            vv = np.polyfit(ts, ctr[ts], 1)[0]
            cv_err.append(float(np.linalg.norm(ctr[lv] + vv * (tau - lv) - ctr[tau])))
        As, s = astar(o, v, tau, ctr[tau])
        cover.append(s >= 0.5)
        prec.append(float(As[tok_target.ravel()].sum()))
        mass_other.append(float(As[tok_other.ravel()].sum()))
        for name, src in (att or {}).items():
            Ah = src[r][BAND].astype(np.float32).mean(0); Ah = Ah / Ah.sum()
            ovl.setdefault(name, []).append((float(np.minimum(Ah, As).sum()), float(np.minimum(1.0 / (C * S), As).sum()),
                                             float(Ah[tok_target.ravel()].sum() / max(tok_target.sum(), 1) * C * S),
                                             float(As[tok_target.ravel()].sum() / max(tok_target.sum(), 1) * C * S), bool(vis[tau] >= 0.5)))
        for name, src in (att_tt or {}).items():
            Ah = src[r][BAND].astype(np.float32).mean(0); Ah = Ah / Ah.sum()
            ovl.setdefault(name + "@timetable", []).append((float(np.nan), float(np.nan),
                                                            float(Ah[tok_target.ravel()].sum() / max(tok_target.sum(), 1) * C * S), float(np.nan), bool(vis[tau] >= 0.5)))
    rec.update(land_err=land_err, stay_err=stay_err, cv_err=cv_err, prec=prec, cover=cover, mass_other=mass_other, ovl=ovl)
    return rec


def summarize(recs):
    out = {"n_clip": len(recs)}
    vr = np.array([r.get("v_read_ratio", np.nan) for r in recs], float)
    out["v_read_ratio_median"] = round(float(np.nanmedian(vr)), 3)
    out["v_read_off50"] = int(np.nansum(np.abs(vr - 1) > 0.5)); out["v_read_n"] = int(np.isfinite(vr).sum())
    for k_ in ("land_err", "stay_err", "cv_err"):
        a = np.concatenate([np.array(r[k_]) for r in recs if r[k_]]) if any(r[k_] for r in recs) else np.array([np.nan])
        out[k_ + "_median"] = round(float(np.nanmedian(a)), 3)
        out[k_ + "_within1"] = round(float(np.nanmean(a <= 1.0)), 3)
    # 미래 칸별 착지 오차 (앞 · 뒤)
    le = [r["land_err"] for r in recs if len(r["land_err"]) == K]
    if le:
        out["land_err_by_step_median"] = [round(float(x), 2) for x in np.median(np.array(le), 0)]
        se = [r["stay_err"] for r in recs if len(r["stay_err"]) == K]
        out["stay_err_by_step_median"] = [round(float(x), 2) for x in np.median(np.array(se), 0)] if se else None
    pr = np.concatenate([r["prec"] for r in recs]); mo = np.concatenate([r["mass_other"] for r in recs]); cv = np.concatenate([r["cover"] for r in recs])
    out["astar_target_mass_median"] = round(float(np.median(pr)), 3); out["astar_target_mass_ge0.5"] = round(float((pr >= 0.5).mean()), 3)
    out["astar_other_object_mass_median"] = round(float(np.median(mo)), 3); out["astar_defined"] = round(float(cv.mean()), 3)
    models = sorted({m for r in recs for m in r["ovl"]})
    out["alignment"] = {}
    for m in models:
        rows = np.array([x for r in recs for x in r["ovl"].get(m, [])], float)
        if not len(rows):
            continue
        d = {"n": int(len(rows))}
        for nm, sel in (("all", np.ones(len(rows), bool)), ("reappeared", rows[:, 4] > 0), ("hidden", rows[:, 4] == 0)):
            if not sel.any():
                continue
            d[nm] = {"overlap_median": None if np.isnan(rows[sel, 0]).all() else round(float(np.nanmedian(rows[sel, 0])), 3),
                     "overlap_uniform_median": None if np.isnan(rows[sel, 1]).all() else round(float(np.nanmedian(rows[sel, 1])), 3),
                     "model_path_strength_median": round(float(np.nanmedian(rows[sel, 2])), 2),
                     "teacher_path_strength_median": None if np.isnan(rows[sel, 3]).all() else round(float(np.nanmedian(rows[sel, 3])), 2),
                     "n": int(sel.sum())}
        out["alignment"][m] = d
    return out


def run_vob():
    res = {}
    for nm, lab in (("S1", "1/3"), ("T1", "2/3"), ("orig", "1")):
        z0 = np.load(STAGE0 / f"vob_{nm}.npz")
        f = RT / f"vob_{nm}.npz"
        d = np.load(f) if f.exists() else None
        recs = []
        for i in range(len(z0["z"])):
            cen, vis = z0["cen"][i], z0["vis"][i]
            tt = np.stack([box_tokens(cen[t]) & (vis[t] >= 0.5) for t in range(C)])
            att = {m: d[f"att_{m}"][i, 0] for m in ("release", "vobftg")} if d is not None else None
            att_tt = {m: d[f"att_{m}"][i, 1] for m in ("release", "vobftg")} if (d is not None and d[f"att_release"].shape[1] > 1) else None
            recs.append(clip_record(z0["z"][i].astype(np.float32), cen, vis, tt, np.zeros_like(tt), att, att_tt))
        res[lab] = summarize(recs)
        print(f"[vob {lab}]", json.dumps(res[lab], ensure_ascii=False), flush=True)
    return res


def run_bt(name):
    f = RT / f"{name}.npz"
    if not f.exists():
        print(f"!! 없음 {f}"); return None
    d = np.load(f, allow_pickle=True)
    recs, metas = [], []
    for i in range(len(d["z"])):
        tk = int(np.argmax(d["tgt"][i]))
        occ = d["occ"][i].astype(np.float32)                     # (16, KMAX, 256)
        vis = d["vis"][i]
        tt = np.stack([(occ[t, tk] >= 0.25) & (vis[t, tk] >= 0.5) for t in range(C)])
        ot = np.stack([np.any([(occ[t, j] >= 0.25) for j in range(occ.shape[1]) if j != tk], 0) if occ.shape[1] > 1 else np.zeros(S, bool) for t in range(C)])
        att = {m: d[f"att_{m}"][i, 0] for m in ("release", "vobftg")}
        recs.append(clip_record(d["z"][i].astype(np.float32), d["ctr"][i][:, tk], vis[:, tk], tt, ot, att))
        metas.append(dict(d["meta"][i]))
    out = {"all": summarize(recs)}
    for key, fn in (("flat", lambda m: m["place"] == "flat"), ("ramp_up", lambda m: m["ramp_mode"] == "up"), ("ramp_down", lambda m: m["ramp_mode"] == "down"),
                    ("flat_ball", lambda m: m["place"] == "flat" and m["target_kind"] in ("sphere", "basketball", "soccer_ball", "beach_ball")),
                    ("flat_slide", lambda m: m["place"] == "flat" and m["target_kind"] not in ("sphere", "basketball", "soccer_ball", "beach_ball"))):
        sel = [r for r, m in zip(recs, metas) if fn(m)]
        if sel:
            out[key] = summarize(sel)
    print(f"[{name}]", json.dumps(out["all"], ensure_ascii=False), flush=True)
    return out


def main():
    res = {"definition": {"sigma_tok": SIG, "band_layers": BAND, "objectness": "stage0 object_score_w (BlenderTrack v1)", "tau": TAU, "kappa": KAP,
                          "velocity": "feature correspondence k=1-3 fwd+bwd median, window radius 3"}}
    res["vobench_speed_sweep"] = run_vob()
    for nm in ("v3", "v3t"):
        res[nm] = run_bt(nm)
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RES / "route_teacher.json", "w"), indent=1, ensure_ascii=False)
    print("[done]", RES / "route_teacher.json")


if __name__ == "__main__":
    main()
