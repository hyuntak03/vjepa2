#!/usr/bin/env python3
"""scaffold 2 판 0 단계 — (T, 256) 물체 지도 모듈만 학습해 본다 (2026-11-01, 사용자 설계 · 문서 predictor_lab/scaffold/README.md).

무엇 (predictor 없음 · encoder 고정 · 학습 = 지도 모듈 하나, app/vjepa_frozen/scaffold.py MapScaffold)
  입력 = 문맥 encoder 토큰 (8 칸 × 256, 1280) → 출력 = 시간 칸마다 물체 확률 (문맥 8 + 미래 8 칸, 256)
  라벨 = BlenderTrack v3 의 가려진 부분까지 포함한 물체 점유 (모든 물체 합; 64×64 마스크를 토큰 16×16 로, 칸 = 샘플 2 장 평균)
데이터
  학습 = v3 train 가림막 장면 (가리는 물체 공 · 상자는 마스크가 없어 뺌) · 렌더 타이밍 합격 · 물체 있는 판 + 물체 뺀 판 = 10,376 clip
  창 = v3 설계 그대로 64 장 간격 2 → 32 샘플 (문맥 16 = 8 칸 · 미래 16 = 8 칸), 좌우 반전 0.5
  encoder = 릴리즈 ViT-H 문맥 encoder, 채점 하네스 · 1 판 0 단계와 같은 경로 (fp32 + autocast fp16) — VOBench 속도 sweep z 를 그대로 쓰려고
평가 (판정 기준은 돌리기 전에 고정, 2026-11-01 사용자 확인)
  학습 안 = v3 test 가림막 장면 2,598 clip: 미래 칸 물체 중심 오차 < "마지막 자리에 머무르기" 기준선 (물체 수 1 · 2 · 3 별,
           마지막 문맥 자리에서 1 토큰 이상 움직인 물체-칸 — 11-01 수정, 정지 물체는 머무름 오차 0 이라)
  학습 밖 = VOBench 속도 sweep 27 block (1/3 · 2/3 · 1 배): 세 속도 모두 기울기 비 중앙값 0.7–1.3 · 50 % 넘게 틀린 영상 ≤ 3 / 27
  지도 → 물체 중심: 참 물체 중심으로 나눈 Voronoi 구역마다 최고 확률 pk (< 0.1 이면 놓침) · p ≥ 0.5 pk 칸의 확률 무게중심 (물체 하나면 구역 = 전체)

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P predictor_lab/scaffold/stage0_map.py train       # GPU 1 장: 학습 → 평가 (→ predictor_lab/scaffold/results/stage0_map.json)
  $P predictor_lab/scaffold/stage0_map.py eval        # GPU 1 장: 저장한 모듈로 평가만
  STEPS=20 EVAL_LIMIT=48 TAG=smoke $P ... train       # 배관 점검
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
from app.vjepa_frozen import scaffold as SC  # noqa: E402

V3 = Path("/data/hyuntak/project/2026/2027_cvpr/nfs_cache/blender_track/v3")
STAGE0 = Path("/data/hyuntak/project/2026/2027_cvpr/nfs_cache/scaffold_stage0")        # VOBench 속도 sweep z (1 판 0 단계에서 뽑음)
RES = REPO / "predictor_lab/scaffold/results"
MODEL_CFG = REPO / "cvpr/results/eval/testbed/vo_moving_occluder_vith/_resolved.yaml"  # 릴리즈 ViT-H (vith) 병합 config — 모델 절만 쓴다
IDX = np.arange(0, 64, 2)
NC, NF, S, KMAX = 8, 8, 256, 3
STEPS = int(os.environ.get("STEPS", 3000))
BS = int(os.environ.get("BS", 16))
NW = int(os.environ.get("NW", 10))
LR, WD, WARM = 3e-4, 0.05, 200
DEPTH = int(os.environ.get("DEPTH", 4))
TAG = os.environ.get("TAG", "map_d4")
EVAL_LIMIT = int(os.environ.get("EVAL_LIMIT", 0))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
XY = SC.token_coords().numpy()                                             # (256, 2) (열, 행)
GR, GC = np.meshgrid(np.arange(64, dtype=np.float32), np.arange(64, dtype=np.float32), indexing="ij")   # 64×64 마스크 행 · 열


# ----------------------------------------------------------------------------- 데이터
def rows(split: str):
    out = []
    for r in csv.DictReader(open(REPO / f"data_csv/blender_track_v3/{split}_arms.csv")):
        if r["variant"] in ("pos_a", "pos_b") and r["occ_type"] == "barrier" and r["render_timing_ok"] == "1":
            out.append(dict(cid=os.path.basename(os.path.realpath(r["file"]))[:-4], arm=r["variant"], count=int(r["count_a"]),
                            place=r["place"], target_mode=r["target_mode"], occ_motion=r["occ_motion"], camera=r["camera"],
                            objset=r["objset"], block=r["block_id"]))
    return out


class V3Clips(torch.utils.data.Dataset):
    """반환: x (3, 32, 256, 256) · occ (16, 256) 점유 비율 · ctr (16, K, 2) 물체 중심 (열, 행) · pres (16, K) 화면 안 ·
    vis (16, K) 보이는 비율 · tgt (K,) target 인가 · i"""

    def __init__(self, rs, flip: bool):
        self.rs, self.flip = rs, flip

    def __len__(self):
        return len(self.rs)

    def __getitem__(self, i):
        import decord
        r = self.rs[i]
        fr = decord.VideoReader(str(V3 / "clips" / f"{r['cid']}.mp4"), num_threads=1).get_batch(list(map(int, IDX))).asnumpy()
        L = np.load(V3 / "labels" / f"{r['cid']}.npz")
        n = int(L["amodal"].shape[1])
        occ = np.zeros((32, 64, 64), np.float32)
        ctr = np.full((16, KMAX, 2), np.nan, np.float32); pres = np.zeros((16, KMAX), bool)
        vis = np.full((16, KMAX), np.nan, np.float32); tgt = np.zeros(KMAX, bool)
        if n:
            am = np.unpackbits(L["amodal"][IDX], axis=-1).astype(np.float32)                  # (32, n, 64, 64)
            occ = am.max(1)                                                                   # 모든 물체 합
            k = min(n, KMAX)
            # 물체 중심 = 가려진 부분 포함 마스크의 무게중심 (토큰 좌표: 마스크 칸 c → c/4 − 0.375). 투영 중심 (center_px) 은
            # 마스크 무게중심과 행 방향으로 ~0.7 토큰 어긋나 (CPU 점검 11-01) 지도 모델에만 불리하므로 쓰지 않는다
            mk = am[:, :k]; ms = mk.sum((2, 3))
            with np.errstate(invalid="ignore", divide="ignore"):
                cx = (mk * GC[None, None]).sum((2, 3)) / ms; cy = (mk * GR[None, None]).sum((2, 3)) / ms
            c = np.stack([cx, cy], -1) / 4.0 - 0.375                                          # (32, k, 2) 토큰 (열, 행)
            an = L["amodal_n"][IDX, :k].astype(np.float32); vn = L["visible_n"][IDX, :k].astype(np.float32)
            ctr[:, :k] = c.reshape(16, 2, k, 2).mean(1)
            pres[:, :k] = (an > 0).reshape(16, 2, k).all(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                vis[:, :k] = np.where(an > 0, vn / np.maximum(an, 1), np.nan).reshape(16, 2, k).mean(1)
            tgt[:k] = [str(x) == "target" for x in L["roles"][:k]]
        occ = occ.reshape(16, 2, 16, 4, 16, 4).mean((1, 3, 5))                                # (16, 16, 16) 칸 · 행 · 열
        x = (torch.from_numpy(fr).permute(3, 0, 1, 2).float() / 255.0 - MEAN) / STD
        if self.flip and random.random() < 0.5:
            x = x.flip(-1); occ = occ[..., ::-1]; ctr[..., 0] = 15.0 - ctr[..., 0]
        return x, torch.from_numpy(np.ascontiguousarray(occ).reshape(16, S)), ctr, pres, vis, tgt, i


# ----------------------------------------------------------------------------- encoder (고정)
def load_encoder(dev):
    import te_v11_score as te
    cfg = yaml.safe_load(open(MODEL_CFG))
    return te.Scorer(cfg, dev, [])                       # predictor 없이 (문맥 · 타깃 encoder 만 올라간다)


def encode(sc, x):
    b = sc.bundle
    x = x.to(b.device, b.dtype, non_blocking=True)
    ci, _ = sc.idx(x.size(0))
    with torch.no_grad(), sc.autocast():
        z = b.context_encoder(x, masks=[ci])            # (B, 8·256, D) — 1 판 0 단계 · 채점 하네스와 같은 호출
    return z.float()


# ----------------------------------------------------------------------------- 학습
def train(dev):
    sc = load_encoder(dev)
    D = sc.D
    tr = rows("train")
    ds = V3Clips(tr, flip=True)
    sampler = torch.utils.data.RandomSampler(ds, replacement=False, num_samples=STEPS * BS)
    dl = torch.utils.data.DataLoader(ds, batch_size=BS, sampler=sampler, num_workers=NW, pin_memory=True, drop_last=True,
                                     persistent_workers=NW > 0, prefetch_factor=4 if NW else None)
    torch.manual_seed(0); random.seed(0)
    model = SC.MapScaffold(D, depth=DEPTH).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    print(f"[train] clip {len(tr)} · step {STEPS} · batch {BS} · 모듈 파라미터 {n_par / 1e6:.2f} M · 깊이 {DEPTH}", flush=True)
    log, t0, tl = [], time.time(), time.time()
    for step, (x, occ, *_rest) in enumerate(dl):
        lr = LR * min(1.0, (step + 1) / WARM) * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
        for g in opt.param_groups:
            g["lr"] = lr
        z = encode(sc, x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(z)
        bce = F.binary_cross_entropy_with_logits(logit.float(), occ.to(dev, non_blocking=True), reduction="none").mean((0, 2))  # (16,) 칸별
        loss = bce.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == STEPS - 1:
            rec = dict(step=step, loss=round(loss.item(), 5), ctx=round(bce[:NC].mean().item(), 5), fut=round(bce[NC:].mean().item(), 5),
                       fut_last=round(bce[-1].item(), 5), gn=round(float(gn), 3), lr=round(lr, 7), s_per_step=round((time.time() - tl) / (50 if step else 1), 3))
            tl = time.time()
            log.append(rec)
            print(f"[train] {rec}", flush=True)
        if (step + 1) % 1000 == 0 or step == STEPS - 1:
            save(model, D, log, step + 1)
    print(f"[train] 끝 {time.time() - t0:.0f}s", flush=True)
    return sc, model


def save(model, D, log, step):
    RES.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": {"dim_in": D, "d": 384, "depth": DEPTH, "heads": 12, "n_ctx": NC, "n_fut": NF},
                "step": step, "log": log, "data": "blender_track v3 train barrier · timing ok · pos_a+pos_b", "seed": 0},
               RES / f"{TAG}.pt")


def load_model(dev, D):
    ck = torch.load(RES / f"{TAG}.pt", map_location="cpu")
    a = ck["arch"]
    model = SC.MapScaffold(a["dim_in"], a["d"], a["depth"], a["heads"], a["n_ctx"], a["n_fut"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    return model, ck.get("log", [])


@torch.no_grad()
def probs(model, z):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.sigmoid(model(z).float())


# ----------------------------------------------------------------------------- 지도 → 물체 중심
def centers(P, ctr, pres):
    """P (16, 256) 확률 · ctr (16, K, 2) · pres (16, K) → 추정 중심 (16, K, 2) (놓침 = nan) · 구역 최고 확률 pk (16, K)."""
    T, K = pres.shape
    est = np.full((T, K, 2), np.nan, np.float32); pk = np.zeros((T, K), np.float32)
    for t in range(T):
        ks = np.nonzero(pres[t])[0]
        if not len(ks):
            continue
        d = ((XY[:, None] - ctr[t, ks][None]) ** 2).sum(-1)                  # (256, |ks|)
        own = d.argmin(1)
        for j, k in enumerate(ks):
            p = P[t][own == j]; xy = XY[own == j]
            m = float(p.max()); pk[t, k] = m
            if m < 0.1:
                continue
            sel = p >= 0.5 * m
            est[t, k] = (p[sel, None] * xy[sel]).sum(0) / p[sel].sum()
    return est, pk


def boot_med(v, groups, n=1000, seed=0):
    """clip 단위 bootstrap 중앙값 95 % CI."""
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by = [np.nonzero(inv == g)[0] for g in range(len(ug))]
    rng = np.random.default_rng(seed)
    meds = []
    for _ in range(n):
        pick = rng.integers(0, len(ug), len(ug))
        meds.append(np.median(v[np.concatenate([idx_by[g] for g in pick])]))
    return [round(float(np.percentile(meds, 2.5)), 3), round(float(np.percentile(meds, 97.5)), 3)]


# ----------------------------------------------------------------------------- 평가: v3 test (학습 안)
@torch.no_grad()
def eval_v3(sc, model, dev):
    rs = rows("test")
    if EVAL_LIMIT:
        rs = rs[:EVAL_LIMIT]
    dl = torch.utils.data.DataLoader(V3Clips(rs, flip=False), batch_size=16, num_workers=NW)
    N = len(rs)
    PP = np.zeros((N, 16, S), np.float16); OCC = np.zeros((N, 16, S), np.float16)
    CTR = np.zeros((N, 16, KMAX, 2), np.float32); PRES = np.zeros((N, 16, KMAX), bool); VIS = np.zeros((N, 16, KMAX), np.float32)
    TGT = np.zeros((N, KMAX), bool)
    bce_sum = torch.zeros(16, device=dev)
    for x, occ, ctr, pres, vis, tgt, i in dl:
        z = encode(sc, x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(z).float()
        bce_sum += F.binary_cross_entropy_with_logits(lo, occ.to(dev), reduction="none").mean(2).sum(0)
        i = i.numpy()
        PP[i] = torch.sigmoid(lo).cpu().numpy(); OCC[i] = occ.numpy(); CTR[i] = ctr.numpy(); PRES[i] = pres.numpy(); VIS[i] = vis.numpy(); TGT[i] = tgt.numpy()
    bce = (bce_sum / N).cpu().numpy()

    recs = []                                                               # (clip, 칸, 물체) 단위 기록
    read = []
    for n in range(N):
        est, pk = centers(PP[n].astype(np.float32), CTR[n], PRES[n])
        r = rs[n]
        for k in range(KMAX):
            # 문맥 칸 읽기: 보이는 비율 ≥ 0.5 인 칸
            for t in range(NC):
                if PRES[n, t, k] and VIS[n, t, k] >= 0.5:
                    e = np.linalg.norm(est[t, k] - CTR[n, t, k]) if np.isfinite(est[t, k, 0]) else np.inf
                    read.append(e)
            ref = [t for t in range(NC) if PRES[n, t, k]]
            if not ref:
                continue
            stay = CTR[n, ref[-1], k]
            hid_end = bool(VIS[n, NC - 1, k] < 0.1) if PRES[n, NC - 1, k] else False
            for t in range(NC, 16):
                if not PRES[n, t, k]:
                    continue
                em = float(np.linalg.norm(est[t, k] - CTR[n, t, k])) if np.isfinite(est[t, k, 0]) else np.inf
                es = float(np.linalg.norm(stay - CTR[n, t, k]))
                recs.append(dict(clip=n, t=t, k=k, em=em, es=es, miss=not np.isfinite(em), target=bool(TGT[n, k]), hid_end=hid_end,
                                 arm=r["arm"], count=r["count"], place=r["place"], target_mode=r["target_mode"], camera=r["camera"],
                                 moved=float(np.linalg.norm(CTR[n, t, k] - stay))))
    read = np.array(read)
    out = {"n_clip": N, "bce_ctx": round(float(bce[:NC].mean()), 5), "bce_fut": round(float(bce[NC:].mean()), 5),
           "bce_by_cell": [round(float(b), 5) for b in bce],
           "read_visible_ctx": {"n": int(len(read)), "within1tok": round(float((read <= 1).mean()), 3), "err_median": round(float(np.median(read)), 3)}}

    def summ(sel_fn, name):
        rr = [x for x in recs if sel_fn(x)]
        if not rr:
            return
        em = np.array([x["em"] for x in rr]); es = np.array([x["es"] for x in rr]); cl = np.array([x["clip"] for x in rr])
        dif = np.where(np.isfinite(em), em, 99.0) - es
        out.setdefault("future", {})[name] = {
            "n_obj_cell": len(rr), "n_clip": int(len(np.unique(cl))), "miss": round(float(np.mean([x["miss"] for x in rr])), 3),
            "err_median_model": round(float(np.median(em)), 3), "err_median_stay": round(float(np.median(es)), 3),
            "diff_median_model_minus_stay": round(float(np.median(dif)), 3), "diff_ci": boot_med(dif, cl),
            "moved_median": round(float(np.median([x["moved"] for x in rr])), 3)}
        mv = np.array([x["moved"] >= 1.0 for x in rr])
        if mv.any():
            out["future"][name]["moved_ge1tok"] = {
                "n_obj_cell": int(mv.sum()), "miss": round(float(np.mean(~np.isfinite(em[mv]))), 3),
                "err_median_model": round(float(np.median(em[mv])), 3), "err_median_stay": round(float(np.median(es[mv])), 3),
                "diff_median_model_minus_stay": round(float(np.median(dif[mv])), 3), "diff_ci": boot_med(dif[mv], cl[mv])}

    summ(lambda x: True, "all")
    for c in (1, 2, 3):
        summ(lambda x, c=c: x["arm"] == "pos_a" and x["count"] == c, f"A_count{c}")
    summ(lambda x: x["target"], "target")
    summ(lambda x: x["target"] and x["hid_end"], "target_hidden_at_ctx_end")
    summ(lambda x: x["target"] and x["hid_end"] and x["target_mode"] == "moving", "target_hidden_moving")
    summ(lambda x: x["target"] and x["hid_end"] and x["target_mode"] == "moving" and x["place"] == "flat", "target_hidden_moving_flat")
    summ(lambda x: x["target"] and x["hid_end"] and x["target_mode"] == "moving" and x["place"] == "ramp", "target_hidden_moving_ramp")
    summ(lambda x: x["target"] and x["target_mode"] == "static", "target_static")
    summ(lambda x: not x["target"], "distractor")
    for cam in ("front", "side", "bird"):
        summ(lambda x, cam=cam: x["target"] and x["hid_end"] and x["camera"] == cam, f"target_hidden_{cam}")
    # 빈 장면 (물체 뺀 판 · 물체 1 개 장면 = 물체 0) — 헛 물체
    empty = [n for n in range(N) if rs[n]["arm"] == "pos_b" and rs[n]["count"] == 1 and not PRES[n].any()]
    if empty:
        mx = PP[empty][:, NC:].astype(np.float32).max(-1)
        out["empty_scene"] = {"n_clip": len(empty), "future_max_p_mean": round(float(mx.mean()), 3), "future_cells_maxp_ge_0.5": round(float((mx >= 0.5).mean()), 3)}
    # 판정 (학습 안, 돌리기 전에 고정): 물체 있는 판의 물체 수 1 · 2 · 3 모두 미래 칸 중심 오차 중앙값 모델 < 머무름 (놓침 = 무한 오차).
    # 11-01 수정 (학습 결과 보기 전, 학습 안 한 점검 job 만 본 뒤): 정지 물체는 머무름 오차가 정확히 0 이라 모든 물체-칸으로는 판정이 성립하지 않는다
    #   → 마지막 문맥 자리에서 1 토큰 이상 움직인 물체-칸만으로 판정한다 (모든 물체-칸 값도 같이 낸다)
    fut = out.get("future", {})
    out["verdict_in_domain"] = all("moved_ge1tok" in fut.get(f"A_count{c}", {}) and
                                   fut[f"A_count{c}"]["moved_ge1tok"]["err_median_model"] < fut[f"A_count{c}"]["moved_ge1tok"]["err_median_stay"]
                                   for c in (1, 2, 3))
    ex = list(range(min(48, N)))
    return out, dict(v3_P=PP[ex], v3_occ=OCC[ex], v3_ctr=CTR[ex], v3_pres=PRES[ex], v3_cid=np.array([rs[n]["cid"] for n in ex]))


# ----------------------------------------------------------------------------- 평가: VOBench 속도 sweep (학습 밖)
SPEEDS = [("S1", "1/3"), ("T1", "2/3"), ("orig", "1")]


def slope_ratio(est_x, true_x):
    """미래 칸 (8 개) 에서 추정 x 의 직선 기울기 / 같은 칸 참 x 의 기울기. 추정이 2 칸 미만이면 nan."""
    t = np.arange(NC, 16); ok = np.isfinite(est_x)
    if ok.sum() < 2:
        return np.nan
    return float(np.polyfit(t[ok], est_x[ok], 1)[0] / np.polyfit(t[ok], true_x[ok], 1)[0])


@torch.no_grad()
def eval_vob(model, dev, D):
    res, ex = {}, {}
    est_all, cen_all = {}, {}
    old = old_calculator()
    for nm, lab in SPEEDS:
        d = np.load(STAGE0 / f"vob_{nm}.npz")
        Z = d["z"]; cen = d["cen"]; vis = d["vis"]
        P = np.concatenate([probs(model, torch.from_numpy(Z[i:i + 8]).to(dev).float().reshape(-1, NC * S, D)).cpu().numpy() for i in range(0, len(Z), 8)])
        est = np.stack([centers(P[n], cen[n][:, None], np.ones((16, 1), bool))[0][:, 0] for n in range(len(P))])   # (N, 16, 2)
        est_all[nm], cen_all[nm] = est, cen
        ex[f"vob_{nm}_P"] = P.astype(np.float16); ex[f"vob_{nm}_cen"] = cen; ex[f"vob_{nm}_vis"] = vis
    keep = np.abs(cen_all["S1"][:, -1, 0] - cen_all["orig"][:, -1, 0]) >= 1.5               # 1 판 0 단계와 같은 27 block
    for nm, lab in SPEEDS:
        est, cen = est_all[nm], cen_all[nm]
        r = np.array([slope_ratio(est[n, NC:, 0], cen[n, NC:, 0]) for n in range(len(est))])[keep]
        fail = (~np.isfinite(r)) | (np.abs(r - 1) > 0.5)
        em = np.linalg.norm(est[:, NC:] - cen[:, NC:], axis=-1)[keep]; em = np.where(np.isfinite(em), em, np.inf)
        es = np.linalg.norm(cen[:, NC - 1:NC] - cen[:, NC:], axis=-1)[keep]
        ro = np.array([slope_ratio(old[nm][n, :, 0], cen[n, NC:, 0]) for n in range(len(est))])[keep]
        res[lab] = {"slope_ratio_median": round(float(np.nanmedian(r)), 3) if np.isfinite(r).any() else None,
                    "slope_ratio_per_block": [None if not np.isfinite(v) else round(float(v), 2) for v in r],
                    "n_fail": int(fail.sum()), "n_nan": int((~np.isfinite(r)).sum()),
                    "future_miss": round(float(np.isinf(em).mean()), 3),
                    "future_err_median": round(float(np.median(em)), 3), "stay_err_median": round(float(np.median(es)), 3),
                    "end_err_median": round(float(np.median(em[:, -1])), 3),
                    "future_col_err_median": round(float(np.median(np.where(np.isfinite(est[:, NC:, 0]), np.abs(est[:, NC:, 0] - cen[:, NC:, 0]), np.inf)[keep])), 3),
                    "stay_col_err_median": round(float(np.median(np.abs(cen[:, NC - 1:NC, 0] - cen[:, NC:, 0])[keep])), 3),
                    "old_calc_slope_ratio_median": round(float(np.median(ro)), 3), "old_calc_n_fail": int((np.abs(ro - 1) > 0.5).sum())}
    pe = {nm: est_all[nm][:, -1, 0] for nm, _ in SPEEDS}
    t1, t3 = cen_all["S1"][:, -1, 0], cen_all["orig"][:, -1, 0]
    fr = ((pe["S1"] - pe["orig"]) / (t1 - t3))[keep]
    out = {"n_block": int(keep.sum()), "by_speed": res,
           "follow_ratio_old_metric": {"median": round(float(np.nanmedian(fr)), 3) if np.isfinite(fr).any() else None,
                                       "n_negative": int((fr < 0).sum()), "n_nan": int((~np.isfinite(fr)).sum())}}
    out["verdict_out_of_domain"] = all(res[lab]["slope_ratio_median"] is not None and 0.7 <= res[lab]["slope_ratio_median"] <= 1.3
                                       and res[lab]["n_fail"] <= 3 for _, lab in SPEEDS)
    return out, ex


def old_calculator():
    """1 판 계산기 (물체다움 w · 직선) 의 미래 자리 — 같은 block 대조용. {speed: (N, 8, 2)}"""
    st = json.load(open(RES / "stage0.json"))["read"]
    sd = torch.load(RES / "object_score_w.pt", map_location="cpu")
    M = SC.ObjectScore(sd["lin.weight"].shape[1]); M.load_state_dict(sd)
    out = {}
    for nm, _ in SPEEDS:
        d = np.load(STAGE0 / f"vob_{nm}.npz")
        a = torch.cat([M(torch.from_numpy(d["z"][i:i + 8]).float()) for i in range(0, len(d["z"]), 8)]).detach()
        xf, _, _, _ = SC.calc(a, st["tau"], torch.arange(NC, 16), "visible", 1, kappa=st["kappa"])
        out[nm] = xf.numpy()
    return out


# ----------------------------------------------------------------------------- 실행
def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "eval"
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    t0 = time.time()
    if part == "train":
        sc, model = train(dev)
    else:
        sc = load_encoder(dev)
        model, _ = load_model(dev, sc.D)
    model.eval()
    _, log = load_model(dev, sc.D)
    v3, ex1 = eval_v3(sc, model, dev)
    print(json.dumps(v3, ensure_ascii=False), flush=True)
    vob, ex2 = eval_vob(model, dev, sc.D)
    print(json.dumps(vob, ensure_ascii=False), flush=True)
    res = {"tag": TAG, "date": time.strftime("%F %T"), "steps": STEPS, "batch": BS, "depth": DEPTH, "lr": LR, "eval_limit": EVAL_LIMIT,
           "train_log_last": log[-5:], "v3_test": v3, "vobench_speed_sweep": vob, "wall_s": round(time.time() - t0)}
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RES / f"stage0_{TAG}.json", "w"), indent=1, ensure_ascii=False)
    np.savez_compressed(RES / f"stage0_{TAG}_examples.npz", **ex1, **ex2)
    print(f"[done] {RES / f'stage0_{TAG}.json'}", flush=True)


if __name__ == "__main__":
    main()
