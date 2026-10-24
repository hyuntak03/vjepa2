#!/usr/bin/env python3
"""운반 계수 추정기 (G19, 진행자 2 · 진행자 1 합의 정의, 2026-10-22). 정의 정본 = paper_lab/EVIDENCE.md §3-2a.

움직이는 칸 (|d| >= 반 칸 = 8 px) 에서  y ≈ a·m + c·s  를 칸 × 1280 차원 전부에 걸쳐 함께 맞춘다 (최소제곱, 상수항 없음, a · c 스칼라).
  m = 같은 칸 문맥 8 블록 평균,  s = 옮긴 마지막 관측 블록 (블록 7, 프레임 14–15),  y = 블록 8 (프레임 16–17) 자리의 무엇.
  y = 실제 다음 블록 → 그 판의 κ,  y = predictor 첫 걸음 rollout → 그 모델의 옮긴 계수 c.
  s 판: (a) 블록 7 토큰 격자를 q + B(q)/16 에서 bilinear 로 읽음 (주) · (a') 칸 반올림 · (b...) 픽셀 와핑 뒤 인코딩 (S_MAKERS, 진행자 1).
  B(q) = 도착 칸 q 의 내용이 블록 7 에서 있던 곳까지의 변위 (px, 256 화면) — 자연 = Farneback 역방향 흐름 (16→14, 17→15) 의 칸 평균,
  렌더 = 라벨 변위 (물체 칸만, 배경 0).
  칸 묶음: 'all' = 움직이는 칸 전부, 'cons' = 앞-뒤 흐름 일관 (칸 평균 오차 <= 1 px) 칸만 (렌더는 all 과 같음).
도구 검사 (EVIDENCE §3-2a): y = 복사 (블록 7) → |c| <= 0.10,  y = 정답 자리 오라클 (VOBench) · (b) 블록 → c >= 0.8.

  # 1) 세트 캐시 (표적 encoder 한 번, 모델과 무관): 블록 0–8 토큰 · 변위 · 칸 · 모델과 무관한 y 의 통계
  CUDA_VISIBLE_DEVICES=2 python paper_lab/scripts/carry_coef.py prep --set natural   # natural12 · vobench · phys2d
  # 2) 모델마다 첫 걸음 rollout → c 통계
  CUDA_VISIBLE_DEVICES=2 python paper_lab/scripts/carry_coef.py model --ckpt z_training/vast/ar_state/latest.pt --name vast70 --sets natural,vobench,phys2d
  # 3) 표 (영상 단위 bootstrap 95 % 구간) · 짝지은 차이
  python paper_lab/scripts/carry_coef.py analyze --names vast70 state_nosup --sets natural,vobench,phys2d
  python paper_lab/scripts/carry_coef.py compare --a vast70 --b state_nosup --set vobench
  python paper_lab/scripts/carry_coef.py selftest          # CPU, 합성 격자로 회귀 · warp 검사
산출: 캐시 /local_datasets/world/world_analysis/cache/paper_lab_carry/<set>/ · 통계 paper_lab/_scratch/carry/<set>/<name>.npz · 표 paper_lab/_scratch/carry/table_*.{json,md}
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p_ in (str(REPO), str(REPO / "paper_lab"), str(REPO / "paper_lab/eval"), str(REPO / "z_research/scripts/analysis")):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

S, NB, C, D, G, CELL = 256, 16, 8, 1280, 16, 16
NKEEP = C + 1                                   # 블록 0..8 만 캐시 (문맥 8 + 다음 블록)
MOVE_PX, CONS_PX = 8.0, 1.0
CACHE = Path("/local_datasets/world/world_analysis/cache/paper_lab_carry")
OUT = REPO / "paper_lab/_scratch/carry"
SETS = ("natural", "natural12", "vobench", "phys2d", "natural_s1", "natural8_s1")   # _s1 = #81 사전 등록 확인 (PLAN §11-1): 같은 셔플 순서에서 4 fps 로 유효한 clip 의 201–500 번째 (natural 200 과 겹치지 않음), 8 fps 판은 같은 clip
PHYS2D_ROOT = Path("/local_datasets/world/training_data/phys2d_sup_v1/val")   # ⚠️ phys2d_single_v1 은 이 노드 (vll5) 에 없다 — 같은 생성기 계열의 대리
SVARS = ("a", "a1", "a_rot", "a1_rot")          # 토큰 옮기기 판 (a: bilinear, a1: 칸 반올림) + 위약 판 (_rot: 같은 크기 · 90° 돌린 방향, 사후 2026-10-22 06:20) + S_MAKERS 의 픽셀 판
CELLSETS = ("all", "cons", "wide")                # wide = 렌더 판 넓힌 칸 (상자와 겹친 면적 >= 25 %, 민감도용; 자연은 all 과 같음)

# 픽셀 와핑 뒤 인코딩 판 (진행자 1 이 채운다): name -> fn(x (B,3,32,256,256) 정규화 tensor, infos [dict], bundle) -> (B, G, G, D) tensor (LN 공간)
#   infos[b] 에 있는 것: natural = {"flow_b": (2, 256, 256, 2) F(16→14) · F(17→15) px, "frames_u8": None}; 렌더 = {"objs": [obj dict]} (아래 _render_objs)
S_MAKERS: dict = {}


# ---------------------------------------------------------------- 공통 수학 (numpy / torch 둘 다)
def warp_tokens(h7, disp, mode="a"):
    """h7 (B, G, G, D) torch, disp (B, G, G, 2) px (도착 → 출발, x y) → (s (B, G, G, D), ok (B, G, G) bool).
    a = bilinear (grid_sample, align_corners=False: 칸 중심 (i+0.5)/G), a1 = 칸 반올림. 출발점이 화면 밖이면 ok=False."""
    import torch
    import torch.nn.functional as F
    B = h7.shape[0]
    yy, xx = torch.meshgrid(torch.arange(G, device=h7.device), torch.arange(G, device=h7.device), indexing="ij")
    cx = (xx.float() + 0.5).unsqueeze(0) + disp[..., 0] / CELL              # 출발점 (칸 단위, 칸 중심 = i + 0.5)
    cy = (yy.float() + 0.5).unsqueeze(0) + disp[..., 1] / CELL
    if mode == "a1":
        ix = torch.floor(cx).long(); iy = torch.floor(cy).long()           # 출발점이 든 칸 = 칸 반올림
        ok = (ix >= 0) & (ix < G) & (iy >= 0) & (iy < G)
        ixc, iyc = ix.clamp(0, G - 1), iy.clamp(0, G - 1)
        bidx = torch.arange(B, device=h7.device).view(B, 1, 1).expand(B, G, G)
        return h7[bidx, iyc, ixc], ok
    ok = (cx >= 0.5) & (cx <= G - 0.5) & (cy >= 0.5) & (cy <= G - 0.5)     # bilinear 이웃 넷이 모두 화면 안
    grid = torch.stack([cx / G * 2 - 1, cy / G * 2 - 1], -1)               # (B, G, G, 2) 정규화 (x, y)
    s = F.grid_sample(h7.permute(0, 3, 1, 2).float(), grid.float(), mode="bilinear", padding_mode="border", align_corners=False)
    return s.permute(0, 2, 3, 1), ok


def ls_stats(m, s, y):
    """칸 (n, D) 셋 → 최소제곱 충분 통계 [Σmm, Σms, Σss, Σmy, Σsy, Σyy, n·D] (float64 numpy)."""
    m, s, y = m.double(), s.double(), y.double()
    return np.array([(m * m).sum().item(), (m * s).sum().item(), (s * s).sum().item(), (m * y).sum().item(),
                     (s * y).sum().item(), (y * y).sum().item(), float(m.numel())])


def solve(st):
    """충분 통계 (…, 7) → (a, c, R² 비 (잔차 / Σyy))."""
    st = np.asarray(st, np.float64)
    A = np.stack([np.stack([st[..., 0], st[..., 1]], -1), np.stack([st[..., 1], st[..., 2]], -1)], -2)
    b = np.stack([st[..., 3], st[..., 4]], -1)
    det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (A[..., 1, 1] * b[..., 0] - A[..., 0, 1] * b[..., 1]) / det
        c = (A[..., 0, 0] * b[..., 1] - A[..., 0, 1] * b[..., 0]) / det
        res = st[..., 5] - 2 * (a * st[..., 3] + c * st[..., 4]) + a * a * st[..., 0] + 2 * a * c * st[..., 1] + c * c * st[..., 2]
        frac = res / st[..., 5]
    return a, c, frac


def l1_irls(m, s, y, iters=25, eps=1e-4):
    """L1 회귀 (보조) — 같은 꼴 y ≈ a·m + c·s, IRLS. 입력 (N,) 평탄 torch. 점추정만."""
    import torch
    m, s, y = m.double(), s.double(), y.double()
    w = torch.ones_like(y)
    a = c = 0.0
    for _ in range(iters):
        A = torch.stack([torch.stack([(w * m * m).sum(), (w * m * s).sum()]), torch.stack([(w * m * s).sum(), (w * s * s).sum()])])
        b = torch.stack([(w * m * y).sum(), (w * s * y).sum()])
        a, c = torch.linalg.solve(A, b).tolist()
        w = 1.0 / (y - a * m - c * s).abs().clamp(min=eps)
    return a, c


# ---------------------------------------------------------------- 세트 읽기
def _tf():
    from analysis.intphys2.dataset import build_video_transform
    return build_video_transform(256, 1.0)


def _denorm_grey(x):
    """x (3, T, 256, 256) 정규화 tensor → (T, 256, 256) uint8 grey."""
    import passline as PL
    a = (x.cpu().float().numpy() * PL.IMNET_STD + PL.IMNET_MEAN).clip(0, 1)
    return (255 * (0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2])).astype(np.uint8)


def flow_disp(x):
    """자연 영상: Farneback 역방향 흐름 → 칸 변위 B (G, G, 2) px (도착 → 출발), 일관 칸 (G, G), 픽셀 흐름 (2, 256, 256, 2)."""
    import cv2
    g = _denorm_grey(x)
    fb = lambda a, b: cv2.calcOpticalFlowFarneback(g[a], g[b], None, 0.5, 3, 15, 3, 5, 1.2, 0)
    b16, b17 = fb(16, 14), fb(17, 15)                                      # 도착 장의 화소 → 출발 장
    f14, f15 = fb(14, 16), fb(15, 17)                                      # 앞-뒤 검사용
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)

    def fb_err(bk, fw):
        mx, my = xx + bk[..., 0], yy + bk[..., 1]
        fws = np.stack([cv2.remap(fw[..., i], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for i in range(2)], -1)
        return np.linalg.norm(bk + fws, axis=-1)
    err = 0.5 * (fb_err(b16, f14) + fb_err(b17, f15))
    bk = 0.5 * (b16 + b17)
    disp = bk.reshape(G, CELL, G, CELL, 2).mean((1, 3))
    cons = err.reshape(G, CELL, G, CELL).mean((1, 3)) <= CONS_PX
    return disp.astype(np.float32), cons, np.stack([b16, b17]).astype(np.float32)


def _cells_of_box(cx, cy, hx, hy):
    """도착 물체 상자 (중심 · 반폭, 256 px) → 칸 중심이 상자 안 (반폭 최소 반 칸) 인 칸 (G, G) bool."""
    c = np.arange(G) * CELL + CELL / 2
    hx, hy = max(hx, CELL / 2), max(hy, CELL / 2)
    return (np.abs(c[None, :] - cx) <= hx) & (np.abs(c[:, None] - cy) <= hy)


def _overlap_cells(cx, cy, hx, hy, frac=0.25):
    """상자와 겹친 면적이 칸 면적의 frac 이상인 칸 (G, G) bool."""
    lo = np.arange(G) * CELL
    ox = np.clip(np.minimum(lo + CELL, cx + hx) - np.maximum(lo, cx - hx), 0, None)
    oy = np.clip(np.minimum(lo + CELL, cy + hy) - np.maximum(lo, cy - hy), 0, None)
    return (oy[:, None] * ox[None, :]) >= frac * CELL * CELL


def _render_disp(objs):
    """objs = [{"src": (x, y), "dst": (x, y), "half": (hx, hy)}] 256 px → (disp (G,G,2), 후보 칸 (중심 규칙), 넓힌 후보 칸).
    중심 규칙 = 칸 중심이 도착 상자 (반폭 최소 반 칸) 안. 넓힌 규칙 = 중심 규칙 ∪ 상자와 겹친 면적 >= 25 %. 두 물체가 겹친 칸은 둘 다에서 뺀다."""
    disp = np.zeros((G, G, 2), np.float32)
    cnt = np.zeros((G, G), np.int32)
    cen = np.zeros((G, G), bool)
    for o in objs:
        mc = _cells_of_box(o["dst"][0], o["dst"][1], *o["half"])
        mw = mc | _overlap_cells(o["dst"][0], o["dst"][1], *o["half"])
        disp[mw] = np.array(o["src"], np.float32) - np.array(o["dst"], np.float32)
        cnt += mw
        cen |= mc
    one = cnt == 1
    return disp, cen & one, one


def natural_items(n, fps, skip=0):
    """skip > 0 (사전 등록 확인 판): 같은 셔플 순서에서 **4 fps 기준으로 유효한** clip 을 앞에서 skip 개 건너뛴다 → fps 와 무관하게
    같은 clip 목록 (4 fps 로 유효하면 더 높은 fps 로도 유효하다). skip = 0 이면 예전과 같다 (그 fps 로 유효한 앞 n 개)."""
    import decord
    import passline as PL
    lines = [l.split()[0] for l in open(PL.K400_LIST) if l.strip()]
    rng = np.random.default_rng(0)
    rng.shuffle(lines)                                                        # 하네스 natural 과 같은 순서 · 같은 가운데 32 장
    tf_ = _tf()
    got = 0
    seen4 = 0
    for f in lines:
        if got >= n:
            break
        try:
            vr = decord.VideoReader(str(PL.K400_DIR / f), num_threads=1)
            if skip:
                step4 = max(1, int(round(vr.get_avg_fps() / 4)))
                if len(vr) < 32 * step4:
                    continue
                seen4 += 1
                if seen4 <= skip:
                    continue
            step = max(1, int(round(vr.get_avg_fps() / fps)))
            if len(vr) < 32 * step:
                continue
            st = (len(vr) - 32 * step) // 2
            x = tf_(vr.get_batch(list(range(st, st + 32 * step, step))).asnumpy())
        except Exception:
            continue
        disp, cons, flow_b = flow_disp(x)
        got += 1
        yield f, x, dict(disp=disp, cons=cons, cand=np.ones((G, G), bool), flow_b=flow_b)


def vobench_items(n):
    import yaml
    import design_headroom as H
    import passline as PL
    from evals.world_model_analysis.data import WMADataset
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir)
    cfg = yaml.safe_load(open(H.SETS["vobench"]["cfg"]))
    ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=dict(ir.rp.MODEL), features={"cache_dir": "/tmp"}, surprise={}))
    rows = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
    ids = [i for i, rec in enumerate(ds.records) if (r := rows.get(rec.video_id)) and r["occlusion"] == "visible"
           and r["motion"] != "flat_static" and r["plausible"] == "1"]
    for i in ids[:n]:
        r = rows[ds.records[i].video_id]
        lab = json.load(open(H.FR / Path(r["file_name"]).parent / "label.json"))["labels"]
        L = lab.get("A" if r["variant"] in ("pos_a", "imp_ab") else "B")
        if not L or L[0] is None:
            continue
        P = np.array([[L[3 * s]["u"], L[3 * s]["v"]] for s in range(32)], np.float64)
        hid = np.array([L[3 * s]["hidden"] for s in range(32)], np.float64)
        if hid[14:18].max() >= 0.5:
            continue
        sc = PL.LABEL_SCALE
        src, dst = P[14:16].mean(0) * sc, P[16:18].mean(0) * sc
        half = (0.5 * sc * np.mean([L[3 * s]["ext_u"] for s in (16, 17)]), 0.5 * sc * np.mean([L[3 * s]["ext_v"] for s in (16, 17)]))
        objs = [dict(src=tuple(src), dst=tuple(dst), half=half)]
        disp, cand, wide = _render_disp(objs)
        x = ds.clip(i)
        yield ds.records[i].video_id, x, dict(disp=disp, cons=np.ones((G, G), bool), cand=cand, cand_wide=wide, objs=objs, P=P, hid=hid)


def phys2d_items(n):
    import decord
    rows = [l.split()[0] for l in open(PHYS2D_ROOT / "index.csv") if l.strip()]
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    tf_ = _tf()
    got = 0
    for mp4 in rows:
        if got >= n:
            break
        stem = mp4[:-4]
        try:
            z = np.load(stem + "_states.npz")
            vr = decord.VideoReader(mp4, num_threads=1)
        except Exception:
            continue
        T = min(len(vr), z["pos"].shape[0])
        pick = None
        for st in range(0, T - 31, 8):                                     # 블록 7→8 에 움직이는 물체가 있는 첫 시작
            f = np.arange(st + 14, st + 18)
            pos, pres, hidd, vis = z["pos"][f], z["present"][f], z["hidden"][f], z["vis_frac"][f]
            ok = pres.all(0) & ~hidd.any(0) & (vis.min(0) > 0.9)
            mv = np.linalg.norm(pos[2:].mean(0) - pos[:2].mean(0), axis=-1)
            if (ok & (mv >= MOVE_PX)).any():
                pick = st
                break
        if pick is None:
            continue
        f = np.arange(pick + 14, pick + 18)
        objs = []
        for j in range(z["pos"].shape[1]):
            if not (z["present"][f, j].all() and not z["hidden"][f, j].any() and z["vis_frac"][f, j].min() > 0.9):
                continue
            bb = z["bbox"][f[2:], j].mean(0)
            objs.append(dict(src=tuple(z["pos"][f[:2], j].mean(0)), dst=tuple(z["pos"][f[2:], j].mean(0)),
                             half=(0.5 * (bb[2] - bb[0]), 0.5 * (bb[3] - bb[1]))))
        disp, cand, wide = _render_disp(objs)
        x = tf_(vr.get_batch(list(range(pick, pick + 32))).asnumpy())
        got += 1
        yield f"{Path(stem).name}@{pick}", x, dict(disp=disp, cons=np.ones((G, G), bool), cand=cand, cand_wide=wide, objs=objs)


def items(setname, n):
    if setname == "natural":
        return natural_items(n, 4)
    if setname == "natural12":
        return natural_items(n, 12)
    if setname == "natural_s1":
        return natural_items(n, 4, skip=200)
    if setname == "natural8_s1":
        return natural_items(n, 8, skip=200)
    if setname == "vobench":
        return vobench_items(n)
    if setname == "phys2d":
        return phys2d_items(n)
    raise SystemExit(setname)


# ---------------------------------------------------------------- 회귀 한 판 (배치)
def cell_masks(info_b, ok_by_var):
    """(G,G) 움직이는 칸 = 후보 · |disp| >= 8 px · 그 s 판의 출발점이 화면 안. 묶음 all / cons."""
    big = np.linalg.norm(info_b["disp"], axis=-1) >= MOVE_PX
    mv = info_b["cand"] & big
    wide = info_b.get("cand_wide", info_b["cand"]) & big
    out = {}
    for v, ok in ok_by_var.items():
        base = mv & ok
        out[(v, "all")] = base
        out[(v, "cons")] = base & info_b["cons"]
        out[(v, "wide")] = wide & ok
    return out


def batch_stats(hb, infos, ys, svars_extra):
    """hb (B, 9, G, G, D) torch float, infos [dict], ys {name: (B, G, G, D)} → ({(svar, cellset, y): (B, 7)}, pooled {(svar,cellset,y): [m,s,y 평탄]}).
    svars_extra {name: (B, G, G, D)} = 픽셀 판 s."""
    import torch
    B = hb.shape[0]
    m = hb[:, :C].mean(1)
    h7 = hb[:, C - 1]
    disp = torch.as_tensor(np.stack([i["disp"] for i in infos]), device=hb.device)
    svals, oks = {}, {}
    disp_rot = torch.stack([-disp[..., 1], disp[..., 0]], -1)              # 위약: 같은 크기, 90° 돌린 방향 (공간 상관은 같고 실제 움직임과는 다름)
    for v in SVARS:
        base = v.split("_")[0]
        svals[v], oks[v] = warp_tokens(h7, disp_rot if v.endswith("_rot") else disp, base)
    for v in SVARS:
        if v.endswith("_rot"):
            oks[v] = oks[v.split("_")[0]]                                   # 위약 판은 실제 판과 **같은 칸** 에서 (화면 밖 출발점은 가장자리로 고정)
    for v, t in svars_extra.items():
        svals[v], oks[v] = t, oks["a"]                                     # 픽셀 판은 bilinear 판과 같은 칸에서
    stats, pooled = {}, {}
    for b in range(B):
        masks = cell_masks(infos[b], {v: oks[v][b].cpu().numpy() for v in svals})
        for (v, cs), mk in masks.items():
            mk_t = torch.as_tensor(mk, device=hb.device)
            mm, ss = m[b][mk_t], svals[v][b][mk_t]
            for yn, yv in ys.items():
                yy = yv[b][mk_t]
                key = (v, cs, yn)
                stats.setdefault(key, np.zeros((B, 7)))[b] = ls_stats(mm, ss, yy.float()) if mk.any() else np.zeros(7)
                if mk.any():
                    pooled.setdefault(key, [[], [], []])
                    for li, t in zip(pooled[key], (mm, ss, yy)):
                        li.append(t.float().reshape(-1)[::4].cpu())                 # L1 보조용 (4 개 중 1 개 차원, 메모리)
    return stats, pooled


def finish(stats_list, pooled, names):
    """배치 통계를 합쳐 {key: (N, 7)} + L1 점추정."""
    import torch
    keys = sorted({k for st in stats_list for k in st})
    out = {k: np.concatenate([st.get(k, np.zeros((len(n), 7))) for st, n in zip(stats_list, names)]) for k in keys}
    l1 = {}
    for k, (ms, ss, ys) in pooled.items():
        if ms:
            l1[k] = l1_irls(torch.cat(ms), torch.cat(ss), torch.cat(ys))
    return out, l1


def save_stats(path, stats, l1, vids, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = {"|".join(k): v for k, v in stats.items()}
    arr.update({"l1|" + "|".join(k): np.array(v) for k, v in l1.items()})
    arr["video"] = np.array(vids)
    if extra:
        arr["meta"] = np.array(json.dumps(extra))
    np.savez(path, **arr)


# ---------------------------------------------------------------- prep (세트 캐시 + 모델과 무관한 y)
def prep(setname, n, bs):
    import torch
    import passline as PL
    from app.vjepa_frozen.ar import encode_blocks
    dev = torch.device("cuda", 0)
    model, bundle = PL._load_model("z_training/vast/ar_state/latest.pt", dev)           # encoder 만 쓴다 (predictor 무관)
    cdir = CACHE / setname
    cdir.mkdir(parents=True, exist_ok=True)
    hs, infos_all, vids, stats_list, names_list = [], [], [], [], []
    pooled = {}
    buf = []
    t0 = time.time()

    def flush():
        if not buf:
            return
        x = torch.stack([b[1] for b in buf]).to(dev, dtype=bundle.dtype)
        infos = [b[2] for b in buf]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hb = encode_blocks(bundle.target_encoder, x, 2).float().view(len(buf), NB, G, G, D)
            ys = {"true": hb[:, C], "copy": hb[:, C - 1], "mean8": hb[:, :C].mean(1)}
            if setname == "vobench":
                ys["oracle"] = torch.stack([PL._oracle_render(hb[b], i["P"], i["hid"])[0] for b, i in enumerate(infos)])
                np.save(cdir / f"y_oracle_{len(hs):04d}.npy", ys["oracle"].half().cpu().numpy())     # restat 용 (06:20 추가)
            extra = {k: fn(x, infos, bundle).float() for k, fn in S_MAKERS.items()}
            for k, t in extra.items():
                ys["s_" + k] = t                                                      # (b) 블록 자체도 y 로 (도구 검사 c >= 0.8)
            st, pl = batch_stats(hb[:, :NKEEP], infos, ys, extra)
        stats_list.append(st); names_list.append([b[0] for b in buf])
        for k, v in pl.items():
            pooled.setdefault(k, [[], [], []])
            for a_, b_ in zip(pooled[k], v):
                a_.extend(b_)
        hs.append(hb[:, :NKEEP].half().cpu().numpy())
        for k in extra:
            np.save(cdir / f"s_{k}_{len(hs) - 1:04d}.npy", extra[k].half().cpu().numpy())
        for b in buf:
            vids.append(b[0])
            infos_all.append({k: v for k, v in b[2].items() if k in ("disp", "cons", "cand", "cand_wide")})
        buf.clear()
        print(f"[prep {setname}] {len(vids)} clips ({time.time() - t0:.0f}s)", flush=True)

    for it in items(setname, n):
        buf.append(it)
        if len(buf) == bs:
            flush()
    flush()
    H_ = np.concatenate(hs)
    np.save(cdir / "h.npy", H_)
    np.savez(cdir / "meta.npz", video=np.array(vids), disp=np.stack([i["disp"] for i in infos_all]),
             cons=np.stack([i["cons"] for i in infos_all]), cand=np.stack([i["cand"] for i in infos_all]),
             cand_wide=np.stack([i.get("cand_wide", i["cand"]) for i in infos_all]),
             extras=np.array(sorted(S_MAKERS)), batch=np.array(bs))
    stats, l1 = finish(stats_list, pooled, names_list)
    save_stats(OUT / setname / "_data.npz", stats, l1, vids, {"set": setname, "n": len(vids), "phys2d_root": str(PHYS2D_ROOT)})
    print(f"[prep {setname}] done {len(vids)} clips, h {H_.shape} -> {cdir}", flush=True)
    _print_quick(setname, "_data")


# ---------------------------------------------------------------- restat (캐시에서 모델과 무관한 y 통계를 다시 셈, 사후 판 추가용)
def restat(setname):
    """캐시 (h · 변위 · 칸 · s_pix · 있으면 y_oracle) 로 _data 통계를 지금의 SVARS 로 다시 낸다 → _data_restat.npz. GPU 있으면 GPU."""
    import torch
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    cdir = CACHE / setname
    H_ = np.load(cdir / "h.npy", mmap_mode="r")
    meta = np.load(cdir / "meta.npz")
    vids = list(meta["video"]); cbs = int(meta["batch"]); extras = [str(e) for e in meta["extras"]]
    wide = meta["cand_wide"] if "cand_wide" in meta.files else meta["cand"]
    stats_list, names_list, pooled = [], [], {}
    for bi, b0 in enumerate(range(0, len(vids), cbs)):
        hb = torch.as_tensor(np.asarray(H_[b0:b0 + cbs]), device=dev).float()
        B = hb.shape[0]
        infos = [dict(disp=meta["disp"][b0 + j], cons=meta["cons"][b0 + j], cand=meta["cand"][b0 + j], cand_wide=wide[b0 + j]) for j in range(B)]
        extra = {e: torch.as_tensor(np.load(cdir / f"s_{e}_{bi:04d}.npy"), device=dev).float() for e in extras}
        ys = {"true": hb[:, C], "copy": hb[:, C - 1], "mean8": hb[:, :C].mean(1)}
        if (cdir / f"y_oracle_{bi:04d}.npy").exists():
            ys["oracle"] = torch.as_tensor(np.load(cdir / f"y_oracle_{bi:04d}.npy"), device=dev).float()
        for k, t in extra.items():
            ys["s_" + k] = t
        with torch.no_grad():
            st, pl = batch_stats(hb, infos, ys, extra)
        stats_list.append(st); names_list.append(vids[b0:b0 + B])
        for k, v in pl.items():
            pooled.setdefault(k, [[], [], []])
            for a_, b_ in zip(pooled[k], v):
                a_.extend(b_)
    stats, l1 = finish(stats_list, pooled, names_list)
    save_stats(OUT / setname / "_data_restat.npz", stats, l1, vids, {"set": setname, "n": len(vids), "restat": True})
    _print_quick(setname, "_data_restat")


def placebo_table(setname, name="_data_restat", ys=("true", "copy", "oracle", "s_pix", "model"), B=1000):
    """위약 차분 κ_Δ = c(y; s_실제) − c(y; s_rot), 같은 영상 bootstrap (같은 재표집으로 짝지음)."""
    st, _, _ = _load_stats(setname, name)
    rows = []
    svs = {k[0] for k in st}
    bases = sorted(v for v in svs if not v.endswith("_rot") and v + "_rot" in svs)   # a · a1 · 픽셀 판 (pix ↔ pix_rot) 모두 — 같은 칸 (a 의 실제 판 칸)
    for base in bases:
        for cs in CELLSETS:
            for yn in ys:
                k1, k2 = (base, cs, yn), (base + "_rot", cs, yn)
                if k1 not in st or k2 not in st:
                    continue
                x1, x2 = st[k1], st[k2]
                c1 = solve(x1.sum(0))[1]; c2 = solve(x2.sum(0))[1]
                idx = np.random.default_rng(0).integers(0, len(x1), size=(B, len(x1)))
                d = solve(x1[idx].sum(1))[1] - solve(x2[idx].sum(1))[1]
                rows.append(dict(set=setname, name=name, s=base, cells=cs, y=yn, c_real=float(c1), c_rot=float(c2), delta=float(c1 - c2),
                                 ci=[float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]))
                print(f"  {setname:9s} {name:14s} s={base:4s} cells={cs:4s} y={yn:7s} c_real={c1:+.3f} c_rot={c2:+.3f} Δ={c1 - c2:+.3f} [{rows[-1]['ci'][0]:+.3f}, {rows[-1]['ci'][1]:+.3f}]")
    return rows


# ---------------------------------------------------------------- model (첫 걸음 rollout → c)
def run_model(ckpt, name, sets, bs, ctx_in):
    import torch
    import passline as PL
    from app.vjepa_frozen.ar import block_indices
    dev = torch.device("cuda", 0)
    model, bundle = PL._load_model(ckpt, dev)
    _, mcfg = PL._install_method(ckpt)
    if mcfg is not None:
        raise SystemExit(f"{ckpt}: 입력 변환이 걸린 체크포인트 (pl_method infer={mcfg}) — 캐시 토큰은 원래 공간이라 이 추정기에서 뺀다")
    pred = bundle.predictor
    for setname in sets:
        cdir = CACHE / setname
        H_ = np.load(cdir / "h.npy", mmap_mode="r")
        meta = np.load(cdir / "meta.npz")
        vids = list(meta["video"])
        cbs = int(meta["batch"])
        extras = [str(e) for e in meta["extras"]]
        stats_list, names_list, pooled = [], [], {}
        t0 = time.time()
        for bi, b0 in enumerate(range(0, len(vids), cbs)):
            hb = torch.as_tensor(np.asarray(H_[b0:b0 + cbs]), device=dev).float()
            B = hb.shape[0]
            wide = meta["cand_wide"] if "cand_wide" in meta.files else meta["cand"]
            infos = [dict(disp=meta["disp"][b0 + j], cons=meta["cons"][b0 + j], cand=meta["cand"][b0 + j], cand_wide=wide[b0 + j]) for j in range(B)]
            extra = {e: torch.as_tensor(np.load(cdir / f"s_{e}_{bi:04d}.npy"), device=dev).float() for e in extras}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h_in = hb[:, C - ctx_in:C].reshape(B, ctx_in * S, D).to(torch.bfloat16)
                p1 = pred.rollout(h_in, block_indices(B, ctx_in, S, dev), 1).float().view(B, 1, G, G, D)[:, 0]
                st, pl = batch_stats(hb, infos, {"model": p1}, extra)
            stats_list.append(st); names_list.append(vids[b0:b0 + B])
            for k, v in pl.items():
                pooled.setdefault(k, [[], [], []])
                for a_, b_ in zip(pooled[k], v):
                    a_.extend(b_)
        stats, l1 = finish(stats_list, pooled, names_list)
        save_stats(OUT / setname / f"{name}.npz", stats, l1, vids, {"ckpt": str(ckpt), "ctx_in": ctx_in, "set": setname})
        print(f"[model {name}] {setname} {len(vids)} clips ({time.time() - t0:.0f}s)", flush=True)
        _print_quick(setname, name)


# ---------------------------------------------------------------- 분석
def _load_stats(setname, name):
    z = np.load(OUT / setname / f"{name}.npz", allow_pickle=False)
    st = {tuple(k.split("|")): z[k] for k in z.files if k not in ("video", "meta") and not k.startswith("l1|")}
    l1 = {tuple(k.split("|")[1:]): z[k] for k in z.files if k.startswith("l1|")}
    return st, l1, list(z["video"])


def boot(st, B=1000, seed=0):
    """영상 단위 bootstrap → (a, c, frac 점추정, c 95 % 구간, 쓴 영상 수)."""
    used = st[:, 6] > 0
    a, c, fr = solve(st.sum(0))
    idx = np.random.default_rng(seed).integers(0, len(st), size=(B, len(st)))
    _, cb, _ = solve(st[idx].sum(1))
    return float(a), float(c), float(fr), [float(np.nanpercentile(cb, 2.5)), float(np.nanpercentile(cb, 97.5))], int(used.sum())


def _print_quick(setname, name):
    st, l1, _ = _load_stats(setname, name)
    for k in sorted(st):
        a, c, fr, ci, nu = boot(st[k], B=200)
        l1s = f"{l1[k][1]:+.3f}" if k in l1 else "—"
        print(f"  {setname:9s} {name:14s} s={k[0]:4s} cells={k[1]:4s} y={k[2]:8s}  a={a:+.3f} c={c:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
              f"  resid={fr:.3f}  videos={nu}  L1 c={l1s}", flush=True)


def analyze(names, sets):
    rows = []
    for setname in sets:
        for nm in ["_data"] + list(names):
            if not (OUT / setname / f"{nm}.npz").exists():
                continue
            st, l1, _ = _load_stats(setname, nm)
            for k in sorted(st):
                a, c, fr, ci, nu = boot(st[k])
                rows.append(dict(set=setname, name=nm, s=k[0], cells=k[1], y=k[2], a=a, c=c, c_lo=ci[0], c_hi=ci[1], resid=fr,
                                 videos=nu, l1_a=float(l1[k][0]) if k in l1 else None, l1_c=float(l1[k][1]) if k in l1 else None))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_".join(names) if len(names) <= 3 else f"{len(names)}models"
    json.dump(rows, open(OUT / f"table_{tag}.json", "w"), indent=1)
    lines = ["| 판 | 모델 / y | s | 칸 | a | c [95 %] | 잔차 비 | 영상 | L1 c |", "|---|---|---|---|---:|---|---:|---:|---:|"]
    for r in rows:
        who = r["y"] if r["name"] == "_data" else r["name"]
        if r["name"] != "_data" and r["y"] != "model":
            continue
        l1s = f"{r['l1_c']:+.3f}" if r["l1_c"] is not None else "—"
        lines.append(f"| {r['set']} | {who} | {r['s']} | {r['cells']} | {r['a']:+.3f} | {r['c']:+.3f} [{r['c_lo']:+.3f}, {r['c_hi']:+.3f}] | "
                     f"{r['resid']:.3f} | {r['videos']} | {l1s} |")
    (OUT / f"table_{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def compare(a, b, setname, svar="a", cells="all", B=1000):
    """같은 영상의 짝지은 bootstrap: c_b − c_a."""
    sa, _, va = _load_stats(setname, a)
    sb, _, vb = _load_stats(setname, b)
    assert va == vb, "같은 캐시의 영상이어야 한다"
    ka, kb = (svar, cells, "model") if a != "_data" else (svar, cells, "true"), (svar, cells, "model") if b != "_data" else (svar, cells, "true")
    xa, xb = sa[ka], sb[kb]
    _, ca, _ = solve(xa.sum(0)); _, cb, _ = solve(xb.sum(0))
    idx = np.random.default_rng(0).integers(0, len(xa), size=(B, len(xa)))
    d = solve(xb[idx].sum(1))[1] - solve(xa[idx].sum(1))[1]
    res = dict(set=setname, a=a, b=b, s=svar, cells=cells, c_a=float(ca), c_b=float(cb), diff=float(cb - ca),
               ci=[float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))])
    print(json.dumps(res, ensure_ascii=False))
    return res


# ---------------------------------------------------------------- CPU 자기 검사
def selftest():
    import torch
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    Bt = 3
    h = torch.randn(Bt, NKEEP, G, G, D) * 0.3
    h[:, :C] += torch.randn(Bt, 1, G, G, D)                                  # 칸마다 시간에 걸쳐 같은 내용 + 잡음
    disp = np.zeros((Bt, G, G, 2), np.float32)
    disp[:, 4:10, 4:10] = [-16.0, 0.0]                                       # 도착 칸 q 의 내용은 왼쪽 한 칸에서 왔다 (오른쪽으로 16 px 이동)
    infos = [dict(disp=disp[b], cons=np.ones((G, G), bool), cand=np.ones((G, G), bool)) for b in range(Bt)]
    s_true, _ = warp_tokens(h[:, C - 1], torch.as_tensor(disp), "a1")
    y_true = 0.3 * h[:, :C].mean(1) + 0.6 * s_true + 0.05 * torch.randn(Bt, G, G, D)
    ys = {"y": y_true, "copy": h[:, C - 1], "shift": s_true, "mean8": h[:, :C].mean(1)}
    st, pl = batch_stats(h, infos, ys, {})
    out, l1 = finish([st], pl, [[f"v{i}" for i in range(Bt)]])
    ok = True
    for (v, cs, yn), x in sorted(out.items()):
        if cs != "all":
            continue
        a, c, fr = solve(x.sum(0))
        print(f"  s={v:3s} y={yn:6s} a={a:+.3f} c={c:+.3f} resid={fr:.3f}  L1 c={l1[(v, cs, yn)][1]:+.3f}")
        if v.endswith("_rot"):                                              # 위약 판: 공간 상관 없는 합성 격자에서 c ≈ 0
            if yn in ("y", "shift"):
                ok &= abs(c) < 0.05
            continue
        if yn == "y":
            ok &= abs(c - 0.6) < 0.05 and abs(a - 0.3) < 0.05
        if yn == "shift":
            ok &= abs(c - 1) < 1e-3
        if yn == "mean8":
            ok &= abs(a - 1) < 1e-3 and abs(c) < 1e-3
    # bilinear 반 칸 이동: 이웃 두 칸 평균과 같아야 한다
    d2 = torch.zeros(1, G, G, 2); d2[..., 0] = -8.0
    sa, okm = warp_tokens(h[:1, C - 1], d2, "a")
    ref = 0.5 * (h[0, C - 1, 5, 5] + h[0, C - 1, 5, 4])
    ok &= torch.allclose(sa[0, 5, 5], ref, atol=1e-4) and bool(okm[0, 5, 5])
    # 칸 반올림 (a1 = 출발점이 든 칸): -8 px → 출발점이 칸 경계 (x = 5.0) → 자기 칸 (반올림 0 칸), -12 px → 왼쪽 칸
    s1, _ = warp_tokens(h[:1, C - 1], d2, "a1")
    ok &= torch.allclose(s1[0, 5, 5], h[0, C - 1, 5, 5])
    d3 = torch.zeros(1, G, G, 2); d3[..., 0] = -12.0
    s3, _ = warp_tokens(h[:1, C - 1], d3, "a1")
    ok &= torch.allclose(s3[0, 5, 5], h[0, C - 1, 5, 4])
    print("SELFTEST", "OK" if ok else "FAIL")
    return ok


def main():
    import torch
    torch.set_num_threads(int(os.environ.get("CARRY_THREADS", 4)))          # 공용 노드: 스레드 과다구독이면 CPU 를 다 잡고 멈춘 듯 돈다 (실측)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--set", required=True, choices=SETS); p.add_argument("--n", type=int, default=200); p.add_argument("--bs", type=int, default=8)
    p = sub.add_parser("model"); p.add_argument("--ckpt", required=True); p.add_argument("--name", required=True)
    p.add_argument("--sets", default="natural,vobench,phys2d"); p.add_argument("--bs", type=int, default=8); p.add_argument("--ctx-in", type=int, default=C)
    p = sub.add_parser("analyze"); p.add_argument("--names", nargs="*", default=[]); p.add_argument("--sets", default="natural,vobench,phys2d")
    p = sub.add_parser("compare"); p.add_argument("--a", required=True); p.add_argument("--b", required=True); p.add_argument("--set", required=True)
    p.add_argument("--s", default="a"); p.add_argument("--cells", default="all")
    p = sub.add_parser("restat"); p.add_argument("--set", required=True, choices=SETS)
    p = sub.add_parser("placebo"); p.add_argument("--set", required=True); p.add_argument("--name", default="_data_restat")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.set, a.n, a.bs)
    elif a.cmd == "model":
        run_model(a.ckpt, a.name, a.sets.split(","), a.bs, a.ctx_in)
    elif a.cmd == "analyze":
        analyze(a.names, a.sets.split(","))
    elif a.cmd == "restat":
        restat(a.set)
    elif a.cmd == "placebo":
        placebo_table(a.set, a.name)
    elif a.cmd == "compare":
        compare(a.a, a.b, a.set, a.s, a.cells)
    else:
        sys.exit(0 if selftest() else 1)


if __name__ == "__main__":
    main()
