#!/usr/bin/env python3
"""adapter 1 단계 — 파라미터 없는 속도 읽기가 학습 밖 (VOBench) 에서 속도를 따라가나 (2026-11-01, 사용자 결정: 2 판 취소 · adapter 디벨롭).

adapter = ① 속도 읽기 (feature 대응) · ② 두 겹 상태 · ③ 옮기기 (splatting). 이 스크립트는 ① 만 잰다 — 여기서 느린 속도를 못 잡으면 ② · ③ 은 의미 없다.
학습 없음 · GPU 없음. 입력 = 1 판 0 단계가 뽑아 둔 VOBench 속도 sweep 문맥 z (27 block × 속도 1/3 · 2/3 · 1, 문맥 8 칸 × 256 토큰 × 1280).

속도 읽기 (토큰마다, 파라미터 0 — 창 반경 R 과 칸 간격 K 만 고정값)
  ẑ_t(r) = LN(z_t(r)) / ‖·‖                                   (학습 계수 없는 층 정규화 뒤 단위 길이)
  S_{t,k}(r, r') = ẑ_t(r) · ẑ_{t+k}(r'),  r' ∈ r 주변 (2R+1)² 창
  d_{t,k}(r) = argmax_{r'} S − r  + 포물선 보간 (열 · 행 따로, 칸 안 ±0.5)
  v_{t,k}(r) = d_{t,k}(r) / k                                    (토큰 / 칸)
물체 속도 (평가 전용): 속도장을 참 물체 중심 (라벨) 에서 이중선형으로 읽는다. 두 칸 모두 보이는 (보이는 비율 ≥ 0.9) 쌍만,
  마지막으로 보이는 칸 쪽 4 칸 안의 (t, k) 쌍들의 중앙값 = v̂.
  ⚠️ adapter 는 물체 자리를 추정하지 않는다 (토큰을 그 자리에서 그대로 옮긴다) — 그래서 참 중심에서 읽는 것이 adapter 와 같은 조건이다.
판정 (오늘 판정과 같은 꼴)
  읽기 비 = v̂_x / (문맥 끝 참 속도_x)  ·  굴리기 기울기 비 = v̂_x / (미래 참 기울기_x)  (등속으로 굴리면 미래 기울기 = v̂)
  통과 = 세 속도 모두 굴리기 기울기 비 중앙값 0.7–1.3 · 50 % 넘게 틀린 block ≤ 3 / 27 (사전 등록 그대로)
  따라감 비 (옛 지표) = (끝 자리 예측 1/3 배 − 1 배) / (참 끝 자리 1/3 배 − 1 배), 끝 자리 예측 = 마지막 보인 참 자리 + v̂ · 남은 칸

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P predictor_lab/scaffold/adapter_step1_velocity.py            # → predictor_lab/scaffold/results/adapter_step1.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
STAGE0 = Path("/data/hyuntak/project/2026/2027_cvpr/nfs_cache/scaffold_stage0")
RES = REPO / "predictor_lab/scaffold/results"
SPEEDS = [("S1", "1/3"), ("T1", "2/3"), ("orig", "1")]
G, NC = 16, 8
R = 3                     # 창 반경 (토큰) — 1 배 속도 최대 ~1.1 토큰/칸 × k 3 = 3.3 을 덮는다
KS = (1, 2, 3)
VIS_OK = 0.9


def flow(z: torch.Tensor, k: int) -> torch.Tensor:
    """z (T, 256, D) → 칸 t → t+k 변위장 (T−k, 16, 16, 2) (열, 행) · 최고 유사도 (T−k, 16, 16)."""
    T = z.shape[0]
    zn = F.normalize(F.layer_norm(z.float(), z.shape[-1:]), dim=-1).view(T, G, G, -1)
    W = 2 * R + 1
    a, b = zn[:-k].reshape(T - k, G * G, -1), zn[k:].reshape(T - k, G * G, -1)
    full = torch.bmm(a, b.transpose(1, 2))                                        # (T−k, 256, 256) 모든 토큰 쌍 유사도 (행렬곱 한 번)
    yy, xx = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij")
    oy, ox = torch.meshgrid(torch.arange(-R, R + 1), torch.arange(-R, R + 1), indexing="ij")
    ny, nx = yy[..., None, None] + oy, xx[..., None, None] + ox                   # (16, 16, W, W) 창 안 이웃 좌표
    inside = (ny >= 0) & (ny < G) & (nx >= 0) & (nx < G)
    nid = (ny.clamp(0, G - 1) * G + nx.clamp(0, G - 1)).view(G * G, W * W)
    sims = torch.gather(full, 2, nid[None].expand(T - k, -1, -1)).view(T - k, G, G, W, W)
    sims = torch.where(inside[None], sims, torch.full_like(sims, -1.0))           # 화면 밖 = 유사도 −1
    flat = sims.view(T - k, G, G, -1)
    best, idx = flat.max(-1)
    iy, ix = idx // W, idx % W

    def sub(s_m, s_0, s_p):                                                       # 포물선 꼭짓점 (−0.5 ~ 0.5)
        den = s_m - 2 * s_0 + s_p
        off = torch.where(den.abs() > 1e-6, 0.5 * (s_m - s_p) / den, torch.zeros_like(den))
        return off.clamp(-0.5, 0.5)

    def at(yy, xx):
        yy = yy.clamp(0, W - 1); xx = xx.clamp(0, W - 1)
        return torch.gather(flat, -1, (yy * W + xx).unsqueeze(-1)).squeeze(-1)

    s0 = best
    ox = torch.where((ix > 0) & (ix < W - 1), sub(at(iy, ix - 1), s0, at(iy, ix + 1)), torch.zeros_like(s0))
    oy = torch.where((iy > 0) & (iy < W - 1), sub(at(iy - 1, ix), s0, at(iy + 1, ix)), torch.zeros_like(s0))
    d = torch.stack([ix.float() - R + ox, iy.float() - R + oy], -1)
    return d, best


def bilinear(field: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """field (16, 16, C) 를 토큰 좌표 xy (열, 행) 에서 이중선형으로."""
    x, y = np.clip(xy[0], 0, G - 1), np.clip(xy[1], 0, G - 1)
    x0, y0 = int(np.floor(x)), int(np.floor(y)); x1, y1 = min(x0 + 1, G - 1), min(y0 + 1, G - 1)
    fx, fy = x - x0, y - y0
    return ((1 - fx) * (1 - fy) * field[y0, x0] + fx * (1 - fy) * field[y0, x1] + (1 - fx) * fy * field[y1, x0] + fx * fy * field[y1, x1])


def main():
    out = {"window_radius": R, "gaps": list(KS), "vis_ok": VIS_OK, "by_speed": {}}
    est = {}
    D = {nm: np.load(STAGE0 / f"vob_{nm}.npz") for nm, _ in SPEEDS}
    cen = {nm: D[nm]["cen"] for nm, _ in SPEEDS}
    keep = np.abs(cen["S1"][:, -1, 0] - cen["orig"][:, -1, 0]) >= 1.5            # 1 판 0 단계 · 2 판과 같은 27 block
    for nm, lab in SPEEDS:
        Z, C, V = D[nm]["z"], cen[nm], D[nm]["vis"]
        N = len(Z)
        vhat = np.full((N, 2), np.nan); last = np.full(N, -1); vctx = np.full(N, np.nan); vfut = np.full(N, np.nan)
        conf = []
        for n in range(N):
            z = torch.from_numpy(Z[n]).float().reshape(NC, G * G, -1)
            vis_t = np.nonzero(V[n, :NC] >= VIS_OK)[0]
            if len(vis_t) < 2:
                continue
            lv = int(vis_t.max()); last[n] = lv
            cand = []
            for k in KS:
                d, best = flow(z, k)
                for t in range(max(0, lv - 4), lv - k + 1):
                    if V[n, t] >= VIS_OK and V[n, t + k] >= VIS_OK:
                        cand.append(bilinear(d[t].numpy(), C[n, t]) / k)
                        conf.append(float(bilinear(best[t].numpy()[..., None], C[n, t])[0]))
            if cand:
                vhat[n] = np.median(np.array(cand), 0)
            tv = vis_t[vis_t >= lv - 4]
            vctx[n] = np.polyfit(tv, C[n, tv, 0], 1)[0] if len(tv) >= 2 else np.nan
            vfut[n] = np.polyfit(np.arange(NC, 16), C[n, NC:, 0], 1)[0]
        est[nm] = dict(vhat=vhat, last=last)
        rr = (vhat[:, 0] / vctx)[keep]; rs = (vhat[:, 0] / vfut)[keep]
        fail = ~np.isfinite(rs) | (np.abs(rs - 1) > 0.5)
        out["by_speed"][lab] = {
            "true_ctx_speed_median": round(float(np.nanmedian(np.abs(vctx[keep]))), 3),
            "est_speed_median": round(float(np.nanmedian(np.abs(vhat[keep, 0]))), 3),
            "read_ratio_median": round(float(np.nanmedian(rr)), 3), "read_ratio_iqr": [round(float(x), 3) for x in np.nanpercentile(rr, [25, 75])],
            "read_n_off50": int((~np.isfinite(rr) | (np.abs(rr - 1) > 0.5)).sum()),
            "roll_slope_ratio_median": round(float(np.nanmedian(rs)), 3), "roll_n_fail": int(fail.sum()), "n_nan": int((~np.isfinite(rs)).sum()),
            "wrong_direction": int((np.sign(vhat[keep, 0]) != np.sign(vctx[keep])).sum()),
            "row_speed_abs_median": round(float(np.nanmedian(np.abs(vhat[keep, 1]))), 3),
            "match_similarity_median": round(float(np.median(conf)), 3) if conf else None,
            "per_block_read_ratio": [None if not np.isfinite(x) else round(float(x), 2) for x in rr]}
    # 따라감 비 (옛 지표)
    def end(nm):
        e = est[nm]; Cn = cen[nm]
        return np.array([Cn[n, e["last"][n], 0] + e["vhat"][n, 0] * (15 - e["last"][n]) if e["last"][n] >= 0 else np.nan for n in range(len(Cn))])
    fr = ((end("S1") - end("orig")) / (cen["S1"][:, -1, 0] - cen["orig"][:, -1, 0]))[keep]
    out["follow_ratio_old_metric"] = {"median": round(float(np.nanmedian(fr)), 3), "sum_ratio": round(float(np.nansum((end("S1") - end("orig"))[keep]) / np.sum((cen["S1"][:, -1, 0] - cen["orig"][:, -1, 0])[keep])), 3),
                                      "n_negative": int((fr < 0).sum()), "reference_pft": 0.167}
    out["verdict_roll"] = all(0.7 <= out["by_speed"][lab]["roll_slope_ratio_median"] <= 1.3 and out["by_speed"][lab]["roll_n_fail"] <= 3 for _, lab in SPEEDS)
    out["n_block"] = int(keep.sum())
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RES / "adapter_step1.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "by_speed"}, ensure_ascii=False))
    for lab, v in out["by_speed"].items():
        print(lab, {k: x for k, x in v.items() if k != "per_block_read_ratio"})
        print("   per block read ratio", v["per_block_read_ratio"])


if __name__ == "__main__":
    sys.exit(main())
