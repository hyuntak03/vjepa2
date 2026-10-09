#!/usr/bin/env python3
"""SC2v2 — encoder 와 독립인 픽셀 optical flow 를 유사 정답으로 한 라벨 없는 지평 곡선 (2026-09-25).

왜: SC2 1 판 (e_ssv2_track.py) 의 유사 정답 = "target encoder 템플릿 argmax 궤적" 이 SSv2 에서 무효였다
  (슬롯 사이 평균 5.5 칸 점프, 55 % 가 ≥ 4 칸; 질의 선택 에너지가 clip 사이 평평 0.99–1.04).
그래서 유사 정답을 encoder 와 무관한 Farneback flow 추적으로 바꾸고, encoder 템플릿 판독은 **flow 와 교차 검증** 한다.

clip 배치 (두 모드 같음): 32 장 = 문맥 16 장 (8 튜블릿) + 미래 16 장 (8 튜블릿), 256².
  ssv2 : e_ssv2_extract.load_clip 과 같은 전처리 (균등 32 장).
  v3   : RollOut_v3 raw 0..63 stride 2 (P2 s2_C16 과 같은 프레임). GT px (288 기준, CELL 18) → 256 격자로 환산해 보정에 쓴다.
flow: 회색 256² 연속 프레임 쌍마다 cv2.calcOpticalFlowFarneback (0.5, 3, 15, 3, 5, 1.2). 점 추적 = 현재 위치 주변 16×16 창의 flow 중앙값으로 한 프레임씩 이동.
질의 q (인과): pick_query — 문맥 끝 (프레임 14→15) flow 크기 최대 토큰 주변의 움직임 화소 중심을 한 프레임 민 자리 = 출발점 (프레임 15), q = 그 토큰.
  flow 유사 정답 f_i = 슬롯 i (프레임 16+2i, 17+2i) 추적 위치 평균의 칸.   문맥 끝 속도 v = 프레임 12→15 추적 변위 / 3 (px/프레임).
판독 (슬롯 i): 템플릿 Tc = hc_7[q] (인과, 문맥 16 장만 본 target) · T = h_7[q] (표준 양방향).
  cos 지도 전부 저장 (float16): [h·Tc, p·Tc, h·T, p·T] × 8 슬롯 × 256.
저장: cos.npy (n, 4, 8, 256) · flow.npy (n, 8, 2) px (x, y; 256 격자) · q.npy (n, 8) = (qy, qx, 에너지, |v|, 출발점 x, y px, vx, vy); v = 문맥 끝 속도 px/프레임 (프레임 12→15 역추적)
      v3 이면 gt.npy (n, 8, 2) px (256 격자), gtq.npy (n, 2) 문맥 끝 GT px.
분석: e_flowtrack_analyze.py

  ARLIB_PRED_CKPT=<ckpt> python e_flowtrack.py --mode ssv2 --gpus 8 --out <dir>
  python e_flowtrack.py --mode v3 --every 4 --gpus 8 --out <dir>
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arlib  # noqa: E402
from h4_single_query_v3 import idx_range  # noqa: E402
from e_ssv2_extract import load_clip, SSV2, MEAN, STD  # noqa: E402

S_TOK, D, NF, TC, G, PX = 256, 1280, 32, 8, 16, 16
V3_FR = list(range(0, 64, 2)); V3_CELL = 18.0; V3_SCALE = 256.0 / (V3_CELL * G)
FREE = ("flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc")


def to_gray(x):
    """정규화된 (3, T, H, W) → uint8 회색 (T, H, W)."""
    import cv2
    u = ((x * STD + MEAN).clamp(0, 1) * 255).byte().permute(1, 2, 3, 0).numpy()
    return np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in u])


def flows(g):
    import cv2
    return np.stack([cv2.calcOpticalFlowFarneback(g[t], g[t + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0) for t in range(len(g) - 1)])   # (T-1, H, W, 2)


def med_flow(fl, x, y, r=PX // 2):
    H, W = fl.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    w = fl[max(0, yi - r):min(H, yi + r), max(0, xi - r):min(W, xi + r)].reshape(-1, 2)
    return np.median(w, 0) if len(w) else np.zeros(2)


def track(fl, x, y, t0, t1):
    """프레임 t0 의 (x, y) 를 t1 까지 (t1 > t0 정방향, t1 < t0 역방향 — 역은 flow 부호 반전 근사) 추적. 프레임별 위치 dict."""
    pos = {t0: (x, y)}
    if t1 > t0:
        for t in range(t0, t1):
            d = med_flow(fl[t], x, y); x, y = float(np.clip(x + d[0], 0, 255)), float(np.clip(y + d[1], 0, 255)); pos[t + 1] = (x, y)
    else:
        for t in range(t0, t1, -1):
            d = med_flow(fl[t - 1], x, y); x, y = float(np.clip(x - d[0], 0, 255)), float(np.clip(y - d[1], 0, 255)); pos[t - 1] = (x, y)
    return pos


def pick_query(fl):
    """문맥 끝 flow (프레임 14→15, fl[14]) 크기가 최대인 토큰 q0 → 그 주변 3×3 토큰 창에서 크기 ≥ 창 최대의 절반인 화소의 크기 가중 중심
    (= 프레임 14 의 움직이는 것) → 그 자리 flow 로 한 프레임 밀어 프레임 15 출발점 (cx, cy). 질의 토큰 q = 출발점이 든 토큰.
    (토큰 중심 출발은 배경 flow 를 타서 v3 GT 1 칸 안 17–31 %; 12→15 합산 중심은 2–3 프레임 뒤처져 41–51 % 였다.)"""
    mag = np.linalg.norm(fl[14], axis=-1)
    e = mag.reshape(G, PX, G, PX).mean((1, 3)).reshape(-1); q0 = int(e.argmax()); y0, x0 = (q0 // G) * PX, (q0 % G) * PX
    ya, yb, xa, xb = max(0, y0 - PX), min(256, y0 + 2 * PX), max(0, x0 - PX), min(256, x0 + 2 * PX)
    w = mag[ya:yb, xa:xb]; m = np.where(w >= 0.5 * w.max(), w, 0.0)
    yy, xx = np.mgrid[ya:yb, xa:xb]
    cy, cx = float((yy * m).sum() / m.sum()) + 0.5, float((xx * m).sum() / m.sum()) + 0.5
    d = med_flow(fl[14], cx, cy); cx, cy = float(np.clip(cx + d[0], 0, 255)), float(np.clip(cy + d[1], 0, 255))
    q = int(min(cy // PX, G - 1)) * G + int(min(cx // PX, G - 1))
    return e, q, cx, cy


def _worker(rank, world, a, items):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    cfg = arlib.v3_cfg(16, 16); bundle = arlib.build_bundle(cfg, dev, window=NF); pred = bundle.predictor; ac = arlib.autocast_ctx(cfg)
    O = {nm: np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="r+") for nm in (["cos", "flow", "q"] + (["gt", "gtq"] if a.mode == "v3" else []))}
    ci = idx_range(0, TC, 1, dev); ti = idx_range(TC, TC, 1, dev)
    if a.mode == "v3":
        ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))); recs = ds64.records
        from h23_extract_v3 import arr
    mine = list(range(rank, len(items), world)); t0 = time.time()
    for n_done, row in enumerate(mine):
        if a.mode == "ssv2":
            x0 = load_clip(items[row])
        else:
            i = items[row]; x0 = ds64.clip(i)[:, V3_FR].float()
            raw = recs[i].raw; gx, gy = arr(raw["px_x_by_sample"]) * V3_SCALE, arr(raw["px_y_by_sample"]) * V3_SCALE
            O["gt"][row] = [[(gx[V3_FR[16 + 2 * k]] + gx[V3_FR[17 + 2 * k]]) / 2, (gy[V3_FR[16 + 2 * k]] + gy[V3_FR[17 + 2 * k]]) / 2] for k in range(8)]
            O["gtq"][row] = [(gx[V3_FR[14]] + gx[V3_FR[15]]) / 2, (gy[V3_FR[14]] + gy[V3_FR[15]]) / 2]
        g = to_gray(x0.cpu()); fl = flows(g)                                              # (31, 256, 256, 2)
        e, q, cx, cy = pick_query(fl); qy, qx = q // G, q % G
        fw = track(fl, cx, cy, 15, 31); bw = track(fl, cx, cy, 15, 12)
        O["flow"][row] = [[(fw[16 + 2 * k][0] + fw[17 + 2 * k][0]) / 2, (fw[16 + 2 * k][1] + fw[17 + 2 * k][1]) / 2] for k in range(8)]
        vx, vy = (cx - bw[12][0]) / 3.0, (cy - bw[12][1]) / 3.0
        O["q"][row] = [qy, qx, float(e[q]), float(np.hypot(vx, vy)), cx, cy, vx, vy]
        x = x0[None].to(dev, bundle.dtype)
        with torch.inference_mode(), ac():
            h = F.layer_norm(bundle.target_encoder(x), (D,)).float().reshape(16, S_TOK, D)
            hc = F.layer_norm(bundle.target_encoder(x[:, :, :16]), (D,)).float().reshape(8, S_TOK, D)
            z = bundle.context_encoder(x, masks=[ci])
            p = pred(z, ci, ti, mask_index=0).float().reshape(8, S_TOK, D)
        Tc, T = F.normalize(hc[7, q], dim=-1), F.normalize(h[7, q], dim=-1)
        hn, pn = F.normalize(h[8:], dim=-1), F.normalize(p, dim=-1)
        O["cos"][row] = torch.stack([hn @ Tc, pn @ Tc, hn @ T, pn @ T]).half().cpu().numpy()
        if rank == 0 and n_done % 20 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분 남은 {el/(n_done+1)*(len(mine)-n_done-1)/60:.1f} 분", flush=True)
    for v_ in O.values(): v_.flush()
    print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("ssv2", "v3"), default="ssv2"); ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count()); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=4, help="v3: 가능 변이 중 몇 개마다 하나")
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.mode == "ssv2":
        import decord
        items, labels = [], []
        for lab, c in enumerate(("left", "right")):
            for f in sorted((SSV2 / c).glob("*.mp4")):
                if len(decord.VideoReader(str(f), num_threads=1)) >= NF: items.append(f); labels.append(lab)
        meta = dict(files=[str(f) for f in items], labels=labels)
    else:
        R = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))).records
        items = [i for i, r in enumerate(R) if r.plausible == "1" and r.raw["scenario"] in FREE][:: a.every]
        meta = dict(video_ids=[R[i].video_id for i in items], traj=[[R[i].raw["scenario"], R[i].raw["primary"], R[i].raw["secondary"]] for i in items])
    if a.limit:
        k = np.linspace(0, len(items) - 1, a.limit).round().astype(int); items = [items[i] for i in k]
        meta = {kk: [vv[i] for i in k] for kk, vv in meta.items()}
    n = len(items)
    shapes = dict(cos=((n, 4, 8, 256), np.float16), flow=((n, 8, 2), np.float32), q=((n, 8), np.float32))
    if a.mode == "v3": shapes.update(gt=((n, 8, 2), np.float32), gtq=((n, 2), np.float32))
    for nm, (sh, dt) in shapes.items():
        m = np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="w+", dtype=dt, shape=sh); m[:] = np.nan; del m
    json.dump(dict(n=n, mode=a.mode, predictor=os.environ.get("ARLIB_PRED_CKPT", "release"), cos_rows=["h.Tc", "p.Tc", "h.T", "p.T"], **meta),
              open(Path(a.out) / "meta.json", "w"))
    print("clips", n, flush=True)
    torch.set_num_threads(2)
    mp.spawn(_worker, args=(a.gpus, a, items), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
