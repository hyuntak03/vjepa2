#!/usr/bin/env python3
"""E-SSv2 추출 — 실영상 (SSv2 좌/우 운동 722 clip) 에서 encoder 는 읽고, predictor 는 상태를 진전시키는가.

clip: 원본 전체에서 균등 32 장 (np.linspace; 원본 ≥ 32 프레임만 → 프레임 복제 없음) → 256×256 (정사각 눌러 넣기,
      bilinear antialias=False) → ImageNet 정규화. 문맥 = 앞 16 장 (8 튜블릿), 미래 = 뒤 16 장 (8 튜블릿).
      역재생 = 같은 32 장의 순서만 뒤집음 (방향 라벨이 뒤집힌다; 문맥은 원래의 뒤 절반을 거꾸로 튼 것).
표현 (수치 경로 = 표준 채점: fp32 + autocast fp16, mask_index 0; 문맥 encoder = online, 표적 = EMA):
  h   = LN(target_encoder(32 장))                  (16, 256, D)   표준 표적 (양방향 — 미래 정보가 문맥 슬롯에 번진다)
  hc  = LN(target_encoder(앞 16 장))               ( 8, 256, D)   인과 표적 (문맥만) → 공정한 복사 기준선 hc[7]
  z   = context_encoder(32 장, masks=문맥)          ( 8, 256, D)   predictor 입력
  zf  = LN(context_encoder(32 장, 마스크 없음))     (16, 256, D)   encoder 대조 (시간 정렬이 대각이어야 한다)
  p   = predictor(z)                               ( 8, 256, D)
저장 (clip × {정, 역}):
  feats  (n, 2, 56, D) fp16  토큰 평균 — 순서 z0..7 | h0..15 | hc0..7 | p0..7 | zf0..15   (방향 probe 용)
  featd  같은 것을 **동적 토큰** (h 의 32 장 시간 분산 상위 25 %) 평균으로
  D_p    (n, 2, 8, 16)  mean|p_i − h_j|  (동적 토큰)       D_pa  같은 것을 전 토큰
  D_z    (n, 2, 8, 16)  mean|zf_{8+i} − h_j| (동적 토큰)  — encoder 대조
  D_pn   (n, 2, 8, 16)  mean|p_i − h^B_j| (동적 토큰 = A 의 것) — B = 같은 rank 가 바로 앞에 처리한 clip (귀무)
  cmap   (n, 2, 3, 8, 256) fp16  토큰별 변화량: ‖h_{8+i} − h_7‖₁/D, ‖p_i − h_7‖₁/D, ‖p_i − hc_7‖₁/D  (변화 지도 일치)
  l1     (n, 2, 8, 6)  동적 토큰에서 mean|x − h_{8+i}|,  x ∈ {p_i, h_7 (새는 복사), hc_7 (인과 복사), zf_{8+i}},
                       + 전 토큰에서 p_i, hc_7
  E_ssv2 계획·공식: auto_research/Archive/E_SSV2_PLAN_2026-09-25.md

  sub.sh -J ar_ssv2 -o auto_research/logs/ssv2_%j.out --export=ALL,CMD="python auto_research/scripts/e_ssv2_extract.py --gpus 8" run8.sbatch
  python auto_research/scripts/e_ssv2_extract.py --gpus 1 --limit 4 --out /tmp/ssv2_smoke
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arlib  # noqa: E402
from h4_single_query_v3 import idx_range  # noqa: E402

S_TOK, D, NF, TC, G = 256, 1280, 32, 8, 16
SSV2 = Path("/data2/local_datasets/world/world_analysis/ssv2_VP_default")
OUT_DEFAULT = "/data2/local_datasets/world/world_analysis/cache/auto_research/ssv2_e"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


def load_clip(f):
    import decord
    vr = decord.VideoReader(str(f), num_threads=2)
    idx = np.linspace(0, len(vr) - 1, NF).round().astype(int)
    x = torch.from_numpy(vr.get_batch(list(idx)).asnumpy()).permute(3, 0, 1, 2).float() / 255.0
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).transpose(0, 1)
    return (x - MEAN) / STD


def _worker(rank, world, a, files):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    cfg = arlib.v3_cfg(16, 16); bundle = arlib.build_bundle(cfg, dev, window=NF); pred = bundle.predictor
    ac = arlib.autocast_ctx(cfg)
    M = {nm: np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="r+") for nm in ("feats", "featd", "D_p", "D_pa", "D_z", "D_pn", "cmap", "l1", "l1b", "cmapb", "D_o1", "D_o2", "D_oa", "D_oe", "proj")}
    Rp = torch.tensor(np.random.default_rng(123).standard_normal((D, 64)) / 8.0, dtype=torch.float32, device=dev)   # 고정 무작위 투영 (JL)
    prevp = None
    mine = list(range(rank, len(files), world)); prevh = None; t0 = time.time()
    ci = idx_range(0, TC, 1, dev); ti = idx_range(TC, TC, 1, dev)
    for n_done, row in enumerate(mine):
        x0 = load_clip(files[row])
        for di, x in enumerate((x0, torch.flip(x0, dims=[1]))):
            x = x[None].to(dev, bundle.dtype)
            with torch.inference_mode(), ac():
                h = F.layer_norm(bundle.target_encoder(x), (D,)).float().reshape(16, S_TOK, D)
                hc = F.layer_norm(bundle.target_encoder(x[:, :, :16]), (D,)).float().reshape(8, S_TOK, D)
                z = bundle.context_encoder(x, masks=[ci])
                zf = F.layer_norm(bundle.context_encoder(x).float(), (D,)).reshape(16, S_TOK, D)
                p = pred(z, ci, ti, mask_index=0).float().reshape(8, S_TOK, D)
                zz = F.layer_norm(z.float(), (D,)).reshape(8, S_TOK, D)
            var = h.var(0).mean(-1)                                            # (256,) 시간 분산
            dyn = var >= torch.quantile(var, 0.75)
            allrep = torch.cat([zz, h, hc, p, zf], 0)                          # (56, 256, D)
            M["feats"][row, di] = allrep.mean(1).half().cpu().numpy()
            M["featd"][row, di] = allrep[:, dyn].mean(1).half().cpu().numpy()
            hd = h[:, dyn]                                                     # (16, nd, D)
            M["D_p"][row, di] = (p[:, None, dyn] - hd[None]).abs().mean((-1, -2)).cpu().numpy()
            M["D_pa"][row, di] = (p[:, None] - h[None]).abs().mean((-1, -2)).cpu().numpy()
            M["D_z"][row, di] = (zf[8:, None, dyn] - hd[None]).abs().mean((-1, -2)).cpu().numpy()
            if prevh is not None:
                M["D_pn"][row, di] = (p[:, None, dyn] - prevh[:, dyn][None]).abs().mean((-1, -2)).cpu().numpy()
            M["cmap"][row, di, 0] = (h[8:] - h[7:8]).abs().mean(-1).half().cpu().numpy()
            M["cmap"][row, di, 1] = (p - h[7:8]).abs().mean(-1).half().cpu().numpy()
            M["cmap"][row, di, 2] = (p - hc[7:8]).abs().mean(-1).half().cpu().numpy()
            fut = h[8:, dyn]
            M["l1"][row, di] = torch.stack([(p[:, dyn] - fut).abs().mean((-1, -2)), (h[7:8, dyn] - fut).abs().mean((-1, -2)),
                                            (hc[7:8, dyn] - fut).abs().mean((-1, -2)), (zf[8:, dyn] - fut).abs().mean((-1, -2)),
                                            (p - h[8:]).abs().mean((-1, -2)), (hc[7:8] - h[8:]).abs().mean((-1, -2))], -1).cpu().numpy()
            # 흐림 기준선 (L1 은 평균 회귀를 보상한다 — p 의 복사 대비 이득이 '진전' 인지 '흐림' 인지 가른다)
            hcm = hc.mean(0, keepdim=True)                                                        # 문맥 시간 평균
            hcb = F.avg_pool2d(hc[7].reshape(G, G, D).permute(2, 0, 1)[None], 3, 1, 1, count_include_pad=False)[0].permute(1, 2, 0).reshape(1, S_TOK, D)  # 공간 3×3 흐림
            hmix = 0.5 * (hc[7:8] + hcm)
            pm = p.mean(0, keepdim=True)                                                          # p 의 슬롯 평균 (슬롯 구분을 지운 p)
            cols = []
            for bl in (hcm, hcb, hmix, pm):
                cols += [(bl[:, dyn] - fut).abs().mean((-1, -2)), (bl - h[8:]).abs().mean((-1, -2))]
            M["l1b"][row, di] = torch.stack(cols, -1).cpu().numpy()                               # (8, 8): [hcm dyn, hcm all, hcb dyn, hcb all, mix dyn, mix all, pmean dyn, pmean all]
            M["cmapb"][row, di, 0] = (hc[7] - hc[6]).abs().mean(-1).half().cpu().numpy()           # 마지막 관측 운동 지도 (인과)
            M["cmapb"][row, di, 1] = (h[7] - h[6]).abs().mean(-1).half().cpu().numpy()             # 같은 것을 표준 h 로
            # 양성 대조: 흐리지만 진전하는 가짜 예측도 시간 정렬이 대각으로 나오는가
            o1 = F.avg_pool2d(h[8:].reshape(8, G, G, D).permute(0, 3, 1, 2), 3, 1, 1, count_include_pad=False).permute(0, 2, 3, 1).reshape(8, S_TOK, D)   # 공간 흐림 진실
            o2 = 0.5 * h[8:] + 0.5 * hcm                                                              # 진실을 문맥 평균으로 반 희석
            M["D_o1"][row, di] = (o1[:, None, dyn] - hd[None]).abs().mean((-1, -2)).cpu().numpy()
            M["D_o2"][row, di] = (o2[:, None, dyn] - hd[None]).abs().mean((-1, -2)).cpu().numpy()
            # 토큰 수준 양성 대조 2 (적대 검증 권고): 복사 쪽으로 흐린 진실 a∈{.7,.8,.9}, 진실 + 다른 clip 의 p 오차
            for k_, a_ in enumerate((0.7, 0.8, 0.9)):
                oa = (1 - a_) * h[8:] + a_ * hc[7:8]
                M["D_oa"][row, di, k_] = (oa[:, None, dyn] - hd[None]).abs().mean((-1, -2)).cpu().numpy()
            if prevp is not None:
                oe = h[8:] + (prevp[0] - prevp[1])                                            # 직전 clip 의 (p − h_fut) 오차
                M["D_oe"][row, di] = (oe[:, None, dyn] - hd[None]).abs().mean((-1, -2)).cpu().numpy()
            # 토큰 수준 β 용 투영 저장: h 0..15 | p 0..7 | hc_7  → (25, 256, 64)
            M["proj"][row, di] = (torch.cat([h, p, hc[7:8]], 0) @ Rp).half().cpu().numpy()
            if di == 0: nexth = h; nextp = (p, h[8:])
        prevh = nexth; prevp = nextp
        if rank == 0 and n_done % 20 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분 남은 {el/(n_done+1)*(len(mine)-n_done-1)/60:.1f} 분", flush=True)
    for v in M.values(): v.flush()
    print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--root", default=str(SSV2))
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    import decord
    files, labels, nfr = [], [], []
    for lab, c in enumerate(("left", "right")):
        for f in sorted((Path(a.root) / c).glob("*.mp4")):
            n = len(decord.VideoReader(str(f), num_threads=1))
            if n >= NF: files.append(f); labels.append(lab); nfr.append(n)
    if a.limit:
        k = np.linspace(0, len(files) - 1, a.limit).round().astype(int); files = [files[i] for i in k]; labels = [labels[i] for i in k]; nfr = [nfr[i] for i in k]
    n = len(files)
    for nm, sh, dt in (("feats", (n, 2, 56, D), np.float16), ("featd", (n, 2, 56, D), np.float16), ("D_p", (n, 2, 8, 16), np.float32),
                       ("D_pa", (n, 2, 8, 16), np.float32), ("D_z", (n, 2, 8, 16), np.float32), ("D_pn", (n, 2, 8, 16), np.float32),
                       ("cmap", (n, 2, 3, 8, S_TOK), np.float16), ("l1", (n, 2, 8, 6), np.float32),
                       ("l1b", (n, 2, 8, 8), np.float32), ("cmapb", (n, 2, 2, S_TOK), np.float16),
                       ("D_o1", (n, 2, 8, 16), np.float32), ("D_o2", (n, 2, 8, 16), np.float32),
                       ("D_oa", (n, 2, 3, 8, 16), np.float32), ("D_oe", (n, 2, 8, 16), np.float32), ("proj", (n, 2, 25, S_TOK, 64), np.float16)):
        v = np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="w+", dtype=dt, shape=sh); v[:] = np.nan; del v
    json.dump(dict(n=n, files=[str(f) for f in files], labels=labels, n_frames=nfr,
                   feats_order="z0..7 | h0..15 | hc0..7 | p0..7 | zf0..15", dirs=["forward", "reversed"]),
              open(Path(a.out) / "meta.json", "w"))
    print("clips", n, "left", labels.count(0), "right", labels.count(1), flush=True)
    mp.spawn(_worker, args=(a.gpus, a, files), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
