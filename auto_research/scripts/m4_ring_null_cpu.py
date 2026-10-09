#!/usr/bin/env python3
"""M4 거리 맞춤 (ring) 귀무 — CPU fp32 부분집합 (2026-09-25, 적대 검증 코드 `verify0925/M4/m4_null_cpu.py` 를 레포로 옮김).

왜: `m4_locality_stride_v3.py` 는 attention 을 key 집합 질량 5 개로만 저장해서 "물체를 본다" 와 "가까운 key 를 본다" (공간 국소성) 를
가를 수 없다. 여기서는 query 행 전체 attention 을 CPU 로 다시 뽑아 같은 거리의 비물체 key 밀도를 귀무로 쓴다.

선택: M4 의 588 clip (v3_m4/meta.json) 에서 궤적마다 첫 clip (궤적 84 개) → --n-clips 로 균등 솎음 (기본 84 = 궤적당 1).
      --chunk i --nchunks k 로 나눠 돌리고 m4_analyze.py 가 합친다 (srun6 60 분 제한).
수치 경로: CPU fp32, autocast 없음 (GPU 원판은 fp32 + autocast fp16). predictor 는 릴리즈 (ARLIB_PRED_CKPT 가 있으면 그것).
표적 (템플릿): 표준 양방향 h = LN(target_encoder(창 전체)) — 적중 판정과 P-query 템플릿 모두.

query 두 종 (슬롯 t 마다 1 개):
  T = 진실 물체 칸의 미래 토큰 (M4 와 같음)
  P = p_t 에서 외형 템플릿 (문맥 마지막 튜블릿 h 의 물체 토큰, 미래 진실 안 씀) 의 cos argmax 칸 = p 가 스스로 물체를 둔 칸
key 집합 (문맥 튜블릿 u 마다, query 3×3 자기 기둥은 뺀다):
  obj    = 물체 3×3 (화면 안 튜블릿만; in_frame_by_sample 두 프레임 모두 > 0)
  objall = 물체 3×3 (화면 밖 튜블릿도 가장자리 칸으로 잘라 넣음 = GPU 원판 obj_traj 정의)
  ring   = obj key 마다 같은 튜블릿 · 같은 Chebyshev 거리의 비물체 key 평균 밀도 (obj key 수로 가중) — 공간 국소성 귀무
  refl   = 물체 궤적을 query 기준 점대칭한 3×3 (같은 거리, 물체 없음)
  비율은 모두 ×균등 (균등 = 그 층 문맥 질량 / 문맥 key 수, 실제 key 수로 나눈 밀도).  층별 값을 저장하고 층 평균은 분석에서.
head 별: T query 의 obj / ring ×U 를 (층, head) 로 저장 (Hook(per_head=True); head 평균은 원본 hook 과 같다).
재현 검사 (chunk 0 첫 clip): 원래 predictor p vs hook(kmask=None) p vs hook(kmask=전부 True) p 의 상대 L1.

  srun6.sh 8 24G <python> m4_ring_null_cpu.py --arm s1_C16 --chunk 0 --nchunks 3
출력: <out>/ringnull_<arm>_c<i>of<k>.json  (기본 out = /data2/.../auto_research/v3_m4_ringnull)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arlib  # noqa: E402
from predictor_hooks import Hook, patch_attention  # noqa: E402

C = Path("/data2/local_datasets/world/world_analysis/cache/auto_research")
G, S, D, CELL = 16, 256, 1280, 18.0
ARMS = {"s1_C16": (1, 16), "s2_C16": (2, 0), "s4_C8": (4, 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS)); ap.add_argument("--n-clips", type=int, default=84)
    ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--m4", default=str(C / "v3_m4")); ap.add_argument("--out", default=str(C / "v3_m4_ringnull"))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True); torch.set_num_threads(a.threads)
    s, r0 = ARMS[a.arm]
    fr = list(range(r0, 64, s)); nc = sum(f < 32 for f in fr) // 2; nf = (len(fr) - 2 * nc) // 2
    meta = json.load(open(Path(a.m4) / "meta.json")); arm_i = [x["name"] for x in meta["arms"]].index(a.arm)
    KN = np.load(Path(a.m4) / "knock.npy", mmap_mode="r")
    traj = ["|".join(map(str, t)) for t in meta["traj"]]
    first = {}
    for i, t in enumerate(traj): first.setdefault(t, i)
    pick = sorted(first.values()); pick = pick[:: max(1, len(pick) // a.n_clips)][: a.n_clips]
    mine = pick[a.chunk:: a.nchunks]
    t0 = time.time()
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))); recs = ds64.records; rid = {r.video_id: i for i, r in enumerate(recs)}
    dev = torch.device("cpu")
    bundle = arlib.build_bundle(arlib.v3_cfg(4, 60), dev, window=64); pred = bundle.predictor
    ci = torch.arange(0, nc * S)[None]; ti = torch.arange(nc * S, (nc + nf) * S)[None]
    repro = None
    if a.chunk == 0:                                                         # 원래 predictor vs hook (막음 없음 · 전부 True mask)
        xc = ds64.clip(rid[meta["video_ids"][mine[0]]])[None].to(dev, bundle.dtype)[:, :, fr]
        with torch.inference_mode():
            z = bundle.context_encoder(xc, masks=[ci]); p_ref = pred(z, ci, ti, mask_index=0).float()
    hook = Hook(per_head=True); patch_attention(pred, hook)
    if a.chunk == 0:
        with torch.inference_mode():
            hook.kmask = None; p_none = pred(z, ci, ti, mask_index=0).float()
            N = (nc + nf) * S; hook.kmask = torch.ones(N, N, dtype=torch.bool); p_true = pred(z, ci, ti, mask_index=0).float(); hook.kmask = None
        rel = lambda p: float(((p - p_ref).abs().mean() / p_ref.abs().mean()).item())
        repro = dict(clip=meta["video_ids"][mine[0]], arm=a.arm, dtype=str(bundle.dtype),
                     hook_none_vs_orig_rel_l1=rel(p_none), hook_alltrue_vs_orig_rel_l1=rel(p_true))
        print("repro", repro, flush=True)
    print("load", round(time.time() - t0), "s", "dtype", bundle.dtype, "threads", a.threads, "clips", len(mine), flush=True)
    ky, kx = np.divmod(np.arange(S), G)

    def cells(raw):
        x = np.array(raw["px_x_by_sample"].split(), float); y = np.array(raw["px_y_by_sample"].split(), float)
        inf = np.array(raw["in_frame_by_sample"].split(), float)
        out, infr = [], []
        for u in range(nc + nf):
            f0, f1 = fr[2 * u], fr[2 * u + 1]
            out.append((int(np.clip(np.floor((x[f0] + x[f1]) / 2 / CELL), 0, G - 1)), int(np.clip(np.floor((y[f0] + y[f1]) / 2 / CELL), 0, G - 1))))
            infr.append(min(inf[f0], inf[f1]) > 0)
        return out, infr

    def dens(Actx, q, ccells, cin):
        """Actx (..., nc, S) 한 query 의 문맥 attention (앞 축 = 층 또는 층×head). q=(x,y) → ×균등 dict."""
        qx, qy = q; lead = Actx.shape[:-2]
        cheb = np.maximum(np.abs(kx - qx), np.abs(ky - qy)); selfm = cheb <= 1
        unif = Actx.sum((-1, -2)) / (nc * S)
        objm = np.zeros((nc, S), bool); objall = np.zeros((nc, S), bool); reflm = np.zeros((nc, S), bool)
        for u in range(nc):
            cx, cy = ccells[u]
            w = (np.maximum(np.abs(kx - cx), np.abs(ky - cy)) <= 1) & ~selfm
            objall[u] = w
            if not cin[u]: continue
            objm[u] = w
            rx, ry = 2 * qx - cx, 2 * qy - cy
            if 0 <= rx < G and 0 <= ry < G:
                reflm[u] = (np.maximum(np.abs(kx - rx), np.abs(ky - ry)) <= 1) & ~selfm
        reflm &= ~objm
        if objm.sum() == 0: return None
        res = dict(obj=(Actx * objm).sum((-1, -2)) / objm.sum() / unif)
        res["objall"] = (Actx * objall).sum((-1, -2)) / objall.sum() / unif if objall.sum() else np.full(lead, np.nan)
        exp = np.zeros(lead); cnt = 0
        for u in range(nc):
            if not objm[u].any(): continue
            for dlt in np.unique(cheb[objm[u]]):
                ring = (cheb == dlt) & ~objm[u]
                if ring.sum() == 0: continue
                nobj = int((objm[u] & (cheb == dlt)).sum())
                exp = exp + Actx[..., u, ring].mean(-1) * nobj; cnt += nobj
        res["ring"] = (exp / cnt) / unif if cnt else np.full(lead, np.nan)
        res["refl"] = (Actx * reflm).sum((-1, -2)) / reflm.sum() / unif if reflm.sum() else np.full(lead, np.nan)
        res["objdist"] = float(np.broadcast_to(cheb, (nc, S))[objm].mean())
        res["ctx"] = Actx.sum((-1, -2))
        return res

    rows = []
    for n_done, mi in enumerate(mine):
        v = meta["video_ids"][mi]; i = rid[v]; raw = recs[i].raw
        cc, cin = cells(raw); lc = cc[nc - 1]; tcs = cc[nc:]
        xc = ds64.clip(i)[None].to(dev, bundle.dtype)[:, :, fr]
        with torch.inference_mode():
            h = F.layer_norm(bundle.target_encoder(xc), (D,)).float().reshape(nc + nf, S, D)
            z = bundle.context_encoder(xc, masks=[ci])
            hook.kmask = None; hook.store = []
            hook.rec_rows = torch.tensor([[nc * S + t * S + tcs[t][1] * G + tcs[t][0] for t in range(nf)]])
            p = pred(z, ci, ti, mask_index=0).float().reshape(nf, S, D)
            AT = torch.stack(hook.store, 1)[0].float().numpy()               # (L, H, nf, N)  (store 층마다 (B, H, R, N))
            app = h[nc - 1, lc[1] * G + lc[0]]
            pa, ht = [], []
            for t in range(nf):
                q = int((F.normalize(p[t], dim=-1) @ F.normalize(app, dim=-1)).argmax()); pa.append((q % G, q // G))
                b = int((F.normalize(p[t], dim=-1) @ F.normalize(h[nc + t, tcs[t][1] * G + tcs[t][0]], dim=-1)).argmax()); ht.append((b % G, b // G))
            hook.store = []
            hook.rec_rows = torch.tensor([[nc * S + t * S + pa[t][1] * G + pa[t][0] for t in range(nf)]])
            _ = pred(z, ci, ti, mask_index=0)
            AP = torch.stack(hook.store, 1)[0].float().numpy()               # (L, H, nf, N)
            hook.rec_rows = None; hook.store = []
        for t in range(nf):
            row = dict(vid=v, m4_row=int(mi), traj=traj[mi], law=meta["scenario"][mi], slot=t,
                       d=float(np.hypot(tcs[t][0] - lc[0], tcs[t][1] - lc[1])), dC=int(max(abs(tcs[t][0] - lc[0]), abs(tcs[t][1] - lc[1]))),
                       n_ctx_off=int(nc - sum(cin[:nc])),
                       d_papp=float(np.hypot(pa[t][0] - lc[0], pa[t][1] - lc[1])),
                       papp_to_true=int(max(abs(pa[t][0] - tcs[t][0]), abs(pa[t][1] - tcs[t][1]))),
                       hit=int(max(abs(ht[t][0] - tcs[t][0]), abs(ht[t][1] - tcs[t][1])) <= 1),
                       hit_gpu=float(KN[mi, arm_i, t, -1]))
            for lab, Aq, q in (("T", AT, tcs[t]), ("P", AP, pa[t])):
                Ah = Aq[:, :, t, :nc * S].reshape(Aq.shape[0], Aq.shape[1], nc, S)   # (L, H, nc, S)
                r = dens(Ah.mean(1), q, cc[:nc], cin[:nc])                            # head 평균 (원본 hook 과 같은 값)
                if r is None: continue
                row[f"{lab}_objdist"] = r["objdist"]; row[f"{lab}_ctx"] = float(r["ctx"].mean())
                for k in ("obj", "objall", "ring", "refl"):
                    row[f"{lab}_{k}_mean"] = float(np.nanmean(r[k])); row[f"{lab}_{k}_L"] = np.round(r[k], 4).tolist()
                if lab == "T":
                    rh = dens(Ah, q, cc[:nc], cin[:nc])                                # (L, H)
                    row["T_obj_LH"] = np.round(rh["obj"], 3).tolist(); row["T_ring_LH"] = np.round(rh["ring"], 3).tolist()
            rows.append(row)
        print(f"{n_done + 1}/{len(mine)} {time.time() - t0:.0f}s", flush=True)
        json.dump(dict(arm=a.arm, n_ctx=nc, n_fut=nf, chunk=a.chunk, nchunks=a.nchunks, n_clips_total=len(pick), done=n_done + 1, of=len(mine),
                       repro=repro, rows=rows), open(Path(a.out) / f"ringnull_{a.arm}_c{a.chunk}of{a.nchunks}.json", "w"))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
