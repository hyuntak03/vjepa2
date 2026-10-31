"""참 미래 h 로 배운 물체 위치 자를 AR predictor 의 예측 미래 p 에 토큰 단위로 건다 — VOBench (2026-10-26).

layer_velocity_probe (평균 풀링) 의 후속: "속도 정보가 있다" 와 "물체가 실제로 옮겨진 미래를 만든다" 를 가른다.

추출: 블록별 LN(target_encoder) 16 블록 (문맥 8 + 참 미래 8) = h, AR rollout 미래 8 블록 = p. 토큰 그대로 (256, 1280) float16.
자 (파라미터 1,281): 토큰마다 선형 점수 → 256 칸 softmax → 칸 중심의 기대값 = 위치 (칸, 1 칸 = 32 px @ 512).
  학습 = train 절반 영상의 h 16 블록 중 물체가 보이는 블록, 목표 = 참 위치 둘레 가우시안 (σ 0.75 칸) 의 CE.
  p 는 학습에 안 쓴다. h 와 p 는 같은 공간 (affine-free LN, 1280) 이다.
속도 = 걸음 사이 위치 차분 (칸 / 걸음). 속도용 학습 파라미터는 없다.
자 검증: (1) h test 절반 위치 오차 (2) 최대 질량 (물체다운 토큰이 있나; 균등 = 1/256)
  (3) 파라미터 없는 교차검증 — 마지막으로 보이는 문맥 블록에서 참 칸의 h 토큰을 본으로, 코사인 유사도 최대 칸.
분할: block 단위 50/50 (운동 층화, seed 0) — layer_velocity_probe 와 같다. 수치는 test 절반.

  PY=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python
  $PY z_research/scripts/analysis/vobench_ar_token_transfer.py extract --gpus 4 [--name vast70 --ckpt ... --tok_dir <scratch>]
  $PY z_research/scripts/analysis/vobench_ar_token_transfer.py analyze [--name vast70 --tok_dir <scratch>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import vobench_ar_layer_velocity_probe as lv  # noqa: E402  (rows_obj · block_split · FR · RES_CFG)

S, T, C, G, D = 256, 8, 8, 16, 1280
CELL = 32.0                                   # px @ 512


def out_dir(name):
    return REPO / f"z_analysis/ar_token_transfer/{name}"


def _worker(rank, world, name, ckpt, tok_dir):
    import torch, yaml
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import block_indices, encode_blocks
    from evals.world_model_analysis.data import WMADataset
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    cfg = yaml.safe_load(open(lv.RES_CFG))
    model = dict(cfg["model"], predictor_checkpoint=str(REPO / ckpt))
    bundle = build_from_config(model, dev)
    data = dict(cfg["data"], index_csv="index.csv", frames_root=str(lv.FR))
    ds = WMADataset(dict(data=data, model=model, features={"cache_dir": "/tmp"}, surprise={}))
    want = {r["video_id"] for r in lv.rows_obj()}
    ids = [i for i, r in enumerate(ds.records) if r.video_id in want]
    mine = ids[rank * len(ids) // world:(rank + 1) * len(ids) // world]
    H = np.lib.format.open_memmap(Path(tok_dir) / f"h{rank}.npy", "w+", np.float16, (len(mine), 2 * T, S, D))
    P = np.lib.format.open_memmap(Path(tok_dir) / f"p{rank}.npy", "w+", np.float16, (len(mine), T, S, D))
    vids = []; t0 = time.time()
    with torch.no_grad():
        for k in range(0, len(mine), 16):
            ch = mine[k:k + 16]
            clips = torch.stack([ds.clip(i) for i in ch]).to(dev, dtype=bundle.dtype)
            B = clips.size(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hb = encode_blocks(bundle.target_encoder, clips, 2)
                idx = block_indices(B, 2 * T, S, dev)
                p = bundle.predictor.rollout(hb[:, :C * S], idx[:, :C * S], T)
            H[k:k + B] = hb.float().view(B, 2 * T, S, D).half().cpu().numpy()
            P[k:k + B] = p.float().view(B, T, S, D).half().cpu().numpy()
            vids += [ds.records[i].video_id for i in ch]
            if rank == 0 and k % 160 == 0:
                print(f"[r0] {k + B}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    H.flush(); P.flush()
    np.save(Path(tok_dir) / f"vid{rank}.npy", np.array(vids))
    print(f"[r{rank}] done {len(mine)} ({time.time() - t0:.0f}s)", flush=True)


def extract(gpus, name, ckpt, tok_dir):
    import torch
    Path(tok_dir).mkdir(parents=True, exist_ok=True)
    torch.multiprocessing.spawn(_worker, args=(gpus, name, ckpt, tok_dir), nprocs=gpus, join=True)


# ---------------------------------------------------------------- 분석

def block_labels(vids):
    """블록 16 개마다 물체 위치 (칸, x 가로 · y 세로) 와 보임 여부 + 메타."""
    rows = {r["video_id"]: r for r in lv.rows_obj()}
    lab, XY, VIS, meta = {}, [], [], []
    for v in vids:
        r = rows[v]; arm = "A" if r["variant"] == "pos_a" else "B"
        b = lv.FR / Path(r["file_name"]).parent
        if b not in lab:
            lab[b] = json.load(open(b / "label.json"))["labels"]
        L = lab[b][arm]
        xy, vis = [], []
        for blk in range(2 * T):
            fr = [L[3 * (2 * blk)], L[3 * (2 * blk + 1)]]
            xy.append([np.mean([f["u"] for f in fr]) / CELL, np.mean([f["v"] for f in fr]) / CELL])
            vis.append(max(f["hidden"] for f in fr) < 0.5)
        XY.append(xy); VIS.append(vis)
        meta.append(dict(block=r["block_id"], motion=r["motion"], occ=r["occlusion"]))
    return np.array(XY, np.float32), np.array(VIS), meta


def load_tokens(tok_dir):
    d = Path(tok_dir); ranks = sorted(int(f.stem[3:]) for f in d.glob("vid*.npy"))
    vids = np.concatenate([np.load(d / f"vid{r}.npy") for r in ranks])
    H = [np.load(d / f"h{r}.npy", mmap_mode="r") for r in ranks]
    P = [np.load(d / f"p{r}.npy", mmap_mode="r") for r in ranks]
    return vids, H, P


def cat_rows(arrs, sel):
    """rank 별 memmap 을 이어 붙인 것처럼 행 sel 을 뽑는다."""
    off = np.cumsum([0] + [len(a) for a in arrs]); out = []
    for i in sel:
        r = np.searchsorted(off, i, side="right") - 1; out.append(arrs[r][i - off[r]])
    return np.stack(out)


def analyze(name, tok_dir, epochs=3000, sigma=0.75):
    import torch
    dev = "cuda"
    vids, Hs, Ps = load_tokens(tok_dir)
    XY, VIS, meta = block_labels(vids)
    n = len(meta)
    g = np.array([m["block"] for m in meta]); mot = np.array([m["motion"] for m in meta]); occ = np.array([m["occ"] for m in meta])
    tr = lv.block_split(g, mot); te = ~tr
    cy, cx = np.meshgrid(np.arange(G) + 0.5, np.arange(G) + 0.5, indexing="ij")
    cen = torch.tensor(np.stack([cx.ravel(), cy.ravel()], 1), dtype=torch.float32, device=dev)   # (256, 2) 칸 중심 (x, y)

    # ---- 자 학습: train 영상 h 의 보이는 블록
    itr = np.where(tr)[0]
    Htr = torch.tensor(cat_rows(Hs, itr), device=dev)                    # (ntr, 16, 256, 1280) fp16
    m_tr = torch.tensor(VIS[itr], device=dev)
    Xtr = Htr[m_tr]; Ytr = torch.tensor(XY[itr], device=dev)[m_tr]       # (nb, 256, 1280) · (nb, 2)
    del Htr
    mu = Xtr.float().mean((0, 1)); sd = Xtr.float().std((0, 1)) + 1e-6
    head = torch.nn.Linear(D, 1).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator(device=dev).manual_seed(0)

    def logits(X):                                                          # (b, 256, D) → (b, 256)
        return head(((X.float() - mu) / sd)).squeeze(-1)

    def soft_target(Y):
        d2 = ((cen[None] - Y[:, None]) ** 2).sum(-1)
        return torch.softmax(-d2 / (2 * sigma ** 2), -1)
    for it in range(epochs):
        b = torch.randint(0, len(Xtr), (512,), device=dev, generator=gen)
        loss = -(soft_target(Ytr[b]) * torch.log_softmax(logits(Xtr[b]), -1)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0:
            print(f"[readout] it {it} loss {loss.item():.3f}", flush=True)
    del Xtr

    def read(X):                                                            # (b, 256, D) → 기대값 xy (b,2), 최대 질량 (b,), argmax xy (b,2)
        out_xy, out_m, out_a = [], [], []
        with torch.no_grad():
            for k in range(0, len(X), 256):
                pr = torch.softmax(logits(X[k:k + 256]), -1)
                out_xy.append(pr @ cen); out_m.append(pr.max(-1).values); out_a.append(cen[pr.argmax(-1)])
        return torch.cat(out_xy).cpu().numpy(), torch.cat(out_m).cpu().numpy(), torch.cat(out_a).cpu().numpy()

    def template_xy(Xq, tmpl):                                              # 파라미터 없음: 코사인 최대 칸
        with torch.no_grad():
            s = torch.nn.functional.normalize(Xq.float(), dim=-1) @ torch.nn.functional.normalize(tmpl.float(), dim=-1)[:, :, None]
            a = s.squeeze(-1).argmax(-1)
        return cen[a].cpu().numpy(), s.squeeze(-1).max(-1).values.cpu().numpy()

    # ---- test 영상: h (16 블록) · p (8 블록) 읽기
    ite = np.where(te)[0]
    R = dict(pa_xy=np.zeros((len(ite), T, 2)), ha_xy=np.zeros((len(ite), 2 * T, 2)), h_xy=np.zeros((len(ite), 2 * T, 2)), h_m=np.zeros((len(ite), 2 * T)), p_xy=np.zeros((len(ite), T, 2)),
             p_m=np.zeros((len(ite), T)), pt_xy=np.full((len(ite), T, 2), np.nan), ht_xy=np.full((len(ite), T, 2), np.nan),
             pt_s=np.full((len(ite), T), np.nan), last=np.zeros(len(ite), int))
    for k in range(0, len(ite), 32):
        sel = ite[k:k + 32]
        Hb = torch.tensor(cat_rows(Hs, sel), device=dev); Pb = torch.tensor(cat_rows(Ps, sel), device=dev)
        B = len(sel)
        xy, m, a = read(Hb.flatten(0, 1)); R["h_xy"][k:k + B] = xy.reshape(B, 2 * T, 2); R["h_m"][k:k + B] = m.reshape(B, 2 * T)
        R["ha_xy"][k:k + B] = a.reshape(B, 2 * T, 2)
        xy, m, a = read(Pb.flatten(0, 1)); R["p_xy"][k:k + B] = xy.reshape(B, T, 2); R["p_m"][k:k + B] = m.reshape(B, T)
        R["pa_xy"][k:k + B] = a.reshape(B, T, 2)
        for j, i in enumerate(sel):
            vis_ctx = np.where(VIS[i, :C])[0]
            if len(vis_ctx) == 0:
                continue
            lb = int(vis_ctx[-1]); R["last"][k + j] = lb
            cxy = XY[i, lb]; ti = int(np.clip(np.floor(cxy[1]), 0, G - 1)) * G + int(np.clip(np.floor(cxy[0]), 0, G - 1))
            tmpl = Hb[j, lb, ti][None].expand(T, -1)
            R["pt_xy"][k + j], R["pt_s"][k + j] = template_xy(Pb[j], tmpl)
            R["ht_xy"][k + j], _ = template_xy(Hb[j, C:], tmpl)

    # ---- 요약
    XYt, VISt = XY[ite], VIS[ite]; mott, occt = mot[ite], occ[ite]
    res = {"n_test": int(len(ite)), "n_readout_train_blocks": int(m_tr.sum().item()), "uniform_mass": 1 / S}
    vf = VISt[:, C:]
    res["check_h_future_err_cells"] = float(np.linalg.norm(R["h_xy"][:, C:] - XYt[:, C:], axis=-1)[vf].mean())
    res["check_h_ctx_err_cells"] = float(np.linalg.norm(R["h_xy"][:, :C] - XYt[:, :C], axis=-1)[VISt[:, :C]].mean())
    res["check_h_template_err_cells"] = float(np.nanmean(np.linalg.norm(R["ht_xy"] - XYt[:, C:], axis=-1)[vf]))
    res["mass_h_future_visible"] = float(R["h_m"][:, C:][vf].mean())
    res["mass_h_future_hidden"] = float(R["h_m"][:, C:][~vf].mean()) if (~vf).any() else None
    groups = {}
    last_xy = XYt[np.arange(len(ite)), R["last"]]                          # 마지막으로 보인 문맥 블록의 참 위치
    for o, osel in (("not_late", occt != "late"), ("late", occt == "late")):
        for m_ in np.unique(mott):
            s = osel & (mott == m_)
            if not s.any():
                continue
            truth = XYt[s, C:]
            ax = truth[:, -1] - last_xy[s]; nrm = np.linalg.norm(ax, axis=-1, keepdims=True)
            ax = np.where(nrm > 1e-6, ax / np.maximum(nrm, 1e-6), 0)          # 영상마다 운동 축 (마지막 관측 → 참 t7)
            prog = lambda xy: ((xy - last_xy[s][:, None]) * ax[:, None]).sum(-1)   # 축 위 진행 (칸)
            d = dict(n=int(s.sum()),
                     true_prog=prog(truth).mean(0).tolist(),
                     p_prog=prog(R["p_xy"][s]).mean(0).tolist(),
                     h_prog=prog(R["h_xy"][s, C:]).mean(0).tolist(),
                     pa_prog=prog(R["pa_xy"][s]).mean(0).tolist(), ha_prog=prog(R["ha_xy"][s, C:]).mean(0).tolist(),
                     pa_err=np.linalg.norm(R["pa_xy"][s] - truth, axis=-1).mean(0).tolist(),
                     ctr_last=float(np.linalg.norm(last_xy[s] - G / 2, axis=-1).mean()),
                     ctr_true=np.linalg.norm(truth - G / 2, axis=-1).mean(0).tolist(),
                     ctr_p=np.linalg.norm(R["p_xy"][s] - G / 2, axis=-1).mean(0).tolist(),
                     ctr_pa=np.linalg.norm(R["pa_xy"][s] - G / 2, axis=-1).mean(0).tolist(),
                     dir_to_ctr=float(np.mean((((G / 2) - last_xy[s]) * ax).sum(-1))),
                     pt_prog=np.nanmean(prog(R["pt_xy"][s]), 0).tolist(),
                     p_err=np.linalg.norm(R["p_xy"][s] - truth, axis=-1).mean(0).tolist(),
                     h_err=np.linalg.norm(R["h_xy"][s, C:] - truth, axis=-1).mean(0).tolist(),
                     pt_err=np.nanmean(np.linalg.norm(R["pt_xy"][s] - truth, axis=-1), 0).tolist(),
                     p_mass=R["p_m"][s].mean(0).tolist(), h_mass=R["h_m"][s, C:].mean(0).tolist(),
                     pt_cos=np.nanmean(R["pt_s"][s], 0).tolist(),
                     true_vis=VISt[s, C:].mean(0).tolist())
            for k_ in ("true", "p", "h", "pt", "pa"):                                # 속도 = 진행 차분 (칸 / 걸음), t0 은 마지막 관측에서
                pr = np.r_[0.0, d[f"{k_}_prog"]]
                d[f"{k_}_vel"] = np.diff(pr).tolist()
            groups[f"{o}/{m_}"] = d
    res["groups"] = groups
    od = out_dir(name); od.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(od / "transfer.json", "w"), indent=1)
    torch.save(dict(head=head.state_dict(), mu=mu.cpu(), sd=sd.cpu(), sigma=sigma), od / "readout.pt")
    print(json.dumps({k: v for k, v in res.items() if k != "groups"}, indent=1))
    f = lambda a: " ".join(f"{x:5.2f}" for x in a)
    for k, d in groups.items():
        print(f"\n{k} (n {d['n']}) 마지막 관측→중심 (운동 축 성분) {d['dir_to_ctr']:+.2f}\n  true prog {f(d['true_prog'])}\n  p    prog {f(d['p_prog'])}\n  pArg prog {f(d['pa_prog'])}\n  pT   prog {f(d['pt_prog'])}"
              f"\n  hArg prog {f(d['ha_prog'])}\n  ctr p    {f(d['ctr_p'])}\n  ctr pArg {f(d['ctr_pa'])}\n  ctr true {f(d['ctr_true'])}"
              f"\n  h    prog {f(d['h_prog'])}\n  p mass   {f(d['p_mass'])}\n  h mass   {f(d['h_mass'])}\n  p err    {f(d['p_err'])}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract", "analyze"])
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--name", default="vast70")
    ap.add_argument("--ckpt", default="z_training/vast/ar_state/latest.pt")
    ap.add_argument("--tok_dir", required=True, help="토큰 memmap 자리 (로컬 디스크, 약 19 GB)")
    a = ap.parse_args()
    if a.phase == "extract":
        extract(a.gpus, a.name, a.ckpt, a.tok_dir)
    else:
        analyze(a.name, a.tok_dir)
