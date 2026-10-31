#!/usr/bin/env python3
"""보조 기억 모듈 in-domain 검사 — 줄어든 손실이 재등장 물체 칸에서 왔나 (2026-10-31, 사용자: "측정해보던가").

질문: release_aux_perm (고정 릴리즈 predictor + 보조 기억 모듈, phys3d_perm_v1 로 20 epoch) 이 학습 손실을 3 % 줄였는데 VOBench late 는 안 올랐다.
  그 3 % 가 영속 (가렸다 다시 나오는 물체를 미래에 다시 그리기) 에서 왔나, 아니면 배경 · 보이는 물체의 일반 보정에서 왔나.
비교: 같은 입력에서 릴리즈 predictor 단독 (gate 0 과 비트 단위로 같다) vs 학습된 모듈. 정답 = LN(target encoder 32 장 전체) 의 미래 토큰.
  학습과 같은 구성: phys3d_perm_v1 val 876 clip (학습에 안 쓴 것), 32 장 4 fps, 문맥 C ∈ {10, 12, 14, 16} 장 → 나머지 미래 전부.
칸 분류 (미래 블록 t = 연속 2 장, 16×16 칸; 물체 상자는 states.npz bbox, 보임 = vis_frac > 0.5 · inframe_frac > 0.5 를 두 장 모두):
  재등장 = 블록 t 에서 보이고, 문맥 마지막 블록에서 안 보이고, 문맥 앞쪽에서는 보였던 물체의 상자 칸
  계속 보임 = 문맥 마지막 블록과 블록 t 에서 모두 보이는 물체의 상자 칸
  새로 들어옴 = 문맥에서 한 번도 안 보였던 물체
  가려짐 = 블록 t 에서 화면 안인데 안 보이는 (가림막 뒤) 물체의 상자 칸
  배경 = 어느 물체 상자에도 안 걸린 칸
점수: 칸 분류마다 이득 % = (릴리즈 L1 − 모듈 L1) / 릴리즈 L1 (토큰 평균 L1 의 비). clip 단위 평균 차이의 표준오차를 같이.

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/aux_perm_indomain.py [--ckpt e20.pt] [--n 876]
출력: z_analysis/results/aux_perm_indomain/<ckpt>.json
"""
import argparse, csv, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="release_aux_perm")
ap.add_argument("--ckpt", default="e20.pt")
ap.add_argument("--n", type=int, default=876)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--mem", type=float, default=0.3)
a = ap.parse_args()
OUT = os.path.join(REPO, "z_analysis/results/aux_perm_indomain"); os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(a.mem)
from app.vjepa_frozen import utils as U
from app.vjepa_frozen.aux_memory import AuxRelease

RD = os.path.join(REPO, "z_training/runs", a.run)
cfg = yaml.safe_load(open(os.path.join(RD, "config.yaml"))); MD = cfg["model"]
NF, G, S, TUB = 32, 16, 256, 2
CTX = [10, 12, 14, 16]
state = U.load_base_checkpoint(MD["checkpoint"])
ctx_enc, tgt_enc = U.build_frozen_encoders(MD, state, dev, torch.bfloat16, NF)
pred = U.build_predictor(MD, ctx_enc.embed_dim, state, dev, NF)
model = AuxRelease(pred, dict(MD["aux"]), ctx_enc.embed_dim, 256, 16, 2, NF).to(dev)
ck = torch.load(os.path.join(RD, a.ckpt), map_location="cpu", weights_only=False)
print(model.load_state_dict(ck["predictor"], strict=True), "epoch", ck.get("epoch"), "gates", {k: round(v, 4) for k, v in model.gates().items()}, flush=True)
model.eval(); del state, ck
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CATS = ["재등장", "계속 보임", "새로 들어옴", "가려짐", "배경"]


def cells(box):
    x0, y0, x1, y1 = box
    if not np.isfinite([x0, y0, x1, y1]).all(): return set()
    return {(r, c) for r in range(max(0, int(y0 // 16)), min(G, int(y1 // 16) + 1)) for c in range(max(0, int(x0 // 16)), min(G, int(x1 // 16) + 1))}


def cell_labels(st, C):
    """(미래 블록 수, 256) 칸 분류 번호 (CATS 순서, 겹치면 앞 번호가 이긴다)."""
    vis = (st["vis_frac"] > 0.5) & (st["inframe_frac"] > 0.5)               # (32, O)
    inf = st["inframe_frac"] > 0.5
    bv = vis[0::2] & vis[1::2]                                             # (16 블록, O)
    Cb = C // TUB; O = vis.shape[1]
    lab = np.full((NF // TUB - Cb, S), 4, np.int8)
    for t in range(Cb, NF // TUB):
        cat_cells = [set() for _ in range(4)]
        for o in range(O):
            box_cells = cells(st["bbox"][2 * t, o]) | cells(st["bbox"][2 * t + 1, o])
            seen_ctx = bv[:Cb, o].any()
            if bv[t, o]:
                k = 1 if bv[Cb - 1, o] else (0 if seen_ctx else 2)
            elif inf[2 * t, o] and inf[2 * t + 1, o] and seen_ctx:
                k = 3
            else:
                continue
            cat_cells[k] |= box_cells
        for k in (3, 2, 1, 0):                                             # 앞 번호가 이기게 뒤에서부터 칠한다
            for (r, c) in cat_cells[k]:
                lab[t - Cb, r * G + c] = k
    return lab


paths = [r[0].split()[0] for r in csv.reader(open("/workspace/dataset/training_data/phys3d_perm_v1/val/index.csv"))][: a.n]
scen = {p: p.split("/")[-2] for p in paths}
import decord
rec = []                                                                   # clip × C 마다: 분류별 (릴리즈 L1 합, 모듈 L1 합, 칸 수)
for b0 in range(0, len(paths), a.bs):
    bp = paths[b0:b0 + a.bs]
    fr = np.stack([decord.VideoReader(p, num_threads=2).get_batch(list(range(NF))).asnumpy() for p in bp])   # (B, 32, 256, 256, 3)
    sts = [np.load(p[:-4] + "_states.npz") for p in bp]
    x = torch.from_numpy(fr).to(dev).permute(0, 4, 1, 2, 3).float().div_(255)
    x = ((x - MEAN) / STD).to(torch.bfloat16)
    B = x.size(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h_full = F.layer_norm(tgt_enc(x).float(), (1280,))                 # (B, 4096, D)
        for C in CTX:
            n_ctx = C // TUB * S
            me = torch.arange(n_ctx, device=dev).unsqueeze(0).expand(B, -1).contiguous()
            mp_ = torch.arange(n_ctx, NF // TUB * S, device=dev).unsqueeze(0).expand(B, -1).contiguous()
            z = ctx_enc(x, masks=[me])
            h = h_full[:, n_ctx:]
            p_rel = model.pred(z, me, mp_, mask_index=0).float()
            p_aux = model(z, me, mp_, mask_index=0).float()
            l_rel = (p_rel - h).abs().mean(-1).view(B, -1, S).cpu().numpy()    # (B, 미래 블록, 256)
            l_aux = (p_aux - h).abs().mean(-1).view(B, -1, S).cpu().numpy()
            for i in range(B):
                lab = cell_labels(sts[i], C)
                d = {"clip": bp[i], "scen": scen[bp[i]], "C": C}
                for k, nm in enumerate(CATS):
                    m = lab == k
                    d[nm] = [float(l_rel[i][m].sum()), float(l_aux[i][m].sum()), int(m.sum())]
                rec.append(d)
    if (b0 // a.bs) % 10 == 0: print(f"{b0 + B}/{len(paths)}", flush=True)

json.dump(rec, open(os.path.join(OUT, f"{a.run}_{a.ckpt[:-3]}.json"), "w"))


def summ(rr, nm):
    rr = [r for r in rr if r[nm][2] > 0]
    if not rr: return float("nan"), float("nan"), 0, 0
    R_ = sum(r[nm][0] for r in rr); A_ = sum(r[nm][1] for r in rr); N_ = sum(r[nm][2] for r in rr)
    pc = np.array([(r[nm][0] - r[nm][1]) / r[nm][0] for r in rr]) * 100     # clip 단위 이득 %
    return 100 * (R_ - A_) / R_, pc.std(ddof=1) / np.sqrt(len(pc)) if len(pc) > 1 else float("nan"), len(rr), N_


print(f"\n{a.run}/{a.ckpt} · phys3d_perm_v1 val {len(paths)} clip · 이득 % = (릴리즈 − 모듈) / 릴리즈 L1  [± clip 단위 SE · clip 수 · 칸 수]")
for C in CTX + [None]:
    rr = [r for r in rec if C is None or r["C"] == C]
    tot = [sum(r[nm][0] for nm in CATS) for r in rr], [sum(r[nm][1] for nm in CATS) for r in rr]
    head = f"  C={C if C else '전체':>3}  모든 칸 {100 * (sum(tot[0]) - sum(tot[1])) / sum(tot[0]):+5.2f}  |"
    print(head + "  ".join(f"{nm} {g:+5.2f}±{se:.2f} ({n},{nc})" for nm in CATS for g, se, n, nc in [summ(rr, nm)]))
print("\n시나리오별 (C 전체) — 재등장 / 계속 보임 / 배경")
for sc in sorted(set(scen.values())):
    rr = [r for r in rec if r["scen"] == sc]
    print(f"  {sc:14s} " + "  ".join(f"{nm} {g:+5.2f} (n {n})" for nm in ("재등장", "계속 보임", "배경") for g, se, n, nc in [summ(rr, nm)]))
