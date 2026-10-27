"""릴리즈 predictor 도 미래 블록에 encoder 꼴의 속도를 안 만드나 (2026-10-27, GPU 1 장).
사용자: "release predictor 도 한번 진단해 볼래? 얘도 미래 부분 만드는 순간 encoder 의 표현처럼 속도 안 만드나?"
릴리즈 predictor: 문맥 16 장 (블록 0–7) → 미래 블록 8–15 를 한 번에 (되먹임 없음). 정답 = 32 장 한 번에 LN(target_encoder) (objcell_concept_cache.py 와 같은 계산).
  → 릴리즈의 "encoder 꼴" 은 블록 AR 의 2 장 encoder 가 아니라 이 32 장 target encoder 토큰이다.
시험: 3D 합성 물체 하나 val 350 편 · 판독기 학습: 같은 세트 train 앞 N 편 (ar_layers meta). 블록 b = 원본 0, 3, …, 93 장 중 2b, 2b+1.
판독기 (attentive, phys_only_ar_velocity.py 와 같은 구조 · 설정, 속도 · 위치 두 머리):
  ① encoder 꼴 판독기 = train 의 32 장 target 토큰 (블록 2–15, b−1 · b 물체 보임) 으로 학습 → val target (상한) · val 릴리즈 예측 (블록 8–15) 에 시험
  ② 자기 판독기 = train 의 릴리즈 예측 (블록 8–15, b 물체 보임) 으로 학습 → val 릴리즈 예측에 시험
지표: 블록 안 속도 R² · 위치 R² (미래 블록 전체 · 걸음 k = b − 7 별) · 걸음별 따라 움직인 비율 (블록 7 기준, 실제 이동 ≥ 16 px).
  CUDA_VISIBLE_DEVICES=0 MEM_GB=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/release_velocity.py --n_train 1500
출력: z_analysis/results/release_velocity.json
"""
import argparse, json, math, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser(); ap.add_argument("--n_train", type=int, default=1500); ap.add_argument("--n_val", type=int, default=350)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--pred_ckpt", default=None, help="predictor 만 이 체크포인트로 (post-FT run 의 latest.pt)"); ap.add_argument("--out", default="release_velocity.json"); a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO)
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 6)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
from analysis.intphys2.model import build_from_config
from evals.world_model_analysis.eval import _context_target_indices
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
MCFG = {"arch_name": "vit_huge", "checkpoint": CK, "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True, "uniform_power": False, "dual_encoder": True,
        "context_encoder_key": "encoder", "target_encoder_key": "target_encoder", "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10},
        "dtype": "float32", "autocast": "float16", "window_size": 32}
if a.pred_ckpt:
    MCFG["predictor_checkpoint"] = a.pred_ckpt   # 2026-10-27: VOBench post-FT 릴리즈 (vobench_release_postft_g_e20)
rel = build_from_config(MCFG, dev); rel.context_encoder.half(); rel.target_encoder.half()
ci, ti = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=256, batch_size=1, device=dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
IDX = [3 * k for k in range(32)]; S, D = 256, 1280


def collect(meta, with_steps):
    """T: target 토큰 (블록 2–15 표본), LT; P: 릴리즈 예측 (블록 8–15 중 b 물체 보임), LP (라벨 + 블록 7 위치 + 걸음)."""
    T, LT, P, LP = [], [], [], []
    for n, m in enumerate(meta):
        B = m["blocks"]
        fr = decord.VideoReader(m["clip"], num_threads=4).get_batch(IDX).asnumpy()
        xn = ((torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0 - MEAN) / STD).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            z = rel.context_encoder(xn.half(), masks=[ci]); p = rel.predictor(z, ci, ti, mask_index=0); h = rel.target_encoder(xn.half())
        Pm = p.float()[0].view(8, S, D).half().cpu().numpy(); Tm = F.layer_norm(h.float(), (D,))[0].view(16, S, D).half().cpu().numpy()
        for b in range(2, 16):
            if B[b]["ok"] and B[b - 1]["ok"]:
                T.append(Tm[b]); LT.append([*B[b]["disp"], *B[b]["ctr"], *B[b - 1]["ctr"], b])
        last = B[7]["ctr"] if B[7]["ok"] else [np.nan, np.nan]
        for b in range(8, 16):
            if B[b]["ok"]:
                P.append(Pm[b - 8]); LP.append([*B[b]["disp"], *B[b]["ctr"], *last, b])
        if (n + 1) % 250 == 0:
            print(f"  {n + 1}/{len(meta)}", flush=True)
    return np.stack(T), np.array(LT, np.float64), np.stack(P), np.array(LP, np.float64)


class AttnProbe(nn.Module):
    def __init__(s, d=1280, heads=16, outs=(2, 2)):
        super().__init__()
        s.ln = nn.LayerNorm(d); s.pos = nn.Parameter(torch.randn(256, d) * 0.02); s.q = nn.Parameter(torch.randn(len(outs), d) * 0.02)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True); s.ln2 = nn.LayerNorm(d); s.heads = nn.ModuleList([nn.Linear(d, o) for o in outs])

    def forward(s, x):
        x = s.ln(x) + s.pos; o, _ = s.attn(s.q.expand(len(x), -1, -1), x, x, need_weights=False); o = s.ln2(o)
        return [h(o[:, i]) for i, h in enumerate(s.heads)]


def train_probe(X, L):
    yv, yp = L[:, 0:2], L[:, 2:4]; st = (yv.mean(0), yv.std(0) + 1e-6, yp.mean(0), yp.std(0) + 1e-6)
    Yv = torch.from_numpy(((yv - st[0]) / st[1]).astype(np.float32)); Yp = torch.from_numpy(((yp - st[2]) / st[3]).astype(np.float32))
    m = AttnProbe().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=a.epochs * math.ceil(len(X) / 128), pct_start=0.05)
    g = torch.Generator().manual_seed(0)
    for ep in range(a.epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ov, op = m(torch.from_numpy(X[b.numpy()]).to(dev).float())
                loss = F.mse_loss(ov.float(), Yv[b].to(dev)) + F.mse_loss(op.float(), Yp[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()

    def read(Xq):
        V, P_ = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(Xq), 64):
                o = m(torch.from_numpy(Xq[i:i + 64]).to(dev).float()); V.append(o[0].float().cpu().numpy()); P_.append(o[1].float().cpu().numpy())
        return np.concatenate(V) * st[1] + st[0], np.concatenate(P_) * st[3] + st[2]
    return read


r2 = lambda P_, Y: round(float(1 - ((P_ - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()), 3)


def future_scores(V, Pp, L):
    out = {"all": {"velocity_r2": r2(V, L[:, 0:2]), "position_r2": r2(Pp, L[:, 2:4]), "n": int(len(L))}}
    for k in range(1, 9):
        s = L[:, 6] == 7 + k
        d = L[:, 2:4] - L[:, 4:6]; mv = s & np.isfinite(d).all(1) & (np.linalg.norm(np.nan_to_num(d), axis=1) >= 16)
        fol = float(((Pp[mv] - L[mv, 4:6]) * d[mv]).sum() / (d[mv] ** 2).sum()) if mv.sum() else None
        out[str(k)] = {"velocity_r2": r2(V[s], L[s, 0:2]) if s.sum() > 10 else None, "follow": None if fol is None else round(fol, 3), "n": int(s.sum()), "n_moved16": int(mv.sum())}
    return out


meta_tr = json.load(open("/workspace/auto_research_cache/ar_layers/train_meta.json"))[: a.n_train]
meta_va = json.load(open("/workspace/auto_research_cache/ar_layers/val_meta.json"))[: a.n_val]
print("train 추출", flush=True); Ttr, LTtr, Ptr, LPtr = collect(meta_tr, False)
print("val 추출", flush=True); Tva, LTva, Pva, LPva = collect(meta_va, True)
print(f"표본 train target {len(LTtr)} 예측 {len(LPtr)} · val target {len(LTva)} 예측 {len(LPva)} · GPU 최대 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
R = {"n": {"train_target": int(len(LTtr)), "train_pred": int(len(LPtr)), "val_target": int(len(LTva)), "val_pred": int(len(LPva))}}
f_enc = train_probe(Ttr, LTtr)
Vt, Pt = f_enc(Tva); R["encoder_form_probe_on_target"] = {"velocity_r2": r2(Vt, LTva[:, 0:2]), "position_r2": r2(Pt, LTva[:, 2:4])}
R["encoder_form_probe_on_release_pred"] = future_scores(*f_enc(Pva), LPva)
f_own = train_probe(Ptr, LPtr)
R["own_probe_on_release_pred"] = future_scores(*f_own(Pva), LPva)
# 같은 미래 블록의 target 토큰에 encoder 꼴 판독기 (걸음별 상한)
fut = LTva[:, 6] >= 8
R["encoder_form_probe_on_target_future_blocks"] = {"velocity_r2": r2(Vt[fut], LTva[fut, 0:2]), "position_r2": r2(Pt[fut], LTva[fut, 2:4]), "n": int(fut.sum())}
print(json.dumps(R, indent=1))
R["predictor_checkpoint"] = a.pred_ckpt
json.dump(R, open(os.path.join(REPO, "z_analysis/results", a.out), "w"), indent=1)
