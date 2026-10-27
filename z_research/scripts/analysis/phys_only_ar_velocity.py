"""합성만으로 학습한 블록 AR 도 출력에 블록 안 속도가 안 담기나 (2026-10-26, GPU 1 장 · 6 GB).
사용자: "phys only 로 학습한 ar 도 predictor output 에 속도가 안 담기나?"
모델 (구조 · 손실 같음, 데이터만 다름):
  natural      = natural_ar_vw_state (K710 · SSv2 · HowTo, 70 epoch)                         — 기준 (tf_output_velocity.json 의 0.04 재현)
  phys3d_single = natural_ar_vw_state_phys3d_only (3D 합성 물체 하나 v1 만, 20 epoch)       — 시험 세트와 같은 분포
  phys3d_multi  = natural_ar_vw_state_phys3d_multi_only (3D 합성 다중 물체 v1 만, epoch 34 에서 멈춤) — 같은 렌더러, 다른 세트
시험: 3D 합성 물체 하나 v1 val 350 편 (ar_layers val_meta). 판독기 학습: 같은 세트 train 앞 N 편 (ar_layers train_meta).
  블록 = 원본 0, 3, …, 93 장 중 2b, 2b+1 (4 fps), 블록 16. teacher forcing 출력 tf[b−1] = 블록 b 예측, 대상 블록 b = 2–15 중 b−1 · b 둘 다 물체 보임.
판독기 두 가지 (attentive probe, attnprobe_train.py 와 같은 구조 · 설정, 속도 · 위치 두 머리):
  ① encoder 판독기 = 기존 attnprobe_enc.pt 를 그대로 (encoder 토큰으로 학습) — 0.04 와 같은 자
  ② 모델 자기 판독기 = 그 모델의 teacher forcing 출력 (train N 편) 으로 새로 학습 → val 출력에서 시험. 같은 방식으로 encoder 토큰에도 학습 (상한)
지표: 블록 안 속도 R² · 위치 R² · 한 블록 이동 따라 움직인 비율 (실제 ≥ 8 px) · 굴림 (문맥 8 → 8 걸음) 걸음별 따라 움직인 비율 (블록 7 기준, 실제 ≥ 16 px, 자기 판독기).
  CUDA_VISIBLE_DEVICES=0 MEM_GB=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/phys_only_ar_velocity.py --n_train 1500
출력: z_analysis/results/phys_only_ar_velocity.json
"""
import argparse, json, math, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser(); ap.add_argument("--n_train", type=int, default=1500); ap.add_argument("--n_val", type=int, default=350)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--models", default="natural,phys3d_single,phys3d_multi"); ap.add_argument("--out", default="phys_only_ar_velocity.json"); a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 6)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT
sys.argv = _argv
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
S, D, NB, C, K = 256, 1280, 16, 8, 8; IDX = [3 * k for k in range(32)]
ALL = {"natural": "natural_ar_vw_state", "phys3d_single": "natural_ar_vw_state_phys3d_only", "phys3d_multi": "natural_ar_vw_state_phys3d_multi_only",
       "mix_event": "natural_ar_vw_state_phys3d_event"}   # 2026-10-26 추가: 자연 영상 + 3D 합성 사건 구간 보조 batch (token 25 %) + 상태 감독 ×0.05, 70 epoch
MODELS = {k: ALL[k] for k in a.models.split(",")}
OUT = os.path.join(REPO, "z_analysis/results", a.out)

enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
_load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.half().to(dev).eval()
ARS = {k: FT.load_pred(v, "latest.pt").to(dev).eval() for k, v in MODELS.items()}
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
idx_all = torch.arange(NB * S, device=dev).unsqueeze(0)


def collect(meta, with_rollout):
    """→ X {'enc': [...], model: [...]} (대상 블록 토큰 (256, 1280) fp16), LB (n, 7) = [속도 2, 위치 2, 직전 위치 2, 블록], RO {model: [(clip, 8, 256, 1280)]}, RL (clip 별 블록 7 · 8–15 라벨)."""
    X = {k: [] for k in ["enc", *MODELS]}; LB = []; RO = {k: [] for k in MODELS}; RL = []
    for n, m in enumerate(meta):
        B = m["blocks"]
        fr = decord.VideoReader(m["clip"], num_threads=4).get_batch(IDX).asnumpy()
        x = (torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0 - MEAN) / STD
        xb = x.reshape(3, NB, 2, 256, 256).permute(1, 0, 2, 3, 4)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = torch.cat([F.layer_norm(enc2(xb[i:i + 4].half().contiguous()).float(), (D,)) for i in range(0, NB, 4)])
            outs = {k: ar.forward_seq(h.reshape(1, -1, D), idx_all).float()[0].view(NB, S, D) for k, ar in ARS.items()}
            ros = {k: ar.rollout_kv(h[:C].reshape(1, -1, D), idx_all[:, :C * S], K).float()[0].view(K, S, D) for k, ar in ARS.items()} if with_rollout else None
        sel = [b for b in range(2, NB) if B[b]["ok"] and B[b - 1]["ok"]]
        if sel:
            X["enc"].append(h[sel].half().cpu().numpy())
            for k in MODELS:
                X[k].append(outs[k][[b - 1 for b in sel]].half().cpu().numpy())
            LB += [[*B[b]["disp"], *B[b]["ctr"], *B[b - 1]["ctr"], b] for b in sel]
        if with_rollout and B[C - 1]["ok"]:
            for k in MODELS:
                RO[k].append(ros[k].half().cpu().numpy())
            RL.append([[*B[b]["ctr"], float(B[b]["ok"])] for b in range(C - 1, NB)])
        if (n + 1) % 250 == 0:
            print(f"  {n + 1}/{len(meta)}", flush=True)
    X = {k: np.concatenate(v) for k, v in X.items()}
    RO = {k: (np.stack(v) if v else None) for k, v in RO.items()}
    return X, np.array(LB, np.float64), RO, np.array(RL, np.float64)


class AttnProbe(nn.Module):
    def __init__(s, d=1280, heads=16, outs=(48, 2, 2)):
        super().__init__()
        s.ln = nn.LayerNorm(d); s.pos = nn.Parameter(torch.randn(256, d) * 0.02); s.q = nn.Parameter(torch.randn(len(outs), d) * 0.02)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True); s.ln2 = nn.LayerNorm(d); s.heads = nn.ModuleList([nn.Linear(d, o) for o in outs])

    def forward(s, x):
        x = s.ln(x) + s.pos; o, _ = s.attn(s.q.expand(len(x), -1, -1), x, x, need_weights=False); o = s.ln2(o)
        return [h(o[:, i]) for i, h in enumerate(s.heads)]


def readout(model, X, nv, sv):
    V, P = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), 64):
            o = model(torch.from_numpy(X[i:i + 64]).to(dev).float()); V.append(o[-2].float().cpu().numpy()); P.append(o[-1].float().cpu().numpy())
    return np.concatenate(V) * sv[0] + nv[0], np.concatenate(P) * sv[1] + nv[1]


def train_probe(X, LB):
    """속도 · 위치 두 머리 attentive probe — attnprobe_train.py 와 같은 설정 (AdamW 3e-4 · wd 0.05 · 묶음 128 · OneCycle), epoch 고정."""
    yv, yp = LB[:, 0:2], LB[:, 2:4]; mv_, sv_, mp_, sp_ = yv.mean(0), yv.std(0) + 1e-6, yp.mean(0), yp.std(0) + 1e-6
    Yv = torch.from_numpy(((yv - mv_) / sv_).astype(np.float32)); Yp = torch.from_numpy(((yp - mp_) / sp_).astype(np.float32))
    m = AttnProbe(outs=(2, 2)).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05)
    steps = a.epochs * math.ceil(len(X) / 128); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.05)
    g = torch.Generator().manual_seed(0)
    for ep in range(a.epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ov, op = m(torch.from_numpy(X[b.numpy()]).to(dev).float())
                loss = F.mse_loss(ov.float(), Yv[b].to(dev)) + F.mse_loss(op.float(), Yp[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval(); return m, (mv_, mp_), (sv_, sp_)


r2 = lambda P, Y: float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def scores(V, P, LB):
    yv, yp, prev = LB[:, 0:2], LB[:, 2:4], LB[:, 4:6]; mv = np.linalg.norm(yp - prev, axis=1) >= 8
    return {"velocity_r2": round(r2(V, yv), 3), "position_r2": round(r2(P, yp), 3),
            "follow_one_block": round(float(((P[mv] - prev[mv]) * (yp[mv] - prev[mv])).sum() / ((yp[mv] - prev[mv]) ** 2).sum()), 3), "n": int(len(LB)), "n_moved8": int(mv.sum())}


meta_tr = json.load(open("/workspace/auto_research_cache/ar_layers/train_meta.json"))[: a.n_train]
meta_va = json.load(open("/workspace/auto_research_cache/ar_layers/val_meta.json"))[: a.n_val]
print("train 추출", flush=True); Xtr, Ltr, _, _ = collect(meta_tr, False)
print("val 추출", flush=True); Xva, Lva, ROva, RLva = collect(meta_va, True)
print("표본 train", len(Ltr), "val", len(Lva), flush=True)
R = {"models": MODELS, "n_train_samples": int(len(Ltr)), "n_val_samples": int(len(Lva)),
     "note": "phys3d_multi 는 70 epoch 계획 중 epoch 34 에서 멈춘 체크포인트 (latest.pt). mix_event 는 상태 감독 손실도 있다"}
# ① encoder 판독기 그대로 (0.04 와 같은 자)
Lt = np.load("/workspace/auto_research_cache/attnprobe/train_enc_lab.npy")
pe = AttnProbe().to(dev); pe.load_state_dict(torch.load("/workspace/auto_research_cache/attnprobe/attnprobe_enc.pt", map_location="cpu")); pe.eval()
nv, sv = (Lt[:, 2:4].mean(0), Lt[:, 4:6].mean(0)), (Lt[:, 2:4].std(0) + 1e-6, Lt[:, 4:6].std(0) + 1e-6)
R["encoder_probe"] = {k: scores(*readout(pe, Xva[k], nv, sv), Lva) for k in ["enc", *MODELS]}
print("① encoder 판독기:", json.dumps(R["encoder_probe"]), flush=True)
# ② 자기 판독기
R["own_probe"] = {}; R["rollout_follow_own_probe"] = {}
for k in ["enc", *MODELS]:
    pm, nv_, sv_ = train_probe(Xtr[k], Ltr)
    R["own_probe"][k] = scores(*readout(pm, Xva[k], nv_, sv_), Lva)
    print(f"② 자기 판독기 {k}: {R['own_probe'][k]}", flush=True)
    if k in MODELS and ROva[k] is not None:
        ro = ROva[k]; n_c = ro.shape[0]
        _, Pr = readout(pm, ro.reshape(-1, S, D), nv_, sv_); Pr = Pr.reshape(n_c, K, 2)
        last = RLva[:, 0, 0:2]; res = {}
        for s in range(K):
            tru = RLva[:, s + 1, 0:2]; okb = RLva[:, s + 1, 2] > 0; d = tru - last; mv = okb & (np.linalg.norm(d, axis=1) >= 16)
            res[str(s + 1)] = {"follow": round(float(((Pr[mv, s] - last[mv]) * d[mv]).sum() / (d[mv] ** 2).sum()), 3) if mv.sum() else None, "n": int(mv.sum())}
        R["rollout_follow_own_probe"][k] = res
        print(f"   굴림 따라 움직인 비율 {k}: " + " ".join(f"{s}:{v['follow']}" for s, v in res.items()), flush=True)
    del pm; torch.cuda.empty_cache()
json.dump(R, open(OUT, "w"), indent=1)
print("saved", OUT)
