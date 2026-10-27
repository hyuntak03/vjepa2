"""motion 진단 (2026-10-24, 사용자 "일단 진단 해봐"). 학습 없는 비교 — 고정된 V-JEPA 정답 토큰 쌍이 움직임 코드 d 를 얼마나 설명하나.

학습 중 로그: 정답 쌍 오차 / 멈춤 기준 오차 ≈ 0.84–0.87 (설명 13–16 %) 에서 안 내려간다. 원인 후보 둘을 가른다:
  (ii) 차분 예측기 구조가 약하다 (칸 평균)            → B (같은 d, 같은 구조 끝까지) · C (같은 d, attention 모으기)
  (i)  d 자체가 토큰에 없는 변화까지 담는다 (SIGReg)  → D (차분 판 × 가중치 맞바꿈으로 새로 맞추기)
데이터 = 학습 로더와 같은 전처리 (var_window 표 · ClipTransform), 클립마다 W 장 (블록 W/2 개). 클립 단위 80/20 분할.
지표 (보류 20 %): R² = 1 − 오차 / Var(d) · 멈춤 비 = 오차 / 오차(뒤 블록 = 앞 블록). GPU 메모리를 --mem_gb 로 묶는다 (학습과 같은 GPU).

  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/motion_diag.py --run natural_ar_vw_motion
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
import yaml
from torch import nn

sys.path.insert(0, ".")
from app.vjepa_frozen import ar as AR, motion as MOM, utils as U          # noqa: E402
from app.vjepa_frozen.data import ClipTransform, normalize_clips           # noqa: E402
from app.vjepa_frozen.var_window import VarWindowBatches                   # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run", default="natural_ar_vw_motion"); p.add_argument("--ckpt", default="latest.pt")
p.add_argument("--clips", type=int, default=450); p.add_argument("--W", type=int, default=16)
p.add_argument("--steps", type=int, default=1500); p.add_argument("--bs", type=int, default=64)
p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--mem_gb", type=float, default=4.5)
p.add_argument("--only", default="")
a = p.parse_args()
dev = torch.device("cuda:0")
torch.cuda.set_per_process_memory_fraction(a.mem_gb * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
torch.manual_seed(0); random.seed(0)
RUN = f"z_training/runs/{a.run}"
cfg = yaml.safe_load(open(f"{RUN}/config.yaml"))
ck = torch.load(f"{RUN}/{a.ckpt}", map_location="cpu", weights_only=False)
MO, TUB, S, RES = ck["motion_cfg"], 2, 256, int(cfg["data"]["resolution"])
log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)
log(f"ckpt {RUN}/{a.ckpt} epoch {ck.get('epoch')}")

# ---- 데이터: 학습과 같은 표 · 전처리 ------------------------------------------------
aug = cfg["data"].get("aug") or {}
tf = ClipTransform(RES, hflip_p=0.0, random_resized_crop=aug.get("random_resized_crop"), square_crop=aug.get("square_crop", "none"), out_uint8=True)
ds = VarWindowBatches(cfg["data"]["var_window"], tf, TUB, seed=4242, rank=0)
def read(i):
    rng = random.Random(f"diag-{i}")
    t = ds.tables[i % len(ds.tables)]
    for _ in range(5):
        try:
            return t.name, tf(ds._read(t, rng.randrange(t.first_ok(a.W), len(t.paths)), a.W, rng))
        except Exception:
            continue
    return None
with ThreadPoolExecutor(32) as ex:
    got = [g for g in ex.map(read, range(a.clips)) if g is not None]
names = [g[0] for g in got]; frames = [g[1] for g in got]                  # (3, W, H, W) uint8
log(f"클립 {len(frames)} ({ {n: names.count(n) for n in set(names)} })")

# ---- 정답 토큰 (블록별 LN(target_encoder)) ----------------------------------------
state = U.load_base_checkpoint(cfg["model"]["checkpoint"])
_, enc = U.build_frozen_encoders(cfg["model"], state, dev, torch.bfloat16, int(cfg["data"]["n_frames"]))
del state
H = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i in range(0, len(frames), 4):
        x = normalize_clips(torch.stack(frames[i:i + 4]).to(dev))
        H.append(AR.encode_blocks(enc, x.to(torch.bfloat16), TUB).to(torch.bfloat16).cpu())
H = torch.cat(H).view(len(frames), a.W // TUB, S, -1)                       # (N, nb, S, D)
del enc; torch.cuda.empty_cache()
log(f"정답 토큰 {tuple(H.shape)}")
NB = a.W // TUB
idx = list(range(len(frames))); random.Random(1).shuffle(idx)
cut = int(0.8 * len(idx)); TR, TE = idx[:cut], idx[cut:]
trans = lambda cl: [(c, b) for c in cl for b in range(NB - 1)]
TRT, TET = trans(TR), trans(TE)

def diffs(cb, kind):
    """전이 목록 → 차분 영상 (M, ch, H, W). block = 블록 b+1 프레임 i − 블록 b 프레임 i (0.5 초, 6 채널, 학습과 같음) /
    boundary = 프레임 2b+2 − 2b+1 (0.25 초, 블록 경계, 3 채널)."""
    out = []
    for c, b in cb:
        f = normalize_clips(frames[c][None, :, 2 * b: 2 * b + 4].to(dev))[0]  # (3, 4, H, W) 블록 b · b+1
        if kind == "block":
            out.append(torch.cat([f[:, 2] - f[:, 0], f[:, 3] - f[:, 1]], 0))
        else:
            out.append(f[:, 2] - f[:, 1])
    return torch.stack(out)

def pairs(cb):
    za = torch.stack([H[c, b] for c, b in cb]).to(dev); zb = torch.stack([H[c, b + 1] for c, b in cb]).to(dev)
    return za, zb

class AttnPred(nn.Module):
    """더 센 예측기: 칸마다 [z_b, z_{b+1} − z_b] → MLP → transformer 1 층 → query 8 개 attention 모으기 → MLP."""
    def __init__(self, D=1280, h=512, out=192, q=8):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(2 * D, h), nn.LayerNorm(h), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(1, S, h)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.mix = nn.TransformerEncoderLayer(h, 8, 2 * h, 0.0, "gelu", batch_first=True, norm_first=True)
        self.q = nn.Parameter(torch.randn(1, q, h) * 0.02)
        self.att = nn.MultiheadAttention(h, 8, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(q * h), nn.Linear(q * h, h), nn.GELU(), nn.Linear(h, out))
    def forward(self, za, zb):
        u = self.mix(self.tok(torch.cat([za, zb - za], -1)) + self.pos)
        o, _ = self.att(self.q.expand(u.size(0), -1, -1), u, u)
        return self.head(o.flatten(1)).float()

@torch.no_grad()
def evaluate(enc_fn, pred, kind, cb=TET, bs=64):
    """보류 전이에서 R² (오차 / Var(d)) · 멈춤 비. enc_fn(차분) → d."""
    D_, F_, C_ = [], [], []
    for i in range(0, len(cb), bs):
        sub = cb[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d = enc_fn(diffs(sub, kind)).float()
            za, zb = pairs(sub)
            D_.append(d); F_.append(pred(za, zb).float()); C_.append(pred(za, za).float())
    d, f, c = torch.cat(D_), torch.cat(F_), torch.cat(C_)
    var = float(d.var(0).mean()); ef = float((f - d).pow(2).mean()); ec = float((c - d).pow(2).mean())
    return {"var_d": round(var, 4), "err": round(ef, 4), "err_copy": round(ec, 4), "R2": round(1 - ef / var, 4), "copy_ratio": round(ef / ec, 4)}

def train(enc, pred, kind, w_sig, w_fit, steps, fixed_enc=False):
    params = list(pred.parameters()) + ([] if fixed_enc else list(enc.parameters()))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.04)
    sig = MOM.SIGReg(17, 1024, 128.0).to(dev)
    rng = random.Random(7)
    for s in range(steps):
        sub = rng.sample(TRT, a.bs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if fixed_enc:
                with torch.no_grad():
                    d = enc(diffs(sub, kind)).float()
            else:
                d = enc(diffs(sub, kind)).float()
            za, zb = pairs(sub)
            l_fit = F.mse_loss(pred(za, zb), d)
            loss = w_fit * l_fit + (0.0 if fixed_enc else w_sig * sig(d))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if s % 500 == 0 or s == steps - 1:
            log(f"  step {s} fit {float(l_fit):.4f}")

R = {}
def cur_motion():
    m = MOM.Motion(MO, 1280, RES, 16, TUB).to(dev); m.load_state_dict(ck["motion"]); m.eval()
    return m
want = lambda k: (not a.only) or k in a.only.split(",")

if want("A"):   # A. 지금 체크포인트 그대로
    m = cur_motion()
    R["A_current"] = evaluate(m.enc, m.pred, "block"); log("A 지금 체크포인트", R["A_current"])
if want("B"):   # B. 같은 d (고정), 같은 구조 차분 예측기를 새로 끝까지
    m = cur_motion(); p_ = MOM.DiffPredictor(1280, 512, int(MO.get("dim", 192))).to(dev)
    train(m.enc, p_, "block", 0, 1.0, a.steps, fixed_enc=True); p_.eval()
    R["B_same_d_meanpool"] = evaluate(m.enc, p_, "block"); log("B 같은 d · 칸 평균", R["B_same_d_meanpool"])
if want("C"):   # C. 같은 d (고정), attention 모으기 예측기
    m = cur_motion(); p_ = AttnPred(out=int(MO.get("dim", 192))).to(dev)
    train(m.enc, p_, "block", 0, 1.0, a.steps, fixed_enc=True); p_.eval()
    R["C_same_d_attn"] = evaluate(m.enc, p_, "block"); log("C 같은 d · attention", R["C_same_d_attn"])
for kind in ("block", "boundary"):   # D. 새로 맞추기: 차분 판 × (SIGReg, 맞추기) 가중치
    for w_sig, w_fit in ((2.0, 0.5), (0.5, 2.0)):
        k = f"D_{kind}_sig{w_sig}_fit{w_fit}"
        if not want("D"):
            continue
        enc_ = MOM.DiffEncoder(RES, 16, 6 if kind == "block" else 3, 192, 12, 3, 192, 2048, ckpt=False).to(dev)
        p_ = MOM.DiffPredictor(1280, 512, 192).to(dev)
        enc_.train(); p_.train()
        train(enc_, p_, kind, w_sig, w_fit, a.steps); enc_.eval(); p_.eval()
        R[k] = evaluate(enc_, p_, kind); log(k, R[k])
out = f"{RUN}/motion_diag_{a.ckpt.replace('.pt', '')}.json"
json.dump({"args": vars(a), "n_clips": len(frames), "n_test_trans": len(TET), "results": R}, open(out, "w"), indent=1)
log("저장", out)
