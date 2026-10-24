#!/usr/bin/env python3
"""[P] 개입 1b: 흐름 없는 비교 보조 과제 이어 학습 (auto_research, 2026-10-21). ar_stab_ft.py 복사, 보조 정답만 바꿨다.
보조 정답 (토큰에서 바로, 문맥만): 위치 b (= 입력 블록 b, b = 1..6) 의 칸 c 에서 cos(h_b(c), h_{b−1}(c + d)), d ∈ 5×5 (−2..2 칸, 화면 밖 제외)
  → s' = 최대 cos, u' = 그 d (dy, dx). 머리 = 층 L 은닉 → 선형 3 출력. 보조 손실 = |ŝ' − s'| + ½ · 평균 |û' − u'| / 2.
이하 원본 설명:
[P] 개입: 안정도 보조 과제 이어 학습 (auto_research, 2026-10-21). z_training · app · src 는 수정하지 않는다 (import 만).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/ar_stab_ft.py --name ft_stab_l05_lr1e4 --lam 0.5 --lr 1e-4

[T] ar_crps_ft.py 의 L1 팔과 같은 설정 (기준 AR latest, 자연 영상 학습 캐시 900 창, 배치 8, 2000 step, AdamW wd 0, 기울기 자르기 1.0, 같은 표집 seed, TF 위치 0..6).
더한 것: predictor 층 L (기본 4) 출력 은닉 (384) 에 선형 머리 → ŝ (칸마다). 손실 = TF L1 + λ · 평균 |ŝ − s|.
보조 정답 s (문맥만): 위치 b (= 입력 블록 b, b = 1..6) 의 칸 c 에서 j = c + U[b](c) (U[b] = 블록 b → b−1 칸 흐름, 등속 가정으로 다음 블록 c 에 올 내용의 지금 자리),
  s = cos(h_b(j), h_{b−1}(j + U[b](j))). 두 자리 모두 화면 안인 칸만. 흐름 = natural_train/*_Uflow.npy ([T] 계산, RAFT 칸 평균).
저장: /workspace/auto_research_cache/ar_ft/<name>.pt (predictor state · stab_head · args · log) — [T] 평가기와 같은 형식."""
import argparse, glob, os, sys, time
import numpy as np, torch, torch.nn as nn, yaml

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--lam", type=float, default=0.5)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lr_head", type=float, default=1e-3)
ap.add_argument("--layer", type=int, default=4)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cache", default="/workspace/auto_research_cache/natural_train")
ap.add_argument("--out", default="/workspace/auto_research_cache/ar_ft")
a = ap.parse_args()
torch.set_num_threads(8)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _clean_backbone_key                       # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                 # noqa: E402
dev = torch.device("cuda")
torch.manual_seed(a.seed); np.random.seed(a.seed)
S, G = 256, 16


def load_pred(run="natural_ar_vw_state", ck="latest.pt"):
    D = os.path.join(REPO, "z_training/runs", run)
    pc = yaml.safe_load(open(os.path.join(D, "config.yaml")))["model"]["predictor"]
    m = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
                             num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
    w = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D, ck), map_location="cpu", weights_only=False)["predictor"]).items()
         if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
    m.load_state_dict(w, strict=True)
    return m


pred = load_pred().to(dev); pred.train()
head = nn.Linear(384, 3).to(dev)
nn.init.normal_(head.weight, std=0.01); nn.init.zeros_(head.bias)
with torch.no_grad():
    head.bias[0] = 0.5
HID = {}
pred.predictor_blocks[a.layer - 1].register_forward_hook(lambda mod, inp, out: HID.__setitem__("h", out))

hfiles = sorted(glob.glob(os.path.join(a.cache, "*_h.npy")))
srcs = [np.load(f, mmap_mode="r") for f in hfiles]
flows = [np.load(f.replace("_h.npy", "_Uflow.npy"), mmap_mode="r") for f in hfiles]
sizes = [x.shape[0] for x in srcs]
print(f"캐시 {[os.path.basename(f) for f in hfiles]} 창 {sizes} · 층 {a.layer} · λ {a.lam} · lr {a.lr}", flush=True)
cyx = torch.stack(torch.meshgrid(torch.arange(G, device=dev), torch.arange(G, device=dev), indexing="ij"), -1).reshape(S, 2).float()   # (S, (y, x))
P0 = torch.stack([cyx[:, 1] + 0.5, cyx[:, 0] + 0.5], -1)                       # (S, (x, y)) 칸 중심


def cell_of(p):                                                                 # p (..., 2) (x, y) 칸 단위 → 칸 번호, 화면 안 여부
    f = torch.floor(p).long()
    ok = (f >= 0).all(-1) & (f < G).all(-1)
    return f[..., 1].clamp(0, G - 1) * G + f[..., 0].clamp(0, G - 1), ok


OFF = torch.tensor([(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)], device=dev)          # (25, 2)


def stab_target(x, U=None):
    """x (B, 8, S, D). return s' (B, 6, S), u' (B, 6, S, 2) (dy, dx), ok (B, 6, S) — 위치 b = 1..6."""
    B = x.shape[0]
    xn = torch.nn.functional.normalize(x, dim=-1)
    yy, xx = cyx[:, 0].long(), cyx[:, 1].long()
    ss, uu = [], []
    for b in range(1, 7):
        cs = []
        for dy, dx in OFF.tolist():
            ty, tx = yy + dy, xx + dx
            ok = (ty >= 0) & (ty < G) & (tx >= 0) & (tx < G)
            j = (ty.clamp(0, G - 1) * G + tx.clamp(0, G - 1))
            c = (xn[:, b] * xn[:, b - 1][:, j]).sum(-1)                         # (B, S)
            cs.append(torch.where(ok[None], c, torch.full_like(c, -2.0)))
        cs = torch.stack(cs, -1)                                                # (B, S, 25)
        m, k = cs.max(-1)
        ss.append(m); uu.append(OFF[k].float())
    s_ = torch.stack(ss, 1); u_ = torch.stack(uu, 1)
    return s_, u_, torch.ones_like(s_, dtype=torch.bool)


opt = torch.optim.AdamW([{"params": pred.parameters(), "lr": a.lr}, {"params": head.parameters(), "lr": a.lr_head}], weight_decay=0.0)
rng = np.random.default_rng(a.seed)
t0 = time.time(); log = []
for it in range(1, a.steps + 1):
    si = rng.integers(len(srcs), size=a.bs)
    wi = [rng.integers(sizes[s]) for s in si]
    xb = torch.from_numpy(np.stack([srcs[s][w] for s, w in zip(si, wi)])).to(dev).float()            # (B, 8, S, D)
    ub = torch.from_numpy(np.stack([np.asarray(flows[s][w], np.float32) for s, w in zip(si, wi)])).to(dev).reshape(a.bs, 8, S, 2)
    x = xb.reshape(a.bs, -1, 1280)
    y = x[:, S:]
    with torch.no_grad():
        s_tgt, u_tgt, ok = stab_target(xb)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        idx = torch.arange(x.size(1), device=dev).unsqueeze(0).expand(a.bs, -1)
        p = pred.forward_seq(x, idx)[:, :-S].float()
        l1 = (p - y).abs().mean()
        hid = HID["h"].float().view(a.bs, 8, S, -1)[:, 1:7]                     # 위치 1..6
        ho = head(hid).float()
        s_hat, u_hat = ho[..., 0], ho[..., 1:]
    okf = ok.float()
    aux = ((s_hat - s_tgt).abs() * okf).sum() / okf.sum().clamp_min(1) + 0.5 * ((u_hat - u_tgt).abs().mean(-1) / 2).mean()
    loss = l1 + a.lam * aux
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(pred.parameters()) + list(head.parameters()), 1.0); opt.step()
    if it % 100 == 0 or it == 1:
        with torch.no_grad():
            m_ = ok
            sh, st = s_hat[m_].float(), s_tgt[m_].float()
            corr = float(torch.corrcoef(torch.stack([sh, st]))[0, 1]) if sh.numel() > 10 else float("nan")
            base_aux = float((st - st.mean()).abs().mean())
        rec = dict(step=it, loss=float(loss), l1=float(l1), aux=float(aux), aux_const=base_aux, corr=corr, sec=time.time() - t0)
        log.append(rec)
        print(f"[{a.name}] {it} L1 {float(l1):.4f} · 보조 {float(aux):.4f} (상수 예측 {base_aux:.4f}) · ŝ–s 상관 {corr:.3f} ({time.time() - t0:.0f}s)", flush=True)
os.makedirs(a.out, exist_ok=True)
torch.save({"predictor": pred.state_dict(), "stab_head": head.state_dict(), "args": vars(a), "log": log}, os.path.join(a.out, f"{a.name}.pt"))
print("saved", os.path.join(a.out, f"{a.name}.pt"), flush=True)
