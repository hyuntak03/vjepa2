#!/usr/bin/env python3
"""kind=ar predictor 가 문맥의 어디를 보나 (2026-10-13). VOBench visible vs late, 이동 clip, 가능 arm (pos_a).

  python z_research/scripts/analysis/ar_attention_lag.py --run natural_ar_vw_state --n 40 --threads 128   # CPU
  (GPU 를 쓰려면 --device cuda)

채점과 같은 경로: stride 3 → 32 장 = 16 블록, 블록별 LN(target_encoder), 문맥 8 블록 → rollout 8 블록 (predictor.rollout_kv).
1. attention 시간 분포: rollout 동안 SDPA 를 감싸 softmax(qkᵀ/√d) 를 직접 계산, query 블록 → key 블록별 질량 (head·query 평균), 층별.
2. 물체 가시성: label.json 의 hidden 으로 블록별 물체가 보였는지. late 는 문맥 끝 블록에서 가려진다.
3. 문맥 자르기: 문맥을 마지막 L 블록만 남기고 (위치는 그대로) rollout → 예측 변화 / 원래 오차.
결과: <out>/ar_attention_lag.json + 표 출력. 파라미터 없음 (학습 없음)."""
import argparse, glob, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key
from src.models.rollout_predictor import vit_prefix_predictor
import src.models.utils.modules as MODS

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="natural_ar_vw_state")
ap.add_argument("--vobench", default="/workspace/dataset/benchmarks/VOBench_v1")
ap.add_argument("--n", type=int, default=40, help="조건 (visible / late) 마다 clip 수")
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=128)
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_num_threads(a.threads)
dev = a.device
RUN = os.path.join(REPO, "z_training/runs", a.run)
OUT = a.out or os.path.join(RUN, "eval/ar_attention_lag")
os.makedirs(OUT, exist_ok=True)
import yaml
cfg = yaml.safe_load(open(os.path.join(RUN, "config.yaml")))
pc = cfg["model"]["predictor"]
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc, _load_checkpoint(CK)["target_encoder"], tag="t")
enc = enc.to(dev).eval()
pred = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384,
                            depth=12, num_heads=12, num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
sd = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(RUN, "latest.pt"), map_location="cpu", weights_only=False)["predictor"]).items()
      if not k.startswith("hint_attn.")}
print("predictor load:", pred.load_state_dict(sd, strict=True))
pred = pred.to(dev).eval()
S, C, K, L = 256, 8, 8, 12
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
IDX = list(range(0, 96, 3))[:32]

# ---- SDPA 감싸기: 기록 중일 때만 가중치 계산 ----
_sdpa = F.scaled_dot_product_attention
REC = {"on": False, "calls": []}
def sdpa_rec(q, k, v, *args, **kw):
    if not REC["on"]:
        return _sdpa(q, k, v, *args, **kw)
    w = torch.softmax((q @ k.transpose(-2, -1)) / q.size(-1) ** 0.5, dim=-1)       # (B, h, Nq, Nk)
    nkb = k.size(2) // S
    mass = w.reshape(*w.shape[:3], nkb, S).sum(-1).mean(dim=(1, 2))                 # (B, nkb) query·head 평균
    REC["calls"].append(mass.detach().cpu().numpy())
    return w @ v
MODS.F.scaled_dot_product_attention = sdpa_rec

def load_clip(d):
    fr = [np.asarray(Image.open(f"{d}/pos_a/{i:06d}.png").convert("RGB")) for i in IDX]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 255.0
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
    x = (x - MEAN) / STD
    lab = {r["f"]: r for r in json.load(open(f"{d}/label.json"))["labels"]["A"]}
    vis = np.array([1 - (lab[IDX[2 * b]]["hidden"] + lab[IDX[2 * b + 1]]["hidden"]) / 2 for b in range(16)])
    return x, vis

@torch.no_grad()
def encode(x):
    xb = x.reshape(3, 16, 2, 256, 256).permute(1, 0, 2, 3, 4)                          # (16, 3, 2, H, W)
    h = torch.cat([enc(xb[i:i + 4].to(dev)) for i in range(0, 16, 4)], 0)               # (16, 256, D)
    return F.layer_norm(h, (h.size(-1),)).reshape(1, 16 * S, -1)

@torch.no_grad()
def rollout(h, keep):
    """문맥 마지막 keep 블록만 (절대 위치 유지). return (예측 (K*S, D), 기록 [층 × 걸음] 질량 목록)."""
    c0 = C - keep
    idx = torch.arange(c0 * S, C * S, device=dev).unsqueeze(0)
    REC["calls"] = []; REC["on"] = True
    p = pred.rollout_kv(h[:, c0 * S: C * S], idx, K)
    REC["on"] = False
    return p[0], REC["calls"]

MOV = ("flat_vel", "flat_acc", "flat_dec", "ramp_down")
res = {}
for occ in ("visible", "late"):
    pat = f"{a.vobench}/*/*/{occ}/*" if occ == "visible" else f"{a.vobench}/*/*/{occ}_k*/*"     # 가림은 <시점>_k{1,2,3}
    dirs = sorted(d for d in glob.glob(pat) if any(m in os.path.basename(d) for m in MOV))
    rng = np.random.default_rng(0); dirs = list(rng.permutation(dirs))[: a.n]
    lagm = np.zeros((L, K, C + K)); visw = []; ko = {1: [], 2: [], 4: []}; t0 = time.time()
    for ci, d in enumerate(dirs):
        x, vis = load_clip(d)
        h = encode(x)
        p, calls = rollout(h, C)
        # calls: 문맥 C 블록 (블록마다 층 L 번) + rollout K-1 걸음 (걸음마다 층 L 번). rollout 예측 걸음 r 의 query 블록 = C-1+r
        calls = calls[(C - 1) * L:]                                                    # 문맥 마지막 블록 (=걸음 0 예측) 부터
        for r in range(K):
            for l in range(L):
                m = calls[r * L + l][0]                                                # (C+r,) key 블록별 질량
                lagm[l, r, : len(m)] += m[::-1]                                        # lag 0 = 자기 블록
        # 물체가 보였던 문맥 블록이 받는 질량 (rollout 걸음 평균, 층 평균)
        visw.append((vis[:C], np.mean([np.mean([calls[r * L + l][0][:C] for l in range(L)], 0) for r in range(K)], 0)))
        tgt = h[0, C * S:]
        base = (p - tgt).abs().mean().item()
        for keep in ko:
            pk, _ = rollout(h, keep)
            ko[keep].append([((pk - p)[r * S:(r + 1) * S].abs().mean().item() / max(1e-9, (p - tgt)[r * S:(r + 1) * S].abs().mean().item()))
                             for r in range(K)])
        if ci % 10 == 0:
            print(f"[{occ}] {ci}/{len(dirs)} {time.time() - t0:.0f}s", flush=True)
    n = len(dirs)
    lagm /= n
    res[occ] = {"n": n, "lag_mass": lagm.tolist(),
                "vis": [v.tolist() for v, _ in visw], "ctx_mass": [w.tolist() for _, w in visw],
                "knockout_rel_change": {str(k): np.mean(v, 0).tolist() for k, v in ko.items()}}

json.dump(res, open(f"{OUT}/ar_attention_lag.json", "w"))
# ---- 표 ----
for occ, R in res.items():
    lm = np.array(R["lag_mass"])                                                       # (L, K, C+K)
    print(f"\n=== {occ} (n={R['n']}) — rollout 걸음 r 에서 query 가 lag 블록 전에 준 attention 질량 (층 평균) ===")
    print("걸음 | " + " ".join(f"lag{j:>2}" for j in range(10)) + " | 문맥 전체 · 자기 rollout")
    for r in range(K):
        row = lm[:, r].mean(0)
        ctx = row[r:r + C].sum(); own = row[:r].sum()                                  # lag < r = rollout 이 만든 블록, lag ≥ r = 문맥
        print(f" r{r}  | " + " ".join(f"{row[j]:.3f}" for j in range(10)) + f" | {ctx:.3f} · {own:.3f}")
    lay = lm.mean(1)
    print("층별 lag0 질량 (걸음 평균):", " ".join(f"{lay[l, 0]:.2f}" for l in range(L)))
    print("층별 lag≥2 질량:", " ".join(f"{lay[l, 2:].sum():.2f}" for l in range(L)))
    ko = R["knockout_rel_change"]
    for k, v in ko.items():
        if not isinstance(v, list):
            continue
        print(f"문맥 마지막 {k} 블록만 남기면 예측 변화 / 원래 오차 (걸음 0..7):", " ".join(f"{x:.2f}" for x in v))
    vis = np.array(R["vis"]); cm = np.array(R["ctx_mass"])
    if occ == "late":
        hid = vis < 0.5
        print(f"late: 문맥 블록 중 물체 가려진 블록 비율 {hid.mean():.2f} | 물체 보이던 블록이 받는 질량 평균 {cm[~hid].mean():.4f} vs 가려진 블록 {cm[hid].mean():.4f}")
