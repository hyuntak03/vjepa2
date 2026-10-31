#!/usr/bin/env python3
"""인접 두 블록 토큰에서 칸별 광학 흐름이 읽히나 — 고정 encoder 토큰 (1280) vs 학습된 AR 의 입력 투영 (384) (2026-10-27, 사용자 요청).
목적: "입력 투영을 MLP 로 바꾸고 384 차이에 흐름 head" 안의 전제 (지금 1280 → 384 선형 투영이 움직임을 버린다) 를 학습 전에 확인.

  CUDA_VISIBLE_DEVICES=7 python z_research/scripts/analysis/flow_probe_tokens.py --run natural_ar_vw_state latest.pt --n 40

데이터: 실제 영상 (K710 · SSv2 · HowTo, fps10t 표, 16 장 = 8 블록) 각 --n clip · VOBench visible pos_a --n clip (30 fps 3 장 간격 32 장 = 16 블록).
정답 흐름: RAFT 장 2b → 2b+2 (블록 b → b+1, 0.2 초), 16×16 칸 평균, 칸 단위.
특징 (칸마다): ① Δh = h_{b+1} − h_b (1280, LN target encoder)  ② [h_b, h_{b+1}] (2560)
              ③ Δz = E(h_{b+1}) − E(h_b) (384, E = AR 의 predictor_embed)  ④ [z_b, z_{b+1}] (768)
읽기: 선형 (ridge) 과 작은 MLP (은닉 512, 2 층) — clip 단위 반 맞춤 / 반 확인. 보고: 흐름 (dx, dy) R² (전체 칸 · 움직이는 칸 |흐름| ≥ 0.25) · 움직이는 칸 방향 cos 중앙값."""
import argparse, csv, glob, os, random, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml, decord
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key
from app.vjepa_frozen import ar as AR
from app.vjepa_frozen.var_window import _Table
ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs=2, default=["natural_ar_vw_state", "latest.pt"])
ap.add_argument("--n", type=int, default=40)
a = ap.parse_args()
dev = torch.device("cuda"); G = 16; torch.manual_seed(0)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, _load_checkpoint(CK)["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
sd = _clean_backbone_key(torch.load(os.path.join(REPO, "z_training/runs", a.run[0], a.run[1]), map_location="cpu", weights_only=False)["predictor"])
E = nn.Linear(1280, 384).to(dev); E.load_state_dict({"weight": sd["predictor_embed.weight"], "bias": sd["predictor_embed.bias"]}); E.eval()
print(f"입력 투영 ← {a.run[0]}:{a.run[1]}", flush=True)

def nat_jobs(name, path):
    t = _Table(name, os.path.join(REPO, path), 1.0); lo = t.first_ok(16); r = random.Random(f"fpt-{name}")
    return [(name, t.paths[j], t.fps[j], r.random()) for j in (r.randrange(lo, len(t.paths)) for _ in range(a.n))]
jobs = nat_jobs("k710", "data_csv/k710_320/fps10t_table.tsv") + nat_jobs("ssv2", "data_csv/ssv2/fps10t_table.tsv") + nat_jobs("how", "data_csv/ht100m_1pct/fps10t_table.tsv")
vb = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == "visible" and r["variant"] == "pos_a"]
jobs += [("vobench", os.path.join("/workspace/dataset/benchmarks/VOBench_v1", vb[i]["file_name"]), 30, None) for i in np.random.default_rng(0).permutation(len(vb))[: a.n]]

def load(job):
    name, p, f, u = job
    if name == "vobench":
        x = torch.stack([torch.from_numpy(np.asarray(Image.open(os.path.join(p, f"{3 * k:06d}.png")).convert("RGB"))) for k in range(32)])
    else:
        vr = decord.VideoReader(p, num_threads=2); n = len(vr); tmax = (n - 1) / f - 1.5
        if tmax < 0: return None
        t0 = u * tmax; x = torch.from_numpy(vr.get_batch([min(n - 1, int(round((t0 + k / 10) * f))) for k in range(16)]).asnumpy())
    x = x.to(dev).permute(0, 3, 1, 2).float(); H, W = x.shape[-2:]; s = 256 / min(H, W)
    x = F.interpolate(x, size=(max(256, round(H * s)), max(256, round(W * s))), mode="bilinear", align_corners=False)
    H, W = x.shape[-2:]; return x[:, :, (H - 256) // 2:(H - 256) // 2 + 256, (W - 256) // 2:(W - 256) // 2 + 256]

D = {}      # src → list of (clip id, h_b, h_n, flow) per 블록 쌍, 칸 펼침
for ci, job in enumerate(jobs):
    try: x = load(job)
    except Exception as e: print("skip", e); continue
    if x is None: continue
    nb = x.size(0) // 2
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = AR.encode_blocks(tgt, ((x.permute(1, 0, 2, 3)[None] / 255 - MEAN) / STD).to(torch.bfloat16), 2).float()[0].view(nb, G * G, -1)
        a1 = x[0:2 * nb - 2:2] / 127.5 - 1; a2 = x[2:2 * nb:2] / 127.5 - 1
        with torch.autocast("cuda", dtype=torch.float16):
            fl = raft(a1.contiguous(), a2.contiguous(), num_flow_updates=12)[-1].float()     # 2b → 2b+2 (앞으로)
        fc = (F.avg_pool2d(fl, 16) / 16).flatten(2).transpose(1, 2)                          # (nb-1, 256, 2)
    D.setdefault(job[0], []).append((h[:-1].half().cpu(), h[1:].half().cpu(), fc.cpu()))
    if ci % 40 == 0: print(f"{ci}/{len(jobs)}", flush=True)

def r2(p, y): return float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
def fit_eval(Xa, Ya, Xb, Yb, mlp):
    if not mlp:
        Xa1 = torch.cat([Xa, torch.ones(len(Xa), 1, device=dev)], 1); Xb1 = torch.cat([Xb, torch.ones(len(Xb), 1, device=dev)], 1)
        W = torch.linalg.solve(Xa1.T @ Xa1 + 1.0 * torch.eye(Xa1.size(1), device=dev), Xa1.T @ Ya)
        return Xb1 @ W
    m = nn.Sequential(nn.LayerNorm(Xa.size(1)), nn.Linear(Xa.size(1), 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 2)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(15):
        perm = torch.randperm(len(Xa), device=dev)
        for i in range(0, len(Xa), 4096):
            j = perm[i:i + 4096]; l = F.smooth_l1_loss(m(Xa[j]), Ya[j]); opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad(): return m(Xb)

print(f"\n흐름 읽기 (확인 반, R² 전체 칸 / 움직이는 칸 · 움직이는 칸 방향 cos 중앙)")
print(f"{'데이터':8s} {'특징':22s} {'선형':>22s} {'MLP':>22s}")
for src, L in D.items():
    half = len(L) // 2
    def cat(sl, k):
        return torch.cat([torch.cat([l[k].reshape(-1, l[k].size(-1)) for l in sl])]).to(dev)
    ha, na, fa = cat(L[:half], 0).float(), cat(L[:half], 1).float(), cat(L[:half], 2).float()
    hb, nb_, fb = cat(L[half:], 0).float(), cat(L[half:], 1).float(), cat(L[half:], 2).float()
    with torch.no_grad():
        za, zna, zb, znb = E(ha), E(na), E(hb), E(nb_)
    mv = fb.norm(dim=1) >= 0.25
    feats = {"① Δh (1280)": (na - ha, nb_ - hb), "② [h_b, h_b+1] (2560)": (torch.cat([ha, na], 1), torch.cat([hb, nb_], 1)),
             "③ Δz 투영 (384)": (zna - za, znb - zb), "④ [z_b, z_b+1] 투영 (768)": (torch.cat([za, zna], 1), torch.cat([zb, znb], 1))}
    for fn, (Xa, Xb) in feats.items():
        cells = []
        for mlp in (False, True):
            if not mlp and Xa.size(1) > 1300: cells.append("—"); continue
            p = fit_eval(Xa, fa, Xb, fb, mlp)
            cos = F.cosine_similarity(p[mv], fb[mv], dim=1).median().item() if mv.any() else float("nan")
            cells.append(f"{r2(p, fb):5.2f} / {r2(p[mv], fb[mv]):5.2f} · {cos:4.2f}")
        print(f"{src:8s} {fn:22s} {cells[0]:>22s} {cells[1]:>22s}   (칸 {len(fb):,}, 움직임 {int(mv.sum()):,})", flush=True)
