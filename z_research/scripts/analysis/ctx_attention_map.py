#!/usr/bin/env python3
"""문맥 attention 지도 — AR (블록 문맥) vs ctxt_video (영상 문맥), VOBench late · visible (2026-10-17).

  python z_research/scripts/analysis/ctx_attention_map.py --runs natural_ar_vw_state natural_ar_vw_state_ctxt_video --n 40 --device cuda

무엇을 재나: **첫 예측 블록 (블록 C) 을 만드는 query** (= 마지막 문맥 자리의 256 토큰) 가 문맥 토큰 (C 블록 × 256) 에 준 attention.
  층 · head · query 평균. 채점 (testbed c16t32) 과 같은 프레임: VOBench 30 fps PNG 를 3 장 간격 → 32 장 = 16 블록, 문맥 8 블록.
  * AR (kind=ar)       : 문맥 = 블록마다 따로 target_encoder → LN(h). 블록 C−1 을 채울 때의 attention (KV 캐시 경로 그대로)
  * ctxt_video (ctx_state): 문맥 = encoder 로 문맥 16 장 한 번 (영상). prefill 의 마지막 문맥 블록 query 만 골라 기록
  ⚠️ 영상 문맥의 "블록 b 토큰" 은 encoder 가 시간 양방향으로 섞은 표현이다 — 블록 b 에 준 attention 이 곧 프레임 b 를 봤다는 뜻은 아니다
출력: <out>/ctx_attention.json (조건 × run: 블록별 질량, 물체 보이던 / 가려진 블록 질량) + fig_ctx_attention_<occ>.png (clip × 문맥 블록 heatmap)
파라미터 없음 (학습 없음)."""
import argparse, glob, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key
from src.models.rollout_predictor import vit_prefix_predictor
import src.models.utils.modules as MODS
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["natural_ar_vw_state", "natural_ar_vw_state_ctxt_video"])
ap.add_argument("--vobench", default="/workspace/dataset/benchmarks/VOBench_v1")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--n_fig", type=int, default=4)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default=os.path.join(REPO, "z_research/VOBench_v1/figures/ctx_attention"))
a = ap.parse_args()
dev = a.device
os.makedirs(a.out, exist_ok=True)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
S, C, G = 256, 8, 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
IDX = list(range(0, 96, 3))[:32]
state = _load_checkpoint(CK)

# ---- SDPA 감싸기: 기록 중일 때 query 구간을 골라 key 토큰별 가중 (head · query 평균) ----
_sdpa = F.scaled_dot_product_attention
REC = {"on": False, "q": None, "calls": []}
def sdpa_rec(q, k, v, *args, **kw):
    if not REC["on"]:
        return _sdpa(q, k, v, *args, **kw)
    qs = q if REC["q"] is None else q[:, :, REC["q"][0]:REC["q"][1]]
    w = torch.softmax((qs.float() @ k.float().transpose(-2, -1)) / q.size(-1) ** 0.5, dim=-1)   # (B, h, nq, Nk)
    REC["calls"].append(w.mean(dim=(1, 2))[0].cpu().numpy())                                    # (Nk,)
    return _sdpa(q, k, v, *args, **kw)
MODS.F.scaled_dot_product_attention = sdpa_rec

def load_clip(d):
    fr = [np.asarray(Image.open(f"{d}/pos_a/{i:06d}.png").convert("RGB")) for i in IDX]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 255.0
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
    lab = {r["f"]: r for r in json.load(open(f"{d}/label.json"))["labels"]["A"]}
    vis = np.array([1 - (lab[IDX[2 * b]]["hidden"] + lab[IDX[2 * b + 1]]["hidden"]) / 2 for b in range(16)])
    return (x - MEAN) / STD, x, vis

class Model:
    def __init__(self, run):
        cfg = yaml.safe_load(open(os.path.join(REPO, "z_training/runs", run, "config.yaml")))
        pc = cfg["model"]["predictor"]; self.kind = pc.get("kind", "ar"); self.run = run
        kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384,
                  depth=12, num_heads=12, num_mask_tokens=10, mask_mode=self.kind)
        if self.kind == "ar":
            kw["ar_feedback"] = pc.get("ar_feedback", "output")
        self.pred = vit_prefix_predictor(**kw)
        sd = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(REPO, "z_training/runs", run, "latest.pt"),
              map_location="cpu", weights_only=False)["predictor"]).items() if not k.startswith(("hint_attn.", "dyn_head."))}
        print(run, self.kind, self.pred.load_state_dict(sd, strict=True))
        self.pred = self.pred.to(dev).eval()
        self.L = len(self.pred.predictor_blocks)
        if self.kind == "ar":
            self.enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
            _load_state_dict(self.enc, state["target_encoder"], tag="t")
        else:
            key = cfg["model"].get("context_encoder_key", "encoder")
            self.enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2 * C, use_rope=True, uniform_power=False)
            _load_state_dict(self.enc, state[key], tag=key)
        self.enc = self.enc.to(dev).eval()

    @torch.no_grad()
    def attention(self, x):
        """return (L, C*S) 첫 예측 query 가 문맥 토큰에 준 attention."""
        xc = x[:, : 2 * C]
        idx = torch.arange(C * S, device=dev).unsqueeze(0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            if self.kind == "ar":
                xb = xc.reshape(3, C, 2, 256, 256).permute(1, 0, 2, 3, 4).to(dev)
                h = F.layer_norm(self.enc(xb).float(), (1280,)).reshape(1, C * S, -1)
                REC["calls"] = []; REC["q"] = None; REC["on"] = True
                self.pred.rollout_kv(h, idx, 1)                          # 블록 0..C-1 을 하나씩 채운다 — 마지막 블록 (C-1) 의 층별 호출이 첫 예측
                REC["on"] = False
                calls = REC["calls"][-self.L:]                           # 블록 C-1 단계: key = 블록 0..C-1
            else:
                z = self.enc(xc.unsqueeze(0).to(dev)).float()             # (1, C*S, D) 문맥 영상 한 번
                REC["calls"] = []; REC["q"] = ((C - 1) * S, C * S); REC["on"] = True
                self.pred.rollout_state_kv(z, idx, 1)                    # prefill 의 마지막 문맥 블록 query = 블록 C 의 상태
                REC["on"] = False; REC["q"] = None
                calls = REC["calls"][: self.L]
        return np.stack(calls)

models = [Model(r) for r in a.runs]
MOV = ("flat_vel", "flat_acc", "flat_dec", "ramp_down")
res = {}
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
for occ in ("late", "visible"):
    pat = f"{a.vobench}/*/*/{occ}/*" if occ == "visible" else f"{a.vobench}/*/*/{occ}_k*/*"
    dirs = sorted(d for d in glob.glob(pat) if any(m in os.path.basename(d) for m in MOV))
    dirs = list(np.random.default_rng(0).permutation(dirs))[: a.n]
    acc = {m.run: {"blk": [], "vis_mass": [], "hid_mass": []} for m in models}
    figdat = []
    t0 = time.time()
    for ci, d in enumerate(dirs):
        x, raw, vis = load_clip(d)
        maps = {}
        for m in models:
            A = m.attention(x).mean(0)                                   # (C*S,) 층 평균
            blk = A.reshape(C, S).sum(1)
            acc[m.run]["blk"].append(blk)
            hid = vis[:C] < 0.5
            if hid.any() and (~hid).any():
                acc[m.run]["vis_mass"].append(blk[~hid].mean()); acc[m.run]["hid_mass"].append(blk[hid].mean())
            maps[m.run] = A.reshape(C, G, G)
        if ci < a.n_fig:
            figdat.append((d, raw, vis, maps))
        if ci % 10 == 0:
            print(f"[{occ}] {ci}/{len(dirs)} {time.time() - t0:.0f}s", flush=True)
    res[occ] = {r: {"n": len(v["blk"]), "block_mass": np.mean(v["blk"], 0).tolist(),
                    "mass_visible_block": float(np.mean(v["vis_mass"])) if v["vis_mass"] else None,
                    "mass_hidden_block": float(np.mean(v["hid_mass"])) if v["hid_mass"] else None} for r, v in acc.items()}
    # 그림: clip 마다 행 = 원본 문맥 프레임 + run 별 heatmap
    nr = len(figdat) * (1 + len(models))
    fig, ax = plt.subplots(nr, C, figsize=(C * 1.4, nr * 1.4))
    for i, (d, raw, vis, maps) in enumerate(figdat):
        base = i * (1 + len(models))
        for b in range(C):
            img = raw[:, 2 * b + 1].permute(1, 2, 0).numpy().clip(0, 1)
            ax[base, b].imshow(img); ax[base, b].set_xticks([]); ax[base, b].set_yticks([])
            ax[base, b].set_title(f"blk {b}" + (" (hidden)" if vis[b] < 0.5 else ""), fontsize=6)
            for j, m in enumerate(models):
                hm = maps[m.run][b]; aa = ax[base + 1 + j, b]
                aa.imshow(img); aa.imshow(np.kron(hm, np.ones((16, 16))), cmap="inferno", alpha=0.6,
                                         vmin=0, vmax=max(1e-9, max(maps[m.run].max() for _ in [0])))
                aa.set_xticks([]); aa.set_yticks([])
                if b == 0:
                    aa.set_ylabel(("AR (block ctx)" if m.kind == "ar" else "ctxt_video") + f"\n{maps[m.run][b].sum():.2f}", fontsize=6)
                else:
                    aa.set_ylabel(f"{maps[m.run][b].sum():.2f}", fontsize=6)
        ax[base, 0].set_ylabel(os.path.basename(d)[:22], fontsize=5)
    fig.suptitle(f"VOBench {occ}: attention of the first-prediction queries over context tokens (layer/head/query mean). "
                 f"Numbers = mass per block", fontsize=8)
    plt.tight_layout(); fig.savefig(f"{a.out}/fig_ctx_attention_{occ}.png", dpi=90); plt.close(fig)

json.dump(res, open(f"{a.out}/ctx_attention.json", "w"), indent=1)
for occ, R in res.items():
    print(f"\n=== {occ} — 첫 예측 query 의 문맥 블록별 attention 질량 (블록 0 … 7, 층 평균) ===")
    for r, v in R.items():
        print(f"{r:34s} n{v['n']} | " + " ".join(f"{x:.3f}" for x in v["block_mass"])
              + (f" | 물체 보이던 블록 {v['mass_visible_block']:.3f} vs 가려진 블록 {v['mass_hidden_block']:.3f}" if v["mass_hidden_block"] is not None else ""))
