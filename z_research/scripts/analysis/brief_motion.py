#!/usr/bin/env python3
"""간략 움직임 진단 (2026-10-17, 사용자 요청 — 100 샘플): AR (블록 문맥) vs ctxt_video (영상 문맥) 가 운동을 얼마나 잇나.

  python z_research/scripts/analysis/brief_motion.py --device cuda

샘플 = brief_diag.py 와 같은 규칙 (VOBench primitive pos_a, 움직이는 clip, late 70 + visible 30, seed 0).
1. 속도 probe: 문맥 끝 (블록 C-1) 의 물체 자리 3×3 토큰 평균 → 물체 속도 (블록당 이동, 토큰 단위 du · dv; 블록 C-3 → C-1 평균) ridge 5-fold R².
   지점: 문맥 표현 · predictor 4 · 12 층. 가려진 블록이면 "있어야 할 자리" 를 쓴다
2. 미래 위치 (파라미터 없음): 예측 블록 (걸음 r) 에서 정답 물체 모양 (정답 블록의 물체 자리 3×3 평균) 과 cosine 이 가장 큰 토큰 자리 →
   실제 물체 위치와의 거리 (토큰, 1 토큰 = 16 px @256). 같은 방법을 정답 블록 자신 (상한) · 블록 복사 (마지막 문맥 블록) 에도 적용
   물체가 보이는 블록만 센다 (가려진 걸음은 뺀다)
3. 위를 운동 종류별 (flat_vel · flat_acc · flat_dec · ramp_up · ramp_down)
출력: <out>/brief_motion.json + 표."""
import argparse, csv, json, os, sys, collections
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key
from src.models.rollout_predictor import vit_prefix_predictor
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["natural_ar_vw_state", "natural_ar_vw_state_ctxt_video"])
ap.add_argument("--vobench", default="/workspace/dataset/benchmarks/VOBench_v1")
ap.add_argument("--n_late", type=int, default=70)
ap.add_argument("--n_vis", type=int, default=30)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default=os.path.join(REPO, "z_research/VOBench_v1/figures/brief_diag"))
a = ap.parse_args(); dev = a.device; os.makedirs(a.out, exist_ok=True)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
S, C, K, G = 256, 8, 8, 16
LAYERS = [4, 12]; STEPS = list(range(K))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
IDX = list(range(0, 96, 3))[:32]
state = _load_checkpoint(CK)
tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tenc, state["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()

rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv")))
        if r["variant"] == "pos_a" and r["objset"] == "primitive" and r["motion"] != "flat_static"]
rng = np.random.default_rng(0)
late = [r for r in rows if r["occlusion"] == "late"]; vis = [r for r in rows if r["occlusion"] == "visible"]
pick = list(rng.permutation(late))[: a.n_late] + list(rng.permutation(vis))[: a.n_vis]

def load(r):
    d = os.path.join(a.vobench, r["file_name"])
    fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in IDX]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 255.0
    W0 = x.shape[-1]
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
    lab = {q["f"]: q for q in json.load(open(os.path.join(os.path.dirname(d), "label.json")))["labels"]["A"]}
    posf, hid = [], []
    for b in range(16):
        q = [lab[IDX[2 * b]], lab[IDX[2 * b + 1]]]
        posf.append((np.mean([t["v"] for t in q]) / W0 * G, np.mean([t["u"] for t in q]) / W0 * G)); hid.append(np.mean([t["hidden"] for t in q]))
    return (x - MEAN) / STD, np.array(posf), np.array(hid)

def rc(p):
    return int(np.clip(p[0], 0, G - 1)), int(np.clip(p[1], 0, G - 1))

def region(p):
    r0, c0 = rc(p)
    return [rr * G + cc for rr in range(max(0, r0 - 1), min(G, r0 + 2)) for cc in range(max(0, c0 - 1), min(G, c0 + 2))]

@torch.no_grad()
def blocks(x):
    xb = x.reshape(3, 16, 2, 256, 256).permute(1, 0, 2, 3, 4).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.cat([tenc(xb[i:i + 8]) for i in range(0, 16, 8)], 0).float()
    return F.layer_norm(h, (1280,)).reshape(1, 16 * S, -1)

def locate(blk, tmpl):
    """blk (S, D) · tmpl (D,) → cosine 최대 토큰의 (행, 열)."""
    c = F.cosine_similarity(blk, tmpl.unsqueeze(0), dim=-1)
    i = int(c.argmax()); return np.array([i // G, i % G], float)

class Model:
    def __init__(self, run):
        cfg = yaml.safe_load(open(os.path.join(REPO, "z_training/runs", run, "config.yaml")))
        pc = cfg["model"]["predictor"]; self.kind = pc.get("kind", "ar"); self.name = run
        kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384,
                  depth=12, num_heads=12, num_mask_tokens=10, mask_mode=self.kind)
        if self.kind == "ar":
            kw["ar_feedback"] = pc.get("ar_feedback", "output")
        self.pred = vit_prefix_predictor(**kw)
        sd = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(REPO, "z_training/runs", run, "latest.pt"),
              map_location="cpu", weights_only=False)["predictor"]).items() if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
        self.pred.load_state_dict(sd, strict=True); self.pred = self.pred.to(dev).eval()
        self.hid = {}
        for l in LAYERS:
            self.pred.predictor_blocks[l - 1].register_forward_hook(lambda m, i, o, l=l: self.hid.__setitem__(l, o.detach()))
        if self.kind != "ar":
            key = cfg["model"].get("context_encoder_key", "encoder")
            self.enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2 * C, use_rope=True, uniform_power=False)
            _load_state_dict(self.enc, state[key], tag=key); self.enc = self.enc.to(dev).eval()

    @torch.no_grad()
    def go(self, x, hb):
        idx = torch.arange(C * S, device=dev).unsqueeze(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = hb[:, : C * S] if self.kind == "ar" else self.enc(x[:, : 2 * C].unsqueeze(0).to(dev)).float()
            self.hid = {}
            if self.kind == "ar":
                self.pred.forward_seq(ctx, idx)
                p = self.pred.rollout_kv(ctx, idx, K).float()[0]
            else:
                self.pred.forward_state(ctx, ctx.new_zeros(1, 0, 384), idx, C)
                p = self.pred.render(self.pred.rollout_state_kv(ctx, idx, K)).float()[0]
        return ctx.float()[0], {l: self.hid[l].float()[0] for l in LAYERS}, p

models = [Model(r) for r in a.runs]
F_ = {m.name: collections.defaultdict(list) for m in models}; V = []; mot = []; occ = []
PE = {m.name: [] for m in models}; PE["copy"] = []; PE["target"] = []
for i, r in enumerate(pick):
    x, posf, hid = load(r); hb = blocks(x); tgt = hb[0]
    V.append((posf[C - 1] - posf[C - 3]) / 2.0); mot.append(r["motion"]); occ.append(r["occlusion"])
    R = region(posf[C - 1])
    for m in models:
        ctx, hd, p = m.go(x, hb)
        F_[m.name]["ctx_last"].append(ctx[(C - 1) * S + torch.tensor(R)].mean(0).cpu().numpy())
        for l in LAYERS:
            F_[m.name][f"pred_L{l}"].append(hd[l][(C - 1) * S + torch.tensor(R)].mean(0).cpu().numpy())
        e = []
        for st in STEPS:
            b = C + st
            if hid[b] >= 0.5:
                e.append(np.nan); continue
            tmpl = tgt[b * S + torch.tensor(region(posf[b]))].mean(0)
            e.append(np.linalg.norm(locate(p[st * S:(st + 1) * S], tmpl) - posf[b]))
        PE[m.name].append(e)
    ec, et = [], []
    for st in STEPS:
        b = C + st
        if hid[b] >= 0.5:
            ec.append(np.nan); et.append(np.nan); continue
        tmpl = tgt[b * S + torch.tensor(region(posf[b]))].mean(0)
        ec.append(np.linalg.norm(locate(tgt[(C - 1) * S: C * S], tmpl) - posf[b]))
        et.append(np.linalg.norm(locate(tgt[b * S:(b + 1) * S], tmpl) - posf[b]))
    PE["copy"].append(ec); PE["target"].append(et)
    if i % 25 == 0:
        print(f"{i}/{len(pick)}", flush=True)

def ridge_r2(X, Y):
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64); n = len(Y)
    folds = np.arange(n) % 5; np.random.default_rng(0).shuffle(folds); best = -9
    for lam in (10.0, 100.0, 1000.0, 10000.0):
        pr = np.zeros_like(Y)
        for f in range(5):
            tr, te = folds != f, folds == f
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; A = (X[tr] - mu) / sd; B = (X[te] - mu) / sd
            ym = Y[tr].mean(0)
            Wt = A.T @ np.linalg.solve(A @ A.T + lam * np.eye(A.shape[0]), Y[tr] - ym)
            pr[te] = B @ Wt + ym
        r2 = 1 - ((pr - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
        best = max(best, r2)
    return float(best)

occ = np.array(occ); mot = np.array(mot); V = np.array(V)
res = {"speed_tok_per_block_mean": float(np.linalg.norm(V, axis=1).mean()), "velocity_r2": {}, "pos_err": {}, "pos_err_by_motion": {}}
print(f"\n물체 평균 속도 {res['speed_tok_per_block_mean']:.2f} 토큰/블록 (1 토큰 = 16 px)")
print("\n[1] 속도 (du, dv) 선형 probe R² — late / visible")
for m in models:
    res["velocity_r2"][m.name] = {nm: {o: ridge_r2(np.array(F_[m.name][nm])[occ == o], V[occ == o]) for o in ("late", "visible")} for nm in F_[m.name]}
    print(f"  {m.name}: " + "  ".join(f"{nm} {v['late']:.2f}/{v['visible']:.2f}" for nm, v in res["velocity_r2"][m.name].items()))
print("\n[2] 미래 물체 위치 오차 (토큰, 물체 보이는 걸음만) — 걸음 0 … 7")
for o in ("late", "visible"):
    print(f"  ({o})")
    for nm, E in PE.items():
        E = np.array(E, float)[occ == o]
        mm = np.nanmean(E, 0); res["pos_err"].setdefault(nm, {})[o] = [None if np.isnan(v) else round(float(v), 2) for v in mm]
        print(f"   {nm[-22:]:>22s} " + " ".join("  -  " if np.isnan(v) else f"{v:5.2f}" for v in mm))
print("\n[3] 운동 종류별 미래 위치 오차 (걸음 3 ~ 7 평균, late + visible)")
for mv in sorted(set(mot)):
    sel = mot == mv
    line = {nm: float(np.nanmean(np.array(E, float)[sel][:, 3:])) for nm, E in PE.items()}
    res["pos_err_by_motion"][mv] = {k: round(v, 2) for k, v in line.items()}
    print(f"   {mv:10s} n{sel.sum():3d} " + "  ".join(f"{k[-13:]} {v:.2f}" for k, v in line.items()))
json.dump(res, open(f"{a.out}/brief_motion.json", "w"), indent=1)
