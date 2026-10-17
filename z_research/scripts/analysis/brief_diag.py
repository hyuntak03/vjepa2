#!/usr/bin/env python3
"""간략 진단 (2026-10-17, 사용자 요청 — 100 샘플): AR (블록 문맥) vs ctxt_video (영상 문맥) 가 무엇을 못하나.

  python z_research/scripts/analysis/brief_diag.py --device cuda

데이터: VOBench primitive, 가능 arm (pos_a), 움직이는 clip — late 70 + visible 30. 채점과 같은 프레임 (30 fps PNG 3 장 간격 → 32 장 = 16 블록, 문맥 8 블록).
1. 선형 probe (물체 모양 6 종 · 색 8 종) — 물체 자리 3×3 토큰 평균을 입력으로, ridge 닫힌 해 + 5-fold (λ 는 격자에서 CV 최고, 약간 낙관적).
   지점: 마지막 문맥 블록 표현 (AR: 블록 LN(h) / ctxt: 영상 z) · predictor 4 · 8 · 12 층 (첫 예측 자리) · 예측 출력 (걸음 0 · 3 · 7) · 정답 (상한)
   물체 자리 = 그 블록 프레임의 label (u, v) — 가려진 블록이면 "물체가 있어야 할 자리"
2. attention (predictor 1 · 4 · 8 · 12 층): 첫 예측 query (마지막 문맥 블록의 물체 자리 3×3) 가 준 질량 —
   (a) 마지막 블록 (b) 앞 블록들 중 물체가 보이던 블록의 물체 자리 (c) 앞 블록의 나머지
3. 예측 오차 (L1) 를 물체 자리 / 배경으로 나눠, 블록 복사 기준선과 비교 (걸음 0 · 3 · 7)
출력: <out>/brief_diag.json + 표. 파라미터 학습 없음 (probe 만)."""
import argparse, csv, json, os, sys
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
ap.add_argument("--n_late", type=int, default=70)
ap.add_argument("--n_vis", type=int, default=30)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default=os.path.join(REPO, "z_research/VOBench_v1/figures/brief_diag"))
a = ap.parse_args(); dev = a.device; os.makedirs(a.out, exist_ok=True)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
S, C, K, G = 256, 8, 8, 16
LAYERS = [1, 4, 8, 12]; STEPS = [0, 3, 7]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
IDX = list(range(0, 96, 3))[:32]
state = _load_checkpoint(CK)
tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tenc, state["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()

# ---- 샘플 ----
rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv")))
        if r["variant"] == "pos_a" and r["objset"] == "primitive" and r["motion"] != "flat_static"]
rng = np.random.default_rng(0)
late = [r for r in rows if r["occlusion"] == "late"]; vis = [r for r in rows if r["occlusion"] == "visible"]
pick = list(rng.permutation(late))[: a.n_late] + list(rng.permutation(vis))[: a.n_vis]
SH = sorted({r["obj_a"].split("/")[0] for r in pick}); CO = sorted({r["obj_a"].split("/")[1] for r in pick})

def load(r):
    d = os.path.join(a.vobench, r["file_name"])
    fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in IDX]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 255.0
    W0 = x.shape[-1]
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
    lab = {q["f"]: q for q in json.load(open(os.path.join(os.path.dirname(d), "label.json")))["labels"]["A"]}
    pos, hid = [], []
    for b in range(16):
        q = [lab[IDX[2 * b]], lab[IDX[2 * b + 1]]]
        u = np.mean([t["u"] for t in q]) / W0 * G; v = np.mean([t["v"] for t in q]) / W0 * G
        pos.append((int(np.clip(v, 0, G - 1)), int(np.clip(u, 0, G - 1)))); hid.append(np.mean([t["hidden"] for t in q]))
    return (x - MEAN) / STD, pos, np.array(hid)

def region(rc):     # 3×3 토큰 인덱스 (블록 안)
    r0, c0 = rc
    return [rr * G + cc for rr in range(max(0, r0 - 1), min(G, r0 + 2)) for cc in range(max(0, c0 - 1), min(G, c0 + 2))]

@torch.no_grad()
def blocks(x):
    xb = x.reshape(3, 16, 2, 256, 256).permute(1, 0, 2, 3, 4).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.cat([tenc(xb[i:i + 8]) for i in range(0, 16, 8)], 0).float()
    return F.layer_norm(h, (1280,)).reshape(1, 16 * S, -1)

# ---- attention 기록 (선택 층, 첫 예측 query 의 물체 자리) ----
_sdpa = F.scaled_dot_product_attention
REC = {"on": False, "qidx": None, "calls": []}
def sdpa_rec(q, k, v, *args, **kw):
    if REC["on"]:
        qi = None
        if q.size(2) == k.size(2) == C * S:                 # ctx_state: 문맥 한 번에 (query 절대 인덱스)
            qi = REC["qidx"]
        elif q.size(2) == S and k.size(2) == C * S:         # ar: 블록별로 나뉜 호출 중 마지막 문맥 블록 query → 문맥 전체 key
            qi = [t - (C - 1) * S for t in REC["qidx"]]
        if qi is not None:
            qs = q[:, :, qi]
            w = torch.softmax((qs.float() @ k.float().transpose(-2, -1)) / q.size(-1) ** 0.5, dim=-1).mean(dim=(1, 2))[0]
            REC["calls"].append(w.cpu().numpy())
    return _sdpa(q, k, v, *args, **kw)
MODS.F.scaled_dot_product_attention = sdpa_rec

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
    def go(self, x, hb, pos):
        idx = torch.arange(C * S, device=dev).unsqueeze(0)
        q = [(C - 1) * S + t for t in region(pos[C - 1])]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = hb[:, : C * S] if self.kind == "ar" else self.enc(x[:, : 2 * C].unsqueeze(0).to(dev)).float()
            REC["calls"] = []; REC["qidx"] = q; REC["on"] = True; self.hid = {}
            if self.kind == "ar":
                self.pred.forward_seq(ctx, idx)                                     # 위치 C-1 출력 = 블록 C 예측 (rollout 첫 걸음과 같다)
            else:
                self.pred.forward_state(ctx, ctx.new_zeros(1, 0, 384), idx, C)
            REC["on"] = False
            att = [REC["calls"][l - 1] for l in LAYERS]
            hid = {l: self.hid[l].float()[0] for l in LAYERS}                        # (N, 384) 층 출력
            if self.kind == "ar":
                p = self.pred.rollout_kv(ctx, idx, K).float()[0]
            else:
                p = self.pred.render(self.pred.rollout_state_kv(ctx, idx, K)).float()[0]
        return ctx.float()[0], hid, att, p

models = [Model(r) for r in a.runs]
feats = {m.run: {} for m in models}; Y = {"shape": [], "color": []}
attn = {m.run: {l: [] for l in LAYERS} for m in models}
err = {m.run: {"obj": [], "bg": []} for m in models}; errc = {"obj": [], "bg": []}; occ = []
def add(run, name, v):
    feats[run].setdefault(name, []).append(v.cpu().numpy())
for i, r in enumerate(pick):
    x, pos, hid = load(r); hb = blocks(x)
    Y["shape"].append(SH.index(r["obj_a"].split("/")[0])); Y["color"].append(CO.index(r["obj_a"].split("/")[1])); occ.append(r["occlusion"])
    tgt = hb[0]
    for m in models:
        ctx, hd, att, p = m.go(x, hb, pos)
        R = region(pos[C - 1])
        add(m.run, "ctx_last", ctx[(C - 1) * S + torch.tensor(R)].mean(0))
        for l in LAYERS:
            add(m.run, f"pred_L{l}", hd[l][(C - 1) * S + torch.tensor(R)].mean(0))
        for st in STEPS:
            Rs = region(pos[C + st]); add(m.run, f"out_r{st}", p[st * S + torch.tensor(Rs)].mean(0))
        # attention: 첫 예측 query → (a) 마지막 블록 (b) 물체 보이던 앞 블록의 물체 자리 (c) 앞 블록 나머지
        for l, w in zip(LAYERS, att):
            w = w[: C * S]; last = w[(C - 1) * S:].sum()
            seen = [b for b in range(C - 1) if hid[b] < 0.5]
            objm = sum(w[b * S + np.array(region(pos[b]))].sum() for b in seen)
            attn[m.run][l].append([last, objm, w[: (C - 1) * S].sum() - objm, len(seen)])
        # 예측 오차 (물체 자리 vs 배경)
        eo, eb = [], []
        for st in STEPS:
            Rs = np.array(region(pos[C + st])); msk = np.zeros(S, bool); msk[Rs] = True
            d = (p[st * S:(st + 1) * S] - tgt[(C + st) * S:(C + st + 1) * S]).abs().mean(-1).cpu().numpy()
            eo.append(d[msk].mean()); eb.append(d[~msk].mean())
        err[m.run]["obj"].append(eo); err[m.run]["bg"].append(eb)
    for st in STEPS:
        pass
    co, cb = [], []
    for st in STEPS:
        Rs = np.array(region(pos[C + st])); msk = np.zeros(S, bool); msk[Rs] = True
        d = (tgt[(C - 1) * S: C * S] - tgt[(C + st) * S:(C + st + 1) * S]).abs().mean(-1).cpu().numpy()
        co.append(d[msk].mean()); cb.append(d[~msk].mean())
    errc["obj"].append(co); errc["bg"].append(cb)
    for st in STEPS:
        Rs = region(pos[C + st]); feats.setdefault("_tgt", {}).setdefault(f"tgt_r{st}", []).append(tgt[(C + st) * S + torch.tensor(Rs)].mean(0).cpu().numpy())
    if i % 20 == 0:
        print(f"{i}/{len(pick)}", flush=True)

# ---- ridge probe (닫힌 해, 5-fold, λ 격자 중 CV 최고) ----
def probe(X, y, ncls, mask=None):
    X = np.asarray(X, np.float64); y = np.asarray(y)
    if mask is not None:
        X, y = X[mask], y[mask]
    n = len(y); folds = np.arange(n) % 5; np.random.default_rng(0).shuffle(folds)
    best = 0
    for lam in (1.0, 10.0, 100.0, 1000.0):
        ok = 0
        for f in range(5):
            tr, te = folds != f, folds == f
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            A = (X[tr] - mu) / sd; B = (X[te] - mu) / sd
            T = np.eye(ncls)[y[tr]]
            Wt = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ T) if A.shape[1] <= A.shape[0] else A.T @ np.linalg.solve(A @ A.T + lam * np.eye(A.shape[0]), T)
            ok += (np.argmax(B @ Wt, 1) == y[te]).sum()
        best = max(best, ok / n)
    return 100 * best

occ = np.array(occ); res = {"n": len(pick), "n_late": int((occ == "late").sum()), "n_vis": int((occ == "visible").sum()),
                             "chance": {"shape": 100 / len(SH), "color": 100 / len(CO)}, "probe": {}, "attn": {}, "err": {}}
names = ["ctx_last"] + [f"pred_L{l}" for l in LAYERS] + [f"out_r{s}" for s in STEPS]
for m in models:
    res["probe"][m.run] = {nm: {t: {sub: probe(feats[m.run][nm], Y[t], len(SH) if t == "shape" else len(CO), occ == sub if sub != "all" else None)
                                    for sub in ("late", "visible")} for t in ("shape", "color")} for nm in names}
    A_ = {l: np.array(attn[m.run][l]) for l in LAYERS}
    res["attn"][m.run] = {l: {occ_: dict(zip(["last_block", "obj_in_seen_earlier", "other_earlier"], A_[l][occ == occ_, :3].mean(0).round(4).tolist()))
                              for occ_ in ("late", "visible")} for l in LAYERS}
    res["err"][m.run] = {k: {occ_: np.array(v)[occ == occ_].mean(0).round(4).tolist() for occ_ in ("late", "visible")} for k, v in err[m.run].items()}
res["probe"]["target"] = {nm: {t: {sub: probe(feats["_tgt"][nm], Y[t], len(SH) if t == "shape" else len(CO), occ == sub) for sub in ("late", "visible")}
                               for t in ("shape", "color")} for nm in feats["_tgt"]}
res["err"]["copy"] = {k: {occ_: np.array(v)[occ == occ_].mean(0).round(4).tolist() for occ_ in ("late", "visible")} for k, v in errc.items()}
json.dump(res, open(f"{a.out}/brief_diag.json", "w"), indent=1, default=float)

print(f"\nn={res['n']} (late {res['n_late']} · visible {res['n_vis']}) | 우연: 모양 {res['chance']['shape']:.1f} · 색 {res['chance']['color']:.1f}")
print("\n[1] 선형 probe 정확도 (%) — late / visible")
for t in ("shape", "color"):
    print(f"  ({t})")
    for nm in names:
        print(f"   {nm:9s} " + "  ".join(f"{m.run[-13:]:>13s} {res['probe'][m.run][nm][t]['late']:5.1f}/{res['probe'][m.run][nm][t]['visible']:5.1f}" for m in models))
    for nm in feats["_tgt"]:
        print(f"   {nm:9s} 정답 {res['probe']['target'][nm][t]['late']:5.1f}/{res['probe']['target'][nm][t]['visible']:5.1f}")
print("\n[2] attention (첫 예측 query = 마지막 문맥 블록의 물체 자리) — 마지막 블록 / 앞 블록 물체 자리 (보이던 블록) / 앞 블록 나머지")
for m in models:
    for l in LAYERS:
        A_ = res["attn"][m.run][l]
        print(f"   {m.run[-13:]:>13s} L{l:2d}  late " + " ".join(f"{v:.3f}" for v in A_["late"].values()) + "  | visible " + " ".join(f"{v:.3f}" for v in A_["visible"].values()))
print("\n[3] 예측 오차 L1 — 물체 자리 / 배경, 걸음 0 · 3 · 7")
for nm in [m.run for m in models] + ["copy"]:
    E = res["err"][nm]
    print(f"   {nm[-13:]:>13s}  late 물체 " + " ".join(f"{v:.3f}" for v in E["obj"]["late"]) + " 배경 " + " ".join(f"{v:.3f}" for v in E["bg"]["late"])
          + " | visible 물체 " + " ".join(f"{v:.3f}" for v in E["obj"]["visible"]) + " 배경 " + " ".join(f"{v:.3f}" for v in E["bg"]["visible"]))
