#!/usr/bin/env python3
"""VOBench 채점을 물체 자리 토큰만으로 다시 (2026-10-17, 사용자 질문) — AR vs ctxt_video.

  python z_research/scripts/analysis/obj_region_surprise.py --occ late --device cuda

채점과 같은 구성: 30 fps PNG 3 장 간격 → 32 장 = 16 블록, 문맥 8 블록 → rollout 8 블록, 타깃 = 블록별 LN(target_encoder), L1.
쌍 = (pos_a, imp_ab) · (pos_b, imp_ba) — 쌍 안 문맥이 같다 (문맥 예측 한 번). 정답 = surprise(imp) > surprise(pos).
surprise: (full) 미래 8 블록 전체 토큰 평균 — 기존 채점 재현용 / (obj) 미래 블록마다 라벨 A · B 물체 중심 3×3 합집합 토큰만 /
  (far) 물체 중심에서 체비셰프 거리 > 3 토큰 (먼 배경, 가림막은 섞일 수 있음).
출력: <out>/obj_region_surprise_<occ>.json + 표 (objset · 위반별, 방향 A / B)."""
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
ap.add_argument("--occ", default="late")
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default=os.path.join(REPO, "z_research/VOBench_v1/figures/brief_diag"))
a = ap.parse_args(); dev = a.device
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
S, C, K, G = 256, 8, 8, 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
IDX = list(range(0, 96, 3))[:32]
state = _load_checkpoint(CK)
tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tenc, state["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()

def frames(d):
    fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in IDX]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 255.0
    W0 = x.shape[-1]
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
    return (x - MEAN) / STD, W0

def far_tokens(label_path, W0, rad=3):
    """미래 블록마다 라벨 A · B 물체 중심에서 체비셰프 거리 > rad 인 토큰 (먼 배경). 가림막 위치 라벨은 없어 가림막은 섞일 수 있다."""
    L = json.load(open(label_path))["labels"]; out = []
    for b in range(C, C + K):
        near = set()
        for key in ("A", "B"):
            lab = {q["f"]: q for q in L.get(key, [])}
            q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
            if any(t is None for t in q):
                continue
            r0 = int(np.clip(np.mean([t["v"] for t in q]) / W0 * G, 0, G - 1)); c0 = int(np.clip(np.mean([t["u"] for t in q]) / W0 * G, 0, G - 1))
            near |= {rr * G + cc for rr in range(max(0, r0 - rad), min(G, r0 + rad + 1)) for cc in range(max(0, c0 - rad), min(G, c0 + rad + 1))}
        out.append(sorted(set(range(S)) - near))
    return out


def obj_tokens(label_path, W0):
    """미래 블록 C..C+K-1 마다 라벨 A · B 물체 중심 3×3 합집합 (블록 안 토큰 인덱스 목록)."""
    L = json.load(open(label_path))["labels"]; out = []
    for b in range(C, C + K):
        s = set()
        for key in ("A", "B"):
            lab = {q["f"]: q for q in L.get(key, [])}
            q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
            if any(t is None for t in q):
                continue
            r0 = int(np.clip(np.mean([t["v"] for t in q]) / W0 * G, 0, G - 1)); c0 = int(np.clip(np.mean([t["u"] for t in q]) / W0 * G, 0, G - 1))
            s |= {rr * G + cc for rr in range(max(0, r0 - 1), min(G, r0 + 2)) for cc in range(max(0, c0 - 1), min(G, c0 + 2))}
        out.append(sorted(s))
    return out

@torch.no_grad()
def blocks(x):
    xb = x.reshape(3, 16, 2, 256, 256).permute(1, 0, 2, 3, 4).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.cat([tenc(xb[i:i + 8]) for i in range(0, 16, 8)], 0).float()
    return F.layer_norm(h, (1280,)).reshape(16 * S, -1)

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
        if self.kind != "ar":
            key = cfg["model"].get("context_encoder_key", "encoder")
            self.enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2 * C, use_rope=True, uniform_power=False)
            _load_state_dict(self.enc, state[key], tag=key); self.enc = self.enc.to(dev).eval()

    @torch.no_grad()
    def predict(self, x, h):
        idx = torch.arange(C * S, device=dev).unsqueeze(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.kind == "ar":
                p = self.pred.rollout_kv(h[: C * S].unsqueeze(0), idx, K)
            else:
                z = self.enc(x[:, : 2 * C].unsqueeze(0).to(dev)).float()
                p = self.pred.render(self.pred.rollout_state_kv(z, idx, K))
        return p.float()[0]

rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == a.occ]
blk = collections.defaultdict(dict)
for r in rows:
    blk[r["block_id"]][r["variant"]] = r
models = [Model(r) for r in a.runs]
res = {m.name: [] for m in models}
for bi, (b, arms) in enumerate(sorted(blk.items())):
    d0 = os.path.join(a.vobench, arms["pos_a"]["file_name"])
    X, H = {}, {}
    for v in ("pos_a", "pos_b", "imp_ab", "imp_ba"):
        X[v], W0 = frames(os.path.join(a.vobench, arms[v]["file_name"])); H[v] = blocks(X[v])
    ot = obj_tokens(os.path.join(os.path.dirname(d0), "label.json"), W0)
    oidx = torch.tensor([s_ * S + t for s_, toks in enumerate(ot) for t in toks], device=dev)
    ft = far_tokens(os.path.join(os.path.dirname(d0), "label.json"), W0)
    fidx = torch.tensor([s_ * S + t for s_, toks in enumerate(ft) for t in toks], device=dev)
    for m in models:
        for pos, imp, dirn in (("pos_a", "imp_ab", "A"), ("pos_b", "imp_ba", "B")):
            p = m.predict(X[pos], H[pos])
            sc = {}
            for v in (pos, imp):
                e = (p - H[v][C * S:]).abs().mean(-1)               # (K*S,) 토큰별 L1
                sc[v] = (float(e.mean()), float(e[oidx].mean()) if len(oidx) else float("nan"), float(e[fidx].mean()) if len(fidx) else float("nan"))
            res[m.name].append({"block": b, "dir": dirn, "objset": arms[pos]["objset"], "viol": arms[pos]["violation"],
                                "full": sc[imp][0] > sc[pos][0], "obj": sc[imp][1] > sc[pos][1], "far": sc[imp][2] > sc[pos][2],
                                "motion": arms[pos]["motion"], "k": arms[pos].get("k"),
                                "margin_full": sc[imp][0] - sc[pos][0], "margin_obj": sc[imp][1] - sc[pos][1]})
    if bi % 40 == 0:
        print(f"{bi}/{len(blk)}", flush=True)

os.makedirs(a.out, exist_ok=True)
json.dump(res, open(f"{a.out}/obj_region_surprise_{a.occ}.json", "w"), indent=1)
print(f"\nVOBench {a.occ} — 쌍 정확도 (%): 화면 전체 surprise → 물체 자리만 surprise  [방향 A / B]")
for m in models:
    R = res[m.name]
    for ob in ("primitive", "realistic"):
        for vi in ("permanence", "shape", "color", None):
            sel = [x for x in R if x["objset"] == ob and (vi is None or x["viol"] == vi)]
            f = 100 * np.mean([x["full"] for x in sel]); o = 100 * np.mean([x["obj"] for x in sel])
            fa = 100 * np.mean([x["full"] for x in sel if x["dir"] == "A"]); fb = 100 * np.mean([x["full"] for x in sel if x["dir"] == "B"])
            oa = 100 * np.mean([x["obj"] for x in sel if x["dir"] == "A"]); ob_ = 100 * np.mean([x["obj"] for x in sel if x["dir"] == "B"])
            mf = np.mean([x["margin_full"] for x in sel]); mo = np.mean([x["margin_obj"] for x in sel]); fr = 100 * np.mean([x["far"] for x in sel])
            print(f"  {m.name[-22:]:>22s} {ob:9s} {str(vi or '평균'):10s} n{len(sel):3d}  전체 {f:5.1f} [{fa:3.0f}/{fb:3.0f}] → 물체 {o:5.1f} [{oa:3.0f}/{ob_:3.0f}] · 먼 배경 {fr:5.1f}  | margin {mf:+.4f} → {mo:+.4f}")
