#!/usr/bin/env python3
"""VOBench late — 물체 자리 (여유 반경별) 만 채점 (2026-10-18, 사용자 질문). 블록 AR 과 영상 문맥 AR 을 같은 자로.

  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python z_research/scripts/analysis/obj_region_margin.py \
      --run natural_ar_vw_state latest.pt --run natural_ar_vw_state_vidctx e38.pt

쌍 = (pos_a, imp_ab) · (pos_b, imp_ba), 문맥이 같아 예측 하나. 정답 = surprise(imp) > surprise(pos).
토큰 집합: 미래 블록마다 라벨 A · B 물체 중심에서 체비셰프 반경 r 안 (r = 1 → 3×3, 2 → 5×5, 3 → 7×7) · 화면 전체 · 먼 배경 (반경 > 3).
kind=ar · ar_input=block: 입력 · 정답 = 블록별 LN(target_encoder) / ar_input=video: 입력 = target_encoder 문맥 pass, 정답 = 전체 pass (창).
채점 간격은 VOBench 채점과 같다 (30 fps 3 프레임, 32 장 = 16 블록, 문맥 8 블록 → rollout 8 블록)."""
import argparse, csv, json, os, sys, collections
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
import yaml
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs=2, action="append", required=True, metavar=("RUN", "CKPT"))
ap.add_argument("--vobench", default="/workspace/dataset/benchmarks/VOBench_v1")
ap.add_argument("--set", default="late", choices=["late", "gravity"], help="gravity (2026-10-23): block = 문맥 하나 + 미래 여럿, 정답 = 가능 미래가 그 칸 안 surprise 최저. 칸 = 모든 미래 (가능 · 불가능) 의 물체 자리 합집합")
a = ap.parse_args()
dev = "cuda"; S, C, K, G = 256, 8, 8, 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
IDX = list(range(0, 96, 3))[:32]
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
state = _load_checkpoint(CK)
def enc(nf):
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=nf, use_rope=True, uniform_power=False)
    _load_state_dict(e, state["target_encoder"], tag="t"); return e.to(dev).eval()
E2, E32 = enc(2), enc(32)
LN = lambda t: F.layer_norm(t, (t.size(-1),))

def load(d):
    fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in IDX]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 255.0; W0 = x.shape[-1]
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
    return (x - MEAN) / STD, W0

@torch.no_grad()
def feats(x):
    xb = x.reshape(3, 16, 2, 256, 256).permute(1, 0, 2, 3, 4).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hb = LN(torch.cat([E2(xb[i:i + 8]) for i in range(0, 16, 8)], 0).float()).reshape(16 * S, -1)       # 블록별
        xv = x.unsqueeze(0).to(dev)
        idx = torch.arange(C * S, device=dev).unsqueeze(0)
        zc = LN(E32(xv, masks=[idx]).float())[0]                                                             # 문맥 pass
        zf = LN(E32(xv).float())[0]                                                                          # 전체 pass (창)
    return hb, zc, zf

def tokens(label_path, W0, keys=("A", "B")):
    L = json.load(open(label_path))["labels"]; cen = []
    for b in range(C, C + K):
        cs = []
        for key in keys:
            lab = {q["f"]: q for q in L.get(key, [])}
            q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
            if all(t is not None for t in q):
                cs.append((int(np.clip(np.mean([t["v"] for t in q]) / W0 * G, 0, G - 1)), int(np.clip(np.mean([t["u"] for t in q]) / W0 * G, 0, G - 1))))
        cen.append(cs)
    def within(r):
        out = []
        for kb, cs in enumerate(cen):
            s_ = set()
            for (r0, c0) in cs:
                s_ |= {rr * G + cc for rr in range(max(0, r0 - r), min(G, r0 + r + 1)) for cc in range(max(0, c0 - r), min(G, c0 + r + 1))}
            out += [kb * S + t for t in sorted(s_)]
        return out
    near3 = set(within(3))
    far = [t for t in range(K * S) if t not in near3]
    return {"r1": within(1), "r2": within(2), "r3": within(3), "r4": within(4), "full": list(range(K * S)), "far": far}   # r4 = 9×9 (2026-10-20)

class Model:
    def __init__(self, run, ck):
        D = os.path.join(REPO, "z_training/runs", run); c = yaml.safe_load(open(os.path.join(D, "config.yaml")))
        pc = c["model"]["predictor"]; self.video = pc.get("ar_input") == "video"; self.name = f"{run}:{ck}"
        assert pc.get("kind") == "ar", "kind=ar 만"
        kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
                  num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
        for k in ("gating_residual", "gating_anchor", "gating_out_ln"):
            if pc.get(k) is not None: kw[k] = pc[k]
        self.p = vit_prefix_predictor(**kw)
        sd = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D, ck), map_location="cpu", weights_only=False)["predictor"]).items()
              if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
        self.p.load_state_dict(sd, strict=True); self.p = self.p.to(dev).eval()
    @torch.no_grad()
    def predict(self, hb, zc):
        idx = torch.arange(C * S, device=dev).unsqueeze(0)
        ctx = zc if self.video else hb[: C * S]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.p.rollout_kv(ctx.unsqueeze(0), idx, K).float()[0]
    def target(self, hb, zf):
        return (zf if self.video else hb)[C * S:]

models = [Model(r, c) for r, c in a.run]
if a.set == "gravity":   # 2026-10-23 사용자: AR · 2D · 3D 감독의 gravity 물체 자리 채점
    rows = list(csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1_gravity/index.csv"))))
    blk = collections.defaultdict(dict)
    for r in rows: blk[r["block_id"]][r["variant"]] = r
    res = {m.name: collections.defaultdict(list) for m in models}
    root = os.path.join(a.vobench, "gravity")
    for bi, (b, arms) in enumerate(sorted(blk.items(), key=lambda kv: int(kv[0]))):
        any_ = next(iter(arms.values())); bdir = os.path.dirname(os.path.join(root, any_["file_name"]))
        lab = json.load(open(os.path.join(bdir, "label.json")))
        letter = {x[0]: x[2] for x in lab["block"]["arms"]}                  # 팔 이름 → 미래 궤적 글자 (pos → P, imp_float → F …)
        F_ = {}; W0 = None
        for v, r in arms.items():
            x, W0 = load(os.path.join(root, r["file_name"])); F_[v] = feats(x)
        T = tokens(os.path.join(bdir, "label.json"), W0, keys=tuple(sorted(set(letter[v] for v in arms))))
        for m in models:
            hb, zc, zf = F_["pos"]; p = m.predict(hb, zc)                   # 문맥 공유 (f0–47 같은 프레임) → 예측 하나
            err = {v: (p - m.target(F_[v][0], F_[v][2])).abs().mean(-1) for v in arms}
            for key, t in T.items():
                if t:
                    tt = torch.tensor(t, device=dev); s_ = {v: float(e[tt].mean()) for v, e in err.items()}
                    res[m.name][key].append(float(min(s_, key=s_.get) == "pos"))
                    res[m.name][f"{key}_{any_['objset']}_{any_['law']}"].append(float(min(s_, key=s_.get) == "pos"))
        if bi % 50 == 0: print(f"{bi}/{len(blk)}", flush=True)
    print("\nVOBench gravity block 정답률 (%) — 가능 미래가 그 칸 안 surprise 최저. 우연 ledge 50 · arc 11.1 · 전체 37")
    print(f"  {'모델':44s} {'3×3':>6s} {'5×5':>6s} {'7×7':>6s} {'9×9':>6s} {'전체':>6s} {'먼 배경':>7s}")
    for m in models:
        R = res[m.name]; f = lambda k: 100 * np.nanmean(R[k]) if R[k] else float("nan")
        print(f"  {m.name:44s} {f('r1'):6.1f} {f('r2'):6.1f} {f('r3'):6.1f} {f('r4'):6.1f} {f('full'):6.1f} {f('far'):7.1f}   (n {len(R['full'])} block)")
        for key, nm in (("r1", "3×3"), ("full", "전체")):
            print(f"  {'':44s} {nm}: " + " · ".join(f"{o} {law} {f(f'{key}_{o}_{law}'):.1f}" for o in ("primitive", "realistic") for law in ("ledge", "arc")))
    sys.exit(0)
rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == "late"]
blk = collections.defaultdict(dict)
for r in rows: blk[r["block_id"]][r["variant"]] = r
res = {m.name: collections.defaultdict(list) for m in models}
copy = collections.defaultdict(lambda: collections.defaultdict(list))
for bi, (b, arms) in enumerate(sorted(blk.items())):
    F_ = {}; W0 = None
    for v in ("pos_a", "imp_ab", "pos_b", "imp_ba"):
        x, W0 = load(os.path.join(a.vobench, arms[v]["file_name"])); F_[v] = feats(x)
    T = tokens(os.path.join(os.path.dirname(os.path.join(a.vobench, arms["pos_a"]["file_name"])), "label.json"), W0)
    for m in models:
        for pv, iv in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
            hb, zc, zf = F_[pv]; p = m.predict(hb, zc)
            ep = (p - m.target(hb, zf)).abs().mean(-1); ei = (p - m.target(*F_[iv][:1], F_[iv][2])).abs().mean(-1)
            for key, t in T.items():
                if t:
                    tt = torch.tensor(t, device=dev); res[m.name][key].append(float(ei[tt].mean() > ep[tt].mean()))
            res[m.name]["viol_" + arms[pv]["violation"]].append(float(ei[torch.tensor(T["r4"], device=dev)].mean() > ep[torch.tensor(T["r4"], device=dev)].mean()) if T["r4"] else np.nan)
            res[m.name][f"o_{arms[pv]['objset']}_{arms[pv]['violation']}"].append(float(ei[torch.tensor(T["r4"], device=dev)].mean() > ep[torch.tensor(T["r4"], device=dev)].mean()) if T["r4"] else np.nan)
    if bi % 50 == 0: print(f"{bi}/{len(blk)}", flush=True)

print("\nVOBench late 쌍 정확도 (%) — 물체 자리 반경별 · 화면 전체 · 먼 배경 (반경 > 3). 우연 50")
print(f"  {'모델':44s} {'3×3':>6s} {'5×5':>6s} {'7×7':>6s} {'9×9':>6s} {'전체':>6s} {'먼 배경':>7s} | 9×9 위반별 (영속 / 모양 / 색)")
for m in models:
    R = res[m.name]; f = lambda k: 100 * np.nanmean(R[k]) if R[k] else float("nan")
    print(f"  {m.name:44s} {f('r1'):6.1f} {f('r2'):6.1f} {f('r3'):6.1f} {f('r4'):6.1f} {f('full'):6.1f} {f('far'):7.1f} | "
          f"{f('viol_permanence'):.1f} / {f('viol_shape'):.1f} / {f('viol_color'):.1f}   (n {len(R['full'])} 쌍)")
    print(f"  {'':44s} 9×9 primitive 영속 / 모양 / 색 " + " / ".join(f"{f('o_primitive_' + v):.1f}" for v in ("permanence", "shape", "color"))
          + "   realistic " + " / ".join(f"{f('o_realistic_' + v):.1f}" for v in ("permanence", "shape", "color")))
