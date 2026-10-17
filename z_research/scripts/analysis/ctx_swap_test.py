#!/usr/bin/env python3
"""문맥 바꿔치기 시험 (--keep_last k: 마지막 k 블록은 진짜, 앞쪽만 바꿈 — 앞쪽 문맥을 쓰나) — predictor 가 문맥을 실제로 보고 미래를 만드나 (2026-10-17, 사용자 질문).

  python z_research/scripts/analysis/ctx_swap_test.py --runs natural_ar_vw_state natural_ar_vw_state_ctxt_video --n 60 --device cuda

같은 clip 의 진짜 문맥으로 굴린 예측 p 와, **다른 clip 의 문맥**으로 굴린 예측 p_swap 을 비교한다 (rollout 8 블록, 채점과 같은 프레임).
  변화 = |p_swap − p| / |p − 정답|  (걸음별, 토큰 평균 L1).  0 에 가까우면 문맥을 안 쓴다 (어떤 문맥이든 같은 미래)
  같이 잼: 정답 대비 오차 — 진짜 문맥 |p − 정답| vs 바꾼 문맥 |p_swap − 정답| (바꾼 문맥이면 오차가 커져야 정상)
  기준선: 블록 복사 (마지막 문맥 블록 LN(h) 를 미래 블록마다) 의 오차
정답 = 블록별 LN(target_encoder) (두 run 모두 타깃이 블록 단위). VOBench pos_a, 이동 clip, visible · late.
⚠️ 영상 문맥판 (ctxt_video) 은 문맥 = encoder 영상 한 번, AR 은 블록별 target_encoder. 파라미터 없음."""
import argparse, glob, json, os, sys
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
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--device", default="cuda")
ap.add_argument("--keep_last", type=int, default=0, help="문맥 마지막 몇 블록은 진짜로 두고 앞쪽만 바꾼다 (0 = 문맥 전부 바꿈)")
ap.add_argument("--out", default=os.path.join(REPO, "z_research/VOBench_v1/figures/ctx_attention"))
a = ap.parse_args(); dev = a.device; os.makedirs(a.out, exist_ok=True)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
S, C, K = 256, 8, 8
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
IDX = list(range(0, 96, 3))[:32]
state = _load_checkpoint(CK)
tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tenc, state["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()

def load_clip(d):
    fr = [np.asarray(Image.open(f"{d}/pos_a/{i:06d}.png").convert("RGB")) for i in IDX]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 255.0
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
    return (x - MEAN) / STD

@torch.no_grad()
def blocks(x):     # (3, 32, H, W) -> (1, 16*S, D) 블록별 LN(target_encoder)
    xb = x.reshape(3, 16, 2, 256, 256).permute(1, 0, 2, 3, 4).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.cat([tenc(xb[i:i + 8]) for i in range(0, 16, 8)], 0).float()
    return F.layer_norm(h, (1280,)).reshape(1, 16 * S, -1)

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
        self.pred.load_state_dict(sd, strict=True); self.pred = self.pred.to(dev).eval()
        if self.kind != "ar":
            key = cfg["model"].get("context_encoder_key", "encoder")
            self.enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2 * C, use_rope=True, uniform_power=False)
            _load_state_dict(self.enc, state[key], tag=key); self.enc = self.enc.to(dev).eval()

    @torch.no_grad()
    def context(self, x, hb):
        if self.kind == "ar":
            return hb[:, : C * S]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.enc(x[:, : 2 * C].unsqueeze(0).to(dev)).float()

    @torch.no_grad()
    def rollout(self, ctx):
        idx = torch.arange(C * S, device=dev).unsqueeze(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.kind == "ar":
                p = self.pred.rollout_kv(ctx, idx, K)
            else:
                p = self.pred.render(self.pred.rollout_state_kv(ctx, idx, K))
        return p.float()[0]

def l1(a_, b_):     # 걸음별 (K,)
    return (a_ - b_).abs().mean(-1).reshape(K, S).mean(1).cpu().numpy()

models = [Model(r) for r in a.runs]
MOV = ("flat_vel", "flat_acc", "flat_dec", "ramp_down")
res = {}
for occ in ("visible", "late"):
    pat = f"{a.vobench}/*/*/{occ}/*" if occ == "visible" else f"{a.vobench}/*/*/{occ}_k*/*"
    dirs = sorted(d for d in glob.glob(pat) if any(m in os.path.basename(d) for m in MOV))
    dirs = list(np.random.default_rng(1).permutation(dirs))[: a.n]
    xs = [load_clip(d) for d in dirs]; hbs = [blocks(x) for x in xs]
    R = {m.run: {"err_true": [], "err_swap": [], "change": []} for m in models}; copy = []
    for i in range(len(dirs)):
        j = (i + len(dirs) // 2) % len(dirs)                                      # 다른 clip (다른 장면 · 물체)
        tgt = hbs[i][0, C * S:]
        copy.append(l1(hbs[i][0, (C - 1) * S: C * S].repeat(K, 1), tgt))
        for m in models:
            p = m.rollout(m.context(xs[i], hbs[i]))
            if a.keep_last:   # 앞쪽 C-keep 블록만 다른 clip 것으로 (프레임 단위로 섞어 영상 문맥도 다시 encoding)
                kf = 2 * (C - a.keep_last)
                xm = xs[i].clone(); xm[:, :kf] = xs[j][:, :kf]
                hm = hbs[i].clone(); hm[:, : (C - a.keep_last) * S] = hbs[j][:, : (C - a.keep_last) * S]
                ps = m.rollout(m.context(xm, hm))
            else:
                ps = m.rollout(m.context(xs[j], hbs[j]))
            e = l1(p, tgt); R[m.run]["err_true"].append(e); R[m.run]["err_swap"].append(l1(ps, tgt))
            R[m.run]["change"].append(l1(ps, p) / np.maximum(e, 1e-9))
    res[occ] = {"copy_err": np.mean(copy, 0).tolist(),
                **{r: {k: np.mean(v, 0).tolist() for k, v in d.items()} for r, d in R.items()}}
    print(f"\n=== {occ} (n={len(dirs)}) — 걸음 0 … 7 ===")
    print(f"{'블록 복사 오차':30s} " + " ".join(f"{x:.3f}" for x in res[occ]["copy_err"]))
    for r in R:
        d = res[occ][r]
        print(f"{r:30s} 진짜 문맥 오차 " + " ".join(f"{x:.3f}" for x in d["err_true"]))
        print(f"{'':30s} 바꾼 문맥 오차 " + " ".join(f"{x:.3f}" for x in d["err_swap"]))
        print(f"{'':30s} 예측 변화 / 오차 " + " ".join(f"{x:.2f}" for x in d["change"]))
json.dump(res, open(f"{a.out}/ctx_swap_test{'_keep' + str(a.keep_last) if a.keep_last else ''}.json", "w"), indent=1)
