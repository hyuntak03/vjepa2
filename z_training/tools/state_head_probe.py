#!/usr/bin/env python3
"""상태 head 위치 오차를 자리별로 나눠 잰다 (2026-10-19, 사용자 질문: "context 쪽이랑 prediction 쪽 나눠서").

  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python z_training/tools/state_head_probe.py \
      --run natural_ar_vw_state_phys2d_light --ckpt latest.pt [--n 192] [--ctx 8]

phys2d_sup_v2 **val** 경계 행 (예측 첫 장 = 사건), 문맥 C 장 · 예측 16 장, 반전 없음. 체크포인트의 predictor 와 state_head 를 그대로 쓴다.
자리 (블록 단위, 라벨 = 그 블록 마지막 장):
  tf_ctx   TF 출력이 문맥 안 블록 1..C-1 을 예측 (진짜 과거를 보고 한 블록 앞)
  tf_pred  TF 출력이 예측 구간 블록 C..  을 예측 (진짜 과거 — 예측 구간 정답까지 보고 한 블록 앞)
  rollout  상태 rollout 이 블록 C..C+R-1 을 예측 (문맥만 보고)
  h_ctx / h_pred  같은 head 를 **정답 토큰** (LN target_encoder 블록) 에 바로 건 것 = 이 자로 읽을 수 있는 최선에 가깝다
  copy     라벨만으로: 마지막 문맥 블록 위치를 그대로 (예측 구간)
위치 오차 px (보임 / 가림, 평균 · 중앙) · 존재 recall (있는 정체를 켰나) · 없음 정확도 · bacc."""
import argparse, bisect, os, random, sys
import numpy as np, torch, torch.nn.functional as F, yaml
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                                                  # noqa: E402
from app.vjepa_frozen import state_head as SHM                                                                 # noqa: E402
from app.vjepa_frozen import ar as AR                                                                          # noqa: E402
from app.vjepa_frozen.var_window import VarWindowBatches                                                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ckpt", default="latest.pt")
ap.add_argument("--n", type=int, default=192, help="창 수")
ap.add_argument("--ctx", type=int, default=8, help="문맥 장 수 (tubelet 배수)")
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--manifest", default="/workspace/dataset/training_data/phys2d_sup_v2/val/boundary.tsv")
a = ap.parse_args()
dev, S, TUB, K = "cuda", 256, 2, 16
D = os.path.join(REPO, "z_training/runs", a.run)
cfg = yaml.safe_load(open(os.path.join(D, "config.yaml")))
ck = torch.load(os.path.join(D, a.ckpt), map_location="cpu", weights_only=False)

pc = cfg["model"]["predictor"]
kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
          num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
pred = vit_prefix_predictor(**kw)
pred.load_state_dict({k: v for k, v in _clean_backbone_key(ck["predictor"]).items()}, strict=True)
pred = pred.to(dev).eval()
head = SHM.build_head(ck.get("state_head_cfg") or cfg["state_head"], 1280)
head.load_state_dict(ck["state_head"]); head = head.to(dev).eval()
CKE = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc, _load_checkpoint(CKE)["target_encoder"], tag="t"); enc = enc.to(dev).eval()

ds = VarWindowBatches.__new__(VarWindowBatches)
ds.tubelet = TUB; ds.transform = type("T", (), {"resolution": 256})()
ds._init_boundary(dict(name="val", p=1.0, manifest=a.manifest, stride=1, context_frames=[a.ctx], predict_frames=K,
                       crop_ratio=[1, 1], hflip_p=0.0, labels="phys2d_sup"))
rng = random.Random(0)
lo = bisect.bisect_left(ds.bnd_cmax, a.ctx)
items = [ds.bnd_items[rng.randrange(lo, len(ds.bnd_items))] for _ in range(a.n)]
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
C = a.ctx // TUB; nb = (a.ctx + K) // TUB; R = nb - C

acc = {k: {"vis": [], "hid": [], "rec": [], "neg": []} for k in ("tf_ctx", "tf_pred", "rollout", "h_ctx", "h_pred", "copy")}


def add(key, out, lab):
    """out: head 출력 (B,T,48,…) cpu · lab: 같은 블록 라벨"""
    err = (out["pos"].float() - lab["pos"]).norm(dim=-1) * 256
    ms, mh = lab["m_state"], lab["hidden"]
    mv = ms & ~mh
    acc[key]["vis"].append(err[mv]); acc[key]["hid"].append(err[mh])
    on = out["exist"] > 0
    acc[key]["rec"].append(on[lab["seen"]].float()); acc[key]["neg"].append((~on)[~lab["seen"]].float())


with torch.no_grad():
    for i in range(0, len(items), a.bs):
        xs, labs = [], []
        for _, path, b in items[i:i + a.bs]:
            x, lab = ds._read_boundary(path, b, a.ctx, rng)
            xs.append(x); labs.append(lab)
        x = torch.stack(xs).to(dev).float().div_(255.0)
        x = (x - MEAN) / STD
        lab = {k: torch.stack([l[k] for l in labs]) for k in labs[0]}           # (B, nb, 48, …)
        sl = lambda a_, b_: SHM.slice_blocks(lab, a_, b_)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = AR.encode_blocks(enc, x, TUB)                                    # (B, nb*S, D)
            idx = AR.block_indices(h.size(0), nb, S, dev)
            p_tf, p_ar = pred(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": R})
            o_tf, o_ar, o_h = head(p_tf.float(), S), head(p_ar.float(), S), head(h.float(), S)
        o_tf = {k: v.float().cpu() for k, v in o_tf.items()}; o_ar = {k: v.float().cpu() for k, v in o_ar.items()}
        o_h = {k: v.float().cpu() for k, v in o_h.items()}
        cut = lambda o, a_, b_: {k: v[:, a_:b_] for k, v in o.items()}
        add("tf_ctx", cut(o_tf, 0, C - 1), sl(1, C))                             # TF 위치 0..C-2 → 블록 1..C-1
        add("tf_pred", cut(o_tf, C - 1, nb - 1), sl(C, nb))                      # TF 위치 C-1..nb-2 → 블록 C..nb-1
        add("rollout", o_ar, sl(C, C + R))
        add("h_ctx", cut(o_h, 1, C), sl(1, C))
        add("h_pred", cut(o_h, C, nb), sl(C, nb))
        cp = {"pos": lab["pos"][:, C - 1:C].expand(-1, R, -1, -1), "exist": torch.where(lab["seen"][:, C - 1:C], 1.0, -1.0).expand(-1, R, -1)}
        add("copy", cp, sl(C, nb))
        print(f"{min(i + a.bs, len(items))}/{len(items)}", end="\r", flush=True)

print(f"\n{a.run}:{a.ckpt} · phys2d_sup_v2 val {len(items)} 창 · 문맥 {a.ctx} 장 ({C} 블록) · 예측 {K} 장 ({R} 블록) · head {(ck.get('state_head_cfg') or {}).get('type', 'attn')}")
print(f"  {'자리':10s} {'보임 평균/중앙 px':>18s} {'가림 평균/중앙 px':>18s} {'있음 recall':>11s} {'없음 정확':>9s} {'bacc':>6s}")
for k, v in acc.items():
    vis, hid = torch.cat(v["vis"]), torch.cat(v["hid"])
    rec, neg = torch.cat(v["rec"]).mean().item(), torch.cat(v["neg"]).mean().item()
    f = lambda t: f"{t.mean():7.1f} /{t.median():6.1f}" if t.numel() else "      —       "
    print(f"  {k:10s} {f(vis):>18s} {f(hid):>18s} {100 * rec:10.1f}% {100 * neg:8.1f}% {50 * (rec + neg):5.1f}")
