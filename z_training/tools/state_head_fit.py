#!/usr/bin/env python3
"""상태 head 만 따로 학습해서 자로 쓸 만한지 본다 (2026-10-19, 사용자 지시: "학습 멈추고 head 만 따로 학습해서 성능 나오는지").

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_training/tools/state_head_fit.py \
      --run natural_ar_vw_state_phys2d --ckpt latest.pt --head '{"type": "light"}' --tag light_plain [--steps 3000]

predictor (체크포인트) 와 encoder 는 고정, head 만 새로 초기화해 Adam (lr 1e-3, clip 없음) 으로 학습한다 —
main 의 IdReadout 학습 조건 (Adam 1e-3, 따로 학습) 에 맞춘다. 학습 = phys2d_sup_v2 train 경계 창, 평가 = val 경계 창.
손실 · 라벨 · 자리 (TF 블록 1.. · rollout C..) 는 학습 때와 같다 (state_head.ar_state_loss). --w 로 손실 가중치
(예: '{"exist_neg": 0.1}'). 평가 표는 state_head_probe.py 와 같은 자리 (tf_ctx · tf_pred · rollout · copy).
산출: z_training/runs/<run>/head_fit/<tag>.json (+ 같은 이름 .pt)."""
import argparse, bisect, json, os, random, sys, time
import numpy as np, torch, yaml
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
ap.add_argument("--head", default='{"type": "light"}', help="state_head config (json)")
ap.add_argument("--w", default="{}", help="손실 가중치 json (exist · pos · vel · exist_neg)")
ap.add_argument("--tag", required=True)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--ctx", type=int, nargs="+", default=[4, 8, 16])
ap.add_argument("--eval_every", type=int, default=1000)
ap.add_argument("--n_eval", type=int, default=192)
ap.add_argument("--root", default="/workspace/dataset/training_data/phys2d_sup_v2")
a = ap.parse_args()
dev, S, TUB, K = "cuda", 256, 2, 16
torch.manual_seed(0)
D = os.path.join(REPO, "z_training/runs", a.run)
cfg = yaml.safe_load(open(os.path.join(D, "config.yaml")))
ck = torch.load(os.path.join(D, a.ckpt), map_location="cpu", weights_only=False)
pc = cfg["model"]["predictor"]
pred = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12,
                            num_heads=12, num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
pred.load_state_dict(_clean_backbone_key(ck["predictor"]), strict=True)
pred = pred.to(dev).eval().requires_grad_(False)
CKE = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc, _load_checkpoint(CKE)["target_encoder"], tag="t"); enc = enc.to(dev).eval().requires_grad_(False)
HC = json.loads(a.head); W = json.loads(a.w)
head = SHM.build_head(HC, 1280).to(dev)
opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.0)
n_par = sum(p.numel() for p in head.parameters())


def source(split, ctxs):
    ds = VarWindowBatches.__new__(VarWindowBatches)
    ds.tubelet = TUB; ds.transform = type("T", (), {"resolution": 256})()
    ds._init_boundary(dict(name=split, p=1.0, manifest=f"{a.root}/{split}/boundary.tsv", stride=1, context_frames=ctxs, predict_frames=K,
                           crop_ratio=[1, 1], hflip_p=0.0, labels="phys2d_sup"))
    return ds


MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)


def batch(ds, items, c, rng):
    xs, labs = [], []
    for _, path, b in items:
        x, lab = ds._read_boundary(path, b, c, rng)
        xs.append(x); labs.append(lab)
    x = (torch.stack(xs).to(dev).float().div_(255.0) - MEAN) / STD
    return x, {k: torch.stack([l[k] for l in labs]).to(dev) for k in labs[0]}


@torch.no_grad()
def features(x, c):
    C = c // TUB; nb = (c + K) // TUB
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = AR.encode_blocks(enc, x, TUB)
        p_tf, p_ar = pred(h, ar={"idx": AR.block_indices(h.size(0), nb, S, dev), "n_ctx_blocks": C, "rollout_steps": nb - C})
    return p_tf.float(), p_ar.float(), C, nb - C


tr = source("train", a.ctx); va = source("val", [8])
rng = random.Random(0)
lo_va = bisect.bisect_left(va.bnd_cmax, 8)
eval_items = [va.bnd_items[rng.randrange(lo_va, len(va.bnd_items))] for _ in range(a.n_eval)]


@torch.no_grad()
def evaluate():
    head.eval()
    acc = {k: {"vis": [], "hid": [], "rec": [], "neg": [], "id": [], "conc": []} for k in ("tf_ctx", "tf_pred", "rollout", "copy")}

    def add(key, out, lab):
        err = (out["pos"].float() - lab["pos"]).norm(dim=-1) * 256
        mh = lab["hidden"]; mv = lab["m_state"] & ~mh
        acc[key]["vis"].append(err[mv].cpu()); acc[key]["hid"].append(err[mh].cpu())
        on = out["exist"] > 0
        acc[key]["rec"].append(on[lab["seen"]].float().cpu()); acc[key]["neg"].append((~on)[~lab["seen"]].float().cpu())
        if "obj" in out:     # 정체 정확도: 보인 정체 중 49 지선다 최고가 자기 정체
            ids = torch.arange(out["obj"].size(-2), device=out["obj"].device).expand_as(lab["seen"])
            acc[key]["id"].append((out["obj"].argmax(-1) == ids)[lab["seen"]].float().cpu())
        if "attn" in out:    # 집중도 w_max × 256 (균등 = 1)
            acc[key]["conc"].append((out["attn"].amax(-1) * out["attn"].size(-1))[lab["seen"]].float().cpu())
    erng = random.Random(1)
    for i in range(0, len(eval_items), a.bs):
        x, lab = batch(va, eval_items[i:i + a.bs], 8, erng)
        p_tf, p_ar, C, R = features(x, 8); nb = C + R
        o_tf, o_ar = head(p_tf, S), head(p_ar, S)
        sl = lambda a_, b_: SHM.slice_blocks(lab, a_, b_)
        cut = lambda o, a_, b_: {k: v[:, a_:b_] for k, v in o.items()}
        add("tf_ctx", cut(o_tf, 0, C - 1), sl(1, C)); add("tf_pred", cut(o_tf, C - 1, nb - 1), sl(C, nb)); add("rollout", o_ar, sl(C, nb))
        add("copy", {"pos": lab["pos"][:, C - 1:C].expand(-1, R, -1, -1), "exist": torch.where(lab["seen"][:, C - 1:C], 1.0, -1.0).expand(-1, R, -1)}, sl(C, nb))
    head.train()
    res = {}
    for k, v in acc.items():
        vis, hid = torch.cat(v["vis"]), torch.cat(v["hid"]); rec, neg = torch.cat(v["rec"]).mean().item(), torch.cat(v["neg"]).mean().item()
        res[k] = {"vis_mean": vis.mean().item(), "vis_med": vis.median().item(), "hid_mean": hid.mean().item() if hid.numel() else None,
                  "hid_med": hid.median().item() if hid.numel() else None, "rec": rec, "neg": neg, "bacc": (rec + neg) / 2,
                  "id_acc": torch.cat(v["id"]).mean().item() if v["id"] else None, "conc": torch.cat(v["conc"]).mean().item() if v["conc"] else None}
    return res


def show(step, res):
    print(f"[step {step}] " + " | ".join(f"{k} 보임 {r['vis_mean']:.1f}/{r['vis_med']:.1f} 가림 {r['hid_mean'] or 0:.1f}/{r['hid_med'] or 0:.1f} "
                                          f"rec {100 * r['rec']:.0f}% bacc {100 * r['bacc']:.0f}"
                                          + (f" 정체 {100 * r['id_acc']:.0f}%" if r.get("id_acc") is not None else "")
                                          + (f" 집중 {r['conc']:.1f}" if r.get("conc") is not None else "") for k, r in res.items()), flush=True)


print(f"{a.run}:{a.ckpt} · head {HC} ({n_par:,} 파라미터) · w {W} · Adam lr {a.lr} · {a.steps} step × {a.bs} 창", flush=True)
hist, t0 = [], time.time()
for step in range(1, a.steps + 1):
    c = rng.choice(a.ctx); lo = bisect.bisect_left(tr.bnd_cmax, c)
    items = [tr.bnd_items[rng.randrange(lo, len(tr.bnd_items))] for _ in range(a.bs)]
    x, lab = batch(tr, items, c, rng)
    p_tf, p_ar, C, R = features(x, c)
    loss, log = SHM.ar_state_loss(head, p_tf, p_ar, lab, S, C, R, W, {})
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.3f} rollout 위치 {log['ar_pos_px_vis']:.1f}px"
              + (f" 정체 {100 * log['ar_id_acc']:.0f}%" if "ar_id_acc" in log else "") + (f" 집중 {log['ar_conc']:.1f}" if "ar_conc" in log else "")
              + (f" τ {head.log_tau.exp().item():.2f}" if hasattr(head, "log_tau") else "") + f" ({time.time() - t0:.0f}s)", flush=True)
    if step % a.eval_every == 0 or step == a.steps:
        r = evaluate(); show(step, r); hist.append({"step": step, **r})
out = os.path.join(D, "head_fit"); os.makedirs(out, exist_ok=True)
json.dump({"run": a.run, "ckpt": a.ckpt, "head": HC, "w": W, "params": n_par, "lr": a.lr, "steps": a.steps, "bs": a.bs, "hist": hist},
          open(os.path.join(out, f"{a.tag}.json"), "w"), indent=1)
torch.save(head.state_dict(), os.path.join(out, f"{a.tag}.pt"))
print("저장", os.path.join(out, f"{a.tag}.json"))
