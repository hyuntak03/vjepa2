"""같은 움직임 자로 여러 predictor 를 잰다 (2026-10-24, 사용자 "rollout 이 42 % 가는 거면 이전이랑 비슷한 거 아니야?").

자 = 한 run (기본 motion v3) 의 차분 encoder · 차분 예측기 (학습 중 체크포인트, eval). 학습 로그의 비율과 같은 정의:
  d 분산 대비 오차 — 정답 쌍 (h_b, h_{b+1}) · TF (h_b, p̂_{b+1}) · 굴린 예측 (p̂_b, p̂_{b+1}) · 멈춤 (h_b, h_b)
  메운 몫 = (멈춤 − 예측) / (멈춤 − 정답 쌍) — 0 % 멈춘 세계, 100 % 정답 쌍만큼
⚠ v3 는 이 자를 낮추도록 학습됐다 (자기 자에 유리). 기준 AR · v1 은 이 자를 본 적이 없다 → 기준 AR 의 값이 "이 자로 본 원래 움직임 양".
데이터 = 학습 표 (var_window) 의 보류 아닌 무작위 클립, W 장 = 문맥 C 블록 + 예측 W/2 − C 블록. GPU 메모리 --mem_gb 로 묶는다.

  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/motion_gap_compare.py \
      --models natural_ar_vw_state@e4 natural_ar_vw_state@latest natural_ar_vw_motion@e4 natural_ar_vw_motion_v2@e4 natural_ar_vw_motion_v3@latest
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, ".")
from analysis.intphys2.model import _clean_backbone_key                    # noqa: E402
from app.vjepa_frozen import ar as AR, motion as MOM, utils as U            # noqa: E402
from app.vjepa_frozen.data import ClipTransform, normalize_clips            # noqa: E402
from app.vjepa_frozen.var_window import VarWindowBatches                    # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor              # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--reader", default="natural_ar_vw_motion_v3@latest")
p.add_argument("--models", nargs="+", required=True)
p.add_argument("--clips", type=int, default=320); p.add_argument("--W", type=int, default=16); p.add_argument("--C", type=int, default=4)
p.add_argument("--bs", type=int, default=4); p.add_argument("--mem_gb", type=float, default=4.5)
p.add_argument("--out", default="")
a = p.parse_args()
dev = torch.device("cuda:0")
torch.cuda.set_per_process_memory_fraction(a.mem_gb * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)
TUB, S = 2, 256
NB = a.W // TUB; C = a.C; R = NB - C


def ck_path(spec):
    run, ck = spec.split("@")
    return run, f"z_training/runs/{run}/{ck}.pt"


def load_pred(spec):
    run, path = ck_path(spec)
    pc = yaml.safe_load(open(f"z_training/runs/{run}/config.yaml"))["model"]["predictor"]
    kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
              num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
    for k_ in ("gating_residual", "gating_anchor", "gating_out_ln"):
        if pc.get(k_) is not None:
            kw[k_] = pc[k_]
    m = vit_prefix_predictor(**kw)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    w = {k: v for k, v in _clean_backbone_key(ck["predictor"]).items() if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
    m.load_state_dict(w, strict=True)
    return m.to(dev).eval(), ck.get("epoch")


rrun, rpath = ck_path(a.reader)
rck = torch.load(rpath, map_location="cpu", weights_only=False)
cfg = yaml.safe_load(open(f"z_training/runs/{rrun}/config.yaml"))
MO = rck["motion_cfg"]
mod = MOM.Motion(MO, 1280, 256, 16, TUB).to(dev); mod.load_state_dict(rck["motion"]); mod.eval()
log(f"자 = {a.reader} (epoch {rck.get('epoch')}) · diff {MO.get('diff', 'block')} · 창 {a.W} 장 = 문맥 {C} + 예측 {R} 블록")

aug = cfg["data"].get("aug") or {}
tf = ClipTransform(256, hflip_p=0.0, random_resized_crop=aug.get("random_resized_crop"), square_crop=aug.get("square_crop", "none"), out_uint8=True)
ds = VarWindowBatches(cfg["data"]["var_window"], tf, TUB, seed=777, rank=0)
def read(i):
    rng = random.Random(f"gap-{i}")
    t = ds.tables[i % len(ds.tables)]
    for _ in range(5):
        try:
            return tf(ds._read(t, rng.randrange(t.first_ok(a.W), len(t.paths)), a.W, rng))
        except Exception:
            continue
    return None
with ThreadPoolExecutor(32) as ex:
    frames = [g for g in ex.map(read, range(a.clips)) if g is not None]
log(f"클립 {len(frames)}")

state = U.load_base_checkpoint(cfg["model"]["checkpoint"])
_, enc = U.build_frozen_encoders(cfg["model"], state, dev, torch.bfloat16, int(cfg["data"]["n_frames"]))
del state
H = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i in range(0, len(frames), a.bs):
        x = normalize_clips(torch.stack(frames[i:i + a.bs]).to(dev))
        H.append(AR.encode_blocks(enc, x.to(torch.bfloat16), TUB).to(torch.bfloat16).cpu())
H = torch.cat(H)
del enc; torch.cuda.empty_cache()
log(f"정답 토큰 {tuple(H.shape)}")

res = {}
for spec in a.models:
    pred, ep = load_pred(spec)
    acc = {k: 0.0 for k in ("fit", "tf", "ar", "copy", "dvar", "l1_tf", "l1_ar")}; n = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(frames), a.bs):
            x = normalize_clips(torch.stack(frames[i:i + a.bs]).to(dev))
            h = H[i:i + a.bs].to(dev).float()
            B = h.size(0)
            idx = AR.block_indices(B, NB, S, dev)
            p_tf, p_ar = pred(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": R})
            _, lg = MOM.motion_losses(mod, x, h, p_tf.float(), p_ar.float(), S, C, R, MO)
            for k in ("fit", "tf", "ar", "copy", "dvar"):
                acc[k] += lg[k] * B
            acc["l1_tf"] += float(AR.l1(p_tf, h[:, S:], 1.0)) * B
            acc["l1_ar"] += float(AR.l1(p_ar, h[:, C * S:], 1.0)) * B
            n += B
    m = {k: v / n for k, v in acc.items()}
    v = m["dvar"]; gap = (m["copy"] - m["fit"])
    m.update(fit_r=m["fit"] / v, tf_r=m["tf"] / v, ar_r=m["ar"] / v, copy_r=m["copy"] / v,
             fill_tf=(m["copy"] - m["tf"]) / gap if gap > 0 else float("nan"), fill_ar=(m["copy"] - m["ar"]) / gap if gap > 0 else float("nan"), epoch=ep)
    res[spec] = m
    log(f"{spec:36s} (e{ep})  정답 쌍 {m['fit_r']:.3f}  TF {m['tf_r']:.3f}  굴린 {m['ar_r']:.3f}  멈춤 {m['copy_r']:.3f}  "
        f"| 메운 몫 TF {100 * m['fill_tf']:5.1f} %  굴린 {100 * m['fill_ar']:5.1f} %  | 잠재 L1 TF {m['l1_tf']:.3f} 굴린 {m['l1_ar']:.3f}")
    del pred; torch.cuda.empty_cache()
out = a.out or f"z_training/runs/{rrun}/motion_gap_compare.json"
json.dump({"args": vars(a), "reader_epoch": rck.get("epoch"), "results": res}, open(out, "w"), indent=1)
log("저장", out)
