#!/usr/bin/env python3
"""블록 AR predictor 가 register 토큰을 얼마나 쓰는지 (2026-10-27, 사용자 요청). CPU 로도 돈다 (학습이 GPU 를 꽉 쓸 때).

  /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/register_usage.py --run natural_ar_vw_state_reg4 e6.pt --device cpu --n 16

데이터: phys3d_single_v1 val, 창 raw 0 간격 3, 문맥 블록 8 → 굴린 8 걸음 (TF 는 블록 1..15).
① attention: 새 query (패치 / register) 가 register key 에 주는 확률 질량 — 층별 평균, TF (문맥 채우기) · 굴린 걸음 나눠서.
   기준 = 균등이면 받을 몫 (보이는 key 중 register 비율).
② 크기: predictor_norm 전 은닉의 토큰 노름 — register vs 패치 (registers 논문의 "노름 큰 토큰" 확인).
③ 지우기 (같은 가중치): (a) 넘김 끔 (reg_carry none: 블록마다 학습 register 만) (b) register 없이 (n_registers 0 경로 = 옛 코드)
   → TF · 굴린 latent L1 (정답 LN(target_encoder) 대비) 이 얼마나 바뀌나. 바뀌지 않으면 register 를 쓰지 않는 것."""
import argparse, glob, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml, decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                                                  # noqa: E402
from app.vjepa_frozen import ar as AR                                                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs=2, required=True, metavar=("RUN", "CKPT"))
ap.add_argument("--device", default="cpu")
ap.add_argument("--n", type=int, default=16)
a = ap.parse_args()
dev = torch.device(a.device); torch.set_num_threads(32)
S, C, K, ST = 256, 8, 8, 3
T = 2 * (C + K)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(dev); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(dev)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, _load_checkpoint(CK)["target_encoder"], tag="t")
tgt = tgt.to(dev, torch.bfloat16 if dev.type == "cuda" else torch.float32).eval()
D = os.path.join(REPO, "z_training/runs", a.run[0])
pc = yaml.safe_load(open(os.path.join(D, "config.yaml")))["model"]["predictor"]
Nr = int(pc.get("n_registers", 0))
sd = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D, a.run[1]), map_location="cpu", weights_only=False)["predictor"]).items()}
def build(nr, carry):
    m = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
                             num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"), n_registers=nr, reg_carry=carry)
    m.load_state_dict(sd if nr else {k: v for k, v in sd.items() if k != "registers"}, strict=True)
    return m.to(dev).eval()
full, nocarry, noreg = build(Nr, "state"), build(Nr, "none"), build(0, "state")
print(f"{a.run[0]}:{a.run[1]} · register {Nr} 개 · |학습 register| {sd['registers'].norm(dim=-1).mean():.2f}", flush=True)

# ---- attention 기록: SDPA 를 감싸 새 query 가 register key 에 주는 질량을 잰다 -----------------------------------
REC = {"on": False, "phase": None, "layer": 0, "stats": {}}
_sdpa = F.scaled_dot_product_attention
def sdpa(q, k, v, *args, **kw):
    if REC["on"]:
        Bq, Hh, Nq, Dh = q.shape; Nk = k.size(2)
        p = torch.softmax((q.float() @ k.float().transpose(-1, -2)) / Dh ** 0.5, dim=-1)        # (B, H, Nq, Nk)
        isreg = (torch.arange(Nk, device=q.device) % (S + Nr)) >= S
        mreg = p[..., isreg].sum(-1)                                                           # (B, H, Nq)
        L = REC["layer"] % 12; REC["layer"] += 1
        key = (REC["phase"], L)
        acc = REC["stats"].setdefault(key, [[], [], []])
        acc[0].append(float(mreg[..., :S].mean())); acc[1].append(float(mreg[..., S:].mean())); acc[2].append(float(isreg.float().mean()))
    return _sdpa(q, k, v, *args, **kw)
F.scaled_dot_product_attention = sdpa

paths = sorted(p for p in glob.glob("/workspace/dataset/training_data/phys3d_single_v1/val/clips/*/*.mp4") if np.load(p[:-4] + "_states.npz")["pos"].shape[0] >= 96)
paths = [paths[i] for i in np.random.default_rng(0).permutation(len(paths))[: a.n]]
L1 = {k: [[], []] for k in ("register 그대로", "넘김 끔", "register 없음")}
NORM = [[], []]
for ci, p in enumerate(paths):
    vr = decord.VideoReader(p, num_threads=2)
    x = torch.from_numpy(vr.get_batch([ST * k for k in range(T)]).asnumpy()).permute(3, 0, 1, 2)[None].float().div_(255).to(dev)
    if x.shape[-1] != 256:
        x = F.interpolate(x[0].transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)[None]
    with torch.no_grad():
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            h = AR.encode_blocks(tgt, ((x - MEAN) / STD).to(next(tgt.parameters()).dtype), 2).float()
        idx = torch.arange(h.size(1), device=dev)[None]
        # ① ② attention · 노름 (register 그대로 모델): TF 문맥 채우기 + 굴린
        REC["on"], REC["layer"] = True, 0
        REC["phase"] = "tf"
        _, saved = full._ar_reg_run(h[:, : C * S], idx[:, : C * S], keep=C - 1)
        REC["phase"] = "ro"
        pr, caches, stp, reg = saved; pos = idx[:, (C - 1) * S: C * S]
        for _ in range(1, K):
            pos = pos + S
            pr, caches, stp, str_ = full._ar_step_kv_reg(stp + full.type_embed[1], pos, caches, reg)
            reg = full._reg_next(str_)
        REC["on"] = False
        # 노름: 은닉 (predictor_norm 전) — 마지막 블록 하나에 hook
        hid = {}
        hk = full.predictor_norm.register_forward_hook(lambda m, i, o: hid.__setitem__("h", i[0]))
        full._ar_reg_run(h[:, : C * S], idx[:, : C * S])
        hk.remove()
        nn_ = hid["h"][0].norm(dim=-1); NORM[0].append(float(nn_[:S].mean())); NORM[1].append(float(nn_[S:].mean()))
        # ③ 지우기
        for name, m in (("register 그대로", full), ("넘김 끔", nocarry), ("register 없음", noreg)):
            p_tf, p_ar = m(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": K})
            L1[name][0].append(float((p_tf - h[:, S:]).abs().mean())); L1[name][1].append(float((p_ar - h[:, C * S:]).abs().mean()))
    print(f"clip {ci + 1}/{len(paths)}", flush=True)

print("\n① 새 query 가 register key 에 주는 attention 질량 (층별, 평균)  · 균등 몫 = 보이는 key 중 register 비율")
print(f"  {'층':>3s}  {'TF 패치→reg':>12s} {'TF reg→reg':>11s} {'굴린 패치→reg':>13s} {'굴린 reg→reg':>12s} {'균등 몫':>8s}")
for L in range(12):
    t, r = REC["stats"].get(("tf", L)), REC["stats"].get(("ro", L))
    print(f"  {L + 1:3d}  {np.mean(t[0]):12.4f} {np.mean(t[1]):11.4f} {np.mean(r[0]):13.4f} {np.mean(r[1]):12.4f} {np.mean(r[2]):8.4f}")
print(f"\n② 은닉 노름 (마지막 문맥 블록, predictor_norm 전): 패치 {np.mean(NORM[0]):.2f} · register {np.mean(NORM[1]):.2f}")
print("\n③ 지우기 — latent L1 (정답 대비, 클립 평균) TF / 굴린")
b = L1["register 그대로"]
for name, v in L1.items():
    print(f"  {name:14s} TF {np.mean(v[0]):.4f} ({100 * (np.mean(v[0]) / np.mean(b[0]) - 1):+.1f}%) · 굴린 {np.mean(v[1]):.4f} ({100 * (np.mean(v[1]) / np.mean(b[1]) - 1):+.1f}%)")
