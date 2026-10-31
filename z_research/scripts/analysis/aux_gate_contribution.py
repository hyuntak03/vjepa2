#!/usr/bin/env python3
"""보조 모듈이 고정 릴리즈 predictor 에 실제로 얼마나 더해지나 (2026-10-31, 사용자: "gating 잘 안쓰는 거 같은데").

gate 값 하나로는 못 읽는다 — 더해지는 양은 gate × proj(attn(m)) 이고 proj 가중치도 학습된다.
재는 것 (문맥 16 장 → 미래 8 블록, 미래 토큰만):
  층별 상대 크기 = ‖gate·cross 출력‖ / ‖그 층 은닉 h‖ (토큰마다 L2, 평균)
  출력 변화 = mean |p_모듈 − p_릴리즈| / mean |p_릴리즈 − 정답|  (= 오차 대비 얼마나 움직였나)
  L1 이득 % = (릴리즈 L1 − 모듈 L1) / 릴리즈 L1
데이터: phys3d_perm_v1 val 앞 n clip (학습 생성기) · VOBench late 가능 변이 n clip (처음 보는 생성기).

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/aux_gate_contribution.py --run release_aux_perm_sup --ckpt latest.pt
"""
import argparse, csv, json, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="release_aux_perm_sup")
ap.add_argument("--ckpt", default="latest.pt")
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--mem", type=float, default=0.25)
a = ap.parse_args()
dev = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(a.mem)
from app.vjepa_frozen import utils as U
from app.vjepa_frozen.aux_memory import AuxRelease
RD = os.path.join(REPO, "z_training/runs", a.run)
cfg = yaml.safe_load(open(os.path.join(RD, "config.yaml"))); MD = cfg["model"]
NF, S, C = 32, 256, 16
state = U.load_base_checkpoint(MD["checkpoint"])
ctx_enc, tgt_enc = U.build_frozen_encoders(MD, state, dev, torch.bfloat16, NF)
model = AuxRelease(U.build_predictor(MD, ctx_enc.embed_dim, state, dev, NF), dict(MD["aux"]), ctx_enc.embed_dim, 256, 16, 2, NF).to(dev)
ck = torch.load(os.path.join(RD, a.ckpt), map_location="cpu", weights_only=False)
model.load_state_dict(ck["predictor"], strict=True); model.eval(); ep = ck.get("epoch"); del state, ck
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
REC = {}
for i, cr in model.cross.items():                       # cross 출력 / 입력 크기 기록
    def hook(mod, inp, out, i=i):
        REC.setdefault(i, []).append(float((out.float().norm(dim=-1) / inp[0].float().norm(dim=-1).clamp_min(1e-6)).mean()))
    cr.register_forward_hook(hook)


def clips(kind):
    if kind == "perm":
        import decord
        ps = [ln.split()[0] for ln in open(os.path.join(REPO, "data_csv/phys3d_perm_v1/val_occ.csv"))]
        for p in [ps[j] for j in np.random.default_rng(0).permutation(len(ps))[: a.n]]:
            yield decord.VideoReader(p, num_threads=2).get_batch(list(range(NF))).asnumpy()
        return
    from PIL import Image
    R = "/workspace/dataset/benchmarks/VOBench_v1"
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1_late/index.csv"))) if r["variant"] in ("pos_a", "pos_b")]
    for r in [rows[j] for j in np.random.default_rng(0).permutation(len(rows))[: a.n]]:
        fr = np.stack([np.asarray(Image.open(os.path.join(R, r["file_name"], f"{3 * j:06d}.png")).convert("RGB")) for j in range(NF)])
        yield F.interpolate(torch.from_numpy(fr).permute(0, 3, 1, 2).float(), size=(256, 256), mode="bilinear",
                            align_corners=False).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


me = torch.arange(C // 2 * S, device=dev)[None]; mp_ = torch.arange(C // 2 * S, NF // 2 * S, device=dev)[None]
print(f"{a.run}/{a.ckpt} (epoch {ep}) · gate {', '.join(f'{k + 1}층 {v:+.4f}' for k, v in model.gates().items())}")
for kind in ("perm", "vobench"):
    REC.clear(); mv, er, lr_, la = [], [], [], []
    for fr in clips(kind):
        x = ((torch.from_numpy(fr).to(dev).permute(3, 0, 1, 2)[None].float().div_(255) - MEAN) / STD).to(torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = F.layer_norm(tgt_enc(x).float(), (1280,))[:, C // 2 * S:]
            z = ctx_enc(x, masks=[me])
            pr = model.pred(z, me, mp_, mask_index=0).float(); pa = model(z, me, mp_, mask_index=0).float()
        mv.append(float((pa - pr).abs().mean())); er.append(float((pr - h).abs().mean()))
        lr_.append(float((pr - h).abs().mean())); la.append(float((pa - h).abs().mean()))
    rel = " · ".join(f"{int(k) + 1}층 {100 * np.mean(v):.2f} %" for k, v in sorted(REC.items(), key=lambda t: int(t[0])))
    print(f"  {kind:8s} 은닉 대비 더해지는 크기: {rel} | 출력 변화 / 오차 {100 * np.mean(mv) / np.mean(er):.2f} % | L1 이득 {100 * (np.mean(lr_) - np.mean(la)) / np.mean(lr_):+.2f} %")
