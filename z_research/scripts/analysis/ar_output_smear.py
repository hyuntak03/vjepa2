"""번짐 가설 — 자연 영상 AR 출력은 물체를 움직임 방향으로 길게 번지게 놓나 (2026-10-26, GPU 1 장 · 6 GB).
사용자: "ㄱㄱ". 앞 측정: 자연 영상 AR 출력의 속도는 encoder 꼴이 아니다 (ar_output_velocity_linear.json). 추측: 물체를 마지막 자리와 갈 자리 사이에 번지게 놓고,
  번짐의 방향 · 길이가 속도를 담는다. encoder 는 속도를 두 장 사이 시간 변화로 담으므로 encoder 판독기는 못 읽는다.
측정 (3D 합성 물체 하나 val 350 편, teacher forcing 출력, 대상 블록 b = 2–15 중 b−1 · b 물체 보임 · 블록 사이 이동 |d| ≥ 12 px):
  배경 기준: 칸마다 그 영상에서 물체 중심이 그 칸에서 가장 먼 블록의 encoder 토큰 (카메라 고정).
  물체 지도 o(c) = ‖X(c) − 배경(c)‖₂, 두 위치 중점에서 6 칸 넘게 먼 칸의 중앙값을 빼고 0 아래는 0, 중점 5 칸 안만.
  X = 정답 블록 encoder 토큰 · 모델 출력 · (참고) 두 블록 encoder 토큰 평균 (순수 번짐 기준).
  지표: 무게 중심 진행 (직전 위치 → 실제 위치 방향, 0 = 직전 자리, 1 = 실제 자리) · 움직임 방향 퍼짐 σ∥ · 수직 퍼짐 σ⊥ · 비 σ∥/σ⊥ (중앙값).
모델: natural_ar_vw_state (자연) · natural_ar_vw_state_phys3d_only (3D 합성만) · natural_ar_vw_state_phys3d_event (자연 + 합성 보조).
  CUDA_VISIBLE_DEVICES=0 MEM_GB=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/ar_output_smear.py
출력: z_analysis/results/ar_output_smear.json
"""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F
import decord
REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 6)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT
sys.argv = _argv
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
S, D, NB = 256, 1280, 16; IDX = [3 * k for k in range(32)]
MODELS = {"natural": "natural_ar_vw_state", "phys3d_only": "natural_ar_vw_state_phys3d_only", "mix_event": "natural_ar_vw_state_phys3d_event"}
enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
_load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.half().to(dev).eval()
ARS = {k: FT.load_pred(v, "latest.pt").to(dev).eval() for k, v in MODELS.items()}
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
idx_all = torch.arange(NB * S, device=dev).unsqueeze(0)
gy, gx = np.mgrid[0:16, 0:16]; CC = np.stack([gx.ravel() * 16 + 8.0, gy.ravel() * 16 + 8.0], 1)       # 칸 중심 (x, y) px


def shape_stats(o, prev, cur):
    d = cur - prev; u = d / np.linalg.norm(d); nrm = np.array([-u[1], u[0]]); mid = (prev + cur) / 2
    dist = np.linalg.norm(CC - mid, axis=1)
    o = o - np.median(o[dist > 6 * 16]); o[(o < 0) | (dist > 5 * 16)] = 0
    if o.sum() <= 0:
        return None
    w = o / o.sum(); c = (w[:, None] * CC).sum(0); r = CC - c
    sa = np.sqrt((w * (r @ u) ** 2).sum()); sp = np.sqrt((w * (r @ nrm) ** 2).sum())
    return [float((c - prev) @ u / np.linalg.norm(d)), float(sa), float(sp), float(sa / max(sp, 1e-6))]


meta = json.load(open("/workspace/auto_research_cache/ar_layers/val_meta.json"))
ST = {k: [] for k in ["truth", "two_block_average", *MODELS]}
for n, m in enumerate(meta):
    B = m["blocks"]
    fr = decord.VideoReader(m["clip"], num_threads=4).get_batch(IDX).asnumpy()
    x = (torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0 - MEAN) / STD
    xb = x.reshape(3, NB, 2, 256, 256).permute(1, 0, 2, 3, 4)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.cat([F.layer_norm(enc2(xb[i:i + 4].half().contiguous()).float(), (D,)) for i in range(0, NB, 4)])
        outs = {k: ar.forward_seq(h.reshape(1, -1, D), idx_all).float()[0].view(NB, S, D).cpu().numpy() for k, ar in ARS.items()}
    h = h.cpu().numpy()
    ctr = np.array([b_["ctr"] for b_ in B]); far = np.array([[np.linalg.norm(CC[c] - ctr[b]) if np.isfinite(ctr[b]).all() else 999 for b in range(NB)] for c in range(S)])
    bg = h[far.argmax(1), np.arange(S)]                                                       # (256, 1280)
    for b in range(2, NB):
        if not (B[b]["ok"] and B[b - 1]["ok"]):
            continue
        prev, cur = ctr[b - 1], ctr[b]
        if np.linalg.norm(cur - prev) < 12:
            continue
        maps = {"truth": h[b], "two_block_average": 0.5 * (h[b - 1] + h[b]), **{k: outs[k][b - 1] for k in MODELS}}
        res = {k: shape_stats(np.linalg.norm(v - bg, axis=1), prev, cur) for k, v in maps.items()}
        if all(r is not None for r in res.values()):
            for k, r in res.items():
                ST[k].append(r)
    if (n + 1) % 100 == 0:
        print(f"  {n + 1}/{len(meta)}", flush=True)
R = {"n": len(ST["truth"])}
for k, v in ST.items():
    v = np.array(v); R[k] = {"centroid_progress_med": round(float(np.median(v[:, 0])), 3), "sigma_along_px_med": round(float(np.median(v[:, 1])), 1),
                             "sigma_perp_px_med": round(float(np.median(v[:, 2])), 1), "along_over_perp_med": round(float(np.median(v[:, 3])), 3)}
print(json.dumps(R, indent=1)); json.dump(R, open(os.path.join(REPO, "z_analysis/results/ar_output_smear.json"), "w"), indent=1)
