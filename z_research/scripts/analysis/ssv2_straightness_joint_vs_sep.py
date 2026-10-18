#!/usr/bin/env python3
"""SSv2 곧음 — 프레임을 따로 encoding vs 한 번에 encoding (2026-10-15, 사용자 지시). ssv2_straightness.py 와 같은 200 영상.

같은 16 프레임 (각 영상 0, 2, …, 30 번째, 12 fps 로 2.7 s) · 같은 16 시점으로 맞춘다 — 프레임마다 두 번 이어 붙여 튜블릿 (2 프레임) 하나가 프레임 하나가 되게:
  sep   : 프레임 t 하나 → [f_t, f_t] (2 프레임) 를 encoder 에 따로 16 번 → 시간 칸 1 개 → 공간 평균 z̄_t   (이미지 encoder 처럼 — Hénaff 판과 같은 질문)
  joint : [f_0, f_0, f_2, f_2, …, f_30, f_30] (32 프레임) 를 한 번에 → 시간 칸 16 개 → 칸마다 공간 평균 z̄_t
  pixel : 프레임 t 64×64 회색
지표 (ssv2_straightness.py 와 같은 정의): φ_curv = mean_t θ_t (rad), φ_ang = std_t θ_t, R²_line · R²_quad. 대조군 = 시점 순서 섞기.
encoder = ViT-H context encoder (online `encoder`) · target encoder (EMA), 마지막 층, bf16.

  CUDA_VISIBLE_DEVICES=<gpu> python z_research/scripts/analysis/ssv2_straightness_joint_vs_sep.py
  → z_research/VOBench_v1/vjepa2/ssv2_straightness/joint_vs_sep.json (+ joint_vs_sep_traj.npz)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
from ssv2_straightness import r2  # noqa: E402

D = REPO / "z_research/VOBench_v1/vjepa2/ssv2_straightness"
CFG = REPO / "analysis/intphys2/configs/bench_intphys2_main_vith.yaml"
FR = list(range(0, 32, 2))   # 16 프레임


def curv(Z):
    v = np.diff(Z, axis=0); a, b = v[:-1], v[1:]
    c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12); th = np.arccos(np.clip(c, -1, 1))
    return float(th.mean()), float(th.std())


def main():
    import torch, yaml
    from decord import VideoReader, cpu
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.dataset import build_video_transform
    paths = list(np.load(D / "per_video.npz", allow_pickle=True)["paths"])
    cfg = yaml.safe_load(open(CFG)); dev = torch.device("cuda:0"); tf = build_video_transform(256, 1.0)
    mk = lambda w: build_from_config({**cfg["model"], "window_size": w, "dtype": "bfloat16"}, dev)
    Bj, Bs = mk(32), mk(2)
    out = {k: [] for k in ("pixel", "sep_enc", "sep_tgt", "joint_enc", "joint_tgt")}
    for i, p in enumerate(paths):
        fr = VideoReader(p, ctx=cpu(0)).get_batch(FR).asnumpy()                              # (16, H, W, 3)
        x = tf(list(np.repeat(fr, 2, axis=0))).to(dev)                                       # (3, 32, 256, 256): 프레임마다 두 번
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for enc_name, key in (("context_encoder", "enc"), ("target_encoder", "tgt")):
                tj = getattr(Bj, enc_name)(x[None])[0].float().view(16, 256, -1).mean(1)       # joint: 시간 칸 16
                xs = x.view(3, 16, 2, 256, 256).permute(1, 0, 2, 3, 4)                        # (16, 3, 2, 256, 256): 프레임마다 따로
                ts = getattr(Bs, enc_name)(xs).float().mean(1)                                # (16, 1280)
                out[f"joint_{key}"].append(tj.cpu().numpy()); out[f"sep_{key}"].append(ts.cpu().numpy())
        g = torch.nn.functional.interpolate(torch.from_numpy(fr).float().mean(-1)[:, None], size=(64, 64), mode="bilinear", align_corners=False)[:, 0]
        out["pixel"].append(g.reshape(16, -1).numpy())
        if i % 50 == 0: print(f"{i}/{len(paths)}", flush=True)
    np.savez(D / "joint_vs_sep_traj.npz", paths=np.array(paths), **{k: np.stack(v) for k, v in out.items()})
    summ = {"n_videos": len(paths), "frames": "0,2,…,30 (16 장)", "spaces": {}}
    for k, Zs in out.items():
        r = {}
        for kind in ("real", "shuffle"):
            cs, an, rl, rq = [], [], [], []
            for j, Z in enumerate(Zs):
                ZZ = Z if kind == "real" else Z[np.random.default_rng(1000 + j).permutation(16)]
                c, a = curv(ZZ); cs.append(c); an.append(a); rl.append(r2(ZZ, 1)); rq.append(r2(ZZ, 2))
            r[kind] = dict(phi_curv=float(np.mean(cs)), phi_curv_deg=float(np.degrees(np.mean(cs))), phi_ang=float(np.mean(an)), r2_line=float(np.mean(rl)), r2_quad=float(np.mean(rq)))
            print(f"{k:10s} {kind:7s} | φ_curv {r[kind]['phi_curv']:.3f} rad ({r[kind]['phi_curv_deg']:.1f}°) · φ_ang {r[kind]['phi_ang']:.3f} | R²_line {r[kind]['r2_line']:.3f} · R²_quad {r[kind]['r2_quad']:.3f}", flush=True)
        summ["spaces"][k] = r
    json.dump(summ, open(D / "joint_vs_sep.json", "w"), indent=1, ensure_ascii=False); print("saved", D / "joint_vs_sep.json")


if __name__ == "__main__":
    main()
