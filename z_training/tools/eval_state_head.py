#!/usr/bin/env python3
"""state_head 채점 — phys2d_sup val 창에서 **순수 rollout** (TF 없음) 으로 head 를 읽는다 (2026-10-19).

  python z_training/tools/eval_state_head.py --run z_training/runs/phys2d_sup_vast_ar_state_pft [--ckpt latest.pt] [--gpu 0]

문맥 C 블록 (기본 8) → rollout K 블록 (8). 블록마다 head 출력 (존재 · 위치 · 속도) 을 라벨 (그 블록 마지막 장) 과 비교한다.
위치 기준선 두 개 (라벨로 만든 '완벽한 지각' 기준선 — 모델보다 유리하다):
  copy  마지막으로 보인 문맥 블록의 위치 그대로
  cv    그 위치 + 그 블록 속도 × 지난 장 수 (등속 외삽)
나누는 축: 보임 / 가려짐 (목표 블록에서), 걸음 r, 범주 (gravity · permanence · solidity · mixed).
출력: <run>/eval/state_head_<ckpt>.json · .md
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from app.vjepa_frozen import ar as AR  # noqa: E402
from app.vjepa_frozen import state_head as SHM  # noqa: E402
from app.vjepa_frozen import utils as U  # noqa: E402
from app.vjepa_frozen.data import ClipTransform, Phys2DSupDataset, normalize_clips  # noqa: E402

FRAMES_PER_BLOCK = 2


def baselines(lab, C, K):
    """lab (B,T,48,...) numpy. return copy · cv 위치 (B,K,48,2) /256 단위, ok (B,48) 마지막으로 보인 문맥 블록이 있나"""
    vis = lab["m_state"][:, :C] & ~lab["hidden"][:, :C]                       # (B,C,48)
    B = vis.shape[0]
    has = vis.any(1)
    last = np.where(has, C - 1 - np.argmax(vis[:, ::-1], axis=1), 0)            # (B,48)
    bi = np.arange(B)[:, None]
    ki = np.arange(48)[None]
    p0 = lab["pos"][bi, last, ki]                                               # (B,48,2)
    v = lab["vel"][bi, last, ki]
    v0 = v[..., :2] * (v[..., 2:] * SHM.SPEED_SCALE) / 256.0                     # px/장 → /256
    dt = ((C + np.arange(K))[None, :, None] - last[:, None, :]) * FRAMES_PER_BLOCK   # (B,K,48) 장
    copy = np.broadcast_to(p0[:, None], (B, K, 48, 2))
    cv = p0[:, None] + v0[:, None] * dt[..., None]
    return copy, cv, has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--csv", default=os.path.join(ROOT, "data_csv/phys2d_sup_v1/val_T32C16s1.csv"))
    ap.add_argument("--context", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cpu" if a.cpu else f"cuda:{a.gpu}")
    ck_path = os.path.join(a.run, a.ckpt)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    MD = dict(cfg["model"])
    n_frames = int(cfg["data"]["n_frames"])
    state = U.load_base_checkpoint(MD["checkpoint"])
    dt = torch.bfloat16
    _, tgt = U.build_frozen_encoders(MD, state, dev, dt, n_frames)
    del state
    pred = U.build_predictor(dict(MD, predictor_init_checkpoint=ck_path), tgt.embed_dim, {"_": 1}, dev, n_frames).eval()
    SH = ck.get("state_head_cfg") or {}
    head = SHM.StateHead(tgt.embed_dim, num_heads=int(SH.get("num_heads", 16)), hidden=int(SH.get("hidden", 512))).to(dev).eval()
    head.load_state_dict(ck["state_head"])
    tub, S = int(MD.get("tubelet_size", 2)), (int(MD.get("img_size", 256)) // int(MD.get("patch_size", 16))) ** 2

    ds = Phys2DSupDataset({"csv": a.csv, "limit": a.limit or None}, n_frames, ClipTransform(256, out_uint8=True), hflip_p=0.0, tubelet=tub)
    with open(a.csv, newline="") as f:
        meta = [(r["scenario"], r["category"]) for r in csv.DictReader(f)][: len(ds)]
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=6,
                                     collate_fn=lambda b: (torch.stack([x[0][0] for x in b]),
                                                           {k: torch.stack([x[3][k] for x in b]) for k in b[0][3]}))
    C = a.context
    nb = n_frames // tub
    K = nb - C
    E = collections.defaultdict(list)          # (구분 키) → 오차 목록
    exist_hits = []
    k0 = 0
    ac = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", dtype=torch.bfloat16)
    with torch.no_grad():
        for clips, lab in dl:
            B = clips.size(0)
            x = normalize_clips(clips.to(dev))
            with ac:
                h = AR.encode_blocks(tgt, x.to(dt), tub)
                idx = AR.block_indices(B, nb, S, dev)
                p_ar = pred.rollout(h[:, : C * S], idx[:, : C * S], K)
                o = head(p_ar, S)
            L = {k: v.numpy() for k, v in lab.items()}
            pp = o["pos"].float().cpu().numpy()                                      # (B,K,48,2)
            ex = o["exist"].float().cpu().numpy() > 0
            exist_hits.append((ex == L["seen"][:, C:]).reshape(-1))
            copy, cv, has = baselines(L, C, K)
            gt = L["pos"][:, C:]
            m = L["m_state"][:, C:] & has[:, None, :]
            hid = L["hidden"][:, C:]
            mov = L["vel"][:, C:, :, 2] * SHM.SPEED_SCALE > SHM.DIR_MIN_SPEED          # 목표 블록에서 움직이는 물체
            for name, P in (("model", pp), ("copy", copy), ("cv", cv)):
                err = np.linalg.norm(P - gt, axis=-1) * 256.0                        # (B,K,48)
                for b in range(B):
                    cat = meta[k0 + b][1]
                    for r in range(K):
                        for kk in np.nonzero(m[b, r])[0]:
                            vis_s = "hid" if hid[b, r, kk] else "vis"
                            e = float(err[b, r, kk])
                            E[(name, vis_s, "all", -1)].append(e)
                            E[(name, vis_s, "all", r)].append(e)
                            E[(name, vis_s, cat, -1)].append(e)
                            if mov[b, r, kk]:
                                E[(name, vis_s, "moving", -1)].append(e)
                                E[(name, vis_s, "moving", r)].append(e)
            k0 += B
    ex_acc = float(np.concatenate(exist_hits).mean())

    def med(key):
        v = E.get(key, [])
        return (float(np.median(v)), float(np.mean(v)), len(v)) if v else (float("nan"), float("nan"), 0)

    res = {"ckpt": ck_path, "epoch": ck.get("epoch"), "n_windows": len(ds), "C": C, "K": K, "exist_acc_rollout": ex_acc, "pos": {}}
    for key in E:
        res["pos"]["|".join(map(str, key))] = med(key)
    lines = [f"# state_head 채점 — `{ck_path}` (epoch {ck.get('epoch')}), phys2d val {len(ds)} 창, 문맥 {C} 블록 → rollout {K}",
             "", f"- 존재 정확도 (rollout 블록, 48 쿼리): **{ex_acc * 100:.1f}%**", "",
             "## 위치 오차 (px, 중앙 / 평균, n) — 모델 vs 라벨 기준선 (copy = 마지막으로 보인 자리, cv = 등속 외삽)", "",
             "| 구분 | 모델 | copy | cv | n |", "|---|---:|---:|---:|---:|"]
    for vis_s in ("vis", "hid"):
        for cat in ("all", "moving", "gravity", "permanence", "solidity", "mixed"):
            r_ = [med((nm, vis_s, cat, -1)) for nm in ("model", "copy", "cv")]
            lines.append(f"| {vis_s} · {cat} | {r_[0][0]:.1f} / {r_[0][1]:.1f} | {r_[1][0]:.1f} / {r_[1][1]:.1f} | {r_[2][0]:.1f} / {r_[2][1]:.1f} | {r_[0][2]:,} |")
    lines += ["", "## 걸음별 (중앙 px, 움직이는 물체만 — 정지 물체는 copy 가 0 이라 중앙값을 지배한다)", "", "| r | 보임 모델 | copy | cv | 가려짐 모델 | copy | cv |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in range(K):
        v = [med((nm, s, "moving", r))[0] for s in ("vis", "hid") for nm in ("model", "copy", "cv")]
        lines.append(f"| {r + 1} | " + " | ".join(f"{x:.1f}" for x in v) + " |")
    od = os.path.join(a.run, "eval")
    os.makedirs(od, exist_ok=True)
    stem = os.path.join(od, f"state_head_{os.path.splitext(a.ckpt)[0]}")
    with open(stem + ".json", "w") as f:
        json.dump(res, f, indent=1)
    with open(stem + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
