#!/usr/bin/env python3
"""경로 선생 진단 1/2 (GPU) — 문맥 latent 뽑기 + 릴리즈 · VOBench PFT 의 미래 query attention 기록 (2026-11-01, 사용자 요청).

물음: (1) 경로 선생 (학습 없는 feature 대응 → "미래 자리 r 에 올 물체는 문맥의 어느 토큰인가" 목표 경로 A*) 은 얼마나 정확한가 (v3 · v3t · VOBench).
      (2) 실제 predictor 의 attention 은 A* 와 얼마나 일치하나 — PFT 의 경로가 속도를 따르나, 외운 시간표 자리를 따르나.
이 스크립트는 재료만 만든다. 계산 · 판정은 route_teacher.py (CPU).

판 (set)
  vob_S1 · vob_T1 · vob_orig  VOBench 속도 sweep (1/3 · 2/3 · 1 배, 30 block). 문맥 z 는 1 판 0 단계가 뽑아 둔 것을 그대로 쓴다 (다시 안 뽑음).
  v3 · v3t                     BlenderTrack test 의 물체 있는 판 (pos_a) · target 움직임 · 렌더 타이밍 합격 (v3 는 가림막 장면만 — 가리는 물체는 마스크가 없다).
                               seed 0 으로 각 N 개. 창 = 설계 그대로 64 장 간격 2 → 32 샘플 (문맥 8 칸 · 미래 8 칸).
모델 = 채점 하네스와 같은 ViT-H (문맥 = online encoder, fp32 + autocast fp16). predictor = release · vobftg (VOBench split v2 + gravity PFT).
attention 기록 (E37 transfer_gap_r4_attn.py 와 같은 가로채기): predictor 의 각 층 sdpa 에서, 미래 칸 r (0..7) 마다
  query = 그 칸의 참 물체 중심 3×3 (상자 중심 1–14 로 자름) 9 토큰 평균 · head 평균 → 문맥 key 2,048 개 위의 질량 (+ 미래 key 질량 합).
  VOBench 1/3 · 2/3 배는 query 를 하나 더: 같은 block 의 1 배 판 참 자리 (= PFT 가 외운 학습셋 시간표 자리, transfer_gap Q3).
저장 /data/hyuntak/project/2026/2027_cvpr/nfs_cache/route_teacher/<set>.npz
  z (N, 8, 256, D) fp16 (v3 · v3t 만) · 라벨 · att_<model> (N, Q, 8, 12, 2048) fp16 · fut_<model> (N, Q, 8, 12) 미래 key 질량

  sbatch 본체: predictor_lab/scaffold/route_job.sh   ($P route_extract.py [N])
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
OUT = Path("/data/hyuntak/project/2026/2027_cvpr/nfs_cache/route_teacher")
STAGE0 = Path("/data/hyuntak/project/2026/2027_cvpr/nfs_cache/scaffold_stage0")
BT = {"v3": Path("/data/hyuntak/project/2026/2027_cvpr/nfs_cache/blender_track/v3"), "v3t": Path("/data/hyuntak/project/2026/2027_cvpr/nfs_cache/blender_track/v3t")}
MODEL_CFG = REPO / "cvpr/results/eval/testbed/vo_moving_occluder_vith/_resolved.yaml"
PREDS = [("release", None), ("vobftg", str(REPO / "z_training/runs/vobench_release_postft_g/latest.pt"))]
IDX = np.arange(0, 64, 2)
S, G, C, K, L = 256, 16, 8, 8, 12
KMAX = 3
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
GR, GC = np.meshgrid(np.arange(64, dtype=np.float32), np.arange(64, dtype=np.float32), indexing="ij")


# ----------------------------------------------------------------------------- v3 · v3t 표본 · 라벨
def bt_rows(name, n, seed=0):
    rows = [r for r in csv.DictReader(open(REPO / f"data_csv/blender_track_{name}/test_arms.csv"))
            if r["variant"] == "pos_a" and r["render_timing_ok"] == "1" and r["target_mode"] == "moving" and r["occ_type"] == "barrier"]
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(rows), min(n, len(rows)), replace=False))
    return [rows[i] for i in pick]


def bt_item(name, r):
    import decord
    cid = os.path.basename(os.path.realpath(r["file"]))[:-4]
    fr = decord.VideoReader(str(BT[name] / "clips" / f"{cid}.mp4"), num_threads=2).get_batch(list(map(int, IDX))).asnumpy()
    Lb = np.load(BT[name] / "labels" / f"{cid}.npz")
    n = int(Lb["amodal"].shape[1]); k = min(n, KMAX)
    am = np.unpackbits(Lb["amodal"][IDX, :k], axis=-1).astype(np.float32)          # (32, k, 64, 64)
    vi = np.unpackbits(Lb["visible"][IDX, :k], axis=-1).astype(np.float32)
    ms = am.sum((2, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.stack([(am * GC).sum((2, 3)) / ms, (am * GR).sum((2, 3)) / ms], -1) / 4.0 - 0.375   # (32, k, 2) 토큰 (열, 행) — 가려진 부분 포함 마스크 무게중심
        vf = np.where(ms > 0, vi.sum((2, 3)) / np.maximum(ms, 1), np.nan)
    ctr = np.full((16, KMAX, 2), np.nan, np.float32); vis = np.full((16, KMAX), np.nan, np.float32); pres = np.zeros((16, KMAX), bool)
    ctr[:, :k] = c.reshape(16, 2, k, 2).mean(1); vis[:, :k] = vf.reshape(16, 2, k).mean(1); pres[:, :k] = (ms > 0).reshape(16, 2, k).all(1)
    occ = vi.reshape(32, k, 16, 4, 16, 4).mean((3, 5)).reshape(16, 2, k, 256).mean(1)            # (16, k, 256) 칸별 보이는 점유 비율
    tocc = np.zeros((16, KMAX, 256), np.float16); tocc[:, :k] = occ
    tgt = np.zeros(KMAX, bool); tgt[:k] = [str(x) == "target" for x in Lb["roles"][:k]]
    x = (torch.from_numpy(fr).permute(3, 0, 1, 2).float() / 255.0 - MEAN) / STD
    meta = {kk: r[kk] for kk in ("block_id", "place", "ramp_mode", "target_kind", "occ_motion", "occ_sub", "camera", "objset", "count_a", "event_s", "event_e", "f_hide")}
    meta["cid"] = cid
    return x, dict(ctr=ctr, vis=vis, pres=pres, occ=tocc, tgt=tgt), meta


# ----------------------------------------------------------------------------- attention 가로채기
class AttnRec:
    """predictor forward 동안 각 층 sdpa 에서 지정 query 들의 문맥 key 질량을 기록한다 (q · k 는 RoPE 뒤 값)."""

    def __init__(self):
        self.on = False
        self._sdpa = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = self._hook

    def _hook(self, q, k, v, *a, **kw):
        out = self._sdpa(q, k, v, *a, **kw)
        if self.on and q.shape[2] == 2 * C * S and k.shape[2] == 2 * C * S:
            QI = self.QI                                                               # (B, Q, K, 9) 토큰 번호
            B, H = q.shape[:2]
            qq = torch.gather(q, 2, QI.view(B, 1, -1, 1).expand(B, H, QI[0].numel(), q.shape[-1]))      # (B, H, Q·K·9, d)
            A = torch.softmax(qq.float() @ k.float().transpose(-1, -2) * q.shape[-1] ** -0.5, -1)     # (B, H, Q·K·9, 4096)
            A = A.mean(1).view(B, QI.shape[1], K, 9, -1).mean(3)                                       # (B, Q, K, 4096)
            self.att[:, :, :, self.l] = A[..., :C * S].half().cpu()
            self.fut[:, :, :, self.l] = A[..., C * S:].sum(-1).float().cpu()
            self.l += 1
        return out

    def run(self, mod, z, ci, ti, QI, autocast):
        B = z.shape[0]
        self.QI = QI
        self.att = torch.zeros(B, QI.shape[1], K, L, C * S, dtype=torch.float16)
        self.fut = torch.zeros(B, QI.shape[1], K, L)
        self.l = 0; self.on = True
        with torch.no_grad(), autocast:
            mod(z, ci, ti, mask_index=0)
        self.on = False
        assert self.l == L, self.l
        return self.att.numpy(), self.fut.numpy()


def qidx(ctr_fut):
    """ctr_fut (B, Q, K, 2) 토큰 (열, 행) → 3×3 query 토큰 번호 (B, Q, K, 9) (상자 중심 1–14, 미래 칸 C+r)."""
    B, Q = ctr_fut.shape[:2]
    out = np.zeros((B, Q, K, 9), np.int64)
    for b in range(B):
        for q in range(Q):
            for r in range(K):
                x, y = ctr_fut[b, q, r]
                x = int(np.clip(np.round(x if np.isfinite(x) else 7), 1, G - 2)); y = int(np.clip(np.round(y if np.isfinite(y) else 7), 1, G - 2))
                out[b, q, r] = [(C + r) * S + (y + dy) * G + (x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    return out


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    import te_v11_score as te
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    cfg = yaml.safe_load(open(MODEL_CFG))
    sc = te.Scorer(cfg, dev, PREDS, keep_own=True)
    b = sc.bundle; D = sc.D
    rec = AttnRec()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # VOBench 속도 sweep — 저장된 z
    vob = {nm: np.load(STAGE0 / f"vob_{nm}.npz") for nm in ("S1", "T1", "orig")}
    for nm in ("S1", "T1", "orig"):
        d = vob[nm]; Z = d["z"]; cen = d["cen"]; n = len(Z)
        qsets = [cen[:, C:]] + ([vob["orig"]["cen"][:, C:]] if nm != "orig" else [])          # 참 자리 · (느린 판) 1 배 시간표 자리
        ctrq = np.stack(qsets, 1)                                                               # (n, Q, K, 2)
        res = {f"att_{t}": [] for t, _ in PREDS} | {f"fut_{t}": [] for t, _ in PREDS}
        for i0 in range(0, n, 8):
            z = torch.from_numpy(Z[i0:i0 + 8]).to(dev).float().reshape(-1, C * S, D)
            ci, ti = sc.idx(z.shape[0])
            QI = torch.from_numpy(qidx(ctrq[i0:i0 + 8])).to(dev)
            for (tag, _), mod in zip(PREDS, sc.mods):
                a, f = rec.run(mod, z, ci, ti, QI, sc.autocast())
                res[f"att_{tag}"].append(a); res[f"fut_{tag}"].append(f)
        np.savez(OUT / f"vob_{nm}.npz", cen=cen, vis=d["vis"], vid=d["vid"], qsets=np.array(["true", "timetable"][:len(qsets)]),
                 **{k_: np.concatenate(v_) for k_, v_ in res.items()})
        print(f"[vob_{nm}] {n} block {time.time() - t0:.0f}s", flush=True)

    # v3 · v3t
    for name in ("v3", "v3t"):
        rows = bt_rows(name, N)
        Zs, labs, metas = [], [], []
        res = {f"att_{t}": [] for t, _ in PREDS} | {f"fut_{t}": [] for t, _ in PREDS}
        for i0 in range(0, len(rows), 8):
            items = [bt_item(name, r) for r in rows[i0:i0 + 8]]
            x = torch.stack([it[0] for it in items]).to(dev, b.dtype)
            ci, ti = sc.idx(x.shape[0])
            with torch.no_grad(), sc.autocast():
                z = b.context_encoder(x, masks=[ci]).float()
            lab = [it[1] for it in items]
            tk = [int(np.argmax(l_["tgt"])) for l_ in lab]
            ctrq = np.stack([l_["ctr"][C:, j][None] for l_, j in zip(lab, tk)])                  # (B, 1, K, 2) target 참 자리
            QI = torch.from_numpy(qidx(ctrq)).to(dev)
            for (tag, _), mod in zip(PREDS, sc.mods):
                a, f = rec.run(mod, z, ci, ti, QI, sc.autocast())
                res[f"att_{tag}"].append(a); res[f"fut_{tag}"].append(f)
            Zs.append(z.view(-1, C, S, D).half().cpu().numpy()); labs += lab; metas += [it[2] for it in items]
            if i0 % 80 == 0:
                print(f"[{name}] {i0 + len(items)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
        np.savez(OUT / f"{name}.npz", z=np.concatenate(Zs), **{k_: np.stack([l_[k_] for l_ in labs]) for k_ in labs[0]},
                 meta=np.array([list(m.items()) for m in metas], dtype=object), **{k_: np.concatenate(v_) for k_, v_ in res.items()})
        print(f"[{name}] {len(rows)} clip 저장 {time.time() - t0:.0f}s", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
