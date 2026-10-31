"""BlenderTrack 학습 데이터 — 배치마다 (보폭, 문맥 길이) 를 고르고, 샘플마다 late 창을 라벨로 찾아 자른다 (2026-10-31, 사용자 결정).

데이터: z_research/scripts/data/blender_track (눈높이 Blender 렌더, 30 fps × 180 장, 256²) — clip 폴더마다 rgb/0000.png … · labels.npz.
메타: z_training/data/build_blender_track_index.py 가 만든 <meta>.npz (clip 이름 · target 의 프레임별 보이는 비율).

창 (수집기에서, DataLoader worker 안):
  배치마다 보폭 s ∈ strides (30 fps 프레임), 문맥 C ∈ context_frames (샘플) 를 하나 고른다. 창 = n_frames 샘플 (기본 32).
  샘플마다 late_p 확률로 late 창 — 문맥 마지막 샘플에서 target 이 화면 안에서 가려져 있고 (보이는 비율 < hid_vis),
  문맥 앞쪽에서 보였고 (≥ seen_vis), 미래에서 다시 보이는 창 — 을 그 (s, C) 의 후보 중에서 고른다 (clip 균등 → 시작점 균등).
  나머지는 sampler 가 준 clip 에서 시작점 균등.
라벨 (블록 = 튜블릿 2 샘플):
  pos (nb, 2)  target 투영 중심의 [행, 열] 토큰 칸 (칸 c 의 중심 = c), 블록 평균
  present ()   1 (BlenderTrack 의 target 은 늘 있다; 화면 밖은 binding_mask 가 pos 범위로 거른다)
  hid (nb,)    1 − 보이는 비율 (화면 밖 = 1)
  occ (nb, 16, 16)  amodal 점유 비율 = 64×64 amodal 칸을 4×4 로 묶고 블록 2 장 평균 (가려져도 물체가 있는 곳)
증강: 좌우 반전만 (라벨도 같이). 자르기 · 크기 바꾸기는 안 한다 (렌더가 이미 256²).
"""
from __future__ import annotations

import logging
import os
import random
from typing import Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)
G = 16          # 토큰 격자
L = 64          # 라벨 격자


class BlenderTrackDataset(torch.utils.data.Dataset):
    """__getitem__ 은 clip 번호만 돌려준다. 읽기 · 자르기 · 라벨은 BlenderTrackCollator 가 (s, C) 를 고른 뒤 한다."""

    def __init__(self, spec: dict, n_frames: int, transform):
        m = np.load(spec["meta"], allow_pickle=False)
        self.root = str(spec.get("root") or m["root"])
        self.clips: List[str] = [str(c) for c in m["clip"]]
        self.vis = m["vis"].astype(np.float32)            # (N, T) target 보이는 비율, 화면 밖 = nan
        self.T = int(self.vis.shape[1])
        lim = int(spec.get("limit") or 0)
        if lim:
            self.clips, self.vis = self.clips[:lim], self.vis[:lim]
        self.n_frames, self.transform = int(n_frames), transform
        self.sample_weights = None
        logger.info(f"[blender_track] {spec['meta']}: {len(self.clips)} clips × {self.T} 프레임, root {self.root}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i: int):
        return int(i)

    def read(self, c: int, idx: np.ndarray) -> torch.Tensor:
        from PIL import Image
        d = os.path.join(self.root, self.clips[c], "rgb")
        return torch.from_numpy(np.stack([np.asarray(Image.open(f"{d}/{int(f):04d}.png").convert("RGB")) for f in idx]))

    def labels(self, c: int, idx: np.ndarray, flip: bool, tub: int = 2) -> Dict[str, torch.Tensor]:
        z = np.load(os.path.join(self.root, self.clips[c], "labels.npz"))
        n = len(idx); nb = n // tub
        am = np.unpackbits(z["amodal"][idx, 0], axis=-1).astype(np.float32)          # (n, 64, 64)
        occ = am.reshape(nb, tub, G, L // G, G, L // G).mean((1, 3, 5))                 # (nb, 16, 16)
        u, v = z["center_px"][idx, 0, 0].astype(np.float32), z["center_px"][idx, 0, 1].astype(np.float32)
        vis = self.vis[c, idx]
        hid = np.where(np.isnan(vis), 1.0, 1.0 - vis).astype(np.float32)
        if flip:
            occ = occ[..., ::-1]
            u = 256.0 - u
        pos = np.stack([v / 16.0 - 0.5, u / 16.0 - 0.5], -1).reshape(nb, tub, 2).mean(1)
        return {"pos": torch.from_numpy(pos.astype(np.float32)), "present": torch.tensor(1.0),
                "hid": torch.from_numpy(hid.reshape(nb, tub).mean(1)), "occ": torch.from_numpy(np.ascontiguousarray(occ))}


class BlenderTrackCollator:
    """clip 번호 배치 → (clips (B,3,n,H,W), enc, pred, info{labels}). 배치마다 (s, C) 하나, 샘플마다 late 창 late_p."""

    def __init__(self, ds: BlenderTrackDataset, mask_sampler, cfg: dict, tubelet: int = 2):
        self.ds, self.masks, self.tub = ds, mask_sampler, int(tubelet)
        self.n = ds.n_frames
        self.strides = [int(x) for x in cfg.get("strides", [2, 3, 4])]
        self.ctx = [int(x) for x in cfg.get("context_frames", [8, 12, 16, 20, 24])]
        self.p_late = float(cfg.get("late_p", 0.5))
        hid_vis, seen_vis = float(cfg.get("hid_vis", 0.1)), float(cfg.get("seen_vis", 0.5))
        for C in self.ctx:
            if C % self.tub or not (0 < C < self.n):
                raise ValueError(f"blender_track context_frames={C}: tubelet 정렬이고 0<C<{self.n}")
        for s in self.strides:
            if (self.n - 1) * s + 1 > ds.T:
                raise ValueError(f"blender_track stride {s}: 창 {(self.n - 1) * s + 1} 프레임 > clip {ds.T}")
        on = ~np.isnan(ds.vis)
        vis = np.nan_to_num(ds.vis, nan=0.0)
        hid, seen = on & (vis < hid_vis), on & (vis >= seen_vis)
        self.cand, msg = {}, []
        for s in self.strides:
            span = (self.n - 1) * s + 1
            J = np.arange(ds.T - span + 1)[:, None] + s * np.arange(self.n)[None]     # (O, n)
            H, S = hid[:, J], seen[:, J]                                               # (N, O, n)
            for C in self.ctx:
                ok = H[:, :, C - 1] & S[:, :, :C - 1].any(-1) & S[:, :, C:].any(-1)   # (N, O)
                cl = np.nonzero(ok.any(1))[0]
                self.cand[(s, C)] = (cl, [np.nonzero(ok[c])[0] for c in cl])
                msg.append(f"s{s}C{C}:{len(cl)}")
        logger.info(f"[blender_track] late 후보 clip 수 (보폭 s, 문맥 C 샘플) {' '.join(msg)} · late_p {self.p_late}")

    def __call__(self, batch):
        s, C = random.choice(self.strides), random.choice(self.ctx)
        span = (self.n - 1) * s + 1
        cl, offs = self.cand[(s, C)]
        xs, labs, n_late = [], [], 0
        for i in batch:
            if len(cl) and random.random() < self.p_late:
                k = random.randrange(len(cl)); c = int(cl[k]); o = int(random.choice(offs[k])); n_late += 1
            else:
                c = int(i); o = random.randint(0, self.ds.T - span)
            idx = o + s * np.arange(self.n)
            x, prm = self.ds.transform(self.ds.read(c, idx), return_params=True)
            if prm["HW"] != (256, 256) or prm["box"] != (0, 0, 256, 256):
                raise ValueError("blender_track: 라벨은 256² 원본 좌표다 — 자르기 · 크기 바꾸기 증강을 끄라")
            xs.append(x)
            labs.append(self.ds.labels(c, idx, prm["flip"], self.tub))
        clips = torch.stack(xs)
        enc, pred = self.masks.temporal_prefix(len(batch), self.n, C)
        labels = {k: torch.stack([b[k] for b in labs]) for k in labs[0]}
        info = {"type": f"bt s{s} C{C}", "context_frames": C, "stride": s, "n_late": n_late, "labels": labels}
        return clips, enc, pred, info
