"""Push-T (DINO-WM `pusht_noise`) 궤적 데이터셋 — action 조건부 AR predictor 학습용 (2026-10-09).

데이터 (`${WORLD_ROOT}/action_planning/dino_wm/pusht_noise/{train,val}`):
  obses/episode_%03d.mp4   224x224, 프레임 = sim 스텝 (train 18,685 / val 21 episode, 길이 49~246)
  states.pth (E, 246, 5)   [agent_x, agent_y, block_x, block_y, block_angle]  (0 패딩)
  rel_actions.pth (E, 246, 2)  agent 이동 Δxy (px)        seq_lengths.pkl  episode 길이
규약 (DINO-WM `TrajSlicerDataset` 과 같다):
  frameskip f (기본 5): 관측 t = 프레임 start + f·t.  action_t = 스텝 [start+f·t, start+f·(t+1)) 의 f 개 Δxy 를 **이어 붙인 2f 차원**
  = 관측 t 에서 t+1 로 가는 동안의 action (AC 의 a_k 와 같은 자리).  정규화는 train 의 action 평균/표준편차 (차원별) 로.
블록 = 관측 1 장을 **2 번 복제**한 tubelet (V-JEPA 2-AC 와 동일). 224 -> 256 리사이즈 (bilinear, antialias 없음 — 채점 관례).
좌우 반전·RRC 는 쓰지 않는다 — action 의 방향·크기가 화면 좌표에 묶여 있다.
"""
from __future__ import annotations

import json
import os
import pickle
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


class PushTDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str, n_blocks: int, frameskip: int, resolution: int,
                 fixed_starts: Optional[List[int]] = None, action_stats: Optional[dict] = None, seed: int = 0,
                 subset_frac: float = 1.0, subset_seed: int = 0, virtual_len: int = 0):
        self.root, self.split = root, split
        self.states, self.actions, self.lengths, split_ids = self._load_arrays(root, split)
        self.nb, self.fs, self.res = int(n_blocks), int(frameskip), int(resolution)
        self.span = self.nb * self.fs                       # 관측 nb 장 + 마지막 블록의 action f 개까지 필요한 스텝 수
        # 짧은 episode 는 미리 뺀다 (런타임 재추첨 금지 — CLAUDE.md 7-1). split_ids = 이 split 의 episode (None = 전부)
        cand = range(len(self.lengths)) if split_ids is None else split_ids
        self.ids = [i for i in cand if self.lengths[i] >= self.span]
        if len(self.ids) < len(cand):
            print(f"[{type(self).__name__}:{split}] {len(cand) - len(self.ids)} episode 가 {self.span} 스텝보다 짧아 제외")
        # ── 저데이터 실험 (2026-10-10): episode 의 subset_frac 만 쓴다 (subset_seed 로 고정 추출 — 두 팔이 같은 부분집합).
        #    virtual_len 을 주면 epoch 길이를 그 값으로 고정해 (부분집합을 순환) step / epoch 과 lr 스케줄을 전체 데이터 run 과 같게 둔다.
        #    action 정규화 통계는 늘 **전체 train** 의 것 (부분집합마다 바뀌면 action 단위가 팔마다 달라진다).
        self.subset_frac = float(subset_frac)
        if self.subset_frac < 1.0:
            k = max(1, int(round(len(self.ids) * self.subset_frac)))
            perm = list(self.ids); random.Random(subset_seed).shuffle(perm)       # 고정 순열의 앞 k 개 → 1 % ⊂ 10 % ⊂ 100 % (포함 관계)
            self.ids = sorted(perm[:k])
            print(f"[pusht:{split}] subset_frac {self.subset_frac}: {k} episode (seed {subset_seed})")
        self.virtual_len = int(virtual_len)
        self.action_stats = action_stats or type(self).compute_action_stats(root)
        self.a_mean = torch.tensor(self.action_stats["mean"]).view(1, 2)
        self.a_std = torch.tensor(self.action_stats["std"]).view(1, 2)
        # fixed_starts: val 용. episode 마다 이 시작점들에서 창을 낸다 (결정론). None 이면 무작위 시작 (train)
        self.fixed = None
        if fixed_starts is not None:
            self.fixed = [(i, s) for i in self.ids for s in fixed_starts if s + self.span <= self.lengths[i]]
        self.rng = random.Random(seed)
        self._vr = {}

    # ------------------------------------------------------------------ 파일 형식 (환경마다 다르다)
    ACTION_SCALE = 100.0        # env.step 이 action × 이 값을 쓴다 (Push-T: px/100). planner 가 읽는다
    STATE_DIM_PROPRIO = 2

    def _load_arrays(self, root: str, split: str):
        """Push-T: <root>/<split>/{states.pth, rel_actions.pth, seq_lengths.pkl, obses/episode_%03d.mp4}. split_ids None = 폴더 전체."""
        d = os.path.join(root, split)
        if not os.path.isdir(os.path.join(d, "obses")):
            raise FileNotFoundError(f"Push-T {split} 이 없다: {d}/obses")
        self.d = d
        states = torch.load(os.path.join(d, "states.pth")).float()
        actions = torch.load(os.path.join(d, "rel_actions.pth")).float()
        with open(os.path.join(d, "seq_lengths.pkl"), "rb") as f:
            lengths = [int(x) for x in pickle.load(f)]
        n = min(len(lengths), states.size(0), actions.size(0))
        return states[:n], actions[:n], lengths[:n], None

    def _frames(self, i: int, t_idx: List[int]) -> np.ndarray:
        """episode i 의 관측 t_idx → (len, H, W, 3) uint8."""
        return self._reader(i).get_batch(t_idx).asnumpy()

    # ------------------------------------------------------------------ stats
    @classmethod
    def compute_action_stats(cls, root: str) -> dict:
        """train split 의 유효 스텝 action 평균/표준편차 (차원별). json 캐시. val 도 **train 통계**를 쓴다."""
        path = os.path.join(root, "action_stats_train.json")
        if os.path.isfile(path):
            return json.load(open(path))
        obj = cls.__new__(cls)
        states, a, L, ids = obj._load_arrays(root, "train")
        ids = range(len(L)) if ids is None else ids
        mask = torch.zeros(a.shape[:2], dtype=torch.bool)
        for i in ids:
            mask[i, :int(L[i])] = True
        v = a[mask]
        st = {"mean": v.mean(0).tolist(), "std": v.std(0).tolist(), "n_steps": int(mask.sum())}
        try:
            json.dump(st, open(path, "w"))
        except OSError:
            pass
        return st

    # ------------------------------------------------------------------ io
    def __len__(self):
        if self.fixed is not None:
            return len(self.fixed)
        return max(len(self.ids), self.virtual_len)

    def _reader(self, i: int):
        import decord
        vr = self._vr.get(i)
        if vr is None:
            vr = decord.VideoReader(os.path.join(self.d, "obses", f"episode_{i:03d}.mp4"), num_threads=1)
            if len(self._vr) > 64:
                self._vr.clear()
            self._vr[i] = vr
        return vr

    def window(self, i: int, start: int) -> Dict[str, torch.Tensor]:
        """episode i, 시작 스텝 start -> clips (3, 2*nb, r, r) uint8 · actions (nb, 2f) 정규화 · proprio (nb, 2) · state (nb, 5)"""
        t_idx = [start + self.fs * t for t in range(self.nb)]
        fr = self._frames(i, t_idx)                                           # (nb, H, W, 3) uint8
        x = torch.from_numpy(fr).permute(0, 3, 1, 2).float().div_(255.0)      # (nb, 3, H, W)
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False)
        x = x.mul_(255.0).round_().clamp_(0, 255).to(torch.uint8)             # (nb, 3, r, r)
        clips = x.unsqueeze(2).repeat(1, 1, 2, 1, 1)                           # (nb, 3, 2, r, r)  프레임 복제 -> tubelet
        clips = clips.permute(1, 0, 2, 3, 4).reshape(3, 2 * self.nb, self.res, self.res)
        a = self.actions[i, start: start + self.span]                           # (nb*f, 2)
        a = ((a - self.a_mean) / self.a_std).reshape(self.nb, -1)               # 차원별 정규화 뒤 f 개 이어 붙임
        st = self.states[i, t_idx]                                               # (nb, state_dim)
        return {"clips": clips, "actions": a, "proprio": st[:, : self.STATE_DIM_PROPRIO], "state": st,
                "episode": torch.tensor(i), "start": torch.tensor(start)}

    def __getitem__(self, k: int):
        if self.fixed is not None:
            i, s = self.fixed[k]
        else:
            i = self.ids[k % len(self.ids)]                                  # virtual_len 이면 부분집합을 순환
            s = self.rng.randint(0, self.lengths[i] - self.span)
        return self.window(i, s)


class WallDataset(PushTDataset):
    """DINO-WM `wall_single` (2026-10-11): 1,920 episode × 50 step, 점 (dot) 이 벽의 문을 지나 반대편 목표로 간다.
    states.pth (E, 50, 2) = 점 위치, actions.pth (E, 50, 2) (env 단위, action_scale 1), 관측 = `obses_u8.npy` (E, 51, 224, 224, 3) uint8
    (`z_research/action_planning/data/convert_wall_obs.py` 로 만든 memmap). split = DINO-WM 과 같은 무작위 90 / 10
    (torch.randperm, Generator seed 42) → train 1,728 / val 192. door / wall 위치는 이 세트에서 전부 30 / 32 로 고정이다."""
    # ⚠️ 공개 env (`env/wall/envs/wall.py:_generate_transition`) 는 `location + action * 2` 인데 데이터 궤적은 `+ action × 1` 로 움직인다
    #    (2026-10-11 실측: 데이터 action 을 그대로 재생하면 10 step 에 6~7 어긋남). 모델은 데이터 동역학을 배우므로
    #    planner 는 env 에 **action / 2** 를 넣어야 계획대로 움직인다 → ACTION_SCALE 2 (env 단위 × 2 = 데이터 단위).
    ACTION_SCALE = 2.0
    STATE_DIM_PROPRIO = 2
    SPLIT_SEED, TRAIN_FRAC = 42, 0.9

    def _load_arrays(self, root: str, split: str):
        p = os.path.join(root, "obses_u8.npy")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Wall uint8 관측이 없다: {p} — convert_wall_obs.py 를 먼저 돌릴 것")
        self.d, self._obs_path, self._mm = root, p, None
        states = torch.load(os.path.join(root, "states.pth")).float()
        actions = torch.load(os.path.join(root, "actions.pth")).float()
        E, T = actions.shape[:2]
        perm = torch.randperm(E, generator=torch.Generator().manual_seed(self.SPLIT_SEED)).tolist()
        k = int(self.TRAIN_FRAC * E)
        ids = sorted(perm[:k]) if split == "train" else sorted(perm[k:])
        return states, actions, [T] * E, ids

    def _frames(self, i: int, t_idx: List[int]) -> np.ndarray:
        if self._mm is None:                                   # worker 마다 lazily 연다
            self._mm = np.load(self._obs_path, mmap_mode="r")
        return np.ascontiguousarray(self._mm[i, t_idx])


def make_traj_dataset(spec: dict, split: str, **kw):
    """data.datasets 의 type (pusht | wall) 에 맞는 궤적 데이터셋."""
    cls = {"pusht": PushTDataset, "wall": WallDataset}[spec["type"]]
    return cls(spec["root"], split, **kw)


ACTION_TYPES = ("pusht", "wall")


class PushTCollator:
    """(clips uint8 (B,3,2nb,r,r), None, None, info{actions (B,nb,2f), proprio, state, episode, start}) — train.py 규약에 맞춘다."""
    def __call__(self, batch):
        out = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
        clips = out.pop("clips")
        return clips, None, None, out
