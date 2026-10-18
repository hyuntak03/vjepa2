"""가변 창 batch 로더 (2026-10-02, 사용자 결정) — AR(ctx_state) 학습용.

step 마다 (문맥 C 장, 예측 K 장) 을 고르고, 창 W = C + K 장을 **fps 4 로, 패딩 없이** 채울 수 있는 데이터셋·영상에서만 뽑는다.

  * 시간 척도 통일: 모든 데이터를 fps 4 (fstp = ceil(원본 fps) // 4, 최소 1) 의 **정확한 정수 간격**으로 뽑는다.
    (VideoDataset 의 linspace 방식과 달리 간격이 흔들리지 않는다.) 블록 = 2 장 = 0.5 s 근처
  * 패딩 금지: 원본에서 그 간격으로 W 장이 실제로 나오는 영상만 쓴다 (`build_fps4_table.py` 의 n4 >= W).
    AR 에서 패딩 (마지막 프레임 반복) 은 예측 구간을 정지 화면으로 만들어 "복사" 가 정답이 된다
  * 데이터셋 선택: W 를 채울 수 있는 영상이 `min_eligible` 편 이상인 데이터셋 중에서 weight 비율로 하나 (batch 단위).
    SSv2 는 짧아서 (중앙 45 장 @ 12 fps) 작은 창에만, 긴 창은 K710 · HowTo 가 맡는다
  * 토큰 예산 batch: rank 당 B = clamp(token_budget // W, b_min, b_max). 짧은 창은 많이, 긴 창은 적게
  * rank 동기화: (C, K, 데이터셋, B) 는 (seed, step) 만으로 정해진다 → 모든 rank·worker 가 같은 값.
    영상 선택은 (seed, step, rank) 로 rank 마다 다르다
  * worker w 는 step start + w, start + w + nw, … 를 만든다. DataLoader(batch_size=None) 는 worker 를 돌아가며
    하나씩 꺼내므로 step 순서가 그대로 유지된다 (이어 돌리기는 start_step 으로)

yield: (clips uint8 (B, 3, W, H, W), info dict) — info: step, C, K (장), context_blocks, predict_blocks, dataset, B, W

출처: origin/vast `app/vjepa_frozen/var_window.py` (b71df3e). 2026-10-16 에 vll7 로 그대로 가져왔다 (로직 무수정).
쓰는 곳: `data.var_window` 가 있으면 `app/vjepa_frozen/train.py` 가 `build_dataset`/`build_loader` 대신 이것을 쓴다.
  * 고정 창 샘플러 (기존 `build_dataset` + `MaskSampler`) 와의 차이 — 고정 창은 `data.n_frames` 장을 **linspace** 로 뽑고
    (fps 가 영상마다 흔들린다) `mask.prefix_window` 의 (C, K) 목록에서 조합을 고른다. 창 길이가 늘 같아 batch 도 고정이다.
    가변 창은 창 길이마다 batch 를 바꾸고 (토큰 예산), 간격을 fps 4 정수로 고정하고, 패딩을 금지한다.
  * 표는 `z_training/data/build_fps4_table.py` 가 만든다.
"""
from __future__ import annotations

import bisect
import itertools
import logging
import random
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class _Table:
    """build_fps4_table.py 의 tsv: path, fps, n_frames, fstp, n4. n4 오름차순으로 정렬해 둔다 (적격 영상 = 꼬리)."""

    def __init__(self, name: str, path: str, weight: float, min_eligible: int = None):
        self.min_eligible = None if min_eligible is None else int(min_eligible)   # 2026-10-15 (vast): 데이터셋별 덮어쓰기 (없으면 전역)
        rows = []
        for ln in open(path, encoding="utf-8"):
            p, f, n, fstp, n4 = ln.rstrip("\n").split("\t")
            if int(n4) > 0:
                rows.append((int(n4), p, int(n), int(fstp)))
        rows.sort(key=lambda r: r[0])
        self.name, self.weight = name, float(weight)
        self.n4 = [r[0] for r in rows]
        self.paths = [r[1] for r in rows]
        self.nframes = [r[2] for r in rows]
        self.fstp = [r[3] for r in rows]

    def first_ok(self, W: int) -> int:
        """n4 >= W 인 첫 위치. 적격 = [first_ok, len)."""
        return bisect.bisect_left(self.n4, W)

    def n_ok(self, W: int) -> int:
        return len(self.n4) - self.first_ok(W)


class VarWindowBatches(torch.utils.data.IterableDataset):
    def __init__(self, cfg: Dict, transform, tubelet: int, seed: int, rank: int, start_step: int = 0):
        super().__init__()
        self.tables = [_Table(d["name"], d["table"], d.get("weight", 1.0), d.get("min_eligible")) for d in cfg["datasets"]]
        self.C_list = [int(c) for c in cfg["context_frames"]]
        self.K_list = [int(k) for k in cfg["predict_frames"]]
        for v in self.C_list + self.K_list:
            if v <= 0 or v % tubelet:
                raise ValueError(f"var_window: 문맥/예측 장 수 {v} 는 tubelet({tubelet}) 의 배수여야 한다")
        self.budget = int(cfg["token_budget"])          # rank 당 B × W (장) 상한
        self.b_min, self.b_max = int(cfg.get("b_min", 4)), int(cfg.get("b_max", 512))
        self.min_eligible = int(cfg.get("min_eligible", 1000))
        self.tubelet, self.seed, self.rank = tubelet, int(seed), int(rank)
        self.start_step = int(start_step)
        self.transform = transform
        self.max_retry = int(cfg.get("max_retry", 20))
        combos = [(C, K) for C in self.C_list for K in self.K_list]
        self.combos = [(C, K) for C, K in combos if self._eligible(C + K)]
        dropped = sorted(set(combos) - set(self.combos))
        if dropped:
            logger.warning(f"var_window: 어떤 데이터셋도 채우지 못하는 조합은 뺐다: {dropped}")
        if not self.combos:
            raise ValueError("var_window: 쓸 수 있는 (C, K) 조합이 없다")

    # ---------------------------------------------------------------- 계획 (rank 공통)
    def _eligible(self, W: int) -> List[_Table]:
        # 데이터셋별 min_eligible 이 있으면 그것, 없으면 전역 (vast 73ae468 이식, 2026-10-17).
        # ⚠️ 이 줄이 없던 판 (b71df3e) 에서는 데이터셋 dict 의 min_eligible 이 **조용히 무시**됐다 —
        #    ctxt_video_tgtvideo run (10-17) 이 그 상태로 돌아, ssv2 가 적격 2,820 편뿐인 W=24 칸까지 들어갔다
        #    (의도 76.7 : 23.3 -> 실측 70.0 : 30.0).
        return [t for t in self.tables if t.n_ok(W) >= (self.min_eligible if t.min_eligible is None else t.min_eligible)]

    def plan(self, step: int):
        rng = random.Random(f"{self.seed}-plan-{step}")
        C, K = rng.choice(self.combos)
        W = C + K
        el = self._eligible(W)
        t = rng.choices(el, weights=[x.weight for x in el], k=1)[0]
        B = max(self.b_min, min(self.b_max, self.budget // W))
        return C, K, W, t, B

    def describe(self) -> str:
        lines = []
        for C, K in self.combos:
            W = C + K
            el = self._eligible(W)
            B = max(self.b_min, min(self.b_max, self.budget // W))
            lines.append(f"C{C}K{K} (W {W}, B {B}): " + ", ".join(f"{t.name} {t.n_ok(W):,}" for t in el))
        return "\n".join(lines)

    # ---------------------------------------------------------------- 읽기
    def _read(self, t: _Table, j: int, W: int, rng: random.Random):
        import decord
        vr = decord.VideoReader(t.paths[j], num_threads=1)
        n, fstp = len(vr), t.fstp[j]
        span = (W - 1) * fstp + 1
        if n < span:                                   # 표와 실제 길이가 다르면 (헤더 오류) 버린다
            raise ValueError(f"짧다: {t.paths[j]} n={n} span={span}")
        s = rng.randint(0, n - span)
        idx = list(range(s, s + span, fstp))
        return vr.get_batch(idx).asnumpy()            # (W, H, W, 3) uint8

    def make(self, step: int):
        C, K, W, t, B = self.plan(step)
        rng = random.Random(f"{self.seed}-data-{step}-{self.rank}")
        lo = t.first_ok(W)
        clips, fails = [], 0
        while len(clips) < B:
            j = rng.randrange(lo, len(t.paths))
            try:
                clips.append(self.transform(self._read(t, j, W, rng)))
            except Exception as e:
                fails += 1
                if fails > self.max_retry * B:
                    raise RuntimeError(f"var_window: 읽기 실패가 너무 많다 ({t.name}, W={W}): {e}")
        x = torch.stack(clips)                         # (B, 3, W, H, W) uint8
        info = {"step": step, "C": C, "K": K, "W": W, "B": B, "dataset": t.name,
                "context_blocks": C // self.tubelet, "predict_blocks": K // self.tubelet,
                "gap_blocks": 0, "win_blocks": W // self.tubelet, "context_frames": C,
                "type": f"var C{C}K{K} {t.name} B{B}", "read_fails": fails}
        return x, info

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi is not None else (0, 1)
        for step in itertools.count(self.start_step + wid, nw):
            yield self.make(step)


def build_var_window_loader(cfg: Dict, transform, tubelet: int, seed: int, rank: int, start_step: int,
                            num_workers: int, pin_mem: bool, prefetch_factor: int, timeout_s: float):
    ds = VarWindowBatches(cfg, transform, tubelet, seed, rank, start_step)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=None, num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=False, timeout=float(timeout_s) if num_workers > 0 else 0,
        **({"prefetch_factor": int(prefetch_factor)} if num_workers > 0 else {}))
    return ds, loader
