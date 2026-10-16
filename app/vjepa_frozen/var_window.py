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
"""
from __future__ import annotations

import bisect
import itertools
import logging
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class _Table:
    """build_fps4_table.py 의 tsv: path, fps, n_frames, fstp, n4. n4 오름차순으로 정렬해 둔다 (적격 영상 = 꼬리)."""

    def __init__(self, name: str, path: str, weight: float, min_eligible: int = None):
        self.min_eligible = None if min_eligible is None else int(min_eligible)   # 2026-10-15: 데이터셋별 덮어쓰기 (없으면 전역)
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
        # mode=multistart (2026-10-14, 사용자 결정): (C, K) 대신 데이터셋을 먼저 고르고 (가중치 = 프레임 비율, step 마다 토큰 수가 같으므로)
        # 그 데이터셋의 창 후보에서 W 를 고른다. rollout 시작점은 [min_ctx, nb-1] 을 n 등분해 구간마다 하나 (층화)
        self.mode = str(cfg.get("mode", "ck"))
        if self.mode == "multistart":
            self._init_multistart(cfg, tubelet, seed, rank, start_step, transform)
            return
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
        self._init_fixed(cfg.get("fixed"))
        self._init_boundary(cfg.get("boundary"))
        combos = [(C, K) for C in self.C_list for K in self.K_list]
        self.combos = [(C, K) for C, K in combos if self._eligible(C + K)]
        dropped = sorted(set(combos) - set(self.combos))
        if dropped:
            logger.warning(f"var_window: 어떤 데이터셋도 채우지 못하는 조합은 뺐다: {dropped}")
        if not self.combos:
            raise ValueError("var_window: 쓸 수 있는 (C, K) 조합이 없다")

    # ---------------------------------------------------------------- 고정 창 PNG 소스 (2026-10-15, VOBench 학습 split)
    def _init_fixed(self, fx):
        """fixed: {p: step 비율, sources: [{name, list: csv (file_name 열), frames_root, frames_pattern, start, stride, n_frames, context_frames}]}.
        step 마다 별도 난수 (seed-fixed-step) 로 p 확률만큼 고정 창 step 이 된다 — 나머지 step 의 계획 · 데이터는 fixed 없을 때와 같다.
        고정 창은 평가 (cvpr testbed c16t32) 와 같은 프레임 (start, start+stride, …) · 같은 문맥 경계 — late 가림이 경계에 걸린다."""
        self.fixed_p, self.fixed_items = 0.0, []
        if not fx:
            return
        self.fixed_p = float(fx["p"])
        self.fixed_spec = None
        for src in fx["sources"]:
            cf = src["context_frames"]
            cf = tuple(int(v) for v in (cf if isinstance(cf, (list, tuple)) else [cf]))   # 2026-10-15: 문맥 길이 후보 (step 마다 무작위)
            spec = (int(src.get("start", 0)), int(src["stride"]), int(src["n_frames"]), cf)
            if self.fixed_spec not in (None, spec):
                raise ValueError("fixed: 소스들의 창 (start, stride, n_frames, context_frames) 이 같아야 한다")
            self.fixed_spec = spec
            rows = list(__import__("csv").DictReader(open(src["list"])))
            pat = src.get("frames_pattern", "{file_name}/{frame:06d}.png")
            self.fixed_items += [(src["name"], src["frames_root"], pat, r["file_name"]) for r in rows]
        st, sd, n, cs = self.fixed_spec
        if n % self.tubelet or any(c % self.tubelet or not 0 < c < n for c in cs):
            raise ValueError("fixed: n_frames · context_frames 는 tubelet 의 배수, 0 < 문맥 < n_frames")
        logger.info(f"var_window fixed: p {self.fixed_p}, clip {len(self.fixed_items)} 개, 프레임 {st}+{sd}k (k<{n}), 문맥 후보 {list(cs)} 장")

    def _is_fixed(self, step):
        return self.fixed_p > 0 and random.Random(f"{self.seed}-fixed-{step}").random() < self.fixed_p

    def make_fixed(self, step):
        st, sd, n, cs = self.fixed_spec
        c = random.Random(f"{self.seed}-fixedplan-{step}").choice(cs)     # 모든 rank 같은 문맥 길이
        W = n
        B = max(self.b_min, min(self.b_max, self.budget // W))
        rng = random.Random(f"{self.seed}-fixeddata-{step}-{self.rank}")
        from PIL import Image
        clips = []
        for _ in range(B):
            name, root, pat, fn = self.fixed_items[rng.randrange(len(self.fixed_items))]
            fr = [np.asarray(Image.open(os.path.join(root, pat.format(file_name=fn, frame=st + sd * k))).convert("RGB")) for k in range(n)]
            clips.append(self.transform(np.stack(fr)))
        x = torch.stack(clips)
        names = sorted({it[0] for it in self.fixed_items})
        info = {"step": step, "C": c, "K": n - c, "W": W, "B": B, "dataset": "+".join(names),
                "context_blocks": c // self.tubelet, "predict_blocks": (n - c) // self.tubelet,
                "gap_blocks": 0, "win_blocks": W // self.tubelet, "context_frames": c,
                "type": f"var C{c}K{n - c} vob B{B}", "read_fails": 0}
        return x, info

    # ---------------------------------------------------------------- 경계 정렬 mp4 소스 (2026-10-16, object_permanence_320)
    def _init_boundary(self, bx):
        """boundary: {p, manifest (tsv: path, n_frames, boundary_frame), stride, context_frames [..], predict_frames,
        crop_ratio [lo, hi], hflip_p, name}. 입력 · 타깃을 이어 붙인 영상에서 **예측 첫 장 = boundary_frame** 이 되도록 뽑는다:
        문맥 b − stride·C … b − stride, 예측 b … b + stride·(K−1). rollout 이 항상 경계에서 시작한다.
        공간: 높이 전체 · 폭 = 높이 × r (r ~ log-uniform crop_ratio) 을 가로 무작위 위치에서 잘라 정사각으로 리사이즈
        (화면비 왜곡 ≤ hi — 기본 aug 의 ratio 상한 1.35 안). 좌우 반전 hflip_p.
        step 판정 · 문맥 길이는 (seed, step) 전용 난수 — 이 키가 없으면 기존과 완전히 같다."""
        self.bnd_p, self.bnd_items = 0.0, []
        if not bx:
            return
        import csv as _csv
        self.bnd_p = float(bx["p"])
        self.bnd_stride = int(bx["stride"])
        self.bnd_K = int(bx["predict_frames"])
        self.bnd_ratio = tuple(float(v) for v in bx.get("crop_ratio", [1.0, 1.3333]))
        self.bnd_hflip = float(bx.get("hflip_p", 0.5))
        self.bnd_name = str(bx.get("name", "boundary"))
        C = [int(c) for c in bx["context_frames"]]
        for v in C + [self.bnd_K]:
            if v <= 0 or v % self.tubelet:
                raise ValueError(f"boundary: 문맥/예측 장 수 {v} 는 tubelet({self.tubelet}) 의 배수여야 한다")
        s = self.bnd_stride
        for r in _csv.DictReader(open(bx["manifest"]), delimiter="\t"):
            b, n = int(r["boundary_frame"]), int(r["n_frames"])
            if n - b >= s * (self.bnd_K - 1) + 1:             # 예측 K 장이 실제로 나오는 영상만 (패딩 금지)
                self.bnd_items.append((b // s, r["path"], b))   # (경계 앞 최대 문맥 장 수, 경로, 경계)
        self.bnd_items.sort(key=lambda it: it[0])
        self.bnd_cmax = [it[0] for it in self.bnd_items]
        self.bnd_C = [c for c in C if self._bnd_n_ok(c) > 0]
        logger.info(f"var_window boundary: p {self.bnd_p}, clip {len(self.bnd_items)} 개, stride {s}, 문맥 후보 "
                    + ", ".join(f"C{c}:{self._bnd_n_ok(c):,}" for c in self.bnd_C) + f", 예측 {self.bnd_K} 장, crop 비 {self.bnd_ratio}")

    def _bnd_n_ok(self, c):
        return len(self.bnd_cmax) - bisect.bisect_left(self.bnd_cmax, c)

    def _is_boundary(self, step):
        return self.bnd_p > 0 and random.Random(f"{self.seed}-bnd-{step}").random() < self.bnd_p

    def _read_boundary(self, path, b, c, rng):
        import math
        import decord
        import torch.nn.functional as F
        K, s, r = self.bnd_K, self.bnd_stride, self.transform.resolution
        vr = decord.VideoReader(path, num_threads=1)
        idx = [b - s * (c - k) for k in range(c)] + [b + s * k for k in range(K)]
        if idx[0] < 0 or idx[-1] >= len(vr):
            raise ValueError(f"창이 영상 밖: {path} b={b} n={len(vr)}")
        fr = torch.from_numpy(vr.get_batch(idx).asnumpy())                      # (W, H, Wd, 3) uint8
        H, Wd = fr.shape[1], fr.shape[2]
        lo, hi = math.log(self.bnd_ratio[0]), math.log(self.bnd_ratio[1])
        cw = min(Wd, int(round(H * math.exp(rng.uniform(lo, hi)))))
        left = rng.randint(0, Wd - cw)
        x = fr[:, :, left:left + cw].permute(0, 3, 1, 2).float()                 # (W, 3, H, cw)
        x = F.interpolate(x, size=(r, r), mode="bilinear", align_corners=False).transpose(0, 1)   # (3, W, r, r)
        if rng.random() < self.bnd_hflip:
            x = x.flip(-1)
        return x.round_().clamp_(0, 255).to(torch.uint8).contiguous()

    def make_boundary(self, step):
        c = random.Random(f"{self.seed}-bndplan-{step}").choice(self.bnd_C)     # 모든 rank 같은 문맥 길이
        K = self.bnd_K
        W = c + K
        B = max(self.b_min, min(self.b_max, self.budget // W))
        rng = random.Random(f"{self.seed}-bnddata-{step}-{self.rank}")
        lo = bisect.bisect_left(self.bnd_cmax, c)
        clips, fails = [], 0
        while len(clips) < B:
            _, path, b = self.bnd_items[rng.randrange(lo, len(self.bnd_items))]
            try:
                clips.append(self._read_boundary(path, b, c, rng))
            except Exception as e:
                fails += 1
                if fails > self.max_retry * B:
                    raise RuntimeError(f"var_window boundary: 읽기 실패가 너무 많다: {e}")
        x = torch.stack(clips)                                                   # (B, 3, W, r, r) uint8
        info = {"step": step, "C": c, "K": K, "W": W, "B": B, "dataset": self.bnd_name,
                "context_blocks": c // self.tubelet, "predict_blocks": K // self.tubelet,
                "gap_blocks": 0, "win_blocks": W // self.tubelet, "context_frames": c,
                "type": f"var C{c}K{K} {self.bnd_name} B{B}", "read_fails": fails}
        return x, info

    # ---------------------------------------------------------------- multistart (2026-10-14)
    def _init_multistart(self, cfg, tubelet, seed, rank, start_step, transform):
        self.budget = int(cfg["token_budget"])
        self.b_min, self.b_max = int(cfg.get("b_min", 4)), int(cfg.get("b_max", 512))
        self.min_eligible = int(cfg.get("min_eligible", 1000))
        self.tubelet, self.seed, self.rank = tubelet, int(seed), int(rank)
        self.start_step, self.transform = int(start_step), transform
        self.max_retry = int(cfg.get("max_retry", 20))
        st = cfg.get("starts") or {}
        self.n_starts = int(st.get("n", 4))
        self.min_ctx = int(st.get("min_ctx_blocks", 2))
        self.max_rollout = int(st.get("max_rollout", 16))
        self.windows = {}
        for d, t in zip(cfg["datasets"], self.tables):
            ws = [int(w) for w in d["windows"]]
            for w in ws:
                if w % tubelet or w // tubelet < self.min_ctx + 1:
                    raise ValueError(f"multistart: {t.name} 창 {w} 는 tubelet 배수이고 {self.min_ctx + 1} 블록 이상이어야 한다")
            ok = [w for w in ws if t.n_ok(w) >= self.min_eligible]
            if not ok:
                raise ValueError(f"multistart: {t.name} 에서 채울 수 있는 창이 없다 ({ws})")
            dropped = sorted(set(ws) - set(ok))
            if dropped:
                logger.warning(f"multistart: {t.name} 창 {dropped} 는 적격 영상 < {self.min_eligible} 라 뺐다")
            self.windows[t.name] = ok
        self.combos = [(t.name, w) for t in self.tables for w in self.windows[t.name]]

    def _starts(self, nb, rng):
        pos = list(range(self.min_ctx, nb))                      # 시작점 c = rollout 전 관측 블록 수
        n = min(self.n_starts, len(pos))
        chunks = np.array_split(np.array(pos), n)
        return [int(rng.choice(list(c))) for c in chunks]

    def plan_ms(self, step: int):
        rng = random.Random(f"{self.seed}-plan-{step}")
        t = rng.choices(self.tables, weights=[x.weight for x in self.tables], k=1)[0]
        W = rng.choice(self.windows[t.name])
        nb = W // self.tubelet
        starts = self._starts(nb, rng)
        B = max(self.b_min, min(self.b_max, self.budget // W))
        return t, W, B, starts

    def make_ms(self, step: int):
        t, W, B, starts = self.plan_ms(step)
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
        nb = W // self.tubelet
        rl = [min(self.max_rollout, nb - c) for c in starts]
        info = {"step": step, "W": W, "B": B, "dataset": t.name, "win_blocks": nb, "starts": starts, "rollouts": rl,
                "context_frames": starts[0] * self.tubelet, "gap_blocks": 0,
                "type": f"ms W{W} {t.name} B{B} s{'-'.join(map(str, starts))}", "read_fails": fails}
        return torch.stack(clips), info

    # ---------------------------------------------------------------- 계획 (rank 공통)
    def _eligible(self, W: int) -> List[_Table]:
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
        if self.mode == "multistart":
            return "\n".join(f"{n} W{w} (B {max(self.b_min, min(self.b_max, self.budget // w))}): 적격 {next(t for t in self.tables if t.name == n).n_ok(w):,}"
                             for n, w in self.combos)
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
            if self.mode == "multistart":
                yield self.make_ms(step)
            elif self._is_fixed(step):
                yield self.make_fixed(step)
            elif self._is_boundary(step):
                yield self.make_boundary(step)
            else:
                yield self.make(step)


def build_var_window_loader(cfg: Dict, transform, tubelet: int, seed: int, rank: int, start_step: int,
                            num_workers: int, pin_mem: bool, prefetch_factor: int, timeout_s: float):
    ds = VarWindowBatches(cfg, transform, tubelet, seed, rank, start_step)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=None, num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=False, timeout=float(timeout_s) if num_workers > 0 else 0,
        **({"prefetch_factor": int(prefetch_factor)} if num_workers > 0 else {}))
    return ds, loader
