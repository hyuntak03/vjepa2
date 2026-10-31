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
                rows.append((int(n4), p, int(n), int(fstp), int(f)))
        rows.sort(key=lambda r: r[0])
        self.name, self.weight = name, float(weight)
        self.n4 = [r[0] for r in rows]
        self.paths = [r[1] for r in rows]
        self.nframes = [r[2] for r in rows]
        self.fstp = [r[3] for r in rows]
        self.fps = [r[4] for r in rows]                # 2026-10-27 (vast 245b255 이식): 시간 기준 표 (fstp < 0) 에서 쓴다

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
        # 2026-10-19: 창 길이 상한 (장). 기본 없음 = 종전 동작.
        #   왜: 모든 창을 채우는 데이터셋 (예: phys2d_v1 은 4 fps 원본 64~128 장이라 W=64 까지 100 %) 을 섞으면
        #   자연 영상이 못 채우던 긴 칸이 되살아나 **그 칸이 그 데이터셋 100 %** 가 된다. 상한으로 격자를 고정해
        #   먼저 돈 run (ctxt_video: W <= 40) 과 같은 칸에서 비교한다. 올리려면 SET 으로 끈다.
        self.max_window = cfg.get("max_window")
        self.max_window = None if self.max_window is None else int(self.max_window)
        # 2026-10-22: flow 보조 라벨 (app/vjepa_frozen/flow_labels.py). 키가 없으면 종전과 완전히 같다.
        #   flow: {roots: {<데이터셋 이름>: <캐시 dir>}, grid: 16, patch: 16, require: true}
        #   ⚠️ 켜면 창 시작을 fps 4 격자에 맞춘다 (snap_start) — 캐시가 전역 fps 4 인접 쌍만 담기 때문이다.
        #      시작 위상이 fstp 가지에서 1 가지로 줄어든다. **먼저 돈 run 과 샘플링이 다르다** (한 변수 대조가 아니다).
        self.flow_cfg = cfg.get("flow") or None
        self.snap_start = bool(cfg.get("snap_start", bool(self.flow_cfg)))
        self._fc = None
        # 2026-10-20 (vast f6e7934 이식): 학습 경로에서도 boundary 를 쓴다. 키가 없으면 bnd_p 0 이라 종전과 완전히 같다
        self._init_boundary(cfg.get("boundary"))
        combos = [(C, K) for C in self.C_list for K in self.K_list]
        self.combos = [(C, K) for C, K in combos
                       if self._eligible(C + K) and (self.max_window is None or C + K <= self.max_window)]
        dropped = sorted(set(combos) - set(self.combos))
        if dropped:
            logger.warning(f"var_window: 뺀 조합 (적격 데이터셋 없음 또는 max_window={self.max_window} 초과): {dropped}")
        if not self.combos:
            raise ValueError("var_window: 쓸 수 있는 (C, K) 조합이 없다")

    # ---------------------------------------------------------------- boundary 소스 (vast b888c61 이식, 2026-10-19)
    #   사건 경계에 맞춘 창 (예측 첫 장 = 전환 근처) + `labels: phys2d_sup` 이면 같은 창의 상태 라벨을 함께 낸다.
    #   처음에는 **`z_training/tools/state_head_fit.py` · `state_head_probe.py` 가 쓰려고만** 가져왔다.
    #   ⚠️ 정정 (2026-10-20) — "학습 경로에서는 안 부른다" 는 더 이상 사실이 아니다. vast f6e7934 의 `boundary.mix=sum`
    #      (합성을 step 교대가 아니라 **매 step 보조 손실**로) 을 이식하면서 `__init__` 에서 부르고 `__iter__` 가 분기한다.
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
        # 2026-10-19: labels: phys2d_sup — 같은 창의 상태 라벨 (z_training/data/phys2d_sup_labels.py → state_head.pack_labels) 을 info["labels"] 에.
        #   좌표 라벨이라 자르기 · 화면비 바꾸기를 금지한다 (crop_ratio [1, 1], 정사각 원본). 반전은 영상 · 라벨에 같이
        self.bnd_labels = bx.get("labels")
        if self.bnd_labels not in (None, "phys2d_sup"):
            raise ValueError(f"boundary.labels 는 phys2d_sup 만: {self.bnd_labels!r}")
        C = [int(c) for c in bx["context_frames"]]
        for v in C + [self.bnd_K]:
            if v <= 0 or v % self.tubelet:
                raise ValueError(f"boundary: 문맥/예측 장 수 {v} 는 tubelet({self.tubelet}) 의 배수여야 한다")
        s = self.bnd_stride
        for r in _csv.DictReader(open(bx["manifest"]), delimiter="\t"):
            b, n = int(r["boundary_frame"]), int(r["n_frames"])
            c0 = int(r.get("ctx_start") or 0)                  # 선택 열: 다른 전환이 없는 문맥의 가장 이른 장 (phys2d)
            if n - b >= s * (self.bnd_K - 1) + 1:             # 예측 K 장이 실제로 나오는 영상만 (패딩 금지)
                self.bnd_items.append(((b - c0) // s, r["path"], b))   # (경계 앞 최대 문맥 장 수, 경로, 경계)
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
        flip = rng.random() < self.bnd_hflip
        if flip:
            x = x.flip(-1)
        x = x.round_().clamp_(0, 255).to(torch.uint8).contiguous()
        if not self.bnd_labels:
            return x
        if cw != Wd or H != Wd:
            raise ValueError(f"boundary labels: 자르기 · 비정사각 금지 (H {H} W {Wd} crop {cw}) — crop_ratio [1, 1] 로")
        from app.vjepa_frozen.state_head import pack_labels
        from z_training.data.phys2d_sup_labels import load_clip, window_labels
        lab = window_labels(load_clip(path), start=idx[0], n_frames=len(idx), stride=s, tubelet=self.tubelet, reduce="last")
        return x, {k: torch.from_numpy(v) for k, v in pack_labels(lab, flip=flip).items()}

    def make_boundary(self, step):
        c = random.Random(f"{self.seed}-bndplan-{step}").choice(self.bnd_C)     # 모든 rank 같은 문맥 길이
        K = self.bnd_K
        W = c + K
        B = max(self.b_min, min(self.b_max, self.budget // W))
        rng = random.Random(f"{self.seed}-bnddata-{step}-{self.rank}")
        lo = bisect.bisect_left(self.bnd_cmax, c)
        clips, labs, fails = [], [], 0
        while len(clips) < B:
            _, path, b = self.bnd_items[rng.randrange(lo, len(self.bnd_items))]
            try:
                got = self._read_boundary(path, b, c, rng)
                if self.bnd_labels:
                    got, lab = got
                    labs.append(lab)
                clips.append(got)
            except Exception as e:
                fails += 1
                if fails > self.max_retry * B:
                    raise RuntimeError(f"var_window boundary: 읽기 실패가 너무 많다: {e}")
        x = torch.stack(clips)                                                   # (B, 3, W, r, r) uint8
        info = {"step": step, "C": c, "K": K, "W": W, "B": B, "dataset": self.bnd_name,
                "context_blocks": c // self.tubelet, "predict_blocks": K // self.tubelet,
                "gap_blocks": 0, "win_blocks": W // self.tubelet, "context_frames": c,
                "type": f"var C{c}K{K} {self.bnd_name} B{B}", "read_fails": fails}
        if labs:
            info["labels"] = {k: torch.stack([l[k] for l in labs]) for k in labs[0]}       # (B, 블록, 48, …)
        return x, info

    # ---------------------------------------------------------------- multistart (2026-10-14)

    # ---------------------------------------------------------------- flow 라벨 (2026-10-22)
    def _flow_cache(self, name: str):
        """memmap 은 fork 뒤에 열어야 안전하다 — worker 안에서 처음 쓸 때 연다."""
        if self._fc is None:
            self._fc = {}
        if name not in self._fc:
            from app.vjepa_frozen.flow_labels import FlowCache
            root = (self.flow_cfg.get("roots") or {}).get(name)       # str 또는 shard 목록
            self._fc[name] = FlowCache(root) if root else False
        return self._fc[name]

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
        if fstp < 0:                                   # 2026-10-27 시간 기준 표 (build_time_fps_table.py): 장 k = 가장 가까운 원본 프레임 round((t0 + k/T)·fps)
            T_, f_ = -fstp, float(t.fps[j])
            n_t = int((n - 1) / f_ * T_) + 1           # 전역 시간 격자 장 수 (flow 캐시 build_flow_cache.py 와 같은 정의)
            if n_t < W:
                raise ValueError(f"짧다: {t.paths[j]} n={n} 창 {(W - 1) / T_:.2f}s")
            if self.snap_start:                        # flow 캐시와 맞춘다: 시작을 1/T 초 격자 위 장 j0 에.
                j0 = rng.randint(0, n_t - W)           #   ⚠️ 식을 build_flow_cache.read_fps4 와 **똑같이** 쓴다 — (j0/T + k/T)·f 는 .5 경계에서 반올림이 갈린다
                idx = [min(n - 1, int(round((j0 + k) / T_ * f_))) for k in range(W)]
            else:                                      # vast 판과 같다: 연속 시작
                t0 = rng.uniform(0.0, (n_t - W) / T_); j0 = -1
                idx = [min(n - 1, int(round((t0 + k / T_) * f_))) for k in range(W)]
            return vr.get_batch(idx).asnumpy(), j0
        span = (W - 1) * fstp + 1
        if n < span:                                   # 표와 실제 길이가 다르면 (헤더 오류) 버린다
            raise ValueError(f"짧다: {t.paths[j]} n={n} span={span}")
        s = rng.randint(0, n - span)
        if self.snap_start:
            s = (s // fstp) * fstp                     # 전역 fps 4 격자에 맞춘다 (flow 캐시의 쌍 색인과 맞추려고)
        idx = list(range(s, s + span, fstp))
        return vr.get_batch(idx).asnumpy(), s // fstp  # (W, H, W, 3) uint8, fps 4 시작 색인

    def make(self, step: int):
        C, K, W, t, B = self.plan(step)
        rng = random.Random(f"{self.seed}-data-{step}-{self.rank}")
        lo = t.first_ok(W)
        clips, fails = [], 0
        fc = self._flow_cache(t.name) if self.flow_cfg else None
        flows, valids = [], []
        while len(clips) < B:
            j = rng.randrange(lo, len(t.paths))
            try:
                arr, j0 = self._read(t, j, W, rng)
                if fc is None:
                    clips.append(self.transform(arr))
                    continue
                from app.vjepa_frozen import flow_labels as FL
                nb_w = W // self.tubelet
                hz = [int(h) for h in self.flow_cfg.get("horizons", [1, 2, 4, 8])]
                got = fc.span(t.paths[j], j0, W + max(hz) - 1) if fc else None
                if got is None:
                    raise KeyError(f"flow 캐시에 없다: {t.paths[j]} j0={j0}")
                d, ok = FL.block_traj(got[0], got[1], nb_w, self.tubelet, hz)
                clip, prm = self.transform(arr, return_params=True)
                f, v = FL.apply_transform_traj(d, ok, prm["box"], prm["HW"], prm["flip"],
                                               int(self.flow_cfg.get("grid", 16)), self.transform.resolution,
                                               int(self.flow_cfg.get("patch", 16)))
                clips.append(clip); flows.append(f); valids.append(v)
            except Exception as e:
                fails += 1
                if fails > self.max_retry * B:
                    raise RuntimeError(f"var_window: 읽기 실패가 너무 많다 ({t.name}, W={W}): {e}")
        x = torch.stack(clips)                         # (B, 3, W, H, W) uint8
        info = {"step": step, "C": C, "K": K, "W": W, "B": B, "dataset": t.name,
                "context_blocks": C // self.tubelet, "predict_blocks": K // self.tubelet,
                "gap_blocks": 0, "win_blocks": W // self.tubelet, "context_frames": C,
                "type": f"var C{C}K{K} {t.name} B{B}", "read_fails": fails}
        if flows:
            info["flow"] = torch.stack(flows)          # (B, nb, H, 2, g, g) 패치 단위 누적 변위
            info["flow_valid"] = torch.stack(valids)   # (B, nb, H, g, g) [0, 1]
        return x, info

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi is not None else (0, 1)
        for step in itertools.count(self.start_step + wid, nw):
            if self._is_boundary(step):          # 2026-10-20: boundary.p (mix=sum 이면 합성 로더가 p 1.0)
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
