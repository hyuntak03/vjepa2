#!/usr/bin/env python3
"""[X1] 이식 실패 조사 공통 도구 (2026-10-30) — 메타 · split v2 · 튜블릿 상태 · 라벨 · 영역 · 판독기 생성.

transfer_gap_{heads,pairdiff,ladder,rescore}.py 가 import 한다. 단독 실행 없음.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
TUBE = Path("/local_datasets/world/world_analysis/cache/vobench_v1_vith_tubeprobe")
TG = Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap")
RES = REPO / "z_analysis/transfer_gap/results/q1q2"
HEADS = TG / "heads"
MAPS = TG / "maps"
SPLIT = REPO / "data_csv/vobench_v1_split_v2"
NT, NF, S, D, GRID, CELL = 16, 8, 256, 1280, 16, 32.0
OPT = dict(epochs=35, batch_size=16, lr=1e-4, weight_decay=0.05)


def meta():
    return json.load(open(TUBE / "meta.json"))["meta"]


def train_blocks():
    return {r["block_id"] for r in csv.DictReader((SPLIT / "index_train.csv").open())}


def tub_state(samples):
    s = samples.replace("|", "")
    out = []
    for t in range(NT):
        a, b = s[2 * t], s[2 * t + 1]
        out.append("vis" if a == b == "." else "occ" if a == b == "H" else "mix")
    return out


def label_of(m):
    return m["obj_a"] if m["variant"] == "pos_a" else m["obj_b"]


def classes(M, objset):
    labs = sorted({label_of(m) for m in M if m["objset"] == objset})
    return labs, {l: j for j, l in enumerate(labs)}


def load_tracks():
    tr = np.load(TG / "tracks.npz")
    return {b: i for i, b in enumerate(tr["block_id"])}, tr["uv"], tr["hidden"], tr["ext"]


def cell_of(uv):
    c = int(np.clip(uv[0] // CELL, 0, GRID - 1)); r = int(np.clip(uv[1] // CELL, 0, GRID - 1))
    return r, c


def region(uv, ext=None, pad=1):
    """물체 중심 uv (px@512) 의 3×3 칸 — ext (반폭 px) 가 주어지면 외곽 상자 + pad 칸 (최소 3×3)."""
    r, c = cell_of(uv)
    r0, r1, c0, c1 = r - 1, r + 1, c - 1, c + 1
    if ext is not None:
        c0 = min(c0, int((uv[0] - ext[0]) // CELL) - pad); c1 = max(c1, int((uv[0] + ext[0]) // CELL) + pad)
        r0 = min(r0, int((uv[1] - ext[1]) // CELL) - pad); r1 = max(r1, int((uv[1] + ext[1]) // CELL) + pad)
    r0, c0 = max(r0, 0), max(c0, 0); r1, c1 = min(r1, GRID - 1), min(c1, GRID - 1)
    return [rr * GRID + cc for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)]


def last_visible_ctx(hid):
    vis = [t for t in range(NF) if not hid[t]]
    return vis[-1] if vis else 0


def make_head(n_cls, dev):
    import sys
    sys.path.insert(0, str(REPO))
    from src.models.attentive_pooler import AttentiveClassifier
    return AttentiveClassifier(embed_dim=D, num_heads=16, mlp_ratio=4.0, depth=1, num_classes=n_cls).to(dev)


def bootstrap_mean(vals, groups, n=2000, seed=0):
    """block 단위 bootstrap — vals 는 표본별 값, groups 는 표본의 block. 평균 · 95 % CI."""
    vals = np.asarray(vals, float); groups = np.asarray(groups)
    ub = np.unique(groups)
    if len(ub) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0, n_block=0)
    idx = {b: np.where(groups == b)[0] for b in ub}
    rng = np.random.default_rng(seed)
    sums = np.array([vals[idx[b]].sum() for b in ub]); cnts = np.array([len(idx[b]) for b in ub])
    bs = []
    for _ in range(n):
        s = rng.integers(0, len(ub), len(ub))
        bs.append(sums[s].sum() / max(cnts[s].sum(), 1))
    return dict(mean=float(vals.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(len(vals)), n_block=int(len(ub)))
