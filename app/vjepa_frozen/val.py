"""학습 중 in-loop 검증: 고정창 surprise 로 matched-pair 정확도.

evals/world_model_analysis/eval.py:run_surprise + score_blocks(pairing=matched) 의 축약판이다.
같은 토큰 인덱스, 같은 거리(mean|p − LN(h)|), 같은 짝짓기(pair_id 로 가능1+불가능1).
정식 수치는 학습이 끝난 뒤 z_training/eval.sh (= run.sh surprise_c16t32) 로 다시 낸다 —
여기 값은 **수렴 감시용**이고 표본이 작다 (n_blocks_per_type × 조건 수 × 2 쌍).

val 블록 키 (resolve_train.py 가 `dataset:` 이름을 data 로 풀어 준다)
  data:  root / index_csv / frames_root / frames_pattern / frames_start / frames_stride /
         block_column / pair_column / plausible_column / type_column
  n_blocks_per_type : type_column 값마다 몇 block 을 쓸지 (seed 로 고정 추출)
  context_length    : 문맥 프레임 수 (기본 16 = surprise_c16t32)
  batch_size        : rank 당
"""
from __future__ import annotations

import csv
import logging
import os
import random
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist

from app.vjepa_frozen.data import ClipTransform, FramesIndexDataset

logger = logging.getLogger(__name__)


def _select_rows(spec: dict, n_blocks_per_type: int, seed: int) -> List[dict]:
    ipath = os.path.join(spec["root"], spec.get("index_csv", "index.csv"))
    with open(ipath, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bcol = spec.get("block_column", "block_id")
    tcol = spec.get("type_column")
    by_type = defaultdict(dict)                 # type -> block_id -> [rows]
    for r in rows:
        by_type[r.get(tcol, "") if tcol else ""].setdefault(r[bcol], []).append(r)
    chosen = []
    for t in sorted(by_type):
        bids = sorted(by_type[t])
        random.Random(seed).shuffle(bids)
        for b in bids[: n_blocks_per_type]:
            chosen += by_type[t][b]
    return chosen


class ValSet:
    def __init__(self, cfg_val: dict, n_frames: int, resolution: int):
        spec = dict(cfg_val["data"])
        spec["include"] = {}                    # 가능·불가능 전부
        self.spec = spec
        self.ds = FramesIndexDataset(spec, n_frames, ClipTransform(resolution))
        self.ds.rows = _select_rows(spec, int(cfg_val.get("n_blocks_per_type", 8)), int(cfg_val.get("seed", 0)))
        self.bcol = spec.get("block_column", "block_id")
        self.pcol = spec.get("pair_column", "pair_id")
        self.plcol = spec.get("plausible_column", "plausible")
        self.tcol = spec.get("type_column")
        if not self.ds.rows:
            raise ValueError("val: 선택된 행이 0개")
        for k in (self.bcol, self.pcol, self.plcol):
            if k not in self.ds.rows[0]:
                raise KeyError(f"val index 에 {k!r} 컬럼이 없다 (matched pairing 에 필요)")

    def __len__(self):
        return len(self.ds.rows)


@torch.no_grad()
def run_val(vs: ValSet, ctx_enc, tgt_enc, predictor, device, autocast_dtype, rank: int, ws: int,
            tubelet: int, spatial: int, context_length: int, batch_size: int, mask_index: int,
            loss_exp: float, target_ln: bool) -> Dict:
    predictor.eval()
    n_frames = vs.ds.n_frames
    n_ctx = context_length // tubelet * spatial
    N = n_frames // tubelet * spatial
    ci = torch.arange(n_ctx, device=device).unsqueeze(0)
    ti = torch.arange(n_ctx, N, device=device).unsqueeze(0)
    idx = list(range(rank, len(vs), ws))
    out = {}
    t0 = time.time()
    ac = torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype else torch.autocast("cuda", enabled=False)
    for s in range(0, len(idx), batch_size):
        chunk = idx[s: s + batch_size]
        x = torch.stack([vs.ds.transform(vs.ds.read_uint8(i)) for i in chunk]).to(device, non_blocking=True)
        n = len(chunk)
        with ac:
            xe = x.to(next(tgt_enc.parameters()).dtype)
            h = tgt_enc(xe)
            h = torch.gather(h, 1, ti.expand(n, -1).unsqueeze(-1).expand(-1, -1, h.size(-1)))
            if target_ln:
                h = torch.nn.functional.layer_norm(h, (h.size(-1),))
            z = ctx_enc(xe, masks=[ci.expand(n, -1)])
            p = predictor(z, ci.expand(n, -1), ti.expand(n, -1), mask_index=mask_index)
        d = (p.float() - h.float()).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
        for k, i in enumerate(chunk):
            out[vs.ds.rows[i]["video_id"]] = float(d[k])
    # gather
    if dist.is_available() and dist.is_initialized() and ws > 1:
        gathered = [None] * ws
        dist.all_gather_object(gathered, out)
        merged = {}
        for g in gathered:
            merged.update(g)
    else:
        merged = out
    predictor.train()
    if rank != 0:
        return {}
    return _score(vs, merged, time.time() - t0)


def _score(vs: ValSet, sur: Dict[str, float], elapsed: float) -> Dict:
    blocks = defaultdict(list)
    for r in vs.ds.rows:
        blocks[r[vs.bcol]].append(r)
    hits_by_type = defaultdict(list)
    pos_s, neg_s, ties = [], [], 0
    for bid, recs in blocks.items():
        by_pair = defaultdict(list)
        for r in recs:
            by_pair[r[vs.pcol]].append(r)
        for pid, prs in by_pair.items():
            pos = [r for r in prs if r[vs.plcol] == "1"]
            neg = [r for r in prs if r[vs.plcol] != "1"]
            if len(pos) != 1 or len(neg) != 1:
                raise ValueError(f"val block {bid} pair {pid}: 가능 {len(pos)} / 불가능 {len(neg)} (1+1 이어야 한다)")
            a, b = sur[pos[0]["video_id"]], sur[neg[0]["video_id"]]
            hit = 1.0 if a < b else (0.5 if a == b else 0.0)
            ties += int(a == b)
            t = pos[0].get(vs.tcol, "") if vs.tcol else "all"
            hits_by_type[t].append(hit)
            pos_s.append(a)
            neg_s.append(b)
    all_hits = [h for v in hits_by_type.values() for h in v]
    res = {"acc": float(np.mean(all_hits)), "n_pair": len(all_hits), "n_ties": ties,
           "surprise_pos": float(np.mean(pos_s)), "surprise_neg": float(np.mean(neg_s)),
           "margin": float(np.mean(np.array(neg_s) - np.array(pos_s))),
           "by_type": {t: {"acc": float(np.mean(v)), "n_pair": len(v)} for t, v in sorted(hits_by_type.items())},
           "elapsed_s": round(elapsed, 1)}
    return res


def format_val(res: Dict) -> str:
    if not res:
        return ""
    s = (f"val acc {100*res['acc']:.1f}% (n_pair {res['n_pair']}, ties {res['n_ties']}) | "
         f"surprise pos {res['surprise_pos']:.4f} neg {res['surprise_neg']:.4f} margin {res['margin']:+.4f} | "
         f"{res['elapsed_s']}s")
    if len(res["by_type"]) > 1:
        s += "\n      " + "  ".join(f"{t}: {100*v['acc']:.1f}" for t, v in res["by_type"].items())
    return s
