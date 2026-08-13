"""world_model_analysis 용 데이터셋 로더.

두 config 스키마를 모두 받는다.
  occlusion_v2 : block_column/variant_column/plausible_column/type_column 이 있고
                 surprise 채점은 전체 block, probing 은 probing.block_types 로 걸러진 부분집합.
  probe_set    : 위반 변이가 없어 block 개념이 없고 group_column(is_static) 만 있다.

feature 캐시는 항상 **전체 비디오**에 대해 만든다. probing 은 그 위에서 마스크로 부분집합만 쓴다.
"""
from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    import decord
except ImportError:  # pragma: no cover
    decord = None

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


@dataclass
class Rec:
    video_id: str
    file: str
    group: str                    # 결과 표의 행 구분 (variant 또는 is_static)
    plausible: str = "1"
    block_id: str = ""
    block_type: str = ""
    split: str = "train"
    in_probe: bool = True         # probing 대상인가 (occlusion 의 presence block 제외 등)
    labels: Dict[str, int] = field(default_factory=dict)
    strata: Dict[str, str] = field(default_factory=dict)


class WMADataset:
    def __init__(self, cfg: dict, limit: Optional[int] = None):
        assert decord is not None, "decord 필요"
        d, P = cfg["data"], cfg.get("probing") or {}
        self.root = d["root"]
        self.n_frames = int(d.get("n_frames", 40))
        rows = list(csv.DictReader(open(os.path.join(self.root, d.get("index_csv", "index.csv")))))

        self.block_col = d.get("block_column")
        gcol = d.get("group_column") or d.get("variant_column") or "variant"
        pcol = d.get("plausible_column", "plausible")
        tcol = d.get("type_column")

        if limit and self.block_col:                       # smoke: 앞 N block 만
            keep = sorted({r[self.block_col] for r in rows}, key=lambda v: int(v))[:limit]
            rows = [r for r in rows if r[self.block_col] in set(keep)]
        elif limit:
            # group_column 값마다 고르게 뽑는다. index.csv 가 그룹별로 정렬돼 있으면
            # (probe_set 은 static 512개가 앞에 몰려 있다) 앞에서 그냥 자르는 순간
            # 한 그룹만 남고, fit_groups=["0"] 인 job 이 학습셋 0개로 죽는다.
            by: Dict[str, List[dict]] = defaultdict(list)
            for r in rows:
                by[r[gcol]].append(r)
            per = max(1, (limit * 4) // len(by))
            keep = {x["video_id"] for g in by.values() for x in g[:per]}
            rows = [r for r in rows if r["video_id"] in keep]

        # probing 대상 필터 (예: occlusion 의 presence block 제외)
        want_types = set(P.get("block_types") or [])
        # 라벨 (probing 이 없으면 빈 dict)
        self.targets = P.get("targets", {})
        self.class_index = {t: {c: i for i, c in enumerate(s["classes"])}
                            for t, s in self.targets.items()}

        split_of = self._split(rows, P.get("split") or d.get("split") or {"mode": "none"})
        strata_cols = d.get("strata") or []

        self.records: List[Rec] = []
        for r in rows:
            in_probe = (not want_types) or (tcol and r.get(tcol) in want_types)
            lab = {}
            if in_probe:
                for t, s in self.targets.items():
                    v = r[s["column"]]
                    if v not in self.class_index[t]:
                        raise KeyError(f"{r['video_id']}: {s['column']}={v!r} 가 {t} 클래스 목록에 없다")
                    lab[t] = self.class_index[t][v]
            self.records.append(Rec(
                video_id=r["video_id"], file=r["file"], group=r[gcol],
                plausible=r.get(pcol, "1"),
                block_id=r.get(self.block_col, "") if self.block_col else "",
                block_type=r.get(tcol, "") if tcol else "",
                split=split_of.get(r["video_id"], "train"), in_probe=in_probe, labels=lab,
                strata={c: r[c] for c in strata_cols if c in r}))

        self._probe = np.array([x.in_probe for x in self.records])
        fitv = P.get("fit_variants")
        self._fitable = (np.array([x.group in set(fitv) for x in self.records])
                         if fitv else np.ones(len(self.records), bool))

        # eval_variants — 평가에 쓸 변이. 생략하면 전부 (probe_set 은 생략한다).
        #
        # occlusion 에서 이게 필요한 이유: 불가능 변이를 평가에 넣으면 세 표현 중
        # 둘은 중복이고 하나는 구조적 오답이라 overall 이 통째로 오염된다.
        #   contextF  ctx_masked 는 미래 토큰을 transformer 이전에 떨궈 1~32 만 본다.
        #             imp_ab 는 pos_a 와 context 픽셀 차이가 0 이므로 표현이 비트 단위로
        #             같다 -> 같은 텐서를 평가셋에 두 번 넣는 것.
        #   pred      predictor 입력은 context 뿐이라 위와 같은 이유로 imp_ab == pos_a.
        #   targetF   imp_ab 는 shape_before=star 인데 실제 렌더는 shape_after=cross 다.
        #             라벨이 before 이므로 인코더가 정확할수록 0 점이 된다.
        # 불가능 변이를 다시 보고 싶으면 이 키를 지우면 원래대로 돌아온다.
        evalv = P.get("eval_variants")
        self._evalable = (np.array([x.group in set(evalv) for x in self.records])
                          if evalv else np.ones(len(self.records), bool))

        # per_group 표에 찍을 그룹. 평가에 안 들어가는 변이는 n=0 행만 만들므로 뺀다.
        self.groups = sorted({x.group for x in self.records if x.in_probe and
                              (not evalv or x.group in set(evalv))}) or \
                      sorted({x.group for x in self.records})

    # ---- split ---------------------------------------------------------------
    @staticmethod
    def _split(rows, cfg) -> Dict[str, str]:
        if cfg.get("mode") != "ratio":
            return {r["video_id"]: r.get("split", "train") for r in rows}
        gb, frac, seed = cfg.get("group_by", "block_id"), float(cfg.get("train_frac", .5)), int(cfg.get("seed", 0))
        strat = cfg.get("stratify_by")
        gs = {}
        for r in rows:
            k = r[strat] if strat else "_all"
            if gs.setdefault(r[gb], k) != k:
                raise ValueError(f"stratify_by={strat!r} 가 {gb}={r[gb]!r} 안에서 일정하지 않다")
        train = set()
        for k in sorted(set(gs.values())):
            g = sorted([b for b, v in gs.items() if v == k])
            random.Random(seed).shuffle(g)
            train |= set(g[: int(round(len(g) * frac))])
        return {r["video_id"]: ("train" if r[gb] in train else "val") for r in rows}

    # ---- 접근 ----------------------------------------------------------------
    def __len__(self):
        return len(self.records)

    def clip(self, i) -> torch.Tensor:
        vr = decord.VideoReader(os.path.join(self.root, self.records[i].file), num_threads=1)
        a = torch.from_numpy(vr.get_batch(list(range(self.n_frames))).asnumpy())
        return ((a.permute(3, 0, 1, 2).float() / 255.0) - MEAN) / STD

    def labels(self, t) -> torch.Tensor:
        return torch.tensor([x.labels.get(t, 0) for x in self.records], dtype=torch.long)

    def group_mask(self, groups: Sequence[str]) -> np.ndarray:
        g = set(groups)
        return np.array([x.group in g for x in self.records])

    def split_masks(self):
        """probing 용 (train, val). probing 대상이 아닌 비디오는 양쪽에서 빠진다."""
        tr = np.array([x.split == "train" for x in self.records]) & self._probe & self._fitable
        va = np.array([x.split == "val" for x in self.records]) & self._probe & self._evalable
        return tr, va

    def blocks(self) -> Dict[str, List[Rec]]:
        out = defaultdict(list)
        for x in self.records:
            if x.block_id:
                out[x.block_id].append(x)
        return dict(out)

    def summary(self) -> str:
        from collections import Counter
        s = f"{len(self)} videos"
        if self.block_col:
            b = self.blocks()
            s += f" | {len(b)} block"
            if any(x.block_type for x in self.records):
                s += " " + str(dict(Counter(x.block_type for x in self.records)))
        s += f" | probing 대상 {int(self._probe.sum())}"
        tr, va = self.split_masks()
        s += f" (train {int(tr.sum())} / val {int(va.sum())})"
        s += f" | groups {self.groups}"
        return s
