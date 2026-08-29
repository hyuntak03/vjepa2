"""world_model_analysis 용 데이터셋 로더.

두 config 스키마를 모두 받는다.
  occlusion_v2 : block_column/variant_column/plausible_column/type_column 이 있고
                 surprise 채점은 전체 block, probing 은 probing.block_types 로 걸러진 부분집합.
  probe_set    : 위반 변이가 없어 block 개념이 없고 group_column(is_static) 만 있다.

feature 캐시는 항상 **전체 비디오**에 대해 만든다. probing 은 그 위에서 마스크로 부분집합만 쓴다.
"""
from __future__ import annotations

import csv
import logging
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

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

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
    pair_id: str = ""             # block 안에서 문맥이 일치하는 (가능, 불가능) 쌍 (Garrido get_matches)
    split: str = "train"
    in_probe: bool = True         # probing 대상인가 (occlusion 의 presence block 제외 등)
    labels: Dict[str, int] = field(default_factory=dict)
    strata: Dict[str, str] = field(default_factory=dict)
    raw: Dict[str, str] = field(default_factory=dict)   # index.csv 원본 행. PNG 경로 조립에 쓴다


class WMADataset:
    def __init__(self, cfg: dict, limit: Optional[int] = None):
        assert decord is not None, "decord 필요"
        d, P = cfg["data"], cfg.get("probing") or {}
        self.root = d["root"]
        self.n_frames = int(d.get("n_frames", 40))

        # ---- data.resolution ------------------------------------------------------
        # 모델에 넣을 한 변의 크기. 원본이 이 크기가 아니면 clip() 에서 리사이즈한다.
        #
        # 왜 검사까지 하는가: surprise 경로는 context/target 토큰을 **인덱스로** 나눈다
        # (_context_target_indices 가 spatial_tokens = (model.img_size/patch)^2 를 쓴다).
        # 그런데 encoder 는 입력 해상도에 맞춰 토큰 수가 달라지고, predictor 는
        # 빌드 시점 grid(=model.img_size/patch)로 flat index -> (t,h,w) 를 되돌린다.
        # 그래서 실제 영상 크기 != model.img_size 면
        #   영상이 더 작으면  -> 인덱스가 실제 토큰 수를 넘어 CUDA assert 로 죽고
        #   영상이 더 크면    -> 인덱스가 전부 범위 안이라 **에러 없이 엉뚱한 구간**을
        #                        context/target 으로 잘라 쓴다 (조용히 틀린 수치).
        # 실제로 IntPhys1_full 원본은 288x288 이고 model.img_size 는 256 이다.
        self.resolution = int(d["resolution"]) if d.get("resolution") else None
        img_size = (cfg.get("model") or {}).get("img_size")
        if self.resolution and img_size and int(img_size) != self.resolution:
            raise ValueError(
                f"data.resolution({self.resolution}) != model.img_size({int(img_size)}). "
                "두 값은 반드시 같아야 한다 — 토큰 인덱스가 model.img_size 기준으로 만들어지고, "
                "predictor 의 RoPE grid 도 거기서 고정된다.")
        self._resized_note = False

        # ---- 프레임 소스: mp4 디코딩 대신 원본 PNG 를 직접 읽을 수 있다 --------------
        # frames_root 를 주면 PNG 경로로 간다. 인코더는 프레임 단위로 받으므로 컨테이너를
        # 거칠 이유가 없고, 코덱 손실과 yuv420 크로마 서브샘플링을 피할 수 있다.
        # 경로는 index.csv 의 컬럼으로 조립한다. scene/ 만 쓴다 (masks, depth 는 안 읽는다).
        #   frames_root:    /local_datasets/world/world_analysis/IntPhys1_dev_frame_png
        #   frames_pattern: "{block}/{quadruplet}/{run}/scene/scene_{frame:03d}.png"
        #   frames_start:   1        파일 번호가 1부터면 1 (scene_001.png)
        self._frames_root = d.get("frames_root")
        self._frames_pattern = d.get("frames_pattern",
                                     "{block}/{quadruplet}/{run}/scene/scene_{frame:03d}.png")
        self._frames_start = int(d.get("frames_start", 1))
        # frames_stride: 원본에서 몇 프레임마다 하나씩 뽑을지. 1 = 연속(기본, 기존 동작).
        #   예) 100프레임 @16fps 를 stride 3 으로 32장 -> raw 0,3,...,93 (실효 5.33fps)
        self._frames_stride = int(d.get("frames_stride", 1))
        if self._frames_stride < 1:
            raise ValueError(f"data.frames_stride 는 1 이상이어야 한다: {self._frames_stride}")
        if self._frames_root and not os.path.isdir(self._frames_root):
            raise FileNotFoundError(f"data.frames_root 가 없다: {self._frames_root}")
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
                pair_id=r.get(d.get("pair_column", "pair_id"), ""),
                split=split_of.get(r["video_id"], "train"), in_probe=in_probe, labels=lab,
                strata={c: r[c] for c in strata_cols if c in r}, raw=dict(r)))

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

    def _read_frames(self, i) -> torch.Tensor:
        """-> (T, H, W, 3) uint8. mp4 를 디코딩하거나 원본 PNG 를 직접 읽는다.

        data.frames_root 를 주면 PNG 경로로 간다. 인코더는 어차피 프레임 단위로 받으므로
        컨테이너를 거칠 이유가 없고, 코덱 손실·크로마 서브샘플링을 피할 수 있다.
        경로는 index.csv 의 컬럼으로 조립한다 (frames_pattern).
        """
        rec = self.records[i]
        if self._frames_root:
            assert Image is not None, "PNG 프레임을 읽으려면 Pillow 가 필요하다"
            fr = []
            for f in range(self._frames_start,
                           self._frames_start + self.n_frames * self._frames_stride,
                           self._frames_stride):
                p = os.path.join(self._frames_root,
                                 self._frames_pattern.format(frame=f, **rec.raw))
                fr.append(np.asarray(Image.open(p).convert("RGB")))
            return torch.from_numpy(np.stack(fr))
        vr = decord.VideoReader(os.path.join(self.root, rec.file), num_threads=1)
        return torch.from_numpy(vr.get_batch(list(range(self.n_frames))).asnumpy())

    def clip(self, i) -> torch.Tensor:
        a = self._read_frames(i)
        x = a.permute(3, 0, 1, 2).float() / 255.0                # (3, T, H, W), 0..1
        r = self.resolution
        if r and tuple(x.shape[-2:]) != (r, r):
            # 정사각 원본(IntPhys 288x288)이라 shorter-side resize + center crop 과 결과가 같다.
            #
            # antialias=False 인 이유: 공식 Garrido eval 과 V-JEPA 계열 전처리가 전부
            # antialias 없는 bilinear 다.
            #   jepa-intuitive-physics/evaluation_code/src/utils/video/transforms.py:572
            #     torch.nn.functional.interpolate(cropped, size=(h,w),
            #                                     mode='bilinear', align_corners=False)
            # 다운샘플 품질만 보면 antialias=True 가 낫지만, 사전학습·공식평가와 같은
            # 커널을 쓰는 쪽이 분포가 맞다.
            if not self._resized_note:
                self._resized_note = True
                # debug 로 둔다 (기본 INFO 라 안 찍힌다). 리사이즈 여부는 config 의
                # data.resolution 으로 이미 명시돼 있고 summary.json 에도 남는다.
                logging.getLogger("wma").debug(
                    f"[data] {tuple(x.shape[-2:])} -> ({r}, {r}) 리사이즈 "
                    f"(data.resolution, bilinear/antialias off = 공식과 동일)")
            x = torch.nn.functional.interpolate(
                x.transpose(0, 1), size=(r, r), mode="bilinear",
                align_corners=False).transpose(0, 1).contiguous()
        return (x - MEAN) / STD

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
