"""무엇을 '성능' 으로 잴 것인가 — 갈아끼울 수 있게 분리한다.

  --metrics surprise_acc              기본. 확립된 채점 정확도
  --metrics surprise_acc,pred_drift   여러 개를 한 번에

새 지표를 붙이는 법: 아래 `@register("이름")` 을 단 클래스를 하나 더 쓴다.
필요한 입력은 클래스 속성으로 선언한다 — run.py 가 그걸 보고 **target 인코더를 돌릴지
말지 정한다** (needs_h=False 인 지표만 요청하면 인코더 하나를 통째로 아낀다).

    needs_h      LN(target_encoder(clip))[미래] 가 필요한가
    needs_clean  knockout 안 한 predictor 출력이 필요한가

⚠️ `surprise_acc` 는 `evals.world_model_analysis.eval.score_blocks` 를 **그대로 재사용**한다.
   채점 정의(matched pairing, 동점 0.5, block 평균)가 본 파이프라인과 갈리지 않게 하려는
   것이다. 여기서 다시 구현하지 말 것.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from evals.world_model_analysis.eval import score_blocks

REGISTRY: dict[str, type] = {}


def register(name):
    def deco(cls):
        cls.metric_name = name
        REGISTRY[name] = cls
        return cls
    return deco


class Metric:
    needs_h = False
    needs_clean = False
    metric_name = "?"

    def __init__(self, ds, cfg):
        self.ds, self.cfg = ds, cfg
        self.btype = {r.video_id: r.block_type for r in ds.records}

    def update(self, vids, p, h, p_clean):
        raise NotImplementedError

    def result(self) -> dict:
        raise NotImplementedError

    def per_video(self) -> dict[str, dict[str, float]]:
        """{필드이름: {video_id: 값}}. run.py 가 npz 로 남겨 merge.py / 그림이 축별로 다시 집계한다."""
        return {}


def _by_type(self, per_video: dict) -> dict:
    d = defaultdict(list)
    for v, x in per_video.items():
        d[self.btype.get(v, "?")].append(x)
    return {k: round(float(np.mean(v)), 4) for k, v in sorted(d.items())}


@register("surprise_acc")
class SurpriseAcc(Metric):
    """확립된 채점 — surprise = mean|p − LN(h)[미래]|, matched pair, chance 50%."""

    needs_h = True

    def __init__(self, ds, cfg):
        super().__init__(ds, cfg)
        self.sup: dict[str, float] = {}

    def update(self, vids, p, h, p_clean):
        # eval.run_surprise 의 _distance(kind='l1', loss_exp=1.0) 과 같은 값이다:
        #   mean(|p − h|^1.0) / 1.0  을 클립마다
        e = (p.float() - h.float()).abs().mean(dim=(1, 2))
        for v, s in zip(vids, e.tolist()):
            self.sup[v] = s

    def result(self):
        res, _ = score_blocks(self.ds, self.sup, self.cfg)
        o = res["overall"]
        out = {"acc": round(100 * o["block_pairwise"], 3), "n_block": o["n_block"],
               "n_pair": o["n_pair"], "n_ties": o["n_ties"],
               "perfect_ratio": round(o["perfect_ratio"], 4), "pairing": res["pairing"]}
        if "by_block_type" in res:
            out["by_block_type"] = {k: round(100 * v["block_pairwise"], 3)
                                    for k, v in res["by_block_type"].items()}
        return out

    def per_video(self):
        return {"surprise": self.sup}


@register("surprise_l1")
class SurpriseL1(Metric):
    """채점 전의 원값 — surprise 자체가 얼마나 커졌나. 정확도와 따로 움직일 수 있다."""

    needs_h = True

    def __init__(self, ds, cfg):
        super().__init__(ds, cfg)
        self.sup: dict[str, float] = {}

    def update(self, vids, p, h, p_clean):
        e = (p.float() - h.float()).abs().mean(dim=(1, 2))
        for v, s in zip(vids, e.tolist()):
            self.sup[v] = s

    def result(self):
        return {"mean": round(float(np.mean(list(self.sup.values()))), 5),
                "by_block_type": _by_type(self, self.sup)}

    def per_video(self):
        return {"surprise": self.sup}


@register("pred_drift")
class PredDrift(Metric):
    """예측 자체가 얼마나 움직였나 — |p − p_clean| / |p_clean| 와 토큰별 코사인.

    채점을 안 거치므로 **target 인코더가 필요 없다.** 이 지표만 쓰면 forward 가 절반이다.
    """

    needs_clean = True

    def __init__(self, ds, cfg):
        super().__init__(ds, cfg)
        self.rel: dict[str, float] = {}
        self.cos: dict[str, float] = {}

    def update(self, vids, p, h, p_clean):
        a, b = p.float(), p_clean.float()
        rel = (a - b).abs().mean(dim=(1, 2)) / b.abs().mean(dim=(1, 2)).clamp_min(1e-9)
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean(dim=1)
        for v, r, c in zip(vids, rel.tolist(), cos.tolist()):
            self.rel[v], self.cos[v] = r, c

    def result(self):
        return {"rel_l1": round(float(np.mean(list(self.rel.values()))), 5),
                "cosine": round(float(np.mean(list(self.cos.values()))), 5),
                "rel_l1_by_block_type": _by_type(self, self.rel)}

    def per_video(self):
        return {"drift_rel": self.rel, "drift_cos": self.cos}


def build(names: list[str], ds, cfg) -> dict[str, Metric]:
    bad = [n for n in names if n not in REGISTRY]
    if bad:
        raise ValueError(f"모르는 지표 {bad}. 있는 것: {sorted(REGISTRY)}")
    return {n: REGISTRY[n](ds, cfg) for n in names}
