#!/usr/bin/env python3
"""위치 자 비교 — **존재 판정 오류**와 **좌표 오류**를 따로, 그리고 교차해서 본다.

두 오류는 경로가 달라서 한쪽이 맞고 다른 쪽이 틀리는 게 가능하다 (2026-09-23 실측:
presence +5.3 으로 '있음' 을 맞게 답하면서 좌표는 141 px 틀린 경우). 그래서 항상 따로 낸다.

  존재 판정  ROC/PR (문턱 무관) · val 에서 고른 문턱 하나로 출처별 오탐 · 미탐 · 전체 오류율
  좌표      평균/중앙/p90/p99/최대 px · 1 칸(18px)·2 칸·100 px 초과 비율
  교차      '있음' 을 맞게 답한 칸에서만 좌표 오차를 다시 잰다

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/presence_readout_compare.py
  $P z_research/scripts/analysis/presence_readout_compare.py --md   # RESULTS.md 에 붙일 표로
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
import sys; sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))   # noqa: E702
from rollout3_paths import PRES as RES   # noqa: E402  자 폴더는 한 곳에서 (R3_DECODER, 기본 presence)
def train_index():
    """자를 **배운 세트**의 인덱스. `summary.json` 의 `train_set` 을 따른다 (없으면 옛 training_v6).

    ⚠️ 하드코딩하면 자를 새 세트로 다시 배운 뒤 **옛 라벨로 채점**한다 (2026-09-24 에 막았다).
    """
    S = json.loads((RES / "summary.json").read_text())
    return ROOT / f"data_csv/rollout_v2_{S.get('train_set', 'training_v6')}/index_probe.csv"
CELL, NORM = 18.0, 144.0                                   # 1 칸 px · 정규화 기준 (px/144 - 1)


def load(rep, kind):
    f = RES / rep / f"preds_{kind}.npz"
    return np.load(f, allow_pickle=True) if f.exists() else None


def groups(z, rows):
    """양성 / 제외 / 화면 밖 / 빈 장면 — test 안에서만."""
    n = len(rows)
    a = lambda s: np.array([float(v) for v in str(s).split()], np.float32)
    inf_ = (np.stack([a(r["in_frame_by_sample"]) for r in rows]) > 0).reshape(n, 16, 2).all(2)[:, 8:]
    emp = np.array([r["scenario"] == "empty" for r in rows])[:, None].repeat(8, 1)
    pos, ign = z["pos"], z["ign"]
    neg = (~pos) & (~ign)
    T = z["test"][:, None]
    return dict(양성=T & pos, 제외=T & ign, 화면밖=T & neg & ~emp & ~inf_, 빈장면=T & neg & emp), T & pos, T & neg


def main(as_md=False):
    rows = list(csv.DictReader(train_index().open()))
    S = json.loads((RES / "summary.json").read_text())
    kinds = sorted({k for v in S["reps"].values() for k in v})
    out = []
    P = (lambda *a: out.append(" ".join(str(x) for x in a)))

    P("## 1. 존재 판정 오류  (test, 문턱 = val 에서 전체 음성 5 % 오탐 지점)\n")
    P("| 표현 | probe | ROC | PR | 문턱 | 미탐 (있는데 없다) | 오탐·빈장면 | 오탐·화면밖 | **전체 오류율** |")
    P("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    ERR = {}
    for rep in ("p", "z", "h"):
        for kind in kinds:
            z = load(rep, kind)
            if z is None: continue
            g, MP, MN = groups(z, rows)
            thr = S["reps"][rep][kind]["thr_val_fpr5"]
            PR_ = z["presence"]
            fn = (PR_[MP] <= thr).mean()
            fp_e = (PR_[g["빈장면"]] > thr).mean() if g["빈장면"].any() else np.nan
            fp_o = (PR_[g["화면밖"]] > thr).mean() if g["화면밖"].any() else np.nan
            tot = ((PR_[MP] <= thr).sum() + (PR_[MN] > thr).sum()) / (MP.sum() + MN.sum())
            ERR[(rep, kind)] = (thr, tot)
            b = (lambda s, v: f"**{s}**" if v else s)
            P(f"| {rep} | {kind} | {S['reps'][rep][kind]['roc_auc']:.3f} | {S['reps'][rep][kind]['pr_auc']:.3f} | "
              f"{thr:+.2f} | {100*fn:.1f}% | {100*fp_e:.1f}% | {100*fp_o:.1f}% | {100*tot:.2f}% |")

    P("\n## 2. 좌표 오류  (모든 GT 양성 tubelet, px. 1 칸 = 18 px = 물체 반폭)\n")
    P("| 표현 | probe | 평균 | 중앙 | p90 | p99 | 최대 | >18px | >36px | >100px |")
    P("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rep in ("p", "z", "h"):
        for kind in kinds:
            z = load(rep, kind)
            if z is None: continue
            _, MP, _ = groups(z, rows)
            e = np.linalg.norm(z["pred"] - z["truth"], axis=-1)[MP] * NORM
            P(f"| {rep} | {kind} | {e.mean():.1f} | {np.median(e):.1f} | {np.percentile(e,90):.1f} | "
              f"{np.percentile(e,99):.1f} | {e.max():.1f} | {100*(e>18).mean():.1f}% | "
              f"{100*(e>36).mean():.1f}% | {100*(e>100).mean():.2f}% |")

    P("\n## 3. 교차 — 존재를 **맞게** 답한 칸에서만 좌표를 다시 재면\n")
    P("| 표현 | probe | n | 평균 px | p99 px | >100px | (전체 대비 평균 변화) |")
    P("|---|---|---:|---:|---:|---:|---:|")
    for rep in ("p", "z", "h"):
        for kind in kinds:
            z = load(rep, kind)
            if z is None: continue
            _, MP, _ = groups(z, rows)
            thr = ERR[(rep, kind)][0]
            ok = MP & (z["presence"] > thr)                 # 있음을 맞게 답한 양성
            err = np.linalg.norm(z["pred"] - z["truth"], axis=-1) * NORM
            e0, e1 = err[MP], err[ok]
            P(f"| {rep} | {kind} | {int(ok.sum()):,} | {e1.mean():.1f} | {np.percentile(e1,99):.1f} | "
              f"{100*(e1>100).mean():.2f}% | {e1.mean()-e0.mean():+.1f} |")
    P("\n⚠️ 두 오류는 경로가 다르다 — presence 를 맞게 답하고도 좌표가 100 px 넘게 틀릴 수 있다.")
    print("\n".join(out))
    if as_md:
        (RES / "COMPARE.md").write_text("# 존재 판정 오류 vs 좌표 오류\n\n" + "\n".join(out) + "\n")
        print(f"\n→ {RES/'COMPARE.md'}")


def bias():
    """attn 자의 **좌표 치우침** (offset) 을 학습셋 test 에서 재어 `attn_bias_px.json` 에 남긴다.

    attn 은 pooled 벡터에서 좌표를 **회귀**하므로 학습 평균 쪽으로 수축한다 → 표현마다
    고정 오프셋이 남는다. 정의는 **test 분할 × 양성 칸**의 평균 `(pred - truth)` px 하나다
    (존재 문턱으로 거르든 안 거르든 0.1 px 안에서 같다: 2026-09-23 실측).
    쓰는 쪽은 읽은 좌표에서 이 값을 **빼면** 된다.
    """
    out = {}
    for rep in ("p", "z", "h"):
        z = load(rep, "attn")
        if z is None:
            continue
        e = (z["pred"] - z["truth"]) * NORM
        m = z["test"][:, None] & z["pos"]
        out[rep] = [round(float(v), 2) for v in e[m].mean(0)]
        print(f"  {rep}: dx {out[rep][0]:+6.2f} px  dy {out[rep][1]:+6.2f} px   (n={int(m.sum()):,})")
    (RES / "attn_bias_px.json").write_text(json.dumps(out, indent=1))
    print(f"→ {RES/'attn_bias_px.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--bias", action="store_true", help="attn_bias_px.json 만 다시 쓴다")
    a = ap.parse_args()
    bias() if a.bias else main(a.md)
