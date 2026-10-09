#!/usr/bin/env python3
"""배경 토큰은 왜 Δ 를 갖나 — 두 검사 (2026-10-07, v11_realistic · ledge, token_surprise.npz 필요).

  1 픽셀 차 구간별 분해: 미래 토큰마다 가능 · 불가능 프레임의 픽셀 최대 차 (모델 입력과 같은 256 bilinear, antialias 없음,
    튜블릿 2 샘플 · 3 채널, 0–255) 를 구간 (=0 / 1–4 / 4–16 / 16–64 / >64) 으로 나눠 토큰 비율 · 쌍 Δ 기여 · 토큰당 |Δ|.
    → 픽셀이 완전히 같은 토큰도 Δ 를 갖는가 (갖는다면 원인은 픽셀이 아니라 target encoder 의 전역 attention)
  2 부호 일치: 쌍마다 배경 토큰 기여 · 물체 토큰 기여 (물체 마스크 = gif_token_surprise.obj_mask) 의 부호가 쌍 전체와 같은가
    → 배경 Δ 가 잡음 (부호 들쭉날쭉) 인가, 같은 판단의 메아리 (부호 일치) 인가

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/token_surprise_background.py   → z_research/v11_realistic/figures/token_surprise/background_checks.md
"""
from __future__ import annotations
import csv, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/figures"))
import gif_token_surprise as G                                    # noqa: E402  (load · pair_delta · obj_mask 를 같이 쓴다)

BINS = [(-1, 1, "=0"), (1, 4, "1–4"), (4, 16, "4–16"), (16, 64, "16–64"), (64, 256, ">64")]


def load256(path):
    x = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)[None].float()
    return torch.nn.functional.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=False)[0].numpy()


def tokdiff(fr, a, b):
    out = np.zeros((8, 16, 16))
    for t in range(8):
        for s in (16 + 2 * t, 17 + 2 * t):
            d = np.abs(load256(fr / a / f"{3 * s:06d}.png") - load256(fr / b / f"{3 * s:06d}.png")).max(0)
            out[t] = np.maximum(out[t], d.reshape(16, 16, 16, 16).max((1, 3)))
    return out


def main():
    jobs = []
    for name, pairs in [("ledge", [("ledge", "pos_fall", "imp_float")]), ("realistic", [("A", "pos_a", "imp_ab"), ("B", "pos_b", "imp_ba")])]:
        E, B, fr = G.load(name)
        md = {r["name"]: r for r in csv.DictReader((fr / "metadata.csv").open())}
        for d, pv, iv in pairs:
            for b in B.values():
                key = "ledge" if name == "ledge" else f"{b[pv]['condition']} {d}"
                jobs.append((key, fr, b[pv], b[iv], G.pair_delta(E, b[pv], b[iv]), G.obj_mask(b[pv], md) | G.obj_mask(b[iv], md)))
    with ThreadPoolExecutor(48) as ex:
        P = list(ex.map(lambda j: tokdiff(j[1], j[2]["file_name"], j[3]["file_name"]), jobs))
    res = {}
    for j, p in zip(jobs, P):
        res.setdefault(j[0], []).append((p, j[4], j[5]))
    md = ["## 1. 픽셀 차 구간별 — 토큰 비율 / 쌍 Δ 기여 / 토큰당 |Δ|", "",
          "| 조건 · 방향 | n | " + " | ".join(f"픽셀 차 {l}" for _, _, l in BINS) + " |", "|---|---:|" + "---|" * len(BINS)]
    for k in sorted(res):
        Pk = np.stack([x[0] for x in res[k]]); D = np.stack([x[1] for x in res[k]])
        cells = []
        for lo, hi, _ in BINS:
            m = (Pk > lo) & (Pk <= hi)
            cells.append(f"{100 * m.mean():.1f} % · {(D * m).sum() / D.size:+.4f} · {np.abs(D[m]).mean() if m.any() else 0:.3f}")
        md.append(f"| {k} | {len(Pk)} | " + " | ".join(cells) + " |")
    md += ["", "## 2. 부호 일치 — 쌍마다", "", "| 조건 · 방향 | n | 배경 기여 부호 = 쌍 전체 부호 | 배경 기여 > 0 | 물체 기여 > 0 |", "|---|---:|---:|---:|---:|"]
    for k in sorted(res):
        v = np.array([(x[1][~x[2]].sum(), x[1][x[2]].sum(), x[1].sum()) for x in res[k]])
        md.append(f"| {k} | {len(v)} | {100 * np.mean(np.sign(v[:, 0]) == np.sign(v[:, 2])):.1f} % | {100 * np.mean(v[:, 0] > 0):.1f} % | {100 * np.mean(v[:, 1] > 0):.1f} % |")
    out = G.OUT / "background_checks.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md)); print(f"\n→ {out}")


if __name__ == "__main__":
    main()
