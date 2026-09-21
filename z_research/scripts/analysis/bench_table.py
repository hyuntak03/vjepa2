#!/usr/bin/env python
"""Garrido 벤치마크 표를 **산출물에서 다시 계산해** markdown 으로 낸다. GPU 불필요.

    python z_research/scripts/analysis/bench_table.py            # 화면
    python z_research/scripts/analysis/bench_table.py --readme   # README.md §4 표 갱신

지표는 **쌍 비교 AvgSurprise 하나**다 (PROTOCOLS.md §1-b). MaxSurprise 는 2026-09-21 에 철회했다 —
그건 단일 영상 + AUROC 용이다 (논문 Table S5).

수치를 손으로 옮겨 적지 않기 위한 스크립트다. 문서에 박힌 값과 어긋나면 **이쪽이 맞다**
(CLAUDE.md §7-4: 수치는 문서가 아니라 산출물에서 다시 계산해 검증한다).

채점은 `garrido_rescore.py` 가 한다 — 공식 축약 규칙이 데이터셋마다 다르기 때문이다
(IntPhys = Filtered(min) / GRASP·InfLevel = property 마다 최고 C, PROTOCOLS.md §3).
창(C+M)마다 실행이 나뉘어 있고 A.8 은 그 창들을 **하나의 탐색 공간**으로 보므로 합쳐서 고른다.

읽는 곳
    z_research/Benchmarks/exp_results/intphys1_sliding__<데이터셋>_<모델>_w{16,32}/per_window.json
"""
from __future__ import annotations
import argparse, pathlib, re, sys

REPO = pathlib.Path(__file__).resolve().parents[3]
BASE = REPO / "z_research/Benchmarks/exp_results"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from garrido_rescore import collect, collect_inflevel      # noqa: E402

MODELS = [("vith", "V-JEPA 2 ViT-H"), ("vjepa21g", "V-JEPA 2.1 ViT-g"), ("videomae2g", "VideoMAEv2-g")]
# 공식 보고 단위. InfLevel-lab 은 **한 데이터셋 3 property** 라 macro 로 합친다
# (`utils.py PROPERTIES_BY_DATASET['inflevel_lab']`). 아래 property 별 줄은 참고용이다.
ROWS = [("IntPhys 1 dev", "intphys1_dev"),
        ("GRASP level2", "grasp_level2"),
        ("InfLevel-lab (3 property macro)", "inflevel_lab"),
        ("  ┗ continuity", "inflevel_continuity"),
        ("  ┗ solidity ¹", "inflevel_solidity"),
        ("  ┗ gravity ¹", "inflevel_gravity")]


def build() -> str:
    got = {}
    for _, ds in ROWS:
        for m, _ in MODELS:
            r = collect_inflevel(BASE, m) if ds == "inflevel_lab" else collect(BASE, ds, m)
            if r:
                got[(ds, m)] = r["best"]

    out = ["| 벤치마크 | 최고 설정 | n_pair | " + " | ".join(n for _, n in MODELS) + " |",
           "|---|---|---:|" + "---:|" * len(MODELS)]
    for label, ds in ROWS:
        cells, settings, npair = [], set(), 0
        for m, _ in MODELS:
            b = got.get((ds, m))
            if b is None:
                cells.append("*미완*"); continue
            cells.append(f"**{b['accuracy']:.2f}**" if b else "—")
            settings.add(b["combo"]); npair = max(npair, b["n_pair"])
        out.append(f"| {label} | {' · '.join(sorted(settings)) or '—'} | {npair or '—'} | "
                   + " | ".join(cells) + " |")

    # 모델마다 어떤 칸이 뽑혔는지 (A.8 은 descriptive 라 반드시 같이 낸다)
    out += ["", "선택된 A.8 칸 (property 별 축약은 `garrido_rescore.py` 출력 참고):", "",
            "| 벤치마크 | " + " | ".join(n for _, n in MODELS) + " |",
            "|---|" + "---|" * len(MODELS)]
    for label, ds in ROWS:
        cells = []
        for m, _ in MODELS:
            b = got.get((ds, m))
            cells.append(f"`{b['combo']}`" if b else "—")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out += ["", "¹ **원리적으로 못 푸는 property** — 컵 상태가 공식 로더가 잘라내는 priming 구간에만 나온다",
            "(Garrido App. E). 50 근처가 정상이고 모델 비교에 쓰지 않는다."]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", action="store_true")
    a = ap.parse_args()
    table = build()
    if not a.readme:
        print(table); sys.exit(0)
    p = REPO / "z_research/Benchmarks/README.md"
    s = p.read_text()
    m = re.search(r"(<!--TABLE:start-->\n)(.*?)(<!--TABLE:end-->)", s, re.S)
    if not m:
        sys.exit("README.md 에 <!--TABLE:start--> / <!--TABLE:end--> 표시가 없다")
    p.write_text(s[:m.start()] + m.group(1) + table + "\n" + m.group(3) + s[m.end():])
    print(f"갱신: {p}")
