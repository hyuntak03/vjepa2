#!/usr/bin/env python3
"""VOBench_v1 asymmetry 비교 (2026-10-10) — late 클립만, 창 16:12 (c16t28) 두 판을 같은 block · 같은 쌍으로 맞댄다.

  original   창 시작 0 (가림 = 문맥 끝 k + 미래 앞 k)                   exp_results/surprise_c16t32__vobench_v1_vith_wc16t28
  asymmetry  창 시작 3k (가림 = 문맥 끝 2k, 미래는 첫 샘플부터 보임)     exp_results/surprise_c16t32__vobench_v1_asymmetry_vith_wc16t28
             불가능 arm 의 문맥 = 가능 arm 프레임 (바꿈이 문맥 / 미래 경계) — z_research/scripts/data/build_vobench_asym.py
  표: objset × 위반 (있는데→있는 A / 없는데→없는 B) · 운동 · k, 정확도 [block 부트스트랩 95 % CI] 와 Δ = asymmetry − original [짝 부트스트랩].
  검증: 재계산 overall 이 각 summary.json 과 같아야 한다 (original 은 late 블록만 다시 세므로 by_block_type late 칸과 대조).

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/vobench_asym_compare.py --model vjepa2     # → z_research/VOBench_v1/vjepa2/surprise_asymmetry/tables.{md,json}
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "z_research/VOBench_v1"; EXP = ROOT / "exp_results"
RUNS = {"vjepa2": ("V-JEPA 2 ViT-H", "surprise_c16t32__vobench_v1_vith_wc16t28", "surprise_c16t32__vobench_v1_asymmetry_vith_wc16t28")}
VIOL = ["permanence", "shape", "color"]
MOT = ["flat_static", "flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up"]
RNG = np.random.default_rng(0)


def pairs(run, blocks):
    S = json.load(open(EXP / run / "per_block.json"))["per_video_surprise"]
    out = {}
    for b, v in blocks.items():
        for d, pos, imp in (("A", "pos_a", "imp_ab"), ("B", "pos_b", "imp_ba")):
            sp, si = S[v[pos]["video_id"]], S[v[imp]["video_id"]]
            out[(b, d)] = 1.0 if sp < si else (0.5 if sp == si else 0.0)
    return out


def boot(keys, H):
    blk = defaultdict(list)
    for k in keys:
        blk[k[0]].append(H[k])
    m = np.array([np.mean(v) for v in blk.values()])
    bs = m[RNG.integers(0, len(m), (2000, len(m)))].mean(1)
    return 100 * m.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def dboot(keys, H0, H1):
    blk = defaultdict(list)
    for k in keys:
        blk[k[0]].append((H0[k], H1[k]))
    a = np.array([np.mean([x for x, _ in v]) for v in blk.values()]); b = np.array([np.mean([y for _, y in v]) for v in blk.values()])
    ix = RNG.integers(0, len(a), (2000, len(a))); bs = (b[ix] - a[ix]).mean(1)
    return 100 * (b.mean() - a.mean()), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="vjepa2", choices=list(RUNS)); a = ap.parse_args()
    label, r0, r1 = RUNS[a.model]
    rows = [r for r in csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv")) if r["occlusion"] == "late"]
    blocks = defaultdict(dict)
    for r in rows:
        blocks[r["block_id"]][r["variant"]] = r
    H0, H1 = pairs(r0, blocks), pairs(r1, blocks)
    meta = {b: v["pos_a"] for b, v in blocks.items()}
    # 검증: asymmetry 는 overall, original 은 late 칸들
    s1 = json.load(open(EXP / r1 / "summary.json"))["surprise"]
    assert abs(boot(list(H1), H1)[0] / 100 - s1["overall"]["block_pairwise"]) < 1e-9
    s0 = json.load(open(EXP / r0 / "summary.json"))["surprise"]["by_block_type"]
    for bt, v in s0.items():
        if bt.endswith("/late"):
            ks = [k for k in H0 if meta[k[0]]["condition"] == bt]
            assert abs(boot(ks, H0)[0] / 100 - v["block_pairwise"]) < 1e-9, bt

    cell = lambda x: f"{x[0]:.1f} [{x[1]:.0f}, {x[2]:.0f}]"
    dcell = lambda x: ("**" if x[1] > 0 or x[2] < 0 else "") + f"{x[0]:+.1f} [{x[1]:+.0f}, {x[2]:+.0f}]" + ("**" if x[1] > 0 or x[2] < 0 else "")
    J = {}; md = [f"# VOBench_v1 asymmetry — {label}, late 만, 창 16:12 (c16t28)", "",
                  "original = 창 시작 0 (가림 = 문맥 끝 k + 미래 앞 k) · asymmetry = 창 시작 3k (가림 = 문맥 끝 2k, 미래는 첫 샘플부터 보임, 바꿈이 문맥 / 미래 경계).",
                  "칸 = 정확도 % [block 부트스트랩 95 % CI]. Δ = asymmetry − original [짝 부트스트랩], **굵게** = CI 가 0 을 안 포함. "
                  "A = 있는데→있는 (permanence 불가능 = 사라짐), B = 없는데→없는 (불가능 = 나타남). 재계산이 두 summary.json 과 일치 (assert).", ""]
    for o in ("primitive", "realistic"):
        J[o] = {}
        md += [f"## {o}", "", "| | original | asymmetry | Δ | original A / B | asymmetry A / B |", "|---|---:|---:|---:|---:|---:|"]
        for v in VIOL + ["all"]:
            ks = [k for k in H0 if meta[k[0]]["objset"] == o and (v == "all" or meta[k[0]]["violation"] == v)]
            kA = [k for k in ks if k[1] == "A"]; kB = [k for k in ks if k[1] == "B"]
            r = dict(original=boot(ks, H0), asymmetry=boot(ks, H1), delta=dboot(ks, H0, H1),
                     A=(boot(kA, H0)[0], boot(kA, H1)[0]), B=(boot(kB, H0)[0], boot(kB, H1)[0]), n=len(ks))
            J[o][v] = r
            md.append(f"| {'**전체**' if v == 'all' else v} | {cell(r['original'])} | {cell(r['asymmetry'])} | {dcell(r['delta'])} | "
                      f"{r['A'][0]:.0f} / {r['B'][0]:.0f} | {r['A'][1]:.0f} / {r['B'][1]:.0f} |")
        md += ["", f"**{o} — permanence × 운동 (A 있는데→있는 / B 없는데→없는)**", "", "| 운동 | original A / B | asymmetry A / B |", "|---|---:|---:|"]
        for m in MOT:
            ks = [k for k in H0 if meta[k[0]]["objset"] == o and meta[k[0]]["violation"] == "permanence" and meta[k[0]]["motion"] == m]
            f = lambda H, d: 100 * np.mean([H[k] for k in ks if k[1] == d])
            J[o][f"perm_{m}"] = dict(original=(f(H0, "A"), f(H0, "B")), asymmetry=(f(H1, "A"), f(H1, "B")), n_pairs_per_dir=len(ks) // 2)
            md.append(f"| {m} | {f(H0, 'A'):.0f} / {f(H0, 'B'):.0f} | {f(H1, 'A'):.0f} / {f(H1, 'B'):.0f} |")
        md += ["", f"**{o} — k 별 전체**", "", "| k | original | asymmetry | Δ |", "|---|---:|---:|---:|"]
        for kk in ("1", "2", "3"):
            ks = [k for k in H0 if meta[k[0]]["objset"] == o and meta[k[0]]["k"] == kk]
            r = dict(original=boot(ks, H0), asymmetry=boot(ks, H1), delta=dboot(ks, H0, H1)); J[o][f"k{kk}"] = r
            md.append(f"| {kk} | {cell(r['original'])} | {cell(r['asymmetry'])} | {dcell(r['delta'])} |")
        md.append("")
    md += ["## 재현", "", "```bash", "P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python",
           "$P z_research/scripts/data/build_vobench_asym.py",
           "GPUS=4 RESULTS_ROOT=legacy SET=\"window.n_frames=28\" bash cvpr/eval/testbed/run.sh vobench_v1_asymmetry vith",
           "GPUS=4 RESULTS_ROOT=legacy SET=\"window.n_frames=28\" bash cvpr/eval/testbed/run.sh vobench_v1 vith",
           f"$P z_research/scripts/analysis/vobench_asym_compare.py --model {a.model}", "```"]
    out = ROOT / a.model / "surprise_asymmetry"; out.mkdir(parents=True, exist_ok=True)
    (out / "tables.md").write_text("\n".join(md) + "\n"); (out / "tables.json").write_text(json.dumps(J, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
