#!/usr/bin/env python3
"""VOBench_v1 채점표 (2026-10-10). `cvpr/eval/testbed` (= surprise_c16t32) 실행의 per_block.json + index 에서 다시 계산한다.

  쌍 = matched pair. A = (pos_a, imp_ab) · B = (pos_b, imp_ba). hit = 1[S(pos) < S(imp)] (+0.5 동점).
  permanence 에서 A 의 불가능 = 물체→빈 (사라짐), B 의 불가능 = 빈→물체 (나타남). shape · color 는 A→B / B→A.
  표: **objset (primitive · realistic) 마다 따로** — 위반 × 가림 시점 (A/B 방향 병기) · 위반 × k · 운동 · 배경 · 물체 · 복사 기준선.

검증: 재계산 overall · by_block_type 이 summary.json 과 다르면 죽는다. CI = block 단위 부트스트랩 95 % (2,000 회).

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/vobench_tables.py --model vjepa2           # → z_research/VOBench_v1/vjepa2/surprise/tables.{md,json}
  $P z_research/scripts/analysis/vobench_tables.py --model dino_foresight   # → z_research/VOBench_v1/dino_foresight/surprise/tables.{md,json}
  (입력 = 하네스 원시 산출물 z_research/VOBench_v1/exp_results/<run>/ — 레지스트리 results_root 라 거기 둔다)
  모델 = MODELS 표 (채점 실행 폴더 · 복사 실행 폴더 · 이름). 복사 실행은 두 꼴을 읽는다:
    te 배열 (copy_baseline.sh: meta.json + l1/copy/hcopy.npy, V-JEPA 계열) · 채점기 per_block.json (dinof 의 model.dinof_copy=true)
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "z_research/VOBench_v1"            # 정리본: <모델>/surprise/ · <모델>/probing/
OUT = ROOT / "exp_results"                         # 하네스 원시 산출물 (레지스트리 results_root)
INDEX = REPO / "data_csv/vobench_v1/index.csv"
MODELS = {   # 이름 → (라벨, 채점 실행, 복사 실행)
    "vjepa2": ("V-JEPA 2 ViT-H", OUT / "surprise_c16t32__vobench_v1_vith", OUT / "copy__vobench_v1_vith"),
    "dino_foresight": ("DINO-Foresight (highres 448)", OUT / "surprise_c16t32__vobench_v1_dinof_highres",
                       OUT / "surprise_c16t32__vobench_v1_dinof_highres_copy"),
    # AR: 타깃 = 블록별 LN(target_encoder) — 복사는 같은 공간의 copy_blk (te 가 own kind=ar 일 때 쓴다)
    "ariel_ar_ep40": ("Ariel AR predictor ep40 (V-JEPA 2 ViT-H frozen)", OUT / "surprise_c16t32__vobench_v1_vith_ariel_ar_ep40",
                      OUT / "copy__vobench_v1_vith_ariel_ar_ep40"),
    # ar_state: copy_blk 는 encoder 만으로 정해져 ar 의 복사 실행을 그대로 쓴다 ('own' 검사는 predictor 가 달라 copy_scores 가 뺀다)
    "ariel_ar_state_ep40": ("Ariel AR-state predictor ep40 (V-JEPA 2 ViT-H frozen)", OUT / "surprise_c16t32__vobench_v1_vith_ariel_ar_state_ep40",
                            OUT / "copy__vobench_v1_vith_ariel_ar_ep40"),
    # vast 브랜치 (2026-10-13): 채점 = analysis/intphys2/model.py _ArScorer (블록별 타깃, 힌트 없음).
    #   복사 = 같은 공간 블록 복사 vast_ar_blockcopy (채점기 per_block.json, 마지막 문맥 블록 LN(h) × 8 = copy_blk 와 같은 정의)
    "vast_ar_vw_state_hint": ("Vast AR-state + hint 10% (K710+SSv2+HowTo, 20,580 step)", OUT / "surprise_c16t32__vobench_v1_vast_ar_vw_state_hint",
                              OUT / "surprise_c16t32__vobench_v1_vast_ar_blockcopy"),
    "vast_ar_vw_state": ("Vast AR-state (K710+SSv2+HowTo, 20,580 step)", OUT / "surprise_c16t32__vobench_v1_vast_ar_vw_state",
                         OUT / "surprise_c16t32__vobench_v1_vast_ar_blockcopy"),
    "vast_ar_vw_hint": ("Vast AR (output feedback) + hint sin2 (K710+SSv2+HowTo, 20,580 step)", OUT / "surprise_c16t32__vobench_v1_vast_ar_vw_hint",
                        OUT / "surprise_c16t32__vobench_v1_vast_ar_blockcopy"),
    "vast_ar_vw_state_delta": ("Vast AR-state + delta L1 (K710+SSv2+HowTo, 20,580 step)", OUT / "surprise_c16t32__vobench_v1_vast_ar_vw_state_delta",
                               OUT / "surprise_c16t32__vobench_v1_vast_ar_blockcopy"),
    # ctxt_video (2026-10-16): kind=ctx_state — 문맥 = 창 단위 encoder z, 타깃 = 블록별 LN(target_encoder) (ctx_target=block). 복사 기준선은 같은 타깃 공간의 블록 복사
    "vast_ar_vw_state_ctxt_video": ("Vast ctxt_video (video context + block TF + state rollout, 20,580 step)",
                                    OUT / "surprise_c16t32__vobench_v1_vast_ar_vw_state_ctxt_video",
                                    OUT / "surprise_c16t32__vobench_v1_vast_ar_blockcopy"),
    "vast_ar_vw_state_op": ("Vast AR-state + object_permanence 10% (boundary-aligned windows, 20,580 step)",
                            OUT / "surprise_c16t32__vobench_v1_vast_ar_vw_state_op",
                            OUT / "surprise_c16t32__vobench_v1_vast_ar_blockcopy"),
    "vast_tube_cd64": ("Vast tube baseline (release recipe, cooldown64)", OUT / "surprise_c16t32__vobench_v1_vast_tube_cd64", None),
    # VideoMAEv2: 창 16 고정 → c8t16_s6 (stride 6, 문맥 8 · 미래 8). 타깃 = 정규화 픽셀. 복사 경로 없음
    "videomae_v2": ("VideoMAEv2-g (window c8t16_s6)", OUT / "surprise_c16t32__vobench_v1_videomae2g_wc8t16_s6", None),
}
RUN, COPY = MODELS["vjepa2"][1], MODELS["vjepa2"][2]
VIOL = ["permanence", "shape", "color"]
OCC = ["visible", "early", "mid", "late"]
MOT = ["flat_static", "flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up"]
RNG = np.random.default_rng(0)


def _pred_ckpt(cfg: dict):
    return (cfg.get("model") or {}).get("predictor_checkpoint")


def copy_scores(cdir: Path, run: Path = None):
    """복사 실행 → {video_id: S}. te 배열이면 셋 (own = predictor 재계산 · copy = LN(z)[7] 복사 · hcopy = h[7] 복사),
    채점기 실행 (per_block.json, dinof_copy) 이면 copy 하나."""
    if cdir is None:
        return None
    if (cdir / "per_block.json").exists() and not (cdir / "meta.json").exists():
        return {"copy": {k: float(v) for k, v in json.load(open(cdir / "per_block.json"))["per_video_surprise"].items()}}
    if not (cdir / "meta.json").exists():
        return None
    meta = json.load(open(cdir / "meta.json"))
    l1, cp, hc = (np.load(cdir / f"{k}.npy") for k in ("l1", "copy", "hcopy"))
    k = meta["pred_tags"].index("own")
    v = meta["video_id"]
    if meta.get("pred_space", {}).get("own") == "ar_block":          # AR: 복사도 같은 블록별 공간 (copy_blk) 이어야 l1 과 비교된다
        cp = np.load(cdir / "copy_blk.npy")
    out = {n: dict(zip(v, x.astype(float))) for n, x in (("own", l1[:, k].mean(1)), ("copy", cp.mean(1)), ("hcopy", hc.mean(1)))}
    if run is not None:
        import yaml
        if _pred_ckpt(meta.get("resolved_config") or {}) != _pred_ckpt(yaml.safe_load(open(run / "_resolved.yaml"))):
            out.pop("own")          # 복사 실행의 predictor 가 채점 실행과 다르다 — 재계산 대조는 의미가 없다
    return out


def hit(sp, si):
    return 1.0 if sp < si else (0.5 if sp == si else 0.0)


def load(run: Path, C=None):
    S = json.load(open(run / "per_block.json"))["per_video_surprise"]
    rows = list(csv.DictReader(open(INDEX)))
    by = defaultdict(dict)
    for r in rows:
        by[r["block_id"]][r["variant"]] = r
    P = []          # 쌍 하나 = dict
    for bid, v in by.items():
        r = v["pos_a"]
        for d, pos, imp in (("A", "pos_a", "imp_ab"), ("B", "pos_b", "imp_ba")):
            sp, si = S[v[pos]["video_id"]], S[v[imp]["video_id"]]
            ex = {}
            if C:
                vp, vi = v[pos]["video_id"], v[imp]["video_id"]
                ex = {f"hit_{n}": hit(C[n][vp], C[n][vi]) for n in C}
            P.append(dict(block=bid, dir=d, hit=hit(sp, si), gap=(si - sp) / sp, **ex,
                          objset=r["objset"], viol=r["violation"], occ=r["occlusion"], k=r["k"], motion=r["motion"],
                          bg=r["background"], obj=r["obj_a"].split("/")[0], cond=r["condition"]))
    return P


def stat(ps, key="hit"):
    """정확도 % · block 부트스트랩 CI · A · B · n 쌍."""
    if not ps:
        return None
    blk = defaultdict(list)
    for p in ps:
        blk[p["block"]].append(p[key])
    m = np.array([np.mean(v) for v in blk.values()])
    bs = m[RNG.integers(0, len(m), (2000, len(m)))].mean(1)
    dA = [p[key] for p in ps if p["dir"] == "A"]; dB = [p[key] for p in ps if p["dir"] == "B"]
    return dict(acc=100 * m.mean(), lo=100 * np.percentile(bs, 2.5), hi=100 * np.percentile(bs, 97.5),
                A=100 * np.mean(dA) if dA else float("nan"), B=100 * np.mean(dB) if dB else float("nan"),
                n=len(ps), gap=100 * float(np.median([p["gap"] for p in ps])))


def cell(s, ab=True):
    if s is None:
        return "—"
    t = f"{s['acc']:.1f} [{s['lo']:.0f}, {s['hi']:.0f}]"
    return t + (f" · {s['A']:.0f}/{s['B']:.0f}" if ab else "")


def grid(P, rkey, rvals, ckey, cvals, filt=lambda p: True, ab=True, total=True):
    """행 rkey × 열 ckey 표 (md 줄 목록, json dict)."""
    head = f"| {rkey} \\ {ckey} | " + " | ".join(cvals) + (" | 전체 |" if total else " |")
    lines = [head, "|---|" + "---:|" * (len(cvals) + total)]
    js = {}
    for rv in rvals:
        row = [stat([p for p in P if filt(p) and p[rkey] == rv and p[ckey] == cv]) for cv in cvals]
        if total:
            row.append(stat([p for p in P if filt(p) and p[rkey] == rv]))
        js[rv] = dict(zip(list(cvals) + (["all"] if total else []), row))
        lines.append(f"| `{rv}` | " + " | ".join(cell(s, ab) for s in row) + " |")
    if total:
        row = [stat([p for p in P if filt(p) and p[ckey] == cv]) for cv in cvals] + [stat([p for p in P if filt(p)])]
        js["all"] = dict(zip(list(cvals) + ["all"], row))
        lines.append("| **전체** | " + " | ".join(cell(s, ab) for s in row) + " |")
    return lines, js


def dstat(ps):
    """Δ = 모델 − 복사 (pt). 같은 block 을 다시 뽑는 짝 부트스트랩 95 % CI."""
    blk = defaultdict(list)
    for p in ps:
        blk[p["block"]].append((p["hit"], p["hit_copy"]))
    m = np.array([np.mean([x for x, _ in v]) for v in blk.values()]); c = np.array([np.mean([y for _, y in v]) for v in blk.values()])
    ix = RNG.integers(0, len(m), (2000, len(m)))
    bs = (m[ix] - c[ix]).mean(1)
    return 100 * (m.mean() - c.mean()), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def copy_grid(P, rkey, rvals, ckey, cvals, filt=lambda p: True):
    """칸 = 모델 · 복사 (A/B) · Δ [CI]."""
    lines = [f"| {rkey} \\ {ckey} | " + " | ".join(cvals) + " | 전체 |", "|---|" + "---:|" * (len(cvals) + 1)]
    js = {}
    for rv in list(rvals) + ["all"]:
        cells, js[rv] = [], {}
        for cv in list(cvals) + ["all"]:
            ps = [p for p in P if filt(p) and (rv == "all" or p[rkey] == rv) and (cv == "all" or p[ckey] == cv)]
            if not ps:
                cells.append("—"); continue
            m, c = stat(ps), stat(ps, "hit_copy"); d = dstat(ps)
            js[rv][cv] = dict(model=m, copy=c, delta=d)
            sig = "**" if d[1] > 0 or d[2] < 0 else ""
            cells.append(f"{m['acc']:.1f} · {c['acc']:.1f} ({c['A']:.0f}/{c['B']:.0f}) · {sig}{d[0]:+.1f} [{d[1]:+.0f}, {d[2]:+.0f}]{sig}")
        lines.append(("| **전체** | " if rv == "all" else f"| `{rv}` | ") + " | ".join(cells) + " |")
    return lines, js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vjepa2", choices=list(MODELS))
    ap.add_argument("--out", default=None, help="기본 z_research/VOBench_v1/<모델>/surprise/")
    a = ap.parse_args()
    label, run, cdir = MODELS[a.model]
    out = Path(a.out) if a.out else ROOT / a.model / "surprise"
    C = copy_scores(cdir, run)
    P = load(run, C)
    summ = json.load(open(run / "summary.json"))["surprise"]

    # 검증: 채점기와 같은 수
    ov = stat(P)
    assert len(P) == summ["overall"]["n_pair"], (len(P), summ["overall"]["n_pair"])
    assert abs(ov["acc"] / 100 - summ["overall"]["block_pairwise"]) < 1e-9, (ov["acc"], summ["overall"]["block_pairwise"])
    for k, v in summ["by_block_type"].items():
        s = stat([p for p in P if p["cond"] == k])
        assert abs(s["acc"] / 100 - v["block_pairwise"]) < 1e-9, (k, s["acc"], v["block_pairwise"])
    n_tie = sum(p["hit"] == 0.5 for p in P)

    OBJSETS = ("primitive", "realistic")
    md = [f"# VOBench_v1 — {label} `surprise_c16t32` 채점표 (objset 별)", "",
          f"실행 `{run.relative_to(REPO) if run.is_relative_to(REPO) else run}` · matched pair {len(P)} · 동점 {n_tie}. "
          f"재계산 overall · by_block_type {len(summ['by_block_type'])} 칸이 summary.json 과 일치 (검증 통과).",
          "**primitive · realistic 은 따로 기록한다** (물체 · 색 집합이 달라 합친 평균은 쓰지 않는다).",
          "칸 = 정확도 % [block 부트스트랩 95 % CI] · A / B 방향 정확도. permanence 의 A = 물체→빈 (사라짐), B = 빈→물체 (나타남).", ""]
    J = {"overall_pooled": ov, "n_ties": n_tie}
    for o in OBJSETS:
        J[o] = {"overall": stat([p for p in P if p["objset"] == o])}
    md += ["| objset | overall | 쌍 |", "|---|---:|---:|"]
    md += [f"| `{o}` | {cell(J[o]['overall'])} | {J[o]['overall']['n']} |" for o in OBJSETS]
    md += [f"| (참고) 합침 | {cell(ov)} | {ov['n']} |", ""]

    HAS_OWN = bool(C) and "own" in C
    if C:
        # 검증: 복사 실행이 같은 clip 을 덮는가 (+ te 배열이면 그 안에서 다시 지은 predictor 가 채점기와 같은 쌍 판정을 내는가)
        S = json.load(open(run / "per_block.json"))["per_video_surprise"]
        assert set(C["copy"]) == set(S), "복사 실행의 video_id 가 채점 실행과 다르다"
        if HAS_OWN:
            rel = max(abs(C["own"][v] - S[v]) / S[v] for v in S)
            flip = sum(p["hit"] != p["hit_own"] for p in P)
            J["copy_check"] = dict(own_vs_harness_max_rel=rel, own_vs_harness_pair_flips=flip)

    for o in OBJSETS:
        f0 = lambda p, o=o: p["objset"] == o
        Jo = J[o]
        md += [f"## {o}", "", f"overall **{cell(Jo['overall'])}**", ""]
        md += [f"### {o} 1. 위반 × 가림 시점", ""]
        l, Jo["viol_x_occ"] = grid(P, "viol", VIOL, "occ", OCC, filt=f0); md += l + [""]
        md += [f"### {o} 2. 위반 × k (가림 시점별)", ""]
        for oc in ("early", "mid", "late"):
            md += [f"**{oc}**", ""]
            l, Jo[f"viol_x_k_{oc}"] = grid(P, "viol", VIOL, "k", ["1", "2", "3"], filt=lambda p, oc=oc, o=o: p["objset"] == o and p["occ"] == oc)
            md += l + [""]
        md += [f"### {o} 3. 운동 × 가림 시점 (위반 셋 합침)", ""]
        l, Jo["motion_x_occ"] = grid(P, "motion", MOT, "occ", OCC, filt=f0); md += l + [""]
        md += [f"### {o} 4. 운동 × 위반 — late 만", ""]
        l, Jo["motion_x_viol_late"] = grid(P, "motion", MOT, "viol", VIOL, filt=lambda p, o=o: p["objset"] == o and p["occ"] == "late"); md += l + [""]
        md += [f"### {o} 5. 배경 × 가림 시점", ""]
        l, Jo["bg_x_occ"] = grid(P, "bg", ["v11", "assets"], "occ", OCC, filt=f0); md += l + [""]
        md += [f"### {o} 6. 물체 (obj_a) × 위반 — 가림 시점 합침, A/B 생략", ""]
        objs = sorted({p["obj"] for p in P if p["objset"] == o})
        l, Jo["obj"] = grid(P, "obj", objs, "viol", VIOL, filt=f0, ab=False); md += l + [""]
        if C:
            Po = [p for p in P if p["objset"] == o]
            cp, d = stat(Po, "hit_copy"), dstat(Po)
            Jo["copy"] = dict(copy=cp, delta=d)
            if HAS_OWN:
                hc, own = stat(Po, "hit_hcopy"), stat(Po, "hit_own"); d_own = dstat([dict(p, hit=p["hit_own"]) for p in Po])
                Jo["copy"].update(hcopy=hc, own=own, delta_own=d_own)
                how = (("copy S = mean_j mean|hb[7] − hb[8+j]| (AR 공간: 블록별 LN(target_encoder) 의 마지막 문맥 블록, te `copy_blk`). 같은 쌍 · 같은 채점. "
                        if a.model == "ariel_ar_ep40" else
                        "copy S = mean_j mean|LN(z)[7] − h[8+j]| (문맥 encoder 마지막 튜블릿, `cvpr/eval/testbed/copy_baseline.sh`). 같은 쌍 · 같은 채점. ") + (
                       f"복사 실행 안에서 다시 지은 predictor 는 채점기와 S 상대차 최대 {rel:.1e} · 쌍 판정 뒤집힘 (전체) {flip} 개 (autocast fp16 배치 차이, CLAUDE.md §7-2) — "
                       f"이 objset 에서 그 판 모델 {own['acc']:.1f}, 같은 파이프라인 Δ {d_own[0]:+.1f} [{d_own[1]:+.1f}, {d_own[2]:+.1f}]. 아래 모델 = 채점기."))
                ref = ("" if a.model == "ariel_ar_ep40" else   # AR 의 hcopy 는 표준 공간이라 l1 과 비교 불가
                       f" (참고: target 쪽 복사 h[7] {hc['acc']:.1f} — 32 장 전부를 본 양방향 표현이라 예측 없는 기준선이 아니다)")
            else:
                how = (f"복사 = 같은 채점기의 블록 복사 모델 `vast_ar_blockcopy` (`predictor.ar_copy`, 마지막 문맥 블록 LN(target_encoder) 를 미래 8 블록에 복사 = copy_blk), 실행 `{cdir.name}`. 같은 쌍 · 같은 채점."
                       if "blockcopy" in cdir.name else
                       f"복사 = 같은 채점기에 `model.dinof_copy=true` (마지막 문맥 프레임 특징을 미래 16 장에 복사), 실행 `{cdir.name}`. 같은 쌍 · 같은 채점.")
                ref = ""
            md += [f"### {o} 7. 복사 기준선 — 예측 없이 마지막 문맥 특징을 미래에 복사", "", how, "",
                   f"**모델 {Jo['overall']['acc']:.1f} · 복사 {cell(cp)} · Δ {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]**{ref}", "",
                   "칸 = 모델 · 복사 (복사 A/B) · Δ = 모델 − 복사 [짝 block 부트스트랩 95 % CI]. **굵게** = CI 가 0 을 안 포함.", ""]
            l, Jo["copy_viol_x_occ"] = copy_grid(P, "viol", VIOL, "occ", OCC, filt=f0); md += l + [""]
            md += ["**late — 운동 × 위반**", ""]
            l, Jo["copy_motion_x_viol_late"] = copy_grid(P, "motion", MOT, "viol", VIOL, filt=lambda p, o=o: p["objset"] == o and p["occ"] == "late")
            md += l + [""]

    RECIPE = {
        "vjepa2": ["GPUS=4 RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1 vith",
                   "GPUS=4 OUTDIR=z_research/VOBench_v1/exp_results/copy__vobench_v1_vith bash cvpr/eval/testbed/copy_baseline.sh vobench_v1 vith"],
        "dino_foresight": ["GPUS=4 RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1 dinof_highres",
                           "COPY=1 GPUS=4 RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1 dinof_highres"],
        "ariel_ar_ep40": ["GPUS=4 RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1 vith_ariel_ar_ep40",
                          "GPUS=4 OUTDIR=z_research/VOBench_v1/exp_results/copy__vobench_v1_vith_ariel_ar_ep40 bash cvpr/eval/testbed/copy_baseline.sh vobench_v1 vith_ariel_ar_ep40"],
        "ariel_ar_state_ep40": ["GPUS=4 DECODE_WORKERS=8 BATCH_SIZE=16 bash z_research/scripts/run.sh surprise_c16t32 vobench_v1 vith_ariel_ar_state_ep40",
                                "# 복사 = ariel_ar_ep40 의 copy__ 실행 (copy_blk, predictor 무관)"],
        "videomae_v2": ["GPUS=4 WINDOW=c8t16_s6 RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1 videomae2g"]}
    for _m in ("vast_ar_vw_state_hint", "vast_ar_vw_state", "vast_ar_vw_hint", "vast_ar_vw_state_delta", "vast_tube_cd64", "vast_ar_vw_state_ctxt_video", "vast_ar_vw_state_op"):   # vast 브랜치 (2026-10-13, Vast 8 GPU)
        RECIPE[_m] = [f"RESULTS_ROOT=legacy GPUS=8 bash cvpr/eval/testbed/run.sh vobench_v1 {_m}",
                      "RESULTS_ROOT=legacy GPUS=8 bash cvpr/eval/testbed/run.sh vobench_v1 vast_ar_blockcopy   # 복사 (블록 공간)"]
    md += ["## 재현", "", "```bash"] + RECIPE[a.model] + [
           f"python z_research/scripts/analysis/vobench_tables.py --model {a.model}", "```"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables.md").write_text("\n".join(md) + "\n")
    (out / "tables.json").write_text(json.dumps(J, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
