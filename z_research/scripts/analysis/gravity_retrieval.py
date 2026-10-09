#!/usr/bin/env python3
"""gravity_realistic — p 는 미래 넷 중 어느 것을 만드나 (2026-10-07).

block = 과거 하나 + 미래 다섯 (문맥 픽셀 동일, 감사 mismatch 0). 미래 k ∈ {arc (가능), rise, float, line, stop}.
  (2026-10-07 20:36 데이터 재렌더로 stop (그 자리에서 멈춤) 이 더해졌다 — 옛 넷 판 결과는 exp_results/_previous_20261007_4futures/)
  s_k = mean_{토큰, 채널} |p − LN(h_k)|   (표준 surprise_c16t32 채점값, per_block.json)
  고른 미래 = argmin_k s_k  (p 가 가장 가까운 미래 = "p 가 만드는 미래" 의 표면 측도). chance = 1 / 미래 수 (다섯이면 20 %)
  쌍 정확도 (가능 vs 불가능 하나) = [s_imp > s_arc], 상대 차 = (s_imp − s_arc) / s_arc
CI = block 부트스트랩 95 % (2,000 회). 조건 = 꼭대기 위치 (arc_pre f57 · arc_apex f48 · arc_post f42).

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/gravity_retrieval.py [--split]   → z_research/v11_realistic/exp_results/gravity_retrieval.{md,json}
     (--split: §7 arc vs float 를 공 · 그림자 · 나머지 토큰으로 나눔, 프레임 읽기 몇 분)
"""
from __future__ import annotations
import collections, csv, json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "z_research/v11_realistic/exp_results/surprise_c16t32__gravity_realistic_vith"
IDX = REPO / "data_csv/intphysgen_gravity_realistic/index.csv"
META = Path("/local_datasets/world/world_analysis/IntPhysGen_gravity_realistic/metadata.csv")
OUT = REPO / "z_research/v11_realistic/exp_results"
FUT = ["pos_arc", "imp_rise", "imp_float", "imp_line", "imp_stop"]
LAB = {"pos_arc": "arc (possible)", "imp_rise": "rise", "imp_float": "float", "imp_line": "line", "imp_stop": "stop"}
CONDS = ["arc_pre", "arc_apex", "arc_post"]
RNG = np.random.default_rng(0)


def boot(x, f=np.mean, n=2000):
    x = np.asarray(x, float)
    b = [f(x[RNG.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(f(x)), *np.percentile(b, [2.5, 97.5])


def main():
    S = json.loads((RUN / "per_block.json").read_text())["per_video_surprise"]
    want = json.loads((RUN / "summary.json").read_text())["surprise"]["overall"]["block_pairwise"]
    meta = {r["name"]: r for r in csv.DictReader(META.open())}
    B = collections.defaultdict(dict)
    for r in csv.DictReader(IDX.open()):
        B[r["block_id"]][r["variant"]] = r
    rows = []
    for b, v in B.items():
        fut = [k for k in FUT if k in v]
        s = np.array([S[v[k]["video_id"]] for k in fut])
        m = meta[v["pos_arc"]["video_id"]]
        rows.append(dict(block=b, cond=v["pos_arc"]["condition"], s=s, pick=fut[int(np.argmin(s))],
                         shape=m["shape_pre"], env=m["env"], dir=m["travel_direction"]))
    # 검증: 쌍 정확도 평균 = summary.json (cross = 가능 × 불가능 3)
    NF = len(rows[0]["s"]); FUT[:] = FUT[:NF]
    got = np.mean([[r["s"][i] > r["s"][0] for i in range(1, NF)] for r in rows])
    if abs(got - want) > 1e-9:
        sys.exit(f"검증 실패: 재계산 {got:.6f} != summary {want:.6f}")
    md, js = [], {"n_block": len(rows), "pairwise_overall": got}
    md += [f"# gravity_realistic — p 가 고르는 미래 (block {len(rows)}, 미래 {NF}, chance {100 / NF:.0f} %)", "",
           f"쌍 정확도 전체 {100 * got:.1f} % ({len(rows) * (NF - 1)} 쌍, summary.json 과 일치 검증) · [ ] = block 부트스트랩 95 % CI", "",
           "## 1. 고른 미래 (argmin surprise) 비율 %", "", "| 조건 | n | " + " | ".join(LAB[k] for k in FUT) + " |", "|---|---:|" + "---:|" * NF]
    js["n_future"] = NF
    js["pick"] = {}
    for c in CONDS + ["all"]:
        R = [r for r in rows if c == "all" or r["cond"] == c]
        cells = []
        js["pick"][c] = {}
        for k in FUT:
            m, lo, hi = boot([r["pick"] == k for r in R]); js["pick"][c][k] = [m, lo, hi]
            cells.append(f"{100 * m:.1f} [{100 * lo:.0f}, {100 * hi:.0f}]")
        md.append(f"| `{c}` | {len(R)} | " + " | ".join(cells) + " |")
    md += ["", "## 2. 쌍 정확도 (가능 vs 불가능 하나) % · 상대 차 (s_imp − s_arc) / s_arc 중앙 %", "",
           "| 조건 | " + " | ".join(f"vs {LAB[k]}" for k in FUT[1:]) + " |", "|---|" + "---:|" * (NF - 1)]
    js["pairwise"] = {}
    for c in CONDS + ["all"]:
        R = [r for r in rows if c == "all" or r["cond"] == c]
        cells = []; js["pairwise"][c] = {}
        for i, k in enumerate(FUT[1:], start=1):
            m, lo, hi = boot([r["s"][i] > r["s"][0] for r in R]); g = np.median([100 * (r["s"][i] - r["s"][0]) / r["s"][0] for r in R])
            js["pairwise"][c][k] = dict(acc=[m, lo, hi], rel_gap_median_pct=float(g))
            cells.append(f"{100 * m:.1f} [{100 * lo:.0f}, {100 * hi:.0f}] · {g:+.2f}")
        md.append(f"| `{c}` | " + " | ".join(cells) + " |")
    md += ["", "## 3. 순위 — 미래마다 평균 순위 (1 = p 에 가장 가까움)", "", "| 조건 | " + " | ".join(LAB[k] for k in FUT) + " |", "|---|" + "---:|" * NF]
    js["rank"] = {}
    for c in CONDS + ["all"]:
        R = [r for r in rows if c == "all" or r["cond"] == c]
        rk = np.array([np.argsort(np.argsort(r["s"])) + 1 for r in R])
        js["rank"][c] = dict(zip(FUT, rk.mean(0).tolist()))
        md.append(f"| `{c}` | " + " | ".join(f"{x:.2f}" for x in rk.mean(0)) + " |")
    md += ["", "## 4. 고른 미래 — 방향 · 물체별 (조건 합침)", "", "| 묶음 | n | " + " | ".join(LAB[k] for k in FUT) + " |", "|---|---:|" + "---:|" * NF]
    for key in ("dir", "shape"):
        for val in sorted({r[key] for r in rows}):
            R = [r for r in rows if r[key] == val]
            md.append(f"| {key}={val} | {len(R)} | " + " | ".join(f"{100 * np.mean([r['pick'] == k for r in R]):.0f}" for k in FUT) + " |")

    # 5. 위치 기준선 — p 가 그 규칙대로 미래를 만든다면 네 미래 중 무엇을 골랐을까 (metadata 궤적, 2026-10-07 18:07 재조립판)
    #    문맥 끝 속도 = 샘플 15 − 14 (px/샘플). 미래 샘플 16..31 에서 평균 위치 거리 최소인 미래.
    #    ⚠️ metadata 는 계획 위치다 — 렌더 위치와 최대 6.5–13.7 px 어긋난다 (생성 로그 pixel check). 미래끼리의 차는 수십 px 라 고르는 결과는 안 바뀐다고 봤다
    md += ["", "## 5. 위치 기준선 — 그 규칙이면 고를 미래 (block 수)", "",
           "| 조건 | 문맥 끝 수직 속도 px/샘플 (− = 위) | 등속 (CV) | 높이 유지 (수평 CV + 수직 0) | 그 자리 멈춤 | p (surprise) |", "|---|---:|---|---|---|---|"]
    def xy(v):
        r = meta[v["video_id"]]
        return np.stack([np.array(r["object_px_x_by_sample"].split(), float), np.array(r["object_px_y_by_sample"].split(), float)], 1)
    js["baseline"] = {}
    for c in CONDS:
        cnt = {"cv": collections.Counter(), "hold": collections.Counter(), "frozen": collections.Counter()}; vz = []
        for b, v in B.items():
            if v["pos_arc"]["condition"] != c:
                continue
            P0 = xy(v["pos_arc"]); vel = P0[15] - P0[14]; t = np.arange(1, 17)[:, None]; vz.append(vel[1])
            # 높이 유지 = 연결 지점 (f48 = 샘플 15 + 한 걸음) 의 높이를 유지하며 수평만 등속. float 의 정의와 같다.
            # ⚠️ 2026-10-07 정정 — 처음엔 샘플 15 (f45) 높이로 잡아, 문맥 끝 수직 속도가 큰 재렌더판 (−6 px/샘플) 에서 한 걸음 차가 판정을 뒤집었다
            hold = np.stack([P0[15][0] + t[:, 0] * vel[0], np.full(16, P0[15][1] + vel[1])], 1)
            for nm, ref in (("cv", P0[15] + t * vel), ("hold", hold), ("frozen", np.repeat(P0[15][None], 16, 0))):
                d = [np.linalg.norm(xy(v[k])[16:] - ref, axis=1).mean() for k in FUT]
                cnt[nm][FUT[int(np.argmin(d))]] += 1
        pk = collections.Counter(r["pick"] for r in rows if r["cond"] == c)
        f = lambda C: " · ".join(f"{LAB[k]} {C[k]}" for k in FUT if C[k])
        js["baseline"][c] = dict(vz=float(np.mean(vz)), cv=dict(cnt["cv"]), hold=dict(cnt["hold"]), frozen=dict(cnt["frozen"]), p=dict(pk))
        md.append(f"| `{c}` | {np.mean(vz):+.1f} | {f(cnt['cv'])} | {f(cnt['hold'])} | {f(cnt['frozen'])} | {f(pk)} |")

    # 6. 튜블릿별 — 미래 튜블릿 t 마다 p 가 가장 가까운 미래 (token_surprise.npz, 물체 토큰 = 미래 전부의 물체 자리 합집합 3×3)
    #    합 (§1) 이 어느 시점에서 정해지는지 본다 (2026-10-08 사용자 질문 "오르는 중에 왜 arc 가 아니라 float 인가")
    tok = RUN / "token_surprise.npz"
    if tok.exists():
        sys.path.insert(0, str(REPO / "z_research/scripts/figures"))
        from gif_token_surprise import obj_mask
        z = np.load(tok); E = dict(zip(z["video_id"], z["e"]))
        md += ["", "## 6. 튜블릿별 — p 가 가장 가까운 미래 % (전체 토큰 / 물체 토큰)", ""]
        js["per_tubelet"] = {}
        for c in CONDS:
            bl = [v for v in B.values() if v["pos_arc"]["condition"] == c]
            pa = np.zeros((8, NF)); po = np.zeros((8, NF))
            for v in bl:
                e = np.stack([E[v[k]["video_id"]] for k in FUT]); M = np.zeros((8, 16, 16), bool)
                for k in FUT:
                    M |= obj_mask(v[k], meta)
                for t in range(8):
                    pa[t, np.argmin(e[:, t].reshape(NF, -1).mean(1))] += 1
                    po[t, np.argmin(e[:, t][:, M[t]].mean(1))] += 1
            pa, po = 100 * pa / len(bl), 100 * po / len(bl)
            # 공 칸만으로 고른 미래 (16 샘플 · 8 튜블릿 전체를 합쳐 argmin, 공 칸 = 다섯 미래 물체 자리 합집합) + block 부트스트랩 CI
            ball_pick = []
            for v in bl:
                e = np.stack([E[v[k]["video_id"]] for k in FUT]); M = np.zeros((8, 16, 16), bool)
                for k in FUT:
                    M |= obj_mask(v[k], meta)
                ball_pick.append(FUT[int(np.argmin(e[:, M].mean(1)))])
            js.setdefault("pick_ball", {})[c] = {k: list(boot([x == k for x in ball_pick])) for k in FUT}
            js["per_tubelet"][c] = dict(all=pa.tolist(), obj=po.tolist())
            md += [f"`{c}` (n={len(bl)})", "", "| t | " + " | ".join(LAB[k] for k in FUT) + " |", "|---|" + "---:|" * NF]
            for t in range(8):
                md.append(f"| t{t} | " + " | ".join(f"{a:.0f} / {o:.0f}" for a, o in zip(pa[t], po[t])) + " |")
            pb = js["pick_ball"][c]
            md.append("공 칸만으로 고른 미래 (8 튜블릿 합): " + " · ".join(f"{LAB[k]} {100 * pb[k][0]:.0f}" for k in FUT if pb[k][0] > 0))
            md.append("")

    # 7. arc vs float 차를 공 자리 / 그림자 / 나머지 로 나눈다 (--split, 프레임을 읽어 몇 분) — 2026-10-08
    #    공 = obj_mask (arc ∪ float), 그림자 = 두 미래 프레임 픽셀 최대 차 > 16 (256 입력) 인데 공 밖, 나머지 = 픽셀이 거의 같은 칸
    if "--split" in sys.argv and tok.exists():
        import torch
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor
        def load256(pth):
            x = torch.from_numpy(np.array(Image.open(pth).convert("RGB"))).permute(2, 0, 1)[None].float()
            return torch.nn.functional.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=False)[0].numpy()
        def work(v):
            Pd = np.zeros((8, 16, 16))
            for t in range(8):
                for q in (16 + 2 * t, 17 + 2 * t):
                    d = np.abs(load256(META.parent / v["pos_arc"]["file_name"] / f"{3 * q:06d}.png") - load256(META.parent / v["imp_float"]["file_name"] / f"{3 * q:06d}.png")).max(0)
                    Pd[t] = np.maximum(Pd[t], d.reshape(16, 16, 16, 16).max((1, 3)))
            Mo = obj_mask(v["pos_arc"], meta) | obj_mask(v["imp_float"], meta); Ms = (Pd > 16) & ~Mo
            d = E[v["imp_float"]["video_id"]] - E[v["pos_arc"]["video_id"]]
            return v["pos_arc"]["condition"], [d[m].sum() / d.size for m in (Mo, Ms, ~Mo & ~Ms)]
        with ThreadPoolExecutor(48) as ex:
            SP = list(ex.map(work, B.values()))
        md += ["## 7. arc vs float — p 가 arc 쪽인 block % (공 자리 / 그림자 / 나머지, 기여 ×1e3)", "",
               "| 조건 | 전체 | 공 자리 | 그림자 | 나머지 (픽셀 ≤ 16) |", "|---|---:|---:|---:|---:|"]
        js["arc_vs_float_split"] = {}
        for c in CONDS:
            a = np.array([x[1] for x in SP if x[0] == c]); tot = a.sum(1)
            js["arc_vs_float_split"][c] = dict(total=float(tot.mean()), ball=float(a[:, 0].mean()), shadow=float(a[:, 1].mean()), rest=float(a[:, 2].mean()))
            md.append(f"| `{c}` | {100 * (tot > 0).mean():.0f} ({1e3 * tot.mean():+.2f}) | {100 * (a[:, 0] > 0).mean():.0f} ({1e3 * a[:, 0].mean():+.2f}) | "
                      f"{100 * (a[:, 1] > 0).mean():.0f} ({1e3 * a[:, 1].mean():+.2f}) | {100 * (a[:, 2] > 0).mean():.0f} ({1e3 * a[:, 2].mean():+.2f}) |")
    (OUT / "gravity_retrieval.md").write_text("\n".join(md) + "\n")
    (OUT / "gravity_retrieval.json").write_text(json.dumps(js, indent=1))
    print("\n".join(md)); print(f"\n→ {OUT}/gravity_retrieval.md")


if __name__ == "__main__":
    main()
