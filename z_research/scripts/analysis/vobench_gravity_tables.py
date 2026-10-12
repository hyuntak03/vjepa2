#!/usr/bin/env python3
"""VOBench_v1 gravity 채점표 (2026-10-10). `cvpr/eval/testbed` 실행 (cross 채점) 의 per_block.json + token_surprise.npz 에서 다시 계산한다.

block = 과거 하나 + 미래 여럿 (f0–47 공유). s_k = mean_{토큰, 채널} |p − LN(h_k)| (표준 surprise).
  ledge : 쌍 정확도 = 1[s_float > s_pos] (+0.5 동점). chance 50 %
  arc   : 고른 미래 = argmin_k s_k over 9 미래 (pos · rise · float · line · stop · stop2 · stop4 · stop8 · stop12). chance 11 %
          "p 가 가장 가까운 미래" 를 predictor 가 그린 미래의 표면 측도로 읽는다
  두 가지 토큰 범위:  full = 미래 8 튜블릿 전부 (= 채점값) · post = 두 프레임이 모두 f48 이후인 튜블릿만 (sliding 창에서는
          미래 앞쪽이 아직 공유 구간이다 — 그 튜블릿의 h 차이는 target encoder 의 전역 attention 이 만든 것이라 뺀다)
  복사  : ec = |LN(z)[문맥 마지막 튜블릿] − LN(h_k)| (token_surprise_maps.py --copy) 로 같은 고르기 · 같은 쌍 정확도
  위치 규칙 기준선: label.json 의 물체 중심 (u, v) 로, 문맥 끝 두 샘플에서 (등속 · 높이 유지 · 그 자리 멈춤) 을 이어 그렸을 때
          미래 샘플 (post) 평균 거리로 가장 가까운 후보 — "그 규칙으로 그리면 무엇을 고르나"
검증: 재계산 쌍 정확도 (block 평균 · 쌍 평균) 가 summary.json 과 다르면 죽는다. 토큰 지도는 token_surprise_maps.py 가 이미 검증.
CI = block 부트스트랩 95 % (2,000 회). **primitive · realistic 은 따로** (VOBench 관례), 합침은 참고.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/vobench_gravity_tables.py [--model vjepa2|ariel_ar_ep40]    # → z_research/VOBench_v1/<model>/gravity/tables.{md,json}
"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "z_research/VOBench_v1/exp_results"
# 모델: 이름 → (라벨, 레지스트리 모델 이름). 정리본 = z_research/VOBench_v1/<이름>/gravity/ (2026-10-11 --model 추가)
MODELS = {"vjepa2": ("V-JEPA 2 ViT-H", "vith"),
          "ariel_ar_ep40": ("Ariel AR predictor ep40 (V-JEPA 2 ViT-H frozen; 타깃 = 블록별 LN(target_encoder), 복사 = copy_blk)", "vith_ariel_ar_ep40")}
_ap = __import__("argparse").ArgumentParser()
_ap.add_argument("--model", default="vjepa2", choices=list(MODELS))
MODEL = _ap.parse_args().model
MLABEL, MREG = MODELS[MODEL]
OUTD = REPO / "z_research/VOBench_v1" / MODEL / "gravity"
INDEX = REPO / "data_csv/vobench_v1_gravity/index.csv"
FR = Path("/data2/local_datasets/world/Benchmark/VOBench_v1/gravity")
SPLICE = 48
# 창: 이름 → 실행 폴더. 표준 = c16t32 (stride 3, 문맥 끝 f45). sliding = stride 2 · 시작 o → 문맥 끝 E = o + 30
WINDOWS = {"c16t32": EXP / f"surprise_c16t32__vobench_v1_gravity_{MREG}"}
for o in (0, 3, 6, 9, 12, 15):
    WINDOWS[f"s2o{o}"] = EXP / f"surprise_c16t32__vobench_v1_gravity_{MREG}_wc16t32s2o{o}"
ARC = ["pos", "imp_rise", "imp_float", "imp_line", "imp_stop", "imp_stop2", "imp_stop4", "imp_stop8", "imp_stop12"]
LAB = {"pos": "arc (possible)", "imp_rise": "rise", "imp_float": "float", "imp_line": "line", "imp_stop": "stop",
       "imp_stop2": "stop2", "imp_stop4": "stop4", "imp_stop8": "stop8", "imp_stop12": "stop12"}
LET = {"pos": "P", "imp_rise": "R", "imp_float": "F", "imp_line": "L", "imp_stop": "S", "imp_stop2": "S2", "imp_stop4": "S4",
       "imp_stop8": "S8", "imp_stop12": "S12"}
OBJ = ["primitive", "realistic"]
RNG = np.random.default_rng(0)


def boot(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * 3
    b = x[RNG.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(x.mean()), *map(float, np.percentile(b, [2.5, 97.5]))]


def pct(c):
    return f"{100 * c[0]:.1f} [{100 * c[1]:.0f}, {100 * c[2]:.0f}]"


def window_frames(run: Path):
    d = yaml.safe_load((run / "_resolved.yaml").read_text())["data"]
    st, sr, n = int(d.get("frames_start", 0)), int(d.get("frames_stride", 1)), int(d["n_frames"])
    fr = [st + sr * k for k in range(n)]
    tub = [(fr[16 + 2 * t], fr[17 + 2 * t]) for t in range(8)]
    post = [t for t in range(8) if tub[t][0] >= SPLICE]
    return dict(start=st, stride=sr, E=fr[15], ctx_prev=fr[14], future=fr[16:], tubelets=tub, post=post)


def load_run(run: Path, rows, blocks):
    S = {k: float(v) for k, v in json.loads((run / "per_block.json").read_text())["per_video_surprise"].items()}
    summ = json.loads((run / "summary.json").read_text())["surprise"]["overall"]
    # 검증: cross 쌍 정확도
    hits = {b: [float(S[n["video_id"]] > S[p["video_id"]]) + 0.5 * (S[n["video_id"]] == S[p["video_id"]])
                for p in v if p["plausible"] == "1" for n in v if n["plausible"] == "0"] for b, v in blocks.items()}
    got_block = float(np.mean([np.mean(h) for h in hits.values()]))
    got_pair = float(np.mean([x for h in hits.values() for x in h]))
    want = summ.get("block_pairwise")
    if want is None or min(abs(got_block - want), abs(got_pair - want)) > 1e-9:
        sys.exit(f"검증 실패 {run.name}: 재계산 block {got_block:.6f} / pair {got_pair:.6f} != summary {want}")
    z = np.load(run / "token_surprise.npz")
    vi = {v: i for i, v in enumerate(z["video_id"])}
    return S, z["e"], z["ec"] if "ec" in z else None, vi, dict(block_pairwise=got_block, pair_mean=got_pair)


def main():
    rows = list(csv.DictReader(INDEX.open()))
    blocks = defaultdict(list)
    for r in rows:
        blocks[r["block_id"]].append(r)
    binfo = {b: dict(objset=v[0]["objset"], law=v[0]["law"], cell=v[0]["cell"], src=v[0]["source_block"],
                     path=str(Path(v[0]["file_name"]).parent), apex=v[0]["apex_frame"], arms={r["variant"]: r["video_id"] for r in v})
             for b, v in blocks.items()}
    lab = {b: json.loads((FR / i["path"] / "label.json").read_text())["labels"] for b, i in binfo.items()}
    uv = {b: {k: np.array([[e["u"], e["v"]] for e in L[k]]) for k in L} for b, L in lab.items()}   # 프레임별 (96, 2)

    js = {"definition": __doc__.split("\n\n")[1], "windows": {}}
    md = [f"# VOBench_v1 gravity — {MLABEL} 채점표", "", f"실행 `{WINDOWS['c16t32'].relative_to(REPO)}`.", "",
          "재계산 쌍 정확도가 창마다 summary.json 과 일치 (스크립트 검증). [ ] = block 부트스트랩 95 % CI. **primitive · realistic 따로.**",
          "칸 = 고른 비율 % (argmin surprise). full = 미래 8 튜블릿 전부 (= 채점값) · post = 두 프레임이 모두 f48 이후인 튜블릿만.", ""]
    avail = {w: r for w, r in WINDOWS.items() if (r / "token_surprise.npz").exists()}
    if "c16t32" not in avail:
        sys.exit("표준 창 c16t32 의 token_surprise.npz 가 없다")
    for w, run in avail.items():
        W = window_frames(run)
        S, E_, EC, vi, chk = load_run(run, rows, blocks)
        post = W["post"]
        res = {"frames": W, "check": chk, "blocks": {}}
        for b, i in binfo.items():
            arms = ARC if i["law"] == "arc" else ["pos", "imp_float"]
            ix = [vi[i["arms"][a]] for a in arms]
            e, sfull = E_[ix], np.array([S[i["arms"][a]] for a in arms])
            spost = e[:, post].mean((1, 2, 3))
            # 공 칸: post 튜블릿마다 후보 전부의 공 중심 토큰 3×3 합집합 (512 px → 16 토큰 = 32 px/토큰)
            M = np.zeros((8, 16, 16), bool)
            for t in post:
                for a in arms:
                    for f in W["tubelets"][t]:
                        u_, v_ = uv[b][LET[a]][f]
                        i_, j_ = int(np.clip(v_ // 32, 0, 15)), int(np.clip(u_ // 32, 0, 15))
                        M[t, max(0, i_ - 1):i_ + 2, max(0, j_ - 1):j_ + 2] = True
            sball = e[:, M].mean(1)
            d = dict(arms=arms, s_full=sfull.tolist(), s_post=spost.tolist(),
                     pick_full=arms[int(np.argmin(sfull))], pick_post=arms[int(np.argmin(spost))],
                     s_ball=sball.tolist(), pick_ball=arms[int(np.argmin(sball))], ball_frac=float(M[post].mean()),
                     pick_tub=[arms[int(np.argmin(e[:, t].mean((1, 2))))] for t in range(8)])
            if EC is not None:
                ec = EC[ix]
                cf, cp = ec.mean((1, 2, 3)), ec[:, post].mean((1, 2, 3))
                d.update(c_full=cf.tolist(), c_post=cp.tolist(), cpick_full=arms[int(np.argmin(cf))], cpick_post=arms[int(np.argmin(cp))])
            # 위치 규칙 기준선 (post 미래 샘플)
            U = uv[b]; P = U["P"]; fut = [f for f in W["future"] if f >= SPLICE]
            vel = (P[W["E"]] - P[W["ctx_prev"]]) / W["stride"]
            rules = {"const_vel": np.array([P[W["E"]] + vel * (f - W["E"]) for f in fut]),
                     "hold_height": np.array([[P[W["E"]][0] + vel[0] * (f - W["E"]), P[W["E"]][1]] for f in fut]),
                     "stop_here": np.array([P[W["E"]] for f in fut])}
            dist = {a: [float(np.linalg.norm(U[LET[a]][fut] - rr, axis=1).mean()) for rr in rules.values()] for a in arms}
            d["rule_pick"] = {k: min(arms, key=lambda a: dist[a][j]) for j, k in enumerate(rules)}
            d["sep_px"] = {a: float(np.linalg.norm(U[LET[a]][fut] - P[fut], axis=1).mean()) for a in arms}   # pos 와의 평균 거리 (구별 가능성)
            res["blocks"][b] = d
        js["windows"][w] = res

    def sel(w, law, objset=None, cell=None):
        return [(b, js["windows"][w]["blocks"][b]) for b, i in binfo.items()
                if i["law"] == law and (objset is None or i["objset"] == objset) and (cell is None or i["cell"] == cell)]

    # arc 셀은 꼭대기 프레임 내림차순 = 문맥이 오르는 중 → 꼭대기 → 내려가는 중 (apexNN = 꼭대기 프레임)
    ACELLS = sorted({i["cell"] for i in binfo.values() if i["law"] == "arc"}, key=lambda c: -int(c[4:]))

    def phase(rel):
        return "오르는 중" if rel < -3 else "꼭대기" if rel <= 3 else "내려가는 중"     # ±3 프레임 = 표준 창 한 샘플

    def groups(law):
        cells = ACELLS if law == "arc" else sorted({i["cell"] for i in binfo.values() if i["law"] == law})
        for o in OBJ:
            for c in cells:
                yield f"{o} · {c}", o, c
            yield f"**{o} 전체**", o, None
        yield "(참고) 합침", None, None

    # ── 1. ledge ──
    md += ["## 1. ledge — 낙하 (가능) vs 같은 높이 직진 (float), 쌍 정확도 %", "",
           "| 창 (문맥 끝) | 묶음 | n | 모델 | 복사 | Δ 모델 − 복사 | 상대 차 중앙 (s_float − s_pos)/s_pos % |", "|---|---|---:|---:|---:|---:|---:|"]
    js["ledge"] = {}
    for w in avail:
        Wf = js["windows"][w]["frames"]
        for name, o, c in groups("ledge"):
            R = sel(w, "ledge", o, c)
            m = [float(d["s_full"][1] > d["s_full"][0]) for _, d in R]
            cp = [float(d["c_full"][1] > d["c_full"][0]) for _, d in R] if "c_full" in R[0][1] else []
            dl = boot(np.array(m) - np.array(cp)) if cp else [np.nan] * 3
            g = float(np.median([100 * (d["s_full"][1] - d["s_full"][0]) / d["s_full"][0] for _, d in R]))
            js["ledge"][f"{w}|{name}"] = dict(n=len(R), model=boot(m), copy=boot(cp) if cp else None, delta=dl, rel_gap_median_pct=g)
            if w == "c16t32" or c is None:
                md.append(f"| `{w}` (f{Wf['E']}) | {name} | {len(R)} | {pct(boot(m))} | {pct(boot(cp)) if cp else '—'} | "
                          f"{100 * dl[0]:+.1f} [{100 * dl[1]:+.0f}, {100 * dl[2]:+.0f}] | {g:+.2f} |")
    # ── 2. arc 표준 창 ──
    W0 = js["windows"]["c16t32"]["frames"]
    md += ["", f"## 2. arc — 표준 창 c16t32 (stride 3, 문맥 끝 f{W0['E']}, 미래 f{W0['future'][0]}–{W0['future'][-1]}): 고른 미래 % (chance 11 %)", "",
           " · ".join(f"`{c}` = 꼭대기 f{c[4:]} (문맥 끝 − 꼭대기 {W0['E'] - int(c[4:]):+d} 프레임, 문맥 끝 = **{phase(W0['E'] - int(c[4:]))}**)"
                      for c in ACELLS) + ".", ""]
    js["arc_pick"] = {}
    for key, title in (("pick_full", "모델 · full (= 채점값)"), ("pick_post", "모델 · post"),
                       ("pick_ball", "모델 · 공 칸만 (post 튜블릿, 후보 9 개 공 자리 합집합 3×3)"), ("cpick_full", "복사 · full")):
        md += [f"### {title}", "", "| 묶음 | n | " + " | ".join(LAB[a] for a in ARC) + " |", "|---|---:|" + "---:|" * len(ARC)]
        for name, o, c in groups("arc"):
            R = sel("c16t32", "arc", o, c)
            cells = {a: boot([d[key] == a for _, d in R]) for a in ARC}
            js["arc_pick"][f"c16t32|{key}|{name}"] = cells
            md.append(f"| {name} | {len(R)} | " + " | ".join(f"{100 * cells[a][0]:.0f}" for a in ARC) + " |")
        md.append("")
    # 쌍 정확도
    md += ["### 쌍 정확도 (가능 vs 불가능 하나) % — 모델 / 복사", "", "| 묶음 | " + " | ".join(f"vs {LAB[a]}" for a in ARC[1:]) + " |",
           "|---|" + "---:|" * (len(ARC) - 1)]
    js["arc_pairwise"] = {}
    for name, o, c in groups("arc"):
        R = sel("c16t32", "arc", o, c)
        cells = []
        for k in range(1, len(ARC)):
            m = boot([d["s_full"][k] > d["s_full"][0] for _, d in R]); cp = boot([d["c_full"][k] > d["c_full"][0] for _, d in R])
            js["arc_pairwise"][f"{name}|{ARC[k]}"] = dict(model=m, copy=cp)
            cells.append(f"{100 * m[0]:.0f} / {100 * cp[0]:.0f}")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    # 튜블릿별
    md += ["", "### 튜블릿별 — 튜블릿 t 에서 p 가 가장 가까운 미래 % (전체 토큰, 표준 창)", ""]
    for o in OBJ:
        for c in ACELLS:
            R = sel("c16t32", "arc", o, c)
            md += [f"`{o} · {c}` (n={len(R)})", "", "| t (프레임) | " + " | ".join(LAB[a] for a in ARC) + " |", "|---|" + "---:|" * len(ARC)]
            for t in range(8):
                fr = W0["tubelets"][t]
                md.append(f"| t{t} (f{fr[0]}–{fr[1]}) | " + " | ".join(f"{100 * np.mean([d['pick_tub'][t] == a for _, d in R]):.0f}" for a in ARC) + " |")
            md.append("")
    # 튜블릿 묶음 — 앞 (t0–t2, f48–63: arc 와 stop12 가 f60 까지 픽셀이 같다) · 뒤 (t3–t7, f66–93)
    KIND = {"pos": "arc", "imp_rise": "rise", "imp_float": "float · line", "imp_line": "float · line"}
    md += ["### 튜블릿 묶음 — 앞 (t0–t2) · 뒤 (t3–t7) 에서 p 가 가장 가까운 후보의 종류 % (전체 토큰, 튜블릿 × block 평균)", "",
           "멈춤 = stop · stop2 · stop4 · stop8 · stop12. 앞 묶음에서 arc 와 늦은 멈춤 (stop8 · stop12) 은 f56 / f60 까지 픽셀이 같아 갈리지 않는다.", "",
           "| 묶음 | n | 앞: arc | 앞: rise | 앞: float · line | 앞: 멈춤 | 뒤: arc | 뒤: rise | 뒤: float · line | 뒤: 멈춤 |", "|---|---:|" + "---:|" * 8]
    js["tubelet_groups"] = {}
    for name, o, c in groups("arc"):
        R = sel("c16t32", "arc", o, c); row = {}
        for gname, ts in (("early", range(0, 3)), ("late", range(3, 8))):
            ks = [KIND.get(d["pick_tub"][t], "freeze") for _, d in R for t in ts]
            row[gname] = {k: float(np.mean([x == k for x in ks])) for k in ("arc", "rise", "float · line", "freeze")}
        js["tubelet_groups"][name] = row
        md.append(f"| {name} | {len(R)} | " + " | ".join(f"{100 * row[g][k]:.0f}" for g in ("early", "late")
                                                         for k in ("arc", "rise", "float · line", "freeze")) + " |")
    md.append("")
    # 위치 규칙
    md += ["### 위치 규칙 기준선 — 문맥 끝 두 샘플에서 그 규칙으로 그리면 가장 가까운 후보 (block 수) · 모델 post", "",
           "| 묶음 | n | 등속 이어 가기 | 높이 유지 (수평만) | 그 자리 멈춤 | **모델 (post)** |", "|---|---:|---|---|---|---|"]

    def tally(xs):
        u, n = np.unique(xs, return_counts=True)
        return " · ".join(f"{LAB[a]} {k}" for a, k in sorted(zip(u, n), key=lambda z: -z[1]))
    for name, o, c in groups("arc"):
        R = sel("c16t32", "arc", o, c)
        md.append(f"| {name} | {len(R)} | " + " | ".join(tally([d["rule_pick"][k] for _, d in R]) for k in ("const_vel", "hold_height", "stop_here"))
                  + f" | {tally([d['pick_post'] for _, d in R])} |")
    md += ["", "후보가 pos 에서 떨어진 정도 (post 미래 샘플 평균 px, 512 px 화면): " +
           " · ".join(f"{LAB[a]} {np.mean([d['sep_px'][a] for _, d in sel('c16t32', 'arc')]):.0f}" for a in ARC[1:]), ""]
    # ── 3. sliding ──
    sl = [w for w in avail if w != "c16t32"]
    if sl:
        md += ["## 3. sliding — stride 2 창 (c16 / p16, 32 샘플 = 62 프레임), 시작을 밀어 문맥 끝 E 를 바꾼다", "",
               "E − apex < 0 = 문맥이 **오르는 중** 에 끝남 · 0 = 꼭대기 · > 0 = 내려가는 중. 고른 미래는 **post** (갈린 뒤 튜블릿만). "
               "미래 창이 E+32 에서 끝나 stop8 · stop12 는 창 안에서 pos 와 거의 같을 수 있다 (`sep_px` 열).", ""]
        js["sliding"] = {}
        for o in OBJ:
            md += [f"### {o}", "", "| 창 | E | 셀 | E − apex | post 튜블릿 | " + " | ".join(LAB[a] for a in ARC) + " | 복사 고름 (post) | ledge 쌍 정확도 |",
                   "|---|---:|---|---:|---:|" + "---:|" * len(ARC) + "---|---:|"]
            rows_ = []
            for w in sl + ["c16t32"]:
                Wf = js["windows"][w]["frames"]
                for c in ACELLS:
                    R = sel(w, "arc", o, c)
                    ap = int(c[4:]); rel = Wf["E"] - ap
                    cells = {a: float(np.mean([d["pick_post"] == a for _, d in R])) for a in ARC}
                    js["sliding"][f"{o}|{w}|{c}"] = dict(E=Wf["E"], rel=rel, pick_post=cells,
                                                         cpick_post={a: float(np.mean([d["cpick_post"] == a for _, d in R])) for a in ARC},
                                                         sep_px={a: float(np.mean([d["sep_px"][a] for _, d in R])) for a in ARC})
                    lg = np.mean([d["s_full"][1] > d["s_full"][0] for _, d in sel(w, "ledge", o)])
                    rows_.append((rel, f"| `{w}` | f{Wf['E']} | {c} | {rel:+d} | {len(Wf['post'])} | " + " | ".join(f"{100 * cells[a]:.0f}" for a in ARC)
                                  + f" | {tally([d['cpick_post'] for _, d in R])} | {100 * lg:.0f} |"))
            md += [r for _, r in sorted(rows_, key=lambda z: z[0])] + [""]
    # ── 4. 멈춤 후보를 문맥 끝 기준으로 — stop 계열은 고정 프레임 (FREEZE) 에서 멈추므로 E 가 바뀌면 뜻이 바뀐다.
    #    D = 멈춘 프레임 − E = 문맥 끝 뒤 진짜 운동을 따라간 프레임 수. 고른 후보를 D 로 다시 센다
    FREEZE = {"imp_stop": 45, "imp_stop2": 50, "imp_stop4": 52, "imp_stop8": 56, "imp_stop12": 60}
    md += ["## 4. 고른 미래를 문맥 끝 기준으로 — D = 멈춘 프레임 − 문맥 끝 E (진짜 운동을 따라간 프레임 수)", "",
           "stop · stop2 · stop4 · stop8 · stop12 는 f45 · 50 · 52 · 56 · 60 에서 멈춘다 — 창마다 D 가 다르다. "
           "칸 = 고른 block 수 (post), `D=` 는 그 후보의 D. 마지막 열 = 고른 멈춤 후보의 D 중앙 (프레임 · 샘플).", ""]
    js["freeze_delay"] = {}
    for o in OBJ:
        md += [f"### {o}", "", "| 창 | E − apex | 셀 | 그 창의 멈춤 후보 D | 고른 것 | 멈춤 아닌 것 고름 | D 중앙 (프레임 / 샘플) | 공 칸: D 중앙 · 멈춤 아닌 것 |",
               "|---|---:|---|---|---|---:|---|---|"]
        rows_ = []
        for w in sl + ["c16t32"]:  # 표준 창은 늘 포함
            Wf = js["windows"][w]["frames"]
            for c in ACELLS:
                R = sel(w, "arc", o, c); rel = Wf["E"] - int(c[4:])
                Ds = sorted((FREEZE[a] - Wf["E"], a) for a in FREEZE)
                picks = [d["pick_post"] for _, d in R]
                dpk = [FREEZE[p] - Wf["E"] for p in picks if p in FREEZE]
                bpk = [d["pick_ball"] for _, d in R]
                dball = [FREEZE[p] - Wf["E"] for p in bpk if p in FREEZE]
                bnon = " · ".join(f"{LAB[p]} {bpk.count(p)}" for p in dict.fromkeys(bpk) if p not in FREEZE)
                non = sum(p not in FREEZE for p in picks)
                js["freeze_delay"][f"{o}|{w}|{c}"] = dict(E=Wf["E"], rel=rel, stride=Wf["stride"], D_picked=dpk, n_non_freeze=non)
                med = float(np.median(dpk)) if dpk else float("nan")
                u, n = np.unique([f"D={FREEZE[p] - Wf['E']}" if p in FREEZE else LAB[p] for p in picks], return_counts=True)
                rows_.append((rel, f"| `{w}` | {rel:+d} | {c} | " + " · ".join(f"{LAB[a]} {D}" for D, a in Ds) + " | "
                              + " · ".join(f"{a} {k}" for a, k in sorted(zip(u, n), key=lambda z: -z[1])) + f" | {non} | "
                              + f"{med:.0f} / {med / Wf['stride']:.1f} | "
                              + (f"{np.median(dball):.0f}" if dball else "—") + f" · {bnon or '—'} |"))
                js["freeze_delay"][f"{o}|{w}|{c}"].update(D_picked_ball=dball, pick_ball=bpk)
        md += [r for _, r in sorted(rows_, key=lambda z: z[0])] + [""]
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "tables.md").write_text("\n".join(md) + "\n")
    (OUTD / "tables.json").write_text(json.dumps(js, indent=1, default=float))
    print("\n".join(md))
    print(f"\n→ {OUTD / 'tables.md'}")


if __name__ == "__main__":
    main()
