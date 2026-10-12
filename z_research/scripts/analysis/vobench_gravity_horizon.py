#!/usr/bin/env python3
"""VOBench_v1 gravity arc — 예측 길이 M (16 · 8 · 4 · 2 샘플) 를 줄여도 "몇 샘플 가다 멈춤" 이 유지되나 (2026-10-11).

문맥 16 샘플 (stride 3, raw 0–45) 고정, 미래 M 샘플 (raw 48 … 45+3M). 창 c16t{16+M}.
가설 — 절대: 문맥 끝에서 일정 거리 (프레임) 만 진짜 운동을 잇고 멈춘다 (M 과 무관) / 상대: 멈춤이 예측 길이에 비례한다.

(1) 창 안 멈춤 시점 D — 고른 후보 = argmin_k s_k (미래 9 개, 창 안 토큰 전부 또는 공 칸). 멈춤 후보 stopK 의 멈춘 프레임
    F(stop)=45 · F(stop2)=50 · F(stop4)=52 · F(stop8)=56 · F(stop12)=60, arc = 멈추지 않음. 창 끝 Fend = 45 + 3M 에서 자른다:
    Feff = min(F, Fend), D = Feff − 45. Feff ≥ Fend − 1 인 후보는 창 안 모든 샘플에서 arc 와 1 프레임 이내다 ("창 끝까지 움직임").
    절대 → D 가 M 과 무관 (Fend 가 작으면 "창 끝까지"), 상대 → D / (3M) 가 일정.
(2) 튜블릿별 움직임 vs 정지 — 튜블릿 t (raw 48+6t, 51+6t) 에서 e(p, h_arc) < e(p, h_stop) 인 block 비율
    (stop = f45 자리 정지). 전체 토큰 · 공 칸 (두 후보 공 자리 합집합 3×3). 절대 시간축 (프레임) 에 M 마다 겹친다.
    절대 → 곡선이 겹친다, 상대 → M 이 짧을수록 일찍 떨어진다. CI = block 부트스트랩 95 %.
입력 = 창마다 surprise 실행의 token_surprise.npz (token_surprise_maps.py, 토큰 평균 = 채점값 검증 끝). primitive · realistic 따로.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/vobench_gravity_horizon.py    # → z_research/VOBench_v1/vjepa2/gravity_horizon/{tables.md,tables.json,figures/}
"""
from __future__ import annotations
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "z_research/VOBench_v1/exp_results"
OUT = REPO / "z_research/VOBench_v1/vjepa2/gravity_horizon"
INDEX = REPO / "data_csv/vobench_v1_gravity/index.csv"
FR = Path("/data2/local_datasets/world/Benchmark/VOBench_v1/gravity")
RUNS = {16: EXP / "surprise_c16t32__vobench_v1_gravity_vith", 8: EXP / "surprise_c16t32__vobench_v1_gravity_vith_wc16t24",
        4: EXP / "surprise_c16t32__vobench_v1_gravity_vith_wc16t20", 2: EXP / "surprise_c16t32__vobench_v1_gravity_vith_wc16t18"}
ARC = ["pos", "imp_rise", "imp_float", "imp_line", "imp_stop", "imp_stop2", "imp_stop4", "imp_stop8", "imp_stop12"]
LET = dict(zip(ARC, ["P", "R", "F", "L", "S", "S2", "S4", "S8", "S12"]))
FREEZE = {"imp_stop": 45, "imp_stop2": 50, "imp_stop4": 52, "imp_stop8": 56, "imp_stop12": 60}
CELLS = [("apex66", "rising"), ("apex48", "apex"), ("apex30", "descending")]
OBJ = ["primitive", "realistic"]
RNG = np.random.default_rng(0)


def boot(x, n=2000):
    x = np.asarray(x, float)
    b = x[RNG.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(x.mean()), *map(float, np.percentile(b, [2.5, 97.5]))]


def ball_mask(U, cands, frames):
    m = np.zeros((16, 16), bool)
    for a in cands:
        for f in frames:
            u, v = U[LET[a]][f]
            i, j = int(np.clip(v // 32, 0, 15)), int(np.clip(u // 32, 0, 15))
            m[max(0, i - 1):i + 2, max(0, j - 1):j + 2] = True
    return m


def main():
    rows = [r for r in csv.DictReader(INDEX.open()) if r["law"] == "arc"]
    blocks = defaultdict(dict)
    for r in rows:
        blocks[r["block_id"]][r["variant"]] = r
    info = {b: dict(objset=v["pos"]["objset"], cell=v["pos"]["cell"]) for b, v in blocks.items()}
    UV = {}
    for b, v in blocks.items():
        L = json.loads((FR / Path(v["pos"]["file_name"]).parent / "label.json").read_text())["labels"]
        UV[b] = {k: np.array([[e["u"], e["v"]] for e in L[k]]) for k in L}
    have = {M: r for M, r in RUNS.items() if (r / "token_surprise.npz").exists()}
    if 16 not in have:
        sys.exit("M=16 (표준 창) token_surprise.npz 가 없다")
    res = {}
    for M, run in sorted(have.items(), reverse=True):
        z = np.load(run / "token_surprise.npz"); vi = {v: i for i, v in enumerate(z["video_id"])}; E = z["e"]
        T = E.shape[1]; assert T == M // 2, (M, E.shape)
        Fend = 45 + 3 * M
        tub = [(48 + 6 * t, 51 + 6 * t) for t in range(T)]
        per = {}
        for b, v in blocks.items():
            e = E[[vi[v[a]["video_id"]] for a in ARC]]                       # (9, T, 16, 16)
            U = UV[b]
            Mb = np.stack([ball_mask(U, ARC, tub[t]) for t in range(T)])        # 공 칸 (후보 9 개 합집합)
            s_all, s_ball = e.mean((1, 2, 3)), e[:, Mb].mean(1)
            d = {}
            for tag, s in (("all", s_all), ("ball", s_ball)):
                pk = ARC[int(np.argmin(s))]
                Feff = min(FREEZE[pk], Fend) if pk in FREEZE else (Fend if pk == "pos" else None)
                # 끝까지 = 창의 마지막 샘플 (Fend) 에서도 arc 와 1 프레임 이내 (예: M 4 의 stop8 = f56, M 2 의 stop2 = f50)
                d[tag] = dict(pick=pk, D=None if Feff is None else Feff - 45, moves_to_end=Feff is not None and Feff >= Fend - 1)
            # (2) 튜블릿별 arc vs stop
            ia, is_ = ARC.index("pos"), ARC.index("imp_stop")
            mv_all, mv_ball = [], []
            for t in range(T):
                mb = ball_mask(U, ["pos", "imp_stop"], tub[t])
                mv_all.append(bool(e[ia, t].mean() < e[is_, t].mean()))
                mv_ball.append(bool(e[ia, t][mb].mean() < e[is_, t][mb].mean()))
            d["move_all"], d["move_ball"] = mv_all, mv_ball
            per[b] = d
        res[M] = dict(Fend=Fend, tubelets=tub, per=per)

    js = {"definition": __doc__.split("\n\n")[1], "M": {}}
    md = ["# VOBench_v1 gravity arc — 예측 길이 M 을 줄이면 (2026-10-11)", "",
          "문맥 16 샘플 고정 (stride 3, 문맥 끝 f45), 미래 M 샘플 = f48 … f(45+3M). 창 c16t{16+M}. 칸마다 12 block. primitive · realistic 따로.",
          "정의는 스크립트 머리 (`z_research/scripts/analysis/vobench_gravity_horizon.py`).", ""]
    md += ["## 1. 창 안 멈춤 시점 D (고른 후보의 멈춘 프레임을 창 끝에서 자름 − 45)", "",
           "칸 = 고른 block 수. `끝까지` = 창 안 모든 샘플에서 arc 와 1 프레임 이내 (arc 또는 창 끝 직전 · 뒤에 멈추는 stopK). `D=k` = 문맥 끝 뒤 k 프레임 가서 멈춤. 그 밖 = rise · float · line.", ""]
    for tag, tname in (("all", "전체 토큰 (= 채점값)"), ("ball", "공 칸만")):
        md += [f"### {tname}", "", "| objset · 문맥 | M (창 끝) | 끝까지 | 멈춤 D 분포 | 그 밖 | 멈춘 것의 D 중앙 · D / (3M) |", "|---|---|---:|---|---|---|"]
        for o in OBJ:
            for cell, ph in CELLS:
                for M in sorted(res, reverse=True):
                    R = [d[tag] for b, d in res[M]["per"].items() if info[b]["objset"] == o and info[b]["cell"] == cell]
                    end = sum(x["moves_to_end"] for x in R)
                    Ds = [x["D"] for x in R if x["D"] is not None and not x["moves_to_end"]]
                    oth = Counter(x["pick"].replace("imp_", "") for x in R if x["D"] is None)
                    med = float(np.median(Ds)) if Ds else float("nan")
                    js["M"].setdefault(str(M), {})[f"{tag}|{o}|{cell}"] = dict(n=len(R), to_end=end, D=Ds, other=dict(oth))
                    md.append(f"| {o} · {ph} | {M} (f{res[M]['Fend']}) | {end} | " + (" · ".join(f"D={k} {n}" for k, n in sorted(Counter(Ds).items())) or "—")
                              + f" | {' · '.join(f'{k} {n}' for k, n in oth.items()) or '—'} | "
                              + (f"{med:.0f} · {med / (3 * M):.2f}" if Ds else "—") + " |")
        md.append("")
    md += ["## 2. 튜블릿별 — p 가 정지 (f45) 보다 진짜 운동 (arc) 에 가까운 block %", "",
           "열 = 튜블릿의 절대 프레임. 절대 가설이면 M 이 달라도 같은 프레임에서 같은 값, 상대 가설이면 M 이 짧을수록 일찍 떨어진다.", ""]
    js["move"] = {}
    for tag, tname in (("move_all", "전체 토큰"), ("move_ball", "공 칸만")):
        md += [f"### {tname}", ""]
        for o in OBJ:
            for cell, ph in CELLS:
                md += [f"`{o} · {ph}` (n=12)", "", "| M | " + " | ".join(f"f{48 + 6 * t}–{51 + 6 * t}" for t in range(8)) + " |", "|---|" + "---:|" * 8]
                for M in sorted(res, reverse=True):
                    R = [d[tag] for b, d in res[M]["per"].items() if info[b]["objset"] == o and info[b]["cell"] == cell]
                    cells = []
                    for t in range(8):
                        if t < M // 2:
                            c = boot([r[t] for r in R]); js["move"][f"{tag}|{o}|{cell}|{M}|{t}"] = c
                            cells.append(f"{100 * c[0]:.0f}")
                        else:
                            cells.append("")
                    md.append(f"| {M} | " + " | ".join(cells) + " |")
                md.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables.md").write_text("\n".join(md) + "\n")
    (OUT / "tables.json").write_text(json.dumps(js, indent=1))
    print("\n".join(md)); print("→", OUT / "tables.md")


if __name__ == "__main__":
    main()
