"""자로 읽은 예측 궤적이 실제 운동을 얼마나 따라가나 — 속도 · 가속도 (2026-10-14).
클립마다 t9–t15 의 위치를 (t8 은 late 에서 물체가 실제로 가려진 튜블릿이라 뺀다) 실제 운동 축 (마지막 관측 → 각 튜블릿의 진실 위치 방향, 평균) 에 투영한 진행량 s(t) 에
2 차식 s = a + b·t + c·t² 를 맞춘다 (t = 0..6 튜블릿). 진실 궤적도 같은 식으로 맞춘다. 단위 = 칸 (32 px @ 512) · 튜블릿.
  속도 비 = 예측 총 이동 (s(6) − s(0), 적합값) / 진실 총 이동      (1 = 맞음, < 1 = 덜 감)
  가속 = 2c (칸/튜블릿²): 가속 (flat_acc · ramp_down) 은 +, 감속 (flat_dec · ramp_up) 은 −, 등속 0 이 진실
  방향 cos = 예측 변위 벡터 (t9→t15) 와 진실 변위 벡터의 cos
t9–t15 에 p 가 '없음' (argmax) 인 튜블릿이 하나라도 있는 클립은 뺀다 (n 에 남은 수). hh = h 를 h 자로 읽은 것 (자 자체의 한계, 위치를 실제 프레임에서 읽음).
범위: split v2 test block · primitive · 물체 있는 가능 영상 · 이동 운동.

  python z_research/scripts/analysis/vobench_readout_motion.py [--occ late | seen (= visible·early·mid)] [--names "rel relft vast70 vast70ft"]
  → z_research/VOBench_v1/vjepa2_pft/readout_figures/motion_fidelity_<occ>.{md,json}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import vobench_ar_readout as V  # noqa: E402

OUT = REPO / "z_research/VOBench_v1/vjepa2_pft/readout_figures"
MOTS = {"flat_vel": "const. vel.", "flat_acc": "accel.", "ramp_down": "ramp down (accel.)", "flat_dec": "decel.", "ramp_up": "ramp up (decel.)"}
J0 = 1                                         # t9 부터
TQ = np.arange(8.0 - J0)


def fit(s):
    c2, c1, c0 = np.polyfit(TQ, s, 2)
    return c0, c1, c2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--occ", default="late"); ap.add_argument("--names", default="rel relft vast70 vast70ft")
    a = ap.parse_args(); names = a.names.split()
    test = {r["block_id"] for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_test.csv"))}
    occs = ("visible", "early", "mid") if a.occ == "seen" else (a.occ,)
    rows = [r for r in V.rows_primitive() if r["block_id"] in test and r["occlusion"] in occs and r["motion"] in MOTS and r["variant"] in ("pos_a", "pos_b")]
    Z = {n: V.load(n) for n in names}; vid = {n: {v: i for i, v in enumerate(z["video_id"])} for n, z in Z.items()}
    R = defaultdict(lambda: defaultdict(list)); lab = {}
    for r in rows:
        arm = "A" if r["variant"] == "pos_a" else "B"; obj = r["obj_a"] if arm == "A" else r["obj_b"]
        if obj == "empty": continue
        bdir = V.FR / Path(r["file_name"]).parent
        if bdir not in lab: lab[bdir] = json.load(open(bdir / "label.json"))["labels"]
        L = lab[bdir][arm]
        P = np.array([[L[3 * s]["u"], L[3 * s]["v"]] for s in range(32)], float)
        truth = np.array([(P[16 + 2 * j] + P[17 + 2 * j]) / 2 for j in range(J0, 8)])
        hid = np.array([L[3 * s]["hidden"] for s in range(32)], bool)
        last = max([s for s in range(16) if not hid[s]] or [15])
        ax = truth[-1] - P[last]; nrm = np.linalg.norm(ax)
        if nrm < 1e-6: continue
        ax = ax / nrm
        st = (truth - P[last]) @ ax / V.CELL; _, tb, tc = fit(st)
        K = len(TQ) - 1; T_disp, T_acc = (tb * K + tc * K * K), 2 * tc
        R[("truth", r["motion"])]["disp"].append(T_disp); R[("truth", r["motion"])]["acc"].append(T_acc)
        for n in names:
            i = vid[n][r["video_id"]]
            for tag in ("pp", "hh"):
                if tag == "hh" and n != names[0]: continue
                key = n if tag == "pp" else "h (ruler on h)"
                if (Z[n][f"{tag}_prob"][i, J0:].argmax(-1) == V.NONE).any(): R[(key, r["motion"])]["dropped"].append(1); continue
                xy = np.array([V.to512(Z[n][f"{tag}_xy"][i, j]) for j in range(J0, 8)])
                sp = (xy - P[last]) @ ax / V.CELL; _, pb, pc = fit(sp)
                d = xy[-1] - xy[0]; dt = truth[-1] - truth[0]
                cos = float(d @ dt / (np.linalg.norm(d) * np.linalg.norm(dt) + 1e-9))
                R[(key, r["motion"])]["ratio"].append((pb * K + pc * K * K) / T_disp if abs(T_disp) > 1e-3 else np.nan)
                R[(key, r["motion"])]["acc"].append(2 * pc); R[(key, r["motion"])]["acc_true"].append(T_acc)
                R[(key, r["motion"])]["cos"].append(cos)
                R[(key, r["motion"])]["err_end"].append(np.linalg.norm(xy[-1] - truth[-1]) / V.CELL)
    order = ["h (ruler on h)"] + names
    md = [f"# 예측 궤적의 운동 충실도 — {a.occ} (split v2 test · primitive · 이동 운동)", "",
          "속도 비 = 예측 총 이동 / 진실 총 이동 (1 = 맞음) · 가속 = 2c 칸/튜블릿² (진실 괄호) · cos = 변위 방향 · 끝 오차 = t15 위치 오차 칸 · n = '없음' 없는 클립 / 전체.", ""]
    J = {}
    for mo, ml in MOTS.items():
        tr = R[("truth", mo)]
        md += [f"## {ml}  (진실 총 이동 {np.median(tr['disp']):.2f} 칸 · 진실 가속 {np.median(tr['acc']):+.3f})", "",
               "| 모델 | n | 속도 비 (중앙값) | 가속 2c (중앙값) | 가속 부호 맞음 | 방향 cos | t15 위치 오차 (칸) |", "|---|---:|---:|---:|---:|---:|---:|"]
        for k in order:
            d = R[(k, mo)]; nd = len(d.get("dropped", [])); n = len(d.get("ratio", []))
            if not n:
                md.append(f"| {k} | 0 / {nd} | — | — | — | — | — |"); continue
            acc, acct = np.array(d["acc"]), np.array(d["acc_true"])
            sign = np.mean(np.sign(acc) == np.sign(acct)) * 100 if mo != "flat_vel" else np.nan
            row = dict(n=n, n_all=n + nd, ratio=float(np.nanmedian(d["ratio"])), acc=float(np.median(acc)), acc_true=float(np.median(acct)),
                       sign=float(sign), cos=float(np.median(d["cos"])), err_end=float(np.mean(d["err_end"])))
            J[f"{k}/{mo}"] = row
            md.append(f"| {k} | {n} / {n + nd} | {row['ratio']:.2f} | {row['acc']:+.3f} ({row['acc_true']:+.3f}) | "
                      f"{'—' if mo == 'flat_vel' else f'{sign:.0f} %'} | {row['cos']:.2f} | {row['err_end']:.2f} |")
        md.append("")
    md += ["## 재현", "", "```bash", f"python z_research/scripts/analysis/vobench_readout_motion.py --occ {a.occ} --names \"{a.names}\"", "```"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"motion_fidelity_{a.occ}.md").write_text("\n".join(md)); json.dump(J, open(OUT / f"motion_fidelity_{a.occ}.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
