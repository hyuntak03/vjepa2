#!/usr/bin/env python3
"""RollOutV3 문서 (`README.md` · `Archive/IDENTITY_R8_2026-09-26.md`) 의 **표를 산출물에서 다시 찍는다** (2026-09-26).

문서의 수치는 이 출력에서 옮긴다. 문서를 고치면 이것을 다시 돌려 대조한다. CPU 몇 초, GPU 없음.

정의 (그림 스크립트와 같다):
  '있음'         정체 자: score = log max_c P(c) − log P(없음) > 0 (= 57-way argmax ≠ 없음). presence 자: logit > thr_val_fpr5
  위치            readout xy·144 (+144) − `attn_bias_px.json` (자의 학습셋 test × 양성 평균 치우침)
  v3 튜블릿 진실   샘플 32 + 2t, 33 + 2t 의 평균 (문맥은 샘플 32 에서 끝난다, `rollout3_window_readout.py` SPLIT)
  v3 복사          문맥 마지막 튜블릿 (샘플 30 · 31) 의 진실 자리에 그대로 둔 것
  v11 튜블릿 진실  샘플 16 + 2t, 17 + 2t 의 평균.  마지막 본 자리 = 샘플 16 − k − 1 (가림 없음은 15)
  조합 맞음        57-way argmax = clip 의 (모양, 색). "'있음' 칸 중" 과 "전체 clip 중" 을 따로 적는다 (우연 1/56 = 1.8 %)

  $P z_research/scripts/analysis/rollout3_doc_numbers.py [--decoder identity_r8]
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
EXP = ROOT / "z_research/RollOutV3/exp_results"
AUD = ROOT / "z_research/RollOutV3/audit"
FIG = ROOT / "z_research/RollOutV3/figures"
import os
TRAIN_META = Path(os.environ.get("R3_TRAIN_META", "/data2/local_datasets/world/world_analysis/RollOut_v2_training_v8/metadata.csv"))   # 2026-09-30: 로그인 노드 (/data2 없음) 용 덮어쓰기
R, SPLIT = 144.0, 32
f0 = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}"
f1 = lambda x: f"{100 * x:.1f}"


def head(t):
    print(f"\n## {t}")


def dec_info(dec):
    S = json.loads((EXP / dec / "summary.json").read_text())
    b = json.loads((EXP / dec / "attn_bias_px.json").read_text())
    ident = "none_index" in S
    thr = {r: (0.0 if ident else S["reps"][r]["attn"]["thr_val_fpr5"]) for r in "pzh"}
    return S, b, thr, ident


def mean_nan(a, m):
    return float(a[m].mean()) if m.any() else float("nan")


# ------------------------------------------------------------------ 자

def decoder_tables(dec):
    head(f"§3-1 학습셋 test ({dec} summary.json; 위치 = 양성 전부, 치우침 빼기 전, 칸)")
    for d in (dec, "identity"):
        if not (EXP / d / "summary.json").exists(): continue
        S = json.loads((EXP / d / "summary.json").read_text())
        for r in "pzh":
            if r not in S["reps"]: continue                      # 2026-09-30: kind=ar 자 (identity_ar40) 는 z 가 없다
            s = S["reps"][r]
            print(f"  {d:12s} {r}  조합 {f1(s['acc57'])}  모양 {f1(s['shape_acc'])}  색 {f1(s['color_acc'])}  없음 AUROC {s['none_auroc']:.4f}  "
                  f"위치 {s['center_mean']:.2f} 칸  argmax recall {f1(s['argmax_recall'])} · 오탐 {f1(s['argmax_fpr'])}")
    head("§3-1 학습셋 test 의 판 빈 장면 '있음' — 판 상태별 (argmax)")
    M = {r["name"]: r for r in csv.DictReader(TRAIN_META.open())}

    def pstate(r, f):
        rs, re_, fs, fe = (int(r[k]) for k in ("occ_rise_start", "occ_rise_end", "occ_fall_start", "occ_fall_end"))
        return "lying" if (f < rs or f > fe) else ("standing" if re_ <= f <= fs else "moving")
    S = json.loads((EXP / dec / "summary.json").read_text()); none = S["none_index"]
    for r_ in "ph":
        d = np.load(EXP / dec / r_ / "preds.npz", allow_pickle=True); cnt = {}
        for i in np.where(d["test"])[0]:
            m = M[d["video_id"][i]]
            if m.get("shape_pre") != "none" or not m.get("occ_rise_start"):
                continue
            for t in range(8):
                if d["cls"][i, t] != none:
                    continue
                st = {pstate(m, 3 * x) for x in (16 + 2 * t, 17 + 2 * t)}; st = "moving" if len(st) > 1 else st.pop()
                cnt.setdefault(st, []).append(d["prob"][i, t].argmax() != none)
        print(f"  {r_}: " + " · ".join(f"{k} {f1(np.mean(v))} % (n {len(v)})" for k, v in sorted(cnt.items())))
    head("§1 · §3-2 v11 판 빈 장면 '있음' (audit/training_v8/panel_false_alarm_<자>.json; p / h / p '있음' 중 판 위)")
    for d in ("presence", "identity", dec):
        f = AUD / "training_v8" / f"panel_false_alarm_{d}.json"
        if f.exists():
            J = json.loads(f.read_text())["results"]
            print(f"  {d:12s} " + "  ".join(f"{k} {f0(v['p']['say'])}/{f0(v['h']['say'])}/{f0(v['p'].get('on_panel_given_say'))}" for k, v in J.items()))
    for d in ("identity", dec):
        z = np.load(EXP / f"v11_{d}" / "readings.npz", allow_pickle=True)
        g = ~z["obj"] & (z["sym_k"].astype(int) == 0)
        print(f"  {d:12s} 판 없는 빈 장면 (가림 없음 · 빈 clip {g.sum()}): p {f1((z['p'][g, -8:, 2] > 0).mean())} · h {f1((z['h'][g, -8:, 2] > 0).mean())} %")


# ------------------------------------------------------------------ v3

def v3_tables(dec):
    idx = {r["video_id"]: r for r in csv.DictReader((ROOT / "data_csv/rollout_v3/index.csv").open())}
    head("§3-4 v3 위치 오차 — 16 창 · 가능 clip · '있음' 칸 · 치우침 뺌; 앞 t0–t3 / 뒤 t≥4, 평균 |x| / |y| px (앞 부호 y)")
    for d, w in (("presence", "windows"), ("identity", "windows_identity"), (dec, f"windows_{dec}")):
        z = np.load(EXP / w / "readings.npz", allow_pickle=True); S, b, thr, _ = dec_info(d)
        POS = z["role"] == "roll"; L = z["truth"][POS] * R; out = []
        for rep in ("p", "z"):
            F, B = [], []
            for c in (4, 8, 16, 32):
                for p in (4, 8, 16, 32):
                    tp = p // 2; gt = L[:, SPLIT:SPLIT + p].reshape(len(L), tp, 2, 2).mean(2)
                    v = z[f"C{c}_P{p}_{rep}"][POS]; say = v[..., 2] > thr[rep]
                    e = v[..., :2] * R - np.array(b[rep], np.float32) - gt
                    t = np.arange(tp)[None].repeat(len(L), 0)
                    F.append(e[say & (t < 4)]); B.append(e[say & (t >= 4)])
            F, B = np.concatenate(F), np.concatenate(B)
            out.append(f"{rep} 앞 {np.abs(F[:, 0]).mean():.1f} / {np.abs(F[:, 1]).mean():.1f} · 뒤 {np.abs(B[:, 0]).mean():.1f} / {np.abs(B[:, 1]).mean():.1f} "
                       f"(앞 부호 y {F[:, 1].mean():+.1f})")
        print(f"  {d:12s} " + "   ".join(out))

    S, b, thr, _ = dec_info(dec)
    z = np.load(EXP / f"windows_{dec}" / "readings.npz", allow_pickle=True)
    POS = z["role"] == "roll"; vid = z["video_id"][POS]; law = z["scenario"][POS]
    combo = np.array([S["shapes"].index(idx[v]["shape_pre"]) * 8 + S["colors"].index(idx[v]["color_pre"]) for v in vid])
    L = z["truth"][POS] * R
    for c, p in ((16, 32), (16, 16), (32, 32), (8, 32)):
        tp = p // 2; gt = L[:, SPLIT:SPLIT + p].reshape(len(L), tp, 2, 2).mean(2); copy = L[:, SPLIT - 2:SPLIT].mean(1)
        vp = z[f"C{c}_P{p}_p"][POS]; say = vp[..., 2] > 0
        okp = z[f"C{c}_P{p}_p_prob"][POS].argmax(-1) == combo[:, None]
        okz = z[f"C{c}_P{p}_z_prob"][POS].argmax(-1) == combo[:, None]
        xy = vp[..., :2] * R - np.array(b["p"], np.float32)
        closer = np.linalg.norm(xy - gt, axis=-1) < np.linalg.norm(copy[:, None] - gt, axis=-1)
        head(f"§4 v3 C{c}/P{p} — 튜블릿 t0…t{tp - 1} (가능 clip {POS.sum()})")
        print("  p '있음'              ", [f0(say[:, t].mean()) for t in range(tp)])
        print("  p 조합 맞음 (전체)     ", [f0(okp[:, t].mean()) for t in range(tp)])
        print("  z 조합 맞음 (전체)     ", [f0(okz[:, t].mean()) for t in range(tp)])
        print("  p 가 복사보다 가까움 ('있음' 칸)", [f0(mean_nan(closer[:, t], say[:, t])) for t in range(tp)])
        if (c, p) == (16, 32):
            print("  법칙별 복사보다 가까움 t15:", " · ".join(f"{l} {f0(mean_nan(closer[law == l, 15], say[law == l, 15]))}" for l in np.unique(law)))


# ------------------------------------------------------------------ v11

def v11_prep(z, S, b):
    n = len(z["sym_k"]); k = z["sym_k"].astype(int)
    mo = np.array(["static" if c.startswith("static") else ("flat" if "flat" in c else "ramp") for c in z["condition"]])
    L = z["truth"] * R + R; X = L[:, 16:].reshape(n, 8, 2, 2).mean(2)
    combo = np.array([S["shapes"].index(s) * 8 + S["colors"].index(c) if s in S["shapes"] else -1 for s, c in zip(z["shape_pre"], z["color_pre"])])
    return dict(k=k, mo=mo, L=L, X=X, obj=z["obj"], td=z["travel_dir"], say=z["p"][:, -8:, 2] > 0, hsay=z["h"][:, -8:, 2] > 0,
                xy=z["p"][:, -8:, :2] * R + R - np.array(b["p"]), ok=z["p_prob"][:, -8:].argmax(-1) == combo[:, None],
                hid=z["hidden"][:, 16:].reshape(n, 8, 2))


def v11_tables(dec):
    S, b, _, _ = dec_info(dec)
    V = v11_prep(np.load(EXP / f"v11_{dec}" / "readings.npz", allow_pickle=True), S, b)
    VM = v11_prep(np.load(EXP / f"v11_mid_{dec}" / "readings.npz", allow_pickle=True), S, b)
    head("§5-1 p '있음' — 물체 clip, 튜블릿 t3–t6 평균 (문맥 끝 k=0–4 | 문맥 중간 k=1–4)")
    for m in ("static", "flat", "ramp"):
        a = [f0(V["say"][V["obj"] & (V["mo"] == m) & (V["k"] == kk)][:, 3:7].mean()) for kk in range(5)]
        c = [f0(VM["say"][VM["obj"] & (VM["mo"] == m) & (VM["k"] == kk)][:, 3:7].mean()) for kk in range(1, 5)]
        print(f"  {m:6s} 끝 {a}   중간 {c}")
    head("§3-3 h '있음' — 문맥 끝 가림 (k≥1) 물체 clip")
    g = V["obj"] & (V["k"] >= 1)
    per = [V["hsay"][V["obj"] & (V["mo"] == m) & (V["k"] == kk), t].mean() for m in ("static", "flat", "ramp") for kk in range(1, 5) for t in range(3, 7)]
    print(f"  t3–t6 (다시 보인 칸), (운동, k, t) 칸별 최소–최대 {f0(min(per))}–{f0(max(per))} %")
    for m in ("static", "flat", "ramp"):
        gm = g & (V["mo"] == m); both = V["hid"][gm].all(2); one = V["hid"][gm].any(2) & ~both; hs = V["hsay"][gm]
        print(f"  {m:6s} 두 샘플 모두 가림 {f0(hs[both].mean())} % (n {both.sum()}) · 한 샘플만 {f0(hs[one].mean())} % (n {one.sum()})")
    head("§5-2 '있을' 때 무엇이 어디에 — 튜블릿 t1…t6: '있음' / 조합 ('있음' 칸 중) / 진실까지 px / 마지막 본 자리까지 px / (진실↔마지막 본 자리)")
    for m in ("ramp", "flat"):
        for kk in (1, 2, 3, 4):
            gm = V["obj"] & (V["mo"] == m) & (V["k"] == kk); ls = V["L"][gm, 16 - kk - 1]; cells = []
            for t in range(1, 7):
                s = V["say"][gm, t]; xy = V["xy"][gm, t]
                cells.append(f"t{t} {f0(s.mean())}/{f0(mean_nan(V['ok'][gm, t], s))}/"
                             f"{np.linalg.norm(xy - V['X'][gm, t], axis=-1)[s].mean():.0f}/{np.linalg.norm(xy - ls, axis=-1)[s].mean():.0f}/"
                             f"{np.linalg.norm(V['X'][gm, t] - ls, axis=-1).mean():.0f}")
            print(f"  {m} k{kk}: " + " | ".join(cells))
    gm = V["obj"] & (V["mo"] == "ramp") & (V["k"] == 2)
    for tdv, nm in ((1, "l2r"), (-1, "r2l")):
        gg = gm & (V["td"] == tdv); s = V["say"][gg, 1]
        print(f"  ramp k2 t1 {nm}: 진실까지 {np.linalg.norm(V['xy'][gg, 1] - V['X'][gg, 1], axis=-1)[s].mean():.0f} px · "
              f"마지막 본 자리까지 {np.linalg.norm(V['xy'][gg, 1] - V['L'][gg, 13], axis=-1)[s].mean():.0f} px")
    for kk in range(1, 5):
        gm = V["obj"] & (V["mo"] == "static") & (V["k"] == kk); s = V["say"][gm, 3:7]
        d = np.linalg.norm(V["xy"][gm, 3:7] - V["X"][gm, 3:7], axis=-1)
        print(f"  static k{kk} (t3–t6): '있음' {f0(s.mean())} · 조합 ('있음' 칸 중) {f0(mean_nan(V['ok'][gm, 3:7], s))} · 진실까지 {d[s].mean():.0f} px")
    gm = V["obj"] & (V["mo"] == "static") & (V["k"] == 0)
    print("  static 가림 없음 p 조합 (전체) t0…t7:", [f0(x) for x in V["ok"][gm].mean(0)])
    head("§7 t7 — 물체가 화면을 나가는 중: p '있음' / h '있음' / p '있음' 칸의 진실까지 px")
    for m in ("static", "flat", "ramp"):
        cells = []
        for kk in range(5):
            gm = V["obj"] & (V["mo"] == m) & (V["k"] == kk); s = V["say"][gm, 7]
            cells.append(f"k{kk} {f0(s.mean())}/{f0(V['hsay'][gm, 7].mean())}/{np.linalg.norm(V['xy'][gm, 7] - V['X'][gm, 7], axis=-1)[s].mean():.0f}")
        print(f"  {m:6s} " + "  ".join(cells))
    head("§5-4 문맥 끝 가림 (k=1–4) 에서 '있음' 은 누구인가 — t1–t6, clip 당 '있음' 튜블릿 수 분포 · 모양별 '있음' (그중 모양 맞음)")
    z = np.load(EXP / f"v11_{dec}" / "readings.npz", allow_pickle=True); SH = S["shapes"]
    pm = z["p_prob"][:, -8:, :56].astype(np.float32).reshape(len(V["k"]), 8, 7, 8).sum(-1).argmax(-1)
    for m in ("flat", "ramp", "static"):
        for tag, kk in (("가림 없음", [0]), ("끝 가림 k1–4", [1, 2, 3, 4])):
            g = V["obj"] & (V["mo"] == m) & np.isin(V["k"], kk); say = V["say"][g][:, 1:7]
            if kk != [0]:
                print(f"  {m:6s} clip 당 '있음' 튜블릿 0…6 개 (%): {np.round(100 * np.bincount(say.sum(1), minlength=7) / g.sum()).astype(int).tolist()}")
            rows = []
            for s_ in SH:
                gg = g & (z["shape_pre"] == s_); pr = V["say"][gg][:, 1:7]; rd = pm[gg][:, 1:7][pr]
                rows.append(f"{s_} {f0(pr.mean())}({f0((rd == SH.index(s_)).mean() if len(rd) else float('nan'))})")
            print(f"  {m:6s} {tag:10s} " + " · ".join(rows))
    d = np.load(EXP / dec / "p" / "preds.npz", allow_pickle=True); te = d["test"]; cls = d["cls"][te]
    sy = d["prob"][te].argmax(-1) != S["none_index"]; pos = (cls >= 0) & (cls < S["none_index"])
    print("  학습셋 test p 자 '있음' recall 모양별: " + " · ".join(f"{s_} {f1(sy[pos & (cls // 8 == i)].mean())}" for i, s_ in enumerate(SH)))
    g = V["obj"] & (V["mo"] != "static") & (V["k"] >= 1)
    print("  h (실제 프레임) t3–t6 '있음' 모양별: " + " · ".join(f"{s_} {f0(V['hsay'][g & (z['shape_pre'] == s_)][:, 3:7].mean())}" for s_ in SH))
    head("§5-4b k 경향 — t3–t6, '있음' % · '있음' 중 모양 유지 % · 색 유지 % (유지 = 57-way argmax 조합의 모양 · 색)")
    top = z["p_prob"][:, -8:].argmax(-1); sy = top != S["none_index"]
    st_ = np.array([SH.index(x) if x in SH else -1 for x in z["shape_pre"]])[:, None]
    ct_ = np.array([S["colors"].index(x) if x in S["colors"] else -1 for x in z["color_pre"]])[:, None]
    shk, cok = sy & (top // 8 == st_), sy & (top % 8 == ct_)
    zm = np.load(EXP / f"v11_mid_{dec}" / "readings.npz", allow_pickle=True)
    for tag, zz, ks in (("끝", z, (0, 1, 2, 3, 4)), ("중간", zm, (1, 2, 3, 4))):
        kz = zz["sym_k"].astype(int); mz = np.array([("static" if c.startswith("static") else ("flat" if "flat" in c else "ramp")) for c in zz["condition"]])
        tz = zz["p_prob"][:, -8:].argmax(-1); syz = tz != S["none_index"]
        sz = np.array([SH.index(x) if x in SH else -1 for x in zz["shape_pre"]])[:, None]; cz = np.array([S["colors"].index(x) if x in S["colors"] else -1 for x in zz["color_pre"]])[:, None]
        for m in ("static", "flat", "ramp"):
            cells = []
            for kk in ks:
                g = zz["obj"] & (mz == m) & (kz == kk); sp = syz[g][:, 3:7]
                cells.append(f"k{kk} {f0(sp.mean())}/{f0(mean_nan((syz & (tz // 8 == sz))[g][:, 3:7], sp))}/{f0(mean_nan((syz & (tz % 8 == cz))[g][:, 3:7], sp))}")
            print(f"  {tag:2s} 가림 {m:6s} " + "  ".join(cells))
    rt = V["obj"] & (V["mo"] != "static") & np.isin(z["shape_pre"], ["sphere", "torus"])
    app = np.array([f"{a_}|{b_}|{e_}" for a_, b_, e_ in zip(z["shape_pre"], z["color_pre"], z["env"])]); units = np.unique(app[rt])
    rng_ = np.random.default_rng(0)
    def dk(us, kept):
        c1 = rt & (V["k"] == 1) & np.isin(app, us); c4 = rt & (V["k"] == 4) & np.isin(app, us)
        return mean_nan(kept[c4][:, 3:7], sy[c4][:, 3:7]) - mean_nan(kept[c1][:, 3:7], sy[c1][:, 3:7])
    for name, kept in (("색 유지", cok), ("모양 유지", shk)):
        bs = [dk(rng_.choice(units, len(units)), kept) for _ in range(500)]
        print(f"  sphere·torus 이동 끝 가림: '있음' 중 {name} k4 − k1 = {100 * dk(units, kept):+.1f}%p "
              f"[{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}] (겉모습 {len(units)} 개 bootstrap 500)")
    head("§5-5 정체 (모양+색 둘 다 맞음, 전체 clip 칸 %) — 마지막으로 본 자리에서 진짜 물체까지 거리 구간별 (칸 수 200 이상만)")
    bins = [0, 20, 40, 60, 80, 100, 130, 170]
    def dtab(dist, ok_, name):
        cells = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (dist >= lo) & (dist < hi)
            cells.append(f"{lo}-{hi} {f0(ok_[m].mean()) if m.sum() >= 200 else '-'}")
        print(f"  {name:22s} " + " | ".join(cells))
    idx3 = {r["video_id"]: r for r in csv.DictReader((ROOT / "data_csv/rollout_v3/index.csv").open())}
    w3 = np.load(EXP / f"windows_{dec}" / "readings.npz", allow_pickle=True)
    P3 = (w3["role"] == "roll") & ~np.isin(w3["scenario"], ["ledge", "wall"]); L3 = w3["truth"][P3] * R
    c3 = np.array([SH.index(idx3[v]["shape_pre"]) * 8 + S["colors"].index(idx3[v]["color_pre"]) for v in w3["video_id"][P3]])
    d3 = np.linalg.norm(L3[:, 32:64].reshape(len(L3), 16, 2, 2).mean(2) - L3[:, 30:32].mean(1)[:, None], axis=-1)
    for rep in ("p", "z"):
        dtab(d3.ravel(), (w3[f"C16_P32_{rep}_prob"][P3].argmax(-1) == c3[:, None]).ravel(), f"v3 C16/P32 {rep}")
    for fn, ks, name in ((f"v11_{dec}", [0], "v11 가림 없음 이동"), (f"v11_mid_{dec}", [1, 2, 3, 4], "v11 문맥 중간 가림 이동")):
        zz = np.load(EXP / fn / "readings.npz", allow_pickle=True); kz = zz["sym_k"].astype(int)
        g = zz["obj"] & np.array([not c.startswith("static") for c in zz["condition"]]) & np.isin(kz, ks); Lz = zz["truth"][g] * R
        dz = np.linalg.norm(Lz[:, 16:].reshape(len(Lz), 8, 2, 2).mean(2)[:, :7] - Lz[:, 15][:, None], axis=-1)
        cz = np.array([SH.index(a_) * 8 + S["colors"].index(b_) for a_, b_ in zip(zz["shape_pre"][g], zz["color_pre"][g])])[:, None]
        for rep in (("p", "h") if ks == [0] else ("p",)):
            dtab(dz.ravel(), (zz[f"{rep}_prob"][g, -8:].argmax(-1)[:, :7] == cz).ravel(), f"{name} {rep}")
    head("§5-3 x 부호 오차 (figures/v11_signed_error/values.json; '있음' 절반 미만 칸 = —)")
    sv = json.loads((FIG / "v11_signed_error" / "values.json").read_text())
    for e in sv:
        if e["axis"] == "x" and e["motion"] != "static" and e["cond"] in ("visible", "last_k1", "last_k2", "last_k3", "last_k4"):
            print(f"  {e['motion']:5s} {e['cond']:8s} t1…t6 {['—' if v is None else f'{v:+.0f}' for v in e['err'][1:7]]}   "
                  f"멈춤 기준선 {['—' if v is None else f'{v:+.0f}' for v in e['stop'][1:7]]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", default="identity_r8")
    a = ap.parse_args()
    decoder_tables(a.decoder); v3_tables(a.decoder); v11_tables(a.decoder)


if __name__ == "__main__":
    main()
