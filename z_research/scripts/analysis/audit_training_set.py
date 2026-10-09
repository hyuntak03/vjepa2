#!/usr/bin/env python3
"""자 (위치 · 있음/없음 · 정체) **학습셋 감사** — 렌더된 세트 폴더 하나를 받아 metadata · 프레임을 직접 본다 (CPU, 수 분).

2026-09-25 에 training_v8 (옛 RollOut_v2_training) · RollOut_v2_training_v7 에서 찾은 결함을 전부 검사한다:
  A 구조     clip 수 · 열 · 이름 중복 · 샘플 열 길이 32 · 폴더 · 클립당 PNG 수 · PNG 크기 (표본)
  B 구성     arm × 방해물 (없음 / 소나무 / 판) × 물체 유무. **빈 칸이 있으면 가중치로 못 고친다**
  C 짝       빈 장면마다 같은 장면 (방해물 자세 · 판 타이밍 · 구조물 · 배경 · 방향) 의 물체 clip 이 있나
  D 자세     물체 clip 에만 / 빈 clip 에만 나오는 방해물 자세 (v7 에서 소나무 자세가 완전히 갈렸다)
  E 라벨     in_frame · prop_intersects · scenery 합, 정체 자 라벨 (조합 / 없음 / 제외, 미래 8 튜블릿), 조합당 양성 수
  F 의존     튜블릿 단위 P(있음 | 방해물 종류 · 구조물 · 판 자세), 상호정보량 원시 · balance_weight 가중
  G 가중치   balance_weight 가 방해물 종류마다 다른가 (v7: 판이 '평지' 로 묶여 전부 0.982)
  H 판       v11 가림막과 재질 · 깊이 · 화면 폭 / 높이 · rise-fall 타이밍 · 위치
  I 누수     양성 튜블릿 px_x · px_y 를 모양 · 색 · 배경 · 튜블릿 번호 원-핫으로 회귀한 R²
  J 커버리지 도달 가능 띠 (u 19–269, v 20–152) 의 18 px 칸 점유
  K 가림 열  has_occlusion · hidden_* · sym_k 가 채워진 clip 과 실제 가림 여부
  L 판 속성  판 clip 튜블릿에서 **보이는** 판 속성 (위치 · 폭 · 높이 · 폭×높이 · 상태) ↔ 있음/없음, 순열 귀무와 함께

  $P z_research/scripts/analysis/audit_training_set.py --root /data2/local_datasets/world/world_analysis/RollOut_v2_training_v8
      → z_research/RollOutV3/audit/training_sets/<폴더 이름>.json (+ 표준 출력 보고)
"""
from __future__ import annotations
import argparse, collections, csv, json, os, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
OUT = ROOT / "z_research/RollOutV3/audit/training_sets"
V11 = Path("/local_datasets/world/world_analysis/IntPhysGen_v11/metadata.csv")
SH = ["capsule", "cone", "cube", "cylinder", "pyramid", "sphere", "torus"]
CO = ["blue", "cyan", "green", "magenta", "orange", "purple", "red", "yellow"]
W, T0 = 288, 8
SCENE_KEYS = ["scenario", "condition", "env", "travel_direction", "surface", "occluder", "occluder_x_cm", "occ_width_cm",
              "occ_height_cm", "occ_depth_cm", "occ_rise_start", "occ_rise_end", "occ_fall_start", "occ_fall_end",
              "prop_x", "prop_y", "prop_w", "prop_h", "prop_thick", "prop_pitch", "ledge_z", "ramp_theta", "wall_x",
              "cam_y", "cam_z", "cam_pitch"]


def A_(s, n=32):
    s = str(s).split()
    return np.array([float(v) for v in s], np.float32) if s else np.zeros(n, np.float32)


def kind(r):
    if r.get("occluder") == "Trapdoor":
        return "panel"
    if float(r.get("prop_w") or 0) > 0:
        return "pine"
    return "none"


def stage(r):
    sc = r["scenario"]
    for k in ("ledge", "ramp", "wedge", "wall"):
        if k in sc:
            return "ramp" if k == "wedge" else k
    return "flat"


def pose(r):
    k = kind(r)
    if k == "panel":
        return f"panel w{r['occ_width_cm']} h{r['occ_height_cm']} x{r['occluder_x_cm']} rise{r['occ_rise_start']}-{r['occ_rise_end']} fall{r['occ_fall_start']}-{r['occ_fall_end']}"
    if k == "pine":
        return f"pine {float(r['prop_w']):.0f}x{float(r['prop_h']):.0f} t{float(r['prop_thick']):.0f} p{float(r['prop_pitch']):+.0f} y{float(r['prop_y']):.0f}"
    return "none"


def mi(x, y, w):
    w = w / w.sum(); o = 0.0
    for a in np.unique(x):
        pa = w[x == a].sum()
        for b in np.unique(y):
            pab = w[(x == a) & (y == b)].sum(); pb = w[y == b].sum()
            if pab > 0:
                o += pab * np.log2(pab / (pa * pb))
    return float(o)


def r2(y, groups):
    """y 를 범주 원-핫 (여러 개) 로 선형 회귀한 R² — 범주 평균으로 설명되는 분산 비율."""
    X = [np.ones(len(y))]
    for g in groups:
        u = np.unique(g)
        X += [(g == v).astype(float) for v in u[1:]]
    X = np.stack(X, 1); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--png-sample", type=int, default=300)
    a = ap.parse_args()
    root = Path(a.root); rep = {}; say = lambda *x: print(*x, flush=True)
    M = list(csv.DictReader((root / "metadata.csv").open())); n = len(M)
    say(f"# {root.name}  ({n} clip)")

    # ── A 구조
    bys = [k for k in M[0] if k.endswith("_by_sample")]
    badlen = {k: sum(1 for r in M if r[k].strip() and len(r[k].split()) != 32) for k in bys}
    names = collections.Counter(r["name"] for r in M)
    def cnt_png(r):
        d = root / r["file_name"]
        return len([f for f in os.listdir(d) if f.endswith(".png")]) if d.is_dir() else -1
    with ThreadPoolExecutor(32) as ex:
        npng = list(ex.map(cnt_png, M))
    from PIL import Image
    random.seed(0); samp = random.sample(M, min(a.png_sample, n)); sizes = collections.Counter()
    for r in samp:
        d = root / r["file_name"]
        for f in ("000000.png", "000093.png"):
            try:
                sizes[Image.open(d / f).size] += 1
            except Exception as e:
                sizes[f"err {type(e).__name__}"] += 1
    rep["A"] = dict(n=n, n_cols=len(M[0]), dup_names=sum(v > 1 for v in names.values()), bad_len=badlen,
                    missing_dir=sum(x < 0 for x in npng), png_per_clip=dict(collections.Counter(npng)), png_sizes={str(k): v for k, v in sizes.items()})
    say(f"A 구조: 열 {len(M[0])} · 이름 중복 {rep['A']['dup_names']} · 길이 틀린 샘플 열 {sum(badlen.values())} · 폴더 없음 {rep['A']['missing_dir']} · "
        f"clip 당 PNG {dict(collections.Counter(npng))} · PNG 크기 (표본) {dict(sizes)}")

    empty = np.array([r["shape_pre"] == "none" for r in M]); K = np.array([kind(r) for r in M])
    ST = np.array([stage(r) for r in M]); SC = np.array([r["scenario"] for r in M]); P_ = np.array([pose(r) for r in M])

    # ── B 구성
    tab = collections.Counter((r["scenario"], kind(r), "빈" if r["shape_pre"] == "none" else "물체") for r in M)
    rep["B"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(tab.items())}
    say("B 구성 (arm | 방해물 | 물체):")
    for sc in sorted(set(SC)):
        row = [f"{kd}: 물체 {tab.get((sc, kd, '물체'), 0)} / 빈 {tab.get((sc, kd, '빈'), 0)}" for kd in ("none", "pine", "panel")
               if tab.get((sc, kd, '물체'), 0) + tab.get((sc, kd, '빈'), 0)]
        say(f"   {sc:8s} " + "  ·  ".join(row))
    holes = [f"{kd}" for kd in ("none", "pine", "panel") if (K == kd).any() and not ((K == kd) & empty).any()]
    holes += [f"{kd}(물체 없음)" for kd in ("none", "pine", "panel") if (K == kd).any() and not ((K == kd) & ~empty).any()]
    rep["B_holes"] = holes
    say(f"   빈 칸 (방해물 종류가 한쪽에만): {holes or '없음 ✓'}")

    # ── C 짝
    key = lambda r: tuple(r.get(k, "") for k in SCENE_KEYS)
    obj_keys = collections.Counter(key(r) for r in M if r["shape_pre"] != "none")
    emp_keys = collections.Counter(key(r) for r in M if r["shape_pre"] == "none")
    unpaired = sum(v for k, v in emp_keys.items() if k not in obj_keys)
    rep["C"] = dict(n_empty=int(empty.sum()), empty_scenes=len(emp_keys), empty_unpaired=unpaired,
                    obj_scenes=len(obj_keys), obj_scenes_with_empty=sum(1 for k in obj_keys if k in emp_keys))
    say(f"C 짝: 빈 clip {int(empty.sum())} 중 같은 장면의 물체 clip 이 없는 것 {unpaired} · 물체 장면 {len(obj_keys)} 중 빈 짝 있음 {rep['C']['obj_scenes_with_empty']}")

    # ── D 자세
    pc = collections.defaultdict(lambda: [0, 0])
    for p, e in zip(P_, empty):
        pc[p][int(e)] += 1
    only_e = {p: v[1] for p, v in pc.items() if v[0] == 0 and p != "none"}
    only_o = {p: v[0] for p, v in pc.items() if v[1] == 0 and p != "none"}
    frac = {p: round(v[1] / (v[0] + v[1]), 3) for p, v in pc.items()}
    rep["D"] = dict(n_poses=len(pc), empty_only=only_e, object_only=only_o,
                    empty_frac_range=[min(frac.values()), max(frac.values())])
    say(f"D 자세: {len(pc)} 가지 · 빈 clip 에만 {len(only_e)} ({sum(only_e.values())} clip) · 물체 clip 에만 {len(only_o)} ({sum(only_o.values())} clip) · "
        f"자세별 빈 비율 {min(frac.values()):.3f}–{max(frac.values()):.3f}")

    # ── E 라벨 (정체 자 규칙, 미래 8 튜블릿)
    inf = np.stack([A_(r["in_frame_by_sample"]) > 0 for r in M]); pi = np.stack([A_(r["prop_intersects_by_sample"]) > 0 for r in M])
    si = np.stack([A_(r["scenery_intersects_by_sample"]) > 0 for r in M])
    px = np.stack([A_(r["object_px_x_by_sample"]) for r in M]); py = np.stack([A_(r["object_px_y_by_sample"]) for r in M])
    rad = np.array([float(r["obj_apparent_px"] or 0) / 2 for r in M])[:, None]
    tub = lambda x, h: (x.reshape(n, 16, 2).all(2) if h == "all" else x.reshape(n, 16, 2).any(2))[:, T0:]
    out = (px < -rad) | (px > W + rad) | (py < -rad) | (py > W + rad)
    pos = tub(inf & ~pi & ~si, "all"); scen = tub(si, "any")
    none = (empty[:, None] | tub(out, "all")) & ~pos & ~scen; ign = ~pos & ~none
    part = ~empty[:, None] & ~pos & ~none & ~scen & ~tub(pi, "any")
    combo = np.array([SH.index(r["shape_pre"]) * 8 + CO.index(r["color_pre"]) if r["shape_pre"] in SH else -1 for r in M])
    cc = np.bincount(np.repeat(combo[:, None], 8, 1)[pos], minlength=56)
    rep["E"] = dict(prop_intersects=int(pi.sum()), scenery=int(si.sum()), empty_inframe=int(inf[empty].sum()),
                    tub_combo=int(pos.sum()), tub_none=int(none.sum()), tub_none_empty=int((none & empty[:, None]).sum()),
                    tub_none_offscreen=int((none & ~empty[:, None]).sum()), tub_excluded=int(ign.sum()),
                    tub_excl_edge_or_mixed=int(part.sum()), per_combo=[int(cc.min()), int(np.median(cc)), int(cc.max())])
    say(f"E 라벨: prop_intersects {int(pi.sum())} · scenery {int(si.sum())} · 빈 clip 인데 in_frame=1 {int(inf[empty].sum())} · "
        f"튜블릿 조합 {int(pos.sum())} / 없음 {int(none.sum())} (빈 {rep['E']['tub_none_empty']} · 통째로 밖 {rep['E']['tub_none_offscreen']}) / "
        f"제외 {int(ign.sum())} · 조합당 양성 {cc.min()}–{cc.max()} (중앙 {int(np.median(cc))})")

    # ── F 의존 · G 가중치
    bw = np.array([float(r["balance_weight"] or 1) for r in M]); use = pos | none
    rep8 = lambda v: np.repeat(v[:, None], 8, 1)[use]
    Y = pos[use]; Wt = rep8(bw); Hent = float(-(Y.mean() * np.log2(Y.mean()) + (1 - Y.mean()) * np.log2(1 - Y.mean())))
    F = dict(label_entropy=Hent)
    for nm, arr in (("방해물 종류", K), ("구조물", ST), ("방해물 자세", P_), ("arm", SC)):
        X = rep8(arr)
        F[nm] = dict(mi_raw=round(mi(X, Y, np.ones_like(Wt)), 4), mi_weighted=round(mi(X, Y, Wt), 4),
                     p_present={str(v): [round(float(Y[X == v].mean()), 3), round(float(np.average(Y[X == v], weights=Wt[X == v])), 3)]
                                for v in np.unique(X) if nm != "방해물 자세"})
    rep["F"] = F
    say(f"F 의존 (튜블릿, 라벨 엔트로피 {Hent:.3f} bits):")
    for nm in ("방해물 종류", "구조물", "방해물 자세", "arm"):
        f = F[nm]
        say(f"   {nm:8s} MI 원시 {f['mi_raw']:.4f} → 가중 {f['mi_weighted']:.4f}" +
            ("   P(있음) 원시/가중: " + "  ".join(f"{k} {v[0]:.3f}/{v[1]:.3f}" for k, v in f["p_present"].items()) if f["p_present"] else ""))
    G = {}
    for kd in ("none", "pine", "panel"):
        for e in (False, True):
            m = (K == kd) & (empty == e)
            if m.any():
                G[f"{kd}|{'빈' if e else '물체'}"] = [round(float(bw[m].min()), 3), round(float(bw[m].max()), 3)]
    rep["G"] = G
    say("G 가중치 범위 (방해물|물체): " + "  ".join(f"{k} {v[0]}–{v[1]}" for k, v in G.items()))

    # ── H 판 vs v11
    pn = [r for r in M if kind(r) == "panel"]
    if pn:
        V = [r for r in csv.DictReader(V11.open()) if r.get("occ_width_cm")]
        rng = lambda rows, k: [min(float(r[k]) for r in rows), max(float(r[k]) for r in rows)]
        hpx = lambda rows: [round(min(float(r["occ_height_cm"]) / (2 * 769.0909 * (float(r["occ_depth_cm"])) / 1410) * W for r in rows), 1),
                            round(max(float(r["occ_height_cm"]) / (2 * 769.0909 * (float(r["occ_depth_cm"])) / 1410) * W for r in rows), 1)]
        rep["H"] = dict(
            train=dict(color=dict(collections.Counter(r["occ_color"] for r in pn)), depth=rng(pn, "occ_depth_cm"), w_px=rng(pn, "occ_apparent_px"),
                       h_px=hpx(pn), rise_fall=sorted({(int(r["occ_rise_start"]), int(r["occ_fall_end"])) for r in pn}),
                       x=sorted({float(r["occluder_x_cm"]) for r in pn}), n_empty=sum(r["shape_pre"] == "none" for r in pn), n_obj=sum(r["shape_pre"] != "none" for r in pn)),
            v11=dict(color=dict(collections.Counter(r["occ_color"] for r in V)), depth=rng(V, "occ_depth_cm"), w_px=rng(V, "occ_apparent_px"), h_px=hpx(V),
                     rise_fall=sorted({(int(r["occ_rise_start"]), int(r["occ_fall_end"])) for r in V})))
        t, v = rep["H"]["train"], rep["H"]["v11"]
        say(f"H 판: 학습 clip {len(pn)} (물체 {t['n_obj']} · 빈 {t['n_empty']}) · 재질 {t['color']} vs v11 {v['color']} · 깊이 {t['depth']} vs {v['depth']} · "
            f"폭 px {t['w_px']} vs {v['w_px']} · 높이 px {t['h_px']} vs {v['h_px']}")
        say(f"   rise-fall (시작, 끝) 학습 {t['rise_fall']}\n                     v11 {v['rise_fall']}")

    # ── I 누수 (양성 튜블릿의 중심 px)
    m = pos
    cx = ((px.reshape(n, 16, 2).mean(2))[:, T0:])[m]; cy = ((py.reshape(n, 16, 2).mean(2))[:, T0:])[m]
    g = lambda arr: np.repeat(arr[:, None], 8, 1)[m]
    tt = np.tile(np.arange(8), (n, 1))[m]
    rep["I"] = {nm: [round(r2(cx, [gg]), 4), round(r2(cy, [gg]), 4)] for nm, gg in
                (("모양", g(np.array([r["shape_pre"] for r in M]))), ("색", g(np.array([r["color_pre"] for r in M]))),
                 ("배경", g(np.array([r["env"] for r in M]))), ("튜블릿 번호", tt), ("방해물 종류", g(K)), ("구조물", g(ST)))}
    say("I 누수 R² (px_x / px_y): " + "  ".join(f"{k} {v[0]:.4f}/{v[1]:.4f}" for k, v in rep["I"].items()))

    # ── J 커버리지
    band = (cx >= 19) & (cx <= 269) & (cy >= 20) & (cy <= 152)
    cells = set(zip((cx[band] // 18).astype(int), (cy[band] // 18).astype(int)))
    allc = {(i, j) for i in range(19 // 18, 269 // 18 + 1) for j in range(20 // 18, 152 // 18 + 1)}
    rep["J"] = dict(cells_occupied=len(cells & allc), cells_total=len(allc))
    say(f"J 커버리지: 18 px 칸 {len(cells & allc)} / {len(allc)}")

    # ── K 가림 열
    ho = np.array([r.get("has_occlusion") == "1" for r in M])
    hid = np.array([bool(r.get("hidden_start")) and r.get("hidden_start") not in ("", "None") for r in M])
    rep["K"] = dict(has_occlusion=int(ho.sum()), hidden_filled=int(hid.sum()),
                    by_kind={kd: [int((ho & (K == kd)).sum()), int((hid & (K == kd)).sum())] for kd in ("none", "pine", "panel")},
                    panel_depth_vs_object=sorted({(r.get("occ_depth_cm"), r.get("obj_depth_cm")) for r in pn}) if pn else None)
    say(f"K 가림 열: has_occlusion=1 {int(ho.sum())} · hidden_* 채워짐 {int(hid.sum())} · 종류별 {rep['K']['by_kind']} · "
        f"(판 깊이, 물체 깊이) {rep['K']['panel_depth_vs_object']}")

    # ── L 판의 **보이는** 속성 ↔ 있음/없음 (판 clip 의 튜블릿). 한 튜블릿에서 자가 보는 것은 판의 위치 · 폭 · 높이 · 상태다.
    #    2026-09-26: 새 판 빈 장면이 상태는 맞췄지만 위치 (x 3 곳뿐) · 폭×높이 조합 (12 개가 빈 장면에만) 이 달랐다
    if pn:
        def pstate(r, f):
            rs, re_, fs, fe = (int(r[k]) for k in ("occ_rise_start", "occ_rise_end", "occ_fall_start", "occ_fall_end"))
            return "lying" if (f < rs or f > fe) else ("standing" if re_ <= f <= fs else "moving")
        idx = {r["name"]: i for i, r in enumerate(M)}
        Ys, XS = [], collections.defaultdict(list)
        for r in pn:
            i = idx[r["name"]]
            for t in range(8):
                if not (pos[i, t] or none[i, t]):
                    continue
                s2 = [16 + 2 * t, 17 + 2 * t]; st = {pstate(r, 3 * x) for x in s2}; st = "moving" if len(st) > 1 else st.pop()
                Ys.append(int(pos[i, t]))
                XS["위치 x"].append(r["occluder_x_cm"]); XS["폭"].append(r["occ_width_cm"]); XS["높이"].append(r["occ_height_cm"])
                XS["폭×높이"].append(r["occ_width_cm"] + "x" + r["occ_height_cm"]); XS["상태"].append(st)
                XS["전부"].append("|".join([r["occluder_x_cm"], r["occ_width_cm"], r["occ_height_cm"], st]))
        Ys = np.array(Ys); rng = np.random.default_rng(0); L = {}
        for f, v in XS.items():
            v = np.array(v); m_ = mi(v, Ys, np.ones(len(Ys)))
            nul = float(np.mean([mi(v, rng.permutation(Ys), np.ones(len(Ys))) for _ in range(5)]))
            L[f] = dict(mi=round(m_, 4), null=round(nul, 4),
                        p_present={k: round(float(Ys[v == k].mean()), 3) for k in np.unique(v)} if f != "전부" else None)
        rep["L"] = L
        say(f"L 판의 보이는 속성 ↔ 있음 (판 clip 튜블릿 {len(Ys)}, P(있음) {Ys.mean():.3f}): " +
            "  ".join(f"{k} {v['mi']:.4f} (귀무 {v['null']:.4f})" for k, v in L.items()))
        for f in ("위치 x", "폭×높이", "상태"):
            zero = [k for k, v in L[f]["p_present"].items() if v in (0.0, 1.0)]
            say(f"   {f}: P(있음) {min(L[f]['p_present'].values()):.2f}–{max(L[f]['p_present'].values()):.2f}" + (f" · 한쪽만 나오는 값 {zero}" if zero else ""))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{root.name}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    say(f"→ {OUT / (root.name + '.json')}")


if __name__ == "__main__":
    main()
