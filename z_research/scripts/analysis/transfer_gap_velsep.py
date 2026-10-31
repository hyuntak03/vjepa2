#!/usr/bin/env python3
r"""[X1] R3 E33 — 가림막과 분리된 속도 판독: 문맥 끝에 가려진 물체의 속도를 온라인 문맥 encoder 의 가려진 튜블릿이 담나 (2026-10-30).

왜: R2 E24 의 "late z 가 운동 종류 100 % · 가로 속도 R² 0.99" 는 VOBench 가림막 가로 위치가 운동 종류마다 달라 (가속 +213 · 감속 −191 ·
    등속 +12 cm) 가림막 위치만으로 설명된다 ([R] R2). 속도가 가림막과 독립으로 변하는 판이 필요하다.
세트: VOBench 보폭 2 판 (frames_start 15 · stride 2 → 원프레임 15, 17, …, 77; 문맥 = 원프레임 15–45, 문맥 끝은 원판과 같은 45).
    같은 장면 · 같은 가림막 화소에서 물체 화면 속도만 2/3 배. (후보 v11 은 속력이 조건마다 하나, RollOut_v2_training 은 문맥 끝에
    실제로 가려지는 이동 영상이 없다 — BOARD [X1] R3 사전 등록 참고.)
정의
  판 s ∈ {s3 (원판, 원프레임 6t, 6t+3), s2 (원프레임 15+4t, 15+4t+2)}, 튜블릿 t = 0..7 (문맥).
  z = LN(context_encoder(문맥 16 장)) — 릴리즈 ViT-H 온라인 encoder, probing 관례 bfloat16 · autocast 없음 (E21 · transfer_gap_extract 와 같은 함수).
  중심 c_t = 튜블릿 두 장의 label (u, v) 평균 (px@512). 가려짐 h_t = 두 장 hidden 의 최소값. 문맥 끝 속도 v = c_7 − c_6 (px / 튜블릿).
  특징: F7 = z_7 화면 평균 (1280) · F7loc = z_7 의 참 (가려진) 물체 자리 3×3 평균 · Ftl = 마지막으로 다 보인 문맥 튜블릿 (h_t < 0.1) 의 화면 평균 · F67 = [z_6, z_7] 화면 평균.
  공변량 C = 장면 칸 (운동 × objset × 배경 × k) 원핫 + 가림막 가로 위치 · 폭 (cm, label.json block.geometry; 두 판에서 같다).
  ridge (특징 표준화, λ 는 바깥 train 안 4 겹으로). C + F 는 두 단계 (바깥 train 에서 C 로 맞추고 남은 몫을 F 로), block 5 겹 (같은 block 의 두 판 · 두 팔은 같은 겹), 겹 배정 seed 0 · 1.
  R²_C · R²_F · R²_CF (CV) · 부분 R² = 1 − SSE_CF / SSE_C. CI = block bootstrap 2,000 (CV 예측 고정, block 재표집).
  같은 장면 두 판 기울기: Δv̂ = v̂(s3) − v̂(s2) (F 만의 CV 예측) 를 Δv 에 최소제곱 (원점 통과) · 상관.
  보폭 판별: F 로 판 (s3 / s2) 을 ridge 판별 (block 5 겹) — 빈 장면 팔에서 우연 (50 %) 이어야 장면에 보폭 단서가 없다.
세트 (행 = (영상, 판)):
  late23  : late k2 · k3, 이동, 물체 팔, 두 판 모두 h_7 ≥ 0.9 (주 판정)
  late1   : late k1 (t7 반쯤 가려짐, 참고)
  vis     : 가림 없음 (visible), 이동, 물체 팔 — F7 은 보이는 튜블릿 (양성 대조)
  empty23 : late k2 · k3 permanence 빈 장면 팔 (pos_b), 목표 = 같은 block 물체 팔 (pos_a) 의 같은 판 속도 (음성 대조)
판정 (BOARD [X1] R3 사전 등록): 서다 = late23 F7 부분 R² ≥ 0.5 (CI 하한 > 0.3) & 기울기 ≥ 0.5 & empty23 부분 R² ≤ 0.1 ·
  기각 = 부분 R² ≤ 0.1 또는 기울기 ≤ 0.2 · 그 사이 중간 · empty23 부분 R² > 0.2 면 판정 불가.
관문: s3 판 z 가 기존 캐시 z_late_ctx.npy (같은 함수) 와 같은 영상에서 상대 L1 < 1e-2.
입력  tubeprobe meta · data_csv/vobench_v1/index.csv · /data2/local_datasets/world/Benchmark/VOBench_v1/<file_name>/../label.json
출력  /data2/local_datasets/world/world_analysis/cache/transfer_gap/E33/z_{s3,s2}.npy (n, 8, 256, 1280) fp16 + list.json
      z_analysis/transfer_gap/results/q1q2/E33_velsep.json

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/transfer_gap_velsep.py list
  $P z_research/scripts/analysis/transfer_gap_velsep.py extract --gpu 0 [--limit 8]     (GPU 1, 약 10 분)
  $P z_research/scripts/analysis/transfer_gap_velsep.py analyze
  $P z_research/scripts/analysis/transfer_gap_velsep.py transfer      # E33b (사후 추가): 물체 팔 판독기 → 빈 장면 팔, 존재 대비 Δz_7
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_gap_common as C  # noqa: E402

NF, S, D = C.NF, C.S, C.D
ARR = C.TG / "E33"
OUT = C.RES
VOB = Path("/data2/local_datasets/world/Benchmark/VOBench_v1")
VERS = {"s3": (0, 3), "s2": (15, 2)}


def frames_of(ver, t):
    st, sd = VERS[ver]
    return st + sd * 2 * t, st + sd * (2 * t + 1)


def build_list():
    M = C.meta(); idx = {r["video_id"]: r for r in csv.DictReader(open(C.REPO / "data_csv/vobench_v1/index.csv"))}
    items = []
    for j, m in enumerate(M):
        if m["occlusion"] not in ("late", "visible") or m["motion"] == "flat_static":
            continue
        r = idx[m["video_id"]]; bdir = VOB / Path(r["file_name"]).parent
        L = json.load(open(bdir / "label.json")); arm = "A" if m["variant"] == "pos_a" else "B"
        g = L["block"]["geometry"]
        lab = L["labels"][arm]
        tr = {} if lab else None                     # 빈 장면 팔 (permanence pos_b) 은 label 이 없다
        for ver in (VERS if lab else ()):
            cs, hs = [], []
            for t in range(NF):
                f0, f1 = frames_of(ver, t)
                a, b = lab[f0], lab[f1]
                assert a["f"] == f0 and b["f"] == f1
                cs.append([(a["u"] + b["u"]) / 2, (a["v"] + b["v"]) / 2]); hs.append(min(a["hidden"], b["hidden"]))
            tr[ver] = dict(c=cs, h=hs)
        bg = L["block"]["design"]["background"]
        items.append(dict(tp=j, video_id=m["video_id"], block=m["block_id"], variant=m["variant"], objset=m["objset"], violation=m["violation"],
                          occ=m["occlusion"], k=int(m["k"]), motion=m["motion"], label=C.label_of(m), background=bg,
                          panel_x=g.get("panel_x"), panel_w=g.get("panel_w"), track=tr))
    ARR.mkdir(parents=True, exist_ok=True)
    json.dump(items, open(ARR / "list.json", "w"))
    print(len(items), "clips →", ARR / "list.json")
    return items


def extract(gpu, limit):
    import torch, torch.nn as nn, yaml
    sys.path.insert(0, str(C.REPO))
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.data import WMADataset
    from evals.world_model_analysis.attn_probe import _ClipDS
    torch.set_num_threads(6)
    dev = torch.device(f"cuda:{gpu}"); torch.cuda.set_device(dev)
    items = json.load(open(ARR / "list.json"))[: limit or None]
    base = yaml.safe_load(open(C.REPO / "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/_resolved.yaml"))
    base["model"]["dtype"] = "bfloat16"; base["model"]["autocast"] = "none"; base["features"] = {"decode_workers": 6}
    model = build_from_config(base["model"], dev)
    suf = "_smoke" if limit else ""
    for ver, (st, sd) in VERS.items():
        cfg = json.loads(json.dumps(base)); cfg["data"]["frames_start"] = st; cfg["data"]["frames_stride"] = sd
        ds = WMADataset(cfg); vid2rec = {r.video_id: i for i, r in enumerate(ds.records)}
        Z = np.lib.format.open_memmap(ARR / f"z_{ver}{suf}.npy", "w+", np.float16, (len(items), NF, S, D))
        loader = torch.utils.data.DataLoader(_ClipDS(cfg, [vid2rec[it["video_id"]] for it in items]), batch_size=8, shuffle=False,
                                             num_workers=5, pin_memory=True)
        w, t0 = 0, time.time()
        with torch.inference_mode():
            for x in loader:
                b = x.size(0)
                x = x.to(dev, model.dtype, non_blocking=True)
                ci, _ = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=S, batch_size=b, device=dev)
                z = model.context_encoder(x, masks=[ci]); z = z[-1] if isinstance(z, list) else z
                Z[w:w + b] = nn.functional.layer_norm(z.float(), (z.size(-1),)).cpu().numpy().reshape(b, NF, S, -1).astype(np.float16); w += b
                if w % 160 < 8:
                    print(f"{ver} {w}/{len(items)} {time.time() - t0:.0f}s", flush=True)
        Z.flush(); del Z
        print(f"{ver} done {len(items)} in {time.time() - t0:.0f}s", flush=True)
    (ARR / f"DONE{suf}").write_text("ok")


# ---------------------------------------------------------------- 분석

def _fit(Xa, Ya, lam):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; A = (Xa - mu) / sd; ym = Ya.mean(0)
    if A.shape[0] < A.shape[1]:
        W = A.T @ np.linalg.solve(A @ A.T + lam * np.eye(len(A)), Ya - ym)
    else:
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ya - ym))
    return lambda Xb: ((Xb - mu) / sd) @ W + ym


def cv_pred(X, Y, groups, seed, lams=(1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5), k=5):
    ub = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(ub); folds = np.array_split(ub, k)
    pred = np.zeros_like(Y, dtype=np.float64)
    for f in folds:
        te = np.isin(groups, f); tr = ~te
        gtr = groups[tr]; ubt = np.unique(gtr); rng.shuffle(ubt); inner = np.array_split(ubt, 4)
        best, bl = None, None
        for lam in lams:
            err = 0.0
            for g in inner:
                v = np.isin(gtr, g)
                err += ((_fit(X[tr][~v], Y[tr][~v], lam)(X[tr][v]) - Y[tr][v]) ** 2).sum()
            if best is None or err < best:
                best, bl = err, lam
        pred[te] = _fit(X[tr], Y[tr], bl)(X[te])
    return pred


def cv_pred_two_stage(Cv, X, Y, groups, seed, lams=(1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5), k=5):
    """공변량 먼저 (Frisch–Waugh 식): 바깥 train 에서 C 로 맞춘 뒤 남은 몫을 X 로 맞춘다. λ 는 각 단계 안쪽 4 겹. 예측 = Ĉ + X̂(나머지)."""
    ub = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(ub); folds = np.array_split(ub, k)
    pred = np.zeros_like(Y, dtype=np.float64)

    def pick(A, B, gtr):
        ubt = np.unique(gtr); rng.shuffle(ubt); inner = np.array_split(ubt, 4); best, bl = None, None
        for lam in lams:
            err = 0.0
            for g in inner:
                v = np.isin(gtr, g); err += ((_fit(A[~v], B[~v], lam)(A[v]) - B[v]) ** 2).sum()
            if best is None or err < best:
                best, bl = err, lam
        return bl
    for f in folds:
        te = np.isin(groups, f); tr = ~te; gtr = groups[tr]
        fc = _fit(Cv[tr], Y[tr], pick(Cv[tr], Y[tr], gtr)); res = Y[tr] - fc(Cv[tr])
        fx = _fit(X[tr], res, pick(X[tr], res, gtr))
        pred[te] = fc(Cv[te]) + fx(X[te])
    return pred


def r2(y, p):
    return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def boot_partial(y, pc, pcf, groups, n=2000, seed=0):
    ub = np.unique(groups); rng = np.random.default_rng(seed); gi = {g: np.where(groups == g)[0] for g in ub}; vals = []
    for _ in range(n):
        ix = np.concatenate([gi[g] for g in rng.choice(ub, len(ub))])
        vals.append(1 - ((pcf[ix] - y[ix]) ** 2).sum() / ((pc[ix] - y[ix]) ** 2).sum())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def analyze():
    items = json.load(open(ARR / "list.json")); Zs = {v: np.load(ARR / f"z_{v}.npy", mmap_mode="r") for v in VERS}
    out = {"definition": __doc__.split("정의")[1].split("입력")[0].strip(), "n_clips": len(items)}
    # 관문: s3 판 vs 기존 z_late_ctx
    zl = np.load(C.TG / "z_late_ctx.npy", mmap_mode="r"); zm = json.load(open(C.TG / "z_late_meta.json"))["tp"]
    pos = {tp: q for q, tp in enumerate(zm)}; gate = []
    for i, it in enumerate(items):
        if it["tp"] in pos and len(gate) < 20:
            a = np.asarray(Zs["s3"][i], np.float32); b = np.asarray(zl[pos[it["tp"]]], np.float32)
            gate.append(float(np.abs(a - b).mean() / np.abs(b).mean()))
    out["gate_vs_z_late_ctx_relL1"] = dict(n=len(gate), max=max(gate) if gate else None, mean=float(np.mean(gate)) if gate else None,
                                          note="z_late_ctx 는 묶음 4 로 뽑았다 (이번은 8) — bf16 묶음 잡음. 같은 묶음 8 인 z_vis_ctx 와는 아래 관문")
    zv = np.load(C.TG / "z_vis_ctx.npy", mmap_mode="r"); zvm = json.load(open(C.TG / "z_vis_meta.json"))["tp"]; pv = {tp: q for q, tp in enumerate(zvm)}; gv = []
    for i, it in enumerate(items):
        if it["tp"] in pv and len(gv) < 20:
            a = np.asarray(Zs["s3"][i], np.float32); b = np.asarray(zv[pv[it["tp"]]], np.float32); gv.append(float(np.abs(a - b).mean() / np.abs(b).mean()))
    out["gate_vs_z_vis_ctx_relL1"] = dict(n=len(gv), max=max(gv) if gv else None)
    print("gate", out["gate_vs_z_late_ctx_relL1"], flush=True)
    by_block = defaultdict(dict)
    for i, it in enumerate(items):
        by_block[it["block"]][it["variant"]] = i

    def feats(i, ver, ref):
        z = np.asarray(Zs[ver][i], np.float32); tr = items[ref]["track"][ver]     # 빈 장면 팔은 같은 block 물체 팔의 자리를 쓴다
        c7 = tr["c"][7]; reg = C.region(c7)
        tl = [t for t in range(NF) if tr["h"][t] < 0.1]; tl = tl[-1] if tl else 0
        return dict(F7=z[7].mean(0), F7loc=z[7, reg].mean(0), Ftl=z[tl].mean(0), F67=np.concatenate([z[6].mean(0), z[7].mean(0)])), tl

    sets = {}
    for i, it in enumerate(items):
        if it["label"] == "empty":
            continue
        okh = all(it["track"][v]["h"][7] >= 0.9 for v in VERS)
        if it["occ"] == "late" and it["k"] in (2, 3) and okh:
            sets.setdefault("late23", []).append((i, i))
        elif it["occ"] == "late" and it["k"] == 1:
            sets.setdefault("late1", []).append((i, i))
        elif it["occ"] == "visible":
            sets.setdefault("vis", []).append((i, i))
    for b, d in by_block.items():
        if "pos_a" in d and "pos_b" in d and items[d["pos_b"]]["label"] == "empty" and items[d["pos_a"]]["label"] != "empty":
            ia, ie = d["pos_a"], d["pos_b"]; it = items[ia]
            if it["occ"] == "late" and it["k"] in (2, 3) and all(it["track"][v]["h"][7] >= 0.9 for v in VERS):
                sets.setdefault("empty23", []).append((ie, ia))      # 특징은 빈 팔, 목표는 물체 팔
    res = {}
    for sname, pairs in sets.items():
        rows = []; F = defaultdict(list)
        for (fi, yi) in pairs:
            for ver in VERS:
                f, tl = feats(fi, ver, yi); it = items[yi]; tr = it["track"][ver]
                v = np.array(tr["c"][7]) - np.array(tr["c"][6])
                for kk, vv in f.items():
                    F[kk].append(vv)
                rows.append(dict(i=yi, ver=ver, block=it["block"], vu=float(v[0]), vv=float(v[1]), tl=tl, cell=f"{it['motion']}|{it['objset']}|{it['background']}|{it['k']}",
                                 px=float(it["panel_x"] or 0.0), pw=float(it["panel_w"] or 0.0), motion=it["motion"]))
        F = {k: np.stack(v).astype(np.float64) for k, v in F.items()}
        g = np.array([r["block"] for r in rows]); Y = np.array([[r["vu"], r["vv"]] for r in rows])
        cells = sorted({r["cell"] for r in rows})
        Cv = np.array([[1.0 if r["cell"] == c else 0.0 for c in cells] + [r["px"] / 100, r["pw"] / 100] for r in rows])
        verb = np.array([1.0 if r["ver"] == "s3" else 0.0 for r in rows])
        o = dict(n_rows=len(rows), n_block=int(len(np.unique(g))), n_cells=len(cells), v_std_u=float(Y[:, 0].std()), v_std_v=float(Y[:, 1].std()),
                 within_cell_std_u=float(np.sqrt(np.mean([np.var(Y[[r['cell'] == c for r in rows], 0]) for c in cells]))),
                 tl_hist={str(k): int(v) for k, v in zip(*np.unique([r["tl"] for r in rows], return_counts=True))})
        for comp, ci in (("u", 0), ("v", 1)):
            y = Y[:, ci]
            if y.std() < 1e-6:
                continue
            for seed in (0, 1):
                pc = cv_pred(Cv, y[:, None], g, seed)[:, 0]
                o[f"{comp}|s{seed}|R2_C"] = round(r2(y, pc), 4)
                for fk in ("F7", "F7loc", "Ftl", "F67"):
                    pf = cv_pred(F[fk], y[:, None], g, seed)[:, 0]
                    pcf = cv_pred_two_stage(Cv, F[fk], y[:, None], g, seed)[:, 0]
                    part = 1 - ((pcf - y) ** 2).sum() / ((pc - y) ** 2).sum()
                    # 같은 장면 두 판: 예측 차 vs 참 차
                    key = defaultdict(dict)
                    for q, r in enumerate(rows):
                        key[r["i"]][r["ver"]] = q
                    a = np.array([key[i]["s3"] for i in key]); b_ = np.array([key[i]["s2"] for i in key])
                    dv = y[a] - y[b_]; dp = pf[a] - pf[b_]; dpc = pcf[a] - pcf[b_]
                    slope = float((dp * dv).sum() / (dv * dv).sum()); slope_cf = float((dpc * dv).sum() / (dv * dv).sum())
                    corr = float(np.corrcoef(dp, dv)[0, 1]) if dp.std() > 0 else 0.0
                    ent = dict(R2_F=round(r2(y, pf), 4), R2_CF=round(r2(y, pcf), 4), partial_R2=round(float(part), 4),
                               within_scene_slope=round(slope, 3), within_scene_slope_CF=round(slope_cf, 3), within_scene_corr=round(corr, 3),
                               mean_true_dv=round(float(dv.mean()), 3), mean_abs_true_dv=round(float(np.abs(dv).mean()), 3))
                    if seed == 0:
                        ent["partial_R2_CI"] = [round(x, 4) for x in boot_partial(y, pc, pcf, g)]
                    o[f"{comp}|s{seed}|{fk}"] = ent
                    print(sname, comp, seed, fk, ent, flush=True)
        # 보폭 판별 (판 s3 = 1)
        for fk in ("F7", "Ftl"):
            accs = []
            for seed in (0, 1):
                pv = cv_pred(F[fk], verb[:, None], g, seed)[:, 0]
                accs.append(float(((pv > 0.5) == (verb > 0.5)).mean()))
            o[f"stride_decode|{fk}"] = [round(a, 3) for a in accs]
        res[sname] = o
    out["sets"] = res
    d = res.get("late23", {}).get("u|s0|F7", {}); e = res.get("empty23", {}).get("u|s0|F7", {})
    pr, lo, sl, ep = d.get("partial_R2"), (d.get("partial_R2_CI") or [None])[0], d.get("within_scene_slope"), e.get("partial_R2")
    if ep is not None and ep > 0.2:
        verdict = "판정 불가 (빈 장면 대조 부분 R² > 0.2)"
    elif pr is not None and pr >= 0.5 and lo is not None and lo > 0.3 and sl >= 0.5 and (ep is None or ep <= 0.1):
        verdict = "서다"
    elif pr is not None and (pr <= 0.1 or sl <= 0.2):
        verdict = "기각"
    else:
        verdict = "중간"
    out["verdict"] = dict(rule="서다 = late23 F7 부분 R² ≥ 0.5 (CI 하한 > 0.3) & 기울기 ≥ 0.5 & empty23 부분 R² ≤ 0.1 · 기각 = 부분 R² ≤ 0.1 또는 기울기 ≤ 0.2 · "
                               "empty23 > 0.2 면 판정 불가", late23_partial_R2=pr, late23_CI_low=lo, late23_slope=sl, empty23_partial_R2=ep, verdict=verdict)
    print(out["verdict"], flush=True)
    json.dump(out, open(OUT / "E33_velsep.json", "w"), indent=1, ensure_ascii=False)


def transfer():
    """E33b — 물체 팔로 학습한 속도 판독기를 같은 장면 빈 팔에 건다 (겹 밖 block). + 존재 대비 Δz_7 판."""
    items = json.load(open(ARR / "list.json")); Zs = {v: np.load(ARR / f"z_{v}.npy", mmap_mode="r") for v in VERS}
    by_block = defaultdict(dict)
    for i, it in enumerate(items):
        by_block[it["block"]][it["variant"]] = i
    ok = lambda it: it["occ"] == "late" and it["k"] in (2, 3) and all(it["track"][v]["h"][7] >= 0.9 for v in VERS)  # noqa: E731
    obj = [i for i, it in enumerate(items) if it["label"] != "empty" and ok(it)]
    pairs = [(d["pos_a"], d["pos_b"]) for b, d in by_block.items() if "pos_b" in d and items[d["pos_b"]]["label"] == "empty" and ok(items[d["pos_a"]])]

    def F(i, ver, ref, kind):
        z = np.asarray(Zs[ver][i], np.float32)
        if kind == "F7":
            return z[7].mean(0)
        if kind == "F7loc":
            return z[7, C.region(items[ref]["track"][ver]["c"][7])].mean(0)
        return np.concatenate([z[6].mean(0), z[7].mean(0)])
    vel = lambda i, ver: float(items[i]["track"][ver]["c"][7][0] - items[i]["track"][ver]["c"][6][0])  # noqa: E731
    out = {"n_obj_clips": len(obj), "n_perm_blocks": len(pairs)}
    for kind in ("F7", "F7loc", "F67"):
        X = np.stack([F(i, v, i, kind) for i in obj for v in VERS]).astype(np.float64)
        y = np.array([vel(i, v) for i in obj for v in VERS]); g = np.array([items[i]["block"] for i in obj for v in VERS])
        Xe = {(a, v): F(e, v, a, kind) for a, e in pairs for v in VERS}
        res = {}
        for seed in (0, 1):
            ub = np.unique(g); rng = np.random.default_rng(seed); rng.shuffle(ub); folds = np.array_split(ub, 5)
            po, pe = {}, {}
            for f in folds:
                te = np.isin(g, f); tr = ~te; gtr = g[tr]; ubt = np.unique(gtr); rng.shuffle(ubt); inner = np.array_split(ubt, 4); best, bl = None, None
                for lam in (1e-1, 1, 10, 100, 1e3, 1e4, 1e5):
                    err = 0.0
                    for gg in inner:
                        v_ = np.isin(gtr, gg); err += ((_fit(X[tr][~v_], y[tr][~v_, None], lam)(X[tr][v_])[:, 0] - y[tr][v_]) ** 2).sum()
                    if best is None or err < best:
                        best, bl = err, lam
                fn = _fit(X[tr], y[tr][:, None], bl)
                for q in np.where(te)[0]:
                    po[q] = fn(X[q:q + 1])[0, 0]
                fset = set(f.tolist())
                for a, e in pairs:
                    if items[a]["block"] in fset:
                        for v in VERS:
                            pe[(a, v)] = fn(Xe[(a, v)][None].astype(np.float64))[0, 0]
            qi = {(i, v): q for q, (i, v) in enumerate([(i, v) for i in obj for v in VERS])}
            A = [a for a, _ in pairs]
            dv = np.array([vel(a, "s3") - vel(a, "s2") for a in A])
            do = np.array([po[qi[(a, "s3")]] - po[qi[(a, "s2")]] for a in A]); de = np.array([pe[(a, "s3")] - pe[(a, "s2")] for a in A])
            dv_all = np.array([vel(i, "s3") - vel(i, "s2") for i in obj]); do_all = np.array([po[qi[(i, "s3")]] - po[qi[(i, "s2")]] for i in obj])
            bo = float((do * dv).sum() / (dv * dv).sum()); be = float((de * dv).sum() / (dv * dv).sum())
            # 빈 팔의 절대 예측이 물체 팔 예측과 얼마나 닮나 (장면 단서만으로 속도를 내나)
            vo = np.array([po[qi[(a, v)]] for a in A for v in VERS]); ve = np.array([pe[(a, v)] for a in A for v in VERS]); vt = np.array([vel(a, v) for a in A for v in VERS])
            res[f"s{seed}"] = dict(slope_obj_perm=round(bo, 3), slope_obj_all=round(float((do_all * dv_all).sum() / (dv_all * dv_all).sum()), 3),
                                   slope_empty=round(be, 3), rho=round(be / bo, 3) if abs(bo) > 1e-6 else None,
                                   R2_obj_perm=round(r2(vt, vo), 3), R2_empty_vs_truth=round(r2(vt, ve), 3), corr_empty_obj_pred=round(float(np.corrcoef(vo, ve)[0, 1]), 3))
        out[kind] = res; print(kind, res, flush=True)
    # 존재 대비 Δz_7 (물체 − 빈, 같은 장면 · 같은 판) — permanence block 만, block 5 겹 F 만
    for kind in ("F7", "F67"):
        X = np.stack([F(a, v, a, kind) - F(e, v, a, kind) for a, e in pairs for v in VERS]).astype(np.float64)
        y = np.array([vel(a, v) for a, e in pairs for v in VERS]); g = np.array([items[a]["block"] for a, e in pairs for v in VERS])
        rr = {}
        for seed in (0, 1):
            pf = cv_pred(X, y[:, None], g, seed)[:, 0]
            dv = y[0::2] - y[1::2]; dp = pf[0::2] - pf[1::2]
            rr[f"s{seed}"] = dict(R2=round(r2(y, pf), 3), within_scene_slope=round(float((dp * dv).sum() / (dv * dv).sum()), 3), corr=round(float(np.corrcoef(dp, dv)[0, 1]), 3))
        out[f"deltaz|{kind}"] = rr; print("deltaz", kind, rr, flush=True)
    b = out["F7"]["s0"]
    rho = b["rho"]
    out["verdict"] = dict(rule="ρ = b_emp / b_obj ≤ 0.3 → 물체 내용 · ≥ 0.7 → 장면 · 표본 시점 단서로 설명 · 그 사이 섞임", rho_F7_s0=rho, rho_F7_s1=out["F7"]["s1"]["rho"],
                          verdict=("물체 내용" if rho is not None and rho <= 0.3 else "장면 · 표본 시점 단서" if rho is not None and rho >= 0.7 else "섞임"))
    print(out["verdict"], flush=True)
    json.dump(out, open(OUT / "E33b_transfer.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["list", "extract", "analyze", "transfer"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.phase == "list":
        build_list()
    elif a.phase == "extract":
        extract(a.gpu, a.limit)
    elif a.phase == "transfer":
        transfer()
    else:
        analyze()
