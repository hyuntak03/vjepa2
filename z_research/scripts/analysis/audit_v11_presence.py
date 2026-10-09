#!/usr/bin/env python3
"""감사 — `RollOutV3/figures/v11_presence` 의 주장을 다시 계산한다 (2026-09-25).

무엇을 다시 재나 (카드의 주장 (a)~(e) + 복사 기준선 + k 별 + mid):
  (a) last 가림 (k=2..4) 의 움직이는 물체 (flat · ramp) 는 p 의 미래 t0~t4/t5 에서 '없다' 로 읽히고 visible · mid 는 '있다'
  (b) 문맥 끝에 가려진 멈춘 물체 (static) 는 '있다'
  (c) 자 검사 — h 는 물체가 보이고 가장자리에서 36 px 이상 안쪽인 칸에서 '있다' 로 읽는가
  (d) 가장자리 · 화면 밖 칸 (flat t5~t7, ramp t6~t7) 은 해석할 수 없다
  (e) 빈 장면 (vanish pos_b, obj=False) 의 오탐 바닥 — 튜블릿마다
  + 궤적 구조 (진실 (32,2) 을 1 px 로 반올림해 해시) · 궤적 단위 부트스트랩
  + 복사 기준선 — 문맥 마지막 튜블릿 (idx 7) 을 자로 읽은 값을 미래 8 튜블릿 전부의 예측으로 둔다
      copy_zfull : readings.npz 의 z[:,7]  (context encoder 가 **32 샘플 전부**를 본 것 — 미래가 새어 들어올 수 있다)
      copy_zctx  : context encoder 를 **문맥 16 샘플만**으로 다시 돌린 튜블릿 7 (predictor 입력과 같은 조건, 새지 않음)
      copy_hctx  : target encoder 를 문맥 16 샘플만으로 돌린 튜블릿 7
      copy_gt    : 위치만 — 문맥에서 마지막으로 보인 샘플의 진실 위치
  + 문턱 하나 (자마다 thr_val_fpr5), 문턱 없는 AUROC (물체 vs 빈 장면, 같은 조건) 를 보조로

두 단계:
  extract : GPU. 문맥 16 샘플만으로 z·h 를 돌려 8 튜블릿을 자로 읽는다 → _audit/v11_presence/ctx_only_readings.npz
  analyze : CPU. 모든 표 · results.json

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  CUDA_VISIBLE_DEVICES=7 $P z_research/scripts/analysis/audit_v11_presence.py extract --gpus 1
  $P z_research/scripts/analysis/audit_v11_presence.py analyze
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import PRES, EXP, check, fingerprint          # noqa: E402

OUT = ROOT / "z_research/RollOutV3/audit/v11_presence"
RES, TC, TP, KMIN, EDGE, W = 144.0, 8, 8, 2, 36.0, 288.0
NB = 2000
MOTIONS = ("static", "flat", "ramp")


def motion(c):
    return "static" if c.startswith("static") else ("flat" if "flat" in c else "ramp")


# ----------------------------------------------------------------------------------------------- extract (GPU)
def _worker(rank, world, n, tmp, bs):
    import torch, torch.nn.functional as F
    from rollout3_window_readout import MODEL, readout_cls, prefetch
    from analysis.intphys2.model import build_from_config
    from evals.world_model_analysis.data import WMADataset
    torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
    NS, S, D = 16, 256, 1280
    R = readout_cls(); heads = {}
    for rep in ("z", "h"):
        hd = R("attn").to(dev).eval()
        hd.load_state_dict(torch.load(PRES / rep / "readout_attn.pt", map_location="cpu")); heads[rep] = hd
    ln = lambda x: F.layer_norm(x, (x.size(-1),))
    bundle = build_from_config({**MODEL, "window_size": NS}, dev)
    data = dict(root=str(tmp), index_csv="index.csv", frames_root="/local_datasets/world/world_analysis/IntPhysGen_v11",
                frames_pattern="{file_name}/{frame:06d}.png", frames_start=0, frames_stride=3, n_frames=NS, resolution=256)
    ds = WMADataset(dict(data=data, model={**MODEL, "window_size": NS}, features={"cache_dir": "/tmp"}, surprise={}))
    mm = np.lib.format.open_memmap(tmp / "ctx.npy", mode="r+")
    idx = list(range(rank, n, world)); pool = ThreadPoolExecutor(max_workers=8); t0 = time.time(); done = 0
    with torch.no_grad():
        for ids, clips in prefetch(ds, idx, bs, pool):
            B = clips.size(0); ids = np.asarray(ids)
            clips = clips.to(dev, dtype=bundle.dtype, non_blocking=True)
            for j, (rep, mod) in enumerate((("z", "context_encoder"), ("h", "target_encoder"))):
                fo = getattr(bundle, mod)(clips); fo = fo[-1] if isinstance(fo, list) else fo
                tok = ln(fo.float()).reshape(B * (NS // 2), S, D).half()
                xy, pres, amap = heads[rep](tok.float())
                mm[ids, j, :, 0:2] = xy.reshape(B, NS // 2, 2).cpu().numpy()
                mm[ids, j, :, 2] = pres.reshape(B, NS // 2).cpu().numpy()
                mm[ids, j, :, 3] = amap.max(-1).values.reshape(B, NS // 2).cpu().numpy()
            done += B
            if rank == 0 and done % (bs * 20) < bs:
                el = time.time() - t0
                print(f"  {done*world:,}/{n:,}  {el:.0f}s  남은 {el*(n/(done*world)-1)/60:.1f}분", flush=True)
    mm.flush()


def extract(a):
    import torch.multiprocessing as mp
    import v11_readout_online as V
    tmp = Path(os.environ.get("AUDIT_TMP", "/tmp/claude-1070/-data-hyuntak-project-2026-2027-cvpr-vjepa2/"
                              "4cf9ae70-0bc8-4a81-8199-b3268c1624a3/scratchpad/audit/ctx_only"))
    tmp.mkdir(parents=True, exist_ok=True)
    rows, vids = [], []
    for timing, name in (("late", "v11"), ("mid", "v11_mid")):
        r, *_ = V.select(timing)
        z = np.load(EXP / name / "readings.npz", allow_pickle=True)
        assert [x["video_id"] for x in r] == list(z["video_id"]), f"{name}: index 순서가 readings 와 다르다"
        rows += r; vids += [x["video_id"] for x in r]
    if a.limit:
        rows, vids = rows[:a.limit], vids[:a.limit]
    n = len(rows)
    with (tmp / "index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    np.lib.format.open_memmap(tmp / "ctx.npy", mode="w+", dtype=np.float32, shape=(n, 2, 8, 4))
    print(f"[extract] {n:,} clip · 문맥 16 샘플만 · z/h", flush=True)
    t0 = time.time()
    mp.spawn(_worker, args=(a.gpus, n, tmp, a.bs), nprocs=a.gpus, join=True)
    print(f"[extract] {(time.time()-t0)/60:.1f}분", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(np.load(tmp / "ctx.npy"))
    np.savez_compressed(OUT / ("ctx_only_readings" + ("_smoke" if a.limit else "") + ".npz"),
                        video_id=np.array(vids), z=arr[:, 0], h=arr[:, 1], decoder_fp=np.array(fingerprint(PRES)))
    print("→", OUT)


# ----------------------------------------------------------------------------------------------- analyze (CPU)
S_ = json.loads((PRES / "summary.json").read_text())
THR = {r: S_["reps"][r]["attn"]["thr_val_fpr5"] for r in ("p", "z", "h")}
BIAS = {r: np.array(v, np.float32) for r, v in json.loads((PRES / "attn_bias_px.json").read_text()).items()}
rng = np.random.default_rng(0)


def load():
    ctx = None
    f = OUT / "ctx_only_readings.npz"
    if f.exists():
        ctx = np.load(f, allow_pickle=True); check(ctx, PRES)
        ctxmap = {v: i for i, v in enumerate(ctx["video_id"])}
    D = {}
    for name in ("v11", "v11_mid"):
        z = np.load(EXP / name / "readings.npz", allow_pickle=True); check(z, PRES)
        n = len(z["video_id"]); k = z["sym_k"].astype(int)
        Ts = z["truth"] * RES + RES
        d = dict(name=name, n=n, k=k, Ts=Ts, T=Ts.reshape(n, 16, 2, 2).mean(2),
                 hid=z["hidden"].reshape(n, 16, 2).any(-1), inf=z["in_frame"].reshape(n, 16, 2).all(-1),
                 obj=z["obj"], mot=np.array([motion(c) for c in z["condition"]]), cond=z["condition"],
                 tim=np.where(k == 0, "visible", "last" if name == "v11" else "mid"),
                 blk=np.array([name + ":" + b for b in z["block_id"]]), tdir=z["travel_dir"], vid=z["video_id"],
                 vtype=z["violation_type"], variant=z["variant"])
        d["inside"] = ((d["T"] >= EDGE) & (d["T"] <= W - EDGE)).all(-1)
        d["visin"] = ~d["hid"] & d["inf"] & d["inside"]               # 보이고 · 화면 안 · 가장자리 36 px 안쪽
        hc = z["hidden"][:, :16]
        d["sseen"] = np.array([max([s for s in range(16) if not hc[i, s]], default=-1) for i in range(n)])
        for rep in ("p", "z", "h"):
            v = z[rep]
            d[rep + "_xy"] = v[..., 0:2] * RES + RES - BIAS[rep]
            d[rep + "_lg"] = v[..., 2]; d[rep + "_say"] = v[..., 2] > THR[rep]; d[rep + "_top1"] = v[..., 3]
        if ctx is not None:
            ii = np.array([ctxmap[v] for v in z["video_id"]])
            for rep in ("z", "h"):
                v = ctx[rep][ii]                                     # (n,8,4) 문맥 튜블릿 0..7
                d["c" + rep + "_xy"] = v[..., 0:2] * RES + RES - BIAS[rep]
                d["c" + rep + "_lg"] = v[..., 2]; d["c" + rep + "_say"] = v[..., 2] > THR[rep]
        # 궤적 해시 — 진실 (32,2) 을 1 px 로 반올림
        tr = np.round(Ts).astype(np.int32)
        d["traj"] = np.array([hashlib.sha1(t.tobytes()).hexdigest()[:10] for t in tr])
        D[name] = d
    return D, ctx is not None


def sel(D, tim, mo, objflag=True, ks=None):
    d = D["v11_mid"] if tim == "mid" else D["v11"]
    m = (d["obj"] == objflag) & (d["tim"] == tim) & (d["mot"] == mo)
    if tim != "visible":
        m &= (d["k"] >= KMIN) if ks is None else np.isin(d["k"], ks)
    return d, m


def cluster_boot(vals, units, nb=NB):
    """vals (n, T) 를 units 로 묶어 재표집 (units 안의 clip 도 다시 뽑는 두 단계). → (T,2) 2.5/97.5 %."""
    vals = np.asarray(vals, float)
    if vals.ndim == 1:
        vals = vals[:, None]
    uu, inv = np.unique(units, return_inverse=True)
    groups = [np.where(inv == u)[0] for u in range(len(uu))]
    out = np.empty((nb, vals.shape[1]))
    for b in range(nb):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in (groups[p] for p in pick)])
        out[b] = np.nanmean(vals[idx], 0)
    return np.percentile(out, [2.5, 97.5], axis=0).T


def auroc(pos, neg):
    """문턱 없는 분리 — P(logit_obj > logit_empty). 순위 기반 (동점 0.5)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort().astype(float) + 1
    # 동점 평균 순위
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def l2(a, b):
    return np.linalg.norm(a - b, axis=-1)


def r3(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r3(v) for v in x]
    x = float(x)
    return None if not np.isfinite(x) else round(x, 3)


def trajectory_structure(D):
    out = {}
    for name, d in D.items():
        for tim in np.unique(d["tim"]):
            for mo in MOTIONS:
                for ob in (True, False):
                    m = (d["tim"] == tim) & (d["mot"] == mo) & (d["obj"] == ob)
                    if tim != "visible":
                        m &= d["k"] >= KMIN
                    if not m.any():
                        continue
                    c = Counter(d["traj"][m])
                    perk = {int(kk): len(set(d["traj"][m & (d["k"] == kk)])) for kk in np.unique(d["k"][m])}
                    out[f"{mo}|{tim}|{'obj' if ob else 'empty'}"] = dict(
                        n_clip=int(m.sum()), n_traj=len(c), copies=sorted(c.values()), n_traj_per_k=perk,
                        n_block=len(set(d["blk"][m])), dirs=dict(Counter(map(int, d["tdir"][m]))))
    # 같은 운동에서 timing · k 를 가로질러 궤적이 같은가
    cross = {}
    for mo in MOTIONS:
        sets = {}
        for name, d in D.items():
            for tim in np.unique(d["tim"]):
                m = d["obj"] & (d["tim"] == tim) & (d["mot"] == mo)
                for kk in np.unique(d["k"][m]):
                    sets[f"{tim}_k{kk}"] = set(d["traj"][m & (d["k"] == kk)])
        allt = set().union(*sets.values())
        cross[mo] = dict(n_distinct_total=len(allt), per_cell={k: len(v) for k, v in sets.items()},
                         shared_by_all=len(set.intersection(*sets.values())))
    # 진실 x 궤적 (방향별 대표) — 기술용
    return out, cross


def unit_ids(d, m, mo):
    """부트스트랩 단위. 궤적이 k·timing 을 가로질러 거의 같으면 (궤적, k) 를 쓴다 — 가림막 폭이 k 와 함께 바뀌어
    장면이 다르다. static 은 자리 8 개라 궤적만으로도 단위가 된다."""
    return np.array([f"{t}|{kk}" for t, kk in zip(d["traj"][m], d["k"][m])])


def main_analyze():
    D, have_ctx = load()
    OUT.mkdir(parents=True, exist_ok=True)
    R = {"thr": THR, "have_ctx_only_copy": have_ctx}
    ts, cross = trajectory_structure(D)
    R["trajectory_structure"] = ts; R["trajectory_cross"] = cross
    lines = []
    P = lines.append
    f3 = lambda a: " ".join(f"{x:5.2f}" if (x is not None and np.isfinite(x)) else "   · " for x in a)
    f1 = lambda a: " ".join(f"{x:5.0f}" if (x is not None and np.isfinite(x)) else "    ·" for x in a)
    P(f"문턱 {json.dumps({k: round(v, 3) for k, v in THR.items()})}  ctx-only 복사 {'있음' if have_ctx else '없음'}\n")
    P("## 궤적 구조 (진실 32x2 를 1 px 반올림 해시, k>=2)")
    for k_, v in ts.items():
        P(f"  {k_:22s} clip {v['n_clip']:5d}  궤적 {v['n_traj']:3d}  k별 {v['n_traj_per_k']}  복사수 {sorted(set(v['copies']))}  block {v['n_block']}  방향 {v['dirs']}")
    for mo, v in cross.items():
        P(f"  {mo}: 전체 distinct {v['n_distinct_total']} · 모든 칸 공유 {v['shared_by_all']} · {v['per_cell']}")

    cells = {}
    for tim in ("visible", "mid", "last"):
        for mo in MOTIONS:
            d, m = sel(D, tim, mo); d0, me = sel(D, tim, mo, objflag=False)
            U = unit_ids(d, m, mo); Ue = d0["blk"][me]
            Tf = d["T"][:, TC:]; vis_in = d["visin"][:, TC:]
            c = {"n_obj": int(m.sum()), "n_empty": int(me.sum()), "n_units": len(set(U))}
            # presence
            for rep, key, off in (("p", "p", 0), ("h", "h", TC), ("z", "z", TC)):
                say = d[rep + "_say"][:, off:off + TP]
                c[key] = say[m].mean(0); c[key + "_ci"] = cluster_boot(say[m], U)
                c[key + "_null"] = say[me].mean(0); c[key + "_null_ci"] = cluster_boot(say[me], Ue)
                lg = d[rep + "_lg"][:, off:off + TP]
                c[key + "_auroc"] = np.array([auroc(lg[m, t], lg[me, t]) for t in range(TP)])
            # 방향 (궤적) 별 p · h
            for dd in (1, -1):
                mm = m & (d["tdir"] == dd)
                if mm.any():
                    c[f"p_dir{dd:+d}"] = d["p_say"][mm].mean(0); c[f"h_dir{dd:+d}"] = d["h_say"][mm, TC:].mean(0)
            # (궤적, k) 단위 값의 범위
            ug = defaultdict(list)
            for i in np.where(m)[0]:
                ug[f"{d['traj'][i]}|{d['k'][i]}"].append(i)
            uv = np.array([d["p_say"][g].mean(0) for g in ug.values()])
            c["p_unit_min"], c["p_unit_max"] = uv.min(0), uv.max(0)
            # 진실 가시성
            c["frac_visible"] = (~d["hid"][:, TC:] & d["inf"][:, TC:])[m].mean(0)
            c["frac_vis_in"] = vis_in[m].mean(0)
            # 보이고 안쪽인 칸만의 p · h (자 검사 (c))
            c["p_visin"] = np.array([d["p_say"][m & vis_in[:, t], t].mean() if (m & vis_in[:, t]).any() else np.nan for t in range(TP)])
            c["h_visin"] = np.array([d["h_say"][m & vis_in[:, t], TC + t].mean() if (m & vis_in[:, t]).any() else np.nan for t in range(TP)])
            c["z_visin"] = np.array([d["z_say"][m & vis_in[:, t], TC + t].mean() if (m & vis_in[:, t]).any() else np.nan for t in range(TP)])
            # 복사 — presence: 문맥 튜블릿 7 을 미래 전부에
            c["copy_zfull"] = float(d["z_say"][m, TC - 1].mean()); c["copy_zfull_null"] = float(d["z_say"][me, TC - 1].mean())
            c["copy_hfull"] = float(d["h_say"][m, TC - 1].mean()); c["copy_hfull_null"] = float(d["h_say"][me, TC - 1].mean())
            c["copy_zfull_auroc"] = auroc(d["z_lg"][m, TC - 1], d["z_lg"][me, TC - 1])
            if have_ctx:
                for rep in ("z", "h"):
                    s7 = d["c" + rep + "_say"][:, 7]
                    c[f"copy_{rep}ctx"] = float(s7[m].mean()); c[f"copy_{rep}ctx_null"] = float(s7[me].mean())
                    c[f"copy_{rep}ctx_ci"] = cluster_boot(s7[m], U)[0]
                    c[f"copy_{rep}ctx_auroc"] = auroc(d["c" + rep + "_lg"][m, 7], d["c" + rep + "_lg"][me, 7])
                # p − 복사 (clip 짝지은 차이, (궤적,k) 단위 CI)
                dif = d["p_say"][m].astype(float) - d["cz_say"][m, 7:8].astype(float)
                c["p_minus_copyzctx"] = dif.mean(0); c["p_minus_copyzctx_ci"] = cluster_boot(dif, U)
                # ctx-only 와 full-clip z 의 문맥 튜블릿 일치 (sanity)
                c["ctx_vs_full_z_agree_t0_7"] = (d["cz_say"][m] == d["z_say"][m, :TC]).mean(0)
            # 위치 — p 가 '있다' 고 한 clip 만
            say = d["p_say"]
            xseen = d["Ts"][np.arange(d["n"]), np.maximum(d["sseen"], 0)]
            dflt = d["p_xy"][me].mean(0)                               # 빈 장면에서 p 자가 내는 평균 좌표 (자 기본값)
            # 같은 궤적 · 같은 k 의 빈 장면에서 p 자가 내는 평균 좌표 (가림막 · 판이 같은 자리) — 위치 귀무
            dft = np.full((d["n"], TP, 2), np.nan)
            for i in np.where(m)[0]:
                mm_ = me & (d["traj"] == d["traj"][i]) & (d["k"] == d["k"][i])
                if mm_.any():
                    dft[i] = d["p_xy"][mm_].mean(0)
            copyxy_z = d["z_xy"][:, TC - 1]
            cz = d["cz_xy"][:, 7] if have_ctx else None
            for nm in ("p_l2", "h_l2", "copygt_l2", "copyz_l2", "copyzctx_l2", "dflt_l2", "p_to_copygt", "p_to_dflt",
                       "otherdir_l2", "p_l2_visin", "p_adv", "gt_adv", "p_top1", "h_top1", "dflt_traj_l2", "p_to_dflt_traj"):
                c[nm] = np.full(TP, np.nan)
            for t in range(TP):
                s = m & say[:, t]
                if s.sum() == 0:
                    continue
                c["p_l2"][t] = l2(d["p_xy"][s, t], Tf[s, t]).mean()
                c["copygt_l2"][t] = l2(xseen[s], Tf[s, t]).mean()
                c["copyz_l2"][t] = l2(copyxy_z[s], Tf[s, t]).mean()
                if have_ctx:
                    c["copyzctx_l2"][t] = l2(cz[s], Tf[s, t]).mean()
                c["dflt_l2"][t] = l2(dflt[t][None], Tf[s, t]).mean()
                c["p_to_copygt"][t] = l2(d["p_xy"][s, t], xseen[s]).mean()
                c["p_to_dflt"][t] = l2(d["p_xy"][s, t], dflt[t][None]).mean()
                c["dflt_traj_l2"][t] = np.nanmean(l2(dft[s, t], Tf[s, t]))
                c["p_to_dflt_traj"][t] = np.nanmean(l2(d["p_xy"][s, t], dft[s, t]))
                sv = s & vis_in[:, t]
                if sv.any():
                    c["p_l2_visin"][t] = l2(d["p_xy"][sv, t], Tf[sv, t]).mean()
                if mo != "static":
                    c["p_adv"][t] = ((d["p_xy"][s, t, 0] - xseen[s, 0]) * d["tdir"][s]).mean()
                    c["gt_adv"][t] = ((Tf[s, t, 0] - xseen[s, 0]) * d["tdir"][s]).mean()
                    # 위치 귀무: 반대 방향 궤적의 같은 튜블릿 진실 (같은 운동 · 같은 timing 의 다른 궤적)
                    oth = []
                    for dd in (1, -1):
                        ss = s & (d["tdir"] == dd); mo_ = m & (d["tdir"] == -dd)
                        if ss.any() and mo_.any():
                            oth.append(l2(d["p_xy"][ss, t], Tf[mo_, t].mean(0)[None]))
                    if oth:
                        c["otherdir_l2"][t] = np.concatenate(oth).mean()
                sh = m & d["h_say"][:, TC + t]
                if sh.any():
                    c["h_l2"][t] = l2(d["h_xy"][sh, TC + t], Tf[sh, t]).mean()
            c["p_top1"] = d["p_top1"][m].mean(0); c["h_top1"] = d["h_top1"][m, TC:].mean(0)
            cells[f"{mo}|{tim}"] = c

    # k 별 (last · mid)
    perk = {}
    for tim in ("last", "mid"):
        for mo in MOTIONS:
            for kk in (1, 2, 3, 4):
                d, m = sel(D, tim, mo, ks=[kk]); d0, me = sel(D, tim, mo, objflag=False, ks=[kk])
                e = dict(n=int(m.sum()), p=d["p_say"][m].mean(0), h=d["h_say"][m, TC:].mean(0),
                         null=d["p_say"][me].mean(0), vis=(~d["hid"][:, TC:] & d["inf"][:, TC:])[m].mean(0),
                         copy_zfull=float(d["z_say"][m, TC - 1].mean()),
                         p_ci=cluster_boot(d["p_say"][m], unit_ids(d, m, mo)))
                if have_ctx:
                    e["copy_zctx"] = float(d["cz_say"][m, 7].mean()); e["copy_hctx"] = float(d["ch_say"][m, 7].mean())
                    e["copy_zctx_null"] = float(d["cz_say"][me, 7].mean())
                perk[f"{mo}|{tim}|k{kk}"] = e

    # ------------------------------------------------------------------ 출력 (표)
    for tim in ("visible", "mid", "last"):
        for mo in MOTIONS:
            c = cells[f"{mo}|{tim}"]
            P(f"\n## {tim} · {mo}   clip {c['n_obj']} (단위 (궤적,k) {c['n_units']}) · 빈 장면 {c['n_empty']}")
            P(f"  진실 보임           {f3(c['frac_visible'])}")
            P(f"  진실 보임&안쪽      {f3(c['frac_vis_in'])}")
            P(f"  p '있다'            {f3(c['p'])}")
            P(f"    CI lo (궤적,k)    {f3(c['p_ci'][:, 0])}")
            P(f"    CI hi             {f3(c['p_ci'][:, 1])}")
            P(f"    단위 min          {f3(c['p_unit_min'])}")
            P(f"    단위 max          {f3(c['p_unit_max'])}")
            for dd in ("+1", "-1"):
                if f"p_dir{dd}" in c:
                    P(f"    방향 {dd}          {f3(c['p_dir' + dd])}   | h {f3(c['h_dir' + dd])}")
            P(f"  p 빈장면 (귀무)     {f3(c['p_null'])}")
            P(f"    귀무 CI hi        {f3(c['p_null_ci'][:, 1])}")
            P(f"  p AUROC 물체vs빈    {f3(c['p_auroc'])}")
            P(f"  h '있다'            {f3(c['h'])}   (귀무 {f3(c['h_null'])})")
            P(f"  h AUROC             {f3(c['h_auroc'])}")
            P(f"  z(full) '있다'      {f3(c['z'])}")
            P(f"  p  보임&안쪽만      {f3(c['p_visin'])}")
            P(f"  h  보임&안쪽만      {f3(c['h_visin'])}")
            cp = f"  복사 presence: z_full t7 {c['copy_zfull']:.3f} (빈 {c['copy_zfull_null']:.3f}, AUROC {c['copy_zfull_auroc']:.3f})"
            cp += f" · h_full t7 {c['copy_hfull']:.3f} (빈 {c['copy_hfull_null']:.3f})"
            if have_ctx:
                cp += (f"\n                 z_ctx t7 {c['copy_zctx']:.3f} [{c['copy_zctx_ci'][0]:.3f},{c['copy_zctx_ci'][1]:.3f}] (빈 {c['copy_zctx_null']:.3f}, AUROC {c['copy_zctx_auroc']:.3f})"
                       f" · h_ctx t7 {c['copy_hctx']:.3f} (빈 {c['copy_hctx_null']:.3f}, AUROC {c['copy_hctx_auroc']:.3f})")
                cp += f"\n                 ctx-only z vs full z 문맥 튜블릿 일치율 {f3(c['ctx_vs_full_z_agree_t0_7'])}"
                cp += (f"\n  p − 복사(z_ctx)     {f3(c['p_minus_copyzctx'])}"
                       f"\n    CI lo             {f3(c['p_minus_copyzctx_ci'][:, 0])}"
                       f"\n    CI hi             {f3(c['p_minus_copyzctx_ci'][:, 1])}")
            P(cp)
            P(f"  p L2 ('있다')       {f1(c['p_l2'])}")
            P(f"  p L2 보임&안쪽      {f1(c['p_l2_visin'])}")
            P(f"  h L2                {f1(c['h_l2'])}")
            P(f"  복사 L2 GT마지막    {f1(c['copygt_l2'])}")
            P(f"  복사 L2 z_full t7   {f1(c['copyz_l2'])}")
            if have_ctx:
                P(f"  복사 L2 z_ctx t7    {f1(c['copyzctx_l2'])}")
            P(f"  기본값 L2 (빈장면 p 평균좌표) {f1(c['dflt_l2'])}")
            P(f"  |p - GT마지막|      {f1(c['p_to_copygt'])}")
            P(f"  |p - 기본값|        {f1(c['p_to_dflt'])}")
            P(f"  빈장면 p 좌표 (같은 궤적·k) L2 {f1(c['dflt_traj_l2'])}")
            P(f"  |p - 같은 궤적 빈장면 p| {f1(c['p_to_dflt_traj'])}")
            if mo != "static":
                P(f"  다른 방향 L2 (귀무) {f1(c['otherdir_l2'])}")
                P(f"  x 전진 p            {f1(c['p_adv'])}")
                P(f"  x 전진 진실         {f1(c['gt_adv'])}")
            P(f"  top1 p / h          {f3(c['p_top1'])} / {f3(c['h_top1'])}")
    P("\n## k 별")
    for key, e in perk.items():
        s = f"  {key:14s} n {e['n']:4d}  p {f3(e['p'])} | h {f3(e['h'])} | 귀무 {f3(e['null'])} | 보임 {f3(e['vis'])} | copy z_full {e['copy_zfull']:.2f}"
        if have_ctx:
            s += f" z_ctx {e['copy_zctx']:.2f} (빈 {e['copy_zctx_null']:.2f}) h_ctx {e['copy_hctx']:.2f}"
        P(s)

    # 자 없는 교차검증 — v11_full 표준 surprise (matched), vanish 쌍 A (문맥에 물체: pos_a 미래=물체 vs imp_ab 미래=빈)
    # "p 가 물체 미래 쪽에 더 가깝다" = S(pos_a) < S(imp_ab). 쌍 B (문맥 빈 장면) 는 대조.
    PB = ROOT / "z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/surprise_c16t32__v11_full_vith/per_block.json"
    IX = ROOT / "data_csv/intphysgen_v11_full/index.csv"
    lf = {}
    if PB.exists() and IX.exists():
        sv = json.loads(PB.read_text())["per_video_surprise"]
        byb = defaultdict(dict)
        for r in csv.DictReader(IX.open()):
            if r["violation_type"] == "vanish":
                byb[r["block_id"]][r["variant"]] = r
        acc = defaultdict(list)
        for b, vv in byb.items():
            r = vv["pos_a"]; cond = r["condition"]
            tim = "visible" if "visible" in cond else ("mid" if cond.endswith("_mid") else ("early" if cond.endswith("_early") else "last"))
            kk = int(r["video_id"].split("_k")[1].split("_")[0]) if "_k" in r["video_id"] else 0
            mo = motion(cond)
            if all(v in sv for v in (vv["pos_a"]["video_id"], vv["imp_ab"]["video_id"], vv["pos_b"]["video_id"], vv["imp_ba"]["video_id"])):
                A = sv[vv["pos_a"]["video_id"]] < sv[vv["imp_ab"]["video_id"]]
                B = sv[vv["pos_b"]["video_id"]] < sv[vv["imp_ba"]["video_id"]]
                acc[(tim, mo, kk)].append((A, B))
        for key in sorted(acc):
            a_ = np.array(acc[key], float)
            ci = cluster_boot(a_[:, 0], np.arange(len(a_)))[0]
            lf["|".join(map(str, key))] = dict(n_block=len(a_), A_obj_ctx=r3(a_[:, 0].mean()), A_ci=r3(ci), B_empty_ctx=r3(a_[:, 1].mean()))
        P("\n## 자 없는 교차검증 (v11_full surprise_c16t32, vanish 쌍 A: p 가 물체 미래에 더 가까운 block 비율 / 쌍 B: 빈 미래에 더 가까운 비율)")
        for key, e in lf.items():
            P(f"  {key:18s} n {e['n_block']:3d}  A {e['A_obj_ctx']:.3f} [{e['A_ci'][0]:.3f},{e['A_ci'][1]:.3f}]  B {e['B_empty_ctx']:.3f}")
    R["label_free_vanish"] = lf

    R["cells"] = {k: {kk: r3(vv) if not isinstance(vv, int) else vv for kk, vv in v.items()} for k, v in cells.items()}
    R["per_k"] = {k: {kk: r3(vv) if not isinstance(vv, int) else vv for kk, vv in v.items()} for k, v in perk.items()}
    (OUT / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
    (OUT / "tables.txt").write_text("\n".join(lines))
    print("\n".join(lines)); print("→", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "analyze"])
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    extract(a) if a.stage == "extract" else main_analyze()
