#!/usr/bin/env python3
r"""[X1] R2 E25 — 가중치 보간 · 층 묶음 교환: predictor 가 만든 출력에서 이식과 영속이 같이 움직이나 (H2-7) (2026-10-30).

정의
  predictor 가중치 두 벌: 릴리즈 θ_R (z_training/runs/release_vith/latest.pt) · VOBench PFT θ_P (z_training/runs/vobench_release_postft_g/latest.pt).
  판 11 개 (학습 없음, 가중치를 섞기만 한다):
    lam{0,25,50,75,100} : θ = (1 − λ) θ_R + λ θ_P
    onlyP_G{1,2,3}      : 묶음 G 만 θ_P, 나머지 θ_R
    onlyR_G{1,2,3}      : 묶음 G 만 θ_R, 나머지 θ_P
    G1 = predictor_embed (입력 사상) + mask_tokens + predictor_blocks.0–3 / G2 = blocks.4–7 / G3 = blocks.8–11 + predictor_norm + predictor_proj
  묶음 크기 4 (캐시와 같음; 8 이면 bf16 묶음 잡음으로 캐시 대비 상대 L1 3.5 % 차 — 2026-10-30 첫 실행). 문맥 encoder (온라인, 릴리즈 ViT-H) 는 영상마다 한 번만 돌리고 판마다 predictor 만 다시 돌린다. 정밀도 = probing 관례 (bfloat16, autocast none)
  — 캐시 p (tubeprobe) · p_relftg 와 같은 경로. λ = 0 / 1 판이 캐시와 같은지 상대 L1 로 관문을 낸다.
판마다 지표 (split v2 test block 의 가능 영상 720 개):
  s_pos(j) = mean_{t,c,d} |p_j − h_j| , s_imp(j) = mean |p_j − h_imp(j)|  (p = predictor 원값, h = LN(target_encoder(32 장)) 캐시, 짝은 같은 문맥)
  정답 = s_imp > s_pos (matched) · 방향 A = (pos_a, imp_ab) · B = (pos_b, imp_ba)
  ε = mean s_pos(판) / mean s_pos(lam0)
  h 판독기 이식 = E3 의 h 판독기 (시드 0 · 1, attention 풀링, 위치 정보 없음) 를 LN(p) 에 건 결합 · 모양 정확도:
      S1 보임 조건 · 이동 · t8–10 보임 / S2 같은 조건 t11–15 보임 / S4 late · 이동 · 다시 보이는 걸음
  쌍 차이 (모양 · 색 · permanence block, 두 팔): Δp̃_t = LN(p_A,t) − LN(p_B,t), Δh_t = h_A,t − h_B,t, O = 참 물체 영역 (외곽 상자 + 1 칸)
      ρ_O = ‖Δp̃_t[O]‖ / ‖Δh_t[O]‖ · 따라감 f_O = 1 + ⟨s*_O, d참⟩ / |d참|² (s*_O = Δh_t[O] 꼴을 Δp̃ 에서 찾는 최선 칸 이동, |d참| ≥ 1 칸만) — E2 와 같은 정의
판정 (PLAN_R2 §3 H2-7): 분리 판 = (이식 ≥ 50 & A ≤ 30) 또는 (A ≥ 70 & 이식 ≤ 20). 없고 11 판 순위 상관 ≥ 0.8 → '같이 온다'. ε > 1.1 인 판은 뺀다.
입력: 캐시 h.npy · h_imp_fut.npy · tracks.npz · heads/<objset>_h_s{0,1}.pt · VOBench 프레임 (WMADataset)
출력: results/q1q2_r2/E25_rows_r{rank}.jsonl (영상 · 판별 행) → summarize → results/q1q2_r2/E25_stitch.json

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/transfer_gap_stitch.py run --rank 0 --world 2 --gpu 0 [--limit 4]
  $P z_research/scripts/analysis/transfer_gap_stitch.py summarize
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_gap_common as C  # noqa: E402

NT, NF, S, D, G = C.NT, C.NF, C.S, C.D, C.GRID
OUT = C.REPO / "z_analysis/transfer_gap/results/q1q2_r2"
REL = C.REPO / "z_training/runs/release_vith/latest.pt"
PFT = C.REPO / "z_training/runs/vobench_release_postft_g/latest.pt"
VARIANTS = ["lam0", "lam25", "lam50", "lam75", "lam100", "onlyP_G1", "onlyP_G2", "onlyP_G3", "onlyR_G1", "onlyR_G2", "onlyR_G3"]


def group_of(k):
    if k.startswith(("predictor_embed", "mask_tokens")):
        return 1
    if k.startswith("predictor_blocks."):
        b = int(k.split(".")[1])
        return 1 if b <= 3 else 2 if b <= 7 else 3
    if k.startswith(("predictor_norm", "predictor_proj")):
        return 3
    return 0   # 체크포인트 밖 버퍼 — 바꾸지 않는다


def variant_states(sd_r, sd_p):
    out = {}
    for v in VARIANTS:
        sd = {}
        for k in sd_r:
            a, b = sd_r[k], sd_p.get(k, sd_r[k])
            if group_of(k) == 0:
                sd[k] = a
                continue
            if v.startswith("lam"):
                lam = int(v[3:]) / 100.0
                sd[k] = ((1 - lam) * a.float() + lam * b.float()).to(a.dtype)
            elif v.startswith("onlyP"):
                sd[k] = b if group_of(k) == int(v[-1]) else a
            else:
                sd[k] = a if group_of(k) == int(v[-1]) else b
        out[v] = sd
    return out


def run(rank, world, gpu, limit, bs=4):
    import torch
    sys.path.insert(0, str(C.REPO))
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.data import WMADataset
    from evals.world_model_analysis.attn_probe import _ClipDS
    import transfer_gap_pairdiff as PD
    import yaml
    torch.set_num_threads(6)
    dev = torch.device(f"cuda:{gpu}"); torch.cuda.set_device(dev)
    LN, _shift_stats, region_shift = PD.torch_tools(dev)
    cfg = yaml.safe_load(open(C.REPO / "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/_resolved.yaml"))
    cfg["model"]["dtype"] = "bfloat16"; cfg["model"]["autocast"] = "none"; cfg["features"] = {"decode_workers": 6}
    ds = WMADataset(cfg); vid2rec = {r.video_id: i for i, r in enumerate(ds.records)}
    model = build_from_config(dict(cfg["model"], predictor_checkpoint=str(REL)), dev)
    pred = model.predictor
    msd = pred.state_dict()
    sd_r = {k: v.detach().clone() for k, v in msd.items()}
    raw_p = torch.load(PFT, map_location="cpu", weights_only=False)["predictor"]
    missing = [k for k in msd if k not in raw_p]
    assert all(group_of(k) == 0 for k in missing), f"PFT 에 없는 파라미터 {missing[:5]}"
    print(f"[r{rank}] 체크포인트 밖 키 {len(missing)} 개 (릴리즈 유지): {missing[:3]}", flush=True)
    sd_p = {k: raw_p[k].to(dev, msd[k].dtype) for k in msd if k in raw_p}
    raw_r = torch.load(REL, map_location="cpu", weights_only=False)["predictor"]
    d_rel = max(float((sd_r[k].float() - raw_r[k].to(dev).float()).abs().max()) for k in msd if k in raw_r)
    print(f"[r{rank}] release state vs file max|Δ| {d_rel:.2e} (bf16 cast)", flush=True)
    VS = variant_states(sd_r, sd_p)

    M = C.meta(); trb = C.train_blocks(); bmap, UV, HID, EXT = C.load_tracks()
    H = np.load(C.TUBE / "h.npy", mmap_mode="r"); HI = np.load(C.TG / "h_imp_fut.npy", mmap_mode="r")
    P0 = np.load(C.TUBE / "p.npy", mmap_mode="r"); PF0 = np.load(C.TG / "p_relftg.npy", mmap_mode="r")
    IM = json.load(open(C.TG / "imp_meta.json"))["meta"]; imp_of = {m["pos_tp"]: q for q, m in enumerate(IM)}
    blocks = defaultdict(dict)
    for j, m in enumerate(M):
        blocks[m["block_id"]][m["variant"]] = j
    bl = sorted([b for b in blocks if b not in trb], key=int)
    bl = bl[rank::world][: limit or None]
    heads = {}
    for objset in ("primitive", "realistic"):
        labs, L = C.classes(M, objset)
        for s in (0, 1):
            clf = C.make_head(len(labs), dev); clf.load_state_dict(torch.load(C.HEADS / f"{objset}_h_s{s}.pt", map_location=dev)); clf.eval()
            heads[(objset, s)] = (clf, labs, L)

    vids = []
    for b in bl:
        vids += [blocks[b]["pos_a"], blocks[b]["pos_b"]]
    loader = torch.utils.data.DataLoader(_ClipDS(cfg, [vid2rec[M[j]["video_id"]] for j in vids]), batch_size=bs, shuffle=False, num_workers=5, pin_memory=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fo = open(OUT / f"E25_rows_r{rank}{'_smoke' if limit else ''}.jsonl", "w")
    tens = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(dev)
    gate = defaultdict(list)
    w, t0 = 0, time.time()
    with torch.no_grad():
        for x in loader:
            b_ = x.size(0); jj = vids[w:w + b_]; w += b_
            x = x.to(dev, model.dtype, non_blocking=True)
            ci, ti = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=S, batch_size=b_, device=dev)
            z = model.context_encoder(x, masks=[ci]); z = z[-1] if isinstance(z, list) else z
            hpos = torch.stack([tens(H[j, NF:]) for j in jj]); himp = torch.stack([tens(HI[imp_of[j]]) for j in jj])
            hfull = {j: hpos[q] for q, j in enumerate(jj)}
            for v in VARIANTS:
                pred.load_state_dict(VS[v], strict=False)
                p = pred(z, ci, ti, mask_index=0).float().reshape(b_, NF, S, -1)
                if v == "lam0":
                    for q, j in enumerate(jj):
                        ref = tens(P0[j]); gate["lam0_vs_cache_p"].append(float((p[q] - ref).abs().mean() / ref.abs().mean()))
                if v == "lam100":
                    for q, j in enumerate(jj):
                        ref = tens(PF0[j]); gate["lam100_vs_cache_pft"].append(float((p[q] - ref).abs().mean() / ref.abs().mean()))
                s_pos = (p - hpos).abs().mean((1, 2, 3)); s_imp = (p - himp).abs().mean((1, 2, 3))
                pt = LN(p)
                # 판독 (h 판독기, 시드 0 · 1)
                reads = {}
                for q, j in enumerate(jj):
                    m = M[j]
                    if C.label_of(m) == "empty":
                        continue
                    st = C.tub_state(m["samples"]); moving = m["motion"] != "flat_static"; late = m["occlusion"] == "late"
                    ks = [k for k in range(NF) if st[NF + k] == "vis"]
                    if not ks:
                        continue
                    res = []
                    for s in (0, 1):
                        clf, labs, L = heads[(m["objset"], s)]
                        gt = L[C.label_of(m)]
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pr = clf(pt[q, ks]).float().argmax(1).cpu().numpy()
                        res.append([(int(a) == gt, labs[int(a)].split("/")[0] == labs[gt].split("/")[0]) for a in pr])
                    reads[j] = dict(ks=[NF + k for k in ks], acc=[(res[0][i][0] + res[1][i][0]) / 2 for i in range(len(ks))],
                                    shape=[(res[0][i][1] + res[1][i][1]) / 2 for i in range(len(ks))])
                # 쌍 차이 (block 의 두 팔이 같은 묶음에 있을 때)
                pdiff = {}
                for q in range(0, b_, 2):
                    ja, jb = jj[q], jj[q + 1]
                    m = M[ja]
                    if M[jb]["block_id"] != m["block_id"]:
                        continue
                    bi = bmap[m["block_id"]]; uvA, extA = UV[bi, 0], EXT[bi, 0]
                    hidA = HID[bi, 0]; hid = hidA | (HID[bi, 1] if m["violation"] != "permanence" else hidA)
                    tl = C.last_visible_ctx(hid)
                    dP = pt[q] - pt[q + 1]; dH = hfull[ja] - hfull[jb]
                    rr = []
                    for k in range(NF):
                        t = NF + k
                        O = C.region(uvA[t], extA[t])
                        rho = float(dP[k, O].norm() / dH[k, O].norm().clamp_min(1e-12))
                        cso = region_shift(dP[k], dH[k], set(O)); sO = PD.SH[int(cso.argmax())]
                        dtrue = np.array(C.cell_of(uvA[t])) - np.array(C.cell_of(uvA[tl]))
                        fO = float(1 + (sO[0] * dtrue[0] + sO[1] * dtrue[1]) / (dtrue ** 2).sum()) if np.hypot(*dtrue) >= 1 else None
                        rr.append(dict(t=t, rho_O=rho, followO=fO, kO=float(cso.max())))
                    pdiff[m["block_id"]] = rr
                for q, j in enumerate(jj):
                    m = M[j]
                    r = dict(v=v, j=j, block=m["block_id"], variant=m["variant"], objset=m["objset"], violation=m["violation"], occlusion=m["occlusion"],
                             k=m["k"], motion=m["motion"], s_pos=float(s_pos[q]), s_imp=float(s_imp[q]))
                    if j in reads:
                        r["read"] = reads[j]
                    if m["variant"] == "pos_a" and m["block_id"] in pdiff:
                        r["pdiff"] = pdiff[m["block_id"]]
                    fo.write(json.dumps(r) + "\n")
            pred.load_state_dict(VS["lam0"], strict=False)
            if rank == 0 or w % 80 < bs:
                print(f"[r{rank}] {w}/{len(vids)} {time.time() - t0:.0f}s · gate " + " ".join(f"{k} {np.mean(v):.2e}" for k, v in gate.items()), flush=True)
    fo.close()
    json.dump({k: dict(mean=float(np.mean(v)), max=float(np.max(v)), n=len(v)) for k, v in gate.items()},
              open(OUT / f"E25_gate_r{rank}{'_smoke' if limit else ''}.json", "w"), indent=1)
    print(f"[r{rank}] done {time.time() - t0:.0f}s", flush=True)


def summarize(smoke=False):
    import glob
    from scipy.stats import spearmanr
    suf = "_smoke" if smoke else ""
    rows = []
    for f in sorted(glob.glob(str(OUT / f"E25_rows_r[0-9]{suf}.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    gates = {}
    for f in sorted(glob.glob(str(OUT / f"E25_gate_r[0-9]{suf}.json"))):
        for k, v in json.load(open(f)).items():
            gates.setdefault(k, []).append(v)
    by = defaultdict(list)
    for r in rows:
        by[r["v"]].append(r)
    base_spos = np.mean([r["s_pos"] for r in by["lam0"]])
    out = {"definition": __doc__, "gates": gates, "variants": {}}

    def pair_acc(sel):
        v = [(float(r["s_imp"] > r["s_pos"]), r["block"]) for r in sel]
        return C.bootstrap_mean([100 * a for a, _ in v], [b for _, b in v]) if v else None

    for v in VARIANTS:
        R = by[v]
        if not R:
            continue
        d = {"n_videos": len(R), "eps": float(np.mean([r["s_pos"] for r in R]) / base_spos)}
        d["acc_all"] = pair_acc(R)
        for o in ("primitive", "realistic", "both"):
            ob = [r for r in R if o == "both" or r["objset"] == o]
            mv = [r for r in ob if r["motion"] != "flat_static"]
            for viol in ("permanence", "shape", "color"):
                for dirn, var in (("A", "pos_a"), ("B", "pos_b")):
                    sel = [r for r in mv if r["occlusion"] == "late" and r["violation"] == viol and r["variant"] == var]
                    d[f"{o}|late_move|{viol}|{dirn}"] = pair_acc(sel)
            d[f"{o}|late_move|shapecolor|A"] = pair_acc([r for r in mv if r["occlusion"] == "late" and r["violation"] in ("shape", "color") and r["variant"] == "pos_a"])
            d[f"{o}|vis_cond|all"] = pair_acc([r for r in ob if r["occlusion"] != "late"])
            # 판독
            for scen, f in (("S1", lambda r, t: r["occlusion"] != "late" and r["motion"] != "flat_static" and t <= 10),
                            ("S2", lambda r, t: r["occlusion"] != "late" and r["motion"] != "flat_static" and t >= 11),
                            ("S4", lambda r, t: r["occlusion"] == "late" and r["motion"] != "flat_static")):
                vals, shp, grp = [], [], []
                for r in ob:
                    if "read" not in r:
                        continue
                    for t, a, s in zip(r["read"]["ks"], r["read"]["acc"], r["read"]["shape"]):
                        if f(r, t):
                            vals.append(100 * a); shp.append(100 * s); grp.append(r["block"])
                if vals:
                    d[f"{o}|read_{scen}"] = C.bootstrap_mean(vals, grp); d[f"{o}|read_{scen}_shape"] = C.bootstrap_mean(shp, grp)
            # 쌍 차이
            for viol_g, vf in (("id", lambda r: r["violation"] in ("shape", "color")), ("perm", lambda r: r["violation"] == "permanence")):
                for scen, f in (("S2", lambda r, t: r["occlusion"] != "late" and t >= 11), ("S4", lambda r, t: r["occlusion"] == "late")):
                    rho, fol, grp, grpf = [], [], [], []
                    for r in ob:
                        if "pdiff" not in r or not vf(r) or r["motion"] == "flat_static":
                            continue
                        for e in r["pdiff"]:
                            if f(r, e["t"]):
                                rho.append(e["rho_O"]); grp.append(r["block"])
                                if e["followO"] is not None:
                                    fol.append(e["followO"]); grpf.append(r["block"])
                    if rho:
                        d[f"{o}|{viol_g}|{scen}|rho_O"] = C.bootstrap_mean(rho, grp)
                    if fol:
                        d[f"{o}|{viol_g}|{scen}|followO"] = C.bootstrap_mean(fol, grpf)
        out["variants"][v] = d
    # 판정
    vs = [v for v in VARIANTS if v in out["variants"] and out["variants"][v]["eps"] <= 1.1]
    get = lambda v, k: (out["variants"][v].get(k) or {}).get("mean", float("nan"))
    tab = {v: dict(A=get(v, "both|late_move|permanence|A"), B=get(v, "both|late_move|permanence|B"),
                   read_S4=get(v, "both|read_S4"), read_S2=get(v, "both|read_S2"), read_S4_shape=get(v, "both|read_S4_shape"),
                   shapecolor_A=get(v, "both|late_move|shapecolor|A"), acc_all=get(v, "acc_all"), eps=out["variants"][v]["eps"],
                   rho_S4=get(v, "both|id|S4|rho_O"), follow_S4=get(v, "both|id|S4|followO"),
                   rho_S2=get(v, "both|id|S2|rho_O"), follow_S2=get(v, "both|id|S2|followO")) for v in vs}
    sep = {v: t for v, t in tab.items() if (t["read_S4"] >= 50 and t["A"] <= 30) or (t["A"] >= 70 and t["read_S4"] <= 20)}
    sep2 = {v: t for v, t in tab.items() if (t["read_S2"] >= 50 and t["A"] <= 30) or (t["A"] >= 70 and t["read_S2"] <= 20)}
    rc = {}
    for k in ("read_S4", "read_S2", "rho_S4", "follow_S4", "shapecolor_A"):
        x = [tab[v][k] for v in vs]; y = [tab[v]["A"] for v in vs]
        rc[f"spearman_A_vs_{k}"] = float(spearmanr(x, y).correlation) if len(vs) > 2 else None
    out["table"] = tab
    out["verdict"] = dict(excluded_eps_gt_1p1=[v for v in VARIANTS if v in out["variants"] and v not in vs], separated_read_S4=list(sep), separated_read_S2=list(sep2),
                          rank_corr=rc,
                          rule="분리 판 = (이식 ≥ 50 & A ≤ 30) 또는 (A ≥ 70 & 이식 ≤ 20). 없고 순위 상관 ≥ 0.8 → 같이 온다")
    json.dump(out, open(OUT / f"E25_stitch{suf}.json", "w"), indent=1, ensure_ascii=False)
    for v, t in tab.items():
        print(v, {k: round(x, 3) if isinstance(x, float) else x for k, x in t.items()})
    print(out["verdict"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "summarize"])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bs", type=int, default=4, help="묶음 크기 — 캐시 (tubeprobe · p_relftg) 는 4 로 뽑았다. 짝수여야 쌍 차이가 같은 묶음에 든다")
    a = ap.parse_args()
    if a.phase == "run":
        run(a.rank, a.world, a.gpu, a.limit, a.bs)
    else:
        summarize(a.smoke)
