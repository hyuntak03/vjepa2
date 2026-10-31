#!/usr/bin/env python3
r"""[X1] R2 E21 — 메아리 대조: late p 의 정체 대비는 '가려진 꼴' 인가 '입력 (온라인) encoder 의 꼴' 인가 (H1-E1 · H1-E2 · H1-E0) (2026-10-30).

추출 (phase extract, GPU 1): 보임 조건 (visible · early · mid) · 이동 · 모양 · 색 block 전부 (두 objset, train + test) 의 두 팔 문맥
  z_vis_ctx[q] = LN(context_encoder(문맥 16 장))[튜블릿 t0..t7]  (560, 8, 256, 1280) fp16 — 릴리즈 ViT-H, probing 관례 bfloat16 · autocast none
  (transfer_gap_extract.py 의 z_late_ctx 와 같은 함수 · 같은 LN)
분석 (phase analyze, CPU): κ(X, Y) = max_{s∈[−9,9]×[−3,3]} cos(X, T_s Y) (전 칸) · κ_O(X, Y; O) = max_s cos(X[O+s], Y[O]) — E2c 와 같은 정의.
  (a) z 바닥: late Δp̃_t 대 같은 조건 (objset · 위반 · k · 운동) 다른 block b′ 의 Δz_7^{b′} (O = b′ 의 t7 자리). block 당 짝 최대 8 개 평균
  (b) encoder 종류: late Δp̃_t 대 Δz_7 (온라인, 가려진 t7) vs Δh_7 (EMA, 같은 t7)
  (c) 보임 조건 S2 (t11–15 보임): Δp̃_t 대 {Δz_끝 (온라인, 마지막 문맥 튜블릿 t7, 보임) · Δh_7 (EMA, 같은 t7) · Δh_t (참 미래)} — 바닥 (a) 판도
  (d) 영역 맞춤판: 모든 비교 대상에 같은 영역 O = Δp̃_t 에너지 최대 3×3 · 같은 탐색 폭 → κ_Op(X, Y) = max_s cos(X[O], Y[O−s])
  (e) E17 late 판 D_X 는 E17_fisher.json 에서 읽어 같이 적는다.
판정 (PLAN_R2 §5 E21): H1-E2 = late κ(z_7) − κ(h_7) ≥ 0.02 (CI 하한 > 0) & 보임 S2 에서도 κ(z_끝) − κ(h_7) ≥ 0.02 (CI > 0) & (d) 같은 순서.
  H1-E1 = late 에서만 κ(z_7) ≈ κ(h_7) > κ(h_t). H1-E0 = 바닥을 빼면 차이가 CI 안.
출력: /data2/.../transfer_gap/z_vis_ctx.npy + z_vis_meta.json · results/q1q2_r2/E21_echo.json

  $P z_research/scripts/analysis/transfer_gap_echo2.py extract --gpu 0 [--limit 8]
  $P z_research/scripts/analysis/transfer_gap_echo2.py analyze
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


def vis_list():
    M = C.meta()
    blocks = defaultdict(dict)
    for j, m in enumerate(M):
        blocks[m["block_id"]][m["variant"]] = j
    bl = sorted([b for b in blocks if M[blocks[b]["pos_a"]]["occlusion"] != "late" and M[blocks[b]["pos_a"]]["motion"] != "flat_static"
                 and M[blocks[b]["pos_a"]]["violation"] in ("shape", "color")], key=int)
    tp = []
    for b in bl:
        tp += [blocks[b]["pos_a"], blocks[b]["pos_b"]]
    return M, tp


def extract(gpu, limit):
    import torch, torch.nn as nn
    sys.path.insert(0, str(C.REPO))
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.data import WMADataset
    from evals.world_model_analysis.attn_probe import _ClipDS
    import yaml
    torch.set_num_threads(6)
    dev = torch.device(f"cuda:{gpu}"); torch.cuda.set_device(dev)
    cfg = yaml.safe_load(open(C.REPO / "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/_resolved.yaml"))
    cfg["model"]["dtype"] = "bfloat16"; cfg["model"]["autocast"] = "none"; cfg["features"] = {"decode_workers": 6}
    ds = WMADataset(cfg); vid2rec = {r.video_id: i for i, r in enumerate(ds.records)}
    M, tp = vis_list()
    tp = tp[: limit or None]
    name = "z_vis_ctx" + ("_smoke" if limit else "")
    Z = np.lib.format.open_memmap(C.TG / f"{name}.npy", "w+", np.float16, (len(tp), NF, S, D))
    json.dump(dict(n=len(tp), note="z_vis_ctx[q] = LN(context_encoder(문맥 16))[t0..7] of tubeprobe 번호 tp[q] (보임 조건 · 이동 · 모양 · 색 두 팔)",
                   tp=tp, video_id=[M[j]["video_id"] for j in tp]), open(C.TG / f"{name.replace('ctx', 'meta')}.json", "w"))
    model = build_from_config(cfg["model"], dev)
    bs = 8
    loader = torch.utils.data.DataLoader(_ClipDS(cfg, [vid2rec[M[j]["video_id"]] for j in tp]), batch_size=bs, shuffle=False, num_workers=5, pin_memory=True)
    w, t0 = 0, time.time()
    with torch.inference_mode():
        for x in loader:
            b = x.size(0)
            x = x.to(dev, model.dtype, non_blocking=True)
            ci, _ = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=S, batch_size=b, device=dev)
            z = model.context_encoder(x, masks=[ci]); z = z[-1] if isinstance(z, list) else z
            zn = nn.functional.layer_norm(z.float(), (z.size(-1),)).cpu().numpy().reshape(b, NF, S, -1).astype(np.float16)
            Z[w:w + b] = zn; w += b
            if w % 80 < bs:
                print(f"{w}/{len(tp)} {time.time() - t0:.0f}s", flush=True)
    Z.flush()
    # 관문: late 캐시와 같은 함수인지 — 같은 영상이 없으므로, 첫 영상의 z 를 다시 한 번 뽑아 결정론 확인
    print(f"done {len(tp)} in {time.time() - t0:.0f}s → {C.TG / name}.npy", flush=True)


def analyze(gpu=-1):
    import torch
    import transfer_gap_pairdiff as PD
    dev = torch.device(f"cuda:{gpu}") if gpu >= 0 else torch.device("cpu")
    torch.set_num_threads(12)
    LN, shift_stats, region_shift = PD.torch_tools(dev)
    SH = PD.SH
    M = C.meta(); bmap, UV, HID, EXT = C.load_tracks(); trb = C.train_blocks()
    H = np.load(C.TUBE / "h.npy", mmap_mode="r"); P = np.load(C.TUBE / "p.npy", mmap_mode="r"); PF = np.load(C.TG / "p_relftg.npy", mmap_mode="r")
    ZL = np.load(C.TG / "z_late_ctx.npy", mmap_mode="r"); zl = {j: q for q, j in enumerate(json.load(open(C.TG / "z_late_meta.json"))["tp"])}
    ZV = np.load(C.TG / "z_vis_ctx.npy", mmap_mode="r"); zv = {j: q for q, j in enumerate(json.load(open(C.TG / "z_vis_meta.json"))["tp"])}
    blocks = defaultdict(dict)
    for j, m in enumerate(M):
        blocks[m["block_id"]][m["variant"]] = j
    load = lambda arr, j: torch.from_numpy(np.asarray(arr[j], np.float32)).to(dev)
    SHT = torch.tensor(SH, device=dev)

    def kOp(X, Y, O):
        """영역 맞춤판: X 의 영역 O (Δp 에너지 최대 3×3) 를 고정하고 Y 에서 같은 꼴을 찾는다 — max_s cos(X[O], Y[O − s])."""
        O = torch.tensor(sorted(O), device=dev); r, c = O // C.GRID, O % C.GRID
        rr = r[None] - SHT[:, :1]; cc = c[None] - SHT[:, 1:]
        valid = ((rr >= 0) & (rr < C.GRID) & (cc >= 0) & (cc < C.GRID)).float()
        Ys = Y[(rr.clamp(0, C.GRID - 1) * C.GRID + cc.clamp(0, C.GRID - 1))] * valid[..., None]
        Xo = X[O]
        ip = torch.einsum("sod,od->s", Ys, Xo)
        return float((ip / (Ys.flatten(1).norm(dim=1) * Xo.norm()).clamp_min(1e-12)).max())

    def energy_box(X):
        e = (X ** 2).sum(-1); r, c = divmod(int(e.argmax()), C.GRID)
        return set(C.region(((c + 0.5) * C.CELL, (r + 0.5) * C.CELL)))

    rng = np.random.default_rng(0)
    rows = []
    t0 = time.time()
    # ---- late (a) (b) (d)
    late_bl = [b for b in blocks if M[blocks[b]["pos_a"]]["occlusion"] == "late" and M[blocks[b]["pos_a"]]["motion"] != "flat_static"
               and M[blocks[b]["pos_a"]]["violation"] in ("shape", "color")]
    vis_bl = [b for b in blocks if blocks[b]["pos_a"] in zv]
    cache_dz = {}

    def dz_late(b):
        if b not in cache_dz:
            cache_dz[b] = load(ZL, zl[blocks[b]["pos_a"]]) - load(ZL, zl[blocks[b]["pos_b"]])
        return cache_dz[b]

    grp = defaultdict(list)
    for b in late_bl:
        m = M[blocks[b]["pos_a"]]; grp[(m["objset"], m["violation"])].append(b)
    for n_, b in enumerate(sorted(late_bl, key=int)):
        ja, jb = blocks[b]["pos_a"], blocks[b]["pos_b"]; m = M[ja]
        bi = bmap[b]; uv, ext = UV[bi, 0], EXT[bi, 0]
        hid = HID[bi, 0] | HID[bi, 1]; tl = C.last_visible_ctx(hid); st = C.tub_state(m["samples"])
        dH = load(H, ja) - load(H, jb); dZ = dz_late(b)
        O7 = set(C.region(uv[7], ext[7])); Otl = set(C.region(uv[tl], ext[tl]))
        partners = [c for c in grp[(m["objset"], m["violation"])] if c != b]
        partners = list(rng.choice(partners, size=min(8, len(partners)), replace=False))
        for mdl, arr in (("rel", P), ("pft", PF)):
            dP = LN(load(arr, ja)) - LN(load(arr, jb))
            for k in range(NF):
                t = NF + k
                if st[t] != "vis":
                    continue
                X = dP[k]; Ot = set(C.region(uv[t], ext[t])); Op = energy_box(X)
                r = dict(block=b, objset=m["objset"], violation=m["violation"], kk=m["k"], motion=m["motion"], t=t, model=mdl, scen="late", test=b not in trb)
                for nm, Y, O in (("h_t", dH[t], Ot), ("z_7", dZ[7], O7), ("h_7", dH[7], O7), ("z_end", dZ[tl], Otl)):
                    r[f"k_{nm}"] = float(shift_stats(X[None], Y[None]).max()); r[f"kO_{nm}"] = float(region_shift(X, Y, O).max()); r[f"kOp_{nm}"] = kOp(X, Y, Op)
                fl = defaultdict(list)
                for c in partners:
                    dZc = dz_late(c); bc = bmap[c]; O7c = set(C.region(UV[bc, 0][7], EXT[bc, 0][7]))
                    fl["k"].append(float(shift_stats(X[None], dZc[7][None]).max())); fl["kO"].append(float(region_shift(X, dZc[7], O7c).max())); fl["kOp"].append(kOp(X, dZc[7], Op))
                for kk_, v in fl.items():
                    r[f"{kk_}_floor_z7"] = float(np.mean(v))
                rows.append(r)
        if n_ % 20 == 0:
            print(f"late {n_}/{len(late_bl)} {time.time() - t0:.0f}s", flush=True)
    # ---- 보임 조건 S2 (c) (d)
    cache_v = {}

    def dz_vis(b):
        if b not in cache_v:
            cache_v[b] = load(ZV, zv[blocks[b]["pos_a"]]) - load(ZV, zv[blocks[b]["pos_b"]])
        return cache_v[b]
    grpv = defaultdict(list)
    for b in vis_bl:
        m = M[blocks[b]["pos_a"]]; grpv[(m["objset"], m["violation"])].append(b)
    for n_, b in enumerate(sorted(vis_bl, key=int)):
        ja, jb = blocks[b]["pos_a"], blocks[b]["pos_b"]; m = M[ja]
        bi = bmap[b]; uv, ext = UV[bi, 0], EXT[bi, 0]
        hid = HID[bi, 0] | HID[bi, 1]; tl = C.last_visible_ctx(hid); st = C.tub_state(m["samples"])
        if st[7] != "vis":
            continue
        dH = load(H, ja) - load(H, jb); dZ = dz_vis(b)
        O7 = set(C.region(uv[7], ext[7]))
        partners = [c for c in grpv[(m["objset"], m["violation"])] if c != b]
        partners = list(rng.choice(partners, size=min(8, len(partners)), replace=False))
        for mdl, arr in (("rel", P), ("pft", PF)):
            dP = LN(load(arr, ja)) - LN(load(arr, jb))
            for k in range(NF):
                t = NF + k
                if t < 11 or st[t] != "vis":
                    continue
                X = dP[k]; Ot = set(C.region(uv[t], ext[t])); Op = energy_box(X)
                r = dict(block=b, objset=m["objset"], violation=m["violation"], kk=m["k"], motion=m["motion"], t=t, model=mdl, scen="vis_S2", test=b not in trb)
                for nm, Y, O in (("h_t", dH[t], Ot), ("z_end", dZ[7], O7), ("h_7", dH[7], O7)):
                    r[f"k_{nm}"] = float(shift_stats(X[None], Y[None]).max()); r[f"kO_{nm}"] = float(region_shift(X, Y, O).max()); r[f"kOp_{nm}"] = kOp(X, Y, Op)
                fl = defaultdict(list)
                for c in partners:
                    dZc = dz_vis(c); bc = bmap[c]; O7c = set(C.region(UV[bc, 0][7], EXT[bc, 0][7]))
                    fl["k"].append(float(shift_stats(X[None], dZc[7][None]).max())); fl["kO"].append(float(region_shift(X, dZc[7], O7c).max())); fl["kOp"].append(kOp(X, dZc[7], Op))
                for kk_, v in fl.items():
                    r[f"{kk_}_floor_z7"] = float(np.mean(v))
                rows.append(r)
        if n_ % 40 == 0:
            print(f"vis {n_}/{len(vis_bl)} {time.time() - t0:.0f}s", flush=True)
    # ---- 요약
    out = {"definition": __doc__, "summary": {}}
    for scen in ("late", "vis_S2"):
        for mdl in ("rel", "pft"):
            for o in ("primitive", "realistic", "both"):
                sel = [r for r in rows if r["scen"] == scen and r["model"] == mdl and (o == "both" or r["objset"] == o)]
                if not sel:
                    continue
                g = [r["block"] for r in sel]
                d = dict(n=len(sel), n_block=len(set(g)))
                for pre in ("k", "kO", "kOp"):
                    for nm in ("h_t", "z_7", "h_7", "z_end"):
                        if f"{pre}_{nm}" in sel[0]:
                            d[f"{pre}_{nm}"] = C.bootstrap_mean([r[f"{pre}_{nm}"] for r in sel], g)
                    if f"{pre}_floor_z7" in sel[0]:
                        d[f"{pre}_floor_z7"] = C.bootstrap_mean([r[f"{pre}_floor_z7"] for r in sel], g)
                    zname = "z_7" if scen == "late" else "z_end"
                    d[f"{pre}_diff_z_minus_h7"] = C.bootstrap_mean([r[f"{pre}_{zname}"] - r[f"{pre}_h_7"] for r in sel], g)
                    d[f"{pre}_diff_z_minus_floor"] = C.bootstrap_mean([r[f"{pre}_{zname}"] - r[f"{pre}_floor_z7"] for r in sel], g)
                    d[f"{pre}_diff_h7_minus_ht"] = C.bootstrap_mean([r[f"{pre}_h_7"] - r[f"{pre}_h_t"] for r in sel], g)
                    d[f"{pre}_diff_z_minus_ht"] = C.bootstrap_mean([r[f"{pre}_{zname}"] - r[f"{pre}_h_t"] for r in sel], g)
                out["summary"][f"{scen}|{mdl}|{o}"] = d
    # 판정 (릴리즈, both)
    ver = {}
    for pre in ("kO", "k", "kOp"):
        L_ = out["summary"].get(f"late|rel|both", {}); V_ = out["summary"].get(f"vis_S2|rel|both", {})
        if not L_ or not V_:
            continue
        dl, dv = L_[f"{pre}_diff_z_minus_h7"], V_[f"{pre}_diff_z_minus_h7"]
        e2 = dl["mean"] >= 0.02 and dl["lo"] > 0 and dv["mean"] >= 0.02 and dv["lo"] > 0
        e1 = (not (dv["mean"] >= 0.02 and dv["lo"] > 0)) and abs(dl["mean"]) < 0.02 and L_[f"{pre}_diff_h7_minus_ht"]["lo"] > 0
        fl = L_[f"{pre}_diff_z_minus_floor"]
        e0 = fl["lo"] <= 0 <= fl["hi"]
        ver[pre] = dict(late_z7_minus_h7=dl, vis_zend_minus_h7=dv, late_z7_minus_floor=fl, late_h7_minus_ht=L_[f"{pre}_diff_h7_minus_ht"],
                        H1_E2=bool(e2), H1_E1=bool(e1), H1_E0=bool(e0))
    try:
        ef = json.load(open(OUT / "E17_fisher.json"))
        ver["E17_late_D"] = {k: {s: v["late_sources"][s]["D"] for s in ("h_t", "z_7", "h_7", "z_end")} for k, v in ef["tasks"].items() if "late_sources" in v}
    except Exception as ex:  # noqa: BLE001
        ver["E17_late_D"] = f"E17 없음 ({ex})"
    out["verdict"] = ver
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "E21_echo.json", "w"), indent=1, ensure_ascii=False, default=float)
    with open(OUT / "E21_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(ver, ensure_ascii=False, default=float)[:3000], f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract", "analyze"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.phase == "extract":
        extract(a.gpu, a.limit)
    else:
        analyze(a.gpu)
