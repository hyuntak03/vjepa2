#!/usr/bin/env python3
"""transfer_gap R4 E38c — 릴리즈 attention 길을 더 세게 키우면 late surprise 가 오르나 (2026-10-30, 리더 [L], 사용자 요청 "세기 키워봐").

E38 (transfer_gap_r4_ko.py) 의 '키우기' 를 β 4 → 8 · 12 · 16 · 24 · 강제 (그 query 는 물체 key 만 보게) 로 넓히고,
E38 에서 빠진 모양 · 색 위반의 surprise 정확도까지 같은 판에서 잰다. E38 의 정의 · 층 띠 (L3–L11) · 자리 규칙을 그대로 쓴다
(같은 모듈의 geom · key_sets · q_true 를 가져온다).

판: VOBench late · 이동 (정지 제외) block 전부 (split v2 train + test; 릴리즈는 VOBench 로 학습하지 않았으므로 둘 다 held-out),
    위반 셋 (사라짐 · 모양 · 색) × 물체 세트 둘. 프레임 = VOBench 256² 사본 (학습 로더와 비트 동일, 채점 하네스와는 반올림 ≤ 0.5/255).
    문맥 encoder = online, 목표 = LN(target_encoder(32 장)), float32 + autocast float16.
조건 (릴리즈 predictor, 층 3–11 의 모든 head):
    clean          아무것도 안 바꿈
    obj_b{β}       다시 나타난 미래 튜블릿의 참 물체 3×3 query → 보이는 문맥 튜블릿의 물체 3×3 key 의 logit 에 +β
    obj_hard       같은 query 가 물체 key 만 보게 (그 밖의 key 에 −1e4)
    ctl_b{β} · ctl_hard   key 를 같은 튜블릿에서 세로 5 칸 옮긴 상자로 (대조)
    + VOBench PFT clean (참고)
    마스크는 block 의 물체 팔 (pos_a) 라벨로 정하고 matched 쌍의 두 문맥 (A · B) 에 같은 것을 쓴다 (E38 과 같음; 한 block 의 A · B 는 경로가 같다).
지표:
    정확도  A = s(imp_ab) > s(pos_a) (p 는 pos_a 문맥), B = s(imp_ba) > s(pos_b) (p 는 pos_b 문맥), s = mean |p − LN(h)[미래]|
    사라짐 block 만: 존재 대비 D = ‖p_A − p_B‖ (토큰별) 의 3×3 상자 최대 자리가 참 자리 1 칸 안인 비율 · 참 상자 / 상자 평균 (E38 과 같은 계산)
    CI = block bootstrap 2,000.
산출: z_analysis/transfer_gap/results/r4_L/E38c/sw_r<rank>.npz → E38c_steer_sweep.json

재현:
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  sbatch -p batch_vll -w vll2 --gres=gpu:2 -c 16 --mem=90G --time=0-03:00:00 \
    --wrap "$P z_research/scripts/analysis/transfer_gap_r4_steer_sweep.py run --gpus 2"
  $P z_research/scripts/analysis/transfer_gap_r4_steer_sweep.py summarize
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import transfer_gap_r4_ko as KO  # noqa: E402

S, NB, C, K, G = KO.S, KO.NB, KO.C, KO.K, KO.G
OUTR = REPO / "z_analysis/transfer_gap/results/r4_L"
OUT = OUTR / "E38c"
FRAMES = REPO / "z_pixel_decoder/runs/_frames/VOBench_v1_256full"
LAYERS = list(range(3, 12))
BETAS = (4.0, 8.0, 12.0, 16.0, 24.0)
CTL_BETAS = (8.0, 16.0)


def build_blocks():
    sp = {}
    for s in ("train", "test"):
        for r in csv.DictReader(open(REPO / f"data_csv/vobench_v1_split_v2/index_{s}.csv")):
            sp[r["video_id"]] = s
    by = defaultdict(dict)
    for r in csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv")):
        if r["occlusion"] != "late" or r["motion"] == "flat_static":
            continue
        by[(r["objset"], r["block_id"])][r["variant"]] = r
    out = []
    for key, v in sorted(by.items()):
        if not all(x in v for x in ("pos_a", "pos_b", "imp_ab", "imp_ba")):
            continue
        r0 = v["pos_a"]
        out.append(dict(block="|".join(key), split=sp.get(r0["video_id"], "?"), viol=r0["violation"],
                        objset=r0["objset"], k=r0["k"], motion=r0["motion"], rows=v))
    return out


def _worker(rank, world, bs, limit):
    import torch
    import copy as _copy
    from analysis.intphys2.model import build_from_config, _clean_backbone_key, _load_state_dict
    from analysis.intphys2.surprise import _context_target_indices
    from analysis.attention.knockout import runner as RN
    from evals.world_model_analysis.data import WMADataset
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    model = dict(checkpoint=str(REPO / "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth"),
                 arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, window_size=32, use_rope=True, uniform_power=False,
                 dual_encoder=True, context_encoder_key="encoder", target_encoder_key="target_encoder",
                 predictor=dict(embed_dim=384, depth=12, num_heads=12, num_mask_tokens=10), dtype="float32")
    bundle = build_from_config(model, dev)
    rel = bundle.predictor
    pft = _copy.deepcopy(rel); st = torch.load(REPO / KO.CKPT["vobftg"], map_location="cpu", weights_only=False)
    assert not (set(pft.state_dict()) - set(_clean_backbone_key(st["predictor"])))
    _load_state_dict(pft, st["predictor"], tag="vobftg"); pft = pft.to(dev).eval()

    TR = KO.tracks()
    blocks = build_blocks()
    if limit:
        blocks = blocks[:limit]
    mine = blocks[rank::world]
    idir = OUT / f"index_r{rank}"; idir.mkdir(parents=True, exist_ok=True)
    rows = [r for b in mine for r in b["rows"].values()]
    with open(idir / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    ds = WMADataset(dict(data=dict(root=str(idir), index_csv="index.csv", frames_root=str(FRAMES), frames_pattern="{file_name}/{frame:06d}.png",
                                   frames_start=0, frames_stride=3, n_frames=32, resolution=256, block_column="block_id", pair_column="pair_id",
                                   variant_column="variant", plausible_column="plausible", type_column="condition"),
                         model=model, features={"cache_dir": "/tmp"}, surprise={}))
    di = {r.video_id: i for i, r in enumerate(ds.records)}
    N = NB * S
    L = len(rel.predictor_blocks); l0 = min(LAYERS)
    ac = torch.autocast("cuda", dtype=torch.float16)

    def enc(vids):
        x = torch.stack([ds.clip(di[v]) for v in vids]).to(dev)
        B = x.size(0)
        ci, ti = _context_target_indices(ctx_frames=2 * C, tgt_frames=2 * K, tubelet_size=2, spatial_tokens=S, batch_size=B, device=dev)
        with ac:
            z = bundle.context_encoder(x, masks=[ci]); z = z[-1] if isinstance(z, list) else z
            h = bundle.target_encoder(x); h = h[-1] if isinstance(h, list) else h
        h = torch.nn.functional.layer_norm(h.float(), (h.size(-1),))[:, C * S:]
        return z, h, ci, ti

    def steer(Qs, Ks, B, beta, hard=False):
        am = torch.zeros(B, 1, N, N, dtype=torch.float16, device=dev)
        for j, (qs, ks) in enumerate(zip(Qs, Ks)):
            if len(qs) and len(ks):
                qq = torch.as_tensor(qs, device=dev); kk = torch.as_tensor(ks, device=dev)
                if hard:
                    am[j, 0, qq, :] = -1e4
                    am[j, 0, qq[:, None], kk[None, :]] = 0.0
                else:
                    am[j, 0, qq[:, None], kk[None, :]] = beta
        return am

    def run(pm, z, ci, ti, conds):
        out = {}
        with ac:
            p0, prefix, masks, n_ctx = RN.clean_forward(pm, z, ci, ti, mask_index=0, keep_prefix=True)
            out["clean"] = p0.float()
            for nm, am in conds.items():
                ams = [am if l in LAYERS else None for l in range(L)]
                out[nm] = RN.resume(pm, prefix[l0], l0, masks, n_ctx, ams).float()
        return out

    acc = defaultdict(list); t0 = time.time(); verify = None
    l1 = lambda p, h: (p - h).abs().mean((1, 2)).cpu().numpy()  # noqa: E731
    for gi in range(0, len(mine), bs):
        gg = mine[gi:gi + bs]; B = len(gg)
        va = [g["rows"]["pos_a"]["video_id"] for g in gg]; vai = [g["rows"]["imp_ab"]["video_id"] for g in gg]
        vb_ = [g["rows"]["pos_b"]["video_id"] for g in gg]; vbi = [g["rows"]["imp_ba"]["video_id"] for g in gg]
        with torch.no_grad():
            zA, hA, ci, ti = enc(va); _, hAi, _, _ = enc(vai); zB, hB, _, _ = enc(vb_); _, hBi, _, _ = enc(vbi)
            Gm = [KO.geom(*TR[v][:2]) for v in va]
            KS = [KO.key_sets(*g) for g in Gm]; QS = [KO.q_true(*g) for g in Gm]
            Ko = [k[0] for k in KS]; Kc = [k[1] for k in KS]
            conds = {}
            for bt in BETAS:
                conds[f"obj_b{bt:g}"] = steer(QS, Ko, B, bt)
            conds["obj_hard"] = steer(QS, Ko, B, 0.0, hard=True)
            for bt in CTL_BETAS:
                conds[f"ctl_b{bt:g}"] = steer(QS, Kc, B, bt)
            conds["ctl_hard"] = steer(QS, Kc, B, 0.0, hard=True)
            pa = run(rel, zA, ci, ti, conds); pb = run(rel, zB, ci, ti, conds)
            qa = run(pft, zA, ci, ti, {}); qb = run(pft, zB, ci, ti, {})
            pa["pft_clean"] = qa["clean"]; pb["pft_clean"] = qb["clean"]
            if verify is None:
                with ac:
                    p0, prefix, masks, n_ctx = RN.clean_forward(rel, zA, ci, ti, mask_index=0, keep_prefix=True)
                    r0 = RN.resume(rel, prefix[0], 0, masks, n_ctx, None).float()
                verify = float((r0 - p0.float()).abs().max())
            Hd = (hA - hB).view(B, K, G, G, -1)
            for cn in pa:
                Pd = (pa[cn] - pb[cn]).view(B, K, G, G, -1)
                cs = np.full((B, K), np.nan, np.float32); rt = np.full((B, K), np.nan, np.float32)
                for j in range(B):
                    yx_, vb_m = Gm[j]
                    for t in range(K):
                        if vb_m[C + t]:
                            y, x = yx_[C + t]
                            a = Pd[j, t, y - 1:y + 2, x - 1:x + 2].mean((0, 1)); b = Hd[j, t, y - 1:y + 2, x - 1:x + 2].mean((0, 1))
                            cs[j, t] = float(torch.nn.functional.cosine_similarity(a, b, dim=0)); rt[j, t] = float(a.norm() / b.norm())
                acc[f"{cn}__boxcos"].append(cs); acc[f"{cn}__boxratio"].append(rt)
                D = (pa[cn] - pb[cn]).norm(dim=-1).view(B, K, G, G)
                Db = torch.nn.functional.avg_pool2d(D.reshape(B * K, 1, G, G), 3, 1, 1).view(B, K, G, G)[:, :, 1:-1, 1:-1]
                acc[f"{cn}__sA_pos"].append(l1(pa[cn], hA)); acc[f"{cn}__sA_imp"].append(l1(pa[cn], hAi))
                acc[f"{cn}__sB_pos"].append(l1(pb[cn], hB)); acc[f"{cn}__sB_imp"].append(l1(pb[cn], hBi))
                acc[f"{cn}__Db"].append(Db.cpu().numpy().astype(np.float32))
        for kk in ("block", "split", "viol", "objset", "k", "motion"):
            acc[f"meta__{kk}"].append(np.array([g[kk] for g in gg]))
        acc["meta__yx"].append(np.stack([g[0] for g in Gm])); acc["meta__vb"].append(np.stack([g[1] for g in Gm]))
        if rank == 0:
            print(f"[r0] {gi + B}/{len(mine)} ({time.time() - t0:.0f}s) verify {verify}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"sw_r{rank}.npz", verify=verify, **{k: np.concatenate(v) for k, v in acc.items()})
    print(f"[r{rank}] done {time.time() - t0:.0f}s verify {verify}", flush=True)


def summarize():
    Z = defaultdict(list); ver = []
    for f in sorted(OUT.glob("sw_r*.npz")):
        z = np.load(f, allow_pickle=True)
        ver.append(float(z["verify"]))
        for k in z.files:
            if k != "verify":
                Z[k].append(z[k])
    Z = {k: np.concatenate(v) for k, v in Z.items()}
    blocks, viol, split = Z["meta__block"], Z["meta__viol"], Z["meta__split"]
    yx, vb = Z["meta__yx"], Z["meta__vb"]
    conds = sorted({k.split("__")[0] for k in Z if k.endswith("__sA_pos")})
    res = {}
    for c in conds:
        A = (Z[f"{c}__sA_imp"] > Z[f"{c}__sA_pos"]).astype(float)
        Bv = (Z[f"{c}__sB_imp"] > Z[f"{c}__sB_pos"]).astype(float)
        r = {}
        for sub, m in [("all", np.ones(len(blocks), bool)), ("test", split == "test")]:
            for v in ("permanence", "shape", "color", "any"):
                mm = m & ((viol == v) if v != "any" else True)
                if mm.sum() == 0:
                    continue
                both = np.concatenate([A[mm], Bv[mm]]); bb = np.concatenate([blocks[mm], blocks[mm]])
                r[f"{sub}|{v}"] = dict(n_block=int(mm.sum()), A=KO.boot_mean(A[mm], blocks[mm]), B=KO.boot_mean(Bv[mm], blocks[mm]),
                                       overall=KO.boot_mean(both, bb))
            mm = m & (viol == "permanence")
            Db = Z[f"{c}__Db"]; hit, con, bl = [], [], []
            for i in np.where(mm)[0]:
                for t in range(K):
                    if not vb[i, C + t]:
                        continue
                    y, x = yx[i, C + t]; mp = Db[i, t]
                    a = np.unravel_index(np.argmax(mp), mp.shape); a = (a[0] + 1, a[1] + 1)
                    hit.append(float(max(abs(a[0] - y), abs(a[1] - x)) <= 1)); con.append(float(mp[y - 1, x - 1] / mp.mean())); bl.append(blocks[i])
            if hit:
                r[f"{sub}|permanence_map"] = dict(within1=KO.boot_mean(np.array(hit), np.array(bl)), contrast_mean=float(np.mean(con)), n_step=len(hit))
            for v in ("permanence", "shape", "color"):
                mm = m & (viol == v)
                cs = Z[f"{c}__boxcos"][mm]; rt = Z[f"{c}__boxratio"][mm]
                r[f"{sub}|{v}_box"] = dict(cos=float(np.nanmean(cs)), ratio=float(np.nanmean(rt)))
        res[c] = r
    out = dict(definition=__doc__.split("판:")[1].split("산출")[0].strip(), layers=LAYERS, verify_max_abs=ver, results=res)
    OUTR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUTR / "E38c_steer_sweep.json", "w"), indent=1, ensure_ascii=False)
    order = ["clean"] + [f"obj_b{b:g}" for b in BETAS] + ["obj_hard"] + [f"ctl_b{b:g}" for b in CTL_BETAS] + ["ctl_hard", "pft_clean"]
    for sub in ("all", "test"):
        print(f"\n== {sub} (verify {ver})")
        print(f"{'cond':12s} {'perm A':>7s} {'perm B':>7s} {'shape':>7s} {'color':>7s} {'late 전체':>9s} {'1칸안':>7s} {'배율':>6s} "
              f"{'존재cos':>7s} {'존재비':>6s} {'정체cos':>7s} {'정체비':>6s}")
        for c in order:
            if c not in res:
                continue
            r = res[c]; g = lambda k, f: r[k][f][0] if k in r else float("nan")  # noqa: E731
            pm = r.get(f"{sub}|permanence_map", {}); pb_ = r.get(f"{sub}|permanence_box", {}); sb = r.get(f"{sub}|shape_box", {})
            print(f"{c:12s} {g(sub + '|permanence', 'A'):7.1f} {g(sub + '|permanence', 'B'):7.1f} {g(sub + '|shape', 'overall'):7.1f} "
                  f"{g(sub + '|color', 'overall'):7.1f} {g(sub + '|any', 'overall'):9.1f} "
                  f"{pm.get('within1', [float('nan')])[0]:7.1f} {pm.get('contrast_mean', float('nan')):6.2f} "
                  f"{pb_.get('cos', float('nan')):7.2f} {pb_.get('ratio', float('nan')):6.2f} {sb.get('cos', float('nan')):7.2f} {sb.get('ratio', float('nan')):6.2f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--gpus", type=int, default=1); ap.add_argument("--bs", type=int, default=2); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "summarize":
        return summarize()
    import torch.multiprocessing as mp
    if a.gpus == 1:
        _worker(0, 1, a.bs, a.limit)
    else:
        ps = [mp.get_context("spawn").Process(target=_worker, args=(r, a.gpus, a.bs, a.limit)) for r in range(a.gpus)]
        [p.start() for p in ps]; [p.join() for p in ps]
        bad = [p.exitcode for p in ps if p.exitcode != 0]
        if bad:
            raise SystemExit(f"worker 실패 {bad}")


if __name__ == "__main__":
    main()
