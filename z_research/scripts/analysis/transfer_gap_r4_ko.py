#!/usr/bin/env python3
"""transfer_gap R4 E38 — 미래 → 과거 물체 자리 attention 을 막으면 (PFT) · 키우면 (릴리즈) late 영속이 바뀌나 (2026-10-30, 실험 A [X1]).

E37 은 attention 질량만 쟀다 (질량 ≠ 사용). 여기서는 그 간선을 실제로 끊거나 키운 뒤 출력이 바뀌는지 본다.
엔진: analysis/attention/knockout/runner.py (clean_forward · resume: 층별 residual 을 받아 두고 l0 부터 attn_mask 로 다시 돈다;
       resume(…, None) 이 실제 forward 와 같은지 첫 배치에서 검사해 결과에 남긴다).

정의 (VOBench split v2, late · 이동, 256² 사본; 문맥 encoder = online, 목표 = LN(target_encoder(32 장)), float32 + autocast float16):
  Kobj   = 보이는 문맥 튜블릿 (두 장 모두 보임) 마다 물체 자리 3×3 (블록 평균 토큰, 중심 1–14) 의 토큰 합집합 — 물체가 있는 문맥 팔의 라벨.
  Kctl   = 같은 튜블릿에서 그 상자를 세로로 5 칸 옮긴 상자 (행 ≥ 6 이면 위로, 아니면 아래로) — 같은 개수 · 물체 상자와 안 겹침.
  막기   = 층 띠의 각 층에서 (모든 미래 토큰 → Kobj) 간선을 끊는다 (bool attn_mask). 대조 막기 = Kctl.
  키우기 = 릴리즈의 같은 층 띠에서 (다시 나타난 미래 튜블릿의 참 물체 자리 3×3 query → Kobj) logit 에 +β. 대조 = Kctl.
  같은 마스크를 matched 쌍의 두 미래 · 빈 장면 팔 문맥에도 쓴다 (마스크는 물체 팔 라벨로 정함).
  ① 사라짐 A = s(imp_ab) > s(pos_a), B = s(imp_ba) > s(pos_b)  (s = mean |p − LN(h)[미래]|, permanence block)
  ② 존재 대비 지도 D = ‖p(물체 팔 문맥) − p(빈 장면 팔 문맥)‖₂ (토큰별). 다시 나타난 미래 튜블릿마다 3×3 상자 합 최대 자리가
     참 자리 1 칸 안 (체비쇼프) 인 비율 · 참 상자 / 상자 평균 (대비 배율).
  ③ 국소 판독 = p 의 참 자리 3×3 평균 (다시 나타난 미래 튜블릿마다) 에 건 선형 판독기 (능형 회귀 → 원핫, λ 는 train block 5 겹으로 고름).
     과제 = primitive 모양 · realistic 모양 · primitive 색 (+ realistic 색). 판독기는 **train block 의 깨끗한 PFT p** 로만 맞추고 test block 에 건다.
     릴리즈 수준은 릴리즈 깨끗한 p 로 맞춘 자기 판독기 (같은 절차) 로 낸다.
E38b (계획 밖 추가, --set v11): 같은 막기 · 키우기를 v11 split test 가림 이동 사라짐 224 block 에 (VOBench PFT 의 학습 생성기 밖). 국소 판독 없음.
층 띠: E37 (a) 에서 VOBench PFT 의 층별 R ≥ 2 인 층 (BOARD 에 E38 결과 전에 적음) — --layers 로 준다.

재현:
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  sbatch -p batch_vll -w vll4 --gres=gpu:2 ... --wrap "$P z_research/scripts/analysis/transfer_gap_r4_ko.py run --layers 1-11 --gpus 2"
  $P z_research/scripts/analysis/transfer_gap_r4_ko.py summarize --layers 1-11     # CPU → results/r4_x1/E38_ko.json
산출: z_analysis/transfer_gap/results/r4_x1/E38/{perm,feat}_r<rank>.npz , E38_ko.json
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
S, NB, C, K, G = 256, 16, 8, 8, 16
OUTR = REPO / "z_analysis/transfer_gap/results/r4_x1"
OUT = OUTR / "E38"
STAGE = OUTR / "stage"
FRAMES = REPO / "z_pixel_decoder/runs/_frames/VOBench_v1_256_x1r4"
CKPT = {"vobftg": "z_training/runs/vobench_release_postft_g/latest.pt"}
BETAS = (2.0, 4.0)


def parse_layers(s):
    out = []
    for p in s.split(","):
        a, b = (p.split("-") + [p])[:2]
        out += list(range(int(a), int(b) + 1))
    return sorted(set(out))


def tracks():
    T = np.load(STAGE / "vobench_tracks.npz", allow_pickle=True)
    return {v: (T["P"][i], T["hid"][i], bool(T["ok"][i])) for i, v in enumerate(T["vid"])}


def geom(P, hid):
    tok = np.stack([np.clip(P[:, 1] // 32, 0, G - 1), np.clip(P[:, 0] // 32, 0, G - 1)], 1).reshape(NB, 2, 2).mean(1)
    yx = np.clip(np.floor(tok + 0.5).astype(int), 1, G - 2)
    ins = (P[:, 0] >= 0) & (P[:, 0] < 512) & (P[:, 1] >= 0) & (P[:, 1] < 512)
    vb = ((hid < 0.5) & ins).reshape(NB, 2).all(1)
    return yx, vb


def box(b, y, x):
    return [b * S + (y + dy) * G + (x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def key_sets(yx, vb):
    ko, kc = [], []
    for b in range(C):
        if not vb[b]: continue
        y, x = yx[b]; yc = y - 5 if y >= 6 else y + 5
        ko += box(b, y, x); kc += box(b, int(np.clip(yc, 1, G - 2)), x)
    return np.array(ko), np.array(kc)


def q_true(yx, vb):
    q = []
    for t in range(C, NB):
        if vb[t]: q += box(t, *yx[t])
    return np.array(q, dtype=np.int64)


def build_groups():
    sp = {}
    for s in ("train", "test"):
        for r in csv.DictReader(open(REPO / f"data_csv/vobench_v1_split_v2/index_{s}.csv")):
            sp[r["video_id"]] = s
    by = defaultdict(dict)
    for r in csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv")):
        if r["occlusion"] != "late" or r["motion"] == "flat_static": continue
        by[(r["objset"], r["block_id"])][r["variant"]] = r
    perm, feat = [], []
    for key, v in by.items():
        r0 = v["pos_a"]; split = sp.get(r0["video_id"], "?")
        if r0["violation"] == "permanence" and split == "test":
            perm.append(dict(block="|".join(key), cell=f"k{r0['k']}|{r0['motion']}|{r0['objset']}", rows=v))
        if r0["violation"] in ("shape", "color"):
            for var in ("pos_a", "pos_b"):
                r = v[var]; lab = (r["obj_a"] if var == "pos_a" else r["obj_b"])
                feat.append(dict(block="|".join(key), split=split, task=f"{r['objset']}_{r0['violation']}", row=r,
                                 label=lab.split("/")[0] if r0["violation"] == "shape" else lab.split("/")[-1]))
    return perm, feat


def build_groups_v11():
    """E38b (계획 밖 추가): v11 split test 가림 이동 사라짐 block — 물체 팔 (pos_a) 궤적 = 원본 metadata (288 px → 토큰 px // 18),
    가림 = 문맥 끝 k + 미래 앞 k 샘플. 반환: perm 목록 (feat 없음) 과 video_id → (P(512 px 환산), hid) 궤적."""
    import pandas as pd
    by = defaultdict(dict)
    for r in csv.DictReader(open(STAGE / "idx_v11sub3/index.csv")):
        if r["condition"] in ("moving_occlusion", "moving_occlusion_flat"):
            by[r["block_id"]][r["variant"]] = r
    meta = pd.read_csv(STAGE / "v11_meta_sub.csv").set_index("name")
    TR, perm = {}, []
    for b, v in by.items():
        r = v["pos_a"]; m = meta.loc[r["video_id"]]
        px = np.array([float(x) for x in str(m["object_px_x_by_sample"]).split()]); py = np.array([float(x) for x in str(m["object_px_y_by_sample"]).split()])
        k = int(r["sym_k"]); hid = np.zeros(32); hid[16 - k:16 + k] = 1.0
        P = np.stack([px, py], 1) * (512 / 288)                                  # geom() 는 512 px · 32 px 칸을 가정 → 같은 토큰 (px // 18)
        TR[r["video_id"]] = (P, hid, True)
        perm.append(dict(block=b, cell=f"{r['condition']}|k{k}", rows=v))
    return perm, TR


def _worker(rank, world, layers, bs, limit, setname="vobench"):
    import torch
    from analysis.intphys2.model import build_from_config, _clean_backbone_key, _load_state_dict
    from analysis.intphys2.surprise import _context_target_indices
    from analysis.attention.knockout import runner as RN
    from evals.world_model_analysis.data import WMADataset
    import copy as _copy
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    model = dict(checkpoint=str(REPO / "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth"),
                 arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, window_size=32, use_rope=True, uniform_power=False,
                 dual_encoder=True, context_encoder_key="encoder", target_encoder_key="target_encoder",
                 predictor=dict(embed_dim=384, depth=12, num_heads=12, num_mask_tokens=10), dtype="float32")
    bundle = build_from_config(model, dev)
    preds = {"release": bundle.predictor}
    for name, path in CKPT.items():
        pm = _copy.deepcopy(bundle.predictor); st = torch.load(REPO / path, map_location="cpu", weights_only=False)
        assert (st.get("arch") or {}).get("kind") in (None, "default", "oneshot")
        assert not (set(pm.state_dict()) - set(_clean_backbone_key(st["predictor"])))
        _load_state_dict(pm, st["predictor"], tag=name); preds[name] = pm.to(dev).eval()
    if setname == "v11":
        perm, TR = build_groups_v11(); feat = []
    else:
        TR = tracks()
        perm, feat = build_groups()
    if limit: perm, feat = perm[:limit], feat[:2 * limit]
    rows = {}
    for g in perm:
        for r in g["rows"].values(): rows[r["video_id"]] = r
    for f in feat: rows[f["row"]["video_id"]] = f["row"]
    idir = OUT / ("index" if setname == "vobench" else f"index_{setname}"); idir.mkdir(parents=True, exist_ok=True)
    allr = list(rows.values())
    with open(idir / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allr[0].keys())); w.writeheader(); w.writerows(allr)
    fr = FRAMES if setname == "vobench" else REPO / "z_pixel_decoder/runs/_frames/IntPhysGen_v11_sub"
    ds = WMADataset(dict(data=dict(root=str(idir), index_csv="index.csv", frames_root=str(fr), frames_pattern="{file_name}/{frame:06d}.png",
                                   frames_start=0, frames_stride=3, n_frames=32, resolution=256, block_column="block_id", pair_column="pair_id",
                                   variant_column="variant", plausible_column="plausible", type_column="condition"),
                         model=model, features={"cache_dir": "/tmp"}, surprise={}))
    di = {r.video_id: i for i, r in enumerate(ds.records)}
    N = NB * S
    L = len(bundle.predictor.predictor_blocks); l0 = min(layers)
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

    def ko_mask(Ksets, B):
        am = torch.ones(B, 1, N, N, dtype=torch.bool, device=dev)
        for j, ks in enumerate(Ksets):
            if len(ks): am[j, 0, C * S:, torch.as_tensor(ks, device=dev)] = False
        return am

    def steer_mask(Qsets, Ksets, B, beta):
        am = torch.zeros(B, 1, N, N, dtype=torch.float16, device=dev)
        for j, (qs, ks) in enumerate(zip(Qsets, Ksets)):
            if len(qs) and len(ks):
                qq = torch.as_tensor(qs, device=dev); kk = torch.as_tensor(ks, device=dev)
                am[j, 0, qq[:, None], kk[None, :]] = beta
        return am

    def run_conds(pm, z, ci, ti, B, conds):
        """conds: {name: per-layer mask 또는 None} → {name: p (B, C*S, D) float32}"""
        out = {}
        with ac:
            p0, prefix, masks, n_ctx = RN.clean_forward(pm, z, ci, ti, mask_index=0, keep_prefix=True)
            out["clean"] = p0.float()
            for nm, am in conds.items():
                ams = [am if l in layers else None for l in range(L)]
                out[nm] = RN.resume(pm, prefix[l0], l0, masks, n_ctx, ams).float()
        return out, (p0, prefix, masks, n_ctx)

    verify = {}
    acc = defaultdict(list)
    t0 = time.time()
    mine_p = perm[rank::world]; mine_f = feat[rank::world]
    # ── permanence block: 4 clip
    for gi in range(0, len(mine_p), bs):
        gg = mine_p[gi:gi + bs]; B = len(gg)
        va = [g["rows"]["pos_a"]["video_id"] for g in gg]; vai = [g["rows"]["imp_ab"]["video_id"] for g in gg]
        vb_ = [g["rows"]["pos_b"]["video_id"] for g in gg]; vbi = [g["rows"]["imp_ba"]["video_id"] for g in gg]
        with torch.no_grad():
            zA, hA, ci, ti = enc(va); _, hAi, _, _ = enc(vai); zB, hB, _, _ = enc(vb_); _, hBi, _, _ = enc(vbi)
            G_ = [geom(*TR[v][:2]) for v in va]
            KS = [key_sets(*g) for g in G_]; QS = [q_true(*g) for g in G_]
            m_ko = ko_mask([k[0] for k in KS], B); m_kc = ko_mask([k[1] for k in KS], B)
            res = {}
            for name, pm in preds.items():
                conds = {"ko_obj": m_ko, "ko_ctl": m_kc} if name != "release" else {}
                if name == "release":
                    for bt in BETAS:
                        conds[f"st_obj_b{bt:g}"] = steer_mask(QS, [k[0] for k in KS], B, bt)
                        conds[f"st_ctl_b{bt:g}"] = steer_mask(QS, [k[1] for k in KS], B, bt)
                pa, aux = run_conds(pm, zA, ci, ti, B, conds)
                pb, _ = run_conds(pm, zB, ci, ti, B, conds)
                if not verify.get(name):
                    with ac:
                        r0 = RN.resume(pm, aux[1][0], 0, aux[2], aux[3], None).float()
                    verify[name] = float((r0 - pa["clean"]).abs().max())
                for cn in pa:
                    l1 = lambda p, h: (p - h).abs().mean((1, 2)).cpu().numpy()
                    D = (pa[cn] - pb[cn]).norm(dim=-1).view(B, K, G, G)
                    Db = torch.nn.functional.avg_pool2d(D.reshape(B * K, 1, G, G), 3, 1, 1).view(B, K, G, G)[:, :, 1:-1, 1:-1]
                    res[f"{name}__{cn}"] = dict(sA_pos=l1(pa[cn], hA), sA_imp=l1(pa[cn], hAi), sB_pos=l1(pb[cn], hB), sB_imp=l1(pb[cn], hBi),
                                                Db=Db.cpu().numpy().astype(np.float32))
            for k_, d in res.items():
                for kk, vv in d.items(): acc[f"perm__{k_}__{kk}"].append(vv)
            acc["perm__block"].append(np.array([g["block"] for g in gg])); acc["perm__cell"].append(np.array([g["cell"] for g in gg]))
            acc["perm__yx"].append(np.stack([g[0] for g in G_])); acc["perm__vb"].append(np.stack([g[1] for g in G_]))
        if rank == 0: print(f"[r0] perm {gi + B}/{len(mine_p)} ({time.time() - t0:.0f}s) verify {verify}", flush=True)
    # ── 모양 · 색 block 의 국소 특징 (clip 마다 자기 팔 라벨)
    for gi in range(0, len(mine_f), bs):
        ff = mine_f[gi:gi + bs]; B = len(ff)
        vids = [f["row"]["video_id"] for f in ff]
        with torch.no_grad():
            z, h, ci, ti = enc(vids)
            G_ = [geom(*TR[v][:2]) for v in vids]
            KS = [key_sets(*g) for g in G_]; QS = [q_true(*g) for g in G_]
            m_ko = ko_mask([k[0] for k in KS], B); m_kc = ko_mask([k[1] for k in KS], B)
            for name, pm in preds.items():
                conds = {"ko_obj": m_ko, "ko_ctl": m_kc} if name != "release" else {f"st_obj_b{BETAS[-1]:g}": steer_mask(QS, [k[0] for k in KS], B, BETAS[-1]),
                                                                                    f"st_ctl_b{BETAS[-1]:g}": steer_mask(QS, [k[1] for k in KS], B, BETAS[-1])}
                pp, _ = run_conds(pm, z, ci, ti, B, conds)
                for cn, p in pp.items():
                    P_ = p.view(B, K, G, G, -1)
                    F_ = np.full((B, K, P_.size(-1)), np.nan, np.float16)
                    for j in range(B):
                        yx, vb = G_[j]
                        for t in range(K):
                            if vb[C + t]:
                                y, x = yx[C + t]; F_[j, t] = P_[j, t, y - 1:y + 2, x - 1:x + 2].mean((0, 1)).cpu().numpy().astype(np.float16)
                    acc[f"feat__{name}__{cn}"].append(F_)
            Hf = h.view(B, K, G, G, -1); FH = np.full((B, K, Hf.size(-1)), np.nan, np.float16)
            for j in range(B):
                yx, vb = G_[j]
                for t in range(K):
                    if vb[C + t]:
                        y, x = yx[C + t]; FH[j, t] = Hf[j, t, y - 1:y + 2, x - 1:x + 2].mean((0, 1)).cpu().numpy().astype(np.float16)
            acc["feat__h"].append(FH)
            for k_ in ("block", "split", "task", "label"):
                acc[f"feat__{k_}"].append(np.array([f[k_] for f in ff]))
            acc["feat__vid"].append(np.array(vids))
        if rank == 0 and (gi // bs) % 5 == 0: print(f"[r0] feat {gi + B}/{len(mine_f)} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / (f"ko_r{rank}.npz" if setname == "vobench" else f"ko_{setname}_r{rank}.npz"), layers=np.array(layers), verify=json.dumps(verify),
             **{k: np.concatenate(v) for k, v in acc.items()})
    print(f"[r{rank}] done {time.time() - t0:.0f}s verify {verify}", flush=True)


# ───────────────────────── 요약
def load(setname="vobench"):
    Z = {}
    for f in sorted(OUT.glob("ko_r*.npz" if setname == "vobench" else f"ko_{setname}_r*.npz")):
        z = np.load(f, allow_pickle=True)
        for k in z.files:
            if k in ("layers", "verify"): Z[k] = z[k]; continue
            Z.setdefault(k, []).append(z[k])
    return {k: (np.concatenate(v) if isinstance(v, list) else v) for k, v in Z.items()}


def boot_mean(x, blocks, nb=2000, seed=0):
    ub, inv = np.unique(blocks, return_inverse=True)
    s = np.bincount(inv, x, len(ub)); n = np.bincount(inv, None, len(ub))
    rng = np.random.default_rng(seed); bs = []
    for _ in range(nb):
        ii = rng.integers(0, len(ub), len(ub)); bs.append(s[ii].sum() / n[ii].sum())
    return [float(100 * s.sum() / n.sum()), float(100 * np.percentile(bs, 2.5)), float(100 * np.percentile(bs, 97.5))]


def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ Y)
    return mu, sd, W


def ridge_pred(m, X):
    mu, sd, W = m
    return ((X - mu) / sd) @ W


def summarize(layers, setname="vobench"):
    Z = load(setname)
    out = {"definition": __doc__.split("정의")[1].split("층 띠")[0].strip(), "layers": layers, "verify_max_abs": json.loads(str(Z["verify"]))}
    conds = sorted({k.split("__")[1] + "__" + k.split("__")[2] for k in Z if k.startswith("perm__") and k.endswith("__sA_pos")})
    blocks = Z["perm__block"]; yx = Z["perm__yx"]; vb = Z["perm__vb"]
    res = {}
    for c in conds:
        A = (Z[f"perm__{c}__sA_imp"] > Z[f"perm__{c}__sA_pos"]).astype(float)
        Bv = (Z[f"perm__{c}__sB_imp"] > Z[f"perm__{c}__sB_pos"]).astype(float)
        Db = Z[f"perm__{c}__Db"]                                  # (n, K, 14, 14) 상자 중심 1..14
        hit, con, bl = [], [], []
        for i in range(len(blocks)):
            for t in range(K):
                if not vb[i, C + t]: continue
                y, x = yx[i, C + t]
                m = Db[i, t]; a = np.unravel_index(np.argmax(m), m.shape); a = (a[0] + 1, a[1] + 1)
                hit.append(float(max(abs(a[0] - y), abs(a[1] - x)) <= 1)); con.append(float(m[y - 1, x - 1] / m.mean())); bl.append(blocks[i])
        hit, con, bl = np.array(hit), np.array(con), np.array(bl)
        res[c] = dict(A=boot_mean(A, blocks), B=boot_mean(Bv, blocks), within1=boot_mean(hit, bl),
                      contrast=[float(con.mean()), float(np.median(con))], n_block=int(len(np.unique(blocks))), n_step=int(len(hit)))
    out["perm"] = res
    if setname != "vobench":
        out["set"] = setname
        g0 = lambda c, k: res[c][k][0] if c in res else np.nan
        dA = g0("vobftg__clean", "A") - g0("vobftg__ko_obj", "A"); dAc = g0("vobftg__clean", "A") - g0("vobftg__ko_ctl", "A")
        out["verdict_ko"] = dict(dA=dA, dA_ctl=dAc, within1_drop=g0("vobftg__clean", "within1") - g0("vobftg__ko_obj", "within1"),
                                 within1_drop_ctl=g0("vobftg__clean", "within1") - g0("vobftg__ko_ctl", "within1"),
                                 verdict=("v11 이득이 그 간선에 기댄다" if dA >= 30 and dAc < dA / 3 else "기대지 않는다" if dA < 10 else "부분"))
        out["verdict_steer"] = {f"b{bt:g}": dict(dA_obj=g0(f"release__st_obj_b{bt:g}", "A") - g0("release__clean", "A"),
                                                 dA_ctl=g0(f"release__st_ctl_b{bt:g}", "A") - g0("release__clean", "A")) for bt in BETAS}
        json.dump(out, open(OUTR / f"E38b_ko_{setname}.json", "w"), indent=1, ensure_ascii=False)
        for c, d in res.items():
            print(f"{c:28s} A {d['A'][0]:5.1f} [{d['A'][1]:.0f},{d['A'][2]:.0f}]  B {d['B'][0]:5.1f}  1칸안 {d['within1'][0]:5.1f} [{d['within1'][1]:.0f},{d['within1'][2]:.0f}]  대비배율 {d['contrast'][0]:.2f}  (blk {d['n_block']}, step {d['n_step']})")
        print(out["verdict_ko"]); print(out["verdict_steer"])
        return
    # 국소 판독
    task = Z["feat__task"]; split = Z["feat__split"]; lab = Z["feat__label"]; fb = Z["feat__block"]
    fconds = sorted({k[len("feat__"):] for k in Z if k.startswith("feat__") and "__" in k[len("feat__"):]})
    rd = {}
    for tk in sorted(set(task)):
        sel = task == tk
        classes = sorted(set(lab[sel])); ci = {c: i for i, c in enumerate(classes)}
        def rows(cn, which):
            F = Z[f"feat__{cn}"] if cn != "h" else Z["feat__h"]
            X, y, b = [], [], []
            for i in np.nonzero(sel & (split == which))[0]:
                for t in range(K):
                    if not np.isnan(F[i, t, 0]): X.append(F[i, t].astype(np.float32)); y.append(ci[lab[i]]); b.append(fb[i])
            return np.array(X), np.array(y), np.array(b)
        def fit(cn):
            X, y, b = rows(cn, "train")
            if len(X) == 0: return None
            Y = np.eye(len(classes))[y]
            ub = np.unique(b); rng = np.random.default_rng(0); fold = {u: f for u, f in zip(ub, rng.permutation(len(ub)) % 5)}
            fo = np.array([fold[u] for u in b]); best = None
            for lam in (1e1, 1e2, 1e3, 1e4):
                ok = []
                for f in range(5):
                    m = ridge_fit(X[fo != f], Y[fo != f], lam); ok.append((ridge_pred(m, X[fo == f]).argmax(1) == y[fo == f]).mean())
                if best is None or np.mean(ok) > best[0]: best = (np.mean(ok), lam)
            return ridge_fit(X, Y, best[1]), best[1]
        r = {"classes": classes, "chance": 100 / len(classes)}
        m_pft = fit("vobftg__clean"); m_rel = fit("release__clean")
        for cn in fconds:
            X, y, b = rows(cn, "test")
            if len(X) == 0: continue
            mm = m_rel if cn.startswith("release") else m_pft
            if mm is None: continue
            r[cn] = boot_mean((ridge_pred(mm[0], X).argmax(1) == y).astype(float), b) + [float(mm[1])]
            if cn.startswith("release"):
                r[cn + "__pft_readout"] = boot_mean((ridge_pred(m_pft[0], X).argmax(1) == y).astype(float), b)
        rd[tk] = r
    out["readout"] = rd
    # 판정 (사전 등록)
    P = res
    def g(c, k): return P[c][k][0] if c in P else np.nan
    dA = g("vobftg__clean", "A") - g("vobftg__ko_obj", "A"); dAc = g("vobftg__clean", "A") - g("vobftg__ko_ctl", "A")
    gapW = g("vobftg__clean", "within1") - g("release__clean", "within1"); dW = g("vobftg__clean", "within1") - g("vobftg__ko_obj", "within1")
    dWc = g("vobftg__clean", "within1") - g("vobftg__ko_ctl", "within1")
    rdrops = []
    for tk, r in rd.items():
        if "vobftg__clean" in r and "release__clean" in r:
            gap = r["vobftg__clean"][0] - r["release__clean"][0]
            rdrops.append(dict(task=tk, gap=gap, drop=r["vobftg__clean"][0] - r.get("vobftg__ko_obj", [np.nan])[0],
                               drop_ctl=r["vobftg__clean"][0] - r.get("vobftg__ko_ctl", [np.nan])[0]))
    second = (gapW > 0 and dW >= gapW / 3) or any(d["gap"] > 0 and d["drop"] >= d["gap"] / 3 for d in rdrops)
    ctl_ok = (dAc < dA / 3) and (dWc < max(dW, 1e-9) / 3 or dW <= 0)
    if dA >= 30 and second and ctl_ok: v = "경로가 필요하다"
    elif dA < 10 and (gapW <= 0 or dW < 0.1 * gapW) and all(d["drop"] < 0.1 * max(d["gap"], 1e-9) for d in rdrops if d["gap"] > 0): v = "필요 없다 (다른 경로)"
    else: v = "부분"
    out["verdict_ko"] = dict(dA=dA, dA_ctl=dAc, within1_gap=gapW, within1_drop=dW, within1_drop_ctl=dWc, readout=rdrops, verdict=v)
    st = {}
    for bt in BETAS:
        a0 = g("release__clean", "A"); ao = g(f"release__st_obj_b{bt:g}", "A"); acn = g(f"release__st_ctl_b{bt:g}", "A")
        st[f"b{bt:g}"] = dict(dA_obj=ao - a0, dA_ctl=acn - a0)
    up = [k for k, d in st.items() if d["dA_obj"] >= 20 and d["dA_ctl"] < 5]
    both = [k for k, d in st.items() if d["dA_obj"] >= 20 and d["dA_ctl"] >= 5]
    out["verdict_steer"] = dict(st, verdict=("그 경로를 키우면 사라짐 정답이 오른다" if up else "경로 특이성 없음" if both else "오르지 않는다"))
    json.dump(out, open(OUTR / "E38_ko.json", "w"), indent=1, ensure_ascii=False)
    for c, d in res.items():
        print(f"{c:28s} A {d['A'][0]:5.1f} [{d['A'][1]:.0f},{d['A'][2]:.0f}]  B {d['B'][0]:5.1f}  1칸안 {d['within1'][0]:5.1f} [{d['within1'][1]:.0f},{d['within1'][2]:.0f}]  대비배율 {d['contrast'][0]:.2f}  (blk {d['n_block']}, step {d['n_step']})")
    for tk, r in rd.items():
        print(tk, {k: (round(v[0], 1) if isinstance(v, list) else v) for k, v in r.items() if k != "classes"})
    print(out["verdict_ko"]); print(out["verdict_steer"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--layers", default="5-11"); ap.add_argument("--gpus", type=int, default=1); ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--set", default="vobench", choices=["vobench", "v11"])
    a = ap.parse_args(); layers = parse_layers(a.layers)
    if a.cmd == "run":
        import torch
        torch.multiprocessing.spawn(_worker, args=(a.gpus, layers, a.bs, a.limit, a.set), nprocs=a.gpus, join=True)
    summarize(layers, a.set)
