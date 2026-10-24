#!/usr/bin/env python3
"""G19 · G20 · G21 확장 (진행자 1, 2026-10-22) — 추정기는 carry_coef.py (진행자 2) 를 그대로 쓴다. 정의 · 판정 규칙 = EVIDENCE §3-2a.

이 파일이 더하는 것:
  (1) 운반 연산 (b) 'pix' — 픽셀을 실제 변위만큼 옮긴 뒤 같은 encoder 로 인코딩 (carry_s_pixel.py). carry_coef.S_MAKERS 에 등록.
  (2) encoder 변형 prep (G20 · G21): 같은 세트 · 같은 칸 · 같은 회귀를 다른 블록 인코딩에서.
        vjepa2   = 연속 2 장 블록 (carry_coef.prep 과 같은 것; 이 파일로 돌리면 pix 가 같이 들어간다)
        dup      = 같은 V-JEPA 2 encoder, 블록 (f_{2b+1}, f_{2b+1}) — 한 장 복제 (V-JEPA 2-AC 꼴)
        dinov2   = DINOv2 ViT-B/14-reg 최종 층 패치 토큰 (224 px → 16 × 16), 블록 = 두 장 평균, 블록별 LN
  (3) encprops (G21): 떨림 (±1/255 균등 잡음) · 칸 미만 매끄러움 (반 칸 = 8 px 평행 이동) — 블록 7, 같은 영상, encoder 별.
  (4) G20 블록 안 움직임: VOBench 보이는 이동에서 물체 3 × 3 칸 토큰 평균 → 블록 안 변위 (라벨, 256 px) 의 ridge 판독,
      묶음 (source_block) 단위로 반을 떼어 보류 R² · 라벨 섞은 대조. (vjepa2 · dup 를 한 패스에서)

  CUDA_VISIBLE_DEVICES=2 python paper_lab/scripts/carry_ext.py prep --set natural --enc vjepa2     # (= carry_coef prep + pix)
  CUDA_VISIBLE_DEVICES=2 python paper_lab/scripts/carry_ext.py prep --set vobench --enc dup
  CUDA_VISIBLE_DEVICES=2 python paper_lab/scripts/carry_ext.py prep --set natural --enc dinov2
  CUDA_VISIBLE_DEVICES=2 python paper_lab/scripts/carry_ext.py encprops --set natural --n 100
  CUDA_VISIBLE_DEVICES=2 python paper_lab/scripts/carry_ext.py velprobe --n 400
산출: 통계 paper_lab/_scratch/carry/<set>/_data[_<enc>].npz (carry_coef 형식) · encprops/velprobe json 은 paper_lab/_scratch/carry/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import carry_coef as CC          # noqa: E402  (sys.path 도 여기서 잡힌다)
import carry_s_pixel as SP       # noqa: E402

G, C, NB, CELL = CC.G, CC.C, CC.NB, CC.CELL
DINO_REPO = "/data/hyuntak/project/2026/2027_cvpr/DINO-Foresight/third_party/dinov2"
DINO_W = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth"
ENC = {"name": "vjepa2", "fn": None}          # 현재 encoder (pix 가 같은 encoder 로 인코딩하도록)


# ---------------------------------------------------------------- encoders: x (B, 3, T, 256, 256) 정규화 → (B, T//2, G, G, D) LN
def enc_vjepa2(bundle, x):
    from app.vjepa_frozen.ar import encode_blocks
    B, T = x.shape[0], x.shape[2]
    h = encode_blocks(bundle.target_encoder, x, 2).float()
    return h.view(B, T // 2, G, G, h.shape[-1])


def enc_dup(bundle, x):
    xd = x.clone()
    xd[:, :, 0::2] = x[:, :, 1::2]                       # 블록 (f_{2b+1}, f_{2b+1}) — 블록의 뒤 장을 두 번 (AC 꼴)
    return enc_vjepa2(bundle, xd)


_DINO = {}


def _dino(dev):
    import torch
    if "m" not in _DINO:
        m = torch.hub.load(DINO_REPO, "dinov2_vitb14_reg", source="local", pretrained=False)
        sd = torch.load(DINO_W, map_location="cpu")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"DINOv2 weights missing {missing[:5]}")
        _DINO["m"] = m.to(dev).eval()
    return _DINO["m"]


def _dino_frames(x, frames_idx, noise_amp=0.0, gen=None):
    """x (B, 3, T, 256, 256) 정규화 → 고른 장마다 DINOv2 (224 px bilinear 축소 뒤, 잡음은 224 에서 = 그 encoder 의 입력 해상도)
    최종 층 정규화 패치 토큰 (B, len(frames_idx), G, G, 768)."""
    import torch
    import torch.nn.functional as F
    B, _, T, H, W = x.shape
    m = _dino(x.device)
    fr = x[:, :, frames_idx].transpose(1, 2).reshape(-1, 3, H, W).float()
    fr = F.interpolate(fr, size=(224, 224), mode="bilinear", align_corners=False, antialias=False)
    if noise_amp:
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        fr = fr + (torch.rand(fr.shape, generator=gen, device=x.device) * 2 - 1) * noise_amp / std
    outs = []
    with torch.autocast("cuda", enabled=False):
        for k in range(0, fr.shape[0], 64):
            outs.append(m.forward_features(fr[k:k + 64])["x_norm_patchtokens"].float())   # (n, 256, 768)
    return torch.cat(outs).view(B, len(frames_idx), G, G, -1)


def enc_dinov2(bundle, x, noise_amp=0.0, gen=None):
    """주 판 (진행자 2 검토 반영): 블록 b = DINOv2(f_{2b+1}) **한 장** (블록의 뒤 장 = G20 복제 판과 같은 시점) → LN.
    image encoder world model (DINO-WM · DINO-world · DINO-Foresight · AC 의 복제) 과 같은 '한 장 = 한 상태'."""
    import torch.nn.functional as F
    T = x.shape[2]
    tok = _dino_frames(x, list(range(1, T, 2)), noise_amp, gen)
    return F.layer_norm(tok, (tok.shape[-1],))


def enc_dinov2avg(bundle, x, noise_amp=0.0, gen=None):
    """보조 판: 블록 = 두 장 DINOv2 토큰 평균 → LN (평균이 떨림을 ~1/√2 로 깎고 움직임을 흐리므로 떨림 비교에는 안 쓴다)."""
    import torch.nn.functional as F
    B, T = x.shape[0], x.shape[2]
    tok = _dino_frames(x, list(range(T)), noise_amp, gen).view(B, T // 2, 2, G, G, -1).mean(2)
    return F.layer_norm(tok, (tok.shape[-1],))


ENCODERS = {"vjepa2": enc_vjepa2, "dup": enc_dup, "dinov2": enc_dinov2, "dinov2avg": enc_dinov2avg}


# ---------------------------------------------------------------- (b) pix: S_MAKERS 등록
def _rot90(v):
    """(…, 2) 변위 (x, y) 를 90° 돌림 — 위약 옮기기 (같은 크기, 실제와 다른 방향; 공간 상관은 같다). 사후 판 (§3-2a)."""
    v = np.asarray(v, np.float32)
    return np.stack([-v[..., 1], v[..., 0]], -1)


def _pix_block(x_b, info, rot=False):
    """한 영상 → 옮긴 블록 프레임 (3, 2, H, W). 자연 = info['flow_b'] (F(16→14), F(17→15)) 로 역방향 와핑,
    렌더 = info['objs'] 상자 (도착 중심 ± 반폭) 안만 라벨 변위로 옮기고 밖은 문맥 16 장 화소 중앙값 배경."""
    import torch
    if "flow_b" in info:
        outs = [SP.backward_warp(x_b[:, s], _rot90(info["flow_b"][i]) if rot else info["flow_b"][i]) for i, (s, t) in enumerate(SP.SRC_TGT)]
        return torch.stack(outs, 1)
    _, T, H, W = x_b.shape
    bg = x_b[:, :16].float().median(dim=1).values.to(x_b.dtype)
    ys, xs = np.mgrid[0:H, 0:W]
    outs = []
    for s, t in SP.SRC_TGT:
        disp = np.zeros((H, W, 2), np.float32)
        inside = np.zeros((H, W), bool)
        for o in info["objs"]:
            hx, hy = max(o["half"][0], 1.0), max(o["half"][1], 1.0)
            m = (np.abs(xs - o["dst"][0]) <= hx) & (np.abs(ys - o["dst"][1]) <= hy)
            d_ = np.array(o["src"], np.float32) - np.array(o["dst"], np.float32)
            disp[m] = _rot90(d_) if rot else d_
            inside |= m
        moved = SP.backward_warp(x_b[:, s], disp)
        outs.append(torch.where(torch.from_numpy(inside).to(x_b.device)[None], moved, bg))
    return torch.stack(outs, 1)


def make_pix(x, infos, bundle, rot=False):
    import torch
    blk = torch.stack([_pix_block(x[b], infos[b], rot) for b in range(x.shape[0])])     # (B, 3, 2, H, W)
    h = ENC["fn"](bundle, blk)                                                       # (B, 1, G, G, D)
    return h[:, 0]


def make_pix_rot(x, infos, bundle):
    return make_pix(x, infos, bundle, rot=True)


# ---------------------------------------------------------------- prep (carry_coef.prep 의 encoder 일반판)
def prep(setname, n, bs, enc):
    import torch
    import passline as PL
    if enc == "vjepa2":                       # 기본 encoder: carry_coef.prep 그대로 (모델 패스용 캐시 h.npy · s_pix 까지) + pix 등록
        ENC["name"], ENC["fn"] = enc, ENCODERS[enc]
        CC.S_MAKERS.clear()
        CC.S_MAKERS["pix"] = make_pix
        CC.S_MAKERS["pix_rot"] = make_pix_rot
        return CC.prep(setname, n, bs)
    dev = torch.device("cuda", 0)
    model, bundle = PL._load_model("z_training/vast/ar_state/latest.pt", dev)
    ENC["name"], ENC["fn"] = enc, ENCODERS[enc]
    CC.S_MAKERS.clear()
    CC.S_MAKERS["pix"] = make_pix
    CC.S_MAKERS["pix_rot"] = make_pix_rot
    tag = "" if enc == "vjepa2" else f"_{enc}"
    stats_list, names_list, pooled, vids = [], [], {}, []
    buf, t0 = [], time.time()

    def flush():
        if not buf:
            return
        x = torch.stack([b[1] for b in buf]).to(dev, dtype=bundle.dtype)
        infos = [b[2] for b in buf]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hb = ENC["fn"](bundle, x).float()                                        # (B, 16, G, G, D)
            ys = {"true": hb[:, C], "copy": hb[:, C - 1], "mean8": hb[:, :C].mean(1)}
            if setname == "vobench":
                ys["oracle"] = torch.stack([PL._oracle_render(hb[b], i["P"], i["hid"])[0] for b, i in enumerate(infos)])
            extra = {"pix": make_pix(x, infos, bundle).float(), "pix_rot": make_pix_rot(x, infos, bundle).float()}
            ys["s_pix"] = extra["pix"]
            st, pl = CC.batch_stats(hb[:, :CC.NKEEP], infos, ys, extra)
        stats_list.append(st); names_list.append([b[0] for b in buf])
        for k, v in pl.items():
            pooled.setdefault(k, [[], [], []])
            for a_, b_ in zip(pooled[k], v):
                a_.extend(b_)
        vids.extend(b[0] for b in buf)
        buf.clear()
        print(f"[ext prep {setname} {enc}] {len(vids)} clips ({time.time() - t0:.0f}s)", flush=True)

    for it in CC.items(setname, n):
        buf.append(it)
        if len(buf) == bs:
            flush()
    flush()
    stats, l1 = CC.finish(stats_list, pooled, names_list)
    CC.save_stats(CC.OUT / setname / f"_data{tag}.npz", stats, l1, vids, {"set": setname, "n": len(vids), "enc": enc})
    print(f"[ext prep {setname} {enc}] done {len(vids)}", flush=True)
    CC._print_quick(setname, f"_data{tag}")


# ---------------------------------------------------------------- encprops (G21): 떨림 · 칸 미만 매끄러움
def encprops(setname, n, bs):
    import torch
    import passline as PL
    dev = torch.device("cuda", 0)
    model, bundle = PL._load_model("z_training/vast/ar_state/latest.pt", dev)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
    res = {e: {"jit": [], "ref": [], "half": [], "nbr": [], "full": []} for e in ("vjepa2", "dup", "dinov2", "dinov2avg")}   # dup = 한 장 복제 블록 (G23b, 리더 지시 20:15)
    g = torch.Generator(device=dev); g.manual_seed(0)
    buf, done = [], 0

    def shift(blk, px):                                                              # 오른쪽으로 px 평행 이동 (가장자리 복제)
        return torch.cat([blk[..., :1].expand(*blk.shape[:-1], px), blk[..., :-px]], -1)

    def flush():
        nonlocal done
        x = torch.stack([b[1] for b in buf]).to(dev, dtype=torch.float32)[:, :, 14:16]   # 블록 7 만 (B, 3, 2, H, W)
        noise = (torch.rand(x.shape, generator=g, device=dev) * 2 - 1) * (1.0 / 255.0) / std   # V-JEPA 2 입력 해상도 (256) 에서
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for e in res:
                f = ENCODERS[e]
                t0_ = f(bundle, x.to(bundle.dtype)).float()[:, 0]                       # (B, G, G, D)
                if e in ("vjepa2", "dup"):
                    tn = f(bundle, (x + noise).to(bundle.dtype)).float()[:, 0]
                else:                                                                   # DINOv2: 잡음은 224 px 입력에서 (보간이 섭동을 깎지 않게)
                    tn = f(bundle, x.to(bundle.dtype), noise_amp=1.0 / 255.0, gen=g).float()[:, 0]
                th = f(bundle, shift(x, 8).to(bundle.dtype)).float()[:, 0]              # 반 칸
                tf = f(bundle, shift(x, 16).to(bundle.dtype)).float()[:, 0]             # 한 칸 (대조)
                res[e]["jit"].append((tn - t0_).norm(dim=-1)[:, 1:-1, 1:-1].mean().item())
                res[e]["ref"].append((t0_ - t0_.roll(1, 0)).norm(dim=-1)[:, 1:-1, 1:-1].mean().item())   # 다른 영상 같은 칸
                avg = 0.5 * (t0_[:, :, 1:] + t0_[:, :, :-1])                            # 칸 p 와 왼쪽 이웃의 평균
                res[e]["half"].append((th[:, 1:-1, 2:-1] - avg[:, 1:-1, 1:-1]).norm(dim=-1).mean().item())
                res[e]["nbr"].append((t0_[:, 1:-1, 2:-1] - t0_[:, 1:-1, 1:-2]).norm(dim=-1).mean().item())
                res[e]["full"].append((tf[:, 1:-1, 2:-1] - t0_[:, 1:-1, 1:-2]).norm(dim=-1).mean().item())
        done += len(buf)
        buf.clear()
        print(f"[encprops {setname}] {done}", flush=True)

    for it in CC.items(setname, n):
        buf.append(it)
        if len(buf) == bs:
            flush()
    if len(buf) >= 2:
        flush()
    out = {}
    for e, r in res.items():
        J, R, Hh, Nn, Ff = (np.mean(r[k]) for k in ("jit", "ref", "half", "nbr", "full"))
        out[e] = {"jitter_over_other_video": J / R, "half_shift_vs_neighbour_avg_over_neighbour_diff": Hh / Nn,
                  "full_shift_vs_left_neighbour_over_neighbour_diff": Ff / Nn, "n_clips": done,
                  "raw": {"jit": J, "ref": R, "half": Hh, "nbr": Nn, "full": Ff}}
    p = CC.OUT / f"encprops_{setname}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- velprobe (G20): 블록 안 움직임
def velprobe(n, bs):
    import torch
    import passline as PL
    import design_headroom as H
    dev = torch.device("cuda", 0)
    model, bundle = PL._load_model("z_training/vast/ar_state/latest.pt", dev)
    rows = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
    feats = {"vjepa2": [], "dup": [], "dup2": [], "dinov2": [], "dinov2_2": []}
    base = ("vjepa2", "dup", "dinov2")
    ys_, grp = [], []
    buf = []

    def flush():
        x = torch.stack([b[1] for b in buf]).to(dev, dtype=bundle.dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hs = {e: ENCODERS[e](bundle, x).float() for e in base}
        for j, (vid, _, info) in enumerate(buf):
            P, hid = info["P"] * PL.LABEL_SCALE, info["hid"]
            for b in range(2, 8):                                                       # 문맥 블록 2–7, 블록 b−1 · b 의 네 장 모두 물체가 보일 때
                if hid[2 * b - 2:2 * b + 2].max() >= 0.5:
                    continue
                c = P[2 * b:2 * b + 2].mean(0)
                cy, cx = int(np.clip(c[1] // CELL, 1, G - 2)), int(np.clip(c[0] // CELL, 1, G - 2))
                f_ = {e: hs[e][j, :, cy - 1:cy + 2, cx - 1:cx + 2].mean((1, 2)).cpu().numpy() for e in base}   # (16, D) — 같은 칸
                feats["vjepa2"].append(f_["vjepa2"][b])                                       # (i) 연속 블록 하나 (도구 검사 등급)
                feats["dup"].append(f_["dup"][b])                                             # (i) 복제 블록 하나
                feats["dup2"].append(np.concatenate([f_["dup"][b - 1], f_["dup"][b]]))          # (i′) 복제 블록 둘 이어 붙임 = predictor 가 블록 사이에서 얻는 정보
                feats["dinov2"].append(f_["dinov2"][b])
                feats["dinov2_2"].append(np.concatenate([f_["dinov2"][b - 1], f_["dinov2"][b]]))
                ys_.append(P[2 * b + 1] - P[2 * b])                                     # 블록 안 변위 (256 px)
                grp.append(rows[vid]["source_block"])
        buf.clear()
        print(f"[velprobe] samples {len(ys_)}", flush=True)

    for it in CC.items("vobench", n):
        buf.append(it)
        if len(buf) == bs:
            flush()
    if buf:
        flush()
    Y = np.asarray(ys_, np.float64)
    grp = np.asarray(grp)
    ug = np.unique(grp)
    rng = np.random.default_rng(0)
    test_g = set(rng.choice(ug, len(ug) // 2, replace=False))
    te = np.array([g_ in test_g for g_ in grp])
    out = {"n_samples": int(len(Y)), "n_groups": int(len(ug)), "target": "within-block displacement P[2b+1]-P[2b] (256 px), blocks 2-7",
           "note": "(i) vjepa2 vs dup = tool check (renders have no motion blur); (i') vjepa2 vs dup2 = main (information recoverable across blocks)"}

    def ridge_r2(X, Y, tr, te, lams=(1e-2, 1e-1, 1, 10, 100, 1000)):
        Xm, Ym = X[tr].mean(0), Y[tr].mean(0)
        Xt, Yt = X[tr] - Xm, Y[tr] - Ym
        idx = np.arange(tr.sum()); rng2 = np.random.default_rng(1); rng2.shuffle(idx)
        half = len(idx) // 2
        best = None
        for lam in lams:                                                                 # 학습 쪽 안에서 λ 고르기
            a, b_ = idx[:half], idx[half:]
            W = np.linalg.solve(Xt[a].T @ Xt[a] + lam * np.eye(X.shape[1]), Xt[a].T @ Yt[a])
            err = ((Xt[b_] @ W - Yt[b_]) ** 2).sum()
            if best is None or err < best[0]:
                best = (err, lam)
        W = np.linalg.solve(Xt.T @ Xt + best[1] * np.eye(X.shape[1]), Xt.T @ Yt)
        pred = (X[te] - Xm) @ W + Ym
        ss_res = ((pred - Y[te]) ** 2).sum()
        ss_tot = ((Y[te] - Y[te].mean(0)) ** 2).sum()
        return 1 - ss_res / ss_tot, best[1]

    for e, F_ in feats.items():
        X = np.asarray(F_, np.float64)
        r2, lam = ridge_r2(X, Y, ~te, te)
        Ysh = Y.copy(); rng.shuffle(Ysh)
        r2s, _ = ridge_r2(X, Ysh, ~te, te)
        out[e] = {"heldout_R2": float(r2), "lambda": lam, "shuffled_label_R2": float(r2s)}
    p = CC.OUT / "velprobe_vobench.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))


def velpos(n):
    """G20 교란 기준선 (진행자 2 지적, CPU): 라벨 자리 (블록 중심 u, v) + 움직임 종류만으로 블록 안 변위를 판독.
    velprobe 와 같은 표본 선택 · 같은 묶음 보류 분할 (rng 0). 특징 = 움직임 종류별 (1, u, v, u², v², uv)."""
    import yaml
    import design_headroom as H
    import passline as PL
    from evals.world_model_analysis.data import WMADataset
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("ir", CC.REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir)
    cfg = yaml.safe_load(open(H.SETS["vobench"]["cfg"]))
    ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=dict(ir.rp.MODEL), features={"cache_dir": "/tmp"}, surprise={}))
    rows = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
    ids = [i for i, rec in enumerate(ds.records) if (r := rows.get(rec.video_id)) and r["occlusion"] == "visible"
           and r["motion"] != "flat_static" and r["plausible"] == "1"]
    motions = sorted({rows[ds.records[i].video_id]["motion"] for i in ids})
    X, Y, grp = [], [], []
    for i in ids[:n]:                                                                   # = carry_coef.vobench_items 의 선택
        r = rows[ds.records[i].video_id]
        lab = json.load(open(H.FR / Path(r["file_name"]).parent / "label.json"))["labels"]
        L = lab.get("A" if r["variant"] in ("pos_a", "imp_ab") else "B")
        if not L or L[0] is None:
            continue
        P = np.array([[L[3 * s_]["u"], L[3 * s_]["v"]] for s_ in range(32)], np.float64) * PL.LABEL_SCALE
        hid = np.array([L[3 * s_]["hidden"] for s_ in range(32)], np.float64)
        if hid[14:18].max() >= 0.5:
            continue
        for b in range(2, 8):                                                           # = velprobe 의 블록 선택
            if hid[2 * b - 2:2 * b + 2].max() >= 0.5:
                continue
            u, v = P[2 * b:2 * b + 2].mean(0) / 256.0
            base = np.array([1.0, u, v, u * u, v * v, u * v])
            f = np.zeros(len(motions) * 6)
            k = motions.index(r["motion"])
            f[6 * k:6 * k + 6] = base
            X.append(f); Y.append(P[2 * b + 1] - P[2 * b]); grp.append(r["source_block"])
    X, Y, grp = np.asarray(X), np.asarray(Y), np.asarray(grp)
    ug = np.unique(grp)
    rng = np.random.default_rng(0)
    test_g = set(rng.choice(ug, len(ug) // 2, replace=False))
    te = np.array([g_ in test_g for g_ in grp])
    out = {"n_samples": int(len(Y)), "n_groups": int(len(ug)), "motions": motions}
    for lam in (1e-4, 1e-2, 1.0):
        W = np.linalg.solve(X[~te].T @ X[~te] + lam * np.eye(X.shape[1]), X[~te].T @ Y[~te])
        pred = X[te] @ W
        out[f"pos_motion_R2_lam{lam}"] = float(1 - ((pred - Y[te]) ** 2).sum() / ((Y[te] - Y[te].mean(0)) ** 2).sum())
    p_ = CC.OUT / "velprobe_posbaseline.json"
    json.dump(out, open(p_, "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    import os
    import torch
    torch.set_num_threads(int(os.environ.get("CARRY_THREADS", 4)))          # 공용 노드: CPU 스레드 묶음 (carry_coef 와 같음)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--set", required=True, choices=CC.SETS); p.add_argument("--enc", default="vjepa2", choices=list(ENCODERS))
    p.add_argument("--n", type=int, default=200); p.add_argument("--bs", type=int, default=8)
    p = sub.add_parser("encprops"); p.add_argument("--set", default="natural", choices=CC.SETS); p.add_argument("--n", type=int, default=100); p.add_argument("--bs", type=int, default=8)
    p = sub.add_parser("velprobe"); p.add_argument("--n", type=int, default=400); p.add_argument("--bs", type=int, default=8)
    p = sub.add_parser("velpos"); p.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.set, a.n, a.bs, a.enc)
    elif a.cmd == "encprops":
        encprops(a.set, a.n, a.bs)
    elif a.cmd == "velpos":
        velpos(a.n)
    else:
        velprobe(a.n, a.bs)


if __name__ == "__main__":
    main()
