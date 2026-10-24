#!/usr/bin/env python3
"""[P] 위약 뺀 운반 신뢰 b — (a) 데이터 최적 vs 모델 · (b) 모델 넷 · (c) 문맥 안정도 구간 조건화 (학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/placebo_b.py [--limit N]

사전 등록: 진행판 [P] 06:52 UTC (KST 15:52). 식은 리더 transport_placebo.py 와 같다.
움직이는 칸 |U[t+1]| ≥ 0.5 칸, t = 4 · 5 · 6. y ≈ a·m + b·W, m = 블록 t−4..t 같은 칸 평균.
  W (토큰 판) = h_t 를 U[t+1] 로 옮김 (칸 내림, 화면 밖은 제자리). W_rot = U[t+1] 을 90° 돌려 옮김 (위약).
  W (픽셀 판) = 블록 t 의 두 장을 칸 흐름 (bilinear 로 픽셀에 늘림 × 16) 으로 뒤로 샘플해 옮긴 뒤 목표 encoder 로 다시 인코딩 (LN). 위약도 같게.
  위약 뺀 b = b(W) − b(W_rot). 정답: data = h_{t+1} · copy = h_t (0 이어야 함) · 모델 넷의 TF 한 걸음 예측.
모델: base = natural_ar_vw_state · vob = natural_ar_vw_state_vob · syn = natural_ar_vw_state_phys3d_only · ctxt = natural_ar_vw_state_ctxt_video (모두 latest).
  ctxt 는 z (블록 0..3, online encoder 영상 모드 + 문맥 마스크) + 관측 블록 4..t (forward_state → render).
문맥 안정도 s = cos(h_t(j), h_{t−1}(p)), j = c + U[t+1](c), p = j + U[t](j) ([T] context_reliability), 경계 0.35 · 0.449 · 0.557 · 0.724.
세트: K710 보류 150 (메타에서 다시 디코딩) · HT100M 160 · SSv2 160 ([E] enc_noise 토큰 · 픽셀 캐시 + natural_transport_ctx5 흐름).
출력: auto_research/results/placebo_b.json."""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F, yaml
import decord

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--nb", type=int, default=1000)
ap.add_argument("--out", default="/workspace/world_model/vjepa2/auto_research/results/placebo_b.json")
a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT  # noqa: E402
sys.argv = _argv
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key  # noqa: E402
from app.vjepa_frozen import utils as UU  # noqa: E402
from app.vjepa_frozen import ar as AR  # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda")
S, G, D = 256, 16, 1280
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
R90 = np.array([[0.0, -1.0], [1.0, 0.0]], np.float32)
EDGES = np.array([0.35, 0.449, 0.557, 0.724])
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
sd = _load_checkpoint(CK)
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, sd["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()
onl = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=32, use_rope=True, uniform_power=False)
_load_state_dict(onl, sd["encoder"], tag="o"); onl = onl.to(dev, torch.bfloat16).eval()
del sd
MODELS = {"base": FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval(),
          "vob": FT.load_pred("natural_ar_vw_state_vob", "latest.pt").to(dev).eval(),
          "syn": FT.load_pred("natural_ar_vw_state_phys3d_only", "latest.pt").to(dev).eval()}
D_ = os.path.join(REPO, "z_training/runs/natural_ar_vw_state_ctxt_video")
cfg = yaml.safe_load(open(os.path.join(D_, "config.yaml")))
ctxt = UU.build_predictor(cfg["model"], 1280, None, torch.device("cpu"), int(cfg["data"].get("n_frames", 64)))
ctxt.load_state_dict({k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D_, "latest.pt"), map_location="cpu", weights_only=False)["predictor"]).items()
                      if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}, strict=True)
ctxt = ctxt.to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
TARGETS = ("data", "copy", "base", "vob", "syn", "ctxt")
KINDS = ("tok_true", "tok_rot", "pix_true", "pix_rot")


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), ok


def warp_tok(ht, u):                                                           # 리더 transport_placebo.warp 와 같다
    j, ok = cell_of(P0 + u)
    out = ht[j].copy(); out[~ok] = ht[~ok]
    return out


def windows():
    nh = "/workspace/auto_research_cache/natural_holdout"
    Hn = np.load(f"{nh}/k710_320_holdout_h.npy", mmap_mode="r"); Un = np.load(f"{nh}/k710_320_holdout_U.npy", mmap_mode="r")
    meta = json.load(open(f"{nh}/k710_320_holdout_meta.json"))
    for w in range(min(Hn.shape[0], a.limit or 10 ** 9)):
        m = meta[w]; vr = decord.VideoReader(m["path"], num_threads=1)
        idx = [m["start"] + m["fstp"] * k for k in range(16)]
        if idx[-1] >= len(vr):
            continue
        fr = vr.get_batch(idx).asnumpy(); Hh, Ww = fr.shape[1:3]; sd_ = min(Hh, Ww); y0, x0 = (Hh - sd_) // 2, (Ww - sd_) // 2
        x = torch.from_numpy(fr[:, y0:y0 + sd_, x0:x0 + sd_]).to(dev).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round()
        yield "k710_holdout", w, np.asarray(Hn[w], np.float32), np.asarray(Un[w], np.float32).reshape(8, S, 2), x
    EN = "/workspace/auto_research_cache/enc_noise"
    for src in ("ht100m", "ssv2"):
        Uw = np.load(os.path.join(REPO, f"auto_research/results/natural_transport_ctx5/rows_{src}.npz"))["Uwin"].astype(np.float32)
        w0 = 0
        for sh in (0, 1):
            T_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r"); PX = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_pix.npy"), mmap_mode="r")
            for w in range(min(T_.shape[0], a.limit or 10 ** 9)):
                yield src, w0 + w, np.asarray(T_[w, 0], np.float32), Uw[w0 + w].reshape(8, S, 2), torch.from_numpy(np.asarray(PX[w])).to(dev).permute(0, 3, 1, 2).float()
            w0 += T_.shape[0]


yy, xx = torch.meshgrid(torch.arange(256, device=dev).float(), torch.arange(256, device=dev).float(), indexing="ij")


@torch.no_grad()
def pix_moved(fr2, u):                                                         # fr2 (2,3,256,256) 0..255, u (S,2) 칸 단위 → 옮긴 두 장 (뒤로 샘플)
    f = torch.from_numpy(u).to(dev).view(1, G, G, 2).permute(0, 3, 1, 2) * 16.0   # (1,2,16,16) px
    f = F.interpolate(f, size=(256, 256), mode="bilinear", align_corners=False)[0]  # (2,256,256)
    gx = (xx + f[0]) / 255 * 2 - 1; gy = (yy + f[1]) / 255 * 2 - 1
    grid = torch.stack([gx, gy], -1)[None].expand(2, -1, -1, -1)
    return F.grid_sample(fr2, grid, mode="bilinear", padding_mode="border", align_corners=True)


@torch.no_grad()
def enc_blocks(frs):                                                           # (n·2,3,256,256) 0..255 → (n, S, D) LN
    xx_ = (frs / 255.0).permute(1, 0, 2, 3)[None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = AR.encode_blocks(tgt, ((xx_ - MEAN) / STD).to(torch.bfloat16), 2)
    return h.float()[0].view(frs.shape[0] // 2, S, -1).cpu().numpy()


REC = {}                                                                       # src → list of dict(win, key → arrays)
t0 = time.time()
for n_done, (src, w, h, U, x) in enumerate(windows()):
    hx = torch.from_numpy(h).to(dev)
    P = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for nm, mdl in MODELS.items():
            P[nm] = mdl.forward_seq(hx[:7].reshape(1, -1, D), torch.arange(7 * S, device=dev).unsqueeze(0)).float()[0].view(7, S, -1).cpu().numpy()   # P[t] = 블록 t+1
        xn = ((x / 255.0).permute(1, 0, 2, 3)[None] - MEAN) / STD
        z = onl(xn.to(torch.bfloat16), masks=[torch.arange(4 * S, device=dev).unsqueeze(0)]).float()
        st_ = ctxt.forward_state(z, ctxt.embed_obs(hx[4:7].reshape(1, -1, D)), torch.arange(7 * S, device=dev).unsqueeze(0), 4)
        pc = ctxt.render(st_).float()[0].view(4, S, -1).cpu().numpy()          # 블록 4..7
    # 픽셀 판 W: 세 걸음 × (실제, 위약) 한 번에 인코딩
    frs = []
    for t in (4, 5, 6):
        for uu in (U[t + 1], U[t + 1] @ R90.T):
            frs.append(pix_moved(x[2 * t:2 * t + 2], uu))
    Wp = enc_blocks(torch.cat(frs).clamp(0, 255))                              # (6, S, D)
    rec = {"win": w}
    for i, t in enumerate((4, 5, 6)):
        sel = np.linalg.norm(U[t + 1], axis=1) >= 0.5
        if sel.sum() < 2:
            continue
        m = h[t - 4:t + 1].mean(0)
        Wk = {"tok_true": warp_tok(h[t], U[t + 1]), "tok_rot": warp_tok(h[t], U[t + 1] @ R90.T), "pix_true": Wp[2 * i], "pix_rot": Wp[2 * i + 1]}
        ys = {"data": h[t + 1], "copy": h[t], "base": P["base"][t], "vob": P["vob"][t], "syn": P["syn"][t], "ctxt": pc[t - 3]}
        j, _ = cell_of(P0 + U[t + 1]); pp, _ = cell_of(P0[j] + U[t][j])
        hj, hp = h[t][j], h[t - 1][pp]
        stab = (hj * hp).sum(1) / (np.linalg.norm(hj, axis=1) * np.linalg.norm(hp, axis=1) + 1e-9)
        bins = np.digitize(stab, EDGES)
        for bsel_name, bmask in [("all", sel)] + [(f"bin{k}", sel & (bins == k)) for k in range(5)]:
            if bmask.sum() < 2:
                continue
            for kind, Wv in Wk.items():
                X = np.stack([m[bmask].ravel(), Wv[bmask].ravel()], 1).astype(np.float64)
                key = (bsel_name, kind)
                d = rec.setdefault(key, {"XtX": np.zeros((2, 2)), **{tg: np.zeros(2) for tg in TARGETS}, "n": 0})
                d["XtX"] += X.T @ X
                for tg, y in ys.items():
                    d[tg] += X.T @ y[bmask].ravel().astype(np.float64)
                d["n"] += int(bmask.sum())
    REC.setdefault(src, []).append(rec)
    if n_done % 40 == 0:
        print(f"{src} {w} · {n_done} 창 · {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ 요약 (창 bootstrap, 같은 재표본으로 모든 차이)
rng = np.random.default_rng(0)


def stack(recs):
    keys = sorted({k for r in recs for k in r if k != "win"})
    A = {k: np.stack([r[k]["XtX"] if k in r else np.zeros((2, 2)) for r in recs]) for k in keys}                       # (n,2,2)
    V = {k: np.stack([np.stack([r[k][tg] if k in r else np.zeros(2) for tg in TARGETS]) for r in recs]) for k in keys}   # (n,T,2)
    return keys, A, V


def fit_all(st, wts):
    keys, A, V = st
    out = {}
    for key in keys:
        Aw = np.einsum("n,nij->ij", wts, A[key]); Vw = np.einsum("n,ntj->tj", wts, V[key])
        if abs(np.linalg.det(Aw)) > 1e-9:
            sol = np.linalg.solve(Aw, Vw.T)                                       # (2,T)
            for ti, tg in enumerate(TARGETS):
                out[(key, tg)] = sol[1, ti]
    return out


def summarize(recs):
    st_ = stack(recs)
    base_fit = fit_all(st_, np.ones(len(recs)))
    boots = [fit_all(st_, np.bincount(rng.integers(0, len(recs), len(recs)), minlength=len(recs)).astype(float)) for _ in range(a.nb)]

    def stat(fn):
        v0 = fn(base_fit); vb = np.array([fn(b) for b in boots])
        vb = vb[np.isfinite(vb)]
        return dict(v=round(float(v0), 4), ci=[round(float(np.percentile(vb, 2.5)), 4), round(float(np.percentile(vb, 97.5)), 4)])

    def net(f, bn, ver, tg):
        try:
            return f[((bn, f"{ver}_true"), tg)] - f[((bn, f"{ver}_rot"), tg)]
        except KeyError:
            return np.nan
    r = {"n_windows": len(recs), "n_cells": int(sum(rr[("all", "tok_true")]["n"] for rr in recs if ("all", "tok_true") in rr))}
    for ver in ("tok", "pix"):
        e = {}
        for tg in TARGETS:
            e[f"b_true_{tg}"] = round(float(base_fit.get((("all", f"{ver}_true"), tg), np.nan)), 4)
            e[f"b_rot_{tg}"] = round(float(base_fit.get((("all", f"{ver}_rot"), tg), np.nan)), 4)
            e[f"net_{tg}"] = stat(lambda f, tg_=tg: net(f, "all", ver, tg_))
        for tg in ("base", "vob", "syn", "ctxt"):
            e[f"ratio_{tg}"] = stat(lambda f, tg_=tg: net(f, "all", ver, tg_) / net(f, "all", ver, "data"))
        for tg in ("vob", "syn", "ctxt"):
            e[f"diff_{tg}_minus_base"] = stat(lambda f, tg_=tg: net(f, "all", ver, tg_) - net(f, "all", ver, "base"))
        bins = {}
        for tg in ("data", "copy", "base", "vob", "syn", "ctxt"):
            bins[tg] = [stat(lambda f, k=k, tg_=tg: net(f, f"bin{k}", ver, tg_)) for k in range(5)]
            bins[f"slope_{tg}"] = stat(lambda f, tg_=tg: net(f, "bin4", ver, tg_) - net(f, "bin0", ver, tg_))
        e["bins"] = bins
        r[ver] = e
    return r


res = {"definition": "사전 등록 [P] 06:52 UTC (KST 15:52); 식은 리더 transport_placebo.py"}
for src, recs in REC.items():
    res[src] = summarize(recs)
res["natural_holdout_pooled(k710+ht100m)"] = summarize(REC.get("k710_holdout", []) + REC.get("ht100m", []))
for nm, r in res.items():
    if not isinstance(r, dict) or "tok" not in r:
        continue
    print(f"\n== {nm}: 창 {r['n_windows']} · 칸 {r['n_cells']}")
    for ver in ("tok", "pix"):
        e = r[ver]
        print(f"  [{ver}] " + " · ".join(f"{tg} {e[f'net_{tg}']['v']:+.4f} {e[f'net_{tg}']['ci']}" for tg in TARGETS))
        print(f"        모델/데이터 " + " · ".join(f"{tg} {e[f'ratio_{tg}']['v']:.2f} {e[f'ratio_{tg}']['ci']}" for tg in ("base", "vob", "syn", "ctxt")))
        print(f"        기준 대비 " + " · ".join(f"{tg} {e[f'diff_{tg}_minus_base']['v']:+.4f} {e[f'diff_{tg}_minus_base']['ci']}" for tg in ("vob", "syn", "ctxt")))
        b = e["bins"]
        print(f"        안정도 기울기 " + " · ".join(f"{tg} {b[f'slope_{tg}']['v']:+.4f} {b[f'slope_{tg}']['ci']}" for tg in ("data", "copy", "base", "ctxt")))
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
print("저장", a.out, f"{time.time() - t0:.0f}s")
