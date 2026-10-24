#!/usr/bin/env python3
"""[T] 기존 lasthole 체크포인트 재평가 (학습 없음) — 사전 등록 진행판 [T] KST 20:11 (auto_research, 2026-10-22).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore auto_research/scripts/lasthole_eval.py [--parts stop,transport,vob,ip2] [--limit N]

모델 (읽기만): L = natural_ar_vw_state_ctxt_video_c4b_lasthole latest (epoch 5) · C6 = natural_ar_vw_state_ctxt_video e6 (epoch 6, 짝 대조)
  · C4 = 같은 run e4 (대조의 학습 단계 흔들림) · base = 블록 AR natural_ar_vw_state latest (참고).
영상 문맥 경로 ([비판] crit_pmi_ctxt · [P] placebo_b 와 같다): z = online encoder 영상 모드 (문맥 장, 마스크 = 문맥 전부) →
  한 걸음 = forward_state(z, 관측 블록) → render, 굴림 = rollout_state_kv(z) → render. z 는 세 영상 문맥 모델이 공유한다.
(1) stop: K710 보류 · SSv2 (pmi_cont 캐시, 블록 10) — 문맥 4 블록 한 걸음. B′ 참 다음 vs 다시 인코딩한 멈춤 (h'_t) · 뒤로 · 건너뜀 승률, C′ 8 후보 순위,
    복사 연속 판정자. 다시 인코딩 잡음은 모델을 다 지은 뒤 seed 0 (crit_pmi_fresh 와 같은 식, ±1/255 장마다).
(2) transport: SSv2 수동 · 행위자 (ssv2_classes) · K710 보류 (natural_holdout), 블록 8. 영상 문맥 = z(블록 0–3) + 관측 블록 4–6 → 블록 5–7,
    블록 AR = TF. 위약 뺀 운반 (transport_placebo 와 같은 회귀, t = 4 · 5 · 6). 복사 정답 검사.
(3) vob: VOBench 끝 가림 216 block — 문맥 8 블록 (16 장) → 8 블록 굴림. 화면 · 물체 자리 (라벨 중심 3×3) 토큰 L1, 쌍 정확도 (움직임 · 정지 × 위반 종류).
(4) ip2: IntPhys 2 Main ([E] 캐시: 블록 토큰 32 · 6 fps 픽셀 64) — 영상 문맥 = z(블록 0–3) + 관측 블록 4–30 → 블록 4–31 예측, 블록 AR = TF.
    surprise = 블록 5–31 평균 L1. 표준 보고 = [비판] 분해 (정확도 · 문맥 항, 장면 bootstrap), 조건별.
모델 짝 차이: 창 · 장면 bootstrap 으로 L − C6, C6 − C4. 출력 results/lasthole_eval.json."""
import argparse, collections, json, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F, yaml, decord
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--parts", default="stop,transport,vob,ip2")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/lasthole_eval.json"))
a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                     # noqa: E402
sys.argv = _argv
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from app.vjepa_frozen import utils as UU                                     # noqa: E402
from app.vjepa_frozen import ar as AR                                        # noqa: E402
from crit_ip2_pmi import decomp                                              # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda")
S, D, G = 256, 1280, 16
PARTS = set(a.parts.split(","))
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
sd = _load_checkpoint(CK)
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, sd["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()
onl = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=32, use_rope=True, uniform_power=False)
_load_state_dict(onl, sd["encoder"], tag="o"); onl = onl.to(dev, torch.bfloat16).eval()
del sd
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)


def load_ctx(run, ck):
    Dd = os.path.join(REPO, "z_training/runs", run)
    cfg = yaml.safe_load(open(os.path.join(Dd, "config.yaml")))
    m = UU.build_predictor(cfg["model"], 1280, None, torch.device("cpu"), int(cfg["data"].get("n_frames", 64)))
    m.load_state_dict({k: v for k, v in _clean_backbone_key(torch.load(os.path.join(Dd, ck), map_location="cpu", weights_only=False)["predictor"]).items()
                       if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}, strict=True)
    return m.to(dev).eval()


CTX = {"L": load_ctx("natural_ar_vw_state_ctxt_video_c4b_lasthole", "latest.pt"), "C6": load_ctx("natural_ar_vw_state_ctxt_video", "e6.pt"),
       "C4": load_ctx("natural_ar_vw_state_ctxt_video", "e4.pt")}
base = FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval()
MODELS = list(CTX) + ["base"]
torch.manual_seed(0)
EMPTY = torch.zeros(1, 0, 384, device=dev)


def norm(x):                                                                 # x (T,3,256,256) 0..255 → (1,3,T,256,256)
    return ((x / 255).permute(1, 0, 2, 3)[None] - MEAN) / STD


@torch.no_grad()
def zctx(xn_ctx):                                                            # (1,3,2C,256,256) → z (1, C·S, D)
    C = xn_ctx.shape[2] // 2
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return onl(xn_ctx.to(torch.bfloat16), masks=[torch.arange(C * S, device=dev).unsqueeze(0)]).float()


@torch.no_grad()
def ctx_onestep(m, z, C, h_obs=None):                                        # 관측 블록 F 개 → 블록 C..C+F 예측 (F+1, S, D)
    F_ = 0 if h_obs is None else h_obs.shape[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fut = EMPTY if F_ == 0 else m.embed_obs(h_obs.reshape(1, F_ * S, D))
        st_ = m.forward_state(z, fut, torch.arange((C + F_) * S, device=dev).unsqueeze(0), C)
        return m.render(st_).float()[0].view(F_ + 1, S, D)


@torch.no_grad()
def ctx_roll(m, z, C, K):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        st_ = m.rollout_state_kv(z, torch.arange(C * S, device=dev).unsqueeze(0), K)
        return m.render(st_).float()[0].view(K, S, D)


@torch.no_grad()
def base_tf(h):                                                              # (nb,S,D) → (nb−1,S,D) 위치 b → 블록 b+1
    nb = h.shape[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return base.forward_seq(h.reshape(1, nb * S, D), torch.arange(nb * S, device=dev).unsqueeze(0)).float()[0].view(nb, S, D)[:-1]


@torch.no_grad()
def base_roll(hc, K):
    C = hc.shape[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return base.rollout_kv(hc.reshape(1, C * S, D), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, D)


def decode(meta_row, nf):
    vr = decord.VideoReader(meta_row["path"], num_threads=1)
    idx = [meta_row["start"] + meta_row["fstp"] * k for k in range(nf)]
    if idx[-1] >= len(vr):
        return None
    fr = vr.get_batch(idx).asnumpy(); Hh, Ww = fr.shape[1:3]; s_ = min(Hh, Ww); y0, x0 = (Hh - s_) // 2, (Ww - s_) // 2
    x = torch.from_numpy(fr[:, y0:y0 + s_, x0:x0 + s_]).to(dev).permute(0, 3, 1, 2).float()
    return F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round()


def boot(x, rng, n=2000):
    x = np.asarray(x, float); m_ = len(x)
    bs = np.sort([x[rng.integers(0, m_, m_)].mean() for _ in range(n)])
    return [float(x.mean()), float(bs[int(0.025 * n)]), float(bs[int(0.975 * n) - 1])]


l1 = lambda p, q: float((p - q).abs().mean())
RES = json.load(open(a.out)) if os.path.exists(a.out) and not a.limit else {}
t0 = time.time()

# ---------------------------------------------------------------- (1) 멈춤 편향
if "stop" in PARTS:
    B_T = [3, 4, 5, 6]; C_T = [3, 4, 5, 6, 7, 8]; NB = 10
    PC = "/workspace/auto_research_cache/pmi_cont"
    out = {}
    for st in ("k710_320_holdout", "ssv2"):
        H = np.load(f"{PC}/{st}_h.npy", mmap_mode="r"); U = np.load(f"{PC}/{st}_U.npy"); meta = json.load(open(f"{PC}/{st}_meta.json"))
        N = len(meta) if not a.limit else a.limit
        med = np.median(U[:N, 4:10].astype(np.float32).reshape(N, 6, -1, 2), axis=2); still = (np.linalg.norm(med, axis=-1).max(1) < 0.25)
        crng = random.Random(1); rows = []
        for i in range(N):
            x = decode(meta[i], 2 * NB)
            h = torch.from_numpy(np.asarray(H[i], np.float32)).to(dev)
            xr = (x + torch.randint(0, 2, x.shape, device=dev).float() * 2 - 1).clamp(0, 255)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hn = AR.encode_blocks(tgt, norm(xr).to(torch.bfloat16), 2)[0].view(NB, S, -1).float()
            xn = norm(x)
            P = {mk: [] for mk in MODELS}
            for c in range(6):
                z = zctx(xn[:, :, 2 * c:2 * (c + 4)])
                for mk, m in CTX.items():
                    P[mk].append(ctx_onestep(m, z, 4)[0])
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):        # 마지막 위치 = 블록 c+4 예측 (base_tf 는 마지막 위치를 버리므로 쓰지 않음)
                    P["base"].append(base.forward_seq(h[c:c + 4].reshape(1, 4 * S, D), torch.arange(4 * S, device=dev).unsqueeze(0)).float()[0, -S:])
            cands = {t: sorted([t + 1] + crng.sample([k for k in range(NB) if k != t + 1], 7)) for t in C_T}
            r = {"still": bool(still[i])}
            for mk in MODELS:
                for t in B_T:
                    p = P[mk][t - 3]
                    r[f"B|{mk}|{t}|true"] = l1(p, hn[t + 1])
                    for nm, k in (("h_t", t), ("h_tm1", t - 1), ("h_tp2", t + 2)):
                        r[f"B|{mk}|{t}|{nm}"] = l1(p, hn[k])
                for t in C_T:
                    p = P[mk][t - 3]; d = {k: l1(p, hn[k]) for k in cands[t]}
                    r[f"C|{mk}|{t}|pick"] = int(min(d, key=d.get) - (t + 1))
            for t in C_T:
                dh = {k: l1(hn[k], h[t]) for k in cands[t]}
                r[f"C|copycont|{t}|pick"] = int(min(dh, key=dh.get) - (t + 1))
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"  stop {st} {i + 1}/{N} ({time.time() - t0:.0f}s)", flush=True)
        brng = np.random.default_rng(0); o = {"n": N, "n_still": int(still.sum())}
        win = lambda rr, mk, neg: np.mean([float(rr[f"B|{mk}|{t}|true"] < rr[f"B|{mk}|{t}|{neg}"]) for t in B_T])
        acc = lambda rr, mk: np.mean([float(rr[f"C|{mk}|{t}|pick"] == 0) for t in C_T])
        for sub, sel in (("all", [True] * len(rows)), ("still", [r_["still"] for r_ in rows])):
            rs = [r_ for r_, s_ in zip(rows, sel) if s_]
            if not rs:
                continue
            oo = {}
            for mk in MODELS:
                for neg in ("h_t", "h_tm1", "h_tp2"):
                    oo[f"B|{mk}|{neg}"] = boot([win(r_, mk, neg) for r_ in rs], brng)
                oo[f"C|{mk}"] = boot([acc(r_, mk) for r_ in rs], brng)
                oo[f"C|{mk}|minus_copycont"] = boot([acc(r_, mk) - acc(r_, "copycont") for r_ in rs], brng)
            oo["C|copycont"] = boot([acc(r_, "copycont") for r_ in rs], brng)
            for a_, b_ in (("L", "C6"), ("C6", "C4")):
                oo[f"B|{a_}-{b_}|h_t"] = boot([win(r_, a_, "h_t") - win(r_, b_, "h_t") for r_ in rs], brng)
                oo[f"C|{a_}-{b_}"] = boot([acc(r_, a_) - acc(r_, b_) for r_ in rs], brng)
            o[sub] = oo
        out[st] = o
        oa = o["all"]
        print(f"== stop {st} (창 {N}): 참 < 멈춤 (다시 인코딩) " + " · ".join(f"{mk} {100 * oa[f'B|{mk}|h_t'][0]:.1f}" for mk in MODELS)
              + f" | 8 후보 순위 " + " · ".join(f"{mk} {100 * oa[f'C|{mk}'][0]:.1f}" for mk in MODELS) + f" · 복사 연속 {100 * oa['C|copycont'][0]:.1f}"
              + f" | L − C6: 멈춤 승률 {100 * oa['B|L-C6|h_t'][0]:+.1f} [{100 * oa['B|L-C6|h_t'][1]:+.1f}, {100 * oa['B|L-C6|h_t'][2]:+.1f}] · 순위 {100 * oa['C|L-C6'][0]:+.1f} [{100 * oa['C|L-C6'][1]:+.1f}, {100 * oa['C|L-C6'][2]:+.1f}]"
              + f" | C6 − C4: 멈춤 {100 * oa['B|C6-C4|h_t'][0]:+.1f} · 순위 {100 * oa['C|C6-C4'][0]:+.1f}", flush=True)
    RES["stop"] = out

# ---------------------------------------------------------------- (2) 위약 뺀 운반
if "transport" in PARTS:
    R90 = np.array([[0.0, -1.0], [1.0, 0.0]], np.float32)
    cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)

    def warp(ht, u):
        s_ = np.floor(P0 + u).astype(int); ok = (s_ >= 0).all(1) & (s_ < G).all(1)
        j = np.clip(s_[:, 1], 0, G - 1) * G + np.clip(s_[:, 0], 0, G - 1)
        o_ = ht[j].copy(); o_[~ok] = ht[~ok]; return o_
    SETS = {"ssv2_passive": "/workspace/auto_research_cache/ssv2_classes/passive", "ssv2_actor": "/workspace/auto_research_cache/ssv2_classes/actor",
            "k710_holdout": "/workspace/auto_research_cache/natural_holdout/k710_320_holdout"}
    out = {}
    YS = ["data", "copy"] + MODELS
    for sn, pre in SETS.items():
        H = np.load(f"{pre}_h.npy", mmap_mode="r"); U = np.load(f"{pre}_U.npy").astype(np.float32).reshape(-1, 8, S, 2); meta = json.load(open(f"{pre}_meta.json"))
        N = H.shape[0] if not a.limit else min(a.limit, H.shape[0])
        recs, recon = [], []
        for w in range(N):
            if not any((np.linalg.norm(U[w, t + 1], axis=1) >= 0.5).sum() >= 2 for t in (4, 5, 6)):
                continue
            x = decode(meta[w], 16)
            if x is None:
                continue
            hh = torch.from_numpy(np.asarray(H[w], np.float32)).to(dev)
            if len(recon) < 5:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    hc = AR.encode_blocks(tgt, norm(x).to(torch.bfloat16), 2)[0].view(8, S, -1).float()
                recon.append(float((hc - hh).abs().mean() / hh.abs().mean()))
            z = zctx(norm(x)[:, :, :8])
            Pm = {mk: ctx_onestep(m, z, 4, hh[4:7]).cpu().numpy() for mk, m in CTX.items()}      # (4,S,D) = 블록 4..7
            Pb = base_tf(hh).cpu().numpy()                                                       # (7,S,D) 위치 t → t+1
            hn = hh.cpu().numpy()
            for t in (4, 5, 6):
                sel = np.linalg.norm(U[w, t + 1], axis=1) >= 0.5
                if sel.sum() < 2:
                    continue
                m_ = hn[t - 4:t + 1].mean(0)
                ys = {"data": hn[t + 1], "copy": hn[t], "base": Pb[t], **{mk: Pm[mk][t + 1 - 4] for mk in CTX}}
                r = {"win": w}
                for kind in ("true", "rot"):
                    Wk = warp(hn[t], U[w, t + 1] if kind == "true" else U[w, t + 1] @ R90.T)
                    X = np.stack([m_[sel].ravel(), Wk[sel].ravel()], 1).astype(np.float64)
                    r[f"XtX_{kind}"] = X.T @ X
                    for nm, yy in ys.items():
                        r[f"Xty_{kind}_{nm}"] = X.T @ yy[sel].ravel().astype(np.float64)
                recs.append(r)
            if (w + 1) % 100 == 0:
                print(f"  transport {sn} {w + 1}/{N} ({time.time() - t0:.0f}s)", flush=True)

        def bfit(rr, kind, nm):
            A_ = sum(r_[f"XtX_{kind}"] for r_ in rr); bv = sum(r_[f"Xty_{kind}_{nm}"] for r_ in rr)
            return np.linalg.solve(A_, bv)[1]
        rng = np.random.RandomState(0)
        wins = np.array([r_["win"] for r_ in recs]); uw = np.unique(wins); ix = {u: np.where(wins == u)[0] for u in uw}
        boots = [np.concatenate([ix[u] for u in rng.choice(uw, len(uw))]) for _ in range(300)]
        e = {"n_windows": int(len(uw)), "recon_rel": float(np.mean(recon)) if recon else None}
        BS = {}
        for nm in YS:
            bt, br = bfit(recs, "true", nm), bfit(recs, "rot", nm)
            BS[nm] = np.array([bfit([recs[i] for i in ii], "true", nm) - bfit([recs[i] for i in ii], "rot", nm) for ii in boots])
            e[nm] = dict(b_true=float(bt), b_rot=float(br), net=float(bt - br), ci=np.percentile(BS[nm], [2.5, 97.5]).tolist())
        for a_, b_ in (("L", "C6"), ("C6", "C4"), ("C6", "base")):
            d_ = BS[a_] - BS[b_]; e[f"{a_}-{b_}"] = dict(diff=float(e[a_]["net"] - e[b_]["net"]), ci=np.percentile(d_, [2.5, 97.5]).tolist())
        out[sn] = e
        print(f"== transport {sn} (창 {e['n_windows']}, 캐시 재현 {e['recon_rel']}): " + " · ".join(f"{nm} {e[nm]['b_true']:.3f}/{e[nm]['b_rot']:.3f}→{e[nm]['net']:+.4f}" for nm in YS)
              + f" | L − C6 {e['L-C6']['diff']:+.4f} {np.round(e['L-C6']['ci'], 4).tolist()} · C6 − C4 {e['C6-C4']['diff']:+.4f}", flush=True)
    RES["transport"] = out

# ---------------------------------------------------------------- (3) VOBench 끝 가림
if "vob" in PARTS:
    VR = "/workspace/dataset/benchmarks/VOBench_v1"; HV = np.load("/workspace/auto_research_cache/vobench_late/h.npy", mmap_mode="r")
    MV = json.load(open("/workspace/auto_research_cache/vobench_late/meta.json"))
    IDX = list(range(0, 96, 3))[:32]
    MEANn = MEAN[0]; STDn = STD[0]

    def vframes(d, idx):
        fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in idx]
        x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 255.0
        W0 = x.shape[-1]
        x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
        return ((x.to(dev) - MEANn) / STDn)[None], W0

    def centers(d, W0, key):
        L_ = json.load(open(os.path.join(d, "label.json")))["labels"]; lab = {q["f"]: q for q in L_.get(key, [])}; cs = []
        for b in range(16):
            q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
            cs.append(None if any(t is None for t in q) else (int(np.clip(np.mean([t["v"] for t in q]) / W0 * G, 0, G - 1)), int(np.clip(np.mean([t["u"] for t in q]) / W0 * G, 0, G - 1))))
        return cs
    res = []
    for m in MV[: a.limit or None]:
        d0 = os.path.join(VR, m["files"]["pos_a"]); xa, W0 = vframes(d0, IDX[:16])
        cen = {k: centers(os.path.dirname(d0), W0, k) for k in ("A", "B")}
        otok = []
        for j in range(8):
            s_ = set()
            for k in ("A", "B"):
                c_ = cen[k][8 + j]
                if c_ is not None:
                    s_ |= {rr * G + cc for rr in range(max(0, c_[0] - 1), min(G, c_[0] + 2)) for cc in range(max(0, c_[1] - 1), min(G, c_[1] + 2))}
            otok.append(sorted(s_))
        oidx = torch.tensor([j * S + t for j, tt in enumerate(otok) for t in tt], device=dev, dtype=torch.long)
        mv = "static" if m["motion"] == "flat_static" else "moving"
        for pi, ii, arm in ((0, 2, "pos_a"), (1, 3, "pos_b")):
            xp = xa if arm == "pos_a" else vframes(os.path.join(VR, m["files"][arm]), IDX[:16])[0]
            z = zctx(xp)
            hp = torch.from_numpy(np.asarray(HV[m["i"], pi], np.float32)).to(dev); hi = torch.from_numpy(np.asarray(HV[m["i"], ii], np.float32)).to(dev)
            preds = {mk: ctx_roll(mm, z, 8, 8) for mk, mm in CTX.items()}; preds["base"] = base_roll(hp[:8], 8)
            rec = {"mv": mv, "viol": m["violation"]}
            for mk, pr in preds.items():
                pr = pr.reshape(8 * S, D)
                e = {nm: (pr - hh[8:].reshape(8 * S, D)).abs().mean(-1) for nm, hh in (("pos", hp), ("imp", hi))}
                rec[f"{mk}|full"] = float(e["imp"].mean()) > float(e["pos"].mean())
                rec[f"{mk}|obj"] = (float(e["imp"][oidx].mean()) > float(e["pos"][oidx].mean())) if len(oidx) else None
            res.append(rec)
    out = {}
    brng = np.random.default_rng(0)
    for mv in ("moving", "static"):
        for vi in (None, "permanence", "shape", "color"):
            sel = [r_ for r_ in res if r_["mv"] == mv and (vi is None or r_["viol"] == vi)]
            if not sel:
                continue
            key = mv + ("" if vi is None else f"/{vi}")
            o = {"n": len(sel)}
            for mk in MODELS:
                for kk in ("obj", "full"):
                    v = [float(r_[f"{mk}|{kk}"]) for r_ in sel if r_[f"{mk}|{kk}"] is not None]
                    o[f"{mk}|{kk}"] = 100 * float(np.mean(v))
            for a_, b_ in (("L", "C6"), ("C6", "C4")):
                for kk in ("obj", "full"):
                    dd = [float(r_[f"{a_}|{kk}"]) - float(r_[f"{b_}|{kk}"]) for r_ in sel if r_[f"{a_}|{kk}"] is not None]
                    o[f"{a_}-{b_}|{kk}"] = [100 * x_ for x_ in boot(dd, brng)]
            out[key] = o
            print(f"== VOBench {key} (쌍 {o['n']}): 물체 자리 " + " · ".join(f"{mk} {o[f'{mk}|obj']:.1f}" for mk in MODELS)
                  + " | 화면 " + " · ".join(f"{mk} {o[f'{mk}|full']:.1f}" for mk in MODELS)
                  + f" | L − C6 물체 {o['L-C6|obj'][0]:+.1f} [{o['L-C6|obj'][1]:+.1f}, {o['L-C6|obj'][2]:+.1f}] · C6 − C4 {o['C6-C4|obj'][0]:+.1f}", flush=True)
    RES["vob"] = out

# ---------------------------------------------------------------- (4) IntPhys 2
if "ip2" in PARTS:
    C2 = "/workspace/auto_research_cache/intphys2/Main"
    M2 = json.load(open(f"{C2}/meta.json")); H2 = np.load(f"{C2}/h.npy", mmap_mode="r"); PX = np.load(f"{C2}/pix.npy", mmap_mode="r")
    rows = []
    for m in M2[: (4 * a.limit) or None]:
        h = torch.from_numpy(np.asarray(H2[m["row"]], np.float32)).to(dev)
        x = torch.from_numpy(np.asarray(PX[m["row"], :8])).to(dev).permute(0, 3, 1, 2).float()
        z = zctx(norm(x))
        r = dict(scene=m["scene_index"], pair=m["pair_id"], imp=bool(m["is_impossible"]), condition=m["condition"], camera=m["camera"])
        for mk, mm in CTX.items():
            p = ctx_onestep(mm, z, 4, h[4:31])                               # (28,S,D) = 블록 4..31
            r[mk] = float((p[1:] - h[5:]).abs().mean())
        r["base"] = float((base_tf(h)[4:] - h[5:]).abs().mean())
        rows.append(r)
        if len(rows) % 200 == 0:
            print(f"  ip2 {len(rows)}/{len(M2)} ({time.time() - t0:.0f}s)", flush=True)
    out = {}
    for mk in MODELS:
        o = {"all": decomp(rows, mk, True, random.Random(0))}
        for cv in sorted({r_["condition"] for r_ in rows}):
            o[cv] = decomp([r_ for r_ in rows if r_["condition"] == cv], mk, True, random.Random(0))
        out[mk] = o
    # 짝 차이 (장면): 문맥 항 i = ½(Δ1 + Δ2), Δ = 상대 차
    sc = collections.defaultdict(dict)
    for r_ in rows:
        sc[r_["scene"]][("1" if r_["pair"] == "1" else "2") + ("I" if r_["imp"] else "P")] = r_
    f_ = lambda x_, y_: (x_ - y_) / ((x_ + y_) / 2)
    brng = np.random.default_rng(0)
    for a_, b_ in (("L", "C6"), ("C6", "C4")):
        di, da = [], []
        for s_, v in sc.items():
            if len(v) != 4:
                continue
            ia = 0.5 * (f_(v["1I"][a_], v["1P"][a_]) + f_(v["2I"][a_], v["2P"][a_])); ib = 0.5 * (f_(v["1I"][b_], v["1P"][b_]) + f_(v["2I"][b_], v["2P"][b_]))
            aa = 0.5 * ((v["1I"][a_] > v["1P"][a_]) + (v["2I"][a_] > v["2P"][a_])); ab = 0.5 * ((v["1I"][b_] > v["1P"][b_]) + (v["2I"][b_] > v["2P"][b_]))
            di.append(ia - ib); da.append(aa - ab)
        out[f"{a_}-{b_}"] = {"ctx_term": boot(di, brng), "acc": [100 * x_ for x_ in boot(da, brng)]}
    RES["ip2"] = out
    for mk in MODELS:
        d_ = out[mk]["all"]
        print(f"== IntPhys 2 {mk}: 정확도 {d_['acc']:.1f} [{d_['acc_ci'][0]:.1f}, {d_['acc_ci'][1]:.1f}] · 결과 항 몫 {d_['outcome_share']:.2f} · 문맥 항 {d_['mean_i']:+.4f} [{d_['mean_i_ci'][0]:+.4f}, {d_['mean_i_ci'][1]:+.4f}]"
              + " | " + " · ".join(f"{cv} {out[mk][cv]['acc']:.1f}" for cv in out[mk] if cv != "all"), flush=True)
    for k in ("L-C6", "C6-C4"):
        print(f"   {k}: 정확도 차 {out[k]['acc'][0]:+.1f} [{out[k]['acc'][1]:+.1f}, {out[k]['acc'][2]:+.1f}] · 문맥 항 차 {out[k]['ctx_term'][0]:+.4f} [{out[k]['ctx_term'][1]:+.4f}, {out[k]['ctx_term'][2]:+.4f}]", flush=True)

json.dump(RES, open(a.out if not a.limit else a.out.replace(".json", "_smoke.json"), "w"), indent=1)
print(f"저장 ({time.time() - t0:.0f}s)", flush=True)
