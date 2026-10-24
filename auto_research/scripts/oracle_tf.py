#!/usr/bin/env python3
"""[P] 방법 B 상한 (학습 없음) — TF 한 걸음 오라클: m ⊙ (정답 변위로 옮긴 블록 t) + (1 − m) ⊙ 기준 AR 예측 (auto_research, 2026-10-21).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/oracle_tf.py

(1) 3D 합성 검증 세트 (내 캐시, 라벨): t = 7..14 움직인 쌍 (이동 > 8 px, 두 블록 보임). 변위 = 라벨 물체 이동 (블록 t+1 둘째 장 − 블록 t 둘째 장), m = 블록 t+1 물체 칸 (덮임 ≥ 0.5).
    예측기: 기준 AR · 오라클 반올림 (정수 칸 이동) · 오라클 쌍선형 · 정답. 지표: 운반 비 (새 칸 [8 지연 + 옮긴 복사] 회귀의 옮긴 복사 계수 / 정답의 계수),
    새 칸 존재감 · 떠난 칸 잔상 (선형 템플릿 점수 비), 운반 이득 (transport_gain.py 읽기, TF), 칸 L1 (전체 · 새 칸).
(4) 자연 보류: K710 보류 (natural_holdout, RAFT 칸 흐름) · HT100M ([E] 캐시 + natural_transport_ctx5 흐름), t = 4 · 5 · 6. 변위 = −U[t+1] 역방향 (내용 출처 = c + U[t+1](c)),
    m = |U[t+1]| ≥ 0.5 칸. TF L1 전체 · 움직이는 칸 (기준 vs 오라클 두 판)."""
import glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml

torch.set_num_threads(8)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _clean_backbone_key                       # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                 # noqa: E402
dev = torch.device("cuda")
S, G, C, NB, L = 256, 16, 8, 16, 8
pc = yaml.safe_load(open(os.path.join(REPO, "z_training/runs/natural_ar_vw_state/config.yaml")))["model"]["predictor"]
model = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
                             num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
model.load_state_dict({k: v for k, v in _clean_backbone_key(torch.load(os.path.join(REPO, "z_training/runs/natural_ar_vw_state/latest.pt"), map_location="cpu", weights_only=False)["predictor"]).items()
                       if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}); model = model.to(dev).eval()
gy, gx = torch.meshgrid(torch.arange(G, device=dev).float(), torch.arange(G, device=dev).float(), indexing="ij")
gy, gx = gy.reshape(S), gx.reshape(S)


def warp(hsrc, dyx, mode):
    """hsrc (S, D). dyx (S, 2) = 출처 칸 오프셋 (dy, dx): 출력(c) = hsrc(c + dyx(c)). 화면 밖은 제자리."""
    py, px = gy + dyx[:, 0], gx + dyx[:, 1]
    if mode == "round":
        ry, rx = torch.round(py).long(), torch.round(px).long()
        ok = (ry >= 0) & (ry < G) & (rx >= 0) & (rx < G)
        j = ry.clamp(0, G - 1) * G + rx.clamp(0, G - 1)
        out = hsrc[j]; out[~ok] = hsrc[~ok]
        return out
    y0, x0 = torch.floor(py), torch.floor(px); wy, wx = py - y0, px - x0
    out = torch.zeros_like(hsrc); wsum = torch.zeros(S, device=dev)
    for dy, wyy in ((0, 1 - wy), (1, wy)):
        for dx, wxx in ((0, 1 - wx), (1, wx)):
            yy, xx = (y0 + dy).long(), (x0 + dx).long()
            ok = (yy >= 0) & (yy < G) & (xx >= 0) & (xx < G)
            w = wyy * wxx * ok
            out += w[:, None] * hsrc[yy.clamp(0, G - 1) * G + xx.clamp(0, G - 1)]; wsum += w
    out = torch.where(wsum[:, None] > 1e-6, out / wsum.clamp_min(1e-6)[:, None], hsrc)
    return out


def cover(bbox):
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return oy[..., :, None] * ox[..., None, :]


def dil(m, k=1):
    m = m.reshape(G, G).copy()
    for _ in range(k):
        o = m.copy()
        o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
        o[1:, 1:] |= m[:-1, :-1]; o[:-1, :-1] |= m[1:, 1:]; o[1:, :-1] |= m[:-1, 1:]; o[:-1, 1:] |= m[1:, :-1]
        m = o
    return m.reshape(-1)


cells3 = lambda cx_, cy_: [y * G + x for y in range(max(0, cy_ - 1), min(G, cy_ + 2)) for x in range(max(0, cx_ - 1), min(G, cx_ + 2))]
cell_of = lambda p: (int(np.clip(p[0] // 16, 0, G - 1)), int(np.clip(p[1] // 16, 0, G - 1)))
yyc, xxc = np.divmod(np.arange(S), G); cen = np.stack([xxc * 16 + 8, yyc * 16 + 8], 1).astype(np.float32)


def readout(blk, tmpl):
    cs = (F.normalize(blk.float(), dim=-1) @ tmpl).cpu().numpy()
    j = int(cs.argmax()); nb = cells3(j % G, j // G)
    w = np.clip(cs[nb], 0, None) + 1e-6
    return (cen[nb] * w[:, None]).sum(0) / w.sum()


NAMES = ["base", "oracle_round", "oracle_bilin", "truth"]
REG = {n: [np.zeros((9, 9)), np.zeros(9)] for n in NAMES}; REGT = [np.zeros((9, 9)), np.zeros(9)]
PRES = {n: {"new": [], "old": []} for n in NAMES}; PRES_REF = {"new": [], "old": []}
L1 = {n: {"all": [], "new": []} for n in NAMES}
GAIN = {n: [] for n in NAMES}
for f in sorted(glob.glob("/workspace/auto_research_cache/phys3d/h_val_target_s*.npy")):
    sh = f.split("_s")[-1].split(".")[0]
    X = np.load(f, mmap_mode="r"); lab = np.load(f"/workspace/auto_research_cache/phys3d/lab_val_s{sh}.npz")
    for w in range(X.shape[0]):
        cv = cover(np.nan_to_num(lab["bbox"][w], nan=-100.0)); cmax = np.maximum(cv[0::2], cv[1::2]).reshape(NB, S)
        objb = cmax >= 0.25; touch = cmax > 0
        ok_ = lab["inframe_frac"][w] > 0.9
        visb = (lab["vis_frac"][w][0::2] > 0.7) & (lab["vis_frac"][w][1::2] > 0.7) & ok_[0::2] & ok_[1::2]
        far = ~dil(touch.any(0), 2); pos = lab["pos"][w]
        ts = [t for t in range(C - 1, NB - 1) if visb[t] and visb[t + 1] and np.isfinite(pos[[2 * t + 1, 2 * t + 3]]).all()
              and np.linalg.norm(pos[2 * t + 3] - pos[2 * t + 1]) > 8 and far.sum() >= 10]
        if not ts:
            continue
        h = torch.from_numpy(np.asarray(X[w], dtype=np.float32)).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p_tf = model.forward_seq(h[:NB - 1].reshape(1, -1, h.size(-1)), torch.arange((NB - 1) * S, device=dev).unsqueeze(0)).float()[0].view(NB - 1, S, -1)
        farm = torch.from_numpy(far).to(dev)
        for t in ts:
            d = pos[2 * t + 1] - pos[2 * t + 3]                                           # 출처 오프셋 (x, y) px = 물체 이동의 반대
            dyx = torch.tensor([d[1] / 16, d[0] / 16], device=dev, dtype=torch.float32).expand(S, 2)
            arr = torch.from_numpy(cmax[t + 1] >= 0.5).to(dev)
            P = {"base": p_tf[t], "truth": h[t + 1]}
            for mode in ("round", "bilin"):
                o = p_tf[t].clone(); o[arr] = warp(h[t], dyx, mode)[arr]; P[f"oracle_{mode}"] = o
            new = objb[t + 1] & ~objb[t]; old = objb[t] & ~objb[t + 1]
            if not (new.any() and old.any()):
                continue
            nm, om = torch.from_numpy(new).to(dev), torch.from_numpy(old).to(dev)
            mc = torch.roll(h[t].view(G, G, -1), shifts=(int(round(-d[1] / 16)), int(round(-d[0] / 16))), dims=(0, 1)).reshape(S, -1)
            Xr = torch.stack([h[t - k][nm] for k in range(L)] + [mc[nm]])               # (9, n, D)
            XtX = torch.einsum("knd,lnd->kl", Xr, Xr).double().cpu().numpy()
            T_hat = F.normalize(h[t][torch.from_numpy(objb[t]).to(dev)].mean(0), dim=-1)
            sc = lambda x: (x @ T_hat) - (x @ T_hat)[farm].mean()
            s_true, s_copy = sc(h[t + 1]), sc(h[t])
            PRES_REF["new"].append((float(s_true[nm].mean()), float(s_copy[nm].mean()))); PRES_REF["old"].append((float(s_true[om].mean()), float(s_copy[om].mean())))
            tmpl = F.normalize(h[t][cells3(*cell_of(pos[2 * t + 1]))].mean(0), dim=-1)
            for n, o in P.items():
                REG[n][0] += XtX; REG[n][1] += torch.einsum("knd,nd->k", Xr, o[nm]).double().cpu().numpy()
                so = sc(o)
                PRES[n]["new"].append(float(so[nm].mean())); PRES[n]["old"].append(float(so[om].mean()))
                L1[n]["all"].append(float((o - h[t + 1]).abs().mean())); L1[n]["new"].append(float((o[nm] - h[t + 1][nm]).abs().mean()))
                q = readout(o, tmpl); dtr = pos[2 * t + 3] - pos[2 * t + 1]
                if np.linalg.norm(dtr) > 16:
                    GAIN[n].append(float(((q - pos[2 * t + 1]) * dtr).sum() / (dtr ** 2).sum()))
    print(f, flush=True)

res = {"phys3d": {}}
lam = lambda A: 1e-6 * np.trace(A) / 9 * np.eye(9)
bt = np.linalg.solve(REG["truth"][0] + lam(REG["truth"][0]), REG["truth"][1])[L]
rn = np.array(PRES_REF["new"]); ro = np.array(PRES_REF["old"])
print(f"\n== 3D 합성 TF 한 걸음 (움직인 쌍 {len(L1['base']['all'])}, 운반 이득 n {len(GAIN['base'])})")
for n in NAMES:
    b = np.linalg.solve(REG[n][0] + lam(REG[n][0]), REG[n][1])[L]
    pn, po = np.array(PRES[n]["new"]), np.array(PRES[n]["old"])
    pres = (pn.mean() - rn[:, 1].mean()) / (rn[:, 0].mean() - rn[:, 1].mean()); trail = (po.mean() - ro[:, 0].mean()) / (ro[:, 1].mean() - ro[:, 0].mean())
    e = dict(beta_ratio=float(b / bt), presence_new=float(pres), trail_old=float(trail), gain_median=float(np.median(GAIN[n])),
             l1_all=float(np.mean(L1[n]["all"])), l1_new=float(np.mean(L1[n]["new"])))
    res["phys3d"][n] = e
    print(f"  {n:13s} 운반 비 {e['beta_ratio']:.2f} · 새 칸 존재감 {e['presence_new']:.2f} · 떠난 칸 잔상 {e['trail_old']:.2f} · 운반 이득 {e['gain_median']:.2f} · L1 전체 {e['l1_all']:.4f} · 새 칸 {e['l1_new']:.4f}")

# ---------------- (4) 자연 보류
G_, P0 = G, None
cyx = np.divmod(np.arange(S), G); P0 = np.stack([cyx[1] + 0.5, cyx[0] + 0.5], 1)


def nat_windows(src):
    if src == "k710_holdout":
        H = np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_h.npy", mmap_mode="r"); U = np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_U.npy", mmap_mode="r")
        for w in range(H.shape[0]):
            yield np.asarray(H[w], np.float32), np.asarray(U[w], np.float32).reshape(8, S, 2)
    else:
        Uw = np.load(os.path.join(REPO, f"auto_research/results/natural_transport_ctx5/rows_{src}.npz"))["Uwin"].astype(np.float32); w0 = 0
        for sh in (0, 1):
            T_ = np.load(f"/workspace/auto_research_cache/enc_noise/{src}_target_encoder_s{sh}_tok.npy", mmap_mode="r")
            for w in range(T_.shape[0]):
                yield np.asarray(T_[w, 0], np.float32), Uw[w0 + w].reshape(8, S, 2)
            w0 += T_.shape[0]


for src in ("k710_holdout", "ht100m"):
    acc = {n: {"all": [], "mov": []} for n in ("base", "oracle_round", "oracle_bilin", "copy")}
    for hn, U in nat_windows(src):
        h = torch.from_numpy(hn).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = model.forward_seq(h[:7].reshape(1, -1, 1280), torch.arange(7 * S, device=dev).unsqueeze(0)).float()[0].view(7, S, -1)
        for t in (4, 5, 6):
            u = torch.from_numpy(U[t + 1]).to(dev)                                          # (S, 2) (dx, dy) 칸: 출처 = c + u
            mov = u.norm(dim=-1) >= 0.5
            if mov.sum() == 0:
                continue
            dyx = torch.stack([u[:, 1], u[:, 0]], -1)
            P = {"base": p[t], "copy": h[t]}
            for mode in ("round", "bilin"):
                o = p[t].clone(); o[mov] = warp(h[t], dyx, mode)[mov]; P[f"oracle_{mode}"] = o
            for n, o in P.items():
                e = (o - h[t + 1]).abs().mean(-1)
                acc[n]["all"].append(float(e.mean())); acc[n]["mov"].append(float(e[mov].mean()))
    res[src] = {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in acc.items()}
    print(f"\n== {src} TF 한 걸음 L1 (전체 · 움직이는 칸 = 흐름 ≥ 0.5 칸)")
    for n, d in res[src].items():
        print(f"  {n:13s} 전체 {d['all']:.4f} · 움직이는 칸 {d['mov']:.4f}  (기준 대비 {100 * (d['all'] / res[src]['base']['all'] - 1):+.1f} % · {100 * (d['mov'] / res[src]['base']['mov'] - 1):+.1f} %)")
json.dump(res, open(os.path.join(REPO, "auto_research/results/oracle_tf.json"), "w"), indent=1)
