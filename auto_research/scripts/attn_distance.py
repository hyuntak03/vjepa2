#!/usr/bin/env python3
"""[E] B. attention 거리 병목 — 기준 블록 AR vs 합성 전용 블록 AR (auto_research, 2026-10-22). 학습 없음 · 체크포인트 읽기만.

  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/attn_distance.py

두 데이터:
  (A) 3D 합성 검증 세트 캐시 (블록 토큰 16 개/창): 한 걸음 TF (블록 0..t → t+1, t = 7..14), 두 블록 모두 보이는 걸음.
  (B) 통제 합성: 3D 합성 영상의 시간 중앙값 배경 + 물체 픽셀 (16 개). 블록마다 v 칸 (가로) 등속, 블록 안 두 장은 v/2 칸 간격.
      문맥 C 블록 → 블록 C 를 TF. v = 0 · 1 · 2 · 3 · 4 · 6 (C = 2), v = 0 · 1 · 2 · 3 (C = 4).
잴 것: 예측 블록 t+1 에서 물체가 새로 들어오는 칸 (블록 t 덮임 0, t+1 덮임 ≥ 0.5) 의 query (위치 t 의 그 칸) 가 준 attention 질량 (층별 · 헤드 평균):
  obj_last = 블록 t 의 물체 칸 · obj_prev = 블록 t−1 의 물체 칸 · self = 자기 칸 (블록 t). 균등 기준 = key 수 / 볼 수 있는 key 수.
  정지 (0 칸) 줄은 "블록 t 의 물체 칸 query → 같은 블록 물체 칸 key" 기준.
knockout: 들어오는 칸 query 가 obj_last key 를 못 보게 (모든 층) / 대조 = 같은 수의 먼 배경 key. 새 칸 존재감 = Σ⟨p − h_t, h_{t+1} − h_t⟩ / Σ|h_{t+1} − h_t|² (들어오는 칸).
SDPA 를 가로채 (src.models.utils.modules 의 PrefixSpec 경로: 블록마다 SDPA 한 번, key 길이 = (블록 + 1)·S) 지정한 query 행만 softmax 를 다시 계산한다."""
import argparse, glob, json, math, os, sys
import numpy as np, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--p3", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--models", nargs="+", default=["natural_ar_vw_state:latest.pt", "natural_ar_vw_state_phys3d_only:e10.pt"])
ap.add_argument("--n_obj", type=int, default=16)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/attn_distance.json"))
a = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                          # noqa: E402
sys.argv = _argv
from src.models.utils import modules as MOD                                      # noqa: E402
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda"); G, S, D, R = 16, 256, 1280, 256
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)

# ---------------- SDPA 가로채기
REC = {"on": False, "nb": 0, "cnt": 0, "q": {}, "keys": {}, "ko": {}, "mass": {}}
_sdpa = F.scaled_dot_product_attention


def sdpa(q, k, v, *args, **kw):
    out = _sdpa(q, k, v, *args, **kw)
    if not REC["on"] or kw.get("attn_mask") is not None:
        return out
    b = k.shape[2] // S - 1
    layer = REC["cnt"] // REC["nb"]; REC["cnt"] += 1
    need_rec, need_ko = b in REC["q"], b in REC["ko"]
    if not (need_rec or need_ko):
        return out
    rows = REC["ko"][b][0] if need_ko else REC["q"][b]
    qq = q[:, :, rows].float()
    lg = qq @ k.float().transpose(-1, -2) / math.sqrt(q.size(-1))               # (B, h, nq, Sk)
    if need_ko:
        lg[..., REC["ko"][b][1]] = float("-inf")
        w = torch.softmax(lg, -1)
        o = (w @ v.float()).to(out.dtype)
        out = out.clone(); out[:, :, rows] = o
    else:
        w = torch.softmax(lg, -1)
        wm = w.mean(1)[0]                                                        # (nq, Sk) 헤드 평균
        for name, kidx in REC["keys"][b].items():
            REC["mass"].setdefault((b, name), {})[layer] = float(wm[:, kidx].sum(-1).mean()) if len(kidx) else float("nan")
    return out


MOD.F.scaled_dot_product_attention = sdpa


def run(model, h, nb, q=None, keys=None, ko=None):
    """h (nb, S, D) → 예측 (nb, S, D) (위치 b = 블록 b+1). q/keys: 기록, ko: {b: (rows, keys)}."""
    REC.update(on=True, nb=nb, cnt=0, q=q or {}, keys=keys or {}, ko=ko or {}, mass={})
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = h.reshape(1, nb * S, D)
        p = model.forward_seq(x, torch.arange(nb * S, device=dev).unsqueeze(0)).float()[0].view(nb, S, D)
    REC["on"] = False
    return p, dict(REC["mass"])


def presence(p, ht, hn, cells):
    d = (hn - ht)[cells]; return float(((p - ht)[cells] * d).sum() / (d * d).sum().clamp_min(1e-9))


def cover(bbox):
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return (oy[..., :, None] * ox[..., None, :]).reshape(*bbox.shape[:-1], -1)


def far_cells(objmask, rng, n):
    m = objmask.reshape(G, G)
    yy, xx = np.where(m)
    far = np.ones((G, G), bool)
    if len(yy):
        far[max(yy.min() - 4, 0):yy.max() + 5, max(xx.min() - 4, 0):xx.max() + 5] = False
    cand = np.where(far.reshape(-1))[0]
    return rng.choice(cand, min(n, len(cand)), replace=False) if len(cand) else np.array([], int)


def dbin(d):
    for lo, hi, nm in ((0, 0.5, "0"), (0.5, 1.5, "1"), (1.5, 2.5, "2"), (2.5, 3.5, "3"), (3.5, 5.0, "4"), (5.0, 7.0, "6"), (7.0, 99, "8")):
        if lo <= d < hi:
            return nm


models = {}
for spec in a.models:
    run_, ck = spec.split(":")
    models[f"{run_}:{ck}"] = FT.load_pred(run_, ck).to(dev).eval()
rng = np.random.default_rng(0)
ACC = {}     # (데이터, 모델, 거리) → {"mass": {name: [ (12,) ...]}, "pres": [], "pres_ko": [], "pres_ctl": [], "uni": []}


def put(key, field, val):
    ACC.setdefault(key, {}).setdefault(field, []).append(val)


def analyse(tag, h, steps, mname, model):
    """h (nb, S, D) torch, steps: list of dict(t, E(rows idx), obj_last, obj_prev, d)."""
    nb = h.shape[0]
    q, keys = {}, {}
    for s in steps:
        q[s["t"]] = s["Q"]
        keys[s["t"]] = {"obj_last": [s["t"] * S + c for c in s["obj_last"]], "obj_prev": [(s["t"] - 1) * S + c for c in s["obj_prev"]],
                        "self": [s["t"] * S + c for c in s["Q"]]}
    p, mass = run(model, h, nb, q=q, keys=keys)
    for s in steps:
        key = (tag, mname, dbin(s["d"]))
        for nm in ("obj_last", "obj_prev", "self"):
            put(key, f"mass_{nm}", [mass.get((s["t"], nm), {}).get(l, float("nan")) for l in range(12)])
        put(key, "uni_obj_last", len(s["obj_last"]) / ((s["t"] + 1) * S))
        if s["moving"]:
            t = s["t"]; ht, hn = h[t], h[t + 1]
            pr0 = presence(p[t], ht, hn, s["Q"])
            pk, _ = run(model, h[:t + 1], t + 1, ko={t: (s["Q"], [t * S + c for c in s["obj_last"]])})
            ctl = far_cells(s["objmask"], rng, len(s["obj_last"]))
            pc, _ = run(model, h[:t + 1], t + 1, ko={t: (s["Q"], [t * S + c for c in ctl])})
            put(key, "pres", pr0); put(key, "pres_ko", presence(pk[t], ht, hn, s["Q"])); put(key, "pres_ctl", presence(pc[t], ht, hn, s["Q"]))


# ---------------- (A) 3D 합성 검증 세트 캐시
files = sorted(glob.glob(os.path.join(a.p3, "h_val_target_s*.npy")))
nwin = 0
for f in files:
    sh = f.split("_s")[-1].split(".")[0]
    lab = np.load(os.path.join(a.p3, f"lab_val_s{sh}.npz")); Hc = np.load(f, mmap_mode="r")
    for w in range(Hc.shape[0]):
        if a.limit and nwin >= a.limit:
            break
        cv = cover(np.nan_to_num(lab["bbox"][w], nan=-100.0))                     # (32, 256)
        cvb = np.maximum(cv[0::2], cv[1::2])                                     # (16, 256)
        vis = lab["vis_frac"][w]; pos = lab["pos"][w]
        steps = []
        for t in range(7, 15):
            if not (vis[2 * t:2 * t + 4].min() > 0.7):
                continue
            pc, pn = pos[2 * t:2 * t + 2].mean(0), pos[2 * t + 2:2 * t + 4].mean(0)
            if not (np.isfinite(pc).all() and np.isfinite(pn).all()):
                continue
            d = float(np.linalg.norm(pn - pc) / 16)
            obj_last = np.where(cvb[t] >= 0.5)[0]; obj_prev = np.where(cvb[t - 1] >= 0.5)[0]
            if len(obj_last) == 0:
                continue
            if d < 0.5:
                steps.append(dict(t=t, Q=list(obj_last), obj_last=list(obj_last), obj_prev=list(obj_prev), d=d, moving=False, objmask=cvb[t] > 0))
            else:
                E = np.where((cvb[t + 1] >= 0.5) & (cvb[t] == 0))[0]
                if len(E) == 0:
                    continue
                steps.append(dict(t=t, Q=list(E), obj_last=list(obj_last), obj_prev=list(obj_prev), d=d, moving=True, objmask=(cvb[t] > 0) | (cvb[t + 1] > 0)))
        if not steps:
            continue
        h = torch.from_numpy(np.asarray(Hc[w], np.float32)).to(dev)
        for mname, m in models.items():
            analyse("A", h, steps, mname, m)
        nwin += 1
        if nwin % 50 == 0:
            print(f"(A) 창 {nwin}", flush=True)

# ---------------- (B) 통제 합성 (등속 붙이기)
enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc, _load_checkpoint(CK)["target_encoder"], tag="t"); enc = enc.to(dev, torch.bfloat16).eval()


@torch.no_grad()
def encode(blocks):                                   # (nb, 2, R, R, 3) 0..255
    x = (blocks.div(255).permute(0, 4, 1, 2, 3) - MEAN) / STD
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hh = enc(x.to(torch.bfloat16)).float()
    return F.layer_norm(hh, (hh.size(-1),))


cand = []
for f in sorted(glob.glob(os.path.join(a.p3, "pix_val_s*.npy"))):
    sh = f.split("_s")[-1].split(".")[0]
    lab = np.load(os.path.join(a.p3, f"lab_val_s{sh}.npz"))
    for w in range(lab["bbox"].shape[0]):
        bb = lab["bbox"][w]
        if not np.isfinite(bb).all():
            continue
        cx = (bb[:, 0] + bb[:, 2]) / 2; cy = (bb[:, 1] + bb[:, 3]) / 2; size = (bb[:, 2] - bb[:, 0]).mean()
        if np.hypot(cx.max() - cx.min(), cy.max() - cy.min()) > 3 * size and 20 < size < 52:
            cand.append((f, w, sh))
rng.shuffle(cand)
objs = []
for f, w, sh in cand:
    if len(objs) >= a.n_obj:
        break
    lab = np.load(os.path.join(a.p3, f"lab_val_s{sh}.npz")); bb = lab["bbox"][w]
    fr = torch.from_numpy(np.asarray(np.load(f, mmap_mode="r")[w])).float().to(dev)
    Bg = fr.median(0).values
    ok = [i for i in range(32) if lab["vis_frac"][w][i] > 0.95 and bb[i, 1] > 40 and bb[i, 3] < R - 40 and bb[i, 0] > 8 and bb[i, 2] < R - 8]
    if not ok:
        continue
    i = ok[0]; x0, y0, x1, y1 = [int(v) for v in bb[i]]
    diff = (fr[i] - Bg).abs().mean(-1)
    M = torch.zeros(R, R, dtype=torch.bool, device=dev)
    M[y0 - 2:y1 + 3, max(x0 - 2, 0):x1 + 3] = diff[y0 - 2:y1 + 3, max(x0 - 2, 0):x1 + 3] > 6
    if M.sum() < 200:
        continue
    # 물체를 왼쪽 끝 (x 8 px) 으로 옮긴 마스크 · 픽셀
    ys, xs = torch.where(M); sx = 8 - int(xs.min())
    M0 = torch.roll(M, sx, 1); O0 = torch.roll(fr[i], sx, 1)
    objs.append((Bg, O0, M0, int(xs.max() - xs.min()) + 1))
print(f"(B) 물체 {len(objs)} 개", flush=True)


def comp(Bg, O, M, dx):
    Ms = torch.roll(M, dx, 1); out = Bg.clone(); out[Ms] = torch.roll(O, dx, 1)[Ms]
    return out


for (Bg, O, M, wpx) in objs:
    for C, vs in ((2, (0, 1, 2, 3, 4, 6)), (4, (0, 1, 2, 3))):
        for v in vs:
            if 8 + C * v * 16 + v * 8 + wpx > R:                                   # 예측 블록 (둘째 장) 까지 화면 안
                continue
            blocks = torch.stack([torch.stack([comp(Bg, O, M, b * v * 16), comp(Bg, O, M, b * v * 16 + v * 8)]) for b in range(C + 1)])
            h = encode(blocks)                                                  # (C+1, S, D)
            cvs = [torch.maximum(blk_m[0], blk_m[1]) for blk_m in
                   [torch.stack([torch.roll(M, b * v * 16, 1), torch.roll(M, b * v * 16 + v * 8, 1)]).float().view(2, G, 16, G, 16).mean(dim=(2, 4)).reshape(2, -1)
                    for b in range(C + 1)]]
            cvb = torch.stack(cvs).cpu().numpy()                               # (C+1, 256)
            t = C - 1
            obj_last = np.where(cvb[t] >= 0.5)[0]; obj_prev = np.where(cvb[t - 1] >= 0.5)[0]
            if v == 0:
                st = dict(t=t, Q=list(obj_last), obj_last=list(obj_last), obj_prev=list(obj_prev), d=0.0, moving=False, objmask=cvb[t] > 0)
            else:
                E = np.where((cvb[t + 1] >= 0.5) & (cvb[t] == 0))[0]
                if len(E) == 0 or len(obj_last) == 0:
                    continue
                st = dict(t=t, Q=list(E), obj_last=list(obj_last), obj_prev=list(obj_prev), d=float(v), moving=True, objmask=(cvb[t] > 0) | (cvb[t + 1] > 0))
            for mname, m in models.items():
                analyse(f"B_C{C}", h, [st], mname, m)

# ---------------- 요약
res = {}
for (tag, mname, b), d in sorted(ACC.items(), key=lambda kv: (kv[0][0], kv[0][1], "0123468".index(kv[0][2]))):
    r = {"n": len(d["mass_obj_last"]), "uni": float(np.mean(d["uni_obj_last"]))}
    for nm in ("obj_last", "obj_prev", "self"):
        arr = np.array(d[f"mass_{nm}"])                                       # (n, 12)
        r[f"{nm}_by_layer"] = np.nanmean(arr, 0).tolist()
    ol = np.array(r["obj_last_by_layer"])
    r["obj_last_max"] = float(np.nanmax(ol)); r["obj_last_argmax_layer"] = int(np.nanargmax(ol) + 1)
    r["obj_last_max_over_uni"] = r["obj_last_max"] / r["uni"]
    r["obj_last_mean_over_uni"] = float(np.nanmean(ol)) / r["uni"]
    if "pres" in d:
        for k in ("pres", "pres_ko", "pres_ctl"):
            r[k] = float(np.mean(d[k]))
        r["ko_drop"] = 1 - r["pres_ko"] / r["pres"] if r["pres"] > 1e-6 else None
        r["ctl_drop"] = 1 - r["pres_ctl"] / r["pres"] if r["pres"] > 1e-6 else None
    res[f"{tag}|{mname}|{b}"] = r
json.dump(res, open(a.out, "w"), indent=1, default=float)
print(f"\n{'데이터|모델|거리':62s} {'n':>4s} {'균등':>6s} {'obj_last 최대 (층)':>16s} {'÷균등':>6s} {'평균÷균등':>8s} {'obj_prev 최대':>12s} {'self 최대':>9s} {'존재감':>7s} {'막음':>6s} {'대조':>6s}")
for k, r in res.items():
    pr = f"{r['pres']:7.3f} {r['pres_ko']:6.3f} {r['pres_ctl']:6.3f}" if "pres" in r else "      —      —      —"
    print(f"{k:62s} {r['n']:4d} {r['uni']:6.4f} {r['obj_last_max']:9.4f} ({r['obj_last_argmax_layer']:2d}) {r['obj_last_max_over_uni']:6.1f} {r['obj_last_mean_over_uni']:8.1f} "
          f"{np.nanmax(r['obj_prev_by_layer']):12.4f} {np.nanmax(r['self_by_layer']):9.4f} {pr}")
print("→", a.out)
