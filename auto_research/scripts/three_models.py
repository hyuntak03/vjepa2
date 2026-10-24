#!/usr/bin/env python3
"""[E] B 확장 — 세 블록 AR (기준 · VOBench 이어 학습 · 합성 전용) 을 같은 창 · 같은 지표로 (auto_research, 2026-10-22). 학습 없음 · 읽기만.

  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/three_models.py

데이터: (A) 3D 합성 검증 세트 캐시 (라벨 → 물체 칸) · (V) VOBench 끝 가림 공유 캐시 (가능 팔 pos_a; 물체 칸 = 같은 block pos_a · pos_b 픽셀 차 > 12 단계인 칸).
한 걸음 TF (블록 t → t+1). 칸 무리: 들어오는 칸 (블록 t 물체 없음, t+1 물체) · 물체 칸 (블록 t) · 정지 칸 (블록 t−7..t+1 물체가 2 칸 안에 안 온 칸).
  1) 들어오는 칸 query → 블록 t 물체 칸 key attention (층별, 헤드 평균) · 균등 대비 — 이동 거리 (칸) 별
  2) 문맥 평균에서 떨어진 정도 = Σ|p − m| / Σ|y − m|, m = 같은 칸 블록 t−7..t 평균 (무리별)
  3) 층별 표현 유사도 = 기준과의 선형 CKA (층 0 = 입력 임베딩, 1..12 블록 출력) — 세 모델은 따로 학습돼 은닉 기저가 달라 원 좌표 차이는 뜻이 없다. 출력 (같은 토큰 공간) 은 상대 차이로 — 무리별"""
import argparse, glob, json, math, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--p3", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--vob", default="/workspace/auto_research_cache/vobench_late")
ap.add_argument("--vroot", default="/workspace/dataset/benchmarks/VOBench_v1")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/three_models.json"))
a = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                          # noqa: E402
sys.argv = _argv
from src.models.utils import modules as MOD                                      # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda"); G, S, D = 16, 256, 1280
MODELS = {"base": ("natural_ar_vw_state", "latest.pt"), "vob": ("natural_ar_vw_state_vob", "latest.pt"), "syn": ("natural_ar_vw_state_phys3d_only", "e10.pt")}
NETS = {k: FT.load_pred(r, c).to(dev).eval() for k, (r, c) in MODELS.items()}

REC = {"on": False, "nb": 0, "cnt": 0, "q": {}, "keys": {}, "mass": {}}
_sdpa = F.scaled_dot_product_attention


def sdpa(q, k, v, *args, **kw):
    out = _sdpa(q, k, v, *args, **kw)
    if not REC["on"] or kw.get("attn_mask") is not None:
        return out
    b = k.shape[2] // S - 1
    layer = REC["cnt"] // REC["nb"]; REC["cnt"] += 1
    if b in REC["q"]:
        w = torch.softmax(q[:, :, REC["q"][b]].float() @ k.float().transpose(-1, -2) / math.sqrt(q.size(-1)), -1).mean(1)[0]
        for nm, kidx in REC["keys"][b].items():
            REC["mass"].setdefault((b, nm), {})[layer] = float(w[:, kidx].sum(-1).mean()) if len(kidx) else float("nan")
    return out


MOD.F.scaled_dot_product_attention = sdpa
HID = []
for net in NETS.values():
    for blk in net.predictor_blocks:
        blk.register_forward_hook(lambda m, i, o: HID.append(o.detach().float()) if REC["on"] else None)


def run(net, h, q, keys):
    nb = h.shape[0]
    HID.clear()
    REC.update(on=True, nb=nb, cnt=0, q=q, keys=keys, mass={})
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = h.reshape(1, nb * S, D)
        emb = net._ar_embed_obs(x).float()
        p = net.forward_seq(x, torch.arange(nb * S, device=dev).unsqueeze(0)).float()[0].view(nb, S, D)
    REC["on"] = False
    hid = [emb[0].view(nb, S, -1)] + [z[0].view(nb, S, -1) for z in HID]          # 13 개 (층 0..12)
    return p, dict(REC["mass"]), hid


def dil(m, k=2):
    m = m.reshape(G, G).copy()
    for _ in range(k):
        o = m.copy(); o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
        o[1:, 1:] |= m[:-1, :-1]; o[:-1, :-1] |= m[1:, 1:]; o[1:, :-1] |= m[:-1, 1:]; o[:-1, 1:] |= m[1:, :-1]; m = o
    return m.reshape(-1)


def dbin(d):
    for lo, hi, nm in ((0, 0.5, "0"), (0.5, 1.5, "1"), (1.5, 2.5, "2"), (2.5, 3.5, "3"), (3.5, 99, "4+")):
        if lo <= d < hi:
            return nm


ACC = {}
FEAT = {}


def put(k, v):
    ACC.setdefault(k, []).append(v)


def analyse(tag, h, steps):
    """steps: dict(t, enter, obj, static, d)."""
    nb = h.shape[0]
    q = {s["t"]: list(s["enter"]) for s in steps if len(s["enter"])}
    keys = {s["t"]: {"obj_last": [s["t"] * S + c for c in s["obj"]]} for s in steps if len(s["enter"])}
    outs = {}
    for name, net in NETS.items():
        outs[name] = run(net, h, q, keys)
    for s in steps:
        t = s["t"]; m_ = h[max(0, t - 7):t + 1].mean(0); y = h[t + 1]
        groups = {"enter": s["enter"], "static": s["static"], "obj": s["obj"]}
        for name, (p, mass, hid) in outs.items():
            for gn, cells in groups.items():
                if len(cells) == 0:
                    continue
                c = torch.as_tensor(cells, device=dev)
                put((tag, name, gn, "dep"), float((p[t][c] - m_[c]).abs().sum() / (y[c] - m_[c]).abs().sum().clamp_min(1e-9)))
                ht_ = h[t][c]; pc_, yc_ = p[t][c], y[c]
                put((tag, name, gn, "err_copy"), float((pc_ - yc_).abs().sum() / (ht_ - yc_).abs().sum().clamp_min(1e-9)))      # < 1 = 복사보다 정답에 가깝다
                put((tag, name, gn, "err_mean"), float((pc_ - yc_).abs().sum() / (m_[c] - yc_).abs().sum().clamp_min(1e-9)))    # < 1 = 문맥 평균보다 가깝다
                put((tag, name, gn, "pres"), float(((pc_ - ht_) * (yc_ - ht_)).sum() / ((yc_ - ht_) ** 2).sum().clamp_min(1e-9)))  # 복사 → 정답 쪽 사영
                put((tag, name, gn, "cos"), float(F.cosine_similarity(pc_, yc_, dim=-1).mean()))
                cs = c[torch.randperm(len(c), device=dev)[:8]]                    # CKA 용 표본 (걸음당 8 칸)
                for l in range(13):
                    FEAT.setdefault((tag, gn, name, l), []).append(hid[l][t][cs].cpu())
                if name != "base":                                               # 출력 (같은 토큰 공간) 차이는 바로 비교 가능
                    put((tag, name, gn, "outdiff"), float((p[t][c] - outs["base"][0][t][c]).norm(dim=-1).mean() / outs["base"][0][t][c].norm(dim=-1).mean()))
            if len(s["enter"]):
                put((tag, name, "enter", f"attn_d{dbin(s['d'])}"), [mass.get((t, "obj_last"), {}).get(l, float("nan")) for l in range(12)])
                put((tag, name, "enter", f"uni_d{dbin(s['d'])}"), len(s["obj"]) / ((t + 1) * S))


def cover(bbox):
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return (oy[..., :, None] * ox[..., None, :]).reshape(*bbox.shape[:-1], -1)


# ---------------- (A) 3D 합성
nwin = 0
for f in sorted(glob.glob(os.path.join(a.p3, "h_val_target_s*.npy"))):
    sh = f.split("_s")[-1].split(".")[0]
    lab = np.load(os.path.join(a.p3, f"lab_val_s{sh}.npz")); Hc = np.load(f, mmap_mode="r")
    for w in range(Hc.shape[0]):
        if a.limit and nwin >= a.limit:
            break
        cvb = cover(np.nan_to_num(lab["bbox"][w], nan=-100.0)); cvb = np.maximum(cvb[0::2], cvb[1::2])
        vis = lab["vis_frac"][w]; pos = lab["pos"][w]
        steps = []
        for t in range(7, 15):
            if not vis[2 * t:2 * t + 4].min() > 0.7:
                continue
            pc, pn = pos[2 * t:2 * t + 2].mean(0), pos[2 * t + 2:2 * t + 4].mean(0)
            if not (np.isfinite(pc).all() and np.isfinite(pn).all()):
                continue
            obj = np.where(cvb[t] >= 0.5)[0]
            if len(obj) == 0:
                continue
            enter = np.where((cvb[t + 1] >= 0.5) & (cvb[t] == 0))[0]
            static = np.where(~dil((cvb[t - 7:t + 2] > 0).any(0)))[0]
            steps.append(dict(t=t, enter=enter, obj=obj, static=static, d=float(np.linalg.norm(pn - pc) / 16)))
        if steps:
            analyse("A", torch.from_numpy(np.asarray(Hc[w], np.float32)).to(dev), steps)
            nwin += 1
            if nwin % 50 == 0:
                print(f"(A) {nwin}", flush=True)

# ---------------- (V) VOBench 끝 가림
meta = json.load(open(os.path.join(a.vob, "meta.json"))); HV = np.load(os.path.join(a.vob, "h.npy"), mmap_mode="r")


def frames(rel):
    fr = [np.asarray(Image.open(os.path.join(a.vroot, rel, f"{3 * k:06d}.png")).convert("RGB")) for k in range(32)]
    x = torch.from_numpy(np.stack(fr)).to(dev).permute(0, 3, 1, 2).float()
    return F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).round()   # (32, 3, 256, 256)


for bi, mrec in enumerate(meta):
    if a.limit and bi >= a.limit:
        break
    xa, xb = frames(mrec["files"]["pos_a"]), frames(mrec["files"]["pos_b"])
    cd = (xa - xb).abs().mean(1).view(32, G, 16, G, 16).mean(dim=(2, 4)).reshape(32, -1).cpu().numpy()
    om = np.maximum(cd[0::2], cd[1::2]) > 12                                    # (16, 256) 블록별 물체 칸 (두 팔의 물체가 다른 칸)
    cen = [np.array(np.unravel_index(np.where(om[b])[0], (G, G))).mean(1) if om[b].any() else None for b in range(16)]
    steps = []
    for t in range(7, 15):
        if not (om[t].any() and om[t + 1].any()):
            continue
        enter = np.where(om[t + 1] & ~om[t])[0]; obj = np.where(om[t])[0]
        static = np.where(~dil(om[t - 7:t + 2].any(0)))[0]
        steps.append(dict(t=t, enter=enter, obj=obj, static=static, d=float(np.linalg.norm(cen[t + 1] - cen[t]))))
    if steps:
        analyse(f"V_{'static' if mrec['motion'] == 'flat_static' else 'moving'}", torch.from_numpy(np.asarray(HV[bi, 0], np.float32)).to(dev), steps)
    if bi % 50 == 0:
        print(f"(V) {bi}/{len(meta)}", flush=True)

# ---------------- 요약
res = {}
for k, v in ACC.items():
    arr = np.array(v, dtype=np.float64)
    res["|".join(k)] = {"n": len(v), "mean": (np.nanmean(arr, 0).tolist() if arr.ndim > 1 else float(np.nanmean(arr)))}
def cka(X, Y):                                                                   # 선형 CKA (Kornblith 등 2019) — 기저에 무관한 표현 유사도
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float((Y.T @ X).pow(2).sum() / ((X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt()))
for tag in sorted({k[0] for k in FEAT}):
    for gn in ("enter", "obj", "static"):
        for m in ("vob", "syn"):
            vals = []
            for l in range(13):
                if (tag, gn, "base", l) not in FEAT:
                    break
                X = torch.cat(FEAT[(tag, gn, "base", l)]).double(); Y = torch.cat(FEAT[(tag, gn, m, l)]).double()
                vals.append(cka(X, Y))
            if vals:
                res[f"{tag}|{m}|{gn}|cka"] = {"n": int(sum(len(x) for x in FEAT[(tag, gn, 'base', 0)])), "mean": vals}
json.dump(res, open(a.out, "w"), indent=1)
for tag in sorted({k.split("|")[0] for k in res}):
    print(f"\n==== {tag}")
    for gn in ("enter", "obj", "static"):
        print(f"  [문맥 평균에서 떨어진 정도 · {gn}] " + " · ".join(f"{m} {res[f'{tag}|{m}|{gn}|dep']['mean']:.3f} (n {res[f'{tag}|{m}|{gn}|dep']['n']})" for m in NETS if f"{tag}|{m}|{gn}|dep" in res))
        print(f"  [정답과 맞는 정도 · {gn}] " + " · ".join(f"{m}: 오차/복사 {res[f'{tag}|{m}|{gn}|err_copy']['mean']:.3f} · 오차/문맥평균 {res[f'{tag}|{m}|{gn}|err_mean']['mean']:.3f} · 정답 쪽 사영 {res[f'{tag}|{m}|{gn}|pres']['mean']:.3f} · cos {res[f'{tag}|{m}|{gn}|cos']['mean']:.3f}" for m in NETS if f"{tag}|{m}|{gn}|err_copy" in res))
        for m in ("vob", "syn"):
            k = f"{tag}|{m}|{gn}|cka"
            if k in res:
                print(f"  [기준과의 표현 유사도 (선형 CKA) {m} · {gn}] 층 0..12: " + " ".join(f"{x:.2f}" for x in res[k]["mean"])
                      + f" | 출력 차이 (기준 대비) {res[f'{tag}|{m}|{gn}|outdiff']['mean']:.2f}")
    for db in ("1", "2", "3", "4+"):
        row = []
        for m in NETS:
            ka, ku = f"{tag}|{m}|enter|attn_d{db}", f"{tag}|{m}|enter|uni_d{db}"
            if ka in res:
                L = np.array(res[ka]["mean"]); row.append(f"{m} 최대 {np.nanmax(L):.3f} (층 {int(np.nanargmax(L)) + 1}) ÷균등 {np.nanmax(L) / res[ku]['mean']:.0f} (n {res[ka]['n']})")
        if row:
            print(f"  [attention 이동 {db} 칸] " + " · ".join(row))
print("→", a.out)
