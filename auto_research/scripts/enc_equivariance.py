#!/usr/bin/env python3
"""고리 1 [E] — 블록 토큰은 평행 이동에 등변인가 (리더 과제, [T] 의 "움직이는 물체 안쪽 0.53 vs 정지 0.18" 공백) (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=3 /venv/vjepa/bin/python -W ignore auto_research/scripts/enc_equivariance.py

모든 토큰 = 블록 (2 장) 별 LN(enc), bf16 (학습 정답과 같은 처리). 거리 = 칸별 차원 평균 |Δh|, 척도 = 다른 창 같은 칸 거리 (토큰 변화 비율).
검사 1 — 화면 전체 평행 이동 (오른쪽 s px, s = 8 · 16 · 32):
  reflect  왼쪽을 반사 패딩, 오른쪽을 잘라냄          (phys3d · vobench · k710)
  roll     순환 이동 (내용 손실 없음, 이음매가 화면 안으로)  (phys3d · vobench · k710)
  pan      더 넓은 원본에서 256 창을 s px 옆에서 다시 자름 = 실제 카메라 팬 (k710: 높이 256 으로 줄인 원본)
  s = 16k: 등변 오차 |h(Sx)(r, col) − h(x)(r, col−k)| (안쪽 칸: 가장자리 2 칸 · 이동으로 들어온 칸 제외) · 고정 칸 변화 |h(Sx)(c) − h(x)(c)|
  s = 8  : 가장 가까운 두 칸 (col, col−1) 과 그 평균까지의 거리
  기준: ±1/255 섭동 |h(x+n) − h(x)| (같은 칸) · 무작위 초기화 encoder 의 같은 양
검사 2 — 물체만 이동 (3D 합성):
  (a) 실제 블록 쌍 (캐시 토큰): 물체 안쪽 칸 (두 장 모두 덮임 > 0.6) c 에서 |h_{t+1}(c + round(d/16)) − h_t(c)|, d = 블록 중심 변위.
      정지 물체 (|d| < 1) 의 |h_{t+1}(c) − h_t(c)| 와 비교. 변위의 칸 아래 위상 |d − 16·round(d/16)| 별 · 블록 안 움직임 크기별.
  (b) 합성 합성 (개입): 창의 시간 중앙값 = 배경, 한 장에서 물체 픽셀을 떼어 배경 위에 붙인다.
      S0 = (물체@0, 물체@0) 정지 블록 · S_s = (물체@s, 물체@s) (s = 8 · 16 · 32, 정지) · M_v = (물체@0, 물체@v) (블록 안 움직임 v = 4 · 8 · 16).
      등변 오차 (S_s vs S0 를 s/16 칸 옮겨 비교) · 움직임 효과 |h(M_v) − h(S0)| · 먼 칸 번짐 · ±1/255 바닥."""
import argparse, csv, glob, json, os, random, sys
import numpy as np, torch, torch.nn.functional as F
import decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/workspace/auto_research_cache/enc_noise")
ap.add_argument("--p3", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--n_obj", type=int, default=120)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/enc_equivariance.json"))
a = ap.parse_args()
torch.set_num_threads(8)
dev = torch.device("cuda"); G, R = 16, 256
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
SD = _load_checkpoint(CK)["target_encoder"]
ENC = {}
for name in ("target", "random"):
    torch.manual_seed(0)
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    if name == "target":
        _load_state_dict(e, SD, tag="t")
    ENC[name] = e.to(dev, torch.bfloat16).eval()
del SD
gen = torch.Generator(device=dev).manual_seed(0)


@torch.no_grad()
def enc(name, px):                    # px (B,2,R,R,3) float 0..255 → (B,G,G,D)
    x = (px.div(255).permute(0, 4, 1, 2, 3) - MEAN) / STD
    out = []
    for i in range(0, x.size(0), 8):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = ENC[name](x[i:i + 8].to(torch.bfloat16)).float()
        out.append(F.layer_norm(h, (h.size(-1),)))
    return torch.cat(out).view(-1, G, G, out[0].size(-1))


def dist(a_, b_):
    return (a_ - b_).abs().mean(-1)


def shift_img(x, s, mode):            # x (B,2,R,R,3), 오른쪽으로 s px
    if mode == "roll":
        return torch.roll(x, s, dims=3)
    xp = x.permute(0, 1, 4, 2, 3).reshape(-1, 3, R, R)
    xp = F.pad(xp, (s, 0, 0, 0), mode="reflect")[..., :R]
    return xp.view(x.shape[0], 2, 3, R, R).permute(0, 1, 3, 4, 2)


def interior(k, mode):                # 비교할 칸 (row, col) 마스크: col 은 이동 뒤 자리
    m = torch.zeros(G, G, dtype=torch.bool, device=dev)
    if mode == "roll":
        m[2:G - 2, 2 + k:G - 2] = True                                   # 이음매 근처 제외
    else:
        m[2:G - 2, max(2, k + 2):G - 2] = True
    return m


def test1(name, X, mode, Xpan=None):
    """X (B,2,R,R,3) float. Xpan: {s: (B,2,R,R,3)} 실제 팬 (그때 mode='pan')."""
    h0 = enc(name, X)
    n = torch.randint(-1, 2, X.shape, device=dev, generator=gen).float()
    nz = dist(enc(name, (X + n).clamp(0, 255)), h0)
    r = {"noise": float(nz[:, 2:G - 2, 2:G - 2].mean())}
    hr = h0.roll(1, 0)
    r["cross"] = float(dist(hr, h0)[:, 2:G - 2, 2:G - 2].mean())
    for s in (8, 16, 32):
        xs = Xpan[s] if mode == "pan" else shift_img(X, s, mode)
        hs = enc(name, xs)
        if s % 16 == 0:
            k = s // 16
            ref = h0.roll(k, 2)                                            # ref(r, col) = h0(r, col−k)
            m = interior(k, mode)
            r[f"eq{s}"] = float(dist(hs, ref)[:, m].mean())
            r[f"fixed{s}"] = float(dist(hs, h0)[:, m].mean())
            # 가장자리에서 먼 칸만 (col ≥ k+5, row 4..11)
            m2 = torch.zeros_like(m); m2[4:12, k + 5:12] = True
            r[f"eq{s}_core"] = float(dist(hs, ref)[:, m2].mean())
        else:
            m = interior(1, mode)
            r["half_same"] = float(dist(hs, h0)[:, m].mean())
            r["half_prev"] = float(dist(hs, h0.roll(1, 2))[:, m].mean())
            r["half_avg"] = float(dist(hs, 0.5 * (h0 + h0.roll(1, 2)))[:, m].mean())
    return r


RES = {"test1": {}, "test2_real": {}, "test2_synth": {}}
rng = np.random.default_rng(0)
# ---------------- 검사 1 데이터
srcs = {}
P3 = np.load(sorted(glob.glob(os.path.join(a.p3, "pix_val_s*.npy")))[0], mmap_mode="r")
srcs["phys3d"] = torch.from_numpy(np.asarray(P3[rng.choice(P3.shape[0], a.n, replace=False)][:, :2])).float().to(dev)
fs = sorted(glob.glob(os.path.join(a.dir, "vobench_target_encoder_s*_pix.npy")))
if fs:
    V = np.load(fs[0], mmap_mode="r")
    srcs["vobench"] = torch.from_numpy(np.asarray(V[rng.choice(V.shape[0], min(a.n, V.shape[0]), replace=False)][:, :2])).float().to(dev)
# k710: 높이 256 으로 줄인 원본에서 가로 오프셋 o, o+s 로 자른다 (실제 팬)
rows = list(csv.reader(open(os.path.join(REPO, "data_csv/k710_320/fps4_table.tsv")), delimiter="\t"))
random.Random(1).shuffle(rows)
K, KP = [], {8: [], 16: [], 32: []}
for r_ in rows:
    if len(K) >= a.n:
        break
    try:
        vr = decord.VideoReader(r_[0], num_threads=2)
        sd = int(r_[3]); s0 = random.Random(r_[0]).randrange(0, max(1, len(vr) - sd - 1))
        fr = torch.from_numpy(vr.get_batch([s0, s0 + sd]).asnumpy()).to(dev).permute(0, 3, 1, 2).float()
    except Exception:
        continue
    H_, W_ = fr.shape[-2:]
    Wn = int(round(W_ * R / H_))
    if Wn < R + 40:
        continue
    fr = F.interpolate(fr, size=(R, Wn), mode="bilinear", align_corners=False).round().clamp(0, 255).permute(0, 2, 3, 1)   # (2,R,Wn,3)
    o = (Wn - R - 32) // 2 + 32
    K.append(fr[:, :, o:o + R])
    for s in (8, 16, 32):
        KP[s].append(fr[:, :, o - s:o - s + R])                           # 창을 왼쪽으로 s → 내용은 오른쪽으로 s
srcs["k710"] = torch.stack(K)
KP = {s: torch.stack(v) for s, v in KP.items()}
for name in ("target", "random"):
    RES["test1"][name] = {}
    for src, X in srcs.items():
        for mode in (["reflect", "roll"] + (["pan"] if src == "k710" else [])):
            r = test1(name, X, mode, KP if mode == "pan" else None)
            RES["test1"][name][f"{src}/{mode}"] = r
            c = r["cross"]
            print(f"[검사1 {name:6s}] {src:8s} {mode:7s} | 척도 {c:.3f} · ±1/255 {r['noise'] / c:5.1%} | "
                  f"16px 등변 오차 {r['eq16'] / c:5.1%} (안쪽 {r['eq16_core'] / c:5.1%}) vs 고정 칸 변화 {r['fixed16'] / c:5.1%} | "
                  f"32px 등변 {r['eq32'] / c:5.1%} vs 고정 {r['fixed32'] / c:5.1%} | 8px: 같은 칸 {r['half_same'] / c:5.1%} · 왼칸 {r['half_prev'] / c:5.1%} · 평균 {r['half_avg'] / c:5.1%}", flush=True)

# ---------------- 검사 2 (a): 실제 블록 쌍 (캐시 토큰)
def cover(bbox):
    x0, y0, x1, y1 = [bbox[..., i, None] for i in range(4)]
    c = np.arange(G) * 16.0
    ox = np.clip(np.minimum(c + 16, x1) - np.maximum(c, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c + 16, y1) - np.maximum(c, y0), 0, 16) / 16
    return oy[..., :, None] * ox[..., None, :]


acc = {}
def put(k, v):
    acc.setdefault(k, []).append(v)


for f in sorted(glob.glob(os.path.join(a.p3, "h_val_target_s*.npy"))):
    sh = f.split("_s")[-1].split(".")[0]
    lab = np.load(os.path.join(a.p3, f"lab_val_s{sh}.npz")); Hc = np.load(f, mmap_mode="r")
    for w in range(Hc.shape[0]):
        bb = np.nan_to_num(lab["bbox"][w], nan=-100.0); cv = cover(bb)                 # (32,G,G)
        ins = np.minimum(cv[0::2], cv[1::2]) > 0.6                                       # (16,G,G) 물체 안쪽
        pos = lab["pos"][w]
        h = None
        for t in range(15):
            if not ins[t].any():
                continue
            pc_t = pos[2 * t:2 * t + 2].mean(0); pc_n = pos[2 * t + 2:2 * t + 4].mean(0)
            d = pc_n - pc_t
            if not np.isfinite(d).all():
                continue
            if h is None:
                h = np.asarray(Hc[w], np.float32)
            dn = np.linalg.norm(d); kk = np.round(d / 16).astype(int)
            wmov = np.linalg.norm(pos[2 * t + 1] - pos[2 * t])                         # 블록 안 움직임 (px/장)
            ys, xs = np.where(ins[t])
            yt, xt = ys + kk[1], xs + kk[0]
            ok = (yt >= 0) & (yt < G) & (xt >= 0) & (xt < G)
            ok &= ins[t + 1][np.clip(yt, 0, G - 1), np.clip(xt, 0, G - 1)]
            if not ok.any():
                continue
            ys, xs, yt, xt = ys[ok], xs[ok], yt[ok], xt[ok]
            ht, hn = h[t].reshape(G, G, -1), h[t + 1].reshape(G, G, -1)
            eqd = np.abs(hn[yt, xt] - ht[ys, xs]).mean(-1).mean()
            fix = np.abs(hn[ys, xs] - ht[ys, xs]).mean(-1).mean()
            if dn < 1:
                put("static_obj", eqd)
            else:
                ph = np.abs(d - 16 * kk).max()
                put(f"move_phase{'<4' if ph < 4 else ('<8' if ph < 8.1 else '>8')}", eqd)
                put(f"move_inblock{'<2' if wmov < 2 else ('<5' if wmov < 5 else '>5')}px", eqd)
                put("move_all", eqd); put("move_all_fixedcell", fix)
                if ph < 4 and wmov < 2:
                    put("move_phase<4_inblock<2", eqd)
RES["test2_real"] = {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in acc.items()}
print("\n[검사2 (a) 실제 쌍, 물체 안쪽 칸 · 칸만큼 옮겨 비교] " + " · ".join(f"{k} {v['mean']:.3f} (n {v['n']})" for k, v in sorted(RES["test2_real"].items())), flush=True)

# ---------------- 검사 2 (b): 합성 (배경 = 시간 중앙값, 물체 붙이기)
files = sorted(glob.glob(os.path.join(a.p3, "pix_val_s*.npy")))
cand = []
for f in files:
    sh = f.split("_s")[-1].split(".")[0]
    lab = np.load(os.path.join(a.p3, f"lab_val_s{sh}.npz"))
    for w in range(lab["bbox"].shape[0]):
        bb = lab["bbox"][w]
        if not np.isfinite(bb).all():
            continue
        cx = (bb[:, 0] + bb[:, 2]) / 2; cy = (bb[:, 1] + bb[:, 3]) / 2
        size = (bb[:, 2] - bb[:, 0]).mean()
        if np.hypot(cx.max() - cx.min(), cy.max() - cy.min()) > 3 * size and 20 < size < 70:
            cand.append((f, w))
rng.shuffle(cand)
syn = {}
def sput(k, v):
    syn.setdefault(k, []).append(v)


nobj = 0
for f, w in cand:
    if nobj >= a.n_obj:
        break
    sh = f.split("_s")[-1].split(".")[0]
    lab = np.load(os.path.join(a.p3, f"lab_val_s{sh}.npz"))
    fr = torch.from_numpy(np.asarray(np.load(f, mmap_mode="r")[w])).float().to(dev)         # (32,R,R,3)
    Bg = fr.median(0).values
    bb = lab["bbox"][w]
    # 물체가 화면 가운데 쪽 · 오른쪽으로 32 px 여유가 있는 장
    ok = [i for i in range(32) if bb[i, 0] > 24 and bb[i, 2] < R - 56 and bb[i, 1] > 24 and bb[i, 3] < R - 24 and lab["vis_frac"][w][i] > 0.95]
    if not ok:
        continue
    i = ok[len(ok) // 2]
    x0, y0, x1, y1 = [int(v) for v in bb[i]]
    diff = (fr[i] - Bg).abs().mean(-1)
    M = torch.zeros(R, R, dtype=torch.bool, device=dev)
    M[max(y0 - 2, 0):y1 + 3, max(x0 - 2, 0):x1 + 3] = diff[max(y0 - 2, 0):y1 + 3, max(x0 - 2, 0):x1 + 3] > 6
    if M.sum() < 150:
        continue
    def comp(dx):
        img = Bg.clone()
        Ms = torch.roll(M, dx, 1); src = torch.roll(fr[i], dx, 1)
        img[Ms] = src[Ms]
        return img
    C = {dx: comp(dx) for dx in (0, 4, 8, 16, 32)}
    blocks = torch.stack([torch.stack([C[0], C[0]]), torch.stack([C[8], C[8]]), torch.stack([C[16], C[16]]), torch.stack([C[32], C[32]]),
                          torch.stack([C[0], C[4]]), torch.stack([C[0], C[8]]), torch.stack([C[0], C[16]]),
                          torch.stack([C[0], C[0]]) + torch.randint(-1, 2, (2, R, R, 3), device=dev, generator=gen).float()]).clamp(0, 255)
    cvr = M.float().view(G, 16, G, 16).mean(dim=(1, 3))                                         # 칸별 물체 덮임 (S0)
    obj = cvr > 0.3
    far = torch.ones(G, G, dtype=torch.bool, device=dev)
    oy, ox = torch.where(cvr > 0)
    far[max(int(oy.min()) - 3, 0):int(oy.max()) + 4, max(int(ox.min()) - 3, 0):int(ox.max()) + 6] = False
    if obj.sum() < 2 or far.sum() < 20:
        continue
    for name in ("target", "random"):
        hh = enc(name, blocks)                                                                  # (8,G,G,D)
        h0 = hh[0]
        sput(f"{name}/noise_obj", float(dist(hh[7], h0)[obj].mean())); sput(f"{name}/noise_far", float(dist(hh[7], h0)[far].mean()))
        for j, s in ((2, 16), (3, 32)):
            k = s // 16
            sput(f"{name}/eq{s}_obj", float(dist(hh[j].roll(-k, 1), h0)[obj].mean()))      # h_s(r, c+k) vs h0(r, c)
            sput(f"{name}/fixed{s}_obj", float(dist(hh[j], h0)[obj].mean()))
            sput(f"{name}/far{s}", float(dist(hh[j], h0)[far].mean()))
        sput(f"{name}/half8_same_obj", float(dist(hh[1], h0)[obj].mean()))
        sput(f"{name}/half8_avg_obj", float(dist(hh[1], 0.5 * (h0 + h0.roll(1, 1)))[obj].mean()))
        for j, v in ((4, 4), (5, 8), (6, 16)):
            sput(f"{name}/motion{v}_obj", float(dist(hh[j], h0)[obj].mean())); sput(f"{name}/motion{v}_far", float(dist(hh[j], h0)[far].mean()))
        sput(f"{name}/cross_far", float(dist(h0, enc(name, torch.stack([torch.stack([fr[0], fr[1]])]))[0])[far].mean()) if False else 0.0)
    nobj += 1
RES["test2_synth"] = {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in syn.items() if not k.endswith("cross_far")}
print(f"\n[검사2 (b) 합성, 물체 {nobj} 개]")
for name in ("target", "random"):
    S = {k.split('/')[1]: v["mean"] for k, v in RES["test2_synth"].items() if k.startswith(name + "/")}
    print(f"  {name:6s} ±1/255 물체 {S['noise_obj']:.3f} 먼 {S['noise_far']:.3f} | 16px 이동 등변 오차 (물체) {S['eq16_obj']:.3f} vs 고정 칸 {S['fixed16_obj']:.3f} · 먼 칸 번짐 {S['far16']:.3f} | "
          f"32px 등변 {S['eq32_obj']:.3f} vs 고정 {S['fixed32_obj']:.3f} · 먼 칸 {S['far32']:.3f} | 8px 같은 칸 {S['half8_same_obj']:.3f} 평균 {S['half8_avg_obj']:.3f} | "
          f"블록 안 움직임 4/8/16px: 물체 {S['motion4_obj']:.3f} {S['motion8_obj']:.3f} {S['motion16_obj']:.3f} · 먼 칸 {S['motion4_far']:.3f} {S['motion8_far']:.3f} {S['motion16_far']:.3f}", flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(RES, open(a.out, "w"), indent=1)
print("→", a.out)
