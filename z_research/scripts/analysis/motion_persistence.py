#!/usr/bin/env python3
"""운동 지속성 측정 (2026-10-18) — 학습 데이터에서 "관성" 이 얼마나 맞나. 학습 없음, encoder 만.

  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python z_research/scripts/analysis/motion_persistence.py --validate     # VOBench 로 추적기 검증
  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python z_research/scripts/analysis/motion_persistence.py --n 150        # K710 · SSv2 · HowTo

질문: 모델 예측의 움직임이 실제의 0.3–0.5 배인 것은 (a) 데이터에서 관성이 원래 약해서 L1 이 그 중앙값을 맞게 배운 것인가,
     (b) 데이터는 관성이 강한데 모델이 섞어서 줄인 것인가.
방법 (블록 H_b = LN(target_encoder(블록 b 단독)), 16×16 칸):
  1 걸음 추적  칸 p 의 특징 → 다음 블록에서 반경 3 칸 (7×7) 안 cosine s(q), w = softmax(s/τ), τ 0.05, p' = Σ w q (소수 위치), d = p' − p
  k 걸음       위치는 소수로 누적 (탐색 원점은 반올림 칸), D_k = p_k − p_0
  필터         매 걸음 max s ≥ 0.7 · 왕복 (k 걸음 뒤 거꾸로 추적해 1 칸 이내 복귀) · 관측 속도 ‖v‖ ≥ 0.5 칸 (v = 직전 블록 → 이 블록 한 걸음, 거꾸로 추적)
  카메라 보정  블록 쌍마다 전체 칸 1 걸음 변위의 중앙값 g 를 뺀다. 어느 블록이든 ‖g‖ > 1 칸이면 clip 제외
  ρ_k = (D_k · v) / (k ‖v‖²) → r_k = median ρ_k (1 등속, 0 멈춤, <0 반대) · π_k = mean 1(‖D_k − k v‖ < ‖D_k‖) · c_k = median cos(D_k, v)
간격: 학습과 같게 — 데이터셋마다 fps4 표의 step (30 fps → 7 프레임) 간격으로 18 장 = 9 블록 (블록당 약 0.47 s).
  VOBench 검증은 채점과 같은 간격 (30 fps 3 프레임, 블록 0.2 s).
함정: 추적은 보이는 동안만 (가림은 신뢰도 필터로 빠진다 → 영속은 따로). 0.5 칸 / 블록보다 느린 운동은 잘 안 잡힌다.
"""
import argparse, csv, json, os, sys, random, collections
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--validate", action="store_true")
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--tau", type=float, default=0.05)
ap.add_argument("--vobench", default="/workspace/dataset/benchmarks/VOBench_v1")
ap.add_argument("--out", default=os.path.join(REPO, "z_research/VOBench_v1/figures/brief_diag"))
a = ap.parse_args()
dev = "cuda"; G, S, Rr, KS = 16, 256, 3, (1, 3, 5, 7)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
state = _load_checkpoint(CK)
tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tenc, state["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
YX = torch.stack(torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij"), -1).view(S, 2).float().to(dev)   # 칸 (행, 열)
NEAR = (YX.unsqueeze(1) - YX.unsqueeze(0)).abs().max(-1).values <= Rr                                          # (S, S) 반경 3

@torch.no_grad()
def encode(x):
    """x (3, T, 256, 256) 정규화 → (T/2, S, D) 블록별 LN."""
    T = x.shape[1] // 2
    xb = x[:, : 2 * T].reshape(3, T, 2, 256, 256).permute(1, 0, 2, 3, 4).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.cat([tenc(xb[i:i + 8]) for i in range(0, T, 8)], 0).float()
    return F.normalize(F.layer_norm(h, (h.size(-1),)), dim=-1)          # cosine 용으로 단위 벡터

def step(Ha, Hb, pos):
    """pos (N, 2) 소수 위치 → 반올림 칸에서 Hb 로 1 걸음. return 새 위치 (N, 2), 최대 유사도 (N,)."""
    cell = pos.round().clamp(0, G - 1); ci = (cell[:, 0] * G + cell[:, 1]).long()
    s = Ha[ci] @ Hb.T                                                   # (N, S)
    s = s.masked_fill(~NEAR[ci], -1e4)
    w = torch.softmax(s / a.tau, -1)
    p_ = w @ YX                                                         # (N, 2)
    step.peak = w.max(-1).values                                        # 2026-10-18 v2: 모호한 매칭 (반복 무늬 · 평면) 거르기용
    return pos + (p_ - cell), s.max(-1).values

def camera(H):
    """블록 쌍마다 전체 칸 1 걸음 변위 중앙값 (T-1, 2)."""
    g = []
    for b in range(H.size(0) - 1):
        p1, _ = step(H[b], H[b + 1], YX.clone())
        g.append((p1 - YX).median(0).values)
    return torch.stack(g)

PEAK = 0.5
def measure(H, b0, cells, cam=True):
    """v2 (2026-10-18): **앞으로만** 추적. cells (N,) = 블록 b0-1 의 칸 → 한 걸음 = 관측 속도 v → 이어서 k 걸음 = D_k.
    (v1 은 b0 에서 뒤로 · 앞으로 따로 추적해 b0 의 추적 오차가 v 와 D 에 반대 부호로 들어가 D ≈ −v 인공물이 났다.)
    매 걸음 max cos ≥ 0.7 · softmax 최대 가중 ≥ 0.5 (모호한 매칭 제외) · 왕복 (b0+k → b0-1) 1 칸 이내."""
    T = H.size(0)
    g = camera(H) if cam else torch.zeros(T - 1, 2, device=dev)
    if cam and (g.norm(dim=-1) > 1).any():
        return None
    start = YX[cells].clone()
    p0, s0 = step(H[b0 - 1], H[b0], start); ok = (s0 >= 0.7) & (step.peak >= PEAK)
    v = (p0 - start) - g[b0 - 1]
    ok &= v.norm(dim=-1) >= 0.5
    out = {}; pos = p0.clone()
    for k in range(1, max(KS) + 1):
        if b0 + k >= T:
            break
        pos, sk = step(H[b0 + k - 1], H[b0 + k], pos); ok = ok & (sk >= 0.7) & (step.peak >= PEAK)
        if k in KS:
            back = pos.clone(); ok_b = torch.ones_like(ok)
            for j in range(k, -1, -1):                                  # 왕복: b0+k → … → b0-1
                back, sj = step(H[b0 + j], H[b0 + j - 1], back); ok_b &= sj >= 0.7
            rt = (back - start).norm(dim=-1) <= 1.0
            D = (pos - p0) - g[b0: b0 + k].sum(0)
            m = ok & rt & ok_b
            out[k] = (D[m].cpu().numpy(), v[m].cpu().numpy())
    return out

def stats(pairs):
    """[(D (n,2), v (n,2))] 를 모아 r · π · c 와 bootstrap CI."""
    D = np.concatenate([p[0] for p in pairs]) if pairs else np.zeros((0, 2)); V = np.concatenate([p[1] for p in pairs]) if pairs else np.zeros((0, 2))
    return D, V

MIN_N = 10 if a.validate else 20
def summarize(D, V, k, rng):
    if len(D) < MIN_N:
        return None
    rho = (D * V).sum(1) / (k * (V ** 2).sum(1))
    pi = (np.linalg.norm(D - k * V, axis=1) < np.linalg.norm(D, axis=1)).astype(float)
    cosv = (D * V).sum(1) / (np.linalg.norm(D, axis=1) * np.linalg.norm(V, axis=1) + 1e-9)
    bs = [np.median(rho[rng.integers(0, len(rho), len(rho))]) for _ in range(500)]
    bp = [pi[rng.integers(0, len(pi), len(pi))].mean() for _ in range(500)]
    return {"n": int(len(D)), "r": float(np.median(rho)), "r_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "pi": float(pi.mean()), "pi_ci": [float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))], "c": float(np.median(cosv))}

def report(name, per_k, rng, excl=None):
    line = f"  {name:22s}" + (f" 제외 {excl:4.1f}%" if excl is not None else "")
    for k in KS:
        D, V = per_k.get(k, (np.zeros((0, 2)), np.zeros((0, 2))))
        s = summarize(D, V, k, rng)
        line += f" | k{k} " + ("—" if s is None else f"r {s['r']:.2f}[{s['r_ci'][0]:.2f},{s['r_ci'][1]:.2f}] π {s['pi']:.2f} c {s['c']:.2f} n{s['n']}")
    print(line, flush=True)

rng = np.random.default_rng(0)
if a.validate:
    from PIL import Image
    IDX = list(range(0, 96, 3))[:32]
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv")))
            if r["variant"] in ("pos_a", "pos_b") and r["occlusion"] == "visible" and r["motion"] != "flat_static"]
    by = collections.defaultdict(lambda: collections.defaultdict(list)); err = collections.defaultdict(list)
    for r in rows:
        d = os.path.join(a.vobench, r["file_name"])
        fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in IDX]
        x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 255.0; W0 = x.shape[-1]
        x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
        H = encode((x - MEAN) / STD)
        lab = {q["f"]: q for q in json.load(open(os.path.join(os.path.dirname(d), "label.json")))["labels"]["A"]}
        P = [np.array([np.mean([lab[IDX[2 * b + t]]["v"] for t in (0, 1)]), np.mean([lab[IDX[2 * b + t]]["u"] for t in (0, 1)])]) / W0 * G for b in range(16)]
        b0 = 8
        r0, c0 = int(np.clip(round(P[b0 - 1][0]), 1, G - 2)), int(np.clip(round(P[b0 - 1][1]), 1, G - 2))
        cells = [(r0 + dr) * G + (c0 + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]   # 물체 3×3
        res = measure(H, b0, torch.tensor(cells, device=dev), cam=False)    # VOBench 는 카메라 고정
        if res is None:
            continue
        vgt = P[b0] - P[b0 - 1]
        for k, (D, V) in res.items():
            if len(D):
                by[r["motion"]][k].append((D, V))
                err[k].extend(np.linalg.norm(D - (P[b0 + k] - P[b0])[None], axis=1).tolist())   # D 는 b0 → b0+k
            # GT 기준 r (추적 없이) — 운동 종류별 기대값 확인용
            Dg = (P[b0 + k] - P[b0])[None]; by["GT:" + r["motion"]][k].append((Dg, vgt[None]))
    print("=== 추적기 검증 (VOBench visible, 물체 중심 칸, b0 = 8) ===")
    print("  추적 D_k vs GT 변위 오차 (칸, 통과한 궤적): " + " ".join(f"k{k} {np.median(err[k]):.2f} (n{len(err[k])})" for k in KS if err[k]))
    for mo in sorted(m for m in by if not m.startswith("GT:")):
        report(f"추적 {mo}", {k: stats(v) for k, v in by[mo].items()}, rng)
        report(f"  GT {mo}", {k: stats(v) for k, v in by["GT:" + mo].items()}, rng)
    sys.exit(0)

# ---- 학습 데이터 ------------------------------------------------------------------------------
import decord
if "--shift" in sys.argv or True:
    pass
decord.bridge.set_bridge("torch")
TABLES = {"k710": "data_csv/k710_320/fps4_table.tsv", "ssv2": "data_csv/ssv2/fps4_table.tsv", "how": "data_csv/ht100m_1pct/fps4_table.tsv"}
NF_OF = {"k710": 18, "ssv2": 12, "how": 18}                              # 9 블록 (SSv2 는 짧아 6 블록 → k ≤ 4)
# 2026-10-18 v2: 자연 무늬 검증 — K710 프레임 하나를 블록마다 s 픽셀씩 밀어 만든 합성 이동 (진짜 r = 1, 카메라 보정 끔)
_rows = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(REPO, TABLES["k710"]))]
random.Random(1).shuffle(_rows)
for spx in (8, 16, 24):
    pairs = collections.defaultdict(list); got = 0
    for r in _rows:
        if got >= 30:
            break
        try:
            fr = decord.VideoReader(r[0], num_threads=4)[0].float() / 255.0              # (H, W, 3)
        except Exception:
            continue
        img = fr.permute(2, 0, 1); h0, w0 = img.shape[-2:]; s_ = 256 / min(h0, w0)
        img = F.interpolate(img[None], size=(round(h0 * s_), round(w0 * s_)), mode="bilinear", align_corners=False)[0]
        need = 256 + spx * 9
        if img.shape[-1] < need:
            img = F.interpolate(img[None], size=(round(img.shape[-2] * need / img.shape[-1]) + 1, need), mode="bilinear", align_corners=False)[0]
        y0 = (img.shape[-2] - 256) // 2
        seq = [img[:, y0:y0 + 256, b * spx: b * spx + 256] for b in range(9) for _ in (0, 1)]   # 블록 안 2 장은 같은 그림
        x = torch.stack(seq, 1)
        H = encode((x - MEAN) / STD)
        res = measure(H, 1, torch.arange(S, device=dev), cam=False)
        got += 1
        for k, (D, V) in res.items():
            if len(D):
                pairs[k].append((-D, -V))                                 # 창이 오른쪽으로 → 내용은 왼쪽으로. 부호를 맞춰 "진짜 r = 1"
    report(f"합성 이동 {spx}px ({spx / 16:.1f}칸/블록)", {k: stats(v) for k, v in pairs.items()}, rng)
summary = {}
for name, tp in TABLES.items():
    rows = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(REPO, tp))]
    NF = NF_OF[name]
    rows = [r for r in rows if int(r[2]) >= int(r[3]) * (NF - 1) + 1]
    random.Random(0).shuffle(rows)
    acc = collections.defaultdict(list); spd = collections.defaultdict(lambda: collections.defaultdict(list)); n_ok = n_ex = 0
    for r in rows:
        if n_ok + n_ex >= a.n:
            break
        path, fps, nfr, st = r[0], float(r[1]), int(r[2]), int(r[3])
        try:
            vr = decord.VideoReader(path, num_threads=4)
            start = random.Random(path).randint(0, max(0, len(vr) - 1 - st * (NF - 1)))
            fr = vr.get_batch([start + st * j for j in range(NF)]).float() / 255.0     # (T, H, W, 3)
        except Exception:
            continue
        x = fr.permute(3, 0, 1, 2)
        h0, w0 = x.shape[-2:]; s_ = 256 / min(h0, w0)
        x = F.interpolate(x.transpose(0, 1), size=(max(256, round(h0 * s_)), max(256, round(w0 * s_))), mode="bilinear", align_corners=False).transpose(0, 1)
        hh, ww = x.shape[-2:]; x = x[:, :, (hh - 256) // 2:(hh - 256) // 2 + 256, (ww - 256) // 2:(ww - 256) // 2 + 256]
        H = encode((x - MEAN) / STD)
        res = measure(H, 1, torch.arange(S, device=dev))                  # 시작 = 블록 0 의 모든 칸
        if res is None:
            n_ex += 1; continue
        n_ok += 1
        for k, (D, V) in res.items():
            if len(D):
                acc[k].append((D, V))
                sp = np.linalg.norm(V, axis=1)
                for lo, hi, lb in ((0.5, 1, "0.5–1"), (1, 2, "1–2"), (2, 99, "≥2")):
                    m = (sp >= lo) & (sp < hi)
                    if m.any():
                        spd[lb][k].append((D[m], V[m]))
    print(f"\n=== {name} (clip {n_ok}, 카메라 제외 {n_ex}) ===")
    report("전체", {k: stats(v) for k, v in acc.items()}, rng, excl=100 * n_ex / max(1, n_ok + n_ex))
    for lb in ("0.5–1", "1–2", "≥2"):
        report(f"속도 {lb}", {k: stats(v) for k, v in spd[lb].items()}, rng)
    summary[name] = {str(k): summarize(*stats(v), k, rng) for k, v in acc.items()}
json.dump(summary, open(os.path.join(a.out, "motion_persistence.json"), "w"), indent=1)
