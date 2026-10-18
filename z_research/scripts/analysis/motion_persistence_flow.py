#!/usr/bin/env python3
"""운동 지속성 측정 — 픽셀 추적판 (2026-10-18). 학습 데이터에서 "관성" 이 얼마나 맞나. GPU · 모델 불필요 (OpenCV).

  /venv/vjepa/bin/python z_research/scripts/analysis/motion_persistence_flow.py --validate    # 합성 이동 + VOBench GT 로 검증
  /venv/vjepa/bin/python z_research/scripts/analysis/motion_persistence_flow.py --n 200        # K710 · SSv2 · HowTo

토큰 추적판 (motion_persistence.py) 은 폐기: ViT 토큰이 16 px 격자에 묶여 칸 사이 이동 (0.5 · 1.5 칸) 에서 r 이 −0.02 · 0.54 로 틀렸다
  (정확히 1 칸일 때만 1.00). 자연 영상 움직임은 대부분 칸 사이라 쓸 수 없다.
방법: 모델과 같은 화면 (짧은 변 256 → 가운데 256 crop), 블록마다 첫 프레임 (블록 = 학습 간격의 2 장).
  블록 b0-1 에서 Shi-Tomasi 특징점 → 피라미드 Lucas-Kanade 로 블록마다 추적 (픽셀 이하 정밀도)
  필터: LK status · 매 걸음 왕복 오차 < 0.5 px · 관측 속도 ‖v‖ ≥ 8 px (= 0.5 칸) · 카메라: 블록 쌍마다 전체 점 변위 중앙값 g 를 빼고 ‖g‖ > 16 px 면 clip 제외
  v = 블록 b0-1 → b0 한 걸음 (카메라 보정), D_k = b0 → b0+k 누적 (보정)
  ρ_k = (D_k · v) / (k ‖v‖²) → r_k = median (1 등속, 0 멈춤) · π_k = mean 1(‖D_k − k v‖ < ‖D_k‖) · c_k = median cos(D_k, v)
간격: 학습과 같게 데이터셋마다 fps4 표의 step. 블록 9 개 (SSv2 는 6 개). VOBench 는 채점 간격 (30 fps 3 프레임, 블록 0.2 s)."""
import argparse, csv, json, os, sys, random, collections
import numpy as np, cv2
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
ap = argparse.ArgumentParser()
ap.add_argument("--validate", action="store_true")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--vobench", default="/workspace/dataset/benchmarks/VOBench_v1")
ap.add_argument("--out", default=os.path.join(REPO, "z_research/VOBench_v1/figures/brief_diag"))
a = ap.parse_args()
KS = (1, 3, 5, 7)
LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def view256(img):
    """RGB/BGR uint8 (H, W, 3) → gray 256×256 (짧은 변 256, 가운데 crop) — 모델 입력과 같은 화면."""
    h, w = img.shape[:2]; s = 256 / min(h, w)
    img = cv2.resize(img, (max(256, round(w * s)), max(256, round(h * s))), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]; y0, x0 = (h - 256) // 2, (w - 256) // 2
    g = img[y0:y0 + 256, x0:x0 + 256]
    return cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) if g.ndim == 3 else g

SURV = []          # (시작 점 수, 끝까지 남은 점 수) — 생존 편향 확인용
def track(frames, b0, mask=None, cam=True, sub=1):
    """frames: gray (N, 256, 256). 블록 b 의 프레임 = frames[b * sub] (sub = 블록 사이 촘촘한 추적 간격 수).
    2026-10-18 v3: 블록 간격 (학습 0.47 s) 이 점 추적에 너무 길어 1–5 걸음 안에 점이 다 사라지고 정지 배경만 남았다 (생존 편향, r ≈ 0).
    → 블록 사이 원본 프레임을 촘촘히 따라가며 추적하고 위치는 블록 시점에서만 뽑는다. return {k: (D, v)} 또는 None (카메라 제외)."""
    T = (len(frames) - 1) // sub + 1
    p = cv2.goodFeaturesToTrack(frames[(b0 - 1) * sub], maxCorners=400, qualityLevel=0.01, minDistance=5, mask=mask)
    if p is None or len(p) < 5:
        return {}
    cur = p.reshape(-1, 2); pts = [cur]; ok = np.ones(len(p), bool)
    for f in range((b0 - 1) * sub, (T - 1) * sub):
        q, st, _ = cv2.calcOpticalFlowPyrLK(frames[f], frames[f + 1], cur.reshape(-1, 1, 2).astype(np.float32), None, **LK)
        r, st2, _ = cv2.calcOpticalFlowPyrLK(frames[f + 1], frames[f], q, None, **LK)
        fb = np.linalg.norm(r.reshape(-1, 2) - cur, axis=1)
        ok &= (st.flatten() == 1) & (st2.flatten() == 1) & (fb < 0.5)
        cur = q.reshape(-1, 2)
        if (f + 1) % sub == 0:
            pts.append(cur)
    SURV.append((len(p), int(ok.sum())))
    P = np.stack(pts)                                                    # (T-b0+1, n, 2): 인덱스 0 = 블록 b0-1
    disp = P[1:] - P[:-1]
    if cam:
        g = np.array([np.median(d[ok], axis=0) if ok.sum() >= 5 else np.zeros(2) for d in disp])
        if (np.linalg.norm(g, axis=1) > 16).any():
            return None
    else:
        g = np.zeros((len(disp), 2))
    v = disp[0] - g[0]
    keep = ok & (np.linalg.norm(v, axis=1) >= 8)
    out = {}
    for k in KS:
        if k >= len(P) - 1 + 1 or 1 + k >= len(P):
            continue
        D = (P[1 + k] - P[1]) - g[1:1 + k].sum(0)
        out[k] = (D[keep], v[keep])
    return out

def summarize(D, V, k, rng, min_n=20):
    if len(D) < min_n:
        return None
    rho = (D * V).sum(1) / (k * (V ** 2).sum(1))
    pi = (np.linalg.norm(D - k * V, axis=1) < np.linalg.norm(D, axis=1)).astype(float)
    cs = (D * V).sum(1) / (np.linalg.norm(D, axis=1) * np.linalg.norm(V, axis=1) + 1e-9)
    bs = [np.median(rho[rng.integers(0, len(rho), len(rho))]) for _ in range(500)]
    bp = [pi[rng.integers(0, len(pi), len(pi))].mean() for _ in range(500)]
    return {"n": int(len(D)), "r": float(np.median(rho)), "r_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "pi": float(pi.mean()), "pi_ci": [float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))], "c": float(np.median(cs))}

def cat(pairs):
    if not pairs:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])

def report(name, per_k, rng, extra="", min_n=20):
    line = f"  {name:26s}{extra}"
    res = {}
    for k in KS:
        s = summarize(*cat(per_k.get(k, [])), k, rng, min_n)
        res[k] = s
        line += f" | k{k} " + ("—" if s is None else f"r {s['r']:.2f}[{s['r_ci'][0]:.2f},{s['r_ci'][1]:.2f}] π {s['pi']:.2f} c {s['c']:.2f} n{s['n']}")
    print(line, flush=True)
    return res

rng = np.random.default_rng(0)
import decord
TABLES = {"k710": "data_csv/k710_320/fps4_table.tsv", "ssv2": "data_csv/ssv2/fps4_table.tsv", "how": "data_csv/ht100m_1pct/fps4_table.tsv"}

if a.validate:
    print("=== 검증 1: 합성 이동 (K710 프레임을 블록마다 s px 밀기, 진짜 r = 1, 카메라 보정 끔) ===")
    rows = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(REPO, TABLES["k710"]))]; random.Random(1).shuffle(rows)
    imgs = []
    for r in rows:
        if len(imgs) >= 30:
            break
        try:
            imgs.append(decord.VideoReader(r[0], num_threads=4)[0].asnumpy())
        except Exception:
            pass
    for spx in (4, 8, 12, 16, 24):
        acc = collections.defaultdict(list)
        for im in imgs:
            h, w = im.shape[:2]; s = 256 / min(h, w); im2 = cv2.resize(im, (round(w * s) + spx * 10, round(h * s)))
            y0 = (im2.shape[0] - 256) // 2
            frames = [cv2.cvtColor(im2[y0:y0 + 256, b * spx: b * spx + 256], cv2.COLOR_RGB2GRAY) for b in range(9)]
            for k, (D, V) in track(frames, 1, cam=False).items():
                acc[k].append((D, V))
        report(f"합성 {spx}px ({spx / 16:.2f}칸/블록)", acc, rng)
    print("\n=== 검증 2: VOBench visible (물체 주변 점, GT r 과 비교) ===")
    from PIL import Image
    IDX = list(range(0, 96, 3))[:32]
    vrows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv")))
             if r["variant"] in ("pos_a", "pos_b") and r["occlusion"] == "visible" and r["motion"] != "flat_static"]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in vrows:
        d = os.path.join(a.vobench, r["file_name"])
        ims = [np.asarray(Image.open(f"{d}/{IDX[2 * b]:06d}.png").convert("RGB")) for b in range(16)]
        W0 = ims[0].shape[1]
        frames = [cv2.cvtColor(cv2.resize(im, (256, 256)), cv2.COLOR_RGB2GRAY) for im in ims]
        lab = {q["f"]: q for q in json.load(open(os.path.join(os.path.dirname(d), "label.json")))["labels"]["A"]}
        P = [np.array([lab[IDX[2 * b]]["u"], lab[IDX[2 * b]]["v"]]) / W0 * 256 for b in range(16)]     # (x, y) px
        b0 = 8; m = np.zeros((256, 256), np.uint8)
        cv2.circle(m, (int(P[b0 - 1][0]), int(P[b0 - 1][1])), 24, 255, -1)
        for k, (D, V) in track(frames, b0, mask=m, cam=False).items():
            by[r["motion"]][k].append((D, V))
        vg = P[b0] - P[b0 - 1]
        for k in KS:
            by["GT:" + r["motion"]][k].append(((P[b0 + k] - P[b0])[None], vg[None]))
    for mo in sorted(x for x in by if not x.startswith("GT:")):
        report(f"추적 {mo}", by[mo], rng, min_n=10); report(f"   GT {mo}", by["GT:" + mo], rng, min_n=10)
    sys.exit(0)

summary = {}
NB = {"k710": 9, "ssv2": 6, "how": 9}
for name, tp in TABLES.items():
    rows = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(REPO, tp))]
    nb = NB[name]
    rows = [r for r in rows if int(r[2]) > int(r[3]) * 2 * (nb - 1)]
    random.Random(0).shuffle(rows)
    acc = collections.defaultdict(list); spd = collections.defaultdict(lambda: collections.defaultdict(list)); n_ok = n_ex = 0
    for r in rows:
        if n_ok + n_ex >= a.n:
            break
        path, st = r[0], int(r[3])
        try:
            vr = decord.VideoReader(path, num_threads=4)
            span = st * 2 * (nb - 1)
            start = random.Random(path).randint(0, max(0, len(vr) - 1 - span))
            sub = max(1, (st * 2) // 2)                                  # 블록 사이를 원본 2 프레임마다 (SSv2 12 fps 는 매 프레임에 가깝게)
            stride = (st * 2) // sub
            frames = [view256(f) for f in vr.get_batch([start + stride * j for j in range(sub * (nb - 1) + 1)]).asnumpy()]
        except Exception:
            continue
        res = track(frames, 1, sub=sub)
        if res is None:
            n_ex += 1; continue
        n_ok += 1
        for k, (D, V) in res.items():
            if len(D):
                acc[k].append((D, V)); sp = np.linalg.norm(V, axis=1) / 16
                for lo, hi, lb in ((0.5, 1, "0.5–1칸"), (1, 2, "1–2칸"), (2, 99, "≥2칸")):
                    mm = (sp >= lo) & (sp < hi)
                    if mm.any():
                        spd[lb][k].append((D[mm], V[mm]))
    sv = np.array(SURV[-n_ok - n_ex:]) if SURV else np.zeros((1, 2))
    print(f"\n=== {name} (clip {n_ok}, 카메라 제외 {n_ex} = {100 * n_ex / max(1, n_ok + n_ex):.1f}%, 블록 {nb}, "
          f"추적점 생존 {100 * sv[:, 1].sum() / max(1, sv[:, 0].sum()):.0f}%) ===")
    summary[name] = {str(k): v for k, v in report("전체", acc, rng).items()}
    for lb in ("0.5–1칸", "1–2칸", "≥2칸"):
        report(f"속도 {lb}", spd[lb], rng)
json.dump(summary, open(os.path.join(a.out, "motion_persistence_flow.json"), "w"), indent=1)
