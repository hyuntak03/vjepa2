#!/usr/bin/env python3
"""VOBench 가림 판독 — 물체가 보이는 블록 / 완전히 가린 블록에서 물체 있음 · 모양 · 색이 읽히나 (2026-10-29, 사용자 요청).

사용자: "폴더 새로 만들어서 확실하게 probing 실험. (N, D) 단위, attention pooling 하면 (1, D) 로 읽어야. k=1 은 제외 —
         2 장이 1 블록이니 2 장 모두 가려진 블록을 읽어야 완벽히 가려진 것".

표현 (릴리즈 V-JEPA 2 ViT-H, 32 장 · 3 장 간격, 문맥 16 장 = 8 블록 · 미래 8 블록, bf16 — probing 관례).
  캐시: /workspace/auto_research_cache/vobench_attnprobe_release/ (vobench_attnprobe_release.py extract · extract_empty 가 만든 것)
  z = LN(문맥 encoder(문맥 16 장)) 의 문맥 블록 · p = predictor 출력 (1280, 출력 투영 뒤) 의 미래 블록 · h = LN(target encoder(32 장)) 의 미래 블록
판독기 입력 = 블록 하나의 토큰 (256, 1280) → attention pooling (질의 1 개, 16 head, cross-attention 블록 1 개) → (1, 1280) → 선형 분류.
블록 범주 = VOBench 장별 가림 기록 (metadata samples) 의 그 블록 두 장: '..' 보임 · 'HH' 완전 가림 · 나머지 (반쯤) 는 버림.
  z 는 문맥 절반, p · h 는 미래 절반의 기록을 쓴다 (p 의 가린 블록 = 실제 미래에서 두 장 다 가린 블록 = late k2·k3 의 미래 블록 1).
영상: k 2·3 (early · mid · late 전부). 목표: obj = 물체 있음 (permanence block 의 pos_a) vs 빈 장면 (pos_b, 같은 장면 · 같은 가림판) /
  shape (6) · color (8) = 물체 영상의 가능 변이 (pos_a · pos_b, 불가능 변이는 쓰지 않는다 — CLAUDE.md §1-4).
맞춤:
  - 보임 판독기 · 가림 판독기를 따로 학습하고 각자 자기 범주에서만 시험한다 (정보 존재)
  - 학습 표본 수를 모든 표현 · 범주에서 같게: obj 36 · shape 90 · color 90 (p · h 의 가린 블록이 허락하는 최대)
  - 위치 맞춤: 가린 블록 하나마다 같은 블록 위치의 보이는 블록 하나를 짝지어 학습 · 시험 (비복원)
  - 학습 걸음 600 (묶음 16, AdamW 1e-3 one-cycle, wd 0.05, bf16) 고정
  - block 단위 5-fold (조건 × 위반 종류 비율 유지), 씨앗 3 개 (fold 배정 · 표본 뽑기 · 초기화가 바뀐다)
  - 대조: 라벨 섞기 (씨앗 0) — 우연 수준이 나와야 한다 · 다수 클래스 기준선
이식 (z → p): z 의 보임 / 가림 판독기를 p 의 같은 범주 블록 (p 의 시험 묶음과 같은 것) 에 그대로 건다.

  for j in 0 1 2 3 4 5 6 7 8; do CUDA_VISIBLE_DEVICES=$((j % 8)) /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/vobench_occlusion_probe.py probe --job $j & done; wait
  /venv/vjepa/bin/python z_research/scripts/analysis/vobench_occlusion_probe.py report
출력: z_analysis/vobench_occlusion_probe/results/  (그림은 z_research/scripts/figures/plot_vobench_occlusion_probe.py)
"""
import argparse, collections, csv, json, os, sys
import numpy as np, torch, torch.nn as nn
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
ap = argparse.ArgumentParser()
ap.add_argument("phase", choices=["probe", "report", "probe_t_vel", "report_t_vel", "probe_cf", "report_cf", "probe_pool", "report_pool"])
ap.add_argument("--job", type=int, default=0, help="0–8 = (z, p, h) × (obj, shape, color) / 9–17 = (z, p, h) × (vel, acc, acc3)")
ap.add_argument("--focus", default="all", choices=["all", "boundary"],
                help="boundary = z 를 문맥 마지막 블록 하나로 (2026-10-29). late k2·k3 은 문맥 마지막 블록과 미래 첫 블록이 함께 가려져 "
                     "z (마지막 문맥 블록) · p · h (미래 첫 블록) 가 같은 영상 · 같은 표본 수 · 위치 하나로 맞춰진다")
a = ap.parse_args()
CACHE = "/workspace/auto_research_cache/vobench_attnprobe_release"
OUT = os.path.join(REPO, "z_analysis/vobench_occlusion_probe/results"); os.makedirs(OUT, exist_ok=True)
META = "/workspace/dataset/benchmarks/VOBench_v1/metadata.csv"
NB, S, D = 8, 256, 1280
SHAPES = ["capsule", "cone", "cube", "cylinder", "sphere", "torus"]
COLORS = ["blue", "cyan", "green", "magenta", "orange", "purple", "red", "yellow"]
NCLS = {"obj": 2, "shape": len(SHAPES), "color": len(COLORS), "acc3": 3, "vel": 2, "acc": 2}   # vel · acc 는 회귀 (출력 2 = 화면 u, v)
NTRAIN = {"obj": 36, "shape": 90, "color": 90, "vel": 180, "acc": 180, "acc3": 90}
REG = ("vel", "acc")
# 2026-10-29 속도 · 가속도 추가 (사용자 "velocity 나 acceleration 도 읽어봐. 표본 잘 해서"):
#   라벨 = VOBench label.json 의 장별 물체 화면 위치 (u, v, 512 px, 가려진 장도 실제 위치가 있다) 에 블록 중심 ±6 장 (원본 30 fps) 2 차 곡선을 맞춰
#   블록 중심의 속도 (px/s) · 가속도 (px/s²), 256 px 기준. 같은 운동이면 영상끼리 값이 거의 같다 (운동 6 종이 6 무리) → 회귀 R² 는 사실상 운동 구분.
#   그래서 acc3 = 평지 등속 · 가속 · 감속 (같은 순간 속도 ≈ 61 px/s) 만으로 "빨라지나 / 그대로 / 느려지나" 3 지선다 — 순수 가속도 판독.
#   표본: realistic 물체 영상도 쓴다 (모양 · 색 라벨이 필요 없다) → 가린 미래 블록 120 → 240 개. 캐시 *_real (vobench_attnprobe_release.py extract_real)
ROOT = "/workspace/dataset/benchmarks/VOBench_v1"
ACC3 = {"flat_vel": 0, "flat_acc": 1, "flat_dec": 2}
SEEDS, STEPS, BS = (0, 1, 2), 600, 16


def videos(tgt):
    """return [(키, 원본 ('M' 물체 영상 / 'E' 빈 장면), 캐시 행, 라벨, meta)] — k 2·3 만."""
    M = json.load(open(os.path.join(CACHE, "meta.json"))); Me = json.load(open(os.path.join(CACHE, "meta_empty.json")))
    if tgt in REG or tgt == "acc3":
        Mr = json.load(open(os.path.join(CACHE, "meta_real.json")))
        V = [(m["video_id"], src, i, ACC3.get(m["motion"], -1), m) for src, MM in (("M", M), ("R", Mr)) for i, m in enumerate(MM) if m["k"] in (2, 3)]
        return [v for v in V if v[4]["motion"] in ACC3] if tgt == "acc3" else V
    if tgt == "obj":
        V = [(m["video_id"], "M", i, 1, m) for i, m in enumerate(M) if m["violation"] == "permanence" and m["k"] in (2, 3)] + \
            [(m["video_id"], "E", i, 0, m) for i, m in enumerate(Me) if m["k"] in (2, 3)]
    else:
        V = [(m["video_id"], "M", i, m[tgt], m) for i, m in enumerate(M) if m["k"] in (2, 3)]
    return V


def categories(V, half):
    md = {r["file_name"]: r["samples"].split("|")[half] for r in csv.DictReader(open(META))}
    f = lambda pr: "visible" if pr == ".." else "occluded" if pr == "HH" else "mixed"
    return [[f(md[v[4]["file_name"]][2 * b:2 * b + 2]) for b in range(NB)] for v in V]


def folds(V, seed):
    """block 단위 5-fold. 조건 (가림 시점 × k) × 위반 종류 안에서 섞어 나눈다."""
    rng = np.random.default_rng(1000 + seed); by = collections.defaultdict(set)
    for v in V:
        m = v[4]; by[(m["occlusion"], m["k"], m["violation"])].add(m["block_id"])
    out = {}
    for key in sorted(by):
        for j, b in enumerate(rng.permutation(sorted(by[key], key=int))):
            out[b] = j % 5
    return out


def load(rep, V, dev):
    A = {k_: np.load(os.path.join(CACHE, f"{rep}{sfx_}.npy"), mmap_mode="r") for k_, sfx_ in (("M", ""), ("E", "_empty"), ("R", "_real"))
         if any(v[1] == k_ for v in V)}
    return torch.from_numpy(np.stack([A[v[1]][v[2]] for v in V])).to(dev).view(len(V) * NB, S, D)


def motion_labels(V, half, tgt):
    """(영상 × 블록, 2) 속도 (px/s) 또는 가속도 (px/s²), 256 px 기준. 블록 b 의 원본 장 = 6 (b + 8·half), +3."""
    out = np.zeros((len(V) * NB, 2), np.float32)
    for j, v in enumerate(V):
        fn = v[4]["file_name"]; arm = os.path.basename(fn)
        L = json.load(open(os.path.join(ROOT, os.path.dirname(fn), "label.json")))["labels"]["A" if arm == "pos_a" else "B"]
        T = np.array([[e["u"], e["v"]] for e in L], float)
        for b in range(NB):
            raw0 = 6 * (b + NB * half); tc = raw0 + 1.5; t = np.arange(max(0, raw0 - 6), min(len(T), raw0 + 10))
            c = np.linalg.lstsq(np.stack([np.ones_like(t), t - tc, (t - tc) ** 2], 1).astype(float), T[t], rcond=None)[0]
            out[j * NB + b] = c[1] * 30 / 2 if tgt == "vel" else 2 * c[2] * 900 / 2
    return out


def fit(X, y, tr, ncls, seed, dev, steps=None):
    steps = steps or STEPS
    from src.models.attentive_pooler import AttentiveClassifier
    torch.manual_seed(seed)
    clf = AttentiveClassifier(embed_dim=D, num_heads=16, mlp_ratio=4.0, depth=1, num_classes=ncls).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    trt = torch.tensor(tr, device=dev); g = torch.Generator(device=dev).manual_seed(seed); perm, k = None, 0
    reg = y.is_floating_point()
    if reg:                                                         # 회귀: 학습 표본의 평균 · 표준편차로 표준화
        clf.mu = y[trt].mean(0); clf.sd = y[trt].std(0) + 1e-6
    clf.train()
    for _ in range(steps):
        if perm is None or k + BS > len(tr):
            perm, k = trt[torch.randperm(len(tr), device=dev, generator=g)], 0
        ib = perm[k:k + BS]; k += BS
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = clf(X[ib].float())
        loss = nn.functional.mse_loss(out.float(), (y[ib] - clf.mu) / clf.sd) if reg else nn.functional.cross_entropy(out.float(), y[ib])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    clf.eval(); clf.reg = reg
    return clf


def predict(clf, X, ii, dev):
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s0 in range(0, len(ii), 64):
            o = clf(X[torch.tensor(ii[s0:s0 + 64], device=dev)].float()).float()
            out.append((o * clf.sd + clf.mu).cpu().numpy() if clf.reg else o.argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, int)


def score(rr):
    """분류 = 정확도 %, 회귀 = R² % (u · v 합친 1 − SSE / SST, SST 는 그 시험 묶음 평균 기준)."""
    if not rr:
        return float("nan")
    if isinstance(rr[0]["y"], list):
        Y = np.array([r["y"] for r in rr]); P = np.array([r["pred"] for r in rr])
        return 100 * float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
    return 100 * float(np.mean([r["pred"] == r["y"] for r in rr]))


def paired(occ, vis_pool, rng):
    """가린 표본 하나마다 같은 위치의 보이는 표본 하나 (비복원). 짝이 없으면 그 가린 표본도 뺀다."""
    pool = collections.defaultdict(list)
    for t in vis_pool:
        pool[t % NB].append(t)
    for b in pool:
        rng.shuffle(pool[b])
    o2, v2 = [], []
    for t in occ:
        if pool[t % NB]:
            o2.append(t); v2.append(pool[t % NB].pop())
    return o2, v2


def split(V, cat, fo, fold, n_train, rng, only_b=None):
    """return 학습 (가림, 보임) · 시험 (가림, 보임). 시험 = 그 fold 의 가린 블록 전부 + 위치 짝 보이는 블록. only_b = 쓸 블록 위치 하나로 제한."""
    ids = lambda tr, c: [j * NB + b for j, v in enumerate(V) if (fo[v[4]["block_id"]] != fold) == tr for b in range(NB)
                         if cat[j][b] == c and (only_b is None or b == only_b)]
    occ_tr = ids(True, "occluded"); rng.shuffle(occ_tr); occ_tr = occ_tr[:n_train]
    occ_tr, vis_tr = paired(occ_tr, ids(True, "visible"), rng)
    occ_te, vis_te = paired(ids(False, "occluded"), ids(False, "visible"), rng)
    return occ_tr, vis_tr, occ_te, vis_te


if a.phase == "probe":
    rep, tgt = ([(r, t) for r in ("z", "p", "h") for t in ("obj", "shape", "color")] + [(r, t) for r in ("z", "p", "h") for t in ("vel", "acc", "acc3")])[a.job]
    dev = torch.device("cuda"); V = videos(tgt); ncls = NCLS[tgt]; reg = tgt in REG
    cat = categories(V, 0 if rep == "z" else 1)
    lab = lambda half: (torch.tensor(motion_labels(V, half, tgt), device=dev) if reg
                        else torch.tensor([v[3] for v in V], device=dev).repeat_interleave(NB))
    X = load(rep, V, dev); y = lab(0 if rep == "z" else 1)
    if rep == "z":                                                  # 이식 대상: p 와 그 범주 · 라벨 (미래 절반)
        XP = load("p", V, dev); cat_p = categories(V, 1); yP = lab(1)
    # 대조 (운동 목표만, 2026-10-29): 같은 장면 · 같은 가림판에서 물체만 없는 영상 (primitive permanence 의 pos_b). 라벨은 같은 block 의
    #   물체 영상 (pos_a) 의 운동. 물체가 없는데도 읽히면 판독기는 물체가 아니라 장면 (경사면 · 가림판 자리 등) 에서 운동을 읽은 것이다.
    VE = None
    if tgt in REG or tgt == "acc3":
        Me = json.load(open(os.path.join(CACHE, "meta_empty.json")))
        posa = {v[4]["block_id"]: j for j, v in enumerate(V) if v[1] == "M" and v[4]["violation"] == "permanence" and v[4]["variant"] == "pos_a"}
        VE = [(m["video_id"], "E", i, ACC3.get(m["motion"], -1), m) for i, m in enumerate(Me) if m["k"] in (2, 3) and m["block_id"] in posa
              and (tgt != "acc3" or m["motion"] in ACC3)]
        XE = load(rep, VE, dev); cat_e = categories(VE, 0 if rep == "z" else 1)
        yE = y.view(len(V), NB, -1)[[posa[v[4]["block_id"]] for v in VE]].reshape(len(VE) * NB, -1) if reg else \
             torch.tensor([v[3] for v in VE], device=dev).repeat_interleave(NB)
        if not reg:
            yE = yE.view(-1)
    rec, log = [], []
    for seed in SEEDS:
        fo = folds(V, seed)
        for fold in range(5):
            rng = np.random.default_rng(seed * 100 + fold)
            occ_tr, vis_tr, occ_te, vis_te = split(V, cat, fo, fold, NTRAIN[tgt], rng, only_b=NB - 1 if (rep == "z" and a.focus == "boundary") else None)
            if rep == "z":                                          # p 의 시험 묶음은 p job 과 같은 rng 순서로 만든다
                _, _, p_occ_te, p_vis_te = split(V, cat_p, fo, fold, NTRAIN[tgt], np.random.default_rng(seed * 100 + fold))
            for c, tr, te in (("visible", vis_tr, vis_te), ("occluded", occ_tr, occ_te)):
                ytr = y[torch.tensor(tr, device=dev)].cpu().numpy()
                maj = ytr.mean(0).tolist() if reg else int(np.bincount(ytr, minlength=ncls).argmax())   # 회귀: 학습 평균 (기준선)
                clf = fit(X, y, tr, ncls, seed * 10 + fold, dev)
                ptr = predict(clf, X, tr, dev)
                tra = float(1 - ((ptr - ytr) ** 2).sum() / ((ytr - ytr.mean(0)) ** 2).sum()) if reg else float((ptr == ytr).mean())
                def add(panel, ii, q, Y):
                    Yc = Y.cpu().numpy()
                    for t_, qq in zip(ii, q):
                        j, b = divmod(t_, NB)
                        rec.append(dict(seed=seed, fold=fold, panel=panel, cat=c, vid=V[j][0], block_id=V[j][4]["block_id"], b=b,
                                        motion=V[j][4]["motion"], y=(Yc[t_].tolist() if reg else int(Yc[t_])),
                                        pred=(np.asarray(qq).tolist() if reg else int(qq)), maj=maj))
                add("self", te, predict(clf, X, te, dev), y)
                if rep == "z":
                    tp = p_vis_te if c == "visible" else p_occ_te
                    add("z2p", tp, predict(clf, XP, tp, dev), yP)
                if VE is not None:                                  # 빈 장면 대조: 시험 fold 의 빈 영상, 같은 범주 · 같은 위치
                    pos_ok = {t_ % NB for t_ in te}
                    te_e = [j * NB + b for j, v in enumerate(VE) if fo[v[4]["block_id"]] == fold for b in range(NB) if cat_e[j][b] == c and b in pos_ok]
                    if te_e:
                        q = predict(clf, XE, te_e, dev); Yc = yE.cpu().numpy()
                        for t_, qq in zip(te_e, q):
                            j, b = divmod(t_, NB)
                            rec.append(dict(seed=seed, fold=fold, panel="empty", cat=c, vid=VE[j][0], block_id=VE[j][4]["block_id"], b=b,
                                            motion=VE[j][4]["motion"], y=(Yc[t_].tolist() if reg else int(Yc[t_])),
                                            pred=(np.asarray(qq).tolist() if reg else int(qq)), maj=maj))
                del clf
                if seed == 0:                                       # 대조: 학습 라벨 섞기
                    ysh = y.clone(); ti = torch.tensor(tr, device=dev); ysh[ti] = y[ti][torch.randperm(len(tr), device=dev)]
                    clf = fit(X, ysh, tr, ncls, 999 + fold, dev); add("shuffle", te, predict(clf, X, te, dev), y); del clf
                log.append(dict(seed=seed, fold=fold, cat=c, n_train=len(tr), n_test=len(te), train_acc=tra))
                print(f"[{rep}/{tgt}] seed {seed} fold {fold} {c:8s} 학습 {len(tr):3d} (acc {100 * tra:.0f}) · 시험 {len(te):3d}", flush=True)
    sfx = "_boundary" if (rep == "z" and a.focus == "boundary") else ""
    json.dump(dict(rep=rep, tgt=tgt, focus=a.focus, records=rec, log=log), open(os.path.join(OUT, f"probe_{rep}_{tgt}{sfx}.json"), "w"))
    for panel in sorted({r["panel"] for r in rec}):
        for c in ("visible", "occluded"):
            rr = [r for r in rec if r["panel"] == panel and r["cat"] == c]
            print(f"{rep} {tgt} {panel:7s} {c:8s} {score(rr):5.1f} (n {len(rr)})")
    print("끝")

elif a.phase == "probe_t_vel":
    # 2026-10-29 사용자: "속도 관련 그래프도" — late k1 미래 블록별 shape / color 그림 (vobench_attnprobe_release.py probe_t) 과 같은 틀의 속도판.
    #   --job = 미래 블록 t (0–7). 판독기 입력 = 미래 블록 t 의 토큰 (256, 1280). 영상 = primitive + realistic 물체 영상 전부 (모든 조건 · 모든 k, 1,200 개).
    #   block 단위 5-fold 로 학습 (전 조건), 시험 기록은 전부 남기고 그림은 late k1 (120 개) · visible 을 쓴다. 씨앗 3, 600 걸음.
    #   판독: h→h (encoder, 실제 미래) · p→p (predictor) · h→p (encoder 판독기를 predictor 에). 라벨 = 그 블록 중심 속도 (px/s, 256 px).
    t = a.job; dev = torch.device("cuda")
    M = json.load(open(os.path.join(CACHE, "meta.json"))); Mr = json.load(open(os.path.join(CACHE, "meta_real.json")))
    V = [(m["video_id"], src, i, -1, m) for src, MM in (("M", M), ("R", Mr)) for i, m in enumerate(MM)]
    def load_t(rep):
        A = {k_: np.load(os.path.join(CACHE, f"{rep}{sfx_}.npy"), mmap_mode="r") for k_, sfx_ in (("M", ""), ("R", "_real"))}
        return torch.from_numpy(np.stack([A[v[1]][v[2], t * S:(t + 1) * S] for v in V])).to(dev)   # 미래 블록 t (캐시는 미래 2048 토큰)
    X = {r_: load_t(r_) for r_ in ("h", "p")}
    y = torch.tensor(motion_labels(V, 1, "vel").reshape(len(V), NB, 2)[:, t], device=dev)
    cond_ = lambda m: "visible" if m["occlusion"] == "visible" else f"{m['occlusion']} k{m['k']}"
    rec = []
    for seed in SEEDS:
        fo = folds(V, seed)
        for fold in range(5):
            tr = [j for j, v in enumerate(V) if fo[v[4]["block_id"]] != fold]; te = [j for j, v in enumerate(V) if fo[v[4]["block_id"]] == fold]
            for fs in ("h", "p"):
                clf = fit(X[fs], y, tr, 2, seed * 10 + fold, dev)
                for es in (("h", "p") if fs == "h" else ("p",)):
                    q = predict(clf, X[es], te, dev); Yc = y.cpu().numpy()
                    rec += [dict(seed=seed, fold=fold, panel=f"{fs}->{es}", vid=V[j][0], block_id=V[j][4]["block_id"], cond=cond_(V[j][4]),
                                 objset="primitive" if V[j][1] == "M" else "realistic", y=Yc[j].tolist(), pred=np.asarray(qq).tolist()) for j, qq in zip(te, q)]
                del clf
            print(f"[t{t + 1}] seed {seed} fold {fold} 학습 {len(tr)} 시험 {len(te)}", flush=True)
    od = os.path.join(OUT, "per_t_vel"); os.makedirs(od, exist_ok=True)
    json.dump(rec, open(os.path.join(od, f"t{t + 1}.json"), "w"))
    for pn in ("h->h", "p->p", "h->p"):
        print(f"t{t + 1} {pn} late k1 R² {score([r for r in rec if r['panel'] == pn and r['cond'] == 'late k1']):.1f} · visible {score([r for r in rec if r['panel'] == pn and r['cond'] == 'visible']):.1f}")
    print("끝")

elif a.phase == "probe_cf":
    # 2026-10-29 사용자: "encoder 는 context — 앞 부분 (1–16 장, 15·16 가려짐) k=2 로. (b) predictor 는 17–32 장 (17·18 가려짐). (c) transfer".
    #   → 시험 영상 = late k2 (문맥 끝 2 장 + 미래 앞 2 장 가림). 그림의 x 축 = 블록 (2 장씩).
    #   job 0–7 = 블록 b: (a) 문맥 encoder 의 문맥 블록 b 로 판독기 학습 · 시험 / (b) predictor 의 미래 블록 b 로 학습 · 시험
    #   job 8–10 = (c) transfer (shape · color · vel): 문맥 encoder 의 문맥 블록 8 개를 모두 모아 판독기 하나를 학습 (1,500 걸음) → predictor 미래 블록 1–8 에 그대로
    #   학습 = 전 조건 · 모든 k 의 영상 (block 단위 5-fold), 씨앗 3. 목표: shape · color (primitive 600) · vel (primitive + realistic 1,200, 그 블록 중심 속도)
    dev = torch.device("cuda")
    M = json.load(open(os.path.join(CACHE, "meta.json"))); Mr = json.load(open(os.path.join(CACHE, "meta_real.json")))
    cond_ = lambda m: "visible" if m["occlusion"] == "visible" else f"{m['occlusion']} k{m['k']}"
    def vids(tgt):
        return [(m["video_id"], src, i, (m[tgt] if tgt != "vel" else -1), m) for src, MM in ((("M", M), ("R", Mr)) if tgt == "vel" else (("M", M),))
                for i, m in enumerate(MM)]
    def load_blocks(rep, V, blocks):
        A = {k_: np.load(os.path.join(CACHE, f"{rep}{sfx_}.npy"), mmap_mode="r") for k_, sfx_ in (("M", ""), ("R", "_real"))}
        return torch.from_numpy(np.stack([np.concatenate([A[v[1]][v[2], b * S:(b + 1) * S] for b in blocks]) for v in V])).to(dev).view(len(V) * len(blocks), S, D)
    def labels(V, tgt, half, blocks):
        if tgt == "vel":
            return torch.tensor(motion_labels(V, half, "vel").reshape(len(V), NB, 2)[:, blocks].reshape(-1, 2), device=dev)
        return torch.tensor([v[3] for v in V], device=dev).repeat_interleave(len(blocks))
    od = os.path.join(OUT, "ctx_fut"); os.makedirs(od, exist_ok=True)
    jobs = [("block", b) for b in range(NB)] + [("transfer", t_) for t_ in ("shape", "color", "vel")]
    kind, arg = jobs[a.job]
    for tgt in (("shape", "color", "vel") if kind == "block" else (arg,)):
        V = vids(tgt); ncls = 2 if tgt == "vel" else NCLS[tgt]; rec = []
        if kind == "block":
            b = arg
            X = {"z": load_blocks("z", V, [b]), "p": load_blocks("p", V, [b])}
            Y = {"z": labels(V, tgt, 0, [b]), "p": labels(V, tgt, 1, [b])}
        else:
            XZ = load_blocks("z", V, list(range(NB))); YZ = labels(V, tgt, 0, list(range(NB)))
            XP = load_blocks("p", V, list(range(NB))); YP = labels(V, tgt, 1, list(range(NB)))
        for seed in SEEDS:
            fo = folds(V, seed)
            for fold in range(5):
                tr = [j for j, v in enumerate(V) if fo[v[4]["block_id"]] != fold]; te = [j for j, v in enumerate(V) if fo[v[4]["block_id"]] == fold]
                if kind == "block":
                    for rep in ("z", "p"):
                        clf = fit(X[rep], Y[rep], tr, ncls, seed * 10 + fold, dev)
                        q = predict(clf, X[rep], te, dev); Yc = Y[rep].cpu().numpy(); del clf
                        rec += [dict(seed=seed, fold=fold, panel=rep, b=b, vid=V[j][0], block_id=V[j][4]["block_id"], cond=cond_(V[j][4]),
                                     y=(Yc[j].tolist() if tgt == "vel" else int(Yc[j])), pred=(np.asarray(qq).tolist() if tgt == "vel" else int(qq)))
                                for j, qq in zip(te, q)]
                else:
                    trz = [j * NB + bb for j in tr for bb in range(NB)]
                    clf = fit(XZ, YZ, trz, ncls, seed * 10 + fold, dev, steps=1500)
                    Yc = YP.cpu().numpy()
                    for bb in range(NB):
                        ii = [j * NB + bb for j in te]; q = predict(clf, XP, ii, dev)
                        rec += [dict(seed=seed, fold=fold, panel="z->p", b=bb, vid=V[t_ // NB][0], block_id=V[t_ // NB][4]["block_id"], cond=cond_(V[t_ // NB][4]),
                                     y=(Yc[t_].tolist() if tgt == "vel" else int(Yc[t_])), pred=(np.asarray(qq).tolist() if tgt == "vel" else int(qq)))
                                for t_, qq in zip(ii, q)]
                    del clf
                print(f"[{kind} {arg} / {tgt}] seed {seed} fold {fold} 학습 영상 {len(tr)}", flush=True)
        json.dump(rec, open(os.path.join(od, f"{tgt}_b{arg + 1}.json" if kind == "block" else f"{tgt}_transfer.json"), "w"))
        for pn in sorted({r["panel"] for r in rec}):
            print(f"{kind} {arg} {tgt} {pn} late k2: " + " ".join(f"{score([r for r in rec if r['panel'] == pn and r['cond'] == 'late k2' and r['b'] == bb]):.1f}"
                                                           for bb in sorted({r["b"] for r in rec})))
    print("끝")

elif a.phase == "probe_pool":
    # 2026-10-29 사용자: "predictor 가 예상한 미래에서 진짜 color 정보 다 안 읽히는 게 맞아?" · "표본이 영상당 T 개 있는 거잖아".
    #   판독기 하나 = 블록 하나 (256, D) 입력, 미래 8 블록을 모두 학습 표본으로 (영상 180 × 8 = 1,440, 블록 위치 섞음). 2,000 걸음.
    #   학습 · 시험 모두 한 무리 안에서: late (k1·2·3, primitive 물체 영상 180) / early+mid (같은 수 180 을 block 단위로 뽑음, 대조).
    #   시험: 블록별 정확도 + 영상 단위 (8 블록의 log-softmax 합으로 한 번 고름 — 미래 어딘가에 색이 있으면 잡힌다). 씨앗 3 × block 단위 5-fold.
    #   --job 0–7 = (p, h) × (color, shape) × (late, early+mid). 라벨 섞기 대조 (씨앗 0).
    rep, tgt, grp = [(r_, t_, g_) for r_ in ("p", "h") for t_ in ("color", "shape") for g_ in ("late", "earlymid")][a.job]
    dev = torch.device("cuda")
    M = json.load(open(os.path.join(CACHE, "meta.json")))
    late = [i for i, m in enumerate(M) if m["occlusion"] == "late"]
    em_blocks = sorted({m["block_id"] for m in M if m["occlusion"] in ("early", "mid")}, key=int)
    em_pick = set(np.random.default_rng(7).permutation(em_blocks)[:len({M[i]["block_id"] for i in late})])
    rows = late if grp == "late" else [i for i, m in enumerate(M) if m["occlusion"] in ("early", "mid") and m["block_id"] in em_pick]
    V = [(M[i]["video_id"], "M", i, M[i][tgt], M[i]) for i in rows]
    A = np.load(os.path.join(CACHE, f"{rep}.npy"), mmap_mode="r")
    X = torch.from_numpy(np.stack([A[v[2]] for v in V])).to(dev).view(len(V) * NB, S, D)
    y = torch.tensor([v[3] for v in V], device=dev).repeat_interleave(NB); ncls = NCLS[tgt]
    def logits(clf, ii):
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s0 in range(0, len(ii), 64):
                out.append(torch.log_softmax(clf(X[torch.tensor(ii[s0:s0 + 64], device=dev)].float()).float(), -1).cpu().numpy())
        return np.concatenate(out)
    rec = []
    for seed in SEEDS:
        fo = folds(V, seed)
        for fold in range(5):
            trv = [j for j, v in enumerate(V) if fo[v[4]["block_id"]] != fold]; tev = [j for j, v in enumerate(V) if fo[v[4]["block_id"]] == fold]
            tr = [j * NB + b for j in trv for b in range(NB)]; te = [j * NB + b for j in tev for b in range(NB)]
            for kind in (("real", "shuffle") if seed == 0 else ("real",)):
                yy = y
                if kind == "shuffle":                               # 영상 단위로 라벨을 섞는다 (같은 영상 8 블록은 같은 라벨 유지)
                    perm = np.random.default_rng(fold).permutation(len(trv)); yy = y.clone()
                    for a_, b_ in zip(trv, np.array(trv)[perm]):
                        yy[a_ * NB:(a_ + 1) * NB] = y[b_ * NB]
                clf = fit(X, yy, tr, ncls, seed * 10 + fold, dev, steps=2000)
                L = logits(clf, te).reshape(len(tev), NB, ncls); del clf
                for k_, j in enumerate(tev):
                    rec.append(dict(seed=seed, fold=fold, kind=kind, vid=V[j][0], block_id=V[j][4]["block_id"], cond=f"{V[j][4]['occlusion']} k{V[j][4]['k']}",
                                    y=int(V[j][3]), block_pred=L[k_].argmax(-1).tolist(), video_pred=int(L[k_].sum(0).argmax())))
            print(f"[{rep}/{tgt}/{grp}] seed {seed} fold {fold} 학습 표본 {len(tr)}", flush=True)
    od = os.path.join(OUT, "pool"); os.makedirs(od, exist_ok=True)
    json.dump(rec, open(os.path.join(od, f"{rep}_{tgt}_{grp}.json"), "w"))
    rr = [r for r in rec if r["kind"] == "real"]
    print(f"{rep} {tgt} {grp}: 블록별 " + " ".join(f"{100 * np.mean([r['block_pred'][b] == r['y'] for r in rr]):.0f}" for b in range(NB)) +
          f" · 영상 단위 {100 * np.mean([r['video_pred'] == r['y'] for r in rr]):.1f} · 섞기 {100 * np.mean([r['video_pred'] == r['y'] for r in rec if r['kind'] == 'shuffle']):.1f}")
    print("끝")

elif a.phase == "report_cf":
    rng = np.random.default_rng(0); res = {}
    def boot(rr):
        by = collections.defaultdict(list)
        for r in rr:
            by[r["block_id"]].append(r)
        bl = list(by)
        bs = [score([r for ix in rng.integers(0, len(bl), len(bl)) for r in by[bl[ix]]]) for _ in range(1000)]
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    od = os.path.join(OUT, "ctx_fut")
    for tgt in ("shape", "color", "vel"):
        R = [r for b in range(1, NB + 1) for r in json.load(open(os.path.join(od, f"{tgt}_b{b}.json")))] + json.load(open(os.path.join(od, f"{tgt}_transfer.json")))
        for c in ("late k2", "visible"):
            print(f"=== {tgt} · {c} ({'R²' if tgt == 'vel' else '정확도'} %)")
            for pn in ("z", "p", "z->p"):
                row = []
                for b in range(NB):
                    rr = [r for r in R if r["panel"] == pn and r["cond"] == c and r["b"] == b]; sc = score(rr); lo, hi = boot(rr)
                    res[f"{tgt}|{c}|{pn}|b{b + 1}"] = dict(score=sc, ci=[lo, hi], n_per_seed=len(rr) / len(SEEDS)); row.append(f"{sc:6.1f}")
                print(f"  {pn:5s} " + " ".join(row))
    json.dump(res, open(os.path.join(od, "summary.json"), "w"), ensure_ascii=False, indent=1)

elif a.phase == "report_t_vel":
    rng = np.random.default_rng(0); res = {}
    def boot(rr):
        by = collections.defaultdict(list)
        for r in rr:
            by[r["block_id"]].append(r)
        bl = list(by)
        bs = [score([r for ix in rng.integers(0, len(bl), len(bl)) for r in by[bl[ix]]]) for _ in range(1000)]
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    for c in ("late k1", "visible"):
        print(f"=== {c} · 속도 R² % (95 % CI)")
        for t in range(1, NB + 1):
            R = json.load(open(os.path.join(OUT, "per_t_vel", f"t{t}.json")))
            row = []
            for pn in ("h->h", "p->p", "h->p"):
                rr = [r for r in R if r["panel"] == pn and r["cond"] == c]; sc = score(rr); lo, hi = boot(rr)
                res[f"{c}|t{t}|{pn}"] = dict(r2=sc, ci=[lo, hi], n_per_seed=len(rr) / len(SEEDS)); row.append(f"{pn} {sc:6.1f} [{lo:6.1f}, {hi:6.1f}]")
            print(f"t{t}  " + " · ".join(row))
    json.dump(res, open(os.path.join(OUT, "per_t_vel", "summary.json"), "w"), ensure_ascii=False, indent=1)

else:
    # 분류 = 정확도 %, 회귀 (vel · acc) = R² %. CI = block 단위 부트스트랩 (2,000 회, 다시 뽑은 block 의 기록으로 같은 점수를 다시 잰다).
    # 기준선: 분류 = 학습 다수 클래스, 회귀 = 학습 평균을 늘 내는 예측. 라벨 섞기 = 씨앗 0 에서 학습 라벨을 섞은 판독기.
    rng = np.random.default_rng(0); res = {}
    def boot(rr):
        by = collections.defaultdict(list)
        for r in rr:
            by[r["block_id"]].append(r)
        bl = list(by)
        bs = [score([r for ix in rng.integers(0, len(bl), len(bl)) for r in by[bl[ix]]]) for _ in range(2000)]
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    print(f"{'표현':12s} {'목표':6s} {'범주':9s} {'점수':>7s} {'95% CI':>15s} {'균형정확도':>9s} {'기준선':>8s} {'라벨섞기':>8s} {'학습n':>6s} {'학습점수':>7s} {'시험n':>6s}")
    for rep in ("z_boundary", "z", "p", "h"):
        for tgt in ("obj", "shape", "color", "vel", "acc", "acc3"):
            fn = os.path.join(OUT, f"probe_{rep.split('_')[0]}_{tgt}{'_boundary' if rep.endswith('boundary') else ''}.json")
            if not os.path.exists(fn):
                continue
            J = json.load(open(fn)); R = J["records"]; reg = tgt in REG
            for panel in ("self", "z2p", "empty"):
                for c in ("visible", "occluded"):
                    rr = [r for r in R if r["panel"] == panel and r["cat"] == c]
                    if not rr:
                        continue
                    sc = score(rr); lo, hi = boot(rr)
                    bal = float("nan") if reg else 100 * np.mean([np.mean([r["pred"] == r["y"] for r in rr if r["y"] == k]) for k in sorted({r["y"] for r in rr})])
                    base = score([dict(r, pred=r["maj"]) for r in rr])
                    sh = [r for r in R if r["panel"] == "shuffle" and r["cat"] == c]
                    shs = score(sh) if (sh and panel == "self") else float("nan")
                    lg = [l for l in J["log"] if l["cat"] == c]
                    name = {"self": rep, "z2p": "z→p", "empty": rep + "·빈장면"}[panel] + ("(경계)" if rep.endswith("boundary") and panel == "z2p" else "")
                    res[f"{name}|{tgt}|{c}"] = dict(acc=sc, ci=[lo, hi], bal_acc=bal, majority=base, shuffle=shs, chance=0.0 if reg else 100 / NCLS[tgt],
                                                     metric="R2" if reg else "accuracy", n_train=float(np.mean([l["n_train"] for l in lg])),
                                                     train_acc=100 * float(np.mean([l["train_acc"] for l in lg])), n_test_per_seed=len(rr) / len(SEEDS))
                    print(f"{name:12s} {tgt:6s} {c:9s} {sc:7.1f} {f'[{lo:.1f}, {hi:.1f}]':>15s} {bal:9.1f} {base:8.1f} {shs:8.1f} "
                          f"{res[f'{name}|{tgt}|{c}']['n_train']:6.0f} {res[f'{name}|{tgt}|{c}']['train_acc']:7.1f} {len(rr) / len(SEEDS):6.0f}")
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=1)
