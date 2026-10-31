#!/usr/bin/env python3
"""릴리즈 ViT-H · VOBench_v1 late (문맥 끝 가림) — 시간 블록마다 물체 probing (2026-10-30, z_analysis/release).

묻는 것: late k 마다, 문맥 블록 1–8 과 예측 블록 9–16 의 각 블록 표현에서 물체 정보 (모양 · 색 · 있음/없음) 가 읽히나.
  블록 t = 튜블릿 t = 모델 입력 연속 2 샘플 (1-idx, 블록 1–8 = 문맥 raw 0–45, 9–16 = 미래 raw 48–93).
  표현 (모두 릴리즈, probing 관례 bfloat16, 기존 캐시):
    z[t] = LN(context_encoder(문맥 16 장))[t]        t = 1..8   predictor 의 입력 (transfer_gap z_late_ctx.npy)
    h[t] = LN(target_encoder(32 장))[t]              t = 1..16  실제 영상의 encoder 표현 (tubeprobe h.npy)
    p[t] = predictor(z, mask_index 0)[t]             t = 9..16  예측 (tubeprobe p.npy)
  표본 = late 의 가능 변이 432 clip (k 1·2·3 각 144, CLAUDE.md §1-4). 분할 = split v2 의 4 중항 (block_id) 단위 216 / 216,
    2-fold 교차 (한쪽으로 학습 → 다른 쪽 평가, 뒤집어 한 번 더) 라 모든 clip 이 한 번씩 test 가 된다.
  ★ 지금 판 (2026-10-30 사용자 결정 "attentive 말고 mean pooling + linear") — phase `pool` → `linear` (CPU):
    입력 = 블록 t 의 공간 토큰 256 개 평균 (1, 1280). 표본 = clip × 블록 → 구간 (문맥 z 블록 1–8 / 예측 p 블록 9–16) 마다
    선형 층 하나 (1280 → 클래스) 를 그 구간의 모든 블록 행으로 학습하고 블록마다 평가한다 (학습 행 fold 당 90 × 8 = 720).
    특징 표준화 (train fold 통계) · 교차 엔트로피 + L2 λ · 전배치 L-BFGS. λ ∈ {1e-4 … 1} 는 train fold 안의 4 중항 3 겹 교차로 고른다.
    비교용으로 같은 선형 층을 블록마다 따로 학습한 판 (행 90) 도 json 에 남긴다 (`per_block`).
    → results/late_block_linear.json
  이전 판 (attentive) — phase `probe` (GPU): head = 표현 · 블록 · 과제마다 따로 (그 블록 표현으로 학습 → 같은 표현 · 같은 블록의 held-out 에서 잼).
    AttentiveClassifier (depth 1, query 1, head 16) · 100 epoch, batch 16, lr 3e-4 OneCycle, wd 0.05 · head 시드 0–2.
    ⚠️ attn_30 (35 epoch, lr 1e-4) 은 학습 clip 90 개에서 덜 학습된다 — 보이는 문맥 블록 3 의 색 30 → 64 %, 모양 75 → 93 %
    (점검 job 235860, 로그 z_analysis/release/01_block_probe/logs/tune_235860.log). 첫 판 (attn_30, split v2 한 방향) 은 results/v1/late_block_probe_v0_attn30.json.
    학습은 k 1·2·3 을 합쳐서, 평가는 k 마다.
  과제: shape (primitive 물체 clip, 6 류, 우연 16.7) · color (primitive 물체 clip, 8 색, 우연 12.5)
        · exist (permanence clip: 물체 있음 pos_a vs 빈 장면 pos_b, primitive + realistic, 우연 50)
  CI = test 4 중항 재표집 (2000 회) 95 %, 값은 head 시드 3 개 평균의 clip 정답률.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/release_late_block_probe.py stage            # vll5 (캐시가 노드 로컬) → NFS 사본 ~9 GB
  for T in shape color exist; do sbatch -p batch_vll -w vll2 --gres=gpu:1 -c 8 --mem=64G --time=0-02:00:00 \
    --wrap "$P z_research/scripts/analysis/release_late_block_probe.py probe --task $T"; done   # → results/late_block_probe_<task>.json (이전 판)
  $P z_research/scripts/analysis/release_late_block_probe.py pool             # 블록 평균 → NFS pooled.npz (CPU, ~1 분)
  sbatch -p batch_vll -w vll2 --gres=gpu:1 -c 8 --mem=16G --wrap "$P z_research/scripts/analysis/release_late_block_probe.py linear"
  VOBench_v2 (2026-10-31, 설계 칸마다 장면 3 개 = v1 의 3 배): release_v2_extract.py 로 z · p · pooled 를 뽑은 뒤
    linear --set v2 --modes shared --task <과제>  → results/late_block_linear_v2_<과제>.json (설계 칸 단위 2 겹)
                                                                               # → results/late_block_linear.json (GPU 몇 분, CPU 6 코어면 ~35 분)
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TUBE = Path("/local_datasets/world/world_analysis/cache/vobench_v1_vith_tubeprobe")
TG = Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap")
STAGE = Path("/data/hyuntak/project/2026/2027_cvpr/nfs_cache/release_late_vobench")
SPLIT = REPO / "data_csv/vobench_v1_split_v2/index_train.csv"
OUT_BASE = REPO / "z_analysis/release/01_block_probe/results"   # 2026-10-31 폴더 정리: v1 · v2 하위 폴더
OUT = OUT_BASE / "v1"
OUT_OF = lambda dset: OUT_BASE / dset
NT, NF = 16, 8
OPT = dict(epochs=100, batch_size=16, lr=3e-4, weight_decay=0.05)
SEEDS = (0, 1, 2)


def label_of(m):
    return m["obj_a"] if m["variant"] == "pos_a" else m["obj_b"]


# ------------------------------------------------------------------ stage (vll5)
def stage():
    M = json.load(open(TUBE / "meta.json"))["meta"]
    idx = [i for i, m in enumerate(M) if m["occlusion"] == "late"]
    zm = json.load(open(TG / "z_late_meta.json"))
    assert zm["tp"] == idx and all(M[t]["video_id"] == v for t, v in zip(zm["tp"], zm["video_id"])), "z_late 순서 불일치"
    STAGE.mkdir(parents=True, exist_ok=True)
    H = np.load(TUBE / "h.npy", mmap_mode="r"); P = np.load(TUBE / "p.npy", mmap_mode="r")
    Z = np.load(TG / "z_late_ctx.npy", mmap_mode="r")
    assert H.shape[1:] == (NT, 256, 1280) and P.shape[1:] == (NF, 256, 1280) and Z.shape == (len(idx), NF, 256, 1280)
    for name, src, rows in (("z", Z, list(range(len(idx)))), ("h", H, idx), ("p", P, idx)):
        t0 = time.time()
        out = np.lib.format.open_memmap(STAGE / f"{name}.npy.tmp", mode="w+", dtype=np.float16, shape=(len(rows),) + src.shape[1:])
        for j, r in enumerate(rows):
            out[j] = src[r]
        out.flush(); del out
        (STAGE / f"{name}.npy.tmp").rename(STAGE / f"{name}.npy")
        print(f"{name} {len(rows)} clip {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(meta=[M[i] for i in idx], tube_index=idx, src=dict(h=str(TUBE / "h.npy"), p=str(TUBE / "p.npy"), z=str(TG / "z_late_ctx.npy"))),
              open(STAGE / "meta.json", "w"))
    (STAGE / "_DONE").write_text(f"late clips {len(idx)}\n")


# ------------------------------------------------------------------ probe (GPU)
def tasks(M):
    """과제 → (clip 행 목록, 라벨 배열, 클래스 이름)."""
    out = {}
    prim = [i for i, m in enumerate(M) if m["objset"] == "primitive" and label_of(m) != "empty"]
    for name, f in (("shape", lambda l: l.split("/")[0]), ("color", lambda l: l.split("/")[1])):
        labs = [f(label_of(M[i])) for i in prim]
        cls = sorted(set(labs))
        out[name] = (prim, np.array([cls.index(l) for l in labs]), cls)
    perm = [i for i, m in enumerate(M) if m["violation"] == "permanence"]
    out["exist"] = (perm, np.array([int(label_of(M[i]) != "empty") for i in perm]), ["empty", "object"])
    return out


def fit_eval(Xtr, ytr, Xte, ncls, seed, dev):
    import torch, torch.nn as nn
    from src.models.attentive_pooler import AttentiveClassifier
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    clf = AttentiveClassifier(embed_dim=Xtr.shape[-1], num_heads=16, mlp_ratio=4.0, depth=1, num_classes=ncls).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=OPT["lr"], weight_decay=OPT["weight_decay"])
    bs = OPT["batch_size"]; steps = OPT["epochs"] * ((len(Xtr) + bs - 1) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, OPT["lr"], total_steps=steps)
    Xtr_t = torch.from_numpy(Xtr).to(dev); y = torch.from_numpy(ytr).to(dev)
    for _ in range(OPT["epochs"]):
        clf.train(); perm = rng.permutation(len(Xtr))
        for s0 in range(0, len(Xtr), bs):
            sel = torch.from_numpy(perm[s0:s0 + bs]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = nn.functional.cross_entropy(clf(Xtr_t[sel].float()), y[sel])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    clf.eval()

    def pred(X):
        o = []
        with torch.no_grad():
            for s0 in range(0, len(X), 64):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o.append(clf(torch.from_numpy(X[s0:s0 + 64]).to(dev).float()).float().argmax(1).cpu().numpy())
        return np.concatenate(o)
    return pred(Xtr), pred(Xte)


def boot_ci(correct, groups, n=2000, seed=0):
    """correct: clip 별 정답률 (시드 평균), groups: 4 중항 id. 4 중항 재표집 95 % CI."""
    ug = sorted(set(groups)); gi = {g: [j for j, x in enumerate(groups) if x == g] for g in ug}
    s = np.array([correct[gi[g]].sum() for g in ug]); c = np.array([len(gi[g]) for g in ug])
    rng = np.random.default_rng(seed); b = rng.integers(0, len(ug), size=(n, len(ug)))
    v = s[b].sum(1) / c[b].sum(1)
    return [100 * float(np.percentile(v, 2.5)), 100 * float(np.percentile(v, 97.5))]


def probe(gpu, task_sel=None, reps=("z", "p")):
    import torch
    sys.path.insert(0, str(REPO))
    assert (STAGE / "_DONE").exists(), "stage 먼저"
    dev = torch.device(f"cuda:{gpu}")
    M = json.load(open(STAGE / "meta.json"))["meta"]
    train_b = {r["block_id"] for r in csv.DictReader(SPLIT.open())}
    t0 = time.time()
    R = {name: np.load(STAGE / f"{name}.npy") for name in reps}                    # RAM 으로 (z · p 각 2.3 GB, h 4.5 GB)
    print(f"loaded {[(k, v.shape) for k, v in R.items()]} {time.time() - t0:.0f}s", flush=True)
    blocks_of = {r: {"z": range(1, NF + 1), "h": range(1, NT + 1), "p": range(NF + 1, NT + 1)}[r] for r in reps}
    col = lambda rep, b: b - 1 if rep in ("z", "h") else b - 1 - NF                   # 블록 (1-idx) → 배열 열
    res = dict(opt=OPT, seeds=list(SEEDS), split=str(SPLIT.relative_to(REPO)), folds=2, reps=list(reps), n_clips=len(M), tasks={})
    for task, (rows, y, cls) in tasks(M).items():
        if task_sel and task != task_sel:
            continue
        fa = np.array([M[i]["block_id"] in train_b for i in rows])                 # fold A = split v2 train 4 중항
        te = list(range(len(rows)))                                                 # 2-fold: 모든 clip 이 한 번씩 test
        ri = np.array(rows)
        T = res["tasks"][task] = dict(classes=cls, chance=100.0 / len(cls) if task != "exist" else 50.0,
                                      n_fold=[int(fa.sum()), int((~fa).sum())], n_test={k: sum(M[i]["k"] == k for i in rows) for k in ("1", "2", "3")}, cells={})
        print(f"[{task}] classes {len(cls)} fold {T['n_fold']} test {T['n_test']} · 학습 클래스 최소 "
              f"{min(np.bincount(y[fa], minlength=len(cls)).min(), np.bincount(y[~fa], minlength=len(cls)).min())}", flush=True)
        for rep, bl in blocks_of.items():
            for b in bl:
                X = np.ascontiguousarray(R[rep][ri, col(rep, b)])                   # (n, 256, 1280) fp16
                cor = np.zeros((len(SEEDS), len(rows))); tr_acc = []
                for si, s in enumerate(SEEDS):
                    for f in (fa, ~fa):
                        ptr, pte = fit_eval(X[f], y[f], X[~f], len(cls), s, dev)
                        cor[si, ~f] = pte == y[~f]; tr_acc.append(100 * float((ptr == y[f]).mean()))
                cm = cor.mean(0)
                cell = dict(train_acc=float(np.mean(tr_acc)), all=100 * float(cm.mean()), by_k={})
                for k in ("1", "2", "3"):
                    jj = [q for q, j in enumerate(te) if M[rows[j]]["k"] == k]
                    g = [M[rows[te[q]]]["block_id"] for q in jj]
                    cell["by_k"][k] = dict(n=len(jj), acc=100 * float(cm[jj].mean()), ci=boot_ci(cm[jj], g),
                                           seed_acc=[100 * float(cor[si, jj].mean()) for si in range(len(SEEDS))])
                T["cells"][f"{rep}{b}"] = cell
                print(f"[{task}] {rep} 블록 {b:2d}  train {cell['train_acc']:5.1f} · test k1 {cell['by_k']['1']['acc']:5.1f} "
                      f"k2 {cell['by_k']['2']['acc']:5.1f} k3 {cell['by_k']['3']['acc']:5.1f} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / (f"late_block_probe_{task_sel}.json" if task_sel else "late_block_probe.json")
    json.dump(res, open(fn, "w"), indent=1, ensure_ascii=False)
    print("done", fn, flush=True)


# ------------------------------------------------------------------ pool · linear (CPU, 지금 판)
LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def pool():
    out = {}
    for name in ("z", "p"):
        A = np.load(STAGE / f"{name}.npy", mmap_mode="r")
        m = np.zeros(A.shape[:2] + A.shape[3:], np.float32)
        for i in range(0, len(A), 16):
            m[i:i + 16] = np.asarray(A[i:i + 16], dtype=np.float32).mean(2)
        out[name] = m
        print(name, m.shape, flush=True)
    np.savez(STAGE / "pooled.npz", **out)


def _dev():
    import torch
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _fit_linear(X, y, ncls, lam):
    import torch
    dev = _dev()
    X = torch.from_numpy(X).to(dev); y = torch.from_numpy(y).to(dev)
    W = torch.zeros(X.shape[1], ncls, device=dev, requires_grad=True); b = torch.zeros(ncls, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=500, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, y) + lam * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach()


def _predict(W, b, X):
    import torch
    return (torch.from_numpy(np.ascontiguousarray(X)).to(W.device) @ W + b).argmax(1).cpu().numpy()


def _rows(F, clips, blocks, col):
    """clip × 블록 행. F (clip, 8, D) → (len(clips)·len(blocks), D), 행 순서 = clip 바깥 · 블록 안쪽."""
    return F[clips][:, [col(b) for b in blocks]].reshape(-1, F.shape[-1])


def _train(F, y, clips, blocks, col, ncls, groups, seed=0):
    """λ 를 train clip 안의 4 중항 3 겹 교차로 고르고, 고른 λ 로 전체 train 행에 학습. 표준화 통계는 train 행."""
    rng = np.random.default_rng(seed)
    ug = sorted(set(groups[clips])); rng.shuffle(ug); fold_of = {g: j % 3 for j, g in enumerate(ug)}
    fo = np.array([fold_of[groups[c]] for c in clips])
    nb = len(blocks)
    score = {}
    for lam in LAMS:
        acc = []
        for f in range(3):
            tr, va = clips[fo != f], clips[fo == f]
            Xtr = _rows(F, tr, blocks, col); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            W, b = _fit_linear((Xtr - mu) / sd, np.repeat(y[tr], nb), ncls, lam)
            acc.append((_predict(W, b, (_rows(F, va, blocks, col) - mu) / sd) == np.repeat(y[va], nb)).mean())
        score[lam] = float(np.mean(acc))
    lam = max(LAMS, key=lambda l: (score[l], l))
    Xtr = _rows(F, clips, blocks, col); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    W, b = _fit_linear((Xtr - mu) / sd, np.repeat(y[clips], nb), ncls, lam)
    tr_acc = float((_predict(W, b, (Xtr - mu) / sd) == np.repeat(y[clips], nb)).mean())
    return (W, b, mu, sd), lam, score, tr_acc


STAGE_V2 = Path("/data/hyuntak/project/2026/2027_cvpr/nfs_cache/release_late_vobench_v2")


def cell_of(m):
    """VOBench_v2 설계 칸 (replica 셋이 궤적 · 가림막을 공유) — 분할 · 재표집 단위."""
    return "|".join(m[k] for k in ("objset", "violation", "background", "motion", "occlusion", "k"))


def v2_folds(M):
    """설계 칸 단위 2 겹: (objset, violation, k) 층마다 칸을 seed 0 으로 섞어 절반을 겹 A 로."""
    strata = {}
    for m in M:
        strata.setdefault((m["objset"], m["violation"], m["k"]), set()).add(cell_of(m))
    fold_a = set()
    for key in sorted(strata):
        cells = sorted(strata[key]); np.random.default_rng(0).shuffle(cells)
        fold_a |= set(cells[: len(cells) // 2])
    return fold_a


def v1_random_folds(M, seed):
    """점검용: VOBench_v1 장면 (block_id) 단위 무작위 2 겹, (objset, violation, k) 층마다 절반."""
    strata = {}
    for m in M:
        strata.setdefault((m["objset"], m["violation"], m["k"]), set()).add(m["block_id"])
    fold_a = set()
    for key in sorted(strata):
        bl = sorted(strata[key]); np.random.default_rng(seed).shuffle(bl)
        fold_a |= set(bl[: len(bl) // 2])
    return fold_a


def linear(dset="v1", task_sel=None, modes=("shared", "per_block"), fold_seed=None):
    import torch
    torch.set_num_threads(6)
    stage = STAGE if dset == "v1" else STAGE_V2
    M = json.load(open(stage / "meta.json"))["meta"]
    Fz = np.load(stage / "pooled.npz")
    F = {"z": Fz["z"], "p": Fz["p"]}
    if dset == "v1":
        train_b = {r["block_id"] for r in csv.DictReader(SPLIT.open())} if fold_seed is None else v1_random_folds(M, fold_seed)
        groups = np.array([m["block_id"] for m in M])
        in_a = lambda m: m["block_id"] in train_b
        split_desc = str(SPLIT.relative_to(REPO)) if fold_seed is None else f"장면 무작위 2 겹 seed {fold_seed} ((objset, violation, k) 층마다 절반)"
    else:
        fold_a = v2_folds(M)
        groups = np.array([cell_of(m) for m in M])
        in_a = lambda m: cell_of(m) in fold_a
        split_desc = "VOBench_v2 설계 칸 단위 2 겹 ((objset, violation, k) 층마다 seed 0 절반), 재표집 · λ 교차도 설계 칸 단위"
    seg = {"z": range(1, NF + 1), "p": range(NF + 1, NT + 1)}
    col = {"z": lambda b: b - 1, "p": lambda b: b - 1 - NF}
    res = dict(probe="block mean (256 토큰 평균) · 표준화 · 선형 층 · CE + L2 (λ 는 train 안 4 중항 3 겹 교차) · L-BFGS",
               dataset=dset, lams=list(LAMS), split=split_desc, folds=2, n_clips=len(M), tasks={})
    t0 = time.time()
    for task, (rows, y_rows, cls) in tasks(M).items():
        if task_sel and task != task_sel:
            continue
        rows = np.array(rows); y = np.full(len(M), -1); y[rows] = y_rows
        fa = np.array([in_a(M[i]) for i in rows])
        T = res["tasks"][task] = dict(classes=cls, chance=100.0 / len(cls) if task != "exist" else 50.0,
                                      n_fold_clips=[int(fa.sum()), int((~fa).sum())],
                                      n_test={k: int(sum(M[i]["k"] == k for i in rows)) for k in ("1", "2", "3")},
                                      shared={}, per_block={}, lam={})
        for rep in ("z", "p"):
            bl = list(seg[rep])
            for mode in modes:
                cor = {b: np.zeros(len(M)) for b in bl}; lam_log = []
                for f in (fa, ~fa):
                    trc, tec = rows[f], rows[~f]
                    for grp in ([bl] if mode == "shared" else [[b] for b in bl]):
                        (W, b_, mu, sd), lam, score, tr_acc = _train(F[rep], y, trc, grp, col[rep], len(cls), groups)
                        lam_log.append(dict(blocks=grp, lam=lam, train_acc=100 * tr_acc, inner=score))
                        for bb in grp:
                            pr = _predict(W, b_, (F[rep][tec][:, col[rep](bb)] - mu) / sd)
                            cor[bb][tec] = pr == y[tec]
                if mode == "shared":
                    T.setdefault("clip_correct", {})
                    for bb in bl:
                        T["clip_correct"][f"{rep}{bb}"] = [int(v) for v in cor[bb][rows]]
                    T["clip_video_id"] = [M[i]["video_id"] for i in rows]
                for bb in bl:
                    cell = dict(by_k={})
                    for k in ("1", "2", "3"):
                        jj = np.array([i for i in rows if M[i]["k"] == k])
                        cell["by_k"][k] = dict(n=len(jj), acc=100 * float(cor[bb][jj].mean()), ci=boot_ci(cor[bb][jj], list(groups[jj])))
                    T[mode][f"{rep}{bb}"] = cell
                T["lam"][f"{rep}_{mode}"] = lam_log
                line = " ".join(f"{T[mode][f'{rep}{bb}']['by_k']['2']['acc']:3.0f}" for bb in bl)
                print(f"[{task}] {rep} {mode:9s} k2: {line}  λ {[d['lam'] for d in lam_log][:4]} ({time.time() - t0:.0f}s)", flush=True)
    OUT = OUT_OF(dset); OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / ("late_block_linear" + ("" if dset == "v1" else f"_{dset}") + (f"_fs{fold_seed}" if fold_seed is not None else "")
                + (f"_{task_sel}" if task_sel else "") + ".json")
    json.dump(res, open(fn, "w"), indent=1, ensure_ascii=False)
    print("done", fn, flush=True)


def attn_shared(dset="v2", task_sel=None, epochs=20, seeds=(0,)):
    """attentive head (fit_eval 과 같은 구조 · lr 3e-4 OneCycle) 를 구간 공유로: 행 = clip × 구간 블록, 블록마다 평가 (2026-10-31, 사용자 "attn probe 로").
    입력 = 블록 하나의 256 토큰. 분할 · 재표집은 linear 와 같다. → results/late_block_attn_<set>_<과제>.json"""
    import torch
    sys.path.insert(0, str(REPO))
    stage = STAGE if dset == "v1" else STAGE_V2
    M = json.load(open(stage / "meta.json"))["meta"]
    t0 = time.time()
    R = {name: np.load(stage / f"{name}.npy") for name in ("z", "p")}
    print(f"loaded {[(k, v.shape) for k, v in R.items()]} {time.time() - t0:.0f}s", flush=True)
    if dset == "v1":
        train_b = {r["block_id"] for r in csv.DictReader(SPLIT.open())}
        groups = np.array([m["block_id"] for m in M]); in_a = lambda m: m["block_id"] in train_b
    else:
        fold_a = v2_folds(M); groups = np.array([cell_of(m) for m in M]); in_a = lambda m: cell_of(m) in fold_a
    seg = {"z": range(1, NF + 1), "p": range(NF + 1, NT + 1)}
    col = {"z": lambda b: b - 1, "p": lambda b: b - 1 - NF}
    OPT["epochs"] = epochs; OPT["lr"] = 3e-4
    dev = torch.device("cuda:0")
    res = dict(probe=f"attentive (depth 1, query 1, head 16) · 구간 공유 · {OPT} · head 시드 {list(seeds)}", dataset=dset, folds=2, tasks={})
    for task, (rows, y_rows, cls) in tasks(M).items():
        if task_sel and task != task_sel:
            continue
        rows = np.array(rows); y = np.full(len(M), -1); y[rows] = y_rows
        fa = np.array([in_a(M[i]) for i in rows])
        T = res["tasks"][task] = dict(classes=cls, chance=100.0 / len(cls) if task != "exist" else 50.0,
                                      n_fold_clips=[int(fa.sum()), int((~fa).sum())], shared={}, train_acc={})
        for rep in ("z", "p"):
            bl = list(seg[rep]); cols = [col[rep](b) for b in bl]
            cor = {b: np.zeros((len(seeds), len(M))) for b in bl}; tr_acc = []
            for f in (fa, ~fa):
                trc, tec = rows[f], rows[~f]
                Xtr = np.ascontiguousarray(R[rep][trc][:, cols].reshape(-1, S_TOK, R[rep].shape[-1]))
                ytr = np.repeat(y[trc], len(bl))
                Xte = np.ascontiguousarray(R[rep][tec][:, cols].reshape(-1, S_TOK, R[rep].shape[-1]))
                for si, sd in enumerate(seeds):
                    ptr, pte = fit_eval(Xtr, ytr, Xte, len(cls), sd, dev)
                    tr_acc.append(100 * float((ptr == ytr).mean()))
                    pte = pte.reshape(len(tec), len(bl))
                    for q, b in enumerate(bl):
                        cor[b][si, tec] = pte[:, q] == y[tec]
                print(f"[{task}] {rep} fold done ({time.time() - t0:.0f}s)", flush=True)
            T["train_acc"][rep] = tr_acc
            T.setdefault("clip_correct", {}); T["clip_video_id"] = [M[i]["video_id"] for i in rows]
            for b in bl:
                cm = cor[b].mean(0)
                T["clip_correct"][f"{rep}{b}"] = [float(v) for v in cm[rows]]
                cell = dict(by_k={})
                for k in ("1", "2", "3"):
                    jj = np.array([i for i in rows if M[i]["k"] == k])
                    cell["by_k"][k] = dict(n=len(jj), acc=100 * float(cm[jj].mean()), ci=boot_ci(cm[jj], list(groups[jj])))
                T["shared"][f"{rep}{b}"] = cell
            line = " ".join(f"{T['shared'][f'{rep}{b}']['by_k']['2']['acc']:3.0f}" for b in bl)
            print(f"[{task}] {rep} shared k2: {line} · train acc {np.mean(tr_acc):.0f}", flush=True)
    OUT = OUT_OF(dset); OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / (f"late_block_attn_{dset}" + (f"_{task_sel}" if task_sel else "") + ".json")
    json.dump(res, open(fn, "w"), indent=1, ensure_ascii=False)
    print("done", fn, flush=True)


S_TOK = 256
ATTN_OUT = REPO / "z_analysis/release/01_block_probe/figures/v2_attn"     # <occ>/results/ (2026-10-31 사용자: v2_attn 안에 late · visible)


def _ln(a, chunk=64):
    """토큰마다 학습 계수 없는 층 정규화 (h 와 같은 정규화) — fp16 배열, 덩어리로."""
    out = np.empty_like(a)
    for i in range(0, len(a), chunk):
        x = np.asarray(a[i:i + chunk], np.float32)
        x = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-6)
        out[i:i + chunk] = x.astype(a.dtype)
    return out


def attn_occ(occ="late", task_sel=None, epochs=20, seeds=(0,)):
    """VOBench_v2 · attentive · 구간 공유 (attn_shared 와 같은 판독기 · 분할 · 재표집) + (c) 판독기 이식 (2026-10-31).
    자기 판독: z (블록 1–8) · p (9–16) · h (9–16, 미래 프레임의 target encoder) — 같은 표현으로 학습 · 평가.
    이식: h 판독기 → LN(p) 와 z 판독기 → LN(p) (블록 9–16, 같은 평가 겹). p 는 h 처럼 토큰마다 학습 계수 없는 층 정규화.
    occ = late (k 1–3) · visible (k 0). → figures/v2_attn/<occ>/results/<occ>_block_attn_v2_<과제>.json"""
    import torch
    sys.path.insert(0, str(REPO))
    stage = Path(f"/data/hyuntak/project/2026/2027_cvpr/nfs_cache/release_{occ}_vobench_v2")
    M = json.load(open(stage / "meta.json"))["meta"]
    t0 = time.time()
    R = {"z": np.load(stage / "z.npy"), "p": np.load(stage / "p.npy"), "h": np.load(stage / "h_fut.npy")}
    R["pn"] = _ln(R["p"])
    print(f"loaded {[(k, v.shape) for k, v in R.items()]} {time.time() - t0:.0f}s", flush=True)
    fold_a = v2_folds(M); groups = np.array([cell_of(m) for m in M]); in_a = lambda m: cell_of(m) in fold_a
    seg = {"z": range(1, NF + 1), "p": range(NF + 1, NT + 1), "h": range(NF + 1, NT + 1)}
    col = {"z": lambda b: b - 1, "p": lambda b: b - 1 - NF, "h": lambda b: b - 1 - NF}
    fut = list(range(NF + 1, NT + 1)); fcols = [b - 1 - NF for b in fut]
    OPT["epochs"] = epochs; OPT["lr"] = 3e-4
    dev = torch.device("cuda:0")
    res = dict(probe=f"attentive (depth 1, query 1, head 16) · 구간 공유 · {OPT} · head 시드 {list(seeds)}", dataset="v2", occlusion=occ,
               transfer="h 판독기 → LN(p) · z 판독기 → LN(p), 블록 9–16, 같은 평가 겹", folds=2, tasks={})
    for task, (rows, y_rows, cls) in tasks(M).items():
        if task_sel and task != task_sel:
            continue
        rows = np.array(rows); y = np.full(len(M), -1); y[rows] = y_rows
        ks = sorted({M[i]["k"] for i in rows})
        fa = np.array([in_a(M[i]) for i in rows])
        T = res["tasks"][task] = dict(classes=cls, chance=100.0 / len(cls) if task != "exist" else 50.0, ks=ks,
                                      n_fold_clips=[int(fa.sum()), int((~fa).sum())], shared={}, transfer={}, train_acc={})
        cors = {}
        for rep in ("z", "p", "h"):
            bl = list(seg[rep]); cols = [col[rep](b) for b in bl]
            cor = {b: np.zeros((len(seeds), len(M))) for b in bl}
            xfer = {b: np.zeros((len(seeds), len(M))) for b in fut} if rep in ("z", "h") else None
            tr_acc = []
            for f in (fa, ~fa):
                trc, tec = rows[f], rows[~f]
                Xtr = np.ascontiguousarray(R[rep][trc][:, cols].reshape(-1, S_TOK, R[rep].shape[-1]))
                ytr = np.repeat(y[trc], len(bl))
                Xte = R[rep][tec][:, cols].reshape(-1, S_TOK, R[rep].shape[-1])
                if xfer is not None:                                        # 같은 head 로 LN(p) 도 읽는다
                    Xte = np.concatenate([Xte, R["pn"][tec][:, fcols].reshape(-1, S_TOK, R["pn"].shape[-1])], 0)
                Xte = np.ascontiguousarray(Xte)
                for si, sd in enumerate(seeds):
                    ptr, pte = fit_eval(Xtr, ytr, Xte, len(cls), sd, dev)
                    tr_acc.append(100 * float((ptr == ytr).mean()))
                    own = pte[: len(tec) * len(bl)].reshape(len(tec), len(bl))
                    for q, b in enumerate(bl):
                        cor[b][si, tec] = own[:, q] == y[tec]
                    if xfer is not None:
                        tp = pte[len(tec) * len(bl):].reshape(len(tec), len(fut))
                        for q, b in enumerate(fut):
                            xfer[b][si, tec] = tp[:, q] == y[tec]
                print(f"[{occ} {task}] {rep} fold done ({time.time() - t0:.0f}s)", flush=True)
            T["train_acc"][rep] = tr_acc
            T.setdefault("clip_correct", {}); T["clip_video_id"] = [M[i]["video_id"] for i in rows]

            def cellstat(cm):
                d = dict(by_k={})
                for k in ks:
                    jj = np.array([i for i in rows if M[i]["k"] == k])
                    d["by_k"][k] = dict(n=len(jj), acc=100 * float(cm[jj].mean()), ci=boot_ci(cm[jj], list(groups[jj])))
                return d
            for b in bl:
                cm = cor[b].mean(0); T["clip_correct"][f"{rep}{b}"] = [float(v) for v in cm[rows]]
                T["shared"][f"{rep}{b}"] = cellstat(cm)
            if xfer is not None:
                for b in fut:
                    cm = xfer[b].mean(0); T["clip_correct"][f"{rep}2p{b}"] = [float(v) for v in cm[rows]]
                    T["transfer"][f"{rep}2p{b}"] = cellstat(cm)
            k0 = ks[len(ks) // 2]
            line = " ".join(f"{T['shared'][f'{rep}{b}']['by_k'][k0]['acc']:3.0f}" for b in bl)
            extra = (" · →p " + " ".join(f"{T['transfer'][f'{rep}2p{b}']['by_k'][k0]['acc']:3.0f}" for b in fut)) if xfer is not None else ""
            print(f"[{occ} {task}] {rep} k{k0}: {line}{extra} · train acc {np.mean(tr_acc):.0f}", flush=True)
    out = ATTN_OUT / occ / "results"; out.mkdir(parents=True, exist_ok=True)
    fn = out / (f"{occ}_block_attn_v2" + (f"_{task_sel}" if task_sel else "") + ".json")
    json.dump(res, open(fn, "w"), indent=1, ensure_ascii=False)
    print("done", fn, flush=True)


SPLIT_V2 = REPO / "data_csv/vobench_v2_split/split.json"
BIND_OUT = REPO / "z_analysis/release/03_binding_train/probe"
NFS = Path("/data/hyuntak/project/2026/2027_cvpr/nfs_cache")


def attn_split(occ="late", task_sel=None, models=("release", "a0", "a1", "a2"), epochs=20, seed=0):
    """binding 학습 평가 (2026-10-31): VOBench_v2 궤적 묶음 train 쪽 clip 으로 판독기 학습 → test 쪽 clip 에서 블록마다.
    z 판독기 (블록 1–8) 는 모든 모델 공통 (encoder 고정) — 한 번 학습해 z 자기 판독 + 각 모델 LN(p) 이식.
    p 자기 판독은 모델마다 그 모델의 p (train 쪽) 로 학습. 모델 p: release = nfs_cache/release_<occ>_vobench_v2/p.npy,
    팔 = nfs_cache/bind_v2/<팔>_<occ>/p.npy (같은 clip 순서 — meta 로 확인). → 03_binding_train/probe/<occ>_<과제>.json"""
    import torch
    sys.path.insert(0, str(REPO))
    base = NFS / f"release_{occ}_vobench_v2"
    M = json.load(open(base / "meta.json"))["meta"]
    vids = [m["video_id"] for m in M]
    train_g = set(json.load(open(SPLIT_V2))["train_groups"])
    grp = lambda m: "|".join(m[k] for k in ("objset", "background", "motion", "occlusion", "k"))
    groups = np.array([cell_of(m) for m in M])
    t0 = time.time()
    Z = np.load(base / "z.npy")
    P = {}
    for mdl in models:
        d = base if mdl == "release" else NFS / "bind_v2" / f"{mdl}_{occ}"
        mm = json.load(open(d / "meta.json"))["meta"] if (d / "meta.json").exists() else M
        assert [m["video_id"] for m in mm] == vids, f"{mdl}: clip 순서가 다르다"
        P[mdl] = np.load(d / "p.npy")
    PN = {k: _ln(v) for k, v in P.items()}
    print(f"loaded z {Z.shape} · p {list(P)} {time.time() - t0:.0f}s", flush=True)
    OPT["epochs"] = epochs; OPT["lr"] = 3e-4
    dev = torch.device("cuda:0")
    ctx, fut = list(range(1, NF + 1)), list(range(NF + 1, NT + 1))
    res = dict(probe=f"attentive · 구간 공유 · {OPT} · seed {seed}", occlusion=occ, split=str(SPLIT_V2.relative_to(REPO)),
               models=list(models), tasks={})
    for task, (rows, y_rows, cls) in tasks(M).items():
        if task_sel and task != task_sel:
            continue
        rows = np.array(rows); y = np.full(len(M), -1); y[rows] = y_rows
        tr = rows[np.array([grp(M[i]) in train_g for i in rows])]; te = rows[np.array([grp(M[i]) not in train_g for i in rows])]
        ks = sorted({M[i]["k"] for i in te})
        T = res["tasks"][task] = dict(classes=cls, chance=100.0 / len(cls) if task != "exist" else 50.0, ks=ks,
                                      n_train=int(len(tr)), n_test=int(len(te)), z={}, p={}, z2p={}, train_acc={})

        def stat(cor):                                            # cor: (len(te),) 정답 0/1
            d = {}
            for k in ks:
                jj = np.array([q for q, i in enumerate(te) if M[i]["k"] == k])
                d[k] = dict(n=len(jj), acc=100 * float(cor[jj].mean()), ci=boot_ci(cor[jj].astype(float), list(groups[te][jj])))
            return d

        def rows_of(A, idx, blocks, off):
            return np.ascontiguousarray(A[idx][:, [b - 1 - off for b in blocks]].reshape(-1, S_TOK, A.shape[-1]))
        # z 판독기: 학습 = train 쪽 z (블록 1–8) · 평가 = test 쪽 z + 각 모델 LN(p) (블록 9–16)
        Xte = [rows_of(Z, te, ctx, 0)] + [rows_of(PN[mdl], te, fut, NF) for mdl in models]
        ptr, pte = fit_eval(rows_of(Z, tr, ctx, 0), np.repeat(y[tr], NF), np.concatenate(Xte, 0), len(cls), seed, dev)
        T["train_acc"]["z"] = 100 * float((ptr == np.repeat(y[tr], NF)).mean())
        n1 = len(te) * NF
        own = pte[:n1].reshape(len(te), NF)
        for q, b in enumerate(ctx):
            T["z"][f"z{b}"] = stat(own[:, q] == y[te])
        for mi, mdl in enumerate(models):
            tp = pte[n1 * (1 + mi): n1 * (2 + mi)].reshape(len(te), NF)
            T["z2p"][mdl] = {f"{b}": stat(tp[:, q] == y[te]) for q, b in enumerate(fut)}
        print(f"[{occ} {task}] z head done ({time.time() - t0:.0f}s)", flush=True)
        for mdl in models:                                         # p 자기 판독 (모델마다)
            ptr, pte = fit_eval(rows_of(P[mdl], tr, fut, NF), np.repeat(y[tr], NF), rows_of(P[mdl], te, fut, NF), len(cls), seed, dev)
            T["train_acc"][f"p_{mdl}"] = 100 * float((ptr == np.repeat(y[tr], NF)).mean())
            tp = pte.reshape(len(te), NF)
            T["p"][mdl] = {f"{b}": stat(tp[:, q] == y[te]) for q, b in enumerate(fut)}
            k0 = ks[len(ks) // 2]
            print(f"[{occ} {task}] {mdl}: p self k{k0} " + " ".join(f"{T['p'][mdl][str(b)][k0]['acc']:3.0f}" for b in fut)
                  + " · z→p " + " ".join(f"{T['z2p'][mdl][str(b)][k0]['acc']:3.0f}" for b in fut) + f" ({time.time() - t0:.0f}s)", flush=True)
    BIND_OUT.mkdir(parents=True, exist_ok=True)
    fn = BIND_OUT / (f"{occ}" + (f"_{task_sel}" if task_sel else "") + ".json")
    json.dump(res, open(fn, "w"), indent=1, ensure_ascii=False)
    print("done", fn, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["stage", "probe", "pool", "linear", "attn", "attn_occ", "attn_split"])
    ap.add_argument("--models", default="release,a0,a1,a2")
    ap.add_argument("--occ", default="late", choices=["late", "visible"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=OPT["epochs"])
    ap.add_argument("--task", default=None, choices=["shape", "color", "exist"])
    ap.add_argument("--reps", default="z,p", help="z,p (기본) · h 도 쓰려면 z,h,p")
    ap.add_argument("--set", default="v1", choices=["v1", "v2"], help="linear: VOBench_v1 (split v2) · VOBench_v2 (설계 칸 2 겹)")
    ap.add_argument("--modes", default="shared,per_block")
    ap.add_argument("--fold-seed", type=int, default=None, help="v1 점검: split v2 파일 대신 장면 무작위 2 겹")
    a = ap.parse_args()
    OPT["epochs"] = a.epochs
    if a.phase == "probe":
        probe(a.gpu, a.task, tuple(a.reps.split(",")))
    elif a.phase == "linear":
        linear(a.set, a.task, tuple(a.modes.split(",")), a.fold_seed)
    elif a.phase == "attn_split":
        attn_split(a.occ, a.task, tuple(a.models.split(",")))
    elif a.phase == "attn_occ":
        attn_occ(a.occ, a.task, epochs=a.epochs if a.epochs != 100 else 20)
    elif a.phase == "attn":
        attn_shared(a.set, a.task, epochs=a.epochs if a.epochs != 100 else 20)
    else:
        {"stage": stage, "pool": pool}[a.phase]()
