#!/usr/bin/env python3
"""**위치 + 정체 (모양 × 색 56 조합 + 없음) 자** — presence 자의 '있다/없다' 를 57 분류로 바꾼 것 (2026-09-25 사용자 설계).

왜: presence 하나로는 "p 가 물체를 만들었는데 안 움직인다" 와 "p 가 물체를 못 만든다" 를 가를 수 없다.
     판 오탐도 '있다' 를 낸다. **정체까지 맞으면** (우연 1/56) p 가 그 물체를 만든 것이다.

자 구조 (presence 자와 query · pooling 이 같다):
  q (1280) → a = softmax(tok · q / √D) → pooled = Σ a · tok
  xy  = Linear(pooled) (2)        위치 head
  cls = Linear(pooled) (57)       정체 head: 모양 7 × 색 8 = 56 조합 (index = 모양 · 8 + 색, 이름은 정렬 순) + 56 = 없음
  파라미터 1,280 + 2,562 + 72,960 = 76,802

라벨 (미래 튜블릿 8 개, 튜블릿 = 샘플 2 개, 둘 다 만족해야 그 라벨):
  조합   물체 **전체**가 화면 안 · 판 · 구조물과 안 겹침 (= presence 자의 양성, `visible_by_sample`)
  없음   빈 장면 clip, 또는 물체가 **통째로** 화면 밖 (중심이 화면 밖으로 반폭 이상: px < −r 또는 > 288 + r, r = 반폭 17.8 px)
  제외   그 밖 전부 — 가장자리에 걸림 (색·모양이 일부 보이는데 '없음' 이라 가르치지 않는다, 2026-09-25 사용자 결정),
         튜블릿 안에서 한 샘플만 보임, 판 · 구조물과 겹침
손실: 위치 MSE (조합 칸만, cell × balance 가중) + λ · CE (조합 + 없음 칸, balance 가중), λ = 1.
split · 가중치 · 특징 · 학습 루프 설정은 presence 자 (`rollout2_presence_readout.py`) 와 같다. 특징은 재추출하지 않는다.

평가 (test, block = clip 단위, scenario 층화 50 / 15 / 35 — presence 자와 같은 split):
  정체     조합 칸에서 57-way 정답률 · '있다' 가정 56-way 정답률 · 모양 / 색 주변 확률 정답률 (우연 1/56 · 1/7 · 1/8)
  없음     P(없음) 의 AUROC (없음 칸 vs 조합 칸), val 없음 칸 5 % 오탐 문턱에서 test 오탐 · 유지율, argmax 규칙 오탐 · 유지율
  위치     조합 칸 중심 오차 (칸 = 18 px)
  무대 가족별 (plain · prop · ledge · wedge · wall · edge · empty)

출력: z_research/RollOutV3/exp_results/identity/{summary.json, <rep>/{readout.pt, preds.npz}}

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/rollout2_identity_readout.py --dry                         # 라벨 개수만 (GPU 0 장)
  $P z_research/scripts/analysis/rollout2_identity_readout.py --limit 64 --epochs 3 --reps h --gpus 1  # 배관 점검 → identity/_smoke
  $P z_research/scripts/analysis/rollout2_identity_readout.py --bias                          # 학습 뒤: 좌표 치우침 attn_bias_px.json
  $P z_research/scripts/analysis/rollout2_identity_readout.py --launch --name identity_v2      # 새 자를 exp_results/identity_v2 에 (끝나면 치우침까지)
  #   학습셋을 바꾸려면 PRESENCE_SET=<data_csv/rollout_v2_ 뒤 이름> IDENTITY_FEAT_DIR=<특징 폴더> 를 앞에 붙인다
  #   (특징은 rollout2_presence_readout.py 의 추출 단계가 만든다 — clip 순서가 index_probe.csv 와 같아야 한다)
  $P z_research/scripts/analysis/rollout2_identity_readout.py --launch                       # 본 실행: p=GPU0,1 · z=2,3 · h=4,5 동시, 적재는 p → z → h 순
"""
from __future__ import annotations
import argparse, importlib.util, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
os.environ.setdefault("PRESENCE_SET", "training_v8")
spec = importlib.util.spec_from_file_location("rp", ROOT / "z_research/scripts/analysis/rollout2_presence_readout.py")
rp = importlib.util.module_from_spec(spec); sys.modules["rp"] = rp; spec.loader.exec_module(rp)   # split · 가중치 · Shards 를 그대로 쓴다

FEAT = Path(os.environ.get("IDENTITY_FEAT_DIR", str(rp.FEAT_DEFAULT_DISK)))   # 기본 = /data2/.../cache/rollout2_<PRESENCE_SET>_feats
OUT = Path(os.environ.get("IDENTITY_OUT", ROOT / "z_research/RollOutV3/exp_results/identity"))   # 속도 측정은 임시 폴더로
S, T, D, RES, CELL, W = rp.S, rp.T, rp.D, rp.RES, rp.CELL, 288
SHAPES = ["capsule", "cone", "cube", "cylinder", "pyramid", "sphere", "torus"]
COLORS = ["blue", "cyan", "green", "magenta", "orange", "purple", "red", "yellow"]
NONE = len(SHAPES) * len(COLORS)                   # 56
NC = NONE + 1


class IdReadout(nn.Module):
    """presence 자와 같은 query · pooling. head = xy (2) + 정체 (57)."""

    def __init__(self, d=D):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) / d ** 0.5)
        self.out = nn.Linear(d, 2)
        self.cls = nn.Linear(d, NC)

    def forward(self, tok):                          # (B,256,D) fp16 → xy (B,2), logits (B,57), map (B,256)
        dt = tok.dtype
        a = torch.softmax((tok @ (self.q / D ** 0.5).to(dt)).float(), -1)
        pooled = torch.bmm(a.to(dt).unsqueeze(1), tok).squeeze(1).float()
        return self.out(pooled), self.cls(pooled), a


def labels():
    """→ rows, L (n,8,2), cls (n,8) int (−1 = 제외), 그리고 진단용 표시들."""
    rows, L, pos, _ = rp.labels()
    n = len(rows)
    arr = rp.arr
    px = np.stack([arr(r["px_x_by_sample"]) if r["px_x_by_sample"].strip() else np.zeros(32, np.float32) for r in rows])
    py = np.stack([arr(r["px_y_by_sample"]) if r["px_y_by_sample"].strip() else np.zeros(32, np.float32) for r in rows])
    rad = np.array([float(r["obj_apparent_px"] or 0) / 2 for r in rows], np.float32)[:, None]
    out = (px < -rad) | (px > W + rad) | (py < -rad) | (py > W + rad)          # 물체가 통째로 화면 밖
    out_t = out.reshape(n, 16, 2).all(2)[:, T:]
    scen = (np.stack([arr(r["scenery_intersects_by_sample"]) if r["scenery_intersects_by_sample"].strip()
                      else np.zeros(32, np.float32) for r in rows]) > 0).reshape(n, 16, 2).any(2)[:, T:]
    empty = np.array([r["scenario"] == "empty" or r["shape_pre"] == "none" for r in rows])   # training_r8: 빈 장면이 물체 arm 안에 있다
    none = (empty[:, None] | out_t) & ~pos & ~scen                              # 구조물 속은 화면 밖이어도 뺀다 (presence 자와 같다)
    combo = np.full(n, -1, np.int64)
    for i, r in enumerate(rows):
        if not empty[i]:
            combo[i] = SHAPES.index(r["shape_pre"]) * len(COLORS) + COLORS.index(r["color_pre"])
    cls = np.full((n, T), -1, np.int64)
    cls[pos] = np.repeat(combo[:, None], T, 1)[pos]
    cls[none] = NONE
    assert (combo[~empty] >= 0).all() and not (pos & none).any()
    return rows, L, cls, empty


def split(rows):
    """presence 자와 **같은** split (block 단위, scenario 층화, 50 / 15 / 35, seed 0)."""
    blk = np.array([r["block_id"] for r in rows]); sc = np.array([r["scenario"] for r in rows])
    rng = np.random.RandomState(0); part = {}
    for s_ in np.unique(sc):
        b = rng.permutation(np.unique(blk[sc == s_])); n1 = int(round(len(b) * .5)); n2 = int(round(len(b) * .65))
        for x in b[:n1]: part[x] = 0
        for x in b[n1:n2]: part[x] = 1
        for x in b[n2:]: part[x] = 2
    g = np.array([part[x] for x in blk])
    return g == 0, g == 1, g == 2


def marg(prob):
    """57 확률 → 모양 (…,7) · 색 (…,8) 주변 확률 ('있다' 조건부)."""
    p = prob[..., :NONE].reshape(*prob.shape[:-1], len(SHAPES), len(COLORS))
    return p.sum(-1), p.sum(-2)


CH = 8                                             # 적재 단위 (clip). 8 clip = 42 MB


def _read(X, cid):
    a = np.empty((len(cid),) + X.shape[1:], np.float16)
    for j, c in enumerate(cid):
        a[j] = X[c]                                  # memmap 에서 clip 하나 (5.2 MB 연속) 를 실제로 읽는다
    return a


class GpuTrain:
    """**train clip 만** GPU 에 싣는다 — 매 epoch 읽는 것은 이것뿐이다 (7,180 clip = 35 GiB).
    val · test 는 끝에 한 번 흘려 읽는다 (`stream`). 적재는 스레드 여러 개로 한다.

    ⚠️ 2026-09-25 실측 (이 노드): 특징 파일은 `/data2` HDD RAID 에 있다. 스레드 1 개 0.7 GB/s → 8 개 이상 1.5 GB/s (포화).
       페이지 캐시에 있으면 1 개 2.9 GB/s → 16 개 30.7 GB/s. presence 자의 Shards 는 스레드 1 개로 전부 (75 GB) 를 실었다.
    """

    def __init__(self, path, ids, devs, threads):
        X = np.load(path, mmap_mode="r")
        parts = np.array_split(np.sort(ids), len(devs))
        self.devs = devs
        self.owner = np.full(len(X), -1, np.int64); self.slot = np.full(len(X), -1, np.int64)
        self.shards = []
        for i, (d, sub) in enumerate(zip(devs, parts)):
            self.shards.append(torch.empty((len(sub),) + X.shape[1:], dtype=torch.float16, device=d))
            self.owner[sub] = i; self.slot[sub] = np.arange(len(sub))
        jobs = [(i, k, sub[k:k + CH]) for i, sub in enumerate(parts) for k in range(0, len(sub), CH)]

        def work(job):
            i, k, cid = job
            a = _read(X, cid)
            self.shards[i][k:k + len(cid)].copy_(torch.from_numpy(a))
            return a.nbytes
        t0 = time.time()
        with ThreadPoolExecutor(threads) as ex:
            nb = sum(ex.map(work, jobs))
        for d in devs:
            torch.cuda.synchronize(d)
        self.load_s, self.gb = time.time() - t0, nb / 1e9


def stream(path, ids, threads, ch=32, depth=8):
    """ids 를 ch clip 씩 스레드로 미리 읽어 돌려준다. 앞서 읽는 양은 depth 묶음으로 막는다 (RAM 폭주 방지)."""
    X = np.load(path, mmap_mode="r")
    chunks = [ids[k:k + ch] for k in range(0, len(ids), ch)]
    with ThreadPoolExecutor(threads) as ex:
        fut = [ex.submit(_read, X, c) for c in chunks[:depth]]
        for j, c in enumerate(chunks):
            if j + depth < len(chunks):
                fut.append(ex.submit(_read, X, chunks[j + depth]))
            yield c, fut[j].result(); fut[j] = None


def train(rep, rows, L, cls, tr, va, te, args, Wc, Wb, log=print):
    """p · z · h 하나. 이 프로세스에 보이는 GPU 전부를 train 샤드로 쓴다 (`--launch` 가 CUDA_VISIBLE_DEVICES 로 나눠 준다).

    step 안에 **CPU↔GPU 동기화가 없다** (2026-09-25 수정). presence 자 루프에는 step 마다 세 가지가 있었다 —
    손실을 float() 로 꺼내기 · 배치 번호 torch.as_tensor(…, device) · 샤드 슬롯 torch.as_tensor(…, device).
    pageable 복사는 그 장치 stream 의 앞 작업을 기다리므로 장치들이 번갈아 쉬었다.
    여기서는 epoch 마다 번호 전체를 한 번에 올리고, 손실은 GPU 에 쌓아 로그 줄에서만 꺼낸다.
    """
    devs = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    tr_i = np.where(tr)[0]
    X = GpuTrain(FEAT / f"{rep}.npy", tr_i, devs, args.threads)
    log(f"    [{rep}] train {len(tr_i)} clip = {X.gb:.1f} GB → GPU {len(devs)} 장 ({X.load_s:.0f}s, {X.gb/max(X.load_s,1e-9):.2f} GB/s)  "
        + " ".join(f"cuda:{d.index} {torch.cuda.memory_allocated(d)/2**30:.1f}GiB" for d in devs))
    if args.marker:
        Path(args.marker).touch()                    # --launch 가 이걸 보고 다음 표현의 적재를 시작한다
    nd = len(devs)
    Y = [torch.from_numpy(L).to(d) for d in devs]
    C = [torch.from_numpy(cls).to(d) for d in devs]
    pos_np = cls >= 0; is_obj = (cls >= 0) & (cls < NONE)
    Wxy_np = (Wc * Wb[:, None]).astype(np.float32); Wce_np = np.repeat(Wb[:, None], T, 1).astype(np.float32)
    WX = [torch.from_numpy(Wxy_np).to(d) for d in devs]; WC = [torch.from_numpy(Wce_np).to(d) for d in devs]
    torch.manual_seed(0)
    ms = [IdReadout().to(d) for d in devs]
    for m in ms[1:]:
        m.load_state_dict(ms[0].state_dict())
    opt = torch.optim.Adam(ms[0].parameters(), lr=args.lr, fused=True)      # 커널 하나 (foreach 는 여러 개)
    t0 = time.time()
    for ep in range(args.epochs):
        perm = np.random.default_rng(ep).permutation(tr_i)
        steps = [perm[k:k + args.bs] for k in range(0, len(perm), args.bs)]
        # epoch 의 번호를 장치별로 한 번에 올린다: 전역 clip 번호 · 샤드 슬롯 · step 경계
        gid, slot, off = [[] for _ in devs], [[] for _ in devs], [[0] for _ in devs]
        for b in steps:
            own = X.owner[b]
            for i in range(nd):
                sel = b[own == i]; gid[i].append(sel); slot[i].append(X.slot[sel]); off[i].append(off[i][-1] + len(sel))
        G_ = [torch.as_tensor(np.concatenate(g), device=d) for g, d in zip(gid, devs)]
        S_ = [torch.as_tensor(np.concatenate(s_), device=d) for s_, d in zip(slot, devs)]
        den_xy = [max(float((is_obj[b] * Wxy_np[b]).sum()), 1e-6) for b in steps]     # 분모 = **배치 전체** 가중치 합
        den_ce = [max(float((pos_np[b] * Wce_np[b]).sum()), 1e-6) for b in steps]
        acc = [torch.zeros(2, device=d) for d in devs]
        te0 = time.time()
        for si, b in enumerate(steps):
            for m in ms:
                m.zero_grad(set_to_none=True)
            losses = []
            for i in range(nd):
                a0, a1 = off[i][si], off[i][si + 1]
                if a1 == a0:
                    continue
                g = G_[i][a0:a1]
                tok = X.shards[i].index_select(0, S_[i][a0:a1]).reshape(-1, S, D)
                y = Y[i].index_select(0, g).reshape(-1, 2); c = C[i].index_select(0, g).reshape(-1)
                wx = WX[i].index_select(0, g).reshape(-1); wc = WC[i].index_select(0, g).reshape(-1)
                xy, lg, _ = ms[i](tok)
                obj = ((c >= 0) & (c < NONE)).float(); use = (c >= 0).float()
                s_xy = (((xy - y) ** 2).mean(1) * obj * wx).sum() / den_xy[si]
                s_ce = (F.cross_entropy(lg, c.clamp(min=0), reduction="none") * use * wc).sum() / den_ce[si]
                losses.append(s_xy + args.lam * s_ce)
                acc[i] += torch.stack([s_xy.detach(), s_ce.detach()]) * len(b)
            # backward 는 **한 번에** — autograd 엔진이 장치마다 스레드를 따로 돌려 두 장치의 역전파가 겹친다.
            # 장치마다 .backward() 를 부르면 하나 끝나야 다음이 시작된다 (2026-09-25 실측: GPU 사용률 24–27 %, 파이썬 병목)
            torch.autograd.backward(losses)
            if nd > 1:                               # 납작한 벡터 하나로 gradient 를 모으고 파라미터를 다시 뿌린다
                for m in ms:
                    for p_ in m.parameters():
                        if p_.grad is None:
                            p_.grad = torch.zeros_like(p_)
                gv = parameters_to_vector([p_.grad for p_ in ms[0].parameters()])
                for m in ms[1:]:
                    gv += parameters_to_vector([p_.grad for p_ in m.parameters()]).to(devs[0])
                vector_to_parameters(gv, [p_.grad for p_ in ms[0].parameters()])
            opt.step()
            if nd > 1:
                with torch.no_grad():
                    pv = parameters_to_vector(ms[0].parameters())
                    for m, d in zip(ms[1:], devs[1:]):
                        vector_to_parameters(pv.to(d), m.parameters())
        if ep % 20 == 0 or ep == args.epochs - 1:
            tot = sum(a_.cpu() for a_ in acc).numpy() / len(perm)            # 여기서만 꺼낸다
            log(f"    [{rep}] epoch {ep:3d}  xy {tot[0]:.4f}  ce {tot[1]:.4f}  ({time.time()-t0:.0f}s, "
                f"이 epoch {time.time()-te0:.2f}s, step {1000*(time.time()-te0)/len(steps):.1f} ms)")
    train_s = time.time() - t0

    # ── 예측: train 은 GPU 샤드에서, val · test 는 흘려 읽는다 ──
    for m in ms:
        m.eval()
    n = len(rows)
    XY = np.empty((n, T, 2), np.float32); PR = np.empty((n, T, NC), np.float32); A1 = np.empty((n, T), np.float32)

    def put(ids, xy, lg, a):
        XY[ids] = xy.reshape(-1, T, 2).cpu().numpy(); PR[ids] = torch.softmax(lg, -1).reshape(-1, T, NC).cpu().numpy()
        A1[ids] = a.max(-1).values.reshape(-1, T).cpu().numpy()
    t1 = time.time()
    with torch.no_grad():
        for i in range(nd):
            ids = np.where(X.owner == i)[0]; ids = ids[np.argsort(X.slot[ids])]
            for k0 in range(0, len(ids), 64):
                put(ids[k0:k0 + 64], *ms[i](X.shards[i][k0:k0 + len(ids[k0:k0 + 64])].reshape(-1, S, D)))
        rest = np.where(~tr)[0]
        for j, (cid, a) in enumerate(stream(FEAT / f"{rep}.npy", rest, args.threads)):
            i = j % nd
            put(cid, *ms[i](torch.from_numpy(a).to(devs[i]).reshape(-1, S, D)))
    log(f"    [{rep}] 학습 {train_s/60:.1f}분 · 예측 {time.time()-t1:.0f}s (val·test {len(rest)} clip 흘려 읽기)")
    rec = evaluate(rows, L, cls, XY, PR, A1, va, te)
    rec["n_params"] = int(sum(p_.numel() for p_ in ms[0].parameters()))
    rec["timing"] = dict(load_s=round(X.load_s, 1), load_gb=round(X.gb, 1), train_min=round(train_s / 60, 2), n_gpu=nd)
    d = out_root(args) / rep; d.mkdir(parents=True, exist_ok=True)
    torch.save(ms[0].state_dict(), d / "readout.pt")
    np.savez_compressed(d / "preds.npz", xy=XY, prob=PR.astype(np.float16), top1=A1, truth=L, cls=cls, train=tr, val=va, test=te,
                        video_id=np.array([r["video_id"] for r in rows]), scenario=np.array([r["scenario"] for r in rows]))
    del X, ms; torch.cuda.empty_cache()
    return rec


def evaluate(rows, L, cls, XY, PR, A1, va, te):
    obj = (cls >= 0) & (cls < NONE); none = cls == NONE
    to, tn = te[:, None] & obj, te[:, None] & none
    pred57 = PR.argmax(-1); pred56 = PR[..., :NONE].argmax(-1)
    ps, pc = marg(PR); s_true, c_true = cls // len(COLORS), cls % len(COLORS)
    pnone = PR[..., NONE]
    vn = 1 - pnone[va[:, None] & none]
    thr = float(np.quantile(vn, 0.95)) if len(vn) else 0.5                     # '있다' 점수 = 1 − P(없음), val 없음 칸 5 % 오탐
    err = np.linalg.norm(XY - L, axis=-1) * RES / CELL
    fam = np.array([rp.family(r["scenario"]) for r in rows])[:, None].repeat(T, 1)
    rec = dict(
        n_test=dict(obj=int(to.sum()), none=int(tn.sum())),
        acc57=float((pred57[to] == cls[to]).mean()), acc56_given_present=float((pred56[to] == cls[to]).mean()),
        shape_acc=float((ps.argmax(-1)[to] == s_true[to]).mean()), color_acc=float((pc.argmax(-1)[to] == c_true[to]).mean()),
        chance=dict(combo=1 / NONE, shape=1 / len(SHAPES), color=1 / len(COLORS)),
        none_auroc=rp.roc_auc(np.r_[1 - pnone[to], 1 - pnone[tn]], np.r_[np.ones(int(to.sum()), bool), np.zeros(int(tn.sum()), bool)]),
        thr_val_fpr5=thr, recall_at_thr=float((1 - pnone[to] > thr).mean()), fpr_at_thr=float((1 - pnone[tn] > thr).mean()),
        argmax_recall=float((pred57[to] != NONE).mean()), argmax_fpr=float((pred57[tn] != NONE).mean()),
        center_mean=float(err[to].mean()), center_med=float(np.median(err[to])),
        attn_top1_obj=float(A1[to].mean()), attn_top1_none=float(A1[tn].mean()) if tn.any() else None,
        shape_confusion=np.bincount(s_true[to] * 7 + ps.argmax(-1)[to], minlength=49).reshape(7, 7).tolist(),
        color_confusion=np.bincount(c_true[to] * 8 + pc.argmax(-1)[to], minlength=64).reshape(8, 8).tolist(),
        by_family={})
    for fm in sorted(set(fam[:, 0])):
        mo, mn = to & (fam == fm), tn & (fam == fm)
        rec["by_family"][fm] = dict(
            n_obj=int(mo.sum()), n_none=int(mn.sum()),
            acc57=float((pred57[mo] == cls[mo]).mean()) if mo.any() else None,
            shape_acc=float((ps.argmax(-1)[mo] == s_true[mo]).mean()) if mo.any() else None,
            color_acc=float((pc.argmax(-1)[mo] == c_true[mo]).mean()) if mo.any() else None,
            fpr_at_thr=float((1 - pnone[mn] > thr).mean()) if mn.any() else None,
            center_mean=float(err[mo].mean()) if mo.any() else None)
    return rec


def out_root(args):
    return OUT / "_smoke" if args.limit else OUT


def summarize(r):
    return (f"정체 57 {100*r['acc57']:.1f} · 56|있다 {100*r['acc56_given_present']:.1f} · 모양 {100*r['shape_acc']:.1f} · "
            f"색 {100*r['color_acc']:.1f} · 없음 AUROC {r['none_auroc']:.3f} · 문턱 오탐 {100*r['fpr_at_thr']:.1f} / 유지 "
            f"{100*r['recall_at_thr']:.1f} · 중심 {r['center_mean']:.2f} 칸")


def write_summary(o, reps, a, n):
    prev = json.loads((o / "summary.json").read_text()) if (o / "summary.json").exists() else {}
    prev.setdefault("reps", {}).update(reps)
    prev.update(n_clip=n, epochs=a.epochs, lr=a.lr, bs=a.bs, lam=a.lam, train_set=os.environ["PRESENCE_SET"], shapes=SHAPES, colors=COLORS,
                none_index=NONE, label_rule="조합 = 물체 전체 화면 안 · 판/구조물 안 겹침 | 없음 = 빈 장면 또는 통째로 화면 밖 | 제외 = 나머지")
    (o / "summary.json").write_text(json.dumps(prev, indent=1, ensure_ascii=False))


def launch(a):
    """표현마다 자식 프로세스 하나 (GPU 묶음 따로). **적재 순서 = --reps 순서**: 앞 표현이 GPU 에 다 실리면 (표시 파일)
    다음 표현을 띄운다 — 디스크 (HDD RAID 1.5 GB/s) 를 두 프로세스가 나눠 쓰지 않게. 학습은 겹쳐 돈다."""
    plan = dict(x.split("=") for x in a.plan.split())
    o = out_root(a); o.mkdir(parents=True, exist_ok=True)
    procs = {}
    for rep in a.reps:
        d = o / rep; d.mkdir(parents=True, exist_ok=True)
        mk = d / ".loaded"; mk.unlink(missing_ok=True)
        cmd = [sys.executable, "-u", __file__, "--reps", rep, "--child", "--marker", str(mk), "--epochs", str(a.epochs),
               "--lr", str(a.lr), "--bs", str(a.bs), "--lam", str(a.lam), "--threads", str(a.threads)]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        if a.name:
            cmd += ["--name", a.name]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": plan[rep]}
        logf = (d / "train.log").open("w")
        procs[rep] = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
        print(f"[launch] {rep} → GPU {plan[rep]} · 로그 {d/'train.log'}", flush=True)
        while not mk.exists() and procs[rep].poll() is None:
            time.sleep(2)
        if procs[rep].poll() not in (None, 0):
            print(f"[launch] {rep} 가 적재 중에 죽었다 (exit {procs[rep].returncode}) — {d/'train.log'}", flush=True)
        else:
            print(f"[launch] {rep} 적재 끝 → 다음", flush=True)
    bad = []
    for rep, pr in procs.items():
        pr.wait(); print(f"[launch] {rep} 끝 (exit {pr.returncode})", flush=True)
        if pr.returncode:
            bad.append(rep)
    reps = {}
    for rep in a.reps:
        f = o / rep / "summary_rep.json"
        if f.exists():
            reps[rep] = json.loads(f.read_text()); print(f"  [{rep}] {summarize(reps[rep])}", flush=True)
    write_summary(o, reps, a, int(json.loads((o / a.reps[0] / "summary_rep.json").read_text()).get("n_clip", 0)) if reps else 0)
    if not bad and not a.limit:
        write_bias()                                   # 그림 스크립트가 자 폴더에서 읽는다 (attn_bias_px.json)
    print(f"→ {o}/summary.json" + (f"  ⚠️ 실패: {bad}" if bad else ""), flush=True)
    sys.exit(1 if bad else 0)


def write_bias():
    """좌표 치우침 `attn_bias_px.json` — presence 자와 **같은 정의**: test 분할 × 조합 (양성) 칸의 평균 (pred − truth) px.
    이름도 같게 둔다 (그림 스크립트가 자 폴더에서 이 이름으로 읽는다). 쓰는 쪽은 읽은 좌표에서 이 값을 뺀다."""
    out = {}
    for rep in ("p", "z", "h"):
        f = OUT / rep / "preds.npz"
        if not f.exists():
            continue
        z = np.load(f)
        m = z["test"][:, None] & (z["cls"] >= 0) & (z["cls"] < NONE)
        e = (z["xy"] - z["truth"]) * RES
        out[rep] = [round(float(e[m][:, 0].mean()), 2), round(float(e[m][:, 1].mean()), 2)]
    (OUT / "attn_bias_px.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias", action="store_true", help="attn_bias_px.json 만 쓴다 (학습 뒤)")
    ap.add_argument("--reps", nargs="+", default=["p", "z", "h"], choices=["p", "z", "h"], help="순서 = 적재 순서")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=32, help="batch = clip 수 (presence 자와 같다)")
    ap.add_argument("--lam", type=float, default=1.0, help="정체 손실 가중 (튜닝하지 않는다)")
    ap.add_argument("--threads", type=int, default=16, help="특징 읽기 스레드 (디스크는 8 개 이상에서 포화)")
    ap.add_argument("--limit", type=int, default=0, help="앞 N/2 · 뒤 N/2 clip 만 (배관 점검, 특징도 그만큼만 읽는다)")
    ap.add_argument("--dry", action="store_true", help="라벨 개수만 세고 끝")
    ap.add_argument("--launch", action="store_true", help="표현마다 자식 프로세스 · GPU 묶음 (--plan)")
    ap.add_argument("--plan", default="p=0,1 z=2,3 h=4,5", help="표현 → CUDA_VISIBLE_DEVICES")
    ap.add_argument("--name", default=None,
                    help="자 폴더 이름 → exp_results/<이름>/ (기본 identity). 새 자를 옛 자 옆에 둘 때. 그림은 new_archive_redraw.py --decoder <이름>")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--marker", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    global OUT
    if a.name:
        OUT = ROOT / "z_research/RollOutV3/exp_results" / a.name
    if a.bias:
        write_bias()
        return
    if a.launch:
        launch(a)
        return

    rows, L, cls, empty = labels()
    Wc, Wb = rp.weights(rows)
    tr, va, te = split(rows)
    n = len(rows)
    cnt = np.bincount(cls[cls >= 0], minlength=NC)
    print(f"[data] {n} clip (빈 장면 {int(empty.sum())}) · 미래 튜블릿 {n*T}: 조합 {int(cnt[:NONE].sum())} "
          f"(조합당 {cnt[:NONE].min()}–{cnt[:NONE].max()}) · 없음 {int(cnt[NONE])} "
          f"(빈 장면 {int((cls[empty] == NONE).sum())} · 통째로 화면 밖 {int((cls[~empty] == NONE).sum())}) · 제외 {int((cls < 0).sum())}", flush=True)
    print(f"[split] train {tr.sum()} / val {va.sum()} / test {te.sum()} clip", flush=True)
    if a.dry:
        return
    global FEAT
    if a.limit:                                    # 앞 N/2 · 뒤 N/2 clip 만 — 특징도 잘라 /dev/shm 에 둔다
        keep = np.r_[np.arange(a.limit // 2), np.arange(n - a.limit // 2, n)]      # 앞 (물체) · 뒤 (빈 장면 포함) 반반
        rows = [rows[i] for i in keep]; L, cls, Wc, Wb = L[keep], cls[keep], Wc[keep], Wb[keep]
        tr, va, te = tr[keep], va[keep], te[keep]
        tr[: a.limit // 2], va[a.limit // 2: 3 * a.limit // 4], te[3 * a.limit // 4:] = True, True, True
        tr[a.limit // 2:] = False; va[: a.limit // 2] = False; va[3 * a.limit // 4:] = False; te[: 3 * a.limit // 4] = False
        small = Path(f"/dev/shm/identity_smoke_{os.getpid()}"); small.mkdir(exist_ok=True)
        for r_ in a.reps:
            np.save(small / f"{r_}.npy", np.load(FEAT / f"{r_}.npy", mmap_mode="r")[keep])
        FEAT = small
        n = len(rows)
    res = {}
    for rep in a.reps:
        print(f"[train] {rep}  GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}", flush=True)
        res[rep] = r = train(rep, rows, L, cls, tr, va, te, a, Wc, Wb, log=lambda *x: print(*x, flush=True))
        print(f"  [{rep}] {summarize(r)}", flush=True)
        d = out_root(a) / rep; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_rep.json").write_text(json.dumps({**r, "n_clip": n}, indent=1, ensure_ascii=False))
    if a.limit:
        import shutil; shutil.rmtree(FEAT, ignore_errors=True)
    if not a.child:
        write_summary(out_root(a), res, a, n)
        print(f"→ {out_root(a)}/summary.json", flush=True)


if __name__ == "__main__":
    main()
