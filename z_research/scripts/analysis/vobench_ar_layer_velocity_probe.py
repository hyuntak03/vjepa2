"""VOBench 가능 영상에서 AR predictor 의 층별 · 미래 걸음별 상태로 속도 · 운동 방향을 선형으로 읽는다 (2026-10-26).

추출 (vast ar_state ep70 기본): 블록별 LN(target_encoder) 문맥 8 블록 → rollout_kv 와 같은 KV 루프로 미래 8 블록.
  걸음 r 의 예측을 내는 토큰 (r=0: 마지막 문맥 블록 자리, r>=1: 직전 상태 토큰) 의 층별 출력을 토큰 평균 (N 축) 으로 저장.
  L 축 = 0 입력 임베딩 · 1–12 블록 출력 · 13 predictor_norm (= 상태, 384) → feat (n, 14, 8, 384)
  + p   = 최종 출력 LN(proj(상태)) 토큰 평균 (n, 8, 1280)
  + h   = 참 미래 블록 LN(target_encoder) 토큰 평균 (n, 8, 1280)      — 천장 (encoder 가 그 블록을 봤을 때)
  + hc  = 마지막 문맥 블록 LN(target_encoder) 토큰 평균 (n, 1280)       — 복사 기준선 (예측 없이 문맥 끝만)
  최종 p 가 predictor.rollout 출력과 같은지 첫 배치에서 assert.
라벨 (label.json 의 월드 좌표 x · z, cm): 미래 튜블릿 j (샘플 16+2j · 17+2j = raw 3s) 두 장 사이 속도 (cm/s).
  접선 가속도 a_j = (|v(raw f1)| − |v(raw f0)|) / 0.1 s, |v(f)| = 중앙차분 (cm/s²). 부호 3 종 = 가속 (flat_acc · ramp_down) / 등속 (static · flat_vel) / 감속 (flat_dec · ramp_up).
  가능 영상 (pos_a · pos_b) 중 물체가 있는 것만 (CLAUDE.md §1-4). 빈 장면은 뺀다.
probe: linear probe (특징 표준화 · 회귀 타깃 표준화 MSE · 분류 CE). block 안 pos_a · pos_b 가 같은 운동이라 분할은 block 단위.
  (a) (층, 걸음) 마다 따로 선형층 (AdamW 상수 lr, full batch) · block 단위 50/50 분할 (운동 층화), test 절반에서 잰다 — — 속력 R² · vx · vz R² · 가속도 R² · 방향 4 종 (static / flat / ramp_down / ramp_up) · 가속 부호 3 종 정확도
  (b) 층마다 걸음 공유 probe 하나 — 걸음에 따라 읽힌 속력이 실제처럼 바뀌나 (운동별 궤적)

  PY=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python
  $PY z_research/scripts/analysis/vobench_ar_layer_velocity_probe.py extract --gpus 4 [--name vast70 --ckpt z_training/vast/ar_state/latest.pt]
  $PY z_research/scripts/analysis/vobench_ar_layer_velocity_probe.py analyze [--name vast70]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
FR = Path(os.environ.get("VOB_FRAMES", "/local_datasets/world/testbed/VOBench_v1"))
INDEX = REPO / "data_csv/vobench_v1/index.csv"
RES_CFG = REPO / "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/_resolved.yaml"
S, T, C = 256, 8, 8
DIRS = ["static", "flat", "ramp_down", "ramp_up"]
ACCS = ["dec", "const", "acc"]


def out_dir(name):
    return REPO / f"z_analysis/ar_layer_velocity_probe/{name}"


def rows_obj():
    """가능 영상 중 물체가 있는 것."""
    out = []
    for r in csv.DictReader(INDEX.open()):
        if r["variant"] not in ("pos_a", "pos_b"):
            continue
        if (r["obj_a"] if r["variant"] == "pos_a" else r["obj_b"]) == "empty":
            continue
        out.append(r)
    return out


def _rollout_layers(pred, x_ctx, idx_ctx, n_steps):
    """rollout_kv (ar_state, action 없음, gating 없음) 를 그대로 따라가며 층별 출력의 토큰 평균을 모은다.
    return feats (B, L=depth+2, n_steps, 384) float32, p (B, n_steps*S, out)."""
    from src.models.utils.modules import PrefixSpec
    assert pred.mask_mode == "ar_state" and pred.n_act == 0 and not pred.gating_residual
    Hg, Wg = pred.grid_height, pred.grid_width
    B = x_ctx.size(0)
    feats = [[None] * n_steps for _ in range(len(pred.predictor_blocks) + 2)]
    h = pred.predictor_embed(x_ctx) + pred.type_embed[0]
    feats[0][0] = h.view(B, C, S, -1)[:, -1].float().mean(1)
    spec = PrefixSpec(1, C - 1, S, future_causal=True)
    caches = []
    for i, blk in enumerate(pred.predictor_blocks):
        h, kv = blk.forward_kv(h, idx_ctx, Hg, Wg, None, action_tokens=0, spec=spec)
        caches.append(kv)
        feats[i + 1][0] = h.view(B, C, S, -1)[:, -1].float().mean(1)
    st = pred.predictor_norm(h.view(B, C, S, -1)[:, -1])
    feats[-1][0] = st.float().mean(1)
    preds = [pred._ar_out(st)]
    pos = idx_ctx[:, -S:]
    for r in range(1, n_steps):
        pos = pos + S
        h = st + pred.type_embed[1]
        feats[0][r] = h.float().mean(1)
        for i, blk in enumerate(pred.predictor_blocks):
            h, caches[i] = blk.forward_kv(h, pos, Hg, Wg, caches[i], action_tokens=0, spec=None)
            feats[i + 1][r] = h.float().mean(1)
        st = pred.predictor_norm(h)
        feats[-1][r] = st.float().mean(1)
        preds.append(pred._ar_out(st))
    import torch
    F = torch.stack([torch.stack(f, 1) for f in feats], 1)
    return F, torch.cat(preds, 1)


def _worker(rank, world, name, ckpt):
    import torch, yaml
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import block_indices, encode_blocks
    from evals.world_model_analysis.data import WMADataset
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    cfg = yaml.safe_load(open(RES_CFG))
    model = dict(cfg["model"], predictor_checkpoint=str(REPO / ckpt))
    bundle = build_from_config(model, dev)
    pred = bundle.predictor
    data = dict(cfg["data"], index_csv="index.csv", frames_root=str(FR))
    ds = WMADataset(dict(data=data, model=model, features={"cache_dir": "/tmp"}, surprise={}))
    want = {r["video_id"] for r in rows_obj()}
    ids = [i for i, r in enumerate(ds.records) if r.video_id in want]
    mine = ids[rank * len(ids) // world:(rank + 1) * len(ids) // world]
    out = defaultdict(list); t0 = time.time(); checked = False
    with torch.no_grad():
        for k in range(0, len(mine), 16):
            ch = mine[k:k + 16]
            clips = torch.stack([ds.clip(i) for i in ch]).to(dev, dtype=bundle.dtype)
            B = clips.size(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hb = encode_blocks(bundle.target_encoder, clips, 2)            # (B, 16*S, 1280)
                idx = block_indices(B, 2 * T, S, dev)
                F, p = _rollout_layers(pred, hb[:, :C * S], idx[:, :C * S], T)
                if not checked:
                    ref = pred.rollout(hb[:, :C * S], idx[:, :C * S], T)
                    rel = ((p.float() - ref.float()).abs().mean() / ref.float().abs().mean()).item()
                    assert rel < 2e-2, f"KV 층별 루프가 rollout 과 다르다: rel {rel:.3g}"
                    print(f"[r{rank}] rollout 대조 rel {rel:.2e}", flush=True); checked = True
            hf = hb.float().view(B, 2 * T, S, -1)
            out["feat"].append(F.half().cpu().numpy())
            out["p"].append(p.float().view(B, T, S, -1).mean(2).half().cpu().numpy())
            out["h"].append(hf[:, T:].mean(2).half().cpu().numpy())
            out["hc"].append(hf[:, T - 1].mean(1).half().cpu().numpy())
            out["video_id"] += [ds.records[i].video_id for i in ch]
            if rank == 0 and k % 160 == 0:
                print(f"[r0] {k + B}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    d = out_dir(name); d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"shard{rank}.npz", video_id=np.array(out.pop("video_id")), **{k: np.concatenate(v) for k, v in out.items()})
    print(f"[r{rank}] done {len(mine)} ({time.time() - t0:.0f}s)", flush=True)


def extract(gpus, name, ckpt):
    import torch
    torch.multiprocessing.spawn(_worker, args=(gpus, name, ckpt), nprocs=gpus, join=True)


# ---------------------------------------------------------------- 분석

def load(name):
    Z = defaultdict(list)
    for f in sorted(out_dir(name).glob("shard*.npz")):
        z = np.load(f)
        for k in z.files:
            Z[k].append(z[k])
    return {k: np.concatenate(v) for k, v in Z.items()}


def labels(vids):
    """월드 속도 (n, T, 2) [vx, vz] cm/s · 문맥 끝 속도 (n, 2) · 방향 클래스 · 메타."""
    rows = {r["video_id"]: r for r in rows_obj()}
    lab, V, Vc, A, Ac, meta = {}, [], [], [], [], []
    for v in vids:
        r = rows[v]; arm = "A" if r["variant"] == "pos_a" else "B"
        b = FR / Path(r["file_name"]).parent
        if b not in lab:
            lab[b] = json.load(open(b / "label.json"))["labels"]
        x = np.array([[l["x"], l["z"]] for l in lab[b][arm]], float)
        V.append([(x[3 * (17 + 2 * j)] - x[3 * (16 + 2 * j)]) / 0.1 for j in range(T)])
        Vc.append((x[45] - x[42]) / 0.1)
        sp = lambda f: np.linalg.norm(x[f + 1] - x[f - 1]) * 15.0               # 중앙차분 속력 cm/s (30 fps)
        A.append([(sp(3 * (17 + 2 * j)) - sp(3 * (16 + 2 * j))) / 0.1 for j in range(T)])
        Ac.append((sp(45) - sp(42)) / 0.1)
        m = r["motion"]
        meta.append(dict(block=r["block_id"], motion=m, occ=r["occlusion"], objset=r["objset"],
                         acc=ACCS.index("acc" if m in ("flat_acc", "ramp_down") else ("dec" if m in ("flat_dec", "ramp_up") else "const")),
                         dir=DIRS.index("static" if m == "flat_static" else ("flat" if m.startswith("flat") else m))))
    return np.array(V), np.array(Vc), np.array(A), np.array(Ac), meta


def block_split(groups, mot, seed=0):
    """block 단위 50/50, 운동 종류로 층화 (CLAUDE.md §1-5). return train 마스크."""
    rng = np.random.default_rng(seed); tr_blocks = set()
    for m in np.unique(mot):
        ub = np.unique(groups[mot == m]); rng.shuffle(ub); tr_blocks |= set(ub[: len(ub) // 2])
    return np.array([b in tr_blocks for b in groups])


def linear_probe(X, Yr, yc, tr, epochs=2000, lr=1e-3, wd=1e-4):
    """선형층 하나 (AdamW 상수 lr, full batch, GPU). Yr (n, r) 회귀 (MSE, 타깃 표준화) · yc [(n,) 정수, k] 분류 (CE).
    특징 표준화는 train 통계. return 회귀 예측 (n, r) · 분류 예측 [(n,)]."""
    import torch
    dev = "cuda"
    X = torch.tensor(X, dtype=torch.float32, device=dev); trm = torch.tensor(tr, device=dev)
    mu, sd = X[trm].mean(0), X[trm].std(0) + 1e-6; X = (X - mu) / sd
    Yr = torch.tensor(Yr, dtype=torch.float32, device=dev); ym, ys = Yr[trm].mean(0), Yr[trm].std(0) + 1e-6
    Yn = (Yr - ym) / ys
    ks = [k for _, k in yc]; yct = [torch.tensor(y, device=dev) for y, _ in yc]
    head = torch.nn.Linear(X.size(1), Yr.size(1) + sum(ks)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        o = head(X[trm]); loss = torch.nn.functional.mse_loss(o[:, :Yr.size(1)], Yn[trm]); c = Yr.size(1)
        for y, k in zip(yct, ks):
            loss = loss + torch.nn.functional.cross_entropy(o[:, c:c + k], y[trm]); c += k
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        o = head(X); pr = (o[:, :Yr.size(1)] * ys + ym).cpu().numpy(); c = Yr.size(1); pc = []
        for k in ks:
            pc.append(o[:, c:c + k].argmax(1).cpu().numpy()); c += k
    return pr, pc


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def analyze(name):
    Z = load(name)
    V, Vc, Acc, Accc, meta = labels(Z["video_id"])
    n = len(meta)
    g = np.array([m["block"] for m in meta])
    mot = np.array([m["motion"] for m in meta]); occ = np.array([m["occ"] for m in meta])
    D = np.array([m["dir"] for m in meta]); Ak = np.array([m["acc"] for m in meta])
    spd = np.linalg.norm(V, axis=-1)                            # (n, T)
    moving = mot != "flat_static"
    reps = {f"L{l}": Z["feat"][:, l].astype(np.float32) for l in range(Z["feat"].shape[1])}
    reps["p"] = Z["p"].astype(np.float32); reps["h(천장)"] = Z["h"].astype(np.float32)
    reps["hc(복사)"] = np.repeat(Z["hc"][:, None].astype(np.float32), T, 1)
    tr = block_split(g, mot); te = ~tr; mv = moving & te
    res = {"n": n, "n_train": int(tr.sum()), "n_test": int(te.sum()), "per_t": {}, "shared": {}}
    # 라벨만의 기준선: 문맥 끝 속도로 미래 속도를 선형 예측 (등속 외삽이 얼마나 맞나)
    # + 문맥 끝 가속도까지 주면 (등가속 외삽) — 라벨이 문맥에서 얼마나 정해지나의 상한 쪽 참고
    base, base_a = {}, {}
    for t in range(T):
        P, _ = linear_probe(np.c_[Vc, np.linalg.norm(Vc, axis=1)], spd[:, t:t + 1], [], tr)
        base[t] = r2(spd[mv, t], P[mv, 0])
        P, _ = linear_probe(np.c_[Vc, np.linalg.norm(Vc, axis=1), Accc], np.c_[spd[:, t], Acc[:, t]], [], tr)
        base_a[t] = dict(speed_r2_moving=r2(spd[mv, t], P[mv, 0]), acc_r2_moving=r2(Acc[mv, t], P[mv, 1]))
    res["label_baseline_speed_r2_moving"] = base
    res["label_baseline_with_ctx_acc"] = base_a
    for rep, X in reps.items():
        row = {}
        for t in range(T):
            P, (pd, pa) = linear_probe(X[:, t], np.c_[spd[:, t], V[:, t], Acc[:, t]], [(D, len(DIRS)), (Ak, len(ACCS))], tr)
            row[t] = dict(speed_r2=r2(spd[te, t], P[te, 0]), speed_r2_moving=r2(spd[mv, t], P[mv, 0]),
                          vx_r2=r2(V[te, t, 0], P[te, 1]), vz_r2=r2(V[mv, t, 1], P[mv, 2]),
                          acc_r2=r2(Acc[te, t], P[te, 3]), acc_r2_moving=r2(Acc[mv, t], P[mv, 3]),
                          dir_acc=float((pd[te] == D[te]).mean()), accsign_acc=float((pa[te] == Ak[te]).mean()),
                          speed_mae=float(np.abs(spd[te, t] - P[te, 0]).mean()))
        res["per_t"][rep] = row
        # 걸음 공유 probe: (n*T) 행, block 단위 CV — 운동별 읽힌 속력 궤적
        PP, _ = linear_probe(X.reshape(n * T, -1), np.c_[spd.reshape(n * T), Acc.reshape(n * T)], [], np.repeat(tr, T))
        Ps = PP[:, 0].reshape(n, T); Pa = PP[:, 1].reshape(n, T)
        res["shared"][rep] = {
            "speed_r2_moving": r2(spd[mv].ravel(), Ps[mv].ravel()),
            "acc_r2_moving": r2(Acc[mv].ravel(), Pa[mv].ravel()),
            "acc_traj": {m: {"true": Acc[(mot == m) & te].mean(0).tolist(), "read": Pa[(mot == m) & te].mean(0).tolist()}
                         for m in np.unique(mot)},
            "traj": {o: {m: {"true": spd[(mot == m) & te & sel].mean(0).tolist(), "read": Ps[(mot == m) & te & sel].mean(0).tolist(),
                             "n": int(((mot == m) & te & sel).sum())} for m in np.unique(mot)}
                     for o, sel in (("all", np.ones(n, bool)), ("late", occ == "late"), ("not_late", occ != "late"))},
        }
        print(f"{rep:8s} spd " + " ".join(f"{row[t]['speed_r2_moving']:5.2f}" for t in range(T))
              + f" | acc " + " ".join(f"{row[t]['acc_r2_moving']:5.2f}" for t in range(T))
              + f" | shared spd {res['shared'][rep]['speed_r2_moving']:.2f} acc {res['shared'][rep]['acc_r2_moving']:.2f}", flush=True)
    json.dump(res, open(out_dir(name) / "probe.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract", "analyze"])
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--name", default="vast70")
    ap.add_argument("--ckpt", default="z_training/vast/ar_state/latest.pt")
    a = ap.parse_args()
    if a.phase == "extract":
        extract(a.gpus, a.name, a.ckpt)
    else:
        analyze(a.name)
