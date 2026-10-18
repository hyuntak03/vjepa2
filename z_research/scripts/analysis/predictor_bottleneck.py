#!/usr/bin/env python3
"""predictor 병목 진단 (2026-10-12) — release · AR ep40 · AR-state ep40 를 같은 판 (VOB_readout_primitive, 직선 등속, 가림 없음) 에서.

질문: "성능이 안 나온다" 가 아니라 **어디서 운동 · 정체 정보가 사라지고, 그게 무엇을 요구하나.**
VOB_readout 는 32 장 10 fps → 16 블록 (튜블릿 2). 문맥 C = 8 블록, 예측 K = 8 블록 (k = 1..8 → 블록 t = 7 + k).

조건 (AR · AR-state 만; release 는 한 번에 8 블록이라 roll 하나):
  roll  : 문맥 8 블록 → 자기 예측을 되먹이며 8 걸음 (표준 rollout)
  tf    : 참 블록을 넣는다 (teacher forcing) — 걸음 k 의 입력 = 참 블록 0..t−1
  injL  : 걸음 k (≥ 3) 의 입력 = 참 문맥 + 자기 예측 (블록 8..t−2) + **마지막 입력 블록만 참** (블록 t−1)
  injO  : 걸음 k (≥ 2) 의 입력 = 참 블록 0..t−2 + **마지막 입력 블록만 자기 예측** (tf 가 낸 블록 t−1)
걸음마다 저장 (fp16):
  자 읽기 (그 모델 p 로 배운 identity_vobprim_<name>/p 자): 위치 xy · P(없음) · 정체 top1  — roll · tf · injL · injO, 그리고 roll 의 입력 블록 위치
  은닉 st = predictor_norm 출력 (384, 그리기 전 — AR-state 에서는 이것이 되먹이는 '상태') 과 그린 출력 r (1280): 블록 전체 평균 · 참 물체 자리 3×3 평균
  참 h 블록 (같은 공간: AR 은 블록별 LN(h), release 는 창 LN(h)) 의 같은 두 풀링 — 천장
  새로움 (파라미터 없음): 토큰마다 1 − max_c cos(토큰, 문맥 토큰 c) — 출력 · 참 미래 · (복사 = 0). 전체 평균 · 물체 3×3 평균
  L1 오차 몫: |출력 − 참| 와 |복사 − 참| 의 물체 3×3 평균 · 그 밖 평균 (복사 = 문맥 마지막 블록; release 는 LN(z) 마지막 블록)

  python predictor_bottleneck.py extract --name ar40 --ckpt z_training/ariel/ar_future_1e_5/latest.pt --gpus 4
  python predictor_bottleneck.py extract --name ar_state40 --ckpt z_training/ariel/ar_state_future_1e_5/latest.pt --gpus 4
  python predictor_bottleneck.py extract --name release --gpus 4
산출: z_research/VOBench_v1/bottleneck/<name>/shard{r}.npz  (분석은 predictor_bottleneck_analyze.py)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.environ.setdefault("PRESENCE_SET", "vob_readout_primitive")
os.environ.setdefault("PRESENCE_STRIDE", "1")
os.environ.setdefault("PRESENCE_FRAMES", "/data2/local_datasets/world/world_analysis/VOB_readout_primitive")
S, NB, C, K, D = 256, 16, 8, 8, 1280
GRID = 16
CELL = 18.0                                   # 288 px 좌표에서 토큰 1 칸
NONE = 56
OUT = REPO / "z_research/VOBench_v1/bottleneck"
INDEX = REPO / "data_csv/rollout_v2_vob_readout_primitive/index.csv"
META = Path("/data2/local_datasets/world/world_analysis/VOB_readout_primitive/metadata.csv")
SPLIT = REPO / "z_research/VOBench_v1/readout/identity_vobprim_ar40/p/preds.npz"     # 자들과 같은 block 단위 split


def arr(s):
    return np.array([float(v) for v in str(s).split()], np.float32)


def select(n_train, n_test, seed=0):
    """움직이는 물체 clip (train / test, 자 split 그대로). 속도 · 장면 섞이게 무작위 추출."""
    z = np.load(SPLIT, allow_pickle=True)
    meta = {r["name"]: r for r in csv.DictReader(META.open())}
    rows = list(csv.DictReader(INDEX.open()))
    assert [r["video_id"] for r in rows] == [str(v) for v in z["video_id"]]
    rng = np.random.default_rng(seed)
    out = []
    for part, n in (("train", n_train), ("test", n_test)):
        cand = [i for i, r in enumerate(rows) if z[part][i] and meta[r["video_id"]]["has_object"] in ("1", "True", "true")
                and np.hypot(float(meta[r["video_id"]]["vx_cm_s"] or 0), float(meta[r["video_id"]]["vz_cm_s"] or 0)) > 1]
        rng.shuffle(cand)
        out += [(i, part) for i in sorted(cand[:n])]
    return out, rows


def truth(rows, i):
    """블록 중심 (16, 2) 288 px · 블록 보임 (16,)."""
    r = rows[i]
    px = arr(r["px_x_by_sample"]).reshape(16, 2).mean(1)
    py = arr(r["px_y_by_sample"]).reshape(16, 2).mean(1)
    vis = arr(r["visible_by_sample"]).reshape(16, 2).min(1) > 0
    return np.stack([px, py], -1), vis


def local_idx(xy):
    """288 px 중심 → 3×3 토큰 인덱스 (9,) (경계 자름)."""
    c = int(np.clip(xy[0] // CELL, 0, GRID - 1)); r = int(np.clip(xy[1] // CELL, 0, GRID - 1))
    ids = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = min(max(r + dr, 0), GRID - 1), min(max(c + dc, 0), GRID - 1)
            ids.append(rr * GRID + cc)
    return ids


def _worker(rank, world, name, ckpt, n_train, n_test, bs):
    import torch
    import torch.nn.functional as F
    from importlib.util import module_from_spec, spec_from_file_location
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import block_indices, encode_blocks
    from evals.world_model_analysis.data import WMADataset
    from src.models.utils.modules import PrefixSpec
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir); rp = ir.rp
    model = dict(rp.MODEL) if name == "release" else dict(rp.MODEL, predictor_checkpoint=str(REPO / ckpt))
    bundle = build_from_config(model, dev)
    pred = bundle.predictor
    kind = getattr(pred, "mask_mode", "release")
    is_ar = kind in ("ar", "ar_state")
    is_state = kind == "ar_state"
    R = ir.IdReadout().to(dev).eval()
    R.load_state_dict(torch.load(REPO / f"z_research/VOBench_v1/readout/identity_vobprim_{name}/p/readout.pt", map_location=dev))
    ds = WMADataset(dict(data=rp.DATA, model=model, features={"cache_dir": "/tmp"}, surprise={}))
    items, rows = select(n_train, n_test)
    mine = items[rank * len(items) // world:(rank + 1) * len(items) // world]
    assert all(ds.records[i].video_id == rows[i]["video_id"] for i, _ in mine[:5])
    LN = lambda x: F.layer_norm(x.float(), (x.size(-1),))

    # ---- AR 내부 한 번 돌리기 (forward_seq / rollout_state 와 같은 계산, 은닉도 돌려준다) ----
    def emb_true(hb_blocks):
        x = pred.predictor_embed(hb_blocks)
        return x + pred.type_embed[0] if is_state else x

    def emb_own(st, rend):
        return st + pred.type_embed[1] if is_state else pred.predictor_embed(rend)

    def run(tok):
        Bn, N, _ = tok.shape; nb = N // S
        idx = torch.arange(N, device=dev).unsqueeze(0).expand(Bn, -1)
        sp = PrefixSpec(1, nb - 1, S, future_causal=True) if nb > 1 else None
        h = pred._run_blocks(tok, idx, sp)
        st = pred.predictor_norm(h)
        return st, F.layer_norm(pred.predictor_proj(st), (D,))

    def blk(x, b):
        return x[:, b * S:(b + 1) * S]

    def readout(rend):                                   # (B, S, D) → xy (B,2) px, pnone, top1
        xy, lg, _ = R(rend.half())
        pr = torch.softmax(lg.float(), -1)
        return ((xy.float() + 1) * 144).cpu().numpy(), pr[:, NONE].cpu().numpy(), pr[:, :NONE].argmax(-1).cpu().numpy()

    def pools(x, loc):                                   # x (B,S,d), loc (B,9) → global mean (B,d), local mean (B,d)
        g = x.float().mean(1)
        l = torch.gather(x.float(), 1, loc.unsqueeze(-1).expand(-1, -1, x.size(-1))).mean(1)
        return g.half().cpu().numpy(), l.half().cpu().numpy()

    def novelty(x, ctxn, loc):                           # x (B,S,D), ctxn (B, C*S, D) 정규화된 문맥 → (B,), (B,)
        xn = F.normalize(x.float(), dim=-1)
        mx = torch.bmm(xn, ctxn.transpose(1, 2)).amax(-1).float()      # (B, S)
        nov = 1 - mx
        return nov.mean(1).cpu().numpy(), torch.gather(nov, 1, loc).mean(1).cpu().numpy()

    def err(x, h, loc):                                  # L1 토큰 오차 → 물체 3×3 평균, 그 밖 평균, 전체
        e = (x.float() - h.float()).abs().mean(-1).float()              # (B, S)
        m = torch.zeros_like(e).scatter_(1, loc, 1.0)
        obj = (e * m).sum(1) / m.sum(1)
        bg = (e * (1 - m)).sum(1) / (1 - m).sum(1)
        return obj.cpu().numpy(), bg.cpu().numpy(), e.mean(1).cpu().numpy()

    rec = {}

    def put(key, val):
        rec.setdefault(key, []).append(val)

    t0 = time.time(); done = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b0 in range(0, len(mine), bs):
            ch = mine[b0:b0 + bs]
            x = torch.stack([ds.clip(i) for i, _ in ch]).to(dev, dtype=bundle.dtype); B = x.size(0)
            T = np.stack([truth(rows, i)[0] for i, _ in ch]); V = np.stack([truth(rows, i)[1] for i, _ in ch])
            loc = torch.tensor(np.array([[local_idx(T[b, t]) for t in range(NB)] for b in range(B)]), device=dev)   # (B,16,9)
            put("row", np.array([i for i, _ in ch])); put("part", np.array([p for _, p in ch])); put("truth", T); put("vis", V)
            if is_ar:
                hb = encode_blocks(bundle.target_encoder, x, 2)                                    # (B, 16S, D) LN
                hfut = [blk(hb, C + k - 1) for k in range(1, K + 1)]
                ctx_ref = F.normalize(hb[:, :C * S].float(), dim=-1)
                copy = blk(hb, C - 1)
                # roll
                tok = emb_true(hb[:, :C * S]); st_roll, r_roll = [], []
                for k in range(1, K + 1):
                    st, rd = run(tok)
                    st, rd = blk(st, C + k - 2), blk(rd, C + k - 2)
                    st_roll.append(st); r_roll.append(rd)
                    tok = torch.cat([tok, emb_own(st, rd)], 1)
                # 검사: 첫 배치에서 공식 rollout 과 같은가
                if b0 == 0:
                    ref = pred.rollout(hb[:, :C * S], block_indices(B, C, S, dev), K)
                    dmax = (torch.cat(r_roll, 1).float() - ref.float()).abs().max().item()
                    dmean = (torch.cat(r_roll, 1).float() - ref.float()).abs().mean().item()
                    print(f"[r{rank}] {name} rollout 재구현 mean|Δ| = {dmean:.2e} max|Δ| = {dmax:.2e} (bf16 묶음 차; 걸음 1 은 0)", flush=True)
                # tf
                st_all, r_all = run(emb_true(hb[:, :(NB - 1) * S]))
                st_tf = [blk(st_all, C + k - 2) for k in range(1, K + 1)]; r_tf = [blk(r_all, C + k - 2) for k in range(1, K + 1)]
                # injL (k ≥ 3): 참 문맥 + 자기 예측 블록 C..t−2 + 참 블록 t−1
                st_L, r_L = {}, {}
                for k in range(3, K + 1):
                    own = [emb_own(st_roll[j], r_roll[j]) for j in range(k - 2)]     # 블록 C..C+k−3 (= t−2 까지)
                    tok = torch.cat([emb_true(hb[:, :C * S])] + own + [emb_true(blk(hb, C + k - 2))], 1)
                    st, rd = run(tok); st_L[k], r_L[k] = blk(st, C + k - 2), blk(rd, C + k - 2)
                # injO (k ≥ 2): 참 블록 0..t−2 + tf 가 낸 블록 t−1
                st_O, r_O = {}, {}
                for k in range(2, K + 1):
                    tok = torch.cat([emb_true(hb[:, :(C + k - 2) * S]), emb_own(st_tf[k - 2], r_tf[k - 2])], 1)
                    st, rd = run(tok); st_O[k], r_O[k] = blk(st, C + k - 2), blk(rd, C + k - 2)
                conds = {"roll": (st_roll, r_roll), "tf": (st_tf, r_tf)}
            else:                                                                                  # release: 창 한 번, 문맥 16 장 → 8 블록
                from analysis.intphys2.surprise import _context_target_indices
                ci, ti = _context_target_indices(ctx_frames=2 * C, tgt_frames=2 * K, tubelet_size=2, spatial_tokens=S, batch_size=B, device=dev)
                cap = {}
                hk = pred.predictor_norm.register_forward_hook(lambda m, i, o: cap.__setitem__("st", o))
                z = bundle.context_encoder(x, masks=[ci]); z = z[-1] if isinstance(z, list) else z
                p = pred(z, ci, ti, mask_index=0); hk.remove()
                stf = cap["st"][:, -K * S:]
                h = bundle.target_encoder(x); h = LN(h[-1] if isinstance(h, list) else h)
                hfut = [blk(h, C + k - 1) for k in range(1, K + 1)]
                ctx_ref = F.normalize(h[:, :C * S].float(), dim=-1)
                copy = LN(blk(z, C - 1))
                st_roll = [blk(stf, k - 1) for k in range(1, K + 1)]; r_roll = [blk(p, k - 1) for k in range(1, K + 1)]
                conds = {"roll": (st_roll, r_roll)}
                st_L = r_L = st_O = r_O = {}
            # ---- 걸음별 기록 ----
            for k in range(1, K + 1):
                t = C + k - 1; lc = loc[:, t]
                g, l = pools(hfut[k - 1], lc); put(f"h_g{k}", g); put(f"h_l{k}", l)
                a, o = novelty(hfut[k - 1], ctx_ref, lc); put(f"nov_true_all{k}", a); put(f"nov_true_obj{k}", o)
                eo, eb, ea = err(copy, hfut[k - 1], lc); put(f"err_copy_obj{k}", eo); put(f"err_copy_bg{k}", eb); put(f"err_copy_all{k}", ea)
                todo = dict(conds)
                if k in r_L: todo["injL"] = ({k: st_L[k]}, {k: r_L[k]})
                if k in r_O: todo["injO"] = ({k: st_O[k]}, {k: r_O[k]})
                for cn, (stl, rl) in todo.items():
                    st = stl[k] if isinstance(stl, dict) else stl[k - 1]; rd = rl[k] if isinstance(rl, dict) else rl[k - 1]
                    xy, pn, tp = readout(rd); put(f"{cn}_xy{k}", xy); put(f"{cn}_pn{k}", pn); put(f"{cn}_top{k}", tp)
                    g, l = pools(st, lc); put(f"{cn}_st_g{k}", g); put(f"{cn}_st_l{k}", l)
                    g, l = pools(rd, lc); put(f"{cn}_r_g{k}", g); put(f"{cn}_r_l{k}", l)
                    a, o = novelty(rd, ctx_ref, lc); put(f"{cn}_nov_all{k}", a); put(f"{cn}_nov_obj{k}", o)
                    eo, eb, ea = err(rd, hfut[k - 1], lc); put(f"{cn}_err_obj{k}", eo); put(f"{cn}_err_bg{k}", eb); put(f"{cn}_err_all{k}", ea)
            # 문맥 마지막 블록 풀링 (속도 · 정체 문맥 천장)
            if is_ar:
                g, l = pools(blk(hb, C - 1), loc[:, C - 1]); put("ctx_g", g); put("ctx_l", l)
            done += B
            if rank == 0 and (b0 // bs) % 10 == 0:
                print(f"[r0] {name} {done}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"shard{rank}.npz", **{k: np.concatenate(v) for k, v in rec.items()})
    print(f"[r{rank}] {name} done {len(mine)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=1600)
    ap.add_argument("--n-test", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    import torch
    torch.multiprocessing.spawn(_worker, args=(a.gpus, a.name, a.ckpt, a.n_train, a.n_test, a.bs), nprocs=a.gpus, join=True)
