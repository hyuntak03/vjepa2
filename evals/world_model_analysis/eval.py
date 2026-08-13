"""World model analysis — probing + IntPhys 방식 surprise 채점.

    python -m evals.main --fname configs/world_model_analysis/<cfg>.yaml \
                         --devices cuda:0 cuda:1 cuda:2 cuda:3

evals/main.py 가 GPU 당 프로세스를 띄우고 init_distributed 를 부른 뒤 config 의
eval_name(=world_model_analysis) 으로 이 파일의 main() 을 호출한다.

병렬화 방식 (중요)
------------------
probe head 는 DDP 로 data-parallel 하지 **않는다**. 대신 (job x target) 단위로
rank 에 나눠 각 rank 가 서로 다른 head 를 통째로 학습한다. 이유:
  * DDP data-parallel 은 effective batch = batch_size x world_size 가 되어
    단일 GPU 로 낸 기존 결과와 수치가 달라진다. head 를 나누면 수치가 그대로다.
  * head 학습은 서로 독립이라 통신이 전혀 필요 없다.
feature 추출과 surprise forward 는 비디오 단위로 rank 에 나눈다 (여기가 진짜 병목).

읽는 법
------
probing 표의 'self' 행 = 그 표현 자체로 학습·평가한 값 = 정보 존재 여부(상한).
다른 행 = 그 head 를 frozen 으로 이식한 값. 둘의 격차가 표현 형식 차이의 크기다.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist

from analysis.intphys2.model import build_from_config
from analysis.intphys2.surprise import _context_target_indices, _distance
from evals.analysis_vlm.occlusion_identity import forward as fwd
from evals.analysis_vlm.occlusion_identity import probe as probelib
from evals.analysis_vlm.occlusion_identity.cache import TokenCache
from evals.world_model_analysis import schema
from evals.world_model_analysis.data import WMADataset
import logging as _logging
import sys as _sys

# evals/main.py 의 process_main 이 logging.basicConfig() 로 루트 로거를 다시 잡고
# rank != 0 은 ERROR 로 낮춘다. 그 설정과 무관하게 모든 rank 의 진행 상황이 보이도록
# 전용 로거에 stderr 핸들러를 직접 붙이고 propagate 를 끈다 (중복 출력 방지).
# 로그가 안 보이면 어느 rank 가 어디서 멈췄는지 진단할 방법이 없다.
logger = _logging.getLogger("wma")
if not logger.handlers:
    _h = _logging.StreamHandler(_sys.stderr)
    _h.setFormatter(_logging.Formatter("[wma] %(message)s"))
    logger.addHandler(_h)
    logger.propagate = False
logger.setLevel(_logging.INFO)


def _rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _gather(obj):
    """모든 rank 의 파이썬 객체를 리스트로 모은다 (rank 0 에서만 쓴다)."""
    r, ws = _rank()
    if ws == 1:
        return [obj]
    out = [None] * ws
    dist.all_gather_object(out, obj)
    return out


# --------------------------------------------------------------------- feature

def extract(ds, cfg, sources, device, recache=False, smoke=False):
    """토큰을 base 시퀀스 단위로 memmap 에 캐시하고, 읽기 전용 핸들을 돌려준다."""
    rank, ws = _rank()
    m, f = cfg["model"], cfg.get("features", cfg.get("surprise", {}))
    ts = int(m.get("tubelet_size", 2))
    spatial = (int(m.get("img_size", 256)) // int(m.get("patch_size", 16))) ** 2
    n_frames, ctx = ds.n_frames, int(f.get("context_length", 32))
    embed = {"vit_large": 1024, "vit_huge": 1280, "vit_giant": 1408}[m.get("arch_name", "vit_large")]
    counts = fwd.base_token_counts(sources, n_frames, ctx, ts, spatial)

    cache = TokenCache(f["cache_dir"], cfg.get("tag", "wma"), [r.video_id for r in ds.records],
                       counts, embed, dtype=f.get("cache_dtype", "float16"))
    if rank == 0:
        logger.info(f"[feat] bases {counts} -> {cache.nbytes()/2**30:.1f} GiB @ {cache.dir}")
    if not recache and cache.matches():
        if rank == 0:
            logger.info("[feat] cache hit")
        return cache.open_read()

    bundle = build_from_config(m, device)
    mm = cache.open_write() if rank == 0 else None
    _barrier()
    if rank != 0:                       # rank 0 이 파일을 만든 뒤에 열어야 한다
        mm = {b: np.load(cache._path(b), mmap_mode="r+") for b in counts}

    idx = list(range(rank, len(ds), ws))          # 비디오를 rank 로 분배
    kw = dict(context_length=ctx, mask_index=int(f.get("mask_index", 0)),
              out_dtype=getattr(torch, f.get("cache_dtype", "float16")))
    bs = int(f.get("batch_size", 4))
    t_dec = t_fwd = t_wr = 0.0                      # 디코딩 / GPU forward / 디스크 쓰기
    t_all = time.time()
    for s in range(0, len(idx), bs):
        chunk = idx[s : s + bs]
        t0 = time.time()
        clips = torch.stack([ds.clip(i) for i in chunk])
        t1 = time.time()
        out = fwd.extract_batch(clips, bundle, sources, debug=(smoke and s == 0 and rank == 0), **kw)
        t2 = time.time()
        for b, v in out.items():
            arr = v.numpy()
            for k, i in enumerate(chunk):
                mm[b][i] = arr[k]
        t_dec += t1 - t0; t_fwd += t2 - t1; t_wr += time.time() - t2
        if (s // bs) % 10 == 0:
            logger.info(f"[feat] rank{rank} {s+len(chunk)}/{len(idx)} "
                        f"(decode {t_dec:.1f}s / fwd {t_fwd:.1f}s / write {t_wr:.1f}s)")
    logger.info(f"[feat] rank{rank} 끝 {len(idx)}개 {time.time()-t_all:.1f}s "
                f"= decode {t_dec:.1f}s + fwd {t_fwd:.1f}s + write {t_wr:.1f}s")
    for v in mm.values():
        v.flush()
    del bundle, mm
    torch.cuda.empty_cache()
    _barrier()
    if rank == 0:
        cache.finalize({b: np.load(cache._path(b), mmap_mode="r") for b in counts})
        logger.info(f"[feat] cached -> {cache.dir}")
    _barrier()
    return cache.open_read()


# --------------------------------------------------------------------- probing

def run_probing(ds, bases, cfg, device):
    rank, ws = _rank()
    P = cfg["probing"]
    m = cfg["model"]
    ts = int(m.get("tubelet_size", 2))
    spatial = (int(m.get("img_size", 256)) // int(m.get("patch_size", 16))) ** 2
    ctx = int(cfg.get("features", cfg.get("surprise", {})).get("context_length", 32))

    jobs, sources = schema.expand(P, ds.n_frames, ctx, ts)
    spec = P["probes"][P.get("probe", "attentive")]
    opt = P["optims"][P.get("optim", "attn_default")]
    if rank == 0:
        logger.info("[probe] 학습 계획 (train 1개당 head 1개, eval 만큼 평가)\n"
                    + schema.describe(jobs, sources, ts, spatial))

    def view(name):
        a, b = fwd.token_slice(sources[name], ts, spatial)
        return bases[sources[name]["base"] if "base" in sources[name]
                     else fwd.base_name(sources[name]["encoder"], sources[name]["window"])][:, a:b]

    tnames = list(P["targets"])
    train_m, val_m = ds.split_masks()
    va = np.nonzero(val_m)[0]
    n_train_of = [int((train_m & (ds.group_mask(j["groups"]) if j["groups"]
                                  else np.ones_like(train_m))).sum()) for j in jobs]

    # ---- 모든 rank 가 같은 검증을 먼저 한다 -----------------------------------
    # 한 rank 만 죽으면 나머지는 아래 _gather 에서 영원히 spin 한다 (GPU 100% 인데
    # 진행이 없다). 그래서 학습을 시작하기 전에 전체 job 을 다 같이 검사해서
    # 죽더라도 모든 rank 가 동시에 같은 이유로 죽게 만든다.
    bad = [f"{j['fit']}(groups={j['groups']})" for j, n in zip(jobs, n_train_of) if n == 0]
    if bad:
        raise ValueError(
            "학습셋이 비어 있는 job 이 있다: " + ", ".join(bad)
            + f"\n  데이터의 group_column 값 = {ds.groups} / train 마스크 {int(train_m.sum())}개."
            + "\n  fit_groups_sweep 에 데이터에 없는 그룹이 있거나, limit 표본이 한쪽 그룹에 쏠렸다.")

    # ---- head 를 토큰 수 가중으로 분배한다 ------------------------------------
    # units[rank::ws] 는 4096-token head 가 특정 rank 에 몰릴 수 있고, 실제로
    # fit_groups_sweep 과 주기가 맞으면 그룹까지 rank 별로 쏠린다.
    # attentive probe 는 T-token self-attention 이라 비용이 T^2 에 가깝게 늘어난다.
    def _cost(ji, _t):
        j = jobs[ji]
        tok = lambda nm: fwd.token_slice(sources[nm], ts, spatial)[1] - fwd.token_slice(sources[nm], ts, spatial)[0]
        unit = lambda T: T * (T + 1024)                     # attn(T^2) + mlp(T*D)
        c = 3 * int(opt["num_epochs"]) * n_train_of[ji] * unit(tok(j["fit"]))   # fwd+bwd
        return c + sum(len(va) * unit(tok(e)) for e in j["evals"])              # eval fwd
    units = [(ji, t) for ji in range(len(jobs)) for t in tnames]
    load, buckets = [0] * ws, [[] for _ in range(ws)]
    for u in sorted(units, key=lambda u: -_cost(*u)):        # LPT greedy
        k = min(range(ws), key=lambda i: (load[i], i))
        buckets[k].append(u)
        load[k] += _cost(*u)
    mine = buckets[rank]
    if rank == 0:
        logger.info("[probe] head 분배(가중) " + " ".join(
            f"rank{i}:{len(buckets[i])}개/{load[i]/max(1,max(load)):.2f}" for i in range(ws)))

    results = []
    for ji, tname in mine:
        job = jobs[ji]
        K = len(P["targets"][tname]["classes"])
        y = ds.labels(tname)
        fit_m = train_m & (ds.group_mask(job["groups"]) if job["groups"] else np.ones_like(train_m))
        tr = np.nonzero(fit_m)[0]
        t0 = time.time()
        head_tag = f"{job['fit']}({tname}, groups={job['groups']})"
        logger.info(f"[probe] rank{rank} {head_tag} 학습 시작 "
                    f"n_train={len(tr)} epochs={opt['num_epochs']} "
                    f"[{len(results)+1}/{len(mine)}]")

        def _ep(ep, n_ep, loss, acc, _tag=head_tag, _t=t0):
            el = time.time() - _t
            logger.info(f"[probe] rank{rank} {_tag} ep {ep}/{n_ep} "
                        f"loss={loss:.4f} acc={acc:.4f} | {el:.0f}s 경과, "
                        f"남은 예상 {el/ep*(n_ep-ep):.0f}s")

        pr = probelib.train_probe(view(job["fit"])[tr], y[tr], spec, K,
                                  num_epochs=int(opt["num_epochs"]), batch_size=int(opt["batch_size"]),
                                  lr=float(opt["lr"]), weight_decay=float(opt["weight_decay"]),
                                  device=device, log=_ep)
        t_fit = time.time() - t0
        rec = {"fit": job["fit"], "groups": job["groups"], "target": tname,
               "n_train": int(fit_m.sum()), "train_acc": pr.train_acc,
               "chance": 1.0 / K, "evals": {}}
        yv = y[va]
        for ev in job["evals"]:
            pred = probelib.predict(pr, view(ev), rows=va)     # val 행만 forward
            cell = {"overall": probelib.accuracy(pred, yv), "per_group": {}}
            for g in ds.groups:
                sel = torch.from_numpy((val_m & ds.group_mask([g]))[va])
                cell["per_group"][g] = {"n": int(sel.sum()),
                                        "acc": probelib.accuracy(pred[sel], yv[sel]),
                                        "bacc": probelib.balanced_accuracy(pred[sel], yv[sel])}
            rec["evals"][ev] = cell
        results.append(rec)
        logger.info(f"[probe] rank{rank} {job['fit']}({tname}, groups={job['groups']}) "
                    f"train_acc={pr.train_acc:.4f} | fit {t_fit:.1f}s "
                    f"eval {time.time()-t0-t_fit:.1f}s")
    logger.info(f"[probe] rank{rank} head {len(mine)}개 완료, gather 대기")
    return [r for part in _gather(results) for r in part]


# -------------------------------------------------------------------- surprise

@torch.inference_mode()
def run_surprise(ds, cfg, device):
    """비디오마다 독립 forward (block 안에서 context 가 공유되지 않는다)."""
    rank, ws = _rank()
    S = cfg["surprise"]
    bundle = build_from_config(cfg["model"], device)
    ts, spatial = bundle.tubelet_size, bundle.num_spatial_tokens
    ctx, N = int(S.get("context_length", 32)), ds.n_frames
    bs = int(S.get("batch_size", 4))
    ci, ti = _context_target_indices(ctx_frames=ctx, tgt_frames=N - ctx, tubelet_size=ts,
                                     spatial_tokens=spatial, batch_size=bs, device=device)
    idx = list(range(rank, len(ds), ws))
    out = {}
    t_all = time.time()
    logger.info(f"[surprise] rank{rank} 시작 {len(idx)}개 (batch {bs})")
    for s in range(0, len(idx), bs):
        chunk = idx[s : s + bs]
        x = torch.stack([ds.clip(i) for i in chunk]).to(device, bundle.dtype)
        n = len(chunk)
        z = bundle.context_encoder(x, masks=[ci[:n]])
        p = bundle.predictor(z, ci[:n], ti[:n], mask_index=int(S.get("mask_index", 0))).float()
        h = bundle.target_encoder(x)
        h = torch.gather(h, 1, ti[:n].unsqueeze(-1).expand(-1, -1, h.size(-1)))
        if S.get("target_layer_norm", True):
            h = torch.nn.functional.layer_norm(h, (h.size(-1),))
        for k, i in enumerate(chunk):
            out[ds.records[i].video_id] = float(
                _distance(p[k : k + 1], h[k : k + 1].float(), S.get("distance", "l1"),
                          loss_exp=float(S.get("loss_exp", 1.0))))
        if (s // bs) % 20 == 0:                  # rank0 만 찍으면 어느 rank 가 느린지 안 보인다
            logger.info(f"[surprise] rank{rank} {s+n}/{len(idx)} "
                        f"({time.time()-t_all:.0f}s 경과)")
    logger.info(f"[surprise] rank{rank} 끝 {len(idx)}개 {time.time()-t_all:.1f}s, gather 대기")
    del bundle
    torch.cuda.empty_cache()
    merged = {}
    for d in _gather(out):
        merged.update(d)
    return merged


def score_blocks(ds, surprise, cfg):
    """block 안 (가능 2) x (불가능 2) = 4쌍 전수 비교. IntPhys relative classification."""
    per_block, dist_cnt = [], defaultdict(int)
    for bid, recs in ds.blocks().items():
        pos = [r for r in recs if r.plausible == "1"]
        neg = [r for r in recs if r.plausible != "1"]
        if not pos or not neg:
            continue
        hits = [1.0 if surprise[p.video_id] < surprise[n.video_id]
                else (0.5 if surprise[p.video_id] == surprise[n.video_id] else 0.0)
                for p in pos for n in neg]
        acc = float(np.mean(hits))
        per_block.append({"block_id": bid, "block_type": recs[0].block_type,
                          "n_pairs": len(hits), "acc": acc})
        dist_cnt[round(acc, 2)] += 1

    def agg(rows):
        return {"n_block": len(rows), "n_pair": sum(r["n_pairs"] for r in rows),
                "block_pairwise": float(np.mean([r["acc"] for r in rows])) if rows else float("nan"),
                "perfect_ratio": float(np.mean([r["acc"] == 1.0 for r in rows])) if rows else float("nan")}

    res = {"overall": agg(per_block), "chance": 0.5,
           "block_distribution": {str(k): v for k, v in sorted(dist_cnt.items())}}
    if "block_type" in (cfg["scoring"].get("breakdown") or []):
        by = defaultdict(list)
        for r in per_block:
            by[r["block_type"]].append(r)
        res["by_block_type"] = {k: agg(v) for k, v in sorted(by.items())}
    return res, per_block


# ------------------------------------------------------------------------ main

def main(args_eval, resume_preempt=False):
    cfg = args_eval
    rank, ws = _rank()
    device = torch.device("cuda:0")           # main.py 가 CUDA_VISIBLE_DEVICES 로 고정해 준다
    torch.cuda.set_device(device)

    # rank 마다 torch 가 코어를 전부 잡으면(기본값) 4랭크 x 48스레드가 48코어를 놓고 싸운다.
    # 실측: clip() 디코딩+정규화가 39ms/video -> 268ms/video 로 7배 느려졌다.
    # 여기 나눠 잡는 것만으로 feature 추출이 그만큼 돌아온다 (수치는 안 변한다).
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 8) // max(1, ws))))
    # init_distributed 는 bind 실패를 삼키고 조용히 (ws=1, rank=0) 으로 폴백한다.
    # 그러면 rank 마다 world_size 가 달라지는 split-brain 이 되는데, 각 rank 는 자기
    # world_size 만 보므로 스스로 알아챌 수 없다. run.sh 가 넘긴 기대값과 대조해서
    # 조용히 틀린 결과를 내는 대신 여기서 죽인다.
    want = os.environ.get("WMA_EXPECT_WS")
    if want and int(want) != ws:
        raise RuntimeError(
            f"rank{rank}: world_size={ws} 인데 {want} 를 기대했다. DDP 초기화가 실패했다.\n"
            f"  포트 37129 를 이전 run 이 잡고 있을 때 생긴다. 남은 프로세스를 정리하고 다시 띄울 것:\n"
            f"    ps -eo pid,args | grep spawn_main")

    out_dir = cfg["output_dir"]
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"[wma] {cfg.get('tag')} | world_size={ws} | out={out_dir}")

    ds = WMADataset(cfg, limit=cfg.get('limit'))
    if rank == 0:
        logger.info(f"[data] {ds.summary()}")

    report = {"tag": cfg.get("tag"), "world_size": ws, "n_videos": len(ds), "config": cfg}

    # ---- surprise (occlusion 처럼 위반 변이가 있는 데이터셋에서만) --------------
    if cfg.get("surprise") and cfg.get("scoring"):
        sup = run_surprise(ds, cfg, device)
        if rank == 0:
            res, per_block = score_blocks(ds, sup, cfg)
            report["surprise"] = res
            with open(os.path.join(out_dir, "per_block.json"), "w") as f:
                json.dump({"per_video_surprise": sup, "per_block": per_block}, f, indent=2)
            o = res["overall"]
            logger.info(f"[surprise] block_pairwise={o['block_pairwise']:.4f} "
                        f"(chance 0.5, block {o['n_block']}, pair {o['n_pair']}) "
                        f"| 4쌍 전부 정답 block 비율={o['perfect_ratio']:.4f}")
            for k, v in res.get("by_block_type", {}).items():
                logger.info(f"           {k:16s} {v['block_pairwise']:.4f} (block {v['n_block']})")
    _barrier()

    # ---- probing --------------------------------------------------------------
    P = cfg.get("probing") or {}
    if P.get("enabled", False):
        ts = int(cfg["model"].get("tubelet_size", 2))
        ctxl = int(cfg.get("features", cfg.get("surprise", {})).get("context_length", 32))
        _, sources = schema.expand(P, ds.n_frames, ctxl, ts)
        srcs = [dict(s, base=fwd.base_name(s["encoder"], s["window"])) for s in sources.values()]
        bases = extract(ds, cfg, srcs, device,
                        recache=bool(cfg.get("recache")), smoke=bool(cfg.get("smoke")))
        recs = run_probing(ds, bases, cfg, device)
        if rank == 0:
            report["probing"] = recs
            for r in sorted(recs, key=lambda x: (x["fit"], x["target"])):
                for ev, c in r["evals"].items():
                    tag = "self" if ev == r["fit"] else f"-> {ev}"
                    logger.info(f"[probe] {r['fit']:20s} {r['target']:6s} "
                                f"groups={str(r['groups'] or '전체'):8s} {tag:24s} "
                                f"acc={c['overall']:.4f}  (chance {r['chance']:.3f}, "
                                f"train_acc={r['train_acc']:.3f}, n_train={r['n_train']})")
    _barrier()

    if rank == 0:
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"[wma] done -> {out_dir}")
    return report
