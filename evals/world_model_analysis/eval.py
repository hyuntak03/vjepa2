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

import contextlib
import json
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

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

from tqdm.auto import tqdm
import warnings as _warnings

# torch 내부(src/models/utils/modules.py)가 torch.backends.cuda.sdp_kernel() 을 쓰는데
# 2.x 에서 deprecated 라 forward 마다 FutureWarning 이 rank 수만큼 쏟아진다.
# 우리가 고칠 수 있는 코드가 아니고 동작에도 영향이 없어서 여기서 막는다.
_warnings.filterwarnings("ignore", message=r".*sdp_kernel.*", category=FutureWarning)

# 버퍼링을 코드에서 잡는다. 이게 없으면 `python -u` 로 띄워야만 진행 상황이 보이고,
# 안 그러면 로그가 블록 단위로 몰려 나와서 멈춘 것처럼 보인다.
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):        # 이미 닫혔거나 reconfigure 가 없는 스트림
        pass


class _TqdmHandler(_logging.StreamHandler):
    """진행바를 지우지 않고 로그를 찍는다.

    그냥 stderr 에 쓰면 tqdm 이 그리고 있는 줄 위에 겹쳐서 화면이 깨진다.
    tqdm.write 는 바를 잠깐 지우고 쓴 뒤 다시 그린다. 바가 없으면 그냥 stderr 다.
    """

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=_sys.stderr)
        except Exception:                        # noqa: BLE001  로깅이 실행을 막으면 안 된다
            self.handleError(record)


# evals/main.py 의 process_main 이 logging.basicConfig() 로 루트 로거를 다시 잡고
# rank != 0 은 ERROR 로 낮춘다. 그 설정과 무관하게 모든 rank 의 진행 상황이 보이도록
# 전용 로거에 stderr 핸들러를 직접 붙이고 propagate 를 끈다 (중복 출력 방지).
# 로그가 안 보이면 어느 rank 가 어디서 멈췄는지 진단할 방법이 없다.
logger = _logging.getLogger("wma")
if not logger.handlers:
    _h = _TqdmHandler(_sys.stderr)
    _h.setFormatter(_logging.Formatter("[wma] %(message)s"))
    logger.addHandler(_h)
    logger.propagate = False
logger.setLevel(_logging.INFO)


def _bar(iterable, desc, rank, total=None, unit="it"):
    """rank0 만 진행바를 그린다 (4개가 같이 그리면 화면이 엉킨다).

    파일로 리다이렉트되면(sbatch, tee) tqdm 이 \\r 로 도배하므로 그때는 끄고,
    기존 로그 줄로만 진행을 남긴다 (monitor.sh 도 그 줄을 읽는다).
    WMA_BAR=on/off 로 강제할 수 있다.
    """
    mode = os.environ.get("WMA_BAR", "auto")
    on = (rank == 0) and (mode == "on" or (mode == "auto" and _sys.stderr.isatty()))
    return tqdm(iterable, desc=desc, total=total, unit=unit, disable=not on,
                file=_sys.stderr, dynamic_ncols=True, leave=False)


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

    # ── 디코딩을 GPU forward 와 겹친다 ─────────────────────────────────────────
    # 원래는 clips = stack([ds.clip(i) ...]) 를 동기로 부르고 나서 forward 를 돌려
    # 둘이 완전히 직렬이었다. 실측 PNG 디코딩은 클립당 약 180 ms(32장 x 5.6 ms)라
    # 그동안 GPU 가 통째로 논다. 워커 스레드가 다음 배치를 미리 디코딩해 둔다.
    #   * 스레드로 충분하다 — PIL 은 디코딩 중 GIL 을 놓는다. 프로세스와 달리
    #     dataset 사본이 안 생기고 memmap 핸들도 그대로 쓴다.
    #   * ⚠️ 결과는 기존과 **비트 단위로 같다**. chunk 순서대로 future 를 받고
    #     mm[b][i] 로 인덱스를 명시하므로 완료 순서가 결과에 영향을 주지 않는다.
    #   * rank 당 CPU 를 넘기지 말 것 (§7-1: 과다구독이면 디코딩이 7배 느려진다).
    # os.cpu_count() 는 노드 전체를 세므로 SLURM 이 준 몫을 못 본다. affinity 를 쓴다.
    try:
        _ncpu = len(os.sched_getaffinity(0))
    except AttributeError:                          # 이 플랫폼엔 없다
        _ncpu = os.cpu_count() or 8
    # 실측(ViT-H, v10): 워커 8 이면 디코딩 174 -> 38 ms/clip 로 forward(118) 아래로 내려가
    # 완전히 숨는다. 16 은 과다구독으로 52 ms 로 되돌아간다 (§7-1).
    nw = int(f.get("decode_workers", 0)) or max(1, min(8, _ncpu // max(1, ws)))
    depth = max(1, int(f.get("prefetch_batches", 2)))
    chunks = [idx[s : s + bs] for s in range(0, len(idx), bs)]

    t_dec = t_fwd = t_wr = 0.0                      # 디코딩 대기 / GPU forward / 디스크 쓰기
    t_all = time.time()
    if rank == 0:
        logger.info(f"[feat] batch={bs} decode_workers={nw} prefetch={depth} "
                    f"({len(chunks)} batch/rank)")
    ex = ThreadPoolExecutor(max_workers=nw)
    pend = deque()

    def _submit(c):
        pend.append((c, [ex.submit(ds.clip, i) for i in c]))

    try:
        for c in chunks[:depth]:
            _submit(c)
        for k in _bar(range(len(chunks)), f"추출 rank{rank}", rank,
                      total=len(chunks), unit="batch"):
            chunk, futs = pend.popleft()
            if k + depth < len(chunks):
                _submit(chunks[k + depth])          # 기다리기 전에 채워 워커가 놀지 않게
            t0 = time.time()
            clips = torch.stack([fu.result() for fu in futs])
            t1 = time.time()
            out = fwd.extract_batch(clips, bundle, sources,
                                    debug=(smoke and k == 0 and rank == 0), **kw)
            t2 = time.time()
            for b, v in out.items():
                arr = v.numpy()
                for j, i in enumerate(chunk):
                    mm[b][i] = arr[j]
            t_dec += t1 - t0; t_fwd += t2 - t1; t_wr += time.time() - t2
            if k % 10 == 0:
                logger.info(f"[feat] rank{rank} {min((k + 1) * bs, len(idx))}/{len(idx)} "
                            f"(decode대기 {t_dec:.1f}s / fwd {t_fwd:.1f}s / write {t_wr:.1f}s)")
    finally:
        ex.shutdown(wait=True)
    logger.info(f"[feat] rank{rank} 끝 {len(idx)}개 {time.time()-t_all:.1f}s "
                f"= decode대기 {t_dec:.1f}s + fwd {t_fwd:.1f}s + write {t_wr:.1f}s "
                f"(workers={nw}; decode대기가 0 에 가까우면 완전히 숨은 것)")
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
    preds = []                       # 샘플별 예측 덤프 (predictions.json)
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

    # ---- head 를 rank 로 쪼개지 않는다. 모든 rank 가 모든 head 를 같이 돈다 --------
    # 예전에는 (job x target) 를 rank 에 나눠 각 rank 가 서로 다른 head 를 통째로
    # 학습했다. head 마다 비용이 크게 달라서 (contextF 4096 tok 이 나머지의 ~8배)
    # 가벼운 head 를 받은 rank 가 10분 넘게 놀았고, NCCL collective timeout(기본
    # 600s)에 걸려 job 이 통째로 죽었다. 실제로 겪었다.
    #
    # 지금은 head 하나를 모든 rank 에 올리고 미니배치를 쪼갠다 (data-parallel).
    # global batch 는 optims.batch_size 그대로고 rank 당 그 1/world_size 를 본다
    # (probe.train_probe 주석 참고). 노는 rank 가 없고, 무거운 head 도 world_size
    # 배로 빨라진다. VRAM 도 batch 가 줄어 rank 당 ~4GiB 로 내려간다.
    units = [(ji, t) for ji in range(len(jobs)) for t in tnames]
    if rank == 0:
        logger.info(f"[probe] head {len(units)}개를 {ws} rank data-parallel 로 순차 학습 "
                    f"(global batch {opt['batch_size']}, rank 당 "
                    f"{max(1, int(opt['batch_size']) // ws)})")

    results = []
    for u_i, (ji, tname) in enumerate(units):
        job = jobs[ji]
        K = len(P["targets"][tname]["classes"])
        y = ds.labels(tname)
        fit_m = train_m & (ds.group_mask(job["groups"]) if job["groups"] else np.ones_like(train_m))
        tr = np.nonzero(fit_m)[0]
        t0 = time.time()
        head_tag = f"{job['fit']}({tname}, groups={job['groups']})"
        if rank == 0:
            logger.info(f"[probe] rank{rank} {head_tag} 학습 시작 "
                        f"n_train={len(tr)} epochs={opt['num_epochs']} "
                        f"[{u_i+1}/{len(units)}]")

        def _ep(ep, n_ep, loss, acc, _tag=head_tag, _t=t0):
            if rank != 0:                      # 전 rank 가 같은 head 를 도니 한 번만 찍는다
                return
            el = time.time() - _t
            logger.info(f"[probe] rank{rank} {_tag} ep {ep}/{n_ep} "
                        f"loss={loss:.4f} acc={acc:.4f} | {el:.0f}s 경과, "
                        f"남은 예상 {el/ep*(n_ep-ep):.0f}s")

        pr = probelib.train_probe(view(job["fit"])[tr], y[tr], spec, K,
                                  num_epochs=int(opt["num_epochs"]), batch_size=int(opt["batch_size"]),
                                  lr=float(opt["lr"]), weight_decay=float(opt["weight_decay"]),
                                  device=device, log=_ep, ddp=(ws > 1),
                                  epoch_iter=lambda eps, _d=f"probe [{u_i+1}/{len(units)}] {head_tag}":
                                      _bar(eps, _d, rank, total=int(opt["num_epochs"]), unit="ep"))
        t_fit = time.time() - t0
        rec = {"fit": job["fit"], "groups": job["groups"], "target": tname,
               "n_train": int(fit_m.sum()), "train_acc": pr.train_acc,
               "chance": 1.0 / K, "evals": {}}
        yv = y[va]
        for ev in job["evals"]:
            pred = probelib.predict(pr, view(ev), rows=va, ddp=(ws > 1))   # val 행만, rank 로 분할
            cell = {"overall": probelib.accuracy(pred, yv), "per_group": {}}
            for g in ds.groups:
                sel = torch.from_numpy((val_m & ds.group_mask([g]))[va])
                cell["per_group"][g] = {"n": int(sel.sum()),
                                        "acc": probelib.accuracy(pred[sel], yv[sel]),
                                        "bacc": probelib.balanced_accuracy(pred[sel], yv[sel])}
            rec["evals"][ev] = cell
            # 샘플별 예측을 따로 모아 둔다. summary.json 은 정확도만 담아서
            # confusion matrix 같은 걸 나중에 뽑으려면 head 를 다시 학습해야 했다
            # (실제로 한 번 그렇게 낭비했다). 여기서 덤프해 두면 재학습이 필요 없다.
            preds.append({"fit": job["fit"], "groups": job["groups"], "target": tname,
                          "eval": ev, "pred": [int(v) for v in pred.tolist()]})
        results.append(rec)
        if rank == 0:
            logger.info(f"[probe] rank{rank} {head_tag} "
                        f"train_acc={pr.train_acc:.4f} | fit {t_fit:.1f}s "
                        f"eval {time.time()-t0-t_fit:.1f}s")
    if rank == 0:
        logger.info(f"[probe] head {len(units)}개 전부 완료")
    # val 행의 신원과 정답을 같이 실어 보낸다 (predictions.json 을 자기완결적으로)
    meta = {"val_video_ids": [ds.records[i].video_id for i in va],
            "val_groups": [ds.records[i].group for i in va],
            "targets": {t: {"classes": P["targets"][t]["classes"],
                            "gold": [int(v) for v in ds.labels(t)[va].tolist()]}
                        for t in tnames},
            "heads": preds}
    return results, meta   # 모든 rank 가 같은 결과를 갖는다. gather 가 필요 없다.


# -------------------------------------------------------------------- surprise

def _frame_budget(n_frames: int, skip: int, budget: str):
    """frame skip 을 적용한 뒤 실제로 쓰는 프레임 인덱스.

    official  : 공식 구현과 같은 개수. IntPhysDataset 이
                    num_frames  = 99 // frame_step          (eval.py:357)
                    length_clip = num_frames * frame_step
                    frames_all[start:start+length_clip:frame_step]
                로 자르기 때문에 100프레임 영상에서 (100-1)//skip 개가 나온다.
                skip=2 -> 49개(0..96), skip=5 -> 19개(0..90), skip=10 -> 9개(0..80).

                ⚠️ 99 는 프로토콜이 아니라 방어 코드다. 100//skip 을 쓰면 length_clip 이
                   100 이 되어 아래 줄이 터진다:
                       start = np.random.randint(0, len(frames_all) - length_clip)
                   (intphys_dataset.py:100, high=0 이면 ValueError)
                   그래서 마지막 1~3 프레임이 프로토콜 의도와 무관하게 버려진다.
                   논문 식 S3 의 t 범위는 T-(C+M) 까지라 영상 전체를 쓴다.

    full      : range(0, n_frames, skip). 논문 식 S3 를 그대로 읽은 것. window 가 1개 더 나온다.
    """
    if budget == "official":
        n = (n_frames - 1) // skip
        return [i * skip for i in range(n)]
    if budget == "full":
        return list(range(0, n_frames, skip))
    raise ValueError(f"frame_budget={budget!r}; official | full 만 된다")


def _intphys1_windows(n_frames: int, skip: int, wsize: int, ctx_lens, stride: int, tubelet: int,
                      frame_budget: str = "full"):
    """Garrido et al. A.7/A.8 의 sliding window -> [(C, ctx_frames, tgt_frames), ...].

        S_t = || p(f(V[t:t+C])) - g(V[t:t+C+M]) ||_1        (식 S2)

    - frame skip 으로 먼저 솎은 뒤 그 위에서 window 를 민다
    - t 는 stride 간격 (논문 s=2, "predicting starting from frames 1, 3, 5, etc.")
    - window 크기 = C + M 로 고정. context C 만 바뀌므로 이동 횟수는 C 와 무관하게 같다
    - t 범위는 t <= n - (C+M)

    ⚠️ growing context 는 넣지 않는다. 그건 IntPhys2 가 600프레임 + window 48 때문에
       추가한 것이고(예측이 너무 늦게 시작되는 문제), 100프레임 + window 16/32 에서는
       불필요하다. IntPhys1 은 Garrido 원 프로토콜을 따른다.
    """
    fr = _frame_budget(n_frames, skip, frame_budget)
    n = len(fr)
    out = []
    for C in ctx_lens:
        M = wsize - C
        if C <= 0 or M <= 0 or C % tubelet or M % tubelet or wsize > n:
            continue
        for t in range(0, n - wsize + 1, stride):
            out.append((C, [fr[i] for i in range(t, t + C)],
                        [fr[i] for i in range(t + C, t + wsize)]))
    return out


@torch.inference_mode()
def run_surprise_intphys1(ds, cfg, device):
    """IntPhys1 채점 프로토콜 (Garrido et al., "Intuitive physics understanding emerges...").

    config 로 켠다 (`surprise.mode: intphys1`). 안 켜면 기존 단일 window 경로 그대로다.

    논문 A.8 의 grid 를 한 번의 실행에서 전부 돈다 (모델을 한 번만 올리려고):
        frame skip x window size(C+M) 조합마다 따로 채점하고, 최고값도 같이 보고한다
        ("we use different context length ... and report the maximal accuracy across them")
    비디오당 값 두 개: AvgSurprise(짝 비교용) / MaxSurprise(단일 영상 분류용).
    시작점마다 context 길이들은 min 으로 합친다 (논문의 IntPhys 전용 처리).
    """
    rank, ws = _rank()
    S = cfg["surprise"]
    W = S.get("intphys1", {}) or {}
    bundle = build_from_config(cfg["model"], device)
    ts, spatial = bundle.tubelet_size, bundle.num_spatial_tokens

    skips = [int(x) for x in W.get("frame_skips", [2, 5, 10])]
    wsizes = [int(x) for x in W.get("window_sizes", [16, 32])]
    mult = [int(x) for x in W.get("context_mult", [2, 4, 6, 8, 10])]
    stride = int(W.get("stride", 2))
    reduce_c = str(W.get("context_reduce", "min"))
    aggs = list(W.get("aggregate", ["avg", "max"]))
    max_batch = int(W.get("max_batch", 16))          # ViT-H 는 낮춰야 한다
    # official = 공식 구현의 프레임 예산((T-1)//skip). full = 논문 식 S3 그대로. _frame_budget 참고.
    frame_budget = str(W.get("frame_budget", "full"))
    mask_index = int(S.get("mask_index", 0))
    dist, lexp = S.get("distance", "l1"), float(S.get("loss_exp", 1.0))

    # ---- autocast (공식 Garrido 구현과 맞추는 스위치) ---------------------------
    # 공식은 가중치를 fp32 로 두고 forward 만 autocast 한다:
    #   jepa-intuitive-physics/evaluation_code/evals/intuitive_physics/eval.py:437
    #     with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
    #   (플래그 이름은 bfloat16 인데 실제 dtype 은 float16 이다)
    # 우리 기본값은 그래프 전체를 bfloat16 으로 캐스팅하는 것(model.dtype)이라 정밀도가
    # 다르다. bf16 는 가수 7비트, fp16 는 10비트다.
    #   model.dtype: float32 + model.autocast: float16  -> 공식과 동일
    #   model.dtype: bfloat16 + autocast 없음            -> 예전 동작
    _AC = {"float16": torch.float16, "fp16": torch.float16,
           "bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "none": None, "": None}
    ac_name = str(cfg["model"].get("autocast", "none")).lower()
    if ac_name not in _AC:
        raise ValueError(f"model.autocast={ac_name!r}; {sorted(_AC)} 중 하나여야 한다")
    ac_dtype = _AC[ac_name]

    def _ac():
        return (torch.autocast("cuda", dtype=ac_dtype) if ac_dtype
                else contextlib.nullcontext())

    # 논문: context lengths = [2,4,6,8,10] x (C+M)/16
    combos = []
    for sk in skips:
        for wz in wsizes:
            cl = [m * wz // 16 for m in mult]
            wins = _intphys1_windows(ds.n_frames, sk, wz, cl, stride, ts, frame_budget)
            if wins:
                combos.append((f"skip{sk}_w{wz}", wins, cl))
    if not combos:
        raise ValueError("유효한 window 조합이 없다. frame_skips/window_sizes 가 영상 길이"
                         f"({ds.n_frames}프레임)에 비해 큰지 확인할 것.")
    # 시작부 설정 요약은 debug 로 내린다 (기본 INFO 라 안 찍힌다). 같은 내용이
    # summary.json 의 report["config"] 에 그대로 들어가므로 정보가 사라지지는 않는다.
    if rank == 0:
        for name, wins, cl in combos:
            logger.debug(f"[surprise] intphys1 {name}: ctx={cl} stride={stride} "
                         f"-> window {len(wins)}개/video (이동 {len(wins)//len(cl)}회 x C {len(cl)}개)")
        logger.debug(f"[surprise] 총 {sum(len(w) for _, w, _ in combos)} forward/video, "
                     f"reduce={reduce_c}, aggregate={aggs}, frame_budget={frame_budget}")

    idx = list(range(rank, len(ds), ws))
    out = {}
    t_all = time.time()
    # video_batch: 한 번에 GPU 에 올리는 영상 수.
    #
    # max_batch 만 올려도 소용이 없다. 배치는 같은 (C, M) 모양끼리만 묶이는데
    # 100프레임 기준 그 그룹 크기가 skip2_w16=18 / skip2_w32=10 / skip5_w16=3 으로
    # 이미 고정이라, max_batch 를 18 이상 줘도 배치가 더 커지지 않는다.
    # 배치를 실제로 키우려면 **영상을 여러 개 겹쳐야** 한다 (모양이 같으니 그냥 쌓인다).
    # video_batch=V 면 그룹 크기가 그대로 V 배가 되고, 그 위에서 max_batch 가 상한이 된다.
    #
    # video_batch=1 이면 예전과 배치 구성이 완전히 같다 (수치도 동일).
    # V>1 이면 배치가 커지면서 bf16 reduction 순서가 달라져 마지막 자리가 흔들릴 수 있다.
    vbs = max(1, int(W.get("video_batch", 1)))
    # window 별 원값을 남긴다. min-over-C 가 실제로 어떤 C 를 골랐는지, 그리고
    # context 가 이미 위반 프레임을 포함한 window 가 결과에 얼마나 기여했는지
    # 사후에 따지려면 집계 전 값이 있어야 한다 (평소엔 꺼둔다).
    dump_windows = bool(W.get("dump_windows", False))
    win_dump = defaultdict(list)          # video_id -> [(combo, start, C, value), ...]
    if rank == 0:
        logger.debug(f"[surprise] rank{rank} 시작 {len(idx)}개/rank "
                     f"(video_batch={vbs}, max_batch={max_batch})")
    steps = range(0, len(idx), vbs)
    # rank0 만 바를 그린다 (4개가 같이 그리면 화면이 엉킨다). 대신 전체 영상 수를 기준으로
    # 잡고 스텝마다 world_size 배로 올린다 — 모든 rank 가 같은 양을 같은 속도로 처리하므로
    # rank0 진행률 x world_size 가 전체 진행률의 좋은 추정이다.
    bar = _bar(None, "surprise", rank, total=len(ds), unit="vid")
    for j, s0 in enumerate(steps):
        vidx = idx[s0 : s0 + vbs]
        clips = [ds.clip(i).to(device, bundle.dtype) for i in vidx]
        res_of = [{} for _ in vidx]
        for name, wins, _ in combos:
            per_start = [defaultdict(list) for _ in vidx]
            # ---- window 시작점 단위로 묶는다 -------------------------------------
            # 같은 시작점 t 의 window 는 C 와 무관하게 프레임 집합이 같다
            # (cf + tf = t 부터 wsize 프레임). 그래서 target encoder 는 시작점마다
            # **한 번만** 돌리면 되고, C 는 그 출력에서 future 토큰을 어디서 자를지만
            # 정한다. 예전에는 (C, M) 모양마다 돌려서 C 개수(=5)만큼 중복 계산했다.
            # (공식 구현도 CTXT_LEN 루프 안에서 target encoder 를 다시 부른다 —
            #  eval.py:466. 결과는 같고 비용만 5배다.)
            # 실측 기준 encoder 토큰 처리량이 ~58% 줄어든다.
            #
            # 배치 구성(시작점 우선, 그 안에서 영상 순)과 chunk 경계는 예전과 동일하게
            # 유지한다. bf16 reduction 순서가 그대로라 수치가 비트 단위로 같다.
            by_start = defaultdict(list)          # 시작 프레임 -> [(C, cf, tf), ...]
            for C, cf, tf in wins:
                by_start[cf[0]].append((C, cf, tf))
            starts = sorted(by_start)
            full_of = {st: l[0][1] + l[0][2] for st, l in by_start.items()}   # 시작점 -> window 프레임
            wsize = len(full_of[starts[0]])
            cl_list = sorted({C for C, _, _ in wins})
            items = [(v, st) for st in starts for v in range(len(vidx))]
            for b0 in range(0, len(items), max_batch):     # ViT-H 는 배치를 끊어야 안 터진다
                gb = items[b0 : b0 + max_batch]
                x = torch.stack([clips[v][:, full_of[st]] for v, st in gb])
                with _ac():
                    h_full = bundle.target_encoder(x)      # ★ window 전체(C+M) 를 시작점당 1회
                for C in cl_list:
                    ci, ti = _context_target_indices(
                        ctx_frames=C, tgt_frames=wsize - C, tubelet_size=ts,
                        spatial_tokens=spatial, batch_size=len(gb), device=device)
                    with _ac():
                        z = bundle.context_encoder(x, masks=[ci])  # context 토큰만 (C 마다 필요)
                        p = bundle.predictor(z, ci, ti, mask_index=mask_index)
                        h = torch.gather(h_full, 1,
                                         ti.unsqueeze(-1).expand(-1, -1, h_full.size(-1)))
                        if S.get("target_layer_norm", True):
                            # 공식도 autocast 안에서 LN 한다 (eval.py:467). autocast 는
                            # layer_norm 을 fp32 로 승격하므로 실제 계산은 fp32 다.
                            h = torch.nn.functional.layer_norm(h, (h.size(-1),))
                    d = (p.float() - h.float()).abs()
                    sv = (d.pow(lexp).mean(dim=(1, 2)) / lexp) if dist == "l1" else d.mean(dim=(1, 2))
                    for (v, st), val in zip(gb, sv.tolist()):
                        # ★ 버킷 키는 window 시작점 t 다. 논문의
                        #   "the minimal surprise over all context lengths C for each
                        #    starting frame t" 에서 t 는 S_t = ||p(f(V[t:t+C])) - g(V[t:t+C+M])||
                        #   의 t, 즉 context 가 시작하는 프레임이다.
                        #   t+C 로 묶으면 C 마다 키가 달라져서 시작점이 다른 window 들이
                        #   한 버킷에 섞이고, 가장자리 버킷은 후보가 1개만 남아 min 이
                        #   무의미해진다 (측정: 같은 forward 결과에서 0.5 까지 벌어졌다).
                        per_start[v][st].append(val)
                        if dump_windows:
                            win_dump[ds.records[vidx[v]].video_id].append(
                                (name, int(st), int(C), float(val)))
            for v in range(len(vidx)):
                per_t = [(min(vv) if reduce_c == "min" else float(np.mean(vv)))
                         for vv in per_start[v].values()]
                res_of[v][f"{name}/avg"] = float(np.mean(per_t))
                res_of[v][f"{name}/max"] = float(np.max(per_t))
        for v, i in enumerate(vidx):
            out[ds.records[i].video_id] = res_of[v]
        del clips
        bar.update(len(vidx) * ws)
        if j % max(1, 20 // vbs) == 0:
            logger.info(f"[surprise] rank{rank} {s0+len(vidx)}/{len(idx)} "
                        f"({time.time()-t_all:.0f}s 경과)")
    bar.close()
    logger.info(f"[surprise] rank{rank} 끝 {len(idx)}개 {time.time()-t_all:.1f}s, gather 대기")
    del bundle
    torch.cuda.empty_cache()
    merged = {}
    for d in _gather(out):
        merged.update(d)
    if dump_windows:
        wd = {}
        for d in _gather(dict(win_dump)):
            wd.update(d)
        if rank == 0:
            out_dir = cfg["output_dir"]
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "per_window.json"), "w") as f:
                json.dump({"schema": ["combo", "window_start_frame", "context_length", "surprise"],
                           "windows": wd}, f)
            logger.info(f"[surprise] per-window 덤프 -> {out_dir}/per_window.json "
                        f"({sum(len(v) for v in wd.values())} 값)")
    keys = [f"{name}/{a}" for name, _, _ in combos for a in aggs]
    return merged, keys


# ---------------------------------------------------------------------------
# token_subset — 채점에 쓸 타깃 토큰을 좁힌다 (희석 제거 실험)
#
#   all             전체 타깃 토큰 (기존 동작. 항상 같이 낸다)
#   object          물체가 있어야 할 자리 주변 토큰만
#   post_divergence divergence_frame 이후 tubelet 만
#   object_postdiv  위 둘의 교집합
#
# ★ 마스크는 video 가 아니라 **block 단위**로 만든다.
#   문맥일치 쌍 (pos_obj, imp_vanish) 은 미래에 물체가 있는 쪽과 없는 쪽이다.
#   각자 자기 궤적으로 마스크를 만들면 imp_vanish 는 마스크가 비어버리고,
#   무엇보다 두 영상을 서로 다른 토큰 집합에서 재는 셈이라 비교가 성립하지 않는다.
#   그래서 block 안 4개 변이의 물체 위치를 합집합으로 묶어 "물체가 있어야 할 자리"를
#   만들고, 그 하나를 4개 영상 전부에 똑같이 적용한다.
#
# 토큰 배치는 PatchEmbed3D 그대로 — tubelet 이 바깥, 그 안에서 spatial.
#   t = tub_local * spatial + (py * grid + px),  tub_local = (frame - ctx) // tubelet
# ---------------------------------------------------------------------------
_SUBSET_KEYS = ("all", "object", "post_divergence", "object_postdiv")


def _frames_table(root: str):
    """frames.csv -> {video_id: [(frame, x_px, y_px, present), ...]}"""
    import csv
    path = os.path.join(root, "frames.csv")
    if not os.path.exists(path):
        return None
    out = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["video_id"]].append((int(r["frame"]), float(r["x_px"]),
                                       float(r["y_px"]), int(r["present"])))
    return out


def _token_subset_masks(ds, cfg, ctx: int, tubelet: int, spatial: int):
    """{mode: {video_id: bool ndarray (T,)}} 와 실제로 쓸 mode 목록을 돌려준다.

    설정이 없거나 frames.csv 가 없으면 ('all',) 만 돌려준다 (기존 동작).
    """
    TS = (cfg.get("scoring") or {}).get("token_subset") or {}
    modes = TS.get("modes")
    if modes is None:
        m = str(TS.get("mode", "all"))
        modes = ["all"] if m == "all" else ["all", m]
    modes = [m for m in modes if m in _SUBSET_KEYS]
    if "all" not in modes:
        modes = ["all"] + modes
    if modes == ["all"]:
        return {"all": None}, ["all"]

    N, res = ds.n_frames, int((cfg["data"].get("resolution") or 256))
    n_tub = (N - ctx) // tubelet
    grid = int(round(spatial ** 0.5))
    assert grid * grid == spatial, f"spatial={spatial} 가 정사각 격자가 아니다"
    patch = res / grid
    radius = int(TS.get("object_radius", 1))

    frames = _frames_table(ds.root)
    if frames is None:
        raise ValueError(f"token_subset={modes} 인데 {ds.root}/frames.csv 가 없다")

    # ---- block 단위 물체 격자 (union over variants) -------------------------
    by_block = defaultdict(list)
    for r in ds.records:
        by_block[r.block_id].append(r)

    masks = {m: {} for m in modes}
    cover = defaultdict(list)
    for bid, recs in by_block.items():
        obj = np.zeros((n_tub, grid, grid), dtype=bool)
        for r in recs:
            for f, x, y, present in frames.get(r.video_id, ()):
                if not present or f < ctx or f >= N:
                    continue
                tb = (f - ctx) // tubelet
                px, py = int(x // patch), int(y // patch)
                x0, x1 = max(0, px - radius), min(grid, px + radius + 1)
                y0, y1 = max(0, py - radius), min(grid, py + radius + 1)
                obj[tb, y0:y1, x0:x1] = True
        obj = obj.reshape(n_tub, spatial)

        dv = int(recs[0].raw.get("divergence_frame") or 0)
        post = np.zeros((n_tub, spatial), dtype=bool)
        for tb in range(n_tub):
            if ctx + tb * tubelet + tubelet - 1 >= dv:      # tubelet 의 마지막 프레임 기준
                post[tb] = True

        built = {"all": np.ones((n_tub, spatial), dtype=bool), "object": obj,
                 "post_divergence": post, "object_postdiv": obj & post}
        for m in modes:
            v = built[m].reshape(-1)
            if not v.any():
                raise ValueError(f"block {bid} 의 token_subset='{m}' 마스크가 비었다")
            cover[m].append(v.mean())
            for r in recs:
                masks[m][r.video_id] = v
    for m in modes:
        if m != "all":
            logger.info(f"[token_subset] {m:16s} 평균 커버리지 "
                        f"{np.mean(cover[m]) * 100:.2f}% ({np.mean(cover[m]) * n_tub * spatial:.0f}"
                        f"/{n_tub * spatial} 토큰)")
    masks["all"] = None                      # 전체는 마스크 없이 그냥 mean
    return masks, modes


@torch.inference_mode()
def run_surprise(ds, cfg, device):
    """비디오마다 독립 forward (block 안에서 context 가 공유되지 않는다)."""
    rank, ws = _rank()
    S = cfg["surprise"]
    bundle = build_from_config(cfg["model"], device)
    ts, spatial = bundle.tubelet_size, bundle.num_spatial_tokens
    ctx, N = int(S.get("context_length", 32)), ds.n_frames
    bs = int(S.get("batch_size", 4))
    decode_workers = int(S.get("decode_workers", 0))
    ci, ti = _context_target_indices(ctx_frames=ctx, tgt_frames=N - ctx, tubelet_size=ts,
                                     spatial_tokens=spatial, batch_size=bs, device=device)
    masks, modes = _token_subset_masks(ds, cfg, ctx, ts, spatial)
    kind, exp = S.get("distance", "l1"), float(S.get("loss_exp", 1.0))

    # Match the Garrido/IntPhys numerical path when requested: keep checkpoint
    # weights in FP32, but execute encoder/predictor/target forward in FP16.
    # Previously only the sliding path honored model.autocast; the fixed-window
    # path silently ignored it and would therefore run full FP32.
    _ac_types = {"float16": torch.float16, "fp16": torch.float16,
                 "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                 "none": None, "": None}
    ac_name = str(cfg["model"].get("autocast", "none")).lower()
    if ac_name not in _ac_types:
        raise ValueError(f"model.autocast={ac_name!r}; {sorted(_ac_types)} 중 하나여야 한다")
    ac_dtype = _ac_types[ac_name]
    if modes != ["all"] and not (kind == "l1" and exp == 1.0):
        raise ValueError(f"token_subset 은 distance=l1, loss_exp=1.0 에서만 쓴다 "
                         f"(지금 {kind}/{exp}). 토큰별로 분해되는 거리여야 부분합이 의미가 있다.")

    idx = list(range(rank, len(ds), ws))
    out = {}
    t_all = time.time()
    logger.info(f"[surprise] rank{rank} 시작 {len(idx)}개 (batch {bs}, "
                f"decode_workers {decode_workers}) | subset={modes}")
    steps = range(0, len(idx), bs)
    pool = ThreadPoolExecutor(max_workers=decode_workers) if decode_workers > 1 else None
    try:
        for s in _bar(steps, f"surprise rank{rank}", rank, total=len(steps), unit="batch"):
            chunk = idx[s : s + bs]
            clips = list(pool.map(ds.clip, chunk)) if pool else [ds.clip(i) for i in chunk]
            x = torch.stack(clips).to(device, bundle.dtype)
            n = len(chunk)
            ac = (torch.autocast("cuda", dtype=ac_dtype) if ac_dtype else contextlib.nullcontext())
            with ac:
                z = bundle.context_encoder(x, masks=[ci[:n]])
                p = bundle.predictor(z, ci[:n], ti[:n], mask_index=int(S.get("mask_index", 0)))
                h = bundle.target_encoder(x)
                h = torch.gather(h, 1, ti[:n].unsqueeze(-1).expand(-1, -1, h.size(-1)))
                if S.get("target_layer_norm", True):
                    h = torch.nn.functional.layer_norm(h, (h.size(-1),))
            p = p.float()
            if modes == ["all"]:                       # 기존 경로 그대로 (수치 보존)
                for k, i in enumerate(chunk):
                    out[ds.records[i].video_id] = {"all": float(
                        _distance(p[k : k + 1], h[k : k + 1].float(), kind, loss_exp=exp))}
            else:
                # 토큰별 오차 e[t] = mean_D |p-h|. subset 점수 = 선택된 t 의 평균.
                # mask 가 전체면 mean(e) == mean_{t,d}|p-h| 라 'all' 이 기존 값과 정확히 같다.
                e = (p - h.float()).abs().mean(-1).cpu().numpy()          # (n, T)
                for k, i in enumerate(chunk):
                    vid = ds.records[i].video_id
                    out[vid] = {m: float(e[k].mean() if masks[m] is None else e[k][masks[m][vid]].mean())
                                for m in modes}
            if (s // bs) % 20 == 0:                  # rank0 만 찍으면 어느 rank 가 느린지 안 보인다
                logger.info(f"[surprise] rank{rank} {s+n}/{len(idx)} "
                            f"({time.time()-t_all:.0f}s 경과)")
    finally:
        if pool:
            pool.shutdown()
    logger.info(f"[surprise] rank{rank} 끝 {len(idx)}개 {time.time()-t_all:.1f}s, gather 대기")
    del bundle
    torch.cuda.empty_cache()
    merged = {}
    for d in _gather(out):
        merged.update(d)
    return merged


def _block_pairs(recs, pairing: str):
    """block 안에서 채점할 (가능, 불가능) 쌍 목록.

    matched (기본, Garrido 공식 코드와 같음)
        4중항 안에서 **문맥이 일치하는 2쌍만** 본다. 짝은 index.csv 의 pair_id 로 오고,
        그건 z_scripts/world_model_analysis/build_intphys1_pairs.py 가 픽셀 분기로 구한다.
        공식 구현:
            evaluation_code/evals/intuitive_physics/utils.py:157-175
            get_breaking_points -> get_matches -> [[0,p],[나머지 둘]]
            eval.py:496  for match in matches: all_losses.append(losses[match])
        설정이 다른 쌍은 위반 전부터 문맥이 갈라져 있어서, surprise 차이가 위반 때문인지
        설정 때문인지 분리되지 않는다. 그래서 논문은 그 쌍을 아예 안 본다.

    cross
        (가능 2) x (불가능 2) = 4쌍 전수. 예전 동작. pair_id 가 없을 때만 쓸 것.
    """
    if pairing == "cross":
        pos = [r for r in recs if r.plausible == "1"]
        neg = [r for r in recs if r.plausible != "1"]
        return [(p, n) for p in pos for n in neg]

    by_pair = defaultdict(list)
    for r in recs:
        by_pair[r.pair_id].append(r)
    if "" in by_pair:
        raise ValueError(
            "scoring.pairing='matched' 인데 index.csv 에 pair_id 가 없다.\n"
            "  python z_scripts/world_model_analysis/build_intphys1_pairs.py 로 만들 것.\n"
            "  (전수 비교로 돌아가려면 scoring.pairing: cross)")
    out = []
    for pid, prs in sorted(by_pair.items()):
        pos = [r for r in prs if r.plausible == "1"]
        neg = [r for r in prs if r.plausible != "1"]
        if len(pos) != 1 or len(neg) != 1:
            raise ValueError(f"pair_id={pid} 가 가능1+불가능1 이 아니다 "
                             f"(가능 {len(pos)} / 불가능 {len(neg)}). index.csv 를 다시 만들 것.")
        out.append((pos[0], neg[0]))
    return out


def single_video_metrics(ds, surprise):
    """단일 영상 분류 (쌍 없이). 공식 compute_metrics 의 뒷부분과 같은 정의.

        evaluation_code/evals/intuitive_physics/eval.py:271-324

    논문 A.7:
        "A harder task, but closer to reality, is to look at the surprises of possible and
         impossible videos individually rather than in pairs. Being able to separate
         possible and impossible videos without pairs is significantly harder."
    그리고 Figure S1 오른쪽 열이 이 과제다. 논문은 여기에 **MaxSurprise** 를 권한다
    ("a maximum surprise score can be used on unique videos by eliminating the surprise
      contribution coming from the complexity of the scene").

    돌려주는 값
      auroc     : 불가능 영상의 surprise 가 가능 영상보다 클 확률 (동점 0.5).
                  공식 roc_curve(labels=가능, score=-surprise) 의 AUC 와 같다.
      abs_acc   : 공식 "Absolute Accuracy". 가능 영상 surprise 의 90 퍼센타일을 임계값으로
                  잡고 (가능 < th) + (불가능 > th) 를 센다. 임계값을 가능 영상만으로
                  정하므로 정직한 (라벨을 안 쓰는) 분류기다.
      best_acc  : 임계값을 100 등분해 훑은 최고 정확도 = oracle 상한.
                  ⚠️ 공식 코드는 이 줄에서 분모를 data1.shape[0] 두 번 더한다
                     (eval.py:301). 가능/불가능 수가 같으면 결과가 같다. 여기서는 제대로 쓴다.
    """
    pos = np.array([surprise[r.video_id] for r in ds.records if r.plausible == "1"], float)
    neg = np.array([surprise[r.video_id] for r in ds.records if r.plausible != "1"], float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (neg[:, None] > pos[None, :]).sum() + 0.5 * (neg[:, None] == pos[None, :]).sum()
    auroc = float(gt / (len(pos) * len(neg)))

    th = float(np.sort(pos)[min(int(np.ceil(0.90 * len(pos))), len(pos) - 1)])
    abs_acc = float(((pos < th).sum() + (neg > th).sum()) / (len(pos) + len(neg)))

    grid = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 100)
    accs = [((pos < t).sum() + (neg > t).sum()) / (len(pos) + len(neg)) for t in grid]
    b = int(np.argmax(accs))
    return {"auroc": auroc, "abs_acc": abs_acc, "abs_threshold": th,
            "best_abs_acc": float(accs[b]), "best_threshold": float(grid[b]),
            "n_possible": len(pos), "n_impossible": len(neg)}


def score_blocks(ds, surprise, cfg):
    """block 안에서 (가능, 불가능) 쌍을 비교한다. IntPhys relative classification.

    쌍을 어떻게 고르는지는 scoring.pairing 이 정한다 (_block_pairs 주석 참고).
    """
    # 기본은 cross (예전 동작). pair_id 가 없는 기존 셋(occlusion_v2, intphys1_dev)이
    # 그대로 돌아야 해서다. IntPhys1 4중항은 config 에서 matched 를 명시한다.
    pairing = str(cfg["scoring"].get("pairing", "cross"))
    if pairing not in ("matched", "cross"):
        raise ValueError(f"scoring.pairing={pairing!r}; matched | cross 만 된다")
    per_block, dist_cnt = [], defaultdict(int)
    for bid, recs in ds.blocks().items():
        pairs = _block_pairs(recs, pairing)
        if not pairs:
            continue
        # 공식은 strict '<' 라 동점을 오답으로 센다 (eval.py:277). 우리는 0.5 를 주는 대신
        # 동점 개수를 같이 남긴다. 동점이 0 이면 두 규칙의 결과는 완전히 같다.
        hits = [1.0 if surprise[p.video_id] < surprise[n.video_id]
                else (0.5 if surprise[p.video_id] == surprise[n.video_id] else 0.0)
                for p, n in pairs]
        n_ties = sum(1 for h in hits if h == 0.5)
        acc = float(np.mean(hits))
        per_block.append({"block_id": bid, "block_type": recs[0].block_type,
                          "n_pairs": len(hits), "n_ties": n_ties, "acc": acc})
        dist_cnt[round(acc, 2)] += 1

    def agg(rows):
        return {"n_block": len(rows), "n_pair": sum(r["n_pairs"] for r in rows),
                "n_ties": sum(r["n_ties"] for r in rows),
                "block_pairwise": float(np.mean([r["acc"] for r in rows])) if rows else float("nan"),
                "perfect_ratio": float(np.mean([r["acc"] == 1.0 for r in rows])) if rows else float("nan")}

    res = {"overall": agg(per_block), "chance": 0.5, "pairing": pairing,
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

    # ---- surprise (위반 변이가 있는 데이터셋에서만) ----------------------------
    # surprise.mode 로 분기한다. 기본은 single (기존 동작 그대로).
    #   single   : 40프레임 클립 하나에서 context/target 한 번만. 구간을 미리 잘라둔 셋용.
    #   intphys1 : Garrido et al. 의 IntPhys 프로토콜. 영상 전체를 sliding window 로 훑고
    #              frame skip x window size grid 를 돌아 Avg/Max 로 집계한다.
    if cfg.get("surprise") and cfg.get("scoring"):
        mode = str(cfg["surprise"].get("mode", "single"))
        if mode == "intphys1":
            sup_raw, aggs = run_surprise_intphys1(ds, cfg, device)
        else:
            sup_raw = run_surprise(ds, cfg, device)
            aggs = list(next(iter(sup_raw.values())).keys()) if sup_raw else ["all"]

        if rank == 0:
            report["surprise_mode"] = mode
            report["surprise"] = {}
            dump = {"per_video_surprise": sup_raw, "per_block": {}}
            for a in aggs:
                sup = {k: v[a] for k, v in sup_raw.items()}
                res, per_block = score_blocks(ds, sup, cfg)
                # 단일 영상 분류(AUROC / absolute accuracy)는 기본 off.
                # 논문 A.7 이 "significantly harder" 라고 따로 떼어 둔 보조 지표고,
                # IntPhys 대표 수치는 쌍 비교(block_pairwise)다. 필요하면
                # scoring.single_video: true 로 켤 것.
                if cfg["scoring"].get("single_video", False):
                    res["single_video"] = single_video_metrics(ds, sup)
                report["surprise"][a] = res
                dump["per_block"][a] = per_block
                o = res["overall"]
                sv = res.get("single_video") or {}
                logger.info(f"[surprise/{a:16s}] block_pairwise={o['block_pairwise']:.4f} "
                            f"(block {o['n_block']}, pair {o['n_pair']}, tie {o['n_ties']}) "
                            f"| 전부정답={o['perfect_ratio']:.4f}"
                            + (f" | AUROC={sv['auroc']:.4f} abs_acc={sv['abs_acc']:.4f}"
                               if sv else ""))
            if mode == "intphys1":
                # 논문: "report the maximal accuracy across them"
                for a in ("avg", "max"):
                    cand = {k: v for k, v in report["surprise"].items() if k.endswith("/" + a)}
                    if not cand:
                        continue
                    best = max(cand, key=lambda k: cand[k]["overall"]["block_pairwise"])
                    report[f"best_{a}"] = {"key": best, **cand[best]["overall"]}
                    logger.info(f"[surprise] BEST({a}) = {best} "
                                f"{cand[best]['overall']['block_pairwise']:.4f}")
                    for k, v in cand[best].get("by_block_type", {}).items():
                        logger.info(f"              {k:16s} {v['block_pairwise']:.4f} "
                                    f"(block {v['n_block']})")
            elif aggs == ["all"]:     # 기존 소비자를 위해 예전 모양 유지
                report["surprise"] = report["surprise"]["all"]
                dump["per_video_surprise"] = {k: v["all"] for k, v in sup_raw.items()}
                dump["per_block"] = dump["per_block"]["all"]
            with open(os.path.join(out_dir, "per_block.json"), "w") as f:
                json.dump(dump, f, indent=2)
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
        recs, pmeta = run_probing(ds, bases, cfg, device)
        if rank == 0:
            report["probing"] = recs
            with open(os.path.join(out_dir, "predictions.json"), "w") as f:
                json.dump(pmeta, f)
            logger.info(f"[probe] 예측 덤프 -> {out_dir}/predictions.json "
                        f"(val {len(pmeta['val_video_ids'])}개 x head {len(pmeta['heads'])}개)")
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
