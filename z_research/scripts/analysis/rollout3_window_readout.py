#!/usr/bin/env python3
"""RollOut v3 — **16 개 (문맥, 예측) 창**에서 p·z·h 의 위치·존재를 읽는다.

목적: RollOutV2 의 dynamics 해석 (거리 한계 2~3 칸, 속도를 이어 가지 않음) 이
**창 크기에 의존하는지**를 없애는 것. 판정은 규칙 기반으로 사용자가 한다 — 여기서는 재기만 한다.

창: 사건은 **f32** 고정, 문맥 `[32-C, 32)`, 예측 `[32, 32+P)`, `C·P ∈ {4,8,16,32}`.
16 조합이 64 프레임 안에 다 들어간다. `wall`·`ledge` 는 f32 까지 pos/imp 가 픽셀 단위로 같아
matched pairing 이 성립한다 (실측 0.000 px, `build_rollout3_index.py` [6]).

⚠️ **창마다 모델을 다시 짓는다.** `window_size` 가 RoPE grid_depth 를 정하므로 `C+P` 가 다르면
   다른 빌드다 (IntPhys1 Garrido 프로토콜과 같은 이유, CLAUDE.md §2-2). 서로 다른 총합은 10 개라
   그 단위로 묶어 빌드를 재사용한다.

⚠️ **자는 학습된 `attn` 를 그대로 건다** (RollOutV3 presence, 학습셋). 창이 바뀌면 RoPE 격자가
   바뀌니 자가 그대로 통한다는 보장이 없다 → **`h` 를 같은 창에서 같이 재는 것이 그 검증이다.**
   `h` 가 1 칸 아래로 유지되면 자가 전이된 것이고, 무너지는 창은 **판정 불가**로 뺀다.

⚠️ **배치 크기가 읽은 값을 ~1 px 흔든다.** bf16 은 배치에 따라 커널·누적 순서가 달라지고
   soft-argmax 가 그걸 증폭한다 (봉우리가 흐린 칸에서만: 차이>5px 칸의 top1 0.445 vs 일치 칸 0.538).
   실측 batch1 대비 중앙 0.81 px · p99 5.7 px. 1 칸이 18 px 이라 무시할 수준이지만
   **재현성은 ~1 px 이라고 적는다** (CLAUDE.md §7-2 와 같은 현상).

저장: 특징이 아니라 **읽은 값만** 남긴다 (튜블릿마다 x, y, presence score, top1 질량).
  정체 자 (`R3_DECODER=identity`) 면 `C{c}_P{p}_{rep}_prob` (…, 57) 도 — score 의 뜻과 문턱은 rollout3_paths.present_thr
  16 창 x 3,136 clip x ≤16 튜블릿 x 3 표현 x 4 값 ≈ 5 MB.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/rollout3_window_readout.py --limit 16          # 배관 점검
  $P z_research/scripts/analysis/rollout3_window_readout.py --reps p h          # 본 실행
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT))
SRC = Path(os.environ.get("R3_SRC", "/data2/local_datasets/world/world_analysis/RollOut_v3"))   # 2026-09-30: 다른 노드 스테이징용 덮어쓰기
INDEX = ROOT / "data_csv/rollout_v3/index.csv"
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
# 자 폴더·출력 폴더는 한 곳에서 (R3_DECODER / R3_OUT). mp.spawn 워커도 환경변수를 물려받아 같은 값을 쓴다
from rollout3_paths import PRES as READOUT_DIR, WIN as OUT, fingerprint, load_head, is_identity   # noqa: E402
TMP = Path(os.environ.get("R3_TMP", "/dev/shm/rollout3_windows"))

NF, SPLIT = 64, 32
CTX, PRD = (4, 8, 16, 32), (4, 8, 16, 32)
WINDOWS = [(c, p) for c in CTX for p in PRD]
S, D, RES, CELL = 256, 1280, 144.0, 18.0

MODEL = dict(                                            # configs/protocols/models.md `vith`
    checkpoint=str(ROOT / "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/"
                          "b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth"),
    arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, window_size=32,
    use_rope=True, uniform_power=False,
    dual_encoder=True, context_encoder_key="encoder", target_encoder_key="target_encoder",
    predictor=dict(embed_dim=384, depth=12, num_heads=12, num_mask_tokens=10),
    dtype="bfloat16",
)
if os.environ.get("PRED_CKPT"):                          # 2026-09-28: 다른 predictor 판 (encoder 는 릴리즈 그대로, arch.kind 는 체크포인트가 선언)
    MODEL["predictor_checkpoint"] = os.environ["PRED_CKPT"]
DATA = dict(root=str(ROOT / "data_csv/rollout_v3"), index_csv="index.csv",
            frames_root=str(SRC / "Images"), frames_pattern="{file_name}/{frame:06d}.png",
            frames_start=0, frames_stride=1, n_frames=32, resolution=256, block_column="block_id")


def readout_cls():
    """학습된 자의 정의를 presence 스크립트에서 그대로 가져온다 (정의를 복제하지 않는다)."""
    spec = importlib.util.spec_from_file_location(
        "rp", ROOT / "z_research/scripts/analysis/rollout2_presence_readout.py")
    m = importlib.util.module_from_spec(spec); sys.modules["rp"] = m; spec.loader.exec_module(m)
    return m.Readout


def labels():
    rows = list(csv.DictReader(INDEX.open()))
    a = lambda s: np.array([float(v) for v in str(s).split()], np.float32)
    n = len(rows)
    px = np.stack([a(r["px_x_by_sample"]) for r in rows])
    py = np.stack([a(r["px_y_by_sample"]) for r in rows])
    L = np.stack([px / RES - 1, py / RES - 1], -1)                 # (n, 64, 2) 자와 같은 정규화
    inf = np.stack([a(r["in_frame_by_sample"]) for r in rows]) > 0  # (n, 64)
    return rows, L.astype(np.float32), inf


def key(rep, head):
    """읽은 값의 이름. 자기 자를 쓰면 표현 이름 그대로, 남의 자를 걸면 `p@h` 처럼 적는다."""
    return rep if rep == head else f"{rep}@{head}"


def tmp_path(c, p, rep):
    return TMP / f"C{c}_P{p}_{rep}.npy"


def prefetch(ds, idx, bs, pool, depth=3):
    """clip 을 **스레드로 미리 읽어** GPU 계산 뒤에 숨긴다.

    ⚠️ `DataLoader(num_workers=N)` 은 여기서 손해다 — 창마다 DataLoader 를 새로 만들어야 해서
       프로세스를 16 창 x N 번 새로 띄운다 (2026-09-23 실측: 16 clip 점검이 7 분 → 14 분).
       PIL 디코드는 GIL 을 놓으므로 스레드로 충분하고 띄우는 비용이 0 이다.
    """
    chunks = [idx[k:k + bs] for k in range(0, len(idx), bs)]
    fut = {}

    def submit(ch):
        for i in ch:
            if i not in fut:
                fut[i] = pool.submit(ds.clip, i)

    for ch in chunks[:depth]:
        submit(ch)
    for j, ch in enumerate(chunks):
        if j + depth < len(chunks):
            submit(chunks[j + depth])
        yield ch, torch.stack([fut.pop(i).result() for i in ch])


def _worker(rank, world, args, n):
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    from analysis.intphys2.model import build_from_config
    from evals.world_model_analysis.data import WMADataset
    from evals.analysis_vlm.occlusion_identity.forward import extract_batch
    import decord  # noqa: F401  (WMADataset 이 쓴다)

    # 자는 종류 (presence / 정체) 와 무관하게 load_head 로 — (xy, score, top1, prob|None) 을 돌려준다 (rollout3_paths)
    need = set(args.reps) | (set(args.cross) if args.cross else set())
    heads = {rep: load_head(rep, dev, READOUT_DIR) for rep in need}
    ident = is_identity(READOUT_DIR)
    # (표현, 그 표현에 걸 자) 목록. `--cross h z` 면 p 토큰에 h 자와 z 자를 **추가로** 건다.
    #   ⚠️ `p` 는 `LN(target encoder)` = `h` 를 맞추도록 학습됐으므로 **h 자를 p 에 거는 것이 원리적**이다.
    #      `z` 자는 공간이 달라 대조군이다 — 실패해도 "정보가 없다" 의 증거가 아니다.
    apply_to = [(r, r) for r in args.reps]
    for r in args.reps:
        for hr in (args.cross or []):
            if hr != r:
                apply_to.append((r, hr))
    ln = lambda x: F.layer_norm(x, (x.size(-1),))
    idx = list(range(rank, n, world))
    pool = ThreadPoolExecutor(max_workers=args.workers)

    # ⚠️ **창 목록은 `args` 로 받는다.** `mp.spawn` 워커는 모듈을 다시 import 하므로 `main()` 에서
    #    모듈 전역 `WINDOWS` 를 줄여도 **워커에는 안 닿는다** (2026-09-24 에 여기서 죽었다).
    by_total = {}
    for c, p in args.win:
        by_total.setdefault(c + p, []).append((c, p))
    t0 = time.time()
    for wi, (total, wins) in enumerate(sorted(by_total.items())):
        bundle = build_from_config({**MODEL, "window_size": total}, dev)
        from analysis.predictors import is_ar
        ar = is_ar(bundle.predictor)
        if ar:
            from app.vjepa_frozen.ar import encode_blocks, block_indices
            assert "z" not in args.reps, "kind=ar 에는 z 가 없다 — --reps p h (h = 블록별 LN(target_encoder))"
        # ⚠️ batch 1 이면 clip 당 ~262 ms 가 **커널 실행 오버헤드**로 나간다 (2026-09-23 프로파일:
        #    predictor 경로가 1,024 토큰이든 8,192 토큰이든 144 ms 로 같았다). 토큰 예산으로 묶는다.
        ntok = (total // 2) * S
        bs = max(1, min(args.max_bs, args.token_budget // ntok))
        for c, p in wins:
            ds = WMADataset(dict(data={**DATA, "frames_start": SPLIT - c, "n_frames": total},
                                 model={**MODEL, "window_size": total},
                                 features={"cache_dir": "/tmp"}, surprise={}))
            tc, tp = c // 2, p // 2
            mm = {key(r, hr): np.lib.format.open_memmap(tmp_path(c, p, key(r, hr)), mode="r+")
                  for r, hr in apply_to}
            mp_ = {key(r, hr): np.lib.format.open_memmap(tmp_path(c, p, key(r, hr) + "_prob"), mode="r+")
                   for r, hr in apply_to} if ident else {}
            done_c = 0
            with torch.no_grad():
                for ids, clips in prefetch(ds, idx, bs, pool):
                    B = clips.size(0)
                    clips = clips.to(dev, dtype=bundle.dtype, non_blocking=True)
                    feats = {}
                    if ar:                           # p = rollout(블록별 h[문맥], tp), h = 블록별 h[미래] (ar_scoring 과 같은 정의)
                        hb = encode_blocks(bundle.target_encoder, clips, 2)
                        idx_ = block_indices(B, tc + tp, S, dev)
                        if "p" in args.reps:
                            with torch.autocast("cuda", dtype=torch.bfloat16):   # 2026-09-30: AR 은 autocast 없이는 SDPA dtype 오류
                                feats["p"] = bundle.predictor.rollout(hb[:, :tc * S], idx_[:, :tc * S], tp).reshape(B * tp, S, D).half()
                        if "h" in args.reps:
                            feats["h"] = hb[:, tc * S:].reshape(B * tp, S, D).half()
                    if "p" in args.reps and not ar:
                        o = extract_batch(clips, bundle, [{"base": "predictor"}],
                                          context_length=c, mask_index=0, out_dtype=torch.float16)
                        feats["p"] = o["predictor"].reshape(B * tp, S, D).to(dev)
                    for rep, mod in (("z", "context_encoder"), ("h", "target_encoder")):
                        if rep not in args.reps or ar:
                            continue
                        fo = getattr(bundle, mod)(clips)
                        fo = fo[-1] if isinstance(fo, list) else fo
                        feats[rep] = ln(fo.float()).reshape(B, -1, S, D)[:, tc:].reshape(B * tp, S, D).half()
                    ids = np.asarray(ids)
                    for r, hr in apply_to:
                        tok = feats[r]
                        xy, score, top1, prob = heads[hr](tok.float())
                        k_ = key(r, hr)
                        mm[k_][ids, :, 0:2] = xy.reshape(B, tp, 2).cpu().numpy()
                        mm[k_][ids, :, 2] = score.reshape(B, tp).cpu().numpy()
                        mm[k_][ids, :, 3] = top1.reshape(B, tp).cpu().numpy()
                        if prob is not None:
                            mp_[k_][ids] = prob.reshape(B, tp, -1).cpu().numpy().astype(np.float16)
                    done_c += B
                    if rank == 0 and done_c % (bs * 5) < bs:
                        frac = (wi + done_c / len(idx)) / len(by_total)
                        el = time.time() - t0
                        print(f"    [win C{c}_P{p}] bs={bs} {done_c*world:,}/{n:,} clip  {el:.0f}s  "
                              f"(전체 {100*frac:.0f}%, 남은 {el*(1-frac)/max(frac,1e-9)/60:.0f}분)", flush=True)
            for v in list(mm.values()) + list(mp_.values()):
                v.flush()
        del bundle
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=["p", "h"], choices=["p", "z", "h"],
                    help="h 는 자 전이 검증용 대조라 빼지 말 것")
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count() or 1)
    ap.add_argument("--limit", type=int, default=0, help="clip 수 제한 (배관 점검)")
    ap.add_argument("--token-budget", type=int, default=32768, help="배치 하나가 쓸 토큰 수 상한")
    ap.add_argument("--max-bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6, help="PNG 읽기 스레드 (GPU 당)")
    ap.add_argument("--cross", nargs="*", default=[], choices=["p", "z", "h"],
                    help="다른 표현에서 배운 자를 **추가로** 건다 (이식). 예: --reps p --cross h z")
    ap.add_argument("--windows", nargs="*", default=None, metavar="CxP",
                    help="창을 제한한다. 예: --windows 16x32 32x32 (기본은 16 개 전부)")
    a = ap.parse_args()

    global WINDOWS
    if a.windows:
        want = {tuple(int(v) for v in w.lower().split("x")) for w in a.windows}
        WINDOWS = [w for w in WINDOWS if w in want]
        assert WINDOWS, f"고른 창이 없다: {a.windows}"
    a.win = list(WINDOWS)                       # 워커에 실어 보낸다 (전역은 안 닿는다)
    KEYS = [key(r, r) for r in a.reps] + [key(r, hr) for r in a.reps for hr in a.cross if hr != r]
    rows, L, inf = labels()
    n = a.limit or len(rows)
    print(f"[data] {len(rows)} clip · 창 {len(WINDOWS)} 개 · 읽는 값 {KEYS}"
          + (f"  ⚠️ --limit {a.limit}" if a.limit else ""), flush=True)

    TMP.mkdir(parents=True, exist_ok=True)
    ident = is_identity(READOUT_DIR)
    for c, p in WINDOWS:
        for k_ in KEYS:
            np.lib.format.open_memmap(tmp_path(c, p, k_), mode="w+", dtype=np.float32,
                                      shape=(n, p // 2, 4))          # x, y, presence score, top1
            if ident:                                                # 정체 자: 57 확률 (56 조합 + 없음) 도 남긴다
                np.lib.format.open_memmap(tmp_path(c, p, k_ + "_prob"), mode="w+", dtype=np.float16,
                                          shape=(n, p // 2, 57))
    t0 = time.time()
    mp.spawn(_worker, args=(a.gpus, a, n), nprocs=a.gpus, join=True)
    print(f"[extract] 끝 {(time.time()-t0)/60:.0f}분", flush=True)

    out = OUT / ("_smoke" if a.limit else "")
    out.mkdir(parents=True, exist_ok=True)
    keep = dict(video_id=np.array([r["video_id"] for r in rows[:n]]),
                scenario=np.array([r["scenario"] for r in rows[:n]]),
                role=np.array([r["role"] for r in rows[:n]]),
                block_id=np.array([r["block_id"] for r in rows[:n]]),
                holdout=np.array([int(r["holdout"]) for r in rows[:n]]),
                primary=np.array([float(r["primary"]) for r in rows[:n]]),
                truth=L[:n], in_frame=inf[:n])
    for c, p in WINDOWS:
        keep.update({f"C{c}_P{p}_{k_}": np.asarray(np.load(tmp_path(c, p, k_))) for k_ in KEYS})
        if ident:
            keep.update({f"C{c}_P{p}_{k_}_prob": np.asarray(np.load(tmp_path(c, p, k_ + "_prob"))) for k_ in KEYS})
    # ⚠️ 표현을 나눠 돌릴 수 있게 **기존 readings.npz 에 병합**한다 (--reps z 만 따로 붙이는 경우).
    #    2026-09-23: p·h 를 먼저 뽑고 z 를 나중에 붙였다. clip 수가 다르면 병합하지 않는다.
    # ⚠️ **자 지문을 싣는다.** 읽는 쪽이 문턱·치우침과 같은 자로 읽혔는지 확인한다 (rollout3_paths.check)
    fp = fingerprint(READOUT_DIR)
    keep["decoder_fp"] = np.array(fp); keep["decoder_dir"] = np.array(str(READOUT_DIR))
    prev = out / "readings.npz"
    if prev.exists():
        old = dict(np.load(prev, allow_pickle=True))
        if str(old.get("decoder_fp")) != fp:          # 다른 자로 읽힌 값과는 섞지 않는다 — 새로 쓴다
            print(f"[merge] 기존 readings.npz 는 다른 자 ({old.get('decoder_fp')}) 로 읽혔다 → 병합하지 않고 새로 쓴다", flush=True)
        elif len(old.get("video_id", [])) == n:
            old.update(keep); keep = old
            print(f"[merge] 기존 readings.npz 와 병합 — 표현 "
                  f"{sorted({k.split('_')[-1] for k in keep if k.startswith('C')})}", flush=True)
        else:
            print(f"[merge] clip 수가 달라 병합하지 않는다 ({len(old.get('video_id', []))} vs {n})", flush=True)
    np.savez_compressed(out / "readings.npz", **keep)
    (out / "meta.json").write_text(json.dumps(
        dict(n_clip=n, windows=WINDOWS, reps=a.reps, split=SPLIT, n_frames=NF, decoder_fp=fp,
             cell_px=CELL, norm=RES, readout=str(READOUT_DIR)), indent=1))
    print(f"→ {out}/readings.npz", flush=True)


if __name__ == "__main__":
    main()
