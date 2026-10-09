#!/usr/bin/env python3
"""IntPhysGen v11 — **visible · 늦은 가림 (late)** 에서 p·z·h 의 위치·존재를 온라인으로 읽는다 (RollOutV3 자).

질문: p 의 미래는 **마지막으로 본 상태를 연장**한 것인가. 늦은 가림은 문맥 끝 k 샘플이 가려져 있어
"마지막 관측 상태" (물체 안 보임) 와 "세계 상태" (물체 있음, 미래 k 샘플 뒤 다시 보임) 가 갈린다.

대상: v11 `index_probe.csv` 의 **가능 변이만** (CLAUDE.md §1-4), 조건 6 개 = visible 3 (k=0) + late 가림 3 (k=1..4).
  물체 있음 (`probe_type=obj`) 6,720 clip — vanish · shape · color 의 pos_a/pos_b
  빈 장면   (`probe_type=empty`) 1,344 clip — vanish 의 pos_b. **v11 장면에서 자의 '없다' 를 검사하는 음성**
창: 표준 v11 프로토콜 — raw 100 프레임 stride 3 → 32 샘플, 문맥 16 샘플 (8 튜블릿), 미래 16 샘플 (8 튜블릿).

저장 (특징이 아니라 **읽은 값만**, `x, y, presence score, top1 질량`; 문턱은 rollout3_paths.present_thr):
  정체 자 (`R3_DECODER=identity R3_OUT=identity`) 면 `{p,z,h}_prob` (…, 57) 도, 출력은 v11_identity/ · v11_mid_identity/
  p  (n, 8, 4)   미래 8 튜블릿            — 문맥 16 샘플만 본다
  z  (n, 16, 4)  32 샘플 전부 (문맥 포함) — context encoder 가 창 전체를 본다 = 자 검사용
  h  (n, 16, 4)  32 샘플 전부             — target encoder = 자 검사용
  진실: metadata `object_px_{x,y}_by_sample` (288 px, 자와 같은 정규화 px/144 − 1),
        가려진 샘플 = metadata `hidden_start..hidden_end` (raw) 에 드는 샘플 (late k 는 16−k..16+k−1)

⚠️ **이 자는 가림 음성을 배운 적이 없다** (RollOutV3 README §4-2). 가려진 샘플에서 `h`·`z` 가 '있다' 고 하는 비율이
   곧 자의 가림 편향이다 — p 를 읽기 전에 그것부터 본다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/v11_readout_online.py --limit 16 --gpus 1     # 배관 점검
  $P z_research/scripts/analysis/v11_readout_online.py --gpus 8                # 본 실행 → RollOutV3/exp_results/v11/readings.npz
  $P z_research/scripts/analysis/v11_readout_online.py --gpus 8 --timing mid   # 문맥 중간 가림 (visible 은 다시 안 뽑는다) → exp_results/v11_mid/
     mid: 샘플 7 부터 k 개 가려졌다가 문맥 안에서 다시 보인다 (raw 21 ~ 21+3(k−1))
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import PRES as READOUT_DIR, EXP, TAG, fingerprint, load_head, is_identity   # noqa: E402
from rollout3_window_readout import MODEL, readout_cls, prefetch            # noqa: E402  (모델·자 정의를 복제하지 않는다)

FR = Path(os.environ.get("V11_FRAMES", "/local_datasets/world/world_analysis/IntPhysGen_v11"))   # 2026-09-30: 다른 노드 스테이징용 덮어쓰기
INDEX = ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"
META = FR / "metadata.csv"
SFX = f"_{TAG}" if TAG else ""                                             # R3_OUT=identity → v11_identity/ · v11_mid_identity/
OUT = EXP / f"v11{SFX}"
TMP = Path(os.environ.get("V11_TMP", "/dev/shm/v11_readout"))
VIS = ["static_visible", "moving_visible_flat", "moving_visible"]
OCC = ["static_occlusion", "moving_occlusion_flat", "moving_occlusion"]
SUF = {"late": "", "mid": "_mid", "early": "_early"}                       # v11 조건 이름 접미사
CONDS = VIS + OCC                                                          # main() 에서 --timing 으로 다시 정한다
NS, CS, S, D, RES, W = 32, 16, 256, 1280, 144.0, 288
TC, TP = CS // 2, (NS - CS) // 2
DATA = dict(root=str(TMP), index_csv="index.csv", frames_root=str(FR), frames_pattern="{file_name}/{frame:06d}.png",
            frames_start=0, frames_stride=3, n_frames=NS, resolution=256)
KEEP = ["video_id", "block_id", "variant", "condition", "motion", "violation_type", "probe_type", "sym_k",
        "occ_timing", "shape_pre", "color_pre", "env"]


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", s.strip()) if v], np.float32)


def select(timing="late"):
    """가능 변이. late = visible 3 + late 가림 3 조건, 그 밖 = 그 타이밍의 가림 3 조건만. 순서는 index_probe 순서 그대로."""
    csv.field_size_limit(10 ** 9)
    conds = VIS + OCC if timing == "late" else [c + SUF[timing] for c in OCC]
    tim = ("", "late") if timing == "late" else (timing,)
    rows = [r for r in csv.DictReader(INDEX.open())
            if r["plausible"] == "1" and r["condition"] in conds and r["occ_timing"] in tim
            and r["probe_type"] in ("obj", "empty")]
    meta = {m["name"]: m for m in csv.DictReader(META.open())}
    n = len(rows)
    L = np.zeros((n, NS, 2), np.float32); hid = np.zeros((n, NS), bool)
    obj = np.array([r["probe_type"] == "obj" for r in rows]); tdir = np.zeros(n, np.int8)
    for i, r in enumerate(rows):
        m = meta[r["video_id"]]
        L[i, :, 0], L[i, :, 1] = arr(m["object_px_x_by_sample"]), arr(m["object_px_y_by_sample"])
        tdir[i] = {"l2r": 1, "r2l": -1}.get(m.get("travel_direction", ""), 0)
        try:
            hs, he = float(m["hidden_start"]), float(m["hidden_end"])
            if he >= hs:
                hid[i] = [hs <= 3 * t <= he for t in range(NS)]
        except (KeyError, ValueError):
            pass
    inf = (L[..., 0] >= 0) & (L[..., 0] < W) & (L[..., 1] >= 0) & (L[..., 1] < W)
    # 검사: late k 의 가려진 샘플은 16−k..16+k−1 이어야 한다 (plot_v11_vanish_readout.py 에서 확인한 규약)
    for i, r in enumerate(rows):
        if r["occ_timing"] == "late" and obj[i]:
            k = int(r["sym_k"]); want = np.zeros(NS, bool); want[CS - k:CS + k] = True
            if not (hid[i] == want).all():
                sys.exit(f"가려진 샘플 규약이 다르다: {r['video_id']} k={k} {np.where(hid[i])[0]}")
    return rows, L / RES - 1, inf, hid, obj, tdir


def _worker(rank, world, args, n):
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    from analysis.intphys2.model import build_from_config
    from evals.world_model_analysis.data import WMADataset
    from evals.analysis_vlm.occlusion_identity.forward import extract_batch
    from analysis.predictors import is_ar
    from app.vjepa_frozen.ar import encode_blocks, block_indices
    import decord  # noqa: F401

    heads = {rep: load_head(rep, dev, READOUT_DIR) for rep in args.reps}   # 자 종류와 무관 (rollout3_paths)
    ident = is_identity(READOUT_DIR)
    ln = lambda x: F.layer_norm(x, (x.size(-1),))
    bundle = build_from_config({**MODEL, "window_size": NS}, dev)
    ar = is_ar(bundle.predictor)
    assert not (ar and 'z' in args.reps), 'kind=ar 에는 z 가 없다 — --reps p h'
    ds = WMADataset(dict(data=DATA, model={**MODEL, "window_size": NS}, features={"cache_dir": "/tmp"}, surprise={}))
    bs = max(1, min(args.max_bs, args.token_budget // ((NS // 2) * S)))
    mm = {r: np.lib.format.open_memmap(TMP / f"{r}.npy", mode="r+") for r in args.reps}
    mp_ = {r: np.lib.format.open_memmap(TMP / f"{r}_prob.npy", mode="r+") for r in args.reps} if ident else {}
    idx = list(range(rank, n, world)); pool = ThreadPoolExecutor(max_workers=args.workers)
    t0, done = time.time(), 0
    with torch.no_grad():
        for ids, clips in prefetch(ds, idx, bs, pool):
            B = clips.size(0); ids = np.asarray(ids)
            clips = clips.to(dev, dtype=bundle.dtype, non_blocking=True)
            feats = {}
            if ar:                                       # 2026-09-30 kind=ar: p = rollout(블록별 h[문맥 8], 8), h = 블록별 LN(h) 16 튜블릿 (rollout3_window_readout 과 같은 정의)
                hb = encode_blocks(bundle.target_encoder, clips, 2)
                idx_ = block_indices(B, NS // 2, S, dev)
                if "p" in args.reps:
                    with torch.autocast("cuda", dtype=torch.bfloat16):   # 2026-09-30: AR 은 autocast 없이는 SDPA dtype 오류
                        feats["p"] = (bundle.predictor.rollout(hb[:, :TC * S], idx_[:, :TC * S], TP).reshape(B * TP, S, D).half(), TP)
                if "h" in args.reps:
                    feats["h"] = (hb.reshape(B * (NS // 2), S, D).half(), NS // 2)
            elif "p" in args.reps:
                o = extract_batch(clips, bundle, [{"base": "predictor"}], context_length=CS, mask_index=0,
                                  out_dtype=torch.float16)
                feats["p"] = (o["predictor"].reshape(B * TP, S, D).to(dev), TP)
            for rep, mod in (("z", "context_encoder"), ("h", "target_encoder")):
                if rep not in args.reps or ar:           # 필요 없는 표현은 ViT-H 한 바퀴를 아낀다 (2026-09-26)
                    continue
                fo = getattr(bundle, mod)(clips); fo = fo[-1] if isinstance(fo, list) else fo
                feats[rep] = (ln(fo.float()).reshape(B * (NS // 2), S, D).half(), NS // 2)
            for rep, (tok, nt) in feats.items():
                xy, score, top1, prob = heads[rep](tok.float())
                mm[rep][ids, :, 0:2] = xy.reshape(B, nt, 2).cpu().numpy()
                mm[rep][ids, :, 2] = score.reshape(B, nt).cpu().numpy()
                mm[rep][ids, :, 3] = top1.reshape(B, nt).cpu().numpy()
                if prob is not None:
                    mp_[rep][ids] = prob.reshape(B, nt, -1).cpu().numpy().astype(np.float16)
            done += B
            if rank == 0 and done % (bs * 10) < bs:
                el = time.time() - t0; frac = done / len(idx)
                print(f"    bs={bs} {done*world:,}/{n:,} clip  {el:.0f}s  남은 {el*(1-frac)/max(frac,1e-9)/60:.1f}분  "
                      f"최대 VRAM {torch.cuda.max_memory_allocated(dev)/2**30:.1f} GiB", flush=True)
    for v in list(mm.values()) + list(mp_.values()):
        v.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count() or 1)
    ap.add_argument("--limit", type=int, default=0, help="clip 수 제한 (배관 점검)")
    ap.add_argument("--token-budget", type=int, default=65536, help="배치 하나의 토큰 수 상한 (4,096 토큰/clip)")
    ap.add_argument("--max-bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8, help="PNG 읽기 스레드 (GPU 당)")
    ap.add_argument("--timing", default="late", choices=list(SUF), help="가림 타이밍. late 만 visible 을 같이 뽑는다")
    ap.add_argument("--reps", nargs="+", default=["p", "z", "h"], choices=["p", "z", "h"],
                    help="뽑을 표현. 그림 (p) · 판 검사 (p, h) 만 필요하면 p h — z 한 바퀴 (약 1/3) 를 아낀다")
    a = ap.parse_args()
    global OUT, CONDS
    if a.timing != "late":
        OUT = EXP / f"v11_{a.timing}{SFX}"; CONDS = [c + SUF[a.timing] for c in OCC]

    rows, L, inf, hid, obj, tdir = select(a.timing)
    if a.limit:                                  # 배관 점검 — 조건마다 고르게
        pick = sorted({i for c in CONDS for i in [j for j, r in enumerate(rows) if r["condition"] == c][:max(1, a.limit // 6)]})
        rows = [rows[i] for i in pick]; L, inf, hid, obj, tdir = L[pick], inf[pick], hid[pick], obj[pick], tdir[pick]
    n = len(rows)
    print(f"[data] {n:,} clip (물체 {int(obj.sum()):,} · 빈 장면 {int((~obj).sum()):,}) · 조건 {len(CONDS)}"
          + (f"  ⚠️ --limit {a.limit}" if a.limit else ""), flush=True)
    TMP.mkdir(parents=True, exist_ok=True)
    with (TMP / "index.csv").open("w", newline="") as f:          # 로더는 index.csv 만 읽는다
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    ident = is_identity(READOUT_DIR)
    for rep, nt in [(r_, n_) for r_, n_ in (("p", TP), ("z", NS // 2), ("h", NS // 2)) if r_ in a.reps]:
        np.lib.format.open_memmap(TMP / f"{rep}.npy", mode="w+", dtype=np.float32, shape=(n, nt, 4))
        if ident:                                            # 정체 자: 57 확률 (56 조합 + 없음)
            np.lib.format.open_memmap(TMP / f"{rep}_prob.npy", mode="w+", dtype=np.float16, shape=(n, nt, 57))
    t0 = time.time()
    mp.spawn(_worker, args=(a.gpus, a, n), nprocs=a.gpus, join=True)
    print(f"[extract] 끝 {(time.time()-t0)/60:.1f}분", flush=True)

    out = OUT / ("_smoke" if a.limit else ""); out.mkdir(parents=True, exist_ok=True)
    keep = {k: np.array([r[k] for r in rows]) for k in KEEP}
    keep.update(truth=L, in_frame=inf, hidden=hid, obj=obj, travel_dir=tdir,
                **{r: np.asarray(np.load(TMP / f"{r}.npy")) for r in a.reps},
                **({f"{r}_prob": np.asarray(np.load(TMP / f"{r}_prob.npy")) for r in a.reps} if ident else {}),
                decoder_fp=np.array(fingerprint(READOUT_DIR)), decoder_dir=np.array(str(READOUT_DIR)))
    np.savez_compressed(out / "readings.npz", **keep)
    (out / "meta.json").write_text(json.dumps(dict(n_clip=n, conds=CONDS, n_samples=NS, context_samples=CS,
                                                   stride=3, decoder_fp=str(keep["decoder_fp"])), indent=1))
    print(f"→ {out}/readings.npz", flush=True)


if __name__ == "__main__":
    main()
