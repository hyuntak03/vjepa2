#!/usr/bin/env python3
"""transfer_gap Q3 — 한 번의 encoder 통과로 predictor 여럿 + 복사 기준선을 채점한다 (E11 · E12 · E13). 2026-10-30, 실험 B [X2].

엔진은 `z_research/scripts/analysis/te_v11_score.py` 그대로다 (Scorer · _worker · write_summary 를 import, 파일 수정 없음).
다른 점은 config 하나뿐이다: te 는 데이터셋이 v11_split_test 로 고정이고 창이 c16t32 · stride 3 으로 고정이라,
여기서는 **cvpr/harness/resolve.py (eval/testbed 프로토콜) 에 --set / --window 를 넘겨** config 를 만든 뒤 같은 runner 로 돌린다.
  · 창 c16t28 (T5) 이면 미래 튜블릿이 6 개다 → spawn 된 rank 에서 te.N_SLOT 을 6 으로 바꾼 뒤 te._worker 를 부른다.
  · 수치 규약 검사: float32 + autocast float16 · L1 · 목표 LN · mask 0 · dual encoder (online 문맥 / EMA 목표) · resolution == img_size.

정의 (te 와 같다, clip i · predictor k · 미래 슬롯 j)
  l1[i,k,j]  = mean | p_k[j] − LN(target_encoder(전체 창))[문맥 뒤 j] |   — 슬롯 평균 = 표준 surprise
  copy[i,j]  = mean | LN(z)[문맥 마지막 튜블릿] − h[문맥 뒤 j] |          — 복사 기준선 (예측 없음)
  kind=ar predictor 는 l1 이 블록별 LN(target) 공간이고 copy_blk 가 그 복사다 (쌍 정확도만 비교)

출력 (--out, 기본 /data2/local_datasets/world/world_analysis/cache/transfer_gap/te_<name>/)
  l1.npy (n,K,S) · pz · pstep · copy (n,S) · hcopy · hstep · done · meta.json (video_id 순서 · 열 · 병합 config) · summary.json

재현 (sbatch 안에서; GPU 는 보이는 앞 N 장)
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  R=z_training/runs
  $P z_research/scripts/analysis/transfer_gap_te.py --name T0 --dataset vobench_v1_split2_test \
     --set data.root=data_csv/transfer_gap_vob data.index_csv=index_T0.csv \
           data.frames_root=/data2/local_datasets/world/world_analysis/cache/transfer_gap/frames_T0 \
     --preds release=$R/release_vith/latest.pt,vobftg=$R/vobench_release_postft_g/latest.pt --gpus 4
  # T1: --set data.frames_start=15 data.frames_stride=2   · T5: --dataset vobench_v1_asymmetry --set window.n_frames=28 ...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import yaml  # noqa: E402

import te_v11_score as te  # noqa: E402

CACHE = Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap")


def resolve(dataset: str, model: str, sets: list, window: str | None, limit: int) -> tuple[dict, str]:
    with tempfile.TemporaryDirectory(prefix="tg_te_") as td:
        o, m = Path(td) / "c.yaml", Path(td) / "m.json"
        args = [sys.executable, str(REPO / "cvpr/harness/resolve.py"), "eval/testbed", dataset, model, "-o", str(o),
                "--meta", str(m), "--quiet"]
        if window:
            args += ["--window", window]
        for kv in sets:
            args += ["--set", kv]
        if limit:
            args += ["--limit", str(limit)]
        env = {k: v for k, v in os.environ.items() if k not in ("TAG", "OUTDIR")}
        r = subprocess.run(args, text=True, capture_output=True, env=env, cwd=str(REPO))
        if r.returncode:
            te.die(f"resolve.py 실패:\n{r.stdout}\n{r.stderr}")
        text = o.read_text()
    cfg = yaml.safe_load(text)
    mo, s, d = cfg["model"], cfg["surprise"], cfg["data"]
    want = [(mo.get("dtype"), "float32"), (str(mo.get("autocast")).lower(), "float16"), (s.get("distance"), "l1"),
            (float(s.get("loss_exp")), 1.0), (bool(s.get("target_layer_norm")), True), (int(s.get("mask_index")), 0),
            (bool(mo.get("dual_encoder")), True), (mo.get("context_encoder_key"), "encoder"),
            (mo.get("target_encoder_key"), "target_encoder"), (int(d.get("resolution")), int(mo.get("img_size"))),
            (cfg["scoring"].get("pairing"), "matched")]
    bad = [(g, w) for g, w in want if g != w]
    if bad:
        te.die(f"병합 config 가 surprise 규약과 다르다: (got, want) = {bad}")
    return cfg, text


def _worker_wrap(rank, nslot, world, a, cfg, preds, todo, ds_index, out):
    te.N_SLOT = nslot                         # spawn 된 rank 는 te 를 새로 import 하므로 여기서 바꾼다 (T5: 6)
    te._worker(rank, world, a, cfg, preds, todo, ds_index, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="vith")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--window")
    ap.add_argument("--preds", required=True, help="tag=/path.pt,… (경로 predictor)")
    ap.add_argument("--out")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    a = ap.parse_args()
    a.preset, a.allow_foreign_base, a.allow_shared_device = None, False, False
    sets = [(kv.split("=", 1)[0] + "=" + str((REPO / kv.split("=", 1)[1]).resolve()))
            if kv.startswith(("data.root=",)) and not kv.split("=", 1)[1].startswith("/") else kv for kv in a.set]
    cfg, text = resolve(a.dataset, a.model, sets, a.window, a.limit)
    N, C = int(cfg["data"]["n_frames"]), int(cfg["surprise"]["context_length"])
    nslot = (N - C) // int(cfg["model"]["tubelet_size"])
    te.N_SLOT = nslot
    a.name = a.name
    preds, _ = te.parse_preds(argparse.Namespace(preds=a.preds, name=a.name, preset=None, model=a.model))
    infos = [te.pred_info(t, p, cfg) for t, p in preds]
    for inf in infos:
        if inf["path"] and not te._same_base(inf.get("base_checkpoint"), cfg["model"]["checkpoint"]):
            te.die(f"{inf['tag']}: base_checkpoint {inf.get('base_checkpoint')} ≠ {cfg['model']['checkpoint']}")
    has_ar = any(inf["space"] == "ar_block" for inf in infos)
    ds = te.WMADataset(cfg)
    ds_index = te.select_rows(ds, a.limit)
    n, K = len(ds_index), len(preds)
    out = Path(a.out) if a.out else CACHE / (f"te_{a.name}" + (f"_limit{a.limit}" if a.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    meta = {"script": str(Path(__file__).relative_to(REPO)), "model": a.model, "preset": a.name, "protocol": "eval/testbed",
            "dataset": a.dataset, "sets": sets, "window": a.window, "n_slot": nslot, "limit": a.limit, "n": n,
            "ds_index": ds_index, "index_csv": str(Path(cfg["data"]["root"]) / cfg["data"]["index_csv"]),
            "pred_tags": [t for t, _ in preds], "preds": infos, "pred_space": {i["tag"]: i["space"] for i in infos},
            "D": te.encoder_dim(cfg), "resolved_config": cfg,
            "arrays": "te_v11_score.py 와 같다 (l1/pz/pstep (n,K,S), copy/hcopy/hstep (n,S))"}
    for c in te.META_COLS + ("objset", "violation", "occlusion", "k", "motion", "background", "obj_a", "obj_b", "samples"):
        meta[c] = [ds.records[i].raw.get(c) for i in ds_index]
    meta["video_id"] = [ds.records[i].video_id for i in ds_index]
    mp_ = out / "meta.json"
    if mp_.exists():
        old = json.load(open(mp_))
        diff = [k for k in ("dataset", "n", "video_id", "pred_tags", "sets", "n_slot") if old.get(k) != meta.get(k)]
        if diff:
            te.die(f"{out} 에 다른 실행 산출물 ({diff}) — --out 을 바꿀 것")
        print(f"[resume] {out}")
    else:
        for k in te.ARR_K:
            m = np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=np.float32, shape=(n, K, nslot)); m[:] = np.nan; m.flush(); del m
        for k in te.arr_names(has_ar)[len(te.ARR_K):]:
            m = np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=np.float32, shape=(n, nslot)); m[:] = np.nan; m.flush(); del m
        m = np.lib.format.open_memmap(out / "done.npy", mode="w+", dtype=np.uint8, shape=(n,)); m[:] = 0; m.flush(); del m
        json.dump(meta, open(mp_, "w"), ensure_ascii=False)
        (out / "_resolved.yaml").write_text(text)
    done = np.load(out / "done.npy")
    todo = [int(r) for r in np.flatnonzero(done == 0)]
    print(f"out={out}  n={n} 남은 {len(todo)}  preds={[t for t, _ in preds]}  slots={nslot}  gpus={a.gpus}", flush=True)
    if todo:
        world = min(a.gpus, len(todo), torch.cuda.device_count())
        mp.spawn(_worker_wrap, args=(nslot, world, a, cfg, preds, todo, ds_index, str(out)), nprocs=world, join=True)
    done = np.load(out / "done.npy")
    if not done.all():
        te.die(f"done 이 아닌 행 {int((done == 0).sum())}")
    if not (out / "summary.json").exists():
        te.write_summary(out, json.load(open(mp_)))
    print("ALL DONE", out, flush=True)


if __name__ == "__main__":
    main()
