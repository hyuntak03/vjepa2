#!/usr/bin/env python3
"""[X1] E0 — 이식 실패 조사 (z_analysis/transfer_gap) 용 토큰 추출 (2026-10-30).

정의 (릴리즈 ViT-H, probing 관례 bfloat16 · autocast none — vobench_tubelet_probe.py 의 cfg_probe · _worker 와 같은 함수 · 같은 LN):
  h_imp_fut[i] = LN(target_encoder(불가능 영상 i 의 32 장))[튜블릿 t8..t15]          (1440, 8, 256, 1280) fp16
  p_relftg[j]  = predictor_PFT(context_encoder(가능 영상 j 의 문맥 16 장), mask_index 0)  (1440, 8, 256, 1280) fp16
                 PFT = z_training/runs/vobench_release_postft_g/latest.pt, 순서는 tubeprobe meta.json 과 같다
  z_late_ctx[k]= LN(context_encoder(late 가능 영상 k 의 문맥 16 장))[튜블릿 t0..t7]    (432, 8, 256, 1280) fp16
  tracks.npz   = block 마다 팔 A · B 의 튜블릿 16 개 중심 (u, v px@512, 샘플 s → raw 3s 의 두 샘플 평균) · 가려짐 (둘 중 하나라도 hidden) · ext_u/ext_v (최대)
입력: data_csv/vobench_v1/index.csv · tubeprobe meta.json (가능 영상 순서) · VOBench 프레임 · label.json
출력: /data2/local_datasets/world/world_analysis/cache/transfer_gap/{h_imp_fut.npy, imp_meta.json, p_relftg.npy, z_late_ctx.npy, z_late_meta.json, tracks.npz, DONE.json}

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/transfer_gap_extract.py tracks                      # CPU, 1 분
  $P z_research/scripts/analysis/transfer_gap_extract.py extract --gpus 4 [--limit 8] # sbatch GPU 4
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = Path(os.environ.get("TG_CACHE", "/data2/local_datasets/world/world_analysis/cache/transfer_gap"))
TUBE = Path("/local_datasets/world/world_analysis/cache/vobench_v1_vith_tubeprobe")
INDEX = REPO / "data_csv/vobench_v1/index.csv"
FR = Path("/data2/local_datasets/world/Benchmark/VOBench_v1")
PFT = REPO / "z_training/runs/vobench_release_postft_g/latest.pt"
NT, NF, S, D = 16, 8, 256, 1280


def cfg_probe():
    import yaml
    cfg = yaml.safe_load(open(REPO / "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/_resolved.yaml"))
    cfg["model"]["dtype"] = "bfloat16"
    cfg["model"]["autocast"] = "none"
    cfg["features"] = {"decode_workers": 6}
    return cfg


def plan(limit=None):
    """레코드 순서 · 짝 · late 목록. WMADataset 순서 = index.csv 순서."""
    rows = list(csv.DictReader(INDEX.open()))
    M = json.load(open(TUBE / "meta.json"))["meta"]
    pos_vid = [m["video_id"] for m in M]
    rec_pos = [i for i, r in enumerate(rows) if r["plausible"] == "1"]
    assert [rows[i]["video_id"] for i in rec_pos] == pos_vid, "tubeprobe meta 순서가 index.csv 가능 순서와 다르다"
    tp_of = {(m["block_id"], m["variant"]): j for j, m in enumerate(M)}
    imp = []
    for i, r in enumerate(rows):
        if r["plausible"] != "0":
            continue
        pv = {"imp_ab": "pos_a", "imp_ba": "pos_b"}[r["variant"]]
        imp.append(dict(rec=i, video_id=r["video_id"], block_id=r["block_id"], variant=r["variant"], pos_tp=tp_of[(r["block_id"], pv)],
                        objset=r["objset"], violation=r["violation"], occlusion=r["occlusion"], k=r["k"], motion=r["motion"],
                        obj_a=r["obj_a"], obj_b=r["obj_b"], samples=r["samples"]))
    late = [j for j, m in enumerate(M) if m["occlusion"] == "late"]
    if limit:
        imp, late = imp[:limit], late[:limit]
        rec_pos = rec_pos[:limit]
    return rows, M, rec_pos, imp, late


# ------------------------------------------------------------------ tracks (CPU)
def tracks():
    rows = list(csv.DictReader(INDEX.open()))
    blocks = {}
    for r in rows:
        blocks.setdefault(r["block_id"], Path(r["file_name"]).parent)
    bids = sorted(blocks, key=int)
    uv = np.zeros((len(bids), 2, NT, 2), np.float32); hid = np.zeros((len(bids), 2, NT), bool)
    ext = np.zeros((len(bids), 2, NT, 2), np.float32); present = np.zeros((len(bids), 2), bool)
    for bi, b in enumerate(bids):
        lab = json.load(open(FR / blocks[b] / "label.json"))["labels"]
        for ai, arm in enumerate("AB"):
            L = lab.get(arm)
            if not L:
                continue
            present[bi, ai] = True
            s = [L[3 * k] for k in range(32)]
            uv[bi, ai] = np.array([[x["u"], x["v"]] for x in s], np.float32).reshape(NT, 2, 2).mean(1)
            hid[bi, ai] = np.array([x["hidden"] for x in s], bool).reshape(NT, 2).any(1)
            ext[bi, ai] = np.array([[x["ext_u"], x["ext_v"]] for x in s], np.float32).reshape(NT, 2, 2).max(1)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "tracks.npz", block_id=np.array(bids), uv=uv, hidden=hid, ext=ext, present=present)
    print(f"tracks: {len(bids)} blocks, present A {present[:,0].sum()} B {present[:,1].sum()} → {OUT/'tracks.npz'}")


# ------------------------------------------------------------------ extract (GPU)
def _worker(rank, world, cfg, jobs, limit):
    import torch, torch.nn as nn
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.attn_probe import _ClipDS, _ac_dtype, _autocast
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}"); torch.set_num_threads(4)
    mine = jobs[rank::world]
    HI = np.load(OUT / "h_imp_fut.npy", mmap_mode="r+")
    PF = np.load(OUT / "p_relftg.npy", mmap_mode="r+")
    ZL = np.load(OUT / "z_late_ctx.npy", mmap_mode="r+")
    model = build_from_config(dict(cfg["model"], predictor_checkpoint=str(PFT)), dev)
    bs = 4
    ci, ti = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=S, batch_size=bs, device=dev)
    ac = _ac_dtype("none")
    # jobs = [(rec, kind, slot_imp, slot_pos, slot_late)] — 묶음 단위로 같은 kind 끼리
    for kind in ("imp", "pos"):
        sub = [j for j in mine if j[1] == kind]
        if not sub:
            continue
        loader = torch.utils.data.DataLoader(_ClipDS(cfg, [j[0] for j in sub]), batch_size=bs, shuffle=False, num_workers=5, pin_memory=True)
        w, t0 = 0, time.time()
        with torch.inference_mode():
            for x in loader:
                b = x.size(0); jj = sub[w:w + b]
                x = x.to(dev, model.dtype, non_blocking=True)
                with _autocast(ac):
                    if kind == "imp":
                        h = model.target_encoder(x); h = h[-1] if isinstance(h, list) else h
                        h = nn.functional.layer_norm(h, (h.size(-1),)).float().reshape(b, NT, S, -1)[:, NF:]
                        hn = h.cpu().numpy().astype(np.float16)
                        for q, j in enumerate(jj):
                            HI[j[2]] = hn[q]
                    else:
                        z = model.context_encoder(x, masks=[ci[:b]]); z = z[-1] if isinstance(z, list) else z
                        p = model.predictor(z, ci[:b], ti[:b], mask_index=0)
                        pn = p.float().cpu().numpy().reshape(b, NF, S, -1).astype(np.float16)
                        zn = nn.functional.layer_norm(z.float(), (z.size(-1),)).cpu().numpy().reshape(b, NF, S, -1).astype(np.float16)
                        for q, j in enumerate(jj):
                            PF[j[3]] = pn[q]
                            if j[4] >= 0:
                                ZL[j[4]] = zn[q]
                w += b
                if rank == 0 and w % 40 < bs:
                    print(f"[r0] {kind} {w}/{len(sub)} ({time.time() - t0:.0f}s)", flush=True)
        print(f"[r{rank}] {kind} done {len(sub)} in {time.time() - t0:.0f}s", flush=True)
    HI.flush(); PF.flush(); ZL.flush()


def extract(gpus, limit):
    import torch
    cfg = cfg_probe()
    rows, M, rec_pos, imp, late = plan(limit)
    from evals.world_model_analysis.data import WMADataset
    assert [r.video_id for r in WMADataset(cfg).records] == [r["video_id"] for r in rows], "WMADataset 순서 ≠ index.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    nI, nP, nL = len(imp), len(rec_pos), len(late)
    np.lib.format.open_memmap(OUT / "h_imp_fut.npy", "w+", np.float16, (nI, NF, S, D)).flush()
    np.lib.format.open_memmap(OUT / "p_relftg.npy", "w+", np.float16, (nP, NF, S, D)).flush()
    np.lib.format.open_memmap(OUT / "z_late_ctx.npy", "w+", np.float16, (nL, NF, S, D)).flush()
    json.dump(dict(n=nI, note="imp 영상 순서 = index.csv 불가능 순서. pos_tp = 짝 가능 영상의 tubeprobe 번호 (imp_ab↔pos_a, imp_ba↔pos_b)", meta=imp),
              open(OUT / "imp_meta.json", "w"))
    late_slot = {j: q for q, j in enumerate(late)}
    json.dump(dict(n=nL, note="z_late_ctx[q] = LN(context_encoder(문맥 16))[t0..7] of tubeprobe 번호 tp[q]",
                   tp=late, video_id=[M[j]["video_id"] for j in late]), open(OUT / "z_late_meta.json", "w"))
    jobs = [(m["rec"], "imp", q, -1, -1) for q, m in enumerate(imp)]
    jobs += [(rec, "pos", -1, j, late_slot.get(j, -1)) for j, rec in enumerate(rec_pos)]
    print(f"extract imp {nI} · pos {nP} (late {nL}) on {gpus} GPU → {OUT}", flush=True)
    torch.multiprocessing.spawn(_worker, args=(gpus, cfg, jobs, limit), nprocs=gpus, join=True)
    json.dump(dict(done=True, time=time.strftime("%F %T"), limit=limit, pft=str(PFT)), open(OUT / ("DONE.json" if not limit else "DONE_smoke.json"), "w"))


def warm_list():
    """프레임 예열용 파일 목록 (stride 3, 32 장 × 2880 영상)."""
    for r in csv.DictReader(INDEX.open()):
        for f in range(0, 96, 3):
            print(FR / r["file_name"] / f"{f:06d}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tracks", "extract", "warmlist"])
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.phase == "tracks":
        tracks()
    elif a.phase == "warmlist":
        warm_list()
    else:
        extract(a.gpus, a.limit or None)
