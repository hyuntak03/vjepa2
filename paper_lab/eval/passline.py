#!/usr/bin/env python
"""Pass-line evaluation harness for ONE block-AR predictor checkpoint (paper_lab, 2026-10-21).

Gives, in one run, every pass-line cell whose scoring tool passes the true-trajectory oracle check, each with the
same-space baseline trio (copy · context-mean copy · true-trajectory oracle) and chance:

  carry  ★ PRIMARY (agreed with 진행자 2): visible moving objects (VOBench, occlusion = visible, motion != static, possible clips):
             carry gain = (d_pred · d_true) / |d_true|^2, median, k1 and k1-8.  d from the last-context-block label position.
             Predicted position = parameter-free readout (template = mean of the last context block's target tokens on the
             labelled object 3x3, normalised; cosine to the 256 cells of a predicted block; arg-max cell; cos-weighted
             centre of its 3x3, cos clipped at 0; pixels at 256 res).  True position = render label.
             Included: |d_true| > 16 px and object visible in that block.  Rows: model · copy · ctxmean8 · true-target upper bound.
  objpair    pair accuracy with OBJECT-REGION scoring (mask = union of both futures' object 3x3 from labels) and ALL-TOKEN
             scoring, VOBench cells: visible/moving, early/moving, late/moving (placement, main), late permanence (presence,
             all-token = auxiliary).  Ties 0.5.
  gravity    future choice (arg-min surprise over candidate futures is the true one), arc by apex and ledge, all-token and
             object-region (mask = union over all candidate futures' object 3x3).
  natcarry   (opt-in set) natural-video carry gain with Farneback flow pseudo labels (encoder-free), kept only where the readout on the
             true target agrees with the flow track (<= 24 px); the true-target 'upper bound' is therefore biased towards 1 by the filter.
  ip1late    (opt-in set) IntPhys 1 late occlusion (uniform-d, 180 pairs, all-token; no object labels -> no oracle, copy and context-mean only).
  natural    rollout L1 per step vs copy and context-mean on a natural-video set disjoint from training
             (default: official Kinetics-400 val, n = 200, seed 0, 32 frames at --fps).

Validity rule (F4, round-1 review): a cell is VALID only if the oracle beats both chance and copy with the oracle's
95 % bootstrap lower bound above max(chance, copy).  Invalid cells are printed as "판정 불가" and excluded from pass/fail.

Uses existing code by import only (nothing in z_research / analysis / app / evals is modified):
  design_headroom (track_vobench, track_gravity, plan_positions, tok, SETS, FR, GRAV_ARM), analysis.intphys2.model.build_from_config,
  app.vjepa_frozen.ar (encode_blocks, block_indices), evals.world_model_analysis.data.WMADataset,
  rollout2_identity_readout (base MODEL dict), analysis.intphys2.dataset.build_video_transform.

  python paper_lab/eval/passline.py extract --ckpt z_training/vast/ar_state/latest.pt --name vast70 --gpus 1
  python paper_lab/eval/passline.py analyze --name vast70
  python paper_lab/eval/passline.py table  --names vast70 ariel_ar40      # side by side
  python paper_lab/eval/passline.py compare --names pilotA0_e10 pilotA1_e10   # paired block-bootstrap difference on the primary cell (+ per violation)
  python paper_lab/eval/passline.py compare --names pilotA0_e5 pilotA1_e5 pilotA0_e10 pilotA1_e10   # difference-in-differences e5 -> e10
Out: paper_lab/eval/results/<name>/{vobench_r*.npz, gravity_r*.npz, natural_r*.npz, passline.json, passline.md}
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
os.chdir(REPO)
import design_headroom as H  # noqa: E402

OUTROOT = REPO / "paper_lab/eval/results"
S, NB, C, K, D, G = 256, 16, 8, 8, 1280, 16
RENDERS = ("model", "copy", "mean8", "oracle")
# PASSLINE_CTX_IN (2026-10-22): number of context blocks FED to the predictor (the last ones before the future, re-indexed
# from position 0, as in training).  Default = C = 8 (standard).  Only the "model" render changes; targets, copy, the
# 8-block context mean and the oracle stay exactly as in the standard run.  For short-context checkpoints (e.g. C in {1,2}).
CTX_IN = int(os.environ.get("PASSLINE_CTX_IN", C))
assert 1 <= CTX_IN <= C, CTX_IN
# PASSLINE_TSPACE (2026-10-22, 진행자 2 요청): "raw" (default) = every model scored against the ORIGINAL block tokens.
# "train" = the method's own training space: targets AND every baseline (copy, context mean, oracle, readouts) are built from
# the method's target view (pl_method.json infer.target, e.g. the MNF jitter-suppressing projection).  Numbers are then only
# comparable as differences to the baselines of the same space.  Fails if the checkpoint has no non-raw target view.
TSPACE = os.environ.get("PASSLINE_TSPACE", "raw")
assert TSPACE in ("raw", "train"), TSPACE
_PM = None
K400_LIST = Path("/data2/local_datasets/Kinetics400/Kinetics-400/kinetics400_val_list_videos.txt")
K400_DIR = Path("/data2/local_datasets/Kinetics400/Kinetics-400/videos_val")
LABEL_SCALE = 0.5  # VOBench labels are in 512-px frame coordinates; tokens are 256-px input
IP1LATE_CSV = Path("/data/hyuntak/project/2026/2027_cvpr/wt_intphys1-late-uni/data_csv/intphys1_late_uni")  # read-only (index lives only in that worktree)
IP1LATE_FRAMES = REPO / "data_gen/IntPhys1_late_uni"
IMNET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1)
IMNET_STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1)


def flow_track(x_norm):
    """Natural-video pseudo label (same recipe as auto_research/scripts/e_flowtrack.py, re-implemented, encoder-free):
    Farneback flow on 256x256 grey frames; query = the 16x16 cell with the largest flow at frames 14->15; start = flow-weighted
    centre of that cell at frame 15; track forward by the median flow of a 16x16 window.  Returns (query cell (y, x), last-context
    position (px, mean of frames 14-15), future block positions (K, 2) px, query flow magnitude)."""
    import cv2
    a = (x_norm.cpu().float().numpy() * IMNET_STD + IMNET_MEAN).clip(0, 1)            # (3, T, H, W)
    g = (255 * (0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2])).astype(np.uint8)          # (T, H, W)
    fl = np.stack([cv2.calcOpticalFlowFarneback(g[t], g[t + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0) for t in range(g.shape[0] - 1)])
    mag = np.linalg.norm(fl[14], axis=-1)                                               # flow 14 -> 15
    cellmag = mag.reshape(G, 16, G, 16).mean((1, 3))
    qy, qx = np.unravel_index(int(np.argmax(cellmag)), cellmag.shape)
    win = mag[qy * 16:(qy + 1) * 16, qx * 16:(qx + 1) * 16]
    yy, xx = np.mgrid[0:16, 0:16]
    w = win + 1e-6
    p15 = np.array([qx * 16 + (xx * w).sum() / w.sum(), qy * 16 + (yy * w).sum() / w.sum()])
    def med(t, p):
        x0, y0 = int(np.clip(p[0] - 8, 0, 240)), int(np.clip(p[1] - 8, 0, 240))
        return np.median(fl[t][y0:y0 + 16, x0:x0 + 16].reshape(-1, 2), 0)
    p14 = p15 - med(14, p15)                                                            # approx. back one frame
    pos = {15: p15}
    p = p15.copy()
    for t in range(15, g.shape[0] - 1):
        p = np.clip(p + med(t, p), 0, 255.999)
        pos[t + 1] = p.copy()
    fut = np.array([(pos[16 + 2 * k] + pos[17 + 2 * k]) / 2 for k in range(K)])
    return (int(qy), int(qx)), np.clip((p14 + p15) / 2, 0, 255.999), fut, float(cellmag[qy, qx])


# ---------------------------------------------------------------- helpers (pure numpy, also used by analyze)
def cell_mask(points, dil=1):
    """points: (K, 2) label xy (512 px) or None per step -> bool (K, G, G)"""
    M = np.zeros((K, G, G), bool)
    for k, p in enumerate(points):
        if p is None or not (0 <= p[0] < 512 and 0 <= p[1] < 512):
            continue
        y, x = H.tok(p)
        M[k, max(y - dil, 0):y + dil + 1, max(x - dil, 0):x + dil + 1] = True
    return M


def future_track(lab_arm):
    if not lab_arm or lab_arm[0] is None:
        return None, None
    P = np.array([[lab_arm[3 * s]["u"], lab_arm[3 * s]["v"]] for s in range(32)], np.float64)
    hid = np.array([lab_arm[3 * s]["hidden"] for s in range(32)], np.float64)
    return P, hid


def boot_ci(x, B=2000, seed=0):
    x = np.asarray([v for v in x if v == v], np.float64)
    if len(x) == 0:
        return [float("nan")] * 3 + [0]
    bs = np.random.default_rng(seed).choice(x, size=(B, len(x))).mean(1)
    return [100 * x.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), int(len(x))]


# ---------------------------------------------------------------- GPU extract
def _load_model(ckpt, dev):
    from importlib.util import module_from_spec, spec_from_file_location
    from analysis.intphys2.model import build_from_config
    os.environ.setdefault("PRESENCE_SET", "vob_readout_primitive")
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec)
    spec.loader.exec_module(ir)
    model = dict(ir.rp.MODEL, predictor_checkpoint=str(REPO / ckpt) if not os.path.isabs(ckpt) else ckpt)
    bundle = build_from_config(model, dev)
    if getattr(bundle.predictor, "mask_mode", "release") not in ("ar", "ar_state"):
        raise SystemExit("passline.py takes block-AR predictors only (kind ar / ar_state)")
    return model, bundle


def _oracle_render(hblk_b, P, hid):
    """true-trajectory oracle = design_headroom's move_or construction (last fully visible object 3x3 pasted at the true
    future position on the last context block with the object erased). Re-implemented here because it lives inside a worker."""
    import torch  # noqa: F401
    pos, last_vis, ctx_c = H.plan_positions(P, hid)
    canvas = hblk_b[C - 1].clone()
    if hid[14] < 0.5 and hid[15] < 0.5:
        ry, rx = H.tok(ctx_c[C - 1])
        far = [bb for bb in range(C) if np.abs(np.array(H.tok(ctx_c[bb])) - np.array((ry, rx))).max() >= 3]
        if far:
            src = hblk_b[far[0]]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = ry + dy, rx + dx
                    if 0 <= yy < G and 0 <= xx < G:
                        canvas[yy, xx] = src[yy, xx]
    if last_vis is None:
        return canvas.unsqueeze(0).expand(K, -1, -1, -1)
    oy, ox = H.tok(ctx_c[last_vis])
    src = hblk_b[last_vis]
    frames = canvas.unsqueeze(0).repeat(K, 1, 1, 1)
    for k in range(K):
        p = pos["move_or"][k]
        if not (0 <= p[0] < 512 and 0 <= p[1] < 512):
            continue
        cy, cx = H.tok(p)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                sy, sx, ty, tx = oy + dy, ox + dx, cy + dy, cx + dx
                if 0 <= sy < G and 0 <= sx < G and 0 <= ty < G and 0 <= tx < G:
                    frames[k, ty, tx] = src[sy, sx]
    return frames


def _readout(blocks, tmpl):
    """blocks (K, G, G, D) tokens, tmpl (D,) unit -> (K, 2) xy at 256 px and (K,) max cos"""
    import torch
    t = torch.nn.functional.normalize(blocks.float(), dim=-1) @ tmpl                       # (K, G, G)
    out, mx = [], []
    for k in range(blocks.shape[0]):
        c = t[k]
        idx = int(torch.argmax(c))
        y, x = divmod(idx, G)
        num, den = np.zeros(2), 0.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < G and 0 <= xx < G:
                    w = max(float(c[yy, xx]), 0.0)
                    num += w * np.array([16 * xx + 8, 16 * yy + 8])
                    den += w
        out.append(num / den if den > 0 else np.array([16 * x + 8, 16 * y + 8], float))
        mx.append(float(c[y, x]))
    return np.array(out), np.array(mx)


def _install_method(ckpt):
    """paper_lab method checkpoints carry an input transform (pl_method.json next to the checkpoint, 진행자 2).
    Returns the ORIGINAL block encoder (scoring targets are always original block tokens) and the method cfg (or None)."""
    from app.vjepa_frozen import ar as AR
    cfg_path = Path(ckpt if os.path.isabs(ckpt) else REPO / ckpt).parent / "pl_method.json"
    if not cfg_path.exists():
        print("[passline] 평가 때 건 성분: 없음 (pl_method.json 없음 — 원래 입력)", flush=True)
        if TSPACE == "train":
            raise SystemExit("PASSLINE_TSPACE=train needs a pl_method.json with a non-raw target view")
        return AR.encode_blocks, None
    sys.path.insert(0, str(REPO / "paper_lab"))
    from method import patch as PM
    raw = json.load(open(cfg_path))
    # new format {"method", "infer", "train_only"}; old flat dict = "method". ONLY inference components are installed:
    # install_eval drops train-only keys (input occluder occ_px / token occluder occ / carry-weighted loss cw) by itself.
    cfg = raw.get("infer", raw.get("method", {})) if isinstance(raw, dict) and ({"infer", "method"} & set(raw)) else raw
    PM.install_eval(cfg)
    applied = dict(PM.CFG)
    leaked = [k for k in getattr(PM, "TRAIN_ONLY", ()) if k in applied]
    if leaked:
        raise SystemExit(f"train-only components would be applied at eval: {leaked}")
    print(f"[passline] 평가 때 건 성분 (eval-time method components): {json.dumps(applied, ensure_ascii=False) if applied else '없음 (none)'}"
          f" · 학습 전용 (안 걸었다): {json.dumps(raw.get('train_only', {}), ensure_ascii=False) if isinstance(raw, dict) else '{}'}", flush=True)
    # only an input transform changes what the predictor sees; otherwise skip the second encoding
    global _PM
    _PM = PM
    if TSPACE == "train":
        if applied.get("target", "raw") == "raw":
            raise SystemExit("PASSLINE_TSPACE=train but this checkpoint has no non-raw target view")
        print(f"[passline] 채점 공간 = 학습 공간 (표적 · 기준선 셋 모두 target view '{applied['target']}')", flush=True)
        return PM._ORIG_ENC, applied
    return PM._ORIG_ENC, (applied if applied.get("input", "raw") != "raw" else None)


def _worker(rank, world, devices, ckpt, name, sets, bs, n_nat, fps, limit=0, fast=False):
    import torch
    import yaml
    from app.vjepa_frozen import ar as AR
    from app.vjepa_frozen.ar import block_indices
    from evals.world_model_analysis.data import WMADataset
    dev_id = devices[rank]
    torch.cuda.set_device(dev_id)
    torch.set_num_threads(4)
    dev = torch.device("cuda", dev_id)
    print(f"[passline] predictor 입력 문맥 블록 수 = {CTX_IN} (표준 {C})", flush=True)
    model, bundle = _load_model(ckpt, dev)
    pred = bundle.predictor
    orig_enc, mcfg = _install_method(ckpt)
    out = OUTROOT / name
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        json.dump({"ckpt": str(ckpt), "method_cfg": mcfg, "fps_natural": fps, "n_natural": n_nat, "sets": sets, "ctx_in_blocks": CTX_IN, "target_space": TSPACE},
                  open(out / "meta.json", "w"), indent=1)

    def run_batch(x):
        """model input = AR.encode_blocks (module attribute, so a method patch applies); targets and all baselines use
        the ORIGINAL block tokens, so every model is scored in the same target space."""
        if TSPACE == "train":
            h_in = AR.encode_blocks(bundle.target_encoder, x, 2).float()      # patched: also fills HOLD["h_tgt"] (target view)
            h_tgt = _PM.HOLD["h_tgt"].float()
        else:
            h_tgt = orig_enc(bundle.target_encoder, x, 2).float()
            h_in = h_tgt if mcfg is None else AR.encode_blocks(bundle.target_encoder, x, 2).float()
        B = x.size(0)
        ar = pred.rollout(h_in[:, (C - CTX_IN) * S:C * S].to(torch.bfloat16), block_indices(B, CTX_IN, S, dev), K).float()
        return h_tgt.view(B, NB, G, G, D), ar.view(B, K, G, G, D)

    for setname in sets:
        if setname not in ("vobench", "gravity", "vobd"):
            continue
        if setname == "vobd" and "vobench" in sets:
            continue                                   # D maps are already saved with the vobench pass
        sp = H.SETS["vobench" if setname == "vobd" else setname]
        cfg = yaml.safe_load(open(sp["cfg"]))
        ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=model, features={"cache_dir": "/tmp"}, surprise={}))
        rows = {r["video_id"]: r for r in csv.DictReader(open(sp["index"]))}
        ids = [i for i, rec in enumerate(ds.records) if rec.video_id in rows]
        if (fast and setname == "vobench") or setname == "vobd":   # primary + carry cells only: late occlusion (all motions) and visible moving
            ids = [i for i in ids if rows[ds.records[i].video_id]["occlusion"] == "late"
                   or (rows[ds.records[i].video_id]["occlusion"] == "visible" and rows[ds.records[i].video_id]["motion"] != "flat_static")]
        if limit:
            ids = ids[:limit]
        mine = ids[rank * len(ids) // world:(rank + 1) * len(ids) // world]
        tr = H.track_gravity if setname == "gravity" else H.track_vobench
        rec = defaultdict(list)
        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b0 in range(0, len(mine), bs):
                ch = mine[b0:b0 + bs]
                x = torch.stack([ds.clip(i) for i in ch]).to(dev, dtype=bundle.dtype)
                hblk, ar = run_batch(x)
                fut = hblk[:, C:]
                for b, i in enumerate(ch):
                    r = rows[ds.records[i].video_id]
                    t = tr(r)
                    rend = {"model": ar[b], "copy": hblk[b, C - 1].unsqueeze(0).expand(K, -1, -1, -1),
                            "mean8": hblk[b, :C].mean(0).unsqueeze(0).expand(K, -1, -1, -1)}
                    rend["oracle"] = rend["copy"] if t is None else _oracle_render(hblk[b], *t)
                    if setname != "vobd":
                        for m in RENDERS:
                            em = (rend[m].float() - fut[b]).abs().mean(-1)
                            rec[f"S_{m}"].append(em.mean((-1, -2)).cpu().numpy())
                            rec[f"E_{m}"].append(em.half().cpu().numpy())
                    if setname in ("vobench", "vobd"):
                        # false-carry maps (B2): per-cell deviation from the context-mean copy, model / copy / oracle / truth
                        for m, rr in (("model", rend["model"]), ("copy", rend["copy"]), ("oracle", rend["oracle"]), ("true", fut[b])):
                            rec[f"D_{m}"].append((rr.float() - rend["mean8"].float()).abs().mean(-1).half().cpu().numpy())
                    # carry readout (positions in 256 px) for model / copy / mean8 / true target
                    pr = np.full((4, K, 2), np.nan, np.float32)
                    pc = np.full((4, K), np.nan, np.float32)
                    if t is not None:
                        P, hid = t
                        if hid[14] < 0.5 and hid[15] < 0.5:
                            c_last = P[14:16].mean(0)
                            ty_, tx_ = H.tok(c_last)
                            patch = hblk[b, C - 1, max(ty_ - 1, 0):ty_ + 2, max(tx_ - 1, 0):tx_ + 2].reshape(-1, D).float().mean(0)
                            tmpl = torch.nn.functional.normalize(patch, dim=0)
                            for j, blocks in enumerate((rend["model"], rend["copy"], rend["mean8"], fut[b])):
                                xy, mx = _readout(blocks, tmpl)
                                pr[j], pc[j] = xy, mx
                    if setname != "vobd":
                        rec["readout"].append(pr)
                        rec["readout_cos"].append(pc)
                    rec["video_id"].append(ds.records[i].video_id)
                if rank == 0 and (b0 // bs) % 10 == 0:
                    print(f"[r0] {setname} {b0 + len(ch)}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
        np.savez(out / f"{setname}_r{rank}.npz", video_id=np.array(rec.pop("video_id")), **{k: np.stack(v) for k, v in rec.items()})
        print(f"[r{rank}] {setname} done ({time.time() - t0:.0f}s)", flush=True)

    if "ip1late" in sets:
        ip_cfg = dict(root=str(IP1LATE_CSV), index_csv="index.csv", frames_root=str(IP1LATE_FRAMES), frames_pattern="{video_id}/{frame:06d}.png",
                      frames_start=0, frames_stride=1, block_column="block_id", pair_column="pair_id", variant_column="variant",
                      plausible_column="plausible", type_column="block_type", resolution=256, n_frames=32)
        ds = WMADataset(dict(data=ip_cfg, model=model, features={"cache_dir": "/tmp"}, surprise={}))
        ids = list(range(len(ds.records)))
        if limit:
            ids = ids[:limit]
        mine = ids[rank * len(ids) // world:(rank + 1) * len(ids) // world]
        rec = defaultdict(list)
        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b0 in range(0, len(mine), bs):
                ch = mine[b0:b0 + bs]
                x = torch.stack([ds.clip(i) for i in ch]).to(dev, dtype=bundle.dtype)
                hblk, ar = run_batch(x)
                fut = hblk[:, C:]
                for b, i in enumerate(ch):
                    for m, rr in (("model", ar[b]), ("copy", hblk[b, C - 1].unsqueeze(0).expand(K, -1, -1, -1)),
                                  ("mean8", hblk[b, :C].mean(0).unsqueeze(0).expand(K, -1, -1, -1))):
                        rec[f"S_{m}"].append((rr.float() - fut[b]).abs().mean(-1).mean((-1, -2)).cpu().numpy())
                    rec["video_id"].append(ds.records[i].video_id)
        np.savez(out / f"ip1late_r{rank}.npz", video_id=np.array(rec.pop("video_id")), **{k: np.stack(v) for k, v in rec.items()})
        print(f"[r{rank}] ip1late done ({time.time() - t0:.0f}s)", flush=True)

    if "natural" in sets or "natcarry" in sets:
        import decord
        from analysis.intphys2.dataset import build_video_transform
        lines = [l.split()[0] for l in open(K400_LIST) if l.strip()]
        rng = np.random.default_rng(0)
        rng.shuffle(lines)
        tf_ = build_video_transform(256, 1.0)
        rec = defaultdict(list)
        got, t0 = 0, time.time()
        cand = lines[rank::world]
        need = n_nat // world + (1 if rank < n_nat % world else 0)
        xs, names = [], []

        def flush():
            if not xs:
                return
            x = torch.stack(xs).to(dev, dtype=bundle.dtype)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hblk, ar = run_batch(x)
            fut = hblk[:, C:]
            for b in range(x.size(0)):
                rends = (("model", ar[b]), ("copy", hblk[b, C - 1].unsqueeze(0).expand(K, -1, -1, -1)),
                         ("mean8", hblk[b, :C].mean(0).unsqueeze(0).expand(K, -1, -1, -1)))
                for m, rr in rends:
                    rec[f"L_{m}"].append((rr.float() - fut[b]).abs().mean(-1).mean((-1, -2)).cpu().numpy())
                rec["video"].append(names[b])
                if "natcarry" in sets:
                    (qy, qx), c_last, ftrk, qmag = flow_track(x[b])
                    patch = hblk[b, C - 1, max(qy - 1, 0):qy + 2, max(qx - 1, 0):qx + 2].reshape(-1, D).float().mean(0)
                    tmpl = torch.nn.functional.normalize(patch, dim=0)
                    pr = np.full((4, K, 2), np.nan, np.float32)
                    pc = np.full((4, K), np.nan, np.float32)
                    for j, blocks in enumerate((ar[b], rends[1][1], rends[2][1], fut[b])):
                        pr[j], pc[j] = _readout(blocks, tmpl)
                    rec["nc_readout"].append(pr)
                    rec["nc_cos"].append(pc)
                    rec["nc_flow"].append(ftrk.astype(np.float32))
                    rec["nc_last"].append(np.asarray(c_last, np.float32))
                    rec["nc_qmag"].append(qmag)
            xs.clear()
            names.clear()

        for f in cand:
            if got >= need:
                break
            try:
                vr = decord.VideoReader(str(K400_DIR / f), num_threads=1)
                step = max(1, int(round(vr.get_avg_fps() / fps)))
                if len(vr) < 32 * step:
                    continue
                st = (len(vr) - 32 * step) // 2
                xs.append(tf_(vr.get_batch(list(range(st, st + 32 * step, step))).asnumpy()))
                names.append(f)
                got += 1
            except Exception:
                continue
            if len(xs) == bs:
                flush()
        flush()
        nc = {k: np.stack(rec.pop(k)) for k in [k for k in list(rec) if k.startswith("nc_")]}
        if nc:
            np.savez(out / f"natcarry_r{rank}.npz", video=np.array(rec["video"]), **nc)
        np.savez(out / f"natural_r{rank}.npz", video=np.array(rec.pop("video")), **{k: np.stack(v) for k, v in rec.items()})
        print(f"[r{rank}] natural done {got} clips ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- CPU analyze
def _load(name, setname):
    Z = defaultdict(list)
    for f in sorted((OUTROOT / name).glob(f"{setname}_r*.npz")):
        z = np.load(f, allow_pickle=True)
        for k in z.files:
            Z[k].append(z[k])
    return {k: np.concatenate(v) for k, v in Z.items()} if Z else None


def _valid(oracle_ci, copy_pt, chance):
    return bool(oracle_ci[1] > max(chance, copy_pt))


def analyze(name, split):
    res = {"name": name, "split": split, "cells": {}}
    test_v = {r["block_id"] for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_test.csv"))}
    test_g = {r["block_id"] for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_gravity_split/index_test.csv"))}

    # ---- VOBench pairs + carry
    Z = _load(name, "vobench")
    if Z is not None:
        vid = {v: i for i, v in enumerate(Z["video_id"])}
        rows = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
        blk = defaultdict(dict)
        for v, r in rows.items():
            blk[r["source_block"]][r["variant"]] = v
        acc = defaultdict(lambda: defaultdict(list))
        gains = defaultdict(lambda: defaultdict(list))
        for sb, arms in blk.items():
            if not all(a in arms for a in ("pos_a", "pos_b", "imp_ab", "imp_ba")) or not all(arms[a] in vid for a in arms):
                continue
            r = rows[arms["pos_a"]]
            if split == "test" and str(r["block_id"]) not in test_v:
                continue
            mot = "static" if r["motion"] == "flat_static" else "moving"
            lab = json.load(open(H.FR / Path(r["file_name"]).parent / "label.json"))["labels"]
            for pos, imp, fa, fb in (("pos_a", "imp_ab", "A", "B"), ("pos_b", "imp_ba", "B", "A")):
                pts = []
                for arm in (fa, fb):
                    Pf, _ = future_track(lab.get(arm))
                    pts.append(None if Pf is None else Pf[16:].reshape(8, 2, 2).mean(1))
                M = np.zeros((K, G, G), bool)
                for p in pts:
                    if p is not None:
                        M |= cell_mask(p)
                ip, ii = vid[arms[pos]], vid[arms[imp]]
                keys = [f"{r['occlusion']}/{mot}"]
                if r["occlusion"] == "late" and mot == "moving":
                    keys.append(f"late/moving/{r['violation']}")
                for m in RENDERS:
                    sp_, si_ = Z[f"S_{m}"][ip].mean(), Z[f"S_{m}"][ii].mean()
                    hit = 1.0 if si_ > sp_ else (0.5 if si_ == sp_ else 0.0)
                    Ep = Z[f"E_{m}"][ip].astype(np.float32)[M].mean() if M.any() else np.nan
                    Ei = Z[f"E_{m}"][ii].astype(np.float32)[M].mean() if M.any() else np.nan
                    hit_o = np.nan if not M.any() else (1.0 if Ei > Ep else (0.5 if Ei == Ep else 0.0))
                    for k in keys:
                        acc[f"all-token {k}"][m].append(hit)
                        acc[f"object-region {k}"][m].append(hit_o)
            # carry gain on visible moving possible clips
            if r["occlusion"] == "visible" and mot == "moving":
                for v in (arms["pos_a"], arms["pos_b"]):
                    rr = rows[v]
                    arm = "A" if rr["variant"] == "pos_a" else "B"
                    P, hid = future_track(lab.get(arm))
                    if P is None:
                        continue
                    i = vid[v]
                    c_last = P[14:16].mean(0) * LABEL_SCALE
                    truth = P[16:].reshape(8, 2, 2).mean(1) * LABEL_SCALE
                    vis = np.array([hid[16 + 2 * k] < 0.5 and hid[17 + 2 * k] < 0.5 for k in range(K)])
                    RO = Z["readout"][i]
                    for k in range(K):
                        dt = truth[k] - c_last
                        if np.linalg.norm(dt) <= 16 or not vis[k] or np.isnan(RO[0, k, 0]):
                            continue
                        for j, m in enumerate(("model", "copy", "mean8", "truth")):
                            g = float(np.dot(RO[j, k] - c_last, dt) / np.dot(dt, dt))
                            gains["all"][m].append(g)
                            if k == 0:
                                gains["k1"][m].append(g)
        for cell, d in acc.items():
            ci = {m: boot_ci(v) for m, v in d.items()}
            res["cells"][cell] = {"chance": 50.0, **ci, "valid": _valid(ci["oracle"], ci["copy"][0], 50.0)}
        for kk, d in gains.items():
            res["cells"][f"carry gain {kk}"] = {m: [float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)), len(v)]
                                                for m, v in d.items()}
            res["cells"][f"carry gain {kk}"]["valid"] = True  # the true-target upper bound is reported, not an oracle test

    # ---- B2 false carry: deviation from the context-mean copy in cells that should not change
    #      (possible clips only; static clips: all cells; moving clips: cells > 3 cells (Chebyshev) from the object's
    #      positions over all 32 frames).  Reported per step for model / copy / oracle / truth (truth = the real change floor).
    Zd = Z if (Z is not None and "D_model" in Z) else _load(name, "vobd")
    if Zd is not None and "D_model" in Zd:
        rows = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
        acc = defaultdict(lambda: defaultdict(list))
        for i, v in enumerate(Zd["video_id"]):
            r = rows.get(str(v))
            if r is None or r["variant"] not in ("pos_a", "pos_b"):
                continue
            if split == "test" and str(r["block_id"]) not in test_v:
                continue
            static = r["motion"] == "flat_static"
            if static:
                M = np.ones((K, G, G), bool)
            else:
                lab = json.load(open(H.FR / Path(r["file_name"]).parent / "label.json"))["labels"]
                P, _ = future_track(lab.get("A" if r["variant"] == "pos_a" else "B"))
                if P is None:
                    continue
                near = np.zeros((G, G), bool)
                for p in P:
                    if 0 <= p[0] < 512 and 0 <= p[1] < 512:
                        y, x = H.tok(p)
                        near[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4] = True
                M = np.broadcast_to(~near, (K, G, G))
            if not M.any():
                continue
            cell = "false carry static clips" if static else "false carry far cells (moving clips)"
            for m in ("model", "copy", "oracle", "true"):
                Dm = Zd[f"D_{m}"][i].astype(np.float32)
                acc[cell][m].append([Dm[k][M[k]].mean() for k in range(K)])
        for cell, d in acc.items():
            res["cells"][cell] = {m: np.mean(np.array(v), 0).tolist() for m, v in d.items()}
            res["cells"][cell]["n_clips"] = len(d["model"])
            res["cells"][cell]["valid"] = True

    # ---- gravity future choice
    Zg = _load(name, "gravity")
    if Zg is not None:
        vidg = {v: i for i, v in enumerate(Zg["video_id"])}
        rowsg = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["gravity"]["index"]))}
        blk = defaultdict(dict)
        for v, r in rowsg.items():
            blk[r["source_block"]][r["variant"]] = v
        pick = defaultdict(lambda: defaultdict(list))
        for sb, arms in blk.items():
            r = rowsg[arms["pos"]]
            if split == "test" and str(r["block_id"]) not in test_g:
                continue
            if not all(a in vidg for a in arms.values()):
                continue
            cells = ["ledge"] if r["law"] == "ledge" else ["arc", f"arc/{r['cell']}"]
            lab = json.load(open(H.FR / "gravity" / Path(r["file_name"]).parent / "label.json"))["labels"]
            M = np.zeros((K, G, G), bool)
            for a in arms:
                Pf, _ = future_track(lab.get(H.GRAV_ARM.get(a, "")))
                if Pf is not None:
                    M |= cell_mask(Pf[16:].reshape(8, 2, 2).mean(1))
            for m in RENDERS:
                sc = {a: Zg[f"S_{m}"][vidg[v]].mean() for a, v in arms.items()}
                ok = float(min(sc, key=sc.get) == "pos")
                so = {a: Zg[f"E_{m}"][vidg[v]].astype(np.float32)[M].mean() for a, v in arms.items()} if M.any() else None
                ok_o = np.nan if so is None else float(min(so, key=so.get) == "pos")
                for c in cells:
                    pick[f"all-token gravity {c}"][m].append(ok)
                    pick[f"object-region gravity {c}"][m].append(ok_o)
            for c in cells:
                pick[f"all-token gravity {c}"]["_n_arms"].append(len(arms))
                pick[f"object-region gravity {c}"]["_n_arms"].append(len(arms))
        for cell, d in pick.items():
            chance = 100.0 / np.mean(d.pop("_n_arms"))
            ci = {m: boot_ci(v) for m, v in d.items()}
            res["cells"][cell] = {"chance": chance, **ci, "valid": _valid(ci["oracle"], ci["copy"][0], chance)}

    # ---- natural video
    Zn = _load(name, "natural")
    if Zn is not None:
        Lm, Lc, La = Zn["L_model"], Zn["L_copy"], Zn["L_mean8"]
        res["cells"]["natural L1"] = {
            "n": int(len(Lm)), "model": Lm.mean(0).tolist(), "copy": Lc.mean(0).tolist(), "mean8": La.mean(0).tolist(),
            "gain_over_copy_pct": (100 * (1 - Lm.mean(0) / Lc.mean(0))).tolist(),
            "gain_over_mean8_pct": (100 * (1 - Lm.mean(0) / La.mean(0))).tolist(), "valid": True}

    # ---- IntPhys 1 late occlusion (no object labels -> no oracle; copy and context-mean are the predictor-free baselines)
    Zi = _load(name, "ip1late")
    if Zi is not None:
        vid = {v: i for i, v in enumerate(Zi["video_id"])}
        rows = list(csv.DictReader(open(IP1LATE_CSV / "index.csv")))
        test_ids = None
        if split == "test":   # fold-0 test (only meaningful for checkpoints trained on fold 0)
            test_ids = {r["video_id"] for r in csv.DictReader(open(str(IP1LATE_CSV) + "_cv/fold0_test.csv"))}
        pairs = defaultdict(dict)
        for r in rows:
            pairs[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r
        acc = defaultdict(lambda: defaultdict(list))
        for key, pr in pairs.items():
            if "pos" not in pr or "imp" not in pr or pr["pos"]["video_id"] not in vid or pr["imp"]["video_id"] not in vid:
                continue
            if test_ids is not None and pr["pos"]["video_id"] not in test_ids:
                continue
            for m in ("model", "copy", "mean8"):
                sp_, si_ = Zi[f"S_{m}"][vid[pr["pos"]["video_id"]]].mean(), Zi[f"S_{m}"][vid[pr["imp"]["video_id"]]].mean()
                hit = 1.0 if si_ > sp_ else (0.5 if si_ == sp_ else 0.0)
                for c in ("ip1late all", f"ip1late {pr['pos']['group']}"):
                    acc[c][m].append(hit)
        for cell, d in acc.items():
            ci = {m: boot_ci(v) for m, v in d.items()}
            res["cells"][f"all-token {cell}"] = {"chance": 50.0, **ci, "oracle": [float("nan")] * 3 + [0],
                                                 "valid": "no oracle (no object labels) — vs predictor-free baselines only"}

    # ---- natural-video carry with flow pseudo labels (agreement rule: keep steps where the readout on the TRUE target
    #      lands within 24 px of the flow track — the two pseudo labels agree, auto_research reading rule 12)
    Zc = _load(name, "natcarry")
    if Zc is not None:
        gains = defaultdict(lambda: defaultdict(list))
        nagree = ntot = 0
        for i in range(len(Zc["video"])):
            RO, F_, cl = Zc["nc_readout"][i], Zc["nc_flow"][i], Zc["nc_last"][i]
            for k in range(K):
                dt = F_[k] - cl
                if np.linalg.norm(dt) <= 16 or np.isnan(RO[3, k, 0]):
                    continue
                ntot += 1
                if np.linalg.norm(RO[3, k] - F_[k]) > 24:
                    continue
                nagree += 1
                for j, m in enumerate(("model", "copy", "mean8", "truth")):
                    g = float(np.dot(RO[j, k] - cl, dt) / np.dot(dt, dt))
                    gains["all"][m].append(g)
                    if k == 0:
                        gains["k1"][m].append(g)
        for kk, d in gains.items():
            res["cells"][f"natural carry gain {kk}"] = {m: [float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)), len(v)]
                                                        for m, v in d.items()}
            res["cells"][f"natural carry gain {kk}"]["valid"] = True
        res["cells"]["natural carry agreement"] = {"agree": nagree, "moved_steps": ntot, "valid": True}

    out = OUTROOT / name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"passline_{split}.json", "w"), indent=1)
    (out / f"passline_{split}.md").write_text(render_md([res]))
    print(render_md([res]))


ORDER = ["carry gain k1", "carry gain all",
         "object-region visible/moving", "object-region early/moving", "object-region late/moving",
         "object-region late/moving/permanence", "object-region late/moving/shape", "object-region late/moving/color",
         "object-region late/static",
         "all-token late/moving/permanence",
         "all-token gravity arc", "all-token gravity arc/apex30", "all-token gravity arc/apex48", "all-token gravity arc/apex66",
         "all-token gravity ledge", "object-region gravity arc", "object-region gravity ledge", "natural L1",
         "natural carry gain k1", "natural carry gain all", "all-token ip1late all", "all-token ip1late moving/occluded",
         "all-token ip1late moving/visible", "all-token ip1late static/occluded", "all-token ip1late static/visible"]


def render_md(results):
    names = [r["name"] for r in results]
    lines = [f"| cell | " + " | ".join(names) + " | copy | ctx-mean8 | oracle / upper | chance | valid |", "|" + "---|" * (len(names) + 6)]
    base = results[0]["cells"]
    for cell in ORDER:
        if cell not in base:
            continue
        b = base[cell]
        if cell.startswith("carry") or cell.startswith("natural carry gain"):
            vals = [f"{r['cells'][cell]['model'][0]:.2f} (n {r['cells'][cell]['model'][3]})" if cell in r["cells"] else "—" for r in results]
            lines.append(f"| {cell} | " + " | ".join(vals) + f" | {b['copy'][0]:.2f} | {b['mean8'][0]:.2f} | {b['truth'][0]:.2f} | 0 = stays · 1 = true | upper bound |")
        elif cell == "natural L1":
            vals = [f"{np.mean(r['cells'][cell]['model']):.3f} (copy −{np.mean(r['cells'][cell]['gain_over_copy_pct']):.1f} %)" if cell in r["cells"] else "—" for r in results]
            lines.append(f"| {cell} (mean over steps, n {b['n']}) | " + " | ".join(vals) + f" | {np.mean(b['copy']):.3f} | {np.mean(b['mean8']):.3f} | — | — | yes |")
        else:
            vals = [f"{r['cells'][cell]['model'][0]:.1f} [{r['cells'][cell]['model'][1]:.0f}, {r['cells'][cell]['model'][2]:.0f}]" if cell in r["cells"] else "—" for r in results]
            v = b["valid"] if isinstance(b["valid"], str) else ("yes" if b["valid"] else "**판정 불가**")
            lines.append(f"| {cell} (n {b['model'][3]}) | " + " | ".join(vals) +
                         f" | {b['copy'][0]:.1f} | {b['mean8'][0]:.1f} | {b['oracle'][0]:.1f} | {b['chance']:.1f} | {v} |")
    for cell in ("false carry static clips", "false carry far cells (moving clips)"):
        if cell in base:
            b = base[cell]
            vals = [f"{np.mean(r['cells'][cell]['model']):.3f}" if cell in r["cells"] else "—" for r in results]
            lines.append(f"| {cell}: mean \\|p − ctx-mean8\\| (n {b['n_clips']}; lower = less invented change) | " + " | ".join(vals) +
                         f" | copy {np.mean(b['copy']):.3f} | 0 | oracle {np.mean(b['oracle']):.3f} · truth {np.mean(b['true']):.3f} | — | yes |")
    if "natural carry agreement" in base:
        a = base["natural carry agreement"]
        lines.append(f"| natural carry: steps where flow and target readout agree | {a['agree']} / {a['moved_steps']} moved steps | | | | | |")
    return "\n".join(lines) + "\n"


def _pair_hits(name, occ="late", mot="moving", scoring="object-region", split="all", render="model", margin=False, far=None):
    """per-pair hits of the MODEL render, keyed by (source_block, direction) — for paired comparisons between checkpoints.
    margin=True (2026-10-22, R2 라운드 2 정의): returns the continuous object-region margin E(imp) − E(pos) instead of the hit
    (positive = the right side); render = model | copy | mean8 | oracle."""
    Z = _load(name, "vobench")
    vid = {v: i for i, v in enumerate(Z["video_id"])}
    rows = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
    test_v = {r["block_id"] for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_test.csv"))}
    blk = defaultdict(dict)
    for v, r in rows.items():
        blk[r["source_block"]][r["variant"]] = v
    out = {}
    for sb, arms in blk.items():
        if not all(a in arms for a in ("pos_a", "pos_b", "imp_ab", "imp_ba")) or not all(arms[a] in vid for a in arms):
            continue
        r = rows[arms["pos_a"]]
        m_ = "static" if r["motion"] == "flat_static" else "moving"
        if r["occlusion"] != occ or m_ != mot or (split == "test" and str(r["block_id"]) not in test_v):
            continue
        lab = json.load(open(H.FR / Path(r["file_name"]).parent / "label.json"))["labels"]
        for pos, imp, fa, fb in (("pos_a", "imp_ab", "A", "B"), ("pos_b", "imp_ba", "B", "A")):
            ip, ii = vid[arms[pos]], vid[arms[imp]]
            if scoring == "object-region":
                M = np.zeros((K, G, G), bool)
                for arm in (fa, fb):
                    Pf, _ = future_track(lab.get(arm))
                    if Pf is not None:
                        M |= cell_mask(Pf[16:].reshape(8, 2, 2).mean(1))
                if far is not None:
                    # '먼 칸' (PLAN §6, 2026-10-22): keep only future blocks whose possible-future object centre is > far px
                    # (512-px label scale; 48 = 1.5 cells) from the last visible context position (context arm = fa)
                    Pa, ha = future_track(lab.get(fa))
                    vis = [t for t in range(16) if Pa is not None and ha[t] < 0.5]
                    if Pa is None or not vis:
                        continue
                    ck = Pa[16:].reshape(8, 2, 2).mean(1)
                    keep = np.linalg.norm(ck - Pa[vis[-1]], axis=1) > far
                    M[~keep] = False
                if not M.any():
                    continue
                a_, b_ = Z["E_" + render][ip].astype(np.float32)[M].mean(), Z["E_" + render][ii].astype(np.float32)[M].mean()
            else:
                a_, b_ = Z["S_" + render][ip].mean(), Z["S_" + render][ii].mean()
            out[(sb, pos)] = float(b_ - a_) if margin else (1.0 if b_ > a_ else (0.5 if b_ == a_ else 0.0))
    return out


def compare(names, split, B=1000, occ="late", mot="moving", far=None):
    """Paired block-bootstrap (source blocks resampled, B = 1000) on the primary cell (object-region, late, moving, 360 pairs)
    and on each violation (120 pairs each).  2 names: difference b − a.  4 names (a1 b1 a2 b2): difference-in-differences
    (b2 − a2) − (b1 − a1), e.g. (A1_e10 − A0_e10) − (A1_e5 − A0_e5)."""
    rows = {r["source_block"]: r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
    H_ = [_pair_hits(n, occ=occ, mot=mot, split=split, far=far) for n in names]
    keys = sorted(set.intersection(*[set(h) for h in H_]))
    if len(names) == 2:
        val = {k: H_[1][k] - H_[0][k] for k in keys}
        label = f"{names[1]} − {names[0]}"
    elif len(names) == 4:
        val = {k: (H_[3][k] - H_[2][k]) - (H_[1][k] - H_[0][k]) for k in keys}
        label = f"({names[3]} − {names[2]}) − ({names[1]} − {names[0]})"
    else:
        raise SystemExit("compare takes 2 or 4 names")
    # continuous margin (R2 라운드 2): change of E(imp) − E(pos) in units of the mean |oracle margin| (oracle of the first name)
    M_ = [_pair_hits(n, occ=occ, mot=mot, split=split, margin=True, far=far) for n in names]
    MO = _pair_hits(names[0], occ=occ, mot=mot, split=split, render="oracle", margin=True, far=far)
    scale = float(np.mean([abs(MO[k]) for k in keys])) if keys else float("nan")
    if len(names) == 2:
        mval = {k: (M_[1][k] - M_[0][k]) / scale for k in keys}
    else:
        mval = {k: ((M_[3][k] - M_[2][k]) - (M_[1][k] - M_[0][k])) / scale for k in keys}
    rng = np.random.default_rng(0)
    out = {"label": label, "split": split, "cell": f"object-region {occ}/{mot}", "B": B, "rows": {},
           "margin_note": "margin_share_pct = change of the object-region margin E(imp)-E(pos) in % of the mean |oracle margin| (R2 round-2 definition)"}
    for sub in ("all", "permanence", "shape", "color"):
        ks = [k for k in keys if sub == "all" or rows[k[0]]["violation"] == sub]
        blocks = sorted({k[0] for k in ks})
        by = defaultdict(list)
        for k in ks:
            by[k[0]].append(val[k])
        byM = defaultdict(list)
        for k in ks:
            byM[k[0]].append(mval[k])
        bs, bsm = [], []
        for _ in range(B):
            pick = rng.integers(0, len(blocks), len(blocks))
            bs.append(np.mean([v for j in pick for v in by[blocks[j]]]))
            bsm.append(np.mean([v for j in pick for v in byM[blocks[j]]]))
        accs = {n: 100 * np.mean([H_[i][k] for k in ks]) for i, n in enumerate(names)}
        out["rows"][sub] = {"n_pairs": len(ks), "n_blocks": len(blocks), "acc": accs,
                            "diff": 100 * np.mean([val[k] for k in ks]),
                            "ci95": [100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)],
                            "n_discordant": int(sum(1 for k in ks if val[k] != 0)),
                            "margin_share_pct": {"diff": 100 * np.mean([mval[k] for k in ks]),
                                                 "ci95": [100 * np.percentile(bsm, 2.5), 100 * np.percentile(bsm, 97.5)]}}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if far is not None:
        out["far_px_label512"] = far
    sfx = "" if far is None else f"_far{int(far)}"
    (OUTROOT / f"compare_{'__'.join(names)}_{split}_{occ}_{mot}{sfx}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def table(names, split):
    rs = [json.load(open(OUTROOT / n / f"passline_{split}.json")) for n in names]
    print(render_md(rs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract", "analyze", "table", "compare"])
    ap.add_argument("--ckpt")
    ap.add_argument("--name")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--sets", default="vobench,gravity,natural")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--devices", default=None, help="comma list of cuda ids (default 0..gpus-1)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--n-natural", type=int, default=200)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--split", default="all", choices=["all", "test"])
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N clips per synthetic set")
    ap.add_argument("--cell", default="late/moving", help="compare: occlusion/motion cell (B1 = late/static)")
    ap.add_argument("--far", type=float, default=None, help="compare: far cells only — future blocks whose possible-future object is > FAR px (512 label scale; 48 = 1.5 cells) from the last visible context position")
    ap.add_argument("--fast", action="store_true", help="VOBench: only late-occlusion and visible-moving clips (primary + carry cells)")
    a = ap.parse_args()
    if a.phase == "extract":
        import torch
        devs = [int(d) for d in a.devices.split(",")] if a.devices else list(range(a.gpus))
        torch.multiprocessing.spawn(_worker, args=(len(devs), devs, a.ckpt, a.name, a.sets.split(","), a.bs, a.n_natural, a.fps, a.limit, a.fast),
                                    nprocs=len(devs), join=True)
    elif a.phase == "analyze":
        analyze(a.name, a.split)
    elif a.phase == "compare":
        compare(a.names, a.split, occ=a.cell.split("/")[0], mot=a.cell.split("/")[1], far=a.far)
    else:
        table(a.names, a.split)
