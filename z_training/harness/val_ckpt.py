#!/usr/bin/env python
"""학습 run 의 **in-loop val 을 체크포인트만으로 다시 낸다** (학습 루프 · DDP 없이, GPU 1 장).

왜: Ariel 학습 코드 (`z_ariel/vjepa2_train_code_20260927`) 를 이식한 뒤 우리 레포가 **같은 숫자**를 내는지
확인하는 정합성 검사다. 체크포인트 안의 `config` (학습 때 병합된 것) 를 그대로 쓰고, 이 서버에 맞지 않는
경로 세 개만 바꾼다 (base model.pth · val index root · val frames_root). 평가 규칙은 학습 루프와 같은 함수다:
  kind=ar            -> app/vjepa_frozen/ar.py::run_val_ar  (주 = rollout surprise, 보조 = tf)
  그 밖 (oneshot/prefix/…) -> app/vjepa_frozen/val.py::run_val (mask token 규약)
정밀도도 학습 루프와 같다: encoder 가중치 = meta.encoder_dtype (기본 bf16), autocast = meta.dtype (기본 bf16),
predictor 가중치 fp32. torch.compile 은 기본 끔 (`--compile` 로 켠다; Ariel run 은 켜고 돌았다 — bf16 잡음 수준 차이).

  PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  CUDA_VISIBLE_DEVICES=0 $PY z_training/harness/val_ckpt.py z_training/ariel/block_causal_future_only_1e_5/latest.pt \
      --ref z_ariel/vjepa2_train_code_20260927/z_training/runs/nat_prefix_scratch/metrics.jsonl
  (--ref 를 주면 같은 epoch 의 기록과 acc · surprise_pos · surprise_neg · margin 을 나란히 찍는다)

출력: stdout 에 요약, `--out` 을 주면 json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from app.vjepa_frozen import utils as U                       # noqa: E402
from app.vjepa_frozen import ar as AR                         # noqa: E402
from app.vjepa_frozen.val import ValSet, run_val              # noqa: E402

# 이 서버 (vll5) 의 경로. 학습 config 는 Ariel 서버 경로 (/nas2 · ${DATA_CSV}) 를 담고 있다.
VITH = ("/data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/"
        "b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
IP1_ROOT = "/local_datasets/world/world_analysis/IntPhys1_dev_videos"
IP1_FRAMES = "/local_datasets/world/world_analysis/IntPhys1_dev_frame_png"


def _snapshot_tail(p: str) -> str:
    """HF 스냅샷 경로의 꼬리 (snapshots/<hash>/original/model.pth) — 서버가 달라도 같은 파일인지 본다."""
    parts = p.replace("\\", "/").split("/")
    return "/".join(parts[parts.index("snapshots"):]) if "snapshots" in parts else os.path.basename(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="predictor-only 체크포인트 (format vjepa_frozen/predictor-only/v1, config 포함)")
    ap.add_argument("--base", default=VITH, help="frozen encoder 가 든 model.pth (기본 models.md 의 vith)")
    ap.add_argument("--val-root", default=IP1_ROOT)
    ap.add_argument("--frames-root", default=IP1_FRAMES)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--compile", action="store_true", help="동결 encoder torch.compile (Ariel run 과 같게)")
    ap.add_argument("--autocast", default=None, help="덮어쓰기 (기본 = 체크포인트 config meta.dtype)")
    ap.add_argument("--ref", default=None, help="Ariel metrics.jsonl — 같은 epoch 의 val 과 나란히 비교")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg, arch = ck["config"], ck.get("arch") or {}
    M, MD, D, L, V = cfg.get("meta", {}), dict(cfg["model"]), cfg["data"], cfg.get("loss", {}), dict(cfg["val"])
    if _snapshot_tail(MD["checkpoint"]) != _snapshot_tail(a.base):
        raise SystemExit(f"base encoder 가 다르다: 학습 {MD['checkpoint']} vs 여기 {a.base}")
    MD["checkpoint"] = a.base
    V["data"] = dict(V["data"], root=a.val_root, frames_root=a.frames_root)
    kind = str((MD.get("predictor") or {}).get("kind", "oneshot"))
    if arch.get("kind") and arch["kind"] != kind:
        raise SystemExit(f"arch.kind={arch['kind']} 와 config kind={kind} 가 다르다")
    n_frames, res = int(D["n_frames"]), int(D.get("resolution", 256))
    tubelet, patch = int(MD.get("tubelet_size", 2)), int(MD.get("patch_size", 16))
    spatial = (res // patch) ** 2
    ac_dtype = U.parse_dtype(a.autocast or M.get("dtype", "bfloat16"))
    enc_dtype = U.parse_dtype(M.get("encoder_dtype", "bfloat16")) or torch.float32
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)

    t0 = time.time()
    state = U.load_base_checkpoint(MD["checkpoint"])
    ctx_enc, tgt_enc = U.build_frozen_encoders(MD, state, dev, enc_dtype, n_frames)
    if a.compile:
        tgt_enc = torch.compile(tgt_enc, mode="default", dynamic=True)
        ctx_enc = tgt_enc if MD.get("context_encoder_key") == MD.get("target_encoder_key") else \
            torch.compile(ctx_enc, mode="default", dynamic=True)
    pred = U.build_predictor(MD, 1280 if a.compile else ctx_enc.embed_dim, None, dev, n_frames)
    msg = pred.load_state_dict(U._clean_backbone_key(ck["predictor"]) if hasattr(U, "_clean_backbone_key")
                               else ck["predictor"], strict=True)
    del state
    print(f"[val_ckpt] {a.ckpt} | kind {kind} (mask_mode {getattr(pred, 'mask_mode', '-')}) | epoch {ck.get('epoch')} "
          f"step {ck.get('step')} | encoders {MD.get('context_encoder_key')}/{MD.get('target_encoder_key')} "
          f"{'shared' if ctx_enc is tgt_enc else 'dual'} {enc_dtype} | autocast {ac_dtype} | load {msg} | "
          f"{time.time()-t0:.0f}s", flush=True)

    vs = ValSet(V, n_frames, res)
    ctxl, bs = int(V.get("context_length", 16)), int(V.get("batch_size", 8))
    loss_exp = float(L.get("loss_exp", 1.0))
    t1 = time.time()
    if kind == "ar":
        res_ = AR.run_val_ar(vs, tgt_enc, pred, dev, ac_dtype, 0, 1, tubelet, spatial, ctxl, bs, loss_exp)
    else:
        res_ = run_val(vs, ctx_enc, tgt_enc, pred, dev, ac_dtype, 0, 1, tubelet, spatial, ctxl, bs,
                       int(MD.get("mask_index", 0)), loss_exp, bool(L.get("target_layer_norm", True)))
    print(f"[val_ckpt] {len(vs)} clips, context {ctxl} / n_frames {n_frames}, {time.time()-t1:.0f}s", flush=True)
    keys = ("acc", "n_ties", "surprise_pos", "surprise_neg", "margin")
    print("  ours : " + "  ".join(f"{k} {res_.get(k)}" for k in keys)
          + "  | " + "  ".join(f"{t} {v['acc']:.4f}" for t, v in (res_.get("by_type") or {}).items()))
    if res_.get("tf"):
        print("  ours tf: " + "  ".join(f"{k} {res_['tf'].get(k)}" for k in keys))
    ref = None
    if a.ref:
        for line in open(a.ref):
            j = json.loads(line)
            if int(j.get("epoch", -1)) == int(ck.get("epoch", -2)):
                ref = j.get("val")
        if ref:
            print("  ref  : " + "  ".join(f"{k} {ref.get(k)}" for k in keys)
                  + "  | " + "  ".join(f"{t} {v['acc']:.4f}" for t, v in (ref.get("by_type") or {}).items()))
            if ref.get("tf"):
                print("  ref tf: " + "  ".join(f"{k} {ref['tf'].get(k)}" for k in keys))
        else:
            print(f"  ref  : epoch {ck.get('epoch')} 기록 없음 ({a.ref})")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"ckpt": a.ckpt, "epoch": ck.get("epoch"), "kind": kind, "compile": a.compile,
                       "autocast": str(ac_dtype), "ours": res_, "ref": ref}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
