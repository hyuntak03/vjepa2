#!/usr/bin/env python3
"""학습 run 의 predictor-only 체크포인트를 **릴리즈 형식의 통짜 파일**로 내보낸다.

  python z_training/harness/export_ckpt.py z_training/runs/<name>/latest.pt -o /data2/.../<name>_full.pt

평소에는 필요 없다 — 채점은 `bash z_training/eval.sh <run>` 이 base checkpoint + predictor_checkpoint 로
바로 한다. 통짜 파일은 (1) configs/protocols/models.md 에 모델로 등록하거나 (2) 남에게 줄 때 쓴다.
ViT-H 는 encoder 두 벌이 들어가 ~5 GB 라 NFS 에 쓰면 90 초쯤 걸린다.

키는 릴리즈 model.pth 와 같다: encoder / target_encoder / predictor (module.backbone.* 접두어) / epoch ...
opt / scaler 는 넣지 않는다 (app/vjepa/train.py 의 load_checkpoint 는 opt 를 요구하므로 그 경로로는 못 이어 돈다).
"""
import argparse
import os
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("run_ckpt", help="z_training/runs/<name>/latest.pt 또는 run 폴더")
ap.add_argument("-o", "--output", default=None)
a = ap.parse_args()

p = a.run_ckpt
if os.path.isdir(p):
    p = os.path.join(p, "latest.pt")
ck = torch.load(p, map_location="cpu", weights_only=False)
if "predictor" not in ck or "base_checkpoint" not in ck:
    raise SystemExit(f"{p}: vjepa_frozen 체크포인트가 아니다 (predictor / base_checkpoint 키 필요)")
base_path = ck["base_checkpoint"]
print(f"run ckpt : {p}  (epoch {ck.get('epoch')}, step {ck.get('step')}, loss {ck.get('loss')})")
print(f"base     : {base_path}")
t0 = time.time()
base = torch.load(base_path, map_location="cpu", weights_only=False)
print(f"base loaded {time.time()-t0:.0f}s; keys {list(base)}")

pred_sd = {f"module.backbone.{k}": v for k, v in ck["predictor"].items()}
out = {"encoder": base["encoder"], "target_encoder": base["target_encoder"], "predictor": pred_sd,
       "epoch": ck.get("epoch", 0), "loss": ck.get("loss"), "batch_size": ck.get("config", {}).get("data", {}).get("batch_size"),
       "world_size": None, "lr": ck.get("config", {}).get("optimization", {}).get("lr"),
       "export": {"from": os.path.abspath(p), "base_checkpoint": base_path, "arch": ck.get("arch"),
                  "arch_name": ck.get("config", {}).get("model", {}).get("arch_name"),
                  "context_encoder_key": ck.get("context_encoder_key"), "target_encoder_key": ck.get("target_encoder_key"),
                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}}
o = a.output or os.path.join(os.path.dirname(os.path.abspath(p)), f"export_full_{os.path.basename(p)}")
t0 = time.time()
torch.save(out, o)
print(f"wrote {o} ({os.path.getsize(o)/1e9:.2f} GB, {time.time()-t0:.0f}s)")
print("models.md 에 등록하려면:\n"
      f"## <이름>\n\narch_name: {out['export']['arch_name'] or '<config.model.arch_name 없음 — 직접 적을 것>'}\ncheckpoint: {o}")
