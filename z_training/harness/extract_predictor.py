#!/usr/bin/env python3
"""릴리즈 model.pth 에서 **predictor 만** 떼어 predictor-only 파일로 저장한다.

  python z_training/harness/extract_predictor.py                 # vith -> z_training/runs/release_vith/latest.pt
  python z_training/harness/extract_predictor.py --model vitl

용도
  * post-FT 의 출발선: bash z_training/eval.sh release_vith v11_split_test  (릴리즈 predictor 그대로 채점)
  * 학습 run 의 e{N}.pt 와 같은 형식이라 같은 도구로 비교·채점된다 (app/vjepa_frozen/utils.save_checkpoint)
파일 형식: {"predictor": <module./backbone. 뗀 state_dict>, "base_checkpoint", "epoch": 0, "format": ...}. opt 는 없다.
"""
import argparse
import os
import sys
import time

import torch
import yaml

ROOT = "/data/hyuntak/project/2026/2027_cvpr/vjepa2"
sys.path.insert(0, f"{ROOT}/z_research/scripts/harness")
from resolve import parse_registry  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="vith", help="configs/protocols/models.md 이름")
ap.add_argument("-o", "--output", default=None, help="기본 z_training/runs/release_<model>/latest.pt")
a = ap.parse_args()
MD = parse_registry(f"{ROOT}/configs/protocols/models.md", "checkpoint")
if a.model not in MD:
    raise SystemExit(f"모델 '{a.model}' 이 models.md 에 없다: {sorted(MD)}")
base = MD[a.model]["checkpoint"]
out = a.output or f"{ROOT}/z_training/runs/release_{a.model}/latest.pt"
t0 = time.time()
ck = torch.load(base, map_location="cpu", weights_only=False)
sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ck["predictor"].items()}
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save({"predictor": sd, "base_checkpoint": base, "epoch": 0, "step": 0, "loss": None,
            "context_encoder_key": "encoder", "target_encoder_key": "target_encoder",
            "arch": {"arch_name": MD[a.model]["arch_name"]}, "format": "vjepa_frozen/predictor-only/v1 (release extract)"}, out)
# eval.sh 가 model.base 를 읽을 수 있게 최소 config 를 옆에 둔다
with open(os.path.join(os.path.dirname(out), "config.yaml"), "w") as f:
    yaml.safe_dump({"model": {"base": a.model, "checkpoint": base, "arch_name": MD[a.model]["arch_name"]},
                    "note": "릴리즈 predictor 추출본. 학습 run 이 아니다."}, f, allow_unicode=True)
print(f"{len(sd)} tensors -> {out} ({os.path.getsize(out)/1e6:.0f} MB, {time.time()-t0:.0f}s)")
print(f"채점: GPUS=8 bash z_training/eval.sh release_{a.model} v11_split_test")
