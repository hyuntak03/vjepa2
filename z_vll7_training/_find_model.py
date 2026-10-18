#!/usr/bin/env python3
"""run 폴더를 가리키는 cvpr/registry/models.yaml 항목 찾기 (eval.sh 보조).

    python z_vll7_training/_find_model.py <models.yaml> <run 폴더>
    -> "<모델이름> <레지스트리가 적은 체크포인트 파일명>"  (없으면 아무것도 안 찍는다)

파일 이름이 달라도 **폴더만 같으면** 찾는다 — eval.sh 가 CKPT 가 다르면 SET 으로 덮는다.
"""
import os
import sys

import yaml

reg, d = sys.argv[1], os.path.abspath(sys.argv[2])
root = os.environ.get("VJEPA2_ROOT", "")
y = yaml.safe_load(open(reg, encoding="utf-8")) or {}
for name, v in y.items():
    if not isinstance(v, dict):
        continue
    p = str(v.get("predictor_checkpoint", "")).replace("${CVPR_REPO}", root)
    if p and os.path.abspath(os.path.dirname(p)) == d:
        print(name, os.path.basename(p))
        break
