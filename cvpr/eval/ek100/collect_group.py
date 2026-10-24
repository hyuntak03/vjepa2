#!/usr/bin/env python3
"""EK100 predictor_group 결과 추출 (vast 2026-10-21). 팔마다 val_metrics__<팔>.jsonl 의 마지막 (기본 epoch 20) 줄 →
verb / noun / action = mean class recall@5, head 중 best (시트 "EK100 subset" 칸의 정의, main GROUP4_CAP16METRIC 과 같다).

  /venv/vjepa/bin/python cvpr/eval/ek100/collect_group.py [결과 폴더] [--epoch 20]
  기본 폴더: cvpr/results/eval/ek100/action_anticipation_frozen/ek100_vast_ek100_g5
"""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("folder", nargs="?", default=os.path.join(os.path.dirname(__file__), "../../results/eval/ek100/action_anticipation_frozen/ek100_vast_ek100_g5"))
ap.add_argument("--epoch", type=int, default=20)
a = ap.parse_args()
fs = sorted(glob.glob(os.path.join(a.folder, "val_metrics__*.jsonl")))
if not fs:
    raise SystemExit(f"val_metrics__*.jsonl 없음: {a.folder}")
print(f"{'팔':<22s} {'epoch':>5s} {'verb':>7s} {'noun':>7s} {'action':>7s}   head 별 action (lr)                 n_clips")
for f in fs:
    rows = [json.loads(l) for l in open(f) if l.strip()]
    rows = [r for r in rows if not r.get("val_only")]
    r = next((x for x in rows if x["epoch"] == a.epoch), rows[-1])
    ph = " ".join(f"{h['action']['recall']:.2f}" for h in r.get("per_head", []))
    lrs = ",".join(f"{h['lr']:g}" for h in r.get("heads", []))
    flag = "" if r["epoch"] == a.epoch else f"   ⚠️ epoch {a.epoch} 아직 없음 (마지막 {r['epoch']})"
    print(f"{r['model']:<22s} {r['epoch']:>5d} {r['verb']['recall']:7.2f} {r['noun']['recall']:7.2f} {r['action']['recall']:7.2f}   {ph} ({lrs})   {r['n_clips']}/{r['expected_clips']}{flag}")
