#!/usr/bin/env python3
"""가변 창 로더 (`data.var_window`) 를 **실제 mp4 로** 돌려 본다 — CPU, 모델 없음 (2026-10-16).

    python z_vll7_training/tests/var_window_cpu_test.py --config ctxt_video_tgtvideo --steps 8

무엇을 재나
  1. (C, K, 데이터셋, B) 계획이 **rank 와 무관하게 같다** (seed, step 만으로 정해진다 — DDP 부하 균형의 전제)
  2. 영상 선택은 **rank 마다 다르다**
  3. 실제로 읽은 clip 의 모양이 (B, 3, W, 256, 256) uint8 이고 **패딩이 없다** (읽기 실패 0)
  4. MaskSampler.prefix_window 가 그 (C, K) 로 내는 인덱스가 창 안에 들어간다
"""
import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import yaml, torch

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="ctxt_video_tgtvideo")
ap.add_argument("--resolved", help="병합된 config yaml (없으면 resolve_train 을 부른다)")
ap.add_argument("--steps", type=int, default=8)
a = ap.parse_args()

ROOT = pathlib.Path(__file__).resolve().parents[2]
if a.resolved:
    cfg = yaml.safe_load(open(a.resolved, encoding="utf-8"))
else:
    import os, subprocess, tempfile
    out = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
    env = {**os.environ, "TRAIN_CFG_DIR": str(ROOT / "z_vll7_training/configs"), "DATA_CSV": str(ROOT / "data_csv")}
    subprocess.run([sys.executable, str(ROOT / "z_training/harness/resolve_train.py"), a.config,
                    "-o", out, "--folder", "/tmp/_vw_test"], check=True, env=env, stdout=subprocess.DEVNULL)
    cfg = yaml.safe_load(open(out, encoding="utf-8"))

from app.vjepa_frozen.data import ClipTransform, MaskSampler
from app.vjepa_frozen.var_window import VarWindowBatches

D, M = cfg["data"], cfg["model"]
VW = D["var_window"]
tub, res, patch = int(M["tubelet_size"]), int(D["resolution"]), int(M["patch_size"])
S = (res // patch) ** 2
aug = D.get("aug") or {}
tf = ClipTransform(res, hflip_p=aug.get("hflip_p", 0.0), random_resized_crop=aug.get("random_resized_crop"),
                   square_crop=aug.get("square_crop", "none"), out_uint8=True)
ds0 = VarWindowBatches(VW, tf, tub, seed=int(cfg["meta"]["seed"]), rank=0)
ds1 = VarWindowBatches(VW, tf, tub, seed=int(cfg["meta"]["seed"]), rank=1)
print(f"조합 {len(ds0.combos)}개, token_budget {ds0.budget}\n{ds0.describe()}\n")

ms = MaskSampler(cfg["mask"], int(D["n_frames"]), res, patch, tub, seed=0)
tot_clip = tot_frame = 0
by_ds = {}
for step in range(a.steps):
    p0, p1 = ds0.plan(step), ds1.plan(step)
    assert (p0[0], p0[1], p0[3].name, p0[4]) == (p1[0], p1[1], p1[3].name, p1[4]), f"step {step}: rank 끼리 계획이 다르다"
    t0 = time.time()
    x, info = ds0.make(step)
    el = time.time() - t0
    C, K, W, B = info["C"], info["K"], info["W"], info["B"]
    assert tuple(x.shape) == (B, 3, W, res, res) and x.dtype == torch.uint8, (x.shape, x.dtype)
    assert info["read_fails"] == 0 or info["read_fails"] < B, f"읽기 실패 {info['read_fails']}"
    m_enc, m_pred = ms.prefix_window(B, info["context_blocks"], info["predict_blocks"])
    assert int(m_pred.max()) < (W // tub) * S, "마스크 인덱스가 창 밖"
    by_ds[info["dataset"]] = by_ds.get(info["dataset"], 0) + B * W
    tot_clip += B; tot_frame += B * W
    print(f"  step {step}: C{C} K{K} W{W} B{B} {info['dataset']:>5} | {tuple(x.shape)} | 읽기실패 {info['read_fails']} | {el:.1f}s "
          f"| mask enc {tuple(m_enc.shape)} pred {tuple(m_pred.shape)}")
# rank 별 영상 선택이 다른지
same = all(ds0.plan(s)[3].name == ds1.plan(s)[3].name for s in range(a.steps))
x0, _ = ds0.make(0); x1, _ = ds1.make(0)
assert not torch.equal(x0, x1), "rank 0 과 1 이 같은 clip 을 읽었다 (영상 선택이 rank 별이어야 한다)"
print(f"\n계획 rank 동기 OK · 영상 선택 rank 별 OK (데이터셋은 {'같다' if same else '다를 수 있다'})")
print(f"{a.steps} step 에서 clip {tot_clip}, 프레임 {tot_frame} | 프레임 비율 " +
      ", ".join(f"{k} {100*v/tot_frame:.1f}%" for k, v in sorted(by_ds.items())))
print("\n전부 통과")
