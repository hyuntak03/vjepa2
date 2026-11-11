"""고정 창 slot 라벨 검사 (CPU, 2026-11-08). /venv/vjepa/bin/python z_training/tests/slot_fixed_labels_test.py
① 크롭 없음: var_window._slot_labels_fixed == data.Mp4ListDataset._slot_labels 와 같은 계산 (4×4 평균 · 블록 평균)
② 크롭 · 반전: 마스크 무게 중심이 같은 창으로 옮긴 물체 중심 (center_px) 과 1 토큰 안에서 맞는다"""
import os, random, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np, torch, decord
from app.vjepa_frozen.data import ClipTransform
from app.vjepa_frozen.var_window import _slot_labels_fixed
L = [l.strip() for l in open("data_csv/blender_track_v4bc4/arms/all_mp4list.csv") if l.strip()]
random.seed(0)
paths = random.sample(L, 24)
idx = list(range(0, 64, 2))
# ①
tf = ClipTransform(256, hflip_p=0.0, random_resized_crop=None, square_crop="center", out_uint8=True)
for p in paths[:6]:
    arr = decord.VideoReader(p, num_threads=1).get_batch(idx).asnumpy()
    tf(arr)
    m, r = _slot_labels_fixed(p, idx, tf)
    z = np.load(p.replace("/clips/", "/labels/")[:-4] + ".npz")
    k = z["visible"].shape[1]
    v = np.unpackbits(z["visible"][np.asarray(idx)], axis=-1).astype(np.float32)
    ref = np.zeros((16, 3, 16, 16), np.float32)
    ref[:, :k] = v.reshape(16, 2, k, 16, 4, 16, 4).mean(axis=(1, 4, 6))
    ref = torch.from_numpy((ref * 255).round().astype(np.uint8)).reshape(16, 3, 256)
    assert torch.equal(m, ref), p
print("① 크롭 없음: 기존 라벨과 같음")
# ②
tf = ClipTransform(256, hflip_p=0.5, random_resized_crop={"scale": [0.3, 1.0], "ratio": [0.75, 1.35]}, square_crop="center", out_uint8=True)
errs, n_crop = [], 0
for p in paths:
    arr = decord.VideoReader(p, num_threads=1).get_batch(idx).asnumpy()
    for _ in range(3):
        tf(arr)
        m, r = _slot_labels_fixed(p, idx, tf)
        top, left, h, w, H, W = tf.last_box
        n_crop += (h, w) != (H, W)
        z = np.load(p.replace("/clips/", "/labels/")[:-4] + ".npz")
        if z["center_px"].shape[1] == 0:                              # 빈 장면 (물체 없음)
            continue
        cen = z["center_px"][np.asarray(idx), 0]                      # (T, 2) target 중심 (x, y)
        vis = z["visible_n"][np.asarray(idx), 0]
        for b in range(16):
            mm = m[b, 0].float().view(16, 16)
            if mm.sum() < 255 * 2 or vis[2 * b] < 40:
                continue
            ys, xs = torch.meshgrid(torch.arange(16.) + 0.5, torch.arange(16.) + 0.5, indexing="ij")
            cy, cx = float((mm * ys).sum() / mm.sum()), float((mm * xs).sum() / mm.sum())
            ex = (cen[2 * b, 0] + cen[2 * b + 1, 0]) / 2
            ey = (cen[2 * b, 1] + cen[2 * b + 1, 1]) / 2
            tx, ty = (ex - left) / w * 16, (ey - top) / h * 16
            if tf.last_flip:
                tx = 16 - tx
            if 0 < tx < 16 and 0 < ty < 16:
                errs.append(np.hypot(cx - tx, cy - ty))
errs = np.array(errs)
print(f"② 크롭 {n_crop}/{len(paths) * 3} · 블록 {len(errs)} 개: 중심 오차 중앙값 {np.median(errs):.2f} 토큰 · 90 % {np.percentile(errs, 90):.2f} · 최대 {errs.max():.2f}")
assert np.median(errs) < 1.0, "크롭 라벨이 물체 중심과 안 맞는다"
print("slot_fixed_labels_test 통과")
