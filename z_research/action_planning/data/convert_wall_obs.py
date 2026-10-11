"""DINO-WM wall_single 관측 (episode 마다 (51, 3, 224, 224) float32 .pth, 30 MB) → uint8 memmap 하나 (2026-10-11).
    python z_research/action_planning/data/convert_wall_obs.py
출력: <root>/obses_u8.npy  shape (E, 51, 224, 224, 3) uint8 (약 14.8 GB). 학습 로더가 창 하나만 memmap 으로 읽는다.
⚠️ 원본은 65 px 렌더를 224 px 로 bilinear 확대한 float 라 픽셀의 약 16 % 가 정수가 아니다 → uint8 반올림 오차 ≤ 0.5/255 (2026-10-11 실측).
   Push-T 학습의 uint8 전송과 같은 크기이고, planning 때 시뮬레이터 관측도 같은 반올림을 거쳐 학습·평가가 일치한다.
변환 뒤 episode 몇 개를 "원본 반올림" 과 비트 대조하고 최대 오차를 찍는다."""
import os, sys, numpy as np, torch
from concurrent.futures import ThreadPoolExecutor
R = sys.argv[1] if len(sys.argv) > 1 else "/data2/local_datasets/world/action_planning/dino_wm/wall_single"
E = torch.load(os.path.join(R, "actions.pth")).shape[0]
o0 = torch.load(os.path.join(R, "obses", "episode_000.pth"))
T, C, H, W = o0.shape
out = os.path.join(R, "obses_u8.npy")
mm = np.lib.format.open_memmap(out + ".tmp", mode="w+", dtype=np.uint8, shape=(E, T, H, W, C))
def one(i):
    o = torch.load(os.path.join(R, "obses", f"episode_{i:03d}.pth"))
    assert tuple(o.shape) == (T, C, H, W), (i, o.shape)
    mm[i] = o.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    return i
with ThreadPoolExecutor(8) as ex:
    for k, _ in enumerate(ex.map(one, range(E))):
        if k % 200 == 0: print(f"{k}/{E}", flush=True)
mm.flush(); del mm
os.replace(out + ".tmp", out)
chk = np.load(out, mmap_mode="r")
for i in (0, E // 2, E - 1):
    o = torch.load(os.path.join(R, "obses", f"episode_{i:03d}.pth"))
    assert np.array_equal(chk[i], o.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()), i
    print(f"episode {i}: 반올림 최대 오차 {float((o - o.round()).abs().max()):.3f} / 255")
print(f"done {out} {chk.shape} — 원본 반올림과 비트 일치 (episode 0, {E//2}, {E-1})")
