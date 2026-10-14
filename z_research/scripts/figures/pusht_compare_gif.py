"""Push-T planning GIF 비교: 목표마다 [post-FT | scratch | goal] 을 한 프레임에 나란히 (2026-10-10).

    python z_research/scripts/figures/pusht_compare_gif.py [--src z_research/action_planning/figures/gif_2026-10-10]

입력 = plan_pusht.py --gif 가 만든 GIF (프레임 = 34 px 글자 띠 + [시뮬레이터 224 | 6 | 목표 224]).
출력 (같은 폴더): compare_goalNN.gif (목표별), compare_all.gif (6 목표 격자), compare_sheet.png (처음·중간·마지막).
짧은 쪽 (성공해 일찍 끝난 팔) 은 마지막 프레임을 반복해 길이를 맞춘다. 라벨은 영어 (한글 폰트 없음, CLAUDE.md §8-4).
"""
import argparse, glob, os, re
import imageio.v3 as iio
import imageio
import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="z_research/action_planning/figures/gif_2026-10-10")
ap.add_argument("--scale", type=int, default=2)
ap.add_argument("--fps", type=float, default=12)
ap.add_argument("--arms", nargs=2, default=["pusht_ar_postft", "pusht_ar_scratch"], help="왼쪽 · 가운데 팔의 run 이름 (2026-10-12)")
ap.add_argument("--labels", nargs=2, default=["post-FT (natural-video AR init, lr 5e-5)", "scratch (lr 3e-4)"])
a = ap.parse_args()
ARMS = list(zip(a.arms, a.labels))
GREEN, RED = (0, 130, 0), (200, 0, 0)


def load(arm, k):
    f = glob.glob(f"{a.src}/{arm}_goal{k:02d}_[SF]*.gif")
    assert len(f) == 1, (arm, k, f)
    ok = "SUCCESS" in f[0]
    fr = iio.imread(f[0])                                   # (n, 258, 454, 3)
    sim, goal = fr[:, 34:, :224], fr[0, 34:, 230:454]
    return sim, goal, ok, len(fr)


def up(x):
    return np.asarray(Image.fromarray(x).resize((x.shape[1] * a.scale, x.shape[0] * a.scale), Image.NEAREST))


goals = sorted({int(m.group(1)) for f in glob.glob(f"{a.src}/{a.arms[0]}_goal*.gif") if (m := re.search(r"goal(\d+)_", f))})
per_goal = {}
for k in goals:
    sims, oks, ns = [], [], []
    for arm, _ in ARMS:
        s, g, ok, n = load(arm, k); sims.append(s); oks.append(ok); ns.append(n); goal = g
    T = max(len(s) for s in sims)
    sims = [np.concatenate([s, np.repeat(s[-1:], T - len(s), 0)]) if len(s) < T else s for s in sims]
    frames = []
    for t in range(T):
        row = np.concatenate([up(sims[0][t]), np.full((224 * a.scale, 8, 3), 255, np.uint8),
                              up(sims[1][t]), np.full((224 * a.scale, 8, 3), 255, np.uint8), up(goal)], axis=1)
        im = Image.new("RGB", (row.shape[1], row.shape[0] + 46), (255, 255, 255)); im.paste(Image.fromarray(row), (0, 46))
        d = ImageDraw.Draw(im); W = 224 * a.scale + 8
        for j, (arm, lab) in enumerate(ARMS):
            d.text((4 + j * W, 2), lab, fill=(0, 0, 0))
            st = ("SUCCESS" if oks[j] else "FAIL") + (f" (done at frame {ns[j] - 16})" if oks[j] and t >= ns[j] - 16 else "")
            d.text((4 + j * W, 18), st, fill=GREEN if oks[j] else RED)
        d.text((4 + 2 * W, 2), f"goal #{k}", fill=(0, 0, 0)); d.text((4 + 2 * W, 18), f"step {t}/{T - 1}", fill=(0, 0, 0))
        frames.append(np.asarray(im))
    per_goal[k] = frames
    imageio.mimsave(f"{a.src}/compare_goal{k:02d}.gif", frames, duration=1.0 / a.fps, loop=0)
    print(f"compare_goal{k:02d}.gif  {a.arms[0]} {'S' if oks[0] else 'F'}  {a.arms[1]} {'S' if oks[1] else 'F'}  ({T} frames)")

# 전체 격자 (목표마다 한 줄) — 크기 때문에 scale 1 로 다시 줄인다
T = max(len(v) for v in per_goal.values())
grid = []
for t in range(T):
    rows = []
    for k in goals:
        fr = per_goal[k][min(t, len(per_goal[k]) - 1)]
        rows.append(np.asarray(Image.fromarray(fr).resize((fr.shape[1] // a.scale, fr.shape[0] // a.scale), Image.BILINEAR)))
    grid.append(np.concatenate(rows, axis=0))
imageio.mimsave(f"{a.src}/compare_all.gif", grid, duration=1.0 / a.fps, loop=0)
sheet = []
for k in goals:
    v = per_goal[k]; pick = [0, len(v) // 2, len(v) - 1]
    sheet.append(np.concatenate([v[i] for i in pick], axis=1))
Image.fromarray(np.concatenate(sheet, axis=0)).save(f"{a.src}/compare_sheet.png")
print(f"compare_all.gif ({T} frames), compare_sheet.png")
