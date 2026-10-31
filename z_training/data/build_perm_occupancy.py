#!/usr/bin/env python3
"""phys3d_perm_v1 물체 칸 라벨 — 보조 기억 모듈 감독용 (2026-10-31, 사용자: "모듈한테 context 보고 물체 놓아져야될 칸 supervision 주면서 일반화 되도록").

블록 t (연속 2 장) × 16×16 칸마다 한 값 (int8). 문맥 C 장에 따라 "문맥에서 본 물체" 가 달라지므로 C ∈ {10, 12, 14, 16} 마다 따로:
   0 = 배경 (물체 없음, 또는 문맥에서 본 적 없는 물체가 가려져 있음 — 문맥으로는 알 수 없다)
   1 = 보이는 물체 (문맥에서 본 적 있는 물체가 그 블록에서 보임) — 그 물체 상자 칸
   2 = 가려진 물체 (문맥에서 본 적 있는 물체가 화면 안인데 가림막 뒤) — 그 물체 상자 칸 (= 영속: 안 보여도 거기 있다)
  -1 = 무시 (문맥에서 한 번도 안 보였던 물체가 새로 보임 — 문맥으로는 알 수 없다)
보임 = vis_frac > 0.5 · inframe_frac > 0.5 (그 장). 블록에서 한 장이라도 보이면 보임, 두 장 모두 화면 안 · 안 보임이면 가려짐.
칸 = 그 장들의 상자 (states.npz bbox, 256 px) 가 걸치는 16 px 칸의 합집합. 겹치면 1 > -1 > 2 > 0 순으로 이긴다 (정답 토큰에 보이는 것이 이긴다).

출력 (data_csv/phys3d_perm_v1/ 아래, gitignore):
  occ_{split}.npy   (N, 4, 16, 256) int8 — 행 = index.csv 순서, 둘째 축 = C ∈ (10, 12, 14, 16)
  {split}_occ.csv   "<mp4> <행 번호>" — VideoDataset 이 둘째 열을 라벨로 넘겨 주므로 학습 루프가 행 번호로 라벨을 찾는다

  /venv/vjepa/bin/python z_training/data/build_perm_occupancy.py --split val

이동량 라벨 (2026-10-31, 사용자 "상대 표현 괜찮은데" → "다시 올리던가") — `disp_{split}.npy` (N, 4, 16, 256, 2) int8:
  칸 (r, c) · 블록 t = 문맥에서 본 물체가 **마지막으로 보인 장** 에 그 칸을 덮었으면, 그 물체 중심이 그 장 → 블록 t 둘째 장까지 움직인 양 (dx, dy).
  단위 1/4 칸 (4 px), ±127 로 자름 (±31.75 칸), 라벨 없음 = −128. 겹치면 작은 물체가 이긴다. 가려져 있어도 · 화면 밖으로 나가도 상자 중심이 유한하면 라벨을 단다.
  칸에 묶인 상대량이라 화면을 평행이동해도 같다 — 절대 칸 분류 (위) 와 달리 "이 장면이면 저 자리" 로 외울 수 없다.
  근거: 문맥 encoder 토큰의 속도 선형 probe 가 phys3d → VOBench 로 방향 cos 0.96 으로 넘어간다 (z_analysis/results/velocity_probe_cross_generator).
"""
import argparse, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ROOT = "/workspace/dataset/training_data/phys3d_perm_v1"
CTX = (10, 12, 14, 16)
NF, G, TUB = 32, 16, 2


def cells(box):
    x0, y0, x1, y1 = box
    if not np.isfinite(box).all() or x1 < 0 or y1 < 0 or x0 > 256 or y0 > 256: return []
    rs = range(max(0, int(y0 // 16)), min(G, int(y1 // 16) + 1)); cs = range(max(0, int(x0 // 16)), min(G, int(x1 // 16) + 1))
    return [r * G + c for r in rs for c in cs]


def labels(path):
    st = np.load(path[:-4] + "_states.npz")
    vis = (st["vis_frac"] > 0.5) & (st["inframe_frac"] > 0.5)             # (32, O)
    inf = (st["inframe_frac"] > 0.5) & st["present"]
    O = vis.shape[1]
    out = np.zeros((len(CTX), NF // TUB, G * G), np.int8)
    for ci, C in enumerate(CTX):
        seen = vis[:C].any(0)                                              # (O,) 문맥에서 본 적 있나
        for t in range(NF // TUB):
            lay = {2: set(), -1: set(), 1: set()}
            for o in range(O):
                fr = (2 * t, 2 * t + 1)
                vf = [f for f in fr if vis[f, o]]
                if vf:
                    k = 1 if seen[o] else -1
                    for f in vf: lay[k].update(cells(st["bbox"][f, o]))
                elif seen[o] and all(inf[f, o] for f in fr):
                    for f in fr: lay[2].update(cells(st["bbox"][f, o]))
            for k in (2, -1, 1):                                           # 뒤에 칠한 것이 이긴다
                out[ci, t, sorted(lay[k])] = k
    return out


def displacements(path):
    st = np.load(path[:-4] + "_states.npz")
    vis = (st["vis_frac"] > 0.5) & (st["inframe_frac"] > 0.5)
    b = st["bbox"]; ctr = np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2], -1)    # (32, O, 2) px
    area = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    O = vis.shape[1]
    out = np.full((len(CTX), NF // TUB, G * G, 2), -128, np.int8)
    for ci, C in enumerate(CTX):
        src = []
        for o in range(O):
            fs = np.where(vis[:C, o])[0]
            if len(fs) and st["present"][fs[-1], o]:
                src.append((area[fs[-1], o], o, fs[-1], cells(b[fs[-1], o])))
        for _, o, f, cl in sorted(src, key=lambda s_: -s_[0]):              # 큰 물체부터 칠해 작은 물체가 이긴다
            for t in range(NF // TUB):
                c1 = ctr[2 * t + 1, o]
                if not (np.isfinite(c1).all() and st["present"][2 * t + 1, o]): continue
                d = np.clip(np.round((c1 - ctr[f, o]) / 4.0), -127, 127).astype(np.int8)   # 1/4 칸 = 4 px
                out[ci, t, cl] = d
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--disp", action="store_true", help="이동량 라벨 disp_{split}.npy 만 만든다")
    a = ap.parse_args()
    if a.disp:
        paths = [ln.split()[0] for ln in open(f"{ROOT}/{a.split}/index.csv") if ln.strip()]
        with ProcessPoolExecutor(32) as ex:
            Dl = np.stack(list(ex.map(displacements, paths, chunksize=64)))
        np.save(os.path.join(REPO, "data_csv/phys3d_perm_v1", f"disp_{a.split}.npy"), Dl)
        fut = np.concatenate([Dl[:, ci, C // TUB:].reshape(-1, 2) for ci, C in enumerate(CTX)])
        ok = fut[:, 0] != -128; mag = np.hypot(*(fut[ok].astype(float) / 4).T)
        print(f"{a.split}: {Dl.shape} · 미래 칸 중 라벨 {100 * ok.mean():.2f} % · 이동 (칸) 중앙 {np.median(mag):.2f} p90 {np.percentile(mag, 90):.2f} · "
              f"0 이동 {100 * (mag == 0).mean():.1f} % · 잘림 {100 * (np.abs(fut[ok]) == 127).any(1).mean():.2f} %")
        sys.exit()
    paths = [ln.split()[0] for ln in open(f"{ROOT}/{a.split}/index.csv") if ln.strip()]
    with ProcessPoolExecutor(32) as ex:
        L = np.stack(list(ex.map(labels, paths, chunksize=64)))
    od = os.path.join(REPO, "data_csv/phys3d_perm_v1")
    np.save(os.path.join(od, f"occ_{a.split}.npy"), L)
    with open(os.path.join(od, f"{a.split}_occ.csv"), "w") as f:
        f.writelines(f"{p} {i}\n" for i, p in enumerate(paths))
    fut = np.concatenate([L[:, ci, C // TUB:].ravel() for ci, C in enumerate(CTX)])
    print(f"{a.split}: {len(paths)} clip → {L.shape} · 미래 칸 비율 " +
          " · ".join(f"{n} {100 * (fut == k).mean():.2f} %" for k, n in ((0, "배경"), (1, "보임"), (2, "가려짐"), (-1, "무시"))))
