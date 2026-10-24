#!/usr/bin/env python3
"""[T] 학습 전 확인: 픽셀 공간 "무름" 증강 (블록마다 물체 색 변환) 이 latent 블록 토큰에서 실제로 안정도 차이로 나타나나 (auto_research, 2026-10-21).

  OMP_NUM_THREADS=8 /venv/vjepa/bin/python auto_research/scripts/synth_label_check.py

합성 보조 캐시 (synth_h.npy · synth_lab.npz) 에서 단단함 / 무름 별로:
  (i)  물체가 거의 안 움직인 블록 쌍 (중심 변위 < 4 px, 두 블록 보임): 물체 칸 (bbox 덮임 ≥ 0.5) 의 |h_b − h_{b−1}| (차원 평균) — 같은 칸이라 이동 없음, 변화 = 외형 변화 + nuisance
  (ii) 움직인 블록 쌍 (> 8 px): 새로 덮인 칸에서 데이터 운반 계수 b (정답 ≈ a·같은 칸 문맥 평균 + b·옮긴 h_{b−1})
  (iii) 화면 전체 (물체 밖 포함) 의 블록 간 변화 — 증강이 물체 밖 칸에도 번지나 (비국소성)
판정: 무름의 (i) 이 단단함보다 뚜렷이 크고 (ii) b 가 뚜렷이 낮으면 라벨이 latent 에서 의미가 있다."""
import json, os
import numpy as np, torch

torch.set_num_threads(8)
D = "/workspace/auto_research_cache/synth_aux"
H = np.load(os.path.join(D, "synth_h.npy"), mmap_mode="r"); L = np.load(os.path.join(D, "synth_lab.npz"))
G, S = 16, 256
c16 = np.arange(G) * 16.0


def cover(bb):
    x0, y0, x1, y1 = bb
    ox = np.clip(np.minimum(c16 + 16, x1) - np.maximum(c16, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c16 + 16, y1) - np.maximum(c16, y0), 0, 16) / 16
    return (oy[:, None] * ox[None, :]).reshape(-1)


st = {k: {"chg_obj": [], "chg_far": [], "XtX": np.zeros((2, 2)), "Xty": np.zeros(2), "n_mov": 0} for k in ("rigid", "soft")}
for w in range(H.shape[0]):
    h = np.asarray(H[w], np.float32); kind = "rigid" if L["stable"][w] else "soft"
    pos, bb, vis = L["pos"][w], L["bbox"][w], L["vis"][w]
    for b in range(1, 8):
        if not (vis[b] and vis[b - 1]) or not (np.isfinite(bb[b]).all() and np.isfinite(bb[b - 1]).all()):
            continue
        d = pos[b] - pos[b - 1]; dn = np.linalg.norm(d)
        cb, cp = cover(bb[b]), cover(bb[b - 1])
        far = (cb == 0) & (cp == 0)
        if dn < 4:
            m = (cb >= 0.5) & (cp >= 0.5)
            if m.any():
                st[kind]["chg_obj"].append(float(np.abs(h[b][m] - h[b - 1][m]).mean()))
            if far.any():
                st[kind]["chg_far"].append(float(np.abs(h[b][far] - h[b - 1][far]).mean()))
        elif dn > 8 and b >= 2:
            new = np.where((cb >= 0.5) & (cp == 0))[0]
            if new.size == 0:
                continue
            dx, dy = int(round(d[0] / 16)), int(round(d[1] / 16))
            ys, xs = np.divmod(new, G); sj = np.clip(ys - dy, 0, G - 1) * G + np.clip(xs - dx, 0, G - 1)
            X = np.stack([h[max(0, b - 5):b].mean(0)[new].ravel(), h[b - 1][sj].ravel()], 1).astype(np.float64)
            st[kind]["XtX"] += X.T @ X; st[kind]["Xty"] += X.T @ h[b][new].ravel().astype(np.float64); st[kind]["n_mov"] += 1
res = {}
for k, v in st.items():
    bcoef = float(np.linalg.solve(v["XtX"], v["Xty"])[1]) if v["n_mov"] else float("nan")
    res[k] = dict(change_obj_static=float(np.mean(v["chg_obj"])), n_static=len(v["chg_obj"]), change_far=float(np.mean(v["chg_far"])), b_moving=bcoef, n_moving=v["n_mov"])
    print(f"{k:5s}: 정지 물체 칸 블록 간 변화 {res[k]['change_obj_static']:.3f} (쌍 {res[k]['n_static']}) · 물체 밖 칸 변화 {res[k]['change_far']:.3f} · 움직인 물체 데이터 b {bcoef:.3f} (쌍 {v['n_mov']})")
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/synth_label_check.json"), "w"), indent=1)
