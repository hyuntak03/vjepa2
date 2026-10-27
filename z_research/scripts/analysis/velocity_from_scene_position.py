"""대조: 3D 합성에서 장면 종류 · 물체 위치 · 블록 번호만으로 블록 안 속도를 얼마나 맞히나 (2026-10-26, CPU, 모델 없음).
자연 영상 AR 출력에서 '자기 판독기' 가 속도를 R² 0.64 로 읽는데 (phys_only_ar_velocity.json), 그게 움직임 특징인지 장면 · 위치 단서인지 가르려고.
표본 규칙은 tf_output_velocity.py 와 같다 (대상 블록 b = 2–15, b−1 · b 물체 보임). 학습 = ar_layers train_meta 전체, 시험 = val_meta.
  ① 선형: 장면 원핫 ⊗ [1, x, y, b] (장면마다 따로 기울기) · ② kNN (같은 장면 안, (x, y, b) 거리, k = 20 평균)
  ③ 참고 — 직전 블록 실제 속도를 그대로 (데이터 지속성)
  /venv/vjepa/bin/python z_research/scripts/analysis/velocity_from_scene_position.py → z_analysis/results/velocity_from_scene_position.json
"""
import json
import numpy as np
SRC = "/workspace/auto_research_cache/ar_layers"


def rows(split):
    out = []
    for m in json.load(open(f"{SRC}/{split}_meta.json")):
        for b in range(2, 16):
            bl, bp = m["blocks"][b], m["blocks"][b - 1]
            if bl["ok"] and bp["ok"]:
                out.append((m["scen"], *bl["ctr"], b, *bl["disp"], *bp["disp"]))
    return out


tr, va = rows("train"), rows("val")
scens = sorted({r[0] for r in tr})
def X(rs):
    A = np.zeros((len(rs), len(scens) * 4))
    for i, r in enumerate(rs):
        j = scens.index(r[0]) * 4; A[i, j:j + 4] = [1, r[1] / 256, r[2] / 256, r[3] / 16]
    return A
Ytr = np.array([r[4:6] for r in tr]); Yva = np.array([r[4:6] for r in va])
r2 = lambda P, Y: round(float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()), 3)
W = np.linalg.lstsq(X(tr), Ytr, rcond=None)[0]; P1 = X(va) @ W
P2 = np.zeros_like(Yva); Ftr = np.array([[r[1], r[2], r[3] * 16] for r in tr]); Str = np.array([r[0] for r in tr])
for i, r in enumerate(va):
    m = Str == r[0]; d = np.linalg.norm(Ftr[m] - np.array([r[1], r[2], r[3] * 16]), axis=1); P2[i] = Ytr[m][np.argsort(d)[:20]].mean(0)
P3 = np.array([r[6:8] for r in va])
R = {"n_train": len(tr), "n_val": len(va), "linear_scene_x_position_block": r2(P1, Yva), "knn20_scene_position_block": r2(P2, Yva), "copy_previous_block_velocity": r2(P3, Yva)}
print(json.dumps(R, indent=1)); json.dump(R, open("/workspace/world_model/vjepa2/z_analysis/results/velocity_from_scene_position.json", "w"), indent=1)
