#!/usr/bin/env python3
"""[T] 진단 (학습 없음): 어떤 표적 공간이 "옮긴 내용이 정답과 맞는 정도" (내용 일치 성분 κ = 실제 흐름 b) 를 올리나 (auto_research, 2026-10-21).

  OMP_NUM_THREADS=8 /venv/vjepa/bin/python auto_research/scripts/target_space_b.py

[E] 수집 캐시 (블록 8 개 창; 변형 0 = 학습 정답 16 비트, 2 · 3 = ±1/255, 4 = 재압축) + 흐름 (natural_transport_ctx5 의 칸 흐름).
움직이는 칸 (실제 흐름 ≥ 0.5 칸/블록) 에서 정답 ≈ a·(같은 칸 5 블록 평균) + b·(h_t 를 실제 흐름으로 옮긴 것), 맞춤 반 · 확인 반.
표적 공간 (증거 · 정답 모두 같은 변환):
  orig      원래 블록 토큰
  denoise4  같은 프레임 네 인코딩 평균 (픽셀 미세 변화 nuisance 일부 제거)
  blur3     칸 3×3 평균 (토큰 공간의 위치 흐림 = 칸 아래 위상 · 블록 안 움직임 세부를 깎음, nuisance 도 칸 평균만큼 줄어듦)
  blur3_dn  둘 다
  blur5     칸 5×5 평균
자연 (K710 · SSv2 · HT100M) vs 3D 합성. 가설: 내용 일치 성분이 위상 · 세부 탓이면 흐림에서 자연/합성 비가 1 쪽으로 간다. nuisance 오염 탓이면 denoise 에서 오른다."""
import json, os
import numpy as np, torch, torch.nn.functional as F

torch.set_num_threads(8)
EN = "/workspace/auto_research_cache/enc_noise"
RD = os.path.join(os.path.dirname(__file__), "../results/natural_transport_ctx5")
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G)
P0 = np.stack([cx + 0.5, cy + 0.5], 1)


def blur(h, k):                                       # h (nb, 256, D) → 칸 k×k 평균 (가장자리 반사)
    g = h.view(h.shape[0], G, G, -1).permute(0, 3, 1, 2)
    g = F.avg_pool2d(F.pad(g, (k // 2,) * 4, mode="replicate"), k, 1)
    return g.permute(0, 2, 3, 1).reshape(h.shape)


SPACES = {"orig": ([0], 1), "denoise4": ([0, 2, 3, 4], 1), "blur3": ([0], 3), "blur3_dn": ([0, 2, 3, 4], 3), "blur5": ([0], 5)}
res = {}
for src in ("k710", "ssv2", "ht100m", "phys3d"):
    U = np.load(os.path.join(RD, f"rows_{src}.npz"))["Uwin"].astype(np.float32)
    acc = {}
    w0 = 0
    for sh in (0, 1):
        T_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")
        for w in range(T_.shape[0]):
            raw = torch.from_numpy(np.asarray(T_[w]).astype(np.float32))        # (5, 8, 256, D)
            half = w % 2
            for name, (vs, k) in SPACES.items():
                h = raw[vs].mean(0)
                if k > 1:
                    h = blur(h, k)
                for t in (4, 5, 6):
                    u1 = U[w0 + w, t + 1].reshape(S, 2)
                    mov = np.linalg.norm(u1, axis=1) >= 0.5
                    if mov.sum() < 4:
                        continue
                    sp = np.floor(P0 + u1).astype(int); okp = (sp >= 0).all(1) & (sp < G).all(1)
                    j = np.where(okp, np.clip(sp[:, 1], 0, G - 1) * G + np.clip(sp[:, 0], 0, G - 1), np.arange(S))
                    mt = torch.from_numpy(mov)
                    X = torch.stack([h[t - 4:t + 1].mean(0)[mt], h[t][torch.from_numpy(j)][mt]], -1).reshape(-1, 2).double()
                    y = h[t + 1][mt].reshape(-1).double()
                    A = acc.setdefault((name, half), [torch.zeros(2, 2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)])
                    A[0] += X.T @ X; A[1] += X.T @ y
        w0 += T_.shape[0]
    r = {}
    for name in SPACES:
        bs = [float(torch.linalg.solve(*acc[(name, hh)])[1]) for hh in (0, 1)]
        r[name] = bs
    res[src] = r
    print(f"{src:7s} " + " · ".join(f"{n} {np.mean(v):.3f}" for n, v in r.items()), flush=True)
nat = {n: float(np.mean([np.mean(res[s][n]) for s in ("k710", "ssv2", "ht100m")])) for n in SPACES}
syn = {n: float(np.mean(res["phys3d"][n])) for n in SPACES}
print("자연 평균 / 3D 합성 / 비: " + " · ".join(f"{n} {nat[n]:.3f} / {syn[n]:.3f} / {nat[n] / syn[n]:.2f}" for n in SPACES))
res["summary"] = {"natural": nat, "phys3d": syn}
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/target_space_b.json"), "w"), indent=1)
