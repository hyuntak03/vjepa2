#!/usr/bin/env python3
"""[T] 방법 B′ 판정 (1): 변위 머리 u 가 문맥을 쓰나 — 무학습 기준과 비교 (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore auto_research/scripts/u_judge.py --ft <이름> [<이름> ...] [--K 4]
  (--ft 없이 돌리면 무학습 기준만 낸다)

데이터: K710 보류 150 창 (블록 8 개, /workspace/auto_research_cache/natural_holdout: 토큰 · 역방향 칸 흐름 U · 순방향 칸 흐름 V).
  U[b] = 블록 b 대표 장 → 블록 b−1 대표 장 (칸 단위, (x, y)), V[b] = 블록 b → b+1. 정답 = U[b+1] ("블록 b+1 의 칸 c 내용이 블록 b 의 어디서 왔나").
  잴 칸 = 움직이는 칸 (|U[b+1]| ≥ 0.5 칸) 중 앞-뒤 일관 (|U[b+1](c) + V[b](c + U[b+1](c))| < 0.25 칸, 가장 가까운 칸).
한 걸음 (TF, 위치 b = 1..6 → 블록 b+1): 모델 u vs 무학습 기준 ① 0 운동 ② 같은 칸 마지막 관측 흐름 U[b] 그대로 ③ 화면 중앙값 흐름 (카메라만).
굴림 (문맥 블록 0..3 → 블록 4..7, K 걸음): 모델 u (굴린 상태에서 읽음) vs ① 0 ② U[3] 그대로 ③ U[3] 화면 중앙값.
지표: 끝점 오차 (칸) 평균. 사전 등록 기준 (진행판 [T] KST 02:29): 모델 ≤ 0.9 × (세 기준 중 가장 작은 것), 한 걸음 · 굴림 둘 다."""
import argparse, json, os, sys
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--ft", nargs="*", default=[])
ap.add_argument("--ftdir", default="/workspace/auto_research_cache/ar_ft")
ap.add_argument("--hold", default="/workspace/auto_research_cache/natural_holdout/k710_320_holdout")
ap.add_argument("--K", type=int, default=4)
ap.add_argument("--bs", type=int, default=6)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/u_judge.json"))
a = ap.parse_args()
torch.set_num_threads(8)
G, S = 16, 256
H = np.load(a.hold + "_h.npy", mmap_mode="r")
U = np.load(a.hold + "_U.npy").astype(np.float32).reshape(-1, 8, S, 2)
V = np.load(a.hold + "_Vflow.npy").astype(np.float32).reshape(-1, 8, S, 2)
N = H.shape[0]
yy, xx = np.divmod(np.arange(S), G)


def mask_for(ut, vb):                                                      # ut: 정답 U[b+1] (n,S,2), vb: V[b] (n,S,2)
    cx = np.clip(np.rint(xx[None] + ut[..., 0]), 0, G - 1).astype(int); cy = np.clip(np.rint(yy[None] + ut[..., 1]), 0, G - 1).astype(int)
    vj = np.take_along_axis(vb, (cy * G + cx)[..., None].repeat(2, -1), 1)
    cons = np.linalg.norm(ut + vj, axis=-1) < 0.25
    return cons & (np.linalg.norm(ut, axis=-1) >= 0.5), cons


def epe(p, t, m):
    return float((np.linalg.norm(p - t, axis=-1) * m).sum()), int(m.sum())


def base_preds(prev):                                                      # prev (n,S,2) = 마지막 관측 흐름
    med = np.median(prev, axis=1, keepdims=True).repeat(S, 1)
    return {"zero": np.zeros_like(prev), "last_flow": prev, "camera": med}


res = {"tf": {}, "roll": {}}
acc = {}


def add(key, part, p, t, m):
    s_, n_ = epe(p, t, m)
    acc.setdefault((part, key), [0.0, 0]); acc[(part, key)][0] += s_; acc[(part, key)][1] += n_


# 무학습 기준
for b in range(1, 7):
    m, _ = mask_for(U[:, b + 1], V[:, b])
    for k, p in base_preds(U[:, b]).items():
        add(k, "tf", p, U[:, b + 1], m)
for j in range(a.K):
    tb = 4 + j
    if tb >= 8:
        break
    m, _ = mask_for(U[:, tb], V[:, tb - 1])
    for k, p in base_preds(U[:, 3]).items():
        add(k, "roll", p, U[:, tb], m)

if a.ft:
    sys.path.insert(0, os.path.dirname(__file__))
    _argv = sys.argv; sys.argv = [sys.argv[0], "--name", "x"]
    import disp_ft as DF                                                   # noqa: E402  (DispWrap · load_pred 재사용)
    sys.argv = _argv
    dev = torch.device("cuda")
    for name in a.ft:
        ck = torch.load(os.path.join(a.ftdir, f"{name}.pt"), map_location="cpu", weights_only=False)
        W = DF.DispWrap(DF.FT.load_pred("natural_ar_vw_state", "latest.pt"))
        W.pred.load_state_dict(ck["predictor"]); W.u_head.load_state_dict(ck["u_head"]); W.m_head.load_state_dict(ck["m_head"])
        W = W.to(dev).eval()
        with torch.no_grad():
            for i0 in range(0, N, a.bs):
                x = torch.from_numpy(np.asarray(H[i0:i0 + a.bs], np.float32)).to(dev)            # (B,8,S,D)
                B = x.size(0)
                _, u, _ = W.heads(x.view(B, -1, x.size(-1)))
                u = u.cpu().numpy()                                                            # (B,8,S,2) 위치 b → 블록 b+1
                for b in range(1, 7):
                    m, _ = mask_for(U[i0:i0 + B, b + 1], V[i0:i0 + B, b])
                    add(name, "tf", u[:, b], U[i0:i0 + B, b + 1], m)
                xc = x[:, :4].reshape(B, 4 * S, -1)
                idx = torch.arange(4 * S, device=dev).unsqueeze(0).expand(B, -1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, st = W.pred.rollout_kv(xc, idx, a.K, return_states=True)
                ur = W.u_head(st.float()).view(B, a.K, S, 2).cpu().numpy()
                for j in range(a.K):
                    tb = 4 + j
                    if tb >= 8:
                        break
                    m, _ = mask_for(U[i0:i0 + B, tb], V[i0:i0 + B, tb - 1])
                    add(name, "roll", ur[:, j], U[i0:i0 + B, tb], m)
        del W; torch.cuda.empty_cache()

for (part, key), (s_, n_) in acc.items():
    res[part][key] = {"epe": s_ / max(n_, 1), "n_cells": n_}
for part in ("tf", "roll"):
    bmin = min(res[part][k]["epe"] for k in ("zero", "last_flow", "camera"))
    res[part]["best_baseline"] = bmin
    for name in a.ft:
        res[part][name]["pass"] = bool(res[part][name]["epe"] <= 0.9 * bmin)
    print(f"== {part} (움직이는 일관 칸 끝점 오차, 칸): " + " · ".join(f"{k} {v['epe']:.3f}" for k, v in res[part].items() if isinstance(v, dict))
          + f" | 기준 최소 {bmin:.3f} → 통과선 {0.9 * bmin:.3f}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
