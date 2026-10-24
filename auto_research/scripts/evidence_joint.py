#!/usr/bin/env python3
"""[S] 후보 F 측정의 사후 진단 (판정 밖, 학습 없음, GPU 없음): 들어오는 칸의 정답 · 기준 AR 예측 · 영상 문맥 예측을 세 증거로 **함께** 회귀한다.
  y ≈ a·x1 + b·E3x + c·Ec
  x1 = 같은 칸 5 블록 평균, E3x = 옛 자리 (흐름이 고른 블록 t 의 자리 s) 둘레 3×3 평균 (들어오는 칸 c 자신은 뺌), Ec = 같은 칸 마지막 블록 (복사 증거).
⚠️ 흐름이 고른 옛 자리 토큰은 데이터 통계 진단용 회귀자다 (GUARDRAILS 2-1, 예측기 아님).
질문: 기준 AR 의 "덜 옮김" 은 옮길 내용을 덜 쓰는 것 (b 모델 < b 데이터) 인가, 옛 내용을 덜 지우는 것 (c 모델 > c 데이터) 인가. 영상 문맥은 어느 쪽에서 다른가.
데이터 = [P] 공유 forward (ctxt_compare): K710 보류 · HT100M (자연), 3D 합성. t = 4, 5, 6 (예측 = TF 한 걸음; 영상 문맥은 t = 4 만 영상 문맥에서 나온 예측, 5 · 6 은 블록 관측 TF 라 따로 낸다).

  OMP_NUM_THREADS=8 /venv/vjepa/bin/python auto_research/scripts/evidence_joint.py"""
import json
import numpy as np, torch

torch.set_num_threads(8)
CC = "/workspace/auto_research_cache/ctxt_compare"; REPO = "/workspace/world_model/vjepa2"
G, S = 16, 256
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
FLOW = {"k710_holdout": np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_U.npy").astype(np.float32),
        "ht100m": np.load(f"{REPO}/auto_research/results/natural_transport_ctx5/rows_ht100m.npz")["Uwin"].astype(np.float32),
        "phys3d": np.load(f"{REPO}/auto_research/results/natural_transport_ctx5/rows_phys3d.npz")["Uwin"].astype(np.float32)}
TG = ("y", "base", "ctxt")


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.where(ok, np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), np.arange(S)), ok


per = {}
for src, U in FLOW.items():
    meta = json.load(open(f"{CC}/{src}_meta.json"))
    CX, GT, PB, PC = (np.load(f"{CC}/{src}_{k}.npy", mmap_mode="r") for k in ("ctx", "gt", "pred_base", "pred_ctxt"))
    for i, w in enumerate(meta["windows"]):
        h = torch.from_numpy(np.concatenate([np.asarray(CX[i]), np.asarray(GT[i])]).astype(np.float32))
        pb = torch.from_numpy(np.asarray(PB[i]).astype(np.float32)); pc = torch.from_numpy(np.asarray(PC[i]).astype(np.float32))
        for t in (4, 5, 6):
            u1 = U[w, t + 1].reshape(S, 2)
            sel = np.where(np.linalg.norm(u1, axis=1) >= 0.5)[0]
            js, ok1 = cell_of(P0 + u1)
            sel = sel[ok1[sel]]
            if sel.size == 0:
                continue
            c = torch.from_numpy(sel); sj = torch.from_numpy(js[sel])
            ys, xs = sj // G, sj % G; yc, xc = c // G, c % G
            cnt = torch.zeros(sel.size); acc = torch.zeros(sel.size, h.shape[-1])
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy_, xx_ = ys + dy, xs + dx
                    ok = (yy_ >= 0) & (yy_ < G) & (xx_ >= 0) & (xx_ < G) & ~((yy_ == yc) & (xx_ == xc))
                    acc += h[t][(yy_.clamp(0, G - 1) * G + xx_.clamp(0, G - 1))] * ok[:, None].float(); cnt += ok.float()
            X = torch.stack([h[t - 4:t + 1].mean(0)[c], acc / cnt.clamp_min(1)[:, None], h[t][c]])          # (3, n, D)
            Y = {"y": h[t + 1][c], "base": pb[t + 1 - 4][c], "ctxt": pc[t + 1 - 4][c]}
            Gm = torch.einsum("knd,lnd->kl", X, X).double().numpy()
            row = [Gm] + [torch.einsum("knd,nd->k", X, Y[k]).double().numpy() for k in TG] + [float((Y[k].double() ** 2).sum()) for k in TG]
            key = (src, "t4" if t == 4 else "t56")
            per.setdefault(key, []).append(row)


def solve(rows):
    Gm = sum(r[0] for r in rows); o = {}
    for j, k in enumerate(TG):
        b = sum(r[1 + j] for r in rows); yy = sum(r[4 + j] for r in rows)
        w = np.linalg.solve(Gm, b); o[k] = w; o[k + "_r2"] = 1 - (yy - 2 * w @ b + w @ Gm @ w) / yy
    return o


rng = np.random.default_rng(0); out = {}
groups = {"natural": [("k710_holdout",), ("ht100m",)], "phys3d": [("phys3d",)]}
for gname, srcs in groups.items():
    for tk in ("t4", "t56"):
        rows = sum((per.get((s[0], tk), []) for s in srcs), [])
        v = solve(rows); W = len(rows)
        bs = [solve([rows[i] for i in rng.integers(0, W, W)]) for _ in range(300)]
        res = {}
        for k in TG:
            lo, hi = np.percentile([b_[k] for b_ in bs], [2.5, 97.5], axis=0)
            res[k] = dict(a=[round(float(v[k][0]), 3), round(float(lo[0]), 3), round(float(hi[0]), 3)],
                          b_move=[round(float(v[k][1]), 3), round(float(lo[1]), 3), round(float(hi[1]), 3)],
                          c_copy=[round(float(v[k][2]), 3), round(float(lo[2]), 3), round(float(hi[2]), 3)], r2=round(float(v[k + "_r2"]), 3))
        out[f"{gname}|{tk}"] = res
        print(f"== {gname} {tk} (행 {W})")
        for k in TG:
            r = res[k]
            print(f"  {k:5s} a(같은 칸 평균) {r['a'][0]:+.3f}  b(옛 자리 3×3) {r['b_move'][0]:+.3f} [{r['b_move'][1]:+.3f},{r['b_move'][2]:+.3f}]  c(같은 칸 복사) {r['c_copy'][0]:+.3f} [{r['c_copy'][1]:+.3f},{r['c_copy'][2]:+.3f}]  R² {r['r2']:.3f}")
json.dump(out, open(f"{REPO}/auto_research/results/evidence_joint.json", "w"), indent=1)
