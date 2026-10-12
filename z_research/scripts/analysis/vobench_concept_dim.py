#!/usr/bin/env python3
"""VOBench_v1 정체 개념 벡터 (difference-in-means) — 시점마다 정체가 같은 방향으로 표현되나 · h 와 p 가 같은 방향인가 (2026-10-10).

  입력 = vobench_tubelet_probe.py 캐시 (h: LN(target_encoder(32)) 튜블릿 16 개 · p: predictor 미래 튜블릿 8 개, 릴리즈 ViT-H, bf16).
  x[s, t] = mean_{256 공간 토큰} 표현[s, t]                                   (N, D) → (1, D) 풀링
  μ[src, t] = mean_{물체 샘플} x            c[src, t, k] = mean_{클래스 k} x − μ[src, t]      (difference-in-means; 빈 장면 제외)
  샘플: 가능 변이 · 물체 클립. h = 가림 없는 튜블릿 (metadata samples '..'), p = 문맥 끝 보임 (visible · early · mid) 의 미래 튜블릿.
  block 50/50 (seed 0, probe 와 같은 분할). 개념 벡터는 train block, 분류는 test block.

  ① 시점 간 일관성  cos(c[h,t,k], c[h,t',k]) 클래스 평균 (16×16) · p (8×8)
  ② h–p 정렬        cos(c[h,t,k], c[p,t,k]) (t = 8–15)
  ③ 신호 크기        ‖c[p,t,k]‖ / ‖c[h,t,k]‖ 클래스 평균
  ④ 잡음 천장        cos(c_train[src,t,k], c_test[src,t,k])  (같은 t, 다른 block 반)
  ⑤ 개념 분류        test 샘플 x − μ_train 를 cos 최대 개념 클래스로: h→h · p→p · h 개념→p (t 별, chance = 1/클래스 수)

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/vobench_concept_dim.py      # → z_research/VOBench_v1/vjepa2/probing/concept/
"""
from __future__ import annotations
import json, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
CACHE = Path("/local_datasets/world/world_analysis/cache/vobench_v1_vith_tubeprobe")
OUT = REPO / "z_research/VOBench_v1/vjepa2/probing/concept"
NT, NF = 16, 8


def tub_state(samples):
    s = samples.replace("|", "")
    return ["vis" if s[2 * t] == s[2 * t + 1] == "." else "occ" if s[2 * t] == s[2 * t + 1] == "H" else "mix" for t in range(NT)]


def label_of(m):
    return m["obj_a"] if m["variant"] == "pos_a" else m["obj_b"]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    M = json.load(open(CACHE / "meta.json"))["meta"]
    H = np.load(CACHE / "h.npy", mmap_mode="r"); P = np.load(CACHE / "p.npy", mmap_mode="r")
    t0 = time.time()
    Xh = np.stack([H[i].astype(np.float32).mean(1) for i in range(len(M))])          # (N, 16, D)
    Xp = np.stack([P[i].astype(np.float32).mean(1) for i in range(len(M))])          # (N, 8, D)
    print(f"pooled h {Xh.shape} p {Xp.shape} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}; md = ["# VOBench_v1 정체 개념 벡터 (difference-in-means) — V-JEPA 2 ViT-H", "",
                    "정의는 `z_research/scripts/analysis/vobench_concept_dim.py` 머리말. 개념 = 클래스 평균 − 물체 샘플 전체 평균 (시점 · 표현마다), 토큰 256 개 평균 풀링.", ""]
    for o in ("primitive", "realistic"):
        rows = [i for i, m in enumerate(M) if m["objset"] == o and label_of(m) != "empty"]
        blocks = sorted({M[i]["block_id"] for i, m in enumerate(M) if m["objset"] == o})
        rng = np.random.default_rng(0); rng.shuffle(blocks); trb = set(blocks[: len(blocks) // 2])   # probe 와 같은 분할
        labs = sorted({label_of(M[i]) for i in rows}); L = {l: j for j, l in enumerate(labs)}
        st = {i: tub_state(M[i]["samples"]) for i in rows}
        def S(src, t, split):
            if src == "h":
                ii = [i for i in rows if st[i][t] == "vis"]
            else:
                ii = [i for i in rows if M[i]["occlusion"] != "late" and st[i][t] == "vis"]
            return [i for i in ii if (M[i]["block_id"] in trb) == (split == "train")]
        X = {"h": lambda ii, t: Xh[ii, t], "p": lambda ii, t: Xp[ii, t - NF]}
        def concepts(src, t, split):
            ii = S(src, t, split); x = X[src](ii, t); mu = x.mean(0)
            y = np.array([L[label_of(M[i])] for i in ii]); C = {}
            for k in set(y):
                C[k] = x[y == k].mean(0) - mu
            return C, mu
        C = {(src, t, sp): concepts(src, t, sp) for src, ts in (("h", range(NT)), ("p", range(NF, NT))) for t in ts for sp in ("train", "test")}
        def avg_cos(A, B):
            ks = sorted(set(A) & set(B)); return float(np.mean([cos(A[k], B[k]) for k in ks])), len(ks)
        r = {"classes": len(labs)}
        # ① 시점 간 일관성 (train 반 개념끼리)
        r["h_time"] = [[avg_cos(C[("h", t, "train")][0], C[("h", u, "train")][0])[0] for u in range(NT)] for t in range(NT)]
        r["p_time"] = [[avg_cos(C[("p", t, "train")][0], C[("p", u, "train")][0])[0] for u in range(NF, NT)] for t in range(NF, NT)]
        # ④ 잡음 천장 (같은 t, train 반 vs test 반)
        r["h_split"] = [avg_cos(C[("h", t, "train")][0], C[("h", t, "test")][0])[0] for t in range(NT)]
        r["p_split"] = [avg_cos(C[("p", t, "train")][0], C[("p", t, "test")][0])[0] for t in range(NF, NT)]
        # ② h–p 정렬 · ③ 크기 비
        r["hp_cos"] = [avg_cos(C[("h", t, "train")][0], C[("p", t, "train")][0])[0] for t in range(NF, NT)]
        r["norm_ratio"] = []
        for t in range(NF, NT):
            A, B = C[("h", t, "train")][0], C[("p", t, "train")][0]; ks = sorted(set(A) & set(B))
            r["norm_ratio"].append(float(np.mean([np.linalg.norm(B[k]) / np.linalg.norm(A[k]) for k in ks])))
        r["h_norm"] = [float(np.mean([np.linalg.norm(v) for v in C[("h", t, "train")][0].values()])) for t in range(NT)]
        r["p_norm"] = [float(np.mean([np.linalg.norm(v) for v in C[("p", t, "train")][0].values()])) for t in range(NF, NT)]
        # ⑤ 개념 분류 (train 개념 · train μ 로 test 샘플)
        def classify(csrc, tsrc, t):
            Cc, mu = C[(csrc, t, "train")]; ks = sorted(Cc); W = np.stack([Cc[k] / np.linalg.norm(Cc[k]) for k in ks])
            ii = S(tsrc, t, "test"); x = X[tsrc](ii, t)
            mu_t = C[(tsrc, t, "train")][1]                       # 대상 표현 자기 μ 로 중심화 (평행 이동 차이 제거)
            z = x - mu_t; z = z / np.linalg.norm(z, axis=1, keepdims=True)
            pred = np.array(ks)[(z @ W.T).argmax(1)]; y = np.array([L[label_of(M[i])] for i in ii])
            return 100 * float((pred == y).mean()), len(ii)
        r["cls_h_h"] = [classify("h", "h", t)[0] for t in range(NT)]
        r["cls_p_p"] = [classify("p", "p", t)[0] for t in range(NF, NT)]
        r["cls_h_p"] = [classify("h", "p", t)[0] for t in range(NF, NT)]
        r["cls_p_h"] = [classify("p", "h", t)[0] for t in range(NF, NT)]
        r["n_test"] = {"h": [len(S("h", t, "test")) for t in range(NT)], "p": [len(S("p", t, "test")) for t in range(NF, NT)]}
        res[o] = r
        f = lambda v: " ".join(f"{x:5.2f}" for x in v); g = lambda v: " ".join(f"{x:5.1f}" for x in v)
        md += [f"## {o} (클래스 {len(labs)}, chance {100 / len(labs):.1f} %)", "",
               "| t | " + " | ".join(f"t{t}" for t in range(NF, NT)) + " |", "|---|" + "---:|" * NF,
               "| ④ 잡음 천장 h (train↔test 반) | " + " | ".join(f"{x:.2f}" for x in r["h_split"][NF:]) + " |",
               "| ④ 잡음 천장 p | " + " | ".join(f"{x:.2f}" for x in r["p_split"]) + " |",
               "| ② cos(c_h, c_p) | " + " | ".join(f"{x:.2f}" for x in r["hp_cos"]) + " |",
               "| ③ ‖c_p‖ / ‖c_h‖ | " + " | ".join(f"{x:.2f}" for x in r["norm_ratio"]) + " |",
               "| ⑤ 분류 h 개념 → h % | " + " | ".join(f"{x:.0f}" for x in r["cls_h_h"][NF:]) + " |",
               "| ⑤ 분류 p 개념 → p % | " + " | ".join(f"{x:.0f}" for x in r["cls_p_p"]) + " |",
               "| ⑤ 분류 h 개념 → p % | " + " | ".join(f"{x:.0f}" for x in r["cls_h_p"]) + " |",
               "| ⑤ 분류 p 개념 → h % | " + " | ".join(f"{x:.0f}" for x in r["cls_p_h"]) + " |", "",
               f"① h 시점 간 cos — t0 과의: {f(r['h_time'][0])}", f"   t7 과의: {f(r['h_time'][7])}",
               f"① p 시점 간 cos — t8 과의: {f(r['p_time'][0])}", f"⑤ h 개념 → h, t0–t15: {g(r['cls_h_h'])}", ""]
        print("\n".join(md[-16:]), flush=True)
    json.dump(res, open(OUT / "concept_dim.json", "w"), indent=1)
    md += ["## 재현", "", "```bash", "$P z_research/scripts/analysis/vobench_tubelet_probe.py extract --gpus 4   # 캐시",
           "$P z_research/scripts/analysis/vobench_concept_dim.py", "```"]
    (OUT / "RESULTS.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
