"""attn_release_vs_ar.py 결과 JSON → 표 (markdown). 2026-11-06."""
import argparse
import json

import numpy as np

C, NB = 8, 16
ORDER = ["visible/moving", "early/moving", "mid/moving", "late/moving", "visible/static", "late/static"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    a = ap.parse_args()
    d = json.load(open(a.json))
    print("verify:", d.get("verify", d.get("verify_last")))
    A = d["attn"]

    print("\n## (A) 미래 블록 query 의 attention 시간 분포 (층 · 헤드 · query · 예측 블록 9–16 평균)")
    print("| 장면 | 모델 | 마지막 문맥 블록 8 | 문맥 1–7 | 미래 자리 (릴리즈 mask · AR 되먹임) | 문맥 안에서 블록 8 몫 (균등 12.5 %) | 첫 예측 블록 9 만: 블록 8 몫 / 문맥 1–7 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for g in ORDER:
        for mdl in ("release", "ar"):
            k = f"{mdl}|{g}"
            if k not in A:
                continue
            m = np.array(A[k]["mass"]).mean((0, 1))             # (f, t)
            fut = m[C:]
            last, early, gen = fut[:, C - 1].mean(), fut[:, :C - 1].sum(1).mean(), fut[:, C:].sum(1).mean()
            share = (fut[:, C - 1] / fut[:, :C].sum(1)).mean()
            f9 = m[C]
            print(f"| {g} | {'V-JEPA 2' if mdl == 'release' else 'AR'} | {100*last:.1f} | {100*early:.1f} | {100*gen:.1f} | {100*share:.1f} | {100*f9[C-1]:.1f} / {100*f9[:C-1].sum():.1f} |")

    print("\n## (A') 층별: 미래 query 가 문맥 블록 8 에 준 몫 (문맥 안 비율, %) — 가시 · late 움직임")
    print("| 모델 · 장면 | " + " | ".join(f"층 {i+1}" for i in range(12)) + " |")
    print("|---|" + "---:|" * 12)
    for g in ("visible/moving", "late/moving"):
        for mdl in ("release", "ar"):
            m = np.array(A[f"{mdl}|{g}"]["mass"]).mean(1)[:, C:]   # (L, f, t)
            sh = (m[:, :, C - 1] / m[:, :, :C].sum(-1)).mean(1)
            print(f"| {'V-JEPA 2' if mdl == 'release' else 'AR'} · {g} | " + " | ".join(f"{100*x:.0f}" for x in sh) + " |")

    print("\n## (B) 물체 query (예측 블록에서 물체가 보이는 칸 3×3) → 문맥 블록의 물체 자리 3×3 질량 (층 · 헤드 · query 평균, %)")
    print("| 장면 | 모델 | 문맥 1–7 물체 자리 합 | 블록 8 물체 자리 | 문맥 물체 자리 / 문맥 전체 (면적 비 3.5 %) | 미래 자리의 물체 자리 합 | 물체 query 수 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for g in ORDER:
        for mdl in ("release", "ar"):
            k = f"{mdl}|{g}"
            if k not in A:
                continue
            o, ob = np.array(A[k]["omass"]).mean(0), np.array(A[k]["obmass"]).mean(0)   # (f, t)
            nobj = np.array(A[k]["nobj"])
            fs = [f for f in range(C, NB) if nobj[f] > 0]
            if not fs:
                continue
            w = nobj[fs] / nobj[fs].sum()
            ctx_e = (o[fs, :C - 1].sum(1) * w).sum()
            ctx_l = (o[fs, C - 1] * w).sum()
            frac = ((o[fs, :C].sum(1) / ob[fs, :C].sum(1)) * w).sum()
            gen = (o[fs, C:].sum(1) * w).sum()
            print(f"| {g} | {'V-JEPA 2' if mdl == 'release' else 'AR'} | {100*ctx_e:.2f} | {100*ctx_l:.2f} | {100*frac:.1f} | {100*gen:.2f} | {int(nobj[fs].sum())} |")

    print("\n## (C) 문맥 encoder 물체 대비 OC_t (물체 자리 차이 / 먼 배경 차이; 1 = 물체 정보 없음), 블록 1–8")
    E = d["enc"]
    names = {"window": "V-JEPA 2 문맥 (16 장 한 번에, 양방향)", "causal": "같은 encoder, 블록 t 까지만 (인과)",
             "block": "같은 encoder, 블록 하나씩", "ar_input": "AR 입력 (target encoder, 블록 하나씩)"}
    for g in ("visible/moving", "late/moving", "mid/moving", "early/moving"):
        print(f"\n**{g}** (n = {len(E['window'][g])})\n")
        print("| 인코딩 | " + " | ".join(f"블록 {t+1}" for t in range(C)) + " |")
        print("|---|" + "---:|" * C)
        for k in ("window", "causal", "block", "ar_input"):
            oc = np.array([x["oc"] for x in E[k][g]])
            print(f"| {names[k]} | " + " | ".join(f"{v:.2f}" for v in oc.mean(0)) + " |")
        vis = np.array([x["vis"] for x in E["window"][g]]).mean(0)
        print("| (물체 가시 비율) | " + " | ".join(f"{v:.2f}" for v in vis) + " |")

    print("\n## (C') late 를 가림 길이 k 로 — 가림 직전 보이는 블록에서 window − causal (양수 = 뒤 프레임이 대비를 키움)")
    print("| k | n | 마지막 보이는 블록 | window | causal | 차이 |")
    print("|---|---:|---|---:|---:|---:|")
    for kk in ("1", "2", "3"):
        rows_w = [x for x in E["window"]["late/moving"] if str(x["k"]) == kk]
        rows_c = [x for x in E["causal"]["late/moving"] if str(x["k"]) == kk]
        if not rows_w:
            continue
        vis = np.array(rows_w[0]["vis"])
        lv = int(max(i for i in range(C) if vis[i] >= 0.99))
        w = np.mean([x["oc"][lv] for x in rows_w]); c = np.mean([x["oc"][lv] for x in rows_c])
        print(f"| {kk} | {len(rows_w)} | 블록 {lv+1} | {w:.2f} | {c:.2f} | {w-c:+.2f} |")


if __name__ == "__main__":
    main()
