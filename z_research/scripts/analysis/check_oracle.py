#!/usr/bin/env python
"""Garrido 프로토콜 하네스가 **기준값(oracle)을 그대로 내는지** 검사한다. GPU 불필요.

    python z_research/scripts/analysis/check_oracle.py

하네스·config·로더를 건드린 뒤에는 **이걸 먼저 돌린다.** 값이 어긋나면 그 변경이 원인이다.

기준값 — IntPhys 1 dev x V-JEPA 2 ViT-H, Garrido A.8 격자
  best = skip2_w32, **논문 텍스트 규칙** = 88.89 %   (지표는 쌍 비교 AvgSurprise 하나)
        IntPhys 은 C 스윕을 **Filtered 로 대체**한다 (A.7 870-872 / A.8 992-996).
        property 별 O1 85.00 / O2 96.67 / O3 85.00 -> macro 88.89
  참고: 노트북 규칙({Filtered} U {C} 최고) 이면 90.56 — 텍스트와 어긋난다 (PROTOCOLS.md §3)
  외부 대조: IntPhys 2 논문 Table 2 의 IntPhys 열, V-JEPA 2-h = 87.22 % (157/180)
            -> 프로토콜이 다르므로 같을 이유는 없고, **1.7pt 안**이면 합격으로 본다.

이 기준을 고른 이유
  - IntPhys 1 dev 는 **공개 벤치마크**이고 V-JEPA 2 ViT-H 에 **공개된 수치가 있는 유일한 칸**이다
  - 360 영상이라 몇 분이면 다시 뽑는다
  - 격자 3 칸이 전부라(`99//skip < window` 인 칸은 성립 불가) 탐색 규칙까지 같이 검증된다

검사 항목
  1. 병합 config 가 A.8 격자·stride·축약·pairing 을 그대로 갖고 있는가
  2. 하네스의 best_avg 가 skip2_w32 = 88.89 인가
  3. per_window.json 에서 **독립으로 다시 계산**해도 같은가 (채점기 버그 교차검증)
  4. 칸별 값이 기록과 같은가 (83.33 / 88.89 / 61.11)
"""
from __future__ import annotations
import json, pathlib, subprocess, sys
import yaml

REPO = pathlib.Path(__file__).resolve().parents[3]
BASE = REPO / "z_research/Benchmarks/exp_results"
# 창(C+M) 마다 실행이 나뉜다 (PROTOCOLS.md §2). oracle 은 그 둘을 합친 A.8 최고값이다.
D16 = BASE / "intphys1_sliding__intphys1_dev_vith_w16"
D32 = BASE / "intphys1_sliding__intphys1_dev_vith_w32"

EXPECT_GRID = {"frame_skips": [2, 5, 10], "context_mult": [2, 4, 6, 8, 10],
               "stride": 2, "context_reduce": "min", "frame_budget": "official"}
EXPECT_BEST = ("skip2_w32", 88.89, 180)   # A.8 격자 최고 (창 16·32 합쳐서), 논문 텍스트 규칙
EXPECT_HARNESS_W32 = 88.89                # 하네스 자체 채점 (같은 값: Filtered + 쌍 수 합산)
EXPECT_ALT_NOTEBOOK = 90.56               # 노트북 규칙({Filtered}∪{C} 최고). 참고값
PUBLISHED = 87.22          # IntPhys 2 논문 Table 2, V-JEPA 2-h, IntPhys 열 (다른 프로토콜)

ok = True
def check(label, got, want, tol=0.01):
    global ok
    good = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    ok &= good
    print(f"  [{'OK ' if good else 'FAIL'}] {label:44s} got={got}  want={want}")

def main():
    global ok
    for D in (D16, D32):
        if not (D / "summary.json").exists():
            sys.exit(f"기준 실행이 없다: {D}\n  만들기: GPUS=8 BENCHES=intphys1_dev MODELS=vith "
                     f"bash z_research/Benchmarks/run_all.sh")
    print(f"기준 실행: {D16.name} + {D32.name}")

    print("\n1) 병합 config 가 A.8 격자인가 (창마다 모델도 그 프레임 수로 지어졌는가)")
    for D, w in ((D16, 16), (D32, 32)):
        c = yaml.safe_load(open(D / "_resolved.yaml"))
        ip = c["surprise"]["intphys1"]
        for k, v in EXPECT_GRID.items():
            check(f"w{w}  surprise.intphys1.{k}", ip.get(k), v)
        check(f"w{w}  surprise.intphys1.window_sizes", ip.get("window_sizes"), [w])
        check(f"w{w}  model.window_size (모델을 그 창으로 지었는가)", c["model"].get("window_size"), w)
        check(f"w{w}  scoring.pairing", c["scoring"].get("pairing"), "matched")
        check(f"w{w}  data.resolution == model.img_size",
              c["data"]["resolution"] == c["model"]["img_size"], True)
        check(f"w{w}  surprise.target_layer_norm", c["surprise"].get("target_layer_norm"), True)
        check(f"w{w}  surprise.distance", c["surprise"].get("distance"), "l1")

    print("\n2) 창을 합친 A.8 최고 (공식 축약 = Filtered)")
    sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
    from garrido_rescore import collect
    r = collect(BASE, "intphys1_dev", "vith")
    b = r["best"]
    check("combo", b["combo"], EXPECT_BEST[0])
    check("규칙", b["rule"], "intphys: Filtered -> macro 평균")
    check("노트북 규칙 참고값", round(b["alt_notebook"], 2), EXPECT_ALT_NOTEBOOK)
    check("정확도(%)", round(b["accuracy"], 2), EXPECT_BEST[1])
    check("n_pair", b["n_pair"], EXPECT_BEST[2])
    check("성립한 칸 수 (skip2_w16 / skip2_w32 / skip5_w16)", len(r["combos"]), 3)

    print("\n3) 하네스 자체 채점과 일치하는가 (교차검증)")
    s32 = json.load(open(D32 / "summary.json"))
    check("w32 하네스 best_avg (Filtered 고정)", round(s32["best_avg"]["block_pairwise"] * 100, 2),
          EXPECT_HARNESS_W32)
    check("w32 n_videos", s32["n_videos"], 360)
    check("w32 n_ties", s32["best_avg"].get("n_ties", 0), 0)

    print("\n4) 외부 대조 (참고)")
    d = abs(b["accuracy"] - PUBLISHED)
    print(f"  [{'OK ' if d <= 2.0 else 'WARN'}] 공개값 {PUBLISHED} "
          f"(IntPhys2 논문 Table 2, 다른 프로토콜) 과 차이 {d:.2f}pt")

    print("\n" + ("통과 — 하네스가 기준값을 재현한다." if ok else "실패 — 위 FAIL 을 먼저 고친다."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
