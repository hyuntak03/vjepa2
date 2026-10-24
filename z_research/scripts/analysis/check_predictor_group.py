#!/usr/bin/env python
"""model.predictor_group 이 단독 실행과 **비트 단위로 같은지** 검사한다 (2026-10-21).

왜 필요한가 — predictor_group 은 frozen encoder 를 팔끼리 공유해 같은 ViT-H forward 를
한 번만 돌리는 최적화다 (GRASP 세 팔 약 6 h -> 약 2.5 h). **채점 엔진을 건드린 것**이므로
"빨라졌다" 만으로는 못 쓴다. 팔마다 단독 실행과 per-video surprise 가 정확히 같아야 한다.

무엇을 재나 — 작은 부분집합(LIMIT)에서
  (1) 그룹 실행 1 번  -> 팔마다 폴더가 생긴다 (단독과 **같은 이름**이라 옆으로 치워 둔다)
  (2) 팔마다 단독 실행
하고 `per_block.json` 의 `per_video_surprise` 를 (영상 x combo) 전수 비교한다.
`max|Δ| == 0.0` 이어야 통과다 — 허용오차를 두지 않는다. 연산 순서를 안 바꿨으므로 정확히 같아야 한다.

    python z_research/scripts/analysis/check_predictor_group.py \
        --dataset grasp_level2_vll7 --group vith_g1_phys2d --window garrido_w16 --limit 24

⚠️ GPUS=1 로 돈다 — DDP 는 rank 분배가 끼어들어 비교가 흐려진다 (CLAUDE.md §7-2).
⚠️ 통과해도 보장 범위는 **kind=ar · ar_state** 뿐이다 (엔진이 다른 kind 는 거부한다).
"""
import argparse, json, os, shutil, subprocess, sys, yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(REPO, "cvpr/results/eval/intphys1")


def run(dataset, model, window, limit):
    env = {**os.environ, "GPUS": "1", "WINDOWS": window, "LIMIT": str(limit), "FORCE": "1"}
    print(f"\n$ GPUS=1 WINDOWS={window} LIMIT={limit} bash cvpr/eval/intphys1/run.sh {dataset} {model}", flush=True)
    if subprocess.run(["bash", "cvpr/eval/intphys1/run.sh", dataset, model], cwd=REPO, env=env).returncode:
        sys.exit(f"실행 실패: {model}")


def per_video(d):
    f = os.path.join(d, "per_block.json")
    if not os.path.isfile(f):
        sys.exit(f"없다: {f}")
    return json.load(open(f))["per_video_surprise"]


def compare(a_dir, b_dir, label):
    A, B = per_video(a_dir), per_video(b_dir)
    if set(A) != set(B):
        print(f"  ✗ {label}: 영상 집합이 다르다 (그룹 {len(A)} vs 단독 {len(B)})")
        return False
    worst, n = 0.0, 0
    for vid in A:
        if set(A[vid]) != set(B[vid]):
            print(f"  ✗ {label}: combo 키가 다르다 ({vid})")
            return False
        for k in A[vid]:
            worst = max(worst, abs(A[vid][k] - B[vid][k]))
            n += 1
    ok = worst == 0.0
    print(f"  {'✓' if ok else '✗'} {label}: 영상 {len(A)} x combo {n // max(1, len(A))} = {n} 값, max|Δ| = {worst:.3e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="grasp_level2_vll7")
    ap.add_argument("--group", default="vith_g1_phys2d")
    ap.add_argument("--window", default="garrido_w16")
    ap.add_argument("--limit", type=int, default=24)
    a = ap.parse_args()

    reg = yaml.safe_load(open(os.path.join(REPO, "cvpr/registry/models.yaml")))
    arms = [g["name"] for g in reg[a.group]["predictor_group"]]
    print(f"팔 {len(arms)}: {arms}")
    sfx = f"_w{a.window}_smoke{a.limit}"

    run(a.dataset, a.group, a.window, a.limit)
    grp = {}
    for m in arms:                      # 그룹이 쓴 팔 폴더를 치워 둔다 (단독이 같은 이름을 쓴다)
        d = f"{OUT}/{a.dataset}_{m}{sfx}"
        g = d + "__grp"
        shutil.rmtree(g, ignore_errors=True)
        if not os.path.isdir(d):
            sys.exit(f"그룹이 팔 폴더를 안 만들었다: {d}")
        shutil.move(d, g)
        grp[m] = g

    for m in arms:
        run(a.dataset, m, a.window, a.limit)

    print("\n=== 비트 단위 대조 (그룹 vs 단독) ===")
    allok = all(compare(grp[m], f"{OUT}/{a.dataset}_{m}{sfx}", m) for m in arms)
    print("\n" + ("통과 — predictor_group 은 단독 실행과 같은 값을 낸다." if allok
                  else "실패 — predictor_group 을 쓰지 말 것."))
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
