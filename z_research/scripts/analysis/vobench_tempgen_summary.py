"""vobench_tempgen.py 결과 정리 (2026-10-10) — 시드마다 8×8 행렬 (학습 시점 × 평가 시점) 을 모아 RESULTS.md · summary.json.

  지표 (물체 샘플 정답률 %, 시드 평균 [최소, 최대]):
    대각 = 같은 시점 학습 · 평가 / 앞으로 = 학습 < 평가 (멀리 4 칸 이상 따로) / 뒤로 = 학습 > 평가
    t8 head = t8 에서만 학습한 head 를 t8 … t15 에 / late = p head (t8 · t15 학습) 를 late 의 p (물체, 섞임 제외) 에
  행렬 4 종: p→p · h→h · h→p · p→h (학습 표현 → 평가 표현).
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

T = list(range(8, 16))
PAIRS = [("p", "p"), ("h", "h"), ("h", "p"), ("p", "h")]


def mat(R, tr, ev, key="acc"):
    return np.array([[R[f"{tr}@t{a}"][ev][str(b)][key] for b in T] for a in T])


def stats(A):
    n = len(T)
    sel = lambda f: float(np.mean([A[i, j] for i in range(n) for j in range(n) if f(i, j)]))
    return dict(diag=float(np.diag(A).mean()), fwd=sel(lambda i, j: j > i), fwd_far=sel(lambda i, j: j - i >= 4),
                bwd=sel(lambda i, j: j < i), bwd_far=sel(lambda i, j: i - j >= 4))


def agg(vals):
    v = np.array(vals, float); return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()), n=len(v))


def main(out: Path):
    runs = defaultdict(list)
    for f in sorted((out / "runs").glob("tempgen_*_s*.json")):
        j = json.load(open(f)); runs[(j["objset"], j["label"])].append(j)
    S = {}; md = ["# VOBench_v1 temporal generalization — V-JEPA 2 ViT-H (릴리즈)", "",
                  "시점 하나 (t_train) 에서만 AttentiveClassifier (depth 1 · query 1 · head 16, 전체 토큰 256) 를 학습하고 미래 시점 t8–t15 전부에 건다. "
                  "학습 = train block · 문맥 끝 보임 조건 (visible · early · mid) 의 물체 + 빈 장면, 그 시점 '보임' 튜블릿. 평가 = test block 물체 샘플 정답률 %. "
                  "정의 · 재현은 `z_research/scripts/analysis/vobench_tempgen.py` 머리말. 칸 = 시드 평균 [최소, 최대].", ""]
    c = lambda d: f"{d['mean']:.1f} [{d['min']:.0f}, {d['max']:.0f}]"
    for label in ("shape", "color", "joint"):
        md += [f"## 라벨 = {label}", ""]
        for o in ("primitive", "realistic"):
            js = runs.get((o, label), [])
            if not js:
                continue
            seeds = sorted(j["seed"] for j in js); n_cls = len(js[0]["classes"]) - 1
            key = f"{o}/{label}"; S[key] = {"seeds": seeds, "chance_object": 100 / n_cls, "n_train": {}, "pairs": {}, "t8_head": {}, "late": {}}
            S[key]["n_train"] = {k: js[0]["res"][k]["n_train"] for k in js[0]["res"]}
            md += [f"**{o}** (시드 {seeds}, 물체 chance {100 / n_cls:.1f} %)", "",
                   "| 학습 → 평가 | 같은 시점 | 앞으로 (학습 < 평가) | 앞으로 4 칸 이상 | 뒤로 (학습 > 평가) | 뒤로 4 칸 이상 |", "|---|---:|---:|---:|---:|---:|"]
            for tr, ev in PAIRS:
                st = [stats(mat(j["res"], tr, ev)) for j in js]
                d = {k: agg([s[k] for s in st]) for k in st[0]}
                S[key]["pairs"][f"{tr}->{ev}"] = dict(d, matrix_mean=np.mean([mat(j["res"], tr, ev) for j in js], 0).round(2).tolist())
                md.append(f"| {tr} → {ev} | {c(d['diag'])} | {c(d['fwd'])} | {c(d['fwd_far'])} | {c(d['bwd'])} | {c(d['bwd_far'])} |")
            md += ["", "t8 에서만 학습한 head → 평가 시점 t8 … t15 (정답 %, 괄호 = none 으로 답한 %, 시드 평균):", "",
                   "| head | 평가 | " + " | ".join(f"t{t}" for t in T) + " |", "|---|---|" + "---:|" * len(T)]
            for tr in ("p", "h"):
                for ev in ("p", "h", "late"):
                    A = np.array([[j["res"][f"{tr}@t8"][ev][str(t)]["acc"] for t in T] for j in js])
                    N = np.array([[j["res"][f"{tr}@t8"][ev][str(t)]["none"] for t in T] for j in js])
                    S[key]["t8_head"][f"{tr}->{ev}"] = dict(acc=A.mean(0).round(2).tolist(), acc_min=A.min(0).tolist(), acc_max=A.max(0).tolist(), none=N.mean(0).round(2).tolist())
                    md.append(f"| {tr}@t8 | {ev if ev != 'late' else 'p late'} | " + " | ".join(f"{a:.0f} ({b:.0f})" for a, b in zip(A.mean(0), N.mean(0))) + " |")
            for tt in (8, 15):
                v = [np.mean([j["res"][f"p@t{tt}"]["late"][str(t)]["acc"] for t in T]) for j in js]
                S[key]["late"][f"p@t{tt}"] = agg(v)
            md += ["", f"late (문맥 끝 가림) 의 p, 미래 전체 평균: p@t8 head {c(S[key]['late']['p@t8'])} · p@t15 head {c(S[key]['late']['p@t15'])} (chance {100 / n_cls:.1f})", ""]
    md += ["## 단서", "",
           "- 시점 하나 · train 반이라 학습 샘플이 objset 마다 약 270 개 — joint (48 + none · 37 + none) 는 클래스당 약 5 개라 불안정하다. shape · color 를 주 근거로 쓴다",
           "- '앞으로 / 뒤로' 비대칭은 h → h 대조와 같이 읽는다 (물체가 움직여도 h 는 평평 — 위치 특이 query 탓이 아니다)",
           "- 문맥 끝 보임 조건만 학습 · 평가 (late 는 따로). 위치가 아니라 정체만 잰다",
           "- **다수 클래스 기준선 (late test 물체 샘플, 2026-10-10 index 에서 셈)**: primitive shape 20.7 · color 17.4 · joint 5.8 / "
           "realistic shape 19.6 · **color 78.4 (`native`)** · joint 14.8. **realistic color 는 물체의 79 % 가 `native` 라 정확도가 기준선에 묻힌다** — realistic 색 결과는 전부 이 기준선과 같이 읽는다",
           "- late: t8 head 는 chance 인데 **t15 head 는 shape 를 다수 기준선의 약 2 배 (primitive 41 · realistic 38 vs 20) 로 읽는다** (realistic joint 34 vs 15). "
           "late p 에 먼 미래 p 와 같은 약한 형식의 모양 정보가 남아 있을 수 있다 — 1 차, late p 로 직접 학습한 held-out 상한으로 확인 전",
           "- 시드 = head 초기화 · 순서만 (block 분할은 고정). 칸별 시드 범위는 대부분 ±2–5", "",
           "## 재현", "", "```bash", "P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python",
           "$P z_research/scripts/analysis/vobench_tubelet_probe.py extract --gpus 4      # 캐시 (h 16 튜블릿 · p 8)",
           "for s in 0 1 2; do for o in primitive realistic; do for l in joint shape color; do",
           "  $P z_research/scripts/analysis/vobench_tempgen.py run $o $l <gpu> $s; done; done; done",
           "$P z_research/scripts/analysis/vobench_tempgen.py summarize", "```"]
    (out / "summary.json").write_text(json.dumps(S, indent=1))
    (out / "RESULTS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
