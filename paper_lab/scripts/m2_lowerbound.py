#!/usr/bin/env python3
"""M2 (옮기는 연산을 구조에) 의 학습 없는 하한 — 진행자 2, 2026-10-22 (PLAN v4 §11). CPU, 읽기만.

질문: 마지막으로 보인 물체 토큰 (3x3) 을 '문맥에서 외삽한 궤적' 위에 붙이는 연산만으로 (학습 없음),
      끝 가림 움직이는 물체의 물체 자리 판정 · 자리 여유 · 먼 칸 자리 여유에서 기준 블록 AR 을 넘는가.
렌더 (이미 계산된 칸별 토큰 오차 지도, 같은 표적 = 블록별 LN 표적 encoder 토큰):
  z_research/VOBench_v1/headroom/vobench_r*.npz (design_headroom.py, 2026-10-13): copy · move_or (참 궤적) · move_cv (마지막 두 '보인'
  샘플로 등속 외삽) · move_ca (보인 샘플 2 차 맞춤). ⚠️ 문맥의 라벨 위치를 쓴다 = 지각이 완벽하다는 가정 (관측만 판이 아니다 —
  관측만 판은 물체 추적까지 관측에서 해야 한다. 이 하한이 지면 관측만 판도 진다).
  paper_lab/eval/results/<모델>/vobench_r0.npz (진행자 1 하네스): 모델 rollout · 복사 · 문맥 평균 · 하네스 오라클.
채점 = R2 라운드 2 정의 (물체 3x3 칸 합집합, 두 arm, 8 미래 블록) + 먼 칸 판 (가능한 미래 물체가 마지막으로 보인 문맥 자리에서 48 px
(512 라벨) 넘게 떨어진 미래 블록만, 진행자 1 `--far 48` 과 같은 정의).
  python paper_lab/scripts/m2_lowerbound.py
"""
import csv, glob, json, os
from collections import defaultdict
import numpy as np

R = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/"
RES = R + "paper_lab/eval/results/"
HR = R + "z_research/VOBench_v1/headroom/"
FR = "/data2/local_datasets/world/Benchmark/VOBench_v1/"
FAR = 48.0
rows = {r["video_id"]: r for r in csv.DictReader(open(R + "data_csv/vobench_v1/index.csv"))}
blk = defaultdict(dict)
for v, r in rows.items():
    blk[r["source_block"]][r["variant"]] = v


def track(L):
    if not L or L[0] is None:
        return None, None
    P = np.array([[L[3 * s]["u"], L[3 * s]["v"]] for s in range(32)], float)
    hid = np.array([L[3 * s]["hidden"] for s in range(32)], float)
    return P, hid


def masks(lab, ctx_arm):
    """(전체 마스크, 먼 칸 마스크) (8, 16, 16) — 두 arm 의 물체 3x3 합집합. 먼 칸 = 문맥 arm 의 미래 물체 중심이
    마지막으로 보인 문맥 자리에서 FAR px 넘게 떨어진 미래 블록."""
    M = np.zeros((8, 16, 16), bool)
    for a in ("A", "B"):
        P, _ = track(lab.get(a))
        if P is None:
            continue
        fut = P[16:].reshape(8, 2, 2).mean(1)
        for k in range(8):
            u, v = fut[k]
            if 0 <= u < 512 and 0 <= v < 512:
                y, x = int(v // 32), int(u // 32)
                M[k, max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = True
    far = np.zeros(8, bool)
    P, hid = track(lab.get(ctx_arm))
    if P is not None:
        vis = [s for s in range(16) if hid[s] < 0.5]
        if vis:
            last = P[vis[-1]]
            fut = P[16:].reshape(8, 2, 2).mean(1)
            far = np.linalg.norm(fut - last[None], axis=1) > FAR
    Mf = M.copy()
    Mf[~far] = False
    return M, Mf


def pairs(occ="late", moving=True):
    out = []
    for sb, arms in blk.items():
        if not all(a in arms for a in ("pos_a", "pos_b", "imp_ab", "imp_ba")):
            continue
        r = rows[arms["pos_a"]]
        if r["occlusion"] != occ or ((r["motion"] != "flat_static") != moving):
            continue
        lab = json.load(open(FR + os.path.dirname(r["file_name"]) + "/label.json"))["labels"]
        for pos, imp, d in (("pos_a", "imp_ab", "A"), ("pos_b", "imp_ba", "B")):
            M, Mf = masks(lab, d)
            if M.any():
                out.append((sb, d, r["violation"], arms[pos], arms[imp], M, Mf))
    return out


def load_headroom():
    E, idx = defaultdict(list), []
    for f in sorted(glob.glob(HR + "vobench_r*.npz")):
        z = np.load(f, allow_pickle=True)
        idx += list(z["video_id"])
        for k in z.files:
            if k.startswith("E_"):
                E[k].append(z[k])
    return {v: i for i, v in enumerate(idx)}, {k: np.concatenate(v) for k, v in E.items()}


def load_harness(name):
    z = np.load(RES + name + "/vobench_r0.npz", allow_pickle=True)
    return {v: i for i, v in enumerate(z["video_id"])}, {k: z[k] for k in z.files if k.startswith("E_")}


def margin(src, render, P, far=False):
    idx, E = src
    out = []
    for sb, d, viol, vp, vi, M, Mf in P:
        mk = Mf if far else M
        if not mk.any():
            out.append(np.nan); continue
        ep = E["E_" + render][idx[vp]].astype(np.float32)[mk].mean()
        ei = E["E_" + render][idx[vi]].astype(np.float32)[mk].mean()
        out.append(ei - ep)
    return np.array(out)


def acc(m):
    m = m[~np.isnan(m)]
    return 100 * np.where(m > 0, 1.0, np.where(m == 0, 0.5, 0.0)).mean()


def boot_diff(d, P, B=2000, seed=0):
    ok = ~np.isnan(d)
    sbs = sorted({p[0] for p, o in zip(P, ok) if o}); pos = defaultdict(list)
    for i, (p, o) in enumerate(zip(P, ok)):
        if o:
            pos[p[0]].append(i)
    groups = [np.array(pos[s]) for s in sbs]
    rng = np.random.default_rng(seed); out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        out[b] = np.concatenate([d[groups[j]] for j in pick]).mean()
    return d[ok].mean(), np.percentile(out, 2.5), np.percentile(out, 97.5), int(ok.sum())


def main():
    P = pairs()
    hr = load_headroom()
    base = load_harness("vast70")
    # 같은 표적인지 확인: 두 파일의 복사 오차 지도가 같아야 한다
    common = [p[3] for p in P if p[3] in hr[0] and p[3] in base[0]][:50]
    dmax = max(float(np.abs(hr[1]["E_copy"][hr[0][v]].astype(np.float32) - base[1]["E_copy"][base[0][v]].astype(np.float32)).max()) for v in common)
    print(f"pairs {len(P)} (far blocks present in {sum(p[6].any() for p in P)}) · target check: max |E_copy(headroom) - E_copy(harness)| = {dmax:.4f}")
    res = {"n_pairs": len(P), "target_check_max_abs": dmax, "rows": {}}
    for far in (False, True):
        tag = "far48" if far else "all"
        mo = margin(base, "oracle", P, far)
        scale = np.nanmean(np.abs(mo))
        mb = margin(base, "model", P, far)
        print(f"\n== {tag}: oracle scale {scale:.4f}")
        for lab, src, rnd in (("copy (harness)", base, "copy"), ("ctx-mean8 (harness)", base, "mean8"), ("oracle (harness)", base, "oracle"),
                              ("base block AR (vast70)", base, "model"), ("move_or (true track, headroom)", hr, "move_or"),
                              ("move_cv (const-vel extrapolation)", hr, "move_cv"), ("move_ca (quadratic fit)", hr, "move_ca"),
                              ("copy (headroom)", hr, "copy")):
            m = margin(src, rnd, P, far)
            d = (m - mb) / scale
            mean, lo, hi, n = boot_diff(d, P)
            share = np.nanmean(m) / scale
            print(f"  {lab:36s} acc {acc(m):5.1f}  margin share {100 * share:+6.1f} %   vs base AR {100 * mean:+6.1f} % [{100 * lo:+6.1f}, {100 * hi:+6.1f}]  n {n}")
            res["rows"][f"{tag}|{lab}"] = dict(acc=acc(m), margin_share=float(share), diff_vs_base=[float(mean), float(lo), float(hi)], n=n)
    os.makedirs(R + "paper_lab/_scratch/m2", exist_ok=True)
    json.dump(res, open(R + "paper_lab/_scratch/m2/lowerbound_headroom.json", "w"), indent=1)


if __name__ == "__main__":
    main()
