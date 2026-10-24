"""[비판] 블록 AR 은 실제 영상에서 "연속성의 PMI" 를 갖나 (학습 없음; 리더 KST 18:00 배정 + 추가 지시, 사전 등록 진행판 KST 18:02 · 18:0x).

창 = 블록 10 개 (K710 보류 · SSv2, cache_natural 로 새로 인코딩). 예측 = TF 한 걸음 (문맥 4 블록 → 다음 블록), surprise = mean |p − 후보| (칸 · 차원 평균).
A. 같은 미래 · 다른 문맥 (원 등록): y=8 ← 자기 4–7 / 같은 클립 0–3 (간격 4); y=9 ← 자기 5–8 / 같은 클립 1–4 (4) · 0–3 (5); 대조 = 다른 클립 같은 자리 문맥.
   복사 기준 = 문맥 마지막 블록을 예측으로.
B. 같은 문맥 · 다른 미래 후보 (가까운 음성): t = 3–6, 문맥 t−3..t, 참 h_t+1 vs h_t · h_t−1 · h_t+2 · h_t+3 · 먼 시각 (|k−(t+1)| ≥ 4 평균) · 다른 클립 h_t+1.
   대조 판정자 "복사 연속" = h_t 와의 L1 (h_t 음성 칸은 정의 안 됨).
C. 8 후보 순위: t+1 = 4–9, 참 + 같은 클립 다른 블록 7 개 (나머지 9 개 중 무작위). 판정자: AR (p 에 가장 가까운) · 중심성 (후보 평균에 가장 가까운) ·
   복사 연속 (h_t 에 가장 가까운, h_t 자신 제외). 주 규칙 = 모든 자리, 미리 정한 부분집합 = 참 자리 8 · 9.
보고는 창 단위 bootstrap 95 %. 세트별 · 카메라 정지 창 (블록 4–9 화면 칸 흐름 중앙값 크기 모두 < 0.25 칸) 따로.
"""
import os, sys, json, argparse, random, collections, numpy as np, torch

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT  # noqa: E402
sys.argv = _argv

MODELS = {"base": ("natural_ar_vw_state", "latest.pt"), "vob": ("natural_ar_vw_state_vob", "latest.pt"),
          "syn": ("natural_ar_vw_state_phys3d_only", "latest.pt"), "syn_e10": ("natural_ar_vw_state_phys3d_only", "e10.pt")}
S, D, NB = 256, 1280, 10
A_DESIGN = [(8, 4, [(0, 4)]), (9, 5, [(1, 4), (0, 5)])]          # (y, 자기 문맥 시작, [(같은 클립 문맥 시작, 간격)])
B_T = [3, 4, 5, 6]
C_T = [3, 4, 5, 6, 7, 8]                                          # 참 = t+1 = 4..9


def boot(x, rng, n=4000):
    x = np.asarray(x, float); m = len(x)
    bs = np.sort([x[rng.integers(0, m, m)].mean() for _ in range(n)])
    return [float(x.mean()), float(bs[int(0.025 * n)]), float(bs[int(0.975 * n) - 1])]


def l1(a, b):
    return float((a - b).abs().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/auto_research_cache/pmi_cont")
    ap.add_argument("--sets", default="k710_320_holdout,ssv2")
    ap.add_argument("--out", default=os.path.join(REPO, "auto_research/results/crit_pmi_cont.json"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(8)
    dev = torch.device("cuda")
    nets = {k: FT.load_pred(r, c).to(dev).eval() for k, (r, c) in MODELS.items()}
    idx = torch.arange(4 * S, device=dev).unsqueeze(0)
    res, allrows = {}, {}
    for st in a.sets.split(","):
        H = np.load(os.path.join(a.cache, f"{st}_h.npy"), mmap_mode="r"); U = np.load(os.path.join(a.cache, f"{st}_U.npy"))
        N = H.shape[0] if not a.limit else a.limit
        rng = random.Random(0); perm = list(range(N))
        while any(p == i for i, p in enumerate(perm)):
            rng.shuffle(perm)
        med = np.median(U[:N, 4:10].astype(np.float32).reshape(N, 6, -1, 2), axis=2)
        still = (np.linalg.norm(med, axis=-1).max(1) < 0.25)
        crng = random.Random(1)
        rows = []
        for i in range(N):
            h = torch.from_numpy(np.asarray(H[i], dtype=np.float32)).to(dev)
            hj = torch.from_numpy(np.asarray(H[perm[i]], dtype=np.float32)).to(dev)
            ctx = [h[c:c + 4] for c in range(0, 6)] + [hj[4:8], hj[5:9]]      # 0..5 자기 클립 시작, 6 = 다른 클립 4–7, 7 = 다른 클립 5–8
            X = torch.stack(ctx).reshape(len(ctx), 4 * S, D)
            r = {"i": i, "still": bool(still[i])}
            # C 의 후보 묶음 (모델과 무관하게 한 번 정함)
            cands = {}
            for t in C_T:
                others = [k for k in range(NB) if k != t + 1]
                cands[t] = sorted([t + 1] + crng.sample(others, 7))
            preds = {}
            with torch.no_grad():
                for mk, net in nets.items():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        preds[mk] = net.forward_seq(X, idx.expand(len(ctx), -1)).float()[:, -S:]   # 문맥 시작 c → 블록 c+4 예측
                preds["copy"] = X.view(len(ctx), 4, S, D)[:, -1]
                for mk, P in preds.items():
                    # A
                    for (y, cs, others) in A_DESIGN:
                        r[f"A|{mk}|{y}|own"] = l1(P[cs], h[y])
                        for (co, g) in others:
                            r[f"A|{mk}|{y}|same_g{g}"] = l1(P[co], h[y])
                        r[f"A|{mk}|{y}|other_clip"] = l1(P[6 if y == 8 else 7], h[y])
                    if mk == "copy":
                        continue
                    # B (문맥 시작 = t−3)
                    for t in B_T:
                        p = P[t - 3]
                        r[f"B|{mk}|{t}|true"] = l1(p, h[t + 1])
                        r[f"B|{mk}|{t}|h_t"] = l1(p, h[t]); r[f"B|{mk}|{t}|h_tm1"] = l1(p, h[t - 1])
                        r[f"B|{mk}|{t}|h_tp2"] = l1(p, h[t + 2]); r[f"B|{mk}|{t}|h_tp3"] = l1(p, h[t + 3])
                        far = [k for k in range(NB) if abs(k - (t + 1)) >= 4]
                        r[f"B|{mk}|{t}|far"] = float(np.mean([l1(p, h[k]) for k in far]))
                        r[f"B|{mk}|{t}|other_clip"] = l1(p, hj[t + 1])
                    # C
                    for t in C_T:
                        p = P[t - 3]
                        d = {k: l1(p, h[k]) for k in cands[t]}
                        pick = min(d, key=d.get)
                        r[f"C|{mk}|{t}|pick"] = int(pick - (t + 1))
                # B 의 문맥 없는 대조: 복사 연속 (h_t 와의 거리)
                for t in B_T:
                    ht = h[t]
                    for nm, k in (("true", t + 1), ("h_tm1", t - 1), ("h_tp2", t + 2), ("h_tp3", t + 3)):
                        r[f"B|copycont|{t}|{nm}"] = l1(ht, h[k])
                    far = [k for k in range(NB) if abs(k - (t + 1)) >= 4 and k != t]
                    r[f"B|copycont|{t}|far"] = float(np.mean([l1(ht, h[k]) for k in far]))
                    r[f"B|copycont|{t}|other_clip"] = l1(ht, hj[t + 1])
                # C 의 문맥 없는 판정자
                for t in C_T:
                    cs_ = cands[t]
                    Cm = torch.stack([h[k] for k in cs_]); mu = Cm.mean(0)
                    dc = {k: l1(h[k], mu) for k in cs_}
                    r[f"C|central|{t}|pick"] = int(min(dc, key=dc.get) - (t + 1))
                    dh = {k: l1(h[k], h[t]) for k in cs_ if k != t}
                    r[f"C|copycont|{t}|pick"] = int(min(dh, key=dh.get) - (t + 1))
            rows.append(r)
            if i % 50 == 0:
                print(st, i, flush=True)
        allrows[st] = rows
        brng = np.random.default_rng(0)
        out = {"n": N, "n_still": int(still.sum())}
        for sub, sel in (("all", np.ones(N, bool)), ("still", still), ("moving", ~still)):
            rs = [rr for rr, s_ in zip(rows, sel) if s_]
            if len(rs) < 5:
                continue
            o = {"n": len(rs)}
            # A
            for mk in list(MODELS) + ["copy"]:
                for neg in ("same", "other_clip"):
                    win, rel = [], []
                    for rr in rs:
                        w_, d_ = [], []
                        for (y, cs, others) in A_DESIGN:
                            own = rr[f"A|{mk}|{y}|own"]
                            negs = [f"A|{mk}|{y}|same_g{g}" for (_, g) in others] if neg == "same" else [f"A|{mk}|{y}|other_clip"]
                            for nk in negs:
                                w_.append(float(own < rr[nk])); d_.append((rr[nk] - own) / own)
                        win.append(np.mean(w_)); rel.append(np.mean(d_))
                    o[f"A|{mk}|{neg}"] = {"win": boot(win, brng), "rel": boot(rel, brng)}
            for mk in MODELS:
                dw = []
                for rr in rs:
                    v = []
                    for (y, cs, others) in A_DESIGN:
                        for (_, g) in others:
                            v.append(float(rr[f"A|{mk}|{y}|own"] < rr[f"A|{mk}|{y}|same_g{g}"]) - float(rr[f"A|copy|{y}|own"] < rr[f"A|copy|{y}|same_g{g}"]))
                    dw.append(np.mean(v))
                o[f"A|{mk}|minus_copy"] = {"dwin": boot(dw, brng)}
            # B
            for mk in list(MODELS) + ["copycont"]:
                for neg in ("h_t", "h_tm1", "h_tp2", "h_tp3", "far", "other_clip"):
                    if mk == "copycont" and neg == "h_t":
                        continue
                    win, rel = [], []
                    for rr in rs:
                        w_ = [float(rr[f"B|{mk}|{t}|true"] < rr[f"B|{mk}|{t}|{neg}"]) for t in B_T]
                        d_ = [(rr[f"B|{mk}|{t}|{neg}"] - rr[f"B|{mk}|{t}|true"]) / rr[f"B|{mk}|{t}|true"] for t in B_T]
                        win.append(np.mean(w_)); rel.append(np.mean(d_))
                    o[f"B|{mk}|{neg}"] = {"win": boot(win, brng), "rel": boot(rel, brng)}
            # C
            for mk in list(MODELS) + ["central", "copycont"]:
                for subset, ts in (("all_pos", C_T), ("end_pos", [7, 8])):
                    acc = [np.mean([float(rr[f"C|{mk}|{t}|pick"] == 0) for t in ts]) for rr in rs]
                    o[f"C|{mk}|{subset}"] = {"acc": boot(acc, brng)}
                cnt = collections.Counter(rr[f"C|{mk}|{t}|pick"] for rr in rs for t in C_T)
                tot = sum(cnt.values())
                o[f"C|{mk}|pick_dist"] = {str(k): round(v / tot, 3) for k, v in sorted(cnt.items())}
                o[f"C|{mk}|by_pos"] = {str(t + 1): float(np.mean([rr[f"C|{mk}|{t}|pick"] == 0 for rr in rs])) for t in C_T}
            for mk in MODELS:
                for crit in ("central", "copycont"):
                    for subset, ts in (("all_pos", C_T), ("end_pos", [7, 8])):
                        dd = [np.mean([float(rr[f"C|{mk}|{t}|pick"] == 0) - float(rr[f"C|{crit}|{t}|pick"] == 0) for t in ts]) for rr in rs]
                        o[f"C|{mk}|minus_{crit}|{subset}"] = {"d": boot(dd, brng)}
            out[sub] = o
        res[st] = out
        json.dump(res, open(a.out, "w"), indent=1)
        json.dump(allrows, open(a.out.replace(".json", "_rows.json"), "w"))
        # 짧은 출력
        print(f"\n== {st}  창 {N} (카메라 정지 {int(still.sum())})", flush=True)
        for sub in ("all", "still"):
            if sub not in out:
                continue
            o = out[sub]
            print(f" [{sub}] n {o['n']}")
            for mk in list(MODELS) + ["copy"]:
                w, w2 = o[f"A|{mk}|same"]["win"], o[f"A|{mk}|other_clip"]["win"]
                extra = f"  모델−복사 {100*o[f'A|{mk}|minus_copy']['dwin'][0]:+5.1f} [{100*o[f'A|{mk}|minus_copy']['dwin'][1]:+5.1f},{100*o[f'A|{mk}|minus_copy']['dwin'][2]:+5.1f}]" if mk != "copy" else ""
                print(f"  A {mk:8s} 같은클립 {100*w[0]:5.1f} [{100*w[1]:5.1f},{100*w[2]:5.1f}] 상대차 {100*o[f'A|{mk}|same']['rel'][0]:+6.2f}%  다른클립 {100*w2[0]:5.1f}{extra}")
            for mk in list(MODELS) + ["copycont"]:
                s = "  B " + f"{mk:8s} " + " ".join(f"{neg} {100*o[f'B|{mk}|{neg}']['win'][0]:5.1f}({100*o[f'B|{mk}|{neg}']['rel'][0]:+5.1f}%)" for neg in ("h_t", "h_tm1", "h_tp2", "h_tp3", "far", "other_clip") if f"B|{mk}|{neg}" in o)
                print(s)
            for mk in list(MODELS) + ["central", "copycont"]:
                c1, c2 = o[f"C|{mk}|all_pos"]["acc"], o[f"C|{mk}|end_pos"]["acc"]
                print(f"  C {mk:8s} 순위 정확도 전체 {100*c1[0]:5.1f} [{100*c1[1]:5.1f},{100*c1[2]:5.1f}] 끝자리 {100*c2[0]:5.1f} [{100*c2[1]:5.1f},{100*c2[2]:5.1f}]  고른 자리 {o[f'C|{mk}|pick_dist']}")


if __name__ == "__main__":
    main()
