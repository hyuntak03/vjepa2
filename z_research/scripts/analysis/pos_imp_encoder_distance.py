"""가능 vs 불가능 영상 사이 encoder L1 거리 — 32 장 한 번에, 미래 부분 (17–32 장 = 블록 8–15) (2026-10-27, GPU 1 장).
사용자: "possible, impossible 이 2 개의 L1 distance 만 재 달라 이거야".
쌍 = 같은 (block_id, pair_id) 의 가능 1 · 불가능 1 (문맥 장이 같다). encoder 두 개 (context `encoder` · target `target_encoder`), 토큰마다 affine 없는 LN.
  거리 = 같은 자리 토큰끼리 |T_가능 − T_불가능| 의 평균 (칸 256 · 차원 1280), 블록 (시간 칸) 마다 따로 → 미래 = 블록 8–15 평균, 문맥 = 블록 0–7 평균 (확인용).
참고 크기: (i) VOBench 가능–가능 = pos_a vs pos_b (같은 블록, 물체만 다름) (ii) 가능 영상의 자기 변화 = 미래 블록 vs 자기 마지막 문맥 블록 (encoder_change_surprise 의 copy 점수와 같은 정의).
데이터: IntPhys1 late (180 쌍) · IntPhys2 위반 정렬 (500 쌍) · VOBench 전체 (1,440 쌍). 읽기 · 전처리는 encoder_change_surprise.py 와 같다.
  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/pos_imp_encoder_distance.py
출력: z_analysis/results/pos_imp_encoder_distance.json
"""
import collections, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, os.path.join(REPO, "z_research/scripts/analysis")); sys.path.insert(0, REPO)
sys.argv = [sys.argv[0], "--sets", ""]                                   # encoder_change_surprise 의 모델 · 읽기 함수만 쓴다 (본체 루프는 sets 가 비어 안 돈다)
import importlib.util
spec = importlib.util.spec_from_file_location("ecs", os.path.join(REPO, "z_research/scripts/analysis/encoder_change_surprise.py"))
src = open(spec.origin).read().split("\nR = {}\n")[0]                    # 본체 앞까지만 실행 (모델 · rows_of · load)
ns = {"__name__": "ecs"}; exec(compile(src, spec.origin, "exec"), ns)
ENC, rows_of, load, D, S = ns["ENC"], ns["rows_of"], ns["load"], ns["D"], ns["S"]


def tokens(r):
    xn = load(r["_src"]); out = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for en, enc in ENC.items():
            out[en] = F.layer_norm(enc(xn.half()).float(), (D,))[0].view(16, S, D)
    return out


R = {}
for name in ("intphys1_late", "intphys2_uni", "vobench_all"):
    rows = rows_of(name); groups = collections.defaultdict(dict)
    for r in rows:
        groups[(r["block_id"], r["pair_id"])]["imp" if r["plausible"] == "0" else "pos"] = r
    pairs = [g for g in groups.values() if "pos" in g and "imp" in g]
    vob_pos = collections.defaultdict(dict)
    if name == "vobench_all":
        for r in rows:
            if r["variant"] in ("pos_a", "pos_b"):
                vob_pos[r["block_id"]][r["variant"]] = r
    rec = collections.defaultdict(list); t0 = time.time(); cache = {}
    for n, p in enumerate(pairs):
        tp, ti = tokens(p["pos"]), tokens(p["imp"])
        for en in ENC:
            d = (tp[en] - ti[en]).abs().mean((1, 2)).cpu().numpy()                    # 블록 16 개
            rec[f"{en}_pos_imp_by_block"].append(d); rec[f"{en}_self_change"].append(float((tp[en][8:] - tp[en][7:8]).abs().mean()))
        rec["group"].append(p["pos"]["_grp"])
        if name == "vobench_all" and p["pos"]["variant"] == "pos_a":                  # 가능–가능 (pos_a vs pos_b), 블록마다 한 번
            ob = vob_pos[p["pos"]["block_id"]].get("pos_b")
            if ob is not None:
                tb = tokens(ob)
                for en in ENC:
                    rec[f"{en}_pos_pos_by_block"].append((tp[en] - tb[en]).abs().mean((1, 2)).cpu().numpy())
        if n % 300 == 0:
            print(f"  {name} {n}/{len(pairs)} · {(time.time() - t0) / (n + 1):.2f} s/쌍", flush=True)
    res = {"n_pairs": len(pairs)}
    for en in ENC:
        A = np.stack(rec[f"{en}_pos_imp_by_block"])
        e = {"pos_vs_imp_future_mean": round(float(A[:, 8:].mean()), 5), "pos_vs_imp_future_median_over_pairs": round(float(np.median(A[:, 8:].mean(1))), 5),
             "pos_vs_imp_context_mean": round(float(A[:, :8].mean()), 5), "pos_vs_imp_by_block": [round(float(v), 5) for v in A.mean(0)],
             "possible_self_change_future_vs_last_context": round(float(np.mean(rec[f"{en}_self_change"])), 5)}
        if rec.get(f"{en}_pos_pos_by_block"):
            B = np.stack(rec[f"{en}_pos_pos_by_block"]); e["pos_a_vs_pos_b_future_mean"] = round(float(B[:, 8:].mean()), 5); e["pos_a_vs_pos_b_context_mean"] = round(float(B[:, :8].mean()), 5)
        g = np.array(rec["group"]); e["pos_vs_imp_future_by_group"] = {k: round(float(A[g == k, 8:].mean()), 5) for k in sorted(set(g))}
        res[en] = e
    R[name] = res
    print(name, {en: {k: v for k, v in res[en].items() if not k.endswith(("by_block", "by_group"))} for en in ENC}, flush=True)
json.dump(R, open(f"{REPO}/z_analysis/results/pos_imp_encoder_distance.json", "w"), indent=1)
print("saved")
