"""VOBench 끝 가림 (late) 채점 — 블록 AR 의 미래 8 블록을 8 걸음 / 2 걸음 단위 / 1 걸음 으로 만들 때 (2026-10-27, CPU).
사용자: "2 step 씩 roll out 하면 최종적으로 predictor 가 만든 8 block (나눠서 예측했지만) 으로 VOBench 성능 어떻게 나와?" → "late 만".
캐시: /workspace/auto_research_cache/vobench_late/h.npy (216 블록 × 4 팔 [pos_a, pos_b, imp_ab, imp_ba] × 16 블록 × 256 × 1280, 블록별 LN(target_encoder)) · meta.json.
  채점 간격 (30 fps 3 장마다, 블록 0.2 s). 쌍 = (pos_a, imp_ab) · (pos_b, imp_ba) — 문맥 블록 0–7 이 같다. 432 쌍.
미래 8 블록 (8–15) 을 만드는 방식 (모델 natural_ar_vw_state latest, 상태 되먹임):
  8step : 실제 0–7 → 8 … 15 굴림 (지금 채점과 같다; 가능 · 불가능이 같은 예측을 쓴다)
  2step : 실제 0–7 → 8 · 9 / 실제 0–9 → 10 · 11 / 0–11 → 12 · 13 / 0–13 → 14 · 15 — 영상마다 자기 실제 블록으로
  1step : 블록 b 마다 실제 0 … b−1 → b (teacher forcing)
  ⚠️ 2step · 1step 은 블록 10 부터 그 영상의 경계 뒤 실제 블록이 문맥이 된다 → '국소 예측 오차' 를 잰다 (sliding 채점과 비슷).
surprise = 미래 블록 8–15 의 토큰 L1 평균. 맞음 = 불가능 surprise > 가능 surprise. 표: 전체 · 사물 (primitive = P / realistic = R) · 움직임 / 정지 · 위반 종류.
  for k in 0 1 2 3; do /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/vobench_late_rollout_schemes.py --shard $k --nshard 4 & done; wait
  /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/vobench_late_rollout_schemes.py --merge --nshard 4   → z_analysis/results/vobench_late_rollout_schemes.json
"""
import argparse, collections, json, os, sys
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--merge", action="store_true"); ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--fix8", action="store_true", help="2026-10-27: 조각에 저장된 8step 값만 쌍별 문맥으로 다시 계산 (GPU 가능)"); a = ap.parse_args()
torch.set_num_threads(a.threads)
REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT
sys.argv = _argv
CA = "/workspace/auto_research_cache/vobench_late"; S, D, NB, C = 256, 1280, 16, 8
H = np.load(f"{CA}/h.npy", mmap_mode="r"); M = json.load(open(f"{CA}/meta.json"))
if a.fix8:                                                                     # 2026-10-27: 8step 정정판 — 쌍마다 자기 문맥에서 굴려 조각 행의 8step 만 바꾼다
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 1.5)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
    ar8 = FT.load_pred("natural_ar_vw_state", "latest.pt").float().eval().to(dev); ix = torch.arange(C * S, device=dev).unsqueeze(0)
    OUTD = "/workspace/auto_research_cache/vobench_late_schemes"
    for k in range(a.nshard):
        rows_k = json.load(open(f"{OUTD}/shard{k}.json")); mine = [m for j, m in enumerate(M) if j % a.nshard == k]; assert len(rows_k) == 2 * len(mine)
        for n, m in enumerate(mine):
            for q, (pi, ii) in enumerate(((0, 2), (1, 3))):
                hp = torch.from_numpy(H[m["i"], pi].astype(np.float32)).to(dev); hi = torch.from_numpy(H[m["i"], ii].astype(np.float32)).to(dev)
                with torch.no_grad():
                    p8 = ar8.rollout_kv(hp[:C].reshape(1, C * S, D), ix, NB - C)[0].view(NB - C, S, D)
                r = rows_k[2 * n + q]; assert r["objset"] == m["objset"] and r["violation"] == m["violation"]
                r["8step"] = bool(float((p8 - hi[C:]).abs().mean()) > float((p8 - hp[C:]).abs().mean()))
        json.dump(rows_k, open(f"{OUTD}/shard{k}.json", "w")); print("fixed shard", k, flush=True)
    a.merge = True
ar = FT.load_pred("natural_ar_vw_state", "latest.pt").float().eval(); T1 = ar.type_embed[1]; idx_all = torch.arange(NB * S).unsqueeze(0)


def futures(h):
    """h (16, 256, D) 실제 블록 → {'1step', '2step'} 각 (8, 256, D) = 블록 8–15 예측 (그 영상의 실제 블록을 문맥으로)."""
    one, two = {}, {}
    with torch.no_grad():
        caches = None
        for j in range(NB - 1):
            p1, caches, stt = ar._ar_step_kv(h[j][None], idx_all[:, j * S:(j + 1) * S], caches)
            if j >= C - 1:
                one[j + 1] = p1[0]
            if j >= C - 1 and (j - (C - 1)) % 2 == 0 and j + 2 < NB:
                p2, _, _ = ar._ar_step_kv(stt + T1, idx_all[:, (j + 1) * S:(j + 2) * S], caches, embedded=True)
                two[j + 1] = p1[0]; two[j + 2] = p2[0]
    return {"1step": torch.stack([one[b] for b in range(C, NB)]), "2step": torch.stack([two[b] for b in range(C, NB)])}


rows = []
OUTD = "/workspace/auto_research_cache/vobench_late_schemes"; os.makedirs(OUTD, exist_ok=True)
for n, m in enumerate(M if a.merge else [m for k, m in enumerate(M) if k % a.nshard == a.shard]):
    if a.merge:
        break
    hs = [torch.from_numpy(H[m["i"], k].astype(np.float32)) for k in range(4)]
    with torch.no_grad():                                                     # 2026-10-27 정정: 쌍마다 자기 문맥 (pos_a 또는 pos_b) 에서 굴린다 — 처음 판은 pos_a 문맥 하나를 두 쌍에 썼다
        p8s = {pi: ar.rollout_kv(hs[pi][:C].reshape(1, C * S, D), idx_all[:, :C * S], NB - C)[0].view(NB - C, S, D) for pi in (0, 1)}
    fut = [futures(h) for h in hs]
    for pi, ii in ((0, 2), (1, 3)):
        sur = lambda pred, h: float((pred - h[C:]).abs().mean())
        r = {"objset": m["objset"], "violation": m["violation"], "motion": "static" if m["motion"] == "flat_static" else "moving"}
        r["8step"] = sur(p8s[pi], hs[ii]) > sur(p8s[pi], hs[pi])            # 문맥이 같아 예측 하나를 두 영상에 비교
        for sch in ("2step", "1step"):
            r[sch] = sur(fut[ii][sch], hs[ii]) > sur(fut[pi][sch], hs[pi])  # 영상마다 자기 실제 블록으로 만든 예측
        rows.append(r)
    if (n + 1) % 40 == 0:
        print(f"  {n + 1}/{len(M)}", flush=True)
if not a.merge:
    json.dump(rows, open(f"{OUTD}/shard{a.shard}.json", "w")); print("shard saved", len(rows)); sys.exit(0)
rows = [r for k in range(a.nshard) for r in json.load(open(f"{OUTD}/shard{k}.json"))]
R = {"n_pairs": len(rows)}
for sch in ("8step", "2step", "1step"):
    e = {"all": round(100 * np.mean([r[sch] for r in rows]), 1)}
    for key in ("objset", "motion", "violation"):
        for v in sorted({r[key] for r in rows}):
            e[f"{key}={v}"] = round(100 * np.mean([r[sch] for r in rows if r[key] == v]), 1)
    R[sch] = e; print(sch, e, flush=True)
json.dump(R, open(f"{REPO}/z_analysis/results/vobench_late_rollout_schemes.json", "w"), indent=1)
print("saved")
