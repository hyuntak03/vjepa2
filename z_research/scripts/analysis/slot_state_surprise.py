"""상태 공간 surprise (2026-11-07, 학습 없음): 예측 latent 없이 맞추기 판 물체 slot 의 주소 질량만으로 판정한다. 외형 거리를 쓰지 않는다.
규칙 (결과를 보기 전에 고정): c = min(1, 문맥 블록 최대 질량), f = min(1, 미래 블록 최대 질량), S = |c − f|. 불가능 S > 가능 S 면 정답, 같으면 0.5.
  물체 slot = 학습 분포에서 dice 로 고른 slot 4. 벤치 라벨은 안 쓴다. 질량 1 = 토큰 한 칸 몫 (v4bc4 보이는 블록 평균 3.84, 가려진 블록 0.19).
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 /venv/vjepa/bin/python z_research/scripts/analysis/slot_state_surprise.py --board real
출력: z_analysis/results/slot_diag_20261107/state_surprise_<board>.json
"""
import argparse, csv, json, os, sys
os.environ.setdefault("WMA_AR_BLOCK_CACHE", "0")
import numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(__file__))
from slot_diag import OUT, RES, ROOT, S, WMADataset, boards, build_from_config, ln

MODEL, KOBJ = "vast_ar_vw_state_v4bc4_slotmatch", 4
B = {"real": ("real world (all 95)", f"{RES}/real_world_permanence_uniform_{MODEL}", None),
     "vob_panel": ("VOBench v3 late · panel", f"{RES}/vobench_v3_{MODEL}", "vobench_v3"),
     "vob_mover": ("VOBench v3 late · moving occluder", f"{RES}/vobench_v3_{MODEL}", "vobench_v3"),
     "sur": ("IntPhys1 Surrogate", f"{RES}/intphys1_surrogate_{MODEL}", "intphys1_surrogate"),
     "ip2": ("IntPhys2 late · permanence (A=pair 1, B=pair 2)", f"{RES}/intphys2_late_final_{MODEL}", None),
     "ip1": ("IntPhys1 late · O1 occluded", f"{ROOT}/cvpr/results/eval/intphys1_late/intphys1_late_uni_{MODEL}", None)}
ap = argparse.ArgumentParser(); ap.add_argument("--board", required=True, choices=list(B)); a = ap.parse_args()
bn, path, mv = B[a.board]
cfg = yaml.safe_load(open(f"{path}/_resolved.yaml"))
blk = boards()[bn][1]
if mv:
    mset = {r["block_id"] for r in csv.DictReader(open(os.path.join(ROOT, f"data_csv/{mv}/index.csv"))) if r["motion"] != "flat_static"}
    blk = {k: v for k, v in blk.items() if k in mset}
held = set(boards()["real world (held-out 48)"][1]) if a.board == "real" else None
dev = torch.device("cuda:0")
bd = build_from_config(cfg["model"], dev)
core = bd.predictor.core
ds = WMADataset(cfg)
pos = {r.video_id: i for i, r in enumerate(ds.records)}
C = int(cfg["surprise"].get("context_length", 16)) // 2


def surprise(vids):
    x = torch.stack([ds.clip(pos[v]) for v in vids]).to(dev, bd.dtype)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        hb = ln(bd.target_encoder(x)).to(torch.float16)
        nb = hb.size(1) // S
        _, att, _ = core._slot_run(hb.view(x.size(0), nb, S, -1))
    m = att[:, :, KOBJ].float().sum(-1)                      # (n, nb) 물체 slot 질량
    c = m[:, :C].max(1).values.clamp_max(1.0)
    f = m[:, C:].max(1).values.clamp_max(1.0)
    return (c - f).abs().cpu().numpy(), c.cpu().numpy(), f.cpu().numpy(), nb


rows = []
for b in sorted(blk):
    vids = [blk[b]["A"][0], blk[b]["A"][1], blk[b]["B"][0], blk[b]["B"][1]]      # pos_A imp_A pos_B imp_B
    s, c, f, nb = surprise(vids)
    win = lambda p, i: 1.0 if s[i] > s[p] else (0.5 if s[i] == s[p] else 0.0)   # noqa: E731
    rows.append((b, win(0, 1), win(2, 3), float(c[0]), float(c[2]), float(f[0]), float(f[1]), float(f[2]), float(f[3])))
A, Bq = 100 * np.mean([r[1] for r in rows]), 100 * np.mean([r[2] for r in rows])
res = {"board": a.board, "n": len(rows), "C": C, "n_blocks": int(nb), "score": float((A + Bq) / 2), "A": float(A), "B": float(Bq),
       "ctx_mass_A": float(np.mean([r[3] for r in rows])), "ctx_mass_B": float(np.mean([r[4] for r in rows])),
       "fut_pos_A": float(np.mean([r[5] for r in rows])), "fut_imp_A": float(np.mean([r[6] for r in rows])),
       "fut_pos_B": float(np.mean([r[7] for r in rows])), "fut_imp_B": float(np.mean([r[8] for r in rows]))}
if held:
    h = [r for r in rows if r[0] in held]
    Ah, Bh = 100 * np.mean([r[1] for r in h]), 100 * np.mean([r[2] for r in h])
    res["held48"] = {"n": len(h), "score": float((Ah + Bh) / 2), "A": float(Ah), "B": float(Bh)}
json.dump(res, open(os.path.join(OUT, f"state_surprise_{a.board}.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, ensure_ascii=False))
