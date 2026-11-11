"""맞추기 판 주소 (slot attention 맵) 에서 물체 상태를 바로 읽을 수 있나 (2026-11-07, 학습 없음).
물체 slot 주소의 무게 중심 → 위치 오차 (px), 주소 질량 → 보임 여부, 무게 중심 변화 → 속도. v4bc4 클립 (학습 분포).
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 /venv/vjepa/bin/python z_research/scripts/analysis/slot_address_state.py
출력: z_analysis/results/slot_diag_20261107/address_state.json
"""
import json, os, random, sys
os.environ.setdefault("WMA_AR_BLOCK_CACHE", "0")
import numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(__file__))
from slot_diag import LIST, OUT, RES, S, build_from_config, ln, load_clip
from slot_appearance_diag import MEAN, STD, tf_clip

MODEL, KOBJ, N = "vast_ar_vw_state_v4bc4_slotmatch", 4, 192
dev = torch.device("cuda:0")
b = build_from_config(yaml.safe_load(open(f"{RES}/real_world_permanence_uniform_{MODEL}/_resolved.yaml"))["model"], dev)
core = b.predictor.core
paths = [r.strip().split()[0] for r in open(LIST) if r.strip()]
random.Random(2).shuffle(paths)
pk = []
for p in paths:
    with np.load(p.replace("/clips/", "/labels/")[:-4] + ".npz") as z:
        if "target" in [str(r) for r in z["roles"]]:
            pk.append(p)
    if len(pk) == N:
        break
ys, xs = np.divmod(np.arange(256), 16)
xs, ys = torch.tensor(xs * 16 + 8.0, device=dev), torch.tensor(ys * 16 + 8.0, device=dev)
res = {}
for tf in ("none", "gray"):
    C_, T_, V_, Mass = [], [], [], []
    for i in range(0, N, 8):
        it = [load_clip(p, MEAN, STD) for p in pk[i:i + 8]]
        x = torch.stack([tf_clip(a[0], tf) for a in it]).to(dev, b.dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            hb = ln(b.target_encoder(x)).to(torch.float16)
            _, att, _ = core._slot_run(hb.view(x.size(0), 16, S, -1))
        a = att[:, :, KOBJ].float()
        m = a.sum(-1).clamp_min(1e-6)
        C_.append(torch.stack([(a * xs).sum(-1) / m, (a * ys).sum(-1) / m], -1).cpu().numpy())
        Mass.append(a.sum(-1).cpu().numpy())
        for L in (q[2] for q in it):
            f = np.arange(16) * 4
            T_.append((L["ctr"][f] + L["ctr"][f + 2]) / 2)
            vf = L["vis_n"] / np.maximum(L["am_n"], 1)
            V_.append((vf[f] + vf[f + 2]) / 2)
    C_, T_, V_, Mass = np.concatenate(C_), np.stack(T_), np.stack(V_), np.concatenate(Mass)
    vis, hid = V_ >= .5, V_ < .05
    e = np.linalg.norm(C_ - T_, axis=-1)
    dC, dT = (C_[:, 1:] - C_[:, :-1]), (T_[:, 1:] - T_[:, :-1])
    both = vis[:, 1:] & vis[:, :-1]
    r2 = 1 - ((dC[both] - dT[both]) ** 2).sum() / ((dT[both] - dT[both].mean(0)) ** 2).sum()
    res[tf] = {"ctr_max": float(np.abs(T_).max()), "n_vis": int(vis.sum()), "n_hid": int(hid.sum()),
               "err_px_median": float(np.median(e[vis])), "err_px_mean": float(e[vis].mean()), "err_px_p90": float(np.percentile(e[vis], 90)),
               "mass_vis": float(Mass[vis].mean()), "mass_hid": float(Mass[hid].mean()),
               "vel_r2": float(r2), "true_step_px": float(np.linalg.norm(dT[both], axis=-1).mean())}
json.dump(res, open(os.path.join(OUT, "address_state.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
