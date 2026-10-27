"""굴림에서 속도가 걸음마다 일정 비율로 줄어드나 (p^k 꼴) — 블록 AR natural_ar_vw_state, 3D 합성 val (2026-10-27, CPU).
사용자: "일단 확인해 봐". 그림: 지금 encoder 표현에서 차원별 중앙값은 속도를 평균처럼 비율 p 만큼 남긴다 → L1 최적이 걸음마다 속도를 p 배로 줄이고,
  굴리면 p, p², p³ … 으로 쌓여 멈춘다. 이 그림이 맞으면 걸음별 '이어 간 몫' 이 기하급수로 줄어야 하고, teacher forcing (매 걸음 실제 입력) 은 줄지 않아야 한다.
입력: ar_layers val 캐시 (350 편) — ro_out (8, 256, 1280) = 문맥 블록 0–7 → 블록 8–15 굴림 출력 · tf_out (16, 256, 1280) = 블록 b+1 teacher forcing 출력.
판독기: attnprobe_ro.pt (굴림 출력 토큰으로 학습한 attentive probe, 속도 · 위치 머리; motion_intervention.py 와 같은 것).
지표 (걸음 k = 1 … 8, 블록 7+k):
  속도 이득 = Σ(읽은 블록 안 속도 · 실제) / Σ|실제|²  (실제 |속도| ≥ 2 px 인 표본)
  위치 증분 이득 = Σ(Δ읽은 위치 · Δ실제 위치) / Σ|Δ실제|², Δ = 블록 7+k 위치 − 블록 7+k−1 위치 (k = 1 은 블록 7 의 실제 위치에서), 실제 |Δ| ≥ 4 px
  같은 지표를 teacher forcing 출력 (블록 8–15 를 실제 문맥으로 1 걸음씩) 에도.
  기하 맞춤: 굴림 이득 g_k ≈ a · p^(k−1) 을 걸음 1–8 에 최소제곱 (log) 으로 → p.
  /venv/vjepa/bin/python z_research/scripts/analysis/rollout_geometric_decay.py   → z_analysis/results/rollout_geometric_decay.json
"""
import json, os
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(32)
SRC = "/workspace/auto_research_cache/ar_layers"; PD = "/workspace/auto_research_cache/attnprobe"


class AttnProbe(nn.Module):
    def __init__(s, d=1280, heads=16, outs=(48, 2, 2)):
        super().__init__()
        s.ln = nn.LayerNorm(d); s.pos = nn.Parameter(torch.randn(256, d) * 0.02); s.q = nn.Parameter(torch.randn(len(outs), d) * 0.02)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True); s.ln2 = nn.LayerNorm(d); s.heads = nn.ModuleList([nn.Linear(d, o) for o in outs])

    def forward(s, x):
        x = s.ln(x) + s.pos; o, _ = s.attn(s.q.expand(len(x), -1, -1), x, x, need_weights=False); o = s.ln2(o)
        return [h(o[:, i]) for i, h in enumerate(s.heads)]


Lr = np.load(f"{PD}/train_ro_lab.npy"); vm, vs, pm, ps = Lr[:, 2:4].mean(0), Lr[:, 2:4].std(0) + 1e-6, Lr[:, 4:6].mean(0), Lr[:, 4:6].std(0) + 1e-6
probe = AttnProbe(); probe.load_state_dict(torch.load(f"{PD}/attnprobe_ro.pt", map_location="cpu")); probe.eval()


def read(tok):
    with torch.no_grad():
        o = probe(torch.from_numpy(tok.astype(np.float32)))
    return o[1].numpy() * vs + vm, o[2].numpy() * ps + pm


meta = json.load(open(f"{SRC}/val_meta.json"))
rows = {"ro": [], "tf": []}                     # (걸음, 읽은 속도 2, 실제 속도 2, 읽은 위치 2, 실제 위치 2, 직전 실제 위치 2, 직전 읽은 위치 2, 직전 ok)
for m in meta:
    B = m["blocks"]
    if not B[7]["ok"]:
        continue
    path = f"{SRC}/val/{m['i']:05d}.npz"; z = np.load(path); ro, tf = z["ro_out"], z["tf_out"]; z.close()
    fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    for tag, toks in (("ro", ro), ("tf", tf[7:15])):          # tf[b] = 블록 b+1 예측 → 블록 8–15
        V, P = read(toks)
        prev_read = np.array(B[7]["ctr"], float)              # 걸음 1 의 '직전 읽은 위치' = 블록 7 실제 위치 (굴림 시작점)
        for k in range(1, 9):
            b = 7 + k; bl, bp = B[b], B[b - 1]
            if bl["ok"]:
                rows[tag].append([k, *V[k - 1], *bl["disp"], *P[k - 1], *bl["ctr"], *(bp["ctr"] if bp["ok"] else [np.nan, np.nan]),
                                  *(prev_read if tag == "ro" or k == 1 else (np.array(bp["ctr"], float) if bp["ok"] else [np.nan, np.nan])), float(bp["ok"])])
            prev_read = P[k - 1]
R = {}
for tag in ("ro", "tf"):
    A = np.array(rows[tag], np.float64); res = {}
    for k in range(1, 9):
        S = A[A[:, 0] == k]; vr, vt = S[:, 1:3], S[:, 3:5]; mv = np.linalg.norm(vt, axis=1) >= 2
        vg = float((vr[mv] * vt[mv]).sum() / (vt[mv] ** 2).sum()) if mv.sum() else None
        ok = S[:, 13] > 0; dt = S[:, 7:9] - S[:, 9:11]; dr = S[:, 5:7] - S[:, 11:13]; m2 = ok & np.isfinite(dt).all(1) & np.isfinite(dr).all(1) & (np.linalg.norm(np.nan_to_num(dt), axis=1) >= 4)
        pg = float((dr[m2] * dt[m2]).sum() / (dt[m2] ** 2).sum()) if m2.sum() else None
        res[str(k)] = {"velocity_gain": None if vg is None else round(vg, 3), "n_v": int(mv.sum()), "position_increment_gain": None if pg is None else round(pg, 3), "n_p": int(m2.sum())}
    R[tag] = res
    for key in ("velocity_gain", "position_increment_gain"):
        g = np.array([res[str(k)][key] for k in range(1, 9)], float); okg = g > 0.02
        if okg.sum() >= 3:
            sl, ic = np.polyfit(np.arange(1, 9)[okg] - 1, np.log(g[okg]), 1); R[tag][f"{key}_geometric_fit"] = {"p": round(float(np.exp(sl)), 3), "a": round(float(np.exp(ic)), 3)}
    print(tag, " ".join(f"{k}: v {v['velocity_gain']} Δpos {v['position_increment_gain']}" for k, v in res.items() if k.isdigit()), {k: v for k, v in R[tag].items() if "fit" in k}, flush=True)
json.dump(R, open("/workspace/world_model/vjepa2/z_analysis/results/rollout_geometric_decay.json", "w"), indent=1)
print("saved")
