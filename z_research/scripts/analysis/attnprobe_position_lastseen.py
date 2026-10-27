"""attentive probe 가 AR 예측에서 읽는 위치는 진짜 위치인가, 마지막에 본 위치인가 (2026-10-26 저녁, CPU).
사용자: "속도는 아예 안 읽히는 걸로 봐야겠네. 근데 위치는 왜 읽히는 거야?"
probe: attnprobe_train.py 가 저장한 attnprobe_{ro,enc}.pt (블록 토큰 (256, 1280) 위, 질의 3 개). 표본: val AR 예측 출력 토큰 (블록 8–15, 물체 보임).
비교: probe 가 낸 위치 P 를 (1) 그 블록의 진짜 중심 Y, (2) 블록 7 (마지막에 본) 중심 Y7 과 R² 로. 기준: Y7 로 Y 를 맞힌 R² (최소제곱, train 에서 맞춤 → val).
  /venv/vjepa/bin/python z_research/scripts/analysis/attnprobe_position_lastseen.py   → z_analysis/results/attnprobe_position_lastseen.json
"""
import json
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(32)
D = "/workspace/auto_research_cache/attnprobe"; M = "/workspace/auto_research_cache/ar_layers"


class AttnProbe(nn.Module):
    def __init__(s, d=1280, heads=16, outs=(48, 2, 2)):
        super().__init__()
        s.ln = nn.LayerNorm(d); s.pos = nn.Parameter(torch.randn(256, d) * 0.02); s.q = nn.Parameter(torch.randn(len(outs), d) * 0.02)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True); s.ln2 = nn.LayerNorm(d); s.heads = nn.ModuleList([nn.Linear(d, o) for o in outs])

    def forward(s, x):
        x = s.ln(x) + s.pos; o, w = s.attn(s.q.expand(len(x), -1, -1), x, x, need_weights=True); o = s.ln2(o)
        return [h(o[:, i]) for i, h in enumerate(s.heads)], w


r2 = lambda P, Y: float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
meta = {m["i"]: m for m in json.load(open(f"{M}/val_meta.json"))}; tmeta = {m["i"]: m for m in json.load(open(f"{M}/train_meta.json"))}
X, L = np.load(f"{D}/val_ro_tok.npy", mmap_mode="r"), np.load(f"{D}/val_ro_lab.npy")
last = np.array([meta[int(i)]["blocks"][7]["ctr"] if meta[int(i)]["blocks"][7]["ok"] else [np.nan, np.nan] for i in L[:, 6]], dtype=np.float64)
ok7 = ~np.isnan(last).any(1); Y = L[:, 4:6].astype(np.float64); step = (L[:, 7] - 7).astype(int)
# 기준: train 에서 Y7 → Y 최소제곱
Lt = np.load(f"{D}/train_ro_lab.npy"); lt = np.array([tmeta[int(i)]["blocks"][7]["ctr"] if tmeta[int(i)]["blocks"][7]["ok"] else [np.nan, np.nan] for i in Lt[:, 6]])
mt = ~np.isnan(lt).any(1); B = np.linalg.lstsq(np.c_[lt[mt], np.ones(mt.sum())], Lt[mt, 4:6], rcond=None)[0]
Lab_t = np.load(f"{D}/train_enc_lab.npy"); pm, ps = Lab_t[:, 4:6].mean(0), Lab_t[:, 4:6].std(0) + 1e-6   # enc probe 의 라벨 표준화 (attnprobe_train 과 같게)
pm_ro, ps_ro = Lt[:, 4:6].mean(0), Lt[:, 4:6].std(0) + 1e-6
res = {"baseline_lastseen_to_true": {}}
for k in range(1, 9):
    m = ok7 & (step == k); res["baseline_lastseen_to_true"][k] = round(r2(np.c_[last[m], np.ones(m.sum())] @ B, Y[m]), 3)
for name, (pm_, ps_) in (("ro", (pm_ro, ps_ro)), ("enc", (pm, ps))):
    model = AttnProbe(); model.load_state_dict(torch.load(f"{D}/attnprobe_{name}.pt", map_location="cpu")); model.eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            o, _ = model(torch.from_numpy(np.ascontiguousarray(X[i:i + 128])).float()); P.append(o[2].numpy())
    P = np.concatenate(P) * ps_ + pm_
    res[f"probe_{name}"] = {k: {"r2_vs_true": round(r2(P[ok7 & (step == k)], Y[ok7 & (step == k)]), 3), "r2_vs_lastseen": round(r2(P[ok7 & (step == k)], last[ok7 & (step == k)]), 3),
                               "n": int((ok7 & (step == k)).sum())} for k in range(1, 9)}
for key in ("probe_ro", "probe_enc"):
    print(key, "진짜 위치 R²     ", [res[key][k]["r2_vs_true"] for k in range(1, 9)])
    print(key, "마지막 본 위치 R²", [res[key][k]["r2_vs_lastseen"] for k in range(1, 9)])
print("기준: 마지막 본 위치로 진짜 위치 맞히기 R²", [res["baseline_lastseen_to_true"][k] for k in range(1, 9)])
print("n", [res["probe_ro"][k]["n"] for k in range(1, 9)])
json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/attnprobe_position_lastseen.json", "w"), indent=1)
