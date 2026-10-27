"""위치 판독 R² 가 왜 의미 없나 — "마지막에 본 위치" 만으로 거의 같은 R² 가 나온다 (2026-10-26 저녁, CPU).
사용자: "위치에 대한 그래프는 다 필요 없네. 의미 없는 거니까. position 그래프가 의미 없다는 걸 증명하는 figure 하나 따로".
probe: attnprobe_train.py 의 attnprobe_enc.pt (encoder 토큰으로 학습) · attnprobe_ro.pt (AR 예측 출력 토큰으로 학습), 블록 토큰 (256, 1280) 위 attentive probe.
표본: val 350 편, 블록 b = 8–15 (예측 걸음 k = b − 7) 와 블록 7 모두 물체 보일 때. 위치 = 물체 중심 (x, y) px (256 화면).
  (a) 위치 R²: encoder probe → encoder 토큰 (상한) · AR probe → AR 예측 출력 토큰 · 기준선 = 블록 7 위치만으로 직선 회귀 (train 에서 맞춤, 모델 없음)
  (b) 데이터: 클립 사이 위치 흩어짐 (x · y 표준편차 평균) vs 블록 7 → 블록 b 이동 거리 (중앙값 · 75 %)
  (c) 변위 R²: 표적 = 진짜 위치 − 블록 7 위치. 예측 변위 = probe 가 낸 위치 − 블록 7 위치 (새로 학습하지 않음). encoder 상한 · AR 예측
      · 기준선 (모델 없음, 걸음마다 train 에서 직선 회귀): 장면 종류 13 원핫 + 블록 7 위치 → 변위
  (d) 따라 움직인 비율 (2026-10-26 사용자: "(c) 는 이해하기 힘들다"): 실제로 16 px 이상 움직인 표본에서 Σ(예측 변위 · 실제 변위) / Σ|실제 변위|² (1 = 실제만큼, 0 = 제자리)
  표본별 값은 position_readout_meaningless_samples.npz 로 저장
  /venv/vjepa/bin/python z_research/scripts/analysis/position_readout_meaningless.py   → z_analysis/results/position_readout_meaningless.json
"""
import json
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(48)
D = "/workspace/auto_research_cache/attnprobe"; M = "/workspace/auto_research_cache/ar_layers"


class AttnProbe(nn.Module):
    def __init__(s, d=1280, heads=16, outs=(48, 2, 2)):
        super().__init__()
        s.ln = nn.LayerNorm(d); s.pos = nn.Parameter(torch.randn(256, d) * 0.02); s.q = nn.Parameter(torch.randn(len(outs), d) * 0.02)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True); s.ln2 = nn.LayerNorm(d); s.heads = nn.ModuleList([nn.Linear(d, o) for o in outs])

    def forward(s, x):
        x = s.ln(x) + s.pos; o, _ = s.attn(s.q.expand(len(x), -1, -1), x, x, need_weights=False); o = s.ln2(o)
        return [h(o[:, i]) for i, h in enumerate(s.heads)]


r2 = lambda P, Y: float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
vmeta = {m["i"]: m for m in json.load(open(f"{M}/val_meta.json"))}; tmeta = {m["i"]: m for m in json.load(open(f"{M}/train_meta.json"))}
last_of = lambda meta, I: np.array([meta[int(i)]["blocks"][7]["ctr"] if meta[int(i)]["blocks"][7]["ok"] else [np.nan, np.nan] for i in I], dtype=np.float64)


def probe_positions(name, X, label_train):
    pm, ps = label_train[:, 4:6].mean(0), label_train[:, 4:6].std(0) + 1e-6          # attnprobe_train.py 와 같은 라벨 표준화
    model = AttnProbe(); model.load_state_dict(torch.load(f"{D}/attnprobe_{name}.pt", map_location="cpu")); model.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            P.append(model(torch.from_numpy(np.ascontiguousarray(X[i:i + 128])).float())[2].numpy())
    return np.concatenate(P) * ps + pm


SETS, SETS_I = {}, {}
for name, kind in (("enc", "enc"), ("ro", "ro")):
    X, L = np.load(f"{D}/val_{kind}_tok.npy", mmap_mode="r"), np.load(f"{D}/val_{kind}_lab.npy")
    keep = (L[:, 7] >= 8) & (L[:, 7] <= 15); idx = np.nonzero(keep)[0]
    P = probe_positions(name, X[idx], np.load(f"{D}/train_{kind}_lab.npy")); Lk = L[idx]
    SETS[name] = (P, Lk[:, 4:6].astype(np.float64), last_of(vmeta, Lk[:, 6]), (Lk[:, 7] - 7).astype(int))
    SETS_I[name] = Lk[:, 6]
# 기준선: 블록 7 위치 → 블록 b 위치 (train 직선 회귀, 걸음 구분 없이)
Lt = np.load(f"{D}/train_ro_lab.npy"); lt = last_of(tmeta, Lt[:, 6]); mt = ~np.isnan(lt).any(1)
B = np.linalg.lstsq(np.c_[lt[mt], np.ones(mt.sum())], Lt[mt, 4:6], rcond=None)[0]
SC = sorted({m["scen"] for m in tmeta.values()})
def scene_feat(meta, I, Y7):
    return np.c_[np.eye(len(SC))[[SC.index(meta[int(i)]["scen"]) for i in I]], Y7, np.ones(len(I))]
res = {"position_r2": {"encoder_probe_on_encoder": {}, "ar_probe_on_ar_prediction": {}, "baseline_last_seen_only": {}},
       "displacement_r2": {"encoder_probe_on_encoder": {}, "ar_probe_on_ar_prediction": {}, "baseline_scene_plus_last_seen": {}}, "data": {}, "n": {}}
for k in range(1, 9):
    for name, key in (("enc", "encoder_probe_on_encoder"), ("ro", "ar_probe_on_ar_prediction")):
        P, Y, Y7, st = SETS[name]; m = (st == k) & ~np.isnan(Y7).any(1)
        res["position_r2"][key][k] = round(r2(P[m], Y[m]), 3); res["displacement_r2"][key][k] = round(r2(P[m] - Y7[m], Y[m] - Y7[m]), 3)
        if name == "ro":
            res["position_r2"]["baseline_last_seen_only"][k] = round(r2(np.c_[Y7[m], np.ones(m.sum())] @ B, Y[m]), 3); res["n"][k] = int(m.sum())
            mk = (Lt[:, 7] == 7 + k) & mt; Ft = scene_feat(tmeta, Lt[mk, 6], lt[mk]); Bs = np.linalg.lstsq(Ft, Lt[mk, 4:6] - lt[mk], rcond=None)[0]
            res["displacement_r2"]["baseline_scene_plus_last_seen"][k] = round(r2(scene_feat(vmeta, np.load(f"{D}/val_ro_lab.npy")[(np.load(f"{D}/val_ro_lab.npy")[:, 7] >= 8)][m][:, 6] if False else SETS_I["ro"][m], Y7[m]) @ Bs, Y[m] - Y7[m]), 3)
            d = np.linalg.norm(Y[m] - Y7[m], axis=1)
            res["data"][k] = {"spread_px": round(float(Y[m].std(0).mean()), 1), "move_median_px": round(float(np.median(d)), 1), "move_p75_px": round(float(np.percentile(d, 75)), 1)}
res["follow_ratio"] = {}
for name, key in (("enc", "encoder_probe_on_encoder"), ("ro", "ar_probe_on_ar_prediction")):
    P, Y, Y7, st = SETS[name]; res["follow_ratio"][key] = {}
    for k in range(1, 9):
        m = (st == k) & ~np.isnan(Y7).any(1); t = Y[m] - Y7[m]; pd = P[m] - Y7[m]; mv = np.linalg.norm(t, axis=1) >= 16
        res["follow_ratio"][key][k] = {"ratio": round(float((pd[mv] * t[mv]).sum() / (t[mv] ** 2).sum()), 3), "n_moved": int(mv.sum())}
np.savez("/workspace/world_model/vjepa2/z_analysis/results/position_readout_meaningless_samples.npz", **{f"{n}_{a}": v for n, (P, Y, Y7, st) in SETS.items() for a, v in (("pred", P), ("true", Y), ("last", Y7), ("step", st))})
json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/position_readout_meaningless.json", "w"), indent=1)
for grp in ("position_r2", "displacement_r2"):
    for key, v in res[grp].items():
        print(f"{grp:16s} {key:28s}", [v[k] for k in range(1, 9)])
print("data", [res["data"][k] for k in range(1, 9)]); print("n", [res["n"][k] for k in range(1, 9)])
for key, v in res["follow_ratio"].items():
    print("따라 움직인 비율", key, [v[k]["ratio"] for k in range(1, 9)], "n", [v[k]["n_moved"] for k in range(1, 9)])
