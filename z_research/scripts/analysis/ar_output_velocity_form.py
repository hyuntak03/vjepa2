"""자연 영상 AR 출력의 속도는 encoder 와 '다른 꼴' 인가, 같은 꼴인데 작아진 것인가 (2026-10-26, CPU).
사용자: "왜 encoder 랑 다르게 그게 형성된 거지".
앞 측정 (phys_only_ar_velocity.json): teacher forcing 출력 속도 R² — encoder 판독기 0.04, 그 모델 출력으로 학습한 판독기 0.64.
가설 (a) 같은 방향, 작은 크기: L1 최적이 블록 안 움직임 성분을 중앙값 쪽 (0 근처) 으로 줄였다 → encoder 판독기의 출력이 정답과 같은 방향으로 움직이되 기울기가 작다.
     이 경우 encoder 판독기 출력에 1 차 보정 (기울기 · 절편) 만 하면 R² 가 크게 오른다.
가설 (b) 다른 꼴: encoder 판독기 출력이 정답과 상관이 낮다 → 보정해도 안 오른다.
입력: ar_layers val 캐시 (val 350 편, tf_out = natural_ar_vw_state teacher forcing 출력, enc = 정답 블록) · attnprobe_enc.pt (tf_output_velocity.py 와 같은 표본 1,923 개).
지표 (축 x · y 따로, 그리고 둘 합): 상관 r · 기울기 (판독 = 기울기 × 정답 + 절편, 최소제곱) · 보정 전 R² · 보정 후 R² (영상 절반으로 보정, 나머지 절반 시험, 두 번 바꿔 평균).
  /venv/vjepa/bin/python z_research/scripts/analysis/ar_output_velocity_form.py   → z_analysis/results/ar_output_velocity_form.json
"""
import json, os
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(32)
SRC = "/workspace/auto_research_cache/ar_layers"; D = "/workspace/auto_research_cache/attnprobe"


class AttnProbe(nn.Module):
    def __init__(s, d=1280, heads=16, outs=(48, 2, 2)):
        super().__init__()
        s.ln = nn.LayerNorm(d); s.pos = nn.Parameter(torch.randn(256, d) * 0.02); s.q = nn.Parameter(torch.randn(len(outs), d) * 0.02)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True); s.ln2 = nn.LayerNorm(d); s.heads = nn.ModuleList([nn.Linear(d, o) for o in outs])

    def forward(s, x):
        x = s.ln(x) + s.pos; o, _ = s.attn(s.q.expand(len(x), -1, -1), x, x, need_weights=False); o = s.ln2(o)
        return [h(o[:, i]) for i, h in enumerate(s.heads)]


Lt = np.load(f"{D}/train_enc_lab.npy"); vm, vs = Lt[:, 2:4].mean(0), Lt[:, 2:4].std(0) + 1e-6
model = AttnProbe(); model.load_state_dict(torch.load(f"{D}/attnprobe_enc.pt", map_location="cpu")); model.eval()
meta = json.load(open(f"{SRC}/val_meta.json"))
TO, EN, Y, CL = [], [], [], []
for m in meta:
    path = f"{SRC}/val/{m['i']:05d}.npz"; z = np.load(path); tfo, enc = z["tf_out"], z["enc"]; z.close()
    fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    for b in range(2, 16):
        if m["blocks"][b]["ok"] and m["blocks"][b - 1]["ok"]:
            TO.append(tfo[b - 1]); EN.append(enc[b]); Y.append(m["blocks"][b]["disp"]); CL.append(m["i"])
Y = np.array(Y, np.float64); CL = np.array(CL)


def run(X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            out.append(model(torch.from_numpy(np.stack(X[i:i + 128]).astype(np.float32)))[1].numpy())
    return np.concatenate(out) * vs + vm


def form(V, Y):
    res = {}
    for nm, ax in (("x", [0]), ("y", [1]), ("xy", [0, 1])):
        v, y = V[:, ax].ravel(), Y[:, ax].ravel()
        r = float(np.corrcoef(v, y)[0, 1]); slope = float(np.polyfit(y, v, 1)[0])
        r2_raw = float(1 - ((v - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        r2_cal = []
        for half in (0, 1):
            tr = (CL % 2 == half); te = ~tr
            A = np.c_[V[tr][:, ax].ravel(), np.ones(tr.sum() * len(ax))]; w = np.linalg.lstsq(A, Y[tr][:, ax].ravel(), rcond=None)[0]
            p = np.c_[V[te][:, ax].ravel(), np.ones(te.sum() * len(ax))] @ w; yt = Y[te][:, ax].ravel()
            r2_cal.append(1 - ((p - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
        res[nm] = {"corr": round(r, 3), "slope_readout_per_true": round(slope, 3), "R2_raw": round(r2_raw, 3), "R2_after_linear_recalibration": round(float(np.mean(r2_cal)), 3)}
    return res


R = {"n": int(len(Y)), "encoder_tokens": form(run(EN), Y), "natural_ar_tf_output": form(run(TO), Y)}
print(json.dumps(R, indent=1))
json.dump(R, open("/workspace/world_model/vjepa2/z_analysis/results/ar_output_velocity_form.json", "w"), indent=1)
