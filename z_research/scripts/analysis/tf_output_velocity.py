"""teacher forcing 출력에도 속도가 죽어 있나 — 실제 문맥을 보고 낸 다음 블록 예측의 블록 안 속도 · 한 블록 이동 (2026-10-26 저녁, CPU).
사용자: "context 자리에서도 TF 한다고 미래 만드는데 거기서도 속도 죽은 채로 있을 것 같음. 자꾸 input 으로 속도 들어오니까 계속 조금씩 살아나는 것".
입력: ar_layers 전체 토큰 캐시 (val 350 편). tf_out[j] = 블록 AR natural_ar_vw_state 가 실제 블록 0..j 를 보고 낸 블록 j+1 예측 (1280, 256 칸). enc[b] = 블록 b 정답.
판독기: attnprobe_enc.pt (encoder 토큰으로 학습한 attentive probe, 모양×색 · 속도 · 위치, attnprobe_train.py) 를 그대로.
  블록 b = j+1 (2–15) 중 물체 보이는 것: 출력 속도 R² (블록 b 안 변위) · 출력 위치 R² vs encoder 블록 b (상한).
  한 블록 이동: 실제 이동 = 위치_b − 위치_j (실제), 예측 이동 = probe 위치 (tf_out[j]) − 위치_j. 실제 이동 ≥ 8 px 표본에서 따라 움직인 비율 Σ(예측 · 실제) / Σ|실제|².
  굴림 걸음 1 (= tf_out[7], 블록 8) 은 따로도 적는다.
  /venv/vjepa/bin/python z_research/scripts/analysis/tf_output_velocity.py   → z_analysis/results/tf_output_velocity.json
"""
import json, os
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(48)
SRC = "/workspace/auto_research_cache/ar_layers"; D = "/workspace/auto_research_cache/attnprobe"


class AttnProbe(nn.Module):
    def __init__(s, d=1280, heads=16, outs=(48, 2, 2)):
        super().__init__()
        s.ln = nn.LayerNorm(d); s.pos = nn.Parameter(torch.randn(256, d) * 0.02); s.q = nn.Parameter(torch.randn(len(outs), d) * 0.02)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True); s.ln2 = nn.LayerNorm(d); s.heads = nn.ModuleList([nn.Linear(d, o) for o in outs])

    def forward(s, x):
        x = s.ln(x) + s.pos; o, _ = s.attn(s.q.expand(len(x), -1, -1), x, x, need_weights=False); o = s.ln2(o)
        return [h(o[:, i]) for i, h in enumerate(s.heads)]


Lt = np.load(f"{D}/train_enc_lab.npy"); vm, vs, pm, ps = Lt[:, 2:4].mean(0), Lt[:, 2:4].std(0) + 1e-6, Lt[:, 4:6].mean(0), Lt[:, 4:6].std(0) + 1e-6
model = AttnProbe(); model.load_state_dict(torch.load(f"{D}/attnprobe_enc.pt", map_location="cpu")); model.eval()
meta = json.load(open(f"{SRC}/val_meta.json"))
TO, EN, LB = [], [], []
for m in meta:
    path = f"{SRC}/val/{m['i']:05d}.npz"; z = np.load(path); tfo, enc = z["tf_out"], z["enc"]; z.close()
    fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    for b in range(2, 16):
        bl, bp = m["blocks"][b], m["blocks"][b - 1]
        if bl["ok"] and bp["ok"]:
            TO.append(tfo[b - 1]); EN.append(enc[b]); LB.append([bl["disp"][0], bl["disp"][1], bl["ctr"][0], bl["ctr"][1], bp["ctr"][0], bp["ctr"][1], b])
LB = np.array(LB, np.float64); print("표본", len(LB), flush=True)


def run(X):
    V, P = [], []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            o = model(torch.from_numpy(np.stack(X[i:i + 128]).astype(np.float32))); V.append(o[1].numpy()); P.append(o[2].numpy())
    return np.concatenate(V) * vs + vm, np.concatenate(P) * ps + pm


r2 = lambda P, Y: float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
Vt, Pt = run(TO); Ve, Pe = run(EN)
Y_v, Y_p, prev, blk = LB[:, 0:2], LB[:, 2:4], LB[:, 4:6], LB[:, 6].astype(int)
mv = np.linalg.norm(Y_p - prev, axis=1) >= 8
ratio = lambda P, m: float(((P[m] - prev[m]) * (Y_p[m] - prev[m])).sum() / ((Y_p[m] - prev[m]) ** 2).sum())
res = {"n": int(len(LB)), "all_blocks_2_15": {"tf_output": {"velocity_r2": round(r2(Vt, Y_v), 3), "position_r2": round(r2(Pt, Y_p), 3), "follow_ratio_move8px": round(ratio(Pt, mv), 3)},
                                           "encoder": {"velocity_r2": round(r2(Ve, Y_v), 3), "position_r2": round(r2(Pe, Y_p), 3), "follow_ratio_move8px": round(ratio(Pe, mv), 3)},
                                           "n_moved8": int(mv.sum())}}
for nm, sel in (("context_blocks_2_8", blk <= 8), ("block_8_rollout_step1_equiv", blk == 8), ("blocks_9_15", blk >= 9)):
    res[nm] = {"tf_output_velocity_r2": round(r2(Vt[sel], Y_v[sel]), 3), "encoder_velocity_r2": round(r2(Ve[sel], Y_v[sel]), 3),
               "tf_output_follow": round(ratio(Pt, sel & mv), 3), "encoder_follow": round(ratio(Pe, sel & mv), 3), "n": int(sel.sum()), "n_moved8": int((sel & mv).sum())}
json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/tf_output_velocity.json", "w"), indent=1)
print(json.dumps(res, indent=1))
