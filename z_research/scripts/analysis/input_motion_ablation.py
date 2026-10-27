"""입력 블록 안 움직임을 지우면 다음 블록 예측의 이동이 줄어드나 — 인과 확인 (2026-10-26 저녁, GPU 1 장 · 6 GB).
사용자: "TF 에서도 속도 죽은 채로 있고, input 으로 속도가 들어오니까 조금씩 살아나는 것" → "해 봐".
3D 합성 val (ar_layers val_meta, 앞 N 편). 블록 = 연속 2 장 (원본 0, 3, …, 93 장 중 2b, 2b+1). 두 가지 입력:
  보통 = 블록마다 (f_2b, f_2b+1) 를 encoder 에 · 정지 = 블록마다 (f_2b, f_2b) — 블록 안 움직임만 0, 블록 사이 위치 변화는 그대로.
블록 AR natural_ar_vw_state: teacher forcing (forward_seq) 출력 tf_out[j] = 블록 j+1 예측 / 굴림 (문맥 8 블록 → 8 걸음).
위치 판독: attnprobe_ro.pt (AR 굴림 출력 토큰으로 학습한 attentive probe; 예측 토큰 분포에 맞음) · 속도 판독은 같은 probe.
지표: 한 블록 이동 따라 움직인 비율 (실제 이동 ≥ 8 px, 기준 = 입력 블록 j 의 실제 위치) · 굴림 따라 움직인 비율 (실제 이동 ≥ 16 px, 기준 = 블록 7 위치) · 출력 속도 R².
  CUDA_VISIBLE_DEVICES=0 MEM_GB=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/input_motion_ablation.py --n 150
출력: z_analysis/results/input_motion_ablation.json
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=150); a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 6)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT
sys.argv = _argv
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
S, C, K, D = 256, 8, 8, 1280; IDX = [3 * k for k in range(32)]
enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
_load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.half().to(dev).eval()
ar = FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)


class AttnProbe(nn.Module):
    def __init__(s, d=1280, heads=16, outs=(48, 2, 2)):
        super().__init__()
        s.ln = nn.LayerNorm(d); s.pos = nn.Parameter(torch.randn(256, d) * 0.02); s.q = nn.Parameter(torch.randn(len(outs), d) * 0.02)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True); s.ln2 = nn.LayerNorm(d); s.heads = nn.ModuleList([nn.Linear(d, o) for o in outs])

    def forward(s, x):
        x = s.ln(x) + s.pos; o, _ = s.attn(s.q.expand(len(x), -1, -1), x, x, need_weights=False); o = s.ln2(o)
        return [h(o[:, i]) for i, h in enumerate(s.heads)]


PD = "/workspace/auto_research_cache/attnprobe"; Lr = np.load(f"{PD}/train_ro_lab.npy")
vm, vs, pm, ps = Lr[:, 2:4].mean(0), Lr[:, 2:4].std(0) + 1e-6, Lr[:, 4:6].mean(0), Lr[:, 4:6].std(0) + 1e-6
probe = AttnProbe().to(dev); probe.load_state_dict(torch.load(f"{PD}/attnprobe_ro.pt", map_location="cpu")); probe.eval()


def read(tok):                                                         # (n, 256, 1280) → 속도 (n,2), 위치 (n,2) px
    with torch.no_grad():
        o = probe(tok.float()); return o[1].cpu().numpy() * vs + vm, o[2].cpu().numpy() * ps + pm


def encode(x, static):                                                 # x (3, 32, 256, 256) 정규화 → (16, 256, 1280)
    xb = x.reshape(3, 16, 2, 256, 256).permute(1, 0, 2, 3, 4)            # (16, 3, 2, H, W)
    if static:
        xb = xb[:, :, :1].expand(-1, -1, 2, -1, -1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.cat([F.layer_norm(enc2(xb[i:i + 4].half().contiguous()).float(), (D,)) for i in range(0, 16, 4)])


meta = json.load(open("/workspace/auto_research_cache/ar_layers/val_meta.json"))[: a.n]
rec = {m_: {"tf_pred": [], "tf_true": [], "tf_prev": [], "tf_vel_p": [], "tf_vel_t": [], "ro_pred": [], "ro_true": [], "ro_last": [], "ro_vel_p": [], "ro_vel_t": []} for m_ in ("normal", "static")}
idx_all = torch.arange(16 * S, device=dev).unsqueeze(0)
for n, m in enumerate(meta):
    fr = decord.VideoReader(m["clip"], num_threads=4).get_batch(IDX).asnumpy()
    x = (torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0 - MEAN) / STD
    B = m["blocks"]
    for mode in ("normal", "static"):
        h = encode(x, mode == "static")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = ar.forward_seq(h.reshape(1, -1, D), idx_all)
            out = (out[0] if isinstance(out, tuple) else out).float()[0].view(16, S, D)          # tf_out[j] = 블록 j+1 예측
            ro = ar.rollout_kv(h[:C].reshape(1, C * S, D), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, D)
        js = [j for j in range(1, 15) if B[j]["ok"] and B[j + 1]["ok"]]
        if js:
            v, p = read(out[js]); r_ = rec[mode]
            r_["tf_pred"] += list(p); r_["tf_true"] += [B[j + 1]["ctr"] for j in js]; r_["tf_prev"] += [B[j]["ctr"] for j in js]; r_["tf_vel_p"] += list(v); r_["tf_vel_t"] += [B[j + 1]["disp"] for j in js]
        ks = [k for k in range(8) if B[8 + k]["ok"] and B[7]["ok"]]
        if ks:
            v, p = read(ro[ks]); r_ = rec[mode]
            r_["ro_pred"] += [(k, *pp) for k, pp in zip(ks, p)]; r_["ro_true"] += [B[8 + k]["ctr"] for k in ks]; r_["ro_last"] += [B[7]["ctr"] for k in ks]; r_["ro_vel_p"] += list(v); r_["ro_vel_t"] += [B[8 + k]["disp"] for k in ks]
    if n % 30 == 0:
        print(f"{n}/{len(meta)}", flush=True)

r2 = lambda P, Y: float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
res = {"n_clips": len(meta)}
for mode, r_ in rec.items():
    P, Y, Pv = np.array(r_["tf_pred"]), np.array(r_["tf_true"]), np.array(r_["tf_prev"]); t = Y - Pv; d = np.linalg.norm(t, axis=1) >= 8
    res[mode] = {"tf_one_block_follow_ratio": round(float(((P[d] - Pv[d]) * t[d]).sum() / (t[d] ** 2).sum()), 3), "tf_n_moved8": int(d.sum()),
                 "tf_output_velocity_r2": round(r2(np.array(r_["tf_vel_p"]), np.array(r_["tf_vel_t"])), 3)}
    RP = np.array(r_["ro_pred"]); RY, RL = np.array(r_["ro_true"]), np.array(r_["ro_last"]); st = RP[:, 0].astype(int); Pp = RP[:, 1:]; tt = RY - RL; dd = np.linalg.norm(tt, axis=1) >= 16
    res[mode]["rollout_follow_ratio_by_step"] = {int(k) + 1: round(float(((Pp[(st == k) & dd] - RL[(st == k) & dd]) * tt[(st == k) & dd]).sum() / max(1e-9, (tt[(st == k) & dd] ** 2).sum())), 3) for k in range(8)}
    res[mode]["rollout_output_velocity_r2"] = round(r2(np.array(r_["ro_vel_p"]), np.array(r_["ro_vel_t"])), 3)
    print(mode, res[mode], flush=True)
json.dump(res, open(f"{REPO}/z_analysis/results/input_motion_ablation.json", "w"), indent=1)
