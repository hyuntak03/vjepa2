"""굴림에서 속도 되살리기 개입 + attention 배분 분석 — 블록 AR natural_ar_vw_state (2026-10-26 저녁, GPU 1 장 · 6 GB).
사용자: "개입해서 최종 output future block 에서 속도 회복되면 좋고. 왜 이렇게 됐는지 — supervision · mask · attention 구조" → "ㅇㅇ".
근거: input_motion_ablation.json — AR 은 이동을 주로 입력 블록 안 움직임에서 끌어낸다. 굴림 입력 (자기 상태) 에는 그 움직임이 약하다.
움직임 성분 Δ_p = 상태_p(보통 입력) − 상태_p(정지 입력) (teacher forcing 자리 p 의 predictor 마지막 정규화 출력 (384); 정지 = 블록 두 장을 같은 장으로).
굴림 걸음 r (자리 7+r, 입력 = 블록 7+r 를 예측한 상태) 의 입력:
  0 기본       : 상태 + type[1]
  1 정답 상태  : teacher forcing 상태_{7+r−1}(보통) + type[1]                      (상한)
  2 정답 움직임: 상태 + type[1] + Δ_{7+r−1}                                        (상한, 움직임만)
  3 이어 붙임  : 상태 + type[1] + shift(Δ_7, d_r) — Δ_7 = 마지막 문맥 자리의 움직임 성분, d_r = 예측 출력 (블록 7+r) 에서 probe 로 읽은 물체 위치 − 블록 7 위치 (칸 단위)   (정답 없음)
  3b 제자리    : 상태 + type[1] + Δ_7                                               (정답 없음)
  (2026-10-26 추가: 상태 경로는 움직임을 잘 못 나른다 → 관측 입력 경로로)
  4 출력 되먹임: embed_obs(예측 출력 블록 7+r) = predictor_embed(출력) + type[0]  (움직임 추가 없음)
  5 출력 되먹임 + 관측 움직임: 4 + shift(M_7, d_r), M_7 = embed_obs(블록 7 보통) − embed_obs(블록 7 정지)   (정답 없음)
  6 실제 블록 관측 입력: embed_obs(실제 블록 7+r)                                 (상한, teacher forcing 과 같은 입력)
판독: attnprobe_ro.pt (굴림 출력 토큰으로 학습한 attentive probe). 걸음별 따라 움직인 비율 (실제 이동 ≥ 16 px, 기준 = 블록 7 위치) · 출력 속도 R².
attention: ACRoPEAttention.forward_kv 를 같은 계산 + 가중치 기록판으로 바꿔 끼움. 굴림 (조건 0) 자리 9–15 의 질의 vs 같은 자리에 실제 블록을 넣은 경우 (teacher forcing 과 같은 계산).
  키 블록별 질량: 문맥 0–6 · 마지막 문맥 7 · 그 뒤 (8 … 자리−1) · 자기 블록. 질의 = 전체 칸 / 실제 물체 칸. 층 × head.
  CUDA_VISIBLE_DEVICES=0 MEM_GB=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/motion_intervention.py --n 150 --n_attn 40
출력: z_analysis/results/motion_intervention.json
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=150); ap.add_argument("--n_attn", type=int, default=40); a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 6)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT
sys.argv = _argv
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
import src.models.utils.modules as MOD
from src.models.utils.modules import rotate_queries_or_keys
S, C, K, D = 256, 8, 8, 1280; IDX = [3 * k for k in range(32)]

# ---- attention 기록판 (계산은 원본 forward_kv 와 같다)
REC = {"on": False, "obj_q": None, "stats": []}


def forward_kv_rec(self, x, mask, H, W, past=None):
    B, N, Cc = x.size()
    m = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
    d_mask, h_mask, w_mask = self.separate_positions(m, H, W); h_mask *= self.grid_size / H; w_mask *= self.grid_size / W
    qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4); q, k, v = qkv[0], qkv[1], qkv[2]; s = 0
    qd = rotate_queries_or_keys(q[..., s:s + self.d_dim], pos=d_mask); kd = rotate_queries_or_keys(k[..., s:s + self.d_dim], pos=d_mask); s += self.d_dim
    qh = rotate_queries_or_keys(q[..., s:s + self.h_dim], pos=h_mask); kh = rotate_queries_or_keys(k[..., s:s + self.h_dim], pos=h_mask); s += self.h_dim
    qw = rotate_queries_or_keys(q[..., s:s + self.w_dim], pos=w_mask); kw = rotate_queries_or_keys(k[..., s:s + self.w_dim], pos=w_mask); s += self.w_dim
    if s < self.head_dim:
        q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1); k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
    else:
        q = torch.cat([qd, qh, qw], dim=-1); k = torch.cat([kd, kh, kw], dim=-1)
    if past is not None:
        k = torch.cat([past[0], k], dim=2); v = torch.cat([past[1], v], dim=2)
    if REC["on"]:
        att = torch.softmax((q.float() @ k.float().transpose(-1, -2)) * q.shape[-1] ** -0.5, dim=-1)          # (B, h, N, M)
        nbk = att.shape[-1] // S; mass = att.view(B, self.num_heads, N, nbk, S).sum(-1)[0]                      # (h, N, nbk)
        oq = REC["obj_q"]
        REC["stats"].append((self._lid, mass.mean(1).cpu().numpy(), mass[:, oq].mean(1).cpu().numpy() if oq else None))
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob)
    x = x.transpose(1, 2).reshape(B, N, Cc); x = self.proj_drop(self.proj(x))
    return x, (k, v)


MOD.ACRoPEAttention.forward_kv = forward_kv_rec

enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
_load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.half().to(dev).eval()
ar = FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval()
for i, blk in enumerate(ar.predictor_blocks):
    blk.attn._lid = i
T1 = ar.type_embed[1]
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


def read(tok):
    with torch.no_grad():
        o = probe(tok.float()); return o[1].float().cpu().numpy() * vs + vm, o[2].float().cpu().numpy() * ps + pm


def encode(x, static):
    xb = x.reshape(3, 16, 2, 256, 256).permute(1, 0, 2, 3, 4)
    if static:
        xb = xb[:, :, :1].expand(-1, -1, 2, -1, -1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.cat([F.layer_norm(enc2(xb[i:i + 4].half().contiguous()).float(), (D,)) for i in range(0, 16, 4)])


def shift_map(M, dy, dx):                                             # M (256, 384) 을 16×16 격자에서 (dy, dx) 칸 옮김, 빈 자리 0
    g = M.view(16, 16, -1); out = torch.zeros_like(g)
    ys, yd = (slice(0, 16 - dy), slice(dy, 16)) if dy >= 0 else (slice(-dy, 16), slice(0, 16 + dy))
    xs, xd = (slice(0, 16 - dx), slice(dx, 16)) if dx >= 0 else (slice(-dx, 16), slice(0, 16 + dx))
    out[yd, xd] = g[ys, xs]; return out.view(256, -1)


idx_all = torch.arange(16 * S, device=dev).unsqueeze(0)
CONDS = ["0_base", "1_oracle_state", "2_oracle_motion", "3_carry_shifted", "3b_carry_inplace", "4_output_feedback", "5_output_feedback_plus_motion", "6_oracle_observed"]
REC_OUT = {c: {"pred": [], "vel": [], "true": [], "last": [], "vtrue": [], "step": []} for c in CONDS}
ATT = {"rollout": [], "real": []}
meta = json.load(open("/workspace/auto_research_cache/ar_layers/val_meta.json"))[: a.n]
for n, m in enumerate(meta):
    B_ = m["blocks"]
    if not B_[7]["ok"]:
        continue
    fr = decord.VideoReader(m["clip"], num_threads=4).get_batch(IDX).asnumpy()
    x = (torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0 - MEAN) / STD
    hn, hs = encode(x, False), encode(x, True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, stn = ar.forward_seq(hn.reshape(1, -1, D), idx_all, return_states=True); _, sts = ar.forward_seq(hs.reshape(1, -1, D), idx_all, return_states=True)
    stn, sts = stn.float()[0].view(16, S, -1), sts.float()[0].view(16, S, -1); Dl = stn - sts
    with torch.no_grad():
        M7 = (ar._ar_embed_obs(hn[7][None]) - ar._ar_embed_obs(hs[7][None])).float()[0]   # 관측 입력 공간의 움직임 성분
    last = np.array(B_[7]["ctr"])
    do_attn = n < a.n_attn
    for cond in CONDS:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            caches = None
            for b in range(C):
                p, caches, stt = ar._ar_step_kv(hn[b][None], idx_all[:, b * S:(b + 1) * S], caches)
            outs = [p.float()[0]]; pos = idx_all[:, (C - 1) * S: C * S]
            for r in range(1, K):
                pos = pos + S; inp = stt.float() + T1
                if cond == "1_oracle_state":
                    inp = stn[7 + r - 1][None] + T1
                elif cond == "2_oracle_motion":
                    inp = inp + Dl[7 + r - 1][None]
                elif cond == "3b_carry_inplace":
                    inp = inp + Dl[7][None]
                elif cond == "3_carry_shifted":
                    _, pp = read(outs[-1][None]); d = (pp[0] - last) / 16.0
                    inp = inp + shift_map(Dl[7], int(round(d[1])), int(round(d[0])))[None]
                elif cond == "4_output_feedback":
                    inp = ar._ar_embed_obs(outs[-1][None].to(stt.dtype)).float()
                elif cond == "5_output_feedback_plus_motion":
                    _, pp = read(outs[-1][None]); d = (pp[0] - last) / 16.0
                    inp = ar._ar_embed_obs(outs[-1][None].to(stt.dtype)).float() + shift_map(M7, int(round(d[1])), int(round(d[0])))[None]
                elif cond == "6_oracle_observed":
                    inp = ar._ar_embed_obs(hn[7 + r][None]).float()
                if cond == "0_base" and do_attn:
                    bl = B_[7 + r]; REC["on"] = True; REC["obj_q"] = bl["cells"] if bl["ok"] and bl["cells"] else None; REC["stats"] = []
                p, caches, stt = ar._ar_step_kv(inp.to(stt.dtype), pos, caches, embedded=True)
                if cond == "0_base" and do_attn:
                    REC["on"] = False; ATT["rollout"].append((r, list(REC["stats"])))
                outs.append(p.float()[0])
        P = torch.stack(outs); ks = [k for k in range(8) if B_[8 + k]["ok"]]
        if ks:
            v, pp = read(P[ks]); R_ = REC_OUT[cond]
            R_["pred"] += list(pp); R_["vel"] += list(v); R_["true"] += [B_[8 + k]["ctr"] for k in ks]; R_["last"] += [list(last)] * len(ks); R_["vtrue"] += [B_[8 + k]["disp"] for k in ks]; R_["step"] += [k + 1 for k in ks]
    if do_attn:                                                       # 같은 자리에 실제 블록을 넣은 경우 (teacher forcing 과 같은 계산)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            caches = None
            for b in range(C):
                _, caches, _ = ar._ar_step_kv(hn[b][None], idx_all[:, b * S:(b + 1) * S], caches)
            for r in range(1, K):
                bl = B_[7 + r]; REC["on"] = True; REC["obj_q"] = bl["cells"] if bl["ok"] and bl["cells"] else None; REC["stats"] = []
                _, caches, _ = ar._ar_step_kv(hn[7 + r][None], idx_all[:, (7 + r) * S:(8 + r) * S], caches)
                REC["on"] = False; ATT["real"].append((r, list(REC["stats"])))
    if n % 25 == 0:
        print(f"{n}/{len(meta)}", flush=True)

r2 = lambda P, Y: float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
res = {"n_clips": len(meta), "conditions": {}}
for cond, R_ in REC_OUT.items():
    P, Y, L, st = np.array(R_["pred"]), np.array(R_["true"]), np.array(R_["last"]), np.array(R_["step"]); t = Y - L; mv = np.linalg.norm(t, axis=1) >= 16
    fr_ = {int(k): round(float(((P[(st == k) & mv] - L[(st == k) & mv]) * t[(st == k) & mv]).sum() / max(1e-9, (t[(st == k) & mv] ** 2).sum())), 3) for k in range(1, 9)}
    res["conditions"][cond] = {"follow_ratio_by_step": fr_, "follow_ratio_steps3_8": round(float(((P[(st >= 3) & mv] - L[(st >= 3) & mv]) * t[(st >= 3) & mv]).sum() / (t[(st >= 3) & mv] ** 2).sum()), 3),
                               "velocity_r2_all": round(r2(np.array(R_["vel"]), np.array(R_["vtrue"])), 3),
                               "velocity_r2_steps3_8": round(r2(np.array(R_["vel"])[st >= 3], np.array(R_["vtrue"])[st >= 3]), 3), "n_moved16_by_step": {int(k): int(((st == k) & mv).sum()) for k in range(1, 9)}}
    print(cond, res["conditions"][cond], flush=True)


def summarize(lst, use_obj):
    """키 블록 묶음별 질량: 문맥 0–6 · 마지막 문맥 7 · 그 뒤 8…p−1 · 자기 p. 층 × head 평균, 걸음별."""
    out = {}
    for r in range(1, K):
        acc = []
        for rr, stats in lst:
            if rr != r:
                continue
            per_layer = np.zeros((12, 12, 4))
            ok = True
            for lid, mall, mobj in stats:
                mm = mobj if use_obj else mall
                if mm is None:
                    ok = False; break
                p = mm.shape[-1] - 1                                  # 자기 블록 = 마지막 키 블록 (자리 7+r)
                per_layer[lid] = np.stack([mm[:, :7].sum(-1), mm[:, 7], mm[:, 8:p].sum(-1) if p > 8 else np.zeros(12), mm[:, p]], -1)
            if ok:
                acc.append(per_layer)
        if acc:
            out[r] = np.mean(acc, 0)                                     # (12 층, 12 head, 4)
    return out


res["attention"] = {}
for kind in ("rollout", "real"):
    for use_obj in (False, True):
        sm = summarize(ATT[kind], use_obj); key = f"{kind}_{'objq' if use_obj else 'allq'}"
        res["attention"][key] = {int(r): {"layer_mean": np.round(v.mean(1), 3).tolist(), "all_mean": np.round(v.mean((0, 1)), 3).tolist()} for r, v in sm.items()}
        print(key, {r: res["attention"][key][r]["all_mean"] for r in res["attention"][key]}, flush=True)
json.dump(res, open(f"{REPO}/z_analysis/results/motion_intervention.json", "w"), indent=1)
print("saved")
