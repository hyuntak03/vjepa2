#!/usr/bin/env python3
"""시간 누적 하이브리드 (natural_ar_vw_state_tmem) 의 블록 기억이 실제로 쓰이는지 잰다 (2026-11-08, GPU 불필요 · CPU).

real world permanence val 클립 몇 개 (가능 영상) 를 학습 val 과 같은 로더 · 블록별 encoder 로 넣고:
  1. 누적 층마다 출력에 들어가는 두 몫의 크기 — ‖σ(g_own)⊙O_own‖ vs ‖σ(g_mem)⊙O_mem‖ (토큰 평균), 게이트 평균
  2. 잊기 α (head 평균 · 최소 · 최대, 8 블록 유지율 α^8) · 쓰기 β
  3. 기억 읽기를 0 으로 껐을 때 (전 누적 층) — 예측 변화 mean|Δp| 와 타깃 L1 변화 (TF · rollout 각각)
기억이 빈 첫 블록 호출은 1 · 2 통계에서 뺀다.

  /venv/vjepa/bin/python z_research/scripts/analysis/tmem_mem_diag.py [--ckpt z_training/runs/natural_ar_vw_state_tmem/latest.pt] [--n 6]
"""
import argparse, json, os, sys, time
from collections import defaultdict

import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from app.vjepa_frozen import utils as U                      # noqa: E402
from app.vjepa_frozen.ar import encode_blocks                  # noqa: E402
from app.vjepa_frozen.val import ValSet                        # noqa: E402
from src.models.utils.modules import ACRoPEAttention           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="natural_ar_vw_state_tmem")
ap.add_argument("--ckpt", default=None)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--threads", type=int, default=32)
ap.add_argument("--out", default=f"{REPO}/z_analysis/results/tmem_diag_20261108")
a = ap.parse_args()
torch.set_num_threads(a.threads)
cfg = yaml.safe_load(open(f"{REPO}/z_training/runs/{a.run}/config.yaml"))
MD, V = cfg["model"], cfg["val"]
ckpt = a.ckpt or f"{REPO}/z_training/runs/{a.run}/latest.pt"
dev = torch.device("cpu")

t0 = time.time()
state = U.load_base_checkpoint(MD["checkpoint"])
_, tgt = U.build_frozen_encoders(MD, state, dev, torch.float32, 32)
pred = U.build_predictor(MD, tgt.embed_dim, None, dev, 32)
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
pred.load_state_dict(ck["predictor"], strict=True)
pred.eval()
print(f"모델 {ckpt} (epoch {ck.get('epoch')}) · {time.time() - t0:.0f}s", flush=True)

vs = ValSet(V, 32, int(MD.get("img_size", 256)))
rows = [i for i, r in enumerate(vs.ds.rows) if str(r[vs.plcol]) == "1"][: a.n]
x = torch.stack([vs.ds.transform(vs.ds.read_uint8(i)) for i in rows]).float()
with torch.no_grad():
    h = encode_blocks(tgt, x, 2)                                # (B, 16*S, D) LN
S = pred.tokens_per_block
nb = h.size(1) // S
C, K = 8, nb - 8
idx = torch.arange(nb * S).unsqueeze(0).expand(h.size(0), -1)
print(f"클립 {len(rows)} · 블록 {nb} (문맥 {C} · 미래 {K}) · {time.time() - t0:.0f}s", flush=True)

# ---- 누적 층 계측 -----------------------------------------------------------------
lids = {}
for i, blk in enumerate(pred.predictor_blocks):
    if getattr(blk.attn, "mem_gate", None) is not None:
        blk.attn._lid = i
        lids[i] = blk.attn
ST = defaultdict(lambda: defaultdict(list))
FLAG = {"zero_mem": False, "rec": False}
_orig_ab = ACRoPEAttention._mem_ab


def _ab(self, x_in, nblk, S_):
    al, be = _orig_ab(self, x_in, nblk, S_)
    if FLAG["rec"]:
        ST[self._lid]["alpha"].append(al.detach().flatten(0, 0))           # (B, H, nblk)
        ST[self._lid]["beta"].append(be.detach().mean().item())
    return al, be


def _mix(self, x_in, o_own, o_mem):
    B_, N_, C_ = x_in.shape
    g = torch.sigmoid(self.mem_gate(x_in))
    o1 = o_own.transpose(1, 2).reshape(B_, N_, C_)
    o2 = o_mem.transpose(1, 2).reshape(B_, N_, C_).to(o1.dtype)
    p1, p2 = g[..., :C_] * o1, g[..., C_:] * o2
    if FLAG["rec"] and o2.abs().sum() > 0:                       # 빈 기억 (첫 블록) 은 뺀다
        d = ST[self._lid]
        d["own"].append(p1.norm(dim=-1).mean().item()); d["mem"].append(p2.norm(dim=-1).mean().item())
        d["g_own"].append(g[..., :C_].mean().item()); d["g_mem"].append(g[..., C_:].mean().item())
        d["own_raw"].append(o1.norm(dim=-1).mean().item()); d["mem_raw"].append(o2.norm(dim=-1).mean().item())
    if FLAG["zero_mem"]:
        p2 = p2 * 0
    return self.proj_drop(self.proj(p1 + p2))


ACRoPEAttention._mem_ab = _ab
ACRoPEAttention._mem_mix = _mix


def run():
    with torch.no_grad():
        p_tf = pred.forward_seq(h[:, :-S], idx[:, :-S])           # 위치 b = 블록 b+1 예측
        p_ar = pred.rollout_kv(h[:, : C * S], idx[:, : C * S], K)
    return p_tf, p_ar


FLAG["rec"] = True
p_tf, p_ar = run()
FLAG["rec"] = False
FLAG["zero_mem"] = True
q_tf, q_ar = run()
FLAG["zero_mem"] = False

tgt_tf, tgt_ar = h[:, S:], h[:, C * S:]
l1 = lambda p, t: (p - t).abs().mean().item()
res = {"ckpt": ckpt, "epoch": ck.get("epoch"), "n_clips": len(rows), "layers": {}, "ablation": {}}
print("\n층 | 자기 블록 몫 ‖σ·O_own‖ | 기억 몫 ‖σ·O_mem‖ | 기억/자기 | g_own | g_mem | ‖O_mem‖ 원값 | α 평균 (최소–최대 head) | α^8 | β")
for i, att in lids.items():
    d = ST[i]
    al = torch.cat([t.reshape(-1, t.size(1), t.size(2)).permute(1, 0, 2).reshape(t.size(1), -1) for t in d["alpha"]], 1)  # (H, ·)
    ah = al.mean(1)
    m = lambda k: sum(d[k]) / max(1, len(d[k]))
    row = dict(own=m("own"), mem=m("mem"), ratio=m("mem") / max(1e-9, m("own")), g_own=m("g_own"), g_mem=m("g_mem"),
               mem_raw=m("mem_raw"), own_raw=m("own_raw"), alpha=ah.mean().item(), alpha_min=ah.min().item(),
               alpha_max=ah.max().item(), keep8=(ah ** 8).mean().item(), beta=m("beta"))
    res["layers"][i] = row
    print(f"{i:2d} | {row['own']:.3f} | {row['mem']:.3f} | {row['ratio']:.2f} | {row['g_own']:.2f} | {row['g_mem']:.2f} | "
          f"{row['mem_raw']:.3f} | {row['alpha']:.3f} ({row['alpha_min']:.3f}–{row['alpha_max']:.3f}) | {row['keep8']:.3f} | {row['beta']:.2f}")
for name, p, q, t in (("TF (관측 입력)", p_tf, q_tf, tgt_tf), ("rollout (문맥 8 → 미래 8)", p_ar, q_ar, tgt_ar)):
    r = dict(l1=l1(p, t), l1_nomem=l1(q, t), dp=(p - q).abs().mean().item())
    if name.startswith("rollout"):
        r["l1_by_step"] = [l1(p[:, k * S:(k + 1) * S], t[:, k * S:(k + 1) * S]) for k in range(K)]
        r["l1_nomem_by_step"] = [l1(q[:, k * S:(k + 1) * S], t[:, k * S:(k + 1) * S]) for k in range(K)]
    res["ablation"][name] = r
    print(f"\n{name}: 타깃 L1 {r['l1']:.4f} → 기억 끔 {r['l1_nomem']:.4f} ({(r['l1_nomem'] / r['l1'] - 1) * 100:+.1f} %) · 예측 변화 mean|Δp| {r['dp']:.4f}")
    if "l1_by_step" in r:
        print("   미래 걸음별 L1 (켬 / 끔):", " ".join(f"{u:.3f}/{v:.3f}" for u, v in zip(r["l1_by_step"], r["l1_nomem_by_step"])))
os.makedirs(a.out, exist_ok=True)
fn = f"{a.out}/mem_diag_e{ck.get('epoch')}.json"
json.dump(res, open(fn, "w"), indent=1)
print(f"\n→ {fn} · {time.time() - t0:.0f}s")
