#!/usr/bin/env python3
"""[물리] 실험 1 — 기억 슬롯 (존재 · 위치 · 보임) 모듈: 합성 렌더러 정답으로만 학습, 두 팔 (A: 가림 동안 정답 포함 · P: 보일 때만 정답) (auto_research, 2026-10-22).

  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/mem_slot.py --arm A --steps 3000
  스모크: --smoke 1 (20 step · 검증 8 창)

블록 AR · 사용자 run · z_training · app 은 건드리지 않는다. 입력은 고정 목표 encoder 블록 토큰 (칸마다 LN, 캐시 그대로).
모델 (블록마다 되풀이, 슬롯 K = 4):
  사전 (예측)   : prior_k = s_k + MLP(s_k) → 슬롯끼리 self-attention 한 층 — 이 블록을 보기 전에 낸 예측
  읽기 (주소)   : q = W_q prior, 칸 키 · 값 = W(LN 토큰) + 칸 위치 임베딩. 슬롯 attention 꼴 (칸마다 슬롯 사이 softmax → 칸 합 1 로 정규화)
                  = 주소 a_k (256,) · 읽은 값 r_k = Σ_c a_k(c) v(c)
  갱신 (사후)   : s_k = GRU(r_k, prior_k)
  머리          : [존재 logit · 위치 (칸 단위 x, y) · 속도 (칸/블록) · 보임 logit] 을 사전 · 사후 둘 다에서
손실 (물체 1 개 데이터, 창마다 맞는 슬롯 하나 = 보이는 블록의 사후 위치 오차가 가장 작은 슬롯, 기울기 없음):
  존재 BCE (맞는 슬롯 1, 나머지 0) · 위치 Huber δ 0.5 칸 · 속도 Huber · 보임 BCE (맞는 슬롯) — 사전 · 사후 모두
  감독 블록: 팔 A = 모든 블록 (가려진 블록에도 "있음 · 위치") · 팔 P = 보이는 블록만 (가려진 블록은 어느 손실에도 안 들어감)
데이터 (--data):
  state_obs (기본, 리더 지시: [T] 관측용 판독 head 와 같은 캐시 · 라벨 로더 재사용) = /workspace/auto_research_cache/state_obs/phys3d_single_v1_{train,val}_h.npy
      + _lab.npz (사용자 pack_labels 48 칸; clip 마다 물체 하나 → 그 정체 칸 k). 학습 600 clip · 보류 300 clip × 블록 16.
      감독 블록: 팔 A = m_state (상태 있음 · 가려져도) · 팔 P = m_state ∧ ¬hidden. 보임 = m_state ∧ ¬hidden.
  synth (대비책, [T] 캐시가 없을 때만) = synth_aux (phys3d_v1 train, 창 800 × 블록 8) / 검증 = phys3d 검증 캐시 (창 358 × 블록 16, 블록 대표 = 둘째 장)
검증 지표 (사전 등록 1 번):
  가려진 블록 위치 오차 (칸) — 사후 · 사전 — vs 복사 기준 (마지막으로 보인 라벨 위치를 그대로)
  존재 AUROC "가려짐 vs 없음" = 맞는 슬롯의 가려진 블록 존재 확률 vs 나머지 슬롯의 존재 확률
  보이는 블록 위치 오차 · 보임 AUROC (참고)
저장: /workspace/auto_research_cache/mem_slot/<arm>[_seed].pt (모델 · 설정 · 검증 지표)
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

S, G, D_IN = 256, 16, 1280
SYN = "/workspace/auto_research_cache/synth_aux"
VAL = "/workspace/auto_research_cache/phys3d"
OUT = "/workspace/auto_research_cache/mem_slot"
SOBS = os.environ.get("MEM_SLOT_SOBS", "/workspace/auto_research_cache/state_obs")
DOM = "phys3d_single_v1"


class MemSlot(nn.Module):
    def __init__(self, K=4, D=256, heads=4):
        super().__init__()
        self.K, self.D = K, D
        self.inp = nn.Sequential(nn.LayerNorm(D_IN, elementwise_affine=False), nn.Linear(D_IN, D))
        self.cell = nn.Parameter(torch.randn(S, D) * 0.02)
        self.kv = nn.Linear(D, 2 * D)
        self.q = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D))
        self.slot0 = nn.Parameter(torch.randn(K, D) * 0.5)
        self.trans = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.mix = nn.TransformerEncoderLayer(D, heads, 2 * D, dropout=0.0, batch_first=True, norm_first=True)
        self.gru = nn.GRUCell(D, D)
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, 6))   # 존재 · x · y · vx · vy · 보임

    def heads(self, s):
        o = self.head(s)
        return dict(exist=o[..., 0], pos=o[..., 1:3] + G / 2, vel=o[..., 3:5], vis=o[..., 5])

    def forward(self, x, read_mask=None):
        """x (B, T, 256, 1280) → 블록마다 사전 · 사후 머리 출력, 주소 (B, T, K, 256).
        read_mask (B, T) bool: False 인 블록은 읽지 않는다 (굴리기 — 사전만으로 이어 감)."""
        B, T = x.shape[:2]
        s = self.slot0[None].expand(B, -1, -1).contiguous()
        pri, post, attn = [], [], []
        for t in range(T):
            p = s + self.trans(s) if t > 0 else s
            p = self.mix(p)
            f = self.inp(x[:, t].float()) + self.cell
            k, v = self.kv(f).chunk(2, -1)
            lg = torch.einsum("bkd,bcd->bkc", self.q(p), k) / self.D ** 0.5
            a = lg.softmax(1) + 1e-8                       # 칸마다 슬롯 사이 경쟁
            a = a / a.sum(-1, keepdim=True)                # 슬롯마다 칸 합 1 = 주소
            r = torch.einsum("bkc,bcd->bkd", a, v)
            s_new = self.gru(r.reshape(-1, self.D), p.reshape(-1, self.D)).view(B, self.K, self.D)
            if read_mask is not None:
                m = read_mask[:, t].view(B, 1, 1).float()
                s_new = m * s_new + (1 - m) * p
            s = s_new
            pri.append(self.heads(p)); post.append(self.heads(s)); attn.append(a)
        stack = lambda L: {k: torch.stack([d[k] for d in L], 1) for k in L[0]}
        return stack(pri), stack(post), torch.stack(attn, 1)


def match_slot(post, pos, sup):
    """창마다 맞는 슬롯: 감독 · 보이는 블록에서 사후 위치 오차 합이 가장 작은 슬롯 (기울기 없음). pos (B,T,2) 칸, sup (B,T) bool"""
    with torch.no_grad():
        err = (post["pos"] - pos[:, :, None]).abs().sum(-1)            # (B,T,K)
        err = (err * sup[:, :, None].float()).sum(1)                   # (B,K)
        return err.argmin(-1)


def losses(pri, post, pos, vis, sup, j):
    """pos (B,T,2) 칸 · vis (B,T) bool · sup (B,T) bool (손실을 거는 블록) · j (B,) 맞는 슬롯"""
    B, T, K = post["exist"].shape
    oh = F.one_hot(j, K).float()[:, None].expand(B, T, K)                  # 맞는 슬롯 1
    w = sup.float()
    vel = torch.zeros_like(pos); vel[:, 1:] = pos[:, 1:] - pos[:, :-1]
    wv = w.clone(); wv[:, 0] = 0; wv[:, 1:] = wv[:, 1:] * w[:, :-1]
    L = {}
    for nm, o in (("pri", pri), ("post", post)):
        tw = w if nm == "post" else torch.cat([torch.zeros_like(w[:, :1]), w[:, 1:]], 1)   # 사전 t=0 은 정보 없음
        L[f"{nm}_exist"] = (F.binary_cross_entropy_with_logits(o["exist"], oh, reduction="none").mean(-1) * tw).sum() / tw.sum().clamp_min(1)
        sel = lambda z: z.gather(2, j.view(B, 1, 1, *([1] * (z.dim() - 3))).expand(B, T, 1, *z.shape[3:])).squeeze(2)
        L[f"{nm}_pos"] = (F.huber_loss(sel(o["pos"]), pos, delta=0.5, reduction="none").sum(-1) * tw).sum() / tw.sum().clamp_min(1)
        L[f"{nm}_vel"] = (F.huber_loss(sel(o["vel"]), vel, delta=0.5, reduction="none").sum(-1) * wv).sum() / wv.sum().clamp_min(1)
        L[f"{nm}_vis"] = (F.binary_cross_entropy_with_logits(sel(o["vis"]), vis.float(), reduction="none") * tw).sum() / tw.sum().clamp_min(1)
    return L


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]); rk = allv.argsort().argsort() + 1
    return float((rk[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_state_obs(split):
    """[T] 캐시 → H (N,16,256,1280) mmap · POS (N,16,2) 칸 · VIS (N,16) · SUPA (N,16) (상태 있음)"""
    H = np.load(os.path.join(SOBS, f"{DOM}_{split}_h.npy"), mmap_mode="r")
    L = np.load(os.path.join(SOBS, f"{DOM}_{split}_lab.npz"))
    k = (L["seen"] | L["m_state"]).any(1).argmax(-1)                     # (N,) 물체 정체 칸
    ix = np.arange(len(k))
    ms, hid = L["m_state"][ix, :, k], L["hidden"][ix, :, k]              # (N,16)
    pos = L["pos"][ix, :, k] * G                                        # (N,16,2) 칸 (x, y)
    return H, pos.astype(np.float32), ms & ~hid, ms


def load_synth_val():
    H, P, V = [], [], []
    for sh in range(4):
        h = np.load(os.path.join(VAL, f"h_val_target_s{sh}.npy"), mmap_mode="r")
        lb = np.load(os.path.join(VAL, f"lab_val_s{sh}.npz"))
        H.append(h); P.append(lb["pos"][:, 1::2] / 16.0); V.append(~lb["hidden"][:, 1::2])   # 블록 대표 = 둘째 장
    P, V = np.concatenate(P), np.concatenate(V)
    return H, P, V, np.ones_like(V)


def load_data(src, smoke):
    if src == "state_obs":
        Ht, Pt, Vt, St = load_state_obs("train"); val = load_state_obs("val")
        n = 64 if smoke else len(Pt)
        X = torch.from_numpy(np.asarray(Ht[:n]).copy())
        return X, torch.from_numpy(Pt[:n]), torch.from_numpy(Vt[:n]), torch.from_numpy(St[:n]), ([val[0]], val[1], val[2], val[3])
    X = torch.from_numpy(np.load(os.path.join(SYN, "synth_h.npy"), mmap_mode="r")[: 64 if smoke else None].copy())
    lb = np.load(os.path.join(SYN, "synth_lab.npz"))
    P = torch.from_numpy(lb["pos"][: len(X)] / 16.0).float(); V = torch.from_numpy(lb["vis"][: len(X)])
    return X, P, V, torch.ones_like(V), load_synth_val()


@torch.no_grad()
def evaluate(model, dev, val, limit=0):
    model.eval()
    H, P, V, SA = val
    rows = dict(hid_post=[], hid_pri=[], hid_copy=[], vis_post=[], ex_hid=[], ex_other=[], vis_p=[], vis_y=[])
    i0 = 0
    for h in H:
        n = h.shape[0] if not limit else min(h.shape[0], max(1, limit // 4))
        for w in range(n):
            x = torch.from_numpy(np.asarray(h[w], dtype=np.float32))[None].to(dev)
            pri, post, _ = model(x)
            pos = P[i0 + w]; vis = V[i0 + w]; sa = SA[i0 + w]
            pp = post["pos"][0].cpu().numpy(); pr = pri["pos"][0].cpu().numpy()      # (T,K,2)
            if vis.sum() == 0:
                continue
            j = int(np.abs(pp - pos[:, None]).sum(-1)[vis].sum(0).argmin())              # 보이는 블록으로만 맞춤
            ex = torch.sigmoid(post["exist"][0]).cpu().numpy(); vp = torch.sigmoid(post["vis"][0, :, j]).cpu().numpy()
            last = None
            for t in range(pos.shape[0]):
                if vis[t]:
                    last = pos[t]; rows["vis_post"].append(np.linalg.norm(pp[t, j] - pos[t]))
                elif last is not None and sa[t]:
                    rows["hid_post"].append(np.linalg.norm(pp[t, j] - pos[t])); rows["hid_pri"].append(np.linalg.norm(pr[t, j] - pos[t]))
                    rows["hid_copy"].append(np.linalg.norm(last - pos[t])); rows["ex_hid"].append(ex[t, j])
                if sa[t]:
                    rows["ex_other"].extend([ex[t, k] for k in range(ex.shape[1]) if k != j])
                    rows["vis_p"].append(vp[t]); rows["vis_y"].append(bool(vis[t]))
        i0 += h.shape[0]
    model.train()
    vy = np.array(rows["vis_y"]); vpp = np.array(rows["vis_p"])
    m = lambda k: float(np.mean(rows[k])) if rows[k] else float("nan")
    return dict(hidden_err_post=m("hid_post"), hidden_err_prior=m("hid_pri"), hidden_err_copy=m("hid_copy"),
                hidden_ratio_post=m("hid_post") / max(m("hid_copy"), 1e-6), visible_err_post=m("vis_post"),
                exist_auroc_hidden_vs_none=auroc(rows["ex_hid"], rows["ex_other"]), vis_auroc=auroc(vpp[vy], vpp[~vy]),
                n_hidden=len(rows["hid_post"]), n_visible=len(rows["vis_post"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "P"], required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--data", choices=["state_obs", "synth"], default="state_obs" if os.path.exists(os.path.join(SOBS, f"{DOM}_val_lab.npz")) else "synth")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(8)
    dev = torch.device("cuda")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    X, POS, VIS, SUPA, VALD = load_data(a.data, a.smoke)                     # X (N,T,256,1280) fp16 RAM
    print(f"데이터 {a.data} {tuple(X.shape)} 읽기 {time.time() - t0:.0f} 초 · 상태 있는 블록 중 가려진 몫 {1 - (VIS & SUPA).float().sum() / SUPA.float().sum():.3f}", flush=True)
    model = MemSlot(K=a.K).to(dev)
    print(f"파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = 20 if a.smoke else a.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    hist = []
    for it in range(1, steps + 1):
        idx = torch.randint(0, len(X), (a.bs,))
        x = X[idx].to(dev, non_blocking=True); pos = POS[idx].to(dev); vis = VIS[idx].to(dev); sa = SUPA[idx].to(dev)
        sup = sa.clone() if a.arm == "A" else (vis & sa)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pri, post, _ = model(x)
        pri = {k: v.float() for k, v in pri.items()}; post = {k: v.float() for k, v in post.items()}
        j = match_slot(post, pos, vis & sup)
        L = losses(pri, post, pos, vis, sup, j)
        loss = sum(L.values())
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if it % 50 == 0 or it == 1 or a.smoke:
            print(f"[{a.arm}] {it}/{steps} loss {loss.item():.3f} " + " ".join(f"{k} {v.item():.3f}" for k, v in L.items())
                  + f" · {time.time() - t0:.0f} 초 · GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        if (it % a.eval_every == 0 or it == steps):
            ev = evaluate(model, dev, VALD, limit=8 if a.smoke else 0); ev["step"] = it; hist.append(ev)
            print(f"[{a.arm}] 검증 {it}: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in ev.items()}), flush=True)
    name = f"{a.arm}{'' if a.seed == 0 else f'_s{a.seed}'}{'_smoke' if a.smoke else ''}"
    torch.save(dict(model=model.state_dict(), args=vars(a), hist=hist), os.path.join(OUT, f"{name}.pt"))
    json.dump(dict(args=vars(a), hist=hist), open(os.path.join(os.path.dirname(__file__), f"../results/mem_slot_{name}.json"), "w"), indent=1)
    print(f"저장 {name} · 전체 {time.time() - t0:.0f} 초", flush=True)


if __name__ == "__main__":
    main()
