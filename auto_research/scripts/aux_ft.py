#!/usr/bin/env python3
"""[T] 2 순위 — 합성 데이터 + 보조 head (문맥 쪽 중간 은닉에서 블록 사이 움직임 · 외형 안정도) 이어 학습 (auto_research, 2026-10-21).
z_training · app · src 수정 없음 (import · hook 만).

  CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/aux_ft.py --name aux_w005 --w 0.05

매 step: 자연 영상 8 창 (학습 캐시) + 합성 2 창 (synth_aux: phys3d 학습 분할, 절반 무름 증강). 손실 = 자연 TF L1 + 합성 TF L1 + w · 보조.
보조 head: predictor 6 번째 층 출력 (B, nb·256, 384) 에서 TF 위치 b (= 블록 b 토큰, b = 1..7) 마다 **칸 256 개 전체 attentive 풀링**
  (점수 = 선형 384→1, softmax, 가중 평균) → 선형 384→3: (a) 블록 b−1 → b 물체 변위 (라벨, 칸 단위; Huber δ 0.25),
  (b) 외형 안정도 logit (단단함 1 / 무름 0; BCE). 물체가 두 블록 모두 보일 때만 (a), 보일 때 (b).
  latent 의 물체 정보는 칸 전체에 퍼져 있어 (물체 밖 95 %) 물체 칸만 모으지 않고 칸 전체에서 고르게 했다.
w = 0 이면 대조 (같은 섞기, 보조 없음). 저장: /workspace/auto_research_cache/ar_ft/<name>.pt"""
import argparse, glob, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--w", type=float, default=0.05)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--bs_syn", type=int, default=2)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--lr_head", type=float, default=1e-3)
ap.add_argument("--stab", type=int, default=1, help="0 = 외형 안정도 (b) 손실 끔 — 증강이 latent 에서 신뢰도 차이를 거의 안 만들어서 (synth_label_check)")
ap.add_argument("--layer", type=int, default=5, help="hook 할 predictor 층 (0 부터; 5 = 12 층 중 6 번째)")
ap.add_argument("--cache", default="/workspace/auto_research_cache/natural_train")
ap.add_argument("--synth", default="/workspace/auto_research_cache/synth_aux")
ap.add_argument("--out", default="/workspace/auto_research_cache/ar_ft")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(__file__))
_argv = sys.argv; sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                     # noqa: E402
sys.argv = _argv
torch.set_num_threads(8)
dev = torch.device("cuda"); S = 256


class AuxHead(nn.Module):
    def __init__(s):
        super().__init__()
        s.score = nn.Linear(384, 1); s.out = nn.Linear(384, 3)

    def forward(s, hid):                                                   # hid (B, nb, S, 384) → (B, nb, 3)
        att = torch.softmax(s.score(hid).squeeze(-1), dim=-1)
        return s.out((att[..., None] * hid).sum(2))


if __name__ == "__main__":
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    pred = FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev); pred.train()
    head = AuxHead().to(dev)
    HOOK = {}
    pred.predictor_blocks[a.layer].register_forward_hook(lambda m, i, o: HOOK.__setitem__("h", o))
    hfiles = sorted(glob.glob(os.path.join(a.cache, "*_h.npy")))
    nat = [np.load(f, mmap_mode="r") for f in hfiles]; nsz = [x.shape[0] for x in nat]
    SY = np.load(os.path.join(a.synth, "synth_h.npy"), mmap_mode="r"); LB = np.load(os.path.join(a.synth, "synth_lab.npz"))
    opt = torch.optim.AdamW([{"params": pred.parameters(), "lr": a.lr}, {"params": head.parameters(), "lr": a.lr_head}], weight_decay=0.0)
    rng = np.random.default_rng(a.seed)

    def tf(x):
        idx = torch.arange(x.size(1), device=dev).unsqueeze(0).expand(x.size(0), -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return pred.forward_seq(x, idx).float()

    t0 = time.time()
    for it in range(1, a.steps + 1):
        si = rng.integers(len(nat), size=a.bs)
        xn = torch.from_numpy(np.stack([nat[s][rng.integers(nsz[s])] for s in si])).to(dev).float()
        wi = rng.integers(SY.shape[0], size=a.bs_syn)
        xs = torch.from_numpy(np.stack([np.asarray(SY[w], np.float32) for w in wi])).to(dev)
        pn = tf(xn.view(a.bs, -1, 1280))[:, :-S]
        l_nat = (pn - xn[:, 1:].reshape(a.bs, -1, 1280)).abs().mean()
        ps = tf(xs.view(a.bs_syn, -1, 1280))
        hid = HOOK["h"].float().view(a.bs_syn, 8, S, -1)
        l_syn = (ps[:, :-S] - xs[:, 1:].reshape(a.bs_syn, -1, 1280)).abs().mean()
        loss = l_nat + l_syn
        l_aux = torch.zeros((), device=dev); disp_err = float("nan"); acc = float("nan")
        if a.w > 0:
            o = head(hid[:, 1:])                                           # 위치 b = 1..7
            pos = torch.from_numpy(LB["pos"][wi]).to(dev).float(); vis = torch.from_numpy(LB["vis"][wi]).to(dev)
            stab = torch.from_numpy(LB["stable"][wi]).to(dev).float()
            d = (pos[:, 1:] - pos[:, :-1]) / 16.0
            mv = (vis[:, 1:] & vis[:, :-1] & torch.isfinite(d).all(-1)).float()
            d = torch.nan_to_num(d)
            l_d = (F.huber_loss(o[..., :2], d, reduction="none", delta=0.25).sum(-1) * mv).sum() / mv.sum().clamp_min(1)
            vb = vis[:, 1:].float()
            l_s = (F.binary_cross_entropy_with_logits(o[..., 2], stab[:, None].expand(-1, 7), reduction="none") * vb).sum() / vb.sum().clamp_min(1)
            l_aux = l_d + (l_s if a.stab else 0.0 * l_s)
            loss = loss + a.w * l_aux
            disp_err = float((((o[..., :2] - d).norm(dim=-1)) * mv).sum() / mv.sum().clamp_min(1))
            acc = float((((o[..., 2] > 0).float() == stab[:, None]).float() * vb).sum() / vb.sum().clamp_min(1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0); opt.step()
        if it % 100 == 0 or it == 1:
            print(f"[{a.name}] {it} 자연 L1 {float(l_nat):.4f} · 합성 L1 {float(l_syn):.4f} · 보조 {float(l_aux):.3f} (변위 오차 {disp_err:.3f} 칸 · 안정도 정확도 {acc:.2f}) ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save({"predictor": pred.state_dict(), "aux_head": head.state_dict(), "args": vars(a), "kind": "aux"}, os.path.join(a.out, f"{a.name}.pt"))
    print("saved", flush=True)
