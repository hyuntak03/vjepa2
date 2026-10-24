#!/usr/bin/env python3
"""[T] 반사실 쌍으로 블록 AR 이어 학습 (틀 안 · 곁가지 없음 · 토큰 L1 그대로) — 사전 등록 진행판 [T] KST 06:1x (auto_research, 2026-10-22).
z_training · app · src 수정 없음 (import 만).

  CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/cf_ft.py --name cf_b --arm b [--steps 3000]

매 step: 자연 영상 8 창 (학습 캐시, 블록 8) + 팔에 따라
  a: 없음 (대조, 같은 step · 학습률)
  b: 반사실 train 2 창 (자연의 25 %)
  c: 반사실 train 2 창 + 일반 3D 합성 2 창 (도메인 단서 대조: 같은 렌더 계열 · 장애물 규칙 없음)
반사실 창 = clip 블록 0..7 (사건 장 중앙값 22 → 블록 3–4 에 사건; 사건 전 문맥 3–4 블록을 창 앞쪽에 둔다).
  장애물 있음 (W_pos) · 없음 (N_pos) 가능 변형만 (train 에는 불가능 변형이 없다).
일반 3D 합성 창 = phys3d 학습 분할 인코딩 (블록 8, 4 fps) 중 외형 증강 없는 판 (synth_aux, stable = 1).
손실 = 모든 창의 TF 한 걸음 토큰 L1 평균 (창 종류별 평균을 더하지 않고, 기준 AR 학습과 같은 꼴로 블록 토큰 L1 의 창 가중 평균).
저장: /workspace/auto_research_cache/cf_ft/<name>.pt (predictor state_dict + args)."""
import argparse, glob, os, sys, time
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--arm", required=True, choices=["a", "b", "c"])
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--bs_cf", type=int, default=2)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--nb", type=int, default=8)
ap.add_argument("--cache", default="/workspace/auto_research_cache/natural_train")
ap.add_argument("--cf", default="/workspace/auto_research_cache/phys3d_cf_v1")
ap.add_argument("--synth", default="/workspace/auto_research_cache/synth_aux")
ap.add_argument("--out", default="/workspace/auto_research_cache/cf_ft")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(__file__))
_argv = sys.argv; sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                     # noqa: E402
sys.argv = _argv
torch.set_num_threads(8)
dev = torch.device("cuda"); S = 256


def load_cf():
    import json
    H = np.load(os.path.join(a.cf, "train_h.npy"), mmap_mode="r")
    M = json.load(open(os.path.join(a.cf, "train_meta.json")))
    rows = [m["row"] for m in M if m["variant"] in ("W_pos", "N_pos")]
    return H, np.array(rows)


if __name__ == "__main__":
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    pred = FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev); pred.train()
    nat = [np.load(f, mmap_mode="r") for f in sorted(glob.glob(os.path.join(a.cache, "*_h.npy")))]; nsz = [x.shape[0] for x in nat]
    CFH, cf_rows = load_cf() if a.arm in ("b", "c") else (None, None)
    if a.arm == "c":
        SY = np.load(os.path.join(a.synth, "synth_h.npy"), mmap_mode="r")
        sy_rows = np.where(np.load(os.path.join(a.synth, "synth_lab.npz"))["stable"].astype(bool))[0]
    opt = torch.optim.AdamW(pred.parameters(), lr=a.lr, weight_decay=0.0)
    rng = np.random.default_rng(a.seed)

    def tf_l1(x):                                                           # x (B, nb, S, D) → 창마다 TF 한 걸음 L1 (B,)
        B = x.size(0)
        xs = x.reshape(B, -1, x.size(-1))
        idx = torch.arange(xs.size(1), device=dev).unsqueeze(0).expand(B, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = pred.forward_seq(xs, idx).float()[:, :-S]
        return (p - x[:, 1:].reshape(B, -1, x.size(-1))).abs().mean((1, 2))

    t0 = time.time()
    for it in range(1, a.steps + 1):
        si = rng.integers(len(nat), size=a.bs)
        parts = [torch.from_numpy(np.stack([nat[s][rng.integers(nsz[s])] for s in si])).to(dev).float()]
        kinds = ["nat"]
        if CFH is not None:
            ci = rng.choice(cf_rows, a.bs_cf, replace=False)
            parts.append(torch.from_numpy(np.stack([np.asarray(CFH[r, :a.nb], np.float32) for r in ci])).to(dev)); kinds.append("cf")
        if a.arm == "c":
            yi = rng.choice(sy_rows, a.bs_cf, replace=False)
            parts.append(torch.from_numpy(np.stack([np.asarray(SY[r, :a.nb], np.float32) for r in yi])).to(dev)); kinds.append("syn")
        ls = [tf_l1(x) for x in parts]
        loss = torch.cat(ls).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0); opt.step()
        if it % 100 == 0 or it == 1:
            print(f"[{a.name}] {it} " + " · ".join(f"{k} L1 {float(l.mean()):.4f}" for k, l in zip(kinds, ls)) + f" ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save({"predictor": pred.state_dict(), "args": vars(a), "kind": "cf_ft"}, os.path.join(a.out, f"{a.name}.pt"))
    print("saved", flush=True)
