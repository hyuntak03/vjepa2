#!/usr/bin/env python3
"""[S] 영상 문맥 모델의 추가 운반은 자연 영상 L1 에서도 이득인가 (학습 없음, forward 만).

  CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/ctxt_natural_l1.py [--limit N]

모델 (latest, 읽기만): 블록 AR = natural_ar_vw_state · 영상 문맥 = natural_ar_vw_state_ctxt_video. 같은 프레임 · 같은 정답 (블록별 LN(목표 encoder)).
창 = 16 장 = 블록 8 (K710 보류 val_min224, 학습 표와 겹침 0; 칸 흐름 캐시). 문맥 C = 4 블록 → 굴린 예측 K = 4 블록 (두 모델 모두 학습한 길이).
  블록 AR: 문맥 블록 토큰 → rollout_kv (상태 되먹임)
  영상 문맥: online encoder 를 클립에 문맥 마스크로 한 번 (채점 · 기존 측정과 같은 방식) → rollout_state_kv → render
칸 무리 (걸음 k 의 정답 블록 C+k): 움직이는 칸 = |U[C+k]| ≥ 0.5 칸 (블록 C+k−1 → C+k), 정지 칸 = |U[C+k]| < 0.1 이고 |U[C+k−1]| < 0.1.
지표 (칸 합의 비, 창 bootstrap):
  L1(예측, 정답) · 복사 (마지막 문맥 블록) 대비 · 문맥 평균 대비
  정답 쪽 사영 = Σ⟨p − h_{C−1}, h_{C+k} − h_{C−1}⟩ / Σ|h_{C+k} − h_{C−1}|²  (0 = 복사, 1 = 정답만큼)"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F, yaml
import decord

REPO = "/workspace/world_model/vjepa2"
CACHE = "/workspace/auto_research_cache"
ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--report", action="store_true")
ap.add_argument("--C", type=int, default=4); ap.add_argument("--K", type=int, default=4)
a = ap.parse_args()
OUTN = os.path.join(REPO, "auto_research/results/ctxt_natural_l1.npz")
S, G = 256, 16
KEYS = ("l1", "copy", "mean", "proj_num", "proj_den", "n")


def run():
    sys.path.insert(0, REPO)
    from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key
    from src.models.rollout_predictor import vit_prefix_predictor
    from app.vjepa_frozen import utils as U
    from app.vjepa_frozen import ar as AR
    torch.set_num_threads(8); dev = torch.device("cuda")
    CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    sd = _load_checkpoint(CK)
    tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    _load_state_dict(tgt, sd["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()
    onl = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=32, use_rope=True, uniform_power=False)
    _load_state_dict(onl, sd["encoder"], tag="o"); onl = onl.to(dev, torch.bfloat16).eval()
    del sd

    def load_pred(run):
        D = os.path.join(REPO, "z_training/runs", run)
        c = yaml.safe_load(open(os.path.join(D, "config.yaml"))); pc = c["model"]["predictor"]
        if pc.get("kind") == "ctx_state":
            m = U.build_predictor(c["model"], 1280, None, torch.device("cpu"), int(c["data"].get("n_frames", 64)))
        else:
            m = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384,
                                     depth=12, num_heads=12, num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
        w = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D, "latest.pt"), map_location="cpu", weights_only=False)["predictor"]).items()
             if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
        m.load_state_dict(w, strict=True); return m.to(dev).eval()
    ar, cx = load_pred("natural_ar_vw_state"), load_pred("natural_ar_vw_state_ctxt_video")
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
    C, K = a.C, a.K
    meta = json.load(open(f"{CACHE}/natural_holdout/k710_320_holdout_meta.json"))
    Uk = np.load(f"{CACHE}/natural_holdout/k710_320_holdout_U.npy").astype(np.float32)
    if a.limit:
        meta = meta[: a.limit]
    rec = {f"{m}|{g}": [] for m in ("ar", "cx") for g in ("mov", "sta")}
    SH = os.path.join(CACHE, "ctxt_natural"); os.makedirs(SH, exist_ok=True)          # [P] 와 공유: 두 모델 예측 (창, 걸음, 칸, 1280) fp16
    PA = np.lib.format.open_memmap(os.path.join(SH, f"ar_pred_C{C}K{K}.npy"), mode="w+", dtype=np.float16, shape=(len(meta), K, S, 1280))
    PC = np.lib.format.open_memmap(os.path.join(SH, f"cx_pred_C{C}K{K}.npy"), mode="w+", dtype=np.float16, shape=(len(meta), K, S, 1280))
    ok = np.zeros(len(meta), bool)
    t0 = time.time()
    for wi, mt in enumerate(meta):
        vr = decord.VideoReader(mt["path"], num_threads=1)
        idx = [mt["start"] + mt["fstp"] * k for k in range(16)]
        if idx[-1] >= len(vr):
            continue
        fr = vr.get_batch(idx).asnumpy()
        Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
        x = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round() / 255.0
        xn = ((x.permute(1, 0, 2, 3)[None]) - MEAN) / STD                                         # (1,3,16,H,W)
        mg = np.linalg.norm(Uk[wi].reshape(8, S, 2), axis=-1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hb = AR.encode_blocks(tgt, xn.to(torch.bfloat16), 2).float()[0].view(8, S, -1)
            idx_ctx = torch.arange(C * S, device=dev).unsqueeze(0)
            p_ar = ar.rollout_kv(hb[:C].reshape(1, C * S, -1), idx_ctx, K).float()[0].view(K, S, -1)
            z = onl(xn.to(torch.bfloat16), masks=[idx_ctx]).float()
            p_cx = cx.render(cx.rollout_state_kv(z, idx_ctx, K)).float()[0].view(K, S, -1)
        PA[wi] = p_ar.half().cpu().numpy(); PC[wi] = p_cx.half().cpu().numpy(); ok[wi] = True
        cp, mu = hb[C - 1], hb[:C].mean(0)
        for name, p in (("ar", p_ar), ("cx", p_cx)):
            per = {"mov": np.zeros((K, len(KEYS))), "sta": np.zeros((K, len(KEYS)))}
            for k in range(K):
                y = hb[C + k]
                gm = {"mov": mg[C + k] >= 0.5, "sta": (mg[C + k] < 0.1) & (mg[C + k - 1] < 0.1)}
                l1 = (p[k] - y).abs().mean(-1); lc = (cp - y).abs().mean(-1); lm = (mu - y).abs().mean(-1)
                num = ((p[k] - cp) * (y - cp)).sum(-1); den = ((y - cp) ** 2).sum(-1)
                for g, m in gm.items():
                    mm = torch.from_numpy(m).to(dev)
                    if mm.sum() == 0:
                        continue
                    per[g][k] += [float(l1[mm].sum()), float(lc[mm].sum()), float(lm[mm].sum()), float(num[mm].sum()), float(den[mm].sum()), float(mm.sum())]
            for g in per:
                rec[f"{name}|{g}"].append(per[g])
        if wi % 25 == 0:
            print(f"{wi}/{len(meta)} ({time.time() - t0:.0f}s)", flush=True)
    PA.flush(); PC.flush(); np.save(os.path.join(SH, f"ok_C{C}K{K}.npy"), ok)
    json.dump(dict(meta="natural_holdout/k710_320_holdout_meta.json (같은 순서, 블록 토큰 = natural_holdout/k710_320_holdout_h.npy)", C=C, K=K,
                   ar="natural_ar_vw_state latest rollout_kv", cx="natural_ar_vw_state_ctxt_video latest render(rollout_state_kv(online 문맥 마스크))"),
              open(os.path.join(SH, f"README_C{C}K{K}.json"), "w"), ensure_ascii=False)
    np.savez(OUTN, **{k: np.stack(v) for k, v in rec.items()})
    print("끝", flush=True)


def report():
    P = np.load(OUTN); rng = np.random.default_rng(0)
    out = {}
    for g in ("mov", "sta"):
        A, B = P[f"ar|{g}"], P[f"cx|{g}"]                                            # (W, K, 6)
        W = A.shape[0]

        def stats(X):
            s = X.sum(0)                                                            # (K, 6)
            return np.stack([s[:, 0] / s[:, 5], s[:, 0] / s[:, 1], s[:, 0] / s[:, 2], s[:, 3] / s[:, 4]], 1)   # (K, 4): L1 · /복사 · /평균 · 사영
        sa, sb = stats(A), stats(B)
        bs = []
        for _ in range(1000):
            ii = rng.integers(0, W, W)
            bs.append(np.stack([stats(A[ii]), stats(B[ii]), stats(B[ii]) / stats(A[ii])]))
        bs = np.stack(bs); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        names = ("L1", "L1/copy", "L1/mean", "proj")
        for k in range(A.shape[1]):
            for j, nm in enumerate(names):
                out[f"{g}|step{k + 1}|{nm}"] = dict(ar=round(float(sa[k, j]), 4), ar_ci=[round(float(lo[0, k, j]), 4), round(float(hi[0, k, j]), 4)],
                                                     cx=round(float(sb[k, j]), 4), cx_ci=[round(float(lo[1, k, j]), 4), round(float(hi[1, k, j]), 4)],
                                                     cx_over_ar=round(float(sb[k, j] / sa[k, j]), 4), ratio_ci=[round(float(lo[2, k, j]), 4), round(float(hi[2, k, j]), 4)],
                                                     n_cells=int(A[:, k, 5].sum()))
                v = out[f"{g}|step{k + 1}|{nm}"]
                print(f"{g} 걸음 {k + 1} {nm:8s} AR {v['ar']:.4f} [{v['ar_ci'][0]:.4f},{v['ar_ci'][1]:.4f}]  ctxt {v['cx']:.4f} [{v['cx_ci'][0]:.4f},{v['cx_ci'][1]:.4f}]  ctxt/AR {v['cx_over_ar']:.4f} [{v['ratio_ci'][0]:.4f},{v['ratio_ci'][1]:.4f}]  칸 {v['n_cells']}")
    json.dump(out, open(os.path.join(REPO, "auto_research/results/ctxt_natural_l1.json"), "w"), indent=1)


if __name__ == "__main__":
    report() if a.report else run()
