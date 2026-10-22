#!/usr/bin/env python
"""궤적 보조 손실의 **상한**을 잰다 (2026-10-22, 사용자 지시 "해보던가").

왜: `ar_vw_shortctx_traj` 에서 궤적 head 가 세 지평선 모두 0.97 (0 예측 = 1.00) 에 멈췄다.
    "predictor 가 못 담았다" 인지 "애초에 선형으로 읽을 수 있는 게 없다" 인지 가려야 한다.
    CLAUDE.md §12 Q8 이 요구하는 **encoder 대조** 다 — 학습 때 빠뜨렸다.

무엇을: 학습과 **똑같은 head · 똑같은 손실**을 세 입력에 각각 붙여 held-out 상대 L1 을 잰다.
    h      타깃 encoder 의 **진짜** 블록 토큰 LN(h_b)  → 상한.
           완벽한 predictor 는 p_b = h_b 를 내므로, 어떤 predictor 도 이보다 잘할 수 없다.
    p      지금 학습 중인 predictor 의 TF 출력 p_b (latest.pt)  → 학습 로그 재현 검사
    hC     마지막 **문맥** 블록 토큰 LN(h_{C-1})  → 파라미터 있는 "복사" 대조.
           미래 블록을 안 보고 문맥 끝만 보고 읽는다. h 가 이것보다 나아야 '미래 토큰에 정보가 있다'

읽는 법: 전부 "0 을 예측했을 때" 로 나눈 상대 L1. 1.00 = 0 예측과 같음.
    등속 외삽 (파라미터 0) 도 같은 held-out 에서 같이 낸다.
    ⚠️ 학습된 자다 — 값 하나로 결론 내지 말고 세 입력의 **차이**로 읽는다 (§7-3).
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

sys.path.insert(0, "/data/hyuntak/project/2026/2027_cvpr/vjepa2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="z_vll7_training/configs/ar_vw_shortctx_traj.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=int, default=260, help="rank 당 토큰 예산 (작게 — 학습이 VRAM 을 쓰고 있다)")
    ap.add_argument("--train-batches", type=int, default=240)
    ap.add_argument("--eval-batches", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--predictor", default="", help="latest.pt (주면 p 도 잰다)")
    args = ap.parse_args()

    import yaml
    from app.vjepa_frozen import ar as AR
    from app.vjepa_frozen import flow_head as FHM
    from app.vjepa_frozen import utils as U
    from app.vjepa_frozen.data import ClipTransform, normalize_clips
    from app.vjepa_frozen.var_window import VarWindowBatches

    import subprocess
    cfg = yaml.safe_load(subprocess.run(
        ["bash", "z_vll7_training/train.sh", args.config.split("/")[-1].replace(".yaml", "")],
        capture_output=True, text=True, env={**__import__("os").environ, "DRYRUN": "1"},
        cwd="/data/hyuntak/project/2026/2027_cvpr/vjepa2").stdout.split("--- DRYRUN: 병합된 config ---\n")[1])

    dev = torch.device(args.device)
    MD, D = cfg["model"], cfg["data"]
    VW = dict(D["var_window"]); VW["token_budget"] = args.budget
    FZ = [int(h) for h in VW["flow"]["horizons"]]
    SUP = [int(h) for h in (cfg["flow_head"].get("supervise") or FZ[1:])]
    si = [FZ.index(h) for h in SUP]
    tub = int(MD.get("tubelet_size", 2))

    state = U.load_base_checkpoint(MD["checkpoint"])
    _, tgt = U.build_frozen_encoders(MD, state, dev, torch.bfloat16, int(D["n_frames"]))
    S = (int(MD.get("img_size", 256)) // int(MD.get("patch_size", 16))) ** 2

    pred = None
    if args.predictor:
        pred = U.build_predictor(MD, tgt.embed_dim, None, dev, int(D["n_frames"]))
        ck = torch.load(args.predictor, map_location="cpu", weights_only=False)
        pred.load_state_dict(ck["predictor" if "predictor" in ck else "model"]); pred.eval().requires_grad_(False)
        print(f"predictor <- {args.predictor} (epoch {ck.get('epoch')})", flush=True)

    tf_ = ClipTransform(int(MD.get("img_size", 256)), hflip_p=(D.get("aug") or {}).get("hflip_p", 0.5),
                        random_resized_crop=(D.get("aug") or {}).get("random_resized_crop"),
                        square_crop=(D.get("aug") or {}).get("square_crop", "none"), out_uint8=True)
    ds = VarWindowBatches(VW, tf_, tub, seed=1234, rank=0)

    arms = {"h": FHM.TrajHead(tgt.embed_dim, n_h=len(SUP)).to(dev),
            "hC": FHM.TrajHead(tgt.embed_dim, n_h=len(SUP)).to(dev)}
    if pred is not None:
        arms["p"] = FHM.TrajHead(tgt.embed_dim, n_h=len(SUP)).to(dev)
    opts = {k: torch.optim.AdamW(v.parameters(), lr=args.lr, weight_decay=0.0) for k, v in arms.items()}

    def feats(step):
        """한 batch → {팔: (입력 (B, T*S, D), 라벨 (B, T, H, 2, g, g), 유효 (B, T, H, g, g))}"""
        clips, info = ds.make(step)
        C, nb = int(info["context_blocks"]), int(info["win_blocks"])
        if nb < 2:
            return None
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            h = AR.encode_blocks(tgt, normalize_clips(clips.to(dev)).to(torch.bfloat16), tub)   # (B, nb*S, Dm) LN
        FL = info["flow"].to(dev)                                                       # (B, nb, H_all, 2, g, g)
        VL = info["flow_valid"].to(dev)
        T = nb - 1
        y, v = FL[:, 1:1 + T][:, :, si], VL[:, 1:1 + T][:, :, si]
        # 등속 외삽 (파라미터 0): 비감독 첫 지평선 h0 의 **진짜** 변위를 비례로 늘린다
        cv = torch.stack([FL[:, 1:1 + T][:, :, 0] * (h_ / FZ[0]) for h_ in SUP], 2)
        out = {"h": (h[:, S:].float(), y, v),                                           # 블록 1..nb-1 의 진짜 토큰
               "hC": (h[:, (C - 1) * S: C * S].float().repeat(1, T, 1), y, v)}          # 문맥 끝 블록을 T 번
        if pred is not None:
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                idx = AR.block_indices(h.size(0), nb, S, dev)
                p_tf, _ = pred(h, ar={"idx": idx, "n_ctx_blocks": C, "rollout_steps": 1})
            out["p"] = (p_tf.float(), y, v)
        return out, cv

    def rel(pred_, tgt_, ok):
        w = ok.unsqueeze(3)
        num = ((pred_ - tgt_).abs() * w).sum(dim=(0, 1, 4, 5)).sum(-1)
        den = (w.sum(dim=(0, 1, 3, 4, 5)) * 2).clamp_min(1e-6)
        z = ((tgt_.abs() * w).sum(dim=(0, 1, 4, 5)).sum(-1) / den).clamp_min(1e-3)
        return num / den / z, den

    t0 = time.time()
    for st in range(args.train_batches):
        got = feats(st)
        if got is None:
            continue
        b, _cv = got
        for k, head in arms.items():
            x, y, v = b[k]
            r, _ = rel(head(x, S), y, v)
            opts[k].zero_grad(set_to_none=True); r.mean().backward(); opts[k].step()
        if (st + 1) % 40 == 0:
            print(f"  학습 {st+1}/{args.train_batches}  {time.time()-t0:.0f}s", flush=True)

    acc = {k: torch.zeros(len(SUP), device=dev) for k in list(arms) + ["등속", "0 예측"]}
    n = 0
    with torch.no_grad():
        for st in range(100000, 100000 + args.eval_batches):
            got = feats(st)
            if got is None:
                continue
            b, cv = got
            for k, head in arms.items():
                acc[k] += rel(head(b[k][0], S), b[k][1], b[k][2])[0]
            _, y, v = b["h"]
            acc["등속"] += rel(cv, y, v)[0]
            acc["0 예측"] += rel(torch.zeros_like(y), y, v)[0]
            n += 1
    NAME = {"h": "h  진짜 블록 토큰", "p": "p  predictor 출력", "hC": "hC 문맥 끝 토큰",
            "등속": "등속 (파라미터 0)", "0 예측": "0 예측"}
    print(f"\n=== held-out {n} batch · 상대 L1 (1.00 = 0 예측) ===")
    print(f"{'입력':<22}" + "".join(f"{str(h//2)+' 블록':>11}" for h in SUP))
    for k in list(arms) + ["등속", "0 예측"]:
        print(f"{NAME[k]:<22}" + "".join(f"{x:11.3f}" for x in (acc[k] / max(n, 1)).tolist()))
    print("\n읽는 법: h 가 상한이다 (완벽한 predictor 는 p = h 를 낸다).")
    print("  h ≈ 1.00  -> 선형으로 읽을 게 애초에 없다. 지금 실패는 설계 탓이다")
    print("  h << 1.00 -> 읽을 수 있다. p 와의 차이가 predictor 가 못 담은 양이다")
    print("  h vs hC   -> 미래 블록 토큰이 문맥 끝 토큰보다 더 아는가 (아니면 자가 자리 prior 를 외운 것)")


if __name__ == "__main__":
    main()
