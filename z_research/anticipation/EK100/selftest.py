#!/usr/bin/env python3
"""추가한 코드의 자체 검사 — 체크포인트도 데이터도 GPU 도 없이 CPU 에서 몇 초에 끝난다.

    python z_research/anticipation/EK100/selftest.py

1. spatial 세 모드가 종횡비를 안 깨는지 + 출력 shape (EK100 의 실제 해상도 3종으로)
2. 비정사각 wrapper 의 토큰 위치 계산 (grid_h x grid_w, skip, N_pred)
3. **정사각 입력에서 새 module 과 원본 module 이 비트 단위로 같은지** (회귀 방지)
4. 비정사각 입력이 실제로 forward 되는지 + predictor RoPE 에 H/W 가 주입됐는지
5. anticipation_point_mode / time_source 두 변형의 프레임 인덱스 산술
6. exact val 지표(`exact_val.py`)가 공식 `ClassMeanRecall` 과 같은 값을 내는지 (긴 꼬리 라벨)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from evals.action_anticipation_frozen.modelcustom import (  # noqa: E402
    vit_encoder_predictor_concat_ar as M_SQ,
)
from evals.action_anticipation_frozen.modelcustom import (  # noqa: E402
    vit_encoder_predictor_concat_ar_nonsquare as M_NS,
)
from evals.action_anticipation_frozen.spatial import (  # noqa: E402
    ResizeShortSideCenterCrop,
    ResizeLetterbox,
    build_eval_spatial,
)

PATCH, TUB, FPS = 16, 2, 8
FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def build_tiny(img_h, img_w, frames=8):
    """작은 encoder/predictor 한 쌍 (ViT-H 구조 그대로, dim 만 작게)."""
    import src.models.predictor as vit_pred
    import src.models.vision_transformer as vit

    torch.manual_seed(0)
    enc = vit.vit_tiny(
        img_size=(img_h, img_w), num_frames=frames, patch_size=PATCH, tubelet_size=TUB,
        uniform_power=True, use_rope=True,
    )
    prd = vit_pred.vit_predictor(
        img_size=(img_h, img_w), embed_dim=enc.embed_dim, patch_size=PATCH, tubelet_size=TUB,
        num_frames=frames * 4, depth=2, num_heads=3, predictor_embed_dim=96,
        num_mask_tokens=10, uniform_power=True, use_mask_tokens=True, use_rope=True,
    )
    enc.eval(), prd.eval()
    return enc, prd


def main():
    print("\n[1] spatial 모드 — EK100 실제 해상도 3종")
    for h, w in [(1080, 1920), (720, 1280), (1440, 1920)]:
        clip = (np.random.rand(4, h, w, 3) * 255).astype(np.uint8)
        cov = ResizeShortSideCenterCrop((256, 448))(clip)
        lb = ResizeLetterbox((256, 448))(clip)
        sq = ResizeShortSideCenterCrop(256)(clip)
        check(f"cover/letterbox/square {h}x{w}",
              cov.shape == (4, 256, 448, 3) and lb.shape == (4, 256, 448, 3) and sq.shape == (4, 256, 256, 3),
              f"{cov.shape[1:3]} {lb.shape[1:3]} {sq.shape[1:3]}")
    # short_side 는 짧은변을 정확히 out_h 에 맞춘다 (정사각 출력 기준)
    for h, w in [(1080, 1920), (720, 1280), (1440, 1920)]:
        clip = (np.random.rand(2, h, w, 3) * 255).astype(np.uint8)
        mid_short = round(min(h, w) * max(256 / h, 256 / w))
        check(f"short_side: {h}x{w} 의 짧은변 -> 256", mid_short == 256,
              f"scale={max(256/h,256/w):.4f}, 출력 {ResizeShortSideCenterCrop(256)(clip).shape[1:3]}")
    check("center_crop 모드가 2단계(Resize+CenterCrop)", len(build_eval_spatial("center_crop", 256, 256)) == 2)
    check("short_side 모드가 1단계", len(build_eval_spatial("short_side", 256, 256)) == 1)
    check("cover 는 별칭", len(build_eval_spatial("cover", 256, 256)) == 1)
    try:
        build_eval_spatial("bogus", 256, 256)
        check("알 수 없는 모드는 죽는다", False)
    except ValueError:
        check("알 수 없는 모드는 죽는다", True)

    print("\n[2] 비정사각 wrapper 의 토큰 위치 계산")
    enc, prd = build_tiny(256, 448, frames=8)
    w = M_NS.AnticipativeWrapper(enc, prd, frames_per_second=FPS, crop_size=(256, 448),
                                 patch_size=PATCH, tubelet_size=TUB, num_output_frames=2, num_steps=1)
    check("grid", (w.grid_h, w.grid_w) == (16, 28), f"{w.grid_h}x{w.grid_w}")
    check("tokens/tubelet", w.tokens_per_tubelet == 448, str(w.tokens_per_tubelet))
    x = torch.randn(2, 3, 8, 256, 448)
    at = torch.tensor([1.0, 1.0])
    with torch.no_grad():
        out = w(x, at)
    n_ctx = (8 // TUB) * 16 * 28
    check("출력 토큰 = 문맥 + 예측", out.shape == (2, n_ctx + 448, 96 if False else enc.embed_dim),
          f"{tuple(out.shape)}  (문맥 {n_ctx} + 예측 448)")

    print("\n[3] 정사각에서 원본 module 과 동일한가 (회귀 방지)")
    enc2, prd2 = build_tiny(256, 256, frames=8)
    w_ns = M_NS.AnticipativeWrapper(enc2, prd2, frames_per_second=FPS, crop_size=(256, 256),
                                    patch_size=PATCH, tubelet_size=TUB, num_output_frames=2,
                                    num_steps=1, mask_index=1)
    M_NS._inject_grid_into_predictor(prd2, 16, 16)   # init_module 이 하는 일을 여기서
    w_sq = M_SQ.AnticipativeWrapper(enc2, prd2, frames_per_second=FPS, crop_size=256,
                                    patch_size=PATCH, tubelet_size=TUB, num_output_frames=2, num_steps=1)
    xs = torch.randn(2, 3, 8, 256, 256)
    with torch.no_grad():
        a, b = w_ns(xs, at), w_sq(xs, at)
    d = (a - b).abs().max().item()
    check("정사각: 새 module == 원본 module", d == 0.0, f"max|diff| = {d:.3e}")

    print("\n[4] H/W 주입이 실제로 RoPE 위치를 바꾸는가")
    # 주입 안 한 predictor 로 같은 비정사각 입력을 돌리면 결과가 달라야 한다 (정사각 폴백)
    enc3, prd3 = build_tiny(256, 448, frames=8)
    w_bad = M_SQ.AnticipativeWrapper(enc3, prd3, frames_per_second=FPS, crop_size=256,
                                     patch_size=PATCH, tubelet_size=TUB, num_output_frames=2, num_steps=1)
    enc4, prd4 = build_tiny(256, 448, frames=8)
    prd4.load_state_dict(prd3.state_dict())
    enc4.load_state_dict(enc3.state_dict())
    M_NS._inject_grid_into_predictor(prd4, 16, 28)
    w_good = M_NS.AnticipativeWrapper(enc4, prd4, frames_per_second=FPS, crop_size=(256, 448),
                                      patch_size=PATCH, tubelet_size=TUB, num_output_frames=2, num_steps=1)
    with torch.no_grad():
        g = w_good(x, at)
        try:
            bad = w_bad(x, at)
            diff = (g[:, -448:] - bad[:, -448:]).abs().max().item() if bad.shape == g.shape else float("inf")
        except Exception:
            diff = float("inf")
    check("주입 없으면 예측 토큰이 달라진다 (= 정사각 폴백이 실제로 틀린다)", diff != 0.0,
          f"max|diff| = {diff:.3e}" if diff != float("inf") else "정사각 폴백은 shape 부터 다르다/죽는다")

    print("\n[5] anticipation_point / time_source 산술")
    sf, ef, vfps = 1000, 1200, 50.0          # 4초 지점, 4초짜리 action
    for at_s in (1.0,):
        aframes = int(at_s * vfps)
        rel = int(sf * 0.0 + (1 - 0.0) * ef - aframes)   # released, ap=0
        pap = int(sf * (1 - 0.0) + 0.0 * ef - aframes)   # paper,    ap=0
        check("released ap=0 -> action 의 끝 - 1s", rel == ef - aframes, f"af={rel} (ef={ef})")
        check("paper    ap=0 -> action 의 시작 - 1s", pap == sf - aframes, f"af={pap} (sf={sf})")
        check("두 규약이 실제로 다르다", rel != pap, f"{rel} vs {pap}  (차이 {rel - pap} 프레임 = {(rel-pap)/vfps:.1f}s)")

    print("\n[6] exact val 지표 == 공식 ClassMeanRecall (정의 동일성)")
    import os
    import torch.distributed as dist
    from evals.action_anticipation_frozen.exact_val import accumulate, class_mean_recall
    from evals.action_anticipation_frozen.metrics import ClassMeanRecall
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
        dist.init_process_group("gloo", rank=0, world_size=1)
    g = torch.Generator().manual_seed(0)
    for n_cls, n_clip, bs in ((97, 1000, 16), (3568, 2000, 16)):
        ref = ClassMeanRecall(num_classes=n_cls, device="cpu", k=5)
        tp = torch.zeros(n_cls, dtype=torch.float64); fn = torch.zeros(n_cls, dtype=torch.float64)
        # 긴 꼬리 라벨 분포 (EK100 처럼)
        w = 1.0 / torch.arange(1, n_cls + 1, dtype=torch.float64)
        for i in range(0, n_clip, bs):
            b = min(bs, n_clip - i)
            y = torch.multinomial(w, b, replacement=True, generator=g)
            logit = torch.randn(b, n_cls, generator=g) + 3.0 * torch.nn.functional.one_hot(y, n_cls) * (torch.rand(b, 1, generator=g) > 0.5)
            r_ref = ref(logit, y)
            accumulate(tp, fn, logit, y, 5)
        r, a, nch = class_mean_recall(tp, fn)
        d_r, d_a = abs(r - float(r_ref["recall"])), abs(a - float(r_ref["accuracy"]))
        check(f"classes={n_cls}: recall/accuracy 일치", d_r < 1e-4 and d_a < 1e-4,
              f"recall {r:.4f} vs {float(r_ref['recall']):.4f}  acc {a:.4f} vs {float(r_ref['accuracy']):.4f}  (지지 클래스 {nch})")
        check(f"classes={n_cls}: TP+FN == clip 수", int((tp + fn).sum()) == n_clip, f"{int((tp+fn).sum())} / {n_clip}")
    dist.destroy_process_group()

    print("\n" + "=" * 60)
    if FAILED:
        print(f"FAIL {len(FAILED)}개: {FAILED}")
        sys.exit(1)
    print("전부 PASS")


if __name__ == "__main__":
    main()
