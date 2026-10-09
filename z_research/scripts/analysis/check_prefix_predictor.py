"""`kind: prefix` predictor 가 학습 때와 같은 attention 으로 도는지 확인한다.

시작점 `z_training/ARIEL_CHECKPOINTS.md` (옛 `z_training/ariel/README.md` 는 2026-09-27 폴더 정리 때 사라졌다).
구현 정본은 **Ariel 학습 코드** `src/models/rollout_predictor.py` (2026-09-23 수령, 2026-09-27 판으로 갱신),
우리 독립 복원은 `analysis/predictors/prefix_causal.py` (`kind: prefix_xcheck`, 대조 전용).

  python z_research/scripts/analysis/check_prefix_predictor.py \
      --ckpt z_training/ariel/block_causal_future_only_1e_5/latest.pt --device cuda:0

  ⚠️ 2026-09-27 — 예시의 옛 체크포인트 `block_causal_future_only_1e_5_ep_5` 는 지워졌다 (아래 "실측 2026-09-23" 은 옛 epoch
     체크포인트 시절의 기록이고 새 판으로 다시 재지 않았다). 지금 prefix 판은 `block_causal_future_only_1e_5/latest.pt` (epoch 45, arch.kind prefix). 이 검사는 prefix 전용이다 —
     kind=ar (`ar_future_1e_5`) 는 release 규약 forward 가 막혀 있어 여기서 돌지 않는다 (AR 검사는 `z_training/tests/ar_cpu_test.py`).

검사 (전부 파라미터 없음. 1·2 는 GPU 불필요)
  1. mask 의미   — 문맥↔문맥 전부 / 문맥→미래 0 / 미래→미래 block-causal
  2. **인과 불변**— 미래 slot 을 뒤에서 잘라내도 앞 slot 출력이 안 변한다.
                   full attention 이면 반드시 달라진다.
  3. **구현 대조**— 학습 경로(ariel prefix/split) 대 dense 마스크 대 우리 독립 복원.
                   그리고 `mask_mode=full` 이 릴리즈 predictor 와 같은지 (ACBlock == Block).
                   **실측 2026-09-23**: split≡dense 0.0 / 복원 3.4e-07 / full≡릴리즈 3.2e-07.
  4. kind 대조   — prefix vs full 출력 차 (마스크를 틀리면 얼마나 달라지나: 상대 **0.5 규모**.
                   난수 문맥이라 실행마다 0.48~0.57 로 흔들린다 — 크기만 읽을 것)
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from analysis.predictors import build as build_predictor                    # noqa: E402
from analysis.predictors.prefix_causal import prefix_causal_attn_mask       # noqa: E402
from src.models.rollout_predictor import build_prefix_mask                  # noqa: E402


def _geom(img, frames, enc_dim):
    return dict(img_size=(img, img), patch_size=16, num_frames=frames, tubelet_size=2,
                embed_dim=enc_dim, predictor_embed_dim=384, depth=12, num_heads=12,
                num_mask_tokens=10, use_rope=True, use_sdpa=True,
                use_silu=False, wide_silu=True)


def _make(kind, sd, dev, **g):
    kw = dict(g)
    if kind in ("default", "oneshot", "prefix_xcheck"):
        kw.update(use_mask_tokens=True, uniform_power=False)
    m = build_predictor(kind, **kw)
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict()}, strict=True)
    return m.to(dev).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frames", type=int, default=32, help="window = C + M (프레임)")
    ap.add_argument("--ctx-frames", type=int, default=16)
    ap.add_argument("--img", type=int, default=64, help="검사 1·2 용 축소 해상도")
    ap.add_argument("--img-full", type=int, default=256, help="검사 3·4 용 실제 해상도")
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ck["predictor"].items()}
    arch = ck.get("arch") or {}
    enc_dim = int(arch.get("encoder_embed_dim", 1280))
    print(f"ckpt {a.ckpt}\n  arch.kind = {arch.get('kind')!r} | encoder_embed_dim {enc_dim} | "
          f"epoch {ck.get('epoch')} step {ck.get('step')} train_loss {ck.get('loss'):.4f}")

    tub = 2
    # ── 1. mask 의미 (Ariel 의 build_prefix_mask 와 우리 것을 같이 본다) ─────────
    S = (a.img // 16) ** 2
    Tc, Tt = a.ctx_frames // tub, (a.frames - a.ctx_frames) // tub
    n_ctx, n_tgt, N = Tc * S, Tt * S, (Tc + Tt) * S
    idx = torch.arange(N).unsqueeze(0)
    is_ctx = torch.zeros(1, N, dtype=torch.bool); is_ctx[:, :n_ctx] = True
    m_ours = prefix_causal_attn_mask(idx, is_ctx, S)[0, 0]
    m_ariel = build_prefix_mask(idx[0] // S, Tc)
    slot = idx[0] // S
    c, t = slice(0, n_ctx), slice(n_ctx, None)
    print(f"\n[1] mask 의미  ({a.img}px -> slot 당 {S} 토큰 | 문맥 slot {Tc} / 미래 slot {Tt})")
    print(f"    문맥->문맥 전부 attend : {bool(m_ours[c, c].all())}")
    print(f"    문맥->미래 전부 차단   : {bool((~m_ours[c, t]).all())}")
    print(f"    미래->문맥 전부 attend : {bool(m_ours[t, c].all())}")
    print(f"    미래->미래 block-causal: "
          f"{bool((m_ours[t, t] == (slot[t].unsqueeze(0) <= slot[t].unsqueeze(1))).all())}")
    same_mask = bool((m_ours == m_ariel).all())
    print(f"    **Ariel build_prefix_mask 와 동일** : {same_mask}")

    # ── 2·3·4. 실제 해상도에서 출력 비교 ──────────────────────────────────────
    S = (a.img_full // 16) ** 2
    n_ctx, n_tgt = Tc * S, Tt * S
    g = _geom(a.img_full, a.frames, enc_dim)
    B = 2
    z = torch.randn(B, n_ctx, enc_dim, device=dev)
    ci = torch.arange(n_ctx, device=dev).unsqueeze(0).expand(B, -1).contiguous()

    def run(kind, n_slots, **kw):
        m = _make(kind, sd, dev, **g, **kw)
        ti = torch.arange(n_ctx, n_ctx + n_slots * S, device=dev).unsqueeze(0).expand(B, -1).contiguous()
        with torch.inference_mode():
            o = m(z, ci, ti, mask_index=0).float().cpu()
        del m
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return o

    print("\n[2] 인과 불변 (미래 slot 을 Tt -> Tt//2 로 줄였을 때 앞쪽 출력 변화)")
    half = max(1, Tt // 2)
    for kind in ("prefix", "default"):
        full, short = run(kind, Tt), run(kind, half)
        d = (full[:, : half * S] - short).abs().max().item()
        scale = full[:, : half * S].abs().mean().item()
        # fp32 SDPA 는 시퀀스 길이가 바뀌면 reduction 순서가 바뀌어 1e-6 규모가 남는다.
        # 그건 "불변" 이다. full attention 은 상대 1 규모로 달라지므로 구분이 명확하다.
        print(f"    {kind:14s} max|Δ| = {d:.3e}  (상대 {d/scale:.4f})   "
              + ("불변 -> block-causal 확인" if d / scale < 1e-4 else "변함 -> full attention"))

    print("\n[3] 구현 대조 — 기준 = prefix/split (학습 때 실제 경로)")
    ref = run("prefix", Tt, prefix_impl="split")
    scale = ref.abs().mean().item()
    for label, kind, kw in [("prefix / dense 마스크", "prefix", {"prefix_impl": "dense"}),
                            ("우리 독립 복원 (xcheck)", "prefix_xcheck", {})]:
        d = (run(kind, Tt, **kw) - ref).abs()
        print(f"    {label:26s} max|Δ| = {d.max().item():.3e}  상대 = {d.mean().item()/scale:.3e}")
    rel = run("default", Tt)                       # 릴리즈 predictor (full self-attention)
    # Ariel 판을 mask_mode=full 로 돌려 릴리즈와 같은지 본다 (= ACBlock 이 Block 과 같은가)
    ar_full = _make("prefix", sd, dev, **g)
    ar_full.mask_mode = "full"
    ti = torch.arange(n_ctx, n_ctx + Tt * S, device=dev).unsqueeze(0).expand(B, -1).contiguous()
    with torch.inference_mode():
        of = ar_full(z, ci, ti, mask_index=0).float().cpu()
    d = (of - rel).abs()
    print(f"    {'mask_mode=full vs 릴리즈':26s} max|Δ| = {d.max().item():.3e}  "
          f"상대 = {d.mean().item()/rel.abs().mean().item():.3e}   (ACBlock == Block)")

    print(f"\n[4] prefix vs 릴리즈(full) 출력 상대차 = "
          f"{(ref - rel).abs().mean().item()/rel.abs().mean().item():.4f}  "
          "(0 이면 mask 가 안 걸린 것이다)")
    print(f"    출력 shape {tuple(ref.shape)}  (기대 {(B, n_tgt, enc_dim)})")
    print("\n=> 구조 검사 " + ("통과" if same_mask else "실패 — 마스크가 Ariel 것과 다르다"))
    return 0 if same_mask else 1


if __name__ == "__main__":
    raise SystemExit(main())
