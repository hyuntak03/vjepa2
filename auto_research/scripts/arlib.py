"""auto_research 공용 도구 — frozen 릴리즈 V-JEPA 2 ViT-H 의 층별 forward.

한 번의 forward 로 다음을 같이 꺼낸다 (전부 표준 채점 `surprise_c16t32` 와 같은 수치 경로:
fp32 가중치 + autocast fp16, 문맥 16 장 = 튜블릿 8 개, 미래 16 장 = 튜블릿 8 개):

  z_l   문맥 encoder (online `encoder`) 블록 l 출력 = residual stream (norm 전), 문맥 토큰만   (B, 2048, 1280)
  z     문맥 encoder 최종 출력 (encoder.norm, affine) = predictor 입력                          (B, 2048, 1280)
  q_l   predictor 블록 l 출력 (norm 전), [문맥 2048 | 미래 2048] 정렬 순서                     (B, 4096, 384)
  p     predictor 출력 (predictor_norm → predictor_proj), 미래 토큰                             (B, 2048, 1280)
  h     LN(target_encoder(clip 32 장)) — 표준 채점의 표적 (affine-free LN)                      (B, 4096, 1280)
  lens  predictor 렌즈 p_l = predictor_proj(predictor_norm(q_l[미래])) — 파라미터 0 개 (모델 자신의 머리)

토큰 배치 (PatchEmbed3D): 튜블릿이 바깥, 그 안에 16×16 공간 패치.  index = t*256 + y*16 + x.
predictor 안에서는 문맥·미래 인덱스를 argsort 로 정렬하므로 문맥 (0..2047) 이 앞, 미래 (2048..4095) 가 뒤다.

검증 (validate_*): hook 을 단 forward 의 p 가 hook 없는 forward 와 비트 단위로 같은지, 렌즈 L11 == p 인지,
표준 채점 per_video_surprise 와 같은 surprise 가 나오는지.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.intphys2.model import build_from_config                     # noqa: E402
from evals.world_model_analysis.data import WMADataset                      # noqa: E402
from evals.analysis_vlm.occlusion_identity.forward import _context_target_indices  # noqa: E402

CFG_DIR = ROOT / "auto_research" / "configs"
G = 16            # 공간 패치 한 변
S = G * G         # 튜블릿당 토큰
TS = 2            # tubelet
N_ENC = 32        # ViT-H 블록 수
N_PRED = 12       # predictor 블록 수


def load_cfg(name: str) -> dict:
    """auto_research/configs/<name>.yaml (resolve.py 산출물)."""
    cfg = yaml.safe_load(open(CFG_DIR / f"{name}.yaml"))
    cfg.pop("probing", None)
    return cfg


def build(cfg: dict, device: torch.device):
    ds = WMADataset(cfg)
    bundle = build_from_config(cfg["model"], device)
    return ds, bundle


def autocast_ctx(cfg: dict):
    name = str(cfg["model"].get("autocast", "none")).lower()
    dt = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}.get(name)
    return (lambda: torch.autocast("cuda", dtype=dt)) if dt else contextlib.nullcontext


def ci_ti(B: int, ctx_frames: int, tgt_frames: int, device):
    return _context_target_indices(ctx_frames=ctx_frames, tgt_frames=tgt_frames, tubelet_size=TS,
                                   spatial_tokens=S, batch_size=B, device=device)


class Taps:
    """블록 출력 hook. reduce(x) 로 GPU 에서 줄인 뒤 보관한다 (기본: 그대로)."""

    def __init__(self, blocks, layers: Iterable[int], reduce=None):
        self.layers = sorted(set(int(l) for l in layers))
        self.reduce = reduce
        self.out: Dict[int, torch.Tensor] = {}
        self._h = [blocks[l].register_forward_hook(self._mk(l)) for l in self.layers]

    def _mk(self, l):
        def f(_m, _i, o):
            self.out[l] = self.reduce(o) if self.reduce is not None else o
        return f

    def close(self):
        for h in self._h:
            h.remove()


@torch.inference_mode()
def forward(bundle, clips: torch.Tensor, cfg: dict, *, ctx_frames: int = 16,
            enc_layers: Iterable[int] = (), pred_layers: Iterable[int] = (),
            enc_reduce=None, pred_reduce=None, want_h: bool = True, want_lens: bool = False,
            pred_kwargs: Optional[dict] = None) -> dict:
    """clips (B, 3, T, H, W) 정규화된 입력. 반환 dict 는 GPU 텐서 (호출자가 줄여서 CPU 로)."""
    device = bundle.device
    B, T = clips.size(0), clips.size(2)
    tgt_frames = T - ctx_frames
    ci, ti = ci_ti(B, ctx_frames, tgt_frames, device)
    x = clips.to(device=device, dtype=bundle.dtype)
    ac = autocast_ctx(cfg)
    out = {}
    et = Taps(bundle.context_encoder.blocks, enc_layers, enc_reduce) if enc_layers else None
    pl = sorted(set(pred_layers) | ({N_PRED - 1} if want_lens else set()))
    pt = Taps(bundle.predictor.predictor_blocks, pl, None) if pl else None
    try:
        with ac():
            z = bundle.context_encoder(x, masks=[ci])
            p = bundle.predictor(z, ci, ti, mask_index=int(cfg["surprise"].get("mask_index", 0)), **(pred_kwargs or {}))
            if want_h:
                h = bundle.target_encoder(x)
                h = F.layer_norm(h, (h.size(-1),))
                out["h"] = h
            if want_lens and pt is not None:
                pred = bundle.predictor
                n_ctx = ci.size(1)
                out["lens"] = {l: pred.predictor_proj(pred.predictor_norm(q[:, n_ctx:])) for l, q in pt.out.items()}
    finally:
        if et: et.close()
        if pt: pt.close()
    out["z"], out["p"] = z, p
    if et: out["z_layers"] = et.out
    if pt:
        want = set(pred_layers)
        out["q_layers"] = {l: (pred_reduce(q) if pred_reduce else q) for l, q in pt.out.items() if l in want}
    return out


def l1_surprise(p: torch.Tensor, h_future: torch.Tensor) -> torch.Tensor:
    """표준 채점과 같은 식: mean |p − LN(h)| (토큰·채널 평균) → (B,)."""
    return (p.float() - h_future.float()).abs().mean(dim=(1, 2))


def future_slice(ctx_frames: int = 16, tgt_frames: int = 16):
    a = (ctx_frames // TS) * S
    return slice(a, a + (tgt_frames // TS) * S)


def tok(t: int, y: int, x: int) -> int:
    return t * S + y * G + x


# ---------------------------------------------------------------- RollOut_v3 (주 데이터, 2026-09-24)
# 3,136 clip × 64 프레임 (30 fps, stride 1), 사건 f32. 창 = 문맥 [SPLIT−C, SPLIT) + 예측 [SPLIT, SPLIT+P).
# RollOutV3 와 같은 16 창 (C·P ∈ {4,8,16,32}). 분할점 SPLIT 은 기본 32 (ledge·wall 사건 = f32).
V3_ROOT = Path("/data2/local_datasets/world/world_analysis/RollOut_v3")
V3_INDEX = ROOT / "data_csv/rollout_v3/index_probe.csv"
V3_WINDOWS = [(c, p) for c in (4, 8, 16, 32) for p in (4, 8, 16, 32)]


def v3_cfg(C: int, P: int, split: int = 32) -> dict:
    """표준 채점과 같은 수치 경로 (fp32 + autocast fp16, mask_index 0) 의 v3 창 config."""
    base = load_cfg("surprise_c16t32__v11_vith")
    model = dict(base["model"], window_size=C + P)
    data = dict(root=str(V3_INDEX.parent), index_csv=V3_INDEX.name,
                frames_root=str(V3_ROOT / "Images"), frames_pattern="{file_name}/{frame:06d}.png",
                frames_start=split - C, frames_stride=1, n_frames=C + P, resolution=256,
                block_column="block_id", pair_column="pair_id", variant_column="variant",
                plausible_column="plausible", type_column="scenario")
    return dict(eval_name="world_model_analysis", data=data, model=model,
                surprise=dict(base["surprise"], context_length=C),
                features={"cache_dir": "/tmp"}, scoring=base.get("scoring", {}))


def v3_dataset(C: int, P: int, split: int = 32):
    assert 0 <= split - C and split + P <= 64, (C, P, split)
    return WMADataset(v3_cfg(C, P, split))


def build_bundle(cfg: dict, device, window: Optional[int] = None):
    """환경변수 ARLIB_PRED_CKPT 가 있으면 predictor 만 그 체크포인트로 바꾼다 (z_training 학습 run · Ariel;
    attention 규칙 kind 는 체크포인트의 arch.kind 를 로더가 읽는다 — analysis/intphys2/model.py)."""
    import os
    m = dict(cfg["model"])
    if window is not None:
        m["window_size"] = int(window)
    pc = os.environ.get("ARLIB_PRED_CKPT")
    if pc:
        m["predictor_checkpoint"] = pc
        print(f"[arlib] predictor <- {pc}", flush=True)
    return build_from_config(m, device)
