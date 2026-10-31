"""Cosmos-1.0-Autoregressive-4B 독립 구현 (analysis/cosmos_ar.py) 검증 — 학습 없음.

  (a) 실제 영상에서 teacher forcing 미래 NLL 이 균등 (ln 64000 = 11.07) 보다 훨씬 낮은가
  (b) 미래 프레임 순서를 섞거나 / 다른 영상의 미래로 바꾸면 미래 NLL 이 오르는가
  (c) 공식 cosmos_predict1 Transformer (megatron 등은 빈 모듈로 막고 import) 의 로짓과 숫자 대조
      - 전체 12800 토큰 teacher forcing 로짓
      - 공식 생성 경로의 한 단계: 9 프레임 문맥 (잠재 2 칸, 5120 토큰) prefill 의 마지막 로짓

결과: cvpr/results/eval/cosmos/validate_cosmos_ar4b/validate.json

python z_research/scripts/analysis/cosmos_ar_validate.py [--skip-official]
"""
import argparse
import glob
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from analysis.cosmos_ar import (AR_CKPT, CosmosTokenizer, frames_to_input, load_ar,  # noqa: E402
                                token_nll)

OFFICIAL_ROOT = os.path.join(os.path.dirname(REPO), "cosmos-predict1")
OUT = os.path.join(REPO, "cvpr/results/eval/cosmos/validate_cosmos_ar4b")
VOB = "/workspace/dataset/benchmarks/VOBench_v1"
IP2 = "/workspace/dataset/benchmarks/IntPhys2/Main"
K400 = "/workspace/dataset/training_data/K710_320/videos/k400"
UNIFORM = math.log(64000)


def load_vobench(rel):
    from PIL import Image
    idx = [0] + list(range(0, 96, 3))
    return np.stack([np.array(Image.open(f"{VOB}/{rel}/{i:06d}.png").convert("RGB")) for i in idx])


def load_mp4(path, stride, start=0, n=33):
    import decord
    vr = decord.VideoReader(path)
    idx = np.clip(np.arange(start, start + n * stride, stride), 0, len(vr) - 1)
    return vr.get_batch(idx).asnumpy()


def score(model, tk, frames, n_ctx=3):
    vid, valid = frames_to_input(frames, "pad")
    idx, agree = tk.encode(vid)
    tok = idx.reshape(1, -1).cuda()
    nll = token_nll(model, tok, tuple(idx.shape[1:]), n_ctx)[0]
    v = valid.cuda()
    # 문맥 NLL (잠재 1..n_ctx-1, 첫 토큰 제외) 도 같이 — 문맥 자체가 균등보다 낮은가
    nll_ctx = token_nll(model, tok, tuple(idx.shape[1:]), 1)[0][: n_ctx - 1]
    return {"future": float(nll[:, v].mean()), "future_per_latent": [float(nll[i][v].mean()) for i in range(nll.shape[0])],
            "ctx_latent1_2": float(nll_ctx[:, v].mean()), "tok_agree": agree}, tok


# ----------------------------------------------------------------------------- 공식 코드 대조
def _stub_modules():
    class _Any:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return None
        def __getattr__(self, n): return _Any()

    class _Mod(types.ModuleType):
        def __getattr__(self, n):
            if n.startswith("__"):
                raise AttributeError(n)
            return _Any

    for name in ["megatron", "megatron.core", "megatron.core.parallel_state", "megatron.core.tensor_parallel",
                 "megatron.core.tensor_parallel.mappings", "megatron.core.tensor_parallel.utils",
                 "omegaconf", "loguru", "loguru._logger"]:
        sys.modules[name] = _Mod(name)
    log = types.ModuleType("cosmos_predict1.utils.log")
    for fn in ["debug", "info", "warning", "error", "critical", "success"]:
        setattr(log, fn, lambda *a, **k: None)
    sys.modules["cosmos_predict1.utils.log"] = log
    sys.path.insert(0, OFFICIAL_ROOT)
    import cosmos_predict1.utils as cu
    cu.log = log


def official_logits(tokens, positions_full, prompt_len):
    """공식 Transformer 로 (전체 seq 로짓[positions_full], prompt prefill 마지막 로짓) 을 낸다."""
    _stub_modules()
    from cosmos_predict1.autoregressive.networks.transformer import Transformer
    params = dict(dim=4096, n_layers=16, n_heads=32, n_kv_heads=8, head_dim=None, vocab_size=64000,
                  ffn_hidden_size=14336, norm_eps=1e-5, norm_type="rmsnorm", rope_theta=500000,
                  apply_abs_pos_emb=False, max_batch_size=1, max_seq_len=12800, fuse_qkv=False,
                  causal_mask=True, precision="bfloat16", use_qk_normalization=True,
                  tensor_model_parallel_size=1, apply_yarn=False, yarn_scale=None, yarn_beta_fast=None,
                  yarn_beta_slow=None, original_seq_len=8192, rope_dim="3D", pytorch_rope_version="v2",
                  original_latent_shape=None, pad_to_multiple_of=64, insert_cross_attn=False,
                  insert_cross_attn_every_k_layers=1, context_dim=1024, num_video_frames=33,
                  video_latent_shape=[5, 40, 64])
    torch.set_default_device("cuda")
    with torch.device("cuda"):
        net = Transformer(params, tokenizer_config=None)
    torch.set_default_device("cpu")
    sd = torch.load(AR_CKPT, map_location="cpu", weights_only=True, mmap=True)
    sd = {k[len("model."):]: v for k, v in sd.items()}
    res = net.load_state_dict(sd, strict=True)
    net = net.cuda().eval()
    with torch.no_grad():
        L = tokens.shape[1]
        full = net(tokens, input_pos=torch.arange(L, device="cuda"))[0, positions_full].float().cpu()
        pre = net(tokens[:, :prompt_len], input_pos=torch.arange(prompt_len, device="cuda"))[0, -1].float().cpu()
    del net
    torch.cuda.empty_cache()
    return full, pre, {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-official", action="store_true")
    ap.add_argument("--only-official", action="store_true", help="(a)(b) 는 기존 validate.json 을 두고 (c) 만")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    rep = {"uniform_nll": UNIFORM}
    if args.only_official and os.path.exists(os.path.join(OUT, "validate.json")):
        rep = json.load(open(os.path.join(OUT, "validate.json")))

    model, res = load_ar()
    rep["strict_load"] = {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys)}
    tk = CosmosTokenizer()

    k400 = sorted(glob.glob(f"{K400}/*.mp4"))[:6]
    ip2 = sorted(glob.glob(f"{IP2}/Videos/*.mp4"))[:3]
    vob = ["primitive/permanence/late_k1/v11_flat_static_cube_purple/pos_a",
           "realistic/shape/late_k2/" + sorted(os.listdir(f"{VOB}/realistic/shape/late_k2"))[0] + "/pos_a",
           "primitive/color/late_k3/" + sorted(os.listdir(f"{VOB}/primitive/color/late_k3"))[0] + "/pos_a"]
    sets = {"k400_stride2": [load_mp4(p, 2) for p in k400],
            "intphys2_6fps": [load_mp4(p, 10, start=60) for p in ip2],
            "vobench_10fps": [load_vobench(r) for r in vob]}
    rep["videos"] = {"k400_stride2": k400, "intphys2_6fps": ip2, "vobench_10fps": vob}

    # (a) + (b)
    t0 = time.time()
    for name, vids in ([] if args.only_official else sets.items()):
        rows = []
        for i, fr in enumerate(vids):
            r0, tok0 = score(model, tk, fr)
            sh = fr.copy()
            perm = rng.permutation(np.arange(17, 33))
            sh[17:] = fr[perm]
            r_sh, _ = score(model, tk, sh)
            rev = fr.copy()
            rev[17:] = fr[17:][::-1]
            r_rev, _ = score(model, tk, rev)
            other = vids[(i + 1) % len(vids)]
            sw = fr.copy()
            o = other[17:]
            if o.shape[1:3] != fr.shape[1:3]:          # 해상도가 다르면 맞춘다
                o = torch.nn.functional.interpolate(torch.as_tensor(o).permute(0, 3, 1, 2).float(),
                                                    size=fr.shape[1:3], mode="bilinear", antialias=True)
                o = o.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
            sw[17:] = o
            r_sw, _ = score(model, tk, sw)
            fz = fr.copy()
            fz[17:] = fr[16]                              # 미래 = 문맥 마지막 프레임 정지
            r_fz, _ = score(model, tk, fz)
            fl = fr.copy()
            fl[17:] = fr[17:, :, ::-1]                    # 미래만 좌우 반전 (내용 같고 배치가 문맥과 어긋남)
            r_fl, _ = score(model, tk, fl)
            rows.append({"orig": r0, "shuffled_future": r_sh, "reversed_future": r_rev, "swapped_future": r_sw,
                         "frozen_future": r_fz, "hflip_future": r_fl})
            print(name, i, f"orig {r0['future']:.3f} ctx {r0['ctx_latent1_2']:.3f} | shuf {r_sh['future']:.3f} "
                  f"rev {r_rev['future']:.3f} swap {r_sw['future']:.3f} frozen {r_fz['future']:.3f} "
                  f"hflip {r_fl['future']:.3f}", flush=True)
        rep[name] = rows
        rep[name + "_mean"] = {k: float(np.mean([r[k]["future"] for r in rows]))
                               for k in ["orig", "shuffled_future", "reversed_future", "swapped_future",
                                         "frozen_future", "hflip_future"]}
    if not args.only_official:
        rep["ab_seconds"] = time.time() - t0

    # (c) 공식 대조 — K400 첫 영상 토큰
    if not args.skip_official:
        _, tok = score(model, tk, sets["k400_stride2"][0])
        S = 40 * 64
        pos_full = torch.arange(3 * S - 1, 5 * S - 1)                 # 미래 2 잠재 칸을 예측하는 위치
        with torch.no_grad():
            mine_full = model(tok, (5, 40, 64))[0, pos_full.cuda()].float().cpu()
            mine_pre = model(tok[:, :2 * S], (5, 40, 64))[0, -1].float().cpu()
        tgt = tok[0, pos_full + 1].cpu()
        del model
        torch.cuda.empty_cache()
        off_full, off_pre, off_res = official_logits(tok, pos_full.cuda(), 2 * S)

        def nll(lg):
            return float(-torch.log_softmax(lg, -1).gather(-1, tgt[:, None]).mean())
        d = (mine_full - off_full).abs()
        rep["official"] = {
            "official_strict_load": off_res,
            "full_max_abs_diff": float(d.max()), "full_mean_abs_diff": float(d.mean()),
            "full_logit_std": float(off_full.std()),
            "full_argmax_agree": float((mine_full.argmax(-1) == off_full.argmax(-1)).float().mean()),
            "future_nll_mine": nll(mine_full), "future_nll_official": nll(off_full),
            "prefill_last_max_abs_diff": float((mine_pre - off_pre).abs().max()),
            "prefill_last_argmax_same": bool(mine_pre.argmax() == off_pre.argmax()),
            "prefill_last_top5_mine": mine_pre.topk(5).indices.tolist(),
            "prefill_last_top5_official": off_pre.topk(5).indices.tolist(),
        }
        print(json.dumps(rep["official"], indent=1))
    json.dump(rep, open(os.path.join(OUT, "validate.json"), "w"), indent=1)
    print({k: v for k, v in rep.items() if k.endswith("_mean")})


if __name__ == "__main__":
    main()
