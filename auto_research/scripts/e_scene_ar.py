#!/usr/bin/env python3
"""I-scene 판독을 자기회귀 predictor (Ariel 팔 C, kind=ar) 에 건다 — 2026-09-27.

    python e_scene_ar.py --task reach --source pan|ssv2|ek100|v3 --out <dir> [--gpus N] [--n 800] [--limit K]
    python e_scene_ar.py --task perm  --source pan --out <dir>            (화면 고정 가림 띠, e_scene_permanence 와 같은 설계)

predictor (모두 frozen ViT-H 위):
  R  = 릴리즈 (표준 경로: z = online encoder(문맥 16 장), mask token → 미래 8 슬롯 한 번에)
  F  = Ariel full-attention ep40   (표준 경로, kind oneshot)    ─┐ 같은 데이터 · 예산, 규칙만 다르다
  B  = Ariel prefix ep45           (표준 경로, kind prefix)     ─┘
  AR = Ariel 자기회귀 ep18/40      (kind ar: 입력 = 타깃 = LN(target_encoder(블록 2 장)) 블록별, mask token 없음)

AR 의 두 팔:
  AR:roll = rollout(h_blk[0..7], 8)        자기 예측을 되먹여 8 블록   ← 주 팔 (closure 가 구조로 들어 있다)
  AR:tf   = forward_seq(h_blk[0..14])[7..] 진짜 블록을 되먹인 한 스텝 앞 예측 (ar_z 의 AR 판; 쌍 안에서 다르다 — 판독 전용)

판독 (파라미터 0, e_scene_reach 와 같다): ĉ = argmax_r cos(p_i[s], T[r]), 적중 = Chebyshev(ĉ, u) ≤ 1.
  템플릿 T: 표준 predictor 는 hc_L = LN(target_encoder(문맥 16 장))[튜블릿 7] (기존과 같다).
           AR 은 T_ar = h_blk[7] = LN(target_encoder(프레임 14–15 만)) — AR 이 실제로 입력으로 먹은 마지막 관측.
           대조로 AR:roll 을 hc_L 로도 읽는다 (AR:roll_x).
  encoder 천장 (합의 토큰): enc = readout(h[8..15] (창 전체 인코딩), hc_L)  ← 표준 팔용
                            enc_ar = readout(h_blk[8..15], T_ar)             ← AR 팔용 (meta.enc_for 가 팔마다 어느 것인지 적는다)
저장 (e_scene_reach{,_nat} 와 같은 이름): arm_c/arm_ct/arm_cc (n, |arms|, 8, 256), enc · enc_ar (n, 3, 8, 256), 자연 영상은 src/disp/srccv.
정밀도: 표준 팔 = fp32 가중치 + fp16 autocast (채점 관례), AR = bf16 autocast (학습과 같다; fp16 넘침 방지) — 판독은 fp32.
⚠️ AR 은 12 fps · SSv2+K400 으로 학습했다. 팬 (인공) · SSv2 (균등 32 장) · EK100 (≈15 fps) 는 다른 팔과 같은 입력이다 (비교 조건 동일).
⚠️ AR ep18 은 40 epoch 중 18 (학습 중). 끝나면 같은 스크립트로 다시 건다.
환경: SCENE_SSV2=<dir> 로 SSv2 클립 위치를 바꿀 수 있다 (vll2 에 tar 로 옮긴 판). SCENE_PAN_LIST 는 e_scene_reach 와 같다.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arlib  # noqa: E402
from h4_single_query_v3 import idx_range  # noqa: E402
import e_scene_reach as ES  # noqa: E402
import e_scene_reach_nat as EN  # noqa: E402
import e_scene_permanence as EP  # noqa: E402
from app.vjepa_frozen.ar import encode_blocks, block_indices  # noqa: E402

S_TOK, D, NF, TC, G = 256, 1280, 32, 8, 16
Z = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/ariel")
CKPT = {"F": Z / "full_attn_future_pred_1e_5/latest.pt", "B": Z / "block_causal_future_only_1e_5/latest.pt",
        "AR": Path(os.environ.get("SCENE_AR_CKPT", str(Z / "ar_future_1e_5/latest.pt")))}   # SCENE_AR_CKPT: 고정 사본 (예: z_training/ariel_eval/ckpt/ar_ep38.pt); latest.pt 는 학습 중 덮인다


def ckpt_epoch(path):
    try:
        st = torch.load(str(path), map_location="cpu", weights_only=False); return int(st.get("epoch", -1))
    except Exception:
        return -1
STD_PREDS = ["R", "F", "B"]
ARMS = [("R", "base"), ("F", "base"), ("B", "base"), ("AR", "roll"), ("AR", "tf"), ("AR", "roll_x")]
# SCENE_AR_EXTRA="AR2=/path/latest.pt,AR3=..." : kind=ar 인 predictor 를 더 건다 (roll · tf 두 팔). 2026-09-29 context_full_ar_1e_5 용.
AR_EXTRA = [kv.split("=", 1) for kv in os.environ.get("SCENE_AR_EXTRA", "").split(",") if "=" in kv]
for _k, _p in AR_EXTRA:
    CKPT[_k] = Path(_p); ARMS += [(_k, "roll"), (_k, "tf")]
AR_PREDS = ["AR"] + [k for k, _ in AR_EXTRA]
ARM_NAMES = [f"{p}:{a}" for p, a in ARMS]
ENC_FOR = {f"{p}:{a}": ("enc" if p in STD_PREDS or a == "roll_x" else "enc_ar") for p, a in ARMS}


def ssv2_items(limit):
    import decord
    root = Path(os.environ.get("SCENE_SSV2", str(EN.SSV2)))
    out = []
    for c in ("left", "right"):
        for f in sorted((root / c).glob("*.mp4")):
            if len(decord.VideoReader(str(f), num_threads=1)) >= NF: out.append(dict(file=str(f)))
    if limit:
        k = np.linspace(0, len(out) - 1, limit).round().astype(int); out = [out[i] for i in k]
    return out


def items_for(a):
    if a.source == "pan": return ES.pan_items(a.limit or a.n)
    if a.source == "ssv2": return ssv2_items(a.limit)
    return EN.items_for(a.source, a.limit)


def ctx_ar_predict(pred, z, hb):
    """ctx_ar (Ariel 팔 C', 2026-09-29): 문맥 = 창 단위 z (1, 8*S, D) (online encoder, prefix 팔과 같은 입력), 미래 = 블록별 LN(h).
    p_tf = forward_mixed(z, hb[8..14], idx, C=8) → 블록 8..15 예측 (진짜 블록 되먹임); p_roll = rollout_ctx(z, idx_ctx, 8, first=p_tf[:S]) (자기 되먹임)."""
    idx = block_indices(1, 16, S_TOK, hb.device)
    p_tf = pred.forward_mixed(z, hb[:, TC * S_TOK: 15 * S_TOK], idx[:, : 15 * S_TOK], TC)
    p_roll = pred.rollout_ctx(z, idx[:, : TC * S_TOK], 8, first=p_tf[:, :S_TOK])
    for nm, t in (("roll", p_roll), ("tf", p_tf)):
        if not torch.isfinite(t).all(): raise FloatingPointError(f"ctx_ar {nm}: inf/nan")
    return p_roll.float().reshape(8, S_TOK, D), p_tf.float().reshape(8, S_TOK, D)


def ar_predict(pred, hb):
    """hb (1, 16*S, D) LN 블록 토큰 → (p_roll (8,S,D), p_tf (8,S,D))."""
    idx = block_indices(1, 16, S_TOK, hb.device)
    p_roll = pred.rollout(hb[:, : TC * S_TOK], idx[:, : TC * S_TOK], 8)
    p_tf = pred.forward_seq(hb[:, :-S_TOK], idx[:, :-S_TOK])[:, (TC - 1) * S_TOK:]
    for nm, t in (("roll", p_roll), ("tf", p_tf)):
        if not torch.isfinite(t).all(): raise FloatingPointError(f"AR {nm}: inf/nan")
    return p_roll.float().reshape(8, S_TOK, D), p_tf.float().reshape(8, S_TOK, D)


def _worker(rank, world, a, items):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    cfg = arlib.v3_cfg(16, 16); os.environ.pop("ARLIB_PRED_CKPT", None)
    bundle = arlib.build_bundle(cfg, dev, window=NF); ac = arlib.autocast_ctx(cfg)
    bf = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    preds = {"R": bundle.predictor}; KIND = {}
    for k in ("F", "B") + tuple(AR_PREDS):
        preds[k], kind = ES.load_predictor(cfg["model"], CKPT[k], dev, bundle.dtype, NF)
        assert {"F": ("default", "oneshot"), "B": ("prefix",)}.get(k, ("ar", "ctx_ar")).__contains__(kind), (k, kind)
        KIND[k] = kind
    ci = idx_range(0, TC, 1, dev); ti = idx_range(TC, TC, 1, dev)
    ds64 = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))) if a.source == "v3" else None
    names = {"reach": ("arm_c", "arm_ct", "arm_cc", "enc", "enc_ar") + (("src", "disp", "srccv") if a.source != "pan" else ()),
             "perm": ("cos_prev", "cos_last", "argmax_prev", "hid", "enc_prev", "enc_last", "enc_prev_ar", "enc_last_ar")}[a.task]
    O = {nm: np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="r+") for nm in names}
    mine = list(range(rank, len(items), world)); t0 = time.time()
    for n_done, row in enumerate(mine):
        it = items[row]
        if a.task == "reach":
            if a.source == "pan":
                x0 = ES.load_pan(it); src, _ = ES.pan_src(it["v"], it["dir"])
            else:
                x0, gt = EN.load_item(a.source, it, ds64); src, disp, scv = EN.src_maps(a.source, x0, gt)
                O["src"][row] = src; O["disp"][row] = disp; O["srccv"][row] = scv
        else:
            k = it["k"]; m = (k + 1) // 2
            x0 = EP.occlude(ES.load_pan(it), k); sp = EP.src_prev(it["v"], it["dir"], m); s0 = ES.pan_src(it["v"], it["dir"])[0]
            hid = np.zeros((8, S_TOK), bool)
            for i in range(8):
                x15 = (np.arange(S_TOK) % G) * 16 + 8 + it["dir"] * it["v"] * (2 * (i + 1) - 0.5)
                hid[i] = (x15 >= EP.BX0) & (x15 < EP.BX1)
            O["hid"][row] = hid
        x = x0[None].to(dev, bundle.dtype)
        with torch.inference_mode():
            with ac():
                h = F.layer_norm(bundle.target_encoder(x), (D,)).float().reshape(16, S_TOK, D)
                hc = F.layer_norm(bundle.target_encoder(x[:, :, :16]), (D,)).float().reshape(8, S_TOK, D)
                z = bundle.context_encoder(x, masks=[ci])
                p_std = {k: preds[k](z, ci, ti, mask_index=0).float().reshape(8, S_TOK, D) for k in STD_PREDS}
            with bf():
                hb = encode_blocks(bundle.target_encoder, x, 2)                          # (1, 16*S, D) LN
                p_ar = {k: (ctx_ar_predict(preds[k], z, hb) if KIND.get(k) == "ctx_ar" else ar_predict(preds[k], hb)) for k in AR_PREDS}
                p_roll, p_tf = p_ar["AR"]
            hb = hb.float().reshape(16, S_TOK, D); hcL = hc[7]; T_ar = hb[7]
            if a.task == "reach":
                for nm, X, T in (("enc", h[8:], hcL), ("enc_ar", hb[8:], T_ar)):
                    c_, ct_, cc_ = ES.readout(X, T, src); O[nm][row] = np.stack([c_.astype(np.float32), ct_.astype(np.float32), cc_.astype(np.float32)], 0)
                outs = {"R:base": (p_std["R"], hcL), "F:base": (p_std["F"], hcL), "B:base": (p_std["B"], hcL),
                        "AR:roll": (p_roll, T_ar), "AR:tf": (p_tf, T_ar), "AR:roll_x": (p_roll, hcL)}
                for k in AR_PREDS[1:]: outs[f"{k}:roll"], outs[f"{k}:tf"] = (p_ar[k][0], T_ar), (p_ar[k][1], T_ar)
                for ai, nm in enumerate(ARM_NAMES):
                    X, T = outs[nm]; c_, ct_, cc_ = ES.readout(X, T, src)
                    O["arm_c"][row, ai] = c_; O["arm_ct"][row, ai] = ct_; O["arm_cc"][row, ai] = cc_
            else:
                spv = torch.as_tensor(sp, device=dev); s0v = torch.as_tensor(s0, device=dev)
                def cos_at(X, Hs, srcv):
                    cs = F.normalize(X, dim=-1) @ F.normalize(Hs, dim=-1).T
                    return cs, torch.gather(cs, 2, srcv.clamp(min=0)[..., None])[..., 0].masked_fill(srcv < 0, float("nan"))
                hp, hl = hc[TC - 1 - m], hc[TC - 1]; hpa, hla = hb[TC - 1 - m], hb[TC - 1]
                O["enc_prev"][row] = cos_at(h[8:], hp, spv)[1].half().cpu().numpy(); O["enc_last"][row] = cos_at(h[8:], hl, s0v)[1].half().cpu().numpy()
                O["enc_prev_ar"][row] = cos_at(hb[8:], hpa, spv)[1].half().cpu().numpy(); O["enc_last_ar"][row] = cos_at(hb[8:], hla, s0v)[1].half().cpu().numpy()
                outs = {"R:base": (p_std["R"], hp, hl), "F:base": (p_std["F"], hp, hl), "B:base": (p_std["B"], hp, hl),
                        "AR:roll": (p_roll, hpa, hla), "AR:tf": (p_tf, hpa, hla), "AR:roll_x": (p_roll, hp, hl)}
                for k in AR_PREDS[1:]: outs[f"{k}:roll"], outs[f"{k}:tf"] = (p_ar[k][0], hpa, hla), (p_ar[k][1], hpa, hla)
                for ai, nm in enumerate(ARM_NAMES):
                    X, Hp, Hl = outs[nm]; cs, ct = cos_at(X, Hp, spv)
                    O["cos_prev"][row, ai] = ct.half().cpu().numpy(); O["argmax_prev"][row, ai] = cs.argmax(-1).short().cpu().numpy()
                    O["cos_last"][row, ai] = cos_at(X, Hl, s0v)[1].half().cpu().numpy()
        if rank == 0 and n_done % 10 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분 남은 {el/(n_done+1)*(len(mine)-n_done-1)/60:.1f} 분", flush=True)
    for v_ in O.values(): v_.flush()
    print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("reach", "perm"), default="reach"); ap.add_argument("--source", choices=("pan", "ssv2", "ek100", "v3"), default="pan")
    ap.add_argument("--out", required=True); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--n", type=int, default=800); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    assert a.task == "reach" or a.source == "pan", "perm 은 팬만"
    items = items_for(a); n = len(items); A = len(ARMS)
    if a.task == "perm":
        for j, it in enumerate(items): it["k"] = EP.KS[j % len(EP.KS)]
        shapes = dict(cos_prev=((n, A, 8, S_TOK), np.float16), cos_last=((n, A, 8, S_TOK), np.float16), argmax_prev=((n, A, 8, S_TOK), np.int16),
                      hid=((n, 8, S_TOK), np.bool_), enc_prev=((n, 8, S_TOK), np.float16), enc_last=((n, 8, S_TOK), np.float16),
                      enc_prev_ar=((n, 8, S_TOK), np.float16), enc_last_ar=((n, 8, S_TOK), np.float16))
    else:
        shapes = dict(arm_c=((n, A, 8, S_TOK), np.int16), arm_ct=((n, A, 8, S_TOK), np.float16), arm_cc=((n, A, 8, S_TOK), np.float16),
                      enc=((n, 3, 8, S_TOK), np.float32), enc_ar=((n, 3, 8, S_TOK), np.float32))
        if a.source != "pan": shapes.update(src=((n, 8, S_TOK), np.int16), disp=((n, 8, S_TOK), np.float16), srccv=((n, 8, S_TOK), np.int16))
    for nm, (sh, dt) in shapes.items():
        m = np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="w+", dtype=dt, shape=sh)
        if dt != np.bool_: m[:] = (-1 if dt == np.int16 else np.nan)
        del m
    json.dump(dict(n=n, task=a.task, source=a.source, items=items, arms=ARM_NAMES, preds=ARM_NAMES, enc_for=ENC_FOR, ckpt={k: str(v) for k, v in CKPT.items()}, ckpt_epoch={k: ckpt_epoch(v) for k, v in CKPT.items()},
                   band=[EP.BX0, EP.BX1], ks=EP.KS, speeds=ES.SPEEDS, enc_rows=["argmax", "cos_true_src", "cos_copy"]), open(Path(a.out) / "meta.json", "w"))
    print("clips", n, "arms", ARM_NAMES, flush=True); torch.set_num_threads(4)
    mp.spawn(_worker, args=(a.gpus, a, items), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
