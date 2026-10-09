#!/usr/bin/env python3
"""I-scene — 장면 단위 reach 와 인과 개입 (2026-09-25). 물체 하나가 아니라 **토큰 장 전체** 가 얼마나 멀리 운반되나, 그리고 무엇이 그 한계를 정하나.

데이터 (--source pan): Kinetics-400 val 의 정지 프레임 한 장 위에 256² 창을 일정 속도로 가로로 민다 (인공 카메라 팬).
  모든 토큰의 출처가 **정확히** 알려진다: 미래 슬롯 i 의 칸 s 내용은 마지막 문맥 튜블릿 (L=7) 에서 x + dir·v·2(i+1) px 에 있었다.
  속도 v ∈ {2, 4, 8, 12} px/프레임 (= 슬롯당 0.25 / 0.5 / 1 / 1.5 칸). 32 장 = 문맥 16 + 미래 16. 프레임은 K400 을 높이 384 로 키운 뒤 가운데 줄에서 자른다.
판독 (파라미터 없음): 미래 토큰 X_i[s] 의 출처 찾기 ĉ = argmax_r cos(X_i[s], hc_L[r]) (hc = 문맥 16 장만 본 LN target, 인과).
  저장: ĉ (int16) · cos(X_i[s], hc_L[u]) (진짜 출처) · cos(X_i[s], hc_L[s]) (복사) — X = h_{8+i} (encoder 천장) 와 predictor 팔마다.
개입 팔 (전부 frozen, scene_hooks.SceneHook):
  release: base (+attention 기록) · rule_full (검사: base 와 같아야) · rule_prefix · rule_iso (I3 규칙)
           · ropeF{0.5,0.75} (I2: 미래 질의 행만 공간 RoPE 축척; 스모크에서 전 토큰 축척은 가까운 거리까지 무너뜨려 해석 불가)
           · temp{1.5,2} (정답 위치 없이 미래 질의 attention 날카롭히기)
           · bias{1,2,4} (I1: 미래 질의 (i,s) → 진짜 출처 key (L, u) 에 logit +β; 팬은 등속이라 문맥 외삽 = 정답) · bias4_rev (대조: 반대 방향)
           · supp_src4 (필요성: 진짜 출처 key 에 −4) · anticopy{2,4} (정답 없음: 자기 기둥 = 모든 문맥 튜블릿의 같은 칸 key 에 −β)
  Ariel ep43: base (+기록) · rule_full · ropeF0.75 · temp2 · bias{2,4} · supp_src4 · anticopy4       pv1: base (+기록) · temp2 · bias4 · anticopy4
  attention 기록 (base 셋): 미래 질의 (i,s) 의 확률 질량 — 진짜 출처 key · 같은 거리 ring 평균 · 복사 key (마지막 문맥 튜블릿), 층 평균과 세 구간.
  E (encoder 등변성, v=2 clip): 창을 3 칸 (48 px) 민 clip 의 z · h 가 원래의 3 칸 이동판과 같은가 (정렬 cos vs 비정렬 cos).
검사: 각 rank 첫 clip 에서 hook (개입 없음) 출력 vs 원래 출력 상대 L1 (release · Ariel) 을 로그에 남긴다.

  python e_scene_reach.py --gpus 8 --n 800 --out <dir>          (--limit 로 smoke)
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
from e_ssv2_extract import MEAN, STD  # noqa: E402
import scene_hooks as SH  # noqa: E402

S_TOK, D, NF, TC, G, PX = 256, 1280, 32, 8, 16, 16
K400 = Path("/data2/local_datasets/Kinetics-400/videos_val")
SPEEDS = [2, 4, 8, 12]
ZH = 384
Z = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training")
ARIEL = Z / "ariel/block_causal_future_only_1e_5/latest.pt"   # ⚠️ 2026-09-27: 옛 ep_43 삭제 → 같은 run 최종 ep45 (다른 파일). 09-25~26 결과 (_stage/results_vll3) 는 ep43 기준
PV1 = Z / "runs/predictor_v1_postft/latest.pt"
ARMS = [("R", "base", dict(rec=True)), ("R", "rule_full", dict(rule="full")), ("R", "rule_prefix", dict(rule="prefix")), ("R", "rule_iso", dict(rule="isolated")),
        ("R", "ropeF0.5", dict(pos_scale=0.5, pos_rows="future")), ("R", "ropeF0.75", dict(pos_scale=0.75, pos_rows="future")),
        ("R", "temp1.5", dict(temp=1.5)), ("R", "temp2", dict(temp=2.0)),
        ("R", "bias1", dict(bias=1.0)), ("R", "bias2", dict(bias=2.0)), ("R", "bias4", dict(bias=4.0)), ("R", "bias4_rev", dict(bias=4.0, rev=True)),
        ("R", "supp_src4", dict(bias=-4.0)), ("R", "anticopy2", dict(anticopy=2.0)), ("R", "anticopy4", dict(anticopy=4.0)),
        ("R", "qshift", dict(qshift=True)),
        ("A", "base", dict(rec=True)), ("A", "rule_full", dict(rule="full")), ("A", "ropeF0.75", dict(pos_scale=0.75, pos_rows="future")),
        ("A", "temp2", dict(temp=2.0)), ("A", "bias2", dict(bias=2.0)), ("A", "bias4", dict(bias=4.0)), ("A", "supp_src4", dict(bias=-4.0)), ("A", "anticopy4", dict(anticopy=4.0)),
        ("P", "base", dict(rec=True)), ("P", "temp2", dict(temp=2.0)), ("P", "bias4", dict(bias=4.0)), ("P", "anticopy4", dict(anticopy=4.0))]
REC_PREDS = ["R", "A", "P"]
# 추가 predictor (학습 run 평가): SCENE_EXTRA="T0=/path/latest.pt,T1=/path/latest.pt" → 각각 base(+기록) · bias4 · anticopy4 팔.
#   SCENE_ONLY_EXTRA=1 이면 R 의 base 만 남기고 (대조) 나머지 기본 팔은 뺀다.
EXTRA = [kv.split("=", 1) for kv in os.environ.get("SCENE_EXTRA", "").split(",") if "=" in kv]
if EXTRA and os.environ.get("SCENE_ONLY_EXTRA"):
    ARMS = [a for a in ARMS if a[0] in ("R", "A", "P") and a[1] in ("base", "bias4")]
for _k, _p in EXTRA:
    ARMS += [(_k, "base", dict(rec=True)), (_k, "bias4", dict(bias=4.0)), (_k, "anticopy4", dict(anticopy=4.0))]
    REC_PREDS.append(_k)
ARM_NAMES = [f"{p}:{a}" for p, a, _ in ARMS]


def load_predictor(model_cfg, ckpt, device, dtype, window):
    """build_from_config 의 predictor 부분만 (encoder 를 다시 올리지 않는다)."""
    from analysis.intphys2 import model as M
    from analysis import predictors as PV
    pst = M._load_checkpoint(str(ckpt)); pc = model_cfg.get("predictor", {}) or {}
    kind = str(((pst.get("arch") or {}).get("kind")) or "default")
    kk = {k: (pst.get("arch") or {})[k] for k in PV.KIND_ONLY_KEYS if k in (pst.get("arch") or {})}
    pred = M._build_predictor(img_size=int(model_cfg.get("img_size", 256)), patch_size=16, tubelet_size=2, num_frames=window, encoder_embed_dim=D,
                              predictor_embed_dim=int(pc.get("embed_dim", 384)), predictor_depth=int(pc.get("depth", 12)),
                              predictor_num_heads=int(pc.get("num_heads", 12)), num_mask_tokens=int(pc.get("num_mask_tokens", 10)),
                              use_rope=True, uniform_power=False, kind=kind, kind_kwargs=kk)
    psd = M._clean_backbone_key(pst["predictor"]); missing = sorted(set(pred.state_dict()) - set(psd))
    assert not missing, missing[:5]
    M._load_state_dict(pred, pst["predictor"], tag=f"predictor<-{Path(ckpt).name}")
    return pred.to(device=device, dtype=dtype).eval(), kind


def pan_items(n):
    lst = os.environ.get("SCENE_PAN_LIST")                # 고정 clip 목록 (다른 노드에서 같은 선택을 재현; exp_results/scene/pan_items_1600.json)
    if lst:
        items = json.load(open(lst)); items = items[:n] if n < len(items) else items
        return [dict(it) for it in items]
    import decord
    fs = sorted(K400.glob("*.mp4"))
    out, i = [], 0
    step = max(1, len(fs) // (n * 3))
    for f in fs[::step]:
        if len(out) >= n: break
        try:
            vr = decord.VideoReader(str(f), num_threads=1); hh, ww = vr[0].shape[:2]
        except Exception:
            continue
        W = int(round(ww * ZH / hh))
        v = SPEEDS[len(out) % len(SPEEDS)]; dr = 1 if (len(out) // len(SPEEDS)) % 2 == 0 else -1
        if W < 256 + 31 * max(SPEEDS) + 8: continue
        out.append(dict(file=str(f), v=v, dir=dr, W=W, fidx=int(len(vr) * 0.4)))
    return out


def load_pan(it, shift_px=0):
    import decord
    vr = decord.VideoReader(it["file"], num_threads=1)
    fr = torch.from_numpy(vr[it["fidx"]].asnumpy()).permute(2, 0, 1).float()[None] / 255.0
    fr = F.interpolate(fr, size=(ZH, it["W"]), mode="bilinear", align_corners=False, antialias=False)[0]
    y0 = (ZH - 256) // 2; W, v, dr = it["W"], it["v"], it["dir"]
    c = (W - 256) // 2
    xs = []
    for f in range(NF):
        o = int(round(c + dr * v * (f - 15.5))) + shift_px
        o = max(0, min(W - 256, o))
        xs.append(fr[:, y0:y0 + 256, o:o + 256])
    x = torch.stack(xs, 1)                                                     # (3, 32, 256, 256)
    return (x - MEAN) / STD


def pan_src(v, dr, rev=False):
    """(8, 256) 출처 flat 인덱스 (−1 = 창 밖) 와 연속 변위 (칸)."""
    src = np.full((8, S_TOK), -1, np.int64); disp = np.zeros((8, S_TOK), np.float32)
    for i in range(8):
        sh = dr * v * 2 * (i + 1) * (-1 if rev else 1)
        for y in range(G):
            for x in range(G):
                xp = PX * x + PX / 2 + sh
                if 0 <= xp < 256: src[i, y * G + x] = y * G + int(xp // PX)
                disp[i, y * G + x] = abs(v * 2 * (i + 1)) / PX
    return src, disp


def readout(X, hcL, src):
    """X (8,256,D), hcL (256,D) → ĉ (8,256), cos 진짜 출처, cos 복사."""
    Xn, Hn = F.normalize(X.float(), dim=-1), F.normalize(hcL.float(), dim=-1)
    cs = Xn @ Hn.T                                                             # (8, 256, 256)
    am = cs.argmax(-1)
    sv = torch.as_tensor(src, device=X.device)
    ct = torch.gather(cs, 2, sv.clamp(min=0)[..., None])[..., 0].masked_fill(sv < 0, float("nan"))
    cc = cs.diagonal(dim1=1, dim2=2)
    return am.short().cpu().numpy(), ct.half().cpu().numpy(), cc.half().cpu().numpy()


def bias_matrix(src, beta, dev):
    N = 16 * S_TOK; B = torch.zeros(N, N, device=dev)
    for i in range(8):
        for s in range(S_TOK):
            if src[i, s] >= 0: B[(8 + i) * S_TOK + s, 7 * S_TOK + src[i, s]] = beta
    return B


def anticopy_matrix(beta, dev):
    """정답 없음: 미래 질의 (i, s) → 모든 문맥 튜블릿의 같은 칸 s key (자기 기둥) 에 logit −β."""
    N = 16 * S_TOK; B = torch.zeros(N, N, device=dev); s = torch.arange(S_TOK, device=dev)
    for i in range(8):
        for t in range(TC):
            B[(8 + i) * S_TOK + s, t * S_TOK + s] = -beta
    return B


def qshift_matrix(src, dev):
    """(8, 256) 출처 → (N, 2): 미래 질의 (i, s) 의 RoPE 위치를 출처 칸 u 로 옮기는 오프셋 (u − s, 칸). 문맥 행과 출처 없는 질의는 0."""
    N = 16 * S_TOK; Q = torch.zeros(N, 2, device=dev); s = np.arange(S_TOK)
    for i in range(8):
        u = src[i]; ok = u >= 0
        dh = np.where(ok, u // G - s // G, 0); dw = np.where(ok, u % G - s % G, 0)
        Q[(8 + i) * S_TOK: (9 + i) * S_TOK, 0] = torch.as_tensor(dh, device=dev, dtype=torch.float32)
        Q[(8 + i) * S_TOK: (9 + i) * S_TOK, 1] = torch.as_tensor(dw, device=dev, dtype=torch.float32)
    return Q


def rec_spec(src, dev):
    """미래 질의 2048 행 (정렬 순서 2048..4095) 의 attention 기록 명세: 출처 key · 같은 Chebyshev 거리 ring (출처 제외) · 복사 key (마지막 문맥 튜블릿 안)."""
    s_idx = np.tile(np.arange(S_TOK), 8); sv = src.reshape(-1)
    sx, sy = s_idx % G, s_idx // G; kx, ky = np.arange(S_TOK) % G, np.arange(S_TOK) // G
    dk = np.maximum(np.abs(kx[None] - sx[:, None]), np.abs(ky[None] - sy[:, None]))       # (2048, 256)
    ds = np.where(sv >= 0, np.maximum(np.abs(sv % G - sx), np.abs(sv // G - sy)), -1)
    ring = (dk == ds[:, None]) & (np.arange(S_TOK)[None] != sv[:, None]) & (sv[:, None] >= 0)
    t = lambda a_, dt=torch.long: torch.as_tensor(a_, device=dev, dtype=dt)
    return dict(q_rows=t(np.arange(8 * S_TOK, 16 * S_TOK)), key_src=t(sv), ring=t(ring, torch.bool), key_copy=t(s_idx), l_off=7 * S_TOK)


def _worker(rank, world, a, items):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    cfg = arlib.v3_cfg(16, 16); os.environ.pop("ARLIB_PRED_CKPT", None)
    bundle = arlib.build_bundle(cfg, dev, window=NF); ac = arlib.autocast_ctx(cfg)
    preds = {"R": bundle.predictor}
    preds["A"], ka = load_predictor(cfg["model"], ARIEL, dev, bundle.dtype, NF)
    preds["P"], kp = load_predictor(cfg["model"], PV1, dev, bundle.dtype, NF)
    assert ka == "prefix" and kp in ("default", "oneshot"), (ka, kp)
    for _k, _p in EXTRA:
        preds[_k], _ = load_predictor(cfg["model"], _p, dev, bundle.dtype, NF)
    ci = idx_range(0, TC, 1, dev); ti = idx_range(TC, TC, 1, dev)
    O = {nm: np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="r+") for nm in ("arm_c", "arm_ct", "arm_cc", "enc", "E", "att") + (("cosmap",) if a.cosmaps else ())}
    hooks = None; mine = list(range(rank, len(items), world)); t0 = time.time()
    for n_done, row in enumerate(mine):
        it = items[row]; x = load_pan(it)[None].to(dev, bundle.dtype)
        src, _ = pan_src(it["v"], it["dir"]); src_rev, _ = pan_src(it["v"], it["dir"], rev=True)
        with torch.inference_mode(), ac():
            h = F.layer_norm(bundle.target_encoder(x), (D,)).float().reshape(16, S_TOK, D)
            hc = F.layer_norm(bundle.target_encoder(x[:, :, :16]), (D,)).float().reshape(8, S_TOK, D)
            z = bundle.context_encoder(x, masks=[ci])
            if hooks is None:                                                  # 첫 clip: hook 재현 검사
                p0 = {k: preds[k](z, ci, ti, mask_index=0).float() for k in "RA"}
                hooks = {k: SH.patch(preds[k], SH.SceneHook()) for k in preds}
                for k in hooks.values(): k.n_ctx_blocks = TC
                rl = {k: float((preds[k](z, ci, ti, mask_index=0).float() - p0[k]).abs().mean() / p0[k].abs().mean()) for k in "RA"}
                print(f"[rank{rank}] hook 재현 상대 L1 {rl}", flush=True)
            hcL = hc[7]
            ec, ect, ecc = readout(h[8:], hcL, src)
            O["enc"][row] = np.stack([ec.astype(np.float32), ect.astype(np.float32), ecc.astype(np.float32)], 0)
            rec = rec_spec(src, dev)
            for ai, (pk, name, kw) in enumerate(ARMS):
                hk = hooks[pk]; hk.reset(); hk.n_ctx_blocks = TC
                hk.rule = kw.get("rule"); hk.pos_scale = kw.get("pos_scale"); hk.pos_rows = kw.get("pos_rows", "all"); hk.temp = kw.get("temp")
                if "bias" in kw: hk.bias = bias_matrix(src_rev if kw.get("rev") else src, kw["bias"], dev)
                if "anticopy" in kw: hk.bias = anticopy_matrix(kw["anticopy"], dev)
                if kw.get("qshift"): hk.qshift = qshift_matrix(src, dev)
                if kw.get("rec"): hk.rec = rec
                p = preds[pk](z, ci, ti, mask_index=0).float().reshape(8, S_TOK, D)
                c_, ct_, cc_ = readout(p, hcL, src)
                if a.cosmaps and kw.get("rec"):
                    O["cosmap"][row, REC_PREDS.index(pk)] = (F.normalize(p, dim=-1) @ F.normalize(hcL, dim=-1).T).half().cpu().numpy()
                O["arm_c"][row, ai] = c_; O["arm_ct"][row, ai] = ct_; O["arm_cc"][row, ai] = cc_
                if kw.get("rec"):
                    R_ = torch.stack(hk.rec_out).float()                                # (L, 3, 2048)
                    Lh = R_.shape[0]; g = [R_.mean(0), R_[:Lh // 3].mean(0), R_[Lh // 3:2 * Lh // 3].mean(0), R_[2 * Lh // 3:].mean(0)]
                    O["att"][row, REC_PREDS.index(pk)] = torch.stack(g).reshape(4, 3, 8, S_TOK).half().cpu().numpy()
                hk.reset()
            if it["v"] == SPEEDS[0]:                                          # E: encoder 등변성 (창 3 칸 이동)
                xs = load_pan(it, shift_px=48)[None].to(dev, bundle.dtype)
                zs = bundle.context_encoder(xs, masks=[ci]).float().reshape(8, G, G, -1); z8 = z.float().reshape(8, G, G, -1)
                hs = F.layer_norm(bundle.target_encoder(xs), (D,)).float().reshape(16, G, G, D); h8 = h.reshape(16, G, G, D)
                cosf = lambda A, B: F.cosine_similarity(A, B, dim=-1).mean().item()
                # 창을 +48 px 밀면 내용이 3 칸 왼쪽으로: zs[..., x] ≈ z[..., x+3]  (가장자리 3 칸 제외)
                O["E"][row] = [cosf(zs[:, 2:-2, 2:-5], z8[:, 2:-2, 5:-2]), cosf(zs[:, 2:-2, 2:-5], z8[:, 2:-2, 2:-5]),
                               cosf(hs[:, 2:-2, 2:-5], h8[:, 2:-2, 5:-2]), cosf(hs[:, 2:-2, 2:-5], h8[:, 2:-2, 2:-5])]
        if rank == 0 and n_done % 10 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분 남은 {el/(n_done+1)*(len(mine)-n_done-1)/60:.1f} 분", flush=True)
    for v_ in O.values(): v_.flush()
    print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--n", type=int, default=800); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cosmaps", action="store_true", help="base 팔의 cos 지도 전체 (n, |rec_preds|, 8, 256, 256) fp16 저장")
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    items = pan_items(a.limit or a.n); n = len(items)
    shapes = dict(arm_c=((n, len(ARMS), 8, S_TOK), np.int16), arm_ct=((n, len(ARMS), 8, S_TOK), np.float16), arm_cc=((n, len(ARMS), 8, S_TOK), np.float16),
                  enc=((n, 3, 8, S_TOK), np.float32), E=((n, 4), np.float32), att=((n, len(REC_PREDS), 4, 3, 8, S_TOK), np.float16))
    if a.cosmaps: shapes["cosmap"] = ((n, len(REC_PREDS), 8, S_TOK, S_TOK), np.float16)
    for nm, (sh, dt) in shapes.items():
        m = np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="w+", dtype=dt, shape=sh); m[:] = (-1 if dt == np.int16 else np.nan); del m
    json.dump(dict(n=n, items=items, arms=ARM_NAMES, rec_preds=REC_PREDS, speeds=SPEEDS, enc_rows=["argmax", "cos_true_src", "cos_copy"], E_cols=["z_aligned", "z_null", "h_aligned", "h_null"],
                   att_axes=["pred R/A/P", "layers mean/first third/mid/last", "p_src/p_ring/p_copy", "slot", "token"]),
              open(Path(a.out) / "meta.json", "w"))
    print("clips", n, "arms", len(ARMS), flush=True)
    torch.set_num_threads(4)
    mp.spawn(_worker, args=(a.gpus, a, items), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
