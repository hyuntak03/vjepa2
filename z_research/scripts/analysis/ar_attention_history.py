#!/usr/bin/env python3
"""AR predictor 는 과거 문맥을 보나 — rollout 중 attention 을 직접 잰다 (2026-10-13, 사용자 요청).

물음 (사용자 10-13): "vast ar_state 70 epoch 로 attention map 을 그려서 실제로 과거 context 를 잘 attend 안 하는지 보자.
attention 크기의 변화를 보고 싶고, 물체 위치를 가정해서도."
anchor 이식 (predictor_anchor_test) 은 출력 쪽 증거였다 — 블록 6 을 바꿔도 AR 의 미래가 안 바뀐다. 여기서는 내부를 본다.

판 AR (--ckpt 있음):
  문맥 = 블록별 LN(target_encoder) 8 블록 (프레임 0–15), 미래 8 걸음 rollout (rollout_kv, 값은 rollout 과 같음 — 스크립트가 검사).
  걸음 r 에서 블록 C+r 을 예측하는 query = 위치 C+r−1 의 토큰 256 개 (r=0 은 마지막 문맥 블록, r≥1 은 앞 걸음의 상태/출력 토큰).
  key = 문맥 블록 0..C−1 + 앞 걸음 블록 C..C+r−1.  12 층 × 12 head 의 softmax(qkᵀ/√d) 를 그대로 계산한다.
판 릴리즈 (--ckpt 없음, name = vjepa2, 2026-10-14 추가 · one-shot 체크포인트 (릴리즈 post-FT) 도 같은 분기):
  문맥 = 문맥 encoder (online) 의 프레임 0–15 토큰, 미래 8 블록 = mask token 으로 **한 번에** 예측 (표준 surprise 와 같은 호출).
  토큰은 위치순 (블록 0..15) 으로 정렬돼 full attention — 층마다 sdpa 호출 하나 (q = k = 16 블록 × 256).
  '걸음 r' = query 블록 C+r (미래 mask 토큰). key = 16 블록 전부 (자기 블록 · 뒤 미래 블록 포함). 배열 끝 축 = 16 (AR 은 15).
  ⚠️ 입력 공간이 다르다 — AR 은 블록별 LN(target_encoder), 릴리즈는 문맥 encoder. 질량 분포만 비교한다.
재는 것:
  bm_all[r, l, h, b]  : 모든 query 평균, key 블록 b 에 간 질량 (균등 = 1/(C+r))
  bm_obj[r, l, h, b]  : query = 물체가 **예측 대상 블록 C+r 에서 있을 자리** 3×3 (물체 위치 '가정' — 라벨의 참 자리)
  om_obj[r, l, h, b]  : 그 query 가 key 블록 b 의 **물체 자리 3×3** 에 준 질량 (블록 안 균등 = 9/256)
  om_ql[r, l, h, b]   : 대조 2 (10-14) — 같은 query 가 key 블록 b 에서 **query 와 같은 좌표** 3×3 에 준 질량 (공간 국소성 기준선;
                        먼 걸음은 거울 자리가 화면 밖이라 om_mir 가 비어서 추가. 물체 이동 ≥ 3 토큰인 clip 에서만 om_obj 와 비교)
  om_mir[r, l, h, b]  : 대조 — 같은 query 가 key 블록 b 에서 **물체 자리를 query 자리 기준으로 뒤집은 자리** 3×3 에 준 질량
                        (query 와의 거리가 같다 → RoPE 공간 국소성만으로는 om_obj = om_mir. om_obj > om_mir 여야 '물체를 본다')
  hid[b]              : key 블록 b 에서 물체가 가려졌나 (late 판에서 마지막으로 보인 블록을 가르려고)
  objmap              : 물체 query 의 head 평균 attention 을 key 블록마다 물체 중심 7×7 로 잘라 합한 것 (그림용)
  ex_*                : 예시 clip 몇 개의 원 attention (head 평균, 그림용)
세트: vob_visible / vob_late (VOBench_v1 이동 · 가능 변이 · 물체 있음), ssv2 (val, 물체 라벨 없음 → bm_all 만).
⚠️ attention 질량 ≠ 정보 사용 (value 크기 · residual 경로). 출력 쪽 인과 증거는 anchor 이식 · 역사 가치 실험과 같이 읽는다.

  python ar_attention_history.py --name vast_ar_state_ep70 --ckpt z_training/vast/ar_state/latest.pt --set vob_late --gpus 4
  python ar_attention_history.py --name vast_ar_state_ep70 --set vob_late --summarize
  python ar_attention_history.py --name vjepa2 --set vob_late --gpus 4                    # 릴리즈 (--ckpt 없음)
  python ar_attention_history.py --name vobench_release_postft_ep30 --ckpt z_training/runs/vobench_release_postft/latest.pt --set vob_late --gpus 4
      # 릴리즈 post-FT (2026-10-14): kind 는 체크포인트가 정한다 (default → 릴리즈 분기). VOBench 판은 split_v2 train / test 요약도 같이 낸다
산출: z_research/VOBench_v1/analysis/attention_map/<name>/<set>_r*.npz, <set>.json  (name = 모델 폴더: vjepa2 · vast_ar_state_ep70 · ariel_ar_ep40 · ariel_ar_state_ep40)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))

S, NB, C, K, G, H_, L_ = 256, 16, 8, 8, 16, 12, 12
NBMAX = C + K - 1
OUT = REPO / "z_research/VOBench_v1/analysis/attention_map"                          # 모델별 폴더 OUT/<name>/
VAL = REPO / "data_csv/ssv2/val_min48.csv"
W = 3                                  # 물체 중심 7×7 (반폭 3)
N_EX = 3                               # rank 마다 저장할 예시 clip


def obj_tokens(P, hid):
    """VOBench 라벨 (512 px, 샘플 32) → 블록 16 개의 물체 토큰 (y, x) 과 가림 여부."""
    import design_headroom as Hm
    Pb = P.reshape(NB, 2, 2).mean(1); hb = hid.reshape(NB, 2).max(1) > 0.5
    return np.array([Hm.tok(p) for p in Pb]), hb


def nbhd(yx, r=1):
    m = np.zeros((G, G), bool); y, x = yx
    m[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1] = True
    return m.reshape(-1)


def _worker(rank, world, name, ckpt, setname, n, bs):
    import torch
    import torch.nn.functional as F
    from importlib.util import module_from_spec, spec_from_file_location
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import block_indices, encode_blocks
    from evals.world_model_analysis.data import WMADataset
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir); rp = ir.rp
    model = dict(rp.MODEL) if not ckpt else dict(rp.MODEL, predictor_checkpoint=str(REPO / ckpt))
    bundle = build_from_config(model, dev); pred = bundle.predictor
    release = getattr(pred, "mask_mode", None) not in ("ar", "ar_state")                  # 릴리즈 · 릴리즈 post-FT (one-shot, full attention)
    if release: assert not hasattr(pred, "mask_mode"), pred.mask_mode                     # prefix · causal 은 이 분기로 오면 안 된다
    NBK = NB if release else NBMAX                                                       # key 블록 축 길이

    if setname == "ssv2":
        import decord
        from analysis.intphys2.dataset import build_video_transform
        rows = [r for r in csv.reader(open(VAL), delimiter=" ")]
        rng = np.random.default_rng(0); rng.shuffle(rows); items = [r[0] for r in rows[:n]]
        tf_ = build_video_transform(256, 1.0)

        def load(f):
            try:
                vr = decord.VideoReader(f, num_threads=1)
                return (tf_(vr.get_batch(list(range(32))).asnumpy()), None, None, f) if len(vr) >= 32 else None
            except Exception:
                return None
    else:
        import yaml
        import design_headroom as Hm
        cfg = yaml.safe_load(open(Hm.SETS["vobench"]["cfg"]))
        ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=model, features={"cache_dir": "/tmp"}, surprise={}))
        want = setname.split("_")[1]
        ix = {r["video_id"]: r for r in csv.DictReader(open(Hm.SETS["vobench"]["index"]))}
        items = [i for i, rec in enumerate(ds.records) if (r := ix.get(rec.video_id)) and r["occlusion"] == want
                 and r["motion"] != "flat_static" and r["variant"] in ("pos_a", "pos_b") and r["obj_a" if r["variant"] == "pos_a" else "obj_b"] != "empty"][:n]

        def load(i):
            r = ix[ds.records[i].video_id]; tr = Hm.track_vobench(r)
            if tr is None: return None
            yx, hb = obj_tokens(*tr)
            if not (0 <= tr[0][:, 0].min() and tr[0][:, 0].max() < 512 and 0 <= tr[0][:, 1].min() and tr[0][:, 1].max() < 512): return None
            return ds.clip(i), yx, hb, ds.records[i].video_id
    mine = items[rank * len(items) // world:(rank + 1) * len(items) // world]

    # ── attention 가로채기
    #    AR: rollout_kv 에서 query = 블록 하나 (S 토큰), key ≥ C 블록인 호출이 층마다 걸음마다 정확히 하나
    #    릴리즈: predictor 의 full attention (q = k = NB 블록) 이 층마다 하나
    _sdpa = F.scaled_dot_product_attention
    REC = {"on": False, "i": 0}

    def sdpa(q, k, v, *a, **kw):
        out = _sdpa(q, k, v, *a, **kw)
        hit = (q.shape[2] == NB * S and k.shape[2] == NB * S) if release else (q.shape[2] == S and k.shape[2] >= C * S)
        if REC["on"] and hit:
            REC["fn"](q, k, REC["i"]); REC["i"] += 1
        return out
    F.scaled_dot_product_attention = sdpa

    acc = {}

    def put(k_, v_): acc.setdefault(k_, []).append(v_)
    objmap = np.zeros((K, L_, NBK, 2 * W + 1, 2 * W + 1), np.float64); n_map = 0
    ex = []
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b0 in range(0, len(mine), bs):
            got = [(f,) + g for f in mine[b0:b0 + bs] if (g := load(f)) is not None]
            if not got: continue
            x = torch.stack([g[1] for g in got]).to(dev, dtype=bundle.dtype); B = x.size(0)
            has_obj = got[0][2] is not None
            if not release:
                hb = encode_blocks(bundle.target_encoder, x, 2)
                idx = block_indices(B, C, S, dev)
            bm_all = np.full((B, K, L_, H_, NBK), np.nan, np.float32)
            bm_obj = np.full_like(bm_all, np.nan); om_obj = np.full_like(bm_all, np.nan); om_mir = np.full_like(bm_all, np.nan)
            om_ql = np.full_like(bm_all, np.nan)                                                 # 대조 2 (10-14): key 블록 b 에서 query 와 같은 좌표 3×3
            exA = np.zeros((B, K, L_, 2, NBK * S), np.float16) if (len(ex) < N_EX and has_obj) else None
            if has_obj:
                YX = np.stack([g[2] for g in got]); HID = np.stack([g[3] for g in got])          # (B, 16, 2), (B, 16)
                KM = torch.tensor(np.stack([[nbhd(YX[j, b]) for b in range(NB)] for j in range(B)]), device=dev)   # (B, 16, S) key 물체 자리
                QM = KM[:, C:]                                                                  # 걸음 r 의 query = 블록 C+r 물체 자리
                MIR = np.zeros((B, K, NB, S), bool); MOK = np.zeros((B, K, NB), bool)            # 거울 대조 key
                for j in range(B):
                    for r_ in range(K):
                        for b in range(NB):
                            m_ = 2 * YX[j, C + r_] - YX[j, b]
                            if (m_ >= 1).all() and (m_ <= G - 2).all() and np.abs(m_ - YX[j, b]).max() >= 3:
                                MIR[j, r_, b] = nbhd(m_); MOK[j, r_, b] = True
                MIR = torch.tensor(MIR, device=dev)

            def fn(q, k, i):
                if release:                                                                     # 층 i, 미래 query 블록마다
                    for r in range(K):
                        qs = slice((C + r) * S, (C + r + 1) * S)
                        core(r, i, torch.softmax(q[:, :, qs].float() @ k.float().transpose(-1, -2) * q.shape[-1] ** -0.5, -1))
                else:
                    r, l = divmod(i, L_)
                    core(r, l, torch.softmax(q.float() @ k.float().transpose(-1, -2) * q.shape[-1] ** -0.5, -1))   # (B, H, S, nb*S)

            def core(r, l, A):
                nb = A.shape[-1] // S
                Ab = A.view(B, H_, S, nb, S)
                bm_all[:, r, l, :, :nb] = Ab.sum(-1).mean(2).cpu().numpy()
                if not has_obj: return
                qm = QM[:, r].float()                                                           # (B, S)
                Aq = (Ab * qm[:, None, :, None, None]).sum(2) / qm.sum(1)[:, None, None, None]  # (B, H, nb, S)
                bm_obj[:, r, l, :, :nb] = Aq.sum(-1).cpu().numpy()
                om_obj[:, r, l, :, :nb] = (Aq * KM[:, None, :nb].float()).sum(-1).cpu().numpy()
                mm = (Aq * MIR[:, r, None, :nb].float()).sum(-1).cpu().numpy()
                om_mir[:, r, l, :, :nb] = np.where(MOK[:, None, r, :nb], mm, np.nan)
                om_ql[:, r, l, :, :nb] = (Aq * qm[:, None, None, :]).sum(-1).cpu().numpy()     # 물체가 안 움직였으면 om_obj 와 같다 → 이동 거리로 거른다
                am = Aq.mean(1).view(B, nb, G, G).cpu().numpy()                                 # head 평균, 물체 중심 crop
                for j in range(B):
                    for b in range(nb):
                        y, x_ = YX[j, b]; pad = np.pad(am[j, b], W)
                        objmap[r, l, b] += pad[y:y + 2 * W + 1, x_:x_ + 2 * W + 1]
                if exA is not None:
                    exA[:, r, l, 0, :nb * S] = A.mean((1, 2)).cpu().numpy().astype(np.float16)
                    exA[:, r, l, 1, :nb * S] = Aq.mean(1).reshape(B, -1).cpu().numpy().astype(np.float16)
            if release:
                from analysis.intphys2.surprise import _context_target_indices
                ci, ti = _context_target_indices(ctx_frames=2 * C, tgt_frames=2 * K, tubelet_size=2, spatial_tokens=S, batch_size=B, device=dev)
                z = bundle.context_encoder(x, masks=[ci]); z = z[-1] if isinstance(z, list) else z
                REC.update(on=True, i=0, fn=fn); p = pred(z, ci, ti, mask_index=0); REC["on"] = False
                assert REC["i"] == L_, REC["i"]
            else:
                REC.update(on=True, i=0, fn=fn)
                p = pred.rollout_kv(hb[:, :C * S], idx, K)
                REC["on"] = False
                assert REC["i"] == K * L_, REC["i"]
            if b0 == 0 and not release:                                                          # rollout_kv == rollout 검사
                p_ref = pred.rollout(hb[:, :C * S], idx, K)
                rel = ((p.float() - p_ref.float()).abs().mean() / p_ref.float().abs().mean()).item()
                print(f"[r{rank}] rollout_kv vs rollout rel {rel:.2e}", flush=True); put("kv_rel", np.array([rel]))
            put("bm_all", bm_all.astype(np.float16)); put("vid", np.array([g[4] for g in got]))   # clip id (post-FT train/test 나누기)
            if has_obj:
                put("bm_obj", bm_obj.astype(np.float16)); put("om_obj", om_obj.astype(np.float16)); put("om_mir", om_mir.astype(np.float16)); put("om_ql", om_ql.astype(np.float16)); put("hid", HID); put("yx", YX)
                n_map += B
                if exA is not None:
                    for j in range(B):
                        if len(ex) < N_EX: ex.append((got[j][0], exA[j], YX[j], HID[j]))
            if rank == 0 and (b0 // bs) % 5 == 0: print(f"[r0] {name} {setname} {b0 + B}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    (OUT / name).mkdir(parents=True, exist_ok=True)
    extra = {}
    if ex:
        extra = dict(ex_item=np.array([e[0] for e in ex]), ex_A=np.stack([e[1] for e in ex]), ex_yx=np.stack([e[2] for e in ex]), ex_hid=np.stack([e[3] for e in ex]))
    np.savez(OUT / name / f"{setname}_r{rank}.npz", objmap=objmap, n_map=n_map, kind=np.array(["release" if release else "ar"]), **{k_: np.concatenate(v_) for k_, v_ in acc.items()}, **extra)


def load_all(setname, name):
    Z = {}
    for f in sorted((OUT / name).glob(f"{setname}_r*.npz")):
        z = np.load(f)
        for k in z.files: Z.setdefault(k, []).append(z[k])
    out = {}
    for k, v in Z.items():
        out[k] = sum(v) if k in ("objmap", "n_map") else v[0] if k == "kind" else np.concatenate([np.atleast_1d(a) for a in v])
    return out


SPLIT = REPO / "data_csv/vobench_v1_split_v2"                                        # vobench_release_postft 의 학습 split (block 단위, train 720 / test 1440)
PER_CLIP = ("bm_all", "bm_obj", "om_obj", "om_mir", "om_ql", "hid", "yx", "vid")


def summarize(setname, name, subset=None):
    Z = load_all(setname, name)
    if subset:                                                                                # clip 을 split_v2 의 train / test 로 거른다 (objmap 은 합이라 못 거름)
        keep_ids = {r["video_id"] for r in csv.DictReader(open(SPLIT / f"index_{subset}.csv"))}
        keep = np.array([v in keep_ids for v in Z["vid"]])
        Z = {k: (v[keep] if k in PER_CLIP else v) for k, v in Z.items()}
    bm = Z["bm_all"].astype(np.float32)                                                       # (N, K, L, H, NBMAX 또는 릴리즈 NB)
    release = "kind" in Z and str(Z["kind"][0]) == "release"
    res = {"n": int(bm.shape[0]), "split": subset, "kind": "release" if release else "ar", "kv_rel_max": float(Z["kv_rel"].max()) if "kv_rel" in Z else None}
    hm = np.nanmean(bm, axis=3)                                                               # head 평균 (N, K, L, NBMAX)
    prof = np.nanmean(hm, axis=(0, 2))                                                        # 층 평균 (K, NBMAX)
    res["block_mass_all"] = prof.tolist()
    # 요약: 걸음 r 에서 (문맥 마지막 블록 · 문맥 앞 7 블록 · 앞 걸음들) 질량, 균등 대비
    def split(P):
        rows = []
        for r in range(K):
            nb = C + r; p = P[r, :nb]
            d = dict(last_ctx=float(p[C - 1]), early_ctx=float(p[:C - 1].sum()), own_steps=float(p[C:nb].sum()) if r else 0.0,
                     newest=float(p[nb - 1]), uniform_block=1.0 / nb)
            if release:                                                                       # 릴리즈: 앞 미래 블록 (mask) · 자기 블록 · 뒤 미래 블록
                d.update(self_block=float(P[r, nb]), later_future=float(P[r, nb + 1:].sum()), uniform_block=1.0 / NB)
            rows.append(d)
        return rows
    res["split_all"] = split(prof)
    res["split_all_by_layer"] = [split(np.nanmean(hm[:, :, l], 0)) for l in range(L_)]
    print(f"\n== {name} · {setname}{' · ' + subset if subset else ''} (clip {res['n']}, {res['kind']}" + (f", rollout_kv vs rollout 상대차 최대 {res['kv_rel_max']:.1e})" if res["kv_rel_max"] is not None else ")"))
    print("  모든 query, 층·head 평균 — 걸음 r: [문맥 앞 7 블록 | 문맥 마지막 블록 | 자기 앞 걸음들 | 가장 새 블록]  (블록당 균등)")
    for r, d in enumerate(res["split_all"]):
        print(f"   r{r}: early {d['early_ctx']:.3f} | last {d['last_ctx']:.3f} | own {d['own_steps']:.3f} | newest {d['newest']:.3f}"
              + (f" | self {d['self_block']:.3f} | later {d['later_future']:.3f}" if release else "") + f"   (균등 {d['uniform_block']:.3f})")
    if "bm_obj" in Z:
        bo = np.nanmean(Z["bm_obj"].astype(np.float32), 3); oo = np.nanmean(Z["om_obj"].astype(np.float32), 3)   # (N, K, L, NBMAX)
        hid = Z["hid"]                                                                        # (N, 16)
        # 물체 query 가 key 블록 b 의 물체 자리에 준 질량 / 그 블록 질량 → 블록 안 물체 몫 (균등 9/256 = 0.035)
        share = oo / np.maximum(bo, 1e-9)
        res["obj_query_block_mass"] = np.nanmean(bo, (0, 2)).tolist()
        res["obj_query_obj_mass"] = np.nanmean(oo, (0, 2)).tolist()
        res["obj_share_in_block"] = np.nanmean(share, (0, 2)).tolist()
        # 마지막으로 보인 문맥 블록 (late 판의 핵심)
        lv = np.array([max([b for b in range(C) if not h[b]], default=-1) for h in hid])
        res["last_visible_ctx_block_hist"] = np.bincount(lv + 1, minlength=C + 1).tolist()
        ok = lv >= 0
        def at(Aarr, sel):                                                                    # clip 마다 블록 lv 의 값 (N, K, L)
            return np.stack([Aarr[i, :, :, lv[i]] for i in np.nonzero(sel)[0]])
        if ok.any():
            res["obj_mass_at_last_visible"] = np.nanmean(at(oo, ok), (0, 2)).tolist()
            res["obj_share_at_last_visible"] = np.nanmean(at(share, ok), (0, 2)).tolist()
            res["block_mass_at_last_visible"] = np.nanmean(at(bo, ok), (0, 2)).tolist()
        om = np.nanmean(Z["om_mir"].astype(np.float32), 3)
        both = ~np.isnan(om)
        res["obj_vs_mirror"] = [[[float(np.nanmean(np.where(both[:, r, :, b], oo[:, r, :, b], np.nan))), float(np.nanmean(om[:, r, :, b]))] for b in range(C + r)] for r in range(K)]
        if ok.any():
            sel = np.nonzero(ok)[0]
            o_lv = np.stack([oo[i, :, :, lv[i]] for i in sel]); m_lv = np.stack([om[i, :, :, lv[i]] for i in sel])
            bb = ~np.isnan(m_lv)
            res["last_visible_obj_vs_mirror"] = [[float(np.nanmean(np.where(bb[:, r], o_lv[:, r], np.nan))), float(np.nanmean(m_lv[:, r]))] for r in range(K)]
        res["obj_mass_at_last_ctx"] = np.nanmean(oo[:, :, :, C - 1], (0, 2)).tolist()
        res["obj_share_at_last_ctx"] = np.nanmean(share[:, :, :, C - 1], (0, 2)).tolist()
        res["frac_hidden_last_ctx"] = float(hid[:, C - 1].mean())
        print("  물체 query (블록 C+r 의 참 물체 자리 3×3) — 블록별 [블록 질량 / 그 블록 물체 자리 몫 (균등 0.035)], 층·head 평균")
        for r in range(K):
            nb = C + r
            print(f"   r{r}: " + " ".join(f"b{b}:{res['obj_query_block_mass'][r][b]:.3f}/{res['obj_share_in_block'][r][b]:.2f}" for b in range(nb)))
        print(f"  마지막으로 보인 문맥 블록 분포 (−1 = 없음, 0..7): {res['last_visible_ctx_block_hist']}  · 블록 C−1 가림 비율 {res['frac_hidden_last_ctx']:.2f}")
        if ok.any():
            print("  마지막으로 보인 블록의 물체 자리: 질량 " + " ".join(f"{v:.3f}" for v in res["obj_mass_at_last_visible"])
                  + " | 블록 안 몫 " + " ".join(f"{v:.2f}" for v in res["obj_share_at_last_visible"]))
        if "last_visible_obj_vs_mirror" in res:
            print("  마지막으로 보인 블록: 물체 자리 vs 거울 자리 질량 (거리 같음) " + " ".join(f"{a:.4f}/{b:.4f}" for a, b in res["last_visible_obj_vs_mirror"]))
        print("  문맥 마지막 블록의 물체 자리: 질량 " + " ".join(f"{v:.3f}" for v in res["obj_mass_at_last_ctx"])
              + " | 블록 안 몫 " + " ".join(f"{v:.2f}" for v in res["obj_share_at_last_ctx"]))
    json.dump(res, open(OUT / name / f"{setname}{'_' + subset if subset else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--ckpt", default="")
    ap.add_argument("--set", default="vob_late", choices=["vob_visible", "vob_late", "vob_early", "vob_mid", "ssv2"])
    ap.add_argument("--gpus", type=int, default=4); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--split", choices=["train", "test"], help="VOBench clip 을 split_v2 train / test 로 거른 요약 (<set>_<split>.json) — post-FT 비교용")
    a = ap.parse_args()
    if not a.summarize:
        import torch
        torch.multiprocessing.spawn(_worker, args=(a.gpus, a.name, a.ckpt, a.set, a.n, a.bs), nprocs=a.gpus, join=True)
    summarize(a.set, a.name)
    if a.set != "ssv2" and "vid" in load_all(a.set, a.name):
        for sp in ("train", "test"): summarize(a.set, a.name, sp)
    elif a.split:
        summarize(a.set, a.name, a.split)
