#!/usr/bin/env python3
"""phys2d_v1 **held-out** 창에서 predictor 여러 개의 예측 L1 을 **복사 기준선과 같이** 낸다 (시나리오별).

왜 이게 필요한가 — `z_training/RESULTS_2026-09-19.md` 의 주장 틀은 "post-FT 는 **학습 도메인 안에서만** 오른다"
인데, phys2d post-FT (2026-10-18/19) 는 지금 **도메인 밖** (IntPhys1 in-loop val) 만 쟀다. 안쪽이 비면
"도메인 안에서만 오른다" 와 "학습이 그냥 망했다" 를 못 가른다. 이 스크립트가 안쪽을 채운다.

⚠️ **이건 VoE 가 아니라 예측 정확도다.** phys2d_v1 에는 가능/불가능 쌍이 없다 (2026-10-19 확인:
   `boundary.tsv`/`manifest.tsv` 에 plausible·violation 컬럼 없음, 28,900 clip 전부 물리적으로 맞다).
   따라서 여기서 "불가능을 구분한다" 는 주장은 **나올 수 없다** — 그건 IntPhys1 · v11 채점에서만 나온다.

## 정의 (창 i, predictor k, 미래 슬롯 j = 0..7)

창은 `build_phys2d_window_index.py` 가 고른 T=32 · C=16 · stride 1 짜리다 (사건이 문맥에 없고 예측 구간에 있다).
수치 관례는 `surprise_c16t32` 와 같다 (CLAUDE.md §1-7): fp32 가중치 + autocast fp16,
target 은 affine 없는 토큰 LN, L1, mask_index 0.

    z      = context_encoder(문맥 16 장)                 (online `encoder`)        (B, 8·S, D)
    h      = LN(target_encoder(32 장))[미래 8 튜블릿]    (affine 없는 LN)          (B, 8·S, D)
    p_k    = predictor_k(z, 문맥 idx, 미래 idx, mask_index=0)                       (B, 8·S, D)
    zT     = LN(z)[문맥 마지막 튜블릿 7]                 (te_v11_score.py 와 같은 식)

    l1  [i,k,j] = mean | p_k[j] − h[j] |     ← 예측 오차 (작을수록 좋다)
    copy[i,  j] = mean | zT    − h[j] |      ← 예측 없는 복사 기준선
    gain[i,k,j] = copy[i,j] − l1[i,k,j]      ← **0 이하면 그 predictor 는 복사보다 못하다**

⚠️ 두 팔의 predictor 는 학습 창 길이가 다르다 (A 32 프레임 · B 16 프레임). 여기서는 **둘 다 32 프레임 창으로**
   채점한다 — 릴리즈 predictor 자체가 16 프레임으로 학습되고 32 프레임 프로토콜에서 쓰이므로 (CLAUDE.md §1-7)
   같은 관례다. 이렇게 해야 두 팔이 같은 자로 재어진다 (학습 창 교락은 남지만 **채점 창 교락은 없앴다**).

## 출력 — <out>/

    l1.npy    (n, K, 8) float32      copy.npy (n, 8) float32
    meta.json                        창 행 (path·start·scenario·category) · predictor 태그/경로 · 설정 · D · S
    summary.json                     시나리오 × predictor 평균 L1 · copy · gain

## 쓰는 법

    PY=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python     # vll7 은 sm_120 이라 이 env 만 돈다
    CUDA_VISIBLE_DEVICES=0 $PY z_research/scripts/analysis/phys2d_heldout_l1.py \
        --preds release,futureA=z_vll7_training/runs/phys2d_future_postft/latest.pt,tubeletB=z_vll7_training/runs/phys2d_tubelet_postft/latest.pt
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.vjepa_frozen import utils as U
from app.vjepa_frozen.data import ClipTransform

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEF_PARAMS = os.path.join(REPO, "z_vll7_training/runs/phys2d_future_postft/params-train.yaml")
DEF_CSV = os.path.join(REPO, "data_csv/phys2d_v1/val_T32C16s1.csv")
DEF_OUT = os.path.join(REPO, "z_research/previous/TrainingEffects/exp_results/phys2d_heldout")


def read_rows(csv_path, limit=None):
    import csv as _csv
    with open(csv_path) as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path}: 빈 파일")
    if limit:
        rows = rows[:limit]
    return rows


def parse_preds(spec, base_ck):
    """'release,tag=/abs/path.pt' -> [(tag, path|None)]  (None = base model.pth 의 릴리즈 predictor)"""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "release":
            out.append(("release", None))
        elif "=" in tok:
            tag, p = tok.split("=", 1)
            p = p if os.path.isabs(p) else os.path.join(REPO, p)
            if not os.path.isfile(p):
                raise FileNotFoundError(f"{tag}: {p} 가 없다")
            out.append((tag, p))
        else:
            raise ValueError(f"predictor 스펙을 모르겠다: {tok!r} (release | tag=/path.pt)")
    if not out:
        raise ValueError("--preds 가 비었다")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=DEF_PARAMS, help="model 블록을 읽을 params-train.yaml")
    ap.add_argument("--csv", default=DEF_CSV, help="창 인덱스 (build_phys2d_window_index.py 산출물)")
    ap.add_argument("--preds", default="release,"
                    "futureA=z_vll7_training/runs/phys2d_future_postft/latest.pt,"
                    "tubeletB=z_vll7_training/runs/phys2d_tubelet_postft/latest.pt")
    ap.add_argument("--context", type=int, default=16, help="문맥 프레임 수")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=DEF_OUT)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.params))
    MD, DD = cfg["model"], cfg["data"]
    n_frames, res = int(DD["n_frames"]), int(DD.get("resolution", 256))
    tub, psz = int(MD.get("tubelet_size", 2)), int(MD.get("patch_size", 16))
    if a.context % tub:
        raise ValueError(f"context {a.context} 가 tubelet {tub} 로 안 나뉜다")
    S = (res // psz) ** 2
    n_ctx, N = a.context // tub * S, n_frames // tub * S
    n_slot = (n_frames - a.context) // tub

    rows = read_rows(a.csv, a.limit or None)
    bad = {int(r["n_frames"]) for r in rows} - {n_frames}
    if bad:
        raise ValueError(f"{a.csv}: 행의 n_frames={sorted(bad)} 가 params 의 data.n_frames={n_frames} 와 다르다")

    dev = torch.device("cuda:0")
    state = U.load_base_checkpoint(MD["checkpoint"])
    # surprise_c16t32 관례: fp32 가중치 + autocast fp16 (CLAUDE.md §1-7)
    ctx_enc, tgt_enc = U.build_frozen_encoders(MD, state, dev, torch.float32, n_frames)
    preds = []
    for tag, path in parse_preds(a.preds, MD["checkpoint"]):
        m = dict(MD)
        if path:
            m["predictor_init_checkpoint"] = path
        p = U.build_predictor(m, ctx_enc.embed_dim, state, dev, n_frames).eval()
        for q in p.parameters():
            q.requires_grad_(False)
        preds.append((tag, path, p))
    print(f"[info] predictor {len(preds)}개: {[t for t, _, _ in preds]}", flush=True)

    # 학습 config 의 aug 는 hflip 뿐 (rrc null, square_crop none) — 채점은 증강 없이
    tf = ClipTransform(res, hflip_p=0.0, random_resized_crop=None, square_crop="none")
    import decord
    ci = torch.arange(n_ctx, device=dev).unsqueeze(0)
    ti = torch.arange(n_ctx, N, device=dev).unsqueeze(0)
    n, K = len(rows), len(preds)
    l1 = np.zeros((n, K, n_slot), np.float32)
    cp = np.zeros((n, n_slot), np.float32)

    def red(x):                                    # (B, n_slot*S, D) -> (B, n_slot)
        return x.abs().float().view(x.size(0), n_slot, S, x.size(-1)).mean(dim=(2, 3)).cpu().numpy()

    t0 = time.time()
    for s in range(0, n, a.batch):
        chunk = rows[s: s + a.batch]
        clips = []
        for r in chunk:
            idx = np.arange(int(r["start"]), int(r["start"]) + n_frames * int(r["stride"]),
                            int(r["stride"]), dtype=np.int64)
            vr = decord.VideoReader(r["path"], num_threads=1)
            if idx[-1] >= len(vr):
                raise IndexError(f"{r['path']}: 창 끝 {idx[-1]} >= 프레임 {len(vr)}")
            clips.append(tf(torch.from_numpy(vr.get_batch(idx).asnumpy())))
        x = torch.stack(clips).to(dev, non_blocking=True)
        b = x.size(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            h = tgt_enc(x)
            h = torch.gather(h, 1, ti.expand(b, -1).unsqueeze(-1).expand(-1, -1, h.size(-1)))
            h = torch.nn.functional.layer_norm(h, (h.size(-1),))
            z = ctx_enc(x, masks=[ci.expand(b, -1)])
            zT = torch.nn.functional.layer_norm(z, (z.size(-1),))[:, -S:]      # 문맥 마지막 튜블릿
            cp[s: s + b] = red(zT.repeat(1, n_slot, 1) - h)
            for k, (_, _, p) in enumerate(preds):
                l1[s: s + b, k] = red(p(z, ci.expand(b, -1), ti.expand(b, -1), mask_index=int(MD.get("mask_index", 0))) - h)
        if (s // a.batch) % 20 == 0:
            el = time.time() - t0
            print(f"[{s + b}/{n}] {el:.0f}s  eta {el / (s + b) * (n - s - b):.0f}s", flush=True)

    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "l1.npy"), l1)
    np.save(os.path.join(a.out, "copy.npy"), cp)
    tags = [t for t, _, _ in preds]
    json.dump({"tags": tags, "paths": {t: p for t, p, _ in preds}, "csv": a.csv, "params": a.params,
               "n": n, "K": K, "n_slot": n_slot, "S": S, "D": int(ctx_enc.embed_dim),
               "n_frames": n_frames, "context": a.context, "resolution": res,
               "dtype": "fp32 weights + autocast fp16 (surprise_c16t32 관례)",
               "rows": [{k: r[k] for k in ("path", "start", "stride", "scenario", "category",
                                           "n_future_events", "first_future_event") if k in r} for r in rows]},
              open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False, indent=1)

    # ── 시나리오별 요약
    scen = np.array([r["scenario"] for r in rows])
    L, C = l1.mean(2), cp.mean(1)                    # (n, K), (n,)
    summ = {"overall": {"n": n, "copy": float(C.mean()),
                        **{t: {"l1": float(L[:, k].mean()), "gain": float((C - L[:, k]).mean())}
                           for k, t in enumerate(tags)}}, "by_scenario": {}}
    for sc in sorted(set(scen)):
        m = scen == sc
        summ["by_scenario"][sc] = {"n": int(m.sum()), "copy": float(C[m].mean()),
                                   **{t: {"l1": float(L[m, k].mean()), "gain": float((C[m] - L[m, k]).mean())}
                                      for k, t in enumerate(tags)}}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)

    w = max(len(s) for s in summ["by_scenario"]) + 1
    print(f"\n{'scenario':{w}s} {'n':>5s} {'copy':>8s}" + "".join(f"{t:>10s}{'Δ':>8s}" for t in tags))
    for sc, d in list(summ["by_scenario"].items()) + [("OVERALL", summ["overall"])]:
        print(f"{sc:{w}s} {d['n']:5d} {d['copy']:8.4f}" +
              "".join(f"{d[t]['l1']:10.4f}{d[t]['gain']:+8.4f}" for t in tags))
    print(f"\nΔ = copy − l1 (양수 = 복사보다 낫다).  {a.out}")


if __name__ == "__main__":
    main()
