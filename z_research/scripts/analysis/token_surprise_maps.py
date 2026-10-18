#!/usr/bin/env python3
"""토큰별 surprise 지도 — `surprise_c16t32` 채점을 **토큰 단위로 다시** 계산해 저장한다 (2026-10-07).

채점 (evals/world_model_analysis/eval.py `run_surprise`) 은 영상마다 mean_{토큰, 채널} |p − LN(h)| 스칼라 하나만 남긴다.
여기서는 같은 config (`<run>/_resolved.yaml`) · 같은 forward (fp32 가중치 + fp16 autocast, 같은 batch) 로
토큰별 e[t, i, j] = mean_채널 |p − LN(h)| 를 남긴다. 미래 8 튜블릿 × 16 × 16 패치.

  Δ[t, i, j] = e_imp − e_pos   (matched pair: 문맥이 같아 p 가 같다)
  mean(Δ) = s_imp − s_pos     ← 채점 차이와 **정확히** 같은 양 (토큰 평균 = 영상 surprise)
  Δ > 0 = 그 토큰이 불가능 쪽을 더 놀랍게 본다 (정답 방향)

검증: 영상마다 mean(e) 가 `<run>/per_block.json` 의 per_video_surprise 와 같아야 한다 (상대 1e-3 넘으면 죽는다;
fp16 batch 구성이 채점 때와 달라 마지막 자리가 흔들릴 수 있어 0 이 아니라 문턱을 둔다 — CLAUDE.md §7-2).

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/token_surprise_maps.py --run z_research/previous/v11_realistic/exp_results/surprise_c16t32__v11_realistic_ledge_vith --gpus 4
  → <run>/token_surprise.npz  (video_id (n,), e (n, 8, 16, 16) float32, s_check (n,))
  --copy (2026-10-10): 같은 forward 에서 복사 기준선 토큰 지도 ec 도 남긴다 —
     ec[t, i, j] = mean_채널 |LN(z)[문맥 마지막 튜블릿, i, j] − LN(h)[t, i, j]|   (cvpr/eval/testbed/copy_baseline 의 copy 와 같은 정의)
  predictor kind=ar (Ariel 팔 C, 2026-10-11): 채점기 (eval.py) 와 같은 경로 — 블록별 LN(target_encoder) 문맥 → rollout,
     타깃도 블록별 (ARS.encode_blocks · rollout_from_blocks = ar_window). 복사 ec 는 같은 공간의 copy_blk = h_blk[문맥 마지막 블록].
     ⚠️ 표준과 타깃 공간이 다르다 — e · ec 의 크기는 다른 kind 와 비교하지 않는다 (고르기 · 쌍 판정만).
"""
from __future__ import annotations
import argparse, contextlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def shard(run: Path, i: int, n: int, out: Path, copy: bool = False):
    import torch, yaml
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.data import WMADataset
    from analysis import predictors as PV
    from analysis.predictors import ar_scoring as ARS
    cfg = yaml.safe_load((run / "_resolved.yaml").read_text())
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    torch.set_num_threads(8)
    ds = WMADataset(cfg)
    S = cfg["surprise"]
    bundle = build_from_config(cfg["model"], dev)
    ts, sp = bundle.tubelet_size, bundle.num_spatial_tokens
    ctx, N, bs = int(S["context_length"]), ds.n_frames, int(os.environ.get("TOKEN_BS") or S.get("batch_size", 4))   # TOKEN_BS: 다른 job 과 GPU 를 나눌 때 batch 를 줄인다 (2026-10-15)
    ci, ti = _context_target_indices(ctx_frames=ctx, tgt_frames=N - ctx, tubelet_size=ts, spatial_tokens=sp, batch_size=bs, device=dev)
    ac = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(str(cfg["model"].get("autocast", "none")).lower())
    assert S.get("distance", "l1") == "l1" and float(S.get("loss_exp", 1.0)) == 1.0, "토큰 분해는 l1 · loss_exp 1 에서만"
    if PV.is_ctx_ar(bundle.predictor):
        raise SystemExit("kind=ctx_ar 는 아직 지원하지 않는다")
    ar = PV.is_ar(bundle.predictor)
    if ar:
        ARS.check_ar_ready(bool(S.get("target_layer_norm", True)))
    idx = list(range(i, len(ds), n))
    vids, E, EC = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            chunk = idx[s:s + bs]; m = len(chunk)
            x = torch.stack([ds.clip(k) for k in chunk]).to(dev, bundle.dtype)
            with (torch.autocast("cuda", dtype=ac) if ac else contextlib.nullcontext()):
                if ar:                                                      # eval.py 의 ar_window 와 같은 두 단계
                    hb = ARS.encode_blocks(bundle.target_encoder, x, ts)    # (m, nb*S, D) 블록별 · 이미 LN
                    p, h, _ = ARS.rollout_from_blocks(bundle.predictor, hb, ctx // ts, sp)
                    zl_ar = hb[:, (ctx // ts - 1) * sp:(ctx // ts) * sp]    # copy_blk: 문맥 마지막 블록 (LN 공간)
                else:
                    z = bundle.context_encoder(x, masks=[ci[:m]])
                    p = bundle.predictor(z, ci[:m], ti[:m], mask_index=int(S.get("mask_index", 0)))
                    h = bundle.target_encoder(x)
                    h = torch.gather(h, 1, ti[:m].unsqueeze(-1).expand(-1, -1, h.size(-1)))
                    if S.get("target_layer_norm", True):
                        h = torch.nn.functional.layer_norm(h, (h.size(-1),))
            e = (p.float() - h.float()).abs().mean(-1)                      # (m, T)  — eval.py token_subset 경로와 같은 식
            E.append(e.reshape(m, (N - ctx) // ts, int(sp ** 0.5), int(sp ** 0.5)).cpu().numpy().astype(np.float32))
            if copy:                                                        # 복사: 문맥 encoder 마지막 튜블릿 LN(z) 를 미래 튜블릿마다
                zl = zl_ar.float() if ar else torch.nn.functional.layer_norm(z[:, -sp:].float(), (z.size(-1),))
                ec = (zl.unsqueeze(1) - h.float().reshape(m, -1, sp, h.size(-1))).abs().mean(-1)
                EC.append(ec.reshape(m, -1, int(sp ** 0.5), int(sp ** 0.5)).cpu().numpy().astype(np.float32))
            vids += [ds.records[k].video_id for k in chunk]
            if (s // bs) % 10 == 0:
                print(f"[shard {i}] {s + m}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
    np.savez(out, video_id=np.array(vids), e=np.concatenate(E), **({"ec": np.concatenate(EC)} if copy else {}))
    print(f"[shard {i}] 끝 {len(vids)} 개 {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--merge-only", action="store_true", help="이미 뽑은 _token_shards 만 합치고 검증 (GPU 불필요)")
    ap.add_argument("--copy", action="store_true", help="복사 기준선 토큰 지도 ec 도 저장")
    a = ap.parse_args()
    run = a.run if a.run.is_absolute() else REPO / a.run
    if a.shard is not None:
        return shard(run, a.shard, a.gpus, a.out, a.copy)
    tmp = run / "_token_shards"; tmp.mkdir(exist_ok=True)
    procs = []
    for i in ([] if a.merge_only else range(a.gpus)):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        procs.append(subprocess.Popen([sys.executable, "-u", __file__, "--run", str(run), "--gpus", str(a.gpus), "--shard", str(i),
                                       "--out", str(tmp / f"s{i}.npz")] + (["--copy"] if a.copy else []), env=env))
    bad = [i for i, p in enumerate(procs) if p.wait() != 0]
    if bad:
        sys.exit(f"shard {bad} 실패")
    vid, e, ec = [], [], []
    for i in range(a.gpus):
        with np.load(tmp / f"s{i}.npz") as q:          # 핸들을 닫아야 NFS 에서 지울 수 있다 (.nfs 잔여 파일)
            vid.append(q["video_id"]); e.append(q["e"])
            if "ec" in q:
                ec.append(q["ec"])
    vid, e = np.concatenate(vid), np.concatenate(e)
    o = np.argsort(vid); vid, e = vid[o], e[o]
    extra = {"ec": np.concatenate(ec)[o]} if ec else {}
    # 검증: 토큰 평균 = 채점 때의 영상 surprise
    S = json.loads((run / "per_block.json").read_text())["per_video_surprise"]
    ref = np.array([S[v] for v in vid]); got = e.reshape(len(e), -1).mean(1)
    rel = np.abs(got - ref) / ref
    print(f"[검증 1] 영상 {len(vid)} 개 · 토큰 평균 vs per_block.json 상대 오차 최대 {rel.max():.2e} · 중앙 {np.median(rel):.2e}")
    # 문턱 2e-3: 채점 때와 GPU 수 · batch 구성이 다르면 fp16 autocast 의 마지막 자리가 흔들린다 (CLAUDE.md §7-2;
    # 2026-10-07 실측 — ledge 6.0e-4 · 실물 1.1e-3, 둘 다 중앙 1e-4 대). 처음 둔 1e-3 은 근거 없이 정한 값이었다.
    if rel.max() > 2e-3:
        sys.exit("검증 실패 — 토큰 평균이 채점 값과 다르다")
    # 검증 2: matched pair 판정 (불가능 > 가능) 이 채점과 같은가 — 지도가 채점과 같은 결론을 내는지가 쓰임새의 전제다
    import csv
    cfg_root = Path(next(l.split(":", 1)[1].strip() for l in (run / "_resolved.yaml").read_text().splitlines() if l.strip().startswith("root:")))
    rows = list(csv.DictReader((cfg_root / "index.csv").open()))
    grp = {}
    for r in rows:                                     # 문맥 묶음 (block, pair_id) — 불가능이 여럿이면 가능과 하나씩 (gravity_realistic)
        grp.setdefault((r["block_id"], r["pair_id"]), []).append(r)
    G = dict(zip(vid, got)); flips, n, near = 0, 0, 0
    for v in grp.values():
        ps = [r["video_id"] for r in v if r["plausible"] == "1"]
        if len(ps) != 1 or ps[0] not in G:
            continue
        for q in (r["video_id"] for r in v if r["plausible"] != "1"):
            n += 1
            a_ref, a_got = S[q] > S[ps[0]], G[q] > G[ps[0]]
            if a_ref != a_got:
                flips += 1
                near += abs(S[q] - S[ps[0]]) / S[ps[0]] < 4e-3
    print(f"[검증 2] matched pair {n} 개 중 판정이 채점과 다른 쌍 {flips} 개 (그중 |차| < 0.4 % 근접 tie {near} 개)")
    if flips - near > 0:
        sys.exit("검증 실패 — 근접 tie 가 아닌 쌍의 판정이 채점과 다르다")
    np.savez_compressed(run / "token_surprise.npz", video_id=vid, e=e, s_check=got, s_ref=ref, **extra)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"→ {run / 'token_surprise.npz'}  e {e.shape}")


if __name__ == "__main__":
    main()
