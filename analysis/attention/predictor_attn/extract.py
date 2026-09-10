#!/usr/bin/env python3
"""predictor attention 집계 추출 — 어떤 프레임을 attend 하나, 얼마나 멀리 보나.

  python -m analysis.attention.predictor_attn.extract --dataset v11 --model vith \
      --group-by condition --per-group 32

무엇이 나오나
-------------
  <outdir>/attn.npz       층 x 헤드 x 질의튜블릿 집계 전부 (hooks.py 의 docstring 참고)
  <outdir>/summary.json   메타 + 검증 + 사람이 읽는 층별 요약표

경로·컬럼·체크포인트는 전부 `configs/protocols/{datasets,models}.md` 레지스트리에서
온다. 이 스크립트는 `z_research/scripts/harness/resolve.py` 를 그대로 불러 병합 config 를
만들므로, 데이터셋을 새로 붙일 때 고칠 곳은 여전히 `datasets.md` 섹션 하나뿐이다.
outdir 기본값도 하네스 관례를 따른다:
    <results_root>/predictor_attn__<데이터셋>_<모델>
ViT-H / ViT-L 둘 다 `--model` 인자 하나로 바뀐다 (predictor 는 H·L 이 완전히 같고
인코더 embed_dim 만 1280 / 1024 로 다르다 — models.md 참고).

⚠️ 토큰 캐시를 못 쓴다. 캐시의 `ctx_masked.npy` 는 `_ln(z)` 인데 predictor 가 학습 때
   받은 것은 **LN 이전의 z** 다 (occlusion_identity/forward.py:157-163). 그래서 문맥
   인코더를 실제로 돌린다. 대신 target 인코더는 부르지 않는다 (채점이 아니므로 불필요).

⚠️ 문맥일치 쌍은 predictor 입력이 비트 단위로 같다 (CLAUDE.md §1-2). 즉 pos_a 와
   imp_ab 의 attention 은 **정확히 동일**하다. 기본 `--variants pos_a pos_b` 는 고유
   문맥만 남긴다. `--variants all` 로 끌 수 있지만 그러면 같은 값을 두 번 세는 것이다.

⚠️ 표본 추출이다. `--per-group` 이 그룹당 클립 수이고 `--seed` 로 고정된다.
   attention 은 클립 간 분산이 작지만, 결론을 내기 전에 --per-group 을 두 배로 올려
   값이 안 움직이는지 확인할 것.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from analysis.attention.predictor_attn.geometry import Grid          # noqa: E402
from analysis.attention.predictor_attn.hooks import AttnRecorder     # noqa: E402
from analysis.intphys2.model import build_from_config                # noqa: E402
from analysis.intphys2.surprise import _context_target_indices       # noqa: E402
from evals.world_model_analysis.data import WMADataset               # noqa: E402

RESOLVE = ROOT / "z_research/scripts/harness/resolve.py"


def resolve_config(protocol: str, dataset: str, model: str, sets: list[str]) -> dict:
    """run.sh 와 같은 경로로 병합 config 를 만든다 (경로·인덱스 실물 검사 포함)."""
    with tempfile.NamedTemporaryFile("r", suffix=".yaml", delete=False) as f:
        tmp = f.name
    cmd = [sys.executable, str(RESOLVE), protocol, dataset, model, "-o", tmp]
    for kv in sets:
        cmd += ["--set", kv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode:
        sys.stderr.write(r.stderr)
        sys.exit(f"resolve.py 실패 ({protocol}/{dataset}/{model})")
    cfg = yaml.safe_load(open(tmp))
    os.unlink(tmp)
    return cfg


def sample_groups(ds, cols, variants, variant_col, per_group, seed):
    """-> {group_key: [record index, ...]}  그룹은 index.csv 컬럼 조합으로 만든다."""
    buckets = collections.defaultdict(list)
    for i, rec in enumerate(ds.records):
        raw = rec.raw
        if variants and variant_col and raw.get(variant_col) not in variants:
            continue
        key = "|".join(str(raw.get(c, "?")) for c in cols) if cols else "all"
        buckets[key].append(i)
    rng = np.random.default_rng(seed)
    out = {}
    for k in sorted(buckets):
        idx = np.asarray(buckets[k])
        if per_group and len(idx) > per_group:
            idx = idx[np.sort(rng.permutation(len(idx))[:per_group])]
        out[k] = idx.tolist()
    return out


def batched(ds, indices, bs, workers):
    """클립 디코딩(PNG 32장, 단일 스레드 ~172ms)을 forward 와 겹친다."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        chunks = [indices[a:a + bs] for a in range(0, len(indices), bs)]
        pending = collections.deque()
        for c in chunks[:2]:
            pending.append((c, ex.submit(lambda cc: [ds.clip(i) for i in cc], c)))
        for nxt in chunks[2:]:
            c, fut = pending.popleft()
            pending.append((nxt, ex.submit(lambda cc: [ds.clip(i) for i in cc], nxt)))
            yield c, torch.stack(fut.result())
        while pending:
            c, fut = pending.popleft()
            yield c, torch.stack(fut.result())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default="surprise_c16t32",
                    help="프레임 배치·dtype 관례를 가져올 프로토콜 (기본: 채점 경로와 동일)")
    ap.add_argument("--dataset", default="v11")
    ap.add_argument("--model", default="vith", help="vith | vitl (models.md)")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="K=V",
                    help="병합 config 점 경로 덮어쓰기 (run.sh 의 SET= 과 같다)")
    ap.add_argument("--group-by", default="condition",
                    help="index.csv 컬럼을 콤마로. 빈 문자열이면 그룹 하나로 묶는다")
    ap.add_argument("--variants", nargs="*", default=["pos_a", "pos_b"],
                    help="고유 문맥만 남긴다. 'all' 이면 끄기")
    ap.add_argument("--per-group", type=int, default=32, help="그룹당 클립 수 (0 = 전부)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--q-chunk", type=int, default=512, help="질의 청크. S 의 배수")
    ap.add_argument("--workers", type=int, default=8, help="PNG 디코드 스레드")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("-o", "--outdir", default=None)
    a = ap.parse_args()

    cfg = resolve_config(a.protocol, a.dataset, a.model, a.sets)
    grid = Grid.from_config(cfg)
    outdir = Path(a.outdir) if a.outdir else \
        Path(cfg["output_dir"]).parent / f"predictor_attn__{a.dataset}_{a.model}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"  grid     : {grid.describe()}")
    print(f"  output   : {outdir}")

    ds = WMADataset(cfg, limit=cfg.get("limit"))
    cols = [c for c in a.group_by.split(",") if c.strip()]
    vcol = cfg["data"].get("variant_column") or "variant"
    variants = None if (not a.variants or a.variants == ["all"]) else set(a.variants)
    groups = sample_groups(ds, cols, variants, vcol, a.per_group, a.seed)
    gkeys = list(groups)
    n_total = sum(len(v) for v in groups.values())
    print(f"  groups   : {len(gkeys)}  x  클립 {[len(groups[k]) for k in gkeys]}  "
          f"= {n_total}  (variants={sorted(variants) if variants else 'all'})")
    if not n_total:
        sys.exit("표본이 0개다. --group-by / --variants 를 확인할 것")

    device = torch.device(a.device)
    torch.cuda.set_device(device)
    t0 = time.time()
    bundle = build_from_config(cfg["model"], device)
    print(f"  model    : {cfg['model']['arch_name']}  embed {bundle.embed_dim}  "
          f"dtype {bundle.dtype}  ({time.time()-t0:.0f}s 로딩)")

    feat = cfg.get("features") or cfg.get("surprise") or {}
    mask_index = int(feat.get("mask_index", 0))
    ac_name = cfg["model"].get("autocast")
    ac = (torch.autocast("cuda", dtype=getattr(torch, ac_name)) if ac_name
          else contextlib.nullcontext())
    ctx_f, tgt_f = grid.ctx_frames, grid.n_frames - grid.ctx_frames

    rec = AttnRecorder(bundle.predictor, grid, len(gkeys), device,
                       q_chunk=a.q_chunk, verify=not a.no_verify)
    t0, done = time.time(), 0
    with rec, torch.no_grad():
        for gi, gk in enumerate(gkeys):
            for idxs, clips in batched(ds, groups[gk], a.batch_size, a.workers):
                clips = clips.to(device=device, dtype=bundle.dtype, non_blocking=True)
                B = clips.size(0)
                ctx_idx, tgt_idx = _context_target_indices(
                    ctx_frames=ctx_f, tgt_frames=tgt_f, tubelet_size=grid.tubelet,
                    spatial_tokens=grid.S, batch_size=B, device=device)
                rec.set_group(gi, B)
                with ac:
                    z = bundle.context_encoder(clips, masks=[ctx_idx])
                    z = z[-1] if isinstance(z, list) else z
                    bundle.predictor(z, ctx_idx, tgt_idx, mask_index=mask_index)
                done += B
                el = time.time() - t0
                print(f"\r  {done}/{n_total} clip  {el:6.1f}s  "
                      f"({1000*el/max(1,done):.0f} ms/clip)  [{gk}]   ", end="", flush=True)
    print()

    out = rec.numpy()
    # 정합성: mass 의 행 합이 1 이어야 한다 (softmax 를 튜블릿으로 접은 것뿐이다)
    rs = out["mass"].sum(-1)                              # (G, L, H, T)
    seen = np.broadcast_to(out["n_query"][:, None, None, :] > 0, rs.shape)
    err = float(np.abs(rs[seen] - 1).max())
    if err > 1e-3:
        sys.exit(f"검증 실패: mass 행 합이 1 에서 {err:.2e} 만큼 벗어났다")

    np.savez_compressed(
        outdir / "attn.npz",
        groups=np.array(gkeys), group_by=np.array(cols or ["all"]),
        T=grid.T, C=grid.C, S=grid.S, G=grid.G, tubelet=grid.tubelet,
        patch=grid.patch, n_frames=grid.n_frames, frames_stride=grid.stride,
        n_layers=rec.L, n_heads=rec.H, **out)

    C, T = grid.C, grid.T
    fut = slice(C, T)                                     # 미래(mask token) 질의
    lay = []
    for li in range(rec.L):
        m = out["mass"][:, li].mean(0)                    # (H, T, T) 그룹 평균
        lay.append(dict(
            layer=li,
            ctx_mass=round(float(m[:, fut, :C].sum(-1).mean()), 4),
            self_tub=round(float(np.mean([m[:, t, t] for t in range(C, T)])), 4),
            sdist_px=round(float(out["sdist_px"][:, li, :, fut].mean()), 2),
            tdist_frames=round(float(out["tdist_frames"][:, li, :, fut].mean()), 3),
            toff_frames=round(float(out["toff_frames"][:, li, :, fut].mean()), 3),
            entropy=round(float(out["entropy"][:, li, :, fut].mean()), 3),
            same_pos_ratio=round(float(out["same"][:, li, :, fut].sum(-1).mean()
                                       * grid.S), 2)))
    summary = dict(
        meta=dict(protocol=a.protocol, dataset=a.dataset, model=a.model,
                  arch=cfg["model"]["arch_name"], embed_dim=int(bundle.embed_dim),
                  dtype=str(bundle.dtype), autocast=ac_name,
                  n_layers=rec.L, n_heads=rec.H, grid=grid.describe(),
                  group_by=cols, groups=gkeys,
                  n_clips={k: int(v) for k, v in zip(gkeys, out["n_clips"])},
                  variants=sorted(variants) if variants else "all",
                  per_group=a.per_group, seed=a.seed,
                  entropy_max=round(float(np.log(grid.N)), 3)),
        verify=dict(row_sum_max_err=err, **(rec.verify_stat or {})),
        by_layer=lay, config=cfg)
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    v = rec.verify_stat
    print(f"  검증     : mass 행 합 오차 {err:.2e}"
          + (f" | 재계산 attn@v vs 실제 출력 max|Δ| {v['max_abs_diff']:.3e} "
             f"(출력 평균크기 {v['mean_abs_out']:.3f})" if v else ""))
    print(f"\n  미래 질의 기준 층별 요약 (그룹 평균, 엔트로피 상한 {np.log(grid.N):.2f})")
    print(f"  {'층':>3} {'문맥질량':>8} {'자기튜블릿':>10} {'공간거리px':>11} "
          f"{'시간거리f':>10} {'부호Δt':>8} {'엔트로피':>9} {'동일패치배율':>12}")
    for r in lay:
        print(f"  {r['layer']:>3} {r['ctx_mass']:>8.3f} {r['self_tub']:>10.3f} "
              f"{r['sdist_px']:>11.1f} {r['tdist_frames']:>10.2f} {r['toff_frames']:>8.2f} "
              f"{r['entropy']:>9.2f} {r['same_pos_ratio']:>12.1f}")
    print(f"\n  [saved] {outdir}/attn.npz + summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
