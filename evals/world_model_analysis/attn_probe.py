"""Attentive probing — 세 지점의 표현에서 정보가 읽히는지 본다.

  z : context_encoder(x, masks=[ctx_idx])      -> (N_ctx, D)   문맥 구간(1~C) 표현
  p : predictor(z, ctx_idx, tgt_idx)           -> (N_tgt, D)   예측한 미래(C+1~T)
  h : LayerNorm(target_encoder(x)[tgt_idx])    -> (N_tgt, D)   실제 미래  ← 상한/대조군

세 소스 모두 토큰 수가 같다 (C == T-C 인 대칭 프로토콜):
  N_ctx = N_tgt = (16 / tubelet 2) * (256/16)^2 = 8 * 256 = 2048,  D = 1280 (ViT-H)

토큰을 그대로 AttentiveClassifier(=V-JEPA attentive probe)에 넣어 학습/평가한다.
분할은 block 단위로 묶어 같은 block 의 영상이 train/test 로 쪼개지지 않게 한다.

읽는 법
  z 높고 p 낮고 h 높음  -> 정보는 문맥에 있었는데 predictor 가 가림 구간에서 잃었다
  z 낮음                 -> 애초에 문맥 표현에 정보가 안 들어갔다
  p 높은데 surprise 실패 -> 정보는 있는데 전역 mean-token L1 채점이 못 쓴다

  bash z_scripts/world_model_analysis/run_attn_probe.sh attn_probe_v8_vith
"""

import argparse
import collections
import contextlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

from analysis.intphys2.model import build_from_config
from analysis.intphys2.surprise import _context_target_indices
from evals.world_model_analysis.data import WMADataset
from src.models.attentive_pooler import AttentiveClassifier

ALL_SOURCES = ("z", "p", "h")

_AC = {"none": None, "null": None, "float32": None,
       "float16": torch.float16, "fp16": torch.float16,
       "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


def _ac_dtype(name):
    n = str(name).lower()
    if n not in _AC:
        raise ValueError(f"autocast={name!r}; {sorted(_AC)} 중 하나여야 한다")
    return _AC[n]


def _autocast(dtype):
    return torch.autocast("cuda", dtype=dtype) if dtype else contextlib.nullcontext()


def _bar(it, desc, rank, unit="it"):
    """rank 마다 한 줄을 쓰는 진행 막대. stderr 로 나간다 (stdout 은 로그 파일).

    disable=None(비-TTY 자동 끄기)은 쓰지 않는다 — 실행 스크립트가 stderr 를 그대로
    터미널에 흘리는데도 상황에 따라 TTY 판정이 어긋나 막대가 통째로 사라진다.
    """
    return tqdm(it, desc=desc, position=rank, leave=True, dynamic_ncols=True,
                file=sys.stderr, unit=unit, mininterval=0.5, disable=False)


# ---------------------------------------------------------------- 특징 추출
def _signature(cfg, sources, world):
    d, s = cfg["data"], cfg["surprise"]
    return {
        "root": d["root"],
        "index_csv": d.get("index_csv", "index.csv"),
        "frames_root": d.get("frames_root"),
        "frames_start": d.get("frames_start", 0),
        "frames_stride": d.get("frames_stride", 1),
        "n_frames": d["n_frames"],
        "resolution": d.get("resolution", 256),
        "context_length": s["context_length"],
        "target_layer_norm": s.get("target_layer_norm", True),
        "mask_index": s.get("mask_index", 0),
        "checkpoint": cfg["model"]["checkpoint"],
        "dtype": cfg["model"].get("dtype", "bfloat16"),
        "autocast": str(cfg["model"].get("autocast", "none")).lower(),
        "limit": cfg.get("limit"),
        "sources": list(sources),
        "world": int(world),
    }


def _bounds(n, world):
    """rank 별 [start, end) 연속 구간. 연속이라 전역 인덱스 -> shard 매핑이 단순하다."""
    e = np.linspace(0, n, world + 1).astype(int)
    return list(zip(e[:-1], e[1:]))


def _shard_path(cache_dir, src, r, world):
    return os.path.join(cache_dir, f"tok_{src}_s{r}of{world}.npy")


class _ClipDS(torch.utils.data.Dataset):
    """PNG 디코딩을 워커 프로세스로 넘기기 위한 얇은 래퍼. 워커마다 WMADataset 을 따로 만든다."""

    def __init__(self, cfg, indices):
        self.cfg, self.indices, self._ds = cfg, list(indices), None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        if self._ds is None:
            self._ds = WMADataset(self.cfg)
        return self._ds.clip(self.indices[j])


def extract(cfg, cache_dir, device, sources, rank=0, world=1):
    """이 rank 가 맡은 구간의 z/p/h 를 뽑아 shard npy 로 저장한다."""
    os.makedirs(cache_dir, exist_ok=True)
    sig = _signature(cfg, sources, world)
    sig_path = os.path.join(cache_dir, f"signature_s{rank}of{world}.json")
    meta_path = os.path.join(cache_dir, f"meta_s{rank}of{world}.json")
    paths = {s: _shard_path(cache_dir, s, rank, world) for s in sources}

    recache = cfg.get("features", {}).get("recache", False)
    if (not recache and os.path.exists(sig_path) and json.load(open(sig_path)) == sig
            and os.path.exists(meta_path) and all(os.path.exists(v) for v in paths.values())):
        print(f"[rank {rank}] 캐시 재사용", flush=True)
        return

    ds = WMADataset(cfg)
    n = len(ds) if cfg.get("limit") is None else min(len(ds), int(cfg["limit"]))
    lo, hi = _bounds(n, world)[rank]
    idxs = list(range(lo, hi))

    model = build_from_config(cfg["model"], device)
    ac = _ac_dtype(cfg["model"].get("autocast", "none"))
    s = cfg["surprise"]
    fx = cfg.get("features", {})
    bs = int(fx.get("batch_size", 8))

    ctx_full, tgt_full = _context_target_indices(
        ctx_frames=s["context_length"],
        tgt_frames=ds.n_frames - s["context_length"],
        tubelet_size=model.tubelet_size,
        spatial_tokens=model.num_spatial_tokens,
        batch_size=bs,
        device=device,
    )
    n_tok, dim = tgt_full.shape[1], model.embed_dim
    if ctx_full.shape[1] != n_tok:
        raise ValueError(
            f"context 토큰 {ctx_full.shape[1]} != target 토큰 {n_tok}. "
            "z/p/h 를 같은 shape 로 다루려면 context_length == n_frames - context_length 여야 한다")

    store = np.dtype(fx.get("dtype", "float16"))
    print(f"[rank {rank}] extract {lo}~{hi} (n={len(idxs)}) tokens={n_tok} dim={dim} "
          f"batch={bs} dtype={store} sources={list(sources)}", flush=True)

    mm = {k: np.lib.format.open_memmap(v, "w+", store, (len(idxs), n_tok, dim))
          for k, v in paths.items()}

    loader = torch.utils.data.DataLoader(
        _ClipDS(cfg, idxs), batch_size=bs, shuffle=False,
        num_workers=int(fx.get("decode_workers", 8)),
        pin_memory=True, persistent_workers=False,
    )

    # 진행 막대는 stderr 로 나간다 (stdout 은 로그 파일). rank 마다 줄을 하나씩 차지한다.
    # 터미널이 아니면 (disable=None) 알아서 꺼져서 로그에 \r 이 쌓이지 않는다.
    bar = _bar(loader, f"extract r{rank}", rank, unit="batch")

    w = 0
    with torch.inference_mode():
        for x in bar:
            b = x.size(0)
            x = x.to(device, model.dtype, non_blocking=True)
            ci, ti = ctx_full[:b], tgt_full[:b]
            out = {}
            with _autocast(ac):
                zc = model.context_encoder(x, masks=[ci])
                if isinstance(zc, list):
                    zc = zc[-1]
                if "z" in paths:
                    out["z"] = zc
                if "p" in paths:
                    out["p"] = model.predictor(zc, ci, ti, mask_index=s.get("mask_index", 0))
                if "h" in paths:
                    hh = model.target_encoder(x)
                    if isinstance(hh, list):
                        hh = hh[-1]
                    hh = torch.gather(hh, 1, ti.unsqueeze(-1).expand(-1, -1, hh.size(-1)))
                    if s.get("target_layer_norm", True):
                        hh = nn.functional.layer_norm(hh, (hh.size(-1),))
                    out["h"] = hh
            for k, v in out.items():
                mm[k][w:w + b] = v.float().cpu().numpy().astype(store)
            w += b
            bar.set_postfix_str(f"{w}/{len(idxs)} clip")
    bar.close()

    for v in mm.values():
        v.flush()
    json.dump([{"video_id": ds.records[i].video_id, **dict(ds.records[i].raw)} for i in idxs],
              open(meta_path, "w"))
    json.dump(sig, open(sig_path, "w"), indent=1)
    del model, mm
    torch.cuda.empty_cache()
    print(f"[rank {rank}] extract 완료 ({len(idxs)}개)", flush=True)


def merge(cfg, cache_dir, sources, world):
    """rank shard 들의 meta 를 순서대로 이어붙이고 manifest 를 쓴다 (토큰은 복사하지 않는다)."""
    sig = _signature(cfg, sources, world)
    metas, sizes = [], []
    for r in range(world):
        sp = os.path.join(cache_dir, f"signature_s{r}of{world}.json")
        if not os.path.exists(sp) or json.load(open(sp)) != sig:
            raise RuntimeError(f"shard {r} 의 서명이 없거나 다르다 -> extract 를 다시 돌려라 ({sp})")
        m = json.load(open(os.path.join(cache_dir, f"meta_s{r}of{world}.json")))
        metas += m
        sizes.append(len(m))
    manifest = {"world": world, "sizes": sizes, "sources": list(sources),
                "paths": {s: [_shard_path(cache_dir, s, r, world) for r in range(world)]
                          for s in sources}}
    json.dump(metas, open(os.path.join(cache_dir, "meta.json"), "w"))
    json.dump(manifest, open(os.path.join(cache_dir, "manifest.json"), "w"), indent=1)
    json.dump(sig, open(os.path.join(cache_dir, "signature.json"), "w"), indent=1)
    print(f"[merge] {len(metas)}개 샘플, shard {sizes}", flush=True)
    return metas, manifest


class ShardedTokens:
    """rank shard npy 여러 개를 하나의 (N, T, D) 배열처럼 fancy-index 로 읽는다."""

    def __init__(self, paths):
        self.mm = [np.load(p, mmap_mode="r") for p in paths]
        self.off = np.cumsum([0] + [m.shape[0] for m in self.mm])
        self.shape = (int(self.off[-1]), self.mm[0].shape[1], self.mm[0].shape[2])
        self.dtype = self.mm[0].dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        out = np.empty((len(idx),) + self.shape[1:], dtype=self.dtype)
        which = np.searchsorted(self.off, idx, side="right") - 1
        for s in np.unique(which):
            m = which == s
            out[m] = self.mm[s][idx[m] - self.off[s]]
        return out


# ---------------------------------------------------------------- 과제 구성
def build_task(meta, spec):
    """index.csv 컬럼으로 (샘플 인덱스, 라벨, 클래스 이름) 을 만든다."""
    idx = list(range(len(meta)))
    for col, allowed in (spec.get("subset") or {}).items():
        allowed = {str(a) for a in allowed}
        idx = [i for i in idx if str(meta[i].get(col, "")) in allowed]

    col = spec["label_column"]
    binz = spec.get("binarize")
    if binz:
        neg = {str(a) for a in binz["negative"]}
        lab = {i: (0 if str(meta[i].get(col, "")) in neg else 1) for i in idx}
        names = binz.get("names", ["negative", "positive"])
    else:
        drop = {str(a) for a in (spec.get("exclude_labels") or [])}
        idx = [i for i in idx if str(meta[i].get(col, "")) not in drop]
        names = sorted({str(meta[i][col]) for i in idx})
        c2i = {c: j for j, c in enumerate(names)}
        lab = {i: c2i[str(meta[i][col])] for i in idx}

    if spec.get("min_per_class"):
        cnt = collections.Counter(lab.values())
        keep = {c for c, v in cnt.items() if v >= int(spec["min_per_class"])}
        idx = [i for i in idx if lab[i] in keep]
        old = sorted(keep)
        remap = {c: j for j, c in enumerate(old)}
        lab = {i: remap[lab[i]] for i in idx}
        names = [names[c] for c in old]
    return idx, lab, names


def token_slice(spec, src, cfg, n_tok):
    """task 의 token_frames 를 토큰 slice 로 바꾼다.

    token_frames 는 **소스별로** 준다. 각 소스가 덮는 구간이 다르기 때문이다.
      z : 문맥 프레임 1~C     (v8 기준 1~16, 가림 구간은 13~16 = raw 36,39,42,45)
      p : 미래 프레임 1~(T-C) (v8 기준 1~16, 가림 구간은 1~4  = raw 48,51,54,57)
      h : p 와 동일
    예) token_frames: {z: [13, 16], p: [1, 4], h: [1, 4]}  -> 가림 구간 토큰만
    """
    tf = (spec.get("token_frames") or {}).get(src)
    if not tf:
        return None
    tub = int(cfg["model"].get("tubelet_size", 2))
    spatial = (int(cfg["model"]["img_size"]) // int(cfg["model"]["patch_size"])) ** 2
    f0, f1 = int(tf[0]) - 1, int(tf[1])
    if f0 % tub or f1 % tub:
        raise ValueError(f"token_frames {tf} 가 tubelet({tub}) 정렬이 아니다")
    a, b = f0 // tub * spatial, f1 // tub * spatial
    if not (0 <= a < b <= n_tok):
        raise ValueError(f"token_frames {tf} -> 토큰 [{a},{b}) 가 범위(0,{n_tok}) 밖")
    return slice(a, b)


def make_splits(meta, idx, pc, group_by):
    """(train, test) 목록. 분할은 항상 block 단위로 묶는다.

      folds >= 2   : k-fold 교차검증  -> k개 split
      folds == 1   : holdout 비율만큼 한 번만 떼어낸 단일 split
      holdout == 0 : train == test == 전체 (학습 정확도. 일반화 아님)
    """
    k = int(pc.get("folds", 5))
    groups = sorted({str(meta[i][group_by]) for i in idx})

    if k >= 2:
        assign = {g: j % k for j, g in enumerate(groups)}
        folds = [[i for i in idx if assign[str(meta[i][group_by])] == f] for f in range(k)]
        return [([i for j, fo in enumerate(folds) if j != f for i in fo], folds[f])
                for f in range(len(folds))]

    ho = float(pc.get("holdout", 0.2))
    if ho <= 0:
        return [(list(idx), list(idx))]

    n_test = max(1, int(round(len(groups) * ho)))
    step = max(1, len(groups) // n_test)
    test_g = {groups[j] for j in range(0, min(len(groups), n_test * step), step)}
    te = [i for i in idx if str(meta[i][group_by]) in test_g]
    tr = [i for i in idx if str(meta[i][group_by]) not in test_g]
    return [(tr, te)]


# ---------------------------------------------------------------- 학습/평가
def _batch(X, ids, sl, device):
    a = np.asarray(X[ids], dtype=np.float32)
    if sl is not None:
        a = a[:, sl]
    return torch.from_numpy(a).to(device, non_blocking=True)


def train_eval(X, lab, n_cls, splits, pc, device, seed, sl=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    bs = int(pc.get("batch_size", 16))
    epochs = int(pc.get("epochs", 25))
    amp = _ac_dtype(pc.get("amp", "bfloat16"))
    accs = []
    for tr, te in splits:
        if not te or not tr:
            continue
        clf = AttentiveClassifier(
            embed_dim=X.shape[-1],
            num_heads=int(pc.get("num_heads", 16)),
            mlp_ratio=float(pc.get("mlp_ratio", 4.0)),
            depth=int(pc.get("depth", 1)),
            num_classes=n_cls,
        ).to(device)
        opt = torch.optim.AdamW(
            clf.parameters(),
            lr=float(pc.get("lr", 1e-3)),
            weight_decay=float(pc.get("weight_decay", 0.05)),
        )
        steps = epochs * max(1, (len(tr) + bs - 1) // bs)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, float(pc.get("lr", 1e-3)), total_steps=steps)
        y_tr = torch.tensor([lab[i] for i in tr], device=device)

        for _ in range(epochs):
            clf.train()
            perm = np.random.permutation(len(tr))
            for s in range(0, len(tr), bs):
                sel = perm[s: s + bs]
                xb = _batch(X, [tr[j] for j in sel], sl, device)
                with _autocast(amp):
                    loss = nn.functional.cross_entropy(clf(xb), y_tr[sel])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sch.step()

        clf.eval()
        ok = 0
        with torch.no_grad():
            for s in range(0, len(te), bs):
                ii = te[s: s + bs]
                xb = _batch(X, ii, sl, device)
                with _autocast(amp):
                    pred = clf(xb).float().argmax(1)
                ok += int((pred.cpu().numpy() == np.array([lab[i] for i in ii])).sum())
        accs.append(ok / len(te) * 100.0)
        del clf
        torch.cuda.empty_cache()
    return float(np.mean(accs)), accs


# ---------------------------------------------------------------- 요약표
SRC_LABEL = {
    "z": "z  (context encoder — 문맥 구간 표현)",
    "p": "p  (predictor — 예측한 미래)",
    "h": "h  (target encoder — 실제 미래, 상한)",
}


def _summary(results, sources):
    tasks = []
    for key in results:
        t = key.rsplit("/", 1)[0]
        if t not in tasks:
            tasks.append(t)
    w = max([len(t) for t in tasks] + [10]) + 2
    out = ["=" * (w + 62), "ATTENTIVE PROBE 요약", "=" * (w + 62)]

    for src in sources:
        rows = [(t, results[f"{t}/{src}"]) for t in tasks if f"{t}/{src}" in results]
        if not rows:
            continue
        out += ["", f"[{SRC_LABEL.get(src, src)}]"]
        out.append(
            f"{'과제':<{w}}{'n':>5}{'train':>7}{'test':>6}{'클래스':>7}"
            f"{'chance':>9}{'최빈':>8}{'정확도':>10}{'chance 대비':>12}{'편차':>8}"
        )
        out.append("-" * (w + 62))
        for t, r in rows:
            out.append(
                f"{t:<{w}}{r['n']:>5}{r.get('n_train', 0):>7}{r.get('n_test', 0):>6}"
                f"{r['num_classes']:>7}{r['chance']:>8.1f}%"
                f"{r['majority']:>7.1f}%{r['mean']:>9.2f}%"
                f"{r['mean'] - r['chance']:>+11.2f}p{r['std']:>8.2f}"
            )

    # ---- 세 지점 나란히. chance 초과분(=실제로 읽힌 정보량) 기준으로 비교한다 ----
    if len([s for s in ("z", "p", "h") if s in sources]) >= 2:
        out += ["", "[z / p / h — chance 초과분 기준. p/h 가 낮으면 predictor 가 정보를 잃은 것]"]
        out.append(f"{'과제':<{w}}{'z':>9}{'p':>9}{'h':>9}"
                   f"{'z-chance':>10}{'p-chance':>10}{'h-chance':>10}{'p/h':>8}{'z/h':>8}")
        out.append("-" * (w + 73))
        for t in tasks:
            g = {}
            for s in ("z", "p", "h"):
                r = results.get(f"{t}/{s}")
                g[s] = (r["mean"], r["mean"] - r["chance"]) if r else None
            if g["h"] is None or g["p"] is None:
                continue
            gh = g["h"][1]
            ph = g["p"][1] / gh * 100 if gh > 1e-9 else float("nan")
            zh = g["z"][1] / gh * 100 if (g["z"] and gh > 1e-9) else float("nan")
            zm = f"{g['z'][0]:>8.2f}%" if g["z"] else f"{'-':>9}"
            zg = f"{g['z'][1]:>+9.2f}p" if g["z"] else f"{'-':>10}"
            out.append(f"{t:<{w}}{zm}{g['p'][0]:>8.2f}%{g['h'][0]:>8.2f}%"
                       f"{zg}{g['p'][1]:>+9.2f}p{g['h'][1]:>+9.2f}p{ph:>7.1f}%{zh:>7.1f}%")
    out.append("=" * (w + 62))
    return "\n".join(out)


# ---------------------------------------------------------------- main
def _enabled_tasks(cfg):
    return [t for t in cfg["tasks"] if t.get("enabled", True)]


def _sources(cfg):
    s = cfg.get("features", {}).get("sources", ["z", "p", "h"])
    bad = [x for x in s if x not in ALL_SOURCES]
    if bad:
        raise ValueError(f"features.sources 에 모르는 값 {bad}; {list(ALL_SOURCES)} 중에서 골라라")
    return list(s)


def main(cfg, phase="all", rank=0, world=1, extract_world=None):
    out_dir = cfg["output_dir"]
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    cache_dir = cfg.get("features", {}).get("cache_dir") or os.path.join(out_dir, "features")
    sources = _sources(cfg)
    ew = int(extract_world or cfg.get("features", {}).get("extract_world", 1))

    # ---------------- 요약 전용 ----------------
    if phase == "summary":
        results = {}
        for f in sorted(os.listdir(shard_dir)):
            if f.endswith(".json"):
                results.update(json.load(open(os.path.join(shard_dir, f))))
        if not results:
            print("[summary] shard 결과가 없다", flush=True)
            return {}
        order = [f"{t['name']}/{s}" for t in _enabled_tasks(cfg) for s in sources]
        results = {k: results[k] for k in order if k in results}
        summary = _summary(results, sources)
        print("\n" + summary, flush=True)
        json.dump(results, open(os.path.join(out_dir, "attn_probe.json"), "w"),
                  indent=1, ensure_ascii=False)
        open(os.path.join(out_dir, "attn_probe.txt"), "w").write(summary + "\n")
        print(f"\n[saved] {out_dir}/attn_probe.json", flush=True)
        return results

    if phase == "merge":
        merge(cfg, cache_dir, sources, ew)
        return {}

    device = torch.device(cfg.get("device", "cuda:0"))
    torch.cuda.set_device(device)

    # ---------------- 특징 추출 ----------------
    if phase == "extract":
        extract(cfg, cache_dir, device, sources, rank=rank, world=ew)
        return {}
    if phase == "all":
        extract(cfg, cache_dir, device, sources, rank=0, world=1)
        merge(cfg, cache_dir, sources, 1)
        ew = 1

    # ---------------- probe ----------------
    manifest = json.load(open(os.path.join(cache_dir, "manifest.json")))
    meta = json.load(open(os.path.join(cache_dir, "meta.json")))
    pc = cfg.get("probe", {})
    seeds = pc.get("seeds", [0])
    group_by = pc.get("group_by", "block_id")
    X = {s: ShardedTokens(manifest["paths"][s]) for s in sources}
    n_tok = X[sources[0]].shape[1]

    combos = [(t, s) for t in _enabled_tasks(cfg) for s in sources][rank::world]
    if not combos:
        print(f"[rank {rank}] 할당된 조합 없음", flush=True)
        return {}

    results = {}
    bar = _bar(combos, f"probe  r{rank}", rank, unit="job")
    for spec, src in bar:
        name = spec["name"]
        bar.set_postfix_str(f"{name}/{src}")
        idx, lab, names = build_task(meta, spec)
        if not idx:
            print(f"[rank {rank}] skip {name}: 대상 표본 없음", flush=True)
            continue
        sl = token_slice(spec, src, cfg, n_tok)
        cnt = collections.Counter(lab.values())
        chance = 100.0 / len(names)
        major = max(cnt.values()) / len(idx) * 100.0
        splits = make_splits(meta, idx, pc, group_by)
        ntok_used = (sl.stop - sl.start) if sl else n_tok
        print(
            f"[rank {rank}] {name}/{src}  n={len(idx)} 클래스={len(names)} "
            f"chance={chance:.1f}% 최빈={major:.1f}% 토큰={ntok_used} "
            f"(train {len(splits[0][0])} / test {len(splits[0][1])}, split {len(splits)}개)",
            flush=True,
        )
        runs = [train_eval(X[src], lab, len(names), splits, pc, device, sd, sl) for sd in seeds]
        m = float(np.mean([r[0] for r in runs]))
        sdv = float(np.std([r[0] for r in runs])) if len(runs) > 1 else float(np.std(runs[0][1]))
        print(f"[rank {rank}] {name}/{src}  ->  {m:6.2f}%  "
              f"(split별 {[round(a, 1) for a in runs[0][1]]})", flush=True)
        results[f"{name}/{src}"] = {
            "mean": m, "std": sdv,
            "runs": [{"mean": r[0], "folds": r[1]} for r in runs],
            "n": len(idx), "num_classes": len(names), "class_names": names,
            "class_counts": {names[c]: v for c, v in sorted(cnt.items())},
            "chance": chance, "majority": major,
            "n_train": len(splits[0][0]), "n_test": len(splits[0][1]),
            "n_splits": len(splits), "n_tokens": int(ntok_used),
        }
    bar.close()

    json.dump(results, open(os.path.join(shard_dir, f"shard{rank}.json"), "w"),
              indent=1, ensure_ascii=False)

    if world == 1 and phase == "all":
        return main(cfg, phase="summary", rank=0, world=1)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fname", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--phase", default="all",
                    choices=["all", "extract", "merge", "probe", "summary"])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--extract-world", type=int, default=None)
    a = ap.parse_args()
    c = yaml.safe_load(open(a.fname))
    if a.device:
        c["device"] = a.device
    main(c, phase=a.phase, rank=a.rank, world=a.world_size, extract_world=a.extract_world)
