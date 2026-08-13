"""Occlusion identity eval — entry point.

    python -m evals.analysis_vlm.occlusion_identity.eval --config <cfg.yaml> [--device cuda:0]

The YAML drives everything. `features.windows` names frame ranges, `features.sources`
pairs each window with an encoder (target / ctx_masked / predictor / isolated), and
`experiments:` is a LIST -- each entry is a self-contained probing experiment with its
own fit source, transfer targets, label convention, probe head and optimization. All
experiments share one feature-extraction pass (the expensive part) and one token cache.

A single experiment:
    fit    one probe head on `fit_on` (+ `fit_labels`, `fit_variants`, train split)
    apply  that head VERBATIM to every source in `apply_to`, scored per variant
           against both label conventions (`before` / `after`)
    (opt)  `independent_probes` additionally fits a fresh head on each apply_to source,
           which upper-bounds what is decodable there -- so "the transfer failed" can be
           told apart from "the information is not there".
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from analysis.intphys2.model import build_from_config                      # noqa: E402
from evals.analysis_vlm.occlusion_identity import forward as fwd           # noqa: E402
from evals.analysis_vlm.occlusion_identity import probe as probelib        # noqa: E402
from evals.analysis_vlm.occlusion_identity.cache import TokenCache         # noqa: E402
from evals.analysis_vlm.occlusion_identity.dataset import OcclusionVideos  # noqa: E402


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, limit, sources) -> OcclusionVideos:
    d = cfg["data"]
    variants = list(dict.fromkeys(
        list(d.get("eval_variants", [])) + list(d.get("extra_variants", []))
        + [v for e in cfg["experiments"] for v in e.get("fit_variants", ["possible"])]))
    return OcclusionVideos(
        d["root"], d.get("index_csv", "index.csv"),
        targets=d["targets"], variants=variants, n_frames=int(d.get("n_frames", 40)),
        split_cfg=d.get("split"),
        occlusion_windows=sorted({tuple(s["window"]) for s in sources}),
        limit_scenes=limit,
        group_column=d.get("group_column", "variant"),
        strata=d.get("strata", []),
    )


# ------------------------------------------------------------------ extraction

def extract_all(ds, cfg, sources, device, *, use_cache, recache, smoke):
    f = cfg["features"]
    n_frames, ctx_len = int(cfg["data"].get("n_frames", 40)), int(f.get("context_length", 32))
    ts = int(cfg["model"].get("tubelet_size", 2))
    S = (int(cfg["model"].get("img_size", 256)) // int(cfg["model"].get("patch_size", 16))) ** 2
    embed_dim = {"vit_large": 1024, "vit_huge": 1280, "vit_giant": 1408}[
        cfg["model"].get("arch_name", "vit_large")]
    counts = fwd.base_token_counts(sources, n_frames, ctx_len, ts, S)

    cache = TokenCache(f["cache_dir"], cfg.get("tag", "feats"),
                       [r.video_id for r in ds.records], counts, embed_dim,
                       dtype=f.get("cache_dtype", "float16"))
    print(f"[feat] bases: " + ", ".join(f"{b}({n} tok)" for b, n in counts.items())
          + f"  -> cache {cache.nbytes()/2**30:.1f} GiB at {cache.dir}")
    if use_cache and not recache and cache.matches():
        print("[feat] cache hit")
        return cache.open_read()

    print(f"[feat] building model on {device} ...")
    bundle = build_from_config(cfg["model"], device)
    assert bundle.embed_dim == embed_dim and bundle.num_spatial_tokens == S, (
        f"cache sizing mismatch: embed {bundle.embed_dim} vs {embed_dim}, "
        f"spatial {bundle.num_spatial_tokens} vs {S}")

    # --no-cache means "do not touch the cache at all": extract into plain in-RAM arrays
    # instead of memmaps, so a smoke run can never leave a partial/other-shaped cache
    # behind for the next real run to trip over.
    if use_cache:
        mm = cache.open_write()
    else:
        mm = {b: np.zeros((len(ds), n, embed_dim), dtype=cache.dtype)
              for b, n in counts.items()}
        print(f"[feat] --no-cache: keeping {cache.nbytes()/2**30:.1f} GiB in RAM, not writing")
    kw = dict(context_length=ctx_len, mask_index=int(f.get("mask_index", 0)),
              out_dtype=getattr(torch, f.get("cache_dtype", "float16")))
    bs = int(f.get("batch_size", 4))
    t0, done = time.time(), 0
    for idx, clips in ds.loader(bs, num_workers=int(f.get("num_workers", 6))):
        out = fwd.extract_batch(clips, bundle, sources, debug=(smoke and done == 0), **kw)
        lo = int(idx[0])
        for b, v in out.items():
            mm[b][lo : lo + len(idx)] = v.numpy()
        done += len(idx)
        if done % max(bs, 20) < bs or done == len(ds):
            el = time.time() - t0
            print(f"  [{done}/{len(ds)}] {el/done:.2f}s/video  eta={(len(ds)-done)*el/done/60:.1f}min")
    del bundle
    torch.cuda.empty_cache()
    if not use_cache:
        return mm                       # plain in-RAM arrays; nothing was written
    cache.finalize(mm)
    del mm
    print(f"[feat] cached -> {cache.dir}")
    return cache.open_read()


# ---------------------------------------------------------------------- probing

def _mask(ds, *, split=None, variants=None) -> np.ndarray:
    m = np.ones(len(ds.records), dtype=bool)
    if split is not None:
        m &= np.array([r.split == split for r in ds.records])
    if variants is not None:
        vs = set(variants)
        m &= np.array([r.variant in vs for r in ds.records])
    return m


def _numkey(v):
    """층화 값 정렬용: 숫자면 숫자로, 아니면 문자열로."""
    try:
        return (0, float(v))
    except (TypeError, ValueError):
        return (1, str(v))


def _labels(ds, name, kind) -> torch.Tensor:
    attr = "labels_after" if kind == "after" else "labels_before"
    return torch.tensor([getattr(r, attr)[name] for r in ds.records], dtype=torch.long)


def _view(bases, src, ts, S):
    """Token view (N, T, D) for one source, as a slice of its base memmap."""
    a, b = fwd.token_slice(src, ts, S)
    return bases[src["base"]][:, a:b]


def run_experiment(exp, ds, bases, cfg, src_of, ts, S, device):
    d = cfg["data"]
    fit_on = exp["fit_on"]
    kind = exp.get("fit_labels", "before")
    fit_variants = exp.get("fit_variants", ["possible"])
    apply_to = exp.get("apply_to", [fit_on])
    tnames = exp.get("targets", [t["name"] for t in d["targets"]])
    spec = exp.get("probe", {"type": "linear", "pooling": "mean", "pre_norm": True})
    o = exp.get("optimization", {})
    okw = dict(num_epochs=int(o.get("num_epochs", 200)),
               batch_size=int(o.get("batch_size", 16)), lr=float(o.get("lr", 1e-3)),
               weight_decay=float(o.get("weight_decay", 0.01)),
               seed=int(o.get("seed", 0)), device=device)
    ev_variants = exp.get("eval_variants", d["eval_variants"])
    extra_variants = exp.get("extra_variants", d.get("extra_variants", []))

    train_m = _mask(ds, split="train", variants=fit_variants)
    val_m = _mask(ds, split="val")
    tr_idx = np.nonzero(train_m)[0]
    print(f"\n[{exp['name']}] probe={probelib.probe_name(spec)} fit_on={fit_on} "
          f"labels={kind} variants={fit_variants} | train n={len(tr_idx)} "
          f"| epochs={okw['num_epochs']} lr={okw['lr']}")

    results, preds = {}, {}
    for tname in tnames:
        t = next(x for x in d["targets"] if x["name"] == tname)
        K = len(t["classes"])
        Xfit = _view(bases, src_of[fit_on], ts, S)
        pr = probelib.train_probe(Xfit[tr_idx], _labels(ds, tname, kind)[tr_idx],
                                  spec, K, **okw)
        print(f"  [{tname}] fitted: n={pr.n_train} train_acc={pr.train_acc:.4f} "
              f"loss={pr.train_loss:.4f}"
              + ("   !! UNDERFIT (raise num_epochs / lr)" if pr.train_acc < 0.99 else ""))

        # 1/K is the wrong reference when the label distribution is skewed or when some
        # configured classes never occur (e.g. the 48px set has 7 shapes, not 8, and
        # `ring` is only 3.6% of scenes). Report what always-guess-the-most-common gets.
        y_all = _labels(ds, tname, kind)
        m_base = val_m & _mask(ds, variants=fit_variants)
        yv = y_all[m_base]
        present = int(len(torch.unique(y_all)))
        majority = (float(torch.bincount(yv, minlength=K).max()) / len(yv)) if len(yv) else float("nan")

        entry = {"probe": probelib.probe_name(spec), "spec": spec, "num_classes": K,
                 "classes_present": present, "chance": 1.0 / K,
                 "majority_baseline": majority,
                 "fit_on": fit_on, "fit_labels": kind,
                 "fit_variants": fit_variants,
                 "train": {"n": pr.n_train, "acc": pr.train_acc, "loss": pr.train_loss},
                 "sources": {}}
        for sname in apply_to:
            src = src_of[sname]
            X = _view(bases, src, ts, S)
            pred = probelib.predict(pr, X)
            preds[(tname, sname)] = pred
            f0, f1 = src["window"]
            hf = np.array([r.hidden_frac.get(f"{f0}-{f1}", float("nan")) for r in ds.records])
            row = {"window": [f0, f1], "encoder": src["encoder"], "n_tokens": X.shape[1],
                   # 가림이 없는 데이터셋(probe set)은 hidden_frac 자체가 없다 -> nan
                   "mean_hidden_frac": (float(np.nanmean(hf[val_m]))
                                        if np.isfinite(hf[val_m]).any() else float("nan")),
                   "per_variant": {}}
            for v in list(ev_variants) + list(extra_variants):
                m = val_m & _mask(ds, variants=[v])
                yb_m, ya_m = _labels(ds, tname, "before")[m], _labels(ds, tname, "after")[m]
                cell = {"n": int(m.sum()),
                        "acc_before": probelib.accuracy(pred[m], yb_m),
                        "bacc_before": probelib.balanced_accuracy(pred[m], yb_m),
                        "acc_after": probelib.accuracy(pred[m], ya_m)}
                if np.isfinite(hf).any() and np.nanmax(hf) > 0:
                    fh = m & (hf >= 1.0)
                    cell["n_fully_hidden"] = int(fh.sum())
                    cell["acc_before_fully_hidden"] = probelib.accuracy(
                        pred[fh], _labels(ds, tname, "before")[fh])
                if v in extra_variants:
                    cell["note"] = "identity undefined after the reveal (object annihilated)"
                row["per_variant"][v] = cell
            # 추가 층화: data.strata 로 지정한 컬럼 값별 정확도 (예: obj_px, is_static).
            # 하나의 head 를 전체 train 으로 학습하고 val 을 쪼개서 보는 것이라,
            # 층별로 head 를 따로 학습하는 것과 달리 표본을 나눠 쓰지 않는다.
            strata_cols = cfg["data"].get("strata", [])
            if strata_cols:
                row["per_stratum"] = {}
                for col in strata_cols:
                    vals = sorted({r.strata.get(col, "?") for r in ds.records}, key=_numkey)
                    row["per_stratum"][col] = {}
                    for sv in vals:
                        sm = val_m & np.array([r.strata.get(col, "?") == sv for r in ds.records])
                        ykb = _labels(ds, tname, kind)[sm]
                        row["per_stratum"][col][sv] = {
                            "n": int(sm.sum()),
                            "acc": probelib.accuracy(pred[sm], ykb),
                            "bacc": probelib.balanced_accuracy(pred[sm], ykb)}
            if exp.get("independent_probes", False):
                ipr = probelib.train_probe(X[tr_idx], _labels(ds, tname, kind)[tr_idx],
                                           spec, K, **okw)
                m = val_m & _mask(ds, variants=fit_variants)
                ip = probelib.predict(ipr, X)[m]
                yk = _labels(ds, tname, kind)[m]
                row["independent"] = {
                    "train_acc": ipr.train_acc, "n": int(m.sum()),
                    "val_acc": probelib.accuracy(ip, yk),
                    "val_bacc": probelib.balanced_accuracy(ip, yk)}
            entry["sources"][sname] = row
        results[tname] = entry
    return {"name": exp["name"], "meta": exp, "results": results, "preds": preds,
            "train_mask": train_m, "val_mask": val_m}


# ---------------------------------------------------------------------- reports

def _wrap(text: str, indent: str, width: int = 96) -> list:
    import textwrap
    return textwrap.wrap(str(text), width=width,
                         initial_indent=indent, subsequent_indent=" " * len(indent)) or []


def _render(exp_out, cfg) -> list:
    """One experiment's block: its stated purpose first, then the numbers.

    The purpose fields (title/question/expect/read_as) live in the YAML and are echoed
    here on purpose -- months later the summary.txt has to say what the experiment was
    asking, not just what it measured.
    """
    m = exp_out.get("meta", {})
    L = ["", "=" * 108, f"### {exp_out['name']}" + (f" — {m['title']}" if m.get("title") else ""),
         "=" * 108]
    for key, label in (("question", "무엇을 보려는가"), ("expect", "예상"),
                       ("read_as", "결과 읽는 법")):
        if m.get(key):
            L += _wrap(m[key], f"  [{label}] ")
    L.append("")
    for tname, e in exp_out["results"].items():
        L += ["-" * 108,
              f"[{tname}]  {e['classes_present']}-way (설정 {e['num_classes']}), "
              f"chance {e['chance']:.3f}, 최빈 baseline {e['majority_baseline']:.3f}"
              f"   probe={e['probe']}   head 학습: {e['fit_on']} "
              f"(labels={e['fit_labels']}, variants={e['fit_variants']}, "
              f"n={e['train']['n']}, train_acc={e['train']['acc']:.4f})",
              "-" * 108,
              f"  {'source':22s} {'enc':11s} {'frames':>7s} {'tok':>5s} {'hid':>5s} "
              f"{'variant':11s} {'n':>4s} {'acc_bef':>8s} {'bacc_bef':>8s} {'acc_aft':>8s} "
              f"{'hidOnly':>8s} {'indep':>7s} {'i_bacc':>7s} {'i_trn':>6s}"]
        for sname, row in e["sources"].items():
            ind = row.get("independent", {})
            f0, f1 = row["window"]
            for i, (v, c) in enumerate(row["per_variant"].items()):
                # source / window / token-count / hidden-frac / independent-probe are
                # per-source, so print them only on that source's first variant row.
                if i == 0:
                    c_src, c_enc = sname, row["encoder"]
                    c_win, c_tok = f"{f0}-{f1}", str(row["n_tokens"])
                    c_hid = ("-" if not np.isfinite(row["mean_hidden_frac"])
                             else f"{row['mean_hidden_frac']:.2f}")
                    c_ind = f"{ind['val_acc']:.4f}" if ind else ""
                    c_ib = f"{ind['val_bacc']:.4f}" if ind else ""
                    c_it = f"{ind['train_acc']:.3f}" if ind else ""
                else:
                    c_src = c_enc = c_win = c_tok = c_hid = c_ind = c_ib = c_it = ""
                c_ho = (f"{c['acc_before_fully_hidden']:.4f}"
                        if c.get("n_fully_hidden") else "-")
                L.append(
                    f"  {c_src:22s} {c_enc:11s} {c_win:>7s} {c_tok:>5s} {c_hid:>5s} "
                    f"{v:11s} {c['n']:4d} {c['acc_before']:8.4f} {c['bacc_before']:8.4f} "
                    f"{c['acc_after']:8.4f} {c_ho:>8s} {c_ind:>7s} {c_ib:>7s} {c_it:>6s}")
            L.append("")
        # 층화 리포트 (있을 때만)
        first_src = next(iter(e["sources"].values()), {})
        if "per_stratum" in first_src:
            L.append(f"  ── 층화 (한 head 를 전체 train 으로 학습, val 만 쪼갬) ──")
            for col in first_src["per_stratum"]:
                vals = list(first_src["per_stratum"][col])
                L.append(f"    [{col}]" + "".join(f"{v:>14s}" for v in vals))
                for sname, row in e["sources"].items():
                    ps = row.get("per_stratum", {}).get(col, {})
                    cells = "".join(
                        f"{ps[v]['acc']:>8.4f}({ps[v]['n']:>3d})" if v in ps else f"{'-':>14s}"
                        for v in vals)
                    L.append(f"    {sname:20s}" + cells)
                L.append("")
    return L


def write_outputs(ds, cfg, outs, sources, src_of, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    d = cfg["data"]
    cls = {t["name"]: t["classes"] for t in d["targets"]}
    windows = sorted({tuple(s["window"]) for s in sources})

    cols = ["video_id", "scene_id", "variant", "split"]
    cols += [f"hidden_{w[0]}-{w[1]}" for w in windows]
    for n in cls:
        cols += [f"{n}_before", f"{n}_after"]
    pred_cols = []
    for o in outs:
        for (tname, sname) in o["preds"]:
            pred_cols.append((o["name"], tname, sname))
    cols += [f"{e}|{t}|{s}" for e, t, s in pred_cols]
    with open(os.path.join(out_dir, "per_video.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(ds.records):
            row = {"video_id": r.video_id, "scene_id": r.scene_id,
                   "variant": r.variant, "split": r.split}
            for win in windows:
                k = f"{win[0]}-{win[1]}"
                row[f"hidden_{k}"] = round(r.hidden_frac.get(k, float("nan")), 4)
            for n in cls:
                row[f"{n}_before"] = cls[n][r.labels_before[n]]
                row[f"{n}_after"] = cls[n][r.labels_after[n]]
            for o in outs:
                for (tname, sname), pred in o["preds"].items():
                    row[f"{o['name']}|{tname}|{sname}"] = cls[tname][int(pred[i])]
            w.writerow(row)

    # confusion matrices: <experiment>__<target>__<source>__<variant>.npy, rows=truth
    # (before-label), cols=prediction, in data.targets[].classes order. Keyed by
    # experiment so two experiments over the same source never overwrite each other.
    if cfg.get("report", {}).get("confusion", True):
        cdir = os.path.join(out_dir, "confusion")
        os.makedirs(cdir, exist_ok=True)
        for o in outs:
            for (tname, sname), pred in o["preds"].items():
                K = len(cls[tname])
                yb = _labels(ds, tname, "before")
                for v in d["eval_variants"]:
                    m = o["val_mask"] & _mask(ds, variants=[v])
                    np.save(os.path.join(cdir, f"{o['name']}__{tname}__{sname}__{v}.npy"),
                            probelib.confusion(pred[m], yb[m], K))

    json.dump({"tag": cfg.get("tag"), "n_videos": len(ds.records),
               "n_scenes": len({r.scene_id for r in ds.records}),
               "experiments": {o["name"]: o["results"] for o in outs}, "config": cfg},
              open(os.path.join(out_dir, "summary.json"), "w"), indent=2)

    L = ["=" * 108, "OCCLUSION IDENTITY — V-JEPA 2 (encoder/predictor frozen)", "=" * 108,
         f"videos={len(ds.records)}  scenes={len({r.scene_id for r in ds.records})}",
         "",
         "■ source 이름 규칙:  <인코더><넣은 프레임 수>__<읽은 토큰 구간>",
         "    tgt40__visible  = target encoder(EMA) 에 40프레임을 넣고, 그 중 1~8 프레임 토큰만 읽음",
         "    ctx8__occluded  = context encoder(online) 에 23~30 프레임'만' 넣고 그 토큰을 읽음",
         "",
         "  인코더 3종:",
         "    tgt  target encoder  = EMA teacher. predictor 가 맞추도록 학습된 쪽",
         "    ctx  context encoder = online. predictor 에 실제로 입력되는 쪽",
         "    pred predictor       = context 만 보고 미래 8프레임을 상상한 출력",
         "  넣은 프레임 수가 중요한 이유: 인코더는 bidirectional 이라 40프레임을 넣으면 어느 구간",
         "  토큰이든 나머지 전 구간을 attend 한다. 8프레임'만' 넣은 버전(tgt8/ctx8)은 그 경로를",
         "  끊은 대조군이라, 두 값을 비교하면 '그 구간 픽셀에서 온 정보'와 '문맥에서 흘러온 정보'가",
         "  분리된다.",
         "",
         "■ 컬럼:",
         "    acc_before  가림 이전 물체 기준 정확도  <- 가림 구간과 predictor 예측의 정답",
         "    acc_after   리빌 후 실제 렌더된 물체 기준  <- imp_shape/imp_color 에서만 달라짐",
         "    hid         그 구간에서 공이 안 보이는 프레임 비율(평균)",
         "    hidOnly     8프레임 내내 완전히 가려진 scene 만 골라낸 acc_before",
         "    bacc_bef    클래스별 recall 의 macro 평균. 라벨 분포가 다른 데이터셋끼리 비교할 때 이걸 볼 것",
         "                (24px shape 은 8종/최빈 0.171, 48px shape 은 7종/최빈 0.221 이라 raw acc 는 비교 불가)",
         "    indep       그 source 자체로 head 를 새로 학습했을 때의 정확도 = 디코딩 상한.",
         "    i_bacc      indep 의 macro 평균 recall / i_trn = indep 의 train_acc (과적합 판정용)",
         "                이식값이 낮은데 indep 이 높으면 '정보는 있는데 인코딩 방식이 다름',",
         "                indep 도 낮으면 '정보가 애초에 없음'.",
         "    variant     possible / imp_shape(모양 바뀜) / imp_color(색 바뀜) / imp_vanish(사라짐)",
         ""]
    for o in outs:
        L += _render(o, cfg)
    txt = "\n".join(L)
    open(os.path.join(out_dir, "summary.txt"), "w").write(txt + "\n")
    print("\n" + txt + f"\n-> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only use the first N scenes")
    ap.add_argument("--only", default=None, help="comma-separated experiment names to run")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-cache", dest="no_cache", action="store_true")
    ap.add_argument("--recache", action="store_true")
    a = ap.parse_args()

    cfg = _load_cfg(a.config)
    device = torch.device(a.device)
    ts = int(cfg["model"].get("tubelet_size", 2))
    S = (int(cfg["model"].get("img_size", 256)) // int(cfg["model"].get("patch_size", 16))) ** 2
    sources = fwd.resolve_sources(cfg["features"], int(cfg["data"].get("n_frames", 40)),
                                  int(cfg["features"].get("context_length", 32)), ts)
    src_of = {s["name"]: s for s in sources}

    exps = cfg["experiments"]
    if a.only:
        want = {x.strip() for x in a.only.split(",")}
        exps = [e for e in exps if e["name"] in want]
        assert exps, f"--only {a.only!r} matched no experiment"
    for e in exps:                      # fail fast on typos before the expensive pass
        for sname in [e["fit_on"]] + list(e.get("apply_to", [])):
            assert sname in src_of, (
                f"experiment {e['name']!r}: unknown source {sname!r}; "
                f"defined: {sorted(src_of)}")

    ds = build_dataset(cfg, a.limit, sources)
    print(f"[data] {ds.summary()}")
    print("[data] sources: " + ", ".join(
        f"{s['name']}({s['encoder']} {s['window'][0]}-{s['window'][1]})" for s in sources))
    print(f"[data] experiments: {[e['name'] for e in exps]}")

    bases = extract_all(ds, cfg, sources, device, use_cache=not a.no_cache,
                        recache=a.recache, smoke=a.smoke)
    pdev = "cuda" if torch.cuda.is_available() and device.type == "cuda" else "cpu"
    outs = [run_experiment(e, ds, bases, cfg, src_of, ts, S, pdev) for e in exps]
    write_outputs(ds, cfg, outs, sources, src_of,
                  a.out_dir or cfg.get("output_dir", "./occlusion_identity_out"))


if __name__ == "__main__":
    main()
