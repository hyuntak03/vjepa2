#!/usr/bin/env python3
"""VOBench_v1 튜블릿 단위 정체 probe — encoder 에서 '가림 없는' 튜블릿으로 학습한 head 를 encoder 가림 · predictor 보임 · predictor 가림에 건다 (2026-10-10).

묻는 것: h (target encoder) 로 학습한 정체 head 가 p (predictor 출력) 에 그대로 통하나. 통하면 "predictor 가 물체 정보를 만들기만 하면
encoder 공간에 정렬된다" 는 기대의 근거가 된다. (p 가 문맥 끝 가림에서 물체 정보를 잃는다는 것은 이미 안다 — ④ 는 재확인.)

  표현 (릴리즈 ViT-H, probing 관례 bfloat16 — attn_probe.py extract 와 같은 함수 · 같은 LN):
    h[t] = LN(target_encoder(32 장))[튜블릿 t] (t = 0..15, 공간 토큰 256)        ← attn_probe 는 미래 8 개만 저장한다. 여기는 16 개 전부
    p[t] = predictor(context_encoder(문맥 16 장), mask_index 0)[튜블릿 t] (t = 8..15)
  샘플 = (클립, 튜블릿 t). 가능 변이 (pos_a · pos_b) 만 (CLAUDE.md §1-4).
  가림 판정 = metadata `samples` (모델 입력 32 샘플마다 '.' 보임 / 'H' 가림): 튜블릿 t = 샘플 2t, 2t+1. 둘 다 '.' = 보임, 둘 다 'H' = 가림, 섞이면 뺀다.
  라벨 = (모양/색) 결합 + none (빈 장면 = permanence pos_b; 빈 장면은 시점과 무관하게 none).
  head = AttentiveClassifier (depth 1, query 1, head 16) · attn_30 (35 epoch, batch 16, lr 1e-4 OneCycle, wd 0.05) — objset 마다 따로.
  학습 = train block 의 h 보임 튜블릿 (물체 + 빈 장면). block 단위 50/50 (seed 0).
  평가 (test block, 고정 head):
    ① h_vis      h, 보임 튜블릿                                     천장
    ② h_occ      h, 가림 튜블릿 (early · mid · late)                 encoder 가 가려진 물체의 정체를 유지하나
    ③ p_vis      p, 미래 튜블릿 · 문맥 끝이 보임 (visible · early · mid) · 정답이 보임    h head 가 p 에 통하나
    ④ p_late     p, 미래 튜블릿 · late (문맥 끝에서 가림), 정답 가림 / 재등장 따로
  지표 = 결합 정확도 · none 이라고 답한 비율 · 다른 물체로 답한 비율 · 모양만 · 색만 맞힌 비율 (+ 미래 튜블릿 t 별).

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/vobench_tubelet_probe.py extract --gpus 4      # → cache/vobench_v1_vith_tubeprobe/{h,p}.npy (fp16, ~23 GB)
  $P z_research/scripts/analysis/vobench_tubelet_probe.py probe                 # → z_research/VOBench_v1/vjepa2/probing/results/<tag>/
  TUBEPROBE_LAYER=16 $P ... extract --gpus 4 · TUBEPROBE_LAYER=16 $P ... probe --tag L16_e35_s0   # h = target encoder 16 번째 block (32 중 절반)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
CFG = REPO / "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/_resolved.yaml"
BASE_CACHE = Path("/local_datasets/world/world_analysis/cache/vobench_v1_vith_tubeprobe")
# TUBEPROBE_LAYER = L (1–32): h 를 target encoder 의 L 번째 block 출력 (out_layers, encoder norm + affine-free LN) 으로 (2026-10-13, 중간층 비교).
#   비우면 마지막 층 (기존). mp.spawn 워커가 모듈을 다시 import 하므로 환경변수로 싣는다. p 는 기존 캐시를 링크한다 (층과 무관).
LAYER = int(os.environ.get("TUBEPROBE_LAYER", "0") or 0)
CACHE = BASE_CACHE if not LAYER else BASE_CACHE.with_name(BASE_CACHE.name + f"_L{LAYER}")
OUT = REPO / "z_research/VOBench_v1/vjepa2/probing/results"
NT, NF, S = 16, 8, 256                     # 튜블릿 (전체 · 미래), 공간 토큰
OPT = dict(epochs=35, batch_size=16, lr=1e-4, weight_decay=0.05)   # configs/protocols/attn_probe.yaml attn_30


def cfg_probe():
    cfg = yaml.safe_load(open(CFG))
    cfg["model"]["dtype"] = "bfloat16"     # probing 관례 (CLAUDE.md §1-7)
    cfg["model"]["autocast"] = "none"
    cfg["features"] = {"decode_workers": 6}
    return cfg


def possible_index(cfg):
    from evals.world_model_analysis.data import WMADataset
    ds = WMADataset(cfg)
    idx = [i for i, r in enumerate(ds.records) if r.plausible == "1"]
    return ds, idx


# ------------------------------------------------------------------ extract
def _worker(rank, world, cfg, idx):
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.attn_probe import _ClipDS, _ac_dtype, _autocast
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    torch.set_num_threads(4)
    N = len(idx)
    lo, hi = N * rank // world, N * (rank + 1) // world
    H = np.load(CACHE / "h.npy", mmap_mode="r+")
    Pm = np.load(CACHE / "p.npy", mmap_mode="r" if LAYER else "r+")
    model = build_from_config(cfg["model"], dev)
    if LAYER:
        vit = [m for m in model.target_encoder.modules() if hasattr(m, "out_layers") and hasattr(m, "blocks")]
        assert len(vit) == 1 and LAYER <= len(vit[0].blocks), (len(vit), LAYER)
        vit[0].out_layers = [LAYER - 1]
    bs = 4
    ci, ti = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=S, batch_size=bs, device=dev)
    loader = torch.utils.data.DataLoader(_ClipDS(cfg, idx[lo:hi]), batch_size=bs, shuffle=False, num_workers=6, pin_memory=True)
    ac = _ac_dtype(cfg["model"].get("autocast", "none"))
    w, t0 = lo, time.time()
    with torch.inference_mode():
        for x in loader:
            b = x.size(0)
            x = x.to(dev, model.dtype, non_blocking=True)
            with _autocast(ac):
                if not LAYER:
                    z = model.context_encoder(x, masks=[ci[:b]])
                    z = z[-1] if isinstance(z, list) else z
                    p = model.predictor(z, ci[:b], ti[:b], mask_index=0)               # (b, 8·S, D)
                h = model.target_encoder(x)
                h = h[-1] if isinstance(h, list) else h                             # (b, 16·S, D)
                h = nn.functional.layer_norm(h, (h.size(-1),))
            H[w:w + b] = h.float().cpu().numpy().reshape(b, NT, S, -1).astype(np.float16)
            if not LAYER:
                Pm[w:w + b] = p.float().cpu().numpy().reshape(b, NF, S, -1).astype(np.float16)
            w += b
            if rank == 0 and (w - lo) % 40 < bs:
                print(f"[r0] {w - lo}/{hi - lo} clip ({time.time() - t0:.0f}s)", flush=True)
    H.flush(); Pm.flush()
    print(f"[r{rank}] done {hi - lo} clip {time.time() - t0:.0f}s", flush=True)


def extract(gpus):
    cfg = cfg_probe()
    ds, idx = possible_index(cfg)
    CACHE.mkdir(parents=True, exist_ok=True)
    D = 1280
    np.lib.format.open_memmap(CACHE / "h.npy", "w+", np.float16, (len(idx), NT, S, D)).flush()
    if LAYER:
        if not (CACHE / "p.npy").exists():
            os.symlink(BASE_CACHE / "p.npy", CACHE / "p.npy")     # p 는 층과 무관 — 기존 캐시
    else:
        np.lib.format.open_memmap(CACHE / "p.npy", "w+", np.float16, (len(idx), NF, S, D)).flush()
    meta = [dict(video_id=ds.records[i].video_id, **{k: ds.records[i].raw[k] for k in
             ("block_id", "variant", "objset", "violation", "occlusion", "k", "motion", "obj_a", "obj_b", "samples")}) for i in idx]
    json.dump(dict(cfg=cfg, n=len(idx), meta=meta), open(CACHE / "meta.json", "w"))
    print(f"extract {len(idx)} possible clips on {gpus} GPU → {CACHE}", flush=True)
    torch.multiprocessing.spawn(_worker, args=(gpus, cfg, idx), nprocs=gpus, join=True)
    json.dump(dict(done=True, time=time.strftime("%F %T")), open(CACHE / "DONE.json", "w"))


# ------------------------------------------------------------------ probe
def tub_state(samples):
    s = samples.replace("|", "")
    assert len(s) == 32, samples
    out = []
    for t in range(NT):
        a, b = s[2 * t], s[2 * t + 1]
        out.append("vis" if a == b == "." else "occ" if a == b == "H" else "mix")
    return out


def label_of(m):
    return m["obj_a"] if m["variant"] == "pos_a" else m["obj_b"]


def run_objset(objset, M, H, Pm, dev, seed=0, train_src="h"):
    rows = [i for i, m in enumerate(M) if m["objset"] == objset]
    labs = sorted({label_of(M[i]) for i in rows})
    assert "empty" in labs
    L = {l: j for j, l in enumerate(labs)}
    NONE = L["empty"]
    blocks = sorted({M[i]["block_id"] for i in rows})
    rng = np.random.default_rng(0); rng.shuffle(blocks)              # 분할은 늘 seed 0 (head 시드와 분리)
    train_b = set(blocks[: len(blocks) // 2])
    st = {i: tub_state(M[i]["samples"]) for i in rows}
    empty = {i: label_of(M[i]) == "empty" for i in rows}

    # 학습: train block · h · 가림 없는 튜블릿 (빈 장면은 시점 무관 none — 그중 '보임' 시점만 써서 물체 클립과 같은 시점 분포)
    if train_src == "h":      # 기본: encoder h 의 가림 없는 튜블릿 (t0–15)
        tr = [(i, t) for i in rows if M[i]["block_id"] in train_b for t in range(NT) if st[i][t] == "vis"]
    else:                      # p 상한: predictor 미래 튜블릿 (t8–15), 문맥 끝이 보이는 조건 (visible · early · mid) 만 — late 는 평가에만
        tr = [(i, t) for i in rows if M[i]["block_id"] in train_b and M[i]["occlusion"] != "late" for t in range(NF, NT) if st[i][t] == "vis"]
    y_tr = np.array([L[label_of(M[i])] for i, _ in tr])
    print(f"[{objset}] train_src {train_src} · classes {len(labs)} · train blocks {len(train_b)} · train tubelets {len(tr)} "
          f"(none {int((y_tr == NONE).sum())}) · 클래스당 최소 {np.bincount(y_tr, minlength=len(labs)).min()}", flush=True)

    # 이 objset 의 h · p 를 RAM 으로 (순차 읽기)
    t0 = time.time()
    ridx = np.array(rows); pos = {i: j for j, i in enumerate(rows)}
    Hs = np.asarray(H[ridx]); Ps = np.asarray(Pm[ridx])
    print(f"[{objset}] loaded h {Hs.shape} p {Ps.shape} in {time.time() - t0:.0f}s", flush=True)

    from src.models.attentive_pooler import AttentiveClassifier
    torch.manual_seed(seed); np.random.seed(seed)
    clf = AttentiveClassifier(embed_dim=Hs.shape[-1], num_heads=16, mlp_ratio=4.0, depth=1, num_classes=len(labs)).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=OPT["lr"], weight_decay=OPT["weight_decay"])
    bs = OPT["batch_size"]; steps = OPT["epochs"] * ((len(tr) + bs - 1) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, OPT["lr"], total_steps=steps)
    ytr = torch.tensor(y_tr, device=dev)
    xb_of = lambda arr, pairs: torch.from_numpy(np.stack([arr[pos[i]][t] for i, t in pairs]).astype(np.float32)).to(dev)
    t0 = time.time()
    for ep in range(OPT["epochs"]):
        clf.train(); perm = np.random.permutation(len(tr)); tot = 0.0
        for s0 in range(0, len(tr), bs):
            sel = perm[s0:s0 + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                xb = xb_of(Hs, [tr[j] for j in sel]) if train_src == "h" else xb_of(Ps, [(tr[j][0], tr[j][1] - NF) for j in sel])
                loss = nn.functional.cross_entropy(clf(xb), ytr[sel])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); tot += float(loss) * len(sel)
        if ep % 5 == 4 or ep == OPT["epochs"] - 1:
            print(f"[{objset}] epoch {ep + 1}/{OPT['epochs']} loss {tot / len(tr):.4f} ({time.time() - t0:.0f}s)", flush=True)
    clf.eval()

    def predict(arr, pairs, off=0):
        out = []
        with torch.no_grad():
            for s0 in range(0, len(pairs), 64):
                pp = pairs[s0:s0 + 64]
                x = torch.from_numpy(np.stack([arr[pos[i]][t - off] for i, t in pp]).astype(np.float32)).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out.append(clf(x).float().argmax(1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, int)

    te = [i for i in rows if M[i]["block_id"] not in train_b]
    cats = {
        "① h_vis · object": ("h", [(i, t) for i in te if not empty[i] for t in range(NT) if st[i][t] == "vis"]),
        "① h_vis · empty": ("h", [(i, t) for i in te if empty[i] for t in range(NT) if st[i][t] == "vis"]),
        "② h_occ · object": ("h", [(i, t) for i in te if not empty[i] for t in range(NT) if st[i][t] == "occ"]),
        "② h_occ · empty": ("h", [(i, t) for i in te if empty[i] for t in range(NT) if st[i][t] == "occ"]),
        "③ p_vis · object": ("p", [(i, t) for i in te if not empty[i] and M[i]["occlusion"] != "late" for t in range(NF, NT) if st[i][t] == "vis"]),
        "③ p_vis · empty": ("p", [(i, t) for i in te if empty[i] and M[i]["occlusion"] != "late" for t in range(NF, NT)]),
        "④ p_late · object, 정답 가림": ("p", [(i, t) for i in te if not empty[i] and M[i]["occlusion"] == "late" for t in range(NF, NT) if st[i][t] == "occ"]),
        "④ p_late · object, 정답 재등장": ("p", [(i, t) for i in te if not empty[i] and M[i]["occlusion"] == "late" for t in range(NF, NT) if st[i][t] == "vis"]),
        "④ p_late · empty": ("p", [(i, t) for i in te if empty[i] and M[i]["occlusion"] == "late" for t in range(NF, NT)]),
        # ⑤ late 클립 시간축 (그림용): h 는 t0–15 의 보임 · 가림 튜블릿 전부, p 는 t8–15 의 보임 · 가림 전부 (섞임 제외)
        "⑤ h_late_all · object": ("h", [(i, t) for i in te if not empty[i] and M[i]["occlusion"] == "late" for t in range(NT) if st[i][t] != "mix"]),
        "⑤ p_late_all · object": ("p", [(i, t) for i in te if not empty[i] and M[i]["occlusion"] == "late" for t in range(NF, NT) if st[i][t] != "mix"]),
    }
    res = {"objset": objset, "train_src": train_src, "classes": labs, "n_train": len(tr), "opt": OPT, "cats": {}}
    for name, (src, pairs) in cats.items():
        if not pairs:
            continue
        pred = predict(Hs if src == "h" else Ps, pairs, off=0 if src == "h" else NF)
        gt = np.array([L[label_of(M[i])] for i, _ in pairs])
        sh = lambda j: labs[j].split("/")[0]; co = lambda j: labs[j].split("/")[-1]
        isobj = gt != NONE
        r = dict(n=len(pairs), acc=100 * float((pred == gt).mean()), none=100 * float((pred == NONE).mean()),
                 other=100 * float(((pred != gt) & (pred != NONE)).mean()))
        if isobj.all():
            r["shape"] = 100 * float(np.mean([sh(a) == sh(b) for a, b in zip(pred, gt)]))
            r["color"] = 100 * float(np.mean([co(a) == co(b) for a, b in zip(pred, gt)]))
        by_t = defaultdict(list)
        for (i, t), a, b in zip(pairs, pred, gt):
            by_t[t].append((a == b, a == NONE))
        r["by_t"] = {int(t): dict(n=len(v), acc=100 * float(np.mean([x for x, _ in v])), none=100 * float(np.mean([y for _, y in v])))
                     for t, v in sorted(by_t.items())}
        by_occ = defaultdict(list)
        for (i, t), a, b in zip(pairs, pred, gt):
            by_occ[f"{M[i]['occlusion']}_k{M[i]['k']}"].append(a == b)
        r["by_occ"] = {k: dict(n=len(v), acc=100 * float(np.mean(v))) for k, v in sorted(by_occ.items())}
        r["samples"] = [[M[i]["video_id"], int(t), st[i][t], M[i]["k"], int(a_), int(b_)] for (i, t), a_, b_ in zip(pairs, pred, gt)]   # [clip, t, 상태, k, pred, gt]
        res["cats"][name] = r
        print(f"[{objset}] {name:30s} n={r['n']:5d} acc {r['acc']:5.1f} · none {r['none']:5.1f} · other {r['other']:5.1f}"
              + (f" · shape {r['shape']:5.1f} · color {r['color']:5.1f}" if "shape" in r else ""), flush=True)
    del Hs, Ps, clf; torch.cuda.empty_cache()
    return res


def probe(gpu, objsets=("primitive", "realistic"), tag="", seed=0, train_src="h"):
    meta = json.load(open(CACHE / "meta.json")); M = meta["meta"]
    assert (CACHE / "DONE.json").exists(), "extract 가 끝나지 않았다"
    H = np.load(CACHE / "h.npy", mmap_mode="r"); Pm = np.load(CACHE / "p.npy", mmap_mode="r")
    dev = torch.device(f"cuda:{gpu}")
    out = OUT / tag if tag else OUT
    out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for o in objsets:
        allres[o] = run_objset(o, M, H, Pm, dev, seed=seed, train_src=train_src)
        json.dump(allres[o], open(out / f"{o}.json", "w"), indent=1, ensure_ascii=False)
    md = [("# VOBench_v1 튜블릿 정체 probe — h (가림 없음) 로 학습한 head 를 h 가림 · p 에 (ViT-H 릴리즈)" if train_src == "h" else
           "# VOBench_v1 튜블릿 정체 probe — **p 로 학습** (문맥 끝 보임 조건의 미래 튜블릿) 한 head 를 p late 에 (ViT-H 릴리즈)"), "",
          f"head = AttentiveClassifier (depth 1, query 1, head 16) · {OPT} · 라벨 = (모양/색) 결합 + none · block 50/50 (seed 0) · objset 따로.",
          "정의 · 재현은 `z_research/scripts/analysis/vobench_tubelet_probe.py` 머리말. 칸 = n · 결합 정확도 · none 이라고 답한 % · 다른 물체로 답한 % · 모양만 · 색만 맞힘.", ""]
    for o, r in allres.items():
        md += [f"## {o} (클래스 {len(r['classes'])}, 학습 튜블릿 {r['n_train']})", "",
               "| 칸 | n | 결합 정확도 | none % | 다른 물체 % | 모양 | 색 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, c in r["cats"].items():
            md.append(f"| {name} | {c['n']} | {c['acc']:.1f} | {c['none']:.1f} | {c['other']:.1f} | "
                      f"{c.get('shape', float('nan')):.1f} | {c.get('color', float('nan')):.1f} |")
        md += ["", "미래 튜블릿 t 별 결합 정확도 (none %):", "",
               "| 칸 | " + " | ".join(f"t{t}" for t in range(NF, NT)) + " |", "|---|" + "---:|" * NF]
        for name, c in r["cats"].items():
            if name.startswith(("③", "④")):
                md.append(f"| {name} | " + " | ".join(f"{c['by_t'][t]['acc']:.0f} ({c['by_t'][t]['none']:.0f})" if t in c["by_t"] else "—"
                                                     for t in range(NF, NT)) + " |")
        md.append("")
    md += ["## 재현", "", "```bash", "P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python",
           "$P z_research/scripts/analysis/vobench_tubelet_probe.py extract --gpus 4",
           "$P z_research/scripts/analysis/vobench_tubelet_probe.py probe", "```"]
    (out / "RESULTS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract", "probe"])
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--objsets", default="primitive,realistic")
    ap.add_argument("--epochs", type=int, default=OPT["epochs"])
    ap.add_argument("--seed", type=int, default=0, help="head 초기화 · 순서 시드 (block 분할은 늘 seed 0)")
    ap.add_argument("--tag", default="", help="결과 하위 폴더 (vjepa2/probing/results/<tag>/)")
    ap.add_argument("--train-src", default="h", choices=["h", "p"], help="h = encoder 보임으로 학습 (이식) · p = predictor 보임 미래로 학습 (p 상한)")
    a = ap.parse_args()
    OPT["epochs"] = a.epochs
    extract(a.gpus) if a.phase == "extract" else probe(a.gpu, tuple(a.objsets.split(",")), a.tag, a.seed, a.train_src)
