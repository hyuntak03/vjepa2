#!/usr/bin/env python3
"""VOBench 에서 릴리즈 V-JEPA 2 predictor 의 미래 예측 p 에 물체 모양 · 색이 읽히나, 그리고 그게 encoder 공간 h 와 맞춰져 있나 (2026-10-29, 사용자 요청).

사용자: "VOBench 기준으로 attentive probing 해서 last k = 1 기준으로 (T, D) shape 으로 obj shape, color 각각 읽도록 해봐".

표현 (surprise_c16t32 / attn_probe 와 같은 정의 — evals/analysis_vlm/occlusion_identity/forward.py:extract_batch):
  32 장 (raw 0,3,…,93), 문맥 16 장 = 8 블록, 미래 16 장 = 8 블록. ViT-H 릴리즈 체크포인트, dual encoder, dtype bfloat16 (probing 관례).
  z = LN(context encoder(문맥 토큰만))  (2048, 1280)
  p = predictor(z) 미래 토큰            (2048, 1280)
  h = LN(target encoder(32 장))[미래]   (2048, 1280)
probe 입력 (T, D): ① 토큰 T = 2048 (8 블록 × 256 칸) 그대로 ② 블록별 공간 평균 T = 8.
데이터: VOBench primitive 의 가능 변이 (pos_a = obj_a, pos_b = obj_b) 중 물체가 있는 것 (permanence pos_b 는 빈 장면이라 뺀다) — 600 clip.
  라벨 = obj 의 모양 (6) · 색 (8). 불가능 변이는 쓰지 않는다 (CLAUDE.md §1-4).
분할: block 단위 5-fold (조건 비율 유지). 모든 clip 이 한 번씩 시험된다. 학습 집합 두 가지: 전 조건 / late 만 (k 1·2·3).
판독: z→z · h→h · p→p (정보가 있나) · h→p (encoder 에서 배운 판독기가 예측에 통하나 = 정렬) · p→h.
판독기: AttentiveClassifier (질의 1, 16 head, depth 1) — evals/world_model_analysis/attn_probe.py 의 train_eval 과 같은 학습 (AdamW 1e-3 one-cycle · wd 0.05 · 묶음 16 · 25 epoch · bf16).

  # 1) 추출 (GPU 1 장, ~5 분)
  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_research/scripts/analysis/vobench_attnprobe_release.py extract
  # 2) probe (조합마다 GPU 1 장)
  for g in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$g /venv/vjepa/bin/python z_research/scripts/analysis/vobench_attnprobe_release.py probe --job $g & done
  # 3) 표
  /venv/vjepa/bin/python z_research/scripts/analysis/vobench_attnprobe_release.py report
출력: z_analysis/results/vobench_attnprobe_release/  (캐시는 /workspace/auto_research_cache/vobench_attnprobe_release/)
"""
import argparse, collections, csv, json, os, sys
import numpy as np, torch, torch.nn as nn, yaml
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
ap = argparse.ArgumentParser()
ap.add_argument("phase", choices=["extract", "extract_iso", "probe", "report", "probe_t", "report_t", "probe_vis", "report_vis", "probe_k", "report_k", "report_vis23", "extract_empty", "probe_fig", "extract_permimp", "l1_static", "extract_permimpba", "l1_dir", "l1_dir_decomp", "l1_copy", "extract_real"])
ap.add_argument("--job", type=int, default=0)
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--match", default="count", choices=["count", "pos"], help="probe_fig: count = 보임 학습 표본을 수만 맞춤 / pos = 가린 블록과 같은 위치의 보이는 블록으로 하나씩 짝 (시험도 그 위치만), 2026-10-29")
ap.add_argument("--subset", default="all", choices=["all", "k23"], help="probe_vis: k23 = k 2·3 영상만 (2026-10-29 사용자 선택, 전 타이밍)")
a = ap.parse_args()
CACHE = "/workspace/auto_research_cache/vobench_attnprobe_release"
OUT = os.path.join(REPO, "z_analysis/results/vobench_attnprobe_release")
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)
SRCS = ("z", "p", "h")
NT, D, NB = 2048, 1280, 8
SHAPES = ["capsule", "cone", "cube", "cylinder", "sphere", "torus"]
COLORS = ["blue", "cyan", "green", "magenta", "orange", "purple", "red", "yellow"]


def meta():
    rows = []
    for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))):
        if r["objset"] != "primitive" or r["plausible"] != "1":
            continue
        obj = r["obj_a"] if r["variant"] == "pos_a" else r["obj_b"]
        if obj == "empty":
            continue
        sh, co = obj.split("/")
        rows.append(dict(video_id=r["video_id"], file_name=r["file_name"], block_id=r["block_id"], variant=r["variant"],
                         occlusion=r["occlusion"], k=int(r["k"]), violation=r["violation"], motion=r["motion"],
                         shape=SHAPES.index(sh), color=COLORS.index(co)))
    return rows


def meta_empty():
    """permanence block 의 pos_b = 같은 장면 · 같은 가림판, 물체 없음 (obj 존재 판독의 0 라벨, 2026-10-29)."""
    rows = []
    for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))):
        if r["objset"] == "primitive" and r["plausible"] == "1" and r["variant"] == "pos_b" and r["obj_b"] == "empty":
            rows.append(dict(video_id=r["video_id"], file_name=r["file_name"], block_id=r["block_id"], variant=r["variant"],
                             occlusion=r["occlusion"], k=int(r["k"]), violation=r["violation"], motion=r["motion"], shape=-1, color=-1))
    return rows


def meta_real():
    """realistic 물체 영상의 가능 변이 (물체 있는 것만) — 2026-10-29 속도 · 가속도 판독 표본을 늘리려고. 모양 · 색 라벨은 없다 (-1)."""
    rows = []
    for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))):
        if r["objset"] != "realistic" or r["plausible"] != "1":
            continue
        obj = r["obj_a"] if r["variant"] == "pos_a" else r["obj_b"]
        if obj == "empty":
            continue
        rows.append(dict(video_id=r["video_id"], file_name=r["file_name"], block_id=r["block_id"], variant=r["variant"],
                         occlusion=r["occlusion"], k=int(r["k"]), violation=r["violation"], motion=r["motion"], obj=obj, shape=-1, color=-1))
    return rows


def meta_permimp(variant="imp_ab"):
    """permanence block 의 imp_ab (variant="imp_ba" 면 빈 장면에 물체가 가림 뒤에서 나타나는 영상, 문맥은 pos_b 와 같다) = 물체가 있다가 가림 뒤에서 사라지는 영상 (채점의 불가능 쪽, 2026-10-29). 문맥은 pos_a 와 픽셀 단위로 같다."""
    rows = []
    for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))):
        if r["objset"] == "primitive" and r["violation"] == "permanence" and r["variant"] == variant:
            rows.append(dict(video_id=r["video_id"], file_name=r["file_name"], block_id=r["block_id"], variant=r["variant"],
                             occlusion=r["occlusion"], k=int(r["k"]), violation=r["violation"], motion=r["motion"], shape=-1, color=-1))
    return rows


def cond(m):
    return "visible" if m["occlusion"] == "visible" else f"{m['occlusion']} k{m['k']}"


def folds(M):
    """block 단위 5-fold, 조건 (+위반 종류) 비율 유지. return block_id → fold."""
    rng = np.random.default_rng(0)
    byc = collections.defaultdict(list)
    for b in sorted({m["block_id"] for m in M}, key=int):
        mb = next(m for m in M if m["block_id"] == b)
        byc[(cond(mb), mb["violation"])].append(b)
    fold_of = {}
    for key, bl in byc.items():
        for j, b in enumerate(rng.permutation(bl)):
            fold_of[b] = j % 5
    return fold_of


def fit(Xf, y, tr, ncls, epochs, seed, dev, bs=16):
    """AttentiveClassifier 학습 (attn_probe.py train_eval 과 같은 설정). return (판독기, 학습 정확도)."""
    from src.models.attentive_pooler import AttentiveClassifier
    torch.manual_seed(seed); np.random.seed(seed)
    clf = AttentiveClassifier(embed_dim=D, num_heads=16, mlp_ratio=4.0, depth=1, num_classes=ncls).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * ((len(tr) + bs - 1) // bs))
    trt = torch.tensor(tr, device=dev)
    for ep in range(epochs):
        clf.train()
        perm = trt[torch.randperm(len(tr), device=dev)]
        for s0 in range(0, len(tr), bs):
            ib = perm[s0:s0 + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = nn.functional.cross_entropy(clf(Xf[ib].float()), y[ib])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    clf.eval()
    pr = np.concatenate([predict(clf, Xf, tr[s0:s0 + 64], dev) for s0 in range(0, len(tr), 64)])
    return clf, 100 * float((pr == y[torch.tensor(tr, device=dev)].cpu().numpy()).mean())


def predict(clf, Xe, ii, dev):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return clf(Xe[torch.tensor(ii, device=dev)].float()).argmax(1).cpu().numpy()


if a.phase in ("extract", "extract_iso", "extract_empty", "extract_permimp", "extract_permimpba", "extract_real"):
    # extract_iso (2026-10-29): 문맥 블록마다 문맥 encoder 를 그 블록 2 장에만 따로 건다 → zi (600, 2048, D). 시간 attention 이 없는 대조군
    from analysis.intphys2.model import build_from_config
    from evals.analysis_vlm.occlusion_identity import forward as fwd
    from evals.world_model_analysis.data import WMADataset
    M = {"extract_empty": meta_empty, "extract_permimp": meta_permimp, "extract_permimpba": lambda: meta_permimp("imp_ba"), "extract_real": meta_real}.get(a.phase, meta)()
    sfx = {"extract_empty": "_empty", "extract_permimp": "_permimp", "extract_permimpba": "_permimpba", "extract_real": "_real"}.get(a.phase, "")          # extract_empty: 빈 장면 120 개 → z_empty / p_empty / h_empty
    idx_dir = os.path.join(CACHE, "index" + sfx); os.makedirs(idx_dir, exist_ok=True)
    src = {r["video_id"]: r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv")))}
    with open(os.path.join(idx_dir, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(src.values())).keys())); w.writeheader()
        for m in M:
            w.writerow(src[m["video_id"]])
    ck = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    cfg = {"data": dict(root=idx_dir, index_csv="index.csv", frames_root="/workspace/dataset/benchmarks/VOBench_v1",
                        frames_pattern="{file_name}/{frame:06d}.png", frames_start=0, frames_stride=3, block_column="block_id",
                        pair_column="pair_id", variant_column="variant", plausible_column="plausible", type_column="condition",
                        n_frames=32, resolution=256),
           "model": dict(arch_name="vit_huge", checkpoint=ck, img_size=256, patch_size=16, tubelet_size=2, window_size=32,
                         use_rope=True, uniform_power=False, dual_encoder=True, context_encoder_key="encoder",
                         target_encoder_key="target_encoder", predictor=dict(embed_dim=384, depth=12, num_heads=12, num_mask_tokens=10),
                         dtype="bfloat16")}
    ds = WMADataset(cfg)
    assert [r.video_id for r in ds.records] == [m["video_id"] for m in M], "index 순서가 다르다"
    dev = torch.device("cuda")
    bundle = build_from_config(cfg["model"], dev)
    iso = a.phase == "extract_iso"
    # ⚠️ 2026-10-29 정정: 예전엔 SRCS 파일을 먼저 w+ 로 연 뒤 _empty 로 바꿔 열어 extract_empty 가 z/p/h.npy 를 120 행으로 덮어썼다. 파일 이름에 sfx 를 붙여 한 번만 연다
    mm = {s: np.lib.format.open_memmap(os.path.join(CACHE, f"{s}{sfx}.npy"), mode="w+", dtype=np.float16, shape=(len(M), NT, D))
          for s in (("zi",) if iso else SRCS)}
    srcs = [{"base": f"isolated_ctx:{2 * b}_{2 * b + 2}"} for b in range(NB)] if iso else [{"base": "target"}, {"base": "ctx_masked"}, {"base": "predictor"}]
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(8); bs = 4
    for s0 in range(0, len(M), bs):
        ii = list(range(s0, min(len(M), s0 + bs)))
        clips = torch.stack(list(ex.map(ds.clip, ii)))
        with torch.no_grad():
            o = fwd.extract_batch(clips, bundle, srcs, context_length=16, mask_index=0, out_dtype=torch.float16)
        if iso:
            for j, i in enumerate(ii):
                mm["zi"][i] = torch.cat([o[f"isolated_ctx:{2 * b}_{2 * b + 2}"][j] for b in range(NB)]).numpy()
            continue
        assert o["target"].shape[1] == 2 * NT and o["ctx_masked"].shape[1] == NT and o["predictor"].shape[1] == NT
        for j, i in enumerate(ii):
            mm["h"][i] = o["target"][j, NT:].numpy(); mm["z"][i] = o["ctx_masked"][j].numpy(); mm["p"][i] = o["predictor"][j].numpy()
        if s0 % 100 == 0:
            print(f"{s0}/{len(M)}", flush=True)
    for v in mm.values():
        v.flush()
    json.dump(M, open(os.path.join(CACHE, f"meta{sfx}.json"), "w"))
    print("끝", len(M))

elif a.phase == "probe":
    from src.models.attentive_pooler import AttentiveClassifier
    M = json.load(open(os.path.join(CACHE, "meta.json")))
    JOBS = [(inp, tr, tg) for inp in ("tokens", "pooled") for tr in ("all", "late") for tg in ("shape", "color")]
    inp, trset, tgt = JOBS[a.job]
    dev = torch.device("cuda")
    X = {}
    for s in SRCS:
        x = torch.from_numpy(np.load(os.path.join(CACHE, f"{s}.npy"), mmap_mode="r")[:]).to(dev)
        X[s] = x.view(len(M), NB, 256, D).mean(2) if inp == "pooled" else x            # (N, T, D) fp16
    y = torch.tensor([m[tgt] for m in M], device=dev)
    ncls = len(SHAPES if tgt == "shape" else COLORS)
    # block 단위 5-fold, 조건 (+위반 종류) 비율 유지
    rng = np.random.default_rng(0)
    byc = collections.defaultdict(list)
    for b in sorted({m["block_id"] for m in M}, key=int):
        mb = next(m for m in M if m["block_id"] == b)
        byc[(cond(mb), mb["violation"])].append(b)
    fold_of = {}
    for key, bl in byc.items():
        bl = list(rng.permutation(bl))
        for j, b in enumerate(bl):
            fold_of[b] = j % 5
    PAIRS = [("z", "z"), ("h", "h"), ("p", "p"), ("h", "p"), ("p", "h")]
    pred = {f"{fs}->{es}": np.full(len(M), -1) for fs, es in PAIRS}
    bs = 16
    for fold in range(5):
        te = [i for i, m in enumerate(M) if fold_of[m["block_id"]] == fold]
        tr = [i for i, m in enumerate(M) if fold_of[m["block_id"]] != fold and (trset == "all" or m["occlusion"] == "late")]
        for fs in ("z", "h", "p"):
            torch.manual_seed(fold); np.random.seed(fold)
            clf = AttentiveClassifier(embed_dim=D, num_heads=16, mlp_ratio=4.0, depth=1, num_classes=ncls).to(dev)
            opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=0.05)
            steps = a.epochs * ((len(tr) + bs - 1) // bs)
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
            trt = torch.tensor(tr, device=dev)
            for ep in range(a.epochs):
                clf.train()
                perm = trt[torch.randperm(len(tr), device=dev)]
                for s0 in range(0, len(tr), bs):
                    ib = perm[s0:s0 + bs]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = nn.functional.cross_entropy(clf(X[fs][ib].float()), y[ib])
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            clf.eval()
            with torch.no_grad():
                tra = []
                for s0 in range(0, len(tr), 64):
                    ib = trt[s0:s0 + 64]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        tra.append((clf(X[fs][ib].float()).argmax(1) == y[ib]).float())
                for f2, es in PAIRS:
                    if f2 != fs:
                        continue
                    for s0 in range(0, len(te), 64):
                        ib = torch.tensor(te[s0:s0 + 64], device=dev)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pred[f"{fs}->{es}"][te[s0:s0 + 64]] = clf(X[es][ib].float()).argmax(1).cpu().numpy()
            print(f"[{inp}/{trset}/{tgt}] fold {fold} fit {fs} · train n {len(tr)} acc {100 * torch.cat(tra).mean():.1f}", flush=True)
            del clf, opt
    json.dump({k: v.tolist() for k, v in pred.items()}, open(os.path.join(OUT, f"pred_{inp}_{trset}_{tgt}{'' if a.epochs == 25 else f'_e{a.epochs}'}.json"), "w"))  # 2026-10-29: epoch 를 바꾼 판은 이름에 _e{N}
    print("끝", inp, trset, tgt)

elif a.phase == "probe_t":
    # 2026-10-29 사용자: "p->p 이거 late k1 이거 t 별로 보여줘봐". 미래 블록 t 하나의 토큰 (256, D) 만 넣는다. --job = t (0–7).
    # 학습 집합 두 가지: 전 조건 25 epoch · late 만 100 epoch (late 만 25 epoch 은 p 판독기가 덜 학습됐다). 판독 h→h · p→p · h→p.
    M = json.load(open(os.path.join(CACHE, "meta.json"))); dev = torch.device("cuda"); t = a.job
    X = {s_: torch.from_numpy(np.ascontiguousarray(np.load(os.path.join(CACHE, f"{s_}.npy"), mmap_mode="r")[:, t * 256:(t + 1) * 256])).to(dev) for s_ in ("h", "p")}
    fold_of = folds(M); os.makedirs(os.path.join(OUT, "per_t"), exist_ok=True)
    for trset, ep in (("all", 25), ("late", 100)):
        for tgt in ("shape", "color"):
            y = torch.tensor([m[tgt] for m in M], device=dev); ncls = len(SHAPES if tgt == "shape" else COLORS)
            pred = {k: np.full(len(M), -1) for k in ("h->h", "p->p", "h->p")}
            for fold in range(5):
                te = [i for i, m in enumerate(M) if fold_of[m["block_id"]] == fold]
                tr = [i for i, m in enumerate(M) if fold_of[m["block_id"]] != fold and (trset == "all" or m["occlusion"] == "late")]
                for fs in ("h", "p"):
                    clf, tra = fit(X[fs], y, tr, ncls, ep, fold, dev)
                    for es in (("h", "p") if fs == "h" else ("p",)):
                        pred[f"{fs}->{es}"][te] = predict(clf, X[es], te, dev)
                    print(f"[t{t + 1}/{trset}/{tgt}] fold {fold} fit {fs} · train n {len(tr)} acc {tra:.1f}", flush=True)
                    del clf
            json.dump({k: v.tolist() for k, v in pred.items()}, open(os.path.join(OUT, "per_t", f"t{t + 1}_{trset}_{tgt}.json"), "w"))
    print("끝", t + 1)

elif a.phase == "probe_vis":
    # 2026-10-29 사용자: "context encoder 도 … 물체가 보일 때랑 안 보일 때 각각 probe 학습해서 성능 좀 재봐".
    # 문맥 블록 b (2 장) 하나의 토큰 (256, D). 블록 범주는 VOBench metadata 의 samples (장별 . 보임 / H 가림): '..' 보임 · 'HH' 가림 · 나머지 섞임.
    # 판독기 둘: 보이는 블록으로 학습 · 가린 블록으로 학습 (보임 학습 표본은 가림 학습 표본 수와 같게 무작위로 줄인다). 둘 다 시험 fold 의 모든 블록에 건다.
    # --job: 0 z·모양 1 z·색 2 zi·모양 3 zi·색  (z = 문맥 16 장을 한 번에 넣은 문맥 encoder, zi = 블록 2 장만 넣은 문맥 encoder)
    src, tgt = [("z", "shape"), ("z", "color"), ("zi", "shape"), ("zi", "color")][a.job]
    M = json.load(open(os.path.join(CACHE, "meta.json"))); dev = torch.device("cuda")
    md = {r["file_name"]: r["samples"].split("|")[0] for r in csv.DictReader(open("/workspace/dataset/benchmarks/VOBench_v1/metadata.csv"))}
    catof = lambda pr: "visible" if pr == ".." else "hidden" if pr == "HH" else "mixed"
    cats = [[catof(md[m["file_name"]][2 * b:2 * b + 2]) for b in range(NB)] for m in M]
    Xb = torch.from_numpy(np.load(os.path.join(CACHE, f"{src}.npy"), mmap_mode="r")[:]).to(dev).view(len(M) * NB, 256, D)   # 표본 = i·8 + b
    yb = torch.tensor([m[tgt] for m in M], device=dev).repeat_interleave(NB)
    ncls = len(SHAPES if tgt == "shape" else COLORS)
    fold_of = folds(M); rec = []
    keep = [m["k"] in (2, 3) if a.subset == "k23" else True for m in M]
    for fold in range(5):
        trv = [i for i, m in enumerate(M) if fold_of[m["block_id"]] != fold and keep[i]]
        tev = [i for i, m in enumerate(M) if fold_of[m["block_id"]] == fold and keep[i]]
        tr_h = [i * NB + b for i in trv for b in range(NB) if cats[i][b] == "hidden"]
        tr_v_all = [i * NB + b for i in trv for b in range(NB) if cats[i][b] == "visible"]
        tr_v = sorted(np.random.default_rng(fold).choice(tr_v_all, len(tr_h), replace=False).tolist())
        te = [i * NB + b for i in tev for b in range(NB)]
        out = {}
        for nm, tr in (("vis", tr_v), ("hid", tr_h)):
            clf, tra = fit(Xb, yb, tr, ncls, 25, fold, dev)
            out[nm] = np.concatenate([predict(clf, Xb, te[s0:s0 + 64], dev) for s0 in range(0, len(te), 64)])
            print(f"[{src}/{tgt}] fold {fold} 학습 {nm} n {len(tr)} acc {tra:.1f}", flush=True)
            del clf
        for j, sid in enumerate(te):
            i, b = divmod(sid, NB)
            rec.append(dict(i=i, b=b, cat=cats[i][b], y=M[i][tgt], vis=int(out["vis"][j]), hid=int(out["hid"][j])))
    od = os.path.join(OUT, "ctx_vis" if a.subset == "all" else "ctx_vis_k23"); os.makedirs(od, exist_ok=True)
    json.dump(rec, open(os.path.join(od, f"{src}_{tgt}.json"), "w"))
    print("끝", src, tgt)

elif a.phase == "report_vis":
    M = json.load(open(os.path.join(CACHE, "meta.json"))); res = {}
    md = {r["file_name"]: r["samples"].split("|")[0] for r in csv.DictReader(open("/workspace/dataset/benchmarks/VOBench_v1/metadata.csv"))}
    def order(r):   # 가린 블록이 연속 가림의 몇 번째인가 (1 부터)
        s_ = md[M[r["i"]]["file_name"]]; k = 0
        for b in range(r["b"], -1, -1):
            if s_[2 * b:2 * b + 2] != "HH":
                break
            k += 1
        return k
    for src in ("z", "zi"):
        R = {tgt: json.load(open(os.path.join(OUT, "ctx_vis", f"{src}_{tgt}.json"))) for tgt in ("shape", "color")
             if os.path.exists(os.path.join(OUT, "ctx_vis", f"{src}_{tgt}.json"))}
        if len(R) < 2:
            continue
        acc = lambda rs, k: 100 * float(np.mean([r[k] == r["y"] for r in rs])) if rs else float("nan")
        name = "z (문맥 16 장을 한 번에)" if src == "z" else "zi (블록 2 장만 따로)"
        print(f"\n=== {name} · 정확도 % 모양 / 색 (chance 16.7 / 12.5) ===")
        print(f"{'행':10s} {'보임판독→보임':>14s} {'가림판독→가림':>14s} {'보임판독→가림':>14s} {'가림판독→보임':>14s} {'n 보임/가림':>12s}")
        rows = [(f"t{b + 1}", lambda r, b=b: r["b"] == b) for b in range(NB)] + [("전체", lambda r: True)] + \
               [(f"가림 {k}번째", lambda r, k=k: r["cat"] != "hidden" or order(r) == k) for k in (1, 2, 3)]
        for lab, f in rows:
            cell = []
            for tr_, ev in (("vis", "visible"), ("hid", "hidden"), ("vis", "hidden"), ("hid", "visible")):
                cell.append(" / ".join(f"{acc([r for r in R[t] if f(r) and r['cat'] == ev], tr_):5.1f}" for t in ("shape", "color")))
            nv = sum(1 for r in R["shape"] if f(r) and r["cat"] == "visible"); nh = sum(1 for r in R["shape"] if f(r) and r["cat"] == "hidden")
            res[f"{src}|{lab}"] = cell + [nv, nh]
            print(f"{lab:10s} " + " ".join(f"{c:>14s}" for c in cell) + f" {nv:5d}/{nh:<5d}")
        mx = [r for r in R["shape"] if r["cat"] == "mixed"]
        print(f"섞인 블록 (n {len(mx)}): 보임판독 " + " / ".join(f"{acc([r for r in R[t] if r['cat'] == 'mixed'], 'vis'):.1f}" for t in R) +
              " · 가림판독 " + " / ".join(f"{acc([r for r in R[t] if r['cat'] == 'mixed'], 'hid'):.1f}" for t in R))
    json.dump(res, open(os.path.join(OUT, "ctx_vis", "summary.json"), "w"), ensure_ascii=False, indent=1)

elif a.phase == "probe_k":
    # 2026-10-29 사용자: "16 frames 넣고. k = 2 부터 … 1~7 block probe 결과, 8 block probe 결과 이렇게 보여달라고".
    # 한 조건의 영상만 쓴다 (60 개). z = 문맥 16 장을 한 번에 넣은 문맥 encoder. 판독기 둘을 따로 학습:
    #   앞 블록 (물체 보임) 토큰 전부 — late k2 · visible 은 블록 1–7, late k3 은 블록 7 이 반쯤 가려서 1–6 /  블록 8 토큰 (late k2 · k3 은 두 장 다 가림)
    # 표본이 작아 (fold 당 학습 48) epoch 200. --job: 0–5 = (late k2, late k3, visible) × (모양, 색)
    # 6–9 (같은 날 추가): 한 조건 60 개로는 visible 블록 8 도 73 / 40 이라 판독기가 표본에 묶인다 → 두 조건씩 합친 120 개
    #   late k2+k3 (블록 8 두 장 가림, 앞 블록 1–6 보임) · visible+early k1 (블록 8 보임 — 대조)
    c_, tgt = [(c, t) for c in ("late k2", "late k3", "visible", "late k2+late k3", "visible+early k1") for t in ("shape", "color")][a.job]
    M = json.load(open(os.path.join(CACHE, "meta.json"))); dev = torch.device("cuda")
    ii = [i for i, m in enumerate(M) if cond(m) in c_.split("+")]
    Z = torch.from_numpy(np.load(os.path.join(CACHE, "z.npy"), mmap_mode="r")[ii]).to(dev)
    nb_front = 6 if "late k3" in c_ else 7
    heads = {f"블록 1–{nb_front}": Z[:, :nb_front * 256], "블록 8": Z[:, 7 * 256:]}
    if c_ == "visible+early k1":                     # early k1 은 블록 3 이 가려서 앞 블록 판독기는 뺀다
        heads.pop("블록 1–7")
    y = torch.tensor([M[i][tgt] for i in ii], device=dev); ncls = len(SHAPES if tgt == "shape" else COLORS)
    fold_of = folds(M); fo = np.array([fold_of[M[i]["block_id"]] for i in ii]); res = {}
    for nm, X in heads.items():
        pr = np.full(len(ii), -1); tra_ = []
        for fold in range(5):
            tr = np.where(fo != fold)[0].tolist(); te = np.where(fo == fold)[0].tolist()
            clf, tra = fit(X, y, tr, ncls, 200, fold, dev); tra_.append(tra)
            pr[te] = predict(clf, X, te, dev); del clf
        res[nm] = dict(acc=100 * float((pr == y.cpu().numpy()).mean()), n=len(ii), train_acc=float(np.mean(tra_)), pred=pr.tolist())
        print(f"[{c_}/{tgt}] {nm} 시험 {res[nm]['acc']:.1f} · 학습 {res[nm]['train_acc']:.1f} · n {len(ii)}", flush=True)
    os.makedirs(os.path.join(OUT, "ctx_k"), exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "ctx_k", f"{c_.replace(' ', '_')}_{tgt}.json"), "w"), ensure_ascii=False)
    print("끝")

elif a.phase == "report_k":
    for c_ in ("late k2", "late k3", "visible", "late k2+late k3", "visible+early k1"):
        fn = lambda t: os.path.join(OUT, "ctx_k", f"{c_.replace(' ', '_')}_{t}.json")
        if not os.path.exists(fn("color")):
            continue
        R = {t: json.load(open(fn(t))) for t in ("shape", "color")}
        print(c_, " · ".join(f"{h} {R['shape'][h]['acc']:.1f} / {R['color'][h]['acc']:.1f} (학습 {R['shape'][h]['train_acc']:.0f}/{R['color'][h]['train_acc']:.0f})" for h in R["shape"]),
              "n", next(iter(R["shape"].values()))["n"])

elif a.phase == "report_vis23":
    # 보임 판독기는 보이는 블록에서만, 가림 판독기는 가린 블록에서만 잰다 (정보 존재). 행 = 조건 × 블록 위치.
    M = json.load(open(os.path.join(CACHE, "meta.json"))); od = os.path.join(OUT, "ctx_vis_k23")
    R = {t: json.load(open(os.path.join(od, f"z_{t}.json"))) for t in ("shape", "color")}
    acc = lambda rs, k: (100 * float(np.mean([r[k] == r["y"] for r in rs])), len(rs)) if rs else (float("nan"), 0)
    res = {}
    print("16 장 문맥 encoder · 블록 하나 (256, 1280) · 정확도 % 모양 / 색 (chance 16.7 / 12.5)")
    print(f"{'조건':10s} {'블록':>4s} {'보이는 블록 (보임 판독기)':>22s} {'가린 블록 (가림 판독기)':>22s}")
    for c_ in [f"{o} k{k}" for o in ("early", "mid", "late") for k in (2, 3)]:
        for b in list(range(NB)) + ["전체"]:
            f = lambda r: cond(M[r["i"]]) == c_ and (b == "전체" or r["b"] == b)
            cell = {}
            for ev, k in (("visible", "vis"), ("hidden", "hid")):
                v = [acc([r for r in R[t] if f(r) and r["cat"] == ev], k) for t in ("shape", "color")]
                cell[ev] = v
            if cell["hidden"][0][1] == 0 and b != "전체":
                continue
            st = lambda v: f"{v[0][0]:5.1f} / {v[1][0]:5.1f} (n {v[0][1]:3d})" if v[0][1] else "–"
            res[f"{c_}|{b}"] = cell
            print(f"{c_:10s} {('t' + str(b + 1)) if b != '전체' else b:>4s} {st(cell['visible']):>22s} {st(cell['hidden']):>22s}")
    for ev, k in (("visible", "vis"), ("hidden", "hid")):
        v = [acc([r for r in R[t] if r["cat"] == ev], k) for t in ("shape", "color")]
        print(f"모든 조건 {ev}: {v[0][0]:.1f} / {v[1][0]:.1f} (n {v[0][1]})")
    json.dump(res, open(os.path.join(od, "summary.json"), "w"), ensure_ascii=False, indent=1)

elif a.phase == "probe_fig":
    # 2026-10-29 사용자: "(a) 는 ctx 해서 visible, occluded 이렇게만 (obj, shape, color), (b) 를 predictor 로".
    # 표현: z = 문맥 encoder (16 장, 문맥 블록 1–8) / p = predictor 출력 1280 (미래 블록 1–8). 입력 = 블록 하나 (256, 1280).
    # 블록 범주 = 그 블록 두 장의 가림 (VOBench samples: z 는 문맥 절반, p 는 미래 절반 — p 의 '가린 블록' 은 실제 미래에서 가려진 블록).
    # 목표: obj = 물체 있음 (permanence block 의 pos_a) vs 빈 장면 (pos_b), 같은 장면·같은 가림판 / shape · color = 물체 영상.
    # 영상: k 2·3 (전 타이밍). 판독기: 보임 · 가림 따로, 학습 표본 수를 같게 (보임을 줄임), 각자 자기 블록에서 시험. block 단위 5-fold.
    # --job: 0–5 = (z, p) × (obj, shape, color), 6–8 = h × (obj, shape, color) (실제 미래, 대조군 — 범주는 p 와 같이 미래 절반)
    # ⚠️ 2026-10-29 사용자 지적 ("obj 는 occluded 가 더 잘하네"): --match count 는 표본 수만 맞춰, p 의 가린 블록 (late k2·k3 미래 블록 1 하나) 과
    #    보이는 블록 (6 조건 × 미래 1–8) 의 구성이 달랐다. --match pos 가 위치까지 맞춘 판 (fig_vis_pm/)
    rep, tgt = [(r_, t_) for r_ in ("z", "p", "h") for t_ in ("obj", "shape", "color")][a.job]
    pm = a.match == "pos"
    M = json.load(open(os.path.join(CACHE, "meta.json"))); Me = json.load(open(os.path.join(CACHE, "meta_empty.json"))); dev = torch.device("cuda")
    md = {r["file_name"]: r["samples"] for r in csv.DictReader(open("/workspace/dataset/benchmarks/VOBench_v1/metadata.csv"))}
    half = 0 if rep == "z" else 1                                   # z = 문맥 절반, p · h = 미래 절반
    ALL = M + Me                                                   # 전역 번호 = M 의 600 다음에 빈 장면 120
    if tgt == "obj":
        vids = [g for g, m in enumerate(ALL) if m["violation"] == "permanence" and m["k"] in (2, 3)]
        lab = {g: int(g < len(M)) for g in vids}; ncls = 2
    else:
        vids = [g for g, m in enumerate(M) if m["k"] in (2, 3)]
        lab = {g: M[g][tgt] for g in vids}; ncls = len(SHAPES if tgt == "shape" else COLORS)
    need = sorted(vids); pos = {g: j for j, g in enumerate(need)}
    # 2026-10-29 (c) 추가: rep = z 이면 같은 판독기를 p 의 같은 범주 블록에도 건다 (문맥 encoder → predictor 이식). 결과는 zp_{tgt}.json
    def load(r_):
        a_main = np.load(os.path.join(CACHE, f"{r_}.npy"), mmap_mode="r"); a_emp = np.load(os.path.join(CACHE, f"{r_}_empty.npy"), mmap_mode="r")
        return torch.from_numpy(np.stack([a_main[g] if g < len(M) else a_emp[g - len(M)] for g in need])).to(dev).view(len(need) * NB, 256, D)
    X = load(rep); XP = load("p") if rep == "z" else None
    y = torch.tensor([lab[g] for g in need], device=dev).repeat_interleave(NB)
    catof = lambda pr: "visible" if pr == ".." else "occluded" if pr == "HH" else "mixed"
    cat = {g: [catof(md[ALL[g]["file_name"]].split("|")[half][2 * b:2 * b + 2]) for b in range(NB)] for g in need}
    cat_p = {g: [catof(md[ALL[g]["file_name"]].split("|")[1][2 * b:2 * b + 2]) for b in range(NB)] for g in need}   # p 블록 = 미래 절반
    occ_pos = {b for g in need for b in range(NB) if cat[g][b] == "occluded"}; occ_pos_p = {b for g in need for b in range(NB) if cat_p[g][b] == "occluded"}
    print(f"[{rep}/{tgt}] match {a.match} · 가린 블록 위치 {sorted(occ_pos)}", flush=True)
    fold_of = folds(M); rec = []; rec_t = []
    for fold in range(5):
        trv = [g for g in need if fold_of[ALL[g]["block_id"]] != fold]; tev = [g for g in need if fold_of[ALL[g]["block_id"]] == fold]
        sid = lambda vs, c, cc=cat: [pos[g] * NB + b for g in vs for b in range(NB) if cc[g][b] == c]
        tr_o = sid(trv, "occluded"); rng_ = np.random.default_rng(fold)
        if pm:                                                      # 가린 블록 하나마다 같은 위치의 보이는 블록 하나 (비복원)
            pool = collections.defaultdict(list)
            for t_ in sid(trv, "visible"):
                pool[t_ % NB].append(t_)
            for b_ in pool:
                rng_.shuffle(pool[b_])
            tr_v = sorted(pool[t_ % NB].pop() for t_ in tr_o if pool[t_ % NB])
        else:
            tr_v = sorted(rng_.choice(sid(trv, "visible"), len(tr_o), replace=False).tolist())
        for c, tr in (("visible", tr_v), ("occluded", tr_o)):
            ep = max(25, int(np.ceil(600 / max(1, (len(tr) + 15) // 16))))           # 표본이 작으면 최소 600 걸음
            clf, tra = fit(X, y, tr, ncls, ep, fold, dev)
            te = [t_ for t_ in sid(tev, c) if not pm or t_ % NB in occ_pos]; pr = predict(clf, X, te, dev) if te else []
            rec += [dict(g=need[t_ // NB], b=t_ % NB, cat=c, y=int(y[t_]), pred=int(q)) for t_, q in zip(te, pr)]
            if XP is not None:
                tp = [t_ for t_ in sid(tev, c, cat_p) if not pm or t_ % NB in occ_pos_p]; qp = predict(clf, XP, tp, dev) if tp else []
                rec_t += [dict(g=need[t_ // NB], b=t_ % NB, cat=c, y=int(y[t_]), pred=int(q)) for t_, q in zip(tp, qp)]
            print(f"[{rep}/{tgt}] fold {fold} {c} 학습 n {len(tr)} ep {ep} acc {tra:.1f} · 시험 n {len(te)}", flush=True)
            del clf
    FV = os.path.join(OUT, "fig_vis_pm" if pm else "fig_vis"); os.makedirs(FV, exist_ok=True)
    json.dump(rec, open(os.path.join(FV, f"{rep}_{tgt}.json"), "w"))
    if XP is not None:
        json.dump(rec_t, open(os.path.join(FV, f"zp_{tgt}.json"), "w"))
        for c in ("visible", "occluded"):
            rr = [r for r in rec_t if r["cat"] == c]
            print(f"z→p {tgt} {c}: {100 * np.mean([r['pred'] == r['y'] for r in rr]):.1f} (n {len(rr)})")
    for c in ("visible", "occluded"):
        rr = [r for r in rec if r["cat"] == c]
        print(f"{rep} {tgt} {c}: {100 * np.mean([r['pred'] == r['y'] for r in rr]):.1f} (n {len(rr)})")
    print("끝")

elif a.phase == "l1_static":
    # 2026-10-29 사용자: "static occlusion 에선 L1 비교할 때 predictor 는 항상 물체가 없는 미래를 생각해. 이상하잖아".
    # 쌍: permanence block 의 pos_a (물체 계속 있음) ↔ imp_ab (물체가 가림 뒤에서 사라짐). 문맥이 픽셀 단위로 같아 p 가 같다 (확인).
    # 채점 = mean|p − h| (미래 2048 토큰). 정답 = s(imp) > s(pos). ⚠️ 캐시는 bf16 (probing 관례) — 공식 채점 (fp32 + fp16 autocast) 과 수치가 조금 다를 수 있다.
    # 분해: 물체 토큰 = h_pos 와 h_imp 가 가장 다른 미래 토큰 상위 5 % (≈ 102 개). 차이 s(imp) − s(pos) 를 물체 토큰 몫과 나머지 몫으로 나눈다.
    # 물체다움 o = d(p, h_imp) / (d(p, h_pos) + d(p, h_imp)) — 물체 토큰에서, 0.5 보다 크면 p 가 물체 쪽에 가깝다.
    M = json.load(open(os.path.join(CACHE, "meta.json"))); Mi = json.load(open(os.path.join(CACHE, "meta_permimp.json")))
    L = lambda n_: np.load(os.path.join(CACHE, f"{n_}.npy"), mmap_mode="r")
    h, pp, hi, pi = L("h"), L("p"), L("h_permimp"), L("p_permimp")
    ipos = {m["block_id"]: i for i, m in enumerate(M) if m["violation"] == "permanence" and m["variant"] == "pos_a"}
    rows = []; pdiff = 0.0
    for j, m in enumerate(Mi):
        i = ipos[m["block_id"]]
        P = pp[i].astype(np.float32); Hp = h[i].astype(np.float32); Hi = hi[j].astype(np.float32)
        pdiff = max(pdiff, float(np.abs(P - pi[j].astype(np.float32)).max()))
        dp = np.abs(P - Hp).mean(-1); di = np.abs(P - Hi).mean(-1)                 # (2048,) 토큰별 거리
        dh = np.abs(Hp - Hi).mean(-1); obj = dh >= np.quantile(dh, 0.95)
        o = di[obj] / (dp[obj] + di[obj])
        rows.append(dict(motion=m["motion"], occ=m["occlusion"], k=m["k"], ok=float(di.mean() > dp.mean()),
                         gap=float(di.mean() - dp.mean()), gap_obj=float((di - dp)[obj].sum() / NT), gap_rest=float((di - dp)[~obj].sum() / NT),
                         o=float(np.median(o)), dist_obj=float(dp[obj].mean()), dist_obj_imp=float(di[obj].mean())))
    print(f"permanence 물체→빈 쌍 {len(rows)} · p (pos_a) 와 p (imp_ab) 최대 차이 {pdiff:.2e} (문맥이 같으면 0 근처)")
    print(f"{'motion':12s} {'가림':8s} {'n':>3s} {'정답%':>6s} {'차이 전체':>10s} {'물체 토큰 몫':>11s} {'나머지 몫':>10s} {'물체다움 o':>10s}")
    for mo in ("flat_static", "전체"):
        for oc in ("visible", "early", "mid", "late"):
            r_ = [r for r in rows if (mo == "전체" or r["motion"] == mo) and r["occ"] == oc]
            f = lambda k_: np.mean([r[k_] for r in r_])
            print(f"{mo:12s} {oc:8s} {len(r_):3d} {100 * f('ok'):6.1f} {1e3 * f('gap'):+10.2f} {1e3 * f('gap_obj'):+11.2f} {1e3 * f('gap_rest'):+10.2f} {np.median([r['o'] for r in r_]):10.3f}")
    print("(차이 ×1e-3, + = 정답 쪽 = p 가 물체 있는 미래에 더 가깝다)")
    json.dump(rows, open(os.path.join(OUT, "l1_static_perm.json"), "w"))

elif a.phase == "l1_dir":
    # 2026-10-29 사용자: "vobench late static 67 아니야? (있음→있음, 없음→없음) 각각". permanence 두 방향:
    #   A = 문맥에 물체 · pos_a (계속 있음) vs imp_ab (사라짐)    B = 빈 문맥 · pos_b (계속 없음) vs imp_ba (나타남). 정답 = s(imp) > s(pos)
    M = json.load(open(os.path.join(CACHE, "meta.json"))); Me = json.load(open(os.path.join(CACHE, "meta_empty.json")))
    Ma = json.load(open(os.path.join(CACHE, "meta_permimp.json"))); Mb = json.load(open(os.path.join(CACHE, "meta_permimpba.json")))
    L = lambda n_: np.load(os.path.join(CACHE, f"{n_}.npy"), mmap_mode="r")
    S = lambda P, H: float(np.abs(P.astype(np.float32) - H.astype(np.float32)).mean())
    ia = {m["block_id"]: i for i, m in enumerate(M) if m["violation"] == "permanence" and m["variant"] == "pos_a"}
    ib = {m["block_id"]: i for i, m in enumerate(Me)}
    pa, ha, pe, he, hia, pib, hib = L("p"), L("h"), L("p_empty"), L("h_empty"), L("h_permimp"), L("p_permimpba"), L("h_permimpba")
    rows = []
    for j, m in enumerate(Ma):
        i = ia[m["block_id"]]; rows.append(dict(d="A", motion=m["motion"], occ=m["occlusion"], ok=S(pa[i], hia[j]) > S(pa[i], ha[i])))
    mx = 0.0
    for j, m in enumerate(Mb):
        i = ib[m["block_id"]]; mx = max(mx, float(np.abs(pe[i].astype(np.float32) - pib[j].astype(np.float32)).max()))
        rows.append(dict(d="B", motion=m["motion"], occ=m["occlusion"], ok=S(pe[i], hib[j]) > S(pe[i], he[i])))
    print(f"B 문맥 일치 확인: p(pos_b) vs p(imp_ba) 최대 차이 {mx:.2e}")
    MOT = ("flat_static", "flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up")
    for oc in ("visible", "early", "mid", "late"):
        print(f"{oc:8s} " + " · ".join(f"{mo} {100 * np.mean([r['ok'] for r in rows if r['d'] == 'A' and r['occ'] == oc and r['motion'] == mo]):.0f}/"
                                       f"{100 * np.mean([r['ok'] for r in rows if r['d'] == 'B' and r['occ'] == oc and r['motion'] == mo]):.0f}" for mo in MOT))
    json.dump(rows, open(os.path.join(OUT, "l1_perm_dir.json"), "w"))

elif a.phase == "l1_dir_decomp":
    # 2026-10-29 사용자: "문맥 끝까지 비어 있는 경우 (B) 는 왜 chance 인가". late · 방향별로 판정 여유와 그 출처를 나눈다.
    #   신호 = 두 미래 사이 거리 mean|h_pos − h_imp| (얼마나 다른 미래인가) · 여유 = s(imp) − s(pos) (+ = 정답) · 상대 여유 = 여유 / s(pos)
    #   물체 토큰 = 두 미래가 가장 다른 토큰 상위 5 %. 물체다움 o = d(p, h_빈) / (d(p, h_물체) + d(p, h_빈)) — 물체 토큰에서, 0.5 보다 크면 p 가 물체 있는 미래 쪽
    #   (A 는 h_물체 = h_pos, B 는 h_물체 = h_imp 나타남)
    M = json.load(open(os.path.join(CACHE, "meta.json"))); Me = json.load(open(os.path.join(CACHE, "meta_empty.json")))
    Ma = json.load(open(os.path.join(CACHE, "meta_permimp.json"))); Mb = json.load(open(os.path.join(CACHE, "meta_permimpba.json")))
    L = lambda n_: np.load(os.path.join(CACHE, f"{n_}.npy"), mmap_mode="r")
    ia = {m["block_id"]: i for i, m in enumerate(M) if m["violation"] == "permanence" and m["variant"] == "pos_a"}
    ib = {m["block_id"]: i for i, m in enumerate(Me)}
    pa, ha, pe, he, hia, hib = L("p"), L("h"), L("p_empty"), L("h_empty"), L("h_permimp"), L("h_permimpba")
    rows = []
    for d_, MM, idx, P_, Hp_, Hi_ in (("A", Ma, ia, pa, ha, hia), ("B", Mb, ib, pe, he, hib)):
        for j, m in enumerate(MM):
            if m["occlusion"] != "late":
                continue
            i = idx[m["block_id"]]
            P = P_[i].astype(np.float32); Hp = Hp_[i].astype(np.float32); Hi = Hi_[j].astype(np.float32)
            dp = np.abs(P - Hp).mean(-1); di = np.abs(P - Hi).mean(-1); dh = np.abs(Hp - Hi).mean(-1)
            obj = dh >= np.quantile(dh, 0.95)
            d_obj, d_emp = (dp, di) if d_ == "A" else (di, dp)            # p 와 물체 있는 미래 / 빈 미래 거리
            rows.append(dict(d=d_, motion=m["motion"], ok=float(di.mean() > dp.mean()), sig=float(dh.mean()), gap=float(di.mean() - dp.mean()),
                             rel=float((di.mean() - dp.mean()) / dp.mean()), gobj=float((di - dp)[obj].sum() / NT), grest=float((di - dp)[~obj].sum() / NT),
                             o=float(np.median((d_emp / (d_obj + d_emp))[obj]))))
    print("late · permanence · 운동별 평균 (×1e-3 은 거리 단위)")
    print(f"{'방향':4s} {'운동':12s} {'정답%':>6s} {'두 미래 거리':>11s} {'여유':>8s} {'상대 여유':>9s} {'물체 몫':>8s} {'나머지 몫':>9s} {'물체다움 o':>10s}")
    for d_ in ("A", "B"):
        for mo in ("flat_static", "flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up"):
            r_ = [r for r in rows if r["d"] == d_ and r["motion"] == mo]; f = lambda k_: np.mean([r[k_] for r in r_])
            print(f"{d_:4s} {mo:12s} {100 * f('ok'):6.0f} {1e3 * f('sig'):11.1f} {1e3 * f('gap'):+8.2f} {100 * f('rel'):+8.2f}% {1e3 * f('gobj'):+8.2f} {1e3 * f('grest'):+9.2f} {np.median([r['o'] for r in r_]):10.3f}")
    json.dump(rows, open(os.path.join(OUT, "l1_perm_dir_decomp.json"), "w"))

elif a.phase == "l1_copy":
    # 2026-10-29: late 에서 두 미래가 갈리는 토큰 (상위 5 %) 에서 p 가 무엇에 가장 가까운가 — 물체 있는 미래 · 빈 미래 · 복사 (문맥 마지막 블록의 같은 칸, LN(z),
    #   README §3-4 복사 기준선과 같은 정의). 복사에 가장 가까우면 "p 가 그 자리에 가림판/마지막 문맥을 그대로 둔다".
    M = json.load(open(os.path.join(CACHE, "meta.json"))); Me = json.load(open(os.path.join(CACHE, "meta_empty.json")))
    Ma = json.load(open(os.path.join(CACHE, "meta_permimp.json"))); Mb = json.load(open(os.path.join(CACHE, "meta_permimpba.json")))
    L = lambda n_: np.load(os.path.join(CACHE, f"{n_}.npy"), mmap_mode="r")
    ia = {m["block_id"]: i for i, m in enumerate(M) if m["violation"] == "permanence" and m["variant"] == "pos_a"}
    ib = {m["block_id"]: i for i, m in enumerate(Me)}
    pa, ha, za, pe, he, ze, hia, hib = L("p"), L("h"), L("z"), L("p_empty"), L("h_empty"), L("z_empty"), L("h_permimp"), L("h_permimpba")
    R_ = collections.defaultdict(list)
    for d_, MM, idx, P_, Hp_, Z_, Hi_ in (("A", Ma, ia, pa, ha, za, hia), ("B", Mb, ib, pe, he, ze, hib)):
        for j, m in enumerate(MM):
            if m["occlusion"] != "late":
                continue
            i = idx[m["block_id"]]
            P = P_[i].astype(np.float32); Hp = Hp_[i].astype(np.float32); Hi = Hi_[j].astype(np.float32)
            C = np.tile(Z_[i][7 * 256:].astype(np.float32), (NB, 1))                       # 복사: 문맥 마지막 블록을 미래 8 블록에
            dh = np.abs(Hp - Hi).mean(-1); obj = dh >= np.quantile(dh, 0.95)
            Hobj, Hemp = (Hp, Hi) if d_ == "A" else (Hi, Hp)
            dd = [float(np.abs(P[obj] - X_[obj]).mean()) for X_ in (Hobj, Hemp, C)]
            R_[(d_, "static" if m["motion"] == "flat_static" else "moving")].append(dd + [int(np.argmin(dd))])
    print("late · 두 미래가 갈리는 토큰에서 p 까지 거리 (평균) · 가장 가까운 것 비율")
    for k_, v in sorted(R_.items()):
        v = np.array(v)
        print(f"{k_[0]} {k_[1]:7s} n {len(v):2d} · 물체 있는 미래 {v[:, 0].mean():.3f} · 빈 미래 {v[:, 1].mean():.3f} · 복사 {v[:, 2].mean():.3f} · "
              f"가장 가까움 물체 {100 * np.mean(v[:, 3] == 0):.0f}% 빈 {100 * np.mean(v[:, 3] == 1):.0f}% 복사 {100 * np.mean(v[:, 3] == 2):.0f}%")

elif a.phase == "report_t":
    M = json.load(open(os.path.join(CACHE, "meta.json"))); res = {}
    for trset in ("all", "late"):
        for c in ("late k1", "visible"):
            sel = np.array([cond(m) == c for m in M])
            print(f"\n=== {c} · 학습 {'전 조건 (25 ep)' if trset == 'all' else 'late 만 (100 ep)'} · 정확도 % (chance 모양 16.7 / 색 12.5) · n {sel.sum()} ===")
            print(f"{'t':>3s} {'p→p 모양':>9s} {'p→p 색':>8s} {'h→p 모양':>9s} {'h→p 색':>8s} {'h→h 모양':>9s} {'h→h 색':>8s}")
            for t in range(1, NB + 1):
                row = []
                for k in ("p->p", "h->p", "h->h"):
                    for tgt in ("shape", "color"):
                        fn = os.path.join(OUT, "per_t", f"t{t}_{trset}_{tgt}.json")
                        if not os.path.exists(fn):
                            row.append(float("nan")); continue
                        v = np.array(json.load(open(fn))[k]); y = np.array([m[tgt] for m in M])
                        row.append(100 * float((v[sel] == y[sel]).mean()))
                res[f"{trset}|{c}|t{t}"] = row
                print(f"{t:3d} " + " ".join(f"{x:9.1f}" if j % 2 == 0 else f"{x:8.1f}" for j, x in enumerate(row)))
    json.dump(res, open(os.path.join(OUT, "per_t", "summary.json"), "w"), ensure_ascii=False, indent=1)

else:
    M = json.load(open(os.path.join(CACHE, "meta.json")))
    conds = ["visible"] + [f"{o} k{k}" for o in ("early", "mid", "late") for k in (1, 2, 3)]
    res = {}
    import glob
    for fn in sorted(glob.glob(os.path.join(OUT, "pred_*.json"))):
                inp, trset, tgt = os.path.basename(fn)[5:-5].split("_")[:3]
                P = json.load(open(fn)); y = np.array([m[tgt] for m in M])
                print(f"\n=== {os.path.basename(fn)} · 입력 {inp} · 학습 {trset} · 목표 {tgt} (chance {100 / (6 if tgt == 'shape' else 8):.1f}) ===")
                print(f"{'판독':8s} " + " ".join(f"{c:>9s}" for c in conds))
                for k, v in P.items():
                    v = np.array(v); row = {}
                    for c in conds:
                        sel = np.array([cond(m) == c for m in M]) & (v >= 0)
                        row[c] = (100 * float((v[sel] == y[sel]).mean()), int(sel.sum())) if sel.sum() else (float("nan"), 0)
                    res[f"{os.path.basename(fn)}|{k}"] = row
                    print(f"{k:8s} " + " ".join(f"{row[c][0]:9.1f}" for c in conds))
                print("n        " + " ".join(f"{row[c][1]:9d}" for c in conds))
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=1)
