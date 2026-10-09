#!/usr/bin/env python3
"""TrainingEffects — v11 정체 (읽기) · 배치 : predictor 마다 풀링 probe + 기하를 다시 돌리고 릴리즈와 나란히 놓는다.

무엇을 하나
  그룹 A predictor (encoder 는 frozen 릴리즈 ViT-H, predictor 만 다르다) 의 p 로 기존 두 분석을 **코드 수정 없이** 다시 돌린다.
    v11_pooled_probe.py    (PROBE.md)    — 조건 안 정체 probe · 교차 이식 · encoder 머리 → p (raw / shift / Procrustes)
    v11_pooled_geometry.py (GEOMETRY.md) — 클래스 평균 부분공간 공유 몫 · resid 이식 · 노름 · 이동 몫
  두 스크립트는 환경변수 V11P_CACHE (meta.npz · z.npy · p.npy · h.npy 가 있는 폴더) / V11P_PROBE_OUT, geometry 는 --out 을 받는다.
  z · h · meta 는 릴리즈 캐시를 그대로 (symlink) 쓰고 p.npy 만 predictor 마다 바꾼다 → p 와 무관한 수치는 전부 릴리즈와 같아야 한다
  (report 가 JSON 잎 단위로 확인한다 = z/h 재사용 증명).

단계 (subcommand)
  prep     cache/training_effects/v11readout_curves/pool_p__<tag> (16,128 clip) 에서 sel=late ∧ obj 9,408 clip 을 골라
           릴리즈 캐시 순서 (video_id 로 확인) 대로 /data2/.../training_effects/v11pooled_<tag>/p.npy (float16, (9408,8,1280)) 를 쓴다.
           meta.npz · z.npy · h.npy 는 릴리즈 캐시로 symlink. done.npy 가 그 9,408 행에서 전부 1 이 아니면 죽는다.
           검사: pool_z · pool_h ↔ 릴리즈 z · h (같은 encoder 경로 → 상대 L2 ~1e-9), release 의 pool_p ↔ 릴리즈 p (bf16 배치 잡음 ~1e-3).
           tag 'release' 는 검사만 하고 **본 행은 원래 릴리즈 캐시** 를 쓴다 (정본 수치 재현). 다시 뽑은 release 는 'release_bs16'
           (= 다른 predictor 와 같은 추출 패스, 배치 16) 으로 따로 돌려 추출 잡음 바닥으로 쓴다.
  probe    v11_pooled_probe.main() 을 그대로 부른다 (V11P_CACHE / V11P_PROBE_OUT 설정 후 import). 결과 파일은 원 스크립트가 쓴다.
           이 래퍼가 더하는 것 두 가지 (머리 · 특징 · λ 선택 · 수치는 원 코드 그대로):
             (1) evaluate() 를 감싸 **clip 별 정오 벡터**를 perclip.npz 로 남긴다 (results.json 에 없다) —
                 v11_split test 부분집합 정확도와 predictor − release 짝 차이 (같은 clip, block bootstrap) 에 쓴다.
             (2) train_head() 캐시 — 입력 (X · y · 마스크 · λ 후보 · 장치 종류 · probe 코드) sha1 이 같은 머리는 먼저 실행의 것을 쓴다.
                 p 와 무관한 머리만 맞는다 (적중 = 특징이 바이트 단위로 같다는 증거). 공용 GPU/CPU 시간 때문.
           --root probe_cpu --device cpu : predictor 비교 세트 (2026-09-25 실행; 공용 GPU 포화로 CPU 가 더 빨랐다).
           --root probe --device cuda:N  : 정본 재현 검사 (release). --skip-tubelet = 원 스크립트 옵션 ((5b) 생략 → ridge 로 대신).
  geometry v11_pooled_geometry.py 를 subprocess 로 (V11P_CACHE=… --out …) — 원 명령 그대로.
  ridge    PROBE.md §8 닫힌 해 ridge 재검사 (one-vs-rest, λ=10, 같은 표준화 · 같은 split) 를 predictor 마다 (CPU).
           원래 10 칸 + logistic 표의 p 행 전부에 대응하는 ridge 행 → 방향 일치 검사. clip 별 정오도 남긴다.
           정본 10 칸은 릴리즈에서 한 자리도 다르지 않아야 한다 (shape 99.4/96.8/96.0/18.9/24.7/23.6/99.8/19.2/94.6/19.8 …).
  curves   학습 곡선 (preset curves 22 tag, CPU): ridge 핵심 행 + resid 부분공간 공유 몫 (geometry 와 같은 식, 결정적 부분만).
  report   모든 results.json · ridge.json · perclip.npz · curves.json → v11_identity/IDENTITY.md · identity.json · 그림.

정의 (원 스크립트와 같다)
  z_all = z 문맥 8 튜블릿 평균, p_fut = p 미래 8 튜블릿 평균, h_fut = h 튜블릿 8–15 평균. visible = k 0, late = k 1–4 (문맥 끝 가림).
  split = WMADataset._split (block, 0.5, seed 0, condition 층화) — probe test 4,716 clip.
  v11_split test 부분집합 = probe test ∩ data_csv/intphysgen_v11_split/index_test.csv (block 단위로 나뉜다; v11_postft 가 안 본 block).
  CI = test (궤적, k) 묶음 bootstrap 1,000 회 (궤적 = meta.truth 경로가 같은 clip; 2026-09-25 적대 검증 반영 — 이전 block 단위는 1–2 clip 이라
       사실상 clip 단위였다), 짝 차이는 같은 묶음 추출. 부분공간 bootstrap 은 절반 안 묶음 재추출 + basic 구간, |Δ| < SUB_MIN 은 주장 안 함.
  report 가 문서 맨 위에 정정 블록 (md_correction) 을 쓴다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/te_analyze_identity.py prep --tags release v11_e10 ip1_e40 ip2_e80 pv1_e15 ariel_ep43
  OMP_NUM_THREADS=8 $P z_research/scripts/analysis/te_analyze_identity.py probe --tag v11_e10 --device cpu --skip-tubelet --root probe_cpu
  $P z_research/scripts/analysis/te_analyze_identity.py geometry --tag v11_e10 --device cuda:6
  $P z_research/scripts/analysis/te_analyze_identity.py ridge --tags release release_bs16 v11_e10 ip1_e40 ip2_e80 pv1_e15 ariel_ep43
  $P z_research/scripts/analysis/te_analyze_identity.py curves
  $P z_research/scripts/analysis/te_analyze_identity.py report
"""
from __future__ import annotations
import argparse, hashlib, json, linecache, os, subprocess, sys, time
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] in ("ridge", "curves", "report"):
    # CPU 단계만: 공용 노드 (96 코어) 에서 BLAS 스레드가 과다구독되면 작은 행렬곱이 ~25 배 느려진다 (실측 30 ms → 1 ms).
    # probe / geometry 는 원 스크립트의 기본값 그대로 둔다 (정본 수치 재현).
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "8")
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
ANA = ROOT / "z_research/scripts/analysis"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ANA))
PY = "/data/hyuntak/anaconda3/envs/vjepa2/bin/python"
REL = Path("/local_datasets/world/world_analysis/cache/v11_pooled_vith")
CROOT = Path("/data2/local_datasets/world/world_analysis/cache/training_effects")
CURVES = CROOT / "v11readout_curves"
OUTD = ROOT / "z_research/TrainingEffects/v11_identity"
SPLIT_TEST = ROOT / "data_csv/intphysgen_v11_split/index_test.csv"
SPLIT_TRAIN = ROOT / "data_csv/intphysgen_v11_split/index_train.csv"
REF_PROBE = ROOT / "z_research/RollOutV3/figures/v11_probe/results.json"
REF_GEOM = ROOT / "z_research/RollOutV3/figures/v11_geometry/results.json"

FINAL = ["release", "v11_e10", "ip1_e40", "ip2_e80", "pv1_e15", "ariel_ep43"]
RUNS = ["release", "release_bs16", "v11_e10", "ip1_e40", "ip2_e80", "pv1_e15", "ariel_ep43"]
CURVE_TAGS = ["release", "v11_e1", "v11_e2", "v11_e3", "v11_e5", "v11_e10",
              "pv1_e1", "pv1_e3", "pv1_e5", "pv1_e10", "pv1_e15",
              "ip1_e5", "ip1_e10", "ip1_e20", "ip1_e40", "ip2_e10", "ip2_e40", "ip2_e80",
              "ariel_ep5", "ariel_ep19", "ariel_ep30", "ariel_ep43"]
LABEL = {"release": "release", "release_bs16": "release (re-extracted, bs16)", "v11_e10": "v11_postft e10",
         "ip1_e40": "intphys1_postft e40", "ip2_e80": "intphys2_postft e80", "pv1_e15": "predictor_v1_postft e15",
         "ariel_ep43": "ariel scratch ep43"}
DOMAIN = {"release": "—", "release_bs16": "—", "v11_e10": "v11 (same generator, held-out blocks)", "ip1_e40": "no",
          "ip2_e80": "no", "pv1_e15": "partly (occlusion k=4 arm)", "ariel_ep43": "no"}
NBOOT = 1000
SUB_MIN = 0.01          # 2026-09-25: 부분공간 공유 몫 차이는 이 크기 미만이면 주장하지 않는다 (적대 검증)
TARGETS = {"shape": "shape_pre", "color": "color_pre"}


def cache_of(tag: str) -> Path:
    return REL if tag == "release" else CROOT / f"v11pooled_{'release' if tag == 'release_bs16' else tag}"


def src_tag(tag: str) -> str:
    return "release" if tag == "release_bs16" else tag


def sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ============================================================================= prep
def rel_l2(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def cmd_prep(a):
    from te_v11_readout import load
    M = load(a.src)
    mt = M["meta"]
    sel = (mt["sel"].astype(str) == "late") & mt["obj"].astype(bool)
    idx = np.where(sel)[0]
    ref = dict(np.load(REL / "meta.npz"))
    assert len(idx) == len(ref["video_id"]) == 9408, (len(idx), len(ref["video_id"]))
    assert (mt["video_id"].astype(str)[idx] == ref["video_id"].astype(str)).all(), "clip 순서가 릴리즈 캐시와 다르다"
    for k in ("block_id", "condition", "sym_k", "shape_pre", "color_pre", "env", "occ_timing"):
        assert (mt[k].astype(str)[idx] == ref[k].astype(str)).all(), k
    done = np.asarray(M["done"])[idx]
    log(f"done {int(done.sum())}/{len(idx)} (sel=late ∧ obj)")
    if not done.all() and not a.allow_partial:
        raise SystemExit("추출이 아직 안 끝났다 (done=0 행이 있다). --allow-partial 은 검사용으로만")
    ok = done.astype(bool)
    zr = np.load(REL / "z.npy", mmap_mode="r"); hr = np.load(REL / "h.npy", mmap_mode="r"); pr = np.load(REL / "p.npy", mmap_mode="r")
    zi = np.asarray(M["pool_z"][idx[ok]]); hi = np.asarray(M["pool_h"][idx[ok]])
    checks = dict(src=str(a.src), n_sel=int(len(idx)), n_done=int(ok.sum()),
                  z_rel_l2=rel_l2(zi, zr[ok]), z_max_abs=float(np.abs(zi.astype(np.float32) - zr[ok].astype(np.float32)).max()),
                  h_rel_l2=rel_l2(hi, hr[ok]), h_max_abs=float(np.abs(hi.astype(np.float32) - hr[ok].astype(np.float32)).max()),
                  tags={})
    log(f"z ↔ release rel L2 {checks['z_rel_l2']:.2e} · h {checks['h_rel_l2']:.2e}")
    assert checks["z_rel_l2"] < 1e-4 and checks["h_rel_l2"] < 1e-4, "encoder 경로가 릴리즈 캐시와 다르다"
    for tag in a.tags:
        key = f"pool_p__{tag}"
        assert key in M, key
        p = np.asarray(M[key][idx]).astype(np.float16)
        assert p.shape == (9408, 8, 1280) and p.dtype == pr.dtype, (p.shape, p.dtype)
        assert np.isfinite(p[ok].astype(np.float32)).all(), tag
        pc = p[ok].astype(np.float32); prc = pr[ok].astype(np.float32)
        cos = (pc * prc).sum(-1) / np.linalg.norm(pc, axis=-1) / np.linalg.norm(prc, axis=-1)
        c = dict(p_rel_l2_vs_release=rel_l2(pc, prc), p_min_cos_vs_release=float(cos.min()),
                 p_mean_cos_vs_release=float(cos.mean()),
                 p_mean_norm_fut=float(np.linalg.norm(pc.mean(1), axis=1).mean()))
        d = CROOT / f"v11pooled_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        if a.allow_partial:
            c["written"] = False
        else:
            tmp = d / "p.npy.tmp.npy"
            np.save(tmp, p); os.replace(tmp, d / "p.npy")
            for f in ("meta.npz", "z.npy", "h.npy"):
                if (d / f).is_symlink() or (d / f).exists():
                    (d / f).unlink()
                (d / f).symlink_to(REL / f)
            (d / "README.json").write_text(json.dumps(dict(
                tag=tag, p=f"{CURVES.name}/pool_p__{tag}.npy rows sel=late∧obj, release meta order (video_id checked)",
                meta_z_h="symlink → " + str(REL), made_by="z_research/scripts/analysis/te_analyze_identity.py prep",
                date=time.strftime("%Y-%m-%d %H:%M")), indent=1))
            c["written"] = True
            c["p_sha1"] = hashlib.sha1(p.tobytes()).hexdigest()[:12]
        checks["tags"][tag] = c
        log(f"{tag:12s} p rel L2 vs release {c['p_rel_l2_vs_release']:.3e}  min cos {c['p_min_cos_vs_release']:.6f}  "
            f"|p_fut| {c['p_mean_norm_fut']:.2f}  written={c['written']}")
    if "release" in checks["tags"]:
        r = checks["tags"]["release"]
        assert r["p_rel_l2_vs_release"] < 1e-2 and r["p_min_cos_vs_release"] > 0.999, r
    if not a.allow_partial:
        (OUTD / "prep").mkdir(parents=True, exist_ok=True)
        (OUTD / "prep" / "prep_checks.json").write_text(json.dumps(checks, indent=1))


# ============================================================================= probe (wrapper)
def cmd_probe(a):
    tag = a.tag
    cache = cache_of(tag)
    out = OUTD / a.root / tag
    for f in ("meta.npz", "z.npy", "p.npy", "h.npy"):
        assert (cache / f).exists(), cache / f
    os.environ["V11P_CACHE"] = str(cache)
    os.environ["V11P_PROBE_OUT"] = str(out)
    import v11_pooled_probe as vp                                           # env 를 읽고 CACHE/OUT 을 정한다
    assert Path(vp.CACHE) == cache and Path(vp.OUT) == out, (vp.CACHE, vp.OUT)
    orig = vp.evaluate
    recs, corrs = [], []

    def evaluate(head, X, y, block, groups, n_cls):                          # 원 함수 그대로 + clip 별 정오 기록
        res, corr = orig(head, X, y, block, groups, n_cls)
        f = sys._getframe(1)
        if f.f_code.co_name == "<lambda>":
            f = f.f_back
        L = f.f_locals
        rec = dict(fn=f.f_code.co_name, line=f.f_lineno, src=linecache.getline(f.f_code.co_filename, f.f_lineno).strip(),
                   tname=str(L.get("tname")))
        for v in ("key", "vn", "src", "t", "sn", "nm", "fname", "hname"):
            if v in L and isinstance(L[v], (str, int, np.integer)):
                rec["v_" + v] = str(L[v])
        recs.append(rec); corrs.append(np.asarray(corr, bool))
        return res, corr
    vp.evaluate = evaluate
    # 머리 캐시: 입력 (X · y · train 마스크 · hold-out 마스크 · 클래스 수 · λ 후보 · 장치 종류 · probe 코드) 의 sha1 이 같으면
    # 먼저 끝난 실행이 저장한 머리를 그대로 쓴다. p 와 무관한 머리 (z_all · z_hid · h_fut …) 만 맞는다 — p 가 바뀌면 X 해시가 달라진다.
    # 적중 = 그 특징이 **바이트 단위로 릴리즈 실행과 같다**는 증거. 공용 GPU 에서 LBFGS 가 느려 (적중이면 머리 학습을 건너뛴다) 넣었다.
    stats = dict(hit=0, miss=0, hit_keys=[], miss_keys=[])
    vp.train_head = cached_train_head(vp, OUTD / "probe" / "_headcache", stats)
    orig_plot = vp.plot

    def plot(R):                                                             # --skip-tubelet 이면 원 plot() 의 튜블릿 그림이 KeyError
        try:                                                                 # (fig_probe.png 는 그 전에 저장된다). 원 스크립트는 --tag 로 피한다
            orig_plot(R)
        except KeyError:
            if not a.skip_tubelet:
                raise
            log("plot: fig_probe_tubelet 생략 (--skip-tubelet)")
    vp.plot = plot
    sys.argv = ["v11_pooled_probe.py", "--device", a.device] + (["--skip-tubelet"] if a.skip_tubelet else [])
    t0 = time.time()
    vp.main()
    np.savez_compressed(out / "perclip.npz", corr=np.stack(corrs), recs=np.array(json.dumps(recs)),
                        video_id=np.load(cache / "meta.npz")["video_id"])
    (out / "run_info.json").write_text(json.dumps(dict(
        tag=tag, cache=str(cache), device=a.device, skip_tubelet=bool(a.skip_tubelet), elapsed_s=round(time.time() - t0, 1), head_cache=stats,
        probe_script_sha1=sha1(ANA / "v11_pooled_probe.py"), wrapper="te_analyze_identity.py probe (evaluate() 기록만 추가)",
        p_sha1=sha1(cache / "p.npy") if tag != "release" else "original release cache"), indent=1))
    log(f"probe {tag} → {out}  ({(time.time()-t0)/60:.1f} min, {len(recs)} evaluate 호출 기록)")


def cached_train_head(vp, HC, stats):
    """v11_pooled_probe.train_head 를 감싼다. 입력 (X · y · train 마스크 · hold-out 마스크 · 클래스 수 · λ 후보 · 장치 종류 · probe 코드)
    의 sha1 이 같으면 먼저 끝난 실행이 저장한 머리를 그대로 쓴다. p 와 무관한 머리 (z_all · z_hid · h_fut …) 만 맞는다 — p 가 바뀌면
    X 해시가 달라진다. 적중 = 그 특징이 **바이트 단위로 먼저 실행과 같다** 는 증거. 공용 GPU 에서 LBFGS 가 느려 넣었다."""
    HC = Path(HC); HC.mkdir(parents=True, exist_ok=True)
    orig_th = vp.train_head
    code = sha1(Path(vp.__file__))

    def train_head(X, y, tr, inner_fit, n_cls, dev):
        hsh = hashlib.sha1()
        for arr in (np.ascontiguousarray(X), np.asarray(y), np.asarray(tr), np.asarray(inner_fit)):
            hsh.update(arr.dtype.str.encode()); hsh.update(str(arr.shape).encode()); hsh.update(arr.tobytes())
        hsh.update(f"{n_cls}|{vp.LAMS}|{torch_dev_type(dev)}|{code}".encode())
        key = hsh.hexdigest()[:20]
        f = HC / f"{key}.npz"
        if f.exists():
            d = np.load(f, allow_pickle=False)
            stats["hit"] += 1; stats["hit_keys"].append(key)
            return vp.Head(d["mu"], d["sd"], d["W"], d["b"], float(d["lam"]), json.loads(str(d["info"])))
        hd = orig_th(X, y, tr, inner_fit, n_cls, dev)
        stats["miss"] += 1; stats["miss_keys"].append(key)
        tmp = HC / f".{key}.{os.getpid()}.npz"
        np.savez(tmp, mu=hd.mu, sd=hd.sd, W=hd.W, b=hd.b, lam=hd.lam, info=np.array(json.dumps(hd.info)))
        os.replace(tmp, f)
        return hd
    return train_head


def torch_dev_type(dev):
    return getattr(dev, "type", str(dev).split(":")[0])


def cmd_geometry(a):
    cache = cache_of(a.tag)
    out = OUTD / "geometry" / a.tag
    env = dict(os.environ, V11P_CACHE=str(cache))
    cmd = [PY, str(ANA / "v11_pooled_geometry.py"), "--device", a.device, "--out", str(out)]
    log("V11P_CACHE=" + str(cache) + " " + " ".join(cmd))
    subprocess.run(cmd, env=env, check=True, cwd=str(ROOT))


# ============================================================================= shared data
def load_cache(tag):
    c = cache_of(tag)
    m = np.load(c / "meta.npz")
    meta = {k: m[k] for k in m.files}
    z = np.load(c / "z.npy").astype(np.float64); p = np.load(c / "p.npy").astype(np.float64)
    h = np.load(c / "h.npy").astype(np.float64)
    return meta, z, p, h


def splits(meta):
    from evals.world_model_analysis.data import WMADataset
    vid, blk, cond = (meta[k].astype(str) for k in ("video_id", "block_id", "condition"))
    rows = [dict(video_id=v, block_id=b, condition=c) for v, b, c in zip(vid, blk, cond)]
    s = WMADataset._split(rows, dict(mode="ratio", group_by="block_id", train_frac=0.5, seed=0, stratify_by="condition"))
    tr = np.array([s[v] == "train" for v in vid])
    import csv
    te_ids = {r["video_id"] for r in csv.DictReader(open(SPLIT_TEST))}
    tr_ids = {r["video_id"] for r in csv.DictReader(open(SPLIT_TRAIN))}
    v11te = np.array([v in te_ids for v in vid]); v11tr = np.array([v in tr_ids for v in vid])
    assert not (v11te & v11tr).any() and (v11te | v11tr).all()
    for b in np.unique(blk):                                                 # block 단위로 갈리는지
        mk = blk == b
        assert v11te[mk].all() or (~v11te[mk]).all(), b
    return tr, v11te


def traj_clusters(meta):
    """(궤적, k) CI 단위 (2026-09-25 적대 검증 반영). 궤적 = meta['truth'] (32 프레임 물체 위치) 가 같은 clip 묶음
    (운동 조건 x-궤적 2 개씩 + 정지 8 자리 = 12), k = sym_k. block (1–2 clip) 은 사실상 clip 단위라 거의-복제를 못 묶는다."""
    t = np.round(np.asarray(meta["truth"], np.float64).reshape(len(meta["truth"]), -1), 3)
    _, inv = np.unique(t, axis=0, return_inverse=True)
    return np.array([f"{a}|{b}" for a, b in zip(np.asarray(inv).ravel(), meta["sym_k"].astype(str))])


# ============================================================================= ridge
def ridge_head(X, y, trm, C, lam=10.0):
    """ridge_check.py 와 같은 닫힌 해. 반환 f(Xq, mu=None, sd=None) — mu·sd 를 주면 그것으로 표준화 (restd 대조)."""
    mu = X[trm].mean(0); sd = X[trm].std(0); sd = np.where(sd < 1e-6, 1, sd)
    Xs = (X[trm] - mu) / sd
    Y = np.eye(C)[y[trm]]
    ym = Y.mean(0); xm = Xs.mean(0)
    A = Xs - xm
    W = np.linalg.solve(A.T @ A + lam * np.eye(X.shape[1]), A.T @ (Y - ym))

    def f(Xq, mu_=None, sd_=None):
        m_, s_ = (mu, sd) if mu_ is None else (mu_, sd_)
        return (((Xq - m_) / s_ - xm) @ W + ym).argmax(1)
    return f


def procrustes(src, dst, fm):
    mp_, ms_ = src[fm].mean(0), dst[fm].mean(0)
    U, _, Vt = np.linalg.svd((src[fm] - mp_).T @ (dst[fm] - ms_))
    return (src - mp_) @ (U @ Vt) + ms_


def ridge_rows(meta, z, p, h, tr, full=True):
    """→ {target: {row: (n,) bool 정오}}. 원 10 칸 (ridge_check.py) + logistic 표의 p 행 대응."""
    te = ~tr
    k = meta["sym_k"].astype(int); vis, late, late2 = k == 0, k >= 1, k >= 2
    hid = meta["hidden"]; fullh = hid[:, 0:16:2] & hid[:, 1:16:2]; cnt = fullh.sum(1)
    z_all, p_fut, h_fut = z.mean(1), p.mean(1), h[:, 8:].mean(1)
    z_hid = np.where(cnt[:, None] > 0, (z * fullh[..., None]).sum(1) / np.maximum(cnt, 1)[:, None], np.nan)
    out = {}
    for t, col in TARGETS.items():
        cls = sorted(set(meta[col].astype(str))); y = np.searchsorted(cls, meta[col].astype(str)); C = len(cls)
        R = {}
        hz = ridge_head(z_all, y, tr, C); hh = ridge_head(h_fut, y, tr, C)
        hpl = ridge_head(p_fut, y, tr & late, C); hpv = ridge_head(p_fut, y, tr & vis, C)
        R["p_vis_head"] = hpv(p_fut) == y
        R["p_late_head"] = hpl(p_fut) == y
        ml, sl = p_fut[tr & late].mean(0), p_fut[tr & late].std(0)
        R["p_vis_head_restd"] = hpv(p_fut, ml, np.where(sl < 1e-6, 1, sl)) == y     # (5x) restd 대응
        if full:
            for t_ in range(8):                                                   # (5b) 대응: 미래 튜블릿 하나씩, late 머리
                R[f"p_late_head_t{t_}"] = ridge_head(p[:, t_], y, tr & late, C)(p[:, t_]) == y
        pv = procrustes(p_fut, z_all, tr & vis); pl = procrustes(p_fut, z_all, tr & late); pa = procrustes(p_fut, z_all, tr)
        R["zhead_p_raw"] = hz(p_fut) == y
        R["zhead_p_shift"] = hz(p_fut - p_fut[tr].mean(0) + z_all[tr].mean(0)) == y
        R["zhead_p_procr_vis"] = hz(pv) == y
        R["zhead_p_procr_late"] = hz(pl) == y
        R["zhead_p_procr_all"] = hz(pa) == y
        R["hhead_p_raw"] = hh(p_fut) == y
        R["hhead_p_procr_vis"] = hh(procrustes(p_fut, h_fut, tr & vis)) == y
        R["hhead_p_procr_late"] = hh(procrustes(p_fut, h_fut, tr & late)) == y
        if full:                                                              # p 와 무관한 원 칸 (재현 검사용)
            R["z_all"] = hz(z_all) == y
            hzh = ridge_head(z_hid, y, tr & late2, C)
            R["z_hid"] = hzh(np.nan_to_num(z_hid)) == y
            R["zhead_zhid"] = hz(np.nan_to_num(z_hid)) == y
            R["h_fut"] = hh(h_fut) == y
        out[t] = R
    return out


# 원 10 칸 (PROBE.md §8) = (행, 부분)
RIDGE10 = [("z_all", "late"), ("z_hid", "late2"), ("p_late_head", "late"), ("p_vis_head", "late"), ("zhead_zhid", "late2"),
           ("zhead_p_raw", "vis"), ("zhead_p_procr_vis", "vis"), ("zhead_p_procr_vis", "late"),
           ("zhead_p_procr_late", "late"), ("hhead_p_raw", "vis")]
RIDGE10_REF = {"shape": [99.4, 96.8, 96.0, 18.9, 24.7, 23.6, 99.8, 19.2, 94.6, 19.8],
               "color": [98.3, 93.4, 95.5, 30.6, 18.9, 18.5, 99.0, 35.8, 94.4, 20.9]}


def masks_of(meta, tr, v11te):
    k = meta["sym_k"].astype(int)
    te = ~tr
    base = dict(vis=te & (k == 0), late=te & (k >= 1), late2=te & (k >= 2))
    out = {}
    for nm, m in base.items():
        out[nm] = m
        out[nm + "|v11test"] = m & v11te
    return out


def cmd_ridge(a):
    OUT = OUTD / "ridge"; OUT.mkdir(parents=True, exist_ok=True)
    for tag in a.tags:
        t0 = time.time()
        meta, z, p, h = load_cache(tag)
        tr, v11te = splits(meta)
        R = ridge_rows(meta, z, p, h, tr)
        Mk = masks_of(meta, tr, v11te)
        acc = {t: {r: {g: round(100 * float(c[m].mean()), 2) for g, m in Mk.items()} for r, c in rows.items()} for t, rows in R.items()}
        ten = {t: [round(100 * float(R[t][r][Mk[g]].mean()), 1) for r, g in RIDGE10] for t in R}
        # 다시 나누기: train = v11_postft 가 학습한 절반 (index_train), test = 안 본 절반 (index_test). block 단위 (확인됨)
        R2 = ridge_rows(meta, z, p, h, ~v11te, full=False)
        k = meta["sym_k"].astype(int)
        Mk2 = dict(vis=v11te & (k == 0), late=v11te & (k >= 1))
        acc2 = {t: {r: {g: round(100 * float(c[m].mean()), 2) for g, m in Mk2.items()} for r, c in rows.items()} for t, rows in R2.items()}
        np.savez_compressed(OUT / f"perclip_{tag}.npz", **{f"{t}|{r}": c for t, rows in R.items() for r, c in rows.items()},
                            **{f"resplit|{t}|{r}": c for t, rows in R2.items() for r, c in rows.items()})
        J = dict(tag=tag, cache=str(cache_of(tag)), lam=10.0, acc=acc, acc_v11resplit=acc2, ridge10=ten, ridge10_ref=RIDGE10_REF,
                 elapsed_s=round(time.time() - t0, 1))
        (OUT / f"ridge_{tag}.json").write_text(json.dumps(J, indent=1))
        log(f"ridge {tag}: shape {ten['shape']}  color {ten['color']}  ({time.time()-t0:.0f}s)")
        if tag == "release":
            assert ten == RIDGE10_REF, ("릴리즈 ridge 10 칸이 정본과 다르다", ten)


# ============================================================================= curves (CPU, 결정적 부분만)
def resid_view(X, meta, tr):
    env, cond, sk = (meta[k].astype(str) for k in ("env", "condition", "sym_k"))
    cell = np.array([f"{e}|{c}|{k}" for e, c, k in zip(env, cond, sk)])
    u, inv = np.unique(cell, return_inverse=True)
    M = np.zeros((len(u), X.shape[1])); n = np.bincount(inv[tr], minlength=len(u)).astype(float)
    np.add.at(M, inv[tr], X[tr]); M /= n[:, None]
    return X - M[inv]


def class_means(X, yi, C):
    M = np.stack([X[yi == c].mean(0) for c in range(C)])
    return M - M.mean(0)


def basis(Mc, r):
    return np.linalg.svd(Mc, full_matrices=False)[2][:r].T


def captured(Mc, U):
    return float(((Mc @ U) ** 2).sum() / (Mc ** 2).sum())


def xfit_norm(XA, XB, y, m, tr, C):
    """geometry.subspace_section 의 xfit_captured_norm 과 같은 식 (A·B 의 train/test 절반 교차, split-half 천장으로 나눔)."""
    r = C - 1
    H = {}
    for nm, X in (("A", XA), ("B", XB)):
        for hf, mh in (("tr", m & tr), ("te", m & ~tr)):
            Mc = class_means(X[mh], y[mh], C); H[nm, hf] = (Mc, basis(Mc, r))
    cap = (captured(H["A", "te"][0], H["B", "tr"][1]) + captured(H["A", "tr"][0], H["B", "te"][1]) +
           captured(H["B", "te"][0], H["A", "tr"][1]) + captured(H["B", "tr"][0], H["A", "te"][1])) / 4
    sA = (captured(H["A", "te"][0], H["A", "tr"][1]) + captured(H["A", "tr"][0], H["A", "te"][1])) / 2
    sB = (captured(H["B", "te"][0], H["B", "tr"][1]) + captured(H["B", "tr"][0], H["B", "te"][1])) / 2
    return cap / np.sqrt(sA * sB)


COLOR_ORDER = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]


def geom_core(meta, z, p, h, tr):
    k = meta["sym_k"].astype(int); vis = k == 0
    X = dict(z_all=z.mean(1), p_fut=p.mean(1), h_fut=h[:, 8:].mean(1))
    Xr = {kk: resid_view(v, meta, tr) for kk, v in X.items()}
    out = {}
    for lname, col, classes in (("shape", "shape_pre", None), ("color", "color_pre", COLOR_ORDER)):
        ys = meta[col].astype(str); classes = classes or sorted(set(ys)); C = len(classes)
        y = np.array([classes.index(v) for v in ys])
        for sub, m in (("visible", vis), ("late", ~vis)):
            for a_, b_ in (("z_all", "p_fut"), ("h_fut", "p_fut"), ("z_all", "h_fut")):
                out[f"{lname}|{a_}-{b_}|{sub}"] = xfit_norm(Xr[a_], Xr[b_], y, m, tr, C)
    return out


def cmd_curves(a):
    from te_v11_readout import load
    M = load(CURVES)
    mt = M["meta"]
    idx = np.where((mt["sel"].astype(str) == "late") & mt["obj"].astype(bool))[0]
    done = np.asarray(M["done"])[idx]
    assert done.all() or a.allow_partial, f"done {int(done.sum())}/{len(idx)}"
    ref = dict(np.load(REL / "meta.npz"))
    assert (mt["video_id"].astype(str)[idx] == ref["video_id"].astype(str)).all()
    meta = {k: ref[k] for k in ref}
    z = np.load(REL / "z.npy").astype(np.float64); h = np.load(REL / "h.npy").astype(np.float64)
    tr, v11te = splits(meta)
    Mk = masks_of(meta, tr, v11te)
    res = {}
    for tag in a.tags:
        t0 = time.time()
        p = np.asarray(M[f"pool_p__{tag}"][idx]).astype(np.float64)
        R = ridge_rows(meta, z, p, h, tr, full=False)
        acc = {t: {r: {g: round(100 * float(c[mm].mean()), 2) for g, mm in Mk.items()} for r, c in rows.items()} for t, rows in R.items()}
        G = geom_core(meta, z, p, h, tr)
        pf, hf = p.mean(1), h[:, 8:].mean(1)
        kk = meta["sym_k"].astype(int)
        cos = {sub: float((np.sum(pf[m] * hf[m], 1) / np.linalg.norm(pf[m], axis=1) / np.linalg.norm(hf[m], axis=1)).mean())
               for sub, m in (("visible", ~tr & (kk == 0)), ("late", ~tr & (kk >= 1)))}
        res[tag] = dict(ridge=acc, subspace=G, cos_pfut_hfut=cos, n_done=int(done.sum()))
        log(f"curves {tag:11s} ridge vis→late {acc['shape']['p_vis_head']['late']:.1f}/{acc['color']['p_vis_head']['late']:.1f}"
            f"  z-p share vis {G['shape|z_all-p_fut|visible']:.3f}/{G['color|z_all-p_fut|visible']:.3f}  ({time.time()-t0:.0f}s)")
    (OUTD / "curves").mkdir(parents=True, exist_ok=True)
    (OUTD / "curves" / "curves.json").write_text(json.dumps(dict(tags=a.tags, n_clip=int(len(idx)), res=res), indent=1))


# ============================================================================= report — 공통
def leaves(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, path + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, path + (i,))
    else:
        yield path, o


def get(o, path):
    for k in path:
        o = o[k]
    return o


P_DEP_PROBE = {"exp5_p_vis_head", "exp5_p_late_head", "exp5x_cross_aligned", "exp5_shuffled_vis", "exp5_shuffled_late",
               "exp6", "exp6h", "geometry"}


def probe_pdep(path):
    """probe results.json 잎이 p 에 의존하는가 (None = 비교 안 함: 실행 정보)."""
    if path[0] == "meta":
        return None
    if path[0] == "diffs":
        return ("/exp5" in path[1]) or ("/exp6" in path[1])
    if path[0] == "sanity":
        return "p_fut" in path
    if path[0] in TARGETS:
        if path[1] in P_DEP_PROBE:
            return True
        if path[1] == "exp5b_tubelet":
            return path[2] == "p"
        if path[1] == "exp6_ref":
            return path[2] in ("p_own_heads", "p_all_head", "p_all_head_on_p")
    return False


def geom_pdep(path):
    if path[0] == "meta":
        return None
    if tuple(path[:2]) in (("pca", "joint"), ("tsne", "joint")):
        return True
    return any("p_fut" in str(x) for x in path)


def reuse_check(J, Jref, pdep):
    ref = dict(leaves(Jref))
    n_ind = n_same = n_dep = 0
    bad, maxd = [], 0.0
    for path, v in leaves(J):
        d = pdep(path)
        if d is None:
            continue
        if d:
            n_dep += 1
            continue
        n_ind += 1
        r = ref.get(path, "__missing__")
        if r == v:
            n_same += 1
        else:
            bad.append(path)
            if isinstance(v, (int, float)) and isinstance(r, (int, float)):
                maxd = max(maxd, abs(v - r))
    return dict(n_p_independent=n_ind, n_identical=n_same, n_p_dependent=n_dep, n_mismatch=len(bad),
                max_abs_diff=maxd, first_mismatch=[[str(x) for x in b] for b in bad[:5]])


def full_equal(J, Jref, skip=("meta",), drop=()):
    keep = lambda p: p[0] not in skip and not any(d in p for d in drop)                 # noqa: E731
    a = {p: v for p, v in leaves(J) if keep(p)}
    b = {p: v for p, v in leaves(Jref) if keep(p)}
    only_a = [p for p in a if p not in b]; only_b = [p for p in b if p not in a]
    diff = [p for p in a if p in b and a[p] != b[p]]
    return dict(n_leaves=len(a), n_shared=len(a) - len(only_a), n_value_diff=len(diff), n_only_new=len(only_a), n_only_ref=len(only_b),
                n_diff=len(diff) + len(only_b), first_diff=[[str(x) for x in d] for d in sorted(diff, key=str)[:5]])


def acc_ci(c, blk, m, seed=0):
    ub, inv = np.unique(blk[m], return_inverse=True)
    cs, ns = np.bincount(inv, weights=c[m].astype(float)), np.bincount(inv).astype(float)
    ix = np.random.default_rng(seed).integers(0, len(ub), size=(NBOOT, len(ub)))
    boot = cs[ix].sum(1) / ns[ix].sum(1)
    return dict(acc=100 * float(c[m].mean()), lo=100 * float(np.percentile(boot, 2.5)), hi=100 * float(np.percentile(boot, 97.5)),
                n=int(m.sum()), n_blocks=int(len(ub)))


def paired_ci(c1, c0, blk, m, seed=0):
    """(c1 − c0) 정확도 차, 같은 clip · 같은 block 추출."""
    ub, inv = np.unique(blk[m], return_inverse=True)
    a = np.bincount(inv, weights=c1[m].astype(float)); b = np.bincount(inv, weights=c0[m].astype(float))
    n = np.bincount(inv).astype(float)
    ix = np.random.default_rng(seed).integers(0, len(ub), size=(NBOOT, len(ub)))
    boot = (a[ix].sum(1) - b[ix].sum(1)) / n[ix].sum(1)
    return dict(d=100 * float(c1[m].mean() - c0[m].mean()), lo=100 * float(np.percentile(boot, 2.5)),
                hi=100 * float(np.percentile(boot, 97.5)))


def sign_of(ci):
    return 1 if ci["lo"] > 0 else (-1 if ci["hi"] < 0 else 0)


# logistic 행 (probe results.json 과 perclip 기록을 잇는다). (id, 라벨, src 접두, 지역변수 조건, 부분, results 경로, ridge 행)
LROWS = [
    ("p_vis_vis", "p visible 머리 → visible", "ev5vv, c5vv = E(h5v", {}, "vis", ("exp5_p_vis_head", "test_visible", "visible"), "p_vis_head"),
    ("p_late_late", "p late 머리 → late", "ev5ll, c5ll = E(h5l", {}, "late", ("exp5_p_late_head", "test_late", "late"), "p_late_head"),
    ("p_vis_late", "p visible 머리 → late (교차)", "ev5vv, c5vv = E(h5v", {}, "late", ("exp5_p_vis_head", "test_late", "late"), "p_vis_head"),
    ("p_late_vis", "p late 머리 → visible (교차)", "ev5ll, c5ll = E(h5l", {}, "vis", ("exp5_p_late_head", "test_visible", "visible"), "p_late_head"),
    ("p_vis_late_restd", "p visible 머리 → late, 재표준화 (5x)", "vis_head_on_late_restd=E(", {}, "late",
     ("exp5x_cross_aligned", "vis_head_on_late_restd", "late"), "p_vis_head_restd"),
]
_RID = {"zhead": {"raw", "shift", "procr_vis", "procr_late", "procr_all"}, "hhead": {"raw", "procr_vis", "procr_late"}}
for _hk, _key, _rid in (("z", "exp6", "zhead"), ("h", "exp6h", "hhead")):
    for _vn, _vl, _rs in (("raw", "raw", "raw"), ("shift", "평균 이동", "shift"), ("procrustes", "Procrustes (visible 짝)", "procr_vis"),
                          ("procrustes_fit_late", "Procrustes (late 짝)", "procr_late"), ("procrustes_fit_all", "Procrustes (전체 짝)", "procr_all")):
        for _g, _gl in (("vis", "visible"), ("late", "late")):
            LROWS.append((f"{_hk}_{_rs}_{_g}", f"{_hk} 머리 → p, {_vl} → {_gl}", "evv, cv = E(hd, X, Gv)", {"key": _key, "vn": _vn}, _g,
                          (_key, _vn, f"test_{_gl}", _gl), f"{_rid}_p_{_rs}" if _rs in _RID[_rid] else None))
for _t in range(8):
    LROWS.append((f"p_late_t{_t}", f"p late 머리, 미래 튜블릿 t{_t}", 'T["exp5b_tubelet"][src]["late"]', {"src": "p", "t": str(_t)}, "late",
                  ("exp5b_tubelet", "p", "late", _t, "test", "late"), f"p_late_head_t{_t}"))
LROW = {r[0]: r for r in LROWS}
# 한 줄 표에 쓰는 핵심 행
KEY_ROWS = ["p_vis_vis", "p_late_late", "p_vis_late", "p_late_vis", "z_raw_vis", "z_raw_late", "z_procr_vis_vis",
            "z_procr_vis_late", "z_procr_late_late", "z_procr_all_late", "h_raw_vis", "h_raw_late", "h_procr_vis_late"]


def load_perclip(tag, vid, root="probe"):
    d = np.load(OUTD / root / tag / "perclip.npz")
    assert (d["video_id"].astype(str) == vid).all(), tag
    return d["corr"], json.loads(str(d["recs"]))


def pick(corr, recs, tname, prefix, cond):
    ix = [i for i, r in enumerate(recs) if r["tname"] == tname and r["src"].startswith(prefix)
          and all(r.get("v_" + k) == v for k, v in cond.items())]
    assert ix, (tname, prefix, cond)
    c = corr[ix[0]]
    for i in ix[1:]:
        assert (corr[i] == c).all(), (tname, prefix, cond)
    return c


# ============================================================================= report — 부분공간 bootstrap (geometry 식 그대로 + block 재추출)
def wmeans(X, y, w, C):
    Y = np.eye(C)[y] * w[:, None]
    M = (Y.T @ X) / Y.sum(0)[:, None]
    return M - M.mean(0)


def subspace_boot(reps, meta, tr, extra_mask, nboot=200, seed=0, unit=None):
    """reps {name: resid (n,D)}. → {label|a-b|sub: (n_boot+1,) 배열}, 0 번째 = 재추출 없음 (= geometry xfit_captured_norm).
    재추출: train 절반 · test 절반 각각에서 block 을 복원추출 (절반이 섞이지 않는다) → 모든 predictor 에 같은 추출 (짝 비교)."""
    k = meta["sym_k"].astype(int); vis = k == 0
    blk = meta["block_id"].astype(str) if unit is None else unit          # 2026-09-25: (궤적, k) 단위 재추출 (unit)
    rng = np.random.default_rng(seed)
    out = {}
    pairs = [(a, b) for a in reps for b in reps if a < b]
    for lname, col, classes in (("shape", "shape_pre", None), ("color", "color_pre", COLOR_ORDER)):
        ys = meta[col].astype(str); classes = classes or sorted(set(ys)); C = len(classes); r = C - 1
        y = np.array([classes.index(v) for v in ys])
        for sub, sm in (("visible", vis), ("late", ~vis)):
            m = sm & extra_mask
            halves = {}
            for hf, mh in (("tr", m & tr), ("te", m & ~tr)):
                ii = np.where(mh)[0]; ub, inv = np.unique(blk[ii], return_inverse=True)
                halves[hf] = (ii, inv, len(ub))
            vals = {pq: [] for pq in pairs}
            Xh = {(nm, hf): np.ascontiguousarray(X[ii]) for nm, X in reps.items() for hf, (ii, _, _) in halves.items()}
            for b in range(nboot + 1):
                W = {}
                for hf, (ii, inv, nb) in halves.items():
                    W[hf] = np.ones(len(ii)) if b == 0 else np.bincount(rng.integers(0, nb, nb), minlength=nb)[inv].astype(float)
                H = {}
                for nm in reps:
                    for hf, (ii, inv, nb) in halves.items():
                        Mc = wmeans(Xh[nm, hf], y[ii], W[hf], C); H[nm, hf] = (Mc, basis(Mc, r))
                sh = {nm: (captured(H[nm, "te"][0], H[nm, "tr"][1]) + captured(H[nm, "tr"][0], H[nm, "te"][1])) / 2 for nm in reps}
                if b == 0:                                                     # split-half 천장 (정규화 분모) 도 남긴다
                    for nm in reps:
                        out[f"{lname}|ceil|{nm}|{sub}"] = np.array([sh[nm]])
                for a_, b_ in pairs:
                    cap = (captured(H[a_, "te"][0], H[b_, "tr"][1]) + captured(H[a_, "tr"][0], H[b_, "te"][1]) +
                           captured(H[b_, "te"][0], H[a_, "tr"][1]) + captured(H[b_, "tr"][0], H[a_, "te"][1])) / 4
                    vals[a_, b_].append(cap / np.sqrt(sh[a_] * sh[b_]))
            for (a_, b_), v in vals.items():
                out[f"{lname}|{a_}-{b_}|{sub}"] = np.array(v)
    return out


# ============================================================================= report
def fmt_ci(s, nd=1):
    return f"{s['acc']:.{nd}f} [{s['lo']:.{nd}f}, {s['hi']:.{nd}f}]"


def fmt_d(s, nd=1, mark=None):
    t = f"{s['d']:+.{nd}f} [{s['lo']:+.{nd}f}, {s['hi']:+.{nd}f}]"
    return t


def cmd_report(a):
    import matplotlib
    matplotlib.use("Agg")
    t0 = time.time()
    PR = a.probe_root
    runs = [t for t in RUNS if (OUTD / PR / t / "results.json").exists()]
    gruns = [t for t in RUNS if (OUTD / "geometry" / t / "results.json").exists()]
    rruns = [t for t in RUNS if (OUTD / "ridge" / f"ridge_{t}.json").exists()]
    assert "release" in runs and "release" in gruns and "release" in rruns, (runs, gruns, rruns)
    log(f"probe {runs}\n geometry {gruns}\n ridge {rruns}")
    meta = {k: v for k, v in np.load(REL / "meta.npz").items()}
    vid = meta["video_id"].astype(str)
    blk = traj_clusters(meta)          # 2026-09-25 정정: CI 단위 = (궤적, k) 묶음 (이전: block_id = 1–2 clip, 사실상 clip 단위)
    tr, v11te = splits(meta)
    te = ~tr
    k = meta["sym_k"].astype(int)
    G = {"vis": te & (k == 0), "late": te & (k >= 1)}
    SC = {"full": lambda m: m, "v11test": lambda m: m & v11te}
    PJ = {t: json.loads((OUTD / PR / t / "results.json").read_text()) for t in runs}
    GJ = {t: json.loads((OUTD / "geometry" / t / "results.json").read_text()) for t in gruns}
    RJ = {t: json.loads((OUTD / "ridge" / f"ridge_{t}.json").read_text()) for t in rruns}
    RC = {t: dict(np.load(OUTD / "ridge" / f"perclip_{t}.npz")) for t in rruns}
    J = dict(meta=dict(date=time.strftime("%Y-%m-%d %H:%M"), script="z_research/scripts/analysis/te_analyze_identity.py report",
                       probe_root=PR, probe_device={t: json.loads((OUTD / PR / t / "run_info.json").read_text())["device"] for t in runs},
                       probe_skip_tubelet={t: json.loads((OUTD / PR / t / "run_info.json").read_text()).get("skip_tubelet") for t in runs},
                       probe_head_cache={t: {k: v for k, v in json.loads((OUTD / PR / t / "run_info.json").read_text()).get("head_cache", {}).items()
                                             if k in ("hit", "miss")} for t in runs},
                       probe_elapsed_s={t: json.loads((OUTD / PR / t / "run_info.json").read_text())["elapsed_s"] for t in runs},
                       runs=runs, geometry_runs=gruns, ridge_runs=rruns, n_boot=NBOOT,
                       ci_unit="(trajectory, k) cluster; trajectory = identical meta.truth path",
                       n_clusters=dict(all=int(len(set(blk))), test_vis=int(len(set(blk[te & (k == 0)]))),
                                       test_late=int(len(set(blk[te & (k >= 1)]))),
                                       test_v11te_vis=int(len(set(blk[te & (k == 0) & v11te]))),
                                       test_v11te_late=int(len(set(blk[te & (k >= 1) & v11te]))))))

    # ---- 1. 재현 · 재사용 검사
    skip5b = bool(PJ["release"]["meta"].get("skip_tubelet"))
    REFP = json.loads(REF_PROBE.read_text())
    chk = dict(release_probe_vs_reference=full_equal(PJ["release"], REFP, drop=("exp5b_tubelet",) if skip5b else ()),
               release_probe_skip_tubelet=skip5b,
               release_geometry_vs_reference=full_equal(GJ["release"], json.loads(REF_GEOM.read_text())),
               release_ridge10=dict(got=RJ["release"]["ridge10"], ref=RIDGE10_REF, equal=RJ["release"]["ridge10"] == RIDGE10_REF),
               probe_reuse={t: reuse_check(PJ[t], PJ["release"], probe_pdep) for t in runs if t != "release"},
               geometry_reuse={t: reuse_check(GJ[t], GJ["release"], geom_pdep) for t in gruns if t != "release"})
    # 정본 재현 (GPU 실행, 있으면) + 비교 세트 release 와 정본의 정확도 차 (장치가 다르면 0 이 아닐 수 있다)
    gp = OUTD / "probe" / "release" / "results.json"
    if gp.exists():
        GPJ = json.loads(gp.read_text())
        chk["release_probe_gpu_vs_reference"] = full_equal(GPJ, REFP, drop=("exp5b_tubelet",) if GPJ["meta"].get("skip_tubelet") else ())
        ri = OUTD / "probe" / "release" / "run_info.json"
        chk["release_probe_gpu_device"] = json.loads(ri.read_text())["device"] if ri.exists() else GPJ["meta"].get("device")
    REFL = dict(leaves(REFP))
    accd = [abs(v - REFL.get(p, v)) for p, v in leaves(PJ["release"])
            if p and p[-1] == "acc" and "exp5b_tubelet" not in p and p[0] != "meta" and isinstance(v, (int, float))]
    chk["release_probe_vs_reference_acc"] = dict(n_acc=len(accd), max_abs_pt=max(accd) if accd else None,
                                                 n_nonzero=int(sum(x > 0 for x in accd)), mean_abs_pt=float(np.mean(accd)) if accd else None)
    # ridge 의 p 무관 행 (z_all · z_hid · zhead_zhid · h_fut) 도 clip 단위로 같아야
    chk["ridge_reuse"] = {t: all((RC[t][f"{tt}|{r}"] == RC["release"][f"{tt}|{r}"]).all()
                                 for tt in TARGETS for r in ("z_all", "z_hid", "zhead_zhid", "h_fut")) for t in rruns if t != "release"}
    prep = json.loads((OUTD / "prep" / "prep_checks.json").read_text()) if (OUTD / "prep" / "prep_checks.json").exists() else None
    chk["prep"] = prep
    J["checks"] = chk
    log("release probe == reference:", chk["release_probe_vs_reference"]["n_diff"] == 0,
        "· geometry:", chk["release_geometry_vs_reference"]["n_diff"] == 0, "· ridge10:", chk["release_ridge10"]["equal"])

    # ---- 2. logistic 행: clip 별 정오 → 전체 test (results.json 과 대조) · v11 test 부분 · 릴리즈 짝 차이
    PC = {t: load_perclip(t, vid, PR) for t in runs}
    LR = {}
    n_verified = 0
    LOG_ROWS = [r for r in LROWS if not r[0].startswith("p_late_t")]                  # (5b) 는 --skip-tubelet → ridge 로만
    for t in runs:
        LR[t] = {}
        for tn in TARGETS:
            LR[t][tn] = {}
            for rid, lab, pre, cond, g, jp, rrow in LOG_ROWS:
                c = pick(*PC[t], tn, pre, cond)
                ref = get(PJ[t][tn], jp)["acc"]
                got = round(100 * float(c[G[g]].mean()), 2)
                assert abs(got - ref) < 1e-6, (t, tn, rid, got, ref)          # 행 대응이 맞는지 results.json 으로 확인
                n_verified += 1
                o = {"c": c}
                for sc, f in SC.items():
                    m = f(G[g])
                    o[sc] = acc_ci(c, blk, m)
                if rrow is not None:
                    rc = RC[t][f"{tn}|{rrow}"]
                    o["ridge_c"] = rc
                    for sc, f in SC.items():
                        o["ridge_" + sc] = acc_ci(rc, blk, f(G[g]))
                LR[t][tn][rid] = o
    log(f"logistic 행 {n_verified} 개가 results.json 과 일치 (clip 별 기록 → 정확도)")
    # 최적화 잡음 바닥: 같은 release p 에 같은 코드를 GPU (정본 장치) 와 CPU 로 맞춘 머리의 정확도 차 (행마다, probe test 전체)
    NOISE = {tn: {} for tn in TARGETS}
    if gp.exists():
        for tn in TARGETS:
            for rid, lab, pre, cond, g, jp, rrow in LOG_ROWS:
                NOISE[tn][rid] = abs(get(PJ["release"][tn], jp)["acc"] - get(GPJ[tn], jp)["acc"])
    J["opt_noise"] = NOISE
    for t in runs:
        for tn in TARGETS:
            for rid, *_ in LOG_ROWS:
                o, o0 = LR[t][tn][rid], LR["release"][tn][rid]
                g = LROW[rid][4]
                nz = NOISE[tn].get(rid, 0.0)
                o["opt_noise"] = nz
                for sc, f in SC.items():
                    m = f(G[g])
                    o["d_" + sc] = paired_ci(o["c"], o0["c"], blk, m)
                    if "ridge_c" in o:
                        o["dr_" + sc] = paired_ci(o["ridge_c"], o0["ridge_c"], blk, m)
                        sl, sr = sign_of(o["d_" + sc]), sign_of(o["dr_" + sc])
                        # 방향 주장 = logistic · ridge 짝 CI 가 같은 방향으로 0 을 벗어나고, |Δ logistic| 이 최적화 잡음 바닥보다 크다
                        o["claim_" + sc] = sl if (sl == sr and abs(o["d_" + sc]["d"]) > nz) else 0
                    else:
                        o["claim_" + sc] = None
    J["logistic"] = {t: {tn: {rid: {kk: v for kk, v in o.items() if kk not in ("c", "ridge_c")} for rid, o in d.items()}
                         for tn, d in LR[t].items()} for t in runs}
    J["lambda"] = {t: {tn: dict(p_vis=PJ[t][tn]["exp5_p_vis_head"]["head"]["lam"], p_late=PJ[t][tn]["exp5_p_late_head"]["head"]["lam"],
                                p_all=PJ[t][tn]["exp6_ref"]["p_all_head"]["lam"]) for tn in TARGETS} for t in runs}
    J["probe_misc"] = {t: dict(
        procrustes_z={kk: round(v, 4) for kk, v in PJ[t]["shape"]["exp6"]["procrustes_fit"].items()},
        procrustes_h={kk: round(v, 4) for kk, v in PJ[t]["shape"]["exp6h"]["procrustes_fit"].items()},
        geometry={kk: round(v, 4) for kk, v in PJ[t]["shape"]["geometry"].items()},
        p_norm=PJ[t]["sanity"]["feature_norm_mean"]["p_fut"]) for t in runs}
    # ridge 재분할 (train = v11_postft 학습 절반, test = 안 본 절반)
    J["ridge_resplit"] = {t: RJ[t]["acc_v11resplit"] for t in rruns}
    RT = {}
    for t in rruns:
        RT[t] = {}
        for tn in TARGETS:
            RT[t][tn] = []
            for i in range(8):
                c, c0 = RC[t][f"{tn}|p_late_head_t{i}"], RC["release"][f"{tn}|p_late_head_t{i}"]
                RT[t][tn].append(dict(full=acc_ci(c, blk, G["late"]), v11test=acc_ci(c, blk, G["late"] & v11te),
                                      d_full=paired_ci(c, c0, blk, G["late"]), d_v11test=paired_ci(c, c0, blk, G["late"] & v11te)))
    J["ridge_tubelet"] = RT
    J["ref_logistic_tubelet"] = {tn: [d["test"]["late"]["acc"] for d in json.loads(REF_PROBE.read_text())[tn]["exp5b_tubelet"]["p"]["late"]]
                                 for tn in TARGETS}
    J["ridge10"] = {t: RJ[t]["ridge10"] for t in rruns}

    # ---- 3. 기하: results.json 행 + 부분공간 bootstrap (전체 · v11 test 만)
    GR = {}
    for t in gruns:
        g = GJ[t]; S = g["subspace"]["resid"]; Cl = g["class"]; Dm = g["domain"]
        o = {}
        for l in TARGETS:
            for s in ("visible", "late"):
                for a_, b_ in (("z_all", "p_fut"), ("h_fut", "p_fut"), ("h_ctx", "p_fut"), ("z_all", "h_fut"), ("z_all", "h_ctx"), ("h_ctx", "h_fut")):
                    o[f"share|{l}|{a_}-{b_}|{s}"] = S[f"{l}|{a_}:{s}|{b_}:{s}"]["xfit_captured_norm"]
                o[f"xfitperm95|{l}|z_all-p_fut|{s}"] = S[f"{l}|z_all:{s}|p_fut:{s}"]["xfit_perm"]["captured_p95"]
            o[f"share|{l}|p_vis-p_late"] = S[f"{l}|p_fut:visible|p_fut:late"]["xfit_captured_norm"]
            for r in ("z_all", "p_fut", "h_fut", "h_ctx"):
                o[f"resid_v2l|{l}|{r}"] = 100 * Cl["resid"][f"{l}|{r}|visible->late"]["linprobe"]
                o[f"resid_l2v|{l}|{r}"] = 100 * Cl["resid"][f"{l}|{r}|late->visible"]["linprobe"]
                o[f"raw_v2l|{l}|{r}"] = 100 * Cl["raw"][f"{l}|{r}|visible->late"]["linprobe"]
                o[f"fisher_resid|{l}|{r}|visible"] = Cl["resid"][f"{l}|{r}|visible"]["fisher"]
                o[f"fisher_resid|{l}|{r}|late"] = Cl["resid"][f"{l}|{r}|late"]["fisher"]
        for s in ("visible", "late"):
            nm = Dm["norms"][s]
            o[f"norm|p|{s}"] = nm["p_fut"]["mean_norm"]; o[f"norm|h_fut|{s}"] = nm["h_fut"]["mean_norm"]
            o[f"rms|p|{s}"] = nm["p_fut"]["rms_radius"]; o[f"rms_ratio|p/h_fut|{s}"] = nm["p_fut"]["rms_radius"] / nm["h_fut"]["rms_radius"]
            o[f"offset|h_fut-p|{s}"] = Dm["pairs"][f"h_fut|p_fut|{s}"]["offset_share_of_paired_sqdist"]
            o[f"offset|z_all-p|{s}"] = Dm["pairs"][f"z_all|p_fut|{s}"]["offset_share_of_paired_sqdist"]
            o[f"cka_resid|z_all-p|{s}"] = g["cka"]["resid"][s]["linear"]["z_all|p_fut"]
            o[f"cka_resid|p-h_fut|{s}"] = g["cka"]["resid"][s]["linear"]["p_fut|h_fut"]
        o["var|p|cell"] = g["variance"]["p_fut|all"]["cell"]; o["var|p|resid_shape"] = g["variance"]["p_fut|all"]["resid_shape"]
        GR[t] = o
    # 부분공간 bootstrap — p 만 predictor 마다 바뀐다
    log("부분공간 bootstrap …")
    zr, hr = np.load(REL / "z.npy").astype(np.float64), np.load(REL / "h.npy").astype(np.float64)
    base = dict(z_all=zr.mean(1), h_fut=hr[:, 8:].mean(1), h_ctx=hr[:, :8].mean(1))
    ptags = list(gruns) + [t for t in ("release_bs16",) if t not in gruns and (cache_of(t) / "p.npy").exists()]
    ptags = [t for t in RUNS if t in ptags]
    J["subspace_tags"] = ptags
    Pf = {t: np.load(cache_of(t) / "p.npy").astype(np.float64).mean(1) for t in ptags}
    hf_ = base["h_fut"]
    J["cos_pfut_hfut"] = {sc: {t: {g: float((np.sum(Pf[t][f(G[g])] * hf_[f(G[g])], 1) / np.linalg.norm(Pf[t][f(G[g])], axis=1)
                                            / np.linalg.norm(hf_[f(G[g])], axis=1)).mean()) for g in G} for t in ptags}
                          for sc, f in SC.items()}
    if gp.exists():                                                             # GPU 정본 release (CPU 재적합과 대조)
        J["gpu_release_acc"] = {tn: {rid: get(GPJ[tn], LROW[rid][5])["acc"] for rid in ("p_late_late", "p_vis_vis", "h_raw_vis", "z_raw_vis")}
                                for tn in TARGETS}
    SB = {}
    for scope, em, trm in (("full", np.ones(len(vid), bool), tr), ("v11test", v11te, tr)):
        cell_tr = trm & em
        reps = {kk: resid_view(v, meta, cell_tr) for kk, v in base.items()}
        reps.update({f"p__{t}": resid_view(Pf[t], meta, cell_tr) for t in ptags})
        SB[scope] = subspace_boot(reps, meta, trm, em, nboot=200, seed=0, unit=blk)
    # 대조: 재추출 0 번째 = geometry results.json (전체)
    gdev = 0.0
    for t in gruns:
        for l in TARGETS:
            for s in ("visible", "late"):
                for a_, key in (("z_all", "z_all-p_fut"), ("h_fut", "h_fut-p_fut"), ("h_ctx", "h_ctx-p_fut")):
                    mine = SB["full"][f"{l}|{min(a_, 'p__' + t)}-{max(a_, 'p__' + t)}|{s}"][0]
                    gdev = max(gdev, abs(mine - GR[t][f"share|{l}|{key}|{s}"]))
    chk["subspace_reimpl_max_abs_dev_vs_geometry"] = gdev
    log(f"부분공간 재구현 ↔ geometry results.json 최대 차 {gdev:.2e}")
    assert gdev < 1e-4, gdev
    SBR = {}
    for scope in SB:
        SBR[scope] = {}
        for t in ptags:
            SBR[scope][t] = {}
            for l in TARGETS:
                for s in ("visible", "late"):
                    for a_ in ("z_all", "h_fut", "h_ctx"):
                        kk = f"{l}|{min(a_, 'p__' + t)}-{max(a_, 'p__' + t)}|{s}"
                        k0 = f"{l}|{min(a_, 'p__release')}-{max(a_, 'p__release')}|{s}"
                        v, v0 = SB[scope][kk], SB[scope][k0]
                        d = v[1:] - v0[1:]; d0 = float(v[0] - v0[0])
                        # 2026-09-25 정정: 치우친 재추출 통계 → basic (반사) bootstrap 구간 [2θ − q97.5, 2θ − q2.5] (점추정이 항상 안에)
                        SBR[scope][t][f"{l}|{a_}-p|{s}"] = dict(
                            v=float(v[0]), lo=float(2 * v[0] - np.percentile(v[1:], 97.5)), hi=float(2 * v[0] - np.percentile(v[1:], 2.5)),
                            d=d0, dlo=float(2 * d0 - np.percentile(d, 97.5)), dhi=float(2 * d0 - np.percentile(d, 2.5)))
                    SBR[scope][t][f"{l}|ceil_p|{s}"] = float(SB[scope][f"{l}|ceil|p__{t}|{s}"][0])
                for s in ("visible", "late"):                                  # 참고: encoder 끼리 (predictor 무관)
                    kk = f"{l}|h_fut-z_all|{s}"
                    v0_ = float(SB[scope][kk][0])
                    SBR[scope][t][f"{l}|z_all-h_fut|{s}"] = dict(v=v0_, lo=2 * v0_ - float(np.percentile(SB[scope][kk][1:], 97.5)),
                                                                hi=2 * v0_ - float(np.percentile(SB[scope][kk][1:], 2.5)))
    J["geometry"] = GR
    J["subspace_boot"] = SBR

    # ---- 4. 학습 곡선
    CJ = json.loads((OUTD / "curves" / "curves.json").read_text()) if (OUTD / "curves" / "curves.json").exists() else None
    J["curves"] = CJ
    (OUTD / "identity.json").write_text(json.dumps(J, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    figs = make_figures(J, LR, runs, gruns, OUTD)
    write_md(J, runs, gruns, rruns, figs, time.time() - t0)
    log(f"→ {OUTD / 'IDENTITY.md'}  ({time.time()-t0:.0f}s)")


# ============================================================================= figures
C_BLUE, C_ORANGE, C_GREEN = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, DGRAY = "#8c8c8c", "#333333"
SHORT = {"release": "release", "release_bs16": "release\n(bs16)", "v11_e10": "v11\ne10", "ip1_e40": "ip1\ne40",
         "ip2_e80": "ip2\ne80", "pv1_e15": "pv1\ne15", "ariel_ep43": "ariel\nep43"}


def _mpl():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _plabel(ax, s, y=-0.22):
    ax.text(0.5, y, s, transform=ax.transAxes, ha="center", va="top", fontsize=10)


HEAT_P = [("p_vis_vis", "p vis head -> vis"), ("p_late_late", "p late head -> late"), ("p_vis_late", "p vis head -> late"),
          ("p_late_vis", "p late head -> vis"), ("z_raw_vis", "z head -> p raw, vis"), ("z_raw_late", "z head -> p raw, late"),
          ("z_procr_vis_vis", "z head -> p Procr(vis), vis"), ("z_procr_vis_late", "z head -> p Procr(vis), late"),
          ("z_procr_late_late", "z head -> p Procr(late), late"), ("h_raw_vis", "h head -> p raw, vis"),
          ("h_raw_late", "h head -> p raw, late"), ("h_procr_vis_late", "h head -> p Procr(vis), late")]


def make_figures(J, LR, runs, gruns, out):
    plt = _mpl()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("bgo", [C_BLUE, "#e6e6e6", C_ORANGE])
    others = [t for t in runs if t != "release"]
    figs = {}
    # ---- (1) Δ heatmap: logistic 행 (a) + 기하 (b)
    rows = [(tn, rid, f"{tn} | {lab}") for tn in TARGETS for rid, lab in HEAT_P]
    Dm = np.array([[LR[t][tn][rid]["d_full"]["d"] for t in others] for tn, rid, _ in rows])
    Cm = np.array([[LR[t][tn][rid]["claim_full"] or 0 for t in others] for tn, rid, _ in rows])
    gothers = [t for t in gruns if t != "release"]
    SBF = J["subspace_boot"]["full"]
    grows = []
    for l in TARGETS:
        for a_, an in (("z_all", "z"), ("h_fut", "h_fut")):
            for s in ("visible", "late"):
                grows.append((f"{l} | share {an}-p, {s} (x100)", [100 * SBF[t][f"{l}|{a_}-p|{s}"]["d"] for t in gothers],
                              [((SBF[t][f"{l}|{a_}-p|{s}"]["dlo"] > 0) - (SBF[t][f"{l}|{a_}-p|{s}"]["dhi"] < 0))
                               * (abs(SBF[t][f"{l}|{a_}-p|{s}"]["d"]) >= SUB_MIN) for t in gothers]))
        grows.append((f"{l} | resid vis->late probe p", [J["geometry"][t][f"resid_v2l|{l}|p_fut"] - J["geometry"]["release"][f"resid_v2l|{l}|p_fut"]
                                                          for t in gothers], [0] * len(gothers)))
    Gm = np.array([g[1] for g in grows]); Gc = np.array([g[2] for g in grows])
    fig, axes = plt.subplots(1, 2, figsize=(15, 9.5), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.9))
    for ax, M, Cc, labs, cols, lab in ((axes[0], Dm, Cm, [r[2] for r in rows], others, "(a) linear probe, delta vs release (pt)"),
                                        (axes[1], Gm, Gc, [g[0] for g in grows], gothers, "(b) class geometry, delta vs release")):
        v = max(1.0, float(np.nanmax(np.abs(M))))
        im = ax.imshow(M, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([SHORT[t] for t in cols], fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=7.5, color="#111111",
                        fontweight="bold" if Cc[i, j] != 0 else "normal")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        ax.spines[:].set_visible(False)
        _plabel(ax, lab, y=-0.08)
    fig.savefig(out / "fig_identity_delta.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    figs["delta"] = "fig_identity_delta.png"

    # ---- (2) 핵심 행 절대값 small multiples (CI)
    panels = [("p_late_late", "p late head -> late (%)"), ("p_vis_late", "p vis head -> late (%)"), ("z_raw_late", "z head -> p raw, late (%)"),
              ("z_procr_vis_late", "z head -> p Procr(vis), late (%)"), ("share_z", "class-mean share z-p, resid, visible"),
              ("share_h", "class-mean share h_fut-p, resid, late")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), gridspec_kw=dict(hspace=0.55, wspace=0.28))
    for pi, (ax, (key, ylab)) in enumerate(zip(axes.flat, panels)):
        cols = runs if key.startswith(("p_", "z_", "h_")) else gruns
        x = np.arange(len(cols))
        for j, (tn, col, off) in enumerate((("shape", C_BLUE, -0.12), ("color", C_ORANGE, 0.12))):
            if key.startswith("share"):
                a_, s = ("z_all", "visible") if key == "share_z" else ("h_fut", "late")
                st = [SBF[t][f"{tn}|{a_}-p|{s}"] for t in cols]
                yv = np.array([d["v"] for d in st]); lo = np.array([d["lo"] for d in st]); hi = np.array([d["hi"] for d in st])
                ref = SBF[cols[0]][f"{tn}|z_all-h_fut|{s}"]["v"]
                ax.axhline(ref, color=col, ls=":", lw=1)
            else:
                st = [LR[t][tn][key]["full"] for t in cols]
                yv = np.array([d["acc"] for d in st]); lo = np.array([d["lo"] for d in st]); hi = np.array([d["hi"] for d in st])
            ax.errorbar(x + off, yv, yerr=[np.clip(yv - lo, 0, None), np.clip(hi - yv, 0, None)], fmt="o", ms=6, color=col, ecolor=col, elinewidth=1.2, capsize=2,
                        label=tn if pi == 0 else None)
            ax.axhline(yv[0], color=GRAY, ls="--", lw=0.9)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in cols], fontsize=8)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.axvspan(-0.5, 0.5 if "release_bs16" not in cols else 1.5, color="#f0f0f0", zorder=0)
        _plabel(ax, f"({'abcdef'[pi]})", y=-0.3)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(out / "fig_identity_rows.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    figs["rows"] = "fig_identity_rows.png"

    # ---- (3) 학습 곡선
    CJ = J.get("curves")
    if CJ:
        res = CJ["res"]
        fams = [("v11", "v11_e", True), ("pv1", "pv1_e", True), ("ip1", "ip1_e", True), ("ip2", "ip2_e", True), ("ariel", "ariel_ep", False)]
        metr = [("read", "ridge p late head -> late (%)"), ("v2l", "ridge p vis head -> late (%)"),
                ("procr", "ridge z head -> p Procr(vis), late (%)"), ("share", "class-mean share z-p, resid, visible")]

        def val(tag, m, tn):
            r = res[tag]
            if m == "read":
                return r["ridge"][tn]["p_late_head"]["late"]
            if m == "v2l":
                return r["ridge"][tn]["p_vis_head"]["late"]
            if m == "procr":
                return r["ridge"][tn]["zhead_p_procr_vis"]["late"]
            return r["subspace"][f"{tn}|z_all-p_fut|visible"]
        fig, axes = plt.subplots(len(metr), len(fams), figsize=(17, 12), sharey="row", gridspec_kw=dict(hspace=0.55, wspace=0.12))
        pl = iter("abcdefghijklmnopqrst")
        for i, (m, ylab) in enumerate(metr):
            for j, (fam, pre, post) in enumerate(fams):
                ax = axes[i, j]
                tags = [t for t in CJ["tags"] if t.startswith(pre) and t in res]
                ep = [int(t[len(pre):]) for t in tags]
                for tn, col in (("shape", C_BLUE), ("color", C_ORANGE)):
                    y0 = val("release", m, tn)
                    ax.axhline(y0, color=GRAY, ls="--", lw=0.9)
                    xs, ys = ([0] + ep, [y0] + [val(t, m, tn) for t in tags]) if post else (ep, [val(t, m, tn) for t in tags])
                    ax.plot(xs, ys, "-o", color=col, ms=4, lw=1.6, label=tn if (i == 0 and j == 0) else None)
                if post:
                    ax.plot([0], [val("release", m, "shape")], "o", color=DGRAY, ms=5, zorder=5)
                    ax.plot([0], [val("release", m, "color")], "o", color=DGRAY, ms=5, zorder=5)
                ax.set_xlabel("epoch", fontsize=8)
                if j == 0:
                    ax.set_ylabel(ylab, fontsize=8)
                if i == 0:
                    ax.set_title(fam, fontsize=10)
                _plabel(ax, f"({next(pl)})", y=-0.32)
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.savefig(out / "fig_identity_curves.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        figs["curves"] = "fig_identity_curves.png"
    return figs


# ============================================================================= markdown
def write_md(J, runs, gruns, rruns, figs, elapsed):
    (OUTD / "IDENTITY.md").write_text(build_md(J, runs, gruns, rruns, figs, elapsed))


def _c(o, nd=1):
    return f"{o['acc']:.{nd}f}"


def _cell(o, sc="full", nd=1, ci=False, ref=False):
    """값 (Δ vs release). claim ±1 이면 굵게 (logistic · ridge 가 같은 방향으로 CI 가 0 을 벗어남). ref = release 칸 (값만)."""
    v = f"{o[sc]['acc']:.{nd}f}"
    d = o.get("d_" + sc)
    if d is None or ref:
        return v
    dd = f"{d['d']:+.{nd}f}" + (f" [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]" if ci else "")
    cl = o.get("claim_" + sc)
    t = f"{v} ({dd})"
    return f"**{t}**" if cl else t


def md_tables(J, runs, gruns, rruns):
    L = J["logistic"]; out = []
    P = out.append
    others = [t for t in runs if t != "release"]
    hdr = "| 행 | " + " | ".join(LABEL[t] for t in runs) + " |\n|---|" + "---:|" * len(runs)
    hdrn = "| 행 | " + " | ".join(LABEL[t] for t in runs) + " | 최적화 잡음 (pt) |\n|---|" + "---:|" * (len(runs) + 1)
    nz = lambda tn, rid: f" {L['release'][tn][rid].get('opt_noise', 0):.1f} |"                               # noqa: E731
    # 표 1 읽기
    P("### 표 1. 읽기 — 조건 안 정체 (logistic probe, probe test 전체 4,716 clip; 우연 shape 14.3 · color 12.5)\n")
    P("셀 = 정확도 % (Δ vs release, pt). **굵게** = 방향 주장 조건 셋을 다 만족: logistic 과 ridge 의 짝 차이 95 % CI ((궤적, k) 묶음 bootstrap) 가 **같은 방향으로** 0 을 벗어나고, "
      "|Δ logistic| 이 그 행의 **최적화 잡음** (같은 release p 에 같은 코드를 GPU · CPU 로 맞춘 두 머리의 정확도 차) 보다 크다.\n")
    for tn in TARGETS:
        P(f"**{tn}**\n\n" + hdrn)
        for rid in ("p_vis_vis", "p_late_late"):
            P(f"| {LROW[rid][1]} | " + " | ".join(_cell(L[t][tn][rid], ref=t == "release") for t in runs) + " |" + nz(tn, rid))
        RT = J["ridge_tubelet"]
        P("| (ridge) late 머리, 미래 튜블릿 하나씩 t0–t7 최저 · 최고 | " + " | ".join(
            (f"{min(d['full']['acc'] for d in RT[t][tn]):.1f} · {max(d['full']['acc'] for d in RT[t][tn]):.1f}" if t in RT else "—")
            for t in runs) + " | — |")
        P("")
    # 표 1b 튜블릿별 (ridge)
    RT = J["ridge_tubelet"]; rt = [t for t in runs if t in RT]
    P("### 표 1b. 미래 튜블릿 하나씩 — late 머리 → late (ridge λ=10; probe test 전체). 셀 = 정확도 % (Δ vs release, pt; * = 짝 95 % CI 가 0 을 벗어남)\n")
    P("| predictor · 속성 | " + " | ".join(f"t{i}" for i in range(8)) + " |\n|---|" + "---:|" * 8)
    for tn in TARGETS:
        for t in rt:
            cells = []
            for d in RT[t][tn]:
                if t == "release":
                    cells.append(f"{d['full']['acc']:.1f}")
                else:
                    sig = "*" if (d["d_full"]["lo"] > 0 or d["d_full"]["hi"] < 0) else ""
                    cells.append(f"{d['full']['acc']:.1f} ({d['d_full']['d']:+.1f}{sig})")
            P(f"| {LABEL[t]} · {tn} | " + " | ".join(cells) + " |")
    P("")
    # 표 2 배치
    P("### 표 2. 배치 — 조건 사이 이식 · encoder 머리 → p (logistic, probe test 전체)\n")
    P("셀 규칙은 표 1 과 같다. raw = 머리를 그대로 · Procrustes = train clip (p, z) 짝으로 맞춘 직교 사상 (라벨 안 씀) 뒤. "
      "visible 짝으로 맞춘 사상을 late 에 거는 행 (`Procrustes (visible 짝) → late`) 이 질문 (iii) 이다.\n")
    for tn in TARGETS:
        P(f"**{tn}**\n\n" + hdrn)
        for rid in ("p_vis_late", "p_late_vis", "p_vis_late_restd", "z_raw_vis", "z_raw_late", "z_shift_vis", "z_shift_late",
                    "z_procr_vis_vis", "z_procr_vis_late", "z_procr_late_vis", "z_procr_late_late", "z_procr_all_vis", "z_procr_all_late",
                    "h_raw_vis", "h_raw_late", "h_procr_vis_vis", "h_procr_vis_late", "h_procr_late_late"):
            P(f"| {LROW[rid][1]} | " + " | ".join(_cell(L[t][tn][rid], ref=t == "release") for t in runs) + " |" + nz(tn, rid))
        P("")
    # 표 3 v11 test 부분집합
    n_sub = {g: L["release"]["shape"][r]["v11test"]["n"] for g, r in (("visible", "p_vis_vis"), ("late", "p_late_late"))}
    P(f"### 표 3. v11_split test 부분집합 (probe test ∩ `index_test.csv` — visible {n_sub['visible']} · late {n_sub['late']} clip) — Δ vs release [95 % CI]\n")
    P("v11_postft 가 학습에 쓰지 않은 block 만. 셀 = logistic Δ [CI] / ridge Δ [CI]. **굵게** = 둘 다 같은 방향으로 0 을 벗어남.\n")
    h3 = "| 행 | release 값 (logistic / ridge) | " + " | ".join(LABEL[t] for t in others) + " |\n|---|---:|" + "---:|" * len(others)
    for tn in TARGETS:
        P(f"**{tn}**\n\n" + h3)
        for rid in KEY_ROWS:
            o0 = L["release"][tn][rid]
            rv = f"{o0['v11test']['acc']:.1f} / {o0['ridge_v11test']['acc']:.1f}" if "ridge_v11test" in o0 else f"{o0['v11test']['acc']:.1f} / —"
            cells = []
            for t in others:
                o = L[t][tn][rid]; d = o["d_v11test"]
                c = f"{d['d']:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}]"
                if "dr_v11test" in o:
                    r = o["dr_v11test"]; c += f" / {r['d']:+.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}]"
                cells.append(f"**{c}**" if o.get("claim_v11test") else c)
            P(f"| {LROW[rid][1]} | {rv} | " + " | ".join(cells) + " |")
        P("")
    # 표 4 기하
    SB = J["subspace_boot"]; GR = J["geometry"]
    go = [t for t in gruns]
    so = J["subspace_tags"]
    P("### 표 4. 기하 — 클래스 평균 부분공간 공유 몫 (resid, 교차 적합 / split-half 천장) · resid 이식 · 노름\n")
    P("공유 몫 셀 = 값 [basic bootstrap 95 % CI] (Δ vs release [CI]); bootstrap 200 회, train · test 절반 안에서 (궤적, k) 묶음 복원추출, 모든 predictor 에 같은 추출 (짝). "
      f"**굵게** = 짝 CI 가 0 을 벗어나고 |Δ| ≥ {SUB_MIN}. "
      "값은 geometry results.json 과 같다 (재구현 최대 차는 표 7). encoder 끼리 (z–h_fut) 는 predictor 와 무관한 기준.\n")
    hg = "| 행 | " + " | ".join(LABEL[t] for t in go) + " |\n|---|" + "---:|" * len(go)
    hs = "| 행 | " + " | ".join(LABEL[t] for t in so) + " |\n|---|" + "---:|" * len(so)
    for scope, stitle in (("full", "9,408 clip 전체"), ("v11test", "v11_split test clip 만 (4,704)")):
        P(f"**{stitle}**\n\n" + hs)
        for tn in TARGETS:
            for a_, an in (("z_all", "z"), ("h_fut", "h_fut"), ("h_ctx", "h_ctx")):
                for s in ("visible", "late"):
                    cells = []
                    for t in so:
                        o = SB[scope][t][f"{tn}|{a_}-p|{s}"]
                        c = f"{o['v']:.3f} [{o['lo']:.3f}, {o['hi']:.3f}]"
                        if t != "release":
                            sig = ((o["dlo"] > 0) - (o["dhi"] < 0)) if abs(o["d"]) >= SUB_MIN else 0     # 2026-09-25: |Δ| < 0.01 는 굵게 안 함
                            c += f" ({o['d']:+.3f} [{o['dlo']:+.3f}, {o['dhi']:+.3f}])"
                            c = f"**{c}**" if sig else c
                        cells.append(c)
                    P(f"| {tn} · {an}–p · {s} | " + " | ".join(cells) + " |")
            for s in ("visible", "late"):
                o = SB[scope]["release"][f"{tn}|z_all-h_fut|{s}"]
                P(f"| {tn} · (기준) z–h_fut · {s} | " + " | ".join(f"{o['v']:.3f}" for _ in so) + " |")
                P(f"| {tn} · p 의 split-half 천장 (정규화 분모) · {s} | " + " | ".join(f"{SB[scope][t][f'{tn}|ceil_p|{s}']:.3f}" for t in so) + " |")
        P("")
    P("**resid 이식 · 노름 · 이동 몫** (geometry results.json 그대로; CI 없음)\n\n" + hg)
    rows = []
    for tn in TARGETS:
        rows += [(f"{tn} · resid visible→late 선형 probe, p (%)", f"resid_v2l|{tn}|p_fut", 1),
                 (f"{tn} · 〃 z_all · h_fut (기준)", None, (tn,)),
                 (f"{tn} · resid late→visible 선형 probe, p (%)", f"resid_l2v|{tn}|p_fut", 1),
                 (f"{tn} · raw visible→late 선형 probe, p (%)", f"raw_v2l|{tn}|p_fut", 1),
                 (f"{tn} · 같은 p 의 visible–late 공유 몫", f"share|{tn}|p_vis-p_late", 3)]
    rows += [("p 평균 노름 visible / late", ("norm|p|visible", "norm|p|late"), 2),
             ("RMS 반경 비 p / h_fut, visible / late", ("rms_ratio|p/h_fut|visible", "rms_ratio|p/h_fut|late"), 3),
             ("h_fut–p 짝 제곱거리 중 중심 이동 몫, visible / late", ("offset|h_fut-p|visible", "offset|h_fut-p|late"), 3),
             ("z_all–p 〃, visible / late", ("offset|z_all-p|visible", "offset|z_all-p|late"), 3),
             ("CKA resid z_all–p, visible / late", ("cka_resid|z_all-p|visible", "cka_resid|z_all-p|late"), 3),
             ("CKA resid p–h_fut, visible / late", ("cka_resid|p-h_fut|visible", "cka_resid|p-h_fut|late"), 3),
             ("장면 칸이 설명하는 p 분산 몫", "var|p|cell", 3)]
    for lab, key, nd in rows:
        if key is None:
            tn = nd[0]
            P(f"| {lab} | " + " | ".join(f"{GR[t][f'resid_v2l|{tn}|z_all']:.1f} · {GR[t][f'resid_v2l|{tn}|h_fut']:.1f}" for t in go) + " |")
        elif isinstance(key, tuple):
            P(f"| {lab} | " + " | ".join(" / ".join(f"{GR[t][k_]:.{nd}f}" for k_ in key) for t in go) + " |")
        else:
            P(f"| {lab} | " + " | ".join(f"{GR[t][key]:.{nd}f}" for t in go) + " |")
    P("")
    # 표 5 Procrustes 잔차 · cos
    PM = J["probe_misc"]
    P("### 표 5. Procrustes 잔차 · 코사인 (probe results.json)\n\n" + hdr)
    for lab, fn in (("z 사상 (visible 짝) 상대 잔차: train visible", lambda t: PM[t]["procrustes_z"]["rel_residual_train_visible"]),
                    ("〃 test visible", lambda t: PM[t]["procrustes_z"]["rel_residual_test_visible"]),
                    ("〃 test late", lambda t: PM[t]["procrustes_z"]["rel_residual_test_late"]),
                    ("z 사상 (late 짝) → test late", lambda t: PM[t]["procrustes_z"]["procrustes_fit_late_rel_residual_test_late"]),
                    ("h_fut 사상 (visible 짝) → test visible · late",
                     lambda t: f"{PM[t]['procrustes_h']['rel_residual_test_visible']:.3f} · {PM[t]['procrustes_h']['rel_residual_test_late']:.3f}"),
                    ("cos(p_fut, h_fut) test visible · late",
                     lambda t: f"{PM[t]['geometry']['cos_pfut_hfut_test_visible']:.3f} · {PM[t]['geometry']['cos_pfut_hfut_test_late']:.3f}"),
                    ("cos(p_fut, z_all) test visible · late",
                     lambda t: f"{PM[t]['geometry']['cos_pfut_zall_test_visible']:.3f} · {PM[t]['geometry']['cos_pfut_zall_test_late']:.3f}"),
                    ("‖p_fut‖ 평균 (전체)", lambda t: PM[t]["p_norm"])):
        P(f"| {lab} | " + " | ".join(v if isinstance(v, str) else f"{v:.3f}" for v in (fn(t) for t in runs)) + " |")
    P("")
    # 표 6 ridge
    P("### 표 6. ridge 재검사 (닫힌 해, λ=10) — PROBE.md §8 의 10 칸 + 재분할\n")
    names = ["(2) z→z late", "(3) z_hid", "(5) p late→late", "(5) p vis→late", "(4) z→z_hid", "(6a) z→p raw vis",
             "(6c) Procr(vis) vis", "(6c) Procr(vis) late", "(6c) Procr(late) late", "(6h) h→p raw vis"]
    hr = "| 칸 | " + " | ".join(LABEL[t] for t in rruns) + " |\n|---|" + "---:|" * len(rruns)
    for tn in TARGETS:
        P(f"**{tn}** (probe split, test 전체)\n\n" + hr)
        for i, nm in enumerate(names):
            P(f"| {nm} | " + " | ".join(f"{J['ridge10'][t][tn][i]:.1f}" for t in rruns) + " |")
        P("")
    RS = J["ridge_resplit"]
    P("**재분할** — train = v11_postft 가 학습한 block 절반 (`index_train.csv`), test = 안 본 절반 (`index_test.csv`); 머리·Procrustes 모두 train 절반으로 맞춤\n\n" + hr)
    for tn in TARGETS:
        for r, g, lab in (("p_late_head", "late", "p late→late"), ("p_vis_head", "late", "p vis→late"), ("zhead_p_raw", "late", "z→p raw late"),
                          ("zhead_p_procr_vis", "vis", "Procr(vis) vis"), ("zhead_p_procr_vis", "late", "Procr(vis) late"),
                          ("zhead_p_procr_late", "late", "Procr(late) late")):
            P(f"| {tn} · {lab} | " + " | ".join(f"{RS[t][tn][r][g]:.1f}" for t in rruns) + " |")
    P("")
    # 표 7 검사
    C = J["checks"]
    P("### 표 7. 재현 · 재사용 검사\n")
    if C.get("release_probe_gpu_vs_reference"):
        g_ = C["release_probe_gpu_vs_reference"]
        P(f"- **GPU** ({C.get('release_probe_gpu_device')}) release probe ↔ 정본 (`RollOutV3/figures/v11_probe/results.json`, (5b) 제외): 공통 잎 {g_['n_shared']} 개 중 값이 다른 것 "
          f"**{g_['n_value_diff']}**, 정본에만 있는 잎 {g_['n_only_ref']} (새 실행에만 있는 잎 {g_['n_only_new']} = 정본 이후 스크립트에 더해진 (5c) · (6e) 대조)")
    c_ = C["release_probe_vs_reference"]; ra = C["release_probe_vs_reference_acc"]
    P(f"- **CPU** release probe (비교 세트) ↔ 정본: 공통 잎 {c_['n_shared']} 개 중 값이 다른 것 {c_['n_value_diff']} (L-BFGS 수치 경로 차). "
      f"정확도 칸 {ra['n_acc']} 개 중 {ra['n_nonzero']} 개가 다르고 최대 {ra['max_abs_pt']:.2f} pt · 평균 {ra['mean_abs_pt']:.3f} pt → 행별 '최적화 잡음' 으로 방향 주장 조건에 넣었다")
    P(f"- release geometry results.json ↔ 정본: 잎 {C['release_geometry_vs_reference']['n_leaves']} 개 중 다른 것 **{C['release_geometry_vs_reference']['n_diff']}**")
    P(f"- release ridge 10 칸 × 2 ↔ PROBE.md §8: 일치 = **{C['release_ridge10']['equal']}**")
    P(f"- 부분공간 재구현 (이 스크립트) ↔ geometry `xfit_captured_norm`: 최대 차 {C['subspace_reimpl_max_abs_dev_vs_geometry']:.1e}")
    if C.get("prep"):
        pp = C["prep"]
        P(f"- prep: 추출 pool_z ↔ 릴리즈 z 상대 L2 {pp['z_rel_l2']:.1e} · pool_h ↔ h {pp['h_rel_l2']:.1e} (같은 encoder 경로). "
          f"release pool_p ↔ 릴리즈 p 상대 L2 {pp['tags']['release']['p_rel_l2_vs_release']:.1e}, 최소 cos {pp['tags']['release']['p_min_cos_vs_release']:.6f} (배치 16 vs 32 의 bf16 잡음)")
        P("- prep: predictor p ↔ 릴리즈 p 상대 L2 — " + " · ".join(f"{t} {v['p_rel_l2_vs_release']:.3f}" for t, v in pp["tags"].items()))
    P("\n| predictor | probe: p 무관 잎 (같음 / 전체) | p 의존 잎 | 최대 차 | geometry: p 무관 잎 (같음 / 전체) | p 의존 잎 | ridge p 무관 행 clip 단위 같음 |\n|---|---:|---:|---:|---:|---:|---:|")
    for t in runs:
        if t == "release":
            continue
        a_ = C["probe_reuse"][t]; b_ = C["geometry_reuse"].get(t, {})
        P(f"| {LABEL[t]} | {a_['n_identical']} / {a_['n_p_independent']} | {a_['n_p_dependent']} | {a_['max_abs_diff']:.2g} | "
          f"{b_.get('n_identical', '—')} / {b_.get('n_p_independent', '—')} | {b_.get('n_p_dependent', '—')} | {C['ridge_reuse'].get(t, '—')} |")
    P("")
    # 표 8 곡선
    CJ = J.get("curves")
    if CJ:
        res = CJ["res"]
        P(f"### 표 8. 학습 곡선 (ridge λ=10 · 부분공간 공유 몫, probe split, n = {CJ['n_clip']} clip; CPU)\n")
        P("| tag | p late→late (s / c) | p vis→late (s / c) | z→p raw late (s / c) | Procr(vis) late (s / c) | 공유 몫 z–p vis (s / c) | 공유 몫 h_fut–p late (s / c) | cos(p,h_fut) late |\n|---|---:|---:|---:|---:|---:|---:|---:|")
        for t in CJ["tags"]:
            r = res[t]; R = r["ridge"]; S = r["subspace"]
            f2 = lambda row, g: f"{R['shape'][row][g]:.1f} / {R['color'][row][g]:.1f}"
            P(f"| {t} | {f2('p_late_head', 'late')} | {f2('p_vis_head', 'late')} | {f2('zhead_p_raw', 'late')} | {f2('zhead_p_procr_vis', 'late')} | "
              f"{S['shape|z_all-p_fut|visible']:.3f} / {S['color|z_all-p_fut|visible']:.3f} | {S['shape|h_fut-p_fut|late']:.3f} / {S['color|h_fut-p_fut|late']:.3f} | "
              f"{r['cos_pfut_hfut']['late']:.3f} |")
        P("")
    # 표 9 λ
    P("### 표 9. logistic λ 선택 (block hold-out; 후보 1e-4 … 1e-1)\n\n| predictor | shape p vis / late / 전체 | color p vis / late / 전체 |\n|---|---|---|")
    for t in runs:
        lm = J["lambda"][t]
        P(f"| {LABEL[t]} | " + " | ".join(f"{lm[tn]['p_vis']:g} / {lm[tn]['p_late']:g} / {lm[tn]['p_all']:g}" for tn in TARGETS) + " |")
    P("")
    return "\n".join(out)


TRAINED = ["v11_e10", "ip1_e40", "ip2_e80", "pv1_e15", "ariel_ep43"]
SHORTL = {"release": "release", "release_bs16": "release(bs16)", "v11_e10": "v11", "ip1_e40": "ip1", "ip2_e80": "ip2",
          "pv1_e15": "pv1", "ariel_ep43": "ariel"}


def _rng(vals, nd=1):
    vals = list(vals)
    return f"{min(vals):.{nd}f}–{max(vals):.{nd}f}" if vals else "—"


def _per(J, runs, fn, nd=1, tags=None):
    tags = [t for t in (tags or TRAINED) if t in runs]
    return " · ".join(f"{SHORTL[t]} {fn(t):.{nd}f}" for t in tags)


def md_correction(J, runs, gruns):
    """2026-09-25 적대 검증 (`../Archive/verify_raw/verify_identity.json`) 반영 목록. 이전 문장 · 수치는 고정 문자열 (이전 문서),
    지금 수치는 이번 실행 J 에서."""
    L = J["logistic"]; SBt = J["subspace_boot"]["v11test"]; CS = J["cos_pfut_hfut"]["v11test"]
    acc = lambda t, tn, r: L[t][tn][r]["v11test"]["acc"]                       # noqa: E731
    dv = lambda t, tn, r, k_="d_v11test": L[t][tn][r][k_]                       # noqa: E731
    fd = lambda o: f"{o['d']:+.1f} [{o['lo']:+.1f}, {o['hi']:+.1f}]"            # noqa: E731
    tr_ = [t for t in TRAINED if t in runs]
    rl = {tn: _rng([acc(t, tn, "p_late_late") for t in tr_]) for tn in TARGETS}
    nc = J["meta"]["n_clusters"]; gpu = J.get("gpu_release_acc") or {}
    lost = [f"{SHORTL[t]} {tn} {LROW[r][1]} {fd(dv(t, tn, r))}" for t in tr_ for tn in TARGETS for r in
            ("z_procr_vis_late", "h_raw_vis") if not L[t][tn][r].get("claim_v11test") and abs(dv(t, tn, r)["d"]) >= 2]
    it = [
        f"**범위** — 이전: 한 줄 수치를 probe test 전체 (4,716 clip; 절반이 v11_postft 학습 block) 에서 인용 → 지금: **v11_split test block** 부분집합에서 인용. "
        f"(i) late 머리 → late 이전 shape 96.5–97.7 · color 94.6–97.8 % → 지금 shape {rl['shape']} · color {rl['color']} % "
        f"(release {acc('release', 'shape', 'p_late_late'):.1f} / {acc('release', 'color', 'p_late_late'):.1f}).",
        f"**CI 단위** — 이전: block bootstrap (block = 1–2 clip, 사실상 clip 단위) 에 '짝 비교는 괜찮고 절대 CI 폭만 과소' → **철회**. 지금: (궤적, k) 묶음 bootstrap "
        f"(궤적 = 같은 `truth` 경로; 묶음 전체 {nc['all']}, test∩v11test visible {nc['test_v11te_vis']} · late {nc['test_v11te_late']}). 굵게 판정을 전부 다시 했다. "
        "검증자는 궤적 16 · 묶음 40 으로 셌다 — 정의가 다르다 (여기 궤적은 물체 위치 경로만, 조건 이름은 안 씀). "
        + ("v11test 에서 |Δ| ≥ 2 pt 인데 굵게가 아닌 칸: " + "; ".join(lost) + "." if lost else ""),
        f"**v11 late→late** — 이전: 'v11 은 late→late 가 두 속성 모두 조금 오른다 (굵게, shape +0.8)' → 지금: shape {fd(dv('v11_e10', 'shape', 'p_late_late'))} · "
        f"color {fd(dv('v11_e10', 'color', 'p_late_late'))} (v11test)"
        + (f"; GPU 정본 release (shape {gpu['shape']['p_late_late']:.1f}) 대비는 {L['v11_e10']['shape']['p_late_late']['full']['acc'] - gpu['shape']['p_late_late']:+.1f} "
           f"= 최적화 잡음 수준. 정체 읽기 개선이 아니다." if gpu else "."),
        "**release 열** — 비교 세트의 release 는 CPU L-BFGS 재적합이다 (정본 PROBE.md 는 GPU). "
        + (f"GPU 정본: p late→late shape {gpu['shape']['p_late_late']:.1f} (CPU {L['release']['shape']['p_late_late']['full']['acc']:.1f}), "
           f"h 머리 → p raw visible shape {gpu['shape']['h_raw_vis']:.1f} (CPU {L['release']['shape']['h_raw_vis']['full']['acc']:.1f})." if gpu else ""),
        f"**'배치는 shape 만'** — 이전: 'color 는 어디서도 안 오른다' → **철회**. 지금: color 는 z–p 공유 몫이 약간 내려가지만 h_fut–p (학습 표적) 로는 "
        f"v11 {SBt['v11_e10']['color|h_fut-p|visible']['d']:+.3f} / {SBt['v11_e10']['color|h_fut-p|late']['d']:+.3f} · pv1 {SBt['pv1_e15']['color|h_fut-p|visible']['d']:+.3f} / "
        f"{SBt['pv1_e15']['color|h_fut-p|late']['d']:+.3f} (visible / late, v11test) 로 약간 오른다; p 의 color split-half 천장 "
        f"{_rng([SBt[t][f'color|ceil_p|{s_}'] for t in ['release'] + tr_ if t in SBt for s_ in ('visible', 'late')], 2)} 라 노이즈가 크다.",
        f"**raw 머리 이식** — 이전: '(shape 만) raw 머리 이식도 같은 쪽이다' → 지금: h_fut 머리 → p raw (visible) 는 **두 속성 모두** 오른다 "
        f"(v11 shape {fd(dv('v11_e10', 'shape', 'h_raw_vis'))} · color {fd(dv('v11_e10', 'color', 'h_raw_vis'))}; pv1 shape {fd(dv('pv1_e15', 'shape', 'h_raw_vis'))} · "
        f"color {fd(dv('pv1_e15', 'color', 'h_raw_vis'))}). shape/color 차이는 resid z–p 부분공간 지표에서만 보인다.",
        f"**도메인** — 이전: '합성 물리 post-FT 에서만 · IntPhys 계열에서는 안 보인다' → 지금: 같은 생성기 학습 (v11 · pv1) 에서 크고, ip1 은 일부 "
        f"(cos late {CS['release']['late']:.3f} → {CS['ip1_e40']['late']:.3f}; h raw vis shape logistic {fd(dv('ip1_e40', 'shape', 'h_raw_vis'))} · ridge "
        f"{fd(dv('ip1_e40', 'shape', 'h_raw_vis', 'dr_v11test'))}), ip2 · ariel 은 거의 없다.",
        f"**(iii) 정렬** — 이전: '어떤 학습도 late p 를 visible p 자리에 놓게 만들지 않았다' (범위만) → 지금: 격차는 남지만 ip2 가 부분적으로 줄인다 "
        f"(Procrustes (visible 짝) → late shape logistic {fd(dv('ip2_e80', 'shape', 'z_procr_vis_late'))} · ridge {fd(dv('ip2_e80', 'shape', 'z_procr_vis_late', 'dr_v11test'))}); "
        f"pv1 · ariel 은 logistic 과 ridge 의 부호가 갈린다 → 이식 크기는 방향 이상으로 읽지 않는다.",
        f"**부분공간 CI** — 이전: 절반 안 block 재추출 percentile 구간 (점추정이 자기 CI 밖인 칸이 있었다) → 지금: 절반 안 (궤적, k) 묶음 재추출 + basic (반사) 구간, "
        f"|Δ| < {SUB_MIN} 는 굵게 안 함. 이전 굵게였던 ariel ±0.002–0.005 같은 차이는 주장하지 않는다.",
        "**미적용** — (a) 부분공간 bootstrap 에서 절반 분할 자체를 다시 뽑기 (분할 고정, 묶음만 재추출); "
        "(b) logistic probe 머리를 v11_split train 절반만으로 다시 맞추기 (probe 재실행 필요 — ridge 재분할 표 6 만 있다); "
        "(c) 다중 비교 보정; (d) (5b) 튜블릿별 logistic (GPU 시간; ridge 로만); (e) 학습 seed 반복 (재학습 필요).",
    ]
    return ("> ⚠️ **정정 (2026-09-25, 적대 검증 반영)** — 무엇이 바뀌었나 (이전 문장 → 지금 문장, 수치는 이번 재실행). "
            "검증 원문: `../Archive/verify_raw/verify_identity.json`.\n>\n" + "\n".join(f"> - {x}" for x in it) + "\n")


def md_head(J, runs, gruns, figs):
    L = J["logistic"]; SB = J["subspace_boot"]["full"]; GR = J["geometry"]
    tr_ = [t for t in TRAINED if t in runs]; tg = [t for t in TRAINED if t in gruns]
    acc = lambda t, tn, r, sc="full": L[t][tn][r][sc]["acc"]
    out = []
    P = out.append
    P("# v11 정체 (읽기) · 배치 — 학습한 predictor 별 (2026-09-25)\n")
    P(md_correction(J, runs, gruns))
    P("> 그룹 A (encoder = frozen 릴리즈 ViT-H, predictor 만 다르다). 상위 [`../README.md`](../README.md) · 모델 [`../MODELS.md`](../MODELS.md) · "
      "원 분석 [`RollOutV3/figures/v11_probe/PROBE.md`](../../RollOutV3/figures/v11_probe/PROBE.md) · "
      "[`v11_geometry/GEOMETRY.md`](../../RollOutV3/figures/v11_geometry/GEOMETRY.md).\n"
      "> **이 문서의 수치는 전부 `te_analyze_identity.py report` 가 산출물 (probe · geometry `results.json`, clip 별 정오 `perclip.npz`, "
      "ridge · 곡선 JSON, 풀링 특징 배열) 에서 다시 계산해 썼다.** 손으로 옮긴 숫자는 없다. 전체 값은 `identity.json`.\n")
    P("## 한 줄\n")
    sc = "v11test"                                                              # 2026-09-25: 한 줄 수치는 v11_split test 부분집합
    SBt = J["subspace_boot"]["v11test"]; CS = J["cos_pfut_hfut"]["v11test"]
    dd = lambda t, tn, r, k_="d_v11test": L[t][tn][r][k_]                       # noqa: E731
    fd = lambda o: f"{o['d']:+.1f} [{o['lo']:+.1f}, {o['hi']:+.1f}]"            # noqa: E731
    rl = {tn: [acc(t, tn, "p_late_late", sc) for t in tr_] for tn in TARGETS}
    r0 = {tn: acc("release", tn, "p_late_late", sc) for tn in TARGETS}
    P("**(i) p 의 조건 안 정체 decodability 는 천장 근처로 남는다 — (ii) v11 과 같은 생성기로 학습한 두 predictor (v11_postft · Predictor_v1_postft) 만 "
      "p 를 절대 좌표에서 h_fut 쪽으로 크게 옮긴다 (shape · color 둘 다; 클래스 평균 부분공간이 encoder 쪽으로 뚜렷이 가는 건 shape) — 학습 도메인 안의 결과이고 "
      "IntPhys1 은 일부, IntPhys2 · ariel 은 거의 안 움직인다 — (iii) visible 짝으로 맞춘 정렬은 여전히 late 에 안 통한다 (ip2 만 부분 개선).**\n")
    P(f"- 범위: 아래 수치는 전부 **v11_split test block** (v11_postft 가 학습에 쓰지 않은 절반; probe test ∩ `index_test.csv`) 이고, "
      f"짝 95 % CI 는 **(궤적, k) 묶음** bootstrap 이다 (test∩v11test 묶음 visible {J['meta']['n_clusters']['test_v11te_vis']} · late {J['meta']['n_clusters']['test_v11te_late']}). "
      "probe 머리는 probe split train 절반으로 맞췄다 (v11_postft 학습 clip 이 섞인다 — 단서).")
    gpu = J.get("gpu_release_acc")
    P(f"- **(i) 읽기** — 학습한 predictor {len(tr_)} 개 모두 predictor 마다 다시 맞춘 late 머리 → late clip "
      f"shape {_rng(rl['shape'])} % · color {_rng(rl['color'])} % (release {r0['shape']:.1f} / {r0['color']:.1f}, 우연 14.3 / 12.5). "
      f"v11 의 shape {dd('v11_e10', 'shape', 'p_late_late')['d']:+.1f} [{dd('v11_e10', 'shape', 'p_late_late')['lo']:+.1f}, {dd('v11_e10', 'shape', 'p_late_late')['hi']:+.1f}] pt 는 "
      + (f"GPU 정본 release (probe test 전체 {gpu['shape']['p_late_late']:.1f}) 대비로는 {acc('v11_e10', 'shape', 'p_late_late') - gpu['shape']['p_late_late']:+.1f} 로 "
         f"그 행의 CPU/GPU 최적화 잡음 ({L['release']['shape']['p_late_late'].get('opt_noise', 0):.1f}) 수준이다 — " if gpu else "")
      + "정체 읽기 개선이라고 쓰지 않는다. 천장 행이라 predictor 순위도 못 매긴다.")
    if tg:
        zs = {tn: {t: SBt[t][f"{tn}|z_all-p|visible"] for t in ["release"] + tg} for tn in TARGETS}
        enc = SBt["release"]["shape|z_all-h_fut|visible"]["v"]
        czd = " · ".join(f"{SHORTL[t]} {zs['color'][t]['d']:+.3f}" for t in tg)
        P(f"- **(ii) 절대 좌표 이동 (두 속성 모두)** — cos(p_fut, h_fut) late release {CS['release']['late']:.3f} → v11 {CS['v11_e10']['late']:.3f} · pv1 {CS['pv1_e15']['late']:.3f} "
          f"(ip1 {CS['ip1_e40']['late']:.3f} · ip2 {CS['ip2_e80']['late']:.3f} · ariel {CS['ariel_ep43']['late']:.3f}). 고정 h_fut 머리를 p 에 그대로 걸면 (visible, logistic) "
          f"shape {acc('release', 'shape', 'h_raw_vis', sc):.1f} → v11 {acc('v11_e10', 'shape', 'h_raw_vis', sc):.1f} · pv1 {acc('pv1_e15', 'shape', 'h_raw_vis', sc):.1f} %, "
          f"color {acc('release', 'color', 'h_raw_vis', sc):.1f} → v11 {acc('v11_e10', 'color', 'h_raw_vis', sc):.1f} · pv1 {acc('pv1_e15', 'color', 'h_raw_vis', sc):.1f} % "
          f"(Δ v11 shape {fd(dd('v11_e10', 'shape', 'h_raw_vis'))} · color {fd(dd('v11_e10', 'color', 'h_raw_vis'))}; "
          f"pv1 shape {fd(dd('pv1_e15', 'shape', 'h_raw_vis'))} · color {fd(dd('pv1_e15', 'color', 'h_raw_vis'))}).")
        P(f"- **(ii) 클래스 평균 부분공간** (resid, 교차 적합 / split-half 천장) — shape z–p (visible) release {zs['shape']['release']['v']:.3f} → "
          + " · ".join(f"{SHORTL[t]} {zs['shape'][t]['v']:.3f}" for t in tg) + f" (encoder 끼리 z–h_fut {enc:.3f}): v11 · pv1 이 encoder 쪽으로 간다. "
          f"color 는 z 기준 공유 몫이 {czd} 로 약간 내려가지만, "
          f"h_fut 기준 (학습 표적) 으로는 v11 {SBt['v11_e10']['color|h_fut-p|visible']['d']:+.3f} / {SBt['v11_e10']['color|h_fut-p|late']['d']:+.3f} · "
          f"pv1 {SBt['pv1_e15']['color|h_fut-p|visible']['d']:+.3f} / {SBt['pv1_e15']['color|h_fut-p|late']['d']:+.3f} (visible / late) 로 약간 오른다. "
          f"p 의 color split-half 천장이 {_rng([SBt[t][f'color|ceil_p|{s_}'] for t in ['release'] + tg for s_ in ('visible', 'late')], 2)} "
          f"(shape {_rng([SBt[t][f'shape|ceil_p|{s_}'] for t in ['release'] + tg for s_ in ('visible', 'late')], 2)}) 라 color 부분공간 추정은 노이즈가 크다 — "
          "'shape 만 움직인다' 고 쓰지 않는다.")
        P(f"- **도메인** — 이 이동은 **같은 생성기 학습 (in-domain)** 결과다 (v11_postft = 궤적 재생, pv1 = 같은 생성기 · 7 모양 중 5). "
          f"IntPhys1 post-FT 는 일부만 움직인다: h_fut 머리 → p raw (visible, shape) logistic {fd(dd('ip1_e40', 'shape', 'h_raw_vis'))} · "
          f"ridge {fd(dd('ip1_e40', 'shape', 'h_raw_vis', 'dr_v11test'))}, shape z–p (visible) {SBt['ip1_e40']['shape|z_all-p|visible']['d']:+.3f}. "
          f"IntPhys2 · ariel 은 거의 없다 (h raw vis shape ip2 {fd(dd('ip2_e80', 'shape', 'h_raw_vis'))} · ariel {fd(dd('ariel_ep43', 'shape', 'h_raw_vis'))}).")
    pv = {tn: [acc(t, tn, "z_procr_vis_late", sc) for t in tr_] for tn in TARGETS}
    pl = {tn: [acc(t, tn, "z_procr_late_late", sc) for t in tr_] for tn in TARGETS}
    P(f"- **(iii) 정렬** — visible clip 짝으로 맞춘 직교 사상 (z 머리) 을 late clip 에 걸면 shape {_rng(pv['shape'])} % · color {_rng(pv['color'])} % "
      f"(release {acc('release', 'shape', 'z_procr_vis_late', sc):.1f} / {acc('release', 'color', 'z_procr_vis_late', sc):.1f}), late 짝으로 맞추면 "
      f"{_rng(pl['shape'])} / {_rng(pl['color'])} % — 어느 predictor 도 late 짝 수준에 못 미친다. "
      f"ip2 만 부분 개선 (shape logistic {fd(dd('ip2_e80', 'shape', 'z_procr_vis_late'))} · ridge {fd(dd('ip2_e80', 'shape', 'z_procr_vis_late', 'dr_v11test'))}). "
      f"이식 크기는 probe 에 달려 있다 — pv1 logistic {dd('pv1_e15', 'shape', 'z_procr_vis_late')['d']:+.1f} vs ridge {dd('pv1_e15', 'shape', 'z_procr_vis_late', 'dr_v11test')['d']:+.1f}, "
      f"ariel {dd('ariel_ep43', 'shape', 'z_procr_vis_late')['d']:+.1f} vs {dd('ariel_ep43', 'shape', 'z_procr_vis_late', 'dr_v11test')['d']:+.1f} (shape).")
    P("- **주장하지 않는 것** — 약 5 pt 이하의 정확도 차와 0.01 이하의 공유 몫 차는 (궤적, k) 단위 CI 에서 대부분 유의하지 않으므로 주장하지 않는다. "
      "방향 주장 (표의 **굵게**) 은 logistic · ridge 짝 CI 가 같은 방향으로 0 을 벗어나고 최적화 잡음보다 큰 칸만. predictor 마다 한 seed. 다중 비교 보정 없음.\n")
    if figs:
        P("그림: " + " · ".join(f"[`{v}`]({v})" for v in figs.values()) + "\n")
        P(f"![delta]({figs['delta']})\n")
    return "\n".join(out)


def _claims(J, t, sc):
    L = J["logistic"][t]
    up, dn = [], []
    for tn in TARGETS:
        for rid in KEY_ROWS:
            c = L[tn][rid].get("claim_" + sc)
            if c == 1:
                up.append(f"{tn} {LROW[rid][1]} ({L[tn][rid]['d_' + sc]['d']:+.1f})")
            elif c == -1:
                dn.append(f"{tn} {LROW[rid][1]} ({L[tn][rid]['d_' + sc]['d']:+.1f})")
    return up, dn


def md_read(J, runs, gruns, figs):
    L = J["logistic"]; SB = J["subspace_boot"]; GR = J["geometry"]; PM = J["probe_misc"]
    tr_ = [t for t in TRAINED if t in runs]; tg = [t for t in TRAINED if t in gruns]
    acc = lambda t, tn, r, sc="full": L[t][tn][r][sc]["acc"]
    out = []; P = out.append
    P("## 읽기\n")
    P("### (i) 정체는 p 에 남는가 — 남는다\n")
    wmin = min(acc(t, tn, r) for t in runs for tn in TARGETS for r in ("p_vis_vis", "p_late_late"))
    P(f"- 같은 조건 안 probe (visible→visible, late→late) 는 모든 predictor 에서 {wmin:.1f} % 이상이다. 천장 근처라 predictor 사이의 작은 차이는 "
      "이식 행 만큼 믿지 않는다 (PROBE.md §6). v11 의 late→late 차 (shape "
      f"{L['v11_e10']['shape']['p_late_late']['d_v11test']['d']:+.1f} · color {L['v11_e10']['color']['p_late_late']['d_v11test']['d']:+.1f}, v11test) 는 "
      "최적화 잡음 바닥 수준이라 개선으로 읽지 않는다 (⚠️ 정정 2026-09-25 — 이전 '두 속성 모두 조금 오른다 (굵게)'). ariel 의 color late→late 는 logistic 만 "
      f"{L['ariel_ep43']['color']['p_late_late']['d_full']['d']:+.1f} (ridge {L['ariel_ep43']['color']['p_late_late']['dr_full']['d']:+.1f}) 이다.")
    RT = J["ridge_tubelet"]
    lt = {t: min(min(d["full"]["acc"] for d in RT[t][tn]) for tn in TARGETS) for t in RT}
    P(f"- 미래 튜블릿 하나씩 (late 머리, **ridge** — logistic (5b) 는 공용 GPU 시간 때문에 `--skip-tubelet` 로 뺐다) 의 최저값 (shape · color 중): "
      + " · ".join(f"{SHORTL[t]} {lt[t]:.1f}" for t in RT) + " % — t7 까지 떨어지지 않는다. 튜블릿별 값과 짝 차이는 표 1b. "
      f"(참고: 정본 release 의 logistic (5b) late 는 shape {min(J['ref_logistic_tubelet']['shape']):.1f}–{max(J['ref_logistic_tubelet']['shape']):.1f} · "
      f"color {min(J['ref_logistic_tubelet']['color']):.1f}–{max(J['ref_logistic_tubelet']['color']):.1f} %.)")
    P("- 짝 차이 (같은 clip, (궤적, k) 묶음 bootstrap) 로 방향이 선 행 (logistic · ridge 둘 다, probe test 전체):")
    for t in tr_:
        up, dn = _claims(J, t, "full")
        P(f"  - **{LABEL[t]}** — ↑ " + ("; ".join(up) if up else "없음") + " / ↓ " + ("; ".join(dn) if dn else "없음"))
    P("- 같은 판정을 v11_split test 부분집합 (v11_postft 가 안 본 block) 에서:")
    for t in tr_:
        up, dn = _claims(J, t, "v11test")
        P(f"  - **{LABEL[t]}** — ↑ " + ("; ".join(up) if up else "없음") + " / ↓ " + ("; ".join(dn) if dn else "없음"))
    conf = [(t, tn, r) for t in tr_ for tn in TARGETS for r in KEY_ROWS
            if "dr_full" in L[t][tn][r] and sign_of(L[t][tn][r]["d_full"]) * sign_of(L[t][tn][r]["dr_full"]) == -1]
    P(f"- **logistic 과 ridge 가 반대 방향으로 CI 를 벗어난 칸** (probe test 전체, 핵심 행): {len(conf)} 개 — "
      + ("; ".join(f"{SHORTL[t]} {tn} {LROW[r][1]} (logistic {L[t][tn][r]['d_full']['d']:+.1f} / ridge {L[t][tn][r]['dr_full']['d']:+.1f})" for t, tn, r in conf) if conf else "없음")
      + ". 교차 이식 행의 크기 · 부호는 probe 에 달려 있다는 PROBE.md §6 의 경고가 predictor 비교에도 그대로다.")
    P(f"- v11 의 뚜렷한 **감소** 하나: shape `p late 머리 → visible` {L['v11_e10']['shape']['p_late_vis']['d_full']['d']:+.1f} pt "
      f"(v11 test 부분 {L['v11_e10']['shape']['p_late_vis']['d_v11test']['d']:+.1f}) — late clip 으로 배운 p 머리가 visible 에 덜 통하게 됐다. "
      f"ip1 은 이식 · 정렬 행 여럿이 같이 내려간다 (shape vis→late {L['ip1_e40']['shape']['p_vis_late']['d_full']['d']:+.1f}, "
      f"Procrustes (late 짝) → late {L['ip1_e40']['shape']['z_procr_late_late']['d_full']['d']:+.1f}).")
    if "release_bs16" in runs:
        nz = [abs(L["release_bs16"][tn][r]["d_full"]["d"]) for tn in TARGETS for r in KEY_ROWS]
        P(f"- 추출 잡음 바닥: 같은 릴리즈 predictor 를 다른 predictor 와 같은 패스 (배치 16) 로 다시 뽑은 p 는 핵심 행에서 |Δ| ≤ {max(nz):.1f} pt "
          f"(중앙 {np.median(nz):.1f}). 이보다 작은 차이는 읽지 않는다.")
    if tg:
        P("\n### (ii) p 의 클래스 배치가 encoder 쪽으로 가는가 — v11 · pv1 (같은 생성기) 에서; 부분공간으로는 shape 가 뚜렷, color 는 지표에 따라 갈린다\n")
        for tn in TARGETS:
            parts = []
            for t in tg:
                o = SB["full"][t][f"{tn}|z_all-p|visible"]
                parts.append(f"{SHORTL[t]} {o['v']:.3f} (Δ {o['d']:+.3f} [{o['dlo']:+.3f}, {o['dhi']:+.3f}])")
            P(f"- {tn}, z–p 공유 몫 (resid, visible): release {SB['full']['release'][f'{tn}|z_all-p|visible']['v']:.3f} → " + " · ".join(parts))
        for tn in TARGETS:
            parts = [f"{SHORTL[t]} {SB['full'][t][f'{tn}|h_fut-p|late']['v']:.3f}" for t in tg]
            P(f"- {tn}, h_fut–p 공유 몫 (resid, late): release {SB['full']['release'][f'{tn}|h_fut-p|late']['v']:.3f} → " + " · ".join(parts)
              + f" (encoder 끼리 z–h_fut late {SB['full']['release'][f'{tn}|z_all-h_fut|late']['v']:.3f})")
        order = {sc: sorted(tg + ["release"], key=lambda t: -SB[sc][t]["shape|z_all-p|visible"]["v"]) for sc in ("full", "v11test")}
        same = order["full"] == order["v11test"]
        P(f"- v11_split test clip 만으로 다시 재면 (표 4 아래 블록) shape z–p (visible) 의 순서는 "
          + " > ".join(SHORTL[t] for t in order["v11test"]) + (" — 전체와 **같다**. v11_postft 가 학습한 clip 의 p 가 끌어올린 값이 아니다."
                                                              if same else f" — 전체 ({' > '.join(SHORTL[t] for t in order['full'])}) 와 **다르다**."))
        P(f"- color 는 다르게 움직인다: z 와의 공유 몫은 안 오르는데 (위), 같은 p 안의 visible–late 클래스 부분공간 공유 몫은 "
          + " · ".join(f"{SHORTL[t]} {GR[t]['share|color|p_vis-p_late']:.3f}" for t in ["release"] + tg)
          + f" 로 v11 · pv1 에서 오르고, resid visible→late color probe 도 v11 {GR['v11_e10']['resid_v2l|color|p_fut']:.1f} (release {GR['release']['resid_v2l|color|p_fut']:.1f}) 로 오른다. "
          "즉 v11 · pv1 의 color 배치는 조건 사이에서는 더 일관되고, z 기준 공유 몫은 약간 내려가지만 h_fut (학습 표적) 기준으로는 v11test 에서 "
          f"v11 {SB['v11test']['v11_e10']['color|h_fut-p|visible']['d']:+.3f} · pv1 {SB['v11test']['pv1_e15']['color|h_fut-p|visible']['d']:+.3f} (visible) 로 약간 오른다. "
          "p 의 color split-half 천장이 낮아 (표 4 위 단락) color 부분공간 추정은 노이즈가 크다 (⚠️ 정정 2026-09-25 — 이전 'encoder 의 color 방향 쪽으로는 가지 않는다').")
        P("- resid visible→late 선형 probe (p): shape " + " · ".join(f"{SHORTL[t]} {GR[t]['resid_v2l|shape|p_fut']:.1f}" for t in ["release"] + tg)
          + f" % (z_all {GR['release']['resid_v2l|shape|z_all']:.1f} · h_fut {GR['release']['resid_v2l|shape|h_fut']:.1f}); color "
          + " · ".join(f"{SHORTL[t]} {GR[t]['resid_v2l|color|p_fut']:.1f}" for t in ["release"] + tg)
          + f" % (z_all {GR['release']['resid_v2l|color|z_all']:.1f} · h_fut {GR['release']['resid_v2l|color|h_fut']:.1f}). CI 없음 — 참고.")
        P("- p 의 크기 · 이동 몫: RMS 반경 비 p/h_fut (late) " + " · ".join(f"{SHORTL[t]} {GR[t]['rms_ratio|p/h_fut|late']:.3f}" for t in ["release"] + tg)
          + "; h_fut–p 짝 거리 중 중심 이동 몫 (late) " + " · ".join(f"{SHORTL[t]} {GR[t]['offset|h_fut-p|late']:.3f}" for t in ["release"] + tg)
          + "; cos(p_fut, h_fut) test late " + " · ".join(f"{SHORTL[t]} {PM[t]['geometry']['cos_pfut_hfut_test_late']:.3f}" for t in ["release"] + tr_) + ".")
        P("- 읽는 법: v11 · pv1 은 p 를 **절대 좌표에서** h_fut 에 붙이고 (cos · 이동 몫 · raw h 머리 이식이 같이 움직인다), shape 의 클래스 평균 방향도 "
          "encoder 의 것과 더 겹치게 된다. color 의 클래스 평균 방향은 z 기준으로는 그러지 않고 h_fut 기준으로는 약간 그렇다 — 두 속성의 차이는 resid 부분공간 지표에서만 보인다. "
          "왜 다른지는 이 자료로 못 가른다 (학습셋의 색 구성 · 풀링 벡터에서 색 몫이 0.5 % 로 작다는 것 등이 후보일 뿐이다).")
    P("\n### (iii) visible 에서 맞춘 정렬이 late 에 통하게 되는가 — 아니다 (ip2 만 부분 개선)\n")
    for tn in TARGETS:
        P(f"- {tn}: Procrustes (visible 짝) → late " + " · ".join(f"{SHORTL[t]} {acc(t, tn, 'z_procr_vis_late'):.1f}" for t in runs)
          + " % vs (late 짝) → late " + " · ".join(f"{SHORTL[t]} {acc(t, tn, 'z_procr_late_late'):.1f}" for t in runs) + " %.")
    P("- 상대 잔차 (visible 짝 사상, test late): " + " · ".join(f"{SHORTL[t]} {PM[t]['procrustes_z']['rel_residual_test_late']:.3f}" for t in runs)
      + " vs test visible " + " · ".join(f"{SHORTL[t]} {PM[t]['procrustes_z']['rel_residual_test_visible']:.3f}" for t in runs) + ".")
    P("- 즉 v11 · pv1 이 p 를 h 쪽으로 옮겨도 '가림 끝 clip 의 p 는 visible clip 의 p 와 다른 자리에 놓인다' 는 릴리즈의 성질은 그대로다. "
      "v11_postft 는 가림 팔 (late 가림) 을 학습했는데도 그렇다. "
      f"도메인 밖 ip2 가 가장 많이 줄인다 (shape v11test logistic {L['ip2_e80']['shape']['z_procr_vis_late']['d_v11test']['d']:+.1f} · "
      f"ridge {L['ip2_e80']['shape']['z_procr_vis_late']['dr_v11test']['d']:+.1f}) — 그래도 late 짝 수준에는 한참 못 미친다.")
    CJ = J.get("curves")
    if CJ:
        res = CJ["res"]
        P("\n### 학습 곡선 (ridge · 부분공간, 표 8 · `fig_identity_curves.png`)\n")
        for fam, pre in (("v11", "v11_e"), ("pv1", "pv1_e"), ("ip1", "ip1_e"), ("ip2", "ip2_e"), ("ariel", "ariel_ep")):
            tags = [t for t in CJ["tags"] if t.startswith(pre)]
            P(f"- {fam}: shape 공유 몫 z–p (visible) " + " → ".join(f"{res[t]['subspace']['shape|z_all-p_fut|visible']:.3f}" for t in (["release"] if fam != "ariel" else []) + tags)
              + "; color " + " → ".join(f"{res[t]['subspace']['color|z_all-p_fut|visible']:.3f}" for t in (["release"] if fam != "ariel" else []) + tags)
              + "; ridge h 머리 → p raw (visible, shape) " + " → ".join(f"{res[t]['ridge']['shape']['hhead_p_raw']['vis']:.1f}" for t in (["release"] if fam != "ariel" else []) + tags))
        P("- 곡선의 release 점은 다른 tag 와 같은 추출 패스 (배치 16) 의 p 다 (표 1–7 의 release 는 원래 캐시). ariel 은 scratch 라 release 에서 출발하지 않는다.")
    return "\n".join(out)


def md_tail(J, runs, figs, elapsed):
    out = []; P = out.append
    P("## 단서\n")
    P("- **v11 은 궤적이 적다.** 운동 조건마다 x-궤적 2 개 (좌→우 · 우→좌), 정지 8 자리. v11_split test block 도 train 과 궤적 · env · 모양 · 색을 공유한다 → "
      "v11_postft 의 이득은 **궤적 재생 · 도메인 안** 이득이다. pv1 (Predictor_v1_training) 도 같은 생성기 · v11 의 7 모양 중 5 (cone · torus 없음) · 8 색 중 6 · env 3 개다.")
    P("- **v11_postft 누수** — 풀링 9,408 clip 의 절반 (4,704) 이 v11_postft 학습셋 (`index_train.csv`) 에 있다. probe test 중 v11_split test 에 드는 것만 따로 (표 3), "
      "ridge 는 v11_split 로 다시 나눈 판 (train = 학습한 절반) 도 (표 6 아래). 부분공간 공유 몫은 v11_split test clip 만으로도 다시 쟀다 (표 4). "
      "그래도 probe **머리**는 학습 clip 의 p 로 맞춘 것이 섞인다 (probe split 이 v11_split 과 다르다).")
    P("- **CI 단위 (⚠️ 정정 2026-09-25)** — 이전 판은 block (1–2 clip) bootstrap 에 '짝 비교는 괜찮다' 고 썼다 — **철회**. block 은 사실상 clip 단위라 "
      "같은 궤적 · k 를 공유하는 거의-복제를 못 묶었고, 올바른 단위에서 짝 CI 가 1.5–4 배 넓어져 작은 굵게 여럿이 사라진다 (검증 원문). 지금은 모든 정확도 CI · 짝 CI · "
      f"부분공간 재추출을 (궤적, k) 묶음 ({J['meta']['n_clusters']['all']} 개) 으로 한다. 묶음 수가 적어 (test∩v11test late {J['meta']['n_clusters']['test_v11te_late']}) CI 자체가 거칠다. "
      "부분공간 bootstrap 은 절반 분할을 고정한 채 묶음만 재추출한다 (분할 변동은 안 들어간다 — 미적용).")
    P("- **이식 수치의 크기는 probe 에 따라 크게 흔들린다** (PROBE.md §6: ridge vs logistic 18–60 pt 차). 그래서 방향만 두 방법이 같을 때 주장하고, 크기는 같은 probe 끼리만 비교한다. "
      f"logistic λ 는 p 머리 {3 * len(TARGETS) * len(runs)} 개 중 "
      f"{sum(1 for t in runs for tn in TARGETS for k_ in ('p_vis', 'p_late', 'p_all') if J['lambda'][t][tn][k_] == 1e-4)} 개가 격자 끝 1e-4 (표 9).")
    P("- **부분공간 bootstrap 의 치우침** — 복원추출은 클래스 평균에 잡음을 더해 공유 몫 분포를 점추정에서 비켜 놓는다 (이전 판은 점추정이 자기 CI 밖인 칸이 있었다, "
      "예: v11 color h_fut–p late 0.774 [0.778, 0.810]). 지금은 basic (반사) 구간 [2θ − q97.5, 2θ − q2.5] 을 쓰고, "
      f"|Δ| < {SUB_MIN} 인 공유 몫 차이는 CI 와 무관하게 주장하지 않는다.")
    P("- **p 는 다시 LN 하지 않는다** (z · h 는 토큰 LN). p 노름 · 퍼짐의 변화 일부는 정규화 차이일 수 있다 — '학습된 기하' 로 바로 읽지 않는다.")
    P("- **풀링 벡터다.** 평균 풀링은 위치를 버리고 장면 칸이 분산의 ~78 % 다. '정체가 p 에 있다' 는 풀링 선형 probe 기준이고, 토큰 수준 배치는 이 자료로 못 본다. "
      "위치 (자, attention 3×3 질량) 주장은 이 문서에 없다 — 자 판독은 `../v11readout/`.")
    P("- **A/B 방향** (CLAUDE.md §1-7): 이 분석은 가능 변이만 쓰는 probing 이라 vanish 쌍의 방향 (물체→빈 · 빈→물체) 이 없다. 채점의 방향별 정확도는 `../v11score/`.")
    C = J["checks"]; M = J["meta"]
    ra = C.get("release_probe_vs_reference_acc", {})
    gv = C.get("release_probe_gpu_vs_reference")
    P(f"- **장치** — predictor 비교용 logistic probe 는 **CPU** 에서 돌렸다 (공용 GPU 가 다른 추출로 포화돼 L-BFGS 가 머리당 70–135 s; CPU 8 스레드는 ~50 s). "
      f"같은 장치끼리라 p 무관 행은 실행 사이에 비트 단위로 같다 (표 7). CPU release ↔ 정본 (GPU) 의 정확도 칸 {ra.get('n_acc')} 개 중 {ra.get('n_nonzero')} 개가 다르고 "
      f"최대 {ra.get('max_abs_pt'):.2f} pt (L-BFGS 수치 경로 차) — 행마다 이 차를 '최적화 잡음' 으로 방향 주장 조건에 넣었다."
      + (f" 같은 코드를 GPU 에서 돌린 release 는 정본과 공통 잎 {gv['n_shared']} 개 중 **{gv['n_value_diff']} 개** 다르다 ((5b) 제외)." if gv else ""))
    P("- **(5b) 튜블릿별 logistic 은 뺐다** (`--skip-tubelet`, 원 스크립트의 옵션) — 머리 32 개가 p 의존 머리의 대부분이라 공용 GPU·CPU 시간이 안 됐다. "
      "튜블릿별 읽기는 ridge 로 모든 predictor 에 대해 냈다 (표 1b); 정본 release 의 logistic (5b) 는 참고로만.")
    P("- **ariel 은 post-FT 가 아니다** — attention 규칙 (block-causal) · 데이터 (SSv2 + K400) · scratch · epoch 네 가지가 같이 다르다. ariel 의 차이를 한 원인에 붙이지 않는다.")
    P("- **원인을 학습 목적함수에 걸지 않는다.** 'p 가 h 쪽으로 간다' 는 관측이고, 두 속성의 부분공간 지표가 왜 다른지도 여기서 답하지 않는다.")
    P("- 한 번씩만 학습했다 (seed 0) — 학습 분산을 모른다. release · release(bs16) 차이는 추출 잡음 바닥일 뿐 학습 잡음이 아니다.")
    P("- 가림 (late) 은 testbed 다 — 부분공간 차이는 visible 에서 이미 보인다 (GEOMETRY.md). late 조건에는 가림막이 장면에 있다 (§8-5 설계).")
    P("\n## 재현\n")
    P("```bash\ncd /data/hyuntak/project/2026/2027_cvpr/vjepa2\nP=/data/hyuntak/anaconda3/envs/vjepa2/bin/python\n"
      "# 0) 특징 (GPU, 별도 실행): cache/training_effects/v11readout_curves/ (pool_z · pool_h · pool_p__<tag>, 16,128 clip)\n"
      "$P z_research/scripts/analysis/te_v11_readout.py --preset curves --gpus 4\n"
      "# 1) predictor 별 풀링 캐시 (sel=late ∧ obj 9,408 clip, 릴리즈 순서; z·h·meta 는 릴리즈 symlink)\n"
      "$P z_research/scripts/analysis/te_analyze_identity.py prep --tags release v11_e10 ip1_e40 ip2_e80 pv1_e15 ariel_ep43\n"
      "# 2) probe 비교 세트 (CPU, float64 L-BFGS, 8 스레드; release 먼저 → 나머지는 p 무관 머리를 캐시에서). release 는 원래 캐시,\n"
      "#    release_bs16 은 다른 predictor 와 같은 패스로 다시 뽑은 릴리즈 (추출 잡음 바닥)\n"
      "export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8\n"
      "for t in release v11_e10 ip1_e40 ip2_e80 pv1_e15 ariel_ep43 release_bs16; do\n"
      "  $P z_research/scripts/analysis/te_analyze_identity.py probe --tag $t --device cpu --skip-tubelet --root probe_cpu\n"
      "done\n"
      "#    정본 재현 검사 (GPU, 정본과 같은 장치): → v11_identity/probe/release\n"
      "$P z_research/scripts/analysis/te_analyze_identity.py probe --tag release --device cuda:0 --skip-tubelet\n"
      "#    = V11P_CACHE=<cache> V11P_PROBE_OUT=v11_identity/<root>/<t> $P z_research/scripts/analysis/v11_pooled_probe.py --device <d> --skip-tubelet\n"
      "#      (래퍼가 더하는 것: evaluate() 의 clip 별 정오 기록 · 입력 sha1 이 같은 머리의 캐시. 머리 · 수치는 원 코드 그대로)\n"
      "# 3) geometry (GPU 1 장씩) = V11P_CACHE=<cache> $P z_research/scripts/analysis/v11_pooled_geometry.py --device cuda:0 --out v11_identity/geometry/<t>\n"
      "for t in release v11_e10 ip1_e40 ip2_e80 pv1_e15 ariel_ep43 release_bs16; do\n"
      "  $P z_research/scripts/analysis/te_analyze_identity.py geometry --tag $t --device cuda:0\n"
      "done\n"
      "# 4) CPU\n"
      "$P z_research/scripts/analysis/te_analyze_identity.py ridge      # PROBE.md §8 ridge 재검사 × predictor (+ v11_split 재분할)\n"
      "$P z_research/scripts/analysis/te_analyze_identity.py curves     # 22 tag 학습 곡선 (ridge 핵심 행 + 부분공간)\n"
      "$P z_research/scripts/analysis/te_analyze_identity.py report     # → IDENTITY.md · identity.json · fig_identity_*.png\n```\n")
    P(f"산출물: `probe/<tag>/{{results.json, perclip.npz, run_info.json, fig_probe*.png}}` · `geometry/<tag>/{{results.json, fig_*.png}}` · `ridge/` · `curves/curves.json` · "
      f"`prep/prep_checks.json` · `identity.json`. 캐시 `/data2/local_datasets/world/world_analysis/cache/training_effects/v11pooled_<tag>/`. report {elapsed:.0f} s (CPU).")
    return "\n".join(out)


def build_md(J, runs, gruns, rruns, figs, elapsed):
    return (md_head(J, runs, gruns, figs) + "\n## 표\n\n" + md_tables(J, runs, gruns, rruns) + "\n" + md_read(J, runs, gruns, figs)
            + "\n\n" + md_tail(J, runs, figs, elapsed) + "\n")


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("prep"); s.add_argument("--tags", nargs="+", default=FINAL); s.add_argument("--src", default=str(CURVES))
    s.add_argument("--allow-partial", action="store_true")
    s = sp.add_parser("probe"); s.add_argument("--tag", required=True); s.add_argument("--device", default="cuda:0")
    s.add_argument("--skip-tubelet", action="store_true", help="원 스크립트의 --skip-tubelet ((5b) 튜블릿별 머리 생략 — 공용 GPU 시간 때문)")
    s.add_argument("--root", default="probe", help="출력 = v11_identity/<root>/<tag> (probe = GPU, probe_cpu = CPU 비교 세트)")
    s = sp.add_parser("geometry"); s.add_argument("--tag", required=True); s.add_argument("--device", default="cuda:0")
    s = sp.add_parser("ridge"); s.add_argument("--tags", nargs="+", default=RUNS)
    s = sp.add_parser("curves"); s.add_argument("--tags", nargs="+", default=CURVE_TAGS); s.add_argument("--allow-partial", action="store_true")
    s = sp.add_parser("report"); s.add_argument("--probe-root", default="probe_cpu",
                                                 help="predictor 비교에 쓰는 probe 세트 (같은 장치끼리). probe/release 는 정본 재현 검사용 (GPU)")
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    {"prep": cmd_prep, "probe": cmd_probe, "geometry": cmd_geometry, "ridge": cmd_ridge, "curves": cmd_curves,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
