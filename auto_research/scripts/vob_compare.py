#!/usr/bin/env python3
"""[P] 기준 AR vs VOBench 섞은 AR — 어디서 오르고 무엇이 바뀌었나 (출력 · 채점, 학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/vob_compare.py

사전 등록: 진행판 [P] KST 04:35.
모델: base = natural_ar_vw_state latest · vob = natural_ar_vw_state_vob latest (같은 레시피, 배치 10 % VOBench 학습 쪽 가능 영상).
(1) VOBench 끝 가림 (공유 캐시 vobench_late, 216 block): 문맥 8 블록 → 굴린 예측 8 블록, 미래 L1 (화면 전체 · 물체 자리 = oracle_vob 의 oidx).
    쌍 (pos_a, imp_ab) = 짝 1 · (pos_b, imp_ba) = 짝 2. 영속은 obj_b = empty → 짝 1 = 사라짐, 짝 2 = 나타남. 보류 block = vobench_v1_heldout 시험 쪽.
(2) 출력이 문맥에서 떨어지는 정도 (같은 칸 문맥 평균 m 기준): dev = Σ‖p − m‖₁ / Σ‖y − m‖₁, 존재감 (투영) = Σ⟨p − m, y − m⟩ / Σ‖y − m‖²,
    copy 기준 = Σ‖p − h_last‖₁ / Σ‖y − h_last‖₁. 칸 무리: 들어오는 칸 (VOBench = 그 팔 물체의 라벨 칸 (보일 때) · 자연 / 3D 합성 = 흐름 ≥ 0.5 칸),
    정지 칸 (VOBench = 어느 블록에도 물체 칸이 아니고 미래 변화가 그 무리의 중앙값 아래 · 자연 / 3D 합성 = 이 블록 · 앞 블록 흐름 < 0.25 칸).
    자연 · 3D 합성 = [T] ft_eval_reliability 의 창 (K710 보류 · HT100M · 3D 합성, 블록 8) 에서 TF 한 걸음 (블록 b+1 을 0..b 로, b = 1..6).
출력: auto_research/results/vob_compare.json."""
import collections, csv, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT  # noqa: E402
sys.argv = _argv
torch.set_num_threads(8)
dev = torch.device("cuda")
S, G, C, K = 256, 16, 8, 8
MODELS = {"base": FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval(), "vob": FT.load_pred("natural_ar_vw_state_vob", "latest.pt").to(dev).eval()}
OUT = os.path.join(REPO, "auto_research/results/vob_compare.json")
VB = "/workspace/dataset/benchmarks/VOBench_v1"
IDX = list(range(0, 96, 3))[:32]


def acc_stats(sums):                                                            # sums: 무리 → [Σ‖p−m‖₁, Σ‖y−m‖₁, Σ⟨p−m,y−m⟩, Σ‖y−m‖², Σ‖p−h‖₁, Σ‖y−h‖₁, n]
    return {g: dict(dev=round(v[0] / max(v[1], 1e-9), 3), presence=round(v[2] / max(v[3], 1e-9), 3), vs_copy=round(v[4] / max(v[5], 1e-9), 3), n=int(v[6])) for g, v in sums.items()}


def add(sums, key, p, y, m, hl):
    if p.shape[0] == 0:
        return
    a = sums.setdefault(key, [0.0] * 7)
    a[0] += float((p - m).abs().sum()); a[1] += float((y - m).abs().sum()); a[2] += float(((p - m) * (y - m)).sum()); a[3] += float(((y - m) ** 2).sum())
    a[4] += float((p - hl).abs().sum()); a[5] += float((y - hl).abs().sum()); a[6] += p.shape[0]


result = {}
t0 = time.time()
# ------------------------------------------------------------------ (1) + (2) VOBench 끝 가림
idx_rows = collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))):
    if r["occlusion"] == "late":
        idx_rows[r["block_id"]][r["variant"]] = r
held = {r["source_block"] for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1_heldout/index.csv")))}
meta = json.load(open("/workspace/auto_research_cache/vobench_late/meta.json"))
H = np.load("/workspace/auto_research_cache/vobench_late/h.npy", mmap_mode="r")
ARMS = ["pos_a", "pos_b", "imp_ab", "imp_ba"]
from PIL import Image


def centers(label_path, W0):
    L_ = json.load(open(label_path))["labels"]; out = {}
    for key in ("A", "B"):
        lab = {q["f"]: q for q in L_.get(key, [])}; cs = []
        for b in range(16):
            q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
            if any(t is None for t in q):
                cs.append(None); continue
            r0 = int(np.clip(np.mean([t["v"] for t in q]) / W0 * G, 0, G - 1)); c0 = int(np.clip(np.mean([t["u"] for t in q]) / W0 * G, 0, G - 1))
            cs.append((r0, c0, float(np.mean([t["hidden"] for t in q]))))
        out[key] = cs
    return out


def region(rc, rad=1):
    r0, c0 = rc[:2]
    return [rr * G + cc for rr in range(max(0, r0 - rad), min(G, r0 + rad + 1)) for cc in range(max(0, c0 - rad), min(G, c0 + rad + 1))]


recs = []
VSUM = {n: {} for n in MODELS}
for bi, m in enumerate(meta):
    arms = idx_rows[m["block_id"]]
    d0 = os.path.join(VB, arms["pos_a"]["file_name"])
    W0 = Image.open(f"{d0}/{IDX[0]:06d}.png").size[0]
    cn = centers(os.path.join(os.path.dirname(d0), "label.json"), W0)
    allobj = set()
    for key in ("A", "B"):
        for c_ in cn[key]:
            if c_ is not None:
                allobj |= set(region(c_))
    otok = []
    for j in range(K):
        s_ = set()
        for key in ("A", "B"):
            if cn[key][C + j] is not None:
                s_ |= set(region(cn[key][C + j]))
        otok.append(sorted(s_))
    oidx = torch.tensor([j * S + t for j in range(K) for t in otok[j]], device=dev)
    hh = {v: torch.from_numpy(np.asarray(H[m["i"], k], dtype=np.float32)).to(dev) for k, v in enumerate(ARMS)}
    mv = "static" if arms["pos_a"]["motion"] == "flat_static" else "moving"
    viol = arms["pos_a"]["violation"]; heldout = arms["pos_a"]["source_block"] in held
    empty_b = arms["pos_a"].get("obj_b", "") == "empty"
    for pid, (pos, imp, key) in (("1", ("pos_a", "imp_ab", "A")), ("2", ("pos_b", "imp_ba", "B"))):
        ctx = hh[pos][:C]; mctx = ctx.mean(0); hl = ctx[-1]
        for n, pr in MODELS.items():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                p = pr.rollout_kv(ctx.reshape(1, C * S, -1), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, -1)
            sc = {}
            for v in (pos, imp):
                e = (p.reshape(K * S, -1) - hh[v][C:].reshape(K * S, -1)).abs().mean(-1)
                sc[v] = (float(e.mean()), float(e[oidx].mean()) if len(oidx) else float("nan"))
            direction = ("vanish" if pid == "1" else "appear") if (viol == "permanence" and empty_b) else f"pair{pid}"
            recs.append(dict(block=m["block_id"], model=n, pair=pid, mv=mv, viol=viol, k=int(arms[pos]["k"]), held=heldout, dir=direction,
                             full=sc[imp][0] > sc[pos][0], obj=sc[imp][1] > sc[pos][1], s_pos=sc[pos], s_imp=sc[imp]))
            # (2) 가능 미래 기준 출력 성질
            y = hh[pos][C:]
            if not (viol == "permanence" and empty_b and key == "B"):
                for j in range(K):
                    c_ = cn[key][C + j]
                    if c_ is None or c_[2] >= 0.5:
                        continue
                    cells = torch.tensor(region(c_), device=dev)
                    add(VSUM[n], f"{mv}|enter|{'held' if heldout else 'train'}", p[j, cells], y[j, cells], mctx[cells], hl[cells])
            nonobj = torch.tensor([c for c in range(S) if c not in allobj], device=dev)
            chg = (y[:, nonobj] - mctx[nonobj]).abs().mean(-1).max(0).values
            st = nonobj[chg <= chg.median()]
            add(VSUM[n], f"{mv}|static|{'held' if heldout else 'train'}", p[:, st].reshape(-1, 1280), y[:, st].reshape(-1, 1280), mctx[st].repeat(K, 1), hl[st].repeat(K, 1))
    if bi % 40 == 0:
        print(f"VOBench {bi}/{len(meta)} ({time.time() - t0:.0f}s)", flush=True)


def table(rs):
    out = {}
    keys = {"all": lambda r: True}
    for mv in ("moving", "static"):
        keys[mv] = (lambda mv_: lambda r: r["mv"] == mv_)(mv)
        for vi in ("permanence", "shape", "color"):
            keys[f"{mv}|{vi}"] = (lambda mv_, vi_: lambda r: r["mv"] == mv_ and r["viol"] == vi_)(mv, vi)
        for dr in ("vanish", "appear", "pair1", "pair2"):
            keys[f"{mv}|{dr}"] = (lambda mv_, dr_: lambda r: r["mv"] == mv_ and r["dir"] == dr_)(mv, dr)
    for split in ("held", "train"):
        for g, fn in keys.items():
            sel = [r for r in rs if fn(r) and r["held"] == (split == "held")]
            if sel:
                out[f"{split}|{g}"] = dict(n=len(sel), obj=round(100 * np.mean([r["obj"] for r in sel]), 1), full=round(100 * np.mean([r["full"] for r in sel]), 1))
    return out


result["vobench_acc"] = {n: table([r for r in recs if r["model"] == n]) for n in MODELS}
result["vobench_out"] = {n: acc_stats(v) for n, v in VSUM.items()}
for n in MODELS:
    t = result["vobench_acc"][n]
    print(n, {k: (v["obj"], v["full"], v["n"]) for k, v in t.items() if k.split("|")[0] == "held" and k.count("|") <= 2}, flush=True)
    print(n, result["vobench_out"][n], flush=True)
# 짝 대응 차 (보류, 물체 자리): vob − base
def paired(rs, fn, field):
    a = {(r["block"], r["pair"]): r[field] for r in rs if r["model"] == "base" and fn(r)}
    b = {(r["block"], r["pair"]): r[field] for r in rs if r["model"] == "vob" and fn(r)}
    ks = sorted(set(a) & set(b)); d = np.array([float(b[k]) - float(a[k]) for k in ks]) * 100
    blocks = np.array([k[0] for k in ks]); ub = np.unique(blocks); rng = np.random.default_rng(0)
    bs = [np.concatenate([d[blocks == u] for u in rng.choice(ub, len(ub))]).mean() for _ in range(2000)]
    return dict(diff=round(float(d.mean()), 1), ci=[round(float(np.quantile(bs, 0.025)), 1), round(float(np.quantile(bs, 0.975)), 1)], n=len(ks))
result["vobench_paired_held"] = {}
for g, fn in (("moving", lambda r: r["mv"] == "moving" and r["held"]), ("static", lambda r: r["mv"] == "static" and r["held"]),
              ("moving|vanish", lambda r: r["mv"] == "moving" and r["held"] and r["dir"] == "vanish"),
              ("moving|appear", lambda r: r["mv"] == "moving" and r["held"] and r["dir"] == "appear")):
    for field in ("obj", "full"):
        result["vobench_paired_held"][f"{g}|{field}"] = paired(recs, fn, field)
print(result["vobench_paired_held"], flush=True)

# ------------------------------------------------------------------ (2) 자연 보류 · 3D 합성 (TF 한 걸음)
def windows(src):
    if src == "k710_holdout":
        Hn = np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_h.npy", mmap_mode="r")
        Un = np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_U.npy", mmap_mode="r")
        for w in range(Hn.shape[0]):
            yield np.asarray(Hn[w], np.float32), np.asarray(Un[w], np.float32).reshape(8, S, 2)
    else:
        EN = "/workspace/auto_research_cache/enc_noise"
        Un = np.load(os.path.join(REPO, f"auto_research/results/natural_transport_ctx5/rows_{src}.npz"))["Uwin"].astype(np.float32)
        w0 = 0
        for sh in (0, 1):
            T_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")
            for w in range(T_.shape[0]):
                yield np.asarray(T_[w, 0], np.float32), Un[w0 + w].reshape(8, S, 2)
            w0 += T_.shape[0]


result["tf_out"] = {}
for src in ("k710_holdout", "ht100m", "phys3d"):
    SUM = {n: {} for n in MODELS}
    for hn, U in windows(src):
        h = torch.from_numpy(hn).to(dev)
        idx = torch.arange(7 * S, device=dev).unsqueeze(0)
        for n, pr in MODELS.items():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                P = pr.forward_seq(h[:7].reshape(1, -1, 1280), idx).float()[0].view(7, S, -1)
            for b in range(1, 7):
                mctx = h[: b + 1].mean(0); y = h[b + 1]; p = P[b]; hl = h[b]
                u1 = np.linalg.norm(U[b + 1], axis=1); u0 = np.linalg.norm(U[b], axis=1)
                ent = torch.from_numpy(np.where(u1 >= 0.5)[0]).to(dev); st = torch.from_numpy(np.where((u1 < 0.25) & (u0 < 0.25))[0]).to(dev)
                add(SUM[n], "enter", p[ent], y[ent], mctx[ent], hl[ent]); add(SUM[n], "static", p[st], y[st], mctx[st], hl[st])
    result["tf_out"][src] = {n: acc_stats(v) for n, v in SUM.items()}
    print(src, result["tf_out"][src], f"({time.time() - t0:.0f}s)", flush=True)
json.dump(dict(result=result, recs=recs), open(OUT, "w"), indent=1, ensure_ascii=False)
print("저장", OUT)
