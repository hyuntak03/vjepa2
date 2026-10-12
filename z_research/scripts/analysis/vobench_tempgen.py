"""VOBench_v1 temporal generalization (2026-10-10) — 시점 하나에서만 attentive head 를 학습해 모든 미래 시점 t8–15 에 건다.

  묻는 것: p (predictor) 의 정체 표현 형식이 시점 사이에 일정한가 (h 대조). 캐시 = vobench_tubelet_probe.py extract (릴리즈 ViT-H, bf16).

  head = AttentiveClassifier (depth 1, query 1, head 16), 전체 토큰 256. 라벨 = joint (모양/색 + none) · shape (+ none) · color (+ none).
  학습 = (src ∈ {h, p}) × (t_train ∈ 8..15), train block, 문맥 끝 보임 조건 (visible · early · mid) 의 물체 + 빈 장면, 그 시점에 '보임' 튜블릿.
  평가 = test block, 모든 t_eval ∈ 8..15: p (visible · early · mid, 보임) · h (보임) · p late (물체, 섞임 제외).
  head 시드만 바꾼다 (block 분할은 늘 seed 0, probe 와 같음). 결과 runs/tempgen_<objset>_<label>_s<seed>.json, 정리 = summarize.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/vobench_tempgen.py run <objset> <label> <gpu> <seed>    # label ∈ joint · shape · color
  $P z_research/scripts/analysis/vobench_tempgen.py summarize                          # → RESULTS.md · summary.json
"""
import json, sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/data/hyuntak/project/2026/2027_cvpr/vjepa2"); sys.path.insert(0, "/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/scripts/analysis")
import vobench_concept_dim as C
from src.models.attentive_pooler import AttentiveClassifier
from pathlib import Path
OUT = Path(__file__).resolve().parents[3] / "z_research/VOBench_v1/vjepa2/probing/temporal_generalization"
if sys.argv[1] == "summarize":
    import vobench_tempgen_summary; vobench_tempgen_summary.main(OUT); sys.exit()
o, LT, gpu, SEED = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]); dev = torch.device(f"cuda:{gpu}")
M = json.load(open(C.CACHE / "meta.json"))["meta"]
rows = [i for i, m in enumerate(M) if m["objset"] == o]
blocks = sorted({M[i]["block_id"] for i in rows}); rng = np.random.default_rng(0); rng.shuffle(blocks); trb = set(blocks[:len(blocks) // 2])
st = {i: C.tub_state(M[i]["samples"]) for i in rows}
def lab(i):
    l = C.label_of(M[i])
    if l == "empty": return "none"
    return l if LT == "joint" else (l.split("/")[0] if LT == "shape" else l.split("/")[-1])
classes = sorted({lab(i) for i in rows}); K = {c: j for j, c in enumerate(classes)}; NONE = K["none"]
t0 = time.time()
H = np.load(C.CACHE / "h.npy", mmap_mode="r"); P = np.load(C.CACHE / "p.npy", mmap_mode="r")
ridx = np.array(rows); pos = {i: j for j, i in enumerate(rows)}
Hs = np.asarray(H[ridx, 8:16]); Ps = np.asarray(P[ridx])                    # (n, 8, 256, D) fp16 — 미래 구간만
print(f"[{o}/{LT}] loaded {time.time() - t0:.0f}s · classes {len(classes)}", flush=True)
X = {"h": Hs, "p": Ps}
def pairs(split, src, t, cond):
    return [i for i in rows if (M[i]["block_id"] in trb) == (split == "train") and cond(M[i]) and (st[i][t] == "vis" if src != "late" else st[i][t] != "mix")]
nonlate = lambda m: m["occlusion"] != "late"
late_obj = lambda m: m["occlusion"] == "late" and C.label_of(m) != "empty"
def xb(src, ii, t):
    return torch.from_numpy(np.asarray(X[src][[pos[i] for i in ii], t - 8], np.float32)).to(dev)
def train(src, t):
    tr = pairs("train", src, t, nonlate); y = torch.tensor([K[lab(i)] for i in tr], device=dev)
    torch.manual_seed(SEED); np.random.seed(SEED)
    clf = AttentiveClassifier(embed_dim=1280, num_heads=16, mlp_ratio=4.0, depth=1, num_classes=len(classes)).to(dev)
    ep, bs, lr = 40, 16, 1e-4
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=ep * ((len(tr) + bs - 1) // bs))
    for _ in range(ep):
        clf.train(); perm = np.random.permutation(len(tr))
        for s in range(0, len(tr), bs):
            sel = perm[s:s + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = nn.functional.cross_entropy(clf(xb(src, [tr[j] for j in sel], t)), y[sel])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    clf.eval(); return clf, len(tr)
def evaluate(clf, src, t, cond, src_key):
    ii = pairs("test", src_key, t, cond)
    ii = [i for i in ii if C.label_of(M[i]) != "empty"]                     # 정답률은 물체 샘플로
    if not ii: return None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pr = torch.cat([clf(xb(src, ii[s:s + 64], t)).float().argmax(1) for s in range(0, len(ii), 64)]).cpu().numpy()
    y = np.array([K[lab(i)] for i in ii])
    return dict(n=len(ii), acc=100 * float((pr == y).mean()), none=100 * float((pr == NONE).mean()))
res = {}
for src in ("p", "h"):
    for tt in range(8, 16):
        clf, ntr = train(src, tt)
        r = {"n_train": ntr, "p": {}, "h": {}, "late": {}}
        for te in range(8, 16):
            r["p"][te] = evaluate(clf, "p", te, nonlate, "p"); r["h"][te] = evaluate(clf, "h", te, lambda m: True, "h")
            r["late"][te] = evaluate(clf, "p", te, late_obj, "late")
        res[f"{src}@t{tt}"] = r
        print(f"[{o}/{LT}] train {src}@t{tt} (n={ntr}) → p: " + " ".join(f"{r['p'][te]['acc']:4.0f}" for te in range(8, 16))
              + " | h: " + " ".join(f"{r['h'][te]['acc']:4.0f}" for te in range(8, 16)) + f" ({time.time() - t0:.0f}s)", flush=True)
        del clf; torch.cuda.empty_cache()
(OUT / "runs").mkdir(parents=True, exist_ok=True)
json.dump(dict(objset=o, label=LT, seed=SEED, classes=classes, opt=dict(epochs=40, batch_size=16, lr=1e-4, weight_decay=0.05), res=res),
          open(OUT / "runs" / f"tempgen_{o}_{LT}_s{SEED}.json", "w"), indent=1)
print("done", flush=True)
