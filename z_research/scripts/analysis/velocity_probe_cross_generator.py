#!/usr/bin/env python3
"""문맥 토큰의 속도 신호가 생성기를 넘어 같은 방식으로 읽히나 (2026-10-31, 사용자: "상대 표현 괜찮은데" → "일단 측정해봐").

왜: 보조 모듈 (release_aux_perm_sup) 은 phys3d 안에서는 물체를 따라가는데 VOBench 에서는 "문맥 끝 그대로" 수준이다
  (z_analysis/results/aux_occ_generalization). 상대 표현 (마지막 자리에서 얼마나 움직였나) 으로 감독하면 넘어갈지는
  입력 (문맥 encoder 토큰) 의 속도 신호가 생성기에 무관한지에 달려 있다.
무엇: 문맥 encoder (릴리즈 `encoder`, 모듈 입력과 같은 경로) 의 블록 5 · 6 (장 10–13) 토큰을 물체 상자 칸에서 평균 → (2 × 1280)
  → 선형 ridge 로 물체 속도 (vx, vy) [px / 장, 256 해상도, 장 10 → 12 중심 이동 / 2] 를 읽는다. 장 10–13 모두 보이는 물체만.
  ① phys3d 로 맞춤 → phys3d held-out (같은 생성기) · VOBench late (처음 보는 생성기) 에 그대로
  ② VOBench 안에서 맞춤 (5-fold, 같은 영상의 두 팔은 같은 fold) = VOBench 토큰에 속도가 선형으로 있는가의 상한
  점수: R² (축별, 0 = 평균 예측) · 피어슨 r (축별, 크기 · 치우침과 무관) · 방향 코사인 (움직이는 것, |v| > 1 px/장)
  phys3d 표준화 통계를 VOBench 에도 그대로 쓴다 (정직한 이식).
읽는 법: ① 의 VOBench R² 가 ② 에 가까우면 → 속도 신호가 생성기와 무관 → 상대 이동량 감독이 넘어갈 근거가 있다.
  ① ≪ ② 이면 → 토큰에 속도는 있지만 생성기마다 읽는 방향이 다르다 → phys3d 하나로 배운 규칙은 안 넘어간다 (데이터 다양화가 먼저).

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/velocity_probe_cross_generator.py
출력: z_analysis/results/velocity_probe_cross_generator/summary.json
"""
import argparse, csv, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
ap = argparse.ArgumentParser()
ap.add_argument("--n_train", type=int, default=2000, help="phys3d 학습 쪽 clip 수 (맞춤용)")
ap.add_argument("--mem", type=float, default=0.25)
a = ap.parse_args()
OUT = os.path.join(REPO, "z_analysis/results/velocity_probe_cross_generator"); os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(a.mem)
from app.vjepa_frozen import utils as U
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
MD = dict(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, use_rope=True, uniform_power=False,
          context_encoder_key="encoder", target_encoder_key="encoder")
ctx_enc, _ = U.build_frozen_encoders(MD, U.load_base_checkpoint(CK), dev, torch.bfloat16, 32)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
NF, G, S, C = 32, 16, 256, 16
me = torch.arange(C // 2 * S, device=dev)[None]


def cells(x0, y0, x1, y1):
    return [r * G + c for r in range(max(0, int(y0 // 16)), min(G, int(y1 // 16) + 1)) for c in range(max(0, int(x0 // 16)), min(G, int(x1 // 16) + 1))]


def feats(fr, objs):
    """fr (32, 256, 256, 3) uint8, objs = [(box10, box12, v)] → [(2560 특징, v)]."""
    x = ((torch.from_numpy(fr).to(dev).permute(3, 0, 1, 2)[None].float().div_(255) - MEAN) / STD).to(torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = ctx_enc(x, masks=[me]).float()[0].view(C // 2, S, -1)            # (8 블록, 256, 1280)
    out = []
    for b10, b12, v in objs:
        c5, c6 = cells(*b10), cells(*b12)
        if not c5 or not c6: continue
        out.append((torch.cat([z[5, c5].mean(0), z[6, c6].mean(0)]).cpu().numpy(), v))
    return out


def phys3d(split, n):
    import decord
    ps = sorted(glob.glob(f"/workspace/dataset/training_data/phys3d_perm_v1/{split}/clips/*/*.mp4"))
    ps = [ps[i] for i in np.random.default_rng(0).permutation(len(ps))[:n]]
    X, Y, M = [], [], []
    for k, p in enumerate(ps):
        s = np.load(p[:-4] + "_states.npz"); vis = (s["vis_frac"] > 0.5) & (s["inframe_frac"] > 0.5); b = s["bbox"]
        objs = []
        for o in range(vis.shape[1]):
            if vis[10:14, o].all():
                c10 = ((b[10, o, 0] + b[10, o, 2]) / 2, (b[10, o, 1] + b[10, o, 3]) / 2); c12 = ((b[12, o, 0] + b[12, o, 2]) / 2, (b[12, o, 1] + b[12, o, 3]) / 2)
                objs.append((tuple(b[10, o]), tuple(b[12, o]), ((c12[0] - c10[0]) / 2, (c12[1] - c10[1]) / 2)))
        if not objs: continue
        for f_, v in feats(decord.VideoReader(p, num_threads=2).get_batch(list(range(NF))).asnumpy(), objs):
            X.append(f_); Y.append(v); M.append(p.split("/")[-2])
        if k % 500 == 0: print(f"phys3d {split} {k}/{len(ps)} · 표본 {len(X)}", flush=True)
    return np.array(X), np.array(Y), M


def vobench():
    from PIL import Image
    R = "/workspace/dataset/benchmarks/VOBench_v1"
    X, Y, M, grp = [], [], [], []
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1_late/index.csv"))) if r["variant"] in ("pos_a", "pos_b")]
    for k, r in enumerate(rows):
        arm = "A" if r["variant"] == "pos_a" else "B"
        if (r["obj_a"] if arm == "A" else r["obj_b"]) == "empty": continue
        L = json.load(open(os.path.join(R, os.path.dirname(r["file_name"]), "label.json")))["labels"][arm]
        if any(L[3 * j]["hidden"] >= 0.5 for j in range(10, 14)): continue
        bx = lambda j: tuple(np.array([L[3 * j]["u"] - L[3 * j].get("ext_u", L[3 * j]["r_px"]), L[3 * j]["v"] - L[3 * j].get("ext_v", L[3 * j]["r_px"]),
                                       L[3 * j]["u"] + L[3 * j].get("ext_u", L[3 * j]["r_px"]), L[3 * j]["v"] + L[3 * j].get("ext_v", L[3 * j]["r_px"])]) / 2)
        v = ((L[36]["u"] - L[30]["u"]) / 4, (L[36]["v"] - L[30]["v"]) / 4)        # (장 12 − 장 10) / 2, 512 → 256
        fr = np.stack([np.asarray(Image.open(os.path.join(R, r["file_name"], f"{3 * j:06d}.png")).convert("RGB")) for j in range(NF)])
        fr = F.interpolate(torch.from_numpy(fr).permute(0, 3, 1, 2).float(), size=(256, 256), mode="bilinear",
                           align_corners=False).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
        for f_, vv in feats(fr, [(bx(10), bx(12), v)]):
            X.append(f_); Y.append(vv); M.append(r["motion"]); grp.append(r["source_block"])
        if k % 100 == 0: print(f"vobench {k}/{len(rows)} · 표본 {len(X)}", flush=True)
    return np.array(X), np.array(Y), M, grp


def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean(0)))
    return lambda Xn: ((Xn - mu) / sd) @ W + Y.mean(0)


def score(Yt, Yp):
    r2 = [1 - ((Yt[:, i] - Yp[:, i]) ** 2).sum() / ((Yt[:, i] - Yt[:, i].mean()) ** 2).sum() for i in range(2)]
    r = [float(np.corrcoef(Yt[:, i], Yp[:, i])[0, 1]) for i in range(2)]
    mv = np.linalg.norm(Yt, axis=1) > 1.0
    cos = float(np.mean((Yt[mv] * Yp[mv]).sum(1) / (np.linalg.norm(Yt[mv], axis=1) * np.linalg.norm(Yp[mv], axis=1) + 1e-9))) if mv.any() else float("nan")
    return dict(r2_x=float(r2[0]), r2_y=float(r2[1]), r_x=r[0], r_y=r[1], cos_moving=cos, n=int(len(Yt)), n_moving=int(mv.sum()))


Xtr, Ytr, _ = phys3d("train", a.n_train)
Xte, Yte, Mte = phys3d("val", 876)
Xv, Yv, Mv, Gv = vobench()
print(f"표본: phys3d 맞춤 {len(Xtr)} · phys3d held-out {len(Xte)} · VOBench {len(Xv)}", flush=True)
# λ: phys3d held-out 의 절반으로 고르고 나머지 절반 + VOBench 에 보고
half = np.arange(len(Xte)) % 2 == 0
best = max([10.0 ** k for k in range(0, 6)], key=lambda lam: np.mean([score(Yte[half], ridge_fit(Xtr, Ytr, lam)(Xte[half]))[k] for k in ("r2_x", "r2_y")]))
P = ridge_fit(Xtr, Ytr, best)
res = {"lambda": best, "phys3d_heldout": score(Yte[~half], P(Xte[~half])), "vobench_from_phys3d": score(Yv, P(Xv))}
# VOBench 안 맞춤 (5-fold, source_block 단위) — λ 는 같은 격자에서 fold 평균 최고
ug = sorted(set(Gv)); fold = {g: i % 5 for i, g in enumerate(np.random.default_rng(0).permutation(ug))}
fid = np.array([fold[g] for g in Gv])
def cv(lam):
    Yp = np.zeros_like(Yv)
    for f in range(5): Yp[fid == f] = ridge_fit(Xv[fid != f], Yv[fid != f], lam)(Xv[fid == f])
    return Yp
lam_v = max([10.0 ** k for k in range(0, 6)], key=lambda lam: np.mean([score(Yv, cv(lam))[k] for k in ("r2_x", "r2_y")]))
res["vobench_within_cv"] = {**score(Yv, cv(lam_v)), "lambda": lam_v}
res["vobench_from_phys3d_by_motion"] = {m: score(Yv[np.array(Mv) == m], P(Xv[np.array(Mv) == m])) for m in sorted(set(Mv))}
res["target_std"] = {"phys3d": Ytr.std(0).tolist(), "vobench": Yv.std(0).tolist()}
json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
f = lambda d: f"R² x {d['r2_x']:+.3f} y {d['r2_y']:+.3f} · r x {d['r_x']:.3f} y {d['r_y']:.3f} · 방향 cos {d['cos_moving']:.3f} (n {d['n']}, 움직임 {d['n_moving']})"
print(f"\nλ phys3d {best:g} · VOBench 안 {lam_v:g} · 속도 표준편차 phys3d {np.round(Ytr.std(0), 2)} · VOBench {np.round(Yv.std(0), 2)} px/장")
print(f"  ① phys3d → phys3d held-out : {f(res['phys3d_heldout'])}")
print(f"  ① phys3d → VOBench          : {f(res['vobench_from_phys3d'])}")
print(f"  ② VOBench 안 5-fold (상한)   : {f(res['vobench_within_cv'])}")
for m, d in res["vobench_from_phys3d_by_motion"].items():
    print(f"     VOBench {m:12s} (phys3d 맞춤) {f(d)}")
