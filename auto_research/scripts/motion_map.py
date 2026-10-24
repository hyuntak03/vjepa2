#!/usr/bin/env python3
"""[물리] 실제 영상 움직임 예측 가능성 지도 — 출처 × 간격 × 문맥 모형 (흐름, 학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 /venv/vjepa/bin/python -W ignore auto_research/scripts/motion_map.py [--n 120] [--limit 0]

사전 등록: 진행판 [물리] 09:18 UTC (KST 18:18), 리더 배정 KST 18:15.
출처: SSv2 수동 물리 23 템플릿 · SSv2 행위자 18 템플릿 · K710 공/던지기 49 라벨 · K710 몸/중력 22 라벨 · HT100M · 3D 합성 (기준).
창: 가운데 3.5 초 (HT100M · 3D 는 기존 시작 장). 간격 Δt = 1/16 · 1/8 · 1/4 · 1/2 초 → 장 간격 max(1, round(Δt·fps)).
흐름 U[i] = RAFT(장 i+1 → 장 i) 16 px 칸 평균 (칸 단위). 카메라 = cam_state.fit_cam 강건 호모그래피.
움직이는 칸: |다음 움직임| ≥ 실제 Δt 칸 (초당 1 칸). 판 (가) 카메라 정지 걸음 (네 흐름 모두 카메라 < 문턱), 흐름 그대로 / (나) 모든 걸음, 흐름 − 카메라.
내용 따라가기: c → s1 = c + U[t+1](c) → s2 = s1 + U[t](s1) → s3 = s2 + U[t−1](s2). Y = V[t+1](c), X1 = V[t](s1), X2 = V[t−1](s2), X3 = V[t−2](s3).
모형: 제자리 0 · 등속 X1 · 등가속 2X1 − X2 · ridge [X1, X2, X3, 1] (짝수 창 맞춤, 맞춤 쪽 5 겹 λ, 홀수 창 평가).
지표: R² = 1 − Σ|Y−Ŷ|²/Σ|Y−Ȳ|² · 방향 45° 안 · 칸 몫 (움직이는 칸 ÷ 그 판의 유효 걸음 전체 칸)."""
import argparse, collections, csv, glob, json, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--cap", type=int, default=64, help="걸음마다 저장하는 움직이는 칸 최대 (칸 몫은 전체로 셈)")
ap.add_argument("--out", default="/workspace/world_model/vjepa2/auto_research/results/motion_map.json")
a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--parts", "", "--limit", "1"]
import cam_state as CS                                                          # fit_cam · cam_mag (부분 실행 없음)
sys.argv = _argv
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
torch.set_num_threads(8)
dev = torch.device("cuda")
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights      # noqa: E402
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
G, R, S = 16, 256, 256
DTS = (1 / 16, 1 / 8, 1 / 4, 1 / 2)
SPAN = 3.5
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)
rng = random.Random(0)

PASSIVE = ["Dropping something behind something", "Dropping something in front of something", "Dropping something into something",
           "Dropping something next to something", "Dropping something onto something",
           "Letting something roll along a flat surface", "Letting something roll down a slanted surface",
           "Letting something roll up a slanted surface, so it rolls back down", "Lifting something up completely, then letting it drop down",
           "Something falling like a feather or paper", "Something falling like a rock",
           "Throwing something", "Throwing something against something", "Throwing something in the air and letting it fall", "Throwing something onto a surface",
           "Rolling something on a flat surface", "Something colliding with something and both are being deflected", "Something being deflected from something",
           "Pushing something so that it falls off the table", "Putting something on the edge of something so it is not supported and falls down",
           "Putting something onto something else that cannot support it so it falls down",
           "Lifting a surface with something on it until it starts sliding down", "Putting something that can't roll onto a slanted surface, so it slides down"]
ACTOR = ["Moving something up", "Moving something down", "Moving something towards the camera", "Moving something away from the camera",
         "Moving something closer to something", "Moving something away from something",
         "Pushing something from left to right", "Pushing something from right to left", "Pulling something from left to right", "Pulling something from right to left",
         "Moving something and something closer to each other", "Moving something and something away from each other",
         "Lifting something up completely without letting it drop down",
         "Holding something", "Holding something next to something", "Holding something in front of something", "Holding something behind something",
         "Holding something over something"]
BALL = ['bouncing ball (not juggling)', 'bowling', 'catching or throwing baseball', 'catching or throwing frisbee', 'catching or throwing softball', 'curling (sport)', 'disc golfing',
        'dodgeball', 'dribbling basketball', 'drop kicking', 'dunking basketball', 'golf chipping', 'golf driving', 'golf putting', 'hammer throw', 'hitting baseball', 'hurling (sport)',
        'javelin throw', 'juggling balls', 'juggling soccer ball', 'kicking field goal', 'kicking soccer ball', 'passing American football (in game)',
        'passing American football (not in game)', 'passing soccer ball', 'playing badminton', 'playing basketball', 'playing billiards', 'playing cricket', 'playing field hockey',
        'playing ice hockey', 'playing kickball', 'playing marbles', 'playing netball', 'playing ping pong', 'playing polo', 'playing road hockey', 'playing shuffleboard',
        'playing squash or racquetball', 'playing tennis', 'playing volleyball', 'shooting basketball', 'shot put', 'skipping stone', 'throwing ball',
        'throwing ball (not baseball or American football)', 'throwing discus', 'throwing snowballs', 'throwing water balloon']
JUMP = ['backflip (human)', 'base jumping', 'bouncing on trampoline', 'bungee jumping', 'cartwheeling', 'diving cliff', 'faceplanting', 'falling off bike', 'falling off chair',
        'gymnastics tumbling', 'high jump', 'hurdling', 'jumping into pool', 'long jump', 'parkour', 'pole vault', 'ski jumping', 'skydiving', 'somersaulting', 'springboard diving',
        'triple jump', 'vault']
SHARE = {}


def build_lists():
    L = {}
    lab = {}
    for split in ("train", "validation"):
        for x in json.load(open(f"/workspace/dataset/training_data/SSv2/labels/{split}.json")):
            lab[x["id"]] = x["template"].replace("[", "").replace("]", "")
    tab = [l.rstrip("\n").split("\t") for l in open(os.path.join(REPO, "data_csv/ssv2/fps4_table.tsv"))]
    ids = [(r[0], lab.get(os.path.basename(r[0])[:-5])) for r in tab if int(r[2]) >= 30]
    for g, cl in (("ssv2_passive", PASSIVE), ("ssv2_actor", ACTOR)):
        cs = set(cl); rows = [p for p, t in ids if t in cs]
        SHARE[g] = len(rows) / max(1, sum(1 for _, t in ids if t))
        random.Random(0).shuffle(rows); L[g] = [(p, None) for p in rows[: a.n]]
    kl = {}
    for f in glob.glob("/workspace/dataset/training_data/K710/source_annotations/*/*.csv"):
        for r in csv.DictReader(open(f)):
            kl[r["youtube_id"]] = r["label"]
    tab = [l.rstrip("\n").split("\t") for l in open(os.path.join(REPO, "data_csv/k710_320/fps4_table.tsv"))]
    lk = [(r[0], kl.get(os.path.basename(r[0])[:11])) for r in tab if int(r[2]) >= int(r[1]) * 3]
    for g, cl in (("k710_ball", BALL), ("k710_jump", JUMP)):
        cs = set(cl); rows = [p for p, t in lk if t in cs]
        SHARE[g] = len(rows) / max(1, len(lk))
        random.Random(0).shuffle(rows); L[g] = [(p, None) for p in rows[: a.n]]
    L["ht100m"] = [(m["path"], m["start"]) for m in json.load(open("/workspace/auto_research_cache/natural_train/ht100m_1pct_meta.json"))[: a.n]]
    ms = sum([json.load(open(f"/workspace/auto_research_cache/phys3d/meta_val_s{k}.json")) for k in range(4)], [])
    random.Random(0).shuffle(ms)
    L["phys3d"] = [(os.path.join("/workspace/dataset/training_data/phys3d_single_v1", m["path"]), int(m.get("r0", 0))) for m in ms[: a.n]]
    if a.limit:
        L = {k: v[: a.limit] for k, v in L.items()}
    return L


def prep(fr):
    Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
    x = torch.from_numpy(np.ascontiguousarray(fr[:, y0:y0 + side, x0:x0 + side])).to(dev).permute(0, 3, 1, 2).float()
    if side != R:
        x = F.interpolate(x, size=(R, R), mode="bilinear", align_corners=False).clamp(0, 255).round()
    return x / 127.5 - 1


@torch.no_grad()
def cell_flow(x):                                                               # (n,3,R,R) → (n−1, 256, 2): U[i] = 장 i+1 → 장 i
    out = []
    for k in range(1, x.shape[0], 16):
        b_ = x[k:k + 16]; a_ = x[k - 1:k - 1 + b_.shape[0]]
        out.append(raft(b_, a_, num_flow_updates=12)[-1])
    f = torch.cat(out)
    return (F.avg_pool2d(f, 16) / 16).permute(0, 2, 3, 1).reshape(f.shape[0], S, 2).cpu().numpy().astype(np.float64)


def load(path, start):
    vr = decord.VideoReader(path, num_threads=2)
    fps = vr.get_avg_fps(); n = int(round(SPAN * fps)) + 1
    if start is None:
        start = max(0, (len(vr) - n) // 2)
    if start + n > len(vr):
        start = max(0, len(vr) - n)
    idx = list(range(start, min(len(vr), start + n)))
    return vr.get_batch(idx).asnumpy(), fps


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), ok


DATA = {}                                                                     # (src, dt, mode) → dict
DTR = collections.defaultdict(list)
L = build_lists()
print("데이터 몫 (영상):", {k: round(v, 4) for k, v in SHARE.items()}, "· 창 수:", {k: len(v) for k, v in L.items()}, flush=True)
t0 = time.time(); nv = 0
for src, items in L.items():
    for wi, (path, start) in enumerate(items):
        try:
            fr, fps = load(path, start)
        except Exception as e:
            print("skip", path, e, flush=True); continue
        x_all = prep(fr); nv += 1
        for dt in DTS:
            step = max(1, int(round(dt * fps))); dtr = step / fps
            x = x_all[::step]
            if x.shape[0] < 5:
                continue
            U = cell_flow(x)
            DTR[(src, dt)].append(dtr)
            thr = dtr                                                           # 초당 1 칸
            Hs = [CS.fit_cam(U[i])[0] for i in range(U.shape[0])]
            Fc = np.stack([CS.cam_flow(H) for H in Hs]); mag = np.array([CS.cam_mag(H) for H in Hs])
            Rr = U - Fc
            for mode in ("still", "camrm"):
                d = DATA.setdefault((src, dt, mode), dict(Y=[], X1=[], X2=[], X3=[], win=[], n_obj=0, n_all=0))
                V = U if mode == "still" else Rr
                for t in range(2, U.shape[0] - 1):
                    if mode == "still" and not (mag[t - 2:t + 2] < thr).all():
                        continue
                    d["n_all"] += S
                    y = V[t + 1]; mov = np.linalg.norm(y, axis=1) >= thr
                    s1, o1 = cell_of(P0 + U[t + 1]); s2, o2 = cell_of(P0[s1] + U[t][s1]); s3, o3 = cell_of(P0[s2] + U[t - 1][s2])
                    sel = np.where(mov & o1 & o2 & o3)[0]
                    d["n_obj"] += int(mov.sum())
                    if sel.size == 0:
                        continue
                    if sel.size > a.cap:
                        sel = np.random.RandomState(wi * 1000 + t).choice(sel, a.cap, replace=False)
                    d["Y"].append(y[sel]); d["X1"].append(V[t][s1[sel]]); d["X2"].append(V[t - 1][s2[sel]]); d["X3"].append(V[t - 2][s3[sel]])
                    d["win"].append(np.full(sel.size, wi))
        if (wi + 1) % 20 == 0:
            print(f"{src} {wi + 1}/{len(items)} · {time.time() - t0:.0f}s", flush=True)


def metrics(Y, P):
    r2 = 1 - ((Y - P) ** 2).sum() / max(((Y - Y.mean(0)) ** 2).sum(), 1e-12)
    cs = (Y * P).sum(1) / (np.linalg.norm(Y, axis=1) * np.linalg.norm(P, axis=1) + 1e-12)
    return float(r2), float((cs > 0.7071).mean())


def ridge_fit(X, Y, lam):
    A = X.T @ X + lam * np.diag([1, 1, 1, 1, 1, 1, 0]); return np.linalg.solve(A, X.T @ Y)


res = {"share_videos": SHARE, "n_videos": nv}
for (src, dt, mode), d in DATA.items():
    if not d["Y"]:
        continue
    Y, X1, X2, X3, win = (np.concatenate(d[k]) for k in ("Y", "X1", "X2", "X3", "win"))
    te = win % 2 == 1; tr = ~te
    Xr = np.c_[X1, X2, X3, np.ones(len(Y))]
    e = dict(dt_real=float(np.mean(DTR[(src, dt)])), n_cells_saved=int(len(Y)), n_test=int(te.sum()), cell_share=d["n_obj"] / max(d["n_all"], 1), n_all_cells=int(d["n_all"]),
             mean_disp=float(np.linalg.norm(Y, axis=1).mean()))
    preds = {"stay": np.zeros_like(Y), "constvel": X1, "constacc": 2 * X1 - X2}
    if te.sum() < 100:                                                          # 칸이 거의 없으면 R² 가 0/0 으로 1 이 된다 — 재지 않는다 (판정에서 빠짐)
        res[f"{src}|{dt:.4f}|{mode}"] = e
        print(f"{src:13s} Δt {dt:.4f} {mode:5s} 칸 몫 {e['cell_share']:.3f} n {e['n_test']} — 평가 칸 < 100, 건너뜀", flush=True)
        continue
    for nm, P in preds.items():
        e[f"{nm}_all"] = metrics(Y, P); e[nm] = metrics(Y[te], P[te])
    if tr.sum() > 50 and te.sum() > 50:
        f5 = win[tr] // 2 % 5
        lams = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
        sc = []
        for lam in lams:
            ss = 0.0
            for k in range(5):
                m = f5 != k
                if m.sum() < 20 or (~m).sum() < 20:
                    continue
                W = ridge_fit(Xr[tr][m], Y[tr][m], lam * m.sum() / 1000); ss += ((Y[tr][~m] - Xr[tr][~m] @ W) ** 2).sum()
            sc.append(ss)
        lam = lams[int(np.argmin(sc))]
        W = ridge_fit(Xr[tr], Y[tr], lam * tr.sum() / 1000)
        e["ridge"] = metrics(Y[te], Xr[te] @ W); e["ridge_lambda"] = lam
        e["ridge_W_X1"] = W[:2].tolist(); e["ridge_W_X2"] = W[2:4].tolist()
    res[f"{src}|{dt:.4f}|{mode}"] = e
    print(f"{src:13s} Δt {dt:.4f} (실제 {e['dt_real']:.3f}) {mode:5s} 칸 몫 {e['cell_share']:.3f} n {e['n_test']:6d} | R² · 45°: "
          + " · ".join(f"{nm} {e[nm][0]:+.2f}/{e[nm][1]:.2f}" for nm in ("stay", "constvel", "constacc", "ridge") if nm in e), flush=True)
best = []
for k, e in res.items():
    if not isinstance(e, dict) or "|" not in k:
        continue
    for nm in ("constvel", "constacc", "ridge"):
        if nm in e:
            best.append((e[nm][0], k, nm, e["cell_share"]))
best.sort(reverse=True)
res["top"] = best[:20]
nat_hits = [b for b in best if not b[1].startswith("phys3d") and b[0] >= 0.5 and b[3] >= 0.10]
res["verdict_hits"] = nat_hits
print("\n자연 출처에서 R² ≥ 0.5 이고 칸 몫 ≥ 10 % 인 칸:", nat_hits if nat_hits else "없음")
print("상위:", [(round(b[0], 3), b[1], b[2], round(b[3], 3)) for b in best[:12]])
json.dump(res, open(a.out, "w"), indent=1)
print("저장", a.out, f"{time.time() - t0:.0f}s")
