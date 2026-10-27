"""문맥 (teacher forcing) 구간에서도 물체의 속도 · 방향 · 정체가 encoder 처럼 남나 — TF 한 걸음 vs 굴린 예측 (2026-10-26, 학습 없음).
사용자: "context 부분도 같이 봐야겠지? tf 구간이니까. 여기선 속도 어떻게 되는지 봐야될 듯".

판독기 · 부분공간 = 3D 합성 학습 쪽 4,000 편의 물체 칸 encoder 토큰 (extract_objtok.py, 블록 공간 = 블록마다 LN(target_encoder(2 장))) 에서, 시나리오 평균을 뺀 뒤 맞춘다.
  방향 8 구간 (블록 안 변위 ≥ 2 px 인 칸) ridge 분류 · 속도 (vx, vy) ridge 회귀 · 정체 평균 차 부분공간 (48 정체, 상위 10)
시험 = val 96 장 영상 350 편 (pred_objvec 과 같은 목록). 기준 AR natural_ar_vw_state latest.
  gt   블록 b 정답 물체 칸 토큰
  tf   teacher forcing 한 걸음 예측 (블록 0..b−1 정답을 보고 블록 b) — b = 1..15. 문맥 구간 = b 1..7, 미래 구간 = b 8..15
  roll 굴린 예측 (문맥 블록 0..7 → 블록 8..15)
지표 (블록 자리 · 출처별, 보이는 물체 칸):
  dir_acc       방향 8 구간 정확도 (움직이는 칸만, 우연 12.5)
  vel_gain      예측에서 읽은 (vx, vy) 를 실제 (vx, vy) 에 원점 회귀한 기울기 (1 = 속도 유지, 0 = 멈춤). gt 가 판독 상한
  vel_r2        같은 판독의 R²
  id_cos        정체 부분공간 좌표 cos(예측, 같은 블록 정답) (gt 는 1)

  CUDA_VISIBLE_DEVICES=<여유 있는 장> /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/tf_context_motion.py
출력: auto_research/results/tf_context_motion.json
"""
import glob, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
BIG = "/workspace/auto_research_cache/objtok_train"; VALMETA = "/workspace/auto_research_cache/pred_objvec/val_meta.json"
OUT = os.path.join(REPO, "auto_research/results/tf_context_motion.json")
IDX = [3 * k for k in range(32)]; S = 256; D = 1280; NBC = 8
dev = torch.device("cuda"); rng = np.random.default_rng(0)
# 2026-10-26: 다른 세션 학습이 장당 ~84 / 98 GB 를 쓴다 → 이 프로세스 GPU 메모리 상한 (기본 6 GB)
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 6)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)


def ridge(X, Y, lam=1e2):
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    return vt.T @ ((s / (s ** 2 + lam))[:, None] * (u.T @ Y))


def dirbin(d, thr=2.0):
    s = np.hypot(d[:, 0], d[:, 1])
    return np.where(s >= thr, ((np.arctan2(d[:, 1], d[:, 0]) + np.pi) / (2 * np.pi / 8)).astype(int) % 8, -1)


# ---- 판독기 · 부분공간 (train)
X, lab, scn_all = [], [], []
for f in sorted(glob.glob(f"{BIG}/shard*.npz")):
    z = np.load(f, allow_pickle=True); sh = int(os.path.basename(f)[5:-4]); scn = z["scen"]; L = z["lab_ar"]
    X.append(z["tok_ar"].astype(np.float32)); lab.append(L); scn_all += [str(scn[int(r[0]) - sh * 100000]) for r in L]
X = np.concatenate(X); L = np.concatenate(lab); scn_all = np.array(scn_all)
smu = {s: X[scn_all == s].mean(0) for s in np.unique(scn_all)}
Xc = X - np.stack([smu[s] for s in scn_all]); del X
mu, sd = Xc.mean(0), Xc.std(0) + 1e-6
sel = rng.choice(len(Xc), min(len(Xc), 60000), replace=False)
Xs = (Xc[sel] - mu) / sd; d_tr = L[sel, 5:7]; db = dirbin(d_tr); mv = db >= 0
Wdir = ridge(Xs[mv], np.eye(8)[db[mv]] - np.eye(8)[db[mv]].mean(0)); bdir = np.eye(8)[db[mv]].mean(0)
Wvel = ridge(Xs, d_tr - d_tr.mean(0)); bvel = d_tr.mean(0)
idc = (L[:, 1] * 8 + L[:, 2]).astype(int)
M = np.stack([Xc[idc == k].mean(0) for k in range(48) if (idc == k).any()]) - Xc.mean(0)
Qid = np.linalg.svd(M, full_matrices=False)[2][:10].T
log("판독기 학습 표본", len(sel), "움직임", int(mv.sum()))
T = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(dev)
Wdir_t, bdir_t, Wvel_t, bvel_t, Qid_t, mu_t, sd_t = map(T, (Wdir, bdir, Wvel, bvel, Qid, mu, sd))

# ---- 모델 · 데이터
import decord
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FTm
sys.argv = _argv
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.to(dev).eval()
ar = FTm.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
LN = lambda t: F.layer_norm(t, (t.size(-1),))
vmeta = json.load(open(VALMETA))
idx = torch.arange(16 * S, device=dev).unsqueeze(0)
recs = []                                                                          # (출처, 블록, 참 vx, vy, 읽은 vx, vy, 방향 정답, 방향 예측, id_cos)
for i, m in enumerate(vmeta):
    f = m["clip"]; st = np.load(f[:-4] + "_states.npz"); pos = st["pos"][:, 0]; sc = f.split("/")[-2]
    fr = decord.VideoReader(f, num_threads=4).get_batch(IDX).asnumpy()
    xn = (torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0 - MEAN) / STD
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        hb = LN(enc2(xn.view(3, 16, 2, 256, 256).permute(1, 0, 2, 3, 4)).float())                  # (16, S, D)
        tf = ar.forward_seq(hb[:15].reshape(1, -1, D), idx[:, : 15 * S]).float()[0].view(15, S, D)   # 위치 b → 블록 b+1
        ro = ar.rollout_kv(hb[:NBC].reshape(1, -1, D), idx[:, : NBC * S], 16 - NBC).float()[0].view(16 - NBC, S, D)
    smu_c = T(smu.get(sc, mu))
    for b in range(16):
        Lb = m["labels"][b]
        if not (Lb["ok"] and Lb["cells"]):
            continue
        cells = Lb["cells"]; d = pos[IDX[2 * b + 1]] - pos[IDX[2 * b]]
        db_true = int(dirbin(d[None])[0])
        srcs = [("gt", hb[b, cells])]
        if b >= 1:
            srcs.append(("tf", tf[b - 1, cells]))
        if b >= NBC:
            srcs.append(("roll", ro[b - NBC, cells]))
        g = ((hb[b, cells] - smu_c - mu_t) / sd_t)
        for nm, tok in srcs:
            x = ((tok - smu_c - mu_t) / sd_t)
            v = (x @ Wvel_t + bvel_t).mean(0).cpu().numpy()                             # 물체 칸 평균 판독
            dp = int((x @ Wdir_t + bdir_t).mean(0).argmax())
            idcos = float(F.cosine_similarity(((tok - smu_c) - mu_t) @ Qid_t, ((hb[b, cells] - smu_c) - mu_t) @ Qid_t, dim=-1).mean())
            recs.append((nm, b, float(d[0]), float(d[1]), float(v[0]), float(v[1]), db_true, dp, idcos))
    if i % 50 == 0:
        log(f"{i}/{len(vmeta)}")

R = np.array([r[1:] for r in recs], np.float64); src = np.array([r[0] for r in recs])
def summ(mask):
    A = R[mask]
    if len(A) < 10:
        return None
    tv, pv = A[:, 1:3], A[:, 3:5]
    gain = float((tv * pv).sum() / (tv * tv).sum())
    r2 = float(1 - ((pv - tv) ** 2).sum() / ((tv - tv.mean(0)) ** 2).sum())
    mvm = A[:, 5] >= 0
    acc = float((A[mvm, 5] == A[mvm, 6]).mean() * 100) if mvm.any() else float("nan")
    return {"n": int(len(A)), "vel_gain": round(gain, 3), "vel_r2": round(r2, 3), "dir_acc": round(acc, 1), "id_cos": round(float(np.median(A[:, 7])), 3)}
out = {"by_block": {}, "groups": {}}
for nm in ("gt", "tf", "roll"):
    out["by_block"][nm] = {int(b): summ((src == nm) & (R[:, 0] == b)) for b in range(16)}
    out["groups"][nm] = {"context_b1_7": summ((src == nm) & (R[:, 0] >= 1) & (R[:, 0] <= 7)), "future_b8_15": summ((src == nm) & (R[:, 0] >= 8))}
json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
for nm in ("gt", "tf", "roll"):
    print(nm, out["groups"][nm])
for b in (1, 4, 7, 8, 9, 11, 13, 15):
    print("block", b, {nm: out["by_block"][nm].get(b) for nm in ("gt", "tf", "roll")})
log("saved", OUT)
