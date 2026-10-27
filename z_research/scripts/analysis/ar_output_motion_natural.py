"""학습 영상에서 AR 출력이 다음 블록의 블록 안 움직임을 얼마나 담나 — 갈래 (a) vs (b) 가르기 (2026-10-26, GPU 1 장 · 6 GB).
사용자: "해보던가". 앞 측정 motion_predictability_flow.json — 학습 영상에서 다음 블록 안 움직임은 지금 움직임의 0.06–0.20 만 이어진다.
  (a) 모델이 학습 영상의 최적 (조건부 중앙값) 에 있어서 출력에 움직임이 적다  vs  (b) 문맥으로 정해지는 움직임도 출력이 버린다.
가르는 법 (판독기 = 칸 토큰 하나 (1280) → 그 칸 블록 안 RAFT 흐름 (dx, dy) 선형, AdamW · MSE, 입력 표준화, 영상 절반으로 학습 · 나머지 절반 시험):
  자기 블록   : 정답 블록 encoder 토큰 → 그 블록 흐름                      (판독기 검증 · 상한)
  지금 블록   : 직전 블록 encoder 토큰 → 다음 블록 흐름                    (문맥 하나로 다음 움직임을 얼마나 아나 = 하한)
  AR 출력     : 그 블록을 예측한 AR 출력 토큰 → 그 블록 흐름              (출력이 담은 다음 움직임)
  이식        : 자기 블록 판독기를 AR 출력에 그대로 → 출력이 담은 움직임의 크기 (같은 판독기로 정답 블록에서 읽은 크기와 비교)
  (a) 면 AR 출력 R² ≳ 지금 블록 R² (최적 출력은 문맥 전체가 아는 만큼 담는다), (b) 면 AR 출력 R² < 지금 블록 R².
데이터: K710 (학습에 쓴 영상 — 학습 분포에서 최적에 닿았나를 묻는 것이라 맞다). fps4 표와 같은 장 · 가운데 정사각 → 256² (흐름 추출과 같은 화면).
  영상마다 연속 28 장 = 블록 14 (블록 b = 장 2b, 2b+1, 블록 안 흐름 = f[2b]). teacher forcing 대상 블록 2–13 (자리 1–12 출력, 문맥 자리 포함), 굴림 = 문맥 8 블록 → 블록 8–13 (걸음 1–6).
  대상 블록마다 칸 64 개 무작위. 움직이는 칸 = |지금 블록 흐름| ≥ 2 px (0.25 s 당).
  CUDA_VISIBLE_DEVICES=0 MEM_GB=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/ar_output_motion_natural.py --n 1500
출력: z_analysis/results/ar_output_motion_natural.json · 캐시 /workspace/auto_research_cache/ar_motion_natural/
"""
import argparse, glob, json, os, random, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1500); ap.add_argument("--cells", type=int, default=64)
ap.add_argument("--probe_only", action="store_true"); ap.add_argument("--epochs", type=int, default=15); a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
CACHE = "/workspace/auto_research_cache/ar_motion_natural"; os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(REPO, "z_analysis/results/ar_output_motion_natural.json")
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 6)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
S, D, NB, C, K = 256, 1280, 14, 8, 6
TF_T = list(range(2, NB))                          # teacher forcing 대상 블록 (자리 t−1 출력)
NAMES = ["Xcur", "Xout", "Xtrue", "y", "w1"]


def extract():
    _argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
    import ar_crps_ft as FT
    sys.argv = _argv
    from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
    enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    _load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.half().to(dev).eval()
    ar = FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
    table = {}
    for l in open(os.path.join(REPO, "data_csv/k710_320/fps4_table.tsv")):
        r = l.rstrip("\n").split("\t"); table[r[0]] = r
    vids = []
    for ip in sorted(glob.glob("/workspace/dataset/training_data/flow_fps4/k710_320/shard*_index.tsv")):
        mm = np.memmap(ip.replace("_index.tsv", "_flow.f16"), np.float16, "r"); mm = mm[: (mm.size // 512) * 512].reshape(-1, 16, 16, 2)
        for l in open(ip):
            p, r0, n = l.rstrip("\n").split("\t"); r0, n = int(r0), int(n)
            if n >= 2 * NB - 1 and r0 + n <= len(mm) and p in table:
                vids.append((p, mm, r0, n))
    random.Random(0).shuffle(vids)
    rng = np.random.default_rng(0)
    idx_all = torch.arange(NB * S, device=dev).unsqueeze(0)
    TF = {k: [] for k in NAMES}; RO = {k: [] for k in NAMES}; done = 0
    for p, mm, r0, n in vids:
        if done >= a.n:
            break
        _, fps, nf, stp, n4 = table[p]
        try:
            vr = decord.VideoReader(p, num_threads=4)
            fidx = [j for j in range(0, int(nf), int(stp))][: int(n4)]; fidx = [j for j in fidx if j < len(vr)]
            if len(fidx) - 1 != n:                         # 흐름 추출 때와 장 수가 다르면 건너뜀 (자리 어긋남 방지)
                continue
            s0 = int(rng.integers(0, n - (2 * NB - 1) + 1))
            fr = vr.get_batch(fidx[s0: s0 + 2 * NB]).asnumpy()
        except Exception:
            continue
        H, W = fr.shape[1:3]; s_ = min(H, W); y0, x0 = (H - s_) // 2, (W - s_) // 2
        x = torch.from_numpy(np.ascontiguousarray(fr[:, y0:y0 + s_, x0:x0 + s_])).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round()        # 흐름 추출과 같은 화면
        x = ((x.to(dev) / 255.0).permute(1, 0, 2, 3) - MEAN) / STD                                              # (3, 28, 256, 256)
        f = np.asarray(mm[r0 + s0: r0 + s0 + 2 * NB - 1], dtype=np.float32).reshape(-1, S, 2)                   # (27, 256, 2)
        wb = f[0::2]                                                                                             # 블록 b 안 흐름 (14, 256, 2)
        xb = x.reshape(3, NB, 2, 256, 256).permute(1, 0, 2, 3, 4)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = torch.cat([F.layer_norm(enc2(xb[i:i + 7].half().contiguous()).float(), (D,)) for i in range(0, NB, 7)])   # (14, 256, 1280)
            tf = ar.forward_seq(h.reshape(1, -1, D), idx_all).float()[0].view(NB, S, D)                         # 자리 b 출력 = 블록 b+1 예측
            ro = ar.rollout_kv(h[:C].reshape(1, -1, D), idx_all[:, :C * S], K).float()[0].view(K, S, D)         # 블록 C … C+K−1
        h = h.half().cpu().numpy(); tf = tf.half().cpu().numpy(); ro = ro.half().cpu().numpy()
        for t in TF_T:
            c = rng.choice(S, a.cells, replace=False)
            for k_, v in zip(NAMES, (h[t - 1, c], tf[t - 1, c], h[t, c], wb[t, c], wb[t - 1, c])):
                TF[k_].append(v)
        for k in range(K):
            t = C + k; c = rng.choice(S, a.cells, replace=False)
            for k_, v in zip(NAMES, (h[C - 1, c], ro[k, c], h[t, c], wb[t, c], wb[C - 1, c])):
                RO[k_].append(v)
        done += 1
        if done % 100 == 0:
            print(f"  영상 {done}/{a.n}", flush=True)
    for tag, Dd, m in (("tf", TF, len(TF_T)), ("ro", RO, K)):
        for k_, v in Dd.items():
            arr = np.stack(v); np.save(f"{CACHE}/{tag}_{k_}.npy", arr.reshape(done, m, a.cells, -1))
    json.dump({"n_clips": done}, open(f"{CACHE}/meta.json", "w"))


def fit(X, Y, epochs):
    """선형 판독기 (1280 → 2), 입력 표준화 · 출력 표준화 · Huber. X (n, 1280) fp16, Y (n, 2).
    2026-10-26 판독기 검증: MSE 선형은 보류 R² 0.12 (꼬리 지배) — 방향은 읽힌다 (Huber 선형: 방향 cos 중앙 0.86, 부호 맞음 79 %, train 과 같음).
    카메라 보정 흐름은 선형으로 잘 안 읽혀 (cos 0.45) 원래 흐름 (카메라 포함, 학습 타깃과 같은 것) 을 쓴다."""
    mu = X.astype(np.float32).mean(0); sd = X.astype(np.float32).std(0) + 1e-4; ys = Y.std(0) + 1e-6
    lin = nn.Linear(X.shape[1], 2).to(dev); opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-2)
    Xt = torch.from_numpy(X); Yt = torch.from_numpy((Y / ys).astype(np.float32)); mu_t = torch.from_numpy(mu).to(dev); sd_t = torch.from_numpy(sd).to(dev)
    g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(perm), 8192):
            b = perm[i:i + 8192]; xb = (Xt[b].to(dev).float() - mu_t) / sd_t
            loss = F.huber_loss(lin(xb), Yt[b].to(dev), delta=0.5); opt.zero_grad(); loss.backward(); opt.step()   # 흐름 꼬리가 두껍다 (99 % 135 px) → Huber

    def pred(Xq):
        out = []
        with torch.no_grad():
            for i in range(0, len(Xq), 32768):
                out.append(lin((torch.from_numpy(Xq[i:i + 32768]).to(dev).float() - mu_t) / sd_t).cpu().numpy())
        return np.concatenate(out) * ys
    return pred


def score(yhat, y, w1, T=2.0):
    mv = np.linalg.norm(w1, axis=1) >= T
    def r2(m):
        return float(1 - ((y[m] - yhat[m]) ** 2).sum() / ((y[m] - y[m].mean(0)) ** 2).sum())
    a1, yh = w1[mv], yhat[mv]
    my = np.linalg.norm(y, axis=1) >= T                                                      # 방향 지표: 대상 블록이 실제로 움직인 칸
    cs = (yhat[my] * y[my]).sum(1) / (np.linalg.norm(yhat[my], axis=1) * np.linalg.norm(y[my], axis=1) + 1e-6)
    return {"dir_cos_med": float(np.median(cs)), "dir_acc": float(np.mean(cs > 0)), "n_target_moving": int(my.sum()),
            "R2_all": r2(np.ones(len(y), bool)), "R2_moving": r2(mv), "n_moving": int(mv.sum()),
            "r_carried": float(np.median((yh * a1).sum(1) / (a1 ** 2).sum(1))),               # 지금 움직임 방향으로 담은 몫 (중앙)
            "speed_ratio": float(np.median(np.linalg.norm(yh, axis=1) / np.linalg.norm(a1, axis=1)))}


if not a.probe_only:
    extract()
L = {f"{tag}_{k}": np.load(f"{CACHE}/{tag}_{k}.npy", mmap_mode="r") for tag in ("tf", "ro") for k in NAMES}
N = json.load(open(f"{CACHE}/meta.json"))["n_clips"]; tr = np.arange(N) % 2 == 0; te = ~tr
flat = lambda arr, m: np.ascontiguousarray(arr[m]).reshape(-1, arr.shape[-1])
R = {"n_clips": N, "cells_per_block": a.cells}
# teacher forcing (대상 블록 2–13 묶음)
y_tr, y_te, w_te = flat(L["tf_y"], tr), flat(L["tf_y"], te), flat(L["tf_w1"], te)
yd = flat(L["tf_y"], te); R["tf_data_r"] = float(np.median((yd * w_te).sum(1)[np.linalg.norm(w_te, axis=1) >= 2] / (w_te ** 2).sum(1)[np.linalg.norm(w_te, axis=1) >= 2]))
def copy_dir(y, w1, T=2.0):
    m = (np.linalg.norm(y, axis=1) >= T) & (np.linalg.norm(w1, axis=1) >= T)
    cs = (w1[m] * y[m]).sum(1) / (np.linalg.norm(w1[m], axis=1) * np.linalg.norm(y[m], axis=1) + 1e-6)
    return {"dir_cos_med": float(np.median(cs)), "dir_acc": float(np.mean(cs > 0)), "n": int(m.sum())}
R["tf_data_copy_direction"] = copy_dir(y_te, w_te)
P = {}
for name, key in (("self_block_encoder", "Xtrue"), ("current_block_encoder", "Xcur"), ("ar_output", "Xout")):
    P[name] = fit(flat(L[f"tf_{key}"], tr), y_tr, a.epochs)
    R.setdefault("tf", {})[name] = score(P[name](flat(L[f"tf_{key}"], te)), y_te, w_te); print("TF", name, R["tf"][name], flush=True)
R["tf"]["transfer_self_probe_on_ar_output"] = score(P["self_block_encoder"](flat(L["tf_Xout"], te)), y_te, w_te)
R["tf"]["self_probe_on_true_block (calibration)"] = R["tf"]["self_block_encoder"]
print("TF transfer", R["tf"]["transfer_self_probe_on_ar_output"], flush=True)
# 자리별 (문맥 자리 1–7 vs 그 뒤 8–12) — 같은 판독기
for lo, hi, lb in ((0, 7, "context_positions_targets_2_8"), (7, len(TF_T), "later_positions_targets_9_13")):
    yy, ww = flat(L["tf_y"][:, lo:hi], te), flat(L["tf_w1"][:, lo:hi], te)
    R["tf"][f"ar_output_{lb}"] = score(P["ar_output"](flat(L["tf_Xout"][:, lo:hi], te)), yy, ww)
    R["tf"][f"current_block_{lb}"] = score(P["current_block_encoder"](flat(L["tf_Xcur"][:, lo:hi], te)), yy, ww)
    print("TF", lb, "out", R["tf"][f"ar_output_{lb}"]["R2_moving"], "cur", R["tf"][f"current_block_{lb}"]["R2_moving"], flush=True)
# 굴림 걸음별 (블록 8+k−1, 문맥 = 블록 0–7, 지금 블록 = 블록 7)
for k in range(K):
    sl = lambda key, m: flat(L[f"ro_{key}"][:, k:k + 1], m)
    yk_tr, yk_te, wk_te = sl("y", tr), sl("y", te), sl("w1", te)
    mv = np.linalg.norm(wk_te, axis=1) >= 2
    e = {"data_r": float(np.median((yk_te * wk_te).sum(1)[mv] / (wk_te ** 2).sum(1)[mv])), "data_copy_direction": copy_dir(yk_te, wk_te)}
    for name, key in (("current_block_encoder", "Xcur"), ("ar_output", "Xout")):
        e[name] = score(fit(sl(key, tr), yk_tr, a.epochs)(sl(key, te)), yk_te, wk_te)
    e["transfer_self_probe_on_ar_output"] = score(P["self_block_encoder"](sl("Xout", te)), yk_te, wk_te)
    e["self_probe_on_true_block"] = score(P["self_block_encoder"](sl("Xtrue", te)), yk_te, wk_te)
    R.setdefault("rollout", {})[str(k + 1)] = e
    print(f"RO step {k + 1}: data r {e['data_r']:.2f} copy-dir {e['data_copy_direction']['dir_acc']:.2f} | dir_acc cur {e['current_block_encoder']['dir_acc']:.2f} out {e['ar_output']['dir_acc']:.2f} "
          f"| cos cur {e['current_block_encoder']['dir_cos_med']:.2f} out {e['ar_output']['dir_cos_med']:.2f} | R2mov cur {e['current_block_encoder']['R2_moving']:.3f} out {e['ar_output']['R2_moving']:.3f} "
          f"| carried: self-probe on true {e['self_probe_on_true_block']['r_carried']:.2f} on out {e['transfer_self_probe_on_ar_output']['r_carried']:.2f} "
          f"out-probe {e['ar_output']['r_carried']:.2f} cur-probe {e['current_block_encoder']['r_carried']:.2f}", flush=True)
json.dump(R, open(OUT, "w"), indent=1)
print("saved", OUT)
