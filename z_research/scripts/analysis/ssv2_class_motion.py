"""실제 영상으로 '움직임이 이어지는 상식' 을 배울 수 있나 — SSv2 행동 부류별 지속률 + 블록 AR 출력 (2026-10-26, GPU 1 장 · 6 GB).
사용자: "그럼 real world data 로는 배우기 힘든 거야?" → "ㅇ".
앞 측정: 학습 영상 전체의 지속률 (0.5 초 뒤) 0.06–0.20 (motion_predictability_flow.json) · 학습 영상에서 AR 출력은 데이터가 아는 만큼 담는다 (ar_output_motion_natural.json).
부류 (auto_research/scripts/cache_ssv2_classes.py 와 같은 템플릿 목록):
  수동 물리 (passive, 23 템플릿): 손을 떠난 뒤 중력 · 관성 · 충돌이 이동을 정한다 (떨어뜨리기 · 굴러가게 두기 · 던지기 · 밀어서 떨어뜨리기 …)
  행위자 (actor, 18 템플릿): 손이 쥐거나 계속 밀며 이동을 정한다 · 그 밖 (other): 나머지 템플릿 전부
① 지속률 (CPU, 미리 뽑은 RAFT 칸 흐름, motion_predictability_flow.py 와 같은 정의): 부류별 · 수동 물리 템플릿별.
   0.25 초 뒤 (블록 사이) · 0.5 초 뒤 (다음 블록 안, 물체가 갈 칸) 중앙 비율, 원래 흐름 · 카메라 뺀 흐름. 움직이는 칸 = |지금 블록 흐름| ≥ 2 px.
   ⚠️ 물체 칸 라벨이 없어 손 움직임도 섞인다 (던지기 · 떨어뜨리기 앞부분의 손).
② 블록 AR 출력 (natural_ar_vw_state latest): 부류마다 영상 n 편, 4 fps 14 장 = 블록 7 (문맥 4 → 굴림 3 걸음). 방향 판독기 (칸 토큰 → 칸 흐름, 선형 · Huber,
   ar_output_motion_natural.py 와 같다) 를 영상 절반 (세 부류 묶음) 으로 학습, 나머지 절반에서 부류별로 다음 블록 움직임 방향 맞힘:
   실제 흐름 복사 (데이터 기준) · 지금 블록 encoder 판독 · AR 출력 판독.
판정: 수동 물리 지속률이 높고 AR 출력이 복사만큼 → 합성 실패는 도메인 차이 / 지속률 높은데 AR 이 복사보다 한참 아래 → 드물어서 못 배움 /
     수동 물리도 지속률이 낮음 → 실제 영상에서는 0.5 초 단위로 관성이 잘 안 보임.
  CUDA_VISIBLE_DEVICES=0 MEM_GB=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/ssv2_class_motion.py --n 800
출력: z_analysis/results/ssv2_class_motion.json · 캐시 /workspace/auto_research_cache/ssv2_class_motion/
"""
import argparse, collections, glob, json, os, random, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=800); ap.add_argument("--cells", type=int, default=64)
ap.add_argument("--flow_per_template", type=int, default=1500); ap.add_argument("--flow_other", type=int, default=4000)
ap.add_argument("--probe_only", action="store_true"); ap.add_argument("--epochs", type=int, default=15); a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
CACHE = "/workspace/auto_research_cache/ssv2_class_motion"; os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(REPO, "z_analysis/results/ssv2_class_motion.json")
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
GROUPS = ("passive", "actor", "other")
G = 16; S, D, NB, C, K = 256, 1280, 7, 4, 3

lab = {}
for split in ("train", "validation"):
    for x in json.load(open(f"/workspace/dataset/training_data/SSv2/labels/{split}.json")):
        lab[x["id"]] = x["template"].replace("[", "").replace("]", "")
group_of = lambda t: "passive" if t in PASSIVE else ("actor" if t in ACTOR else "other")
vids = []
for ip in sorted(glob.glob("/workspace/dataset/training_data/flow_fps4/ssv2/shard*_index.tsv")):
    mm = np.memmap(ip.replace("_index.tsv", "_flow.f16"), np.float16, "r"); mm = mm[: (mm.size // 512) * 512].reshape(-1, G, G, 2)
    for l in open(ip):
        p, r0, n = l.rstrip("\n").split("\t"); r0, n = int(r0), int(n)
        vid = os.path.basename(p).split(".")[0]
        if vid in lab and n >= 5 and r0 + n <= len(mm):
            vids.append((p, mm, r0, n, lab[vid], group_of(lab[vid])))
random.Random(0).shuffle(vids)
R = {"n_videos_by_group_in_flow_cache": dict(collections.Counter(v[5] for v in vids))}
print("영상 (흐름 있음):", R["n_videos_by_group_in_flow_cache"], flush=True)


# ---------------- ① 지속률 (motion_predictability_flow.py 의 정의)
def persist(F5, T=2.0):
    """F5 (N, 5, 16, 16, 2) = [w0, u0, w1, u1, w2] → 0.25 초 뒤 · 0.5 초 뒤 중앙 비율 (물체가 갈 칸)."""
    W1, U1, W2 = F5[:, 2], F5[:, 3], F5[:, 4]
    n_, rr, cc = np.nonzero(np.linalg.norm(W1, axis=-1) >= T)
    if len(n_) < 50:
        return None
    a1 = W1[n_, rr, cc]
    out = {"n_cells": int(len(n_)), "n_samples": int(len(F5))}
    for name, sc, B in (("lag_0.25s", 1, U1), ("lag_0.5s", 2, W2)):
        dr = np.clip(rr + np.round(sc * a1[:, 1] / 16).astype(int), 0, G - 1); dc = np.clip(cc + np.round(sc * a1[:, 0] / 16).astype(int), 0, G - 1)
        b = B[n_, dr, dc]; out[name] = float(np.median((a1 * b).sum(1) / (a1 ** 2).sum(1)))
    return out


def gather(rows, per_video=3, seed=0):
    rng = np.random.default_rng(seed); F5 = []
    for p, mm, r0, n, t, g in rows:
        for j in rng.choice(np.arange(2, n - 2), size=min(per_video, n - 4), replace=False):
            F5.append(np.asarray(mm[r0 + j - 2: r0 + j + 3], dtype=np.float32))
    return np.stack(F5)


if not a.probe_only:
    by_t = collections.defaultdict(list)
    for v in vids:
        by_t[v[4]].append(v)
    P1 = {}
    for g in GROUPS:
        rows = [v for v in vids if v[5] == g]
        rows = [r for t in sorted({r[4] for r in rows}) for r in by_t[t][: a.flow_per_template]] if g != "other" else rows[: a.flow_other]
        F5 = gather(rows)
        P1[g] = {"raw": persist(F5), "cam": persist(F5 - np.median(F5, axis=(2, 3), keepdims=True)), "n_videos": len(rows)}
        print(f"① {g:8s} 영상 {len(rows):6d} | raw 0.25s {P1[g]['raw']['lag_0.25s']:.2f} 0.5s {P1[g]['raw']['lag_0.5s']:.2f} "
              f"| cam 0.25s {P1[g]['cam']['lag_0.25s']:.2f} 0.5s {P1[g]['cam']['lag_0.5s']:.2f}", flush=True)
    P1["passive_by_template"] = {}
    for t in PASSIVE:
        rows = by_t[t][: a.flow_per_template]
        if len(rows) < 30:
            continue
        F5 = gather(rows)
        m_raw, m_cam = persist(F5), persist(F5 - np.median(F5, axis=(2, 3), keepdims=True))
        P1["passive_by_template"][t] = {"n_videos": len(rows), "raw": m_raw, "cam": m_cam}
        print(f"   {t[:70]:70s} n {len(rows):5d} raw 0.5s {m_raw['lag_0.5s'] if m_raw else float('nan'):.2f} cam 0.5s {m_cam['lag_0.5s'] if m_cam else float('nan'):.2f}", flush=True)
    R["persistence"] = P1
    json.dump(R, open(OUT, "w"), indent=1)

# ---------------- ② 블록 AR 출력
NAMES = ["Xcur", "Xout", "Xtrue", "y", "w1"]


def extract():
    _argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
    import ar_crps_ft as FT
    sys.argv = _argv
    from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
    dev = torch.device("cuda")
    enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    _load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.half().to(dev).eval()
    ar = FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
    table = {}
    for l in open(os.path.join(REPO, "data_csv/ssv2/fps4_table.tsv")):
        r = l.rstrip("\n").split("\t"); table[r[0]] = r
    rng = np.random.default_rng(0); idx_all = torch.arange(NB * S, device=dev).unsqueeze(0)
    store = {g: {"tf": {k: [] for k in NAMES}, "ro": {k: [] for k in NAMES}} for g in GROUPS}
    cnt = collections.Counter()
    for p, mm, r0, n, t, g in vids:
        if cnt[g] >= a.n or n < 2 * NB - 1 or p not in table:
            continue
        _, fps, nf, stp, n4 = table[p]
        try:
            vr = decord.VideoReader(p, num_threads=1)                       # webm (VP9) 은 여러 스레드로 열면 디코딩이 실패한다
            fidx = [j for j in range(0, int(nf), int(stp))][: int(n4)]; fidx = [j for j in fidx if j < len(vr)]
            if len(fidx) - 1 != n:
                continue
            s0 = (n - (2 * NB - 1)) // 2                                    # 영상 가운데 창 (SSv2 행동은 가운데에 있다)
            fr = vr.get_batch(fidx[s0: s0 + 2 * NB]).asnumpy()
        except Exception:
            continue
        H, W = fr.shape[1:3]; s_ = min(H, W); y0, x0 = (H - s_) // 2, (W - s_) // 2
        x = torch.from_numpy(np.ascontiguousarray(fr[:, y0:y0 + s_, x0:x0 + s_])).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round()
        x = ((x.to(dev) / 255.0).permute(1, 0, 2, 3) - MEAN) / STD
        f = np.asarray(mm[r0 + s0: r0 + s0 + 2 * NB - 1], dtype=np.float32).reshape(-1, S, 2); wb = f[0::2]
        xb = x.reshape(3, NB, 2, 256, 256).permute(1, 0, 2, 3, 4)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = F.layer_norm(enc2(xb.half().contiguous()).float(), (D,))
            tf = ar.forward_seq(h.reshape(1, -1, D), idx_all).float()[0].view(NB, S, D)
            ro = ar.rollout_kv(h[:C].reshape(1, -1, D), idx_all[:, :C * S], K).float()[0].view(K, S, D)
        h = h.half().cpu().numpy(); tf = tf.half().cpu().numpy(); ro = ro.half().cpu().numpy()
        st = store[g]
        for tt in range(1, NB):
            c = rng.choice(S, a.cells, replace=False)
            for k_, v in zip(NAMES, (h[tt - 1, c], tf[tt - 1, c], h[tt, c], wb[tt, c], wb[tt - 1, c])):
                st["tf"][k_].append(v)
        for k in range(K):
            tt = C + k; c = rng.choice(S, a.cells, replace=False)
            for k_, v in zip(NAMES, (h[C - 1, c], ro[k, c], h[tt, c], wb[tt, c], wb[C - 1, c])):
                st["ro"][k_].append(v)
        cnt[g] += 1
        if sum(cnt.values()) % 300 == 0:
            print(f"  ② 영상 {dict(cnt)}", flush=True)
        if all(cnt[x] >= a.n for x in GROUPS):
            break
    for g in GROUPS:
        for tag, m in (("tf", NB - 1), ("ro", K)):
            for k_, v in store[g][tag].items():
                np.save(f"{CACHE}/{g}_{tag}_{k_}.npy", np.stack(v).reshape(cnt[g], m, a.cells, -1))
    json.dump(dict(cnt), open(f"{CACHE}/meta.json", "w"))


def fit(X, Y, epochs, dev):
    mu = X.astype(np.float32).mean(0); sd = X.astype(np.float32).std(0) + 1e-4; ys = Y.std(0) + 1e-6
    lin = nn.Linear(X.shape[1], 2).to(dev); opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-2)
    Xt = torch.from_numpy(X); Yt = torch.from_numpy((Y / ys).astype(np.float32)); mu_t = torch.from_numpy(mu).to(dev); sd_t = torch.from_numpy(sd).to(dev)
    g_ = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), generator=g_)
        for i in range(0, len(perm), 8192):
            b = perm[i:i + 8192]; loss = F.huber_loss(lin((Xt[b].to(dev).float() - mu_t) / sd_t), Yt[b].to(dev), delta=0.5)
            opt.zero_grad(); loss.backward(); opt.step()

    def pred(Xq):
        with torch.no_grad():
            return np.concatenate([lin((torch.from_numpy(Xq[i:i + 32768]).to(dev).float() - mu_t) / sd_t).cpu().numpy() for i in range(0, len(Xq), 32768)]) * ys
    return pred


def dir_acc(yhat, y, T=2.0):
    m = np.linalg.norm(y, axis=1) >= T
    cs = (yhat[m] * y[m]).sum(1) / (np.linalg.norm(yhat[m], axis=1) * np.linalg.norm(y[m], axis=1) + 1e-6)
    return {"acc": float(np.mean(cs > 0)), "cos_med": float(np.median(cs)), "n": int(m.sum())}


if not a.probe_only:
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 6)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
    extract()
dev = torch.device("cuda")
R = json.load(open(OUT)) if os.path.exists(OUT) else R
cnt = json.load(open(f"{CACHE}/meta.json"))
L = {(g, tag, k): np.load(f"{CACHE}/{g}_{tag}_{k}.npy", mmap_mode="r") for g in GROUPS for tag in ("tf", "ro") for k in NAMES}
half = {g: (np.arange(cnt[g]) % 2 == 0) for g in GROUPS}
flat = lambda arr, m, sl=slice(None): np.ascontiguousarray(arr[m][:, sl]).reshape(-1, arr.shape[-1])
R["ar_output"] = {"n_videos": cnt}
for tag, steps in (("tf", [None]), ("ro", list(range(K)))):
    for st in steps:
        sl = slice(None) if st is None else slice(st, st + 1)
        key = "teacher_forcing" if st is None else f"rollout_step_{st + 1}"
        cat = lambda nm, tr: np.concatenate([flat(L[(g, tag, nm)], half[g] if tr else ~half[g], sl) for g in GROUPS])
        P = {nm: fit(cat(src, True), cat("y", True), a.epochs, dev) for nm, src in (("current_block", "Xcur"), ("ar_output", "Xout"))}
        res = {}
        for g in GROUPS:
            te = ~half[g]; y, w1 = flat(L[(g, tag, "y")], te, sl), flat(L[(g, tag, "w1")], te, sl)
            res[g] = {"copy_flow": dir_acc(w1, y),                                         # 지금 블록 흐름을 그대로 (판독기와 같은 칸 = 대상 블록이 움직인 칸 전부)
                      "current_block": dir_acc(P["current_block"](flat(L[(g, tag, "Xcur")], te, sl)), y),
                      "ar_output": dir_acc(P["ar_output"](flat(L[(g, tag, "Xout")], te, sl)), y)}
            print(f"② {key:16s} {g:8s} | 복사 {res[g]['copy_flow']['acc']:.3f} · 지금 블록 {res[g]['current_block']['acc']:.3f} · AR 출력 {res[g]['ar_output']['acc']:.3f}", flush=True)
        R["ar_output"][key] = res
json.dump(R, open(OUT, "w"), indent=1)
print("saved", OUT)
