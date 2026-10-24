"""[비판] 상태 head (phys3d_single latest) 가 관측 토큰 (목표 encoder) 에서도 상태를 읽나 — 3D 합성 val (학습 없음, CPU).
head 는 predictor 예측 미래로 학습됐다. 관측 블록 토큰 h[b] 에 걸면 분포 밖일 수 있다 (리더 질문).
위치 오차 (px, 256 해상도): 관측 head(h[b]) vs 라벨[b] · TF 예측 head(p[b]) vs 라벨[b+1] · 복사 (라벨[b] 를 b+1 에) — 물체가 있고 보이는 블록.
"없음" 확률: 보임 / 가림 / 화면 밖 블록별 평균.
사용: python crit_state_head_check.py [--shards 0] [--pred 1]"""
import os, sys, json, argparse, numpy as np, torch
REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT  # noqa: E402
sys.argv = _argv
from app.vjepa_frozen.state_head import SingleStateHead  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--shards", default="0,1,2,3"); ap.add_argument("--pred", type=int, default=1)
ap.add_argument("--run", default="natural_ar_vw_state_phys3d_single"); ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_num_threads(8)
ck = torch.load(os.path.join(REPO, "z_training/runs", a.run, "latest.pt"), map_location="cpu", weights_only=False)
hd = SingleStateHead(); hd.load_state_dict(ck["state_head"]); hd.eval()
W = FT.NoiseWrap(FT.load_pred(a.run, "latest.pt"), 32, 0.0).eval() if a.pred else None
C = "/workspace/auto_research_cache/phys3d/"
E = {"obs": [], "pred": [], "copy": []}; NONE = {"vis": [], "hid": [], "out": []}; NONEp = {"vis": [], "hid": [], "out": []}
for s in a.shards.split(","):
    H = np.load(C + f"h_val_target_s{s}.npy", mmap_mode="r"); L = np.load(C + f"lab_val_s{s}.npz")
    pos = L["pos"]; pres = L["present"]; hid = L["hidden"]; vis = L["vis_frac"]; inf = L["inframe_frac"]
    for i in range(H.shape[0]):
        h = torch.from_numpy(np.asarray(H[i], dtype=np.float32))            # (16, 256, D)
        NB = h.shape[0]
        lb = lambda b: 2 * b + 1                                             # 블록 b 라벨 = 마지막 장
        with torch.no_grad():
            oh = hd(h.reshape(1, -1, 1280), 256)
            ph = oh["pos"][0, :, 0].numpy() * 256; Ph = torch.softmax(oh["obj"][0, :, 0], -1)[:, -1].numpy()
            if W is not None:
                p = W.tf(h.reshape(1, NB * 256, 1280)).float().view(NB, 256, 1280)[:-1]
                op = hd(p.reshape(1, -1, 1280), 256); pp = op["pos"][0, :, 0].numpy() * 256; Pp = torch.softmax(op["obj"][0, :, 0], -1)[:, -1].numpy()
        for b in range(NB):
            f = lb(b); ok = pres[i, f] and vis[i, f] > 0.5 and inf[i, f] > 0.5
            st = "vis" if ok else ("hid" if pres[i, f] and hid[i, f] else "out")
            NONE[st].append(Ph[b])
            if ok:
                E["obs"].append(np.abs(ph[b] - pos[i, f]).sum())
            if b >= 1:
                fp = lb(b - 1)
                if ok and pres[i, fp]:
                    E["copy"].append(np.abs(pos[i, fp] - pos[i, f]).sum())
                    if W is not None:
                        E["pred"].append(np.abs(pp[b - 1] - pos[i, f]).sum())
                if W is not None:
                    NONEp[st].append(Pp[b - 1])
    print("shard", s, "done", flush=True)
res = {k: dict(n=len(v), median=float(np.median(v)) if v else None, mean=float(np.mean(v)) if v else None) for k, v in E.items()}
res["none_prob_obs"] = {k: float(np.mean(v)) if v else None for k, v in NONE.items()}
res["none_prob_pred"] = {k: float(np.mean(v)) if v else None for k, v in NONEp.items()}
print(json.dumps(res, indent=1))
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
