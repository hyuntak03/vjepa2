"""[비판] 3D 상태 head 의 attention 이 관측 토큰에서 물체 칸에 몰리나 (학습 없음, CPU) — Main 62.6 의 "주소" 해석용.
3D 합성 검증 캐시 (관측 블록 토큰 · 라벨 bbox). 블록 b 의 물체 칸 = 블록 마지막 장 bbox 와 겹치는 칸 (보이는 블록만).
보고: attention 질량 중 물체 칸 몫 vs 물체 칸 넓이 (균등이면 같음) — 3D head · 무작위 쿼리 head."""
import os, sys, json, numpy as np, torch
REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO)
from app.vjepa_frozen.state_head import SingleStateHead  # noqa: E402
torch.set_num_threads(8); torch.manual_seed(0)
ck = torch.load(os.path.join(REPO, "z_training/runs/natural_ar_vw_state_phys3d_single/latest.pt"), map_location="cpu", weights_only=False)
H3 = SingleStateHead(); H3.load_state_dict(ck["state_head"]); H3.eval()
HR = SingleStateHead().eval()
C = "/workspace/auto_research_cache/phys3d/"
out = {"head3d": [], "headrand": [], "area": [], "head3d_max_in": []}
for s in range(4):
    H = np.load(C + f"h_val_target_s{s}.npy", mmap_mode="r"); L = np.load(C + f"lab_val_s{s}.npz")
    bbox, vis, pres = L["bbox"], L["vis_frac"], L["present"]
    for i in range(H.shape[0]):
        h = torch.from_numpy(np.asarray(H[i], dtype=np.float32))
        with torch.no_grad():
            a3 = H3(h.reshape(1, -1, 1280), 256)["attn"][0, :, 0].numpy(); ar = HR(h.reshape(1, -1, 1280), 256)["attn"][0, :, 0].numpy()
        for b in range(h.shape[0]):
            f = 2 * b + 1
            if not (pres[i, f] and vis[i, f] > 0.5):
                continue
            x0, y0, x1, y1 = bbox[i, f]                                           # 256 화면 px (가정: x0,y0,x1,y1)
            m = np.zeros((16, 16), bool)
            m[int(max(0, np.floor(y0 / 16))):int(min(16, np.ceil(y1 / 16))), int(max(0, np.floor(x0 / 16))):int(min(16, np.ceil(x1 / 16)))] = True
            m = m.reshape(-1)
            if m.sum() == 0 or m.sum() > 128:
                continue
            out["head3d"].append(a3[b][m].sum()); out["headrand"].append(ar[b][m].sum()); out["area"].append(m.mean())
            out["head3d_max_in"].append(bool(m[a3[b].argmax()]))
res = {k: float(np.mean(v)) for k, v in out.items()}; res["n"] = len(out["area"])
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(REPO, "auto_research/results/crit_head_attn_object.json"), "w"), indent=1)
