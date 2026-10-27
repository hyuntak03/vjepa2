"""encoder 1280 의 모양 · 색 선형 판독 — 물체 칸 평균 vs 화면 전체 평균 vs 배경 (물체 칸 뺀) 평균, 같은 표본 (2026-10-26, CPU).
사용자: "encoder embedding (1280) 에서 shape × color acc 가 저렇게 낮다고? 100 % 가까이 나오는 게 아니라? 다시 검토해봐".
라벨 점검 (같은 날): 8,350 편 전부 모양 · 색 번호가 클립 json 의 주 물체와 일치 (불일치 0), 주 물체는 클립마다 하나.
데이터: ar_layers 전체 토큰 캐시의 encoder 출력 (16, 256, 1280; 블록 단위 LN(target_encoder)). train 앞 N 편 (기본 2,500) · val 350 편, 블록 0–15 중 물체 보이는 블록.
  물체 칸 = val/train_meta 의 블록별 cells (물체 상자 안에 중심이 든 16×16 칸; 비면 물체 중심 칸 하나). 배경 = 나머지 칸.
판독기: 선형층 하나 (모양 6 · 색 8 · 모양×색 48 따로), 입력 표준화, AdamW lr 1e-3 · wd 1e-2 · 묶음 256 · 최대 100 epoch, train 클립 20 % 로 epoch 선택.
  /venv/vjepa/bin/python z_research/scripts/analysis/identity_readout_object_cells.py   → z_analysis/results/identity_readout_object_cells.json
"""
import argparse, copy, json, os
from multiprocessing import Pool
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--n_train", type=int, default=2500); a = ap.parse_args()
SRC = "/workspace/auto_research_cache/ar_layers"


def one(arg):
    split, m = arg
    path = f"{SRC}/{split}/{m['i']:05d}.npz"
    E = np.load(path)["enc"].astype(np.float32)                       # (16, 256, 1280)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    rows = []
    for b, bl in enumerate(m["blocks"]):
        if not bl["ok"]:
            continue
        cells = bl["cells"] or [int(min(15, max(0, bl["ctr"][1] // 16)) * 16 + min(15, max(0, bl["ctr"][0] // 16)))]
        mask = np.zeros(256, bool); mask[cells] = True
        rows.append((E[b, mask].mean(0), E[b].mean(0), E[b, ~mask].mean(0), m["shape"], m["color"], m["i"], len(cells)))
    return rows


def load(split, n):
    meta = json.load(open(f"{SRC}/{split}_meta.json"))[: n or None]
    with Pool(6) as pool:
        R = [r for rs in pool.imap(one, [(split, m) for m in meta], chunksize=4) for r in rs]
    return {"obj": np.stack([r[0] for r in R]), "full": np.stack([r[1] for r in R]), "bg": np.stack([r[2] for r in R]),
            "shape": np.array([r[3] for r in R]), "color": np.array([r[4] for r in R]), "clip": np.array([r[5] for r in R]), "ncell": np.array([r[6] for r in R])}


if __name__ == "__main__":
    import torch, torch.nn.functional as F
    torch.set_num_threads(32)
    T, V = load("train", a.n_train), load("val", 0)
    print(f"표본: train {len(T['shape'])} (클립 {a.n_train}) · val {len(V['shape'])} · 물체 칸 수 중앙값 {np.median(V['ncell'])}", flush=True)

    def fit(X, y, clip, K):
        torch.manual_seed(0); X, y = torch.from_numpy(X), torch.from_numpy(y); h = torch.from_numpy(clip % 5 == 0)
        head = torch.nn.Linear(X.shape[1], K); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2); best = (-1, None)
        Xa, ya, Xh, yh = X[~h], y[~h], X[h], y[h]
        for ep in range(100):
            perm = torch.randperm(len(Xa))
            for i in range(0, len(Xa), 256):
                idx = perm[i:i + 256]; loss = F.cross_entropy(head(Xa[idx]), ya[idx]); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                acc = (head(Xh).argmax(1) == yh).float().mean().item()
            if acc > best[0]:
                best = (acc, copy.deepcopy(head.state_dict()))
        head.load_state_dict(best[1]); return head

    res = {"n_train": int(len(T["shape"])), "n_val": int(len(V["shape"])), "label_check": "8,350 편 모양 · 색 번호 = 클립 json 주 물체 (불일치 0)"}
    for feat in ("obj", "full", "bg"):
        Xt, Xv = T[feat], V[feat]; mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Xt, Xv = ((Xt - mu) / sd).astype(np.float32), ((Xv - mu) / sd).astype(np.float32)
        r = {}
        for task, yt, yv, K in (("shape", T["shape"], V["shape"], 6), ("color", T["color"], V["color"], 8), ("joint48", T["shape"] * 8 + T["color"], V["shape"] * 8 + V["color"], 48)):
            h = fit(Xt, yt.astype(np.int64), T["clip"], K)
            with torch.no_grad():
                r[task] = round(float((h(torch.from_numpy(Xv)).argmax(1).numpy() == yv).mean() * 100), 1)
        res[feat] = r; print(feat, r, flush=True)
    res["val_majority"] = {"shape": round(float(np.bincount(V["shape"]).max() / len(V["shape"]) * 100), 1), "color": round(float(np.bincount(V["color"]).max() / len(V["color"]) * 100), 1)}
    json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/identity_readout_object_cells.json", "w"), indent=1, ensure_ascii=False)
    print(res["val_majority"])
