#!/usr/bin/env python3
"""VOBench 학습 split 의 물체 칸 · 이동량 라벨 — 보조 모듈 감독용 (2026-10-31, 사용자: "지금 모듈 구조 그대로 하고 VOBench 로 학습해보자").

build_perm_occupancy.py 와 같은 정의를 VOBench label.json (팔마다 장별 u, v, ext_u/ext_v (r_px), hidden; 512 px) 으로:
  칸 (16 블록 × 256 칸, int8): 0 배경 · 1 보이는 물체 · 2 가려진 물체 (문맥에서 본 적 있음) · −1 무시 (문맥에서 못 본 물체가 보임)
  이동량 (16 블록 × 256 칸 × 2, int8, 1/4 칸, −128 = 없음): 문맥에서 마지막으로 보인 장의 물체 칸 → 블록 t 둘째 장까지 물체 중심 이동
  보임 = hidden < 0.5. VOBench 물체는 늘 화면 안이다. 팔에 물체가 없으면 (obj 'empty') 전부 배경 / 이동량 없음.
학습 로더 (frames_index) 는 시작 장을 frames_start_choices [0, 1, 2] 에서 고르므로 라벨을 시작마다 만든다:
  행 = index 행 i × 3 + 시작 번호 (FramesIndexDataset 의 row_label: true 가 이 번호를 라벨로 넘긴다). 원본 장 = 시작 + 3 j (j = 0..31).
팔: pos_a → A, pos_b → B, gravity 'pos' → P (가능 궤적).
출력: data_csv/vobench_v1_split_v2g/{occ,disp}_train.npy — (N × 3, 1, 16, 256[, 2]), 둘째 축 = 문맥 [16] (학습 mask 와 같다)

  /venv/vjepa/bin/python z_training/data/build_vobench_occupancy.py
"""
import csv, json, os
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
R = "/workspace/dataset/benchmarks/VOBench_v1"
IDX = os.path.join(REPO, "data_csv/vobench_v1_split_v2g/index_train.csv")
STARTS, STRIDE, NF, TUB, G = (0, 1, 2), 3, 32, 2, 16
CTX = (16,)


def cells(x0, y0, x1, y1):
    return [r * G + c for r in range(max(0, int(y0 // 16)), min(G, int(y1 // 16) + 1)) for c in range(max(0, int(x0 // 16)), min(G, int(x1 // 16) + 1))]


def one(r, s):
    occ = np.zeros((len(CTX), NF // TUB, G * G), np.int8)
    disp = np.full((len(CTX), NF // TUB, G * G, 2), -128, np.int8)
    arm = {"pos_a": "A", "pos_b": "B", "pos": "P"}[r["variant"]]
    if r["variant"] in ("pos_a", "pos_b") and (r["obj_a"] if arm == "A" else r["obj_b"]) == "empty":
        return occ, disp
    L = json.load(open(os.path.join(R, os.path.dirname(r["file_name"]), "label.json")))["labels"][arm]
    fr = [s + STRIDE * j for j in range(NF)]
    vis = np.array([L[f]["hidden"] < 0.5 for f in fr])
    box = [tuple(np.array([L[f]["u"] - L[f].get("ext_u", L[f]["r_px"]), L[f]["v"] - L[f].get("ext_v", L[f]["r_px"]),
                           L[f]["u"] + L[f].get("ext_u", L[f]["r_px"]), L[f]["v"] + L[f].get("ext_v", L[f]["r_px"])]) / 2) for f in fr]
    ctr = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in box])
    for ci, C in enumerate(CTX):
        seen = vis[:C].any()
        for t in range(NF // TUB):
            vf = [f for f in (2 * t, 2 * t + 1) if vis[f]]
            if vf:
                for f in vf: occ[ci, t, cells(*box[f])] = 1 if seen else -1
            elif seen:
                for f in (2 * t, 2 * t + 1): occ[ci, t, cells(*box[f])] = 2
        fs = np.where(vis[:C])[0]
        if len(fs):
            f0 = fs[-1]; src = cells(*box[f0])
            for t in range(NF // TUB):
                disp[ci, t, src] = np.clip(np.round((ctr[2 * t + 1] - ctr[f0]) / 4.0), -127, 127).astype(np.int8)
    return occ, disp


def main():
    rows = list(csv.DictReader(open(IDX)))
    O, D = [], []
    for r in rows:
        for s in STARTS:
            o, d = one(r, s); O.append(o); D.append(d)
    O, D = np.stack(O), np.stack(D)
    np.save(os.path.join(os.path.dirname(IDX), "occ_train.npy"), O); np.save(os.path.join(os.path.dirname(IDX), "disp_train.npy"), D)
    fut = O[:, 0, CTX[0] // TUB:].ravel(); dd = D[:, 0, CTX[0] // TUB:].reshape(-1, 2); ok = dd[:, 0] != -128
    print(f"{len(rows)} 행 × {len(STARTS)} 시작 → occ {O.shape} · 미래 칸 배경 {100 * (fut == 0).mean():.2f} % · 보임 {100 * (fut == 1).mean():.2f} % · "
          f"가려짐 {100 * (fut == 2).mean():.2f} % · 무시 {100 * (fut == -1).mean():.2f} % · 이동량 라벨 칸 {100 * ok.mean():.2f} % · "
          f"이동 (칸) 중앙 {np.median(np.hypot(*(dd[ok].astype(float) / 4).T)):.2f}")


if __name__ == "__main__":
    main()
