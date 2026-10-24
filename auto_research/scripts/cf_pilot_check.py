#!/usr/bin/env python3
"""[물리] 반사실 쌍 시험 렌더 확인 (학습 없음, CPU): 짝 사이 차이가 정말 "장애물 하나" 인지 픽셀 차 지도로.

  /venv/main/bin/python auto_research/scripts/cf_pilot_check.py --root /workspace/dataset/training_data/phys3d_cf_v1/pilot --fig <scratch>/cf_pilot

장면마다 네 변형의 렌더 장 (--keep-frames) 을 읽어:
  (1) 충돌 전 (장 < t_hit − 2): W_pos vs W_imp · N_pos vs N_imp 픽셀 차 (같아야 한다 — 물리 · 보임이 같고 장애물 충돌만 다름)
  (2) 충돌 전: W_pos vs N_pos 픽셀 차가 장애물 투영 상자 (+ 그림자) 안에만 있나 — 차이 픽셀 중 장애물 상자 (여백 12 px) 안 몫
  (3) 충돌 뒤: W_pos vs W_imp (튕김 vs 통과) 차이가 물체 둘레에만 있나 — 차이 픽셀 중 두 물체 bbox (여백 8 px) 안 몫
  (4) 라벨: 네 변형 궤적이 물리 두 판과 맞나 (json label_traj_check_px), 가림 비율
그림: 장면마다 변형 4 줄 × 장 6 칸 + 차이 지도 2 줄 (PNG).
"""
import argparse, glob, json, os
import numpy as np
from PIL import Image

V = ["W_pos", "W_imp", "N_pos", "N_imp"]


def frames(d):
    fs = sorted(glob.glob(os.path.join(d, "f*.png")))
    return np.stack([np.asarray(Image.open(f).convert("RGB"), dtype=np.int16) for f in fs])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--fig", required=True)
    a = ap.parse_args()
    os.makedirs(a.fig, exist_ok=True)
    rows = []
    for sd in sorted(glob.glob(os.path.join(a.root, "scene_*"))):
        if not all(os.path.isdir(os.path.join(sd, f"{v}_frames")) for v in V):
            continue
        M = {v: json.load(open(os.path.join(sd, f"{v}.json"))) for v in V}
        F = {v: frames(os.path.join(sd, f"{v}_frames")) for v in V}
        L = {v: np.load(os.path.join(sd, f"{v}_states.npz")) for v in V}
        th = M["W_pos"].get("t_event")
        if th is None:                                                                # 옛 판: 라벨 궤적이 처음 0.5 px 넘게 갈리는 장
            th = int(np.where(np.linalg.norm(L["W_pos"]["pos"][:, 0] - L["N_pos"]["pos"][:, 0], axis=-1) > 0.5)[0][0])
        T = min(len(F[v]) for v in V)
        dif = lambda x, y, sl: (np.abs(F[x][sl] - F[y][sl]).max(-1) > 10)        # (t, H, W) 차이 픽셀
        pre = slice(0, max(1, th)); post = slice(th + 2, T)
        r = dict(scene=os.path.basename(sd), t_hit=th, kind=M["W_pos"]["obstacle"]["kind"], cam=M["W_pos"]["env"]["camera"]["mode"],
                 screen=M["W_pos"]["screen"] is not None, traj_check_px=max(M[v]["label_traj_check_px"] for v in V),
                 hidden=[round(M[v]["hidden_frac"], 3) for v in V])
        r["pre_Wpos_vs_Wimp_px"] = float(dif("W_pos", "W_imp", pre).mean()); r["pre_Npos_vs_Nimp_px"] = float(dif("N_pos", "N_imp", pre).mean())
        r["pre_maxdiff_W"] = int(np.abs(F["W_pos"][pre] - F["W_imp"][pre]).max()); r["pre_maxdiff_N"] = int(np.abs(F["N_pos"][pre] - F["N_imp"][pre]).max())   # 갈림 전 최대 차 (복사 뒤 0 이어야)
        r["t_event"] = int(th); r["t_hit"] = M["W_pos"]["t_hit"]
        r["label_pre_maxdiff_px"] = float(max(np.abs(L["W_pos"]["pos"][pre, 0] - L[v]["pos"][pre, 0]).max() for v in V))
        # 장애물 투영 상자: W_pos 와 N_pos 의 충돌 전 차이 지도 전체를 상자로 근사하지 않고, 라벨 없는 대신 '차이가 물체 bbox 밖' 몫을 잰다
        d_wn = dif("W_pos", "N_pos", pre)
        bb = L["W_pos"]["bbox"][pre, 0]                                              # (t, 4) px
        obj_mask = np.zeros_like(d_wn)
        for t in range(d_wn.shape[0]):
            x0, y0, x1, y1 = [int(round(v)) for v in bb[t]]
            obj_mask[t, max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8] = True
        r["pre_Wpos_vs_Npos_frac_px"] = float(d_wn.mean())
        r["pre_Wpos_vs_Npos_on_object"] = float((d_wn & obj_mask).sum() / max(d_wn.sum(), 1))   # 충돌 전 차이 중 물체 둘레 몫 (작아야: 차이는 장애물 · 그림자)
        d_post = dif("W_pos", "W_imp", post)
        bw, bi = L["W_pos"]["bbox"][post, 0], L["W_imp"]["bbox"][post, 0]
        m2 = np.zeros_like(d_post)
        for t in range(d_post.shape[0]):
            for b in (bw[t], bi[t]):
                x0, y0, x1, y1 = [int(round(v)) for v in b]
                m2[t, max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8] = True
        r["post_Wpos_vs_Wimp_on_objects"] = float((d_post & m2).sum() / max(d_post.sum(), 1))
        rows.append(r)
        # 그림
        ts = sorted(set([0, max(0, th - 6), th, min(T - 1, th + 6), min(T - 1, th + 18), T - 1]))
        tiles = [np.concatenate([F[v][t] for t in ts], 1) for v in V]
        heat = lambda m: np.repeat((m * 255).astype(np.int16)[..., None], 3, -1)
        full_wn = np.abs(F["W_pos"][:T] - F["N_pos"][:T]).max(-1) > 10
        full_wi = np.abs(F["W_pos"][:T] - F["W_imp"][:T]).max(-1) > 10
        tiles.append(np.concatenate([heat(full_wn[t]) for t in ts], 1))
        tiles.append(np.concatenate([heat(full_wi[t]) for t in ts], 1))
        Image.fromarray(np.clip(np.concatenate(tiles, 0), 0, 255).astype(np.uint8)).save(os.path.join(a.fig, f"{r['scene']}.png"))
        print(json.dumps(r), flush=True)
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), "../results/cf_pilot_check.json"), "w"), indent=1)
    if rows:
        k = ["pre_maxdiff_W", "pre_maxdiff_N", "label_pre_maxdiff_px", "pre_Wpos_vs_Wimp_px", "pre_Npos_vs_Nimp_px", "pre_Wpos_vs_Npos_frac_px", "pre_Wpos_vs_Npos_on_object", "post_Wpos_vs_Wimp_on_objects", "traj_check_px"]
        print("평균", {x: round(float(np.mean([r[x] for r in rows])), 4) for x in k})


if __name__ == "__main__":
    main()
