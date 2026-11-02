#!/usr/bin/env python3
"""BlenderTrack v3 · v3t — 미래 자리를 외워서 맞힐 수 있나 (2026-11-01, 사용자 질문).

물음: 이 학습셋으로 학습하면 predictor 가 "속도로 계산" 대신 "외운 자리" 로 late 미래를 맞힐 수 있나.
      VOBench 는 그럴 수 있었다 (미래 경로 21 가지, test 경로의 67 % · 끝 칸 100 % 가 train 에 있음, 전형 자리 기준선 99–100 %).
방법: target 이 움직이는 물체 있는 판 (pos_a) 의 화면 궤적 (라벨 center_px, 토큰 좌표 = px / 16 − 0.5) 을 train · test 전부 뽑고,
  test 의 미래 자리 (샘플 16–31) 를 다섯 가지로 맞힌다 — 기준선의 '학습' 은 train 통계뿐.
  B0 머무름        : 마지막으로 보인 샘플의 자리
  B1 외운 변위     : 마지막 보인 자리 + train 평균 변위 (같은 방향 · 같은 운동 종류 · 같은 마지막 보인 샘플) — 속도를 안 봄
  B1s 외운 화면 자리: train 평균 화면 자리 (같은 방향 · 운동 종류) — VOBench 식 시간표
  B2 등속          : 마지막 보인 자리 + 문맥 끝 속도 × 경과 (참 위치로 맞춘 속도)
  B3 등가속        : 보이는 문맥 샘플에 2 차식 (참 위치)
  B4 가까운 이웃   : 문맥 궤적 (보이는 샘플의 상대 위치) 이 가장 가까운 train 영상 5 개의 미래 변위 평균 — "외워서 보간"
  판정: B1 · B1s 가 B2 · B3 만큼 맞으면 속도 없이 외워서 풀린다. 크게 나쁘면 외우기로는 안 풀린다 (속도를 읽어야 한다).
  다양성: 문맥 끝 화면 속력 · 가림 길이 · 사라진 자리 · 끝 자리의 퍼짐, 양자화한 미래 경로 (칸) 의 test → train 겹침.
운동 종류: 평지 미끄러짐 (상자 · 통) · 평지 구름 (공) · 빗면 오르기 · 빗면 내리기 (설정 cells.place · ramp_mode · target_kind).

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/blender_v3_memorization.py      # → predictor_lab/results/v3_memorization.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/nfs_cache/blender_track")
OUT = REPO / "predictor_lab/results/v3_memorization.json"
IDX = np.arange(0, 64, 2)
CTX = 16
BALLS = {"sphere", "basketball", "soccer_ball", "beach_ball"}


def motion(r):
    if r["place"] == "ramp":
        return "ramp_" + r["ramp_mode"]
    return "flat_roll" if r["target_kind"] in BALLS else "flat_slide"


def load(name, split):
    rows = [r for r in csv.DictReader(open(REPO / f"data_csv/blender_track_{name}/{split}_arms.csv"))
            if r["variant"] == "pos_a" and r["render_timing_ok"] == "1" and r["target_mode"] == "moving"]

    def one(r):
        cid = os.path.basename(os.path.realpath(r["file"]))[:-4]
        L = np.load(ROOT / name / "labels" / f"{cid}.npz")
        if L["amodal_n"].shape[1] == 0 or str(L["roles"][0]) != "target":
            return None
        c = L["center_px"][IDX, 0].astype(np.float32) / 16.0 - 0.5            # (32, 2)
        a = L["amodal_n"][IDX, 0].astype(np.float32); v = L["visible_n"][IDX, 0].astype(np.float32)
        vis = np.where(a > 0, v / np.maximum(a, 1), 0.0)
        return dict(c=c, vis=vis, mo=motion(r), occ=r["occ_type"] + "_" + r["occ_motion"], cam=r["camera"], set=name)
    with ThreadPoolExecutor(32) as ex:
        out = [x for x in ex.map(one, rows) if x is not None]
    return out


def feats(d):
    vis = d["vis"][:CTX]
    vs = np.nonzero(vis >= 0.9)[0]
    if len(vs) < 3:
        return None
    lv = int(vs.max())
    ts = vs[vs >= lv - 4]
    v = np.polyfit(ts, d["c"][ts], 1)[0]                                        # (2,) 토큰 / 샘플
    q = np.polyfit(vs, d["c"][vs], 2) if len(vs) >= 4 else None
    dirn = int(np.sign(v[0])) if abs(v[0]) > 1e-3 else 0
    hid = np.nonzero(d["vis"] < 0.1)[0]
    reap = [t for t in range(CTX, 32) if d["vis"][t] >= 0.5 and t > CTX]
    return dict(lv=lv, v=v, q=q, dir=dirn, speed=float(np.linalg.norm(v)), hid_len=int(len(hid)),
                t_reap=reap[0] if reap else -1, ctx_rel=None)


def main():
    res = {}
    for name in ("v3", "v3t"):
        tr = [d for d in load(name, "train")]; te = [d for d in load(name, "test")]
        for d in tr + te:
            d["f"] = feats(d)
        tr = [d for d in tr if d["f"]]; te = [d for d in te if d["f"]]
        # train 통계 (B1 · B1s)
        disp = defaultdict(list); scr = defaultdict(list)
        for d in tr:
            f = d["f"]
            disp[(f["dir"], d["mo"], f["lv"])].append(d["c"][CTX:] - d["c"][f["lv"]])
            scr[(f["dir"], d["mo"])].append(d["c"][CTX:])
        disp = {k: np.mean(v, 0) for k, v in disp.items()}; scr = {k: np.mean(v, 0) for k, v in scr.items()}
        disp_any = defaultdict(list)
        for d in tr:
            disp_any[(d["f"]["dir"], d["mo"])].append(d["c"][CTX:] - d["c"][d["f"]["lv"]])
        disp_any = {k: np.mean(v, 0) for k, v in disp_any.items()}
        # B4 이웃: 문맥 보이는 샘플의 (마지막 보인 자리 기준 상대 위치) — 샘플 lv−4..lv 다섯 점 (방향 · 운동 종류 같은 것 중)
        def ctxvec(d):
            f = d["f"]; ts = np.arange(f["lv"] - 4, f["lv"] + 1).clip(0)
            return (d["c"][ts] - d["c"][f["lv"]]).ravel()
        trv = defaultdict(list)
        for d in tr:
            trv[(d["f"]["dir"], d["mo"])].append((ctxvec(d), d["c"][CTX:] - d["c"][d["f"]["lv"]]))
        trv = {k: (np.stack([a for a, _ in v]), np.stack([b for _, b in v])) for k, v in trv.items()}
        errs = defaultdict(lambda: defaultdict(list))
        tf = np.arange(CTX, 32)
        for d in te:
            f = d["f"]; lv = f["lv"]; true = d["c"][CTX:]; base = d["c"][lv]
            preds = {"B0_stay": np.repeat(base[None], 16, 0)}
            k1 = (f["dir"], d["mo"], lv)
            preds["B1_memorized_offset"] = base + (disp[k1] if k1 in disp else disp_any.get((f["dir"], d["mo"]), np.zeros((16, 2))))
            preds["B1s_memorized_screen"] = scr.get((f["dir"], d["mo"]), np.repeat(base[None], 16, 0))
            preds["B2_const_vel"] = base + f["v"][None] * (tf - lv)[:, None]
            if f["q"] is not None:
                preds["B3_const_acc"] = np.stack([np.polyval(f["q"][:, j], tf) for j in range(2)], -1)
            key = (f["dir"], d["mo"])
            if key in trv:
                X, Y = trv[key]; dist = np.linalg.norm(X - ctxvec(d)[None], axis=1)
                nn = np.argsort(dist)[:5]
                preds["B4_nearest_neighbors"] = base + Y[nn].mean(0)
            for k_, p in preds.items():
                e = np.linalg.norm(p - true, axis=1)                             # (16,)
                for grp in ("all", d["mo"]):
                    errs[grp][k_].append(e)
        out = {"n_train": len(tr), "n_test": len(te)}
        for grp, dd in errs.items():
            out[grp] = {k_: {"err_median_tok": round(float(np.median(np.concatenate(v))), 2),
                             "err_end_median_tok": round(float(np.median([x[-1] for x in v])), 2),
                             "within1tok": round(float(np.mean(np.concatenate(v) <= 1.0)), 3), "n": len(v)} for k_, v in dd.items()}
        # 다양성
        sp = np.array([d["f"]["speed"] for d in tr]); hl = np.array([d["f"]["hid_len"] for d in tr])
        lvp = np.array([d["c"][d["f"]["lv"]] for d in tr]); endp = np.array([d["c"][31] for d in tr])
        out["diversity_train"] = {
            "ctx_end_screen_speed_tok_per_sample_pct10_50_90": [round(float(x), 3) for x in np.percentile(sp, [10, 50, 90])],
            "hidden_samples_pct10_50_90": [int(x) for x in np.percentile(hl, [10, 50, 90])],
            "disappear_pos_std_tok": [round(float(x), 2) for x in lvp.std(0)],
            "end_pos_std_tok": [round(float(x), 2) for x in endp.std(0)],
            "motion_counts": {k: int(v) for k, v in zip(*np.unique([d["mo"] for d in tr], return_counts=True))}}
        # 양자화 미래 경로 (VOBench 비교용: 미래 샘플 16–31 의 칸 열 · 행 반올림, 2 샘플마다)
        qp = lambda d: tuple(np.round(d["c"][CTX::2]).astype(int).ravel())
        trs = {qp(d) for d in tr}; tre = {tuple(np.round(d["c"][31]).astype(int)) for d in tr}
        out["path_overlap"] = {"unique_future_paths_train": len(trs), "test_path_in_train": round(float(np.mean([qp(d) in trs for d in te])), 3),
                               "test_end_cell_in_train": round(float(np.mean([tuple(np.round(d["c"][31]).astype(int)) in tre for d in te])), 3)}
        res[name] = out
        print(f"== {name}: train {len(tr)} · test {len(te)}", flush=True)
        print("  다양성", json.dumps(out["diversity_train"], ensure_ascii=False))
        print("  경로 겹침", json.dumps(out["path_overlap"], ensure_ascii=False))
        for grp in ["all"] + sorted(k for k in out if k.startswith(("flat", "ramp"))):
            print(f"  [{grp}] " + " | ".join(f"{k_.split('_', 1)[0]} {v['err_median_tok']:.2f}/{v['err_end_median_tok']:.2f}" for k_, v in out[grp].items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("[done]", OUT)


if __name__ == "__main__":
    main()
