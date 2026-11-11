#!/usr/bin/env python3
"""BlenderTrack v5mc 묶기 (2026-11-02) — 렌더 뒤: 렌더 타이밍 검사 → 최종 폴더로 옮기기 (tar 친화) → 학습 목록 → 시험 VoE 영상 · 목록 → README.

최종 폴더 (<data_root>, 이 폴더 하나를 tar 로 묶는다; 작업 공간 <out> = work/ 는 묶지 않는다)
  README.md                          구조 · 정의 · 재현
  configs/v5mc.yaml                  설계 설정 (그대로 복사)
  lists/train_all.npz                학습 목록 (clip = data_root 기준 상대 경로, 확장자 없음 · vis · layout mp4 · root / labels_root = 만든 때의 절대 경로)
  lists/train_{speed,accel,grid}.npz 묶음 종류별
  lists/{train,test,probe}_manifest.csv   영상마다 한 줄 (묶음 · 단계 · 운동 값 · 가림 샘플 · 상대 경로)
  lists/test_voe_index.csv           시험 VoE (채점기 vobench_v1 형식: file = 절대 경로, file_rel = 상대 경로)
  {train,test,probe}/<묶음 종류>/shard_<묶음 번호 // 100>/<clip>.{mp4,npz,json}   — 폴더 하나에 영상 약 600 (파일 약 1,800)
  test_voe/<묶음 종류>/<묶음>/<변이>.mp4   이어 붙인 불가능 영상 (가능 영상은 위 test/ 의 원본을 가리킨다 — 링크 없음)
옮기기는 같은 파일 시스템 안 os.rename (복사 없음). 렌더 타이밍을 통과하지 못한 묶음은 work/ 에 남긴다 (목록에 없음).
  $P compose_v5mc.py --config configs/v5mc.yaml --workers 64
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_design_v5mc as G  # noqa: E402
from make_v3_meta import vis_of  # noqa: E402

SPLICE = 32
GROUP = {"s": "train", "t": "test", "p": "probe"}
CFG = {}


def _init(cfg):
    CFG["cfg"] = cfg


def render_timing(lab, W, max_off):
    z = np.load(lab)
    if z["amodal"].shape[1] == 0 or str(z["roles"][0]) != "target":
        return None, None, None
    a_, v_ = z["amodal_n"][:, 0].astype(float), z["visible_n"][:, 0].astype(float)
    hid = np.where(a_ > 0, 1.0 - v_ / np.maximum(a_, 1), 0.0)
    bb = z["bbox_px"][:, 0]; S_ = 256
    w_ = np.maximum(bb[:, 2] - bb[:, 0], 1e-6); h_ = np.maximum(bb[:, 3] - bb[:, 1], 1e-6)
    inw = np.clip(np.minimum(bb[:, 2], S_) - np.maximum(bb[:, 0], 0), 0, None) * np.clip(np.minimum(bb[:, 3], S_) - np.maximum(bb[:, 1], 0), 0, None)
    off = np.nan_to_num(1.0 - inw / (w_ * h_), nan=1.0)
    ok, ev = G.timing_ok(hid[::2], off[::2], W, max_off)
    return bool(ok), (list(ev) if ev else None), bool(z["visible_n"][SPLICE - 1, 0] == 0 and z["visible_n"][SPLICE, 0] == 0)


def check(arg):
    cid, is_b = arg
    cfg = CFG["cfg"]; out = cfg["out"]
    have = all(os.path.exists(f"{out}/{d}/{cid}.{e}") for d, e in (("clips", "mp4"), ("labels", "npz"), ("specs", "json")))
    if not have:
        return cid, False, None, None, None
    if is_b:
        return cid, True, True, None, None
    ok, ev, hs = render_timing(f"{out}/labels/{cid}.npz", cfg["timing"], float(cfg["velocity"]["max_off"]))
    return cid, True, ok, ev, hs


def move(arg):
    cid, dst = arg
    out = CFG["cfg"]["out"]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for d, e in (("clips", "mp4"), ("labels", "npz"), ("specs", "json")):
        src = f"{out}/{d}/{cid}.{e}"
        if os.path.exists(src):
            os.rename(src, f"{dst}.{e}")
    return cid


def write_mp4(path, frames, fps, crf, gop):
    H, W = frames.shape[1:3]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-crf", str(crf), "-g", str(gop), "-pix_fmt", "yuv420p", "-movflags", "+faststart", path + ".tmp.mp4"],
                         stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(frames).tobytes()); p.stdin.close(); p.wait()
    os.replace(path + ".tmp.mp4", path)


def make_voe(fam):
    """시험 묶음 하나 → 이어 붙인 불가능 영상 + 목록 줄 (가능 영상 = 옮긴 원본)."""
    import decord
    cfg = CFG["cfg"]; DR = cfg["data_root"]; rc = cfg["render"]
    rows_m = fam["rows"]                                                # 단계 키 → manifest 줄 (rel 경로 포함)
    ftype = fam["ftype"]
    bid = f"v5mc_{fam['split']}_f{fam['family']:06d}"
    d = f"{DR}/test_voe/{ftype}/{bid}"
    os.makedirs(d, exist_ok=True)
    V = {k: decord.VideoReader(f"{DR}/{r['rel']}.mp4")[:].asnumpy() for k, r in rows_m.items()}
    mot = [k for k in rows_m if not k.startswith("B")]
    bkey = {}
    for k, r in rows_m.items():
        if k.startswith("B"):
            for kk in (r.get("b_for") or mot):
                bkey[kk] = k
    out = []

    def add(variant, plaus, pair, path, lvl, lvl_fut):
        out.append({"video_id": f"{bid}_{variant}", "file": path, "file_rel": os.path.relpath(path, DR), "file_name": f"{bid}_{variant}", "block_id": bid,
                    "variant": variant, "plausible": plaus, "pair_id": pair, "ftype": ftype, "level": lvl, "level_fut": lvl_fut, "family": fam["family"],
                    **{k: fam[k] for k in ("place", "objset", "camera", "occ", "ramp_mode")},
                    "info": json.dumps(rows_m.get(lvl, {}).get("level_info"))})
    for b in sorted(set(bkey.values())):
        add(f"pos_{b}", 1, b, f"{DR}/{rows_m[b]['rel']}.mp4", b, b)
    for k in mot:
        b = bkey[k]
        add(f"pos_{k}", 1, k, f"{DR}/{rows_m[k]['rel']}.mp4", k, k)
        p = f"{d}/vanish_{k}.mp4"
        if not os.path.exists(p):
            write_mp4(p, np.concatenate([V[k][:SPLICE], V[b][SPLICE:]]), cfg["fps"], rc["crf"], rc["gop"])
        add(f"vanish_{k}", 0, k, p, k, b)
        p = f"{d}/appear_{k}.mp4"
        if not os.path.exists(p):
            write_mp4(p, np.concatenate([V[b][:SPLICE], V[k][SPLICE:]]), cfg["fps"], rc["crf"], rc["gop"])
        add(f"appear_{k}", 0, b, p, b, k)
    # 바꿔치기: 같은 장면 (같은 쐐기) 의 움직이는 단계끼리만. grid 는 같은 가속도 줄 · 같은 속력 열만
    for k in mot:
        for j in mot:
            if j == k or "static" in k or "static" in j or bkey[k] != bkey[j]:
                continue
            if ftype == "grid" and not (k[1] == j[1] or k[2] == j[2]):
                continue
            p = f"{d}/swap_{k}_{j}.mp4"
            if not os.path.exists(p):
                write_mp4(p, np.concatenate([V[k][:SPLICE], V[j][SPLICE:]]), cfg["fps"], rc["crf"], rc["gop"])
            add(f"swap_{k}_{j}", 0, k, p, k, j)
    return out


README = """# BlenderTrack v5_motion_control (v5mc) — 같은 장면에서 운동만 바꾼 묶음

만든 날 2026-11-02 · 설계 gen_design_v5mc.py · 설정 configs/v5mc.yaml · 계획 predictor_lab/v4vc/PLAN_v5_motion_control_2026-11-02.md

## 한 영상
256 × 256 px · 20 fps · 64 장 (3.2 초) mp4 (Blender 5.0.1 Cycles 16 샘플 + 잡음 제거 · 그림자). 모델은 2 장마다 하나 → 32 샘플 (문맥 16 + 미래 16).
같은 이름의 .npz = 물체 라벨 (프레임마다 화면 중심 · 상자 · amodal · visible 마스크 · 판 각도), .json = 설계 spec (장면 · 운동 전부).

## 묶음
묶음 = 장면 하나 (카메라 · 환경 · 가리는 것 · 물체 생김새 · 방향 · 그 밖 물체) 에서 운동만 바꾼 영상 + 같은 장면에서 물체만 뺀 빈 장면 (B).
모든 단계가 같은 자리를 같은 시각에 지나 완전히 가려지기 시작하고 (late, 샘플 14–15) 창 끝 전에 완전히 다시 보인다.
- speed: 가려질 때 속력 5 단계 (가장 느린 : 가장 빠른 = 3 · 2.5 · 2 중 하나). 평면 trapdoor 는 정지 단계 (k_static) 하나 더.
- accel: 속력 같음, 가속도 4 단계 — 평면 = 보이지 않는 일정한 힘 (감속 · 가속), 빗면 = 경사 각도 (빈 장면도 각도마다 B0 … ).
- grid: 속력 3 × 가속도 3 (키 k<속력><가속도>).

## 폴더
- {train,test,probe}/<묶음 종류>/shard_<묶음 번호 // 100>/<clip>.{mp4,npz,json}
- test_voe/<묶음 종류>/<묶음>/{vanish,appear,swap}_*.mp4 — 시험 불가능 영상 (문맥 끝 프레임 32 에서 이어 붙임)
- lists/: train_all.npz (+ 종류별) · *_manifest.csv · test_voe_index.csv. 경로는 이 폴더 기준 상대 (file_rel · clip). npz 의 root 는 만든 곳의 절대 경로라
  다른 곳에 풀면 root 만 새 자리로 바꾸면 된다.

## 숫자
{numbers}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(a.config))
    out, DR = cfg["out"], cfg["data_root"]
    os.makedirs(f"{DR}/lists", exist_ok=True); os.makedirs(f"{DR}/configs", exist_ok=True)
    shutil.copy(a.config, f"{DR}/configs/v5mc.yaml")
    nums = {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(cfg,)) as ex:
        all_rows = defaultdict(list)
        for split in cfg["splits"]:
            mp = f"{out}/manifest_{split}.json"
            if not os.path.exists(mp):
                continue
            man = json.load(open(mp))
            res = {r[0]: r for r in ex.map(check, [(m["cid"], m["level"].startswith("B")) for m in man], chunksize=16)}
            fams = defaultdict(list)
            for m in man:
                m["split"] = split
                fams[m["family"]].append(m)
            good, moves = [], []
            grp = GROUP[split[0]]
            for f, ms in sorted(fams.items()):
                ok = all(res[m["cid"]][1] and res[m["cid"]][2] for m in ms)
                if not ok:
                    continue
                good.append(f)
                for m in ms:
                    rel = f"{grp}/{m['ftype']}/shard_{f // 100:03d}/{m['cid']}"
                    m["rel"] = rel
                    m["event_render"] = res[m["cid"]][3]
                    m["hidden_splice"] = res[m["cid"]][4]
                    moves.append((m["cid"], f"{DR}/{rel}"))
                    all_rows[grp].append(m)
            list(ex.map(move, moves, chunksize=32))
            nv = sum(len(fams[f]) for f in good)
            nums[split] = {"families": len(good), "clips": nv, "of_selected": len(fams)}
            print(f"{split}: 묶음 {len(good)} / {len(fams)} 통과 · 영상 {nv} → {DR}/{grp}/", flush=True)
        # 목록
        for grp, rows in all_rows.items():
            cols = ["rel", "cid", "family", "level", "ftype", "place", "objset", "camera", "occ", "ramp_mode", "kind", "v_hide", "level_info", "event_render", "b_for"]
            with open(f"{DR}/lists/{grp}_manifest.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
                for r in rows:
                    w.writerow({**r, "level_info": json.dumps(r.get("level_info")), "b_for": json.dumps(r.get("b_for"))})
        tr = all_rows.get("train", [])
        for name, sel in [("all", tr)] + [(ft, [r for r in tr if r["ftype"] == ft]) for ft in ("speed", "accel", "grid")]:
            if not sel:
                continue
            clips = [r["rel"] for r in sel]
            vis = np.stack(list(ex.map(vis_of, [f"{DR}/{c}.npz" for c in clips], chunksize=64)))
            np.savez(f"{DR}/lists/train_{name}.npz", clip=np.array(clips), vis=vis, root=np.array(DR), layout=np.array("mp4"), labels_root=np.array(DR))
            print(f"학습 목록 {name}: 영상 {len(clips)} → {DR}/lists/train_{name}.npz", flush=True)
        # 시험 VoE
        te = all_rows.get("test", [])
        fams = defaultdict(dict)
        meta = {}
        for r in te:
            key = (r["split"], r["family"])
            fams[key][r["level"]] = r
            meta[key] = r
        jobs = [{"split": k[0], "family": k[1], "ftype": meta[k]["ftype"], "rows": v,
                 **{c: meta[k][c] for c in ("place", "objset", "camera", "occ", "ramp_mode")}} for k, v in fams.items()]
        vrows = [x for rr in ex.map(make_voe, jobs, chunksize=2) for x in rr]
        if vrows:
            with open(f"{DR}/lists/test_voe_index.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(vrows[0])); w.writeheader(); w.writerows(vrows)
            print(f"시험 VoE: 묶음 {len(jobs)} · 영상 {len(vrows)} → {DR}/lists/test_voe_index.csv", flush=True)
    txt = "\n".join(f"- {k}: 묶음 {v['families']} (고른 {v['of_selected']} 중 렌더 타이밍 통과) · 영상 {v['clips']}" for k, v in nums.items())
    txt += f"\n- 학습 영상 합 {sum(v['clips'] for k, v in nums.items() if k.startswith('s_'))}"
    open(f"{DR}/README.md", "w").write(README.replace("{numbers}", txt))
    json.dump(nums, open(f"{DR}/lists/counts.json", "w"), indent=1)
    print("V5MC_PACK_DONE")


if __name__ == "__main__":
    main()
