#!/usr/bin/env python3
"""BlenderTrack v4vc 팔 · 목록 (2026-11-02) — 렌더 뒤: 렌더 타이밍 검사 → 학습 목록 (묶음 · 대조) → 시험 VoE 영상 · 목록.

렌더 타이밍: 렌더 라벨로 다시 잰 target 가려진 비율이 late 정의 (gen_design_v4vc.timing_ok, 화면 밖 상한 max_off) 를 지키나.
학습 목록 data_csv/blender_track_v4vc/arms/<이름>.npz (blender_track 로더 형식: clip · vis · root · layout mp4 · labels_root)
  fam  묶음 분할: 모든 단계가 렌더 타이밍을 통과한 묶음만, 묶음마다 K 단계 영상 + 빈 장면 1
  ind  대조 분할: 단계 하나 영상 (렌더 타이밍 통과) + 빈 장면 (상황 안 번호 % K == 0)
  두 목록의 target 있는 영상 수 · 빈 장면 수를 같게 자른다 (fam 은 묶음 통째로, ind 는 장면을 seed 0 으로 무작위).
시험 VoE (test 분할, 모든 단계 통과 · 이음 프레임 31 · 32 에서 모든 단계 target 완전히 가려짐) — 이음은 프레임 32 (문맥 끝 다음)
  A_k            가능 (속도 단계 k)                          B          가능 (빈 장면)
  vanish_k       A_k[0:32] + B[32:64]   사라짐 (불가능)      appear_k   B[0:32] + A_k[32:64]  나타남 (불가능)
  swap_k_j       A_k[0:32] + A_j[32:64] 속도 바꿔치기 (불가능, j ≠ k): 문맥의 속도 k 인데 속도 j 의 때 · 자리로 다시 나온다
  짝 (문맥이 같다): (A_k, vanish_k) · (B, appear_k) · (A_k, swap_k_j)
  출력 <out>/arms/test/<묶음>/*.mp4 · data_csv/blender_track_v4vc/voe_test/index.csv (영상마다 한 줄, 채점기 vobench_v1 형식)
  $P compose_v4vc.py --config configs/v4vc.yaml --workers 60
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_design_v4vc as G  # noqa: E402
from make_v3_meta import vis_of  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
SPLICE = 32
DATA_CSV = os.environ.get("BT_DATA_CSV") or f"{REPO}/data_csv"          # 11-02 Vast: 목록 자리를 env.sh 의 BT_DATA_CSV 로 (비우면 레포 data_csv, 예전과 같음)
DCSV = f"{DATA_CSV}/blender_track_v4vc"                       # main 에서 설정 이름으로 바꾼다 (빗면 v4vcr → blender_track_v4vcr)
CFG = {}


def _init(path):
    CFG["cfg"] = G.load_cfg(path, None)


def cid_of(cfg, split, idx):
    return f"{split}{cfg['splits'][split]['seed']}_{idx:06d}"


def render_timing(cfg, lab):
    z = np.load(lab)
    if z["amodal"].shape[1] == 0 or str(z["roles"][0]) != "target":
        return None, None, None
    a_, v_ = z["amodal_n"][:, 0].astype(float), z["visible_n"][:, 0].astype(float)
    hid = np.where(a_ > 0, 1.0 - v_ / np.maximum(a_, 1), 0.0)
    bb = z["bbox_px"][:, 0]; S_ = 256
    w_ = np.maximum(bb[:, 2] - bb[:, 0], 1e-6); h_ = np.maximum(bb[:, 3] - bb[:, 1], 1e-6)
    inw = np.clip(np.minimum(bb[:, 2], S_) - np.maximum(bb[:, 0], 0), 0, None) * np.clip(np.minimum(bb[:, 3], S_) - np.maximum(bb[:, 1], 0), 0, None)
    off = np.nan_to_num(1.0 - inw / (w_ * h_), nan=1.0)
    ok, ev = G.timing_ok(hid[::2], off[::2], cfg["timing"], float(cfg["velocity"]["max_off"]))
    hidden_splice = bool(z["visible_n"][SPLICE - 1, 0] == 0 and z["visible_n"][SPLICE, 0] == 0)
    return bool(ok), (list(ev) if ev else None), hidden_splice


def check_family(arg):
    split, f = arg
    cfg = CFG["cfg"]
    out = cfg["out"]
    slots = G.render_slots(cfg, split, f)
    if not os.path.exists(f"{out}/design/{cid_of(cfg, split, slots[0][0])}.json"):
        return None                                                     # 고르지 않은 묶음
    rows = []
    for idx, lev in slots:
        c = cid_of(cfg, split, idx)
        have = os.path.exists(f"{out}/clips/{c}.mp4") and os.path.exists(f"{out}/labels/{c}.npz")
        r = {"family": f, "idx": idx, "level": lev, "cid": c, "rendered": have, "timing_ok": None, "event": None, "hidden_splice": None}
        if have and lev != "B":
            r["timing_ok"], r["event"], r["hidden_splice"] = render_timing(cfg, f"{out}/labels/{c}.npz")
        rows.append(r)
    sp = json.load(open(f"{out}/design/{cid_of(cfg, split, slots[0][0])}.json"))
    return {"family": f, "slots": rows, "cell": sp["cell"], "factors": sp["factors"], "velocity": sp["velocity"]}


def write_mp4(path, frames, fps, crf, gop):
    H, W = frames.shape[1:3]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-crf", str(crf), "-g", str(gop), "-pix_fmt", "yuv420p", "-movflags", "+faststart", path + ".tmp.mp4"],
                         stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(frames).tobytes()); p.stdin.close(); p.wait()
    os.replace(path + ".tmp.mp4", path)


def make_test_family(fam):
    import decord
    cfg = CFG["cfg"]
    out, rc = cfg["out"], cfg["render"]
    K = int(cfg["velocity"]["levels"])
    f = fam["family"]
    bid = f"{cfg['name']}_test_f{f:06d}"
    d = f"{out}/arms/test/{bid}"
    os.makedirs(d, exist_ok=True)
    cl = {s["level"]: s["cid"] for s in fam["slots"]}
    V = {lv: decord.VideoReader(f"{out}/clips/{c}.mp4")[:].asnumpy() for lv, c in cl.items()}
    am = {}
    for lv, c in cl.items():
        if lv == "B":
            continue
        z = np.load(f"{out}/labels/{c}.npz")
        a = np.unpackbits(z["amodal"][:, 0], axis=-1).astype(bool)
        am[lv] = np.kron(a[SPLICE - 1] | a[SPLICE], np.ones((4, 4), bool))
    vids = []
    def link(name, c):
        lk = f"{d}/{name}.mp4"
        if os.path.lexists(lk):
            os.remove(lk)
        os.symlink(os.path.relpath(f"{out}/clips/{c}.mp4", d), lk)
        return lk
    def seam(x, y, m):
        return float(np.abs(x[SPLICE].astype(int) - y[SPLICE].astype(int))[~m].max())
    vids.append(dict(variant="B", plausible=1, pair="B", level="", level_fut="", file=link("B", cl["B"])))
    for k in range(K):
        vids.append(dict(variant=f"A_k{k}", plausible=1, pair=f"k{k}", level=k, level_fut=k, file=link(f"A_k{k}", cl[k])))
        p = f"{d}/vanish_k{k}.mp4"
        if not os.path.exists(p):
            write_mp4(p, np.concatenate([V[k][:SPLICE], V["B"][SPLICE:]]), cfg["fps"], rc["crf"], rc["gop"])
        vids.append(dict(variant=f"vanish_k{k}", plausible=0, pair=f"k{k}", level=k, level_fut="B", file=p, seam=seam(V[k], V["B"], am[k])))
        p = f"{d}/appear_k{k}.mp4"
        if not os.path.exists(p):
            write_mp4(p, np.concatenate([V["B"][:SPLICE], V[k][SPLICE:]]), cfg["fps"], rc["crf"], rc["gop"])
        vids.append(dict(variant=f"appear_k{k}", plausible=0, pair="B", level="B", level_fut=k, file=p, seam=seam(V["B"], V[k], am[k])))
        for j in range(K):
            if j == k:
                continue
            p = f"{d}/swap_k{k}_j{j}.mp4"
            if not os.path.exists(p):
                write_mp4(p, np.concatenate([V[k][:SPLICE], V[j][SPLICE:]]), cfg["fps"], rc["crf"], rc["gop"])
            vids.append(dict(variant=f"swap_k{k}_j{j}", plausible=0, pair=f"k{k}", level=k, level_fut=j, file=p, seam=seam(V[k], V[j], am[k] | am[j])))
    c, F, v = fam["cell"], fam["factors"], fam["velocity"]
    ev = {s["level"]: s["event"] for s in fam["slots"] if s["level"] != "B"}
    rows = []
    for x in vids:
        lv = x["level"] if x["level"] != "B" else ""
        rows.append({"video_id": f"{bid}_{x['variant']}", "file": x["file"], "file_name": f"{bid}_{x['variant']}", "block_id": bid,
                     "variant": x["variant"], "plausible": x["plausible"], "pair_id": x["pair"], "condition": f"{c['objset']}/{c['camera']}/{c['occ']}",
                     "objset": c["objset"], "camera": c["camera"], "occ": c["occ"], "occ_sub": c["occ_sub"],
                     "place": c.get("place", "flat"), "ramp_mode": c.get("ramp_mode", ""), "target_kind": c["target_kind"],
                     "target_shape": c["target_shape"], "count": c["count"], "level": x["level"], "level_fut": x["level_fut"],
                     "u_ctx": round(v["u_px_per_sample"][lv], 3) if lv != "" else "", "event_s": ev[lv][0] if lv != "" else "",
                     "event_e": ev[lv][1] if lv != "" else "", "s_in": v["s_in"], "seam": round(x.get("seam", 0.0), 1), "family": f})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--splits", default=None, help="기본 = 설정의 분할 전부")
    a = ap.parse_args()
    cfg = G.load_cfg(a.config, None)
    global DCSV
    DCSV = f"{DATA_CSV}/blender_track_{cfg['name']}"
    splits = (a.splits or " ".join(cfg["splits"])).split()
    s_fam = next((x for x in splits if "train" in x and not x.endswith("_ind")), None)
    s_ind = next((x for x in splits if x.endswith("_ind")), None)
    s_test = next((x for x in splits if "test" in x), None)
    K = int(cfg["velocity"]["levels"])
    os.makedirs(f"{DCSV}/arms", exist_ok=True)
    fams = {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.config,)) as ex:
        for sp in splits:
            NC = int(cfg["splits"][sp]["candidates"])
            fams[sp] = [r for r in ex.map(check_family, [(sp, f) for f in range(NC)], chunksize=8) if r is not None]
            json.dump(fams[sp], open(f"{DCSV}/families_{sp}.json", "w"), default=str)
            ok = [all(s["rendered"] and (s["level"] == "B" or (s["timing_ok"] and s["hidden_splice"])) for s in r["slots"]) for r in fams[sp]]
            nslot = sum(len(r["slots"]) for r in fams[sp]); nren = sum(s["rendered"] for r in fams[sp] for s in r["slots"])
            tok = [s["timing_ok"] for r in fams[sp] for s in r["slots"] if s["level"] != "B" and s["rendered"]]
            print(f"{sp}: 고른 묶음 {len(fams[sp])} · 렌더 {nren}/{nslot} · 렌더 타이밍 통과 (단계 영상) {np.mean(tok) if tok else float('nan'):.3f} · 묶음 전부 통과 {sum(ok)}", flush=True)
        # 학습 목록 (fam · ind) — 영상 수 맞추기
        out = cfg["out"]
        if s_fam and s_fam in fams and not s_ind:                         # 묶음 목록만 (빗면 2 차: 대조 팔 없음)
            fam_ok = [r for r in fams[s_fam] if all(s["rendered"] and (s["level"] == "B" or s["timing_ok"]) for s in r["slots"])]
            clips = [s["cid"] for r in fam_ok for s in r["slots"]]
            vis = np.stack([vis_of(f"{out}/labels/{c}.npz") for c in clips])
            np.savez(f"{DCSV}/arms/fam.npz", clip=np.array(clips), vis=vis, root=np.array(f"{out}/clips"), layout=np.array("mp4"), labels_root=np.array(f"{out}/labels"))
            print(f"학습 목록 fam: 묶음 {len(fam_ok)} · 영상 {len(clips)} → {DCSV}/arms/fam.npz", flush=True)
        if s_fam and s_ind and s_fam in fams and s_ind in fams:
            fam_ok = [r for r in fams[s_fam] if all(s["rendered"] and (s["level"] == "B" or s["timing_ok"]) for s in r["slots"])]
            ind_A = [(r, s) for r in fams[s_ind] for s in r["slots"] if s["level"] != "B" and s["rendered"] and s["timing_ok"]]
            ind_B = [(r, s) for r in fams[s_ind] for s in r["slots"] if s["level"] == "B" and s["rendered"]]
            nA = min(K * len(fam_ok), len(ind_A)); nB = min(len(fam_ok), len(ind_B))
            rng = np.random.default_rng(0)
            fam_keep = fam_ok[:nA // K]
            ind_A = [ind_A[i] for i in sorted(rng.choice(len(ind_A), nA, replace=False))]
            ind_B = [ind_B[i] for i in sorted(rng.choice(len(ind_B), nB, replace=False))]
            fam_clips = [s["cid"] for r in fam_keep for s in r["slots"] if s["level"] != "B"] + [s["cid"] for r in fam_keep[:nB] for s in r["slots"] if s["level"] == "B"]
            ind_clips = [s["cid"] for _, s in ind_A] + [s["cid"] for _, s in ind_B]
            for name, clips in (("fam", fam_clips), ("ind", ind_clips)):
                vis = np.stack([vis_of(f"{out}/labels/{c}.npz") for c in clips])
                np.savez(f"{DCSV}/arms/{name}.npz", clip=np.array(clips), vis=vis, root=np.array(f"{out}/clips"), layout=np.array("mp4"),
                         labels_root=np.array(f"{out}/labels"))
                lv = [int(s["level"]) for r in (fam_keep if name == "fam" else []) for s in r["slots"] if s["level"] != "B"] or [int(s["level"]) for _, s in ind_A]
                print(f"학습 목록 {name}: 영상 {len(clips)} (target 있음 {nA} · 빈 장면 {nB}) · 단계별 {np.bincount(lv).tolist()} → {DCSV}/arms/{name}.npz", flush=True)
        # 시험 VoE
        if s_test and s_test in fams:
            good = [r for r in fams[s_test] if all(s["rendered"] and (s["level"] == "B" or (s["timing_ok"] and s["hidden_splice"])) for s in r["slots"])]
            rows = [x for rr in ex.map(make_test_family, good, chunksize=2) for x in rr]
            os.makedirs(f"{DCSV}/voe_test", exist_ok=True)
            with open(f"{DCSV}/voe_test/index.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            sm = np.array([r["seam"] for r in rows if r["plausible"] == 0])
            print(f"시험 VoE: 묶음 {len(good)} · 영상 {len(rows)} · 이음 (target 밖 화소 최대 차) 중앙 {np.median(sm):.1f} · 95 % {np.percentile(sm, 95):.1f} → {DCSV}/voe_test/index.csv")


if __name__ == "__main__":
    main()
