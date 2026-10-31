#!/usr/bin/env python3
"""transfer_gap Q3 — E11 · E12 데이터 준비 (색인 · 변환 프레임 · 문맥 무결성). 2026-10-30, 실험 B [X2].

정의
  E11  v11 split test 부분집합 = vanish 위반 × {moving_occlusion, moving_occlusion_flat, static_occlusion, moving_visible,
       moving_visible_flat} (k 0–4, 조건당 448 영상). 행은 원 색인 그대로 (열 · 순서 유지), 프레임도 원본.
  E12  VOBench split v2 test (1,440 영상) 를 바꾼 판. 표본 프레임 32 장 (raw 0,3,…,93) 만, 원래 프레임 번호 이름으로 쓴다
       (그래서 하네스의 frames_start 0 · stride 3 이 그대로 맞는다). 크기 조정은 로더와 같은 커널
       (torch bilinear, align_corners=False, antialias 없음) 으로 256 px, uint8 반올림, PNG (압축 1).
       T0  항등 (같은 작성기 — 반올림 잡음의 관문)
       T3  448 px 창을 (x0,y0) ∈ {(0,0),(64,0),(0,64),(64,64)} 에서 잘라 256 으로. 물체 중심 (u, v) 가 표본 프레임 어디서든
           네 창의 교집합 [64, 448) 을 벗어나는 block 은 뺀다 (팔 A · B 둘 다 검사). 상자 규칙 (u±ext_u) 은 70 / 360 block 만 남아 버렸다
       T4  HSV 색상 +180° = 화소마다 c' = max(r,g,b) + min(r,g,b) − c (채도 · 명도 보존, 정확한 식) 를 512 px 에서 건 뒤 크기 조정
       T2  시간 역재생: 링크 f' → 원본 95 − f' (f' = 0,3,…,93, 원본 512 px 그대로 → 하네스가 크기 조정).
           짝을 다시 맺는다: (역 pos_a, 역 imp_ba) = pair A · (역 pos_b, 역 imp_ab) = pair B.
           관문: imp_ab 의 역 문맥 프레임 (원본 50,53,…,95) 이 pos_b 와, imp_ba 의 것이 pos_a 와 화소 단위로 같은 block 만 쓴다.
       T5  창 이동판 vobench_v1_asymmetry (가림이 문맥 끝에서 끝남, 창 c16t28) 의 행 중 source_block 이 split v2 test 인 것.
  무결성: 모든 matched pair (pos_a, imp_ab) · (pos_b, imp_ba) 의 문맥 16 장 (하네스가 읽는 파일) 을 화소 단위로 비교한다.

입력  data_csv/vobench_v1_split_v2/index_test.csv · /data2/local_datasets/world/Benchmark/VOBench_v1/<file_name>/{NNNNNN.png, ../label.json}
      data_csv/intphysgen_v11_split/index_test.csv · /local_datasets/world/world_analysis/IntPhysGen_v11 · data_csv/vobench_v1_asymmetry/index.csv
출력  data_csv/transfer_gap_v11sub/{index.csv, context_integrity.json}
      data_csv/transfer_gap_vob/{index_T0.csv, index_T3.csv, index_T2.csv, index_T5.csv, prep.json, context_integrity_T*.json}
      /data2/local_datasets/world/world_analysis/cache/transfer_gap/frames_{T0,T3_x0y0,T3_x64y0,T3_x0y64,T3_x64y64,T4,T2}/<file_name>/NNNNNN.png

재현
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/transfer_gap_frames.py v11sub            # CPU, 색인 + 무결성 (몇 분)
  $P z_research/scripts/analysis/transfer_gap_frames.py index             # CPU, 몇 초 (T3 걸러내기 · T2 관문 포함)
  $P z_research/scripts/analysis/transfer_gap_frames.py frames --procs 32 # CPU, 프레임 쓰기 (sbatch 로)
  $P z_research/scripts/analysis/transfer_gap_frames.py integrity --procs 32
  $P z_research/scripts/analysis/transfer_gap_frames.py ip1q --procs 32     # 잡음 바닥: IntPhys 1 dev 를 같은 작성기로 재양자화
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
VOB = Path("/data2/local_datasets/world/Benchmark/VOBench_v1")
V11 = Path("/local_datasets/world/world_analysis/IntPhysGen_v11")
ASYM = Path("/data2/local_datasets/world/Benchmark/VOBench_v1_asymmetry")
CACHE = Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap")
OUTV = REPO / "data_csv/transfer_gap_vob"
OUT11 = REPO / "data_csv/transfer_gap_v11sub"
SAMPLES = list(range(0, 96, 3))           # 0,3,…,93 (32 장)
CTX = SAMPLES[:16]                        # raw 0–45
WINS = {"T3_x0y0": (0, 0), "T3_x64y0": (64, 0), "T3_x0y64": (0, 64), "T3_x64y64": (64, 64)}
V11_CONDS = ("moving_occlusion", "moving_occlusion_flat", "static_occlusion", "moving_visible", "moving_visible_flat")


def rd(p):
    return list(csv.DictReader(open(p)))


def wr(p, rows, cols=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = cols or list(rows[0].keys())
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def load_png(p):
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"))


# ----------------------------------------------------------------------------- 무결성
def _cmp_pair(args):
    root, fa, fb, frames, pat = args
    bad = []
    for f in frames:
        a = load_png(os.path.join(root, pat.format(file_name=fa, frame=f)))
        b = load_png(os.path.join(root, pat.format(file_name=fb, frame=f)))
        if a.shape != b.shape or not np.array_equal(a, b):
            bad.append(f)
    return fa, fb, bad


def integrity(rows, root, frames, procs, pat="{file_name}/{frame:06d}.png"):
    """matched pair (block, pair_id) 마다 가능 · 불가능 문맥 프레임 화소 비교."""
    by = defaultdict(dict)
    for r in rows:
        by[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r["file_name"]
    jobs = [(str(root), g["pos"], g["imp"], frames, pat) for g in by.values() if "pos" in g and "imp" in g]
    t0 = time.time()
    with ProcessPoolExecutor(procs) as ex:
        res = list(ex.map(_cmp_pair, jobs, chunksize=8))
    mism = [{"pos": a, "imp": b, "frames": f} for a, b, f in res if f]
    return {"frames_root": str(root), "n_pairs": len(jobs), "context_frames_raw": frames, "n_frames_checked": len(jobs) * len(frames),
            "n_mismatch_pairs": len(mism), "mismatches": mism[:50], "seconds": round(time.time() - t0, 1),
            "checked": time.strftime("%Y-%m-%d %H:%M")}


# ----------------------------------------------------------------------------- E11
def cmd_v11sub(a):
    rows = rd(REPO / "data_csv/intphysgen_v11_split/index_test.csv")
    sub = [r for r in rows if r["violation_type"] == "vanish" and r["condition"] in V11_CONDS]
    wr(OUT11 / "index.csv", sub, list(rows[0].keys()))
    cnt = defaultdict(int)
    for r in sub:
        cnt[(r["condition"], r["sym_k"])] += 1
    print(f"v11sub {len(sub)} 영상", dict(sorted(cnt.items())))
    ci = integrity(sub, V11, CTX, a.procs)
    ci["counts"] = {f"{c}|k{k}": n for (c, k), n in sorted(cnt.items())}
    json.dump(ci, open(OUT11 / "context_integrity.json", "w"), indent=1)
    print("integrity", {k: ci[k] for k in ("n_pairs", "n_frames_checked", "n_mismatch_pairs", "seconds")})


# ----------------------------------------------------------------------------- E12 색인
def labels(file_name):
    return json.load(open(VOB / Path(file_name).parent / "label.json"))["labels"]


def cmd_index(a):
    rows = rd(REPO / "data_csv/vobench_v1_split_v2/index_test.csv")
    cols = list(rows[0].keys())
    OUTV.mkdir(parents=True, exist_ok=True)
    wr(OUTV / "index_T0.csv", rows, cols)
    prep = {"n_T0": len(rows)}
    # T3: 물체 상자가 표본 프레임 어디서든 [64, 448) 밖이면 block 제외 (팔 A · B)
    blocks = defaultdict(list)
    for r in rows:
        blocks[r["block_id"]].append(r)
    keep_box, why = set(), defaultdict(int)
    keep_c = set()
    for b, rs in blocks.items():
        lab = labels(rs[0]["file_name"])
        ok_box, ok_ctr = True, True
        for arm in ("A", "B"):
            L = lab.get(arm) or []                                                # permanence 빈 팔은 라벨이 비어 있다
            for f in SAMPLES:
                if f >= len(L) or L[f].get("u") is None:
                    continue
                e = L[f]
                u, v = float(e["u"]), float(e["v"])
                eu, ev = float(e.get("ext_u", e.get("r_px", 0))), float(e.get("ext_v", e.get("r_px", 0)))
                if not (64 <= u - eu and u + eu < 448 and 64 <= v - ev and v + ev < 448):
                    ok_box = False
                if not (64 <= u < 448 and 64 <= v < 448):
                    ok_ctr = False
        if ok_box:
            keep_box.add(b)
        if ok_ctr:
            keep_c.add(b)
        why["box_out" if not ok_box else "box_in"] += 1
    # ⚠️ 2026-10-30: 상자 규칙은 70 / 360 block (late 칸당 3) 만 남아 판정이 안 된다 → 중심 규칙 (210 block) 을 쓴다. 상자 규칙 수는 기록만.
    t3 = [r for r in rows if r["block_id"] in keep_c]
    wr(OUTV / "index_T3.csv", t3, cols)
    keep = keep_c
    prep["T3"] = {"rule": "물체 중심 (u, v) 가 표본 프레임 전부 · 팔 A·B 전부에서 [64,448) 안 (상자 규칙 u±ext 는 70 block 만 남아 버림)",
                  "n_block_all": len(blocks), "n_block_keep_box_rule": len(keep_box),
                  "n_block_keep_center": len(keep_c), "n_video": len(t3),
                  "by_type": {}}
    bt = defaultdict(lambda: [0, 0])
    for b, rs in blocks.items():
        k = rs[0]["condition"]
        bt[k][1] += 1
        bt[k][0] += b in keep
    prep["T3"]["by_type"] = {k: f"{v[0]}/{v[1]}" for k, v in sorted(bt.items())}
    print("T3 중심 규칙", len(keep_c), "/", len(blocks), "(상자 규칙", len(keep_box), ")")
    # T5: 창 이동판 ∩ split v2 test (source_block 기준)
    srcs = {r["source_block"] for r in rows}
    asym = rd(REPO / "data_csv/vobench_v1_asymmetry/index.csv")
    t5 = [r for r in asym if r["source_block"] in srcs]
    wr(OUTV / "index_T5.csv", t5, list(asym[0].keys()))
    prep["T5"] = {"n_video": len(t5), "n_block": len({r["block_id"] for r in t5}), "n_asym_all": len(asym)}
    print("T5", prep["T5"])
    json.dump(prep, open(OUTV / "prep.json", "w"), indent=1, ensure_ascii=False)


# ----------------------------------------------------------------------------- E12 T2 (관문 + 링크)
REV_CTX = [95 - f for f in CTX]           # 역 영상의 문맥 = 원본 95,92,…,50


def _t2_gate(args):
    b, fn = args                          # fn: variant → file_name
    bad = []
    for imp, pos in (("imp_ab", "pos_b"), ("imp_ba", "pos_a")):
        for f in REV_CTX:
            x = load_png(VOB / fn[imp] / f"{f:06d}.png")
            y = load_png(VOB / fn[pos] / f"{f:06d}.png")
            if not np.array_equal(x, y):
                bad.append((imp, f))
                break
    return b, bad


def cmd_t2(a):
    rows = rd(REPO / "data_csv/vobench_v1_split_v2/index_test.csv")
    cols = list(rows[0].keys())
    by = defaultdict(dict)
    for r in rows:
        by[r["block_id"]][r["variant"]] = r
    jobs = [(b, {v: r["file_name"] for v, r in vs.items()}) for b, vs in by.items() if len(vs) == 4]
    with ProcessPoolExecutor(a.procs) as ex:
        res = list(ex.map(_t2_gate, jobs, chunksize=4))
    ok = {b for b, bad in res if not bad}
    fails = [{"block": b, "bad": bad} for b, bad in res if bad]
    print(f"T2 관문: 통과 {len(ok)} / {len(jobs)}")
    out = []
    root = CACHE / "frames_T2"
    for b in sorted(ok, key=int):
        vs = by[b]
        # 역 영상: 원래 plausible 그대로 (역 pos 는 가능, 역 imp 는 불가능). 짝: A = (pos_a, imp_ba), B = (pos_b, imp_ab)
        for v, pid in (("pos_a", "A"), ("imp_ba", "A"), ("pos_b", "B"), ("imp_ab", "B")):
            r = dict(vs[v])
            r["pair_id"] = pid
            r["file_name"] = r["file_name"]
            out.append(r)
            d = root / r["file_name"]
            d.mkdir(parents=True, exist_ok=True)
            for f in SAMPLES:
                lk = d / f"{f:06d}.png"
                if not lk.is_symlink():
                    os.symlink(VOB / r["file_name"] / f"{95 - f:06d}.png", lk)
    wr(OUTV / "index_T2.csv", out, cols)
    prep = json.load(open(OUTV / "prep.json")) if (OUTV / "prep.json").exists() else {}
    prep["T2"] = {"gate": "역 문맥 (원본 50..95 표본) 에서 imp_ab == pos_b, imp_ba == pos_a 화소 동일", "n_block": len(jobs),
                  "n_pass": len(ok), "fails": fails[:30], "pairs": "A = (rev pos_a, rev imp_ba), B = (rev pos_b, rev imp_ab)"}
    json.dump(prep, open(OUTV / "prep.json", "w"), indent=1, ensure_ascii=False)
    if out:
        ci = integrity(out, root, CTX, a.procs)
        json.dump(ci, open(OUTV / "context_integrity_T2.json", "w"), indent=1)
        print("T2 integrity", ci["n_pairs"], ci["n_mismatch_pairs"])


# ----------------------------------------------------------------------------- E12 프레임 쓰기
def _resize(x):
    """x (H, W, 3) uint8 or float [0,1] → 256 px uint8 (로더 커널: bilinear, align_corners=False, antialias 없음)."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float()
    if x.dtype == np.uint8:
        t = t / 255.0
    if t.shape[-1] != 256 or t.shape[-2] != 256:
        t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
    return (t[0].permute(1, 2, 0).clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()


def _save(arr, p):
    from PIL import Image
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(p, compress_level=1)


def _write_video(args):
    import torch
    torch.set_num_threads(1)
    fn, which = args
    n = 0
    for f in SAMPLES:
        outs = {}
        for T in which:
            p = CACHE / f"frames_{T}" / fn / f"{f:06d}.png"
            if not p.exists():
                outs[T] = p
        if not outs:
            continue
        img = load_png(VOB / fn / f"{f:06d}.png")
        x = img.astype(np.float32) / 255.0
        for T, p in outs.items():
            if T == "T0":
                y = _resize(x)
            elif T in WINS:
                x0, y0 = WINS[T]
                y = _resize(x[y0:y0 + 448, x0:x0 + 448])
            elif T == "T4":
                mx, mn = x.max(-1, keepdims=True), x.min(-1, keepdims=True)
                y = _resize(mx + mn - x)
            else:
                raise ValueError(T)
            _save(y, p)
            n += 1
    return n


def cmd_frames(a):
    rows = rd(OUTV / "index_T0.csv")
    t3 = {r["file_name"] for r in rd(OUTV / "index_T3.csv")}
    which = a.T.split(",")
    jobs = []
    for r in rows:
        w = [T for T in which if T in ("T0", "T4") or (T in WINS and r["file_name"] in t3)]
        if w:
            jobs.append((r["file_name"], w))
    t0 = time.time()
    n = 0
    with ProcessPoolExecutor(a.procs) as ex:
        for i, k in enumerate(ex.map(_write_video, jobs, chunksize=2)):
            n += k
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} 영상, {n} 장, {time.time() - t0:.0f}s", flush=True)
    print(f"frames 끝: {n} 장 {time.time() - t0:.0f}s")


def cmd_integrity(a):
    res = {}
    for T in a.T.split(","):
        rows = rd(OUTV / ("index_T3.csv" if T in WINS else "index_T0.csv"))
        ci = integrity(rows, CACHE / f"frames_{T}", CTX, a.procs)
        json.dump(ci, open(OUTV / f"context_integrity_{T}.json", "w"), indent=1)
        res[T] = (ci["n_pairs"], ci["n_mismatch_pairs"])
        print(T, res[T], flush=True)
    # 작성기 관문 (CPU): T0 프레임 vs 원본을 로더 식으로 크기 조정한 값 — 반올림 오차 최대 0.5/255
    rows = rd(OUTV / "index_T0.csv")[:: max(1, len(rd(OUTV / "index_T0.csv")) // 24)]
    md = 0.0
    for r in rows:
        for f in (0, 45, 93):
            o = load_png(VOB / r["file_name"] / f"{f:06d}.png").astype(np.float32) / 255.0
            import torch
            import torch.nn.functional as F
            ref = F.interpolate(torch.from_numpy(o).permute(2, 0, 1)[None], size=(256, 256), mode="bilinear",
                                align_corners=False)[0].permute(1, 2, 0).numpy()
            w = load_png(CACHE / "frames_T0" / r["file_name"] / f"{f:06d}.png").astype(np.float32) / 255.0
            md = max(md, float(np.abs(ref - w).max()))
    res["T0_writer_max_abs_err_01"] = md
    print("T0 writer max |Δ| (0–1 범위):", md)
    json.dump(res, open(OUTV / "integrity_summary.json", "w"), indent=1)


# ----------------------------------------------------------------------------- 잡음 바닥: IntPhys 1 재양자화 (E12 T0 관문 실패 뒤 추가, 2026-10-30)
IP1 = Path("/local_datasets/world/world_analysis/IntPhys1_dev_frame_png")


def _ip1_video(d):
    import torch
    torch.set_num_threads(1)
    n = 0
    for p in sorted((IP1 / d / "scene").glob("scene_*.png")):
        q = CACHE / "frames_ip1_T0" / d / "scene" / p.name
        if q.exists():
            continue
        _save(_resize(load_png(p).astype(np.float32) / 255.0), q)
        n += 1
    return n


def cmd_ip1q(a):
    """IntPhys 1 dev 프레임을 같은 작성기로 256 px uint8 로 다시 쓴다 (내용 변화 없음, 반올림 ≤ 0.5/255 만). 잡음 바닥 측정용."""
    rows = rd("/local_datasets/world/world_analysis/IntPhys1_dev_videos/index.csv")
    vids = sorted({f"{r['block']}/{r['quadruplet']}/{r['run']}" for r in rows})
    t0 = time.time()
    with ProcessPoolExecutor(a.procs) as ex:
        n = sum(ex.map(_ip1_video, vids, chunksize=4))
    print(f"ip1q: {len(vids)} 영상, {n} 장, {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["v11sub", "index", "t2", "frames", "integrity", "ip1q"])
    ap.add_argument("--procs", type=int, default=min(32, os.cpu_count() or 4))
    ap.add_argument("--T", default="T0,T3_x0y0,T3_x64y0,T3_x0y64,T3_x64y64,T4")
    a = ap.parse_args()
    {"v11sub": cmd_v11sub, "index": cmd_index, "t2": cmd_t2, "frames": cmd_frames, "integrity": cmd_integrity, "ip1q": cmd_ip1q}[a.cmd](a)


if __name__ == "__main__":
    main()
