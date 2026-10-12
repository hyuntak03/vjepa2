#!/usr/bin/env python3
"""VOBench_v1 asymmetry 판 (2026-10-10) — late 가림이 **문맥 끝에서 끝나도록** 창 시작을 k 샘플 (원본 3k 프레임) 미룬다.

원본 late: 가림 샘플 = [16−k, 16+k) (문맥 끝 k + 미래 앞 k, 대칭). 창을 k 샘플 밀면 가림 = [16−2k, 16) 전부 문맥 안 → 미래는 첫 샘플부터 물체가 보인다
(가능한 미래 pos 는 늘 물체가 있다; A|B 바꿈 (원본 샘플 16) 은 문맥 안 가림 구간에서 일어난다).

하네스의 PNG 경로는 클립별 시작 프레임을 지원하지 않아 (frames_start_column 은 mp4 전용) **심볼릭 링크 폴더**로 번호를 다시 매긴다:
  <out_frames>/<file_name>/{i:06d}.png -> <VOBench_v1>/<file_name>/{3k + i:06d}.png   (i = 0 .. 95 − 3k)
영상 길이 96 프레임이라 창은 28 샘플 (문맥 16 + 미래 12, stride 3) 이 최대다 (k = 3: 마지막 원본 9 + 81 = 90).
대상 = late 클립만 (864 arm). index 는 원본 index.csv 의 late 행 + `shift_frames` · `samples_asym` 열.
⚠️ 바꿈 (원본 48) 이후 A · B 렌더는 물체가 가려져 있어도 렌더 잡음만큼 다르다 (2026-10-10 실측: 화면 72 % 가 평균 0.72 단계, 10 단계 초과 0 %).
   밀린 문맥에는 바꿈 이후 프레임이 1–3 장 들어가므로, **불가능 arm 의 문맥 프레임 (새 번호 i < 48) 은 같은 쌍 가능 arm 의 프레임으로 링크**한다.
   → 문맥이 바이트 단위로 같고, 바꿈이 정확히 문맥 / 미래 경계에 온다. 미래 (i ≥ 48) 는 각자 자기 렌더.
문맥 무결성: 쌍마다 밀린 문맥 16 샘플 (원본 3k + 3j) 을 픽셀 단위로 비교. mismatch 가 있으면 죽는다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/data/build_vobench_asym.py
"""
from __future__ import annotations
import csv, json, os, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
SRC = Path("/data2/local_datasets/world/Benchmark/VOBench_v1")
DST = Path("/data2/local_datasets/world/Benchmark/VOBench_v1_asymmetry")
IDX_IN = REPO / "data_csv/vobench_v1/index.csv"
IDX_OUT = REPO / "data_csv/vobench_v1_asymmetry"
N_SAMPLES, STRIDE, CTX = 28, 3, 16


def main():
    rows = [r for r in csv.DictReader(open(IDX_IN)) if r["occlusion"] == "late"]
    pos_of = {(r["block_id"], r["pair_id"]): r["file_name"] for r in rows if r["plausible"] == "1"}
    t0 = time.time(); nl = 0
    for r in rows:
        k = int(r["k"]); sh = 3 * k
        s = r["samples"].replace("|", "")
        asym = s[k:k + N_SAMPLES]
        assert asym[:CTX].endswith("H" * (2 * k)) and "H" not in asym[CTX:], (r["video_id"], s, asym)   # 가림 = 문맥 끝 2k, 미래 전부 보임
        r["shift_frames"] = str(sh); r["samples_asym"] = asym[:CTX] + "|" + asym[CTX:]
        d = DST / r["file_name"]; d.mkdir(parents=True, exist_ok=True)
        for i in range(96 - sh):
            ln = d / f"{i:06d}.png"
            src_arm = pos_of[(r["block_id"], r["pair_id"])] if (r["plausible"] == "0" and i < CTX * STRIDE) else r["file_name"]
            tgt = SRC / src_arm / f"{sh + i:06d}.png"
            if ln.is_symlink() and os.readlink(ln) != str(tgt):
                ln.unlink()
            if not ln.is_symlink():
                os.symlink(tgt, ln); nl += 1
        assert (d / f"{(N_SAMPLES - 1) * STRIDE:06d}.png").exists()
    print(f"late arms {len(rows)} · 새 링크 {nl} ({time.time() - t0:.0f}s)")

    # 문맥 무결성 (밀린 문맥 16 샘플)
    pairs = defaultdict(list)
    for r in rows:
        pairs[(r["block_id"], r["pair_id"])].append(r)
    def check(key):
        v = pairs[key]; p_ = next(x for x in v if x["plausible"] == "1"); n_ = next(x for x in v if x["plausible"] == "0")
        bad = []
        for j in range(CTX):
            f = f"{j * STRIDE:06d}.png"
            if not np.array_equal(np.asarray(Image.open(DST / p_["file_name"] / f)), np.asarray(Image.open(DST / n_["file_name"] / f))):
                bad.append(j)
        return key, bad
    t0 = time.time()
    with ThreadPoolExecutor(32) as ex:
        res = list(ex.map(check, sorted(pairs)))
    mism = {f"{k[0]}/{k[1]}": b for k, b in res if b}
    audit = dict(n_pairs=len(res), n_frames_checked=len(res) * CTX, n_mismatch_pairs=len(mism), mismatch=mism, seconds=round(time.time() - t0, 1),
                 note="밀린 창 (원본 3k + 3j, j < 16) 의 문맥. 바꿈 (원본 48) 이후 프레임이 가린 채로 문맥에 들어간다")
    IDX_OUT.mkdir(parents=True, exist_ok=True)
    (IDX_OUT / "context_integrity.json").write_text(json.dumps(audit, indent=1))
    print(f"문맥 무결성: {len(res)} 쌍 × {CTX} 장 — mismatch {len(mism)} 쌍 ({audit['seconds']} s)")
    if mism:
        print("  예:", list(mism.items())[:5])
        raise SystemExit("문맥 mismatch — 가린 구간에서 A/B 픽셀이 다르다. 평가하지 않는다")
    with open(IDX_OUT / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"index {len(rows)} -> {IDX_OUT / 'index.csv'}")


if __name__ == "__main__":
    main()
