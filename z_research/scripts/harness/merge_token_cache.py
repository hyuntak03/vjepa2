#!/usr/bin/env python3
"""토큰 캐시 두 개를 **제자리에서** 이어 붙인다 (A 뒤에 B 를 append).

왜 필요한가 — `TokenCache.matches()` 는 `video_ids` **완전 일치**를 요구한다
(`evals/analysis_vlm/occlusion_identity/cache.py:57`). 그래서 조건을 더한 데이터셋은
이미 뽑아 둔 clip 까지 **전부 다시 추출**한다. v11(21,504) + early/mid(21,504) 의 경우
32,256~43,008 clip 을 새로 뽑아야 하는데, 이미 있는 절반을 재사용하면 GPU 시간과
디스크를 아낀다.

되는 이유 — `.npy` 는 C-order 라 axis 0 이어붙이기가 **바이트 이어붙이기**와 같다.
헤더는 64바이트 정렬로 패딩돼 있고 `21504 -> 43008` 은 둘 다 5자리라 **헤더 길이가 안 변한다**
(실측 118바이트 유지). 그래서 헤더의 shape 만 고쳐 쓰고 뒤에 붙이면 된다.

⚠️ **A 를 제자리에서 키운다. 실패하면 A 가 깨진다.** A 는 재추출로 복구 가능하지만
   32분(8 GPU)이 든다. `--dry-run` 으로 먼저 확인할 것.
⚠️ 자릿수가 바뀌면(예: 9,999 -> 10,000) 헤더 길이가 달라져 거부한다.
⚠️ base 하나를 붙일 때마다 B 쪽 파일을 지울지(`--free-as-you-go`) 정할 수 있다.
   디스크가 빠듯할 때 쓴다 — 붙이고 검증한 뒤에만 지운다.

병합 후 **인덱스 순서가 `A 의 video_ids + B 의 video_ids` 와 정확히 같아야 한다.**
`z_research/scripts/data/concat_index.py` 가 그 순서로 index.csv 를 만든다.

  python z_research/scripts/harness/merge_token_cache.py \
    --into /local_datasets/world/world_analysis/cache/v11_vith \
    --from /local_datasets/world/world_analysis/cache/v11_earlymid_vith \
    --rename-to v11_full_vith --free-as-you-go
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
from pathlib import Path
import numpy as np

CHUNK = 1 << 28          # 256 MiB


def hdr_of(p: Path):
    with open(p, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(f"{p}: npy 매직이 아니다")
        ver = f.read(2)
        hlen = int.from_bytes(f.read(2), "little")
        raw = f.read(hlen)
    d = eval(raw.decode())            # numpy 가 쓰는 형식 그대로 (dict 리터럴)
    return ver, hlen, raw, d, 10 + hlen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--into", type=Path, required=True, help="A: 여기에 덧붙인다 (제자리 수정)")
    ap.add_argument("--src", "--from", dest="src", type=Path, required=True, help="B: 붙일 캐시")
    ap.add_argument("--rename-to", default=None, help="병합 후 A 디렉토리 이름 (선택)")
    ap.add_argument("--free-as-you-go", action="store_true",
                    help="base 하나를 붙이고 검증한 뒤 B 쪽 파일을 지운다 (디스크가 빠듯할 때)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ma = json.load(open(a.into / "meta.json"))
    mb = json.load(open(a.src / "meta.json"))
    for k in ("embed_dim", "dtype"):
        if ma[k] != mb[k]:
            sys.exit(f"거부: {k} 가 다르다 ({ma[k]} vs {mb[k]})")
    if ma["base_counts"] != mb["base_counts"]:
        sys.exit(f"거부: base_counts 가 다르다\n  A {ma['base_counts']}\n  B {mb['base_counts']}")
    dup = set(ma["video_ids"]) & set(mb["video_ids"])
    if dup:
        sys.exit(f"거부: video_id 가 {len(dup)}개 겹친다 (예: {sorted(dup)[:3]})")
    na, nb = len(ma["video_ids"]), len(mb["video_ids"])
    print(f"  A {a.into.name}: {na} clip\n  B {a.src.name}: {nb} clip  ->  합계 {na + nb}")

    plan = []
    for base in ma["base_counts"]:
        pa, pb = a.into / f"{base}.npy", a.src / f"{base}.npy"
        va, ha, ra, da, oa = hdr_of(pa)
        vb, hb, rb, db, ob = hdr_of(pb)
        if da["descr"] != db["descr"] or da["fortran_order"] or db["fortran_order"]:
            sys.exit(f"거부: {base} dtype/order 불일치")
        if da["shape"][1:] != db["shape"][1:]:
            sys.exit(f"거부: {base} 토큰 shape 불일치 {da['shape']} vs {db['shape']}")
        new = (da["shape"][0] + db["shape"][0],) + da["shape"][1:]
        raw2 = ra.replace(str(da["shape"]).encode(), str(new).encode())
        if len(raw2) != len(ra):
            sys.exit(f"거부: {base} 헤더 길이가 바뀐다 ({len(ra)} -> {len(raw2)}). "
                     "자릿수가 늘어난 경우다. 이 스크립트로는 못 한다")
        exp_a, exp_b = oa + np.prod(da["shape"]) * 2, ob + np.prod(db["shape"]) * 2
        for p, e in ((pa, exp_a), (pb, exp_b)):
            if os.path.getsize(p) != e:
                sys.exit(f"거부: {p} 크기 이상 (={os.path.getsize(p)}, 기대 {e})")
        plan.append((base, pa, pb, ob, raw2, os.path.getsize(pb) - ob))
        print(f"    {base:12s} {da['shape']} + {db['shape']} -> {new}"
              f"   붙일 바이트 {(os.path.getsize(pb) - ob) / 2**30:.1f} GiB")

    plan.sort(key=lambda t: -t[5])            # 큰 것부터 (free-as-you-go 효과가 크다)
    st = os.statvfs(a.into)
    print(f"  여유 {st.f_bavail * st.f_frsize / 2**30:.0f} GiB  |  "
          f"최대 순간 증가 {max(t[5] for t in plan) / 2**30:.0f} GiB"
          f"{' (free-as-you-go)' if a.free_as_you_go else ''}")
    if a.dry_run:
        print("  --dry-run: 아무것도 안 썼다"); return

    for base, pa, pb, ob, raw2, nbytes in plan:
        t0 = time.time()
        with open(pb, "rb") as fb, open(pa, "r+b") as fa:
            fb.seek(ob); fa.seek(0, os.SEEK_END); done = 0
            while True:
                buf = fb.read(CHUNK)
                if not buf:
                    break
                fa.write(buf); done += len(buf)
                print(f"\r    {base:12s} {done / 2**30:7.1f} / {nbytes / 2**30:.1f} GiB"
                      f"  ({done / max(time.time() - t0, 1e-9) / 2**20:.0f} MiB/s)", end="")
            fa.flush(); os.fsync(fa.fileno())
            fa.seek(10); fa.write(raw2); fa.flush(); os.fsync(fa.fileno())
        print(f"\r    {base:12s} 완료 {nbytes / 2**30:.1f} GiB  {time.time() - t0:.0f}s" + " " * 20)
        arr = np.load(pa, mmap_mode="r")      # 헤더가 맞는지 즉시 확인
        assert arr.shape[0] == na + nb, f"{base}: 병합 후 shape 이상 {arr.shape}"
        del arr
        if a.free_as_you_go:
            os.remove(pb); print(f"    {base:12s} B 쪽 삭제 (여유 확보)")

    ma["video_ids"] = ma["video_ids"] + mb["video_ids"]
    json.dump(ma, open(a.into / "meta.json", "w"))
    print(f"  meta.json 갱신: video_ids {na + nb}")
    if a.rename_to:
        dst = a.into.parent / a.rename_to
        a.into.rename(dst); print(f"  이름 변경 -> {dst}")
    if a.free_as_you_go and a.src.exists() and not any(a.src.glob("*.npy")):
        # .npy 는 붙이면서 다 지웠다. 남은 meta.json 만 치운다.
        shutil.rmtree(a.src); print(f"  {a.src.name} 정리")
    print("  끝. 인덱스는 반드시 `A 순서 + B 순서` 로 만들 것 "
          "(z_research/scripts/data/concat_index.py)")


if __name__ == "__main__":
    main()
