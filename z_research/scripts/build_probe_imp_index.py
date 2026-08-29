#!/usr/bin/env python3
"""index_probe.csv -> index_probe_imp.csv. 불가능 변이를 probing 대상으로 여는 인덱스.

왜 별도 인덱스가 필요한가
------------------------
"바뀐 shape/color 를 target encoder 가 읽어내는가" 를 물으려면 두 가지가 필요하다.

1) **불가능 변이를 probing 대상에 넣기.** 기존 attn_probe 는 block_types:[obj] 로
   막아 뒀다. 그 근거(§1-4)는 캐시로 재확인했다 — z(ctx_masked)·p(predictor) 는
   문맥일치 쌍에서 원소의 99.5% 가 비트 단위로 같고 평균 |Δ| = 0.0000 이라
   **정말 중복**이다. 하지만 h(target encoder) 는 32프레임 전부를 보므로 다르다:
       h(imp_ab)[미래] vs h(pos_b)[미래]  평균|Δ| 0.427  (미래 렌더가 같은 픽셀인데도)
       h(imp_ab)[문맥] vs h(pos_a)[문맥]  평균|Δ| 0.345  (문맥 렌더가 바이트 동일인데도)
   그래서 **target run 에 한해서만** 불가능 변이가 새 정보를 준다.

2) **가능 변이로 학습하고 불가능 변이에서 평가하기.** fit_variants/eval_variants 는
   group_column 값으로 거르는데(§3-2), group_column 이 condition 이면 pos/imp 를
   못 가른다. 그래서 `pk_cond = <종류>|<condition>` 결합 컬럼을 만들어 거기에 건다.

vanish 는 뺀다 — imp_ab 는 shape_post 가 'none'(물체가 사라짐), imp_ba 는 shape_pre 가
'none' 이라 8-way 클래스에 안 들어간다. shape 게임 1024 / color 게임 1024 만 쓴다.

⚠️ **행 개수와 순서를 그대로 둔다(8192).** TokenCache.matches() 가 video_ids 로
   서명을 만들기 때문에, 한 줄이라도 빼면 160 GiB 를 다시 뽑는다. probing 대상은
   probe_kind + block_types 로 고르고, 나머지 행은 'skip' 으로 남겨 둔다.

  python z_research/scripts/build_probe_imp_index.py --root data_csv/intphysgen_v10
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path


def kind(r: dict) -> str:
    if r["probe_type"] == "obj":
        return "obj"
    if r["probe_type"] == "imp" and r["game_name"] in ("shape", "color"):
        return f"imp_{r['game_name']}"
    return "skip"                      # empty, imp/vanish — 라벨이 'none' 이라 못 쓴다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data_csv/intphysgen_v10"))
    ap.add_argument("--src", default="index_probe.csv")
    ap.add_argument("--out", default="index_probe_imp.csv")
    a = ap.parse_args()

    rows = list(csv.DictReader((a.root / a.src).open()))
    cols = list(rows[0]) + ["probe_kind", "pk_cond"]
    for r in rows:
        r["probe_kind"] = kind(r)
        r["pk_cond"] = f"{r['probe_kind']}|{r['condition']}"

    out = a.root / a.out
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    c = collections.Counter(r["probe_kind"] for r in rows)
    print(f"wrote {out}: {len(rows)} rows (원본과 같은 순서·개수 -> 토큰 캐시 재사용)")
    for k in ("obj", "imp_shape", "imp_color", "skip"):
        print(f"  {k:<10} {c[k]}")
    # 라벨이 성립하는지 검사 — 'none' 이 하나라도 있으면 eval.py 가 KeyError 로 죽는다
    bad = [(r["video_id"], col, r[col])
           for r in rows if r["probe_kind"] != "skip"
           for col in ("shape_pre", "shape_post", "color_pre", "color_post")
           if r[col] == "none"]
    print(f"  probing 대상 중 'none' 라벨: {len(bad)}건" + (f"  예: {bad[:2]}" if bad else "  (OK)"))
    ch = collections.Counter((r["probe_kind"], r["shape_pre"] == r["shape_post"],
                              r["color_pre"] == r["color_post"])
                             for r in rows if r["probe_kind"] != "skip")
    print("  (종류, shape 불변?, color 불변?) ->", dict(ch))


if __name__ == "__main__":
    main()
