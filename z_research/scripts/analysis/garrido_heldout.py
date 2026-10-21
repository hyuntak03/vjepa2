#!/usr/bin/env python3
"""Garrido A.8 최고 칸 (garrido_rescore.select) 을 **지정한 영상만으로** 다시 고른다 (2026-10-21).

dev 로 학습한 모델 (예: vith_pft_ip1late_f0 — IntPhys 1 dev block 2-fold) 은 IntPhys 1 sliding 을 학습에 안 쓴 block 으로만 세야 한다.
창 (w16 · w32) 결과 폴더의 per_window.json 은 그대로 두고, 쌍을 셀 때 --keep 인덱스의 video_id 만 남긴다 (나머지 축약 · 선택 규칙은 garrido_rescore 그대로).

  python z_research/scripts/analysis/garrido_heldout.py --dataset intphys1_dev --model vith_pft_ip1late_f0 \
      --keep data_csv/intphys1_late_uni_cv/fold0_test.csv [--model-ref vith]
  --model-ref: 같은 영상만으로 센 다른 모델 값 (비교 기준) 을 같이 찍는다."""
import argparse, csv, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import garrido_rescore as G                                                   # noqa: E402

REPO = HERE.parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="intphys1_dev")
ap.add_argument("--model", required=True)
ap.add_argument("--model-ref", nargs="*", default=[])
ap.add_argument("--keep", required=True, help="video_id 열이 있는 csv — 이 영상만 센다")
ap.add_argument("--results", default=str(REPO / "cvpr/results/eval/intphys1"))
a = ap.parse_args()
keep = {r["video_id"] for r in csv.DictReader(open(REPO / a.keep if not a.keep.startswith("/") else a.keep))}
_load = G.load_index


def load_kept(res):
    cfg, meta = _load(res)
    return cfg, {v: r for v, r in meta.items() if v in keep}


G.load_index = load_kept
for m in [a.model] + a.model_ref:
    cb, kind = {}, None
    for w in ("w16", "w32"):
        d = pathlib.Path(a.results) / f"{a.dataset}_{m}_wgarrido_{w}"
        if not (d / "per_window.json").exists():
            print(f"{m}: {d.name} 없음 — 건너뜀"); continue
        r = G.rescore(d); cb.update(r["combos"]); kind = r["dataset_kind"]
    if not cb:
        continue
    b = G.select(cb, kind)
    pg = " · ".join(f"{g} {v['acc']:.1f} (n{v['n']})" for g, v in b["per_group"].items()) if "per_group" in b else ""
    print(f"{m:28s} {b['accuracy']:.2f}  combo {b['combo']}  n_pair {b.get('n_pair')}  | {pg}")
