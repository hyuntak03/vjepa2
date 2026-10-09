#!/usr/bin/env python3
"""옛 레지스트리 (configs/protocols/{datasets,models}.md) -> cvpr/registry/{datasets,models}.yaml 변환.

    python cvpr/harness/import_registry.py            # 두 yaml 을 다시 쓴다 (손으로 고친 항목은 _manual 블록으로 보존)
    python cvpr/harness/import_registry.py --check    # 쓰지 않고 차이만 보고

바꾸는 것
  * 절대경로 -> ${CVPR_*} (cvpr/env.sh 의 변수). 레포를 옮기거나 노드가 바뀌면 env.sh 한 줄만 바꾼다.
  * models: `vith` 와 encoder 키가 전부 같은 섹션은 `base: vith` + 다른 키만 남긴다 (ViT-H 경로 17 회 복사 제거).
  * 실물이 없는 체크포인트/프레임 폴더는 `available: false` + `note` 로 표시한다 (run --list 에서 [사용 불가]).
  * 설명문에서 샌 키 (`index:` 등 — md 파서가 `key:` 줄을 전부 긁는다) 는 버리고 이름을 출력한다.

옛 md 는 그대로 둔다 (z_research/scripts/run.sh 가 아직 읽는다). 여기 yaml 이 cvpr/ 의 정본이다.
수동 항목 (intphys2_main, ek100 등 md 에 없던 것) 은 이 파일 하단 MANUAL_* 에 있다.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # 레포 루트
sys.path.insert(0, str(ROOT / "z_research/scripts/harness"))
from resolve import parse_registry           # noqa: E402  (옛 파서 그대로 — 한 벌만 둔다)

# 절대경로 -> env 변수. 긴 것부터 (접두어 겹침 방지).
ROOTS = [
    (f"{ROOT}/checkpoint", "${CVPR_CKPT}"),
    (f"{ROOT}/data_csv", "${CVPR_DATA_CSV}"),
    (str(ROOT), "${CVPR_REPO}"),
    ("/local_datasets/world/world_analysis/cache", "${CVPR_CACHE}"),
    ("/local_datasets", "${CVPR_LOCAL}"),
    ("/data2", "${CVPR_DATA2}"),
    ("/data/hyuntak/project/2026/2027_cvpr", "${CVPR_NFS}"),
    ("/data/dataset/world/Benchmarks", "${CVPR_BENCH_SRC}"),
    ("/data/jongseo/project", "${CVPR_OTHER_CKPT}"),
]
ENV_DEFAULT = {   # 실물 검사용 (env.sh 와 같은 기본값)
    "CVPR_REPO": str(ROOT), "CVPR_CKPT": f"{ROOT}/checkpoint", "CVPR_DATA_CSV": f"{ROOT}/data_csv",
    "CVPR_CACHE": "/local_datasets/world/world_analysis/cache", "CVPR_LOCAL": "/local_datasets",
    "CVPR_DATA2": "/data2", "CVPR_NFS": "/data/hyuntak/project/2026/2027_cvpr",
    "CVPR_BENCH_SRC": "/data/dataset/world/Benchmarks", "CVPR_OTHER_CKPT": "/data/jongseo/project",
}
DATA_KEYS = {"root", "index_csv", "frames_root", "frames_pattern", "frames_start", "frames_stride",
             "block_column", "pair_column", "variant_column", "group_column", "plausible_column",
             "type_column", "frames_start_column", "n_frames", "resolution"}
META_KEYS = {"raw_frames", "cache_tag", "results_root", "available", "note", "sliding_frame_skips", "overrides"}
STRAY = {"index"}                            # 설명문에서 새는 키


def sub_roots(v):
    if isinstance(v, str):
        for src, dst in ROOTS:
            if v.startswith(src):
                return dst + v[len(src):]
    return v


def expand(v: str) -> str:
    return re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", lambda m: ENV_DEFAULT.get(m.group(1), m.group(0)), v)


def convert_datasets(md: Path):
    out, dropped = {}, []
    for name, sec in parse_registry(str(md), "root").items():
        e = {}
        for k, v in sec.items():
            if k in STRAY:
                dropped.append((name, k)); continue
            if k not in DATA_KEYS | META_KEYS:
                dropped.append((name, k)); continue
            e[k] = sub_roots(v)
        e.setdefault("engine", "wma")
        e["legacy_results_root"] = e.pop("results_root", "${CVPR_REPO}/z_exp/world_model_analysis/results")
        # 실물 검사 (root · frames_root). 없으면 available:false + note
        missing = [k for k in ("root", "frames_root") if k in e and not os.path.exists(expand(e[k]))]
        if missing and e.get("available", True) is not False:
            e["available"] = False
            e["note"] = f"실물 없음 ({', '.join(missing)}: {', '.join(expand(e[k]) for k in missing)}) — 2026-10-09 변환 때 vll5 기준"
        out[name] = e
    return out, dropped


def convert_models(md: Path):
    raw = parse_registry(str(md), "checkpoint")
    base = {k: sub_roots(v) for k, v in raw["vith"].items()}
    out, dropped = {}, []
    for name, sec in raw.items():
        e = {k: sub_roots(v) for k, v in sec.items() if k not in STRAY}
        for k in [k for k in sec if k in STRAY]:
            dropped.append((name, k))
        # vith 와 encoder 가 같은 섹션 -> base: vith
        if name != "vith" and all(e.get(k) == v for k, v in base.items()):
            e = {"base": "vith", **{k: v for k, v in e.items() if k not in base}}
        missing = [k for k in ("checkpoint", "predictor_checkpoint") if k in e and not os.path.exists(expand(e[k]))]
        if missing:
            e["available"] = False
            e["note"] = f"실물 없음 ({', '.join(missing)}) — 2026-10-09 변환 때 vll5 기준. 옛 결과 _resolved.yaml 을 읽기 위한 기록"
        out[name] = e
    return out, dropped


# ── md 에 없던 항목 (분석/학습 하네스가 통짜 yaml 로 들고 있던 것) ──────────────
# Garrido A.8 격자의 벤치별 frame_skips (옛 z_research/Benchmarks/run_all.sh set_bench 의 case 문).
#   GRASP 은 skip 2 를 뺀 부분 탐색 (2026-09-22 결정: 비용 66~71 %, Table S3 도 GRASP 에 skip 10). 보고할 때 "A.8 부분 탐색" 이라고 밝힌다.
MANUAL_PATCH = {
    "intphys1_dev": {"sliding_frame_skips": [2, 5, 10]},
    "grasp_level2": {"sliding_frame_skips": [5, 10]},
    "inflevel_continuity": {"sliding_frame_skips": [5, 10, 20]},
    "inflevel_gravity": {"sliding_frame_skips": [5, 10, 20]},
    "inflevel_solidity": {"sliding_frame_skips": [5, 10, 20]},
    # 데이터셋이 요구하는 채점 옵션. 옛 datasets.md:846 은 `SET="scoring.pairing=cross"` 를 메모로만 들고 있었다.
    "gravity_realistic": {"overrides": {"scoring.pairing": "cross"}},
}

# 모델별 실행 관례 (옛 chain 스크립트가 SET 으로 주던 것). surprise.<키> 는 surprise 블록에 덮인다 (수치 무관한 자원 상한).
MANUAL_MODEL_PATCH = {
    "dinof_highres": {"surprise.batch_size": 8, "surprise.decode_workers": 8,
                      "note": "문맥 4 프레임 고정 — testbed 는 문맥 16 중 마지막 4 장으로 미래를 자기회귀 unroll. IntPhys1 sliding 은 WINDOW=dinof_w5|w8|w16 (run.sh 가 돈다). 복사 기준선 COPY=1 (model.dinof_copy=true)"},
    "dinof_lowres":  {"surprise.batch_size": 8, "surprise.decode_workers": 8, "note": "dinof_highres 와 같고 224 판"},
}

MANUAL_DATASETS = {
    "intphys2_main": {
        "engine": "intphys2",
        "root": "${CVPR_DATA2}/local_datasets/world/Benchmark/IntPhys2",   # stage.sh 가 푼 노드 로컬 사본 (analysis/intphys2/configs/bench_*.yaml)
        "dataset_variant": "intphys2", "split": "Main",
        "legacy_results_root": "${CVPR_REPO}/z_research/Benchmarks/exp_results/intphys2",
        "note": "IntPhys 2 Main 506 쌍. 다른 사본 ${CVPR_LOCAL}/world/IntPhys2 (TEMPLATE/repro 가 쓰던 것) 은 split 폴더 구성이 같다",
    },
    "intphys2_main_nfs": {
        "engine": "intphys2",
        "root": "${CVPR_LOCAL}/world/IntPhys2", "dataset_variant": "intphys2", "split": "Main",
        "legacy_results_root": "${CVPR_REPO}/z_exp/intphys2",
        "note": "vjepa2_vith_intphys2_main.yaml 이 쓰던 사본",
    },
    "ek100": {
        "engine": "ek100",
        "base_path": "${CVPR_DATA2}/local_datasets/EPIC-KITCHENS_resized",
        "dataset_train": "${CVPR_NFS}/epic-kitchens-100-annotations/EPIC_100_train.csv",
        "dataset_val": "${CVPR_NFS}/epic-kitchens-100-annotations/EPIC_100_validation.csv",
        "file_format": 0,
        "legacy_results_root": "${CVPR_REPO}/z_research/anticipation/EK100/exp_results",
        "note": "공식 val 기하로 재인코딩한 256x256 영상 (train 495 + val 138). z_research/anticipation/EK100/README.md §3-1",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ds, d1 = convert_datasets(ROOT / "configs/protocols/datasets.md")
    md, d2 = convert_models(ROOT / "configs/protocols/models.md")
    for k, v in MANUAL_DATASETS.items():
        ds[k] = v
    for k, v in MANUAL_PATCH.items():
        if k in ds:
            ds[k].update(v)
    for k, v in MANUAL_MODEL_PATCH.items():
        if k in md:
            md[k].update(v)
    for name, e in ds.items():
        if e["engine"] != "wma":
            continue
        for k in ("root", "frames_root"):
            pass
    hdr_ds = ("# 데이터셋 레지스트리 (cvpr). configs/protocols/datasets.md 에서 import_registry.py 가 만들었다 (2026-10-09).\n"
              "# 손으로 고쳐도 된다 — 다만 다시 변환하면 덮이므로 바꾼 항목은 아래 _manual 에 옮겨 둘 것.\n"
              "# 키: engine (wma|intphys2|ek100) · raw_frames · cache_tag · legacy_results_root · available · note · 그리고 data: 블록으로 들어가는 키들.\n"
              "# ${CVPR_*} 는 cvpr/env.sh 의 변수다. 설명은 옛 md 의 각 섹션에 그대로 있다.\n")
    hdr_md = ("# 모델 레지스트리 (cvpr). configs/protocols/models.md 에서 import_registry.py 가 만들었다 (2026-10-09).\n"
              "# base: vith -> vith 의 키를 깔고 그 위에 덮는다. `predictor.kind` 같은 점 키는 resolve 가 predictor 블록에 넣는다.\n"
              "# `surprise.<키>` 는 모델별 채점 관례 (예: target_layer_norm: false). available: false 는 실물 없음.\n")
    ds_doc = hdr_ds + yaml.safe_dump(ds, allow_unicode=True, sort_keys=False, width=200)
    md_doc = hdr_md + yaml.safe_dump(md, allow_unicode=True, sort_keys=False, width=200)
    print(f"datasets {len(ds)} (md {len(ds) - len(MANUAL_DATASETS)} + manual {len(MANUAL_DATASETS)}) · models {len(md)}")
    print("dropped keys:", d1 + d2)
    print("unavailable datasets:", [k for k, v in ds.items() if v.get("available") is False])
    print("unavailable models:", [k for k, v in md.items() if v.get("available") is False])
    if a.check:
        return
    (HERE.parent / "registry/datasets.yaml").write_text(ds_doc, encoding="utf-8")
    (HERE.parent / "registry/models.yaml").write_text(md_doc, encoding="utf-8")
    print("wrote cvpr/registry/{datasets,models}.yaml")


if __name__ == "__main__":
    main()
