#!/usr/bin/env python3
"""프로토콜 yaml + 데이터셋/모델 레지스트리(md) -> 실행 가능한 최종 config 하나.

  python z_research/scripts/harness/resolve.py <프로토콜> <데이터셋> [모델] -o out.yaml

프로토콜 yaml 은 **데이터셋과 모델에 무관한 것만** 담는다 (프레임 배치, 채점 규칙,
probe 정의, dtype 관례). 경로·컬럼 이름·체크포인트는 configs/protocols/{datasets,models}.md
에서 온다. 그래서 데이터를 바꿀 때 고치는 곳이 md 한 줄뿐이다.

병합 규칙 — **프로토콜이 이긴다**
    data  = { **datasets.md[<데이터셋>], **프로토콜.data  }
    model = { **models.md[<모델>],       **프로토콜.model }
  프로토콜이 소유한 키(n_frames, resolution, probing 용 index_csv/type_column 등)는
  레지스트리에 뭐가 있든 프로토콜 값이 쓰인다. 반대로 레지스트리에만 있는 키
  (root, frames_pattern, block_column ...)는 그대로 채워진다.

  n_frames: RAW  -> 데이터셋의 raw_frames 로 치환 (영상 전체를 쓰는 프로토콜용)

자동으로 정해지는 것 (환경변수로 덮을 수 있다)
    tag        = <데이터셋.cache_tag>_<모델>                             TAG=...
    output_dir = <데이터셋.results_root>/<프로토콜>__<데이터셋>_<모델>   OUTDIR=...
  results_root 는 datasets.md 에서 온다 (데이터셋별 아카이브 자리).
  없으면 z_exp/world_model_analysis/results 로 떨어진다.
  ⚠️ tag 는 토큰 캐시의 이름이다. 프로토콜이 달라도 (데이터셋, 모델)이 같으면 같은 캐시를
     쓰라고 일부러 프로토콜을 뺐다. 캐시 서명은 video_ids/토큰수/dtype 만 보므로
     **체크포인트·해상도·autocast·재렌더는 서명에 없다. 데이터를 다시 만들면 캐시도 치울 것.**

모델을 로드하기 전에 실물을 검사한다 (경로, 인덱스, resolution==img_size, 프레임 예산).
ViT-H 로딩은 프로세스당 ~2분이라 그 전에 죽는 게 훨씬 싸다.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import yaml

ROOT = "/data/hyuntak/project/2026/2027_cvpr/vjepa2"
META = ("raw_frames", "cache_tag", "results_root", "available", "note")  # data: 로 안 들어가는 키
KV = re.compile(r"^([a-z_][a-z0-9_]*)\s*:\s*(.+?)\s*$")


def parse_registry(path: str, required: str) -> dict:
    """`## 이름` 섹션마다 `key: value` 를 모은다. required 키가 없는 섹션은 설명으로 보고 버린다."""
    out, cur = {}, None
    for ln in open(path, encoding="utf-8"):
        if ln.startswith("## "):
            cur = ln[3:].strip()
            out[cur] = {}
            continue
        if cur is None or ln.lstrip().startswith(("#", "-", ">", "|", "*")):
            continue
        m = KV.match(ln)
        if m:
            try:
                out[cur][m.group(1)] = yaml.safe_load(m.group(2))
            except yaml.YAMLError:
                out[cur][m.group(1)] = m.group(2)
    return {k: v for k, v in out.items() if required in v}


def die(msg: str):
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("protocol")
    ap.add_argument("dataset")
    ap.add_argument("model", nargs="?", default="vith")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--datasets", default=f"{ROOT}/configs/protocols/datasets.md")
    ap.add_argument("--models", default=f"{ROOT}/configs/protocols/models.md")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                    help="병합된 config 를 점 경로로 덮어쓴다. 값은 YAML 로 읽는다.\n"
                         "  --set probing.optim=attn_50\n"
                         "  --set 'probing.fit_groups_sweep=[null]'\n"
                         "  --set probing.targets.shape=null      (null = 그 키를 지운다)\n"
                         "실험 케이스마다 yaml 을 새로 만들지 않기 위한 장치다.")
    a = ap.parse_args()

    # ── 프로토콜 ────────────────────────────────────────────────────────────────
    pf = a.protocol if a.protocol.endswith((".yaml", ".yml")) else \
        f"{ROOT}/configs/protocols/{a.protocol}.yaml"
    if not os.path.isfile(pf):
        avail = sorted(f[:-5] for f in os.listdir(f"{ROOT}/configs/protocols") if f.endswith(".yaml"))
        die(f"프로토콜 없음 -> {pf}\n  사용 가능: {', '.join(avail)}")
    def load_protocol(path, seen=()):
        """`extends: <이름>` 을 따라가 부모 위에 자식을 깊은 병합한다.

        프로토콜 하나가 한 줄(예: fit_groups_sweep)만 다를 때 130줄짜리 문서를
        통째로 복제하지 않으려고 넣었다. 자식이 이긴다. dict 는 재귀 병합,
        list/스칼라는 통째로 교체한다 (fit_groups_sweep 를 덮으려면 교체여야 한다).

        **자식에서 값을 `null` 로 두면 그 키를 지운다.** dict 는 재귀 병합이라
        부모의 항목이 그대로 남는데, 부분집합만 돌리고 싶을 때가 있다:
            probing:
              targets:
                shape: null      # 부모의 shape/env 를 빼고 color 만 남긴다
                env:   null
        """
        c = yaml.safe_load(open(path, encoding="utf-8"))
        base = c.pop("extends", None)
        if not base:
            return c
        if base in seen:
            die(f"extends 순환: {' -> '.join(seen + (base,))}")
        bf = f"{ROOT}/configs/protocols/{base}.yaml"
        if not os.path.isfile(bf):
            die(f"extends 대상 프로토콜 없음 -> {bf}")

        def merge(dst, src):
            for k, v in src.items():
                if v is None:                      # null = 그 키를 지운다
                    dst.pop(k, None)
                elif isinstance(v, dict) and isinstance(dst.get(k), dict):
                    dst[k] = merge(dict(dst[k]), v)
                else:
                    dst[k] = v
            return dst

        return merge(load_protocol(bf, seen + (base,)), c)

    cfg = load_protocol(pf)
    pname = os.path.splitext(os.path.basename(pf))[0]

    # ── 레지스트리 ──────────────────────────────────────────────────────────────
    DS = parse_registry(a.datasets, "root")
    MD = parse_registry(a.models, "checkpoint")
    if a.dataset not in DS:
        die(f"데이터셋 '{a.dataset}' 이 {a.datasets} 에 없다\n  사용 가능: {', '.join(sorted(DS))}")
    if a.model not in MD:
        die(f"모델 '{a.model}' 이 {a.models} 에 없다\n  사용 가능: {', '.join(sorted(MD))}")
    ds, md = dict(DS[a.dataset]), dict(MD[a.model])

    if ds.get("available") is False:
        die(f"데이터셋 '{a.dataset}' 은 available: false 다.\n  {ds.get('note', '')}")

    raw_frames = ds.pop("raw_frames", None)
    cache_tag = ds.pop("cache_tag", a.dataset)
    results_root = ds.pop("results_root", f"{ROOT}/z_exp/world_model_analysis/results")
    for k in META:
        ds.pop(k, None)
        md.pop(k, None)

    # ── 병합: 프로토콜이 이긴다 ─────────────────────────────────────────────────
    cfg["data"] = {**ds, **(cfg.get("data") or {})}
    cfg["model"] = {**md, **(cfg.get("model") or {})}

    if cfg["data"].get("n_frames") == "RAW":
        if raw_frames is None:
            die(f"프로토콜이 n_frames: RAW 를 요구하는데 '{a.dataset}' 에 raw_frames 가 없다")
        cfg["data"]["n_frames"] = int(raw_frames)

    cfg["tag"] = os.environ.get("TAG") or f"{cache_tag}_{a.model}"
    cfg["output_dir"] = os.environ.get("OUTDIR") or \
        f"{results_root}/{pname}__{a.dataset}_{a.model}"

    # ── --set 덮어쓰기 (점 경로) ────────────────────────────────────────────────
    for kv in a.sets:
        if "=" not in kv:
            die(f"--set 은 KEY=VALUE 여야 한다 -> {kv}")
        key, val = kv.split("=", 1)
        try:
            val = yaml.safe_load(val)
        except yaml.YAMLError:
            pass                                   # 파싱 안 되면 문자열 그대로
        node, *rest = key.split(".")
        cur, path = cfg, [node]
        for k in [node] + rest[:-1]:
            if not isinstance(cur.get(k), dict):
                cur[k] = {}
            cur = cur[k]
        leaf = rest[-1] if rest else node
        if val is None:
            cur.pop(leaf, None)
        else:
            cur[leaf] = val

    # ── fit_groups_sweep: auto ─────────────────────────────────────────────────
    # 데이터에 실제로 있는 group 을 읽어 [null, [g1], [g2], ...] 를 만든다.
    # 이게 없으면 조건 구성이 다른 데이터셋마다 프로토콜 yaml 을 새로 떠야 한다
    # (실제로 attn_probe_flat.yaml 이 그 이유로 존재했다).
    P = cfg.get("probing") or {}
    if P.get("enabled") and P.get("fit_groups_sweep") == "auto":
        dd = cfg["data"]
        ipath = os.path.join(dd["root"], dd.get("index_csv", "index.csv"))
        gcol = dd.get("group_column") or dd.get("variant_column") or "variant"
        tcol, want = dd.get("type_column"), set(P.get("block_types") or [])
        seen = []
        with open(ipath, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if want and tcol and r.get(tcol) not in want:
                    continue                       # probing 대상이 없는 group 은 학습셋 0 -> 죽는다
                if r[gcol] not in seen:
                    seen.append(r[gcol])
        P["fit_groups_sweep"] = [None] + [[g] for g in sorted(seen)]

    # ── 모델 로드 전에 실물 검사 ────────────────────────────────────────────────
    d, m = cfg["data"], cfg["model"]
    for label, p in (("model.checkpoint", m.get("checkpoint")),
                     ("data.root", d.get("root")),
                     ("data.frames_root", d.get("frames_root"))):
        if p and not os.path.exists(p):
            die(f"{label} 이 없다 -> {p}")
    idx = os.path.join(d["root"], d.get("index_csv", "index.csv"))
    if not os.path.isfile(idx):
        die(f"index 가 없다 -> {idx}\n  build_*_index.py 를 먼저 돌려야 한다")
    if d.get("resolution") != m.get("img_size"):
        die(f"data.resolution({d.get('resolution')}) != model.img_size({m.get('img_size')}). "
            "작으면 CUDA assert, 크면 에러 없이 엉뚱한 구간을 잘라 쓴다")
    if raw_frames:
        need = d.get("frames_start", 1) + (d["n_frames"] - 1) * d.get("frames_stride", 1)
        if need > raw_frames:
            die(f"프레임 예산 초과: 마지막 프레임 인덱스 {need} > raw_frames {raw_frames} "
                f"(n_frames={d['n_frames']}, stride={d.get('frames_stride', 1)})")
    if cfg.get("probing", {}).get("enabled") and "features" not in cfg:
        die("probing.enabled 인데 features: 블록이 없다. surprise 로 폴백해 cache_dir KeyError 로 죽는다")

    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)
    yaml.safe_dump(cfg, open(a.output, "w", encoding="utf-8"),
                   allow_unicode=True, sort_keys=False)

    if not a.quiet:
        n, st, sr = d["n_frames"], d.get("frames_start", 1), d.get("frames_stride", 1)
        print(f"  protocol : {pname}   ({pf})")
        print(f"  dataset  : {a.dataset}   {d['root']}")
        print(f"  index    : {d.get('index_csv','index.csv')}")
        print(f"  model    : {a.model}   {m['arch_name']}   dtype={m.get('dtype')}"
              f" autocast={m.get('autocast', '-')}")
        print(f"  frames   : n={n} start={st} stride={sr} -> raw {st}..{st + (n-1)*sr}"
              f" (raw_frames={raw_frames})  res={d.get('resolution')}")
        if "surprise" in cfg:
            s = cfg["surprise"]
            print(f"  surprise : mode={s.get('mode','single')} ctx={s.get('context_length','-')}"
                  f" pairing={cfg.get('scoring',{}).get('pairing','cross')}")
        if cfg.get("probing", {}).get("enabled"):
            print(f"  probing  : {list(cfg['probing']['targets'])}  "
                  f"block_types={cfg['probing'].get('block_types')}  "
                  f"optim={cfg['probing'].get('optim')}")
        print(f"  tag      : {cfg['tag']}   (토큰 캐시 이름)")
        print(f"  output   : {cfg['output_dir']}")


if __name__ == "__main__":
    main()
