#!/usr/bin/env python3
"""RollOutV3 의 **경로와 자(decoder) 지문을 한 곳에서** 정한다. 스크립트마다 경로를 하드코딩하지 않는다.

환경변수 두 개로 고른다 (안 주면 옛 동작 그대로):
  R3_DECODER  자 폴더 (exp_results/<이름>).                                  기본 presence
              `identity` = 위치 + 정체 (56 조합 + 없음) 자 (2026-09-25). `load_head` · `present_thr` 가 두 종류를 같은 형식으로 맞춘다
              지금 RollOutV3 문서 · 그림의 자는 `identity_r8` (2026-09-26, 새 학습셋 training_r8). 기본값 presence 는 다른 세션 호환용
  R3_OUT      결과 꼬리표. 주면 exp_results/windows_<꼬리표>/ 와
              figures/<꼬리표>/{windows,examples,readout,windows_gif}/ 로 쓴다.   기본 없음 (windows/, figures/<하위>/)

  예) R3_OUT=training_v8 python z_research/scripts/analysis/rollout3_window_readout.py --reps p z h --gpus 8

⚠️ **readings 에는 자가 구워져 있다.** `readings.npz` 는 특징이 아니라 자가 읽은 좌표·presence 다.
   자를 다시 학습하면 문턱·치우침 (`summary.json`, `attn_bias_px.json`) 은 새 자 것이 되는데 readings 는
   옛 자 것이라, 섞으면 **옛 읽은 값에 새 문턱을 거는 조용히 틀린 그림**이 나온다 (2026-09-24 에 막았다).
   → 쓰는 쪽은 `fingerprint()` 를 싣고, 읽는 쪽은 `check()` 를 부른다.

  $P z_research/scripts/analysis/rollout3_paths.py     # 지금 경로·자 지문·readings 지문
"""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
EXP = Path(os.environ.get("R3_EXP", str(ROOT / "z_research/RollOutV3/exp_results")))   # 2026-09-28: 다른 predictor 판 (ariel/) 은 R3_EXP 로 딴 뿌리
# 옛 파이프라인 (rollout3_rerun.sh 의 profile · windows · examples · readout 그림, presence 자 시기) 이 쓰는 뿌리.
# 2026-09-26 폴더 정리 때 그 그림들을 figures/_superseded/presence_era_2026-09-24/ 로 내렸으므로 다시 돌리면 그 옆에 쓴다
FIGROOT = ROOT / "z_research/RollOutV3/figures/_superseded/presence_era_2026-09-24"
DECODER = os.environ.get("R3_DECODER", "presence")
TAG = os.environ.get("R3_OUT", "")
PRES = EXP / DECODER                                           # 자 (readout_attn.pt · summary.json · attn_bias_px.json)
WIN = EXP / (f"windows_{TAG}" if TAG else "windows")           # v3 16 창에서 읽은 값
V11 = EXP / (f"v11_{TAG}" if TAG else "v11")                   # v11 visible + 문맥 끝 가림에서 읽은 값
V11_MID = EXP / (f"v11_mid_{TAG}" if TAG else "v11_mid")       # v11 문맥 중간 가림
# 지금 그림 폴더의 뿌리 = RollOutV3/figures/ (v3_signed_error{,_hhead} · v3_trajectory · v3_gif · v11_presence · v11_signed_error ·
# v11_readout_overlay · v11_probe · v11_geometry). 2026-09-26 전에는 RollOutV3/new_archive/ 였다 (호환 링크 없음).
# 시험 그림은 R3_FIGROOT 로 딴 데 그린다
ARCHIVE = Path(os.environ.get("R3_FIGROOT", str(ROOT / "z_research/RollOutV3/figures")))


def fig(sub: str) -> Path:
    """그림 폴더. 꼬리표가 있으면 figures/<꼬리표>/<sub>."""
    return FIGROOT / TAG / sub if TAG else FIGROOT / sub


def head_file(rep: str, pres: Path = PRES) -> Path:
    """자 가중치 파일. presence 자는 `readout_attn.pt`, 정체 자 (identity, 2026-09-25) 는 `readout.pt`."""
    for name in ("readout_attn.pt", "readout.pt"):
        f = Path(pres) / rep / name
        if f.exists():
            return f
    return Path(pres) / rep / "readout_attn.pt"


def is_identity(pres: Path = PRES) -> bool:
    """정체 자 (위치 + 모양×색 56 조합 + 없음, `rollout2_identity_readout.py`) 인가."""
    f = Path(pres) / "summary.json"
    return f.exists() and "none_index" in json.loads(f.read_text())


def present_thr(rep: str, pres: Path = PRES) -> float:
    """readings 의 3 번째 값 (`[..., 2]`) 에 거는 '있다' 문턱.
      presence 자  presence logit, 문턱 = val 음성 5 % 오탐 (`thr_val_fpr5`)
      정체 자      score = log max_c P(조합 c) − log P(없음), 문턱 = **0** (= 57-way argmax 가 '없음' 이 아니다).
                   튜닝한 문턱이 없다. 학습셋 test 오탐 · recall: identity_r8 p 0.27 · z 0.05 · h 0.03 % / 99.3 · 99.9 · 99.8 %
                   (옛 identity: p 1.3 · z 0.2 · h 0.3 % / 99.0 · 99.8 · 99.8 %)
    """
    S = json.loads((Path(pres) / "summary.json").read_text())
    return 0.0 if "none_index" in S else float(S["reps"][rep]["attn"]["thr_val_fpr5"])


def load_head(rep: str, dev, pres: Path = PRES):
    """자를 불러 f(tok (B,256,D)) → (xy (B,2), score (B,), top1 (B,), prob (B,57) 또는 None) 을 돌려준다.
    score 는 `present_thr` 와 짝이다. 두 종류의 자를 같은 readings 형식 (x, y, score, top1) 으로 쓰게 한다."""
    import importlib.util, sys, torch
    ident = is_identity(pres)
    mod = "rollout2_identity_readout" if ident else "rollout2_presence_readout"
    if mod not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod, ROOT / f"z_research/scripts/analysis/{mod}.py")
        m = importlib.util.module_from_spec(spec); sys.modules[mod] = m; spec.loader.exec_module(m)
    m = sys.modules[mod]
    h = (m.IdReadout() if ident else m.Readout("attn")).to(dev).eval()
    h.load_state_dict(torch.load(head_file(rep, pres), map_location="cpu"))

    def f(tok):
        if ident:
            xy, lg, a = h(tok)
            lp = torch.log_softmax(lg.float(), -1)
            score = lp[:, :m.NONE].max(-1).values - lp[:, m.NONE]
            return xy, score, a.max(-1).values, lp.exp()
        xy, pres_, a = h(tok)
        return xy, pres_, a.max(-1).values, None
    return f


def fingerprint(pres: Path = PRES, reps=("p", "z", "h")) -> str:
    """p·z·h 자 가중치 파일의 sha1 앞 12 자리 (없는 파일은 'missing' 으로 섞는다)."""
    h = hashlib.sha1()
    for r in reps:
        f = head_file(r, pres)
        h.update(r.encode()); h.update(f.read_bytes() if f.exists() else b"missing")
    return h.hexdigest()[:12]


def stamp(folder: Path, pres: Path = PRES, readings=()):
    """그림 폴더에 **어느 자로 그렸나** 를 남긴다 (`_decoder.json`). new_archive_redraw 가 이걸 보고 옛 그림을 _superseded/ 로 내린다."""
    import datetime
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / "_decoder.json").write_text(json.dumps(dict(
        decoder=Path(pres).name, decoder_dir=str(pres), fingerprint=fingerprint(pres), identity=is_identity(pres),
        readings=[str(r) for r in readings], drawn=datetime.datetime.now().isoformat(timespec="seconds")), indent=1))


def check(readings, pres: Path = PRES):
    """`readings` (np.load 결과) 가 **지금 자**로 읽힌 것인지. 아니면 죽는다."""
    got = str(readings["decoder_fp"]) if "decoder_fp" in readings.files else None
    want = fingerprint(pres)
    if got != want:
        raise SystemExit(
            f"\n⚠️ {WIN}/readings.npz 는 **다른 자**로 읽혔다 (readings {got} vs 자 {pres.name} {want}).\n"
            f"   새 문턱·치우침을 옛 읽은 값에 걸면 조용히 틀린다. 16 창을 이 자로 다시 읽을 것 (RERUN.md §3),\n"
            f"   또는 R3_DECODER 로 그 readings 를 읽은 자를 고를 것.\n")


if __name__ == "__main__":
    import numpy as np
    print(f"자       {PRES}   지문 {fingerprint()}")
    print(f"readings {WIN}")
    f = WIN / "readings.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        print(f"         지문 {str(z['decoder_fp']) if 'decoder_fp' in z.files else '(없음)'}")
    print(f"그림     {fig('<하위>')}")
