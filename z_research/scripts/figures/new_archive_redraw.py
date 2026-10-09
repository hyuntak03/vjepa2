#!/usr/bin/env python3
"""**자 (decoder) 하나로 new_archive 그림 네 폴더를 다시 그린다** — 새 자를 학습하면 이것 한 줄 (2026-09-25).

  폴더                    그림                                              그리는 스크립트
  train_readout           자 자신을 학습셋 held-out test 에서 (존재·정체·위치)  plot_train_readout.py (정체 자만. readings 대신 자의 preds.npz)
  context_to_future       문맥 운동 · 외형이 p 의 미래를 바꾸나 + 정체 유지       plot_context_to_future.py (정체 자만)
  v3_signed_error         v3 16 창 × 8 법칙, 축마다 앞섬 / 뒤처짐 (p 자기 자)   plot_v3_l2_error.py --signed
  v3_signed_error_hhead   같은 그림을 h 자로 p 를 읽어서 (p@h)                 plot_v3_l2_error.py --signed --head h
  v3_trajectory           위치 · 속도 − 시간                                   plot_v3_trajectory.py
  v11_presence            v11 가림 타이밍 × k, p 의 미래에 물체가 있나         plot_v11_presence.py (+ --by-k)
  v11_signed_error        v11 조건 9 개 × 운동 3, 앞섬/뒤처짐 (h 기준선)       plot_v11_signed_error.py

단계 (`--steps`, 기본 전부 차례로):
  check    자 폴더 (`exp_results/<자>`) 에 summary.json · p/z/h 가중치가 있나, readings 가 이 자로 읽혔나 (지문) 를 표로 보인다
  bias     `attn_bias_px.json` 이 없으면 (또는 --force) `decoder_bias.py`
  extract  **이 자로 안 읽힌 readings 만** 다시 읽는다 — v3 16 창 (p z h + p@h) · v11 visible+late · v11 mid.
           v3 는 `--gpus-v3`, v11 은 `--gpus-v11` 에서 **동시에** (v11 은 late → mid). 한쪽만 필요하면 GPU 를 그쪽에 몰아 준다.
           v3 프레임은 /data2 HDD RAID 라 먼저 페이지 캐시에 올린다 — 안 하면 작은 PNG 랜덤 읽기가 17 MB/s 라 GPU 가 논다
           (2026-09-25 실측: 20 GB 예열 32 초, 추출 v3 40 분 · v11 late 20 분 · mid 12 분, GPU 6 + 2 장)
  figs     각 폴더의 지금 그림을 도장 (`_decoder.json`) 을 보고 `_superseded/<그린 자>_<지문>/` 로 내린 뒤 새로 그린다.
           같은 자 (지문 같음) 로 그린 그림이면 내리지 않고 덮는다

readings 는 자마다 따로 쓴다 (`exp_results/{windows,v11,v11_mid}_<꼬리표>`, 꼬리표 기본 = 자 이름; 옛 자 `presence` 는 꼬리표 없음).
그림 스크립트는 지문이 다른 readings 를 받으면 죽는다 (`rollout3_paths.check`) — 옛 값에 새 문턱을 거는 사고를 막는다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/new_archive_redraw.py --decoder identity --dry            # 계획만 (GPU 0 장)
  $P z_research/scripts/figures/new_archive_redraw.py --decoder identity_v2               # 새 자: 치우침 → 읽기 (~40 분, GPU 8) → 그림
  $P z_research/scripts/figures/new_archive_redraw.py --decoder identity --steps figs     # 그림만 (CPU 몇 분)
  $P z_research/scripts/figures/new_archive_redraw.py --decoder presence --steps figs --figroot /tmp/x   # 옛 자로 딴 데에 (시험)
새 자 학습은 `z_research/scripts/analysis/rollout2_identity_readout.py --launch --name <자>` (끝나면 치우침까지 쓴다).
"""
from __future__ import annotations
import argparse, datetime, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
AN, FG = ROOT / "z_research/scripts/analysis", ROOT / "z_research/scripts/figures"
EXP = ROOT / "z_research/RollOutV3/exp_results"
V3_FRAMES = Path("/data2/local_datasets/world/world_analysis/RollOut_v3/Images")
sys.path.insert(0, str(AN))
from rollout3_paths import fingerprint, head_file, is_identity   # noqa: E402  (경로를 인자로 받는 함수만 쓴다)

FOLDERS = {
    "train_readout": [["plot_train_readout.py"]],            # 2026-09-26: 자 자신을 학습셋 test 에서 (맨 먼저 보는 그림, 정체 자만)
    "v3_signed_error": [["plot_v3_l2_error.py", "--signed"]],
    "v3_signed_error_hhead": [["plot_v3_l2_error.py", "--signed", "--head", "h"]],
    "v3_trajectory": [["plot_v3_trajectory.py"]],
    "v11_presence": [["plot_v11_presence.py"], ["plot_v11_presence.py", "--by-k"], ["plot_v11_presence.py", "--identity"], ["plot_v11_presence.py", "--acc57"]],   # --identity · --acc57: 정체 자만
    "v11_signed_error": [["plot_v11_signed_error.py"]],     # 2026-09-26: v11 의 앞섬/뒤처짐 (가림막 오탐 없는 자에서만 뜻이 있다)
    "v11_readout_overlay": [["plot_v11_readout_overlay.py"]],  # 2026-09-26: '있음' 일 때 읽힌 위치를 빈 장면 프레임 위에
    "context_to_future": [["plot_context_to_future.py"]],      # 2026-09-26: 문맥 운동 · 외형이 p 의 미래를 바꾸나 + 정체 (정체 자만)
}
KEEP_SUFFIX = (".png", ".pdf", ".json", ".gif")   # .pdf: 논문 판 그림 (2026-09-27)


def readings(tag):
    sfx = f"_{tag}" if tag else ""
    return {"v3": EXP / f"windows{sfx}" / "readings.npz", "v11": EXP / f"v11{sfx}" / "readings.npz",
            "mid": EXP / f"v11_mid{sfx}" / "readings.npz"}


def fp_of(f: Path):
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    return str(z["decoder_fp"]) if "decoder_fp" in z.files else "(지문 없음)"


def env_for(a):
    return {**os.environ, "R3_DECODER": a.decoder, "R3_OUT": a.tag, "R3_FIGROOT": str(a.figroot)}


def log(*x):
    print(f"[redraw {datetime.datetime.now():%H:%M:%S}]", *x, flush=True)


def check(a):
    pres = EXP / a.decoder
    miss = [str(p) for p in [pres / "summary.json"] + [head_file(r, pres) for r in "pzh"] if not p.exists()]
    if miss:
        sys.exit(f"자 폴더가 덜 됐다: {miss}\n학습: rollout2_identity_readout.py --launch --name {a.decoder}")
    fp = fingerprint(pres)
    log(f"자 {pres}  종류 {'정체 (57 분류)' if is_identity(pres) else 'presence'}  지문 {fp}  "
        f"치우침 {'있음' if (pres / 'attn_bias_px.json').exists() else '없음 → bias 단계가 만든다'}")
    for k, f in readings(a.tag).items():
        got = fp_of(f)
        log(f"  readings {k:4s} {f.parent.name:22s} " + ("없음 → 읽는다" if got is None else
            ("이 자로 읽힘 ✓" if got == fp else f"다른 자 ({got}) → 다시 읽는다")))
    for d in FOLDERS:
        st = a.figroot / d / "_decoder.json"
        s = json.loads(st.read_text()) if st.exists() else None
        log(f"  그림 {d:22s} " + (f"지금 {s['decoder']} ({s['fingerprint']}, {s['drawn']})" if s else "도장 없음"))
    return fp


def bias(a):
    if (EXP / a.decoder / "attn_bias_px.json").exists() and not a.force:
        log("치우침: 이미 있다 (건너뜀)"); return
    run([sys.executable, str(AN / "decoder_bias.py")], env_for(a), None)


def run(cmd, env, logf):
    out = open(logf, "w") if logf else None
    r = subprocess.run(cmd, env=env, stdout=out or subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode:
        tail = (Path(logf).read_text() if logf else r.stdout)[-2000:]
        sys.exit(f"실패 (exit {r.returncode}): {' '.join(map(str, cmd))}\n{tail}")
    return r


def extract(a, fp):
    R = readings(a.tag)
    need = {k: a.force or fp_of(f) != fp for k, f in R.items()}
    if not any(need.values()):
        log("읽기: 세 readings 모두 이 자로 읽혀 있다 (건너뜀)"); return
    g3, g11 = a.gpus_v3, a.gpus_v11
    if not need["v3"]:
        g11 = f"{g3},{g11}"
    if not (need["v11"] or need["mid"]):
        g3 = f"{g3},{g11}"
    ng = lambda g: len(g.split(","))
    logs = EXP / f"redraw_{a.tag or 'presence'}"; logs.mkdir(parents=True, exist_ok=True)
    env = env_for(a); err = []
    if need["v3"]:
        t0 = time.time()
        subprocess.run(["bash", "-c", f'find "{V3_FRAMES}" -name "*.png" -print0 | xargs -0 -P 64 -n 200 cat > /dev/null'])
        log(f"v3 프레임 페이지 캐시 예열 {time.time()-t0:.0f}s")

    def v3():
        cmd = [sys.executable, "-u", str(AN / "rollout3_window_readout.py"), "--reps", *a.v3_reps, "--cross", "h",
               "--gpus", str(ng(g3)), "--token-budget", "65536", "--max-bs", "32"]
        e = {**env, "CUDA_VISIBLE_DEVICES": g3, "R3_TMP": f"/dev/shm/r3w_{a.tag or 'presence'}"}
        log(f"v3 16 창 읽기 → GPU {g3}  (로그 {logs / 'v3.log'})")
        try:
            run(cmd, e, logs / "v3.log")
        except SystemExit as x:
            err.append(str(x))
        shutil.rmtree(e["R3_TMP"], ignore_errors=True)

    def v11():
        for k, extra in (("v11", []), ("mid", ["--timing", "mid"])):
            if not need[k] or err:
                continue
            e = {**env, "CUDA_VISIBLE_DEVICES": g11, "V11_TMP": f"/dev/shm/v11_{k}_{a.tag or 'presence'}"}
            log(f"v11 {k} 읽기 → GPU {g11}  (로그 {logs / f'{k}.log'})")
            try:
                run([sys.executable, "-u", str(AN / "v11_readout_online.py"), "--gpus", str(ng(g11)), "--reps", *a.v11_reps] + extra,
                    e, logs / f"{k}.log")
            except SystemExit as x:
                err.append(str(x))
            shutil.rmtree(e["V11_TMP"], ignore_errors=True)
    th = [threading.Thread(target=f) for f, n in ((v3, need["v3"]), (v11, need["v11"] or need["mid"])) if n]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    if err:
        sys.exit("\n".join(err))
    log(f"읽기 끝 {(time.time()-t0)/60:.1f}분")


def supersede(d: Path, fp):
    files = [f for f in d.iterdir() if f.is_file() and f.suffix in KEEP_SUFFIX] if d.exists() else []
    if not files:
        return
    st = d / "_decoder.json"
    s = json.loads(st.read_text()) if st.exists() else None
    if s and s.get("fingerprint") == fp:
        return                                           # 같은 자로 그린 그림 — 덮는다
    name = f"{s['decoder']}_{s['fingerprint']}" if s else f"unstamped_{datetime.datetime.now():%Y%m%d-%H%M}"
    dst = d / "_superseded" / name
    if dst.exists():
        dst = dst.with_name(dst.name + f"_{datetime.datetime.now():%H%M%S}")
    dst.mkdir(parents=True)
    for f in files:
        shutil.move(str(f), dst / f.name)
    (dst / "README.md").write_text(
        f"# {name} — 자동으로 내림 ({datetime.datetime.now():%Y-%m-%d %H:%M})\n\n"
        f"`new_archive_redraw.py` 가 이 폴더의 그림을 다른 자로 다시 그리면서 내렸다.\n"
        f"이 판을 그린 자: {s['decoder'] + ' (지문 ' + s['fingerprint'] + ', ' + s['drawn'] + ')' if s else '도장이 없어 모른다'}.\n"
        f"다시 그리려면 `new_archive_redraw.py --decoder {s['decoder'] if s else '<자>'} --steps figs`.\n")
    log(f"  {d.name}: 지금 그림 {len(files)} 개 → _superseded/{dst.name}/")


def figs(a, fp):
    env = env_for(a)
    for d, cmds in FOLDERS.items():
        if d in ("train_readout", "context_to_future") and not is_identity(EXP / a.decoder):
            log(f"  {d}: 정체 자가 아니라 건너뜀 (presence 자는 plot_readout_errorbars.py)"); continue
        supersede(a.figroot / d, fp)
        for c in cmds:
            if ("--identity" in c or "--acc57" in c) and not is_identity(EXP / a.decoder):
                continue                                         # 정체 유지 그림은 정체 자에서만
            run([sys.executable, str(FG / c[0])] + c[1:], env, None)
        log(f"  {d}: 그림 끝 ({a.figroot / d})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--decoder", required=True, help="exp_results/ 아래 자 폴더 이름 (예: identity, presence)")
    ap.add_argument("--tag", default=None, help="readings 꼬리표 (기본 = 자 이름, presence 는 '' = 옛 폴더)")
    ap.add_argument("--steps", nargs="+", default=["check", "bias", "extract", "figs"],
                    choices=["check", "bias", "extract", "figs"])
    ap.add_argument("--gpus-v3", default="0,1,2,3,4,5")
    ap.add_argument("--gpus-v11", default="6,7")
    ap.add_argument("--figroot", default=str(ROOT / "z_research/RollOutV3/figures"))
    ap.add_argument("--force", action="store_true", help="이미 이 자로 읽힌 readings · 치우침도 다시")
    # 2026-09-26: 네 폴더 그림과 판 검사에 **쓰이는 표현만** 뽑는다 — ViT-H 한 바퀴씩 아껴 5 단계가 ~35 분 → ~20 분.
    #   v3: p (p 자 · p@h) · z (그림의 기준선). h@h 는 그림에 안 쓰인다 (v3_gif · 감사가 쓰면 --v3-reps p z h)
    #   v11: p (그림) · h (판 오탐 검사). z 는 안 쓰인다 (감사가 쓰면 --v11-reps p z h)
    ap.add_argument("--v3-reps", nargs="+", default=["p", "z"], choices=["p", "z", "h"])
    ap.add_argument("--v11-reps", nargs="+", default=["p", "h"], choices=["p", "z", "h"])
    ap.add_argument("--dry", action="store_true", help="check 만 하고 계획을 보인다")
    a = ap.parse_args()
    a.tag = ("" if a.decoder == "presence" else a.decoder) if a.tag is None else a.tag
    a.figroot = Path(a.figroot)
    fp = check(a)
    if a.dry:
        log(f"계획: {' → '.join(s for s in a.steps if s != 'check')} (그림 뿌리 {a.figroot})"); return
    t0 = time.time()
    if "bias" in a.steps:
        bias(a)
    if "extract" in a.steps:
        extract(a, fp)
    if "figs" in a.steps:
        figs(a, fp)
    log(f"끝 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
