#!/bin/bash
# EK100 anticipation 학습 모니터 (ANSI 색).
#
#   watch -c -n 1 bash z_research/anticipation/EK100/monitor.sh                                          # 가장 최근에 갱신된 run
#   watch -c -n 1 bash z_research/anticipation/EK100/monitor.sh ek100_vith_official256_released_grid8    # TAG 지정
#   NO_COLOR=1 bash z_research/anticipation/EK100/monitor.sh <TAG>                                        # 색 없이 (파일로 남길 때)
#
# 보여 주는 것: SLURM 상태 (대기 중이면 이유) · 설정 요약 · epoch/iter 진행률과 남은 시간 · GPU 메모리 (rank 0 최대, 로그의 [mem]) ·
# 최신 train 지표 · epoch 별 val (val_metrics.jsonl, head 가 여럿이면 지표마다 head 중 max, 열마다 최고값 강조) ·
# 마지막 epoch 의 head 별 표 · "늘 같은 top-5" 경고 · 에러 줄. 로그·결과는 NFS 라 job 이 vll3 에서 돌아도 어느 노드에서 봐도 된다.
set -uo pipefail
ROOT=/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/anticipation/EK100/exp_results/action_anticipation_frozen
if [[ -n "${1:-}" ]]; then RUN="$ROOT/$1"; else RUN=$(ls -td "$ROOT"/*/ 2>/dev/null | head -1); fi
RUN=${RUN%/}; TAG=$(basename "$RUN")
Q=$(squeue -u "$USER" -h -n "ek100_$TAG" -o "%i|%T|%N|%M|%l|%R" 2>/dev/null | head -1)
exec /data/hyuntak/anaconda3/envs/vjepa2/bin/python - "$RUN" "$TAG" "$Q" "$(hostname -s)" <<'PYEOF'
import json, os, re, sys, time
from datetime import datetime
run, tag, q, host = sys.argv[1:5]
C = not os.environ.get("NO_COLOR")
def c(s, *codes):
    return f"\033[{';'.join(map(str, codes))}m{s}\033[0m" if C else str(s)
B, DIM, RED, GRN, YEL, BLU, MAG, CYN, GRY = 1, 2, 31, 32, 33, 34, 35, 36, 90
W = 104
rule = lambda ch="─": c(ch * W, GRY)
def section(t):
    print(c(f"── {t} ", CYN, B) + c("─" * max(W - len(t) - 4, 0), GRY))
def bar(cur, tot, n=30, col=GRN):
    k = round(n * cur / max(tot, 1)); return c("█" * k, col) + c("░" * (n - k), GRY)

print(c("━" * W, CYN))
job = dict(zip(("id", "state", "node", "elapsed", "limit", "reason"), q.split("|"))) if q else {}
title = c(f" EK100 · {tag}", B, CYN)
if job:
    st = job["state"]
    badge = c(f" {st} ", B, 30, 42) if st == "RUNNING" else c(f" {st} ", B, 30, 43)
    where = c(f"@ {job['node']}", B) if st == "RUNNING" else c(f"대기 이유 {job['reason']}", YEL)
    print(f"{title}   {badge}  job {job['id']}  {where}  " + c(f"경과 {job['elapsed']} / {job['limit']}", GRY))
else:
    print(f"{title}   " + c(" NOT IN QUEUE ", B, 30, 47) + c("  (끝났거나 run.sh 로 직접 실행)", GRY))
if not os.path.isdir(run):
    print(c(f" run 폴더 없음: {run}", YEL) + (c("   (job 이 시작하면 생긴다)", GRY) if job else "")); print(c("━" * W, CYN)); sys.exit()
log = os.path.join(run, "stdout.log")
txt = open(log, errors="replace").read() if os.path.exists(log) else ""
age = time.time() - os.path.getmtime(log) if txt else None
age_s = (c(f"{int(age)}s 전 갱신", GRN if age < 180 else RED)) if age is not None else c("아직 없음", YEL)
print(c(" log ", GRY) + c(log.replace("/data/hyuntak/project/2026/2027_cvpr/vjepa2/", ""), GRY) + f"   ({age_s})")
heads = []
try:
    import yaml
    cfg = yaml.safe_load(open(os.path.join(run, "_resolved.yaml")))
    o, d, w = cfg["experiment"]["optimization"], cfg["experiment"]["data"], cfg["model_kwargs"]["wrapper_kwargs"]
    heads = o["multihead_kwargs"]
    hs = f"head 1 (lr {heads[0]['lr']:g}, wd {heads[0]['weight_decay']:g})" if len(heads) == 1 else f"head {len(heads)} (지표 = head 중 max)"
    kv = lambda k, v: c(k, GRY) + " " + c(v, B)
    print(" " + "   ".join([kv("probe", hs), kv("ap", d.get("anticipation_point_mode")), kv("time", d.get("time_source")),
                            kv("mask", w.get("mask_index", 1)), kv("batch", f"{o['batch_size']}/GPU"), kv("epoch", o["num_epochs"])]))
    n_ep = int(o["num_epochs"])
except Exception:
    n_ep = 20
print(c("━" * W, CYN))

T = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
ipe = re.findall(r"Dataloader created\.\.\. iterations per epoch: (\d+)", txt)
ipe = int(ipe[0]) if ipe else 525
epochs = re.findall(r"\]\[main\s*\] Epoch (\d+)", txt)
ep = int(epochs[-1]) if epochs else 0
it = re.findall(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]\[root\s*\]\[train_one_epoch\s*\] \[\s*(\d+)\] acc \(v/n\): ([\d.]+)% \(([\d.]+)% ([\d.]+)%\) recall \(v/n\): ([\d.]+)% \(([\d.]+)% ([\d.]+)%\) \[mem: ([\d.e+]+)\] \[data: ([\d.]+) ms\]", txt)
vals = re.findall(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]\[root\s*\]\[validate_exact\s*\] \[exact val\] itr 0 ", txt)
vend = re.findall(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]\[root\s*\]\[main\s*\] \[exact val\] epoch (\d+)", txt)
in_val = bool(vals) and len(vals) > len(vend)

section("progress")
if it:
    cur = int(it[-1][1]); spi = None
    # 최근 몇 개 로그 줄 (같은 epoch 안) 로 s/iter 을 잰다 — 두 줄만 쓰면 흔들린다
    recent = [x for x in it[-6:] if int(x[1]) <= cur]
    pairs = [(T(b[0]) - T(a[0])).total_seconds() / (int(b[1]) - int(a[1])) for a, b in zip(recent, recent[1:]) if int(b[1]) > int(a[1])]
    if pairs:
        spi = sum(pairs) / len(pairs)
    vsec = (T(vend[-1][0]) - T(vals[len(vend) - 1])).total_seconds() if vend else 240.0
    print(f" epoch {c(f'{ep + 1:>2}', B)}/{n_ep}  {bar(ep, n_ep, 20, BLU)}   iter {c(f'{cur:>3}', B)}/{ipe}  {bar(cur, ipe)}" + (c("   ◀ val 중", YEL, B) if in_val else ""))
    if spi:
        left_ep = (ipe - cur) * spi
        left_all = left_ep + vsec + (n_ep - ep - 1) * (ipe * spi + vsec)
        eta = datetime.fromtimestamp(time.time() + left_all).strftime("%m-%d %H:%M")
        print(" " + c(f"{spi:.2f} s/iter", B) + c(f" (data {float(it[-1][9]):.0f} ms)", GRY) + f"   epoch 당 {c(f'{(ipe * spi + vsec) / 60:.0f}분', B)}"
              + f"   이 epoch 남은 {left_ep/60:.0f}분   전체 남은 " + c(f"{left_all/3600:.1f}시간", B) + "  →  " + c(f"{eta} 끝", MAG, B))
    mem_gb = float(it[-1][8]) / 1024; frac = min(mem_gb / 24.0, 1)
    mcol = RED if mem_gb > 22 else (YEL if mem_gb > 20.5 else GRN)
    print(f" GPU mem (rank 0 최대)  {bar(mem_gb, 24.0, 20, mcol)}  " + c(f"{mem_gb:.1f}", B, mcol) + c(" / 24 GB", GRY) + (c("   ⚠ 한계에 가깝다", RED, B) if mem_gb > 22 else ""))
    a = it[-1]
    print(" train " + c("(epoch 누적, head 중 max)", GRY) + f"   top-5 acc  action {c(a[2], B)}  verb {a[3]}  noun {a[4]}"
          + c("   │   ", GRY) + f"recall@5  action {c(a[5], B)}  verb {a[6]}  noun {a[7]}")
else:
    print(c(" 아직 학습 iter 로그가 없다 (모델 로딩 중이거나 시작 전)", YEL))

vm = os.path.join(run, "val_metrics.jsonl")
rows = [json.loads(l) for l in open(vm)] if os.path.exists(vm) else []
section("val · mean class recall@5" + ("  (head 중 max, 지표마다 따로)" if len(heads) > 1 else ""))
CONST = (6.85, 2.49, 0.45)          # 모든 clip 에 같은 top-5 를 낼 때 (val 클래스 73 / 201 / 1,114)
print(c(f"   {'epoch':>5}   {'verb':>7} {'noun':>7} {'action':>7}   {'clips':>11}", GRY))
if rows:
    best = {k: max(r[k]["recall"] for r in rows) for k in ("verb", "noun", "action")}
    if len(rows) > 12:
        print(c(f"   … 앞 {len(rows) - 12} epoch 생략 (강조 = 전체 epoch 중 최고)", GRY))
    for r in rows[-12:]:
        cell = lambda k: c(f"{r[k]['recall']:7.2f}", B, GRN) if r[k]["recall"] == best[k] else f"{r[k]['recall']:7.2f}"
        const = abs(r["verb"]["recall"] - CONST[0]) < 0.1 and abs(r["noun"]["recall"] - CONST[1]) < 0.1 and abs(r["action"]["recall"] - CONST[2]) < 0.05
        ok = r["n_clips"] == r["expected_clips"]
        print(f"   {r['epoch']:>5}   {cell('verb')} {cell('noun')} {cell('action')}   " + c(f"{r['n_clips']}/{r['expected_clips']}", GRY if ok else RED)
              + (c("   ⚠ 늘 같은 top-5 (붕괴)", RED, B) if const else ""))
else:
    print(c("   (아직 없음 — epoch 이 끝나야 생긴다)", GRY))
print(c(f"   {'paper':>5}   {59.2:7.2f} {54.6:7.2f} {36.5:7.2f}   ViT-H Table 5 (head 20 중 best, 릴리즈 규약)", GRY))

if rows and len(rows[-1].get("per_head", [])) > 1:
    r = rows[-1]; ph = r["per_head"]; hd = r.get("heads", [{}] * len(ph))
    bi = max(range(len(ph)), key=lambda i: ph[i]["action"]["recall"])
    section(f"epoch {r['epoch']} · head 별")
    print(c(f"   {'head':>4}   {'lr':>7} {'wd':>7}   {'verb':>7} {'noun':>7} {'action':>7}", GRY))
    for i, (h, m) in enumerate(zip(hd, ph)):
        const = abs(m["verb"]["recall"] - CONST[0]) < 0.1 and abs(m["action"]["recall"] - CONST[2]) < 0.05
        line = f"   {i:>4}   {h.get('lr', float('nan')):7.0e} {h.get('wd', float('nan')):7.0e}   {m['verb']['recall']:7.2f} {m['noun']['recall']:7.2f} {m['action']['recall']:7.2f}"
        print(c(line, B, GRN) + c("  ◀ action best", GRN) if i == bi else (c(line, RED) + c("  ⚠ 붕괴", RED, B) if const else line))

err = [l for l in txt.splitlines() if re.match(r"^(Traceback|\S*Error:|RuntimeError|.*CUDA out of memory|.*OutOfMemoryError)", l)]
if err:
    section("errors")
    for l in err[-3:]:
        print(c("   " + l[:W - 4], RED))
print(c("━" * W, CYN) + c(f"\n {datetime.now().strftime('%m-%d %H:%M:%S')} · {host}", GRY))
PYEOF
