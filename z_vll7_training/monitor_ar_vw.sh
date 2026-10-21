#!/bin/bash
# =============================================================================
# 짧은 문맥 · 긴 예측 AR (kind=ar, 가변 창) 모니터 — vll7 (2026-10-21)
#
#   watch -c -n 5 bash z_vll7_training/monitor_ar_vw.sh                 # 도는 run 자동 선택
#   watch -c -n 5 bash z_vll7_training/monitor_ar_vw.sh ar_vw_shortctx  # 지정
#   MONITOR_COLOR=0 bash z_vll7_training/monitor_ar_vw.sh               # 색 끄기
#
# z_vll7_training/monitor.sh (헤더 · 진행 · ETA · epoch · GPU · ckpt · val) 를 그대로 쓰고,
# 그 loss 구역만 빼고 (MONITOR_NO_LOSS=1) 이 run 의 **축**으로 다시 찍는다:
#
#   K 축   예측 길이 (블록) 마다 tf · ar · **ar−tf**.  이 run 의 질문이 "긴 예측이 되는가" 이므로
#          **ar−tf 가 K 와 함께 얼마나 벌어지는가**가 주 지표다 (tf = 한 걸음, ar = K 걸음 되먹임).
#   C 축   문맥 1 블록 vs 2 블록 — 문맥을 한 블록 더 주면 얼마나 싸지는가
#   칸     (C,K) 10 칸 전부
#   데이터 k400 : ssv2 실측 (이 격자 기대 59.8 : 40.2 — 긴 창 격자의 76.7 : 23.3 과 다르다)
#
# ⚠️ tf 와 ar 은 **같은 공간의 L1 이지만 자리 수가 다르다** (tf 는 nb−1 자리 전부, ar 은 R 자리).
#    K 가 크면 ar 이 먼 블록을 포함하므로 **K 사이의 ar 절대값 비교는 과제 난이도 차이를 섞는다.**
#    같은 K 안에서 시간에 따른 변화를, 그리고 ar−tf 를 본다.
# ⚠️ 학습 손실이 내려가는 것은 **복사 대비 우위가 아니다.** 채점은 `surprise.ar_baseline=copy_last` 와 같이 낸다.
# =============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh" >/dev/null
RUN=${1:-}
if [[ -z "$RUN" ]]; then
  RUN=$(tmux ls -F '#{session_name}' 2>/dev/null | grep '^train_' | head -1 | sed 's/^train_//')
  [[ -z "$RUN" ]] && RUN=$(ls -t "$VLL7_RUNS" 2>/dev/null | head -1)
fi
D="$VLL7_RUNS/$RUN"
[[ -d "$D" ]] || { echo "run 없음: $D   (있는 것: $(ls "$VLL7_RUNS" 2>/dev/null | tr '\n' ' '))"; exit 1; }

# 마지막 ━ 줄만 빼고 (아래에서 다시 긋는다) 그대로 쓴다
MONITOR_NO_LOSS=1 bash "$(dirname "${BASH_SOURCE[0]}")/monitor.sh" "$RUN" 2>/dev/null \
  | awk 'NR>1 && prev !~ /━━━/ {print prev} {prev=$0} END {if (prev !~ /━━━/) print prev}'

"$VJEPA2_PY" - "$D" <<'EOF'
import os, re, sys
from collections import OrderedDict, Counter, defaultdict
D = sys.argv[1]
col = os.environ.get("MONITOR_COLOR", "1") != "0"
def c(s, a): return f"\033[{a}m{s}\033[0m" if col else str(s)
B, DIM, GRN, YEL, MAG = "1", "2", "32", "33", "35"
W = 78
tl = os.path.join(D, "train.log")

TUB = 2   # tubelet — 로그의 C/K 는 **장** 수다. 블록 = 장 / 2
# [e, i] loss L (avg A) [tf T ar R R=n] ... | mask var CxKy <dataset> B<b>
pat = re.compile(r"\[(\d+), +(\d+)\] loss ([\d.]+) \(avg ([\d.]+)\) \[tf ([\d.]+) ar ([\d.]+) R=(\d+)\].*?"
                 r"\| mask var C(\d+)K(\d+) (\w+) B(\d+)")
seen = OrderedDict()
if os.path.isfile(tl):
    for ln in open(tl, errors="ignore"):
        m = pat.search(ln)
        if m:
            seen[(m.group(1), m.group(2))] = m.groups()
S = list(seen.values())
if not S:
    print(c(" 아직 var_window loss 줄이 없다 (첫 step 대기 — ViT-H 로드 · torch.compile)", DIM))
    print(c("━" * W, DIM)); raise SystemExit

e, i, l, avg, tf, ar, R, Cf, Kf, ds, Bb = S[-1]
print(f" {c('loss', B):>7}  {c(f'{float(l):.4f}', B)}  =  tf {float(tf):.4f}  +  ar {float(ar):.4f}"
      f"   {c(f'(epoch 평균 {float(avg):.4f})', DIM)}"
      f"   {c(f'지금 C{int(Cf)//TUB}K{int(Kf)//TUB} 블록 · {ds} · B{Bb} · R={R}', DIM)}")

REC = S[-600:]                       # 최근 600 step (중복 제거 후) 만
def mean(v): return sum(v) / len(v) if v else float("nan")

# ── K 축 (예측 길이) ────────────────────────────────────────────────────────
byK = defaultdict(lambda: ([], []))
byC = defaultdict(lambda: ([], []))
byCK = defaultdict(lambda: ([], []))
for g in REC:
    k, cc = int(g[8]) // TUB, int(g[7]) // TUB
    for d, key in ((byK, k), (byC, cc), (byCK, (cc, k))):
        d[key][0].append(float(g[4])); d[key][1].append(float(g[5]))

print(f" {c('K 축', B):>7}  {c('예측 길이 (블록)', B+';'+MAG)}    "
      + "   ".join(f"{k:>2}" for k, _ in sorted(byK.items())) + c("     ← 1 블록 = 2 장 = 0.5 s (fps 4)", DIM))
for lab, idx in (("tf", 0), ("ar", 1)):
    print(f" {'':>7}    {lab:<16}  " + "  ".join(f"{mean(v[idx]):5.3f}" for _, v in sorted(byK.items())))
print(f" {'':>7}    {c('ar − tf', B+';'+YEL):<25}  "
      + "  ".join(c(f"{mean(v[1]) - mean(v[0]):+5.3f}", B+';'+YEL) for _, v in sorted(byK.items()))
      + c("   ← 되먹임이 한 걸음 대비 얼마나 나쁜가", DIM))
print(f" {'':>7}    {c('n step', DIM):<18}  " + "  ".join(f"{len(v[0]):5d}" for _, v in sorted(byK.items())))

# ── C 축 · 칸 ──────────────────────────────────────────────────────────────
print(f" {c('C 축', B):>7}  " + "    ".join(
    f"C{cc} 블록: tf {mean(v[0]):.3f} ar {mean(v[1]):.3f} (n {len(v[0])})" for cc, v in sorted(byC.items())))
cells = sorted(byCK.items())
for row in (cells[:5], cells[5:]):
    if row:
        print(f" {'':>7}  " + "  ".join(f"C{a}K{b}:{mean(v[1]):.3f}" for (a, b), v in row) + c("   (ar)", DIM))

# ── 데이터 비율 ────────────────────────────────────────────────────────────
cnt = Counter(g[9] for g in S); tot = sum(cnt.values())
print(f" {c('데이터', B):>7}  " + "  ".join(f"{k} {100 * v / tot:.1f}%" for k, v in cnt.most_common())
      + c("   (step 기준. 이 격자 기대 k400 60.0 : ssv2 40.0)", DIM))
print(c("━" * W, DIM))
EOF
