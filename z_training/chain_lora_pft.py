#!/usr/bin/env python3
"""encoder LoRA 이어 학습 8 판 — 올리기 · 대기 · 채점 대기 스크립트 (2026-11-02, 사용자: "context encoder LoRA 붙여서 같이 학습 ... sheet 에 있는 것들 다",
"GPU 8 장, 한꺼번에", "release 는 한번에 다 올려").

  setsid nohup /venv/vjepa/bin/python z_training/chain_lora_pft.py > z_training/logs/chain_lora_pft.log 2>&1 &
  상태: z_training/logs/chain_lora_pft.state.json   (판마다 상태 · 채점 진행)

하는 일
  1. 학습 8 판을 GPU 8 장 전부로 (`GPUS=8 bash z_training/tmux_train.sh <config>`, 음의 손실 두 판은 worktree vjepa2_negloss 에서) 올린다.
     판끼리 GPU 를 겹쳐 쓴다. 메모리는 **계산으로** 나눈다: GPU 당 예산 BUDGET_MIB (기본 93,000 MiB = 97,887 의 약 95 %) 안에서
     스모크로 잰 판별 최대 사용량 (nvidia-smi, rank 하나 = GPU 하나) 의 합이 넘지 않을 때만 올린다. 실제 남은 메모리 (nvidia-smi) 도 같이 본다.
     릴리즈 4 판은 맨 처음 한꺼번에 (사용자 지시), AR 은 들어가는 만큼 → 나머지는 앞 판이 끝나 자리가 나면 자동으로.
  2. 판이 끝나면 (run 폴더에 e<EP>.pt + 그 run 의 학습 프로세스 없음) 네 벤치를 GPU 8 장으로 채점한다 (채점 batch 16 그대로, 메모리가 모자라면 기다린다):
     IntPhys 1 late (intphys1_late_uni) · IntPhys 2 late final · real world permanence (uniform) · VOBench v3.
  3. 학습이 OOM 등으로 끝 없이 죽으면 (e<EP>.pt 없음) 동시 학습 수가 그때보다 줄었을 때 다시 올린다 (latest.pt 에서 이어 돈다, 최대 2 번).
  4. 다 끝나면 표 (op_cells_vast.py · op_cells_vob3.py) 를 찍는다.

학습 프로세스 찾기: /proc/*/cmdline 을 `^\\S*python\\S* \\S*launch.py --fname \\S*/runs/<run>/config.yaml` 로 (python 으로 시작하는 줄만 —
이 스크립트 자신이나 감시 셸이 패턴 글자를 담아도 잡히지 않는다). pgrep 은 안 쓴다.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

ROOT = "/workspace/world_model/vjepa2"
NEG = "/workspace/world_model/vjepa2_negloss"
PY = "/venv/vjepa/bin/python"
BUDGET = int(os.environ.get("BUDGET_MIB", 93000))
GPU_IDS = "0 1 2 3 4 5 6 7"
LOG_DIR = f"{ROOT}/z_training/logs"
STATE = f"{LOG_DIR}/chain_lora_pft.state.json"
EVAL_LOG = f"{ROOT}/cvpr/results/eval/_logs_lora_20261102"

# (run, checkout, 레지스트리 모델, 마지막 epoch, GPU 당 MiB (스모크 실측 최대 + 여유), 대조 모델)
# GPU 당 MiB = 스모크 실측 (GPU 1 장 · rank batch 그대로 · 끝 val 포함, nvidia-smi 최대) + 700 MiB (DDP · NCCL 버퍼, GPU 2 장 스모크에서 +330~370)
#   실측: 릴리즈 rank 6 = 10,904 · rank 4 = 8,462 · rank 2 = 6,082 / AR rank 6 = 19,694 · rank 2 = 8,936 MiB (z_training/docs/LORA_PFT_2026-11-02.md)
#   AR 은 realbench (짧고 작다) 를 vobench 앞에 둔다 — 첫 판에 릴리즈 4 + AR v4vc · negimp · realbench 7 판이 예산 안에 든다
JOBS = [
    ("v4vc_release_postft_lora",          ROOT, "vith_release_v4vc_lora",            20, int(os.environ.get("MEM_REL6", 11700)), "vith_release_v4vc"),
    ("v4vc_release_negimp_lora",          NEG,  "vith_release_v4vc_negimp_lora",     20, int(os.environ.get("MEM_REL6", 11700)), "vith_release_v4vc_negimp"),
    ("vobench_release_postft_g_e20_lora", ROOT, "vith_release_vobpft_e20_lora",      20, int(os.environ.get("MEM_REL4", 9200)),  "vith_release_vobpft_e20"),
    ("realbench_release_postft_lora",     ROOT, "vith_release_realbench_lora",       30, int(os.environ.get("MEM_REL2", 6800)),  "vith_release_realbench"),
    ("v4vc_ar_vw_state_postft_lora",      ROOT, "vast_ar_vw_state_v4vc_lora",        20, int(os.environ.get("MEM_AR6", 20400)),  "vast_ar_vw_state_v4vc"),
    ("v4vc_ar_vw_state_negimp_lora",      NEG,  "vast_ar_vw_state_v4vc_negimp_lora", 20, int(os.environ.get("MEM_AR6", 20400)),  "vast_ar_vw_state_v4vc_negimp"),
    ("realbench_ar_vw_state_postft_lora", ROOT, "vast_ar_vw_state_realbench_lora",   30, int(os.environ.get("MEM_AR2", 9700)),  "vast_ar_vw_state_realbench"),
    ("vobench_ar_vw_state_postft_g_lora", ROOT, "vast_ar_vw_state_vobpft_lora",      30, int(os.environ.get("MEM_AR6", 20400)),  "vast_ar_vw_state_vobpft"),
]
FIRST_TOGETHER = {"v4vc_release_postft_lora", "v4vc_release_negimp_lora", "vobench_release_postft_g_e20_lora", "realbench_release_postft_lora"}
# 채점 GPU 당 MiB (스모크 실측: 릴리즈 LoRA 9.9 GiB, AR LoRA 15.6 GiB, batch 16) + 여유
EVAL_MEM = {"rel": int(os.environ.get("EVAL_MEM_REL", 11500)), "ar": int(os.environ.get("EVAL_MEM_AR", 17500))}
# 채점 (벤치, 명령). {m} = 모델
BENCHES = [
    ("intphys1_late_uni", f"DS=intphys1_late_uni GPU_IDS='{GPU_IDS}' bash cvpr/eval/intphys1_late/run.sh {{m}}", "cvpr/results/eval/intphys1_late/intphys1_late_uni_{m}"),
    ("intphys2_late_final", f"GPU_IDS='{GPU_IDS}' bash cvpr/eval/testbed/run.sh intphys2_late_final {{m}}", "cvpr/results/eval/testbed/intphys2_late_final_{m}"),
    ("real_world_permanence_uniform", f"GPU_IDS='{GPU_IDS}' bash cvpr/eval/testbed/run.sh real_world_permanence_uniform {{m}}", "cvpr/results/eval/testbed/real_world_permanence_uniform_{m}"),
    ("vobench_v3", f"GPU_IDS='{GPU_IDS}' bash cvpr/eval/testbed/run.sh vobench_v3 {{m}}", "cvpr/results/eval/testbed/vobench_v3_{m}"),
]
# LoRA 없는 대조판 중 채점이 빠진 칸 (vith_release_vobpft_e20 은 IntPhys 2 late final · real world 가 없다) — 표를 나란히 내려고 같이 채운다
EXTRA_EVALS = [("vith_release_vobpft_e20", "rel")]
MAX_RETRY = 2


def log(msg):
    print(f"[chain_lora_pft] {time.strftime('%F %T')} {msg}", flush=True)


def train_pids(run):
    pat = re.compile(r"^\S*python\S* \S*launch.py --fname \S*/runs/" + re.escape(run) + r"/config.yaml")
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            cmd = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
        except Exception:
            continue
        if pat.match(cmd):
            out.append(int(d))
    return out


def gpu_used_max():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout
        return max(int(x) for x in r.split())
    except Exception:
        return 10 ** 9


def gpu_free_min():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout
        return min(int(x) for x in r.split())
    except Exception:
        return 0


class _Adopted:
    """--resume 때 이미 돌던 채점 셸 (자기 세션) 을 pid 로 지켜본다. 끝나면 returncode 0 으로 둔다 (결과는 per_block.json 으로 따로 확인)."""

    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        try:
            st_ = open(f"/proc/{self.pid}/stat").read().split(")")[1].split()[0]
            if st_ != "Z":
                return None
        except Exception:
            pass
        self.returncode = 0
        return 0


def find_eval_pid(m):
    """채점 셸 = 자기 세션의 우두머리 (pid == session id) 이고 cmdline 에 모델 이름이 있다. bash -c 는 마지막 명령을 exec 하므로
    끝 벤치에서는 cmdline 이 `bash cvpr/eval/testbed/run.sh vobench_v3 <m>` 로 바뀐다 — 둘 다 받는다."""
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            cmd = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
            sid = int(open(f"/proc/{d}/stat").read().split(")")[1].split()[3])
        except Exception:
            continue
        if sid == int(d) and cmd.startswith("bash ") and re.search(r"(^| |')" + re.escape(m) + r"( |'|$)", cmd) and ("sleep 30" in cmd or "cvpr/eval/" in cmd):
            return int(d)
    return None


def run_dir(run, co):
    return f"{co}/z_training/runs/{run}"


def main():
    os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(EVAL_LOG, exist_ok=True)
    st = {j[0]: {"state": "pending", "retries": 0, "launched": None, "fail_ntrain": None} for j in JOBS}
    ev_queue = [(m, kind) for m, kind in EXTRA_EVALS]          # (모델, rel|ar) — 학습이 끝난 판이 뒤에 붙는다
    ev_done, ev_running = [], {}                               # 모델 → (Popen, kind, 시작)
    jobs = {j[0]: j for j in JOBS}
    resumed = "--resume" in sys.argv                           # 2026-11-02: 대기 스크립트만 다시 띄울 때 (학습은 tmux 라 그대로 돈다). 채점이 돌고 있지 않을 때만
    if resumed:
        S = json.load(open(STATE))
        for k, v in S["train"].items():
            st[k].update(v)
        ev_queue, ev_done = [tuple(x) for x in S["eval_queue"]], S["eval_done"]
        for x in S.get("eval_running", []):                    # 돌던 채점 셸은 pid 로 넘겨받는다 (자기 세션이라 대기 스크립트가 죽어도 산다)
            m = x if isinstance(x, str) else x[0]
            pid = x[2] if (not isinstance(x, str) and len(x) > 2 and os.path.exists(f"/proc/{x[2]}")) else find_eval_pid(m)
            if pid is None:
                raise SystemExit(f"--resume: 돌던 채점 {m} 의 셸을 못 찾았다")
            ev_running[m] = (_Adopted(pid), "ar" if m.startswith("vast_ar") else "rel", time.time())
            log(f"--resume: 돌던 채점 넘겨받음 {m} (pid {pid})")
    last_launch = [0.0]

    def reserved():
        r = sum(jobs[k][4] for k, v in st.items() if v["state"] in ("launched", "running"))
        r += sum(EVAL_MEM[v[1]] for v in ev_running.values())
        return r

    def fits(need):
        """예약 합 (스모크 실측 최대 + 여유) 이 예산 안이면 된다. 넘어도 실제 사용량 (nvidia-smi 최대 GPU) + need 가 예산 안이고,
        마지막 올림 뒤 5 분이 지났고 (그 판이 최대 메모리까지 올라왔고) 막 올린 판이 없으면 된다 (2026-11-02 사용자: 여유를 너무 보수적으로 잡지 말 것)."""
        if gpu_free_min() < need + 1024:
            return False
        if reserved() + need <= BUDGET:
            return True
        settled = time.time() - last_launch[0] >= 300 and not any(v["state"] == "launched" for v in st.values())
        # ⚠️ 2026-11-02 13:34 정정: 채점은 벤치마다 프로세스를 새로 띄워 벤치 사이에 메모리가 0 이 된다 → 실제 사용량이 채점 몫을 빠뜨려
        #    AR VOBench 판이 OOM 으로 죽었다. 돌고 있는 채점은 실제 사용량과 상관없이 예약량을 더한다 (겹쳐 세어도 안전한 쪽)
        return settled and gpu_used_max() + need + sum(EVAL_MEM[v[1]] for v in ev_running.values()) <= BUDGET

    def launch(run):
        _, co, model, ep, mem, _ = jobs[run]
        if train_pids(run):
            log(f"올림 거부: {run} 학습 프로세스가 이미 있다"); st[run]["state"] = "running"; return
        last_launch[0] = time.time()
        subprocess.run(["tmux", "kill-session", "-t", f"=train_{run}"], capture_output=True)   # 재시도: 남은 로그 창 세션을 치운다 (이 run 이름만)
        env = {**os.environ, "GPUS": "8"}
        env.pop("NAME", None); env.pop("SET", None); env.pop("LIMIT", None); env.pop("DEVICES", None)
        p = subprocess.run(["bash", "z_training/tmux_train.sh", run], cwd=co, env=env, capture_output=True, text=True)
        log(f"올림 {run} (GPU 당 {mem} MiB 예약, 예약 합 {reserved() + mem}/{BUDGET}, 실제 최대 GPU {gpu_used_max()} MiB): {p.stdout.strip().splitlines()[0] if p.stdout.strip() else p.stderr.strip()}")
        st[run].update(state="launched", launched=time.time())

    def dump():
        json.dump({"budget": BUDGET, "reserved": reserved(), "train": st, "eval_queue": ev_queue, "eval_running": [[m, v[1], v[0].pid] for m, v in ev_running.items()],
                   "eval_done": ev_done, "time": time.strftime("%F %T")}, open(STATE, "w"), indent=1, ensure_ascii=False)

    log(f"시작: 학습 {len(JOBS)} 판, 예산 GPU 당 {BUDGET} MiB" + (" (--resume: 상태 파일에서 이어감)" if resumed else ""))
    first = not resumed
    while True:
        # ── 학습 상태 갱신 ────────────────────────────────────────────────────
        for run, v in st.items():
            if v["state"] not in ("launched", "running"):
                continue
            _, co, model, ep, _, _ = jobs[run]
            alive = bool(train_pids(run))
            if alive:
                v["state"] = "running"
                continue
            if v["state"] == "launched" and time.time() - v["launched"] < 300:   # 아직 모델 로딩 전 (tmux → train.sh → launch.py)
                continue
            if os.path.exists(f"{run_dir(run, co)}/e{ep}.pt"):
                v["state"] = "done"; v["t_done"] = time.strftime("%F %T")
                log(f"학습 끝: {run} → 채점 대기열에 {model}")
                ev_queue.append((model, "ar" if "_ar_" in run else "rel"))
            else:
                others = sum(1 for k, x in st.items() if k != run and x["state"] in ("launched", "running"))
                tail = ""
                try:
                    tail = subprocess.run(["tail", "-n", "200", f"{run_dir(run, co)}/train.log"], capture_output=True, text=True).stdout
                except Exception:
                    pass
                oom = "out of memory" in tail.lower()
                v["retries"] += 1
                v["fail_ntrain"] = others
                v["state"] = "failed" if v["retries"] > MAX_RETRY else "retry"
                log(f"학습이 끝 없이 죽었다: {run} (OOM {oom}, 그때 같이 돈 다른 학습 {others}) → {v['state']}")
        # ── 채점 상태 갱신 ────────────────────────────────────────────────────
        for m in list(ev_running):
            p, kind, t0 = ev_running[m]
            if p.poll() is not None:
                ev_done.append({"model": m, "rc": p.returncode, "min": round((time.time() - t0) / 60, 1)})
                log(f"채점 끝: {m} (rc {p.returncode}, {(time.time() - t0) / 60:.1f} 분)")
                del ev_running[m]
        # ── 학습 올리기 ──────────────────────────────────────────────────────
        if first:
            for run in [r for r in st if r in FIRST_TOGETHER]:
                launch(run)
            first = False
            time.sleep(20)
        ntrain = sum(1 for x in st.values() if x["state"] in ("launched", "running"))
        for run, v in st.items():
            if v["state"] not in ("pending", "retry"):
                continue
            if v["state"] == "retry" and ntrain > 0 and ntrain >= v["fail_ntrain"]:
                continue                                       # 재시도: 같이 도는 학습 수가 실패 때보다 줄었을 때만 (그 전엔 건너뛰고 다음 판을 본다)
            need = jobs[run][4]
            if fits(need):
                launch(run); ntrain += 1
                time.sleep(20)
                continue
            break                                              # 처음 올리는 판은 줄 순서를 지킨다 (앞 판이 안 들어가면 뒤 판도 기다린다)
        # ── 채점 올리기 ──────────────────────────────────────────────────────
        while ev_queue:
            m, kind = ev_queue[0]
            if m in ev_running:
                break
            if not fits(EVAL_MEM[kind]):
                break
            last_launch[0] = time.time()
            ev_queue.pop(0)
            cmds = " ; ".join(f"echo '### {b} {m}' ; " + c.format(m=m) for b, c, _ in BENCHES)   # 이미 있는 결과는 launch.sh 가 건너뛴다 (대조판 빈 칸 채우기에 그대로 쓴다)
            f = open(f"{EVAL_LOG}/{m}.log", "a")
            p = subprocess.Popen(["bash", "-c", f"sleep 30; cd {ROOT} && {cmds}"], stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            ev_running[m] = (p, kind, time.time())
            log(f"채점 올림: {m} ({kind}, GPU 당 {EVAL_MEM[kind]} MiB 예약) → {EVAL_LOG}/{m}.log")
        dump()
        # ── 끝? ────────────────────────────────────────────────────────────
        if all(v["state"] in ("done", "failed") for v in st.values()) and not ev_queue and not ev_running:
            break
        time.sleep(30)

    log("학습 · 채점 끝. 표:")
    models = [j[2] for j in JOBS]
    pairs = []
    for j in JOBS:
        pairs += [j[5], j[2]]
    subprocess.run([PY, "z_research/scripts/analysis/op_cells_vast.py", *pairs], cwd=ROOT)
    subprocess.run([PY, "z_research/scripts/analysis/op_cells_vob3.py", *pairs], cwd=ROOT)
    log("CHAIN_LORA_PFT_DONE " + json.dumps({k: v["state"] for k, v in st.items()}))


if __name__ == "__main__":
    main()
