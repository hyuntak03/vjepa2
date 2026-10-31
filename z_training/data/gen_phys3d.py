#!/usr/bin/env python
"""phys3d 대량 생성 드라이버 (2026-10-19). Blender 워커 W 개를 띄워 작업을 나눠 주고, 끝나면 manifest 를 모은다.

    python z_training/data/gen_phys3d.py --out /workspace/dataset/training_data/phys3d_v1/train --per-scenario 2000 --gpus 2,3,4,5,6,7 --gpu-per 4 --cpu-workers 120
    python z_training/data/gen_phys3d.py --out .../val --per-scenario 80 --seed-base 1000000

- 작업 = (시나리오, k). seed = seed_base + k, 주 물체 정체 = k % 48 (시나리오마다 층화).
- 워커 하나 = blender 프로세스 하나. 모든 워커가 같은 작업 목록을 돌며 claims/ 에 O_EXCL 로 표시를 만든 쪽이 가져간다.
  GPU 워커 = Cycles OptiX (CUDA_VISIBLE_DEVICES 를 따른다), CPU 워커 = Cycles CPU 스레드 1.
  ⚠️ EEVEE 는 쓰지 말 것 — EGL 이 CUDA_VISIBLE_DEVICES 를 무시해 워커 전부가 GPU 0 에 붙는다 (2026-10-19, 학습 GPU 메모리 96.9/97.9 GB).
- 이어 하기: 결과 jsonl 에 있는 작업은 건너뛴다. 같은 명령을 다시 치면 된다.
- 산출물: <out>/clips/<scenario>/<scenario>_<k>.{mp4,json,_states.npz}, index.csv · boundary.tsv · manifest.tsv · failed.tsv
  (boundary.tsv 열은 phys2d_sup 과 같다 → var_window boundary 모드 · build_phys2d_window_index.py --dv-row 1.0 그대로)
"""
import argparse, glob, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
BLENDER = os.environ.get("BLENDER", "/workspace/tools/blender-4.2.3-linux-x64/blender")
SCEN = ["drop_bounce", "projectile", "toss_up", "ramp_roll", "ledge_fall", "pendulum", "occ_pass", "moving_screen",
        "drop_behind", "shelf_land", "wall_bounce", "collide", "stack_topple", "container"]
CAT = {"drop_bounce": "gravity", "projectile": "gravity", "toss_up": "gravity", "ramp_roll": "gravity", "ledge_fall": "gravity",
       "pendulum": "gravity", "occ_pass": "permanence", "moving_screen": "permanence", "drop_behind": "permanence",
       "shelf_land": "solidity", "wall_bounce": "solidity", "collide": "solidity", "stack_topple": "solidity", "container": "solidity"}
N_ID = 48
# --single: 물체 1 개 판 (gen_phys3d_single_blender.py, 2026-10-19). uniform.tsv (4 장 간격 경계, ctx_start 0) 를 더 만든다
SCEN_SINGLE = ["drop_bounce", "projectile", "toss_up", "ramp_roll", "ledge_fall", "pendulum", "occ_pass", "moving_screen",
               "drop_behind", "shelf_land", "wall_bounce", "obstacle_bounce", "container"]
CAT.update(obstacle_bounce="solidity")
UNIFORM_STEP, UNIFORM_MIN_CTX, UNIFORM_K = 4, 4, 16


def uniform_rows(T, rng, stride=1):
    """경계 = UNIFORM_STEP 간격 (clip 마다 시작 위치 무작위) 의 모든 자리, 문맥 4 장 이상 · 예측 16 장 (gen_phys2d_single 과 같다).
    stride s (학습이 s 장마다 1 장을 본다, v2 는 3): 간격 · 문맥이 s 배, 예측 16 장은 렌더 s·15 + 1 장. s = 1 이면 v1 과 같다."""
    step, k = UNIFORM_STEP * stride, stride * (UNIFORM_K - 1) + 1
    return list(range(UNIFORM_MIN_CTX * stride + int(rng.integers(0, step)), T - k + 1, step))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-scenario", type=int, required=True)
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--gpus", default="2,3,4,5,6,7", help="GPU 워커를 둘 GPU (빈 문자열 = CPU 만)")
    ap.add_argument("--gpu-per", type=int, default=4, help="GPU 한 장당 워커 (워커 ≈ 0.6~1.6 GB)")
    ap.add_argument("--cpu-workers", type=int, default=120)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--single", action="store_true", help="물체 1 개 판 (gen_phys3d_single_blender.py)")
    ap.add_argument("--single-v2", action="store_true", help="물체 1 개 v2 (gen_phys3d_single_v2_blender.py, 2026-10-21)")
    ap.add_argument("--defer-render", action="store_true", help="multi: 라벨 워커 (CPU, --cpu-workers) 는 장면을 .blend 로 남기고 렌더 워커 (GPU 마다 --render-per) 가 렌더")
    ap.add_argument("--render-per", type=int, default=4)
    ap.add_argument("--flow", action="store_true", help="multi 의 phys3d_flow 판 (gen_phys3d_flow_blender.py, 2026-10-27) — 오래 이어지는 운동 · 시나리오 45")
    ap.add_argument("--perm", action="store_true", help="multi 의 phys3d_perm 판 (gen_phys3d_perm_blender.py, 2026-10-30) — 물체 영속 · 4 fps 32 장 · 가림 5 종")
    ap.add_argument("--multi", action="store_true", help="phys3d_multi (gen_phys3d_multi_blender.py, 2026-10-25) — 4 fps 64 장 · 문맥 끝 고정 · 가림 · 배경")
    ap.add_argument("--single-v3", action="store_true", help="물체 1 개 v3 (gen_phys3d_single_v3_blender.py, 2026-10-23) — 공중 붙잡기 없음 · 4 fps 저장 (--stride 1)")
    ap.add_argument("--stride", type=int, default=1, help="uniform 행: 학습이 몇 장마다 1 장을 보나 (v2 = 3)")
    ap.add_argument("--pin", type=int, default=1, help="CPU 워커를 코어 하나에 묶기 (taskset)")
    a = ap.parse_args()
    if a.multi:
        return main_multi(a)
    v3 = a.single_v3
    a.single_v2 = a.single_v2 or v3                 # v3 는 행 · 열이 v2 와 같다 (장 단위만 4 fps)
    a.single = a.single or a.single_v2
    script = "gen_phys3d_single_v3_blender.py" if v3 else "gen_phys3d_single_v2_blender.py" if a.single_v2 else "gen_phys3d_single_blender.py" if a.single else "gen_phys3d_blender.py"
    names = (SCEN_SINGLE if a.single else SCEN) if a.scenarios == "all" else a.scenarios.split(",")
    clips = os.path.join(a.out, "clips")
    wdir = os.path.join(a.out, "_work")
    os.makedirs(wdir, exist_ok=True)

    if not a.collect_only:
        jobs = [(n, k) for k in range(a.per_scenario) for n in names]       # 섞어서 시나리오가 고르게 진행
        jf = os.path.join(wdir, "jobs.tsv")
        with open(jf, "w") as f:
            for n, k in jobs:
                f.write(f"{n}\t{k}\t{a.seed_base + k}\t{k % N_ID}\n")
        # 이어 하기: 결과가 있는 작업만 가져간 것으로 남기고, 나머지 표시는 지운다 (죽은 워커가 쥐고 있던 것)
        claims = os.path.join(wdir, "claims")
        os.makedirs(claims, exist_ok=True)
        done = {json.loads(l)["job"].replace("/", "_") for f in glob.glob(os.path.join(wdir, "res_*.jsonl")) for l in open(f) if l.strip()}
        for c in os.listdir(claims):
            if c not in done:
                os.remove(os.path.join(claims, c))
        # 워커: GPU 마다 gpu_per 개 (Cycles OptiX, CUDA_VISIBLE_DEVICES 를 따른다) + CPU 워커 (Cycles CPU, 스레드 1)
        spec = [("cycles", g) for g in a.gpus.split(",") if g for _ in range(a.gpu_per)] + [("cpu", "")] * a.cpu_workers
        run = int(time.time())
        procs = []
        cores = sorted(os.sched_getaffinity(0))
        for w, (eng, g) in enumerate(spec):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, OMP_NUM_THREADS="1", PYTHONHASHSEED="0")
            # CPU 워커는 코어 하나에 묶는다 — -t 1 이어도 OIDN 디노이즈 (TBB) 가 코어 수만큼 스레드를 써서 워커가 200 % 넘게 돌고
            # 216 워커에 load 2,400 이 됐다 (2026-10-19). 뒤쪽 코어부터 쓴다 (앞쪽은 학습 dataloader 몫으로 남긴다)
            pin = ["taskset", "-c", str(cores[-1 - (w % len(cores))])] if eng == "cpu" and a.pin else []
            # -t 1: blender 는 기본으로 코어 수 (256) 만큼 스레드를 띄운다 — 워커 150 개면 컨테이너 pids.max (62,720) 에 걸려
            # 일부가 조용히 죽었다 (2026-10-19 시험 156 개 중 62). 렌더 스레드는 setup_render 가 따로 정한다
            cmd = pin + ["nice", "-n", str(a.nice), BLENDER, "-t", "1", "-b", "-noaudio", "-P", os.path.join(HERE, script), "--",
                   "--jobs", jf, "--claims", claims, "--results", os.path.join(wdir, f"res_{run}_{w:03d}.jsonl"), "--out", clips,
                   "--frames", str(a.frames), "--engine", eng]
            procs.append(subprocess.Popen(cmd, stdout=open(os.path.join(wdir, f"log_{w:03d}.txt"), "a"), stderr=subprocess.STDOUT, env=env))
        print(f"워커 {len(spec)} (GPU {sum(e == 'cycles' for e, _ in spec)} · CPU {a.cpu_workers}), 작업 {len(jobs):,} (남은 {len(jobs) - len(done):,})", flush=True)
        t0, total, n0 = time.time(), len(jobs), len(done)
        while any(p.poll() is None for p in procs):
            time.sleep(60)
            n = sum(sum(1 for _ in open(f)) for f in glob.glob(os.path.join(wdir, "res_*.jsonl")))
            el = time.time() - t0
            print(f"[{n:,}/{total:,}] {el / 60:.1f} 분" + (f", 남은 ~{el / max(n - n0, 1) * (total - n) / 60:.0f} 분" if n > n0 else ""), flush=True)
        bad = [w for w, p in enumerate(procs) if p.returncode != 0]
        if bad:
            print(f"⚠️ 워커 {bad} 비정상 종료 — _work/log_*.txt 확인, 같은 명령으로 이어 하기", flush=True)

    res = [json.loads(l) for f in sorted(glob.glob(os.path.join(wdir, "res_*.jsonl"))) for l in open(f) if l.strip()]
    res.sort(key=lambda r: (r["name"], r["k"]))
    good = [r for r in res if r["ok"]]
    with open(os.path.join(a.out, "failed.tsv"), "w") as fh:
        fh.write("name\tk\tmain_identity\twhy\n")
        fh.writelines(f"{r['name']}\t{r['k']}\t{r['main_identity']}\t{'; '.join(r['why'])}\n" for r in res if not r["ok"])
    with open(os.path.join(a.out, "index.csv"), "w") as fh:
        fh.writelines(f"{r['mp4']} 0\n" for r in good)
    bcols = ["path", "n_frames", "boundary_frame", "ctx_start", "video_frames", "event", "event_frame", "obj_role", "other", "dv", "scenario", "category"]
    if a.single_v2:                       # pred_min: 사건의 결과 (가림 → 다시 보임) 가 예측에 들어가려면 필요한 렌더 장 수 · ctx_gate: 앞선 충돌 · 이탈 (v1 의 ctx_start)
        bcols = bcols + ["pred_min", "ctx_gate"]
    mcols = ["category", "name", "k", "n_frames", "attempt", "n_objects", "main_shape", "main_color", "main_identity", "n_occluders", "g",
             "n_events", "n_occ_episodes", "n_rows", "sec"]
    xcols = ["n_ep_target", "n_timed_screens", "hidden_frac", "n_hidden_frames", "n_hidden_spans", "hidden_at_start", "hidden_at_end",
             "n_wall_hits"] + (["max_blind_run"] if a.single_v2 else []) if a.single else []
    nb = nu = 0
    import numpy as np
    fu = open(os.path.join(a.out, "uniform.tsv"), "w") if a.single else None
    with open(os.path.join(a.out, "boundary.tsv"), "w") as fb, open(os.path.join(a.out, "manifest.tsv"), "w") as fm:
        fb.write("\t".join(bcols) + "\n")
        fm.write("\t".join(mcols + xcols + (["seed_retry"] if a.single else [])) + "\n")
        if fu:
            fu.write("\t".join(bcols) + "\n")
        for r in good:
            m = json.load(open(r["mp4"][:-4] + ".json"))
            role = [o["role"] for o in m["objects"]]
            for w in m["rows"]:
                fb.write("\t".join(str(v) for v in [r["mp4"], w["end"], w["boundary"], w["ctx_start"], m["n_frames"], w["event"],
                                                    w["event_frame"], role[w["obj"]], "", w["dv"], r["name"], m["category"]]
                                   + ([w["pred_min"], w["ctx_gate"]] if a.single_v2 else [])) + "\n")
                nb += 1
            mo = m["objects"][0]
            fm.write("\t".join(str(v) for v in [m["category"], r["name"], r["k"], m["n_frames"], m["attempt"], len(m["objects"]),
                                                mo["shape"], mo["color"], mo["identity"], len(m["occluders"]), round(m["env"]["g"], 3),
                                                len(m["events"]), sum(e["kind"] == "occlude" for e in m["events"]), len(m["rows"]),
                                                r.get("sec", "")]
                               + [round(m["metrics"][c], 3) if isinstance(m["metrics"][c], float) else m["metrics"][c] for c in xcols]
                               + ([(m["seed"] - r["seed"]) // 7_919_007] if a.single else [])) + "\n")
            if fu:
                T = m["n_frames"]
                for b in uniform_rows(T, np.random.default_rng([a.seed_base, r["k"], names.index(r["name"]) if r["name"] in names else 0, 77]), a.stride):
                    fu.write("\t".join(str(v) for v in [r["mp4"], T, b, 0, T, "uniform", b, "main", "", 0.0, r["name"], m["category"]]
                                       + ([1, 0] if a.single_v2 else [])) + "\n")
                    nu += 1
    if fu:
        fu.close()
    print(f"{len(good):,}/{len(res):,} 통과 → {a.out}/index.csv · boundary 행 {nb:,} (영상당 {nb / max(1, len(good)):.1f})"
          + (f" · uniform 행 {nu:,} (영상당 {nu / max(1, len(good)):.1f})" if fu else ""))


def multi_names(flow=False, perm=False):
    import re
    files = ["gen_phys3d_multi_blender.py"] + (["gen_phys3d_flow_blender.py"] if flow or perm else []) + (["gen_phys3d_perm_blender.py"] if perm else [])
    return [n for f in files for n in re.findall(r'@scenario\("(\w+)"', open(os.path.join(HERE, f)).read())]


def main_multi(a):
    """phys3d_multi: 작업 (시나리오, k) — C* = C_SET[k mod 5] (생성기가 k 로 층화). 워커는 GPU 마다 gpu_per 개 (렌더 OptiX, 라벨은 CPU 1 코어) + CPU 워커."""
    import numpy as np
    script = "gen_phys3d_perm_blender.py" if a.perm else "gen_phys3d_flow_blender.py" if a.flow else "gen_phys3d_multi_blender.py"
    names = multi_names(a.flow, a.perm) if a.scenarios == "all" else a.scenarios.split(",")
    clips, wdir = os.path.join(a.out, "clips"), os.path.join(a.out, "_work")
    os.makedirs(wdir, exist_ok=True)
    if not a.collect_only:
        jobs = [(n, k) for k in range(a.per_scenario) for n in names]
        jf = os.path.join(wdir, "jobs.tsv")
        with open(jf, "w") as f:
            for n, k in jobs:
                f.write(f"{n}\t{k}\t{a.seed_base + k}\t0\n")
        claims = os.path.join(wdir, "claims")
        os.makedirs(claims, exist_ok=True)
        done = {json.loads(l)["job"].replace("/", "_") for f in glob.glob(os.path.join(wdir, "res_*.jsonl")) for l in open(f) if l.strip()}
        for c in os.listdir(claims):
            if c not in done:
                os.remove(os.path.join(claims, c))
        spec = ([] if a.defer_render else [("cycles", g) for g in a.gpus.split(",") if g for _ in range(a.gpu_per)]) + [("cpu", "")] * a.cpu_workers
        run = int(time.time())
        procs, cores = [], sorted(os.sched_getaffinity(0))
        rprocs = []
        if a.defer_render:                         # 렌더 워커 — GPU 에 오래 산다
            done_flag = os.path.join(wdir, "labels_done")
            if os.path.exists(done_flag):
                os.remove(done_flag)
            rclaims = os.path.join(wdir, "rclaims")
            if os.path.isdir(rclaims):            # 이어 하기: 끝나지 않은 렌더 표시는 지운다
                for c in os.listdir(rclaims):
                    os.remove(os.path.join(rclaims, c))
            for j, g in enumerate([g for g in a.gpus.split(",") if g for _ in range(a.render_per)]):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, OMP_NUM_THREADS="2", PYTHONHASHSEED="0")
                cmd = ["nice", "-n", str(a.nice), BLENDER, "-t", "2", "-b", "-noaudio", "-P", os.path.join(HERE, "render_queue_blender.py"), "--",
                       "--root", clips, "--claims", rclaims, "--done", done_flag, "--engine", "cycles"]
                rprocs.append(subprocess.Popen(cmd, stdout=open(os.path.join(wdir, f"rlog_{j:03d}.txt"), "a"), stderr=subprocess.STDOUT, env=env))
        for w, (eng, g) in enumerate(spec):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, OMP_NUM_THREADS="1", PYTHONHASHSEED="0")
            if a.defer_render:
                env["PHYS3D_DEFER_RENDER"] = "1"
            pin = ["taskset", "-c", str(cores[-1 - (w % len(cores))])] if a.pin else []
            cmd = pin + ["nice", "-n", str(a.nice), BLENDER, "-t", "1", "-b", "-noaudio", "-P", os.path.join(HERE, script), "--",
                         "--jobs", jf, "--claims", claims, "--results", os.path.join(wdir, f"res_{run}_{w:03d}.jsonl"), "--out", clips,
                         "--engine", eng]
            procs.append(subprocess.Popen(cmd, stdout=open(os.path.join(wdir, f"log_{w:03d}.txt"), "a"), stderr=subprocess.STDOUT, env=env))
        print(f"워커 {len(spec)} (GPU {sum(e == 'cycles' for e, _ in spec)} · CPU {a.cpu_workers}), 작업 {len(jobs):,} (남은 {len(jobs) - len(done):,})", flush=True)
        t0, total, n0 = time.time(), len(jobs), len(done)
        while any(p.poll() is None for p in procs):
            time.sleep(60)
            n = sum(sum(1 for _ in open(f)) for f in glob.glob(os.path.join(wdir, "res_*.jsonl")))
            el = time.time() - t0
            print(f"[{n:,}/{total:,}] {el / 60:.1f} 분" + (f", 남은 ~{el / max(n - n0, 1) * (total - n) / 60:.0f} 분" if n > n0 else ""), flush=True)
        bad = [w for w, p in enumerate(procs) if p.returncode != 0]
        if bad:
            print(f"⚠️ 워커 {bad} 비정상 종료 — _work/log_*.txt 확인, 같은 명령으로 이어 하기", flush=True)
        if a.defer_render:
            open(done_flag, "w").close()
            while any(p.poll() is None for p in rprocs):
                time.sleep(30)
                left = len(glob.glob(os.path.join(clips, "*", "*_r.blend")))
                print(f"렌더 남은 {left:,}", flush=True)
    res = [json.loads(l) for f in sorted(glob.glob(os.path.join(wdir, "res_*.jsonl"))) for l in open(f) if l.strip()]
    res.sort(key=lambda r: (r["name"], r["k"]))
    for r in res:                                  # 렌더가 안 된 것 (blend 남음 · mp4 없음) 은 실패로 (라벨만 시험 PHYS3D_NORENDER=1 은 빼고)
        if r["ok"] and not os.path.exists(r["mp4"]) and os.environ.get("PHYS3D_NORENDER") != "1":
            r["ok"], r["why"] = False, ["render missing"]
    good = [r for r in res if r["ok"]]
    with open(os.path.join(a.out, "failed.tsv"), "w") as fh:
        fh.write("name\tk\tC_star\tanchor\twhy\n")
        fh.writelines(f"{r['name']}\t{r['k']}\t{r.get('C_star', '')}\t{r.get('anchor', '')}\t{'; '.join(r['why'])}\n" for r in res if not r["ok"])
    with open(os.path.join(a.out, "index.csv"), "w") as fh:
        fh.writelines(f"{r['mp4']} 0\n" for r in good)
    bcols = ["path", "n_frames", "boundary_frame", "ctx_start", "video_frames", "event", "event_frame", "obj_role", "other", "dv", "scenario",
             "category", "pred_min", "ctx_gate", "anchor", "C_star"]
    mcols = ["category", "name", "k", "n_frames", "attempt", "C_star", "anchor", "anchor_asked", "want_move", "n_objects", "n_arrivals", "main_shape",
             "occ_modes", "n_occ", "mid_occ", "bg_mode", "floor_kind", "wall_kind", "hdri", "g", "speed_scale", "n_events", "n_rows", "sec"]
    nb = nu = 0
    with open(os.path.join(a.out, "boundary.tsv"), "w") as fb, open(os.path.join(a.out, "uniform.tsv"), "w") as fu, \
            open(os.path.join(a.out, "manifest.tsv"), "w") as fm:
        fb.write("\t".join(bcols) + "\n")
        fu.write("\t".join(bcols) + "\n")
        fm.write("\t".join(mcols) + "\n")
        for r in good:
            m = json.load(open(r["mp4"][:-4] + ".json"))
            role = [o["role"] for o in m["objects"]]
            T = m["n_frames"]
            for w in m["rows"]:
                fb.write("\t".join(str(v) for v in [r["mp4"], w["end"], w["boundary"], w["ctx_start"], T, w["event"], w["event_frame"],
                                                    role[w["obj"]], "", w["dv"], r["name"], m["category"], w.get("pred_min", 1),
                                                    w.get("ctx_gate", 0), w.get("anchor", ""), m["C_star"]]) + "\n")
                nb += 1
            occ = m.get("occlusion") or []
            e = m["env"]
            fm.write("\t".join(str(v) for v in [m["category"], r["name"], r["k"], T, m["attempt"], m["C_star"], m["anchor"], m["anchor_asked"],
                                                int(m["want_move"]), len(m["objects"]), m["info"].get("n_arrivals", 0), m["objects"][0]["shape"],
                                                ",".join(o["mode"] for o in occ), len(occ), int(any(o.get("kind") == "mid_context" for o in occ)),
                                                e.get("bg_mode", ""), e.get("floor_kind", ""), e.get("wall_kind", ""), e.get("hdri", ""),
                                                round(e["g"], 3), round(e["speed_scale"], 3), len(m["events"]), len(m["rows"]), r.get("sec", "")]) + "\n")
            for b in uniform_rows(T, np.random.default_rng([a.seed_base, r["k"], names.index(r["name"]) if r["name"] in names else 0, 77]), 1):
                fu.write("\t".join(str(v) for v in [r["mp4"], T, b, 0, T, "uniform", b, "main", "", 0.0, r["name"], m["category"], 1, 0, "", m["C_star"]]) + "\n")
                nu += 1
    print(f"{len(good):,}/{len(res):,} 통과 → {a.out}/index.csv · boundary 행 {nb:,} (영상당 {nb / max(1, len(good)):.1f}) · uniform 행 {nu:,}")


if __name__ == "__main__":
    sys.exit(main())
