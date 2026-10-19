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
    a = ap.parse_args()
    names = SCEN if a.scenarios == "all" else a.scenarios.split(",")
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
        for w, (eng, g) in enumerate(spec):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, OMP_NUM_THREADS="1")
            cmd = ["nice", "-n", str(a.nice), BLENDER, "-b", "-noaudio", "-P", os.path.join(HERE, "gen_phys3d_blender.py"), "--",
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
    mcols = ["category", "name", "k", "n_frames", "attempt", "n_objects", "main_shape", "main_color", "main_identity", "n_occluders", "g",
             "n_events", "n_occ_episodes", "n_rows", "sec"]
    nb = 0
    with open(os.path.join(a.out, "boundary.tsv"), "w") as fb, open(os.path.join(a.out, "manifest.tsv"), "w") as fm:
        fb.write("\t".join(bcols) + "\n")
        fm.write("\t".join(mcols) + "\n")
        for r in good:
            m = json.load(open(r["mp4"][:-4] + ".json"))
            role = [o["role"] for o in m["objects"]]
            for w in m["rows"]:
                fb.write("\t".join(str(v) for v in [r["mp4"], w["end"], w["boundary"], w["ctx_start"], m["n_frames"], w["event"],
                                                    w["event_frame"], role[w["obj"]], "", w["dv"], r["name"], m["category"]]) + "\n")
                nb += 1
            mo = m["objects"][0]
            fm.write("\t".join(str(v) for v in [m["category"], r["name"], r["k"], m["n_frames"], m["attempt"], len(m["objects"]),
                                                mo["shape"], mo["color"], mo["identity"], len(m["occluders"]), round(m["env"]["g"], 3),
                                                len(m["events"]), sum(e["kind"] == "occlude" for e in m["events"]), len(m["rows"]),
                                                r.get("sec", "")]) + "\n")
    print(f"{len(good):,}/{len(res):,} 통과 → {a.out}/index.csv · boundary 행 {nb:,} (영상당 {nb / max(1, len(good)):.1f})")


if __name__ == "__main__":
    sys.exit(main())
