"""plan_pusht.py 가 남긴 재생 기록 (rollout.npz) 으로 GIF 를 다시 만든다 — planning · GPU 없이 시뮬레이터만 돈다 (2026-10-12).

    python z_research/action_planning/render_plan_gifs.py --prefix z_research/action_planning/exp_results/<tag 접두어 (…_shard 앞까지)> \\
        --goals 0,1,2 --out z_research/action_planning/figures/gif_YYYY-MM-DD

--prefix 는 조각 폴더들 (<prefix>_shard*of*) 또는 폴더 하나 (<prefix>) 를 찾는다. --goals 는 전체 목표 번호 (eval_index), 생략하면 전부.
출력 = <out>/gif/<run>_goalNN_{SUCCESS,FAIL}.gif (plan_pusht.py --gif 와 같은 모양 → pusht_compare_gif.py 로 팔끼리 나란히).
재생한 최종 상태가 기록된 e_states 와 다르면 죽는다 (시뮬레이터 재현성 검사).
"""
import argparse, glob, json, os, sys, types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan_pusht as PP  # noqa: E402  (DINO-WM env · save_gifs · make_env 를 그대로 쓴다)

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", required=True)
ap.add_argument("--goals", default=None, help="쉼표로 전체 목표 번호. 생략 = 기록된 목표 전부")
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=float, default=15)
a = ap.parse_args()

dirs = sorted(glob.glob(f"{a.prefix}_shard*of*")) or [a.prefix]
want = None if a.goals is None else {int(x) for x in a.goals.split(",")}
out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
n_done = 0
for d in dirs:
    f = Path(d) / "rollout.npz"
    if not f.is_file():
        sys.exit(f"재생 기록이 없다: {f} (2026-10-12 이전 평가는 저장하지 않았다 — plan_pusht.py --only ... --gif 로 다시 planning)")
    R = np.load(f)
    run = json.load(open(Path(d) / "summary.json"))["run"]
    idx = R["eval_index"].tolist()
    keep = [j for j, k in enumerate(idx) if want is None or k in want]
    if not keep:
        continue
    kind, fs = str(R["env_kind"]), int(R["frameskip"])
    seeds = R["seeds"][keep].tolist()
    env = PP.make_env(kind, len(keep))
    if kind == "wall":
        cfg = json.load(open(Path(d) / "summary.json"))
        import yaml
        rc = yaml.safe_load(open(PP.ROOT / "z_training/runs" / run / "config.yaml"))
        root = next(s["root"] for s in rc["data"]["datasets"] if s.get("type") == "wall")
        door = torch.load(os.path.join(root, "door_locations.pth"))[0, 0, 0]
        wallx = torch.load(os.path.join(root, "wall_locations.pth"))[0, 0, 0]
        env.update_env([{"fix_door_location": door, "fix_wall_location": wallx}] * len(keep))
    e_obses, e_states = env.rollout(seeds, R["state_0"][keep], R["exec_actions"][keep])
    err = float(np.abs(np.asarray(e_states, np.float32) - R["e_states"][keep]).max())
    assert err < 1e-3, f"{d}: 재생 상태가 기록과 다르다 (max |Δ| {err:.3g})"
    obs_g, _ = env.prepare(seeds, R["state_g"][keep])
    obs_g = {k: np.asarray(v)[:, None] for k, v in obs_g.items()}
    cwd = os.getcwd(); os.chdir(out)
    PP.save_gifs(types.SimpleNamespace(run=run), [idx[j] for j in keep], e_obses, obs_g, R["successes"][keep].astype(bool),
                 R["action_len"][keep], fs, torch.from_numpy(R["actions_normalized"][keep]), fps=a.fps)
    os.chdir(cwd)
    n_done += len(keep)
    print(f"{d}: {len(keep)} goals replayed (max |Δstate| {err:.2g})")
print(f"{n_done} GIFs -> {out}/gif")
