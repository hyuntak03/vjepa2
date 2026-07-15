"""Build a per-video keystone-tick lookup for IntPhys 1 dev.

Reads /local_datasets/world/IntPhys1_dev_videos/keystones.csv and recovers the
matched-pair mapping from /local_datasets/world/dev via md5(scene_001.png).

For each of 360 videos, output {video_path: tick} where:
  * impossible video with 1 magic_tick        -> that tick
  * impossible video with 2 magic_ticks       -> midpoint (covers both in a 16f window)
  * possible video (matched to imp with 1)    -> matched imp's tick
  * possible video (matched to imp with 2)    -> matched imp's midpoint

Result is saved to data_csv/IntPhys1/keystones.json for consumption by the
video dataset.

Run: python analysis/intphys1/build_keystones.py
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

KEYSTONES_CSV = "/local_datasets/world/IntPhys1_dev_videos/keystones.csv"
DEV_ROOT = "/local_datasets/world/dev"
VIDEOS_ROOT = "/local_datasets/world/IntPhys1_dev_videos/scene"
OUT_JSON = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/IntPhys1/keystones.json"


def md5_first_frame(block: str, quad: str, run: int) -> str:
    """MD5 of /dev/{block}/{quad}/{run}/scene/scene_001.png -- pre-breakpoint frame is
    byte-identical between two runs that share a matched pair."""
    p = Path(DEV_ROOT) / block / quad / str(run) / "scene" / "scene_001.png"
    return hashlib.md5(p.read_bytes()).hexdigest()


def parse_tick(magic_ticks_str: str) -> int | None:
    """CSV stores '47' or '46 56' -- return the FIRST tick as int, or midpoint for 2-tick."""
    s = str(magic_ticks_str).strip()
    if not s or s.lower() == "nan":
        return None
    parts = s.split()
    if len(parts) == 1:
        return int(parts[0])
    # 2-tick case: midpoint. 16f window centered here covers both breakpoints if <= 8f apart.
    return sum(int(p) for p in parts) // len(parts)


def main():
    df = pd.read_csv(KEYSTONES_CSV, dtype={"quadruplet": str})
    print(f"loaded {len(df)} rows from {KEYSTONES_CSV}")

    # Group by (block, quadruplet), recover matched pairs via md5(scene_001.png)
    ticks_by_scene_id: dict[str, int] = {}
    for (block, quad), grp in df.groupby(["block", "quadruplet"]):
        assert len(grp) == 4, f"{block}/{quad}: expected 4 runs, got {len(grp)}"
        runs = grp["run"].tolist()

        # md5 -> [runs] : two groups of size 2 = the two matched pairs
        md5_groups: dict[str, list[int]] = {}
        for r in runs:
            h = md5_first_frame(block, quad, r)
            md5_groups.setdefault(h, []).append(r)
        assert len(md5_groups) == 2 and all(len(g) == 2 for g in md5_groups.values()), (
            f"{block}/{quad}: md5 grouping did not yield two pairs of size 2: {md5_groups}"
        )

        # For each pair, find the impossible run's tick and share with the possible run.
        for h, pair_runs in md5_groups.items():
            imp_run = None
            imp_tick = None
            poss_run = None
            for r in pair_runs:
                row = grp[grp["run"] == r].iloc[0]
                if row["is_possible"] == 0:  # impossible
                    imp_run = r
                    imp_tick = parse_tick(row["magic_ticks"])
                else:
                    poss_run = r
            assert imp_run is not None and imp_tick is not None and poss_run is not None, (
                f"{block}/{quad}: pair {pair_runs} has no impossible with tick"
            )
            for r in pair_runs:
                scene_id = f"{block}_{quad}_{r}"
                ticks_by_scene_id[scene_id] = imp_tick

    print(f"resolved {len(ticks_by_scene_id)} scene ticks (expected 360)")
    assert len(ticks_by_scene_id) == 360

    # Map scene_id -> mp4 path
    ticks_by_path: dict[str, int] = {
        f"{VIDEOS_ROOT}/{sid}.mp4": t for sid, t in ticks_by_scene_id.items()
    }

    tick_hist = pd.Series(list(ticks_by_scene_id.values())).describe()
    print(f"\ntick distribution:\n{tick_hist}")
    print(f"\nsample entries:")
    for k in list(ticks_by_path.keys())[:5]:
        print(f"  {k} -> tick {ticks_by_path[k]}")

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(ticks_by_path, f, indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
