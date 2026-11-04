#!/usr/bin/env python3
"""real_world_permanence (실제 영상 물체 영속성 2×2, 95 영상 × 4 = 380 clip) → 채점 index (mp4 + 영상별 프레임 번호 32 개) + 문맥 감사 (2026-11-02).

원본: /data/hyuntak/project/2026/2027_cvpr/testbed/real_bench/real_world_permanence (README 참고; Rubric Office 20 + MAGICONTROL 10 + Perception Test 50 + HOOT 2 + Physics-IQ 13).
창 = Rubric uniform 과 같다 (build_rubric_quads_frames.py): t = t_splice (숨은 구간 가운데, 이 프레임부터 뒤쪽 소스),
  문맥 = round(linspace(0, t − 1, 16)) · 미래 = round(linspace(t, n − 1, 16)). 짧은 영상 (Physics-IQ 굴러가기 24–32 프레임, HOOT) 은 번호가 겹칠 수 있다 (같은 프레임 반복).
  60 fps 중복 영상 없음 (index.json duplicated 정보 없음 → 7_ballbounce 도 이 세트에서는 그대로 읽는다; README: 네 영상 모두 같은 프레임 쌍).
변이 → 표준 이름 (Rubric · VOBench 와 같은 2×2): possible_present = pos_a · impossible_vanish = imp_ab (짝 A, 문맥 '있음', 사라짐) ·
  possible_absent = pos_b · impossible_appear = imp_ba (짝 B, 문맥 '없음', 나타남).
감사 (CLAUDE.md §7-4): 짝마다 문맥 16 장을 디코딩해 화소 최대차 · 같은 장 수. 미래 창에 '다시 보임' 프레임이 몇 장 들어가는가.
출력: data_csv/real_world_permanence_uniform/index.csv (file = 원본 폴더 기준 mp4 상대 경로, frames = 32 개 공백 구분) · audit.json
채점 (vast 채점기): 레지스트리 real_world_permanence_uniform (video_root = 원본 폴더, frames_index_column = frames). main 채점기는 data.frames_column=frames
  main 에서 옮김 (origin/main fc59160, 2026-11-02 vast): SRC 기본값만 이 머신 경로로 바꿨다.
  /venv/vjepa/bin/python z_research/scripts/data/build_real_world_permanence_index.py [--no-audit]
"""
import argparse
import csv
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = "/workspace/dataset/benchmarks/real_world_permanence"   # vast (2026-11-02). main 원본: /data/hyuntak/project/2026/2027_cvpr/testbed/real_bench/real_world_permanence
VAR = {"possible_present": ("pos_a", "1", "A"), "impossible_vanish": ("imp_ab", "0", "A"),
       "impossible_appear": ("imp_ba", "0", "B"), "possible_absent": ("pos_b", "1", "B")}
PAIRS = {"A": ("possible_present", "impossible_vanish"), "B": ("possible_absent", "impossible_appear")}
N = 16


def window(t, n):
    ctx = np.round(np.linspace(0, t - 1, N)).astype(int)
    fut = np.round(np.linspace(t, n - 1, N)).astype(int)
    return ctx, fut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--no-audit", action="store_true", help="문맥 디코딩 감사를 건너뛴다")
    a = ap.parse_args()
    J = json.load(open(f"{a.src}/index.json"))
    V = J["videos"]
    clip_by = {(c["video"], c["variant"]): c for c in csv.DictReader(open(f"{a.src}/index.csv"))}
    rows, audit = [], {}
    if not a.no_audit:
        import decord
    for bi, v in enumerate(V):
        vid = v["id"]; n = int(v["num_frames"]); t = int(v["t_splice"])
        rev = int(v["first_visible_after_reveal"])
        ctx, fut = window(t, n)
        idx = np.concatenate([ctx, fut])
        assert (np.diff(idx) >= 0).all() and idx[-1] < n and ctx[-1] < t <= fut[0], (vid, idx)
        c0 = clip_by[(vid, "possible_present")]
        name = vid.replace("/", "__")
        for k, (var, plaus, pair) in VAR.items():
            c = clip_by[(vid, k)]
            assert int(c["num_frames"]) == n and int(c["t_splice"]) == t, (vid, k)
            rows.append(dict(video_id=f"{name}__{var}", file=c["file"], file_name=c["file"][:-4], block_id=str(bi), source_block=vid,
                             variant=var, plausible=plaus, pair_id=pair, condition=c0["outcome"], motion=c0["outcome"],
                             has_occlusion="1", violation_type="permanence", game_name=v["source"], role=var, sym_k="0",
                             objset="real", violation="permanence", occlusion="late", k="0", background=v["source"],
                             obj_a="present", obj_b="absent", ctx_clean_samples="", samples="",
                             source=v["source"], outcome=c0["outcome"], hidden_s=c0["hidden_s"],
                             frames=" ".join(map(str, idx.tolist()))))
        au = dict(frames=idx.tolist(), t=t, n=n, reveal=rev, fut_reveal_frames=int((fut >= rev).sum()),
                  n_unique=int(len(np.unique(idx))))
        if not a.no_audit:
            fr = {k: decord.VideoReader(f"{a.src}/{clip_by[(vid, k)]['file']}", num_threads=1).get_batch(list(ctx)).asnumpy() for k in VAR}
            au["ctx_maxdiff"] = {p: int(np.abs(fr[x].astype(int) - fr[y].astype(int)).max()) for p, (x, y) in PAIRS.items()}
            au["ctx_identical_frames"] = {p: int(sum((fr[x][j] == fr[y][j]).all() for j in range(N))) for p, (x, y) in PAIRS.items()}
            au["ctx_meandiff"] = {p: float(np.abs(fr[x].astype(float) - fr[y].astype(float)).mean()) for p, (x, y) in PAIRS.items()}
        audit[vid] = au
        print(vid, t, n, au.get("ctx_maxdiff"), au.get("ctx_identical_frames"), au["fut_reveal_frames"], flush=True)
    od = f"{REPO}/data_csv/real_world_permanence_uniform"
    os.makedirs(od, exist_ok=True)
    with open(f"{od}/index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(audit, open(f"{od}/audit.json", "w"), indent=1)
    if not a.no_audit:
        md = np.array([[au["ctx_maxdiff"][p] for p in "AB"] for au in audit.values()])
        ident = np.array([[au["ctx_identical_frames"][p] for p in "AB"] for au in audit.values()])
        print(f"문맥 감사: 짝 {md.size} 개 중 화소 완전 일치 {(md == 0).sum()} · 최대차 중앙 {np.median(md):.0f} · 최대 {md.max()} · "
              f"16 장 모두 같은 짝 {(ident == N).sum()}")
    print(f"{len(rows)} 행 -> {od}/index.csv")


if __name__ == "__main__":
    main()
