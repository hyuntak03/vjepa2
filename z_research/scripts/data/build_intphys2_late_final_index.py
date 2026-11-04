#!/usr/bin/env python3
"""IntPhys 2 late **final** (사용자 재라벨 labels_final.csv, 500 쌍) 채점 index — mp4 + 영상별 프레임 번호 32 개 (2026-11-02, vast).

main `cvpr/eval/intphys2_late/build.py --final labels_final.csv --index-only` (origin/main fc59160) 의 build_final 과 같은 행을 만든다.
다른 점은 대응표 하나: main 은 IntPhys2_labeling/labels.csv (file → SceneIndex · pair · 원본 경로) 를 읽었는데 이 머신에는 없어서,
Main/metadata.csv 에서 (SceneIndex = scene, type = "<pair>_Possible" / "<pair>_Impossible") 로 원본 mp4 를 찾는다 (500 쌍 모두 1:1, game_name · condition 일치 확인).
창: labels_final 의 context_frames 16 + future_frames 16 그대로. 가능 영상이 더 짧으면 그 번호를 가능 영상 마지막 장으로 (clamp_pos 열) — main 과 같다.
출력: data_csv/intphys2_late_final/index.csv (file = video_root 기준 상대 경로 "Videos/<hash>.mp4", frames = 32 개 공백 구분) + labels_final_snapshot.csv
채점: 레지스트리 intphys2_late_final (video_root = IntPhys2/Main, frames_index_column = frames). 시트 "IntPhys2 late occ" = condition permanence 120 쌍 (Camera Moving 68 · Fixed 52).
  /venv/vjepa/bin/python z_research/scripts/data/build_intphys2_late_final_index.py
"""
import argparse
import csv
import os
import shutil

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
N_CTX, N_FUT = 16, 16


def nframes(p):
    import decord
    return len(decord.VideoReader(p, num_threads=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="/workspace/world_model/intphys2_label_check/labels_final.csv")
    ap.add_argument("--main-dir", default="/workspace/dataset/benchmarks/IntPhys2/Main")
    ap.add_argument("--out", default=f"{REPO}/data_csv/intphys2_late_final")
    a = ap.parse_args()
    meta = {(r["SceneIndex"], r["type"]): r for r in csv.DictReader(open(f"{a.main_dir}/metadata.csv"))}
    rows = []
    for x in csv.DictReader(open(a.final)):
        rp, ri = meta[(x["scene"], f"{x['pair']}_Possible")], meta[(x["scene"], f"{x['pair']}_Impossible")]
        for r in (rp, ri):
            assert (r["game_name"], r["condition"]) == (x["game_name"], x["condition"]), x["impossible_file"]
        pos, imp = rp["file_name"], ri["file_name"]                 # "Videos/<hash>.mp4"
        fr = list(map(int, x["context_frames"].split())) + list(map(int, x["future_frames"].split()))
        n_pos, n_imp = nframes(f"{a.main_dir}/{pos}"), nframes(f"{a.main_dir}/{imp}")
        assert n_imp == int(x["n_frames"]), (x["impossible_file"], n_imp, x["n_frames"])
        assert len(fr) == N_CTX + N_FUT and max(fr) < n_imp, (x["impossible_file"], max(fr), n_imp)
        fr_pos = [min(f, n_pos - 1) for f in fr]
        base = {"block_id": f"s{x['scene']}", "pair_id": x["pair"], "block_type": x["condition"], "condition": x["condition"],
                "game_name": x["game_name"], "Difficulty": ri["Difficulty"], "Camera": ri["Camera"], "occluder": ri["occluder"],
                "d": int(x["violation_start"]), "d_sheet": x["sheet_violation_start"], "d_src": x["source"], "annotator": x["annotator"],
                "occluded": "0", "n": min(n_pos, n_imp), "clamp_pos": sum(f != g for f, g in zip(fr, fr_pos))}
        assert ri["Camera"] == x["camera"], x["impossible_file"]
        for plaus, path, ff in (("1", pos, fr_pos), ("0", imp, fr)):
            tag = "pos" if plaus == "1" else "imp"
            rows.append({"video_id": f"s{x['scene']}_p{x['pair']}_{tag}", "file": path, "variant": f"{tag}_{x['pair']}",
                         "plausible": plaus, **base, "frames": " ".join(map(str, ff))})
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    shutil.copy(a.final, f"{a.out}/labels_final_snapshot.csv")
    perm = [r for r in rows if r["condition"] == "permanence" and r["plausible"] == "0"]
    print(f"쌍 {len(rows) // 2} (permanence {len(perm)}: Moving {sum(r['Camera'] == 'Moving' for r in perm)} · Fixed {sum(r['Camera'] == 'Fixed' for r in perm)}) "
          f"· 가능 영상 끝 번호 조정 쌍 {sum(1 for r in rows if r['plausible'] == '1' and r['clamp_pos'])} -> {a.out}/index.csv")


if __name__ == "__main__":
    main()
