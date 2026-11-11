"""real world permanence 장면별 카메라 움직임 측정 (2026-11-07, 사용자 "가림막 종류 / 카메라 무빙 이런 거 구분"). 학습 없음 · GPU 없음.
대상을 지운 영상 (possible_absent.mp4) 에서 이웃 프레임 사이의 화면 전체 평행 이동을 위상 상관 (FFT) 으로 잰다 (긴 변 256 px 로 줄인 회색조).
  shift_med · shift_p90 : 프레임당 이동 (px, 긴 변 256 기준) 중앙값 · 90 % — 흔들림 크기
  path · net             : 클립 전체 누적 이동 거리 · 처음→끝 순 이동 (px) — 이동 (팬) 여부
물체 · 손 · 가림막이 움직여도 배경이 화면 대부분이면 배경 이동이 잡힌다 (손이 화면을 크게 가리는 장면은 과대 추정될 수 있다).
  /venv/vjepa/bin/python z_research/scripts/analysis/real_bench_camera_motion.py
"""
import csv, json, os, sys
import numpy as np

SRC = "/workspace/dataset/benchmarks/real_world_permanence"
OUT = os.path.join(os.path.dirname(__file__), "../../../z_analysis/results/real_errors_20261107/camera_motion.csv")


def shift(a, b):
    F = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    r = np.fft.ifft2(F / (np.abs(F) + 1e-6)).real
    y, x = np.unravel_index(np.argmax(r), r.shape)
    H, W = r.shape
    if y > H // 2: y -= H
    if x > W // 2: x -= W
    return float(x), float(y)


def main():
    import decord
    J = json.load(open(f"{SRC}/index.json"))
    rows = []
    for v in J["videos"]:
        vr = decord.VideoReader(f"{SRC}/{v['id']}/possible_absent.mp4", num_threads=4)
        n = len(vr); step = max(1, int(round(float(v.get("fps", 30)) / 15)))      # 약 15 fps 로 훑는다
        idx = list(range(0, n, step))
        fr = vr.get_batch(idx).asnumpy().astype(np.float32).mean(-1)               # (T, H, W) 회색
        H, W = fr.shape[1:]; s = 256 / max(H, W)
        ys = (np.arange(int(H * s)) / s).astype(int); xs = (np.arange(int(W * s)) / s).astype(int)
        g = fr[:, ys][:, :, xs]
        g = g - g.mean((1, 2), keepdims=True)
        win = np.outer(np.hanning(g.shape[1]), np.hanning(g.shape[2]))
        d = np.array([shift(g[i + 1] * win, g[i] * win) for i in range(len(g) - 1)])
        mag = np.hypot(d[:, 0], d[:, 1])
        cum = d.cumsum(0)
        rows.append(dict(scene=v["id"], source=v["source"], fps=v.get("fps"), step=step, n_steps=len(d),
                         shift_med=round(float(np.median(mag)), 3), shift_p90=round(float(np.percentile(mag, 90)), 3),
                         path=round(float(mag.sum()), 1), net=round(float(np.hypot(*cum[-1])), 1)))
        print(rows[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("저장", OUT)


if __name__ == "__main__":
    main()
