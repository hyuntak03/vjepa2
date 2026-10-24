> 📥 **vast 브랜치에서 가져온 문서 (2026-10-24, `origin/vast` c05c259 의 `z_training/docs/PHYS3D_DATASET_2026-10-19.md`, 원문 그대로).**
> 본문의 run 이름 · `/workspace/...` 경로 · 시트 언급은 vast 노드 기준이다 — 이 노드 (vll5) 에 그 산출물이 없을 수 있고, 수치는 그쪽 산출물에서 계산된 것이다.

# phys3d_v1 — Blender 2.5D 물리 + 장마다 상태 라벨 (2026-10-19, ⚠️ 생성 중단 상태)

> **상태 (2026-10-19 12:00 UTC)**: train 30,800 작업 중 **18,063** 끝 (통과 18,039 · 실패 24), val 미시작.
> 사용자 지시 ("3D 멈추고 2D 먼저") 로 멈췄다. 같은 명령으로 이어 한다 (결과 jsonl 에 있는 작업은 건너뛴다).
> 편향 점검 · 창 인덱스 · 수치 정리는 생성이 끝난 뒤 이 문서에 붙인다.
>
> ⚠️ **2026-10-19 발견 — sphere 클립의 보임 라벨 (`vis_frac` · `bbox` · `hidden` · `seen` · `area`) 은 실행마다 다르다.** `sample_points` 의 `calc_loop_triangles` 가
> UV 구에서 다각형 순서를 프로세스마다 달리 내 표본 점이 바뀐다 (위치 · 속도 · 사건은 같다, 나머지 다섯 모양은 같다). 물체 1 개 판
> ([`PHYS3D_SINGLE_DATASET_2026-10-19.md`](PHYS3D_SINGLE_DATASET_2026-10-19.md)) 에서 고쳤고 이 판에는 반영하지 않았다 (이어 하기 중인 데이터와 섞이지 않게).

| | |
|---|---|
| 위치 | `/workspace/dataset/training_data/phys3d_v1/{train,val}` |
| 계획 | train 14 시나리오 × 2,200 = 30,800 · val × 80 = 1,120 (seed_base 1,000,000) |
| 생성기 | `z_training/data/gen_phys3d_blender.py` (Blender 4.2.3 안에서 돈다) · 드라이버 `z_training/data/gen_phys3d.py` · 실행 `z_training/data/run_gen_phys3d_v1.sh` |
| 미리보기 | `/workspace/dataset/training_data/phys3d_examples_v0` (예시 18 편) · 라벨 겹침 `z_training/tools/phys3d_overlay.py` |

## 장면 · 라벨

- 방 (바닥 · 뒷벽 · 옆벽 x = ±3.4) + 보이지 않는 유리벽 y = ±0.9 (2.5D) + 앞쪽 가림막 (y = −1.3, 충돌 없음). 카메라 (±0.6, −6, 1.6~2.4), lens 29.
- 물리: Bullet 강체, 12 fps · 장당 10 substep, 중력 클립마다 3.5~9.81 m/s² (seed 고정).
- 정체 48 = 모양 6 (sphere · cube · cylinder · cone · pyramid · torus) × 색 8 — 2D 와 같은 id 규약 (shape × 8 + color), 시나리오마다 층화.
- 라벨 키 · 단위는 phys2d_sup 과 같다 (pos px, vel px/장, bbox, present · vis_frac · inframe_frac · hidden · seen · impact · identity …) + `pos3d` · `vel3d` (m, m/장).
  보임은 표면 표본 200 점의 카메라 광선 (자기 자신에 먼저 맞아야 보임).
- 시나리오 14: 중력 6 (drop_bounce · projectile · toss_up · ramp_roll · ledge_fall · pendulum) · 영속 3 (occ_pass · moving_screen · drop_behind) ·
  고체성 5 (shelf_land · wall_bounce · collide · stack_topple · container). 비영속 시나리오에도 40 % 로 가림막 하나 더.
- 행: 2D 와 같은 규칙 (전환은 예측 구간, ctx_start = 앞선 충돌 dv ≥ 1.5 m/s · 받침 이탈). 클립마다 C16K16 사건 행이 하나는 있어야 통과.

## 렌더 — ⚠️ EEVEE 금지

- EEVEE (EGL) 는 `CUDA_VISIBLE_DEVICES` 를 무시해 워커 전부가 GPU 0 에 붙었다 (학습 GPU 메모리 96.9 / 97.9 GB). Cycles 만 쓴다.
- Cycles GPU (OptiX): GPU 2~7 에 장당 2 워커 (4 는 OOM), 클립당 39 s. Cycles CPU: 스레드 1, 4 샘플 + OIDN 디노이즈, 클립당 219 s (16 샘플과 눈으로 같다).
  생성분 18,063 중 GPU 11,800 · CPU 6,263.
- 재현: 라벨은 (코드, seed, Blender 4.2.3) 으로 정해진다. 픽셀은 GPU / CPU 렌더가 미세하게 다르고 어느 클립이 GPU 로 가는지는 실행마다 다르다 → 픽셀까지 같아야 하면 `RENDER_GPUS=` (CPU 만).

## 재현

```bash
cd /workspace/world_model/vjepa2
# Blender 4.2.3 LTS: https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz
apt install -y libsm6 libxi6 libxrender1 libxkbcommon0 libgl1
# 예시 한 편
/workspace/tools/blender-4.2.3-linux-x64/blender -b -noaudio -P z_training/data/gen_phys3d_blender.py -- --scenario collide --seed 3 --out /tmp/p3d
# 전체 (이어 하기 가능). 픽셀까지 재현하려면 RENDER_GPUS=
BLENDER=/workspace/tools/blender-4.2.3-linux-x64/blender OUT=/workspace/dataset/training_data/phys3d_v1 RENDER_GPUS=2,3,4,5,6,7 GPU_PER=2 CPU_WORKERS=200 \
  bash z_training/data/run_gen_phys3d_v1.sh
```
