# VOBench late permanence — AR 기본판 vs AR + v4 이어 학습, pixel decoder 그림 (2026-11-02)

열: 원본 (가능 영상) · AR 기본판 · AR + v4 이어 학습. 앞 16 장 = 문맥 (원본), 뒤 16 장 = 각 모델의 블록 AR 예측을 decoder 로 그린 것.
머리줄 correct / wrong = 이 스크립트의 쌍 판정 (AR 채점과 같은 정의). A = 사라짐 짝 (문맥에 물체), B = 나타남 짝. 둘 다 VOBench 로 학습한 적 없다.

## primitive/moving_A (15 쌍) — 맞힌 쌍: AR 15 · AR + v4 15
- `primitive/moving_A/flat_acc__v11_flat_acc_torus_green.gif` — AR correct · AR + v4 correct
- `primitive/moving_A/ramp_down__v11_ramp_down_capsule_purple.gif` — AR correct · AR + v4 correct
- `primitive/moving_A/flat_vel__v11_flat_vel_torus_green.gif` — AR correct · AR + v4 correct
- `primitive/moving_A/flat_dec__v11_flat_dec_cylinder_purple.gif` — AR correct · AR + v4 correct
- `primitive/moving_A/ramp_down__v11_ramp_down_sphere_orange.gif` — AR correct · AR + v4 correct
- `primitive/moving_A/ramp_up__v11_ramp_up_cone_blue.gif` — AR correct · AR + v4 correct
- `primitive/moving_A/flat_vel__v11_flat_vel_cylinder_cyan.gif` — AR correct · AR + v4 correct
- `primitive/moving_A/flat_acc__v11_flat_acc_cone_cyan.gif` — AR correct · AR + v4 correct
- `primitive/moving_A/flat_dec__v11_flat_dec_capsule_purple.gif` — AR correct · AR + v4 correct
- `primitive/moving_A/flat_vel__assets_flat_vel_sphere_red.gif` — AR correct · AR + v4 correct
- `primitive/moving_A/ramp_up__assets_ramp_up_cylinder_magenta.gif` — AR correct · AR + v4 correct
- `primitive/moving_A/flat_dec__assets_flat_dec_torus_orange.gif` — AR correct · AR + v4 correct
- `primitive/moving_A/ramp_up__assets_ramp_up_capsule_cyan.gif` — AR correct · AR + v4 correct
- `primitive/moving_A/flat_acc__assets_flat_acc_torus_purple.gif` — AR correct · AR + v4 correct
- `primitive/moving_A/ramp_down__assets_ramp_down_cube_red.gif` — AR correct · AR + v4 correct

## primitive/moving_B (15 쌍) — 맞힌 쌍: AR 13 · AR + v4 15
- `primitive/moving_B/flat_acc__v11_flat_acc_torus_green.gif` — AR correct · AR + v4 correct
- `primitive/moving_B/ramp_down__v11_ramp_down_capsule_purple.gif` — AR wrong · AR + v4 correct
- `primitive/moving_B/flat_vel__v11_flat_vel_torus_green.gif` — AR correct · AR + v4 correct
- `primitive/moving_B/flat_dec__v11_flat_dec_cylinder_purple.gif` — AR correct · AR + v4 correct
- `primitive/moving_B/ramp_down__v11_ramp_down_sphere_orange.gif` — AR correct · AR + v4 correct
- `primitive/moving_B/ramp_up__v11_ramp_up_cone_blue.gif` — AR correct · AR + v4 correct
- `primitive/moving_B/flat_vel__v11_flat_vel_cylinder_cyan.gif` — AR correct · AR + v4 correct
- `primitive/moving_B/flat_acc__v11_flat_acc_cone_cyan.gif` — AR correct · AR + v4 correct
- `primitive/moving_B/flat_dec__v11_flat_dec_capsule_purple.gif` — AR correct · AR + v4 correct
- `primitive/moving_B/flat_vel__assets_flat_vel_sphere_red.gif` — AR correct · AR + v4 correct
- `primitive/moving_B/ramp_up__assets_ramp_up_cylinder_magenta.gif` — AR wrong · AR + v4 correct
- `primitive/moving_B/flat_dec__assets_flat_dec_torus_orange.gif` — AR correct · AR + v4 correct
- `primitive/moving_B/ramp_up__assets_ramp_up_capsule_cyan.gif` — AR correct · AR + v4 correct
- `primitive/moving_B/flat_acc__assets_flat_acc_torus_purple.gif` — AR correct · AR + v4 correct
- `primitive/moving_B/ramp_down__assets_ramp_down_cube_red.gif` — AR correct · AR + v4 correct

## primitive/static_A (3 쌍) — 맞힌 쌍: AR 3 · AR + v4 3
- `primitive/static_A/flat_static__v11_flat_static_cone_red.gif` — AR correct · AR + v4 correct
- `primitive/static_A/flat_static__assets_flat_static_cylinder_cyan.gif` — AR correct · AR + v4 correct
- `primitive/static_A/flat_static__assets_flat_static_cube_blue.gif` — AR correct · AR + v4 correct

## primitive/static_B (3 쌍) — 맞힌 쌍: AR 3 · AR + v4 3
- `primitive/static_B/flat_static__v11_flat_static_cone_red.gif` — AR correct · AR + v4 correct
- `primitive/static_B/flat_static__assets_flat_static_cylinder_cyan.gif` — AR correct · AR + v4 correct
- `primitive/static_B/flat_static__assets_flat_static_cube_blue.gif` — AR correct · AR + v4 correct

## realistic/moving_A (15 쌍) — 맞힌 쌍: AR 15 · AR + v4 14
- `realistic/moving_A/flat_dec__v11_flat_dec_apple.gif` — AR correct · AR + v4 correct
- `realistic/moving_A/ramp_down__v11_ramp_down_scooter.gif` — AR correct · AR + v4 correct
- `realistic/moving_A/flat_vel__v11_flat_vel_tennis_ball.gif` — AR correct · AR + v4 correct
- `realistic/moving_A/flat_dec__v11_flat_dec_car.gif` — AR correct · AR + v4 correct
- `realistic/moving_A/ramp_up__v11_ramp_up_basketball.gif` — AR correct · AR + v4 correct
- `realistic/moving_A/flat_acc__v11_flat_acc_scooter.gif` — AR correct · AR + v4 correct
- `realistic/moving_A/flat_dec__v11_flat_dec_tennis_ball.gif` — AR correct · AR + v4 correct
- `realistic/moving_A/ramp_down__v11_ramp_down_car.gif` — AR correct · AR + v4 wrong
- `realistic/moving_A/flat_vel__assets_flat_vel_apple.gif` — AR correct · AR + v4 correct
- `realistic/moving_A/flat_vel__assets_flat_vel_tennis_ball.gif` — AR correct · AR + v4 correct
- `realistic/moving_A/flat_acc__assets_flat_acc_basketball.gif` — AR correct · AR + v4 correct
- `realistic/moving_A/ramp_up__assets_ramp_up_scooter.gif` — AR correct · AR + v4 correct
- `realistic/moving_A/flat_acc__assets_flat_acc_bus.gif` — AR correct · AR + v4 correct
- `realistic/moving_A/ramp_down__assets_ramp_down_apple.gif` — AR correct · AR + v4 correct
- `realistic/moving_A/ramp_up__assets_ramp_up_basketball.gif` — AR correct · AR + v4 correct

## realistic/moving_B (15 쌍) — 맞힌 쌍: AR 15 · AR + v4 15
- `realistic/moving_B/flat_dec__v11_flat_dec_apple.gif` — AR correct · AR + v4 correct
- `realistic/moving_B/ramp_down__v11_ramp_down_scooter.gif` — AR correct · AR + v4 correct
- `realistic/moving_B/flat_vel__v11_flat_vel_tennis_ball.gif` — AR correct · AR + v4 correct
- `realistic/moving_B/flat_dec__v11_flat_dec_car.gif` — AR correct · AR + v4 correct
- `realistic/moving_B/ramp_up__v11_ramp_up_basketball.gif` — AR correct · AR + v4 correct
- `realistic/moving_B/flat_acc__v11_flat_acc_scooter.gif` — AR correct · AR + v4 correct
- `realistic/moving_B/flat_dec__v11_flat_dec_tennis_ball.gif` — AR correct · AR + v4 correct
- `realistic/moving_B/ramp_down__v11_ramp_down_car.gif` — AR correct · AR + v4 correct
- `realistic/moving_B/flat_vel__assets_flat_vel_apple.gif` — AR correct · AR + v4 correct
- `realistic/moving_B/flat_vel__assets_flat_vel_tennis_ball.gif` — AR correct · AR + v4 correct
- `realistic/moving_B/flat_acc__assets_flat_acc_basketball.gif` — AR correct · AR + v4 correct
- `realistic/moving_B/ramp_up__assets_ramp_up_scooter.gif` — AR correct · AR + v4 correct
- `realistic/moving_B/flat_acc__assets_flat_acc_bus.gif` — AR correct · AR + v4 correct
- `realistic/moving_B/ramp_down__assets_ramp_down_apple.gif` — AR correct · AR + v4 correct
- `realistic/moving_B/ramp_up__assets_ramp_up_basketball.gif` — AR correct · AR + v4 correct

## realistic/static_A (3 쌍) — 맞힌 쌍: AR 3 · AR + v4 3
- `realistic/static_A/flat_static__v11_flat_static_basketball.gif` — AR correct · AR + v4 correct
- `realistic/static_A/flat_static__assets_flat_static_tennis_ball.gif` — AR correct · AR + v4 correct
- `realistic/static_A/flat_static__assets_flat_static_bus.gif` — AR correct · AR + v4 correct

## realistic/static_B (3 쌍) — 맞힌 쌍: AR 3 · AR + v4 3
- `realistic/static_B/flat_static__v11_flat_static_basketball.gif` — AR correct · AR + v4 correct
- `realistic/static_B/flat_static__assets_flat_static_tennis_ball.gif` — AR correct · AR + v4 correct
- `realistic/static_B/flat_static__assets_flat_static_bus.gif` — AR correct · AR + v4 correct

## 재현

```bash
CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_pixel_decoder/render_compare_vob_ar.py
```
