# 무효 — 첫 실행 (2026-10-10 16:10–16:24, e35 · seed 0)

realistic head 학습이 loss 2.6 에서 멈춰 전부 none 으로 답했다 (`realistic.json`). 같은 설정을 다시 돌린 run 들 (`../tubelet_probe/e35_s0_user/` · `e35_s0_realistic_only/`) 은
정상 (realistic ① 89.6) 이고 두 run 의 수치가 같다. primitive 는 이 실행의 `RESULTS.md` 와 재실행이 소수점까지 같다 → 같은 설정은 결정적으로 재현되는데 이 한 번만 무너졌다. 원인 미상 (일회성).
이 폴더에 있던 primitive.json 은 사용자 재실행이 16:29 에 덮어쓴 것이라 `e35_s0_user/` 로 옮겼다.
