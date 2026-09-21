# 물러난 config (2026-09-21)

**IntPhys 2 프로토콜은 현재 작업 범위 밖이다.** 지금 진행 중인 것은 Garrido 프로토콜
(IntPhys 1 · GRASP · InfLevel) 뿐이고, 그 진입점은 `z_research/Benchmarks/run_all.sh` 다.

여기 있는 것은 실험 하나마다 떠 둔 일회성 config 다. 지우지 않고 내려 둔다.

⚠️ **`vjepa2_vith_intphys1_ip2proto.yaml`** — IntPhys 1 영상을 IntPhys 2 프로토콜로 잰 것.
실측 결과 w16 67.22 / w32 66.11 로, IntPhys 2 논문이 V-JEPA 2-h 에 보고한 **87.22 와 20pt 벌어진다.**
원인 미규명. IntPhys 1 은 **Garrido 프로토콜로만** 보고한다 (A.8 격자 최고 = skip2_w32 88.89).

새 실험이 필요하면 `../intphys2_TEMPLATE.yaml` 에서 시작한다.
