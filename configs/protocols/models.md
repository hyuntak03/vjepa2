# 모델 레지스트리

`z_research/scripts/run.sh <프로토콜> <데이터셋> [모델]` 의 세 번째 인자.
`## 이름` 아래의 `key: value` 만 읽는다 (그 밖의 줄은 전부 설명으로 무시된다).
여기 값들은 최종 config 의 `model:` 블록으로 들어가고, **프로토콜 yaml 의 `model:` 이 이긴다.**

기본은 `vith` 다. IntPhys1 동일 프로토콜에서 ViT-H 88.89% vs ViT-L 64.44% (+24.4pt) —
**외부 벤치마크에서 먼저 정한 것**이라 자체 데이터로 뒤집지 않는다.
자체 데이터에는 ViT-L 이 이기는 축이 있다 (v8 shape violation 83.98 vs 73.83,
2D transit overall 75.39 vs 71.48, Jongseo physv3 92% vs 80%).
**"ViT-H 가 항상 낫다"는 거짓이니 그렇게 쓰지 말 것.** 사후 모델 선택을 안 한다는 원칙일 뿐이다.

predictor(embed 384 / depth 12 / heads 12 / mask token 10)는 H·L 이 완전히 같다.
encoder 만 hidden 1024→1280, depth 24→32 로 커진다.

---

## vith

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth

## vitl

arch_name: vit_large
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f/original/model.pth
