# 물러난 그림

| 폴더 | 무엇 | 왜 물러났나 |
|---|---|---|
| `v1_attn_per_block/late_k{1,2,3}_block_probe.{png,pdf}` | VOBench_v1 · 블록마다 따로 학습한 attentive (100 epoch · lr 3e-4, 2 겹, head 시드 3) | 학습 영상이 겹당 90 개라 판독기가 외운다 (학습 정확도 100 %). 보이는 문맥 블록에서도 색 35–77 % 로 천장에 못 닿고, 같은 보이는 블록이 k 마다 20 pt 갈린다 (2026-10-30). 지금 판은 `../v2_attn/` (VOBench_v2 · 구간 공유) 이다. 값은 `../../results/v1/late_block_probe_{shape,color,exist}.json` |

- 2026-10-31 폴더 정리 때 원래의 `_superseded/` 폴더 (그림 · README) 가 없어져 있었다 (이 작업에서 지운 것은 아님). 결과 json 에서 같은 스크립트로 다시 그렸다.
