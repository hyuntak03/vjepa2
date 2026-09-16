# _superseded — v11 emergence 첫 판 (2026-09-14)

`fig_emergence.*`, `fig_exit_distance.*` 는 `../by_condition/fig_<timing>_k<k>.*` 로 대체됐다.

물러난 이유 (사용자 피드백 "물체가 가림막을 벗어났는지 잘 모르겠다, 시각화가 이상하다"):
- y 축이 "출구 모서리 기준 거리" 하나라 가림막이 공간의 어디에 있는지 안 보였다. 회색 띠는 시간 (진실이 가려진 슬롯) 을 칠해 공간 띠로 오해됐다
- p 를 clip 평균 한 줄로 그려, attention 이 퍼진 기본값 읽기와 물체 읽기가 섞였다
- 미래 8 튜블릿만 그려 early/mid 에서 문맥 안 가림막 통과가 안 보였다
- k=2·4 와 late/mid/early 일부만 담았다

새 판: 가림막을 위치 축의 띠로, 문맥+미래 16 튜블릿, p 는 clip 마다 점 (퍼진 attention 은 회색), 튜블릿별 out / not out / no object 막대, timing × k 12 장.
