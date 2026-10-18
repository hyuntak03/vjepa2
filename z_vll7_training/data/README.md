# z_vll7_training/data — vll7 데이터 인덱스 빌더

| 스크립트 | 무엇 |
|---|---|
| `build_vll7_index.py` | 노드 로컬 mp4 폴더 → `"<mp4> <label>"` csv (`data_csv/vll7_*/all.csv`) |
| `derive_fixed_csv.py` | fps4 표 → **고정 창**용 min-frames csv (`--mode fps4` = fps 4 로 N 장 / `--mode raw` = 원본 N 장) |
| `var_window_report.py` | **가변 창의 실제 샘플링을 돌리기 전에 표로** — (C,K) 칸마다 적격 편수 · batch · 실제 step/clip/frame 비율 |

fps4 표 자체는 공용 `z_training/data/build_fps4_table.py` 가 만든다 (vast 이식).
전체 명령은 `configs/training/datasets.md` 의 vll7 절 또는 `z_vll7_training/README.md` §재현.
