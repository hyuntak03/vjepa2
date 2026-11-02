#!/usr/bin/env python3
"""cells.json (table_v3.py 출력) → 시트 trianing_ablation 에 쓸 행 블록 (행마다 표 1 H–Y · 표 2 H–Z). 2026-11-01.
  $P predictor_lab/v3_ablation/sheet_rows.py [tag …]   (기본: cells.json 에 있는 모든 행)
출력: JSON [{"range": "trianing_ablation!H15:Y15", "values": [[…]]}, …]  — Google Sheets 커넥터로 그대로 쓴다.
"""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import table_v3 as T
cells = json.load(open(REPO / "predictor_lab/results/v3_ablation/cells.json"))
tags = sys.argv[1:] or list(T.ROW)
out = []
for t in tags:
    r = T.ROW[t]
    for r0, ncol in ((r, len(T.T1_COLS)), (r + 42, len(T.T2_COLS))):
        vals = [cells.get(f"{T.col(7 + i)}{r0}") for i in range(ncol)]
        if any(v is not None for v in vals):
            out.append({"range": f"trianing_ablation!H{r0}:{T.col(7 + ncol - 1)}{r0}", "values": [[v if v is not None else "" for v in vals]]})
print(json.dumps(out, ensure_ascii=False))
