#!/usr/bin/env python3
"""시트 새 탭 "v3 축 전이 · 사라짐" 내용 · 서식 (2026-11-02, 사용자 요청). 입력 = axis_transfer_vanish.json.
출력 predictor_lab/results/v3_ablation/sheet_axis/{values.json, format.json} — Google Sheets 커넥터로 보낸다.
정답률 칸 = 채점 산출물에서 잰 값 (표마다 머리 칸에 출처 메모), 전이율 칸 = 같은 표의 정답률 칸을 참조하는 수식."""
import json, string
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = REPO / "predictor_lab/results/v3_ablation/sheet_axis"
J = json.load(open(D / "axis_transfer_vanish.json"))
SID, TAB = 900010, "v3 축 전이 · 사라짐"
C = list(string.ascii_uppercase)
NAVY = {"red": 0.11764706, "green": 0.21960784, "blue": 0.3882353}
GRAY = {"red": 0.9490196, "green": 0.9490196, "blue": 0.9490196}
WHITE = {"red": 1, "green": 1, "blue": 1}
DARK = {"red": 0.14901961, "green": 0.14901961, "blue": 0.14901961}
LINE = {"red": 0.7764706, "green": 0.7764706, "blue": 0.7764706}
BLUE = {"red": 0.16, "green": 0.47, "blue": 0.84}

rows = {}                        # 1-based row → list of cells from column B
def put(r, c, v):
    rows.setdefault(r, {})[c] = v
fmt, merges, notes, heads, groups, levelrows, nrows, bodies, diag, trcells = [], [], [], [], [], [], [], [], [], []
put(2, "B", "BlenderTrack v3 축 전이: 한 축 수준으로 학습한 예측기가 같은 축의 다른 수준에서도 사라짐을 맞히나")
put(3, "B", "정답률 (%) = v3 시험에서 같은 문맥의 (물체가 다시 나오는 미래, 물체가 가림막 뒤에서 사라진 미래) 짝 가운데 사라진 쪽 surprise (미래 8 블록 평균 latent L1) 가 더 큰 비율. 같으면 0.5. 우연 50.")
put(4, "B", "학습 = 그 축 수준의 v3 장면 1,600 개 (영상 3,200) 로 release 예측기를 이어 학습 · 2,000 step · 전역 batch 24 · seed 1 개. 시험 = v3 시험 1,719 장면의 사라짐 짝을 그 축 수준으로 나눔.")
put(5, "B", "전이율 (%) = (그 행 − 원래) ÷ (시험 수준으로 직접 학습한 행 − 원래), 움직이는 물체 기준. 직접 학습한 오름폭이 아래 최소 오름폭보다 작으면 비운다. 굵은 칸 = 같은 수준으로 학습하고 시험.")
put(6, "B", "전이율을 낼 최소 오름폭 (%p)")
put(6, "C", 5)
r = 8
for k, ax in enumerate(J["axes"], 1):
    L = ax["levels"]; nL = len(L); subs = ax["subsets"]
    put(r, "B", f"표 {k}. {ax['axis']}")
    heads.append(r)
    r += 1
    hr = r
    put(r, "B", "학습 데이터 (한 축 수준, 장면 1,600)")
    blocks = []
    col = 2                                                     # C
    for sub in subs:
        title = {"moving": "움직이는 물체 · 사라짐 정답률 (%)", "static": "정지 물체 · 사라짐 정답률 (%)", "all": "사라짐 정답률 (%), 시험 = 물체 움직임 수준"}[sub]
        put(r, C[col], title); merges.append((r, col, col + nL)); blocks.append((sub, col)); col += nL
    trcol = col
    put(r, C[trcol], "전이율 (%), 움직이는 물체" if subs[0] != "all" else "전이율 (%)"); merges.append((r, trcol, trcol + nL))
    last = trcol + nL - 1
    groups.append((hr, last))
    r += 1
    for sub, c0 in blocks:
        for j, l in enumerate(L):
            put(r, C[c0 + j], l)
    for j, l in enumerate(L):
        put(r, C[trcol + j], l)
    levelrows.append((r, last))
    r += 1
    put(r, "B", "시험 짝 수 (n)")
    for sub, c0 in blocks:
        for j, l in enumerate(L):
            put(r, C[c0 + j], ax["n"][f"{sub}|{l}"])
    nrows.append((r, last))
    r += 1
    first = r
    rowof = {}
    for row in ax["rows"]:
        rowof[row["arm"]] = r
        r += 1
    r = first
    trained = {l: [x for x in ax["rows"] if x["name"] == l + "만"][0]["arm"] for l in L}
    for row in ax["rows"]:
        put(r, "B", row["name"])
        for sub, c0 in blocks:
            for j, l in enumerate(L):
                v = row["acc"][f"{sub}|{l}"]
                put(r, C[c0 + j], "—" if v is None else v)
                if row["name"] == l + "만":
                    diag.append((r, c0 + j))
        if row["arm"] not in ("release", "copy"):
            mc0 = blocks[0][1]
            for j, l in enumerate(L):
                me = f"{C[mc0 + j]}{r}"; base = f"{C[mc0 + j]}{rowof['release']}"; own = f"{C[mc0 + j]}{rowof[trained[l]]}"
                put(r, C[trcol + j], f'=IF({own}-{base}<$C$6,"",ROUND(100*({me}-{base})/({own}-{base})))')
                trcells.append((r, trcol + j))
                if row["name"] == l + "만":
                    diag.append((r, trcol + j))
        r += 1
    bodies.append((first, r - 1, last))
    notes.append((hr, 1, "Source: predictor_lab/results/v3_ablation/sheet_axis/axis_transfer_vanish.json (axis_transfer_vanish.py 가 v3 시험 채점 산출물 te/v3test_* 에서 다시 계산), 2026-11-02, Claude 가 실행"))
    r += 1
maxr = r

# values: B2:K{maxr}
grid = []
for rr in range(2, maxr + 1):
    line = [rows.get(rr, {}).get(c, None) for c in C[1:11]]
    grid.append(line)
json.dump({"range": f"'{TAB}'!B2:K{maxr}", "formulas": grid}, open(D / "values.json", "w"), ensure_ascii=False)

def gr(r0, r1, c0, c1):  # 1-based rows inclusive, 0-based cols [c0, c1)
    return {"sheetId": SID, "startRowIndex": r0 - 1, "endRowIndex": r1, "startColumnIndex": c0, "endColumnIndex": c1}
def tf(**k):
    t = {"fontFamily": "Times New Roman", "fontSize": k.pop("size", 10)}; t.update(k); return t
def border(color, style="SOLID", width=1):
    return {"style": style, "width": width, "colorStyle": {"rgbColor": color}}
req = [{"addSheet": {"properties": {"sheetId": SID, "title": TAB, "index": 4, "gridProperties": {"rowCount": maxr + 10, "columnCount": 12, "hideGridlines": True}}}}]
req.append({"repeatCell": {"range": gr(1, maxr + 10, 0, 12), "cell": {"userEnteredFormat": {"textFormat": tf(), "verticalAlignment": "MIDDLE"}}, "fields": "userEnteredFormat.textFormat,userEnteredFormat.verticalAlignment"}})
req.append({"repeatCell": {"range": gr(2, 2, 1, 2), "cell": {"userEnteredFormat": {"textFormat": tf(bold=True, size=14, foregroundColorStyle={"rgbColor": NAVY})}}, "fields": "userEnteredFormat.textFormat"}})
req.append({"repeatCell": {"range": gr(3, 5, 1, 2), "cell": {"userEnteredFormat": {"textFormat": tf(foregroundColorStyle={"rgbColor": {"red": 0.247, "green": 0.247, "blue": 0.247}})}}, "fields": "userEnteredFormat.textFormat"}})
req.append({"repeatCell": {"range": gr(6, 6, 1, 2), "cell": {"userEnteredFormat": {"horizontalAlignment": "RIGHT"}}, "fields": "userEnteredFormat.horizontalAlignment"}})
req.append({"repeatCell": {"range": gr(6, 6, 2, 3), "cell": {"userEnteredFormat": {"backgroundColorStyle": {"rgbColor": {"red": 1, "green": 0.9490196, "blue": 0.8}}, "horizontalAlignment": "CENTER",
            "textFormat": tf(bold=True, foregroundColorStyle={"rgbColor": {"red": 0, "green": 0, "blue": 1}}), "borders": {"bottom": border({"red": 0.486, "green": 0.486, "blue": 0.486})}}},
            "fields": "userEnteredFormat.backgroundColorStyle,userEnteredFormat.horizontalAlignment,userEnteredFormat.textFormat,userEnteredFormat.borders"}})
notes.append((6, 2, "입력값 (사용자가 바꿀 수 있음). 직접 학습한 오름폭이 이보다 작으면 전이율은 잡음이 커서 비운다. Source: Claude 가 정한 기본값, 2026-11-02"))
for h in heads:
    req.append({"repeatCell": {"range": gr(h, h, 1, 2), "cell": {"userEnteredFormat": {"textFormat": tf(bold=True, size=11, foregroundColorStyle={"rgbColor": NAVY})}}, "fields": "userEnteredFormat.textFormat"}})
for (hr, last), (lr, _), (nr, _), (b0, b1, _) in zip(groups, levelrows, nrows, bodies):
    req.append({"repeatCell": {"range": gr(hr, lr, 1, last + 1), "cell": {"userEnteredFormat": {"backgroundColorStyle": {"rgbColor": NAVY}, "horizontalAlignment": "CENTER", "wrapStrategy": "WRAP",
               "textFormat": tf(bold=True, foregroundColorStyle={"rgbColor": WHITE})}}, "fields": "userEnteredFormat.backgroundColorStyle,userEnteredFormat.horizontalAlignment,userEnteredFormat.wrapStrategy,userEnteredFormat.textFormat"}})
    req.append({"mergeCells": {"range": gr(hr, lr, 1, 2), "mergeType": "MERGE_ALL"}})
    req.append({"repeatCell": {"range": gr(nr, nr, 1, last + 1), "cell": {"userEnteredFormat": {"backgroundColorStyle": {"rgbColor": GRAY}, "horizontalAlignment": "CENTER", "textFormat": tf(italic=True)}},
               "fields": "userEnteredFormat.backgroundColorStyle,userEnteredFormat.horizontalAlignment,userEnteredFormat.textFormat"}})
    req.append({"repeatCell": {"range": gr(nr, nr, 1, 2), "cell": {"userEnteredFormat": {"horizontalAlignment": "LEFT"}}, "fields": "userEnteredFormat.horizontalAlignment"}})
    req.append({"repeatCell": {"range": gr(b0, b1, 2, last + 1), "cell": {"userEnteredFormat": {"horizontalAlignment": "CENTER", "numberFormat": {"type": "NUMBER", "pattern": "0.0"}}},
               "fields": "userEnteredFormat.horizontalAlignment,userEnteredFormat.numberFormat"}})
    req.append({"updateBorders": {"range": gr(hr, b1, 1, last + 1), "top": border(DARK, "SOLID_MEDIUM", 2), "bottom": border(DARK, "SOLID_MEDIUM", 2), "left": border(DARK), "right": border(DARK),
               "innerHorizontal": border(LINE), "innerVertical": border(LINE)}})
    # 원래 · 복사 행 (회색 글씨) 와 학습 팔 사이 구분선
    req.append({"repeatCell": {"range": gr(b1 - 2, b1 - 1, 1, last + 1), "cell": {"userEnteredFormat": {"textFormat": tf(foregroundColorStyle={"rgbColor": {"red": 0.4, "green": 0.4, "blue": 0.4}})}}, "fields": "userEnteredFormat.textFormat"}})
for (hr, c0, c1) in merges:
    pass
for (hr, c0, c1) in merges:
    req.append({"mergeCells": {"range": gr(hr, hr, c0, c1), "mergeType": "MERGE_ALL"}})
for (r_, c_) in diag:
    req.append({"repeatCell": {"range": gr(r_, r_, c_, c_ + 1), "cell": {"userEnteredFormat": {"textFormat": tf(bold=True)}}, "fields": "userEnteredFormat.textFormat"}})
# 전이율 칸 색 띠 0 (흰색) ~ 100 (파랑) — trianing_ablation 표 3 · 4 와 같은 약속
tr_ranges = [gr(b0, b1, gl - len_ + 1 if False else 0, 0) for (b0, b1, gl), len_ in []]
trr = {}
for (r_, c_) in trcells:
    trr.setdefault(c_, []).append(r_)
rngs = []
for (b0, b1, last), ax in zip(bodies, J["axes"]):
    nL = len(ax["levels"])
    rngs.append(gr(b0, b1, last + 1 - nL, last + 1))
req.append({"addConditionalFormatRule": {"rule": {"ranges": rngs, "gradientRule": {
    "minpoint": {"colorStyle": {"rgbColor": WHITE}, "type": "NUMBER", "value": "0"},
    "maxpoint": {"colorStyle": {"rgbColor": {"red": 0.42, "green": 0.62, "blue": 0.89}}, "type": "NUMBER", "value": "100"}}}, "index": 0}})
for (r_, c_, text) in notes:
    req.append({"updateCells": {"rows": [{"values": [{"note": text}]}], "fields": "note", "start": {"sheetId": SID, "rowIndex": r_ - 1, "columnIndex": c_}}})
req.append({"updateDimensionProperties": {"range": {"sheetId": SID, "dimension": "COLUMNS", "startIndex": 0, "endIndex": 1}, "properties": {"pixelSize": 20}, "fields": "pixelSize"}})
req.append({"updateDimensionProperties": {"range": {"sheetId": SID, "dimension": "COLUMNS", "startIndex": 1, "endIndex": 2}, "properties": {"pixelSize": 290}, "fields": "pixelSize"}})
req.append({"updateDimensionProperties": {"range": {"sheetId": SID, "dimension": "COLUMNS", "startIndex": 2, "endIndex": 11}, "properties": {"pixelSize": 92}, "fields": "pixelSize"}})
json.dump({"requests": req}, open(D / "format.json", "w"), ensure_ascii=False)
print("rows", maxr, "requests", len(req), "diag", len(diag), "tr", len(trcells))
