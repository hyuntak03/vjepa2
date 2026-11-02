#!/usr/bin/env python3
"""시트 trianing_ablation 을 통째로 짓는다 (2026-11-01) — 내용 · 수식 · 값 (table.json) · 서식 · 조건부 서식.

11-01 사용자: v3t 를 따로 두지 말고 v3 하나로. trapdoor 는 가림 종류의 세 번째 수준. v3 시험 = v3 (3,438 짝) + trapdoor (892 짝).
출력 (Google Sheets 커넥터로 차례로 보낸다) — predictor_lab/results/v3_ablation/sheet/
  00_clear.json      : 서식 · 병합 · 조건부 서식 지우기 (batchUpdate requests)
  10_values_*.json   : {"range": "trianing_ablation!B2:AA60", "rows": [[…]]}  (update_formulas)
  20_format_*.json   : batchUpdate requests (서식 · 병합 · 테두리)
  30_cond.json       : 조건부 서식
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P predictor_lab/v3_ablation/build_sheet.py [--n-cond 17]
"""
import argparse
import json
import string
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTD = REPO / "predictor_lab/results/v3_ablation/sheet"
SID = 202780267
TAB = "trianing_ablation"
COLS = list(string.ascii_uppercase) + ["AA", "AB"]                                  # A … AB (목록이어야 AA 가 한 칸이다)


def ci(c): return COLS.index(c)
def col(i): return COLS[i]
def shift(c, k): return COLS[ci(c) + k]


# ============================================================ 팔 (표 1 의 행 순서)
SEC = "sec"
ARMS = [
    (SEC, "기준"),
    ("copy", "복사 기준선", "-", "예측 없음: 마지막 문맥 latent 를 미래 자리에 그대로 둠", "-", "-", "-"),
    ("release", "release (PFT 없음)", "-", "-", "-", 0, 0),
    ("vobftg", "VOBench PFT", "학습셋", "VOBench_v1 학습 절반 (학습 설정이 달라 참고용)", "release", "-", "-"),
    (SEC, "같은 크기 대조 (v3 모든 칸 섞음)"),
    ("v3_mix1600_s0", "v3 섞음 1,600", "없음", "v3 train 전체에서 무작위 1,600 장면 (trapdoor 없음)", "release", 1600, 3200),
    ("v3_mix1600_t25_s0", "v3 섞음 1,600 + trapdoor 25 %", "trapdoor 비율", "v3 1,200 장면 + trapdoor 400 장면", "v3 섞음 1,600", 1600, 3200),
    ("v3_mix680_s0", "v3 섞음 680", "없음", "v3 train 전체에서 무작위 680 장면", "release", 680, 1360),
    ("v3_full_s0", "v3 전부 (trapdoor 전)", "양", "trapdoor 를 넣기 전 v3 train 전체", "v3 섞음 1,600", 7136, 14272),
    (SEC, "물체 모양"),
    ("v3_prim_s0", "v3 primitive 만", "물체 모양", "기본 도형 물체 장면만", "v3 섞음 1,600", 1600, 3200),
    ("v3_real_s0", "v3 realistic 만", "물체 모양", "실물 모양 물체 (공 · 상자 · 통 등) 장면만", "v3 섞음 1,600", 1600, 3200),
    (SEC, "카메라"),
    ("v3_front_s0", "v3 정면만", "카메라", "정면, 눈높이 카메라 장면만", "v3 섞음 1,600", 1600, 3200),
    ("v3_side_s0", "v3 측면만", "카메라", "비스듬한 측면 (방위 30–60°) 장면만", "v3 섞음 1,600", 1600, 3200),
    ("v3_bird_s0", "v3 위에서만", "카메라", "내려다보는 카메라 (bird eye) 장면만", "v3 섞음 1,600", 1600, 3200),
    (SEC, "가림 종류"),
    ("v3_barrier_s0", "v3 가림막만", "가림 종류", "판 · 벽 · 탁자 · 레일 판이 가리는 장면만", "v3 섞음 1,600", 1600, 3200),
    ("v3_object_s0", "v3 물체 가림만", "가림 종류", "다른 물체 (서 있는 물체 · 굴러오는 공) 가 가리는 장면만", "v3 섞음 1,600", 1600, 3200),
    ("v3t_only_s0", "v3 trapdoor 만", "가림 종류", "바닥에 누운 판이 일어서서 가렸다가 다시 눕는 장면만 (VOBench late 와 같은 꼴)", "v3 섞음 1,600", 1528, 3056),
    (SEC, "가림 움직임"),
    ("v3_occstatic_s0", "v3 고정 가림만", "가림 움직임", "가리는 것이 움직이지 않는 장면만 (판 · 벽 · 탁자 · 서 있는 물체)", "v3 섞음 1,600", 1600, 3200),
    ("v3_occmoving_s0", "v3 움직이는 가림만", "가림 움직임", "가리는 것이 움직이는 장면만 (레일 판 · 굴러오는 공)", "v3 섞음 1,600", 1600, 3200),
    (SEC, "target 움직임"),
    ("v3_tgtmoving_s0", "v3 target 움직임만", "target 움직임", "target 이 굴러가거나 미끄러지는 장면만", "v3 섞음 1,600", 1600, 3200),
    ("v3_tgtstatic_s0", "v3 target 정지만", "target 움직임", "target 은 서 있고 가리는 것이 지나가는 장면만", "v3 섞음 1,600", 1600, 3200),
    (SEC, "장소"),
    ("v3_flat_s0", "v3 평지만", "장소", "평평한 바닥 장면만", "v3 섞음 1,600", 1600, 3200),
    ("v3_ramp_s0", "v3 빗면만", "장소", "쐐기 빗면 (오르기 · 내리기) 장면만", "v3 섞음 1,600", 1600, 3200),
    (SEC, "가림 길이"),
    ("v3_short_s0", "v3 짧은 가림만", "가림 길이", "가려진 시간이 짧은 장면만", "v3 섞음 1,600", 1600, 3200),
    ("v3_long_s0", "v3 긴 가림만", "가림 길이", "가려진 시간이 긴 장면만", "v3 섞음 1,600", 1600, 3200),
    (SEC, "빈 장면"),
    ("v3_aonly_s0", "v3 빈 장면 뺌", "빈 장면", "target 이 있는 영상만 (target 을 뺀 영상 없음), 영상 수는 같게", "v3 섞음 1,600", 3200, 3200),
    (SEC, "정면 · 고정 가림막"),
    ("v3_frontstatic_s0", "v3 정면 · 고정 가림막만", "정면 · 고정 가림막", "정면 카메라 · 고정 가림막 장면 전부 (VOBench 와 닮았다고 본 조각 — VOBench 는 실제로 trapdoor)", "v3 섞음 680", 680, 1360),
]
R1 = 14                                         # 표 1 첫 섹션 줄 = 13, 첫 팔 = 14
ROW, SECROWS = {}, []
r = 13
for a in ARMS:
    if a[0] == SEC:
        SECROWS.append(r)
    else:
        ROW[a[0]] = r
    r += 1
T1_LAST = r - 1                                 # 48
T2_TITLE, T2_H = T1_LAST + 3, T1_LAST + 4       # 51, 52
OFF2 = T2_H - 8                                 # 표 2 = 표 1 + 44 (머리줄 8 ↔ 52 · n · 문턱 · chance 세 줄 뒤에 섹션)
T2_LAST = T1_LAST + OFF2

# 표 1 열 (H…X) + Y 외부 평균
T1_COLS = [("vob_late", s) for s in ("all", "primitive", "realistic", "A", "B")] + [("ip1uni", s) for s in ("all", "moving", "static")] \
    + [("ip2uni", s) for s in ("all", "occluder")] + [("rubric", s) for s in ("all", "occlusion", "containment")] \
    + [("clevrer", s) for s in ("all", "behind", "cross", "overtake")]
T1_AVG = shift("H", len(T1_COLS))               # Y
# 표 2 열 (H…AA) — v3all (v3 + trapdoor)
T2_KEYS = ["all", "primitive", "realistic", "front", "side", "bird", "barrier", "object", "trapdoor", "occ_static", "occ_moving",
           "tgt_moving", "tgt_static", "flat", "ramp", "short", "long", "vob_form", "A", "B"]
T2C = {k: shift("H", i) for i, k in enumerate(T2_KEYS)}
T2_END = T2C["B"]                               # AA
# 배운 칸 (표 2 굵은 테두리)
TRAINED = {"v3_prim_s0": "primitive", "v3_real_s0": "realistic", "v3_front_s0": "front", "v3_side_s0": "side", "v3_bird_s0": "bird",
           "v3_barrier_s0": "barrier", "v3_object_s0": "object", "v3t_only_s0": "trapdoor", "v3_occstatic_s0": "occ_static",
           "v3_occmoving_s0": "occ_moving", "v3_tgtmoving_s0": "tgt_moving", "v3_tgtstatic_s0": "tgt_static", "v3_flat_s0": "flat",
           "v3_ramp_s0": "ramp", "v3_short_s0": "short", "v3_long_s0": "long", "v3_aonly_s0": "A"}
# 표 4 축 (팔, 시험 칸, 칸 이름)
AXES = [("물체 모양", [("v3_prim_s0", "primitive", "primitive"), ("v3_real_s0", "realistic", "realistic")]),
        ("카메라", [("v3_front_s0", "front", "정면"), ("v3_side_s0", "side", "측면"), ("v3_bird_s0", "bird", "위에서")]),
        ("가림 종류", [("v3_barrier_s0", "barrier", "가림막"), ("v3_object_s0", "object", "물체"), ("v3t_only_s0", "trapdoor", "trapdoor")]),
        ("가림 움직임", [("v3_occstatic_s0", "occ_static", "고정"), ("v3_occmoving_s0", "occ_moving", "움직임")]),
        ("target 움직임", [("v3_tgtmoving_s0", "tgt_moving", "움직임"), ("v3_tgtstatic_s0", "tgt_static", "정지")]),
        ("장소", [("v3_flat_s0", "flat", "평지"), ("v3_ramp_s0", "ramp", "빗면")]),
        ("가림 길이", [("v3_short_s0", "short", "짧게"), ("v3_long_s0", "long", "길게")])]
T3_TITLE = T2_LAST + 3                          # 94
T3_H = T3_TITLE + 1
T3_R0 = T3_H + 1
T3_LAST = T3_R0 + len(AXES)                     # 축 7 + 빈 장면 1
T4_TITLE = T3_LAST + 3
BLOCKS = []
r = T4_TITLE + 1
for name, lv in AXES:
    k = len(lv)
    BLOCKS.append(dict(name=name, lv=lv, k=k, r0=r, levels=[r + 3 + i for i in range(k)], mix=r + 3 + k, rel=r + 4 + k, copy=r + 5 + k))
    r += k + 7
T4_LAST = r - 1
COMPS = [("v3 를 섞어 배우면 무엇이 오르나", "v3 효과", "v3 섞음 1,600 − release", "v3_mix1600_s0", "release"),
         ("trapdoor 를 25 % 섞으면", "trapdoor 비율", "v3 섞음 1,600 + trapdoor 25 % − v3 섞음 1,600", "v3_mix1600_t25_s0", "v3_mix1600_s0"),
         ("장면 수 680 → 1,600", "양", "v3 섞음 1,600 − v3 섞음 680", "v3_mix1600_s0", "v3_mix680_s0"),
         ("장면 수 1,600 → 7,136", "양", "v3 전부 − v3 섞음 1,600", "v3_full_s0", "v3_mix1600_s0"),
         ("primitive 와 realistic", "물체 모양", "v3 primitive 만 − v3 realistic 만", "v3_prim_s0", "v3_real_s0"),
         ("정면과 측면", "카메라", "v3 정면만 − v3 측면만", "v3_front_s0", "v3_side_s0"),
         ("정면과 위에서", "카메라", "v3 정면만 − v3 위에서만", "v3_front_s0", "v3_bird_s0"),
         ("측면과 위에서", "카메라", "v3 측면만 − v3 위에서만", "v3_side_s0", "v3_bird_s0"),
         ("가림막과 물체 가림", "가림 종류", "v3 가림막만 − v3 물체 가림만", "v3_barrier_s0", "v3_object_s0"),
         ("trapdoor 와 가림막", "가림 종류", "v3 trapdoor 만 − v3 가림막만", "v3t_only_s0", "v3_barrier_s0"),
         ("trapdoor 와 물체 가림", "가림 종류", "v3 trapdoor 만 − v3 물체 가림만", "v3t_only_s0", "v3_object_s0"),
         ("고정 가림과 움직이는 가림", "가림 움직임", "v3 고정 가림만 − v3 움직이는 가림만", "v3_occstatic_s0", "v3_occmoving_s0"),
         ("target 움직임과 정지", "target 움직임", "v3 target 움직임만 − v3 target 정지만", "v3_tgtmoving_s0", "v3_tgtstatic_s0"),
         ("평지와 빗면", "장소", "v3 평지만 − v3 빗면만", "v3_flat_s0", "v3_ramp_s0"),
         ("짧은 가림과 긴 가림", "가림 길이", "v3 짧은 가림만 − v3 긴 가림만", "v3_short_s0", "v3_long_s0"),
         ("target 을 뺀 영상 (빈 장면) 이 없으면", "빈 장면", "v3 빈 장면 뺌 − v3 섞음 1,600", "v3_aonly_s0", "v3_mix1600_s0"),
         ("정면 · 고정 가림막만 배우면", "정면 · 고정 가림막", "v3 정면 · 고정 가림막만 − v3 섞음 680", "v3_frontstatic_s0", "v3_mix680_s0"),
         ("VOBench 로 배운 것과 v3 섞음", "학습셋", "VOBench PFT − v3 섞음 1,600", "vobftg", "v3_mix1600_s0")]
T5_TITLE = T4_LAST + 2
T5_H = T5_TITLE + 1
T5_THR = T5_H + 2
T5_LAST = T5_THR + len(COMPS)
GL0 = T5_LAST + 3
GLOSS = [
    ("짝 (matched pair)", "문맥 16 장이 화소 단위로 같은 가능 · 불가능 두 영상. 불가능 쪽 surprise 가 더 크면 정답."),
    ("surprise", "predictor 가 낸 미래 latent 와 실제 미래 latent (target encoder, LN) 의 L1 평균."),
    ("late 가림", "문맥 끝 (마지막 관측) 에서 target 이 가려져 있는 상태. v3 는 전부 late."),
    ("trapdoor", "바닥에 카메라 쪽으로 누운 판이 가림 직전 일어서서 target 을 가리고, 가림이 끝나면 다시 눕는 가림막. VOBench late 의 가림막이 이것이다."),
    ("v3 시험", "v3 시험 3,438 짝 + trapdoor 시험 892 짝 = 4,330 짝 (표 2 의 모든 칸이 둘을 합친 값)."),
    ("사라짐 (A)", "문맥에 target 이 있고, 불가능 영상에서는 가림 뒤에서 다시 안 나오는 짝."),
    ("나타남 (B)", "문맥에 target 이 없고, 불가능 영상에서는 가림 뒤에서 target 이 나오는 짝."),
    ("복사 기준선", "마지막 문맥 latent 를 미래 자리에 그대로 둔 예측. 예측 없이 풀리는 정도를 잰다."),
    ("비교 기준", "그 팔과 장면 수가 같고 모든 칸을 섞은 대조 팔. 표 5 가 이 짝으로 차이를 낸다."),
    ("배운 칸 · 밖 칸", "배운 칸 = 그 팔의 학습 데이터와 같은 조건의 시험 칸 (굵은 테두리). 밖 칸 = 같은 축의 나머지 칸."),
    ("이득", "같은 시험 칸에서 release (PFT 없음) 대비 점수 차 (%p)."),
    ("전이율", "밖 칸 이득 ÷ 배운 칸 이득. 1 (100 %) = 다 옮겨 감, 0 = 안 옮겨 감. 배운 칸 이득이 문턱보다 작으면 비움."),
    ("특화 이득", "배운 칸에서 한 칸만 배운 팔 − 모든 칸을 섞어 배운 팔 (같은 장면 수). 양수 = 몰아 배우는 쪽이 그 칸에서 낫다."),
]
GL_LAST = GL0 + len(GLOSS)
LASTCOL = "AA"


def build_values(table):
    G = {}

    def put(cell, v):
        c = cell.rstrip("0123456789"); rr = int(cell[len(c):]); G[(rr, ci(c))] = v

    def val(pn, s, t):
        v = table.get(pn, {}).get(s, {}).get(t)
        return None if v is None else v[0]

    def n_of(pn, s):
        d = table.get(pn, {}).get(s, {})
        return next(iter(d.values()))[1] if d else None

    put("B2", "BlenderTrack v3 학습 ablation: 어떤 학습 데이터가 어떤 벤치로 옮겨 가나")
    put("B3", "모든 팔: release predictor 위 PFT · 2,000 step (20 epoch × 100) · global batch 24 · 같은 lr · seed 0. 값 = matched 짝 정확도 (%), chance 50. "
              "모든 값은 vll7 한 노드에서 같은 통과로 채점 (VO_moving_occluder 는 표에 없음).")
    put("B4", "칸 색: 표 1 · 2 와 표 5 는 문턱 기준 (진한 색 = 문턱 이상, 옅은 색 = 절반 이상). 문턱 = 문턱 배수 × √(2500 / n), n = 그 열의 짝 수. "
              "표 3 · 4 의 이득은 −10 (빨강) ~ 0 (흰색) ~ +10 (초록) 색 띠, 전이율은 0 (흰색) ~ 100 % (파랑).")
    put("B5", "표 1 = 외부 벤치 (전부 object permanence 짝만) · 표 2 = v3 시험 칸 (trapdoor 포함) · 표 3 = 축마다 전이 요약 · 표 4 = 축별 전이 행렬 · "
              "표 5 = 같은 크기 두 팔의 차이 · 용어는 맨 아래.")
    put("B6", "문턱 배수 (표준오차의 몇 배)"); put("C6", 2); put("D6", "전이율 판정: 옮겨 감 (≥)"); put("E6", 0.8); put("F6", "안 옮겨 감 (≤)"); put("G6", 0.3)

    def arm_block(base_title_row, hdr_row, title, cols_spec, add_rows):
        pass

    # ---------------- 표 1
    put("B7", "표 1. 학습 밖 외부 벤치로 옮겨 가나 (object permanence 짝만)")
    for c, t in zip("BCDEFG", ["학습 팔", "바꾼 축", "학습 데이터", "비교 기준", "장면 수", "학습 영상"]):
        put(f"{c}8", t)
    put("H8", "VOBench late (trapdoor) · permanence"); put("M8", "IntPhys 1 late occ · permanence (O1)"); put("P8", "IntPhys 2 late · permanence")
    put("R8", "real bench (Rubric office)"); put("U8", "CLEVRER style (static 제외)"); put(f"{T1_AVG}8", "외부 평균 (각 벤치 전체 열)")
    for i, t in enumerate(["전체", "primitive", "realistic", "사라짐 (A)", "나타남 (B)", "전체", "물체 움직임", "물체 정지", "전체", "가림막 있음",
                           "전체", "occlusion", "containment", "전체", "behind", "cross", "overtake"]):
        put(f"{shift('H', i)}9", t)
    put("B10", "시험 짝 수 (n)"); put("B11", "색 문턱 (%p)"); put("B12", "Chance")
    for i, (pn, s) in enumerate(T1_COLS):
        c = shift("H", i)
        put(f"{c}10", n_of(pn, s)); put(f"{c}11", f"=ROUND($C$6*SQRT(2500/{c}10),1)"); put(f"{c}12", 50)
    put(f"{T1_AVG}10", "-"); put(f"{T1_AVG}11", "=ROUND($C$6*SQRT(2500/H10+2500/M10+2500/P10+2500/R10+2500/U10)/5,1)"); put(f"{T1_AVG}12", 50)
    r = 13
    for a in ARMS:
        if a[0] == SEC:
            put(f"B{r}", a[1])
        else:
            for c, x in zip("BCDEFG", a[1:]):
                put(f"{c}{r}", x)
            for i, (pn, s) in enumerate(T1_COLS):
                put(f"{shift('H', i)}{r}", val(pn, s, a[0]))
            put(f"{T1_AVG}{r}", f'=IF(COUNT(H{r},M{r},P{r},R{r},U{r})=5,AVERAGE(H{r},M{r},P{r},R{r},U{r}),"")')
        r += 1
    # ---------------- 표 2
    put(f"B{T2_TITLE}", "표 2. v3 시험 칸별 — v3 + trapdoor 4,330 짝 (굵은 테두리 = 그 팔이 배운 칸)")
    h = T2_H
    for c, t in zip("BCDEFG", ["학습 팔", "바꾼 축", "학습 데이터", "비교 기준", "장면 수", "학습 영상"]):
        put(f"{c}{h}", t)
    put(f"H{h}", "전체"); put(f"{T2C['primitive']}{h}", "물체 모양"); put(f"{T2C['front']}{h}", "카메라"); put(f"{T2C['barrier']}{h}", "가림 종류")
    put(f"{T2C['occ_static']}{h}", "가림 움직임"); put(f"{T2C['tgt_moving']}{h}", "target 움직임"); put(f"{T2C['flat']}{h}", "장소")
    put(f"{T2C['short']}{h}", "가림 길이"); put(f"{T2C['vob_form']}{h}", "VOBench 꼴 (정면 · trapdoor)"); put(f"{T2C['A']}{h}", "방향")
    sub = {"primitive": "primitive", "realistic": "realistic", "front": "정면", "side": "측면", "bird": "위에서", "barrier": "가림막", "object": "물체",
           "trapdoor": "trapdoor", "occ_static": "고정", "occ_moving": "움직임", "tgt_moving": "움직임", "tgt_static": "정지", "flat": "평지",
           "ramp": "빗면", "short": "짧게", "long": "길게", "A": "사라짐 (A)", "B": "나타남 (B)"}
    for k, t in sub.items():
        put(f"{T2C[k]}{h + 1}", t)
    nr, tr, cr = h + 2, h + 3, h + 4
    put(f"B{nr}", "시험 짝 수 (n)"); put(f"B{tr}", "색 문턱 (%p)"); put(f"B{cr}", "Chance")
    for k in T2_KEYS:
        c = T2C[k]
        put(f"{c}{nr}", n_of("v3all", k)); put(f"{c}{tr}", f"=ROUND($C$6*SQRT(2500/{c}{nr}),1)"); put(f"{c}{cr}", 50)
    r = 13
    for a in ARMS:
        r2 = r + OFF2
        if a[0] == SEC:
            put(f"B{r2}", f"=B{r}")
        else:
            for c in "BCDEFG":
                put(f"{c}{r2}", f"={c}{r}")
            for k in T2_KEYS:
                put(f"{T2C[k]}{r2}", val("v3all", k, a[0]))
        r += 1
    # ---------------- 표 3 (요약)
    def gcell(blk, i, row): return f"{shift('K', i)}{row}"
    put(f"B{T3_TITLE}", "표 3. v3 안 전이 요약: 한 칸만 배우면 같은 축의 다른 칸으로 옮겨 가나 (축마다 한 줄, 표 4 에서 계산)")
    for c, t in zip("BCHIJKLM", ["축", "학습 팔 (각자 한 칸만)", "배운 칸 이득 (%p)", "밖 칸 이득 (%p)", "전이율 (밖 / 배운)",
                                 "섞음 1,600 이득 (같은 칸들 평균, %p)", "특화 이득 (배운 칸에서 − 섞음 1,600, %p)", "판정"]):
        put(f"{c}{T3_H}", t)
    for j, b in enumerate(BLOCKS):
        rr, k = T3_R0 + j, b["k"]
        dg = [f"{shift('K', i)}{b['levels'][i]}" for i in range(k)]
        og = [f"{shift('K', i)}{b['levels'][a_]}" for a_ in range(k) for i in range(k) if i != a_]
        mg = [f"{shift('K', i)}{b['mix']}" for i in range(k)]
        ds = [f"{shift('H', i)}{b['levels'][i]}" for i in range(k)]
        ms = [f"{shift('H', i)}{b['mix']}" for i in range(k)]
        thr = ",".join(f"{T2C[s]}${tr}" for _, s, _ in b["lv"])
        put(f"B{rr}", b["name"]); put(f"C{rr}", " · ".join(f"{lab} 만" for _, _, lab in b["lv"]))
        put(f"H{rr}", f'=IF(COUNT({",".join(dg)})={k},AVERAGE({",".join(dg)}),"")')
        put(f"I{rr}", f'=IF(COUNT({",".join(og)})={len(og)},AVERAGE({",".join(og)}),"")')
        put(f"J{rr}", f'=IF(AND(ISNUMBER(H{rr}),ISNUMBER(I{rr})),IF(H{rr}>=MAX({thr}),I{rr}/H{rr},""),"")')
        put(f"K{rr}", f'=IF(COUNT({",".join(mg)})={k},AVERAGE({",".join(mg)}),"")')
        put(f"L{rr}", f'=IF(COUNT({",".join(ds + ms)})={2 * k},AVERAGE({",".join(ds)})-AVERAGE({",".join(ms)}),"")')
        put(f"M{rr}", f'=IF(ISNUMBER(J{rr}),IF(J{rr}>=$E$6,"옮겨 감",IF(J{rr}<=$G$6,"안 옮겨 감","일부")),"")')
    rr = T3_LAST
    a2, m2, rl = ROW["v3_aonly_s0"] + OFF2, ROW["v3_mix1600_s0"] + OFF2, ROW["release"] + OFF2
    A_, B_ = T2C["A"], T2C["B"]
    put(f"B{rr}", "빈 장면"); put(f"C{rr}", "target 있는 영상만 (배운 칸 = 사라짐 A, 밖 = 나타남 B; 나타남은 학습 전부터 99 % 라 전이율은 읽지 말 것)")
    put(f"H{rr}", f'=IF(AND(ISNUMBER({A_}{a2}),ISNUMBER({A_}{rl})),{A_}{a2}-{A_}{rl},"")')
    put(f"I{rr}", f'=IF(AND(ISNUMBER({B_}{a2}),ISNUMBER({B_}{rl})),{B_}{a2}-{B_}{rl},"")')
    put(f"J{rr}", f'=IF(AND(ISNUMBER(H{rr}),ISNUMBER(I{rr})),IF(H{rr}>={A_}${tr},I{rr}/H{rr},""),"")')
    put(f"K{rr}", f'=IF(COUNT({A_}{m2},{B_}{m2},{A_}{rl},{B_}{rl})=4,AVERAGE({A_}{m2},{B_}{m2})-AVERAGE({A_}{rl},{B_}{rl}),"")')
    put(f"L{rr}", f'=IF(AND(ISNUMBER({A_}{a2}),ISNUMBER({A_}{m2})),{A_}{a2}-{A_}{m2},"")')
    put(f"M{rr}", f'=IF(ISNUMBER(J{rr}),IF(J{rr}>=$E$6,"옮겨 감",IF(J{rr}<=$G$6,"안 옮겨 감","일부")),"")')
    # ---------------- 표 4
    put(f"B{T4_TITLE}", "표 4. 축별 전이 행렬 (행 = 학습 팔, 열 = v3 시험 칸, 굵은 테두리 = 배운 칸)")
    for b in BLOCKS:
        r0, k, lv = b["r0"], b["k"], b["lv"]
        put(f"B{r0}", b["name"]); put(f"B{r0 + 1}", "학습 팔"); put(f"C{r0 + 1}", "학습 데이터")
        put(f"H{r0 + 1}", "점수 (%)"); put(f"K{r0 + 1}", "release 대비 이득 (%p)"); put(f"N{r0 + 1}", "전이율 (배운 칸 이득 = 100 %)")
        for i, (_, _, lab) in enumerate(lv):
            for c0 in "HKN":
                put(f"{shift(c0, i)}{r0 + 2}", f"시험: {lab}")
        rows = [(b["levels"][i], lv[i][0], i) for i in range(k)] + [(b["mix"], "v3_mix1600_s0", None), (b["rel"], "release", None), (b["copy"], "copy", None)]
        for rr, tag, di in rows:
            t2, t1 = ROW[tag] + OFF2, ROW[tag]
            put(f"B{rr}", f"=B{t2}"); put(f"C{rr}", f"=D{t1}")
            for i, (_, s, _) in enumerate(lv):
                sc = T2C[s]
                put(f"{shift('H', i)}{rr}", f'=IF(ISNUMBER({sc}{t2}),{sc}{t2},"")')
                if tag not in ("release", "copy"):
                    x, x0 = f"{shift('H', i)}{rr}", f"{shift('H', i)}{b['rel']}"
                    put(f"{shift('K', i)}{rr}", f'=IF(AND(ISNUMBER({x}),ISNUMBER({x0})),{x}-{x0},"")')
                if di is not None:
                    g, gd = f"{shift('K', i)}{rr}", f"{shift('K', di)}{rr}"
                    put(f"{shift('N', i)}{rr}", f'=IF(AND(ISNUMBER({g}),ISNUMBER({gd})),IF({gd}>={T2C[lv[di][1]]}${tr},{g}/{gd},""),"")')
    # ---------------- 표 5
    put(f"B{T5_TITLE}", "표 5. 같은 크기 두 팔의 차이 (앞 팔 − 뒤 팔, %p)")
    put(f"B{T5_H}", "질문"); put(f"C{T5_H}", "축"); put(f"D{T5_H}", "비교 (앞 팔 − 뒤 팔)")
    put(f"H{T5_H}", "VOBench late (trapdoor) · permanence"); put(f"M{T5_H}", "IntPhys 1 late occ"); put(f"N{T5_H}", "IntPhys 2 late")
    put(f"O{T5_H}", "real bench"); put(f"P{T5_H}", "CLEVRER style"); put(f"Q{T5_H}", "외부 평균"); put(f"R{T5_H}", "v3 시험 (trapdoor 포함)")
    for c, t in zip("HIJKLRSTU", ["전체", "primitive", "realistic", "사라짐 (A)", "나타남 (B)", "전체", "사라짐 (A)", "나타남 (B)", "VOBench 꼴"]):
        put(f"{c}{T5_H + 1}", t)
    put(f"B{T5_THR}", "색 문턱 (%p)")
    T5_1 = dict(H="H", I="I", J="J", K="K", L="L", M="M", N="P", O="R", P="U", Q=T1_AVG)
    T5_2 = dict(R=T2C["all"], S=T2C["A"], T=T2C["B"], U=T2C["vob_form"])
    for c, s in T5_1.items(): put(f"{c}{T5_THR}", f"={s}11")
    for c, s in T5_2.items(): put(f"{c}{T5_THR}", f"={s}{tr}")
    for i, (q, ax, txt, a, b) in enumerate(COMPS):
        rr = T5_THR + 1 + i
        put(f"B{rr}", q); put(f"C{rr}", ax); put(f"D{rr}", txt)
        ra, rb = ROW[a], ROW[b]
        for c, s in T5_1.items():
            put(f"{c}{rr}", f'=IF(AND(ISNUMBER({s}{ra}),ISNUMBER({s}{rb})),{s}{ra}-{s}{rb},"")')
        for c, s in T5_2.items():
            put(f"{c}{rr}", f'=IF(AND(ISNUMBER({s}{ra + OFF2}),ISNUMBER({s}{rb + OFF2})),{s}{ra + OFF2}-{s}{rb + OFF2},"")')
    # ---------------- 용어
    put(f"B{GL0}", "용어")
    for i, (k, v) in enumerate(GLOSS):
        put(f"B{GL0 + 1 + i}", k); put(f"C{GL0 + 1 + i}", v)
    return G


def to_chunks(G, row_ranges):
    out = []
    for r0, r1 in row_ranges:
        rows = []
        for rr in range(r0, r1 + 1):
            row = [G.get((rr, c)) for c in range(1, ci(LASTCOL) + 1)]
            while row and row[-1] is None:
                row.pop()
            rows.append(row if row else [None])
        out.append({"range": f"{TAB}!B{r0}:{LASTCOL}{r1}", "rows": rows})
    return out


# ============================================================ 서식
def rgb(h): h = h.lstrip("#"); return {"red": round(int(h[0:2], 16) / 255, 2), "green": round(int(h[2:4], 16) / 255, 2), "blue": round(int(h[4:6], 16) / 255, 2)}


NAVY, SECC, DARK, MID, LV, LH, WHITE, GREY = "#1F3864", "#D9E1F2", "#252525", "#7E7E7E", "#C7C7C7", "#D8D8D8", "#FFFFFF", "#F3F3F3"


def rng_(a, b):
    ca = a.rstrip("0123456789"); ra = int(a[len(ca):]); cb = b.rstrip("0123456789"); rb = int(b[len(cb):])
    return {"sheetId": SID, "startRowIndex": ra - 1, "endRowIndex": rb, "startColumnIndex": ci(ca), "endColumnIndex": ci(cb) + 1}


def build_format():
    R = []
    def B(c, st="SOLID"): return {"style": st, "color": rgb(c)}
    def fmt(a, b, f, flds): R.append({"repeatCell": {"range": rng_(a, b), "cell": {"userEnteredFormat": f}, "fields": ",".join("userEnteredFormat." + x for x in flds)}})
    def merge(a, b): R.append({"mergeCells": {"range": rng_(a, b), "mergeType": "MERGE_ALL"}})
    def border(a, b, **kw): R.append({"updateBorders": {"range": rng_(a, b), **kw}})
    def header(a, b): fmt(a, b, {"backgroundColor": rgb(NAVY), "textFormat": {"bold": True, "foregroundColor": rgb(WHITE)}, "horizontalAlignment": "CENTER", "wrapStrategy": "WRAP"},
                          ["backgroundColor", "textFormat.bold", "textFormat.foregroundColor", "horizontalAlignment", "wrapStrategy"])
    def box(a, b): border(a, b, top=B(DARK, "SOLID_MEDIUM"), bottom=B(DARK, "SOLID_MEDIUM"), left=B(DARK), right=B(DARK), innerHorizontal=B(LH), innerVertical=B(LV))
    def hinner(a, b): border(a, b, innerHorizontal=B(WHITE), innerVertical=B(WHITE), bottom=B(DARK))
    def vsep(c, r1, r2): border(f"{c}{r1}", f"{c}{r2}", right=B(MID))
    def section(rr, c2): merge(f"B{rr}", f"{c2}{rr}"); fmt(f"B{rr}", f"{c2}{rr}", {"backgroundColor": rgb(SECC), "textFormat": {"bold": True}, "horizontalAlignment": "LEFT"}, ["backgroundColor", "textFormat.bold", "horizontalAlignment"])
    def numfmt(a, b, pat): fmt(a, b, {"numberFormat": {"type": "NUMBER", "pattern": pat}, "horizontalAlignment": "CENTER"}, ["numberFormat", "horizontalAlignment"])
    def grey(a, b): fmt(a, b, {"backgroundColor": rgb(GREY), "textFormat": {"italic": True}}, ["backgroundColor", "textFormat.italic"])
    def boldbox(cell):
        m = B("#000000", "SOLID_MEDIUM"); border(cell, cell, top=m, bottom=m, left=m, right=m); fmt(cell, cell, {"textFormat": {"bold": True}}, ["textFormat.bold"])
    def rowh(rr, px): R.append({"updateDimensionProperties": {"range": {"sheetId": SID, "dimension": "ROWS", "startIndex": rr - 1, "endIndex": rr}, "properties": {"pixelSize": px}, "fields": "pixelSize"}})

    R.append({"updateSheetProperties": {"properties": {"sheetId": SID, "gridProperties": {"hideGridlines": True}}, "fields": "gridProperties.hideGridlines"}})
    fmt("A1", f"AB{GL_LAST + 3}", {"textFormat": {"fontFamily": "Times New Roman", "fontSize": 10}, "verticalAlignment": "MIDDLE"}, ["textFormat.fontFamily", "textFormat.fontSize", "verticalAlignment"])
    for a, b_, px in [(0, 1, 18), (1, 2, 230), (2, 3, 95), (3, 4, 210), (4, 5, 105), (5, 7, 62), (7, 27, 84)]:
        R.append({"updateDimensionProperties": {"range": {"sheetId": SID, "dimension": "COLUMNS", "startIndex": a, "endIndex": b_}, "properties": {"pixelSize": px}, "fields": "pixelSize"}})
    merge("B2", f"{LASTCOL}2"); fmt("B2", "B2", {"textFormat": {"bold": True, "fontSize": 14, "foregroundColor": rgb(NAVY)}}, ["textFormat.bold", "textFormat.fontSize", "textFormat.foregroundColor"])
    for rr in (3, 4, 5): merge(f"B{rr}", f"{LASTCOL}{rr}")
    fmt("B3", "B5", {"wrapStrategy": "WRAP", "textFormat": {"foregroundColor": rgb("#404040")}}, ["wrapStrategy", "textFormat.foregroundColor"])
    fmt("B6", "G6", {"wrapStrategy": "WRAP", "horizontalAlignment": "RIGHT"}, ["wrapStrategy", "horizontalAlignment"])
    for cc in ("C6", "E6", "G6"):
        fmt(cc, cc, {"backgroundColor": rgb("#FFF2CC"), "textFormat": {"bold": True, "foregroundColor": rgb("#0000FF")}, "horizontalAlignment": "CENTER"},
            ["backgroundColor", "textFormat.bold", "textFormat.foregroundColor", "horizontalAlignment"])
        border(cc, cc, top=B(MID), bottom=B(MID), left=B(MID), right=B(MID))
    for rr in (7, T2_TITLE, T3_TITLE, T4_TITLE, T5_TITLE, GL0):
        fmt(f"B{rr}", f"B{rr}", {"textFormat": {"bold": True, "fontSize": 12, "foregroundColor": rgb(NAVY)}}, ["textFormat.bold", "textFormat.fontSize", "textFormat.foregroundColor"])

    def table12(h, last, end, groups, singles, seps, secs, nrow):
        header(f"B{h}", f"{end}{h + 1}")
        for cc in "BCDEFG": merge(f"{cc}{h}", f"{cc}{h + 1}")
        for a, b_ in groups: merge(f"{a}{h}", f"{b_}{h}")
        for cc in singles: merge(f"{cc}{h}", f"{cc}{h + 1}")
        grey(f"B{nrow}", f"{end}{nrow + 2}")
        numfmt(f"H{nrow}", f"{end}{nrow}", "#,##0"); numfmt(f"H{nrow + 1}", f"{end}{last}", "0.0")
        numfmt(f"F{nrow + 3}", f"G{last}", "#,##0")
        fmt(f"C{nrow + 3}", f"C{last}", {"horizontalAlignment": "CENTER"}, ["horizontalAlignment"])
        fmt(f"E{nrow + 3}", f"E{last}", {"horizontalAlignment": "CENTER"}, ["horizontalAlignment"])
        fmt(f"D{nrow + 3}", f"D{last}", {"wrapStrategy": "WRAP"}, ["wrapStrategy"])
        box(f"B{h}", f"{end}{last}"); hinner(f"B{h}", f"{end}{h + 1}")
        for cc in seps: vsep(cc, nrow, last)
        border(f"B{nrow + 2}", f"{end}{nrow + 2}", bottom=B(MID))
        for rr in secs: section(rr, end)
        rowh(h, 34); rowh(h + 1, 34)
    table12(8, T1_LAST, T1_AVG, [("H", "L"), ("M", "O"), ("P", "Q"), ("R", "T"), ("U", "X")], [T1_AVG], ["G", "L", "O", "Q", "T", "X"], SECROWS, 10)
    g2 = [(T2C["primitive"], T2C["realistic"]), (T2C["front"], T2C["bird"]), (T2C["barrier"], T2C["trapdoor"]), (T2C["occ_static"], T2C["occ_moving"]),
          (T2C["tgt_moving"], T2C["tgt_static"]), (T2C["flat"], T2C["ramp"]), (T2C["short"], T2C["long"]), (T2C["A"], T2C["B"])]
    table12(T2_H, T2_LAST, T2_END, g2, ["H", T2C["vob_form"]],
            ["G", "H", T2C["realistic"], T2C["bird"], T2C["trapdoor"], T2C["occ_moving"], T2C["tgt_static"], T2C["ramp"], T2C["long"], T2C["vob_form"]],
            [s + OFF2 for s in SECROWS], T2_H + 2)
    for tag, s in TRAINED.items():
        boldbox(f"{T2C[s]}{ROW[tag] + OFF2}")
    for tag in ("release",):
        fmt(f"B{ROW[tag]}", f"{T1_AVG}{ROW[tag]}", {"textFormat": {"italic": True}}, ["textFormat.italic"])
        fmt(f"B{ROW[tag] + OFF2}", f"{T2_END}{ROW[tag] + OFF2}", {"textFormat": {"italic": True}}, ["textFormat.italic"])
    # 표 3
    header(f"B{T3_H}", f"M{T3_H}"); merge(f"C{T3_H}", f"G{T3_H}"); rowh(T3_H, 48)
    for rr in range(T3_R0, T3_LAST + 1): merge(f"C{rr}", f"G{rr}")
    fmt(f"B{T3_R0}", f"B{T3_LAST}", {"textFormat": {"bold": True}}, ["textFormat.bold"])
    numfmt(f"H{T3_R0}", f"I{T3_LAST}", "+0.0;-0.0;0.0"); numfmt(f"K{T3_R0}", f"L{T3_LAST}", "+0.0;-0.0;0.0"); numfmt(f"J{T3_R0}", f"J{T3_LAST}", "0%")
    fmt(f"M{T3_R0}", f"M{T3_LAST}", {"horizontalAlignment": "CENTER", "textFormat": {"bold": True}}, ["horizontalAlignment", "textFormat.bold"])
    box(f"B{T3_H}", f"M{T3_LAST}"); hinner(f"B{T3_H}", f"M{T3_H}")
    for cc in ("G", "I", "J", "L"): vsep(cc, T3_R0, T3_LAST)
    border(f"B{T3_LAST - 1}", f"M{T3_LAST - 1}", bottom=B(MID))
    # 표 4 블록 (모두 직접 — 블록 수가 적다)
    src = {}                                                           # k 마다 첫 블록만 직접 서식 · 나머지는 서식 복사 (병합 포함)
    for b in BLOCKS:
        r0, k, lvr, last = b["r0"], b["k"], b["levels"], b["copy"]
        e = lambda c0: shift(c0, k - 1)
        if k in src:
            s0 = src[k]
            R.append({"copyPaste": {"source": rng_(f"B{s0['r0']}", f"P{s0['copy']}"), "destination": rng_(f"B{r0}", f"P{last}"), "pasteType": "PASTE_FORMAT"}})
            continue
        src[k] = b
        section(r0, e("N"))
        header(f"B{r0 + 1}", f"G{r0 + 2}"); merge(f"B{r0 + 1}", f"B{r0 + 2}"); merge(f"C{r0 + 1}", f"G{r0 + 2}")
        for c0 in "HKN":
            header(f"{c0}{r0 + 1}", f"{e(c0)}{r0 + 2}"); merge(f"{c0}{r0 + 1}", f"{e(c0)}{r0 + 1}")
        for rr in range(r0 + 3, last + 1): merge(f"C{rr}", f"G{rr}")
        numfmt(f"H{r0 + 3}", f"{e('H')}{last}", "0.0"); numfmt(f"K{r0 + 3}", f"{e('K')}{last}", "+0.0;-0.0;0.0"); numfmt(f"N{r0 + 3}", f"{e('N')}{last}", "0%")
        fmt(f"B{b['rel']}", f"{e('N')}{last}", {"textFormat": {"italic": True}, "backgroundColor": rgb(GREY)}, ["textFormat.italic", "backgroundColor"])
        for c0, c1 in (("B", e("H")), ("K", e("K")), ("N", e("N"))):
            box(f"{c0}{r0 + 1}", f"{c1}{last}"); hinner(f"{c0}{r0 + 1}", f"{c1}{r0 + 2}")
            border(f"{c0}{lvr[-1]}", f"{c1}{lvr[-1]}", bottom=B(MID))
        vsep("G", r0 + 3, last)
        for i, rr in enumerate(lvr):
            boldbox(f"{shift('H', i)}{rr}"); boldbox(f"{shift('K', i)}{rr}")
            fmt(f"{shift('N', i)}{rr}", f"{shift('N', i)}{rr}", {"textFormat": {"foregroundColor": rgb("#999999")}}, ["textFormat.foregroundColor"])
    # 표 5
    h1, thr, last5 = T5_H, T5_THR, T5_LAST
    header(f"B{h1}", f"U{h1 + 1}"); rowh(h1, 34); rowh(h1 + 1, 34)
    for cc in "BC": merge(f"{cc}{h1}", f"{cc}{h1 + 1}")
    merge(f"D{h1}", f"G{h1 + 1}"); merge(f"H{h1}", f"L{h1}"); merge(f"R{h1}", f"U{h1}")
    for cc in "MNOPQ": merge(f"{cc}{h1}", f"{cc}{h1 + 1}")
    for rr in range(thr, last5 + 1): merge(f"D{rr}", f"G{rr}")
    grey(f"B{thr}", f"U{thr}")
    numfmt(f"H{thr}", f"U{thr}", "0.0"); numfmt(f"H{thr + 1}", f"U{last5}", "+0.0;-0.0;0.0")
    fmt(f"C{thr + 1}", f"C{last5}", {"horizontalAlignment": "CENTER"}, ["horizontalAlignment"])
    box(f"B{h1}", f"U{last5}"); hinner(f"B{h1}", f"U{h1 + 1}")
    for cc in ("G", "L", "M", "N", "O", "P", "Q"): vsep(cc, thr, last5)
    border(f"B{thr}", f"U{thr}", bottom=B(MID))
    # 용어
    for rr in range(GL0 + 1, GL_LAST + 1): merge(f"C{rr}", f"P{rr}")
    fmt(f"B{GL0 + 1}", f"B{GL_LAST}", {"textFormat": {"bold": True}}, ["textFormat.bold"])
    fmt(f"C{GL0 + 1}", f"C{GL_LAST}", {"wrapStrategy": "WRAP"}, ["wrapStrategy"])
    border(f"B{GL0 + 1}", f"P{GL_LAST}", top=B(DARK, "SOLID_MEDIUM"), bottom=B(DARK, "SOLID_MEDIUM"), left=B(DARK), right=B(DARK), innerHorizontal=B(LH), innerVertical=B(MID))
    return R


def build_cond():
    R, idx = [], [0]
    GS, GL_, RS, RL = "#B7E1CD", "#E6F4EA", "#F4C7C3", "#FCE8E6"
    def cond(rs, formula, bg):
        R.append({"addConditionalFormatRule": {"index": idx[0], "rule": {"ranges": [rng_(*x) for x in rs],
                  "booleanRule": {"condition": {"type": "CUSTOM_FORMULA", "values": [{"userEnteredValue": formula}]}, "format": {"backgroundColor": rgb(bg)}}}}}); idx[0] += 1
    rel1, rel2, tr2 = ROW["release"], ROW["release"] + OFF2, T2_H + 3
    for (a, b_), rel, t in [((f"H{R1}", f"{T1_AVG}{T1_LAST}"), rel1, 11), ((f"H{R1 + OFF2}", f"{T2_END}{T2_LAST}"), rel2, tr2)]:
        x = a
        cond([(a, b_)], f"=AND(ISNUMBER({x}),ISNUMBER(H${rel}),{x}-H${rel}>=H${t})", GS)
        cond([(a, b_)], f"=AND(ISNUMBER({x}),ISNUMBER(H${rel}),H${rel}-{x}>=H${t})", RS)
        cond([(a, b_)], f"=AND(ISNUMBER({x}),ISNUMBER(H${rel}),{x}-H${rel}>=H${t}/2)", GL_)
        cond([(a, b_)], f"=AND(ISNUMBER({x}),ISNUMBER(H${rel}),H${rel}-{x}>=H${t}/2)", RL)
    x = f"H{T5_THR + 1}"
    for f_, cc in [(f"=AND(ISNUMBER({x}),{x}>=H${T5_THR})", GS), (f"=AND(ISNUMBER({x}),-{x}>=H${T5_THR})", RS),
                   (f"=AND(ISNUMBER({x}),{x}>=H${T5_THR}/2)", GL_), (f"=AND(ISNUMBER({x}),-{x}>=H${T5_THR}/2)", RL)]:
        cond([(x, f"U{T5_LAST}")], f_, cc)
    for t, cc in [("옮겨 감", GS), ("안 옮겨 감", RS), ("일부", "#FFF2CC")]:
        R.append({"addConditionalFormatRule": {"index": idx[0], "rule": {"ranges": [rng_(f"M{T3_R0}", f"M{T3_LAST}")],
                  "booleanRule": {"condition": {"type": "TEXT_EQ", "values": [{"userEnteredValue": t}]}, "format": {"backgroundColor": rgb(cc)}}}}}); idx[0] += 1
    gain = [(f"H{T3_R0}", f"I{T3_LAST}"), (f"K{T3_R0}", f"L{T3_LAST}")]; ratio = [(f"J{T3_R0}", f"J{T3_LAST}")]
    for b in BLOCKS:
        k = b["k"]
        gain.append((f"K{b['levels'][0]}", f"{shift('K', k - 1)}{b['mix']}")); ratio.append((f"N{b['levels'][0]}", f"{shift('N', k - 1)}{b['levels'][-1]}"))
    def grad(rs, lo, mid, hi):
        g = {"minpoint": {"type": "NUMBER", "value": str(lo[0]), "color": rgb(lo[1])}, "maxpoint": {"type": "NUMBER", "value": str(hi[0]), "color": rgb(hi[1])}}
        if mid: g["midpoint"] = {"type": "NUMBER", "value": str(mid[0]), "color": rgb(mid[1])}
        R.append({"addConditionalFormatRule": {"index": idx[0], "rule": {"ranges": [rng_(*x) for x in rs], "gradientRule": g}}}); idx[0] += 1
    grad(gain, (-10, "#E67C73"), (0, "#FFFFFF"), (10, "#57BB8A"))
    grad(ratio, (0, "#FFFFFF"), None, (1, "#6FA8DC"))
    return R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-cond", type=int, default=17, help="지금 탭에 있는 조건부 서식 수 (지우기)")
    a = ap.parse_args()
    table = json.load(open(REPO / "predictor_lab/results/v3_ablation/table.json"))
    OUTD.mkdir(parents=True, exist_ok=True)
    clear = [{"unmergeCells": {"range": {"sheetId": SID, "startRowIndex": 0, "endRowIndex": 260, "startColumnIndex": 0, "endColumnIndex": 28}}},
             {"updateCells": {"range": {"sheetId": SID, "startRowIndex": 0, "endRowIndex": 260, "startColumnIndex": 0, "endColumnIndex": 28}, "fields": "userEnteredValue,userEnteredFormat,note"}},
             {"updateDimensionProperties": {"range": {"sheetId": SID, "dimension": "ROWS", "startIndex": 0, "endIndex": 260}, "properties": {"pixelSize": 21}, "fields": "pixelSize"}}] \
        + [{"deleteConditionalFormatRule": {"sheetId": SID, "index": 0}} for _ in range(a.n_cond)]
    json.dump(clear, open(OUTD / "00_clear.json", "w"), ensure_ascii=False, separators=(",", ":"))
    G = build_values(table)
    chunks = to_chunks(G, [(2, T1_LAST), (T2_TITLE, T2_LAST), (T3_TITLE, T4_TITLE + 1 + 30), (T4_TITLE + 32, GL_LAST)])
    for i, c in enumerate(chunks):
        json.dump(c, open(OUTD / f"10_values_{i}.json", "w"), ensure_ascii=False, separators=(",", ":"))
    F = build_format()
    k = (len(F) + 2) // 3
    for i in range(3):
        json.dump(F[i * k:(i + 1) * k], open(OUTD / f"20_format_{i}.json", "w"), ensure_ascii=False, separators=(",", ":"))
    json.dump(build_cond(), open(OUTD / "30_cond.json", "w"), ensure_ascii=False, separators=(",", ":"))
    lay = dict(T1_LAST=T1_LAST, T2_TITLE=T2_TITLE, T2_LAST=T2_LAST, T3=(T3_TITLE, T3_LAST), T4=(T4_TITLE, T4_LAST), T5=(T5_TITLE, T5_LAST), GL=(GL0, GL_LAST),
               ROW=ROW, OFF2=OFF2, T2C=T2C, T1_AVG=T1_AVG)
    json.dump(lay, open(OUTD / "layout.json", "w"), ensure_ascii=False, indent=1)
    for f in sorted(OUTD.glob("*.json")):
        print(f.name, f.stat().st_size)
    print({k_: v for k_, v in lay.items() if k_ not in ("ROW", "T2C")})


if __name__ == "__main__":
    main()
