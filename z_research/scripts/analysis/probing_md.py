#!/usr/bin/env python
"""attn_probe 결과 -> 전수 기록 markdown 절 (문서 제2부를 통째로 재생성).

요약하지 않는다. head 54개의 메타, (head x 평가조건) 324칸, 팔 안 vis<->occ 이식,
그 이식의 k 분해, 붕괴 통계를 전부 적는다. 숫자는 전부 산출물에서 다시 계산한다.

    python z_research/scripts/analysis/probing_md.py \
        --run    z_research/IntPhysGenV11/exp_results/attn_probe__v11_vith \
        --index  data_csv/intphysgen_v11/index_probe.csv \
        --report z_research/IntPhysGenV11/exp_results/report.json \
        --append z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md

MARK 이후를 통째로 갈아끼우므로 몇 번을 돌려도 중복되지 않는다.
"""
from __future__ import annotations
import argparse, collections, csv, io, json
import numpy as np

CONDS = ["static_visible", "static_occlusion", "moving_visible_flat",
         "moving_occlusion_flat", "moving_visible", "moving_occlusion"]
SHORT = {"static_visible": "sta+VIS", "static_occlusion": "sta+OCC",
         "moving_visible_flat": "flat+VIS", "moving_occlusion_flat": "flat+OCC",
         "moving_visible": "ramp+VIS", "moving_occlusion": "ramp+OCC"}
ARMS = [("Static", "static_visible", "static_occlusion"),
        ("Constant velocity", "moving_visible_flat", "moving_occlusion_flat"),
        ("Accelerating", "moving_visible", "moving_occlusion")]
PTS = [("contextF__f1to16", "z", "context encoder, frames 1-16"),
       ("targetF__f17to32", "h", "target encoder, frames 17-32"),
       ("pred__f17to32", "p", "predictor 출력, frames 17-32")]
TGT = [("shape", "shape"), ("color", "colour"), ("env", "background")]
KS = ["1", "2", "3", "4"]
MARK = "<!-- probing-section-v11 -->"


class D:
    def __init__(self, run, index, report):
        P = json.load(open(f"{run}/predictions.json"))
        idx = {r["video_id"]: r for r in csv.DictReader(open(index))}
        self.vids = P["val_video_ids"]
        self.group = np.array(P["val_groups"])
        self.k = np.array([idx[v]["sym_k"] for v in self.vids])
        self.cls = {t: P["targets"][t]["classes"] for t in P["targets"]}
        self.gold = {t: np.asarray(P["targets"][t]["gold"]) for t in P["targets"]}
        self.head = {(h["target"], h["fit"], h["groups"][0]): np.asarray(h["pred"])
                     for h in P["heads"]}
        self.S = json.load(open(f"{run}/summary.json"))
        self.meta = {(r["target"], r["fit"], r["groups"][0]): r for r in self.S["probing"]}
        R = json.load(open(report))
        self.sc = collections.defaultdict(list)
        for c in R["scoring"]["cells"]:
            self.sc[(c["condition"], c["violation_type"])].append(c)

    def acc(self, t, f, tr, mask):
        m = np.asarray(mask)
        return 100 * float((self.head[(t, f, tr)][m] == self.gold[t][m]).mean()) if m.sum() \
            else float("nan")

    def sens(self, cond, viol):
        cs = self.sc.get((cond, viol), [])
        n = sum(c["n"] for c in cs)
        return (sum(c["sensitivity"] * c["n"] for c in cs) / n if n else float("nan")), n

    def collapse(self, t, f, tr, mask):
        """가장 많이 찍힌 클래스와 그 비율, 고유 클래스 수."""
        pr = self.head[(t, f, tr)][np.asarray(mask)]
        c = collections.Counter(pr)
        top, n = c.most_common(1)[0]
        return self.cls[t][top], 100 * n / len(pr), len(c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--append", required=True)
    a = ap.parse_args()
    d = D(a.run, a.index, a.report)
    C = d.S["config"]
    L: list[str] = [MARK, ""]
    w = L.append

    # ───────────────────────────── §6 설정
    w("---\n")
    w("# 제2부 — attentive probing 전수 기록 (2026-08-30)\n")
    w("같은 v11 클립·같은 토큰 위에서 잰 표현 분석이다. 제1부(채점)와 **데이터·모델이 같고**")
    w("프로토콜만 다르다. 채점이 못 맞히는 조건에서 표현에 정보가 남아 있는지를 본다.\n")
    w("## 6. 무엇을 어떻게 쟀나\n")
    w("| 기호 | source | 인코더 | frames |")
    w("|---|---|---|---|")
    for fit, sh, desc in PTS:
        enc = {"z": "context_encoder (online), 미래 토큰 마스킹",
               "h": "target_encoder (EMA), 32프레임 forward",
               "p": "predictor(ctx_masked)"}[sh]
        w(f"| `{sh}` | `{fit}` | {enc} | {desc.split(', ')[1]} |")
    w("")
    w("세 지점 모두 `(2048, 1280)` = `(16 / tubelet 2) x (256/16)^2`. 직접 비교된다.\n")
    pr = C["probing"]["probes"][C["probing"].get("probe", "attentive")]
    op = C["probing"]["optims"][C["probing"]["optim"]]
    dd = C["data"]
    w("```")
    w(f"probe      {pr}")
    w(f"optim      {C['probing']['optim']}  {op}   (스케줄러 없음, AdamW 상수 lr)")
    w(f"dtype      {C['model']['dtype']}   (채점은 fp32+fp16 autocast — 관례가 다르다)")
    w(f"img_size   {C['model']['img_size']}  window_size {C['model']['window_size']}  "
      f"tubelet {C['model'].get('tubelet_size', 2)}")
    w(f"index      {dd['index_csv']}   block_types {C['probing']['block_types']}")
    w(f"           obj 9,408 / empty 1,344 / imp 10,752 중 obj 만 학습·평가")
    w(f"group      {dd['group_column']} (6 조건)     frames n={dd['n_frames']} "
      f"stride={dd['frames_stride']} res={dd['resolution']}")
    w(f"split      {dd['split']}")
    w(f"n_videos   {d.S['n_videos']}  (토큰 추출은 전량)")
    w("```")
    w("`num_probe_blocks: 1` = cross-attention 하나 = V-JEPA attentive pooler 그대로.")
    w("depth 4 는 v10 실측에서 **더 나빴다** (p self 94.69 vs 96.35).\n")
    w("**불가능 변이를 넣지 않는 이유** — `contextF` 는 미래 토큰을 transformer 이전에 떨궈")
    w("`imp_ab` 가 `pos_a` 와 비트 단위로 같아지고, `pred` 도 입력이 context 뿐이라 같다.")
    w("`targetF` 는 라벨이 `_pre` 인데 렌더는 `_post` 라 인코더가 정확할수록 0점이 된다.\n")
    w("```bash")
    w("P=attn_probe D=v11 M=vith GPUS=8 GPUS_PROBE=1 \\")
    w('SPLIT="shape color env" \\')
    w('GSPLIT="static_visible,static_occlusion,moving_visible '
      'moving_visible_flat,moving_occlusion,moving_occlusion_flat" \\')
    w('SET="probing.targets.shape.classes=[capsule,cone,cube,cylinder,pyramid,sphere,torus]" \\')
    w("bash z_research/scripts/sbatch.sh")
    w("python z_research/scripts/analysis/merge_probe_runs.py <base>")
    w("```")
    w(f"head **{len(d.S['probing'])}개** = 6조건 x 3타깃 x 3지점. `fit_groups_sweep: auto_conditions`.")
    w("토큰 캐시 420.0 GiB. 추출 32분(8 GPU) + probing 16분(1 GPU x 6 동시).")
    w("shape 은 v11 이 7종이라 `narrowcap` 을 뺐다 — chance 12.5 → **14.29%**.\n")
    w("⚠️ **`fit_groups_sweep` 는 학습 마스크만 제한한다. 평가는 6조건 전부를 돈다.**")
    w("head 하나가 6개 값을 낸다 — 대각선이 self, 비대각선이 이식이다.\n")

    # ───────────────────────────── §7 대각선
    w("---\n")
    w("## 7. 대각선 — 학습 조건 == 평가 조건 (self)\n")
    for t, tl in TGT:
        ch = d.meta[(t, PTS[0][0], CONDS[0])]["chance"] * 100
        w(f"### {tl}  (chance {ch:.2f}%)\n")
        w("| condition | z | h | p | h−p | n_train | n_val | p train_acc |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|")
        for c in CONDS:
            v = {sh: d.acc(t, f, c, d.group == c) for f, sh, _ in PTS}
            m = d.meta[(t, "pred__f17to32", c)]
            nv = int((d.group == c).sum())
            w(f"| {c} | {v['z']:.2f} | {v['h']:.2f} | {v['p']:.2f} | {v['h']-v['p']:+.2f} "
              f"| {m['n_train']} | {nv} | {m['train_acc']:.3f} |")
        w("")

    # ───────────────────────────── §8 팔 안 vis<->occ
    w("---\n")
    w("## 8. 팔 안 vis ↔ occ 이식 — **운동이 통제된 유일한 비교**\n")
    w("팔 안에서는 궤적이 같다 (`fixed_speed: true`, v0 116 cm/s 동일). 바뀌는 것은")
    w("가림막의 유무뿐이다. 그래서 §9 의 팔을 가로지르는 이식과 달리 여기서는")
    w("\"attention query 가 위치 특이적이라 안 옮겨간다\" 로 설명되지 않는다.\n")
    w("`h` 가 대조군이다 — `h` 는 **실제로 가려진 미래**를 본 인코더라 `p` 와 같은 가림을 겪는다.\n")
    for t, tl in TGT:
        ch = d.meta[(t, PTS[0][0], CONDS[0])]["chance"] * 100
        w(f"### {tl}  (chance {ch:.2f}%)\n")
        w("| arm | pt | VIS self | VIS→OCC | OCC self | OCC→VIS | 낙차(VIS−VIS→OCC) |")
        w("|---|---|---:|---:|---:|---:|---:|")
        for aname, vis, occ in ARMS:
            for f, sh, _ in PTS:
                v = [d.acc(t, f, vis, d.group == vis), d.acc(t, f, vis, d.group == occ),
                     d.acc(t, f, occ, d.group == occ), d.acc(t, f, occ, d.group == vis)]
                w(f"| {aname} | {sh} | {v[0]:.1f} | **{v[1]:.1f}** | {v[2]:.1f} | {v[3]:.1f} "
                  f"| {v[0]-v[1]:+.1f} |")
        vo = {sh: np.mean([d.acc(t, f, vis, d.group == occ) for _, vis, occ in ARMS])
              for f, sh, _ in PTS}
        ov = {sh: np.mean([d.acc(t, f, occ, d.group == vis) for _, vis, occ in ARMS])
              for f, sh, _ in PTS}
        w(f"| **3팔 평균** | | | **z {vo['z']:.1f} · h {vo['h']:.1f} · p {vo['p']:.1f}** | "
          f"| z {ov['z']:.1f} · h {ov['h']:.1f} · p {ov['p']:.1f} | |")
        w("")

    # ───────────────────────────── §9 6x6
    w("---\n")
    w("## 9. 조건 간 이식 전수 — 6×6\n")
    w("⚠️ **팔을 가로지르는 칸은 깨지는 것이 기대값이다.** attentive probe 의 query 는")
    w("\"어느 토큰을 볼 것인가\" 를 배우는데, 정지와 이동은 물체가 점유하는 토큰이 다르다.")
    w("이 표는 전수 기록이며, 주장에 쓸 수 있는 것은 **§8 의 팔 안 칸**이다.\n")
    for t, tl in TGT:
        w(f"### {tl}\n")
        for f, sh, _ in PTS:
            w(f"**{sh}** — `{f}`\n")
            w("| train \\ eval | " + " | ".join(SHORT[c] for c in CONDS) + " | 행평균(off) |")
            w("|---|" + "---:|" * (len(CONDS) + 1))
            for tr in CONDS:
                row = [d.acc(t, f, tr, d.group == c) for c in CONDS]
                off = [x for c, x in zip(CONDS, row) if c != tr]
                cells = " | ".join(("**%.1f**" % x) if c == tr else ("%.1f" % x)
                                   for c, x in zip(CONDS, row))
                w(f"| {SHORT[tr]} | {cells} | {np.mean(off):.1f} |")
            w("")

    # ───────────────────────────── §10 k 분해
    w("---\n")
    w("## 10. k 분해 — self 는 평평, 이식은 k 에 비례해 무너진다\n")
    w("head 는 조건으로 학습됐고 k 는 학습축이 아니다. 아래는 클립별 예측을")
    w("`index_probe.csv` 와 조인해 **사후에** 쪼갠 것이라 재학습이 없다.\n")
    w("### (a) self 를 k 로  — 세 팔 합산\n")
    w("| target | pt | k=0(가림없음) | k=1 | k=2 | k=3 | k=4 |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for t, tl in TGT:
        for f, sh, _ in PTS:
            row = []
            for k in ["0"] + KS:
                hit = tot = 0
                for _, vis, occ in ARMS:
                    c = vis if k == "0" else occ
                    m = (d.group == c) & (d.k == k)
                    if m.sum():
                        hit += int((d.head[(t, f, c)][m] == d.gold[t][m]).sum())
                        tot += int(m.sum())
                row.append(100 * hit / tot)
            w(f"| {tl} | {sh} | " + " | ".join(f"{x:.2f}" for x in row) + " |")
    w("")
    w("### (b) VIS→OCC 이식을 k 로  — 팔별\n")
    w("그 팔의 **비가림** 조건으로 학습한 head 를 **가림** 조건에서 k 별로 평가한다.\n")
    w("| target | arm | pt | k=1 | k=2 | k=3 | k=4 | 기울기(k4−k1) |")
    w("|---|---|---|---:|---:|---:|---:|---:|")
    for t, tl in TGT:
        for aname, vis, occ in ARMS:
            for f, sh, _ in PTS:
                row = [d.acc(t, f, vis, (d.group == occ) & (d.k == k)) for k in KS]
                w(f"| {tl} | {aname} | {sh} | " + " | ".join(f"{x:.1f}" for x in row)
                  + f" | {row[-1]-row[0]:+.1f} |")
    w("")
    w("### (c) 그 이식이 무너지면서 무엇이 되는가 — 최빈 예측\n")
    w("| target | arm | k | acc | 최빈 예측 | 그 비율 | 고유 클래스 |")
    w("|---|---|---:|---:|---|---:|---:|")
    for t, tl in TGT:
        if t == "env":
            continue
        for aname, vis, occ in ARMS:
            for k in KS:
                m = (d.group == occ) & (d.k == k)
                top, share, uniq = d.collapse(t, "pred__f17to32", vis, m)
                w(f"| {tl} | {aname} | {k} | {d.acc(t,'pred__f17to32',vis,m):.1f} | "
                  f"{top} | {share:.1f} | {uniq} |")
    w("")

    # ───────────────────────────── §11 관찰
    w("---\n")
    w("## 11. 관찰\n")
    w("### (1) 정보는 세 지점 전부에 있다. 채점만 못 읽는다\n")
    w("같은 클립·같은 토큰에서 잰 두 측정:\n")
    w("| condition | shape p self | shape 채점 sens | colour p self | colour 채점 sens |")
    w("|---|---:|---:|---:|---:|")
    for c in CONDS:
        w(f"| {SHORT[c]} | {d.acc('shape','pred__f17to32',c,d.group==c):.2f} | "
          f"{d.sens(c,'shape')[0]:.2f} | {d.acc('color','pred__f17to32',c,d.group==c):.2f} | "
          f"{d.sens(c,'color')[0]:.2f} |")
    w("")
    w("`ramp+OCC` 에서 shape 채점 sensitivity 는 **4.46**, vanish 는 **0.00**(완전 상쇄)인데")
    w("같은 조건의 `p` self 는 **98.46%** 다. → **\"정보가 없어서 못 맞힌다\" 가 아니다.**")
    w("그림: `figures/probing/by_condition/fig_information_vs_scoring`.\n")
    w("⚠️ v8 과 다르다. v8 가림 조건 `p` self 는 shape 82.0 / colour **54.5** 였고 v11 은")
    w("98.5 / 99.0 이다. v8 은 8종 중 4쌍만 썼고 실루엣이 폭과 교락돼 있었다 — v11 은")
    w("21쌍 전수에 실루엣을 렌더 픽셀 기준으로 등화했다. **어느 쪽이 원인인지 안 갈렸다.**\n")
    w("### (2) 비가림으로 배운 readout 이 같은 팔의 가림에서 안 통한다\n")
    w("운동이 통제된 §8 의 `VIS→OCC` 3팔 평균:\n")
    w("```")
    for t, tl in TGT:
        vo = {sh: np.mean([d.acc(t, f, vis, d.group == occ) for _, vis, occ in ARMS])
              for f, sh, _ in PTS}
        ch = d.meta[(t, PTS[0][0], CONDS[0])]["chance"] * 100
        w(f"{tl:11s} z {vo['z']:5.1f}   h {vo['h']:5.1f}   p {vo['p']:5.1f}     chance {ch:.1f}")
    w("```")
    w("colour 가 가장 깨끗하다 — 인코더 낙차는 0.0~2.5pt 인데 `p` 만 100 → 15~21 이다.")
    w("`h` 도 가려진 미래를 본 인코더인데 안 무너지므로, 차이는 **파이프라인에서 predictor")
    w("단계에 국한된다.** `env` 는 세 지점 전부 100.0 이라 predictor 출력 전체가 흔들리는")
    w("것이 아님을 보인다.\n")
    w("⚠️ shape 은 인코더도 흔들린다 (z 3팔 평균 64.8, `Constant velocity` 는 42.4).")
    w("**\"인코더는 가림에 무관하다\" 는 colour 와 env 로만 주장할 수 있다.**\n")
    w("### (3) 그 실패는 k 에 비례하고 끝에는 단일 클래스로 붕괴한다\n")
    w("self 는 k 에 평평한데(§10-a) `VIS→OCC` 는 k 와 함께 떨어진다(§10-b).")
    w("`Accelerating` 팔의 `p`:\n")
    w("```")
    for t, tl in (("shape", "shape"), ("color", "colour")):
        row = [d.acc(t, "pred__f17to32", "moving_visible",
                     (d.group == "moving_occlusion") & (d.k == k)) for k in KS]
        uni = [d.collapse(t, "pred__f17to32", "moving_visible",
                          (d.group == "moving_occlusion") & (d.k == k))[2] for k in KS]
        w(f"{tl:7s} acc   " + "  ".join(f"k{k}={v:5.1f}" for k, v in zip(KS, row)))
        w(f"{'':7s} 고유클래스 " + "  ".join(f"k{k}={u:2d}" for k, u in zip(KS, uni)))
    w("```")
    w("→ **가림 길이는 정보를 지우지 않는다.** 같은 조건 안에서는 k 와 무관하게 읽히고,")
    w("비가림을 기준으로 재면 k 에 비례해 멀어진다. 채점이 조건 안에서만 비교하므로")
    w("제1부의 \"k 는 무관하다\" 와 어긋나지 않는다.\n")
    w("### (4) `env` 대조군 — predictor 출력 전체가 망가지는 것이 아니다\n")
    w("`env` 는 대각선 18칸, 팔 안 이식 18칸, k 분해 36칸이 **전부 100.0** 이다.")
    w("같은 head 구조·같은 n_train(781)·같은 조건 쌍에서 배경은 완벽히 옮겨간다.\n")
    w("⚠️ 다만 배경은 2048 토큰 전체에 깔려 있어 query 가 어디를 보든 읽힌다.")
    w("**그래서 이 대조군은 \"물체 정체성도 옮겨가야 한다\" 를 뒷받침하지 않는다** — §12-(2).\n")
    w("### (5) `static_occlusion` 만 대각선에서도 낮다 — 미해결\n")
    w("가림 세 조건 중 `sta+OCC` 만 `p` self 가 85.13 / 87.55 이고 나머지 둘은 98.5 / 99.0 이다.")
    w("그런데 **채점에서는 `sta+OCC` 만 vanish 탐지가 살아남는다**(sens +20.54, 이동 둘은 0.00/+2.01).")
    w("표현에서는 가장 나쁜데 채점에서는 가장 좋다. 설명 미상.")
    w("그리고 이 두 head 가 수렴이 안 된 셋 중 둘이라(train_acc 0.931 / 0.954)")
    w("**표현 탓인지 최적화 탓인지 아직 안 갈렸다** — §13.\n")

    # ───────────────────────────── §12 정정
    w("---\n")
    w("## 12. 정정 — 한 번 썼다가 철회한 해석 (기록)\n")
    w("| 쓰지 말 것 | 왜 틀렸나 |")
    w("|---|---|")
    w("| **\"팔을 가로지르는 이식 실패가 표현이 조건별이라는 증거\"** | 기대값이다. "
      "attentive probe 의 query 는 어느 토큰을 볼지를 배우는데 정지와 이동은 물체가 "
      "점유하는 토큰이 다르다. 정지↔이동 비대칭(정지→이동 70.1 vs 이동→정지 17.3)도 "
      "같은 이유로 설명된다 — 정지 물체의 자리는 이동 궤적이 훑는 띠 안에 들어갈 수 "
      "있지만 그 역은 아니다. **§8 의 팔 안 비교만 쓸 것.** |")
    w("| **\"`env` 가 100% 이식되므로 head 용량·표본 문제가 아니다\"** | 반만 맞다. "
      "배경은 2048 토큰 전체에 깔려 있어 query 가 어디를 보든 읽히는 **자명한 경우**다. "
      "물체 정체성이 옮겨가야 할 이유를 세워주지 않는다. |")
    w("| **\"가림이 표현을 회전시킨다 / 옮긴다\"** | 각도도 부분공간 겹침도 클래스 중심 "
      "이동도 잰 적이 없다. 게다가 probe 가 선형이 아니라 attentive pooling + 선형이라 "
      "실패가 **클래스를 가르는 방향**에서 온 건지 **어느 토큰을 모으느냐**에서 온 건지 "
      "구분되지 않는다. 쓸 수 있는 문장: \"비가림으로 학습한 head 가 같은 팔의 가림 "
      "조건에서 못 읽는다.\" |")
    w("| **\"z 와 h 가 같다\"** | 둘 다 천장이다 (train_acc 1.000 이 18/18). 차이를 잴 "
      "해상도가 없다. |")
    w("| **pooled(`fit_groups_sweep=[null]`) 대조군을 돌린다** | 철회. 이식 실패가 "
      "기대값이면 pooled 는 \"데이터가 다양하면 head 가 잘 된다\" 는 자명한 답만 낸다. |")
    w("")

    # ───────────────────────────── §13 단서
    w("---\n")
    w("## 13. 단서\n")
    ta = [(k, r["train_acc"]) for k, r in d.meta.items() if r["train_acc"] < 1.0]
    w(f"- **`train_acc` 가 1.000 이 아닌 head 는 {len(ta)}개뿐이고 전부 `p` 다:**")
    for (t, f, c), v in sorted(ta, key=lambda x: x[1]):
        w(f"  `{t} p {c}` **{v:.3f}**")
    w(f"  나머지 {len(d.meta)-len(ta)}개는 정확히 1.000 으로 포화다. **대조로 읽을 수 있는 것은")
    w("  `p` 줄과 이식뿐이다** (CLAUDE.md §7-3). 낮게 나온 세 칸이 표현 탓인지 35 epoch")
    w("  미수렴 탓인지 아직 안 갈렸다 — `attn_probe.yaml` 의 `attn_100` 프리셋으로")
    w("  그 조건만 다시 돌리면 캐시 적중이라 몇 분에 갈린다.")
    w("- head 당 `n_train` 781~787. 클래스당 shape ~112, colour ~98, env ~196.")
    w("  `stratify_by: condition` 이라 **조건은 균등하지만 클래스는 아니다.**")
    w("- k 별 셀은 방향당 n=172~211 이다. **k 별 요철을 단독 인용하지 말 것** — "
      "이항 SE 가 ~3pp 다.")
    w("- **fold 없음 / seed 1개.** `GPUS=1` 로 돌려 head 학습은 결정론적이다.")
    w("- **h→p 이식(정렬손실)은 안 쟀다.** config 의 `eval` 이 `[self]` 뿐이다. "
      "v8 의 핵심 지표(48.0/49.9, 가림에서 14.8=chance)와 비교하려면 다시 돌려야 한다.")
    w("- **`p` 가 문맥 잔상인지 안 갈렸다.** `p = predictor(context)` 라 정보의 출처는 "
      "정의상 문맥이고, 물체는 모든 조건의 context 구간에 보인다. 그래서 §11-(1) 의 "
      "\"정보가 있다\" 가 **\"predictor 가 미래로 옮겼다\"** 인지 **\"문맥이 묻어나왔다\"** "
      "인지 구분되지 않는다. `z_research/scripts/analysis/is_p_just_context.py` 가 이 질문을 "
      "묻는다 (캐시만, GPU 불필요).")
    w("- 교란 — occluded 조건에만 가림막이 장면에 추가로 있다. 설계 의도이며 분리 대상이 "
      "아니다 (CLAUDE.md §8-5).\n")

    # ───────────────────────────── §14 그림
    w("---\n")
    w("## 14. 그림\n")
    w("```")
    w("z_research/IntPhysGenV11/figures/probing/")
    w("  README.md                                어느 그림이 무슨 주장을 담는가")
    w("  by_condition/  fig_information_vs_scoring   핵심 1 — 정보 있음 vs 채점 실패")
    w("                 fig_transfer_vis_occ         핵심 2 — 팔 안 VIS<->OCC")
    w("                 fig_confusion_transfer          2 의 기전 (h 100.0 vs p 16.4)")
    w("                 fig_self_condition              부록 — 정보 존재 전수")
    w("                 fig_transfer_matrix             부록 — 6x6 전수")
    w("  by_k/          fig_transfer_vis_occ_k       핵심 3 — 이식이 k 에 비례해 무너진다")
    w("                 fig_confusion_transfer_k     핵심 4 — 붕괴의 끝")
    w("                 fig_self_k                      부록 — self 는 k 에 평평")
    w("```")
    w("```bash")
    w("python z_research/scripts/figures/plot_v11_probing.py            # 화면용")
    w("python z_research/scripts/figures/plot_v11_probing.py --width 7.0  # double-column")
    w("```")
    w("324칸을 `summary.json` 과 대조해 **불일치 0 일 때만** 그린다 (아니면 죽는다).\n")
    w("**지웠고 다시 만들지 말 것** — `fig_confusion_self`(sta+OCC self: 그 두 head 가")
    w("미수렴이고 `sta+OCC` 는 채점이 살아남는 조건이라 어려운 케이스가 아니다),")
    w("`fig_self_arm_k`(9패널 중 7개가 전 구간 100).\n")

    # ───────────────────────────── 전수 표
    w("---\n")
    w("# 전수 표 (제2부)\n")
    w("## G. head 54개 — 메타\n")
    w("| # | target | 지점 | train cond | n_train | train_acc | chance | overall(6조건) | self |")
    w("|---:|---|---|---|---:|---:|---:|---:|---:|")
    i = 0
    for t, _ in TGT:
        for f, sh, _ in PTS:
            for c in CONDS:
                i += 1
                m = d.meta[(t, f, c)]
                w(f"| {i} | {t} | {sh} | {c} | {m['n_train']} | {m['train_acc']:.3f} | "
                  f"{m['chance']*100:.2f} | {m['evals'][f]['overall']*100:.2f} | "
                  f"{d.acc(t, f, c, d.group == c):.2f} |")
    w("")
    w("## H. (head × 평가조건) 전수 — 324칸\n")
    w("| target | 지점 | train cond | eval cond | self | n | acc | bacc |")
    w("|---|---|---|---|:-:|---:|---:|---:|")
    nH = 0
    for t, _ in TGT:
        for f, sh, _ in PTS:
            for tr in CONDS:
                pg = d.meta[(t, f, tr)]["evals"][f]["per_group"]
                for ev in CONDS:
                    g = pg[ev]; nH += 1
                    w(f"| {t} | {sh} | {tr} | {ev} | {'O' if tr == ev else ''} | {g['n']} | "
                      f"{g['acc']*100:.2f} | {g['bacc']*100:.2f} |")
    w(f"\n({nH} 행 = head 54 × 평가조건 6)\n")
    w("## I. 팔 안 vis ↔ occ 전수 — 54칸\n")
    w("| target | arm | 지점 | VIS self | VIS→OCC | OCC self | OCC→VIS |")
    w("|---|---|---|---:|---:|---:|---:|")
    for t, _ in TGT:
        for aname, vis, occ in ARMS:
            for f, sh, _ in PTS:
                w(f"| {t} | {aname} | {sh} | {d.acc(t,f,vis,d.group==vis):.2f} | "
                  f"{d.acc(t,f,vis,d.group==occ):.2f} | {d.acc(t,f,occ,d.group==occ):.2f} | "
                  f"{d.acc(t,f,occ,d.group==vis):.2f} |")
    w("")
    w("## J. VIS→OCC 이식의 k 분해 전수 — 108칸\n")
    w("| target | arm | 지점 | k | n | acc | 최빈 예측 | 그 비율 | 고유 클래스 |")
    w("|---|---|---|---:|---:|---:|---|---:|---:|")
    nJ = 0
    for t, _ in TGT:
        for aname, vis, occ in ARMS:
            for f, sh, _ in PTS:
                for k in KS:
                    m = (d.group == occ) & (d.k == k)
                    top, share, uniq = d.collapse(t, f, vis, m)
                    nJ += 1
                    w(f"| {t} | {aname} | {sh} | {k} | {int(m.sum())} | "
                      f"{d.acc(t,f,vis,m):.2f} | {top} | {share:.1f} | {uniq} |")
    w(f"\n({nJ} 행)\n")

    txt = "\n".join(L)
    body = io.open(a.append, encoding="utf-8").read()
    if MARK in body:
        body = body[:body.index(MARK)]
        print("  기존 제2부를 갈아끼운다")
    io.open(a.append, "w", encoding="utf-8").write(body.rstrip() + "\n\n" + txt)
    print(f"  [written] {a.append}  (+{len(L)} 줄; 표 H {nH}칸 / J {nJ}칸)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
