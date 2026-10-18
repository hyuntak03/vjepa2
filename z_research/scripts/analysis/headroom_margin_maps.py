#!/usr/bin/env python3
"""matched 쌍 margin (S_imp − S_pos) 이 토큰 지도의 어디서 오나 — design_headroom 의 E_* 오차 지도 (2026-10-13, DESIGN_CHOICES_2026-10-13 §2-4 · §2-5).
part_1 = 물체 영역 (반경 1) 안 / 밖 기여 · part_2 = 반경별 몫 · part_3 = 물체로부터 거리별 토큰당 margin (early, copy 렌더).
  python headroom_margin_maps.py      (design_headroom.py extract 가 먼저)
"""
import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[3])
def part_1():
    """안 / 밖 기여 (반경 1)"""
    import sys, csv, json, numpy as np
    from pathlib import Path
    from collections import defaultdict
    sys.path.insert(0,"z_research/scripts/analysis")
    import design_headroom as H
    Z=H.load("vobench"); vid={v:i for i,v in enumerate(Z["video_id"])}
    rows={r["video_id"]:r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
    blk=defaultdict(dict)
    for v,r in rows.items(): blk[r["source_block"]][r["variant"]]=v
    def fmask(lab, arm, dil=1):
        M=np.zeros((8,16,16),bool); L=lab.get(arm)
        if not L or L[0] is None: return M
        P=np.array([[L[3*s]["u"],L[3*s]["v"]] for s in range(32)])[16:].reshape(8,2,2).mean(1)
        for k,p in enumerate(P):
            if not (0<=p[0]<512 and 0<=p[1]<512): continue
            y,x=H.tok(p); M[k,max(y-dil,0):y+dil+1,max(x-dil,0):x+dil+1]=True
        return M
    out=defaultdict(lambda: defaultdict(list))
    for sb,arms in blk.items():
        r=rows[arms["pos_a"]]; mot="static" if r["motion"]=="flat_static" else "moving"
        lab=json.load(open(H.FR/Path(r["file_name"]).parent/"label.json"))["labels"]
        for pos,imp,d,fa,fb in (("pos_a","imp_ab","A","A","B"),("pos_b","imp_ba","B","B","A")):
            M=fmask(lab,fa)|fmask(lab,fb)
            key=f"{r['violation']}/{r['occlusion']}/{mot}/{d}"
            for m in ("copy","ar","move_or"):
                dE=Z[f"E_{m}"][vid[arms[imp]]].astype(np.float32)-Z[f"E_{m}"][vid[arms[pos]]].astype(np.float32)   # >0 = 정답 쪽
                out[key][m+"_in"].append(dE[M].sum()/dE.size); out[key][m+"_out"].append(dE[~M].sum()/dE.size)
            out[key]["frac_area"].append(M.mean())
    print("셀 · 방향: (S_imp − S_pos) 를 물체 영역 (두 미래의 물체 3×3 합집합) 안 / 밖 기여로 — 양수 = 정답 쪽. 평균 ×1e3")
    for key in sorted(out):
        if "/late/" not in key and "/early/" not in key: continue
        o=out[key]; print(f"  {key:30s} area {np.mean(o['frac_area']):.3f} | " + "  ".join(f"{m}: in {1e3*np.mean(o[m+'_in']):+6.2f} out {1e3*np.mean(o[m+'_out']):+6.2f}" for m in ("copy","ar","move_or")))

def part_2():
    """반경별 몫"""
    import sys, csv, json, numpy as np
    from pathlib import Path
    from collections import defaultdict
    sys.path.insert(0,"z_research/scripts/analysis")
    import design_headroom as H
    Z=H.load("vobench"); vid={v:i for i,v in enumerate(Z["video_id"])}
    rows={r["video_id"]:r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
    blk=defaultdict(dict)
    for v,r in rows.items(): blk[r["source_block"]][r["variant"]]=v
    rpx=defaultdict(list)
    def fmask(lab, arm, dil):
        M=np.zeros((8,16,16),bool); L=lab.get(arm)
        if not L or L[0] is None: return M
        P=np.array([[L[3*s]["u"],L[3*s]["v"]] for s in range(32)])[16:].reshape(8,2,2).mean(1)
        for k,p in enumerate(P):
            if not (0<=p[0]<512 and 0<=p[1]<512): continue
            y,x=H.tok(p); M[k,max(y-dil,0):y+dil+1,max(x-dil,0):x+dil+1]=True
        return M
    out=defaultdict(lambda: defaultdict(list))
    for sb,arms in blk.items():
        r=rows[arms["pos_a"]]; mot="static" if r["motion"]=="flat_static" else "moving"
        lab=json.load(open(H.FR/Path(r["file_name"]).parent/"label.json"))["labels"]
        if lab.get("A") and lab["A"][0]: rpx[r["objset"]].append(lab["A"][0].get("r_px",np.nan))
        for pos,imp,d,fa,fb in (("pos_a","imp_ab","A","A","B"),("pos_b","imp_ba","B","B","A")):
            key=f"{r['violation']}/{r['occlusion']}/{mot}"
            dE={m:Z[f"E_{m}"][vid[arms[imp]]].astype(np.float32)-Z[f"E_{m}"][vid[arms[pos]]].astype(np.float32) for m in ("copy","ar")}
            for dil in (1,2,3,5):
                M=fmask(lab,fa,dil)|fmask(lab,fb,dil)
                for m in dE:
                    tot=dE[m].sum()
                    out[key][f"{m}_d{dil}"].append(dE[m][M].sum()); out[key][f"{m}_tot"].append(tot)
                out[key][f"area_d{dil}"].append(M.mean())
    print("r_px (512 기준) 평균:", {k:round(float(np.nanmean(v)),1) for k,v in rpx.items()}, " 1 토큰 = 32 px")
    print("물체 영역 반경별, 총 margin 중 영역 안 몫 (Σ in / Σ total) — copy | ar   [영역 면적]")
    for key in sorted(out):
        o=out[key]
        s=f"  {key:28s}"
        for dil in (1,2,3,5):
            s+=f"  r{dil}({np.mean(o[f'area_d{dil}']):.2f}) " + " ".join(f"{np.sum(o[f'{m}_d{dil}'])/np.sum(o[f'{m}_tot']):+.2f}" for m in ("copy","ar"))
        print(s)

def part_3():
    """거리별 토큰당 margin (early, copy)"""
    import sys, csv, json, numpy as np
    from pathlib import Path
    from collections import defaultdict
    sys.path.insert(0,"z_research/scripts/analysis")
    import design_headroom as H
    Z=H.load("vobench"); vid={v:i for i,v in enumerate(Z["video_id"])}
    rows={r["video_id"]:r for r in csv.DictReader(open(H.SETS["vobench"]["index"]))}
    blk=defaultdict(dict)
    for v,r in rows.items(): blk[r["source_block"]][r["variant"]]=v
    yy,xx=np.mgrid[0:16,0:16]
    def dist(lab, arms):
        D=np.full((8,16,16),99)
        for arm in arms:
            L=lab.get(arm)
            if not L or L[0] is None: continue
            P=np.array([[L[3*s]["u"],L[3*s]["v"]] for s in range(32)])[16:].reshape(8,2,2).mean(1)
            for k,p in enumerate(P):
                y,x=H.tok(p); D[k]=np.minimum(D[k],np.maximum(abs(yy-y),abs(xx-x)))
        return D
    prof=defaultdict(lambda: defaultdict(list)); grid=defaultdict(lambda: np.zeros((16,16))); n=defaultdict(int)
    for sb,arms in blk.items():
        r=rows[arms["pos_a"]]
        if r["occlusion"]!="early": continue
        mot="static" if r["motion"]=="flat_static" else "moving"
        lab=json.load(open(H.FR/Path(r["file_name"]).parent/"label.json"))["labels"]
        for pos,imp,fa,fb in (("pos_a","imp_ab","A","B"),("pos_b","imp_ba","B","A")):
            D=dist(lab,(fa,fb)); key=f"{r['objset']}/{r['violation']}/{mot}"
            dE=Z["E_copy"][vid[arms[imp]]].astype(np.float32)-Z["E_copy"][vid[arms[pos]]].astype(np.float32)
            for d in range(0,12): 
                m=D==d
                if m.any(): prof[key][d].append(dE[m].mean())
            grid[key]+=dE.mean(0); n[key]+=1
    print("early 셀, copy 렌더: 토큰당 margin (S_imp−S_pos 기여, ×1e3) vs 물체로부터 체비셰프 거리 (토큰)")
    for key in sorted(prof):
        print(f"  {key:30s} "+" ".join(f"{d}:{1e3*np.mean(prof[key][d]):5.1f}" for d in range(12) if len(prof[key][d])>20))
    for key in ("primitive/permanence/moving","realistic/shape/moving"):
        g=grid[key]/n[key]*1e3; print(key, "평균 margin 지도 (×1e3), 행=y"); print(np.array2string(g,precision=0,max_line_width=200,suppress_small=True))

if __name__ == "__main__":
    for f in (part_1, part_2, part_3):
        print("\n#", f.__doc__); f()
