"""Render audited overhead tracks, physical clearances and the recorded policy depth views."""
import argparse
from itertools import combinations, product
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geometry_audit import build_geometry


SIZE = (1280,720)
COLORS = ["#c84440","#22844e","#366ebd"]
ZH = {"success":"通过","collision":"碰撞","timeout":"超时","bypass":"绕行","incomplete":"未完成"}
FONTS = {i:ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",i,index=2)
         for i in (15,17,19,22,26,31)}
LEFT,TOP=44,103
MW,MH=880,484


def projection(positions):
    xmin=min(-2.,float(positions[...,0].min())-.5)
    xmax=max(14.5,float(positions[...,0].max())+.5)
    ymin=min(-4.5,float(positions[...,1].min())-.5)
    ymax=max(4.5,float(positions[...,1].max())+.5)
    scale=min(MW/(xmax-xmin),MH/(ymax-ymin))
    def project(x,y):
        return round((x-.5*(xmin+xmax))*scale+MW/2),round((.5*(ymin+ymax)-y)*scale+MH/2)
    return project,(xmin,xmax,ymin,ymax)


def text(draw,x,y,s,size=19,color="#25333d"):
    draw.text((x,y),str(s),font=FONTS[size],fill=color,stroke_width=0)


def wall_map(geometry,project,limits):
    canvas=Image.new("RGB",(MW,MH),"#f4f6f7")
    d=ImageDraw.Draw(canvas)
    xmin,xmax,ymin,ymax=limits
    for x in range(int(np.ceil(xmin)),int(np.floor(xmax))+1):
        u,_=project(x,0)
        d.line((u,0,u,MH),fill="#e1e7e9")
        if u < MW-64:
            text(d,u+3,MH-24,str(x),15,"#66717a")
    for y in range(int(np.ceil(ymin)),int(np.floor(ymax))+1):
        _,v=project(0,y)
        d.line((0,v,MW,v),fill="#e1e7e9")
        if v > 33:
            text(d,3,v-19,str(y),15,"#66717a")
    for box in geometry.walls.values():
        vertices=(np.array(list(product([-1,1],repeat=3)))*box["half"])@box["axes"]+box["center"]
        pixels=np.int32([project(-p[2],-p[0]) for p in vertices])
        hull=cv2.convexHull(pixels).reshape(-1,2)
        d.polygon([tuple(map(int,p)) for p in hull],fill="#76828a")
    gx,_=project(10,0)
    for v in range(0,MH,13):
        d.line((gx,v,gx,v+7),fill="#746442",width=2)
    text(d,gx+5,6,"x = 10 m",17,"#685c42")
    text(d,9,5,"y（m）",17)
    text(d,MW-60,MH-27,"x（m）",17)
    return canvas


def distances(geometry,poses):
    wall=np.full(len(poses),np.inf)
    for agent in range(3):
        for box in geometry.walls.values():
            loc=geometry.legacy.to_box(poses[:,agent],box)
            solid=np.linalg.norm(np.maximum(np.abs(loc)-box["half"],0),axis=1)
            wall=np.minimum(wall,solid-.10)
    pair=np.min([np.linalg.norm(poses[:,a]-poses[:,b],axis=1) for a,b in combinations(range(3),2)],axis=0)
    return wall,pair


def encode(run,cohort,seed,kind,g):
    index=json.loads((cohort/"geometry.json").read_text())
    rec=next(v for v in index if v["seed"]==seed)
    with np.load(cohort/f"seed_{seed}.npz") as data:
        pos=data["control_xyz"]; sub=data["substep_xyz"]; depth=data["depth"]
    project,limits=projection(pos)
    base=wall_map(g,project,limits)
    wall,peer=distances(g,pos)
    initial=np.stack([np.array([project(x,y) for x,y,_ in pos[:,a]]) for a in range(3)])
    first=rec["first_contact"]
    path=run/"figures"/f"{kind}-seed{seed}.mp4"
    limit=len(pos)-1
    fps=12.5
    tool=subprocess.Popen(["ffmpeg","-loglevel","error","-y","-f","rawvideo","-pixel_format","rgb24","-video_size","1280x720",
                           "-framerate",str(fps),"-i","-","-an","-c:v","libx264","-threads","2","-crf","19","-preset","veryfast",
                           "-pix_fmt","yuv420p","-movflags","+faststart",str(path)],stdin=subprocess.PIPE)
    try:
        for k in range(limit+1):
            tile=base.copy()
            d=ImageDraw.Draw(tile)
            for a in range(3):
                pts=[tuple(p) for p in initial[a,:k+1]]
                if len(pts)>1:
                    d.line(pts,fill=COLORS[a],width=3)
                px,py=pts[-1]
                d.ellipse((px-5,py-5,px+5,py+5),fill=COLORS[a],outline="white",width=1)
                text(d,px+6,py-17,str(a+1),17,COLORS[a])
            if first and k>=first["step"]:
                for a in first["drones"]:
                    x,y,_=sub[first["sample"],a-1]
                    px,py=project(x,y)
                    d.ellipse((px-13,py-13,px+13,py+13),outline="#ae2c27",width=3)
            frame=Image.new("RGB",SIZE,"white"); d=ImageDraw.Draw(frame)
            text(d,LEFT,13,f"三机穿缝 · 种子 {seed}    原生{ZH[rec['native_outcome']]}  /  严验{ZH[rec['strict_outcome']]}",31)
            text(d,LEFT,64,f"俯视轨迹   |   {cohort.name}   |   控制步 {k:03d}/{limit}   ·   仿真时间 {k*.02:4.2f} s   ·   0.25 倍速",19)
            frame.paste(tile,(LEFT,TOP))
            text(d,954,64,"策略深度相机  ·  0–10 m",19)
            for a in range(3):
                y=103+164*a
                pixels=np.uint8(np.clip(1-depth[k,a,0]/10,0,1)*255)
                pic=Image.fromarray(pixels).resize((144,144),Image.Resampling.NEAREST if hasattr(Image,"Resampling") else Image.NEAREST)
                frame.paste(pic,(955,y))
                text(d,1110,y+11,f"{a+1} 号机",22,COLORS[a])
                text(d,1110,y+54,f"高 {pos[k,a,2]:.2f} m",17)
                text(d,1110,y+88,f"y {pos[k,a,1]:+.2f} m",17)
            text(d,953,595,"亮色较近，暗色较远",17,"#586572")
            text(d,44,595,f"此帧墙面净空 {wall[k]:+.3f} m     最近机间中心距 {peer[k]:.3f} m（限值 0.20 m）",22)
            if first and k>=first["step"]:
                labels={"wall_negative_y":"负 y 侧墙","wall_positive_y":"正 y 侧墙","stage_AirHangar":"机库主体","stage_Plane":"地面","stage_Plane.001":"机库平面"}
                for a,b in combinations((1,2,3),2):
                    labels[f"drone_{a}_{b}"]=f"{a}、{b} 号机相遇"
                name="，".join(labels.get(n,n) for n in first["objects"])
                message=f"首次接触：第 {first['step']} 步，第 {first['substep']} 子步，{name}"
                color="#aa312d"
            else:
                message="截至当前，真实轨迹未检测到接触"
                color="#225b47"
            text(d,44,645,message,22,color)
            if k == limit:
                text(d,44,686,f"全程最小机间中心距 {rec['min_peer_m']:.3f} m     两侧穿缝墙最小机体净空 {rec['min_wall_clearance_m']:+.3f} m",17)
                text(d,953,645,f"全队严验{ZH[rec['strict_outcome']]}",26,color)
            # Hold first/last frames so the initial layout and terminal contact can be inspected.
            copies=12 if k in (0,limit) else 1
            for _ in range(copies):
                tool.stdin.write(frame.tobytes())
            if k == limit:
                frame.save(path.with_suffix(".png"))
    finally:
        tool.stdin.close()
        code=tool.wait()
    if code:
        raise RuntimeError(f"ffmpeg exited {code}")
    expected=limit+23
    nframes=int(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
                                         "stream=nb_frames","-of","csv=p=0",str(path)],text=True).strip())
    assert nframes==expected
    return dict(kind=kind,seed=seed,cohort=str(cohort),geometry_outcome=rec["strict_outcome"],first_contact=first,
                frames=nframes,control_steps=limit,physics_substeps=limit*8,fps=fps,video=str(path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("cohort",type=Path)
    p.add_argument("--prefix",default="final")
    args=p.parse_args()
    run,cohort=args.run.resolve(),args.cohort.resolve()
    g=build_geometry(run/"source/datasets/spy_datasets")
    summary=json.loads((cohort/"summary.json").read_text())
    videos=[]
    for mode,tag in (("success","best_recording_seed"),("failure","failure_recording_seed")):
        seed=summary[tag]
        if seed is not None:
            videos.append(encode(run,cohort,seed,f"{args.prefix}-{mode}",g))
    (run/"figures"/f"{args.prefix}-replays.json").write_text(json.dumps(videos,indent=2,ensure_ascii=False)+"\n")


if __name__=="__main__":
    main()
