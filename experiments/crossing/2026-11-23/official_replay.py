"""Replay measured poses using the crossing example's original Habitat top camera."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from paper_runtime import load_example, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("cohort",type=Path)
    parser.add_argument("--prefix",default="official-final")
    args=parser.parse_args()
    run,cohort=args.run.resolve(),args.cohort.resolve()
    rl=load_example(run)
    tally=json.loads((cohort/"summary.json").read_text())
    files=[]
    for kind,seed in (("success",tally["best_recording_seed"]),("failure",tally["failure_recording_seed"])):
        if seed is None:
            continue
        with np.load(cohort/f"seed_{seed}.npz") as data:
            states=data["state"]
        xyz=states[:,:,:3]
        lo,hi=xyz.min(axis=(0,1)),xyz.max(axis=(0,1))
        eye=[float((lo[0]+hi[0])/2),float((lo[1]+hi[1])/2),7.5]
        elevation=np.maximum(eye[2]-xyz[:,:,2],.1)
        span=max(float(np.max(abs(xyz[:,:,0]-eye[0])/elevation)),
                 float(np.max(abs(xyz[:,:,1]-eye[1])/elevation))*960/540)
        fov=float(np.clip(np.degrees(2*np.arctan(1.18*span)),90,155))
        render=dict(mode="fix",view="top",resolution=[540,960],position=torch.tensor([eye]),
                    trajectory=True,line_width=3)
        env=rl.MultiNavigationEnv(num_agent_per_scene=3,num_scene=1,random_kwargs=copy.deepcopy(rl.random_kwargs),
                                  visual=True,max_episode_steps=256,scene_kwargs=dict(path=rl.scene_path,render_settings=render))
        filename=run/"figures"/f"{args.prefix}-{kind}-seed{seed}.mp4"
        process=subprocess.Popen(["ffmpeg","-loglevel","error","-y","-f","rawvideo","-pixel_format","rgb24",
                                  "-video_size","960x540","-framerate","12.5","-i","-","-an","-c:v","libx264",
                                  "-threads","2","-crf","18","-preset","veryfast","-pix_fmt","yuv420p",
                                  "-movflags","+faststart",str(filename)],stdin=subprocess.PIPE)
        try:
            env.reset()
            env.envs.sceneManager._render_camera[0]._sensors["render"].fov=fov
            for k,s in enumerate(states):
                env.envs.sceneManager.set_pose(torch.from_numpy(s[:,:3].copy()),torch.from_numpy(s[:,3:7].copy()))
                rendered=np.asarray(env.render()[0])
                frame=np.ascontiguousarray(rendered[:,:,:3])
                assert frame.shape==(540,960,3) and frame.dtype==np.uint8
                for _ in range(12 if k in (0,len(states)-1) else 1):
                    process.stdin.write(frame.tobytes())
                if k==len(states)-1:
                    Image.fromarray(frame).save(filename.with_suffix(".png"))
        finally:
            process.stdin.close()
            status=process.wait()
            env.close()
        if status:
            raise RuntimeError(f"ffmpeg returned {status}")
        nframes=int(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
                        "stream=nb_frames","-of","csv=p=0",str(filename)],text=True))
        assert nframes==len(states)+22
        files.append(dict(seed=seed,mode=kind,checkpoint=cohort.name,camera="official fixed top",camera_position_enu_m=eye,observer_hfov_deg=fov,
                          frames=nframes,archived_poses=len(states),fps=12.5,video=str(filename)))
    write_json(run/"figures"/f"{args.prefix}-replays.json",files)


if __name__=="__main__":
    main()
