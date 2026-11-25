"""Compare the September simulator's unmodified nearest-point queries with the signed scene geometry."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def to_hab(p):
    x, y, z = p
    return [-float(y), float(z), -float(x)]


def to_enu(p):
    x, y, z = p
    return [-float(z), -float(x), float(y)]


def write(path, data):
    temp = path.with_suffix(path.suffix+".partial")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    temp.replace(path)


def raw(runtime):
    from paper_runtime import load_example, make_env, reset_seed
    run, dest = runtime
    rl = load_example(run)
    env = make_env(rl)
    try:
        reset_seed(env, 1000)
        sim = env.envs.sceneManager.scenes[0]
        root = sim.get_active_scene_graph().get_root_node()
        bounds = root.cumulative_bb
        manager = sim.get_rigid_object_manager()
        rigid = []
        for handle in manager.get_object_handles():
            obj = manager.get_object_by_handle(handle)
            if obj is None:
                rigid.append(dict(handle=handle, accessible_with_rigid_object_getter=False))
                continue
            rigid.append(dict(handle=handle,motion=str(obj.motion_type),position_enu=to_enu(obj.translation),
                              object_world_transform_habitat=np.asarray(obj.transformation).tolist()))

        points = []
        def point(label, p, traces=None):
            hab = np.asarray(to_hab(p),dtype=np.float32)
            hit = sim.get_closest_collision_point(pt=hab.reshape(3,1), max_search_radius=float(env.envs.sceneManager.sensitive_radius))
            pos = to_enu(hit.hit_pos)
            points.append(dict(label=label, enu=list(map(float,p)),native_nearest_enu=pos,
                               native_distance_m=float(np.linalg.norm(np.array(pos)-p)),
                               native_out_of_bounds=bool(hit.is_out_bound),recorded=traces))
        for y, side in ((-2.0,"negative_y"),(2.0,"positive_y")):
            for x in (7.65,7.72,7.79,7.86,8.0,8.14,8.21,8.28,8.35):
                point(f"wall_{side}_x{x:.2f}",[x,y,1.5])
        for y in (-0.68,-0.54,-0.48,0.0,0.48,0.54,0.68):
            point(f"aperture_y{y:.2f}",[8.,y,1.5])
        for x in (-1.05,-1.0,-0.95,-0.88,-0.5,0.0):
            point(f"back_limit_x{x:.2f}",[x,-.4,2.1])
        for z in (-.15,0.,.05,.11,.25,.5,1.5):
            point(f"low_altitude_z{z:.2f}",[4.,0.,z])

        for step,seeds in ((0,(1000,1001,1002,1003,1004,1037,1072,1097)),(1013760,tuple(range(1000,1006)))):
            folder=run/f"evaluation/development/PPO_{step}"
            for seed in seeds:
                trace=folder/f"seed_{seed}.npz"
                with np.load(trace) as d:
                    if step:
                        xyz=d["control_xyz"]
                        frames=np.argmin(np.abs(xyz[:,:,0]-8),axis=0)
                        ids=list(range(3))
                    else:
                        xyz=d["control_xyz"]
                        frames=np.full(3,len(xyz)-1)
                        ids=np.flatnonzero(d["native_collision"][-1])
                    for i in ids:
                        at=int(frames[i]); idx=int(i)
                        p=xyz[at,idx].astype(float)
                        point(f"steps{step}_seed{seed}_uav{idx+1}_frame{at}",p,
                              dict(train_steps=step,seed=seed,drone=idx+1,control_frame=at,
                                   final_outcome=d["native_outcome"].item(),
                                   env_distance_m=float(d["native_distance"][at-1,idx]),
                                   env_collision=bool(d["native_collision"][at-1,idx]),
                                   env_out_of_bounds=bool(d["out_of_bounds"][at-1,idx]),
                                   peer_center_m=np.linalg.norm(p-np.delete(xyz[at],idx,axis=0),axis=1).min().item()))
        write(dest/"native-points.json",dict(frame="ENU x,y,z in metres", native_radius_m=float(env.envs.uav_radius),
              active_root_cumulative_bb_habitat_after_reset=[list(bounds.min),list(bounds.max)],
              active_objects=rigid,points=points))
    finally:
        env.close()


def compare(runtime):
    from geometry_audit import build_geometry
    run,dest=runtime
    native=json.loads((dest/"native-points.json").read_text())
    geom=build_geometry(run/"source/datasets/spy_datasets")
    index=geom.legacy.surface_index()
    def solid_distance(p):
        return min(np.linalg.norm(np.maximum(np.abs(geom.legacy.to_box(np.array([p]),box)[0])-box["half"],0.)) for box in geom.walls.values())
    results=[]
    for point in native["points"]:
        xyz=np.array(point["enu"])
        hit=np.array(point["native_nearest_enu"])
        ref=geom.legacy.probe(index,xyz)
        hit_ref=geom.legacy.probe(index,hit)
        walls=[r for r in ref if r["source"].startswith("wall")]
        stage=[r for r in ref if not r["source"].startswith("wall")]
        results.append(dict(**point,nearest_wall_solid_m=solid_distance(xyz), nearest_wall_face_m=walls[0]["residual_m"],
                        nearest_actual_stage_m=stage[0]["residual_m"],actual_stage_name=stage[0]["source"],
                        nearest_actual_face_name=ref[0]["source"],nearest_actual_face_m=ref[0]["residual_m"],
                        nearest_mesh_to_native_hit_name=hit_ref[0]["source"],native_hit_mesh_residual_m=hit_ref[0]["residual_m"]))
    write(dest/"point-comparison.json",results)

    zero=run/"evaluation/development/PPO_0"
    rows=[]
    for seed in range(1000,1100):
        trace=zero/f"seed_{seed}.npz"
        with np.load(trace) as d:
            p=d["control_xyz"][-1].astype(float)
            pair_dist=np.linalg.norm(p[:,None]-p[None,:],axis=-1)
            np.fill_diagonal(pair_dist,np.inf)
            labels=[]
            details=[]
            for i in np.flatnonzero(d["native_collision"][-1]):
                closest=geom.legacy.probe(index,p[i])[0]
                dist=float(d["native_distance"][-1,i])
                plane_delta=abs(p[i,0]+1)
                if abs(dist-plane_delta)<1e-5 and closest["residual_m"]>.10:
                    label="back_x_minus_1"
                elif abs(dist-pair_dist[i].min())<1e-5:
                    label="peer_center"
                else:
                    label="unresolved"
                labels.append(label)
                details.append(dict(drone=int(i+1),cause=label,enu=p[i].tolist(),native_distance_m=dist,
                                    actual_nearest_face=closest["source"], actual_nearest_face_m=closest["residual_m"],
                                    nearest_peer_m=pair_dist[i].min(),abs_distance_to_x_minus_1_m=plane_delta))
            rows.append(dict(seed=seed,native=d["native_outcome"].item(),
                             native_out_of_bounds=bool(d["out_of_bounds"][-1].any()),
                             causes="+".join(sorted(set(labels))),details=details))
    write(dest/"zero-checkpoint-terminal.json",rows)
    with (zero/"episodes.csv").open() as f:
        strict={int(r["seed"]):r for r in csv.DictReader(f)}
    from collections import Counter
    counts=Counter((v["causes"],strict[v["seed"]]["strict_outcome"]) for v in rows)
    gap=next(row for row in results if row["label"]=="aperture_y0.00")
    center=[r for r in results if r["label"].startswith("wall_") and r["label"].endswith("_x8.00")]
    front=[r for r in results if r["label"].startswith("wall_") and r["label"].endswith("_x7.72")]
    threshold=.10
    assert len(center)==len(front)==2
    assert all(.07<r["native_distance_m"]<threshold and r["native_hit_mesh_residual_m"]<.002 for r in front)
    assert all(r["nearest_wall_solid_m"]<1e-8 and r["native_distance_m"]>.18 for r in center)
    assert .59<gap["native_distance_m"]<.61
    walltraces=[r for r in results if r["recorded"] and r["recorded"]["train_steps"]==1013760]
    matches=sum(abs(r["native_distance_m"]-r["nearest_wall_face_m"])<.003 for r in walltraces)
    back=[r for r in results if r["label"].startswith("back_limit_")]
    sample=next(r for r in results if r["label"].startswith("steps0_seed1000"))
    million=run/"evaluation/development/PPO_1013760"
    episodes=json.loads((million/"geometry.json").read_text())
    discordant=[]
    for result in episodes:
        if result["native_outcome"]!="success" or result["strict_success"]:
            continue
        seed=result["seed"]
        with np.load(million/f"seed_{seed}.npz") as trace:
            controls=trace["control_xyz"].astype(float)
            pairs=[(0,1),(0,2),(1,2)]
            distances=np.column_stack([np.linalg.norm(controls[:,i]-controls[:,j],axis=-1) for i,j in pairs])
            frame,pair_index=map(int,np.unravel_index(np.argmin(distances),distances.shape))
            i,j=pairs[pair_index]
            entry=dict(seed=seed,strict_first_contact=result["first_contact"],
                       strict_minimum_peer_m=result["min_peer_m"],
                       closest_control_frame=frame,closest_control_pair=[i+1,j+1],
                       closest_control_distance_m=float(distances[frame,pair_index]),
                       control_distances_12_13_23_m=distances[frame].tolist(),
                       positions_enu=controls[frame].tolist(),
                       official_mixed_norm_for_drone_1_m=float(np.linalg.norm(controls[frame,1:]-controls[frame,0])),
                       env_active_nearest_distance_m=trace["native_distance"][frame-1].tolist() if frame else None)
            discordant.append(entry)
    write(dest/"million-native-success-strict-collision.json",discordant)
    summary=dict(zero_terminal_native_vs_strict={f"{group}/{outcome}":n for (group,outcome),n in counts.items()},
                 early_actual_mesh_closest_at_hit_min=min(z["actual_nearest_face_m"] for r in rows for z in r["details"] if z["cause"]=="back_x_minus_1"),
                 inspected_million_development_samples=len(walltraces), native_static_distance_matches_nearest_wall_face_within_3mm=matches,
                 wall_face_probes={r["label"]:r["native_distance_m"] for r in front+center+[gap]},
                 back_plane_queries={r["label"]:dict(native_nearest=r["native_nearest_enu"],closest_actual_mesh_to_hit_m=r["native_hit_mesh_residual_m"],out_of_bounds=r["native_out_of_bounds"]) for r in back},
                 sample_seed1000=sample)
    summary["million_dev_episodes_with_negative_wall_skin_clearance"]=sum(r["min_wall_clearance_m"]<0 for r in episodes)
    summary["million_dev_minimum_wall_skin_clearance_m"]=min(r["min_wall_clearance_m"] for r in episodes)
    summary["million_native_success_and_strict_failure"]=len(discordant)
    summary["discordant_control_pair_center_min_less_0p10"]=[r["seed"] for r in discordant if r["closest_control_distance_m"]<.10]
    summary["discordant_control_pair_center_min_at_least_0p10"]=sum(r["closest_control_distance_m"]>=.10 for r in discordant)
    write(dest/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--mesh",action="store_true",help="Use system Python with VTK to match the saved native queries")
    args=p.parse_args()
    run=args.run.resolve()
    dest=run/"smoke/native-geometry"
    dest.mkdir(parents=True,exist_ok=True)
    if args.mesh:
        compare((run,dest))
    else:
        raw((run,dest))
        subprocess.run(["/usr/bin/python3",str(Path(__file__).resolve()),str(run),"--mesh"],check=True)
