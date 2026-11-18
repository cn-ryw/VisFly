"""Audit measured paths against independent meshes, match starts and test paired success."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
import vtk

from replay_paths import archive_root, write_json
from scene_geometry import (RADIUS, classify, minimum_wall_distances, obstacle_boxes,
                            plane_dimensions, surface_for_endpoint, surface_index)
BUDGETS = (500736, 1000704)
SEEDS = range(1000, 1100)
OBS_KEYS = {'state', 'depth', 'target', 'swarm'}
PAIR_TYPES = ('both_success', 'only_500736', 'only_1000704', 'both_fail')


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def independent_endpoint_distances(positions, surfaces):
    """Endpoint rule uses distance to other centers or nearest scene triangle."""
    separation = np.linalg.norm(positions[:, :, None] - positions[:, None, :], axis=-1)
    for drone in range(3):
        separation[:, drone, drone] = np.inf
    closest = separation.min(axis=2)
    cell, subcell, d2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.)
    pt = [0., 0., 0.]
    for t, frame in enumerate(positions):
        for i, (x, y, z) in enumerate(frame):
            query = [-y, z, -x]
            for surface in surfaces.values():
                surface.FindClosestPoint(query, pt, cell, subcell, d2)
                closest[t, i] = min(closest[t, i], math.sqrt(float(d2)))
    return closest


def endpoint_object(drone, surface):
    if surface.startswith('drone_'):
        other = int(surface.removeprefix('drone_'))
        a, b = sorted((drone, other))
        return f'drone_{a}_{b}'
    if surface.startswith('wall_') or surface == 'unclassified':
        return surface
    return 'stage_' + surface


def classify_contacts(step, geometry, position, near_points, endpoint, surfaces):
    if step is None:
        return []
    by_drone_object = defaultdict(set)
    for e in geometry['wall_contacts']:
        if e['step'] == step:
            by_drone_object[(e['drone'], 'wall_' + e['wall'])].add('swept')
    for e in surface_for_endpoint(position[step], near_points[step], endpoint[step], surfaces):
        by_drone_object[(e['drone'], endpoint_object(e['drone'], e['surface']))].add('endpoint')
    return [dict(drone=d, object=obj, via=sorted(via))
            for (d, obj), via in sorted(by_drone_object.items())]


def verify_episode(directory, budget, seed, walls, dimensions, surfaces, baseline, run):
    e = json.loads((directory / 'episode.json').read_text())
    archive = directory / 'trajectory.npz'
    assert e['seed'] == seed and e['agent_transitions'] == budget
    doc = json.loads((baseline / f'train/ppo_{budget}.json').read_text())
    assert e['checkpoint_sha256'] == doc['sha256']
    with np.load(archive) as trace:
        state = trace['state']
        positions = state[:, :, :3].astype(np.float64)
        step = len(positions) - 1
        assert e['episode_length'] == step
        assert positions.shape == (step+1, 3, 3)
        assert np.isfinite(positions).all()
        hashes = {'state': sha_array(state[0]), 'depth': sha_array(trace['depth'][0]),
                  'target': sha_array(trace['target']),
                  'swarm': sha_array(np.stack([np.delete(state[0], i, axis=0) for i in range(3)]))}
        assert hashes == e['initial_observation_sha256']
        np.testing.assert_array_equal(positions[0], e['initial_positions'])
        np.testing.assert_array_equal(positions[-1], e['final_positions'])
        wall, geometry = classify(positions, walls, dimensions)
        nearest_point = trace['collision_point']
        measured_distance = trace['collision_distance']
        np.testing.assert_allclose(np.linalg.norm(positions - nearest_point, axis=-1),
                                   measured_distance, rtol=1e-5, atol=1e-6)
        independent_distance = independent_endpoint_distances(positions, surfaces)
        endpoints = independent_distance < RADIUS
        np.testing.assert_array_equal(endpoints, measured_distance < RADIUS)
        np.testing.assert_array_equal(endpoints, trace['endpoint_contact'])
        np.testing.assert_array_equal(wall, trace['swept_wall_contact'])
        colliding = wall | endpoints
        np.testing.assert_array_equal(colliding, trace['is_collision'])
        np.testing.assert_array_equal(colliding[-1], e['final_collision_flags'])
        np.testing.assert_equal(colliding.any(), e['any_collision'])
        frames = np.flatnonzero(colliding.any(axis=1))
        first = int(frames[0]) if len(frames) else None
        assert first in (None, 0, step)
        events = classify_contacts(first, geometry, positions, nearest_point, endpoints, surfaces)
        drones = sorted({event['drone'] for event in events})
        if first is not None:
            assert drones == (np.flatnonzero(colliding[first]) + 1).tolist()
        safe = bool(geometry['wall_clear_team_pass'] and first is None)
        objects = sorted({event['object'] for event in events})
        assert sum(map(bool, (e['team_success'], e['time_limit'], e['any_collision']))) == 1
        if safe != e['team_success']:
            assert e['team_success'] and not geometry['all_crossings_in_aperture']
        outcome = 'success' if safe else ('bypass' if e['team_success'] else e['end'])
        try:
            trajectory = str(archive.relative_to(run))
        except ValueError:  # Used only by the known-case regression.
            trajectory = str(archive)
        return dict(seed=seed, agent_transitions=budget, length=step, sim_end=e['end'], safe_pass=safe,
                    outcome=outcome, first_contact_step=first, first_contact_drones=drones,
                    first_contact_objects=objects, first_contact_details=events,
                    initial_observation_sha256=hashes, midplane_crossings=geometry['midplane_crossings'],
                    all_past_goal=geometry['all_past_goal'],
                    all_crossings_in_aperture=geometry['all_crossings_in_aperture'],
                    maximum_endpoint_distance_error_m=float(np.max(np.abs(independent_distance-measured_distance))),
                    minimum_wall_distances=minimum_wall_distances(positions, walls) if safe else [],
                    trajectory=trajectory, verified_pose_frames=step+1)


def check_known(walls, dims, surfaces, baseline, run):
    first = verify_episode(baseline / 'evaluation/steps_500736/seed_42', 500736, 42, walls, dims, surfaces, baseline, run)
    second = verify_episode(baseline / 'evaluation/steps_1000704/seed_42', 1000704, 42, walls, dims, surfaces, baseline, run)
    neighbor = verify_episode(baseline / 'evaluation/steps_1000704/seed_43', 1000704, 43, walls, dims, surfaces, baseline, run)
    assert first['safe_pass'] and first['length'] == 51 and first['first_contact_step'] is None
    assert second['length'] == second['first_contact_step'] == 41
    assert second['first_contact_objects'] == ['wall_negative_y'] and second['first_contact_drones'] == [3]
    assert neighbor['length'] == neighbor['first_contact_step'] == 31
    assert neighbor['first_contact_objects'] == ['drone_1_2'] and neighbor['first_contact_drones'] == [1, 2]


def summarize_budget(rows):
    objects, categories = Counter(), Counter()
    category_sets = Counter()
    for row in rows:
        objs = row['first_contact_objects']
        objects.update(objs)
        cats = {o.split('_')[0] for o in objs}
        categories.update(cats)
        if cats:
            category_sets['+'.join(sorted(cats))] += 1
    outcomes = Counter(r['outcome'] for r in rows)
    assert sum(outcomes.values()) == len(rows)
    assert sum(category_sets.values()) == outcomes['collision']
    return dict(episodes=len(rows), **{k:outcomes[k] for k in ('success','collision','timeout','bypass')},
                simulator_counts=dict(Counter(r['sim_end'] for r in rows)),
                mean_length=float(np.mean([r['length'] for r in rows])),
                first_contact_categories=dict(categories), first_contact_category_sets=dict(category_sets),
                first_contact_objects=dict(objects), initial_collision_episodes=sum(r['first_contact_step']==0 for r in rows),
                verified_pose_frames=sum(r['verified_pose_frames'] for r in rows),
                maximum_endpoint_distance_error_m=max(r['maximum_endpoint_distance_error_m'] for r in rows))


def finalize(records, dimensions, output):
    pairs = []
    cells = dict.fromkeys(PAIR_TYPES, 0)
    for seed in SEEDS:
        a, b = (records[k, seed] for k in BUDGETS)
        assert a['initial_observation_sha256'] == b['initial_observation_sha256']
        assert set(a['initial_observation_sha256']) == OBS_KEYS
        outcome = {(True, True):PAIR_TYPES[0], (True, False):PAIR_TYPES[1],
                   (False, True):PAIR_TYPES[2], (False, False):PAIR_TYPES[3]}[a['safe_pass'], b['safe_pass']]
        cells[outcome] += 1
        pairs.append(dict(seed=seed, initial_match=True, pair_outcome=outcome, episodes={str(BUDGETS[0]):a,str(BUDGETS[1]):b}))
    assert len({pair['episodes'][str(BUDGETS[0])]['initial_observation_sha256']['state'] for pair in pairs}) == len(SEEDS)
    summaries = {str(k):summarize_budget([records[k,s] for s in SEEDS]) for k in BUDGETS}
    for k, only in zip(BUDGETS, ('only_500736','only_1000704')):
        assert summaries[str(k)]['success'] == cells['both_success'] + cells[only]
    a, b = cells['only_500736'], cells['only_1000704']
    discordant = a+b
    p = float(binomtest(a, discordant, .5, alternative='two-sided').pvalue) if discordant else 1.
    exact = min(1., 2 * sum(math.comb(discordant,i) for i in range(min(a,b)+1)) / 2**discordant)
    assert math.isclose(p, exact, rel_tol=1e-12)
    decision = ('500736_better' if a>b else '1000704_better') if p<.05 else 'inconclusive'
    test = dict(method='two-sided exact McNemar (conditional binomial on discordant pairs)', p_value=p, alpha=.05,
                discordant=discordant, decision=decision)
    summary = dict(n_pairs=len(pairs), matched_initial_pairs=len(pairs), pair_counts=cells,
                   budgets=summaries, exact_mcnemar=test, seeds=[SEEDS.start,SEEDS.stop-1],
                   aperture_x_ylo_yhi_zlo_zhi_m=dimensions, drone_radius_m=RADIUS,
                   geometry='independent GLB linear swept sphere vs wall solids, per-pose GLB triangle and neighbor-center endpoint distances; pairwise contact radius per original simulator',
                   checkpoint_scope='two checkpoints of one training run; development start seeds only')
    write_json(output / 'pairs.json', dict(pairs=pairs))
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-run", type=Path, help="Directory containing evaluation/steps_*")
    ap.add_argument("--output-dir", type=Path, help="Directory for pairs.json and summary.json")
    ap.add_argument("--check-known-only", action="store_true")
    args = ap.parse_args()
    archive = archive_root()
    baseline = archive / "crossing-corrected-20261118"
    run = (args.input_run or archive / "crossing-paired-1000-1099").resolve()
    if not args.check_known_only and not args.output_dir:
        ap.error("--output-dir is required for a full audit")
    walls = obstacle_boxes()
    dims = plane_dimensions(walls)
    surfaces = surface_index()
    check_known(walls, dims, surfaces, baseline, run)
    if args.check_known_only:
        print("Known path regression passed.")
        return
    rows = {}
    for seed in SEEDS:
        for budget in BUDGETS:
            case = run / "evaluation" / f"steps_{budget}" / f"seed_{seed}"
            rows[budget,seed] = verify_episode(case, budget, seed, walls, dims, surfaces, baseline, run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    finalize(rows, dims, args.output_dir)


if __name__ == "__main__":
    main()
