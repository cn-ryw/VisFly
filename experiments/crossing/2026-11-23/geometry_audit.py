"""Offline ENU trajectory audit for the published garage crossing scene (CPU/VTK)."""
import argparse
import hashlib
import importlib.util
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

RADIUS = 0.10
SUBSTEPS = 8
EPS = 1e-12  # numerical equality at the open collision threshold, metres
LEGACY_DIR = Path(__file__).resolve().parents[1] / "2026-11-18" / "replay"
ASSET_HASHES = {
    "stages/garage_v1.glb": "5a3a093d786e1795ad1c7c1e5a79e2c012fe2497a200018e6a03bfc6693bfb71",
    "self_define_objects/wall.glb": "c267ff2b08450dffe091498508d1b444709ae8100d8ddd80bd826b6f9415b6d8",
    "scene": "cc5aa2971a7bffe378550a9735005ed381575e3c49c589d58a0c708fd170aa34",
}


@dataclass
class StageMesh:
    name: str
    surface: object
    triangles: np.ndarray
    bounds: np.ndarray


@dataclass
class Geometry:
    legacy: object
    walls: dict
    dimensions: tuple
    stage: list


def build_geometry(assets: Path):
    """Load either layout of the identical September/November garage and wall GLBs."""
    assets = Path(assets).resolve()
    scene = Path("garage_crossing/garage_crossing_0.scene_instance.json")
    scene_file = next((root / scene for root in (assets / "configs", assets / "configs/scenes")
                       if (root / scene).is_file()), None)
    if scene_file is None:
        raise FileNotFoundError(f"garage_crossing_0.scene_instance.json under {assets}/configs")
    for name, expected in ASSET_HASHES.items():
        source = scene_file if name == "scene" else assets / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unexpected paper-baseline geometry: {source}")

    # This helper reads its asset directory from module globals. Each geometry gets
    # its own copy so an existing caller can still build another dataset alongside it.
    spec = importlib.util.spec_from_file_location("paper_crossing_legacy_geometry", LEGACY_DIR / "scene_geometry.py")
    legacy = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(LEGACY_DIR))
    try:
        spec.loader.exec_module(legacy)
    finally:
        sys.path.pop(0)
    legacy.ASSETS, legacy.SCENE = assets, scene_file
    walls = legacy.obstacle_boxes()
    leaves = legacy.mesh_leaves(assets / "stages/garage_v1.glb")
    if len(leaves) != 3:
        raise ValueError("Expected AirHangar, Plane and Plane.001 in the pinned GLB")
    stage = []
    for name, poly in zip(("AirHangar", "Plane", "Plane.001"), leaves):
        cells = vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)
        if not (cells[:, 0] == 3).all():
            raise ValueError(f"Non-triangle cell on {name}")
        vertices = vtk_to_numpy(poly.GetPoints().GetData())
        triangles = vertices[cells[:, 1:]]
        if name == "Plane.001" and np.array_equal(triangles, stage[-1].triangles):
            continue  # The GLB lists the same two floor triangles twice at identical coordinates.
        stage.append(StageMesh("stage_" + name, legacy.locator(poly), triangles,
                               np.array(poly.GetBounds()).reshape(3, 2)))
    return Geometry(legacy, walls, legacy.plane_dimensions(walls), stage)


def below(distance, threshold):
    return distance < threshold - EPS


def first_true(flags):
    indices = np.flatnonzero(flags)
    return int(indices[0]) if len(indices) else None


def first_wall(legacy, path, box):
    points = legacy.to_box(path, box)
    half = box["half"]
    at_poses = np.linalg.norm(np.maximum(np.abs(points) - half, 0), axis=1)
    first = first_true(below(at_poses, RADIUS))
    minimum = float(at_poses.min())
    low = np.minimum(points[:-1], points[1:])
    high = np.maximum(points[:-1], points[1:])
    # Segment AABB against the solid provides a lower bound on Euclidean distance.
    lower = np.linalg.norm(np.maximum(np.maximum(low - half, -half - high), 0), axis=1)
    for k in np.flatnonzero(lower < max(minimum, RADIUS)):
        earlier = (first is None or k+1 < first) and below(lower[k], RADIUS)
        if lower[k] >= minimum and not earlier:
            continue
        distance, _, _ = legacy.segment_box(points[k], points[k+1], half)
        minimum = min(minimum, distance)
        if earlier and below(distance, RADIUS):
            first = int(k+1)
    return first, minimum


def first_peer(path, a, b, swept):
    relative = path[:, a] - path[:, b]
    distances = np.linalg.norm(relative, axis=-1)
    if not swept or len(path) == 1:
        return first_true(below(distances, 2*RADIUS if swept else RADIUS)), float(distances.min())
    change = np.diff(relative, axis=0)
    length2 = np.einsum("ij,ij->i", change, change)
    fraction = np.divide(-np.einsum("ij,ij->i", relative[:-1], change), length2,
                         out=np.zeros_like(length2), where=length2 > 0).clip(0, 1)
    separation = np.linalg.norm(relative[:-1] + fraction[:, None] * change, axis=1)
    minima = np.r_[distances[0], separation]
    return first_true(below(minima, 2*RADIUS)), float(minima.min())


class StageQuery:
    """VTK scratch buffers reused by mesh queries. Coordinates here are Habitat XYZ."""
    def __init__(self, mesh):
        self.mesh = mesh
        self.cell = vtk.vtkGenericCell()
        self.cell_id, self.sub_id = vtk.reference(0), vtk.reference(0)
        self.dist2, self.t, self.u = vtk.reference(0.), vtk.reference(0.), vtk.reference(0.)
        self.p, self.q = [0., 0., 0.], [0., 0., 0.]
        self.ids = vtk.vtkIdList()

    def within(self, point, radius):
        found = self.mesh.surface.FindClosestPointWithinRadius(
            point, radius+EPS, self.p, self.cell, self.cell_id, self.sub_id, self.dist2)
        return math.sqrt(max(0., float(self.dist2))) if found else math.inf

    def segment_hits(self, a, b):
        # Every point on the segment is within half its length of the midpoint.
        half_length = 0.5 * np.linalg.norm(b-a)
        mid_dist = self.within((a+b)*0.5, RADIUS + half_length)
        if below(mid_dist, RADIUS):
            return True
        if half_length == 0 or not below(mid_dist - half_length, RADIUS):
            return False
        if below(self.within(a, RADIUS), RADIUS) or below(self.within(b, RADIUS), RADIUS):
            return True
        if self.mesh.surface.IntersectWithLine(
                a, b, EPS, self.t, self.p, self.q, self.sub_id, self.cell_id, self.cell):
            return True
        # Once face crossings and segment endpoints have been tested, the closest
        # pair, if within radius, can be found on a triangle edge. This also covers
        # a line parallel to a face whose orthogonal projection overlaps that face.
        lo, hi = np.minimum(a, b) - RADIUS, np.maximum(a, b) + RADIUS
        self.ids.Reset()
        self.mesh.surface.FindCellsWithinBounds(np.column_stack((lo, hi)).ravel(), self.ids)
        for i in range(self.ids.GetNumberOfIds()):
            tri = self.mesh.triangles[self.ids.GetId(i)]
            if (tri.max(axis=0) < lo).any() or (tri.min(axis=0) > hi).any():
                continue
            for j in range(3):
                d2 = vtk.vtkLine.DistanceBetweenLineSegments(
                    a, b, tri[j-1], tri[j], self.p, self.q, self.t, self.u)
                if d2 < (RADIUS-EPS)**2:
                    return True
        return False

    def first(self, path, swept):
        # Bounds only reject faraway queries. Collision uses the original triangles.
        lo, hi = self.mesh.bounds.T
        margin_lo, margin_hi = lo-RADIUS, hi+RADIUS
        if not swept:
            near = ((path >= margin_lo) & (path <= margin_hi)).all(axis=1)
            for k in np.flatnonzero(near):
                if below(self.within(path[k], RADIUS), RADIUS):
                    return int(k)
            return None
        if below(self.within(path[0], RADIUS), RADIUS):
            return 0
        near = ((np.minimum(path[:-1], path[1:]) <= margin_hi) &
                (np.maximum(path[:-1], path[1:]) >= margin_lo)).all(axis=1)
        for k in np.flatnonzero(near):
            if self.segment_hits(path[k], path[k+1]):
                return int(k+1)
        return None


def classify_path(geometry, path, native_outcome, *, swept, stride):
    events, wall_minima, peer_minima = [], {}, {}
    for drone in range(3):
        for wall, box in geometry.walls.items():
            first, minimum = first_wall(geometry.legacy, path[:, drone], box)
            label = "wall_" + wall
            wall_minima[f"drone_{drone+1}/{label}"] = minimum
            if first is not None:
                events.append((first, label, (drone+1,)))
    habitat = np.ascontiguousarray(np.stack((-path[..., 1], path[..., 2], -path[..., 0]), axis=-1))
    for mesh in geometry.stage:
        query = StageQuery(mesh)
        for drone in range(3):
            first = query.first(habitat[:, drone], swept)
            if first is not None:
                events.append((first, mesh.name, (drone+1,)))
    for a, b in combinations(range(3), 2):
        first, minimum = first_peer(path, a, b, swept)
        label = f"drone_{a+1}_{b+1}"
        peer_minima[label] = minimum
        if first is not None:
            events.append((first, label, (a+1, b+1)))

    crossings = geometry.legacy.passage_points(path, geometry.dimensions)
    for drone_crossings in crossings:
        for crossing in drone_crossings:
            crossing["fractional_step"] /= stride
    through = all(any(c["direction"] == "forward" for c in drone) and
                  all(c["in_aperture"] for c in drone) for drone in crossings)
    past = np.all(path[-1, :, 0] > 10.).item()
    first_contact = None
    if events:
        earliest = min(index for index, _, _ in events)
        ties = sorted((obj, drones) for index, obj, drones in events if index == earliest)
        first_contact = dict(step=(earliest+stride-1)//stride, sample=earliest,
                             objects=sorted({obj for obj, _ in ties}),
                             drones=sorted({drone for _, ds in ties for drone in ds}),
                             details=[dict(object=obj, drones=list(ds)) for obj, ds in ties])
        if stride == SUBSTEPS:
            first_contact["substep"] = (earliest-1) % stride+1 if earliest else 0
    bypass = bool(not events and past and not through)
    success = bool(not events and past and through)
    if events:
        outcome = "collision"
    elif success:
        outcome = "success"
    elif bypass:
        outcome = "bypass"
    elif native_outcome == "timeout":
        outcome = "timeout"
    else:
        outcome = "incomplete"
    return dict(success=success, outcome=outcome, first_contact=first_contact,
                min_peer_m=min(peer_minima.values()), min_peer_by_pair_m=peer_minima,
                min_wall_clearance_m=min(wall_minima.values())-RADIUS,
                min_wall_center_to_solid_by_drone_m=wall_minima, through_gap=through,
                bypass=bypass, all_past_goal=past, midplane_crossings=crossings)


def audit_episode(geometry, control_xyz, substep_xyz, native_outcome, *, control_only=False):
    """Test each centre's linearly swept 10 cm sphere and every pair of relative segments.

    first_contact.step and .sample are the ending control/sample indices of the
    first contacting interval (0 for initial overlap). substep is 1..8 within that
    control. The legacy audit sweeps walls on control chords and samples the
    stage and 10 cm inter-centre criterion at controls. Historical calibration can
    pass control_only=True, which explicitly labels the missing integrator path.
    """
    control = np.asarray(control_xyz, dtype=np.float64)
    substeps = np.asarray(substep_xyz, dtype=np.float64)
    stride = 1 if control_only else SUBSTEPS
    if control.ndim != 3 or control.shape[1:] != (3, 3) or not len(control):
        raise ValueError("control_xyz must contain initial ENU XYZ and every control endpoint, shape [T+1,3,3]")
    if substeps.shape != (stride*(len(control)-1)+1, 3, 3):
        raise ValueError(f"Expected {stride} integrator substeps per control interval plus initial pose")
    if not np.isfinite(control).all() or not np.isfinite(substeps).all():
        raise ValueError("Non-finite ENU position")
    if not np.allclose(substeps[::stride], control, atol=1e-6, rtol=0):
        raise ValueError("Measured substep endpoints do not match control poses (1 micrometre tolerance)")
    native_outcome = str(native_outcome)
    legacy = classify_path(geometry, control, native_outcome, swept=False, stride=1)
    strict = classify_path(geometry, substeps, native_outcome, swept=True, stride=stride)
    return dict(native_outcome=native_outcome, legacy_success=legacy["success"],
                legacy_outcome=legacy["outcome"], legacy_first_contact=legacy["first_contact"],
                legacy_through_gap=legacy["through_gap"], legacy_bypass=legacy["bypass"],
                strict_success=strict.pop("success"), strict_outcome=strict.pop("outcome"), **strict,
                path_sampling="historical_control_chords_only" if control_only else "measured_integrator_substeps",
                control_steps=len(control)-1, measured_substeps=None if control_only else len(substeps)-1,
                aperture_x_ylo_yhi_zlo_zhi_m=list(geometry.dimensions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="Root with configs, stages and self_define_objects")
    parser.add_argument("--output", type=Path, required=True, help="JSON file (list of trajectory results)")
    parser.add_argument("traces", type=Path, nargs="+", help="NPZ with control_xyz, substep_xyz, native_outcome")
    options = parser.parse_args()
    geometry = build_geometry(options.assets)
    rows = []
    for trace in options.traces:
        with np.load(trace, allow_pickle=False) as data:
            rows.append(dict(trajectory=str(trace), **audit_episode(
                geometry, data["control_xyz"], data["substep_xyz"], data["native_outcome"].item())))
    options.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = options.output.with_suffix(options.output.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    tmp.replace(options.output)


if __name__ == "__main__":
    main()
