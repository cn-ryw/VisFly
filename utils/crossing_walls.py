"""Swept-sphere contact with the static cuboids used by the crossing task."""
import json
from functools import lru_cache
from pathlib import Path

import magnum as mn
from magnum import scenetools, trade
import numpy as np


@lru_cache(maxsize=16)
def _mesh_bounds(asset):
    importer = trade.ImporterManager().load_and_instantiate("GltfImporter")
    importer.open_file(asset)
    scene = importer.scene(importer.default_scene)
    vertices = []
    transforms = scenetools.absolute_field_transformations3d(scene, trade.SceneField.MESH)
    for index, transform in zip(scene.field(trade.SceneField.MESH), transforms):
        mesh = importer.mesh(index)
        points = np.asarray(mesh.attribute(trade.MeshAttribute.POSITION))
        matrix = np.asarray(transform)
        vertices.extend(points @ matrix[:3, :3].T + matrix[:3, 3])
    corners = np.unique(vertices, axis=0)
    low, high = corners.min(axis=0), corners.max(axis=0)
    if len(corners) != 8 or not np.all((corners == low) | (corners == high)):
        raise ValueError(f"The crossing wall collision mesh must be a cuboid: {asset}")
    return (low + high) / 2, (high - low) / 2


def load_crossing_walls(scene_path, simulator):
    """Read the same collision mesh and instance transforms used to build Habitat's scene."""
    if not str(scene_path).endswith(".scene_instance.json"):
        return []
    instances = json.loads(Path(scene_path).read_text()).get("object_instances", [])
    walls = [w for w in instances if w["template_name"] == "debug_wall"
             and w.get("motion_type", "").upper() == "STATIC"]
    if not walls:
        return []
    manager = simulator.get_object_template_manager()
    handle, = [h for h in manager.get_template_handles("debug_wall")
               if Path(h).name == "debug_wall.object_config.json"]
    template = manager.get_template_by_handle(handle)
    center, half = _mesh_bounds(template.collision_asset_handle)
    result = []
    for wall in walls:
        w, x, y, z = wall.get("rotation", [1, 0, 0, 0])
        # Use Habitat's stored quaternion, including its original rounding.
        matrix = np.asarray(mn.Quaternion(mn.Vector3(x, y, z), w).to_matrix(), dtype=float)
        column_lengths = np.linalg.norm(matrix, axis=0)
        axes = matrix / column_lengths
        scale = np.asarray(template.scale) * wall.get("uniform_scale", 1.)
        offset = center
        if wall.get("translation_origin", "COM").upper() == "COM":
            offset = center - np.asarray(template.com)
        origin = wall["translation"] + matrix @ (offset * scale)
        result.append((origin, axes, half * np.abs(scale) * column_lengths))
    return result


def segment_hits_box(a, b, half, radius):
    """Exact segment-to-solid AABB distance test, including rounded edge/corner clearance."""
    delta = b - a
    # Each slab crossing changes which coordinates contribute to squared distance.
    varying = delta != 0
    boundaries = np.concatenate(((-half[varying] - a[varying]) / delta[varying],
                                 (half[varying] - a[varying]) / delta[varying]))
    cuts = np.unique(np.r_[0., boundaries[(0 < boundaries) & (boundaries < 1)], 1.])
    for left, right in zip(cuts[:-1], cuts[1:]):
        mid = a + ((left + right) / 2) * delta
        outside = np.abs(mid) > half
        if not outside.any():
            return True
        slope = delta[outside]
        offset = a[outside] - np.sign(mid[outside]) * half[outside]
        denom = slope @ slope
        t = np.clip(-(offset @ slope) / denom, left, right) if denom else left
        nearest = offset + t * slope
        if nearest @ nearest < radius * radius:
            return True
    return False


def swept_wall_hits(walls, starts, ends, radius):
    """Test a batch of center segments in Habitat world coordinates."""
    hit = np.zeros(len(ends), dtype=bool)
    for origin, axes, half in walls:
        a = (starts - origin) @ axes
        b = (ends - origin) @ axes
        # Reject segments outside a slab before testing rounded corners.
        candidates = ((np.minimum(a, b) < half + radius)
                      & (np.maximum(a, b) > -half - radius)).all(axis=1)
        for i in np.flatnonzero(candidates & ~hit):
            hit[i] = segment_hits_box(a[i], b[i], half, radius)
    return hit
