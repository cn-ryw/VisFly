"""Original GLB-triangle and swept-wall audit used for the paired crossing trajectories."""
import itertools
import json
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from replay_paths import REPO

RADIUS = 0.1
ASSETS = REPO / "datasets/visfly-beta"
SCENE = ASSETS / "configs/scenes/garage_crossing/garage_crossing_0.scene_instance.json"

def mesh_leaves(asset):
    reader = vtk.vtkGLTFReader()
    reader.SetFileName(str(asset))
    reader.Update()
    iterator = reader.GetOutput().NewTreeIterator()
    iterator.InitTraversal()
    result = []
    while not iterator.IsDoneWithTraversal():
        result.append(iterator.GetCurrentDataObject())
        iterator.GoToNextItem()
    return result


def locator(poly):
    search = vtk.vtkStaticCellLocator()
    search.SetDataSet(poly)
    search.BuildLocator()
    return search


def surface_index():
    stage = mesh_leaves(ASSETS / "stages/garage_v1.glb")
    names = ["AirHangar", "Plane", "Plane.001"]
    assert len(stage) == len(names)
    index = {name: locator(poly) for name, poly in zip(names, stage)}
    wall = mesh_leaves(ASSETS / "self_define_objects/wall.glb")[0]
    local = vtk_to_numpy(wall.GetPoints().GetData()).astype("float64")
    for obj in json.loads(SCENE.read_text())["object_instances"]:
        assert obj["template_name"] == "debug_wall"
        trans = np.array(obj["translation"])
        quat = np.array(obj["rotation"])
        qw, qv = quat[0], quat[1:]
        poly = vtk.vtkPolyData()
        poly.DeepCopy(wall)
        # Magnum applies the stored quaternion, including its rounded coefficients.
        xyz = trans + local + 2 * np.cross(qv, np.cross(qv, local) + qw * local)
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(xyz, deep=True))
        poly.SetPoints(points)
        name = "wall_positive_y" if trans[0] < 0 else "wall_negative_y"
        index[name] = locator(poly)
    return index


def probe(index, enu):
    query = [-enu[1], enu[2], -enu[0]]
    result = []
    for name, search in index.items():
        closest = [0., 0., 0.]
        cell_id, sub_id, d2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.)
        search.FindClosestPoint(query, closest, cell_id, sub_id, d2)
        cell = search.GetDataSet().GetCell(int(cell_id))
        tri = vtk_to_numpy(cell.GetPoints().GetData())
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        normal = normal / np.linalg.norm(normal)
        enu_normal = [-float(normal[2]), -float(normal[0]), float(normal[1])]
        result.append(dict(source=name, mesh_cell=int(cell_id), normal_enu=enu_normal,
                           residual_m=float(d2) ** 0.5))
    return sorted(result, key=lambda c: c["residual_m"])


def obstacle_boxes():
    # vtkGLTFReader applies the scale stored in the cube's node before returning vertices.
    vertices = vtk_to_numpy(mesh_leaves(ASSETS / "self_define_objects/wall.glb")[0].GetPoints().GetData())
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    assert np.array_equal(-lo, hi) and len(np.unique(vertices, axis=0)) == 8
    boxes = {}
    for obj in json.loads(SCENE.read_text())["object_instances"]:
        assert obj["template_name"] == "debug_wall" and obj["uniform_scale"] == 1
        center = np.asarray(obj["translation"])
        q = np.asarray(obj["rotation"])
        unit = np.eye(3)
        transformed = unit + 2 * np.cross(q[1:], np.cross(q[1:], unit) + q[0] * unit)
        scale = np.linalg.norm(transformed, axis=1)
        axes = transformed / scale[:, None]
        np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-10)
        name = "negative_y" if center[0] > 0 else "positive_y"
        boxes[name] = dict(center=center, axes=axes, half=hi * scale)
    return boxes


def to_box(enu, box):
    hab = np.column_stack((-enu[:, 1], enu[:, 2], -enu[:, 0]))
    return (hab - box["center"]) @ box["axes"].T


def segment_box(a, b, half):
    """Exact Euclidean line-segment distance to an axis-aligned solid box and center intersection."""
    delta = b - a
    cuts = [0., 1.]
    for i, d in enumerate(delta):
        if abs(d) > 1e-15:
            cuts.extend(t for signed_edge in (-half[i], half[i])
                        for t in [(signed_edge - a[i]) / d] if 0 < t < 1)
    cuts = sorted(set(cuts))
    candidates = list(cuts)
    inside = []
    for left, right in zip(cuts, cuts[1:]):
        midpoint = a + ((left + right) / 2) * delta
        active = np.abs(midpoint) > half
        if not active.any():
            inside.append((left, right))
        offset = a[active] - np.sign(midpoint[active]) * half[active]
        direction = delta[active]
        denom = direction @ direction
        if denom > 0:
            candidates.append(float(np.clip(-(offset @ direction) / denom, left, right)))
    times = np.asarray(candidates)
    distances = np.linalg.norm(np.maximum(np.abs(a + times[:, None] * delta) - half, 0), axis=1)
    best = int(np.argmin(distances))
    return float(distances[best]), float(times[best]), inside


def corners_enu(box):
    signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
    hab = signs * box['half'] @ box['axes'] + box['center']
    return np.column_stack((-hab[:,2], -hab[:,0], hab[:,1]))


def plane_dimensions(walls):
    # Mid-plane through both mesh centers. Read inner edge and z extents of
    # actual intersected box edges, including the rounded scene quaternion.
    middle = np.mean([-box['center'][2] for box in walls.values()])
    edges = {}
    for name, box in walls.items():
        vertices = corners_enu(box)
        cut_points = []
        for i, p in enumerate(vertices):
            for bit in (1,2,4):
                j = i ^ bit
                q = vertices[j]
                if i >= j or (p[0]-middle)*(q[0]-middle) >= 0:
                    continue
                cut_points.append(p + (q-p) * (middle-p[0]) / (q[0]-p[0]))
        cut_points = np.array(cut_points)
        assert len(cut_points) == 4
        edges[name] = {'lo':cut_points.min(axis=0), 'hi':cut_points.max(axis=0)}
    return (float(middle),float(edges['negative_y']['hi'][1]),float(edges['positive_y']['lo'][1]),
            float(max(e['lo'][2] for e in edges.values())), float(min(e['hi'][2] for e in edges.values())))


def wall_hits(pos, boxes):
    # Use the prior offline, exact piecewise-quadratic segment-to-box minimum.
    # Bounding boxes are only a cheap reject; near a corner radial distance controls.
    hit = np.zeros(pos.shape[:2], dtype=bool)
    contacts = []
    for drone in range(3):
        for wall, box in boxes.items():
            points = to_box(pos[:,drone], box)
            margin = box['half'] + RADIUS
            low,high = np.minimum(points[:-1],points[1:]), np.maximum(points[:-1],points[1:])
            near = (low < margin).all(axis=-1) & (high > -margin).all(axis=-1)
            for k in np.flatnonzero(near):
                d, _, _ = segment_box(points[k],points[k+1],box['half'])
                if d < RADIUS:
                    hit[k+1,drone] = True
                    contacts.append({'step':int(k+1),'drone':drone+1,'wall':wall,'segment_minimum_m':d})
            dist = np.linalg.norm(np.maximum(np.abs(points[0])-box['half'],0))
            if dist < RADIUS:
                hit[0,drone] = True
                contacts.append({'step':0,'drone':drone+1,'wall':wall,'segment_minimum_m':float(dist)})
    return hit, contacts


def passage_points(pos, dims):
    middle, left, right, bottom, top = dims
    rows = []
    for drone in range(3):
        path = pos[:,drone]
        # Handle exact middle-plane endpoints once (open left and closed right).
        # Record backward passages as well so bypassing in one direction is visible.
        crosses = ((path[:-1,0] < middle) != (path[1:,0] < middle))
        hits = []
        for k in np.flatnonzero(crosses):
            x0,x1 = path[k,0],path[k+1,0]
            u = (middle-x0)/(x1-x0)
            coord = path[k]*(1-u) + path[k+1]*u
            y,z = float(coord[1]),float(coord[2])
            hits.append({'fractional_step':float(k+u), 'direction':'forward' if x1>x0 else 'backward',
                         'y':y,'z':z,'in_aperture':bool(left <= y <= right and bottom <= z <= top)})
        rows.append(hits)
    return rows


def minimum_wall_distances(pos, walls):
    minima = []
    for drone in range(3):
        chosen = None
        for wall, box in walls.items():
            path = to_box(pos[:,drone],box)
            for k,(a,b) in enumerate(zip(path[:-1],path[1:])):
                dist,t,_ = segment_box(a,b,box['half'])
                if chosen is None or dist < chosen['center_to_solid_m']:
                    chosen = {'drone':drone+1,'wall':wall,'center_to_solid_m':dist,'skin_clearance_m':dist-RADIUS,
                              'fractional_step':float(k+t)}
        minima.append(chosen)
    return minima


def classify(pos, walls, dims):
    assert pos.ndim==3 and pos.shape[1:]==(3,3)
    wall_flags, contacts = wall_hits(pos,walls)
    passes = passage_points(pos,dims)
    through = all(any(c['direction']=='forward' for c in path) and
                  all(c['in_aperture'] for c in path) for path in passes)
    past = bool(np.all(pos[-1,:,0] > 10))
    return wall_flags, {
        'wall_contacts':contacts, 'midplane_crossings':passes,
        'all_crossings_in_aperture':through, 'all_past_goal':past,
        'wall_clear_team_pass':bool(through and past and not wall_flags.any()),
    }


def surface_for_endpoint(position, nearest, flags, surfaces):
    matches = []
    for i in np.flatnonzero(flags):
        query = nearest[i]
        distances = np.linalg.norm(position-query,axis=-1)
        distances[i] = np.inf
        mesh = probe(surfaces, query)[0]
        if distances.min() < min(mesh['residual_m'], 1e-5):
            match = f'drone_{int(distances.argmin())+1}'
        elif mesh['residual_m'] < 0.002:
            match = mesh['source']
        else:
            match = 'unclassified'
        matches.append({'drone':int(i+1),'surface':match})
    return matches
