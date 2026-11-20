"""NumPy-only recorded vehicle-box geometry, shared by planning and audits.

Native actor-local bounding boxes are geometric proxies, not collision meshes.
This module does not connect to CARLA, inflate boxes, or infer sensor contacts.
"""
from __future__ import annotations

import itertools
import numpy as np

BOX_ROUNDOFF_M = 1e-6  # Numerical boundary classification, not a safety margin.
HISTORY_GEOMETRY_SCHEMA = 'recorded-common-history-vehicle-obb-v1'


def rotation(degrees):
    roll, pitch, yaw = np.deg2rad(degrees)
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr],
                     [cp*sy, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr],
                     [sp, -cp*sr, cp*cr]])


def ego_bbox(state, ego_index=0):
    """Use measured transforms and actor-local box offsets/rotations, not a point car."""
    extent = np.asarray(state['bbox_extent'])[ego_index]
    offset = np.asarray(state['bbox_offset'])[ego_index]
    angle = np.asarray(state['bbox_rotation_deg'])[ego_index]
    transforms = np.asarray(state['transform_world_vehicle'])[:, ego_index]
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
    local = (signs*extent) @ rotation(angle).T+offset
    corners = np.einsum('tij,kj->tki', transforms[:, :3, :3], local)+transforms[:, None, :3, 3]
    center = np.einsum('tij,j->ti', transforms[:, :3, :3], offset)+transforms[:, :3, 3]
    return center, corners


def box_sat_gap(a, b):
    """3D oriented-box separating-axis gap; its sign tests box intersection.

    Corners follow itertools.product((-1, 1), repeat=3), as in ego_bbox.
    The gap is not Euclidean object/mesh distance or a collision-sensor result.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != (8, 3) or b.shape != (8, 3) or not np.isfinite([a, b]).all():
        raise ValueError('Require two finite eight-corner boxes')
    edges = [p[[4, 2, 1]]-p[0] for p in (a, b)]
    norms = [np.linalg.norm(e, axis=1) for e in edges]
    if any(np.any(n <= 1e-12) for n in norms):
        raise ValueError('Degenerate vehicle box')
    u, v = (e/n[:, None] for e, n in zip(edges, norms))
    if not all(np.allclose(e@e.T, np.eye(3), rtol=0, atol=1e-5) for e in (u, v)):
        raise ValueError('Vehicle-box axes must be orthogonal')
    candidates = np.vstack((u, v, np.cross(u[:, None, :], v[None, :, :]).reshape(-1, 3)))
    lengths = np.linalg.norm(candidates, axis=1)
    axes = candidates[lengths > 1e-10]/lengths[lengths > 1e-10, None]
    pa, pb = a@axes.T, b@axes.T
    return float(np.maximum(pa.min(axis=0)-pb.max(axis=0), pb.min(axis=0)-pa.max(axis=0)).max())


def common_history_vehicle_geometry(states, roles):
    """Check all recorded vehicle bodies at -2..0 s, including t0, only.

    Full replays may include a future; only the required 41-frame prefix is
    inspected. Missing geometry or incomplete history raises ValueError. Valid
    measurements with interior box overlap return a failed geometry report;
    this is distinct from a native collision-sensor event.
    """
    if (not isinstance(roles, (list, tuple)) or not roles
            or any(not isinstance(role, str) or not role for role in roles)
            or len(set(roles)) != len(roles)):
        raise ValueError('Common-history geometry requires distinct recorded vehicle roles')
    times = np.asarray(states.get('time_s'), dtype=float)
    expected = np.arange(41)*.05-2.
    if (times.ndim != 1 or len(times) < 41 or not np.isfinite(times[:41]).all()
            or not np.allclose(times[:41], expected, atol=1e-8, rtol=0)):
        raise ValueError('Common-history geometry requires all 41 observed times from -2 through t0')
    n, count = len(times), len(roles)
    transforms = np.asarray(states.get('transform_world_vehicle'), dtype=float)
    frames = np.asarray(states.get('frame_id'))
    if transforms.shape != (n, count, 4, 4) or not np.isfinite(transforms[:41]).all():
        raise ValueError('Common-history geometry requires finite measured vehicle-to-world transforms')
    if (frames.shape != (n,) or not np.isfinite(frames[:41]).all()
            or not np.equal(frames[:41], np.round(frames[:41])).all()):
        raise ValueError('Common-history geometry requires recorded integer frame IDs')
    transforms = transforms[:41]
    rotation_matrices = transforms[:, :, :3, :3]
    if (not np.allclose(transforms[:, :, 3, :], [0., 0., 0., 1.], atol=1e-8, rtol=0)
            or not np.allclose(rotation_matrices @ np.swapaxes(rotation_matrices, -1, -2),
                               np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(rotation_matrices), 1., atol=1e-5, rtol=0)):
        raise ValueError('Common-history vehicle transforms must be rigid, right-handed homogeneous matrices')
    prefix = {'transform_world_vehicle': transforms}
    for key in ('bbox_extent', 'bbox_offset', 'bbox_rotation_deg'):
        value = np.asarray(states.get(key), dtype=float)
        if value.shape != (count, 3) or not np.isfinite(value).all():
            raise ValueError('Missing or invalid recorded vehicle geometry: ' + key)
        if key == 'bbox_extent' and np.any(value <= 0):
            raise ValueError('Recorded vehicle extents must be strictly positive')
        prefix[key] = value
    corners = [ego_bbox(prefix, index)[1] for index in range(count)]
    overlaps, minimum = [], None
    touching, comparisons = 0, 0
    for index in range(41):
        for i, j in itertools.combinations(range(count), 2):
            gap = box_sat_gap(corners[i][index], corners[j][index])
            row = {'roles': [roles[i], roles[j]], 'sample_index': index,
                   'time_s': float(times[index]), 'frame_id': int(frames[index]), 'sat_gap_m': gap}
            comparisons += 1
            if minimum is None or gap < minimum['sat_gap_m']:
                minimum = row
            if gap < -BOX_ROUNDOFF_M:
                overlaps.append(row)
            elif gap <= BOX_ROUNDOFF_M:
                touching += 1
    return {'schema': HISTORY_GEOMETRY_SCHEMA, 'status': 'failed' if overlaps else 'passed',
            'passed': not overlaps, 'recorded_interval_s': [-2., 0.], 't0_included': True,
            'history_samples_checked': 41, 'vehicle_roles': list(roles),
            'pair_observations_checked': comparisons,
            'method': '15-axis 3D SAT using measured transforms and native local bbox extents/offsets/rotations',
            'bbox_inflation_m': 0., 'numerical_boundary_roundoff_m': BOX_ROUNDOFF_M,
            'failure_rule': 'sat_gap_m < -numerical_boundary_roundoff_m',
            'minimum_sat_gap_m': minimum['sat_gap_m'] if minimum else None,
            'minimum_pair_observation': minimum, 'interior_overlap_observation_count': len(overlaps),
            'interior_overlap_pair_count': len({tuple(row['roles']) for row in overlaps}),
            'interior_overlaps': overlaps, 'touching_or_roundoff_observation_count': touching,
            'future_observations_examined': False,
            'meaning': 'Recorded bounding-box interior overlap is invalid shared setup; it is not a native mesh-contact or collision-sensor assertion.'}
