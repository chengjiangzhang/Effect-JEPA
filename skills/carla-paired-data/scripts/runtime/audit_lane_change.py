#!/usr/bin/env python3
"""Audit measured lateral trajectories against declared adjacent-lane geometry.

No CARLA server is used. CARLA's offline Map API is used only by the CLI for
OpenDRIVE topology diagnostics. Pure trajectory checks are NumPy-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
if __package__:
    from .controllers_v2 import branch_names
    from .vehicle_geometry import ego_bbox, rotation
else:
    from controllers_v2 import branch_names
    from vehicle_geometry import ego_bbox, rotation


def lateral_branches(config):
    """Only planned lateral interventions require lane-change certification."""
    return tuple(b for b in branch_names(config, False) if lane_direction(config, b) is not None)


def lane_direction(config, branch):
    """Resolve physical direction independently from arbitrary condition names."""
    profiles = config.get('action', {}).get('profiles')
    if profiles is None:
        return branch if branch in ('left', 'right') else None
    profile = profiles.get(branch, {})
    if profile.get('kind') != 'lane_change':
        return None
    direction = profile.get('direction')
    if direction not in ('left', 'right'):
        raise ValueError(f'{branch}: lane-change profile has no valid direction')
    return direction


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(points, route):
    """Project XY points onto a polyline; retain correspondence and local normals."""
    points, route = np.asarray(points, float), np.asarray(route, float)
    if route.ndim != 2 or route.shape[1] < 2 or len(route) < 2:
        raise ValueError('Route requires at least two XY points')
    edges = np.diff(route[:, :2], axis=0)
    squared = np.sum(edges**2, axis=1)
    if not np.isfinite(route).all() or np.any(squared < 1e-12):
        raise ValueError('Route must be finite with nonzero XY segments')
    delta = points[:, None, :2]-route[None, :-1, :2]
    fraction = np.clip(np.einsum('nsi,si->ns', delta, edges)/squared, 0., 1.)
    closest = route[None, :-1, :2]+fraction[..., None]*edges
    distance = np.linalg.norm(points[:, None, :2]-closest, axis=-1)
    index = distance.argmin(axis=1)
    rows = np.arange(len(points))
    fraction = fraction[rows, index]
    closest = closest[rows, index]
    tangent = edges[index]/np.sqrt(squared[index, None])
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    offset = np.einsum('ni,ni->n', points[:, :2]-closest, normal)
    return index, fraction, closest, tangent, normal, offset




def sustained_duration(times, mask):
    maximum, start = 0., None
    for t, valid in zip(times, mask):
        if valid:
            start = t if start is None else start
            maximum = max(maximum, float(t-start))
        else:
            start = None
    return maximum


def trajectory_audit(state, config, branch, center_tolerance_m=.5,
                     heading_tolerance_deg=10., hold_s=.5, cutin_margin_m=.25):
    """Check actual progress, cut-in and end-of-record completion independently.

    New config contract: action.lateral_mode='adjacent_lane_route';
    action.lane_changes[side].target_route has the same station correspondence as
    vehicles[ego].route; evidence.target_lane_widths_m gives per-station width.
    Legacy width_m is accepted only when per-station evidence is absent. These thresholds are explicit pilot
    acceptance settings, not a universal definition of safe driving.
    """
    times = np.asarray(state['time_s'], float)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Recorded times must be strictly increasing and finite')
    ego = config['vehicle_roles'].index('ego')
    route = np.asarray(config['vehicles'][ego]['route'], float)
    origin = np.asarray(state['position'])[:, ego]
    offset = project(origin, route)[-1]
    post = times >= -1e-8
    if not post.any():
        raise ValueError('No post-intervention measurements')
    direction = lane_direction(config, branch)
    result = {
        'branch': branch, 'direction': direction,
        'configured_lateral_mode': config['action'].get('lateral_mode', 'route_offset'),
        'configured_lateral_offset_m': config['action'].get('lateral_offset_m'),
        'final_source_route_offset_m': float(offset[-1]),
        'max_abs_source_route_offset_m': float(abs(offset[post]).max()),
        'post_intervention_frames': int(post.sum()),
        'target_declared': direction in config['action'].get('lane_changes', {}),
        'clear_cutin': False, 'completed': False,
    }
    target_spec = config['action'].get('lane_changes', {}).get(direction)
    if config['action'].get('lateral_mode') != 'adjacent_lane_route' or target_spec is None:
        result.update(status='no_declared_adjacent_lane_target',
                      reason='Legacy offset policy has no specified adjacent driving-lane route; cannot certify a lane change.')
        return result
    target = np.asarray(target_spec['target_route'], float)
    if target.shape != route.shape:
        raise ValueError('Source and target routes require identical station correspondence')
    if target_spec.get('available') is False:
        raise ValueError('Requested adjacent lane is explicitly unavailable')
    evidence = target_spec.get('evidence', {})
    width = np.asarray(evidence.get('target_lane_widths_m', target_spec.get('width_m')), float)
    if width.ndim == 0:
        width = np.full(len(route), float(width))
    if width.shape != (len(route),) or not np.isfinite(width).all() or (width <= 0).any():
        raise ValueError('Target width_m must be positive scalar or one width per station')
    center, corners = ego_bbox(state, ego)
    si, sf, source_xy, _, source_normal, _ = project(center, route)
    target_at_source = target[si, :2]*(1-sf[:, None])+target[si+1, :2]*sf[:, None]
    signed_separation = np.einsum('ni,ni->n', target_at_source-source_xy, source_normal)
    side = -1. if direction == 'left' else 1.
    if np.any(side*signed_separation[post] <= 1.5):
        raise ValueError('Target route must lie on requested side at a distinct lane center')
    width_at_source = width[si]*(1-sf)+width[si+1]*sf
    # Shared source/target boundary from the target center and its measured lane
    # width; midpoint alone would be wrong when adjacent lane widths differ.
    boundary = abs(signed_separation)-width_at_source/2
    if np.any(boundary[post] <= 0):
        raise ValueError('Target lane boundary overlaps source lane center')
    center_toward_target = side*np.einsum('ni,ni->n', center[:, :2]-source_xy, source_normal)
    ti, tf, target_xy, tangent, _, target_offset = project(center, target)
    width_at_target = width[ti]*(1-tf)+width[ti+1]*tf
    corner_flat = corners.reshape(-1, 3)
    ci, cf, corner_xy, corner_tangent, _, corner_offsets = project(corner_flat, target)
    corner_width = (width[ci]*(1-cf)+width[ci+1]*cf).reshape(len(times), 8)
    corner_offsets = corner_offsets.reshape(len(times), 8)
    corner_longitudinal_error = abs(np.einsum('ni,ni->n', corner_flat[:, :2]-corner_xy, corner_tangent)).reshape(len(times), 8)
    contained = ((abs(corner_offsets) <= corner_width/2+.02) & (corner_longitudinal_error <= .02)).all(axis=1)
    yaw = np.asarray(state['rotation_deg'])[:, ego, 2]
    target_yaw = np.rad2deg(np.arctan2(tangent[:, 1], tangent[:, 0]))
    heading_error = abs((yaw-target_yaw+180) % 360-180)
    center_longitudinal_error = abs(np.einsum('ni,ni->n', center[:, :2]-target_xy, tangent))
    cutin = post & (center_toward_target >= boundary+cutin_margin_m) & (abs(target_offset) <= width_at_target/2) & (center_longitudinal_error <= .02)
    finished = post & contained & (abs(target_offset) <= center_tolerance_m) & (heading_error <= heading_tolerance_deg)
    final_window = (times >= times[-1]-hold_s-1e-8) & post
    full_window = bool(times[final_window][-1]-times[final_window][0] >= hold_s-1e-8)
    completed = full_window and bool(finished[final_window].all())
    clear_cutin = sustained_duration(times, cutin) >= hold_s-1e-8
    result.update(
        status='completed' if completed else ('clear_cutin' if clear_cutin else 'not_clearly_entered_target_lane'),
        clear_cutin=bool(clear_cutin), completed=bool(completed),
        final_target_center_error_m=float(abs(target_offset[-1])),
        final_heading_error_deg=float(heading_error[-1]),
        final_bbox_within_target_lane=bool(contained[-1]),
        maximum_sustained_cutin_s=sustained_duration(times, cutin),
        maximum_sustained_completion_s=sustained_duration(times, finished),
        target_geometry_evidence={k: v for k, v in target_spec.items() if k not in ('target_route', 'width_m')},
        thresholds={'center_tolerance_m': center_tolerance_m, 'heading_tolerance_deg': heading_tolerance_deg,
                    'hold_s': hold_s, 'cutin_center_beyond_shared_boundary_m': cutin_margin_m,
                    'bbox_lateral_numeric_tolerance_m': .02},
        frames=[{'time_s': float(t), 'source_route_offset_m': float(o),
                 'center_beyond_shared_boundary_m': float(b), 'target_center_error_m': float(e),
                 'heading_error_deg': float(h), 'bbox_inside_target_lane': bool(ins),
                 'clear_cutin_geometry': bool(c), 'completion_geometry': bool(f)}
                for t, o, b, e, h, ins, c, f in zip(times[post], offset[post],
                    (center_toward_target-boundary)[post], abs(target_offset[post]),
                    heading_error[post], contained[post], cutin[post], finished[post])])
    return result


def waypoint_record(waypoint):
    if waypoint is None:
        return None
    return {'road_id': int(waypoint.road_id), 'section_id': int(waypoint.section_id),
            'lane_id': int(waypoint.lane_id), 'width_m': float(waypoint.lane_width),
            'lane_type': str(waypoint.lane_type), 'yaw_deg': float(waypoint.transform.rotation.yaw),
            'is_junction': bool(waypoint.is_junction)}


def same_direction_driving(waypoint, other, carla):
    return bool(other is not None and other.lane_type == carla.LaneType.Driving
                and abs((other.transform.rotation.yaw-waypoint.transform.rotation.yaw+180) % 360-180) < 30)


def topology_at(cmap, position, carla):
    w = cmap.get_waypoint(carla.Location(*map(float, position)), project_to_road=True,
                         lane_type=carla.LaneType.Driving)
    result = {'source': waypoint_record(w)}
    for side in ('left', 'right'):
        other = getattr(w, 'get_'+side+'_lane')()
        result[side] = waypoint_record(other)
        result[side+'_same_direction_driving'] = same_direction_driving(w, other, carla)
    return result


def precontact_geometry(report, collisions):
    """Do not count collision-displaced poses as pre-contact maneuver evidence."""
    first = min((float(event['time_s']) for event in collisions), default=None)
    frames = [row for row in report.get('frames', [])
              if first is None or row['time_s'] < first-1e-8]
    times = np.array([row['time_s'] for row in frames])
    hold = report.get('thresholds', {}).get('hold_s', .5)
    durations = {}
    for name, key in (('clear_cutin', 'clear_cutin_geometry'), ('completion', 'completion_geometry')):
        duration = sustained_duration(times, np.array([row[key] for row in frames], dtype=bool)) if frames else 0.
        durations['maximum_sustained_'+name+'_s'] = duration
        durations[name+'_held_before_contact'] = duration >= hold-1e-8
    return {'first_branch_contact_time_s': first, 'post_t0_frames_before_contact': len(frames),
            'definition': 'Only frames strictly before the first branch contact; full-sequence geometric results remain separate.',
            **durations}


def audit_target_geometry(cmap, config, branch, carla):
    """Re-read map geometry independently instead of trusting config evidence."""
    direction = lane_direction(config, branch)
    spec = config['action'].get('lane_changes', {}).get(direction)
    if spec is None:
        return {'passed': False, 'errors': ['No declared target route']}
    source = np.asarray(config['vehicles'][config['vehicle_roles'].index('ego')]['route'], float)
    target = np.asarray(spec['target_route'], float)
    evidence = spec.get('evidence', {})
    errors, maximum_center_error, maximum_width_error, maximum_lookup_longitudinal_delta = [], 0., 0., 0.
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 4:
        return {'passed': False, 'errors': ['Source/target station correspondence malformed']}
    for name in ('source_lane_keys', 'target_lane_keys', 'source_lane_widths_m', 'target_lane_widths_m'):
        if name not in evidence or len(evidence[name]) != len(source):
            errors.append(f'Missing or wrong-length {name}')
    if errors:
        return {'passed': False, 'errors': errors}
    endpoint_fallbacks = []
    endpoints = None
    def lookup(prefix, row, index):
        nonlocal endpoints
        station_key = prefix+'_opendrive_s_m'
        if station_key not in evidence:
            return cmap.get_waypoint(carla.Location(*map(float, row[:3])), project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
        road, section, lane = map(int, evidence[prefix+'_lane_keys'][index])
        station = float(evidence[station_key][index])
        waypoint = cmap.get_waypoint_xodr(road, lane, station)
        if waypoint is not None:
            return waypoint
        # The CARLA API can reject exact road-length endpoints even though
        # junction.get_waypoints exposes the original valid endpoint object.
        # Recover only that actual object with identical key/s, never invent a
        # route extension or substitute a nearby driving lane.
        if endpoints is None:
            junctions = {w.junction_id: w.get_junction() for w in cmap.generate_waypoints(4.) if w.is_junction}
            endpoints = [w for j in junctions.values() for pair in j.get_waypoints(carla.LaneType.Driving) for w in pair]
        matches = [w for w in endpoints if (w.road_id, w.section_id, w.lane_id) == (road, section, lane)
                   and abs(w.s-station) < 1e-6]
        if matches:
            endpoint_fallbacks.append({'prefix': prefix, 'station_index': index,
                                      'road_section_lane': [road, section, lane], 's_m': station})
            return matches[0]
        return None
    for index, (s, t) in enumerate(zip(source, target)):
        sw, tw = lookup('source', s, index), lookup('target', t, index)
        if sw is None or tw is None:
            errors.append(f'Station {index}: source/target OpenDRIVE station cannot be independently resolved')
            continue
        actual_neighbor = getattr(sw, 'get_'+direction+'_lane')()
        if not same_direction_driving(sw, actual_neighbor, carla):
            errors.append(f'Station {index}: requested side has no same-direction Driving neighbor')
        else:
            location = actual_neighbor.transform.location
            yaw = np.deg2rad(actual_neighbor.transform.rotation.yaw)
            delta = t[:3]-[location.x, location.y, location.z]
            adjacent_error = float(np.hypot(delta[:2] @ [-np.sin(yaw), np.cos(yaw)], delta[2]))
            if adjacent_error > .25:
                errors.append(f'Station {index}: target is not requested adjacent lane center ({adjacent_error:.3f} m)')
        for prefix, row, waypoint in (('source', s, sw), ('target', t, tw)):
            xyz = waypoint.transform.location
            yaw = np.deg2rad(waypoint.transform.rotation.yaw)
            delta = row[:3]-[xyz.x, xyz.y, xyz.z]
            # CARLA's inverse world->s map lookup approximates longitudinal s
            # (0.4 m errors are possible on these curved road assets). The
            # relevant lane-center error is lateral/vertical, not longitudinal.
            distance = float(np.hypot(delta[:2] @ [-np.sin(yaw), np.cos(yaw)], delta[2]))
            maximum_lookup_longitudinal_delta = max(maximum_lookup_longitudinal_delta,
                abs(float(delta[:2] @ [np.cos(yaw), np.sin(yaw)])))
            maximum_center_error = max(maximum_center_error, distance)
            identity = [int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id)]
            if identity != list(evidence[prefix+'_lane_keys'][index]):
                errors.append(f'Station {index}: {prefix} lane identity disagrees with XODR')
            if distance > .25:
                errors.append(f'Station {index}: {prefix} route is {distance:.3f} m from map lane center')
            heading = abs((row[3]-waypoint.transform.rotation.yaw+180) % 360-180)
            if heading > 5.:
                errors.append(f'Station {index}: {prefix} heading disagrees with map')
            width_error = abs(float(evidence[prefix+'_lane_widths_m'][index])-float(waypoint.lane_width))
            maximum_width_error = max(maximum_width_error, width_error)
            if width_error > .01:
                errors.append(f'Station {index}: {prefix} width disagrees with map')
    return {'passed': not errors, 'errors': errors, 'stations_checked': len(source),
            'maximum_lateral_vertical_center_error_m': maximum_center_error, 'maximum_width_error_m': maximum_width_error,
            'maximum_inverse_map_longitudinal_delta_m': maximum_lookup_longitudinal_delta,
            'original_junction_endpoint_fallbacks': endpoint_fallbacks,
            'thresholds': {'lateral_vertical_center_error_m': .25, 'width_error_m': .01, 'heading_error_deg': 5.},
            'method': 'Offline CARLA Map nearest lane, map neighbor lookup on requested side, lane identity, width and heading rechecked independently from declared values.'}


def audit_dataset(data):
    import carla  # Offline Map API only. Never construct a Client.
    result = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'data_root': str(data.resolve()), 'audit_source_sha256': sha256(__file__),
              'method': 'Actual per-frame ego trajectory projected onto its original route local normal; declared target route and actual bounding boxes required for lane-change certification. No world-Y shortcut.',
              'topology_note': 'Adjacent waypoint query at measured maintain t0; junction connector absence does not by itself prove no physically reachable lane. Full-map scan covers non-junction driving lanes at 2 m spacing.',
              'limitations': 'Geometric completion is not collision-free or legal driving. A declared target must also be validated as an actual same-direction driving lane by planning; collision sensor records remain separate.',
              'maps': {}, 'scenes': [], 'errors': []}
    maps = {}
    manifest_path = data/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    planned_scenes = manifest.get('scene_ids', sorted(p.name for p in data.glob('scene_*') if p.is_dir()))
    if not planned_scenes:
        result['errors'].append('No planned scenes')
    for scene_id in planned_scenes:
        folder = data/scene_id
        config_path = folder/'spec.json'
        if not config_path.exists():
            result['errors'].append(f'{scene_id}: missing frozen config')
            continue
        config = json.loads(config_path.read_text())
        roles = config['vehicle_roles']
        if (not roles or roles[0] != 'ego' or roles.count('ego') != 1 or len(set(roles)) != len(roles)
                or roles != [v.get('role') for v in config['vehicles']]):
            result['errors'].append(f'{scene_id}: invalid vehicle roles; sole ego must be index0')
            continue
        planned = lateral_branches(config)
        declared = manifest.get('scene_branches', {}).get(scene_id)
        if declared is not None and list(branch_names(config)) != declared:
            result['errors'].append(f'{scene_id}: manifest/config branch mismatch')
        extras = [b for b in ('left', 'right') if (folder/'branches'/b).exists() and b not in planned]
        if extras:
            result['errors'].append(f'{scene_id}: unplanned lateral captures {extras}')
        map_path = data/config['map_geometry_path']
        name = config['map']
        if name not in maps:
            maps[name] = carla.Map(name, map_path.read_text())
            counts = Counter()
            for w in maps[name].generate_waypoints(2.):
                if w.is_junction or w.lane_type != carla.LaneType.Driving:
                    continue
                left = same_direction_driving(w, w.get_left_lane(), carla)
                right = same_direction_driving(w, w.get_right_lane(), carla)
                counts['both' if left and right else ('left_only' if left else ('right_only' if right else 'neither'))] += 1
            result['maps'][name] = {'xodr_sha256': sha256(map_path), 'nonjunction_sample_distance_m': 2.,
                                    'adjacent_same_direction_driving_counts': {key: counts[key] for key in ('both', 'left_only', 'right_only', 'neither')}}
        scene = {'scene_id': folder.name, 'family': config['family'], 'config_file_sha256': sha256(config_path),
                 'planned_lateral_branches': list(planned), 'branches': {}}
        maintain_file = folder/'branches/maintain/states.npz'
        if maintain_file.exists():
            with np.load(maintain_file, allow_pickle=False) as state:
                index = int(np.argmin(abs(state['time_s'])))
                scene['measured_t0_topology'] = topology_at(maps[name], state['position'][index, roles.index('ego')], carla)
        for branch in planned:
            state_path = folder/'branches'/branch/'states.npz'
            if not state_path.exists():
                scene['branches'][branch] = {'status': 'missing', 'completed': False}
                continue
            with np.load(state_path, allow_pickle=False) as state:
                report = trajectory_audit(state, config, branch)
                # Independent legacy diagnostic: real actor bbox center versus
                # the source driving-lane boundary, using offline map widths.
                ego = config['vehicle_roles'].index('ego')
                center, _ = ego_bbox(state, ego)
                route = np.asarray(config['vehicles'][ego]['route'])
                si, sf, nearest_xy, _, _, center_offsets = project(center, route)
                nearest_z = route[si, 2]*(1-sf)+route[si+1, 2]*sf
                widths = []
                for xy, z in zip(nearest_xy, nearest_z):
                    w = maps[name].get_waypoint(carla.Location(float(xy[0]), float(xy[1]), float(z)),
                        project_to_road=True, lane_type=carla.LaneType.Driving)
                    widths.append(float(w.lane_width))
                post = state['time_s'] >= -1e-8
                toward = (-1 if lane_direction(config, branch) == 'left' else 1)*center_offsets
                beyond = toward-np.asarray(widths)/2
                report['source_lane_width_range_m'] = [float(min(widths)), float(max(widths))]
                report['bbox_center_crossed_source_boundary'] = bool((beyond[post] > 0).any())
                report['bbox_center_max_beyond_source_boundary_m'] = float(beyond[post].max())
            meta = json.loads((state_path.parent/'meta.json').read_text())
            report['states_sha256'] = sha256(state_path)
            report['collision_sensor_records'] = len(meta.get('collisions', []))
            report['precontact_geometry'] = precontact_geometry(report, meta.get('collisions', []))
            if report['target_declared']:
                report['target_map_geometry'] = audit_target_geometry(maps[name], config, branch, carla)
                if not report['target_map_geometry']['passed']:
                    report['trajectory_geometry_status_before_map_check'] = report['status']
                    report.update(status='invalid_target_map_geometry', completed=False, clear_cutin=False)
            scene['branches'][branch] = report
        result['scenes'].append(scene)
    branches = [b for s in result['scenes'] for b in s['branches'].values()]
    offsets = [abs(b['final_source_route_offset_m']) for b in branches if 'final_source_route_offset_m' in b]
    result['summary'] = {'scene_count': len(result['scenes']), 'lateral_branch_count': len(branches),
        'expected_scene_count': len(planned_scenes),
        'planned_lateral_branch_count': len(branches),
        'collected_lateral_branch_count': sum(b['status'] != 'missing' for b in branches),
        'missing_lateral_branch_count': sum(b['status'] == 'missing' for b in branches),
        'declared_target_branch_count': sum(bool(b.get('target_declared')) for b in branches),
        'completed_count': sum(bool(b['completed']) for b in branches),
        'contact_free_completed_count': sum(bool(b['completed']) and b.get('collision_sensor_records') == 0 for b in branches),
        'precontact_clear_cutin_count': sum(bool(b.get('precontact_geometry', {}).get('clear_cutin_held_before_contact')) for b in branches),
        'precontact_completion_count': sum(bool(b.get('precontact_geometry', {}).get('completion_held_before_contact')) for b in branches),
        'clear_cutin_count': sum(bool(b.get('clear_cutin')) for b in branches),
        'bbox_center_crossed_source_boundary_count': sum(bool(b.get('bbox_center_crossed_source_boundary')) for b in branches),
        'final_absolute_original_route_offset_range_m': [min(offsets), max(offsets)] if offsets else None,
        't0_dual_same_direction_neighbor_scene_count': sum(
            bool(s.get('measured_t0_topology', {}).get('left_same_direction_driving')) and
            bool(s.get('measured_t0_topology', {}).get('right_same_direction_driving')) for s in result['scenes'])}
    result['lane_change_applicability'] = 'planned_lateral_branches' if branches else 'no_lateral_branches'
    invalid = any(b.get('status') != 'missing' and not b.get('completed') for b in branches)
    result['status'] = ('failed' if result['errors'] or invalid else 'incomplete'
                        if result['summary']['missing_lateral_branch_count'] else 'passed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = audit_dataset(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report['summary'], indent=2))
    print(args.out.resolve())
    if report['status'] != 'passed':
        raise SystemExit(2 if report['status'] == 'incomplete' else 1)


if __name__ == '__main__':
    main()
