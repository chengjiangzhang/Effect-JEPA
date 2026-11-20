"""Deterministic pedestrian gap acceptance from current measured state only.

This is an executable heuristic, not a collision-avoidance guarantee. Legacy v1
commits after release; v2 continues to evaluate slow/stop/bounded retreat.
No ego action name, desired trajectory, future observation or random generator
is accepted. The same pure functions are replayed by the offline evidence audit.
"""
from __future__ import annotations

import math
import sys
import numpy as np

if __package__:
    from . import pedestrian_crossing
else:
    import pedestrian_crossing

KIND = 'gap_acceptance_v1'
SCHEMA = 'pedestrian-decisions-v1'
CONFLICT_CORRIDOR_M = 1.6
CONFLICT_VERTICAL_SEPARATION_M = 2.5
MOVING_SPEED_M_S = .05
FIELDS = {'kind', 'vehicle_conflicts', 'pedestrian_entry_s', 'pedestrian_exit_s',
          'clearance_time_s', 'clearance_distance_m', 'stopped_speed_m_s',
          'yielding_speed_m_s', 'minimum_deceleration_m_s2'}


def feedback_enabled(spec):
    return spec.get('response_policy') is not None


def decision_schema(specs):
    if any(s.get('response_policy', {}).get('kind') == pedestrian_crossing.KIND_V4 for s in specs):
        return 'pedestrian-decisions-v4'
    if any(s.get('response_policy', {}).get('kind') == pedestrian_crossing.KIND_V3 for s in specs):
        return 'pedestrian-decisions-v3'
    return ('pedestrian-decisions-v2' if any(s.get('response_policy', {}).get('kind') ==
            pedestrian_crossing.KIND for s in specs) else SCHEMA)


def _number(value, label, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f'{label}: expected a finite {"positive" if positive else "nonnegative"} number')
    return float(value)


def route_projection(position, route):
    route = np.asarray(route, dtype=float)
    segments = np.diff(route[:, :2], axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if not len(lengths) or np.any(lengths < 1e-8):
        raise ValueError('Pedestrian feedback requires a nondegenerate declared route')
    fractions = np.clip(np.sum((np.asarray(position)[:2]-route[:-1, :2])*segments, axis=1)/lengths**2, 0, 1)
    closest = route[:-1, :2]+fractions[:, None]*segments
    index = int(np.argmin(np.linalg.norm(closest-np.asarray(position)[:2], axis=1)))
    tangent = segments[index]/lengths[index]
    station = float(np.r_[0., np.cumsum(lengths)][index]+fractions[index]*lengths[index])
    return station, tangent


def crossing_geometry(vehicle_route, pedestrian_route, max_distance_m=CONFLICT_CORRIDOR_M):
    """Continuous segment closest approach; shared compiler/runtime criterion.

    The 1.6m corridor is an authored geometry heuristic, not proof that vehicle
    dimensions or actual trajectories obey the centerline. Runtime separately
    checks actual unlisted vehicle bodies against the remaining walking path.
    """
    vehicle, walker = np.asarray(vehicle_route,float), np.asarray(pedestrian_route,float)
    v, w = vehicle[:,:2], walker[:,:2]
    r, s = np.diff(v,axis=0), np.diff(w,axis=0)
    rv, sw = np.linalg.norm(r,axis=1), np.linalg.norm(s,axis=1)
    if np.any(rv <= 1e-8) or np.any(sw <= 1e-8):
        raise ValueError('Conflict geometry requires nondegenerate route segments')
    vs, ws = np.r_[0.,np.cumsum(rv)], np.r_[0.,np.cumsum(sw)]
    best = None
    def add(t,u,j):
        nonlocal best
        points_v = v[:-1]+t[:,None]*r
        points_w = w[j]+u[:,None]*s[j]
        distances = np.linalg.norm(points_v-points_w,axis=1)
        vz=vehicle[:-1,2]+t*np.diff(vehicle[:,2])
        wz=walker[j,2]+u*(walker[j+1,2]-walker[j,2])
        distances=np.where(np.abs(vz-wz)<=CONFLICT_VERTICAL_SEPARATION_M,distances,np.inf)
        index = int(np.argmin(distances))
        row = {'distance_m':float(distances[index]),'vehicle_s':float(vs[index]+t[index]*rv[index]),
               'pedestrian_s':float(ws[j]+u[index]*sw[j])}
        if best is None or row['distance_m'] < best['distance_m']:
            best = row
    cross = lambda a,b: a[...,0]*b[...,1]-a[...,1]*b[...,0]
    for j,segment in enumerate(s):
        for u0 in (0.,1.):
            t = np.clip(np.sum((w[j]+u0*segment-v[:-1])*r,axis=1)/rv**2,0.,1.)
            add(t,np.full(len(r),u0),j)
        for t0 in (0.,1.):
            u = np.clip(np.sum((v[:-1]+t0*r-w[j])*segment,axis=1)/sw[j]**2,0.,1.)
            add(np.full(len(r),t0),u,j)
        denominator = cross(r,segment)
        valid = np.abs(denominator)>1e-10
        safe = np.where(valid,denominator,1.)
        t = cross(w[j]-v[:-1],segment)/safe
        u = cross(w[j]-v[:-1],r)/safe
        intersect = valid & (t>=0) & (t<=1) & (u>=0) & (u<=1)
        vz=vehicle[:-1,2]+t*np.diff(vehicle[:,2])
        wz=walker[j,2]+u*(walker[j+1,2]-walker[j,2])
        intersect &= np.abs(vz-wz)<=CONFLICT_VERTICAL_SEPARATION_M
        if np.any(intersect):
            k = int(np.flatnonzero(intersect)[0])
            row = {'distance_m':0.,'vehicle_s':float(vs[k]+t[k]*rv[k]),
                   'pedestrian_s':float(ws[j]+u[k]*sw[j])}
            if best is None or row['distance_m'] < best['distance_m']:
                best = row
    return best if best is not None and best['distance_m'] <= max_distance_m else None


def validate_policy(spec, config):
    if not feedback_enabled(spec):
        return
    policy = spec['response_policy']
    version2 = isinstance(policy, dict) and pedestrian_crossing.continuous(policy.get('kind'))
    expected = FIELDS | {'crossing_feedback'} if version2 else FIELDS
    if not isinstance(policy, dict) or set(policy) != expected or policy.get('kind') not in (KIND, pedestrian_crossing.KIND, pedestrian_crossing.KIND_V3, pedestrian_crossing.KIND_V4):
        raise ValueError('Pedestrian response_policy must declare a complete supported versioned contract')
    if version2:
        pedestrian_crossing.validate(policy['crossing_feedback'], policy['kind'])
    if pedestrian_crossing.timed_entry(policy['kind']):
        _number(spec['start_time_s']+2.+float(config.get('driving_warmup_s', 1.))+1e-9, 'v3 earliest decision within auditable common warmup')
    else:
        _number(spec['start_time_s'], 'feedback earliest start_time_s')
    for key in FIELDS-{'kind', 'vehicle_conflicts'}:
        _number(policy[key], key, positive=key in ('clearance_time_s', 'clearance_distance_m', 'minimum_deceleration_m_s2'))
    if not policy['stopped_speed_m_s'] < policy['yielding_speed_m_s'] <= 3.:
        raise ValueError('Require stopped_speed_m_s < yielding_speed_m_s <= 3')
    length = np.linalg.norm(np.diff(np.asarray(spec['route_points'])[:, :2], axis=0), axis=1).sum()
    if not 0 <= policy['pedestrian_entry_s'] < policy['pedestrian_exit_s'] <= length:
        raise ValueError('Pedestrian conflict interval must lie on its route')
    vehicles = {v['role']: v for v in config['vehicles']}
    conflicts = policy['vehicle_conflicts']
    if not isinstance(conflicts, list) or not conflicts:
        raise ValueError('Feedback requires all relevant vehicle conflict intervals')
    seen = set()
    for item in conflicts:
        if not isinstance(item, dict) or set(item) != {'role', 'entry_s', 'exit_s'}:
            raise ValueError('Vehicle conflict requires role/entry_s/exit_s only')
        role = item['role']
        if role not in vehicles or role in seen:
            raise ValueError('Feedback vehicle roles must be existing and unique')
        seen.add(role)
        length = np.linalg.norm(np.diff(np.asarray(vehicles[role]['route'])[:, :2], axis=0), axis=1).sum()
        if not 0 <= _number(item['entry_s'], 'vehicle entry_s') < _number(item['exit_s'], 'vehicle exit_s') <= length:
            raise ValueError('Vehicle conflict interval must lie on its declared route')
    if 'ego' not in seen:
        raise ValueError('Reactive pedestrian must observe ego as a conflict participant')
    required = {role for role,actor in vehicles.items()
                if crossing_geometry(actor['route'],spec['route_points']) is not None}
    if required-seen:
        raise ValueError('Pedestrian feedback omits geometrically conflicting vehicle roles: '+','.join(sorted(required-seen)))


def _rotation(angles):
    roll, pitch, yaw = np.radians(angles)
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr],
                     [cp*sy, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr],
                     [sp, -cp*sr, cp*cr]])


def body_interval(observation, index, route, transform_key):
    """Project measured oriented bbox center/extents, including local offsets."""
    transform = np.asarray(observation[transform_key][index], dtype=float)
    rotation = transform[:3, :3]
    center = transform[:3, 3]+rotation@np.asarray(observation['bbox_offset'][index])
    station, tangent = route_projection(center, route)
    orientation = rotation@_rotation(observation['bbox_rotation_deg'][index])
    half = float(np.abs(np.r_[tangent, 0.]@orientation)@observation['bbox_extent'][index])
    return station-half, station+half, tangent


def unlisted_body_blocks(observed, index, spec, pedestrian_rear):
    """Current measured OBB overlaps the remaining walk corridor, any role."""
    transform = observed['transform_world_vehicle'][index]
    rotation = transform[:3,:3]
    center = transform[:3,3]+rotation@observed['bbox_offset'][index]
    station,tangent = route_projection(center,spec['route_points'])
    route=np.asarray(spec['route_points'],float)
    knots=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1))]
    projected=np.array([np.interp(station,knots,route[:,i]) for i in (0,1)])
    station += float((center[:2]-projected)@tangent)
    normal=np.array([-tangent[1],tangent[0]])
    orientation=rotation@_rotation(observed['bbox_rotation_deg'][index])
    extent=observed['bbox_extent'][index]
    along=float(np.abs(np.r_[tangent,0.]@orientation)@extent)
    lateral=float(np.abs(np.r_[normal,0.]@orientation)@extent)
    vertical=float(np.abs(orientation[2])@extent)
    walk_z=float(np.interp(station,knots,route[:,2]))
    margin=spec['response_policy']['clearance_distance_m']
    intersects = (station+along >= pedestrian_rear-margin and
                  station-along <= spec['response_policy']['pedestrian_exit_s']+margin and
                  abs(float((center[:2]-projected)@normal)) <= lateral+margin and
                  abs(float(center[2]-walk_z)) <= vertical+CONFLICT_VERTICAL_SEPARATION_M)
    return bool(intersects)


def walker_control(position, spec, time_s, dt):
    """Track the fixed polyline from the current location; no future data."""
    result = np.zeros(5, dtype=float)
    if time_s < spec['start_time_s']:
        return result
    route = np.asarray(spec['route_points'], dtype=float)
    lengths = np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)
    stations = np.r_[0., np.cumsum(lengths)]
    station, _ = route_projection(position, route)
    remaining = float(np.linalg.norm(route[-1, :2]-np.asarray(position)[:2]))
    if station >= stations[-1]-.05 and remaining <= .08:
        return result
    target_s = min(station+max(.3, spec['speed_mps']*dt), stations[-1])
    segment = min(int(np.searchsorted(stations, target_s, side='right'))-1, len(lengths)-1)
    target = route[segment, :2]+(target_s-stations[segment])/lengths[segment]*(route[segment+1, :2]-route[segment, :2])
    delta = target-np.asarray(position)[:2]
    norm = float(np.linalg.norm(delta))
    if norm > 1e-8:
        result[:2] = delta/norm
        result[3] = min(spec['speed_mps'], remaining/dt)
    return result


def decision(spec, config, vehicles, people, person_index, time_s, committed=False):
    """Return a branch-blind decision from the same recorded simulation frame."""
    policy = spec['response_policy']
    rear, front, _ = body_interval(people, person_index, spec['route_points'], 'transform_world_actor')
    crossing_time = max(0., policy['pedestrian_exit_s']+policy['clearance_distance_m']-rear)/spec['speed_mps']
    required_gap = crossing_time+policy['clearance_time_s']
    rows = []
    definitions = {v['role']: v for v in config['vehicles']}
    for conflict in policy['vehicle_conflicts']:
        index = config['vehicle_roles'].index(conflict['role'])
        vehicle_rear, vehicle_front, tangent = body_interval(vehicles, index, definitions[conflict['role']]['route'], 'transform_world_vehicle')
        velocity = np.asarray(vehicles['velocity'][index])[:2]
        speed = float(np.linalg.norm(velocity)); forward = float(velocity@tangent)
        acceleration = float(np.asarray(vehicles['acceleration'][index])[:2]@tangent)
        distance = float(conflict['entry_s']-vehicle_front)
        cleared = vehicle_rear > conflict['exit_s']+policy['clearance_distance_m']
        occupied = vehicle_front >= conflict['entry_s']-policy['clearance_distance_m'] and vehicle_rear <= conflict['exit_s']+policy['clearance_distance_m']
        eta = distance/forward if distance > 0 and forward > .05 else None
        stop_distance = forward**2/(2*-acceleration) if forward >= 0 and acceleration <= -policy['minimum_deceleration_m_s2'] else None
        if cleared:
            safe, reason = True, 'rear_physically_cleared'
        elif occupied:
            safe, reason = False, 'vehicle_body_in_conflict_zone'
        elif distance >= policy['clearance_distance_m'] and speed <= policy['stopped_speed_m_s']:
            safe, reason = True, 'observed_stopped_before_conflict'
        elif (distance > policy['clearance_distance_m'] and 0 < forward <= policy['yielding_speed_m_s']
              and speed <= policy['yielding_speed_m_s'] and stop_distance is not None
              and stop_distance+policy['clearance_distance_m'] < distance):
            safe, reason = True, 'observed_slow_deceleration_with_stopping_margin'
        elif eta is not None and eta > required_gap:
            safe, reason = True, 'current_speed_arrival_gap'
        else:
            safe, reason = False, 'approaching_or_not_observed_clear'
        rows.append({'role': conflict['role'], 'safe_gap_observed': safe, 'reason': reason,
            'body_front_route_s_m': vehicle_front, 'body_rear_route_s_m': vehicle_rear,
            'front_to_entry_m': distance, 'speed_m_s': speed, 'forward_speed_m_s': forward,
            'forward_acceleration_m_s2': acceleration, 'entry_eta_s': eta,
            'estimated_braking_distance_m': stop_distance})
    listed={item['role'] for item in policy['vehicle_conflicts']}
    for index,role in enumerate(config['vehicle_roles']):
        if role not in listed:
            blocking=unlisted_body_blocks(vehicles,index,spec,rear)
            rows.append({'role':role,'safe_gap_observed':not blocking,
                'reason':'unlisted_vehicle_body_blocks_walk_path' if blocking else 'unlisted_vehicle_body_outside_walk_path'})
    ready = time_s+1e-9 >= spec['start_time_s']
    release = ready and all(row['safe_gap_observed'] for row in rows)
    allowed = bool(committed or release)
    reason = 'continue_committed_crossing' if committed else 'before_earliest_start' if not ready else 'accept_observed_gap' if release else 'wait_for_observed_gap'
    return {'role': spec['role'], 'kind': KIND, 'reason': reason, 'allow_walk': allowed,
        'committed_before': bool(committed), 'committed_after': allowed,
        'pedestrian_body_front_route_s_m': front, 'pedestrian_body_rear_route_s_m': rear,
        'required_gap_s': required_gap, 'vehicle_checks': rows,
        'safety_guarantee': False}


def controls_and_decisions(config, vehicles, people, time_s, dt, committed, *, final=False):
    controls, decisions = [], []
    next_committed = list(committed)
    for index, spec in enumerate(config.get('pedestrians', [])):
        version2 = pedestrian_crossing.continuous(spec.get('response_policy', {}).get('kind'))
        state = committed[index]
        was_released = state.get('released', False) if isinstance(state, dict) else bool(state)
        if final:
            row = {'role': spec['role'], 'kind': spec['response_policy']['kind'] if feedback_enabled(spec) else 'scripted',
                   'reason': 'unexecuted_final_boundary', 'allow_walk': False,
                   'committed_before': was_released, 'committed_after': was_released}
            if version2:
                state = dict(state) if isinstance(state, dict) else pedestrian_crossing.initial_state()
                row.update(mode='final', state_before=state, state_after=state, signed_command_speed_m_s=0.)
            control = np.zeros(5)
        elif version2:
            control, row, next_committed[index] = pedestrian_crossing.step(
                spec, config, vehicles, people, index, time_s, dt, state, policy_module=sys.modules[__name__])
        elif feedback_enabled(spec):
            row = decision(spec, config, vehicles, people, index, time_s, committed[index])
            next_committed[index] = row['committed_after']
            control = walker_control(people['position'][index], spec, time_s, dt) if row['allow_walk'] else np.zeros(5)
        else:
            control = walker_control(people['position'][index], spec, time_s, dt)
            row = {'role': spec['role'], 'kind': 'scripted', 'reason': 'fixed_start_and_route',
                   'allow_walk': bool(control[3] > 0), 'committed_before': False, 'committed_after': False}
        controls.append(control); decisions.append(row)
    return np.asarray(controls).reshape(-1, 5), decisions, next_committed


def decision_summary(specs, decision_samples, pedestrian_samples):
    """Command release is not measured motion onset; report them separately."""
    summaries={}
    for index,spec in enumerate(specs):
        commanded=next((row['time_s'] for row in decision_samples
            if row['decisions'][index].get('committed_after') and
               not row['decisions'][index].get('committed_before')),None)
        moving=next((float(row['time_s']) for row in pedestrian_samples
                     if np.linalg.norm(row['velocity'][index,:2]) > MOVING_SPEED_M_S),None)
        summaries[spec['role']]={'command_release_time_s':commanded if feedback_enabled(spec) else None,
            'observed_first_moving_time_s':moving,'moving_speed_threshold_m_s':MOVING_SPEED_M_S,
            'motion_definition':'first measured horizontal speed strictly above threshold',
            'reactive_policy':feedback_enabled(spec)}
        if pedestrian_crossing.continuous(spec.get('response_policy', {}).get('kind')):
            changes, last = [], None
            for sample in decision_samples:
                row = sample['decisions'][index]
                if row.get('mode') != last and row.get('mode') != 'final':
                    changes.append({'time_s': sample['time_s'], 'mode': row['mode'], 'reason': row['reason']})
                    last = row['mode']
            stations = [route_projection(s['position'][index], spec['route_points'])[0] for s in pedestrian_samples]
            stopped = next((float(s['time_s']) for s in pedestrian_samples
                if moving is not None and s['time_s'] > moving and np.linalg.norm(s['velocity'][index,:2]) <= MOVING_SPEED_M_S), None)
            summaries[spec['role']].update(commanded_mode_transitions=changes,
                observed_reverse_distance_m=float(np.maximum(0., -np.diff(stations)).sum()),
                observed_stop_after_motion_time_s=stopped,
                motion_feedback_note='Command modes and measured displacement are distinct; local prediction is not a safety guarantee')
    return summaries
