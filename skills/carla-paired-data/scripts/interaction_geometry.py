"""Offline map evidence and config construction for signal, walker and meeting scenes.

No CARLA client connection is made here. Compiled geometry is a production plan,
not evidence that a scene has passed physical or image validation.
"""
from __future__ import annotations

import copy
import itertools
import math

import carla
import numpy as np

# Candidate IDs listed in the official pedestrian catalogue. Availability and
# actual dimensions are checked by the runtime; this is not native validation.
WALKER_BLUEPRINT_CANDIDATES = ('walker.pedestrian.0001', 'walker.pedestrian.0002',
                             'walker.pedestrian.0004', 'walker.pedestrian.0007')
WAITING_CLEARANCE_M = .5


def route_body_samples(route, half_extents=(2.8, 1.2), *, step=.5,
                       bbox_offset=(0., 0.), bbox_yaw_deg=0.):
    """Conservative continuous swept OBB cover, using route poses, not outcomes."""
    route = np.asarray(route, dtype=float)
    if route.ndim != 2 or len(route) < 2 or route.shape[1] < 3 or not np.isfinite(route).all():
        raise ValueError('Swept-body route must have finite nondegenerate segments')
    lengths = np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)
    if np.any(lengths <= 1e-8) or not math.isfinite(step) or step <= 0:
        raise ValueError('Swept-body route must have finite nondegenerate segments')
    stations = np.r_[0., np.cumsum(lengths)]
    headings = (np.radians(route[:, 3]) if route.shape[1] >= 4 else
                np.r_[np.arctan2(np.diff(route[:,1]),np.diff(route[:,0])),
                      math.atan2(route[-1,1]-route[-2,1],route[-1,0]-route[-2,0])])
    headings = np.unwrap(headings)
    # Each interval stays inside one polyline segment. The cover is local:
    # a bend far away must not inflate a straight meeting lane's width.
    boundaries = np.unique(np.r_[np.arange(0., stations[-1], step), stations])
    left_s, right_s = boundaries[:-1], boundaries[1:]
    sample_s = (left_s+right_s)/2
    angle = np.interp(sample_s, stations, headings)
    forward = np.stack((np.cos(angle),np.sin(angle)),axis=-1)
    right = np.stack((-np.sin(angle),np.cos(angle)),axis=-1)
    centre = np.stack((np.interp(sample_s,stations,route[:,0]),
                       np.interp(sample_s,stations,route[:,1])),axis=-1)
    offset = np.asarray(bbox_offset,dtype=float)[:2]
    centre += offset[0]*forward+offset[1]*right
    body_angle = angle+math.radians(bbox_yaw_deg)
    forward = np.stack((np.cos(body_angle),np.sin(body_angle)),axis=-1)
    right = np.stack((-np.sin(body_angle),np.cos(body_angle)),axis=-1)
    half = np.asarray(half_extents,dtype=float)[:2]
    if half.shape != (2,) or np.any(half <= 0) or not np.isfinite(half).all():
        raise ValueError('Positive finite body half-extents are required')
    # Linear centre translation projected onto each OBB axis, plus a rigorous
    # arc-length bound for body/offset rotation over this half interval.
    half_translation = np.stack([np.interp(right_s,stations,route[:,i])
                                 -np.interp(left_s,stations,route[:,i]) for i in (0,1)],axis=-1)/2
    translation_cover = np.stack((abs(np.einsum('ij,ij->i',half_translation,forward)),
                                  abs(np.einsum('ij,ij->i',half_translation,right))),axis=-1)
    half_angle = abs(np.interp(right_s,stations,headings)-np.interp(left_s,stations,headings))/2
    rotation_cover = (float(np.linalg.norm(half))+float(np.linalg.norm(offset)))*half_angle
    guard = float(np.max(np.linalg.norm(half_translation,axis=-1)+rotation_cover))
    return {'station':sample_s,'centre':centre,'forward':forward,'right':right,
            'half':half[None,:]+translation_cover+rotation_cover[:,None],
            'sampling_guard_m':guard}


def _body_parameters(role, measured_actors):
    if measured_actors is not None:
        actor = measured_actors[role]
        return (np.asarray(actor['bbox_extent_m'])[:2], actor.get('bbox_offset_m',[0.,0.])[:2],
                actor.get('bbox_rotation_deg',[0.,0.,0.])[2])
    return ((.4,.4) if role.startswith('pedestrian:') else (2.8,1.2)), (0.,0.), 0.


def waiting_position_check(cfg, station, roles, *, measured_actors=None,
                           clearance_m=WAITING_CLEARANCE_M):
    """Entire parked ego body must avoid each target's full route sweep.

    The static envelope is a conservative planning assumption. A measured-t0
    invocation supplies observed extents, offsets and box rotation instead.
    No target is removed because it is stationary or has no predicted ETA.
    """
    vehicles = {a['role']:a for a in cfg['vehicles']}
    walkers = {a['role']:a for a in cfg.get('pedestrians',[])}
    ego = vehicles['ego']; route = np.asarray(ego['route'],float)
    stations = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1))]
    if not 0 <= station <= stations[-1]:
        raise ValueError('Waiting station is outside the frozen ego route')
    yaw = float(np.interp(station,stations,np.unwrap(np.radians(route[:,3]))))
    actor_forward = np.array([math.cos(yaw),math.sin(yaw)])
    actor_right = np.array([-math.sin(yaw),math.cos(yaw)])
    centre = np.array([np.interp(station,stations,route[:,i]) for i in (0,1)])
    half, offset, box_yaw = _body_parameters('ego',measured_actors)
    centre += offset[0]*actor_forward+offset[1]*actor_right
    yaw += math.radians(box_yaw)
    forward = np.array([math.cos(yaw),math.sin(yaw)])
    right = np.array([-math.sin(yaw),math.cos(yaw)])
    half = np.asarray(half)+clearance_m
    blocked = []
    for role in roles:
        if role in vehicles and role != 'ego':
            actor = vehicles[role]; other_route = actor['route']; params = _body_parameters(role,measured_actors)
        elif role in walkers:
            actor = walkers[role]; other_route = actor['route_points']
            params = _body_parameters(role if measured_actors is not None else 'pedestrian:'+role,measured_actors)
        else:
            raise ValueError('Unknown or ego waiting target: '+role)
        sweep = route_body_samples(other_route,params[0],bbox_offset=params[1],bbox_yaw_deg=params[2])
        delta = sweep['centre']-centre
        overlap = np.ones(len(delta),dtype=bool)
        for axis, extent in ((forward,half[0]),(right,half[1])):
            radius = extent+sweep['half'][:,0]*abs(sweep['forward']@axis)+sweep['half'][:,1]*abs(sweep['right']@axis)
            overlap &= abs(delta@axis) <= radius
        for axes, extent in ((sweep['forward'],sweep['half'][:,0]),(sweep['right'],sweep['half'][:,1])):
            radius = extent+half[0]*abs(axes@forward)+half[1]*abs(axes@right)
            overlap &= abs(np.einsum('ij,ij->i',delta,axes)) <= radius
        if np.any(overlap):
            blocked.append(role)
    return {'clear':not blocked,'blocked_target_roles':blocked,'stop_route_s':float(station),
            'clearance_m':float(clearance_m),'target_roles':list(roles),
            'method':'entire_parked_ego_OBB_vs_continuous_participant_route_swept_OBB_v1',
            'body_source':'measured_t0_bbox' if measured_actors is not None else 'conservative_planning_envelope_requires_native_recheck',
            'sample_step_m':.5}


def safe_waiting_station(cfg, requested_station, roles):
    """Move only the planned waiting point upstream; never move any NPC."""
    for shift in np.arange(0., max(0.,requested_station-2.)+.251, .25):
        station = requested_station-shift
        if station < 2.:
            break
        result = waiting_position_check(cfg,station,roles)
        if result['clear']:
            result['requested_stop_route_s'] = float(requested_station)
            result['upstream_adjustment_m'] = float(shift)
            return result
    raise ValueError('No upstream waiting pose keeps the complete ego body clear of target routes')


def route_signal(cmap, route, geometry):
    """Resolve a signal that controls this actual connector, not a nearby pole."""
    values = np.asarray(route, float)
    station = geometry.cumulative(values)
    approach = cmap.get_waypoint(carla.Location(*values[40, :3]))
    matches = []
    for landmark in approach.get_landmarks_of_type(65., '1000001', False):
        wp = landmark.waypoint
        if wp is None:
            continue
        validities = [list(v) for v in landmark.get_lane_validities()]
        if not any(min(a, b) <= wp.lane_id <= max(a, b) for a, b in validities):
            continue
        xyz = np.asarray(geometry.row(wp)[:3])
        index = int(np.argmin(np.linalg.norm(values[:, :3] - xyz, axis=1)))
        distance = float(np.linalg.norm(values[index, :3] - xyz))
        # Signals returned across another fork do not control this route.
        if not 65 <= index <= 80 or distance > 1.2:
            continue
        matches.append((distance, str(landmark.id), {
            'landmark_id': str(landmark.id), 'landmark_type': landmark.type,
            'road_id': int(wp.road_id), 'lane_id': int(wp.lane_id),
            'opendrive_s_m': float(wp.s), 'lane_validities': validities,
            'signal_route_s_m': float(station[index]),
            # OpenDRIVE CARLA signals lie at/just inside the connector. Keep
            # the vehicle front two metres before that mapped boundary.
            'stop_route_s_m': float(min(station[index] - 2., station[70] - 1.)),
            'route_signal_distance_m': distance,
            'binding': 'runtime TrafficLight.get_opendrive_id equals landmark_id',
        }))
    if not matches:
        raise ValueError('No lane-valid signal landmark on the actual approach connector')
    return min(matches, key=lambda item: item[:2])[2]


def signal_candidate(cmap, candidate, geometry):
    result = copy.deepcopy(candidate)
    ego = route_signal(cmap, result['ego_route'], geometry)
    npc = route_signal(cmap, result['npc_route'], geometry)
    if ego['landmark_id'] == npc['landmark_id']:
        raise ValueError('Independent signal approaches must have distinct signal asset IDs')
    result['signal_approaches'] = {'ego': ego, 'npc': npc}
    result['evidence']['signals'] = {
        'mode': 'frozen programmed native signal actors',
        'approaches': copy.deepcopy(result['signal_approaches']),
        'phase_clock': 'seconds relative to common intervention t0',
    }
    return result


def configure_signals(cfg, candidate, template):
    routes = {'ego': np.asarray(candidate['ego_route']), 'npc': np.asarray(candidate['npc_route'])}
    groups = []
    for group, route in routes.items():
        roles = [a['role'] for a in cfg['vehicles'] if np.array_equal(a['route'], route)]
        approach = candidate['signal_approaches'][group]
        groups.append({'id': group, 'landmark_ids': [approach['landmark_id']], 'roles': roles,
                       'stop_route_s': {r: approach['stop_route_s_m'] for r in roles},
                       'map_evidence': copy.deepcopy(approach)})
    if template == 'left_signalized':
        phases = [(-3., 'Green', 'Green'), (2.5, 'Yellow', 'Yellow'), (3.5, 'Red', 'Red')]
        policy = 'permissive opposing green: ego turns left across the opposing through movement'
    else:
        green = float(cfg['geometry']['traffic_design']['signal_encounter_plan']['npc_green_time_s'])
        phases = [(-100., 'Green', 'Red'), (green - 3.5, 'Yellow', 'Red'),
                  (green - .5, 'Red', 'Red'), (green, 'Red', 'Green')]
        policy = ('initial-clearance encounter: ego starts committed past its stop line; '
                  'queued cross traffic receives green two physics steps before t0 after '
                  '3s yellow and 0.5s all-red; '
                  'the preceding green is a declared schedule, not observed entry evidence')
    cfg['traffic_signal_program'] = {
        'clock': 'relative_to_t0_s', 'groups': groups,
        'phases': [{'start_time_s': t, 'states': {'ego': ego, 'npc': npc}}
                   for t, ego, npc in phases],
        'policy': policy, 'unmatched_signals': 'Off',
        'npc_control': 'live front-to-stopline distance; committed vehicles clear junction',
        'ego_control': 'requested action is not overridden; signal violations are observable outcomes',
    }
    cfg['traffic_lights'] = 'native signal actors bound by OpenDRIVE ID, frozen and set from traffic_signal_program'


def signal_crossing_plan(candidate, speed_scale, style):
    """Fit the current action/opportunity contracts before native qualification.

    The 2.8m half-length is the existing conservative spawn envelope. The
    2.3m front lower bound covers the four native_v2 blueprint bodies, including
    their local offsets (smallest observed forward bound: Mercedes 2.334m).
    These are planning assumptions, not substituted measured-t0 evidence.
    """
    if __package__:
        from .action_opportunity import MIN_OBSERVATION_S
        from .runtime.action_separation import CONTRACT
        from .runtime.ego_decisions import DEFAULT_STOP_RESPONSE_TIME_S, DEFAULT_STOP_MARGIN_M
        from .measured_t0 import travel_bound, arrival_bound
    else:
        from action_opportunity import MIN_OBSERVATION_S
        from runtime.action_separation import CONTRACT
        from runtime.ego_decisions import DEFAULT_STOP_RESPONSE_TIME_S, DEFAULT_STOP_MARGIN_M
        from measured_t0 import travel_bound, arrival_bound
    body_half, front_lower = 2.8, 2.3
    window = float(CONTRACT['future_s'])
    latest_entry = window - MIN_OBSERVATION_S
    pre_t0 = float(CONTRACT['history_s']) + 1.  # unchanged driving warmup
    deceleration = 4.5  # the shared _place_stop_approach / yield_wait design
    dt = float(CONTRACT['physics_dt_s'])
    # Advance this fixed signal programme by two physics intervals, while
    # retaining its yellow/all-red durations. This adds two intervals of
    # prospective observation reserve; it never changes an NPC's live policy.
    observation_reserve = 2. * dt
    green_time = -observation_reserve
    planned_latest_entry = latest_entry - observation_reserve
    own_line = float(candidate['signal_approaches']['ego']['stop_route_s_m'])
    other_line = float(candidate['signal_approaches']['npc']['stop_route_s_m'])
    # Commit the ego *front*, rather than requiring its centre to be a metre
    # past the line. Retain the existing control stop margin beyond the line.
    own_spawn = own_line - front_lower + DEFAULT_STOP_MARGIN_M
    own_entry = float(candidate['ego_conflict_s']) - 4.
    waiting = safe_waiting_station({'vehicles': [
        {'role': 'ego', 'route': candidate['ego_route']},
        {'role': 'interaction_vehicle', 'route': candidate['npc_route']}]},
        own_entry - body_half - 1.5, ['interaction_vehicle'])
    stop_center = waiting['stop_route_s']
    # The full conservative parked ego body must clear the opposing route.
    # Include the same braking, controller response and stop margin as action
    # design; _place_stop_approach must not silently undo this committed pose.
    speeds = []
    contract_minimum_speed = float(CONTRACT['minimum_measured_t0_speed_m_s'])
    # Reserve one interval at the declared design braking rate above the
    # measured speed gate, rather than selecting precisely on its boundary.
    minimum_speed = contract_minimum_speed + deceleration * dt
    for speed in np.arange(contract_minimum_speed, 3.5 * speed_scale + 1e-8, .05):
        speed = round(float(speed), 6)
        if speed < minimum_speed - 1e-8:
            continue
        t0 = own_spawn + pre_t0 * speed
        stopping = (speed * speed / (2. * deceleration)
                    + speed * DEFAULT_STOP_RESPONSE_TIME_S + DEFAULT_STOP_MARGIN_M
                    + speed * dt)
        if (t0 + stopping <= stop_center + 1e-8 and
                0. < (own_entry - t0) / speed <= window):
            speeds.append(speed)
    if not speeds:
        raise ValueError('Signal crossing has no committed-ego speed at or above the distinct-action minimum with full stopping/entry opportunity')
    desired_before_style = 6.2 * speed_scale
    desired = desired_before_style * style['desired_speed_multiplier']
    acceleration = float(style['max_acceleration'])
    # Queue the conservative front at the mapped stop line. IDM minimum_gap
    # remains a live following-policy parameter; it is not an extra setback
    # from this already mapped, unobstructed signal boundary.
    npc_spawn = other_line - body_half
    other_entry = float(candidate['npc_conflict_s']) - 4.
    required = other_entry - npc_spawn
    front_required = required - front_lower
    admission_duration = planned_latest_entry - green_time
    travel = travel_bound(0., acceleration, admission_duration, desired)
    if not 0. < front_required <= travel + 1e-8:
        raise ValueError(f'Signal crossing queued NPC cannot reach body conflict entry with one second remaining ({front_required:.2f}>{travel:.2f} m in {latest_entry:g}s)')
    if min(own_spawn, npc_spawn) < 5.:
        raise ValueError('Signal crossing has insufficient actual mapped route before the committed/queued poses')
    entry_after_green = arrival_bound(front_required, 0., acceleration, desired)
    entry_time = entry_after_green + green_time
    entry_tick = math.ceil(entry_time / dt - 1e-8) * dt
    return {'policy_id': 'signal-crossing-opportunity-v2',
            'ego_spawn_s': own_spawn, 'ego_initial_speed_candidates_m_s': speeds,
            'ego_minimum_design_speed_m_s': minimum_speed,
            'ego_minimum_measured_t0_speed_m_s': contract_minimum_speed,
            'ego_speed_reserve_m_s': deceleration * dt,
            'ego_position_reserve_s': dt,
            'ego_committed_front_margin_m': DEFAULT_STOP_MARGIN_M,
            'ego_waiting_geometry': waiting,
            'ego_design_deceleration_m_s2': deceleration,
            'ego_stop_response_time_s': DEFAULT_STOP_RESPONSE_TIME_S,
            'ego_control_stop_margin_m': DEFAULT_STOP_MARGIN_M,
            'npc_spawn_s': npc_spawn, 'npc_initial_speed_m_s': 0.,
            'npc_desired_speed_before_style_m_s': desired_before_style,
            'npc_distance_to_conflict_entry_m': required,
            'npc_front_to_conflict_entry_bound_m': front_required,
            'npc_optimistic_body_entry_s': entry_time,
            'npc_optimistic_body_entry_after_green_s': entry_after_green,
            'npc_optimistic_body_entry_tick_s': entry_tick,
            'npc_ideal_3s_acceleration_distance_m': travel,
            'npc_ideal_4s_acceleration_distance_m': travel_bound(0., acceleration, window, desired),
            'minimum_response_observation_s': MIN_OBSERVATION_S,
            'response_observation_reserve_s': observation_reserve,
            'latest_planned_target_entry_s': planned_latest_entry,
            'latest_target_entry_s': latest_entry, 'physics_dt_s': dt,
            'conservative_body_half_length_m': body_half,
            'native_v2_front_lower_bound_m': front_lower,
            'npc_green_time_s': green_time, 'ego_initially_committed_past_stop_line': True,
            'initial_entry_legality_observed': False,
            'basis': 'initial mapped routes and conservative body bounds, fixed signal schedule with a recorded two-step advance, full response/braking reserve and declared policy envelope; measured t0 and actual motion require native qualification'}


def _route_at(wp, geometry, before=130, after=150):
    past, future = geometry.extend(wp, before, True), geometry.extend(wp, after)
    if past is None or future is None:
        raise ValueError('Insufficient actual mapped route around candidate')
    points = list(reversed(past)) + [wp] + future
    rows = np.asarray([geometry.row(w) for w in points])
    distances = np.linalg.norm(np.diff(rows[:, :3], axis=0), axis=1)
    if min(distances) < 1e-4 or max(distances) > 2.1:
        raise ValueError('Discontinuous or degenerate mapped route')
    return rows.tolist(), points, float(geometry.cumulative(rows)[before])


def _sidewalk_span(cmap, wp, geometry):
    """Select actual sidewalk centres beyond both sides of one road carriageway."""
    origin = np.asarray(geometry.row(wp)[:3])
    yaw = math.radians(wp.transform.rotation.yaw)
    normal = np.array([-math.sin(yaw), math.cos(yaw), 0.])
    samples = []
    for offset in np.arange(-28., 28.01, .5):
        point = origin + normal * offset
        near = cmap.get_waypoint(carla.Location(*point), project_to_road=False,
                                lane_type=carla.LaneType.Driving | carla.LaneType.Sidewalk)
        if near is not None and near.road_id == wp.road_id and abs(near.s - wp.s) < 1.:
            samples.append((float(offset), near))
    driving = [o for o, w in samples if w.lane_type == carla.LaneType.Driving]
    if not driving:
        raise ValueError('No mapped carriageway under pedestrian crossing')
    ends = []
    for side in (-1, 1):
        limit = min(driving) if side < 0 else max(driving)
        choices = [(abs(o), w) for o, w in samples if w.lane_type == carla.LaneType.Sidewalk
                   and side * (o - limit) > .1]
        if not choices:
            raise ValueError('No actual outside sidewalk on both crossing sides')
        walk = min(choices, key=lambda item: item[0])[1]
        xyz = np.asarray(geometry.row(walk)[:3])
        if abs(xyz[2] - origin[2]) > .6:
            raise ValueError('Pedestrian crossing has incompatible sidewalk elevation')
        ends.append({'position': xyz.tolist(), 'road_id': int(walk.road_id),
                     'lane_id': int(walk.lane_id), 'opendrive_s_m': float(walk.s),
                     'lane_type': 'Sidewalk', 'width_m': float(walk.lane_width)})
    # Enter from the side nearest ego so the four-second observation can contain
    # an encounter without increasing walker speed or moving the pavement.
    ends.sort(key=lambda item: np.linalg.norm(np.asarray(item['position']) - origin))
    if np.linalg.norm(np.asarray(ends[0]['position'])[:2] - origin[:2]) > 5.:
        raise ValueError('Ego lane is too far from an outside sidewalk for this short crossing window')
    if np.linalg.norm(np.asarray(ends[0]['position'])[:2] - np.asarray(ends[1]['position'])[:2]) > 25.:
        raise ValueError('Crossing exceeds the declared 25-metre carriageway span')
    return ends


def pedestrian_candidates(cmap, map_name, template, geometry):
    result, excluded, used = [], [], set()
    for wp in cmap.generate_waypoints(12.):
        if wp.is_junction:
            continue
        key = (int(wp.road_id), int(wp.lane_id), int(wp.s // 25))
        if key in used:
            continue
        used.add(key)
        near_future = geometry.extend(wp, 25)
        near_past = geometry.extend(wp, 20, True)
        if near_future is None or near_past is None:
            continue
        junction_distance = next((i + 1 for i, w in enumerate(near_future) if w.is_junction), None)
        if template == 'pedestrian_straight':
            if junction_distance is not None or any(w.is_junction for w in near_past):
                continue
            if max(geometry.angle(w.transform.rotation.yaw, wp.transform.rotation.yaw)
                   for w in near_past + near_future) > 12.:
                continue
        elif junction_distance is None or not 5 <= junction_distance <= 16:
            continue
        try:
            ends = _sidewalk_span(cmap, wp, geometry)
            route, points, station = _route_at(wp, geometry)
            evidence = {'type': 'sidewalk_to_sidewalk_crossing_of_actual_driving_lanes',
                        'sidewalk_endpoints': ends, 'crossing_road_id': int(wp.road_id),
                        'ego_lane_id': int(wp.lane_id), 'junction_distance_ahead_m': junction_distance,
                        'signals': 'frozen Off; controlled unsignalized pedestrian interaction',
                        'painted_crosswalk_claimed': False}
            result.append({'map': map_name, 'route': route, 'ego_route': route,
                           'origin': geometry.row(wp)[:3], 'ego_conflict_s': station,
                           'anchor': {'road_id': int(wp.road_id), 'lane_id': int(wp.lane_id),
                                      'opendrive_s_m': float(wp.s)}, 'evidence': evidence,
                           'pedestrian_endpoints': ends})
        except ValueError as exc:
            excluded.append({'map': map_name, 'template': template, 'anchor': list(key), 'reason': str(exc)})
    return result, excluded


def narrow_candidates(cmap, map_name, geometry):
    result, excluded, used = [], [], set()
    for wp in cmap.generate_waypoints(20.):
        if wp.is_junction or wp.lane_id != -1:
            continue
        other = wp.get_left_lane()
        if (other is None or other.lane_type != carla.LaneType.Driving or other.lane_id != 1
                or other.road_id != wp.road_id or geometry.angle(wp.transform.rotation.yaw, other.transform.rotation.yaw) < 170):
            continue
        if wp.lane_width + other.lane_width > 6.40001:
            continue
        key = (int(wp.road_id), int(wp.s // 40))
        if key in used:
            continue
        used.add(key)
        try:
            own_route, own_points, own_s = _route_at(wp, geometry)
            other_route, other_points, other_s = _route_at(other, geometry)
            local = own_points[118:143]
            evidence_widths = []
            for w in local:
                opposite = w.get_left_lane()
                outside = w.get_right_lane()
                if (w.is_junction or w.road_id != wp.road_id or opposite is None
                        or opposite.lane_type != carla.LaneType.Driving or opposite.lane_id != 1
                        or opposite.road_id != wp.road_id or opposite.is_junction
                        or (outside is not None and outside.lane_type == carla.LaneType.Driving)
                        or (opposite.get_right_lane() is not None
                            and opposite.get_right_lane().lane_type == carla.LaneType.Driving)):
                    raise ValueError('Meeting segment must be native one-lane-per-direction road without extra driving lanes')
                width = float(w.lane_width + opposite.lane_width)
                if width > 6.40001:
                    raise ValueError('Mapped road is wider than the predeclared 6.4-metre narrow-road threshold')
                evidence_widths.append(width)
            evidence = {'type': 'native_narrow_two_way_road', 'road_id': int(wp.road_id),
                        'ego_lane_id': -1, 'opposing_lane_id': 1,
                        'driving_lanes_per_direction': 1, 'total_width_max_m': max(evidence_widths),
                        'total_width_min_m': min(evidence_widths), 'narrow_width_threshold_m': 6.4,
                        'sample_count': len(evidence_widths), 'single_lane_bottleneck_claimed': False,
                        'priority_policy': 'conservative yielding on declared reduced-clearance road interval; no physical obstruction is invented'}
            result.append({'map': map_name, 'ego_route': own_route, 'npc_route': other_route,
                           'ego_conflict_s': own_s, 'npc_conflict_s': other_s,
                           'origin': geometry.row(wp)[:3], 'evidence': evidence,
                           'anchor': {'road_id': int(wp.road_id), 'lane_id': -1, 'opendrive_s_m': float(wp.s)}})
        except ValueError as exc:
            excluded.append({'map': map_name, 'template': 'narrow_meeting', 'anchor': list(key), 'reason': str(exc)})
    return result, excluded


def _style_vehicles(cars, rng, geometry):
    for actor in cars[1:]:
        actor['blueprint_id'] = geometry.DIVERSE_NPC_MODELS[int(rng.integers(len(geometry.DIVERSE_NPC_MODELS)))]
        actor['blueprint_attributes']['color'] = geometry.DIVERSE_COLORS[int(rng.integers(len(geometry.DIVERSE_COLORS)))]
    if any(geometry.spawn_footprints_overlap(a, b) for a, b in itertools.combinations(cars, 2)):
        raise ValueError('Initial conservative vehicle footprints overlap')


def pedestrian_config(candidate, template, index, seed, rng, speed_scale, geometry, randomization=None):
    route = candidate['ego_route']
    speed = float(rng.uniform(4.5, 5.5)) * speed_scale
    if randomization:
        speed = randomization['parameters']['ego_initial_speed_m_s']*speed_scale
    station = candidate['ego_conflict_s'] - speed * 5.4
    # Leave a gap behind ego so a moving follower does not conceal ego's
    # yielding cue throughout the fixed four-second response window.
    offsets = [0., -28., -52., -76., 24., 46., 68.]
    if randomization:
        from scene_randomization import stream
        count = randomization['participants']['vehicle_count_including_ego']
        # Select a subset before placing: no seven-actor feasibility prerequisite.
        offsets = [0.]+list(stream(randomization['seed'],'participant/pedestrian_traffic').choice(offsets[1:],count-1,replace=False))
    cars = [geometry.vehicle('ego' if i == 0 else f'traffic_{i:02d}', route,
                            station + gap, speed, speed + (.3 if i else 0.))
            for i, gap in enumerate(offsets)]
    _style_vehicles(cars, rng, geometry)
    geom = {'map': candidate['map'], 'type': template, 'origin': candidate['origin'], 'conflicts': [],
            'selection': {'anchor': candidate['anchor'], 'outcome_based_selection': False},
            'interaction_evidence': candidate['evidence'], 'ego_conflict_s': candidate['ego_conflict_s'],
            'traffic_design': {'vehicle_count_including_ego': len(cars), 'pedestrian_count': 1,
                               'reaction_guaranteed': False, 'future_outcome_filtering': False}}
    cfg = geometry.finish_config(template, index, seed, cars, geom, 2., 1.)
    from walker_spawn import crossing_path, blueprint_for
    direction = 0
    points, transform, curb_offset = crossing_path(candidate['pedestrian_endpoints'], direction)
    fallback_blueprint = WALKER_BLUEPRINT_CANDIDATES[int(rng.integers(len(WALKER_BLUEPRINT_CANDIDATES)))]
    blueprint = blueprint_for(randomization['seed']) if randomization else fallback_blueprint
    cfg['pedestrians'] = [{'role': 'crossing_pedestrian',
                          'blueprint_id': blueprint, 'transform': transform,
                          'route_points': points, 'speed_mps': float(rng.uniform(1.25, 1.55)),
                          'start_time_s': 0.,
                          'start_time_semantics': 'earliest observation-based crossing decision, not scheduled departure',
                          'crossing_direction_index': direction, 'geometry_evidence': candidate['evidence'],
                          'initial_opportunity_design': {'sidewalk_start': 'nearest_to_ego_corridor',
                              'toward_curb_offset_m': curb_offset, 'sidewalk_edge_reserve_m': .35,
                              'future_window_s': 4., 'reaction_guaranteed': False},
                          'blueprint_evidence': {'source':'https://carla.readthedocs.io/en/0.9.15/catalogue_pedestrians/',
                              'runtime_availability_verified':False,'runtime_blueprint_and_bbox_check_required':True},
                          'reactive_to_ego': True,
                          'policy': 'reactive_crossing_v2; shared current-state feedback throughout crossing, no branch or future trajectory access'}]
    return cfg, {}


def narrow_config(candidate, index, seed, rng, speed_scale, geometry, randomization=None):
    speed = float(rng.uniform(4.2, 5.0)) * speed_scale
    if randomization:
        speed = randomization['parameters']['ego_initial_speed_m_s']*speed_scale
    cars = []
    streams = [('ego', [0., -17., -34., -51.]), ('npc', [0., -17., -34.])]
    if randomization:
        streams = [(side,[-17.*i for i in range(1+randomization['participants'][key])]) for side,key in
                   [('ego','narrow_ego_followers'),('npc','narrow_oncoming_followers')]]
    for side, offsets in streams:
        route = candidate[side + '_route']
        stream_speed = randomization['parameters']['other_initial_speed_m_s']*speed_scale if randomization and side=='npc' else speed
        station = candidate[side + '_conflict_s'] - stream_speed * 5.8
        if randomization and len(offsets)>1:
            shift=randomization['parameters']['narrow_oncoming_shift_m'] if side=='npc' else 0.
            gap=min(randomization['parameters']['queue_gap_m'],(station+min(0.,shift)-5.)/(len(offsets)-1))
            if gap<11.:
                raise ValueError('Requested narrow queue does not fit the mapped approach')
            offsets=[-i*gap for i in range(len(offsets))]
        for i, offset in enumerate(offsets):
            role = ('ego' if i == 0 else f'ego_follower_{i}') if side == 'ego' else f'oncoming_{i}'
            initial = stream_speed
            if randomization and i:
                from scene_randomization import stream
                initial=max(.5,stream_speed+float(stream(randomization['seed'],f'actor/{role}/initial_relative').uniform(-.45,.45)))
            cars.append(geometry.vehicle(role, route, station + offset, initial, initial + (.2 if role != 'ego' else 0.)))
    _style_vehicles(cars, rng, geometry)
    ego_roles = [a['role'] for a in cars if a['role'] == 'ego' or a['role'].startswith('ego_follower')]
    other_roles = [a['role'] for a in cars if a['role'].startswith('oncoming')]
    # Sample right of way once per scene, never from a branch or its outcome.
    ego_priority = bool(rng.integers(2))
    primary, yielding = (ego_roles, other_roles) if ego_priority else (other_roles, ego_roles)
    intervals = {role: [candidate[side + '_conflict_s'] - 10., candidate[side + '_conflict_s'] + 10.]
                 for side, roles in [('ego', ego_roles), ('npc', other_roles)] for role in roles}
    geom = {'map': candidate['map'], 'type': 'narrow_meeting', 'origin': candidate['origin'], 'conflicts': [],
            'selection': {'anchor': candidate['anchor'], 'outcome_based_selection': False},
            'interaction_evidence': candidate['evidence'],
            'ego_conflict_s': candidate['ego_conflict_s'],
            'traffic_design': {'vehicle_count_including_ego': len(cars), 'reaction_guaranteed': False,
                               'future_outcome_filtering': False},
            'narrow_meeting': {'yield_roles': yielding, 'priority_roles': primary,
                              'priority_assignment': 'ego_stream' if ego_priority else 'oncoming_stream',
                              'priority_fixed_across_branches': True,
                              'yield_route_s': {r: intervals[r] for r in yielding},
                              'priority_route_s': {r: intervals[r] for r in primary},
                              'stop_margin_m': 2., 'clearance_time_s': 3.,
                              'policy': 'conservative priority rule for measured native reduced-clearance interval'}}
    return geometry.finish_config('narrow_meeting', index, seed, cars, geom, 2., 1.), {}
