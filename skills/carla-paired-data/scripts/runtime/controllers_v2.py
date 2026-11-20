"""Closed-loop physical driving policies for CARLA paired data.

NPC logic receives observed state plus fixed routes/right-of-way. It never
receives an ego branch ID, desired future trajectory, or simulation outcome.
"""
from __future__ import annotations

import math
import copy
import numpy as np

if __package__:
    from . import npc_merge
    from .auxiliary_actors import validate_auxiliary_config
    from .ego_decisions import EgoDecisionState, validate_profiles as validate_decision_profiles, longitudinal_target, stop_target, DEFAULT_STOP_RESPONSE_TIME_S
else:
    import npc_merge
    from auxiliary_actors import validate_auxiliary_config
    from ego_decisions import EgoDecisionState, validate_profiles as validate_decision_profiles, longitudinal_target, stop_target, DEFAULT_STOP_RESPONSE_TIME_S


LEGACY_BRANCHES = ('maintain', 'brake', 'accelerate', 'left', 'right', 'maintain_repeat')


def _lane_directions(values):
    if (not isinstance(values, (list, tuple, set, frozenset))
            or any(not isinstance(value, str) or value not in ('left', 'right') for value in values)
            or len(set(values)) != len(values)):
        raise ValueError('available_lane_change_directions must contain distinct left/right directions')
    return tuple(side for side in ('left', 'right') if side in values)


def _profile_names(directions):
    if len(directions) == 2:
        return LEGACY_BRANCHES[:-1]
    if directions:
        return ('maintain', 'brake_light', 'brake', 'accelerate', directions[0])
    return ('maintain', 'brake_light', 'brake', 'accelerate_light', 'accelerate')


def _finite_action_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _validate_action_profiles(settings):
    if settings.get('profile_version') == 2:
        return validate_decision_profiles(settings)
    if 'available_lane_change_directions' not in settings:
        raise ValueError('Explicit action profiles require available_lane_change_directions')
    directions = _lane_directions(settings['available_lane_change_directions'])
    names = _profile_names(directions)
    profiles = settings['profiles']
    if not isinstance(profiles, dict) or set(profiles) != set(names):
        raise ValueError(f'Explicit profiles require exactly five feasible actions: {names}')
    signatures = set()
    for name in names:
        profile = profiles[name]
        if not isinstance(profile, dict):
            raise ValueError(f'{name}: action profile must be an object')
        kind = 'maintain' if name == 'maintain' else 'lane_change' if name in directions else 'longitudinal'
        if profile.get('kind') != kind:
            raise ValueError(f'{name}: expected action kind {kind}')
        acceleration = _finite_action_number(profile.get('acceleration_m_s2'), f'{name}.acceleration_m_s2')
        cap = _finite_action_number(profile.get('speed_increase_cap_m_s'), f'{name}.speed_increase_cap_m_s')
        if cap < 0.:
            raise ValueError(f'{name}: speed increase cap must be nonnegative')
        if kind == 'longitudinal':
            braking = name.startswith('brake')
            if (braking and (acceleration >= 0. or cap != 0.)) or (not braking and (acceleration <= 0. or cap <= 0.)):
                raise ValueError(f'{name}: signed acceleration/cap must produce the declared longitudinal action')
        elif acceleration != 0. or cap != 0.:
            raise ValueError(f'{name}: maintain/lane-change profiles preserve the initial target speed')
        direction = profile.get('direction')
        if (kind == 'lane_change' and direction != name) or (kind != 'lane_change' and direction is not None):
            raise ValueError(f'{name}: direction must name its available target lane only for lane-change actions')
        signatures.add((kind, acceleration, cap, direction))
    if len(signatures) != 5:
        raise ValueError('The five action profiles must represent distinct physical conditions')
    for light, strong in (('brake_light', 'brake'), ('accelerate_light', 'accelerate')):
        if light in profiles and abs(profiles[light]['acceleration_m_s2']) >= abs(profiles[strong]['acceleration_m_s2']):
            raise ValueError(f'{light}: acceleration magnitude must be lower than {strong}')
    return names


def select_action_profiles(available_directions, *, braking_deceleration_m_s2=4.5,
                           acceleration_m_s2=1.5, acceleration_speed_increase_cap_m_s=5.):
    """Return five explicit profiles for the actually available adjacent lanes.

    Store this dict in config['action']['profiles'], declare the same directions
    in action['available_lane_change_directions'], then use branch_names(config)
    for config['branches']. Repeat is an alias, not a sixth independent action.
    Strong longitudinal settings can be supplied from the existing action config.
    """
    directions = _lane_directions(available_directions)
    brake = _finite_action_number(braking_deceleration_m_s2, 'braking_deceleration_m_s2')
    acceleration = _finite_action_number(acceleration_m_s2, 'acceleration_m_s2')
    cap = _finite_action_number(acceleration_speed_increase_cap_m_s, 'acceleration_speed_increase_cap_m_s')
    if min(brake, acceleration, cap) <= 0.:
        raise ValueError('Strong braking, acceleration, and speed increase cap must be positive')
    profiles = {}
    for name in _profile_names(directions):
        if name in directions:
            profile = {'kind': 'lane_change', 'acceleration_m_s2': 0.,
                       'speed_increase_cap_m_s': 0., 'direction': name}
        elif name == 'maintain':
            profile = {'kind': 'maintain', 'acceleration_m_s2': 0., 'speed_increase_cap_m_s': 0.}
        else:
            value = {'brake_light': -1.2, 'brake': -brake,
                     'accelerate_light': .6, 'accelerate': acceleration}[name]
            profile = {'kind': 'longitudinal', 'acceleration_m_s2': value,
                       'speed_increase_cap_m_s': cap if value > 0. else 0.}
        profiles[name] = profile
    _validate_action_profiles({'available_lane_change_directions': directions, 'profiles': profiles})
    return profiles


def branch_names(config, include_repeat=True):
    """Validated scene-specific branches, or the frozen legacy six by default.

    The prescribed order does not depend on JSON object-key ordering. Explicit
    config['branches'], when supplied with profiles, must include exactly the
    five conditions and the maintain_repeat alias in this order.
    """
    settings = config.get('action', {})
    if 'profiles' not in settings:
        return LEGACY_BRANCHES if include_repeat else LEGACY_BRANCHES[:-1]
    names = _validate_action_profiles(settings)
    if any(settings['profiles'][name]['kind'] == 'lane_change' for name in names):
        if settings.get('lateral_mode') != 'adjacent_lane_route':
            raise ValueError('Explicit lane-change profiles require actual adjacent-lane routes')
    validate_lane_change_routes(config['vehicles'][0]['route'], settings)
    complete = names + ('maintain_repeat',)
    declared = config.get('branches')
    if declared is not None and (not isinstance(declared, (list, tuple)) or tuple(declared) != complete):
        raise ValueError(f'config.branches must contain five distinct actions plus maintain_repeat: {complete}')
    return complete if include_repeat else names


def _profile_for_branch(settings, branch):
    name = 'maintain' if branch == 'maintain_repeat' else branch
    if name not in settings['profiles']:
        raise ValueError(f'Unknown action branch for this scene: {branch}')
    return settings['profiles'][name]


def wrap(angle):
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi


def idm(speed, desired, gap=math.inf, leader_speed=0., max_acceleration=2., comfortable_brake=2.8, time_headway=1.25, minimum_gap=2.5):
    speed, desired = max(float(speed), 0.), max(float(desired), .1)
    interaction = 0.
    if math.isfinite(gap):
        wanted = minimum_gap + max(0., speed * time_headway + speed * (speed-leader_speed) /
                                   (2 * math.sqrt(max_acceleration * comfortable_brake)))
        interaction = (wanted / max(gap, .15)) ** 2
    return float(np.clip(max_acceleration * (1-(speed/desired)**4-interaction), -8., max_acceleration))


def route_progress(position, route):
    route = np.asarray(route)
    segments = np.diff(route[:, :2], axis=0)
    length = np.linalg.norm(segments, axis=1)
    fraction = np.clip(np.sum((np.asarray(position)[:2]-route[:-1, :2]) * segments, axis=1) /
                       np.maximum(length**2, 1e-8), 0., 1.)
    closest = route[:-1, :2] + fraction[:, None] * segments
    index = int(np.argmin(np.linalg.norm(closest-np.asarray(position)[:2], axis=1)))
    return float(np.r_[0., np.cumsum(length)][index] + fraction[index] * length[index])


def steer_to_route(position, yaw, speed, route, lateral_offset=0.):
    route = np.asarray(route)
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1))]
    progress = route_progress(position, route)
    target_s = min(distances[-1], progress + max(4., .65 * speed))
    index = min(len(route)-1, int(np.searchsorted(distances, target_s)))
    route_yaw = math.radians(route[index, 3])
    target = route[index, :2] + lateral_offset * np.array([-math.sin(route_yaw), math.cos(route_yaw)])
    delta = target - np.asarray(position)[:2]
    alpha = wrap(math.atan2(delta[1], delta[0])-yaw)
    wheel_angle = math.atan2(2*2.85*math.sin(alpha), max(np.linalg.norm(delta), .5))
    return float(np.clip(wheel_angle/.70, -.65, .65))


def live_leader(index, observed, extents, margin):
    yaw = np.radians(observed["rotation_deg"][:, 2])
    heading = np.array([math.cos(yaw[index]), math.sin(yaw[index])])
    right = np.array([-heading[1], heading[0]])
    best = (math.inf, 0., None)
    for j in range(len(extents)):
        if j == index:
            continue
        # Oncoming crossing cars are handled by the junction conflict policy.
        if abs(wrap(yaw[j]-yaw[index])) > math.radians(55):
            continue
        delta = observed["position"][j, :2]-observed["position"][index, :2]
        forward, lateral = float(delta @ heading), abs(float(delta @ right))
        width = extents[j, 0]*abs(math.sin(yaw[j]-yaw[index])) + extents[j, 1]*abs(math.cos(yaw[j]-yaw[index]))
        if forward <= 0 or lateral > extents[index, 1]+width+margin:
            continue
        gap = forward-extents[index, 0]-extents[j, 0]
        if gap < best[0]:
            best = (gap, max(0., float(observed["velocity"][j, :2] @ heading)), j)
    return best


def project_to_route(position, route):
    """Current position -> route arclength, signed offset and local tangent.

    The route is the NPC's predeclared lane centreline, not an ego trajectory.
    Comparing each actor at its own local tangent avoids widening the lane
    corridor merely because the follower and a neighbour straddle a bend.
    """
    route = np.asarray(route, dtype=float)
    segments = np.diff(route[:, :2], axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1e-8
    if not valid.any():
        raise ValueError('Leader projection requires a nonzero route segment')
    fractions = np.clip(np.sum((np.asarray(position)[:2]-route[:-1, :2])*segments, axis=1) /
                        np.maximum(lengths**2, 1e-8), 0., 1.)
    closest = route[:-1, :2] + fractions[:, None]*segments
    distances = np.linalg.norm(closest-np.asarray(position)[:2], axis=1)
    index = int(np.argmin(np.where(valid, distances, np.inf)))
    tangent = segments[index]/lengths[index]
    right = np.array([-tangent[1], tangent[0]])
    offset = float((np.asarray(position)[:2]-closest[index]) @ right)
    progress = float(np.r_[0., np.cumsum(lengths)][index] + fractions[index]*lengths[index])
    return progress, offset, tangent


def route_leader(index, observed, extents, margin, route):
    """Live leaders in this NPC's route corridor; no branch/future input."""
    own_s, own_offset, own_tangent = project_to_route(observed['position'][index], route)
    yaw = np.radians(observed['rotation_deg'][:, 2])
    own_angle = wrap(yaw[index]-math.atan2(own_tangent[1], own_tangent[0]))
    own_width = extents[index, 0]*abs(math.sin(own_angle)) + extents[index, 1]*abs(math.cos(own_angle))
    own_length = extents[index, 0]*abs(math.cos(own_angle)) + extents[index, 1]*abs(math.sin(own_angle))
    best = (math.inf, 0., None)
    for j in range(len(extents)):
        if j == index:
            continue
        other_s, other_offset, tangent = project_to_route(observed['position'][j], route)
        ahead = other_s-own_s
        angle = wrap(yaw[j]-math.atan2(tangent[1], tangent[0]))
        # Oncoming/perpendicular traffic still uses the fixed conflict policy.
        if ahead <= 0 or abs(angle) > math.radians(55):
            continue
        width = extents[j, 0]*abs(math.sin(angle)) + extents[j, 1]*abs(math.cos(angle))
        if abs(other_offset-own_offset) > own_width+width+margin:
            continue
        length = extents[j, 0]*abs(math.cos(angle)) + extents[j, 1]*abs(math.sin(angle))
        gap = ahead-own_length-length
        if gap < best[0]:
            best = (float(gap), max(0., float(observed['velocity'][j, :2] @ tangent)), j)
    return best


def validate_priority_zone(config):
    """Optional map-derived junction area; absent configs keep legacy behavior."""
    zone = config['geometry'].get('priority_zone')
    if zone is None:
        return
    roles = config['vehicle_roles']
    if not isinstance(zone, dict) or zone.get('yield_role') not in roles[1:]:
        raise ValueError('priority_zone requires an existing NPC yield_role')
    priority = zone.get('priority_roles')
    if (not isinstance(priority, list) or not priority or len(set(priority)) != len(priority)
            or zone['yield_role'] in priority or any(role not in roles for role in priority)):
        raise ValueError('priority_zone requires distinct existing priority_roles excluding yield_role')
    yield_route = config['vehicles'][roles.index(zone['yield_role'])]['route']
    for name, route in (('main', zone.get('main_route')), ('yield', yield_route)):
        route = np.asarray(route, dtype=float)
        if route.ndim != 2 or route.shape[1] != 4 or len(route) < 2 or not np.isfinite(route).all():
            raise ValueError(f'priority_zone {name} route requires finite Nx4 geometry')
        lengths = np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)
        interval = np.asarray(zone.get(f'{name}_zone_s' if name == 'main' else 'yield_route_s'), dtype=float)
        if (np.any(lengths <= 1e-8) or interval.shape != (2,) or not np.isfinite(interval).all()
                or not 0. <= interval[0] < interval[1] <= lengths.sum()+1e-6):
            raise ValueError(f'priority_zone {name} interval must lie on its nondegenerate route')
    bounds = np.asarray(zone.get('main_lateral_bounds_m'), dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError('priority_zone requires finite increasing main_lateral_bounds_m')
    for field, allow_zero in (('stop_margin_m', True), ('clearance_time_s', False), ('lateral_lookahead_s', False)):
        value = zone.get(field)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0 or (value == 0 and not allow_zero)):
            raise ValueError(f'priority_zone requires a finite positive {field}')


def box_route_bounds(index, observed, extents, route, extrapolate_s=0.):
    """Project the current yaw-oriented footprint, not merely its centre.

    The online controller receives actual actor extents and pose. As in the
    existing leader controller, actor-local bbox offsets are not an input;
    the planner records a conservative footprint margin for this limitation.
    """
    half = np.asarray(extents[index, :2], dtype=float)
    if not np.isfinite(half).all() or np.any(half <= 0.):
        raise ValueError('priority_zone requires finite positive measured extents')
    yaw = math.radians(float(observed['rotation_deg'][index, 2]))
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    local = np.array([[-1., -1.], [-1., 1.], [1., 1.], [1., -1.]])*half
    corners = (local @ rotation.T + observed['position'][index, :2]
               + extrapolate_s*observed['velocity'][index, :2])
    projected = [project_to_route(corner, route) for corner in corners]
    stations, offsets = [p[0] for p in projected], [p[1] for p in projected]
    return min(stations), max(stations), min(offsets), max(offsets)


def priority_zone_target(index, observed, config, extents, target):
    """Current-state junction admission; a crossing already inside clears it.

    yield_route_s is a VEHICLE-CENTRE interval already expanded by the planner
    for the crossing footprint over all actual main lanes. Do not subtract its
    half length again. main_zone_s is intersected with observed priority boxes.
    Constant-current-velocity windows are a heuristic, not a collision guarantee.
    """
    zone = config['geometry']['priority_zone']
    spec = config['vehicles'][index]
    own_s, _, tangent = project_to_route(observed['position'][index], spec['route'])
    own_speed = max(0., float(observed['velocity'][index, :2] @ tangent))
    enter, leave = map(float, zone['yield_route_s'])
    main_enter, main_leave = map(float, zone['main_zone_s'])
    lateral_lo, lateral_hi = map(float, zone['main_lateral_bounds_m'])
    lateral_horizon = float(zone['lateral_lookahead_s'])
    phase = 'before_entry' if own_s < enter else ('committed_clearance' if own_s < leave else 'cleared')
    # No invented approach velocity for a stationary actor. For a stopped
    # crossing, use the declared starting window, then re-evaluate every tick.
    arrival = max(enter-own_s, 0.)/own_speed if own_speed > .2 else 0.
    clearance = max(float(zone['clearance_time_s']),
                    max(leave-max(own_s, enter), 0.)/own_speed if own_speed > .2 else 0.)
    checks, triggered = [], []
    for role in zone['priority_roles']:
        other = config['vehicle_roles'].index(role)
        lo, hi, lat_lo, lat_hi = box_route_bounds(other, observed, extents, zone['main_route'])
        future_bounds = box_route_bounds(other, observed, extents, zone['main_route'], lateral_horizon)
        _, _, other_tangent = project_to_route(observed['position'][other], zone['main_route'])
        velocity = float(observed['velocity'][other, :2] @ other_tangent)
        lateral_velocity = float(observed['velocity'][other, :2] @ np.array([-other_tangent[1], other_tangent[0]]))
        lateral_overlap = lat_hi >= lateral_lo and lat_lo <= lateral_hi
        future_lateral_overlap = future_bounds[3] >= lateral_lo and future_bounds[2] <= lateral_hi
        swept_lateral_overlap = max(lat_hi, future_bounds[3]) >= lateral_lo and min(lat_lo, future_bounds[2]) <= lateral_hi
        lateral_entry, lateral_exit = None, None
        if lateral_overlap:
            lateral_entry = 0.
            lateral_exit = ((lateral_hi-lat_lo)/lateral_velocity if lateral_velocity > 1e-6
                            else (lat_hi-lateral_lo)/-lateral_velocity if lateral_velocity < -1e-6
                            else math.inf)
        elif lat_hi < lateral_lo and lateral_velocity > 1e-6:
            lateral_entry = (lateral_lo-lat_hi)/lateral_velocity
            lateral_exit = (lateral_hi-lat_lo)/lateral_velocity
        elif lat_lo > lateral_hi and lateral_velocity < -1e-6:
            lateral_entry = (lat_lo-lateral_hi)/-lateral_velocity
            lateral_exit = (lat_hi-lateral_lo)/-lateral_velocity
        lateral_candidate = (lateral_entry is not None and lateral_entry <= lateral_horizon
                             and swept_lateral_overlap)
        occupied = hi >= main_enter and lo <= main_leave
        t_enter = t_leave = None
        if occupied:
            state = 'occupied'
            t_enter = 0.
            t_leave = ((main_leave-lo)/velocity if velocity > .2
                       else (hi-main_enter)/-velocity if velocity < -.2 else math.inf)
        elif hi < main_enter and velocity > .2:
            state = 'approaching'
            t_enter, t_leave = (main_enter-hi)/velocity, (main_leave-lo)/velocity
        elif lo > main_leave and velocity < -.2:
            state = 'approaching_reverse'
            t_enter, t_leave = (lo-main_leave)/-velocity, (hi-main_enter)/-velocity
        else:
            state = 'stopped_outside' if abs(velocity) <= .2 else 'cleared_or_moving_away'
        # A live box currently occupies the physical corridor even when it is
        # stationary. Outside actors require both a current-motion approach
        # and temporal overlap; central/left-lane merge neighbours are excluded.
        blocks = bool(occupied and lateral_overlap)
        if not blocks and lateral_candidate and t_enter is not None:
            blocks = max(t_enter, lateral_entry, arrival) <= min(t_leave, lateral_exit, arrival+clearance)
        if blocks:
            triggered.append(role)
        checks.append({'role': role, 'box_main_s': [float(lo), float(hi)],
                       'box_lateral_m': [float(lat_lo), float(lat_hi)],
                       'extrapolated_box_lateral_m': [float(future_bounds[2]), float(future_bounds[3])],
                       'current_lateral_overlap': bool(lateral_overlap),
                       'extrapolated_lateral_overlap': bool(future_lateral_overlap),
                       'swept_lateral_overlap': bool(swept_lateral_overlap),
                       'lateral_approach_eta_s': lateral_entry,
                       'lateral_speed_m_s': lateral_velocity,
                       'along_route_speed_m_s': velocity, 'state': state,
                       'arrival_s': t_enter, 'exit_s': t_leave if t_leave is None or math.isfinite(t_leave) else None,
                       'blocks_entry': bool(blocks)})
    yielding = phase == 'before_entry' and bool(triggered)
    stop_gap = enter-own_s-float(zone['stop_margin_m'])
    if yielding:
        speed = float(np.linalg.norm(observed['velocity'][index, :2]))
        stop_acceleration = idm(speed, spec['desired_speed_mps'], max(stop_gap, .15), 0.,
                                **config['controller']['idm'])
        target = min(target, max(0., speed+.75*stop_acceleration))
    return target, {'phase': phase, 'yielding': yielding, 'trigger_roles': triggered,
                    'yield_route_s': [enter, leave], 'main_zone_s': [main_enter, main_leave],
                    'main_lateral_bounds_m': [lateral_lo, lateral_hi],
                    'lateral_lookahead_s': lateral_horizon,
                    'own_route_s': float(own_s), 'stop_gap_m': float(stop_gap),
                    'own_entry_eta_s': arrival, 'clearance_window_s': clearance,
                    'priority': checks}


def signal_constraint(index, observed, config, extents):
    """Read a present signal only; a vehicle past its stop line clears it."""
    role = config['vehicle_roles'][index]
    for group in config.get('traffic_signal_program', {}).get('groups', []):
        if role not in group['roles']:
            continue
        if group['id'] not in observed.get('signal_states', {}):
            raise ValueError('Signalized control requires actual signal observation')
        state = observed['signal_states'][group['id']]
        station, _, _ = project_to_route(observed['position'][index], config['vehicles'][index]['route'])
        distance = float(group['stop_route_s'][role]-station-extents[index, 0])
        return {'group_id': group['id'], 'state': state, 'distance_front_to_stop_line_m': distance,
                'must_stop': bool(state in ('Red', 'Yellow') and distance > 0.),
                'committed_past_stop_line': bool(distance <= 0.)}
    return None


def pedestrian_target(index, observed, config, extents, target):
    """Yield to current walkers on/approaching this vehicle's route corridor.

    Constant-velocity short-horizon extrapolation uses measured walker velocity,
    never its future route or start schedule. Stationary sidewalk walkers do not
    become a virtual obstacle merely because they may cross later.
    """
    people = observed.get('pedestrians')
    if people is None:
        if config.get('pedestrians'):
            raise ValueError('Pedestrian scene requires current pedestrian observations')
        return target, []
    spec = config['vehicles'][index]
    station, offset, _ = project_to_route(observed['position'][index], spec['route'])
    speed = float(np.linalg.norm(observed['velocity'][index, :2]))
    horizon = float(config['controller']['yield_lookahead_s'])
    triggers = []
    for j, position in enumerate(people['position']):
        person_s, lateral, tangent = project_to_route(position, spec['route'])
        own_half = float(extents[index, 1])
        radius = float(np.linalg.norm(people['bbox_extent'][j, :2]))
        lane_half = own_half + radius + .35
        along = float(people['velocity'][j, :2] @ tangent)
        across = float(people['velocity'][j, :2] @ np.array([-tangent[1], tangent[0]]))
        rel = lateral-offset
        lateral_interval = None
        if abs(across) < 1e-6:
            if abs(rel) <= lane_half:
                lateral_interval = (0., math.inf)
        else:
            first, last = sorted(((-lane_half-rel)/across, (lane_half-rel)/across))
            if last >= 0:
                lateral_interval = (max(0., first), last)
        gap = person_s-station-extents[index, 0]-radius
        if lateral_interval is None or gap < -2*extents[index, 0] or lateral_interval[0] > horizon:
            continue
        arrival = max(0., gap)/max(speed-along, .2)
        present = lateral_interval[0] == 0. and gap >= 0.
        potential = max(0., arrival-1.) <= lateral_interval[1] and arrival+1. >= lateral_interval[0]
        if present or potential:
            # If presently approaching laterally, stop before its current route
            # station. A margin avoids planning an exact capsule-to-bumper touch.
            acceleration = idm(speed, spec['desired_speed_mps'], max(gap-1., .15),
                               max(0., along), **config['controller']['idm'])
            target = min(target, max(0., speed+.75*acceleration))
            triggers.append(people['roles'][j])
    return target, triggers


def narrow_meeting_target(index, observed, config, extents, target, memory=None, time_s=0.):
    """Current-state priority/yield policy on a measured narrow two-way road."""
    zone = config['geometry'].get('narrow_meeting')
    role = config['vehicle_roles'][index]
    if zone and zone.get('policy_version') == 'narrow-current-occupancy-v2':
        return narrow_live_target(index,observed,config,extents,target,memory,time_s)
    if zone is None or role not in zone['yield_roles']:
        return target, None
    spec = config['vehicles'][index]
    station, _, _ = project_to_route(observed['position'][index], spec['route'])
    start, end = zone['yield_route_s'][role]
    front, rear = station+extents[index, 0], station-extents[index, 0]
    phase = 'before_entry' if front < start else ('committed_clearance' if rear <= end else 'cleared')
    speed = float(np.linalg.norm(observed['velocity'][index, :2]))
    arrival = max(0., start-front)/speed if speed > .2 else 0.
    triggers = []
    for other_role in zone['priority_roles']:
        other = config['vehicle_roles'].index(other_role)
        other_spec = config['vehicles'][other]
        other_s, _, tangent = project_to_route(observed['position'][other], other_spec['route'])
        lo, hi = zone['priority_route_s'][other_role]
        other_speed = float(observed['velocity'][other, :2] @ tangent)
        head, tail = other_s+extents[other, 0], other_s-extents[other, 0]
        occupied = head >= lo and tail <= hi
        moving = other_speed > .2 and head < lo
        eta = (lo-head)/other_speed if moving else math.inf
        if occupied or (moving and eta <= arrival+zone['clearance_time_s']):
            triggers.append(other_role)
    yielding = phase == 'before_entry' and bool(triggers)
    gap = start-front-zone['stop_margin_m']
    if yielding:
        acceleration = idm(speed, spec['desired_speed_mps'], max(gap, .15), 0., **config['controller']['idm'])
        target = min(target, max(0., speed+.75*acceleration))
    return target, {'phase': phase, 'yielding': yielding, 'trigger_roles': triggers,
                    'distance_front_to_zone_m': float(start-front), 'stop_gap_m': float(gap)}


def narrow_live_target(index, observed, config, extents, target, memory, time_s):
    """Measured two-way clearance, occupancy, and causal release confirmation."""
    zone = config['geometry']['narrow_meeting']; role = config['vehicle_roles'][index]
    # Confirmation memory starts with the recorded common history, so its
    # complete evolution is independently replayable from saved observations.
    if time_s < -2.-1e-9:
        memory = None
    intervals = {**zone['priority_route_s'],**zone['yield_route_s']}
    if role not in intervals:
        return target,None
    own = config['vehicles'][index]; start,end = intervals[role]
    station,_,tangent = project_to_route(observed['position'][index],own['route'])
    own_corners = _actual_corners(observed, extents, index)
    own_stations = [project_to_route(corner, own['route'])[0] for corner in own_corners]
    front,rear = max(own_stations),min(own_stations)
    phase = 'before_entry' if front<start else 'committed_clearance' if rear<=end else 'cleared'
    others=[]; triggers=[]; passable=True
    margin=float(zone['lateral_margin_m'])
    for j,other in enumerate(config['vehicles']):
        if j==index or other['role'] not in intervals:
            continue
        other_s,_,other_tangent=project_to_route(observed['position'][j],other['route'])
        if float(tangent@other_tangent)>-.5:
            continue
        lo,hi=intervals[other['role']]
        other_corners = _actual_corners(observed, extents, j)
        other_stations = [project_to_route(corner, other['route'])[0] for corner in other_corners]
        head,tail=max(other_stations),min(other_stations)
        occupied=head>=lo and tail<=hi
        speed=max(0.,float(observed['velocity'][j,:2]@other_tangent))
        eta=(lo-head)/speed if head<lo and speed>.2 else math.inf
        # Compare mapped centreline separation near this actor, and current
        # lateral separation. A displaced actor cannot inherit nominal clearance.
        route=np.asarray(other['route']); ds=np.linalg.norm(route[:,:2]-observed['position'][index,:2],axis=1)
        near=route[int(np.argmin(ds)),:2]
        normal=np.array([-tangent[1],tangent[0]])
        mapped_lateral=abs(float(normal@(near-observed['position'][index,:2])))
        own_lateral = own_corners[:,:2] @ normal
        other_lateral = other_corners[:,:2] @ normal
        current_gap = max(float(own_lateral.min()-other_lateral.max()),float(other_lateral.min()-own_lateral.max()))
        own_radius = float(np.max(np.abs((own_corners[:,:2]-observed['position'][index,:2])@normal)))
        other_radius = float(np.max(np.abs((other_corners[:,:2]-observed['position'][j,:2])@normal)))
        mapped_gap = mapped_lateral-own_radius-other_radius
        bodies_fit=min(mapped_gap,current_gap)>2*margin
        if not bodies_fit:
            passable=False
        others.append({'role':other['role'],'occupied':bool(occupied),'eta_s':None if not math.isfinite(eta) else eta,
                       'stopped_before_entry':bool(head<lo and speed<=.2),'bodies_fit':bool(bodies_fit),
                       'current_body_lateral_gap_m':current_gap,'mapped_body_lateral_gap_m':mapped_gap})
        if not bodies_fit and (occupied or other['role'] in zone['priority_roles'] and eta<=zone['clearance_time_s']):
            triggers.append(other['role'])
    yielding=phase=='before_entry' and bool(triggers)
    state=memory.setdefault(role,{'clear_since_s':None,'was_yielding':False}) if memory is not None else {'clear_since_s':None,'was_yielding':False}
    if yielding:
        state.update(clear_since_s=None,was_yielding=True)
    elif phase=='before_entry' and state['was_yielding']:
        if state['clear_since_s'] is None:
            state['clear_since_s']=float(time_s)
        if time_s-state['clear_since_s']<zone['release_confirmation_s']-1e-9:
            yielding=True
        else:
            state['was_yielding']=False
    gap=start-front-zone['stop_margin_m']
    speed=float(np.linalg.norm(observed['velocity'][index,:2]))
    if yielding:
        acceleration=idm(speed,own['desired_speed_mps'],max(gap,.15),0.,**config['controller']['idm'])
        target=min(target,max(0.,speed+.75*acceleration))
    elif passable and phase!='cleared' and any(x['occupied'] or x['eta_s'] is not None and x['eta_s']<3. for x in others):
        target=min(target,float(zone['passing_speed_m_s']))
    return target,{'policy_version':zone['policy_version'],'phase':phase,'yielding':bool(yielding),
                  'trigger_roles':triggers,'distance_front_to_zone_m':float(start-front),'stop_gap_m':float(gap),
                  'simultaneous_body_clearance':bool(passable),'observed_opposing_actors':others,'state_after':dict(state)}


def moving_box_overlap_interval(first_position, first_velocity, first_yaw, first_half,
                                second_position, second_velocity, second_yaw, second_half,
                                horizon, margin):
    """Continuous SAT overlap under current velocities and fixed current yaws.

    This is a short-horizon collision heuristic. It does not use a planned
    trajectory or assume that either vehicle will keep its future control.
    """
    first_axes = np.array([[math.cos(first_yaw), math.sin(first_yaw)],
                           [-math.sin(first_yaw), math.cos(first_yaw)]])
    second_axes = np.array([[math.cos(second_yaw), math.sin(second_yaw)],
                            [-math.sin(second_yaw), math.cos(second_yaw)]])
    delta = np.asarray(second_position)[:2]-np.asarray(first_position)[:2]
    relative = np.asarray(second_velocity)[:2]-np.asarray(first_velocity)[:2]
    begin, end = 0., float(horizon)
    for axis in np.vstack((first_axes, second_axes)):
        radius = (float(np.abs(first_axes @ axis) @ np.asarray(first_half)[:2])
                  +float(np.abs(second_axes @ axis) @ np.asarray(second_half)[:2])+float(margin))
        distance, velocity = float(delta @ axis), float(relative @ axis)
        if abs(velocity) < 1e-9:
            if abs(distance) > radius:
                return None
            continue
        lower, upper = sorted(((-radius-distance)/velocity, (radius-distance)/velocity))
        begin, end = max(begin, lower), min(end, upper)
        if begin > end:
            return None
    return (begin, end) if end >= 0. and begin <= horizon else None


def live_collision_constraint(index, observed, config, extents, target):
    """Priority does not authorize driving into an observed conflicting body.

    Every actor uses the same present-state rule. A rear actor catching up does
    not make its leader brake; only an actor whose centre is ahead now can
    trigger this longitudinal emergency constraint. No ego branch or future
    route is read. Existing following/right-of-way policies remain in place.
    """
    settings = config['controller']
    horizon = float(settings.get('collision_lookahead_s', 1.5))
    margin = float(settings.get('collision_margin_m', .25))
    maximum_brake = float(settings.get('max_emergency_deceleration_m_s2', 8.))
    own_yaw = math.radians(float(observed['rotation_deg'][index, 2]))
    heading = np.array([math.cos(own_yaw), math.sin(own_yaw)])
    speed = float(np.linalg.norm(observed['velocity'][index, :2]))
    events = []
    for other in range(len(extents)):
        if other == index:
            continue
        vertical_distance = abs(float(observed['position'][other, 2]-observed['position'][index, 2]))
        if vertical_distance > float(extents[index, 2]+extents[other, 2])+margin:
            continue
        delta = observed['position'][other, :2]-observed['position'][index, :2]
        if float(delta @ heading) <= 0.:
            continue
        interval = moving_box_overlap_interval(observed['position'][index], observed['velocity'][index], own_yaw,
            extents[index], observed['position'][other], observed['velocity'][other],
            math.radians(float(observed['rotation_deg'][other, 2])), extents[other], horizon, margin)
        if interval is None:
            continue
        time_to_contact = interval[0]
        deceleration = min(maximum_brake, speed/max(time_to_contact, .25))
        target = min(target, max(0., speed-.75*deceleration))
        events.append({'role': config['vehicle_roles'][other], 'constant_velocity_margin_contact_time_s': time_to_contact,
                       'current_margin_expanded_boxes_overlap': bool(time_to_contact == 0.)})
    return target, {'triggered': bool(events), 'participants': events, 'horizon_s': horizon,
                    'footprint_margin_m': margin,
                    'assumption': 'present velocity and yaw held constant; not a collision-avoidance guarantee'}


def npc_target(index, observed, config, extents, narrow_memory=None, time_s=0.):
    """No branch parameter: live leader and fixed conflict geometry determine action."""
    spec = config["vehicles"][index]
    # Actor policies are part of the frozen scene specification.  The same
    # override is used for all ego branches and all subordinate constraints.
    override = spec.get('npc_policy', {})
    if override:
        settings = {**config['controller'], **override,
                    'idm': {**config['controller']['idm'], **override.get('idm', {})}}
        config = {**config, 'controller': settings}
    speed = float(np.linalg.norm(observed["velocity"][index, :2]))
    settings = config["controller"]
    geometry = settings.get('leader_geometry', 'heading_projection')
    if geometry == 'route_projection':
        gap, leader_speed, leader_index = route_leader(index, observed, extents, settings['lateral_margin_m'], spec['route'])
    elif geometry == 'heading_projection':
        gap, leader_speed, leader_index = live_leader(index, observed, extents, settings['lateral_margin_m'])
    else:
        raise ValueError(f'Unknown leader_geometry: {geometry}')
    acceleration = idm(speed, spec["desired_speed_mps"], gap, leader_speed, **settings["idm"])
    target = max(0., speed + acceleration * .75)
    reason = "leader" if leader_index is not None else "free"
    roles = config["vehicle_roles"]
    zone = config['geometry'].get('priority_zone')
    zone_applies = zone is not None and spec['role'] == zone['yield_role']
    # The area contract replaces the old centre-lane point conflict only for
    # this yielding actor. Other roles and absent-zone configs remain legacy.
    for conflict in ([] if zone_applies else config["geometry"].get("conflicts", [])):
        if spec["role"] not in conflict["roles"] or spec["role"] == conflict["priority_role"]:
            continue
        other_role = conflict["priority_role"]
        other = roles.index(other_role)
        own_route = spec["route"]
        other_route = config["vehicles"][other]["route"]
        own_distance = conflict["route_s"][spec["role"]] - route_progress(observed["position"][index], own_route)
        other_distance = conflict["route_s"][other_role] - route_progress(observed["position"][other], other_route)
        other_speed = float(np.linalg.norm(observed["velocity"][other, :2]))
        own_arrival = max(own_distance, 0.) / max(speed, .5)
        other_arrival = max(other_distance, 0.) / max(other_speed, .2)
        approaching = other_distance > -(extents[other, 0] + 3.)
        temporal_overlap = abs(own_arrival-other_arrival) < settings["conflict_time_gap_s"]
        occupied = -3. < other_distance < extents[other, 0]+3.
        if own_distance > -2. and approaching and ((other_arrival < settings["yield_lookahead_s"] and temporal_overlap) or occupied):
            # Brake towards a virtual stopped leader before the crossing/merge.
            virtual_gap = max(own_distance-extents[index, 0]-3., .15)
            stop_acceleration = idm(speed, spec["desired_speed_mps"], virtual_gap, 0., **settings["idm"])
            target = min(target, max(0., speed + stop_acceleration*.75))
            reason = "yield_to_live_conflict"
    diagnostic = {"role": spec["role"], "reason": reason, "leader_index": leader_index,
                  "leader_gap_m": None if not math.isfinite(gap) else float(gap)}
    if zone_applies:
        target, diagnostic['priority_zone'] = priority_zone_target(index, observed, config, extents, target)
        if diagnostic['priority_zone']['yielding']:
            diagnostic['reason'] = 'yield_to_priority_zone'
    signal = signal_constraint(index, observed, config, extents)
    if signal is not None:
        diagnostic['signal'] = signal
        if signal['must_stop']:
            stop_acceleration = idm(speed, spec['desired_speed_mps'],
                max(signal['distance_front_to_stop_line_m']-1., .15), 0., **settings['idm'])
            target = min(target, max(0., speed+.75*stop_acceleration))
            diagnostic['reason'] = 'stop_for_signal'
    target, people = pedestrian_target(index, observed, config, extents, target)
    if people:
        diagnostic['pedestrian_yield_roles'] = people
        diagnostic['reason'] = 'yield_to_live_pedestrian'
    target, narrow = narrow_meeting_target(index, observed, config, extents, target,narrow_memory,time_s)
    if narrow is not None:
        diagnostic['narrow_meeting'] = narrow
        if narrow['yielding']:
            diagnostic['reason'] = 'yield_on_narrow_road'
    if config.get('action', {}).get('profile_version') == 2:
        target, safety = live_collision_constraint(index, observed, config, extents, target)
        diagnostic['live_collision_constraint'] = safety
        if safety['triggered']:
            diagnostic['reason'] = 'brake_for_observed_conflicting_body'
    return target, diagnostic


def ego_target(branch, time_s, initial_speed, settings):
    time_s = max(float(time_s), 0.)
    speed, lateral = float(initial_speed), 0.
    if 'profiles' in settings:
        profile = _profile_for_branch(settings, branch)
        if settings.get('profile_version') == 2:
            if profile['kind'] in ('longitudinal', 'commit_go'):
                return longitudinal_target(profile, time_s, initial_speed), lateral
            if profile['kind'] not in ('maintain', 'lane_change'):
                raise ValueError('Decision policies require current observations via ScenarioController')
            return speed, lateral
        acceleration = profile['acceleration_m_s2']
        if profile['kind'] == 'longitudinal':
            speed = (max(0., speed+acceleration*time_s) if acceleration < 0.
                     else speed+min(acceleration*time_s, profile['speed_increase_cap_m_s']))
        return speed, lateral
    if branch == "brake":
        speed = max(0., speed-settings["braking_deceleration_m_s2"]*time_s)
    elif branch == "accelerate":
        speed += min(settings["acceleration_m_s2"]*time_s, settings["acceleration_speed_increase_cap_m_s"])
    elif branch in ("left", "right"):
        # Legacy configurations can contain a within-lane offset.
        # A lane-change config steers on its actual map-derived target route;
        # it must never silently fall back to a guessed lane width here.
        if settings.get('lateral_mode', 'legacy_offset') == 'legacy_offset':
            u = float(np.clip((time_s-settings["lateral_onset_s"])/settings["lateral_duration_s"], 0., 1.))
            lateral = (-1 if branch == "left" else 1)*settings["lateral_offset_m"]*(10*u**3-15*u**4+6*u**5)
    return speed, lateral


def validate_lane_change_routes(source_route, settings):
    """Validate the controller contract; map topology is checked by planning.

    Each target sample must correspond to the same longitudinal map station
    as its source sample. Keeping these pairs is essential on curved roads:
    independently resampled lane routes do not have matching arc lengths.
    The planner additionally records actual road/lane IDs and lane widths.
    """
    directions = (_lane_directions(settings['available_lane_change_directions'])
                  if 'available_lane_change_directions' in settings else ('left', 'right'))
    mode = settings.get('lateral_mode', 'legacy_offset')
    if mode == 'legacy_offset':
        return
    if mode != 'adjacent_lane_route':
        raise ValueError(f'Unknown lateral_mode: {mode}')
    if not directions:
        return
    if settings.get('profile_version') != 2:
        onset = _finite_action_number(settings.get('lateral_onset_s'), 'lateral_onset_s')
        duration = _finite_action_number(settings.get('lateral_duration_s'), 'lateral_duration_s')
        if onset < 0 or duration <= 0:
            raise ValueError('Lane-change onset must be nonnegative and duration positive')
    else:
        for profile in settings.get('profiles', {}).values():
            if profile['kind'] == 'lane_change':
                onset = _finite_action_number(profile.get('onset_s', 0.), 'profile.onset_s')
                duration = _finite_action_number(profile.get('duration_s'), 'profile.duration_s')
                if onset < 0 or duration <= 0:
                    raise ValueError('Lane-change onset must be nonnegative and duration positive')
    source = np.asarray(source_route, dtype=float)
    if source.ndim != 2 or source.shape[1] != 4 or len(source) < 2 or not np.isfinite(source).all():
        raise ValueError('Lane-change source route must be finite Nx4 samples')
    yaw = np.radians(source[:, 3])
    normal = np.column_stack((-np.sin(yaw), np.cos(yaw)))
    for side, sign in (('left', -1), ('right', 1)):
        if side not in directions:
            continue
        if side not in settings.get('lane_changes', {}):
            raise ValueError(f'Missing actual adjacent-lane route for {side}')
        declared = settings['lane_changes'][side]
        evidence = declared.get('evidence', {})
        if declared.get('available') is not True:
            raise ValueError(f'{side}: actual adjacent driving lane must be available')
        required = {'source_lane_keys', 'target_lane_keys', 'source_lane_widths_m',
                    'target_lane_widths_m', 'same_direction', 'target_lane_type', 'station_count'}
        if not required.issubset(evidence):
            raise ValueError(f'{side}: incomplete map-derived adjacent-lane evidence')
        if evidence['same_direction'] is not True or evidence['target_lane_type'] != 'Driving':
            raise ValueError(f'{side}: adjacent lane must be same-direction Driving')
        if evidence['station_count'] != len(source):
            raise ValueError(f'{side}: evidence station_count does not match source route')
        keys = []
        widths = []
        for prefix in ('source', 'target'):
            lane_keys = np.asarray(evidence[f'{prefix}_lane_keys'], dtype=float)
            lane_widths = np.asarray(evidence[f'{prefix}_lane_widths_m'], dtype=float)
            if (lane_keys.shape != (len(source), 3) or not np.isfinite(lane_keys).all()
                    or np.any(lane_keys != np.floor(lane_keys)) or np.any(lane_keys[:, 2] == 0)):
                raise ValueError(f'{side}: {prefix} lane keys require Nx3 actual road/section/lane IDs')
            if lane_widths.shape != (len(source),) or not np.isfinite(lane_widths).all() or np.any(lane_widths <= 0.):
                raise ValueError(f'{side}: {prefix} lane widths require N positive actual widths')
            keys.append(lane_keys)
            widths.append(lane_widths)
        if (np.any(keys[0][:, :2] != keys[1][:, :2]) or np.any(keys[0][:, 2] == keys[1][:, 2])
                or np.any(np.sign(keys[0][:, 2]) != np.sign(keys[1][:, 2]))):
            raise ValueError(f'{side}: paired samples must identify distinct same-direction lanes of the same road section')
        target = np.asarray(declared['target_route'], dtype=float)
        if target.shape != source.shape or not np.isfinite(target).all():
            raise ValueError(f'{side}: target route requires finite paired Nx4 samples')
        separation = np.sum((target[:, :2]-source[:, :2])*normal, axis=1)
        if np.any(sign*separation <= widths[0]/2.):
            raise ValueError(f'{side}: target centreline must be beyond the source lane boundary on the declared side')
        expected_spacing = (widths[0]+widths[1])/2.
        # Map sampling tolerance, not a learned/assumed universal lane width.
        if np.any(np.abs(sign*separation-expected_spacing) > np.maximum(.25, .1*expected_spacing)):
            raise ValueError(f'{side}: target centreline spacing disagrees with recorded lane widths')
        if np.any(np.cos(np.radians(target[:, 3]-source[:, 3])) < math.cos(math.radians(45))):
            raise ValueError(f'{side}: target route must be same-direction driving geometry')
        if np.any(np.linalg.norm(np.diff(target[:, :2], axis=0), axis=1) <= 1e-8):
            raise ValueError(f'{side}: target route has a degenerate segment')


def lane_change_route(source_route, branch, time_s, settings):
    """Smoothly move the tracking reference onto the actual adjacent lane.

    Only the reference path is blended. The actor still evolves through the
    physical simulator under throttle/steer/brake, without transform changes.
    After the transition it tracks the target lane centreline permanently.
    """
    source = np.asarray(source_route, dtype=float)
    onset, duration = settings.get('lateral_onset_s', 0.), settings.get('lateral_duration_s', 1.)
    if 'profiles' in settings:
        profile = _profile_for_branch(settings, branch)
        branch = profile.get('direction') if profile['kind'] == 'lane_change' else None
        if settings.get('profile_version') == 2 and branch:
            onset, duration = profile.get('onset_s', 0.), profile['duration_s']
    if branch not in ('left', 'right') or time_s <= onset:
        return source
    target = np.asarray(settings['lane_changes'][branch]['target_route'], dtype=float)
    u = float(np.clip((time_s-onset)/duration, 0., 1.))
    if u == 1.:
        return target
    weight = 10*u**3-15*u**4+6*u**5
    route = source.copy()
    route[:, :3] += weight*(target[:, :3]-source[:, :3])
    # Interpolate heading across +/-180 without rotating the long way around.
    angle = (target[:, 3]-source[:, 3]+180.) % 360.-180.
    route[:, 3] += weight*angle
    return route


def _actual_corners(observed, extents, index):
    """Actual world OBB, including actor and local-box roll/pitch/yaw/offset.

    Strict decisions cannot infer a tail from an actor origin or a blueprint
    size. Legacy NPC/history geometry remains unchanged elsewhere.
    """
    def rotation(value):
        roll, pitch, yaw = np.deg2rad(value)
        cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
        return np.array([[cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr],
                         [cp*sy, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr],
                         [sp, -cp*sr, cp*cr]])
    required = ('position', 'rotation_deg', 'bbox_offset', 'bbox_rotation_deg')
    if any(key not in observed for key in required):
        raise ValueError('Strict clearance requires actual bbox_offset and bbox_rotation_deg observations')
    values = [np.asarray(observed[key][index], dtype=float) for key in required]
    extent = np.asarray(extents[index], dtype=float)
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in values+[extent]) or np.any(extent <= 0):
        raise ValueError('Strict clearance requires finite actual OBB geometry')
    position, angles, offset, box_angles = values
    actor = rotation(angles)
    signs = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    return position+actor@offset+(signs*extent)@(actor@rotation(box_angles)).T


def _hull(points):
    points = sorted(set(map(tuple, np.asarray(points)[:, :2])))
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def half(rows):
        out=[]
        for p in rows:
            while len(out)>1 and cross(out[-2],out[-1],p)<=0: out.pop()
            out.append(p)
        return out
    return np.asarray(half(points)[:-1]+half(points[::-1])[:-1])


def _polygon_overlap(first, second):
    for poly in (first, second):
        for edge in np.roll(poly,-1,axis=0)-poly:
            axis = np.array([-edge[1],edge[0]])
            a,b = first@axis,second@axis
            if a.max() < b.min()-1e-8 or b.max() < a.min()-1e-8: return False
    return True


def _route_sweep_overlap(corners, ego_corners, ego_position, route, start_s, end_s):
    """Intersect actual body with a piecewise straight swept ego footprint.

    This is current occupancy, not an extrapolated or promised future trajectory.
    The ego box is conservatively projected onto each route segment; local box
    offset and rotated overhang are retained. Elevated non-overlapping bodies
    are excluded by the actual current height interval.
    """
    if corners[:,2].min()>ego_corners[:,2].max()+.1 or corners[:,2].max()<ego_corners[:,2].min()-.1:
        return False
    route = np.asarray(route, dtype=float)
    lengths = np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1)
    stations = np.r_[0.,np.cumsum(lengths)]
    body = _hull(corners)
    relative = ego_corners[:,:2]-np.asarray(ego_position)[:2]
    _,_,ego_tangent=project_to_route(ego_position,route)
    ego_normal=np.array([-ego_tangent[1],ego_tangent[0]])
    along,across=relative@ego_tangent,relative@ego_normal
    for k,length in enumerate(lengths):
        if length <= 1e-8: continue
        lo,hi=max(float(start_s),stations[k]),min(float(end_s),stations[k+1])
        if hi<lo: continue
        tangent=(route[k+1,:2]-route[k,:2])/length
        normal=np.array([-tangent[1],tangent[0]])
        # Rotate the current footprint with the local route heading; its
        # measured heading error/offset/overhang remain in this envelope.
        begin=route[k,:2]+tangent*(lo-stations[k])
        end=route[k,:2]+tangent*(hi-stations[k])
        poly=_hull([base+tangent*a+normal*b for base in (begin,end)
                    for a in (along.min(),along.max()) for b in (across.min(),across.max())])
        if _polygon_overlap(body,poly): return True
    return False


def strict_entry_guard(conflict, observed, config, extents):
    """Current known vehicles/pedestrians and signals, not all map obstacles.

    A vehicle queue in the conflict or within braking reach blocks permission.
    Every frame is re-evaluated, including after the first release. Unknown
    static obstacles are outside this observation contract.
    """
    ego=_actual_corners(observed,extents,0)
    route=config['vehicles'][0]['route']
    position=observed['position'][0]
    station,_,_=project_to_route(position,route)
    speed=float(np.linalg.norm(observed['velocity'][0,:2]))
    nominal=config['action']['nominal_policy']
    decel=float(nominal['comfortable_deceleration_m_s2'])
    tau=float(nominal.get('response_time_s',DEFAULT_STOP_RESPONSE_TIME_S))
    end=max(float(conflict['ego_exit_s']), station+4.+speed*tau+speed*speed/(2*decel))
    ego_front=max(project_to_route(p,route)[0] for p in ego)
    actors=[(role,_actual_corners(observed,extents,k)) for k,role in enumerate(config['vehicle_roles']) if k]
    people=observed.get('pedestrians')
    if people is not None:
        actors += [(role,_actual_corners(people,people['bbox_extent'],k)) for k,role in enumerate(people['roles'])]
    blocking=[]; clearances=[]
    for role,corners in actors:
        low=min(project_to_route(p,route)[0] for p in corners)
        high=max(project_to_route(p,route)[0] for p in corners)
        if high < station-1e-6: continue  # behind the ego, not its entry corridor
        if _route_sweep_overlap(corners,ego,position,route,station,end):
            gap=max(0.,low-ego_front-float(nominal['idm']['minimum_gap']))
            blocking.append(role); clearances.append(gap)
    cap=stop_target(0.,min(clearances),decel,tau)[0] if clearances else None
    return {'clear':not blocking,'blocking_roles':blocking,
        'target_speed_limit_m_s':float(cap) if cap is not None else float(nominal['desired_speed_m_s']),
        'scope':'current_observed_vehicle_and_pedestrian_obbs; signals checked separately',
        'sweep_start_route_s':float(station),'sweep_end_route_s':float(end),
        'minimum_available_stopping_distance_m':min(clearances) if clearances else None}


def decision_conflict_observation(conflict, observed, config, extents):
    """Occupancy and approach estimates from present actor boxes and velocities.

    Route intervals are declared map geometry. No future actor controls, walker
    schedules, or branch labels are consulted. Constant-current-velocity ETA
    is an admission heuristic, not a promise of collision avoidance.
    """
    checks, arrivals, occupied_roles = [], [], []
    roles = config['vehicle_roles']
    strict=any(p.get('release_condition')=='specified_actors_cleared_and_entry_clear_v1'
               and p.get('conflict_id')==conflict['id'] for p in config['action']['profiles'].values())
    ego=_actual_corners(observed,extents,0) if strict else None
    for participant in conflict['participants']:
        role = participant['role']
        if role in roles:
            index = roles.index(role)
            route = config['vehicles'][index]['route']
            position, velocity = observed['position'][index], observed['velocity'][index]
            half = float(extents[index, 0])
            corners=_actual_corners(observed,extents,index) if strict else None
        else:
            people = observed.get('pedestrians')
            if people is None or role not in people.get('roles', []):
                raise ValueError(f'Missing current conflict participant observation: {role}')
            index = people['roles'].index(role)
            spec = next((x for x in config.get('pedestrians', []) if x['role'] == role), None)
            if spec is None:
                raise ValueError(f'Unknown pedestrian conflict participant: {role}')
            route = spec['route_points']
            position, velocity = people['position'][index], people['velocity'][index]
            half = float(np.linalg.norm(people['bbox_extent'][index, :2]))
            corners=_actual_corners(people,people['bbox_extent'],index) if strict else None
        station, _, tangent = project_to_route(position, route)
        along_speed = float(np.asarray(velocity)[:2] @ tangent)
        enter, leave = float(participant['entry_s']), float(participant['exit_s'])
        low, high = station-half, station+half
        occupied = high >= enter and low <= leave
        eta = 0. if occupied else (enter-high)/along_speed if high < enter and along_speed > .05 else math.inf
        if low > leave and along_speed < -.05:
            eta = (low-leave)/-along_speed
        if occupied:
            occupied_roles.append(role)
        arrivals.append(eta)
        checks.append({'role': role, 'route_s': station, 'route_speed_m_s': along_speed,
                       'occupied': bool(occupied), 'arrival_s': eta if math.isfinite(eta) else None})
        if strict:
            body_low=min(project_to_route(p,route)[0] for p in corners)
            body_high=max(project_to_route(p,route)[0] for p in corners)
            overlapping=_route_sweep_overlap(corners,ego,observed['position'][0],
                config['vehicles'][0]['route'],conflict['ego_entry_s'],conflict['ego_exit_s'])
            checks[-1].update(body_route_min_s=float(body_low),body_route_max_s=float(body_high),
                body_past_declared_exit=bool(body_low>leave+1e-6),
                overlaps_ego_conflict_sweep=bool(overlapping),
                body_cleared=bool(body_low>leave+1e-6 and not overlapping),
                clearance_rule='actual_entire_body_beyond_exit_and_outside_ego_sweep_v1')
    next_arrival = min(arrivals, default=math.inf)
    result = {'id': conflict['id'], 'occupied': bool(occupied_roles),
            'occupied_roles': occupied_roles,
            'next_arrival_s': next_arrival if math.isfinite(next_arrival) else None, 'participants': checks}
    if strict: result['entry_guard']=strict_entry_guard(conflict,observed,config,extents)
    return result


def ego_decision_context(observed, config, extents):
    """The same normal driving policy feeds history and nominal continuation.

    This policy has its own declared IDM parameters; changing an NPC style does
    not silently change the ego base policy. It obeys present signals and fixed
    right-of-way while other ego profiles intervene on that continuation.
    """
    action = config['action']
    baseline = action['nominal_policy']
    spec = config['vehicles'][0]
    speed = float(np.linalg.norm(observed['velocity'][0, :2]))
    station, _, _ = project_to_route(observed['position'][0], spec['route'])
    gap, leader_speed, leader = route_leader(0, observed, extents,
        float(baseline.get('lateral_margin_m', .3)), spec['route'])
    acceleration = idm(speed, baseline['desired_speed_m_s'], gap, leader_speed, **baseline['idm'])
    target = max(0., speed+.75*acceleration)
    deceleration = float(baseline['comfortable_deceleration_m_s2'])
    response_time = float(baseline.get('response_time_s', DEFAULT_STOP_RESPONSE_TIME_S))
    diagnostic = {'reason': 'leader' if leader is not None else 'free',
                  'leader_index': leader, 'leader_gap_m': gap if math.isfinite(gap) else None,
                  'policy_parameters': baseline, 'yield_conflicts': []}
    conflicts = {}
    for conflict in action.get('conflicts', []):
        current = decision_conflict_observation(conflict, observed, config, extents)
        conflicts[conflict['id']] = current
        before_entry = station+extents[0, 0] < float(conflict['ego_entry_s'])
        stop = float(conflict.get('stop_route_s', conflict['ego_entry_s']-extents[0, 0]-1.5))
        arrival = max(0., float(conflict['ego_entry_s'])-station-extents[0, 0])/max(speed, .2)
        horizon = arrival+float(baseline['release_gap_s'])
        if (conflict.get('ego_must_yield', True) and before_entry
                and (current['occupied'] or current['next_arrival_s'] is not None and current['next_arrival_s'] <= horizon)):
            target = min(target, stop_target(station, stop, deceleration, response_time)[0])
            diagnostic['yield_conflicts'].append(conflict['id'])
            diagnostic['reason'] = 'yield_to_live_conflict'
    signal = signal_constraint(0, observed, config, extents)
    if signal is not None and signal['must_stop']:
        distance = max(0., signal['distance_front_to_stop_line_m']-1.)
        target = min(target, stop_target(0., distance, deceleration, response_time)[0])
        diagnostic['reason'] = 'stop_for_signal'
    # Use the ego baseline's IDM settings for pedestrian slowing as well.
    local_config = {**config, 'controller': {**config['controller'], 'idm': baseline['idm'],
        'yield_lookahead_s': float(baseline.get('yield_lookahead_s', 4.))}}
    target, people = pedestrian_target(0, observed, local_config, extents, target)
    if people:
        diagnostic.update(reason='yield_to_live_pedestrian', pedestrian_roles=people)
    target, safety = live_collision_constraint(0, observed, config, extents, target)
    diagnostic['live_collision_constraint'] = safety
    if safety['triggered']:
        diagnostic['reason'] = 'brake_for_observed_conflicting_body'
    return {'route_s': station, 'speed': speed, 'nominal_target': target,
            'nominal_desired_speed_m_s': float(baseline['desired_speed_m_s']),
            'nominal_hold': bool(target <= .05), 'nominal_diagnostic': diagnostic,
            'stop_deceleration': deceleration, 'stop_response_time_s': response_time,
            'conflicts': conflicts, 'signal': signal}


def validate_decision_geometry(config):
    """Reject decision waypoints/participants outside their frozen routes."""
    if config['action'].get('profile_version') != 2:
        return
    action = config['action']
    specs = {x['role']: x for x in config['vehicles']}
    specs.update({x['role']: x for x in config.get('pedestrians', [])})
    def length(spec):
        route = np.asarray(spec.get('route', spec.get('route_points')), dtype=float)
        return float(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1).sum())
    ego_length = length(config['vehicles'][0])
    conflicts = {x['id']: x for x in action.get('conflicts', [])}
    for conflict in conflicts.values():
        if float(conflict['ego_exit_s']) > ego_length+1e-6:
            raise ValueError('Ego conflict interval extends beyond the mapped route')
        for actor in conflict['participants']:
            if actor['role'] not in specs:
                raise ValueError('Decision conflict identifies an unknown actor')
            if float(actor['exit_s']) > length(specs[actor['role']])+1e-6:
                raise ValueError('Participant interval extends beyond its mapped route')
    for name, profile in action['profiles'].items():
        if 'stop_route_s' in profile and not 0. <= float(profile['stop_route_s']) <= ego_length:
            raise ValueError(f'{name}: stopping station is outside ego route')
        if profile['kind'] == 'yield_wait' and float(profile['stop_route_s']) >= conflicts[profile['conflict_id']]['ego_entry_s']:
            raise ValueError('yield_wait stopping station must precede conflict entry')
        if 'duration_s' in config:
            if profile.get('onset_s', 0.) >= config['duration_s']:
                raise ValueError(f'{name}: intervention starts outside the future window')
            if float(profile.get('stop_deadline_s', 0.)) > config['duration_s']:
                raise ValueError(f'{name}: stopping deadline exceeds the future window')
            if profile['kind'] == 'lane_change':
                end = (profile.get('onset_s', 0.)+profile['duration_s']
                       +profile.get('settling_time_s', 0.)+profile.get('required_final_hold_s', .5))
                if end > config['duration_s']+1e-9:
                    raise ValueError(f'{name}: insufficient transition, settling and unchanged completion-hold budget')


class ScenarioController:
    def __init__(self, config, dt=.05):
        self.config, self.dt = config, float(dt)
        self.integral = np.zeros(len(config["vehicles"]))
        self.branches = branch_names(config)
        self.ego_decision = EgoDecisionState()
        self.narrow_memory = {}
        self.merge_memory = {}
        self.merge_warmup = []
        self.merge_history_start = None
        self.merge_t0 = None
        self.active_routes = {}
        for actor in config['vehicles']:
            npc_merge.validate(actor)
        validate_lane_change_routes(config['vehicles'][0]['route'], config['action'])
        validate_priority_zone(config)
        validate_auxiliary_config(config)
        validate_decision_geometry(config)

    def controls(self, observed, branch, time_s, extents, ego_base_speed=None):
        if 'profiles' in self.config['action'] and branch not in self.branches:
            raise ValueError(f'Unknown action branch for this scene: {branch}')
        n = len(self.config["vehicles"])
        controls, targets = np.zeros((n, 3)), np.zeros(n)
        diagnostics = []
        speeds = np.linalg.norm(observed["velocity"][:, :2], axis=1)
        has_merge = any(npc_merge.enabled(a) for a in self.config['vehicles'])
        if has_merge and abs(time_s+2.) < 1e-8:
            self.merge_history_start = copy.deepcopy(self.merge_memory)
        if has_merge and abs(time_s) < 1e-8:
            self.merge_t0 = copy.deepcopy(self.merge_memory)
        for i, spec in enumerate(self.config["vehicles"]):
            lateral = 0.
            force_hold = False
            terminal_brake = .4
            route = spec['route']
            if i == 0:
                base = spec["initial_speed_mps"] if ego_base_speed is None else ego_base_speed
                active_branch = "maintain" if time_s < 0 else branch
                if self.config['action'].get('profile_version') == 2:
                    context = ego_decision_context(observed, self.config, extents)
                    if time_s < 0:
                        target, force_hold = context['nominal_target'], context['nominal_hold']
                        diagnostic = {'role': spec['role'], 'kind': 'nominal', 'phase': 'common_history',
                                      'baseline': context['nominal_diagnostic'], 'force_brake_hold': force_hold}
                    else:
                        profile = _profile_for_branch(self.config['action'], branch)
                        if time_s >= profile.get('onset_s', 0.):
                            terminal_brake = float(profile.get('terminal_brake_min', .4))
                        target, force_hold, diagnostic = self.ego_decision.evaluate(
                            active_branch, profile, time_s, base, context)
                        diagnostic['role'] = spec['role']
                    diagnostics.append(diagnostic)
                else:
                    target, lateral = ego_target(active_branch, time_s, base, self.config["action"])
                if self.config['action'].get('lateral_mode') == 'adjacent_lane_route':
                    route = lane_change_route(route, active_branch, time_s, self.config['action'])
                signal = signal_constraint(i, observed, self.config, extents)
                if signal is not None and self.config['action'].get('profile_version') != 2:
                    diagnostics.append({'role': spec['role'], 'reason': 'explicit_ego_intervention',
                                        'signal': signal, 'ego_action_overridden': False})
            else:
                npc_config = self.config
                merge_diagnostic = None
                if npc_merge.enabled(spec):
                    route, merge_diagnostic = npc_merge.decide(i,observed,self.config,extents,
                        self.merge_memory,time_s,self.dt)
                    # Following is computed along the currently chosen route,
                    # including source-lane traffic while waiting/returning.
                    vehicles = list(self.config['vehicles'])
                    vehicles[i] = {**spec,'route':route}
                    npc_config = {**self.config,'vehicles':vehicles}
                target, diagnostic = npc_target(i, observed, npc_config, extents,self.narrow_memory,time_s)
                if merge_diagnostic is not None:
                    diagnostic['dynamic_merge'] = merge_diagnostic
                diagnostics.append(diagnostic)
            self.active_routes[i] = route
            error = target-speeds[i]
            self.integral[i] = np.clip(self.integral[i]+error*self.dt, -1.5, 1.5)
            effort = .16 + .42*error + .10*self.integral[i]
            throttle, brake = (min(effort, .9), 0.) if effort >= 0 else (0., min(-effort*.7, 1.))
            if self.config['action'].get('profile_version') == 2 and (force_hold or target <= .01):
                throttle, brake = 0., max(brake, terminal_brake)
                self.integral[i] = 0.
            steering = steer_to_route(observed["position"][i], math.radians(observed["rotation_deg"][i, 2]), speeds[i], route, lateral)
            controls[i] = [throttle, steering, brake]
            targets[i] = target
        if has_merge and time_s < -2.-1e-8:
            self.merge_warmup.append({'time_s':float(time_s),
                'observed':{key:np.asarray(observed[key]).tolist() for key in
                    ('position','velocity','rotation_deg','bbox_offset','bbox_rotation_deg')},
                'bbox_extent':np.asarray(extents).tolist(),
                'decisions':{d['role']:d['dynamic_merge'] for d in diagnostics if 'dynamic_merge' in d}})
        return controls, targets, diagnostics

    def merge_evidence(self):
        return {'schema':'npc-merge-memory-v1','warmup':self.merge_warmup,
                'history_start_state':self.merge_history_start,
                't0_state':self.merge_t0 if self.merge_t0 is not None else copy.deepcopy(self.merge_memory)}
