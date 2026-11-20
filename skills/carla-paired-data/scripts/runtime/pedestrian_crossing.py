"""Causal, bounded mid-crossing feedback, independent of ego action labels.

Other actors are extrapolated at their *measured current* velocity and fixed
orientation. This is a local risk heuristic, not a human model or safety proof.
Every alternative includes braking/reversal time and a stationary endpoint;
continuous swept-box checks prevent a fast car tunnelling between samples.
"""
from __future__ import annotations

import math
import itertools
import numpy as np

KIND = 'reactive_crossing_v2'
KIND_V3 = 'reactive_crossing_v3'
KIND_V4 = 'reactive_crossing_v4'
DEFAULTS = {'lookahead_s': 2.5, 'body_margin_m': .25, 'caution_margin_m': .75,
            'slow_speed_ratio': .5, 'retreat_speed_m_s': .8, 'max_retreat_m': 1.2,
            'acceleration_m_s2': 1.5, 'deceleration_m_s2': 3., 'resume_clear_s': .4}
MODES = ('continue', 'slow', 'stop', 'retreat')
DEFAULTS_V3 = {**DEFAULTS, 'fast_speed_ratio': 1.5, 'fast_speed_cap_m_s': 2.5}


def continuous(kind):
    return kind in (KIND, KIND_V3, KIND_V4)


def timed_entry(kind):
    return kind in (KIND_V3, KIND_V4)


def validate(parameters, kind=KIND):
    expected = DEFAULTS_V3 if timed_entry(kind) else DEFAULTS
    if not isinstance(parameters, dict) or set(parameters) != set(expected):
        raise ValueError('reactive_crossing_v2 requires complete crossing_feedback parameters')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
           for x in parameters.values()):
        raise ValueError('crossing_feedback parameters must be finite and positive')
    p = parameters
    if not (1. <= p['lookahead_s'] <= 4. and p['body_margin_m'] < p['caution_margin_m'] <= 2.
            and .1 <= p['slow_speed_ratio'] < 1. and p['retreat_speed_m_s'] <= 1.5
            and p['max_retreat_m'] <= 2. and p['acceleration_m_s2'] <= 4.
            and p['deceleration_m_s2'] <= 6. and p['resume_clear_s'] <= 1.):
        raise ValueError('crossing_feedback parameters outside declared engineering bounds')
    if timed_entry(kind) and not (1. < p['fast_speed_ratio'] <= 2. and 1.5 <= p['fast_speed_cap_m_s'] <= 3.):
        raise ValueError('Fast crossing parameters outside declared engineering bounds')


def initial_state():
    return {'released': False, 'mode': 'wait', 'clear_since_s': None,
            'retreat_target_s_m': None, 'retreat_origin_s_m': None,
            'last_command_speed_m_s': 0.}


def route_points(route, stations, braking_extension=False):
    route = np.asarray(route, float)
    lengths = np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    points = np.stack([np.interp(stations, cumulative, route[:, axis]) for axis in range(3)], axis=-1)
    if braking_extension:
        # A measured endpoint overshoot must brake, not snap or reverse. The
        # short inertial extension is included in the swept risk envelope.
        stations = np.asarray(stations)
        for mask, anchor, direction, origin in (
                (stations < 0., route[0], (route[1]-route[0])/lengths[0], 0.),
                (stations > cumulative[-1], route[-1], (route[-1]-route[-2])/lengths[-1], cumulative[-1])):
            points[mask] = anchor+(stations[mask]-origin)[:, None]*direction
    return points


def motion_profile(position, velocity, route, station, tangent, mode, target, spec, dt):
    """Signed route motion: slow before reversing; never teleport to a target."""
    p = spec['response_policy']['crossing_feedback']
    bounded = timed_entry(spec['response_policy'].get('kind'))
    count = int(math.ceil(p['lookahead_s']/dt))
    times = np.arange(count+1)*dt
    end = float(np.linalg.norm(np.diff(np.asarray(route)[:, :2], axis=0), axis=1).sum())
    speed = float(np.dot(np.asarray(velocity)[:2], tangent))
    stations = [station]
    signed_speeds = []
    desired = {'continue': spec['speed_mps'], 'slow': spec['speed_mps']*p['slow_speed_ratio'],
               'stop': 0., 'retreat': -min(p['retreat_speed_m_s'], spec['speed_mps']),
               'fast': min(spec['speed_mps']*p.get('fast_speed_ratio', 1.), p.get('fast_speed_cap_m_s', spec['speed_mps']))}[mode]
    for _ in range(count):
        # Reversal first decelerates to zero, then accelerates in the new direction.
        goal = 0. if speed*desired < 0 else desired
        direction = -1. if desired < 0 else 1.
        bound = target if direction < 0 else end
        remaining = max(0., (bound-stations[-1])*direction)
        stop_speed = math.sqrt(2*p['deceleration_m_s2']*remaining)
        if bounded:
            # Discrete braking needs one tick of margin before its target.
            half_tick = .5*p['deceleration_m_s2']*dt
            stop_speed = math.sqrt(half_tick*half_tick+2*p['deceleration_m_s2']*remaining)-half_tick
            stop_speed = min(stop_speed,remaining/dt)
        goal = math.copysign(min(abs(goal), stop_speed), goal)
        limit = p['deceleration_m_s2'] if abs(goal) < abs(speed) else p['acceleration_m_s2']
        speed += float(np.clip(goal-speed, -limit*dt, limit*dt))
        next_station = stations[-1]+speed*dt
        if not bounded:
            next_station = float(np.clip(next_station, 0., end))
            if mode == 'retreat' and speed < 0:
                next_station = max(target, next_station)
        speed = (next_station-stations[-1])/dt
        signed_speeds.append(speed); stations.append(next_station)
    points = route_points(route, stations, braking_extension=bounded)
    points += np.asarray(position)-points[0]  # retain measured offset, no route snap
    # The route supplies horizontal control; do not predict stair climbing from a polyline.
    points[:, 2] = position[2]
    control = np.zeros(5)
    delta = points[1]-points[0]
    distance = float(np.linalg.norm(delta))
    if distance > 1e-10:
        control[:3] = delta/distance
        control[3] = distance/dt
    return times, points, control, signed_speeds[0]


def swept_interval(local_path, half_extent, times):
    """Continuous segment/slab intersection against a stationary inflated OBB."""
    start, delta = local_path[:-1], np.diff(local_path, axis=0)
    moving = np.abs(delta) > 1e-12
    safe_delta = np.where(moving, delta, 1.)
    a, b = (-half_extent-start)/safe_delta, (half_extent-start)/safe_delta
    lower = np.where(moving, np.minimum(a, b), -np.inf)
    upper = np.where(moving, np.maximum(a, b), np.inf)
    blocked = np.any(~moving & (np.abs(start) > half_extent), axis=1)
    lo = np.maximum(0., lower.max(axis=1))
    hi = np.minimum(1., upper.min(axis=1))
    hits = ~blocked & (hi >= lo)
    if not hits.any():
        return None, 0.
    first = int(np.flatnonzero(hits)[0])
    return float(times[first]+lo[first]*(times[first+1]-times[first])), float(
        np.sum(np.where(hits, np.maximum(0., hi-lo), 0.)*np.diff(times)))


def rollout_envelope(points, walker_half, measured_path=None):
    if measured_path is not None:
        # Enclose both measured-speed and command-speed rollouts, including the
        # space between them. Two clear endpoints alone are not a clear corridor.
        walker_half = walker_half+np.abs(points-measured_path)/2.
        points = (points+measured_path)/2.
    return points, walker_half


def obstacle_travel(obstacle, times):
    velocity = obstacle[4]
    travel = times[:, None]*velocity
    if len(obstacle) > 5 and obstacle[5] > 0:
        speed = float(np.linalg.norm(velocity))
        stopping_time = np.minimum(times, speed/obstacle[5])
        travel = (speed*stopping_time-.5*obstacle[5]*stopping_time**2)[:, None]*velocity/max(speed, 1e-9)
    return travel


def risk(points, times, obstacles, walker_half, parameters, measured_path=None):
    points, walker_half = rollout_envelope(points, walker_half, measured_path)
    checks = []
    for obstacle in obstacles:
        role, center, orientation, extent, velocity = obstacle[:5]
        travel = obstacle_travel(obstacle, times)
        local = (points-center-travel) @ orientation
        # Walker envelope is yaw independent, covering turn/reversal and capsule width.
        inflated = extent+walker_half @ np.abs(orientation)
        if inflated.ndim == 2:
            inflated = np.maximum(inflated[:-1], inflated[1:])
        hard, duration = swept_interval(local, inflated+parameters['body_margin_m'], times)
        caution, _ = swept_interval(local, inflated+parameters['caution_margin_m'], times)
        checks.append({'role': role, 'body_risk_time_s': hard, 'caution_time_s': caution,
                       'body_risk_duration_s': duration})
    hard = [r['body_risk_time_s'] for r in checks if r['body_risk_time_s'] is not None]
    caution = [r['caution_time_s'] for r in checks if r['caution_time_s'] is not None]
    return {'body_clear': not hard, 'caution_clear': not caution,
            'first_body_risk_s': min(hard) if hard else None,
            'first_caution_s': min(caution) if caution else None,
            'body_risk_duration_s': sum(r['body_risk_duration_s'] for r in checks),
            'actor_checks': checks}


def observed_leader_braking(vehicles, config, index):
    """Only extrapolate observed braking behind a stopped same-route leader.

    No planned controls or promise of future yielding. No lane-change route,
    moving leader, lateral escape, or insufficient measured stopping margin is
    treated as a blocked queue. All roles still participate in collision tests.
    """
    from_module = __import__(__package__+'.pedestrian_policy' if __package__ else 'pedestrian_policy', fromlist=['body_interval'])
    actor = config['vehicles'][index]
    if actor.get('route_maneuver'):
        return 0.
    rear, front, tangent = from_module.body_interval(vehicles,index,actor['route'],'transform_world_vehicle')
    speed = float(vehicles['velocity'][index,:2] @ tangent)
    decel = -float(vehicles['acceleration'][index,:2] @ tangent)
    if speed <= .3 or decel < .5 or decel > 6.:
        return 0.
    for j, leader in enumerate(config['vehicles']):
        if j == index or not np.array_equal(actor['route'], leader['route']):
            continue
        lo, hi, _ = from_module.body_interval(vehicles,j,actor['route'],'transform_world_vehicle')
        lateral = abs(float(np.array([-tangent[1],tangent[0]]) @ (vehicles['position'][j,:2]-vehicles['position'][index,:2])))
        if (lo > front+1. and lateral < .5 and np.linalg.norm(vehicles['velocity'][j,:2]) <= .3
                and speed*speed/(2*decel)+1. < lo-front):
            return decel
    return 0.


def obstacles_from(vehicles, people, index, config, rotation, version3=False):
    obstacles = []
    for observed, roles, transform_key, excluded in (
        (vehicles, config['vehicle_roles'], 'transform_world_vehicle', None),
        (people, [s['role'] for s in config.get('pedestrians', [])], 'transform_world_actor', index)):
        for i, role in enumerate(roles):
            if i == excluded:
                continue
            transform = observed[transform_key][i]
            center = transform[:3, 3]+transform[:3, :3] @ observed['bbox_offset'][i]
            orientation = transform[:3, :3] @ rotation(observed['bbox_rotation_deg'][i])
            velocity = np.array(observed['velocity'][i], dtype=float, copy=True)
            velocity[2] = 0.
            obstacle = (role, center, orientation, observed['bbox_extent'][i], velocity)
            if version3 and transform_key == 'transform_world_vehicle':
                obstacle += (observed_leader_braking(vehicles, config, i),)
            obstacles.append(obstacle)
    return obstacles


def least_risk_mode(alternatives, previous, dt):
    """Do not reverse an escape for sub-tick differences in predicted impact time.

    Prefer later impact, then shorter exposure, with half a physics interval as
    tie resolution. Preserve the ongoing response among effectively tied choices.
    None of these alternatives is clear; this ranking never certifies safety.
    """
    tolerance = dt/2.
    latest = max(row['first_body_risk_s'] for row in alternatives.values())
    choices = [mode for mode, row in alternatives.items()
               if row['first_body_risk_s'] >= latest-tolerance]
    shortest = min(alternatives[mode]['body_risk_duration_s'] for mode in choices)
    choices = [mode for mode in choices
               if alternatives[mode]['body_risk_duration_s'] <= shortest+tolerance]
    return previous if previous in choices else choices[0]


def entry_side_refuge(alternatives, paths, obstacles, walker_half, tangent, parameters, times):
    """Prefer a reachable near-side refuge while approaching a threatened lane.

    This is evaluated only when ordinary forward motion has observed risk.
    Full swept-body checks still assess each option, including every actor.
    Stopping at a refuge must clear the existing caution corridor, not just
    touch the minimum body margin and cancel a retreat with zero clearance.
    A retreat may first reach the body-clear corridor; keep its bounded target
    while moving to a place where stopping satisfies the caution margin.
    Once moving away from a threatening lane centre, keep the forward-clearance
    ranking. Never decide from a requested scene phase or an ego action label.
    """
    threats = {r['role'] for r in alternatives['continue']['actor_checks']
               if r['caution_time_s'] is not None}
    checks = []
    refuge_masks = {mode: np.ones(len(times), dtype=bool) for mode in ('stop','retreat','fast') if mode in paths}
    for obstacle in obstacles:
        role, center, orientation, extent = obstacle[:4]
        if role not in threats:
            continue
        normal = orientation[:, 1]
        travel = float(np.r_[tangent, 0.] @ normal)
        lateral = float((paths['continue'][0][0]-center) @ normal)
        # A crossing must have an identifiable entry side. Do not use the
        # long axis of a parallel/strongly oblique obstacle as a lane refuge.
        # Vehicle obstacles carry an observed-braking slot; pedestrians do not.
        # Do not reinterpret another walker's orientation as a traffic lane.
        entry_side = len(obstacle) > 5 and abs(travel) >= .5 and lateral*travel < 0.
        side = -1. if travel > 0. else 1.
        half = float(extent[1]+walker_half @ np.abs(normal)+parameters['body_margin_m'])
        clearances = {}
        for mode in refuge_masks:
            direction = -side if mode == 'fast' else side
            command_path, measured_path = paths[mode]
            midpoint, envelope_half = rollout_envelope(command_path, walker_half, measured_path)
            moving_center = center+obstacle_travel(obstacle, times)
            # Use exactly the same spatial envelope and current-state obstacle
            # prediction as the continuous risk check, including lateral motion.
            margin=parameters['caution_margin_m'] if mode=='stop' else parameters['body_margin_m']
            inflated = extent[1]+envelope_half @ np.abs(normal)+margin
            clearance = direction*((midpoint-moving_center) @ normal)-inflated
            clearances[mode] = float(clearance[-1])
            refuge_masks[mode] &= clearance > 0.
        checks.append({'role':role,'entry_side':bool(entry_side),
                       'signed_lateral_distance_m':lateral,'corridor_half_width_m':half,
                       'endpoint_near_side_clearance_m':clearances})
    exit_times = {}
    for mode, mask in refuge_masks.items():
        stays_clear = np.logical_and.accumulate(mask[::-1])[::-1]
        exit_times[mode] = float(times[np.flatnonzero(stays_clear)[0]]) if stays_clear.any() else None
    evidence = {'policy':'observed_entry_side_reachable_refuge_v3','threats':checks,
                'stop_refuge_clearance_margin_m':parameters['caution_margin_m'],
                'clearance_margin_m_by_mode':{m:parameters['caution_margin_m'] if m=='stop'
                    else parameters['body_margin_m'] for m in refuge_masks},
                'body_sweep_margin_m':parameters['body_margin_m'],
                'dynamic_clearance_times_s':exit_times,
                'includes_braking_reversal_and_command_execution_envelope':True,
                'uses_same_moving_obstacles_and_aabb_envelope_as_risk':True,
                'clearance_times_are_prediction_grid_samples':True}
    if not checks or not all(r['entry_side'] for r in checks):
        return None, evidence
    for mode in ('stop','retreat'):
        if (mode in alternatives and alternatives[mode]['body_clear']
                and all(r['endpoint_near_side_clearance_m'].get(mode,-1.) > 0. for r in checks)):
            forward = exit_times.get('fast')
            if (mode == 'retreat' and alternatives['fast']['body_clear'] and forward is not None
                    and (exit_times[mode] is None or exit_times[mode] > forward+1e-9)):
                continue
            return mode, evidence
    return None, evidence


def step(spec, config, vehicles, people, index, time_s, dt, state, *, policy_module):
    """Return next-interval command, auditable decision, and fresh causal state."""
    state = dict(state) if isinstance(state, dict) else initial_state()
    before = dict(state)
    version3 = timed_entry(spec['response_policy']['kind'])
    gap = policy_module.decision(spec, config, vehicles, people, index, time_s, state['released'])
    released = bool(state['released'] or (gap['allow_walk'] and not version3))
    row = {**gap, 'kind': spec['response_policy']['kind'], 'committed_before': state['released'], 'committed_after': released,
           'state_before': before, 'prediction': 'current_velocity_fixed_orientation_swept_boxes',
           'alternatives': {}, 'unavoidable_predicted_conflict': False}
    if version3:
        row['prediction'] = 'timed_body_sweeps_with_observed_stopped_leader_braking'
    state['released'] = released
    if not released and not version3:
        row.update(mode='wait', state_after=state, signed_command_speed_m_s=0.)
        return np.zeros(5), row, state
    p = spec['response_policy']['crossing_feedback']
    position = people['position'][index]
    station, tangent = policy_module.route_projection(position, spec['route_points'])
    target = (state['retreat_target_s_m'] if state['retreat_target_s_m'] is not None else
              max(0., station-p['max_retreat_m']))
    obstacles = obstacles_from(vehicles, people, index, config, policy_module._rotation, version3)
    transform = people['transform_world_actor'][index]
    offset = transform[:3, :3] @ people['bbox_offset'][index]
    # Center follows actor position plus measured height; horizontal envelope covers offsets.
    extent = np.asarray(people['bbox_extent'][index])
    orientation = transform[:3, :3] @ policy_module._rotation(people['bbox_rotation_deg'][index])
    corners = (np.asarray(list(itertools.product((-1., 1.), repeat=3)))*extent) @ orientation.T
    radius = float(np.linalg.norm(corners[:, :2], axis=1).max()+np.linalg.norm(offset[:2]))
    walker_half = np.array([radius, radius, float(np.abs(orientation[2]) @ extent)])
    if not released and version3:
        # Test the WHOLE remaining forward crossing against each body's timed
        # sweep. A late-arriving car must not block an already-cleared lane.
        # This replaces a single all-lanes gap with geometry, not smaller margins.
        import copy
        entry_spec = copy.deepcopy(spec)
        remaining = float(np.linalg.norm(np.diff(np.asarray(spec['route_points'])[:,:2],axis=0),axis=1).sum())-station
        horizon = max(p['lookahead_s'], remaining/spec['speed_mps']+spec['speed_mps']/p['acceleration_m_s2']+1.)
        entry_spec['response_policy']['crossing_feedback']['lookahead_s'] = horizon
        times, points, _, _ = motion_profile(position, people['velocity'][index],spec['route_points'],station,tangent,'continue',target,entry_spec,dt)
        points[:,2] += offset[2]
        entry = risk(points,times,obstacles,walker_half,p)
        released = time_s+1e-9 >= spec['start_time_s'] and entry['caution_clear']
        row.update(entry_sweep=entry,entry_horizon_s=horizon,committed_after=bool(released),
                   prediction='timed_body_sweeps_with_observed_stopped_leader_braking')
        state['released'] = bool(released)
        if not released:
            row.update(mode='wait',state_after=state,signed_command_speed_m_s=0.,allow_walk=False,
                       reason='before_earliest_start' if time_s < spec['start_time_s'] else 'wait_for_clear_timed_body_sweep')
            return np.zeros(5),row,state
    modes = ('continue','fast','slow','stop','retreat') if version3 else MODES
    controls, speeds, paths = {}, {}, {}
    for mode in modes:
        if mode == 'retreat' and station-target < .02 and state['mode'] != 'retreat':
            continue  # Never step behind the original route start or invent unexplored space.
        times, measured, _, _ = motion_profile(position, people['velocity'][index],
            spec['route_points'], station, tangent, mode, target, spec, dt)
        previous_command = np.r_[tangent*state['last_command_speed_m_s'], 0.]
        times, points, control, speed = motion_profile(position, previous_command,
            spec['route_points'], station, tangent, mode, target, spec, dt)
        points[:, 2] += offset[2]
        measured[:, 2] += offset[2]
        row['alternatives'][mode] = risk(points, times, obstacles, walker_half, p, measured)
        controls[mode], speeds[mode] = control, speed
        paths[mode] = (points, measured)
    alternatives = row['alternatives']
    refuge = None
    if spec['response_policy']['kind'] == KIND_V4 and not alternatives['continue']['caution_clear']:
        refuge, evidence = entry_side_refuge(alternatives,paths,obstacles,walker_half,tangent,p,times)
        row['refuge_comparison'] = evidence
    # Prefer progress when it is clear. Slowing/stopping is not automatically safer in a lane.
    if alternatives['continue']['caution_clear']:
        chosen, reason = 'continue', 'forward_path_clear'
    elif refuge is not None:
        chosen, reason = refuge, ('pause_before_threatened_corridor' if refuge == 'stop'
                                  else 'return_to_reachable_entry_side')
    elif version3 and alternatives['fast']['caution_clear']:
        chosen, reason = 'fast', 'faster_forward_motion_clears_observed_risk'
    elif alternatives['slow']['caution_clear']:
        chosen, reason = 'slow', 'slow_opens_clearance'
    else:
        clear = [m for m in modes if m in alternatives and alternatives[m]['body_clear']]
        if clear:
            chosen = clear[0]
            reason = {'continue': 'continue_clears_conflict', 'fast': 'fast_clears_conflict', 'slow': 'slow_avoids_conflict',
                      'stop': 'stop_outside_predicted_sweep', 'retreat': 'bounded_retreat_to_clear_path'}[chosen]
        else:
            # No candidate is certified clear: report it, do not label a risky pause safe.
            chosen = least_risk_mode(alternatives, state['mode'], dt)
            reason = 'no_clear_alternative_delay_predicted_conflict'
            row['unavoidable_predicted_conflict'] = True
    resuming = (chosen in ('continue','fast') and state['mode'] == 'slow' or
                chosen in ('continue', 'fast', 'slow') and state['mode'] in ('stop', 'retreat'))
    if resuming:
        # Hysteresis is only allowed while the previous response remains body-clear.
        if alternatives[state['mode']]['body_clear']:
            if state['clear_since_s'] is None:
                state['clear_since_s'] = float(time_s)
            if time_s-state['clear_since_s'] < p['resume_clear_s']-1e-9:
                chosen, reason = state['mode'], 'confirm_clear_before_resuming'
        else:
            state['clear_since_s'] = None
    else:
        state['clear_since_s'] = None
    if chosen == 'retreat':
        if state['retreat_origin_s_m'] is None:
            state['retreat_origin_s_m'] = float(station)
            state['retreat_target_s_m'] = float(target)
    elif chosen in ('continue', 'fast', 'slow'):
        state['retreat_origin_s_m'] = state['retreat_target_s_m'] = None
    state['mode'] = chosen
    state['last_command_speed_m_s'] = speeds[chosen]
    row.update(reason=reason, mode=chosen, allow_walk=bool(controls[chosen][3] > 0),
               state_after=state, signed_command_speed_m_s=speeds[chosen],
               route_station_m=float(station), motion_prediction='envelope_of_measured_and_command_speed_rollouts')
    return controls[chosen], row, state
