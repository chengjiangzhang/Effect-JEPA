"""Offline preflight planning from the measured, unmodified common history.

This module never connects to CARLA. Kinematic envelopes identify impossible
contracts and report weak opportunities; they do not predict NPC decisions or
certify the future. The full physics, execution, prefix and RGB gates still run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

if __package__:
    from .runtime.controllers_v2 import project_to_route
    from .runtime.ego_decisions import validate_profiles, DEFAULT_STOP_RESPONSE_TIME_S
    from .runtime.vehicle_geometry import common_history_vehicle_geometry
else:
    from runtime.controllers_v2 import project_to_route
    from runtime.ego_decisions import validate_profiles, DEFAULT_STOP_RESPONSE_TIME_S
    from runtime.vehicle_geometry import common_history_vehicle_geometry

POLICY_VERSION = 'measured-t0-kinematic-planning-v1'
PLAN_SCHEMA = 'measured-t0-plan-v1'
EVIDENCE_SCHEMA = 'measured-t0-evidence-v1'
HISTORY_MODE = 'common_history_only_not_delivery'
DT_S = .05
WINDOW_S = 4.
RECOVERY_OBSERVATION_S = .5  # One saved 2 Hz future interval; not a new execution tolerance.
EPS = 1e-8


class HistoryEvidenceError(ValueError):
    """Missing or inconsistent measurements are not a failed action candidate."""


class HistorySetupError(ValueError):
    """Valid measurements show invalid shared initialization/history."""

    def __init__(self, code, details=None):
        super().__init__(code)
        self.details = details or {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _spec_digest(spec):
    return digest({k: v for k, v in spec.items() if k != 'spec_sha256'})


def _number(value, field, low=0.):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < low:
        raise ValueError(f'{field}: expected a finite number >= {low}')
    return float(value)


def travel_bound(speed, acceleration, duration, cap=None):
    """Distance for a nonnegative speed with a positive constant acceleration cap.

    No tracking error, traffic, friction or route curvature is simulated. This
    optimistic engineering envelope is not a certified bound on CARLA physics.
    """
    speed = _number(speed, 'speed'); acceleration = _number(acceleration, 'acceleration')
    duration = _number(duration, 'duration')
    cap = speed + acceleration * duration if cap is None else max(speed, _number(cap, 'cap'))
    ramp = duration if acceleration <= EPS else min(duration, (cap-speed)/acceleration)
    return speed*duration + acceleration*ramp*(duration-.5*ramp)


def arrival_bound(distance, speed, acceleration=0., cap=None, limit=WINDOW_S):
    if distance <= EPS:
        return 0.
    if travel_bound(speed, acceleration, limit, cap) + EPS < distance:
        return None
    lo, hi = 0., float(limit)
    for _ in range(60):
        mid = (lo+hi)/2
        if travel_bound(speed, acceleration, mid, cap) < distance:
            lo = mid
        else:
            hi = mid
    return hi


def _validated_arrays(path, count, expected_times, *, sentinel=False):
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    n = len(expected_times)
    shapes = {'time_s': (n,), 'position': (n,count,3), 'velocity': (n,count,3),
              'rotation_deg': (n,count,3), 'bbox_extent': (count,3),
              'bbox_offset': (count,3), 'bbox_rotation_deg': (count,3),
              'frame_id': (n,), 'simulation_time_s': (n,)}
    for key, shape in shapes.items():
        value = arrays.get(key)
        if np.shape(value) != shape or not np.isfinite(value).all():
            raise HistoryEvidenceError(f'{path}: missing/invalid {key}, expected {shape}')
    if not np.allclose(arrays['time_s'], expected_times, atol=EPS, rtol=0):
        raise HistoryEvidenceError('Common history must end at t0, contain only -2..0 seconds, and use 20 Hz')
    if not np.all(np.diff(arrays['frame_id']) == 1) or not np.allclose(
            np.diff(arrays['simulation_time_s']), DT_S, atol=1e-5, rtol=0):
        raise HistoryEvidenceError('Measured history frame/physics clock is not consecutive 20 Hz')
    if np.any(arrays['bbox_extent'] <= 0):
        raise HistoryEvidenceError('Actor extents must be strictly positive')
    if sentinel:
        if (np.shape(arrays.get('transform_world_vehicle')) != (n, count, 4, 4)
                or not np.isfinite(arrays['transform_world_vehicle']).all()):
            raise HistoryEvidenceError('History requires complete recorded vehicle transforms for body geometry')
        for key, shape in {'control': (n,count,3), 'target_speed': (n,count)}.items():
            if np.shape(arrays.get(key)) != shape or not np.isfinite(arrays[key]).all() or np.any(arrays[key][-1] != 0):
                raise HistoryEvidenceError('History t0 controls must be an unexecuted zero sentinel')
    return arrays


def load_history(spec, history_root):
    """Read only the dedicated history-only output; never future trajectories."""
    if spec.get('spec_sha256') != _spec_digest(spec):
        raise HistoryEvidenceError('Compiled input spec hash is invalid')
    if 'measured_t0_plan' in spec:
        raise HistoryEvidenceError('Already measured/planned spec cannot replace the immutable initial plan')
    scene = spec.get('scene_id')
    if not isinstance(scene, str) or Path(scene).name != scene:
        raise HistoryEvidenceError('Invalid scene identity')
    root = Path(history_root)
    scene_dir = root/scene
    branch = scene_dir/'branches/common_history'
    original_path, state_path, meta_path = scene_dir/'spec.json', branch/'states.npz', branch/'meta.json'
    original, meta = _read(original_path), _read(meta_path)
    if original != spec:
        raise HistoryEvidenceError('Measured history used a different initial plan')
    if (meta.get('status') != 'complete' or meta.get('history_only_complete') is not True
            or meta.get('capture_mode') != HISTORY_MODE or meta.get('t0_index') != 40
            or meta.get('spec_sha256') != spec['spec_sha256']
            or meta.get('vehicle_roles') != spec['vehicle_roles']):
        raise HistoryEvidenceError('History metadata is incomplete or has a different spec/actor order')
    if meta.get('code_sha256') != spec.get('code_sha256'):
        raise HistoryEvidenceError('History was not collected with the compiled spec runtime')
    times = np.arange(41)*DT_S-2.
    states = _validated_arrays(state_path, len(spec['vehicles']), times, sentinel=True)
    if int(meta.get('t0_frame', -1)) != int(states['frame_id'][-1]):
        raise HistoryEvidenceError('History t0 frame disagrees with metadata')
    collisions = meta.get('collisions')
    if not isinstance(collisions, list):
        raise HistoryEvidenceError('History collision record is missing')
    identity = {'states_sha256': file_sha(state_path), 'meta_sha256': file_sha(meta_path),
                'spec_file_sha256': file_sha(original_path), 't0_index': 40,
                'capture_mode': HISTORY_MODE, 'scene_id': scene,
                't0_frame': int(states['frame_id'][-1]),
                't0_simulation_time_s': float(states['simulation_time_s'][-1])}
    manifest = _read(root/'manifest.json')
    rows = [row for row in manifest.get('branch_results', [])
            if row.get('scene') == scene and row.get('branch') == 'common_history']
    if (manifest.get('status') != 'complete' or manifest.get('capture_mode') != HISTORY_MODE
            or manifest.get('history_only') is not True or manifest.get('future_s') != 0.
            or manifest.get('code_sha256') != spec['code_sha256']
            or manifest.get('frozen_history_spec_sha256', {}).get(scene) != spec['spec_sha256']
            or manifest.get('input_spec_file_sha256', {}).get(scene) != identity['spec_file_sha256']
            or len(rows) != 1 or rows[0].get('status') != 'complete'
            or rows[0].get('states_sha256') != identity['states_sha256']
            or rows[0].get('meta_sha256') != identity['meta_sha256']):
        raise HistoryEvidenceError('History manifest/source/completion/file hashes disagree')
    identity['manifest_sha256'] = file_sha(root/'manifest.json')
    pedestrians = None
    if spec.get('pedestrians'):
        path = branch/'pedestrians.npz'
        pedestrians = _validated_arrays(path, len(spec['pedestrians']), times)
        if not np.array_equal(states['frame_id'], pedestrians['frame_id']) or not np.array_equal(
                states['simulation_time_s'], pedestrians['simulation_time_s']):
            raise HistoryEvidenceError('Pedestrian history is not synchronized with vehicles')
        identity['pedestrians_sha256'] = file_sha(path)
    signals = None
    if spec.get('traffic_signal_program'):
        path = branch/'signal_states.json'; signals = _read(path)
        rows = signals.get('samples', [])
        if signals.get('program') != spec['traffic_signal_program'] or len(rows) != 41:
            raise HistoryEvidenceError('Signal history/program is missing or inconsistent')
        for i, row in enumerate(rows):
            if abs(float(row['time_s'])-times[i]) > EPS:
                raise HistoryEvidenceError('Signal history has the wrong relative timestamps')
        identity['signal_states_sha256'] = file_sha(path)
    try:
        geometry = common_history_vehicle_geometry(states, spec['vehicle_roles'])
    except (ValueError, TypeError) as exc:
        raise HistoryEvidenceError('Invalid common-history vehicle geometry: ' + str(exc)) from exc
    identity['vehicle_geometry'] = geometry
    details = {'history': identity, 'vehicle_geometry': geometry,
               'native_collision_events': collisions,
               'native_collision_event_count': len(collisions)}
    if collisions:
        raise HistorySetupError('initialization_or_common_history_contact', details)
    if not geometry['passed']:
        raise HistorySetupError('common_history_vehicle_bbox_interior_overlap', details)
    return states, pedestrians, signals, identity


def _actor_state(actor, states, index, ego_position, ego_yaw, actor_type='vehicle'):
    route = actor['route'] if actor_type == 'vehicle' else actor['route_points']
    route = np.asarray(route, float)
    if route.ndim != 2 or len(route) < 2 or route.shape[1] < 2 or not np.isfinite(route).all():
        raise HistoryEvidenceError(f'{actor["role"]}: invalid declared route')
    position = states['position'][-1,index]; velocity = states['velocity'][-1,index]
    rotation = states['rotation_deg'][-1,index]
    station, offset, tangent = project_to_route(position, route)
    speed = float(np.linalg.norm(velocity[:2])); along_speed = float(velocity[:2] @ tangent)
    yaw = math.radians(float(rotation[2])); box_yaw = yaw + math.radians(float(states['bbox_rotation_deg'][index,2]))
    tangent_yaw = math.atan2(tangent[1], tangent[0]); relative = box_yaw-tangent_yaw
    extent = states['bbox_extent'][index]; box_offset = states['bbox_offset'][index]
    rotation_xy = np.array([[math.cos(yaw),-math.sin(yaw)], [math.sin(yaw),math.cos(yaw)]])
    center_shift = float((rotation_xy @ box_offset[:2]) @ tangent)
    half = float(abs(math.cos(relative))*extent[0] + abs(math.sin(relative))*extent[1])
    forward = np.array([math.cos(ego_yaw), math.sin(ego_yaw)])
    delta = position[:2]-ego_position[:2]
    return {'role': actor['role'], 'actor_type': actor_type,
        'position_world_m': position.tolist(), 'velocity_world_m_s': velocity.tolist(),
        'rotation_deg': rotation.tolist(), 'speed_m_s': speed, 'route_speed_m_s': along_speed,
        'route_s_m': station, 'route_lateral_offset_m': offset,
        'route_heading_deg': math.degrees(tangent_yaw),
        'heading_error_deg': (float(rotation[2])-math.degrees(tangent_yaw)+180)%360-180,
        'route_sha256': digest(actor['route'] if actor_type == 'vehicle' else actor['route_points']),
        'route_remaining_m': float(np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1).sum()-station),
        'body_front_route_s_m': station+center_shift+half,
        'body_rear_route_s_m': station+center_shift-half,
        'body_half_length_on_route_m': half, 'bbox_extent_m': extent.tolist(),
        'bbox_offset_m': box_offset.tolist(), 'bbox_rotation_deg': states['bbox_rotation_deg'][index].tolist(),
        'relative_to_ego_t0_forward_m': float(delta @ forward),
        'relative_to_ego_t0_right_m': float(delta @ np.array([-forward[1],forward[0]])),
        'spawn_speed_m_s': actor.get('initial_speed_mps'),
        'spawn_route_s_m': actor.get('spawn_route_s')}


def _participant(state, declaration, actor, spec):
    front, rear = state['body_front_route_s_m'], state['body_rear_route_s_m']
    entry, leave = float(declaration['entry_s']), float(declaration['exit_s'])
    speed = max(0., state['route_speed_m_s'])
    is_walker = state['actor_type'] == 'pedestrian'
    reactive_walk = is_walker and actor.get('response_policy',{}).get('kind') in ('gap_acceptance_v1','reactive_crossing_v2','reactive_crossing_v3','reactive_crossing_v4')
    if not is_walker:
        acceleration = float(actor.get('npc_policy', {}).get('idm', {}).get('max_acceleration',
                             spec['controller']['idm']['max_acceleration']))
        cap = max(speed, float(actor['desired_speed_mps']))
    else:
        # The walking speed is a possible motion after admission. A feedback
        # walker may wait indefinitely; its earliest start is not a schedule.
        acceleration = 0.
        cap = _number(actor['speed_mps'], 'walker.speed_mps')
    distance = entry-front; cleared = rear > leave
    occupied = front >= entry and rear <= leave
    eta = 0. if occupied else distance/speed if distance > 0 and speed > .05 else None
    clear_eta = max(0., leave-rear)/speed if speed > .05 else 0. if cleared else None
    if is_walker:
        # Earliest permitted departure gives only an optimistic opportunity
        # bound for feedback walkers. It does not assert actual admission.
        start = max(0., float(actor['start_time_s']))
        walk_time = max(0., distance)/cap if cap > EPS else math.inf
        scripted_entry = start+walk_time
        optimistic = (0. if occupied else scripted_entry
                      if not cleared and scripted_entry <= WINDOW_S+EPS else None)
    else:
        optimistic = None if cleared else arrival_bound(max(0., distance), speed, acceleration, cap)
    return {'role': state['role'], 'entry_s_m': entry, 'exit_s_m': leave,
            'front_to_entry_m': distance, 'rear_to_exit_m': leave-rear,
            'occupied_at_t0': occupied, 'already_cleared_at_t0': cleared,
            'constant_speed_entry_s': eta, 'constant_speed_clear_s': clear_eta,
            'optimistic_entry_s': optimistic, 'maximum_acceleration_m_s2': acceleration,
            'speed_cap_m_s': cap, 'can_reach_entry_within_window_under_envelope': optimistic is not None,
            'stationary_before_entry_is_cleared': False,
            'envelope_basis': ('earliest_feedback_admission_and_walk_speed_opportunity_only'
                              if reactive_walk else 'declared_scripted_walk_start_and_speed_not_reactive_behavior') if is_walker
                              else 'declared_fixed_policy_acceleration_and_desired_speed',
            **({'declared_walk_start_s': float(actor['start_time_s']),
                'declared_walk_speed_m_s': cap, 'departure_time_known':not reactive_walk,
                'clearance_time_guaranteed':False,
                'start_time_semantics':'earliest_feedback_decision' if reactive_walk else 'fixed_scripted_start'} if is_walker else {})}


def analyze_t0(spec, states, pedestrians=None, signals=None):
    """Pure analysis using final common-history state; never read a future state.

    Public for offline diagnostics. Production callers must use plan_from_history
    so measurement identity and the history-only capture contract are validated.
    """
    if abs(float(np.asarray(states['time_s'])[-1])) > EPS or np.any(np.asarray(states['time_s']) > EPS):
        raise HistoryEvidenceError('Planning may use common history only, never future observations')
    validate_profiles(spec['action'])
    if (float(spec['duration_s']) != WINDOW_S or float(spec['history_s']) != 2.
            or float(spec['dt_s']) != DT_S):
        raise HistoryEvidenceError('Measured planning v1 requires the frozen 2s history/4s future/20Hz protocol')
    ego_pos = states['position'][-1,0]; ego_yaw = math.radians(float(states['rotation_deg'][-1,0,2]))
    actors = {a['role']: _actor_state(a, states, i, ego_pos, ego_yaw) for i,a in enumerate(spec['vehicles'])}
    ego_velocity = states['velocity'][-1,0,:2]
    forward = np.array([math.cos(ego_yaw),math.sin(ego_yaw)])
    for actor in actors.values():
        delta = np.asarray(actor['position_world_m'][:2])-ego_pos[:2]
        relative_velocity = np.asarray(actor['velocity_world_m_s'][:2])-ego_velocity
        actor['relative_velocity_ego_t0_forward_m_s'] = float(relative_velocity @ forward)
        actor['relative_velocity_ego_t0_right_m_s'] = float(relative_velocity @ np.array([-forward[1],forward[0]]))
        squared = float(relative_velocity @ relative_velocity)
        closest_t = max(0.,float(-delta @ relative_velocity/squared)) if squared > EPS else None
        actor['constant_velocity_closest_approach_s'] = closest_t
        actor['constant_velocity_closest_approach_distance_m'] = (float(np.linalg.norm(delta+closest_t*relative_velocity))
            if closest_t is not None else None)
        actor['closest_approach_is_not_collision_or_response_label'] = True
    definitions = {a['role']: a for a in spec['vehicles']}
    if spec.get('pedestrians'):
        if pedestrians is None:
            raise HistoryEvidenceError('Measured pedestrian state is required for this template')
        for i, actor in enumerate(spec['pedestrians']):
            actors[actor['role']] = _actor_state(actor, pedestrians, i, ego_pos, ego_yaw, 'pedestrian')
            if 'observed_control' in pedestrians:
                try:
                    from .runtime.walker_motion import examine
                except ImportError:
                    from runtime.walker_motion import examine
                actors[actor['role']]['walker_execution'] = examine(pedestrians, i, actor['role'])
            definitions[actor['role']] = actor
    ego = actors['ego']; speed = ego['speed_m_s']; station = ego['route_s_m']
    baseline = spec['action']['nominal_policy']
    failures, warnings = [], []
    def issue(target, code, branch=None, **details):
        target.append({'code': code, **({'branch': branch} if branch else {}), **details})
    conflicts = {}
    for conflict in spec['action'].get('conflicts', []):
        entries = []
        for part in conflict['participants']:
            if part['role'] not in actors:
                raise HistoryEvidenceError(f'Missing measured conflict participant {part["role"]}')
            entries.append(_participant(actors[part['role']], part, definitions[part['role']], spec))
        distance = float(conflict['ego_entry_s'])-ego['body_front_route_s_m']
        conflicts[conflict['id']] = {'ego_front_to_entry_m': distance,
            'ego_rear_to_exit_m': float(conflict['ego_exit_s'])-ego['body_rear_route_s_m'],
            'ego_constant_speed_entry_s': distance/speed if distance > 0 and speed > .05 else 0. if distance <= 0 else None,
            'participants': entries, 'description': conflict.get('zone_definition')}
        if not any(p['can_reach_entry_within_window_under_envelope'] for p in entries):
            issue(failures, 'no_pending_participant_can_reach_declared_conflict_under_optimistic_envelope',
                  conflict_id=conflict['id'])
        elif not any(p['occupied_at_t0'] or (p['constant_speed_entry_s'] is not None and p['constant_speed_entry_s'] <= WINDOW_S)
                     for p in entries):
            issue(warnings, 'participant_opportunity_requires_acceleration_or_policy_release', conflict_id=conflict['id'])
        eta = conflicts[conflict['id']]['ego_constant_speed_entry_s']
        if eta is None or eta > WINDOW_S:
            issue(warnings, 'reference_constant_speed_approach_exceeds_window', conflict_id=conflict['id'], entry_s=eta)
    branches = {}
    for name, profile in spec['action']['profiles'].items():
        kind = profile['kind']; onset = float(profile.get('onset_s', 0.))
        item = {'kind': kind, 'onset_s': onset, 'static_estimate_not_execution_proof': True}
        branches[name] = item
        if onset >= WINDOW_S:
            issue(failures, 'onset_outside_future_window', name)
        if kind == 'lane_change':
            finish = onset+float(profile['duration_s'])+float(profile.get('settling_time_s',0))+float(profile.get('required_final_hold_s',.5))
            item['declared_completion_and_hold_s'] = finish
            if finish > WINDOW_S+EPS:
                issue(failures, 'lane_transition_settling_and_hold_exceed_window', name)
        stop_kind = kind in ('stop_hold','yield_wait') or kind == 'commit_go' and onset > 0
        if stop_kind:
            acceleration = float(profile.get('comfortable_deceleration_m_s2', baseline['comfortable_deceleration_m_s2']))
            response = float(profile.get('response_time_s', baseline.get('response_time_s',DEFAULT_STOP_RESPONSE_TIME_S)))
            margin = float(profile.get('stop_margin_m',0.))
            coast = onset if kind in ('stop_hold','yield_wait') else 0.
            available = float(profile['stop_route_s'])-station
            required = speed*(coast+response)+speed*speed/(2*acceleration)+margin
            earliest = coast+response+speed/acceleration if speed > .15 else coast
            stopping = {'available_distance_m': available, 'required_response_braking_margin_m': required,
                'ideal_braking_distance_m': speed*speed/(2*acceleration), 'response_time_s': response,
                'optimistic_stop_time_s': earliest, 'stop_deadline_s': profile.get('stop_deadline_s')}
            item['stop_budget'] = stopping
            strict_wait = profile.get('release_condition')=='specified_actors_cleared_and_entry_clear_v1'
            if strict_wait or profile.get('hold_policy')=='hold_until_window_end_v1':
                if __package__:
                    from .interaction_geometry import waiting_position_check
                else:
                    from interaction_geometry import waiting_position_check
                waiting_roles=profile.get('wait_for_roles',list(dict.fromkeys(
                    part['role'] for conflict in spec['action']['conflicts'] for part in conflict['participants'])))
                clearance=waiting_position_check(spec,float(profile['stop_route_s']),waiting_roles,
                                                measured_actors=actors)
                item['waiting_body_clearance']=clearance
                if not clearance['clear']:
                    issue(failures,'waiting_ego_body_obstructs_target_swept_route',name,
                          target_roles=clearance['blocked_target_roles'])
            if available+EPS < required:
                issue(failures, 'measured_t0_stop_distance_budget_insufficient', name, available_m=available, required_m=required)
            if 'stop_deadline_s' in profile:
                deadline = float(profile['stop_deadline_s'])
                stopping['deadline_target_zero_s'] = deadline-response
                if deadline > WINDOW_S+EPS or earliest > deadline+EPS:
                    issue(failures, 'measured_t0_stop_deadline_infeasible', name, earliest_stop_s=earliest, deadline_s=deadline)
            if kind == 'yield_wait':
                wait = float(profile.get('min_wait_s',.25)); deadline = profile.get('stop_deadline_s')
                release_limit = min(WINDOW_S, float(profile['timeout_s']))
                optimistic_release = earliest+wait
                budget = {'optimistic_stop_then_minimum_wait_release_s': optimistic_release,
                    'optimistic_recovery_observation_end_s': optimistic_release+RECOVERY_OBSERVATION_S,
                    'recovery_interval_s': RECOVERY_OBSERVATION_S,
                    'deadline_based_recovery_observation_end_s': float(deadline)+wait+RECOVERY_OBSERVATION_S if deadline is not None else None,
                    'release_event_is_not_guaranteed': True,
                    'timeout_waiting_remains_valid_execution': True,
                    'waiting_rule': profile.get('release_condition', 'current_conflict_gap_not_actor_passage'),
                    'waiting_roles': profile.get('wait_for_roles', [part['role'] for conflict in spec['action']['conflicts']
                        if conflict['id']==profile['conflict_id'] for part in conflict['participants']])}
                if strict_wait:
                    budget.update(release_depends_on_entire_target_body_clear=True,
                        entry_clearance_rechecked_after_release=True,
                        may_release_while_approaching_without_full_stop=True,
                        optimistic_stop_then_minimum_wait_is_not_release_prediction=True)
                wait_conflict = next(c for c in spec['action']['conflicts'] if c['id']==profile['conflict_id'])
                remaining_distance = max(0., float(wait_conflict['ego_entry_s'])
                    -float(profile['stop_route_s'])-(ego['body_front_route_s_m']-station))
                recover_acceleration = float(baseline['idm']['max_acceleration'])
                recover_cap = float(profile.get('resume_speed_m_s',baseline['desired_speed_m_s']))
                budget.update(recovery_distance_from_latest_stop_to_conflict_m=remaining_distance,
                    optimistic_recovery_travel_m=travel_bound(0.,recover_acceleration,
                        max(0.,WINDOW_S-optimistic_release),recover_cap),
                    recovery_envelope_basis='optimistic stop/wait release plus declared nominal acceleration; not actual passage')
                if deadline is not None:
                    budget['deadline_based_recovery_travel_m'] = travel_bound(0.,recover_acceleration,
                        max(0.,WINDOW_S-float(deadline)-wait),recover_cap)
                item['wait_and_recovery_budget'] = budget
                if optimistic_release >= release_limit-EPS or optimistic_release+RECOVERY_OBSERVATION_S > WINDOW_S+EPS:
                    issue(warnings if strict_wait else failures,
                          'strict_wait_may_remain_until_timeout' if strict_wait else 'no_optimistic_stop_wait_recovery_budget',
                          name, **budget)
                elif budget['deadline_based_recovery_observation_end_s'] is not None and budget['deadline_based_recovery_observation_end_s'] > WINDOW_S+EPS:
                    issue(warnings, 'stop_deadline_does_not_reserve_recovery_interval', name, **budget)
        acceleration = max(0., float(profile.get('acceleration_m_s2',0.)))
        cap = speed+float(profile.get('speed_increase_cap_m_s',0.))
        if kind == 'nominal':
            acceleration = float(baseline['idm']['max_acceleration']); cap = max(speed,float(baseline['desired_speed_m_s']))
        if kind in ('maintain','longitudinal','commit_go','nominal','lane_change'):
            # A delayed commit can only lose speed during its pre-onset stop
            # approach; assuming it coasts is deliberately optimistic.
            distance = speed*onset+travel_bound(speed,acceleration,max(0.,WINDOW_S-onset),cap)
            item['optimistic_forward_travel_m'] = distance
            item['optimistic_entry_reachable'] = {key: value['ego_front_to_entry_m'] <= distance+EPS for key,value in conflicts.items()}
            if kind == 'longitudinal' and float(profile['acceleration_m_s2']) < 0 and speed <= float(profile.get('min_target_speed_m_s',0.))+EPS:
                issue(warnings,'deceleration_at_or_below_target_floor',name)
        if ego['route_remaining_m'] < item.get('optimistic_forward_travel_m',0.) + ego['body_half_length_on_route_m']:
            issue(warnings,'optimistic_envelope_reaches_declared_route_end',name)
    for key in conflicts:
        advancing = [value for name,value in branches.items()
                     if value['kind'] in ('maintain','commit_go','nominal')
                     or value['kind']=='longitudinal' and spec['action']['profiles'][name].get('acceleration_m_s2',0)>0]
        if not advancing or not any(value.get('optimistic_entry_reachable',{}).get(key,False) for value in advancing):
            issue(failures,'no_ego_entry_condition_reaches_conflict_under_optimistic_envelope',conflict_id=key)
    static = spec.get('geometry',{}).get('traffic_design',{}).get('authored_initial_conditions',
             spec.get('geometry',{}).get('traffic_design',{}).get('interaction_opportunity',{}))
    drift = {'static_estimated_route_s_m':static.get('estimated_t0_route_s'),
             'static_estimated_speed_m_s':static.get('estimated_t0_speed_m_s')}
    if drift['static_estimated_route_s_m'] is not None:
        drift['measured_minus_estimated_route_s_m'] = station-float(drift['static_estimated_route_s_m'])
    if drift['static_estimated_speed_m_s'] is not None:
        drift['measured_minus_estimated_speed_m_s'] = speed-float(drift['static_estimated_speed_m_s'])
    distinct_opportunity = None
    if 'action_separation_contract' in spec:
        if __package__:
            from .action_opportunity import analyze_opportunity
        else:
            from action_opportunity import analyze_opportunity
        distinct_opportunity = analyze_opportunity(spec, actors, conflicts)
        failures.extend(distinct_opportunity['failure_reasons'])
    if __package__:
        from .encounter_calibration import analyze as analyze_encounter
    else:
        from encounter_calibration import analyze as analyze_encounter
    encounter = analyze_encounter(spec,actors,conflicts,branches)
    if not encounter['passed']:
        issue(failures,'measured_encounter_targets_not_met',rows=encounter['rows'])
    return {'encounter_calibration':encounter,'schema': EVIDENCE_SCHEMA, 'policy_version': POLICY_VERSION, 'passed': not failures,
        'future_window_s': WINDOW_S, 'actors': actors, 'conflicts': conflicts, 'branches': branches,
        'static_design_drift': drift, 'signal_t0': signals['samples'][-1] if signals else None,
        'failure_reasons': failures, 'warnings': warnings,
        'distinct_action_response_opportunity': distinct_opportunity,
        'repair_scope': 'Repair the shared pre-history setup or explicitly authored future plan; rerun common history. Never repair branch-specific initial states/NPC policies.',
        'limitations': ['Kinematic estimates are not simulated execution or NPC outcome predictions.',
            'No future reaction magnitude, passage success or collision outcome is used to select scenes.',
            'Declared conflict intervals may be imperfect; swept-body interaction geometry and native execution remain separate checks.',
            'Correctly waiting to timeout remains a valid future outcome, even when a planning recovery opportunity existed.']}


def _bounds(spec, timing_bounds):
    result = {}
    for name, bound in (timing_bounds or {}).items():
        if name not in spec['action']['profiles'] or not isinstance(bound, dict) or set(bound) != {'onset_s'}:
            raise ValueError('Only explicit action-profile onset_s bounds are supported')
        pair = bound['onset_s']
        if not isinstance(pair, (tuple,list)) or len(pair) != 2:
            raise ValueError('onset_s bounds require [minimum, maximum]')
        lo, hi = [_number(x, 'onset bound') for x in pair]
        original = float(spec['action']['profiles'][name].get('onset_s',0.))
        if lo > hi or hi >= WINDOW_S or not lo-EPS <= original <= hi+EPS:
            raise ValueError('Bounds must contain the original onset and remain in the future window')
        if spec['action']['profiles'][name]['kind'] in ('maintain','nominal') and (lo != 0 or hi != 0):
            raise ValueError('Reference/common baseline onset is immutable')
        if any(abs(x/DT_S-round(x/DT_S)) > 1e-7 for x in (lo,hi)):
            raise ValueError('Onset bounds must lie on the 20 Hz clock')
        result[name] = (lo,hi)
    return result


def assert_shared_setup_unchanged(initial, planned, timing_bounds=None):
    """Allow new proof + spec hash and explicitly bounded ego onset changes only."""
    bounds = _bounds(initial,timing_bounds)
    left, right = copy.deepcopy(initial), copy.deepcopy(planned)
    for value in (left,right):
        value.pop('spec_sha256',None); value.pop('measured_t0_plan',None)
    if set(left['action']['profiles']) != set(right['action']['profiles']):
        raise ValueError('Changing the branch set is forbidden')
    for name, profile in left['action']['profiles'].items():
        other = right['action']['profiles'][name]
        if profile.get('onset_s',0.) != other.get('onset_s',0.):
            value = _number(other.get('onset_s'),f'{name}.onset_s')
            if name not in bounds or not bounds[name][0]-EPS <= value <= bounds[name][1]+EPS:
                raise ValueError('Unapproved onset change')
            if 'onset_s' in profile:
                other['onset_s'] = profile['onset_s']
            else:
                other.pop('onset_s',None)
    if left != right:
        raise ValueError('Measured planning changed shared initialization/history/NPC policy or another frozen contract')
    return True


def plan_from_history(spec, history_root, *, timing_bounds=None):
    """Return an in-memory planned spec/evidence; never modify or overwrite inputs.

    ``status`` is passed/failed. ``planned_spec`` is None on failure. Only an
    explicit ``{branch: {'onset_s': [lo, hi]}}`` permits timing adjustment; no
    deadline, timeout, acceleration, state or policy is silently changed.
    """
    bounds = _bounds(spec,timing_bounds)
    try:
        states,pedestrians,signals,history = load_history(spec,history_root)
    except HistorySetupError as exc:
        evidence = {'schema':EVIDENCE_SCHEMA,'policy_version':POLICY_VERSION,'passed':False,
            'original_history_spec_sha256':spec['spec_sha256'],
            'failure_reasons':[{'code':str(exc),'scope':'shared_history_setup','details':exc.details}],
            'repair_scope':'Repair the shared scene setup and rerun all common history; never repair one branch.'}
        return {'status':'failed','passed':False,'planned_spec':None,'evidence':evidence,
                'failure_reasons':evidence['failure_reasons']}
    except HistoryEvidenceError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HistoryEvidenceError(f'Cannot read complete measured history: {exc}') from exc
    proposed = copy.deepcopy(spec)
    evidence = analyze_t0(proposed,states,pedestrians,signals)
    original_evidence = copy.deepcopy(evidence)
    adjustments = []
    # Each failed branch is repaired independently on the declared 20 Hz grid.
    # Group opportunity failures are never solved by changing the shared setup.
    for name,(lo,hi) in bounds.items():
        failures = [row for row in evidence['failure_reasons'] if row.get('branch') == name]
        if not failures:
            continue
        original = float(proposed['action']['profiles'][name].get('onset_s',0.))
        candidates = sorted((round(t*DT_S,8) for t in range(round(lo/DT_S),round(hi/DT_S)+1)),
                            key=lambda t:(abs(t-original),t))
        for onset in candidates:
            if abs(onset-original) <= EPS:
                continue
            attempt = copy.deepcopy(proposed); attempt['action']['profiles'][name]['onset_s']=onset
            try:
                analyzed = analyze_t0(attempt,states,pedestrians,signals)
            except ValueError:
                continue  # e.g. making two action definitions identical
            if any(row.get('branch') == name for row in analyzed['failure_reasons']):
                continue
            proposed,evidence=attempt,analyzed
            adjustments.append({'branch':name,'field':'onset_s','before':original,'after':onset,
                'explicit_bounds_s':[lo,hi],'resolved_failure_reasons':failures})
            break
    evidence.update(history=history,original_history_spec_sha256=spec['spec_sha256'],
        explicit_timing_bounds=copy.deepcopy(timing_bounds or {}), adjustments=adjustments,
        original_profile_failure_reasons=original_evidence['failure_reasons'])
    evidence['plan_mode'] = 'adjusted_explicit_onsets' if adjustments else 'checked_original_profiles'
    if not evidence['passed']:
        return {'status':'failed','passed':False,'planned_spec':None,'evidence':evidence,
                'failure_reasons':evidence['failure_reasons']}
    proposed['measured_t0_plan'] = {'schema':PLAN_SCHEMA,'policy_version':POLICY_VERSION,
        'original_history_spec_sha256':spec['spec_sha256'],'history':history,
        'evidence_sha256':digest(evidence),'plan_mode':evidence['plan_mode'],
        'explicit_timing_bounds':copy.deepcopy(timing_bounds or {}),'adjustments':adjustments,
        'actual_t0_checked':True,'future_execution_verified':False}
    proposed['spec_sha256'] = _spec_digest(proposed)
    assert_shared_setup_unchanged(spec,proposed,timing_bounds)
    return {'status':'passed','passed':True,'planned_spec':proposed,'evidence':evidence,'failure_reasons':[]}


def summarize_observed_window(spec, physics_scene, *, execution_report=None):
    """Describe actual future timing after physics QA, without any new gate.

    This function is deliberately separate from plan_from_history. Its future
    observations must never be fed back into measured-t0 planning or used to
    select scenes by NPC response. Existing execution QA supplies validity;
    this summary merely makes approach/stop/wait/recovery censoring explicit.
    """
    scene = Path(physics_scene)
    if _read(scene/'spec.json') != spec or spec.get('spec_sha256') != _spec_digest(spec):
        raise HistoryEvidenceError('Physics-window summary has a different or invalid frozen spec')
    names = list(spec.get('branches', list(spec['action']['profiles'])+['maintain_repeat']))
    if len(names) != 6 or set(names) != set(spec['action']['profiles']) | {'maintain_repeat'}:
        raise HistoryEvidenceError('Physics-window summary requires all five model branches and the QA repeat')
    audited = None
    if execution_report is not None:
        audited = execution_report.get('scenes', {}).get(spec['scene_id'])
        if audited is None or audited.get('spec_file_sha256') != file_sha(scene/'spec.json'):
            raise HistoryEvidenceError('Existing execution audit does not bind this physics spec')
    rows = {}; sources = {}
    threshold = float(spec.get('execution_contract', {}).get('stopped_speed_m_s', .25))
    for name in names:
        folder = scene/'branches'/name
        paths = {key:folder/key for key in ('states.npz','meta.json','controller_diagnostics.json')}
        if (folder/'npc_merge_state.json').is_file():
            paths['npc_merge_state.json']=folder/'npc_merge_state.json'
        sources[name] = {key:file_sha(path) for key,path in paths.items()}
        if audited is not None and audited.get('source_files', {}).get(name) != sources[name]:
            raise HistoryEvidenceError('Existing execution audit binds different recorded branch files')
        meta = _read(paths['meta.json'])
        if (meta.get('status') != 'complete' or meta.get('capture_mode') != 'physics_only_preflight_not_delivery'
                or meta.get('branch') != name or meta.get('spec_sha256') != spec['spec_sha256']
                or meta.get('code_sha256') != spec['code_sha256'] or meta.get('vehicle_roles') != spec['vehicle_roles']):
            raise HistoryEvidenceError('Incomplete or mismatched physics-window metadata')
        states = _validated_arrays(paths['states.npz'],len(spec['vehicles']),np.arange(121)*DT_S-2.)
        diagnostics = _read(paths['controller_diagnostics.json'])
        if (not isinstance(diagnostics,list) or len(diagnostics) != 121
                or any(not isinstance(row,list) or len(row) != len(spec['vehicles']) for row in diagnostics)):
            raise HistoryEvidenceError('Physics-window summary requires every recorded controller diagnostic')
        ego_rows = [row[0] for row in diagnostics]
        future = states['time_s'] >= -EPS
        times = states['time_s']; speed = np.linalg.norm(states['velocity'][:,0,:2],axis=1)
        profile = spec['action']['profiles']['maintain' if name=='maintain_repeat' else name]
        def decision_time(field):
            values = {float(row[field]) for row,t in zip(ego_rows,times) if t >= -EPS and row.get(field) is not None}
            if len(values) > 1 or any(not math.isfinite(value) or value < -EPS or value > WINDOW_S+EPS for value in values):
                raise HistoryEvidenceError('Controller decision timestamp is not a single valid latched event')
            return next(iter(values),None)
        stopped, released = decision_time('stopped_at_s'), decision_time('released_at_s')
        timed_out = any(bool(row.get('timed_out')) for row,t in zip(ego_rows,times) if t >= -EPS)
        collision_events = meta.get('collisions')
        if not isinstance(collision_events,list):
            raise HistoryEvidenceError('Physics-window collision observations are missing')
        contact = min((float(event['time_s']) for event in collision_events
                       if event.get('role')=='ego' or event.get('other_role')=='ego'),default=None)
        available = max(0.,WINDOW_S-released) if released is not None else None
        before_contact = max(0.,min(WINDOW_S,contact)-released) if released is not None and contact is not None else available
        stationary = times[future & (speed <= threshold)]
        # Match the controller's 0.15 m/s stop latch when describing movement;
        # keep the execution audit's separate stationary threshold explicit.
        resumed = times[future & (times > (released if released is not None else WINDOW_S)+EPS) & (speed > .15)]
        physical_recovery = max(0.,WINDOW_S-float(resumed[0])) if len(resumed) else None
        qa = audited['branches'][name] if audited is not None else None
        timeout_valid = None
        if profile['kind']=='yield_wait' and timed_out and qa is not None:
            # A contact-censored endpoint cannot establish a complete timeout
            # hold, even if all precontact controls passed the existing gate.
            timeout_valid = None if qa.get('quality_status')=='censored' else qa.get('execution_pass') is True
        rows[name] = {'kind':profile['kind'],'qa_only':name=='maintain_repeat',
            'measured_t0_speed_m_s':float(speed[40]), 'measured_t0_position_world_m':states['position'][40,0].tolist(),
            'declared_stop_deadline_s':profile.get('stop_deadline_s'),
            'declared_min_wait_s':profile.get('min_wait_s'),'declared_timeout_s':profile.get('timeout_s'),
            'stopped_at_s':stopped,'released_at_s':released,
            'observed_first_stationary_time_s':float(stationary[0]) if len(stationary) else None,
            'observed_first_moving_after_release_s':float(resumed[0]) if len(resumed) else None,
            'observed_moving_threshold_m_s':.15,'execution_stationary_threshold_m_s':threshold,
            'remaining_future_after_observed_motion_s':physical_recovery,
            'at_least_one_2hz_interval_after_observed_motion':physical_recovery >= RECOVERY_OBSERVATION_S-EPS if physical_recovery is not None else False,
            'remaining_future_after_release_s':available,
            'at_least_one_2hz_recovery_interval':available >= RECOVERY_OBSERVATION_S-EPS if available is not None else False,
            'precontact_recovery_available_s':before_contact,
            'precontact_recovery_interval_available':before_contact >= RECOVERY_OBSERVATION_S-EPS if before_contact is not None else False,
            'timed_out':timed_out,'waiting_timeout_valid':timeout_valid,
            'waiting_timeout_validity_basis':'existing execution audit; null when not a timeout, not supplied, or contact-censored',
            'execution_pass':qa.get('execution_pass') if qa else None,
            'observed_outcome':qa.get('observed_outcome') if qa else None,
            'first_ego_contact_s':contact,'future_censored_by_contact':contact is not None,
            'terminal_phase':ego_rows[-1].get('phase'),'terminal_speed_m_s':float(speed[-1]),
            'no_release_does_not_imply_execution_failure':True}
    return {'schema':'observed-four-second-window-v1','scene_id':spec['scene_id'],
        'spec_sha256':spec['spec_sha256'],'source_files':sources,'branches':rows,
        'future_window_s':WINDOW_S,'recovery_observation_interval_s':RECOVERY_OBSERVATION_S,
        'execution_report_sha256':digest(execution_report) if execution_report is not None else None,
        'descriptive_only':True,'new_acceptance_gate':False,'used_to_replan_initial_conditions':False,
        'limitations':['Actual future observations; not inputs to measured-t0 planning.',
            'No required NPC passage, reaction magnitude, recovery success, or collision-free outcome.',
            'Insufficient post-release observation is reported, not relabeled as failed strategy execution.']}
