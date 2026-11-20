#!/usr/bin/env python3
"""Audit frozen policies, their actual execution, and observed paired outcomes.

No simulator is required.  Recorded state is used to re-evaluate policies, while
actuator readback and measured motion are independent checks.  Outcomes (including
waiting and contact) are never treated as requested successes. NPC effect size
is descriptive and is not a production acceptance criterion.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

if __package__:
    from .controllers_v2 import (ScenarioController, branch_names, lane_change_route,
        project_to_route, steer_to_route, _actual_corners)
    from .audit_lane_change import trajectory_audit, lane_direction, project
else:
    from controllers_v2 import (ScenarioController, branch_names, lane_change_route,
        project_to_route, steer_to_route, _actual_corners)
    from audit_lane_change import trajectory_audit, lane_direction, project


DEFAULT_EXECUTION_CONTRACT = {
    'command_tolerance': 1e-5,
    'target_tolerance_m_s': 1e-6,
    'speed_tracking_tolerance_m_s': 2.,
    'speed_tracking_grace_s': .75,
    'tracking_violation_duration_s': .75,
    'minimum_speed_change_fraction': .5,
    'minimum_speed_change_m_s': .5,
    'min_precontact_motion_s': .25,
    'stopped_speed_m_s': .25,
    'stop_position_overshoot_m': .5,
    'lane_center_tolerance_m': .5,
    'lane_heading_tolerance_deg': 10.,
    'lane_hold_s': .5,
    'route_lateral_tolerance_m': 1.,
}
DEFAULT_RESPONSE_CONTRACT = {
    'unrelated_roles': [],
    'unrelated_position_tolerance_m': 0.,
    'unrelated_velocity_tolerance_m_s': 0.,
    'unrelated_heading_tolerance_deg': 0.,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as record:
        return {key: record[key] for key in record.files}


def contracts(spec):
    ec = {**DEFAULT_EXECUTION_CONTRACT, **spec.get('execution_contract', {})}
    rc = {**DEFAULT_RESPONSE_CONTRACT, **spec.get('response_contract', {})}
    if set(ec) != set(DEFAULT_EXECUTION_CONTRACT):
        raise ValueError('Unknown execution contract field')
    for key, value in ec.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value < 0:
            raise ValueError(f'Invalid execution contract: {key}')
    if ec['minimum_speed_change_fraction'] > 1 or ec['lane_hold_s'] <= 0:
        raise ValueError('Speed-change fraction must be <=1 and lane hold must be positive')
    if set(rc) != set(DEFAULT_RESPONSE_CONTRACT):
        raise ValueError('Unknown response contract field')
    roles = rc['unrelated_roles']
    if not isinstance(roles, list) or len(set(roles)) != len(roles) or not set(roles).issubset(spec['vehicle_roles'][1:]):
        raise ValueError('Unrelated roles must name distinct configured NPC vehicles')
    for key, value in rc.items():
        if key != 'unrelated_roles' and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0):
            raise ValueError(f'Invalid response contract: {key}')
    return ec, rc


def longest_violation(times, flags):
    """Length of contiguous flagged state intervals; all intervals are recorded."""
    start = None
    longest = 0.
    for index, flag in enumerate(flags):
        if flag:
            start = times[index] if start is None else start
            longest = max(longest, float(times[index]-start))
        else:
            start = None
    return longest


def _observed(state, index, spec, pedestrians=None, signals=None):
    result = {key: state[key][index] for key in ('position', 'velocity', 'rotation_deg')}
    for key in ('bbox_offset','bbox_rotation_deg'):
        if key in state: result[key]=state[key]
    if spec.get('pedestrians'):
        if pedestrians is None:
            raise ValueError('Policy audit requires recorded pedestrian observations')
        result['pedestrians'] = {key: pedestrians[key][index] for key in ('position', 'velocity', 'rotation_deg')}
        result['pedestrians'].update(roles=[p['role'] for p in spec['pedestrians']], bbox_extent=pedestrians['bbox_extent'])
        for key in ('bbox_offset','bbox_rotation_deg'):
            if key in pedestrians: result['pedestrians'][key]=pedestrians[key]
    if spec.get('traffic_signal_program'):
        if signals is None or len(signals['samples']) != len(state['time_s']):
            raise ValueError('Policy audit requires synchronized signal readback')
        row = signals['samples'][index]
        if abs(float(row['time_s'])-float(state['time_s'][index])) > 1e-8:
            raise ValueError('Signal readback timestamp mismatch')
        result['signal_states'] = {}
        for group in spec['traffic_signal_program']['groups']:
            actual = [row['actual_states'][key] for key in group['landmark_ids']]
            if not actual or len(set(actual)) != 1:
                raise ValueError('Signal group actual lamps disagree')
            result['signal_states'][group['id']] = actual[0]
    return result


def _actuator(effort):
    return (min(effort, .9), 0.) if effort >= 0 else (0., min(-effort*.7, 1.))


def replay_policy(state, spec, branch, diagnostics, pedestrians=None, signals=None, merge_evidence=None):
    """Replay targets exactly and PI commands with bounded hidden warmup integral.

    Warmup precedes the saved history. Its initial PI integral is unknown but is
    provably in [-1.5,1.5]. Propagating that interval is honest; resetting it to
    zero and demanding exact command equality would reject valid recordings.
    Steering, hold clamps, target speeds and actuator application are observable.
    """
    ec, _ = contracts(spec)
    times = state['time_s']; n = len(spec['vehicles'])
    if spec['action'].get('profile_version') != 2:
        raise ValueError('Execution gate requires explicit version2 decision profiles')
    if not isinstance(diagnostics, list) or len(diagnostics) != len(times):
        raise ValueError('Missing per-state controller diagnostics')
    controller = ScenarioController(spec)
    if __package__:
        from . import npc_merge
    else:
        import npc_merge
    controller.merge_memory = npc_merge.replay_warmup(spec,merge_evidence)
    expected = np.zeros_like(state['target_speed'])
    decision_rows = []
    integral_low, integral_high = np.full(n, -1.5), np.full(n, 1.5)
    command_errors, diagnostic_errors = [], []
    initial = float(spec['vehicles'][0]['initial_speed_mps'])
    for k, t in enumerate(times):
        if abs(t) <= 1e-8:
            initial = float(np.linalg.norm(state['velocity'][k, 0, :2]))
        observed = _observed(state, k, spec, pedestrians, signals)
        active = 'maintain' if t < -1e-8 or branch == 'maintain_repeat' else branch
        _, targets, computed = controller.controls(observed, active, float(t), state['bbox_extent'], initial)
        expected[k] = targets
        decision_rows.append(computed)
        recorded = diagnostics[k]
        if not isinstance(recorded, list) or len(recorded) != n:
            raise ValueError(f'Missing actor diagnostics at {t}s')
        for actor, (actual, wanted) in enumerate(zip(recorded, computed)):
            if actual.get('dynamic_merge') != wanted.get('dynamic_merge'):
                diagnostic_errors.append({'time_s':float(t),'role':spec['vehicle_roles'][actor],
                    'field':'dynamic_merge','reason':'causal merge decision or memory differs'})
            if wanted.get('narrow_meeting',{}).get('policy_version') == 'narrow-current-occupancy-v2':
                if actual.get('narrow_meeting') != wanted['narrow_meeting']:
                    diagnostic_errors.append({'time_s':float(t),'role':spec['vehicle_roles'][actor],
                                              'field':'narrow_meeting','reason':'causal narrow decision or memory differs'})
            for key in ('role', 'kind', 'phase', 'reason', 'stopped_at_s', 'released_at_s', 'timed_out', 'force_brake_hold'):
                if actual.get(key) != wanted.get(key):
                    diagnostic_errors.append({'time_s': float(t), 'role': spec['vehicle_roles'][actor], 'field': key,
                                              'recorded': actual.get(key), 'replayed': wanted.get(key)})
            if wanted.get('yield_semantics') == 'specified_actors_cleared_and_entry_clear_v1':
                for key in ('yield_semantics','wait_for_roles','pending_wait_roles',
                            'all_wait_targets_cleared','entry_clear','entry_blocking_roles',
                            'entry_permitted_now','release_reason','rolling_release','reblocked','conflict'):
                    if actual.get(key) != wanted.get(key):
                        diagnostic_errors.append({'time_s':float(t),'role':spec['vehicle_roles'][actor],
                            'field':key,'recorded':actual.get(key),'replayed':wanted.get(key)})
            if wanted.get('hold_policy') == 'hold_until_window_end_v1' and actual.get('hold_policy') != wanted['hold_policy']:
                diagnostic_errors.append({'time_s':float(t),'role':spec['vehicle_roles'][actor],
                    'field':'hold_policy','recorded':actual.get('hold_policy'),'replayed':wanted['hold_policy']})
        if k == len(times)-1:
            continue  # final controls are an unexecuted zero sentinel
        speed = np.linalg.norm(state['velocity'][k, :, :2], axis=1)
        error = targets-speed
        integral_low = np.clip(integral_low+error*.05, -1.5, 1.5)
        integral_high = np.clip(integral_high+error*.05, -1.5, 1.5)
        for actor in range(n):
            lo = _actuator(.16+.42*error[actor]+.1*integral_low[actor])
            hi = _actuator(.16+.42*error[actor]+.1*integral_high[actor])
            low, high = np.minimum(lo, hi), np.maximum(lo, hi)
            hold = actor == 0 and computed[0].get('force_brake_hold', False)
            if hold or targets[actor] <= .01:
                terminal_brake = .4
                if actor == 0 and t >= 0.:
                    profile = spec['action']['profiles'][active]
                    if t >= profile.get('onset_s', 0.):
                        terminal_brake = float(profile.get('terminal_brake_min', .4))
                low[0] = high[0] = 0.
                low[1], high[1] = max(low[1], terminal_brake), max(high[1], terminal_brake)
                integral_low[actor] = integral_high[actor] = 0.
            route = controller.active_routes[actor]
            steering = steer_to_route(observed['position'][actor], np.deg2rad(observed['rotation_deg'][actor, 2]),
                                      speed[actor], route, 0.)
            control = state['control'][k, actor]
            tol = ec['command_tolerance']
            if (np.any(control[[0, 2]] < low-tol) or np.any(control[[0, 2]] > high+tol)
                    or abs(control[1]-steering) > tol):
                command_errors.append({'time_s': float(t), 'role': spec['vehicle_roles'][actor],
                    'recorded_throttle_steer_brake': control.tolist(),
                    'allowed_throttle_brake_min': low.tolist(), 'allowed_throttle_brake_max': high.tolist(),
                    'expected_steer': float(steering)})
    if controller.merge_memory and controller.merge_t0 != merge_evidence.get('t0_state'):
        diagnostic_errors.append({'field':'dynamic_merge_t0','reason':'internal state at t0 differs'})
    target_error = np.abs(expected-state['target_speed'])
    readback_error = np.abs(state['observed_control'][1:]-state['control'][:-1])
    return {'target_max_error_m_s': float(target_error.max()),
        'target_error_by_role_m_s': dict(zip(spec['vehicle_roles'], target_error.max(axis=0).tolist())),
        'actuator_readback_max_error': float(readback_error.max()),
        'actuator_readback_error_by_role': dict(zip(spec['vehicle_roles'], readback_error.max(axis=(0, 2)).tolist())),
        'command_error_count': len(command_errors), 'command_errors': command_errors[:30],
        'diagnostic_error_count': len(diagnostic_errors), 'diagnostic_errors': diagnostic_errors[:30],
        'passed': bool(target_error.max() <= ec['target_tolerance_m_s'] and
            readback_error.max() <= ec['command_tolerance'] and not command_errors and not diagnostic_errors)}, decision_rows


def _role_contact(meta, role):
    events = [e for e in meta['collisions'] if e.get('role') == role or e.get('other_role') == role]
    # Static actors may have no other_role, but the attached sensor role remains.
    return min((float(e['time_s']) for e in events), default=None)


def _tracking(times, speed, target, mask, ec):
    errors = np.zeros(len(times))
    for k in np.flatnonzero(mask):
        recent = (times >= times[k]-ec['speed_tracking_grace_s']-1e-8) & (times <= times[k]+1e-8)
        errors[k] = max(0., float(target[recent].min()-speed[k]), float(speed[k]-target[recent].max()))
    violations = mask & (errors > ec['speed_tracking_tolerance_m_s'])
    duration = longest_violation(times, violations)
    return {'max_envelope_error_m_s': float(errors.max()), 'longest_tracking_violation_s': duration,
            'passed': bool(duration < ec['tracking_violation_duration_s']-1e-8),
            'definition': 'Observed speed must track the range of requested targets over the preceding grace window; sustained excess is a control failure.'}


def waiting_footprint_audit(state, spec, profile, ego_rows, precontact_mask, stopped_speed_m_s):
    """Stationary, not-yet-released ego must remain before its conflict entry.

    This is a conservative projection of the actual eight-corner ego OBB onto
    the declared route. It checks the specified conflict, not unknown static
    obstacles, and does not require any participant to cross. A later stop
    after an authorized release is excluded: new traffic can require braking
    inside a conflict which the ego had already been permitted to enter.
    """
    conflicts=spec['action'].get('conflicts',[])
    identity=profile.get('conflict_id')
    selected=[x for x in conflicts if x['id']==identity] if identity is not None else conflicts
    report={'policy':'stationary_unreleased_actual_body_before_declared_conflict_v1',
        'scope':'precontact stationary unreleased ego; declared route-entry geometry only',
        'unknown_obstacles_assessed':False, 'stopped_speed_m_s':float(stopped_speed_m_s),
        'conflict_id':None,'checked_frames':0,'minimum_front_clearance_m':None,
        'frames':[],'violations':[],'passed':False}
    if len(selected)!=1:
        report['failure_reason']='missing_or_ambiguous_declared_waiting_conflict'
        return report
    conflict=selected[0];entry=float(conflict['ego_entry_s'])
    report.update(conflict_id=conflict['id'],ego_entry_s=entry,ego_exit_s=float(conflict['ego_exit_s']))
    speed=np.linalg.norm(state['velocity'][:,0,:2],axis=1)
    route=spec['vehicles'][0]['route']
    for k,t in enumerate(state['time_s']):
        if not precontact_mask[k] or speed[k]>stopped_speed_m_s or ego_rows[k].get('released_at_s') is not None:
            continue
        observed={key:state[key][k] for key in ('position','rotation_deg')}
        for key in ('bbox_offset','bbox_rotation_deg'):
            if key in state: observed[key]=state[key]
        corners=_actual_corners(observed,state['bbox_extent'],0)
        stations=[project_to_route(p,route)[0] for p in corners]
        low,high=float(min(stations)),float(max(stations))
        clearance=entry-high
        row={'time_s':float(t),'body_route_min_s':low,'body_route_max_s':high,
            'front_clearance_to_conflict_m':clearance,'entire_body_before_entry':bool(clearance>=-1e-6)}
        report['frames'].append(row)
        if not row['entire_body_before_entry']:report['violations'].append(row)
    report['checked_frames']=len(report['frames'])
    if report['frames']:
        report['minimum_front_clearance_m']=min(x['front_clearance_to_conflict_m'] for x in report['frames'])
    report['passed']=not report['violations']
    report['failure_reason']=None if report['passed'] else 'ego_waiting_body_intrudes_declared_conflict'
    return report


def branch_audit(state, spec, branch, meta, diagnostics, pedestrians=None, signals=None, merge_evidence=None):
    ec, _ = contracts(spec)
    times = np.asarray(state['time_s'], float)
    n = len(spec['vehicles'])
    if times.shape != (121,) or not np.allclose(times, np.arange(121)*.05-2, atol=1e-8, rtol=0):
        raise ValueError('Execution audit requires fixed -2..4s,20Hz timeline')
    for key, shape in {'position': (121,n,3), 'velocity': (121,n,3), 'rotation_deg': (121,n,3),
                       'control': (121,n,3), 'observed_control': (121,n,3), 'target_speed': (121,n)}.items():
        if np.shape(state[key]) != shape or not np.isfinite(state[key]).all():
            raise ValueError(f'Invalid state field {key}')
    if not isinstance(meta.get('collisions'), list):
        raise ValueError('Collision sensor record missing')
    if any(not isinstance(e.get('time_s'), (int,float)) or not np.isfinite(e['time_s']) for e in meta['collisions']):
        raise ValueError('Collision timestamps must be finite')
    failures = []
    replay, computed = replay_policy(state, spec, branch, diagnostics, pedestrians, signals, merge_evidence)
    if not replay['passed']:
        failures.append('policy_or_actuator_execution_mismatch')
    if any(e['time_s'] <= 1e-8 for e in meta['collisions']):
        failures.append('initialization_or_prefix_collision')
    speeds = np.linalg.norm(state['velocity'][..., :2], axis=-1)
    profile = spec['action']['profiles']['maintain' if branch == 'maintain_repeat' else branch]
    kind = profile['kind']; onset = float(profile.get('onset_s', 0))
    ego_contact = _role_contact(meta, 'ego')
    mask = (times >= onset-1e-8)
    if ego_contact is not None:
        mask &= times < ego_contact-1e-8
    enough = bool(mask.any() and times[mask][-1]-times[mask][0] >= ec['min_precontact_motion_s']-1e-8)
    tracking = _tracking(times, speeds[:,0], state['target_speed'][:,0], mask, ec)
    if enough and not tracking['passed']:
        failures.append('ego_speed_tracking_failed')
    stations = np.array([project_to_route(p, spec['vehicles'][0]['route'])[0] for p in state['position'][:,0]])
    physical = {'tracking': tracking, 'precontact_motion_observed_s':
        float(times[mask][-1]-times[mask][0]) if mask.any() else 0., 'terminal_requirement_checked': False}
    outcome = []
    if kind == 'lane_change':
        hold = max(ec['lane_hold_s'], float(profile.get('required_final_hold_s', .5)))
        lane = trajectory_audit(state, spec, branch, ec['lane_center_tolerance_m'],
                                ec['lane_heading_tolerance_deg'], hold)
        physical['lane_change'] = {k:v for k,v in lane.items() if k != 'frames'}
        if ego_contact is None:
            outcome.append('lane_change_completed' if lane['completed'] else 'lane_change_incomplete')
        else:
            outcome.append('lane_geometry_complete_after_contact' if lane['completed'] else 'lane_change_incomplete')
        attained = lane['completed']
        deadline = onset+float(profile['duration_s'])+float(profile.get('settling_time_s', 0.))+hold
        # A collision does not excuse a requirement that already failed before
        # contact. Never use collision-displaced post-contact geometry as proof
        # of successful completion, either.
        checked = ego_contact is None
        if ego_contact is not None and times[mask].size and times[mask][-1] >= deadline-1e-8:
            prior = times < ego_contact-1e-8
            prestate = {k:(v[prior] if k not in ('bbox_extent','bbox_offset','bbox_rotation_deg')
                          and np.ndim(v) > 0 and len(v) == len(times) else v)
                        for k,v in state.items()}
            prelane = trajectory_audit(prestate, spec, branch, ec['lane_center_tolerance_m'],
                                      ec['lane_heading_tolerance_deg'], hold)
            physical['lane_change_before_contact'] = {k:v for k,v in prelane.items() if k != 'frames'}
            attained = prelane['completed']; checked = True
        if checked:
            physical['terminal_requirement_checked'] = True
            if not attained:
                failures.append('ego_lane_change_not_completed_to_specification')
        physical['lane_completion_deadline_s'] = deadline
    else:
        offsets = np.abs(project(state['position'][:,0], spec['vehicles'][0]['route'])[-1])
        maximum = float(offsets[mask].max()) if mask.any() else None
        physical['max_precontact_route_offset_m'] = maximum
        if enough and maximum > ec['route_lateral_tolerance_m']:
            failures.append('ego_route_tracking_failed')
    if kind in ('longitudinal', 'commit_go') and enough:
        ids = np.flatnonzero(mask)
        desired = float(state['target_speed'][ids[-1],0]-state['target_speed'][ids[0],0])
        realized = float(speeds[ids[-1],0]-speeds[ids[0],0])
        physical.update(requested_speed_change_m_s=desired, actual_speed_change_m_s=realized)
        if abs(desired) >= ec['minimum_speed_change_m_s'] and times[ids[-1]]-times[ids[0]] >= ec['speed_tracking_grace_s']:
            if np.sign(desired)*realized < abs(desired)*ec['minimum_speed_change_fraction']:
                failures.append('ego_requested_speed_change_not_realized')
    ego_rows = [x[0] for x in computed]
    if kind in ('stop_hold','yield_wait') or kind == 'commit_go' and onset > 0:
        stop = float(profile['stop_route_s'])
        hold_mask = np.array([bool(x.get('force_brake_hold')) for x in ego_rows]) & mask
        stopping_mask = np.array([x.get('released_at_s') is None and x.get('phase') not in ('before_onset','committed_entry') for x in ego_rows]) & mask
        if hold_mask.any() and np.any(speeds[hold_mask,0] > ec['stopped_speed_m_s']):
            # A hold clamp may first engage while braking past a stop line;
            # sustained velocity, not that single transition tick, is failure.
            if longest_violation(times, hold_mask & (speeds[:,0] > ec['stopped_speed_m_s'])) > ec['speed_tracking_grace_s']:
                failures.append('ego_failed_to_hold_stationary')
        overshoot = max(0., float((stations[stopping_mask]-stop).max())) if stopping_mask.any() else 0.
        physical['max_stop_overshoot_m'] = overshoot
        if overshoot > ec['stop_position_overshoot_m']:
            failures.append('ego_passed_required_wait_or_stop_position')
        released = next((x.get('released_at_s') for x in ego_rows if x.get('released_at_s') is not None), None)
        stopped = next((x.get('stopped_at_s') for x in ego_rows if x.get('stopped_at_s') is not None), None)
        physical.update(stopped_at_s=stopped, released_at_s=released)
        strict_yield = kind == 'yield_wait' and profile.get('release_condition') == 'specified_actors_cleared_and_entry_clear_v1'
        if strict_yield:
            active_rows=[(float(t),row) for t,row in zip(times,ego_rows) if t>=onset-1e-8]
            invalid=[t for t,row in active_rows if row.get('release_reason') and not (
                row.get('all_wait_targets_cleared') and row.get('entry_clear') and row.get('entry_permitted_now'))]
            if invalid: failures.append('yield_released_without_specified_clearance')
            physical['yield_semantics']={'policy':profile['release_condition'],
                'wait_for_roles':profile['wait_for_roles'], 'invalid_release_times_s':invalid,
                'rolling_release_times_s':[t for t,row in active_rows if row.get('rolling_release')],
                'reblocked_times_s':[t for t,row in active_rows if row.get('reblocked')],
                'targets_cleared_at_end':bool(ego_rows[-1].get('all_wait_targets_cleared')),
                'entry_clear_at_end':bool(ego_rows[-1].get('entry_clear')),
                'target_passage_required_for_release':True,
                'target_passage_required_for_scene_acceptance':False,
                'waiting_timeout_is_valid_policy_outcome':True}
        elif kind == 'yield_wait':
            physical['yield_semantics']={'policy':'legacy_gap_after_stop_v0',
                'strict_specified_actor_clearance_verified':False}
        if strict_yield or kind == 'stop_hold' and profile.get('hold_policy') == 'hold_until_window_end_v1':
            footprint=waiting_footprint_audit(state,spec,profile,ego_rows,mask,ec['stopped_speed_m_s'])
            physical['waiting_footprint']=footprint
            if not footprint['passed']:
                failures.append(footprint['failure_reason'])
        if 'stop_deadline_s' in profile:
            deadline = float(profile['stop_deadline_s'])
            deadline_checked = times[-1] >= deadline-1e-8 and (ego_contact is None or ego_contact > deadline+1e-8)
            before_deadline = mask & (times <= deadline+1e-8)
            stationary = before_deadline & (speeds[:,0] <= ec['stopped_speed_m_s'])
            rolling_exempt = bool(strict_yield and released is not None and released <= deadline+1e-8)
            deadline_checked = deadline_checked and not rolling_exempt
            physical.update(stop_deadline_s=deadline, stop_deadline_checked=bool(deadline_checked),
                            observed_stop_by_deadline=bool(stationary.any()),
                            stop_deadline_released_before_requirement=rolling_exempt)
            if deadline_checked and not stationary.any():
                failures.append('ego_did_not_stop_by_declared_deadline')
        if released is not None:
            outcome.append('wait_released' if kind == 'yield_wait' else 'hold_released')
        elif stopped is not None:
            outcome.append('waiting' if kind == 'yield_wait' else 'stopped_holding')
        else:
            outcome.append('approaching_stop')
        if kind in ('stop_hold','yield_wait') and ego_contact is None and released is None:
            physical['terminal_requirement_checked'] = True
            if speeds[-1,0] > ec['stopped_speed_m_s']:
                failures.append('ego_did_not_stop_within_declared_window')
        if kind == 'yield_wait' and any(x.get('timed_out') for x in ego_rows):
            outcome.append('wait_timeout_hold')
    if kind == 'maintain':
        outcome.append('maintaining_speed')
    if kind == 'nominal':
        outcome.append('normal_policy_continued')
    if kind == 'longitudinal':
        outcome.append('decelerating' if profile['acceleration_m_s2'] < 0 else 'accelerating')
    if kind == 'commit_go':
        outcome.append('entry_attempted')
    if speeds[-1,0] <= ec['stopped_speed_m_s']:
        outcome.append('stationary_at_end')
    if ego_contact is not None:
        outcome.append('collision')
    npc_outcomes = {}
    npc_checks = {}
    for i, actor in enumerate(spec['vehicles'][1:], 1):
        role = actor['role']; contact = _role_contact(meta, role)
        valid = (times >= -1e-8) & (times < contact-1e-8 if contact is not None else True)
        reasons = sorted({rows[i].get('reason','unspecified') for rows,t in zip(computed,times) if t >= -1e-8})
        travelled = project_to_route(state['position'][-1,i], actor['route'])[0]-project_to_route(state['position'][40,i], actor['route'])[0]
        tags = ['stationary_at_end' if speeds[-1,i] <= ec['stopped_speed_m_s'] else 'moving_at_end']
        if any('yield' in r or 'stop' in r or 'brake' in r for r in reasons):
            tags.append('reactive_constraint_activated')
        if speeds[-1,i] <= ec['stopped_speed_m_s'] and 'reactive_constraint_activated' in tags:
            tags.append('waiting')
        if speeds[-1,i] < speeds[40,i]-.5:
            tags.append('decelerated')
        if contact is not None:
            tags.append('collision')
        route = np.asarray(actor['route'])
        route_length = np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1).sum()
        if project_to_route(state['position'][-1,i], route)[0] >= route_length-.5:
            tags.append('declared_route_end_reached')
        merge_rows=[(float(t),row[i]['dynamic_merge']) for row,t in zip(computed,times)
                    if t>=-1e-8 and 'dynamic_merge' in row[i]]
        merge_modes=sorted({d['state_after']['mode'] for _,d in merge_rows})
        merge_transitions=[{'time_s':t,'before':d['state_before']['mode'],
            'after':d['state_after']['mode'],'reason':d['reason'],
            'observed_lateral_progress':d['observed_lateral_progress']}
            for t,d in merge_rows if d['state_before']['mode']!=d['state_after']['mode']]
        npc_outcomes[role] = {'events': tags, 'policy_reasons': reasons,
            'initial_speed_m_s': float(speeds[40,i]), 'final_speed_m_s': float(speeds[-1,i]),
            'route_progress_m': float(travelled), 'first_contact_s': contact,
            'response_required': False, 'cancellation_state_supported': bool(merge_rows),
            'dynamic_merge':{'policy_modes':merge_modes,'policy_transitions':merge_transitions,
                'commands_are_not_proof_of_completed_lateral_motion':True} if merge_rows else None}
        npc_checks[role] = _tracking(times, speeds[:,i], state['target_speed'][:,i], valid, ec)
        if not npc_checks[role]['passed']:
            failures.append(f'npc_speed_tracking_failed:{role}')
    if not enough and ego_contact is None:
        failures.append('insufficient_motion_observations')
    # Null is explicitly unassessed, not a pass. Valid early contacts are retained
    # as censored outcomes by the orchestration layer, with commands still gated.
    execution_pass = False if failures else True if enough else None
    return {'execution_pass': execution_pass, 'failure_reason': failures,
        'quality_status': 'failed' if failures else 'censored' if ego_contact is not None else 'passed',
        'execution_scope': 'strictly_precontact_motion_and_all_policy_commands' if ego_contact is not None else 'whole_window',
        'observed_outcome': {'ego': outcome, 'npcs': npc_outcomes,
            'collision': bool(meta['collisions']), 'ego_collision': ego_contact is not None,
            'first_contact_s': min((float(e['time_s']) for e in meta['collisions']), default=None),
            'censored_after_contact': ego_contact is not None,
            'terminal_completion_assessed': ego_contact is None},
        'checks': {'policy_execution': replay, 'ego_physical_execution': physical, 'npc_physical_execution': npc_checks},
        'execution_contract': ec}


def response_audit(states, spec, pedestrians=None):
    """Pairwise max over all model branches at matching future times, in XY.

    Translation/rotation of a shared t0 frame leaves these horizontal distances
    unchanged. The QA repeat is excluded from effect statistics and measured
    separately; there is no effect-size gate.
    """
    _, contract = contracts(spec)
    actions = list(branch_names(spec, False)); names = list(branch_names(spec))
    reference = states['maintain']; times = reference['time_s']; future = times > 1e-8
    if not future.any():
        raise ValueError('Future state observations are required')
    for name in names:
        if not np.array_equal(states[name]['time_s'], times):
            raise ValueError('Cross-branch reaction comparison requires identical timestamps')
    per_role = {}; failures = []; unrelated = {}
    for actor, role in enumerate(spec['vehicle_roles'][1:],1):
        best = {'max_pairwise_xy_difference_m': 0., 'pair': None, 'time_s': None}
        pairs = []
        for a,b in itertools.combinations(actions,2):
            delta = np.linalg.norm(states[a]['position'][future,actor,:2]-states[b]['position'][future,actor,:2],axis=1)
            peak = int(delta.argmax()); distance = float(delta[peak])
            pairs.append({'branches':[a,b], 'max_xy_difference_m':distance})
            if distance > best['max_pairwise_xy_difference_m']:
                best.update(max_pairwise_xy_difference_m=distance,pair=[a,b],time_s=float(times[future][peak]))
        repeat = float(np.linalg.norm(reference['position'][future,actor,:2]-states['maintain_repeat']['position'][future,actor,:2],axis=1).max())
        per_role[role] = {**best, 'repeat_xy_noise_m': repeat, 'pairwise_maxima': pairs}
        if role in contract['unrelated_roles']:
            maxima = {'position_m': 0., 'velocity_m_s': 0., 'heading_deg': 0.}
            for a,b in itertools.combinations(names,2):
                maxima['position_m'] = max(maxima['position_m'],float(np.linalg.norm(states[a]['position'][future,actor]-states[b]['position'][future,actor],axis=1).max()))
                maxima['velocity_m_s'] = max(maxima['velocity_m_s'],float(np.linalg.norm(states[a]['velocity'][future,actor]-states[b]['velocity'][future,actor],axis=1).max()))
                angle = np.abs((states[a]['rotation_deg'][future,actor,2]-states[b]['rotation_deg'][future,actor,2]+180)%360-180)
                maxima['heading_deg'] = max(maxima['heading_deg'],float(angle.max()))
            ok = (maxima['position_m'] <= contract['unrelated_position_tolerance_m'] and
                  maxima['velocity_m_s'] <= contract['unrelated_velocity_tolerance_m_s'] and
                  maxima['heading_deg'] <= contract['unrelated_heading_tolerance_deg'])
            unrelated[role] = {**maxima,'passed':ok,
                'repeat_xy_noise_m':repeat,
                'repeat_noise_within_position_tolerance':repeat <= contract['unrelated_position_tolerance_m']}
            if not ok:
                failures.append(f'declared_unrelated_npc_changed:{role}')
    maximum = max((x['max_pairwise_xy_difference_m'] for x in per_role.values()),default=0.)
    noise = max((x['repeat_xy_noise_m'] for x in per_role.values()),default=0.)
    target_roles = {p['role'] for c in spec['action'].get('conflicts', []) for p in c['participants']}
    if not target_roles:
        target_roles = set(spec['vehicle_roles'][1:]) - set(contract['unrelated_roles'])
    target_sources = {role: (states, spec['vehicle_roles'].index(role), 'vehicle')
                      for role in target_roles if role in spec['vehicle_roles']}
    if spec.get('pedestrians'):
        if pedestrians is None or set(pedestrians) != set(names):
            raise ValueError('Response diagnostics require every original pedestrian branch including repeat')
        for name in names:
            if not np.array_equal(pedestrians[name]['time_s'], times):
                raise ValueError('Pedestrian response timestamps differ from vehicle observations')
        for index, person in enumerate(spec['pedestrians']):
            target_sources[person['role']] = (pedestrians, index, 'pedestrian')
    targets = {}
    for role, (source, index, actor_type) in sorted(target_sources.items()):
        repeat = float(np.linalg.norm(source['maintain']['position'][future,index,:2] -
                                      source['maintain_repeat']['position'][future,index,:2], axis=1).max())
        pairs = []
        for a,b in itertools.combinations(actions,2):
            delta = np.linalg.norm(source[a]['position'][future,index,:2] - source[b]['position'][future,index,:2], axis=1)
            speed_delta = np.abs(np.linalg.norm(source[a]['velocity'][future,index,:2],axis=1) -
                                 np.linalg.norm(source[b]['velocity'][future,index,:2],axis=1))
            pairs.append({'branches':[a,b], 'max_xy_difference_m':float(delta.max()),
                          'rms_xy_difference_m':float(np.sqrt(np.mean(delta**2))),
                          'max_speed_difference_m_s':float(speed_delta.max()),
                          'above_05_m':bool(delta.max() > .5), 'above_10_m':bool(delta.max() > 1.),
                          'above_five_times_repeat_noise':bool(delta.max() > 5.*repeat)})
        targets[role] = {'actor_type':actor_type, 'repeat_xy_noise_m':repeat, 'pairs':pairs,
            'first_moving_time_s':{name:next((float(t) for t,v in zip(times[future],
                np.linalg.norm(source[name]['velocity'][future,index,:2],axis=1)) if v > .05),None) for name in actions}}
    return {'response_metrics': {'max_npc_displacement_diff_m':maximum,
        'reactive_npc_count_05':sum(x['max_pairwise_xy_difference_m'] > .5 for x in per_role.values()),
        'reactive_npc_count_10':sum(x['max_pairwise_xy_difference_m'] > 1. for x in per_role.values()),
        'effect_strength':'weak' if maximum <= .5 else 'medium' if maximum <= 1. else 'strong',
        'effect_strength_rule':{'weak':'max <= 0.5 m','medium':'0.5 m < max <= 1 m','strong':'max > 1 m'},
        'definition':'max over non-ego vehicles, all unordered pairs of five non-QA branches, and matched t>0 observations of horizontal position difference; common world/shared t0 coordinates',
        'threshold_comparison':'strictly greater than 0.5 or 1.0 m; engineering labels, not statistical significance',
        'relevant_actor_response': {'targets':targets, 'pair_count_per_actor':10,
            'used_as_acceptance_gate':False, 'includes_pedestrians':bool(spec.get('pedestrians')),
            'definition':'Every unordered ego-action pair, per related vehicle or pedestrian; no claim that all targets must react differently'},
        'max_npc_repeat_xy_noise_m':noise,'threshold_05_above_repeat_noise':noise < .5,
        'threshold_10_above_repeat_noise':noise < 1.,'used_as_acceptance_gate':False,'per_npc':per_role},
        'unrelated_invariance':{'passed':not failures,'failure_reason':failures,'contract':contract,'roles':unrelated,
            'tested_roles':list(contract['unrelated_roles']),
            'definition':'all six branches including repeat, all t>0 samples; 3D position/velocity and wrapped yaw; no relabeling of declared unrelated roles'}}


def scene_opportunity(state, spec, pedestrians=None):
    """Conservative, t0-only reachability check, never a reaction-size filter.

    It is possible to pass and then wait for the whole recording or see no NPC
    reaction. It is not possible to pass by placing every participant beyond
    the interaction or outside the reachable four-second route interval.
    """
    ec, _ = contracts(spec)
    t0 = int(np.argmin(abs(state['time_s'])))
    if abs(state['time_s'][t0]) > 1e-8:
        raise ValueError('Interaction opportunity requires a recorded t0')
    horizon = float(spec.get('duration_s',4.))
    if horizon != 4.:
        raise ValueError('Opportunity horizon must match the fixed four-second protocol')
    observations = {}
    for i, actor in enumerate(spec['vehicles']):
        station,offset,tangent = project_to_route(state['position'][t0,i],actor['route'])
        projected = project(state['position'][t0,i][None],actor['route'])[2][0]
        endpoint_error = abs(float((state['position'][t0,i,:2]-projected)@tangent))
        speed = max(0.,float(state['velocity'][t0,i,:2]@tangent))
        heading = float(np.rad2deg(np.arctan2(tangent[1],tangent[0])))
        heading_error = abs(float((state['rotation_deg'][t0,i,2]-heading+180)%360-180))
        policy = actor.get('npc_policy',{})
        acceleration = float(policy.get('idm',{}).get('max_acceleration',spec['controller']['idm']['max_acceleration']))
        if i == 0:
            acceleration = max(float(spec['action']['nominal_policy']['idm']['max_acceleration']),
                               *(max(0.,float(p.get('acceleration_m_s2',0.))) for p in spec['action']['profiles'].values()))
        route_length = float(np.linalg.norm(np.diff(np.asarray(actor['route'])[:,:2],axis=0),axis=1).sum())
        observations[actor['role']] = {'route_s':station,'speed_m_s':speed,
            'conservative_max_forward_distance_m':speed*horizon+.5*acceleration*horizon*horizon,
            'reachable_route_s':[station,min(route_length,station+speed*horizon+.5*acceleration*horizon*horizon)],
            'route_lateral_offset_m':float(offset),'heading_error_deg':heading_error,
            'outside_route_endpoint_m':endpoint_error,
            'on_declared_route':bool(abs(offset)<=ec['route_lateral_tolerance_m'] and heading_error<=45. and endpoint_error<=.5),
            'half_length_m':float(state['bbox_extent'][i,0])}
        if actor.get('route_maneuver',{}).get('policy_version')=='observed-gap-merge-v1':
            m=actor['route_maneuver']
            offsets=[abs(project_to_route(state['position'][t0,i],m[key])[1]) for key in ('source_route','target_route')]
            observations[actor['role']]['on_declared_route']=bool(min(offsets)<=m['lane_half_width_m']+.05 and heading_error<=45. and endpoint_error<=.5)
            observations[actor['role']]['mapped_corridor_offsets_m']=offsets
    if spec.get('pedestrians'):
        if pedestrians is None:
            raise ValueError('Pedestrian opportunity requires recorded pedestrian state at t0')
        for i,actor in enumerate(spec['pedestrians']):
            station,offset,tangent = project_to_route(pedestrians['position'][t0,i],actor['route_points'])
            speed = float(actor['speed_mps'])
            delay = max(0.,float(actor.get('start_time_s',0.)))
            travel = speed*max(0.,horizon-delay)
            observations[actor['role']] = {'route_s':station,'speed_m_s':float(np.linalg.norm(pedestrians['velocity'][t0,i,:2])),
                'conservative_max_forward_distance_m':travel,'reachable_route_s':[station,station+travel],
                'route_lateral_offset_m':float(offset),'on_declared_route':bool(abs(offset)<=.5),
                'half_length_m':float(np.linalg.norm(pedestrians['bbox_extent'][i,:2]))}
    template = spec.get('grid_spec',{}).get('template') or spec.get('grid_cell',{}).get('template') or spec.get('template')
    if template is None:
        template = spec.get('provenance',{}).get('template')
    # The compiler has used both explicit template and grid metadata over its
    # versions; new specs always carry template_id in grid membership.
    if template is None:
        template = spec.get('grid',{}).get('template') or spec.get('template_id')
    evidence = []; failures = []
    ego = observations['ego']
    if not ego['on_declared_route']:
        failures.append('ego_t0_not_on_declared_route')
    def interval_hit(role,entry,exit):
        row=observations[role];lo,hi=row['reachable_route_s'];extent=row['half_length_m']
        return bool(row['on_declared_route'] and lo-extent <= exit and hi+extent >= entry)
    conflicts=spec['action'].get('conflicts',[])
    for conflict in conflicts:
        own_hit=interval_hit('ego',float(conflict['ego_entry_s']),float(conflict['ego_exit_s']))
        participants=[]
        for participant in conflict['participants']:
            role=participant['role']
            if role not in observations:
                raise ValueError(f'Opportunity participant {role} has no recorded initial state')
            hit=interval_hit(role,float(participant['entry_s']),float(participant['exit_s']))
            actor=next((a for a in spec['vehicles'] if a['role']==role),{})
            maneuver=actor.get('route_maneuver')
            already_merged=bool(maneuver and observations[role]['route_s']>=float(maneuver['source_interval_s_m'][1]))
            if maneuver and maneuver.get('policy_version')=='observed-gap-merge-v1':
                # Longitudinal travel while waiting is not completed merging.
                idx=spec['vehicle_roles'].index(role)
                already_merged=abs(project_to_route(state['position'][t0,idx],maneuver['target_route'])[1])<.15
            participants.append({'role':role,'reachable_conflict':hit,'merge_already_completed_at_t0':already_merged,
                                 'valid_opportunity':hit and not already_merged})
        evidence.append({'conflict_id':conflict['id'],'ego_reachable_conflict':own_hit,
            'participants':participants,'valid_opportunity':own_hit and any(p['valid_opportunity'] for p in participants)})
    if conflicts:
        if not any(row['valid_opportunity'] for row in evidence):
            failures.append('no_declared_interaction_reachable_from_actual_t0_within_four_seconds')
    else:
        # Straight-following and ego lane changes do not use a point-conflict.
        # Check actual nearby vehicles on the source or declared target lanes.
        ego_position=state['position'][t0,0];ego_route=spec['vehicles'][0]['route']
        ego_station,_,ego_tangent=project_to_route(ego_position,ego_route)
        is_lateral=any(p['kind']=='lane_change' for p in spec['action']['profiles'].values())
        routes=({side:value['target_route'] for side,value in spec['action'].get('lane_changes',{}).items()}
                if is_lateral else {'source':ego_route})
        maximum_brake=max((max(0.,-float(p.get('acceleration_m_s2',0.))) for p in spec['action']['profiles'].values()),default=0.)
        min_ego_travel=(ego['speed_m_s']*horizon if maximum_brake<=0 else
            (ego['speed_m_s']*horizon-.5*maximum_brake*horizon*horizon if ego['speed_m_s']/maximum_brake>horizon
             else ego['speed_m_s']**2/(2*maximum_brake)))
        for side,route in routes.items():
            own_s,_,tangent=project_to_route(ego_position,route)
            for i,actor in enumerate(spec['vehicles'][1:],1):
                role=actor['role'];row=observations[role]
                station,offset,npc_tangent=project_to_route(state['position'][t0,i],route)
                projected=project(state['position'][t0,i][None],route)[2][0]
                endpoint_error=abs(float((state['position'][t0,i,:2]-projected)@npc_tangent))
                yaw=np.deg2rad(state['rotation_deg'][t0,i,2])
                same_direction=float(np.array([np.cos(yaw),np.sin(yaw)])@npc_tangent)>np.cos(np.deg2rad(45.))
                forward=station-own_s
                policy=actor.get('npc_policy',{}).get('idm',{})
                idm={**spec['controller']['idm'],**policy}
                influence=state['bbox_extent'][0,0]+state['bbox_extent'][i,0]+float(idm['minimum_gap'])+float(idm['time_headway'])*row['speed_m_s']
                if is_lateral:
                    closure=abs(row['speed_m_s']-ego['speed_m_s'])*horizon+.5*float(idm['max_acceleration'])*horizon*horizon
                    close=abs(forward)<=closure+influence
                else:
                    closure=max(0.,row['conservative_max_forward_distance_m']-min_ego_travel)
                    close=forward<0 and -forward<=closure+influence
                valid=bool(same_direction and abs(offset)<=ec['route_lateral_tolerance_m'] and endpoint_error<=.5 and close and row['on_declared_route'])
                evidence.append({'lane':side,'role':role,'relative_route_station_m':float(forward),
                    'route_lateral_offset_m':float(offset),'conservative_closure_plus_influence_m':float(closure+influence),
                    'outside_compared_route_endpoint_m':endpoint_error,
                    'valid_opportunity':valid})
        if not any(row['valid_opportunity'] for row in evidence):
            failures.append('no_nearby_route_compatible_vehicle_for_declared_straight_interaction')
    return {'passed':not failures,'failure_reason':failures,'template':template,'horizon_s':horizon,
        'actual_t0':observations,'evidence':evidence,
        'uses_future_response':False,
        'definition':'Conservative geometric reachability from measured common t0 using frozen routes, maximum configured acceleration and a four-second horizon. Signal-blocked waiting and weak outcomes are permitted; this is necessary opportunity, not promised interaction.'}


def scene_audit(scene, spec):
    scene = Path(scene); states = {}; branches = {}; source = {}
    contracts(spec)
    maintain_pedestrians = None; pedestrian_states = {}
    for branch in branch_names(spec):
        folder = scene/'branches'/branch
        meta = json.loads((folder/'meta.json').read_text())
        if meta.get('status') != 'complete' or meta.get('branch') != branch or meta.get('vehicle_roles') != spec['vehicle_roles']:
            raise ValueError(f'{branch}: incomplete branch metadata or actor identity')
        for key in ('spec_sha256','code_sha256'):
            if not spec.get(key) or meta.get(key) != spec[key]:
                raise ValueError(f'{branch}: frozen {key} mismatch')
        state = load_arrays(folder/'states.npz'); states[branch] = state
        diagnostics = json.loads((folder/'controller_diagnostics.json').read_text())
        people = load_arrays(folder/'pedestrians.npz') if spec.get('pedestrians') else None
        if people is not None:
            pedestrian_states[branch] = people
        if branch == 'maintain':
            maintain_pedestrians = people
        signals = json.loads((folder/'signal_states.json').read_text()) if spec.get('traffic_signal_program') else None
        merge_evidence=json.loads((folder/'npc_merge_state.json').read_text()) if (folder/'npc_merge_state.json').is_file() else None
        branches[branch] = branch_audit(state,spec,branch,meta,diagnostics,people,signals,merge_evidence)
        source[branch] = {name:sha256(folder/name) for name in ('meta.json','states.npz','controller_diagnostics.json')}
        if merge_evidence is not None:
            source[branch]['npc_merge_state.json']=sha256(folder/'npc_merge_state.json')
    response = response_audit(states,spec,pedestrian_states if pedestrian_states else None)
    opportunity = scene_opportunity(states['maintain'],spec,maintain_pedestrians)
    failures = [f'{name}:{reason}' for name,row in branches.items() for reason in row['failure_reason']]
    failures += response['unrelated_invariance']['failure_reason']
    failures += opportunity['failure_reason']
    # Prefix includes t0 states; first intervention control is intentionally not
    # included because it acts on [t0,t0+dt), not on the shared past.
    prefix_failures = []
    base = states['maintain']; hist = base['time_s'] <= 1e-8
    for branch,state in states.items():
        for key in ('position','velocity','rotation_deg','angular_velocity','acceleration','transform_world_vehicle'):
            if not np.array_equal(state[key][hist],base[key][hist]):
                prefix_failures.append(f'{branch}:shared_prefix_{key}_changed')
    failures += prefix_failures
    return {'execution_pass':not failures,'failure_reason':failures,'branches':branches,**response,
        'interaction_opportunity':opportunity,
        'paired_prefix':{'passed':not prefix_failures,'failure_reason':prefix_failures,'definition':'exact per-frame physical states through t0'},
        'source_files':source,'spec_file_sha256':sha256(scene/'spec.json'),
        'censored_branches':[name for name,row in branches.items() if row['quality_status']=='censored'],
        'all_planned_model_branches_retained':list(branch_names(spec,False)),
        'limits':['Censored post-contact completion is not a successful maneuver; valid contact outcomes remain in the scene.',
                  'NPC policy execution is checked, but weak/absent reaction and incomplete NPC merging are not failures.',
                  'Dynamic adjacent merging may wait or cancel early; mapped ramp routes retain their own longitudinal policy.']}


def audit_dataset(data):
    data = Path(data); manifest = json.loads((data/'manifest.json').read_text())
    scenes = {}; errors = []
    for name in manifest['scene_ids']:
        try:
            scene = data/name; spec = json.loads((scene/'spec.json').read_text())
            scenes[name] = scene_audit(scene,spec)
            if not scenes[name]['execution_pass']:
                errors.extend(f'{name}:{failure}' for failure in scenes[name]['failure_reason'])
        except (OSError,ValueError,KeyError,TypeError,IndexError,AssertionError) as exc:
            errors.append(f'{name}:{exc}')
    return {'kind':'strategy_execution_and_observed_outcomes','status':'passed' if scenes and not errors else 'failed',
        'errors':errors,'scenes':scenes,'audit_sha256':sha256(__file__),
        'acceptance_semantics':'Gate policy/actuator errors, physical execution failures, prefix or unrelated-role mismatch; retain valid contact-censored and waiting outcomes. Never gate on NPC effect magnitude.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); report=audit_dataset(args.data)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.out.with_suffix(args.out.suffix+'.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temporary.replace(args.out)
    print(json.dumps({'status':report['status'],'scenes':len(report['scenes']),'errors':report['errors']}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
