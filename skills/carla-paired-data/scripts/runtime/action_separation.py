"""Prospectively frozen five-action eligibility, separate from execution QA.

All ten unordered pairs must differ in executed controls AND physical motion.
Only the shared pre-contact interval can establish eligibility; collision
consequences and NPC responses remain recorded, never manufactured as actions.
Legacy scenes without this contract retain their original diagnostic semantics.
"""
from __future__ import annotations

import itertools
import numpy as np


CONTRACT = {
    'policy_id': 'five-effective-ego-actions-v1',
    'history_s': 2., 'future_s': 4., 'physics_dt_s': .05,
    'control_thresholds': {'throttle': .1, 'steer': .05, 'brake': .1},
    'minimum_control_duration_s': .25,
    'position_threshold_m': 1., 'speed_threshold_m_s': .5,
    'minimum_motion_duration_s': .5,
    'repeat_multiplier': 5.,
    'minimum_measured_t0_speed_m_s': 2.5,
    'pair_count': 10,
    'contact_scope': 'strictly_before_first_ego_contact_in_either_branch',
    'failure_unit': 'whole_scene',
    'npc_response_used_as_gate': False,
}


def validate_contract(contract):
    if contract != CONTRACT:
        raise ValueError('Five-action eligibility contract differs from frozen v1 thresholds')


def _longest(mask, dt, *, intervals):
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    # Control rows represent intervals; physical states represent instants.
    return max(0, longest if intervals else longest - 1) * dt


def _contact(meta):
    events = meta.get('collisions')
    if not isinstance(events, list):
        raise ValueError('Original collision evidence is required for action separation')
    times = []
    for event in events:
        t = float(event['time_s'])
        if not np.isfinite(t):
            raise ValueError('Nonfinite contact time')
        if event.get('role') == 'ego' or event.get('other_role') == 'ego':
            times.append(t)
    return min(times, default=float('inf'))


def audit_separation(states, planned, metadata, contract, ego_index=0):
    """Validate complete original records, then measure sustained differences."""
    validate_contract(contract)
    planned = list(planned)
    if len(planned) != 5 or len(set(planned)) != 5 or 'maintain' not in planned or 'maintain_repeat' in planned:
        raise ValueError('Five model branches including maintain are required')
    names = set(planned) | {'maintain_repeat'}
    if set(states) != names or set(metadata) != names:
        raise ValueError('Five complete branches and the independent QA repeat are required')
    times = np.arange(121) * .05 - 2.
    series, contacts = {}, {}
    for name in names:
        state = states[name]
        if np.shape(state['time_s']) != (121,) or not np.allclose(state['time_s'], times, atol=1e-8, rtol=0):
            raise ValueError('Action separation requires the unchanged -2..4s 20Hz timeline')
        arrays = {}
        for key in ('position', 'velocity', 'control'):
            array = np.asarray(state[key], float)
            if (array.ndim != 3 or array.shape[0] != 121 or array.shape[2] != 3
                    or not 0 <= ego_index < array.shape[1] or not np.isfinite(array).all()):
                raise ValueError(f'{name}: invalid {key} evidence')
            arrays[key] = array[:, ego_index]
        arrays['speed'] = np.linalg.norm(arrays['velocity'][:, :2], axis=1)
        series[name] = arrays
        contacts[name] = _contact(metadata[name])
    future = times > 1e-8
    controls = (times >= -1e-8) & (times < 4. - 1e-8)
    a, b = series['maintain'], series['maintain_repeat']
    noise = {
        'position_m': float(np.linalg.norm(a['position'][future, :2] - b['position'][future, :2], axis=1).max()),
        'speed_m_s': float(np.abs(a['speed'][future] - b['speed'][future]).max()),
        'control': np.abs(a['control'][controls] - b['control'][controls]).max(axis=0).tolist(),
    }
    control_limits = np.maximum(list(CONTRACT['control_thresholds'].values()),
                                np.asarray(noise['control']) * CONTRACT['repeat_multiplier'])
    position_limit = max(CONTRACT['position_threshold_m'], noise['position_m'] * CONTRACT['repeat_multiplier'])
    speed_limit = max(CONTRACT['speed_threshold_m_s'], noise['speed_m_s'] * CONTRACT['repeat_multiplier'])
    pairs, errors = [], []
    for left, right in itertools.combinations(planned, 2):
        a, b = series[left], series[right]
        contact = min(contacts[left], contacts[right])
        # An outgoing interval ending at contact cannot establish control separation.
        control_mask = controls & (times + .05 < contact - 1e-8)
        state_mask = future & (times < contact - 1e-8)
        delta_control = np.abs(a['control'] - b['control'])
        delta_position = np.linalg.norm(a['position'][:, :2] - b['position'][:, :2], axis=1)
        delta_speed = np.abs(a['speed'] - b['speed'])
        control_s = _longest(control_mask & np.any(delta_control >= control_limits, axis=1), .05, intervals=True)
        position_s = _longest(state_mask & (delta_position >= position_limit), .05, intervals=False)
        speed_s = _longest(state_mask & (delta_speed >= speed_limit), .05, intervals=False)
        control_pass = control_s + 1e-8 >= CONTRACT['minimum_control_duration_s']
        motion_pass = max(position_s, speed_s) + 1e-8 >= CONTRACT['minimum_motion_duration_s']
        failures = ([] if control_pass else ['insufficient_sustained_executed_control_difference']) + (
            [] if motion_pass else ['insufficient_sustained_precontact_ego_motion_difference'])
        row = {'branches': [left, right], 'passed': not failures, 'failure_reasons': failures,
               'first_ego_contact_s': contact if np.isfinite(contact) else None,
               'control_difference_duration_s': control_s,
               'position_difference_duration_s': position_s, 'speed_difference_duration_s': speed_s,
               'precontact_position_max_m': float(delta_position[state_mask].max()) if state_mask.any() else 0.,
               'precontact_speed_max_m_s': float(delta_speed[state_mask].max()) if state_mask.any() else 0.}
        pairs.append(row)
        if failures:
            errors.append(f'{left}/{right}: ' + ', '.join(failures))
    speed_t0 = float(series['maintain']['speed'][40])
    if speed_t0 < CONTRACT['minimum_measured_t0_speed_m_s']:
        errors.append('measured_t0_speed_below_distinct_action_design_range')
    return {'status': 'failed' if errors else 'passed', 'errors': errors,
            'contract': contract, 'pair_count': len(pairs), 'passed_pair_count': sum(p['passed'] for p in pairs),
            'all_pairs_passed': not errors, 'measured_t0_speed_m_s': speed_t0,
            'repeat_noise': noise, 'effective_thresholds': {'control': control_limits.tolist(),
                'position_m': position_limit, 'speed_m_s': speed_limit},
            'pairs': pairs, 'npc_response_used_as_gate': False}
