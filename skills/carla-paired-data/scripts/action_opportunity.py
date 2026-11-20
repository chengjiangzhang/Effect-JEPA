"""Prospective response opportunity inside the fixed four-second future.

Uses measured common t0 only. Estimates do not promise actual release or
distinct NPC outcomes; the native qualification reports those separately.
"""
from __future__ import annotations

if __package__:
    from .runtime.action_separation import CONTRACT, validate_contract
else:
    from runtime.action_separation import CONTRACT, validate_contract


MIN_OBSERVATION_S = 1.


def analyze_opportunity(spec, actors, conflicts):
    validate_contract(spec['action_separation_contract'])
    errors, evidence = [], []
    ego = actors['ego']
    if ego['speed_m_s'] < CONTRACT['minimum_measured_t0_speed_m_s']:
        errors.append({'code': 'measured_t0_speed_below_distinct_action_design_range'})
    for identity, conflict in conflicts.items():
        pending = [p for p in conflict['participants'] if not p['already_cleared_at_t0']]
        early = [p for p in pending if p['optimistic_entry_s'] is not None
                 and p['optimistic_entry_s'] <= 4. - MIN_OBSERVATION_S]
        if not early:
            errors.append({'code': 'no_target_entry_opportunity_with_one_second_remaining', 'conflict_id': identity})
    for pedestrian in spec.get('pedestrians', []):
        role = pedestrian['role']; actual = actors[role]; policy = pedestrian['response_policy']
        stop = next(p for p in spec['action']['profiles'].values() if p['kind'] == 'stop_hold')
        response = float(stop['response_time_s'])
        ramp = float(stop['stop_deadline_s']) - response - float(stop['onset_s'])
        # Explicit controller response reserve after the target-speed ramp.
        cue = float(stop['onset_s']) + max(0., ramp * (1. - policy['yielding_speed_m_s'] / max(ego['speed_m_s'], .01))) + response
        admission = max(float(pedestrian['start_time_s']), cue)
        timed_sweep = policy.get('kind') in ('reactive_crossing_v3', 'reactive_crossing_v4')
        if timed_sweep:
            admission = max(0.,float(pedestrian['start_time_s']))
        walk_speed = float(pedestrian['speed_mps'])
        entry = admission + max(0., policy['pedestrian_entry_s'] - actual['body_front_route_s_m']) / walk_speed
        clearance_duration = max(0., policy['pedestrian_exit_s'] - actual['body_rear_route_s_m']) / walk_speed + policy['clearance_time_s']
        other_vehicles = []
        for conflict in policy['vehicle_conflicts']:
            if conflict['role'] == 'ego':
                continue
            other = actors[conflict['role']]
            speed = max(0., other['route_speed_m_s'])
            front = other['body_front_route_s_m'] + speed * admission
            rear = other['body_rear_route_s_m'] + speed * admission
            cleared = rear > conflict['exit_s'] + policy['clearance_distance_m']
            before = front < conflict['entry_s'] - policy['clearance_distance_m']
            gap = (conflict['entry_s'] - policy['clearance_distance_m'] - front) / speed if before and speed > .05 else None
            permits = cleared or (before and (speed <= .05 or gap >= clearance_duration))
            other_vehicles.append({'role': conflict['role'], 'constant_speed_gap_at_cue_s': gap,
                                   'cleared_at_cue_estimate': cleared, 'permits_estimated_crossing': permits})
        row = {'role': role, 'yielding_cue_estimate_s': cue,
               'earliest_policy_decision_s': pedestrian['start_time_s'],
               'entry_after_cue_estimate_s': entry,
               'response_observation_remaining_s': max(0., 4. - entry),
               'crossing_and_clearance_duration_s': clearance_duration,
               'other_vehicle_checks': other_vehicles,
               'uses_future_observations': False, 'actual_admission_or_clearance_guaranteed': False}
        row['entry_policy'] = 'timed_body_sweep_with_continuing_feedback' if timed_sweep else 'whole_crossing_gap'
        row['background_gap_is_admission_gate'] = not timed_sweep
        evidence.append(row)
        if entry + MIN_OBSERVATION_S > 4. + 1e-8:
            errors.append({'code': 'pedestrian_response_opportunity_too_late', 'role': role})
        if not timed_sweep and not all(v['permits_estimated_crossing'] for v in other_vehicles):
            errors.append({'code': 'background_traffic_blocks_designed_pedestrian_opportunity', 'role': role})
    return {'policy_id': 'measured-t0-response-opportunity-v1', 'passed': not errors,
            'history_s': 2., 'future_s': 4., 'minimum_observation_s': MIN_OBSERVATION_S,
            'failure_reasons': errors, 'pedestrians': evidence,
            'basis': 'Common t0 geometry and frozen target/controller/constant-speed estimates only; not a simulation of future responses',
            'future_response_used_as_gate': False}
