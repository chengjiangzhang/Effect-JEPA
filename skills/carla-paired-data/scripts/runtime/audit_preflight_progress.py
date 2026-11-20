#!/usr/bin/env python3
"""Incremental read-only scene checks while a physics preflight is collected.

Only complete branch metadata are read. This quick report supplements the
independent final map, dynamics, physics-deployment and NPC-response audits.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

if __package__:
    from .audit_lane_change import trajectory_audit, precontact_geometry, lane_direction
else:
    from audit_lane_change import trajectory_audit, precontact_geometry, lane_direction
if __package__:
    from .review_common import INITIAL_TOLERANCES, state_errors
else:
    from review_common import INITIAL_TOLERANCES, state_errors
if __package__:
    from .controllers_v2 import branch_names
    from .audit_auxiliary import inspect_auxiliary, compare_auxiliary
else:
    from controllers_v2 import branch_names
    from audit_auxiliary import inspect_auxiliary, compare_auxiliary

PHYSICAL_KEYS = ('position', 'rotation_deg', 'velocity', 'angular_velocity', 'acceleration',
    'transform_world_vehicle', 'control', 'target_speed', 'bbox_extent', 'bbox_offset',
    'bbox_rotation_deg', 'observed_control', 'observed_gear', 'observed_manual_gear_shift',
    'requested_gear', 'requested_manual_gear_shift', 'time_s')


def read_states(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {k: loaded[k] for k in loaded.files}


def physical_differences(a, b, history=False):
    result = {}
    for key in PHYSICAL_KEYS:
        x, y = a[key], b[key]
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f'Invalid physical arrays: {key}')
        if history and not key.startswith('bbox_'):
            # State[t0] is shared, but control[t0] commands the first future
            # interval and may differ legitimately across action branches.
            end = 40 if key in ('control', 'target_speed', 'requested_gear', 'requested_manual_gear_shift') else 41
            x, y = x[:end], y[:end]
        result[key] = float(np.max(np.abs(x-y)))
    return result


def branch_check(folder, config, meta):
    if folder.name not in branch_names(config):
        raise ValueError('Captured branch is unavailable in this scene')
    roles = config['vehicle_roles']
    if (not roles or roles[0] != 'ego' or len(set(roles)) != len(roles)
            or roles != [v['role'] for v in config['vehicles']]):
        raise ValueError('Invalid frozen vehicle role order; ego must be index0')
    for key in ('code_sha256', 'spec_sha256'):
        if not config.get(key) or meta.get(key) != config[key]:
            raise ValueError(f'Branch {key} differs from frozen config')
    if meta.get('vehicle_roles') != roles or meta.get('branch') != folder.name:
        raise ValueError('Recorded branch or actor order differs from config')
    state_path = folder/'states.npz'; states = read_states(state_path)
    dt = np.diff(states['time_s'])
    if (states['time_s'].shape != (121,) or not np.allclose(
            states['time_s'], np.arange(121)*.05-2, atol=1e-8, rtol=0)
            or not np.allclose(dt, .05, atol=1e-9, rtol=0)):
        raise ValueError('Require121 original states over -2..4s in .05s intervals')
    if states['position'].shape != (121, len(roles), 3):
        raise ValueError('Recorded time/actor coverage differs from config')
    physical_differences(states, states)  # Require complete finite physical arrays, including selected scopes.
    auxiliary = inspect_auxiliary(folder, config, meta, states)
    if auxiliary['status'] != 'passed':
        raise ValueError('Auxiliary actors: ' + '; '.join(auxiliary['errors']))
    speeds = np.linalg.norm(states['velocity'][..., :2], axis=-1)
    acceleration = np.diff(speeds, axis=0)/dt[:, None]
    if not np.isfinite(acceleration).all():
        raise ValueError('Nonfinite acceleration')
    locations = np.argwhere(np.abs(acceleration) > 8.+1e-9)
    violations = []
    for index, actor in locations:
        violations.append({'role': config['vehicle_roles'][actor],
            'interval_s': states['time_s'][index:index+2].tolist(),
            'speed_m_s': speeds[index:index+2, actor].tolist(),
            'signed_acceleration_m_s2': float(acceleration[index, actor]),
            'control': states['control'][index, actor].tolist()})
    if not isinstance(meta.get('collisions'), list):
        raise ValueError('Missing collision records')
    result = {'states_sha256': hashlib.sha256(state_path.read_bytes()).hexdigest(),
        'source_sha256': meta['code_sha256'], 'capture_mode': meta.get('capture_mode'),
        'auxiliary_actors': auxiliary,
        'peak_acceleration_m_s2': float(max(0., acceleration.max())),
        'peak_deceleration_m_s2': float(max(0., -acceleration.min())),
        'threshold_violations': violations, 'collision_records': meta['collisions'],
        'collision_record_count': len(meta['collisions']), 'lane_change': None}
    if lane_direction(config, folder.name) is not None:
        geometry = trajectory_audit(states, config, folder.name)
        before = precontact_geometry(geometry, meta['collisions'])
        result['lane_change'] = {'completed': geometry['completed'], 'clear_cutin': geometry['clear_cutin'],
            'final_source_route_offset_m': geometry['final_source_route_offset_m'],
            'precontact_geometry': before}
    contacts = meta['collisions']
    if any(not isinstance(event.get('time_s'), (float, int)) or not np.isfinite(event['time_s']) for event in contacts):
        raise ValueError('Collision events require finite intervention-relative time_s')
    initialization_contacts = [event for event in contacts if event['time_s'] <= 1e-9]
    first_contact = min((event['time_s'] for event in contacts), default=float('inf'))
    precontact_violations = [row for row in violations if row['interval_s'][1] < first_contact-1e-9]
    result['initialization_collisions'] = initialization_contacts
    result['post_intervention_collisions'] = [event for event in contacts if event['time_s'] > 1e-9]
    result['precontact_dynamics_violations'] = precontact_violations
    result['strict_no_contact_complete_lane_passed'] = bool(not violations and not contacts and
        (result['lane_change'] is None or result['lane_change']['completed']))
    result['acceptance_policy'] = ('Reject initialization contacts or precontact dynamics violations. '
        'Retain post-intervention contact outcomes. Lane completion is reported separately from control execution.')
    result['passed'] = not initialization_contacts and not precontact_violations
    return result, states


def replay_check(states_by_branch):
    """Use existing review tolerances, never a maximum mixing physical units."""
    branches = list(states_by_branch)
    base = states_by_branch['maintain']
    shared = {b: physical_differences(states_by_branch[b], base, history=True) for b in branches}
    repeat_states = states_by_branch['maintain_repeat']
    repeat = physical_differences(repeat_states, base)
    checks = {}; violations = []
    for branch in branches:
        states = states_by_branch[branch]
        checks[branch] = {}
        for scope, indices in [('history_and_t0', np.arange(41)), ('t0', [40])]:
            values = state_errors(states, base, indices)
            checks[branch][scope] = values
            violations.extend({'branch': branch, 'scope': scope, 'metric': key,
                'value': value, 'tolerance': INITIAL_TOLERANCES[key]}
                for key, value in values.items() if value > INITIAL_TOLERANCES[key])
    repeat_xy = float(np.linalg.norm(repeat_states['position'][..., :2] - base['position'][..., :2], axis=-1).max())
    return {'acceptance_method': 'review_common.state_errors and INITIAL_TOLERANCES; repeat XY <= 0.05m',
        'predeclared_initial_tolerances': INITIAL_TOLERANCES,
        'state_errors_by_branch': checks, 'initial_tolerance_violations': violations,
        'repeat_full_sequence_max_xy_m': repeat_xy, 'repeat_xy_tolerance_m': .05,
        'history_and_t0_component_max_abs_differences_diagnostic_only': shared,
        'repeat_component_max_abs_differences_diagnostic_only': repeat,
        'passed': not violations and repeat_xy <= .05}


def audit_progress(data, branch_cache=None, history_cache=None):
    data = Path(data); manifest = json.loads((data/'manifest.json').read_text())
    if manifest.get('capture_mode') != 'physics_only_preflight_not_delivery':
        raise ValueError('This progress checker is only for explicitly marked physics preflight')
    branch_cache = {} if branch_cache is None else branch_cache
    history_cache = {} if history_cache is None else history_cache
    scenes = {}; requested_by_scene = {}; planned_by_scene = {}
    observed = completed_scenes = complete_protocol_scenes = 0; failed = []
    for name in manifest['scene_ids']:
        scene = data/name; cfg = json.loads((scene/'spec.json').read_text())
        planned = list(branch_names(cfg)); planned_by_scene[name] = planned
        requested = manifest.get('requested_scene_branches', {}).get(name, manifest.get('requested_branches'))
        branches = list(planned if requested is None else requested)
        if not branches or len(set(branches)) != len(branches) or not set(branches) <= set(planned):
            raise ValueError(f'{name}: requested branches must be distinct feasible configured actions')
        if name in manifest.get('scene_branches', {}) and manifest['scene_branches'][name] != planned:
            raise ValueError(f'{name}: manifest scene branches differ from config')
        requested_by_scene[name] = branches
        row = {'planned_branches': planned, 'requested_branches': branches,
               'branches': {}, 'history_and_repeat': None}
        for branch in branches:
            folder = scene/'branches'/branch; meta_path = folder/'meta.json'
            result = {'status': 'missing'}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta['status'] == 'complete':
                    observed += 1
                    try:
                        fingerprint = tuple((p.stat().st_size, p.stat().st_mtime_ns) if p.is_file() else None for p in
                                            (scene/'spec.json', meta_path, folder/'states.npz',
                                             folder/'pedestrians.npz',folder/'pedestrian_decisions.json',folder/'signal_states.json'))
                        cached = branch_cache.get((name, branch))
                        if cached is None or cached[0].get('input_stat_fingerprint') != fingerprint:
                            branch_cache[name, branch] = branch_check(folder, cfg, meta)
                            branch_cache[name, branch][0]['input_stat_fingerprint'] = fingerprint
                            history_cache.pop(name, None)
                        result = branch_cache[name, branch][0]
                        result['status'] = 'passed_quick_checks' if result['passed'] else 'failed_quick_checks'
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        branch_cache.pop((name, branch), None); history_cache.pop(name, None)
                        result = {'status': 'failed_quick_checks', 'passed': False, 'error': str(exc)}
                    if not result.get('passed'):
                        failed.append(f'{name}/{branch}')
                else:
                    branch_cache.pop((name, branch), None); history_cache.pop(name, None)
                    result = {'status': meta['status']}
                    if meta['status'] == 'failed':
                        failed.append(f'{name}/{branch}')
            else:
                branch_cache.pop((name, branch), None); history_cache.pop(name, None)
            row['branches'][branch] = result
        if all((name, branch) in branch_cache for branch in branches):
            completed_scenes += 1
            if {'maintain', 'maintain_repeat'} <= set(branches):
                if name in history_cache and set(history_cache[name]['state_errors_by_branch']) != set(branches):
                    history_cache.pop(name)
                if name not in history_cache:
                    history_cache[name] = replay_check({branch: branch_cache[name, branch][1] for branch in branches})
                row['history_and_repeat'] = history_cache[name]
                if not history_cache[name]['passed']:
                    failed.append(f'{name}/common_history_or_repeat')
                auxiliary = {branch: compare_auxiliary(scene/'branches'/branch,scene/'branches/maintain',cfg,
                             scope='repeat' if branch=='maintain_repeat' else 'history') for branch in branches}
                row['auxiliary_common_history_and_repeat'] = auxiliary
                if any(record['status']!='passed' for record in auxiliary.values()):
                    failed.append(f'{name}/auxiliary_common_history_or_repeat')
                if set(branches) == set(planned):
                    complete_protocol_scenes += 1
            else:
                row['history_and_repeat_not_evaluated'] = 'Requested subset omits maintain or maintain_repeat'
        scenes[name] = row
    expected = sum(map(len, requested_by_scene.values()))
    expected_full = sum(map(len, planned_by_scene.values()))
    full_protocol = complete_protocol_scenes == len(manifest['scene_ids']) and bool(scenes)
    selected_complete = observed == expected and completed_scenes == len(scenes) and bool(scenes)
    status = 'failed' if failed else 'incomplete'
    if not failed and selected_complete:
        status = 'passed_quick_checks' if full_protocol else 'passed_requested_scope_checks'
    return {'kind': 'incremental_physics_preflight_checks', 'capture_mode': manifest['capture_mode'],
        'delivery_eligible': False, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'data': str(data.resolve()), 'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'status': status,
        'summary': {'expected_branches': expected, 'observed_complete_branches': observed,
                    'full_protocol_expected_branches': expected_full,
                    'full_protocol_coverage_complete': full_protocol,
                    'requested_scene_branches': requested_by_scene,
                    'expected_scenes': len(manifest['scene_ids']), 'complete_scenes': completed_scenes,
                    'failed_checks': failed}, 'scenes': scenes,
        'limits': ['This incremental report does not replace final independent map/physics-deployment/NPC-response audits.',
                   'No RGB was collected; passing physics preflight is not formal dataset delivery.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.data.resolve()):
        parser.error('Derived progress report must remain outside the read-only dataset')
    cache, histories, announced = {}, {}, set()
    while True:
        report = audit_progress(args.data, cache, histories)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n'); temporary.replace(args.out)
        for name, row in report['scenes'].items():
            if row['history_and_repeat'] and name not in announced:
                announced.add(name)
                print(json.dumps({'scene': name,
                    'failures': [v for v in report['summary']['failed_checks'] if v.startswith(name+'/')],
                    'contacts': sum(v.get('collision_record_count', 0) for v in row['branches'].values()),
                    'maximum_acceleration_m_s2': max(v['peak_acceleration_m_s2'] for v in row['branches'].values()),
                    'maximum_deceleration_m_s2': max(v['peak_deceleration_m_s2'] for v in row['branches'].values()),
                    'initial_tolerance_violations': row['history_and_repeat']['initial_tolerance_violations'],
                    'repeat_full_sequence_max_xy_m': row['history_and_repeat']['repeat_full_sequence_max_xy_m']}), flush=True)
        for failure in report['summary']['failed_checks']:
            if failure not in announced:
                announced.add(failure); print(json.dumps({'FAILURE': failure}), flush=True)
        manifest = json.loads((args.data/'manifest.json').read_text())
        if not args.watch or manifest['status'] != 'running':
            print(json.dumps({'status': report['status'], 'summary': report['summary']}), flush=True)
            return 1 if report['status'] == 'failed' else 0
        time.sleep(2)


if __name__ == '__main__':
    raise SystemExit(main())
