#!/usr/bin/env python3
"""Describe realized ego motions while retaining every paired branch.

Legacy numerical duplicate diagnostics retain every branch, including all
post-contact observations. New specs also enforce the frozen ten-pair action
separation contract using pre-contact ego evidence. No NPC response magnitude
is used as a gate; execution and repetition checks remain independent.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

if __package__:
    from .controllers_v2 import branch_names
    from .ego_decisions import longitudinal_target
    from .action_separation import audit_separation
else:
    from controllers_v2 import branch_names
    from ego_decisions import longitudinal_target
    from action_separation import audit_separation


POLICY_VERSION = 'ego-motion-diagnostic-all-branches-v2'
BASE_TOLERANCES = {'position_m': .02, 'speed_m_s': .02, 'yaw_deg': .1}
MAX_TOLERANCES = {'position_m': .05, 'speed_m_s': .05, 'yaw_deg': .5}
REPEAT_MULTIPLIER = 3.
STATIONARY_SPEED_M_S = .15


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ego_series(state, ego_index):
    times = np.asarray(state['time_s'], float)
    if (times.ndim != 1 or len(times) < 3 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or not np.any(np.abs(times) <= 1e-8)
            or not np.any(times > 1e-8)):
        raise ValueError('Strictly increasing finite timeline with t0 and future is required')
    values = {}
    for name in ('position', 'velocity', 'rotation_deg', 'control'):
        array = np.asarray(state[name], float)
        if (array.ndim != 3 or array.shape[0] != len(times) or array.shape[2] != 3
                or not 0 <= ego_index < array.shape[1] or not np.isfinite(array).all()):
            raise ValueError(f'Invalid ego observation array: {name}')
        values[name] = array[:, ego_index]
    values['time_s'] = times
    values['speed'] = np.linalg.norm(values['velocity'], axis=-1)
    return values


def motion_metrics(a, b, ego_index=0, *, history=False):
    """Compare synchronized world-frame ego samples (no branch-wise recentering)."""
    a, b = _ego_series(a, ego_index), _ego_series(b, ego_index)
    if not np.array_equal(a['time_s'], b['time_s']):
        raise ValueError('Branch timelines differ; no interpolation is performed')
    times = a['time_s']
    mask = times <= 1e-8 if history else times > 1e-8
    if not mask.any():
        raise ValueError('Requested motion comparison has no samples')
    distance = np.linalg.norm(a['position']-b['position'], axis=-1)[mask]
    speed = np.abs(a['speed']-b['speed'])[mask]
    yaw = np.abs((a['rotation_deg'][:, 2]-b['rotation_deg'][:, 2]+180.) % 360.-180.)[mask]
    velocity = np.linalg.norm(a['velocity']-b['velocity'], axis=-1)[mask]
    # Recorded control[t] applies over [t,t+dt); final control is a sentinel.
    control_mask = (times < -1e-8 if history else times >= -1e-8) & (np.arange(len(times)) < len(times)-1)
    control = np.abs(a['control']-b['control'])[control_mask]
    return {'position_m': float(distance.max()), 'speed_m_s': float(speed.max()),
            'yaw_deg': float(yaw.max()), 'velocity_vector_difference_m_s': float(velocity.max()),
            'position_rms_m': float(np.sqrt(np.mean(distance**2))),
            'speed_rms_m_s': float(np.sqrt(np.mean(speed**2))),
            'last_position_difference_m': float(distance[-1]),
            'state_sample_count': int(mask.sum()), 'control_interval_count': int(control_mask.sum()),
            'max_abs_control_difference_by_channel': control.max(0).tolist() if len(control) else [0., 0., 0.],
            'identical_ego_control_sequence': bool(not np.any(control)),
            'control_channels': ['throttle', 'steer', 'brake']}


def motion_summary(state, ego_index=0, profile=None):
    """Actual motion in the observation window; ideal commands are separate."""
    values = _ego_series(state, ego_index); times = values['time_s']
    t0 = int(np.flatnonzero(np.abs(times) <= 1e-8)[0]); future = times > 1e-8
    elapsed, speed = times[t0:], values['speed'][t0:]
    stopped = speed <= STATIONARY_SPEED_M_S
    first = np.flatnonzero(stopped)
    longest, final_duration, start = 0., 0., None
    for t, stationary in zip(elapsed, stopped):
        if stationary:
            start = float(t) if start is None else start
            final_duration = float(t)-start; longest = max(longest, final_duration)
        else:
            start = None; final_duration = 0.
    base_xy = float(np.linalg.norm(values['velocity'][t0, :2]))
    result = {'measured_t0_speed_m_s': float(values['speed'][t0]),
              'controller_t0_speed_xy_m_s': base_xy,
              'future_min_speed_m_s': float(values['speed'][future].min()),
              'future_max_speed_m_s': float(values['speed'][future].max()),
              'final_speed_m_s': float(speed[-1]),
              'travelled_distance_3d_m': float(np.linalg.norm(np.diff(values['position'][t0:], axis=0), axis=-1).sum()),
              'displacement_from_t0_3d_m': float(np.linalg.norm(values['position'][-1]-values['position'][t0])),
              'stationary_speed_threshold_m_s': STATIONARY_SPEED_M_S,
              'first_stationary_time_s': float(elapsed[first[0]]) if len(first) else None,
              'maximum_continuous_stationary_s': longest, 'final_continuous_stationary_s': final_duration,
              'future_window_s': [float(elapsed[0]), float(elapsed[-1])],
              'speed_definition': '3D velocity norm; controller target anchoring separately uses XY norm',
              'ideal_command_diagnostic': None}
    if profile is not None and profile['kind'] in ('longitudinal', 'maintain', 'lane_change'):
        targets = np.array([longitudinal_target(profile, time, base_xy) for time in elapsed])
        zero = np.flatnonzero(targets <= .01)
        result['ideal_command_diagnostic'] = {'kind': profile['kind'], 'time_s': elapsed.tolist(),
            'target_speed_m_s': targets.tolist(), 'target_final_speed_m_s': float(targets[-1]),
            'first_zero_target_time_s': float(elapsed[zero[0]]) if len(zero) else None,
            'definition': 'Declared target schedule anchored to measured t0 XY speed; not a claim of realized ideal dynamics'}
    elif profile is not None:
        result['ideal_command_diagnostic'] = {'kind': profile['kind'],
            'definition': 'State/event-dependent policy; target schedule is not inferred from an ideal constant acceleration'}
    return result


def audit_motion_similarity(states_by_branch, planned_branches, ego_index=0):
    """Diagnose complete-link motion groups; retain all planned conditions.

    A candidate joins a group only when it is within tolerance of every existing
    member. This prevents A~B~C from silently merging A and C when A !~ C.
    Tolerances are fixed before capture and may only increase within hard caps
    to account for the independently captured maintain repeat.
    equivalent_motion_count counts non-representative conditions, not pairs.
    """
    planned = list(planned_branches)
    if (len(planned) != 5 or len(set(planned)) != 5 or 'maintain' not in planned
            or 'maintain_repeat' in planned):
        raise ValueError('Require five planned distinct conditions including maintain, excluding QA repeat')
    if set(states_by_branch) != set(planned) | {'maintain_repeat'}:
        raise ValueError('Require all five planned branch observations and maintain_repeat')
    order = ['maintain', *(branch for branch in planned if branch != 'maintain')]
    repeat = motion_metrics(states_by_branch['maintain'], states_by_branch['maintain_repeat'], ego_index)
    required = {key: max(value, REPEAT_MULTIPLIER*repeat[key]) for key, value in BASE_TOLERANCES.items()}
    effective = {key: min(value, MAX_TOLERANCES[key]) for key, value in required.items()}
    noise_errors = [f'{key}: required duplicate tolerance {value:.8g} exceeds hard cap {MAX_TOLERANCES[key]:.8g}'
                    for key, value in required.items() if value > MAX_TOLERANCES[key]+1e-12]
    history = {branch: motion_metrics(states_by_branch['maintain'], states_by_branch[branch], ego_index, history=True)
               for branch in [*order, 'maintain_repeat']}
    history_errors = [f'{branch}: history/t0 {key} differs by {value:.8g}, tolerance {BASE_TOLERANCES[key]:.8g}'
                      for branch, row in history.items() for key, value in row.items()
                      if key in BASE_TOLERANCES and value > BASE_TOLERANCES[key]+1e-12]
    errors = noise_errors+history_errors
    pairs, lookup = [], {}
    for a, b in itertools.combinations(order, 2):
        measured = motion_metrics(states_by_branch[a], states_by_branch[b], ego_index)
        equivalent = all(measured[key] <= value+1e-12 for key, value in effective.items())
        record = {'branches': [a, b], 'metrics': measured, 'equivalent_motion': equivalent,
                  'motion_difference_despite_identical_controls': bool(not equivalent and measured['identical_ego_control_sequence'])}
        pairs.append(record); lookup[frozenset((a, b))] = record
    clusters = []
    if not errors:
        for branch in order:
            for cluster in clusters:
                if all(lookup[frozenset((branch, other))]['equivalent_motion'] for other in cluster):
                    cluster.append(branch)
                    break
            else:
                clusters.append([branch])
    representatives = [cluster[0] for cluster in clusters]
    similar = {branch: {'representative': cluster[0],
                         'metrics': lookup[frozenset((branch, cluster[0]))]['metrics'],
                         'reason': 'Full future ego motion is numerically equivalent; raw capture retained'}
                for cluster in clusters for branch in cluster[1:]}
    return {'status': 'failed' if errors else 'passed', 'errors': errors,
            'policy_version': POLICY_VERSION,
            'scope': 'Numerical within-scene ego-motion duplicates; not controller or interaction validation',
            'model_branches': planned,
            'distinct_motion_representatives': representatives,
            'similar_motion_branches': similar, 'qa_only_branches': ['maintain_repeat'],
            'raw_branches_retained': [*planned, 'maintain_repeat'],
            'distinct_motion_count': None if errors else len(representatives),
            'equivalent_motion_count': None if errors else len(planned)-len(representatives),
            'motion_audit_valid': not errors, 'duplicate_clusters': clusters,
            'tolerances': {'base': dict(BASE_TOLERANCES), 'hard_maximum': dict(MAX_TOLERANCES),
                           'repeat_multiplier': REPEAT_MULTIPLIER, 'required': required,
                           'effective': effective, 'repeat_noise_within_bounds': not noise_errors},
            'repeat_motion_noise': repeat, 'common_history_and_t0': history,
            'realized_motion_by_branch': {branch: motion_summary(states_by_branch[branch], ego_index)
                                          for branch in [*planned, 'maintain_repeat']},
            'pairwise_motion_comparisons': pairs,
            'limits': ['Only ego trajectory positions, speed and wrapped yaw determine duplicates.',
                       'All future samples are used, including post-contact outcomes; endpoints alone do not suffice.',
                       'Equivalent realized motions remain separate branches within the intact paired scene; these are diagnostics only.',
                       'Distinct motion despite identical controls is recorded for execution/repeatability review, not proof of action causality.',
                       'No NPC reaction magnitude or collision-free outcome is used to keep or remove a model branch.',
                       'Grouping is numerical and conservative, not a claim of statistical or causal equivalence.']}


def normalize_motion_report(report):
    """Read older selection reports without discarding historical membership.

    New reports have one model-branch list and diagnostic motion counts. Genuine
    legacy exclusions are preserved separately, never reinterpreted as removal
    decisions made by the current audit.
    """
    result = dict(report)
    branches = result.get('model_branches', result.get('selected_model_branches',
                          result.get('planned_model_branches')))
    if branches is None:
        raise ValueError('Motion report has no model branch list')
    result['model_branches'] = list(branches)
    result.setdefault('motion_audit_valid', result.get('selection_valid',
                      result.get('status') in ('passed', 'accepted')))
    distinct = result.get('unique_condition_count')
    equivalent = result.get('unique_condition_shortfall')
    if distinct is None and equivalent is not None and 'planned_model_branches' in result:
        distinct = len(result['planned_model_branches'])-equivalent
    result.setdefault('distinct_motion_count', distinct)
    result.setdefault('equivalent_motion_count', equivalent)
    planned = result.get('planned_model_branches')
    selected = result.get('selected_model_branches', branches)
    exclusions = result.get('excluded_duplicate_branches', {})
    if exclusions or planned is not None and list(planned) != list(selected):
        result['legacy_branch_selection'] = {
            'planned_model_branches': planned, 'selected_model_branches': list(selected),
            'excluded_duplicate_branches': exclusions}
    for key in ('planned_model_branches', 'selected_model_branches',
                'excluded_duplicate_branches', 'planned_condition_count',
                'unique_condition_count', 'unique_condition_shortfall', 'selection_valid'):
        result.pop(key, None)
    return result


def select_branches(states_by_branch, planned_branches, ego_index=0):
    """Compatibility API for callers using the former selection field names."""
    result = audit_motion_similarity(states_by_branch, planned_branches, ego_index)
    return {**result, 'planned_model_branches': result['model_branches'],
            'selected_model_branches': result['model_branches'] if result['motion_audit_valid'] else [],
            'excluded_duplicate_branches': {}, 'planned_condition_count': len(result['model_branches']),
            'unique_condition_count': result['distinct_motion_count'],
            'unique_condition_shortfall': result['equivalent_motion_count'],
            'selection_valid': result['motion_audit_valid']}


def audit_scene(scene_dir, config=None):
    scene = Path(scene_dir)
    config_path = scene/'spec.json'
    config = json.loads(config_path.read_text()) if config is None else config
    if (config.get('action', {}).get('design_evidence', {}).get('action_design_version') == 'five-effective-ego-actions-v1'
            and 'action_separation_contract' not in config):
        raise ValueError('New action design is missing its required separation contract')
    roles = config['vehicle_roles']
    if (not roles or roles.count('ego') != 1 or len(set(roles)) != len(roles)
            or roles != [vehicle['role'] for vehicle in config['vehicles']]):
        raise ValueError('Frozen role order must uniquely identify ego')
    planned = list(branch_names(config, False)); branches = list(branch_names(config))
    states, sources, decision_summaries, metadata = {}, {}, {}, {}
    for branch in branches:
        folder = scene/'branches'/branch; path = folder/'states.npz'; meta_path = folder/'meta.json'
        meta = json.loads(meta_path.read_text())
        metadata[branch] = meta
        if meta.get('status') != 'complete' or meta.get('branch') != branch or meta.get('vehicle_roles') != roles:
            raise ValueError(f'{branch}: incomplete metadata or role/branch mismatch')
        for key in ('spec_sha256', 'code_sha256'):
            if not config.get(key) or meta.get(key) != config[key]:
                raise ValueError(f'{branch}: frozen {key} identity mismatch')
        with np.load(path, allow_pickle=False) as loaded:
            states[branch] = {key: loaded[key] for key in ('time_s', 'position', 'velocity', 'rotation_deg', 'control')}
        state = states[branch]
        if (state['time_s'].shape != (121,) or not np.allclose(state['time_s'], np.arange(121)*.05-2., atol=1e-8, rtol=0)
                or any(state[key].shape != (121, len(roles), 3) for key in ('position', 'velocity', 'rotation_deg', 'control'))):
            raise ValueError(f'{branch}: require full original -2..4 second 20 Hz timeline and actor coverage')
        sources[branch] = {'states_sha256': digest(path), 'meta_sha256': digest(meta_path)}
        diagnostic_path = folder/'controller_diagnostics.json'
        if diagnostic_path.exists():
            rows = json.loads(diagnostic_path.read_text())
            if not isinstance(rows, list) or len(rows) != len(state['time_s']):
                raise ValueError(f'{branch}: diagnostic timeline does not match recorded states')
            phases, last_phase = [], None
            for time, row in zip(state['time_s'], rows):
                if not isinstance(row, list):
                    raise ValueError(f'{branch}: malformed controller diagnostic row')
                if time < -1e-8:
                    continue
                ego_rows = [actor for actor in row if actor.get('role') == 'ego']
                if config.get('action', {}).get('profile_version') == 2:
                    expected_branch = 'maintain' if branch == 'maintain_repeat' else branch
                    if len(ego_rows) != 1 or ego_rows[0].get('branch') != expected_branch:
                        raise ValueError(f'{branch}: ego decision diagnostic branch mismatch')
                    expected_kind = config['action']['profiles'][expected_branch]['kind']
                    if ego_rows[0].get('kind') != expected_kind:
                        raise ValueError(f'{branch}: ego decision diagnostic policy-kind mismatch')
                if ego_rows:
                    decision = ego_rows[0]; phase = decision.get('phase')
                    if phase != last_phase:
                        phases.append({'time_s': float(time), 'phase': phase}); last_phase = phase
            decision_summaries[branch] = {'phase_transitions': phases, 'observed': True}
            sources[branch]['controller_diagnostics_sha256'] = digest(diagnostic_path)
        elif config.get('action', {}).get('profile_version') == 2:
            raise ValueError(f'{branch}: version2 decision capture requires controller diagnostics')
        else:
            decision_summaries[branch] = {'observed': False, 'reason': 'Legacy capture has no controller decision diagnostics'}
    result = audit_motion_similarity(states, planned, roles.index('ego'))
    if 'action_separation_contract' in config:
        separation = audit_separation(states, planned, metadata,
                                      config['action_separation_contract'], roles.index('ego'))
        result['action_separation'] = separation
        result['errors'].extend(separation['errors'])
        result['status'] = 'failed' if result['errors'] else 'passed'
    profiles = config.get('action', {}).get('profiles', {})
    result['realized_motion_by_branch'] = {branch: motion_summary(state, roles.index('ego'),
        profiles.get('maintain' if branch == 'maintain_repeat' else branch)) for branch, state in states.items()}
    result['ego_decision_diagnostics'] = decision_summaries
    result.update(scene_id=scene.name, config_file_sha256=digest(config_path), sources=sources)
    return result


def audit_dataset(data):
    data = Path(data); manifest_path = data/'manifest.json'
    manifest = json.loads(manifest_path.read_text()); ids = manifest['scene_ids']
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Manifest requires a nonempty distinct scene list')
    scenes, errors = {}, []
    for name in ids:
        try:
            result = audit_scene(data/name)
            declared = manifest.get('scene_branches', {}).get(name)
            if declared is not None and declared != result['raw_branches_retained']:
                raise ValueError('Manifest/config branch order mismatch')
            scenes[name] = result
            if result['status'] != 'passed':
                errors.extend(f'{name}: {error}' for error in result['errors'])
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append(f'{name}: {exc}')
    return {'schema_version': 2, 'kind': 'realized_ego_action_diagnostics',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'data_root': str(data.resolve()),
            'status': 'failed' if errors else 'passed', 'errors': errors, 'scenes': scenes,
            'manifest_sha256': digest(manifest_path), 'audit_sha256': digest(__file__),
            'helper_sha256': {name: digest(Path(__file__).parent/name) for name in ('controllers_v2.py', 'ego_decisions.py', 'action_separation.py')},
            'summary': {'planned_scenes': len(ids), 'audited_scenes': len(scenes),
                        'model_branch_count': sum(len(row['model_branches']) for row in scenes.values()),
                        'scenes_with_equivalent_motion': sum(bool(row['equivalent_motion_count']) for row in scenes.values())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.data.resolve()):
        parser.error('Report must remain outside read-only raw data')
    report = audit_dataset(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix+'.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'); temporary.replace(args.out)
    print(json.dumps({'status': report['status'], **report['summary'], 'errors': report['errors']}, ensure_ascii=False))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
