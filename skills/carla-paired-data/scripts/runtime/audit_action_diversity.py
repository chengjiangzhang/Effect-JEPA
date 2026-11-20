#!/usr/bin/env python3
"""Read-only realized-action differences and recorded vehicle-box geometry.

This reports measured differences and a separately identified numerical ego-motion
duplicate selection. It never filters by NPC response or reconstructs missing
settled observations, and it does not connect to CARLA.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

if __package__:
    from .controllers_v2 import branch_names
else:
    from controllers_v2 import branch_names
if __package__:
    from .vehicle_geometry import ego_bbox, rotation, box_sat_gap, BOX_ROUNDOFF_M
else:
    from vehicle_geometry import ego_bbox, rotation, box_sat_gap, BOX_ROUNDOFF_M
if __package__:
    from .audit_preflight_progress import replay_check
    from .audit_realized_actions import select_branches
else:
    from audit_preflight_progress import replay_check
    from audit_realized_actions import select_branches


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_states(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}



def snapshot_geometry(state, index, roles):
    centers, corners = zip(*(ego_bbox(state, i) for i in range(len(roles))))
    pairs = []
    for i, j in itertools.combinations(range(len(roles)), 2):
        gap = box_sat_gap(corners[i][index], corners[j][index])
        classification = ('separated' if gap > BOX_ROUNDOFF_M else
                          'interior_overlap' if gap < -BOX_ROUNDOFF_M else 'touching_or_roundoff')
        pairs.append({'roles': [roles[i], roles[j]], 'sat_gap_m': gap, 'classification': classification,
                      'bbox_center_xy_distance_m': float(np.linalg.norm(centers[j][index, :2]-centers[i][index, :2]))})
    return {'time_s': float(state['time_s'][index]), 'frame_id': int(state['frame_id'][index]),
            'method': '15-axis 3D SAT using measured vehicle transforms and local bbox offset/rotation/extents',
            'numerical_boundary_roundoff_m': BOX_ROUNDOFF_M, 'pairs': pairs,
            'interior_overlap_pairs': [p['roles'] for p in pairs if p['classification'] == 'interior_overlap'],
            'minimum_sat_gap_m': min((p['sat_gap_m'] for p in pairs), default=None)}


def planned_spawn_geometry(config, state):
    """Config poses plus actual actor-box dimensions; explicitly not observations."""
    transforms = []
    for spec in config['vehicles']:
        pose = spec['transform']; matrix = np.eye(4)
        matrix[:3, :3] = rotation(pose['rotation_deg']); matrix[:3, 3] = pose['position']
        transforms.append(matrix)
    planned = {key: state[key] for key in ('bbox_extent', 'bbox_offset', 'bbox_rotation_deg')}
    planned.update(transform_world_vehicle=np.asarray(transforms)[None], time_s=np.array([0.]), frame_id=np.array([-1]))
    result = snapshot_geometry(planned, 0, config['vehicle_roles'])
    result.pop('time_s'); result.pop('frame_id')
    result['method'] = 'Configured requested spawn transforms plus recorded actor-local boxes; not measured spawn or settled poses'
    result['observed'] = False
    return result


def collision_phases(events, history_start):
    result = {'before_recorded_history': [], 'recorded_history_before_t0': [], 't0': [], 'future': []}
    for event in events:
        t = float(event['time_s'])
        if not np.isfinite(t):
            raise ValueError('Collision time must be finite')
        key = ('before_recorded_history' if t < history_start-1e-8 else
               'recorded_history_before_t0' if t < -1e-8 else 't0' if abs(t) <= 1e-8 else 'future')
        result[key].append(event)
    return result


def pair_metrics(a, b, anchor, events_a=(), events_b=()):
    """B minus A, strict future states; controls start in [0,4) and exclude sentinel."""
    t = a['time_s']
    np.testing.assert_array_equal(t, b['time_s'])
    pa = a['position'][:, 0]@anchor[:3, :3].T+anchor[:3, 3]
    pb = b['position'][:, 0]@anchor[:3, :3].T+anchor[:3, 3]
    delta_xy = pb[:, :2]-pa[:, :2]
    distance = np.linalg.norm(delta_xy, axis=1)
    delta_speed = np.linalg.norm(b['velocity'][:, 0, :2], axis=1)-np.linalg.norm(a['velocity'][:, 0, :2], axis=1)
    delta_control = b['control'][:, 0]-a['control'][:, 0]
    future = t > 1e-8
    controls = (t >= -1e-8) & (np.arange(len(t)) < len(t)-1)
    first_contact = min((float(e['time_s']) for e in [*events_a, *events_b]), default=None)

    def measured(state_mask, control_mask):
        ids, ci = np.flatnonzero(state_mask), np.flatnonzero(control_mask)
        result = {'future_state_count': len(ids), 'applied_future_control_count': len(ci),
                  'state': None, 'control': None}
        if len(ids):
            peak = ids[int(np.argmax(distance[ids]))]
            result['state'] = {'max_ego_xy_difference_m': float(distance[peak]), 'max_xy_time_s': float(t[peak]),
                'rms_ego_xy_difference_m': float(np.sqrt(np.mean(distance[ids]**2))),
                'last_ego_xy_difference_m': float(distance[ids[-1]]),
                'last_ego_delta_xy_B_minus_A_m': delta_xy[ids[-1]].tolist(),
                'max_absolute_speed_difference_m_s': float(np.abs(delta_speed[ids]).max()),
                'rms_speed_difference_m_s': float(np.sqrt(np.mean(delta_speed[ids]**2))),
                'last_speed_difference_B_minus_A_m_s': float(delta_speed[ids[-1]]),
                'numerically_identical_ego_xy': bool(np.array_equal(pa[ids, :2], pb[ids, :2])),
                'numerically_identical_ego_speed': bool(np.all(delta_speed[ids] == 0.))}
        if len(ci):
            result['control'] = {'channels': ['throttle', 'steer', 'brake'],
                'max_abs_difference_by_channel': np.abs(delta_control[ci]).max(axis=0).tolist(),
                'rms_difference_by_channel': np.sqrt(np.mean(delta_control[ci]**2, axis=0)).tolist(),
                'absolute_difference_time_integral_by_channel_s':
                    np.sum(np.abs(delta_control[ci])*np.diff(t)[ci, None], axis=0).tolist(),
                'numerically_identical_applied_controls': bool(np.all(delta_control[ci] == 0.))}
        return result

    pre_state, pre_control = future.copy(), controls.copy()
    if first_contact is not None:
        pre_state &= t < first_contact-1e-9
        pre_control &= np.r_[t[1:], np.inf] < first_contact-1e-9
    return {'all_future': measured(future, controls),
            'strictly_precontact': measured(pre_state, pre_control),
            'first_contact_across_pair_s': first_contact,
            'future_series': {'time_s': t[future].tolist(), 'ego_xy_difference_m': distance[future].tolist(),
                              'speed_difference_B_minus_A_m_s': delta_speed[future].tolist()},
            'definition': 'All future samples retained, including after contact; precontact control intervals end strictly before first callback time.'}


def scene_audit(scene, config):
    roles = config['vehicle_roles']; branches = list(branch_names(config)); actions = list(branch_names(config, False))
    if len(actions) != 5 or len(branches) != 6:
        raise ValueError('Require five configured actions plus maintain_repeat')
    if not roles or roles[0] != 'ego' or len(set(roles)) != len(roles) or roles != [v['role'] for v in config['vehicles']]:
        raise ValueError('Unique actor roles with ego at index0 are required')
    states, metadata, sources = {}, {}, {}
    for branch in branches:
        folder = scene/'branches'/branch; meta_path = folder/'meta.json'
        meta = json.loads(meta_path.read_text())
        if meta.get('status') != 'complete':
            raise ValueError(f'{branch}: only complete metadata may be read')
        if (meta.get('branch') != branch or meta.get('vehicle_roles') != roles or not config.get('spec_sha256')
                or meta.get('spec_sha256') != config['spec_sha256'] or not config.get('code_sha256')
                or meta.get('code_sha256') != config['code_sha256'] or not isinstance(meta.get('collisions'), list)):
            raise ValueError(f'{branch}: source/config/role identity or collision record mismatch')
        state = read_states(folder/'states.npz')
        if state['time_s'].shape != (121,) or not np.allclose(state['time_s'], np.arange(121)*.05-2, atol=1e-8, rtol=0):
            raise ValueError(f'{branch}: expected original -2..4s .05s timeline')
        if state['position'].shape != (121, len(roles), 3):
            raise ValueError(f'{branch}: actor/time coverage mismatch')
        states[branch], metadata[branch] = state, meta
        sources[branch] = {'meta_sha256': digest(meta_path), 'states_sha256': digest(folder/'states.npz')}
    replay = replay_check(states)  # Existing per-unit history/repeat criteria; no new tolerances.
    base = states['maintain']; t0 = int(np.argmin(abs(base['time_s'])))
    anchor = np.linalg.inv(base['transform_world_vehicle'][t0, 0])
    records = {}
    for branch, state in states.items():
        meta = metadata[branch]
        trace = meta.get('initialization_frame_trace', {})
        settled_frame = trace.get('after_braked_settle_frame')
        settled_indices = np.flatnonzero(state['frame_id'] == settled_frame) if settled_frame is not None else []
        settled = (dict(snapshot_geometry(state, int(settled_indices[0]), roles), observed=True)
                   if len(settled_indices) else {'observed': False, 'frame_id': settled_frame,
                    'reason': 'No settled vehicle pose stored; initialization frame IDs alone cannot establish box geometry.'})
        common = state['position']@anchor[:3, :3].T+anchor[:3, 3]
        observed_t0 = []
        for i, role in enumerate(roles):
            observed_t0.append({'role': role, 'position_in_maintain_t0_frame_m': common[t0, i].tolist(),
                'world_heading_deg': float(state['rotation_deg'][t0, i, 2]),
                'heading_relative_to_maintain_ego_deg': float((state['rotation_deg'][t0, i, 2]-base['rotation_deg'][t0, 0, 2]+180)%360-180),
                'world_xy_speed_m_s': float(np.linalg.norm(state['velocity'][t0, i, :2])),
                'bbox_extent_m': state['bbox_extent'][i].tolist()})
        records[branch] = {'settled_geometry': settled,
            'history_start_geometry': snapshot_geometry(state, 0, roles),
            't0_geometry': snapshot_geometry(state, t0, roles),
            'observed_t0_vehicles': observed_t0,
            'collision_callbacks_by_phase': collision_phases(meta['collisions'], state['time_s'][0]),
            'collision_callback_count': len(meta['collisions']),
            'ego_future': {'time_s': state['time_s'][t0+1:].tolist(),
                'position_in_maintain_t0_frame_m': common[t0+1:, 0].tolist(),
                'world_xy_speed_m_s': np.linalg.norm(state['velocity'][t0+1:, 0, :2], axis=-1).tolist(),
                'applied_control_start_time_s': state['time_s'][t0:-1].tolist(),
                'applied_throttle_steer_brake': state['control'][t0:-1, 0].tolist()}}
    pairs = [{'actions': [a, b], **pair_metrics(states[a], states[b], anchor,
        metadata[a]['collisions'], metadata[b]['collisions'])} for a, b in itertools.combinations(actions, 2)]
    repeat = pair_metrics(states['maintain'], states['maintain_repeat'], anchor,
                          metadata['maintain']['collisions'], metadata['maintain_repeat']['collisions'])
    return {'scene': scene.name, 'map': config.get('map'), 'family': config.get('family'),
        'config_file_sha256': digest(scene/'spec.json'), 'code_sha256': config['code_sha256'],
        'source_files': sources, 'actions': actions, 'branches': branches, 'vehicle_roles': roles,
        'action_profiles': config.get('action', {}).get('profiles'), 'history_and_repeat': replay,
        'model_branch_selection': select_branches(states, actions),
        'planned_spawn_geometry': planned_spawn_geometry(config, base), 'branch_records': records,
        'pairwise_action_comparisons': pairs, 'repeat_comparison': repeat,
        'summary': {'pair_count': len(pairs),
            'max_xy_difference_range_m': [min(p['all_future']['state']['max_ego_xy_difference_m'] for p in pairs),
                                        max(p['all_future']['state']['max_ego_xy_difference_m'] for p in pairs)],
            'max_speed_difference_range_m_s': [min(p['all_future']['state']['max_absolute_speed_difference_m_s'] for p in pairs),
                                              max(p['all_future']['state']['max_absolute_speed_difference_m_s'] for p in pairs)],
            'exactly_identical_future_xy_action_pairs': [p['actions'] for p in pairs if p['all_future']['state']['numerically_identical_ego_xy']],
            'collision_callback_count': sum(r['collision_callback_count'] for r in records.values()),
            'history_start_overlap_branches': [b for b, r in records.items() if r['history_start_geometry']['interior_overlap_pairs']],
            't0_overlap_branches': [b for b, r in records.items() if r['t0_geometry']['interior_overlap_pairs']],
            'settled_pose_observed_branches': sum(r['settled_geometry']['observed'] for r in records.values())}}


def audit_dataset(data, cache=None):
    data = Path(data); manifest_path = data/'manifest.json'; manifest = json.loads(manifest_path.read_text())
    cache = {} if cache is None else cache
    scenes, pending, errors, expected_actions = {}, {}, [], {}
    observed = 0
    for name in manifest['scene_ids']:
        try:
            scene = data/name; cfg = json.loads((scene/'spec.json').read_text()); branches = list(branch_names(cfg))
            expected_actions[name] = branches
            if name in manifest.get('scene_branches', {}) and manifest['scene_branches'][name] != branches:
                raise ValueError('Manifest/config action mismatch')
            incomplete = []
            for branch in branches:
                path = scene/'branches'/branch/'meta.json'
                status = json.loads(path.read_text()).get('status') if path.is_file() else 'missing'
                if status == 'complete':
                    observed += 1
                else:
                    incomplete.append({'branch': branch, 'status': status})
            if incomplete:
                pending[name] = incomplete; continue
            files = [scene/'spec.json', *(scene/'branches'/b/f for b in branches for f in ('meta.json', 'states.npz'))]
            fingerprint = tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in files)
            if name not in cache or cache[name][0] != fingerprint:
                cache[name] = (fingerprint, scene_audit(scene, cfg))
            scenes[name] = cache[name][1]
        except (OSError, ValueError, KeyError, TypeError, IndexError, AssertionError) as exc:
            errors.append(f'{name}: {exc}')
    complete = bool(scenes) and len(scenes) == len(manifest['scene_ids']) and not errors and not pending
    return {'kind': 'realized_action_differences_and_initial_box_geometry',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'data': str(data.resolve()),
        'capture_mode': manifest.get('capture_mode'), 'delivery_eligible': False,
        'manifest_sha256': digest(manifest_path), 'audit_sha256': digest(__file__),
        'helper_sha256': {n: digest(Path(__file__).parent/n) for n in ('controllers_v2.py', 'ego_decisions.py', 'audit_lane_change.py', 'audit_preflight_progress.py', 'audit_realized_actions.py')},
        'status': 'complete_measurements' if complete else 'failed' if errors else 'incomplete',
        'coverage_complete': complete, 'expected_scenes': len(manifest['scene_ids']), 'audited_scenes': len(scenes),
        'expected_branches': sum(map(len, expected_actions.values())), 'observed_complete_metadata_branches': observed,
        'pairwise_future_action_comparisons': sum(s['summary']['pair_count'] for s in scenes.values()),
        'errors': errors, 'pending': pending, 'scenes': scenes,
        'limits': ['Pairwise differences are measurements; the separately reported model_branch_selection applies predeclared bounded repeat-aware numerical duplicate tolerances to ego motion only.',
            'SAT uses complete 3D oriented boxes; box overlap is not necessarily a mesh collision. Collision callbacks remain separate and are not deduplicated across actor sensors.',
            'Config spawn geometry is planned, history-start/t0 geometry is observed, and settling is unverified when its pose was never stored.',
            'Only configured dynamic actors are in the box-pair audit; scenery/parked-mesh contacts are represented only by collision callbacks.',
            'All future differences remain available after contact; strict precontact metrics are additional, never a filter that removes failed branches.',
            'No future state, action or initialization history was modified; this is not proof of world-model action understanding.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--watch', action='store_true', help='Write partial report while collecting, final report only with complete coverage')
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.data.resolve()):
        parser.error('Audit output must stay outside read-only data')
    cache, announced = {}, set()
    while True:
        result = audit_dataset(args.data, cache)
        for name, scene in result['scenes'].items():
            if name not in announced:
                print(json.dumps({'scene': name, **scene['summary']}, ensure_ascii=False), flush=True)
                announced.add(name)
        target = args.out if result['coverage_complete'] else args.out.with_name(args.out.stem+'.partial.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.json.tmp'); temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n'); temporary.replace(target)
        manifest = json.loads((args.data/'manifest.json').read_text())
        if result['coverage_complete'] or not args.watch or manifest.get('status') not in ('running', 'prepared'):
            print(json.dumps({k: result[k] for k in ('status', 'audited_scenes', 'expected_scenes', 'pairwise_future_action_comparisons', 'errors')}), flush=True)
            return 0 if result['coverage_complete'] else 1
        time.sleep(20)


if __name__ == '__main__':
    raise SystemExit(main())
