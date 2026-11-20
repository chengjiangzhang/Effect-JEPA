#!/usr/bin/env python3
"""Read-only comparison of formal RGB capture and its physics-only preflight.

This establishes numerical correspondence, not image quality, collision freedom,
or independent acceptance of the formal data. No simulator client is imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

if __package__:
    from .controllers_v2 import branch_names
    from .audit_auxiliary import inspect_auxiliary, compare_auxiliary, collision_identity
else:
    from controllers_v2 import branch_names
    from audit_auxiliary import inspect_auxiliary, compare_auxiliary, collision_identity
if __package__:
    from .review_common import CAMERA_NAMES, INITIAL_TOLERANCES, state_errors, write_json
else:
    from review_common import CAMERA_NAMES, INITIAL_TOLERANCES, state_errors, write_json


LEGACY_SOURCES = ('collect_multicam.py', 'scenarios_v2.py', 'controllers_v2.py')
AUXILIARY_SOURCES = LEGACY_SOURCES + ('auxiliary_actors.py',)
DECISION_SOURCES = AUXILIARY_SOURCES + ('ego_decisions.py',)
LIDAR_SOURCES = DECISION_SOURCES + ('semantic_lidar.py',)
PEDESTRIAN_SOURCES = LIDAR_SOURCES + ('pedestrian_policy.py',)
CROSSING_SOURCES = PEDESTRIAN_SOURCES + ('pedestrian_crossing.py',)
SOURCES = CROSSING_SOURCES + ('npc_merge.py',)
CLOCKS = ('frame_id', 'simulation_time_s')
VECTORS = ('position', 'rotation_deg', 'velocity', 'angular_velocity', 'acceleration',
           'control', 'observed_control')
SCALARS = ('target_speed', 'observed_gear', 'observed_manual_gear_shift',
           'requested_gear', 'requested_manual_gear_shift')
BOXES = ('bbox_extent', 'bbox_offset', 'bbox_rotation_deg')
STATE_ERROR_FIELDS = ('position', 'rotation_deg', 'velocity', 'angular_velocity', 'acceleration')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), sha(raw)


def config_digest(config):
    value = {key: value for key, value in config.items() if key != 'spec_sha256'}
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def source_identity(record):
    values = record.get('code_sha256', {})
    if not isinstance(values, dict) or set(values) not in (set(SOURCES),set(CROSSING_SOURCES),set(PEDESTRIAN_SOURCES),set(LIDAR_SOURCES),set(DECISION_SOURCES),set(AUXILIARY_SOURCES),set(LEGACY_SOURCES)):
        return None
    if any(v.get('route_maneuver',{}).get('policy_version')=='observed-gap-merge-v1' for v in record.get('vehicles',[])) and 'npc_merge.py' not in values:
        return None
    if any(p.get('response_policy') for p in record.get('pedestrians', [])) and 'pedestrian_policy.py' not in values:
        return None
    if any(p.get('response_policy', {}).get('kind') in ('reactive_crossing_v2', 'reactive_crossing_v3', 'reactive_crossing_v4') for p in record.get('pedestrians', [])) and 'pedestrian_crossing.py' not in values:
        return None
    if record.get('semantic_lidar') and 'semantic_lidar.py' not in values:
        return None
    if record.get('action', {}).get('profile_version') == 2 and 'ego_decisions.py' not in values:
        return None
    if (record.get('pedestrians') or record.get('traffic_signal_program') or
            record.get('geometry', {}).get('narrow_meeting')) and 'auxiliary_actors.py' not in values:
        return None
    if any(not isinstance(value, str) or len(value) != 64 or
           any(c not in '0123456789abcdef' for c in value) for value in values.values()):
        return None
    return values


def inventory(root, manifest):
    """Validate the plan independently of which raw branches are complete."""
    scenes = manifest.get('scene_ids')
    if not isinstance(scenes, list) or not scenes or len(set(scenes)) != len(scenes) or any(
            not isinstance(scene, str) or Path(scene).name != scene for scene in scenes):
        raise ValueError('Manifest must specify nonempty, unique scene path components')
    per_scene, configs, issues = {}, {}, []
    declared = manifest.get('scene_branches')
    if declared is not None and (not isinstance(declared, dict) or set(declared) != set(scenes)):
        issues.append('manifest.scene_branches keys differ from scene_ids')
    for scene in scenes:
        cfg, file_sha = read_json(root / scene / 'spec.json')
        names = list(branch_names(cfg))
        per_scene[scene] = names
        configs[scene] = {'path': str(root / scene / 'spec.json'), 'sha256': file_sha,
                          'spec_sha256': cfg.get('spec_sha256')}
        if cfg.get('scene_id') != scene or cfg.get('branches') != names:
            issues.append(f'{scene}: config scene/branch declaration differs from validated action profiles')
        if cfg.get('spec_sha256') != config_digest(cfg):
            issues.append(f'{scene}: config payload/hash identity mismatch')
        if source_identity(cfg) is None or source_identity(cfg) != source_identity(manifest):
            issues.append(f'{scene}: config/manifest collection-source identity mismatch')
        if not cfg.get('vehicle_roles') or cfg['vehicle_roles'][0] != 'ego':
            issues.append(f'{scene}: ego must be the first configured actor')
        if 'profiles' in cfg.get('action', {}) and declared is None:
            issues.append(f'{scene}: dynamic action protocol requires manifest.scene_branches')
        if declared is not None and (not isinstance(declared, dict) or declared.get(scene) != names):
            issues.append(f'{scene}: manifest.scene_branches differs from validated config')
        branch_dir = root / scene / 'branches'
        extras = sorted(p.name for p in branch_dir.iterdir() if p.is_dir() and p.name not in names) if branch_dir.is_dir() else []
        if extras:
            issues.append(f'{scene}: unconfigured actual branch directories: {extras}')
    union = set(name for names in per_scene.values() for name in names)
    declared_union = manifest.get('branches', [])
    if not isinstance(declared_union, list) or len(set(declared_union)) != len(declared_union) or set(declared_union) != union:
        issues.append('manifest.branches differs from the union of validated per-scene branches')
    extras = sorted(p.name for p in root.glob('scene_*') if p.is_dir() and p.name not in scenes)
    if extras:
        issues.append(f'Unconfigured actual scene directories: {extras}')
    return per_scene, configs, issues


def maximum_difference(a, b):
    difference = np.abs(a.astype(np.longdouble) - b.astype(np.longdouble))
    result = float(difference.max())
    return result if np.isfinite(result) else None


def compare_field(name, a, b):
    result = {'data_shape': list(a.shape), 'reference_shape': list(b.shape),
              'data_dtype': str(a.dtype), 'reference_dtype': str(b.dtype),
              'shape_equal': a.shape == b.shape, 'dtype_equal': a.dtype == b.dtype,
              'raw_exact': False, 'raw_max_abs_difference': None}
    for side, values in (('data', a), ('reference', b)):
        numeric = values.dtype.kind in 'biuf'
        result[side + '_finite'] = bool(numeric and values.size and np.isfinite(values).all())
        result[side + '_nonfinite_count'] = int((~np.isfinite(values)).sum()) if numeric else None
    valid = result['shape_equal'] and result['data_finite'] and result['reference_finite']
    if valid:
        result['raw_exact'] = bool(np.array_equal(a, b))
        result['raw_max_abs_difference'] = maximum_difference(a, b)
    if name in CLOCKS:
        result.update(relative_exact=False, relative_max_abs_difference=None,
                      absolute_first_offset_data_minus_reference=None)
        if valid and a.ndim == 1:
            # Keep integer arithmetic for frame IDs; do not coerce large IDs to float.
            relative_a, relative_b = a - a[0], b - b[0]
            result['relative_exact'] = bool(np.array_equal(relative_a, relative_b))
            result['relative_max_abs_difference'] = maximum_difference(relative_a, relative_b)
            offset = a[0].item() - b[0].item()
            result['absolute_first_offset_data_minus_reference'] = offset if np.isfinite(offset) else None
            result['comparison'] = 'relative to first recorded sample; absolute offset retained'
    return result


def compare_arrays(data, reference, config):
    roles = config['vehicle_roles']
    count = round((config['history_s'] + config['duration_s']) / config['dt_s']) + 1
    expected = {key: (count, len(roles), 3) for key in VECTORS}
    expected.update({key: (count, len(roles)) for key in SCALARS})
    expected.update({key: (len(roles), 3) for key in BOXES})
    expected.update(time_s=(count,), frame_id=(count,), simulation_time_s=(count,),
                    transform_world_vehicle=(count, len(roles), 4, 4))
    issues, fields = [], {}
    for name in sorted(set(data) | set(reference) | set(expected)):
        if name not in data or name not in reference:
            fields[name] = {'data_present': name in data, 'reference_present': name in reference}
            issues.append('Missing state field: ' + name)
            continue
        fields[name] = compare_field(name, data[name], reference[name])
        if name in expected:
            fields[name]['expected_shape'] = list(expected[name])
            if data[name].shape != expected[name] or reference[name].shape != expected[name]:
                issues.append('Unexpected full-sequence shape: ' + name)
        if not all(fields[name][key] for key in ('shape_equal', 'dtype_equal', 'data_finite', 'reference_finite')):
            issues.append('State shape, dtype or finiteness mismatch: ' + name)
    if 'time_s' in fields and fields['time_s'].get('raw_exact'):
        times = data['time_s']
        expected_times = np.arange(count) * config['dt_s'] - config['history_s']
        if times.shape != expected_times.shape or not np.allclose(times, expected_times, atol=1e-8, rtol=0):
            issues.append('Relative time grid does not cover configured history/future')
    else:
        issues.append('Relative time_s arrays do not match exactly')
    physical_exact = not issues and all(row.get('raw_exact', False)
                                       for key, row in fields.items() if key not in CLOCKS)
    relative_clock_exact = all(fields.get(key, {}).get('relative_exact', False) for key in CLOCKS)
    diagnostics = {'scope': 'full recorded sequence; diagnostic use of existing initial-state tolerances',
                   'not_a_substitute_for_exact_equivalence': True,
                   'tolerances': dict(INITIAL_TOLERANCES), 'errors': None,
                   'within_tolerance_by_quantity': None, 'all_quantities_within_tolerance': None}
    diagnostic_ready = (fields.get('time_s', {}).get('raw_exact', False) and
                        all(name in fields and fields[name].get('shape_equal') and
                            fields[name].get('data_finite') and fields[name].get('reference_finite') and
                            data[name].shape == expected[name] for name in STATE_ERROR_FIELDS))
    if diagnostic_ready:
        errors = state_errors(data, reference, np.arange(count))
        flags = {key: errors[key] <= limit for key, limit in INITIAL_TOLERANCES.items()}
        diagnostics.update(errors=errors, within_tolerance_by_quantity=flags,
                           all_quantities_within_tolerance=all(flags.values()))
    return {'fields': fields, 'issues': issues, 'physical_fields_exact': bool(physical_exact),
            'relative_clock_sequences_exact': bool(relative_clock_exact),
            'full_sequence_state_diagnostics': diagnostics}


def compare_branch(data_root, reference_root, scene, branch, manifests):
    result = {'scene': scene, 'branch': branch, 'status': 'incomplete', 'identity_matches': False,
              'equivalent_exact': False, 'issues': [], 'inputs': {}}
    configs, metas, arrays, state_paths = {}, {}, {}, {}
    for side, root in (('data', data_root), ('reference', reference_root)):
        scene_path = root / scene
        branch_path = scene_path / 'branches' / branch
        paths = {'config': scene_path / 'spec.json', 'meta': branch_path / 'meta.json',
                 'states': branch_path / 'states.npz'}
        result['inputs'][side] = {key: {'path': str(path), 'exists': path.is_file()}
                                  for key, path in paths.items()}
        for key, path in paths.items():
            if not path.is_file():
                result['issues'].append(f'{side}: missing {key}')
        if any(not path.is_file() for path in paths.values()):
            continue
        configs[side], config_sha = read_json(paths['config'])
        metas[side], meta_sha = read_json(paths['meta'])
        result['inputs'][side]['config']['sha256'] = config_sha
        result['inputs'][side]['meta']['sha256'] = meta_sha
        result['inputs'][side]['meta']['status'] = metas[side].get('status')
        if metas[side].get('status') != 'complete':
            result['issues'].append(f'{side}: branch status is not complete')
            continue
        state_paths[side] = paths['states']
    if len(state_paths) != 2:
        return result
    for side, path in state_paths.items():
        raw = path.read_bytes()
        result['inputs'][side]['states']['sha256'] = sha(raw)
        with np.load(io.BytesIO(raw), allow_pickle=False) as values:
            arrays[side] = {name: values[name] for name in values.files}
    result['status'] = 'mismatch'
    identity_issues = []
    for side in ('data', 'reference'):
        config, meta, manifest = configs[side], metas[side], manifests[side]
        declared = config.get('spec_sha256')
        result['inputs'][side]['spec_sha256'] = declared
        result['inputs'][side]['code_sha256'] = meta.get('code_sha256')
        if not declared or config_digest(config) != declared or meta.get('spec_sha256') != declared:
            identity_issues.append(f'{side}: config payload/hash/meta identity mismatch')
        code = source_identity(config)
        if code is None or source_identity(meta) != code or source_identity(manifest) != code:
            identity_issues.append(f'{side}: collection-source identity differs across config/meta/manifest')
        if config.get('scene_id') != scene or meta.get('branch') != branch or branch not in branch_names(config):
            identity_issues.append(f'{side}: scene/branch identity mismatch')
        if not config.get('vehicle_roles') or meta.get('vehicle_roles') != config['vehicle_roles']:
            identity_issues.append(f'{side}: recorded role ordering differs from config')
    if configs['data'].get('spec_sha256') != configs['reference'].get('spec_sha256'):
        identity_issues.append('Formal/reference spec_sha256 differs')
    if source_identity(metas['data']) != source_identity(metas['reference']):
        identity_issues.append('Formal/reference collection source hashes differ')
    if metas['data'].get('vehicle_roles') != metas['reference'].get('vehicle_roles'):
        identity_issues.append('Formal/reference recorded role ordering differs')
    if metas['reference'].get('capture_mode') != 'physics_only_preflight_not_delivery':
        identity_issues.append('Reference is not labelled physics-only preflight')
    formal_meta = metas['data']
    if formal_meta.get('capture_mode') == 'physics_only_preflight_not_delivery' or \
            set(formal_meta.get('camera_names', [])) != set(CAMERA_NAMES) or not formal_meta.get('rgb_samples', 0):
        identity_issues.append('Formal branch does not declare six-camera RGB capture')
    result['identity_matches'] = not identity_issues
    result['identity_issues'] = identity_issues
    result.update(compare_arrays(arrays['data'], arrays['reference'], configs['data']))
    auxiliary = {}
    for side, root in (('data', data_root), ('reference', reference_root)):
        auxiliary[side] = inspect_auxiliary(root/scene/'branches'/branch, configs[side], metas[side], arrays[side])
        result['issues'].extend(f'{side}: {issue}' for issue in auxiliary[side]['errors'])
    auxiliary['comparison'] = compare_auxiliary(data_root/scene/'branches'/branch,
                                                reference_root/scene/'branches'/branch,configs['data'])
    if configs['data'].get('pedestrians'):
        try:
            if collision_identity(metas['data']) != collision_identity(metas['reference']):
                auxiliary['comparison']['errors'].append('Vehicle/pedestrian collision evidence differs between physics/RGB')
                auxiliary['comparison']['status'] = 'failed'
        except (KeyError, TypeError, ValueError) as exc:
            auxiliary['comparison']['errors'].append('Cannot compare collision evidence: '+str(exc))
            auxiliary['comparison']['status'] = 'failed'
    result['issues'].extend(auxiliary['comparison']['errors'])
    result['auxiliary_actors'] = auxiliary
    auxiliary_passed = all(value['status'] == 'passed' for value in auxiliary.values())
    result['vehicle_physical_fields_exact'] = result['physical_fields_exact']
    result['auxiliary_fields_exact'] = auxiliary_passed
    result['physical_fields_exact'] = result['physical_fields_exact'] and auxiliary_passed
    if any(v.get('route_maneuver',{}).get('policy_version')=='observed-gap-merge-v1' for v in configs['data'].get('vehicles',[])):
        merge={};merge_exact=True
        for side,root in (('data',data_root),('reference',reference_root)):
            folder=root/scene/'branches'/branch
            memory,memory_sha=read_json(folder/'npc_merge_state.json')
            diagnostics,diagnostics_sha=read_json(folder/'controller_diagnostics.json')
            merge[side]={'memory':memory,'decisions':[[row.get('dynamic_merge') for row in frame] for frame in diagnostics]}
            result['inputs'][side]['npc_merge_state_sha256']=memory_sha
            result['inputs'][side]['controller_diagnostics_sha256']=diagnostics_sha
        merge_exact=merge['data']==merge['reference']
        result['dynamic_merge_exact']=merge_exact
        result['physical_fields_exact'] &= merge_exact
        if not merge_exact:
            result['issues'].append('Dynamic merge warmup, t0 memory or policy decisions differ between physics/RGB')
    result['issues'] = identity_issues + result['issues']
    result['equivalent_exact'] = bool(result['identity_matches'] and result['physical_fields_exact']
                                       and result['relative_clock_sequences_exact'] and auxiliary_passed)
    if result['equivalent_exact']:
        result['status'] = 'equivalent_exact'
    return result


def branch_fingerprint(data, reference, scene, branch, manifest_hashes):
    values = [tuple(manifest_hashes)]
    for root in (data, reference):
        for path in (root / scene / 'spec.json', root / scene / 'branches' / branch / 'meta.json',
                     root / scene / 'branches' / branch / 'states.npz',
                     root / scene / 'branches' / branch / 'pedestrians.npz',
                     root / scene / 'branches' / branch / 'pedestrian_decisions.json',
                     root / scene / 'branches' / branch / 'npc_merge_state.json',
                     root / scene / 'branches' / branch / 'controller_diagnostics.json',
                     root / scene / 'branches' / branch / 'signal_states.json'):
            stat = path.stat() if path.is_file() else None
            values.append((str(path), stat.st_size, stat.st_mtime_ns) if stat else (str(path), None))
    return tuple(values)


def audit(data, reference, cache=None):
    data, reference = Path(data).absolute(), Path(reference).absolute()
    manifests, input_identity = {}, {}
    for side, root in (('data', data), ('reference', reference)):
        manifests[side], value = read_json(root / 'manifest.json')
        input_identity[side] = {'root': str(root), 'manifest_sha256': value,
                                'manifest_status': manifests[side].get('status')}
    inventories, protocol_issues = {}, []
    for side, root in (('data', data), ('reference', reference)):
        inventories[side], configs, issues = inventory(root, manifests[side])
        input_identity[side]['configs'] = configs
        protocol_issues.extend(f'{side}: {issue}' for issue in issues)
    scenes, per_scene = manifests['data']['scene_ids'], inventories['data']
    if set(inventories['data']) != set(inventories['reference']):
        protocol_issues.append('Formal/reference scene coverage differs')
    if inventories['data'] != inventories['reference']:
        protocol_issues.append('Formal/reference validated per-scene branches differ')
    for scene in set(inventories['data']) & set(inventories['reference']):
        if input_identity['data']['configs'][scene]['spec_sha256'] != input_identity['reference']['configs'][scene]['spec_sha256']:
            protocol_issues.append(f'{scene}: Formal/reference spec_sha256 differs')
    rows = []
    for scene in scenes:
        for branch in per_scene[scene]:
            fingerprint = branch_fingerprint(data, reference, scene, branch,
                [input_identity[side]['manifest_sha256'] for side in ('data', 'reference')])
            key = (scene, branch)
            try:
                if cache is not None and key in cache and cache[key][0] == fingerprint:
                    row = cache[key][1]
                else:
                    row = compare_branch(data, reference, scene, branch, manifests)
                    if cache is not None:
                        cache.pop(key, None)
                        if 'fields' in row:
                            cache[key] = (fingerprint, row)
            except Exception as error:
                row = {'scene': scene, 'branch': branch, 'status': 'error',
                       'equivalent_exact': False, 'issues': [f'{type(error).__name__}: {error}']}
            rows.append(row)
    complete = all('fields' in row for row in rows)
    exact = not protocol_issues and all(row.get('equivalent_exact', False) for row in rows)
    physical_exact = sum(row.get('identity_matches', False) and row.get('physical_fields_exact', False) for row in rows)
    return {'schema': 'rgb-physics-capture-equivalence-v1',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'inputs': input_identity,
            'tool_sha256': sha(Path(__file__).read_bytes()),
            'state_errors_source_sha256': sha(Path(__file__).with_name('review_common.py').read_bytes()),
            'branch_protocol_source_sha256': sha(Path(__file__).with_name('controllers_v2.py').read_bytes()),
            'status': ('error' if any(row['status'] == 'error' for row in rows) else
                       'mismatch' if protocol_issues else 'equivalent_exact' if complete and exact else 'mismatch' if complete else 'incomplete'),
            'complete': complete, 'equivalent_exact': complete and exact,
            'protocol_issues': protocol_issues, 'scene_branches': per_scene,
            'reference_scene_branches': inventories['reference'],
            'expected_scenes': len(scenes), 'expected_branches': len(rows),
            'compared_branches': sum('fields' in row for row in rows),
            'identity_matched_branches': sum(row.get('identity_matches', False) for row in rows),
            'physical_fields_exact_branches': physical_exact,
            'physical_and_relative_clocks_exact_branches': sum(row.get('equivalent_exact', False) for row in rows),
            'incomplete_branches': [f"{row['scene']}/{row['branch']}" for row in rows if row['status'] == 'incomplete'],
            'mismatched_or_error_branches': [f"{row['scene']}/{row['branch']}" for row in rows
                                             if row['status'] in ('mismatch', 'error')],
            'definitions': {
                'physical_fields_exact': 'Every vehicle/pedestrian state field except absolute frame_id and simulation_time_s matches exactly, with required fields/shapes/dtypes and recorded timeline. Configured traffic-light identities and readback sequences also match; pedestrian-scene collision evidence matches after removing ephemeral actor IDs.',
                'relative_clock_sequences_exact': 'frame_id and simulation_time_s match after subtracting their own first recorded value. No rounding/tolerance is used; absolute offset and raw differences are also reported.',
                'full_sequence_state_diagnostics': 'Five separately dimensioned state_errors; INITIAL_TOLERANCES applied diagnostically over the full sequence, never pooled across units or substituted for exact equality.',
            },
            'limitations': ['Physics-only preflight remains non-delivery data.',
                            'This audit does not decode PNGs or replace formal RGB synchronization, image-quality, collision, dynamics or repeat QA.',
                            'Clock floating-point roundoff is reported as a difference, never silently called exact.',
                            'No audit verdict from another source/config is transferred. Missing/incomplete branches cannot pass.'],
            'branches': rows}


def summary(report):
    return {key: report[key] for key in ('status', 'expected_branches', 'compared_branches',
                                         'physical_fields_exact_branches',
                                         'physical_and_relative_clocks_exact_branches')}


def error_report(data, reference, error):
    return {'schema': 'rgb-physics-capture-equivalence-v1',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'error', 'complete': False, 'equivalent_exact': False,
            'expected_branches': None, 'compared_branches': 0,
            'physical_fields_exact_branches': 0, 'physical_and_relative_clocks_exact_branches': 0,
            'inputs': {'data': {'root': str(Path(data).absolute())},
                       'reference': {'root': str(Path(reference).absolute())}},
            'protocol_issues': [f'{type(error).__name__}: {error}'], 'branches': []}


def watch(data, reference, out, *, batch=None, poll_seconds=20.):
    """Monitor complete raw branches; cached reads never replace the final fresh audit."""
    cache, previous = {}, None
    progress_path = out.with_name(out.stem + '_watch.json')
    while True:
        try:
            report = audit(data, reference, cache)
            if report['complete']:
                # Rehash and reload everything before publishing a final result.
                report = audit(data, reference)
            batch_record = read_json(batch)[0] if batch is not None and batch.is_file() else {}
        except Exception as error:
            report = error_report(data, reference, error)
            write_json(out, report)
            write_json(progress_path, {**report, 'watch_status': 'failed'})
            print(json.dumps(report), flush=True)
            return 1
        batch_status = batch_record.get('status')
        failed = bool(report['protocol_issues'] or report['mismatched_or_error_branches'])
        stopped = bool(batch_status and batch_status not in
                       ('running', 'planned', 'prepared', 'stopped_after_branch_limit'))
        current = (json.dumps(summary(report), sort_keys=True), batch_status,
                   batch_record.get('session_id'), report['inputs']['data']['manifest_sha256'],
                   tuple(report['protocol_issues']))
        if current != previous or report['complete'] or failed or stopped:
            write_json(out, report)
            progress = {'schema': 'capture-equivalence-watch-v1',
                        'updated_utc': datetime.now(timezone.utc).isoformat(),
                        'data': str(data.absolute()), 'reference': str(reference.absolute()),
                        'report': str(out.absolute()), 'batch_status': batch_status,
                        'scene_branches': report['scene_branches'], **summary(report)}
            progress['watch_status'] = ('complete_exact' if report['equivalent_exact'] else
                'mismatch' if failed else 'collection_stopped_incomplete' if stopped else
                'paused_for_renderer_maintenance' if batch_status == 'stopped_after_branch_limit' else
                'waiting_for_remaining_branches')
            write_json(progress_path, progress)
            print(json.dumps(progress), flush=True)
            previous = current
        if report['complete'] or failed or stopped:
            return 0 if report['equivalent_exact'] else 2
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Formal six-camera dataset root')
    parser.add_argument('--reference', type=Path, required=True, help='Physics-only preflight dataset root')
    parser.add_argument('--out', type=Path, required=True, help='Standalone JSON report outside both raw datasets')
    parser.add_argument('--watch', action='store_true', help='Continuously audit newly completed branches; no CARLA access')
    parser.add_argument('--batch', type=Path, help='Optional collection batch.json; stop on failure, wait through renderer maintenance')
    parser.add_argument('--poll-seconds', type=float, default=20.)
    args = parser.parse_args()
    if any(args.out.resolve().is_relative_to(root.resolve()) for root in (args.data, args.reference)):
        parser.error('--out must not modify either raw dataset')
    if not 0 < args.poll_seconds <= 30:
        parser.error('--poll-seconds must be greater than zero and at most 30')
    if args.watch:
        return watch(args.data, args.reference, args.out, batch=args.batch, poll_seconds=args.poll_seconds)
    try:
        report = audit(args.data, args.reference)
    except Exception as error:
        report = error_report(args.data, args.reference, error)
    write_json(args.out, report)
    print(json.dumps(summary(report)))
    return 0 if report['equivalent_exact'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
