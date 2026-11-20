#!/usr/bin/env python3
"""Compare every physics/RGB replay with the measured, pre-intervention prefix.

This checks physical correspondence, not RGB pixels: the measurement has no
cameras. The existing formal RGB consistency audit remains independently required.
No simulator connection is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

# The gate also runs as a standalone runtime CLI. Reuse the same narrow
# approval rule as the planner, rather than maintaining a second allow-list.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measured_t0 import assert_shared_setup_unchanged

if __package__:
    from .audit_capture_equivalence import compare_field, config_digest, source_identity
    from .audit_auxiliary import inspect_auxiliary, signal_identity
    from .controllers_v2 import branch_names
    from .review_common import CAMERA_NAMES, write_json
    from .vehicle_geometry import common_history_vehicle_geometry, HISTORY_GEOMETRY_SCHEMA
else:
    from audit_capture_equivalence import compare_field, config_digest, source_identity
    from audit_auxiliary import inspect_auxiliary, signal_identity
    from controllers_v2 import branch_names
    from review_common import CAMERA_NAMES, write_json
    from vehicle_geometry import common_history_vehicle_geometry, HISTORY_GEOMETRY_SCHEMA

MODE = 'common_history_only_not_delivery'
HISTORY_BRANCH = 'common_history'
OUTGOING = frozenset(('control', 'target_speed', 'requested_gear', 'requested_manual_gear_shift'))
BOXES = frozenset(('bbox_extent', 'bbox_offset', 'bbox_rotation_deg'))
CLOCKS = frozenset(('frame_id', 'simulation_time_s'))
REQUIRED = frozenset(('position', 'rotation_deg', 'velocity', 'angular_velocity', 'acceleration',
    'transform_world_vehicle', 'control', 'target_speed', 'observed_control', 'observed_gear',
    'observed_manual_gear_shift', 'requested_gear', 'requested_manual_gear_shift',
    'time_s', 'frame_id', 'simulation_time_s')) | BOXES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def vehicle_shapes(arrays, count, actor_count):
    shapes = {key:(count,actor_count,3) for key in ('position','rotation_deg','velocity',
              'angular_velocity','acceleration','control','observed_control')}
    shapes.update({key:(count,actor_count) for key in ('target_speed','observed_gear',
        'observed_manual_gear_shift','requested_gear','requested_manual_gear_shift')})
    shapes.update({key:(actor_count,3) for key in BOXES})
    shapes.update({key:(count,) for key in ('time_s','frame_id','simulation_time_s')})
    shapes['transform_world_vehicle']=(count,actor_count,4,4)
    return [f'{key}: shape does not match the frozen actor count / recorded time grid'
            for key,shape in shapes.items() if key not in arrays or arrays[key].shape != shape]


def compare_prefix(actual, reference, *, pedestrian=False):
    errors, fields = [], {}
    required = ((REQUIRED - {'transform_world_vehicle', 'target_speed', 'observed_gear',
        'observed_manual_gear_shift', 'requested_gear', 'requested_manual_gear_shift'}) |
        {'transform_world_actor'}) if pedestrian else REQUIRED
    if set(actual) != set(reference) or not required.issubset(reference):
        errors.append('State field set differs or required physical/control evidence is absent')
    expected_times = np.arange(41)*.05-2.
    for side, arrays, count in (('actual', actual, 121), ('reference', reference, 41)):
        times = arrays.get('time_s')
        if times is None or times.shape != (count,) or not np.array_equal(times, np.arange(count)*.05-2.):
            errors.append(f'{side}: expected the complete {count}-frame scene-relative time grid')
    for key in sorted(set(actual) & set(reference)):
        a, b = actual[key], reference[key]
        if key not in BOXES:
            if a.ndim == 0 or b.ndim == 0 or a.shape[0] != 121 or b.shape[0] != 41:
                errors.append(f'{key}: expected 121 replay / 41 measured time samples')
                continue
            # t0 observed state is shared; outgoing t0 controls belong to the
            # first intervened interval and have no counterpart in measurement.
            end = 40 if key in OUTGOING else 41
            a, b = a[:end], b[:end]
        fields[key] = compare_field(key, a, b)
        field = fields[key]
        match = field.get('relative_exact') if key in CLOCKS else field.get('raw_exact')
        if not (match and field['dtype_equal'] and field['data_finite'] and field['reference_finite']):
            errors.append(f'{key}: measured/replayed common prefix is not exactly equal')
    if 'time_s' in actual and not np.array_equal(actual['time_s'][:41], expected_times):
        errors.append('Replay does not include the full observed t0 boundary')
    return {'status': 'failed' if errors else 'passed', 'errors': errors, 'fields': fields,
            'physical_tolerance': 0., 'clock_policy': 'Exact relative clock; absolute offset reported',
            'requested_control_scope': 't < 0; t0 outgoing controls excluded',
            'observed_state_scope': '-2 <= t <= 0, including prior applied control at t0'}


def compare_signals(actual, reference):
    errors = []
    if actual.get('program') != reference.get('program') or signal_identity(actual) != signal_identity(reference):
        errors.append('Traffic-signal fixed program or stable light identities changed')
    left, right = actual.get('samples', []), reference.get('samples', [])
    if len(left) != 121 or len(right) != 41:
        return errors + ['Traffic-signal replay/reference sample count differs from 121/41']
    def normalize(rows):
        first = rows[0]
        return [{**row, 'frame_id': row['frame_id']-first['frame_id'],
                 'simulation_time_s': row['simulation_time_s']-first['simulation_time_s']}
                for row in rows[:41]]
    if normalize(left) != normalize(right):
        errors.append('Measured/commanded traffic-signal common prefix differs')
    return errors


def check_rgb_samples(folder, states, meta):
    """A rendered replay must sample the same physical prefix; no pixel claim."""
    if not meta.get('camera_names'):
        return {'applicable': False, 'errors': []}
    errors = []
    if set(meta['camera_names']) != set(CAMERA_NAMES) or meta.get('rgb_samples') != 13:
        errors.append('RGB replay does not declare the six-camera / 13-sample protocol')
    samples = read_arrays(folder/'samples.npz')
    indices = np.arange(0, 121, 10)
    for key, expected in (('physics_indices', indices), ('time_s', np.arange(13)*.5-2.),
                          ('frame_id', states['frame_id'][indices])):
        if key not in samples or not np.array_equal(samples[key], expected):
            errors.append('RGB sampling differs from replay physics: '+key)
    poses = samples.get('ego_to_world')
    if poses is None or poses.shape != (13,4,4) or not np.array_equal(
            poses[:5], states['transform_world_vehicle'][indices[:5],0]):
        errors.append('RGB common-prefix ego pose differs from measured physics samples')
    sensor_ids = samples.get('sensor_frame_id')
    if sensor_ids is None or sensor_ids.shape != (13,7) or not np.array_equal(
            sensor_ids[:5], np.repeat(states['frame_id'][indices[:5],None],7,axis=1)):
        errors.append('RGB/LiDAR prefix sensor frames do not match corresponding physics')
    return {'applicable': True, 'errors': errors,
            'pixel_equivalence': 'Not tested here: no RGB exists in history-only measurement',
            'samples_sha256': sha(folder/'samples.npz')}


def audit_branch(data, reference, scene, branch, manifests):
    folder = data/scene/'branches'/branch
    measured = reference/scene/'branches'/HISTORY_BRANCH
    errors, identity = [], {}
    cfgs, metas, arrays, geometry = {}, {}, {}, {}
    for side, root, location in (('actual', data, folder), ('reference', reference, measured)):
        paths = {'spec': root/scene/'spec.json', 'meta': location/'meta.json', 'states': location/'states.npz'}
        identity[side] = {name: {'path': str(path), 'sha256': sha(path)} for name,path in paths.items()}
        cfgs[side], metas[side], arrays[side] = read_json(paths['spec']), read_json(paths['meta']), read_arrays(paths['states'])
        cfg, meta = cfgs[side], metas[side]
        if config_digest(cfg) != cfg.get('spec_sha256') or meta.get('spec_sha256') != cfg.get('spec_sha256'):
            errors.append(f'{side}: spec content hash / capture identity mismatch')
        if cfg.get('scene_id') != scene or meta.get('status') != 'complete':
            errors.append(f'{side}: scene identity mismatch or capture incomplete')
        code = source_identity(cfg)
        if code is None or source_identity(meta) != code or source_identity(manifests[side]) != code:
            errors.append(f'{side}: frozen collector source identity mismatch')
        if meta.get('vehicle_roles') != cfg.get('vehicle_roles') or not cfg.get('vehicle_roles'):
            errors.append(f'{side}: captured vehicle role order differs from spec')
        errors.extend(f'{side}: {error}' for error in vehicle_shapes(arrays[side],
            121 if side=='actual' else 41,len(cfg.get('vehicle_roles',[]))))
        try:
            geometry[side] = common_history_vehicle_geometry(arrays[side], cfg.get('vehicle_roles'))
            if not geometry[side]['passed']:
                errors.append(f'{side}: recorded vehicle bbox interior overlap during -2..0 common history')
        except (ValueError, TypeError) as exc:
            geometry[side] = {'status': 'failed', 'passed': False, 'error': str(exc)}
            errors.append(f'{side}: missing/invalid common-history vehicle geometry: {exc}')
        if not isinstance(meta.get('collisions'), list):
            errors.append(f'{side}: collision sensor evidence is missing')
        elif any(event['time_s'] <= 0 for event in meta['collisions']):
            errors.append(f'{side}: contact during initialization or measured common history')
    cfg, original = cfgs['actual'], cfgs['reference']
    try:
        assert_shared_setup_unchanged(original, cfg,
            cfg.get('measured_t0_plan', {}).get('explicit_timing_bounds'))
    except (ValueError, KeyError, TypeError) as exc:
        errors.append('Shared setup or unapproved onset changed after measurement: '+str(exc))
    if metas['actual'].get('branch') != branch or branch not in branch_names(cfg):
        errors.append('Actual branch not declared in frozen scene')
    ref_meta = metas['reference']
    if (ref_meta.get('capture_mode') != MODE or ref_meta.get('branch') != HISTORY_BRANCH or
            ref_meta.get('history_only_complete') is not True or ref_meta.get('t0_index') != 40 or
            ref_meta.get('physics_frames') != 41):
        errors.append('Reference is not a complete t0-only history measurement')
    manifest = manifests['reference']
    if (manifest.get('frozen_history_spec_sha256', {}).get(scene) != original.get('spec_sha256') or
            manifest.get('input_spec_file_sha256', {}).get(scene) != identity['reference']['spec']['sha256']):
        errors.append('Measured history manifest does not bind the original spec bytes')
    results = [r for r in manifest.get('branch_results', []) if r.get('scene') == scene and r.get('branch') == HISTORY_BRANCH]
    if len(results) != 1 or any(results[0].get(k+'_sha256') != identity['reference'][k]['sha256'] for k in ('meta','states')):
        errors.append('Measured history manifest does not bind complete state/meta evidence')
    linkage = cfg.get('measured_t0_plan')
    if linkage is not None:
        expected = {'states_sha256': identity['reference']['states']['sha256'],
                    'meta_sha256': identity['reference']['meta']['sha256'],
                    'spec_file_sha256': identity['reference']['spec']['sha256']}
        if (linkage.get('original_history_spec_sha256') != original['spec_sha256'] or
                any(linkage.get('history', {}).get(key) != value for key,value in expected.items())):
            errors.append('Final plan is not linked to these measured t0 states and original spec')
    elif cfg['spec_sha256'] != original['spec_sha256']:
        errors.append('Changed future plan lacks measured_t0_plan evidence linkage')
    if any(a.get('route_maneuver',{}).get('policy_version')=='observed-gap-merge-v1' for a in cfg['vehicles']):
        for name in ('npc_merge_state.json',):
            actual_merge=read_json(folder/name); reference_merge=read_json(measured/name)
            if actual_merge != reference_merge:
                errors.append('Dynamic merge common warmup/history-start/t0 memory differs')
        actual_diag=read_json(folder/'controller_diagnostics.json')
        reference_diag=read_json(measured/'controller_diagnostics.json')
        def merge_rows(rows):
            return [[d.get('dynamic_merge') for d in row] for row in rows[:40]]
        if merge_rows(actual_diag)!=merge_rows(reference_diag):
            errors.append('Dynamic merge common-history decisions differ')
    comparison = compare_prefix(arrays['actual'], arrays['reference'])
    errors.extend(comparison['errors'])
    auxiliary = {}
    for side, location in (('actual',folder), ('reference',measured)):
        auxiliary[side] = inspect_auxiliary(location, cfgs[side], metas[side], arrays[side])
        errors.extend(f'{side}: {error}' for error in auxiliary[side]['errors'])
    if cfg.get('pedestrians'):
        auxiliary['comparison'] = compare_prefix(read_arrays(folder/'pedestrians.npz'),
            read_arrays(measured/'pedestrians.npz'), pedestrian=True)
        errors.extend(auxiliary['comparison']['errors'])
    if cfg.get('traffic_signal_program'):
        errors.extend(compare_signals(read_json(folder/'signal_states.json'), read_json(measured/'signal_states.json')))
    rgb = check_rgb_samples(folder, arrays['actual'], metas['actual'])
    errors.extend(rgb['errors'])
    return {'scene': scene, 'branch': branch, 'status': 'failed' if errors else 'passed',
            'errors': errors, 'inputs': identity, 'physical_prefix': comparison,
            'auxiliary': auxiliary, 'rgb_sampling': rgb, 'vehicle_geometry': geometry}


def audit(data, reference):
    data, reference = Path(data).resolve(), Path(reference).resolve()
    manifests = {'actual': read_json(data/'manifest.json'), 'reference': read_json(reference/'manifest.json')}
    errors, rows = [], []
    measured = manifests['reference']
    if measured.get('capture_mode') != MODE or measured.get('status') != 'complete':
        errors.append('Measured history run is incomplete or has the wrong capture mode')
    scenes = manifests['actual'].get('scene_ids', [])
    if not scenes or len(set(scenes)) != len(scenes) or set(scenes) != set(measured.get('scene_ids', [])):
        errors.append('Actual/measured scene coverage differs or is empty')
    for name, expected in measured.get('preflight_sources_sha256', {}).items():
        path = reference/'source'/name
        if Path(name).name != name or not path.is_file() or sha(path) != expected:
            errors.append('Frozen history source snapshot differs: '+name)
    if not measured.get('preflight_sources_sha256'):
        errors.append('Frozen history source snapshot identity is missing')
    for scene in scenes:
        try:
            if Path(scene).name != scene:
                raise ValueError('Invalid scene path')
            cfg = read_json(data/scene/'spec.json')
            branches = list(branch_names(cfg))
            if manifests['actual'].get('scene_branches',{}).get(scene) != branches:
                errors.append(f'{scene}: manifest does not declare every configured action branch')
            for branch in branches:
                try:
                    rows.append(audit_branch(data, reference, scene, branch, manifests))
                except (OSError, KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
                    rows.append({'scene': scene, 'branch': branch, 'status':'failed',
                                 'errors':[f'{type(exc).__name__}: {exc}']})
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f'{scene}: {type(exc).__name__}: {exc}')
    passed = bool(rows) and not errors and all(row['status']=='passed' for row in rows)
    return {'schema':'measured-history-prefix-audit-v1', 'status':'passed' if passed else 'failed',
            'passed':passed, 'errors':errors, 'data':str(data), 'reference':str(reference),
            'manifest_sha256': {side:sha(root/'manifest.json') for side,root in
                                (('actual',data),('reference',reference))},
            'all_configured_branches_required':True, 'branches_checked':len(rows), 'branches':rows,
            'meaning':'Exact observed physical history and t0, with pre-t0 actuation; no future-outcome filtering',
            'rgb_pixels_checked':False, 'vehicle_geometry_policy':HISTORY_GEOMETRY_SCHEMA,
            'vehicle_geometry_source_sha256':sha(Path(__file__).with_name('vehicle_geometry.py'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.data, args.reference)
    write_json(args.out, report)
    print(json.dumps({'status':report['status'], 'branches_checked':report['branches_checked'],
                      'out':str(args.out)}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
