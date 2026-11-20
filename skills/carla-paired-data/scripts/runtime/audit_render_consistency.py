#!/usr/bin/env python3
"""Fail-closed offline RGB repeat gate; engineering tolerances, not noise statistics.

Compare every non-reference branch's common history through t0 and the complete
maintain_repeat trajectory. Inputs are native PNGs; no registration, resampling,
color correction, or automatically relaxed threshold is permitted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
DEFAULT_THRESHOLDS = {'mae_0_255': .25, 'max_tile_mae_0_255': 4.,
                      'fraction_any_channel_gt_32': .001,
                      'fraction_any_channel_gt_8': .005}
STATE_REQUIRED = {'position', 'rotation_deg', 'velocity', 'angular_velocity',
                  'acceleration', 'transform_world_vehicle', 'control',
                  'target_speed', 'bbox_extent', 'bbox_offset',
                  'bbox_rotation_deg', 'time_s', 'frame_id', 'simulation_time_s'}
ACTUATION = {'observed_control', 'observed_gear', 'observed_manual_gear_shift',
             'requested_gear', 'requested_manual_gear_shift'}
REQUESTED = {'control', 'target_speed', 'requested_gear', 'requested_manual_gear_shift'}
BOXES = {'bbox_extent', 'bbox_offset', 'bbox_rotation_deg'}
SAMPLE_REQUIRED = {'time_s', 'physics_indices', 'frame_id', 'sensor_frame_id',
                   'world_to_camera', 'camera_to_ego', 'intrinsics', 'ego_to_world'}
STATIC_CALIBRATION = {'camera_to_ego', 'intrinsics', 'lidar_to_ego'}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def image_metrics(reference, actual, tile_size=100):
    """Native uint8 RGB differences; edge tiles use their actual pixel counts."""
    if reference.dtype != np.uint8 or actual.dtype != np.uint8:
        raise ValueError('Images must be uint8 RGB')
    if reference.shape != actual.shape or reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError('Images must have identical H x W x 3 shapes')
    if not isinstance(tile_size, int) or tile_size <= 0 or min(reference.shape[:2]) <= 0:
        raise ValueError('Require positive image dimensions and integer tile size')
    difference = np.abs(actual.astype(np.int16)-reference.astype(np.int16))
    maximum = difference.max(axis=-1)
    tiles = [(float(difference[y:y+tile_size, x:x+tile_size].mean()), x, y)
             for y in range(0, reference.shape[0], tile_size)
             for x in range(0, reference.shape[1], tile_size)]
    tile, x, y = max(tiles)
    return {'mae_0_255': float(difference.mean()),
            'fraction_any_channel_gt_8': float((maximum > 8).mean()),
            'fraction_any_channel_gt_32': float((maximum > 32).mean()),
            'max_tile_mae_0_255': tile,
            'max_tile_bounds_xyxy': [x, y, min(x+tile_size, reference.shape[1]),
                                     min(y+tile_size, reference.shape[0])],
            'max_channel_difference_0_255': int(maximum.max())}


def metric_failures(metrics, thresholds):
    return {key: {'actual': metrics[key], 'maximum_allowed': limit}
            for key, limit in thresholds.items() if metrics[key] > limit}


def selected_indices(times, repeat=False, requested=False):
    times = np.asarray(times)
    return np.arange(len(times)) if repeat else np.flatnonzero(times < 0 if requested else times <= 0)


class Inputs:
    """Hash the exact bytes decoded, then rehash all compared inputs before pass."""
    def __init__(self, root):
        self.root = root.resolve()
        self.records = {}

    def read(self, path):
        path = Path(path).resolve()
        data = path.read_bytes()
        relative = str(path.relative_to(self.root))
        record = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        old = self.records.setdefault(relative, record)
        if record != old:
            raise ValueError(f'Input changed while auditing: {relative}')
        return data

    def json(self, path):
        value = json.loads(self.read(path))
        if not isinstance(value, dict):
            raise ValueError(f'Expected JSON object: {path}')
        return value

    def npz(self, path):
        with np.load(io.BytesIO(self.read(path)), allow_pickle=False) as archive:
            result = {key: archive[key] for key in archive.files}
        if any(not (np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_)) or not np.isfinite(value).all()
               for value in result.values()):
            raise ValueError(f'Nonfinite or nonnumeric array in {path}')
        return result

    def png(self, path, width, height):
        data = self.read(path)
        with Image.open(io.BytesIO(data)) as im:
            if im.format != 'PNG' or im.mode != 'RGB' or im.size != (width, height):
                raise ValueError(f'Expected native RGB PNG {width}x{height}: {path}')
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            result = np.asarray(im).copy()
        if result.dtype != np.uint8:
            raise ValueError(f'Expected uint8 PNG: {path}')
        return result

    def recheck(self):
        failures = []
        for relative, expected in self.records.items():
            path = self.root/relative
            try:
                data = path.read_bytes()
                if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                    failures.append({'path': relative, 'error': 'Input changed before final rehash'})
            except OSError as exc:
                failures.append({'path': relative, 'error': str(exc)})
        return failures


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_branch(inputs, folder, config, name):
    meta = inputs.json(folder/'meta.json')
    require(meta.get('status') == 'complete', f'{name}: branch is not complete')
    require(meta.get('branch') == name, f'{name}: meta branch differs')
    require(meta.get('spec_sha256') == config['spec_sha256'], f'{name}: config identity differs')
    require(meta.get('vehicle_roles') == config['vehicle_roles'], f'{name}: role ordering differs')
    require(meta.get('camera_names') == list(CAMERAS), f'{name}: camera ordering differs')
    if 'code_sha256' in config:
        require(meta.get('code_sha256') == config['code_sha256'], f'{name}: capture source identity differs')
    state, sample = inputs.npz(folder/'states.npz'), inputs.npz(folder/'samples.npz')
    needed = STATE_REQUIRED | (ACTUATION if config.get('vehicle_physics_profile') else set())
    require(needed <= state.keys(), f'{name}: missing required state arrays {sorted(needed-state.keys())}')
    require(SAMPLE_REQUIRED <= sample.keys(), f'{name}: missing required sample arrays')
    timing = config['timing']
    expected_t = np.arange(round((timing['history_s']+timing['future_s'])/timing['physics_dt_s'])+1)*timing['physics_dt_s']-timing['history_s']
    expected_sample_t = np.arange(round((timing['history_s']+timing['future_s'])/timing['rgb_interval_s'])+1)*timing['rgb_interval_s']-timing['history_s']
    require(np.array_equal(state['time_s'], expected_t), f'{name}: state relative timestamps differ from plan')
    require(np.array_equal(sample['time_s'], expected_sample_t), f'{name}: RGB relative timestamps differ from plan')
    require(np.count_nonzero(state['time_s'] == 0) == 1, f'{name}: missing unique t0')
    indices = sample['physics_indices']
    require(np.issubdtype(indices.dtype, np.integer) and indices.shape == expected_sample_t.shape,
            f'{name}: invalid physics_indices')
    require((indices >= 0).all() and (indices < len(expected_t)).all(), f'{name}: invalid RGB/state indices')
    require(np.array_equal(state['time_s'][indices], sample['time_s']), f'{name}: RGB/state relative times differ')
    require(np.array_equal(state['frame_id'][indices], sample['frame_id']), f'{name}: RGB/world frames differ')
    require(sample['sensor_frame_id'].shape == (len(indices), 7) and
            (sample['sensor_frame_id'] == sample['frame_id'][:, None]).all(), f'{name}: sensor frame synchronization differs')
    n, t, s = len(config['vehicle_roles']), len(expected_t), len(indices)
    shapes = {'position': (t, n, 3), 'rotation_deg': (t, n, 3), 'velocity': (t, n, 3),
              'angular_velocity': (t, n, 3), 'acceleration': (t, n, 3),
              'transform_world_vehicle': (t, n, 4, 4), 'control': (t, n, 3), 'target_speed': (t, n),
              'simulation_time_s': (t,), 'frame_id': (t,), **{k: (n, 3) for k in BOXES}}
    for key, shape in shapes.items():
        require(state[key].shape == shape, f'{name}: invalid shape for state {key}')
    for key in ACTUATION & state.keys():
        shape = (t, n, 3) if key == 'observed_control' else (t, n)
        require(state[key].shape == shape, f'{name}: invalid shape for state {key}')
    require(np.issubdtype(state['frame_id'].dtype, np.integer) and (np.diff(state['frame_id']) == 1).all(),
            f'{name}: recorded physical frames are not consecutive')
    require((np.diff(state['simulation_time_s']) > 0).all(), f'{name}: simulation clock is not increasing')
    for key, shape in {'world_to_camera': (s, 6, 4, 4), 'ego_to_world': (s, 4, 4),
                       'camera_to_ego': (6, 4, 4), 'intrinsics': (6, 3, 3)}.items():
        require(sample[key].shape == shape, f'{name}: invalid shape for sample {key}')
    require(np.array_equal(sample['ego_to_world'], state['transform_world_vehicle'][indices, 0]),
            f'{name}: sampled ego pose differs from states')
    if 'semantic_lidar' in config:
        if __package__:
            from . import semantic_lidar
        else:
            import semantic_lidar
        lidar_report = semantic_lidar.inspect_branch(folder, config, meta, state, sample)
        require(lidar_report['status'] == 'passed', f'{name}: '+ '; '.join(lidar_report['errors']))
        for relative in lidar_report['input_sha256']:
            inputs.read(folder/relative)
    return meta, state, sample


def compare_branch(reference, actual, repeat):
    """Exact observed-history equality; requested t0 controls are interventions."""
    meta_a, state_a, sample_a = reference
    meta_b, state_b, sample_b = actual
    failures, fields = [], []
    for group, a, b in (('states', state_a, state_b), ('samples', sample_a, sample_b)):
        require(a.keys() == b.keys(), f'{group}: array keys differ')
        for key in sorted(a):
            if key in {'frame_id', 'sensor_frame_id', 'lidar_frame_id', 'lidar_timestamp_s'}:
                continue  # Each episode's internal sensor/frame agreement was checked above.
            x, y = a[key], b[key]
            if key == 'simulation_time_s':
                anchor = int(np.flatnonzero(a['time_s'] == 0)[0])
                x, y = x-x[anchor], y-y[anchor]
            static = key in (BOXES if group == 'states' else STATIC_CALIBRATION)
            index = None if static else selected_indices(a['time_s'], repeat, key in REQUESTED)
            if index is not None:
                x, y = x[index], y[index]
            equal = x.shape == y.shape and np.array_equal(x, y)
            field = {'group': group, 'field': key, 'exact': equal,
                     'scope': 'static' if static else ('full_sequence' if repeat else
                              ('t<0_requested_actions' if key in REQUESTED else 't<=0_observed_history'))}
            fields.append(field)
            if not equal:
                failures.append(field)
    # Metadata is an additional identity check, never a substitute for pixels.
    for key in ('weather_actual', 'weather_after_warmup', 'weather_at_history_start',
                'rig', 'render_preset', 'engine_requested', 'code_sha256'):
        if key in meta_a or key in meta_b:
            if meta_a.get(key) != meta_b.get(key):
                failures.append({'group': 'meta', 'field': key, 'exact': False})
    return {'exact': not failures, 'fields': fields, 'failures': failures}


def audit(data, scenes=None, thresholds=None, tile_size=100, final_rehash=True, *, repeat_only=False):
    """Audit full common-history coverage, or an explicitly scoped repeat preflight.

    Repeat-only keeps the original frozen manifest/config intact and does not
    require the intervention branches to exist. Its pass is never a full-scene
    delivery pass; the default full audit must still run after their collection.
    """
    data = Path(data).resolve()
    require(isinstance(repeat_only, bool), 'repeat_only must be boolean')
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    require(thresholds.keys() == DEFAULT_THRESHOLDS.keys(), 'Threshold fields differ')
    require(all(np.isfinite(v) and v >= 0 for v in thresholds.values()), 'Thresholds must be finite and nonnegative')
    require(isinstance(tile_size, int) and tile_size > 0, 'Tile size must be positive')
    inputs = Inputs(data)
    report = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'data_root': str(data), 'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'thresholds': thresholds, 'tile_size_pixels': tile_size,
              'threshold_semantics': 'Fixed engineering tolerances, not statistically estimated noise; no automatic relaxation.',
              'threshold_identity_sha256': canonical_hash({'thresholds': thresholds, 'tile_size_pixels': tile_size}),
              'audit_mode': 'repeat_only' if repeat_only else 'full',
              'scope': ('Preflight only: maintain versus maintain_repeat, complete sequence for all six cameras; no intervention branch coverage.'
                        if repeat_only else 'All non-maintain branch common histories including t0; maintain_repeat full sequence. No duplicate repeat-history pairs.'),
              'state_boundary': 'Observed states/calibration compare through t0; next-interval control/target/requested fields compare strictly before t0. Repeat compares all.',
              'scenes': [], 'pairs': [], 'failures': [], 'input_files': inputs.records,
              'status': 'failed', 'final_rehash': {'performed': False}}
    try:
        manifest = inputs.json(data/'manifest.json')
        planned = manifest['scene_ids']
        selected = list(planned if scenes is None else scenes)
        require(selected and len(selected) == len(set(selected)), 'Require nonempty unique scenes')
        require(set(selected) <= set(planned), 'Requested scenes absent from manifest')
        require(all(Path(s).name == s and s not in {'.', '..'} for s in selected), 'Invalid scene ID')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report['failures'].append({'stage': 'manifest', 'error': str(exc)})
        selected = []
    for scene in selected:
        scene_report = {'scene_id': scene, 'branches': [], 'pair_count': 0}
        report['scenes'].append(scene_report)
        try:
            config = inputs.json(data/scene/'spec.json')
            require(config['scene_id'] == scene, 'Scene ID mismatch')
            require(canonical_hash({k: v for k, v in config.items() if k != 'spec_sha256'}) == config['spec_sha256'],
                    'Invalid frozen config hash')
            names = config['branches']
            require(len(names) == len(set(names)) and {'maintain', 'maintain_repeat'} <= set(names), 'Missing/duplicate reference or repeat branch')
            require(all(Path(n).name == n and n not in {'.', '..'} for n in names), 'Invalid branch name')
            require(manifest.get('scene_branches', {}).get(scene, names) == names, 'Manifest/config branch mismatch')
            rig = config['camera_rig']
            require([c['name'] for c in rig] == list(CAMERAS), 'Require frozen six-camera ordering')
            root = data/scene/'branches'
            reference = load_branch(inputs, root/'maintain', config, 'maintain')
            reference_images = {}
            for name in (['maintain_repeat'] if repeat_only else names):
                if name == 'maintain':
                    continue
                branch_report = {'branch': name, 'status': 'failed'}
                scene_report['branches'].append(branch_report)
                try:
                    actual = load_branch(inputs, root/name, config, name)
                    comparison = compare_branch(reference, actual, name == 'maintain_repeat')
                    branch_report['physical_and_camera_comparison'] = comparison
                    if not comparison['exact']:
                        report['failures'].append({'scene_id': scene, 'branch': name, 'stage': 'physical_or_calibration',
                                                   'fields': comparison['failures']})
                    indices = selected_indices(reference[2]['time_s'], name == 'maintain_repeat')
                    branch_failed = not comparison['exact']
                    for i in indices:
                        for camera in rig:
                            relative = Path('camera')/camera['name']/f'{i:05d}.png'
                            ref_path, path = root/'maintain'/relative, root/name/relative
                            pair = {'scene_id': scene, 'branch': name, 'camera': camera['name'],
                                    'sample_index': int(i), 'time_s': float(reference[2]['time_s'][i]),
                                    'reference_path': str(ref_path.relative_to(data)), 'actual_path': str(path.relative_to(data))}
                            try:
                                cache_key = (int(i), camera['name'])
                                if cache_key not in reference_images:
                                    reference_images[cache_key] = inputs.png(ref_path, camera['width'], camera['height'])
                                observed = inputs.png(path, camera['width'], camera['height'])
                                metrics = image_metrics(reference_images[cache_key], observed, tile_size)
                                failures = metric_failures(metrics, thresholds)
                                pair.update(metrics=metrics, status='failed' if failures else 'passed', exceeded_thresholds=failures)
                            except (OSError, ValueError) as exc:
                                pair.update(status='failed', error=str(exc))
                            if pair['status'] == 'failed':
                                branch_failed = True
                                report['failures'].append({'stage': 'rgb_pair', **pair})
                            report['pairs'].append(pair)
                            scene_report['pair_count'] += 1
                    branch_report['status'] = 'failed' if branch_failed else 'passed'
                except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                    branch_report['error'] = str(exc)
                    report['failures'].append({'scene_id': scene, 'branch': name, 'stage': 'branch_inputs', 'error': str(exc)})
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            scene_report['error'] = str(exc)
            report['failures'].append({'scene_id': scene, 'stage': 'scene_inputs', 'error': str(exc)})
    if final_rehash:
        changed = inputs.recheck()
        report['final_rehash'] = {'performed': True, 'file_count': len(inputs.records), 'failures': changed}
        report['failures'].extend({'stage': 'final_input_rehash', **failure} for failure in changed)
    else:
        report['failures'].append({'stage': 'final_input_rehash', 'error': 'Final input rehash was disabled; cannot issue pass'})
    report['input_manifest_sha256'] = canonical_hash(inputs.records)
    report['summary'] = {'scene_count': len(report['scenes']), 'pair_count': len(report['pairs']),
                         'failed_rgb_pair_count': sum(p['status'] == 'failed' for p in report['pairs']),
                         'failure_count': len(report['failures']), 'input_file_count': len(inputs.records)}
    report['status'] = 'passed' if selected and report['pairs'] and not report['failures'] else 'failed'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='JSON report path')
    parser.add_argument('--scenes', nargs='+')
    parser.add_argument('--repeat-only', action='store_true',
                        help='Preflight maintain versus maintain_repeat only; cannot establish full-scene delivery')
    parser.add_argument('--max-mae', type=float, default=.25)
    parser.add_argument('--max-tile-mae', type=float, default=4.)
    parser.add_argument('--max-fraction-gt32', type=float, default=.001)
    parser.add_argument('--max-fraction-gt8', type=float, default=.005)
    parser.add_argument('--tile-size', type=int, default=100)
    args = parser.parse_args()
    thresholds = {'mae_0_255': args.max_mae, 'max_tile_mae_0_255': args.max_tile_mae,
                  'fraction_any_channel_gt_32': args.max_fraction_gt32,
                  'fraction_any_channel_gt_8': args.max_fraction_gt8}
    try:
        result = audit(args.data, args.scenes, thresholds, args.tile_size, repeat_only=args.repeat_only)
    except (OSError, ValueError) as exc:
        printable_thresholds = {k: v if np.isfinite(v) else repr(v) for k, v in thresholds.items()}
        result = {'status': 'failed', 'audit_mode': 'repeat_only' if args.repeat_only else 'full',
                  'error': str(exc), 'thresholds': printable_thresholds,
                  'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix+'.tmp')
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(args.out)
    print(json.dumps({'status': result['status'], 'audit_mode': result['audit_mode'],
                      **result.get('summary', {}), 'report': str(args.out)}))
    raise SystemExit(0 if result['status'] == 'passed' else 2)


if __name__ == '__main__':
    main()
