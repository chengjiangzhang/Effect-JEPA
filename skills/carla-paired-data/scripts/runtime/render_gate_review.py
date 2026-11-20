"""Read-only, fail-closed binding of the native RGB gate to current raw bytes.

No simulator access and no replacement audit: missing, partial or stale evidence
cannot grant a pass. This also checks every rendered physics frame, not just the
13 saved RGB times. Legacy configurations without the policy are unchanged.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from . import audit_render_consistency as gate
else:
    import audit_render_consistency as gate


POLICY = {'schema_version': 1, 'policy_id': 'native-rgb-repeat-v1',
          'tile_size_pixels': 100, 'thresholds': dict(gate.DEFAULT_THRESHOLDS)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def expected_sensor_tick(config):
    return 0. if 'render_consistency_policy' in config else .05


def render_history_errors(meta, states, config):
    if 'render_consistency_policy' not in config:
        return []
    errors = []
    history = meta.get('render_history')
    if not isinstance(history, dict):
        return ['missing complete render_history ledger']
    if history.get('status') != 'complete' or history.get('physics_frames_checked') != 121:
        errors.append('render_history must cover all 121 physics frames')
    tick = history.get('sensor_tick_s')
    if isinstance(tick, bool) or not isinstance(tick, (int, float)) or tick != 0.:
        errors.append('render_history.sensor_tick_s must be zero')
    names = [c['name'] for c in config['camera_rig']] + [config['semantic_lidar']['name']]
    cameras = history.get('cameras')
    if not isinstance(cameras, list) or [c.get('camera') if isinstance(c, dict) else None for c in cameras] != names:
        return errors + ['render_history requires ordered unique six cameras and semantic LiDAR']
    for name, camera in zip(names, cameras):
        frames = np.asarray(camera.get('frames', []))
        timestamps = np.asarray(camera.get('timestamps_s', []))
        if (frames.shape != (121,) or not np.issubdtype(frames.dtype, np.integer)
                or not np.array_equal(frames, states['frame_id'])):
            errors.append(f'{name}: render_history frame IDs differ from all 121 physical frames')
        if (timestamps.shape != (121,) or not np.issubdtype(timestamps.dtype, np.number)
                or not np.isfinite(timestamps).all()
                or not np.allclose(timestamps, states['simulation_time_s'], rtol=0, atol=1e-6)):
            errors.append(f'{name}: render_history timestamps differ from all 121 physical times')
    return errors


def checked_render_gate(data, run, scene, config):
    """Validate a scene-scoped batch gate, including a fresh SHA sweep.

The result is current at this check; it is not a promise about later mutations.
RGB metric calculations remain in the original source-identified audit report.
"""
    if 'render_consistency_policy' not in config:
        return {'required': False, 'status': 'not_required_legacy'}
    data, run = Path(data).resolve(), Path(run).resolve()
    path = run/'render_consistency'/f'{scene}.json'
    result = {'required': True, 'status': 'failed', 'report_path': str(path),
              'validator_source_sha256': sha(__file__),
              'audit_source_sha256': sha(gate.__file__), 'raw_sha_rechecked': False}
    require = gate.require
    try:
        require(gate.canonical_hash(config['render_consistency_policy']) == gate.canonical_hash(POLICY),
                'Policy differs from frozen native-rgb-repeat-v1 engineering tolerances')
        if not path.is_file():
            return {**result, 'status': 'pending', 'reason': 'Native RGB gate report has not been generated'}
        report_bytes = path.read_bytes()
        result['report_sha256'] = hashlib.sha256(report_bytes).hexdigest()
        report = json.loads(report_bytes)
        require(isinstance(report, dict), 'Native RGB gate report must be a JSON object')
        if report.get('status') == 'partial':
            return {**result, 'status': 'pending', 'reason': 'Native RGB gate has incomplete branch coverage'}
        if report.get('status') != 'passed':
            return {**result, 'reason': 'Native RGB gate reported failure',
                    'reported_status': report.get('status'), 'failures': report.get('failures', report.get('errors', []))}
        require(report.get('schema_version') == 1 and Path(report['data_root']).resolve() == data,
                'Gate schema or dataset root differs')
        require(report['audit_source_sha256'] == result['audit_source_sha256'], 'Gate audit source SHA is stale')
        expected_thresholds = {'thresholds': POLICY['thresholds'], 'tile_size_pixels': POLICY['tile_size_pixels']}
        require(all(report.get(k) == v for k, v in expected_thresholds.items())
                and report['threshold_identity_sha256'] == gate.canonical_hash(expected_thresholds),
                'Gate used different engineering tolerances')
        require(not report.get('failures'), 'Passed gate contains failures')
        require(report['final_rehash']['performed'] is True and not report['final_rehash']['failures'],
                'Gate omitted its final input SHA sweep')
        inputs = gate.Inputs(data)
        manifest = inputs.json(data/'manifest.json')
        actual_config = inputs.json(data/scene/'spec.json')
        require(actual_config == config and actual_config['scene_id'] == scene, 'Current config differs')
        require(gate.canonical_hash({k: v for k, v in config.items() if k != 'spec_sha256'}) == config['spec_sha256'],
                'Invalid current config hash')
        require(scene in manifest['scene_ids'] and manifest['scene_branches'][scene] == config['branches'],
                'Manifest branch inventory differs')
        names = config['branches']
        require(len(names) == len(set(names)) == 6 and {'maintain', 'maintain_repeat'} <= set(names),
                'Policy requires five actions and one repeat')
        require(all(Path(n).name == n and n not in {'.', '..'} for n in [scene]+names), 'Invalid scene or branch name')
        loaded = {name: gate.load_branch(inputs, data/scene/'branches'/name, config, name) for name in names}
        reference = loaded['maintain']
        require(len(reference[1]['time_s']) == 121 and len(reference[2]['time_s']) == 13,
                'Policy requires 121 physical frames and 13 saved RGB times')
        history = gate.selected_indices(reference[2]['time_s'])
        require(len(history) == 5 and reference[2]['time_s'][history[-1]] == 0., 'Policy requires five common-history RGB times through t0')
        require([c['name'] for c in config['camera_rig']] == list(gate.CAMERAS), 'Six camera inventory differs')
        for name, (meta, state, _) in loaded.items():
            errors = render_history_errors(meta, state, config)
            require(not errors, f'{name}: '+ '; '.join(errors))
            if name != 'maintain':
                # Requested t0 controls legitimately differ; observed t0 does not.
                require(gate.compare_branch(reference, loaded[name], name == 'maintain_repeat')['exact'],
                        f'{name}: observed history/repeat states or calibration differ')
        expected = {}
        expected_files = set(inputs.records)
        for name in names:
            if name == 'maintain':
                continue
            for index in gate.selected_indices(reference[2]['time_s'], name == 'maintain_repeat'):
                for camera in gate.CAMERAS:
                    paths = [str(Path(scene)/'branches'/b/'camera'/camera/f'{index:05d}.png') for b in ('maintain', name)]
                    expected[(name, camera, int(index))] = (*paths, float(reference[2]['time_s'][index]))
                    expected_files.update(paths)
        scenes = report['scenes']
        require(len(scenes) == 1 and scenes[0]['scene_id'] == scene and scenes[0]['pair_count'] == len(expected),
                'Gate scene or pair coverage differs')
        branch_reports = scenes[0]['branches']
        require(len(branch_reports) == 5 and {b['branch'] for b in branch_reports} == set(names)-{'maintain'}
                and all(b['status'] == 'passed' and b['physical_and_camera_comparison']['exact'] is True for b in branch_reports),
                'Gate branch comparisons are incomplete or failed')
        require(len(report['pairs']) == len(expected), 'Gate lacks complete RGB pair coverage')
        seen = set()
        for pair in report['pairs']:
            key = (pair['branch'], pair['camera'], pair['sample_index'])
            require(key in expected and key not in seen, 'Gate has missing, duplicate or unexpected RGB pairs')
            seen.add(key)
            require(pair['scene_id'] == scene and (pair['reference_path'], pair['actual_path'], pair['time_s']) == expected[key],
                    'Gate RGB pair identity differs')
            require(pair['status'] == 'passed' and not pair.get('exceeded_thresholds') and not pair.get('error'),
                    'Gate contains failed RGB pairs')
            metrics = pair['metrics']
            require(all(np.isfinite(metrics[k]) and 0 <= metrics[k] <= limit for k, limit in POLICY['thresholds'].items()),
                    'Gate pair metrics exceed frozen tolerances')
        records = report['input_files']
        require(set(records) == expected_files, 'Gate input SHA inventory is incomplete or contains unrelated files')
        require(report['input_manifest_sha256'] == gate.canonical_hash(records), 'Gate input manifest SHA differs')
        require(report['final_rehash']['file_count'] == len(records), 'Gate final rehash count differs')
        require(report['summary'] == {'scene_count': 1, 'pair_count': len(expected), 'failed_rgb_pair_count': 0,
                'failure_count': 0, 'input_file_count': len(records)}, 'Gate summary coverage differs')
        # Hash every compared PNG, even if size and modification time are unchanged.
        for relative, identity in records.items():
            current = data/relative
            require(current.resolve().is_relative_to(data), 'Gate input path escapes dataset')
            if relative in inputs.records:
                require(identity == inputs.records[relative], f'Stale gate input: {relative}')
            else:
                require(current.stat().st_size == identity['bytes'] and sha(current) == identity['sha256'],
                        f'Stale gate input: {relative}')
        require(path.read_bytes() == report_bytes, 'Gate report changed during validation')
        return {**result, 'status': 'passed', 'raw_sha_rechecked': True,
                'input_manifest_sha256': report['input_manifest_sha256'], 'input_file_count': len(records),
                'pair_count': len(expected), 'render_history_frames_per_branch': 121,
                'saved_rgb_times_per_branch': 13, **expected_thresholds}
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        return {**result, 'status': 'stale' if result.get('report_sha256') else 'failed', 'reason': str(exc)}
