"""Offline protocol regression tests; optional real metadata via --data PATH.

All mutations are confined to fresh temporary metadata/NPZ copies. No RGB,
CARLA server, production dataset modifications or future-region labels are used.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

if __package__:
    from . import audit_protocol as protocol
    from .test_semantic_lidar import add_lidar_fixture
else:
    import audit_protocol as protocol
    from test_semantic_lidar import add_lidar_fixture


CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
BRANCHES = ('maintain', 'brake', 'accelerate', 'left', 'right', 'maintain_repeat')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def fixture(data):
    cameras = [{'name': name, 'width': 1600, 'height': 900, 'fov_deg': 70.,
                'transform': {'position': [float(i) / 10, 0., 1.6],
                              'rotation_deg': [0., 0., float(i) * 60.]}}
               for i, name in enumerate(CAMERAS)]
    config = {'camera_rig': cameras, 'vehicle_roles': ['ego', 'rear'],
              'code_sha256': {'capture.py': 'frozen-fixture-code'},
              'render': {'camera_attributes': {'gamma': 2.2}},
              'render_consistency_policy': {},
              'timing': {'weather_warmup_s': 1., 'renderer_prime_s': 1.},
              'weather': {'cloudiness': 5., 'sun_altitude_angle': 45.},
              'branches': list(BRANCHES)}
    config['semantic_lidar'] = protocol.semantic_lidar.default_spec()
    config['spec_sha256'] = hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    states = {'frame_id': np.arange(1000, 1121),
              'transform_world_vehicle': np.broadcast_to(np.eye(4), (121, 2, 4, 4)).copy()}
    states['transform_world_vehicle'][:, 0, 0, 3] = np.arange(121) * .2
    extrinsics = np.array([protocol.rigid_from_spec(c['transform']) for c in cameras])
    intrinsics = []
    for camera in cameras:
        focal = camera['width'] / (2 * np.tan(np.deg2rad(camera['fov_deg']) / 2))
        intrinsics.append([[focal, 0., 800.], [0., focal, 450.], [0., 0., 1.]])
    indices = np.arange(0, 121, 10)
    poses = states['transform_world_vehicle'][indices, 0]
    samples = {'physics_indices': indices, 'ego_to_world': poses,
               'camera_to_ego': extrinsics, 'intrinsics': np.array(intrinsics),
               'world_to_camera': np.linalg.inv(poses[:, None] @ extrinsics[None])}
    rig = []
    for camera in cameras:
        attrs = {'gamma': '2.2', 'role_name': camera['name'], 'image_size_x': '1600',
                 'image_size_y': '900', 'fov': '70', 'sensor_tick': '0',
                 'enable_postprocess_effects': 'true'}
        rig.append({'name': camera['name'], 'attributes': attrs.copy(),
                    'actor_attributes': attrs.copy()})
    prime = {'status': 'complete', 'render_frames': 20, 'images_saved': 0,
             'carla_client_version': '0.9.16', 'carla_server_version': '0.9.16',
             'reset_after': 'reload_world(False) before the ordinary branch initialization',
             'renderer_prime_s': 1., 'physics_dt_s': .05, 'weather_warmup_frames': 20,
             'first_frame': 101, 'last_frame': 120, 'final_simulation_time_s': 6.,
             'camera_names': list(CAMERAS), 'rig': rig,
             'weather_after_warmup': config['weather'], 'weather_actual': config['weather'],
             'sensor_final_frames': {name: 120 for name in CAMERAS},
             'sensor_last_samples': {name: {'frame_id': 120, 'width': 1600, 'height': 900,
                                            'buffer_bytes': 1600 * 900 * 4, 'timestamp': 6.}
                                     for name in CAMERAS}}
    write_json(data / 'manifest.json', {'scene_ids': ['scene_0000'],
                                      'scene_branches': {'scene_0000': list(BRANCHES)}})
    write_json(data / 'scene_0000/spec.json', config)
    for branch in BRANCHES:
        folder = data / 'scene_0000/branches' / branch
        meta = {'status': 'complete', 'branch': branch,
                'carla_client_version': '0.9.16', 'carla_server_version': '0.9.16',
                'spec_sha256': config['spec_sha256'], 'code_sha256': config['code_sha256'],
                'vehicle_roles': config['vehicle_roles'], 'camera_names': list(CAMERAS),
                'rig': rig, 'renderer_warmup': prime,
                **{key: config['weather'] for key in
                   ('weather_actual', 'weather_after_warmup', 'weather_at_history_start')}}
        add_lidar_fixture(folder, config, meta, states, samples)
        write_json(folder / 'meta.json', meta)
        np.savez_compressed(folder / 'states.npz', **states)
        np.savez_compressed(folder / 'samples.npz', **samples)


class ProtocolTests(unittest.TestCase):
    @contextmanager
    def data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'data'
            fixture(root)
            yield root

    def change_meta(self, data, change):
        path = data / 'scene_0000/branches/maintain/meta.json'
        value = json.loads(path.read_text()); change(value); write_json(path, value)

    def change_samples(self, data, change):
        path = data / 'scene_0000/branches/maintain/samples.npz'
        value = protocol.arrays(path); change(value); np.savez_compressed(path, **value)

    def assert_rejected(self, data, reason=None):
        result = protocol.audit(data)
        self.assertEqual(result['status'], 'failed', result)
        self.assertTrue(result['failures'])
        if reason:
            self.assertIn(reason, json.dumps(result['failures']))
        return result

    def test_complete_six_branch_capture_passes_read_only(self):
        with self.data() as data:
            before = inventory(data)
            result = protocol.audit(data)
            self.assertEqual(result['status'], 'passed', result)
            self.assertEqual(result['summary'], {'expected_branches': 6, 'passed_branches': 6,
                                                 'failed_branches': 0})
            self.assertEqual(inventory(data), before)

    def test_missing_required_metadata_is_rejected(self):
        for key in ('status', 'branch', 'spec_sha256', 'code_sha256', 'vehicle_roles',
                    'camera_names', 'rig', 'weather_actual', 'renderer_warmup'):
            with self.subTest(key=key), self.data() as data:
                self.change_meta(data, lambda value: value.pop(key))
                self.assert_rejected(data)

    def test_capture_client_and_server_version_must_match(self):
        for key in ('carla_client_version', 'carla_server_version'):
            for value in ('0.9.15', None):
                with self.subTest(key=key, value=value), self.data() as data:
                    self.change_meta(data, lambda meta: meta.__setitem__(key, value))
                    self.assert_rejected(data)

    def test_malformed_rig_rejected_with_structured_diagnostic(self):
        for value in (None, 'CAM_FRONT', 42):
            with self.subTest(value=value), self.data() as data:
                self.change_meta(data, lambda meta: meta['rig'].__setitem__(0, value))
                result = self.assert_rejected(data)
                self.assertEqual(result['summary']['failed_branches'], 1)

    def test_requested_and_actor_attributes_cannot_override_frozen_setting(self):
        for scopes in (('attributes',), ('actor_attributes',), ('attributes', 'actor_attributes')):
            with self.subTest(scopes=scopes), self.data() as data:
                def change(meta):
                    for scope in scopes:
                        meta['rig'][0][scope]['image_size_x'] = '800'
                self.change_meta(data, change)
                self.assert_rejected(data, 'CAM_FRONT')

    def test_calibration_intrinsics_and_measured_rig_are_checked(self):
        for key in ('intrinsics', 'world_to_camera'):
            with self.subTest(key=key), self.data() as data:
                def change(samples):
                    samples[key].flat[0] += .1
                self.change_samples(data, change)
                self.assert_rejected(data)

    def test_consistent_but_wrong_installation_transform_is_rejected(self):
        with self.data() as data:
            def change(samples):
                samples['camera_to_ego'][0, 0, 3] += .2
                samples['world_to_camera'] = np.linalg.inv(
                    samples['ego_to_world'][:, None] @ samples['camera_to_ego'][None])
            self.change_samples(data, change)
            self.assert_rejected(data, 'frozen camera installation transform')

    def test_missing_nonfinite_and_wrong_shape_calibration_rejected(self):
        operations = (lambda samples: samples.pop('intrinsics'),
                      lambda samples: samples['intrinsics'].__setitem__((0, 0, 0), np.nan),
                      lambda samples: samples.__setitem__('world_to_camera', np.eye(4)))
        for operation in operations:
            with self.subTest(operation=operation), self.data() as data:
                self.change_samples(data, operation)
                self.assert_rejected(data, 'calibration shape')

    def test_sampling_indices_and_pose_must_match_physics(self):
        for key in ('physics_indices', 'ego_to_world'):
            with self.subTest(key=key), self.data() as data:
                def change(samples):
                    if key == 'physics_indices': samples[key][1] += 1
                    else: samples[key][2, 0, 3] += .1
                self.change_samples(data, change)
                self.assert_rejected(data)

    def test_config_tampering_and_branch_manifest_mismatch_rejected(self):
        for name in ('config', 'manifest'):
            with self.subTest(name=name), self.data() as data:
                path = data / ('scene_0000/spec.json' if name == 'config' else 'manifest.json')
                value = json.loads(path.read_text())
                if name == 'config': value['weather']['cloudiness'] = 6.
                else: value['scene_branches']['scene_0000'].pop()
                write_json(path, value)
                self.assert_rejected(data, 'hash mismatch' if name == 'config' else 'manifest/config mismatch')

    def test_prime_episode_must_precede_recorded_frames(self):
        with self.data() as data:
            self.change_meta(data, lambda meta: meta['renderer_warmup'].update(
                first_frame=1000, last_frame=1019))
            self.assert_rejected(data, 'discarded prime episode')

    def test_protocol_audit_does_not_filter_or_derive_future_outcomes(self):
        with self.data() as data:
            # All branch arrays are identical: no strong future response exists.
            # A future contact is an observed outcome, not a sensor-protocol failure.
            self.change_meta(data, lambda meta: meta.update(collisions=[{'time_s': 2.0}]))
            before = inventory(data)
            result = protocol.audit(data)
            self.assertEqual(result['status'], 'passed', result)
            def keys(value):
                if isinstance(value, dict):
                    for key, child in value.items():
                        yield key
                        yield from keys(child)
                elif isinstance(value, list):
                    for child in value: yield from keys(child)
            self.assertFalse(any('reactive' in key.lower() or 'ttc' in key.lower() for key in keys(result)))
            self.assertEqual(inventory(data), before)

    def test_cli_rejects_report_inside_raw_data(self):
        with self.data() as data:
            before = inventory(data)
            command = [sys.executable, str(Path(protocol.__file__).resolve()), '--data', str(data),
                       '--out', str(data / 'audit.json')]
            process = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertIn('outside the read-only raw dataset', process.stderr)
            self.assertEqual(inventory(data), before)


@unittest.skipUnless(os.environ.get('CARLA_PROTOCOL_TEST_DATA'), 'Pass --data for real saved metadata tests')
class RecordedMetadataTests(unittest.TestCase):
    @contextmanager
    def copied_data(self):
        source = Path(os.environ['CARLA_PROTOCOL_TEST_DATA']).resolve()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            for path in source.rglob('*'):
                if path.name in ('manifest.json', 'spec.json', 'meta.json', 'states.npz', 'samples.npz'):
                    destination = target / path.relative_to(source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination)
            yield target

    def test_actual_saved_six_branch_metadata_is_accepted_without_images(self):
        with self.copied_data() as data:
            before = inventory(data)
            report = protocol.audit(data)
            self.assertEqual(report['status'], 'passed', report)
            self.assertEqual(report['summary']['passed_branches'], 6)
            self.assertEqual(inventory(data), before)

    def test_actual_saved_calibration_tampering_is_rejected(self):
        with self.copied_data() as data:
            path = next(data.glob('*/branches/maintain/samples.npz'))
            samples = protocol.arrays(path)
            samples['intrinsics'][3, 0, 0] *= 1.1
            np.savez_compressed(path, **samples)
            report = protocol.audit(data)
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['summary']['failed_branches'], 1)
            self.assertIn('intrinsics differ', json.dumps(report['failures']))


if __name__ == '__main__':
    # Parse the optional real-data path before unittest evaluates skip decorators.
    # Setting it here updates the already-created optional class explicitly.
    if '--data' in sys.argv:
        index = sys.argv.index('--data')
        os.environ['CARLA_PROTOCOL_TEST_DATA'] = sys.argv[index + 1]
        del sys.argv[index:index + 2]
        RecordedMetadataTests.__unittest_skip__ = False
    unittest.main()
