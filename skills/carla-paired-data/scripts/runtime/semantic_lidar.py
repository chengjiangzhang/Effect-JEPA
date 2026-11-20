"""CARLA 0.9.16 semantic LiDAR: native packets, synchronization and read-only QA.

Four little-endian float32 fields followed by two uint32 fields per point.
No intensity/dropoff/noise model is invented: semantic LiDAR has none.
Unmapped ObjIdx values remain raw, branch-local object IDs, not stable tracks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

NAME = 'LIDAR_TOP'
DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                  ('CosAngle', '<f4'), ('ObjIdx', '<u4'), ('ObjTag', '<u4')])
DEFAULT_SPEC = {
    'name': NAME, 'blueprint_id': 'sensor.lidar.ray_cast_semantic',
    'transform': {'position': [0., 0., 2.4], 'rotation_deg': [0., 0., 0.]},
    'attributes': {'channels': 64, 'range': 80., 'points_per_second': 1280000,
                   'rotation_frequency': 20., 'upper_fov': 10., 'lower_fov': -30.,
                   'horizontal_fov': 360., 'sensor_tick': 0.},
    'saved_hz': 2., 'capture_hz': 20.,
    'scan': 'One complete 360-degree ray-cast sweep per physics tick; save at RGB times.',
    'coordinates': 'Sensor-local CARLA axes: x forward, y right, z up; metres.',
    'fields': list(DTYPE.names),
}


def default_spec():
    return copy.deepcopy(DEFAULT_SPEC)


def rigid_from_spec(spec):
    roll, pitch, yaw = np.deg2rad(spec['rotation_deg'])
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = [[cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr],
                      [cp*sy, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr],
                      [sp, -cp*sr, cp*cr]]
    result[:3, 3] = spec['position']
    return result


def validate_spec(spec, physics_dt_s=.05, rgb_interval_s=.5):
    if not isinstance(spec, dict) or spec.get('name') != NAME or spec.get('blueprint_id') != DEFAULT_SPEC['blueprint_id']:
        raise ValueError('Mandatory LIDAR_TOP semantic LiDAR spec missing or invalid')
    attrs = spec.get('attributes')
    if not isinstance(attrs, dict) or set(attrs) != set(DEFAULT_SPEC['attributes']):
        raise ValueError('Semantic LiDAR requires explicit supported attributes')
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) for v in attrs.values()):
        raise ValueError('LiDAR attributes must be finite numeric values')
    for key in ('channels', 'points_per_second'):
        if not isinstance(attrs[key], int) or attrs[key] <= 0:
            raise ValueError(f'LiDAR {key} must be a positive integer')
    if attrs['range'] <= 0 or not -90 <= attrs['lower_fov'] < attrs['upper_fov'] <= 90:
        raise ValueError('LiDAR range or vertical FOV is invalid')
    if attrs['horizontal_fov'] != 360 or attrs['sensor_tick'] != 0 or not np.isclose(attrs['rotation_frequency']*physics_dt_s, 1.):
        raise ValueError('LiDAR must produce one full 360-degree scan on every physics tick')
    if not np.isclose(spec.get('capture_hz', 0)*physics_dt_s, 1.) or not np.isclose(spec.get('saved_hz', 0)*rgb_interval_s, 1.):
        raise ValueError('LiDAR capture/save rate differs from fixed physics/RGB clocks')
    if attrs['points_per_second']*physics_dt_s/attrs['channels'] < 1:
        raise ValueError('LiDAR needs at least one ray per channel per physics frame')
    transform = spec.get('transform', {})
    for key in ('position', 'rotation_deg'):
        value = np.asarray(transform.get(key, []))
        if value.shape != (3,) or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f'LiDAR transform.{key} must contain three finite numbers')
    if spec.get('fields') != list(DTYPE.names):
        raise ValueError('LiDAR fields must preserve the native binary semantic layout')
    return spec


def blueprint(library, spec):
    validate_spec(spec)
    result = library.find(spec['blueprint_id'])
    attributes = {'role_name': spec['name'], **spec['attributes']}
    for key, value in attributes.items():
        if not result.has_attribute(key):
            raise RuntimeError(f'Required semantic LiDAR attribute missing: {key}')
        result.set_attribute(key, str(value))
    return result, attributes


def decode(raw):
    if len(raw) % DTYPE.itemsize:
        raise ValueError(f'Truncated semantic LiDAR buffer: {len(raw)} bytes is not a multiple of {DTYPE.itemsize}')
    points = np.frombuffer(raw, dtype=DTYPE).copy()
    if any(not np.isfinite(points[key]).all() for key in ('x', 'y', 'z', 'CosAngle')):
        raise ValueError('Semantic LiDAR contains non-finite point values')
    if (np.abs(points['CosAngle']) > 1.00001).any():
        raise ValueError('Semantic LiDAR incidence cosine lies outside [-1,1]')
    return points


def packet_arrays(measurement, *, spec, expected_frame, expected_timestamp, relative_time):
    validate_spec(spec)
    if int(measurement.frame) != expected_frame or not np.isclose(float(measurement.timestamp), expected_timestamp, rtol=0, atol=1e-6):
        raise ValueError('LiDAR packet frame/time differs from the saved world/RGB sample')
    points = decode(measurement.raw_data)
    channels = int(measurement.channels)
    if channels != spec['attributes']['channels']:
        raise ValueError('LiDAR packet channel count differs from frozen spec')
    counts = np.asarray([measurement.get_point_count(i) for i in range(channels)], dtype=np.int64)
    if (counts < 0).any() or int(counts.sum()) != len(points):
        raise ValueError('LiDAR per-channel counts disagree with native point buffer')
    world = np.asarray(measurement.transform.get_matrix(), dtype=np.float64)
    if world.shape != (4, 4) or not np.isfinite(world).all():
        raise ValueError('LiDAR measurement world pose is invalid')
    return {'points': points, 'frame_id': np.asarray(expected_frame, dtype=np.int64),
            'timestamp_s': np.asarray(expected_timestamp, dtype=np.float64),
            'time_s': np.asarray(relative_time, dtype=np.float64),
            'lidar_to_world': world, 'lidar_to_ego': rigid_from_spec(spec['transform']),
            'channels': np.asarray(channels, dtype=np.int64),
            'horizontal_angle': np.asarray(measurement.horizontal_angle, dtype=np.float64),
            'point_count_by_channel': counts}


def save_packet(measurement, path, **kwargs):
    arrays = packet_arrays(measurement, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    return len(arrays['points'])


def metadata(spec, requested_attributes, actor_attributes, actor_roles):
    return {**copy.deepcopy(spec), 'attributes': dict(requested_attributes),
            'actor_attributes': dict(actor_attributes),
            'actor_to_role': {str(int(key)): str(value) for key, value in sorted(actor_roles.items())},
            'identity_note': 'Only spawned vehicle/walker ObjIdx IDs map to stable scene roles. All other IDs remain raw branch-local or map-object IDs; no cross-branch track identity is inferred.',
            'files': f'lidar/{spec["name"]}/00000.npz..00012.npz',
            'native_record_bytes': DTYPE.itemsize, 'point_dtype': DTYPE.descr,
            'projection': 'Ray azimuth/elevation calibration in attributes; no pinhole intrinsic K.',
            'samples': 13, 'sensor_frames': 121}


def inspect_branch(folder, spec_config, meta, states, samples):
    """Audit actual native arrays without opening a simulator or rewriting data."""
    errors = []
    result = {'status': 'failed', 'errors': errors, 'sample_count': 0, 'point_count': 0, 'input_sha256': {}}
    try:
        spec = validate_spec(spec_config.get('semantic_lidar'))
        record = meta.get('semantic_lidar')
        if not isinstance(record, dict):
            raise ValueError('Missing semantic_lidar sensor metadata')
        expected_sensor_names = [item['name'] for item in spec_config.get('camera_rig', [])] + [spec['name']]
        if spec_config.get('camera_rig') and meta.get('sensor_names') != expected_sensor_names:
            errors.append('Recorded six RGB plus LiDAR sensor inventory/order differs')
        if record.get('blueprint_id') != spec['blueprint_id'] or record.get('name') != spec['name']:
            errors.append('LiDAR sensor identity differs from frozen spec')
        expected_attributes = {'role_name': spec['name'], **spec['attributes']}
        for scope in ('attributes', 'actor_attributes'):
            attrs = record.get(scope, {})
            for key, value in expected_attributes.items():
                actual = attrs.get(key)
                equal = str(actual) == str(value) if key == 'role_name' else actual is not None and np.isclose(float(actual), float(value), rtol=1e-6, atol=1e-6)
                if not equal:
                    errors.append(f'LiDAR {scope}.{key} differs from frozen spec')
        expected_role_map = {str(actor): role for actor, role in zip(meta.get('actor_ids', []), spec_config['vehicle_roles'])}
        expected_role_map.update({str(item['actor_id']): item['role'] for item in meta.get('pedestrians', [])})
        if len(expected_role_map) != len(spec_config['vehicle_roles']) + len(spec_config.get('pedestrians', [])):
            errors.append('LiDAR stable role mapping lacks actual actor identities')
        if record.get('actor_to_role') != expected_role_map:
            errors.append('LiDAR native actor IDs do not match captured stable vehicle/walker roles')
        indices = np.arange(0, 121, 10)
        if np.asarray(samples.get('sensor_frame_id', [])).shape != (13, 7):
            errors.append('Six RGB plus LiDAR sensor frame matrix must be 13x7')
        shapes = {'lidar_to_world': (13, 4, 4), 'lidar_to_ego': (4, 4), 'lidar_frame_id': (13,), 'lidar_timestamp_s': (13,)}
        for key, shape in shapes.items():
            value = samples.get(key)
            if value is None or value.shape != shape or not np.isfinite(value).all():
                errors.append(f'Missing/invalid LiDAR samples.{key}')
        if errors:
            return result
        local = rigid_from_spec(spec['transform'])
        if not np.allclose(samples['lidar_to_ego'], local, rtol=0, atol=1e-6):
            errors.append('LiDAR extrinsic differs from frozen installation')
        expected_world = samples['ego_to_world'] @ local
        translation_error = float(np.max(np.linalg.norm(samples['lidar_to_world'][:, :3, 3]-expected_world[:, :3, 3], axis=-1)))
        rotation_error = float(np.max(np.abs(samples['lidar_to_world'][:, :3, :3]-expected_world[:, :3, :3])))
        result['calibration'] = {'translation_max_m': translation_error, 'rotation_matrix_max_error': rotation_error}
        # Same engineering bounds as the existing RGB rigid-rig calibration.
        if translation_error > .002 or rotation_error > 1e-4:
            errors.append('LiDAR measured world pose disagrees with rigid ego-mounted extrinsic')
        if not np.issubdtype(samples['lidar_frame_id'].dtype, np.integer):
            errors.append('LiDAR frame IDs must use integer storage')
        if not np.array_equal(samples['lidar_frame_id'], states['frame_id'][indices]) or not np.array_equal(samples['sensor_frame_id'][:, 6], samples['lidar_frame_id']):
            errors.append('LiDAR sampled frame IDs differ from RGB/physics')
        if not np.allclose(samples['lidar_timestamp_s'], states['simulation_time_s'][indices], rtol=0, atol=1e-6):
            errors.append('LiDAR sampled timestamps differ from physics')
        directory = Path(folder)/'lidar'/spec['name']
        files = sorted(directory.glob('*.npz'))
        if [p.name for p in files] != [f'{i:05d}.npz' for i in range(13)]:
            raise ValueError('Require exactly 13 aligned semantic LiDAR NPZ files')
        for index, path in enumerate(files):
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            result['input_sha256'][str(path.relative_to(folder))] = checksum
            with np.load(path, allow_pickle=False) as opened:
                packet = {key: opened[key] for key in opened.files}
            points = packet.get('points')
            if points is None or points.ndim != 1 or points.dtype != DTYPE:
                errors.append(f'{path.name}: invalid native structured points dtype')
                continue
            decode(points.tobytes())
            for key, expected in (('frame_id', samples['lidar_frame_id'][index]), ('time_s', -2+index*.5),
                                  ('timestamp_s', samples['lidar_timestamp_s'][index]), ('channels', spec['attributes']['channels'])):
                value = packet.get(key)
                if value is None or value.shape != () or not np.isfinite(value) or not np.isclose(value, expected, rtol=0, atol=1e-6):
                    errors.append(f'{path.name}: {key} differs from synchronized sample')
            for key in ('frame_id', 'channels'):
                value = packet.get(key)
                if value is None or not np.issubdtype(value.dtype, np.integer):
                    errors.append(f'{path.name}: {key} must use integer storage')
            for key, expected in (('lidar_to_world', samples['lidar_to_world'][index]), ('lidar_to_ego', local)):
                value = packet.get(key)
                if value is None or value.shape != (4, 4) or not np.allclose(value, expected, rtol=0, atol=1e-6):
                    errors.append(f'{path.name}: invalid {key}')
            counts = packet.get('point_count_by_channel')
            if counts is None or counts.shape != (spec['attributes']['channels'],) or not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any() or counts.sum() != len(points):
                errors.append(f'{path.name}: per-channel counts disagree with buffer')
            angle = packet.get('horizontal_angle')
            if angle is None or angle.shape != () or not np.isfinite(angle):
                errors.append(f'{path.name}: invalid horizontal angle')
            if checksum != hashlib.sha256(path.read_bytes()).hexdigest():
                errors.append(f'{path.name}: raw LiDAR input changed during audit')
            result['sample_count'] += 1
            result['point_count'] += len(points)
        if (Path(folder)/'preview').exists() or 'preview' in meta:
            errors.append('Overhead RGB preview is excluded from this capture contract')
    except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, zipfile.BadZipFile) as exc:
        errors.append(f'Semantic LiDAR: {type(exc).__name__}: {exc}')
    result['status'] = 'failed' if errors else 'passed'
    return result
