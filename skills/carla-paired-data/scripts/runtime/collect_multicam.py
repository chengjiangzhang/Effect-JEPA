#!/usr/bin/env python3
"""CARLA six-camera paired data, 2 Hz RGB and 20 Hz physical telemetry.

Every branch is executed in a new Python client by collect_multicam_batch.py.
No learned model, image synthesis, resizing or lossy RGB compression is used.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback

import carla
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if __package__:
    from .controllers_v2 import ScenarioController, branch_names
    from .auxiliary_actors import AuxiliaryActors
    from . import semantic_lidar
else:
    from controllers_v2 import ScenarioController, branch_names
    from auxiliary_actors import AuxiliaryActors
    import semantic_lidar
if __package__:
    from .scenarios_v2 import build_city_scenes, build_scenes, build_extended_lane_change_scenes
else:
    from scenarios_v2 import build_city_scenes, build_scenes, build_extended_lane_change_scenes

DT = .05
COLLISION_DRAIN_WALL_S = .1
BRANCHES = ('maintain','brake','accelerate','left','right','maintain_repeat')
RIG = [
    ('CAM_FRONT', .8, 0., 1.6, 0., 70.),
    ('CAM_FRONT_LEFT', .27, -.55, 1.6, -55., 70.),
    ('CAM_FRONT_RIGHT', .27, .55, 1.6, 55., 70.),
    ('CAM_BACK', -2., 0., 1.6, 180., 110.),
    ('CAM_BACK_LEFT', -.32, -.55, 1.6, -110., 70.),
    ('CAM_BACK_RIGHT', -.32, .55, 1.6, 110., 70.),
]
DEFAULT_XODR = os.environ.get('CARLA_OPENDRIVE_ROOT') or str(
    Path(os.environ.get('CARLA_ROOT', '.'))/'CarlaUE4/Content/Carla/Maps/OpenDrive')
RESOURCES = HERE/'resources'
# Selected from native RGB calibration, rather than a uniform exposure across
# maps with different baked lighting. Freeze these settings within each scene.
EXPOSURE_BY_MAP = {'Town10HD': 4., 'Town03': 3., 'Town04': 3., 'Town05': 3.}
WEATHER_FIELDS = ('cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity',
                  'sun_azimuth_angle', 'sun_altitude_angle', 'fog_density', 'fog_distance',
                  'fog_falloff', 'wetness', 'scattering_intensity', 'mie_scattering_scale',
                  'rayleigh_scattering_scale', 'dust_storm')
CAMERA_EXPOSURE_FIELDS = frozenset(('exposure_compensation', 'shutter_speed', 'iso', 'fstop'))
NATIVE_NPC_BLUEPRINTS = ('vehicle.lincoln.mkz_2020', 'vehicle.mercedes.coupe_2020',
                         'vehicle.dodge.charger_2020')
NATIVE_NPC_COLORS = ('90,90,90', '170,170,170', '105,25,30', '20,45,80')


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def log_capture_progress(scene, branch, phase, started, **details):
    """Small flushed diagnostic records; never alter the simulation clock."""
    print(json.dumps({'event': 'capture_progress', 'time_utc': datetime.now(timezone.utc).isoformat(),
                      'scene': scene, 'branch': branch, 'phase': phase,
                      'elapsed_wall_s': round(time.monotonic()-started, 3), **details},
                     allow_nan=False), flush=True)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {n:hashlib.sha256((HERE/n).read_bytes()).hexdigest()
            for n in ('collect_multicam.py','scenarios_v2.py','controllers_v2.py','npc_merge.py','auxiliary_actors.py','ego_decisions.py','semantic_lidar.py','pedestrian_policy.py','pedestrian_crossing.py')}


def motion_warmup_seconds(cfg):
    """Scene-authored driving prefix; saved history and future clocks stay fixed."""
    declared = cfg.get('driving_warmup_s', cfg.get('timing', {}).get('motion_warmup_s', 1.))
    if (isinstance(declared, bool) or not isinstance(declared, (int, float))
            or not math.isfinite(declared) or not .5 <= declared <= 3.
            or not math.isclose(declared/DT, round(declared/DT), rel_tol=0, abs_tol=1e-8)):
        raise ValueError('driving_warmup_s must be in [.5,3] and aligned to the .05s physics tick')
    recorded = cfg.get('timing', {}).get('motion_warmup_s', declared)
    if not isinstance(recorded, (int, float)) or isinstance(recorded, bool) or recorded != declared:
        raise ValueError('timing.motion_warmup_s differs from authored driving_warmup_s')
    return float(declared)


def resource_path(value):
    """Resolve portable bundled evidence without embedding an install location."""
    value = str(value)
    if value.startswith('resource://'):
        name = value[len('resource://'):]
        if not name or Path(name).name != name:
            raise ValueError('Resource URI must contain a single packaged filename')
        return (RESOURCES/name).resolve(strict=True)
    return Path(value).resolve(strict=True)


def resource_identity(path):
    path = Path(path).resolve(strict=True)
    return 'resource://'+path.name if path.parent == RESOURCES.resolve() else str(path)


def preset_camera_attributes(preset, map_name):
    """A map-specific exposure remains identical across that scene's branches."""
    return dict(preset.get('camera_attributes_by_map', {}).get(map_name, preset['camera_attributes']))


def scene_camera_attributes(preset, cfg):
    """Apply an explicit scene exposure after map defaults, frozen across branches.

    Camera pose, optics, timing and geometry-related postprocessing stay fixed.
    These values are authored reference parameters, not a photometric validation.
    """
    overrides = cfg.get('camera_override', {})
    if not isinstance(overrides, dict) or not set(overrides).issubset(CAMERA_EXPOSURE_FIELDS):
        raise ValueError(f'camera_override permits only {sorted(CAMERA_EXPOSURE_FIELDS)}')
    for key, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'camera_override.{key} must be finite')
        if key != 'exposure_compensation' and value <= 0:
            raise ValueError(f'camera_override.{key} must be positive')
    if overrides and preset.get('scene_weather_policy') != 'frozen-per-scene':
        raise ValueError('Scene exposure overrides require frozen-per-scene rendering')
    return {**preset_camera_attributes(preset, cfg['map']), **overrides}


def vehicle_light_mode(cfg):
    mode = cfg.get('vehicle_light_mode', 'default')
    if mode not in ('default', 'low_beam'):
        raise ValueError('vehicle_light_mode must be default or low_beam')
    return mode


def set_vehicle_lights(client, actors, cfg):
    """Acknowledge the same night-light request for every branch vehicle."""
    mode = vehicle_light_mode(cfg)
    if mode == 'low_beam':
        requested = carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam)
        apply_commands(client, [carla.command.SetVehicleLightState(actor.id, requested) for actor in actors])


def checked_vehicle_lights(actors, roles, cfg):
    """Read after a tick; require position/low beams, permitting dynamic brake bits."""
    mode = vehicle_light_mode(cfg)
    if mode == 'default':
        return {'mode': mode, 'status': 'native_default_not_overridden'}
    if len(actors) != len(roles):
        raise ValueError('Vehicle-light roles do not match actors')
    required = int(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam)
    records = []
    for actor, role in zip(actors, roles):
        actual = int(actor.get_light_state())
        if actual & required != required:
            raise RuntimeError(f'{role}: requested night vehicle lights not active: required={required}, actual={actual}')
        records.append({'role': role, 'actor_id': int(actor.id), 'observed_mask': actual})
    return {'mode': mode, 'status': 'verified', 'required_mask': required,
            'required_flags': ['Position', 'LowBeam'], 'vehicles': records}


def load_render_preset(path):
    """Validate a selected CARLA 0.9.16 preset without contacting the server.

    Evidence hashes establish file identity, not whether requested engine CVars
    actually took effect. Native-image/runtime evidence must establish that.
    Resolved evidence paths keep a frozen copy independent of its new location.
    """
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    preset = json.loads(raw)
    if not isinstance(preset, dict) or preset.get('schema_version') != 'carla-render-preset-v1':
        raise ValueError('Render preset requires schema_version=carla-render-preset-v1')
    if preset.get('example_only'):
        raise ValueError('Example render preset is a schema template, not a selected calibration')
    if preset.get('carla_version') != '0.9.16':
        raise ValueError('Render preset must target CARLA 0.9.16')
    if not isinstance(preset.get('preset_id'), str) or not preset['preset_id'].strip():
        raise ValueError('Render preset requires a nonempty preset_id')
    if preset.get('quality_level') != 'Epic':
        raise ValueError('Render preset must require Epic quality')
    maps = preset.get('maps')
    if not isinstance(maps, list) or not maps or any(not isinstance(m, str) or not m for m in maps):
        raise ValueError('Render preset requires an explicit nonempty maps list')
    weather = preset.get('weather')
    if not isinstance(weather, dict) or set(weather) != set(WEATHER_FIELDS):
        raise ValueError(f'Render preset weather must contain all and only these 14 fields: {WEATHER_FIELDS}')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in weather.values()):
        raise ValueError('Render preset weather values must be finite numbers')
    camera = preset.get('camera_attributes')
    required = {'exposure_mode', 'exposure_compensation', 'shutter_speed', 'iso', 'fstop', 'gamma', 'bloom_intensity'}
    if not isinstance(camera, dict) or not required.issubset(camera):
        raise ValueError(f'Render preset camera_attributes must explicitly contain {sorted(required)}')
    if camera['exposure_mode'] not in ('manual', 'histogram'):
        raise ValueError('Camera exposure_mode must be manual or histogram')
    numeric = required-{'exposure_mode'}
    if camera['exposure_mode'] == 'histogram':
        numeric |= {'exposure_min_bright', 'exposure_max_bright', 'exposure_speed_up', 'exposure_speed_down'}
    for key in numeric:
        value = camera.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Camera {key} must be an explicit finite number')
    if any(camera[k] <= 0 for k in ('shutter_speed', 'iso', 'fstop', 'gamma')):
        raise ValueError('Camera shutter_speed, iso, fstop and gamma must be positive')
    reserved = {'role_name', 'image_size_x', 'image_size_y', 'fov', 'sensor_tick', 'enable_postprocess_effects'}
    if reserved.intersection(camera):
        raise ValueError('Render preset cannot override the frozen camera rig, timing or postprocess contract')
    for key in ('motion_blur_intensity', 'lens_flare_intensity', 'lens_k', 'lens_kcube', 'chromatic_aberration_intensity'):
        if key in camera and camera[key] != 0:
            raise ValueError(f'Render preset must keep {key}=0 for geometric evaluation')
    engine = preset.get('engine')
    if not isinstance(engine, dict) or not isinstance(engine.get('requested'), dict) or not engine['requested']:
        raise ValueError('Render preset requires nonempty engine.requested settings')
    for values, label in ((camera, 'camera_attributes'), (engine['requested'], 'engine.requested')):
        for key, value in values.items():
            if not isinstance(key, str) or not key or not isinstance(value, (str, int, float, bool)):
                raise ValueError(f'{label} settings must have string names and scalar values')
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f'{label}.{key} must be finite')
    # Multi-map batches can reuse measured presets without applying Town05's
    # exposure to Town10HD. Only the calibrated camera settings vary by map;
    # weather and engine settings retain the common, hashed preset identity.
    overrides = preset.get('camera_attributes_by_map', {})
    if not isinstance(overrides, dict) or not set(overrides).issubset(maps):
        raise ValueError('Camera map overrides must name only declared maps')
    for name, attributes in overrides.items():
        if not isinstance(attributes, dict) or set(attributes) != set(camera):
            raise ValueError(f'{name}: camera override must declare the complete same setting keys')
        # This protocol permits only independently calibrated exposure changes.
        if any(attributes[key] != camera[key] for key in camera if key != 'exposure_compensation'):
            raise ValueError(f'{name}: map override may only change exposure_compensation')
        exposure = attributes['exposure_compensation']
        if isinstance(exposure, bool) or not isinstance(exposure, (int, float)) or not math.isfinite(exposure):
            raise ValueError(f'{name}: exposure_compensation must be finite')
    for evidence, label in ((engine.get('evidence'), 'engine.evidence'),
                            (preset.get('calibration_evidence'), 'calibration_evidence')):
        reference = preset.get('validation_status') == 'reference_parameters_requires_local_validation'
        if not isinstance(evidence, list) or (not evidence and not (reference and label == 'calibration_evidence')):
            raise ValueError(f'Render preset requires nonempty {label}')
        for entry in evidence:
            if not isinstance(entry, dict) or not isinstance(entry.get('path'), str) or not entry['path']:
                raise ValueError(f'{label} requires file path and sha256 entries')
            expected = entry.get('sha256')
            if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
                raise ValueError(f'{label} requires a lowercase SHA256 file hash')
            source = (resource_path(entry['path']) if entry['path'].startswith('resource://') else
                      (Path(entry['path']) if Path(entry['path']).is_absolute() else path.parent/entry['path']).resolve(strict=True))
            if not source.is_file():
                raise ValueError(f'Render evidence is not a readable file: {source}')
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Render evidence SHA256 mismatch: {source}')
            entry['path'] = resource_identity(source)
    return preset, {'source_path': str(path), 'source_sha256': hashlib.sha256(raw).hexdigest()}


def check_frozen_render_preset(data, config):
    identity = config.get('render_preset')
    if identity is None:
        if config.get('scene_profile') in ('town10_city_v1', 'lane_change_v1', 'diverse_v1'):
            raise ValueError('City profile requires a frozen render preset')
        return None
    if identity.get('file') != 'render_preset.json':
        raise ValueError('Frozen render preset must be the dataset render_preset.json')
    path = Path(data)/identity['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != identity['sha256']:
        # A published mixed-source release retains its original default and may bind
        # a different sealed preset to an individual scene. Native capture uses the default.
        manifest_file = Path(data) / 'manifest.json'
        manifest = json.loads(manifest_file.read_text()) if manifest_file.is_file() else {}
        scene = config.get('scene_id', '')
        if not isinstance(scene, str):
            raise ValueError('Frozen render preset SHA256 mismatch')
        digits = scene.removeprefix('scene_')
        binding = manifest.get('scene_render_presets', {}).get(scene, {})
        expected_file = f'{scene}/render_preset.json'
        if (manifest.get('status') != 'published_snapshot' or len(digits) < 4 or not digits.isascii() or not digits.isdigit()
                or scene != 'scene_' + digits or binding != {'file': expected_file, 'sha256': identity['sha256']}):
            raise ValueError('Frozen render preset SHA256 mismatch')
        folder = Path(data) / scene
        path = folder / 'render_preset.json'
        if folder.is_symlink() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != identity['sha256']:
            raise ValueError('Frozen per-scene render preset SHA256 mismatch')
    preset, _ = load_render_preset(path)
    weather = config['weather']
    scene_weather = preset.get('scene_weather_policy') == 'frozen-per-scene'
    valid_weather = (set(weather) == set(WEATHER_FIELDS) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in weather.values()))
    if config['map'] not in preset['maps'] or not valid_weather or (not scene_weather and weather != preset['weather']):
        raise ValueError('Frozen scene map/weather differs from selected render preset')
    mode = vehicle_light_mode(config)
    if config['render'].get('vehicle_light_mode', 'default') != mode:
        raise ValueError('Frozen scene vehicle-light policy differs from render settings')
    if config['render']['camera_attributes'] != scene_camera_attributes(preset, config) or config['render']['engine'] != preset['engine']:
        raise ValueError('Frozen scene camera/engine settings differ from selected render preset')
    return preset


def vec(v):return [float(v.x),float(v.y),float(v.z)]


def transform(value):
    p=value['position'];r=value['rotation_deg']
    return carla.Transform(carla.Location(x=p[0],y=p[1],z=p[2]),carla.Rotation(roll=r[0],pitch=r[1],yaw=r[2]))


def transform_dict(t):
    return {'position':vec(t.location),'rotation_deg':[t.rotation.roll,t.rotation.pitch,t.rotation.yaw]}


def configure_world(world):
    settings=world.get_settings();settings.synchronous_mode=True
    settings.fixed_delta_seconds=DT;settings.substepping=True
    settings.max_substep_delta_time=.01;settings.max_substeps=10;settings.no_rendering_mode=False
    world.apply_settings(settings)


def observe(vehicles):
    ts=[a.get_transform() for a in vehicles]
    return {'position':np.array([vec(t.location) for t in ts]),
            'rotation_deg':np.array([[t.rotation.roll,t.rotation.pitch,t.rotation.yaw] for t in ts]),
            'velocity':np.array([vec(a.get_velocity()) for a in vehicles]),
            'angular_velocity':np.array([vec(a.get_angular_velocity()) for a in vehicles]),
            'acceleration':np.array([vec(a.get_acceleration()) for a in vehicles]),
            'bbox_extent':np.array([vec(a.bounding_box.extent) for a in vehicles]),
            'bbox_offset':np.array([vec(a.bounding_box.location) for a in vehicles]),
            'bbox_rotation_deg':np.array([[a.bounding_box.rotation.roll,
                a.bounding_box.rotation.pitch,a.bounding_box.rotation.yaw] for a in vehicles]),
            'transform_world_vehicle':np.array([t.get_matrix() for t in ts])}


def apply_commands(client, commands):
    """Acknowledge every mutation before the sole explicit world.tick().

    Individual actor.apply_control/target_velocity calls are asynchronous RPCs;
    their return does not guarantee the last vehicle's command reached a tick.
    """
    responses = client.apply_batch_sync(commands, False)
    if len(responses) != len(commands):
        raise RuntimeError('Synchronous command batch returned an incomplete response')
    errors = [response.error for response in responses if response.has_error()]
    if errors:
        raise RuntimeError(f'Synchronous command batch failed: {errors}')


def apply(client, vehicles, controls):
    apply_commands(client, [carla.command.ApplyVehicleControl(actor.id,
        carla.VehicleControl(throttle=float(row[0]), steer=float(row[1]), brake=float(row[2]),
                             manual_gear_shift=False, gear=0))
        for actor, row in zip(vehicles, controls)])


def observe_actuation(vehicles):
    values = [actor.get_control() for actor in vehicles]
    n = len(vehicles)
    return {
        'observed_control': np.array([[v.throttle, v.steer, v.brake] for v in values]),
        'observed_gear': np.array([v.gear for v in values]),
        'observed_manual_gear_shift': np.array([v.manual_gear_shift for v in values]),
        'requested_gear': np.zeros(n, dtype=int),
        'requested_manual_gear_shift': np.zeros(n, dtype=bool),
    }


def physics_record(p):
    fields = ('max_rpm', 'moi', 'damping_rate_full_throttle',
              'damping_rate_zero_throttle_clutch_engaged', 'damping_rate_zero_throttle_clutch_disengaged',
              'use_gear_autobox', 'gear_switch_time', 'clutch_strength', 'final_ratio', 'mass',
              'drag_coefficient', 'use_sweep_wheel_collision')
    result = {key: getattr(p, key) for key in fields}
    result['center_of_mass'] = vec(p.center_of_mass)
    result['torque_curve'] = [[v.x, v.y] for v in p.torque_curve]
    result['steering_curve'] = [[v.x, v.y] for v in p.steering_curve]
    result['forward_gears'] = [{'ratio': g.ratio, 'down_ratio': g.down_ratio, 'up_ratio': g.up_ratio}
                               for g in p.forward_gears]
    result['wheels'] = [{**{key: getattr(w, key) for key in (
        'tire_friction', 'damping_rate', 'max_steer_angle', 'radius', 'max_brake_torque',
        'max_handbrake_torque', 'lat_stiff_max_load', 'lat_stiff_value', 'long_stiff_value')},
        'position': vec(w.position)} for w in p.wheels]
    return result


def vehicle_physics_profile(calibration_paths):
    """Require actual stop/restart evidence at both test speeds before freezing.

    This only validates calibration coverage. Full scene trajectories still need
    independent physical review; no simulator parameter guarantees the result.
    """
    if not calibration_paths:
        raise ValueError('lane_change_v1 requires --physics-calibration stop/restart reports')
    required = set(NATIVE_NPC_BLUEPRINTS) | {'vehicle.tesla.model3'}
    evidence, speeds, friction = [], set(), None
    for item in calibration_paths:
        path = resource_path(item)
        record = json.loads(path.read_text())
        params = record.get('parameters', {})
        if record.get('status') != 'passed' or record.get('carla_version') != '0.9.16':
            raise ValueError('Physics calibration must pass with CARLA 0.9.16')
        if params.get('manual_gear_shift') or params.get('gear') != 0 or params.get('brake_cap') != 1.:
            raise ValueError('Physics calibration must cover automatic transmission and full brake')
        value = params.get('tire_friction_override')
        if value is None or not math.isfinite(value) or value <= 0 or params.get('zero_throttle_damping_override') is not None:
            raise ValueError('Require a single positive wheel-friction override')
        if friction is not None and not math.isclose(value, friction):
            raise ValueError('Calibration reports must share one physical profile')
        friction = value
        trials = record.get('trials', [])
        if {row['blueprint'] for row in trials} != required or len(trials) != len(required):
            raise ValueError('Physics calibration must cover all four planned vehicle models')
        for row in trials:
            if row['status'] != 'passed' or not all(row['acceptance'].values()):
                raise ValueError('Every model must pass physical calibration')
            for key in ('peak_speed_deceleration_m_s2', 'peak_speed_acceleration_m_s2'):
                if not math.isfinite(row[key]) or row[key] > 8.:
                    raise ValueError('Calibration exceeds the declared 8 m/s2 threshold')
            wheels = row['applied_physics_readback']['wheels']
            if len(wheels) != 4 or any(not math.isclose(w['tire_friction'], value, abs_tol=1e-6) for w in wheels):
                raise ValueError('Actual physics readback does not match candidate')
        speeds.add(float(params['cruise_target_m_s']))
        evidence.append({'path': resource_identity(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'source_sha256': record['source_sha256']})
    if not {6., 10.}.issubset(speeds):
        raise ValueError('Require natural stop/restart calibration at 6 and 10 m/s')
    return {'mode': 'uniform_wheel_friction', 'wheel_tire_friction': friction,
            'scope': 'All ego/NPC wheels; other native physics unchanged, automatic transmission.',
            'calibration': evidence, 'calibrated_speeds_m_s': sorted(speeds),
            'review_limit_m_s2': 8.,
            'limitations': 'Simulator parameter chosen from vehicle-only dynamics calibration, not a measured real-world friction coefficient. Scene dynamics, steering, contacts and repeat still require review.'}


def apply_physics_profile(client, world, vehicles, roles, profile):
    records = []
    commands = []
    for actor, role in zip(vehicles, roles):
        native = actor.get_physics_control()
        before = physics_record(native)
        wheels = native.wheels
        for wheel in wheels:
            wheel.tire_friction = profile['wheel_tire_friction']
        native.wheels = wheels
        commands.append(carla.command.ApplyVehiclePhysicsControl(actor.id, native))
        records.append({'role': role, 'blueprint_id': actor.type_id, 'native': before})
    # A post-tick readback alone cannot prove every vehicle received the
    # override before that tick. Acknowledge the complete batch first, so the
    # final actor cannot take its first physics step with the old parameters.
    apply_commands(client, commands)
    world.tick(90)
    for actor, entry in zip(vehicles, records):
        entry['applied'] = physics_record(actor.get_physics_control())
        if any(not math.isclose(w['tire_friction'], profile['wheel_tire_friction'], abs_tol=1e-6)
               for w in entry['applied']['wheels']):
            raise RuntimeError('Vehicle physics override failed actual post-tick readback')
    return records


def checked_weather(world, expected):
    weather = world.get_weather()
    actual = {key: float(getattr(weather, key)) for key in (
        'cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity',
        'sun_azimuth_angle', 'sun_altitude_angle', 'fog_density', 'fog_distance',
        'fog_falloff', 'wetness', 'scattering_intensity', 'mie_scattering_scale',
        'rayleigh_scattering_scale', 'dust_storm') if hasattr(weather, key)}
    mismatches = {key: (value, actual.get(key)) for key, value in expected.items()
                  if key not in actual or not math.isclose(actual[key], value, rel_tol=0, abs_tol=1e-4)}
    if mismatches:
        raise RuntimeError(f'Actual weather differs from frozen scene: {mismatches}')
    return actual


def get_image(q, expected):
    while True:
        image=q.get(timeout=90)
        if image.frame==expected:return image
        if image.frame>expected:raise RuntimeError(f'Missing sensor frame {expected}, received {image.frame}')


def receive_frame_barrier(queues, names, expected_frame, expected_time, timeout=30.):
    """Wait for this frame's complete rig before allowing another world tick.

    A world snapshot arrives before GPU sensor processing finishes. One shared
    deadline bounds the whole rig; stale/future frames are errors, not skipped.
    """
    deadline = time.monotonic() + timeout
    images = []
    for name, q in zip(names, queues):
        try:
            image = q.get(timeout=max(0., deadline-time.monotonic()))
        except queue.Empty as exc:
            raise RuntimeError(f'{name}: frame {expected_frame} sensor barrier timed out') from exc
        if image.frame != expected_frame:
            raise RuntimeError(f'{name}: sensor barrier expected {expected_frame}, received {image.frame}')
        if not math.isclose(image.timestamp, expected_time, rel_tol=0, abs_tol=1e-6):
            raise RuntimeError(f'{name}: sensor barrier timestamp differs from physics')
        images.append(image)
    if len(images) != len(names) or len(queues) != len(names):
        raise RuntimeError('sensor barrier camera/queue inventory differs')
    return images


def save_rgb(image,path):
    rgb=np.frombuffer(image.raw_data,dtype=np.uint8).reshape(image.height,image.width,4)[:,:,:3][:,:,::-1].copy()
    Image.fromarray(rgb).save(path,compress_level=6)


def camera_blueprint(library, name, width, height, fov, render):
    bp=library.find('sensor.camera.rgb')
    # Zero means every physics tick. A floating-point .05-second actor timer
    # can skip an intermediate frame even with a nominal .05-second world.
    values={'role_name':name,'image_size_x':width,'image_size_y':height,'fov':fov,'sensor_tick':0.,
            'enable_postprocess_effects':'true','motion_blur_intensity':0.,'lens_flare_intensity':0.,
            'lens_k':0.,'lens_kcube':0.,'chromatic_aberration_intensity':0.,**render}
    recorded={}
    for key,value in values.items():
        if bp.has_attribute(key):
            bp.set_attribute(key,str(value));recorded[key]=str(value)
        else:
            raise RuntimeError(f'Required sensor attribute missing: {key}')
    return bp,recorded


def preserve_static_map_requested(cfg):
    """Diagnostic only: keep map actors, not the previous branch's captures."""
    value = cfg.get('preserve_static_map', False)
    if type(value) is not bool:
        raise ValueError('preserve_static_map must be a bool')
    return value


def check_no_capture_actors(world):
    """Do not reuse a world containing another branch's vehicles or sensors."""
    leftovers = [{'id': int(actor.id), 'type_id': actor.type_id}
                 for actor in world.get_actors()
                 if actor.type_id.startswith(('vehicle.', 'sensor.', 'walker.', 'controller.ai.walker'))]
    if leftovers:
        raise RuntimeError(f'preserve_static_map refuses leftover vehicle/sensor actors: {leftovers}')
    return {'checked': True, 'vehicle_sensor_count': 0}


def finish_capture_cleanup(world, out=None):
    """Publish one post-destroy snapshot; never retry ticks to hide leftovers."""
    report = {'status': 'partial', 'policy': 'exactly_one_tick_after_destroy',
              'world_id_before': int(world.id), 'tick_timeout_s': 90,
              'usage': 'Cleanup only; excluded from RGB history and physical GT.'}
    try:
        report['frame_id'] = int(world.tick(90))
        snapshot = world.get_snapshot()
        report['snapshot_frame_id'] = int(snapshot.frame)
        report['simulation_time_s'] = float(snapshot.timestamp.elapsed_seconds)
        report['world_id_after'] = int(world.id)
        if (report['snapshot_frame_id'] != report['frame_id'] or
                report['world_id_after'] != report['world_id_before'] or
                not math.isfinite(report['simulation_time_s'])):
            raise RuntimeError('World identity, frame or timestamp changed during cleanup snapshot')
        report['actor_check'] = check_no_capture_actors(world)
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        exc.cleanup_report = report
        raise
    finally:
        if out is not None:
            write_json(Path(out)/'cleanup.json', report)
            meta_path = Path(out)/'meta.json'
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                meta['cleanup'] = report
                if report['status'] != 'complete':
                    meta['status_before_cleanup'] = meta.get('status')
                    meta.update(status='failed', stage='capture_cleanup', error=report.get('error'))
                elif meta.get('status') == 'captured_pending_cleanup':
                    meta['status'] = 'complete'
                write_json(meta_path, meta)
    return report


def initialization_world_readback(world, stage):
    settings = world.get_settings()
    snapshot = world.get_snapshot()
    return {'stage': stage, 'world_id': int(world.id), 'map': world.get_map().name,
            'frame': int(snapshot.frame),
            'elapsed_seconds': float(snapshot.timestamp.elapsed_seconds),
            'delta_seconds': float(snapshot.timestamp.delta_seconds),
            'settings': {key: getattr(settings, key) for key in (
                'synchronous_mode', 'fixed_delta_seconds', 'substepping',
                'max_substep_delta_time', 'max_substeps', 'no_rendering_mode')}}


def checked_initialization_readback(world, stage):
    row = initialization_world_readback(world, stage)
    settings = row['settings']
    if not (settings['synchronous_mode'] is True
            and settings['fixed_delta_seconds'] is not None
            and math.isclose(settings['fixed_delta_seconds'], DT, rel_tol=0, abs_tol=1e-9)
            and settings['substepping'] is True
            and math.isclose(settings['max_substep_delta_time'], .01, rel_tol=0, abs_tol=1e-9)
            and settings['max_substeps'] == 10
            and settings['no_rendering_mode'] is False):
        raise RuntimeError(f'Initialization lost synchronous native settings: {row}')
    return row


def initialize_branch_world(client, cfg):
    preserve = preserve_static_map_requested(cfg)
    world = client.get_world()
    incoming_id = int(world.id)
    readbacks = [initialization_world_readback(world, 'incoming')]
    configure_world(world)
    readbacks.append(checked_initialization_readback(world, 'before_map_selection'))
    map_loaded = world.get_map().name.rsplit('/', 1)[-1] != cfg['map']
    if map_loaded:
        world = client.load_world(cfg['map'], reset_settings=False)
    readbacks.append(checked_initialization_readback(world, 'after_map_selection'))
    selected_id = int(world.id)
    if preserve:
        check_no_capture_actors(world)
    configure_world(world)
    if not preserve:
        world = client.reload_world(False)
        configure_world(world)
    readbacks.append(checked_initialization_readback(world, 'after_initial_reset'))
    return world, {'preserve_static_map_requested': preserve,
                   'initialization_readbacks': readbacks,
                   'world_id_before_map_selection': incoming_id,
                   'map_loaded': map_loaded,
                   'world_id_before_initial_reset': selected_id,
                   'world_id_after_initial_reset': int(world.id),
                   'initial_reload_skipped': preserve,
                   'policy': 'reuse_static_map_new_branch_actors' if preserve else 'reload_world_false',
                   'clock_note': 'Preserved worlds keep their absolute simulation clock; this is an initialization diagnostic, not a counterfactual-validity claim.' if preserve else 'Legacy reset protocol.'}


def prime_renderer(client, world, cfg, versions, progress=None):
    """Render and discard a separate episode before native-profile collection.

    A cold renderer can differ from a warmed renderer even after a world reset.
    This pass contains no recorded history, controls or GT. Callbacks queue only
    frame metadata, so full-resolution warmup images are not retained.
    """
    preserve = preserve_static_map_requested(cfg)
    if cfg.get('vehicle_profile') != 'native_v2' or 'render_preset' not in cfg:
        return world, None
    seconds = cfg['timing'].get('renderer_prime_s')
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Native preset collection requires positive timing.renderer_prime_s')
    render_ticks = round(seconds / DT)
    if render_ticks < 1 or not math.isclose(render_ticks * DT, seconds, abs_tol=1e-9):
        raise ValueError('renderer_prime_s must be a whole number of physics ticks')
    if len(cfg['camera_rig']) != 6:
        raise ValueError('Renderer priming requires the complete six-camera rig')
    prime_world_id = int(world.id)
    clear_before = check_no_capture_actors(world) if preserve else None
    weather_ticks = round(cfg['timing']['weather_warmup_s'] / DT)
    world.set_weather(carla.WeatherParameters(**cfg['weather']))
    for _ in range(weather_ticks):
        world.tick(90)
    weather_before = checked_weather(world, cfg['weather'])
    ego = None
    sensors, queues, rig = [], [], []
    try:
        library = world.get_blueprint_library()
        spec = cfg['vehicles'][0]
        bp = library.find(spec['blueprint_id'])
        for key, value in spec.get('attributes', spec.get('blueprint_attributes', {})).items():
            if bp.has_attribute(key):
                bp.set_attribute(key, str(value))
        raised = transform(spec['transform'])
        raised.location.z += 1.
        ego = world.try_spawn_actor(bp, raised)
        if ego is None:
            raise RuntimeError('Renderer prime ego spawn failed')
        apply_commands(client, [carla.command.SetSimulatePhysics(ego.id, False),
                                carla.command.ApplyTransform(ego.id, transform(spec['transform']))])
        set_vehicle_lights(client, [ego], cfg)
        for camera_spec in cfg['camera_rig']:
            bp, requested = camera_blueprint(library, camera_spec['name'], camera_spec['width'],
                camera_spec['height'], camera_spec['fov_deg'], cfg['render']['camera_attributes'])
            sensor = world.spawn_actor(bp, transform(camera_spec['transform']), attach_to=ego,
                                       attachment_type=carla.AttachmentType.Rigid)
            sensors.append(sensor)
            frames = queue.Queue()
            sensor.listen(lambda image, frames=frames: frames.put({
                'frame_id': int(image.frame), 'timestamp': float(image.timestamp),
                'width': int(image.width), 'height': int(image.height),
                'buffer_bytes': len(image.raw_data)}))
            queues.append(frames)
            rig.append({'name': camera_spec['name'], 'attributes_requested': requested,
                        'actor_attributes': dict(sensor.attributes)})
        first_frame = last_frame = None
        received = {}
        for render_index in range(render_ticks):
            last_frame = world.tick(90)
            if first_frame is None:
                first_frame = last_frame
            final_time = float(world.get_snapshot().timestamp.elapsed_seconds)
            # Prime views also need an identical render history. Never enqueue
            # a later world update before this rig's current capture completes.
            for camera_spec, frames in zip(cfg['camera_rig'], queues):
                sample = frames.get(timeout=30)
                if sample['frame_id'] != last_frame:
                    raise RuntimeError('Renderer prime frame barrier mismatch')
                if not math.isclose(sample['timestamp'], final_time, rel_tol=0, abs_tol=1e-6):
                    raise RuntimeError('Renderer prime timestamp barrier mismatch')
                received[camera_spec['name']] = sample
            if progress is not None and ((render_index+1) % 10 == 0 or render_index+1 == render_ticks):
                progress('renderer_prime_frames', completed_ticks=render_index+1, total_ticks=render_ticks)
        final_time = float(world.get_snapshot().timestamp.elapsed_seconds)
        for camera_spec in cfg['camera_rig']:
            sample = received[camera_spec['name']]
            if (sample['width'], sample['height'], sample['buffer_bytes']) != (
                    camera_spec['width'], camera_spec['height'],
                    camera_spec['width'] * camera_spec['height'] * 4):
                raise RuntimeError(f'Renderer prime invalid image buffer: {camera_spec["name"]}')
        report = {'status': 'complete', 'renderer_prime_s': seconds, 'physics_dt_s': DT,
                  'weather_warmup_frames': weather_ticks, 'render_frames': render_ticks,
                  'first_frame': first_frame, 'last_frame': last_frame,
                  'camera_names': [s['name'] for s in cfg['camera_rig']],
                  'final_simulation_time_s': final_time,
                  'sensor_final_frames': {name: sample['frame_id'] for name, sample in received.items()},
                  'sensor_last_samples': received, 'rig': rig,
                  'weather_after_warmup': weather_before,
                  'weather_actual': checked_weather(world, cfg['weather']),
                  'vehicle_lights': checked_vehicle_lights([ego], [spec['role']], cfg),
                  'carla_client_version': versions['client'], 'carla_server_version': versions['server'],
                  'images_saved': 0, 'per_frame_rgb_barrier': True,
                  'usage': 'Discarded renderer priming only; no model input or GT.',
                  'reset_after': 'reload_world(False) before the ordinary branch initialization'}
    finally:
        for sensor in sensors:
            try:
                sensor.stop()
            except RuntimeError:
                pass
        for actor in list(reversed(sensors)) + ([ego] if ego is not None else []):
            try:
                actor.destroy()
            except RuntimeError:
                pass
        if preserve:
            prime_cleanup = finish_capture_cleanup(world)
    if preserve:
        report['capture_actors_before_prime'] = clear_before
        report['capture_actors_after_cleanup'] = prime_cleanup['actor_check']
        report['cleanup'] = prime_cleanup
        report['reset_after'] = 'reload skipped; static map retained after destroying all prime vehicles and cameras'
    else:
        world = client.reload_world(False)
    configure_world(world)
    report['world_id_before_prime'] = prime_world_id
    report['world_id_after_prime'] = int(world.id)
    report['world_reload_skipped'] = preserve
    return world, report


def environment_vehicles(world, origin, radius=120.):
    """Map vehicle geometry before dynamic actors spawn; boxes are WORLD space."""
    result={}
    for name in ('Car','Bus','Truck','Motorcycle','Bicycle','Train'):
        label=getattr(carla.CityObjectLabel,name,None)
        if label is None:continue
        for obj in world.get_environment_objects(label):
            box=obj.bounding_box
            if math.hypot(box.location.x-origin.x,box.location.y-origin.y)>radius:continue
            result[str(obj.id)]={'environment_id':str(obj.id),'name':obj.name,'type':name,
                'transform_world':obj.transform.get_matrix(),'bbox_center_world':vec(box.location),
                'bbox_rotation_deg':[box.rotation.roll,box.rotation.pitch,box.rotation.yaw],
                'bbox_extent':vec(box.extent),
                'world_corners':[vec(p) for p in box.get_world_vertices(carla.Transform())],
                'is_static':True,'velocity_assumption':'fixed map geometry; not measured actor telemetry'}
    return {'objects':[result[k] for k in sorted(result)],'radius_m':radius,
            'query_origin_world':vec(origin),'queried_before_dynamic_spawn':True,
            'identity_note':'Environment geometry IDs; one record is not guaranteed to equal one physical vehicle.'}


def apply_vehicle_profile(config, profile, scene_index):
    """Predeclare NPC appearance by scene/role; every branch shares this config."""
    if profile not in ('legacy_tesla', 'native_v2'):
        raise ValueError(f'Unknown vehicle profile: {profile}')
    vehicles = config['vehicles']
    if not vehicles or vehicles[0]['role'] != 'ego' or vehicles[0]['blueprint_id'] != 'vehicle.tesla.model3':
        raise ValueError('The ego must remain Tesla Model 3 for the fixed camera rig')
    config['vehicle_profile'] = profile
    if profile == 'legacy_tesla':
        if any(v['blueprint_id'] != 'vehicle.tesla.model3' for v in vehicles):
            raise ValueError('legacy_tesla requires the unchanged Tesla vehicle plan')
        return
    for role_index, vehicle in enumerate(vehicles[1:], start=1):
        assignment = int(scene_index)+role_index-1
        vehicle['blueprint_id'] = NATIVE_NPC_BLUEPRINTS[assignment % len(NATIVE_NPC_BLUEPRINTS)]
        attributes = dict(vehicle.get('attributes', vehicle.get('blueprint_attributes', {})))
        attributes.update(role_name=vehicle['role'], color=NATIVE_NPC_COLORS[assignment % len(NATIVE_NPC_COLORS)])
        vehicle['blueprint_attributes'] = attributes
        if 'attributes' in vehicle:
            vehicle['attributes'] = dict(attributes)
    provenance = config.setdefault('provenance', {})
    provenance['vehicle_assignment'] = {
        'profile': profile, 'ego_blueprint_unchanged': 'vehicle.tesla.model3',
        'npc_blueprint_cycle': list(NATIVE_NPC_BLUEPRINTS), 'npc_color_cycle_rgb': list(NATIVE_NPC_COLORS),
        'rule': '(scene_index + vehicle_role_index - 1) modulo each cycle length; ego is role index 0',
        'fixed_across_branches': True, 'outcome_based_selection': False,
        'physical_validation': 'Actual bounding boxes/offsets, route tracking, collisions and same-action repeat must be audited for these models.'}
    footprint = provenance.setdefault('spawn_footprint_check', {})
    if 'applies_to_blueprint' in footprint:
        footprint['original_template_blueprint'] = footprint.pop('applies_to_blueprint')
    footprint.update(scope='Conservative preliminary nominal rectangles only; not measured Gen2 vehicle footprints.',
                     actual_bbox_audit_status='required_after_capture',
                     actual_geometry_fields=['bbox_extent', 'bbox_offset', 'bbox_rotation_deg', 'transform_world_vehicle'])


def plan(args):
    if __package__:
        from .audit_render_consistency import DEFAULT_THRESHOLDS
    else:
        from audit_render_consistency import DEFAULT_THRESHOLDS
    render_policy = {'schema_version': 1, 'policy_id': 'native-rgb-repeat-v1',
                     'tile_size_pixels': 100, 'thresholds': dict(DEFAULT_THRESHOLDS)}
    profile = getattr(args, 'scene_profile', 'legacy_v2')
    builders = {'legacy_v2': build_scenes, 'town10_city_v1': build_city_scenes,
                'lane_change_v1': build_extended_lane_change_scenes}
    if profile == 'diverse_v1':
        if __package__:
            from .scenarios_v2 import build_diverse_scenes
        else:
            from scenarios_v2 import build_diverse_scenes
        builders[profile] = build_diverse_scenes
    if profile not in builders:
        raise ValueError(f'Unknown scene profile: {profile}')
    vehicle_profile = getattr(args, 'vehicle_profile', None) or ('legacy_tesla' if profile == 'legacy_v2' else 'native_v2')
    if vehicle_profile not in ('native_v2', 'legacy_tesla'):
        raise ValueError(f'Unknown vehicle profile: {vehicle_profile}')
    if profile == 'legacy_v2' and vehicle_profile != 'legacy_tesla':
        raise ValueError('native_v2 is declared only for town10_city_v1; legacy_v2 keeps legacy_tesla')
    preset_path = getattr(args, 'render_preset', None)
    if profile in ('town10_city_v1', 'lane_change_v1', 'diverse_v1') and preset_path is None:
        raise ValueError(f'{profile} requires --render-preset with validated image/runtime evidence')
    preset, identity = load_render_preset(preset_path) if preset_path is not None else (None, None)
    scenes=builders[profile](args.opendrive,seed=args.seed,count_per_family=5,history_s=2.,driving_warmup_s=1.)
    if len(scenes) != 20 and not (profile == 'lane_change_v1' and len(scenes) == 10):
        raise RuntimeError(f'Unexpected declared scene count: {len(scenes)}')
    for i, cfg in enumerate(scenes):
        if profile == 'diverse_v1':
            if vehicle_profile != 'native_v2':
                raise ValueError('Diverse scenes require the calibrated vehicle set')
            if (cfg['vehicles'][0]['role'] != 'ego' or cfg['vehicles'][0]['blueprint_id'] != 'vehicle.tesla.model3'
                    or any(v['blueprint_id'] not in (*NATIVE_NPC_BLUEPRINTS, 'vehicle.tesla.model3') for v in cfg['vehicles'][1:])):
                raise ValueError('Diverse scenes require ego first and calibrated vehicle blueprints')
            cfg['vehicle_profile'] = 'native_v2'
            # The builder samples appearance with the layout seed once. Do not
            # overwrite it with the old deterministic per-slot model cycle.
        else:
            apply_vehicle_profile(cfg, vehicle_profile, i)
    if profile in ('lane_change_v1', 'diverse_v1'):
        if vehicle_profile != 'native_v2':
            raise ValueError('Lane-change profile requires calibrated native_v2 NPC vehicles')
        physical_profile = vehicle_physics_profile(getattr(args, 'physics_calibration', None))
        for cfg in scenes:
            cfg['vehicle_physics_profile'] = physical_profile
    if preset is not None:
        if not {c['map'] for c in scenes}.issubset(preset['maps']):
            raise ValueError('Render preset does not cover every selected scene map')
        planned = [args.data/c['scene_id']/'spec.json' for c in scenes] + [args.data/'render_preset.json', args.data/'manifest.json']
        if any(p.exists() for p in planned):
            raise FileExistsError('Use a fresh batch directory for a frozen render preset')
        write_json(args.data/'render_preset.json', preset)
        identity.update(file='render_preset.json', preset_id=preset['preset_id'],
                        sha256=hashlib.sha256((args.data/'render_preset.json').read_bytes()).hexdigest(),
                        engine_note='Requested engine settings and hashed evidence; not an automatic runtime CVar readback.')
    hashes=source_hashes()
    for i,cfg in enumerate(scenes):
        cfg['version']='paired-multicam-v2';cfg['scene_id']=f'scene_{i:04d}'
        cfg['render_consistency_policy'] = render_policy
        cfg['scene_profile']=profile
        cfg['base_seed']=args.seed;cfg.setdefault('seed',args.seed+i)
        cfg['vehicle_roles']=[v['role'] for v in cfg['vehicles']]
        cfg['timing']={'physics_dt_s':DT,'rgb_interval_s':.5,'history_s':2.,'future_s':4.,
                       'weather_warmup_s':1.,
                       'braked_settle_s':1.,'motion_warmup_s':motion_warmup_seconds(cfg),'t0_physics_index':40,'t0_sample_index':4}
        if vehicle_profile == 'native_v2' and preset is not None:
            cfg['timing']['renderer_prime_s'] = 2.
        cfg['control_dispatch']='All vehicle controls and initial velocities acknowledged by apply_batch_sync(do_tick_cue=False) before the sole world.tick.'
        cfg['branches']=list(branch_names(cfg)) if profile == 'diverse_v1' else list(BRANCHES)
        cfg['camera_rig']=[{'name':n,'transform':{'position':[x,y,z],'rotation_deg':[0.,0.,yaw]},
                           'width':1600,'height':900,'fov_deg':fov} for n,x,y,z,yaw,fov in RIG]
        cfg['render']={'quality_level':'Epic','postprocess':True,'rgb_format':'lossless PNG',
                       'sensor_render_hz':20,'saved_rgb_hz':2,
                       'sampling_note':'Render all physics frames to update temporal anti-aliasing; save only every tenth synchronized frame.',
                       'camera_attributes':{'exposure_mode':'manual','shutter_speed':250.,'iso':100.,'fstop':8.,
                                            'exposure_compensation':EXPOSURE_BY_MAP[cfg['map']],
                                            'gamma':2.2,'bloom_intensity':.675},
                       'lighting_protocol':'Fixed map-specific manual exposure selected from real calibration images; identical weather and camera settings for all scene branches; native renderer RGB, no offline color adjustment.',
                       'exceptions':'Motion blur, lens flare, chromatic aberration and distortion disabled for geometric evaluation; lighting/AO/AA/tonemapping retained.'}
        cfg.setdefault('weather',{'cloudiness':15.,'sun_altitude_angle':55.,'sun_azimuth_angle':25.,
                                  'precipitation':0.,'wetness':0.,'fog_density':0.,'wind_intensity':0.})
        if preset is not None:
            cfg['weather'] = dict(preset['weather'])
            cfg['render']['camera_attributes'] = preset_camera_attributes(preset, cfg['map'])
            cfg['render']['engine'] = preset['engine']
            cfg['render']['lighting_protocol'] = 'Complete weather and camera attributes from a frozen, evidence-linked render preset; identical across every branch, no offline color adjustment.'
            cfg['render_preset'] = dict(identity)
        cfg['semantic_lidar'] = semantic_lidar.default_spec()
        cfg['code_sha256']=hashes
        cfg['input_contract']={'visual_input':'six raw ego-mounted RGB cameras','common_history':'maintain samples indices 0..4, shared by every branch',
                               'labels':'common-t0 structured BEV and reactive labels are evaluation targets, never visual model inputs',
                               'model_data_branches':list(branch_names(cfg, include_repeat=False)),
                               'qa_only_branches':['maintain_repeat'],
                               'repeat_usage':'Retain for rendering/physics QA; exclude from training and model-scoring samples. Branch roles do not imply RGB acceptance.'}
        cfg['spec_sha256']=digest(cfg)
        target=args.data/cfg['scene_id']/'spec.json'
        if target.exists():raise FileExistsError(f'Use a fresh batch directory: {target}')
        write_json(target,cfg)
    maps=args.data/'maps';maps.mkdir(exist_ok=True)
    for name in sorted(set(c['map'] for c in scenes)):
        text=(args.opendrive/(name+'.xodr')).read_text();(maps/(name+'.xodr')).write_text(text)
    manifest = {'version':'paired-multicam-v2','status':'planned','scene_profile':profile,'vehicle_profile':vehicle_profile,
               'scene_ids':[c['scene_id'] for c in scenes],
               'branches':list(dict.fromkeys(b for c in scenes for b in c['branches'])),'code_sha256':hashes,
               'rgb_hz':2,'physics_hz':20,'history_s':2,'future_s':4,'rgb_size':[1600,900],
               'model_input':'six-camera RGB; common observed history','evaluation_only':'structured common-t0 BEV and reactive labels'}
    manifest['render_consistency_policy'] = render_policy
    manifest['qa_only_branches'] = ['maintain_repeat']
    manifest['scene_model_branches'] = {
        c['scene_id']: list(branch_names(c, include_repeat=False)) for c in scenes}
    if identity is not None:
        manifest['render_preset'] = identity
    if profile == 'diverse_v1':
        manifest['scene_branches'] = {c['scene_id']: c['branches'] for c in scenes}
        manifest['planned_branch_count'] = sum(len(c['branches']) for c in scenes)
        manifest['branch_union_note'] = 'branches is a vocabulary union, not a Cartesian product; use scene_branches for collection and evaluation.'
        manifest['layout_protocol'] = 'Sample layout and action feasibility once per scene; reuse exact config, common history and NPC policy across all branches.'
    if profile == 'lane_change_v1':
        manifest['requested_scene_count'] = 20
        manifest['planned_scene_count'] = len(scenes)
        manifest['missing_families'] = sorted(set(('rear_following', 'adjacent_parallel',
            'unprotected_intersection', 'merging')) - {c['family'] for c in scenes})
        manifest['action_semantics'] = 'Map-derived adjacent-lane left/right changes; actual completion requires trajectory and bounding-box audit.'
    write_json(args.data/'manifest.json', manifest)
    print('Planned',len(scenes),'scenes:',{f:sum(c['family']==f for c in scenes) for f in set(c['family'] for c in scenes)},flush=True)


def finalize_scene(cfg, *, data, render_preset=None, physics_calibration=None):
    """Freeze one map-resolved scene into the portable collector contract.

    The caller supplies geometry, vehicles, action profiles and scene identity.
    This does not connect to CARLA or claim physical/render acceptance.
    """
    import copy
    if __package__:
        from .audit_render_consistency import DEFAULT_THRESHOLDS
    else:
        from audit_render_consistency import DEFAULT_THRESHOLDS
    cfg = copy.deepcopy(cfg)
    data = Path(data)
    scene_id = cfg.get('scene_id')
    if not isinstance(scene_id, str) or not scene_id or Path(scene_id).name != scene_id:
        raise ValueError('A safe scene_id is required')
    target = data/scene_id/'spec.json'
    if target.exists():
        raise FileExistsError(f'Frozen scene already exists: {target}')
    preset, identity = load_render_preset(render_preset or RESOURCES/'daylight.json')
    if cfg['map'] not in preset['maps']:
        raise ValueError('Render preset must explicitly cover the selected map')
    preset_file = data/'render_preset.json'
    if preset_file.exists():
        if json.loads(preset_file.read_text()) != preset:
            raise ValueError('Cannot change the render preset within a frozen dataset')
    else:
        write_json(preset_file, preset)
    identity.update(file='render_preset.json', preset_id=preset['preset_id'],
                    sha256=hashlib.sha256(preset_file.read_bytes()).hexdigest(),
                    source_path=resource_identity(render_preset or RESOURCES/'daylight.json'),
                    validation_status=preset.get('validation_status', 'evidence-linked-parameters'))
    cfg.update(version='paired-multicam-v2', scene_profile='grid_v1', vehicle_profile='native_v2')
    cfg['vehicle_roles'] = [vehicle['role'] for vehicle in cfg['vehicles']]
    if not cfg['vehicle_roles'] or cfg['vehicle_roles'][0] != 'ego' or len(set(cfg['vehicle_roles'])) != len(cfg['vehicle_roles']):
        raise ValueError('Require one ego first and distinct stable vehicle roles')
    calibrated = set(NATIVE_NPC_BLUEPRINTS) | {'vehicle.tesla.model3'}
    if any(vehicle['blueprint_id'] not in calibrated for vehicle in cfg['vehicles']):
        raise ValueError('Vehicle model lacks bundled physics-profile evidence')
    cfg['vehicle_physics_profile'] = vehicle_physics_profile(physics_calibration or
        [RESOURCES/'physics_6mps.json', RESOURCES/'physics_10mps.json'])
    cfg['timing'] = {'physics_dt_s': DT, 'rgb_interval_s': .5, 'history_s': 2., 'future_s': 4.,
                     'weather_warmup_s': 1., 'braked_settle_s': 1., 'motion_warmup_s': motion_warmup_seconds(cfg),
                     'renderer_prime_s': 2., 't0_physics_index': 40, 't0_sample_index': 4}
    cfg['branches'] = list(branch_names(cfg))
    cfg['camera_rig'] = [{'name': n, 'transform': {'position': [x,y,z], 'rotation_deg': [0.,0.,yaw]},
                         'width': 1600, 'height': 900, 'fov_deg': fov} for n,x,y,z,yaw,fov in RIG]
    cfg['weather'] = {**preset['weather'], **cfg.get('weather_override', {})}
    cfg['vehicle_light_mode'] = vehicle_light_mode(cfg)
    cfg['render'] = {'quality_level': 'Epic', 'postprocess': True, 'rgb_format': 'lossless PNG',
                     'sensor_render_hz': 20, 'saved_rgb_hz': 2,
                     'camera_attributes': scene_camera_attributes(preset, cfg),
                     'vehicle_light_mode': cfg['vehicle_light_mode'],
                     'engine': preset['engine'],
                     'lighting_protocol': 'Native renderer RGB; scene weather, camera attributes and engine settings frozen across branches.',
                     'sampling_note': 'Render every physics tick; save every tenth synchronized frame.'}
    cfg['render_preset'] = identity
    cfg['render_consistency_policy'] = {'schema_version': 1, 'policy_id': 'native-rgb-repeat-v1',
                                       'tile_size_pixels': 100, 'thresholds': dict(DEFAULT_THRESHOLDS)}
    cfg['control_dispatch'] = 'ApplyVehicleControl commands acknowledged before the sole world.tick; control[t0] belongs to the outgoing future interval.'
    cfg['input_contract'] = {'visual_input': 'six raw ego-mounted RGB cameras',
                            'common_history': 'observations and states at -2..0 seconds; past controls strictly before t0',
                            'model_data_branches': list(branch_names(cfg, include_repeat=False)),
                            'qa_only_branches': ['maintain_repeat'],
                            'derived_labels': 'OCC, TTC and reactive masks are deferred; retain all raw states and transforms.'}
    cfg['semantic_lidar'] = semantic_lidar.default_spec()
    cfg['code_sha256'] = source_hashes()
    cfg.pop('spec_sha256', None)
    # Frozen identity must match the JSON representation actually read by every
    # branch; map builders can return tuple-valued lane IDs and anchors.
    cfg = json.loads(json.dumps(cfg, allow_nan=False))
    cfg['spec_sha256'] = digest(cfg)
    check_frozen_render_preset(data, cfg)
    write_json(target, cfg)
    return cfg


def finalize_manifest(configs, data, extra=None):
    """Write identity and explicit per-scene branch sets for finalized configs."""
    data = Path(data)
    if (data/'manifest.json').exists():
        raise FileExistsError('Frozen manifest already exists')
    if not configs or len({cfg['scene_id'] for cfg in configs}) != len(configs):
        raise ValueError('Require at least one scene with unique identities')
    for cfg in configs:
        if cfg['code_sha256'] != source_hashes():
            raise ValueError('Finalized scene source hash is stale')
        if json.loads((data/cfg['scene_id']/'spec.json').read_text()) != cfg:
            raise ValueError('Scene must be finalized on disk first')
    manifest = {'version': 'paired-multicam-v2', 'status': 'planned', 'scene_profile': 'grid_v1',
                'vehicle_profile': 'native_v2', 'scene_ids': [cfg['scene_id'] for cfg in configs],
                'branches': list(dict.fromkeys(branch for cfg in configs for branch in cfg['branches'])),
                'scene_branches': {cfg['scene_id']: cfg['branches'] for cfg in configs},
                'scene_model_branches': {cfg['scene_id']: list(branch_names(cfg, include_repeat=False)) for cfg in configs},
                'qa_only_branches': ['maintain_repeat'], 'planned_branch_count': sum(len(cfg['branches']) for cfg in configs),
                'code_sha256': source_hashes(), 'rgb_hz': 2, 'physics_hz': 20, 'history_s': 2, 'future_s': 4,
                'semantic_lidar': configs[0]['semantic_lidar'],
                'rgb_size': [1600,900], 'model_input': 'six-camera RGB and common observed history',
                'render_consistency_policy': configs[0]['render_consistency_policy'],
                'render_preset': configs[0]['render_preset'],
                'branch_union_note': 'Use scene_branches; branches is a vocabulary union, never a Cartesian product.'}
    if extra:
        if any(key in manifest and manifest[key] != value for key,value in extra.items()):
            raise ValueError('Extra manifest metadata cannot override the collection contract')
        manifest.update(extra)
    write_json(data/'manifest.json', manifest)
    return manifest


def collect(args,cfg,branch,out):
    if cfg['code_sha256']!=source_hashes():raise RuntimeError('Collection source differs from frozen plan')
    if branch not in branch_names(cfg):
        raise ValueError(f'Branch {branch} is absent from this frozen scene plan')
    expected=cfg['spec_sha256'];body={k:v for k,v in cfg.items() if k!='spec_sha256'}
    if digest(body)!=expected:raise RuntimeError('Invalid config content hash')
    semantic_lidar.validate_spec(cfg.get('semantic_lidar'))
    preset = check_frozen_render_preset(args.data, cfg)
    preserve = preserve_static_map_requested(cfg)  # Reject invalid types before any CARLA call.
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'meta.json',{'status':'partial','branch':branch,'spec_sha256':expected})
    progress_started = time.monotonic()
    def progress(phase, **details):
        log_capture_progress(cfg['scene_id'], branch, phase, progress_started, **details)
    progress('connect', map=cfg['map'])
    client=carla.Client(args.host,args.port);client.set_timeout(120)
    versions = {'client': client.get_client_version(), 'server': client.get_server_version()}
    if any(version != '0.9.16' for version in versions.values()):
        raise RuntimeError(f'This project requires CARLA 0.9.16 for client and server: {versions}')
    if preset is not None:
        for version in (client.get_client_version(), client.get_server_version()):
            if version.split('-')[0].split('+')[0] != preset['carla_version']:
                raise RuntimeError(f'Render preset requires CARLA {preset["carla_version"]}, found {version}')
    world_initialization = None
    try:
        progress('world_initialization')
        world, world_initialization = initialize_branch_world(client, cfg)
        progress('renderer_prime')
        world, renderer_warmup = prime_renderer(client, world, cfg, versions, progress=progress)
        progress('renderer_prime_complete')
        world_initialization['world_id_after_prime'] = int(world.id)
        world_initialization['prime_reload_skipped'] = (renderer_warmup['world_reload_skipped']
                                                        if renderer_warmup is not None else None)
    except BaseException as exc:
        write_json(out/'meta.json', {'status': 'failed', 'stage': 'renderer_warmup',
            'branch': branch, 'spec_sha256': expected, 'error': repr(exc),
            'world_initialization': world_initialization,
            'cleanup': getattr(exc, 'cleanup_report', None),
            'traceback': traceback.format_exc()})
        raise
    progress('weather_warmup')
    world.set_weather(carla.WeatherParameters(**cfg['weather']))
    # set_weather is asynchronous. Advance a fixed, pre-actor warmup before
    # validating readback and spawning the cameras; do not label a stale default
    # weather snapshot as the actual conditions of a future capture.
    for _ in range(round(cfg['timing']['weather_warmup_s']/DT)):
        world.tick(90)
    weather_after_warmup = checked_weather(world, cfg['weather'])
    write_json(out/'environment_vehicles.json',environment_vehicles(world,transform(cfg['vehicles'][0]['transform']).location))
    auxiliary = None
    actors=[];sensors=[];collisions=[];camera_actors=[];queues=[];rig_meta=[];future_writes=[]
    collision_lock = threading.Lock()
    rendered_frames = [[] for _ in range(7)]
    rendered_timestamps = [[] for _ in range(7)]
    barrier_records = []
    def receiver(q, camera_index):
        def on_image(image):
            rendered_frames[camera_index].append(int(image.frame))
            rendered_timestamps[camera_index].append(float(image.timestamp))
            q.put(image)
        return on_image
    def capture_tick():
        frame = world.tick(90)
        snapshot = world.get_snapshot()
        if snapshot.frame != frame:
            raise RuntimeError('World changed before sensor barrier; another tick client is active')
        started = time.monotonic()
        images = receive_frame_barrier(queues, [r['name'] for r in rig_meta] + [cfg['semantic_lidar']['name']],
                                       frame, snapshot.timestamp.elapsed_seconds)
        if world.get_snapshot().frame != frame:
            raise RuntimeError('World advanced during sensor barrier')
        barrier_records.append({'frame_id': frame, 'timestamp_s': snapshot.timestamp.elapsed_seconds,
                                'wait_s': time.monotonic()-started, 'camera_count': 6, 'sensor_count': len(images), 'lidar_count': 1})
        return frame, images
    t0_frame=[None];start=time.monotonic()
    writer=ThreadPoolExecutor(max_workers=4)
    try:
        progress('spawn_actors_and_sensor_rig')
        lib=world.get_blueprint_library()
        for spec in cfg['vehicles']:
            bp=lib.find(spec['blueprint_id'])
            for k,v in spec.get('attributes',spec.get('blueprint_attributes',{})).items():
                if bp.has_attribute(k):bp.set_attribute(k,str(v))
            actor=world.try_spawn_actor(bp,transform(spec['transform']))
            if actor is None:raise RuntimeError(f'Spawn failed: {spec["role"]}')
            actors.append(actor)
        apply_commands(client, [carla.command.SetAutopilot(actor.id, False) for actor in actors])
        set_vehicle_lights(client, actors, cfg)
        roles=cfg['vehicle_roles'];actor_roles=dict(zip([a.id for a in actors],roles))
        auxiliary = AuxiliaryActors(world, client, cfg, DT, carla_module=carla)
        actor_roles.update(auxiliary.actor_roles)
        physical_profile = cfg.get('vehicle_physics_profile')
        physical_records = apply_physics_profile(client, world, actors, roles, physical_profile) if physical_profile else None
        final_warmup_ticks = cfg['timing'].get('final_renderer_warmup_ticks', 0)
        if isinstance(final_warmup_ticks, bool) or not isinstance(final_warmup_ticks, int) or final_warmup_ticks < 0:
            raise ValueError('final_renderer_warmup_ticks must be a nonnegative integer')
        if final_warmup_ticks:
            apply_commands(client, [carla.command.SetSimulatePhysics(a.id, False) for a in actors])
        collision_actors = actors + auxiliary.actors
        collision_roles = roles + [spec['role'] for spec in cfg.get('pedestrians', [])]
        for actor,role in zip(collision_actors,collision_roles):
            sensor=world.spawn_actor(lib.find('sensor.other.collision'),carla.Transform(),attach_to=actor)
            def on_collision(event,role=role):
                with collision_lock:
                    collisions.append({'frame_id':int(event.frame),'role':role,'other_actor_id':event.other_actor.id,
                                       'other_role':actor_roles.get(event.other_actor.id,event.other_actor.type_id),'normal_impulse':vec(event.normal_impulse)})
            sensor.listen(on_collision);sensors.append(sensor)
        for spec in cfg['camera_rig']:
            bp,attrs=camera_blueprint(lib,spec['name'],spec['width'],spec['height'],spec['fov_deg'],cfg['render']['camera_attributes'])
            local=transform(spec['transform']);camera=world.spawn_actor(bp,local,attach_to=actors[0],attachment_type=carla.AttachmentType.Rigid)
            q=queue.Queue();camera.listen(receiver(q,len(queues)));queues.append(q);camera_actors.append(camera);sensors.append(camera)
            f=spec['width']/(2*math.tan(math.radians(spec['fov_deg'])/2))
            rig_meta.append({**spec,'attributes':attrs,'actor_attributes':dict(camera.attributes),'camera_to_ego':local.get_matrix(),
                             'intrinsic':[[f,0,spec['width']/2],[0,f,spec['height']/2],[0,0,1]]})
            (out/'camera'/spec['name']).mkdir(parents=True)
        lidar_spec = cfg['semantic_lidar']
        lidar_bp, lidar_attrs = semantic_lidar.blueprint(lib, lidar_spec)
        lidar_local = transform(lidar_spec['transform'])
        lidar_actor = world.spawn_actor(lidar_bp, lidar_local, attach_to=actors[0],
                                        attachment_type=carla.AttachmentType.Rigid)
        sensors.append(lidar_actor)
        q = queue.Queue(); lidar_actor.listen(receiver(q, len(queues))); queues.append(q)
        lidar_meta = semantic_lidar.metadata(lidar_spec, lidar_attrs, dict(lidar_actor.attributes), actor_roles)
        # Spectator placement retains the prior renderer streaming view without
        # spawning or saving an overhead RGB sensor.
        ego_spawn = transform(cfg['vehicles'][0]['transform']); forward = ego_spawn.get_forward_vector()
        world.get_spectator().set_transform(carla.Transform(
            carla.Location(x=ego_spawn.location.x+forward.x*35, y=ego_spawn.location.y+forward.y*35,
                           z=ego_spawn.location.z+85), carla.Rotation(pitch=-90, yaw=ego_spawn.rotation.yaw)))
        final_warmup = None
        if final_warmup_ticks:
            progress('final_world_warmup', total_ticks=final_warmup_ticks)
            # Keep the actual vehicles, rig, spectator and final world alive.
            # Weather is refreshed again by each sensor's BeginPlay in CARLA;
            # therefore readiness must be established after the full rig exists.
            world.set_weather(carla.WeatherParameters(**cfg['weather']))
            warmup_first = None
            for warmup_tick in range(final_warmup_ticks):
                warmup_last, discarded = capture_tick()
                if warmup_first is None:
                    warmup_first = warmup_last
                if (warmup_tick+1) % 20 == 0 or warmup_tick+1 == final_warmup_ticks:
                    progress('final_world_warmup_frames', completed_ticks=warmup_tick+1,
                             total_ticks=final_warmup_ticks)
            expected_warmup = list(range(warmup_first, warmup_last + 1))
            for name, ledger in zip([r['name'] for r in rig_meta] + [cfg['semantic_lidar']['name']], rendered_frames):
                actual = [f for f in ledger if warmup_first <= f <= warmup_last]
                if actual != expected_warmup:
                    raise RuntimeError(f'{name}: missing/repeated frame during final-world warmup')
            final_warmup = {'status': 'complete', 'physics_disabled': True,
                'frames': final_warmup_ticks, 'first_frame': warmup_first, 'last_frame': warmup_last,
                'weather': checked_weather(world, cfg['weather']),
                'spectator': transform_dict(world.get_spectator().get_transform()),
                'reset_after': False, 'images_saved': 0}
            # Drop only discarded warmup frames; none belongs to model history.
            apply_commands(client, [carla.command.ApplyTransform(a.id, transform(s['transform']))
                                    for a, s in zip(actors, cfg['vehicles'])])
            apply_commands(client, [carla.command.SetSimulatePhysics(a.id, True) for a in actors])
        # Renderer warmup is discarded. Reset walkers once before the common
        # 21-tick settle; never teleport them in saved history or future.
        progress('braked_settle', total_ticks=21)
        auxiliary.hold_pedestrians(reset=True)
        warmup_seconds = motion_warmup_seconds(cfg)
        auxiliary.set_signal_time(-cfg['timing']['history_s']-warmup_seconds)
        apply(client, actors, np.tile([0., 0., 1.], (len(actors), 1)))
        frame, images = capture_tick()
        for _ in range(20):
            frame, images = capture_tick()
        velocity_commands = []
        for actor,spec in zip(actors,cfg['vehicles']):
            forward=actor.get_transform().get_forward_vector();v=spec['initial_speed_mps']
            velocity_commands.extend([
                carla.command.ApplyTargetVelocity(actor.id, carla.Vector3D(forward.x*v,forward.y*v,forward.z*v)),
                carla.command.ApplyTargetAngularVelocity(actor.id, carla.Vector3D())])
        apply_commands(client, velocity_commands)
        extents=np.array([vec(a.bounding_box.extent) for a in actors])
        offsets=np.array([vec(a.bounding_box.location) for a in actors])
        boxrot=np.array([[a.bounding_box.rotation.roll,a.bounding_box.rotation.pitch,a.bounding_box.rotation.yaw] for a in actors])
        controller=ScenarioController(cfg,dt=DT)
        progress('motion_warmup', total_ticks=round(warmup_seconds/DT))
        for warmup_step in range(round(warmup_seconds/DT)):
            warmup_time = -cfg['timing']['history_s']-warmup_seconds+warmup_step*DT
            observed = auxiliary.inject_observation(observe(actors), warmup_time)
            c,_,_=controller.controls(observed,'maintain',warmup_time,extents)
            auxiliary.advance_commands(auxiliary.warmup_commands(warmup_time), warmup_time+DT)
            apply(client,actors,c);frame,images=capture_tick()
        weather_at_history_start = checked_weather(world, cfg['weather'])
        lights_at_history_start = checked_vehicle_lights(actors, roles, cfg)
        t0_frame[0]=frame+40
        fields=('position','rotation_deg','velocity','angular_velocity','acceleration','transform_world_vehicle','control','target_speed')
        if physical_profile:
            fields += ('observed_control','observed_gear','observed_manual_gear_shift',
                       'requested_gear','requested_manual_gear_shift')
        arrays={k:[] for k in fields};frames=[];timestamps=[];sample_frames=[];sample_ids=[];world2cam=[];ego_poses=[]
        lidar_world=[];lidar_timestamps=[];lidar_frames=[]
        # The history uses one declared speed policy, not a newly measured target
        # on every replay. Only at the intervention boundary is cruise anchored
        # to the actually observed t0 speed.
        base_speed=float(cfg['vehicles'][0]['initial_speed_mps']);diagnostics=[]
        for step in range(121):
            t=-2+step*DT;obs=auxiliary.inject_observation(observe(actors), t)
            if physical_profile:
                obs.update(observe_actuation(actors))
            if step==40:base_speed=float(np.linalg.norm(obs['velocity'][0,:2]))
            actual_branch='maintain' if step<40 else ('maintain' if branch=='maintain_repeat' else branch)
            c,targets,diagnostic=controller.controls(obs,actual_branch,t,extents,ego_base_speed=base_speed)
            if step==120:c[:]=0
            walker_commands = auxiliary.record(t, final=step == 120)
            for k in fields:arrays[k].append(c.copy() if k=='control' else np.asarray(targets).copy() if k=='target_speed' else obs[k])
            frames.append(frame);timestamps.append(world.get_snapshot().timestamp.elapsed_seconds)
            diagnostics.append(diagnostic)
            if step%10==0:
                sample_index=step//10;sample_frames.append(frame);sample_ids.append([int(im.frame) for im in images])
                world2cam.append([camera.get_transform().get_inverse_matrix() for camera in camera_actors[:6]])
                ego_poses.append(obs['transform_world_vehicle'][0])
                for i,spec in enumerate(cfg['camera_rig']):
                    future_writes.append(writer.submit(save_rgb,images[i],out/'camera'/spec['name']/f'{sample_index:05d}.png'))
                lidar_world.append(images[-1].transform.get_matrix())
                lidar_timestamps.append(float(images[-1].timestamp)); lidar_frames.append(int(images[-1].frame))
                future_writes.append(writer.submit(semantic_lidar.save_packet, images[-1],
                    out/'lidar'/lidar_spec['name']/f'{sample_index:05d}.npz', spec=lidar_spec,
                    expected_frame=frame, expected_timestamp=timestamps[-1], relative_time=t))
                progress('sample_queued', sample_index=sample_index, sample_count=13,
                         physics_step=step, physics_frame_id=int(frame), relative_time_s=round(t, 3))
            if step<120:
                auxiliary.advance_commands(walker_commands, t+DT)
                apply(client,actors,c);frame,images=capture_tick()
        weather_actual = checked_weather(world, cfg['weather'])
        lights_at_end = checked_vehicle_lights(actors, roles, cfg)
        progress('drain_writes', queued_write_count=len(future_writes))
        for future in future_writes:future.result()
        progress('writes_complete', rgb_images=78, lidar_packets=13)
        render_ledger = []
        for index, name in enumerate([r['name'] for r in rig_meta] + [cfg['semantic_lidar']['name']]):
            selected = [(f, t) for f, t in zip(rendered_frames[index], rendered_timestamps[index])
                        if frames[0] <= f <= frames[-1]]
            if [f for f, _ in selected] != frames:
                raise RuntimeError(f'{name}: render history has missing or repeated physics frames')
            if not np.allclose([t for _, t in selected], timestamps, rtol=0, atol=1e-6):
                raise RuntimeError(f'{name}: render-history timestamps differ from physics')
            render_ledger.append({'camera': name, 'frames': [f for f, _ in selected],
                                  'timestamps_s': [t for _, t in selected]})
        values={k:np.asarray(v,dtype=np.float64) for k,v in arrays.items()}
        values.update(time_s=np.arange(121)*DT-2,frame_id=np.asarray(frames,dtype=np.int64),simulation_time_s=np.asarray(timestamps),
                      bbox_extent=extents,bbox_offset=offsets,bbox_rotation_deg=boxrot)
        np.savez_compressed(out/'states.npz',**values)
        auxiliary.save(out)
        np.savez_compressed(out/'samples.npz',time_s=np.arange(13)*.5-2,physics_indices=np.arange(0,121,10),
            frame_id=np.asarray(sample_frames,dtype=np.int64),sensor_frame_id=np.asarray(sample_ids,dtype=np.int64),
            world_to_camera=np.asarray(world2cam),camera_to_ego=np.array([r['camera_to_ego'] for r in rig_meta]),
            intrinsics=np.array([r['intrinsic'] for r in rig_meta]),ego_to_world=np.array(ego_poses),
            lidar_to_world=np.asarray(lidar_world), lidar_to_ego=np.asarray(lidar_local.get_matrix()),
            lidar_frame_id=np.asarray(lidar_frames,dtype=np.int64), lidar_timestamp_s=np.asarray(lidar_timestamps))
        # Match the physics path: drain asynchronous collision callbacks without
        # advancing scene time, then freeze the recorded event list.
        time.sleep(COLLISION_DRAIN_WALL_S)
        with collision_lock:
            recorded_collisions = [{**event, 'time_s': (event['frame_id']-t0_frame[0])*DT}
                                   for event in collisions]
        write_json(out/'controller_diagnostics.json',diagnostics)
        if controller.merge_memory:
            write_json(out/'npc_merge_state.json',controller.merge_evidence())
        meta={'status':'complete','version':'paired-multicam-v2','branch':branch,'spec_sha256':expected,
              'created_utc':datetime.now(timezone.utc).isoformat(),'elapsed_wall_s':time.monotonic()-start,
              'carla_client_version':client.get_client_version(),'carla_server_version':client.get_server_version(),
              'weather_actual':weather_actual,
              'weather_after_warmup':weather_after_warmup,'weather_at_history_start':weather_at_history_start,
              'weather_actual_readback':'After final saved simulation frame; also checked after weather warmup and at history start.',
              'vehicle_lights': {'at_history_start': lights_at_history_start, 'at_final_frame': lights_at_end},
              'control_dispatch':cfg['control_dispatch'],
              'vehicle_roles':roles,'actor_ids':[a.id for a in actors],'camera_names':[r['name'] for r in rig_meta],
              'rig':rig_meta,'semantic_lidar':lidar_meta,
              'sensor_names':[r['name'] for r in rig_meta]+[lidar_spec['name']],
              'motion_warmup_s':warmup_seconds, 'physics_frames':121,'rgb_samples':13,'collisions':recorded_collisions,'t0_frame':t0_frame[0],
              'collision_callback_drain_wall_s':COLLISION_DRAIN_WALL_S,
              'ego_speed_t0_m_s':base_speed,'code_sha256':source_hashes(),'command':sys.argv,
              'control_timing':'control[t] is applied over state[t] -> state[t+1]; last row is unexecuted zero sentinel',
              'coordinates':'CARLA world xyz metres; roll,pitch,yaw degrees; camera pinhole xyz=[sensor_y,-sensor_z,sensor_x]',
              'npc_branch_information':'none; policies see actual positions, velocities, fixed routes and right-of-way only',
              'traffic_lights':'frozen off; intersections are explicitly unsignalized in this controlled experiment'}
        meta.update(auxiliary.metadata())
        if auxiliary.actors:
            meta['collision_sensor_roles'] = collision_roles
        meta['render_history'] = {'sensor_tick_s': 0., 'physics_frames_checked': len(frames),
                                  'status': 'complete', 'cameras': render_ledger}
        meta['rgb_frame_barrier'] = {'protocol': 'wait_for_six_rgb_and_semantic_lidar_before_next_world_tick',
                                     'timeout_s': 30., 'frames': barrier_records}
        meta['final_world_render_warmup'] = final_warmup
        if 'vehicle_profile' in cfg:
            meta['vehicle_profile'] = cfg['vehicle_profile']
            meta['vehicles'] = [{'role': role, 'actor_id': actor.id, 'blueprint_id': actor.type_id,
                                 'actor_attributes': dict(actor.attributes)} for role, actor in zip(roles, actors)]
        if preset is not None:
            meta['render_preset'] = cfg['render_preset']
            meta['engine_requested'] = cfg['render']['engine']['requested']
        if physical_profile:
            meta['vehicle_physics_profile'] = physical_profile
            meta['vehicle_physics'] = physical_records
            meta['control_time_convention'] = 'control and requested_* belong to the next interval (last frame is not executed); observed_* are simulator-returned prior applied controls.'
        meta['world_initialization'] = world_initialization
        if renderer_warmup is not None:
            meta['renderer_warmup'] = renderer_warmup
        if preserve:
            meta['status'] = 'captured_pending_cleanup'
        write_json(out/'meta.json',meta)
        if not preserve:
            print('COMPLETE',cfg['scene_id'],branch,13,'six-camera samples',round(meta['elapsed_wall_s'],1),'s',flush=True)
    except BaseException as exc:
        progress('failed', exception_type=type(exc).__name__, error=str(exc)[:300])
        write_json(out/'meta.json',{'status':'failed','branch':branch,'spec_sha256':expected,'error':repr(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        progress('cleanup')
        writer.shutdown(wait=True)
        for sensor in sensors:
            try:sensor.stop()
            except RuntimeError:pass
        for actor in list(reversed(sensors))+list(reversed(actors)):
            try:actor.destroy()
            except RuntimeError:pass
        if auxiliary is not None:
            auxiliary.close()
        if preserve:
            finish_capture_cleanup(world, out)
            completed_meta = json.loads((out/'meta.json').read_text())
            if completed_meta.get('status') == 'complete':
                print('COMPLETE',cfg['scene_id'],branch,13,'six-camera samples',
                      round(completed_meta['elapsed_wall_s'],1),'s; cleanup verified',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--opendrive',type=Path,default=Path(DEFAULT_XODR))
    parser.add_argument('--seed',type=int,default=20261013)
    parser.add_argument('--scene-profile',choices=('legacy_v2','town10_city_v1','lane_change_v1','diverse_v1'),default='legacy_v2')
    parser.add_argument('--vehicle-profile',choices=('native_v2','legacy_tesla'),
                        help='Defaults to native_v2 for city scenes, otherwise legacy_tesla; ego always Tesla')
    parser.add_argument('--render-preset',type=Path,help='Validated CARLA 0.9.16 complete weather/camera/engine preset; required for the city profile')
    parser.add_argument('--physics-calibration',type=Path,nargs='+',help='Passed four-model full-brake stop/restart calibration at 6 and 10 m/s; required for lane_change_v1')
    parser.add_argument('--scene');parser.add_argument('--branch',help='Must belong to the selected scene config, including its feasible action variants')
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=2000)
    parser.add_argument('--out',type=Path,help='Optional branch destination for isolated preflight/retry')
    args=parser.parse_args();args.data.mkdir(parents=True,exist_ok=True)
    if args.plan_only:plan(args);return
    if not args.scene or not args.branch:parser.error('--scene and --branch are required for actual collection')
    cfg=json.loads((args.data/args.scene/'spec.json').read_text())
    collect(args,cfg,args.branch,args.out or args.data/args.scene/'branches'/args.branch)


if __name__=='__main__':main()
