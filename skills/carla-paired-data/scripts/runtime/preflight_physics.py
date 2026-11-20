#!/usr/bin/env python3
"""Fast physical preflight for a frozen CARLA plan; never RGB delivery data.

No RGB sensors are spawned. Output must be a new independent directory.
Numerical equivalence with formal RGB capture must be measured, not assumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

import numpy as np

if __package__:
    from . import collect_multicam as cm
else:
    import collect_multicam as cm
if __package__:
    from .controllers_v2 import ScenarioController, branch_names
else:
    from controllers_v2 import ScenarioController, branch_names

MODE = 'physics_only_preflight_not_delivery'
HISTORY_MODE = 'common_history_only_not_delivery'
HISTORY_BRANCH = 'common_history'
HERE = Path(__file__).resolve().parent
SOURCE_FILES = ('collect_multicam.py', 'scenarios_v2.py', 'controllers_v2.py', 'npc_merge.py', 'auxiliary_actors.py', 'ego_decisions.py', 'semantic_lidar.py', 'pedestrian_policy.py', 'pedestrian_crossing.py')
LOADED_SOURCE_HASHES = cm.source_hashes()
LOADED_PREFLIGHT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SETTLE_TICKS = 21
COLLISION_DRAIN_WALL_S = .1


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_names(values, available, label):
    if not values:
        return list(available)
    selected = []
    for value in values:
        for item in value.split(','):
            item = item.strip()
            if label == 'scene' and item.isdigit():
                item = f'scene_{int(item):04d}'
            if item not in available:
                raise ValueError(f'Unknown {label}: {item}')
            selected.append(item)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError(f'Select distinct {label}s')
    return selected


def validate_config(config, data, branch=None):
    current = cm.source_hashes()
    if (current != LOADED_SOURCE_HASHES or config.get('code_sha256') != current or
            sha256(__file__) != LOADED_PREFLIGHT_SHA256):
        raise RuntimeError('Collection source differs from loaded code or frozen plan')
    if cm.digest({k: v for k, v in config.items() if k != 'spec_sha256'}) != config.get('spec_sha256'):
        raise RuntimeError('Invalid frozen config content hash')
    if branch is not None and branch not in branch_names(config):
        raise ValueError('Branch is absent from frozen config')
    roles = config['vehicle_roles']
    if (not roles or roles[0] != 'ego' or len(set(roles)) != len(roles) or roles.count('ego') != 1
            or roles != [v['role'] for v in config['vehicles']]):
        raise ValueError('Require unique frozen vehicle roles with the sole ego at index0')
    timing = config['timing']
    expected = {'physics_dt_s': .05, 'history_s': 2., 'future_s': 4.,
                'weather_warmup_s': 1., 'braked_settle_s': 1.,
                't0_physics_index': 40}
    if any(timing.get(key) != value for key, value in expected.items()):
        raise ValueError('Preflight initialization supports only the declared .05s/-2..4 protocol')
    cm.motion_warmup_seconds(config)
    cm.semantic_lidar.validate_spec(config.get('semantic_lidar'))
    profile = config['vehicle_physics_profile']
    actual = cm.vehicle_physics_profile([e['path'] for e in profile['calibration']])
    if actual != profile:
        raise RuntimeError('Physics calibration content/profile differs from frozen plan')
    cm.check_frozen_render_preset(data, config)


def prepare_output(data, out, scene_values=None, branch_values=None, *, history_only=False):
    """Validate before creating anything; preserve config/map bytes exactly."""
    data, out = Path(data).resolve(), Path(out).resolve()
    if history_only and branch_values:
        raise ValueError('--history-only measures the common prefix once; --branches is not applicable')
    if out.exists():
        raise FileExistsError(f'Use a new independent preflight directory: {out}')
    if out.is_relative_to(data) or data.is_relative_to(out):
        raise ValueError('Preflight output must be separate from the frozen input dataset')
    manifest_path = data/'manifest.json'
    original = json.loads(manifest_path.read_text())
    scenes = select_names(scene_values, original['scene_ids'], 'scene')
    requested = select_names(branch_values, original['branches'], 'branch') if branch_values else None
    configs, scene_branches, selected_branches = [], {}, {}
    for scene in scenes:
        config = json.loads((data/scene/'spec.json').read_text())
        validate_config(config, data)
        available = list(branch_names(config))
        if config['scene_id'] != scene or any(b not in available for b in (requested or [])):
            raise ValueError('Config differs from requested scene/branches')
        if scene in original.get('scene_branches', {}) and original['scene_branches'][scene] != available:
            raise ValueError('Manifest scene branches differ from frozen config')
        scene_branches[scene] = available
        selected_branches[scene] = ([HISTORY_BRANCH] if history_only else
                                    list(requested) if requested is not None else available)
        if not (data/'maps'/f"{config['map']}.xodr").is_file():
            raise FileNotFoundError(f"Missing frozen map: {config['map']}")
        configs.append(config)
    all_sources = {**LOADED_SOURCE_HASHES, Path(__file__).name: LOADED_PREFLIGHT_SHA256}
    manifest = {
        'version': 'paired-physics-preflight-v1', 'status': 'prepared',
        'capture_mode': HISTORY_MODE if history_only else MODE,
        'delivery_eligible': False, 'created_utc': now(), 'input_data': str(data),
        'input_manifest_sha256': sha256(manifest_path),
        'scene_ids': scenes, 'branches': list(dict.fromkeys(b for scene in scenes for b in scene_branches[scene])),
        'scene_branches': scene_branches, 'requested_scene_branches': selected_branches,
        'requested_branches': requested, 'requested_branch_count': sum(map(len, selected_branches.values())),
        'code_sha256': LOADED_SOURCE_HASHES, 'preflight_sources_sha256': all_sources,
        'rgb_samples': 0, 'camera_names': [], 'physics_hz': 20, 'history_s': 2.,
        'future_s': 0. if history_only else 4., 'history_only': history_only,
        'frozen_history_spec_sha256': {c['scene_id']: c['spec_sha256'] for c in configs},
        'input_spec_file_sha256': {c['scene_id']: sha256(data/c['scene_id']/'spec.json') for c in configs},
        'equivalence_to_rgb_capture': 'Unverified. Compare original physical state arrays against formal RGB capture before treating this as a numerical proxy.',
        'differences_from_delivery': [
            'No RGB cameras or semantic LiDAR, no discarded renderer-prime episode; world is reloaded before ordinary physics initialization.',
            'no_rendering_mode=True; one collision sensor per configured vehicle.',
            'Fixed21 brake ticks assume the formal first RGB arrives one tick after the brake command, then first_frame+20.',
            'Each branch has a fresh CARLA Client and reloaded world in one Python batch process; formal wrapper uses a fresh Python process per branch.',
        ], 'branch_results': [],
    }
    out.mkdir(parents=True)
    (out/'source').mkdir(); (out/'maps').mkdir()
    for name in all_sources:
        source = Path(__file__) if name == Path(__file__).name else HERE/name
        shutil.copyfile(source, out/'source'/name)
        if sha256(out/'source'/name) != all_sources[name]:
            raise RuntimeError('Source changed while snapshotting preflight code')
    for config in configs:
        (out/config['scene_id']).mkdir()
        shutil.copyfile(data/config['scene_id']/'spec.json', out/config['scene_id']/'spec.json')
    for map_name in {config['map'] for config in configs}:
        shutil.copyfile(data/'maps'/f'{map_name}.xodr', out/'maps'/f'{map_name}.xodr')
    if (data/'render_preset.json').exists():
        shutil.copyfile(data/'render_preset.json', out/'render_preset.json')
    cm.write_json(out/'manifest.json', manifest)
    return configs, selected_branches, manifest


def initialize_motion(client, world, actors, config, auxiliary=None):
    """Match the formal post-profile settle/velocity/motion-warmup sequence."""
    warmup_seconds = cm.motion_warmup_seconds(config)
    trace = {'motion_warmup_s': warmup_seconds, 'before_braked_settle_frame': int(world.get_snapshot().frame)}
    if auxiliary is not None:
        auxiliary.hold_pedestrians(reset=True)
        auxiliary.set_signal_time(-config['timing']['history_s']-warmup_seconds)
    cm.apply(client, actors, np.tile([0., 0., 1.], (len(actors), 1)))
    for _ in range(SETTLE_TICKS):
        world.tick(90)
    trace['after_braked_settle_frame'] = int(world.get_snapshot().frame)
    commands = []
    for actor, spec in zip(actors, config['vehicles']):
        forward = actor.get_transform().get_forward_vector(); speed = spec['initial_speed_mps']
        commands.extend([
            cm.carla.command.ApplyTargetVelocity(actor.id, cm.carla.Vector3D(
                forward.x*speed, forward.y*speed, forward.z*speed)),
            cm.carla.command.ApplyTargetAngularVelocity(actor.id, cm.carla.Vector3D()),
        ])
    cm.apply_commands(client, commands)
    trace['initial_velocity_assignment_frame'] = int(world.get_snapshot().frame)
    extents = np.array([cm.vec(a.bounding_box.extent) for a in actors])
    offsets = np.array([cm.vec(a.bounding_box.location) for a in actors])
    boxrot = np.array([[a.bounding_box.rotation.roll, a.bounding_box.rotation.pitch,
                       a.bounding_box.rotation.yaw] for a in actors])
    controller = ScenarioController(config, dt=cm.DT)
    for warmup_step in range(round(warmup_seconds/cm.DT)):
        warmup_time = -config['timing']['history_s']-warmup_seconds+warmup_step*cm.DT
        observed = cm.observe(actors)
        if auxiliary is not None:
            observed = auxiliary.inject_observation(observed, warmup_time)
        controls, _, _ = controller.controls(observed, 'maintain', warmup_time, extents)
        if auxiliary is not None:
            auxiliary.advance_commands(auxiliary.warmup_commands(warmup_time), warmup_time+cm.DT)
        cm.apply(client, actors, controls); world.tick(90)
    trace['after_motion_warmup_frame'] = int(world.get_snapshot().frame)
    return controller, extents, offsets, boxrot, trace


def record_physics(client, world, actors, config, branch, controller, extents, auxiliary=None,
                   *, history_only=False):
    """Save current state and previous actuation before applying next control."""
    fields = ('position', 'rotation_deg', 'velocity', 'angular_velocity', 'acceleration',
              'transform_world_vehicle', 'control', 'target_speed', 'observed_control',
              'observed_gear', 'observed_manual_gear_shift', 'requested_gear',
              'requested_manual_gear_shift')
    arrays = {key: [] for key in fields}; frames, timestamps, diagnostics = [], [], []
    ego = config.get('vehicle_roles', ['ego']).index('ego')
    base_speed = float(config['vehicles'][ego]['initial_speed_mps'])
    final_step = 40 if history_only else 120
    for step in range(final_step+1):
        t = -2+step*cm.DT
        observed = cm.observe(actors); observed.update(cm.observe_actuation(actors))
        if auxiliary is not None:
            auxiliary.inject_observation(observed, t)
        if step == 40:
            base_speed = float(np.linalg.norm(observed['velocity'][ego, :2]))
        active_branch = 'maintain' if step < 40 else ('maintain' if branch == 'maintain_repeat' else branch)
        if history_only and step == final_step:
            # Observe t0 before any intervention. In particular do not let a
            # candidate future profile mutate controller state or query NPCs.
            controls = np.zeros((len(actors), 3), dtype=np.float64)
            targets = np.zeros(len(actors), dtype=np.float64)
            diagnostic = [{'role': role, 'phase': 'common_history_boundary_observation',
                           'future_control_evaluated': False}
                          for role in config['vehicle_roles']]
        else:
            controls, targets, diagnostic = controller.controls(
                observed, active_branch, t, extents, ego_base_speed=base_speed)
        if step == final_step:
            controls[:] = 0.
        for key in fields:
            arrays[key].append(controls.copy() if key == 'control' else
                               np.asarray(targets).copy() if key == 'target_speed' else observed[key])
        snapshot = world.get_snapshot()
        frames.append(snapshot.frame); timestamps.append(snapshot.timestamp.elapsed_seconds)
        diagnostics.append(diagnostic)
        walker_commands = auxiliary.record(t, final=step == final_step) if auxiliary is not None else None
        if step < final_step:
            if auxiliary is not None:
                auxiliary.advance_commands(walker_commands, t+cm.DT)
            cm.apply(client, actors, controls); world.tick(90)
    values = {key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()}
    values.update(time_s=np.arange(final_step+1)*cm.DT-2, frame_id=np.asarray(frames, dtype=np.int64),
                  simulation_time_s=np.asarray(timestamps))
    return values, diagnostics, base_speed


def collect_branch(data, out, config, branch, host, port, *, history_only=False):
    if history_only and branch != HISTORY_BRANCH:
        raise ValueError('History-only capture must use the common_history destination')
    validate_config(config, data, None if history_only else branch)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); actors, sensors, collisions = [], [], []
    auxiliary = None
    collision_lock = threading.Lock(); versions = {}; trace = {}
    base_meta = {'status': 'partial', 'capture_mode': HISTORY_MODE if history_only else MODE,
                 'delivery_eligible': False, 'history_only': history_only,
                 'history_only_complete': False, 't0_index': 40,
                 'branch': branch, 'spec_sha256': config['spec_sha256'],
                 'code_sha256': cm.source_hashes(), 'preflight_source_sha256': sha256(__file__),
                 'command': sys.argv, 'rgb_samples': 0, 'camera_names': []}
    cm.write_json(out/'meta.json', base_meta)
    try:
        client = cm.carla.Client(host, port); client.set_timeout(120)
        versions = {'client': client.get_client_version(), 'server': client.get_server_version()}
        if any(v != '0.9.16' for v in versions.values()):
            raise RuntimeError(f'CARLA0.9.16 required: {versions}')
        world = client.get_world()
        if world.get_map().name.rsplit('/', 1)[-1] != config['map']:
            world = client.load_world(config['map'])
        cm.configure_world(world); world = client.reload_world(False); cm.configure_world(world)
        settings = world.get_settings(); settings.no_rendering_mode = True; world.apply_settings(settings)
        trace['after_reload_frame'] = int(world.get_snapshot().frame)
        world.set_weather(cm.carla.WeatherParameters(**config['weather']))
        for _ in range(round(config['timing']['weather_warmup_s']/cm.DT)):
            world.tick(90)
        trace['after_weather_warmup_frame'] = int(world.get_snapshot().frame)
        weather_after_warmup = cm.checked_weather(world, config['weather'])
        cm.write_json(out/'environment_vehicles.json', cm.environment_vehicles(
            world, cm.transform(config['vehicles'][config['vehicle_roles'].index('ego')]['transform']).location))
        lib = world.get_blueprint_library()
        for spec in config['vehicles']:
            blueprint = lib.find(spec['blueprint_id'])
            for key, value in spec.get('attributes', spec.get('blueprint_attributes', {})).items():
                if blueprint.has_attribute(key):
                    blueprint.set_attribute(key, str(value))
            actor = world.try_spawn_actor(blueprint, cm.transform(spec['transform']))
            if actor is None:
                raise RuntimeError(f"Spawn failed: {spec['role']}")
            actors.append(actor)
        cm.apply_commands(client, [cm.carla.command.SetAutopilot(a.id, False) for a in actors])
        roles = config['vehicle_roles']; actor_roles = dict(zip([a.id for a in actors], roles))
        auxiliary = cm.AuxiliaryActors(world, client, config, cm.DT, carla_module=cm.carla)
        actor_roles.update(auxiliary.actor_roles)
        trace['before_physics_profile_frame'] = int(world.get_snapshot().frame)
        physical_records = cm.apply_physics_profile(client, world, actors, roles, config['vehicle_physics_profile'])
        trace['after_physics_profile_frame'] = int(world.get_snapshot().frame)
        collision_actors = actors + auxiliary.actors
        collision_roles = roles + [spec['role'] for spec in config.get('pedestrians', [])]
        for actor, role in zip(collision_actors, collision_roles):
            sensor = world.spawn_actor(lib.find('sensor.other.collision'), cm.carla.Transform(), attach_to=actor)
            def on_collision(event, role=role):
                with collision_lock:
                    collisions.append({'frame_id': int(event.frame), 'role': role,
                        'other_actor_id': event.other_actor.id,
                        'other_role': actor_roles.get(event.other_actor.id, event.other_actor.type_id),
                        'normal_impulse': cm.vec(event.normal_impulse)})
            sensor.listen(on_collision); sensors.append(sensor)
        controller, extents, offsets, boxrot, motion_trace = initialize_motion(client, world, actors, config, auxiliary)
        trace.update(motion_trace)
        weather_at_history_start = cm.checked_weather(world, config['weather'])
        values, diagnostics, base_speed = record_physics(client, world, actors, config, branch,
            controller, extents, auxiliary, history_only=history_only)
        values.update(bbox_extent=extents, bbox_offset=offsets, bbox_rotation_deg=boxrot)
        np.savez_compressed(out/'states.npz', **values)
        auxiliary.save(out)
        cm.write_json(out/'controller_diagnostics.json', diagnostics)
        if controller.merge_memory:
            cm.write_json(out/'npc_merge_state.json',controller.merge_evidence())
        # Collision events use an asynchronous sensor stream. Drain callbacks
        # without advancing physical time; no hidden post-recording world ticks.
        time.sleep(COLLISION_DRAIN_WALL_S)
        t0_frame = int(values['frame_id'][40])
        with collision_lock:
            recorded_collisions = [{**event, 'time_s': (event['frame_id']-t0_frame)*cm.DT} for event in collisions]
        trace.update(history_first_frame=int(values['frame_id'][0]), t0_frame=t0_frame,
                     final_recorded_frame=int(values['frame_id'][-1]))
        actual_settings = world.get_settings()
        meta = {**base_meta, 'status': 'complete', 'created_utc': now(),
            'history_only_complete': history_only,
            'elapsed_wall_s': time.monotonic()-start,
            'carla_client_version': versions['client'], 'carla_server_version': versions['server'],
            'vehicle_roles': roles, 'actor_ids': [a.id for a in actors],
            'vehicles': [{'role': r, 'actor_id': a.id, 'blueprint_id': a.type_id,
                          'actor_attributes': dict(a.attributes)} for r, a in zip(roles, actors)],
            'vehicle_physics_profile': config['vehicle_physics_profile'], 'vehicle_physics': physical_records,
            'physics_frames': len(values['time_s']), 'collisions': recorded_collisions, 't0_frame': t0_frame,
            'ego_speed_t0_m_s': base_speed, 'weather_after_warmup': weather_after_warmup,
            'weather_at_history_start': weather_at_history_start,
            'weather_actual': cm.checked_weather(world, config['weather']),
            'control_dispatch': config['control_dispatch'],
            'control_timing': 'control[t] is applied over state[t] -> state[t+1]; last row is unexecuted zero sentinel',
            'control_time_convention': 'control and requested_* belong to the next interval (last frame is not executed); observed_* are simulator-returned prior applied controls.',
            'coordinates': 'CARLA world xyz metres; roll,pitch,yaw degrees',
            'npc_branch_information': 'none; policies see actual positions, velocities, fixed routes and right-of-way only',
            'traffic_lights': 'frozen off; intersections explicitly unsignalized',
            'initialization_frame_trace': trace,
            'initialization_ticks': {'weather_warmup': 20, 'physics_profile': 1,
                                     'braked_settle': SETTLE_TICKS, 'motion_warmup': round(cm.motion_warmup_seconds(config)/cm.DT),
                                     'history': 40, 'future': 0 if history_only else 80},
            'initial_velocity_assignment': 'Exactly once before motion warmup, matching formal collection; none during saved history/future.',
            'no_rendering_mode': bool(actual_settings.no_rendering_mode),
            'world_settings': {key: getattr(actual_settings, key) for key in ('synchronous_mode',
                'fixed_delta_seconds', 'substepping', 'max_substep_delta_time', 'max_substeps')},
            'renderer_prime': 'Skipped: discarded graphics-only episode; formal reloads world afterwards.',
            'collision_callback_drain_wall_s': COLLISION_DRAIN_WALL_S,
            'equivalence_to_rgb_capture': 'Unverified until physical arrays are compared to formal RGB data.',
        }
        meta.update(auxiliary.metadata())
        if auxiliary.actors:
            meta['collision_sensor_roles'] = collision_roles
        if (cm.source_hashes() != LOADED_SOURCE_HASHES or
                sha256(__file__) != LOADED_PREFLIGHT_SHA256):
            raise RuntimeError('Frozen shared source changed during preflight')
        cm.write_json(out/'meta.json', meta)
        return {'scene': config['scene_id'], 'branch': branch, 'status': 'complete',
                'capture_mode': HISTORY_MODE if history_only else MODE,
                'frozen_history_spec_sha256': config['spec_sha256'],
                'meta_sha256': sha256(out/'meta.json'),
                'states_sha256': sha256(out/'states.npz'), 'elapsed_wall_s': meta['elapsed_wall_s']}
    except BaseException as exc:
        cm.write_json(out/'meta.json', {**base_meta, 'status': 'failed', 'error': repr(exc),
                      'traceback': traceback.format_exc(), 'initialization_frame_trace': trace})
        raise
    finally:
        for sensor in sensors:
            try:
                sensor.stop()
            except RuntimeError:
                pass
        for actor in list(reversed(sensors))+list(reversed(actors)):
            try:
                actor.destroy()
            except RuntimeError:
                pass
        if auxiliary is not None:
            auxiliary.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Existing frozen formal plan; read only')
    parser.add_argument('--out', type=Path, required=True, help='New independent data-disk preflight directory')
    parser.add_argument('--scenes', nargs='+', help='Scene IDs/numbers, optionally comma separated; default all')
    parser.add_argument('--branches', nargs='+', help='Branch names, optionally comma separated; default all')
    parser.add_argument('--history-only', action='store_true',
                        help='Observe only the shared -2..0s prefix once, before planning interventions')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    args = parser.parse_args()
    # Same advisory lock as the RGB batch wrapper. Raw/manual CARLA clients must
    # still be coordinated externally; a lock is not simulator authentication.
    key = hashlib.sha256(f'{args.host}:{args.port}'.encode()).hexdigest()[:12]
    with open(f'/tmp/effect_jepa_multicam_{key}.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another collection wrapper owns this CARLA port') from exc
        configs, scene_branches, manifest = prepare_output(args.data, args.out, args.scenes, args.branches,
                                                          history_only=args.history_only)
        manifest.update(status='running', command=sys.argv, started_utc=now())
        cm.write_json(args.out/'manifest.json', manifest)
        try:
            for config in configs:
                for branch in scene_branches[config['scene_id']]:
                    result = collect_branch(args.data, args.out/config['scene_id']/'branches'/branch,
                                            config, branch, args.host, args.port, history_only=args.history_only)
                    manifest['branch_results'].append(result)
                    cm.write_json(args.out/'manifest.json', manifest)
                    print(json.dumps(result), flush=True)
            manifest.update(status='complete', finished_utc=now())
        except BaseException as exc:
            manifest.update(status='failed', error=repr(exc), finished_utc=now())
            raise
        finally:
            cm.write_json(args.out/'manifest.json', manifest)


if __name__ == '__main__':
    main()
