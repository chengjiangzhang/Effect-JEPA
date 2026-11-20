"""Read-only pedestrian and traffic-signal evidence checks.

These optional records are mandatory for scenes that configure those actors.
Vehicle-only captures retain their original schema. No CARLA client is needed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .review_common import INITIAL_TOLERANCES, state_errors
    from .pedestrian_policy import feedback_enabled, controls_and_decisions, validate_policy, decision_summary, decision_schema
else:
    from review_common import INITIAL_TOLERANCES, state_errors
    from pedestrian_policy import feedback_enabled, controls_and_decisions, validate_policy, decision_summary, decision_schema


WALKER_VECTORS = ('position', 'rotation_deg', 'velocity', 'angular_velocity', 'acceleration')
WALKER_BOXES = ('bbox_extent', 'bbox_offset', 'bbox_rotation_deg')
CLOCKS = ('frame_id', 'simulation_time_s')


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pedestrian_shapes(config, count):
    size = len(config.get('pedestrians', []))
    shapes = {key: (count, size, 3) for key in WALKER_VECTORS}
    shapes.update({key: (size, 3) for key in WALKER_BOXES})
    shapes.update(transform_world_actor=(count, size, 4, 4),
                  control=(count, size, 5), observed_control=(count, size, 5),
                  time_s=(count,), frame_id=(count,), simulation_time_s=(count,))
    return shapes


def pedestrian_errors(values, config, meta, vehicle_states):
    errors = []
    specifications = config.get('pedestrians', [])
    roles = [item['role'] for item in specifications]
    records = meta.get('pedestrians', [])
    if not roles or len(set(roles)) != len(roles) or set(roles) & set(config['vehicle_roles']):
        errors.append('Pedestrian roles must be distinct from all vehicle roles')
    if meta.get('pedestrian_roles') != roles or not isinstance(records, list) or \
            len(records) != len(roles) or any(not isinstance(row, dict) for row in records):
        errors.append('Pedestrian metadata identity/order differs from config')
    else:
        actor_ids = []
        for spec, record in zip(specifications, records):
            if record.get('role') != spec['role'] or record.get('blueprint_id') != spec['blueprint_id']:
                errors.append('Pedestrian recorded role/blueprint differs from frozen actor')
            actor = record.get('actor_id')
            if isinstance(actor, bool) or not isinstance(actor, int) or actor < 0:
                errors.append('Pedestrian recorded actor ID is invalid')
            actor_ids.append(actor)
            if not isinstance(record.get('actor_attributes'), dict):
                errors.append('Pedestrian actual actor attributes missing')
        if len(set(actor_ids)) != len(actor_ids) or set(actor_ids) & set(meta.get('actor_ids', [])):
            errors.append('Pedestrian actor IDs duplicate another recorded actor')
    count = len(vehicle_states['frame_id'])
    for key, shape in pedestrian_shapes(config, count).items():
        value = values.get(key)
        if value is None or value.shape != shape or value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            errors.append(f'Invalid finite pedestrian array {key}, expected {shape}')
    if any('pedestrian array' in error for error in errors):
        return errors
    for key in ('time_s', 'frame_id', 'simulation_time_s'):
        if key not in vehicle_states or not np.array_equal(values[key], vehicle_states[key]):
            errors.append(f'Pedestrian {key} differs from vehicle physics samples')
    if not np.issubdtype(values['frame_id'].dtype, np.integer):
        errors.append('Pedestrian frame IDs must be integers')
    if not np.allclose(values['position'], values['transform_world_actor'][..., :3, 3], atol=1e-5, rtol=0):
        errors.append('Pedestrian position differs from measured actor transform')
    rotation = values['transform_world_actor'][..., :3, :3]
    if not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-5, rtol=0) or \
            not np.allclose(np.linalg.det(rotation), 1., atol=1e-5, rtol=0):
        errors.append('Pedestrian transform is not rigid')
    if not np.allclose(values['transform_world_actor'][..., 3, :], [0., 0., 0., 1.], atol=1e-6, rtol=0):
        errors.append('Pedestrian transform has invalid homogeneous row')
    if np.any(values['bbox_extent'] <= 0):
        errors.append('Pedestrian bounding-box extents must be positive')
    for key in ('control', 'observed_control'):
        value = values[key]
        if np.any(value[..., 3] < 0) or np.any(~np.isin(value[..., 4], (0., 1.))) or \
                np.any(np.linalg.norm(value[..., :3], axis=-1) > 1.00001):
            errors.append(f'Pedestrian {key} has invalid direction/speed/jump')
    return errors


def signal_identity(record):
    """Stable light identity excludes per-world-reload CARLA actor IDs."""
    return [{key: value for key, value in group.items() if key != 'actor_ids' and key != 'light_records'} |
            {'light_records': [{key: value for key, value in light.items() if key != 'actor_id'}
                               for light in group.get('light_records', [])]}
            for group in record.get('groups', [])]


def signal_errors(record, config, vehicle_states):
    errors = []
    if not isinstance(record, dict):
        return ['Traffic-signal record must be an object']
    program = config['traffic_signal_program']
    if record.get('program') != program:
        errors.append('Recorded traffic-signal program differs from frozen config')
    declared, groups = program.get('groups', []), record.get('groups', [])
    phases = program.get('phases', [])
    if not isinstance(groups, list) or any(not isinstance(group, dict) for group in groups):
        return errors + ['Traffic-signal groups must be objects']
    if not declared or not phases or len(groups) != len(declared):
        return errors + ['Traffic-signal program/groups/phases missing']
    ids = [group['id'] for group in declared]
    if len(set(ids)) != len(ids) or [group.get('id') for group in groups] != ids:
        errors.append('Traffic-signal group IDs/order differ from config')
    landmarks = []
    for spec, actual in zip(declared, groups):
        for key in ('id', 'landmark_ids', 'roles', 'stop_route_s'):
            if actual.get(key) != spec.get(key):
                errors.append(f'Traffic-signal group {spec["id"]} {key} differs from config')
        lights = actual.get('light_records', [])
        if not isinstance(lights, list) or any(not isinstance(light, dict) for light in lights):
            errors.append('Traffic-signal measured light records must be objects')
            continue
        if not spec.get('landmark_ids') or len(set(spec['landmark_ids'])) != len(spec['landmark_ids']):
            errors.append('Traffic-signal landmark IDs must be nonempty and unique')
        if [light.get('landmark_id') for light in lights] != spec.get('landmark_ids'):
            errors.append('Traffic-signal measured light identity differs from planned landmarks')
        actor_ids = [light.get('actor_id') for light in lights]
        if actual.get('actor_ids') != actor_ids or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in actor_ids):
            errors.append('Traffic-signal actual actor IDs missing or inconsistent')
        for light in lights:
            transform = light.get('transform')
            try:
                position, rotation = transform['position'], transform['rotation_deg']
                valid = np.asarray(position).shape == (3,) and np.asarray(rotation).shape == (3,) and np.isfinite(position + rotation).all()
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                errors.append('Traffic-signal measured light transform missing or invalid')
        landmarks.extend(spec.get('landmark_ids', []))
    if len(set(landmarks)) != len(landmarks):
        errors.append('Traffic-signal landmark belongs to multiple groups')
    starts = np.asarray([phase.get('start_time_s') for phase in phases], dtype=float)
    times = vehicle_states['time_s']
    if not np.isfinite(starts).all() or np.any(np.diff(starts) <= 0) or starts[0] > times[0] + 1e-8:
        return errors + ['Traffic-signal phase starts do not cover the recorded timeline']
    for phase in phases:
        if set(phase.get('states', {})) != set(ids) or any(state not in ('Red', 'Yellow', 'Green') for state in phase.get('states', {}).values()):
            errors.append('Traffic-signal phase group/state declaration invalid')
    samples = record.get('samples', [])
    if not isinstance(samples, list) or len(samples) != len(times):
        return errors + ['Traffic-signal observed sample count differs from physics states']
    for index, (sample, stamp) in enumerate(zip(samples, times)):
        if not isinstance(sample, dict):
            errors.append(f'Traffic-signal sample {index} must be an object')
            continue
        if sample.get('time_s') != float(stamp) or sample.get('frame_id') != int(vehicle_states['frame_id'][index]) or \
                sample.get('simulation_time_s') != float(vehicle_states['simulation_time_s'][index]):
            errors.append(f'Traffic-signal sample {index} is not synchronized to vehicle states')
        phase_index = int(np.searchsorted(starts, float(stamp) + 1e-9, side='right') - 1)
        expected = phases[phase_index]['states']
        if sample.get('phase_index') != phase_index or sample.get('states') != expected:
            errors.append(f'Traffic-signal sample {index} differs from scheduled phase')
        actual = sample.get('actual_states', {})
        if not isinstance(actual, dict) or set(actual) != set(landmarks):
            errors.append(f'Traffic-signal sample {index} lacks per-light readback')
        elif any(actual[landmark] != expected[group['id']] for group in declared for landmark in group['landmark_ids']):
            errors.append(f'Traffic-signal sample {index} readback differs from commanded state')
    return errors


def replay_warmup(record, walkers, config, vehicle_states):
    """Reconstruct the hidden policy state from saved observations, never trust it."""
    specs = config.get('pedestrians', [])
    committed = [False]*len(specs)
    rows = record.get('warmup', [])
    enabled = any(s.get('response_policy', {}).get('kind') in ('reactive_crossing_v3', 'reactive_crossing_v4')
                  and s['start_time_s'] < -2. for s in specs)
    seconds = float(config.get('driving_warmup_s', 1.))
    count = round(seconds/.05) if enabled else 0
    if not isinstance(rows, list) or len(rows) != count:
        return ['Missing or unexpected complete pedestrian warmup evidence'], committed
    errors = []
    previous = None
    for i, row in enumerate(rows):
        expected_time = -2.-seconds+i*.05
        if (abs(row['time_s']-expected_time)>1e-8 or
                row['frame_id'] != int(vehicle_states['frame_id'][0])-count+i or
                abs(row['simulation_time_s']-float(vehicle_states['simulation_time_s'][0])+(count-i)*.05)>1e-6):
            errors.append(f'Warmup observation {i} has wrong frame or time')
        vehicles = {k:np.asarray(v) for k,v in row['vehicles'].items()}
        people = {k:np.asarray(v) for k,v in row['pedestrians'].items()}
        if any(not np.isfinite(v).all() for v in [*vehicles.values(), *people.values()]):
            errors.append(f'Warmup observation {i} is nonfinite')
        control, decisions, committed = controls_and_decisions(
            config, vehicles, people, float(row['time_s']), .05, committed)
        if decisions != row['decisions'] or not np.allclose(row['control'],control,rtol=0,atol=1e-9):
            errors.append(f'Warmup decision {i} differs from causal replay')
        if previous is not None and not np.allclose(people['observed_control'],previous,rtol=0,atol=1e-6):
            errors.append(f'Warmup actuation {i} differs from previous command')
        previous = control
    if previous is not None and not np.allclose(walkers['observed_control'][0],previous,rtol=0,atol=1e-6):
        errors.append('Warmup command does not match first model-history actuation')
    return errors, committed


def pedestrian_decision_errors(record, walkers, config, meta, vehicle_states):
    """Recompute outgoing commands from recorded same-frame observations.

    The final sample is an unexecuted zero sentinel, including history-only t0.
    No future motion, branch label or vehicle requested command enters policy.
    """
    errors = []
    specs = config.get('pedestrians', [])
    for spec in specs:
        validate_policy(spec, config)
    if record.get('schema') != decision_schema(specs) or record.get('policies') != {s['role']:s.get('response_policy') for s in specs}:
        return ['Pedestrian decision schema/policies differ from frozen spec']
    if (meta.get('pedestrian_decision_record') != 'pedestrian_decisions.json'
            or meta.get('pedestrian_feedback_roles') != [s['role'] for s in specs if feedback_enabled(s)]):
        errors.append('Pedestrian feedback metadata is missing or differs from frozen roles')
    samples = record.get('samples', [])
    count = len(vehicle_states['time_s'])
    if not isinstance(samples, list) or len(samples) != count or count not in (41, 121):
        return errors+['Pedestrian decision timeline must include every history/future physics sample']
    measured_samples=[{'time_s':walkers['time_s'][i], 'velocity':walkers['velocity'][i], 'position':walkers['position'][i]} for i in range(count)]
    if record.get('summary') != decision_summary(specs,samples,measured_samples):
        errors.append('Pedestrian commanded release / measured motion onset summary differs from recorded evidence')
    warmup_errors, committed = replay_warmup(record, walkers, config, vehicle_states)
    errors.extend(warmup_errors)
    vehicle_fields = ('position','velocity','acceleration','transform_world_vehicle')
    boxes = ('bbox_extent','bbox_offset','bbox_rotation_deg')
    for index, stamp in enumerate(vehicle_states['time_s']):
        row = samples[index]
        if (not isinstance(row, dict) or row.get('time_s') != float(stamp)
                or row.get('frame_id') != int(vehicle_states['frame_id'][index])
                or row.get('simulation_time_s') != float(vehicle_states['simulation_time_s'][index])
                or row.get('final_boundary') is not (index == count-1)):
            errors.append(f'Pedestrian decision {index} has stale/wrong frame/time/boundary')
            continue
        vehicles = {key: vehicle_states[key][index] for key in vehicle_fields}
        vehicles.update({key: vehicle_states[key] for key in boxes})
        people = {key: walkers[key][index] for key in ('position','velocity','transform_world_actor')}
        people.update({key: walkers[key] for key in boxes})
        controls, expected, committed = controls_and_decisions(
            config, vehicles, people, float(stamp), .05, committed, final=index == count-1)
        if row.get('decisions') != expected:
            errors.append(f'Pedestrian decision {index} differs from current-state policy replay')
        if not np.allclose(walkers['control'][index], controls, rtol=0, atol=1e-9):
            errors.append(f'Pedestrian commanded control {index} differs from policy replay')
    return errors


def inspect_auxiliary(folder, config, meta, vehicle_states):
    folder = Path(folder)
    result = {'status': 'passed', 'errors': [], 'input_sha256': {},
              'pedestrian_count': len(config.get('pedestrians', [])),
              'signal_group_count': len(config.get('traffic_signal_program', {}).get('groups', []))}
    errors = result['errors']
    wanted = {'pedestrians.npz': bool(config.get('pedestrians')),
              'signal_states.json': bool(config.get('traffic_signal_program')),
              'pedestrian_decisions.json': any(feedback_enabled(s) for s in config.get('pedestrians', []))}
    for name, required in wanted.items():
        path = folder / name
        if not required:
            if path.exists():
                errors.append(f'Unconfigured auxiliary record: {name}')
            continue
        if not path.is_file():
            errors.append(f'Missing required auxiliary record: {name}')
            continue
        initial_hash = digest(path); result['input_sha256'][name] = initial_hash
        try:
            if name == 'pedestrians.npz':
                errors.extend(pedestrian_errors(read_arrays(path), config, meta, vehicle_states))
            elif name == 'pedestrian_decisions.json':
                errors.extend(pedestrian_decision_errors(json.loads(path.read_text()),
                    read_arrays(folder/'pedestrians.npz'), config, meta, vehicle_states))
            else:
                errors.extend(signal_errors(json.loads(path.read_text()), config, vehicle_states))
        except (OSError, KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')
        if digest(path) != initial_hash:
            errors.append(f'Auxiliary input changed during audit: {name}')
    if wanted['pedestrians.npz']:
        roles = set(config['vehicle_roles']) | {p['role'] for p in config['pedestrians']}
        sensor_roles = meta.get('collision_sensor_roles', [])
        if len(sensor_roles) != len(roles) or set(sensor_roles) != roles:
            errors.append('Collision-sensor coverage must include every vehicle and pedestrian')
        events = meta.get('collisions')
        if not isinstance(events, list):
            errors.append('Vehicle/pedestrian collision records missing')
        else:
            for event in events:
                try:
                    valid = event['role'] in roles and np.isfinite(event['time_s']) and \
                            isinstance(event['frame_id'], int) and len(event['normal_impulse']) == 3 and \
                            np.isfinite(event['normal_impulse']).all() and isinstance(event['other_actor_id'], int)
                except (TypeError, KeyError):
                    valid = False
                if not valid:
                    errors.append('Invalid vehicle/pedestrian collision identity or measured impulse')
                    continue
                if event['time_s'] <= 1e-9:
                    errors.append('Contact during initialization/common history or at t0')
                times = vehicle_states.get('time_s', [])
                if len(times) > 1:
                    expected_time = times[0] + (event['frame_id']-vehicle_states['frame_id'][0])*(times[1]-times[0])
                    if not np.isclose(event['time_s'], expected_time, rtol=0, atol=1e-7):
                        errors.append('Collision frame ID and intervention-relative time disagree')
    if not wanted['pedestrians.npz'] and (meta.get('pedestrian_roles') or meta.get('pedestrians')):
        errors.append('Metadata records unconfigured pedestrians')
    result['status'] = 'failed' if errors else 'passed'
    return result


def compare_auxiliary(actual_folder, reference_folder, config, *, scope='exact'):
    """Exact physics/RGB correspondence or common-history/repeat QA.

    Signal identities and all scheduled/measured phases are always compared in
    full: signal interventions may not change with the ego action branch.
    """
    if scope not in ('exact', 'history', 'repeat'):
        raise ValueError('Unknown auxiliary comparison scope')
    actual_folder, reference_folder = Path(actual_folder), Path(reference_folder)
    errors, result = [], {'scope': scope, 'errors': []}
    if config.get('pedestrians'):
        try:
            actual = read_arrays(actual_folder / 'pedestrians.npz')
            reference = read_arrays(reference_folder / 'pedestrians.npz')
            if set(actual) != set(reference):
                errors.append('Pedestrian state field sets differ')
            elif scope == 'exact':
                for key in actual:
                    a, b = actual[key], reference[key]
                    if key in CLOCKS:
                        a, b = a-a[0], b-b[0]
                    if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
                        errors.append('Pedestrian physics/RGB state differs: ' + key)
            else:
                count = len(actual['time_s']); history = np.flatnonzero(actual['time_s'] <= 1e-9)
                indices = history if scope == 'history' else np.arange(count)
                state_difference = state_errors(actual, reference, indices)
                result['pedestrian_state_errors'] = state_difference
                limits = INITIAL_TOLERANCES if scope == 'history' else {**INITIAL_TOLERANCES, 'position_m': .05}
                errors.extend(f'Pedestrian {scope} {key} exceeds {limit}' for key, limit in limits.items() if state_difference[key] > limit)
                for key in WALKER_BOXES:
                    if not np.array_equal(actual[key], reference[key]):
                        errors.append('Pedestrian geometry differs: ' + key)
                # The policy is fixed, but future commands may react to ego.
                for key in ('control', 'observed_control'):
                    selected = np.flatnonzero(actual['time_s'] < -1e-9) if scope == 'history' and key == 'control' else indices
                    if not np.allclose(actual[key][selected], reference[key][selected], atol=1e-6, rtol=0):
                        errors.append('Pedestrian common-policy control mismatch: ' + key)
        except (OSError, KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            errors.append(f'Pedestrian comparison cannot complete: {exc}')
    if any(feedback_enabled(s) for s in config.get('pedestrians', [])):
        try:
            a = json.loads((actual_folder/'pedestrian_decisions.json').read_text())
            b = json.loads((reference_folder/'pedestrian_decisions.json').read_text())
            if a.get('schema') != b.get('schema') or a.get('policies') != b.get('policies'):
                errors.append('Pedestrian feedback contract differs across captures')
            def selected(record):
                rows = record['samples']
                frame0, time0 = rows[0]['frame_id'], rows[0]['simulation_time_s']
                result = []
                for row in rows:
                    if scope == 'history' and row['time_s'] >= -1e-9:
                        continue  # t0 outgoing policy is not evaluated in history-only mode.
                    normalized = {**row, 'frame_id':row['frame_id']-frame0,
                                  'simulation_time_s':row['simulation_time_s']-time0}
                    if scope != 'exact':
                        # Preserve existing state tolerances, but require the
                        # same actual feedback decisions and commitment.
                        normalized.pop('simulation_time_s')
                        normalized['decisions'] = [{key:value for key,value in d.items()
                            if key in ('role','kind','reason','allow_walk','committed_before','committed_after','mode','state_before','state_after')}
                            for d in row['decisions']]
                    result.append(normalized)
                return result
            if selected(a) != selected(b):
                errors.append('Pedestrian feedback decisions differ for '+scope)
            def warmup_decisions(record):
                rows = record.get('warmup', [])
                frame0 = record['samples'][0]['frame_id']
                return [{'time_s':r['time_s'],'frame_from_history':r['frame_id']-frame0,
                    'decisions':[{k:v for k,v in d.items() if k in
                        ('role','kind','reason','allow_walk','committed_before','committed_after','mode','state_before','state_after')}
                        for d in r['decisions']]} for r in rows]
            if warmup_decisions(a) != warmup_decisions(b):
                errors.append('Pedestrian warmup policy state differs for '+scope)
        except (OSError, KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            errors.append(f'Pedestrian decision comparison cannot complete: {exc}')
    if config.get('traffic_signal_program'):
        try:
            a = json.loads((actual_folder / 'signal_states.json').read_text())
            b = json.loads((reference_folder / 'signal_states.json').read_text())
            if a.get('program') != b.get('program') or signal_identity(a) != signal_identity(b):
                errors.append('Traffic-signal stable identity/program differs across captures')
            def normalized(record):
                samples = record['samples']; frame0 = samples[0]['frame_id']; time0 = samples[0]['simulation_time_s']
                rows = [{**row, 'frame_id': row['frame_id']-frame0} for row in samples]
                elapsed = np.asarray([row.pop('simulation_time_s')-time0 for row in rows])
                return rows, elapsed
            rows_a, elapsed_a = normalized(a); rows_b, elapsed_b = normalized(b)
            clocks_equal = (elapsed_a.shape == elapsed_b.shape and
                (np.array_equal(elapsed_a, elapsed_b) if scope=='exact' else
                 np.allclose(elapsed_a, elapsed_b, atol=1e-6, rtol=0)))
            if rows_a != rows_b or not clocks_equal:
                errors.append('Traffic-signal sequence differs across captures')
        except (OSError, KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            errors.append(f'Traffic-signal comparison cannot complete: {exc}')
    result['errors'] = errors; result['status'] = 'failed' if errors else 'passed'
    return result


def collision_identity(meta):
    """Sort measured events without comparing ephemeral actor IDs."""
    normalized = []
    for event in meta['collisions']:
        normalized.append({'role': event['role'], 'other_role': event['other_role'],
                           'time_s': event['time_s'], 'relative_frame': event['frame_id']-meta['t0_frame'],
                           'normal_impulse': event['normal_impulse']})
    return sorted(normalized, key=lambda value: json.dumps(value, sort_keys=True))
