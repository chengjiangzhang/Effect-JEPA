"""Deterministic pedestrian and traffic-signal execution for paired scenes.

The caller owns the sole world.tick(). Signal changes for t+dt are sent before
that tick, so readback at t records the actual state visible in the RGB frame.
Walker controls at t are outgoing commands, while observed_control is the
command applied over the preceding interval. No branch ID reaches this module.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

if __package__:
    from .pedestrian_policy import (walker_control, validate_policy, feedback_enabled,
                                    controls_and_decisions, decision_summary, decision_schema)
else:
    from pedestrian_policy import (walker_control, validate_policy, feedback_enabled,
                                   controls_and_decisions, decision_summary, decision_schema)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def signal_phase(program, time_s):
    """Hold first/last declared phase outside schedule; no server wall clock."""
    index = 0
    for i, phase in enumerate(program['phases']):
        if time_s + 1e-9 >= phase['start_time_s']:
            index = i
    return index, program['phases'][index]['states']


def validate_auxiliary_config(config):
    roles = config['vehicle_roles']
    pedestrians = config.get('pedestrians', [])
    if not isinstance(pedestrians, list):
        raise ValueError('pedestrians must be a list')
    seen = set(roles)
    for spec in pedestrians:
        role = spec.get('role')
        if not isinstance(role, str) or not role or role in seen:
            raise ValueError('Pedestrians require unique roles disjoint from vehicle roles')
        seen.add(role)
        if not str(spec.get('blueprint_id', '')).startswith('walker.pedestrian.'):
            raise ValueError('A pedestrian requires an explicit walker.pedestrian blueprint')
        route = np.asarray(spec.get('route_points'), dtype=float)
        if (route.ndim != 2 or route.shape[1] != 3 or len(route) < 2
                or not np.isfinite(route).all()
                or np.any(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1) < .01)):
            raise ValueError('Pedestrian route_points require nondegenerate finite Nx3 samples')
        spawn = spec.get('transform', {})
        if (not isinstance(spawn, dict) or any(np.asarray(spawn.get(k)).shape != (3,)
                or not all(_finite(value) for value in spawn[k]) for k in ('position', 'rotation_deg'))):
            raise ValueError('Pedestrian transform requires finite xyz/roll/pitch/yaw')
        if np.linalg.norm(np.asarray(spawn['position'])[:2]-route[0, :2]) > .5:
            raise ValueError('Pedestrian spawn must begin at its declared route')
        if not _finite(spec.get('speed_mps')) or not 0 < spec['speed_mps'] <= 3.:
            raise ValueError('Pedestrian speed_mps must be in (0,3]')
        earliest = -2.-float(config.get('driving_warmup_s', 1.)) if spec.get('response_policy', {}).get('kind') in ('reactive_crossing_v3', 'reactive_crossing_v4') else -2.
        if not _finite(spec.get('start_time_s')) or spec['start_time_s'] < earliest-1e-9:
            raise ValueError('Pedestrian start precedes the recorded common motion initialization')
        validate_policy(spec, config)
    program = config.get('traffic_signal_program')
    if program is not None:
        groups, phases = program.get('groups', []), program.get('phases', [])
        if not groups or not phases:
            raise ValueError('Traffic signals require groups and phases')
        group_ids, lights, assigned = set(), set(), set()
        for group in groups:
            identity = group.get('id')
            if not isinstance(identity, str) or not identity or identity in group_ids:
                raise ValueError('Traffic signal group IDs must be unique')
            group_ids.add(identity)
            ids = group.get('landmark_ids')
            if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                    or any(not isinstance(i, str) or not i for i in ids) or lights.intersection(ids)):
                raise ValueError('Signal groups require unique disjoint OpenDRIVE landmark IDs')
            lights.update(ids)
            controlled = group.get('roles', [])
            stops = group.get('stop_route_s', {})
            if (not isinstance(controlled, list) or not controlled or len(set(controlled)) != len(controlled)
                    or not set(controlled).issubset(roles) or assigned.intersection(controlled)
                    or set(stops) != set(controlled)):
                raise ValueError('Signal groups require distinct vehicle roles and stop_route_s for each')
            for role in controlled:
                route = np.asarray(config['vehicles'][roles.index(role)]['route'])
                length = np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1).sum()
                if not _finite(stops[role]) or not 0 <= stops[role] <= length:
                    raise ValueError('Signal stop line must lie on its actor route')
            assigned.update(controlled)
        previous = -math.inf
        for phase in phases:
            start = phase.get('start_time_s')
            states = phase.get('states', {})
            if (not _finite(start) or start <= previous or set(states) != group_ids
                    or any(value not in ('Red', 'Yellow', 'Green') for value in states.values())):
                raise ValueError('Signal phases require sorted finite times and one valid state per group')
            previous = start
        if phases[0]['start_time_s'] > -2.:
            raise ValueError('Signal schedule must cover the entire common history')
    zone = config.get('geometry', {}).get('narrow_meeting')
    if zone is not None:
        if zone.get('policy_version') is not None:
            if zone['policy_version'] != 'narrow-current-occupancy-v2':
                raise ValueError('Unsupported narrow-road policy version')
            for name, bounds in (('lateral_margin_m',(.1,1.)),('release_confirmation_s',(.1,1.)),('passing_speed_m_s',(.5,5.))):
                if not _finite(zone.get(name)) or not bounds[0] <= zone[name] <= bounds[1]:
                    raise ValueError('Invalid narrow-road parameter: '+name)
        yielding, priority = zone.get('yield_roles', []), zone.get('priority_roles', [])
        if (not yielding or not priority or len(set(yielding)) != len(yielding)
                or len(set(priority)) != len(priority) or set(yielding).intersection(priority)
                or not set(yielding).issubset(roles) or not set(priority).issubset(roles)):
            raise ValueError('Narrow meeting requires disjoint existing yielding and priority vehicle roles')
        for kind, members in (('yield', yielding), ('priority', priority)):
            intervals = zone.get(f'{kind}_route_s', {})
            if set(intervals) != set(members):
                raise ValueError('Each narrow-road actor requires its own map-derived interval')
            for role, interval in intervals.items():
                route = np.asarray(config['vehicles'][roles.index(role)]['route'])
                length = np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1).sum()
                value = np.asarray(interval, dtype=float)
                if (value.shape != (2,) or not np.isfinite(value).all()
                        or not 0 <= value[0] < value[1] <= length + 1e-6):
                    raise ValueError('Narrow-road interval must lie on the corresponding actor route')
        for key in ('stop_margin_m', 'clearance_time_s'):
            if not _finite(zone.get(key)) or zone[key] <= 0:
                raise ValueError(f'Narrow meeting requires positive {key}')



def _vec(value):
    return [float(value.x), float(value.y), float(value.z)]


def _transform_dict(value):
    return {'position': _vec(value.location),
            'rotation_deg': [value.rotation.roll, value.rotation.pitch, value.rotation.yaw]}


class AuxiliaryActors:
    """Shared implementation used by both physics preflight and RGB capture."""
    def __init__(self, world, client, config, dt=.05, carla_module=None):
        validate_auxiliary_config(config)
        if carla_module is None:
            import carla as carla_module
        self.carla, self.world, self.client, self.config, self.dt = carla_module, world, client, config, dt
        self.actors, self.light_bindings, self.samples, self.signal_samples = [], {}, [], []
        self.specs = config.get('pedestrians', [])
        self.feedback = any(feedback_enabled(spec) for spec in self.specs)
        self.committed = [False]*len(self.specs)
        self.decision_samples = []
        self.warmup_samples = []
        self.warmup_feedback = any(s.get('response_policy', {}).get('kind') in ('reactive_crossing_v3', 'reactive_crossing_v4')
                                   and s['start_time_s'] < -2. for s in self.specs)
        self._current_observation = None
        self.program = config.get('traffic_signal_program')
        self.groups = []
        self._closed = False
        try:
            lights = list(world.get_actors().filter('traffic.traffic_light*'))
            by_id = {}
            for light in lights:
                light.set_state(self.carla.TrafficLightState.Off)
                light.freeze(True)
                by_id.setdefault(str(light.get_opendrive_id()), []).append(light)
            if self.program is not None:
                for group in self.program['groups']:
                    matches = []
                    for landmark in group['landmark_ids']:
                        bound = by_id.get(landmark, [])
                        if len(bound) != 1:
                            raise RuntimeError(f'Signal landmark {landmark}: expected one real light, found {len(bound)}')
                        self.light_bindings[landmark] = bound[0]
                        matches.append(bound[0])
                    self.groups.append({**group, 'actor_ids': [int(light.id) for light in matches],
                        'light_records': [{'landmark_id': landmark, 'actor_id': int(light.id),
                            'transform': _transform_dict(light.get_transform())}
                            for landmark, light in zip(group['landmark_ids'], matches)]})
                self.set_signal_time(-3.)
            library = world.get_blueprint_library()
            for spec in self.specs:
                blueprint = library.find(spec['blueprint_id'])
                for key, value in {**spec.get('attributes', {}), 'role_name': spec['role'], 'is_invincible': 'false'}.items():
                    if blueprint.has_attribute(key):
                        blueprint.set_attribute(key, str(value))
                transform = self._transform(spec['transform'])
                actor = world.try_spawn_actor(blueprint, transform)
                if actor is None:
                    raise RuntimeError(f"Pedestrian spawn failed: {spec['role']}")
                self.actors.append(actor)
            self.hold_pedestrians()
        except BaseException:
            self.close()
            raise

    def _transform(self, item):
        return self.carla.Transform(self.carla.Location(*item['position']),
            self.carla.Rotation(roll=item['rotation_deg'][0], pitch=item['rotation_deg'][1], yaw=item['rotation_deg'][2]))

    def _batch(self, commands):
        if not commands:
            return
        results = self.client.apply_batch_sync(commands, False)
        if len(results) != len(commands) or any(result.has_error() for result in results):
            raise RuntimeError('Auxiliary actor synchronous command failed: '+str([r.error for r in results if r.has_error()]))

    @property
    def actor_roles(self):
        return {actor.id: spec['role'] for actor, spec in zip(self.actors, self.specs)}

    def hold_pedestrians(self, reset=False):
        if reset:
            self.committed = [False]*len(self.specs)
            self.warmup_samples = []
            self._current_observation = None
            self._batch([self.carla.command.ApplyTransform(actor.id, self._transform(spec['transform']))
                         for actor, spec in zip(self.actors, self.specs)])
        self.apply_walkers(np.zeros((len(self.actors), 5)))

    def apply_walkers(self, controls):
        commands = [self.carla.command.ApplyWalkerControl(actor.id, self.carla.WalkerControl(
            direction=self.carla.Vector3D(*map(float, row[:3])), speed=float(row[3]), jump=bool(row[4])))
            for actor, row in zip(self.actors, controls)]
        self._batch(commands)

    def set_signal_time(self, time_s):
        if self.program is not None:
            _, states = signal_phase(self.program, time_s)
            for group in self.program['groups']:
                for landmark in group['landmark_ids']:
                    self.light_bindings[landmark].set_state(getattr(self.carla.TrafficLightState, states[group['id']]))

    def observe_pedestrians(self):
        transforms = [actor.get_transform() for actor in self.actors]
        controls = [actor.get_control() for actor in self.actors]
        return {'position': np.asarray([_vec(t.location) for t in transforms]).reshape(-1, 3),
                'rotation_deg': np.asarray([[t.rotation.roll, t.rotation.pitch, t.rotation.yaw] for t in transforms]).reshape(-1, 3),
                'velocity': np.asarray([_vec(a.get_velocity()) for a in self.actors]).reshape(-1, 3),
                'angular_velocity': np.asarray([_vec(a.get_angular_velocity()) for a in self.actors]).reshape(-1, 3),
                'acceleration': np.asarray([_vec(a.get_acceleration()) for a in self.actors]).reshape(-1, 3),
                'transform_world_actor': np.asarray([t.get_matrix() for t in transforms]).reshape(-1, 4, 4),
                'observed_control': np.asarray([_vec(c.direction)+[c.speed, float(c.jump)] for c in controls]).reshape(-1, 5),
                'bbox_extent': np.asarray([_vec(a.bounding_box.extent) for a in self.actors]).reshape(-1, 3),
                'bbox_offset': np.asarray([_vec(a.bounding_box.location) for a in self.actors]).reshape(-1, 3),
                'bbox_rotation_deg': np.asarray([[a.bounding_box.rotation.roll, a.bounding_box.rotation.pitch,
                    a.bounding_box.rotation.yaw] for a in self.actors]).reshape(-1, 3)}

    def signal_observation(self, time_s):
        if self.program is None:
            return None
        phase, expected = signal_phase(self.program, time_s)
        actual = {key: str(light.get_state()).rsplit('.', 1)[-1] for key, light in self.light_bindings.items()}
        for group in self.program['groups']:
            if any(actual[key] != expected[group['id']] for key in group['landmark_ids']):
                raise RuntimeError(f'Signal readback at {time_s}s differs from frozen schedule: {actual}')
        return {'phase_index': phase, 'states': dict(expected), 'actual_states': actual}

    def inject_observation(self, observed, time_s):
        if self.actors:
            observed['pedestrians'] = {**self.observe_pedestrians(),
                'roles': [s['role'] for s in self.specs],
                'bbox_extent': np.asarray([_vec(a.bounding_box.extent) for a in self.actors])}
        signal = self.signal_observation(time_s)
        if signal is not None:
            observed['signal_states'] = signal['states']
        if self.feedback:
            # Copy only measured fields, before a controller can annotate its
            # outgoing action. The sole world owner has not ticked yet.
            needed = ('position', 'velocity', 'acceleration', 'transform_world_vehicle',
                      'bbox_extent', 'bbox_offset', 'bbox_rotation_deg')
            vehicles = {key: np.asarray(observed[key]).copy() for key in needed}
            people = {key: value.copy() for key, value in observed['pedestrians'].items()
                      if isinstance(value, np.ndarray)}
            self._current_observation = (int(self.world.get_snapshot().frame), float(time_s), vehicles, people)
        return observed

    def record(self, time_s, final=False):
        snapshot = self.world.get_snapshot()
        identity = {'time_s': float(time_s), 'frame_id': int(snapshot.frame),
                    'simulation_time_s': float(snapshot.timestamp.elapsed_seconds)}
        controls = np.zeros((len(self.actors), 5))
        if self.actors:
            if self.feedback:
                cached = self._current_observation
                if cached is None or cached[0] != identity['frame_id'] or cached[1] != float(time_s):
                    raise RuntimeError('Pedestrian feedback requires this same frame/time vehicle observation')
                _, _, vehicles, observed = cached
                controls, rows, self.committed = controls_and_decisions(
                    self.config, vehicles, observed, time_s, self.dt, self.committed, final=final)
                self.decision_samples.append({**identity, 'final_boundary': bool(final), 'decisions': rows})
                self._current_observation = None  # A record may never reuse another frame's feedback.
            else:
                observed = self.observe_pedestrians()
                if not final:
                    controls = np.asarray([walker_control(position, spec, time_s, self.dt)
                                           for position, spec in zip(observed['position'], self.specs)])
            # Bounding boxes are stored once, matching existing NPZ contracts.
            dynamic = {key: value for key, value in observed.items()
                       if key not in ('bbox_extent', 'bbox_offset', 'bbox_rotation_deg')}
            self.samples.append({**dynamic, **identity, 'control': controls.copy()})
        signal = self.signal_observation(time_s)
        if signal is not None:
            self.signal_samples.append({**identity, **signal})
        return controls

    def warmup_commands(self, time_s):
        """Natural walking before model history, with complete causal evidence."""
        if not self.warmup_feedback:
            return np.zeros((len(self.actors), 5))
        snapshot = self.world.get_snapshot()
        cached = self._current_observation
        if cached is None or cached[0] != int(snapshot.frame) or cached[1] != float(time_s):
            raise RuntimeError('Warmup feedback requires the same measured frame')
        _, _, vehicles, people = cached
        controls, decisions, self.committed = controls_and_decisions(
            self.config, vehicles, people, time_s, self.dt, self.committed)
        self.warmup_samples.append({'time_s':float(time_s),'frame_id':int(snapshot.frame),
            'simulation_time_s':float(snapshot.timestamp.elapsed_seconds),
            'vehicles':{k:v.tolist() for k,v in vehicles.items()},
            'pedestrians':{k:v.tolist() for k,v in people.items()},
            'control':controls.tolist(),'decisions':decisions})
        self._current_observation = None
        return controls

    def advance_commands(self, walker_commands, next_time_s):
        self.apply_walkers(walker_commands)
        self.set_signal_time(next_time_s)

    def save(self, out):
        out = Path(out)
        if self.actors:
            values = {key: np.asarray([sample[key] for sample in self.samples]) for key in self.samples[0]}
            values.update(bbox_extent=np.asarray([_vec(a.bounding_box.extent) for a in self.actors]),
                          bbox_offset=np.asarray([_vec(a.bounding_box.location) for a in self.actors]),
                          bbox_rotation_deg=np.asarray([[a.bounding_box.rotation.roll, a.bounding_box.rotation.pitch,
                                                        a.bounding_box.rotation.yaw] for a in self.actors]))
            np.savez_compressed(out/'pedestrians.npz', **values)
        if self.feedback:
            (out/'pedestrian_decisions.json').write_text(json.dumps({
                'schema': decision_schema(self.specs), 'policies': {spec['role']: spec.get('response_policy') for spec in self.specs},
                'observation_semantics': 'Same-frame measured vehicle and pedestrian state; no future actions or branch labels',
                'summary': decision_summary(self.specs, self.decision_samples, self.samples),
                'warmup': self.warmup_samples,
                'samples': self.decision_samples}, indent=2, allow_nan=False)+'\n')
        if self.program is not None:
            (out/'signal_states.json').write_text(json.dumps({'program': self.program,
                'groups': self.groups, 'samples': self.signal_samples}, indent=2, allow_nan=False)+'\n')

    def metadata(self):
        result = {}
        if self.actors:
            result['pedestrian_roles'] = [s['role'] for s in self.specs]
            result['pedestrians'] = [{'role': spec['role'], 'actor_id': int(actor.id),
                'blueprint_id': actor.type_id, 'actor_attributes': dict(actor.attributes)}
                for spec, actor in zip(self.specs, self.actors)]
            result['pedestrian_control_columns'] = ['direction_x', 'direction_y', 'direction_z', 'speed_m_s', 'jump']
            if self.feedback:
                result['pedestrian_decision_record'] = 'pedestrian_decisions.json'
                result['pedestrian_feedback_roles'] = [spec['role'] for spec in self.specs if feedback_enabled(spec)]
                result['pedestrian_policy_note'] = 'Versioned current-state policy; v3 includes bounded faster crossing and auditable common warmup; heuristic, not a safety guarantee'
            result['pedestrian_control_timing'] = 'control[t] is next-interval command; observed_control[t] is prior applied command; final outgoing control is unexecuted zero sentinel'
        if self.program is not None:
            result['traffic_lights'] = 'Fixed scene-relative schedule, per-light readback saved in signal_states.json; NPCs obey stop lines, ego interventions remain explicit'
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        for actor in reversed(self.actors):
            try:
                actor.destroy()
            except RuntimeError:
                pass
