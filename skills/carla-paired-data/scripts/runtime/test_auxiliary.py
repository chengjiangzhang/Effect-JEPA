"""Offline actor/signal lifecycle tests; fake ticks are not CARLA validation."""
import copy
import fnmatch
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import carla
import numpy as np

if __package__:
    from .auxiliary_actors import AuxiliaryActors, signal_phase, validate_auxiliary_config, walker_control
    from .controllers_v2 import ScenarioController, npc_target, select_action_profiles
else:
    from auxiliary_actors import AuxiliaryActors, signal_phase, validate_auxiliary_config, walker_control
    from controllers_v2 import ScenarioController, npc_target, select_action_profiles


def configuration():
    route = [[0., 0., 0., 0.], [100., 0., 0., 0.]]
    return {'vehicle_roles': ['ego', 'npc'],
        'vehicles': [{'role': role, 'route': route, 'initial_speed_mps': 5., 'desired_speed_mps': 8.}
                     for role in ('ego', 'npc')],
        'action': {'available_lane_change_directions': [], 'profiles': select_action_profiles([])},
        'geometry': {}, 'controller': {'leader_geometry': 'route_projection', 'lateral_margin_m': .3,
            'idm': {'max_acceleration': 2., 'comfortable_brake': 2.8, 'time_headway': 1.25, 'minimum_gap': 2.5},
            'conflict_time_gap_s': 2., 'yield_lookahead_s': 4.},
        'pedestrians': [{'role': 'crossing_pedestrian', 'blueprint_id': 'walker.pedestrian.0001',
            'transform': {'position': [12., -4., .8], 'rotation_deg': [0., 0., 90.]},
            'route_points': [[12., -4., 0.], [12., 4., 0.]], 'speed_mps': 1.4, 'start_time_s': 0.}],
        'traffic_signal_program': {'groups': [{'id': 'ego', 'landmark_ids': ['light0'],
             'roles': ['ego'], 'stop_route_s': {'ego': 12.}},
            {'id': 'npc', 'landmark_ids': ['light1'], 'roles': ['npc'], 'stop_route_s': {'npc': 12.}}],
            'phases': [{'start_time_s': -3., 'states': {'ego': 'Green', 'npc': 'Red'}},
                {'start_time_s': 0., 'states': {'ego': 'Yellow', 'npc': 'Red'}},
                {'start_time_s': 1., 'states': {'ego': 'Red', 'npc': 'Green'}}]}}


class FakeList(list):
    def filter(self, pattern):
        return [actor for actor in self if fnmatch.fnmatch(actor.type_id, pattern)]


class FakeLight:
    type_id = 'traffic.traffic_light'
    def __init__(self, identity, actor_id):
        self.identity, self.id = identity, actor_id
        self.current = self.pending = carla.TrafficLightState.Off
    def get_opendrive_id(self): return self.identity
    def set_state(self, value): self.pending = value
    def get_state(self): return self.current
    def freeze(self, value): self.frozen = value
    def get_transform(self): return carla.Transform(carla.Location(x=float(self.id)))


class FakeBlueprint:
    def __init__(self, identity): self.id, self.attrs = identity, {}
    def has_attribute(self, key): return True
    def set_attribute(self, key, value): self.attrs[key] = value


class FakeWalker:
    def __init__(self, blueprint, transform, identity):
        self.id, self.type_id, self.attributes = identity, blueprint.id, dict(blueprint.attrs)
        self.transform = transform
        self.current_control = self.pending_control = carla.WalkerControl()
        self.velocity = carla.Vector3D()
        self.bounding_box = carla.BoundingBox(carla.Location(z=.85), carla.Vector3D(.3, .3, .85))
        self.destroyed = False
    def get_transform(self): return self.transform
    def get_control(self): return self.current_control
    def get_velocity(self): return self.velocity
    def get_acceleration(self): return carla.Vector3D()
    def get_angular_velocity(self): return carla.Vector3D()
    def destroy(self): self.destroyed = True
    def step(self):
        self.current_control = self.pending_control
        self.velocity = self.current_control.direction*self.current_control.speed
        self.transform.location += self.velocity*.05


class FakeWorld:
    def __init__(self, lights=True, fail_spawn_at=None):
        self.lights = [FakeLight('light0', 10), FakeLight('light1', 11)] if lights else []
        self.walkers = []; self.vehicles = []; self.frame = 100; self.fail_spawn_at = fail_spawn_at
    def get_actors(self): return FakeList(self.lights + self.walkers)
    def get_blueprint_library(self): return SimpleNamespace(find=FakeBlueprint)
    def try_spawn_actor(self, blueprint, transform):
        if len(self.walkers) == self.fail_spawn_at: return None
        actor = FakeWalker(blueprint, transform, len(self.walkers)+1000)
        self.walkers.append(actor); return actor
    def tick(self, *unused):
        for light in self.lights: light.current = light.pending
        for walker in self.walkers: walker.step()
        for vehicle in self.vehicles: vehicle.current_control = vehicle.pending_control
        self.frame += 1
        return self.frame
    def get_snapshot(self): return SimpleNamespace(frame=self.frame, timestamp=SimpleNamespace(elapsed_seconds=self.frame*.05))


class FakeClient:
    def __init__(self, world): self.world = world
    def apply_batch_sync(self, commands, tick):
        assert tick is False
        for command in commands:
            actor = next(a for a in self.world.walkers+self.world.vehicles if a.id == command.actor_id)
            if isinstance(command, carla.command.ApplyWalkerControl): actor.pending_control = command.control
            elif isinstance(command, carla.command.ApplyVehicleControl): actor.pending_control = command.control
            elif isinstance(command, carla.command.ApplyTransform): actor.transform = command.transform
            else: raise AssertionError(type(command))
        return [SimpleNamespace(has_error=lambda: False, error='') for _ in commands]


def observations():
    return {'position': np.array([[0., 20., 0.], [0., 0., 0.]]),
            'rotation_deg': np.zeros((2, 3)), 'velocity': np.array([[5., 0., 0.], [5., 0., 0.]])}


class AuxiliaryTests(unittest.TestCase):
    def test_physics_recorder_integrates_actual_helper_clock(self):
        if __package__:
            from .preflight_physics import record_physics
        else:
            from preflight_physics import record_physics
        cfg = configuration(); world = FakeWorld(); client = FakeClient(world)
        runner = AuxiliaryActors(world, client, cfg)
        for i in range(2):
            actor = FakeWalker(FakeBlueprint('vehicle.fixture'), carla.Transform(carla.Location(y=20.*(1-i))), i)
            actor.current_control = actor.pending_control = carla.VehicleControl()
            actor.velocity = carla.Vector3D(x=5.)
            world.vehicles.append(actor)
        runner.set_signal_time(-2.); world.tick()
        extents = np.asarray([[2.,1.,1.], [2.,1.,1.]])
        values, diagnostics, base = record_physics(client, world, world.vehicles, cfg,
            'accelerate', ScenarioController(cfg), extents, auxiliary=runner)
        self.assertEqual(values['position'].shape, (121,2,3))
        self.assertEqual(len(runner.samples), 121)
        np.testing.assert_array_equal(values['frame_id'], [sample['frame_id'] for sample in runner.samples])
        self.assertEqual(runner.signal_samples[40]['actual_states']['light0'], 'Yellow')
        self.assertEqual(runner.samples[40]['observed_control'][0,3], 0.)
        self.assertGreater(runner.samples[41]['observed_control'][0,3], 0.)
        self.assertTrue(any(row.get('signal') for row in diagnostics[40]))

    def test_signal_and_walker_121_frame_timeline(self):
        cfg = configuration(); world = FakeWorld(); runner = AuxiliaryActors(world, FakeClient(world), cfg)
        runner.set_signal_time(-2.); world.tick()
        initial = runner.actors[0].get_transform().location.y
        for step in range(121):
            time = -2+step*.05
            runner.inject_observation(observations(), time)
            controls = runner.record(time, final=step == 120)
            if step < 120:
                runner.advance_commands(controls, time+.05); world.tick()
        with tempfile.TemporaryDirectory() as temporary:
            runner.save(temporary)
            states = np.load(Path(temporary)/'pedestrians.npz')
            self.assertEqual(states['position'].shape, (121, 1, 3))
            self.assertEqual(states['control'].shape, (121, 1, 5))
            np.testing.assert_allclose(states['position'][:41, 0, 1], initial)
            self.assertEqual(states['observed_control'][40, 0, 3], 0.)
            self.assertGreater(states['control'][40, 0, 3], 0.)
            self.assertGreater(states['observed_control'][41, 0, 3], 0.)
            np.testing.assert_allclose(states['control'][-1], 0.)
            signal = json.loads((Path(temporary)/'signal_states.json').read_text())
            self.assertEqual(len(signal['samples']), 121)
            self.assertEqual(signal['samples'][39]['actual_states']['light0'], 'Green')
            self.assertEqual(signal['samples'][40]['actual_states']['light0'], 'Yellow')
            self.assertEqual(signal['samples'][60]['actual_states']['light1'], 'Green')
            np.testing.assert_array_equal(states['frame_id'], [s['frame_id'] for s in signal['samples']])
            if __package__:
                from .audit_auxiliary import inspect_auxiliary
            else:
                from audit_auxiliary import inspect_auxiliary
            vehicle_clocks = {key: states[key] for key in ('frame_id', 'time_s', 'simulation_time_s')}
            meta = {**runner.metadata(), 'actor_ids': [1, 2], 'collisions': [],
                    'collision_sensor_roles': cfg['vehicle_roles']+[p['role'] for p in cfg['pedestrians']]}
            checked = inspect_auxiliary(temporary, cfg, meta, vehicle_clocks)
            self.assertEqual(checked['status'], 'passed', checked['errors'])
        runner.close(); self.assertTrue(world.walkers[0].destroyed)

    def test_signal_readback_not_relabelled_to_requested_state(self):
        world = FakeWorld(); runner = AuxiliaryActors(world, FakeClient(world), configuration()); world.tick()
        runner.set_signal_time(1.)
        with self.assertRaisesRegex(RuntimeError, 'readback'):
            runner.signal_observation(1.)  # no tick: actual light is still the previous phase
        world.tick()
        self.assertEqual(runner.signal_observation(1.)['actual_states']['light1'], 'Green')

    def test_missing_signal_asset_rejected_before_pedestrian_spawn(self):
        world = FakeWorld(lights=False)
        with self.assertRaisesRegex(RuntimeError, 'expected one real light'):
            AuxiliaryActors(world, FakeClient(world), configuration())
        self.assertEqual(world.walkers, [])

    def test_failed_second_spawn_cleans_first(self):
        cfg = configuration(); cfg['pedestrians'].append(copy.deepcopy(cfg['pedestrians'][0]))
        cfg['pedestrians'][1]['role'] = 'other_pedestrian'
        world = FakeWorld(fail_spawn_at=1)
        with self.assertRaisesRegex(RuntimeError, 'spawn failed'):
            AuxiliaryActors(world, FakeClient(world), cfg)
        self.assertTrue(world.walkers[0].destroyed)

    def test_unsupported_duplicate_roles_and_uncovered_history_rejected(self):
        for mutate in (lambda c: c['pedestrians'][0].update(role='ego'),
                       lambda c: c['pedestrians'][0].update(start_time_s=-3.),
                       lambda c: c['traffic_signal_program']['phases'][0].update(start_time_s=-1.)):
            cfg = configuration(); mutate(cfg)
            with self.assertRaises(ValueError): validate_auxiliary_config(cfg)

    def test_walker_holds_before_start_tracks_live_position_and_stops(self):
        spec = configuration()['pedestrians'][0]
        np.testing.assert_array_equal(walker_control([12., -4., 0.], spec, -.05, .05), 0.)
        control = walker_control([11.8, -3., 0.], spec, .5, .05)
        self.assertGreater(control[0], 0.); self.assertGreater(control[1], 0.)
        self.assertAlmostEqual(np.linalg.norm(control[:3]), 1.)
        np.testing.assert_array_equal(walker_control([12., 4., 0.], spec, 4., .05), 0.)

    def test_legacy_empty_auxiliary_writes_no_extra_files(self):
        cfg = configuration(); cfg.pop('pedestrians'); cfg.pop('traffic_signal_program')
        world = FakeWorld(); runner = AuxiliaryActors(world, FakeClient(world), cfg); world.tick()
        self.assertEqual(runner.inject_observation(observations(), -2.).keys(), observations().keys())
        runner.record(-2.)
        with tempfile.TemporaryDirectory() as temporary:
            runner.save(temporary); self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_red_signal_slows_npc_but_does_not_override_ego_action(self):
        cfg = configuration(); cfg.pop('pedestrians')
        obs = observations(); extents = np.array([[2., 1., 1.], [2., 1., 1.]])
        obs['signal_states'] = {'ego': 'Red', 'npc': 'Green'}
        free, _ = npc_target(1, obs, cfg, extents)
        control = ScenarioController(cfg)
        _, targets_green, _ = control.controls(obs, 'accelerate', 1., extents)
        obs['signal_states']['npc'] = 'Red'
        stopped, diagnostic = npc_target(1, obs, cfg, extents)
        _, targets_red, diagnostics = ScenarioController(cfg).controls(obs, 'accelerate', 1., extents)
        self.assertLess(stopped, free)
        self.assertEqual(diagnostic['reason'], 'stop_for_signal')
        self.assertEqual(targets_green[0], targets_red[0])
        self.assertFalse(diagnostics[0]['ego_action_overridden'])
        obs['position'][1, 0] = 14.
        _, diagnostic = npc_target(1, obs, cfg, extents)
        self.assertFalse(diagnostic['signal']['must_stop'])

    def test_pedestrian_response_uses_current_motion_not_future_plan(self):
        cfg = configuration(); cfg.pop('traffic_signal_program')
        obs = observations(); extents = np.array([[2., 1., 1.], [2., 1., 1.]])
        obs['pedestrians'] = {'position': np.array([[12., -4., 0.]]), 'velocity': np.zeros((1,3)),
                              'bbox_extent': np.array([[.3,.3,.85]]), 'roles': ['crossing_pedestrian']}
        free, _ = npc_target(1, obs, cfg, extents)
        obs['pedestrians']['velocity'][0,1] = 1.4
        crossing, diagnostic = npc_target(1, obs, cfg, extents)
        self.assertLess(crossing, free)
        self.assertIn('crossing_pedestrian', diagnostic['pedestrian_yield_roles'])
        cfg['pedestrians'][0]['route_points'] = [[-999.,0.,0.], [999.,0.,0.]]
        unchanged, _ = npc_target(1, obs, cfg, extents)
        self.assertEqual(unchanged, crossing)

    def test_narrow_yielding_releases_when_live_priority_clears(self):
        cfg = configuration(); cfg.pop('pedestrians'); cfg.pop('traffic_signal_program')
        cfg['geometry']['narrow_meeting'] = {'yield_roles': ['npc'], 'priority_roles': ['ego'],
            'yield_route_s': {'npc': [10.,30.]}, 'priority_route_s': {'ego': [10.,30.]},
            'stop_margin_m': 2., 'clearance_time_s': 3.}
        extents = np.array([[2.,1.,1.],[2.,1.,1.]])
        obs = observations(); obs['position'][0,0] = 15.
        constrained, diagnostic = npc_target(1, obs, cfg, extents)
        self.assertTrue(diagnostic['narrow_meeting']['yielding'])
        obs['position'][0,0] = 40.
        clear, diagnostic = npc_target(1, obs, cfg, extents)
        self.assertGreater(clear, constrained)
        self.assertFalse(diagnostic['narrow_meeting']['yielding'])
        obs['position'][0,0] = 15.; obs['position'][1,0] = 15.
        _, diagnostic = npc_target(1, obs, cfg, extents)
        self.assertEqual(diagnostic['narrow_meeting']['phase'], 'committed_clearance')
        self.assertFalse(diagnostic['narrow_meeting']['yielding'])


if __name__ == '__main__':
    unittest.main()
