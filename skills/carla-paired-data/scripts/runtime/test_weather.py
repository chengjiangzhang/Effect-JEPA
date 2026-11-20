"""Offline tests for frozen per-scene exposure and native night-light requests."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import carla

if __package__:
    from . import collect_multicam as capture
    from .controllers_v2 import select_action_profiles
else:
    import collect_multicam as capture
    from controllers_v2 import select_action_profiles


class WeatherRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.preset, _ = capture.load_render_preset(capture.RESOURCES/'daylight.json')

    def test_camera_overrides_apply_after_map_defaults(self):
        cfg = {'map': 'Town05', 'camera_override': {'iso': 400., 'fstop': 2.8}}
        effective = capture.scene_camera_attributes(self.preset, cfg)
        base = capture.preset_camera_attributes(self.preset, 'Town05')
        self.assertEqual(effective, {**base, 'iso': 400., 'fstop': 2.8})
        self.assertEqual(capture.preset_camera_attributes(self.preset, 'Town05'), base)
        self.assertEqual(capture.scene_camera_attributes(self.preset, {'map': 'Town05'}), base)

    def test_camera_override_rejects_geometry_timing_and_nonfinite_values(self):
        for override in ({'fov': 90.}, {'sensor_tick': .5}, {'image_size_x': 800},
                         {'exposure_mode': 'histogram'}, {'iso': 0.}, {'fstop': -1.},
                         {'shutter_speed': True}, {'exposure_compensation': float('nan')},
                         {'iso': float('inf')}, []):
            with self.subTest(override=override), self.assertRaises(ValueError):
                capture.scene_camera_attributes(self.preset, {'map': 'Town05', 'camera_override': override})

    def test_scene_exposure_requires_frozen_scene_policy(self):
        preset = copy.deepcopy(self.preset)
        preset.pop('scene_weather_policy')
        with self.assertRaises(ValueError):
            capture.scene_camera_attributes(preset, {'map': 'Town05', 'camera_override': {'iso': 400.}})

    def test_each_camera_blueprint_gets_scene_exposure_and_keeps_native_rig(self):
        """The blueprint factory shared by prime and collection uses the same recipe."""
        cfg = {'map': 'Town10HD', 'camera_override': {
            'iso': 400., 'fstop': 2.8, 'shutter_speed': 100., 'exposure_compensation': 3.}}
        render = capture.scene_camera_attributes(self.preset, cfg)
        for name, _, _, _, _, fov in capture.RIG:
            attributes = {}
            blueprint = SimpleNamespace(has_attribute=lambda key: True,
                                        set_attribute=attributes.__setitem__)
            library = SimpleNamespace(find=lambda identifier: blueprint)
            _, recorded = capture.camera_blueprint(library, name, 1600, 900, fov, render)
            self.assertEqual(recorded, attributes)
            for key, value in render.items():
                self.assertEqual(recorded[key], str(value))
            self.assertEqual(recorded['role_name'], name)
            self.assertEqual(recorded['image_size_x'], '1600')
            self.assertEqual(recorded['image_size_y'], '900')
            self.assertEqual(recorded['fov'], str(fov))
            self.assertEqual(recorded['sensor_tick'], '0.0')

    def test_scene_exposure_attribute_cannot_be_silently_skipped(self):
        blueprint = SimpleNamespace(has_attribute=lambda key: key != 'iso',
                                    set_attribute=lambda key, value: None)
        library = SimpleNamespace(find=lambda identifier: blueprint)
        with self.assertRaisesRegex(RuntimeError, 'Required sensor attribute missing: iso'):
            capture.camera_blueprint(library, 'CAM_FRONT', 1600, 900, 70., {'iso': 400.})

    def test_night_scene_finalization_and_tamper_checks(self):
        cfg = {'scene_id': 'scene_night', 'map': 'Town05', 'seed': 2, 'family': 'rear_following',
               'weather_override': {'sun_altitude_angle': -20.},
               'camera_override': {'iso': 400., 'fstop': 2.8, 'shutter_speed': 100.},
               'vehicle_light_mode': 'low_beam',
               'vehicles': [{'role': 'ego', 'blueprint_id': 'vehicle.tesla.model3',
                             'route': [[0., 0., 0., 0.], [20., 0., 0., 0.]]}],
               'action': {'available_lane_change_directions': [], 'profiles': select_action_profiles([])}}
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            frozen = capture.finalize_scene(cfg, data=data)
            self.assertEqual(frozen['weather']['sun_altitude_angle'], -20.)
            self.assertEqual(frozen['render']['camera_attributes']['iso'], 400.)
            self.assertEqual(frozen['render']['vehicle_light_mode'], 'low_beam')
            self.assertEqual(json.loads((data/'scene_night/spec.json').read_text()), frozen)
            capture.check_frozen_render_preset(data, frozen)
            bad_camera = copy.deepcopy(frozen)
            bad_camera['render']['camera_attributes']['iso'] = 100.
            with self.assertRaises(ValueError):
                capture.check_frozen_render_preset(data, bad_camera)
            bad_lights = copy.deepcopy(frozen)
            bad_lights['render']['vehicle_light_mode'] = 'default'
            with self.assertRaises(ValueError):
                capture.check_frozen_render_preset(data, bad_lights)

    def test_light_mode_rejects_unknown_names(self):
        for mode in ('high_beam', None, True, 1):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                capture.vehicle_light_mode({'vehicle_light_mode': mode})

    def test_night_lights_request_all_vehicles_before_tick(self):
        actors = [SimpleNamespace(id=101), SimpleNamespace(id=102)]
        with patch.object(capture, 'apply_commands') as apply:
            capture.set_vehicle_lights('client', actors, {'vehicle_light_mode': 'low_beam'})
            client, commands = apply.call_args.args
            self.assertEqual(client, 'client')
            self.assertEqual([command.actor_id for command in commands], [101, 102])
            self.assertEqual([int(command.light_state) for command in commands], [3, 3])

    def test_night_lights_verify_readback_and_permit_brake_bits(self):
        actor = SimpleNamespace(id=1, get_light_state=lambda: carla.VehicleLightState(3 | 8))
        report = capture.checked_vehicle_lights([actor], ['ego'], {'vehicle_light_mode': 'low_beam'})
        self.assertEqual(report['status'], 'verified')
        self.assertEqual(report['vehicles'][0]['observed_mask'], 11)
        actor.get_light_state = lambda: carla.VehicleLightState.Position
        with self.assertRaisesRegex(RuntimeError, 'night vehicle lights not active'):
            capture.checked_vehicle_lights([actor], ['ego'], {'vehicle_light_mode': 'low_beam'})

    def test_night_lights_failed_command_ack_aborts(self):
        response = SimpleNamespace(has_error=lambda: True, error='actor unavailable')
        client = SimpleNamespace(apply_batch_sync=lambda commands, tick: [response])
        with self.assertRaisesRegex(RuntimeError, 'actor unavailable'):
            capture.set_vehicle_lights(client, [SimpleNamespace(id=1)],
                                       {'vehicle_light_mode': 'low_beam'})

    def test_night_lights_incomplete_ack_or_role_mapping_aborts(self):
        client = SimpleNamespace(apply_batch_sync=lambda commands, tick: [])
        with self.assertRaisesRegex(RuntimeError, 'incomplete response'):
            capture.set_vehicle_lights(client, [SimpleNamespace(id=1)],
                                       {'vehicle_light_mode': 'low_beam'})
        with self.assertRaisesRegex(ValueError, 'roles do not match actors'):
            capture.checked_vehicle_lights([SimpleNamespace(id=1)], [],
                                           {'vehicle_light_mode': 'low_beam'})

    def test_default_preserves_existing_light_behavior(self):
        with patch.object(capture, 'apply_commands') as apply:
            capture.set_vehicle_lights('client', [SimpleNamespace(id=1)], {})
            apply.assert_not_called()
        self.assertEqual(capture.checked_vehicle_lights([], [], {})['status'], 'native_default_not_overridden')


if __name__ == '__main__':
    unittest.main()
