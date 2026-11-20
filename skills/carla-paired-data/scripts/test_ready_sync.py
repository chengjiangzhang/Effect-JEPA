"""Offline startup ordering/readback checks. No CARLA process or client is used."""
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import copy
import json
import sys
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import managed_server as ms
from runtime import collect_multicam as cm


class World:
    def __init__(self, name='Town01', identifier=1, events=None, synchronous=False):
        self.events = events if events is not None else []
        self.name, self.id, self.frame = name, identifier, 37
        self.settings = SimpleNamespace(synchronous_mode=synchronous,
            fixed_delta_seconds=.05 if synchronous else None, substepping=True,
            max_substep_delta_time=.01, max_substeps=10, no_rendering_mode=False)
        self.ignore_apply = False

    def get_settings(self):
        return copy.copy(self.settings)

    def apply_settings(self, settings):
        self.events.append(('apply_settings', self.id))
        if not self.ignore_apply:
            self.settings = copy.copy(settings)
        return self.frame

    def get_snapshot(self):
        return SimpleNamespace(frame=self.frame,
            timestamp=SimpleNamespace(elapsed_seconds=1.85, delta_seconds=.05))

    def get_map(self):
        return SimpleNamespace(name='/Game/Carla/Maps/'+self.name)

    def tick(self, *args, **kwargs):
        raise AssertionError('Startup candidate must not pad/tick frames')


def fake_client(world):
    client = Mock()
    client.get_client_version.return_value = '0.9.16'
    client.get_server_version.return_value = '0.9.16'
    client.get_world.return_value = world
    def maps():
        world.events.append(('available_maps', world.id))
        if not world.settings.synchronous_mode:
            raise AssertionError('Map/audit preparation began before readiness synchronization')
        return ['/Game/Carla/Maps/Town05']
    client.get_available_maps.side_effect = maps
    return client


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manager = ms.ManagedServer.__new__(ms.ManagedServer)
        self.manager.host, self.manager.port, self.manager.maps = '127.0.0.1', 2000, ['Town05']
        self.manager._current = {'log_dir': self.temp.name}
        self.manager._child = Mock(returncode=None)
        self.manager._child.poll.return_value = None
        self.manager._refresh_owned = Mock()

    def test_ready_freezes_before_map_query_and_preserves_actual_clock(self):
        world = World()
        with patch.object(ms, '_client', return_value=fake_client(world)):
            result = self.manager._wait_ready()
        self.assertEqual(world.events, [('apply_settings', 1), ('available_maps', 1)])
        row = result['startup_world_readback'][0]
        self.assertFalse(row['before_sync']['settings']['synchronous_mode'])
        self.assertTrue(row['after_sync']['settings']['synchronous_mode'])
        self.assertEqual(row['before_sync']['frame'], 37)
        self.assertEqual(row['after_sync']['frame'], 37)
        self.assertEqual(row['after_sync']['elapsed_seconds'], 1.85)
        saved = json.loads((Path(self.temp.name)/'startup_world_readback.json').read_text())
        self.assertEqual(saved['attempts'], result['startup_world_readback'])

    def test_settings_not_applied_is_infrastructure_failure_before_maps(self):
        world = World(); world.ignore_apply = True
        client = fake_client(world)
        with patch.object(ms, '_client', return_value=client):
            with self.assertRaisesRegex(ms.ManagedServerError, 'did not retain'):
                self.manager._wait_ready()
        client.get_available_maps.assert_not_called()
        row = json.loads((Path(self.temp.name)/'startup_world_readback.json').read_text())['attempts'][0]
        self.assertEqual(row['status'], 'failed')
        self.assertFalse(row['after_sync']['settings']['synchronous_mode'])

    def test_wrong_native_version_never_changes_world(self):
        world = World(); client = fake_client(world)
        client.get_server_version.return_value = '0.9.15'
        with patch.object(ms, '_client', return_value=client):
            with self.assertRaisesRegex(ms.ManagedServerError, '0.9.16 required'):
                self.manager._wait_ready()
        client.get_world.assert_not_called()

    def test_incorrect_fixed_delta_readback_fails(self):
        world = World(synchronous=True)
        row = ms._world_readback(world)
        self.assertTrue(ms._ready_sync_matches(row))
        for key, value in [('fixed_delta_seconds', .1), ('substepping', False),
                           ('max_substep_delta_time', .02), ('max_substeps', 2),
                           ('no_rendering_mode', True)]:
            bad = copy.deepcopy(row); bad['settings'][key] = value
            self.assertFalse(ms._ready_sync_matches(bad), key)

    def initialize(self, *, target='Town05', preserve=False, reset_by_map=False):
        events = []; incoming = World(events=events)
        selected = World(target, 2, events, synchronous=not reset_by_map)
        reloaded = World(target, 3, events, synchronous=True)
        client = Mock(); client.get_world.return_value = incoming
        def load(name, reset_settings):
            self.assertTrue(incoming.settings.synchronous_mode)
            self.assertFalse(reset_settings)
            events.append(('load_world', name, reset_settings))
            return selected
        client.load_world.side_effect = load
        client.reload_world.return_value = reloaded
        with patch.object(cm, 'preserve_static_map_requested', return_value=preserve), \
                patch.object(cm, 'check_no_capture_actors', return_value={'empty':True}):
            world, meta = cm.initialize_branch_world(client, {'map': target})
        return world, meta, client, events

    def test_map_switch_preserves_settings_and_retains_legacy_reload(self):
        world, meta, client, events = self.initialize()
        self.assertEqual(events[0], ('apply_settings', 1))
        client.load_world.assert_called_once_with('Town05', reset_settings=False)
        client.reload_world.assert_called_once_with(False)
        self.assertEqual(world.id, 3)
        self.assertEqual([row['world_id'] for row in meta['initialization_readbacks']], [1,1,2,3])
        self.assertEqual(meta['policy'], 'reload_world_false')

    def test_same_map_has_no_new_map_load_or_frame_padding(self):
        world, meta, client, events = self.initialize(target='Town01')
        client.load_world.assert_not_called()
        client.reload_world.assert_called_once_with(False)
        self.assertEqual([row['frame'] for row in meta['initialization_readbacks']], [37]*4)

    def test_preserve_diagnostic_still_skips_reload(self):
        world, meta, client, events = self.initialize(preserve=True)
        client.reload_world.assert_not_called()
        self.assertTrue(meta['initial_reload_skipped'])
        self.assertEqual(world.id, 2)

    def test_map_reset_settings_loss_fails_before_reload_or_collection(self):
        with self.assertRaisesRegex(RuntimeError, 'lost synchronous'):
            self.initialize(reset_by_map=True)

    def test_collector_readback_keeps_settings_and_actual_clocks(self):
        world = World(synchronous=True)
        row = cm.checked_initialization_readback(world, 'test')
        self.assertEqual(row['frame'], 37)
        self.assertEqual(row['elapsed_seconds'], 1.85)
        world.settings.fixed_delta_seconds = .1
        with self.assertRaisesRegex(RuntimeError, 'lost synchronous'):
            cm.checked_initialization_readback(world, 'test')


if __name__ == '__main__':
    unittest.main()
