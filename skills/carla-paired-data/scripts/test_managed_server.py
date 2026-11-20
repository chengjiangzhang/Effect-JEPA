"""Owned-server lifecycle checks; no simulator, sockets or processes are started."""
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import managed_server as ms


class ManagedServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'carla').mkdir()
        executable = self.root / 'carla/CarlaUE4.sh'
        executable.write_text('#!/bin/sh\n')
        executable.chmod(0o755)
        self.ini = self.root / 'epic.ini'
        self.ini.write_text('[SystemSettings]\n')
        self.config = {'carla_root': str(self.root / 'carla'), 'user': 'test_carla',
                       'log_root': str(self.root / 'logs'), 'engine_ini': str(self.ini)}
        self.path = self.root / 'server.json'
        self.path.write_text(json.dumps(self.config))
        self.user_patch = patch.object(ms.pwd, 'getpwnam', return_value=SimpleNamespace(pw_uid=1000))
        self.user_patch.start()
        self.addCleanup(self.user_patch.stop)
        self.uid_patch = patch.object(ms.os, 'getuid', return_value=0)
        self.uid_patch.start()
        self.native_patch = patch.object(ms.weather_proof, 'launch_proof', return_value={'test_mock': True})
        self.native_patch.start()
        self.addCleanup(self.native_patch.stop)
        self.addCleanup(self.uid_patch.stop)

    def manager(self, **kwargs):
        return ms.ManagedServer(self.path, self.root / 'batch', kwargs.get('maps', ['Town05']),
                                kwargs.get('host', '127.0.0.1'), kwargs.get('port', 2000))

    def fake_launch(self, command, **kwargs):
        pid = 500 + len(self.children)
        child = Mock(pid=pid, returncode=None)
        child.poll.side_effect = lambda: child.returncode
        child.wait.side_effect = lambda: child.returncode
        child.log_dir = Path(command[command.index('--log-dir') + 1])
        self.children[pid] = child
        return child

    def fake_record(self, pid):
        child = self.children.get(pid)
        if child is None or child.returncode is not None:
            return None
        return {'pid': pid, 'ppid': 100, 'pgid': pid, 'state': 'S', 'start_ticks': pid + 1000}

    def fake_signal(self, pid, sig):
        self.assertEqual(sig, signal.SIGTERM)
        child = self.children[pid]
        child.returncode = 241
        (child.log_dir / 'shutdown.json').write_text(json.dumps({'engine_restored': True,
            'stop_requested': True, 'returncode': -15}))

    def lifecycle_mocks(self):
        self.children = {}
        patches = [
            patch.object(ms.subprocess, 'Popen', side_effect=self.fake_launch),
            patch.object(ms, '_record', side_effect=self.fake_record),
            patch.object(ms, '_owned_processes', return_value={}),
            patch.object(ms, '_port_open', return_value=False),
            patch.object(ms.os, 'kill', side_effect=self.fake_signal),
            patch.object(ms.ManagedServer, '_wait_ready', return_value={
                'client': '0.9.16', 'server': '0.9.16', 'available_maps': ['/Game/Carla/Maps/Town05']}),
        ]
        mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)
        return mocks

    def test_recycles_only_owned_launcher_and_preserves_numbered_logs(self):
        popen, _, _, _, kill, _ = self.lifecycle_mocks()
        with self.manager() as manager:
            first = manager.before_attempt('scene_0000', 0, 0)
            first_log = Path(first['log_dir'])
            (first_log / 'server.log').write_text('preserve previous engine output')
            second = manager.before_attempt('scene_0001', 0, 0)
            self.assertEqual(first['epoch'], 1)
            self.assertEqual(second['epoch'], 2)
            self.assertEqual(kill.call_args_list[0].args, (500, signal.SIGTERM))
            self.assertEqual(self.children[500].returncode, 241)
            self.assertEqual(first['status'], 'ready')  # Returned metadata is not mutated by stop.
            self.assertEqual(json.loads((first_log / 'managed.json').read_text())['status'], 'stopped')
            self.assertEqual((first_log / 'server.log').read_text(), 'preserve previous engine output')
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index('--user') + 1], 'test_carla')
            self.assertNotIn('--fixed-seed', command)
            self.assertNotIn('--startup-benchmark', command)
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            self.assertEqual(popen.call_args.kwargs['stdin'], ms.subprocess.DEVNULL)
        self.assertEqual(kill.call_args_list[-1].args, (501, signal.SIGTERM))
        manager.close()  # Idempotent; no repeated signal.
        self.assertEqual(kill.call_count, 2)
        with self.manager() as resumed:
            third = resumed.before_attempt('scene_0002', 0, 0)
            self.assertEqual(third['epoch'], 3)

    def test_busy_port_never_adopts_or_kills_external_server(self):
        with self.manager() as manager, patch.object(ms, '_port_open', return_value=True), \
                patch.object(ms.subprocess, 'Popen') as popen, patch.object(ms.os, 'kill') as kill:
            with self.assertRaisesRegex(ms.ManagedServerError, 'occupied'):
                manager.before_attempt('scene_0000', 0, 0)
            popen.assert_not_called()
            kill.assert_not_called()

    def test_changed_ini_stops_before_new_launch(self):
        with self.manager() as manager, patch.object(ms.subprocess, 'Popen') as popen:
            self.ini.write_text('changed after manager configuration')
            with self.assertRaisesRegex(ms.ManagedServerError, 'changed'):
                manager.before_attempt('scene_0000', 0, 0)
            popen.assert_not_called()

    def test_reused_pid_is_never_signalled(self):
        _, _, _, _, kill, _ = self.lifecycle_mocks()
        manager = self.manager()
        manager.before_attempt('scene_0000', 0, 0)
        with patch.object(ms, '_same_process', return_value=False):
            with self.assertRaisesRegex(ms.ManagedServerError, 'identity changed'):
                manager.before_attempt('scene_0001', 0, 0)
            kill.assert_not_called()
        manager.close()

    def test_unclean_engine_restoration_prevents_second_launch(self):
        popen, _, _, _, _, _ = self.lifecycle_mocks()
        manager = self.manager()
        first = manager.before_attempt('scene_0000', 0, 0)
        def bad_shutdown(pid, sig):
            self.children[pid].returncode = 241
            (self.children[pid].log_dir / 'shutdown.json').write_text('{"engine_restored": false}')
        with patch.object(ms.os, 'kill', side_effect=bad_shutdown):
            with self.assertRaisesRegex(ms.ManagedServerError, 'cleanup unverified'):
                manager.before_attempt('scene_0001', 0, 0)
        self.assertEqual(popen.call_count, 1)
        (Path(first['log_dir']) / 'shutdown.json').write_text('{"engine_restored": true}')
        manager.close()

    def test_surviving_engine_prevents_second_launch(self):
        popen, _, _, _, _, _ = self.lifecycle_mocks()
        manager = self.manager()
        manager.before_attempt('scene_0000', 0, 0)
        survivor = {'pid': 600, 'ppid': 1, 'pgid': 600, 'start_ticks': 1600, 'state': 'S'}
        with patch.object(ms, '_owned_processes', return_value={600: survivor}), \
                patch.object(ms, '_same_process', return_value=True), \
                patch.object(ms, 'ENGINE_EXIT_TIMEOUT_S', 0.):
            with self.assertRaisesRegex(ms.ManagedServerError, 'remaining=\\[600\\]'):
                manager.before_attempt('scene_0001', 0, 0)
        self.assertEqual(popen.call_count, 1)
        manager.close()

    def test_missing_config_is_infrastructure_failure_and_never_creates_batch(self):
        with self.manager() as manager, patch.object(ms.subprocess, 'Popen') as popen:
            self.assertFalse((self.root / 'batch').exists())
            self.path.unlink()
            with self.assertRaisesRegex(ms.ManagedServerError, 'infrastructure failure'):
                manager.before_attempt('scene_0000', 0, 0)
            popen.assert_not_called()

    def test_readiness_failure_cleans_up_owned_launcher(self):
        _, _, _, _, kill, ready = self.lifecycle_mocks()
        ready.side_effect = ms.ManagedServerError('readiness deadline')
        with self.manager() as manager:
            with self.assertRaisesRegex(ms.ManagedServerError, 'readiness deadline'):
                manager.before_attempt('scene_0000', 0, 0)
            self.assertIsNone(manager._child)
        self.assertEqual(kill.call_count, 1)

    def test_transient_port_occupancy_after_exit_waits_then_allows_restart(self):
        popen, _, _, _, kill, _ = self.lifecycle_mocks()
        with self.manager() as manager:
            manager.before_attempt('scene_0000', 0, 0)
            # The port briefly remains reachable after the owned engine exits.
            # False #1 completes cleanup; false #2 is the next startup check.
            with patch.object(ms, '_port_open', side_effect=[True, True, False, False]), \
                    patch.object(ms.time, 'sleep') as sleep:
                second = manager.before_attempt('scene_0001', 0, 0)
                self.assertEqual(sleep.call_count, 2)
                self.assertEqual(second['epoch'], 2)
            self.assertEqual(popen.call_count, 2)
            self.assertEqual(kill.call_args_list[0].args, (500, signal.SIGTERM))

    def test_persistent_port_occupancy_prevents_restart_without_port_owner_kill(self):
        popen, _, _, _, kill, _ = self.lifecycle_mocks()
        manager = self.manager()
        manager.before_attempt('scene_0000', 0, 0)
        with patch.object(ms, '_port_open', return_value=True), \
                patch.object(ms, 'PORT_RELEASE_TIMEOUT_S', 0.):
            with self.assertRaisesRegex(ms.ManagedServerError, 'port remains occupied'):
                manager.before_attempt('scene_0001', 0, 0)
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(kill.call_count, 1)
        self.assertEqual(kill.call_args.args, (500, signal.SIGTERM))
        manager.close()
        self.assertEqual(kill.call_count, 1)

    def test_strict_config_rejects_commands_remote_hosts_and_root_account(self):
        self.path.write_text(json.dumps({**self.config, 'command': 'killall CarlaUE4'}))
        with self.assertRaisesRegex(ms.ManagedServerError, 'Config keys'):
            self.manager()
        self.path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ms.ManagedServerError, 'loopback'):
            self.manager(host='remote.example')
        with patch.object(ms.pwd, 'getpwnam', return_value=SimpleNamespace(pw_uid=0)):
            with self.assertRaisesRegex(ms.ManagedServerError, 'non-root'):
                self.manager()

    def test_second_manager_cannot_acquire_same_log_root(self):
        with self.manager():
            with self.assertRaisesRegex(ms.ManagedServerError, 'setup'):
                self.manager()

    def test_fixed_seed_boolean_is_forwarded_explicitly(self):
        popen, _, _, _, _, _ = self.lifecycle_mocks()
        self.path.write_text(json.dumps({**self.config, 'fixed_seed': True}))
        with self.manager() as manager:
            metadata = manager.before_attempt('scene_0000', 0, 0)
            self.assertEqual(popen.call_args.args[0].count('--fixed-seed'), 1)
            self.assertIs(metadata['fixed_seed'], True)
        for invalid in (1, 0, 'true', None):
            with self.subTest(fixed_seed=invalid):
                self.path.write_text(json.dumps({**self.config, 'fixed_seed': invalid}))
                with self.assertRaisesRegex(ms.ManagedServerError, 'fixed_seed must be a boolean'):
                    self.manager()

    def test_startup_benchmark_is_explicit_boolean_and_part_of_frozen_identity(self):
        popen, _, _, _, _, _ = self.lifecycle_mocks()
        with self.manager() as default:
            original_identity = default.config_sha256
        self.path.write_text(json.dumps({**self.config, 'startup_benchmark': True}))
        with self.manager() as manager:
            self.assertNotEqual(manager.config_sha256, original_identity)
            metadata = manager.before_attempt('scene_0000', 0, 0)
            self.assertEqual(popen.call_args.args[0].count('--startup-benchmark'), 1)
            self.assertIs(metadata['startup_benchmark'], True)
            self.assertEqual(metadata['startup_fixed_delta_seconds'], .05)
        for invalid in (1, 0, 'true', None):
            with self.subTest(startup_benchmark=invalid):
                self.path.write_text(json.dumps({**self.config, 'startup_benchmark': invalid}))
                with self.assertRaisesRegex(ms.ManagedServerError, 'startup_benchmark must be a boolean'):
                    self.manager()

    def test_readiness_checks_native_versions_and_exact_map_names(self):
        manager = self.manager()
        manager._child = Mock(returncode=None)
        manager._child.poll.return_value = None
        log_dir = self.root / 'logs/server_000001'
        log_dir.mkdir()
        manager._current = {'log_dir': str(log_dir)}
        world = Mock(id=1)
        world.get_settings.return_value = SimpleNamespace(
            synchronous_mode=True, fixed_delta_seconds=.05, substepping=True,
            max_substep_delta_time=.01, max_substeps=10, no_rendering_mode=False)
        world.get_snapshot.return_value = SimpleNamespace(frame=42,
            timestamp=SimpleNamespace(elapsed_seconds=2.1, delta_seconds=.05))
        world.get_map.return_value = SimpleNamespace(name='/Game/Carla/Maps/Town05')
        world.apply_settings.return_value = 42
        client = Mock()
        client.get_world.return_value = world
        client.get_client_version.return_value = '0.9.16'
        client.get_server_version.return_value = '0.9.16'
        client.get_available_maps.return_value = ['/Game/Carla/Maps/Town05_Opt']
        with patch.object(manager, '_refresh_owned'), patch.object(ms, '_client', return_value=client):
            with self.assertRaisesRegex(ms.ManagedServerError, 'lacks rendered maps'):
                manager._wait_ready()
            client.get_available_maps.return_value = ['/Game/Carla/Maps/Town05']
            self.assertEqual(manager._wait_ready()['server'], '0.9.16')
            client.get_server_version.return_value = '0.9.15'
            with self.assertRaisesRegex(ms.ManagedServerError, '0.9.16 required'):
                manager._wait_ready()
        manager._child = None
        manager.close()

    def test_readiness_has_bounded_deadline(self):
        manager = self.manager()
        manager._child = Mock(returncode=None)
        manager._child.poll.return_value = None
        manager._current = {'log_dir': str(self.root / 'logs/server_000001')}
        with patch.object(manager, '_refresh_owned'), \
                patch.object(ms.time, 'monotonic', side_effect=[0., 91.]):
            with self.assertRaisesRegex(ms.ManagedServerError, 'within 90s'):
                manager._wait_ready()
        manager._child = None
        manager.close()


if __name__ == '__main__':
    unittest.main()
