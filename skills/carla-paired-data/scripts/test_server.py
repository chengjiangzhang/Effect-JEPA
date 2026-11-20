"""Launcher lifecycle tests with tiny fake executables; never starts CARLA."""
import json
import fcntl
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import server


class ServerTests(unittest.TestCase):
    def setUp(self):
        # Lifecycle fixtures deliberately contain a tiny Python executable, not
        # CARLA. Native asset failures are exercised separately against the real
        # verifier with a synthetic manifest and byte hashes.
        from native_weather_proof import package_bundle
        mock = patch.object(server, 'verify_bundle', return_value=package_bundle())
        self.verifier = mock.start()
        self.addCleanup(mock.stop)
        prepare = patch.object(server.dfao_renderer, 'prepare', side_effect=lambda stock, logs: stock)
        prepare.start()
        self.addCleanup(prepare.stop)
        watcher = patch.object(server.dfao_renderer, 'CacheRead')
        watcher.start().return_value.observe.return_value = True
        self.addCleanup(watcher.stop)

    def setup_fixture(self, temporary, body):
        base=Path(temporary); home=base/'home';home.mkdir()
        root=base/'carla';root.mkdir()
        executable=root/'CarlaUE4.sh'
        executable.write_text('#!'+sys.executable+'\n'+body)
        executable.chmod(0o755)
        requested=base/'requested.ini';requested.write_text('[SystemSettings]\nr.ShadowQuality=5\n')
        ini=home/'.config/Epic/CarlaUE4/Saved/Config/LinuxNoEditor/Engine.ini'
        ini.parent.mkdir(parents=True);ini.write_text('original settings\n');ini.chmod(0o640)
        account=SimpleNamespace(pw_uid=1000,pw_gid=1000,pw_name='fake_carla',pw_dir=str(home))
        argv=['server.py','--carla-root',str(root),'--log-dir',str(base/'logs'),
              '--engine-ini',str(requested),'--stop-timeout','.3']
        return base,account,argv,ini

    def run_main(self, account, argv):
        with patch.object(server.os,'getuid',return_value=1000),patch.object(server.pwd,'getpwuid',return_value=account),patch.object(sys,'argv',argv):
            return server.main()

    def test_environment_matches_native_graphics_launcher(self):
        a=SimpleNamespace(pw_dir='/home/example',pw_name='example')
        env=server.server_environment(a,{'PATH':'/miniconda/bin','LD_PRELOAD':'bad.so','PYTHONPATH':'/tmp/unsafe',
            'LD_LIBRARY_PATH':'/usr/lib/gpu:/opt/miniconda/lib:/opt/miniforge3/lib::/usr/lib64',
            'VK_ICD_FILENAMES':'/etc/vulkan/nvidia.json','API_KEY':'do-not-copy'})
        self.assertEqual(env['PATH'],server.SYSTEM_PATH)
        self.assertEqual(env['LD_LIBRARY_PATH'],'/usr/lib/gpu:/usr/lib64')
        self.assertEqual(env['VK_ICD_FILENAMES'],'/etc/vulkan/nvidia.json')
        self.assertEqual(env['TMP'],env['TMPDIR']);self.assertEqual(env['TEMP'],env['TMPDIR'])
        self.assertEqual(env['CARLA_CACHE_DIR'],'/home/example/.cache/carla')
        for key in ('LD_PRELOAD','PYTHONPATH','API_KEY'):self.assertNotIn(key,env)

    def test_inaccessible_ancestor_identifies_actual_execution_account(self):
        account=SimpleNamespace(pw_name='engine_user',pw_dir='/home/engine')
        def probe(command,**kwargs):
            return SimpleNamespace(returncode=1 if command[-1]=='/root' else 0)
        with patch.object(server.os,'getuid',return_value=0),patch.object(server.shutil,'which',return_value='/usr/sbin/runuser'),patch.object(server.subprocess,'run',side_effect=probe) as run:
            with self.assertRaisesRegex(ValueError,'engine_user cannot traverse directory /root'):
                server.check_account_access(account,Path('/root/carla'))
            self.assertEqual(run.call_args.args[0][:5],['/usr/sbin/runuser','-u','engine_user','--','/usr/bin/test'])

    def test_normal_exit_restores_ini_bytes_and_permissions(self):
        with tempfile.TemporaryDirectory() as td:
            base,account,argv,ini=self.setup_fixture(td,'import os,json\nprint(json.dumps(dict(os.environ)))\n')
            before=ini.stat()
            self.assertEqual(self.run_main(account,argv),0)
            self.assertEqual(ini.read_text(),'original settings\n')
            self.assertEqual(ini.stat().st_mode & 0o777,0o640)
            self.assertEqual(ini.stat().st_mtime_ns,before.st_mtime_ns)
            child_env=json.loads((base/'logs/server.log').read_text())
            self.assertEqual(child_env['HOME'],account.pw_dir)
            self.assertEqual((Path(account.pw_dir)/'xdg-runtime').stat().st_mode & 0o777,0o700)
            self.assertTrue(json.loads((base/'logs/shutdown.json').read_text())['engine_restored'])
            launch=json.loads((base/'logs/launch.json').read_text())
            self.assertFalse(launch['fixed_seed'])
            self.assertNotIn('-FixedSeed',launch['command'])
            self.assertFalse(launch['startup_benchmark'])
            self.assertIsNone(launch['startup_fixed_delta_seconds'])
            self.assertNotIn('-benchmark',launch['command'])
            self.assertNotIn('FPS=20',launch['command'])

    def test_fixed_seed_reaches_engine_and_launch_record(self):
        with tempfile.TemporaryDirectory() as td:
            base,account,argv,ini=self.setup_fixture(td,'import sys,json\nprint(json.dumps(sys.argv[1:]))\n')
            self.assertEqual(self.run_main(account,[*argv,'--fixed-seed']),0)
            child_args=json.loads((base/'logs/server.log').read_text())
            self.assertEqual(child_args.count('-FixedSeed'),1)
            launch=json.loads((base/'logs/launch.json').read_text())
            self.assertTrue(launch['fixed_seed'])
            self.assertEqual(launch['command'][1:],child_args)
            self.assertIn('-quality-level=Epic',child_args)
            self.assertEqual(ini.read_text(),'original settings\n')

    def test_startup_benchmark_reaches_engine_and_records_fixed_timestep(self):
        with tempfile.TemporaryDirectory() as td:
            base,account,argv,ini=self.setup_fixture(td,'import sys,json\nprint(json.dumps(sys.argv[1:]))\n')
            self.assertEqual(self.run_main(account,[*argv,'--fixed-seed','--startup-benchmark']),0)
            child_args=json.loads((base/'logs/server.log').read_text())
            self.assertEqual(child_args.count('-benchmark'),1)
            self.assertEqual(child_args.count('FPS=20'),1)
            self.assertEqual(child_args.count('-FixedSeed'),1)
            launch=json.loads((base/'logs/launch.json').read_text())
            self.assertTrue(launch['startup_benchmark'])
            self.assertEqual(launch['startup_fixed_delta_seconds'],.05)
            self.assertEqual(launch['command'][1:],child_args)
            self.assertEqual(ini.read_text(),'original settings\n')

    def test_launch_error_restores_ini(self):
        with tempfile.TemporaryDirectory() as td:
            base,account,argv,ini=self.setup_fixture(td,'pass\n')
            with patch.object(server.subprocess,'Popen',side_effect=OSError('simulated spawn failure')):
                with self.assertRaisesRegex(OSError,'simulated spawn failure'):self.run_main(account,argv)
            self.assertEqual(ini.read_text(),'original settings\n')
            self.assertTrue(json.loads((base/'logs/shutdown.json').read_text())['engine_restored'])

    def test_account_lock_prevents_concurrent_engine_override(self):
        with tempfile.TemporaryDirectory() as td:
            base,account,argv,ini=self.setup_fixture(td,'pass\n')
            runtime=Path(account.pw_dir)/'xdg-runtime';runtime.mkdir()
            with (runtime/'carla-engine.lock').open('a+b') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError,'Another CARLA launch owns'):
                    self.run_main(account,argv)
            self.assertEqual(ini.read_text(),'original settings\n')

    def test_absent_original_ini_is_removed_after_exit(self):
        with tempfile.TemporaryDirectory() as td:
            base,account,argv,ini=self.setup_fixture(td,'pass\n');ini.unlink()
            self.assertEqual(self.run_main(account,argv),0)
            self.assertFalse(ini.exists())

    def test_external_ini_edit_is_preserved(self):
        body="import os,pathlib\np=pathlib.Path(os.environ['HOME'])/'.config/Epic/CarlaUE4/Saved/Config/LinuxNoEditor/Engine.ini'\np.write_text('external edit\\n')\n"
        with tempfile.TemporaryDirectory() as td:
            base,account,argv,ini=self.setup_fixture(td,body)
            self.assertEqual(self.run_main(account,argv),0)
            self.assertEqual(ini.read_text(),'external edit\n')
            self.assertFalse(json.loads((base/'logs/shutdown.json').read_text())['engine_restored'])

    def test_term_escalates_and_stops_owned_grandchild(self):
        body="""import os,pathlib,signal,subprocess,sys,time
signal.signal(signal.SIGTERM,signal.SIG_IGN)
grandchild=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'])
(pathlib.Path(os.environ['HOME'])/'ready.json').write_text(str(grandchild.pid))
time.sleep(30)
"""
        with tempfile.TemporaryDirectory() as td:
            base,account,argv,ini=self.setup_fixture(td,body)
            ready=Path(account.pw_dir)/'ready.json';cancel=threading.Event()
            def interrupt():
                deadline=time.monotonic()+5
                while not cancel.is_set() and time.monotonic()<deadline:
                    if ready.exists():
                        os.kill(os.getpid(),signal.SIGTERM);return
                    time.sleep(.02)
            monitor=threading.Thread(target=interrupt,daemon=True);monitor.start()
            start=time.monotonic()
            try:code=self.run_main(account,argv)
            finally:cancel.set();monitor.join(timeout=1.)
            self.assertEqual(code,-signal.SIGKILL)
            self.assertLess(time.monotonic()-start,5.)
            grandchild=int(ready.read_text());state=Path(f'/proc/{grandchild}/stat')
            deadline=time.monotonic()+2.
            while state.exists() and state.read_text().split()[2]!='Z' and time.monotonic()<deadline:time.sleep(.02)
            self.assertTrue(not state.exists() or state.read_text().split()[2]=='Z')
            self.assertEqual(ini.read_text(),'original settings\n')
            self.assertTrue(json.loads((base/'logs/shutdown.json').read_text())['stop_requested'])


if __name__=='__main__':unittest.main()
