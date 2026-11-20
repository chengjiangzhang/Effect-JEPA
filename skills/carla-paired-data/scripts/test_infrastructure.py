"""Production infrastructure failures preserve the scene and its repair budget."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pipeline
from test_orchestrator import Fixture


class InfrastructureTests(unittest.TestCase):
    def test_resume_identity_includes_external_engine_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            ini = Path(tmp) / 'external.ini'
            config = Path(tmp) / 'server.json'
            ini.write_text('[Settings]\nquality=original\n')
            pipeline.write(config, {'engine_ini': str(ini)})
            args = fixture.args(managed_server_config=config)
            with fixture.patches():
                before = pipeline.configuration(args)
                ini.write_text('[Settings]\nquality=changed\n')
                after = pipeline.configuration(args)
            self.assertEqual(before['managed_server_config_sha256'], after['managed_server_config_sha256'])
            self.assertNotEqual(before['managed_engine_ini_sha256'], after['managed_engine_ini_sha256'])

    def test_timeout_comes_from_latest_failed_job_not_generic_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            path = reports / 'capture/batch.json'
            pipeline.write(path, {'jobs': [{'status': 'failed', 'timeout': True, 'branch': 'maintain'}]})
            self.assertEqual(pipeline.capture_timeout_job(reports)['branch'], 'maintain')
            for marker in (False, 'true', None):
                pipeline.write(path, {'jobs': [{'status': 'failed', 'timeout': True},
                                              {'status': 'failed', 'timeout': marker, 'returncode': -1}]})
                self.assertIsNone(pipeline.capture_timeout_job(reports))

    def test_capture_timeout_is_infrastructure_not_scene_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            pipeline.write(attempt / 'reports/capture/batch.json', {'jobs': [
                {'status': 'failed', 'branch': 'maintain', 'timeout': True}]})
            with patch.object(pipeline, 'run_command', side_effect=RuntimeError('collector failed')):
                with self.assertRaises(pipeline.CaptureInfrastructureError):
                    pipeline.capture_stage(attempt, attempt / 'physics', '127.0.0.1', 2000)

    def test_infrastructure_interruption_does_not_repair_or_exhaust_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            with fixture.patches(), patch.object(pipeline, 'capture_stage',
                    side_effect=pipeline.CaptureInfrastructureError('capture_branch_timeout')):
                with self.assertRaises(pipeline.CaptureInfrastructureError):
                    pipeline.execute(fixture.args(stage='all'))
            state = json.loads((fixture.output / 'state.json').read_text())
            self.assertEqual(state['status'], 'blocked_infrastructure')
            slot = state['slots']['scene_0000']
            self.assertNotEqual(slot['status'], 'exhausted')
            self.assertEqual(len(slot['attempts']), 1)
            self.assertEqual(slot['attempts'][0]['status'], 'interrupted')
            self.assertFalse((fixture.output / slot['attempts'][0]['path'] / 'failure.json').exists())

    def test_managed_server_boundary_precedes_physics_and_error_aborts(self):
        from managed_server import ManagedServerError
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            class BrokenManager:
                def before_attempt(self, *args):
                    raise ManagedServerError('not ready')
            with fixture.patches(), self.assertRaises(ManagedServerError):
                pipeline._execute(fixture.args(stage='physics'), BrokenManager())
            self.assertEqual(fixture.physics_calls, [])
            state = json.loads((fixture.output / 'state.json').read_text())
            self.assertEqual(state['status'], 'blocked_infrastructure')

    def test_managed_branch_restarts_cannot_bypass_repeat_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            pipeline.write(attempt / 'data/manifest.json', {'scene_ids': ['scene_0000']})
            pipeline.write(attempt / 'data/scene_0000/spec.json', {'weather': {'precipitation': 0}})
            epochs = []
            def restart(*identity):
                epochs.append(identity)
                return {'epoch': len(epochs)}
            def collect(data, reports, host, port, max_new_branches):
                self.assertEqual(max_new_branches, 1)
                if len(epochs) == 2:
                    raise ValueError('RGB repeat preflight failed')
                pipeline.write(reports / 'capture/batch.json', {
                    'status': 'stopped_after_branch_limit', 'completed_jobs': 1, 'total_jobs': 6})
            with patch('controllers_v2.branch_names', return_value=['branch_'+str(i) for i in range(6)]), \
                    patch.object(pipeline, 'collect_capture_pass', side_effect=collect), \
                    patch.object(pipeline, 'audit_command') as final_audit:
                with self.assertRaisesRegex(ValueError, 'RGB repeat preflight failed'):
                    pipeline.capture_stage(attempt, attempt / 'physics', '127.0.0.1', 2000,
                        SimpleNamespace(before_attempt=restart), ('scene_0000', 0, 0))
                self.assertEqual(len(epochs), 2)
                final_audit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
