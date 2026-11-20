"""Real planning/binding on synthetic history; only CARLA and map writing doubled."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pipeline
import planning_stage as stage
from test_measured_t0 import fixture, save_history


class PlanningStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.attempt = Path(self.temp.name) / 'attempt'
        self.spec, self.states = fixture()
        self.scene = self.spec['scene_id']
        self.original = self.attempt / 'initial_plan' / self.scene / 'spec.json'
        pipeline.write(self.original, self.spec)
        pipeline.write(self.attempt / 'initial_plan/manifest.json', {'scene_ids': [self.scene]})
        self.order = []
        self.next_history_collision = False

    def history_command(self, command, log, timeout_s=1800):
        self.assertIn('--history-only', command)
        self.assertEqual(Path(command[command.index('--data')+1]), self.attempt/'initial_plan')
        self.order.append('observe_shared_history')
        root = Path(command[command.index('--out')+1])
        save_history(root, self.spec, self.states,
                     collisions=[{'time_s': -1.}] if self.next_history_collision else [])

    def write_plan(self, data, specs, plan, catalog):
        self.order.append('freeze_measured_plan')
        self.assertEqual(specs[0]['vehicles'], self.spec['vehicles'])
        self.assertEqual(specs[0]['action'], self.spec['action'])
        self.assertTrue(specs[0]['measured_t0_plan']['actual_t0_checked'])
        pipeline.write(data/self.scene/'spec.json', specs[0])
        pipeline.write(data/'manifest.json', {'scene_ids': [self.scene]})

    def run_stage(self):
        with patch.object(pipeline, 'run_command', side_effect=self.history_command), \
             patch.dict('sys.modules', {'scene_compiler': SimpleNamespace(write_plan=self.write_plan)}):
            return stage.prepare_measured_plan(self.attempt, self.scene, 'localhost', 2000,
                                                plan={}, catalog=None)

    def test_measured_initial_state_precedes_plan_and_original_is_immutable(self):
        before = self.original.read_bytes()
        result = self.run_stage()
        self.assertEqual(self.order, ['observe_shared_history','freeze_measured_plan'])
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(before, self.original.read_bytes())
        self.assertEqual(result['evidence']['actors']['ego']['speed_m_s'], 3.)
        self.assertEqual(self.spec['vehicles'][0]['initial_speed_mps'], 4.)
        final = pipeline.read(self.attempt/'data'/self.scene/'spec.json')
        self.assertNotEqual(final['spec_sha256'], self.spec['spec_sha256'])

    def test_resume_reuses_same_history_and_plan_without_replanning(self):
        before = self.run_stage()
        after = self.run_stage()
        self.assertEqual(before, after)
        self.assertEqual(self.order, ['observe_shared_history','freeze_measured_plan'])

    def test_tampered_measurement_stops_without_new_candidate_or_plan(self):
        result = self.run_stage()
        path = self.attempt/result['history_directory']/self.scene/'branches/common_history/states.npz'
        path.write_bytes(path.read_bytes()+b'tamper')
        with self.assertRaises(pipeline.CaptureInfrastructureError):
            self.run_stage()
        self.assertEqual(len(self.order), 2)

    def test_infeasible_real_t0_returns_shared_setup_repair_without_final_data(self):
        self.states['position'][:,0,0] += 8.
        with self.assertRaisesRegex(ValueError,'measured_t0_plan_failed'):
            self.run_stage()
        self.assertEqual(self.order, ['observe_shared_history'])
        self.assertFalse((self.attempt/'data').exists())
        report = pipeline.read(self.attempt/'reports/t0_planning.json')
        self.assertEqual(report['status'], 'failed')
        self.assertIn('measured_t0_stop_distance_budget_insufficient', str(report['failure_reasons']))

    def test_history_contact_is_shared_setup_failure_not_new_future_outcome(self):
        self.next_history_collision = True
        with self.assertRaisesRegex(ValueError, 'measured_t0_plan_failed'):
            self.run_stage()
        self.assertFalse((self.attempt/'data').exists())

    def test_orphan_final_plan_is_not_silently_bound_to_new_history(self):
        pipeline.write(self.attempt/'data/manifest.json', {'scene_ids': [self.scene]})
        with self.assertRaises(pipeline.CaptureInfrastructureError):
            self.run_stage()
        self.assertEqual(self.order, [])

    def test_history_timeout_preserves_attempt_as_infrastructure_interruption(self):
        import subprocess
        with patch.object(pipeline,'run_command',side_effect=subprocess.TimeoutExpired('history',600)):
            with self.assertRaises(pipeline.CaptureInfrastructureError):
                stage.prepare_measured_plan(self.attempt,self.scene,'localhost',2000,plan={},catalog=None)
        self.assertFalse((self.attempt/'data').exists())

    def test_physical_prefix_failure_blocks_execution_audit_and_rgb(self):
        self.run_stage()
        commands = []
        def command(argv, log, timeout_s=1800):
            script = Path(argv[2]).name; commands.append(script)
            target = Path(argv[argv.index('--out')+1])
            if script == 'preflight_physics.py':
                target.mkdir()
            elif script == 'audit_history_prefix.py':
                pipeline.write(target, {'status':'failed','errors':['ego t0 state differs']})
            else:
                self.fail('Should stop before later audits/capture: '+script)
        with patch.object(pipeline,'run_command',side_effect=command):
            with self.assertRaisesRegex(ValueError,'measured_common_history_physics_mismatch'):
                pipeline.physics_stage(self.attempt,self.scene,'localhost',2000)
        self.assertEqual(commands,['preflight_physics.py','audit_history_prefix.py'])

    def test_source_file_mutation_after_final_freeze_is_detected_before_prefix_audit(self):
        self.run_stage()
        changed = copy.deepcopy(self.spec); changed['vehicles'][0]['initial_speed_mps'] += 1
        pipeline.write(self.original, changed)
        with patch.object(pipeline,'audit_command') as audit:
            with self.assertRaises(pipeline.CaptureInfrastructureError):
                stage.audit_measured_prefix(self.attempt,self.attempt/'data','capture')
            audit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
