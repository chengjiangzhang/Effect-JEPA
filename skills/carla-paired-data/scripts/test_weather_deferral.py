"""Offline scheduling tests: a weather hold changes no scientific quota or gates."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pipeline
from plan_grid import build_plan
from test_orchestrator import Fixture, MODEL_BRANCHES


def add_rain(fixture):
    grid = pipeline.read(fixture.grid_path)
    grid['weather_variants'].append({'id': 'light_rain', 'overrides': {'precipitation': 20.}})
    pipeline.write(fixture.grid_path, grid)


def accepted_capture(attempt, physics, host, port):
    scene = pipeline.read(attempt / 'data/manifest.json')['scene_ids'][0]
    (attempt / 'export').mkdir()
    (attempt / 'export/index.html').write_text('fixture')
    for name in ('physics', 'physics_execution', 'action_execution', 'protocol', 'render', 'equivalence'):
        pipeline.write(attempt / 'reports' / f'{name}.json', {'status': 'passed'})
    pipeline.write(attempt / 'reports/action_selection.json', {'status': 'passed', 'scenes': {
        scene: {'model_branches': MODEL_BRANCHES, 'motion_audit_valid': True,
                'distinct_motion_count': 1, 'equivalent_motion_count': 4}}})
    pipeline.write(attempt / 'reports/execution.json', {'status': 'passed', 'scenes': {
        scene: {'execution_pass': True, 'censored_branches': [], 'response_metrics': {
            'max_npc_displacement_diff_m': 0., 'reactive_npc_count_05': 0,
            'reactive_npc_count_10': 0, 'effect_strength': 'weak'}}}})
    return attempt / 'data'


class WeatherDeferralTest(unittest.TestCase):
    def test_all_held_needs_no_native_server_or_candidate_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            with f.patches(), patch.object(pipeline, 'native_server_readiness') as native, \
                    patch.object(f.compiler, 'build_catalog') as catalog, \
                    patch.object(pipeline, 'capture_stage') as capture:
                for resume in (False, True):
                    self.assertEqual(pipeline.execute(f.args(stage='all', resume=resume,
                        defer_weather=['daylight'])), 3)
                native.assert_not_called(); catalog.assert_not_called(); capture.assert_not_called()
            state = f.state()
            self.assertEqual(state['status'], 'incomplete')
            self.assertEqual(state['candidate_specs_started'], 0)
            self.assertEqual(state['slots']['scene_0000']['attempts'], [])
            self.assertEqual(state['slots']['scene_0000']['status'], 'pending')
            self.assertEqual(f.physics_calls, [])
            self.assertFalse((f.output / 'attempts').exists())
            report = pipeline.read(f.output / 'dataset_manifest.json')
            self.assertEqual(report['deferred_scene_count'], 1)
            self.assertEqual(report['remaining_scene_count'], 1)
            self.assertEqual(report['deferred_scenes'][0]['scene_id'], 'scene_0000')

    def test_partial_capture_and_resume_release_keep_identity_and_accepted_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp)); add_rain(f)
            with f.patches(), patch.object(pipeline, 'capture_stage', side_effect=accepted_capture):
                args = dict(count=4, stage='all', dataset_purpose='evaluation')
                self.assertEqual(pipeline.execute(f.args(**args, defer_weather=['light_rain'])), 3)
                before = f.state()
                frozen = {name: (f.output / name).read_bytes()
                          for name in ('plan.json', 'pipeline_config.json', 'grid.json')}
                self.assertEqual(before['candidate_specs_started'], 2)
                self.assertEqual(sum(s['status'] == 'accepted' for s in before['slots'].values()), 2)
                identities = {name: s['delivery_inventory_sha256'] for name, s in before['slots'].items()
                              if s['status'] == 'accepted'}
                report = pipeline.read(f.output / 'dataset_manifest.json')
                self.assertEqual(report['deferred_scene_count'], 2)
                self.assertEqual(report['remaining_scene_count'], 2)
                self.assertEqual(pipeline.execute(f.args(**args, resume=True)), 0)
            after = f.state()
            self.assertEqual(after['status'], 'complete')
            self.assertEqual(after['candidate_specs_started'], 4)
            self.assertEqual(len(f.compiled), 4)
            self.assertEqual(len(f.physics_calls), 4)
            for name, content in frozen.items():
                self.assertEqual((f.output / name).read_bytes(), content)
            for name, inventory in identities.items():
                self.assertEqual(after['slots'][name]['delivery_inventory_sha256'], inventory)
            for slot in after['slots'].values():
                self.assertEqual(slot['status'], 'accepted')
                self.assertNotIn('scheduling_deferral', slot)
                self.assertEqual(slot['response_metrics']['effect_strength'], 'weak')
                self.assertEqual(slot['model_branches'], MODEL_BRANCHES)
            self.assertEqual(after['invocation_schedule']['requested_deferred_weather_ids'], [])
            self.assertTrue(all(call[1]['dataset_purpose'] == 'evaluation' for call in f.curation_calls))
            report = pipeline.read(f.output / 'dataset_manifest.json')
            self.assertEqual(report['deferred_scene_count'], 0)
            self.assertEqual(report['remaining_scene_count'], 0)
            events = [json.loads(line) for line in (f.output / 'events.jsonl').read_text().splitlines()]
            schedules = [event for event in events if event['event'] == 'scheduling_scope']
            self.assertEqual([s['requested_deferred_weather_ids'] for s in schedules], [['light_rain'], []])

    def test_hold_preserves_precompiled_attempt_then_reuses_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            with f.patches():
                self.assertEqual(pipeline.execute(f.args()), 0)
                before = deepcopy(f.state()['slots']['scene_0000']['attempts'])
                with patch.object(pipeline, 'native_server_readiness') as native:
                    self.assertEqual(pipeline.execute(f.args(resume=True, stage='physics',
                        defer_weather=['daylight'])), 3)
                    native.assert_not_called()
                self.assertEqual(f.state()['slots']['scene_0000']['status'], 'compiled')
                self.assertEqual(f.state()['slots']['scene_0000']['attempts'], before)
                self.assertEqual(pipeline.execute(f.args(resume=True, stage='physics')), 0)
            self.assertEqual(len(f.compiled), 1)
            self.assertEqual(f.physics_calls, ['scene_0000'])
            self.assertEqual(f.state()['candidate_specs_started'], 1)

    def test_invalid_weather_rejected_before_freeze_or_native_rpc(self):
        for weather in (['unknown'], ['daylight', 'daylight']):
            with self.subTest(weather=weather), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(Path(tmp))
                with f.patches(), patch.object(pipeline, 'native_server_readiness') as native:
                    with self.assertRaises(ValueError):
                        pipeline.execute(f.args(stage='all', defer_weather=weather))
                    native.assert_not_called()
                self.assertFalse((f.output / 'pipeline_config.json').exists())
                self.assertFalse((f.output / 'state.json').exists())

    def test_exhausted_slots_are_not_resurrected_by_hold_or_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            def fail(job, repair):
                raise ValueError('invalid_geometry')
            f.compile_failure = fail
            with f.patches():
                self.assertEqual(pipeline.execute(f.args()), 3)
                before = deepcopy(f.state()['slots'])
                for weather in (['daylight'], []):
                    self.assertEqual(pipeline.execute(f.args(resume=True, defer_weather=weather)), 3)
                    self.assertEqual(f.state()['slots'], before)
            self.assertEqual(f.state()['candidate_specs_started'], 2)
            self.assertEqual(len(f.compiled), 6)

    def test_real_200_plan_keeps_all_weather_quotas_and_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            plan = build_plan(pipeline.read(pipeline.PACKAGE / 'assets/grid.example.json'),
                              200, 20261103, 3, 400)
            before = deepcopy(plan)
            state = {'slots': {j['scene_id']: {'status': 'pending'} for j in plan['jobs']}}
            scope = pipeline.scheduling_scope(f.args(stage='all', defer_weather=['light_rain']), plan, state)
            self.assertEqual(scope['deferred_scene_count'], 29)
            self.assertEqual(scope['runnable_scene_count'], 171)
            self.assertEqual(scope['candidate_budget_consumed_by_deferral'], 0)
            self.assertEqual(plan, before)
            self.assertEqual(len(set(scope['deferred_scene_ids'] + scope['runnable_scene_ids'])), 200)


if __name__ == '__main__':
    unittest.main()
