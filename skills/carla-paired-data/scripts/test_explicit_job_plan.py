"""Explicit new-cohort jobs; synthetic fixtures only, no native acceptance.

Later derive a subset from frozen parent plan/manifest bytes using
accepted_scenes[*].scene_id, retain remaining jobs in order, and recompute
target_count and weather counts. Record the parent raw plan SHA256. External
ops must cap prior+new consumption; never import parent state or attempts.
This module derives only synthetic plans, never a live remaining plan.
"""
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pipeline
from plan_grid import build_plan, canonical_hash, load_job_plan, planned_weather_counts, validate_job_plan
from test_orchestrator import Fixture

GRID = json.loads((Path(__file__).resolve().parents[1]/'assets/grid.example.json').read_text())


def subset(parent, indices, budget):
    result = deepcopy(parent)
    result['jobs'] = [deepcopy(parent['jobs'][index]) for index in indices]
    result['count'] = len(indices)
    result['max_candidate_specs_in_batch'] = budget
    result['parent_plan_sha256'] = canonical_hash(parent)
    counts = Counter(job['cell_id'] for job in result['jobs'])
    for cell in result['cells']:
        cell['target_count'] = counts[cell['cell_id']]
    result['planned_weather_by_template'] = planned_weather_counts(
        result['jobs'], result['cells'], result['weather_variants'])
    return result


class ExplicitJobPlanValidationTests(unittest.TestCase):
    def setUp(self):
        self.parent = build_plan(GRID, 200, 20261111, 3, 1600)
        self.plan = subset(self.parent, [9, 42, 199], 24)

    def validate(self, plan=None, **kwargs):
        options = dict(count=3, seed=20261111, max_repairs=3, max_candidates=24)
        options.update(kwargs)
        return validate_job_plan(self.plan if plan is None else plan, GRID, **options)

    def test_sparse_subset_keeps_jobs_and_zero_quota_cells(self):
        before = deepcopy(self.parent)
        result = self.validate()
        self.assertEqual([j['scene_id'] for j in result['jobs']], ['scene_0009', 'scene_0042', 'scene_0199'])
        self.assertEqual(result['jobs'], [self.parent['jobs'][i] for i in (9, 42, 199)])
        self.assertEqual(self.parent, before)
        self.assertEqual(len(result['cells']), 36)
        self.assertEqual(sum(c['target_count'] for c in result['cells']), 3)
        self.assertTrue(any(c['target_count'] == 0 for c in result['cells']))
        self.assertEqual(len(result['weather_variants']), 7)
        result['jobs'][0]['seed'] = -1
        self.assertNotEqual(result, self.plan)

    def test_count_seed_and_budget_match_cli(self):
        for options in ({'count': 4}, {'count': 0}, {'seed': 1}, {'max_repairs': 2},
                        {'max_candidates': 23}, {'max_candidates': 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.validate(**options)
        for key in ('count', 'base_seed', 'max_repairs_per_candidate', 'max_candidate_specs_in_batch'):
            plan = deepcopy(self.plan); plan[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(plan)

    def test_missing_unknown_and_execution_fields_rejected(self):
        for key, value in (('accepted_scenes', []), ('state', {}), ('attempts', []),
                           ('candidate_specs_started', 10), ('status', 'complete'),
                           ('execution_ready', True)):
            plan = deepcopy(self.plan); plan[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(plan)
        plan = deepcopy(self.plan); del plan['grid_sha256']
        with self.assertRaises(ValueError): self.validate(plan)

    def test_job_execution_sentinels_and_unknown_fields_rejected(self):
        for key, value in (('stage', 'accepted'), ('executable_config', {}),
                           ('attempt_index', 1), ('attempt_index', False),
                           ('attempts', []), ('status', 'accepted'), ('accepted_data', 'old/data')):
            plan = deepcopy(self.plan); plan['jobs'][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): self.validate(plan)

    def test_scene_ids_and_seeds_unique_and_canonical(self):
        for key, value in (('scene_id', self.plan['jobs'][1]['scene_id']),
                           ('scene_id', '../scene_0009'), ('scene_id', 'scene_9'),
                           ('scene_id', 'scene_00009'), ('scene_id', 'scene_0009/escape'),
                           ('seed', self.plan['jobs'][1]['seed']), ('seed', True),
                           ('seed', -1), ('seed', 2**31)):
            plan = deepcopy(self.plan); plan['jobs'][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): self.validate(plan)

    def test_component_seeds_bound_to_scene_seed(self):
        for change in ('missing', 'value', 'type', 'extra'):
            plan = deepcopy(self.plan); component = plan['jobs'][0]['component_seeds']
            if change == 'missing': del component['npc']
            elif change == 'value': component['layout'] += 1
            elif change == 'type': component['weather'] = str(component['weather'])
            else: component['attempts'] = []
            with self.subTest(change=change), self.assertRaises(ValueError): self.validate(plan)

    def test_cells_match_grid_and_exact_counts(self):
        for change in ('missing', 'duplicate', 'count', 'negative', 'bool', 'style', 'action', 'extra'):
            plan = deepcopy(self.plan); cell = plan['cells'][0]
            if change == 'missing': plan['cells'].pop()
            elif change == 'duplicate': plan['cells'][1] = deepcopy(cell)
            elif change == 'count': cell['target_count'] += 1
            elif change == 'negative': cell['target_count'] = -1
            elif change == 'bool': cell['target_count'] = False
            elif change == 'style': cell['npc_style_parameters']['time_headway'] = 99
            elif change == 'action': cell['candidate_actions'].append('invented')
            else: cell['accepted'] = 0
            with self.subTest(change=change), self.assertRaises(ValueError): self.validate(plan)
        plan = deepcopy(self.plan)
        active = next(c for c in plan['cells'] if c['target_count'])
        inactive = next(c for c in plan['cells'] if not c['target_count'])
        active['target_count'] -= 1; inactive['target_count'] += 1
        with self.assertRaisesRegex(ValueError, 'allocation'): self.validate(plan)

    def test_weather_definition_references_and_statistics_match(self):
        for change in ('variant', 'unknown', 'cell', 'count', 'grid'):
            plan = deepcopy(self.plan)
            if change == 'variant': plan['weather_variants'][0]['overrides']['cloudiness'] = 3
            elif change == 'unknown': plan['jobs'][0]['weather_id'] = 'invented'
            elif change == 'cell': plan['jobs'][0]['cell_id'] = 'invented'
            elif change == 'grid': plan['grid_sha256'] = '0'*64
            else:
                template = next(iter(plan['planned_weather_by_template']))
                plan['planned_weather_by_template'][template]['daylight'] += 1
            with self.subTest(change=change), self.assertRaises(ValueError): self.validate(plan)

    def test_empty_or_malformed_collections_rejected(self):
        for key, value in (('jobs', []), ('jobs', None), ('jobs', [None]*3),
                           ('cells', {}), ('cells', [None]*36), ('weather_variants', [])):
            plan = deepcopy(self.plan); plan[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): self.validate(plan)

    def test_file_exact_bytes_duplicate_keys_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'jobs.json'
            raw = json.dumps(self.plan).encode(); path.write_bytes(raw)
            loaded, digest = load_job_plan(path, GRID, 3, 20261111, 3, 24)
            self.assertEqual(loaded, self.plan)
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            path.write_bytes(raw+b'\n')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                load_job_plan(path, GRID, 3, 20261111, 3, 24, digest)
            for raw in ('{"count": 3, "count": 3}', '{"count": NaN}'):
                path.write_text(raw)
                with self.assertRaises(ValueError): load_job_plan(path, GRID, 3, 20261111, 3, 24)


class ExplicitJobPlanPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.fixture = Fixture(self.root)
        self.grid = pipeline.read(self.fixture.grid_path)
        self.plan = subset(build_plan(self.grid, 5, 19, 2, 10), [1, 4], 4)
        self.path = self.root/'explicit.json'; pipeline.write(self.path, self.plan)
        self.source = patch.object(pipeline, 'source_identity', return_value={'fixture': 'test-only'})
        self.source.start(); self.addCleanup(self.source.stop)

    def args(self, **kwargs):
        values = dict(command='plan', stage='all', count=2, max_candidates=4, job_plan=self.path)
        values.update(kwargs)
        return self.fixture.args(**values)

    def prepare(self, **kwargs):
        with redirect_stdout(io.StringIO()): return pipeline.prepare(self.args(**kwargs))

    def test_prepare_pending_only_and_preview_use_sparse_jobs(self):
        scope = pipeline.preview_scheduling_scope(self.args())
        self.assertEqual(scope['runnable_scene_ids'], ['scene_0001', 'scene_0004'])
        self.assertFalse(self.fixture.output.exists())
        _, cfg, plan, state = self.prepare()
        self.assertEqual(plan, self.plan)
        self.assertEqual(cfg['job_plan_sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual((state['candidate_specs_started'], state['model_calls']), (0, 0))
        self.assertEqual(set(state['slots']), {'scene_0001', 'scene_0004'})
        self.assertTrue(all(x['status'] == 'pending' and x['attempts'] == [] for x in state['slots'].values()))
        self.assertEqual(pipeline.read(self.path), self.plan)

    def test_plan_command_zero_accepted_and_no_native_calls(self):
        with patch.object(pipeline, 'native_server_readiness') as native, redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.execute(self.args()), 0)
        native.assert_not_called()
        report = pipeline.read(self.fixture.output/'dataset_manifest.json')
        self.assertEqual(report['target_scene_count'], 2)
        self.assertEqual(report['accepted_scene_count'], 0)
        self.assertEqual(report['accepted_scenes'], [])

    def test_invalid_plan_rejected_before_output_lock_or_native(self):
        self.plan['jobs'][0]['executable_config'] = {'old': 'execution'}
        pipeline.write(self.path, self.plan)
        with patch.object(pipeline, 'native_server_readiness') as native:
            with self.assertRaises(ValueError): pipeline.execute(self.args(command='run'))
        native.assert_not_called(); self.assertFalse(self.fixture.output.exists())
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse(self.fixture.output.exists())

    def test_resume_keeps_attempts_consumption_and_exhausted_state(self):
        _, cfg, plan, state = self.prepare()
        state['candidate_specs_started'] = 2
        state['slots']['scene_0001'].update(status='exhausted', attempts=[{
            'path': 'attempts/scene_0001/candidate_00/attempt_00', 'candidate_index': 0,
            'attempt_index': 0, 'status': 'failed', 'reason': 'fixture failure'}])
        pipeline.write(self.fixture.output/'state.json', state)
        _, cfg2, plan2, state2 = self.prepare(resume=True)
        self.assertEqual((cfg2, plan2, state2), (cfg, plan, state))
        self.assertEqual(pipeline.preview_scheduling_scope(self.args(resume=True))['runnable_scene_ids'], ['scene_0004'])

    def test_changed_input_bytes_same_semantics_rejected_on_resume(self):
        self.prepare(); self.path.write_bytes(self.path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError, 'differ'): self.prepare(resume=True)
        with self.assertRaisesRegex(ValueError, 'unchanged'):
            pipeline.preview_scheduling_scope(self.args(resume=True))

    def test_tampered_stored_plan_or_state_slot_identity_rejected(self):
        _, _, plan, state = self.prepare()
        changed = deepcopy(plan); changed['jobs'].reverse()
        pipeline.write(self.fixture.output/'plan.json', changed)
        with self.assertRaisesRegex(ValueError, 'Stored plan'): self.prepare(resume=True)
        pipeline.write(self.fixture.output/'plan.json', plan)
        for change in ('missing', 'extra', 'cell'):
            changed = deepcopy(state)
            if change == 'missing': del changed['slots']['scene_0001']
            elif change == 'extra': changed['slots']['scene_9999'] = {'cell_id': plan['jobs'][0]['cell_id']}
            else: changed['slots']['scene_0001']['cell_id'] = 'different'
            pipeline.write(self.fixture.output/'state.json', changed)
            with self.subTest(change=change), self.assertRaises(ValueError): self.prepare(resume=True)

    def test_adding_job_plan_cannot_resume_ordinary_batch(self):
        with redirect_stdout(io.StringIO()): pipeline.prepare(self.args(job_plan=None))
        with self.assertRaisesRegex(ValueError, 'differ'): self.prepare(resume=True)

    def test_ordinary_path_has_no_new_configuration_key(self):
        args = self.args(job_plan=None)
        self.assertNotIn('job_plan_sha256', pipeline.configuration(args))
        with redirect_stdout(io.StringIO()): _, _, plan, _ = pipeline.prepare(args)
        self.assertEqual(plan, build_plan(self.grid, 2, 19, 2, 4))

    def test_sparse_jobs_compile_with_original_weather_and_seed(self):
        with self.fixture.patches(), patch.object(pipeline, 'native_server_readiness') as native:
            self.assertEqual(pipeline.execute(self.args(command='run', stage='compile')), 0)
        native.assert_not_called()
        self.assertEqual([row['job'] for row in self.fixture.compiled], self.plan['jobs'])
        state = self.fixture.state()
        self.assertEqual(state['candidate_specs_started'], 2)
        self.assertEqual(set(state['slots']), {'scene_0001', 'scene_0004'})
        self.assertTrue(all(slot['status'] == 'compiled' for slot in state['slots'].values()))

    def test_input_changed_between_configuration_and_load_does_not_freeze(self):
        original = pipeline.configuration
        def changed_after_hash(args):
            cfg = original(args)
            self.path.write_bytes(self.path.read_bytes()+b'\n')
            return cfg
        with patch.object(pipeline, 'configuration', side_effect=changed_after_hash):
            with self.assertRaisesRegex(ValueError, 'SHA256'): self.prepare()
        self.assertFalse(self.fixture.output.exists())

    def test_invalid_configuration_or_existing_records_rejected_before_side_effects(self):
        for options in ({'maps': ['MissingMap']}, {'candidates_per_slot': 0}):
            with self.subTest(options=options), patch.object(pipeline, 'native_server_readiness') as native:
                with self.assertRaises(ValueError): pipeline.execute(self.args(command='run', **options))
                native.assert_not_called()
                self.assertFalse(self.fixture.output.exists())
        self.fixture.output.mkdir()
        old = self.fixture.output/'old_record.json'; old.write_text('preserve me')
        with self.assertRaises(FileExistsError): pipeline.execute(self.args())
        self.assertEqual(old.read_text(), 'preserve me')
        self.assertFalse((self.fixture.output/'.pipeline.lock').exists())

    def test_cli_support_and_insufficient_budget(self):
        argv = ['plan', '--output', str(self.fixture.output), '--opendrive', str(self.fixture.maps),
                '--maps', 'TownMock', '--grid', str(self.fixture.grid_path), '--job-plan', str(self.path),
                '--count', '2', '--seed', '19', '--max-repairs', '2', '--max-candidates', '4']
        with redirect_stdout(io.StringIO()): self.assertEqual(pipeline.main(argv), 0)
        for budget in (0, 1):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                pipeline.preview_scheduling_scope(self.args(max_candidates=budget))


if __name__ == '__main__':
    unittest.main()
