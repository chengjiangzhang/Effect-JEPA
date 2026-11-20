"""Coverage and split-conditioning contracts without a simulator or model API."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from plan_grid import build_plan
from sampling import coverage_summary, location_id, location_shortlist, weather_variants
import pipeline
from test_orchestrator import FakeCatalog
from test_scene_compiler import Catalog
from scene_compiler import validate_selection

GRID = json.loads((Path(__file__).resolve().parents[1] / 'assets/grid.example.json').read_text())


class SamplingTests(unittest.TestCase):
    def test_all_seven_weather_conditions_represented_within_each_template(self):
        plan = build_plan(GRID, 200, 42)
        self.assertEqual(len(plan['weather_variants']), 7)
        self.assertEqual(set(v['id'] for v in plan['weather_variants']),
                         {'daylight', 'soft_daylight', 'late_afternoon', 'overcast', 'light_rain', 'light_fog', 'night'})
        for counts in plan['planned_weather_by_template'].values():
            self.assertTrue(all(n > 0 for n in counts.values()))
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_small_budget_prioritizes_template_coverage(self):
        for count in (12, 20, 36):
            plan = build_plan(GRID, count, 5)
            cells = {c['cell_id']: c for c in plan['cells']}
            self.assertEqual(len({cells[j['cell_id']]['template'] for j in plan['jobs']}), 12)
            self.assertEqual(len(plan['jobs']), count)

    def test_weather_changes_do_not_change_scene_quota_seed_or_layout(self):
        old = deepcopy(GRID); old['weather_variants'] = old['weather_variants'][:3]
        a, b = build_plan(old, 200, 13), build_plan(GRID, 200, 13)
        self.assertEqual(a['cells'], b['cells'])
        for ja, jb in zip(a['jobs'], b['jobs']):
            self.assertEqual({k: v for k, v in ja.items() if k != 'weather_id'},
                             {k: v for k, v in jb.items() if k != 'weather_id'})

    def test_location_first_and_connector_reuse_is_not_a_new_location(self):
        candidates = [{'candidate_id': x, 'map': 'TownTest', 'anchor': {'junction_id': j, 'connector': x}}
                      for x, j in [('a', 1), ('b', 1), ('c', 2)]]
        used = [{'template': 'left', 'candidate_id': 'a', 'map': 'TownTest',
                 'location_id': location_id(candidates[0])}]
        self.assertEqual(location_id(candidates[0]), location_id(candidates[1]))
        self.assertEqual([c['candidate_id'] for c in location_shortlist(candidates, used, 'left', 10)], ['c'])
        # An infeasible attempted candidate permits another real site; weather is irrelevant.
        self.assertEqual(len(location_shortlist(candidates, used, 'left', 10, ['c'])), 1)

    def test_agent_and_deterministic_paths_cannot_select_easier_weather(self):
        grid = deepcopy(GRID)
        grid['pairs'] = [p for p in grid['pairs'] if p['id'] == 'follower_response_straight']
        plan = build_plan(grid, 7, 8); job = plan['jobs'][0]
        cell = next(c for c in plan['cells'] if c['cell_id'] == job['cell_id'])
        with tempfile.TemporaryDirectory() as tmp:
            state = {'slots': {}, 'model_calls': 0}
            for repair in range(4):
                selection, _ = pipeline.select_spec(None, FakeCatalog(), job, cell, plan, repair, [], Path(tmp), state)
                self.assertEqual(selection['weather_id'], job['weather_id'])
        catalog = Catalog(Path('/unused'), {}, {'follower_response_straight': [{'candidate_id': 'a'}]}, [], {})
        different = next(w['id'] for w in GRID['weather_variants'] if w['id'] != job['weather_id'])
        with self.assertRaisesRegex(ValueError, 'balanced'):
            validate_selection({'weather_id': different}, job, cell, catalog)

    def test_exposure_and_weather_values_validated_early(self):
        for key, value in [('sun_altitude_angle', -100), ('cloudiness', 101), ('precipitation', float('nan')), ('typo', 1)]:
            grid = deepcopy(GRID); grid['weather_variants'][0]['overrides'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                build_plan(grid, 36, 1)
        grid = deepcopy(GRID); grid['weather_variants'][-1]['camera_overrides']['iso'] = 0
        with self.assertRaises(ValueError):
            weather_variants(grid)
        grid = deepcopy(GRID); grid['weather_variants'].append(deepcopy(grid['weather_variants'][0]))
        with self.assertRaises(ValueError):
            weather_variants(grid)

    def test_accepted_distribution_reports_weather_shortfalls(self):
        plan = build_plan(GRID, 20, 2)
        rows = [{'template': 'left_unsignalized', 'weather_id': 'daylight',
                 'base_scene_id': 'base_1', 'location_id': 'location_1'}]
        report = coverage_summary(plan, rows)['left_unsignalized']
        self.assertEqual(report['distinct_base_scenes'], 1)
        self.assertEqual(len(report['missing_weather_ids']), 6)
        self.assertEqual(report['scene_versions'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
