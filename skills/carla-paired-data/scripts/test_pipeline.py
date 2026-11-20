"""Offline behavior checks for proposal balance and preflight decisions; no simulator used."""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from plan_grid import build_plan
from preflight_decision import REQUIRED_CHECKS, decide

SKILL = Path(__file__).resolve().parents[1]


def valid_record():
    branches = ['maintain', 'brake', 'accelerate', 'left', 'right', 'maintain_repeat']
    return {
        'schema_version': 1, 'scene_id': 'synthetic_scene',
        'spec_sha256': hashlib.sha256(b'synthetic test spec').hexdigest(), 'attempt_index': 0,
        'branches_expected': branches, 'branches_checked': list(branches),
        'checks': {k: {'status': 'pass', 'evidence': f'synthetic_fixture#{k}'} for k in REQUIRED_CHECKS},
        'collisions': [], 'observations': {},
    }


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.grid = json.loads((SKILL / 'assets/grid.example.json').read_text())

    def test_two_hundred_balanced_unique_reproducible_proposals(self):
        plan = build_plan(self.grid, 200, 20261028)
        self.assertEqual(plan, build_plan(self.grid, 200, 20261028))
        counts = Counter(j['cell_id'] for j in plan['jobs'])
        self.assertEqual(sum(counts.values()), 200)
        self.assertEqual(len(counts), 36)
        self.assertEqual(set(counts.values()), {5, 6})
        self.assertEqual(counts, {c['cell_id']: c['target_count'] for c in plan['cells']})
        self.assertEqual(len({j['seed'] for j in plan['jobs']}), 200)
        self.assertEqual(len({j['scene_id'] for j in plan['jobs']}), 200)
        self.assertFalse(plan['execution_ready'])
        self.assertTrue(all(j['executable_config'] is None for j in plan['jobs']))

    def test_seed_changes_assignment_and_excluded_pair_never_allocated(self):
        a = build_plan(self.grid, 200, 1)
        b = build_plan(self.grid, 200, 2)
        self.assertNotEqual(a['jobs'], b['jobs'])
        self.assertEqual(len(a['excluded_pairs']), 1)
        self.assertFalse(any(j['cell_id'].startswith('narrow_ramp__') for j in a['jobs']))

    def test_small_budget_keeps_zero_quota_cells_visible(self):
        plan = build_plan(self.grid, 3, 17)
        self.assertEqual(len(plan['cells']), 36)
        self.assertEqual(sum(c['target_count'] == 0 for c in plan['cells']), 33)

    def test_all_twelve_intended_templates_receive_quota(self):
        plan = build_plan(self.grid, 200, 1)
        self.assertEqual(plan['disabled_pairs'], [])
        needed = {'left_signalized', 'cross_signalized', 'pedestrian_straight',
                  'pedestrian_unsignalized', 'narrow_meeting'}
        counts = Counter(c['template'] for c in plan['cells'] if c['target_count'] > 0)
        self.assertEqual(len(counts), 12)
        for template in needed:
            self.assertEqual(counts[template], 3)

    def test_ambiguous_or_invalid_grid_rejected(self):
        for alteration in ('duplicate', 'invalid_status', 'no_reference', 'qa_action'):
            grid = copy.deepcopy(self.grid)
            if alteration == 'duplicate':
                grid['pairs'].append(grid['pairs'][0])
            elif alteration == 'invalid_status':
                grid['pairs'][0]['semantic_status'] = 'proven'
            elif alteration == 'no_reference':
                grid['pairs'][0]['candidate_actions'].remove('maintain')
            else:
                grid['pairs'][0]['candidate_actions'].append('maintain_repeat')
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                build_plan(grid, 200, 1)
        with self.assertRaises(ValueError):
            build_plan(self.grid, 200, 1, max_candidates=199)

    def test_cli_writes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'plan.json'
            cmd = [sys.executable, '-B', str(SKILL / 'scripts/plan_grid.py'),
                   '--grid', str(SKILL / 'assets/grid.example.json'), '--count', '200',
                   '--seed', '123', '--out', str(out)]
            first = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            raw = out.read_bytes()
            second = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(second.returncode, 2)
            self.assertEqual(raw, out.read_bytes())


class DecisionTests(unittest.TestCase):
    def test_complete_physics_ready_not_rgb_approved(self):
        result = decide(valid_record())
        self.assertEqual(result['decision'], 'ready_for_rgb')
        self.assertFalse(result['render_quality_approved'])

    def test_post_action_collision_and_weak_reaction_retained(self):
        record = valid_record()
        record['collisions'] = [{'time_s': 1.85, 'branch': 'left', 'evidence': 'fixture#contact'}]
        record['observations'] = {'weak_npc_response': True, 'maneuver_interrupted_by_contact': True}
        result = decide(record)
        self.assertEqual(result['decision'], 'ready_for_rgb')
        self.assertEqual(len(result['post_intervention_collisions']), 1)

    def test_initial_contact_is_not_an_action_consequence(self):
        for time in (-1.0, 0.0):
            record = valid_record()
            record['collisions'] = [{'time_s': time, 'branch': 'maintain', 'evidence': 'fixture#contact'}]
            self.assertEqual(decide(record)['decision'], 'repair_or_replace')

    def test_failed_control_not_excused_by_future_contact(self):
        record = valid_record()
        record['checks']['action_execution']['status'] = 'fail'
        record['collisions'] = [{'time_s': 1, 'branch': 'left', 'evidence': 'fixture#contact'}]
        self.assertEqual(decide(record)['decision'], 'repair_or_replace')

    def test_repair_budget_exhausts_after_three_modifications(self):
        record = valid_record()
        record['checks']['geometry']['status'] = 'fail'
        for attempt in range(4):
            record['attempt_index'] = attempt
            result = decide(record, max_repairs=3)
            self.assertEqual(result['remaining_repairs'], 3-attempt)
            self.assertEqual(result['decision'], 'repair_or_replace' if attempt < 3 else 'reject_candidate')
        record['attempt_index'] = 4
        with self.assertRaises(ValueError):
            decide(record)

    def test_missing_or_empty_evidence_cannot_pass(self):
        for alteration in ('missing_check', 'empty_evidence', 'partial_branches', 'no_events_audit'):
            record = valid_record()
            if alteration == 'missing_check':
                del record['checks']['repeat_state']
            elif alteration == 'empty_evidence':
                record['checks']['geometry']['evidence'] = ''
            elif alteration == 'partial_branches':
                record['branches_checked'].pop()
            else:
                record['collisions'] = None
            self.assertEqual(decide(record)['decision'], 'needs_evidence')

    def test_invalid_collision_time_or_identity_rejected(self):
        for event in ({'time_s': float('nan'), 'branch': 'left'},
                      {'time_s': 1, 'branch': 'not_in_config'}):
            record = valid_record()
            record['collisions'] = [event]
            with self.assertRaises(ValueError):
                decide(record)

    def test_decision_cli_status_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = valid_record()
            record['checks']['geometry']['status'] = 'fail'
            record['attempt_index'] = 3
            inp, out = Path(tmp) / 'checks.json', Path(tmp) / 'decision.json'
            inp.write_text(json.dumps(record))
            cmd = [sys.executable, '-B', str(SKILL / 'scripts/preflight_decision.py'),
                   '--record', str(inp), '--out', str(out)]
            first = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(first.returncode, 4, first.stderr)
            raw = out.read_bytes()
            self.assertEqual(json.loads(raw)['decision'], 'reject_candidate')
            second = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(second.returncode, 64)
            self.assertEqual(raw, out.read_bytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
