"""Synthetic full-motion selection and delayed-lane audit tests; no CARLA server."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

if __package__:
    from .audit_realized_actions import audit_dataset, audit_scene, motion_metrics, motion_summary, audit_motion_similarity, normalize_motion_report, select_branches
    from .audit_lane_change import lane_direction, lateral_branches, trajectory_audit
    from .test_runtime import fixture_states
else:
    from audit_realized_actions import audit_dataset, audit_scene, motion_metrics, motion_summary, audit_motion_similarity, normalize_motion_report, select_branches
    from audit_lane_change import lane_direction, lateral_branches, trajectory_audit
    from test_runtime import fixture_states


NAMES = ['maintain', 'nominal', 'stop_hold', 'yield_wait', 'commit_go']


def observations(names=NAMES):
    states = {name: fixture_states() for name in [*names, 'maintain_repeat']}
    for state in states.values():
        state['position'][:, 0, 0] = 20.+3.*state['time_s']
        state['velocity'][:, 0, 0] = 3.
        state['transform_world_vehicle'][..., :3, 3] = state['position']
    return states


def change_future(state, *, position=0., speed=0., yaw=0.):
    future = state['time_s'] > 1e-8
    state['position'][future, 0, 0] += position
    state['velocity'][future, 0, 0] += speed
    state['rotation_deg'][future, 0, 2] += yaw


def lane_config():
    route = np.array([[0., 0., 0., 0.], [100., 0., 0., 0.]])
    target = route.copy(); target[:, 1] = -3.5
    names = ['maintain', 'brake', 'accelerate', 'left', 'left_delayed']
    profiles = {'maintain': {'kind': 'maintain'},
                'brake': {'kind': 'longitudinal', 'acceleration_m_s2': -4.5, 'speed_increase_cap_m_s': 0.},
                'accelerate': {'kind': 'longitudinal', 'acceleration_m_s2': 1.5, 'speed_increase_cap_m_s': 5.},
                'left': {'kind': 'lane_change', 'direction': 'left', 'onset_s': .2, 'duration_s': 2.5},
                'left_delayed': {'kind': 'lane_change', 'direction': 'left', 'onset_s': .8, 'duration_s': 2.5}}
    nominal = {'desired_speed_m_s': 3., 'respect_signals': True, 'release_gap_s': 2.,
               'comfortable_deceleration_m_s2': 2.,
               'idm': {'max_acceleration': 2., 'comfortable_brake': 2., 'time_headway': 1.2, 'minimum_gap': 2.5}}
    return {'vehicles': [{'role': 'ego', 'route': route.tolist()}, {'role': 'rear', 'route': route.tolist()}],
            'vehicle_roles': ['ego', 'rear'], 'branches': names+['maintain_repeat'],
            'action': {'profile_version': 2, 'profile_order': names, 'profiles': profiles, 'nominal_policy': nominal,
                       'available_lane_change_directions': ['left'], 'lateral_mode': 'adjacent_lane_route',
                       'lateral_onset_s': .2, 'lateral_duration_s': 2.5,
                       'lane_changes': {'left': {'target_route': target.tolist(), 'width_m': 3.5, 'available': True,
                           'evidence': {'source_lane_keys': [[1, 0, -2]]*2, 'target_lane_keys': [[1, 0, -1]]*2,
                               'source_lane_widths_m': [3.5]*2, 'target_lane_widths_m': [3.5]*2,
                               'same_direction': True, 'target_lane_type': 'Driving', 'station_count': 2}}}}}


class RealizedActionsTests(unittest.TestCase):
    def test_identical_realized_motion_keeps_all_conditions(self):
        result = audit_motion_similarity(observations(), NAMES)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['distinct_motion_representatives'], ['maintain'])
        self.assertEqual(result['model_branches'], NAMES)
        self.assertNotIn('excluded_duplicate_branches', result)
        self.assertNotIn('planned_model_branches', result)
        self.assertNotIn('selected_model_branches', result)
        self.assertTrue(result['motion_audit_valid'])
        self.assertEqual(result['distinct_motion_count'], 1)
        self.assertEqual(result['equivalent_motion_count'], 4)
        self.assertEqual(len(result['similar_motion_branches']), 4)
        self.assertIn('maintain_repeat', result['raw_branches_retained'])

    def test_legacy_api_keeps_old_fields_and_failure_gate(self):
        states = observations()
        result = select_branches(states, NAMES)
        self.assertEqual(result['selected_model_branches'], NAMES)
        self.assertEqual(result['excluded_duplicate_branches'], {})
        self.assertEqual(result['unique_condition_shortfall'], 4)
        self.assertTrue(result['selection_valid'])
        change_future(states['maintain_repeat'], position=.1)
        failed = select_branches(states, NAMES)
        self.assertEqual(failed['selected_model_branches'], [])
        self.assertFalse(failed['selection_valid'])
        self.assertEqual(failed['model_branches'], NAMES)
        self.assertIsNone(failed['equivalent_motion_count'])

    def test_legacy_report_preserves_actual_exclusions_without_restoring_membership(self):
        excluded = {'nominal': {'representative': 'maintain', 'reason': 'old policy'}}
        legacy = {'status': 'passed', 'planned_model_branches': NAMES,
                  'selected_model_branches': ['maintain'], 'excluded_duplicate_branches': excluded,
                  'unique_condition_count': 1, 'unique_condition_shortfall': 4,
                  'selection_valid': True, 'repeat_motion_noise': {'position_m': .001}}
        report = normalize_motion_report(legacy)
        self.assertEqual(report['model_branches'], ['maintain'])
        self.assertEqual(report['distinct_motion_count'], 1)
        self.assertEqual(report['equivalent_motion_count'], 4)
        self.assertEqual(report['repeat_motion_noise'], legacy['repeat_motion_noise'])
        self.assertEqual(report['legacy_branch_selection']['excluded_duplicate_branches'], excluded)
        self.assertEqual(report['legacy_branch_selection']['planned_model_branches'], NAMES)
        self.assertEqual(normalize_motion_report(report), report)
        self.assertIn('selected_model_branches', legacy)  # Input is not rewritten.

    def test_failed_legacy_report_remains_invalid_and_empty(self):
        report = normalize_motion_report({'status': 'failed', 'planned_model_branches': NAMES,
            'selected_model_branches': [], 'excluded_duplicate_branches': {},
            'selection_valid': False, 'unique_condition_shortfall': None})
        self.assertFalse(report['motion_audit_valid'])
        self.assertEqual(report['model_branches'], [])
        self.assertIsNone(report['distinct_motion_count'])
        self.assertIsNone(report['equivalent_motion_count'])

    def test_controls_alone_do_not_make_different_motion(self):
        states = observations(); states['nominal']['control'][40:-1, 0, 0] = .5
        result = audit_motion_similarity(states, NAMES)
        self.assertIn('nominal', result['similar_motion_branches'])
        self.assertEqual(result['similar_motion_branches']['nominal']['metrics']['max_abs_control_difference_by_channel'], [.5, 0., 0.])

    def test_identical_endpoints_do_not_hide_intermediate_motion(self):
        states = observations(); states['nominal']['position'][70, 0, 1] = .2
        result = audit_motion_similarity(states, NAMES)
        self.assertIn('nominal', result['distinct_motion_representatives'])
        pair = result['pairwise_motion_comparisons'][0]
        self.assertEqual(pair['metrics']['last_position_difference_m'], 0.)
        self.assertEqual(pair['metrics']['position_m'], .2)

    def test_speed_and_yaw_each_distinguish_motion(self):
        for alteration in ({'speed': .2}, {'yaw': 1.}):
            states = observations(); change_future(states['nominal'], **alteration)
            self.assertIn('nominal', audit_motion_similarity(states, NAMES)['distinct_motion_representatives'])

    def test_wrapped_yaw_comparison(self):
        states = observations()
        states['maintain']['rotation_deg'][:, 0, 2] = 179.98
        states['nominal']['rotation_deg'][:, 0, 2] = -179.98
        self.assertAlmostEqual(motion_metrics(states['maintain'], states['nominal'])['yaw_deg'], .04)

    def test_final_control_sentinel_excluded_but_t0_control_included(self):
        states = observations(); states['nominal']['control'][-1, 0, 0] = 1.
        self.assertTrue(motion_metrics(states['maintain'], states['nominal'])['identical_ego_control_sequence'])
        states['nominal']['control'][40, 0, 2] = 1.
        self.assertFalse(motion_metrics(states['maintain'], states['nominal'])['identical_ego_control_sequence'])

    def test_npc_changes_do_not_affect_selection(self):
        states = observations(); baseline = audit_motion_similarity(states, NAMES)
        states['nominal']['position'][41:, 1, 0] += 100.
        self.assertEqual(audit_motion_similarity(states, NAMES)['distinct_motion_representatives'], baseline['distinct_motion_representatives'])

    def test_repeat_calibrates_tolerance_only_within_cap(self):
        states = observations(); change_future(states['maintain_repeat'], position=.01)
        change_future(states['nominal'], position=.025)
        result = audit_motion_similarity(states, NAMES)
        self.assertAlmostEqual(result['tolerances']['effective']['position_m'], .03)
        self.assertIn('nominal', result['similar_motion_branches'])
        change_future(states['maintain_repeat'], position=.01)
        result = audit_motion_similarity(states, NAMES)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['distinct_motion_representatives'], [])
        self.assertIsNone(result['equivalent_motion_count'])

    def test_common_t0_mismatch_fails_even_with_distinct_futures(self):
        states = observations(); states['nominal']['position'][40, 0, 0] += .1
        self.assertEqual(audit_motion_similarity(states, NAMES)['status'], 'failed')

    def test_complete_link_prevents_transitive_merging(self):
        states = observations()
        change_future(states['nominal'], position=.015)
        change_future(states['stop_hold'], position=.03)
        result = audit_motion_similarity(states, NAMES)
        self.assertEqual(result['distinct_motion_representatives'], ['maintain', 'stop_hold'])
        self.assertEqual(result['similar_motion_branches']['nominal']['representative'], 'maintain')
        self.assertNotIn('stop_hold', result['duplicate_clusters'][0])

    def test_maintain_wins_even_when_config_order_differs(self):
        names = NAMES[1:]+NAMES[:1]
        self.assertEqual(audit_motion_similarity(observations(names), names)['distinct_motion_representatives'], ['maintain'])

    def test_nonfinite_and_missing_repeat_and_timeline_mismatch_rejected(self):
        states = observations(); states['nominal']['position'][80, 0, 0] = np.nan
        with self.assertRaises(ValueError): audit_motion_similarity(states, NAMES)
        states = observations(); del states['maintain_repeat']
        with self.assertRaises(ValueError): audit_motion_similarity(states, NAMES)
        states = observations(); states['nominal']['time_s'][80] += .001
        with self.assertRaises(ValueError): audit_motion_similarity(states, NAMES)

    def test_future_after_contact_is_not_truncated(self):
        states = observations(); states['nominal']['position'][120, 0, 2] = .4
        self.assertIn('nominal', audit_motion_similarity(states, NAMES)['distinct_motion_representatives'])

    def test_actual_t0_speed_and_ideal_profile_are_separate(self):
        state = observations()['maintain']; state['velocity'][:, 0, 0] = 8.
        state['velocity'][80:, 0, 0] = 0.
        profile = {'kind': 'longitudinal', 'acceleration_m_s2': -1.2, 'speed_increase_cap_m_s': 0.}
        report = motion_summary(state, profile=profile)
        self.assertEqual(report['controller_t0_speed_xy_m_s'], 8.)
        self.assertAlmostEqual(report['ideal_command_diagnostic']['target_final_speed_m_s'], 3.2)
        self.assertEqual(report['final_speed_m_s'], 0.)
        self.assertAlmostEqual(report['first_stationary_time_s'], 2.)
        self.assertAlmostEqual(report['final_continuous_stationary_s'], 2.)
        self.assertAlmostEqual(report['travelled_distance_3d_m'], 12.)

    def test_lane_direction_not_inferred_from_branch_label(self):
        cfg = lane_config()
        self.assertEqual(lane_direction(cfg, 'left_delayed'), 'left')
        self.assertEqual(lateral_branches(cfg), ('left', 'left_delayed'))
        cfg['action']['profiles']['left']['kind'] = 'maintain'
        self.assertIsNone(lane_direction(cfg, 'left'))

    def test_delayed_left_uses_same_target_and_completion_standard(self):
        cfg = lane_config(); state = observations()['maintain']
        u = np.clip((state['time_s']-.8)/2.5, 0., 1.)
        state['position'][:, 0, 1] = -3.5*u
        state['bbox_extent'][0] = [.8, .5, .5]
        state['transform_world_vehicle'][..., :3, 3] = state['position']
        delayed = trajectory_audit(state, cfg, 'left_delayed')
        direct = trajectory_audit(state, cfg, 'left')
        self.assertTrue(delayed['completed'])
        self.assertEqual(delayed['direction'], 'left')
        self.assertEqual(delayed['frames'], direct['frames'])
        self.assertEqual(delayed['thresholds'], direct['thresholds'])

    def test_dataset_cli_reports_selection_and_identity_failures(self):
        names = ['maintain', 'brake', 'accelerate', 'left', 'right']
        cfg = {'vehicle_roles': ['ego', 'rear'], 'vehicles': [{'role': 'ego'}, {'role': 'rear'}],
               'action': {}, 'spec_sha256': 'test-config', 'code_sha256': {'test': 'test-source'}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data = root/'data'; scene = data/'scene_0000'
            scene.mkdir(parents=True)
            (data/'manifest.json').write_text(json.dumps({'scene_ids': ['scene_0000'],
                'scene_branches': {'scene_0000': names+['maintain_repeat']}}))
            (scene/'spec.json').write_text(json.dumps(cfg))
            for name, state in observations(names).items():
                folder = scene/'branches'/name; folder.mkdir(parents=True)
                np.savez_compressed(folder/'states.npz', **state)
                (folder/'meta.json').write_text(json.dumps({'status': 'complete', 'branch': name,
                    'vehicle_roles': cfg['vehicle_roles'], 'spec_sha256': cfg['spec_sha256'], 'code_sha256': cfg['code_sha256']}))
            output = root/'report.json'
            command = [sys.executable, '-B', str(Path(__file__).with_name('audit_realized_actions.py')), '--data', str(data), '--out', str(output)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report['scenes']['scene_0000']['distinct_motion_representatives'], ['maintain'])
            meta = scene/'branches/maintain/meta.json'; body = json.loads(meta.read_text()); body['spec_sha256'] = 'wrong'; meta.write_text(json.dumps(body))
            self.assertEqual(audit_dataset(data)['status'], 'failed')

    def test_v2_requires_actual_controller_diagnostics_and_validates_policy_identity(self):
        cfg = lane_config(); cfg.update(spec_sha256='fixture-config', code_sha256={'fixture': 'source'})
        names = cfg['branches'][:-1]
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary)/'scene_0000'; scene.mkdir()
            (scene/'spec.json').write_text(json.dumps(cfg))
            for name, state in observations(names).items():
                folder = scene/'branches'/name; folder.mkdir(parents=True)
                np.savez_compressed(folder/'states.npz', **state)
                (folder/'meta.json').write_text(json.dumps({'status': 'complete', 'branch': name,
                    'vehicle_roles': cfg['vehicle_roles'], 'spec_sha256': cfg['spec_sha256'], 'code_sha256': cfg['code_sha256']}))
                label = 'maintain' if name == 'maintain_repeat' else name
                rows = [[{'role': 'ego', 'branch': label, 'kind': cfg['action']['profiles'][label]['kind'],
                          'phase': 'common_history' if time < 0 else 'active'}] for time in state['time_s']]
                (folder/'controller_diagnostics.json').write_text(json.dumps(rows))
            result = audit_scene(scene)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['ego_decision_diagnostics']['left_delayed']['phase_transitions'], [{'time_s': 0., 'phase': 'active'}])
            target = scene/'branches/left_delayed/controller_diagnostics.json'
            rows = json.loads(target.read_text()); rows[40][0]['kind'] = 'maintain'; target.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'policy-kind mismatch'): audit_scene(scene)
            target.unlink()
            with self.assertRaisesRegex(ValueError, 'requires controller diagnostics'): audit_scene(scene)


if __name__ == '__main__':
    unittest.main(verbosity=2)
