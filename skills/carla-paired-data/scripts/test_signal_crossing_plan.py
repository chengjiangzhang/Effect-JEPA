"""Signal planning boundary tests; synthetic routes, no RPC or native evidence."""
import copy
import unittest
from pathlib import Path

from action_opportunity import analyze_opportunity
from interaction_geometry import signal_crossing_plan, waiting_position_check
from runtime.action_separation import CONTRACT
from scene_compiler import Catalog, eligible_candidates


STYLE = {'max_acceleration': 1.5, 'minimum_gap': 3., 'desired_speed_multiplier': .9}


def candidate(ego_line=80., npc_line=91.):
    return {'candidate_id': 'cross_signalized-unit', 'map': 'unit',
            'ego_conflict_s': 100., 'npc_conflict_s': 100.,
            'ego_route': [[0., 0., 0., 0.], [200., 0., 0., 0.]],
            'npc_route': [[100., -100., 0., 90.], [100., 100., 0., 90.]],
            'signal_approaches': {'ego': {'stop_route_s_m': ego_line},
                                  'npc': {'stop_route_s_m': npc_line}}}


class SignalCrossingPlanTests(unittest.TestCase):
    def test_current_speed_history_stopping_and_opportunity_contracts_agree(self):
        c = candidate(); p = signal_crossing_plan(c, 1., STYLE)
        self.assertEqual((CONTRACT['history_s'], CONTRACT['future_s'], p['physics_dt_s']), (2., 4., .05))
        self.assertEqual(p['npc_green_time_s'], -.1)
        self.assertEqual(p['npc_initial_speed_m_s'], 0.)
        self.assertFalse(p['initial_entry_legality_observed'])
        for speed in p['ego_initial_speed_candidates_m_s']:
            self.assertGreaterEqual(speed, CONTRACT['minimum_measured_t0_speed_m_s'] + 4.5 * .05)
            station = p['ego_spawn_s'] + 3. * speed
            braking = speed * speed / (2. * p['ego_design_deceleration_m_s2'])
            response = speed * p['ego_stop_response_time_s']
            self.assertGreaterEqual(p['ego_waiting_geometry']['stop_route_s'] - station + 1e-8,
                                    braking + response + p['ego_control_stop_margin_m'] + speed * .05)
        spec = {'action_separation_contract': copy.deepcopy(CONTRACT)}
        result = analyze_opportunity(spec, {'ego': {'speed_m_s': min(p['ego_initial_speed_candidates_m_s'])}},
            {'cross': {'participants': [{'already_cleared_at_t0': False,
                                        'optimistic_entry_s': p['npc_optimistic_body_entry_s']}]}})
        self.assertTrue(result['passed'], result['failure_reasons'])
        self.assertLessEqual(p['npc_optimistic_body_entry_tick_s'], 2.9 + 1e-8)

    def test_low_speed_only_geometry_is_filtered_before_collection(self):
        with self.assertRaisesRegex(ValueError, 'distinct-action minimum'):
            signal_crossing_plan(candidate(ego_line=84.06), 1., STYLE)

    def test_two_tick_planning_reserve_boundary_is_closed_at_two_point_nine(self):
        # A stationary participant at a=1.5 can cover 6.75m in 3s. At this
        # mapped line its conservative body-front distance is exactly 6.75m.
        c = candidate(npc_line=89.75)
        p = signal_crossing_plan(c, 1., STYLE)
        self.assertAlmostEqual(p['npc_optimistic_body_entry_after_green_s'], 3.)
        self.assertAlmostEqual(p['npc_optimistic_body_entry_s'], 2.9)
        self.assertAlmostEqual(p['npc_optimistic_body_entry_tick_s'], 2.9)
        c['signal_approaches']['npc']['stop_route_s_m'] -= .01
        with self.assertRaisesRegex(ValueError, 'one second remaining'):
            signal_crossing_plan(c, 1., STYLE)

    def test_four_second_reach_is_insufficient_for_current_response_contract(self):
        with self.assertRaisesRegex(ValueError, 'one second remaining'):
            signal_crossing_plan(candidate(npc_line=89.5), 1., STYLE)

    def test_body_front_poses_are_not_confused_with_vehicle_centres(self):
        c = candidate(); p = signal_crossing_plan(c, 1., STYLE)
        self.assertAlmostEqual(p['npc_spawn_s'] + p['conservative_body_half_length_m'], 91.)
        self.assertAlmostEqual(p['ego_spawn_s'] + p['native_v2_front_lower_bound_m'] - 80.,
                               p['ego_committed_front_margin_m'])
        cfg = {'vehicles': [{'role': 'ego', 'route': c['ego_route']},
                            {'role': 'interaction_vehicle', 'route': c['npc_route']}]}
        self.assertTrue(waiting_position_check(cfg, p['ego_waiting_geometry']['stop_route_s'],
                                              ['interaction_vehicle'])['clear'])
        # Following distance still belongs to the unchanged live IDM policy;
        # it must not move a stationary front away from a signal boundary.
        other = dict(STYLE, minimum_gap=10.)
        self.assertEqual(p['npc_spawn_s'], signal_crossing_plan(c, 1., other)['npc_spawn_s'])

    def test_speed_scale_cannot_scale_below_the_absolute_action_minimum(self):
        p = signal_crossing_plan(candidate(), .85, STYLE)
        self.assertTrue(all(v >= 2.5 for v in p['ego_initial_speed_candidates_m_s']))

    def test_shortlist_excludes_low_speed_and_late_target_candidates(self):
        good = candidate(); slow = candidate(ego_line=84.06); late = candidate(npc_line=89.5)
        slow['candidate_id'] = 'slow'; late['candidate_id'] = 'late'
        catalog = Catalog(Path('/unused'), {}, {'cross_signalized': [slow, late, good]}, [], {})
        cell = {'template': 'cross_signalized', 'npc_style_parameters': STYLE}
        self.assertEqual(eligible_candidates(catalog, cell), [good])
        catalog.candidates['cross_signalized'] = [slow, late]
        with self.assertRaisesRegex(ValueError, 'three-second target-entry'):
            eligible_candidates(catalog, cell)


if __name__ == '__main__':
    unittest.main(verbosity=2)
