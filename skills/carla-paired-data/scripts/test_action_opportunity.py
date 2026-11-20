"""Pre-intervention layout checks; never use observed future effect labels."""
import copy
import unittest
from action_opportunity import analyze_opportunity
from runtime.action_separation import CONTRACT


def fixture():
    person={'role':'walker','start_time_s':0.,'speed_mps':1.5,
        'response_policy':{'yielding_speed_m_s':2.,'pedestrian_entry_s':1.,'pedestrian_exit_s':4.,
            'clearance_time_s':1.,'clearance_distance_m':1.,
            'vehicle_conflicts':[{'role':'ego','entry_s':20.,'exit_s':24.},
                                 {'role':'follower','entry_s':20.,'exit_s':24.}]}}
    spec={'action_separation_contract':copy.deepcopy(CONTRACT),'pedestrians':[person],
          'action':{'profiles':{'stop':{'kind':'stop_hold','onset_s':0.,'stop_deadline_s':2.25,'response_time_s':.75}}}}
    actors={'ego':{'speed_m_s':5.},'walker':{'body_front_route_s_m':.3,'body_rear_route_s_m':-.3},
            'follower':{'route_speed_m_s':5.,'body_front_route_s_m':-30.,'body_rear_route_s_m':-35.}}
    conflicts={'c':{'participants':[{'already_cleared_at_t0':False,'optimistic_entry_s':.5}]}}
    return spec,actors,conflicts


class OpportunityTests(unittest.TestCase):
    def test_near_side_has_time_to_respond_with_shared_background_gap(self):
        result=analyze_opportunity(*fixture())
        self.assertTrue(result['passed'],result['failure_reasons'])
        self.assertEqual((result['history_s'],result['future_s']),(2.,4.))
        self.assertGreaterEqual(result['pedestrians'][0]['response_observation_remaining_s'],1.)
        self.assertFalse(result['future_response_used_as_gate'])

    def test_far_sidewalk_fails_before_any_future_rollout(self):
        spec,actors,conflicts=fixture();spec['pedestrians'][0]['response_policy']['pedestrian_entry_s']=7.
        result=analyze_opportunity(spec,actors,conflicts)
        self.assertIn('pedestrian_response_opportunity_too_late',[r['code'] for r in result['failure_reasons']])

    def test_moving_follower_cannot_be_ignored_to_claim_a_clear_crossing(self):
        spec,actors,conflicts=fixture();actors['follower']['body_front_route_s_m']=8.
        result=analyze_opportunity(spec,actors,conflicts)
        self.assertIn('background_traffic_blocks_designed_pedestrian_opportunity',[r['code'] for r in result['failure_reasons']])

    def test_stopped_t0_and_end_of_window_encounter_fail(self):
        spec,actors,conflicts=fixture();actors['ego']['speed_m_s']=0.
        conflicts['c']['participants'][0]['optimistic_entry_s']=3.8
        result=analyze_opportunity(spec,actors,conflicts)
        self.assertIn('measured_t0_speed_below_distinct_action_design_range',[r['code'] for r in result['failure_reasons']])
        self.assertIn('no_target_entry_opportunity_with_one_second_remaining',[r['code'] for r in result['failure_reasons']])


if __name__=='__main__':unittest.main(verbosity=2)
