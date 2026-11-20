import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from action_design import turn_speed_plan
from encounter_calibration import _entry_opportunity, _distance_at_eta, _fast_eta, _measured_second_step
from walker_spawn import blueprint_for, check_spec, crossing_path, direction_for, feasible_sites


class CandidateRegressionTests(unittest.TestCase):
    def test_inverse_matches_earliest_profile_with_onset_and_cap(self):
        spec = {'action': {'profiles': {'one': {'acceleration_m_s2': 2., 'onset_s': .3,
                        'speed_increase_cap_m_s': 1.}, 'two': {'acceleration_m_s2': 3., 'onset_s': 1.}}}}
        for t in (.3, .87, 2.15, 3.41, 7.83):
            distance = _distance_at_eta(spec, 3., t)
            self.assertLessEqual(abs(_fast_eta(spec, 3., distance)-t), .050001)
        self.assertGreater(_distance_at_eta(spec, 3., 3.41), 3.*3.41)

    def test_constant_speed_and_common_history_bounds(self):
        self.assertEqual(_distance_at_eta({'action':{'profiles':{}}}, 4., 2.), 8.)
        row = {'error_s': -.5}
        target = {'kind':'pedestrian_clearance','role':'person'}
        revisions = {'layout_revisions':[{'applied_route_shifts_m':{'ego':-4.},
                't0_evidence':{'rows':[{**target,'error_s':-1.5}]}}]}
        self.assertEqual(_measured_second_step(revisions,target,'ego',row,-3.),-2.)
        self.assertEqual(_measured_second_step(revisions,target,'ego',{'error_s':-2.},-3.),-3.)

    def test_early_entry_opportunity_keeps_arrival_difference_and_waits(self):
        spec={'encounter_calibration':{'max_shift_per_revision_m':8.}}
        actors={'ego':{'route_speed_m_s':4.}, 'lead':{'route_speed_m_s':5.,'actor_type':'vehicle'}}
        part={'role':'lead','already_cleared_at_t0':False,'optimistic_entry_s':3.7,
              'maximum_acceleration_m_s2':1.,'speed_cap_m_s':5.,'front_to_entry_m':18.5}
        conflicts={'zone':{'participants':[part]}}
        rows=[{'kind':'arrival_difference','role':'lead','error_s':0.,'tolerance_s':.45}]
        steps=[]
        _entry_opportunity(spec,actors,conflicts,rows,steps)
        shifts={s['role']:s['route_shift_m'] for s in steps}
        self.assertAlmostEqual(shifts['lead'],3.75)
        self.assertAlmostEqual(shifts['ego']/4.,shifts['lead']/5.)
        tight = {'hold':{'stop_budget':{'available_distance_m':5.5,'required_response_braking_margin_m':5.}}}
        rows=[{'kind':'arrival_difference','role':'lead','error_s':0.,'tolerance_s':.45}]
        _entry_opportunity(spec,actors,conflicts,rows,steps:=[],branches=tight)
        self.assertFalse(steps)
        self.assertEqual(rows[-1]['classification'],'no_joint_translation_at_observed_speeds')
        tight['hold']['stop_budget']['available_distance_m']=4.5
        _entry_opportunity(spec,actors,conflicts,[],steps,branches=tight)
        self.assertLess(next(i['route_shift_m'] for i in steps if i['role']=='ego'),-.5)
        part['optimistic_entry_s']=2.9
        _entry_opportunity(spec,actors,conflicts,[],steps:=[])
        self.assertFalse(steps)
        part['optimistic_entry_s']=None
        actors['lead']['route_speed_m_s']=0.
        _entry_opportunity(spec,actors,conflicts,[],steps)
        self.assertFalse(steps)
        part['already_cleared_at_t0']=True
        actors['lead']['route_speed_m_s']=5.
        _entry_opportunity(spec,actors,conflicts,[],steps)
        self.assertFalse(steps)

    def test_arrival_adjustment_preserves_existing_observation_opportunity(self):
        spec = {'encounter_calibration': {'max_shift_per_revision_m': 8.}}
        actors = {'ego': {'route_speed_m_s': 3.}, 'target': {'actor_type': 'vehicle', 'route_speed_m_s': 6.}}
        target = {'role':'target','already_cleared_at_t0':False,'optimistic_entry_s':2.5,
                  'maximum_acceleration_m_s2':0.,'speed_cap_m_s':6.,'front_to_entry_m':15.}
        rows = [{'kind':'arrival_difference','role':'target','error_s':-1.,'tolerance_s':.45}]
        steps = [{'role':'target','route_shift_m':-6.,'basis':'arrival_difference'}]
        _entry_opportunity(spec,actors,{'lane':{'participants':[target]}},rows,steps)
        shifts = {step['role']:step['route_shift_m'] for step in steps}
        self.assertAlmostEqual(shifts['target'], 15.-2.95*6.)
        self.assertAlmostEqual(-1. - shifts['target']/6. + shifts['ego']/3., 0.)
        self.assertEqual(rows[-1]['classification'],'preserve_entry_during_arrival_adjustment')

    def test_entry_second_step_uses_bounded_observed_displacement(self):
        z={'kind':'observation_window','role':'oncoming','estimated_approach_shift_m':2.}
        prev={**z,'estimated_approach_shift_m':3.}
        history={'layout_revisions':[{'applied_route_shifts_m':{'oncoming':3.},
                                     't0_evidence':{'rows':[prev]}}]}
        self.assertEqual(_measured_second_step(history,z,'oncoming',z,2.),4.)
        self.assertEqual(z['correction_method'],'bounded_measured_entry_response')

    def test_pose_and_blueprint_checked_before_location_preference(self):
        ends=[{'position':[0.,0.,0.],'width_m':2.},{'position':[0.,6.,0.],'width_m':2.}]
        self.assertEqual(direction_for(ends,[0.,0.,0.],True),0)
        self.assertEqual(direction_for(ends,[0.,4.,0.],True),1)
        pts,pose,_=crossing_path(ends,0)
        self.assertAlmostEqual(pose['position'][2]-pts[0][2],.8)
        model=blueprint_for(11)
        a,b=[{'candidate_id':k,'pedestrian_endpoints':ends,'origin':[0.,0.,0.]} for k in ('blocked','clear')]
        cat=SimpleNamespace(candidates={'pedestrian_straight':[a,b]},walker_spawns={
            (k,0,model): {'spawned': k=='clear','candidate_id':k,'direction':0,'blueprint_id':model,
                          'transform':pose, 'file':'map.json','sha256':'example'} for k in ('clear','blocked')})
        with patch('scene_compiler.validate_selection',return_value={'layout_seed':11}), \
             patch('scene_randomization.intent',return_value={'participants':{'preferred_pedestrian_direction':1}}):
            self.assertEqual(feasible_sites([a,b],cat,{}, {'template':'pedestrian_straight'},0),[b])
        spec={'grid_spec':{'candidate_id':'clear'},'pedestrians':[{'crossing_direction_index':0,
             'blueprint_id':model,'transform':pose}]}
        self.assertEqual(len(check_spec(spec,cat.walker_spawns)),1)
        spec['pedestrians'][0]['transform']={**pose,'position':[0.,1.,.8]}
        with self.assertRaises(ValueError): check_spec(spec,cat.walker_spawns)

    def test_final_turn_timing_retains_sampled_delay_and_firm_stop(self):
        from interaction_design import action_timing, PHASES
        cfg={'dt_s':.05,'action':{'profiles':{'stop_hold':{'kind':'stop_hold','response_time_s':.75}}},
             'geometry':{'traffic_design':{'turn_speed_plan':{'approach_m_s':3.68,'proposed_approach_m_s':4.39}}},
             'randomization':{'parameters':{'stop_delay_s':.4}}}
        action_timing(cfg,'left_unsignalized',PHASES['left_unsignalized'][0])
        profile=cfg['action']['profiles']['stop_hold']
        self.assertAlmostEqual(profile['onset_s'],.4)
        self.assertAlmostEqual(profile['stop_deadline_s'],.4+.75+1.05)

    def test_left_turn_budget_varies_with_curvature(self):
        straight=np.column_stack((np.arange(200.),np.zeros(200)))
        self.assertAlmostEqual(turn_speed_plan(straight,45.,2.,6.)['commit_increase_cap_m_s'],5.)
        angle=np.linspace(0,2*math.pi,315)
        curved=np.column_stack((15*np.cos(angle),15*np.sin(angle)))
        plan=turn_speed_plan(curved,40.,2.,7.4)
        self.assertAlmostEqual(plan['maximum_curvature_per_m'],1./15,delta=.003)
        self.assertLess(plan['approach_m_s'],7.4)
        self.assertLessEqual((plan['approach_m_s']+plan['commit_increase_cap_m_s'])**2 *
                             plan['maximum_curvature_per_m'],3.5)
        self.assertGreater(plan['commit_increase_cap_m_s'],1.)

if __name__=='__main__':
    unittest.main()
