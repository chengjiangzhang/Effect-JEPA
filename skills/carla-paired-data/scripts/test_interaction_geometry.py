"""Whole-body waiting geometry and frozen pedestrian feedback, no simulator."""
import copy
import unittest

import numpy as np

from action_design import configure_actions, configure_pedestrian_feedback
from interaction_geometry import (WALKER_BLUEPRINT_CANDIDATES, pedestrian_config,
                                  route_body_samples, safe_waiting_station,
                                  waiting_position_check)
from runtime.pedestrian_policy import validate_policy
from scene_compiler import runtime_modules


def crossing_fixture():
    return {'vehicles':[
        {'role':'ego','route':[[0.,0.,0.,0.],[100.,0.,0.,0.]]},
        {'role':'crossing','route':[[10.,-40.,0.,90.],[10.,40.,0.,90.]]}]}


class WaitingGeometryTests(unittest.TestCase):
    def test_center_before_crossing_does_not_mean_whole_body_clear(self):
        cfg=crossing_fixture()
        self.assertFalse(waiting_position_check(cfg,8.,['crossing'])['clear'])
        self.assertTrue(waiting_position_check(cfg,4.,['crossing'])['clear'])

    def test_wait_station_moves_upstream_without_changing_npc_or_route(self):
        cfg=crossing_fixture();before=copy.deepcopy(cfg)
        result=safe_waiting_station(cfg,8.,['crossing'])
        self.assertTrue(result['clear']);self.assertLess(result['stop_route_s'],8.)
        self.assertGreater(result['upstream_adjustment_m'],0.)
        self.assertEqual(cfg,before)

    def test_parallel_route_overlap_cannot_be_relabelled_clear_by_no_eta(self):
        cfg=crossing_fixture();cfg['vehicles'][1]['route']=copy.deepcopy(cfg['vehicles'][0]['route'])
        with self.assertRaisesRegex(ValueError,'No upstream waiting pose'):
            safe_waiting_station(cfg,8.,['crossing'])

    def test_actual_bbox_offset_changes_waiting_clearance(self):
        cfg=crossing_fixture()
        actors={role:{'bbox_extent_m':[1.5,.5,.8],'bbox_offset_m':[0.,0.,0.],
                      'bbox_rotation_deg':[0.,0.,0.]} for role in ('ego','crossing')}
        self.assertTrue(waiting_position_check(cfg,6.,['crossing'],measured_actors=actors)['clear'])
        actors['ego']['bbox_offset_m'][0]=3.
        self.assertFalse(waiting_position_check(cfg,6.,['crossing'],measured_actors=actors)['clear'])

    def test_continuous_sample_cover_has_translation_and_rotation_margin(self):
        straight=route_body_samples([[0.,0.,0.,0.],[2.,0.,0.,0.]])
        turning=route_body_samples([[0.,0.,0.,0.],[2.,0.,0.,30.]])
        self.assertEqual(straight['sampling_guard_m'],.25)
        self.assertGreater(turning['sampling_guard_m'],straight['sampling_guard_m'])
        self.assertTrue(np.all(np.diff(turning['station'])<=.5+1e-9))
        self.assertTrue(np.all(straight['half'][:,1]==1.2))

    def test_distant_bend_does_not_inflate_straight_meeting_lane(self):
        cfg=crossing_fixture()
        cfg['vehicles'][1]['route']=[[100.,3.,0.,180.],[50.,3.,0.,180.],
                                    [0.,3.,0.,180.],[-2.,4.,0.,150.],[-3.,6.,0.,120.]]
        self.assertTrue(waiting_position_check(cfg,25.,['crossing'])['clear'])
        # Actual lane separation below the unchanged 0.5m clearance still fails.
        for row in cfg['vehicles'][1]['route']:row[1]-=.2
        self.assertFalse(waiting_position_check(cfg,25.,['crossing'])['clear'])

    def test_pedestrian_entire_route_is_checked_in_either_direction(self):
        cfg=crossing_fixture();cfg['pedestrians']=[{'role':'walker','route_points':[[10.,-5.,0.],[10.,5.,0.]]}]
        for reverse in (False,True):
            if reverse: cfg['pedestrians'][0]['route_points'].reverse()
            self.assertFalse(waiting_position_check(cfg,8.,['walker'])['clear'])
            self.assertTrue(waiting_position_check(cfg,5.,['walker'])['clear'])


class PedestrianConfigurationTests(unittest.TestCase):
    def test_same_seed_freezes_full_plan_but_cross_scene_actor_attributes_vary(self):
        geometry,_=runtime_modules()
        route=[[float(x),0.,0.,0.] for x in range(500)]
        candidate={'ego_route':route,'ego_conflict_s':200.,'map':'unit','origin':[200.,0.,0.],
                   'anchor':{'road_id':1},'evidence':{},'pedestrian_endpoints':[
                       {'position':[200.,-8.,0.]},{'position':[200.,8.,0.]}]}
        def make(seed):
            cfg,_=pedestrian_config(candidate,'pedestrian_straight',0,seed,np.random.default_rng(seed),1.,geometry)
            configure_actions(cfg,'pedestrian_straight',{},geometry)
            return cfg
        self.assertEqual(make(17),make(17))
        samples=[make(seed) for seed in range(24)]
        walkers=[c['pedestrians'][0] for c in samples]
        self.assertEqual({w['crossing_direction_index'] for w in walkers},{0})
        self.assertEqual({w['blueprint_id'] for w in walkers},set(WALKER_BLUEPRINT_CANDIDATES))
        self.assertEqual({w['start_time_s'] for w in walkers},{0.})
        self.assertGreater(len({w['speed_mps'] for w in walkers}),1)
        for cfg,w in zip(samples,walkers):
            validate_policy(w,cfg)
            self.assertTrue(w['reactive_to_ego'])
            self.assertFalse(w['blueprint_evidence']['runtime_availability_verified'])
            self.assertEqual({p['role'] for p in w['response_policy']['vehicle_conflicts']},set(cfg['vehicle_roles']))
            self.assertEqual(cfg['action']['profiles']['yield_wait']['wait_for_roles'],['crossing_pedestrian'])

    def test_continuous_crossings_include_adjacent_vehicles_but_exclude_overpass(self):
        geometry,_=runtime_modules()
        cfg={'vehicles':[
            {'role':'ego','route':[[0.,0.,0.,0.],[100.,0.,0.,0.]]},
            {'role':'adjacent','route':[[0.,3.5,0.,0.],[100.,3.5,0.,0.]]},
            {'role':'overpass','route':[[0.,0.,8.,0.],[100.,0.,8.,0.]]}],
            'pedestrians':[{'role':'walker','route_points':[[50.,-8.,0.],[50.,8.,0.]],
                            'start_time_s':.5,'speed_mps':1.5}]}
        configure_pedestrian_feedback(cfg,None,geometry)
        walker=cfg['pedestrians'][0];validate_policy(walker,cfg)
        self.assertEqual({p['role'] for p in walker['response_policy']['vehicle_conflicts']},{'ego','adjacent'})
        self.assertLessEqual(walker['response_policy']['pedestrian_entry_s'],8.)
        self.assertGreaterEqual(walker['response_policy']['pedestrian_exit_s'],11.5)
        bad=copy.deepcopy(walker);bad['response_policy']['vehicle_conflicts']=bad['response_policy']['vehicle_conflicts'][:1]
        with self.assertRaises(ValueError):validate_policy(bad,cfg)


if __name__=='__main__':
    unittest.main()
