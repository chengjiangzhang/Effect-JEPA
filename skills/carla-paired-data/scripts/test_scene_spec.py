"""Offline authored-spec rejection and resolution invariants, without CARLA server."""
import copy
import unittest
import numpy as np
from scene_compiler import runtime_modules
from scene_spec import authoring_payload,authoring_schema,apply_authoring
from action_design import configure_actions,profiles_for_template
from model_api import SchemaError


def fixture():
    g,_=runtime_modules()
    route=[[float(x),0.,0.,0.] for x in range(400)]
    actors=[]
    for role,station in [('ego',100.),('follower',85.)]:
        actors.append({'role':role,'route':route,'spawn_route_s':station,
            'transform':g.point_at_s(route,station),'initial_speed_mps':6.,
            'desired_speed_mps':6.,'blueprint_id':'vehicle.tesla.model3',
            'blueprint_attributes':{'color':'0,0,0'}})
    cfg=g.finish_config('rear_following',0,17,actors,
        {'map':'Town04','origin':[100.,0.,0.],'conflicts':[],
         'traffic_design':{'roles_and_slots':[]}},2.,1.)
    configure_actions(cfg,'follower_response_straight',{},g)
    cfg['controller']['leader_geometry']='route_projection'
    return cfg,g


class SpecTests(unittest.TestCase):
    def joint_pedestrian_fixture(self):
        cfg,g=fixture()
        cfg['vehicles'][0].update(initial_speed_mps=3.4,desired_speed_mps=3.4)
        cfg['pedestrians']=[{}]
        cfg['interaction_design']={'requested_phase':'entering',
            'pedestrian_encounter':{'version':'body-relative-arrival-v2'}}
        cfg['action']['conflicts']=[{'id':'pedestrian','ego_entry_s':127.2,'ego_exit_s':131.2,
            'participants':[{'role':'walker','entry_s':2.,'exit_s':5.}]}]
        cfg['action']['profiles']['accelerate'].update(
            acceleration_m_s2=3.5,onset_s=.1,speed_increase_cap_m_s=5.)
        cfg['geometry']['traffic_design']['interaction_opportunity']={
            'estimated_t0_route_s':110.2,'estimated_t0_speed_m_s':3.4}
        return cfg,g

    def test_joint_entry_rechecks_weakened_authored_profile(self):
        cfg,g=self.joint_pedestrian_fixture()
        apply_authoring(cfg,authoring_payload(cfg),g)
        evidence=cfg['geometry']['traffic_design']['interaction_opportunity']['strategy_entry_opportunity']
        self.assertTrue(evidence['passed']);self.assertTrue(evidence['uses_final_authored_profiles'])
        p=authoring_payload(cfg)
        p['profiles']['accelerate'].update(acceleration_m_s2=.1,onset_s=2.9,speed_increase_cap_m_s=.1)
        with self.assertRaisesRegex(ValueError,'one second remaining'):
            apply_authoring(cfg,p,g)

    def test_joint_entry_recomputes_authored_position_speed_and_fork(self):
        for changed in ('station','speed','fork'):
            cfg,g=self.joint_pedestrian_fixture();p=authoring_payload(cfg)
            if changed=='station':p['actors']['ego']['spawn_route_s']+=1.
            if changed=='speed':p['actors']['ego']['initial_speed_mps']+=.1
            if changed=='fork':p['fork_time_s']+=.25
            apply_authoring(cfg,p,g)
            evidence=cfg['geometry']['traffic_design']['interaction_opportunity']['strategy_entry_opportunity']
            expected=p['actors']['ego']['spawn_route_s']+p['fork_time_s']*p['actors']['ego']['initial_speed_mps']
            self.assertAlmostEqual(evidence['estimated_t0_route_s'],expected)
            self.assertAlmostEqual(evidence['distance_to_entry_m'],127.2-expected)
            self.assertEqual(evidence['profiles'][0]['frozen_profile'],cfg['action']['profiles']['accelerate'])
        cfg,g=self.joint_pedestrian_fixture();p=authoring_payload(cfg)
        p['actors']['ego']['spawn_route_s']-=.5
        with self.assertRaisesRegex(ValueError,'one second remaining'):
            apply_authoring(cfg,p,g)

    def test_stopping_deadline_and_physical_margin_survive_authoring(self):
        cfg,g=fixture()
        cfg['vehicles'][1]['route']=[[155.,float(x-155),0.,90.] for x in range(400)]
        cfg['vehicles'][1]['transform']=g.point_at_s(cfg['vehicles'][1]['route'],85.)
        conflict={'id':'crossing','stop_route_s':150.,'ego_entry_s':155.,'ego_exit_s':160.,
                  'participants':[{'role':'follower','entry_s':150.,'exit_s':160.}]}
        cfg['action']['conflicts']=[conflict]
        cfg['action']['profiles']=profiles_for_template('cross_unsignalized',[],conflict)
        cfg['action']['profile_order']=list(cfg['action']['profiles'])
        p=authoring_payload(cfg);apply_authoring(cfg,p,g)
        budget=cfg['geometry']['traffic_design']['authored_initial_conditions']['stop_budgets']['stop_hold']
        self.assertEqual(budget['stop_deadline_s'],3.5)
        self.assertEqual(budget['control_stop_margin_m'],.75)
        self.assertAlmostEqual(budget['required_distance_m'],6.*.75+36./9.+.75)
        # Individually bounded numbers may still define an impossible timing
        # combination. The complete profile must be validated after authoring.
        p['profiles']['stop_hold']['onset_s']=3.
        with self.assertRaisesRegex(ValueError,'no braking time'):apply_authoring(cfg,p,g)

    def test_agent_cannot_remove_waiting_targets_or_enable_timed_stop_release(self):
        cfg,g=fixture()
        conflict={'id':'crossing','stop_route_s':150.,'ego_entry_s':155.,'ego_exit_s':160.,
                  'participants':[{'role':'follower','entry_s':150.,'exit_s':160.}]}
        cfg['action']['conflicts']=[conflict]
        cfg['action']['profiles']=profiles_for_template('cross_unsignalized',[],conflict)
        cfg['action']['profile_order']=list(cfg['action']['profiles'])
        original=authoring_payload(cfg)
        for name,field,value in [('yield_wait','wait_for_roles',[]),
                                 ('yield_wait','release_condition','gap_only'),
                                 ('stop_hold','hold_policy','timed'),
                                 ('stop_hold','hold_until_s',3.)]:
            p=copy.deepcopy(original);p['profiles'][name][field]=value
            with self.subTest(field=field),self.assertRaises(SchemaError):
                apply_authoring(copy.deepcopy(cfg),p,g)

    def test_authored_lane_settling_cannot_consume_required_hold(self):
        cfg,g=fixture()
        cfg['action']['available_lane_change_directions']=['left']
        cfg['action']['profiles']=profiles_for_template('ego_lane_change_straight',['left'],None)
        cfg['action']['profile_order']=list(cfg['action']['profiles'])
        p=authoring_payload(cfg);apply_authoring(cfg,p,g)
        self.assertEqual(cfg['action']['profiles']['left_delayed']['required_final_hold_s'],.5)
        p['profiles']['left_delayed']['settling_time_s']=1.
        with self.assertRaisesRegex(ValueError,'settling'):apply_authoring(cfg,p,g)

    def test_full_spec_changes_actual_pose_speed_policy_and_fork(self):
        cfg,g=fixture();p=authoring_payload(cfg)
        p['actors']['follower'].update(spawn_route_s=80.,initial_speed_mps=7.)
        p['actors']['follower']['npc_policy']['idm']['time_headway']=2.
        p['fork_time_s']=3.5;p['profiles']['brake']['onset_s']=.4
        p['unrelated_roles']=['follower']
        apply_authoring(cfg,p,g)
        self.assertEqual(cfg['vehicles'][1]['transform']['position'][0],80.)
        self.assertEqual(cfg['vehicles'][1]['initial_speed_mps'],7.)
        self.assertEqual(cfg['vehicles'][1]['npc_policy']['idm']['time_headway'],2.)
        self.assertEqual(cfg['driving_warmup_s'],1.5)
        self.assertEqual(cfg['action']['profiles']['brake']['onset_s'],.4)
        self.assertEqual(cfg['response_contract']['unrelated_roles'],['follower'])
        self.assertEqual(cfg['response_contract']['unrelated_position_tolerance_m'],0.)
        self.assertFalse(cfg['authoring_contract']['npc_no_reaction_is_failure'])

    def test_same_inputs_have_same_map_derived_world_pose(self):
        cfg,g=fixture();other=copy.deepcopy(cfg);p=authoring_payload(cfg)
        apply_authoring(cfg,p,g);apply_authoring(other,p,g)
        self.assertEqual(cfg,other)

    def test_overlap_is_rejected_after_authoring(self):
        cfg,g=fixture();p=authoring_payload(cfg);p['actors']['follower']['spawn_route_s']=100.
        with self.assertRaisesRegex(ValueError,'overlap'):apply_authoring(cfg,p,g)

    def test_no_unknown_policies_thresholds_roles_or_world_commands(self):
        for change in ('policy','threshold','role','xyz'):
            cfg,g=fixture();p=authoring_payload(cfg)
            if change=='policy':p['actors']['follower']['npc_policy']['read_future']=True
            elif change=='threshold':p['execution_contract']={'speed_tracking_tolerance_m_s':100}
            elif change=='role':del p['actors']['follower']
            else:p['actors']['ego']['world_xyz']=[1,2,3]
            with self.subTest(change=change),self.assertRaises((SchemaError,ValueError)):apply_authoring(cfg,p,g)

    def test_agent_may_not_replace_mapped_maneuver_kind(self):
        cfg,g=fixture();p=authoring_payload(cfg);p['profiles']['brake']['kind']='lane_change'
        with self.assertRaises(SchemaError):apply_authoring(cfg,p,g)

    def test_inert_reference_acceleration_is_rejected(self):
        cfg,g=fixture();p=authoring_payload(cfg)
        p['profiles']['maintain']['acceleration_m_s2']=4.
        p['profiles']['maintain']['speed_increase_cap_m_s']=10.
        with self.assertRaises(SchemaError):apply_authoring(cfg,p,g)

    def test_fork_requires_actual_physics_tick(self):
        cfg,g=fixture();p=authoring_payload(cfg);p['fork_time_s']=3.01
        with self.assertRaisesRegex(ValueError,'20 Hz'):apply_authoring(cfg,p,g)

    def test_same_future_motion_does_not_delete_branches(self):
        cfg,g=fixture();apply_authoring(cfg,authoring_payload(cfg),g)
        self.assertEqual(len(cfg['action']['profiles']),5)
        self.assertFalse(cfg['action']['design_evidence']['actual_trajectory_dedup_required'])


if __name__=='__main__':unittest.main(verbosity=2)
