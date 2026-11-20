"""Offline checks of explicit ego decisions; no simulator is started."""
from __future__ import annotations

import copy
import json
import unittest
import numpy as np

if __package__:
    from .controllers_v2 import (ScenarioController, branch_names, ego_decision_context,
        decision_conflict_observation, lane_change_route, select_action_profiles, npc_target,
        live_collision_constraint)
    from .ego_decisions import EgoDecisionState, validate_profiles, longitudinal_target, stop_target, deadline_stop_target
else:
    from controllers_v2 import (ScenarioController, branch_names, ego_decision_context,
        decision_conflict_observation, lane_change_route, select_action_profiles, npc_target,
        live_collision_constraint)
    from ego_decisions import EgoDecisionState, validate_profiles, longitudinal_target, stop_target, deadline_stop_target


def fixture():
    baseline = {'desired_speed_m_s': 8., 'idm': {'max_acceleration': 2., 'comfortable_brake': 2.8,
        'time_headway': 1.25, 'minimum_gap': 2.5}, 'release_gap_s': 1.5,
        'comfortable_deceleration_m_s2': 4.5, 'respect_signals': True}
    conflict = {'id': 'crossing', 'ego_entry_s': 20., 'ego_exit_s': 26., 'stop_route_s': 14.,
        'participants': [{'role': 'cross', 'entry_s': 28., 'exit_s': 34.}]}
    profiles = {
        'maintain': {'kind': 'maintain'},
        'decelerate': {'kind': 'longitudinal', 'onset_s': .5, 'acceleration_m_s2': -1.2,
                       'speed_increase_cap_m_s': 0., 'min_target_speed_m_s': 1.},
        'stop_hold': {'kind': 'stop_hold', 'stop_route_s': 14., 'hold_until_s': 4.},
        'yield_wait': {'kind': 'yield_wait', 'stop_route_s': 14., 'conflict_id': 'crossing',
                       'min_wait_s': .25, 'release_gap_s': 1.5, 'timeout_s': 4.},
        'commit_go': {'kind': 'commit_go', 'onset_s': 0., 'acceleration_m_s2': 1.5,
                      'speed_increase_cap_m_s': 5., 'respect_signals': False},
    }
    cfg = {'vehicle_roles': ['ego', 'cross'], 'vehicles': [
        {'role': 'ego', 'route': [[0.,0.,0.,0.], [100.,0.,0.,0.]], 'initial_speed_mps': 8., 'desired_speed_mps': 8.},
        {'role': 'cross', 'route': [[20.,-30.,0.,90.], [20.,70.,0.,90.]], 'initial_speed_mps': 4., 'desired_speed_mps': 4.}],
        'action': {'profile_version': 2, 'schema_version': 2, 'profiles': profiles,
            'profile_order': list(profiles), 'available_lane_change_directions': [], 'nominal_policy': baseline,
            'conflicts': [conflict], 'lateral_mode': 'adjacent_lane_route', 'lateral_onset_s': .2, 'lateral_duration_s': 3.},
        'controller': {'idm': copy.deepcopy(baseline['idm']), 'leader_geometry': 'route_projection',
            'lateral_margin_m': .3, 'conflict_time_gap_s': 2., 'yield_lookahead_s': 4.},
        'geometry': {'conflicts': []}}
    observed = {'position': np.array([[14.,0.,0.], [20.,0.,0.]]),
        'velocity': np.array([[0.,0.,0.], [0.,4.,0.]]),
        'rotation_deg': np.array([[0.,0.,0.], [0.,0.,90.]])}
    extents = np.array([[2.,1.,.8], [2.,1.,.8]])
    return cfg, observed, extents


def context(**changes):
    value = {'route_s': 14., 'speed': 0., 'nominal_target': 0., 'nominal_diagnostic': {},
        'stop_deceleration': 4.5, 'signal': None,
        'conflicts': {'crossing': {'occupied': True, 'next_arrival_s': 0.}}}
    value.update(changes)
    return value


class DecisionTests(unittest.TestCase):
    def test_distant_stop_is_capped_by_measured_speed_and_deadline(self):
        # Native scene_0002 had measured v0~=3 m/s despite authored 5.5 m/s;
        # its maximum stop line stayed 11 m away and spatial-only control
        # did not stop within four seconds. The line is an upper bound.
        cfg,_,_=fixture()
        p={**cfg['action']['profiles']['stop_hold'], 'stop_route_s':50.,
           'stop_deadline_s':3.5, 'stop_margin_m':.75, 'response_time_s':.75}
        state=EgoDecisionState()
        values=[]
        for t in (0., 1., 2., 2.75):
            target,hold,d=state.evaluate('stop_hold',p,t,3.,context(route_s=10.,speed=3.))
            values.append(target)
        self.assertEqual(values[0],3.)
        self.assertAlmostEqual(values[1],3.*(1.-1./2.75))
        self.assertGreater(values[1],values[2]);self.assertEqual(values[-1],0.)
        self.assertTrue(hold);self.assertEqual(d['target_zero_deadline_s'],2.75)
        self.assertEqual(d['remaining_to_stop_m'],40.)

    def test_deadline_stop_can_wait_and_release_upstream_without_npc_forcing(self):
        cfg,_,_=fixture()
        p={**cfg['action']['profiles']['yield_wait'], 'stop_deadline_s':3.5,
           'stop_margin_m':.75,'response_time_s':.75,'resume_speed_m_s':6.}
        state=EgoDecisionState()
        target,hold,d=state.evaluate('yield_wait',p,3.,3.,context(route_s=8.))
        self.assertEqual(target,0.);self.assertTrue(hold)
        self.assertEqual(d['stopped_at_s'],3.)
        self.assertTrue(d['stationary_before_requested_stop'])
        self.assertIsNone(d['released_at_s'])
        clear=context(route_s=8.,conflicts={'crossing':{'occupied':False,'next_arrival_s':None}})
        self.assertEqual(state.evaluate('yield_wait',p,3.1,3.,clear)[0],0.)
        target,hold,d=state.evaluate('yield_wait',p,3.3,3.,clear)
        self.assertEqual(target,6.);self.assertFalse(hold)
        self.assertEqual(d['phase'],'released');self.assertFalse(d['npc_policy_modified'])

    def test_margin_moves_control_target_not_audit_stop_line(self):
        cfg,_,_=fixture()
        p={**cfg['action']['profiles']['stop_hold'],'stop_margin_m':.75}
        target,_,d=EgoDecisionState().evaluate('stop_hold',p,0.,5.,context(route_s=10.,speed=5.))
        self.assertAlmostEqual(target,stop_target(10.,13.25,4.5,.75)[0])
        self.assertEqual(d['stop_route_s'],14.);self.assertEqual(d['control_stop_route_s'],13.25)
        self.assertEqual(d['remaining_to_stop_m'],4.)
        self.assertEqual(d['stop_station_overshoot_m'],0.)

    def test_deadline_budget_rejects_unphysical_requested_deceleration(self):
        with self.assertRaisesRegex(ValueError,'measured t0 speed'):
            deadline_stop_target(0.,0.,1.,8.,4.5,.75)
        self.assertEqual(deadline_stop_target(2.75,0.,3.5,3.,4.5,.75),0.)
        self.assertEqual(deadline_stop_target(.5,1.,3.5,3.,4.5,.75),3.)

    def test_stop_deadline_and_margin_validation(self):
        for patch in ({'stop_deadline_s':.5}, {'stop_deadline_s':4.1},
                      {'stop_deadline_s':3.,'onset_s':2.5}, {'stop_margin_m':-1.},
                      {'stop_margin_m':15.}):
            cfg,_,_=fixture();cfg['action']['profiles']['stop_hold'].update(patch)
            with self.subTest(patch=patch),self.assertRaises(ValueError):
                validate_profiles(cfg['action'])
        cfg,_,_=fixture();cfg['duration_s']=3.
        cfg['action']['profiles']['stop_hold']['stop_deadline_s']=3.5
        with self.assertRaisesRegex(ValueError,'future window'):ScenarioController(cfg)

    def test_stop_reserves_response_distance_before_old_late_braking_point(self):
        # Native failure: at t=1.0 s, remaining=2.8498 m and speed=4.10 m/s;
        # the old ideal envelope still requested cruise. It eventually stopped
        # 1.974 m beyond the line. The new target must already demand slowing.
        distance, speed = 2.8497835735405914, 4.100
        old, _ = stop_target(0., distance, 4.5, response_time_s=0.)
        anticipated, remaining = stop_target(0., distance, 4.5, response_time_s=.75)
        self.assertGreater(old, speed)
        self.assertLess(anticipated, speed-1.)
        self.assertEqual(remaining, distance)
        # Nonzero response time consumes part of the same physical distance.
        self.assertAlmostEqual(anticipated*.75+anticipated**2/9., distance)
        self.assertEqual(stop_target(2., 1., 4.5, .75), (0., -1.))

    def test_declared_response_time_controls_wait_and_nominal_envelopes(self):
        cfg, obs, ext = fixture()
        obs['position'][0, 0] = 10.
        obs['velocity'][0, 0] = 4.
        baseline = cfg['action']['nominal_policy']
        baseline['response_time_s'] = .75
        ctx = ego_decision_context(obs, cfg, ext)
        expected = stop_target(10., 14., 4.5, .75)[0]
        self.assertEqual(ctx['stop_response_time_s'], .75)
        self.assertLessEqual(ctx['nominal_target'], expected)
        profile = {**cfg['action']['profiles']['yield_wait'], 'response_time_s': 1.}
        target, _, diag = EgoDecisionState().evaluate('yield_wait', profile, 0., 4., ctx)
        self.assertAlmostEqual(target, min(4., stop_target(10., 14., 4.5, 1.)[0]))
        self.assertEqual(diag['response_time_s'], 1.)
        for value in (-1., float('nan'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                stop_target(0., 4., 4.5, value)

    def test_versioned_order_survives_sorted_json(self):
        cfg, _, _ = fixture()
        cfg['branches'] = list(branch_names(cfg))
        self.assertEqual(branch_names(json.loads(json.dumps(cfg, sort_keys=True))), tuple(cfg['branches']))

    def test_unknown_kind_is_rejected(self):
        cfg, _, _ = fixture()
        cfg['action']['profiles']['commit_go']['kind'] = 'teleport_first'
        with self.assertRaises(ValueError): validate_profiles(cfg['action'])

    def test_duplicate_profiles_cannot_be_distinguished_by_caption(self):
        cfg, _, _ = fixture()
        cfg['action']['profiles']['commit_go'] = {**cfg['action']['profiles']['stop_hold'], 'description': 'different name'}
        with self.assertRaises(ValueError): validate_profiles(cfg['action'])

    def test_legacy_profile_definitions_remain_supported(self):
        cfg, _, _ = fixture()
        cfg['action'] = {'profiles': select_action_profiles([]), 'available_lane_change_directions': []}
        self.assertEqual(branch_names(cfg), ('maintain','brake_light','brake','accelerate_light','accelerate','maintain_repeat'))

    def test_delay_and_nonzero_deceleration_floor(self):
        cfg, _, _ = fixture()
        p = cfg['action']['profiles']['decelerate']
        self.assertEqual(longitudinal_target(p, .2, 8.), 8.)
        self.assertAlmostEqual(longitudinal_target(p, 4., 8.), 3.8)
        self.assertEqual(longitudinal_target(p, 4., .5), .5)
        self.assertEqual(longitudinal_target(p, 100., 8.), 1.)

    def test_yield_stops_then_releases_only_after_clear_and_wait(self):
        cfg, _, _ = fixture(); state = EgoDecisionState(); p = cfg['action']['profiles']['yield_wait']
        _, hold, d = state.evaluate('yield_wait', p, 1., 8., context())
        self.assertTrue(hold); self.assertEqual(d['stopped_at_s'], 1.)
        clear = context(conflicts={'crossing': {'occupied': False, 'next_arrival_s': None}})
        self.assertEqual(state.evaluate('yield_wait', p, 1.1, 8., clear)[0], 0.)
        self.assertEqual(state.evaluate('yield_wait', p, 1.3, 8., clear)[0], 8.)
        self.assertEqual(state.evaluate('yield_wait', p, 1.4, 8., context())[2]['phase'], 'released')

    def test_yield_does_not_release_for_close_approacher(self):
        cfg, _, _ = fixture(); state=EgoDecisionState(); p=cfg['action']['profiles']['yield_wait']
        state.evaluate('yield_wait',p,0.,8.,context())
        soon=context(conflicts={'crossing': {'occupied':False,'next_arrival_s':.5}})
        self.assertEqual(state.evaluate('yield_wait',p,1.,8.,soon)[0],0.)

    def test_timeout_is_latched_and_does_not_create_forced_gap(self):
        cfg, _, _ = fixture(); state=EgoDecisionState(); p=cfg['action']['profiles']['yield_wait']
        state.evaluate('yield_wait',p,0.,8.,context())
        self.assertEqual(state.evaluate('yield_wait',p,4.,8.,context())[2]['phase'],'timeout_hold')
        clear=context(conflicts={'crossing': {'occupied':False,'next_arrival_s':None}})
        target,hold,d=state.evaluate('yield_wait',p,5.,8.,clear)
        self.assertEqual(target,0.);self.assertTrue(hold);self.assertTrue(d['timed_out'])

    def test_red_signal_prevents_release(self):
        cfg, _, _=fixture();state=EgoDecisionState();p=cfg['action']['profiles']['yield_wait']
        state.evaluate('yield_wait',p,0.,8.,context())
        clear=context(conflicts={'crossing': {'occupied':False,'next_arrival_s':None}},
                      signal={'state':'Red','must_stop':True})
        self.assertEqual(state.evaluate('yield_wait',p,1.,8.,clear)[0],0.)
        clear['signal']={'state':'Green','must_stop':False}
        self.assertEqual(state.evaluate('yield_wait',p,1.1,8.,clear)[0],8.)

    def test_stop_hold_is_not_event_release(self):
        cfg, _, _=fixture();p=cfg['action']['profiles']['stop_hold'];state=EgoDecisionState()
        clear=context(conflicts={'crossing': {'occupied':False,'next_arrival_s':None}})
        self.assertEqual(state.evaluate('stop_hold',p,1.,8.,clear)[0],0.)
        self.assertEqual(state.evaluate('stop_hold',p,4.,8.,clear)[0],0.)
        self.assertEqual(state.evaluate('stop_hold',p,4.1,8.,clear)[0],8.)

    def test_delayed_commit_waits_until_declared_time(self):
        cfg, _, _=fixture();p={**cfg['action']['profiles']['commit_go'],'onset_s':1.,'stop_route_s':14.}
        state=EgoDecisionState()
        self.assertEqual(state.evaluate('commit_delayed',p,.5,8.,context())[0],0.)
        self.assertEqual(state.evaluate('commit_delayed',p,1.,8.,context())[0],8.)
        self.assertEqual(state.evaluate('commit_delayed',p,2.,8.,context())[0],9.5)

    def test_nominal_history_and_continuation_use_identical_baseline(self):
        cfg,obs,ext=fixture();cfg['action']['profiles']['commit_go']={'kind':'nominal'}
        cfg['action']['profiles']['nominal']=cfg['action']['profiles'].pop('commit_go')
        cfg['action']['profile_order']=list(cfg['action']['profiles'])
        before=ScenarioController(cfg);after=ScenarioController(cfg)
        first=before.controls(obs,'nominal',-.01,ext)
        second=after.controls(obs,'nominal',0.,ext)
        np.testing.assert_array_equal(first[0],second[0]);np.testing.assert_array_equal(first[1],second[1])

    def test_common_history_is_same_for_every_branch(self):
        cfg,obs,ext=fixture();results=[]
        for branch in branch_names(cfg):
            results.append(ScenarioController(cfg).controls(obs,branch,-1.,ext))
        for other in results[1:]:
            np.testing.assert_array_equal(results[0][0],other[0])
            np.testing.assert_array_equal(results[0][1],other[1])

    def test_nominal_is_not_maintain_and_is_independent_of_npc_style(self):
        cfg,obs,ext=fixture();obs['velocity'][0,0]=8.
        baseline=ego_decision_context(obs,cfg,ext)
        cfg['controller']['idm']['max_acceleration']=.5
        self.assertEqual(ego_decision_context(obs,cfg,ext)['nominal_target'],baseline['nominal_target'])
        self.assertLess(baseline['nominal_target'],8.)

    def test_no_creep_brake_clamp(self):
        cfg,obs,ext=fixture();controller=ScenarioController(cfg)
        for time in (0.,.05,.1,1.,3.9):
            controls,targets,diagnostics=controller.controls(obs,'stop_hold',time,ext,ego_base_speed=8.)
            self.assertEqual(controls[0,0],0.);self.assertGreaterEqual(controls[0,2],.4)
            self.assertEqual(targets[0],0.)

    def test_declared_terminal_service_brake_changes_only_active_ego_stop(self):
        cfg,obs,ext=fixture();strong=copy.deepcopy(cfg)
        strong['action']['profiles']['stop_hold'].update(terminal_brake_min=1.,onset_s=.5)
        cfg['action']['profiles']['stop_hold']['onset_s']=.5
        obs['velocity'][0,0]=1.4
        for branch in branch_names(cfg):
            for t in (-1.,0.,.45,.5,1.,3.9):
                old=ScenarioController(cfg).controls(obs,branch,t,ext,ego_base_speed=8.)
                new=ScenarioController(strong).controls(obs,branch,t,ext,ego_base_speed=8.)
                np.testing.assert_array_equal(old[0][1:],new[0][1:])
                np.testing.assert_array_equal(old[1],new[1])
                self.assertEqual(old[2],new[2])
                if branch=='stop_hold' and t>=.5:
                    self.assertEqual(new[0][0,0],0.)
                    self.assertEqual(new[0][0,2],1.)
                    self.assertLess(old[0][0,2],new[0][0,2])
                else:
                    np.testing.assert_array_equal(old[0],new[0])

    def test_terminal_brake_does_not_override_nonzero_approach_target(self):
        cfg,obs,ext=fixture();obs['position'][0,0]=0.;obs['velocity'][0,0]=2.
        strong=copy.deepcopy(cfg);strong['action']['profiles']['stop_hold']['terminal_brake_min']=1.
        old=ScenarioController(cfg).controls(obs,'stop_hold',.5,ext,ego_base_speed=8.)
        new=ScenarioController(strong).controls(obs,'stop_hold',.5,ext,ego_base_speed=8.)
        self.assertGreater(new[1][0],.01)
        np.testing.assert_array_equal(old[0],new[0])

    def test_terminal_brake_rejects_out_of_range_and_nonstop_profiles(self):
        for value in (.399,1.001,float('inf'),float('nan'),True):
            cfg,_,_=fixture();cfg['action']['profiles']['stop_hold']['terminal_brake_min']=value
            with self.assertRaises(ValueError):validate_profiles(cfg['action'])
        for name in ('maintain','commit_go','decelerate'):
            cfg,_,_=fixture();cfg['action']['profiles'][name]['terminal_brake_min']=1.
            with self.assertRaises(ValueError):validate_profiles(cfg['action'])

    def test_yield_terminal_brake_preserves_history_approach_and_release(self):
        from test_strict_yield import strict_fixture
        cfg,obs,ext=strict_fixture();strong=copy.deepcopy(cfg)
        strong['action']['profiles']['yield_wait']['terminal_brake_min']=1.
        for position,speed,time,cleared in ((14.,1.6,-.5,False),(0.,3.,.5,False),
                (14.,1.6,.5,False),(10.,3.,.5,True)):
            obs['position'][0,0]=position;obs['velocity'][0,0]=speed
            obs['position'][1,1]=10. if cleared else 0.
            old=ScenarioController(cfg).controls(obs,'yield_wait',time,ext,ego_base_speed=4.)
            new=ScenarioController(strong).controls(obs,'yield_wait',time,ext,ego_base_speed=4.)
            np.testing.assert_array_equal(old[0][1:],new[0][1:])
            np.testing.assert_array_equal(old[1],new[1]);self.assertEqual(old[2],new[2])
            if time>=0 and position==14. and not cleared:
                self.assertTrue(new[2][0]['force_brake_hold'])
                self.assertEqual(new[0][0,0],0.);self.assertEqual(new[0][0,2],1.)
                self.assertLess(old[0][0,2],new[0][0,2])
            else:
                np.testing.assert_array_equal(old[0],new[0])

    def test_commit_changes_only_ego_not_npc_given_same_observation(self):
        cfg,obs,ext=fixture();original=copy.deepcopy(cfg)
        baseline=ScenarioController(cfg).controls(obs,'maintain',1.,ext,ego_base_speed=8.)
        commit=ScenarioController(cfg).controls(obs,'commit_go',1.,ext,ego_base_speed=8.)
        np.testing.assert_array_equal(baseline[0][1:],commit[0][1:])
        self.assertEqual(cfg,original)
        self.assertGreater(commit[1][0],baseline[1][0])

    def test_stateful_controller_rejects_branch_switch(self):
        cfg,obs,ext=fixture();controller=ScenarioController(cfg)
        controller.controls(obs,'maintain',0.,ext)
        with self.assertRaises(ValueError): controller.controls(obs,'stop_hold',.1,ext)

    def test_delayed_lane_change_uses_real_paired_route(self):
        source=np.array([[0.,0.,0.,0.],[30.,0.,0.,0.]])
        target=source.copy();target[:,1]=-3.5
        settings={'profile_version':2,'profiles':{'left_delayed':{'kind':'lane_change','direction':'left','onset_s':.4,'duration_s':3.}},
            'lateral_onset_s':.2,'lateral_duration_s':2.5,'lane_changes':{'left':{'target_route':target.tolist()}}}
        np.testing.assert_array_equal(lane_change_route(source,'left_delayed',.3,settings),source)
        np.testing.assert_allclose(lane_change_route(source,'left_delayed',3.4,settings),target)
        self.assertLess(lane_change_route(source,'left_delayed',1.9,settings)[0,1],0.)

    def test_stationary_occupied_actor_blocks_but_cleared_actor_does_not(self):
        cfg,obs,ext=fixture();conflict=cfg['action']['conflicts'][0]
        obs['velocity'][1,:]=0.
        result=decision_conflict_observation(conflict,obs,cfg,ext)
        self.assertTrue(result['occupied']);self.assertEqual(result['next_arrival_s'],0.)
        obs['position'][1,1]=20.
        result=decision_conflict_observation(conflict,obs,cfg,ext)
        self.assertFalse(result['occupied']);self.assertIsNone(result['next_arrival_s'])

    def test_pedestrian_conflict_reads_current_position_not_start_schedule(self):
        cfg,obs,ext=fixture();cfg['pedestrians']=[{'role':'walker','route_points':[[20.,-10.,0.],[20.,10.,0.]],'start_time_s':999.}]
        cfg['action']['conflicts'][0]['participants']=[{'role':'walker','entry_s':8.,'exit_s':12.}]
        obs['pedestrians']={'roles':['walker'],'position':np.array([[20.,0.,0.]]),
            'velocity':np.zeros((1,3)),'bbox_extent':np.array([[.3,.3,.9]])}
        result=decision_conflict_observation(cfg['action']['conflicts'][0],obs,cfg,ext)
        self.assertTrue(result['occupied'])

    def test_missing_live_signal_observation_is_error(self):
        cfg,obs,ext=fixture()
        cfg['traffic_signal_program']={'groups':[{'id':'ego_light','roles':['ego'],'stop_route_s':{'ego':18.}}]}
        with self.assertRaises(ValueError): ego_decision_context(obs,cfg,ext)

    def test_nominal_respects_scene_priority(self):
        cfg,obs,ext=fixture();obs['velocity'][0,0]=8.
        yielding=ego_decision_context(obs,cfg,ext)['nominal_target']
        cfg['action']['conflicts'][0]['ego_must_yield']=False
        priority=ego_decision_context(obs,cfg,ext)['nominal_target']
        self.assertGreater(priority,yielding)

    def test_zero_initial_speed_wait_restarts_at_declared_resume_speed(self):
        cfg,_,_=fixture();state=EgoDecisionState()
        p={**cfg['action']['profiles']['yield_wait'],'resume_speed_m_s':6.}
        state.evaluate('yield_wait',p,0.,0.,context())
        clear=context(conflicts={'crossing': {'occupied':False,'next_arrival_s':None}})
        self.assertEqual(state.evaluate('yield_wait',p,.5,0.,clear)[0],6.)

    def test_signal_can_turn_red_after_release_before_stop_line(self):
        cfg,_,_=fixture();state=EgoDecisionState();p=cfg['action']['profiles']['yield_wait']
        state.evaluate('yield_wait',p,0.,8.,context())
        clear=context(conflicts={'crossing': {'occupied':False,'next_arrival_s':None}})
        self.assertEqual(state.evaluate('yield_wait',p,.5,8.,clear)[0],8.)
        clear['signal']={'state':'Red','must_stop':True,'distance_front_to_stop_line_m':1.}
        target,hold,diagnostic=state.evaluate('yield_wait',p,.6,8.,clear)
        self.assertEqual(target,0.);self.assertTrue(hold)
        self.assertEqual(diagnostic['phase'],'released_signal_stop')

    def test_stopping_station_outside_geometry_is_rejected(self):
        cfg,_,_=fixture();cfg['action']['profiles']['stop_hold']['stop_route_s']=101.
        with self.assertRaises(ValueError):ScenarioController(cfg)

    def test_undefined_conflict_actor_is_rejected(self):
        cfg,_,_=fixture();cfg['action']['conflicts'][0]['participants'][0]['role']='missing'
        with self.assertRaises(ValueError):ScenarioController(cfg)

    def test_stop_onset_is_not_ignored(self):
        cfg,_,_=fixture();state=EgoDecisionState()
        p={**cfg['action']['profiles']['stop_hold'],'onset_s':1.}
        self.assertEqual(state.evaluate('stop_hold',p,.5,8.,context())[0],8.)
        self.assertEqual(state.evaluate('stop_hold',p,1.,8.,context())[0],0.)

    def test_diagnostics_are_strict_json_serializable(self):
        cfg,obs,ext=fixture();obs['position'][1,1]=20.
        _,_,diagnostics=ScenarioController(cfg).controls(obs,'yield_wait',0.,ext,ego_base_speed=8.)
        json.dumps(diagnostics,allow_nan=False)

    def test_priority_npc_reacts_to_perpendicular_live_body(self):
        cfg,obs,ext=fixture();obs['position'][1]=[20.,-10.,0.]
        obs['position'][0]=[50.,0.,0.];obs['velocity'][0]=0.
        free=npc_target(1,obs,cfg,ext)[0]
        obs['position'][0]=[20.,-5.,0.]
        blocked,diagnostic=npc_target(1,obs,cfg,ext)
        self.assertEqual(free,4.);self.assertLess(blocked,free)
        self.assertEqual(diagnostic['reason'],'brake_for_observed_conflicting_body')

    def test_priority_collision_check_does_not_brake_for_rear_approacher(self):
        cfg,obs,ext=fixture()
        obs['position'][1]=[20.,0.,0.];obs['position'][0]=[20.,-10.,0.]
        obs['rotation_deg'][:,2]=90.;obs['velocity'][0]=[0.,10.,0.]
        target,diagnostic=live_collision_constraint(1,obs,cfg,ext,4.)
        self.assertEqual(target,4.);self.assertFalse(diagnostic['triggered'])

    def test_parallel_neighbor_is_not_collision_blocker(self):
        cfg,obs,ext=fixture()
        obs['position'][1]=[20.,0.,0.];obs['position'][0]=[23.5,1.,0.]
        obs['rotation_deg'][:,2]=90.;obs['velocity'][:]=[0.,4.,0.]
        target,diagnostic=live_collision_constraint(1,obs,cfg,ext,4.)
        self.assertEqual(target,4.);self.assertFalse(diagnostic['triggered'])

    def test_npc_queued_at_red_has_brake_hold_not_pi_creep(self):
        cfg,obs,ext=fixture()
        cfg['traffic_signal_program']={
            'groups':[{'id':'npc_light','roles':['cross'],'landmark_ids':['fixture_light'],
                       'stop_route_s':{'cross':10.}}],
            'phases':[{'start_time_s':-2.,'states':{'npc_light':'Red'}}]}
        obs['position'][1]=[20.,-22.1,0.];obs['velocity'][1]=0.
        obs['position'][0]=[80.,0.,0.];obs['signal_states']={'npc_light':'Red'}
        controller=ScenarioController(cfg)
        for time in (-1.,-.5,0.):
            controls,targets,_=controller.controls(obs,'maintain',time,ext)
            self.assertEqual(targets[1],0.);self.assertEqual(controls[1,0],0.)
            self.assertGreaterEqual(controls[1,2],.4)

    def test_wait_already_past_red_stop_line_can_clear(self):
        cfg,_,_=fixture();state=EgoDecisionState();p=cfg['action']['profiles']['yield_wait']
        state.evaluate('yield_wait',p,0.,8.,context())
        clear=context(conflicts={'crossing': {'occupied':False,'next_arrival_s':None}},
                      signal={'state':'Red','must_stop':False,'committed_past_stop_line':True})
        self.assertEqual(state.evaluate('yield_wait',p,1.,8.,clear)[0],8.)

    def test_initial_queue_upstream_of_stop_station_can_wait_and_resume(self):
        cfg,_,_=fixture();state=EgoDecisionState()
        p={**cfg['action']['profiles']['yield_wait'],'resume_speed_m_s':6.}
        upstream=context(route_s=5.)
        self.assertEqual(state.evaluate('yield_wait',p,0.,0.,upstream)[0],0.)
        upstream['conflicts']['crossing']={'occupied':False,'next_arrival_s':None}
        self.assertEqual(state.evaluate('yield_wait',p,1.,0.,upstream)[0],6.)

    def test_overpass_actor_is_not_a_two_dimensional_collision(self):
        cfg,obs,ext=fixture();obs['position'][1]=[20.,-10.,0.]
        obs['position'][0]=[20.,-5.,6.];obs['velocity'][0]=0.
        target,diagnostic=live_collision_constraint(1,obs,cfg,ext,4.)
        self.assertEqual(target,4.);self.assertFalse(diagnostic['triggered'])


if __name__=='__main__':
    unittest.main()
