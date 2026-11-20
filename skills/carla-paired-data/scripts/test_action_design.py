"""Offline action-library semantics; never starts a CARLA server."""
import copy
import json
import math
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

if __package__:
    from .action_design import profiles_for_template, distinct_profiles_for_template, speed_profile_estimates, configure_actions, conflict_for_config
    from .scene_compiler import runtime_modules
else:
    from action_design import profiles_for_template, distinct_profiles_for_template, speed_profile_estimates, configure_actions, conflict_for_config
    from scene_compiler import runtime_modules


class ActionDesignTests(unittest.TestCase):
    def joint_entry_fixture(self):
        module=sys.modules[configure_actions.__module__];geometry,_=runtime_modules()
        route=[[0.,0.,0.,0.],[200.,0.,0.,0.]]
        cfg={'vehicles':[geometry.vehicle('ego',route,40.,3.4,3.4)],
             'pedestrians':[{}],'history_s':2.,'driving_warmup_s':1.,'duration_s':4.,
             'geometry':{'traffic_design':{}},'interaction_design':{'requested_phase':'entering',
                 'prospective_ego_entry_s':5.,'pedestrian_encounter':{'version':'body-relative-arrival-v2'}},
             'action':{'profiles':{'accelerate':{'kind':'longitudinal','acceleration_m_s2':3.5,
                 'onset_s':.1,'speed_increase_cap_m_s':5.,'min_target_speed_m_s':0.}}}}
        return module,geometry,cfg,{'stop_route_s':94.,'ego_entry_s':100.}

    def test_joint_pedestrian_late_maintain_can_have_capped_strategy_opportunity(self):
        module,geometry,cfg,conflict=self.joint_entry_fixture()
        module._place_stop_approach(cfg,conflict,geometry,delay=.5)
        module.validate_joint_pedestrian_entry(cfg,conflict)
        e=cfg['geometry']['traffic_design']['interaction_opportunity']
        self.assertGreater(e['nominal_constant_speed_entry_after_t0_s'],4.)
        self.assertGreaterEqual(e['available_stop_distance_m'],e['required_stop_distance_m'])
        self.assertTrue(e['strategy_entry_opportunity']['passed'])
        self.assertEqual(e['strategy_entry_opportunity']['entry_horizon_s'],3.)
        self.assertFalse(e['strategy_entry_opportunity']['native_arrival_guaranteed'])
        cfg['action']['profiles']['renamed']=cfg['action']['profiles'].pop('accelerate')
        cfg['future_response']='must_retreat'
        module.validate_joint_pedestrian_entry(cfg,conflict)
        self.assertTrue(e['strategy_entry_opportunity']['passed'])

    def test_joint_entry_rejects_out_of_window_distance_and_retains_other_phase_gate(self):
        module,geometry,cfg,conflict=self.joint_entry_fixture()
        module._place_stop_approach(cfg,conflict,geometry,delay=.5)
        e=cfg['geometry']['traffic_design']['interaction_opportunity']
        distance=module.target_travel_with_response(cfg['action']['profiles']['accelerate'],3.4,3.,.75)
        cfg['vehicles'][0]['spawn_route_s']=100.-distance-3.*3.4
        module.validate_joint_pedestrian_entry(cfg,conflict)
        cfg['vehicles'][0]['spawn_route_s']-=.001
        with self.assertRaisesRegex(ValueError,'one second remaining'):module.validate_joint_pedestrian_entry(cfg,conflict)
        for phase in ('curb_wait','clearing'):
            module,geometry,cfg,conflict=self.joint_entry_fixture()
            cfg['interaction_design']['requested_phase']=phase
            with self.assertRaisesRegex(ValueError,'No nominal ego opportunity'):
                module._place_stop_approach(cfg,conflict,geometry,delay=.5)

    def test_target_travel_uses_cap_and_response_and_matches_runtime_target(self):
        import numpy as np
        module,_,cfg,_=self.joint_entry_fixture();profile=cfg['action']['profiles']['accelerate']
        for cap in (.2,5.,15.):
            for onset in (0.,.5,4.):
                p={**profile,'speed_increase_cap_m_s':cap,'onset_s':onset}
                target=module.target_travel_with_response(p,3.4,3.,.75)
                delayed={**p,'onset_s':onset+.75}
                times=np.linspace(0.,3.,30001)
                speed=np.array([module.longitudinal_target(delayed,t,3.4) for t in times])
                numeric=float(((speed[:-1]+speed[1:])*.5).sum()*.0001)
                self.assertAlmostEqual(target,numeric,places=7)
                if onset>=3.:self.assertAlmostEqual(target,3.4*3.)
        for change in ({'speed_increase_cap_m_s':0.},{'acceleration_m_s2':0.},
                       {'acceleration_m_s2':float('inf')},{'onset_s':-1.}):
            with self.assertRaises(ValueError):module.target_travel_with_response({**profile,**change},3.4,3.,.75)

    def test_joint_low_speed_encounter_repositions_queue_with_stopping_reserve(self):
        module=sys.modules[configure_actions.__module__]
        geometry,_=runtime_modules()
        route=[[0.,0.,0.,0.],[200.,0.,0.,0.]]
        cars=[geometry.vehicle(role,route,station,3.,3.) for role,station in [('ego',40.),('rear',20.)]]
        cfg={'vehicles':cars,'history_s':2.,'driving_warmup_s':1.,'duration_s':4.,
             'geometry':{'traffic_design':{}},'interaction_design':{
                 'requested_phase':'entering','prospective_ego_entry_s':3.5,
                 'pedestrian_encounter':{'version':'body-relative-arrival-v2'}}}
        module._place_stop_approach(cfg,{'stop_route_s':78.,'ego_entry_s':84.},geometry,delay=.5)
        evidence=cfg['geometry']['traffic_design']['interaction_opportunity']
        self.assertLess(evidence['common_ego_stream_spawn_shift_back_m'],0.)
        self.assertAlmostEqual(cars[0]['spawn_route_s']-cars[1]['spawn_route_s'],20.)
        self.assertGreaterEqual(evidence['available_stop_distance_m'],evidence['required_stop_distance_m']-1e-9)
        self.assertLessEqual(evidence['nominal_constant_speed_entry_after_t0_s'],4.)

    def test_nominal_departure_uses_current_map_heading_not_distance_only(self):
        module=sys.modules[configure_actions.__module__]
        from runtime.pedestrian_crossing import DEFAULTS_V3,KIND_V4
        cfg={'dt_s':.05,'history_s':2.,'driving_warmup_s':1.,
             'vehicles':[{'role':'ego','initial_speed_mps':4.,'spawn_route_s':70.,'route':[]}],
             'pedestrians':[{'route_points':[[100.,-3.25,0.],[100.,12.,0.]],'speed_mps':1.5,
                 'start_time_s':-1.,'initial_opportunity_design':{},
                 'response_policy':{'kind':KIND_V4,'crossing_feedback':dict(DEFAULTS_V3)}}]}
        def geometry(yaw):
            return SimpleNamespace(point_at_s=lambda route,station:{'position':[station,0.,0.],
                                                                    'rotation_deg':[0.,0.,yaw]})
        module.nominal_pedestrian_entry(cfg,geometry(0.))
        self.assertTrue(cfg['pedestrians'][0]['initial_opportunity_design']['nominal_entry_sweep']['caution_clear'])
        with self.assertRaisesRegex(ValueError,'nominal pre-t0 pedestrian departure gap'):
            module.nominal_pedestrian_entry(cfg,geometry(15.))

    def configured_catalogue(self, template, speed=2.8):
        # Isolate action authoring from map placement. This is an initial-state
        # policy fixture, not a model of Unreal dynamics or native acceptance.
        conflict = {'id': 'interaction_0', 'stop_route_s': 77.02,
                    'participants': [{'role': name, 'entry_s': 74., 'exit_s': 84.}
                                     for name in ('interaction_vehicle', 'traffic_03', 'traffic_06')]}
        cfg = {'seed': 1, 'dt_s': .05, 'duration_s': 4., 'action': {},
               'vehicles': [{'initial_speed_mps': speed, 'desired_speed_mps': speed}]}
        changes = {'left': {}, 'right': {}} if template == 'ego_lane_change_straight' else {}
        module = sys.modules[configure_actions.__module__]
        with patch.object(module, 'conflict_for_config', return_value=copy.deepcopy(conflict)), \
             patch.object(module, '_place_stop_approach'):
            configure_actions(cfg, template, changes, object())
        expected = distinct_profiles_for_template(template, sorted(changes), conflict, cfg['seed'])
        for profile in expected.values():
            if profile['kind'] in ('stop_hold', 'yield_wait'):
                profile['resume_speed_m_s'] = speed
        return cfg, expected

    def test_signalized_hold_deadline_uses_initial_speed_and_two_physics_ticks(self):
        for speed in (0., 2.5, 2.75, 2.8, 3., 5.):
            with self.subTest(speed=speed):
                cfg, legacy = self.configured_catalogue('cross_signalized', speed)
                hold = cfg['action']['profiles']['stop_hold']
                self.assertAlmostEqual(hold['stop_deadline_s'], .85 + speed/6. + 2*.05)
                self.assertEqual(hold['comfortable_deceleration_m_s2'], 6.)
                self.assertAlmostEqual(hold['response_time_s'], .85)
                for key in set(hold) | set(legacy['stop_hold']):
                    if key not in ('stop_deadline_s', 'response_time_s'):
                        self.assertEqual(hold[key], legacy['stop_hold'][key])

    def test_signalized_yield_and_named_wait_objects_are_unchanged(self):
        cfg, legacy = self.configured_catalogue('cross_signalized')
        for name in legacy:
            if name != 'stop_hold':
                self.assertEqual(cfg['action']['profiles'][name], legacy[name])
        wait = cfg['action']['profiles']['yield_wait']
        self.assertEqual(wait['wait_for_roles'], ['interaction_vehicle', 'traffic_03', 'traffic_06'])
        self.assertEqual(wait['stop_deadline_s'], 3.5)
        self.assertEqual(wait['release_condition'], 'specified_actors_cleared_and_entry_clear_v1')
        self.assertEqual(wait['timeout_behavior'], 'remain_stopped')

    def test_every_other_template_preserves_its_entire_profiles(self):
        grid = json.loads((Path(__file__).resolve().parents[1]/'assets/grid.example.json').read_text())
        for row in grid['pairs']:
            if row.get('enabled') and row['id'] != 'cross_signalized':
                with self.subTest(template=row['id']):
                    cfg, legacy = self.configured_catalogue(row['id'])
                    self.assertEqual(cfg['action']['profiles'], legacy)

    def test_signalized_deadline_reserves_two_ticks_under_existing_six_cap(self):
        if __package__:
            from .runtime.ego_decisions import deadline_stop_target
        else:
            from runtime.ego_decisions import deadline_stop_target
        for speed in (2.5, 2.75, 2.8, 3.):
            cfg, _ = self.configured_catalogue('cross_signalized', speed)
            hold = cfg['action']['profiles']['stop_hold']
            zero_time = hold['stop_deadline_s'] - hold['response_time_s']
            self.assertAlmostEqual(6.*zero_time-speed, .6)
            strong_distance = speed*.85 + speed*speed/12. + .75
            existing_yield_with_position_tick = speed*.75 + speed*speed/9. + .75 + speed*.05
            self.assertLessEqual(strong_distance, existing_yield_with_position_tick)
            self.assertAlmostEqual(strong_distance-existing_yield_with_position_tick,
                                   .05*speed-speed*speed/36.)
            # Existing runtime feasibility still rejects a measured drift that
            # exceeds the unchanged cap; no cap or tolerance is widened.
            self.assertEqual(deadline_stop_target(0., 0., hold['stop_deadline_s'], speed+.6, 6., .85), speed+.6)
            with self.assertRaisesRegex(ValueError, 'physically infeasible'):
                deadline_stop_target(0., 0., hold['stop_deadline_s'], speed+.60001, 6., .85)
            first_zero_tick = math.ceil((zero_time-1e-9)/.05)*.05
            self.assertEqual(deadline_stop_target(first_zero_tick, 0., hold['stop_deadline_s'], speed, 6., .85), 0.)

    def test_tight_spatial_yield_cap_does_not_mask_authored_earlier_hold(self):
        if __package__:
            from .runtime.ego_decisions import deadline_stop_target, stop_target
        else:
            from runtime.ego_decisions import deadline_stop_target, stop_target
        # Same synthetic currently observed station for both policies. No
        # rollout or future NPC response chooses any design parameter here.
        cfg, legacy = self.configured_catalogue('cross_signalized', 2.8)
        profiles = cfg['action']['profiles']; now = .5; remaining = 1.95
        def target(profile):
            return min(2.8, stop_target(0., remaining, profile['comfortable_deceleration_m_s2'], profile['response_time_s'])[0],
                       deadline_stop_target(now, 0., profile['stop_deadline_s'], 2.8,
                                            profile['comfortable_deceleration_m_s2'], profile['response_time_s']))
        old_difference = abs(target(legacy['stop_hold'])-target(legacy['yield_wait']))
        new_difference = abs(target(profiles['stop_hold'])-target(profiles['yield_wait']))
        self.assertLess(old_difference, .2)
        self.assertGreater(new_difference, 1.5)
        self.assertLess(target(profiles['stop_hold']), target(profiles['yield_wait']))

    def test_signalized_repair_keeps_complete_ten_pair_measured_contract(self):
        cfg, _ = self.configured_catalogue('cross_signalized')
        contract = cfg['action_separation_contract']
        self.assertEqual(contract['pair_count'], 10)
        self.assertEqual(contract['minimum_control_duration_s'], .25)
        self.assertEqual(contract['minimum_motion_duration_s'], .5)
        self.assertEqual(contract['position_threshold_m'], 1.)
        self.assertEqual(contract['speed_threshold_m_s'], .5)
        self.assertEqual(contract['control_thresholds'], {'throttle': .1, 'steer': .05, 'brake': .1})
        self.assertEqual(contract['minimum_measured_t0_speed_m_s'], 2.5)
        self.assertFalse(contract['npc_response_used_as_gate'])

    def test_new_catalogue_avoids_nominal_and_short_delayed_lane_aliases(self):
        conflict={'id':'c','stop_route_s':50.,'participants':[{'role':'npc','entry_s':30.,'exit_s':40.}]}
        grid=json.loads((Path(__file__).resolve().parents[1]/'assets/grid.example.json').read_text())
        for row in grid['pairs']:
            if not row.get('enabled'): continue
            for directions in (['left'],['right'],['left','right']):
                p=distinct_profiles_for_template(row['id'],directions,conflict)
                self.assertEqual(len(p),5)
                self.assertLessEqual(set(p),set(row['candidate_actions']))
                self.assertNotIn('nominal',p)
                self.assertFalse(any(n.endswith('_delayed') for n in p) if row['id']=='ego_lane_change_straight' else False)
                if 'stop_hold' in p:
                    self.assertLess(p['stop_hold']['stop_deadline_s'],p['yield_wait']['stop_deadline_s'])
                    self.assertGreater(p['stop_hold']['comfortable_deceleration_m_s2'],p['yield_wait']['comfortable_deceleration_m_s2'])

    def setUp(self):
        self.conflict = {'id': 'interaction_0', 'stop_route_s': 50.,
                         'participants':[{'role':'crossing_vehicle','entry_s':30.,'exit_s':40.}]}

    def test_all_templates_select_five_declared_conditions(self):
        grid = json.loads((Path(__file__).resolve().parents[1] / 'assets/grid.example.json').read_text())
        for pair in grid['pairs']:
            if not pair.get('enabled'):
                continue
            for directions in (['left'], ['right'], ['left', 'right']):
                with self.subTest(template=pair['id'], directions=directions):
                    profiles = profiles_for_template(pair['id'], directions, self.conflict)
                    self.assertEqual(len(profiles), 5)
                    self.assertEqual(next(iter(profiles)), 'maintain')
                    self.assertLessEqual(set(profiles), set(pair['candidate_actions']))

    def test_following_retains_strength_and_braking_timing_scan(self):
        p = profiles_for_template('follower_response_straight', ['left', 'right'], None)
        self.assertEqual(set(p), {'maintain', 'brake_light', 'brake', 'accelerate', 'brake_delayed'})
        self.assertEqual(p['brake']['acceleration_m_s2'], p['brake_delayed']['acceleration_m_s2'])
        self.assertGreater(p['brake_delayed']['onset_s'], p['brake']['onset_s'])

    def test_single_direction_uses_same_real_lane_with_distinct_onset(self):
        for side in ('left', 'right'):
            p = profiles_for_template('ego_lane_change_straight', [side], None)
            self.assertEqual(p[side]['direction'], p[side + '_delayed']['direction'])
            self.assertGreater(p[side + '_delayed']['onset_s'], p[side]['onset_s'])
            for v in p.values():
                if v['kind'] == 'lane_change':
                    self.assertLessEqual(v['onset_s'] + v['duration_s'] + v['settling_time_s'] + v['required_final_hold_s'], 4.)
                    self.assertEqual(v['required_final_hold_s'],.5)
                    self.assertGreaterEqual(v['settling_time_s'],.9)
        with self.assertRaises(ValueError):
            profiles_for_template('ego_lane_change_straight', [], None)

    def test_wait_and_hold_have_different_release_rules(self):
        p = profiles_for_template('left_unsignalized', [], self.conflict)
        self.assertEqual(p['stop_hold']['hold_until_s'], 4.)
        self.assertEqual(p['yield_wait']['conflict_id'], 'interaction_0')
        self.assertEqual(p['yield_wait']['timeout_behavior'], 'remain_stopped')
        self.assertNotIn('hold_until_s', p['yield_wait'])
        self.assertEqual(p['stop_hold']['stop_route_s'], p['yield_wait']['stop_route_s'])
        self.assertEqual(p['stop_hold']['response_time_s'], .75)
        self.assertEqual(p['yield_wait']['response_time_s'], .75)
        self.assertEqual(p['stop_hold']['stop_deadline_s'],3.5)
        self.assertEqual(p['yield_wait']['stop_deadline_s'],3.5)
        self.assertEqual(p['stop_hold']['stop_margin_m'],.75)
        self.assertEqual(p['stop_hold']['hold_policy'],'hold_until_window_end_v1')
        self.assertEqual(p['yield_wait']['release_condition'],'specified_actors_cleared_and_entry_clear_v1')
        self.assertEqual(p['yield_wait']['wait_for_roles'],['crossing_vehicle'])
        self.assertEqual(p['yield_wait']['min_wait_s'],0.)

    def test_commit_delayed_changes_onset_without_changing_npc(self):
        p = profiles_for_template('ego_merge_ramp', [], self.conflict)
        self.assertEqual(p['commit_go']['onset_s'], 0.)
        self.assertEqual(p['commit_delayed']['onset_s'], 1.)
        self.assertEqual(p['commit_delayed']['before_onset'], 'approach_and_stop_at_declared_stop_station')
        self.assertNotIn('npc_control', p['commit_go'])

    def test_eight_mps_target_example_and_low_speed_end_state_alias(self):
        p = profiles_for_template('follower_response_straight', [], None)
        values = speed_profile_estimates(p, 8.)
        self.assertAlmostEqual(values['brake_light']['target_end_speed_m_s'], 3.2)
        self.assertAlmostEqual(values['brake']['target_first_zero_s'], 1.8)
        self.assertGreater(values['brake_delayed']['target_travel_m'], values['brake']['target_travel_m'])
        low = speed_profile_estimates(p, .8)
        self.assertEqual(low['brake']['target_end_speed_m_s'], low['brake_light']['target_end_speed_m_s'])
        self.assertNotEqual(low['brake']['target_travel_m'], low['brake_light']['target_travel_m'])

    def test_deceleration_floor_does_not_accelerate_a_slow_or_stopped_ego(self):
        p = profiles_for_template('merge_straight', [], self.conflict)
        for initial_speed in (0., .5):
            with self.subTest(initial_speed=initial_speed):
                estimate = speed_profile_estimates(p, initial_speed)['decelerate']
                self.assertEqual(estimate['target_end_speed_m_s'], initial_speed)
                self.assertAlmostEqual(estimate['target_travel_m'], 4. * initial_speed)
                self.assertEqual(estimate['target_first_zero_s'], 0. if initial_speed == 0. else None)
        # From 2.2 m/s, one second of braking reaches the 1 m/s floor,
        # then the remaining three seconds stay at that floor.
        estimate = speed_profile_estimates(p, 2.2)['decelerate']
        self.assertAlmostEqual(estimate['target_end_speed_m_s'], 1.)
        self.assertAlmostEqual(estimate['target_travel_m'], 4.6)
        self.assertIsNone(estimate['target_first_zero_s'])

    def test_delayed_braking_includes_pre_onset_travel(self):
        p = profiles_for_template('follower_response_straight', [], None)
        estimate = speed_profile_estimates(p, 4.5)['brake_delayed']
        # 0.6 s at 4.5 m/s, followed by one second braking to rest.
        self.assertAlmostEqual(estimate['target_travel_m'], 4.95)
        self.assertAlmostEqual(estimate['target_first_zero_s'], 1.6)
        self.assertEqual(estimate['target_end_speed_m_s'], 0.)

    def test_acceleration_cap_and_floor_follow_execution_semantics(self):
        p = {'accelerate': {'kind': 'longitudinal', 'onset_s': .5,
             'acceleration_m_s2': 2., 'speed_increase_cap_m_s': 3.,
             'min_target_speed_m_s': 2.}}
        # The deceleration floor must not jump a 1 m/s ego to 2 m/s.
        before = speed_profile_estimates(p, 1., window=.5)['accelerate']
        self.assertEqual(before['target_end_speed_m_s'], 1.)
        self.assertAlmostEqual(before['target_travel_m'], .5)
        # 0.5 s at 1 m/s, 1.5 s ramping to 4 m/s, then 2 s at the cap.
        estimate = speed_profile_estimates(p, 1.)['accelerate']
        self.assertEqual(estimate['target_end_speed_m_s'], 4.)
        self.assertAlmostEqual(estimate['target_travel_m'], 12.25)
        self.assertIsNone(estimate['target_first_zero_s'])

    def test_delayed_commit_does_not_claim_state_independent_travel(self):
        p = profiles_for_template('ego_merge_ramp', [], self.conflict)
        estimate = speed_profile_estimates(p, 2.)['commit_delayed']
        self.assertIsNone(estimate['target_travel_m'])
        self.assertTrue(estimate['before_onset_depends_on_stop_approach'])
        self.assertAlmostEqual(estimate['target_end_speed_m_s'], 6.5)

    def test_pedestrian_nominal_is_policy_not_zero_action(self):
        p = profiles_for_template('pedestrian_straight', [], self.conflict)
        self.assertEqual(p['nominal']['kind'], 'nominal')
        self.assertNotEqual(p['nominal'], p['maintain'])
        self.assertTrue(speed_profile_estimates(p, 5.)['nominal']['depends_on_current_traffic_state'])

    def test_stop_plan_uses_geometry_and_keeps_shared_stream_layout(self):
        geometry, _ = runtime_modules()
        route = [[float(i), 0., 0., 0.] for i in range(220)]
        opposing = [[100., float(i-100), 0., 90.] for i in range(220)]
        cars = [geometry.vehicle('ego', route, 76., 5.),
                geometry.vehicle('follower', route, 58., 5.),
                geometry.vehicle('interaction_vehicle', opposing, 74., 5.)]
        cfg = geometry.finish_config('cross_unsignalized', 0, 1, cars,
            {'type': 'cross_unsignalized', 'map': 'unit', 'origin': [100., 0., 0.],
             'ego_conflict_s': 100., 'conflict_xyz': [100., 0., 0.], 'traffic_design': {}}, 2., 1.)
        before = copy.deepcopy(cfg['vehicles'])
        configure_actions(cfg, 'cross_unsignalized', {}, geometry)
        check = cfg['geometry']['traffic_design']['interaction_opportunity']
        self.assertGreaterEqual(check['available_stop_distance_m'], check['ideal_braking_distance_m'])
        self.assertEqual(check['controller_response_distance_m'], 5.*.75)
        self.assertAlmostEqual(check['required_stop_distance_m'], 5.*.75 + 25./9. + .75)
        self.assertGreaterEqual(check['available_stop_distance_m'] + 1e-9, check['required_stop_distance_m'])
        after = cfg['vehicles']
        self.assertAlmostEqual(after[0]['spawn_route_s'] - after[1]['spawn_route_s'],
                               before[0]['spawn_route_s'] - before[1]['spawn_route_s'])
        self.assertEqual(after[2], before[2])
        self.assertTrue(cfg['action']['nominal_policy']['respect_signals'])
        self.assertEqual(cfg['action']['nominal_policy']['response_time_s'], .75)
        self.assertFalse(cfg['action']['design_evidence']['equal_realized_trajectories_are_distinct_samples'])

    def test_pedestrian_station_uses_continuous_projection_not_nearest_vertex(self):
        geometry, _ = runtime_modules()
        route = [[float(i), 0., 0., 0.] for i in range(120)]
        cfg = {'vehicles': [{'role': 'ego', 'route': route}],
               'geometry': {'ego_conflict_s': 50.},
               'pedestrians': [{'role': 'walker', 'route_points': [[50., -8., 0.], [50., 8., .1]]}]}
        p = conflict_for_config(cfg, 'pedestrian_straight', geometry)['participants'][0]
        self.assertAlmostEqual(p['entry_s'], 6.2)
        self.assertAlmostEqual(p['exit_s'], 9.8)

    def test_signal_encounter_checks_stopped_npc_reach_from_declared_green(self):
        if __package__:
            from .interaction_geometry import signal_crossing_plan
        else:
            from interaction_geometry import signal_crossing_plan
        c = {'ego_conflict_s': 100., 'npc_conflict_s': 100.,
             'ego_route': [[0., 0., 0., 0.], [200., 0., 0., 0.]],
             'npc_route': [[100., -100., 0., 90.], [100., 100., 0., 90.]],
             'signal_approaches': {'ego': {'stop_route_s_m': 80.}, 'npc': {'stop_route_s_m': 91.}}}
        style = {'max_acceleration': 1.5, 'minimum_gap': 3., 'desired_speed_multiplier': .9}
        p = signal_crossing_plan(c, 1., style)
        self.assertEqual(p['npc_green_time_s'], -.1)
        self.assertEqual(p['npc_initial_speed_m_s'], 0.)
        self.assertFalse(p['initial_entry_legality_observed'])
        self.assertGreaterEqual(p['npc_ideal_4s_acceleration_distance_m'], p['npc_distance_to_conflict_entry_m'])
        self.assertTrue(all(v >= 2.5 for v in p['ego_initial_speed_candidates_m_s']))
        self.assertLessEqual(p['npc_optimistic_body_entry_s'], 3.)
        self.assertGreaterEqual(p['npc_ideal_3s_acceleration_distance_m'], p['npc_front_to_conflict_entry_bound_m'])
        c['signal_approaches']['npc']['stop_route_s_m'] = 80.
        with self.assertRaisesRegex(ValueError, 'cannot reach'):
            signal_crossing_plan(c, 1., style)

    def test_nominal_priority_is_consistent_with_npc_yield_rule(self):
        geometry, _ = runtime_modules()
        route = [[float(i), 0., 0., 0.] for i in range(120)]
        other = [[50., float(i - 50), 0., 90.] for i in range(120)]
        cfg = {'vehicles': [{'role': 'ego', 'route': route}, {'role': 'npc', 'route': other}],
               'geometry': {'ego_conflict_s': 50., 'origin': [50., 0., 0.],
                            'conflicts': [{'roles': ['ego', 'npc'], 'priority_role': 'ego'}]}}
        self.assertFalse(conflict_for_config(cfg, 'cross_unsignalized', geometry)['ego_must_yield'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
