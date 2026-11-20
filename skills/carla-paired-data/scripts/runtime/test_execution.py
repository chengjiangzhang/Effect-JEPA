"""Offline execution/paired-outcome checks; synthetic records, no CARLA server."""
import copy
import unittest
import numpy as np

from audit_execution import branch_audit, response_audit, replay_policy, contracts, scene_opportunity
from controllers_v2 import ScenarioController, branch_names, npc_target
from test_decisions import fixture


def straight_spec():
    cfg, _, ext = fixture()
    cfg['duration_s'] = 4.
    cfg['vehicles'][1]['route'] = [[200.,-100.,0.,90.],[200.,100.,0.,90.]]
    cfg['action']['conflicts'][0]['participants'][0].update(entry_s=100.,exit_s=106.)
    return cfg,ext


def recorded(cfg,ext,branch='maintain',ego_speed=None,ego_position=None,npc_speed=4.,npc_position=None):
    times=np.arange(121)*.05-2
    speed=np.full(121,8.) if ego_speed is None else np.asarray(ego_speed)
    x=np.r_[0.,np.cumsum(speed[:-1]*.05)]
    positions=np.zeros((121,2,3));positions[:,0,0]=x
    positions[:,1,0]=200.;positions[:,1,1]=-50+np.arange(121)*.05*npc_speed
    if ego_position is not None:positions[:,0]=ego_position
    if npc_position is not None:positions[:,1]=npc_position
    velocities=np.zeros((121,2,3));velocities[:,0,0]=speed;velocities[:,1,1]=npc_speed
    angles=np.zeros((121,2,3));angles[:,1,2]=90.
    state={'position':positions,'velocity':velocities,'rotation_deg':angles,'time_s':times,
           'bbox_extent':ext,'bbox_offset':np.zeros((2,3)),'bbox_rotation_deg':np.zeros((2,3))}
    tf=np.broadcast_to(np.eye(4),(121,2,4,4)).copy();tf[:,:,:3,3]=positions
    tf[:,1,:2,:2]=[[0.,-1.],[1.,0.]]
    state['transform_world_vehicle']=tf
    controller=ScenarioController(cfg)
    controls=[];targets=[];diagnostics=[]
    for k,t in enumerate(times):
        obs={key:state[key][k] for key in ('position','velocity','rotation_deg')}
        obs.update(bbox_offset=state['bbox_offset'],bbox_rotation_deg=state['bbox_rotation_deg'])
        action='maintain' if k<40 or branch=='maintain_repeat' else branch
        c,v,d=controller.controls(obs,action,t,ext,ego_base_speed=float(speed[40]))
        if k==120:c[:]=0.
        controls.append(c);targets.append(v);diagnostics.append(d)
    state['control']=np.asarray(controls);state['target_speed']=np.asarray(targets)
    state['observed_control']=np.concatenate((state['control'][:1],state['control'][:-1]))
    return state,diagnostics


def lane_spec():
    cfg,ext=straight_spec()
    source=np.array([[float(x),0.,0.,0.] for x in range(-20,101,5)])
    target=source.copy();target[:,1]=-3.5
    cfg['vehicles'][0]['route']=source.tolist()
    cfg['action']['available_lane_change_directions']=['left']
    cfg['action']['lane_changes']={'left':{'available':True,'target_route':target.tolist(),
        'width_m':3.5,'evidence':{'station_count':len(source),'same_direction':True,'target_lane_type':'Driving',
        'source_lane_keys':[[1,0,-2]]*len(source),'target_lane_keys':[[1,0,-1]]*len(source),
        'source_lane_widths_m':[3.5]*len(source),'target_lane_widths_m':[3.5]*len(source)}}}
    cfg['action']['profiles']['commit_go']={'kind':'lane_change','direction':'left','onset_s':.5,'duration_s':2.}
    return cfg,ext


class ExecutionTests(unittest.TestCase):
    def test_terminal_brake_command_is_specified_but_does_not_excuse_motion(self):
        cfg,ext=straight_spec()
        cfg['action']['profiles']['stop_hold'].update(terminal_brake_min=1.,stop_deadline_s=2.25,
            response_time_s=.75,comfortable_deceleration_m_s2=6.)
        state,diag=recorded(cfg,ext,'stop_hold',ego_speed=np.full(121,1.4))
        row=branch_audit(state,cfg,'stop_hold',{'collisions':[]},diag)
        self.assertIn('ego_failed_to_hold_stationary',row['failure_reason'])
        self.assertIn('ego_did_not_stop_by_declared_deadline',row['failure_reason'])
        replay=replay_policy(state,cfg,'stop_hold',diag,contracts(cfg)[0])
        self.assertTrue(replay[0]['passed'],replay[0])
        wrong=copy.deepcopy(state);wrong['control'][90,0,2]=.4
        failed=replay_policy(wrong,cfg,'stop_hold',diag,contracts(cfg)[0])
        self.assertGreater(failed[0]['command_error_count'],0)

    def test_pedestrian_pairwise_response_is_reported_without_forcing_a_difference(self):
        cfg,ext=straight_spec();base,_=recorded(cfg,ext)
        names=branch_names(cfg);states={n:copy.deepcopy(base) for n in names}
        cfg['pedestrians']=[{'role':'walker'}]
        people={n:{'time_s':base['time_s'].copy(),'position':np.zeros((121,1,3)),
                   'velocity':np.zeros((121,1,3))} for n in names}
        people['stop_hold']['position'][41:,0,0]=2.
        people['stop_hold']['velocity'][41:,0,0]=1.
        result=response_audit(states,cfg,people)['response_metrics']['relevant_actor_response']
        self.assertFalse(result['used_as_acceptance_gate'])
        self.assertEqual(result['targets']['walker']['actor_type'],'pedestrian')
        self.assertEqual(len(result['targets']['walker']['pairs']),10)
        self.assertEqual(sum(p['above_10_m'] for p in result['targets']['walker']['pairs']),4)
        self.assertEqual(result['targets']['walker']['repeat_xy_noise_m'],0.)
        with self.assertRaises(ValueError):response_audit(states,cfg)

    def test_maintain_record_passes(self):
        cfg,ext=straight_spec();state,diag=recorded(cfg,ext)
        row=branch_audit(state,cfg,'maintain',{'collisions':[]},diag)
        self.assertTrue(row['execution_pass'],row['failure_reason'])
        self.assertFalse(row['observed_outcome']['collision'])

    def test_unchanged_actual_speed_fails_braking_not_just_control_existence(self):
        cfg,ext=straight_spec();state,diag=recorded(cfg,ext,'decelerate')
        row=branch_audit(state,cfg,'decelerate',{'collisions':[]},diag)
        self.assertFalse(row['execution_pass'])
        self.assertIn('ego_requested_speed_change_not_realized',row['failure_reason'])

    def test_actual_braking_passes(self):
        cfg,ext=straight_spec();times=np.arange(121)*.05-2
        speed=8-1.2*np.maximum(times-.5,0)
        state,diag=recorded(cfg,ext,'decelerate',ego_speed=speed)
        row=branch_audit(state,cfg,'decelerate',{'collisions':[]},diag)
        self.assertTrue(row['execution_pass'],row['failure_reason'])

    def test_wrong_recorded_target_fails_replayed_policy(self):
        cfg,ext=straight_spec();state,diag=recorded(cfg,ext)
        state['target_speed'][60:,0]+=2
        row=branch_audit(state,cfg,'maintain',{'collisions':[]},diag)
        self.assertIn('policy_or_actuator_execution_mismatch',row['failure_reason'])

    def test_actuator_was_not_applied(self):
        cfg,ext=straight_spec();state,diag=recorded(cfg,ext)
        state['observed_control'][50,0,2]=.9
        row=branch_audit(state,cfg,'maintain',{'collisions':[]},diag)
        self.assertFalse(row['checks']['policy_execution']['passed'])

    def test_wrong_commands_do_not_pass_by_matching_readback(self):
        cfg,ext=straight_spec();state,diag=recorded(cfg,ext)
        state['control'][50,0]=[1.,0.,0.];state['observed_control'][51,0]=[1.,0.,0.]
        replay,_=replay_policy(state,cfg,'maintain',diag)
        self.assertGreater(replay['command_error_count'],0)

    def test_collision_outcome_is_retained_and_terminal_assessment_censored(self):
        cfg,ext=straight_spec();state,diag=recorded(cfg,ext)
        meta={'collisions':[{'role':'ego','other_role':'cross','time_s':1.}]}
        row=branch_audit(state,cfg,'maintain',meta,diag)
        self.assertTrue(row['execution_pass'],row['failure_reason'])
        self.assertEqual(row['quality_status'],'censored')
        self.assertFalse(row['observed_outcome']['terminal_completion_assessed'])

    def test_early_collision_does_not_fake_motion_pass(self):
        cfg,ext=straight_spec();state,diag=recorded(cfg,ext)
        row=branch_audit(state,cfg,'maintain',{'collisions':[{'role':'ego','time_s':.05}]},diag)
        self.assertIsNone(row['execution_pass'])
        self.assertEqual(row['quality_status'],'censored')

    def test_preintervention_collision_is_failure(self):
        cfg,ext=straight_spec();state,diag=recorded(cfg,ext)
        row=branch_audit(state,cfg,'maintain',{'collisions':[{'role':'ego','time_s':0.}]},diag)
        self.assertIn('initialization_or_prefix_collision',row['failure_reason'])

    def test_waiting_for_blocked_conflict_is_valid_without_passage(self):
        cfg,_,ext=fixture();cfg['duration_s']=4.
        positions=np.broadcast_to([14.,0.,0.],(121,3));npc=np.broadcast_to([20.,0.,0.],(121,3))
        state,diag=recorded(cfg,ext,'yield_wait',ego_speed=np.zeros(121),ego_position=positions,npc_position=npc,npc_speed=0.)
        row=branch_audit(state,cfg,'yield_wait',{'collisions':[]},diag)
        self.assertTrue(row['execution_pass'],row['failure_reason'])
        self.assertIn('waiting',row['observed_outcome']['ego'])
        self.assertIn('wait_timeout_hold',row['observed_outcome']['ego'])

    def test_bypassing_wait_position_is_failure(self):
        cfg,ext=straight_spec();state,diag=recorded(cfg,ext,'stop_hold')
        row=branch_audit(state,cfg,'stop_hold',{'collisions':[]},diag)
        self.assertIn('ego_passed_required_wait_or_stop_position',row['failure_reason'])

    def test_lane_change_commands_without_actual_lane_entry_fail(self):
        cfg,ext=lane_spec();state,diag=recorded(cfg,ext,'commit_go')
        row=branch_audit(state,cfg,'commit_go',{'collisions':[]},diag)
        self.assertIn('ego_lane_change_not_completed_to_specification',row['failure_reason'])
        self.assertFalse(row['checks']['ego_physical_execution']['lane_change']['completed'])

    def test_contact_does_not_demand_postcontact_lane_completion(self):
        cfg,ext=lane_spec();state,diag=recorded(cfg,ext,'commit_go')
        row=branch_audit(state,cfg,'commit_go',{'collisions':[{'role':'ego','time_s':1.}]},diag)
        self.assertTrue(row['execution_pass'],row['failure_reason'])
        self.assertFalse(row['checks']['ego_physical_execution']['terminal_requirement_checked'])
        self.assertIn('lane_change_incomplete',row['observed_outcome']['ego'])

    def test_late_contact_does_not_excuse_already_failed_lane_change(self):
        cfg,ext=lane_spec();state,diag=recorded(cfg,ext,'commit_go')
        row=branch_audit(state,cfg,'commit_go',{'collisions':[{'role':'ego','time_s':3.8}]},diag)
        self.assertFalse(row['execution_pass'])
        self.assertIn('ego_lane_change_not_completed_to_specification',row['failure_reason'])
        self.assertTrue(row['checks']['ego_physical_execution']['terminal_requirement_checked'])

    def test_declared_lane_settling_is_budgeted_without_changing_final_hold(self):
        cfg,ext=lane_spec();cfg['action']['profiles']['commit_go']['settling_time_s']=.8
        state,diag=recorded(cfg,ext,'commit_go')
        # Contact before the declared completion deadline censors completion,
        # even though the reference transition itself has already finished.
        row=branch_audit(state,cfg,'commit_go',{'collisions':[{'role':'ego','time_s':3.4}]},diag)
        physical=row['checks']['ego_physical_execution']
        self.assertAlmostEqual(physical['lane_completion_deadline_s'],3.8)
        self.assertTrue(row['execution_pass'],row['failure_reason'])
        self.assertFalse(physical['terminal_requirement_checked'])
        # Settling is not a substitute for actual complete lane entry/hold.
        row=branch_audit(state,cfg,'commit_go',{'collisions':[]},diag)
        self.assertIn('ego_lane_change_not_completed_to_specification',row['failure_reason'])
        row=branch_audit(state,cfg,'commit_go',{'collisions':[{'role':'ego','time_s':3.9}]},diag)
        self.assertIn('ego_lane_change_not_completed_to_specification',row['failure_reason'])

    def test_lane_settling_must_fit_alongside_unchanged_half_second_hold(self):
        cfg,ext=lane_spec();cfg['action']['profiles']['commit_go']['settling_time_s']=1.1
        with self.assertRaisesRegex(ValueError,'settling'):ScenarioController(cfg)

    def test_stopping_after_declared_deadline_is_failure_even_if_finally_stopped(self):
        cfg,ext=straight_spec();p=cfg['action']['profiles']['stop_hold']
        p.update(stop_route_s=17.,stop_deadline_s=3.5,stop_margin_m=.75)
        # Keep physical positions safely upstream; isolate deadline from stop
        # line and end-window tests. Speed first reaches zero only at 3.7s.
        times=np.arange(121)*.05-2
        speed=np.where(times<3.7,1.,0.)
        positions=np.zeros((121,3));positions[:,0]=np.minimum(np.arange(121)*.05,5.7)
        state,diag=recorded(cfg,ext,'stop_hold',ego_speed=speed,ego_position=positions)
        row=branch_audit(state,cfg,'stop_hold',{'collisions':[]},diag)
        self.assertIn('ego_did_not_stop_by_declared_deadline',row['failure_reason'])
        self.assertNotIn('ego_did_not_stop_within_declared_window',row['failure_reason'])
        # Collision before the deadline cannot require a post-contact stop.
        row=branch_audit(state,cfg,'stop_hold',{'collisions':[{'role':'ego','time_s':3.} ]},diag)
        self.assertNotIn('ego_did_not_stop_by_declared_deadline',row['failure_reason'])
        self.assertFalse(row['checks']['ego_physical_execution']['stop_deadline_checked'])

    def test_upstream_deadline_stop_is_valid_without_forcing_passage(self):
        cfg,_,ext=fixture();cfg['duration_s']=4.
        cfg['action']['profiles']['yield_wait'].update(stop_deadline_s=3.5,stop_margin_m=.75)
        positions=np.broadcast_to([8.,0.,0.],(121,3));npc=np.broadcast_to([20.,0.,0.],(121,3))
        state,diag=recorded(cfg,ext,'yield_wait',ego_speed=np.zeros(121),ego_position=positions,npc_position=npc,npc_speed=0.)
        row=branch_audit(state,cfg,'yield_wait',{'collisions':[]},diag)
        self.assertTrue(row['execution_pass'],row['failure_reason'])
        self.assertIn('waiting',row['observed_outcome']['ego'])
        self.assertTrue(row['checks']['ego_physical_execution']['observed_stop_by_deadline'])

    def test_reaction_uses_all_pairs_not_just_reference(self):
        cfg,ext=straight_spec();base,_=recorded(cfg,ext)
        states={b:copy.deepcopy(base) for b in branch_names(cfg)}
        states['stop_hold']['position'][41:,1,0]+=.4
        states['decelerate']['position'][41:,1,0]-=.4
        row=response_audit(states,cfg)['response_metrics']
        self.assertAlmostEqual(row['max_npc_displacement_diff_m'],.8)
        self.assertEqual(row['reactive_npc_count_05'],1)
        self.assertEqual(row['reactive_npc_count_10'],0)
        self.assertEqual(row['effect_strength'],'medium')

    def test_exact_thresholds_and_repeat_not_counted_as_effect(self):
        cfg,ext=straight_spec();base,_=recorded(cfg,ext)
        states={b:copy.deepcopy(base) for b in branch_names(cfg)}
        states['stop_hold']['position'][41:,1,0]+=.5
        states['maintain_repeat']['position'][41:,1,0]+=9.
        row=response_audit(states,cfg)['response_metrics']
        self.assertEqual(row['reactive_npc_count_05'],0)
        self.assertEqual(row['effect_strength'],'weak')
        self.assertFalse(row['threshold_05_above_repeat_noise'])

    def test_weak_reaction_is_not_failure(self):
        cfg,ext=straight_spec();base,_=recorded(cfg,ext)
        row=response_audit({b:copy.deepcopy(base) for b in branch_names(cfg)},cfg)
        self.assertEqual(row['response_metrics']['effect_strength'],'weak')
        self.assertTrue(row['unrelated_invariance']['passed'])
        self.assertFalse(row['response_metrics']['used_as_acceptance_gate'])

    def test_declared_unrelated_role_change_is_failure_without_relabeling(self):
        cfg,ext=straight_spec();cfg['response_contract']={'unrelated_roles':['cross']}
        base,_=recorded(cfg,ext);states={b:copy.deepcopy(base) for b in branch_names(cfg)}
        states['commit_go']['position'][60,1,1]+=.01
        row=response_audit(states,cfg)['unrelated_invariance']
        self.assertFalse(row['passed'])
        self.assertIn('declared_unrelated_npc_changed:cross',row['failure_reason'])
        self.assertEqual(row['contract']['unrelated_roles'],['cross'])

    def test_repeat_noise_cannot_enlarge_unrelated_tolerance(self):
        cfg,ext=straight_spec();cfg['response_contract']={'unrelated_roles':['cross']}
        base,_=recorded(cfg,ext);states={b:copy.deepcopy(base) for b in branch_names(cfg)}
        states['maintain_repeat']['position'][60,1,0]+=.001
        row=response_audit(states,cfg)['unrelated_invariance']
        self.assertFalse(row['passed'])
        self.assertFalse(row['roles']['cross']['repeat_noise_within_position_tolerance'])

    def test_unrelated_checks_velocity_and_heading_not_only_position(self):
        cfg,ext=straight_spec();cfg['response_contract']={'unrelated_roles':['cross']}
        base,_=recorded(cfg,ext);states={b:copy.deepcopy(base) for b in branch_names(cfg)}
        states['commit_go']['velocity'][60,1,0]+=.001
        self.assertFalse(response_audit(states,cfg)['unrelated_invariance']['passed'])
        states['commit_go']=copy.deepcopy(base);states['commit_go']['rotation_deg'][60,1,2]+=.01
        self.assertFalse(response_audit(states,cfg)['unrelated_invariance']['passed'])

    def test_npc_actor_policy_override_is_frozen_and_used(self):
        cfg,obs,ext=fixture();obs['position'][0]=[80.,0.,0.];obs['velocity'][1]=0.
        old=npc_target(1,obs,cfg,ext)[0]
        cfg['vehicles'][1]['npc_policy']={'idm':{'max_acceleration':.5}}
        new=npc_target(1,obs,cfg,ext)[0]
        self.assertLess(new,old)
        self.assertEqual(cfg['controller']['idm']['max_acceleration'],2.)

    def test_unknown_contract_and_ego_unrelated_are_rejected(self):
        cfg,_=straight_spec();cfg['execution_contract']={'invented_tolerance':1.}
        with self.assertRaises(ValueError):contracts(cfg)
        cfg.pop('execution_contract');cfg['response_contract']={'unrelated_roles':['ego']}
        with self.assertRaises(ValueError):contracts(cfg)

    def test_actual_t0_unreachable_conflict_rejects_empty_scene_not_weak_effect(self):
        cfg,ext=straight_spec();state,_=recorded(cfg,ext)
        row=scene_opportunity(state,cfg)
        self.assertFalse(row['passed'])
        self.assertFalse(row['uses_future_response'])
        cfg['action']['conflicts'][0]['participants'][0].update(entry_s=70.,exit_s=76.)
        self.assertTrue(scene_opportunity(state,cfg)['passed'])

    def test_already_completed_npc_merge_at_t0_is_invalid_merge_opportunity(self):
        cfg,ext=straight_spec();state,_=recorded(cfg,ext)
        cfg['action']['conflicts'][0]['participants'][0].update(entry_s=70.,exit_s=76.)
        cfg['vehicles'][1]['route_maneuver']={'source_interval_s_m':[40.,50.]}
        row=scene_opportunity(state,cfg)
        self.assertFalse(row['passed'])
        self.assertTrue(row['evidence'][0]['participants'][0]['merge_already_completed_at_t0'])

    def test_past_conflict_cannot_be_recovered_by_high_reachability_speed(self):
        cfg,ext=straight_spec();state,_=recorded(cfg,ext)
        cfg['action']['conflicts'][0]['participants'][0].update(entry_s=70.,exit_s=76.)
        state['position'][40,0,0]=80.
        self.assertFalse(scene_opportunity(state,cfg)['passed'])

    def test_follower_opportunity_accepts_near_rear_vehicle(self):
        cfg,ext=straight_spec();state,_=recorded(cfg,ext)
        cfg['action']['conflicts']=[]
        cfg['vehicles'][1]['route']=cfg['vehicles'][0]['route']
        state['position'][:,1]=state['position'][:,0]-[10.,0.,0.]
        state['rotation_deg'][:,1]=0.;state['velocity'][:,1]=[8.,0.,0.]
        self.assertTrue(scene_opportunity(state,cfg)['passed'])

    def test_far_follower_outside_route_endpoint_not_mistaken_for_nearby(self):
        cfg,ext=straight_spec();state,_=recorded(cfg,ext)
        cfg['action']['conflicts']=[]
        cfg['vehicles'][1]['route']=[[-400.,0.,0.,0.],[100.,0.,0.,0.]]
        state['position'][:,1]=state['position'][:,0]-[200.,0.,0.]
        state['rotation_deg'][:,1]=0.;state['velocity'][:,1]=[8.,0.,0.]
        self.assertFalse(scene_opportunity(state,cfg)['passed'])


if __name__=='__main__':unittest.main()
