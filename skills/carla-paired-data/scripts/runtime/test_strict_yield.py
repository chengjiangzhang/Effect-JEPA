"""Strict observed-actor waiting and re-entry guards; synthetic states only."""
import copy
import unittest
import numpy as np
from controllers_v2 import ego_decision_context, decision_conflict_observation
from ego_decisions import EgoDecisionState, STRICT_YIELD, STRICT_HOLD, validate_profiles
from audit_execution import branch_audit, replay_policy, waiting_footprint_audit
from test_decisions import fixture
from test_execution import recorded


def strict_fixture():
    cfg,obs,ext=fixture()
    cfg['action']['profiles']['yield_wait'].update(release_condition=STRICT_YIELD,
        wait_for_roles=['cross'],min_wait_s=0.)
    cfg['action']['profiles']['stop_hold']['hold_policy']=STRICT_HOLD
    obs['bbox_offset']=np.zeros((2,3));obs['bbox_rotation_deg']=np.zeros((2,3))
    return cfg,obs,ext


def participant(cfg,obs,ext):
    return decision_conflict_observation(cfg['action']['conflicts'][0],obs,cfg,ext)['participants'][0]


def evaluate(state,cfg,obs,ext,t=0.,v0=4.):
    return state.evaluate('yield_wait',cfg['action']['profiles']['yield_wait'],t,v0,
                          ego_decision_context(obs,cfg,ext))


def queue(cfg,obs,ext,x=18.):
    cfg['vehicle_roles'].append('queue')
    cfg['vehicles'].append({**copy.deepcopy(cfg['vehicles'][0]),'role':'queue'})
    for key,value in [('position',[x,0,0]),('velocity',[0,0,0]),('rotation_deg',[0,0,0]),
                      ('bbox_offset',[0,0,0]),('bbox_rotation_deg',[0,0,0])]:
        obs[key]=np.vstack([obs[key],value])
    return np.vstack([ext,[2,1,.8]])


class StrictYieldTests(unittest.TestCase):
    def test_stationary_before_entry_never_means_cleared(self):
        cfg,obs,ext=strict_fixture();obs['position'][1,1]=-10.;obs['velocity'][1]=0
        row=participant(cfg,obs,ext)
        self.assertIsNone(row['arrival_s']);self.assertFalse(row['body_cleared'])
        state=EgoDecisionState()
        for t in (0.,.25,2.,4.):
            target,hold,d=evaluate(state,cfg,obs,ext,t)
            self.assertIsNone(d['released_at_s']);self.assertEqual(d['pending_wait_roles'],['cross'])
            self.assertEqual(target,0.);self.assertTrue(hold)
        self.assertTrue(d['timed_out'])

    def test_front_and_center_clear_but_tail_still_inside(self):
        cfg,obs,ext=strict_fixture();obs['position'][1,1]=5.
        row=participant(cfg,obs,ext)
        self.assertGreater(row['route_s'],34.)
        self.assertEqual(row['body_route_min_s'],33.)
        self.assertFalse(row['body_cleared'])
        obs['position'][1,1]=6.1
        self.assertTrue(participant(cfg,obs,ext)['body_cleared'])

    def test_actual_box_offset_and_local_rotation_used(self):
        cfg,obs,ext=strict_fixture();obs['position'][1,1]=6.1
        obs['bbox_offset'][1,0]=-3.
        self.assertFalse(participant(cfg,obs,ext)['body_cleared'])
        obs['bbox_offset'][1]=0.;obs['position'][1,1]=5.5
        self.assertFalse(participant(cfg,obs,ext)['body_cleared'])
        obs['bbox_rotation_deg'][1,2]=90.
        self.assertTrue(participant(cfg,obs,ext)['body_cleared'])

    def test_missing_actual_box_metadata_fails_closed(self):
        cfg,obs,ext=strict_fixture();del obs['bbox_offset']
        with self.assertRaisesRegex(ValueError,'actual bbox'):
            evaluate(EgoDecisionState(),cfg,obs,ext)

    def test_rolling_release_does_not_force_stop(self):
        cfg,obs,ext=strict_fixture();obs['position'][0,0]=10.;obs['velocity'][0,0]=3.
        obs['position'][1,1]=10.
        target,hold,d=evaluate(EgoDecisionState(),cfg,obs,ext,v0=3.)
        self.assertGreater(target,3.);self.assertFalse(hold)
        self.assertEqual(d['released_at_s'],0.);self.assertTrue(d['rolling_release'])
        self.assertIsNone(d['stopped_at_s']);self.assertTrue(d['entry_permitted_now'])

    def test_named_target_clear_but_other_queue_blocks_entry(self):
        cfg,obs,ext=strict_fixture();obs['position'][1,1]=10.
        ext=queue(cfg,obs,ext)
        target,hold,d=evaluate(EgoDecisionState(),cfg,obs,ext)
        self.assertTrue(d['all_wait_targets_cleared']);self.assertFalse(d['entry_clear'])
        self.assertEqual(d['entry_blocking_roles'],['queue'])
        self.assertIsNone(d['released_at_s']);self.assertEqual(target,0.);self.assertTrue(hold)

    def test_new_obstacle_after_release_brakes_then_reopens(self):
        cfg,obs,ext=strict_fixture();obs['position'][1,1]=10.
        ext=queue(cfg,obs,ext,x=70.);state=EgoDecisionState()
        self.assertIsNotNone(evaluate(state,cfg,obs,ext)[2]['released_at_s'])
        obs['position'][2,0]=18.;obs['velocity'][0,0]=3.
        target,hold,d=evaluate(state,cfg,obs,ext,t=.5)
        self.assertEqual(target,0.);self.assertTrue(hold);self.assertTrue(d['reblocked'])
        self.assertEqual(d['phase'],'released_reblocked');self.assertEqual(d['released_at_s'],0.)
        obs['position'][2,0]=70.
        target,hold,d=evaluate(state,cfg,obs,ext,t=.6)
        self.assertGreater(target,0.);self.assertFalse(hold);self.assertFalse(d['reblocked'])

    def test_target_returns_to_conflict_revokes_current_permission(self):
        cfg,obs,ext=strict_fixture();obs['position'][1,1]=10.;state=EgoDecisionState()
        evaluate(state,cfg,obs,ext)
        obs['position'][1,1]=0.
        target,hold,d=evaluate(state,cfg,obs,ext,t=1.)
        self.assertTrue(d['reblocked']);self.assertEqual(target,0.)
        self.assertEqual(d['pending_wait_roles'],['cross'])

    def test_stopped_after_release_cannot_restart_into_occupied_entry(self):
        cfg,obs,ext=strict_fixture();obs['position'][1,1]=10.
        ext=queue(cfg,obs,ext,x=70.);state=EgoDecisionState()
        evaluate(state,cfg,obs,ext)
        obs['position'][2,0]=25.  # outside immediate overlap, inside conflict entry
        target,hold,d=evaluate(state,cfg,obs,ext,t=.5)
        self.assertTrue(d['reblocked']);self.assertEqual(target,0.);self.assertTrue(hold)

    def test_behind_vehicle_does_not_block_forward_entry(self):
        cfg,obs,ext=strict_fixture();obs['position'][1,1]=10.;ext=queue(cfg,obs,ext,x=5.)
        self.assertTrue(evaluate(EgoDecisionState(),cfg,obs,ext)[2]['entry_permitted_now'])

    def test_elevated_vehicle_does_not_block_ground_route(self):
        cfg,obs,ext=strict_fixture();obs['position'][1,1]=10.;ext=queue(cfg,obs,ext)
        obs['position'][2,2]=10.
        self.assertTrue(evaluate(EgoDecisionState(),cfg,obs,ext)[2]['entry_permitted_now'])

    def test_pedestrian_beyond_underestimated_interval_still_in_ego_sweep(self):
        cfg,obs,ext=strict_fixture()
        cfg['action']['profiles']['yield_wait']['wait_for_roles']=['walker']
        cfg['action']['conflicts'][0]['participants']=[{'role':'walker','entry_s':29.8,'exit_s':30.2}]
        cfg['pedestrians']=[{'role':'walker','route_points':[[20,-30,0],[20,70,0]]}]
        obs['pedestrians']={'roles':['walker'],'position':np.array([[20,.6,0]]),
            'velocity':np.array([[0,1.,0]]),'rotation_deg':np.zeros((1,3)),
            'bbox_extent':np.array([[.3,.3,.8]]),'bbox_offset':np.zeros((1,3)),
            'bbox_rotation_deg':np.zeros((1,3))}
        row=participant(cfg,obs,ext)
        self.assertTrue(row['body_past_declared_exit']);self.assertTrue(row['overlaps_ego_conflict_sweep'])
        self.assertFalse(row['body_cleared'])
        obs['pedestrians']['position'][0,1]=2.
        self.assertTrue(participant(cfg,obs,ext)['body_cleared'])

    def test_stop_hold_never_automatically_resumes(self):
        cfg,obs,ext=strict_fixture();state=EgoDecisionState()
        target,hold,d=state.evaluate('stop_hold',cfg['action']['profiles']['stop_hold'],4.1,4.,
                                     ego_decision_context(obs,cfg,ext))
        self.assertEqual(target,0.);self.assertTrue(hold);self.assertIsNone(d['released_at_s'])
        self.assertEqual(d['hold_policy'],STRICT_HOLD)

    def test_roles_cannot_omit_declared_target_or_use_dwell_shortcut(self):
        cfg,_,_=strict_fixture();validate_profiles(cfg['action'])
        cfg['action']['profiles']['yield_wait']['wait_for_roles']=[]
        with self.assertRaisesRegex(ValueError,'all declared'):validate_profiles(cfg['action'])
        cfg,_,_=strict_fixture();cfg['action']['profiles']['yield_wait']['min_wait_s']=.25
        with self.assertRaisesRegex(ValueError,'rolling release'):validate_profiles(cfg['action'])

    def test_replay_rejects_forged_clearance_diagnostic(self):
        cfg,_,ext=strict_fixture();cfg['duration_s']=4.
        position=np.tile([14.,0.,0.],(121,1));npc=np.tile([20.,-10.,0.],(121,1))
        state,rows=recorded(cfg,ext,'yield_wait',ego_speed=np.zeros(121),ego_position=position,
                            npc_speed=0.,npc_position=npc)
        self.assertTrue(replay_policy(state,cfg,'yield_wait',rows)[0]['passed'])
        rows[60][0]['all_wait_targets_cleared']=True
        report,_=replay_policy(state,cfg,'yield_wait',rows)
        self.assertFalse(report['passed'])
        self.assertIn('all_wait_targets_cleared',[x['field'] for x in report['diagnostic_errors']])

    def test_valid_continuous_wait_does_not_require_target_passage(self):
        cfg,_,ext=strict_fixture();cfg['duration_s']=4.
        cfg['vehicles'][1]['desired_speed_mps']=.01
        position=np.tile([14.,0.,0.],(121,1));npc=np.tile([20.,-10.,0.],(121,1))
        state,rows=recorded(cfg,ext,'yield_wait',ego_speed=np.zeros(121),ego_position=position,
                            npc_speed=0.,npc_position=npc)
        report=branch_audit(state,cfg,'yield_wait',{'collisions':[]},rows)
        self.assertTrue(report['execution_pass'],report['failure_reason'])
        semantic=report['checks']['ego_physical_execution']['yield_semantics']
        self.assertTrue(semantic['target_passage_required_for_release'])
        self.assertFalse(semantic['target_passage_required_for_scene_acceptance'])
        self.assertFalse(semantic['targets_cleared_at_end'])
        self.assertIn('wait_timeout_hold',report['observed_outcome']['ego'])

    def test_rolling_release_exempts_unneeded_stop_deadline(self):
        cfg,_,ext=strict_fixture();cfg['duration_s']=4.
        cfg['action']['profiles']['yield_wait'].update(stop_deadline_s=3.5)
        npc=np.tile([20.,20.,0.],(121,1));npc[:,1]+=np.arange(121)*.2
        state,rows=recorded(cfg,ext,'yield_wait',ego_speed=np.full(121,3.),npc_position=npc)
        report=branch_audit(state,cfg,'yield_wait',{'collisions':[]},rows)
        physical=report['checks']['ego_physical_execution']
        self.assertFalse(physical['stop_deadline_checked'])
        self.assertTrue(physical['stop_deadline_released_before_requirement'])
        self.assertNotIn('ego_did_not_stop_by_declared_deadline',report['failure_reason'])

    def test_stationary_body_intrusion_fails_even_without_center_overshoot(self):
        for branch in ('yield_wait','stop_hold'):
            with self.subTest(branch=branch):
                cfg,_,ext=strict_fixture();cfg['duration_s']=4.
                cfg['vehicles'][1]['desired_speed_mps']=.01
                ext[0,0]=7.  # measured long body crosses entry20 while center14 meets stop14
                pos=np.tile([14.,0.,0.],(121,1));npc=np.tile([20.,-10.,0.],(121,1))
                state,rows=recorded(cfg,ext,branch,ego_speed=np.zeros(121),ego_position=pos,
                                    npc_speed=0.,npc_position=npc)
                report=branch_audit(state,cfg,branch,{'collisions':[]},rows)
                physical=report['checks']['ego_physical_execution']
                self.assertAlmostEqual(physical['max_stop_overshoot_m'],0.)
                self.assertFalse(report['execution_pass'])
                self.assertIn('ego_waiting_body_intrudes_declared_conflict',report['failure_reason'])
                self.assertAlmostEqual(physical['waiting_footprint']['minimum_front_clearance_m'],-1.)

    def test_legal_upstream_stop_change_still_binds_single_conflict(self):
        cfg,_,ext=strict_fixture();cfg['duration_s']=4.
        cfg['vehicles'][1]['desired_speed_mps']=.01
        cfg['action']['profiles']['stop_hold']['stop_route_s']=10.
        pos=np.tile([10.,0.,0.],(121,1));npc=np.tile([20.,-10.,0.],(121,1))
        state,rows=recorded(cfg,ext,'stop_hold',ego_speed=np.zeros(121),ego_position=pos,
                            npc_speed=0.,npc_position=npc)
        report=branch_audit(state,cfg,'stop_hold',{'collisions':[]},rows)
        self.assertTrue(report['execution_pass'],report['failure_reason'])
        footprint=report['checks']['ego_physical_execution']['waiting_footprint']
        self.assertEqual(footprint['conflict_id'],'crossing');self.assertEqual(footprint['checked_frames'],81)
        self.assertEqual(footprint['minimum_front_clearance_m'],8.)

    def test_waiting_footprint_uses_actual_offset_and_heading(self):
        cfg,_,ext=strict_fixture()
        state={'position':np.array([[[14.,0.,0.],[20.,-10.,0.]]]),
            'velocity':np.zeros((1,2,3)),'rotation_deg':np.zeros((1,2,3)),
            'bbox_extent':ext,'bbox_offset':np.array([[5.,0.,0.],[0.,0.,0.]]),
            'bbox_rotation_deg':np.zeros((2,3)),'time_s':np.array([1.])}
        def audit():
            return waiting_footprint_audit(state,cfg,cfg['action']['profiles']['stop_hold'],
                [{'released_at_s':None}],[True],.25)
        self.assertFalse(audit()['passed'])
        state['rotation_deg'][0,0,2]=90.
        self.assertTrue(audit()['passed'])  # offset now lateral; actual front is15, not21

    def test_postrelease_and_postcontact_stops_not_relabelled_as_bad_wait(self):
        cfg,_,ext=strict_fixture()
        state={'position':np.array([[[23.,0.,0.],[20.,-10.,0.]]]),
            'velocity':np.zeros((1,2,3)),'rotation_deg':np.zeros((1,2,3)),
            'bbox_extent':ext,'bbox_offset':np.zeros((2,3)),
            'bbox_rotation_deg':np.zeros((2,3)),'time_s':np.array([2.])}
        for mask,release in (([True],1.),([False],None)):
            result=waiting_footprint_audit(state,cfg,cfg['action']['profiles']['yield_wait'],
                [{'released_at_s':release}],mask,.25)
            self.assertTrue(result['passed']);self.assertEqual(result['checked_frames'],0)

    def test_ambiguous_hold_requires_explicit_conflict_binding(self):
        cfg,_,ext=strict_fixture()
        cfg['action']['conflicts'].append({**cfg['action']['conflicts'][0],'id':'second'})
        state={'velocity':np.zeros((0,2,3)),'time_s':np.array([])}
        profile=cfg['action']['profiles']['stop_hold']
        report=waiting_footprint_audit(state,cfg,profile,[],[],.25)
        self.assertFalse(report['passed']);self.assertIn('ambiguous',report['failure_reason'])
        profile['conflict_id']='crossing'
        self.assertTrue(waiting_footprint_audit(state,cfg,profile,[],[],.25)['passed'])


if __name__=='__main__':unittest.main()
