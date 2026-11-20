"""Offline measured-t0 planning contracts; no simulator or future lookahead."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

if __package__:
    from . import measured_t0 as m
    from .action_design import profiles_for_template
else:
    import measured_t0 as m
    from action_design import profiles_for_template


def fixture():
    conflict = {'id':'c','ego_entry_s':24.,'ego_exit_s':34.,'stop_route_s':19.7,
        'participants':[{'role':'npc','actor_type':'vehicle','entry_s':24.,'exit_s':34.}],
        'ego_must_yield':True}
    profiles = profiles_for_template('ego_merge_ramp',[],conflict)
    # Explicit fixture values, independent of future changes to the default library.
    for profile in profiles.values():
        if 'response_time_s' in profile: profile['response_time_s']=.75
    spec = {'scene_id':'scene_0000','history_s':2.,'duration_s':4.,'dt_s':.05,
        'vehicle_roles':['ego','npc'], 'vehicles':[
            {'role':'ego','route':[[0.,0.,0.,0.],[100.,0.,0.,0.]],'spawn_route_s':4.,
             'initial_speed_mps':4.,'desired_speed_mps':4.},
            {'role':'npc','route':[[0.,4.,0.,0.],[100.,4.,0.,0.]],'spawn_route_s':14.,
             'initial_speed_mps':3.,'desired_speed_mps':4.,'npc_policy':{'idm':{'max_acceleration':2.}}}],
        'controller':{'idm':{'max_acceleration':2.}},
        'action':{'schema_version':2,'profile_version':2,'profiles':profiles,'profile_order':list(profiles),
            'available_lane_change_directions':[], 'conflicts':[conflict],
            'nominal_policy':{'desired_speed_m_s':4.,'idm':{'max_acceleration':2.,'comfortable_brake':2.8,
                'time_headway':1.25,'minimum_gap':2.5},'release_gap_s':1.5,
                'comfortable_deceleration_m_s2':4.5,'response_time_s':.75,'respect_signals':True}},
        'geometry':{'traffic_design':{'authored_initial_conditions':{
            'estimated_t0_route_s':12.,'estimated_t0_speed_m_s':4.}}},'code_sha256':{'controller':'frozen'}}
    times = np.arange(41)*.05-2.
    position = np.zeros((41,2,3)); position[:,0,0]=10+times*3;position[:,1,0]=20+times*3;position[:,1,1]=4.
    velocity = np.zeros_like(position); velocity[:,:,0]=3.
    states={'time_s':times,'position':position,'velocity':velocity,'rotation_deg':np.zeros_like(position),
        'bbox_extent':np.tile([2.5,1.,.8],(2,1)), 'bbox_offset':np.zeros((2,3)),
        'bbox_rotation_deg':np.zeros((2,3)), 'frame_id':np.arange(100,141),
        'simulation_time_s':np.arange(41)*.05+5., 'control':np.zeros((41,2,3)),
        'target_speed':np.zeros((41,2))}
    states['transform_world_vehicle'] = np.tile(np.eye(4), (41,2,1,1))
    states['transform_world_vehicle'][:,:,:3,3] = position
    spec['spec_sha256']=m._spec_digest(spec)
    return spec,states


def save_history(root,spec,states,**updates):
    scene=Path(root)/spec['scene_id']; branch=scene/'branches/common_history';branch.mkdir(parents=True)
    (scene/'spec.json').write_text(json.dumps(spec,indent=2))
    np.savez_compressed(branch/'states.npz',**states)
    meta={'status':'complete','capture_mode':m.HISTORY_MODE,'history_only_complete':True,
        't0_index':40,'spec_sha256':spec['spec_sha256'],'code_sha256':spec['code_sha256'],
        'vehicle_roles':spec['vehicle_roles'],'t0_frame':int(states['frame_id'][-1]),'collisions':[]}
    meta.update(updates);(branch/'meta.json').write_text(json.dumps(meta))
    manifest={'status':'complete','capture_mode':m.HISTORY_MODE,'history_only':True,'future_s':0.,
        'code_sha256':spec['code_sha256'],'frozen_history_spec_sha256':{spec['scene_id']:spec['spec_sha256']},
        'input_spec_file_sha256':{spec['scene_id']:m.file_sha(scene/'spec.json')},
        'branch_results':[{'scene':spec['scene_id'],'branch':'common_history','status':'complete',
            'states_sha256':m.file_sha(branch/'states.npz'),'meta_sha256':m.file_sha(branch/'meta.json')}]}
    (Path(root)/'manifest.json').write_text(json.dumps(manifest))
    return branch


class MeasuredT0Tests(unittest.TestCase):
    def test_original_profile_plan_is_checked_not_falsely_replanned(self):
        spec,states=fixture(); before=copy.deepcopy(spec)
        with tempfile.TemporaryDirectory() as root:
            b=save_history(root,spec,states)
            result=m.plan_from_history(spec,root)
            self.assertEqual(result['status'],'passed')
            self.assertEqual(spec,before)
            planned=result['planned_spec']; proof=planned['measured_t0_plan']
            self.assertEqual(proof['plan_mode'],'checked_original_profiles')
            self.assertTrue(proof['actual_t0_checked']);self.assertFalse(proof['future_execution_verified'])
            self.assertEqual(planned['action'],spec['action'])
            self.assertEqual(planned['vehicles'],spec['vehicles'])
            self.assertEqual(proof['history']['states_sha256'],m.file_sha(b/'states.npz'))
            self.assertEqual(proof['evidence_sha256'],m.digest(result['evidence']))
            self.assertEqual(planned['spec_sha256'],m._spec_digest(planned))
            self.assertNotEqual(planned['spec_sha256'],spec['spec_sha256'])

    def test_real_t0_drift_replaces_no_static_values(self):
        spec,states=fixture(); e=m.analyze_t0(spec,states)
        self.assertEqual(e['actors']['ego']['speed_m_s'],3.)
        self.assertEqual(e['static_design_drift']['measured_minus_estimated_speed_m_s'],-1.)
        self.assertEqual(e['static_design_drift']['measured_minus_estimated_route_s_m'],-2.)
        self.assertEqual(e['conflicts']['c']['participants'][0]['front_to_entry_m'],1.5)

    def test_strict_wait_has_no_forced_dwell_or_promised_release(self):
        spec,states=fixture(); e=m.analyze_t0(spec,states)
        self.assertTrue(e['passed'])
        codes=[x['code'] for x in e['warnings']]
        self.assertNotIn('stop_deadline_does_not_reserve_recovery_interval',codes)
        b=e['branches']['yield_wait']['wait_and_recovery_budget']
        self.assertAlmostEqual(b['deadline_based_recovery_observation_end_s'],4.)
        self.assertTrue(b['timeout_waiting_remains_valid_execution'])
        self.assertLess(b['optimistic_recovery_observation_end_s'],4.)
        self.assertTrue(b['may_release_while_approaching_without_full_stop'])
        self.assertTrue(b['optimistic_stop_then_minimum_wait_is_not_release_prediction'])

    def test_actual_t0_stop_distance_and_deadline_checked(self):
        spec,states=fixture();states['position'][:,0,0]+=8.
        e=m.analyze_t0(spec,states)
        self.assertFalse(e['passed'])
        self.assertIn('measured_t0_stop_distance_budget_insufficient',[x['code'] for x in e['failure_reasons']])
        states['position'][:,0,0]-=8.;states['velocity'][:,0,0]=15.
        e=m.analyze_t0(spec,states)
        self.assertIn('measured_t0_stop_deadline_infeasible',[x['code'] for x in e['failure_reasons']])

    def test_optimistic_stop_wait_recovery_can_be_impossible(self):
        spec,states=fixture();spec['action']['profiles']['yield_wait']['min_wait_s']=3.
        spec['action']['profiles']['yield_wait'].pop('release_condition')
        spec['action']['profiles']['yield_wait'].pop('wait_for_roles')
        e=m.analyze_t0(spec,states)
        self.assertIn('no_optimistic_stop_wait_recovery_budget',[x['code'] for x in e['failure_reasons']])

    def test_delayed_commit_stops_from_t0_not_after_onset(self):
        spec,states=fixture(); e=m.analyze_t0(spec,states)
        expected=3.*.75+3.*3./9.
        self.assertAlmostEqual(e['branches']['commit_delayed']['stop_budget']['required_response_braking_margin_m'],expected)

    def test_waiting_object_stopped_is_not_already_cleared(self):
        spec,states=fixture();states['velocity'][:,1]=0.
        e=m.analyze_t0(spec,states);p=e['conflicts']['c']['participants'][0]
        self.assertIsNone(p['constant_speed_entry_s']);self.assertFalse(p['already_cleared_at_t0'])
        self.assertTrue(p['can_reach_entry_within_window_under_envelope'])
        self.assertTrue(e['passed']) # Can move under fixed policy, not forced to do so.
        self.assertIn('participant_opportunity_requires_acceleration_or_policy_release',[x['code'] for x in e['warnings']])

    def test_all_waiting_objects_already_cleared_requires_shared_setup_repair(self):
        spec,states=fixture();states['position'][:,1,0]+=30
        e=m.analyze_t0(spec,states)
        self.assertFalse(e['passed'])
        self.assertIn('no_pending_participant_can_reach_declared_conflict_under_optimistic_envelope',
                      [x['code'] for x in e['failure_reasons']])

    def test_no_ego_can_reach_conflict_is_design_failure_not_outcome_filter(self):
        spec,states=fixture();spec['action']['conflicts'][0]['ego_entry_s']=80.;spec['action']['conflicts'][0]['ego_exit_s']=90.
        e=m.analyze_t0(spec,states)
        self.assertIn('no_ego_entry_condition_reaches_conflict_under_optimistic_envelope',[x['code'] for x in e['failure_reasons']])
        self.assertTrue(any('No future reaction' in x for x in e['limitations']))

    def test_lane_completion_budget_unchanged(self):
        spec,states=fixture();p=spec['action'];p['available_lane_change_directions']=['left']
        p['profiles']['commit_delayed']={'kind':'lane_change','direction':'left','onset_s':1.,'duration_s':2.2,
            'settling_time_s':.9,'required_final_hold_s':.5,'acceleration_m_s2':0.,'speed_increase_cap_m_s':0.}
        e=m.analyze_t0(spec,states)
        self.assertIn('lane_transition_settling_and_hold_exceed_window',[x['code'] for x in e['failure_reasons']])

    def test_explicit_onset_adjustment_only_and_recorded(self):
        spec,states=fixture();states['position'][:,0,0]+=4.
        spec['action']['profiles']['yield_wait']['onset_s']=1.;spec['spec_sha256']=m._spec_digest(spec)
        with tempfile.TemporaryDirectory() as root:
            save_history(root,spec,states)
            self.assertEqual(m.plan_from_history(spec,root)['status'],'failed')
            bounds={'yield_wait':{'onset_s':[0.,1.]}}
            result=m.plan_from_history(spec,root,timing_bounds=bounds)
            self.assertEqual(result['status'],'passed');planned=result['planned_spec']
            self.assertLess(planned['action']['profiles']['yield_wait']['onset_s'],1.)
            self.assertEqual(planned['action']['profiles']['yield_wait']['stop_deadline_s'],3.5)
            self.assertEqual(planned['action']['profiles']['yield_wait']['timeout_s'],4.)
            self.assertEqual(planned['measured_t0_plan']['plan_mode'],'adjusted_explicit_onsets')
            self.assertTrue(m.assert_shared_setup_unchanged(spec,planned,bounds))
            self.assertTrue(result['evidence']['original_profile_failure_reasons'])

    def test_shared_setup_mutation_is_always_rejected(self):
        spec,_=fixture()
        for mutation in ('npc','route','history','deadline','timeout','newbranch','onset'):
            p=copy.deepcopy(spec)
            if mutation=='npc':p['vehicles'][1]['desired_speed_mps']+=.1
            elif mutation=='route':p['vehicles'][0]['route'][0][0]+=.1
            elif mutation=='history':p['history_s']=3.
            elif mutation=='deadline':p['action']['profiles']['yield_wait']['stop_deadline_s']=4.
            elif mutation=='timeout':p['action']['profiles']['yield_wait']['timeout_s']=5.
            elif mutation=='newbranch':p['action']['profiles']['extra']={}
            else:p['action']['profiles']['commit_delayed']['onset_s']=.5
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                m.assert_shared_setup_unchanged(spec,p)

    def test_non_onset_bounds_and_unbounded_baseline_changes_rejected(self):
        spec,_=fixture()
        bad=[{'yield_wait':{'stop_deadline_s':[3.,4.]}},{'maintain':{'onset_s':[0.,1.]}},
             {'yield_wait':{'onset_s':[.1,1.]}},{'yield_wait':{'onset_s':[0.,.07]}},
             {'yield_wait':{'onset_s':[0.,True]}}]
        for bounds in bad:
            with self.subTest(bounds=bounds),self.assertRaises(ValueError):m._bounds(spec,bounds)

    def test_incomplete_wrong_source_wrong_t0_and_non_history_evidence_rejected(self):
        spec,states=fixture()
        updates=[{'status':'partial'},{'capture_mode':'physics_only_preflight_not_delivery'},
            {'history_only_complete':False},{'t0_index':39},{'t0_frame':999},
            {'code_sha256':{'controller':'other'}},{'vehicle_roles':['npc','ego']}]
        for patch in updates:
            with self.subTest(patch=patch),tempfile.TemporaryDirectory() as root:
                save_history(root,spec,states,**patch)
                with self.assertRaises(m.HistoryEvidenceError):m.plan_from_history(spec,root)

    def test_no_future_lookahead_and_no_executed_t0_control(self):
        spec,states=fixture()
        for key in ('time_s','control','target_speed','frame_id','position'):
            broken=copy.deepcopy(states)
            if key=='time_s':broken[key][-1]=.05
            elif key=='frame_id':broken[key][-1]+=1
            elif key=='position':broken[key][-1,0,0]=np.nan
            else:broken[key][-1]=.1
            with self.subTest(key=key),tempfile.TemporaryDirectory() as root:
                save_history(root,spec,broken)
                with self.assertRaises(m.HistoryEvidenceError):m.plan_from_history(spec,root)
        broken=copy.deepcopy(states);broken['time_s'][-1]=.05
        with self.assertRaises(m.HistoryEvidenceError):m.analyze_t0(spec,broken)

    def test_initial_spec_and_recorded_spec_must_match_exactly(self):
        spec,states=fixture()
        with tempfile.TemporaryDirectory() as root:
            save_history(root,spec,states)
            other=copy.deepcopy(spec);other['vehicles'][1]['desired_speed_mps']+=1.
            with self.assertRaises(m.HistoryEvidenceError):m.plan_from_history(other,root)
            other['spec_sha256']=m._spec_digest(other)
            with self.assertRaises(m.HistoryEvidenceError):m.plan_from_history(other,root)

    def test_missing_pedestrian_or_signal_measurements_do_not_get_invented(self):
        spec,states=fixture();spec['pedestrians']=[{'role':'walker','route_points':[[20.,0.,0.],[20.,10.,0.]]}]
        with self.assertRaises(m.HistoryEvidenceError):m.analyze_t0(spec,states)
        spec.pop('pedestrians');spec['traffic_signal_program']={'groups':[]}
        spec['spec_sha256']=m._spec_digest(spec)
        with tempfile.TemporaryDirectory() as root:
            save_history(root,spec,states)
            with self.assertRaises(m.HistoryEvidenceError):m.plan_from_history(spec,root)

    def test_kinematic_integrals_include_speed_cap_and_static_impossibility(self):
        self.assertAlmostEqual(m.travel_bound(4.,2.,4.,6.),23.)
        self.assertIsNone(m.arrival_bound(10.,2.,0.))
        self.assertAlmostEqual(m.arrival_bound(8.,2.,0.),4.)
        self.assertAlmostEqual(m.arrival_bound(4.,0.,2.),2.)
        self.assertEqual(m.arrival_bound(-1.,0.),0.)
        with self.assertRaises(ValueError):m.travel_bound(2.,float('nan'),4.)


    def test_history_contact_is_shared_setup_failure_not_infrastructure(self):
        spec,states=fixture()
        with tempfile.TemporaryDirectory() as root:
            save_history(root,spec,states,collisions=[{'time_s':-1.,'role':'ego','other_role':'npc'}])
            result=m.plan_from_history(spec,root)
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['failure_reasons'][0]['code'],'initialization_or_common_history_contact')
            self.assertIsNone(result['planned_spec'])

    def test_manifest_incomplete_or_tampered_history_is_infrastructure_error(self):
        spec,states=fixture()
        for change in ('status','states','meta','identity'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as root:
                branch=save_history(root,spec,states);p=Path(root)/'manifest.json';manifest=json.loads(p.read_text())
                if change=='status':manifest['status']='prepared'
                elif change=='identity':manifest['frozen_history_spec_sha256'][spec['scene_id']]='wrong'
                elif change=='states':manifest['branch_results'][0]['states_sha256']='wrong'
                else:manifest['branch_results'][0]['meta_sha256']='wrong'
                p.write_text(json.dumps(manifest))
                with self.assertRaises(m.HistoryEvidenceError):m.plan_from_history(spec,root)

    def test_implicit_original_zero_onset_can_be_explicitly_adjusted(self):
        spec,_=fixture();spec['action']['profiles']['yield_wait'].pop('onset_s')
        adjusted=copy.deepcopy(spec);adjusted['action']['profiles']['yield_wait']['onset_s']=.25
        self.assertTrue(m.assert_shared_setup_unchanged(spec,adjusted,{'yield_wait':{'onset_s':[0.,.25]}}))

    def test_scheduled_stationary_walker_has_an_opportunity_without_reactive_claim(self):
        spec,states=fixture()
        for start,distance,expected in ((0.,4.,True),(1.,4.,True),(4.,4.,False),(5.,1.,False),(0.,9.,False)):
            walker={'role':'walker','route_points':[[0.,4.,0.],[100.,4.,0.]],'speed_mps':1.5,'start_time_s':start}
            measured={'actor_type':'pedestrian','role':'walker','body_front_route_s_m':10.,
                'body_rear_route_s_m':9.5,'route_speed_m_s':0.}
            p=m._participant(measured,{'entry_s':10.+distance,'exit_s':20.+distance},walker,spec)
            with self.subTest(start=start,distance=distance):
                self.assertEqual(p['can_reach_entry_within_window_under_envelope'],expected)
                self.assertIsNone(p['constant_speed_entry_s'])
                self.assertEqual(p['envelope_basis'],'declared_scripted_walk_start_and_speed_not_reactive_behavior')
                self.assertFalse(p['already_cleared_at_t0'])

    def test_reactive_walk_start_is_earliest_opportunity_not_departure_or_clearance(self):
        spec,_=fixture()
        walker={'role':'walker','route_points':[[0.,4.,0.],[100.,4.,0.]],'speed_mps':1.5,
                'start_time_s':.5,'response_policy':{'kind':'gap_acceptance_v1'}}
        measured={'actor_type':'pedestrian','role':'walker','body_front_route_s_m':10.,
                  'body_rear_route_s_m':9.5,'route_speed_m_s':0.}
        p=m._participant(measured,{'entry_s':13.,'exit_s':20.},walker,spec)
        self.assertEqual(p['optimistic_entry_s'],2.5)
        self.assertFalse(p['departure_time_known']);self.assertFalse(p['clearance_time_guaranteed'])
        self.assertIsNone(p['constant_speed_clear_s'])
        self.assertEqual(p['start_time_semantics'],'earliest_feedback_decision')

    def test_strict_wait_checks_measured_whole_body_clearance(self):
        spec,states=fixture()
        # Same center route as the valid fixture, but a wide actual vehicle
        # obstructs the adjacent target's swept path at the declared stop pose.
        states['bbox_extent'][0,1]=4.
        e=m.analyze_t0(spec,states)
        self.assertIn('waiting_ego_body_obstructs_target_swept_route',
                      [x['code'] for x in e['failure_reasons']])


    def test_stationary_scheduled_pedestrian_template_does_not_false_reject(self):
        spec,states=fixture()
        walker={'role':'walker','route_points':[[0.,4.,0.],[100.,4.,0.]],'speed_mps':1.5,'start_time_s':0.}
        spec['pedestrians']=[walker]
        spec['action']['conflicts'][0]['participants']=[{'role':'walker','actor_type':'pedestrian','entry_s':24.,'exit_s':34.}]
        spec['action']['profiles']['yield_wait']['wait_for_roles']=['walker']
        ped={key:(value[:,1:2].copy() if np.ndim(value)==3 else value[1:2].copy()
                  if key in ('bbox_extent','bbox_offset','bbox_rotation_deg') else value.copy())
             for key,value in states.items()}
        ped['velocity'][:]=0.
        e=m.analyze_t0(spec,states,ped)
        self.assertTrue(e['passed'])
        self.assertIn('walker',e['actors'])
        self.assertIsNone(e['conflicts']['c']['participants'][0]['constant_speed_entry_s'])
        self.assertIsNotNone(e['conflicts']['c']['participants'][0]['optimistic_entry_s'])

    def test_corrupt_npz_is_evidence_error_not_setup_failure(self):
        spec,states=fixture()
        with tempfile.TemporaryDirectory() as root:
            branch=save_history(root,spec,states);(branch/'states.npz').write_bytes(b'invalid')
            with self.assertRaises(m.HistoryEvidenceError):m.plan_from_history(spec,root)

    def test_actual_heading_rotates_box_and_relative_velocity(self):
        spec,states=fixture();states['rotation_deg'][-1,1,2]=90.;states['velocity'][-1,1,0]=4.
        e=m.analyze_t0(spec,states);npc=e['actors']['npc']
        self.assertAlmostEqual(npc['body_half_length_on_route_m'],1.)
        self.assertAlmostEqual(npc['relative_velocity_ego_t0_forward_m_s'],1.)
        self.assertEqual(npc['heading_error_deg'],90.)


    def _physics_fixture(self, root, *, release=None, contact=None, execution=True, censored=False):
        spec,prefix=fixture();spec['branches']=list(spec['action']['profiles'])+['maintain_repeat'];spec['spec_sha256']=m._spec_digest(spec)
        scene=Path(root)/spec['scene_id'];scene.mkdir();(scene/'spec.json').write_text(json.dumps(spec))
        qa={'spec_file_sha256':m.file_sha(scene/'spec.json'),'source_files':{},'branches':{}}
        for name in spec['branches']:
            folder=scene/'branches'/name;folder.mkdir(parents=True)
            state={key:(np.concatenate([value,np.repeat(value[-1:],80,axis=0)]) if key not in
                       ('bbox_extent','bbox_offset','bbox_rotation_deg') else value.copy()) for key,value in prefix.items()}
            state['time_s']=np.arange(121)*.05-2.;state['frame_id']=np.arange(100,221);state['simulation_time_s']=np.arange(121)*.05+5.
            state['velocity'][state['time_s']>=2.,0]=0.
            if release is not None:state['velocity'][state['time_s']>release+.1,0,0]=1.
            np.savez_compressed(folder/'states.npz',**state)
            collisions=[] if contact is None else [{'time_s':contact,'role':'ego','other_role':'npc'}]
            (folder/'meta.json').write_text(json.dumps({'status':'complete','capture_mode':'physics_only_preflight_not_delivery',
                'branch':name,'vehicle_roles':spec['vehicle_roles'],'spec_sha256':spec['spec_sha256'],
                'code_sha256':spec['code_sha256'],'collisions':collisions}))
            diag=[[{'stopped_at_s':2. if t>=2. else None,
                    'released_at_s':release if release is not None and t>=release else None,
                    'timed_out':bool(release is None and t>=4.),'phase':'released' if release is not None and t>=release else 'holding'},{}]
                  for t in state['time_s']]
            (folder/'controller_diagnostics.json').write_text(json.dumps(diag))
            qa['source_files'][name]={key:m.file_sha(folder/key) for key in ('states.npz','meta.json','controller_diagnostics.json')}
            qa['branches'][name]={'execution_pass':execution,'quality_status':'censored' if censored else 'passed' if execution else 'failed',
                'observed_outcome':{'ego':['waiting']}}
        return spec,scene,{'status':'passed' if execution else 'failed','scenes':{spec['scene_id']:qa}}

    def test_observed_late_release_is_reported_not_rejected_or_replanned(self):
        with tempfile.TemporaryDirectory() as root:
            spec,scene,qa=self._physics_fixture(root,release=3.6)
            before=copy.deepcopy(spec)
            report=m.summarize_observed_window(spec,scene,execution_report=qa)
            row=report['branches']['yield_wait']
            self.assertAlmostEqual(row['remaining_future_after_release_s'],.4)
            self.assertFalse(row['at_least_one_2hz_recovery_interval'])
            self.assertTrue(row['execution_pass']);self.assertIsNone(row['waiting_timeout_valid'])
            self.assertFalse(report['new_acceptance_gate']);self.assertFalse(report['used_to_replan_initial_conditions'])
            self.assertEqual(spec,before)

    def test_observed_timeout_needs_existing_execution_evidence_not_passage(self):
        for passed,expected in ((True,True),(False,False)):
            with self.subTest(passed=passed),tempfile.TemporaryDirectory() as root:
                spec,scene,qa=self._physics_fixture(root,execution=passed)
                result=m.summarize_observed_window(spec,scene,execution_report=qa)
                row=result['branches']['yield_wait']
                self.assertEqual(row['waiting_timeout_valid'],expected)
                self.assertIsNone(row['released_at_s']);self.assertTrue(row['no_release_does_not_imply_execution_failure'])
                without=m.summarize_observed_window(spec,scene)
                self.assertIsNone(without['branches']['yield_wait']['waiting_timeout_valid'])

    def test_observed_contact_censors_recovery_and_timeout_validity(self):
        with tempfile.TemporaryDirectory() as root:
            spec,scene,qa=self._physics_fixture(root,release=3.2,contact=3.4,censored=True)
            row=m.summarize_observed_window(spec,scene,execution_report=qa)['branches']['yield_wait']
            self.assertTrue(row['at_least_one_2hz_recovery_interval'])
            self.assertFalse(row['precontact_recovery_interval_available'])
            self.assertAlmostEqual(row['precontact_recovery_available_s'],.2)
        with tempfile.TemporaryDirectory() as root:
            spec,scene,qa=self._physics_fixture(root,contact=2.1,censored=True)
            row=m.summarize_observed_window(spec,scene,execution_report=qa)['branches']['yield_wait']
            self.assertIsNone(row['waiting_timeout_valid'])

    def test_observed_report_rejects_unbound_audit_or_incomplete_physics(self):
        with tempfile.TemporaryDirectory() as root:
            spec,scene,qa=self._physics_fixture(root)
            qa['scenes'][spec['scene_id']]['source_files']['yield_wait']['states.npz']='changed'
            with self.assertRaises(m.HistoryEvidenceError):m.summarize_observed_window(spec,scene,execution_report=qa)
            (scene/'branches/maintain_repeat/meta.json').write_text('{}')
            with self.assertRaises(m.HistoryEvidenceError):m.summarize_observed_window(spec,scene)

    def test_no_conflict_following_preserves_zero_weak_response_opportunities(self):
        spec,states=fixture();p=profiles_for_template('follower_response_straight',[],None)
        spec['action'].update(profiles=p,profile_order=list(p),conflicts=[])
        states['velocity'][:,0]=0.
        e=m.analyze_t0(spec,states)
        self.assertTrue(e['passed'])
        self.assertIn('deceleration_at_or_below_target_floor',[x['code'] for x in e['warnings']])


if __name__=='__main__':unittest.main()
