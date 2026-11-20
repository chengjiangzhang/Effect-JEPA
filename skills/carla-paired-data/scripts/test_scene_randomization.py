"""Behavioral invariants of scene sampling and acceptance-bias accounting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from interaction_design import PHASES
from scene_randomization import intent, individualize
from randomization_report import build_report, distribution, trajectory_similarity, conditional_counts, encounter_coverage


class SamplingTests(unittest.TestCase):
    def sample(self,template,phase,seed):
        return intent({'seed':seed,'interaction_phase':phase},{'template':template},{'layout_seed':seed})

    def test_phase_does_not_pin_count_speed_progress_or_style(self):
        for template,phases in PHASES.items():
            for phase in phases:
                samples=[self.sample(template,phase,n) for n in range(80)]
                counts={s['participants']['vehicle_count_including_ego'] for s in samples}
                self.assertGreaterEqual(len(counts),5,(template,phase))
                keys=('npc_green_s','queue_gap_m','action_acceleration_scale') if template=='cross_signalized' else (
                    'ego_initial_speed_m_s','road_anchor_shift_m' if template in ('follower_response_straight','ego_lane_change_straight','merge_straight') else 'queue_gap_m','action_acceleration_scale')
                self.assertTrue(set(keys).issubset(samples[0]['active_parameters']))
                values=np.array([[s['parameters'][k] for k in keys] for s in samples])
                self.assertTrue(np.all(np.std(values,axis=0)>1e-4))
                self.assertLess(abs(np.corrcoef(values.T)[0,1]),.5)
                self.assertLess(abs(np.corrcoef(values.T)[0,2]),.5)
                self.assertTrue(all(s['uses_future_outcomes'] is False for s in samples))

    def test_replay_and_new_seed(self):
        one=self.sample('merge_straight','partial_merge',10)
        self.assertEqual(one,self.sample('merge_straight','partial_merge',10))
        self.assertNotEqual(one,self.sample('merge_straight','partial_merge',11))
        self.assertLess(one['parameters']['merge_onset_s'],0.)
        altered=copy.deepcopy(one);altered['parameters']['queue_gap_m']=999
        self.assertNotEqual(one,altered)

    def test_extra_actor_does_not_redraw_existing_preferences_or_appearance(self):
        class Geometry:
            DIVERSE_NPC_MODELS=['a','b','c']
            DIVERSE_COLORS=['1,2,3','4,5,6']
        def actor(role):return {'role':role,'desired_speed_mps':5.,'blueprint_id':'a','blueprint_attributes':{'color':'1,2,3'}}
        cfg={'randomization':self.sample('merge_ramp','near_together',24),
            'controller':{'idm':{'max_acceleration':2.,'comfortable_brake':2.8,'time_headway':1.25,'minimum_gap':2.5}},
            'vehicles':[actor('ego'),actor('primary')]}
        extra=copy.deepcopy(cfg);extra['vehicles'].append(actor('secondary'))
        individualize(cfg,Geometry);individualize(extra,Geometry)
        self.assertEqual(cfg['vehicles'],extra['vehicles'][:2])
        self.assertNotEqual(extra['vehicles'][1]['npc_policy'],extra['vehicles'][2]['npc_policy'])

    def test_ego_and_pedestrian_only_has_valid_empty_unrelated_roles_schema(self):
        from test_orchestrator import Fixture
        from scene_spec import authoring_schema, authoring_payload
        from model_api import check_schema, validate_json, SchemaError
        cfg=Fixture.scene_spec();cfg['vehicles']=cfg['vehicles'][:1]
        schema=authoring_schema(cfg);check_schema(schema)
        payload=authoring_payload(cfg);validate_json(payload,schema)
        self.assertEqual(payload['unrelated_roles'],[])
        payload['unrelated_roles']=['nonexistent']
        with self.assertRaises(SchemaError):validate_json(payload,schema)

    def test_narrow_participant_counts_keep_both_heads_and_allow_no_followers(self):
        samples=[self.sample('narrow_meeting','waiting_gap',n)['participants'] for n in range(80)]
        self.assertTrue(any(s['narrow_ego_followers']==0 for s in samples))
        self.assertTrue(any(s['narrow_oncoming_followers']==0 for s in samples))
        self.assertTrue(any(s['narrow_ego_followers']>=2 for s in samples))
        for s in samples:
            self.assertEqual(2+s['narrow_ego_followers']+s['narrow_oncoming_followers'],s['vehicle_count_including_ego'])

    def test_narrow_opposing_initial_speed_is_not_bound_to_ego_speed(self):
        for phase in PHASES['narrow_meeting']:
            samples=[self.sample('narrow_meeting',phase,n) for n in range(80)]
            values=np.array([[s['parameters'][k] for k in ('ego_initial_speed_m_s','other_initial_speed_m_s')] for s in samples])
            self.assertLess(abs(np.corrcoef(values.T)[0,1]),.5)
            self.assertTrue(np.any(values[:,0]>values[:,1]))
            self.assertTrue(np.any(values[:,0]<values[:,1]))
            self.assertIn('other_initial_speed_m_s',samples[0]['active_parameters'])


class ReportingTests(unittest.TestCase):
    def test_waiting_exemption_and_cancel_command_are_not_motion_coverage(self):
        records=[{'template':'merge_straight','history_measurements':2,'physics_qualified':True,
            'revisions':[{}, {'rows':[{'within_target':True,'classification':'observed_waiting_no_free_speed_eta'},
                                     {'within_target':False}]}]}]
        merge={'policy_mode_counts':{'aborting':12},'policy_transitions':[{'after':'aborting'}],
            'observed_early_return':[],'observed_target_centre_reached':[],'target_gap_blocked':[]}
        observed=[{'status':'failed','physics_qualified':True,'template':'merge_straight',
            'observed_events':{'accelerate':{'merger':{'dynamic_merge':merge}}}}]
        report=encounter_coverage(records,observed);counts=report['by_template']['merge_straight']
        self.assertEqual(counts['observed_waiting_exempt'],1)
        self.assertEqual(counts['outside_numeric_target'],1)
        self.assertEqual(counts.get('within_numeric_target',0),0)
        actual=report['dynamic_merge_qualified_branch_actors']['merge_straight']
        self.assertEqual(actual['policy_cancel_transitions'],1)
        self.assertEqual(actual['observed_return_segments'],0)
        self.assertFalse(report['outcome_selection'])

    def test_pending_stages_are_separate_from_terminal_failures(self):
        base={'template':'p','proposed':{'vehicle_count':2,'phase':'crossing','weather':'daylight'}}
        rows=[dict(base,status='compiled',compiled=True),
              dict(base,status='physics_checked',compiled=True,physics_qualified=True),
              dict(base,status='failed',compiled=True,physics_qualified=True),
              dict(base,status='started')]
        bins=conditional_counts(rows)['bins']
        for group in bins:
            self.assertEqual(group['compiled_awaiting_physics'],1)
            self.assertEqual(group['physics_awaiting_visual'],1)
            self.assertEqual(group['started_or_interrupted'],1)
            self.assertEqual(group['failed'],1)
            self.assertEqual(group['physics_qualified'],2)

    def test_physics_success_survives_later_visual_failure(self):
        from test_orchestrator import Fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'attempts/s/candidate_00/attempt_00';scene='scene_0000'
            initial=path/'initial_plan'/scene;initial.mkdir(parents=True)
            (initial/'spec.json').write_text(json.dumps(Fixture.scene_spec(scene)))
            plan={'jobs':[{'scene_id':scene,'cell_id':'c'}],'cells':[{'cell_id':'c','template':'follower_response_straight'}]}
            state={'slots':{scene:{'attempts':[{'path':str(path.relative_to(root)),'candidate_index':0,'attempt_index':0,
                'status':'failed','reason':'rgb consistency','completed_stages':{'physics_checked':True}}]}}}
            result=build_report(root,plan,state)
            self.assertEqual(result['stages']['compiled']['count'],1)
            self.assertEqual(result['stages']['physics_qualified']['count'],1)
            self.assertEqual(result['stages']['accepted']['count'],0)

    def test_rejected_proposals_and_unattempted_slots_remain_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cell={'cell_id':'ped','template':'pedestrian_straight'}
            jobs=[{'scene_id':'scene_0000','cell_id':'ped','seed':1,'interaction_phase':'curb_wait','weather_id':'daylight'},
                  {'scene_id':'scene_0001','cell_id':'ped','seed':2,'interaction_phase':'curb_wait','weather_id':'night'}]
            state={'slots':{'scene_0000':{'attempts':[]},'scene_0001':{'attempts':[]}}}
            for i in range(2):
                rel=f'attempts/scene_0000/candidate_00/attempt_{i:02d}';path=root/rel;path.mkdir(parents=True)
                selected={'layout_seed':i,'weather_id':'daylight'}
                (path/'sampling_proposal.json').write_text(json.dumps({'selection':selected,'intent':intent(jobs[0],cell,selected)}))
                state['slots']['scene_0000']['attempts'].append({'path':rel,'candidate_index':0,'attempt_index':i,'status':'failed','reason':'mapped approach too short'})
            report=build_report(root,{'jobs':jobs,'cells':[cell]},state)
            self.assertEqual(report['stages']['planned']['count'],2)
            self.assertEqual(report['stages']['proposed']['count'],2)
            self.assertEqual(report['stages']['rejected_proposals']['count'],2)
            self.assertEqual(report['stages']['accepted']['count'],0)
            self.assertEqual(report['rejections'][0]['count'],2)
            self.assertFalse(report['admission_gate'])

    def test_constant_features_not_reported_as_nan_correlations(self):
        rows=[{'numeric':{'x':2.,'y':float(i),'z':3.*i}} for i in range(10)]
        result=distribution(rows)
        self.assertEqual(len(result['pearson_correlations']),1)
        self.assertAlmostEqual(result['pearson_correlations'][0]['r'],1.)
        json.dumps(result,allow_nan=False)

    def test_similarity_uses_the_whole_window_and_keeps_equal_outcomes(self):
        base={'scene_id':'a','template':'p','roles':['ego','target'],'branches':['maintain'],
              'positions_ego_t0_frame_m':np.zeros((1,9,2,2)).tolist()}
        b=copy.deepcopy(base);b['scene_id']='b'
        c=copy.deepcopy(base);c['scene_id']='c';c['positions_ego_t0_frame_m'][0][3][1][0]=4.
        other=copy.deepcopy(base);other['roles']=['ego','other']
        result=trajectory_similarity([base,b,c,other])
        self.assertEqual(result['comparable_scene_pairs'],3)
        self.assertEqual(result['nearest_pairs'][0]['position_rms_m'],0.)
        self.assertGreater(result['nearest_pairs'][-1]['position_rms_m'],0.)
        self.assertFalse(result['admission_gate'])


if __name__=='__main__':unittest.main(verbosity=2)
