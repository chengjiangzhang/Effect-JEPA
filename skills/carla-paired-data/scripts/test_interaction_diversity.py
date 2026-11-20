"""Behavior, causal information, version compatibility and coverage regressions."""
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

from interaction_design import PHASES,PEDESTRIAN_PACES,assign_phases,phase_for,road_encounter,action_timing,pedestrian_pace_for,freeze_job_strata
from action_design import distinct_profiles_for_template
from plan_grid import build_plan,validate_job_plan
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
import pedestrian_crossing as c
import pedestrian_policy as p
from test_pedestrian_crossing import config,step,people
from test_pedestrian_policy import vehicles
from auxiliary_actors import validate_auxiliary_config
from interaction_coverage import segments,stop_context,reported_npc_rule_triggers
from controllers_v2 import narrow_meeting_target
from test_auxiliary import configuration,observations
from audit_auxiliary import replay_warmup


def new_config():
    cfg=config()
    cfg['pedestrians'][0]['response_policy'].update(kind=c.KIND_V3,crossing_feedback=dict(c.DEFAULTS_V3))
    return cfg


class DiversityTests(unittest.TestCase):
    def test_all_declared_stop_actions_have_service_braking_at_zero_target(self):
        conflict={'stop_route_s':30.,'id':'conflict','participants':[{'role':'target'}]}
        stop_templates=set()
        for template,phases in PHASES.items():
            for phase in phases:
                profiles=distinct_profiles_for_template(template,['left','right'],conflict)
                cfg={'action':{'profiles':profiles}}
                action_timing(cfg,template,phase)
                for name,profile in profiles.items():
                    if profile['kind'] in ('stop_hold','yield_wait'):
                        if profile['kind']=='stop_hold':stop_templates.add(template)
                        self.assertEqual(profile['terminal_brake_min'],1.,(template,phase,name))
                    else:
                        self.assertNotIn('terminal_brake_min',profile,(template,phase,name))
        self.assertEqual(stop_templates,set(PHASES)-{
            'follower_response_straight','ego_lane_change_straight','merge_straight','merge_ramp'})

    def test_reported_triggers_exclude_relations_ego_context_and_unapplied_endpoint(self):
        roles=['ego','target','rear']
        rows=[{'role':'ego','signal':{'must_stop':True,'state':'Red'}},
              {'role':'rear','leader_index':1,'reason':'leader'},
              {'role':'target','leader_index':0,'reason':'brake_for_observed_conflicting_body',
               'signal':{'must_stop':True,'state':'Red'},
               'live_collision_constraint':{'triggered':True,'participants':[{'role':'ego'},{'role':'ego'}]},
               'priority_zone':{'yielding':False,'trigger_roles':['rear']}}]
        result=reported_npc_rule_triggers([rows,rows,rows],roles,[0.,.05,4.])
        self.assertEqual(len(result),2)
        self.assertEqual({r['reported_rule'] for r in result},{'stop_for_signal','live_collision_constraint'})
        for row in result:
            self.assertEqual(row['actor'],'target')
            self.assertEqual(row['decision_samples'],2)
            self.assertEqual(row['last_decision_time_s'],.05)
        rows[2]['live_collision_constraint']['triggered']=False
        rows[2]['signal']['must_stop']=False
        self.assertEqual(reported_npc_rule_triggers([rows],roles,[0.]),[])
        rows[2]['narrow_meeting']={'yielding':True,'trigger_roles':[]}
        confirmation=reported_npc_rule_triggers([rows],roles,[.1])
        self.assertEqual(len(confirmation),1)
        self.assertEqual(confirmation[0]['reported_rule'],'narrow_meeting')
        self.assertIsNone(confirmation[0]['related_actor_or_signal'])

    def test_measured_stillness_is_not_automatically_a_reactive_stop(self):
        times=np.arange(6)*.05
        def classify(command,station,mode):
            return stop_context(segments(np.ones(6,dtype=bool),times),times,np.full(6,command),
                                np.full(6,station),10.6,[mode]*6)[0]['classification']
        self.assertEqual(classify(1.35,9.28,'continue'),'stationary_with_move_command')
        self.assertEqual(classify(0.,9.28,'stop'),'policy_stop_with_measured_stillness')
        self.assertEqual(classify(0.,10.6,'continue'),'near_route_end_with_zero_command')
        self.assertEqual(classify(0.,9.28,'retreat'),'zero_command_other_mode')
        event=stop_context(segments(np.ones(6,dtype=bool),times),times,np.zeros(6),np.full(6,10.6),
                           10.6,['continue']*6,endpoint_distance=np.full(6,3.))[0]
        self.assertFalse(event['near_route_end'])
        mixed=stop_context(segments(np.ones(6,dtype=bool),times),times,np.zeros(6),np.full(6,9.28),
                           10.6,['continue']*5+['stop'])[0]
        self.assertEqual(mixed['classification'],'zero_command_other_mode')

    def test_every_template_has_four_prospective_strata(self):
        grid=json.loads((Path(__file__).parents[1]/'assets/grid.example.json').read_text())
        plan=build_plan(grid,144,43)
        self.assertEqual(len(PHASES),12)
        lookup={x['cell_id']:x['template'] for x in plan['cells']}
        for template,phases in PHASES.items():
            actual=[j['interaction_phase'] for j in plan['jobs'] if lookup[j['cell_id']]==template]
            self.assertEqual(set(actual),set(phases))
            self.assertLessEqual(max(actual.count(x) for x in phases)-min(actual.count(x) for x in phases),1)
        self.assertEqual(plan,validate_job_plan(plan,grid,144,43))

    def test_phase_survives_candidate_repairs_and_rejects_unknown_label(self):
        job={'seed':5,'interaction_phase':'crossing'}
        self.assertEqual(phase_for(job,'pedestrian_straight'),'crossing')
        with self.assertRaises(ValueError):phase_for(job,'narrow_meeting')

    def test_pedestrian_stages_cross_paces_before_capture_without_response_targets(self):
        grid=json.loads((Path(__file__).parents[1]/'assets/grid.example.json').read_text())
        grid['pairs']=[x for x in grid['pairs'] if x['id'].startswith('pedestrian_')]
        plan=build_plan(grid,24,317)
        lookup={c['cell_id']:c['template'] for c in plan['cells']}
        for template in ('pedestrian_straight','pedestrian_unsignalized'):
            jobs=[j for j in plan['jobs'] if lookup[j['cell_id']]==template]
            for phase in ('entering','crossing'):
                values=[j['pedestrian_pace'] for j in jobs if j['interaction_phase']==phase]
                self.assertCountEqual(values,PEDESTRIAN_PACES)
            for job in jobs:
                if job['interaction_phase'] in ('curb_wait','clearing'):
                    self.assertNotIn('pedestrian_pace',job)
        self.assertEqual(plan,validate_job_plan(plan,grid,24,317))

    def test_pedestrian_pace_validation_preserves_old_jobs_and_rejects_invalid_scope(self):
        job={'seed':5,'interaction_phase':'entering','pedestrian_pace':'unhurried'}
        self.assertEqual(pedestrian_pace_for(job,'pedestrian_straight','entering'),'unhurried')
        for template,phase in [('narrow_meeting','entering'),('pedestrian_straight','curb_wait')]:
            with self.assertRaises(ValueError):pedestrian_pace_for(job,template,phase)
        job['pedestrian_pace']='must_retreat'
        with self.assertRaises(ValueError):pedestrian_pace_for(job,'pedestrian_straight','entering')
        job.pop('pedestrian_pace')
        self.assertIn(pedestrian_pace_for(job,'pedestrian_straight','entering'),PEDESTRIAN_PACES)
        for value in (None,[],{},False,1):
            job['pedestrian_pace']=value
            with self.assertRaises(ValueError):pedestrian_pace_for(job,'pedestrian_straight','entering')

    def test_legacy_pace_and_phase_are_frozen_before_candidate_reseeding(self):
        job={'seed':5,'interaction_phase':'entering'}
        frozen=freeze_job_strata(job,'pedestrian_straight')
        self.assertEqual(frozen['pedestrian_pace'],'brisk')
        for seed in (0,1,2,13):
            candidate={**frozen,'seed':seed}
            self.assertEqual(pedestrian_pace_for(candidate,'pedestrian_straight','entering'),'brisk')
        self.assertNotIn('pedestrian_pace',job)
        self.assertEqual(freeze_job_strata({'seed':5},'pedestrian_straight')['interaction_phase'],'entering')

    def test_history_crossing_is_versioned_and_bounded(self):
        cfg=new_config();cfg['pedestrians'][0]['start_time_s']=-1.5
        validate_auxiliary_config(cfg)
        cfg['pedestrians'][0]['start_time_s']=-3.1
        with self.assertRaises(ValueError):validate_auxiliary_config(cfg)
        old=config();old['pedestrians'][0]['start_time_s']=-1.
        with self.assertRaises(ValueError):validate_auxiliary_config(old)

    def test_observed_queue_braking_requires_real_stopped_leader_and_margin(self):
        cfg=new_config();cfg['vehicles'].append(copy.deepcopy(cfg['vehicles'][0]))
        cfg['vehicles'][1]['role']='follower';cfg['vehicle_roles'].append('follower')
        obs=vehicles(cfg,x=0.,speed=4.,acceleration=-2.)
        obs['position'][0,0]=20.;obs['transform_world_vehicle'][0,0,3]=20.;obs['velocity'][0]=0.
        self.assertEqual(c.observed_leader_braking(obs,cfg,1),2.)
        obs['velocity'][0,0]=1.
        self.assertEqual(c.observed_leader_braking(obs,cfg,1),0.)
        obs['velocity'][0]=0.;obs['acceleration'][1]=0.
        self.assertEqual(c.observed_leader_braking(obs,cfg,1),0.)
        obs['acceleration'][1,0]=-2.;cfg['vehicles'][1]['route_maneuver']={'kind':'merge'}
        self.assertEqual(c.observed_leader_braking(obs,cfg,1),0.)

    def test_hidden_state_is_reconstructed_from_complete_common_warmup(self):
        cfg=new_config();cfg['pedestrians'][0]['start_time_s']=-2.5
        observed=vehicles(cfg,x=-100.,speed=0.);person=people(cfg,y=-5.,speed=0.)
        person['observed_control']=np.zeros((1,5));state=[False];rows=[]
        for i in range(20):
            time=-3.+i*.05
            control,decisions,state=p.controls_and_decisions(cfg,observed,person,time,.05,state)
            rows.append({'time_s':time,'frame_id':60+i,'simulation_time_s':3.+i*.05,
                'vehicles':{k:v.tolist() for k,v in observed.items()},
                'pedestrians':{k:v.tolist() for k,v in person.items()},
                'control':control.tolist(),'decisions':decisions})
            person['velocity']=control[:,:3]*control[:,3:4]
            person['position']+=person['velocity']*.05
            person['transform_world_actor'][:,:3,3]=person['position']
            person['observed_control']=control.copy()
        walkers={'observed_control':person['observed_control'][None]}
        states={'frame_id':np.array([80]),'simulation_time_s':np.array([4.])}
        record={'warmup':rows}
        errors,actual=replay_warmup(record,walkers,cfg,states)
        self.assertFalse(errors);self.assertEqual(actual,state);self.assertTrue(actual[0]['released'])
        missing=copy.deepcopy(record);missing['warmup'].pop()
        self.assertTrue(replay_warmup(missing,walkers,cfg,states)[0])
        wrong=copy.deepcopy(record);wrong['warmup'][-1]['decisions'][0]['state_after']['mode']='stop'
        self.assertTrue(replay_warmup(wrong,walkers,cfg,states)[0])

    def test_fast_is_a_risk_response_with_bounded_acceleration(self):
        cfg=new_config()
        control,row,_=step(cfg,y=-2.,car_x=0.,car_speed=8.)
        self.assertEqual(row['mode'],'fast')
        self.assertTrue(row['alternatives']['fast']['caution_clear'])
        self.assertFalse(row['alternatives']['continue']['caution_clear'])
        self.assertLessEqual(control[3],1.4+c.DEFAULTS_V3['acceleration_m_s2']*.05+1e-8)
        _,clear,_=step(cfg,y=-2.,car_x=0.,car_speed=0.)
        self.assertEqual(clear['mode'],'continue')

    def test_branch_labels_do_not_change_pedestrian_decision(self):
        cfg=new_config();a=step(cfg,y=-2.,car_x=0.)
        other=copy.deepcopy(cfg);other['branch']='stop_hold';other['action']={'future':[999], 'name':'accelerate'}
        b=step(other,y=-2.,car_x=0.)
        np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1:],b[1:])

    def test_earliest_start_is_respected_even_with_free_crossing(self):
        cfg=new_config();cfg['pedestrians'][0]['start_time_s']=-1.
        _,row,_=step(cfg,y=-5.,speed=0.,state=False,time=-1.5,car_x=-100.,car_speed=0.)
        self.assertEqual(row['mode'],'wait')
        _,row,_=step(cfg,y=-5.,speed=0.,state=False,time=-.5,car_x=-100.,car_speed=0.)
        self.assertNotEqual(row['mode'],'wait')
        self.assertIn('entry_sweep',row)

    def test_temporal_sampling_does_not_claim_visual_visibility(self):
        t=np.arange(81)*.05
        short=segments((t>=.1)&(t<=.4),t)
        self.assertEqual(len(short),1);self.assertEqual(short[0]['rgb_sample_times'],[])
        self.assertEqual(segments((t>=.1)&(t<=.2),t),[])

    def test_narrow_passes_when_bodies_fit_and_yields_for_real_occupancy(self):
        cfg=configuration();cfg.pop('pedestrians');cfg.pop('traffic_signal_program')
        cfg['vehicles'][1]['route']=[[100.,3.6,0.,180.],[0.,3.6,0.,180.]]
        cfg['geometry']['narrow_meeting']={'yield_roles':['npc'],'priority_roles':['ego'],
            'yield_route_s':{'npc':[10.,30.]},'priority_route_s':{'ego':[10.,30.]},
            'stop_margin_m':2.,'clearance_time_s':3.,'policy_version':'narrow-current-occupancy-v2',
            'lateral_margin_m':.35,'release_confirmation_s':.4,'passing_speed_m_s':3.}
        obs=observations();obs['position'][0]=[15.,0.,0.];obs['position'][1]=[95.,3.6,0.]
        obs.update(bbox_offset=np.zeros((2,3)),bbox_rotation_deg=np.zeros((2,3)))
        obs['rotation_deg'][1,2]=180.
        obs['velocity'][1]=[-3.,0.,0.];ext=np.array([[2.,1.,1.],[2.,1.,1.]])
        _,row=narrow_meeting_target(1,obs,cfg,ext,5.,{},0.)
        self.assertFalse(row['yielding']);self.assertTrue(row['simultaneous_body_clearance'])
        cfg['vehicles'][1]['route']=[[100.,1.8,0.,180.],[0.,1.8,0.,180.]];obs['position'][1,1]=1.8
        memory={};_,row=narrow_meeting_target(1,obs,cfg,ext,5.,memory,0.)
        self.assertTrue(row['yielding'])
        obs['position'][0,0]=40.
        _,row=narrow_meeting_target(1,obs,cfg,ext,5.,memory,.1)
        self.assertTrue(row['yielding'])
        _,row=narrow_meeting_target(1,obs,cfg,ext,5.,memory,.55)
        self.assertFalse(row['yielding'])

    def test_release_confirmation_applies_when_lateral_clearance_opens(self):
        cfg=configuration();cfg.pop('pedestrians');cfg.pop('traffic_signal_program')
        cfg['vehicles'][1]['route']=[[100.,3.,0.,180.],[0.,3.,0.,180.]]
        cfg['geometry']['narrow_meeting']={'yield_roles':['npc'],'priority_roles':['ego'],
            'yield_route_s':{'npc':[10.,30.]},'priority_route_s':{'ego':[10.,30.]},
            'stop_margin_m':2.,'clearance_time_s':3.,'policy_version':'narrow-current-occupancy-v2',
            'lateral_margin_m':.35,'release_confirmation_s':.4,'passing_speed_m_s':3.}
        obs=observations();obs.update(bbox_offset=np.zeros((2,3)),bbox_rotation_deg=np.zeros((2,3)))
        obs['position'][:]=[[15.,0.,0.],[95.,1.8,0.]];obs['rotation_deg'][1,2]=180.
        ext=np.array([[2.4,.9,.7],[2.4,.9,.7]]);memory={}
        self.assertTrue(narrow_meeting_target(1,obs,cfg,ext,5.,memory,0.)[1]['yielding'])
        obs['position'][1,1]=3.
        self.assertTrue(narrow_meeting_target(1,obs,cfg,ext,5.,memory,.05)[1]['yielding'])
        self.assertFalse(narrow_meeting_target(1,obs,cfg,ext,5.,memory,.45)[1]['yielding'])
        self.assertFalse(memory['npc']['was_yielding'])
        obs['rotation_deg'][1,2]=90.
        self.assertFalse(narrow_meeting_target(1,obs,cfg,ext,5.,{},.5)[1]['simultaneous_body_clearance'])
        obs['rotation_deg'][1,2]=180.;obs['bbox_offset'][1,1]=1.
        self.assertFalse(narrow_meeting_target(1,obs,cfg,ext,5.,{},.5)[1]['simultaneous_body_clearance'])

    def test_measured_retreat_overshoot_brakes_without_position_snap(self):
        spec=new_config()['pedestrians'][0]
        _,points,control,speed=c.motion_profile(np.array([0.,.98,0.]),[0.,-.2,0.],
            [[0.,0.,0.],[0.,10.,0.]],.98,[0.,1.],'retreat',1.,spec,.05)
        self.assertAlmostEqual(speed,-.05)
        self.assertAlmostEqual(control[3],.05)
        self.assertLess(points[1,1],points[0,1])
        self.assertGreaterEqual(np.diff(points[:,1]).min(),-.0025-1e-10)
        cfg=new_config();state={**c.initial_state(),'released':True,'mode':'retreat',
            'retreat_origin_s_m':4.7,'retreat_target_s_m':3.5,'last_command_speed_m_s':-.2}
        command,row,_=step(cfg,y=-1.52,speed=-.2,car_x=23.,state=state)
        self.assertEqual(row['mode'],'retreat')
        self.assertAlmostEqual(row['signed_command_speed_m_s'],-.05)
        self.assertLess(command[1],0.)
        for station,mode,target,sign in ((9.999,'continue',0.,1.),(.001,'retreat',0.,-1.)):
            _,points,_,_=c.motion_profile(np.array([0.,station,0.]),[0.,0.,0.],
                [[0.,0.,0.],[0.,10.,0.]],station,[0.,1.],mode,target,spec,.05)
            self.assertGreaterEqual(points[:,1].min(),-1e-10)
            self.assertLessEqual(points[:,1].max(),10.+1e-10)

    def test_lane_gap_places_front_and_rear_and_preserves_real_routes(self):
        import scenarios_v2 as geometry
        def fixture():
            main=[[0.,0.,0.,0.],[400.,0.,0.,0.]];side=[[0.,3.5,0.,0.],[400.,3.5,0.,0.]]
            cars=[geometry.vehicle('ego',main,150.,5.),geometry.vehicle('right_rear',side,132.,5.),
                  geometry.vehicle('parallel_vehicle',side,151.,5.),geometry.vehicle('traffic',side,169.,5.)]
            return {'vehicles':cars,'geometry':{'traffic_design':{'roles_and_slots':[
                {'role':a['role'],'route_lane':'main' if i==0 else 'right'} for i,a in enumerate(cars)]}}}
        wide=fixture();tight=fixture();routes=[copy.deepcopy(a['route']) for a in wide['vehicles']]
        road_encounter(wide,'ego_lane_change_straight',0,np.random.default_rng(3),geometry)
        road_encounter(tight,'ego_lane_change_straight',2,np.random.default_rng(3),geometry)
        self.assertGreater(wide['vehicles'][2]['spawn_route_s']-150.,23.)
        self.assertLess(tight['vehicles'][2]['spawn_route_s']-150.,15.)
        self.assertGreater(150.-wide['vehicles'][1]['spawn_route_s'],24.)
        self.assertEqual(routes,[a['route'] for a in wide['vehicles']])
        for a,b in zip(sorted(wide['vehicles'][1:],key=lambda a:a['spawn_route_s']),
                       sorted(wide['vehicles'][1:],key=lambda a:a['spawn_route_s'])[1:]):
            self.assertGreater(b['spawn_route_s']-a['spawn_route_s'],20.)


if __name__=='__main__':unittest.main(verbosity=2)
