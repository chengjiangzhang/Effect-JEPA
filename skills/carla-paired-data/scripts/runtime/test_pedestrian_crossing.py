"""Behavioral and evidence regressions for current-state mid-crossing feedback."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

if __package__:
    from . import pedestrian_policy as p, pedestrian_crossing as c, audit_auxiliary as audit
    from .test_pedestrian_policy import fixture, vehicles, person
    from .test_auxiliary import FakeWorld, FakeClient
    from .auxiliary_actors import AuxiliaryActors, validate_auxiliary_config
    from .audit_capture_equivalence import source_identity, SOURCES
else:
    import pedestrian_policy as p, pedestrian_crossing as c, audit_auxiliary as audit
    from test_pedestrian_policy import fixture, vehicles, person
    from test_auxiliary import FakeWorld, FakeClient
    from auxiliary_actors import AuxiliaryActors, validate_auxiliary_config
    from audit_capture_equivalence import source_identity, SOURCES


def config():
    cfg = fixture()
    cfg['pedestrians'][0]['response_policy'].update(kind=c.KIND, crossing_feedback=dict(c.DEFAULTS))
    return cfg


def people(cfg, y=-3., speed=1.4):
    observed = person(cfg)
    observed['position'][0, 1] = y
    observed['transform_world_actor'][0, :3, 3] = observed['position'][0]
    observed['velocity'] = np.array([[0., speed, 0.]])
    return observed


def released():
    return {**c.initial_state(), 'released': True, 'mode': 'continue', 'last_command_speed_m_s':1.4}


def step(cfg=None, *, y=-3., speed=1.4, car_x=0., car_speed=8., state=None, time=1., observed=None):
    cfg = cfg or config()
    if state is None:
        state={**released(),'last_command_speed_m_s':speed}
    control, rows, states = p.controls_and_decisions(cfg,
        vehicles(cfg, x=car_x, speed=car_speed) if observed is None else observed,
        people(cfg, y, speed), time, .05, [state])
    return control[0], rows[0], states[0]


class CrossingBehaviorTests(unittest.TestCase):
    def test_waiting_and_legacy_commitment_remain_versioned(self):
        cfg = config(); validate_auxiliary_config(cfg)
        control, row, _ = step(cfg, y=-5, speed=0, state=False, car_x=10)
        self.assertEqual(row['mode'], 'wait'); self.assertEqual(control[3], 0)
        old = fixture()
        control, rows, _ = p.controls_and_decisions(old, vehicles(old), people(old), 1., .05, [True])
        self.assertEqual(rows[0]['reason'], 'continue_committed_crossing')
        self.assertAlmostEqual(control[0,3], 1.4)

    def test_same_pedestrian_different_measured_ego_changes_response(self):
        _, safe, _ = step(car_speed=0)
        command, danger, _ = step(car_speed=8)
        self.assertEqual(safe['mode'], 'continue')
        self.assertEqual(danger['mode'], 'stop')
        self.assertLess(command[3], 1.4)
        self.assertFalse(danger['alternatives']['continue']['body_clear'])
        self.assertTrue(danger['alternatives']['stop']['body_clear'])

    def test_caution_uses_slower_progress_when_it_opens_clearance(self):
        command, row, _ = step(y=-4.)
        self.assertEqual(row['mode'], 'slow')
        self.assertGreater(command[3], 0); self.assertLess(command[3], 1.4)

    def test_retreat_is_selected_when_stopping_leaves_body_in_vehicle_sweep(self):
        command, row, state = step(y=-1.5)
        self.assertEqual(row['mode'], 'retreat')
        self.assertFalse(row['alternatives']['stop']['body_clear'])
        self.assertTrue(row['alternatives']['retreat']['body_clear'])
        # No instantaneous reverse: the first command brakes existing forward motion.
        self.assertGreater(command[1], 0)
        self.assertAlmostEqual(command[3], 1.25)
        self.assertAlmostEqual(state['retreat_origin_s_m']-state['retreat_target_s_m'], 1.2)

    def test_safe_forward_clearance_is_preferred_to_retreat(self):
        _, row, _ = step(y=1.5, car_x=8, car_speed=8)
        self.assertEqual(row['mode'], 'continue')
        self.assertFalse(row['unavoidable_predicted_conflict'])

    def test_retreat_is_rechecked_against_unlisted_rear_vehicle(self):
        cfg=config(); cfg['vehicle_roles'].append('rear_vehicle')
        cfg['vehicles'].append({'role':'rear_vehicle','route':[[0.,-3.,0.],[100.,-3.,0.]]})
        observed=vehicles(cfg,x=0,speed=8)
        observed['position'][1]=[20.,-3.,0.]
        observed['transform_world_vehicle'][1,:3,3]=observed['position'][1]
        observed['velocity'][1]=0
        _, row, _=step(cfg,y=-1.5,observed=observed)
        self.assertFalse(row['alternatives']['retreat']['body_clear'])
        self.assertTrue(row['unavoidable_predicted_conflict'])
        self.assertIn('rear_vehicle',[r['role'] for r in row['alternatives']['retreat']['actor_checks']])

    def test_other_pedestrian_is_an_obstacle(self):
        cfg=config();cfg['pedestrians'].append(copy.deepcopy(cfg['pedestrians'][0]))
        cfg['pedestrians'][1]['role']='other_pedestrian'
        obs=people(cfg,y=-1.5)
        for k,v in list(obs.items()):obs[k]=np.concatenate([v,v.copy()],axis=0)
        obs['position'][1,1]=-2.7;obs['transform_world_actor'][1,:3,3]=obs['position'][1]
        obs['velocity'][1]=0
        _,rows,_=p.controls_and_decisions(cfg,vehicles(cfg,x=0,speed=8),obs,1.,.05,[released(),released()])
        self.assertFalse(rows[0]['alternatives']['retreat']['body_clear'])

    def test_resume_requires_continuous_clearance_and_interrupt_resets_timer(self):
        _,_,state=step()
        for t in (1.05,1.2):
            _,row,state=step(car_speed=0,state=state,time=t,speed=0)
            self.assertEqual(row['mode'],'stop')
        _,_,state=step(state=state,time=1.3,speed=0)
        self.assertIsNone(state['clear_since_s'])
        _,row,state=step(car_speed=0,state=state,time=1.35,speed=0)
        self.assertEqual(row['mode'],'stop')
        _,row,state=step(car_speed=0,state=state,time=1.75,speed=0)
        self.assertEqual(row['mode'],'continue')

    def test_hysteresis_never_holds_an_unsafe_stop(self):
        state={**released(),'mode':'stop'}
        _,row,_=step(y=1.5,car_x=8,car_speed=8,state=state,speed=0)
        self.assertFalse(row['alternatives']['stop']['body_clear'])
        self.assertNotEqual(row['reason'],'confirm_clear_before_resuming')
        self.assertEqual(row['mode'],'continue')

    def test_stop_to_slow_also_requires_clearance_confirmation(self):
        state={**released(),'mode':'stop','last_command_speed_m_s':0.}
        _,row,state=step(y=-4.,car_speed=8.,state=state,speed=0.,time=1.)
        self.assertEqual(row['mode'],'stop')
        self.assertEqual(row['reason'],'confirm_clear_before_resuming')
        _,row,state=step(y=-4.,car_speed=8.,state=state,speed=0.,time=1.4)
        self.assertEqual(row['mode'],'slow')

    def test_no_label_target_action_or_future_trajectory_leaks_into_feedback(self):
        cfg=config();observed=vehicles(cfg,x=0,speed=8)
        original=step(cfg,observed=observed)
        cfg['action']={'branch':'yield_wait','future_ego_positions':[[999,999,0]]}
        observed.update(target_speed=np.array([0.]),control=np.zeros((1,3)),branch='stop_hold')
        changed=step(cfg,observed=observed)
        np.testing.assert_array_equal(original[0],changed[0]); self.assertEqual(original[1:],changed[1:])

    def test_inputs_and_prior_state_are_not_mutated(self):
        state=released();saved=copy.deepcopy(state);cfg=config();old=copy.deepcopy(cfg)
        step(cfg,state=state)
        self.assertEqual(state,saved);self.assertEqual(cfg,old)

    def test_retreat_never_goes_behind_start_or_ratcheted_target(self):
        cfg=config();spec=cfg['pedestrians'][0]
        times,points,_,_=c.motion_profile(np.array([20.,-4.8,0.]),[0.,0.,0.],
            spec['route_points'],.2,[0.,1.],'retreat',0.,spec,.05)
        self.assertGreaterEqual(points[:,1].min(),-5.)
        _,_,state=step(y=-1.5)
        target=state['retreat_target_s_m'];state['mode']='stop'
        _,row,next_state=step(y=-1.5,speed=0,state=state)
        self.assertEqual(next_state['retreat_target_s_m'],target)

    def test_confirmed_forward_progress_ends_the_previous_retreat_episode(self):
        state={**released(),'mode':'stop','retreat_origin_s_m':3.,'retreat_target_s_m':1.8,
               'clear_since_s':.5,'last_command_speed_m_s':0.}
        _,row,next_state=step(y=-4.,car_speed=8.,state=state,speed=0.,time=1.)
        self.assertEqual(row['mode'],'slow')
        self.assertIsNone(next_state['retreat_target_s_m'])
        self.assertIsNone(next_state['retreat_origin_s_m'])

    def test_actuator_lag_does_not_reset_outgoing_speed_ramp_every_tick(self):
        state={**released(),'last_command_speed_m_s':0.}
        commands=[]
        for i in range(10):
            command,row,state=step(state=state,speed=0.,car_speed=0.,time=1.+i*.05)
            commands.append(command[3])
        self.assertGreater(commands[-1],.7)
        np.testing.assert_allclose(np.diff(commands),.075,atol=1e-9)

    def test_risk_encloses_intermediate_motion_not_only_two_clear_endpoints(self):
        times=np.array([0.,.05]);a=np.array([[-3.,0.,0.],[-3.,0.,0.]])
        b=np.array([[3.,0.,0.],[3.,0.,0.]])
        obstacle=[('car',np.zeros(3),np.eye(3),np.ones(3),np.zeros(3))]
        self.assertTrue(c.risk(a,times,obstacle,np.ones(3)*.2,c.DEFAULTS)['body_clear'])
        self.assertTrue(c.risk(b,times,obstacle,np.ones(3)*.2,c.DEFAULTS)['body_clear'])
        self.assertFalse(c.risk(a,times,obstacle,np.ones(3)*.2,c.DEFAULTS,b)['body_clear'])

    def test_continuous_sweep_catches_between_frame_crossing(self):
        first,duration=c.swept_interval(np.array([[-10.,0,0],[10.,0,0]]),np.ones(3),np.array([0.,.05]))
        self.assertAlmostEqual(first,.0225);self.assertAlmostEqual(duration,.005)
        self.assertEqual(c.swept_interval(np.array([[-10.,0,5],[10.,0,5]]),np.ones(3),np.array([0.,.05])),(None,0.))

    def test_unsafe_alternatives_do_not_abandon_escape_for_microsecond_tie(self):
        options = {'slow': {'first_body_risk_s': .917717, 'body_risk_duration_s': .588},
                   'retreat': {'first_body_risk_s': .917712, 'body_risk_duration_s': .005}}
        self.assertEqual(c.least_risk_mode(options, 'retreat', .05), 'retreat')
        options['slow']['body_risk_duration_s'] = .005
        self.assertEqual(c.least_risk_mode(options, 'retreat', .05), 'retreat')
        # A full-tick improvement is meaningful and must still change the response.
        options['slow']['first_body_risk_s'] = .97
        self.assertEqual(c.least_risk_mode(options, 'retreat', .05), 'slow')

    def test_no_false_feedback_from_vehicle_on_overpass(self):
        cfg=config();obs=vehicles(cfg,x=0,speed=8)
        obs['position'][:,2]=10;obs['transform_world_vehicle'][:,2,3]=10
        _,row,_=step(cfg,observed=obs)
        self.assertEqual(row['mode'],'continue')

    def test_invalid_new_contract_and_missing_source_identity_rejected(self):
        for field,value in [('max_retreat_m',-1),('lookahead_s',float('nan')),('slow_speed_ratio',1),('body_margin_m',2)]:
            cfg=config();cfg['pedestrians'][0]['response_policy']['crossing_feedback'][field]=value
            with self.assertRaises(ValueError):validate_auxiliary_config(cfg)
        cfg=config();cfg['code_sha256']={k:'a'*64 for k in SOURCES}
        self.assertIsNotNone(source_identity(cfg));cfg['code_sha256'].pop('pedestrian_crossing.py')
        self.assertIsNone(source_identity(cfg))


def recorded_trace(folder):
    cfg=config();world=FakeWorld(lights=False);aux=AuxiliaryActors(world,FakeClient(world),cfg)
    recorded=[]
    for i in range(121):
        t=-2+i*.05
        # An initially stopped vehicle accelerates after the walker has departed.
        obs=vehicles(cfg,x=0 if t<2 else 8*(t-2),speed=0 if t<2 else 8)
        recorded.append({k:v.copy() for k,v in obs.items()})
        aux.inject_observation(obs,t);command=aux.record(t,final=i==120)
        if i<120:aux.advance_commands(command,t+.05);world.tick()
    aux.save(folder)
    states={k:np.asarray([s[k] for s in recorded]) for k in recorded[0] if k not in audit.WALKER_BOXES}
    states.update({k:recorded[0][k] for k in audit.WALKER_BOXES})
    states.update({k:np.asarray([s[k] for s in aux.samples]) for k in ('time_s','frame_id','simulation_time_s')})
    meta={**aux.metadata(),'actor_ids':[1],'collisions':[],
          'collision_sensor_roles':cfg['vehicle_roles']+[s['role'] for s in cfg['pedestrians']]}
    return cfg,states,meta


class CrossingEvidenceTests(unittest.TestCase):
    def test_full_trace_is_causally_replayable_and_records_actual_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,states,meta=recorded_trace(root)
            checked=audit.inspect_auxiliary(root,cfg,meta,states)
            self.assertEqual(checked['status'],'passed',checked['errors'])
            record=json.loads((root/'pedestrian_decisions.json').read_text())
            self.assertEqual(record['schema'],'pedestrian-decisions-v2')
            modes=[s['decisions'][0]['mode'] for s in record['samples']]
            self.assertIn('continue',modes);self.assertIn('stop',modes)
            summary=record['summary']['crossing_pedestrian']
            self.assertGreater(summary['observed_stop_after_motion_time_s'],summary['observed_first_moving_time_s'])
            self.assertEqual(record['samples'][-1]['decisions'][0]['mode'],'final')

    def test_response_state_and_measured_summary_tampering_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,states,meta=recorded_trace(root)
            path=root/'pedestrian_decisions.json';original=path.read_text()
            record=json.loads(original);record['samples'][80]['decisions'][0]['state_after']['mode']='retreat'
            path.write_text(json.dumps(record))
            self.assertEqual(audit.inspect_auxiliary(root,cfg,meta,states)['status'],'failed')
            record=json.loads(original);record['summary']['crossing_pedestrian']['observed_reverse_distance_m']=999.
            path.write_text(json.dumps(record))
            self.assertEqual(audit.inspect_auxiliary(root,cfg,meta,states)['status'],'failed')

    def test_repeat_and_history_preserve_stateful_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b';a.mkdir();b.mkdir()
            cfg,_,_=recorded_trace(a);recorded_trace(b)
            self.assertEqual(audit.compare_auxiliary(a,b,cfg,scope='repeat')['status'],'passed')
            path=b/'pedestrian_decisions.json';record=json.loads(path.read_text())
            record['samples'][90]['decisions'][0]['state_after']['mode']='retreat';path.write_text(json.dumps(record))
            self.assertEqual(audit.compare_auxiliary(a,b,cfg,scope='repeat')['status'],'failed')
            self.assertEqual(audit.compare_auxiliary(a,b,cfg,scope='history')['status'],'passed')


if __name__ == '__main__':
    unittest.main()
