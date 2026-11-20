"""Causal refuge selection and native-feedback evidence regressions."""
import copy
import unittest
import numpy as np

if __package__:
    from . import pedestrian_crossing as c
    from .test_pedestrian_crossing import config, step
    from .test_pedestrian_policy import vehicles
    from .interaction_coverage import active_feedback,annotate_paired_feedback
else:
    import pedestrian_crossing as c
    from test_pedestrian_crossing import config, step
    from test_pedestrian_policy import vehicles
    from interaction_coverage import active_feedback,annotate_paired_feedback


def v4():
    cfg=config()
    cfg['pedestrians'][0]['response_policy'].update(kind=c.KIND_V4,crossing_feedback=dict(c.DEFAULTS_V3))
    return cfg


class RefugeTests(unittest.TestCase):
    def test_reachable_stop_before_corridor_and_retreat_after_entry(self):
        _,stop,_=step(v4(),y=-2.5,car_x=0,car_speed=8)
        _,retreat,_=step(v4(),y=-1.5,car_x=0,car_speed=8)
        self.assertEqual(stop['mode'],'stop')
        self.assertEqual(retreat['mode'],'retreat')
        self.assertTrue(stop['alternatives']['stop']['body_clear'])
        self.assertFalse(retreat['alternatives']['stop']['body_clear'])
        t=retreat['refuge_comparison']['dynamic_clearance_times_s']
        self.assertLess(t['retreat'],t['fast'])

    def test_forward_escape_when_return_is_not_reachable(self):
        _,row,_=step(v4(),y=-.5,car_x=4,car_speed=8)
        self.assertEqual(row['mode'],'fast')
        self.assertIsNone(row['refuge_comparison']['dynamic_clearance_times_s']['retreat'])
        self.assertTrue(row['alternatives']['fast']['body_clear'])

    def test_no_risk_or_already_clearing_retains_progress(self):
        for args in ({'y':-2.5,'car_x':-100,'car_speed':0},
                     {'y':1.5,'car_x':8,'car_speed':8}):
            _,row,_=step(v4(),**args)
            self.assertEqual(row['mode'],'continue')

    def test_barely_body_clear_is_not_a_completed_refuge(self):
        # The actor is already stationary outside the minimum body envelope,
        # but still within the existing caution corridor of an approaching car.
        # Continue a feasible retreat instead of calling that a safe waiting place.
        _,row,_=step(v4(),y=-1.8,speed=0.,car_x=0,car_speed=8)
        self.assertTrue(row['alternatives']['stop']['body_clear'])
        self.assertFalse(row['alternatives']['stop']['caution_clear'])
        self.assertEqual(row['mode'],'retreat')
        self.assertEqual(row['refuge_comparison']['stop_refuge_clearance_margin_m'],.75)

    def test_scene_intent_branch_and_future_do_not_change_decision(self):
        cfg=v4(); observed=vehicles(cfg,x=0,speed=8)
        expected=step(cfg,y=-1.5,observed=observed)
        cfg['interaction_design']={'requested_phase':'clearing','desired_response':'fast'}
        cfg['action']={'branch':'yield_wait','future_positions':[[999.,999.,0.]]}
        observed.update(control=np.zeros((1,3)),target_speed=np.zeros(1),branch='stop_hold')
        actual=step(cfg,y=-1.5,observed=observed)
        np.testing.assert_array_equal(actual[0],expected[0])
        self.assertEqual(actual[1:],expected[1:])

    def test_rear_actor_still_blocks_retreat(self):
        cfg=v4();cfg['vehicle_roles'].append('rear')
        cfg['vehicles'].append({'role':'rear','route':[[0.,-3.,0.],[100.,-3.,0.]]})
        observed=vehicles(cfg,x=0,speed=8)
        observed['position'][1]=[20.,-3.,0.]
        observed['transform_world_vehicle'][1,:3,3]=observed['position'][1]
        observed['velocity'][1]=0.
        _,row,_=step(cfg,y=-1.5,observed=observed)
        self.assertFalse(row['alternatives']['retreat']['body_clear'])
        self.assertNotEqual(row['mode'],'retreat')

    def test_v3_legacy_ranking_not_silently_redefined(self):
        cfg=v4();cfg['pedestrians'][0]['response_policy']['kind']=c.KIND_V3
        _,row,_=step(cfg,y=-2.5,car_x=0,car_speed=8)
        self.assertEqual(row['mode'],'fast')
        self.assertNotIn('refuge_comparison',row)

    def test_oblique_exit_uses_aabb_not_two_clear_endpoints(self):
        tangent=np.array([np.cos(np.deg2rad(10)),np.sin(np.deg2rad(10))])
        normal=np.array([2**-.5,-2**-.5,0.])
        orientation=np.column_stack(([normal[1],-normal[0],0.],normal,[0.,0.,1.]))
        walker_half=np.array([.4,.4,.9]);extent=np.array([2.,1.,1.])
        half=extent[1]+walker_half@np.abs(normal)+c.DEFAULTS_V3['caution_margin_m']
        delta=.2925*np.r_[tangent,0.]
        middle=-normal*(half+abs(delta@normal)/2.+.01)
        initial=-normal*(half+1.)
        command=np.array([initial,middle+delta/2.]);measured=np.array([initial,middle-delta/2.])
        old=min(-command[-1]@normal-half,-measured[-1]@normal-half)
        self.assertAlmostEqual(old,.01)
        alternatives={'continue':{'actor_checks':[{'role':'ego','caution_time_s':1.}]},
                      'stop':{'body_clear':True},'fast':{'body_clear':False}}
        paths={'continue':(command,measured),'stop':(command,measured),'fast':(command,measured)}
        _,evidence=c.entry_side_refuge(alternatives,paths,
            [('ego',np.zeros(3),orientation,extent,np.zeros(3),0.)],
            walker_half,tangent,c.DEFAULTS_V3,np.array([0.,1.]))
        self.assertLess(evidence['threats'][0]['endpoint_near_side_clearance_m']['stop'],0.)

    def test_exit_prediction_accounts_for_lateral_obstacle_velocity(self):
        cfg=v4();observed=vehicles(cfg,x=0,speed=8)
        observed['velocity'][0,1]=-.5
        _,row,_=step(cfg,y=-2.5,observed=observed)
        self.assertIsNone(row['refuge_comparison']['dynamic_clearance_times_s']['stop'])
        self.assertTrue(row['refuge_comparison']['uses_same_moving_obstacles_and_aabb_envelope_as_risk'])


def event_fixture():
    times=np.arange(-40,81)*.05
    spec={'dt_s':.05,'pedestrians':[{'speed_mps':1.,'route_points':[[0.,0.,0.],[0.,20.,0.]]}]}
    speed=np.ones(len(times));station=np.cumsum(speed)*.05
    records={'time_s':times,'position':np.zeros((len(times),1,3)),
             'velocity':np.zeros((len(times),1,3)), 'observed_control':np.zeros((len(times),1,5))}
    records['position'][:,0,1]=station
    records['velocity'][:,0,1]=speed
    records['observed_control'][:,0,1]=1.
    records['observed_control'][:,0,3]=speed
    decisions={'samples':[{'time_s':float(t-.05),'decisions':[{'mode':'continue'}]} for t in times]}
    return spec,records,decisions,{'collisions':[]}


def set_motion(args,start,end,mode,speed):
    _,r,d,_=args;mask=(r['time_s']>=start-1e-8)&(r['time_s']<=end+1e-8)
    r['velocity'][mask,0,1]=speed
    r['observed_control'][mask,0,1]=1 if speed>=0 else -1
    r['observed_control'][mask,0,3]=abs(speed)
    r['position'][:,0,1]=np.cumsum(r['velocity'][:,0,1])*.05
    for index in np.flatnonzero(mask):d['samples'][index]['decisions'][0]['mode']=mode


class NativeFeedbackEvidenceTests(unittest.TestCase):
    def test_stop_retreat_fast_and_resume_require_aligned_native_motion(self):
        args=event_fixture()
        set_motion(args,.5,1.,'stop',0.)
        set_motion(args,1.5,2.,'retreat',-.8)
        set_motion(args,2.5,3.,'fast',1.5)
        events=active_feedback(*args)['events']
        for name in ('stop','retreat','fast','resume'):self.assertTrue(events[name],name)
        self.assertAlmostEqual(events['stop'][0]['start_s'],.5)
        self.assertAlmostEqual(events['retreat'][0]['reverse_path_m'],.4)

    def test_issued_stop_while_still_moving_is_not_stop_or_recovery(self):
        args=event_fixture()
        set_motion(args,.5,1.,'stop',1.)
        events=active_feedback(*args)['events']
        self.assertFalse(events['stop']);self.assertFalse(events['resume'])

    def test_startup_end_of_route_and_obstacle_stillness_are_not_active_stop(self):
        args=event_fixture();set_motion(args,-2.,1.,'stop',0.)
        self.assertFalse(active_feedback(*args)['events']['stop'])
        args=event_fixture();set_motion(args,.5,1.,'stop',0.)
        args[1]['position'][args[1]['time_s']>=.5,0,1]=20.
        self.assertFalse(active_feedback(*args)['events']['stop'])
        args=event_fixture();set_motion(args,.5,1.,'continue',0.)
        args[1]['observed_control'][:,0,3]=1.
        self.assertFalse(active_feedback(*args)['events']['stop'])

    def test_contact_truncates_feedback_and_short_mode_blips_do_not_count(self):
        args=event_fixture();set_motion(args,.5,.65,'stop',0.)
        set_motion(args,1.5,2.,'fast',1.5)
        args[3]['collisions']=[{'time_s':1.6}]
        events=active_feedback(*args)['events']
        self.assertFalse(events['stop']);self.assertFalse(events['fast'])

    def test_stop_selected_after_preexisting_blockage_is_not_active_stop(self):
        args=event_fixture()
        set_motion(args,-1.,1.5,'continue',0.)
        set_motion(args,1.55,2.5,'stop',0.)
        self.assertFalse(active_feedback(*args)['events']['stop'])

    def test_pre_t0_mode_carried_into_future_is_not_new_feedback(self):
        args=event_fixture();set_motion(args,-.8,1.,'fast',1.5)
        self.assertFalse(active_feedback(*args)['events']['fast'])

    def test_identical_responses_are_not_paired_feedback_coverage(self):
        args=event_fixture();set_motion(args,.5,1.,'stop',0.)
        records={b:copy.deepcopy(args[1]) for b in ('maintain','maintain_repeat','accelerate')}
        reports={b:active_feedback(args[0],records[b],args[2],args[3]) for b in ('maintain','accelerate')}
        annotate_paired_feedback(records,reports)
        self.assertFalse(reports['accelerate']['events']['stop'][0]['paired_motion_evidence']['has_different_response'])
        records['accelerate']['position'][records['accelerate']['time_s']>=.75,0,1]+=.2
        annotate_paired_feedback(records,reports)
        self.assertTrue(reports['accelerate']['events']['stop'][0]['paired_motion_evidence']['has_different_response'])


if __name__=='__main__':unittest.main()
