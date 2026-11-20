"""Feedback, causality and replay-evidence regressions; no simulator required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

if __package__:
    from . import pedestrian_policy as policy, audit_auxiliary as audit
    from .auxiliary_actors import AuxiliaryActors, validate_auxiliary_config
    from .test_auxiliary import FakeWorld, FakeClient, configuration
else:
    import pedestrian_policy as policy, audit_auxiliary as audit
    from auxiliary_actors import AuxiliaryActors, validate_auxiliary_config
    from test_auxiliary import FakeWorld, FakeClient, configuration


def fixture(two_vehicles=False):
    cfg = configuration(); cfg.pop('traffic_signal_program')
    if not two_vehicles:
        cfg['vehicle_roles'] = ['ego']; cfg['vehicles'] = cfg['vehicles'][:1]
    spec = cfg['pedestrians'][0]
    spec.update(route_points=[[20.,-5.,0.],[20.,5.,0.]],
                transform={'position':[20.,-5.,.8],'rotation_deg':[0.,0.,90.]})
    spec['response_policy'] = {'kind':policy.KIND,
        'vehicle_conflicts':[{'role':r,'entry_s':18.,'exit_s':22.} for r in cfg['vehicle_roles']],
        'pedestrian_entry_s':4.,'pedestrian_exit_s':6.,'clearance_time_s':1.,
        'clearance_distance_m':1.,'stopped_speed_m_s':.3,'yielding_speed_m_s':2.,
        'minimum_deceleration_m_s2':.5}
    return cfg


def vehicles(cfg, x=10., speed=8., acceleration=0.):
    n = len(cfg['vehicle_roles'])
    value = {'position':np.tile([x,0.,0.],(n,1)), 'velocity':np.tile([speed,0.,0.],(n,1)),
        'acceleration':np.tile([acceleration,0.,0.],(n,1)),
        'bbox_extent':np.tile([2.,1.,1.],(n,1)), 'bbox_offset':np.zeros((n,3)),
        'bbox_rotation_deg':np.zeros((n,3)), 'transform_world_vehicle':np.tile(np.eye(4),(n,1,1))}
    value['transform_world_vehicle'][:,:3,3] = value['position']
    return value


def person(cfg):
    point = cfg['pedestrians'][0]['transform']['position']
    transform = np.eye(4); transform[:3,3] = point
    return {'position':np.array([point]), 'transform_world_actor':np.array([transform]),
        'bbox_extent':np.array([[.3,.3,.85]]), 'bbox_offset':np.array([[0.,0.,.85]]),
        'bbox_rotation_deg':np.zeros((1,3))}


def decide(cfg, observed, t=0., committed=False):
    return policy.decision(cfg['pedestrians'][0],cfg,observed,person(cfg),0,t,committed)


class PedestrianFeedbackTests(unittest.TestCase):
    def test_actual_fast_approach_waits_stopped_ego_releases(self):
        cfg = fixture()
        waiting = decide(cfg,vehicles(cfg))
        releasing = decide(cfg,vehicles(cfg,speed=0.))
        self.assertFalse(waiting['allow_walk'])
        self.assertTrue(releasing['allow_walk'])
        self.assertEqual(releasing['vehicle_checks'][0]['reason'],'observed_stopped_before_conflict')
        self.assertIsNone(releasing['vehicle_checks'][0]['entry_eta_s'])

    def test_no_action_name_or_future_plan_affects_current_policy(self):
        cfg = fixture(); original = decide(cfg,vehicles(cfg))
        cfg['action'] = {'branch':'brake','future_ego_positions':[[-999.,0,0]]}
        self.assertEqual(original,decide(cfg,vehicles(cfg)))

    def test_body_must_fully_clear_not_merely_center_or_missing_eta(self):
        cfg = fixture()
        self.assertFalse(decide(cfg,vehicles(cfg,x=23.5,speed=0.))['allow_walk'])
        cleared = decide(cfg,vehicles(cfg,x=26.,speed=0.))
        self.assertTrue(cleared['allow_walk'])
        self.assertEqual(cleared['vehicle_checks'][0]['reason'],'rear_physically_cleared')
        crossing = vehicles(cfg,speed=0.); crossing['velocity'][0]=[0.,2.,0.]
        self.assertFalse(decide(cfg,crossing)['allow_walk'])

    def test_stopped_inside_crossing_is_not_yielding(self):
        cfg = fixture()
        row = decide(cfg,vehicles(cfg,x=20.,speed=0.))
        self.assertFalse(row['allow_walk'])
        self.assertEqual(row['vehicle_checks'][0]['reason'],'vehicle_body_in_conflict_zone')

    def test_actual_bbox_offset_changes_occupancy(self):
        cfg = fixture(); observed = vehicles(cfg,x=14.,speed=0.)
        self.assertTrue(decide(cfg,observed)['allow_walk'])
        observed['bbox_offset'][0,0] = 4.
        self.assertFalse(decide(cfg,observed)['allow_walk'])

    def test_ego_yield_cannot_override_adjacent_vehicle_approach(self):
        cfg = fixture(True); observed = vehicles(cfg,speed=0.)
        observed['velocity'][1,0] = 8.
        row = decide(cfg,observed)
        self.assertFalse(row['allow_walk'])
        self.assertTrue(row['vehicle_checks'][0]['safe_gap_observed'])
        self.assertFalse(row['vehicle_checks'][1]['safe_gap_observed'])

    def test_omitted_crossing_vehicle_is_rejected_but_actual_unlisted_body_also_blocks(self):
        cfg=fixture(True);cfg['pedestrians'][0]['response_policy']['vehicle_conflicts'].pop()
        with self.assertRaisesRegex(ValueError,'omits geometrically'):
            validate_auxiliary_config(cfg)
        cfg['vehicles'][1]['route']=[[0.,30.,0.,0.],[100.,30.,0.,0.]]
        validate_auxiliary_config(cfg)
        observed=vehicles(cfg,speed=0.)
        observed['position'][1]=[20.,0.,0.]
        observed['transform_world_vehicle'][1,:3,3]=observed['position'][1]
        row=decide(cfg,observed)
        self.assertFalse(row['allow_walk'])
        self.assertEqual(row['vehicle_checks'][1]['reason'],'unlisted_vehicle_body_blocks_walk_path')
        observed['position'][1]=[20.,30.,0.]
        observed['transform_world_vehicle'][1,:3,3]=observed['position'][1]
        self.assertTrue(decide(cfg,observed)['allow_walk'])

    def test_continuous_crossing_geometry_not_sparse_knots_and_ignores_overpass(self):
        path=[[20.,-5.,0.],[20.,5.,0.]]
        crossing=policy.crossing_geometry([[0.,0.,0.],[100.,0.,0.]],path)
        self.assertEqual(crossing,{'distance_m':0.,'vehicle_s':20.,'pedestrian_s':5.})
        self.assertIsNone(policy.crossing_geometry([[0.,0.,6.],[100.,0.,6.]],path))

    def test_slow_measured_deceleration_and_margin_are_both_required(self):
        cfg = fixture()
        slowing = decide(cfg,vehicles(cfg,speed=1.5,acceleration=-1.))
        self.assertTrue(slowing['allow_walk'])
        self.assertEqual(slowing['vehicle_checks'][0]['reason'],'observed_slow_deceleration_with_stopping_margin')
        self.assertFalse(decide(cfg,vehicles(cfg,x=15.,speed=2.,acceleration=-.5))['allow_walk'])
        self.assertFalse(decide(cfg,vehicles(cfg,x=15.,speed=1.5,acceleration=1.))['allow_walk'])

    def test_gap_requires_full_body_crossing_and_clearance_time(self):
        cfg = fixture(); safe = decide(cfg,vehicles(cfg,x=0.,speed=1.))
        unsafe = decide(cfg,vehicles(cfg,x=0.,speed=5.))
        self.assertTrue(safe['allow_walk']);self.assertFalse(unsafe['allow_walk'])
        self.assertEqual(safe['vehicle_checks'][0]['reason'],'current_speed_arrival_gap')

    def test_earliest_time_and_commitment_are_distinct(self):
        cfg = fixture(); observed = vehicles(cfg,speed=0.)
        self.assertFalse(decide(cfg,observed,t=-.05)['allow_walk'])
        self.assertTrue(decide(cfg,observed,t=0.)['allow_walk'])
        self.assertTrue(decide(cfg,vehicles(cfg),t=.1,committed=True)['allow_walk'])

    def test_contract_rejects_branch_keys_missing_ego_and_invalid_intervals(self):
        for mutate in (lambda p:p.update(branch='brake'),
                       lambda p:p.update(vehicle_conflicts=[]),
                       lambda p:p.update(pedestrian_exit_s=999.),
                       lambda p:p.update(yielding_speed_m_s=.1)):
            cfg = fixture(); mutate(cfg['pedestrians'][0]['response_policy'])
            with self.assertRaises(ValueError): validate_auxiliary_config(cfg)
        validate_auxiliary_config(configuration())  # Legacy scripted data remain readable.

    def test_record_requires_current_frame_not_previous_observation(self):
        cfg=fixture(); world=FakeWorld(lights=False); aux=AuxiliaryActors(world,FakeClient(world),cfg)
        with self.assertRaisesRegex(RuntimeError,'same frame/time'): aux.record(0.)
        aux.inject_observation(vehicles(cfg),0.); world.tick()
        with self.assertRaisesRegex(RuntimeError,'same frame/time'): aux.record(0.)

    def test_current_observation_is_copied_before_outgoing_action_annotation(self):
        cfg=fixture(); world=FakeWorld(lights=False); aux=AuxiliaryActors(world,FakeClient(world),cfg)
        observed=vehicles(cfg);aux.inject_observation(observed,0.)
        observed['velocity'][:]=0.;observed['branch']='brake'
        self.assertEqual(aux.record(0.)[0,3],0.)


def recorded_trace(folder, *, history_only=False):
    cfg=fixture(); world=FakeWorld(lights=False); aux=AuxiliaryActors(world,FakeClient(world),cfg)
    recorded=[];count=41 if history_only else 121
    for index in range(count):
        time=-2+index*.05
        observed=vehicles(cfg,speed=0. if time>=.2 else 8.)
        recorded.append({key:value.copy() for key,value in observed.items()})
        aux.inject_observation(observed,time)
        outgoing=aux.record(time,final=index==count-1)
        if index<count-1:
            aux.advance_commands(outgoing,time+.05);world.tick()
    aux.save(folder)
    states={key:np.asarray([row[key] for row in recorded]) for key in recorded[0] if key not in audit.WALKER_BOXES}
    states.update({key:recorded[0][key] for key in audit.WALKER_BOXES})
    states.update({key:np.asarray([row[key] for row in aux.samples]) for key in ('time_s','frame_id','simulation_time_s')})
    meta={**aux.metadata(),'actor_ids':[1],'collisions':[],
          'collision_sensor_roles':cfg['vehicle_roles']+[s['role'] for s in cfg['pedestrians']]}
    return cfg,states,meta


class PedestrianEvidenceTests(unittest.TestCase):
    def test_native_call_order_trace_recomputes_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cfg,states,meta=recorded_trace(root)
            checked=audit.inspect_auxiliary(root,cfg,meta,states)
            self.assertEqual(checked['status'],'passed',checked['errors'])
            self.assertIn('pedestrian_decisions.json',checked['input_sha256'])
            record=json.loads((root/'pedestrian_decisions.json').read_text())
            self.assertFalse(record['samples'][40]['decisions'][0]['allow_walk'])
            self.assertTrue(record['samples'][45]['decisions'][0]['allow_walk'])
            summary=record['summary'][cfg['pedestrians'][0]['role']]
            self.assertGreater(summary['observed_first_moving_time_s'],summary['command_release_time_s'])
            record['samples'][40]['decisions'][0]['allow_walk']=True
            (root/'pedestrian_decisions.json').write_text(json.dumps(record))
            self.assertEqual(audit.inspect_auxiliary(root,cfg,meta,states)['status'],'failed')

    def test_observed_motion_summary_is_audited_separately_from_allow_walk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cfg,states,meta=recorded_trace(root)
            record=json.loads((root/'pedestrian_decisions.json').read_text())
            summary=record['summary'][cfg['pedestrians'][0]['role']]
            summary['observed_first_moving_time_s']=summary['command_release_time_s']
            (root/'pedestrian_decisions.json').write_text(json.dumps(record))
            checked=audit.inspect_auxiliary(root,cfg,meta,states)
            self.assertTrue(any('motion onset summary' in error for error in checked['errors']))

    def test_missing_feedback_evidence_is_not_accepted_as_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cfg,states,meta=recorded_trace(root)
            (root/'pedestrian_decisions.json').unlink()
            self.assertEqual(audit.inspect_auxiliary(root,cfg,meta,states)['status'],'failed')

    def test_history_boundary_never_executes_future_feedback_and_replays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cfg,states,meta=recorded_trace(root,history_only=True)
            checked=audit.inspect_auxiliary(root,cfg,meta,states)
            self.assertEqual(checked['status'],'passed',checked['errors'])
            data=audit.read_arrays(root/'pedestrians.npz')
            self.assertTrue(np.array_equal(data['control'][-1],np.zeros((1,5))))
            record=json.loads((root/'pedestrian_decisions.json').read_text())
            self.assertEqual(record['samples'][-1]['decisions'][0]['reason'],'unexecuted_final_boundary')

    def test_same_state_repeat_same_decisions_and_changed_future_is_not_prefix_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            a,b=Path(temporary)/'a',Path(temporary)/'b';a.mkdir();b.mkdir()
            cfg,_,_=recorded_trace(a);recorded_trace(b)
            self.assertEqual(audit.compare_auxiliary(a,b,cfg,scope='repeat')['status'],'passed')
            record=json.loads((b/'pedestrian_decisions.json').read_text())
            record['samples'][60]['decisions'][0]['reason']='different_future_response'
            (b/'pedestrian_decisions.json').write_text(json.dumps(record))
            self.assertEqual(audit.compare_auxiliary(a,b,cfg,scope='history')['status'],'passed')
            self.assertEqual(audit.compare_auxiliary(a,b,cfg,scope='repeat')['status'],'failed')


if __name__ == '__main__':
    unittest.main()
