"""Offline t0 measurement identity and replay gate tests; no CARLA RPC."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

if __package__:
    from . import audit_history_prefix as gate
    from . import collect_multicam as cm
    from .preflight_physics import prepare_output, collect_branch, HISTORY_BRANCH, HISTORY_MODE
    from .controllers_v2 import select_action_profiles
    from .test_runtime import fixture_states as base_fixture_states
else:
    import audit_history_prefix as gate
    import collect_multicam as cm
    from preflight_physics import prepare_output, collect_branch, HISTORY_BRANCH, HISTORY_MODE
    from controllers_v2 import select_action_profiles
    from test_runtime import fixture_states as base_fixture_states


def fixture_states():
    # Valid shared initialization: the generic runtime fixture puts both actors
    # at the origin, which is deliberately invalid for the history body gate.
    result = base_fixture_states()
    result['position'][:,1,0] = 10.
    result['transform_world_vehicle'][...,:3,3] = result['position']
    return result


def prefix(values):
    result = {key: value.copy() if key in gate.BOXES else value[:41].copy()
              for key,value in values.items()}
    for key in ('control','target_speed'):
        if key in result:
            result[key][-1] = 0.
    return result


class MeasuredHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data = Path(self.temporary.name)/'data'
        self.history = Path(self.temporary.name)/'history'
        cfg = {'scene_id':'scene_0000', 'map':'Town05', 'seed':2, 'family':'rear_following',
            'dt_s':.05, 'history_s':2., 'duration_s':4.,
            'vehicles':[{'role':role,'initial_speed_mps':5.,'blueprint_id':'vehicle.tesla.model3',
                         'route':[[0.,0.,0.,0.],[100.,0.,0.,0.]]} for role in ('ego','rear')],
            'action':{'available_lane_change_directions':[], 'profiles':select_action_profiles([])}}
        self.cfg = cm.finalize_scene(cfg, data=self.data)
        cm.finalize_manifest([self.cfg], self.data)
        (self.data/'maps').mkdir()
        (self.data/'maps/Town05.xodr').write_text('<OpenDRIVE/>')
        self.original = (self.data/'scene_0000/spec.json').read_bytes()
        _, selected, self.manifest = prepare_output(self.data,self.history,history_only=True)
        self.assertEqual(selected, {'scene_0000':[HISTORY_BRANCH]})
        folder = self.history/'scene_0000/branches'/HISTORY_BRANCH
        folder.mkdir(parents=True)
        np.savez_compressed(folder/'states.npz', **prefix(fixture_states()))
        self.meta = {'status':'complete','spec_sha256':self.cfg['spec_sha256'],
            'code_sha256':self.cfg['code_sha256'],'vehicle_roles':['ego','rear'],'collisions':[]}
        cm.write_json(folder/'meta.json',{**self.meta,'branch':HISTORY_BRANCH,'capture_mode':HISTORY_MODE,
            'physics_frames':41,'history_only_complete':True,'t0_index':40})
        self.manifest.update(status='complete',branch_results=[{'scene':'scene_0000','branch':HISTORY_BRANCH,
            'meta_sha256':gate.sha(folder/'meta.json'),'states_sha256':gate.sha(folder/'states.npz')}])
        cm.write_json(self.history/'manifest.json',self.manifest)
        for branch in self.cfg['branches']:
            location = self.data/'scene_0000/branches'/branch
            location.mkdir(parents=True)
            np.savez_compressed(location/'states.npz', **fixture_states())
            cm.write_json(location/'meta.json',{**self.meta,'branch':branch})

    def test_original_spec_snapshot_and_all_six_branch_gate(self):
        self.assertEqual((self.data/'scene_0000/spec.json').read_bytes(), self.original)
        self.assertEqual((self.history/'scene_0000/spec.json').read_bytes(), self.original)
        report=gate.audit(self.data,self.history)
        self.assertTrue(report['passed'],report)
        self.assertEqual(report['branches_checked'],6)
        self.assertEqual(self.manifest['future_s'],0.)
        self.assertEqual(self.manifest['frozen_history_spec_sha256']['scene_0000'],self.cfg['spec_sha256'])

    def test_future_is_ignored_but_observed_t0_and_past_actuation_are_strict(self):
        actual=fixture_states(); measured=prefix(actual)
        actual['position'][41:] = 90.
        actual['control'][40:] = 1.
        actual['target_speed'][40:] = 5.
        self.assertEqual(gate.compare_prefix(actual,measured)['status'],'passed')
        for key, index in (('position',40),('observed_control',40),('control',39)):
            changed=copy.deepcopy(actual); changed[key][index,0,0] += 1e-8
            self.assertEqual(gate.compare_prefix(changed,measured)['status'],'failed')

    def test_all_branches_required_and_partial_measurement_rejected(self):
        folder=self.data/'scene_0000/branches/accelerate'
        (folder/'states.npz').unlink()
        report=gate.audit(self.data,self.history)
        self.assertFalse(report['passed'])
        self.assertEqual(report['branches_checked'],6)
        self.assertTrue(any(r['branch']=='accelerate' and r['status']=='failed' for r in report['branches']))
        self.manifest['status']='failed'
        cm.write_json(self.history/'manifest.json',self.manifest)
        self.assertIn('incomplete',gate.audit(self.data,self.history)['errors'][0])

    def test_linked_future_onset_allowed_but_shared_weather_rejected(self):
        cfg=copy.deepcopy(self.cfg)
        folder=self.history/'scene_0000/branches'/HISTORY_BRANCH
        cfg['action']['profiles']['accelerate']['onset_s']=.25
        cfg['measured_t0_plan']={'schema':'measured-t0-plan-v1',
            'explicit_timing_bounds':{'accelerate':{'onset_s':[0.,.5]}},
            'original_history_spec_sha256':self.cfg['spec_sha256'], 'history':{
                'states_sha256':gate.sha(folder/'states.npz'), 'meta_sha256':gate.sha(folder/'meta.json'),
                'spec_file_sha256':gate.sha(self.history/'scene_0000/spec.json')}}
        def save():
            cfg['spec_sha256']=gate.config_digest(cfg)
            cm.write_json(self.data/'scene_0000/spec.json',cfg)
            for branch in cfg['branches']:
                cm.write_json(self.data/'scene_0000/branches'/branch/'meta.json',
                    {**self.meta,'spec_sha256':cfg['spec_sha256'],'branch':branch})
        save()
        report=gate.audit(self.data,self.history)
        self.assertTrue(report['passed'],report)
        cfg['weather']['cloudiness']+=1.
        save()
        report=gate.audit(self.data,self.history)
        self.assertFalse(report['passed'])
        self.assertTrue(any('Shared setup' in e for e in report['branches'][0]['errors']))

    def test_signal_readback_prefix_cannot_change(self):
        actual={'program':{'x':'fixture'},'groups':[], 'samples':[
            {'frame_id':i+100,'simulation_time_s':float(i),'time_s':float(i),
             'actual_states':{'stable_light':'Green'}} for i in range(121)]}
        reference=copy.deepcopy(actual);reference['samples']=reference['samples'][:41]
        self.assertEqual(gate.compare_signals(actual,reference),[])
        actual['samples'][40]['actual_states']['stable_light']='Red'
        self.assertTrue(gate.compare_signals(actual,reference))

    def test_failed_connection_preserves_failure_without_complete_measurement(self):
        folder=Path(self.temporary.name)/'failed_history'
        with mock.patch.object(cm.carla,'Client',side_effect=RuntimeError('connection timeout')):
            with self.assertRaisesRegex(RuntimeError,'connection timeout'):
                collect_branch(self.data,folder,self.cfg,HISTORY_BRANCH,'127.0.0.1',2000,history_only=True)
        meta=json.loads((folder/'meta.json').read_text())
        self.assertEqual(meta['status'],'failed')
        self.assertFalse(meta['history_only_complete'])
        self.assertFalse((folder/'states.npz').exists())

    def test_rgb_prefix_must_reference_the_same_physics_and_camera_time(self):
        folder=Path(self.temporary.name)/'rgb';folder.mkdir()
        states=fixture_states();indices=np.arange(0,121,10)
        samples={'physics_indices':indices,'time_s':np.arange(13)*.5-2.,
            'frame_id':states['frame_id'][indices],
            'ego_to_world':states['transform_world_vehicle'][indices,0],
            'sensor_frame_id':np.repeat(states['frame_id'][indices,None],7,axis=1)}
        meta={'camera_names':list(gate.CAMERA_NAMES),'rgb_samples':13}
        np.savez_compressed(folder/'samples.npz',**samples)
        self.assertEqual(gate.check_rgb_samples(folder,states,meta)['errors'],[])
        samples['sensor_frame_id'][4,0]+=1
        np.savez_compressed(folder/'samples.npz',**samples)
        self.assertTrue(gate.check_rgb_samples(folder,states,meta)['errors'])

    def test_pedestrian_t0_observed_control_is_shared_outgoing_control_is_not(self):
        actual=fixture_states()
        actual['transform_world_actor']=actual.pop('transform_world_vehicle')
        for key in ('target_speed','observed_gear','observed_manual_gear_shift','requested_gear','requested_manual_gear_shift'):
            actual.pop(key)
        actual['control']=np.zeros((121,2,5));actual['observed_control']=np.zeros((121,2,5))
        measured=prefix(actual); actual['control'][40,0,3]=1.4
        self.assertEqual(gate.compare_prefix(actual,measured,pedestrian=True)['status'],'passed')
        actual['observed_control'][40,0,3]=1.4
        self.assertEqual(gate.compare_prefix(actual,measured,pedestrian=True)['status'],'failed')


if __name__=='__main__':
    unittest.main(verbosity=2)
