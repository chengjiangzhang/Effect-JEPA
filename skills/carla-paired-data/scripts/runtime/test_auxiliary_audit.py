"""Synthetic audit/export regression tests; no simulator or image collection."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
if __package__:
    from . import audit_auxiliary as audit, audit_capture_equivalence as equivalence
    from .export_common import common_arrays, export
    from .review_common import CAMERA_NAMES
else:
    import audit_auxiliary as audit
    import audit_capture_equivalence as equivalence
    from export_common import common_arrays, export
    from review_common import CAMERA_NAMES


def fixture():
    states = {'time_s':np.arange(121)*.05-2, 'frame_id':np.arange(100,221),
              'simulation_time_s':np.arange(121)*.05+10}
    person = {'role':'crossing_pedestrian','blueprint_id':'walker.pedestrian.0001',
              'transform':{'position':[10.,-2.,.5],'rotation_deg':[0.,0.,90.]},
              'route_points':[[10.,-2.,.5],[10.,6.,.5]],'speed_mps':1.2,'start_time_s':.2}
    group = {'id':'ego_signal','landmark_ids':['42'],'roles':['ego'],'stop_route_s':{'ego':18.}}
    program = {'groups':[group],'phases':[{'start_time_s':-3.,'states':{'ego_signal':'Red'}},
                                        {'start_time_s':1.,'states':{'ego_signal':'Green'}}]}
    config = {'vehicle_roles':['ego'],'pedestrians':[person],'traffic_signal_program':program,
              'history_s':2.,'duration_s':4.,'dt_s':.05}
    walkers = {key:np.zeros(shape) for key,shape in audit.pedestrian_shapes(config,121).items()}
    walkers.update({key:value.copy() for key,value in states.items()})
    walkers['position'][:]=[10.,-2.,.5]; walkers['bbox_extent'][:]=[.3,.3,.9]
    walkers['transform_world_actor'][:]=np.eye(4)
    walkers['transform_world_actor'][...,:3,3]=walkers['position']
    walkers['control'][...,1]=1.; walkers['observed_control'][...,1]=1.
    meta={'vehicle_roles':['ego'],'actor_ids':[1],'pedestrian_roles':[person['role']],
          'pedestrians':[{'role':person['role'],'blueprint_id':person['blueprint_id'],'actor_id':2,
                         'actor_attributes':{'role_name':person['role']}}],
          'collisions':[],'collision_sensor_roles':['ego',person['role']],'t0_frame':140}
    signals={'program':deepcopy(program),'groups':[{**deepcopy(group),'actor_ids':[9],
         'light_records':[{'actor_id':9,'landmark_id':'42','transform':{
             'position':[15.,3.,0.],'rotation_deg':[0.,0.,0.]}}]}],'samples':[]}
    for i,time in enumerate(states['time_s']):
        phase=0 if time<1.-1e-9 else 1; value=program['phases'][phase]['states']
        signals['samples'].append({'time_s':float(time),'frame_id':int(states['frame_id'][i]),
             'simulation_time_s':float(states['simulation_time_s'][i]),'phase_index':phase,
             'states':deepcopy(value),'actual_states':{'42':value['ego_signal']}})
    return config,meta,states,walkers,signals


def save(folder,walkers,signals):
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'pedestrians.npz',**walkers)
    (folder/'signal_states.json').write_text(json.dumps(signals))


class AuxiliaryAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        self.config,self.meta,self.states,self.walkers,self.signals=fixture()

    def inspect(self):
        save(self.path,self.walkers,self.signals)
        return audit.inspect_auxiliary(self.path,self.config,self.meta,self.states)

    def comparison(self,scope='exact',change_walkers=None,change_signals=None):
        a,b=self.path/'a',self.path/'b'; save(a,self.walkers,self.signals)
        changed,signal=deepcopy(self.walkers),deepcopy(self.signals)
        if change_walkers:change_walkers(changed)
        if change_signals:change_signals(signal)
        save(b,changed,signal)
        return audit.compare_auxiliary(b,a,self.config,scope=scope)

    def test_valid_measured_auxiliary_records(self):
        self.assertEqual(self.inspect()['status'],'passed')

    def test_required_files_cannot_be_replaced_by_metadata(self):
        result=audit.inspect_auxiliary(self.path,self.config,self.meta,self.states)
        self.assertEqual(result['status'],'failed'); self.assertEqual(len(result['errors']),2)

    def test_vehicle_only_capture_remains_compatible(self):
        self.assertEqual(audit.inspect_auxiliary(self.path,{'vehicle_roles':['ego']},{},{})['status'],'passed')
        self.assertEqual(audit.compare_auxiliary(self.path,self.path,{})['status'],'passed')

    def test_unconfigured_actor_records_fail(self):
        save(self.path,self.walkers,self.signals)
        result=audit.inspect_auxiliary(self.path,{'vehicle_roles':['ego']},{},self.states)
        self.assertEqual(result['status'],'failed');self.assertEqual(len(result['errors']),2)

    def test_actor_identity_and_clock_failures(self):
        self.meta['pedestrian_roles']=['wrong'];self.walkers['frame_id'][12]+=1
        errors=self.inspect()['errors']
        self.assertTrue(any('identity/order' in e for e in errors));self.assertTrue(any('frame_id differs' in e for e in errors))

    def test_invalid_bbox_transform_and_controls_fail(self):
        self.walkers['bbox_extent'][0,0]=-1;self.walkers['transform_world_actor'][2,0,0,0]=2
        self.walkers['control'][5,0,4]=2
        errors=self.inspect()['errors']
        for word in ('extents','not rigid','direction/speed/jump'):self.assertTrue(any(word in e for e in errors))

    def test_signal_without_per_light_readback_fails(self):
        del self.signals['samples'][30]['actual_states']
        self.assertTrue(any('per-light readback' in e for e in self.inspect()['errors']))

    def test_signal_stale_state_at_transition_fails(self):
        self.signals['samples'][60]['actual_states']['42']='Red'
        self.assertTrue(any('readback differs' in e for e in self.inspect()['errors']))

    def test_signal_identity_and_synchronization_failures(self):
        self.signals['groups'][0]['light_records'][0]['landmark_id']='wrong'
        self.signals['samples'][5]['frame_id']+=1
        errors=self.inspect()['errors']
        self.assertTrue(any('measured light identity' in e for e in errors));self.assertTrue(any('not synchronized' in e for e in errors))

    def test_malformed_signal_readback_is_structured_failure(self):
        self.signals['samples'][7]['actual_states']=None
        self.signals['groups'][0]['light_records'][0]=None
        self.assertEqual(self.inspect()['status'],'failed')

    def test_exact_comparison_ignores_reload_ids_and_absolute_frame_offsets(self):
        def walker(a):a['frame_id']+=500
        def signal(a):
            a['groups'][0]['actor_ids']=[99];a['groups'][0]['light_records'][0]['actor_id']=99
            for row in a['samples']:row['frame_id']+=500
        self.assertEqual(self.comparison(change_walkers=walker,change_signals=signal)['status'],'passed')

    def test_exact_comparison_rejects_changed_walker_control(self):
        self.assertEqual(self.comparison(change_walkers=lambda a:a['control'].__setitem__((80,0,3),1.))['status'],'failed')

    def test_history_allows_future_reaction_but_includes_t0(self):
        def future(a):a['position'][41:,0,0]+=2;a['control'][40:,0,3]=1
        self.assertEqual(self.comparison('history',future)['status'],'passed')
        self.assertEqual(self.comparison('history',lambda a:a['position'].__setitem__((40,0,0),11.))['status'],'failed')

    def test_repeat_rejects_future_walker_divergence(self):
        self.assertEqual(self.comparison('repeat',lambda a:a['position'].__setitem__((90,0,0),11.))['status'],'failed')

    def test_signal_full_future_is_fixed_across_action_branches(self):
        def signal(a):a['samples'][90]['actual_states']['42']='Red'
        self.assertEqual(self.comparison('history',change_signals=signal)['status'],'failed')

    def test_signal_history_clock_roundoff_tolerated_but_timing_drift_rejected(self):
        def shifted(signal):
            for row in signal['samples']:row['simulation_time_s']+=10000.
        self.assertEqual(self.comparison('history',change_signals=shifted)['status'],'passed')
        self.assertEqual(self.comparison('repeat',change_signals=shifted)['status'],'passed')
        self.assertEqual(self.comparison('exact',change_signals=shifted)['status'],'failed')
        def drift(signal):
            shifted(signal);signal['samples'][80]['simulation_time_s']+=1e-4
        self.assertEqual(self.comparison('history',change_signals=drift)['status'],'failed')

    def test_post_action_collision_allowed_measured_impulse_required(self):
        self.meta['collisions']=[{'role':'crossing_pedestrian','other_actor_id':1,'other_role':'ego',
             'time_s':1.,'frame_id':160,'normal_impulse':[1.,0.,0.]}]
        self.assertEqual(self.inspect()['status'],'passed')
        self.meta['collisions'][0]['normal_impulse'][0]=float('nan')
        self.assertEqual(self.inspect()['status'],'failed')

    def test_history_contact_or_missing_collision_sensor_fails(self):
        self.meta['collisions']=[{'role':'crossing_pedestrian','other_actor_id':1,'other_role':'ego',
             'time_s':0.,'frame_id':140,'normal_impulse':[1.,0.,0.]}]
        self.assertTrue(any('Contact during' in e for e in self.inspect()['errors']))
        self.meta['collisions']=[];self.meta['collision_sensor_roles']=['ego']
        self.assertEqual(self.inspect()['status'],'failed')

    def test_collision_comparison_preserves_impulse_excludes_reload_ids(self):
        self.meta['collisions']=[{'role':'crossing_pedestrian','other_actor_id':1,'other_role':'ego',
             'time_s':1.,'frame_id':160,'normal_impulse':[1.,0.,0.]}]
        other=deepcopy(self.meta);other['t0_frame']+=100;other['collisions'][0]['frame_id']+=100
        other['collisions'][0]['other_actor_id']=99
        self.assertEqual(audit.collision_identity(self.meta),audit.collision_identity(other))
        other['collisions'][0]['normal_impulse'][0]=2
        self.assertNotEqual(audit.collision_identity(self.meta),audit.collision_identity(other))

    def test_common_geometry_preserves_type_and_bbox_offset(self):
        self.walkers['bbox_offset'][0]=[.1,.2,.3]
        transform=np.eye(4);transform[0,3]=-5
        result=common_arrays(self.walkers,transform,actor_type='pedestrian')
        self.assertNotIn('transform_ego0_vehicle',result)
        np.testing.assert_allclose(result['position_ego0_m'][0,0],[5.,-2.,.5])
        np.testing.assert_allclose(result['bbox_corners_ego0_m'][0,0].mean(0),[5.1,-1.8,.8])

    def full_states(self):
        states=deepcopy(self.states)
        states.update({key:np.zeros((121,1,3)) for key in equivalence.VECTORS})
        states.update({key:np.zeros((121,1)) for key in equivalence.SCALARS})
        states.update({key:np.zeros((1,3)) for key in equivalence.BOXES})
        states['bbox_extent'][:]=[2.,1.,.8]
        states['transform_world_vehicle']=np.broadcast_to(np.eye(4),(121,1,4,4)).copy()
        return states

    def test_export_includes_separate_pedestrian_tracks_and_measured_signals(self):
        data,out=self.path/'data',self.path/'export';scene=data/'scene_0000';scene.mkdir(parents=True)
        (data/'manifest.json').write_text(json.dumps({'scene_ids':['scene_0000']}))
        (scene/'spec.json').write_text(json.dumps(self.config))
        states=self.full_states();samples={'physics_indices':np.arange(0,121,10),'time_s':states['time_s'][::10],
             'frame_id':states['frame_id'][::10],'intrinsics':np.tile(np.eye(3),(6,1,1)),
             'world_to_camera':np.tile(np.eye(4),(13,6,1,1))}
        for branch in ('maintain','brake','accelerate','left','right','maintain_repeat'):
            folder=scene/'branches'/branch;save(folder,self.walkers,self.signals)
            (folder/'meta.json').write_text(json.dumps({**self.meta,'status':'complete','branch':branch}))
            np.savez_compressed(folder/'states.npz',**states);np.savez_compressed(folder/'samples.npz',**samples)
            for camera in CAMERA_NAMES:
                target=folder/'camera'/camera;target.mkdir(parents=True)
                for i in range(13):(target/f'{i:05d}.png').touch()
        self.assertEqual(export(data,out)['status'],'exported')
        tracks=json.loads((out/'scene_0000/maintain/pedestrian_tracks.json').read_text())
        self.assertEqual(tracks['actor_type'],'pedestrian');self.assertEqual(tracks['track_ids'],['scene_0000:crossing_pedestrian'])
        self.assertTrue((out/'scene_0000/maintain/pedestrians_ego0.npz').is_file())
        self.assertEqual(json.loads((out/'scene_0000/maintain/signal_states.json').read_text()),self.signals)

    def test_capture_equivalence_checks_auxiliary_files_not_only_vehicles(self):
        self.config.update(scene_id='scene_0000',code_sha256={name:'a'*64 for name in equivalence.SOURCES})
        self.config['spec_sha256']=equivalence.config_digest(self.config)
        roots={side:self.path/side for side in ('data','reference')}
        manifests={side:{'code_sha256':self.config['code_sha256']} for side in roots}
        for side,root in roots.items():
            folder=root/'scene_0000/branches/maintain';save(folder,self.walkers,self.signals)
            (root/'scene_0000/spec.json').write_text(json.dumps(self.config))
            meta={**self.meta,'status':'complete','branch':'maintain','spec_sha256':self.config['spec_sha256'],
                  'code_sha256':self.config['code_sha256'],'camera_names':list(CAMERA_NAMES),'rgb_samples':13}
            if side=='reference':meta['capture_mode']='physics_only_preflight_not_delivery'
            (folder/'meta.json').write_text(json.dumps(meta));np.savez_compressed(folder/'states.npz',**self.full_states())
        result=equivalence.compare_branch(roots['data'],roots['reference'],'scene_0000','maintain',manifests)
        self.assertTrue(result['equivalent_exact'],result)
        (roots['data']/'scene_0000/branches/maintain/pedestrians.npz').unlink()
        result=equivalence.compare_branch(roots['data'],roots['reference'],'scene_0000','maintain',manifests)
        self.assertFalse(result['equivalent_exact']);self.assertTrue(result['vehicle_physical_fields_exact'])
        self.assertFalse(result['physical_fields_exact'])


if __name__=='__main__':unittest.main()
