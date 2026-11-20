"""Synthetic native LiDAR packet tests; no CARLA server or dataset access."""
import copy
import json
from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

if __package__:
    from . import semantic_lidar as lidar
else:
    import semantic_lidar as lidar


def measurement(points, world=None, frame=100, timestamp=5., channels=64):
    counts = [len(points)]+[0]*(channels-1)
    return SimpleNamespace(raw_data=points.tobytes(), frame=frame, timestamp=timestamp,
        channels=channels, horizontal_angle=0., get_point_count=lambda index: counts[index],
        transform=SimpleNamespace(get_matrix=lambda: np.eye(4) if world is None else world))


def add_lidar_fixture(folder, config, meta, states, samples):
    spec = config['semantic_lidar']
    meta['sensor_names'] = [item['name'] for item in config.get('camera_rig', [])]+[spec['name']]
    meta['actor_ids'] = list(range(101, 101+len(config['vehicle_roles'])))
    ids = dict(zip(meta['actor_ids'], config['vehicle_roles']))
    attrs = {'role_name': spec['name'], **spec['attributes']}
    meta['semantic_lidar'] = lidar.metadata(spec, attrs, {k: str(v) for k,v in attrs.items()}, ids)
    states.setdefault('simulation_time_s', np.arange(121)*.05+50)
    samples['sensor_frame_id'] = np.repeat(states['frame_id'][::10, None], 7, axis=1)
    samples['lidar_frame_id'] = states['frame_id'][::10].copy()
    samples['lidar_timestamp_s'] = states['simulation_time_s'][::10].copy()
    samples['lidar_to_ego'] = lidar.rigid_from_spec(spec['transform'])
    samples['lidar_to_world'] = samples['ego_to_world'] @ samples['lidar_to_ego']
    points = np.array([(10., 2., -1., .7, 101, 14), (15., 0., -2., .5, 9999, 1)], dtype=lidar.DTYPE)
    for i in range(13):
        packet = measurement(points, samples['lidar_to_world'][i],
            int(samples['lidar_frame_id'][i]), float(samples['lidar_timestamp_s'][i]))
        lidar.save_packet(packet, Path(folder)/'lidar'/lidar.NAME/f'{i:05d}.npz', spec=spec,
            expected_frame=packet.frame, expected_timestamp=packet.timestamp, relative_time=-2+i*.5)


class SemanticLidarTests(unittest.TestCase):
    def setUp(self):
        self.spec = lidar.default_spec()
        self.points = np.array([(10., 2., -1., .7, 4000000001, 14)], dtype=lidar.DTYPE)

    def test_native_binary_layout_preserves_uint_ids_and_incidence(self):
        result = lidar.decode(self.points.tobytes())
        self.assertEqual(result.dtype.itemsize, 24)
        self.assertEqual(int(result['ObjIdx'][0]), 4000000001)
        self.assertEqual(result['ObjIdx'].dtype, np.dtype('uint32'))
        np.testing.assert_array_equal(result, self.points)

    def test_truncated_nonfinite_and_invalid_cosine_rejected(self):
        for raw in (self.points.tobytes()[:-1], b'0'):
            with self.assertRaises(ValueError): lidar.decode(raw)
        for key,value in (('x',np.nan),('CosAngle',1.2)):
            points=self.points.copy();points[key]=value
            with self.assertRaises(ValueError): lidar.decode(points.tobytes())

    def test_frozen_full_sweep_configuration_and_unknown_attributes(self):
        lidar.validate_spec(self.spec)
        for key, value in (('rotation_frequency',2.),('sensor_tick',.5),('horizontal_fov',180.),('channels',0)):
            spec=copy.deepcopy(self.spec);spec['attributes'][key]=value
            with self.assertRaises(ValueError):lidar.validate_spec(spec)
        spec=copy.deepcopy(self.spec);spec['attributes']['noise_stddev']=1.
        with self.assertRaises(ValueError):lidar.validate_spec(spec)

    def test_sensor_blueprint_requires_all_attributes(self):
        values={}
        bp=SimpleNamespace(has_attribute=lambda key:True,set_attribute=lambda key,value:values.update({key:value}))
        library=SimpleNamespace(find=lambda name:bp)
        lidar.blueprint(library,self.spec)
        self.assertEqual(values['rotation_frequency'],'20.0')
        self.assertEqual(values['sensor_tick'],'0.0')
        bp.has_attribute=lambda key:key!='rotation_frequency'
        with self.assertRaises(RuntimeError):lidar.blueprint(library,self.spec)

    def test_native_packet_alignment_and_counts(self):
        packet=measurement(self.points)
        kwargs=dict(spec=self.spec,expected_frame=100,expected_timestamp=5.,relative_time=0.)
        self.assertEqual(lidar.packet_arrays(packet,**kwargs)['points'].shape,(1,))
        for field,value in (('frame',101),('timestamp',5.05),('channels',32)):
            bad=copy.copy(packet);setattr(bad,field,value)
            with self.assertRaises(ValueError):lidar.packet_arrays(bad,**kwargs)
        packet.get_point_count=lambda index:0
        with self.assertRaises(ValueError):lidar.packet_arrays(packet,**kwargs)

    def test_callback_barrier_rejects_missing_future_and_wrong_timestamp(self):
        # Import collector only for its synchronous queue barrier: no Client call.
        if __package__:
            from .collect_multicam import receive_frame_barrier
        else:
            from collect_multicam import receive_frame_barrier
        names=[f'CAM_{i}' for i in range(6)]+[lidar.NAME]
        queues=[queue.Queue() for _ in names]
        for q in queues:q.put(measurement(self.points))
        self.assertEqual(len(receive_frame_barrier(queues,names,100,5.,timeout=.01)),7)
        with self.assertRaises(RuntimeError):receive_frame_barrier(queues,names,100,5.,timeout=.001)
        for q in queues:q.put(measurement(self.points,frame=101))
        with self.assertRaises(RuntimeError):receive_frame_barrier(queues,names,100,5.,timeout=.01)
        queues=[queue.Queue() for _ in names]
        for q in queues:q.put(measurement(self.points,timestamp=5.05))
        with self.assertRaises(RuntimeError):receive_frame_barrier(queues,names,100,5.,timeout=.01)

    def fixture(self, folder):
        config={'semantic_lidar':self.spec,'vehicle_roles':['ego','rear']}
        meta={}
        states={'frame_id':np.arange(1000,1121)}
        samples={'ego_to_world':np.repeat(np.eye(4)[None],13,axis=0)}
        samples['ego_to_world'][:,0,3]=np.arange(13)
        add_lidar_fixture(folder,config,meta,states,samples)
        return config,meta,states,samples

    def test_audit_checks_pose_clock_ids_and_keeps_unknown_raw_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.fixture(folder)
            report=lidar.inspect_branch(folder,*args)
            self.assertEqual(report['status'],'passed',report)
            self.assertEqual(report['point_count'],26)
            self.assertEqual(set(args[1]['semantic_lidar']['actor_to_role']),{'101','102'})
            for key in ('lidar_to_world','lidar_timestamp_s','lidar_frame_id'):
                samples={k:v.copy() for k,v in args[3].items()};samples[key].flat[0]+=1
                self.assertEqual(lidar.inspect_branch(folder,*args[:3],samples)['status'],'failed')

    def test_missing_corrupt_packet_and_role_map_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.fixture(folder)
            args[1]['semantic_lidar']['actor_to_role']['9999']='invented_track'
            self.assertEqual(lidar.inspect_branch(folder,*args)['status'],'failed')
            args[1]['semantic_lidar']['actor_to_role'].pop('9999')
            (Path(folder)/'lidar'/lidar.NAME/'00004.npz').unlink()
            self.assertEqual(lidar.inspect_branch(folder,*args)['status'],'failed')
            args=self.fixture(folder)
            (Path(folder)/'lidar'/lidar.NAME/'00004.npz').write_bytes(b'corrupted-native-npz')
            report=lidar.inspect_branch(folder,*args)
            self.assertEqual(report['status'],'failed')
            self.assertTrue(report['errors'])

    def test_common_frame_export_keeps_raw_lidar_and_registers_transforms(self):
        if __package__:
            from .test_runtime import fixture_states
            from .export_common import export, CAMERA_NAMES
        else:
            from test_runtime import fixture_states
            from export_common import export, CAMERA_NAMES
        cfg = {'vehicle_roles': ['ego', 'rear'], 'vehicles': [{'role':'ego'}, {'role':'rear'}],
               'action': {}, 'semantic_lidar': self.spec}
        with tempfile.TemporaryDirectory() as temporary:
            data, out = Path(temporary)/'data', Path(temporary)/'export'
            scene = data/'scene_0000'; scene.mkdir(parents=True)
            (data/'manifest.json').write_text(json.dumps({'scene_ids':['scene_0000']}))
            (scene/'spec.json').write_text(json.dumps(cfg))
            states=fixture_states(); indices=np.arange(0,121,10)
            for branch in ('maintain','brake','accelerate','left','right','maintain_repeat'):
                folder=scene/'branches'/branch
                samples={'physics_indices':indices,'time_s':states['time_s'][indices],
                         'frame_id':states['frame_id'][indices],
                         'intrinsics':np.broadcast_to(np.eye(3),(6,3,3)),
                         'world_to_camera':np.broadcast_to(np.eye(4),(13,6,4,4)),
                         'ego_to_world':states['transform_world_vehicle'][indices,0]}
                meta={'status':'complete','vehicle_roles':cfg['vehicle_roles'],'collisions':[]}
                add_lidar_fixture(folder,cfg,meta,states,samples)
                (folder/'meta.json').write_text(json.dumps(meta))
                np.savez_compressed(folder/'states.npz',**states)
                np.savez_compressed(folder/'samples.npz',**samples)
                for camera in CAMERA_NAMES:
                    directory=folder/'camera'/camera;directory.mkdir(parents=True)
                    for index in range(13):(directory/f'{index:05d}.png').write_bytes(b'fixture-image-path')
            export(data,out)
            frame=json.loads((out/'scene_0000/frame.json').read_text())
            calibration=json.loads((out/'scene_0000/maintain/lidar.json').read_text())
            self.assertEqual(calibration['stable_actor_tracks']['101'],'scene_0000:ego')
            self.assertTrue(all((out/'scene_0000/maintain'/name).is_file() for name in calibration['raw_files']))
            with np.load(out/'scene_0000/maintain/states_ego0.npz') as result:
                np.testing.assert_allclose(result['lidar_to_ego0'],
                    np.asarray(frame['world_to_ego0']) @ samples['lidar_to_world'])
            self.assertEqual(len(list(data.rglob('lidar/LIDAR_TOP/*.npz'))),78)
            self.assertIn('semantic_lidar', (out/'index.html').read_text())

    def test_no_overhead_preview_and_partial_write(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.fixture(folder)
            (Path(folder)/'preview').mkdir()
            self.assertEqual(lidar.inspect_branch(folder,*args)['status'],'failed')
            bad=measurement(self.points,frame=999)
            target=Path(folder)/'bad.npz'
            with self.assertRaises(ValueError):
                lidar.save_packet(bad,target,spec=self.spec,expected_frame=100,expected_timestamp=5.,relative_time=0.)
            self.assertFalse(target.exists())


if __name__=='__main__':unittest.main(verbosity=2)
