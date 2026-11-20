"""Offline contracts for the portable runtime; no CARLA server required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

if __package__:
    from . import collect_multicam as capture
    from .audit_capture_equivalence import audit as audit_equivalence
    from .audit_preflight_progress import branch_check, physical_differences
    from .controllers_v2 import select_action_profiles
    from .export_common import common_arrays, export as export_common
else:
    import collect_multicam as capture
    from audit_capture_equivalence import audit as audit_equivalence
    from audit_preflight_progress import branch_check, physical_differences
    from controllers_v2 import select_action_profiles
    from export_common import common_arrays, export as export_common


def fixture_states():
    t, n = 121, 2
    result = {key: np.zeros((t,n,3)) for key in
              ('position','rotation_deg','velocity','angular_velocity','acceleration','control','observed_control')}
    result['time_s'] = np.arange(t)*.05-2.
    result['frame_id'] = np.arange(t)
    result['simulation_time_s'] = np.arange(t)*.05
    result['transform_world_vehicle'] = np.broadcast_to(np.eye(4),(t,n,4,4)).copy()
    result['target_speed'] = np.zeros((t,n))
    result['bbox_extent'] = np.ones((n,3))
    result['bbox_offset'] = np.zeros((n,3))
    result['bbox_rotation_deg'] = np.zeros((n,3))
    for key in ('observed_gear','observed_manual_gear_shift','requested_gear','requested_manual_gear_shift'):
        result[key] = np.zeros((t,n))
    return result


class RuntimeTests(unittest.TestCase):
    def test_live_observation_preserves_actual_noncentral_rotated_body_geometry(self):
        vector = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
        pose = SimpleNamespace(location=vector(10., 20., 1.),
            rotation=SimpleNamespace(roll=1., pitch=2., yaw=90.), get_matrix=lambda: np.eye(4))
        actor = SimpleNamespace(get_transform=lambda: pose,
            get_velocity=lambda: vector(1., 2., 0.),
            get_angular_velocity=lambda: vector(0., 0., .2),
            get_acceleration=lambda: vector(-1., 0., 0.),
            bounding_box=SimpleNamespace(extent=vector(2.4, .9, .8), location=vector(.6, -.2, .3),
                rotation=SimpleNamespace(roll=3., pitch=4., yaw=17.)))
        observed = capture.observe([actor])
        np.testing.assert_array_equal(observed['bbox_extent'], [[2.4, .9, .8]])
        np.testing.assert_array_equal(observed['bbox_offset'], [[.6, -.2, .3]])
        np.testing.assert_array_equal(observed['bbox_rotation_deg'], [[3., 4., 17.]])
        np.testing.assert_array_equal(observed['rotation_deg'], [[1., 2., 90.]])

    def test_reactive_policy_source_is_required_and_legacy_capture_remains_readable(self):
        if __package__:
            from .audit_capture_equivalence import source_identity, LIDAR_SOURCES, SOURCES
            from . import collect_multicam_batch, preflight_physics
        else:
            from audit_capture_equivalence import source_identity, LIDAR_SOURCES, SOURCES
            import collect_multicam_batch, preflight_physics
        previous = {name: 'a'*64 for name in LIDAR_SOURCES}
        self.assertEqual(source_identity({'code_sha256': previous}), previous)
        reactive = {'code_sha256': previous, 'pedestrians': [{'response_policy': {'kind': 'gap_acceptance_v1'}}]}
        self.assertIsNone(source_identity(reactive))
        reactive['code_sha256'] = {name: 'a'*64 for name in SOURCES}
        self.assertEqual(source_identity(reactive), reactive['code_sha256'])
        self.assertEqual(set(capture.source_hashes()), set(SOURCES))
        self.assertEqual(set(collect_multicam_batch.SOURCE_FILES), set(SOURCES))
        self.assertEqual(set(preflight_physics.SOURCE_FILES), set(SOURCES))
        merged={'vehicles':[{'route_maneuver':{'policy_version':'observed-gap-merge-v1'}}],
                'code_sha256':dict(capture.source_hashes())}
        self.assertEqual(source_identity(merged),merged['code_sha256'])
        merged['code_sha256'].pop('npc_merge.py')
        self.assertIsNone(source_identity(merged))

    def test_history_excludes_t0_outgoing_control_but_includes_state(self):
        a = fixture_states(); b = copy.deepcopy(a)
        b['control'][40,0,0] = 1.
        b['target_speed'][40,0] = 3.
        self.assertEqual(physical_differences(a,b,history=True)['control'],0.)
        self.assertEqual(physical_differences(a,b,history=True)['target_speed'],0.)
        b['position'][40,0,0] = .3
        self.assertAlmostEqual(physical_differences(a,b,history=True)['position'],.3)
        b['control'][39,0,0] = 1.
        self.assertEqual(physical_differences(a,b,history=True)['control'],1.)

    def check_contact(self, time):
        states = fixture_states()
        cfg = {'vehicle_roles':['ego','rear'],'vehicles':[{'role':'ego'},{'role':'rear'}],
               'code_sha256':{'fixture':'test'},'spec_sha256':'fixture','action':{}}
        meta = {'vehicle_roles':cfg['vehicle_roles'],'branch':'maintain','code_sha256':cfg['code_sha256'],
                'spec_sha256':cfg['spec_sha256'],'collisions':[{'time_s':time}]}
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'maintain';folder.mkdir()
            np.savez_compressed(folder/'states.npz',**states)
            return branch_check(folder,cfg,meta)[0]

    def test_post_action_contact_is_retained(self):
        result=self.check_contact(1.85)
        self.assertTrue(result['passed'])
        self.assertFalse(result['strict_no_contact_complete_lane_passed'])
        self.assertEqual(len(result['post_intervention_collisions']),1)

    def test_initialization_contact_is_rejected(self):
        for time in (-1.,0.):
            self.assertFalse(self.check_contact(time)['passed'])

    def test_common_frame_uses_one_fixed_reference(self):
        states = fixture_states()
        states['position'][:,:,0] = 10.
        states['position'][:,:,1] = 3.
        states['transform_world_vehicle'][...,:3,3] = states['position']
        states['bbox_offset'][:,0] = 2.
        transform=np.array([[0.,1.,0.,-3.],[-1.,0.,0.,10.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
        common=common_arrays(states,transform)
        np.testing.assert_allclose(common['position_ego0_m'],0.)
        np.testing.assert_allclose(common['bbox_corners_ego0_m'].mean(2),np.broadcast_to([0.,-2.,0.],(121,2,3)))
        np.testing.assert_allclose(common['transform_ego0_vehicle'][...,:3,3],common['position_ego0_m'])

    def test_bundled_evidence_resolves_without_project(self):
        preset,_=capture.load_render_preset(capture.RESOURCES/'daylight.json')
        self.assertEqual(preset['validation_status'],'reference_parameters_requires_local_validation')
        self.assertEqual(preset['engine']['evidence'][0]['path'],'resource://engine.ini')
        profile=capture.vehicle_physics_profile(['resource://physics_6mps.json','resource://physics_10mps.json'])
        self.assertEqual(profile['calibrated_speeds_m_s'],[6.,10.])
        self.assertTrue(all(e['path'].startswith('resource://') for e in profile['calibration']))

    def test_finalize_freezes_weather_and_refuses_overwrite(self):
        cfg={'scene_id':'scene_0000','map':'Town05','seed':2,'family':'rear_following',
             'weather':{'sun_altitude_angle':99.},'weather_override':{'cloudiness':30.},
             'vehicles':[{'role':'ego','blueprint_id':'vehicle.tesla.model3','route':[[0.,0.,0.,0.],[20.,0.,0.,0.]]}],
             'action':{'available_lane_change_directions':[],'profiles':select_action_profiles([])}}
        with tempfile.TemporaryDirectory() as temporary:
            data=Path(temporary)
            frozen=capture.finalize_scene(cfg,data=data)
            self.assertEqual(frozen['weather']['cloudiness'],30.)
            self.assertEqual(frozen['weather']['sun_altitude_angle'],45.)
            self.assertEqual(frozen['code_sha256'],capture.source_hashes())
            manifest=capture.finalize_manifest([frozen],data,extra={'plan_identity':'test'})
            self.assertEqual(manifest['planned_branch_count'],6)
            self.assertNotIn('maintain_repeat',manifest['scene_model_branches']['scene_0000'])
            capture.check_frozen_render_preset(data,frozen)
            self.assertEqual(json.loads((data/'scene_0000/spec.json').read_text()),frozen)
            with self.assertRaises(FileExistsError):capture.finalize_scene(cfg,data=data)

    def test_full_equivalence_audit_writes_real_success_and_mismatch(self):
        cfg={'scene_id':'scene_0000','map':'Town05','seed':2,'family':'rear_following',
             'dt_s':.05,'history_s':2.,'duration_s':4.,
             'vehicles':[{'role':role,'blueprint_id':'vehicle.tesla.model3','route':[[0.,0.,0.,0.],[20.,0.,0.,0.]]}
                         for role in ('ego','rear')],
             'action':{'available_lane_change_directions':[],'profiles':select_action_profiles([])}}
        with tempfile.TemporaryDirectory() as temporary:
            data=Path(temporary)/'data'; reference=Path(temporary)/'physics'
            frozen=capture.finalize_scene(cfg,data=data)
            capture.finalize_manifest([frozen],data)
            import shutil
            shutil.copytree(data,reference)
            for root in (data,reference):
                for branch in frozen['branches']:
                    folder=root/'scene_0000/branches'/branch;folder.mkdir(parents=True)
                    np.savez_compressed(folder/'states.npz',**fixture_states())
                    meta={'status':'complete','vehicle_roles':['ego','rear'],'branch':branch,
                          'code_sha256':frozen['code_sha256'],'spec_sha256':frozen['spec_sha256']}
                    if root==reference:meta['capture_mode']='physics_only_preflight_not_delivery'
                    else:meta.update(camera_names=[n[0] for n in capture.RIG],rgb_samples=13)
                    capture.write_json(folder/'meta.json',meta)
            report=audit_equivalence(data,reference)
            self.assertTrue(report['equivalent_exact'],report)
            self.assertEqual(report['status'],'equivalent_exact')
            self.assertEqual(report['compared_branches'],6)
            changed=fixture_states();changed['position'][80,1,0]=.01
            np.savez_compressed(data/'scene_0000/branches/brake/states.npz',**changed)
            report=audit_equivalence(data,reference)
            self.assertFalse(report['equivalent_exact'])
            self.assertEqual(report['status'],'mismatch')

    def test_export_keeps_all_five_paired_branches_with_similar_motion(self):
        if __package__:
            from .test_realized_actions import lane_config, observations
        else:
            from test_realized_actions import lane_config, observations
        from PIL import Image
        cfg = lane_config()
        cfg.update(spec_sha256='fixture-config', code_sha256={'fixture': 'source'})
        with tempfile.TemporaryDirectory() as temporary:
            data, out = Path(temporary)/'data', Path(temporary)/'export'
            capture.write_json(data/'manifest.json', {'scene_ids': ['scene_0000']})
            capture.write_json(data/'scene_0000/spec.json', cfg)
            for name, states in observations(cfg['branches'][:-1]).items():
                folder = data/'scene_0000/branches'/name
                capture.write_json(folder/'meta.json', {'status': 'complete', 'branch': name,
                    'vehicle_roles': cfg['vehicle_roles'], 'actor_ids': [12, 34], 'collisions': [],
                    'spec_sha256': cfg['spec_sha256'], 'code_sha256': cfg['code_sha256']})
                np.savez_compressed(folder/'states.npz', **states)
                indices = np.arange(0, 121, 10)
                np.savez_compressed(folder/'samples.npz', physics_indices=indices,
                    time_s=states['time_s'][indices], frame_id=states['frame_id'][indices],
                    intrinsics=np.broadcast_to(np.eye(3), (6, 3, 3)),
                    world_to_camera=np.broadcast_to(np.eye(4), (13, 6, 4, 4)))
                active = 'maintain' if name == 'maintain_repeat' else name
                capture.write_json(folder/'controller_diagnostics.json', [[{'role': 'ego',
                    'branch': active, 'kind': cfg['action']['profiles'][active]['kind'],
                    'phase': 'common_history' if time < 0 else 'active'}] for time in states['time_s']])
                for camera, *_ in capture.RIG:
                    directory = folder/'camera'/camera; directory.mkdir(parents=True)
                    for index in range(13):
                        Image.new('RGB', (2, 2)).save(directory/f'{index:05d}.png')
            export_common(data, out)
            frame = json.loads((out/'scene_0000/frame.json').read_text())
            self.assertEqual(frame['model_branches'], cfg['branches'][:-1])
            self.assertEqual(frame['equivalent_motion_count'], 4)
            self.assertEqual(frame['distinct_motion_count'], 1)
            self.assertTrue(frame['motion_audit_valid'])
            self.assertNotIn('planned_model_branches', frame)
            self.assertNotIn('excluded_duplicate_branches', frame)
            self.assertNotIn('duplicate_exclusion', (out/'index.html').read_text())
            self.assertEqual(len(list(out.rglob('states_ego0.npz'))), 6)
            self.assertEqual(len(list(data.rglob('states.npz'))), 6)
            self.assertTrue((out/'scene_0000/action_selection.json').is_file())
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['scene_model_branches']['scene_0000'], cfg['branches'][:-1])

    def test_export_creates_all_branches_without_future_region_labels(self):
        from PIL import Image
        from io import BytesIO
        image=BytesIO();Image.new('RGB',(2,2)).save(image,format='PNG')
        image_bytes=image.getvalue()
        cfg={'vehicle_roles':['ego','rear'],'vehicles':[{'role':'ego'},{'role':'rear'}],'action':{}}
        names=['maintain','brake','accelerate','left','right','maintain_repeat']
        with tempfile.TemporaryDirectory() as temporary:
            data=Path(temporary)/'data';out=Path(temporary)/'review'
            capture.write_json(data/'manifest.json',{'scene_ids':['scene_0000']})
            capture.write_json(data/'scene_0000/spec.json',cfg)
            states=fixture_states();indices=np.arange(0,121,10)
            samples={'physics_indices':indices,'time_s':states['time_s'][indices],
                     'frame_id':states['frame_id'][indices],
                     'intrinsics':np.broadcast_to(np.eye(3),(6,3,3)),
                     'world_to_camera':np.broadcast_to(np.eye(4),(13,6,4,4))}
            for branch in names:
                folder=data/'scene_0000/branches'/branch;folder.mkdir(parents=True)
                capture.write_json(folder/'meta.json',{'status':'complete','vehicle_roles':['ego','rear'],
                    'actor_ids':[12,34],'collisions':[]})
                np.savez_compressed(folder/'states.npz',**states)
                np.savez_compressed(folder/'samples.npz',**samples)
                for name,*_ in capture.RIG:
                    camera=folder/'camera'/name;camera.mkdir(parents=True)
                    for index in range(13):(camera/f'{index:05d}.png').write_bytes(image_bytes)
            report=export_common(data,out)
            self.assertEqual(report['scenes'],1)
            self.assertTrue((out/'index.html').is_file())
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertFalse(manifest['reactive_labels_generated'])
            self.assertEqual(len(list(out.rglob('states_ego0.npz'))),6)
            frame=json.loads((out/'scene_0000/frame.json').read_text())
            self.assertEqual(frame['track_ids'],['scene_0000:ego','scene_0000:rear'])
            self.assertNotIn('maintain_repeat',frame['model_branches'])
            with self.assertRaises(FileExistsError):export_common(data,out)


if __name__=='__main__':
    unittest.main()
