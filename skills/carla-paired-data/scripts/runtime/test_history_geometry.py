"""Recorded-history OBB admission tests; no simulator connection."""
from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock

import numpy as np

if __package__:
    from . import vehicle_geometry as geometry
    from . import audit_history_prefix as prefix_gate
    from . import test_history_only as history_fixture
    from . import audit_action_diversity, audit_lane_change
else:
    import vehicle_geometry as geometry
    import audit_history_prefix as prefix_gate
    import test_history_only as history_fixture
    import audit_action_diversity, audit_lane_change

import measured_t0
from test_measured_t0 import fixture, save_history


def boxes(length=41):
    transforms = np.tile(np.eye(4), (length,2,1,1))
    transforms[:,1,0,3] = 8.
    return {'time_s': np.arange(length)*.05-2., 'frame_id': np.arange(length),
            'transform_world_vehicle': transforms, 'bbox_extent': np.ones((2,3)),
            'bbox_offset': np.zeros((2,3)), 'bbox_rotation_deg': np.zeros((2,3))}


class BodyGeometryTests(unittest.TestCase):
    def test_original_public_geometry_apis_reexport_one_implementation(self):
        self.assertIs(audit_lane_change.ego_bbox, geometry.ego_bbox)
        self.assertIs(audit_lane_change.rotation, geometry.rotation)
        self.assertIs(audit_action_diversity.box_sat_gap, geometry.box_sat_gap)
        self.assertEqual(audit_action_diversity.BOX_ROUNDOFF_M, 1e-6)

    def test_separated_bodies_check_all_41_inclusive_times(self):
        report = geometry.common_history_vehicle_geometry(boxes(), ['ego','npc'])
        self.assertTrue(report['passed'])
        self.assertEqual(report['pair_observations_checked'], 41)
        self.assertEqual(report['minimum_sat_gap_m'], 6.)
        self.assertFalse(report['future_observations_examined'])

    def test_t0_overlap_rejected_even_without_callback(self):
        state = boxes(); state['transform_world_vehicle'][40,1,0,3] = 1.9
        report = geometry.common_history_vehicle_geometry(state, ['ego','npc'])
        self.assertFalse(report['passed'])
        self.assertEqual(report['interior_overlap_observation_count'], 1)
        self.assertEqual(report['interior_overlaps'][0]['time_s'], 0.)
        self.assertAlmostEqual(report['interior_overlaps'][0]['sat_gap_m'], -.1)

    def test_future_overlap_does_not_fail_common_history(self):
        state = boxes(121); state['transform_world_vehicle'][41:,1] = state['transform_world_vehicle'][41:,0]
        report = geometry.common_history_vehicle_geometry(state, ['ego','npc'])
        self.assertTrue(report['passed'])
        self.assertEqual(report['pair_observations_checked'], 41)

    def test_elevated_vehicle_with_xy_overlap_is_separated(self):
        state = boxes(); state['transform_world_vehicle'][:,1,:3,3] = [0.,0.,5.]
        self.assertEqual(geometry.common_history_vehicle_geometry(state,['a','b'])['minimum_sat_gap_m'],3.)

    def test_native_local_bbox_rotation_and_offset_are_used(self):
        state = boxes(); state['bbox_extent'][:] = [2.,.2,.2]
        state['transform_world_vehicle'][:,1,:3,3] = [0.,1.5,0.]
        self.assertTrue(geometry.common_history_vehicle_geometry(state,['a','b'])['passed'])
        state['bbox_rotation_deg'][0,2] = 90.
        self.assertFalse(geometry.common_history_vehicle_geometry(state,['a','b'])['passed'])
        state = boxes(); state['bbox_offset'][0,0] = 6.1
        self.assertFalse(geometry.common_history_vehicle_geometry(state,['a','b'])['passed'])

    def test_world_rotation_pitch_roll_and_local_rotation_compose(self):
        state = boxes(); state['bbox_offset'][0] = [.3,.4,.5]
        state['bbox_rotation_deg'][0] = [12.,23.,34.]
        state['transform_world_vehicle'][:,0,:3,:3] = geometry.rotation([15.,25.,35.])
        centers,corners = geometry.ego_bbox(state,0)
        expected_center = geometry.rotation([15.,25.,35.]) @ state['bbox_offset'][0]
        np.testing.assert_allclose(centers[0],expected_center,atol=1e-12)
        composed = geometry.rotation([15.,25.,35.]) @ geometry.rotation([12.,23.,34.])
        edges = corners[0,[4,2,1]]-corners[0,0]
        np.testing.assert_allclose(edges/2.,composed.T,atol=1e-12)
        self.assertTrue(geometry.common_history_vehicle_geometry(state,['a','b'])['passed'])

    def test_numerical_roundoff_is_not_an_inflation_margin(self):
        for separation, passed in [(2.,True),(2.-.5e-6,True),(2.-2e-6,False)]:
            with self.subTest(separation=separation):
                state=boxes();state['transform_world_vehicle'][:,1,0,3]=separation
                self.assertEqual(geometry.common_history_vehicle_geometry(state,['a','b'])['passed'],passed)

    def test_missing_box_transform_or_incomplete_prefix_fails_closed(self):
        for key in ('bbox_offset','bbox_extent','bbox_rotation_deg','transform_world_vehicle','frame_id','time_s'):
            state=boxes();state.pop(key)
            with self.subTest(key=key),self.assertRaises((ValueError,TypeError)):
                geometry.common_history_vehicle_geometry(state,['a','b'])
        for length in (1,40):
            with self.subTest(length=length),self.assertRaises(ValueError):
                geometry.common_history_vehicle_geometry(boxes(length),['a','b'])
        state=boxes();state['time_s'][40]=.05
        with self.assertRaises(ValueError):geometry.common_history_vehicle_geometry(state,['a','b'])

    def test_nonrigid_or_nonfinite_geometry_fails_closed(self):
        for kind in ('scale','reflection','nan','nonpositive_extent'):
            state=boxes()
            if kind=='scale':state['transform_world_vehicle'][:,0,0,0]=2.
            elif kind=='reflection':state['transform_world_vehicle'][:,0,0,0]=-1.
            elif kind=='nan':state['bbox_rotation_deg'][0,0]=np.nan
            else:state['bbox_extent'][0,0]=0.
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                geometry.common_history_vehicle_geometry(state,['a','b'])


class PlannerAdmissionTests(unittest.TestCase):
    def test_overlap_is_structured_setup_failure_before_planning(self):
        spec,states=fixture();states['transform_world_vehicle'][40,1]=states['transform_world_vehicle'][40,0]
        with tempfile.TemporaryDirectory() as root:
            save_history(root,spec,states)
            with mock.patch.object(measured_t0,'analyze_t0',side_effect=AssertionError('must not plan')):
                report=measured_t0.plan_from_history(spec,root)
        self.assertEqual(report['status'],'failed');self.assertIsNone(report['planned_spec'])
        failure=report['failure_reasons'][0]
        self.assertEqual(failure['code'],'common_history_vehicle_bbox_interior_overlap')
        self.assertEqual(failure['scope'],'shared_history_setup')
        self.assertEqual(failure['details']['native_collision_event_count'],0)
        self.assertEqual(failure['details']['vehicle_geometry']['interior_overlaps'][0]['time_s'],0.)
        json.dumps(report,allow_nan=False)

    def test_pipeline_persists_pair_time_and_gap_before_repair(self):
        from test_planning_stage import PlanningStageTests
        harness = PlanningStageTests()
        harness.setUp(); self.addCleanup(harness.doCleanups)
        harness.states['transform_world_vehicle'][40,1] = harness.states['transform_world_vehicle'][40,0]
        with self.assertRaisesRegex(ValueError, 'measured_t0_plan_failed'):
            harness.run_stage()
        self.assertEqual(harness.order, ['observe_shared_history'])
        self.assertFalse((harness.attempt/'data').exists())
        report = json.loads((harness.attempt/'reports/t0_planning.json').read_text())
        failure = report['failure_reasons'][0]
        self.assertEqual(failure['code'], 'common_history_vehicle_bbox_interior_overlap')
        details = failure['details']
        self.assertEqual(details['native_collision_event_count'], 0)
        first = details['vehicle_geometry']['interior_overlaps'][0]
        self.assertEqual(first['roles'], harness.spec['vehicle_roles'])
        self.assertEqual(first['time_s'], 0.)
        self.assertLess(first['sat_gap_m'], -geometry.BOX_ROUNDOFF_M)
        self.assertIn('history', details)

    def test_missing_transform_is_evidence_failure_not_repaired_geometry(self):
        spec,states=fixture();states.pop('transform_world_vehicle')
        with tempfile.TemporaryDirectory() as root:
            save_history(root,spec,states)
            with self.assertRaises(measured_t0.HistoryEvidenceError):
                measured_t0.plan_from_history(spec,root)


class ReplayAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=history_fixture.MeasuredHistoryTests()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)

    def replace_states(self,folder,change):
        states=prefix_gate.read_arrays(folder/'states.npz');change(states)
        np.savez_compressed(folder/'states.npz',**states)

    def rebind_reference(self):
        f=self.fixture;folder=f.history/'scene_0000/branches/common_history'
        f.manifest['branch_results'][0]['states_sha256']=prefix_gate.sha(folder/'states.npz')
        history_fixture.cm.write_json(f.history/'manifest.json',f.manifest)

    def test_identical_invalid_prefix_is_rejected_on_both_sides(self):
        f=self.fixture
        def overlap(s):
            s['position'][40,1]=s['position'][40,0]
            s['transform_world_vehicle'][40,1]=s['transform_world_vehicle'][40,0]
        self.replace_states(f.history/'scene_0000/branches/common_history',overlap);self.rebind_reference()
        for name in f.cfg['branches']:self.replace_states(f.data/'scene_0000/branches'/name,overlap)
        report=prefix_gate.audit(f.data,f.history)
        self.assertFalse(report['passed'])
        for row in report['branches']:
            self.assertEqual(row['physical_prefix']['status'],'passed')
            self.assertFalse(row['vehicle_geometry']['reference']['passed'])
            self.assertFalse(row['vehicle_geometry']['actual']['passed'])

    def test_future_native_collision_and_body_overlap_are_not_prefix_failures(self):
        f=self.fixture
        def future_overlap(s):
            s['position'][41:,1]=s['position'][41:,0]
            s['transform_world_vehicle'][41:,1]=s['transform_world_vehicle'][41:,0]
        for name in f.cfg['branches']:
            folder=f.data/'scene_0000/branches'/name;self.replace_states(folder,future_overlap)
            meta=prefix_gate.read_json(folder/'meta.json');meta['collisions']=[{'time_s':.05,'role':'ego','other_role':'rear'}]
            history_fixture.cm.write_json(folder/'meta.json',meta)
        report=prefix_gate.audit(f.data,f.history)
        self.assertTrue(report['passed'],report)
        self.assertTrue(all(row['vehicle_geometry']['actual']['future_observations_examined'] is False for row in report['branches']))


if __name__=='__main__':
    unittest.main(verbosity=2)
