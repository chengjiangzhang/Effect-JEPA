import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

import encounter_calibration as calibration
import measured_t0
import pipeline
import planning_stage
from test_measured_t0 import fixture,save_history


class CalibrationTests(unittest.TestCase):
    def fixture(self):
        spec,states=fixture()
        spec['encounter_calibration']={'policy_version':calibration.VERSION,'max_revisions':2,
            'max_shift_per_revision_m':8.,'targets':[{'kind':'relative_gap','role':'npc',
                'desired_gap_m':16.,'encounter_time_s':1.5,'tolerance_m':2.}]}
        spec['spec_sha256']=measured_t0._spec_digest(spec)
        return spec,states

    def test_relative_velocity_jointly_changes_gap_at_encounter(self):
        spec,states=self.fixture()
        first=measured_t0.analyze_t0(spec,states)['encounter_calibration']
        self.assertFalse(first['passed']);self.assertEqual(first['corrections'][0]['route_shift_m'],6.)
        states['velocity'][-1,1,0]=7.
        second=measured_t0.analyze_t0(spec,states)['encounter_calibration']
        self.assertTrue(second['passed'])
        self.assertEqual(second['rows'][0]['constant_speed_gap_at_encounter_m'],16.)

    def test_future_observation_is_rejected(self):
        spec,states=self.fixture();states['time_s'][-1]=.05
        with self.assertRaises(measured_t0.HistoryEvidenceError):measured_t0.analyze_t0(spec,states)

    def test_small_mismatch_is_not_a_future_response_requirement(self):
        spec,states=self.fixture();spec['encounter_calibration']['targets'][0]['desired_gap_m']=11.
        report=measured_t0.analyze_t0(spec,states)['encounter_calibration']
        self.assertTrue(report['passed']);self.assertFalse(report['future_response_used'])

    def test_observed_stopped_actor_is_not_assigned_a_free_arrival(self):
        spec,states=self.fixture();spec['geometry']['ego_conflict_s']=30.
        spec['encounter_calibration']['targets']=[{'kind':'arrival_difference','role':'npc',
            'desired_difference_s':0.,'tolerance_s':.45}]
        states['velocity'][-1,1]=0.
        report=measured_t0.analyze_t0(spec,states)['encounter_calibration']
        self.assertTrue(report['passed'])
        self.assertEqual(report['rows'][0]['classification'],'observed_waiting_no_free_speed_eta')

    def test_same_lane_gap_correction_keeps_ego_reference_fixed(self):
        from types import SimpleNamespace
        spec,_=self.fixture();route=[[0.,0.,0.,0.],[200.,0.,0.,0.]]
        spec['interaction_design']={}
        for actor,station in zip(spec['vehicles'],[100.,65.]):
            actor.update(route=route,spawn_route_s=station,initial_speed_mps=6.)
        spec['geometry']['traffic_design']['roles_and_slots']=[{'role':r,'route_lane':'main'} for r in ('ego','npc')]
        geometry=SimpleNamespace(cumulative=lambda r:np.array([0.,200.]),
            point_at_s=lambda r,s:{'position':[s,0.,0.],'rotation_deg':[0.,0.,0.]},spawn_footprints_overlap=lambda a,b:False)
        report={'corrections':[{'role':'npc','route_shift_m':8.,'basis':'relative_gap'}]}
        with patch('scene_compiler.runtime_modules',return_value=(geometry,None)):
            corrected=calibration.corrected_spec(spec,report)
        self.assertEqual(corrected['vehicles'][0]['spawn_route_s'],100.)
        self.assertEqual(corrected['vehicles'][1]['spawn_route_s'],73.)
        self.assertEqual(spec['vehicles'][1]['spawn_route_s'],65.)

    def test_no_future_runs_and_original_history_preserved_across_correction(self):
        spec,states=self.fixture();calls=[]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);scene=spec['scene_id'];initial=root/'initial_plan'/scene/'spec.json'
            pipeline.write(initial,spec);pipeline.write(root/'initial_plan/manifest.json',{'scene_ids':[scene]})
            original=initial.read_bytes()
            def run(command,log,timeout_s):
                self.assertIn('--history-only',command)
                data=Path(command[command.index('--data')+1]);out=Path(command[command.index('--out')+1])
                current=pipeline.read(data/scene/'spec.json');current_states=copy.deepcopy(states)
                shift=current['vehicles'][1]['spawn_route_s']-spec['vehicles'][1]['spawn_route_s']
                current_states['position'][:,1,0]+=shift
                current_states['transform_world_vehicle'][:,1,0,3]+=shift
                calls.append(str(data.name));save_history(out,current,current_states)
            def correct(current,report):
                result=copy.deepcopy(current);result['vehicles'][1]['spawn_route_s']+=report['corrections'][0]['route_shift_m']
                result['encounter_calibration']['layout_revisions']=[{'based_on_spec_sha256':current['spec_sha256']}]
                return result
            def write(data,specs,plan,catalog):
                result=copy.deepcopy(specs[0]);result['spec_sha256']=measured_t0._spec_digest(result)
                pipeline.write(data/scene/'spec.json',result);pipeline.write(data/'manifest.json',{'scene_ids':[scene]})
            with patch.object(pipeline,'run_command',side_effect=run),patch.object(calibration,'corrected_spec',side_effect=correct),patch('scene_compiler.write_plan',side_effect=write):
                record=planning_stage.prepare_measured_plan(root,scene,'localhost',2000,plan={},catalog=None)
                self.assertEqual(record['status'],'passed')
                self.assertEqual(record['initial_plan_directory'],'calibrated_plan_001')
                self.assertEqual(calls,['initial_plan','calibrated_plan_001'])
                self.assertEqual(initial.read_bytes(),original)
                self.assertTrue((root/'common_history_calibration_000'/scene/'branches/common_history/states.npz').is_file())
                planning_stage.verify_planning_record(root)
                # Even the discarded first prefix is bound into the evidence.
                old=root/'common_history_calibration_000'/scene/'branches/common_history/states.npz'
                old.write_bytes(old.read_bytes()+b'changed')
                with self.assertRaises(pipeline.CaptureInfrastructureError):planning_stage.verify_planning_record(root)


if __name__=='__main__':unittest.main()
