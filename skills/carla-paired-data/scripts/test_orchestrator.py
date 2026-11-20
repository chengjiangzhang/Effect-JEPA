#!/usr/bin/env python3
"""Offline pipeline integration tests with isolated compiler/CARLA interfaces.

The production state machine, manifests, filesystem resume and model command
transport are real. Geometry compilation and CARLA execution are explicit test
doubles; these tests do not establish physical or visual acceptance.
"""
from argparse import Namespace
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import shutil
from types import SimpleNamespace
import unittest

import numpy as np
from unittest.mock import patch

import pipeline
from model_api import ModelError
from test_model_api import mock_server, response_for


MODEL_BRANCHES = ['maintain', 'brake_light', 'brake', 'accelerate_light', 'accelerate']


class FakeCatalog:
    def __init__(self):
        self.candidates = {"follower_response_straight": [{"candidate_id": "candidate_a", "map":"TownMock"}]}
        self.hashes = {"TownMock": "frozen-map-hash"}
        self.exclusions = []

    def public(self, template=None):
        return deepcopy(self.candidates["follower_response_straight"])


class Fixture:
    def __init__(self, root):
        self.root = root
        self.output = root / "output"
        self.maps = root / "maps"
        self.maps.mkdir()
        (self.maps / "TownMock.xodr").write_text("<OpenDRIVE>test-only</OpenDRIVE>")
        self.grid_path = root / "grid.json"
        pipeline.write(self.grid_path, {"schema_version": 2,
            "npc_styles": [{"id": "normal", "implementation_status": "implemented", "parameters": {}}],
            "pairs": [{"id": "follower_response_straight", "interaction": "follower_response", "topology": "straight",
                       "semantic_status": "candidate", "implementation_status": "implemented", "enabled": True,
                       "template": "follower_response_straight", "candidate_actions": ["maintain", "brake", "accelerate"]}],
            "weather_variants": [{"id": "daylight", "overrides": {}}]})
        self.compiled = []
        self.compile_failure = None
        self.physics_calls = []
        self.curation_calls = []
        self.planning_calls = []
        self.catalog = FakeCatalog()
        self.geometry = SimpleNamespace(cumulative=lambda route: np.r_[0., np.cumsum(np.linalg.norm(np.diff(np.asarray(route)[:,:2],axis=0),axis=1))],
                                        DIVERSE_NPC_MODELS={'vehicle.tesla.model3'})
        self.compiler = SimpleNamespace(build_catalog=lambda *args, **kwargs: self.catalog,
                                        compile_one=self.compile_one, write_plan=self.write_plan,
                                        validate_selection=lambda selection,job,cell,catalog,repair:{'layout_seed':job['seed']+repair,**selection},
                                        runtime_modules=lambda:(self.geometry, None))

    def args(self, **kwargs):
        values = dict(command="run", output=self.output, carla_root=None, opendrive=str(self.maps),
            maps=["TownMock"], grid=self.grid_path, count=1, seed=19, max_repairs=2,
            candidates_per_slot=2, max_candidates=None, model_config=None, host="127.0.0.1", port=2000,
            resume=False, stage="compile")
        values.update(kwargs)
        return Namespace(**values)

    def compile_one(self, job, cell, catalog, repair, selection):
        self.compiled.append({"job": deepcopy(job), "repair": repair, "selection": deepcopy(selection)})
        if self.compile_failure is not None:
            self.compile_failure(job, repair)
        cfg=self.scene_spec(job['scene_id'])
        if selection.get('scene'):
            for actor in cfg['vehicles']:
                actor.update(deepcopy(selection['scene']['actors'][actor['role']]))
            cfg['driving_warmup_s']=selection['scene']['fork_time_s']-cfg['history_s']
            cfg['action']['profiles']=deepcopy(selection['scene']['profiles'])
        return cfg

    @staticmethod
    def scene_spec(scene_id='scene_0000'):
        profiles={'maintain':{'kind':'maintain'}}
        for name, acceleration in (('brake_light',-1.2),('brake',-4.5),('accelerate_light',.6),('accelerate',1.5)):
            profiles[name]={'kind':'longitudinal','acceleration_m_s2':acceleration,'onset_s':0.,
                            'speed_increase_cap_m_s':5. if acceleration>0 else 0.}
        return {'scene_id':scene_id,'branches':MODEL_BRANCHES+['maintain_repeat'],
            'history_s':2.,'duration_s':4.,'map':'TownMock','driving_warmup_s':1.,'geometry':{},
            'controller':{'idm':{'max_acceleration':1.5,'comfortable_brake':2.,'time_headway':1.5,'minimum_gap':2.5}},
            'vehicles':[{'role':role,'spawn_route_s':station,'initial_speed_mps':8.,'desired_speed_mps':8.,
                'blueprint_id':'vehicle.tesla.model3','route':[[0.,0.,0.,0.],[200.,0.,0.,0.]],
                'transform':{'position':[station,0.,0.],'rotation_deg':[0.,0.,0.]}} for role,station in [('ego',50.),('rear',30.)]],
            'action':{'profiles':profiles}}

    @staticmethod
    def write_plan(data, configs, plan, catalog):
        pipeline.write(data / "manifest.json", {"scene_ids": [cfg["scene_id"] for cfg in configs]})
        for cfg in configs:
            pipeline.write(data / cfg["scene_id"] / "spec.json", cfg)

    def prepare_measured_plan(self, attempt, scene, host, port, **kwargs):
        # Explicit double: no simulator or physical acceptance in orchestration tests.
        self.planning_calls.append(scene)
        if not (attempt / "data").exists():
            shutil.copytree(attempt / "initial_plan", attempt / "data")
        return {"status": "passed"}

    def physics_stage(self, attempt, scene, host, port):
        self.physics_calls.append(scene)
        path = attempt / "physics_000"
        path.mkdir(exist_ok=True)
        return path

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, {"scene_compiler": self.compiler,
            "carla": SimpleNamespace(Client=lambda *args: SimpleNamespace(set_timeout=lambda x: None,
                    get_client_version=lambda: "0.9.16", get_server_version=lambda: "0.9.16",
                    get_available_maps=lambda: ['/Game/Carla/Maps/TownMock']))}))
        stack.enter_context(patch.object(pipeline, "source_identity", return_value={"test_fixture": "frozen"}))
        stack.enter_context(patch.object(pipeline, "verify_native_delivery_proofs"))
        stack.enter_context(patch.object(pipeline, "physics_stage", side_effect=self.physics_stage))
        stack.enter_context(patch("planning_stage.prepare_measured_plan", side_effect=self.prepare_measured_plan))
        stack.enter_context(patch("planning_stage.audit_measured_prefix", return_value={"status": "passed"}))
        stack.enter_context(patch("measured_t0.summarize_observed_window", return_value={"status": "offline_fixture"}))
        stack.enter_context(patch("publish_dataset.publish_dataset", return_value={"status": "offline_fixture"}))
        stack.enter_context(patch.dict(sys.modules, {'curate_dataset':SimpleNamespace(
            curate_dataset=lambda *args, **kwargs:self.curation_calls.append((args,kwargs)))}))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def model_config(self, command_script, **kwargs):
        path = self.root / "model.json"
        pipeline.write(path, {"provider": "command", "model": "offline-fixture",
            "command": [sys.executable, "-c", command_script], "max_retries": 0, "max_calls": 20,
            "timeout_s": 1, **kwargs})
        return path

    def state(self):
        return pipeline.read(self.output / "state.json")


class OrchestratorTest(unittest.TestCase):
    def setUp(self):
        # These state-machine fixtures do not execute an ELF. Real native proofs
        # and missing-proof rejection are covered in test_native_weather_proof.
        self.native_boundary = patch.object(pipeline, 'verify_native_delivery_proofs')
        self.native_boundary.start()
        self.addCleanup(self.native_boundary.stop)
        self.history_boundary = patch("planning_stage.audit_measured_prefix", return_value={"status": "passed"})
        self.history_boundary.start()
        self.addCleanup(self.history_boundary.stop)
        # Orchestration fixtures create no native states or actor records.
        # Coverage mathematics has dedicated tests and native qualification.
        self.coverage_boundary = patch('runtime.interaction_coverage.scene_report',
                                       return_value={'status':'offline_fixture'})
        self.coverage_boundary.start()
        self.addCleanup(self.coverage_boundary.stop)

    def test_compile_then_physics_resume_reuses_frozen_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            with f.patches():
                self.assertEqual(pipeline.execute(f.args()), 0)
                before = pipeline.read(f.output / "attempts/scene_0000/candidate_00/attempt_00/initial_plan/scene_0000/spec.json")
                self.assertEqual(pipeline.execute(f.args(resume=True, stage="physics")), 0)
                after = pipeline.read(f.output / "attempts/scene_0000/candidate_00/attempt_00/data/scene_0000/spec.json")
            self.assertEqual(before, after)
            self.assertEqual(len(f.compiled), 1)
            self.assertEqual(f.physics_calls, ["scene_0000"])
            self.assertEqual(f.state()["slots"]["scene_0000"]["status"], "physics_checked")
            self.assertEqual(pipeline.read(f.output / "dataset_manifest.json")["accepted_scene_count"], 0)

    def test_compile_resume_does_not_downgrade_physics_checked_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            with f.patches():
                self.assertEqual(pipeline.execute(f.args(stage="physics")), 0)
                self.assertEqual(pipeline.execute(f.args(resume=True, stage="compile")), 0)
            self.assertEqual(f.state()["slots"]["scene_0000"]["status"], "physics_checked")
            self.assertEqual(len(f.compiled), 1)
            self.assertEqual(len(f.physics_calls), 1)

    def test_three_repairs_and_candidate_budget_are_finite_and_keep_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            def fail(job, repair):
                raise ValueError("initial_overlap: ego and rear intersect before t0")
            f.compile_failure = fail
            with f.patches():
                self.assertEqual(pipeline.execute(f.args(max_repairs=3)), 3)
                self.assertEqual(pipeline.execute(f.args(max_repairs=3, resume=True)), 3)
            state = f.state()
            attempts = state["slots"]["scene_0000"]["attempts"]
            self.assertEqual(len(attempts), 8)
            self.assertEqual(len(f.compiled), 8)
            self.assertEqual(state["candidate_specs_started"], 2)
            self.assertEqual(state["slots"]["scene_0000"]["status"], "exhausted")
            self.assertEqual(state["status"], "incomplete")
            self.assertTrue(all("initial_overlap" in entry["reason"] for entry in attempts))
            self.assertTrue(all((f.output / entry["path"] / "failure.json").is_file() for entry in attempts))
            manifest = pipeline.read(f.output / "dataset_manifest.json")
            self.assertEqual(manifest["accepted_scene_count"], 0)
            self.assertEqual(next(iter(manifest["cells"].values()))["exhausted"], 1)

    def test_model_selection_real_command_and_resume_do_not_duplicate_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            model = f.model_config("import json,sys; e=json.load(sys.stdin); r=e['request']; print(json.dumps(r['base_spec'] if 'base_spec' in r else {'candidate_id':r['catalog_candidates'][0]['candidate_id'],'layout_seed':12,'speed_scale':1.0,'weather_id':'daylight'}))")
            with f.patches():
                self.assertEqual(pipeline.execute(f.args(model_config=model)), 0)
                self.assertEqual(pipeline.execute(f.args(model_config=model, resume=True, stage="physics")), 0)
            self.assertEqual(f.state()["model_calls"], 2)
            selection = pipeline.read(f.output / "attempts/scene_0000/candidate_00/attempt_00/selection.json")
            self.assertEqual(selection["selection"]["candidate_id"], "candidate_a")
            self.assertEqual(selection['model']['catalog_selection']['calls_used'], 1)
            self.assertEqual(selection['model']['scene_authoring']['calls_used'], 2)
            self.assertEqual(set(selection['selection']['scene']['actors']), {'ego','rear'})
            self.assertEqual(set(selection['selection']['scene']['profiles']), set(MODEL_BRANCHES))
            self.assertEqual(len(f.compiled), 2)

    def test_execution_failure_repairs_authored_policy_and_fork_with_specific_reason(self):
        real_physics_stage=pipeline.physics_stage
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp)); execution_checks=[]
            model=f.model_config("""import json,sys
r=json.load(sys.stdin)['request']
if 'base_spec' not in r:
 print(json.dumps({'candidate_id':'candidate_a','layout_seed':12,'speed_scale':1.0,'weather_id':'daylight'}))
else:
 scene=r['base_spec']
 if r['previous_attempts']:
  assert 'deceleration_not_executed' in r['previous_attempts'][0]['reason']
  assert r['previous_attempts'][0]['previous_selection']['scene']['actors']['rear']['initial_speed_mps']==8.
  scene['actors']['rear']['initial_speed_mps']=6.5
  scene['actors']['rear']['npc_policy']['idm']['comfortable_brake']=3.
  scene['fork_time_s']=4.
 print(json.dumps(scene))
""")
            def run(command,log,timeout_s=1800):
                script=Path(command[2]).name
                target=Path(command[command.index('--out')+1])
                if script=='preflight_physics.py':self.make_vehicle_capture(target)
                elif script=='audit_preflight_progress.py':pipeline.write(target,{'status':'passed_quick_checks'})
                elif script=='audit_execution.py':
                    execution_checks.append(target)
                    failed=len(execution_checks)==1
                    pipeline.write(target,{'status':'failed' if failed else 'passed','scenes':{'scene_0000':{
                        'execution_pass':not failed,'branches':{'brake':{'execution_pass':not failed,
                        'failure_reason':['deceleration_not_executed'] if failed else [],
                        'observed_outcome':'continued_motion' if failed else 'completed'}}}}})
                elif script=='audit_realized_actions.py':pipeline.write(target,{'status':'passed'})
                else:self.fail('Unexpected fake runtime '+script)
            with f.patches(),patch.object(pipeline,'physics_stage',side_effect=real_physics_stage),                 patch.object(pipeline,'run_command',side_effect=run):
                self.assertEqual(pipeline.execute(f.args(stage='physics',model_config=model)),0)
            state=f.state();slot=state['slots']['scene_0000']
            self.assertEqual([a['status'] for a in slot['attempts']],['failed','physics_checked'])
            self.assertIn('deceleration_not_executed',slot['attempts'][0]['reason'])
            self.assertEqual(state['model_calls'],4)
            final=f.output/slot['attempts'][-1]['path']/'data/scene_0000/spec.json'
            cfg=pipeline.read(final)
            self.assertEqual(cfg['vehicles'][1]['initial_speed_mps'],6.5)
            self.assertEqual(cfg['vehicles'][1]['npc_policy']['idm']['comfortable_brake'],3.)
            self.assertEqual(cfg['driving_warmup_s'],2.)
            self.assertEqual(cfg['branches'][:-1],MODEL_BRANCHES)

    def test_resume_rejects_changed_map_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            with f.patches():
                pipeline.prepare(f.args(command="plan"))
                (f.maps / "TownMock.xodr").write_text("<OpenDRIVE>changed map</OpenDRIVE>")
                with self.assertRaises(ValueError):
                    pipeline.prepare(f.args(command="plan", resume=True))

    def test_model_transport_retry_attempts_are_counted_and_persisted(self):
        from scene_spec import authoring_payload
        selection = {"candidate_id":"candidate_a","layout_seed":12,"speed_scale":1.0,"weather_id":"daylight"}
        authored = authoring_payload(Fixture.scene_spec())
        with tempfile.TemporaryDirectory() as tmp, mock_server([
                (429, {"error":"retry"}, {}), (200,response_for("openai_chat",json.dumps(selection)), {}),
                (200,response_for("openai_chat",json.dumps(authored)), {})]) as (base,received):
            f=Fixture(Path(tmp)); model=f.root/'model.json'
            pipeline.write(model, {'provider':'openai_chat','model':'offline-fixture','base_url':base,
                'max_retries':1,'max_calls':3,'retry_backoff_s':0,'timeout_s':1})
            with f.patches():
                self.assertEqual(pipeline.execute(f.args(model_config=model)),0)
                self.assertEqual(pipeline.execute(f.args(model_config=model,resume=True)),0)
            self.assertEqual(len(received),3)
            self.assertEqual(f.state()['model_calls'],3)

    def test_api_budget_exhaustion_stops_without_exhausting_unvisited_scene_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            model = f.model_config("print('{\"unallowed\":2}')", max_calls=1)
            with f.patches():
                try:
                    status = pipeline.execute(f.args(model_config=model, count=2))
                except ModelError:
                    status = 2
            self.assertEqual(status, 2, "An exhausted API budget is an infrastructure stop, not geometry exhaustion")
            self.assertEqual(f.state()["model_calls"], 1)
            self.assertEqual(f.state()["slots"]["scene_0001"]["status"], "pending")

    def test_invalid_model_json_enters_bounded_repair_with_failure_context(self):
        from scene_spec import authoring_payload
        selection={'candidate_id':'candidate_a','layout_seed':12,'speed_scale':1.0,'weather_id':'daylight'}
        authored=authoring_payload(Fixture.scene_spec())
        with tempfile.TemporaryDirectory() as tmp, mock_server([
                (200,response_for('openai_chat','not JSON'),{}),
                (200,response_for('openai_chat',json.dumps(selection)),{}),
                (200,response_for('openai_chat',json.dumps(authored)),{})]) as (base,received):
            f=Fixture(Path(tmp)); model=f.root/'model.json'
            pipeline.write(model,{'provider':'openai_chat','model':'offline-fixture','base_url':base,
                                  'max_retries':0,'max_calls':3,'timeout_s':1})
            with f.patches():self.assertEqual(pipeline.execute(f.args(model_config=model)),0)
            self.assertEqual(len(received),3)
            state=f.state();self.assertEqual(state['model_calls'],3)
            entries=state['slots']['scene_0000']['attempts']
            self.assertEqual([v['status'] for v in entries],['failed','compiled'])
            self.assertEqual(len(f.compiled),2)
            for index in (1,2):
                request=json.loads(received[index]['body']['messages'][1]['content'])['request']
                self.assertEqual(request['attempt_index'],1)
                self.assertIn('not valid JSON',request['previous_attempts'][0]['reason'])

    def test_resume_rejects_changed_code_or_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            with f.patches():
                pipeline.prepare(f.args(command="plan"))
                with self.assertRaises(ValueError):
                    pipeline.prepare(f.args(command="plan", resume=True, seed=20))
                with self.assertRaises(ValueError):
                    pipeline.prepare(f.args(command="plan", resume=True, dataset_purpose='evaluation'))
                with patch.object(pipeline, "source_identity", return_value={"test_fixture": "changed"}):
                    with self.assertRaises(ValueError):
                        pipeline.prepare(f.args(command="plan", resume=True))

    def test_evaluation_purpose_is_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            with f.patches():
                _, cfg, _, _ = pipeline.prepare(f.args(command='plan', dataset_purpose='evaluation'))
                self.assertEqual(cfg['dataset_purpose'], 'evaluation')
                pipeline.prepare(f.args(command='plan', resume=True, dataset_purpose='evaluation'))
                with self.assertRaises(ValueError):
                    pipeline.prepare(f.args(command='plan', resume=True))

    def test_missing_rendered_map_stops_before_freezing_or_retrying(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            with f.patches():
                with self.assertRaisesRegex(ValueError, 'missing rendered maps: Town07'):
                    pipeline.execute(f.args(stage='physics', maps=['Town07']))
            self.assertEqual(f.compiled, [])
            self.assertEqual(f.physics_calls, [])
            self.assertFalse((f.output/'pipeline_config.json').exists())
            self.assertFalse((f.output/'state.json').exists())

    def test_rendered_map_check_does_not_accept_opt_for_non_layered_map(self):
        server = SimpleNamespace(set_timeout=lambda x: None, get_client_version=lambda: '0.9.16',
            get_server_version=lambda: '0.9.16',
            get_available_maps=lambda: ['/Game/Carla/Maps/Town07_Opt'])
        with patch.dict(sys.modules, {'carla': SimpleNamespace(Client=lambda *args: server)}):
            with self.assertRaisesRegex(ValueError, 'missing rendered maps: Town07;'):
                pipeline.native_server_readiness('127.0.0.1', 2000, ['Town07'])
            result = pipeline.native_server_readiness('127.0.0.1', 2000, ['Town07_Opt'])
            self.assertEqual(result['requested_maps'], ['Town07_Opt'])

    def test_global_candidate_limit_preserves_unmet_distribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            def fail(job, repair):
                raise ValueError("invalid geometry")
            f.compile_failure = fail
            with f.patches():
                result = pipeline.execute(f.args(count=2, max_candidates=2, max_repairs=0))
            self.assertEqual(result, 3)
            self.assertEqual(len(f.compiled), 2)
            manifest = pipeline.read(f.output / "dataset_manifest.json")
            self.assertEqual(manifest["target_scene_count"], 2)
            self.assertEqual(manifest["accepted_scene_count"], 0)
            self.assertEqual(next(iter(manifest["cells"].values()))["target"], 2)

    def make_vehicle_capture(self, data):
        pipeline.write(data/'manifest.json', {'scene_ids':['scene_0000']})
        pipeline.write(data/'scene_0000/spec.json', Fixture.scene_spec())

    def test_capture_accepts_exact_equivalence_report_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp) / "attempt"
            self.make_vehicle_capture(attempt/'data')
            selection = {'status': 'passed', 'scenes': {'scene_0000': {
                'selection_valid': True, 'selected_model_branches': ['maintain', 'brake'],
                'excluded_duplicate_branches': {}, 'duplicate_clusters': [['maintain'], ['brake']],
                'tolerances': {}, 'pairwise_motion_comparisons': []}}}
            pipeline.write(attempt / 'reports/action_execution.json', selection)
            def run(command, log, timeout_s=1800):
                script = Path(command[2]).name
                if script == "audit_protocol.py":
                    pipeline.write(attempt / "reports/protocol.json", {"status": "passed"})
                elif script == "audit_render_consistency.py":
                    pipeline.write(attempt / "reports/render.json", {"status": "passed", "failures": []})
                elif script == "audit_capture_equivalence.py":
                    pipeline.write(attempt / "reports/equivalence.json", {"status": "equivalent_exact", "complete": True, "equivalent_exact": True})
                elif script == 'audit_execution.py':
                    pipeline.write(attempt / 'reports/execution.json', {'status':'passed'})
                elif script == 'audit_realized_actions.py':
                    from runtime.audit_realized_actions import normalize_motion_report
                    pipeline.write(attempt / 'reports/action_selection.json', {'status': 'passed', 'scenes': {
                        name: normalize_motion_report(row) for name, row in selection['scenes'].items()}})
                elif script == "export_common.py":
                    pipeline.write(attempt / "export/manifest.json", {"schema": "paired-common-frame-v1",
                        "source_data": "../data", "scene_ids": ["scene_0000"], "raw_data_modified": False,
                        "scene_model_branches": {"scene_0000": ["maintain", "brake"]},
                        "reactive_labels_generated": False, "qa_only_branches": ["maintain_repeat"], "review": "index.html"})
                    (attempt / "export/index.html").write_text("<html>offline export fixture</html>")
            with patch.object(pipeline, "run_command", side_effect=run):
                self.assertEqual(pipeline.capture_stage(attempt, Path(tmp) / "physics", "127.0.0.1", 2000), attempt / "data")

    def test_failed_render_report_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp) / "attempt"
            self.make_vehicle_capture(attempt/'data')
            def run(command, log, timeout_s=1800):
                if Path(command[2]).name == "audit_protocol.py":
                    pipeline.write(attempt / "reports/protocol.json", {"status": "passed"})
                elif Path(command[2]).name == "audit_render_consistency.py":
                    pipeline.write(attempt / "reports/render.json", {"status": "failed", "failures": ["history shadow mismatch"]})
            with patch.object(pipeline, "run_command", side_effect=run), self.assertRaises(ValueError):
                pipeline.capture_stage(attempt, Path(tmp) / "physics", "127.0.0.1", 2000)

    def test_nonzero_render_audit_preserves_specific_failure_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp) / "attempt"
            self.make_vehicle_capture(attempt/'data')
            def run(command, log, timeout_s=1800):
                if Path(command[2]).name == "audit_protocol.py":
                    pipeline.write(attempt / "reports/protocol.json", {"status": "passed"})
                elif Path(command[2]).name == "audit_render_consistency.py":
                    pipeline.write(attempt / "reports/render.json", {"status": "failed", "failures": ["history shadow mismatch"]})
                    raise RuntimeError("audit_render_consistency.py exited 2")
            with patch.object(pipeline, "run_command", side_effect=run), self.assertRaises((ValueError, RuntimeError)) as ctx:
                pipeline.capture_stage(attempt, Path(tmp) / "physics", "127.0.0.1", 2000)
            self.assertIn("history shadow mismatch", str(ctx.exception))

    def test_large_capture_manifest_does_not_hide_render_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp) / "attempt"
            (attempt / "data").mkdir(parents=True)
            def run(command, log, timeout_s=1800):
                pipeline.write(attempt / "reports/capture/batch.json", {
                    "status": "failed", "jobs": [{"irrelevant_log": "x" * 20000}],
                    "error": "render consistency failed"})
                pipeline.write(attempt / "reports/capture/render_consistency/scene_0000.json", {
                    "status": "failed", "failures": [{"stage": "rgb_pair", "branch": "right",
                    "reason": "specific history shadow mismatch"}]})
                raise RuntimeError("batch exited 1")
            with patch.object(pipeline, "run_command", side_effect=run), self.assertRaises(ValueError) as ctx:
                pipeline.capture_stage(attempt, Path(tmp) / "physics", "127.0.0.1", 2000)
            self.assertIn("specific history shadow mismatch", str(ctx.exception))
            self.assertLess(len(str(ctx.exception)), 2000)

    def test_resume_detects_changed_future_png_and_missing_export(self):
        for altered in ('data/scene_0000/future.png', 'export/states_ego0.npz'):
            with self.subTest(altered=altered), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(Path(tmp))
                def capture(attempt, physics, host, port):
                    (attempt / 'data/scene_0000/future.png').write_bytes(b'raw')
                    (attempt / 'export').mkdir()
                    (attempt / 'export/states_ego0.npz').write_bytes(b'arrays')
                    for name in ('physics', 'physics_execution', 'action_execution', 'protocol', 'render', 'equivalence'):
                        pipeline.write(attempt / 'reports' / (name + '.json'), {'status': 'passed'})
                    pipeline.write(attempt / 'reports/action_selection.json', {'status': 'passed', 'scenes': {
                        'scene_0000': {'model_branches': MODEL_BRANCHES, 'motion_audit_valid': True,
                                       'distinct_motion_count': 4, 'equivalent_motion_count': 1}}})
                    pipeline.write(attempt / 'reports/execution.json', {'status':'passed','scenes':{
                        'scene_0000':{'execution_pass':True,'response_metrics':{
                            'max_npc_displacement_diff_m':0.,'reactive_npc_count_05':0,
                            'reactive_npc_count_10':0,'effect_strength':'weak'}}}})
                    return attempt / 'data'
                with f.patches(), patch.object(pipeline, 'capture_stage', side_effect=capture):
                    self.assertEqual(pipeline.execute(f.args(stage='all')), 0)
                    slot = f.state()['slots']['scene_0000']
                    self.assertEqual(slot['model_branches'], MODEL_BRANCHES)
                    self.assertEqual(slot['equivalent_motion_count'], 1)
                    target = (f.output / slot['accepted_data']).parent / altered
                    if altered.endswith('.png'):
                        target.write_bytes(b'bad')  # Same size: must verify content.
                    else:
                        target.unlink()
                    with self.assertRaisesRegex(ValueError, 'Delivery file is missing or changed'):
                        pipeline.execute(f.args(stage='all', resume=True))
                self.assertEqual(f.state()['status'], 'failed_delivery_integrity')
                self.assertEqual(f.state()['slots']['scene_0000']['status'], 'invalid_delivery')
                self.assertEqual(pipeline.read(f.output / 'dataset_manifest.json')['accepted_scene_count'], 0)

    def test_weak_response_is_accepted_without_deleting_paired_branches(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp))
            def capture(attempt,physics,host,port):
                (attempt/'export').mkdir()
                (attempt/'export/index.html').write_text('fixture')
                for name in ('physics','physics_execution','action_execution','protocol','render','equivalence'):
                    pipeline.write(attempt/'reports'/f'{name}.json',{'status':'passed'})
                pipeline.write(attempt/'reports/action_selection.json',{'status':'passed','scenes':{
                    'scene_0000':{'model_branches':MODEL_BRANCHES,'motion_audit_valid':True,
                    'distinct_motion_count':1,'equivalent_motion_count':4}}})
                pipeline.write(attempt/'reports/execution.json',{'status':'passed','scenes':{
                    'scene_0000':{'execution_pass':True,'censored_branches':[],
                    'response_metrics':{'max_npc_displacement_diff_m':0.,'reactive_npc_count_05':0,
                        'reactive_npc_count_10':0,'effect_strength':'weak'}}}})
                return attempt/'data'
            with f.patches(),patch.object(pipeline,'capture_stage',side_effect=capture):
                self.assertEqual(pipeline.execute(f.args(stage='all')),0)
            manifest=pipeline.read(f.output/'dataset_manifest.json')
            self.assertEqual(manifest['accepted_scene_count'],1)
            slot=f.state()['slots']['scene_0000']
            self.assertTrue(slot['execution_pass'])
            self.assertEqual(slot['response_metrics']['effect_strength'],'weak')
            self.assertEqual(slot['model_branches'],MODEL_BRANCHES)
            self.assertNotIn('excluded_duplicate_branches',slot)
            self.assertNotIn('planned_model_branches',slot)
            self.assertEqual(slot['equivalent_motion_count'],4)
            self.assertEqual(len(f.curation_calls),1)

    def test_capture_execution_failure_prevents_export_and_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp)/'attempt';self.make_vehicle_capture(attempt/'data')
            scripts=[]
            def run(command,log,timeout_s=1800):
                script=Path(command[2]).name;scripts.append(script)
                if script=='collect_multicam_batch.py':return
                target=Path(command[command.index('--out')+1])
                if script=='audit_capture_equivalence.py':
                    pipeline.write(target,{'status':'equivalent_exact','complete':True,'equivalent_exact':True})
                elif script=='audit_execution.py':
                    pipeline.write(target,{'status':'failed','scenes':{'scene_0000':{'branches':{
                        'brake':{'execution_pass':False,'failure_reason':['deceleration_not_executed'],
                                 'observed_outcome':'continued_motion'}}}}})
                else:pipeline.write(target,{'status':'passed'})
            with patch.object(pipeline,'run_command',side_effect=run):
                with self.assertRaisesRegex(ValueError,'deceleration_not_executed'):
                    pipeline.capture_stage(attempt,Path(tmp)/'physics','127.0.0.1',2000)
            self.assertNotIn('export_common.py',scripts)
            self.assertNotIn('audit_realized_actions.py',scripts)

    def test_concurrent_production_cannot_share_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            f.output.mkdir()
            with (f.output / '.pipeline.lock').open('a+b') as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with f.patches(), self.assertRaisesRegex(RuntimeError, 'Another pipeline process'):
                    pipeline.execute(f.args())
            self.assertFalse((f.output / 'state.json').exists())

    def test_partial_physics_metadata_is_not_reused_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            physics = attempt / 'physics_000'
            pipeline.write(physics / 'manifest.json', {'scene_ids': ['scene_0000'],
                'requested_scene_branches': {'scene_0000': ['maintain', 'brake']}})
            pipeline.write(physics / 'scene_0000/branches/maintain/meta.json', {'status': 'complete'})
            pipeline.write(physics / 'scene_0000/branches/brake/meta.json', {'status': 'partial'})
            self.assertIsNone(pipeline.latest_physics(attempt))
            pipeline.write(physics / 'scene_0000/branches/brake/meta.json', {'status': 'complete'})
            self.assertEqual(pipeline.latest_physics(attempt), physics)


if __name__ == "__main__":
    unittest.main()
