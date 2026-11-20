"""Portable offline native-evidence checks; never connects to a simulator."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import native_weather_proof as proof
import native_decal_proof
import pipeline


def text(rows):
    return ''.join(json.dumps(row)+'\n' for row in rows)


def header(pid=123):
    return {'kind':'weather_rng_start','pid':pid,'policy':proof.POLICY,'seed':0,
            'algorithm':'glibc_random_r_type3_deg31_sep3','state_bytes':128,'libc_version':'2.35',
            'libc_consumption_preserved':True,'main_thread_only':True,'target_caller':'0x5ab3d5a','test_probe':False}


def draw():
    return {'kind':'weather_draw','event_index':1,'weather_ordinal':1,'context_valid':True,
            'weather_match':True,'actual_libc_value':41,'returned_value':1804289383,'status':'isolated_weather_value',
            'node':'Rain','node_outer':'BP_Weather_C','object_class':'BP_Weather_C','script_size':3565,
            'code_offset':1584,'tid':123}


class WeatherProofTests(unittest.TestCase):
    def test_dry_or_physics_zero_draw_allowed_but_rain_requires_draws(self):
        self.assertEqual(proof.trace_audit(text([header()]),header(),False)['weather_calls'],0)
        with self.assertRaisesRegex(ValueError,'no isolated'):
            proof.trace_audit(text([header()]),header(),True)
        self.assertEqual(proof.trace_audit(text([header(),draw()]),header(),True)['weather_calls'],1)

    def test_context_ordinal_range_and_nonweather_tampering_rejected(self):
        mutations=({'context_valid':False},{'event_index':2},{'weather_ordinal':2},{'tid':321},
                   {'script_size':1},{'code_offset':99},{'returned_value':-1},
                   {'weather_match':False,'status':'unchanged_non_weather','weather_ordinal':0})
        for update in mutations:
            with self.subTest(update=update),self.assertRaises(ValueError):
                proof.trace_audit(text([header(),{**draw(),**update}]),header(),False)

    def test_nonweather_valid_passthrough_and_srand(self):
        row={**draw(),'weather_match':False,'weather_ordinal':0,'returned_value':41,
             'status':'unchanged_non_weather','node':'Other'}
        result=proof.trace_audit(text([header(),{'kind':'actual_libc_srand','seed':0},row]),header(),False)
        self.assertEqual(result['weather_calls'],0)

    def test_footer_counts_complete_lines_unknown_rows_and_cap(self):
        rows=[header(),draw(),{'kind':'weather_rng_end','truncated':False,'weather_calls':1,'selected_events':1}]
        self.assertTrue(proof.trace_audit(text(rows),header(),True)['footer_present'])
        for bad in (text(rows)[:-1],text([*rows,draw()]),text([header(),{'kind':'unexpected'}]),
                    text([header(),draw(),{**rows[-1],'weather_calls':2}]),
                    text([header(),*[{**draw(),'event_index':i+1,'weather_ordinal':i+1} for i in range(1024)]])):
            with self.subTest(size=len(bad)),self.assertRaises(ValueError):proof.trace_audit(bad,header(),False)

    def test_real_package_source_identity_and_archive_include_native_bytes(self):
        bundle=proof.package_bundle();identity=pipeline.source_identity()
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as archive:
            for name,digest in identity.items():archive.writestr(name,(pipeline.PACKAGE/name).read_bytes())
        with zipfile.ZipFile(stream) as archive:
            for name,digest in bundle['manifest']['files'].items():
                path='scripts/runtime/native/weather_rng/'+name
                self.assertEqual(identity[path],digest)
                self.assertEqual(hashlib.sha256(archive.read(path)).hexdigest(),digest)

    def synthetic_bundle(self,root):
        bundle=root/'bundle';bundle.mkdir();native=root/'carla';native.mkdir()
        value=copy.deepcopy(proof.package_bundle()['manifest'])
        for name in value['files']:
            (bundle/name).write_bytes(('nativefixture:'+name).encode());value['files'][name]=proof.sha(bundle/name)
        for name in value['native_resources']:
            f=native/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(name.encode())
            value['native_resources'][name]=proof.sha(f)
        f=bundle/'manifest.json';f.write_text(json.dumps(value))
        return bundle,native,proof.sha(f)

    def test_verifier_rejects_native_library_asset_manifest_and_libc_changes(self):
        with tempfile.TemporaryDirectory() as td:
            bundle,native,digest=self.synthetic_bundle(Path(td))
            with patch.object(proof,'BUNDLE',bundle),patch.object(proof,'MANIFEST_SHA256',digest),\
                    patch.object(proof,'libc_version',return_value='2.35'),\
                    patch.object(proof.platform,'system',return_value='Linux'),patch.object(proof.platform,'machine',return_value='x86_64'):
                self.assertEqual(proof.verify_bundle(native)['manifest_sha256'],digest)
                for target in [bundle/'libweather_rng.so',bundle/'weather_rng.c',bundle/'rand_entry.S',
                               native/'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping',
                               native/'CarlaUE4/Content/Carla/Blueprints/Weather/BP_Weather.uexp',bundle/'manifest.json']:
                    old=target.read_bytes();target.write_bytes(old+b'changed')
                    with self.subTest(file=target.name),self.assertRaises(ValueError):proof.verify_bundle(native)
                    target.write_bytes(old)
                with patch.object(proof,'libc_version',return_value='2.17'),self.assertRaisesRegex(ValueError,'libc'):
                    proof.verify_bundle(native)
                with patch.object(proof.platform,'machine',return_value='aarch64'),self.assertRaisesRegex(ValueError,'platform'):
                    proof.verify_bundle(native)

    def launch_fixture(self,root):
        bundle=proof.package_bundle();folder=root/'epoch';trace=folder/'weather_rng';trace.mkdir(parents=True)
        library=bundle['library'];manifest=bundle['manifest'];engine=root/'carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
        metadata={**bundle,'trace_directory':str(trace)}
        (folder/'weather_rng.json').write_text(json.dumps(metadata))
        launch={'environment':{'LD_PRELOAD':library,'CARLA_WEATHER_RNG_SEED':'0','CARLA_WEATHER_RNG_LOG_DIR':str(trace)}}
        (folder/'launch.json').write_text(json.dumps(launch))
        (trace/'weather-123.jsonl').write_text(text([header(),draw()]))
        if manifest.get('decal_order_fix'):
            (trace/'decal-order-123.jsonl').write_text(text([{
                'kind':'decal_order_start','policy':native_decal_proof.POLICY,'target':'0x3bf64f0',
                'entry_verified':True,'preserved_primary_order':native_decal_proof.PRIMARY,
                'stable_ties':native_decal_proof.TIES}]))
        runtime = proof.dfao_renderer.runtime_path(root/'carla', folder.parent)
        (folder/'dfao_shader_read.json').write_text(json.dumps(proof.dfao_renderer.receipt(root/'carla', runtime)))
        record={'pid':123,'start_ticks':456,'state':'S','argv':[str(runtime/proof.dfao_renderer.SHIPPING)]}
        return folder,root/'carla',record,launch

    def test_launch_proof_requires_owned_exact_pid_header_and_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            folder,native,record,launch=self.launch_fixture(Path(td))
            with patch.object(proof,'process_identity',return_value=record):
                result=proof.launch_proof(folder,native,{123:record})
                self.assertEqual(result['engine']['start_ticks'],456)
                with patch.object(proof,'process_identity',return_value={**record,'state':'R'}):
                    self.assertEqual(result,proof.launch_proof(folder,native,{123:record}))
                for change in ({'start_ticks':455},{'argv':['/unowned/CarlaUE4-Linux-Shipping']},{'state':'Z'}):
                    with patch.object(proof,'process_identity',return_value={**record,**change}),self.assertRaises(ValueError):
                        proof.launch_proof(folder,native,{123:record})
                (folder/'weather_rng/weather-123.jsonl').write_text(text([header(999)]))
                with self.assertRaises(ValueError):proof.launch_proof(folder,native,{123:record})
                (folder/'weather_rng/weather-123.jsonl').write_text(text([header(),draw()]))
                launch['environment']['LD_PRELOAD']='/wrong.so';(folder/'launch.json').write_text(json.dumps(launch))
                with self.assertRaises(ValueError):proof.launch_proof(folder,native,{123:record})

    def test_epoch_snapshot_binds_branch_and_is_independent_of_shutdown_append(self):
        with tempfile.TemporaryDirectory() as td:
            folder,native,record,_=self.launch_fixture(Path(td))
            with patch.object(proof,'process_identity',return_value=record):launch=proof.launch_proof(folder,native,{123:record})
            result=proof.epoch_proof(launch,True,{'branch':'maintain'})
            path=Path(launch['trace_path']);path.write_text(path.read_text()+json.dumps({'kind':'weather_rng_end','weather_calls':1,'selected_events':1,'truncated':False})+'\n')
            proof.verify_proof(result,{'branch':'maintain'},True)
            with self.assertRaisesRegex(ValueError,'bind'):proof.verify_proof(result,{'branch':'brake'},True)
            result['trace_snapshot']=result['trace_snapshot'].replace('1804289383','1804289384')
            with self.assertRaisesRegex(ValueError,'snapshot changed'):proof.verify_proof(result,{'branch':'maintain'},True)

    def test_saved_branch_proof_missing_or_changed_blocks_acceptance(self):
        from managed_server import ManagedServerError
        with tempfile.TemporaryDirectory() as td:
            attempt=Path(td);branch=attempt/'data/scene_0000/branches/maintain';branch.mkdir(parents=True)
            spec=attempt/'data/scene_0000/spec.json';spec.write_text('{"weather":{"precipitation":20}}')
            (branch/'meta.json').write_text('{"status":"complete"}');(branch/'states.npz').write_bytes(b'states')
            with self.assertRaises(ManagedServerError):pipeline.verify_native_branch_proofs(attempt,'scene_0000',['maintain'])
            folder,native,record,_=self.launch_fixture(attempt)
            with patch.object(proof,'process_identity',return_value=record):launch=proof.launch_proof(folder,native,{123:record})
            manager=Mock();manager.weather_proof.side_effect=lambda required,binding:proof.epoch_proof(launch,required,binding)
            pipeline.save_native_branch_proof(attempt,'scene_0000','maintain',manager)
            pipeline.verify_native_branch_proofs(attempt,'scene_0000',['maintain'])
            (branch/'states.npz').write_bytes(b'changed')
            with self.assertRaises(ManagedServerError):pipeline.verify_native_branch_proofs(attempt,'scene_0000',['maintain'])
            with self.assertRaisesRegex(ManagedServerError,'overwrite'):pipeline.save_native_branch_proof(attempt,'scene_0000','maintain',manager)


    def test_physics_and_each_rgb_epoch_are_bound_and_sealed(self):
        from managed_server import ManagedServerError
        from controllers_v2 import branch_names
        with tempfile.TemporaryDirectory() as td:
            attempt=Path(td);data=attempt/'data';scene='scene_0000'
            spec={'weather':{'precipitation':20}};names=list(branch_names(spec))
            pipeline.write(data/'manifest.json',{'scene_ids':[scene]})
            pipeline.write(data/scene/'spec.json',spec)
            physics=attempt/'physics_000'
            pipeline.write(physics/'manifest.json',{'scene_ids':[scene],'requested_scene_branches':{scene:names}})
            def branch_files(base,name):
                folder=base/scene/'branches'/name
                pipeline.write(folder/'meta.json',{'status':'complete'})
                (folder/'states.npz').write_bytes(('states:'+name).encode())
            for name in names:branch_files(physics,name)
            folder,native,record,_=self.launch_fixture(attempt)
            with patch.object(proof,'process_identity',return_value=record):launch=proof.launch_proof(folder,native,{123:record})
            manager=Mock();manager.weather_proof.side_effect=lambda required,binding:proof.epoch_proof(launch,required,binding)
            with self.assertRaises(ManagedServerError):pipeline.native_physics_proof(attempt,scene,physics)
            pipeline.native_physics_proof(attempt,scene,physics,manager)
            pipeline.native_physics_proof(attempt,scene,physics)
            self.assertFalse(manager.weather_proof.call_args.args[0])
            epochs=[]
            def restart(*args):
                epochs.append(args);return {'epoch':len(epochs)}
            manager.before_attempt.side_effect=restart
            def collect(data,reports,host,port,max_new_branches):
                self.assertEqual(max_new_branches,1);i=len(epochs)-1;name=names[i]
                branch_files(data,name)
                pipeline.write(reports/'capture/batch.json',{'status':'complete' if i==5 else 'stopped_after_branch_limit',
                    'total_jobs':6,'completed_jobs':i+1,'jobs':[{'scene':scene,'branch':name,'status':'complete'}]})
            with patch.object(pipeline,'collect_capture_pass',side_effect=collect),                    patch.object(pipeline,'audit_command',side_effect=ValueError('post-native audits reached')):
                with self.assertRaisesRegex(ValueError,'post-native audits reached'):
                    with patch('planning_stage.audit_measured_prefix', return_value={'status':'passed'}):
                        pipeline.capture_stage(attempt,physics,'127.0.0.1',2000,manager,(scene,0,0))
            self.assertEqual(len(epochs),6);self.assertEqual(manager.weather_proof.call_count,7)
            self.assertTrue(all(call.args[0] is True for call in manager.weather_proof.call_args_list[1:]))
            pipeline.verify_native_delivery_proofs(attempt)
            for name in ('physics','physics_execution','execution','action_execution','action_selection','protocol','render','equivalence'):
                pipeline.write(attempt/'reports'/f'{name}.json',{'status':'passed'})
            pipeline.write(attempt/'export/manifest.json',{})
            identity=pipeline.seal_delivery(attempt)
            inventory=pipeline.read(attempt/'reports/delivery_files.json')['files']
            self.assertEqual(len([p for p in inventory if p.startswith('reports/native_weather/')]),7)
            pipeline.verify_delivery(attempt,{'accepted_data':'data','delivery_inventory_sha256':identity})
            (attempt/'reports/native_weather/maintain.json').unlink()
            with self.assertRaises(ManagedServerError):pipeline.seal_delivery(attempt)
            with self.assertRaises(ValueError):pipeline.verify_delivery(attempt,{'accepted_data':'data','delivery_inventory_sha256':identity})

    def test_missing_proof_cannot_reach_capture_audits_or_be_recreated_on_resume(self):
        from managed_server import ManagedServerError
        with tempfile.TemporaryDirectory() as td:
            attempt=Path(td);scene='scene_0000'
            pipeline.write(attempt/'data/manifest.json',{'scene_ids':[scene]})
            pipeline.write(attempt/'data'/scene/'spec.json',{'weather':{'precipitation':0}})
            pipeline.write(attempt/'data'/scene/'branches/maintain/meta.json',{'status':'complete'})
            manager=Mock()
            with patch.object(pipeline,'collect_capture_pass') as collect,patch.object(pipeline,'audit_command') as audit:
                with self.assertRaises(ManagedServerError):
                    pipeline.capture_stage(attempt,attempt/'physics_000','127.0.0.1',2000,manager,(scene,0,0))
                collect.assert_not_called();audit.assert_not_called();manager.before_attempt.assert_not_called()


if __name__=='__main__':unittest.main()
