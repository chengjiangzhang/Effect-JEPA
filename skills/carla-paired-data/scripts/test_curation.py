#!/usr/bin/env python3
"""Synthetic, CPU-only curation tests; no simulator, model API or capture."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import curate_dataset as c


BRANCHES = ['maintain', 'brake', 'accelerate', 'stop_hold', 'yield_wait']


def record(name, x=0., outcome='moving', strength='weak', style='normal', quality=0., template='follower'):
    return {'scene_id': name, 'template': template, 'npc_style': style,
        'metric_group': template+'|1', 'dedup_group': template+'|'+style+'|1',
        'features': {'names': ['x', 'y'], 'values': [float(x), 0.], 'scale_floors': [.1, .1],
            'measured_t0_fingerprint': c.fingerprint(['actual', x]),
            'requested_spawn_fingerprint': c.fingerprint(['spawn', x])},
        'branch_plan_sha256': 'plan', 'outcome_signature': outcome,
        'effect_strength': strength, 'quality_score': quality,
        'model_branches': list(BRANCHES), 'qa_only_branches': ['maintain_repeat']}


def policy(rows, threshold=.1, prototypes=()):
    models = {}
    for row in rows:
        models.setdefault(row['metric_group'], {'feature_names': ['x', 'y'], 'center': [0., 0.],
            'scale': [1., 1.], 'distance_threshold': threshold,
            'mode': 'calibrated_candidate' if threshold else 'exact_only',
            'calibration_prototypes': list(prototypes)})
    return {'schema': c.SCHEMA, 'feature_schema': c.FEATURE_SCHEMA, 'metric_groups': models}


def fixture(root, name='scene_0000', delta=0., outcome='moving', strength='weak', passed=True):
    folder = root/'captures'/name
    state_dir = folder/'branches/maintain'; state_dir.mkdir(parents=True)
    roles = ['ego']+[f'npc_{i}' for i in range(6)]
    positions = np.array([[0., 0., 0.]]+[[20.+i*10.+delta, (-1.)**i*3.5, 0.] for i in range(6)])
    velocity = np.array([[8., 0., 0.]]+[[5., 0., 0.] for _ in range(6)])
    states = {'time_s': np.array([-2., 0., 4.]), 'position': np.stack([positions]*3),
        'velocity': np.stack([velocity]*3), 'rotation_deg': np.zeros((3,7,3)),
        'bbox_extent': np.array([[2.,1.,.75]]*7)}
    np.savez_compressed(state_dir/'states.npz', **states)
    spec = {'scene_id': name, 'map': 'Town04', 'family': 'follower', 'seed': 123,
        'vehicle_roles': roles, 'branches': BRANCHES+['maintain_repeat'],
        'action': {'profiles': {b: {'kind': b} for b in BRANCHES}},
        'grid_spec': {'template': 'follower', 'npc_style': 'normal'},
        'geometry': {'conflict_xyz': [50.,0.,0.], 'ego_conflict_s': 150.},
        'vehicles': [{'role': role, 'initial_speed_mps': float(velocity[i,0]),
            'transform': {'position': positions[i].tolist(), 'rotation_deg': [0.,0.,0.]},
            'route': [[-100.,float(positions[i,1]),0.], [150.,float(positions[i,1]),0.]]} for i,role in enumerate(roles)]}
    c.write(folder/'spec.json', spec)
    response = {'max_npc_displacement_diff_m': {'weak':.1,'medium':.8,'strong':1.5}[strength],
        'reactive_npc_count_05': int(strength != 'weak'), 'reactive_npc_count_10': int(strength == 'strong'),
        'effect_strength': strength}
    execution = {'execution_pass': passed, 'failure_reason': [] if passed else ['controller_failure'],
        'unrelated_invariance': {'passed': True}, 'response_metrics': response,
        'branches': {b: {'execution_pass': True, 'failure_reason': [],
            'observed_outcome': {'ego': [outcome], 'npcs': {role: {'events':[outcome],
                'policy_reasons':['leader_following']} for role in roles[1:]}, 'collision': outcome == 'collision'}} for b in BRANCHES}}
    reports = root/'reports'/name; c.write(reports/'execution.json', {'scenes': {name:execution}})
    entry = {'scene_id': name, 'data': 'captures', 'reports': 'reports/'+name, 'cell_id':'follower.normal',
             'model_branches': BRANCHES, 'qa_only_branches':['maintain_repeat']}
    return entry, spec, states, execution


class FeaturesTest(unittest.TestCase):
    def test_real_coordinate_order_and_rigid_rotation_invariance(self):
        with tempfile.TemporaryDirectory() as d:
            _, spec, states, _ = fixture(Path(d))
            before = c.scene_features(spec, states)
            rotated = deepcopy(states); other = deepcopy(spec)
            rotation = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
            for field in ('position','velocity'):
                rotated[field] = rotated[field] @ rotation.T
            rotated['rotation_deg'][:,:,2] = 90.
            other['geometry']['conflict_xyz'] = [0.,50.,0.]
            for car in other['vehicles']:
                car['route'] = (np.array(car['route'])@rotation.T).tolist()
                car['transform']['position'] = (np.array(car['transform']['position'])@rotation.T).tolist()
                car['transform']['rotation_deg'][2] = 90.
            after = c.scene_features(other, rotated)
            np.testing.assert_allclose(before['values'], after['values'], atol=1e-10)
            self.assertNotEqual(before['measured_t0_fingerprint'], after['measured_t0_fingerprint'])

    def test_ttc_masks_and_cap_are_distinct(self):
        self.assertIsNone(c.cv_disc_ttc([10.,0.],[1.,0.],2.))
        self.assertEqual(c.cv_disc_ttc([10.,0.],[-1.,0.],2.),8.)
        self.assertEqual(c.cv_disc_ttc([1.,0.],[1.,0.],2.),0.)
        self.assertEqual(c.censored_time(None), [20.,0.])
        self.assertEqual(c.censored_time(float('inf')), [20.,0.])
        self.assertEqual(c.censored_time(25.), [20.,1.])

    def test_projection_is_not_nearest_waypoint(self):
        self.assertAlmostEqual(c.route_station([[0.,0.],[10.,0.],[10.,10.]],[7.,2.]),7.)

    def test_nonfinite_state_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _, spec, state, _ = fixture(Path(d))
            state['velocity'][1,2,0] = np.nan
            with self.assertRaisesRegex(ValueError, 'Nonfinite|nonfinite'):
                c.scene_features(spec, state)

    def test_role_names_and_order_do_not_change_exact_identity(self):
        with tempfile.TemporaryDirectory() as d:
            _, spec, state, _ = fixture(Path(d))
            original = c.scene_features(spec, state)
            permutation = [0,6,2,4,1,3,5]
            spec['vehicle_roles'] = [spec['vehicle_roles'][i] for i in permutation]
            spec['vehicles'] = [spec['vehicles'][i] for i in permutation]
            for key in ('position','velocity','rotation_deg'):
                state[key] = state[key][:,permutation]
            state['bbox_extent'] = state['bbox_extent'][permutation]
            changed = c.scene_features(spec,state)
            self.assertEqual(original['values'],changed['values'])
            self.assertEqual(original['measured_t0_fingerprint'],changed['measured_t0_fingerprint'])

    def test_spawn_fingerprint_ignores_route_padding_and_measured_t0_drift(self):
        with tempfile.TemporaryDirectory() as d:
            _, spec, state, _ = fixture(Path(d))
            before=c.scene_features(spec,state)
            other=deepcopy(spec); shifted=deepcopy(state)
            for car in other['vehicles']:
                car['route'].append([200.,car['route'][-1][1],0.])
            shifted['position'][1,:,0]+=.01
            after=c.scene_features(other,shifted)
            self.assertEqual(before['requested_spawn_fingerprint'],after['requested_spawn_fingerprint'])
            self.assertNotEqual(before['measured_t0_fingerprint'],after['measured_t0_fingerprint'])

    def test_action_signature_ignores_design_evidence_and_station_origin(self):
        a={'action':{'profiles':{'stop_hold':{'kind':'stop_hold','stop_route_s':17.,'hold_until_s':4.}},
            'conflicts':[{'id':'main','ego_entry_s':20.}], 'design_evidence':{'initial_speed':8.},
            'nominal_policy':{'desired_speed_m_s':8.},'lane_changes':{'left':{'huge_route':[1,2]}}}}
        b=deepcopy(a);b['action']['profiles']['stop_hold']['stop_route_s']=117.
        b['action']['conflicts'][0]['ego_entry_s']=120.
        b['action']['design_evidence']['initial_speed']=8.001
        b['action']['lane_changes']['left']['huge_route'].append(3)
        self.assertEqual(c.action_plan_signature(a),c.action_plan_signature(b))
        b['action']['nominal_policy']['desired_speed_m_s']=9.
        self.assertNotEqual(c.action_plan_signature(a),c.action_plan_signature(b))

    def test_base_id_is_available_before_finalize_and_ignores_weather(self):
        with tempfile.TemporaryDirectory() as d:
            _,spec,state,_=fixture(Path(d))
            base=c.requested_base_scene_id(spec)
            other=deepcopy(spec);other.pop('vehicle_roles')
            other.update(seed=999,weather_variant_id='night',weather_override={'sun_altitude_angle':-20.},
                         camera_override={'iso':800.},vehicle_light_mode='low_beam')
            self.assertEqual(base,c.requested_base_scene_id(other))
            spec['base_scene_id']=base
            self.assertEqual(c.scene_features(spec,state)['base_scene_id'],base)
            spec['base_scene_id']='base_fake'
            with self.assertRaisesRegex(ValueError,'base_scene_id'):
                c.scene_features(spec,state)

    def test_capture_conditions_use_parameters_not_names(self):
        original={'weather_variant_id':'daylight','weather':{'cloudiness':5.},
                  'render':{'camera_attributes':{'iso':100.}},'vehicle_light_mode':'default'}
        renamed=deepcopy(original);renamed['weather_variant_id']='same_daylight'
        self.assertEqual(c.capture_condition_signature(original),c.capture_condition_signature(renamed))
        for field in ('weather','exposure','lights'):
            changed=deepcopy(original)
            if field=='weather':changed['weather']['cloudiness']=80.
            elif field=='exposure':changed['render']['camera_attributes']['iso']=400.
            else:changed['vehicle_light_mode']='low_beam'
            self.assertNotEqual(c.capture_condition_signature(original),c.capture_condition_signature(changed))


class ClusterTest(unittest.TestCase):
    def test_same_outcome_dedup_whole_scene_keeps_all_branches(self):
        rows=[record('a'),record('b',.01)]
        result=c.curate_records(rows,policy(rows))
        self.assertEqual(result['representatives'],[0])
        self.assertEqual(result['records'][0]['model_branches'],BRANCHES)
        self.assertEqual(result['assignment'],{0:0,1:0})

    def test_contrasting_outcomes_protected_but_same_split(self):
        rows=[record('a'),record('b',.01,'waiting','strong')]
        result=c.curate_records(rows,policy(rows)); splits,membership,_=c.assign_splits(result,0)
        self.assertEqual(len(result['representatives']),2)
        self.assertGreater(result['protected_pair_count'],0)
        self.assertEqual(membership['a'],membership['b'])

    def test_different_action_plans_are_not_deleted(self):
        rows=[record('a'),record('b',.01)]
        rows[1]['branch_plan_sha256']='different'
        self.assertEqual(len(c.curate_records(rows,policy(rows))['representatives']),2)

    def test_cross_style_and_cross_template_exact_identity_share_split(self):
        rows=[record('a'),record('b',style='assertive'),record('c',template='cross')]
        result=c.curate_records(rows,policy(rows))
        self.assertEqual(len(result['representatives']),3)
        _,membership,_=c.assign_splits(result,0)
        self.assertEqual(len(set(membership.values())),1)

    def test_same_requested_spawn_different_realized_initial_not_deleted(self):
        rows=[record('a'),record('b',20.)]
        rows[1]['features']['requested_spawn_fingerprint']=rows[0]['features']['requested_spawn_fingerprint']
        result=c.curate_records(rows,policy(rows))
        self.assertEqual(len(result['representatives']),2)
        self.assertEqual(len(result['components']),1)

    def test_prefer_explicit_quality_then_deterministic_name(self):
        rows=[record('a'),record('b',.01,quality=2.)]
        result=c.curate_records(rows,policy(rows))
        self.assertEqual(result['representatives'],[1])
        self.assertEqual(result['assignment'][0],1)

    def test_representative_radius_blocks_transitive_collapse(self):
        rows=[record('a'),record('b',.1),record('c',.2)]
        result=c.curate_records(rows,policy(rows,.08))
        self.assertEqual(len(result['components']),1)
        self.assertEqual(result['representatives'],[0,2])

    def test_training_calibration_neighbor_forces_whole_component_train(self):
        rows=[record('a'),record('b',.01,'different')]
        prototype={'scene_id':'external', 'values':[0.,0.],
                   'measured_t0_fingerprint':'external','requested_spawn_fingerprint':'external'}
        result=c.curate_records(rows,policy(rows,prototypes=[prototype]))
        _,membership,_=c.assign_splits(result,0)
        self.assertEqual(membership,{'a':'train','b':'train'})

    def test_four_train_anchors_required_for_near_threshold(self):
        rows=[record(str(i),float(i)) for i in range(3)]
        fitted=c.fit_policy(rows)
        self.assertEqual(fitted['metric_groups']['follower|1']['mode'],'exact_only')
        rows.append(record('3',3.))
        fitted=c.fit_policy(rows)
        self.assertGreater(fitted['metric_groups']['follower|1']['distance_threshold'],0.)

    def test_calibration_provenance_records_only_supplied_anchors(self):
        rows=[record('a',0.),record('b',1.),record('c',2.),record('d',3.)]
        fitted=c.fit_policy(rows,source={'kind':'training'})
        self.assertEqual(fitted['metric_groups']['follower|1']['calibration_count'],4)
        far=record('test',1e6)
        c.curate_records(rows+[far],fitted)
        self.assertEqual(fitted['metric_groups']['follower|1']['center'],[1.5,0.])

    def test_coverage_shortfall_is_explicit_for_small_data(self):
        rows=[record('a')]
        result=c.curate_records(rows,policy(rows)); _,_,balance=c.assign_splits(result,0)
        self.assertFalse(balance['coverage_guaranteed'])
        self.assertTrue(balance['coverage_shortfalls'])

    def test_strength_coverage_when_independent_groups_available(self):
        rows=[record(f'{strength}_{i}',i*10.+offset,outcome=strength,strength=strength)
              for offset,strength in enumerate(('weak','medium','strong')) for i in range(10)]
        result=c.curate_records(rows,policy(rows)); _,_,balance=c.assign_splits(result,0)
        self.assertTrue(balance['coverage_guaranteed'])
        for split in c.SPLITS:
            self.assertEqual(set(balance['effect_strength_counts'][split]),{'weak','medium','strong'})

    def test_large_batch_uses_independent_groups_without_pair_table(self):
        rows=[record(str(i),float(i),outcome=str(i)) for i in range(1000)]
        result=c.curate_records(rows,policy(rows))
        self.assertEqual(len(result['representatives']),1000)
        self.assertEqual(len(result['components']),1000)
        self.assertEqual(result['near_pair_count'],0)

    def test_weather_variants_retained_and_share_split_despite_t0_drift(self):
        rows=[record('day'),record('rain',20.),record('night',40.)]
        for row,condition in zip(rows,('day','rain','night')):
            row['features']['requested_spawn_fingerprint']=rows[0]['features']['requested_spawn_fingerprint']
            row['capture_condition_sha256']=condition
        result=c.curate_records(rows,policy(rows))
        self.assertEqual(len(result['representatives']),3)
        _,membership,balance=c.assign_splits(result,0)
        self.assertEqual(len(set(membership.values())),1)
        self.assertEqual(sum(balance['base_scene_counts'].values()),1)
        self.assertEqual(sum(balance['capture_scene_counts'].values()),3)
        self.assertEqual(sum(sum(x.values()) for x in balance['effect_strength_counts'].values()),1)

    def test_same_weather_duplicate_still_removed_other_conditions_kept(self):
        rows=[record(name) for name in ('day','day_repeat','rain','exposure','lights')]
        for row,condition in zip(rows,('day','day','rain','high_iso','low_beam')):
            row['capture_condition_sha256']=condition
        result=c.curate_records(rows,policy(rows))
        self.assertEqual(len(result['representatives']),4)
        self.assertEqual(result['assignment'][1],0)
        self.assertEqual(len(result['components']),1)

    def test_calibration_weather_replicas_do_not_change_scale_or_count(self):
        rows=[record(f'base_{i}',float(i)) for i in range(4)]
        original=c.fit_policy(rows)['metric_groups']['follower|1']
        variants=[]
        for i in range(20):
            variant=deepcopy(rows[0]);variant['scene_id']=f'weather_{i}'
            variant['features']['values']=[0.01*i,0.]
            variant['features']['measured_t0_fingerprint']=f'weather_measured_{i}'
            variant['capture_condition_sha256']=str(i);variants.append(variant)
        fitted=c.fit_policy(rows+variants)['metric_groups']['follower|1']
        self.assertEqual(fitted['calibration_count'],4)
        self.assertEqual(fitted['calibration_capture_count'],24)
        for key in ('center','scale','distance_threshold'):
            self.assertEqual(original[key],fitted[key])
        fitted_three=c.fit_policy(rows[:3]+variants)['metric_groups']['follower|1']
        self.assertEqual(fitted_three['calibration_count'],3)
        self.assertEqual(fitted_three['mode'],'exact_only')

    def test_calibration_weather_variant_with_drift_stays_training(self):
        source=record('calibration',0.)
        target=record('night',1000.)
        target['features']['requested_spawn_fingerprint']=source['features']['requested_spawn_fingerprint']
        target['capture_condition_sha256']='night'
        result=c.curate_records([target],c.fit_policy([source]))
        _,membership,_=c.assign_splits(result,0)
        self.assertEqual(membership,{'night':'train'})


class IntegrationTest(unittest.TestCase):
    def test_evaluation_only_never_fits_and_preserves_weather_and_outcome_contrasts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); entries = []
            for name, delta, outcome, strength in [('a', 0., 'moving', 'weak'),
                ('b', 0., 'moving', 'weak'), ('c', 0., 'waiting', 'strong'),
                ('d', 20., 'collision', 'strong'), ('e', 0., 'moving', 'weak')]:
                entry, spec, _, _ = fixture(root, name, delta, outcome, strength)
                if name == 'e':
                    spec['weather'] = {'sun_altitude_angle': -25.}
                    c.write(root/'captures'/name/'spec.json', spec)
                entries.append(entry)
            manifest = root/'dataset_manifest.json'; c.write(manifest, {'accepted_scenes': entries})
            with patch.object(c, 'fit_policy', side_effect=AssertionError('must not fit evaluation')), \
                 patch.object(c, 'train_anchor', side_effect=AssertionError('must not reserve training')), \
                 patch.object(c, 'assign_splits', side_effect=AssertionError('must not make training splits')):
                report = c.curate_dataset(manifest, root/'curation', dataset_purpose='evaluation')
            self.assertEqual(report['splits'], {'evaluation': ['a', 'c', 'd', 'e']})
            self.assertEqual(report['retained_base_scene_count'], 2)
            self.assertEqual(report['retained_scene_count'], 4)
            self.assertEqual(len(report['excluded_scenes']), 1)
            self.assertTrue(all(row['model_branches'] == BRANCHES for row in report['retained_scenes']))
            self.assertEqual([p.name for p in (root/'curation/splits').iterdir()], ['evaluation.json'])
            self.assertEqual(c.read(root/'curation/curation_policy.json')['metric_groups'], {})
            audit = c.read(root/'curation/leakage_audit.json')
            self.assertFalse(audit['external_training_overlap_checked'])
            self.assertTrue(any(set(g['scene_ids']) == {'a', 'b', 'c', 'e'} for g in audit['components']))
            self.assertTrue(all(g['split'] == 'evaluation' and not g['forced_training_calibration'] for g in audit['components']))

    def test_evaluation_rejects_calibration_and_purpose_changes_in_place(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); manifest = root/'dataset_manifest.json'
            c.write(manifest, {'accepted_scenes': []})
            for kwargs in ({'policy_path': root/'policy.json'}, {'calibration_manifest': manifest}):
                with self.assertRaisesRegex(ValueError, 'accepts no fitted policy or calibration'):
                    c.curate_dataset(manifest, root/'curation', dataset_purpose='evaluation', **kwargs)
            c.curate_dataset(manifest, root/'curation', dataset_purpose='evaluation')
            with self.assertRaisesRegex(ValueError, 'purpose differs'):
                c.curate_dataset(manifest, root/'curation')

    def test_end_to_end_raw_untouched_no_branch_deletion_and_policy_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); entries=[]
            for name,delta,outcome,strength in [('a',0.,'moving','weak'),('b',0.,'moving','weak'),
                ('c',0.,'waiting','strong'),('d',20.,'collision','strong')]:
                entry,_,_,_=fixture(root,name,delta,outcome,strength); entries.append(entry)
            manifest=root/'dataset_manifest.json'; c.write(manifest,{'accepted_scenes':entries})
            raw_before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'captures').rglob('*') if p.is_file()}
            report=c.curate_dataset(manifest,root/'curation',seed=0)
            self.assertEqual(report['retained_scene_count'],3)
            self.assertEqual(len(report['excluded_scenes']),1)
            for row in report['retained_scenes']:
                self.assertEqual(row['model_branches'],BRANCHES)
            self.assertEqual(next(r for r in report['retained_scenes'] if r['scene_id']=='d')['effect_strength'],'strong')
            raw_after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'captures').rglob('*') if p.is_file()}
            self.assertEqual(raw_before,raw_after)
            again=c.curate_dataset(manifest,root/'curation2',policy_path=root/'curation/curation_policy.json',seed=0)
            self.assertEqual(report['splits'],again['splits'])
            self.assertEqual(c.read(root/'curation/leakage_audit.json')['status'],'passed')

    def test_failed_execution_or_missing_audit_is_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            a,_,_,_=fixture(root,'a',passed=False); b,_,_,_=fixture(root,'b')
            (root/'reports/b/execution.json').unlink()
            manifest=root/'dataset_manifest.json'; c.write(manifest,{'accepted_scenes':[a,b]})
            rows,excluded=c.load_scenes(manifest)
            self.assertEqual(rows,[]); self.assertEqual(len(excluded),2)

    def test_valid_early_collision_censor_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); entry,_,_,execution=fixture(root,outcome='collision',strength='strong')
            for branch in execution['branches'].values():
                branch.update(execution_pass=None,quality_status='censored')
                branch['observed_outcome']['censored_after_contact']=True
            c.write(root/'reports/scene_0000/execution.json',{'scenes':{'scene_0000':execution}})
            manifest=root/'dataset_manifest.json';c.write(manifest,{'accepted_scenes':[entry]})
            rows,excluded=c.load_scenes(manifest)
            self.assertEqual(len(rows),1); self.assertEqual(excluded,[])

    def test_existing_train_test_duplicate_is_reported_and_grouped(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            a,_,_,_=fixture(root,'a'); b,_,_,_=fixture(root,'b',outcome='waiting',strength='strong')
            a['split']='train'; b['split']='test'
            manifest=root/'dataset_manifest.json';c.write(manifest,{'accepted_scenes':[a,b]})
            report=c.curate_dataset(manifest,root/'curation')
            audit=c.read(root/'curation/leakage_audit.json')
            self.assertEqual(len(audit['source_split_violations']),1)
            self.assertEqual(len({r['split'] for r in report['retained_scenes']}),1)

    def test_stale_execution_source_or_spec_is_rejected(self):
        for field in ('spec','state'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                root=Path(d);entry,_,_,execution=fixture(root)
                folder=root/'captures/scene_0000'
                execution['spec_file_sha256']=c.file_sha(folder/'spec.json')
                execution['source_files']={'maintain':{'states.npz':c.file_sha(folder/'branches/maintain/states.npz')}}
                c.write(root/'reports/scene_0000/execution.json',{'scenes':{'scene_0000':execution}})
                target=folder/'spec.json' if field=='spec' else folder/'branches/maintain/states.npz'
                with target.open('ab') as stream:stream.write(b' ')
                manifest=root/'dataset_manifest.json';c.write(manifest,{'accepted_scenes':[entry]})
                rows,excluded=c.load_scenes(manifest)
                self.assertEqual(rows,[]);self.assertEqual(len(excluded),1)

    def test_raw_scene_output_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); entry,_,_,_=fixture(root)
            manifest=root/'dataset_manifest.json'; c.write(manifest,{'accepted_scenes':[entry]})
            with self.assertRaisesRegex(ValueError,'overlap'):
                c.curate_dataset(manifest,root/'captures/scene_0000/new')

    def test_empty_manifest_is_reported_without_fabricated_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); manifest=root/'dataset_manifest.json'; c.write(manifest,{'accepted_scenes':[]})
            report=c.curate_dataset(manifest,root/'curation')
            self.assertEqual(report['retained_scene_count'],0)
            self.assertFalse(report['balance']['coverage_guaranteed'])

    def test_captured_weather_family_is_retained_as_one_base(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);entries=[]
            for name,sun in [('day',45.),('night',-20.)]:
                entry,spec,_,_=fixture(root,name)
                spec.update(base_scene_id=c.requested_base_scene_id(spec),weather={'sun_altitude_angle':sun},
                            weather_variant_id=name)
                c.write(root/'captures'/name/'spec.json',spec);entries.append(entry)
            manifest=root/'dataset_manifest.json';c.write(manifest,{'accepted_scenes':entries})
            report=c.curate_dataset(manifest,root/'curation')
            self.assertEqual(report['retained_scene_count'],2)
            self.assertEqual(report['retained_base_scene_count'],1)
            self.assertEqual(len({r['split'] for r in report['retained_scenes']}),1)
            self.assertEqual(report['excluded_scenes'],[])
            self.assertTrue(all(r['model_branches']==BRANCHES for r in report['retained_scenes']))


if __name__=='__main__':
    unittest.main(verbosity=2)
