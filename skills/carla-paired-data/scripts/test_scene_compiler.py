"""Offline compiler contracts; --opendrive also checks real map compilation, no server."""
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from plan_grid import build_plan
from scene_compiler import Catalog, validate_selection, compile_plan, runtime_modules, SUPPORTED_TEMPLATES, build_catalog, candidate_eligibility, eligible_candidates

SKILL=Path(__file__).resolve().parents[1]
OPENDRIVE=None
if '--opendrive' in sys.argv:
    index=sys.argv.index('--opendrive')
    OPENDRIVE=Path(sys.argv[index+1])
    del sys.argv[index:index+2]


class CompilerSelectionTests(unittest.TestCase):
    def setUp(self):
        self.grid=json.loads((SKILL/'assets/grid.example.json').read_text())
        self.plan=build_plan(self.grid,36,20261028)
        self.cell=next(c for c in self.plan['cells'] if c['template']=='ego_merge_ramp')
        self.job=next(j for j in self.plan['jobs'] if j['cell_id']==self.cell['cell_id'])
        self.catalog=Catalog(Path('/unused'),{}, {'ego_merge_ramp':[
            {'candidate_id':'map-ramp-001','map':'Town04','anchor':{'junction_id':1}}]},[],{})

    def test_explicit_ego_templates_and_replaced_follower(self):
        ids={p['id'] for p in self.grid['pairs']}
        self.assertIn('ego_merge_ramp',ids)
        self.assertIn('ego_lane_change_straight',ids)
        self.assertIn('follower_response_straight',ids)
        self.assertNotIn('lead_brake_straight',ids)
        self.assertEqual({c['template'] for c in self.plan['cells']},set(SUPPORTED_TEMPLATES))
        self.assertEqual(len(self.plan['disabled_pairs']),0)
        self.assertEqual(len(SUPPORTED_TEMPLATES),12)

    def test_enabled_unimplemented_type_cannot_gain_quota(self):
        grid=copy.deepcopy(self.grid)
        pair=next(p for p in grid['pairs'] if p['id']=='pedestrian_straight')
        pair['enabled']=True
        pair['implementation_status']='unsupported'
        with self.assertRaises(ValueError):build_plan(grid,200,1)

    def test_candidate_must_belong_to_target_template(self):
        with self.assertRaises(ValueError):
            validate_selection({'candidate_id':'different-map-candidate'},self.job,self.cell,self.catalog)

    def test_model_cannot_inject_world_coordinates_or_shell_fields(self):
        for extra in ({'spawn':[1,2,3]}, {'command':'echo no'}, {'npc_style':'anything'}):
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                validate_selection(extra,self.job,self.cell,self.catalog)

    def test_model_continuous_parameters_have_bounds(self):
        for extra in ({'speed_scale':2.0},{'speed_scale':float('nan')},{'layout_seed':-1},
                      {'layout_seed':True},{'layout_seed':2**31}):
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                validate_selection(extra,self.job,self.cell,self.catalog)

    def test_selection_repeatability_and_bounded_repair(self):
        original=validate_selection(None,self.job,self.cell,self.catalog)
        self.assertEqual(original,validate_selection(None,self.job,self.cell,self.catalog))
        self.assertNotEqual(original['layout_seed'],validate_selection(None,self.job,self.cell,self.catalog,1)['layout_seed'])
        for attempt in (-1,4,True):
            with self.assertRaises(ValueError):validate_selection(None,self.job,self.cell,self.catalog,attempt)


    def test_map_candidate_exclusions_have_separate_denominator(self):
        candidates=[{'candidate_id':'good','map':'Town05'}, {'candidate_id':'bad','map':'Town07'}]
        catalog=Catalog(Path('/unused'),{}, {'cross_signalized':candidates},[],{})
        cell={'cell_id':'signal__normal','template':'cross_signalized','npc_style_parameters':{}}
        def check(candidate,*args):
            if candidate['candidate_id']=='bad':raise ValueError('mapped stopping approach too short')
        with patch('interaction_geometry.signal_crossing_plan',side_effect=check):
            eligible,report=candidate_eligibility(catalog,cell)
            self.assertEqual(eligible,[candidates[0]])
            self.assertEqual(eligible_candidates(catalog,cell),eligible)
            self.assertEqual(report['candidate_count'],2)
            self.assertEqual(report['excluded_candidates'],[{'candidate_id':'bad','map':'Town07','reason':'mapped stopping approach too short'}])


@unittest.skipUnless(OPENDRIVE,'Pass --opendrive for offline real-map integration')
class RealMapIntegrationTests(unittest.TestCase):
    def test_all_cells_attempted_and_missing_quotas_preserved_with_capture_contract(self):
        grid=json.loads((SKILL/'assets/grid.example.json').read_text())
        plan=build_plan(grid,36,20261028)
        with tempfile.TemporaryDirectory(prefix='carla-map-contract-') as tmp:
            data=Path(tmp)/'data'
            result=compile_plan(plan,data,OPENDRIVE)
            self.assertEqual(result['requested'],36)
            proposals=result['sampling_proposals']
            self.assertEqual({r['scene_id'] for r in proposals},{j['scene_id'] for j in plan['jobs']})
            compiled={r['scene_id'] for r in proposals if r['status']=='compiled'}
            self.assertEqual(result['compiled'],len(compiled))
            for job in plan['jobs']:
                if job['scene_id'] not in compiled:
                    rejected=[r for r in proposals if r['scene_id']==job['scene_id']]
                    self.assertEqual(len(rejected),4)
                    self.assertTrue(all(r['status']=='rejected' and r['reason'] for r in rejected))
            runtime_modules()
            from runtime.preflight_physics import validate_config
            manifest=json.loads((data/'manifest.json').read_text())
            self.assertEqual(manifest['planned_branch_count'],6*result['compiled'])
            templates=set()
            seen_weather=set()
            variants={v['id']: v for v in grid['weather_variants']}
            from curate_dataset import requested_base_scene_id
            narrow_priorities=set()
            for name in manifest['scene_ids']:
                cfg=json.loads((data/name/'spec.json').read_text())
                validate_config(cfg,data)
                seen_weather.add(cfg['weather_variant_id'])
                variant=variants[cfg['weather_variant_id']]
                for key, value in variant['overrides'].items():
                    self.assertEqual(cfg['weather'][key], value)
                for key, value in variant.get('camera_overrides',{}).items():
                    self.assertEqual(cfg['render']['camera_attributes'][key], value)
                self.assertEqual(cfg['vehicle_light_mode'],variant.get('vehicle_lights','default'))
                self.assertEqual(cfg['base_scene_id'],requested_base_scene_id(cfg))
                self.assertTrue(cfg['grid_spec']['location_id'].startswith('location_'))
                self.assertEqual(len(cfg['vehicles']),cfg['randomization']['participants']['vehicle_count_including_ego'])
                self.assertGreaterEqual(len(cfg['vehicles']),1)
                self.assertLessEqual(len(cfg['vehicles']),7)
                self.assertEqual(len(cfg['input_contract']['model_data_branches']),5)
                self.assertEqual(cfg['input_contract']['qa_only_branches'],['maintain_repeat'])
                self.assertEqual(cfg['action']['profile_version'],2)
                self.assertEqual(cfg['action_separation_contract']['policy_id'],'five-effective-ego-actions-v1')
                self.assertEqual((cfg['history_s'],cfg['duration_s']),(2.,4.))
                self.assertEqual(cfg['action']['profile_order'],cfg['input_contract']['model_data_branches'])
                self.assertEqual(cfg['action']['nominal_policy']['respect_signals'],True)
                self.assertTrue(cfg['action']['design_evidence']['actual_t0_recheck_required'])
                self.assertFalse(cfg['action']['design_evidence']['actual_trajectory_dedup_required'])
                self.assertTrue(cfg['action']['design_evidence']['equal_realized_trajectories_retained_in_scene'])
                if cfg['action']['conflicts']:
                    opportunity=cfg['geometry']['traffic_design']['interaction_opportunity']
                    if 'strategy_entry_opportunity' in opportunity:
                        self.assertTrue(cfg['grid_spec']['template'].startswith('pedestrian_'))
                        self.assertIn(cfg['interaction_design']['requested_phase'],('entering','crossing'))
                        strategy=opportunity['strategy_entry_opportunity']
                        self.assertTrue(strategy['passed'])
                        self.assertEqual(strategy['entry_horizon_s'],3.)
                        self.assertEqual(strategy['observation_reserve_s'],1.)
                        self.assertTrue(any(p['target_travel_m']>=strategy['distance_to_entry_m']
                                            for p in strategy['profiles']))
                        for p in strategy['profiles']:
                            self.assertEqual(p['frozen_profile'],cfg['action']['profiles'][p['branch']])
                    else:
                        self.assertLessEqual(opportunity['nominal_constant_speed_entry_after_t0_s'],4.)
                    self.assertGreaterEqual(opportunity['available_stop_distance_m'],opportunity['ideal_braking_distance_m'])
                    self.assertFalse(opportunity['outcome_filtering'])
                if cfg['grid_spec']['template']=='follower_response_straight':
                    self.assertEqual(set(cfg['action']['profiles']),{'maintain','brake_light','brake','accelerate','brake_delayed'})
                    self.assertFalse(cfg['action']['lane_changes'])
                if cfg['grid_spec']['template']=='ego_lane_change_straight':
                    self.assertFalse(any(name.endswith('_delayed') for name in cfg['action']['profiles']))
                    self.assertIn('brake', cfg['action']['profiles'])
                    for side,spec in cfg['action']['lane_changes'].items():
                        evidence=spec['evidence']
                        self.assertEqual(len(evidence['source_boundary_lane_change_permissions']),evidence['station_count'])
                        self.assertIn(evidence['intervention_permission_status'],(
                            'waypoint_and_boundary_metadata_allow',
                            'controlled_map_route_intervention_boundary_permission_not_established'))
                templates.add(cfg['grid_spec']['template'])
                if cfg['grid_spec']['template']=='ego_merge_ramp':
                    topology=cfg['geometry']['interaction_evidence']['entry_road_topology']
                    self.assertLess(len(topology['incoming_same_direction_lanes']),
                                    len(cfg['geometry']['interaction_evidence']['main_road_topology']['incoming_same_direction_lanes']))
                    self.assertIn('target_lane_vehicle',cfg['vehicle_roles'])
                if cfg['grid_spec']['template']=='merge_straight':
                    actor=next(a for a in cfg['vehicles'] if a['role']=='parallel_vehicle')
                    self.assertEqual(actor['route_maneuver']['type'],'adjacent_lane_merge')
                if cfg['grid_spec']['template'] in ('left_signalized','cross_signalized'):
                    program=cfg['traffic_signal_program']
                    self.assertEqual(set(sum((g['roles'] for g in program['groups']),[])),set(cfg['vehicle_roles']))
                    self.assertLessEqual(program['phases'][0]['start_time_s'],-2.)
                    self.assertEqual(len({x for g in program['groups'] for x in g['landmark_ids']}),2)
                    for group in program['groups']:
                        evidence=group['map_evidence']
                        self.assertEqual(evidence['landmark_type'],'1000001')
                        self.assertLess(evidence['stop_route_s_m'],evidence['signal_route_s_m'])
                        self.assertLessEqual(evidence['route_signal_distance_m'],1.2)
                    if cfg['grid_spec']['template']=='cross_signalized':
                        for phase in program['phases']:
                            self.assertLessEqual(list(phase['states'].values()).count('Green'),1)
                        p=cfg['geometry']['traffic_design']['signal_encounter_plan']
                        self.assertFalse(p['initial_entry_legality_observed'])
                        green_by_phase={'queue_release':-.1,'moving_release':-.8,
                                        'clearance_early':-1.4,'clearance_late':.45}
                        expected_green=cfg['randomization']['parameters']['npc_green_s']
                        self.assertEqual(p['npc_green_time_s'],expected_green)
                        self.assertLessEqual(p['npc_optimistic_body_entry_tick_s'],2.9+1e-8)
                        self.assertGreaterEqual(cfg['vehicles'][0]['initial_speed_mps'],2.725)
                        self.assertEqual(opportunity['common_ego_stream_spawn_shift_back_m'],0.)
                        self.assertGreaterEqual(p['npc_ideal_4s_acceleration_distance_m'],p['npc_distance_to_conflict_entry_m'])
                        self.assertTrue(cfg['action']['conflicts'][0]['wait_station_beyond_signal_line'])
                        yellow=next(ph['start_time_s'] for ph in program['phases'] if ph['states']['ego']=='Yellow')
                        red=next(ph['start_time_s'] for ph in program['phases'] if ph['states']=={'ego':'Red','npc':'Red'})
                        green=next(ph['start_time_s'] for ph in program['phases'] if ph['states']['npc']=='Green')
                        self.assertAlmostEqual(red-yellow,3.)
                        self.assertAlmostEqual(green-red,.5)
                        self.assertEqual(green,expected_green)
                    else:
                        self.assertLess(cfg['geometry']['interaction_evidence']['ego_signed_turn_deg'],-45)
                        self.assertEqual(program['phases'][0]['states'],{'ego':'Green','npc':'Green'})
                if cfg['grid_spec']['template'].startswith('pedestrian_'):
                    ped=cfg['pedestrians'][0]
                    self.assertEqual(len(cfg['pedestrians']),1)
                    phase=cfg['interaction_design']['requested_phase']
                    if phase=='curb_wait':self.assertEqual(ped['start_time_s'],0.)
                    else:
                        self.assertLess(ped['start_time_s'],0.)
                        self.assertGreaterEqual(ped['start_time_s'],-cfg['history_s']-cfg['driving_warmup_s'])
                        self.assertGreaterEqual(ped['start_time_s'],-5.)
                    self.assertTrue(ped['blueprint_id'].startswith('walker.pedestrian.'))
                    self.assertGreater(len(ped['route_points']),2)
                    self.assertEqual([p['lane_type'] for p in ped['geometry_evidence']['sidewalk_endpoints']],['Sidewalk','Sidewalk'])
                    self.assertFalse(ped['geometry_evidence']['painted_crosswalk_claimed'])
                    self.assertTrue(ped['reactive_to_ego'])
                    self.assertEqual(ped['response_policy']['kind'],'reactive_crossing_v4')
                    if phase in ('entering','crossing'):
                        entry=ped['initial_opportunity_design']['nominal_entry_sweep']
                        self.assertTrue(entry['caution_clear'])
                        self.assertLess(entry['time_s'],0.)
                    self.assertEqual(set(row['role'] for row in ped['response_policy']['vehicle_conflicts']),set(cfg['vehicle_roles']))
                    self.assertFalse(ped['blueprint_evidence']['runtime_availability_verified'])
                    self.assertIn('accelerate',cfg['action']['profiles'])
                    self.assertNotIn('nominal',cfg['action']['profiles'])
                    if cfg['grid_spec']['template']=='pedestrian_unsignalized':
                        self.assertGreaterEqual(ped['geometry_evidence']['junction_distance_ahead_m'],5)
                        self.assertLessEqual(ped['geometry_evidence']['junction_distance_ahead_m'],16)
                if cfg['grid_spec']['template']=='narrow_meeting':
                    evidence=cfg['geometry']['interaction_evidence']
                    self.assertLessEqual(evidence['total_width_max_m'],6.40001)
                    self.assertEqual(evidence['driving_lanes_per_direction'],1)
                    self.assertFalse(evidence['single_lane_bottleneck_claimed'])
                    rule=cfg['geometry']['narrow_meeting']
                    narrow_priorities.add(rule['priority_assignment'])
                    self.assertTrue(rule['priority_fixed_across_branches'])
                    self.assertEqual(cfg['action']['conflicts'][0]['ego_must_yield'],'ego' in rule['yield_roles'])
                    self.assertEqual(set(rule['yield_roles'])|set(rule['priority_roles']),set(cfg['vehicle_roles']))
                    self.assertFalse(set(rule['yield_roles'])&set(rule['priority_roles']))
            self.assertEqual(templates,set(SUPPORTED_TEMPLATES))
            self.assertEqual(narrow_priorities,{'ego_stream','oncoming_stream'})
            with self.assertRaises(FileExistsError):compile_plan(plan,data,OPENDRIVE)

    def test_wider_maps_are_not_relabelled_as_narrow_roads(self):
        catalog=build_catalog(OPENDRIVE,maps=('Town04','Town05','Town10HD'),templates=('narrow_meeting',))
        self.assertEqual(catalog.candidates['narrow_meeting'],[])

    def test_signal_and_sidewalk_evidence_resolve_to_actual_map_assets(self):
        import carla
        catalog=build_catalog(OPENDRIVE,templates=('cross_signalized','left_signalized','pedestrian_straight','pedestrian_unsignalized'))
        for template,candidates in catalog.candidates.items():
            self.assertTrue(candidates,template)
            for candidate in candidates:
                cmap=catalog.maps[candidate['map']]
                if 'signal_approaches' in candidate:
                    identifiers={str(l.id) for l in cmap.get_all_landmarks() if l.type=='1000001'}
                    for evidence in candidate['signal_approaches'].values():
                        self.assertIn(evidence['landmark_id'],identifiers)
                        waypoint=cmap.get_waypoint_xodr(evidence['road_id'],evidence['lane_id'],evidence['opendrive_s_m'])
                        self.assertIsNotNone(waypoint)
                else:
                    for endpoint in candidate['pedestrian_endpoints']:
                        wp=cmap.get_waypoint_xodr(endpoint['road_id'],endpoint['lane_id'],endpoint['opendrive_s_m'])
                        self.assertIsNotNone(wp)
                        self.assertEqual(wp.lane_type,carla.LaneType.Sidewalk)


if __name__=='__main__':unittest.main(verbosity=2)
