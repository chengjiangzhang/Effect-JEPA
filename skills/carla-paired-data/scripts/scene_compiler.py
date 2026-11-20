#!/usr/bin/env python3
"""Compile balanced jobs into frozen CARLA configs using real OpenDRIVE geometry.

This module needs the CARLA Python client but never connects to a server. All
actor positions and routes come from the supplied maps. Physical and RGB
acceptance are subsequent pipeline stages, not claims made by compilation.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
RUNTIME = SCRIPT_ROOT / 'runtime'
SUPPORTED_TEMPLATES = (
    'follower_response_straight', 'ego_lane_change_straight', 'merge_straight',
    'merge_ramp', 'ego_merge_ramp', 'cross_unsignalized', 'left_unsignalized',
    'left_signalized', 'cross_signalized', 'pedestrian_straight',
    'pedestrian_unsignalized', 'narrow_meeting',
)


def runtime_modules():
    # All imported implementation is shipped inside this skill.
    if str(RUNTIME) not in sys.path:
        sys.path.insert(0, str(RUNTIME))
    import scenarios_v2 as geometry
    import controllers_v2 as controllers
    return geometry, controllers


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


@dataclass
class Catalog:
    root: Path
    maps: dict
    candidates: dict
    exclusions: list
    hashes: dict

    def public(self, cell_template=None):
        values = []
        for template, candidates in sorted(self.candidates.items()):
            if cell_template is not None and template != cell_template:
                continue
            for candidate in candidates:
                values.append({
                    'candidate_id': candidate['candidate_id'], 'template': template,
                    'map': candidate['map'], 'anchor': candidate['anchor'],
                    'allowed_lane_changes': list(candidate.get('changes', {})),
                    'geometry_evidence': candidate.get('evidence', {}),
                })
        return values


def _identify(template, candidate):
    candidate = dict(candidate)
    candidate['candidate_id'] = template + '-' + digest(
        [template, candidate['map'], candidate['anchor']])[:16]
    return candidate


def opposing_left_turn_candidates(cmap, map_name, geometry):
    """Enumerate unique mapped left-turn/through connections, without rollout.

    Legacy Town10HD anchors keep their original IDs. New anchors also identify
    the through movement's outgoing lane: distinct lane connections must not
    silently share a candidate ID. Location coverage still uses map+junction.
    """
    import carla
    junctions = {w.junction_id: w.get_junction()
                 for w in cmap.generate_waypoints(4) if w.is_junction}
    legacy = set(geometry.CITY_CROSSING_ANCHORS) if map_name == 'Town10HD' else set()
    candidates, exclusions = [], []
    for jid, junction in sorted(junctions.items()):
        connections = {}
        for start, end in sorted(junction.get_waypoints(carla.LaneType.Driving),
                                 key=lambda pair: (geometry.wp_key(pair[0]), geometry.wp_key(pair[1]))):
            topology = geometry.merge_path_topology(start, end)
            if topology is None or topology['incoming_is_junction']:
                continue
            key = (topology['incoming_road_id'], topology['incoming_lane_id'],
                   topology['outgoing_road_id'], topology['outgoing_lane_id'])
            connections.setdefault(key, []).append((start, end, topology))
        paths = []
        for key, matches in sorted(connections.items()):
            # Match city_connector's requirement of one unambiguous connection.
            if len(matches) != 1:
                exclusions.append({'map': map_name, 'junction_id': jid,
                                   'connection': list(key),
                                   'reason': 'Opposing-left-turn lane connection is ambiguous'})
                continue
            start, end, topology = matches[0]
            turn = float((end.transform.rotation.yaw-start.transform.rotation.yaw+180) % 360-180)
            if not (abs(turn) <= 18 or -120 < turn < -45):
                continue
            route = geometry.path_from_connector(start, end)
            if route is None:
                continue
            paths.append({'key': key, 'route': route, 'topology': topology,
                          'start_yaw': float(start.transform.rotation.yaw),
                          'signed_turn_deg': turn})
        for through in paths:
            if abs(through['signed_turn_deg']) > 18:
                continue
            for turning in paths:
                if not -120 < turning['signed_turn_deg'] < -45:
                    continue
                try:
                    point, through_s, turning_s, evidence = geometry.city_crossing_geometry(through, turning)
                except ValueError:
                    continue
                through_in, through_out = through['key'][:2], through['key'][2:]
                turn_in, turn_out = turning['key'][:2], turning['key'][2:]
                anchor = {'junction_id': jid, 'ego_incoming_lane': turn_in,
                          'ego_outgoing_lane': turn_out, 'opposing_incoming_lane': through_in}
                if (jid, through_in, through_out, turn_in, turn_out) not in legacy:
                    anchor['opposing_outgoing_lane'] = through_out
                candidates.append({'map': map_name, 'junction_id': jid,
                    'ego_route': turning['route'], 'npc_route': through['route'],
                    'ego_conflict_s': turning_s, 'npc_conflict_s': through_s,
                    'origin': point, 'conflict_xyz': point, 'anchor': anchor,
                    'evidence': {**{k:v for k,v in evidence.items() if k != 'npc_must_yield_to'},
                        'type': 'ego_left_turn_across_opposing_through_route',
                        'priority_policy': 'opposing NPC keeps live leader policy; ego follows selected speed action',
                        'ego_maneuver': 'left_turn',
                        'ego_signed_turn_deg': turning['signed_turn_deg'],
                        'signals': 'frozen Off for controlled unprotected interaction'}})
    return candidates, exclusions


def build_catalog(opendrive_root, maps=('Town04', 'Town05', 'Town10HD', 'Town07'), templates=None):
    geometry, _ = runtime_modules()
    import interaction_geometry as interactions
    import carla
    root = Path(opendrive_root).resolve()
    templates = set(SUPPORTED_TEMPLATES if templates is None else templates)
    unknown = templates - set(SUPPORTED_TEMPLATES)
    if unknown:
        raise ValueError(f'Unsupported scene templates: {sorted(unknown)}')
    map_objects, hashes, exclusions = {}, {}, []
    for name in maps:
        if not isinstance(name, str) or Path(name).name != name or name.endswith('.xodr'):
            raise ValueError('Map names must be simple CARLA map names, without paths/extensions')
        source = root / (name + '.xodr')
        raw = source.read_bytes()
        map_objects[name] = carla.Map(name, raw.decode())
        hashes[name] = hashlib.sha256(raw).hexdigest()
    pools = {template: [] for template in sorted(templates)}
    road_templates = templates & {'follower_response_straight', 'ego_lane_change_straight', 'merge_straight'}
    for name, cmap in map_objects.items():
        if road_templates:
            roads, rejected = geometry.diverse_road_candidates(cmap, name)
            exclusions.extend(rejected)
            for road in roads:
                for template in road_templates:
                    if template != 'follower_response_straight' and not road['changes']:
                        continue
                    pools[template].append(_identify(template, road))
        if templates & {'merge_ramp', 'ego_merge_ramp'}:
            for item in geometry.junction_candidates(cmap, name, 'merging'):
                try:
                    point, evidence = geometry.physical_merge_join(item['ego_route'], item['npc_route'])
                    main = item['main_road_topology']; entry = item['entry_road_topology']
                    # A narrower independent feeder joins a continuous wider road.
                    # Connector angles alone cannot establish a ramp merge.
                    if len(main['incoming_same_direction_lanes']) <= len(entry['incoming_same_direction_lanes']):
                        raise ValueError('Receiving road is not wider than the independent feeder')
                    candidate = dict(item, anchor={'junction_id': item['junction_id'],
                        'main_incoming_road': main['incoming_road_id'],
                        'entry_incoming_road': entry['incoming_road_id'],
                        'receiving_road_lane': item['outgoing_lane']},
                        origin=point, evidence={**evidence, 'main_road_topology': main,
                            'entry_road_topology': entry, 'topology_definition':
                            'independent narrower feeder converges onto a lane of a wider through road; highway classification is not inferred'})
                    candidate['ego_conflict_s'] = geometry.city_project_s(item['ego_route'], point)
                    candidate['npc_conflict_s'] = geometry.city_project_s(item['npc_route'], point)
                    for template in templates & {'merge_ramp', 'ego_merge_ramp'}:
                        pools[template].append(_identify(template, candidate))
                except ValueError as exc:
                    exclusions.append({'map': name, 'junction_id': item['junction_id'], 'reason': str(exc)})
        if templates & {'cross_unsignalized', 'cross_signalized'}:
            for item in geometry.junction_candidates(cmap, name, 'unprotected_intersection'):
                candidate = dict(item, anchor={'junction_id': item['junction_id'],
                    'ego_connector_road_id': item['ego_connector_road_id'],
                    'npc_connector_road_id': item['npc_connector_road_id']},
                    origin=item['conflict_xyz'], evidence={'type': 'crossing_driving_lane_connectors',
                        'signals': 'runtime freezes traffic lights Off; this is a controlled unprotected policy, not a claim that the map has no signal fixtures'})
                if 'cross_unsignalized' in templates:
                    pools['cross_unsignalized'].append(_identify('cross_unsignalized', candidate))
                if 'cross_signalized' in templates:
                    try:
                        signaled = interactions.signal_candidate(cmap, candidate, geometry)
                        pools['cross_signalized'].append(_identify('cross_signalized', signaled))
                    except ValueError as exc:
                        exclusions.append({'map': name, 'template': 'cross_signalized',
                                           'junction_id': item['junction_id'], 'reason': str(exc)})
        for template in templates & {'pedestrian_straight', 'pedestrian_unsignalized', 'narrow_meeting'}:
            candidates, rejected = (interactions.narrow_candidates(cmap, name, geometry)
                                    if template == 'narrow_meeting' else
                                    interactions.pedestrian_candidates(cmap, name, template, geometry))
            pools[template].extend(_identify(template, c) for c in candidates)
            exclusions.extend(rejected)
    # Actual opposing-left-turn connections on every supplied map. Ego turns;
    # the independently routed NPC continues through the same-height conflict.
    if templates & {'left_unsignalized', 'left_signalized'}:
        for name, cmap in sorted(map_objects.items()):
            candidates, rejected = opposing_left_turn_candidates(cmap, name, geometry)
            exclusions.extend(rejected)
            for candidate in candidates:
                if 'left_unsignalized' in templates:
                    pools['left_unsignalized'].append(_identify('left_unsignalized', candidate))
                if 'left_signalized' in templates:
                    try:
                        signaled = interactions.signal_candidate(cmap, candidate, geometry)
                        pools['left_signalized'].append(_identify('left_signalized', signaled))
                    except ValueError as exc:
                        exclusions.append({'map': name, 'template': 'left_signalized',
                                           'junction_id': candidate['junction_id'], 'reason': str(exc)})
    for values in pools.values():
        values.sort(key=lambda c: c['candidate_id'])
    return Catalog(root, map_objects, pools, exclusions, hashes)


def candidate_eligibility(catalog, cell):
    """Account for map feasibility separately from sampled scene attempts."""
    template = cell.get('template', cell.get('pair_id'))
    if template not in SUPPORTED_TEMPLATES:
        raise ValueError(f'Unsupported template {template!r}; no substitute topology is allowed')
    candidates = catalog.candidates.get(template, [])
    if not candidates:
        raise ValueError(f'No map-grounded candidates for {template} in supplied maps')
    exclusions = []
    if template == 'cross_signalized':
        from interaction_geometry import signal_crossing_plan
        eligible = []
        for candidate in candidates:
            try:
                signal_crossing_plan(candidate, 1., cell['npc_style_parameters'])
                eligible.append(candidate)
            except ValueError as exc:
                exclusions.append({'candidate_id':candidate['candidate_id'],
                                   'map':candidate['map'],'reason':str(exc)})
        candidates = eligible
    return candidates, {'cell_id':cell.get('cell_id'),'template':template,
        'candidate_count':len(catalog.candidates.get(template,[])),
        'eligible_candidate_ids':[c['candidate_id'] for c in candidates],
        'excluded_candidates':exclusions,
        'unit':'map candidates for this template/style, not sampled proposals or scenes',
        'basis':'existing map and nominal action-opportunity constraints; no future rollout'}


def eligible_candidates(catalog, cell):
    """Apply map/phase/action-opportunity feasibility before coverage ranking."""
    candidates, _ = candidate_eligibility(catalog,cell)
    if not candidates:
        raise ValueError('No signalized crossing satisfies the distinct-action speed/stopping and three-second target-entry opportunity bounds')
    return candidates


def validate_selection(selection, job, cell, catalog, attempt_index=0):
    if type(attempt_index) is not int or not 0 <= attempt_index <= 3:
        raise ValueError('attempt_index must be an integer in [0,3]')
    template = cell.get('template', cell.get('pair_id'))
    if template not in SUPPORTED_TEMPLATES:
        raise ValueError(f'Unsupported template {template!r}; no substitute topology is allowed')
    candidates = (eligible_candidates(catalog, cell) if selection is None or
                  isinstance(selection, dict) and 'candidate_id' not in selection else catalog.candidates.get(template, []))
    if not candidates:
        raise ValueError(f'No map-grounded candidates for {template} in supplied maps')
    seed = int(digest([job['seed'], attempt_index, 'layout'])[:8], 16) % 2**31
    default_candidate = candidates[int(digest([job['seed'], attempt_index, 'location'])[:8], 16) % len(candidates)]
    result = {'candidate_id': default_candidate['candidate_id'], 'layout_seed': seed, 'speed_scale': 1.,
              'weather_id': job.get('weather_id', 'daylight')}
    selection = {} if selection is None else selection
    if not isinstance(selection, dict) or set(selection) - (set(result) | {'scene'}):
        raise ValueError('Selection accepts catalog identifiers plus a complete scene authoring payload')
    result.update(selection)
    if result['candidate_id'] not in {c['candidate_id'] for c in candidates}:
        raise ValueError('Selected candidate_id does not belong to this template/map catalog')
    if type(result['layout_seed']) is not int or not 0 <= result['layout_seed'] < 2**31:
        raise ValueError('layout_seed must be an integer in [0,2^31)')
    if (isinstance(result['speed_scale'], bool) or not isinstance(result['speed_scale'], (int, float))
            or not .85 <= result['speed_scale'] <= 1.15):
        raise ValueError('speed_scale must lie in [.85,1.15]')
    if not isinstance(result['weather_id'], str) or not result['weather_id']:
        raise ValueError('weather_id must name a frozen weather variant')
    if job.get('weather_id') is not None and result['weather_id'] != job['weather_id']:
        raise ValueError('weather_id must match the template-balanced assignment; repairs cannot change weather')
    return result


def _profile(cfg, style):
    if not isinstance(style, dict) or not style:
        raise ValueError('Each cell must carry an implemented npc_style_parameters profile')
    required = {'max_acceleration', 'comfortable_brake', 'time_headway', 'minimum_gap',
                'desired_speed_multiplier', 'conflict_time_gap_s', 'yield_lookahead_s'}
    if set(style) != required or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                   or not math.isfinite(v) or v <= 0 for v in style.values()):
        raise ValueError('NPC profile needs the seven finite positive declared parameters')
    for key in ('max_acceleration', 'comfortable_brake', 'time_headway', 'minimum_gap'):
        cfg['controller']['idm'][key] = style[key]
    for key in ('conflict_time_gap_s', 'yield_lookahead_s'):
        cfg['controller'][key] = style[key]
    cfg['controller']['npc_style_parameters'] = dict(style)
    for actor in cfg['vehicles'][1:]:
        actor['desired_speed_mps'] *= style['desired_speed_multiplier']


def _npc_merge_route(actor, source_ego_route, geometry, pre_t0=3., merge_onset=.5):
    """Freeze legal corridors and intention; actual merging uses observed gaps."""
    source = np.asarray(actor['route'], float)
    main = np.asarray(source_ego_route, float)
    source_s = geometry.cumulative(source)
    start = actor['spawn_route_s'] + actor['initial_speed_mps'] * (pre_t0 + merge_onset)
    end = start + actor['initial_speed_mps'] * 2.5
    if end + 15 > source_s[-1]:
        raise ValueError('NPC adjacent merge does not have enough source route')
    end_index = int(np.searchsorted(source_s, end))
    route = source[:end_index+1].copy()
    main_s = geometry.cumulative(main)
    for i in range(len(route)):
        u = float(np.clip((source_s[i]-start)/(end-start), 0., 1.))
        if u == 0:
            continue
        station = geometry.city_project_s(main, source[i, :3])
        if station < 2 or station > main_s[-1]-15:
            raise ValueError('NPC merge target leaves mapped receiving route')
        target = geometry.point_at_s(main, station)
        xyz = np.asarray(target['position']) - np.array([0., 0., .4])
        weight = 10*u**3-15*u**4+6*u**5
        route[i, :3] += weight*(xyz-route[i, :3])
    final_s = geometry.city_project_s(main, route[-1, :3])
    suffix = main[main_s > final_s+0.1]
    route = np.vstack([route, suffix])
    delta = np.diff(route[:, :2], axis=0)
    route[:-1, 3] = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
    route[-1, 3] = route[-2, 3]
    if np.any(np.linalg.norm(delta, axis=1) <= 1e-5) or np.max(np.linalg.norm(delta, axis=1)) > 2.5:
        raise ValueError('NPC merge route has degenerate/discontinuous samples')
    actor['route'] = route.tolist()
    actor['spawn_route_s'] = geometry.city_project_s(route, actor['transform']['position'])
    actor['route_maneuver'] = {'type': 'adjacent_lane_merge', 'source_interval_s_m': [start, end],
                             'policy': 'fixed map-grounded route; speed responds to current live traffic'}
    # Keep the nominal route for declared conflict geometry and historical
    # readers. It is no longer the route blindly executed by this NPC.
    target = source.copy()
    valid = []
    for i, point in enumerate(source):
        station = geometry.city_project_s(main, point[:3])
        if 2. < station < main_s[-1]-2.:
            mapped=np.array([np.interp(station,main_s,main[:,k]) for k in range(3)])
            distance=float(np.linalg.norm(mapped[:2]-point[:2]))
            j=min(len(main)-2,max(0,int(np.searchsorted(main_s,station)-1)))
            own=source[min(i+1,len(source)-1),:2]-source[max(i-1,0),:2]
            receiving=main[j+1,:2]-main[j,:2]
            same_direction=float(own@receiving)/max(float(np.linalg.norm(own)*np.linalg.norm(receiving)),1e-8)>.9
            if 2.4 <= distance <= 5.5 and same_direction:
                target[i, :3] = mapped
                valid.append(i)
    groups=np.split(np.asarray(valid,dtype=int),np.flatnonzero(np.diff(valid)>1)+1) if valid else []
    valid=next((g for g in groups if len(g) and source_s[g[0]] <= start and source_s[g[-1]] >= end+15.),[])
    if not len(valid):
        raise ValueError('Dynamic NPC merge needs mapped source/target corridors through completion')
    # Mapping must describe the actual adjacent road, with no opposing lane or
    # discontinuous shortcut. Endpoint padding is never an admissible merge.
    separation = np.linalg.norm(target[valid,:2]-source[valid,:2],axis=1)
    actor['route_maneuver'].update(policy_version='observed-gap-merge-v1',
        policy='current target front/rear gaps; wait, confirm, merge, early cancel and retry',
        source_route=source.tolist(),target_route=target.tolist(),target_continuation_route=main.tolist(),
        valid_source_interval_s_m=[float(source_s[valid[0]]),float(source_s[valid[-1]])],
        earliest_start_time_s=float(merge_onset),duration_s=2.5,abort_duration_s=1.8,
        clear_confirmation_s=.2,minimum_gap_m=2.,front_headway_s=.6,rear_headway_s=.6,
        prediction_horizon_s=2.5,comfortable_brake_m_s2=3.,abort_max_progress=.4,
        lane_half_width_m=float(np.median(separation)/2.))


def _road_config(candidate, template, index, seed, rng, speed_scale, phase=None, randomization=None):
    geometry, _ = runtime_modules()
    family = 'rear_following' if template == 'follower_response_straight' else 'adjacent_parallel'
    cars, traffic, changes, unavailable, scope = geometry._diverse_traffic(candidate, family, rng, 3., 0, randomization=randomization)
    for actor in cars:
        actor['initial_speed_mps'] *= speed_scale
        actor['desired_speed_mps'] *= speed_scale
    traffic['speed_scale'] = speed_scale
    traffic['ego_initial_speed_mps'] = cars[0]['initial_speed_mps']
    lookup = {actor['role']:actor for actor in cars}
    for slot in traffic['roles_and_slots']:
        actor = lookup[slot['role']]
        slot['relative_initial_speed_mps'] = actor['initial_speed_mps']-cars[0]['initial_speed_mps']
        slot['desired_speed_mps_before_style'] = actor['desired_speed_mps']
        if 'nominal_t0_longitudinal_offset_m' in slot:
            slot['nominal_t0_longitudinal_offset_m'] = slot['gap_from_ego_station_m']+3.*slot['relative_initial_speed_mps']
    for opportunity in traffic['approach_opportunities']:
        if 'relative_initial_speed_mps' in opportunity:
            opportunity['relative_initial_speed_mps'] *= speed_scale
        if 'nominal_t0_longitudinal_offset_m' in opportunity:
            slot = next(item for item in traffic['roles_and_slots'] if item['role'] == opportunity['role'])
            opportunity['nominal_t0_longitudinal_offset_m'] = slot['nominal_t0_longitudinal_offset_m']
    if template == 'merge_straight':
        actor = next(v for v in cars if v['role'] == 'parallel_vehicle')
        offset = (randomization['parameters']['merge_onset_s'] if randomization else {'waiting_gap':1.1, 'approach_lateral':.4, 'starting_merge':-.3, 'partial_merge':-1.1}.get(phase,.5))
        _npc_merge_route(actor, candidate['route'], geometry, merge_onset=offset)
        if phase == 'partial_merge':
            # This is a longitudinal ego intervention. Its braking/history
            # approach may extend behind the crop made for ego lane changes.
            # Restore real original map samples, never extrapolate a road or
            # move the already-merging NPC back to an easier phase.
            ego = cars[0]
            ego['route'] = copy.deepcopy(candidate['route'])
            ego['spawn_route_s'] = geometry.city_project_s(ego['route'],ego['transform']['position'])
            scope = {'mode':'full_original_main_route_for_partial_npc_merge',
                     'original_cropped_scope':scope,'uses_observed_future_or_model':False,
                     'ego_lane_change_actions':False,
                     'reason':'retain real mapped space for common history and stopping before an already-started merge'}
    geom = {'map': candidate['map'], 'origin': candidate['origin'], 'conflicts': [],
        'type': template, 'selection': {'anchor': candidate['anchor'], 'outcome_based_selection': False},
        'traffic_design': traffic, 'unavailable_lane_change_directions': unavailable, 'ego_route_scope': scope}
    cfg = geometry.finish_config(family, index, seed, cars, geom, 2., 1.)
    return cfg, changes


def audit_boundary_permissions(changes, cmap):
    """Record map permissions and missing/contradictory markings separately."""
    import carla
    for side, spec in changes.items():
        evidence = spec['evidence']
        allowed = carla.LaneChange.Left if side == 'left' else carla.LaneChange.Right
        marking_permissions = []
        for key, station in zip(evidence['source_lane_keys'], evidence['source_opendrive_s_m']):
            wp = cmap.get_waypoint_xodr(int(key[0]), int(key[2]), float(station))
            if wp is None or not int(wp.lane_change) & int(allowed):
                raise ValueError(f'{side}: original OpenDRIVE station has no waypoint lane-change permission')
            marking = getattr(wp, side + '_lane_marking')
            marking_permissions.append(str(marking.lane_change))
        all_markings_allow = all(p in (str(allowed), str(carla.LaneChange.Both)) for p in marking_permissions)
        evidence['source_boundary_lane_change_permissions'] = marking_permissions
        evidence['boundary_permissions_allow_all_stations'] = all_markings_allow
        evidence['intervention_permission_status'] = (
            'waypoint_and_boundary_metadata_allow' if all_markings_allow else
            'controlled_map_route_intervention_boundary_permission_not_established')
        evidence['public_road_legal_permission_verified'] = False
        evidence['permission_api_reference'] = 'https://carla.readthedocs.io/en/0.9.16/python_api/#carla.Waypoint'


def _junction_config(candidate, template, index, seed, rng, speed_scale, style, phase=None, randomization=None):
    geometry, _ = runtime_modules()
    ego_route, other_route = candidate['ego_route'], candidate['npc_route']
    ego_conflict, other_conflict = candidate['ego_conflict_s'], candidate['npc_conflict_s']
    if template == 'ego_merge_ramp':
        ego_route, other_route = other_route, ego_route
        ego_conflict, other_conflict = other_conflict, ego_conflict
    from interaction_design import PHASES
    variant = PHASES[template].index(phase) if phase is not None else 1
    speed = float(rng.uniform(4.1, 7.0))*speed_scale
    other_speed = float(rng.uniform(3.5, 7.5))*speed_scale
    ego_eta = float(rng.uniform(2., 2.8))
    other_eta = ego_eta + (1.2, .1, -.9, -.4)[variant] + float(rng.uniform(-.2,.2))
    parameters = randomization['parameters'] if randomization else {}
    if randomization:
        speed = parameters['ego_initial_speed_m_s']*speed_scale
        other_speed = parameters['other_initial_speed_m_s']*speed_scale
        ego_eta = parameters['ego_nominal_arrival_s']
        other_eta = ego_eta+parameters['other_minus_ego_arrival_s']
    turn = None
    if template in ('left_signalized', 'left_unsignalized'):
        from action_design import turn_speed_plan
        turn = turn_speed_plan(ego_route, ego_conflict, ego_eta, speed)
        speed = turn['approach_m_s']
    target_count = randomization['participants']['vehicle_count_including_ego'] if randomization else 7
    ego_s = ego_conflict - speed*(3.+ego_eta)
    other_s = other_conflict - other_speed*(3.+other_eta)
    signal_opportunity = None
    desired_other = other_speed + .4
    if template == 'cross_signalized':
        from interaction_geometry import signal_crossing_plan
        signal_opportunity = signal_crossing_plan(candidate, speed_scale, style)
        speed = float(rng.uniform(min(signal_opportunity['ego_initial_speed_candidates_m_s']),max(signal_opportunity['ego_initial_speed_candidates_m_s']))) if randomization else float(rng.choice(signal_opportunity['ego_initial_speed_candidates_m_s']))
        ego_s, other_s = signal_opportunity['ego_spawn_s'], signal_opportunity['npc_spawn_s']
        other_speed = 0.
        desired_other = signal_opportunity['npc_desired_speed_before_style_m_s']
        ego_eta = (ego_conflict - ego_s) / speed - 3.
        other_eta = None
    other_role = 'target_lane_vehicle' if template == 'ego_merge_ramp' else ('opposing_vehicle' if template in ('left_unsignalized', 'left_signalized') else 'interaction_vehicle')
    cars = [geometry.vehicle('ego', ego_route, ego_s, speed, priority=0, color='195,35,30'),
            geometry.vehicle(other_role, other_route, other_s, other_speed, desired_other, priority=1)]
    opportunities = [{'role': other_role, 'nominal_conflict_after_t0_s': other_eta}]
    # Add a follower of the receiving stream for ego active merge.
    if template == 'ego_merge_ramp' and (not randomization or randomization['participants']['target_lane_rear']):
        cars.append(geometry.vehicle('target_lane_rear', other_route, other_s-float(rng.uniform(13,20)),
                                     other_speed, other_speed+.7, priority=1))
    attempts = []
    for trial in range(80):
        if len(cars) == target_count:
            break
        own = bool(rng.integers(2))
        force_following = bool(randomization and phase=='continuous_stream' and len(cars)==2)
        if force_following:
            own=False
        route, station, base_speed = (ego_route, ego_s, speed) if own else (other_route, other_s, other_speed)
        spacing = (23., 19., 31., 13.)[variant]
        if template == 'cross_signalized':
            spacing = (12.5,14.,15.5,17.)[variant]
        if randomization:
            spacing = parameters['queue_gap_m']
        options = [-3*spacing,-2*spacing,-spacing] if template == 'cross_signalized' else [-3*spacing,-2*spacing,-spacing,spacing,2*spacing,3*spacing]
        if force_following:
            options=[-spacing,-2*spacing]
        gap = float(rng.choice(options)+rng.uniform(-2.,2.))
        try:
            initial = max(0., base_speed+float(rng.uniform(-.3,.5)))
            desired = base_speed+float(rng.uniform(0.,1.))
            if template == 'cross_signalized' and not own:
                initial, desired = 0., desired_other
            available=float(geometry.cumulative(route)[-1])-(station+gap)
            effective_desired=desired*style['desired_speed_multiplier']
            if station+gap < 20. or available < max(initial,effective_desired)*7.+22.:
                raise ValueError('Background slot has insufficient mapped approach/future space')
            car = geometry.vehicle(f'traffic_{len(cars):02d}', route, station+gap, initial, desired)
            if any(geometry.spawn_footprints_overlap(car, old) for old in cars):
                raise ValueError('initial conservative actor footprints overlap')
            cars.append(car)
        except ValueError as exc:
            attempts.append({'slot_attempt': trial, 'gap_m': gap, 'reason': str(exc)})
    if len(cars) != target_count or any(geometry.spawn_footprints_overlap(a,b) for a,b in itertools.combinations(cars,2)):
        raise ValueError('Cannot place requested disjoint vehicles on the independent approach routes')
    for actor in cars[1:]:
        actor['blueprint_id'] = geometry.DIVERSE_NPC_MODELS[int(rng.integers(len(geometry.DIVERSE_NPC_MODELS)))]
        actor['blueprint_attributes']['color'] = geometry.DIVERSE_COLORS[int(rng.integers(len(geometry.DIVERSE_COLORS)))]
    # Entry NPC yields when ego is on main road. For ego entry/left turn, main
    # traffic retains its live leader policy; future contact is a possible outcome.
    conflicts = []
    if template in ('merge_ramp', 'cross_unsignalized'):
        conflicts.append({'roles':['ego',other_role], 'point':candidate['origin'],
                          'route_s':{'ego':ego_conflict,other_role:other_conflict},
                          'priority_role':'ego','kind':template})
    geom = {'map':candidate['map'], 'origin':candidate['origin'], 'junction_id':candidate['junction_id'],
        'type':template, 'conflicts':conflicts, 'selection':{'anchor':candidate['anchor'],'outcome_based_selection':False},
        'interaction_evidence':candidate['evidence'], 'ego_conflict_s':ego_conflict,
        'npc_conflict_s':other_conflict, 'conflict_xyz':candidate['origin'],
        'traffic_design':{'vehicle_count_including_ego':len(cars),'initial_slot_attempts':attempts,
            'approach_opportunities':opportunities,'ego_nominal_conflict_after_t0_s':ego_eta,
            'reaction_guaranteed':False,'future_outcome_filtering':False}}
    cfg = geometry.finish_config(template, index, seed, cars, geom, 2., 1.)
    if turn is not None:
        cfg['geometry']['traffic_design']['turn_speed_plan'] = turn
    if signal_opportunity is not None:
        cfg['geometry']['traffic_design']['signal_encounter_plan'] = signal_opportunity
    if template in ('left_signalized', 'cross_signalized'):
        from interaction_geometry import configure_signals
        configure_signals(cfg, candidate, template)
    return cfg, {}


def compile_one(job, cell, catalog, attempt_index=0, selection=None):
    geometry, controllers = runtime_modules()
    chosen = validate_selection(selection, job, cell, catalog, attempt_index)
    template = cell.get('template', cell.get('pair_id'))
    from interaction_design import phase_for, diversify, pedestrian_pace_for
    phase = phase_for(job, template)
    pedestrian_pace = pedestrian_pace_for(job, template, phase)
    candidate = next(c for c in catalog.candidates[template] if c['candidate_id'] == chosen['candidate_id'])
    from scene_randomization import intent, individualize, snapshot
    randomization = intent(job,cell,chosen)
    rng = np.random.default_rng(chosen['layout_seed'])
    index = int(job['scene_id'].split('_')[-1])
    if template in {'follower_response_straight','ego_lane_change_straight','merge_straight'}:
        cfg, changes = _road_config(candidate, template, index, job['seed'], rng, chosen['speed_scale'],phase,randomization)
    elif template in {'pedestrian_straight', 'pedestrian_unsignalized'}:
        from interaction_geometry import pedestrian_config
        cfg, changes = pedestrian_config(candidate, template, index, job['seed'], rng, chosen['speed_scale'], geometry,randomization=randomization)
    elif template == 'narrow_meeting':
        from interaction_geometry import narrow_config
        cfg, changes = narrow_config(candidate, index, job['seed'], rng, chosen['speed_scale'], geometry,randomization=randomization)
    else:
        cfg, changes = _junction_config(candidate, template, index, job['seed'], rng, chosen['speed_scale'], cell['npc_style_parameters'],phase,randomization)
    cfg['randomization'] = randomization
    cfg['seed'] = int(job['seed'])
    cfg['scene_id'] = job['scene_id']
    cfg['scene_profile'] = 'portable_grid_v1'
    cfg['vehicle_profile'] = 'native_v2'
    cfg['action'].pop('lateral_offset_m', None)
    if template == 'ego_lane_change_straight':
        audit_boundary_permissions(changes, catalog.maps[cfg['map']])
    diversify(cfg,template,phase,rng,geometry,pedestrian_pace=pedestrian_pace)
    from action_design import configure_actions
    configure_actions(cfg, template, changes, geometry)
    cfg.pop('branches')
    cfg['branches'] = list(controllers.branch_names(cfg))
    actual_actions = set(controllers.branch_names(cfg, include_repeat=False))
    if not actual_actions.issubset(set(cell['candidate_actions'])):
        raise ValueError(f'Actual geometry requires actions outside declared candidate_actions: {sorted(actual_actions-set(cell["candidate_actions"]))}')
    cfg['controller']['leader_geometry'] = 'route_projection'
    cfg['controller']['npc_style'] = cell['npc_style']
    _profile(cfg, cell['npc_style_parameters'])
    individualize(cfg,geometry)
    from scene_spec import apply_authoring, authoring_payload
    apply_authoring(cfg, chosen.get('scene', authoring_payload(cfg)), geometry)
    snapshot(cfg,geometry,catalog.maps[cfg['map']])
    cfg['interaction_design']['initial_actor_parameters'] = [
        {key:copy.deepcopy(actor[key]) for key in ('role','spawn_route_s','initial_speed_mps','desired_speed_mps')}
        for actor in cfg['vehicles']]
    if any(geometry.spawn_footprints_overlap(a,b) for a,b in itertools.combinations(cfg['vehicles'],2)):
        raise ValueError('Diverse initial layout overlaps conservative vehicle footprints')
    cfg['geometry']['traffic_design']['effective_actor_parameters'] = [
        {'role':actor['role'],'initial_speed_mps':actor['initial_speed_mps'],
         'desired_speed_mps':actor['desired_speed_mps'],'blueprint_id':actor['blueprint_id']}
        for actor in cfg['vehicles']]
    geometry.check_lane_change_future_budget(cfg)
    controllers.ScenarioController(cfg)
    for actor in cfg['vehicles']:
        available = float(geometry.cumulative(actor['route'])[-1])-actor['spawn_route_s']
        nominal = max(actor['initial_speed_mps'], actor['desired_speed_mps'])*(cfg['driving_warmup_s']+cfg['history_s']+cfg['duration_s'])
        required = nominal + (12. if actor['role'] == 'ego' else 0.) + 10.
        if available < required:
            raise ValueError(f'{actor["role"]}: insufficient mapped route horizon ({available:.1f} < {required:.1f} m)')
    ego = cfg['vehicles'][0]
    centre = geometry.point_at_s(ego['route'], ego['spawn_route_s']+ego['initial_speed_mps']*(cfg['history_s']+cfg['driving_warmup_s']))['position']
    cfg['local_map_lanes'] = geometry.local_lane_geometry(catalog.maps[cfg['map']].generate_waypoints(2.),centre)
    cfg['local_map_geometry'] = {'center_xyz':centre,'radius_m':120.,'sample_distance_m':2.}
    cfg['map_geometry_path'] = 'maps/'+cfg['map']+'.xodr'
    cfg['provenance'].update(opendrive_sha256=catalog.hashes[cfg['map']],
        compiler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope_status='compiled_geometry_requires_physics_and_rgb_preflight',
        sampling={'layout_seed':chosen['layout_seed'],'scene_parameters_frozen_across_all_branches':True},
        spawn_footprint_check={'shape':'oriented conservative rectangle','length_m':5.6,'width_m':2.3,
                              'all_pairs_disjoint':True,'actual_blueprint_bbox_requires_capture_audit':True})
    cfg['grid_spec'] = {'cell_id':cell['cell_id'],'template':template,'npc_style':cell['npc_style'],
        'attempt_index':attempt_index,'selection':chosen,'candidate_id':chosen['candidate_id'],
        'geometry_validated':True,'physical_validation':'pending','rgb_validation':'pending'}
    from sampling import location_id
    cfg['grid_spec']['location_id'] = location_id(candidate)
    from encounter_calibration import configure as configure_encounter
    configure_encounter(cfg,template)
    return cfg


def write_plan(data, configs, plan, render_preset=None, physics_calibration=None, catalog=None):
    """Freeze a new dataset plan. Existing plans/configs are never overwritten."""
    runtime_modules()
    import collect_multicam as collector
    data = Path(data).resolve()
    if (data/'manifest.json').exists() or any((data/c['scene_id']/'spec.json').exists() for c in configs):
        raise FileExistsError('Choose a fresh output directory for compiled configs')
    from sampling import weather_variants
    from curate_dataset import requested_base_scene_id
    variants = {v['id']: v for v in weather_variants(plan)}
    finalized = []
    for cfg in configs:
        weather_id = cfg['grid_spec']['selection']['weather_id']
        if weather_id not in variants:
            raise ValueError(f'Unknown weather_id {weather_id!r}')
        cfg['weather_override'] = copy.deepcopy(variants[weather_id].get('overrides',{}))
        cfg['weather_variant_id'] = weather_id
        cfg['camera_override'] = copy.deepcopy(variants[weather_id].get('camera_overrides', {}))
        cfg['vehicle_light_mode'] = variants[weather_id].get('vehicle_lights', 'default')
        cfg['base_scene_id'] = requested_base_scene_id(cfg)
        cfg['weather_validation'] = 'configured_requires_native_rgb_and_repeat_review'
        cfg = collector.finalize_scene(cfg,data=data,render_preset=render_preset,physics_calibration=physics_calibration)
        finalized.append(cfg)
    data.mkdir(parents=True,exist_ok=True)
    if catalog is not None:
        for name in sorted({c['map'] for c in finalized}):
            source = catalog.root/(name+'.xodr')
            target = data/'maps'/(name+'.xodr')
            target.parent.mkdir(exist_ok=True)
            if not target.exists(): target.write_bytes(source.read_bytes())
            elif target.read_bytes() != source.read_bytes(): raise ValueError('Frozen map content mismatch')
    extra = {'grid_sha256':plan.get('grid_sha256'), 'grid_cells':plan['cells'],
        'disabled_pairs':plan.get('disabled_pairs',[]), 'excluded_pairs':plan.get('excluded_pairs',[]),
        'requested_scene_count':plan.get('count',len(configs)), 'planned_scene_count':len(configs)}
    manifest = collector.finalize_manifest(finalized,data,extra=extra)
    return manifest


def compile_plan(plan, data, opendrive, maps=('Town04','Town05','Town10HD','Town07'), render_preset=None,
                 physics_calibration=None, max_repairs=3, selected_jobs=None, selections=None):
    if isinstance(plan,(str,Path)): plan=json.loads(Path(plan).read_text())
    if (Path(data)/'manifest.json').exists() or (Path(data)/'compilation.json').exists():
        raise FileExistsError('Choose a fresh output directory for this compiled batch')
    if type(max_repairs) is not int or not 0 <= max_repairs <= 3:
        raise ValueError('max_repairs must be an integer in [0,3]')
    cells={c['cell_id']:c for c in plan['cells']}
    jobs=[j for j in plan['jobs'] if selected_jobs is None or j['scene_id'] in selected_jobs]
    if not jobs: raise ValueError('No selected jobs')
    templates={cells[j['cell_id']]['template'] for j in jobs}
    catalog=build_catalog(opendrive,maps,templates)
    missing=sorted(t for t in templates if not catalog.candidates[t])
    if missing: raise ValueError(f'No actual map candidates for required templates: {missing}; change map availability or explicitly revise the distribution')
    from sampling import location_shortlist, coverage_row
    configs, failures, proposals=[],[],[]
    for job in jobs:
        cell=cells[job['cell_id']]
        cfg=None
        for attempt in range(max_repairs+1):
            choice=(selections or {}).get(job['scene_id']) if attempt==0 else None
            if choice is None:
                variants=plan.get('weather_variants',[{'id':'daylight','overrides':{}}])
                candidates = eligible_candidates(catalog, cell)
                failed_ids = [r['selection'].get('candidate_id') for r in failures if r['scene_id'] == job['scene_id']]
                chosen = location_shortlist(candidates, [coverage_row(c) for c in configs], cell['template'],
                                            [job['seed'], attempt], failed_ids)[0]
                choice={'candidate_id': chosen['candidate_id'],
                        'weather_id':job.get('weather_id',variants[job['component_seeds']['weather']%len(variants)]['id'])}
            try:
                from scene_randomization import intent
                chosen=validate_selection(choice,job,cell,catalog,attempt)
                proposed={'scene_id':job['scene_id'],'cell_id':cell['cell_id'],'attempt_index':attempt,
                          'intent':intent(job,cell,chosen),'selection':chosen,'status':'proposed'}
                proposals.append(proposed)
                cfg=compile_one(job,cell,catalog,attempt,choice)
                proposed['status']='compiled'
                break
            except ValueError as exc:
                if proposals and proposals[-1]['scene_id']==job['scene_id'] and proposals[-1]['attempt_index']==attempt:
                    proposals[-1].update(status='rejected',reason=str(exc))
                failures.append({'scene_id':job['scene_id'],'cell_id':cell['cell_id'],
                                 'attempt_index':attempt,'selection':choice,'reason':str(exc)})
        if cfg is None:
            # Preserve missing quotas explicitly; never backfill with an easier cell.
            continue
        configs.append(cfg)
    report={'status':'compiled' if len(configs)==len(jobs) else 'partial_geometry_failure',
        'requested':len(jobs),'compiled':len(configs),'geometry_attempt_failures':failures,'sampling_proposals':proposals,
        'catalog_counts':{k:len(v) for k,v in catalog.candidates.items()},
        'cell_eligibility':[candidate_eligibility(catalog,cells[key])[1] for key in sorted({j['cell_id'] for j in jobs})],
        'catalog':catalog.public(),'physical_validation':'not_run','rgb_validation':'not_run'}
    if configs:write_plan(data,configs,plan,render_preset,physics_calibration,catalog)
    write_new(Path(data)/'compilation.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--opendrive',type=Path,required=True)
    parser.add_argument('--maps',nargs='+',default=['Town04','Town05','Town10HD','Town07'])
    parser.add_argument('--render-preset',type=Path)
    parser.add_argument('--physics-calibration',type=Path,nargs='+')
    parser.add_argument('--max-repairs',type=int,default=3)
    parser.add_argument('--scenes',nargs='+')
    parser.add_argument('--catalog-out',type=Path,help='Only export map-grounded catalog IDs')
    args=parser.parse_args()
    try:
        if args.catalog_out:
            catalog=build_catalog(args.opendrive,args.maps)
            write_new(args.catalog_out,{'candidates':catalog.public(),'map_sha256':catalog.hashes,
                                        'exclusions':catalog.exclusions})
            print(json.dumps({'status':'catalogued','candidates':len(catalog.public())}))
            return
        if args.plan is None or args.data is None:parser.error('--plan and --data are required for compilation')
        result=compile_plan(args.plan,args.data,args.opendrive,args.maps,args.render_preset,
                            args.physics_calibration,args.max_repairs,args.scenes)
        print(json.dumps({k:result[k] for k in ('status','requested','compiled','catalog_counts')}))
        if result['status']!='compiled':sys.exit(4)
    except (ValueError,KeyError,TypeError,OSError,ImportError) as exc:
        parser.exit(2,f'Error: {exc}\n')


if __name__=='__main__':main()
