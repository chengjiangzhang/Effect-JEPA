"""Template-specific ego interventions and offline feasibility evidence.

The action catalogue is a set of control conditions, not promised outcomes.
All distance/timing calculations below use only frozen initial geometry; actual
four-second trajectories and duplicates must be checked after execution.
"""
from __future__ import annotations

import copy
import itertools
import math
import numpy as np

if __package__:
    from .runtime.ego_decisions import longitudinal_target, DEFAULT_STOP_RESPONSE_TIME_S, DEFAULT_STOP_MARGIN_M
else:
    from runtime.ego_decisions import longitudinal_target, DEFAULT_STOP_RESPONSE_TIME_S, DEFAULT_STOP_MARGIN_M

if __package__:
    from .runtime.action_separation import CONTRACT as ACTION_SEPARATION_CONTRACT
else:
    from runtime.action_separation import CONTRACT as ACTION_SEPARATION_CONTRACT


def _long(acceleration, onset=0., floor=0., cap=5.):
    return {'kind': 'longitudinal', 'onset_s': onset,
            'acceleration_m_s2': acceleration,
            'speed_increase_cap_m_s': cap if acceleration > 0 else 0.,
            'min_target_speed_m_s': floor}


def profiles_for_template(template, directions, conflict, seed=0):
    """Five distinct definitions. Equal realised trajectories are not five samples."""
    maintain = {'kind': 'maintain', 'onset_s': 0., 'acceleration_m_s2': 0.,
                'speed_increase_cap_m_s': 0.}
    profiles = {'maintain': maintain}
    if template == 'follower_response_straight':
        profiles.update(brake_light=_long(-1.2), brake=_long(-4.5),
                        accelerate=_long(1.5), brake_delayed=_long(-4.5, onset=.6))
        return profiles
    if template == 'ego_lane_change_straight':
        if not directions:
            raise ValueError('Ego lane-change template needs at least one mapped target lane')
        def lane(side, onset=0.):
            return {'kind': 'lane_change', 'direction': side, 'onset_s': onset,
                    'duration_s': 2.2, 'settling_time_s': .9,
                    'acceleration_m_s2': 0., 'speed_increase_cap_m_s': 0.,
                    'required_final_hold_s': .5}
        if len(directions) == 1:
            side = directions[0]
            profiles.update(brake=_long(-4.5), accelerate=_long(1.5))
            profiles[side] = lane(side)
            profiles[side + '_delayed'] = lane(side, .4)
        else:
            side = ('left', 'right')[seed % 2]
            profiles.update(left=lane('left'), right=lane('right'))
            profiles[side + '_delayed'] = lane(side, .4)
            profiles['accelerate'] = _long(1.5)
        return profiles
    if conflict is None:
        raise ValueError(f'{template}: a mapped conflict interval is required')
    stop = {'kind': 'stop_hold', 'onset_s': 0., 'stop_route_s': conflict['stop_route_s'],
            'hold_policy': 'hold_until_window_end_v1',
            'stop_deadline_s': 3.5, 'stop_margin_m': DEFAULT_STOP_MARGIN_M,
            'comfortable_deceleration_m_s2': 4.5, 'response_time_s': DEFAULT_STOP_RESPONSE_TIME_S, 'hold_until_s': 4.}
    wait = {'kind': 'yield_wait', 'onset_s': 0., 'stop_route_s': conflict['stop_route_s'],
            'conflict_id': conflict['id'], 'min_wait_s': 0., 'release_gap_s': 1.5,
            'release_condition': 'specified_actors_cleared_and_entry_clear_v1',
            'wait_for_roles': [part['role'] for part in conflict.get('participants',[])],
            'timeout_s': 4., 'timeout_behavior': 'remain_stopped',
            'stop_deadline_s': 3.5, 'stop_margin_m': DEFAULT_STOP_MARGIN_M,
            'comfortable_deceleration_m_s2': 4.5, 'response_time_s': DEFAULT_STOP_RESPONSE_TIME_S}
    commit = {'kind': 'commit_go', 'onset_s': 0., 'acceleration_m_s2': 1.5,
              'speed_increase_cap_m_s': 5., 'stop_route_s': conflict['stop_route_s'],
              'response_time_s': DEFAULT_STOP_RESPONSE_TIME_S,
              'respect_signals': False, 'risk_condition': 'ego does not perform conflict yielding; NPC policy unchanged'}
    nominal = {'kind': 'nominal', 'onset_s': 0.}
    if template in ('merge_straight', 'merge_ramp'):
        profiles.update(decelerate=_long(-1.2, floor=1.), accelerate=_long(1.5),
                        yield_wait=wait, nominal=nominal)
    elif template == 'ego_merge_ramp':
        delayed = copy.deepcopy(commit)
        delayed.update(onset_s=1., before_onset='approach_and_stop_at_declared_stop_station')
        profiles.update(commit_go=commit, commit_delayed=delayed, stop_hold=stop, yield_wait=wait)
    elif template.startswith('pedestrian_'):
        profiles.update(nominal=nominal, decelerate=_long(-1.2, floor=1.), stop_hold=stop, yield_wait=wait)
    elif template in ('left_unsignalized', 'left_signalized', 'cross_unsignalized',
                      'cross_signalized', 'narrow_meeting'):
        profiles.update(decelerate=_long(-1.2, floor=1.), stop_hold=stop, yield_wait=wait, commit_go=commit)
    else:
        raise ValueError(f'No action design for template {template!r}')
    return profiles


def distinct_profiles_for_template(template, directions, conflict, seed=0):
    """New four-second catalogue; actual ten-pair eligibility is a separate gate.

    Nominal continuation is useful in common history but is not reliably a
    distinct future action. Stop and yield also need different braking phases,
    since a correct yield may remain stopped for the whole four-second future.
    """
    profiles = profiles_for_template(template, directions, conflict, seed)
    if template == 'ego_lane_change_straight':
        profiles = {name: p for name, p in profiles.items() if not name.endswith('_delayed')}
        profiles['brake_light' if len(directions) == 1 else 'brake'] = _long(-1.2 if len(directions) == 1 else -4.5)
    elif template.startswith('pedestrian_'):
        profiles = {name: (_long(1.5) if name == 'nominal' else p) for name, p in profiles.items()}
        profiles['accelerate'] = profiles.pop('nominal')
    elif template in ('merge_straight', 'merge_ramp'):
        profiles['brake'] = _long(-4.5)
        profiles.pop('nominal')
    for p in profiles.values():
        if p['kind'] == 'stop_hold':
            p.update(stop_deadline_s=2.25, comfortable_deceleration_m_s2=6.)
        elif p['kind'] == 'yield_wait':
            p.update(stop_deadline_s=3.5, comfortable_deceleration_m_s2=4.5)
    return profiles


def _interval(route, centre, before=4., after=6., geometry=None):
    end = float(geometry.cumulative(route)[-1])
    return max(0., centre - before), min(end, centre + after)


def conflict_for_config(cfg, template, geometry):
    """Declare wait intervals on mapped routes; not reactive GT or collision labels."""
    if template in ('follower_response_straight', 'ego_lane_change_straight'):
        return None
    ego = cfg['vehicles'][0]
    geom = cfg['geometry']
    participants = []
    if template.startswith('pedestrian_'):
        centre = float(geom['ego_conflict_s'])
        entry, exit = _interval(ego['route'], centre, 2., 2., geometry)
        for ped in cfg['pedestrians']:
            path = np.asarray(ped['route_points'])
            delta = np.diff(path[:, :2], axis=0)
            lengths = np.linalg.norm(delta, axis=1)
            walking = np.r_[0., np.cumsum(lengths)]
            point = np.asarray(geometry.point_at_s(ego['route'], centre)['position'])
            along = np.clip(np.einsum('ij,ij->i', point[:2] - path[:-1, :2], delta)
                            / np.maximum(lengths * lengths, 1e-12), 0., 1.)
            nearest = path[:-1, :2] + along[:, None] * delta
            near = int(np.argmin(np.linalg.norm(nearest - point[:2], axis=1)))
            crossing = float(walking[near] + along[near] * lengths[near])
            participants.append({'role': ped['role'], 'actor_type': 'pedestrian',
                'entry_s': max(0., crossing - 1.8),
                'exit_s': min(float(walking[-1]), crossing + 1.8)})
    elif template == 'narrow_meeting':
        rule = geom['narrow_meeting']
        own = rule['priority_route_s'].get('ego', rule['yield_route_s'].get('ego'))
        entry, exit = map(float, own)
        opposite = (rule['yield_route_s'] if 'ego' in rule['priority_roles'] else rule['priority_route_s'])
        participants = [{'role': role, 'actor_type': 'vehicle', 'entry_s': float(value[0]),
                         'exit_s': float(value[1])} for role, value in opposite.items()]
    else:
        if template == 'merge_straight':
            other = next(a for a in cfg['vehicles'] if a['role'] == 'parallel_vehicle')
            a, b = other['route_maneuver']['source_interval_s_m']
            npc_station = (a + b) / 2.
            point = geometry.point_at_s(other['route'], npc_station)['position']
            centre = geometry.city_project_s(ego['route'], point)
            geom['ego_conflict_s'] = centre
            geom['conflict_xyz'] = point
            others = [other]
        else:
            centre = float(geom['ego_conflict_s'])
            point = geom.get('conflict_xyz', geom['origin'])
            others = [a for a in cfg['vehicles'][1:] if not np.array_equal(a['route'], ego['route'])]
        entry, exit = _interval(ego['route'], centre, geometry=geometry)
        for actor in others:
            station = geometry.city_project_s(actor['route'], point)
            own_entry, own_exit = _interval(actor['route'], station, geometry=geometry)
            participants.append({'role': actor['role'], 'actor_type': 'vehicle',
                                 'entry_s': own_entry, 'exit_s': own_exit})
    if not participants:
        raise ValueError(f'{template}: no mapped interaction participants')
    # Centre stop station protects a 5.6-m conservative body before the zone.
    stop_station = entry - 2.8 - 1.5
    if __package__:
        from .interaction_geometry import safe_waiting_station
    else:
        from interaction_geometry import safe_waiting_station
    waiting_geometry = safe_waiting_station(cfg,stop_station,[p['role'] for p in participants])
    stop_station = waiting_geometry['stop_route_s']
    group_id = None
    signal_stop_station = None
    for group in cfg.get('traffic_signal_program', {}).get('groups', []):
        if 'ego' in group['roles']:
            # A permissive left turn may already have passed the green stop
            # line before reaching its oncoming conflict. The waiting station
            # is before that conflict, not automatically back at the signal.
            # Nominal traffic-light handling uses its independently mapped line.
            signal_stop_station = float(group['stop_route_s']['ego']) - 2.8
            group_id = group['id']
    result = {'id': 'interaction_0', 'ego_entry_s': entry, 'ego_exit_s': exit,
              'stop_route_s': stop_station, 'participants': participants,
              'waiting_geometry': waiting_geometry,
              'ego_must_yield': template != 'narrow_meeting' or 'ego' in geom['narrow_meeting']['yield_roles'],
              'zone_definition': 'declared mapped approach interval, not ground-truth reactive or collision region',
              'release_observations': 'current actor route positions and velocities only; no branch or future trajectory'}
    if any(rule.get('priority_role') == 'ego' and 'ego' in rule.get('roles', [])
           for rule in geom.get('conflicts', [])):
        result['ego_must_yield'] = False
    if group_id is not None:
        result['signal_group'] = group_id
        result['signal_stop_center_route_s'] = signal_stop_station
        result['wait_station_beyond_signal_line'] = stop_station > signal_stop_station
        result['signal_policy'] = 'nominal respects the approach signal; intervention stopping/commit is not guaranteed lawful or safe'
    return result


def joint_pedestrian_encounter(cfg):
    design=cfg.get('interaction_design',{})
    return bool(cfg.get('pedestrians') and design.get('pedestrian_encounter',{}).get('version')=='body-relative-arrival-v2'
                and design.get('requested_phase') in ('entering','crossing'))


def target_travel_with_response(profile, speed, window, response):
    """Integrate a capped target-speed ramp after an explicit planning reserve.

    This is not a native-physics lower bound or measured acceleration latency.
    """
    acceleration=float(profile.get('acceleration_m_s2',0.))
    cap=float(profile.get('speed_increase_cap_m_s',0.))
    onset=float(profile.get('onset_s',0.))
    if (profile.get('kind')!='longitudinal' or
            not all(math.isfinite(x) for x in (acceleration,cap,onset,speed,window,response)) or
            acceleration<=0 or cap<=0 or min(onset,speed,window,response)<0):
        raise ValueError('Prospective entry requires a finite positive capped longitudinal profile')
    active=max(0.,window-onset-response)
    ramp=min(active,cap/acceleration)
    return speed*window+.5*acceleration*ramp*ramp+cap*max(0.,active-ramp)


def validate_joint_pedestrian_entry(cfg,conflict,*,final_authoring=False):
    """Require a frozen strategy to have a prospective encounter within 3s."""
    if not joint_pedestrian_encounter(cfg):
        return
    if __package__:
        from .action_opportunity import MIN_OBSERVATION_S
    else:
        from action_opportunity import MIN_OBSERVATION_S
    opportunity=cfg['geometry']['traffic_design']['interaction_opportunity']
    # Authoring can change speed, station, fork time and profiles. Never reuse
    # the earlier placement estimate for a final authored-input check.
    ego=cfg['vehicles'][0]
    speed=float(ego['initial_speed_mps'])
    pre=float(cfg['history_s']+cfg['driving_warmup_s'])
    station=float(ego['spawn_route_s'])+pre*speed
    distance=conflict['ego_entry_s']-station
    window=cfg['duration_s']-MIN_OBSERVATION_S
    profiles=[]
    for name,profile in cfg['action']['profiles'].items():
        if profile['kind']=='longitudinal' and profile.get('acceleration_m_s2',0.)>0:
            travel=target_travel_with_response(profile,speed,window,DEFAULT_STOP_RESPONSE_TIME_S)
            profiles.append({'branch':name,'frozen_profile':copy.deepcopy(profile),'target_travel_m':travel})
    passed=bool(distance>0 and any(x['target_travel_m']+1e-9>=distance for x in profiles))
    opportunity['strategy_entry_opportunity']={
        'policy':'joint-pedestrian-target-speed-entry-v1','passed':passed,
        'distance_to_entry_m':distance,'entry_horizon_s':window,
        'estimated_t0_route_s':station,'estimated_t0_speed_m_s':speed,
        'common_motion_duration_s':pre,
        'observation_reserve_s':MIN_OBSERVATION_S,
        'planning_response_reserve_s':DEFAULT_STOP_RESPONSE_TIME_S,'profiles':profiles,
        'uses_final_authored_profiles':bool(final_authoring),'native_arrival_guaranteed':False,
        'response_reserve_is_not_measured_acceleration_latency':True,
        'future_response_used':False}
    if not passed:
        raise ValueError('No frozen positive-speed strategy provides a prospective pedestrian encounter with one second remaining')


def _place_stop_approach(cfg, conflict, geometry, deceleration=4.5, delay=0.):
    """Ensure the *initial plan* leaves braking distance; no trajectory feedback."""
    if conflict is None:
        return
    ego = cfg['vehicles'][0]
    pre = cfg['history_s'] + cfg['driving_warmup_s']
    v = float(ego['initial_speed_mps'])
    station = ego['spawn_route_s'] + pre * v
    stopping = v * v / (2. * deceleration)
    response_distance = v * (DEFAULT_STOP_RESPONSE_TIME_S+delay)
    shortage = max(0., station + stopping + response_distance + DEFAULT_STOP_MARGIN_M - conflict['stop_route_s'])
    requested_eta = cfg.get('interaction_design', {}).get('prospective_ego_entry_s')
    if requested_eta is not None:
        shortage = max(shortage, station-(conflict['ego_entry_s']-v*requested_eta))
        encounter = cfg.get('interaction_design', {}).get('pedestrian_encounter',{})
        if (encounter.get('version') == 'body-relative-arrival-v2'
                and cfg['interaction_design']['requested_phase'] in ('entering','crossing')):
            # Jointly rescale placement when changing the initial approach
            # speed. Keeping the old high-speed spawn would only increase the
            # arrival time. A signed shift may move the whole queue forwards;
            # the unchanged stopping envelope still bounds the nearest pose.
            requested_station = conflict['ego_entry_s']-v*requested_eta
            stop_bound = conflict['stop_route_s']-stopping-response_distance-DEFAULT_STOP_MARGIN_M
            shortage = station-min(requested_station,stop_bound)
    if shortage:
        same_stream = {slot['role'] for slot in cfg['geometry']['traffic_design'].get('roles_and_slots', [])
                       if slot.get('route_lane') == 'main'}
        for actor in cfg['vehicles']:
            if actor['role'] in same_stream or np.array_equal(actor['route'], ego['route']):
                target = actor['spawn_route_s'] - shortage
                if target < 5.:
                    raise ValueError('Mapped approach cannot provide a stopping distance and common history')
                actor['spawn_route_s'] = target
                actor['transform'] = geometry.point_at_s(actor['route'], target)
        station -= shortage
    if any(geometry.spawn_footprints_overlap(a, b) for a, b in itertools.combinations(cfg['vehicles'], 2)):
        raise ValueError('Interaction approach placement overlaps initial conservative footprints')
    arrival = (conflict['ego_entry_s'] - station) / max(v, .01)
    # An already-moving pedestrian can react within four seconds when ego
    # accelerates even if maintain would arrive later. Its alternate initial
    # opportunity gate runs after the final action profiles are authored.
    if not joint_pedestrian_encounter(cfg) and not 0. < arrival <= cfg['duration_s']:
        raise ValueError(f'No nominal ego opportunity to enter mapped conflict during future window: {arrival:.2f}s')
    cfg['geometry']['traffic_design']['interaction_opportunity'] = {
        'estimated_t0_route_s': station, 'estimated_t0_speed_m_s': v,
        'common_ego_stream_spawn_shift_back_m': shortage,
        'nominal_constant_speed_entry_after_t0_s': arrival,
        'stop_center_route_s': conflict['stop_route_s'],
        'available_stop_distance_m': conflict['stop_route_s'] - station,
        'ideal_braking_distance_m': stopping,
        'design_deceleration_m_s2': deceleration,
        'response_time_s': DEFAULT_STOP_RESPONSE_TIME_S,
        'delayed_brake_onset_s': delay,
        'controller_response_distance_m': response_distance,
        'control_stop_margin_m': DEFAULT_STOP_MARGIN_M,
        'required_stop_distance_m': stopping + response_distance + DEFAULT_STOP_MARGIN_M,
        'basis': 'initial geometry and constant-speed estimate only; measured t0 and trajectories must be checked',
        'outcome_filtering': False,
    }


def speed_profile_estimates(profiles, speed, window=4., dt=.05):
    """Integrate the runtime speed targets; these are not realised trajectories."""
    times = np.arange(0., window + dt * .5, dt)
    values = {}
    for name, p in profiles.items():
        if p['kind'] not in ('maintain', 'longitudinal', 'lane_change', 'commit_go'):
            values[name] = {'depends_on_current_traffic_state': True}
            continue
        onset = p.get('onset_s', 0.)
        target = np.array([longitudinal_target(p, t, speed) for t in times])
        values[name] = {'target_end_speed_m_s': float(target[-1]),
                       'target_travel_m': float(np.sum((target[:-1] + target[1:]) * np.diff(times) / 2.)),
                       'target_first_zero_s': next((float(t) for t, x in zip(times, target) if x <= 1e-9), None),
                       'onset_s': onset, 'ideal_target_only': True}
        if p['kind'] == 'commit_go' and onset:
            # Before entry time the stateful controller approaches its stop
            # station; a speed-only profile cannot estimate that travel.
            values[name]['target_travel_m'] = None
            values[name]['before_onset_depends_on_stop_approach'] = True
    return values


def nominal_pedestrian_entry(cfg, geometry):
    """Check an initial-layout opportunity using the actual mapped approach.

    This predicts a nominal departure from the sidewalk, before native capture.
    It never simulates selected future ego actions or demands a future response.
    Conservative planar bodies are planning bounds, not measured actor boxes.
    """
    from runtime.pedestrian_crossing import motion_profile, risk
    from runtime.pedestrian_policy import route_projection
    ped=cfg['pedestrians'][0]
    when=float(ped['start_time_s'])
    pre=cfg['history_s']+cfg['driving_warmup_s']
    obstacles=[]
    for actor in cfg['vehicles']:
        speed=actor['initial_speed_mps']
        pose=geometry.point_at_s(actor['route'],actor['spawn_route_s']+speed*(pre+when))
        yaw=math.radians(pose['rotation_deg'][2])
        orientation=np.array([[math.cos(yaw),-math.sin(yaw),0.],
                              [math.sin(yaw),math.cos(yaw),0.],[0.,0.,1.]])
        center=np.r_[pose['position'][:2],0.]
        obstacles.append((actor['role'],center,orientation,np.array([2.8,1.15,1.2]),
                          orientation[:,0]*speed,0.))
    position=np.r_[ped['route_points'][0][:2],0.]
    station,tangent=route_projection(position,ped['route_points'])
    estimate=copy.deepcopy(ped)
    p=estimate['response_policy']['crossing_feedback']
    length=float(np.linalg.norm(np.diff(np.array(ped['route_points'])[:,:2],axis=0),axis=1).sum())
    p['lookahead_s']=max(p['lookahead_s'],length/ped['speed_mps']+ped['speed_mps']/p['acceleration_m_s2']+1.)
    times,points,_,_=motion_profile(position,np.zeros(3),ped['route_points'],station,tangent,
                                   'continue',0.,estimate,cfg['dt_s'])
    check=risk(points,times,obstacles,np.array([.3,.3,1.]),p)
    evidence={'time_s':when,'planning_bodies_m':{'vehicle_half':[2.8,1.15,1.2],'walker_half':[.3,.3,1.]},
              'prediction':'mapped current heading and nominal constant speed at earliest departure',
              'native_initial_state_and_future_response_not_claimed':True,**check}
    ped['initial_opportunity_design']['nominal_entry_sweep']=evidence
    if not check['caution_clear']:
        roles=[r['role'] for r in check['actor_checks'] if r['caution_time_s'] is not None]
        raise ValueError('Mapped initial approach provides no nominal pre-t0 pedestrian departure gap: '+','.join(roles))


def turn_speed_plan(route, conflict_s, nominal_arrival_s, proposed_speed):
    """Joint approach/commit speeds from the mapped curve ahead of the fork."""
    xy = np.asarray(route, dtype=float)[:, :2]
    stations = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    start = max(0., conflict_s - nominal_arrival_s * proposed_speed)
    end = min(stations[-1], start + 4. * proposed_speed + .5 * 1.5 * 4.**2 + 3.)
    if end <= start:
        raise ValueError('Left-turn opportunity lies outside the mapped route')
    samples = np.linspace(start, end, max(3, math.ceil(end - start) + 1))
    def points(offset):
        return np.stack([np.interp(samples + offset, stations, xy[:, axis]) for axis in range(2)], axis=1)
    before, centre, after = points(-2.), points(0.), points(2.)
    a, b = centre - before, after - centre
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(a+b, axis=1)
    curvature = np.divide(2 * np.abs(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]), denominator,
                          out=np.zeros_like(denominator), where=denominator > 1.e-8)
    peak = float(curvature.max())
    # Reserve longitudinal tire force for a positive commit at friction 0.6.
    # This only sets action parameters. The original 1 m route audit remains.
    lateral_budget = 3.5
    limit = min(proposed_speed + 5.15, math.sqrt(lateral_budget / max(peak, 1.e-9)))
    speed = min(proposed_speed, limit - 1.35)
    if speed < 2.5:
        raise ValueError('Mapped turn cannot sustain five distinct actions above the minimum fork speed')
    return {'policy': 'mapped-turn-speed-v1', 'curvature_span_m': [start, end],
            'maximum_curvature_per_m': peak, 'design_lateral_acceleration_m_s2': lateral_budget,
            'ceiling_m_s': limit, 'proposed_approach_m_s': proposed_speed, 'approach_m_s': speed,
            'commit_increase_cap_m_s': min(5., limit - speed - .15)}


def configure_actions(cfg, template, changes, geometry):
    conflict = conflict_for_config(cfg, template, geometry)
    if cfg.get('pedestrians') and cfg.get('interaction_design'):
        phase=cfg['interaction_design']['requested_phase']
        pedestrian=cfg['pedestrians'][0]
        interval=conflict['participants'][0]
        if phase != 'curb_wait':
            encounter=cfg['interaction_design'].get('pedestrian_encounter',{})
            station={'entering':max(.1,interval['entry_s']-encounter.get('entry_body_setback_m',.35)),
                     'crossing':interval['entry_s']+encounter.get('crossing_body_progress_m',.65),
                     'clearing':interval['exit_s']-encounter.get('clearing_body_remaining_m',.55)}[phase]
            speed=pedestrian['speed_mps']
            acceleration=cfg.get('randomization',{}).get('pedestrian_policy',{}).get('acceleration_m_s2',1.5)
            ramp_distance=.5*speed*speed/acceleration
            time=(math.sqrt(2.*station/acceleration) if station<ramp_distance else
                  station/speed+.5*speed/acceleration)+.2
            time=min(5.,round(time/.05)*.05)
            pedestrian['start_time_s']=-time
            cfg['driving_warmup_s']=max(1.,round(max(0.,time-2.)/.05)*.05)
            pedestrian['initial_opportunity_design'].update(
                earliest_decision_s=-time,prospective_t0_station_m=station,
                basis='bounded nominal walking from mapped sidewalk; measured stage may differ')
            # Joint arrival design gives an entering walker a plausible gap
            # before the constant-speed ego, while acceleration can change it.
            if encounter and phase in ('entering','crossing'):
                # Conservative lateral body/caution clearance plus the vehicle
                # front reaching this crossing. Geometry and nominal motion,
                # not a simulated future response. Measured t0 is still audited.
                crossing=(interval['entry_s']+interval['exit_s'])/2.
                walker_clear=max(0.,crossing+2.35-station)/speed
                ego_speed=cfg['vehicles'][0]['initial_speed_mps']
                eta=walker_clear+.8/ego_speed+encounter['arrival_slack_s']
                cfg['interaction_design']['prospective_ego_entry_s']=eta
                encounter.update(prospective_walker_clearance_s=walker_clear,
                    prospective_ego_centre_arrival_s=eta+2./ego_speed,
                    pedestrian_t0_station_target_m=station,
                    geometry_and_nominal_estimate_only=True)
            else:
                cfg['interaction_design']['prospective_ego_entry_s']=2.8
    delay=0.
    if cfg.get('interaction_design') and (template.startswith('pedestrian_') or template in ('left_unsignalized','narrow_meeting')):
        from interaction_design import PHASES
        delay=cfg.get('randomization',{}).get('parameters',{}).get('stop_delay_s',(.35,.5,.65,.45)[PHASES[template].index(cfg['interaction_design']['requested_phase'])])
    _place_stop_approach(cfg, conflict, geometry, deceleration=4.5,delay=delay)
    cfg['action_separation_contract'] = copy.deepcopy(ACTION_SEPARATION_CONTRACT)
    action = cfg['action']
    # Only the ego lane-change template varies lateral choice. The other road
    # templates keep their requested interaction and do not add unrelated turns.
    directions = sorted(changes) if template == 'ego_lane_change_straight' else []
    action.update(schema_version=2, profile_version=2, available_lane_change_directions=directions,
        lane_changes={k: copy.deepcopy(changes[k]) for k in directions},
        lateral_mode='adjacent_lane_route',
        conflicts=[conflict] if conflict is not None else [],
        reference='maintain tracks measured t0 speed; common history uses frozen nominal policy',
        nominal_policy={'desired_speed_m_s': cfg['vehicles'][0]['desired_speed_mps'],
            'idm': {'max_acceleration': 2., 'comfortable_brake': 2.8, 'time_headway': 1.25, 'minimum_gap': 2.5},
            'release_gap_s': 1.5, 'comfortable_deceleration_m_s2': 4.5,
            'response_time_s': DEFAULT_STOP_RESPONSE_TIME_S,
            'yield_lookahead_s': 4., 'lateral_margin_m': .3,
            'stop_margin_m': 1., 'respect_signals': True})
    action['profiles'] = distinct_profiles_for_template(template, directions, conflict, cfg['seed'])
    turn = cfg.get('geometry', {}).get('traffic_design', {}).get('turn_speed_plan')
    if turn is not None:
        action['profiles']['commit_go']['speed_increase_cap_m_s'] = turn['commit_increase_cap_m_s']
    if template == 'cross_signalized':
        # This template's tight mapped stop approach can make a long hold
        # deadline nearly alias the yield spatial cap. Command an earlier stop
        # from INITIAL speed only, retaining the existing response reserve and
        # two physics ticks of speed-drift capacity below the declared brake cap.
        stop = action['profiles']['stop_hold']
        # Keep the target ramp, with two further ticks for sampled actuator
        # tracking in the stop-space envelope and physical completion deadline.
        stop['response_time_s'] += 2. * cfg['dt_s']
        zero_duration = (cfg['vehicles'][0]['initial_speed_mps']
                         / stop['comfortable_deceleration_m_s2'] + 2. * cfg['dt_s'])
        stop['stop_deadline_s'] = stop['onset_s'] + stop['response_time_s'] + zero_duration
    if cfg.get('pedestrians'):
        configure_pedestrian_feedback(cfg, conflict, geometry)
        design=cfg.get('interaction_design',{})
        if (design.get('pedestrian_encounter',{}).get('version')=='body-relative-arrival-v2'
                and design['requested_phase'] in ('entering','crossing')):
            nominal_pedestrian_entry(cfg,geometry)
    if cfg.get('interaction_design'):
        from interaction_design import action_timing
        action_timing(cfg,template,cfg['interaction_design']['requested_phase'])
    validate_joint_pedestrian_entry(cfg,conflict)
    for profile in action['profiles'].values():
        if profile['kind'] in ('stop_hold', 'yield_wait'):
            profile['resume_speed_m_s'] = action['nominal_policy']['desired_speed_m_s']
    action['profile_order'] = list(action['profiles'])
    action['design_evidence'] = {'template': template, 'future_window_s': cfg['duration_s'],
        'target_speed_estimates_at_initial_speed': speed_profile_estimates(
            action['profiles'], cfg['vehicles'][0]['initial_speed_mps'], cfg['duration_s']),
        'actual_t0_recheck_required': True, 'actual_trajectory_dedup_required': False,
        'equal_realized_trajectories_retained_in_scene': True,
        'action_design_version': 'five-effective-ego-actions-v1',
        'all_ten_pairs_must_pass_before_acceptance': True,
        'failed_scene_raw_branches_retained': True,
        'equal_realized_trajectories_are_distinct_samples': False,
        'risk_outcomes_retained': True, 'npc_policy_same_for_all_branches': True}
    for name, profile in action['profiles'].items():
        if profile['kind'] == 'lane_change':
            end = (profile['onset_s'] + profile['duration_s']
                   + profile.get('settling_time_s', 0.) + profile['required_final_hold_s'])
            if end > cfg['duration_s'] + 1e-9:
                raise ValueError(f'{name}: declared transition, settling and hold exceed future window')
    return action


def configure_pedestrian_feedback(cfg, conflict, geometry):
    """Freeze feedback conflict geometry before any branch exists or executes."""
    if __package__:
        from .runtime.pedestrian_policy import crossing_geometry
        from .runtime.pedestrian_crossing import KIND_V4 as KIND, DEFAULTS_V3 as DEFAULTS
    else:
        from runtime.pedestrian_policy import crossing_geometry
        from runtime.pedestrian_crossing import KIND_V4 as KIND, DEFAULTS_V3 as DEFAULTS
    for pedestrian in cfg['pedestrians']:
        path = np.asarray(pedestrian['route_points'],float)
        length = np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1)
        cumulative = np.r_[0.,np.cumsum(length)]
        vehicle_conflicts, pedestrian_intervals = [], []
        for actor in cfg['vehicles']:
            crossing_geometry_row = crossing_geometry(actor['route'],path)
            if crossing_geometry_row is None:
                continue
            crossing = crossing_geometry_row['pedestrian_s']
            station = crossing_geometry_row['vehicle_s']
            before,after = _interval(actor['route'],station,2.,2.,geometry)
            vehicle_conflicts.append({'role':actor['role'],'entry_s':before,'exit_s':after})
            pedestrian_intervals.append((max(0.,crossing-1.8),min(float(cumulative[-1]),crossing+1.8)))
        if not vehicle_conflicts or not any(row['role']=='ego' for row in vehicle_conflicts):
            raise ValueError('Reactive pedestrian route must intersect the actual ego corridor')
        pedestrian['response_policy'] = {'kind':KIND,'crossing_feedback':{**DEFAULTS,**cfg.get('randomization',{}).get('pedestrian_policy',{})},'vehicle_conflicts':vehicle_conflicts,
            'pedestrian_entry_s':min(a for a,b in pedestrian_intervals),
            'pedestrian_exit_s':max(b for a,b in pedestrian_intervals),
            'clearance_time_s':1.,'clearance_distance_m':1.,
            'stopped_speed_m_s':.3,'yielding_speed_m_s':2.,'minimum_deceleration_m_s2':.5}
        pedestrian['reactive_to_ego'] = True
        pedestrian['policy'] = KIND+'; shared current-state feedback and recorded initialization, no branch or future trajectory access'
