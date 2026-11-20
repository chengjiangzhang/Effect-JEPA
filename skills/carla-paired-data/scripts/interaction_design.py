"""Prospective encounter strata; sampled once, never from branch outcomes.

Requested stages are design intents. Measured t0 stages and future response
coverage are reported separately; neither is fabricated from these labels.
"""
from __future__ import annotations
import copy
import math
import numpy as np

VERSION = 'interaction-design-v5'
from scene_randomization import parameters, actor_rng
PEDESTRIAN_PACES = {'unhurried': (1.0, 1.2), 'ordinary': (1.2, 1.5), 'brisk': (1.5, 1.8)}
PHASES = {
    'follower_response_straight': ('spaced', 'closing', 'slowing', 'platoon'),
    'ego_lane_change_straight': ('wide_gap', 'closing_rear', 'tight_gap', 'later_entry'),
    'merge_straight': ('waiting_gap', 'approach_lateral', 'starting_merge', 'partial_merge'),
    'merge_ramp': ('ego_first', 'near_together', 'npc_first', 'continuous_stream'),
    'ego_merge_ramp': ('ego_first', 'near_together', 'npc_first', 'continuous_stream'),
    'cross_unsignalized': ('waiting', 'starting', 'approaching', 'entering'),
    'cross_signalized': ('queue_release', 'moving_release', 'clearance_early', 'clearance_late'),
    'left_unsignalized': ('wide_gap', 'near_together', 'oncoming_first', 'continuous_stream'),
    'left_signalized': ('long_green', 'late_green', 'platoon_gap', 'continuous_stream'),
    'pedestrian_straight': ('curb_wait', 'entering', 'crossing', 'clearing'),
    'pedestrian_unsignalized': ('curb_wait', 'entering', 'crossing', 'clearing'),
    'narrow_meeting': ('approaching', 'waiting_gap', 'entered', 'clearing'),
}


def phase_for(job, template):
    phase = job.get('interaction_phase', PHASES[template][job['seed'] % len(PHASES[template])])
    if phase not in PHASES[template]:
        raise ValueError(f'Unsupported interaction phase {phase!r} for {template}')
    return phase


def assign_phases(jobs, cells, seed):
    templates = {c['cell_id']: c['template'] for c in cells}
    counts = {}
    for job in jobs:
        template = templates[job['cell_id']]
        if template not in PHASES:
            continue  # legacy/example grids retain their own contract
        count = counts.get(template, 0)
        job['interaction_phase'] = PHASES[template][(count + seed) % len(PHASES[template])]
        if template.startswith('pedestrian_') and job['interaction_phase'] in ('entering','crossing'):
            # Cross initial stage with walking pace before capture. Three full
            # four-stage cycles cover all pace strata without outcome sampling.
            job['pedestrian_pace'] = tuple(PEDESTRIAN_PACES)[(count//4+seed//4)%3]
        counts[template] = count+1


def pedestrian_pace_for(job, template, phase):
    value = job.get('pedestrian_pace')
    applicable = template.startswith('pedestrian_') and phase in ('entering','crossing')
    if 'pedestrian_pace' in job and (not isinstance(value,str) or not applicable or value not in PEDESTRIAN_PACES):
        raise ValueError('pedestrian_pace requires an entering/crossing pedestrian scene and a declared pace')
    return (value or tuple(PEDESTRIAN_PACES)[job['seed']%3]) if applicable else None


def freeze_job_strata(job, template):
    """Resolve legacy defaults from the slot seed before any candidate reseed."""
    result = copy.deepcopy(job)
    if template in PHASES:
        phase = phase_for(job,template)
        result['interaction_phase'] = phase
        pace = pedestrian_pace_for(job,template,phase)
        if pace is not None:
            result['pedestrian_pace'] = pace
    return result


def place(actor, station, geometry):
    end = float(geometry.cumulative(actor['route'])[-1])
    if not 5. <= station <= end-20.:
        raise ValueError(f"{actor['role']}: encounter layout exceeds mapped route")
    actor['spawn_route_s'] = float(station)
    actor['transform'] = geometry.point_at_s(actor['route'], float(station))


def road_encounter(cfg, template, variant, rng, geometry):
    """Place the whole relevant lane queue, including its nearest front car."""
    ego = cfg['vehicles'][0]
    slots = {s['role']: s for s in cfg['geometry']['traffic_design']['roles_and_slots']}
    lanes = {}
    for actor in cfg['vehicles'][1:]:
        lane = slots[actor['role']]['route_lane']
        if template == 'follower_response_straight' and lane != 'main':
            continue
        own_s = geometry.city_project_s(actor['route'], ego['transform']['position'])
        gap = actor['spawn_route_s']-own_s
        lanes.setdefault(lane, []).append((gap, actor, own_s))
    for lane, actors in lanes.items():
        rear = sorted((a for a in actors if a[0] < 0 and a[1]['role'] != 'parallel_vehicle'), key=lambda a: -a[0])
        front = sorted((a for a in actors if a[0] >= 0 or a[1]['role'] == 'parallel_vehicle'), key=lambda a: a[0])
        following = template == 'follower_response_straight' or lane == 'main'
        near_rear = ((22., 18., 13., 11.5) if following else (25., 18., 16., 21.))[variant]
        near_front = (24., 18., 12., 20.)[variant]
        spacing = (28., 23., 25., 12.)[variant]
        sampled = parameters(cfg)
        if sampled:
            near_rear,near_front,spacing = (sampled[x] for x in ('rear_gap_m','front_gap_m','queue_gap_m'))
        for is_rear, group in ((True, rear), (False, front)):
            actual_spacing = spacing
            if len(group) > 1:
                room = min((own_s-5. if is_rear else float(geometry.cumulative(actor['route'])[-1])-20.-own_s)
                           for _,actor,own_s in group)
                actual_spacing = min(spacing,(room-(near_rear if is_rear else near_front)-2.)/(len(group)-1))
                if actual_spacing < 11.:
                    raise ValueError('Mapped lane cannot contain the selected queue with disjoint bodies')
            for rank, (_, actor, own_s) in enumerate(group):
                local = actor_rng(cfg,actor['role'],'road_layout') if sampled else rng
                gap = (near_rear if is_rear else near_front)+rank*actual_spacing+local.uniform(0., 2.)
                gap *= -1. if is_rear else 1.
                place(actor, own_s+gap, geometry)
                relative = ((-.1, 1.8, -.5, .5)[variant] if is_rear else rng.uniform(.1, .7))
                if sampled:
                    relative = sampled['rear_relative_speed_m_s' if is_rear else 'front_relative_speed_m_s']+local.uniform(-.25,.25)
                actor['initial_speed_mps'] = max(.5, ego['initial_speed_mps']+relative)
                actor['desired_speed_mps'] = max(.5, ego['initial_speed_mps']+(
                    (-.9 if variant == 2 else .8) if is_rear else relative))
                if sampled:
                    actor['desired_speed_mps'] = max(.5,actor['initial_speed_mps']+local.uniform(-.5,.9))
                    if template=='follower_response_straight' and variant==2 and is_rear:
                        actor['desired_speed_mps']=max(.5,actor['initial_speed_mps']-local.uniform(.4,1.4))
                slot = slots[actor['role']]
                slot.update(gap_from_ego_station_m=float(gap), relative_initial_speed_mps=float(relative),
                            desired_speed_mps=actor['desired_speed_mps'])
                slot.pop('nominal_t0_longitudinal_offset_m', None)
                if actor['role'] == 'parallel_vehicle':
                    slot['relation'] = 'nearest front vehicle in a jointly authored target-lane gap'
    cfg['geometry']['traffic_design']['approach_opportunities'] = [
        {'role': actor['role'], 'route_lane': slots[actor['role']]['route_lane'],
         'spawn_gap_m': slots[actor['role']]['gap_from_ego_station_m'],
         'relative_initial_speed_mps': actor['initial_speed_mps']-ego['initial_speed_mps'],
         'actual_t0_requires_capture_audit': True} for actor in cfg['vehicles'][1:]]


def diversify(cfg, template, phase, rng, geometry, pedestrian_pace=None):
    """Vary physical initial parameters before actions and measured-t0 planning."""
    cars = cfg['vehicles']; ego = cars[0]; pre = cfg['history_s']+cfg['driving_warmup_s']
    k = PHASES[template].index(phase)
    sampled = parameters(cfg)
    design = {'version': VERSION, 'requested_phase': phase,
              'phase_is_measured': False, 'uses_future_outcomes': False,
              'scene_parameters_shared_across_branches': True}
    cfg['interaction_design'] = design
    if template in ('follower_response_straight', 'ego_lane_change_straight'):
        road_encounter(cfg, template, k, rng, geometry)
    elif template.startswith('pedestrian_'):
        ped = cfg['pedestrians'][0]
        if k in (1,2):
            # A close low-speed encounter leaves room for the delayed stop,
            # while a subsequent finite acceleration can change the observed
            # gap before the pedestrian has already cleared the lane.
            ego['initial_speed_mps']=sampled.get('ego_initial_speed_m_s',float(rng.uniform(3.0,4.2)))
            ego['desired_speed_mps']=ego['initial_speed_mps']
        ped['start_time_s'] = (0., -.65, -1.3, -2.)[k]
        if k in (1,2):
            pace = pedestrian_pace or tuple(PEDESTRIAN_PACES)[cfg['seed']%3]
            bounds = PEDESTRIAN_PACES[pace]
            ped['speed_mps'] = float(rng.uniform(*bounds))
            design['pedestrian_pace'] = {'stratum':pace,'nominal_speed_range_m_s':list(bounds),
                                         'sampled_speed_m_s':ped['speed_mps'],
                                         'assigned_before_capture':True,'required_response':None}
        else:
            ped['speed_mps'] = float(rng.uniform(1.1,1.75))
        if sampled:
            ped['speed_mps']=sampled['pedestrian_speed_m_s']
            if 'pedestrian_pace' in design:
                design['pedestrian_pace']['sampled_speed_m_s']=ped['speed_mps']
        # Sample encounter conditions before rollout, never a required response.
        cfg['interaction_design']['pedestrian_encounter'] = {
            'version':'body-relative-arrival-v2',
            'entry_body_setback_m':float(rng.uniform(.6,1.1)),
            'crossing_body_progress_m':float(rng.uniform(-.25,.1)),
            'arrival_slack_s':float(rng.uniform(.10,.30)),
            'acceleration_onset_s':float(rng.choice((0.,.2))) if k in (1,2) else 0.,
            'acceleration_m_s2':float(rng.uniform(3.,3.8)) if k in (1,2) else 1.5,
            'outcome_used_for_assignment':False}
        if sampled:
            encounter=design['pedestrian_encounter']
            for key in ('entry_body_setback_m','crossing_body_progress_m','arrival_slack_s'):
                encounter[key]=sampled[key]
            encounter['clearing_body_remaining_m']=sampled['clearing_body_remaining_m']
            if k in (1,2):
                encounter['acceleration_onset_s']=sampled['walker_acceleration_onset_s']
                encounter['acceleration_m_s2']=sampled['walker_encounter_acceleration_m_s2']
        ped['initial_opportunity_design'].update(requested_phase=phase, earliest_decision_s=ped['start_time_s'])
        endpoints = ped['geometry_evidence']['sidewalk_endpoints']
        origin = np.asarray(cfg['geometry']['origin'])
        reverse = cfg['randomization']['participants']['preferred_pedestrian_direction']==1 if sampled else k in (1,3)
        from walker_spawn import crossing_path, direction_for
        if direction_for(endpoints, origin, reverse):
            ped['route_points'], ped['transform'], curb = crossing_path(endpoints, 1)
            ped['crossing_direction_index'] = 1
            ped['initial_opportunity_design'].update(sidewalk_start='opposite_mapped_side_within_five_metres',toward_curb_offset_m=curb)
        # Retain the sampled actors, varying their queue placement. The
        # actual number near the encounter is measured after common history.
        offsets = ([0., -35., -53., -71., 35., 58., 82.],
                   [0., -24., -45., -66., 32., 59., 85.],
                   [0., -32., -50., -68., 45., 68., 91.],
                   [0., -28., -46., -64., 48., 73., 96.])[k]
        ranks={-1:0,1:0}
        for i, actor in enumerate(cars[1:], 1):
            local=actor_rng(cfg,actor['role'],'pedestrian_traffic') if sampled else rng
            gap=offsets[i]+rng.uniform(-2.,2.)
            if sampled:
                side=-1 if actor['spawn_route_s']<ego['spawn_route_s'] else 1
                gap=side*(sampled['pedestrian_rear_gap_m' if side<0 else 'pedestrian_front_gap_m']+ranks[side]*sampled['queue_gap_m']+local.uniform(0.,2.))
                ranks[side]+=1
            place(actor,ego['spawn_route_s']+gap,geometry)
            actor['initial_speed_mps'] = max(.2,ego['initial_speed_mps']+rng.uniform(-1.,1.))
            actor['desired_speed_mps'] = max(.5,actor['initial_speed_mps']+rng.uniform(-.3,.6))
    elif template == 'narrow_meeting':
        zone = cfg['geometry']['narrow_meeting']
        # Width is measured from map; simultaneous passage is decided from
        # observed body widths AND separation, not just this map label.
        zone.update(policy_version='narrow-current-occupancy-v2',
                    lateral_margin_m=.35, release_confirmation_s=.4,
                    passing_speed_m_s=float(rng.uniform(2.5,3.5)))
        roles = cfg['vehicle_roles']
        for side in ('ego', 'oncoming'):
            stream = [a for a in cars if (a['role']=='ego' or a['role'].startswith('ego_follower')) == (side=='ego')]
            head = stream[0]
            gap = sampled.get('queue_gap_m',(29.,42.,20.,35.)[k])
            reserve=(5.-min(0.,sampled['narrow_oncoming_shift_m']) if side=='oncoming' else 5.) if sampled else 30.
            gap = min(gap,(head['spawn_route_s']-reserve)/max(1,len(stream)-1))
            if len(stream)>1 and gap < 11.:
                raise ValueError('Mapped narrow approach cannot contain the selected queue')
            for i, actor in enumerate(stream[1:],1):
                place(actor,head['spawn_route_s']-i*gap+rng.uniform(-1.,1.),geometry)
        # Move the opposing stream's encounter phase, preserving its own gaps.
        oncoming = [a for a in cars if a['role'].startswith('oncoming')]
        shift = sampled.get('narrow_oncoming_shift_m',(-4., -10., 7., 13.)[k])
        for actor in oncoming:
            place(actor,actor['spawn_route_s']+shift,geometry)
    elif template == 'left_signalized':
        yellow = sampled.get('signal_yellow_s',(3.5,1.3,2.7,2.)[k])
        cfg['traffic_signal_program']['phases'] = [
            {'start_time_s':-100.,'states':{'ego':'Green','npc':'Green'}},
            {'start_time_s':yellow,'states':{'ego':'Yellow','npc':'Yellow'}},
            {'start_time_s':yellow+3.,'states':{'ego':'Red','npc':'Red'}}]
        cfg['traffic_signal_program']['policy'] = 'mapped unprotected left turn with opposing permissive green; varying remaining green'
    elif template == 'cross_signalized':
        # Each scenario freezes the same programme for all branches. Preserve
        # yellow/all-red durations while varying actual queue release phase.
        green = sampled.get('npc_green_s',(-.1,-.8,-1.4,.45)[k])
        plan=cfg['geometry']['traffic_design']['signal_encounter_plan']
        plan['base_green_time_s']=plan['npc_green_time_s']
        plan['npc_green_time_s'] = green
        entry=plan['npc_optimistic_body_entry_after_green_s']+green
        plan['npc_optimistic_body_entry_s']=entry
        plan['npc_optimistic_body_entry_tick_s']=float(np.ceil(entry/cfg['dt_s']-1e-8)*cfg['dt_s'])
        plan['base_schedule_distance_bounds_retained_as_reference']=True
        plan['basis']='Mapped committed-ego and queued-NPC poses; actual frozen release time and optimistic entry recomputed; measured t0 is the admission check'
        cfg['traffic_signal_program']['phases'] = [
            {'start_time_s':-100.,'states':{'ego':'Green','npc':'Red'}},
            {'start_time_s':green-3.5,'states':{'ego':'Yellow','npc':'Red'}},
            {'start_time_s':green-.5,'states':{'ego':'Red','npc':'Red'}},
            {'start_time_s':green,'states':{'ego':'Red','npc':'Green'}}]
        cfg['traffic_signal_program']['policy'] = 'mapped committed-ego clearance with variable cross-queue release; initial entry legality unobserved'
    design['initial_actor_parameters'] = [
        {key:copy.deepcopy(actor[key]) for key in ('role','spawn_route_s','initial_speed_mps','desired_speed_mps')}
        for actor in cars]


def action_timing(cfg, template, phase):
    """Keep action kinds/branch compatibility; explicitly version their timing."""
    profiles = cfg['action']['profiles']; k = PHASES[template].index(phase)
    sampled = parameters(cfg)
    for p in profiles.values():
        if p['kind']=='longitudinal':
            p['acceleration_m_s2'] *= sampled.get('action_acceleration_scale',(.9,1.1,.95,1.05)[k])
    encounter = cfg.get('interaction_design',{}).get('pedestrian_encounter')
    if encounter and 'accelerate' in profiles:
        profiles['accelerate'].update(onset_s=encounter['acceleration_onset_s'],
            acceleration_m_s2=encounter['acceleration_m_s2'])
    if template=='follower_response_straight':
        profiles['brake_delayed']['onset_s']=sampled.get('delayed_brake_onset_s',(.45,.6,.8,.7)[k])
    if template == 'ego_lane_change_straight':
        for p in profiles.values():
            if p['kind'] == 'lane_change':
                p['onset_s'] = sampled.get('lane_change_onset_s',(0., .15, .25, .35)[k])
    for profile in profiles.values():
        if profile['kind'] in ('stop_hold','yield_wait'):
            # A zero speed target still needs service braking until the car
            # actually stops. Yield keeps its earlier, nonzero approach ramp
            # and rolling release; the clamp only applies during forced hold.
            profile['terminal_brake_min'] = 1.
    # Delay only when a declared stop has enough prospective space. Signal
    # clearance scenes have a tighter stop envelope and retain immediate hold.
    if template.startswith('pedestrian_') or template in ('left_unsignalized','narrow_meeting'):
        stop = profiles.get('stop_hold')
        if stop:
            delay = sampled.get('stop_delay_s',(.35,.5,.65,.45)[k])
            deadline = delay + 2.25
            turn = cfg.get('geometry', {}).get('traffic_design', {}).get('turn_speed_plan')
            if template == 'left_unsignalized' and turn and turn['approach_m_s'] < turn['proposed_approach_m_s']:
                # A fixed 1.5 s target-speed ramp gives a low-speed tight turn
                # a gentle stop. Keep the sampled delay and a firm braking ramp.
                dt = cfg['dt_s']
                ramp = math.ceil((turn['approach_m_s'] / 4. + 2. * dt) / dt) * dt
                deadline = min(deadline, delay + stop['response_time_s'] + ramp)
            stop.update(onset_s=delay,stop_deadline_s=deadline)
            cfg['action']['timing_semantics'] = 't0 selects the complete policy; stop_hold waits before stronger braking, yield_wait starts earlier'
    cfg['action']['timing_design_version'] = VERSION
