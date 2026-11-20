"""Bounded, prospective scene sampling with independently named random streams.

An intent is a proposal, not measured t0 or a promised response. It is persisted
before compilation so that infeasible proposals remain in the denominator.
There are no rollout, branch-name or response inputs to this sampler.
"""
from __future__ import annotations
import copy
import hashlib
import json
import numpy as np

VERSION = 'joint-scene-randomization-v1'


def active_parameters(template, phase):
    keys={'action_acceleration_scale'}
    if template!='cross_signalized':keys.add('ego_initial_speed_m_s')
    if template in ('follower_response_straight','ego_lane_change_straight','merge_straight'):
        keys.add('road_anchor_shift_m')
        if template!='merge_straight':
            keys.update(('rear_gap_m','front_gap_m','queue_gap_m','rear_relative_speed_m_s','front_relative_speed_m_s'))
        keys.add({'follower_response_straight':'delayed_brake_onset_s','ego_lane_change_straight':'lane_change_onset_s','merge_straight':'merge_onset_s'}[template])
    elif template.startswith('pedestrian_'):
        keys.update(('pedestrian_speed_m_s','pedestrian_rear_gap_m','pedestrian_front_gap_m','queue_gap_m','stop_delay_s'))
        if phase in ('entering','crossing'):
            keys.update(('entry_body_setback_m' if phase=='entering' else 'crossing_body_progress_m',
                'arrival_slack_s','walker_acceleration_onset_s','walker_encounter_acceleration_m_s2'))
        if phase=='clearing':keys.add('clearing_body_remaining_m')
    elif template=='narrow_meeting':
        keys.update(('other_initial_speed_m_s','queue_gap_m','narrow_oncoming_shift_m','stop_delay_s'))
    elif template=='cross_signalized':keys.update(('queue_gap_m','npc_green_s'))
    else:
        keys.update(('other_initial_speed_m_s','ego_nominal_arrival_s','other_minus_ego_arrival_s','queue_gap_m'))
        if template=='left_signalized':keys.add('signal_yellow_s')
        if template=='left_unsignalized':keys.add('stop_delay_s')
    return sorted(keys)


def stream(seed, name):
    value = hashlib.sha256(f'{VERSION}/{int(seed)}/{name}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(value[:8], 'big'))


def intent(job, cell, selection):
    from interaction_design import phase_for, pedestrian_pace_for, PEDESTRIAN_PACES
    template = cell['template']; phase = phase_for(job, template)
    seed = int(selection['layout_seed'])
    def uniform(name, bounds):
        return float(stream(seed, name).uniform(*bounds))
    def integer(name, low, high):
        return int(stream(seed, name).integers(low, high + 1))
    ped = template.startswith('pedestrian_')
    bounds = ((3.,4.3) if phase in ('entering','crossing') else (4.2,6.)) if ped else (
        (3.6,5.5) if template == 'narrow_meeting' else (4.2,7.5))
    pace = pedestrian_pace_for(job, template, phase)
    rear_bounds = (22.,36.) if phase in ('spaced','wide_gap') else (
        (12.,19.) if phase in ('tight_gap','platoon') else (14.,30.))
    relative_bounds = (1.,2.5) if phase in ('closing','closing_rear') else (-1.1,1.6)
    arrival_bounds = {'ego_first':(.6,1.5),'npc_first':(-1.4,-.6),
        'near_together':(-.35,.35),'oncoming_first':(-1.4,-.6),
        'wide_gap':(.8,1.6)}.get(phase,(-1.1,1.3))
    merge_bounds = {'waiting_gap':(.65,1.3),'approach_lateral':(.05,.55),
        'starting_merge':(-.6,-.05),'partial_merge':(-1.45,-.8)}.get(phase,(.1,.7))
    shift_bounds = {'approaching':(-8.,1.),'waiting_gap':(-14.,-7.),
        'entered':(3.,10.),'clearing':(10.,17.)}.get(phase,(-4.,4.))
    green_bounds = {'queue_release':(-.35,.1),'moving_release':(-1.1,-.5),
        'clearance_early':(-1.6,-1.1),'clearance_late':(.15,.65)}.get(phase,(-.5,.5))
    yellow_bounds = {'long_green':(3.1,3.9),'late_green':(.9,1.7),
        'platoon_gap':(2.1,3.3),'continuous_stream':(1.5,2.6)}.get(phase,(1.,3.8))
    parameters = {
        'ego_initial_speed_m_s':uniform('ego_speed',bounds),
        'other_initial_speed_m_s':uniform('other_speed',(3.2,5.7) if template=='narrow_meeting' else (3.5,7.5)),
        'ego_nominal_arrival_s':uniform('ego_arrival',(1.8,2.9)),
        'other_minus_ego_arrival_s':uniform('arrival_difference',arrival_bounds),
        'rear_gap_m':uniform('rear_gap',rear_bounds),
        'front_gap_m':uniform('front_gap',(12.,34.)),
        'queue_gap_m':uniform('queue_gap',(12.,32.)),
        'rear_relative_speed_m_s':uniform('relative_speed',relative_bounds),
        'front_relative_speed_m_s':uniform('front_relative_speed',(-1.,.9)),
        'road_anchor_shift_m':uniform('anchor_shift',(-9.,9.)),
        'merge_onset_s':uniform('merge_progress',merge_bounds),
        'narrow_oncoming_shift_m':uniform('narrow_progress',shift_bounds),
        'npc_green_s':uniform('signal_release',green_bounds),
        'signal_yellow_s':uniform('signal_remaining',yellow_bounds),
        'stop_delay_s':uniform('stop_timing',(.3,.65)),
        'lane_change_onset_s':uniform('lane_change_timing',(.2,.35) if phase=='later_entry' else (0.,.25)),
        'delayed_brake_onset_s':uniform('follower_brake_timing',(.4,.8)),
        'action_acceleration_scale':uniform('action_strength',(.9,1.1)),
        'pedestrian_speed_m_s':uniform('walker_speed',PEDESTRIAN_PACES[pace] if pace else (1.1,1.75)),
        'entry_body_setback_m':uniform('walker_entry_progress',(.6,1.1)),
        'crossing_body_progress_m':uniform('walker_crossing_progress',(-.25,.15)),
        'clearing_body_remaining_m':uniform('walker_clearance_progress',(.3,.9)),
        'arrival_slack_s':uniform('walker_arrival_slack',(.1,.3)),
        'walker_acceleration_onset_s':uniform('walker_encounter_action_timing',(0.,.2)),
        'walker_encounter_acceleration_m_s2':uniform('walker_encounter_action_strength',(3.,3.8)),
        'pedestrian_rear_gap_m':uniform('walker_rear_gap',(24.,45.)),
        'pedestrian_front_gap_m':uniform('walker_front_gap',(30.,50.)),
    }
    count = integer('participant_count',1 if ped else (3 if phase in ('platoon','continuous_stream') else 2),7)
    if template=='ego_lane_change_straight' and phase in ('tight_gap','closing_rear'):
        count=max(3,count)
    # Separate the count from which side has a following queue. A narrow road
    # always retains its two heads; extra actors need not belong to ego's side.
    ego_followers = integer('narrow_ego_followers',0,count-2) if not ped else 0
    if template=='cross_unsignalized':
        other_eta=uniform('crossing_progress',{'waiting':(2.5,3.5),'starting':(1.5,2.5),
            'approaching':(.5,1.5),'entering':(-.3,.5)}[phase])
        parameters['other_minus_ego_arrival_s']=other_eta-parameters['ego_nominal_arrival_s']
    return {'version':VERSION,'seed':seed,'template':template,'phase':phase,
        'pedestrian_pace':pace,'parameters':parameters,'active_parameters':active_parameters(template,phase),
        'participants':{'vehicle_count_including_ego':count,
            'narrow_ego_followers':ego_followers,'narrow_oncoming_followers':max(0,count-2-ego_followers),
            'target_lane_rear':bool(integer('target_lane_rear',0,1)) and count>=3,
            'preferred_pedestrian_direction':integer('walker_direction',0,1)},
        'pedestrian_policy':{'acceleration_m_s2':uniform('walker_acceleration',(1.35,1.65)),
            'resume_clear_s':uniform('walker_confirmation',(.3,.55)),
            'fast_speed_ratio':uniform('walker_fast_ratio',(1.4,1.65))},
        'distribution':'bounded engineering distribution; not calibrated human population',
        'same_for_all_branches':True,'uses_future_outcomes':False,
        'stream_rule':'sha256(version, layout_seed, component or actor role); no sequential global RNG',
        'constraints':'mapped routes, initial disjoint bodies, native common history, five-action execution and original QA'}


def parameters(cfg):
    return cfg.get('randomization',{}).get('parameters',{})


def actor_rng(cfg, role, purpose):
    return stream(cfg['randomization']['seed'],f'actor/{role}/{purpose}')


def individualize(cfg, geometry):
    """Stable per-person preferences; safety envelopes/QA remain unchanged."""
    if 'randomization' not in cfg:
        return
    from scene_spec import actor_policy, NPC_BOUNDS
    for actor in cfg['vehicles']:
        role = actor['role']
        if role != 'ego':
            policy = actor_policy(cfg,actor)
            for key in ('max_acceleration','comfortable_brake','time_headway','minimum_gap'):
                low,high = NPC_BOUNDS[key]
                policy['idm'][key] = float(np.clip(policy['idm'][key] * actor_rng(cfg,role,key).uniform(.8,1.2),low,high))
            for key in ('conflict_time_gap_s','yield_lookahead_s'):
                low,high = NPC_BOUNDS[key]
                policy[key] = float(np.clip(policy[key] * actor_rng(cfg,role,key).uniform(.85,1.15),low,high))
            actor['npc_policy'] = policy
            actor['desired_speed_mps'] = float(np.clip(actor['desired_speed_mps'] *
                actor_rng(cfg,role,'desired_speed').uniform(.92,1.08),.5,15.))
            actor['blueprint_id'] = str(actor_rng(cfg,role,'blueprint').choice(geometry.DIVERSE_NPC_MODELS))
        actor['blueprint_attributes']['color'] = str(actor_rng(cfg,role,'color').choice(geometry.DIVERSE_COLORS))
    cfg['randomization']['individual_policy_semantics'] = (
        'NPC style is a population centre, not an identical scene-wide policy; individual parameters are fixed across time and all branches, possibly latent to a two-second RGB history')


def snapshot(cfg, geometry, cmap=None):
    """Final authored inputs, after physical adjustments; distinct from intent."""
    if 'randomization' not in cfg:
        return
    from scene_spec import actor_policy
    ego = cfg['vehicles'][0]
    rows=[]
    for actor in cfg['vehicles']:
        projected=geometry.city_project_s(actor['route'],ego['transform']['position'])
        rows.append({'role':actor['role'],'spawn_route_s':actor['spawn_route_s'],
            'initial_speed_m_s':actor['initial_speed_mps'],
            'relative_initial_speed_m_s':actor['initial_speed_mps']-ego['initial_speed_mps'],
            'ego_projected_route_gap_m':actor['spawn_route_s']-projected,
            'gap_semantics':'projected onto actor route, not a shared-lane distance claim',
            'desired_speed_m_s':actor['desired_speed_mps'], 'blueprint_id':actor['blueprint_id'],
            'color':actor['blueprint_attributes'].get('color'),
            'policy':actor_policy(cfg,actor) if actor['role']!='ego' else None})
    if cmap is not None:
        import carla
        chains={}
        for actor in cfg['vehicles']:
            chain=[]
            route=actor['route']
            for point in route[::5]+[route[-1]]:
                wp=cmap.get_waypoint(carla.Location(x=float(point[0]),y=float(point[1]),z=float(point[2])),
                    project_to_road=True,lane_type=carla.LaneType.Driving)
                key=[int(wp.road_id),int(wp.lane_id)] if wp else None
                if not chain or chain[-1]!=key:chain.append(key)
            chains[actor['role']]=chain
        cfg['randomization']['mapped_route_combination']={'map':cfg['map'],'actor_road_lane_chains':chains,
            'basis':'sampled route points projected to actual map Driving lanes; identity diagnostic, not proof of legal boundary crossing'}
    cfg['randomization']['effective']={'vehicles':rows,'stage':'initial_compilation_after_authoring',
        'pedestrians':copy.deepcopy(cfg.get('pedestrians',[])),
        'action_profiles':copy.deepcopy(cfg['action']['profiles']),
        'history_s':cfg['history_s'],'future_s':cfg['duration_s'],
        'common_motion_s':cfg['history_s']+cfg['driving_warmup_s'],
        'signals':copy.deepcopy(cfg.get('traffic_signal_program')),
        'measured_t0':False,'adjustment_basis':'map geometry and prospective stopping/encounter budgets; native t0 checked separately'}
    cfg['geometry']['traffic_design']['vehicle_count_including_ego']=len(cfg['vehicles'])
