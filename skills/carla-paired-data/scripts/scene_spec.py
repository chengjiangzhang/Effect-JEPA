"""Editable scene specifications resolved against immutable map routes.

Agent edits are declarative data. Spawn station is measured in metres on a
listed route, and world pose is derived from that route; arbitrary off-road XYZ
and executable policy code are deliberately outside this contract.
"""
from __future__ import annotations
import copy
import itertools
import math
import numpy as np


def obj(properties, required=None):
    return {'type':'object','additionalProperties':False,'properties':properties,
            'required':list(properties) if required is None else required}


def num(low, high):
    return {'type':'number','minimum':low,'maximum':high}


NPC_BOUNDS = {
    'max_acceleration':(.2,5.), 'comfortable_brake':(.5,8.),
    'time_headway':(.3,4.), 'minimum_gap':(.5,10.),
    'lateral_margin_m':(.05,1.5), 'conflict_time_gap_s':(.2,6.),
    'yield_lookahead_s':(.5,8.), 'collision_lookahead_s':(.1,4.),
    'collision_margin_m':(0.,1.), 'max_emergency_deceleration_m_s2':(1.,10.)}
IDM = ('max_acceleration','comfortable_brake','time_headway','minimum_gap')
DEFAULT_NPC = {'lateral_margin_m':.3,'conflict_time_gap_s':1.5,
               'yield_lookahead_s':4.,'collision_lookahead_s':1.5,
               'collision_margin_m':.25,'max_emergency_deceleration_m_s2':8.}
PROFILE_BOUNDS = {'onset_s':(0.,3.9),'acceleration_m_s2':(-8.,4.),
    'speed_increase_cap_m_s':(0.,10.),'min_target_speed_m_s':(0.,15.),
    'duration_s':(1.5,4.),'required_final_hold_s':(.5,1.),
    'settling_time_s':(0.,2.),'stop_deadline_s':(.1,4.),'stop_margin_m':(0.,2.),
    'stop_route_s':(0.,10000.),'comfortable_deceleration_m_s2':(.5,8.),
    'response_time_s':(0.,2.),'terminal_brake_min':(.4,1.),
    'hold_until_s':(0.,4.),'min_wait_s':(0.,4.),'release_gap_s':(.2,6.),
    'timeout_s':(.1,4.),'resume_speed_m_s':(.1,20.)}


def actor_policy(cfg, actor):
    c=cfg['controller']; p=actor.get('npc_policy',{})
    result={k:float(p.get(k,c.get(k,v))) for k,v in DEFAULT_NPC.items()}
    result['idm']={k:float(p.get('idm',{}).get(k,c['idm'][k])) for k in IDM}
    result['leader_geometry']=p.get('leader_geometry',c.get('leader_geometry','route_projection'))
    return result


def authoring_payload(cfg):
    """Compact complete authorable inputs, without huge immutable route arrays."""
    return {'actors':{a['role']:{'spawn_route_s':float(a['spawn_route_s']),
        'initial_speed_mps':float(a['initial_speed_mps']),
        'desired_speed_mps':float(a['desired_speed_mps']),
        'blueprint_id':a['blueprint_id'],
        **({'npc_policy':actor_policy(cfg,a)} if a['role']!='ego' else {})}
        for a in cfg['vehicles']},
        'fork_time_s':float(cfg['history_s']+cfg['driving_warmup_s']),
        'profiles':copy.deepcopy(cfg['action']['profiles']),
        'unrelated_roles':list(cfg.get('response_contract',{}).get('unrelated_roles',[]))}


def _value_schema(value, name):
    if name in PROFILE_BOUNDS:
        return num(*PROFILE_BOUNDS[name])
    if isinstance(value,dict):
        return obj({k:_value_schema(v,k) for k,v in value.items()})
    if isinstance(value,bool): return {'type':'boolean','const':value}
    if isinstance(value,str): return {'type':'string','const':value}
    if isinstance(value,(float,int)): return num(value,value)
    if isinstance(value,list):
        return {'type':'array','minItems':len(value),'maxItems':len(value),
                'items':{'type':'string'},'const':value}
    raise ValueError(f'Unsupported authoring field {name}')


def profile_schema(profile):
    fields={k:_value_schema(v,k) for k,v in profile.items()}
    if profile['kind'] in ('maintain','nominal') and 'onset_s' in fields:
        fields['onset_s']=num(0.,0.)
    if profile['kind'] in ('maintain','lane_change'):
        for key in ('acceleration_m_s2','speed_increase_cap_m_s'):
            if key in fields:fields[key]=num(0.,0.)
    if profile['kind']=='longitudinal':
        if profile['acceleration_m_s2']<0:
            fields['acceleration_m_s2']=num(-8.,-.1)
            fields['speed_increase_cap_m_s']=num(0.,0.)
        else:
            fields['acceleration_m_s2']=num(.1,4.)
    if profile.get('hold_policy')=='hold_until_window_end_v1':
        fields['hold_until_s']=num(4.,4.)
    if profile.get('release_condition')=='specified_actors_cleared_and_entry_clear_v1':
        fields['min_wait_s']=num(0.,0.)
    return obj(fields)


def authoring_schema(cfg):
    from scene_compiler import runtime_modules
    geometry,_=runtime_modules()
    p=authoring_payload(cfg)
    actors={}
    for a in cfg['vehicles']:
        route_end=float(geometry.cumulative(a['route'])[-1])
        upper=route_end-20.
        if a.get('route_maneuver'):
            upper=min(upper,a['route_maneuver']['source_interval_s_m'][0]-5.)
        props={'spawn_route_s':num(2.,upper),'initial_speed_mps':num(0.,15.),
               'desired_speed_mps':num(.1,20.),
               'blueprint_id':{'type':'string','enum':sorted(set(geometry.DIVERSE_NPC_MODELS)|{'vehicle.tesla.model3'})}}
        if a['role']!='ego':
            props['npc_policy']=obj({**{k:num(*NPC_BOUNDS[k]) for k in DEFAULT_NPC},
                'idm':obj({k:num(*NPC_BOUNDS[k]) for k in IDM}),
                'leader_geometry':{'type':'string','enum':['route_projection']}})
        actors[a['role']]=obj(props)
    return obj({'actors':obj(actors),'fork_time_s':num(2.5,5.),
        'profiles':obj({k:profile_schema(v) for k,v in p['profiles'].items()}),
        'unrelated_roles':{'type':'array','items':{'type':'string',**(
            {'enum':[a['role'] for a in cfg['vehicles'][1:]]} if len(cfg['vehicles'])>1 else {})},
                           'maxItems':len(cfg['vehicles'])-1}})


def authoring_context(cfg):
    from scene_compiler import runtime_modules
    geometry,_=runtime_modules()
    return {'base_spec':authoring_payload(cfg),
        'routes':{a['role']:{'length_m':float(geometry.cumulative(a['route'])[-1]),
            'world_route_xyz_yaw':a['route'][::max(1,len(a['route'])//80)],
            'spawn_world_pose':a['transform'],
            'maneuver':a.get('route_maneuver')} for a in cfg['vehicles']},
        'conflicts':cfg['action'].get('conflicts',[]),
        'traffic_signal_program':cfg.get('traffic_signal_program'),
        'frozen_pedestrians':copy.deepcopy(cfg.get('pedestrians',[])),
        'geometry_evidence':cfg['geometry'].get('interaction_evidence'),
        'protocol':{'history_s':2.,'future_s':4.,'physics_dt_s':.05,
                    'fork_origin':'seconds after common controlled motion begins; stored sample times relative to fork'},
        'rules':['Explicitly specify every actor and each of the five action profiles.',
            'Spawn station and velocity are initial conditions before common nominal-policy history.',
            'Keep route topology, participant roles and signal plan fixed; choose another catalog candidate for another road.',
            'NPC parameters are frozen across branches; only ego profiles differ.',
            'Pedestrian feedback policy, route, earliest decision time, and wait_for_roles are fixed for this candidate; do not drop waiting targets or replace strict release rules.',
            'Waiting must leave the entire ego body outside the target actors swept routes, not merely put its centre before a crossing point.',
            'stop_route_s is the latest permitted centre station, not a destination that must be reached; a deadline may stop ego upstream.',
            'Stopping deadlines reserve response_time_s and are checked against measured t0 speed; lane settling_time_s is separate from the unchanged final hold.',
            'Weak/no NPC reaction, correctly sustained waiting and action-caused collisions are valid outcomes.',
            'Do not relax execution tolerances or relabel previously declared unrelated actors to hide a failure.']}


def apply_authoring(cfg, proposal, geometry):
    from model_api import check_schema,validate_json
    schema=authoring_schema(cfg);check_schema(schema);validate_json(proposal,schema)
    fork=float(proposal['fork_time_s'])
    if abs(fork/.05-round(fork/.05))>1e-7:
        raise ValueError('fork_time_s must lie on the 20 Hz physics clock')
    cfg['driving_warmup_s']=round(fork-2.,8)
    cfg['fork']={'time_s':fork,'origin':'common controlled-motion start',
                 'saved_time_origin':'fork is t=0','history_s':2.,'future_s':4.,
                 'reference_branch':'maintain','repeat_branch':'maintain_repeat'}
    for a in cfg['vehicles']:
        patch=proposal['actors'][a['role']]
        a.update(copy.deepcopy(patch))
        a['transform']=geometry.point_at_s(a['route'],a['spawn_route_s'])
        a['initial_state_definition']='mapped route station, route-tangent heading and forward speed before common motion warmup/history'
        if a['role']!='ego':
            a['response_policy_type']='live_observation_route_idm'
            a['response_policy_scope']='identical parameters and implementation in every branch; no branch name or future labels'
    if len(set(proposal['unrelated_roles']))!=len(proposal['unrelated_roles']):
        raise ValueError('unrelated_roles must contain distinct NPC roles')
    if any(geometry.spawn_footprints_overlap(a,b) for a,b in itertools.combinations(cfg['vehicles'],2)):
        raise ValueError('Authored spawn footprints overlap')
    cfg['action']['profiles']=copy.deepcopy(proposal['profiles'])
    cfg['action']['nominal_policy']['desired_speed_m_s']=cfg['vehicles'][0]['desired_speed_mps']
    for name,p in cfg['action']['profiles'].items():
        onset=p.get('onset_s',0.)
        if onset>=4.:raise ValueError(f'{name}: intervention onset is outside the future window')
        if p['kind']=='lane_change' and onset+p['duration_s']+p.get('settling_time_s',0.)+p.get('required_final_hold_s',.5)>4.+1e-9:
            raise ValueError(f'{name}: transition, settling plus required hold exceed 4 seconds')
        if p.get('stop_deadline_s',0.)>cfg['duration_s']:
            raise ValueError(f'{name}: stopping deadline exceeds future window')
        if 'stop_route_s' in p:
            conflicts=cfg['action'].get('conflicts',[])
            bound=min((c['ego_entry_s']-2.8 for c in conflicts),default=math.inf)
            if p['stop_route_s']>bound:raise ValueError(f'{name}: stop centre must keep body before conflict entry')
            if p.get('hold_policy')=='hold_until_window_end_v1' or p.get('release_condition')=='specified_actors_cleared_and_entry_clear_v1':
                from interaction_geometry import waiting_position_check
                roles=p.get('wait_for_roles',list(dict.fromkeys(part['role'] for c in conflicts for part in c['participants'])))
                clearance=waiting_position_check(cfg,float(p['stop_route_s']),roles)
                if not clearance['clear']:
                    raise ValueError(f'{name}: entire waiting ego body obstructs target routes {clearance["blocked_target_roles"]}')
    turn = cfg.get('geometry', {}).get('traffic_design', {}).get('turn_speed_plan')
    if turn is not None:
        ego = cfg['vehicles'][0]
        from action_design import turn_speed_plan
        pre = cfg['history_s'] + cfg['driving_warmup_s']
        speed = float(ego['initial_speed_mps'])
        centre = cfg['geometry']['ego_conflict_s']
        eta = (centre - ego['spawn_route_s']) / max(speed, .01) - pre
        authored_turn = turn_speed_plan(ego['route'], centre, eta, speed)
        requested = cfg['action']['profiles']['commit_go']['speed_increase_cap_m_s']
        if speed + requested > authored_turn['ceiling_m_s'] + 1e-6:
            raise ValueError('Authored left-turn speed exceeds the mapped curvature budget')
        turn['authored_ceiling_m_s'] = authored_turn['ceiling_m_s']
    from runtime.ego_decisions import validate_profiles
    validate_profiles(cfg['action'])
    from action_design import speed_profile_estimates
    cfg['action']['design_evidence'].update(
        target_speed_estimates_at_initial_speed=speed_profile_estimates(cfg['action']['profiles'],cfg['vehicles'][0]['initial_speed_mps']),
        actual_trajectory_dedup_required=False,equal_realized_trajectories_retained_in_scene=True)
    from audit_execution import DEFAULT_EXECUTION_CONTRACT
    cfg['execution_contract']=copy.deepcopy(DEFAULT_EXECUTION_CONTRACT)
    cfg['response_contract']={'unrelated_roles':list(proposal['unrelated_roles']),
        'unrelated_position_tolerance_m':0.,'unrelated_velocity_tolerance_m_s':0.,
        'unrelated_heading_tolerance_deg':0.}
    cfg['spec_schema_version']=3
    cfg['authoring_contract']={'editable_input':copy.deepcopy(proposal),
        'pose_resolution':'world pose derived from explicit per-actor station on frozen map-grounded route',
        'npc_no_reaction_is_failure':False,'deduplication_unit':'whole paired scene group',
        'execution_contract_editable_by_agent':False}
    # Recompute design-only interaction opportunity after all author edits. This
    # is not an outcome-magnitude acceptance filter.
    ego=cfg['vehicles'][0]
    estimate=ego['spawn_route_s']+fork*ego['initial_speed_mps']
    cfg['geometry']['traffic_design']['authored_initial_conditions']={
        'estimated_t0_route_s':estimate,'estimated_t0_speed_m_s':ego['initial_speed_mps'],
        'basis':'constant-speed design estimate, actual common-history rollout is separately audited'}
    # Record response distance after agent edits as well. A common nominal
    # policy may already slow/stop, so this estimate is not a substitute for
    # the measured-t0 execution audit or a reason to reject valid waiting.
    speed = float(ego['initial_speed_mps'])
    baseline = cfg['action']['nominal_policy']
    from runtime.ego_decisions import DEFAULT_STOP_RESPONSE_TIME_S
    cfg['geometry']['traffic_design']['authored_initial_conditions']['stop_budgets'] = {
        name: {'available_distance_m': float(p['stop_route_s'])-estimate
                    - speed*(float(p.get('onset_s', 0.)) if p['kind'] in ('stop_hold', 'yield_wait') else 0.),
               'required_distance_m': speed*float(p.get('response_time_s', baseline.get('response_time_s', DEFAULT_STOP_RESPONSE_TIME_S)))
                    + speed*speed/(2*float(p.get('comfortable_deceleration_m_s2', baseline['comfortable_deceleration_m_s2'])))
                    + float(p.get('stop_margin_m',0.)),
               'control_stop_margin_m': float(p.get('stop_margin_m',0.)),
               'stop_deadline_s': p.get('stop_deadline_s'),
               'deadline_feasibility_basis': 'checked against measured t0 speed at execution; authored spawn speed is not t0 speed',
               'response_time_s': float(p.get('response_time_s', baseline.get('response_time_s', DEFAULT_STOP_RESPONSE_TIME_S)))}
        for name, p in cfg['action']['profiles'].items()
        if p['kind'] in ('stop_hold', 'yield_wait') or p['kind'] == 'commit_go' and p.get('onset_s', 0.) > 0}
    from action_design import joint_pedestrian_encounter, validate_joint_pedestrian_entry
    if joint_pedestrian_encounter(cfg):
        validate_joint_pedestrian_entry(cfg,cfg['action']['conflicts'][0],final_authoring=True)
    return cfg
