"""Causal adjacent-lane merging on two frozen, mapped driving corridors.

The intention and preferred duration are fixed, not the executed lateral path.
Only present observations, fixed geometry and the actor's own memory enter a
decision. No ego action label, future recording or sampled per-frame noise.
"""
from __future__ import annotations

import copy
import math
import numpy as np

VERSION = 'observed-gap-merge-v1'


def enabled(actor):
    return actor.get('route_maneuver', {}).get('policy_version') == VERSION


def validate(actor):
    if not enabled(actor):
        return
    m = actor['route_maneuver']
    source, target = (np.asarray(m[k], float) for k in ('source_route', 'target_route'))
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 4 or len(source) < 3:
        raise ValueError('Dynamic merge requires aligned mapped source/target routes')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite merge corridors')
    continuation=np.asarray(m.get('target_continuation_route',target),float)
    if continuation.ndim!=2 or continuation.shape[1]!=4 or len(continuation)<3 or not np.isfinite(continuation).all():
        raise ValueError('Invalid mapped receiving route continuation')
    interval=np.asarray(m['valid_source_interval_s_m'],float)
    total=float(np.linalg.norm(np.diff(source[:,:2],axis=0),axis=1).sum())
    if interval.shape!=(2,) or not np.isfinite(interval).all() or not 0.<=interval[0]<interval[1]<=total+1e-6:
        raise ValueError('Invalid mapped merge interval')
    for key in ('duration_s', 'abort_duration_s', 'clear_confirmation_s', 'minimum_gap_m',
                'front_headway_s', 'rear_headway_s', 'prediction_horizon_s', 'comfortable_brake_m_s2'):
        if not math.isfinite(float(m[key])) or float(m[key]) <= 0:
            raise ValueError('Invalid dynamic merge parameter: '+key)
    if not 0. < float(m['abort_max_progress']) < .5:
        raise ValueError('Abort must be limited to early, uncommitted merging')


def initial_state():
    return {'mode': 'waiting', 'blend': 0., 'clear_since_s': None,
            'started_at_s': None, 'aborted_at_s': None, 'completed_at_s': None}


def blended_route(maneuver, blend):
    source, target = (np.asarray(maneuver[k], float) for k in ('source_route', 'target_route'))
    route = source.copy()
    u = float(blend)
    weight = 10*u**3-15*u**4+6*u**5
    route[:, :3] += weight*(target[:, :3]-source[:, :3])
    delta = np.diff(route[:, :2], axis=0)
    route[:-1, 3] = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
    route[-1, 3] = route[-2, 3]
    return route


def _project(position, route):
    # Kept in the geometry implementation used by every other controller.
    if __package__:
        from .controllers_v2 import project_to_route
    else:
        from controllers_v2 import project_to_route
    return project_to_route(position, route)


def observed_progress(actor, position):
    m = actor['route_maneuver']
    source, target = np.asarray(m['source_route']), np.asarray(m['target_route'])
    station,offset,_=_project(position,source)
    lower,upper=m['valid_source_interval_s_m']
    if station<lower or station>upper:
        # Continue measuring against the real receiving route, not coincident
        # padding, once the car has passed the admissible merge interval.
        if 'target_continuation_route' not in m:
            return 0.
        target=np.asarray(m['target_continuation_route'])
    ds = abs(offset); dt = abs(_project(position, target)[1])
    return float(np.clip(ds/max(ds+dt, 1e-6), 0., 1.))


def observation_route(actor, position):
    """The nearer legal corridor for geometric reporting, never a new road."""
    if not enabled(actor):
        return actor['route']
    m = actor['route_maneuver']
    return min((m['source_route'], m['target_route'], actor['route']),
               key=lambda route: abs(_project(position, route)[1]))


def gap_check(index, observed, extents, roles, route, params, horizon):
    """Projected body gaps including approaching target-lane rear vehicles.

    Present velocities are extrapolated for a bounded manoeuvre horizon. This
    engineering approximation is recorded, not claimed as an avoidance proof.
    """
    if __package__:
        from .controllers_v2 import _actual_corners
    else:
        from controllers_v2 import _actual_corners
    station, _, tangent = _project(observed['position'][index], route)
    own_speed = max(0., float(observed['velocity'][index, :2] @ tangent))
    own_corners = _actual_corners(observed, extents, index)
    own_s = np.array([_project(p, route)[0] for p in own_corners])
    front, rear = float(own_s.max()), float(own_s.min())
    checks = []
    for j, role in enumerate(roles):
        if j == index:
            continue
        s, lateral, direction = _project(observed['position'][j], route)
        yaw = math.radians(float(observed['rotation_deg'][j, 2]))
        heading = np.array([math.cos(yaw), math.sin(yaw)])
        if float(heading @ direction) < .7:
            continue
        if abs(float(observed['position'][j, 2]-observed['position'][index, 2])) > extents[index,2]+extents[j,2]+.3:
            continue
        # A body touching the receiving corridor counts; source-lane cars do
        # not become target-lane traffic merely because they are nearby.
        cross=float(direction[0]*heading[1]-direction[1]*heading[0])
        half_width = abs(float(heading @ direction))*extents[j,1]+abs(cross)*extents[j,0]
        if abs(lateral) > float(params['lane_half_width_m'])+half_width-.15:
            continue
        corners = _actual_corners(observed, extents, j)
        stations = np.array([_project(p, route)[0] for p in corners])
        other_speed = max(0., float(observed['velocity'][j, :2] @ direction))
        if s >= station:
            side = 'front'; gap = float(stations.min()-front)
            closing = own_speed-other_speed
            required = params['minimum_gap_m']+params['front_headway_s']*own_speed
        else:
            side = 'rear'; gap = float(rear-stations.max())
            closing = other_speed-own_speed
            required = params['minimum_gap_m']+params['rear_headway_s']*other_speed
        required += max(0.,closing)**2/(2.*params['comfortable_brake_m_s2'])
        projected = gap-closing*horizon
        blocked = min(gap,projected) < required
        checks.append({'role':role,'side':side,'body_gap_m':gap,
                       'constant_velocity_gap_at_horizon_m':projected,
                       'required_gap_m':required,'closing_speed_m_s':closing,'blocked':bool(blocked)})
    return {'clear':not any(c['blocked'] for c in checks),'checks':checks,
            'horizon_s':float(horizon),'uses_future_observations':False}


def decide(index, observed, config, extents, memory, time_s, dt):
    actor = config['vehicles'][index]; m = actor['route_maneuver']; role = actor['role']
    before = copy.deepcopy(memory.get(role, initial_state())); state = copy.deepcopy(before)
    actual = observed_progress(actor, observed['position'][index])
    remaining = max(.5, float(m['duration_s'])*(1.-actual))
    horizon = min(float(m['prediction_horizon_s']), remaining)
    target = gap_check(index,observed,extents,config['vehicle_roles'],m['target_route'],m,horizon)
    source = None
    eligible = time_s >= float(m['earliest_start_time_s'])-1e-8
    source_station = _project(observed['position'][index], m['source_route'])[0]
    speed=float(np.linalg.norm(observed['velocity'][index,:2]))
    required_distance=max(10.,speed*(m['duration_s']+m['clear_confirmation_s']+1.)+extents[index,0]+3.)
    eligible &= float(m['valid_source_interval_s_m'][0]) <= source_station <= float(m['valid_source_interval_s_m'][1])-required_distance
    reason = 'waiting_for_intention'
    if state['mode'] == 'waiting':
        state['blend'] = 0.
        if eligible and target['clear']:
            if state['clear_since_s'] is None:
                state['clear_since_s'] = float(time_s)
            reason = 'confirming_target_gap'
            if time_s-state['clear_since_s']+1e-8 >= m['clear_confirmation_s']:
                state.update(mode='merging',started_at_s=float(time_s),clear_since_s=None)
                reason = 'start_in_observed_gap'
        else:
            state['clear_since_s'] = None
            reason = 'target_gap_blocked' if eligible else 'waiting_for_intention'
    elif state['mode'] == 'merging':
        if not target['clear'] and actual < m['abort_max_progress'] and state['blend'] < .5:
            source = gap_check(index,observed,extents,config['vehicle_roles'],m['source_route'],m,
                               min(m['prediction_horizon_s'],m['abort_duration_s']))
            if source['clear']:
                state.update(mode='aborting',aborted_at_s=float(time_s),clear_since_s=None)
                reason = 'cancel_early_merge_target_gap_closed'
            else:
                reason = 'continue_source_return_blocked'
        else:
            reason = 'continue_committed_merge' if not target['clear'] else 'continue_in_observed_gap'
    elif state['mode'] == 'aborting':
        source = gap_check(index,observed,extents,config['vehicle_roles'],m['source_route'],m,
                           min(m['prediction_horizon_s'],m['abort_duration_s']))
        reason = 'returning_to_source' if source['clear'] else 'return_gap_reblocked_hold_lateral'
        if not source['clear'] and target['clear']:
            state['mode']='merging'
            reason='return_blocked_continue_into_clear_target'
    else:
        reason = 'merge_completed'
    if state['mode'] == 'merging':
        state['blend'] = min(1.,state['blend']+dt/m['duration_s'])
        if state['blend'] >= 1.-1e-8 and actual >= .95:
            state.update(mode='complete',completed_at_s=float(time_s),blend=1.)
    elif state['mode'] == 'aborting' and reason!='return_gap_reblocked_hold_lateral':
        state['blend'] = max(0.,state['blend']-dt/m['abort_duration_s'])
        if state['blend'] <= 1e-8 and actual <= .05:
            state.update(mode='waiting',blend=0.,clear_since_s=None)
    memory[role] = state
    diag = {'policy_version':VERSION,'state_before':before,'state_after':copy.deepcopy(state),
            'reason':reason,'observed_lateral_progress':actual,'target_gap':target,'return_gap':source,
            'eligible':bool(eligible),'safety_guarantee':False}
    route=(np.asarray(m['target_continuation_route'],float)
           if state['mode']=='complete' and 'target_continuation_route' in m else blended_route(m,state['blend']))
    return route,diag


def replay_warmup(config, evidence):
    """Reconstruct memory from recorded pre-history observations, not a seed."""
    if not any(enabled(a) for a in config['vehicles']):
        return {}
    if not isinstance(evidence,dict) or evidence.get('schema') != 'npc-merge-memory-v1':
        raise ValueError('Dynamic merge requires full common warmup evidence')
    warmup = float(config.get('driving_warmup_s',1.))
    rows = evidence['warmup']; expected = -2.-warmup+np.arange(round(warmup/.05))*.05
    if len(rows)!=len(expected) or not np.allclose([r['time_s'] for r in rows],expected,atol=1e-8,rtol=0):
        raise ValueError('Incomplete dynamic merge warmup timeline')
    memory = {}
    n = len(config['vehicles'])
    for row in rows:
        observed={k:np.asarray(v,float) for k,v in row['observed'].items()}
        extents=np.asarray(row['bbox_extent'],float)
        for key in ('position','velocity','rotation_deg','bbox_offset','bbox_rotation_deg'):
            if observed[key].shape!=(n,3) or not np.isfinite(observed[key]).all():
                raise ValueError('Invalid merge warmup observation: '+key)
        if extents.shape!=(n,3) or not np.isfinite(extents).all() or np.any(extents<=0):
            raise ValueError('Invalid merge warmup actor extents')
        decisions={}
        for i, actor in enumerate(config['vehicles']):
            if enabled(actor):
                _,d=decide(i,observed,config,extents,memory,row['time_s'],.05)
                decisions[actor['role']]=d
        if decisions!=row['decisions']:
            raise ValueError('Dynamic merge warmup decision/memory cannot be reproduced')
    if memory!=evidence['history_start_state']:
        raise ValueError('Dynamic merge state at history start differs from warmup')
    return memory
