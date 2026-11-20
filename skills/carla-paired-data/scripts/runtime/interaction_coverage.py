"""Read-only measured encounter/response coverage, never an outcome filter.

2 Hz temporal support is not proof of RGB visibility (occlusion is untested).
Thresholds are declared engineering event descriptors, not fitted statistics.
"""
from __future__ import annotations
import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
import numpy as np


def arrays(path):
    with np.load(path,allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def segments(mask,times,minimum=.25):
    indices=np.flatnonzero(mask)
    groups=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    result=[]
    for group in groups:
        if not len(group):
            continue
        start,end=float(times[group[0]]),float(times[group[-1]])
        if end-start+1e-8>=minimum:
            sampled=np.arange(0.,4.0001,.5)
            result.append({'start_s':start,'end_s':end,'duration_s':end-start,
                           'rgb_sample_times':sampled[(sampled>=start-1e-8)&(sampled<=end+1e-8)].tolist()})
    return result


def measured_merge_returns(times, positions, source_route, decisions, contact_s=4.05):
    """Physical return under a preceding abort command, not ratio jitter.

    Use the full recorded history to establish a real outbound excursion after
    a merge start. A command at t acts on the following interval; state_before
    at the next sample identifies the command which produced that motion.
    Each step must move inward by >0.1 mm; pauses split this conservative
    continuous-motion descriptor, so it is not an exhaustive return count.
    These fixed engineering descriptors do not admit or reject a scene.
    """
    from controllers_v2 import project_to_route
    times=np.asarray(times,float); positions=np.asarray(positions,float)
    if len(times)!=len(positions) or len(times)!=len(decisions):
        raise ValueError('Merge return evidence must align with all recorded states')
    distance=np.array([abs(project_to_route(p,source_route)[1]) for p in positions])
    active=np.array([d['state_before']['mode']=='aborting' for d in decisions])
    valid=(times>=-1e-8)&(times<=4.+1e-8)&(times<float(contact_s)-1e-8)
    inward=np.r_[False,np.diff(distance)<-1e-4]
    mask=active&inward&valid&np.r_[False,valid[:-1]]
    indices=np.flatnonzero(mask)
    groups=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    starts=[i for i,d in enumerate(decisions) if d['state_before']['mode']=='waiting'
            and d['state_after']['mode']=='merging']
    result=[]
    for group in groups:
        if not len(group):continue
        begin,end=int(group[0])-1,int(group[-1])
        earlier=[i for i in starts if i<=begin]
        if not earlier:continue  # No recorded outbound baseline: do not assert a return.
        launch=earlier[-1]
        excursion=float(distance[launch:end+1].max()-distance[launch])
        returned=float(distance[begin]-distance[end])
        duration=float(times[end]-times[begin])
        if duration+1e-8<.25 or excursion+1e-8<.05 or returned+1e-8<.05:
            continue
        sampled=np.arange(0.,4.0001,.5)
        result.append({'start_s':float(times[begin]),'end_s':float(times[end]),
            'duration_s':duration,'outbound_start_s':float(times[launch]),
            'outward_excursion_m':excursion,'return_distance_m':returned,
            'source_centre_distance_at_end_m':float(distance[end]),
            'returned_to_source_centre':bool(distance[end]<=.05),
            'rgb_sample_times':sampled[(sampled>=times[begin]-1e-8)&(sampled<=times[end]+1e-8)].tolist()})
    return result


def stop_context(events,times,applied_speed,stations,route_length,modes,desired_speed=None,endpoint_distance=None,outgoing_modes=None):
    """Describe measured stillness without mistaking blocked motion for a decision."""
    for event in events:
        mask=(times>=event['start_s']-1e-8)&(times<=event['end_s']+1e-8)
        commands=np.asarray(applied_speed)[mask]
        counts=Counter(np.asarray(modes,dtype=object)[mask].tolist())
        remaining=np.maximum(0.,float(route_length)-np.asarray(stations)[mask])
        distance=remaining if endpoint_distance is None else np.asarray(endpoint_distance)[mask]
        zero=bool(np.all(commands<=.05))
        endpoint=bool(np.all(remaining<=.05) and np.all(distance<=.05))
        classification=('policy_stop_with_measured_stillness' if zero and set(counts)=={'stop'} else
                        'near_route_end_with_zero_command' if zero and endpoint else
                        'zero_command_other_mode' if zero else 'stationary_with_move_command')
        event.update(classification=classification,applied_policy_mode_counts=dict(counts),
            applied_command_speed_range_m_s=[float(commands.min()),float(commands.max())],
            remaining_route_distance_range_m=[float(remaining.min()),float(remaining.max())],
            endpoint_xy_distance_range_m=[float(distance.min()),float(distance.max())],
            route_end_tolerance_m=.05,zero_command_threshold_m_s=.05,
            near_route_end=endpoint,physical_obstacle_cause='not established by these records',
            frames=[{'time_s':float(times[k]),'applied_policy_mode':modes[k],
                     'outgoing_policy_mode':outgoing_modes[k] if outgoing_modes is not None and times[k]<4.-1e-8 else None,
                     'applied_speed_command_m_s':float(applied_speed[k]),
                     'desired_speed_command_m_s':float(desired_speed[k]) if desired_speed is not None and times[k]<4.-1e-8 else None,
                     'remaining_route_m':float(max(0.,route_length-stations[k]))}
                    for k in np.flatnonzero(mask)])
    return events


def active_feedback(spec, records, decisions, metadata, person_index=0):
    """Joint policy/native-control/motion evidence, before any branch contact.

    This describes generated outcomes. It is never a scene admission filter.
    The decision at t-dt is compared with native applied control/motion at t.
    """
    if __package__:
        from .pedestrian_policy import route_projection
    else:
        from pedestrian_policy import route_projection
    person = spec['pedestrians'][person_index]
    times = np.asarray(records['time_s'])
    dt = float(spec['dt_s'])
    by_time = {round(float(r['time_s']),8):r['decisions'][person_index] for r in decisions['samples']}
    modes = []
    for t in times:
        prior = by_time.get(round(float(t)-dt,8),{})
        modes.append(prior.get('mode','unobserved'))
    modes = np.asarray(modes,dtype=object)
    episode_start=np.empty(len(times))
    for i in range(len(times)):
        episode_start[i]=float(times[i])-dt if i==0 or modes[i]!=modes[i-1] else episode_start[i-1]
    new_episode=episode_start>=-1e-8
    projections = [route_projection(p,person['route_points']) for p in records['position'][:,person_index]]
    station = np.asarray([p[0] for p in projections])
    tangent = np.asarray([p[1] for p in projections])
    velocity = records['velocity'][:,person_index,:2]
    speed = np.linalg.norm(velocity,axis=1)
    signed_speed = np.sum(velocity*tangent,axis=1)
    control = records['observed_control'][:,person_index]
    signed_command = np.sum(control[:,:2]*tangent,axis=1)*control[:,3]
    contact = min((float(e['time_s']) for e in metadata['collisions']),default=float('inf'))
    future = (times>=-1e-8)&(times<=4.+1e-8)&(times<contact-1e-8)
    # A stop command issued after a walker has already been blocked for a
    # while is not an active movement-to-stop response. Read the actual speed
    # at the decision selecting this episode, before its control was applied.
    moving_at_selection=np.zeros(len(times),dtype=bool)
    for i in range(1,len(times)):
        moving_at_selection[i]=(speed[i-1]>.05 if modes[i]!=modes[i-1]
                                else moving_at_selection[i-1])
    length = float(np.linalg.norm(np.diff(np.asarray(person['route_points'])[:,:2],axis=0),axis=1).sum())
    endpoint_distance = np.linalg.norm(records['position'][:,person_index,:2]-np.asarray(person['route_points'])[-1,:2],axis=1)
    still_on_path = (length-station>.05)&(endpoint_distance>.05)
    nominal = float(person['speed_mps'])
    masks = {
        'stop':future&new_episode&moving_at_selection&still_on_path&(modes=='stop')&(control[:,3]<=.05)&(speed<=.05),
        'retreat':future&new_episode&(modes=='retreat')&(signed_command<-.05)&(signed_speed<-.05),
        'fast':future&new_episode&(modes=='fast')&(signed_command>1.15*nominal)&(signed_speed>1.15*nominal)}
    # Recovery needs a preceding *measured* interruption, not just one issued
    # stop/retreat decision while the actor was still moving forwards.
    interruption_ends = []
    for name in ('stop','retreat'):
        for event in segments(masks[name],times):
            selected=(times>=event['start_s']-1e-8)&(times<=event['end_s']+1e-8)
            if name=='retreat' and np.maximum(0.,-np.diff(station[selected])).sum()<=.05:
                continue
            interruption_ends.append(event['end_s'])
    after_interruption = times>min(interruption_ends,default=float('inf'))+1e-8
    masks['resume'] = future&after_interruption&np.isin(modes,('continue','slow','fast'))&(signed_command>.05)&(signed_speed>.05)
    events = {}
    for name, mask in masks.items():
        events[name] = []
        for event in segments(mask,times):
            selected=(times>=event['start_s']-1e-8)&(times<=event['end_s']+1e-8)
            path=station[selected]
            reverse=float(np.maximum(0.,-np.diff(path)).sum())
            if name=='retreat' and reverse<=.05:
                continue
            event.update(reverse_path_m=reverse,forward_path_m=float(np.maximum(0.,np.diff(path)).sum()),
                first_policy_episode_start_s=float(episode_start[selected].min()),
                measured_speed_range_m_s=[float(speed[selected].min()),float(speed[selected].max())],
                applied_command_speed_range_m_s=[float(control[selected,3].min()),float(control[selected,3].max())],
                applied_policy_mode_counts=dict(Counter(modes[selected].tolist())),
                all_samples_before_first_branch_contact=True)
            events[name].append(event)
    return {'schema':'active-pedestrian-feedback-v1','events':events,
            'first_branch_contact_s':contact if np.isfinite(contact) else None,
            'decision_to_native_alignment_s':dt,'minimum_continuous_duration_s':.25,
            'minimum_reverse_path_m':.05,'speed_threshold_m_s':.05,'fast_ratio':1.15,
            'requires_policy_native_control_and_actual_motion':True,
            'requires_post_t0_policy_episode':True,
            'stop_requires_actual_motion_at_policy_selection':True,
            'rgb_sampling_support_is_not_visibility':True,'outcome_admission_gate':False}


def annotate_paired_feedback(records,reports,person_index=0):
    """Add same-time, pre-contact branch contrast to each measured event.

    The 5 cm descriptor is above measured repeat error; it is not scene QA.
    An event identical under every ego intervention is not counted as paired
    response coverage, even if it is a legitimate reaction to background traffic.
    """
    times=records['maintain']['time_s']
    repeat=float(np.linalg.norm(records['maintain']['position'][:,person_index,:2]-
                                records['maintain_repeat']['position'][:,person_index,:2],axis=1).max())
    threshold=max(.05,repeat)
    for branch,report in reports.items():
        for events in report['events'].values():
            for event in events:
                pairs=[]
                for other,other_report in reports.items():
                    if other==branch:continue
                    contact=other_report['first_branch_contact_s']
                    mask=(times>=event['start_s']-1e-8)&(times<=event['end_s']+1e-8)
                    if contact is not None:mask &= times<contact-1e-8
                    distance=np.linalg.norm(records[branch]['position'][:,person_index,:2]-
                                            records[other]['position'][:,person_index,:2],axis=1)
                    value=float(distance[mask].max()) if mask.any() else None
                    pairs.append({'other_branch':other,'comparable_samples_before_either_contact':int(mask.sum()),
                                  'max_position_difference_during_event_m':value,
                                  'above_repeat_and_5cm':value is not None and value>threshold})
                event['paired_motion_evidence']={'pairs':pairs,'repeat_position_noise_m':repeat,
                    'threshold_m':threshold,'has_different_response':any(p['above_repeat_and_5cm'] for p in pairs),
                    'observation_descriptor_not_scene_admission':True}


def measured_vehicle_geometry(spec, state, zero):
    from controllers_v2 import project_to_route, _actual_corners
    observed={k:state[k][zero] for k in ('position','rotation_deg','velocity')}
    for key in ('bbox_offset','bbox_rotation_deg'):observed[key]=state[key]
    ego_route=spec['vehicles'][0]['route']
    ego_s,_,ego_tangent=project_to_route(observed['position'][0],ego_route)
    rows={}
    for i,actor in enumerate(spec['vehicles']):
        station,lateral,tangent=project_to_route(observed['position'][i],actor['route'])
        projected,ego_lateral,_=project_to_route(observed['position'][i],ego_route)
        body=[project_to_route(c,actor['route'])[0] for c in _actual_corners(observed,state['bbox_extent'],i)]
        intervals=[]
        for conflict in spec['action'].get('conflicts',[]):
            if i==0:
                intervals.append((conflict['id'],conflict['ego_entry_s'],conflict['ego_exit_s']))
            else:
                intervals.extend((conflict['id'],p['entry_s'],p['exit_s'])
                                 for p in conflict['participants'] if p['role']==actor['role'])
        phases=[]
        for name,entry,end in intervals:
            front,rear=max(body),min(body)
            phase=('before_entry' if front<entry else 'entering' if rear<entry else
                   'cleared' if rear>end else 'clearing' if front>end else 'inside')
            phases.append({'conflict_id':name,'body_phase':phase,'front_to_entry_m':entry-front,
                           'rear_to_exit_m':end-rear})
        rows[actor['role']]={'route_station_m':station,'route_lateral_offset_m':lateral,
            'relative_ego_route_station_m':projected-ego_s,'relative_ego_route_lateral_m':ego_lateral,
            'relative_forward_speed_m_s':float((observed['velocity'][i,:2]-observed['velocity'][0,:2])@ego_tangent),
            'forward_acceleration_m_s2':float(state['acceleration'][zero,i,:2]@tangent),
            'conflict_phases':phases}
    return rows


def vehicle_constraint_counts(samples,roles):
    counts=Counter()
    for sample in samples:
        for row in sample:
            role=row.get('role','unknown');leader=row.get('leader_index')
            if leader is not None:counts[(role,'following_leader',roles[int(leader)])]+=1
            for key in ('narrow_meeting','priority_zone'):
                value=row.get(key,{})
                if value.get('yielding'):
                    for other in value.get('trigger_roles',[]):counts[(role,key,other)]+=1
            for other in row.get('pedestrian_yield_roles',[]):counts[(role,'pedestrian_yield',other)]+=1
            signal=row.get('signal') or {}
            if signal.get('must_stop'):counts[(role,'stop_for_signal',str(signal.get('state','signal')))]+=1
    return [{'actor':a,'constraint':why,'related_actor_or_signal':b,'samples':n}
            for (a,why,b),n in sorted(counts.items())]


def reported_npc_rule_triggers(samples,roles,times):
    """Count explicit logged triggers, not dominant or marginal constraints.

    A leader relation alone is not a braking trigger. Ego signal diagnostics
    are context, not proof its selected policy obeyed that signal. Target
    speed changes per constraint were not recorded, so cannot be attributed.
    The t=4 decision acts beyond the observation window and is excluded.
    """
    records={}
    for sample,time in zip(samples,times):
        if not 0. <= float(time) < 4.-1e-8:
            continue
        found=set()
        for row in sample:
            actor=row.get('role')
            if actor not in roles[1:]:
                continue
            for key in ('narrow_meeting','priority_zone'):
                value=row.get(key) or {}
                if value.get('yielding'):
                    for other in value.get('trigger_roles') or [None]:
                        found.add((actor,key,None if other is None else str(other)))
            for other in row.get('pedestrian_yield_roles',[]):
                found.add((actor,'pedestrian_yield',str(other)))
            signal=row.get('signal') or {}
            if signal.get('must_stop'):
                found.add((actor,'stop_for_signal',str(signal.get('state','signal'))))
            collision=row.get('live_collision_constraint') or {}
            if collision.get('triggered'):
                for participant in collision.get('participants',[]):
                    found.add((actor,'live_collision_constraint',participant['role']))
        for key in found:
            records.setdefault(key,[]).append(float(time))
    return [{'actor':a,'reported_rule':rule,'related_actor_or_signal':other,
             'decision_samples':len(ts),'first_decision_time_s':min(ts),'last_decision_time_s':max(ts)}
            for (a,rule,other),ts in sorted(records.items(),key=lambda item:tuple('' if v is None else v for v in item[0]))]


def scene_report(root):
    root=Path(root); spec=json.loads((root/'spec.json').read_text())
    names=list(spec['action']['profiles'])
    vehicle={b:arrays(root/'branches'/b/'states.npz') for b in names+['maintain_repeat']}
    reference=vehicle['maintain']; zero=int(np.flatnonzero(reference['time_s']==0.)[0])
    times=reference['time_s'][zero:]; roles=spec['vehicle_roles']
    people={b:arrays(root/'branches'/b/'pedestrians.npz') for b in names+['maintain_repeat']} if spec.get('pedestrians') else {}
    t0={'vehicles':{role:{'position':reference['position'][zero,j].tolist(),
          'speed_m_s':float(np.linalg.norm(reference['velocity'][zero,j,:2])),
          'distance_to_ego_m':float(np.linalg.norm(reference['position'][zero,j,:2]-reference['position'][zero,0,:2]))}
          for j,role in enumerate(roles)},'pedestrians':{}}
    t0['vehicle_encounter_geometry']=measured_vehicle_geometry(spec,reference,zero)
    events={}; pairs={}; blockers={};constraints={}
    for b in names:
        events[b]={}
        if people:
            from pedestrian_policy import body_interval,route_projection
            p=people[b]; obs={k:v[zero] if k not in ('bbox_extent','bbox_offset','bbox_rotation_deg') else v for k,v in p.items()}
            for j,walker in enumerate(spec['pedestrians']):
                role=walker['role']; speed=np.linalg.norm(p['velocity'][zero:,j,:2],axis=1)
                station=np.array([route_projection(x,walker['route_points'])[0] for x in p['position'][zero:,j]])
                rear,front,_=body_interval(obs,j,walker['route_points'],'transform_world_actor')
                entry=walker['response_policy']['pedestrian_entry_s']; end=walker['response_policy']['pedestrian_exit_s']
                if b=='maintain':
                    t0['pedestrians'][role]={'speed_m_s':float(speed[0]),'route_station_m':float(station[0]),
                        'phase':'cleared' if rear>end else 'crossing' if front>=entry else 'approaching' if speed[0]>.05 else 'waiting',
                        'body_front_m':front,'body_rear_m':rear,'conflict_entry_m':entry,'conflict_exit_m':end}
                moving=speed>.05; moved_before=np.maximum.accumulate(moving)
                retreat=np.r_[False,np.diff(station)<-.05*.05]
                events[b][role]={'path_length_m':float(np.linalg.norm(np.diff(p['position'][zero:,j,:2],axis=0),axis=1).sum()),
                    'first_moving_s':float(times[np.flatnonzero(moving)[0]]) if moving.any() else None,
                    'stops_after_moving':segments((speed<=.05)&moved_before,times),
                    'slow_motion':segments((speed>.05)&(speed<walker['speed_mps']*.75),times),
                    'fast_motion':segments(speed>walker['speed_mps']*1.15,times),
                    'retreat':segments(retreat,times),
                    'reverse_path_m':float(np.maximum(0.,-np.diff(station)).sum())}
        for j,role in enumerate(roles[1:],1):
            v=vehicle[b]; speed=np.linalg.norm(v['velocity'][zero:,j,:2],axis=1)
            events[b][role]={'path_length_m':float(np.linalg.norm(np.diff(v['position'][zero:,j,:2],axis=0),axis=1).sum()),
                             'speed_t0_m_s':float(speed[0]),'speed_t4_m_s':float(speed[-1]),
                             'stops_after_moving':segments((speed<=.05)&np.maximum.accumulate(speed>.05),times)}
        path=root/'branches'/b/'controller_diagnostics.json'
        if path.exists():
            record=json.loads(path.read_text()); counts=Counter()
            def walk(x):
                if isinstance(x,dict):
                    if isinstance(x.get('reason'),str): counts[x['reason']]+=1
                    for v in x.values():walk(v)
                elif isinstance(x,list):
                    for v in x:walk(v)
            samples=record[zero:] if isinstance(record,list) else record.get('samples',[])
            walk(samples);blockers[b]=dict(counts)
            constraints[b]={'vehicle_relations':vehicle_constraint_counts(samples,roles),
                'reported_npc_rule_triggers':reported_npc_rule_triggers(samples,roles,times)}
            for j,role in enumerate(roles[1:],1):
                decisions=[row[j].get('dynamic_merge') for row in samples]
                if not decisions or any(d is None for d in decisions):
                    continue
                meta=json.loads((root/'branches'/b/'meta.json').read_text())
                contacts=[c['time_s'] for c in meta.get('collisions',[]) if 'time_s' in c]
                contact=min(contacts,default=4.05)
                progress=np.array([d['observed_lateral_progress'] for d in decisions])
                modes=[d['state_after']['mode'] for d in decisions]
                valid=times<min(contact,4.)-1e-8
                events[b][role]['dynamic_merge']={
                    'policy_mode_counts':dict(Counter(np.asarray(modes)[valid].tolist())),
                    'policy_transitions':[{'time_s':float(t),'before':d['state_before']['mode'],
                        'after':d['state_after']['mode'],'reason':d['reason']}
                        for t,d in zip(times,decisions) if t<min(contact,4.)-1e-8 and d['state_before']['mode']!=d['state_after']['mode']],
                    'target_gap_blocked':segments(np.array([not d['target_gap']['clear'] for d in decisions])&valid,times),
                    'observed_progress_t0':float(progress[0]),
                    'observed_progress_range_precontact':[float(progress[valid].min()),float(progress[valid].max())] if np.any(valid) else None,
                    'observed_progress_decrease':segments((np.r_[False,np.diff(progress)<-1e-4])&valid,times),
                    'observed_early_return':measured_merge_returns(reference['time_s'],
                        vehicle[b]['position'][:,j],spec['vehicles'][j]['route_maneuver']['source_route'],
                        [row[j]['dynamic_merge'] for row in (record if isinstance(record,list) else record['samples'])],contact),
                    'return_semantics':{'policy':'measured-abort-return-v1',
                        'minimum_outward_excursion_m':.05,'minimum_return_distance_m':.05,
                        'minimum_sustained_duration_s':.25,'source_centre_tolerance_m':.05,
                        'each_sample_inward_distance_gt_m':1e-4,
                        'pauses_split_segments_conservative_lower_bound':True,
                        'requires_recorded_merge_start_and_preceding_abort_command':True,
                        'excludes_contact_and_pre_t0_return_motion':True,
                        'progress_decrease_alone_is_not_return':True,
                        'centre_return_is_not_full_body_lane_completion':True},
                    'observed_target_centre_reached':segments((progress>=.95)&valid,times),
                    'progress_semantics':'Centre offset ratio between mapped source/target; not full-body lane completion',
                    'first_scene_contact_s':contact if contact<=4. else None,
                    'future_response_is_not_admission_requirement':True}
        ped_log=root/'branches'/b/'pedestrian_decisions.json'
        if ped_log.exists():
            decision_record=json.loads(ped_log.read_text())
            samples=decision_record['samples']
            counts=Counter()
            risk_counts=Counter()
            for row in samples:
                if row['time_s']<0.:
                    continue
                for decision in row['decisions']:
                    counts[decision.get('reason','unknown')]+=1
                    # These are the forward/entry risks actually considered by
                    # v3, not the legacy whole-crossing-gap diagnostic.
                    sweep=decision.get('entry_sweep') or decision.get('alternatives',{}).get('continue',{})
                    for check in sweep.get('actor_checks',[]):
                        if check.get('caution_time_s') is not None:
                            risk_counts[(decision['role'],check['role'])]+=1
            blockers[b]['pedestrian_decisions']=dict(counts)
            constraints.setdefault(b,{})['pedestrian_forward_risk']=[
                {'actor':a,'risk_actor':other,'samples':n} for (a,other),n in sorted(risk_counts.items())]
            for j,walker in enumerate(spec['pedestrians']):
                role=walker['role'];p=people[b]
                events[b][role]['active_feedback'] = active_feedback(spec,p,decision_record,
                    json.loads((root/'branches'/b/'meta.json').read_text()),j)
                decisions={round(row['time_s'],8):next(d for d in row['decisions'] if d['role']==role) for row in samples}
                def mode(decision):
                    after=decision.get('state_after')
                    return after.get('mode',decision.get('mode','unknown')) if isinstance(after,dict) else decision.get('mode','unknown')
                modes=[];outgoing_modes=[]
                for t in times:
                    # Native readback at t was sent by the preceding decision.
                    modes.append(mode(decisions[round(float(t)-spec['dt_s'],8)]))
                    outgoing_modes.append(mode(decisions[round(float(t),8)]))
                path=np.asarray(walker['route_points'],float)
                length=float(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1).sum())
                station=np.array([route_projection(x,walker['route_points'])[0] for x in p['position'][zero:,j]])
                stop_context(events[b][role]['stops_after_moving'],times,p['observed_control'][zero:,j,3],
                             station,length,modes,p['control'][zero:,j,3],
                             np.linalg.norm(p['position'][zero:,j,:2]-path[-1,:2],axis=1),outgoing_modes)
    for j,walker in enumerate(spec.get('pedestrians',[])):
        role=walker['role']
        annotate_paired_feedback(people,{b:events[b][role]['active_feedback'] for b in names},j)
    all_roles=[(role,j,vehicle) for j,role in enumerate(roles) if j]
    all_roles += [(p['role'],j,people) for j,p in enumerate(spec.get('pedestrians',[]))]
    for role,j,records in all_roles:
        repeat=float(np.linalg.norm(records['maintain']['position'][zero:,j,:2]-records['maintain_repeat']['position'][zero:,j,:2],axis=1).max())
        pairs[role]={'repeat_position_noise_m':repeat,'pairs':[]}
        for a,b in itertools.combinations(names,2):
            d=np.linalg.norm(records[a]['position'][zero:,j,:2]-records[b]['position'][zero:,j,:2],axis=1)
            dv=np.linalg.norm(records[a]['velocity'][zero:,j,:2]-records[b]['velocity'][zero:,j,:2],axis=1)
            pairs[role]['pairs'].append({'a':a,'b':b,'max_position_difference_m':float(d.max()),
                'rms_position_difference_m':float(np.sqrt(np.mean(d*d))),'max_velocity_difference_m_s':float(dv.max()),
                'above_repeat_and_05m':bool(d.max()>max(.5,repeat)),
                'above_repeat_and_1m':bool(d.max()>max(1.,repeat))})
    # Compact complete-window traces for cross-scene diagnostics. This is
    # computed from existing arrays once, never used for scene admission.
    indices=[int(np.argmin(abs(reference['time_s']-t))) for t in np.arange(0.,4.0001,.5)]
    # Use measured yaw even when ego is stationary.
    yaw=float(np.radians(reference['rotation_deg'][zero,0,2]))
    rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    positions=[]
    for b in names:
        position=vehicle[b]['position'][indices,:,:2]
        if people:
            position=np.concatenate((position,people[b]['position'][indices,:,:2]),axis=1)
        positions.append(((position-reference['position'][zero,0,:2]) @ rotation).tolist())
    sketch={'roles':roles+[p['role'] for p in spec.get('pedestrians',[])],'branches':names,'times_s':np.arange(0.,4.0001,.5).tolist(),
            'positions_ego_t0_frame_m':positions}
    return {'schema':'interaction-coverage-v1','scene_id':spec['scene_id'],'motion_sketch':sketch,
            'template':spec['grid_spec']['template'],'design':spec.get('interaction_design'),
            'measured_t0':t0,'observed_events':events,'object_response_pairs':pairs,
            'encounter_calibration_contract':spec.get('encounter_calibration'),
            'controller_reason_counts':blockers,'outcome_admission_gate':False,
            'observed_constraints_by_actor':constraints,
            'constraint_reporting_semantics':{
                'vehicle_relations':'Legacy relation/context counts; not necessarily an active constraint. Ego signal context does not establish policy compliance.',
                'reported_npc_rule_triggers':'Explicit NPC trigger flags and participants at decision times 0 <= t < 4; no inference from leader existence alone.',
                'limitation':'Per-rule target speed changes are not recorded. These counts do not establish the dominant cause of braking or waiting; final reason strings can be overwritten by later rules.'},
            'temporal_support_only':True,'rgb_visibility':'not established by state sampling; inspect rendered images',
            'stationary_event_semantics':'Measured stillness is classified using applied commands, policy mode and remaining route; it is not automatically reactive feedback or a diagnosed obstacle',
            'time_window_s':[-2.,4.],'thresholds':{'moving_speed_m_s':.05,'minimum_event_duration_s':.25,'response_m':[.5,1.]}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();report=scene_report(args.scene)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
