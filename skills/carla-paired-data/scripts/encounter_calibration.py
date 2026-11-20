"""Bounded layout calibration using common history only, before any future.

Targets are frozen at compilation. Each correction creates another complete
shared setup and reruns its history; no recorded actor is moved at t0. Accepted
future reactions, waiting, cancellation and contacts never feed this module.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import math
import numpy as np

LEGACY_VERSION = 'measured-encounter-calibration-v1'
VERSION = 'measured-encounter-calibration-v2'


def configure(spec, template):
    p=spec.get('randomization',{}).get('parameters',{})
    phase=spec.get('interaction_design',{}).get('requested_phase')
    if phase is None:
        phase=spec.get('interaction_design',{}).get('interaction_phase')
    roles=spec['vehicle_roles']; targets=[]
    if template=='merge_straight':
        seed=int.from_bytes(hashlib.sha256((str(spec['seed'])+'/merge-gap').encode()).digest()[:8],'big')
        desired=float(np.random.default_rng(seed).uniform(7.,14.))
        targets.append({'kind':'relative_gap','role':'parallel_vehicle','desired_gap_m':desired,
            'encounter_time_s':max(.5,float(p.get('merge_onset_s',.5))+1.25),'tolerance_m':2.})
    elif template in ('follower_response_straight','ego_lane_change_straight'):
        rear=[r for r in roles if r=='rear' or r in ('left_rear','right_rear')]
        if template=='ego_lane_change_straight' and phase in ('tight_gap','closing_rear') and not rear:
            raise ValueError('Requested target-lane rear interaction requires an actual rear participant')
        selected=rear or (['parallel_vehicle'] if 'parallel_vehicle' in roles else [])
        for role in selected:
            behind=role in rear
            targets.append({'kind':'relative_gap','role':role,
                'desired_gap_m':(-1. if behind else 1.)*float(p.get('rear_gap_m' if behind else 'front_gap_m',20.)),
                'encounter_time_s':1.5,'tolerance_m':3.})
    elif template in ('cross_unsignalized','left_unsignalized','left_signalized','merge_ramp','ego_merge_ramp'):
        role={'cross_unsignalized':'interaction_vehicle','left_unsignalized':'opposing_vehicle',
              'left_signalized':'opposing_vehicle','merge_ramp':'interaction_vehicle','ego_merge_ramp':'target_lane_vehicle'}[template]
        if role in roles:
            targets.append({'kind':'arrival_difference','role':role,
                'desired_difference_s':float(p.get('other_minus_ego_arrival_s',0.)), 'tolerance_s':.45})
    elif template.startswith('pedestrian_') and phase in ('entering','crossing'):
        targets.append({'kind':'pedestrian_clearance','role':spec['pedestrians'][0]['role'],
            'desired_difference_s':float(p.get('arrival_slack_s',.2)),'tolerance_s':.4})
    spec['encounter_calibration']={'policy_version':VERSION,'template':template,'requested_phase':phase,
        'max_revisions':2,'max_shift_per_revision_m':8.,'targets':targets,
        'uses_future_observations':False,'outcome_selection':False,
        'unconstrained_case':'Record actual opportunity; retain clearing, signal-controlled and legitimate waiting cases'}


def _point(route, station):
    route=np.asarray(route,float);length=np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1)
    s=np.r_[0.,np.cumsum(length)]; i=min(len(length)-1,max(0,int(np.searchsorted(s,station,side='right')-1)))
    u=float(np.clip((station-s[i])/max(length[i],1e-8),0,1))
    return route[i,:3]+u*(route[i+1,:3]-route[i,:3])


def _project(position,route):
    from runtime.controllers_v2 import project_to_route
    return project_to_route(position,route)


def _fast_eta(spec, speed, distance):
    """Target-speed envelope, explicitly an estimate rather than a rollout."""
    profiles=spec['action']['profiles'].values()
    positive=[p for p in profiles if p.get('acceleration_m_s2',0)>0]
    if not positive:
        return distance/max(speed,.2)
    estimates=[]
    for p in positive:
        travelled=0.;v=speed
        for step in range(1,161):
            t=step*.05
            if t>float(p.get('onset_s',0))+.25:
                v=min(speed+float(p.get('speed_increase_cap_m_s',5)),v+float(p['acceleration_m_s2'])*.05)
            travelled+=v*.05
            if travelled>=distance:
                estimates.append(t);break
    return min(estimates,default=8.)


def _distance_at_eta(spec, speed, target_s):
    """Inverse of the prospective positive-speed envelope, on its 20 Hz grid."""
    duration = float(np.clip(target_s, 0., 8.))
    positive = [p for p in spec['action']['profiles'].values() if p.get('acceleration_m_s2', 0) > 0]
    if not positive:
        return max(speed, .2) * duration
    # The fastest arrival across profiles is the inverse of their largest
    # travelled distance. All profiles start from the same observed speed.
    distances = []
    for profile in positive:
        velocity = speed
        distance = 0.
        for step in range(1, math.ceil(duration / .05) + 1):
            if step * .05 > profile.get('onset_s', 0.) + .25:
                velocity = min(speed + profile.get('speed_increase_cap_m_s', 5.),
                               velocity + profile['acceleration_m_s2'] * .05)
            distance += velocity * min(.05, max(0., duration - (step - 1) * .05))
        distances.append(distance)
    return max(distances)


def _measured_second_step(contract, target, role, row, proposed):
    """Use the preceding common-history response to the actual applied shift."""
    revisions = contract.get('layout_revisions', [])
    if not revisions or not proposed:
        return proposed
    previous = revisions[-1]
    applied = previous.get('applied_route_shifts_m', {}).get(role)
    field = {'relative_gap':'error_m','observation_window':'estimated_approach_shift_m'}.get(target['kind'], 'error_s')
    prior = next((r for r in previous.get('t0_evidence', {}).get('rows', [])
                  if r['role'] == target['role'] and r['kind'] == target['kind']
                  and r.get('conflict_id') == target.get('conflict_id')), {})
    if not applied or field not in prior:
        return proposed
    change = row[field] - prior[field]
    if abs(change) < 1e-5:
        return proposed
    secant = -row[field] * applied / change
    # A stopped or reacting participant may change regime between two
    # prefixes. Only trust a measured slope consistent with the analytic one.
    if target['kind'] == 'observation_window' and secant / proposed > 0.:
        row['correction_method'] = 'bounded_measured_entry_response'
        return proposed * float(np.clip(secant / proposed, .5, 2.))
    if .5 <= secant / proposed <= 2.:
        row['correction_method'] = 'measured_common_history_secant'
        return secant
    return proposed


def _entry_opportunity(spec, actors, conflicts, rows, corrections, branches=None):
    """Move a still-approaching vehicle's initial queue into the observable window."""
    if __package__:
        from .measured_t0 import travel_bound
        from .action_opportunity import MIN_OBSERVATION_S
    else:
        from measured_t0 import travel_bound
        from action_opportunity import MIN_OBSERVATION_S
    deadline = 4. - MIN_OBSERVATION_S
    limit = spec['encounter_calibration']['max_shift_per_revision_m']
    requested = {entry['role']: entry for entry in corrections}
    budgets = [b['stop_budget'] for b in (branches or {}).values() if 'stop_budget' in b]
    room = min((b['available_distance_m'] - b['required_response_braking_margin_m']
                for b in budgets), default=math.inf)
    ego_speed = actors['ego']['route_speed_m_s']
    if budgets:
        room -= max(.15, .05 * ego_speed)
    for identity, conflict in conflicts.items():
        pending = [p for p in conflict['participants'] if not p['already_cleared_at_t0']]
        def still_enters(part):
            eta = part['optimistic_entry_s']
            if eta is None or eta > deadline:
                return False
            role = part['role']
            shift = requested.get(role, {}).get('route_shift_m', 0.)
            if shift >= 0.:
                return True
            actor = actors[role]
            reach = travel_bound(actor['route_speed_m_s'], part['maximum_acceleration_m_s2'],
                                 deadline - .05, part['speed_cap_m_s'])
            return part['front_to_entry_m'] - shift <= reach

        if any(still_enters(part) for part in pending):
            continue
        candidates = []
        for part in pending:
            actor = actors[part['role']]
            if actor['actor_type'] != 'vehicle' or actor['route_speed_m_s'] <= .5:
                continue
            # Leave one physics tick of margin in the design. The actual QA deadline
            # still comes from the independent measured-t0 check.
            reach = travel_bound(actor['route_speed_m_s'], part['maximum_acceleration_m_s2'],
                                 deadline - .05, part['speed_cap_m_s'])
            candidates.append((part['front_to_entry_m'] - reach, part['role']))
        if not candidates:
            continue
        required, role = min(candidates)
        realign = requested.get(role)
        if required <= 0. and (not realign or realign['route_shift_m'] >= required):
            continue
        eta_target = next((r for r in rows if r['kind'] == 'arrival_difference' and r['role'] == role
                           and 'error_s' in r), None)
        advance = max(required, realign['route_shift_m'] if realign else 0.)
        # Test whether the observation window, relative arrival target and
        # every existing ego stopping budget can coexist at the measured speeds.
        # Leave incompatible setups to the original t0 gate without another replay.
        ego_shift = min(0., room)
        if eta_target:
            target_speed = actors[role]['route_speed_m_s']
            ideal = (advance / target_speed - eta_target['error_s']) * ego_speed
            if ideal - ego_speed * eta_target['tolerance_s'] > room:
                rows.append({'kind':'observation_window','role':role,'conflict_id':identity,
                             'classification':'no_joint_translation_at_observed_speeds',
                             'ego_available_shift_m':room,'minimum_target_shift_m':required,
                             'original_opportunity_gate_required':True,'uses_future_observations':False})
                continue
            ego_shift = min(room, (min(advance, limit)/target_speed - eta_target['error_s']) * ego_speed)
        if abs(ego_shift) > 1.e-6 and 'ego' not in requested:
            entry = {'role':'ego', 'route_shift_m':float(np.clip(ego_shift, -limit, limit)),
                     'basis':'joint_arrival_and_stopping_room'}
            corrections.append(entry)
            requested['ego'] = entry
        measurement = {'kind':'observation_window','role':role,'conflict_id':identity,
                       'deadline_s':deadline,'estimated_approach_shift_m':required,
                       'within_target':False,'uses_future_observations':False}
        if required <= 0.:
            # A rearward ETA correction must leave room to observe the target.
            measurement['classification'] = 'preserve_entry_during_arrival_adjustment'
            applied = advance
        else:
            applied = _measured_second_step(spec['encounter_calibration'], measurement, role, measurement, advance)
        if realign:
            realign.update(route_shift_m=min(applied, limit), basis='arrival_and_observation_window')
        else:
            entry = {'role':role, 'route_shift_m':min(applied, limit), 'basis':'observation_window'}
            requested[role] = entry
            corrections.append(entry)
        rows.append(measurement)


def analyze(spec, actors, conflicts=None, branches=None):
    contract=spec.get('encounter_calibration')
    if not contract:
        return {'applicable':False,'passed':True,'rows':[],'corrections':[]}
    version = contract.get('policy_version', LEGACY_VERSION)
    if version not in (VERSION, LEGACY_VERSION):
        raise ValueError(f'Unknown encounter calibration policy: {version}')
    ego=actors['ego'];definitions={v['role']:v for v in spec['vehicles']};rows=[];corrections=[]
    for target in contract['targets']:
        role=target['role'];other=actors[role];kind=target['kind'];row={**target,'uses_future_observations':False}
        shift=0.;move_role=role
        if kind=='relative_gap':
            route=definitions['ego']['route']
            own_s,_,tan=_project(ego['position_world_m'],route)
            other_s,_,_=_project(other['position_world_m'],route)
            relative=float((np.asarray(other['velocity_world_m_s'])[:2]-np.asarray(ego['velocity_world_m_s'])[:2])@tan)
            predicted=other_s-own_s+relative*target['encounter_time_s']
            error=predicted-target['desired_gap_m'];shift=-error
            row.update(measured_t0_gap_m=other_s-own_s,measured_relative_speed_m_s=relative,
                constant_speed_gap_at_encounter_m=predicted,error_m=error,within_target=abs(error)<=target['tolerance_m'])
        elif kind=='arrival_difference':
            centre=float(spec['geometry']['ego_conflict_s'])
            point=_point(definitions['ego']['route'],centre)
            npc_centre=_project(point,definitions[role]['route'])[0]
            own_distance=centre-ego['route_s_m'];other_distance=npc_centre-other['route_s_m']
            moving=min(ego['route_speed_m_s'],other['route_speed_m_s'])>.5
            if not moving:
                row.update(within_target=True,classification='observed_waiting_no_free_speed_eta',
                           measured_ego_speed_m_s=ego['route_speed_m_s'],measured_object_speed_m_s=other['route_speed_m_s'])
            else:
                own_eta=own_distance/ego['route_speed_m_s'];other_eta=other_distance/other['route_speed_m_s']
                error=other_eta-own_eta-target['desired_difference_s']
                shift=error*other['route_speed_m_s']
                row.update(ego_eta_s=own_eta,object_eta_s=other_eta,error_s=error,
                           within_target=abs(error)<=target['tolerance_s'])
        elif kind=='pedestrian_clearance':
            ped=next(p for p in spec['pedestrians'] if p['role']==role);rule=ped['response_policy']
            conflict=next(c for c in rule['vehicle_conflicts'] if c['role']=='ego')
            measured_walk = other.get('walker_execution')
            if measured_walk is None or not measured_walk['passed']:
                row.update(within_target=False, classification='walker_execution_missing_or_blocked',
                           walker_execution=measured_walk)
                rows.append(row)
                continue
            if other['speed_m_s'] < .05:
                classification = ('observed_waiting_not_forced_to_cross' if measured_walk['t0_or_final_applied_speed_m_s'] < .05
                                  else 'movement_command_active_eta_unavailable')
                row.update(within_target=True, classification=classification,
                           applied_speed_m_s=measured_walk['t0_or_final_applied_speed_m_s'])
            else:
                remaining=max(0.,rule['pedestrian_exit_s']-other['body_rear_route_s_m'])
                clear=remaining/max(other['speed_m_s'],.05)
                distance=max(0.,conflict['entry_s']-ego['body_front_route_s_m'])
                eta=_fast_eta(spec,ego['speed_m_s'],distance)
                error=eta-clear-target['desired_difference_s']
                move_role='ego'
                if version == VERSION:
                    target_eta = clear + target['desired_difference_s']
                    target_distance = _distance_at_eta(spec, ego['speed_m_s'], target_eta)
                    shift = distance - target_distance
                    row.update(correction_method='inverse_target_speed_envelope',
                               observed_entry_distance_m=distance, target_entry_distance_m=target_distance)
                else:
                    shift=error*max(ego['route_speed_m_s'],.5)
                row.update(pedestrian_clear_eta_s=clear,ego_target_speed_entry_eta_s=eta,error_s=error,
                    within_target=abs(error)<=target['tolerance_s'],estimate_is_not_execution_guarantee=True)
        if not row['within_target']:
            if version == VERSION:
                shift = _measured_second_step(contract, target, move_role, row, shift)
            amount=float(np.clip(shift,-contract['max_shift_per_revision_m'],contract['max_shift_per_revision_m']))
            corrections.append({'role':move_role,'route_shift_m':amount,'basis':kind})
        rows.append(row)
    if version == VERSION and conflicts and spec.get('action_separation_contract') and not spec.get('pedestrians'):
        _entry_opportunity(spec, actors, conflicts, rows, corrections, branches)
    return {'applicable':True,'policy_version':version,'passed':not corrections and not any(row.get('classification') == 'walker_execution_missing_or_blocked' for row in rows),'rows':rows,'corrections':corrections,
            'future_response_used':False,'free_arrival_estimates_do_not_override_observed_waiting':True}


def corrected_spec(spec, report):
    """Revise spawn layout, keeping the reference ego fixed for relative gaps.

    Same-lane followers behind the moved actor move with it. An actor ahead
    (especially the ego reference) must not follow that shift: translating both
    ends of a measured gap cannot calibrate the gap. Other-lane queues may move
    together, subject to the full spawn-overlap check below.
    """
    from scene_compiler import runtime_modules
    geometry,_=runtime_modules();out=copy.deepcopy(spec)
    slots={x['role']:x.get('route_lane') for x in spec['geometry'].get('traffic_design',{}).get('roles_and_slots',[])}
    definitions={v['role']:v for v in out['vehicles']};shifts={}
    for correction in report['corrections']:
        role=correction['role'];actor=definitions[role];lane=slots.get(role)
        source=actor.get('route_maneuver',{}).get('source_route',actor['route'])
        anchor_s=_project(_point(actor['route'],actor['spawn_route_s']),source)[0]
        ego_on_source=np.array_equal(definitions['ego']['route'],source) or (lane is not None and slots.get('ego')==lane)
        for other in out['vehicles']:
            common=(other['role']==role or (lane is not None and slots.get(other['role'])==lane)
                    or np.array_equal(other['route'],source))
            if role!='ego' and other['role']=='ego':
                common=False
            if common and ego_on_source and other['role']!=role:
                other_s=_project(_point(other['route'],other['spawn_route_s']),source)[0]
                common=other_s<anchor_s-1e-6
            if common:
                # A shared queue receives one coherent shift, not the sum of
                # multiple noisy requests for actors on that same queue.
                shifts.setdefault(other['role'],float(correction['route_shift_m']))
    # On v2 an explicit queue-head correction takes priority over an inherited
    # follower shift from another queue. Collision and mapped interval gates apply.
    if spec['encounter_calibration'].get('policy_version') == VERSION:
        shifts.update({request['role']: request['route_shift_m'] for request in report['corrections']})
    for role,shift in shifts.items():
        actor=definitions[role];station=float(actor['spawn_route_s'])+shift
        route=actor['route'];end=float(geometry.cumulative(route)[-1])
        upper=end-max(20.,actor['initial_speed_mps']*7.+10.)
        if actor.get('route_maneuver'):
            upper=min(upper,float(actor['route_maneuver']['source_interval_s_m'][0])-5.)
        if not 5.<=station<=upper:
            raise ValueError('Measured encounter correction exceeds mapped spawn interval for '+role)
        actor['spawn_route_s']=station;actor['transform']=geometry.point_at_s(route,station)
    if any(geometry.spawn_footprints_overlap(a,b) for a,b in itertools.combinations(out['vehicles'],2)):
        raise ValueError('Measured encounter correction would overlap shared spawn bodies')
    for key in ('spec_sha256','code_sha256','measured_t0_plan','base_scene_id'):
        out.pop(key,None)
    log=out['encounter_calibration'].setdefault('layout_revisions',[])
    log.append({'based_on_spec_sha256':spec.get('spec_sha256'),'applied_route_shifts_m':shifts,'t0_evidence':report})
    out['interaction_design']['initial_actor_parameters']=[{k:copy.deepcopy(v[k]) for k in
        ('role','spawn_route_s','initial_speed_mps','desired_speed_mps')} for v in out['vehicles']]
    # Refresh effective sampled inputs without rewriting the original proposal.
    from scene_randomization import snapshot
    snapshot(out,geometry)
    return out


def prepare_history(attempt, scene, host, port, plan, catalog, manager=None, attempt_identity=None):
    """Immutable bounded revisions; every decision precedes full future capture."""
    from pipeline import read,write,runtime_command,run_command,CaptureInfrastructureError
    from measured_t0 import load_history,analyze_t0,HistorySetupError,HistoryEvidenceError
    from scene_compiler import write_plan
    import subprocess
    initial=attempt/'initial_plan'; spec=read(initial/scene/'spec.json')
    maximum=int(spec['encounter_calibration']['max_revisions'])
    for revision in range(maximum+1):
        history=attempt/f'common_history_calibration_{revision:03d}'
        if not (history/'manifest.json').is_file() or read(history/'manifest.json').get('status')!='complete':
            if manager is not None:
                if attempt_identity is None or attempt_identity[0]!=scene:
                    raise ValueError('Managed encounter history requires its scene attempt identity')
                epoch=manager.before_attempt(*attempt_identity)
                write(attempt/'reports/server_epochs'/f"{epoch['epoch']:06d}_history.json",epoch)
            try:
                run_command(runtime_command('preflight_physics.py','--data',initial,'--out',history,
                    '--scenes',scene,'--history-only','--host',host,'--port',port),
                    attempt/'reports'/f'common_history_calibration_{revision:03d}.log',600)
            except subprocess.TimeoutExpired as exc:
                raise CaptureInfrastructureError('Encounter common-history collection timed out') from exc
        try:
            states,people,signals,identity=load_history(spec,history)
        except HistorySetupError:
            return initial,history  # original history geometry gate explains the failure
        except (HistoryEvidenceError,FileNotFoundError) as exc:
            raise CaptureInfrastructureError('Encounter history evidence invalid: '+str(exc)) from exc
        try:
            measured=analyze_t0(spec,states,people,signals)
        except HistoryEvidenceError as exc:
            raise CaptureInfrastructureError('Encounter t0 evidence invalid: '+str(exc)) from exc
        report=measured['encounter_calibration']
        record={**report,'revision':revision,'initial_spec_sha256':spec['spec_sha256'],
                'history':identity,'all_future_branches_not_yet_collected':True}
        write(attempt/'reports'/f'encounter_calibration_{revision:03d}.json',record)
        if any(row.get('classification') == 'walker_execution_missing_or_blocked' for row in report['rows']):
            raise ValueError('walker_motion_not_executed_in_common_history: '+str(report['rows']))
        if report['passed']:
            return initial,history
        if revision==maximum:
            raise ValueError('measured_encounter_calibration_budget_exhausted: '+str(report['rows']))
        corrected=corrected_spec(spec,report)
        next_initial=attempt/f'calibrated_plan_{revision+1:03d}'
        if not (next_initial/'manifest.json').exists():
            write_plan(next_initial,[corrected],plan,catalog=catalog)
        else:
            existing=read(next_initial/scene/'spec.json')
            # Finalization only supplies identities. All authored physics,
            # controller and layout fields must match the recomputed revision.
            comparable=copy.deepcopy(existing)
            for key in ('spec_sha256','code_sha256','base_scene_id'):
                comparable.pop(key,None)
            if comparable!=corrected:
                raise CaptureInfrastructureError('Existing calibrated plan does not match measured shared history')
        initial=next_initial;spec=read(initial/scene/'spec.json')
    raise AssertionError('Unreachable calibration loop')
