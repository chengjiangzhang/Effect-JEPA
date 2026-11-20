"""Proposal/acceptance distribution audit. Diagnostics never select scenes."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def distribution(rows):
    keys=sorted({k for row in rows for k,v in row.get('numeric',{}).items() if v is not None})
    result={'count':len(rows),'categorical':{},'numeric':{},'pearson_correlations':[]}
    for key in ('phase','weather','map','vehicle_count','roles','route_combination'):
        result['categorical'][key]=dict(sorted(Counter(str(r.get(key,'unavailable')) for r in rows).items()))
    for key in keys:
        values=np.array([r['numeric'][key] for r in rows if r.get('numeric',{}).get(key) is not None],float)
        result['numeric'][key]={'n':len(values),'distinct_values':len(set(values)),
            'min':float(values.min()),'q10':float(np.quantile(values,.1)),
            'median':float(np.median(values)),'q90':float(np.quantile(values,.9)),
            'max':float(values.max())}
    for a,b in itertools.combinations(keys,2):
        values=np.array([(r['numeric'][a],r['numeric'][b]) for r in rows
            if r.get('numeric',{}).get(a) is not None and r.get('numeric',{}).get(b) is not None],float)
        if len(values)<5 or np.min(np.std(values,axis=0))<1e-10:
            continue
        corr=float(np.corrcoef(values.T)[0,1])
        if abs(corr)>=.8:
            result['pearson_correlations'].append({'a':a,'b':b,'n':len(values),'r':corr})
    result['correlation_note']='Only |r| >= 0.8 with n >= 5; small-sample alerts, not independence tests or failures. Inspect per-phase strata to avoid pooled mixture correlations.'
    return result


def proposal_row(proposal,job,cell):
    design=proposal['intent'];chosen=proposal['selection']
    return {'template':cell['template'],'phase':design['phase'],'weather':chosen.get('weather_id',job.get('weather_id')),
        'map':proposal.get('map','unavailable'),'vehicle_count':design['participants']['vehicle_count_including_ego'],
        'numeric':copy_numeric({k:design['parameters'][k] for k in design.get('active_parameters',design['parameters'])})}


def copy_numeric(values):
    return {k:float(v) for k,v in values.items() if isinstance(v,(float,int)) and not isinstance(v,bool)}


def effective_row(cfg,job,cell):
    rows=cfg.get('randomization',{}).get('effective',{}).get('vehicles',[])
    result={'template':cell['template'],'phase':cfg.get('interaction_design',{}).get('requested_phase'),
        'weather':job.get('weather_id'),'map':cfg['map'],'vehicle_count':len(cfg['vehicles']),
        'roles':','.join(sorted(v['role'] for v in cfg['vehicles'])),'numeric':{},
        'route_combination':json.dumps(cfg.get('randomization',{}).get('mapped_route_combination','unavailable'),sort_keys=True,separators=(',',':'))}
    result['numeric']={'ego_initial_speed_m_s':cfg['vehicles'][0]['initial_speed_mps'],
        'vehicle_count':len(cfg['vehicles']),'common_motion_s':cfg['history_s']+cfg['driving_warmup_s']}
    if rows:
        result['numeric'].update(npc_speed_std_m_s=float(np.std([r['initial_speed_m_s'] for r in rows[1:]])) if len(rows)>1 else None,
            npc_headway_std_s=float(np.std([r['policy']['idm']['time_headway'] for r in rows[1:]])) if len(rows)>1 else None)
    for key,profile in cfg['action']['profiles'].items():
        result['numeric'][key+'_onset_s']=profile.get('onset_s',0.)
    for ped in cfg.get('pedestrians',[]):
        result['numeric'].update(pedestrian_speed_m_s=ped['speed_mps'],
            prospective_pedestrian_t0_station_m=ped['initial_opportunity_design'].get('prospective_t0_station_m'))
    return result


def build_report(output,plan,state):
    output=Path(output);jobs={j['scene_id']:j for j in plan['jobs']};cells={c['cell_id']:c for c in plan['cells']}
    stages=defaultdict(list);failures=Counter();attempts=[];measured=[];sketches=[];encounters=[]
    for name,slot in state['slots'].items():
        job=jobs[name];cell=cells[job['cell_id']]
        stages['planned'].append({'template':cell['template'],'phase':job.get('interaction_phase'),
            'weather':job.get('weather_id'),'numeric':{}})
        for entry in slot['attempts']:
            path=output/entry['path'];status=entry['status']
            row={'scene_id':name,'path':entry['path'],'status':status,'template':cell['template'],
                'candidate_index':entry['candidate_index'],'attempt_index':entry['attempt_index']}
            proposal=path/'sampling_proposal.json'
            if proposal.exists():
                proposed=proposal_row(read(proposal),job,cell)
                stages['proposed'].append(proposed);row['proposed']=proposed
                if status=='failed':stages['rejected_proposals'].append(proposed)
            initial=path/'initial_plan'/name/'spec.json'
            if initial.exists():
                cfg=read(initial);effective=effective_row(cfg,job,cell)
                stages['compiled'].append(effective);row['effective']=effective
                # Later planning can change the prospective ego approach. Read
                # final authored inputs when available, never relabel intent.
                actual=path/'data'/name/'spec.json'
                final=effective_row(read(actual),job,cell) if actual.exists() else effective
                if entry.get('completed_stages',{}).get('physics_checked') or status in ('physics_checked','accepted'):
                    stages['physics_qualified'].append(final)
                if status=='accepted':stages['accepted'].append(final)
                row['compiled']=True
                row['physics_qualified']=bool(entry.get('completed_stages',{}).get('physics_checked') or status in ('physics_checked','accepted'))
                row['accepted']=status=='accepted'
            if status=='failed':
                reason=entry.get('reason','unknown').split(':',1)[0][:180]
                failures[(cell['template'],reason)]+=1;row['rejection_reason']=entry.get('reason')
            calibration=sorted((path/'reports').glob('encounter_calibration_*.json'))
            if calibration:
                encounters.append({'scene_id':name,'attempt':entry['path'],'template':cell['template'],
                    'status':status,'physics_qualified':row.get('physics_qualified',status in ('physics_checked','accepted')),
                    'history_measurements':len(calibration),'revisions':[read(p) for p in calibration]})
            coverage=path/'reports/interaction_coverage_physics.json'
            if coverage.exists():
                observed=read(coverage)
                measured.append({'scene_id':name,'attempt':entry['path'],'status':status,
                    'physics_qualified':row.get('physics_qualified',status in ('physics_checked','accepted')),
                    'template':cell['template'],'phase':job.get('interaction_phase'),
                    'missing_fields':[k for k in ('measured_t0','observed_events','object_response_pairs','controller_reason_counts') if k not in observed],
                    'measured_t0':observed.get('measured_t0'),'observed_events':observed.get('observed_events'),
                    'object_response_pairs':observed.get('object_response_pairs'),
                    'controller_reason_counts':observed.get('controller_reason_counts')})
                if (entry.get('completed_stages',{}).get('physics_checked') or status in ('physics_checked','accepted')) and observed.get('motion_sketch'):
                    sketches.append({'scene_id':name,'template':cell['template'],**observed['motion_sketch']})
            attempts.append(row)
    result={'schema':'scene-randomization-audit-v1','admission_gate':False,
        'count_units':{'planned':'one requested slot','proposed':'one persisted proposal including repairs',
            'compiled':'one compiled attempt, including later failure','physics_qualified':'one successful six-branch attempt',
            'accepted':'one fully accepted visual attempt'},
        'stages':{},'attempts':attempts,'conditional_stage_counts':conditional_counts(attempts),
        'map_candidate_eligibility':read(output/'catalog.json').get('cell_eligibility',[]) if (output/'catalog.json').exists() else [],
        'rejections':[{'template':t,'reason':r,'count':n} for (t,r),n in sorted(failures.items())],
        'native_observations':measured,'trajectory_similarity':trajectory_similarity(sketches),
        'encounter_opportunity_coverage':encounter_coverage(encounters,measured),
        'interpretation':'Intent is not measured progress; compilation is not native acceptance. Legitimate waiting, weak response and post-action contact remain. No outcome-based filtering. Small samples are not proof of population diversity.'}
    for stage in ('planned','proposed','compiled','rejected_proposals','physics_qualified','accepted'):
        rows=stages[stage];by_template={};by_phase={}
        for template in sorted({r['template'] for r in rows}):
            subset=[r for r in rows if r['template']==template]
            by_template[template]=distribution(subset)
            for phase in sorted({str(r.get('phase')) for r in subset}):
                by_phase[template+'/'+phase]=distribution([r for r in subset if str(r.get('phase'))==phase])
        result['stages'][stage]={'count':len(rows),'by_template':by_template,'by_template_phase':by_phase}
    return result


def encounter_coverage(records,observed):
    """Keep numerical opportunity, policy choice and real motion distinct."""
    by_template=defaultdict(Counter);dynamic=defaultdict(Counter)
    for record in records:
        counts=by_template[record['template']];counts['measured_attempts']+=1
        counts['history_revisions']+=record['history_measurements']-1
        counts['physics_qualified_attempts']+=int(record['physics_qualified'])
        final=record['revisions'][-1];counts['unconstrained_attempts']+=int(not final['rows'])
        for row in final['rows']:
            # A stopped actor is exempt from a free-speed ETA; it has NOT
            # satisfied a numerical rendezvous target.
            category=('observed_waiting_exempt' if row.get('classification','').startswith('observed_waiting')
                      else 'within_numeric_target' if row['within_target'] else 'outside_numeric_target')
            counts[category]+=1
    for record in observed:
        if not record.get('physics_qualified',record['status'] in ('physics_checked','accepted')):
            continue
        counts=dynamic[record['template']]
        for branch,roles in (record.get('observed_events') or {}).items():
            for role,events in roles.items():
                merge=events.get('dynamic_merge')
                if merge is None:continue
                counts['qualified_branch_actor_records']+=1
                for mode,n in merge['policy_mode_counts'].items():
                    counts['policy_'+mode+'_present']+=int(n>0)
                counts['policy_cancel_transitions']+=sum(x['after']=='aborting' for x in merge['policy_transitions'])
                counts['observed_return_segments']+=len(merge['observed_early_return'])
                counts['observed_target_centre_segments']+=len(merge['observed_target_centre_reached'])
                counts['target_gap_blocked_segments']+=len(merge['target_gap_blocked'])
    return {'calibration_attempts':records,'by_template':{k:dict(v) for k,v in by_template.items()},
        'dynamic_merge_qualified_branch_actors':{k:dict(v) for k,v in dynamic.items()},
        'calibration_unit':'attempt and target check; rejected and qualified attempts remain separately identified',
        'motion_unit':'qualified branch-actor events, before first scene contact; command transitions are not completed motions',
        'reasonable_no_response_retained':True,'outcome_selection':False,
        'missing_behavior_is_reported_not_filled_by_filtering':True}


def conditional_counts(attempts):
    groups=defaultdict(Counter)
    for row in attempts:
        proposed=row.get('proposed')
        if not proposed:continue
        speed=proposed.get('numeric',{}).get('ego_initial_speed_m_s')
        bins={'vehicle_count':str(proposed['vehicle_count']),'phase':str(proposed['phase']),
              'weather':str(proposed['weather'])}
        if speed is not None:
            bins['proposed_ego_speed_m_s']='below4' if speed<4 else '4to6' if speed<6 else '6andabove'
        for feature,value in bins.items():
            counts=groups[(row['template'],feature,value)];counts['proposed']+=1
            for stage in ('compiled','physics_qualified','accepted'):
                counts[stage]+=int(row.get(stage,False))
            counts['failed']+=int(row['status']=='failed')
            counts['started_or_interrupted']+=int(row['status'] in ('started','interrupted'))
            counts['compiled_awaiting_physics']+=int(row.get('compiled',False) and not row.get('physics_qualified',False) and row['status']!='failed')
            counts['physics_awaiting_visual']+=int(row.get('physics_qualified',False) and not row.get('accepted',False) and row['status']!='failed')
    return {'bins':[{'template':t,'feature':f,'value':v,**dict(n)} for (t,f,v),n in sorted(groups.items())],
        'unit':'proposal attempts, not unique scenes; physics success remains after later visual failure',
        'rates':'Not estimated from unfinished stages. Compare stage counts with their explicit proposal denominator and pending cases.'}


def trajectory_similarity(sketches):
    pairs=[];comparable=0
    for a,b in itertools.combinations(sketches,2):
        if a['template']!=b['template'] or a['roles']!=b['roles'] or a['branches']!=b['branches']:
            continue
        comparable+=1
        delta=np.asarray(a['positions_ego_t0_frame_m'])-np.asarray(b['positions_ego_t0_frame_m'])
        pairs.append({'a':a['scene_id'],'b':b['scene_id'],'template':a['template'],
            'position_rms_m':float(np.sqrt(np.mean(np.sum(delta*delta,axis=-1))))})
    return {'comparable_scene_pairs':comparable,'nearest_pairs':sorted(pairs,key=lambda p:p['position_rms_m'])[:30],
        'basis':'All5 branches and 9 future sensor-time positions in common ego t0 frame; same semantic role set/order and branch set required; no cross-topology equivalence claimed',
        'admission_gate':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=build_report(a.output,read(a.output/'plan.json'),read(a.output/'state.json'))
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
