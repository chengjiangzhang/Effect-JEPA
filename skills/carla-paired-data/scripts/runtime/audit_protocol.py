#!/usr/bin/env python3
"""Audit recorded sensor settings, calibration and renderer-prime evidence.

Reads only raw metadata and numerical arrays. No CARLA connection, image
rewriting, reactive-region selection, TTC or outcome-based filtering is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .audit_multicam import actual_render_checks, calibration_errors, renderer_prime_checks
    from .audit_auxiliary import inspect_auxiliary, compare_auxiliary
else:
    from audit_multicam import actual_render_checks, calibration_errors, renderer_prime_checks
    from audit_auxiliary import inspect_auxiliary, compare_auxiliary


if __package__:
    from . import semantic_lidar
else:
    import semantic_lidar


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path,allow_pickle=False) as value:
        return {key:value[key] for key in value.files}


def same(actual,expected):
    try:
        return bool(np.isfinite(float(actual)) and np.isclose(float(actual),float(expected),rtol=1e-6,atol=1e-6))
    except (ValueError,TypeError):
        return str(actual).lower()==str(expected).lower()


def rigid_from_spec(spec):
    roll,pitch,yaw=np.deg2rad(spec['rotation_deg'])
    cr,sr,cp,sp,cy,sy=np.cos(roll),np.sin(roll),np.cos(pitch),np.sin(pitch),np.cos(yaw),np.sin(yaw)
    matrix=np.eye(4)
    matrix[:3,:3]=[[cp*cy,cy*sp*sr-sy*cr,-cy*sp*cr-sy*sr],
                      [cp*sy,sy*sp*sr+cy*cr,-sy*sp*cr+cy*sr],
                      [sp,-cp*sr,cp*cr]]
    matrix[:3,3]=spec['position']
    return matrix


def inspect_branch(folder,config,branch):
    folder=Path(folder);errors=[]
    result={'status':'failed','errors':errors}
    paths={key:folder/name for key,name in [('metadata','meta.json'),('samples','samples.npz'),('states','states.npz')]}
    result['input_sha256']={key:digest(path) for key,path in paths.items()}
    meta=json.loads(paths['metadata'].read_text());samples=arrays(paths['samples']);states=arrays(paths['states'])
    result['semantic_lidar']=semantic_lidar.inspect_branch(folder,config,meta,states,samples)
    errors.extend(result['semantic_lidar']['errors'])
    result['auxiliary_actors']=inspect_auxiliary(folder,config,meta,states)
    errors.extend(result['auxiliary_actors']['errors'])
    for key in ('carla_client_version', 'carla_server_version'):
        if not isinstance(meta.get(key), str) or not meta[key].startswith('0.9.16'):
            errors.append(f'{key}: CARLA 0.9.16 is required')
    if meta.get('status')!='complete' or meta.get('branch')!=branch:
        errors.append('Metadata status/branch differs from completed expected capture')
    for key in ('spec_sha256','code_sha256','vehicle_roles'):
        if not config.get(key) or meta.get(key)!=config[key]:
            errors.append(f'Metadata {key} differs from frozen scene')
    if 'motion_warmup_s' in config.get('timing', {}):
        warmup = config['timing']['motion_warmup_s']
        if not same(meta.get('motion_warmup_s'), warmup):
            errors.append('Recorded driving warmup differs from frozen timing.motion_warmup_s')
        if 'driving_warmup_s' in config and config['driving_warmup_s'] != warmup:
            errors.append('Authored driving_warmup_s differs from frozen timing')
    cameras=config['camera_rig'];names=[camera['name'] for camera in cameras]
    if len(names)!=6 or len(set(names))!=6 or meta.get('camera_names')!=names:
        errors.append('Require six distinct cameras in the frozen order')
    rig=meta.get('rig',[])
    if not isinstance(rig,list) or len(rig)!=6 or any(not isinstance(r,dict) for r in rig) or [r.get('name') for r in rig]!=names:
        errors.append('Recorded sensor rig names/order differ from scene')
        return result
    result['actual_render']=actual_render_checks(meta,config)
    errors.extend(result['actual_render']['errors'])
    frame_ids=states.get('frame_id')
    if frame_ids is None or frame_ids.shape!=(121,) or not np.issubdtype(frame_ids.dtype,np.integer):
        errors.append('Require 121 integral recorded physics frame IDs')
        return result
    result['renderer_prime']=renderer_prime_checks(meta,config,frame_ids)
    errors.extend(result['renderer_prime']['errors'])
    shapes={'world_to_camera':(13,6,4,4),'camera_to_ego':(6,4,4),
            'ego_to_world':(13,4,4),'intrinsics':(6,3,3),'physics_indices':(13,)}
    for key,shape in shapes.items():
        value=samples.get(key)
        if value is None or value.shape!=shape or not np.isfinite(value).all():
            errors.append(f'Invalid finite calibration shape: samples.{key}, expected {shape}')
    if any('calibration shape' in issue for issue in errors):return result
    indices=samples['physics_indices']
    if not np.issubdtype(indices.dtype,np.integer) or not np.array_equal(indices,np.arange(0,121,10)):
        errors.append('Camera samples do not match the 2 Hz physics indices')
    else:
        transforms=states.get('transform_world_vehicle')
        if transforms is None or transforms.shape!=(121,len(config['vehicle_roles']),4,4):
            errors.append('Recorded actor transform coverage is invalid')
        elif not np.allclose(samples['ego_to_world'],transforms[indices,config['vehicle_roles'].index('ego')],rtol=0,atol=1e-5):
            errors.append('Camera sample ego pose differs from measured physics ego pose')
    result['calibration']=calibration_errors(samples)
    if result['calibration']['translation_max_m']>.002 or result['calibration']['rotation_matrix_max_error']>1e-4:
        errors.append('Camera extrinsics disagree with measured rigid ego-mounted rig')
    for index,(spec,record) in enumerate(zip(cameras,rig)):
        camera=spec['name'];focal=spec['width']/(2*np.tan(np.deg2rad(spec['fov_deg'])/2))
        expected=np.array([[focal,0,spec['width']/2],[0,focal,spec['height']/2],[0,0,1]])
        if not np.allclose(samples['intrinsics'][index],expected,rtol=0,atol=1e-6):
            errors.append(f'{camera}: intrinsics differ from declared resolution/FOV')
        if not np.allclose(samples['camera_to_ego'][index],rigid_from_spec(spec['transform']),rtol=0,atol=1e-5):
            errors.append(f'{camera}: extrinsics differ from frozen camera installation transform')
        expected_attributes={**config['render']['camera_attributes'], 'role_name':camera,
            'image_size_x':spec['width'],'image_size_y':spec['height'],'fov':spec['fov_deg'],
            'sensor_tick':0. if 'render_consistency_policy' in config else .05,
            'enable_postprocess_effects':'true'}
        for key,value in expected_attributes.items():
            for scope in ('attributes','actor_attributes'):
                if key not in record.get(scope,{}) or not same(record[scope][key],value):
                    errors.append(f'{camera}: {scope}.{key} differs from frozen setting')
    changed=[key for key,path in paths.items() if digest(path)!=result['input_sha256'][key]]
    if changed:errors.append('Raw inputs changed during audit: '+', '.join(changed))
    result['status']='failed' if errors else 'passed'
    return result


def audit(data):
    data=Path(data).resolve();manifest_path=data/'manifest.json'
    manifest=json.loads(manifest_path.read_text());initial_manifest_sha=digest(manifest_path)
    result={'schema':'carla-native-capture-protocol-v1','created_utc':datetime.now(timezone.utc).isoformat(),
            'data':str(data),'status':'failed','scene_ids':manifest['scene_ids'],
            'manifest_sha256':initial_manifest_sha,'scenes':{},'failures':[],
            'scope':'Recorded camera/weather attributes, measured calibration, renderer-prime evidence; no reactive labels or TTC.',
            'limitations':'Recorded evidence cannot establish image quality or runtime engine CVar readback. RGB consistency and physics checks remain separate.',
            'audit_source_sha256':digest(__file__),
            'helper_source_sha256':digest(Path(__file__).with_name('audit_multicam.py'))}
    expected=0;passed=0
    for scene in manifest['scene_ids']:
        if not isinstance(scene,str) or Path(scene).name!=scene:
            raise ValueError('Unsafe scene identifier')
        config_path=data/scene/'spec.json';config=json.loads(config_path.read_text())
        config_sha=digest(config_path)
        body={key:value for key,value in config.items() if key!='spec_sha256'}
        content_hash=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        scene_result={'config_file_sha256':config_sha,'branches':{}}
        result['scenes'][scene]=scene_result
        branches=manifest.get('scene_branches',{}).get(scene,config.get('branches',[]))
        if not branches or len(set(branches))!=len(branches) or branches!=config.get('branches'):
            result['failures'].append(f'{scene}: branch manifest/config mismatch')
            continue
        if content_hash!=config.get('spec_sha256'):
            result['failures'].append(f'{scene}: frozen config content hash mismatch')
        for branch in branches:
            expected+=1
            if not isinstance(branch,str) or Path(branch).name!=branch:
                raise ValueError('Unsafe branch identifier')
            try:
                record=inspect_branch(data/scene/'branches'/branch,config,branch)
            except (OSError,KeyError,ValueError,TypeError,IndexError,np.linalg.LinAlgError) as exc:
                record={'status':'failed','errors':[type(exc).__name__+': '+str(exc)]}
            scene_result['branches'][branch]=record
            if record['status']=='passed':passed+=1
            else:result['failures'].append({'scene':scene,'branch':branch,'errors':record['errors']})
        if config.get('pedestrians') or config.get('traffic_signal_program'):
            comparisons = {}
            for branch in branches:
                scope = 'repeat' if branch == 'maintain_repeat' else 'history'
                check = compare_auxiliary(data/scene/'branches'/branch,
                                          data/scene/'branches/maintain',config,scope=scope)
                comparisons[branch] = check
                if check['status'] != 'passed':
                    result['failures'].append({'scene':scene,'branch':branch,'errors':check['errors']})
            scene_result['auxiliary_common_history_and_repeat'] = comparisons
        if digest(config_path)!=config_sha:result['failures'].append(f'{scene}: config changed during audit')
    if digest(manifest_path)!=initial_manifest_sha:result['failures'].append('Manifest changed during audit')
    result['summary']={'expected_branches':expected,'passed_branches':passed,'failed_branches':expected-passed}
    result['status']='passed' if expected and not result['failures'] else 'failed'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.resolve().is_relative_to(args.data.resolve()):
        parser.error('Audit output must be outside the read-only raw dataset')
    result=audit(args.data);args.out.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.out.with_suffix(args.out.suffix+'.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(args.out)
    print(json.dumps({'status':result['status'],'summary':result['summary'],'report':str(args.out)}))
    return 0 if result['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
