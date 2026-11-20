#!/usr/bin/env python3
"""Export exact common-t0 geometry and a native-RGB inspection index.

This creates derived coordinate views only. Raw images/states are never changed,
and no reactive labels, TTC, OCC or model targets inferred from future outcomes
are generated here.
"""
from __future__ import annotations

import argparse
import html
import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

if __package__:
    from .audit_lane_change import rotation
    from .controllers_v2 import branch_names
    from .review_common import CAMERA_NAMES, write_json
    from .audit_auxiliary import inspect_auxiliary, read_arrays
    from . import semantic_lidar
else:
    from audit_lane_change import rotation
    from controllers_v2 import branch_names
    from review_common import CAMERA_NAMES, write_json
    from audit_auxiliary import inspect_auxiliary, read_arrays
    import semantic_lidar


def arrays(path):
    with np.load(path, allow_pickle=False) as loaded:
        result = {key: loaded[key] for key in loaded.files}
    for key, value in result.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite {path}:{key}')
    return result


def common_arrays(states, world_to_ego0, *, actor_type='vehicle'):
    """Use measured actor matrices and local bbox transforms, including offsets."""
    if actor_type not in ('vehicle', 'pedestrian'):
        raise ValueError('Unsupported actor type')
    transforms = np.asarray(states['transform_world_vehicle' if actor_type=='vehicle' else 'transform_world_actor'], dtype=np.float64)
    n = transforms.shape[1]
    converted = np.einsum('ij,tnjk->tnik', world_to_ego0, transforms)
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
    local = np.stack([(signs*states['bbox_extent'][i]) @ rotation(states['bbox_rotation_deg'][i]).T
                      + states['bbox_offset'][i] for i in range(n)])
    corners = np.einsum('tnij,nkj->tnki', converted[..., :3, :3], local) + converted[..., None, :3, 3]
    r = world_to_ego0[:3, :3]
    result = {'time_s': states['time_s'], 'frame_id': states['frame_id'],
              'world_to_ego0': world_to_ego0,
              'transform_ego0_vehicle' if actor_type=='vehicle' else 'transform_ego0_actor': converted,
              'position_ego0_m': np.einsum('ij,tnj->tni', r, states['position']) + world_to_ego0[:3, 3],
              'velocity_ego0_m_s': np.einsum('ij,tnj->tni', r, states['velocity']),
              'acceleration_ego0_m_s2': np.einsum('ij,tnj->tni', r, states['acceleration']),
              'angular_velocity_ego0_deg_s': np.einsum('ij,tnj->tni', r, states['angular_velocity']),
              'bbox_corners_ego0_m': corners, 'control': states['control']}
    if 'observed_control' in states:
        result['observed_control'] = states['observed_control']
    return result


def write_review(out, scene_rows):
    """One HTML entry point with all branches, cameras and recorded times."""
    payload = json.dumps(scene_rows, ensure_ascii=False).replace('<', '\\u003c')
    markup = '''<!doctype html><meta charset="utf-8"><title>Paired CARLA RGB review</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f7fa;color:#182332}select,input{margin:8px;padding:6px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.grid img{width:100%;height:auto}figure{margin:0;background:white;padding:8px}pre{white-space:pre-wrap}a{color:#1763a6}</style>
<h1>Paired CARLA data</h1><p>Native raw RGB and recorded annotations. Repeat is QA only. Select a scene, branch and time.</p>
<label>Scene <select id="scene"></select></label><label>Branch <select id="branch"></select></label>
<label>Time <input id="time" type="range" min="0" max="12" value="4"><span id="stamp"></span></label>
<p id="links"></p><div class="grid" id="images"></div><pre id="summary"></pre>
<script>const scenes=PAYLOAD;const scene=document.getElementById('scene'),branch=document.getElementById('branch'),time=document.getElementById('time');
for(let i=0;i<scenes.length;i++){scene.add(new Option(scenes[i].scene_id,i));}
function show(){const s=scenes[scene.value],b=s.branches[branch.value],k=Number(time.value);document.getElementById('stamp').textContent=b.times_s[k].toFixed(1)+' s';const grid=document.getElementById('images');grid.replaceChildren();for(const [camera,paths] of Object.entries(b.images)){const figure=document.createElement('figure'),caption=document.createElement('figcaption'),a=document.createElement('a'),img=document.createElement('img');caption.textContent=camera;img.src=paths[k];img.alt=camera+' '+b.times_s[k]+' s';a.href=paths[k];a.target='_blank';a.append(img);figure.append(caption,a);grid.append(figure);}const links=document.getElementById('links');links.replaceChildren();for(const [label,href]of Object.entries({spec:s.spec,metadata:b.metadata,states:b.states,common_frame:s.common_frame,...(b.lidar&&b.lidar.length?{semantic_lidar:b.lidar[k],lidar_calibration:b.lidar_calibration}:{}),...(b.auxiliary_links||{})})){const a=document.createElement('a');a.href=href;a.textContent=label;links.append(a,document.createTextNode(' | '));}document.getElementById('summary').textContent=JSON.stringify({scene:s.scene_id,branch:branch.value,qa_only:branch.value==='maintain_repeat',model_branch:s.model_branches.includes(branch.value),similar_motion:s.similar_motion_branches[branch.value]||null,collisions:b.collisions},null,2);}
function changeScene(){branch.replaceChildren();for(const b of Object.keys(scenes[scene.value].branches)){branch.add(new Option(b,b));}show();}scene.onchange=changeScene;branch.onchange=show;time.oninput=show;if(scenes.length)changeScene();</script>'''.replace('PAYLOAD', payload)
    (out/'index.html').write_text(markup)


def export(data, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'Use a new export directory: {out}')
    if out == data or data.is_relative_to(out):
        raise ValueError('Export must not contain the raw dataset')
    manifest = json.loads((data/'manifest.json').read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.common_export_', dir=out.parent))
    rows = []
    try:
        for scene in manifest['scene_ids']:
            folder = data/scene
            config = json.loads((folder/'spec.json').read_text())
            motion_audit = None
            if config.get('action', {}).get('profile_version') == 2:
                if __package__:
                    from .audit_realized_actions import audit_scene
                else:
                    from audit_realized_actions import audit_scene
                motion_audit = audit_scene(folder, config=config)
                if motion_audit.get('status') != 'passed':
                    raise ValueError(f'{scene}: realised action audit failed: {motion_audit}')
            roles = config['vehicle_roles']
            pedestrian_roles = [actor['role'] for actor in config.get('pedestrians', [])]
            names = branch_names(config)
            reference = arrays(folder/'branches/maintain/states.npz')
            matches = np.flatnonzero(np.isclose(reference['time_s'], 0., atol=1e-9, rtol=0.))
            if len(matches) != 1:
                raise ValueError(f'{scene}: require exactly one reference t0')
            t0 = int(matches[0]); ego = roles.index('ego')
            ego0_to_world = reference['transform_world_vehicle'][t0, ego]
            world_to_ego0 = np.linalg.inv(ego0_to_world)
            target = temporary/scene; target.mkdir()
            frame = {'reference_branch': 'maintain', 't0_physics_index': t0,
                     'world_to_ego0': world_to_ego0.tolist(), 'ego0_to_world': ego0_to_world.tolist(),
                     'axes': 'CARLA ego(t0): x forward, y right, z up; metres; one fixed transform for all branches and future times.',
                     'track_ids': [scene+':'+role for role in roles],
                     'pedestrian_track_ids': [scene+':'+role for role in pedestrian_roles],
                     'actor_types': {'vehicles':'vehicle','pedestrians':'pedestrian'},
                     'bbox_corner_axis': '8 actor-local extent-sign combinations transformed through measured bbox offsets/rotations and actor matrices',
                     'model_branches': list(branch_names(config, include_repeat=False)), 'qa_only_branches': ['maintain_repeat']}
            if motion_audit is not None:
                frame.update({key: motion_audit[key] for key in
                              ('model_branches', 'motion_audit_valid', 'distinct_motion_count', 'equivalent_motion_count')})
                write_json(target/'action_selection.json', motion_audit)
            write_json(target/'frame.json', frame)
            entry = {'scene_id': scene, 'spec': os.path.relpath(folder/'spec.json', out),
                     'common_frame': str(Path(scene)/'frame.json'), 'branches': {},
                     'model_branches': frame['model_branches'],
                     'similar_motion_branches': motion_audit['similar_motion_branches'] if motion_audit is not None else {}}
            for branch in names:
                source = folder/'branches'/branch
                meta = json.loads((source/'meta.json').read_text())
                if meta.get('status') != 'complete' or meta.get('vehicle_roles') != roles:
                    raise ValueError(f'{scene}/{branch}: missing capture or role correspondence')
                state = arrays(source/'states.npz'); sample = arrays(source/'samples.npz')
                auxiliary = inspect_auxiliary(source, config, meta, state)
                if auxiliary['status'] != 'passed':
                    raise ValueError(f'{scene}/{branch}: invalid auxiliary records: {auxiliary["errors"]}')
                if not np.array_equal(state['time_s'], reference['time_s']):
                    raise ValueError(f'{scene}/{branch}: state timebase differs')
                indices = sample['physics_indices'].astype(int)
                if not np.array_equal(state['frame_id'][indices], sample['frame_id']):
                    raise ValueError(f'{scene}/{branch}: RGB/physics frames differ')
                converted = common_arrays(state, world_to_ego0)
                converted.update(sample_time_s=sample['time_s'], intrinsics=sample['intrinsics'],
                                 ego0_to_camera=np.einsum('tsij,jk->tsik',sample['world_to_camera'],ego0_to_world))
                if 'semantic_lidar' in config:
                    lidar_report = semantic_lidar.inspect_branch(source, config, meta, state, sample)
                    if lidar_report['status'] != 'passed':
                        raise ValueError(f'{scene}/{branch}: invalid semantic LiDAR: {lidar_report["errors"]}')
                    converted.update(lidar_to_ego0=np.einsum('ij,tjk->tik', world_to_ego0, sample['lidar_to_world']),
                                     lidar_to_ego=sample['lidar_to_ego'],
                                     lidar_frame_id=sample['lidar_frame_id'],
                                     lidar_timestamp_s=sample['lidar_timestamp_s'])
                branch_out = target/branch; branch_out.mkdir()
                np.savez_compressed(branch_out/'states_ego0.npz', **converted)
                write_json(branch_out/'tracks.json', {'track_ids': frame['track_ids'],
                           'actor_ids_this_branch_only': meta['actor_ids'], 'roles': roles,
                           'actor_type': 'vehicle'})
                if pedestrian_roles:
                    pedestrians = read_arrays(source/'pedestrians.npz')
                    converted_pedestrians = common_arrays(pedestrians, world_to_ego0, actor_type='pedestrian')
                    np.savez_compressed(branch_out/'pedestrians_ego0.npz', **converted_pedestrians)
                    write_json(branch_out/'pedestrian_tracks.json', {
                        'track_ids': frame['pedestrian_track_ids'], 'roles': pedestrian_roles,
                        'actor_type': 'pedestrian',
                        'actor_ids_this_branch_only': [actor['actor_id'] for actor in meta['pedestrians']],
                        'control_columns': ['direction_x_world', 'direction_y_world', 'direction_z_world', 'speed_m_s', 'jump'],
                        'raw_record': os.path.relpath(source/'pedestrians.npz',branch_out),
                    })
                if 'semantic_lidar' in config:
                    write_json(branch_out/'lidar.json', {
                        'name': config['semantic_lidar']['name'],
                        'native_record': meta['semantic_lidar'],
                        'point_frame': 'sensor local CARLA xyz metres; use states_ego0.npz lidar_to_ego0 at the matching sample',
                        'formula': 'p_ego0 = lidar_to_ego0[sample] @ [x, y, z, 1]',
                        'raw_files': [os.path.relpath(source/'lidar'/config['semantic_lidar']['name']/f'{i:05d}.npz', branch_out) for i in range(len(sample['time_s']))],
                        'stable_actor_tracks': {key: scene+':'+role for key,role in meta['semantic_lidar']['actor_to_role'].items()},
                        'unmapped_ids': 'Keep raw ObjIdx; do not infer cross-branch track identity for unmapped/map objects.'})
                if config.get('traffic_signal_program'):
                    shutil.copyfile(source/'signal_states.json', branch_out/'signal_states.json')
                image_files = {camera: [source/'camera'/camera/f'{i:05d}.png' for i in range(len(sample['time_s']))]
                               for camera in CAMERA_NAMES}
                if any(not image.is_file() for paths in image_files.values() for image in paths):
                    raise FileNotFoundError(f'{scene}/{branch}: native RGB image missing')
                entry['branches'][branch] = {'times_s': sample['time_s'].tolist(),
                    'images': {camera: [os.path.relpath(image, out) for image in paths] for camera,paths in image_files.items()},
                    'metadata': os.path.relpath(source/'meta.json', out), 'states': os.path.relpath(source/'states.npz', out),
                    'collisions': meta['collisions'],
                    'lidar': [os.path.relpath(source/'lidar'/config['semantic_lidar']['name']/f'{i:05d}.npz', out) for i in range(len(sample['time_s']))] if 'semantic_lidar' in config else [],
                    'lidar_calibration': str(Path(scene)/branch/'lidar.json') if 'semantic_lidar' in config else None,
                    'auxiliary_links': {name: os.path.relpath(source/name, out) for name in
                        ('pedestrians.npz', 'pedestrian_decisions.json', 'signal_states.json') if (source/name).is_file()}}
            rows.append(entry)
        write_json(temporary/'manifest.json', {'schema': 'paired-common-frame-v1',
            'source_data': os.path.relpath(data, out), 'scene_ids': manifest['scene_ids'],
            'raw_data_modified': False, 'reactive_labels_generated': False,
            'qa_only_branches': ['maintain_repeat'], 'review': 'index.html',
            'scene_model_branches': {row['scene_id']: row['model_branches'] for row in rows}})
        write_review(temporary, rows)
        temporary.rename(out)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return {'status': 'exported', 'out': str(out), 'scenes': len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.data, args.out)))


if __name__ == '__main__':
    main()
