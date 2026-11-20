#!/usr/bin/env python3
"""Publish a verified, portable scene-level snapshot; never collect or alter raw data."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

import numpy as np


SCHEMA = 'paired-clean-dataset-v1'
SCENE = re.compile(r'scene_[0-9]+\Z')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative(root, value):
    """Only regular in-tree files/directories; an outward or inward symlink is rejected."""
    p = PurePosixPath(value)
    if not isinstance(value, str) or not value or p.is_absolute() or '..' in p.parts or '\\' in value:
        raise ValueError(f'Unsafe relative path: {value!r}')
    result = root.joinpath(*p.parts)
    current = root
    for part in p.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'Symlink is not a portable sealed file: {current}')
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes root: {value}')
    return result


def file_record(path):
    return {'size': path.stat().st_size, 'sha256': file_sha(path)}


def verify_file(path, expected):
    if not path.is_file() or path.stat().st_size != expected['size'] or file_sha(path) != expected['sha256']:
        raise ValueError(f'Missing or changed sealed file: {path}')


def inventory_tree(root):
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symlink in published tree: {path}')
        if path.is_file() and path != root / 'checksums.json':
            files[str(path.relative_to(root))] = file_record(path)
    return files


def review_links(row):
    yield row['spec']
    yield row['common_frame']
    for branch in row['branches'].values():
        yield branch['metadata']
        yield branch['states']
        yield from branch.get('lidar', [])
        if branch.get('lidar_calibration'):
            yield branch['lidar_calibration']
        yield from branch.get('auxiliary_links', {}).values()
        for paths in branch['images'].values():
            yield from paths


def verify_published_dataset(output):
    """Recheck every byte and every generated viewer link, without the original work tree."""
    output = Path(output).resolve()
    if (output / 'checksums.json').is_symlink():
        raise ValueError('Publication inventory must be a regular file')
    inventory = read(output / 'checksums.json')
    if inventory.get('schema') != SCHEMA:
        raise ValueError('Not a managed clean dataset')
    expected = inventory['files']
    actual_paths = set()
    for path in output.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'Symlink in publication: {path}')
        if path.is_file() and path != output / 'checksums.json':
            actual_paths.add(str(path.relative_to(output)))
    if actual_paths != set(expected):
        raise ValueError('Published inventory differs from actual files; refusing to overwrite extra/missing content')
    for name, record in expected.items():
        verify_file(relative(output, name), record)
    manifest = read(output / 'dataset_manifest.json')
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Missing clean publication identity')
    ids = [entry['scene_id'] for entry in manifest['accepted_scenes']]
    if len(set(ids)) != len(ids) or len(ids) != manifest['published_scene_count']:
        raise ValueError('Invalid published scene count')
    raw_manifest = read(output / 'manifest.json')
    if raw_manifest['scene_ids'] != ids:
        raise ValueError('Raw and publication scene lists disagree')
    review = read(output / 'review.json')
    if [row['scene_id'] for row in review] != ids:
        raise ValueError('Review scene list differs')
    for entry, row in zip(manifest['accepted_scenes'], review):
        scene = entry['scene_id']
        if entry['data'] != scene or entry['reports'] != scene + '/qa':
            raise ValueError('Clean scene paths are not direct children')
        spec = read(relative(output, scene + '/spec.json'))
        if spec['scene_id'] != scene or spec['branches'] != raw_manifest['scene_branches'][scene]:
            raise ValueError('Published spec/branch identity differs')
        if set(spec['branches']) != set(entry['model_branches']) | set(entry['qa_only_branches']):
            raise ValueError('Branch usage does not cover the original branch set')
        seal = output / scene / 'qa/source_delivery_files.json'
        if file_sha(seal) != manifest['source_scene_seals'][scene]:
            raise ValueError('Copied original delivery inventory no longer matches provenance')
        original_files = read(seal)['files']
        bindings = read(output / scene / 'qa/portable_sources.json')
        for original, destination in bindings['files'].items():
            # Operational JSON links change location; every raw/NPZ/map/audit byte must not.
            if destination in bindings['rewritten_common_json']:
                continue
            if expected.get(destination) != original_files[original]:
                raise ValueError(f'Published bytes differ from the original seal: {destination}')
        for ref in [spec.get('map_geometry_path'), spec.get('render_preset', {}).get('file')]:
            if ref and not relative(output, ref).is_file():
                raise ValueError(f'Missing portable spec resource: {ref}')
        for link in review_links(row):
            if not relative(output, link).is_file():
                raise ValueError(f'Broken review link: {link}')
        for path in (output / scene / 'common').rglob('*.json'):
            payload = read(path)
            refs = payload.get('raw_files', []) + ([payload['raw_record']] if 'raw_record' in payload else [])
            for ref in refs:
                target = (path.parent / ref).resolve()
                if not target.is_relative_to(output) or not target.is_file():
                    raise ValueError(f'Broken common-coordinate link: {path}: {ref}')
    return manifest | {'checksums_sha256': file_sha(output / 'checksums.json'), 'verified': True}


def _selection(manifest_path, curation_path):
    manifest = read(manifest_path)
    curation = read(curation_path)
    if curation.get('input_manifest_sha256') != file_sha(manifest_path):
        raise ValueError('Curation does not bind the current accepted-scene manifest; curate this snapshot explicitly')
    entries = {row['scene_id']: row for row in manifest.get('accepted_scenes', [])}
    if len(entries) != len(manifest.get('accepted_scenes', [])):
        raise ValueError('Duplicate source scene IDs')
    retained = curation.get('retained_scenes')
    if not isinstance(retained, list) or not retained:
        raise ValueError('No explicitly curated scenes to publish')
    ids = [row['scene_id'] for row in retained]
    if len(ids) != len(set(ids)) or set(ids) - entries.keys():
        raise ValueError('Curation includes duplicate or non-accepted scenes')
    if curation.get('retained_scene_count') != len(ids):
        raise ValueError('Curation retained count differs')
    splits = curation.get('splits', {})
    split_ids = [scene for scenes in splits.values() for scene in scenes]
    if Counter(split_ids) != Counter(ids):
        raise ValueError('Curation splits must include every retained whole scene exactly once')
    if curation.get('dataset_purpose') == 'evaluation' and set(splits) != {'evaluation'}:
        raise ValueError('Evaluation publication must have only the evaluation split')
    for row in retained:
        if not SCENE.fullmatch(row['scene_id']) or row.get('split') not in splits or row['scene_id'] not in splits[row['split']]:
            raise ValueError('Invalid scene ID or split membership')
    return manifest, curation, [(entries[row['scene_id']], row) for row in sorted(retained, key=lambda x: x['scene_id'])]


def _source(root, entry, curated):
    scene = entry['scene_id']
    data = relative(root, entry['data'])
    folder = data if (data / 'spec.json').is_file() else data / scene
    # Original production format: accepted_data points to attempt/data.
    if folder.is_symlink() or folder.parent != data or data.name != 'data':
        raise ValueError('Expected original sealed attempt/data/scene directory')
    attempt = data.parent
    reports = relative(root, entry['reports'])
    if reports != attempt / 'reports':
        raise ValueError('Reports do not belong to the accepted attempt')
    seal_path = relative(attempt, 'reports/delivery_files.json')
    if file_sha(seal_path) != entry.get('delivery_inventory_sha256'):
        raise ValueError('Original delivery inventory identity changed')
    inventory = read(seal_path)
    if inventory.get('schema') != 'paired-delivery-sha256-v1':
        raise ValueError('Unsupported original delivery seal')
    for name, record in inventory['files'].items():
        verify_file(relative(attempt, name), record)
    for path in folder.rglob('*'):
        if path.is_symlink() or path.is_file() and str(path.relative_to(attempt)) not in inventory['files']:
            raise ValueError('Raw scene contains an unsealed or symlink file')
    spec_path = folder / 'spec.json'
    if curated.get('spec_sha256') != file_sha(spec_path):
        raise ValueError('Curation spec identity is stale')
    spec = read(spec_path)
    model = entry['model_branches']
    qa = entry['qa_only_branches']
    if spec['scene_id'] != scene or len(model) != 5 or len(set(model)) != 5 or qa != ['maintain_repeat']:
        raise ValueError('Require five model conditions and an explicitly QA-only repeat')
    if set(spec['branches']) != set(model + qa) or set(curated['model_branches']) != set(model):
        raise ValueError('Curation or spec would remove an action branch')
    return {'entry': entry, 'curated': curated, 'attempt': attempt, 'data': data, 'folder': folder,
            'seal_path': seal_path, 'inventory': inventory, 'spec': spec}


def _copy_file(source, target, mode, counters):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_record(source) != file_record(target):
            raise ValueError(f'Conflicting shared resource: {target}')
        return
    if mode != 'copy':
        try:
            os.link(source, target)
            counters['hardlinked_files'] += 1
            return
        except OSError as exc:
            if mode == 'hardlink' or exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOSYS, errno.EOPNOTSUPP):
                raise
    shutil.copyfile(source, target)
    counters['copied_files'] += 1


def _review(root, source):
    scene, spec = source['entry']['scene_id'], source['spec']
    motion = read(root / scene / 'common/action_selection.json')
    row = {'scene_id': scene, 'spec': f'{scene}/spec.json', 'common_frame': f'{scene}/common/frame.json',
           'model_branches': source['entry']['model_branches'],
           'similar_motion_branches': motion.get('similar_motion_branches', {}), 'branches': {}}
    for branch in spec['branches']:
        base = Path(scene) / 'branches' / branch
        meta = read(root / base / 'meta.json')
        if meta.get('status') != 'complete':
            raise ValueError(f'Incomplete accepted branch: {scene}/{branch}')
        with np.load(root / base / 'samples.npz', allow_pickle=False) as samples:
            times = samples['time_s'].tolist()
        cameras = meta['camera_names']
        lidar = spec.get('semantic_lidar', {}).get('name')
        row['branches'][branch] = {'times_s': times, 'metadata': str(base / 'meta.json'),
            'states': str(base / 'states.npz'), 'collisions': meta.get('collisions', []),
            'images': {c: [str(base / 'camera' / c / f'{i:05d}.png') for i in range(len(times))] for c in cameras},
            'lidar': [str(base / 'lidar' / lidar / f'{i:05d}.npz') for i in range(len(times))] if lidar else [],
            'lidar_calibration': f'{scene}/common/{branch}/lidar.json' if lidar else None,
            'auxiliary_links': {n: str(base / n) for n in ('pedestrians.npz', 'pedestrian_decisions.json', 'signal_states.json') if (root / base / n).is_file()}}
    return row


def _publish_scene(source, temporary, mode, counters):
    scene, attempt = source['entry']['scene_id'], source['attempt']
    mapping = {}
    for name in source['inventory']['files']:
        p = PurePosixPath(name)
        if p.parts[:2] == ('data', scene):
            dest = Path(*p.parts[1:])
        elif p.parts[:2] == ('export', scene):
            dest = Path(scene, 'common', *p.parts[2:])
        elif p.parts[0] == 'reports':
            dest = Path(scene, 'qa', *p.parts[1:])
        elif p.parts[0].startswith('common_history_'):
            dest = Path(scene, 'qa', 'history', *p.parts)
        elif p.parts[0] == 'initial_plan':
            dest = Path(scene, 'qa', *p.parts)
        elif p.parts[0] == 'data' and name != 'data/manifest.json' and p.parts[1] != '_staging_multicam':
            # Only shared resources are promoted outside the scene.
            if p.parts[1] != 'maps' and name != 'data/render_preset.json':
                continue
            dest = Path(*p.parts[1:])
        else:
            continue
        mapping[attempt / name] = dest
    required = [source['folder'] / 'spec.json', attempt / 'export' / scene / 'frame.json',
                attempt / 'export' / scene / 'action_selection.json', attempt / 'data/manifest.json']
    for path in required:
        if str(path.relative_to(attempt)) not in source['inventory']['files']:
            raise ValueError(f'Unsealed required delivery file: {path}')
    for original, destination in mapping.items():
        target = temporary / destination
        if destination.parts[:2] == (scene, 'common') and original.suffix == '.json':
            value = read(original)
            # These are operational links, not original audit provenance fields.
            for key in ('raw_record', 'raw_files'):
                if key not in value:
                    continue
                refs = value[key] if isinstance(value[key], list) else [value[key]]
                rewritten = []
                for ref in refs:
                    resolved = (original.parent / ref).resolve()
                    if resolved not in mapping:
                        raise ValueError(f'Derived link escapes the sealed scene: {original}: {ref}')
                    rewritten.append(os.path.relpath(temporary / mapping[resolved], target.parent))
                value[key] = rewritten if isinstance(value[key], list) else rewritten[0]
            write(target, value)
        else:
            _copy_file(original, target, mode, counters)
    _copy_file(source['seal_path'], temporary / scene / 'qa/source_delivery_files.json', mode, counters)
    _copy_file(attempt / 'data/manifest.json', temporary / scene / 'qa/source_data_manifest.json', mode, counters)
    write(temporary / scene / 'qa/usage.json', {'model_branches': source['entry']['model_branches'],
        'qa_only_branches': ['maintain_repeat'], 'source_attempt': str(attempt),
        'original_audit_paths': 'Unmodified historical provenance; use the clean root manifests for data access.',
        'original_inventory_sha256': file_sha(source['seal_path'])})
    write(temporary / scene / 'qa/portable_sources.json', {
        'source_attempt': str(attempt), 'path_base': 'dataset root',
        'rewritten_common_json': [str(dest) for dest in mapping.values()
                                  if dest.parts[:2] == (scene, 'common') and dest.suffix == '.json'],
        'files': {str(original.relative_to(attempt)): str(destination) for original, destination in mapping.items()}})
    return _review(temporary, source)


def publish_dataset(dataset_manifest, output, *, curation_manifest=None, link_mode='auto'):
    """Publish an explicit curation, idempotently; update only a verified snapshot from this source."""
    if link_mode not in ('auto', 'hardlink', 'copy'):
        raise ValueError('link_mode must be auto, hardlink or copy')
    manifest_path = Path(dataset_manifest).resolve()
    source_root = manifest_path.parent
    requested_output = Path(output).absolute()
    if requested_output.is_symlink():
        raise ValueError('Output may not be a symlink')
    output = requested_output.resolve()
    if output == source_root or output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise ValueError('Published output must not overlap the original work directory')
    curation_path = Path(curation_manifest).resolve() if curation_manifest else source_root / 'curation/curation_manifest.json'
    if not curation_path.is_file():
        raise ValueError('An explicit completed curation manifest is required; publication never fits or selects scenes')
    source_hash, curation_hash = file_sha(manifest_path), file_sha(curation_path)
    manifest, curation, selected = _selection(manifest_path, curation_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f'.{output.name}.publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = verify_published_dataset(output) if output.exists() else None
        sources = [_source(source_root, entry, curated) for entry, curated in selected]
        if previous:
            if previous['source_manifest'] != str(manifest_path):
                raise ValueError('Existing publication belongs to a different source dataset')
            if previous['source_manifest_sha256'] == source_hash and previous['curation_manifest_sha256'] == curation_hash:
                return previous | {'publication_action': 'verified_existing'}
        seals = {s['entry']['scene_id']: file_sha(s['seal_path']) for s in sources}
        if previous:
            old_seals = previous['source_scene_seals']
            # An accepted raw scene never changes; curation may remove an exact duplicate.
            accepted_seals = {row['scene_id']: row.get('delivery_inventory_sha256') for row in manifest['accepted_scenes']}
            if any(accepted_seals.get(scene) != seal for scene, seal in old_seals.items()):
                raise ValueError('An earlier accepted scene disappeared or changed its raw delivery seal')
        temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}.publishing-', dir=output.parent))
        backup = None
        try:
            counters = Counter()
            review = [_publish_scene(source, temporary, link_mode, counters) for source in sources]
            entries = []
            for source in sources:
                entry = copy.deepcopy(source['entry']); scene = entry['scene_id']
                entry.update(data=scene, reports=scene + '/qa', export=scene + '/common', review='index.html', split=source['curated']['split'])
                entry['source_data'] = source['entry']['data']
                entry['source_inventory'] = scene + '/qa/source_delivery_files.json'
                entries.append(entry)
            ids = [e['scene_id'] for e in entries]
            raw_manifest = copy.deepcopy(read(sources[0]['data'] / 'manifest.json'))
            raw_manifest.update(status='published_snapshot', scene_ids=ids,
                scene_branches={s['spec']['scene_id']: s['spec']['branches'] for s in sources},
                scene_model_branches={e['scene_id']: e['model_branches'] for e in entries},
                branches=sorted({b for s in sources for b in s['spec']['branches']}),
                planned_scene_count=len(ids), published_scene_count=len(ids),
                planned_branch_count=sum(len(s['spec']['branches']) for s in sources), review='index.html')
            write(temporary / 'manifest.json', raw_manifest)
            target = int(manifest.get('target_scene_count', len(ids)))
            bases = len({row['base_scene_id'] for row in curation['retained_scenes']})
            result = {'schema': SCHEMA, 'status': 'published_snapshot', 'dataset_purpose': curation['dataset_purpose'],
                'source_manifest': str(manifest_path), 'source_manifest_sha256': source_hash,
                'curation_manifest': str(curation_path), 'curation_manifest_sha256': curation_hash,
                'source_scene_seals': seals, 'source_accepted_scene_count': len(manifest['accepted_scenes']),
                'published_scene_count': len(ids), 'accepted_scene_count': len(ids), 'retained_base_scene_count': bases,
                'target_scene_count': target, 'remaining_scene_count': max(0, target - bases),
                'target_met': len(ids) == target and bases == target,
                'selection_unit': 'whole explicitly curated paired scene; no branch dropped',
                'qa_only_branches': ['maintain_repeat'], 'accepted_scenes': entries,
                'splits': curation['splits'], 'transfer': dict(counters), 'raw_data_modified': False,
                'immutable_usage': 'Read only. Hardlinked files share storage with original raw files; copy mode provides independent inodes.',
                'capture_provenance': 'Publishing does not re-collect or upgrade the protocol of historical captures.'}
            write(temporary / 'dataset_manifest.json', result)
            write(temporary / 'provenance.json', {'source_manifest': read(manifest_path), 'curation_manifest': curation,
                'source_manifest_sha256': source_hash, 'curation_manifest_sha256': curation_hash})
            for split, scene_ids in curation['splits'].items():
                if not re.fullmatch(r'[a-z_]+', split):
                    raise ValueError('Invalid split name')
                write(temporary / 'splits' / f'{split}.json', {'split': split, 'scene_ids': scene_ids,
                    'scenes': [e for e in entries if e['split'] == split], 'policy_sha256': curation['policy_sha256']})
            write(temporary / 'review.json', review)
            from runtime.export_common import write_review
            write_review(temporary, review)
            write(temporary / 'checksums.json', {'schema': SCHEMA, 'files': inventory_tree(temporary)})
            verified = verify_published_dataset(temporary)
            if file_sha(manifest_path) != source_hash or file_sha(curation_path) != curation_hash:
                raise ValueError('Source manifest or curation changed during publication')
            if output.exists():
                backup = Path(tempfile.mkdtemp(prefix=f'.{output.name}.previous-', dir=output.parent))
                backup.rmdir()
                output.rename(backup)
            try:
                temporary.rename(output)
            except BaseException:
                if backup is not None:
                    backup.rename(output); backup = None
                raise
            if backup is not None:
                shutil.rmtree(backup); backup = None
            return verified | {'publication_action': 'updated' if previous else 'created', 'output': str(output)}
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest', type=Path)
    parser.add_argument('--curation-manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--link-mode', choices=('auto', 'hardlink', 'copy'), default='auto')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only and args.dataset_manifest is None:
        parser.error('--dataset-manifest is required when publishing')
    report = verify_published_dataset(args.output) if args.verify_only else publish_dataset(
        args.dataset_manifest, args.output, curation_manifest=args.curation_manifest, link_mode=args.link_mode)
    print(json.dumps({k: report[k] for k in ('status', 'published_scene_count', 'retained_base_scene_count',
        'target_scene_count', 'remaining_scene_count', 'target_met', 'checksums_sha256')}, indent=2))


if __name__ == '__main__':
    main()
