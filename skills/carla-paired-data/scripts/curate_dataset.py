#!/usr/bin/env python3
"""Curate whole paired scenes without modifying captures or deleting branches.

Only accepted scenes with a passing execution audit enter the model lists. Fit
distance scales and thresholds on training-only calibration scenes, group nearby
initial states before splitting, and deduplicate only same-plan/same-outcome
scenes within a template/style cell. The threshold is an auditable conservative
heuristic derived from the available distribution, not a validated natural-law
boundary. Small calibration groups fall back to exact initial-state duplicates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = 'paired-scene-curation-v1'
FEATURE_SCHEMA = 'ego0-relational-cv-disc-ttc-v1'
TIME_CAP = 20.0
NUMERIC_ROUND_DECIMALS = 6
SPLITS = ('train', 'validation', 'test')
FRACTIONS = (.8, .1, .1)
DEFAULTS = {'calibration_train_fraction': .35, 'minimum_calibration_scenes': 4,
            'nearest_neighbor_quantile': .1, 'distance_multiplier': .5,
            'max_coordinate_multiplier': 2., 'seed': 0}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def requested_base_scene_id(spec):
    """Physical spawn identity independent of seed, weather and capture labels.

    This deliberately groups even differing policies/styles at the same initial
    physical arrangement; they cannot become independent train/test examples.
    Route padding/resampling, actor IDs, colours and list order are ignored.
    """
    spawn = {'map': spec['map'], 'ego': None, 'npcs': []}
    vehicles = {v['role']: v for v in spec['vehicles']}
    roles = spec.get('vehicle_roles', [v['role'] for v in spec['vehicles']])
    if len(vehicles) != len(spec['vehicles']) or roles.count('ego') != 1 or set(roles) != set(vehicles):
        raise ValueError('Invalid actor identities for base_scene_id')
    for role in roles:
        vehicle = vehicles[role]; transform = vehicle['transform']
        row = {'position': np.round(np.asarray(transform['position'], float), NUMERIC_ROUND_DECIMALS).tolist(),
               'rotation_deg': np.round(np.asarray(transform['rotation_deg'], float), NUMERIC_ROUND_DECIMALS).tolist(),
               'initial_speed': round(float(vehicle['initial_speed_mps']), NUMERIC_ROUND_DECIMALS)}
        if role == 'ego':
            spawn['ego'] = row
        else:
            spawn['npcs'].append(row)
    spawn['npcs'].sort(key=fingerprint)
    return 'base_' + fingerprint(spawn)


def base_identity(record):
    # Old captures without explicit base_scene_id use the identical derived key.
    return record.get('base_scene_id', 'base_' + record['features']['requested_spawn_fingerprint'])


def capture_condition_signature(spec):
    """Preserve rendering conditions, not spelling differences in preset IDs."""
    return fingerprint({'weather': spec.get('weather', spec.get('weather_override', {})),
        'camera_attributes': spec.get('render', {}).get('camera_attributes', spec.get('camera_override', {})),
        'vehicle_light_mode': spec.get('vehicle_light_mode', 'default')})


def dedup_signature(record):
    return (record['dedup_group'], record['branch_plan_sha256'], record['outcome_signature'],
            record.get('capture_condition_sha256', 'legacy_unspecified'))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_path(base, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def route_station(route, point):
    """Nearest continuous XY polyline projection, not a nearest waypoint index."""
    route = np.asarray(route, float)
    if route.ndim != 2 or route.shape[0] < 2 or route.shape[1] < 2:
        raise ValueError('A route needs two finite XY points')
    xy = route[:, :2]
    if not np.isfinite(xy).all():
        raise ValueError('Nonfinite route geometry')
    delta = np.diff(xy, axis=0); squared = np.sum(delta * delta, axis=1)
    t = np.clip(np.sum((np.asarray(point)[:2] - xy[:-1]) * delta, axis=1)
                / np.maximum(squared, 1e-12), 0., 1.)
    closest = xy[:-1] + t[:, None] * delta
    index = int(np.argmin(np.sum((closest - np.asarray(point)[:2]) ** 2, axis=1)))
    distance = np.sqrt(squared)
    return float(np.sum(distance[:index]) + t[index] * distance[index])


def censored_time(value):
    # An absent/non-closing event differs explicitly from a finite time >20 s.
    valid = bool(value is not None and np.isfinite(value) and value >= 0.)
    return [min(float(value), TIME_CAP) if valid else TIME_CAP, float(valid)]


def cv_disc_ttc(relative_position, relative_velocity, radius):
    """Constant-velocity 2D bounding-disc first overlap; a feature, not true TTC."""
    p = np.asarray(relative_position, float); v = np.asarray(relative_velocity, float)
    c = float(p @ p - radius * radius)
    if c <= 0.:
        return 0.
    a = float(v @ v); b = 2. * float(p @ v)
    discriminant = b * b - 4. * a * c
    if a <= 1e-12 or b >= 0. or discriminant < 0.:
        return None
    value = (-b - math.sqrt(discriminant)) / (2. * a)
    return value if value >= 0. else None


def conflict_station(spec, role, vehicle):
    geometry = spec.get('geometry', {})
    direct = geometry.get('ego_conflict_s') if role == 'ego' else None
    if direct is not None:
        return float(direct)
    for conflict in geometry.get('conflicts', []):
        if role in conflict.get('route_s', {}):
            return float(conflict['route_s'][role])
    narrow = geometry.get('narrow_meeting', {})
    interval = narrow.get('yield_route_s', {}).get(role) or narrow.get('priority_route_s', {}).get(role)
    if interval:
        return float(interval[0])
    point = geometry.get('conflict_xyz')
    if point is not None and vehicle.get('route'):
        return route_station(vehicle['route'], point)
    return None


def scene_features(spec, states):
    roles = spec['vehicle_roles']
    if not roles or roles.count('ego') != 1:
        raise ValueError('Require exactly one ego role')
    time = np.asarray(states['time_s'], float)
    zero = np.flatnonzero(np.isclose(time, 0., rtol=0., atol=1e-8))
    if len(zero) != 1:
        raise ValueError('Require exactly one measured maintain t0')
    index = int(zero[0]); ego = roles.index('ego')
    position = np.asarray(states['position'][index], float)
    velocity = np.asarray(states['velocity'][index], float)
    rotation = np.asarray(states['rotation_deg'][index], float)
    boxes = np.asarray(states['bbox_extent'], float)
    if boxes.ndim == 3:
        boxes = boxes[index]
    if position.shape != (len(roles), 3) or velocity.shape != position.shape or rotation.shape != position.shape:
        raise ValueError('Invalid actor state shapes')
    if boxes.shape != position.shape or not np.isfinite([position, velocity, rotation, boxes]).all():
        raise ValueError('Invalid/nonfinite state or bounding box')
    yaw = math.radians(float(rotation[ego, 2]))
    # Collector stores [roll,pitch,yaw]. Row-vector world to ego rotation.
    basis = np.asarray([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    local = (position[:, :2] - position[ego, :2]) @ basis
    local_v = (velocity[:, :2] - velocity[ego, :2]) @ basis
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    heading = np.deg2rad(rotation[:, 2] - rotation[ego, 2])
    # Canonical order ignores role labels/actor IDs/seed; spatial ties use velocity.
    order = sorted((i for i in range(len(roles)) if i != ego),
                   key=lambda i: tuple(np.round(np.r_[local[i], local_v[i], heading[i]], 6)))
    vehicles = {v['role']: v for v in spec['vehicles']}

    def eta(i):
        actor = vehicles[roles[i]]; station = conflict_station(spec, roles[i], actor)
        if station is None or not actor.get('route') or speed[i] < 1e-6:
            return censored_time(None)
        remaining = station - route_station(actor['route'], position[i])
        return censored_time(remaining / speed[i] if remaining >= 0. else None)

    names = ['ego.speed_m_s', 'ego.conflict_eta_s', 'ego.conflict_eta_valid']
    values = [float(speed[ego]), *eta(ego)]
    floors = [.1, .1, 1.]
    for k, i in enumerate(order):
        radius = float(np.linalg.norm(boxes[ego, :2]) + np.linalg.norm(boxes[i, :2]))
        row = [*local[i], *local_v[i], math.sin(heading[i]), math.cos(heading[i]),
               *censored_time(cv_disc_ttc(local[i], local_v[i], radius)), *eta(i),
               float(boxes[i, 0]), float(boxes[i, 1])]
        suffixes = ['x_m', 'y_m', 'vx_m_s', 'vy_m_s', 'sin_heading', 'cos_heading',
                    'cv_disc_ttc_s', 'ttc_valid', 'conflict_eta_s', 'conflict_eta_valid',
                    'half_length_m', 'half_width_m']
        names += [f'npc_{k}.{name}' for name in suffixes]
        values += [float(v) for v in row]
        floors += [.25, .25, .1, .1, .02, .02, .1, 1., .1, 1., .05, .05]
    if not np.isfinite(values).all():
        raise ValueError('Features must be finite after explicit TTC/ETA censoring')
    canonical = [ego, *order]
    # World pose is deliberately included: exact identity can bridge style/template.
    physical = {'map': spec['map'], 'ego': np.round(np.r_[position[ego], velocity[ego],
                rotation[ego], boxes[ego]], NUMERIC_ROUND_DECIMALS).tolist(),
                'npcs': sorted(np.round(np.c_[position[order], velocity[order], rotation[order],
                                 boxes[order]], NUMERIC_ROUND_DECIMALS).tolist())}
    base_id = requested_base_scene_id(spec)
    if 'base_scene_id' in spec and spec['base_scene_id'] != base_id:
        raise ValueError('Explicit base_scene_id does not match physical spawn specification')
    return {'schema': FEATURE_SCHEMA, 'names': names, 'values': values, 'scale_floors': floors,
            'canonical_roles': [roles[i] for i in canonical], 'measured_t0_fingerprint': fingerprint(physical),
            'requested_spawn_fingerprint': base_id.removeprefix('base_'), 'base_scene_id': base_id, 'npc_count': len(order),
            'time_cap_s': TIME_CAP, 'time_mask_definition': '1 finite nonnegative event, including censored >20s; 0 absent or non-closing',
            'ttc_definition': 'constant velocity 2D bounding discs from measured t0; not actual future collision time'}


def outcome_signature(spec, execution, canonical_roles):
    """Coarse categorical outcomes only, preserving response/decision contrasts."""
    branches = [b for b in spec['branches'] if b != 'maintain_repeat']
    summaries = {}
    for branch in branches:
        row = execution['branches'][branch]
        outcome = row['observed_outcome']
        summaries[branch] = {'ego': sorted(outcome.get('ego', [])),
            'collision': bool(outcome.get('collision', False)),
            'censored_after_contact': bool(outcome.get('censored_after_contact', False)),
            'npcs': [{'events': sorted(outcome.get('npcs', {}).get(role, {}).get('events', [])),
                      'policy_reasons': sorted(outcome.get('npcs', {}).get(role, {}).get('policy_reasons', []))}
                     for role in canonical_roles[1:]]}
    response = execution['response_metrics']
    return fingerprint({'branches': summaries, 'effect_strength': response['effect_strength'],
                        'reactive_npc_count_05': response['reactive_npc_count_05'],
                        'reactive_npc_count_10': response['reactive_npc_count_10']})


def action_plan_signature(spec):
    """Executable meaning, excluding seed-dependent estimates and map padding."""
    action = spec['action']
    notes = {'description', 'label', 'risk_label', 'risk_condition', 'risk_conditions',
             'design_evidence', 'reference', 'provenance'}
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in notes}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value
    profiles = strip(action['profiles'])
    conflicts = action.get('conflicts', [])
    for profile in profiles.values():
        if 'stop_route_s' not in profile:
            continue
        candidates = [c for c in conflicts if c.get('id') == profile.get('conflict_id')]
        if not candidates:
            candidates = [c for c in conflicts if c.get('ego_entry_s') is not None]
        if candidates:
            stop = float(profile.pop('stop_route_s'))
            chosen = min(candidates, key=lambda conflict: abs(float(conflict['ego_entry_s'])-stop))
            profile['stop_margin_before_conflict_entry_m'] = round(float(chosen['ego_entry_s'])-stop,6)
    return fingerprint({'profiles': profiles, 'nominal_policy': strip(action.get('nominal_policy', {}))})


def verify_audit_sources(folder, spec_path, execution):
    """Reject stale audit reuse if its recorded inputs disagree with raw files."""
    if execution.get('spec_file_sha256') is not None and execution['spec_file_sha256'] != file_sha(spec_path):
        raise ValueError('Execution audit spec_file_sha256 no longer matches actual spec')
    for branch, files in execution.get('source_files', {}).items():
        branch_root = (folder/'branches'/branch).resolve()
        if not branch_root.is_relative_to((folder/'branches').resolve()):
            raise ValueError('Execution source branch escapes raw directory')
        for relative, expected in files.items():
            source = (branch_root/relative).resolve()
            if not source.is_relative_to(branch_root) or not source.is_file() or file_sha(source) != expected:
                raise ValueError(f'Execution audit source hash mismatch: {branch}/{relative}')


def load_scenes(manifest_path):
    manifest_path = Path(manifest_path).resolve(); manifest = read(manifest_path)
    base = manifest_path.parent; records, excluded = [], []
    seen = set()
    for entry in manifest.get('accepted_scenes', []):
        scene_id = entry['scene_id']
        if scene_id in seen:
            raise ValueError(f'Duplicate scene ID in manifest: {scene_id}')
        seen.add(scene_id)
        try:
            data = relative_path(base, entry['data'])
            folder = data if (data / 'spec.json').is_file() else data / scene_id
            spec_path = folder / 'spec.json'; spec = read(spec_path)
            if spec.get('scene_id') != scene_id:
                raise ValueError('Manifest scene_id differs from captured spec')
            report_dir = relative_path(base, entry['reports'])
            # execution.json is the production pipeline name. Standalone auditor
            # spelling is supported explicitly; never silently use physics-only.
            reports = [report_dir / n for n in ('execution.json', 'execution_audit.json')]
            report_path = next((p for p in reports if p.is_file()), None)
            if report_path is None:
                raise ValueError('Missing final capture execution audit')
            report = read(report_path)
            execution = report.get('scenes', {}).get(scene_id)
            if not isinstance(execution, dict) or execution.get('execution_pass') is not True:
                raise ValueError('Final capture execution audit did not pass')
            if execution.get('unrelated_invariance', {}).get('passed') is not True:
                raise ValueError('Unrelated-vehicle invariance audit did not pass')
            verify_audit_sources(folder, spec_path, execution)
            branches = [b for b in spec['branches'] if b != 'maintain_repeat']
            if len(branches) != 5 or len(set(branches)) != 5 or 'maintain' not in branches:
                raise ValueError('Require five planned conditions including maintain')
            for branch in branches:
                result = execution['branches'].get(branch, {})
                outcome = result.get('observed_outcome')
                censored_contact = (result.get('execution_pass') is None
                    and result.get('quality_status') == 'censored' and isinstance(outcome, dict)
                    and outcome.get('collision') is True)
                if (result.get('execution_pass') is not True and not censored_contact) or not isinstance(outcome, dict):
                    raise ValueError(f'Missing/pending/failed action execution: {branch}')
                if not isinstance(outcome.get('ego'), list) or not isinstance(outcome.get('npcs'), dict):
                    raise ValueError(f'Missing categorical ego/NPC outcomes: {branch}')
                for role in spec['vehicle_roles'][1:]:
                    npc = outcome['npcs'].get(role)
                    if not isinstance(npc, dict) or any(not isinstance(npc.get(k), list)
                        or not all(isinstance(v, str) for v in npc[k]) for k in ('events','policy_reasons')):
                        raise ValueError(f'Incomplete observed NPC outcome: {branch}/{role}')
            state_path = folder / 'branches/maintain/states.npz'
            with np.load(state_path, allow_pickle=False) as state:
                feature = scene_features(spec, state)
            grid = spec.get('grid_spec', {})
            template = grid.get('template', spec.get('family'))
            style = grid.get('npc_style', spec.get('controller', {}).get('npc_style'))
            if not isinstance(template, str) or not isinstance(style, str):
                raise ValueError('Require explicit template and NPC style')
            response = execution['response_metrics']
            if response['effect_strength'] not in ('weak', 'medium', 'strong'):
                raise ValueError('Invalid effect_strength')
            quality = float(entry.get('quality_score', 0.))
            if not np.isfinite(quality):
                raise ValueError('Nonfinite quality_score')
            records.append({'scene_id': scene_id, 'template': template, 'npc_style': style,
                'base_scene_id': feature['base_scene_id'],
                'weather_variant_id': spec.get('weather_variant_id'),
                'capture_condition_sha256': capture_condition_signature(spec),
                'cell_id': entry.get('cell_id'), 'metric_group': f'{template}|{feature["npc_count"]}',
                'dedup_group': f'{template}|{style}|{feature["npc_count"]}',
                'features': feature, 'model_branches': branches, 'qa_only_branches': ['maintain_repeat'],
                'branch_plan_sha256': action_plan_signature(spec),
                'outcome_signature': outcome_signature(spec, execution, feature['canonical_roles']),
                'response_metrics': response, 'effect_strength': response['effect_strength'],
                'quality_score': quality, 'quality_basis': 'all mandatory gates passed; optional manifest quality_score, default tied at zero',
                'source_split': entry.get('split'),
                'data': str(data), 'scene_directory': str(folder), 'reports': str(report_dir),
                'spec_sha256': file_sha(spec_path), 'execution_audit_sha256': file_sha(report_path),
                'maintain_states_sha256': file_sha(state_path)})
        except (ValueError, KeyError, OSError, TypeError) as exc:
            excluded.append({'scene_id': scene_id, 'reason': 'quality_or_provenance_unavailable',
                             'detail': str(exc), 'raw_data_modified': False})
    return records, excluded


def train_anchor(record, seed, fraction):
    # Seed alone cannot move the same physical scene between fitting and test.
    key = f'{seed}:{record["features"]["requested_spawn_fingerprint"]}'
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) / 2 ** 64 < fraction


def distances(x, y):
    squared = np.maximum(np.sum(x*x, axis=1)[:, None] + np.sum(y*y, axis=1)[None] - 2*x@y.T, 0.)
    return np.sqrt(squared / max(x.shape[1], 1))


def fit_policy(records, seed=0, source=None):
    groups = defaultdict(list)
    for r in records:
        groups[r['metric_group']].append(r)
    models = {}
    for name, rows in sorted(groups.items()):
        capture_count = len(rows)
        by_base = {}
        for row in sorted(rows, key=lambda r: r['scene_id']):
            by_base.setdefault(base_identity(row), row)
        rows = list(by_base.values())
        values = np.asarray([r['features']['values'] for r in rows], float)
        floors = np.asarray(rows[0]['features']['scale_floors'], float)
        center = np.median(values, axis=0)
        scale = np.maximum(np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0), floors)
        normalized = (values-center)/scale
        nearest = []
        if len(rows) >= DEFAULTS['minimum_calibration_scenes']:
            # Zero distances are valid and lead conservatively to exact-only;
            # never discard them to inflate a threshold on repeated layouts.
            for first in range(0, len(rows), 64):
                d = distances(normalized[first:first+64], normalized)
                d[np.arange(len(d)), np.arange(first, first+len(d))] = np.inf
                nearest.extend(np.min(d, axis=1).tolist())
            q = float(np.quantile(nearest, DEFAULTS['nearest_neighbor_quantile']))
            threshold = q * DEFAULTS['distance_multiplier']
        else:
            q, threshold = None, 0.
        models[name] = {'feature_names': rows[0]['features']['names'], 'center': center.tolist(),
            'scale': scale.tolist(), 'scale_floors': floors.tolist(), 'calibration_count': len(rows),
            'calibration_capture_count': capture_count,
            'calibration_unit': 'one lexicographically first scene per physical base_scene_id within metric group',
            'distance_threshold': threshold, 'nearest_neighbor_quantile_value': q,
            'mode': 'calibrated_candidate' if threshold > 1e-10 else 'exact_only',
            'reason': ('sufficient train-only calibration' if len(rows) >= 4
                       else 'fewer than four train-only calibration scenes; collect calibration data'),
            'calibration_prototypes': [{'scene_id': r['scene_id'], 'values': r['features']['values'],
                'base_scene_id': base_identity(r),
                'measured_t0_fingerprint': r['features']['measured_t0_fingerprint'],
                'requested_spawn_fingerprint': r['features']['requested_spawn_fingerprint']} for r in rows]}
    return {'schema': SCHEMA, 'feature_schema': FEATURE_SCHEMA, 'parameters': {**DEFAULTS, 'seed': seed},
            'fit_source': source or {}, 'threshold_status': 'distribution-derived candidate; review before publication and freeze before evaluation',
            'distance': 'RMS of per-coordinate robust-scaled differences; additionally every coordinate <=2*distance_threshold',
            'scope': 'metric fit per template and actor count; dedup per template AND NPC style; cross-style neighbors share split',
            'time_censor_s': TIME_CAP, 'metric_groups': models}


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size)); self.rank = [0] * size

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]; value = self.parent[value]
        return value

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def normalized_features(rows, model):
    if any(r['features']['names'] != model['feature_names'] for r in rows):
        raise ValueError('Frozen curation feature schema differs')
    scale = np.asarray(model['scale'], float)
    if np.any(scale <= 0.) or not np.isfinite(scale).all():
        raise ValueError('Invalid frozen feature scale')
    return (np.asarray([r['features']['values'] for r in rows]) - np.asarray(model['center'])) / scale


def neighbor_pairs(rows, model):
    if model is None or model['mode'] == 'exact_only':
        return
    threshold = float(model['distance_threshold'])
    x = normalized_features(rows, model)
    for start in range(0, len(rows), 64):
        d = distances(x[start:start+64], x)
        a, b = np.where(d <= threshold)
        for first, second in zip(a, b):
            first = int(first) + start; second = int(second)
            if first >= second:
                continue
            if float(np.max(np.abs(x[first] - x[second]))) <= 2. * threshold:
                yield first, second, float(d[first-start, second])


def equivalent_plan_outcome(a, b):
    return dedup_signature(a) == dedup_signature(b)


def curate_records(records, policy):
    """No raw mutation; full-group representatives and leakage components."""
    records = sorted(records, key=lambda r: r['scene_id']); size = len(records)
    union = UnionFind(size); pair_count = 0; protected = []; protected_count = 0
    groups = defaultdict(list); exact_buckets = defaultdict(list)
    for i, row in enumerate(records):
        groups[row['metric_group']].append(i)
        exact_buckets[('base_scene_id', base_identity(row))].append(i)
        for key in ('measured_t0_fingerprint', 'requested_spawn_fingerprint'):
            exact_buckets[(key, row['features'][key])].append(i)
    exact_links = []
    for bucket, indices in exact_buckets.items():
        for i in indices[1:]:
            union.union(indices[0], i)
        if len(indices) > 1:
            exact_links.append({'identity': bucket[0], 'scene_ids': [records[i]['scene_id'] for i in indices]})
        subsets = defaultdict(list)
        for i in indices:
            row = records[i]; subsets[dedup_signature(row)].append(i)
        if len(subsets) > 1 and len(indices) > 1:
            # A differing outcome or plan is protected even for exactly equal t0.
            for i in indices[1:]:
                if not equivalent_plan_outcome(records[indices[0]], records[i]):
                    protected_count += 1
                    if len(protected) < 2000:
                        protected.append({'a': records[indices[0]]['scene_id'], 'b': records[i]['scene_id'],
                                          'reason': 'same base/initial state; different cell, plan, outcome or capture condition', 'distance': 0.})
    for group, indices in sorted(groups.items()):
        rows = [records[i] for i in indices]; model = policy['metric_groups'].get(group)
        for a, b, distance in neighbor_pairs(rows, model):
            i, j = indices[a], indices[b]; pair_count += 1; union.union(i, j)
            if not equivalent_plan_outcome(records[i], records[j]):
                protected_count += 1
                if len(protected) < 2000:
                    protected.append({'a': rows[a]['scene_id'], 'b': rows[b]['scene_id'],
                        'distance': distance, 'reason': 'different cell, action plan, observed outcome or capture condition; retain both if otherwise representative'})
    # Direct-neighbor representatives prevent single-link chains from collapsing
    # endpoints farther apart than the frozen distance threshold.
    representatives, assignment = [], {}
    keep_by_signature = defaultdict(list)
    for i in sorted(range(size), key=lambda j: (-records[j]['quality_score'], records[j]['scene_id'])):
        row = records[i]
        signature = dedup_signature(row)
        candidates = keep_by_signature[signature]
        previous = [j for j in candidates if row['features']['measured_t0_fingerprint']
                    == records[j]['features']['measured_t0_fingerprint']]
        model = policy['metric_groups'].get(row['metric_group'])
        if not previous and candidates and model and model['mode'] != 'exact_only':
            x = normalized_features([row], model)
            y = normalized_features([records[j] for j in candidates], model)
            close = np.flatnonzero(distances(x, y)[0] <= model['distance_threshold'])
            previous = [candidates[k] for k in close
                        if np.max(np.abs(x[0]-y[k])) <= 2.*model['distance_threshold']]
        if previous:
            best = min(previous, key=lambda j: (-records[j]['quality_score'], records[j]['scene_id']))
            assignment[i] = best
        else:
            assignment[i] = i; representatives.append(i); keep_by_signature[signature].append(i)
    # Every group touching calibration must remain training. This includes
    # external calibration replicas, independent of scene IDs and NPC style.
    calibration_fingerprints = set()
    for model in policy['metric_groups'].values():
        for p in model['calibration_prototypes']:
            calibration_fingerprints.update((p['measured_t0_fingerprint'], p['requested_spawn_fingerprint']))
    forced_train = set(); calibration_overlap = []
    for group, indices in groups.items():
        model = policy['metric_groups'].get(group)
        prototype_x = None
        if model and model['mode'] != 'exact_only' and model['calibration_prototypes']:
            prototype_x = (np.asarray([p['values'] for p in model['calibration_prototypes']])
                           - np.asarray(model['center'])) / np.asarray(model['scale'])
        for i in indices:
            features = records[i]['features']; overlap = any(features[key] in calibration_fingerprints
                for key in ('measured_t0_fingerprint', 'requested_spawn_fingerprint'))
            if not overlap and prototype_x is not None:
                x = normalized_features([records[i]], model)
                candidates = np.flatnonzero(distances(x, prototype_x)[0] <= model['distance_threshold'])
                overlap = any(np.max(np.abs(x[0] - prototype_x[j])) <= 2.*model['distance_threshold'] for j in candidates)
            if overlap:
                forced_train.add(union.find(i)); calibration_overlap.append(records[i]['scene_id'])
    components = defaultdict(list)
    for i in range(size):
        components[union.find(i)].append(i)
    return {'records': records, 'representatives': sorted(representatives), 'assignment': assignment,
            'components': dict(components), 'forced_train': forced_train,
            'calibration_overlap': calibration_overlap, 'near_pair_count': pair_count,
            'protected_pair_count': protected_count, 'protected_pairs_preview': protected,
            'protected_preview_limit': 2000, 'exact_links': exact_links}


def assign_splits(curated, seed):
    records = curated['records']; representatives = set(curated['representatives'])
    strata = lambda i: f'{records[i]["dedup_group"]}|{records[i]["effect_strength"]}'
    # Multiple weather/exposure captures count as one physical base within each
    # stratum. Different observed strength labels still remain visible.
    total = Counter(key for _, key in {(base_identity(records[i]), strata(i)) for i in representatives})
    targets = {s: {key: FRACTIONS[j]*value for key, value in total.items()} for j, s in enumerate(SPLITS)}
    counts = {s: Counter() for s in SPLITS}; assignments = {}; components = curated['components']
    component_counts = {c: Counter(key for _, key in
        {(base_identity(records[i]), strata(i)) for i in ids if i in representatives}) for c, ids in components.items()}
    for c in sorted(curated['forced_train']):
        assignments[c] = 'train'; counts['train'].update(component_counts[c])
    pending = [c for c in components if c not in assignments]
    pending.sort(key=lambda c: (-sum(component_counts[c].values()), fingerprint([seed, [records[i]['scene_id'] for i in components[c]]])))
    for c in pending:
        values = component_counts[c]
        # Minimize change in normalized squared per-stratum count error. Groups
        # are indivisible; coverage impossibility is reported, never concealed.
        def score(split):
            return sum(((counts[split][key]+value-targets[split][key])**2
                        - (counts[split][key]-targets[split][key])**2) / max(total[key], 1)
                       for key, value in values.items())
        selected = min(SPLITS, key=lambda s: (score(s), SPLITS.index(s)))
        assignments[c] = selected; counts[selected].update(values)
    # Reserve at least one independent representative of each observed strength
    # in each split when movable components permit it. A component never splits,
    # and training calibration components can never be moved to evaluation.
    strengths = ('weak', 'medium', 'strong')
    component_strengths = {c: Counter(strength for _, strength in
        {(base_identity(records[i]), records[i]['effect_strength']) for i in ids if i in representatives})
                           for c, ids in components.items()}
    def strength_counts():
        result = {s: Counter() for s in SPLITS}
        for group, split in assignments.items():
            result[split].update(component_strengths[group])
        return result
    for _ in range(9):
        current = strength_counts(); moves = []
        for strength in strengths:
            for target in SPLITS:
                if current[target][strength]:
                    continue
                for group, old in assignments.items():
                    contribution = component_strengths[group]
                    if (old == target or group in curated['forced_train'] or not contribution[strength]
                        or any(current[old][label] - amount <= 0 for label, amount in contribution.items())):
                        continue
                    penalty = sum(((counts[target][key]+amount-targets[target][key])**2
                                  - (counts[target][key]-targets[target][key])**2
                                  + (counts[old][key]-amount-targets[old][key])**2
                                  - (counts[old][key]-targets[old][key])**2) / max(total[key],1)
                                  for key,amount in component_counts[group].items())
                    moves.append((penalty, group, old, target))
        if not moves:
            break
        _, group, old, target = min(moves)
        counts[old].subtract(component_counts[group]); counts[target].update(component_counts[group])
        assignments[group] = target
    splits = {s: [] for s in SPLITS}; membership = {}
    for c, indices in components.items():
        split = assignments[c]
        for i in indices:
            membership[records[i]['scene_id']] = split
            if i in representatives:
                splits[split].append(records[i]['scene_id'])
    for value in splits.values():
        value.sort()
    coverage = {s: dict(Counter(strength for _, strength in
        {(base_identity(records[i]), records[i]['effect_strength']) for i in representatives
         if membership[records[i]['scene_id']] == s})) for s in SPLITS}
    issues = []
    for strength in ('weak', 'medium', 'strong'):
        available = sum(coverage[s].get(strength, 0) for s in SPLITS)
        independent = sum(any(i in representatives and records[i]['effect_strength'] == strength for i in ids)
                          for ids in components.values())
        missing = [s for s in SPLITS if coverage[s].get(strength, 0) == 0]
        if missing:
            issues.append({'effect_strength': strength, 'available_scenes': available,
                           'independent_components': independent, 'missing_splits': missing,
                           'reason': 'insufficient independent groups, forced calibration training, or indivisible stratification; collect more varied scenes; do not move duplicates across splits'})
    return splits, membership, {'target_fractions': dict(zip(SPLITS, FRACTIONS)),
        'balance_unit': 'unique physical base_scene_id per stratum; weather captures do not increase independent coverage',
        'base_scene_counts': {s: len({base_identity(records[i]) for i in representatives
            if membership[records[i]['scene_id']] == s}) for s in SPLITS},
        'capture_scene_counts': {s: len(splits[s]) for s in SPLITS},
        'effect_strength_counts': coverage, 'stratum_counts': {s: dict(counts[s]) for s in SPLITS},
        'coverage_shortfalls': issues, 'coverage_guaranteed': not issues}


def evaluation_policy(seed):
    """Exact identities only: evaluation data never fit a distance metric."""
    return {'schema': SCHEMA, 'feature_schema': FEATURE_SCHEMA,
        'parameters': {'seed': seed}, 'metric_groups': {},
        'fit_source': {'kind': 'evaluation_exact_identities_only', 'fitted_scene_ids': []},
        'threshold_status': 'no distribution fitting; exact identities only',
        'scope': 'within this evaluation batch; no external training overlap audit',
        'time_censor_s': TIME_CAP}


def assign_evaluation(curated):
    records = curated['records']
    retained = [records[i] for i in curated['representatives']]
    if curated['forced_train'] or curated['calibration_overlap']:
        raise ValueError('Evaluation-only curation cannot include training calibration')
    membership = {r['scene_id']: 'evaluation' for r in records}
    splits = {'evaluation': sorted(r['scene_id'] for r in retained)}
    balance = {'target_fractions': {'evaluation': 1.},
        'capture_scene_counts': {'evaluation': len(retained)},
        'base_scene_counts': {'evaluation': len({base_identity(r) for r in retained})},
        'template_counts': dict(sorted(Counter(r['template'] for r in retained).items())),
        'npc_style_counts': dict(sorted(Counter(r['npc_style'] for r in retained).items())),
        'effect_strength_counts': dict(sorted(Counter(r['effect_strength'] for r in retained).items())),
        'coverage_shortfalls': [], 'coverage_guaranteed': False,
        'coverage_note': 'Observed counts only; no train/test stratification or coverage guarantee.'}
    return splits, membership, balance


def curate_dataset(dataset_manifest, out, calibration_manifest=None, policy_path=None, seed=0,
                   dataset_purpose='train_eval'):
    dataset_manifest, out = Path(dataset_manifest).resolve(), Path(out).resolve()
    if dataset_purpose not in ('train_eval', 'evaluation'):
        raise ValueError('dataset_purpose must be train_eval or evaluation')
    if dataset_purpose == 'evaluation' and (policy_path or calibration_manifest):
        raise ValueError('Evaluation-only curation uses exact identities and accepts no fitted policy or calibration manifest')
    if policy_path and calibration_manifest:
        raise ValueError('Use a frozen policy OR a calibration manifest, not both')
    if out == dataset_manifest.parent or dataset_manifest.parent.is_relative_to(out):
        raise ValueError('Curation output must be a separate child/sibling, never overwrite the dataset root')
    existing = out / 'curation_manifest.json'
    if existing.is_file() and read(existing).get('dataset_purpose', 'train_eval') != dataset_purpose:
        raise ValueError('Curation purpose differs; use a new output directory')
    records, quality_excluded = load_scenes(dataset_manifest)
    for record in records:
        raw = Path(record['scene_directory'])
        if out == raw or out.is_relative_to(raw) or raw.is_relative_to(out):
            raise ValueError('Curation output must not overlap any raw scene')
    calibration_excluded = []
    if dataset_purpose == 'evaluation':
        policy = evaluation_policy(seed)
    elif policy_path:
        policy = read(policy_path)
        if policy.get('schema') != SCHEMA or policy.get('feature_schema') != FEATURE_SCHEMA:
            raise ValueError('Frozen curation policy has incompatible schema')
    else:
        if calibration_manifest:
            calibration_path = Path(calibration_manifest).resolve()
            calibration, calibration_excluded = load_scenes(calibration_path)
            source = {'kind': 'separate_training_calibration_manifest', 'path': str(calibration_path),
                      'sha256': file_sha(calibration_path), 'excluded': calibration_excluded}
        else:
            calibration = [r for r in records if train_anchor(r, seed, DEFAULTS['calibration_train_fraction'])]
            source = {'kind': 'deterministically_reserved_training_anchors_before_fit',
                      'manifest_sha256': file_sha(dataset_manifest),
                      'selection': 'hash(seed, requested_spawn_fingerprint) < 0.35; independent of future response and split labels',
                      'anchor_scene_ids': [r['scene_id'] for r in calibration]}
        policy = fit_policy(calibration, seed, source)
    curated = curate_records(records, policy)
    splits, membership, balance = (assign_evaluation(curated) if dataset_purpose == 'evaluation'
                                   else assign_splits(curated, seed))
    records = curated['records']; retained = []; excluded = list(quality_excluded)
    for i, record in enumerate(records):
        if curated['assignment'][i] != i:
            excluded.append({'scene_id': record['scene_id'], 'reason': 'similar_whole_scene_same_plan_and_outcome',
                'representative_scene_id': records[curated['assignment'][i]]['scene_id'],
                'split': membership[record['scene_id']], 'model_branches': record['model_branches'],
                'raw_data_modified': False})
        else:
            retained.append({key: value for key, value in record.items() if key != 'features'}
                            | {'split': membership[record['scene_id']]})
    groups = [{'group_id': f'group_{number:05d}', 'scene_ids': [records[i]['scene_id'] for i in indices],
               'split': membership[records[indices[0]]['scene_id']], 'forced_training_calibration': c in curated['forced_train']}
              for number, (c, indices) in enumerate(sorted(curated['components'].items()))]
    leakage = {'status': 'passed', 'checked_policy': fingerprint(policy),
        'definition': 'No physical base_scene_id, exact-state, exact-requested-spawn or frozen-feature-neighbor connected component crosses train/validation/test, including weather/exposure variants and different outcomes/styles',
        'cross_split_near_duplicate_count': 0, 'near_pair_count': curated['near_pair_count'],
        'exact_identity_groups': curated['exact_links'], 'components': groups,
        'calibration_overlapping_scene_ids_forced_train': curated['calibration_overlap'],
        'limits': 'Conservative measured-feature comparison; no claim of complete semantic or visual scene equivalence. Small calibration cells use exact identities only.'}
    if dataset_purpose == 'evaluation':
        leakage.update(definition='Exact base/initial-state groups stay together within the evaluation-only batch.',
            external_training_overlap_checked=False,
            limits='Only exact identities within this batch were checked. No similarity threshold was fitted; near duplicates and overlap with external training data remain unchecked.')
    by_name = {r['scene_id']: r for r in records}
    leakage['source_split_violations'] = [{'group_id': group['group_id'],
        'source_splits': {scene: by_name[scene].get('source_split') for scene in group['scene_ids']
                          if by_name[scene].get('source_split')},
        'replacement_split': group['split']} for group in groups
        if len({by_name[scene].get('source_split') for scene in group['scene_ids']
                if by_name[scene].get('source_split')}) > 1]
    for group in groups:
        if len({membership[scene] for scene in group['scene_ids']}) != 1:
            raise AssertionError('Near-duplicate scene group leaked across splits')
    report = {'schema': SCHEMA, 'dataset_purpose': dataset_purpose,
        'selection_unit': 'whole paired scene with all five non-QA branches',
        'raw_data_modified': False, 'input_manifest': str(dataset_manifest), 'input_manifest_sha256': file_sha(dataset_manifest),
        'policy_sha256': fingerprint(policy), 'input_accepted_scene_count': len(records)+len(quality_excluded),
        'quality_eligible_scene_count': len(records), 'retained_scene_count': len(retained),
        'quality_eligible_base_scene_count': len({base_identity(r) for r in records}),
        'retained_base_scene_count': len({r['base_scene_id'] for r in retained}),
        'retained_scenes': retained, 'excluded_scenes': excluded, 'splits': splits,
        'protected_contrast_count': curated['protected_pair_count'],
        'protected_contrasts_preview': curated['protected_pairs_preview'], 'protected_preview_limit': 2000,
        'balance': balance, 'missing_calibration_metric_groups': sorted({r['metric_group'] for r in records} - set(policy['metric_groups'])),
        'notes': ['No individual maintain/brake/accelerate branch is removed by curation.',
                  'Collisions and weak responses remain eligible when execution/paired quality checks pass.',
                  'Different observed outcome or action plan protects a nearby scene from duplicate deletion.',
                  'Different weather, camera exposure or vehicle lighting conditions are retained; the same physical base_scene_id always shares one split.',
                  'Weather captures do not increase independent base-scene coverage or training calibration weight.',
                  'Protected contrasts share a split, even if this prevents desired split ratios.',
                  'quality_score is an optional manifest tiebreaker; no invented ranking when absent.',
                  'Features and thresholds use t0 and training-only calibration; observed outcomes only protect contrasts and stratify coverage.']}
    if dataset_purpose == 'evaluation':
        report['missing_calibration_metric_groups'] = []
        report['notes'] = report['notes'][:4] + [
            'All retained paired scenes belong to evaluation; no training or validation split is produced.',
            'No distance scale, similarity threshold, model, or readout is fitted on this batch.',
            'Exact duplicates are removed only when cell, action plan, outcome and capture condition agree.',
            'Independent base-scene and capture-version counts remain separate; external training overlap is not audited.']
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'curation_policy.json', policy)
    write(out / 'feature_table.json', {'schema': FEATURE_SCHEMA, 'scenes': {r['scene_id']: r['features'] for r in records}})
    write(out / 'leakage_audit.json', leakage)
    for name, scene_ids in splits.items():
        write(out / 'splits' / f'{name}.json', {'split': name, 'scene_ids': scene_ids,
            'scenes': [r for r in retained if r['split'] == name], 'policy_sha256': report['policy_sha256']})
    write(out / 'curation_manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--calibration-manifest', type=Path)
    parser.add_argument('--policy', type=Path, help='Reuse frozen training-calibrated policy')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset-purpose', choices=['train_eval', 'evaluation'], default='train_eval')
    args = parser.parse_args()
    result = curate_dataset(args.dataset_manifest, args.out, args.calibration_manifest, args.policy,
                            args.seed, args.dataset_purpose)
    print(json.dumps({'retained': result['retained_scene_count'], 'excluded': len(result['excluded_scenes']),
                      'splits': {k: len(v) for k, v in result['splits'].items()},
                      'coverage_shortfalls': result['balance']['coverage_shortfalls']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
